
"""
    BYM2 <: ComponentModel

v1.9.1 (2026-08-07) - The Besag-York-Mollié 2 (BYM2) model, which provides an intuitive and
well-identified parameterization for spatial effects by separating them into a structured (ICAR)
and an unstructured (IID) component.

# Fields
- `rho::UnivariateDistribution`: The prior for the mixing parameter, controlling the proportion of
  variance attributed to the structured spatial effect.
- `sigma::UnivariateDistribution`: The prior for the overall marginal standard deviation of the
  total spatial effect.
- `method::Symbol`: The computational method for sampling the latent field. Can be `:spectral`
  (default, efficient, and AD-safe) or `:cholesky` (dense factorization, AD-safe but less efficient).
"""
struct BYM2 <: ComponentModel
    rho::UnivariateDistribution
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_CONSTRUCTORS[:bym2] = (p, params) -> BYM2(p.rho, p.sigma, p.method)


# helper to map to classes of methods (data structures), :any mean it can be used in many approaches
MODEL_TO_STRUCTURE_MAP[:bym2] = :spatial

"""
    get_datastructures!(m_type::Type{BYM2}, M::Dict, mod_data::Dict)::Bool

v1.9.1 (2026-08-07) - Data-dependent setup for the BYM2 component.
This method establishes the spatial context for the model. It ensures that a spatial
index variable is provided and that a valid adjacency matrix `W` is available, either
from the module's parameters or the global model configuration. It then sets the
spatial index (`s_idx`) and number of spatial units (`s_N`) in the main configuration `M`.
"""
function get_datastructures!(m_type::Type{BYM2}, M::Dict, mod_data::Dict)::Bool
    data = M[:data]
    params = mod_data[:params]
    variables = mod_data[:variables]

    if isempty(variables)
        error("The BYM2 model requires a spatial index variable, e.g., `random(region, model=:bym2)`.")
    end

    # Handle Adjacency Matrix W
    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr || w_val isa Symbol
            calling_mod = get(M, :calling_module, Main)
            try
                M[:W] = Core.eval(calling_mod, w_val)
            catch e
                error("Could not evaluate `W` argument `$(w_val)` in BYM2 module. Error: $e")
            end
        else
            M[:W] = w_val
        end
    end

    if !haskey(M, :W) || isnothing(M[:W])
        error("The BYM2 model requires an adjacency matrix `W`, but it was not provided.")
    end

    # Handle Spatial Index
    s_var_sym = Symbol(variables[1])
    if !hasproperty(data, s_var_sym)
        error("Spatial index variable ':$s_var_sym' for BYM2 model not found in data.")
    end
    
    M[:s_idx] = data[!, s_var_sym]
    M[:s_N] = size(M[:W], 1)

    if M[:s_N] != length(unique(M[:s_idx]))
        @warn "The number of spatial units implied by `W` ($(M[:s_N])) does not match the number of unique levels in the index variable `$(s_var_sym)` ($(length(unique(M[:s_idx])))). Ensure this is intentional."
    end

    return true
end

"""
    get_precomputes(m::BYM2, M::NamedTuple, mod_data::Dict)::NamedTuple

v1.9.1 (2026-08-07) - Pre-computes the structure matrix for the BYM2 component.
For a BYM2 model, the precision matrix template is based on the ICAR structure.
This function calls the central `build_structure_template` utility to generate this
matrix and its spectral decomposition (`U`, `L`), which are essential for the
AD-safe spectral sampling method.
"""
function get_precomputes(m::BYM2, M::NamedTuple, mod_data::Dict)::NamedTuple
    s_N = get(M, :s_N, 0)
    W = get(M, :W, nothing)

    if s_N == 0 || isnothing(W)
        error("Could not perform pre-computation for BYM2 component '$(mod_data[:key])' because spatial context (s_N and W) is missing.")
    end

    # The BYM2 model's structured component is an ICAR model.
    template = build_structure_template(:bym2, s_N; W=W)
    return (Q_template=template.matrix, U=template.U, L=template.L, scaling_factor=template.scaling_factor)
end

"""
    get_priors(m::BYM2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

v1.9.1 (2026-08-07) - Generates the Turing code for the BYM2 component's priors.
This function creates the code strings for sampling the hyperparameters `sigma` and `rho`,
as well as the standard normal innovations for the structured (`struct`) and
unstructured (`iid`) components of the BYM2 model.
"""
function get_priors(m::BYM2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = size(spec.Q_template, 1)
    is_multivariate = (arch == "multivariate")
    is_shared = get(spec.params, :shared, false)
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ $(_distribution_to_string(m.sigma))")
        push!(priors_acc, "$(v.rho) ~ $(_distribution_to_string(m.rho))")
    end
    push!(priors_acc, "$(v.struct) ~ MvNormal(zeros(T, $(n_latent)), I)")
    push!(priors_acc, "$(v.iid) ~ MvNormal(zeros(T, $(n_latent)), I)")
    return join(priors_acc, "\n    ")
end

"""
    get_updates(m::BYM2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

v1.9.2 (2026-08-07) - Generates the Turing code for the BYM2 component's update logic.
This function implements a conditional logic based on the `method` field of the `BYM2` struct.
- `:spectral` (default): Uses the efficient and AD-safe spectral sampling method.
- `:cholesky`: Uses a dense Cholesky decomposition of the static ICAR precision matrix.
  This provides a computational alternative while remaining AD-safe.
Both methods correctly implement the Riebler parameterization for the BYM2 model.
"""
function get_updates(m::BYM2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "s_idx"

    if m.method == :spectral
        return """
        # --- BYM2 Spectral Assembly: $(spec.key) ---
        # This block constructs the BYM2 spatial effect using a non-centered parameterization
        # based on the spectral decomposition of the ICAR precision matrix. This method is
        # the default as it is efficient and safe for automatic differentiation (AD).

        # 1. Construct the diagonal of the spectral transformation matrix D for the structured part.
        #    The structured part has precision Q_star, so its covariance has eigenvalues 1/L_j.
        #    The standard deviation is 1/sqrt(L_j).
        local diag_D_structured = one(T) ./ sqrt.(spec.L .+ T(noise))
        # The first eigenvalue is 0, corresponding to the null space (intercept).
        # Set its contribution to 0 to enforce the sum-to-zero constraint.
        diag_D_structured[1] = zero(T)

        # 2. Apply the spectral transformation to the standard normal innovations: latent = U * D * z
        local structured_effect = spec.U * (diag_D_structured .* $(v.struct))

        # 3. Combine structured and unstructured components using the Riebler parameterization.
        $(v.latent) = $(v.sigma) .* (sqrt($(v.rho)) .* structured_effect .+ sqrt(one(T) - $(v.rho)) .* $(v.iid))

        # 4. Add the final effect to the linear predictor.
        $(eta_target) .+= view($(v.latent), M.$(index_var))
        """
    elseif m.method == :cholesky
        return """
        # --- BYM2 Cholesky Assembly: $(spec.key) ---
        # This block constructs the BYM2 spatial effect using a dense Cholesky decomposition
        # of the static ICAR precision matrix. This method is AD-safe but can be less
        # efficient than the spectral method for large N.

        # 1. Get the ICAR precision matrix template.
        local Q_template = spec_registry["$(spec.key)"].Q_template
        
        # 2. Perform a dense Cholesky decomposition. This is AD-safe.
        local F_struct = cholesky(Symmetric(Matrix(Q_template) + noise * I))
        
        # 3. Sample the structured component using the Cholesky factor (non-centered).
        local structured_effect = F_struct.L' \\ $(v.struct)
        
        # 4. Apply a soft sum-to-zero constraint for identifiability.
        Turing.@addlogprob! logpdf(Normal(zero(T), T(0.001) * $(spec.hyper.n_latent)), sum(structured_effect))
        
        # 5. Combine structured and unstructured components using the Riebler parameterization.
        $(v.latent) = $(v.sigma) .* (sqrt($(v.rho)) .* structured_effect .+ sqrt(one(T) - $(v.rho)) .* $(v.iid))
        
        # 6. Add the final effect to the linear predictor.
        $(eta_target) .+= view($(v.latent), M.$(index_var))
        """
    else
        error("Unsupported method '$(m.method)' for BYM2 component. Supported methods are :spectral and :cholesky.")
    end
end

"""
    get_effects(m::BYM2, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

v1.9.2 (2026-08-07) - Reconstructs the BYM2 component's effect from posterior samples.
This function is updated to correctly reconstruct the posterior effect regardless of the
`method` used during sampling. It extracts the posterior samples for `sigma`, `rho`,
`struct`, and `iid` and re-runs the Riebler parameterization logic for each sample.
"""
function get_effects(m::BYM2, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    structured_effects = []
    is_multivariate = outcomes_N > 1
    is_shared = get(spec.params, :shared, false)
    n_latent = size(spec.Q_template, 1)

    # Pre-fetch spectral decomposition matrices
    U = spec.hyper.U
    L_eig = spec.hyper.L

    for k in 1:outcomes_N
        outcome_idx = is_multivariate ? k : nothing
        v = generate_full_variable_names(spec, M.model_arch, outcome_idx)
        
        # Handle shared vs. outcome-specific hyperparameters
        sigma_var_name = (is_multivariate && is_shared) ? string(generate_full_variable_names(spec, M.model_arch, nothing).sigma) : string(v.sigma)
        rho_var_name = (is_multivariate && is_shared) ? string(generate_full_variable_names(spec, M.model_arch, nothing).rho) : string(v.rho)

        sigma_samples = get_params_vector(chain, sigma_var_name, 1)
        rho_samples = get_params_vector(chain, rho_var_name, 1)
        struct_samples = get_params_vector(chain, string(v.struct), n_latent)
        iid_samples = get_params_vector(chain, string(v.iid), n_latent)
        
        T = eltype(chain.value)
        effect_k = Matrix{T}(undef, n_latent, n_samples)

        # The reconstruction logic is the same for both methods, as they both
        # sample the same `struct` and `iid` innovations. We just need to reconstruct
        # the structured part from its innovations.
        diag_D_structured = one(T) ./ sqrt.(L_eig .+ T(1e-9))
        diag_D_structured[1] = zero(T)

        for s in 1:n_samples
            sigma_s = sigma_samples[s, 1]
            rho_s = rho_samples[s, 1]
            struct_innov_s = struct_samples[s, :]
            iid_s = iid_samples[s, :]

            # Reconstruct the structured effect from its standard normal innovations
            structured_effect_s = U * (diag_D_structured .* struct_innov_s)
            
            # Combine components using the Riebler parameterization
            effect_k[:, s] = sigma_s .* (sqrt(rho_s) .* structured_effect_s .+ sqrt(one(T) - rho_s) .* iid_s)
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end