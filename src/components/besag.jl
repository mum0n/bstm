
"""
    Besag <: ComponentModel

v1.9.1 (2026-08-07) - The Besag model, a standard Intrinsic Conditional Autoregressive (ICAR) model.
It assumes that the value at a given location is conditionally dependent on its neighbors.
This model is intrinsic, meaning its precision matrix is singular and requires a sum-to-zero
constraint for identifiability.

# Fields
- `sigma::UnivariateDistribution`: The prior for the standard deviation of the spatial effect.
- `method::Symbol`: The computational method for sampling the latent field. Can be `:spectral`
  (default, efficient, and AD-safe) or `:cholesky` (dense factorization, AD-safe but less efficient).
"""
struct Besag <: ComponentModel
    sigma::UnivariateDistribution
    method::Symbol
end
 

COMPONENT_CONSTRUCTORS[:besag] = (p, params) -> Besag(p.sigma, get(params, :method, :spectral))


# helper to map to classes of methods (data structures), :any mean it can be used in many approaches
MODEL_TO_STRUCTURE_MAP[:besag] = :spatial




"""
    get_datastructures!(m_type::Type{Besag}, M::Dict, mod_data::Dict)::Bool

v1.9.1 (2026-08-07) - Data-dependent setup for the Besag component.
This method establishes the spatial context for the model. It ensures that a spatial
index variable is provided and that a valid adjacency matrix `W` is available, either
from the module's parameters or the global model configuration. It then sets the
spatial index (`s_idx`) and number of spatial units (`s_N`) in the main configuration `M`.
"""
function get_datastructures!(m_type::Type{Besag}, M::Dict, mod_data::Dict)::Bool
    data = M[:data]
    params = mod_data[:params]
    variables = mod_data[:variables]

    if isempty(variables)
        error("The Besag model requires a spatial index variable, e.g., `random(region, model=:besag)`.")
    end

    # Handle Adjacency Matrix W
    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr || w_val isa Symbol
            calling_mod = get(M, :calling_module, Main)
            try
                M[:W] = Core.eval(calling_mod, w_val)
            catch e
                error("Could not evaluate `W` argument `$(w_val)` in Besag module. Error: $e")
            end
        else
            M[:W] = w_val
        end
    end

    if !haskey(M, :W) || isnothing(M[:W])
        error("The Besag model requires an adjacency matrix `W`, but it was not provided.")
    end

    # Handle Spatial Index
    s_var_sym = Symbol(variables[1])
    if !hasproperty(data, s_var_sym)
        error("Spatial index variable ':$s_var_sym' for Besag model not found in data.")
    end
    
    M[:s_idx] = data[!, s_var_sym]
    M[:s_N] = size(M[:W], 1)

    if M[:s_N] != length(unique(M[:s_idx]))
        @warn "The number of spatial units implied by `W` ($(M[:s_N])) does not match the number of unique levels in the index variable `$(s_var_sym)` ($(length(unique(M[:s_idx])))). Ensure this is intentional."
    end

    return true
end

"""
    get_precomputes(m::Besag, M::NamedTuple, mod_data::Dict)::NamedTuple

v1.9.1 (2026-08-07) - Pre-computes the structure matrix for the Besag component.
For a Besag model, the precision matrix template is based on the ICAR structure.
This function calls the central `build_structure_template` utility to generate this
matrix and its spectral decomposition (`U`, `L`), which are essential for the
AD-safe spectral sampling method.
"""
function get_precomputes(m::Besag, M::NamedTuple, mod_data::Dict)::NamedTuple
    s_N = get(M, :s_N, 0)
    W = get(M, :W, nothing)

    if s_N == 0 || isnothing(W)
        error("Could not perform pre-computation for Besag component '$(mod_data[:key])' because spatial context (s_N and W) is missing.")
    end

    # The Besag model is an ICAR model.
    template = build_structure_template(:besag, s_N; W=W)
    return (Q_template=template.matrix, U=template.U, L=template.L, scaling_factor=template.scaling_factor)
end

"""
    get_priors(m::Besag, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

v1.9.1 (2026-08-07) - Generates the Turing code for the Besag component's priors.
This function creates the code strings for sampling the `sigma` hyperparameter and the
standard normal innovations (`struct`) for the latent field.
"""
function get_priors(m::Besag, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = size(spec.Q_template, 1)
    is_multivariate = (arch == "multivariate")
    is_shared = get(spec.params, :shared, false)
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ $(_distribution_to_string(m.sigma))")
    end
    push!(priors_acc, "$(v.struct) ~ MvNormal(zeros(T, $(n_latent)), I)")
    return join(priors_acc, "\n    ")
end

"""
    get_updates(m::Besag, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

v1.9.1 (2026-08-07) - Generates the Turing code for the Besag component's update logic.
This function implements a conditional logic based on the `method` field of the `Besag` struct.
- `:spectral` (default): Uses the efficient and AD-safe spectral sampling method.
- `:cholesky`: Uses a dense Cholesky decomposition of the static ICAR precision matrix.
  This provides a computational alternative while remaining AD-safe.
Both methods correctly implement the ICAR model with a sum-to-zero constraint.
"""
function get_updates(m::Besag, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "s_idx"
    n_latent = size(spec.Q_template, 1)

    if m.method == :spectral
        return """
        # --- Besag Spectral Assembly: $(spec.key) ---
        # This block constructs the Besag (ICAR) spatial effect using a non-centered parameterization
        # based on the spectral decomposition of the ICAR precision matrix. This method is
        # the default as it is efficient and safe for automatic differentiation (AD).

        # 1. Construct the diagonal of the spectral transformation matrix D.
        #    The standard deviation is sigma / sqrt(L_j).
        local diag_D = $(v.sigma) ./ sqrt.(spec.L .+ T(noise))

        # 2. Enforce sum-to-zero constraint by zeroing out the component corresponding to the null space.
        #    ICAR has a rank deficiency of 1, so the first eigenvalue is zero.
        diag_D[1] = zero(T)

        # 3. Apply the spectral transformation: latent = U * D * z
        $(v.latent) = spec.U * (diag_D .* $(v.struct))

        # 4. Add the final effect to the linear predictor.
        $(eta_target) .+= view($(v.latent), M.$(index_var))
        """
    elseif m.method == :cholesky
        return """
        # --- Besag Cholesky Assembly: $(spec.key) ---
        # This block constructs the Besag (ICAR) spatial effect using a dense Cholesky decomposition
        # of the static ICAR precision matrix. This method is AD-safe but can be less
        # efficient than the spectral method for large N.

        # 1. Get the ICAR precision matrix template.
        local Q_template = spec_registry["$(spec.key)"].Q_template
        
        # 2. Perform a dense Cholesky decomposition. This is AD-safe.
        local F_struct = cholesky(Symmetric(Matrix(Q_template) + noise * I))
        
        # 3. Sample the latent field using the Cholesky factor (non-centered).
        $(v.latent) = F_struct.L' \\ $(v.struct)
        
        # 4. Apply a soft sum-to-zero constraint for identifiability.
        Turing.@addlogprob! logpdf(Normal(zero(T), T(0.001) * $(n_latent)), sum($(v.latent)))
        
        # 5. Scale by sigma and add to the linear predictor.
        $(v.latent) .*= $(v.sigma)
        $(eta_target) .+= view($(v.latent), M.$(index_var))
        """
    else
        error("Unsupported method '$(m.method)' for Besag component. Supported methods are :spectral and :cholesky.")
    end
end

"""
    get_effects(m::Besag, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

v1.9.1 (2026-08-07) - Reconstructs the Besag component's effect from posterior samples.
This function extracts the posterior samples for `sigma` and `struct` from the MCMC
chain. It then reconstructs the full posterior distribution of the latent Besag field by
re-running the spectral transformation for each posterior sample, enforcing the sum-to-zero
constraint.
"""
function get_effects(m::Besag, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    structured_effects = []
    is_multivariate = outcomes_N > 1
    is_shared = get(spec.params, :shared, false)
    n_latent = size(spec.Q_template, 1)

    U = spec.hyper.U
    L_eig = spec.hyper.L

    for k in 1:outcomes_N
        outcome_idx = is_multivariate ? k : nothing
        v = generate_full_variable_names(spec, M.model_arch, outcome_idx)
        
        sigma_var_name = (is_multivariate && is_shared) ? string(generate_full_variable_names(spec, M.model_arch, nothing).sigma) : string(v.sigma)
        struct_var_name = (is_multivariate && is_shared) ? string(generate_full_variable_names(spec, M.model_arch, nothing).struct) : string(v.struct)

        sigma_samples = get_params_vector(chain, sigma_var_name, 1)
        struct_samples = get_params_vector(chain, struct_var_name, n_latent)
        
        T = eltype(chain.value)
        effect_k = Matrix{T}(undef, n_latent, n_samples)

        diag_D = one(T) ./ sqrt.(L_eig .+ T(1e-9))
        diag_D[1] = zero(T) # Enforce sum-to-zero

        for s in 1:n_samples
            sigma_s = sigma_samples[s, 1]
            struct_s = struct_samples[s, :]
            effect_k[:, s] = U * (diag_D .* struct_s) .* sigma_s
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end

