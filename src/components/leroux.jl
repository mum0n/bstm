

"""
    Leroux <: ComponentModel

v1.9.1 (2026-08-07) - The Leroux model, a proper CAR model that defines spatial correlation
as a convex combination of a spatially structured (ICAR) component and an unstructured (IID)
component. Its precision matrix is `Q = (1-ρ)I + ρQ*`, where `Q*` is the scaled ICAR precision matrix.

# Fields
- `rho::UnivariateDistribution`: The prior for the spatial correlation parameter, bounded on `[0, 1]`.
- `sigma::UnivariateDistribution`: The prior for the overall marginal standard deviation.
- `method::Symbol`: The computational method for sampling the latent field. Can be `:spectral`
  (default, efficient, and AD-safe) or `:cholesky` (dense factorization, AD-safe but less efficient).
"""
struct Leroux <: ComponentModel
    rho::UnivariateDistribution
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_CONSTRUCTORS[:leroux] = (p, params) -> Leroux(p.rho, p.sigma, get(params, :method, :spectral))


# helper to map to classes of methods (data structures), :any mean it can be used in many approaches
MODEL_TO_STRUCTURE_MAP[:leroux] = :spatial


"""
    get_datastructures!(m_type::Type{Leroux}, M::Dict, mod_data::Dict)::Bool

v1.9.1 (2026-08-07) - Data-dependent setup for the Leroux component.
This method establishes the spatial context for the model. It ensures that a spatial
index variable is provided and that a valid adjacency matrix `W` is available. It then
sets the spatial index (`s_idx`) and number of spatial units (`s_N`) in the main
configuration `M`.
"""
function get_datastructures!(m_type::Type{Leroux}, M::Dict, mod_data::Dict)::Bool
    data = M[:data]
    params = mod_data[:params]
    variables = mod_data[:variables]

    if isempty(variables)
        error("The Leroux model requires a spatial index variable, e.g., `random(region, model=:leroux)`.")
    end

    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr || w_val isa Symbol
            calling_mod = get(M, :calling_module, Main)
            try; M[:W] = Core.eval(calling_mod, w_val); catch e; error("Could not evaluate `W` argument `$(w_val)`. Error: $e"); end
        else
            M[:W] = w_val
        end
    end

    if !haskey(M, :W) || isnothing(M[:W]); error("The Leroux model requires an adjacency matrix `W`."); end

    s_var_sym = Symbol(variables[1])
    if !hasproperty(data, s_var_sym); error("Spatial index variable ':$s_var_sym' not found in data."); end
    
    M[:s_idx] = data[!, s_var_sym]
    M[:s_N] = size(M[:W], 1)

    return true
end

"""
    get_precomputes(m::Leroux, M::NamedTuple, mod_data::Dict)::NamedTuple

v1.9.1 (2026-08-07) - Pre-computes the structure matrix for the Leroux component.
The precision matrix template for a Leroux model is based on the ICAR structure.
This function calls `build_structure_template` to generate this matrix and its
spectral decomposition (`U`, `L`), which are essential for the AD-safe spectral sampling method.
"""
function get_precomputes(m::Leroux, M::NamedTuple, mod_data::Dict)::NamedTuple
    s_N = get(M, :s_N, 0)
    W = get(M, :W, nothing)
    if s_N == 0 || isnothing(W); error("Could not perform pre-computation for Leroux component because spatial context (s_N and W) is missing."); end

    template = build_structure_template(:leroux, s_N; W=W)
    return (Q_template=template.matrix, U=template.U, L=template.L, scaling_factor=template.scaling_factor)
end

"""
    get_priors(m::Leroux, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

v1.9.1 (2026-08-07) - Generates the Turing code for the Leroux component's priors.
This function creates the code strings for sampling the hyperparameters `sigma` and `rho`,
as well as the standard normal innovations (`raw`) for the latent field.
"""
function get_priors(m::Leroux, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
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
    push!(priors_acc, "$(v.raw) ~ MvNormal(zeros(T, $(n_latent)), I)")
    return join(priors_acc, "\n    ")
end

"""
    get_updates(m::Leroux, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

v1.9.1 (2026-08-07) - Generates the Turing code for the Leroux component's update logic.
This function implements a conditional logic based on the `method` field of the `Leroux` struct.
- `:spectral` (default): Uses the efficient and AD-safe spectral sampling method.
- `:cholesky`: Uses a dense Cholesky decomposition of the full Leroux precision matrix.
"""
function get_updates(m::Leroux, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "s_idx"

    if m.method == :spectral
        return """
        # --- Leroux Spectral Assembly: $(spec.key) ---
        # This block constructs the Leroux spatial effect using a non-centered parameterization
        # based on the spectral decomposition of the ICAR precision matrix. This is the default method.
        
        # 1. Construct the diagonal of the spectral transformation matrix D.
        #    For Leroux, Q = (1-ρ)I + ρQ*. The eigenvalues are (1-ρ) + ρλ_j.
        #    The standard deviation of the transformed variable is σ / sqrt(eigenvalue).
        local diag_D = $(v.sigma) ./ sqrt.((one(T) .- $(v.rho)) .+ $(v.rho) .* spec.L .+ T(noise))
        
        # 2. Apply the spectral transformation: latent = U * D * z
        $(v.latent) = spec.U * (diag_D .* $(v.raw))
        
        # 3. Add the final effect to the linear predictor.
        $(eta_target) .+= view($(v.latent), M.$(index_var))
        """
    elseif m.method == :cholesky
        return """
        # --- Leroux Cholesky Assembly: $(spec.key) ---
        # This block constructs the Leroux spatial effect using a dense Cholesky decomposition.
        
        # 1. Recompose the full Leroux precision matrix.
        local Q_template = spec_registry["$(spec.key)"].Q_template
        local Q_final = (one(T) - $(v.rho)) .* I(size(Q_template, 1)) .+ $(v.rho) .* Q_template
        
        # 2. Perform a dense Cholesky decomposition.
        local F = cholesky(Symmetric(Matrix(Q_final) + noise * I))
        
        # 3. Sample the latent field using the Cholesky factor (non-centered).
        local unscaled_latent = F.L' \\ $(v.raw)
        $(v.latent) = $(v.sigma) .* unscaled_latent
        
        # 4. Add the final effect to the linear predictor.
        $(eta_target) .+= view($(v.latent), M.$(index_var))
        """
    else
        error("Unsupported method '$(m.method)' for Leroux component. Supported methods are :spectral and :cholesky.")
    end
end

"""
    get_effects(m::Leroux, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

v1.9.1 (2026-08-07) - Reconstructs the Leroux component's effect from posterior samples.
This function extracts the posterior samples for `sigma`, `rho`, and `raw` from the MCMC
chain. It then reconstructs the full posterior distribution of the latent Leroux field by
re-running the spectral transformation for each posterior sample.
"""
function get_effects(m::Leroux, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
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
        rho_var_name = (is_multivariate && is_shared) ? string(generate_full_variable_names(spec, M.model_arch, nothing).rho) : string(v.rho)

        sigma_samples = get_params_vector(chain, sigma_var_name, 1)
        rho_samples = get_params_vector(chain, rho_var_name, 1)
        raw_samples = get_params_vector(chain, string(v.raw), n_latent)
        
        T = eltype(chain.value)
        effect_k = Matrix{T}(undef, n_latent, n_samples)

        for s in 1:n_samples
            sigma_s = sigma_samples[s, 1]
            rho_s = rho_samples[s, 1]
            raw_s = raw_samples[s, :]

            # Re-run the spectral transformation for this posterior sample
            diag_D_s = sigma_s ./ sqrt.((one(T) - rho_s) .+ rho_s .* L_eig .+ T(1e-9))
            effect_k[:, s] = U * (diag_D_s .* raw_s)
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end