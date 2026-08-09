
"""
    NonStationaryVariance <: ComponentModel

A component model for non-stationary variance, where the standard deviation of a base
spatial effect varies across space according to a modifier spatial effect.
It acts as an orchestrator, combining a `base_model` (typically a spatial GMRF)
with a `modifier_model` (typically a spatial smoother) to create a spatially
varying standard deviation.

# Version
v1.0.0 (2026-08-08)

# Mathematical Summary
The component models a non-stationary spatial field \$\\phi(s)\$ where the local
standard deviation is a function of some covariates. The effect is a product of a
stationary base process \$\\phi_{base}\$ and a spatially varying scale \$\\sigma(s)\$:
\$\\phi(s) = \\phi_{base}(s) \\cdot \\sigma(s)\$

1.  **Base Process (\$\\phi_{base}\$)**: This is a standard, zero-mean Gaussian Markov
    Random Field (GMRF) with unit variance, such as an `ICAR` or `Besag` model.
    \$\\phi_{base} \\sim \\mathcal{N}(0, Q_{base}^{-1})\$

2.  **Scale Process (\$\\sigma(s)\$)**: The logarithm of the scale is modeled as a
    smooth function of one or more covariates \$x\$, defined by the `modifier_model`:
    \$\\log(\\sigma(s)) = f(x(s))\$
    where \$f(x)\$ is typically a P-spline or Gaussian Process smoother. Exponentiating
    ensures the standard deviation is always positive.

This structure allows the model to capture heteroscedasticity in the spatial process,
where some regions are inherently more variable than others.

# Assumptions
- The base model is a GMRF with a valid precision structure.
- The modifier model is a smoother that can be evaluated over the spatial domain.

# Best Use Case
Modeling spatial data where the variance is expected to change as a function of
other spatial covariates. For example, modeling species abundance where the
variability is higher in areas with more complex habitats, or modeling air pollution
where variance is higher near industrial centers.

# Key References
- Gelfand, A. E., Schmidt, A. M., Banerjee, S., & Sirmans, C. F. (2005).
  *Nonstationary multivariate process modeling through spatially varying
  coregionalization*. Test, 14(2), 263-312. (For concepts on non-stationary
  spatial modeling).

# Fields
- `base_model::ComponentModel`: The underlying spatial component model (e.g., `ICAR`)
  whose variance is being modified.
- `modifier_model::ComponentModel`: The smoother component model (e.g., `PSpline`)
  that defines the pattern of the log-standard deviation.
"""
struct NonStationaryVariance <: ComponentModel
    base_model::ComponentModel
    modifier_model::ComponentModel
end

# Add to the central component constructor registry.
COMPONENT_TYPE_REGISTRY[:nonstationaryvariance] = NonStationaryVariance
COMPONENT_CONSTRUCTORS[:nonstationaryvariance] = (p, params) -> begin
    base_model_obj = get(
        params, :base_model_obj,
        error("NonStationaryVariance requires a `base_model_obj`.")
    )
    modifier_model_obj = get(
        params, :modifier_model_obj,
        error("NonStationaryVariance requires a `modifier_model_obj`.")
    )
    NonStationaryVariance(base_model_obj, modifier_model_obj)
end

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[:nonstationaryvariance] = :spatial

"""
    get_datastructures!(m_type::Type{<:NonStationaryVariance}, M::Dict, mod_data::Dict)

Delegates data structure setup to both the `base_model` and `modifier_model`.
"""
function get_datastructures!(
    m_type::Type{<:NonStationaryVariance}, M::Dict, mod_data::Dict
)::Bool
    params = mod_data[:params]

    # Delegate to the base model (e.g., ICAR)
    base_model_spec_node = get(params, :base_model_spec, nothing)
    if isnothing(base_model_spec_node)
        error("NonStationaryVariance requires a `base_model_spec` parameter.")
    end
    base_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_base"),
        :type => base_model_spec_node.module_type,
        :variables => get(base_model_spec_node.args, :positional_args, []),
        :params => base_model_spec_node.args
    )
    base_model_type = typeof(mod_data[:component_obj].base_model)
    get_datastructures!(base_model_type, M, base_mod_data)

    # Delegate to the modifier model (e.g., PSpline)
    modifier_model_spec_node = get(params, :modifier_model_spec, nothing)
    if isnothing(modifier_model_spec_node)
        error("NonStationaryVariance requires a `modifier_model_spec` parameter.")
    end
    modifier_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_modifier"),
        :type => modifier_model_spec_node.module_type,
        :variables => get(modifier_model_spec_node.args, :positional_args, []),
        :params => modifier_model_spec_node.args
    )
    modifier_model_type = typeof(mod_data[:component_obj].modifier_model)
    get_datastructures!(modifier_model_type, M, modifier_mod_data)

    return true
end

"""
    get_precomputes(m::NonStationaryVariance, M::NamedTuple, mod_data::Dict)

Delegates pre-computation to both the `base_model` and `modifier_model` and stores
their results in the `hyper` registry.
"""
function get_precomputes(
    m::NonStationaryVariance, M::NamedTuple, mod_data::Dict
)::NamedTuple
    params = mod_data[:params]

    # Precomputes for the base model
    base_model_spec_node = get(params, :base_model_spec, nothing)
    base_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_base"),
        :type => base_model_spec_node.module_type,
        :variables => get(base_model_spec_node.args, :positional_args, []),
        :params => base_model_spec_node.args
    )
    base_precomputes = get_precomputes(m.base_model, M, base_mod_data)

    # Precomputes for the modifier model
    modifier_model_spec_node = get(params, :modifier_model_spec, nothing)
    modifier_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_modifier"),
        :type => modifier_model_spec_node.module_type,
        :variables => get(modifier_model_spec_node.args, :positional_args, []),
        :params => modifier_model_spec_node.args
    )
    modifier_precomputes = get_precomputes(m.modifier_model, M, modifier_mod_data)

    modifier_basis_key = Symbol(join(modifier_mod_data[:variables], "_"))

    return (
        base_precomputes=base_precomputes,
        modifier_precomputes=modifier_precomputes,
        modifier_basis_key=modifier_basis_key
    )
end

"""
    get_priors(m::NonStationaryVariance, spec::NamedTuple, arch::String, outcome_idx, M)

Generates priors by delegating to the child models and adding a prior for the raw
innovations of the base model.
"""
function get_priors(
    m::NonStationaryVariance, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    base_spec_key = Symbol("$(spec.key)_base")
    modifier_spec_key = Symbol("$(spec.key)_modifier")

    base_spec_for_priors = (
        key = base_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.base_model)],
        var = spec.var,
        component_obj = m.base_model,
        params = spec.params,
        hyper = spec.hyper.base_precomputes
    )
    modifier_spec_for_priors = (
        key = modifier_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.modifier_model)],
        var = spec.var,
        component_obj = m.modifier_model,
        params = spec.params,
        hyper = spec.hyper.modifier_precomputes
    )

    # The base model's sigma is not used, as variance is controlled by the modifier.
    # We only need its structural priors (e.g., rho for Leroux).
    base_priors = get_priors(m.base_model, base_spec_for_priors, arch, outcome_idx, M)
    base_priors_cleaned = replace(base_priors, r".*sigma.*" => "")
    
    modifier_priors = get_priors(
        m.modifier_model, modifier_spec_for_priors, arch, outcome_idx, M
    )

    base_p_names = generate_full_variable_names(base_spec_for_priors, arch, outcome_idx)
    n_latent_base = spec.hyper.base_precomputes.n_latent
    base_raw_prior = "$(base_p_names.raw) ~ MvNormal(zeros($(n_latent_base)), I)"

    return """
        # --- Priors for NonStationaryVariance component: $(spec.key) ---
        $(base_priors_cleaned)
        $(modifier_priors)
        $(base_raw_prior)
    """
end

"""
    get_updates(m::NonStationaryVariance, spec::NamedTuple, arch::String, outcome_idx, M)

Generates the Turing code to construct the non-stationary variance effect.
"""
function get_updates(
    m::NonStationaryVariance, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    base_spec_key = Symbol("$(spec.key)_base")
    modifier_spec_key = Symbol("$(spec.key)_modifier")

    base_spec_for_updates = (
        key = base_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.base_model)],
        var = spec.var,
        component_obj = m.base_model,
        params = spec.params,
        hyper = spec.hyper.base_precomputes
    )
    modifier_spec_for_updates = (
        key = modifier_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.modifier_model)],
        var = spec.var,
        component_obj = m.modifier_model,
        params = spec.params,
        hyper = spec.hyper.modifier_precomputes
    )

    modifier_frags = get_updates(
        m.modifier_model, modifier_spec_for_updates, arch, outcome_idx, M
    )
    base_p_names = generate_full_variable_names(base_spec_for_updates, arch, outcome_idx)
    modifier_p_names = generate_full_variable_names(
        modifier_spec_for_updates, arch, outcome_idx
    )

    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    effect_app_regex = Regex("$(eta_target) \\.\\+= .*")
    modifier_update_cleaned = replace(modifier_frags, effect_app_regex => "")

    n_latent_base = spec.hyper.base_precomputes.n_latent
    modifier_basis_key = spec.hyper.modifier_basis_key
    base_model_type_sym = Symbol(lowercase(string(typeof(m.base_model))))

    base_latent_reconstruction_code = """
        local Q_base_template = spec_registry[:$(base_spec_for_updates.key)].hyper.Q_template
        local F_base = cholesky(Symmetric(Matrix(Q_base_template) + M.noise * I))
        local base_latent_raw = F_base.L' \\ $(base_p_names.raw)
        
        if $(base_model_type_sym) in [:icar, :besag]
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_latent_base)), sum(base_latent_raw))
        end
    """

    return """
        # --- NonStationaryVariance Component: $(spec.key) ---
        
        # 1. Realize the log-standard deviation field from the modifier model.
        $(modifier_update_cleaned)
        
        local log_sigma_field = M.basis_matrices[:$(modifier_basis_key)] * $(modifier_p_names.latent)
        local spatially_varying_sigma = exp.(log_sigma_field)
        
        # 2. Realize the raw latent field from the base model.
        $(base_latent_reconstruction_code)
        
        # 3. Combine raw latent field with spatially varying sigma.
        local final_effect_latent = base_latent_raw .* spatially_varying_sigma
        
        # 4. Add the final effect to the linear predictor.
        $(eta_target) .+= view(final_effect_latent, M.s_idx)
    """
end

"""
    get_effects(m::NonStationaryVariance, chain, M::NamedTuple, ...)

Reconstructs the `NonStationaryVariance` component's effect from posterior samples.
"""
function get_effects(
    m::NonStationaryVariance, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    base_spec_key = Symbol("$(spec.key)_base")
    modifier_spec_key = Symbol("$(spec.key)_modifier")

    for k in 1:outcomes_N
        outcome_idx = outcomes_N > 1 ? k : nothing
        
        # --- 1. Reconstruct Modifier Field (log-sigma) ---
        modifier_spec = (
            key = modifier_spec_key,
            structure = MODEL_TO_STRUCTURE_MAP[typeof(m.modifier_model)],
            var = spec.var,
            component_obj = m.modifier_model,
            params = spec.params,
            hyper = spec.hyper.modifier_precomputes
        )
        modifier_results = get_effects(
            m.modifier_model, chain, M, n_samples, outcomes_N, modifier_spec, PS, N_total
        )
        log_sigma_field_samples = modifier_results.structured[k]
        spatially_varying_sigma_samples = exp.(log_sigma_field_samples)

        # --- 2. Reconstruct Base Field ---
        base_spec = (
            key = base_spec_key,
            structure = MODEL_TO_STRUCTURE_MAP[typeof(m.base_model)],
            var = spec.var,
            component_obj = m.base_model,
            params = spec.params,
            hyper = spec.hyper.base_precomputes
        )
        base_p_names = generate_full_variable_names(base_spec, M.model_arch, outcome_idx)
        base_raw_samples = get_params_vector(
            chain, string(base_p_names.raw), base_spec.hyper.n_latent
        )
        
        n_latent_base = base_spec.hyper.n_latent
        Q_base_template = base_spec.hyper.Q_template
        F_base = cholesky(Symmetric(Matrix(Q_base_template) + M.noise * I))
        
        base_latent_raw_samples = zeros(n_samples, n_latent_base)
        for i in 1:n_samples
            base_latent_raw_samples[i, :] = F_base.L' \ base_raw_samples[i, :]
        end

        # --- 3. Combine and Index ---
        s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)
        
        final_effect_k = zeros(N_total, n_samples)
        for i in 1:n_samples
            base_field_at_obs = view(base_latent_raw_samples[i, :], s_idx_full)
            sigma_at_obs = spatially_varying_sigma_samples[:, i]
            final_effect_k[:, i] = base_field_at_obs .* sigma_at_obs
        end
        
        push!(structured_effects, final_effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
