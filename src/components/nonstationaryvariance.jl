"""
    NonStationaryVariance <: ComponentModel

A component model for non-stationary variance, where the standard deviation of a base
spatial effect varies across space according to a modifier spatial effect.
It acts as an orchestrator, combining a `base_model` (typically a spatial GMRF)
with a `modifier_model` (typically a spatial smoother) to create a spatially
varying standard deviation.

# Version
v1.1.1 (2026-08-14)

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

# Computational Methods (for Base Model)
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the base model's precision matrix. Recommended for NUTS.
- `:cholesky` (AD-friendly): Uses a dense Cholesky factorization of the base model's
  precision matrix.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization,
  which is not compatible with most AD backends.

# Inputs
- **Required**:
  - A composition of two `random()` modules using the `∘` operator, e.g.,
    `random(s_idx, model=icar) ∘ random(cov, model=pspline)`.
  - The base model must be a spatial GMRF (e.g., `icar`, `bym2`).
  - The modifier model must be a smoother (e.g., `pspline`, `gp`).
- **Optional**:
  - `method`: `Symbol`, computational method for the base model. Default: `:spectral`.

# Outputs (Parameter Names)
- Parameters are inherited from the child components, prefixed with the main key
  and either `_base` or `_modifier`. For example:
  - `sigma_<key>_modifier`: The standard deviation of the modifier smoother.
  - `rho_<key>_base`: The mixing parameter of the base spatial model (if applicable).
  - `innovations_<key>_modifier`: Innovations for the modifier smoother.
  - `innovations_<key>_base`: Innovations for the base spatial model.
"""
struct NonStationaryVariance <: ComponentModel
    base_model::ComponentModel
    modifier_model::ComponentModel
    method::Symbol
end

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
    method = get(params, :method, :spectral)
    NonStationaryVariance(base_model_obj, modifier_model_obj, method)
end

MODEL_TO_STRUCTURE_MAP[:nonstationaryvariance] = :spatial

function get_precomputes(
    m::NonStationaryVariance, M::NamedTuple, mod_data::Dict
)::NamedTuple
    params = mod_data[:params]

    base_model_spec_node = get(params, :base_node, nothing)
    if isnothing(base_model_spec_node)
        error("NonStationaryVariance requires a `base_node` parameter.")
    end
    base_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_base"),
        :type => base_model_spec_node.module_type,
        :variables => get(base_model_spec_node.args, :positional_args, []),
        :params => base_model_spec_node.args
    )
    base_precomputes = get_precomputes(m.base_model, M, base_mod_data)

    modifier_model_spec_node = get(params, :modifier_node, nothing)
    if isnothing(modifier_model_spec_node)
        error("NonStationaryVariance requires a `modifier_node` parameter.")
    end
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

    base_priors = get_priors(m.base_model, base_spec_for_priors, arch, outcome_idx, M)
    # The base model's sigma is not used; variance is controlled by the modifier.
    base_priors_cleaned = replace(base_priors, r".*sigma.*" => "")
    
    modifier_priors = get_priors(
        m.modifier_model, modifier_spec_for_priors, arch, outcome_idx, M
    )

    return """
        # --- Priors for NonStationaryVariance component: $(spec.key) ---
        $(base_priors_cleaned)
        $(modifier_priors)
    """
end

function get_updates(
    m::NonStationaryVariance, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    base_spec_key = Symbol("$(spec.key)_base")
    modifier_spec_key = Symbol("$(spec.key)_modifier")

    base_spec = (
        key = base_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.base_model)],
        var = spec.var,
        component_obj = m.base_model,
        params = spec.params,
        hyper = spec.hyper.base_precomputes
    )
    modifier_spec = (
        key = modifier_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.modifier_model)],
        var = spec.var,
        component_obj = m.modifier_model,
        params = spec.params,
        hyper = spec.hyper.modifier_precomputes
    )

    # Generate code for the modifier, which defines the log-sigma field.
    modifier_updates = get_updates(m.modifier_model, modifier_spec, arch, outcome_idx, M)
    modifier_latent_var = generate_full_variable_names(modifier_spec, arch, outcome_idx).latent
    modifier_logic = replace(modifier_updates, Regex("$(eta_target) \\.\\+= .*") => "")
    modifier_logic = replace(modifier_logic, modifier_latent_var => "log_sigma_field")

    # Generate code for the base model, which defines the unit-variance field.
    base_updates = get_updates(m.base_model, base_spec, arch, outcome_idx, M)
    base_latent_var = generate_full_variable_names(base_spec, arch, outcome_idx).latent
    base_logic = replace(base_updates, Regex("$(eta_target) \\.\\+= .*") => "")
    base_logic = replace(base_logic, Regex("$(base_latent_var) = .*") => "") # Remove scaling
    base_logic = replace(base_logic, base_latent_var => "base_latent_raw")

    return """
        # --- NonStationaryVariance Component: $(spec.key) ---
        let
            # 1. Realize the log-standard deviation field from the modifier model.
            $(modifier_logic)
            local spatially_varying_sigma = exp.(log_sigma_field)
            
            # 2. Realize the raw latent field from the base model.
            $(base_logic)
            
            # 3. Combine raw latent field with spatially varying sigma.
            local final_effect_latent = base_latent_raw .* spatially_varying_sigma
            
            # 4. Add the final effect to the linear predictor.
            $(eta_target) .+= view(final_effect_latent, M.s_idx)
        end
    """
end


function get_effects(
    m::NonStationaryVariance, chain::Chains, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device
    noise = M.noise
    
    structured_effects = Vector{Matrix{Float64}}()
    
    # --- Index Handling: Combine training and prediction sets on device ---
    s_idx_train = M.s_idx # Already on device
    s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx)
        s_idx_pred_cpu = get(PS.data, :s_idx, [])
        vcat(s_idx_train, to_device(s_idx_pred_cpu))
    else
        s_idx_train
    end
    N_total = length(s_idx_full)

    # --- Define specs for inner models ---
    base_spec = (
        key = Symbol("$(spec.key)_base"),
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.base_model)],
        var = spec.var,
        component_obj = m.base_model,
        params = get(spec.params, :base_node, (; args = Dict())).args,
        hyper = spec.hyper.base_precomputes
    )
    modifier_spec = (
        key = Symbol("$(spec.key)_modifier"),
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.modifier_model)],
        var = spec.var,
        component_obj = m.modifier_model,
        params = get(spec.params, :modifier_node, (; args = Dict())).args,
        hyper = spec.hyper.modifier_precomputes
    )

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        # 1. Get the modifier effect (spatially varying log-sigma)
        # This recursive call returns CPU arrays.
        modifier_results = get_effects(m.modifier_model, chain, modifier_spec, M, PS)
        log_sigma_field_cpu = modifier_results.structured[k]
        
        # Move to device and compute sigma
        log_sigma_field_device = to_device(log_sigma_field_cpu)
        spatially_varying_sigma_device = exp.(log_sigma_field_device)

        # 2. Get the base model's raw innovations
        base_p_names = generate_full_variable_names(base_spec, M.model_arch, k)
        base_innovations_name = _find_parameter(p_names, string(base_p_names.innovations), k, is_multivariate_model)
        
        if isempty(base_innovations_name)
            @warn "Base innovations for NonStationaryVariance component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end
        
        # Extract samples (CPU) and move to device
        base_innovations_cpu = get_params_matrix(chain, base_innovations_name, base_spec.hyper.n_latent)
        base_innovations_device = to_device(base_innovations_cpu)
        
        # 3. Reconstruct the base latent field (unit variance) on the device
        n_latent_base = base_spec.hyper.n_latent
        base_latent_raw_device = to_device(zeros(Float64, n_latent_base, n_samples))

        if m.method == :spectral
            U = base_spec.hyper.U # Already on device
            L = base_spec.hyper.L # Already on device
            diag_D = 1.0 ./ sqrt.(L .+ noise)
            if typeof(m.base_model) in [ICAR, Besag]; diag_D[1] = 0.0; end
            
            # Vectorized reconstruction
            base_latent_raw_device = U * (diag_D .* base_innovations_device')
        else # :cholesky or :cholesky_sparse
            # Use the pre-computed Cholesky factor
            F_base = base_spec.hyper.cholesky_factor
            for i in 1:n_samples
                raw_field = F_base.L' \ base_innovations_device[i, :]
                base_latent_raw_device[:, i] = raw_field .- mean(raw_field)
            end
        end

        # 4. Combine base field and sigma field on the device
        # Index the base field to match the observation structure
        base_field_at_obs_device = base_latent_raw_device[s_idx_full, :]
        
        # Element-wise product
        final_effect_k_device = base_field_at_obs_device .* spatially_varying_sigma_device
        
        # 5. Move final result back to CPU
        push!(structured_effects, Array(final_effect_k_device))
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
