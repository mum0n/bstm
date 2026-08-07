# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    NonStationaryVariance <: ComponentModel

A component model for non-stationary variance, where the standard deviation of a base
spatial effect varies across space according to a modifier spatial effect.
It acts as an orchestrator, combining a `base_model` (typically a spatial GMRF)
with a `modifier_model` (typically a spatial smoother) to create a spatially
varying standard deviation.

# Fields
- `base_model::ComponentModel`: The underlying spatial component model (e.g., `ICAR`, `Besag`)
  whose variance is being modified.
- `modifier_model::ComponentModel`: The spatial smoother component model (e.g., `PSpline`, `GP`)
  that defines the pattern of the log-standard deviation.
"""
struct NonStationaryVariance <: ComponentModel
    base_model::ComponentModel
    modifier_model::ComponentModel
end

# Add to the central component constructor registry.
# This constructor expects the inner models to be already resolved and passed in `params`.
# The `resolve_technical_primitive` function is responsible for resolving the inner models
# and then calling this constructor.
COMPONENT_CONSTRUCTORS[:nonstationary_variance] = (p, params) -> begin
    base_model_obj = get(params, :base_model_obj, error("NonStationaryVariance constructor requires a `base_model_obj` parameter."))
    modifier_model_obj = get(params, :modifier_model_obj, error("NonStationaryVariance constructor requires a `modifier_model_obj` parameter."))
    
    NonStationaryVariance(base_model_obj, modifier_model_obj)
end

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[NonStationaryVariance] = :spatial

"""
    get_datastructures!(m_type::Type{<:NonStationaryVariance}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `NonStationaryVariance` component.
It delegates data structure setup to both the `base_model` and `modifier_model`.
"""
function get_datastructures!(m_type::Type{<:NonStationaryVariance}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]

    # Delegate data structure setup to the base model (spatial GMRF)
    base_model_spec_node = get(params, :base_model_spec, nothing)
    if isnothing(base_model_spec_node)
        error("NonStationaryVariance model requires a `base_model_spec` parameter in its module data.")
    end
    base_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_base"),
        :type => base_model_spec_node.module_type,
        :variables => get(base_model_spec_node.args, :positional_args, []),
        :params => base_model_spec_node.args
    )
    base_model_type = typeof(mod_data[:component_obj].base_model)
    get_datastructures!(base_model_type, M, base_mod_data)

    # Delegate data structure setup to the modifier model (spatial smoother)
    modifier_model_spec_node = get(params, :modifier_model_spec, nothing)
    if isnothing(modifier_model_spec_node)
        error("NonStationaryVariance model requires a `modifier_model_spec` parameter in its module data.")
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
    get_precomputes(m::NonStationaryVariance, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `NonStationaryVariance` component.
It delegates pre-computation to both the `base_model` and `modifier_model` and stores
their results.
"""
function get_precomputes(m::NonStationaryVariance, M::NamedTuple, mod_data::Dict)::NamedTuple
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

    # The modifier model might produce a basis matrix that needs to be stored.
    # This key is used in the `get_updates` function.
    modifier_basis_key = if haskey(modifier_mod_data[:params], :basis_key)
        modifier_mod_data[:params][:basis_key]
    else
        Symbol(join(modifier_mod_data[:variables], "_"))
    end

    return (
        base_precomputes=base_precomputes,
        modifier_precomputes=modifier_precomputes,
        modifier_basis_key=modifier_basis_key
    )
end

"""
    get_priors(m::NonStationaryVariance, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `NonStationaryVariance` component's priors.
It delegates prior generation to both the `base_model` and `modifier_model`, and
defines the raw latent field for the base model.
"""
function get_priors(m::NonStationaryVariance, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    # Generate full variable names for the base and modifier models, prefixed by this component's key.
    base_spec_key = Symbol("$(spec.key)_base")
    modifier_spec_key = Symbol("$(spec.key)_modifier")

    base_spec_for_priors = (
        key = base_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.base_model)],
        var = spec.var, # Inherit variable from NonStationaryVariance for naming consistency
        component_obj = m.base_model,
        params = spec.params, # Pass NonStationaryVariance's params, as inner model's params are nested within
        Q_template = spec.hyper.base_precomputes.Q_template,
        scaling_factor = spec.hyper.base_precomputes.scaling_factor,
        hyper = spec.hyper.base_precomputes
    )
    modifier_spec_for_priors = (
        key = modifier_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.modifier_model)],
        var = spec.var,
        component_obj = m.modifier_model,
        params = spec.params,
        Q_template = spec.hyper.modifier_precomputes.Q_template,
        scaling_factor = spec.hyper.modifier_precomputes.scaling_factor,
        hyper = spec.hyper.modifier_precomputes
    )

    base_priors = get_priors(m.base_model, base_spec_for_priors, arch, outcome_idx, M)
    modifier_priors = get_priors(m.modifier_model, modifier_spec_for_priors, arch, outcome_idx, M)

    # The base model's latent field is sampled as raw innovations.
    base_p_names = generate_full_variable_names(base_spec_for_priors, arch, outcome_idx)
    n_latent_base = spec.hyper.base_precomputes.n_latent
    base_raw_prior = "$(base_p_names.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent_base)), I), :$(base_p_names.raw))"

    return """
        # --- Priors for NonStationaryVariance component: $(spec.key) ---
        $(base_priors)
        $(modifier_priors)
        $(base_raw_prior)
    """
end

"""
    get_updates(m::NonStationaryVariance, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `NonStationaryVariance` component's effect
and adding it to the linear predictor (`eta`).
"""
function get_updates(m::NonStationaryVariance, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    # Generate full variable names for the base and modifier models, prefixed by this component's key.
    base_spec_key = Symbol("$(spec.key)_base")
    modifier_spec_key = Symbol("$(spec.key)_modifier")

    base_spec_for_updates = (
        key = base_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.base_model)],
        var = spec.var,
        component_obj = m.base_model,
        params = spec.params,
        Q_template = spec.hyper.base_precomputes.Q_template,
        scaling_factor = spec.hyper.base_precomputes.scaling_factor,
        hyper = spec.hyper.base_precomputes
    )
    modifier_spec_for_updates = (
        key = modifier_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.modifier_model)],
        var = spec.var,
        component_obj = m.modifier_model,
        params = spec.params,
        Q_template = spec.hyper.modifier_precomputes.Q_template,
        scaling_factor = spec.hyper.modifier_precomputes.scaling_factor,
        hyper = spec.hyper.modifier_precomputes
    )

    modifier_frags = get_updates(m.modifier_model, modifier_spec_for_updates, arch, outcome_idx, M)
    base_p_names = generate_full_variable_names(base_spec_for_updates, arch, outcome_idx)
    modifier_p_names = generate_full_variable_names(modifier_spec_for_updates, arch, outcome_idx)

    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    # Remove any direct `eta` update from the modifier fragment, as NonStationaryVariance will handle it.
    effect_app_regex = Regex("$(eta_target) \\.\\+= .*")
    modifier_update_cleaned = replace(modifier_frags, effect_app_regex => "")

    n_latent_base = spec.hyper.base_precomputes.n_latent
    modifier_basis_key = spec.hyper.modifier_basis_key

    # The base model's latent field is sampled as raw innovations.
    # We need to reconstruct it using its Q_template and the raw innovations.
    # The spectral method is preferred for AD-compatibility.
    base_model_type = typeof(m.base_model)
    base_model_type_sym = Symbol(lowercase(string(base_model_type)))

    base_latent_reconstruction_code = """
        local Q_base_template = spec_registry[:$(base_spec_for_updates.key)].hyper.Q_template
        local F_base = cholesky(Symmetric(Matrix(Q_base_template) + M.noise * I))
        local base_latent_raw::Vector{T} = F_base.L' \\ $(base_p_names.raw)
        
        if $(base_model_type_sym) in [:icar, :besag]
            Turing.@addlogprob! logpdf(Normal(T(0), T(0.001) * $(n_latent_base)), sum(base_latent_raw))
        end
    """

    return """
        # --- NonStationaryVariance Component: $(spec.key) ---
        
        # 1. Realize the log-standard deviation field from the modifier model.
        $(modifier_update_cleaned)
        
        local log_sigma_field::Vector{T} = M.basis_matrices[:$(modifier_basis_key)] * $(modifier_p_names.latent)
        local spatially_varying_sigma::Vector{T} = exp.(log_sigma_field)
        
        # 2. Realize the raw latent field from the base model.
        $(base_latent_reconstruction_code)
        
        # 3. Combine raw latent field with spatially varying sigma.
        local final_effect_latent = zeros(T, M.s_N)
        for s_idx_val in 1:M.s_N
            final_effect_latent[s_idx_val] = base_latent_raw[s_idx_val] * spatially_varying_sigma[s_idx_val]
        end
        
        # 4. Add the final effect to the linear predictor, indexed by the observation's spatial unit.
        eta .+= final_effect_latent[M.s_idx]
    """
end

"""
    get_effects(m::NonStationaryVariance, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `NonStationaryVariance` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::NonStationaryVariance, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    # Generate full variable names for the base and modifier models, prefixed by this component's key.
    base_spec_key = Symbol("$(spec.key)_base")
    modifier_spec_key = Symbol("$(spec.key)_modifier")

    base_spec_for_effects = (
        key = base_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.base_model)],
        var = spec.var,
        component_obj = m.base_model,
        params = spec.params,
        Q_template = spec.hyper.base_precomputes.Q_template,
        scaling_factor = spec.hyper.base_precomputes.scaling_factor,
        hyper = spec.hyper.base_precomputes
    )
    modifier_spec_for_effects = (
        key = modifier_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.modifier_model)],
        var = spec.var,
        component_obj = m.modifier_model,
        params = spec.params,
        Q_template = spec.hyper.modifier_precomputes.Q_template,
        scaling_factor = spec.hyper.modifier_precomputes.scaling_factor,
        hyper = spec.hyper.modifier_precomputes
    )

    # Reconstruct effects for the modifier model (log-sigma field)
    # The modifier's get_effects will return structured effects for its own latent dimension.
    modifier_effects_result = get_effects(m.modifier_model, chain, M, n_samples, outcomes_N, generate_full_variable_names(modifier_spec_for_effects, M.model_arch, nothing), modifier_spec_for_effects, PS, N_total)
    
    # We need the full reconstructed log_sigma_field, not just its summary.
    # This requires re-running the modifier's update logic for each sample.

    # Extract raw samples for modifier's latent field
    modifier_p_names = generate_full_variable_names(modifier_spec_for_effects, M.model_arch, nothing)
    modifier_raw_samples = get(chain, modifier_p_names.raw)
    
    # Assuming modifier has a sigma parameter (e.g., for splines)
    modifier_sigma_samples = if hasproperty(m.modifier_model, :sigma)
        get(chain, modifier_p_names.sigma)
    else
        fill(1.0, n_samples) # Default to 1 if no sigma
    end

    # Reconstruct the log_sigma_field for all spatial units
    modifier_basis_key = spec.hyper.modifier_basis_key
    B_modifier_train = M.basis_matrices[modifier_basis_key]
    B_modifier_full = if !isnothing(PS) && haskey(PS, :basis_matrices) && haskey(PS.basis_matrices, modifier_basis_key)
        vcat(B_modifier_train, PS.basis_matrices[modifier_basis_key])
    else
        B_modifier_train
    end
    n_latent_modifier = size(B_modifier_full, 2) # Number of basis functions

    log_sigma_field_samples = zeros(n_samples, M.s_N)
    for i in 1:n_samples
        current_modifier_raw = modifier_raw_samples[i, :]
        current_modifier_sigma = modifier_sigma_samples[i]

        # Reconstruct modifier's latent coefficients (assuming RW-like penalty)
        U_mod = modifier_spec_for_effects.hyper.U
        L_mod = modifier_spec_for_effects.hyper.L
        
        # For splines, diff_order is typically a field. For other smoothers, it might be 0.
        diff_order_mod = get(m.modifier_model, :diff_order, 0) 
        
        diag_D_mod = current_modifier_sigma ./ sqrt.(L_mod .+ M.noise)
        for j in 1:diff_order_mod; diag_D_mod[j] = 0.0; end # Enforce sum-to-zero for intrinsic penalties
        
        modifier_coeffs = U_mod * (diag_D_mod .* current_modifier_raw)
        
        # Multiply by basis matrix to get log_sigma_field
        log_sigma_field_samples[i, :] = B_modifier_full * modifier_coeffs
    end
    spatially_varying_sigma_samples = exp.(log_sigma_field_samples)

    # Reconstruct effects for the base model (raw latent field)
    base_p_names = generate_full_variable_names(base_spec_for_effects, M.model_arch, nothing)
    base_raw_samples = get(chain, base_p_names.raw)
    n_latent_base = spec.hyper.base_precomputes.n_latent

    Q_base_template = spec.hyper.base_precomputes.Q_template
    noise = M.noise
    F_base = cholesky(Symmetric(Matrix(Q_base_template) + noise * I))
    base_latent_raw_samples = zeros(n_samples, n_latent_base)
    for i in 1:n_samples
        base_latent_raw_samples[i, :] = F_base.L' \ base_raw_samples[i, :]
    end

    # Combine raw latent field with spatially varying sigma
    reconstructed_effects = zeros(n_samples, M.s_N)
    for i in 1:n_samples
        reconstructed_effects[i, :] = base_latent_raw_samples[i, :] .* spatially_varying_sigma_samples[i, :]
    end

    # Determine indices for reconstruction (training or prediction)
    idx_to_use = isnothing(PS) ? M.s_idx : PS.s_idx
    
    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    indexed_mean = mean_effect[idx_to_use]
    indexed_lower = lower_ci[idx_to_use]
    indexed_upper = upper_ci[idx_to_use]

    return (structured=(mean=indexed_mean, lower=indexed_lower, upper=indexed_upper),)
end
