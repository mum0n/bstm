# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    SVC <: ComponentModel

A component model for Spatially Varying Coefficients (SVC), allowing the effect of a
covariate to vary smoothly across space. It acts as an orchestrator, applying an inner
spatial `ComponentModel` to a specified covariate.

# Fields
- `covariate::Symbol`: The symbol of the covariate whose coefficient varies over space.
- `model::ComponentModel`: The inner spatial component model (e.g., `ICAR`, `BYM2`)
  that defines the spatial variation of the coefficient.
"""
struct SVC <: ComponentModel
    covariate::Symbol
    model::ComponentModel # The inner spatial model
end

# Add to the central component constructor registry.
# This constructor expects the inner model to be already resolved and passed in `params`.
# The `resolve_technical_primitive` function is responsible for resolving the inner model
# and then calling this constructor.
COMPONENT_CONSTRUCTORS[:svc] = (p, params) -> begin
    covariate = get(params, :covariate, error("SVC constructor requires a `covariate` parameter."))
    inner_model_obj = get(params, :inner_model_obj, error("SVC constructor requires an `inner_model_obj` parameter."))
    
    SVC(covariate, inner_model_obj)
end

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[SVC] = :spatial

"""
    get_datastructures!(m_type::Type{<:SVC}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `SVC`.
It ensures that the `covariate` variable is present in the data and delegates
data structure setup to the inner spatial model.
"""
function get_datastructures!(m_type::Type{<:SVC}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    cov_var = get(params, :covariate, nothing)

    if isnothing(cov_var)
        error("SVC model requires a `covariate` parameter.")
    end

    if cov_var != Symbol("1") && cov_var != :intercept && !hasproperty(M[:data], cov_var)
        error("Covariate variable ':$cov_var' for SVC model not found in data.")
    end

    # Delegate data structure setup to the inner spatial model
    inner_model_spec_node = get(params, :spatial_model_spec, nothing)
    if isnothing(inner_model_spec_node)
        error("SVC model requires a `spatial_model_spec` parameter in its module data.")
    end

    inner_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_inner"), # Create a unique key for the inner model
        :type => inner_model_spec_node.module_type,
        :variables => get(inner_model_spec_node.args, :positional_args, []),
        :params => inner_model_spec_node.args
    )
    
    # Call the inner model's get_datastructures!
    inner_model_type = typeof(mod_data[:component_obj].model) # Access the actual inner model type
    return get_datastructures!(inner_model_type, M, inner_mod_data)
end

"""
    get_precomputes(m::SVC, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `SVC`.
It delegates pre-computation to the inner spatial model and stores its results.
"""
function get_precomputes(m::SVC, M::NamedTuple, mod_data::Dict)::NamedTuple
    # Delegate pre-computation to the inner spatial model
    inner_model_spec_node = get(mod_data[:params], :spatial_model_spec, nothing)
    if isnothing(inner_model_spec_node)
        error("SVC model's mod_data missing `spatial_model_spec` for precomputes.")
    end

    inner_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_inner"),
        :type => inner_model_spec_node.module_type,
        :variables => get(inner_model_spec_node.args, :positional_args, []),
        :params => inner_model_spec_node.args
    )

    inner_precomputes = get_precomputes(m.model, M, inner_mod_data)
    
    # The SVC component itself doesn't have a Q_template, U, L directly,
    # but its inner model does. We store the inner model's precomputes.
    return (inner_precomputes=inner_precomputes,)
end

"""
    get_priors(m::SVC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `SVC`'s priors.
It delegates prior generation to the inner spatial model.
"""
function get_priors(m::SVC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    # Construct a spec for the inner model to pass to its get_priors function.
    inner_spec_key = Symbol("$(spec.key)_inner")
    inner_spec = (
        key = inner_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.model)], # Get structure of inner model
        var = spec.var, # Inherit variable from SVC for naming consistency
        component_obj = m.model,
        params = spec.params, # Pass SVC's params, as inner model's params are nested within
        Q_template = spec.hyper.inner_precomputes.Q_template, # Use inner model's Q_template
        scaling_factor = spec.hyper.inner_precomputes.scaling_factor,
        hyper = spec.hyper.inner_precomputes # Pass all inner precomputes as hyper for inner model
    )
    
    # Delegate prior generation to the inner spatial model
    return get_priors(m.model, inner_spec, arch, outcome_idx, M)
end

"""
    get_updates(m::SVC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `SVC`'s effect
and adding it to the linear predictor (`eta`).
It delegates the inner model's effect construction and then applies the covariate.
"""
function get_updates(m::SVC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    cov_var = m.covariate

    # Construct a spec for the inner model to pass to its get_updates function.
    inner_spec_key = Symbol("$(spec.key)_inner")
    inner_spec = (
        key = inner_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.model)],
        var = spec.var,
        component_obj = m.model,
        params = spec.params,
        Q_template = spec.hyper.inner_precomputes.Q_template,
        scaling_factor = spec.hyper.inner_precomputes.scaling_factor,
        hyper = spec.hyper.inner_precomputes
    )

    # Get the update code from the inner spatial model.
    inner_frags = get_updates(m.model, inner_spec, arch, outcome_idx, M)
    
    # The inner model's latent effect variable name will be based on its own spec key.
    inner_p_names = generate_full_variable_names(inner_spec, arch, outcome_idx)
    inner_latent_var = inner_p_names.latent

    # Remove any direct `eta` update from the inner fragment, as SVC will handle it.
    effect_app_regex = Regex("$(eta_target) \\.\\+= .*")
    update_inner_cleaned = replace(inner_frags, effect_app_regex => "")

    # The inner model's latent effect is assumed to be indexed by M.s_idx.
    is_intercept = (cov_var == Symbol("1") || cov_var == :intercept)
    
    application_code = if is_intercept
        """
        for i in 1:length($(eta_target))
            $(eta_target)[i] += $(inner_latent_var)[M.s_idx[i]]
        end
        """
    else
        """
        local cov_data::Vector{T} = T.(M.data[!, :$(cov_var)])
        for i in 1:length($(eta_target))
            $(eta_target)[i] += cov_data[i] * $(inner_latent_var)[M.s_idx[i]]
        end
        """
    end
    
    return """
        # --- Spatially Varying Coefficient (SVC) for: $(cov_var) (Component: $(spec.key)) ---
        $(update_inner_cleaned)
        $(application_code)
    """
end

"""
    get_effects(m::SVC, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `SVC`'s effect from the MCMC chain's posterior samples.
It reconstructs the inner model's effects and then applies the covariate.
"""
function get_effects(m::SVC, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    cov_var = m.covariate
    
    # Construct a spec for the inner model to pass to its get_effects function.
    inner_spec_key = Symbol("$(spec.key)_inner")
    inner_spec = (
        key = inner_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.model)],
        var = spec.var,
        component_obj = m.model,
        params = spec.params,
        Q_template = spec.hyper.inner_precomputes.Q_template,
        scaling_factor = spec.hyper.inner_precomputes.scaling_factor,
        hyper = spec.hyper.inner_precomputes
    )

    # Generate parameter names for the inner model
    inner_p_names = generate_full_variable_names(inner_spec, M.model_arch, nothing)

    # Reconstruct the inner model's effects. This returns a NamedTuple with mean, lower, upper.
    inner_effects_result = get_effects(m.model, chain, M, n_samples, outcomes_N, inner_p_names, inner_spec, PS, N_total)
    
    # The inner model's effects are structured (mean, lower, upper) over its latent dimension (M.s_N).
    inner_mean_effect = inner_effects_result.structured.mean
    inner_lower_effect = inner_effects_result.structured.lower
    inner_upper_effect = inner_effects_result.structured.upper

    # Extract covariate data. If PS is provided, use its data, otherwise use M.data.
    is_intercept = (cov_var == Symbol("1") || cov_var == :intercept)
    
    local cov_data_full
    if !is_intercept
        data_source = isnothing(PS) ? M.data : PS.data
        cov_data_full = data_source[!, cov_var]
    end
    
    # The inner effects are reconstructed for the full latent dimension (M.s_N).
    # We need to index these effects by M.s_idx (or PS.s_idx) and multiply by the covariate.
    idx_to_use = isnothing(PS) ? M.s_idx : PS.s_idx

    # Initialize arrays for the final SVC effect
    final_mean_effect = zeros(eltype(inner_mean_effect), length(idx_to_use))
    final_lower_effect = zeros(eltype(inner_lower_effect), length(idx_to_use))
    final_upper_effect = zeros(eltype(inner_upper_effect), length(idx_to_use))

    for i in 1:length(idx_to_use)
        s_idx_val = idx_to_use[i]
        cov_val = is_intercept ? 1.0 : cov_data_full[i]
        
        final_mean_effect[i] = cov_val * inner_mean_effect[s_idx_val]
        final_lower_effect[i] = cov_val * inner_lower_effect[s_idx_val]
        final_upper_effect[i] = cov_val * inner_upper_effect[s_idx_val]
    end

    return (structured=(mean=final_mean_effect, lower=final_lower_effect, upper=final_upper_effect),)
end
