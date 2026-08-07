# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    TVC <: ComponentModel

A component model for Temporally Varying Coefficients (TVC), allowing the effect of a
covariate to vary smoothly over time. It acts as an orchestrator, applying an inner
temporal `ComponentModel` to a specified covariate.

# Fields
- `covariate::Symbol`: The symbol of the covariate whose coefficient varies over time.
- `model::ComponentModel`: The inner temporal component model (e.g., `AR1`, `RW2`)
  that defines the temporal variation of the coefficient.
"""
struct TVC <: ComponentModel
    covariate::Symbol
    model::ComponentModel # The inner temporal model
end

# Add to the central component constructor registry.
# This constructor expects the inner model to be already resolved and passed in `params`.
# The `resolve_technical_primitive` function is responsible for resolving the inner model
# and then calling this constructor.
COMPONENT_CONSTRUCTORS[:tvc] = (p, params) -> begin
    covariate = get(params, :covariate, error("TVC constructor requires a `covariate` parameter."))
    inner_model_obj = get(params, :inner_model_obj, error("TVC constructor requires an `inner_model_obj` parameter."))
    
    TVC(covariate, inner_model_obj)
end

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[TVC] = :temporal

"""
    get_datastructures!(m_type::Type{<:TVC}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `TVC`.
It ensures that the `covariate` variable is present in the data and delegates
data structure setup to the inner temporal model.
"""
function get_datastructures!(m_type::Type{<:TVC}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    cov_var = get(params, :covariate, nothing)

    if isnothing(cov_var)
        error("TVC model requires a `covariate` parameter.")
    end

    if !hasproperty(M[:data], cov_var)
        error("Covariate variable ':$cov_var' for TVC model not found in data.")
    end

    # Delegate data structure setup to the inner temporal model
    # The inner model's mod_data needs to be constructed from the TVC's mod_data.
    inner_model_spec_node = get(params, :temporal_model_spec, nothing)
    if isnothing(inner_model_spec_node)
        error("TVC model requires a `temporal_model_spec` parameter in its module data.")
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
    get_precomputes(m::TVC, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `TVC`.
It delegates pre-computation to the inner temporal model and stores its results.
"""
function get_precomputes(m::TVC, M::NamedTuple, mod_data::Dict)::NamedTuple
    # Delegate pre-computation to the inner temporal model
    inner_model_spec_node = get(mod_data[:params], :temporal_model_spec, nothing)
    if isnothing(inner_model_spec_node)
        error("TVC model's mod_data missing `temporal_model_spec` for precomputes.")
    end

    inner_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_inner"),
        :type => inner_model_spec_node.module_type,
        :variables => get(inner_model_spec_node.args, :positional_args, []),
        :params => inner_model_spec_node.args
    )

    inner_precomputes = get_precomputes(m.model, M, inner_mod_data)
    
    # The TVC component itself doesn't have a Q_template, U, L directly,
    # but its inner model does. We store the inner model's precomputes.
    return (inner_precomputes=inner_precomputes,)
end

"""
    get_priors(m::TVC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `TVC`'s priors.
It delegates prior generation to the inner temporal model.
"""
function get_priors(m::TVC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    # Construct a spec for the inner model to pass to its get_priors function.
    # The inner model's key should be unique, derived from the TVC's key.
    inner_spec_key = Symbol("$(spec.key)_inner")
    inner_spec = (
        key = inner_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.model)], # Get structure of inner model
        var = spec.var, # Inherit variable from TVC for naming consistency
        component_obj = m.model,
        params = spec.params, # Pass TVC's params, as inner model's params are nested within
        Q_template = spec.hyper.inner_precomputes.Q_template, # Use inner model's Q_template
        scaling_factor = spec.hyper.inner_precomputes.scaling_factor,
        hyper = spec.hyper.inner_precomputes # Pass all inner precomputes as hyper for inner model
    )
    
    # Delegate prior generation to the inner temporal model
    return get_priors(m.model, inner_spec, arch, outcome_idx, M)
end

"""
    get_updates(m::TVC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `TVC`'s effect
and adding it to the linear predictor (`eta`).
It delegates the inner model's effect construction and then applies the covariate.
"""
function get_updates(m::TVC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
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

    # Get the update code from the inner temporal model.
    # We need to ensure the inner model's latent effect is assigned to a variable
    # that we can then use, and its direct eta update is suppressed.
    inner_frags = get_updates(m.model, inner_spec, arch, outcome_idx, M)
    
    # The inner model's latent effect variable name will be based on its own spec key.
    inner_p_names = generate_full_variable_names(inner_spec, arch, outcome_idx)
    inner_latent_var = inner_p_names.latent

    # Remove any direct `eta` update from the inner fragment, as TVC will handle it.
    # This regex needs to be robust to different spacing and variable names.
    # It looks for `eta_target .+= ...` or `eta_target[i] += ...`
    effect_app_regex_1 = Regex("$(eta_target) \\.\\+= .*")
    effect_app_regex_2 = Regex("for i in 1:length\\($(eta_target)\\)\\s*$(eta_target)\\[i\\] \\+= .*end")
    
    update_inner_cleaned = replace(inner_frags, effect_app_regex_1 => "")
    update_inner_cleaned = replace(update_inner_cleaned, effect_app_regex_2 => "")

    # The inner model's latent effect is assumed to be indexed by M.t_idx.
    # This is consistent with the original `_generate_component_code_fragments` logic.
    application_code = """
        local cov_data::Vector{T} = T.(M.data[!, :$(cov_var)])
        for i in 1:length($(eta_target))
            $(eta_target)[i] += cov_data[i] * $(inner_latent_var)[M.t_idx[i]]
        end
    """
    
    return """
        # --- Temporally Varying Coefficient (TVC) for: $(cov_var) (Component: $(spec.key)) ---
        $(update_inner_cleaned)
        $(application_code)
    """
end

"""
    get_effects(m::TVC, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `TVC`'s effect from the MCMC chain's posterior samples.
It reconstructs the inner model's effects and then applies the covariate.
"""
function get_effects(m::TVC, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
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
    inner_p_names = generate_full_variable_names(inner_spec, M.model_arch, nothing) # Pass nothing for outcome_idx, as get_effects handles multivariate internally

    # Reconstruct the inner model's effects
    inner_effects_result = get_effects(m.model, chain, M, n_samples, outcomes_N, inner_p_names, inner_spec, PS, N_total)
    
    # The inner model's effects are typically structured (mean, lower, upper) over its latent dimension.
    # For TVC, this is usually M.t_N.
    inner_mean_effect = inner_effects_result.structured.mean
    inner_lower_effect = inner_effects_result.structured.lower
    inner_upper_effect = inner_effects_result.structured.upper

    # Extract covariate data. If PS is provided, use its data, otherwise use M.data.
    data_source = isnothing(PS) ? M.data : PS.data
    cov_data_full = data_source[!, cov_var]
    
    # The inner effects are typically reconstructed for the full latent dimension (e.g., M.t_N).
    # We need to index these effects by M.t_idx (or PS.t_idx) and multiply by the covariate.
    idx_to_use = isnothing(PS) ? M.t_idx : PS.t_idx

    # Initialize arrays for the final TVC effect
    final_mean_effect = zeros(eltype(inner_mean_effect), length(idx_to_use))
    final_lower_effect = zeros(eltype(inner_lower_effect), length(idx_to_use))
    final_upper_effect = zeros(eltype(inner_upper_effect), length(idx_to_use))

    for i in 1:length(idx_to_use)
        t_idx_val = idx_to_use[i]
        cov_val = cov_data_full[i]
        
        final_mean_effect[i] = cov_val * inner_mean_effect[t_idx_val]
        final_lower_effect[i] = cov_val * inner_lower_effect[t_idx_val]
        final_upper_effect[i] = cov_val * inner_upper_effect[t_idx_val]
    end

    return (structured=(mean=final_mean_effect, lower=final_lower_effect, upper=final_upper_effect),)
end
