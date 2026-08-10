"""
    TVC <: ComponentModel

A component model for Temporally Varying Coefficients (TVC), allowing the effect of a
covariate to vary smoothly over time. It acts as an orchestrator, applying an inner
temporal `ComponentModel` to a specified covariate.

# Version
v1.0.2 (2026-08-10)

# Mathematical Summary
A TVC model replaces a fixed regression coefficient \$\\beta\$ with a time-indexed
coefficient \$\\beta(t)\$. The contribution to the linear predictor \$\\eta\$ for an
observation at time \$t_i\$ with covariate value \$x_i\$ is given by:

\$\\eta_i = \\dots + \\beta(t_i) x_i\$

The time-varying coefficient \$\\boldsymbol{\\beta} = (\\beta(t_1), \\dots, \\beta(t_{t_N}))\$
is itself modeled as a latent temporal process, governed by the `inner_model`. For
example, if the inner model is a second-order random walk (RW2), then:

\$\\beta(t) = 2\\beta(t-1) - \\beta(t-2) + \\omega_t, \\quad \\omega_t \\sim \\mathcal{N}(0, \\sigma^2_{\\beta})\$

This allows the model to learn how the influence of a covariate changes over time.

# Computational Methods
The `TVC` component does not have its own methods. The computational method is
determined by the `method` parameter of the inner temporal model. For example, to
use a spectral decomposition for the time-varying coefficient, you would specify:
`... |> random(year, model=rw2, method=:spectral)`

# Fields
- `covariate::Symbol`: The symbol of the covariate whose coefficient varies over time.
- `model::ComponentModel`: The inner temporal component model (e.g., `AR1`, `RW2`)
  that defines the temporal variation of the coefficient.
"""
struct TVC <: ComponentModel
    covariate::Symbol
    model::ComponentModel
end

COMPONENT_TYPE_REGISTRY[:tvc] = TVC

# The constructor is called by `resolve_technical_primitive`, which resolves the
# inner model object and passes it in the `params` dictionary.
COMPONENT_CONSTRUCTORS[:tvc] = (p, params) -> begin
    covariate = get(params, :covariate, error("TVC constructor requires a `covariate` parameter."))
    inner_model_obj = get(params, :inner_model_obj, error("TVC constructor requires an `inner_model_obj` parameter."))
    
    TVC(covariate, inner_model_obj)
end

MODEL_TO_STRUCTURE_MAP[:tvc] = :temporal


"""
    get_datastructures!(m_type::Type{<:TVC}, M::Dict, mod_data::Dict)::Bool

Data-dependent setup for the TVC component.
Ensures the specified `covariate` exists in the data and delegates further temporal
setup to the inner temporal model's `get_datastructures!` method.
"""
function get_datastructures!(m_type::Type{<:TVC}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    cov_var = get(params, :covariate, nothing)

    if isnothing(cov_var)
        error("TVC model '$(mod_data[:key])' requires a `covariate` parameter.")
    end

    if !hasproperty(M[:data], cov_var)
        error("Covariate ':$cov_var' for TVC model '$(mod_data[:key])' not found in data.")
    end

    # Delegate data structure setup to the inner temporal model.
    inner_model_spec_node = get(params, :temporal_model_spec, nothing)
    if isnothing(inner_model_spec_node)
        error("TVC model '$(mod_data[:key])' is missing the inner `temporal_model_spec`.")
    end

    # The component object `m` is not yet the final TVC object at this stage,
    # so we need to get the inner model type from the parsed formula node.
    inner_model_name = get(inner_model_spec_node.args, :model, :rw2)
    if !haskey(COMPONENT_TYPE_REGISTRY, inner_model_name)
        error("Inner model ':$inner_model_name' for TVC not found in COMPONENT_TYPE_REGISTRY.")
    end
    inner_model_type = COMPONENT_TYPE_REGISTRY[inner_model_name]

    inner_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_inner"),
        :type => inner_model_spec_node.module_type,
        :variables => get(inner_model_spec_node.args, :positional_args, []),
        :params => inner_model_spec_node.args
    )
    
    return get_datastructures!(inner_model_type, M, inner_mod_data)
end

"""
    get_precomputes(m::TVC, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes structures for the TVC component.
Delegates pre-computation to the inner temporal model and stores the results.
"""
function get_precomputes(m::TVC, M::NamedTuple, mod_data::Dict)::NamedTuple
    inner_model_spec_node = get(mod_data[:params], :temporal_model_spec, nothing)
    if isnothing(inner_model_spec_node)
        error("TVC model '$(mod_data[:key])' missing `temporal_model_spec` for precomputes.")
    end

    inner_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_inner"),
        :type => inner_model_spec_node.module_type,
        :variables => get(inner_model_spec_node.args, :positional_args, []),
        :params => inner_model_spec_node.args
    )
    
    inner_precomputes = get_precomputes(m.model, M, inner_mod_data)
    
    return (inner_precomputes=inner_precomputes,)
end

"""
    get_priors(m::TVC, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the TVC component's priors.
Delegates prior generation to the inner temporal model, ensuring that parameter
names for the inner model are correctly scoped to avoid collisions.
"""
function get_priors(m::TVC, spec::NamedTuple, arch::String, outcome_idx, M)::String
    inner_spec_key = Symbol("$(spec.key)_inner")
    inner_spec = (
        key = inner_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.model)],
        var = spec.var,
        component_obj = m.model,
        params = spec.params,
        hyper = get(spec.hyper, :inner_precomputes, NamedTuple())
    )
    
    return get_priors(m.model, inner_spec, arch, outcome_idx, M)
end

"""
    get_updates(m::TVC, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates Turing code for the TVC component's effect.
Delegates the construction of the time-varying coefficient field to the inner
model, and then multiplies this field by the specified covariate value for each
observation before adding it to the linear predictor `eta`.
"""
function get_updates(
    m::TVC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    cov_var = m.covariate

    inner_spec_key = Symbol("$(spec.key)_inner")
    inner_spec = (
        key = inner_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.model)],
        var = spec.var,
        component_obj = m.model,
        params = spec.params,
        hyper = get(spec.hyper, :inner_precomputes, NamedTuple())
    )

    inner_updates_code = get_updates(m.model, inner_spec, arch, outcome_idx, M)
    inner_p_names = generate_full_variable_names(inner_spec, arch, outcome_idx)
    inner_latent_var = inner_p_names.latent

    # The inner component's `get_updates` will generate code that includes adding
    # its effect to `eta`. We must remove this part to prevent double-counting,
    # as the TVC component applies the effect in its own way (multiplied by the covariate).
    effect_app_regex = Regex("$(eta_target) .*\\.=")
    update_inner_cleaned = replace(
        inner_updates_code, effect_app_regex => "# (eta update handled by TVC)"
    )

    application_code = "$(eta_target) .+= M.data[!, :$(cov_var)] .* view($(inner_latent_var), M.t_idx)"
    
    return """
        # --- Temporally Varying Coefficient (TVC) for: $(cov_var) ---
        # 1. Generate the latent temporal field for the coefficient.
        $(update_inner_cleaned)

        # 2. Apply the time-varying coefficient to the linear predictor.
        $(application_code)
    """
end

"""
    get_effects(m::TVC, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_total)::NamedTuple

Reconstructs the TVC component's effect from posteriors.
Reconstructs the inner temporal field for each posterior sample and then multiplies
it by the covariate values to get the final TVC effect for each observation.
"""
function get_effects(
    m::TVC, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    cov_var = m.covariate
    
    inner_spec_key = Symbol("$(spec.key)_inner")
    inner_spec = (
        key = inner_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.model)],
        var = spec.var,
        component_obj = m.model,
        params = spec.params,
        hyper = get(spec.hyper, :inner_precomputes, NamedTuple())
    )

    # Recursively call get_effects on the inner temporal model
    inner_effects_result = get_effects(
        m.model, chain, M, n_samples, outcomes_N, inner_spec, PS, N_total
    )
    
    # Get the full covariate vector (training + prediction)
    cov_data_full = if !isnothing(PS) && hasproperty(PS.data, cov_var)
        vcat(M.data[!, cov_var], PS.data[!, cov_var])
    else
        M.data[!, cov_var]
    end

    # The inner effect is already indexed to the observation level, so we just
    # need to multiply it by the covariate.
    structured_effects = Vector{Matrix{Float64}}()
    for k in 1:outcomes_N
        temporal_effect_k = inner_effects_result.structured[k]
        
        # Element-wise multiplication of the covariate and the temporal effect
        final_effect_k = temporal_effect_k .* cov_data_full
        
        push!(structured_effects, final_effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
