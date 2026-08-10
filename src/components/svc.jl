"""
    SVC <: ComponentModel

A component model for Spatially Varying Coefficients (SVC), allowing the effect of a
covariate to vary smoothly across space. It acts as an orchestrator, applying an inner
spatial `ComponentModel` to a specified covariate.

# Version
v1.1.1 (2026-08-10)

# Mathematical Summary
An SVC model replaces a fixed regression coefficient \$\\beta\$ with a spatially-indexed
coefficient \$\\beta(s)\$. The contribution to the linear predictor \$\\eta\$ for an
observation at location \$s_i\$ with covariate value \$x_i\$ is given by:

\$\\eta_i = \\dots + \\beta(s_i) x_i\$

The spatially varying coefficient \$\\boldsymbol{\\beta} = (\\beta(s_1), \\dots, \\beta(s_{s_N}))\$
is itself modeled as a latent Gaussian Process or GMRF, governed by the `inner_model`.

# Computational Methods
The `SVC` component does not have its own methods. The computational method is
determined by the `method` parameter of the inner spatial model. For example, to
use a spectral decomposition for the spatially varying coefficient, you would specify:
`... |> random(s_idx, model=icar, method=:spectral)`

# Fields
- `covariate::Symbol`: The symbol of the covariate whose coefficient varies over space.
- `model::ComponentModel`: The inner spatial component model (e.g., `ICAR`, `BYM2`)
  that defines the spatial variation of the coefficient.
"""
struct SVC <: ComponentModel
    covariate::Symbol
    model::ComponentModel
end

COMPONENT_TYPE_REGISTRY[:svc] = SVC

COMPONENT_CONSTRUCTORS[:svc] = (p, params) -> begin
    covariate = get(params, :covariate, error("SVC constructor requires a `covariate` parameter."))
    inner_model_obj = get(params, :inner_model_obj, error("SVC constructor requires an `inner_model_obj` parameter."))
    
    SVC(covariate, inner_model_obj)
end

MODEL_TO_STRUCTURE_MAP[:svc] = :spatial

"""
    get_datastructures!(m_type::Type{<:SVC}, M::Dict, mod_data::Dict)::Bool

Data-dependent setup for the SVC component.
Ensures the specified `covariate` exists in the data and delegates further spatial
setup to the inner spatial model's `get_datastructures!` method.
"""
function get_datastructures!(m_type::Type{<:SVC}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    cov_var = get(params, :covariate, nothing)

    if isnothing(cov_var)
        error("SVC model '$(mod_data[:key])' requires a `covariate` parameter.")
    end

    # The covariate '1' or :intercept is a special case for a spatially varying intercept.
    if cov_var != Symbol("1") && cov_var != :intercept && !hasproperty(M[:data], cov_var)
        error("Covariate ':$cov_var' for SVC model '$(mod_data[:key])' not found in data.")
    end

    # Delegate data structure setup to the inner spatial model.
    # This ensures that s_idx, s_N, W, etc., are correctly set up based on the
    # inner model's requirements.
    inner_model_spec_node = get(params, :spatial_model_spec, nothing)
    if isnothing(inner_model_spec_node)
        error("SVC model '$(mod_data[:key])' is missing the inner `spatial_model_spec`.")
    end

    # The component object `m` is not yet the final SVC object at this stage,
    # so we need to get the inner model type from the parsed formula node.
    inner_model_name = get(inner_model_spec_node.args, :model, :icar)
    if !haskey(COMPONENT_TYPE_REGISTRY, inner_model_name)
        error("Inner model ':$inner_model_name' for SVC not found in COMPONENT_TYPE_REGISTRY.")
    end
    inner_model_type = COMPONENT_TYPE_REGISTRY[inner_model_name]

    inner_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_inner"), # Create a unique key for the inner model
        :type => inner_model_spec_node.module_type,
        :variables => get(inner_model_spec_node.args, :positional_args, []),
        :params => inner_model_spec_node.args
    )
    
    return get_datastructures!(inner_model_type, M, inner_mod_data)
end

"""
    get_precomputes(m::SVC, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes structures for the SVC component.
Delegates pre-computation to the inner spatial model (e.g., to build its
precision matrix template and spectral decomposition) and stores the results.
"""
function get_precomputes(m::SVC, M::NamedTuple, mod_data::Dict)::NamedTuple
    inner_model_spec_node = get(mod_data[:params], :spatial_model_spec, nothing)
    if isnothing(inner_model_spec_node)
        error("SVC model '$(mod_data[:key])' missing `spatial_model_spec` for precomputes.")
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
    get_priors(m::SVC, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the SVC component's priors.
Delegates prior generation to the inner spatial model, ensuring that parameter
names for the inner model are correctly scoped to avoid collisions.
"""
function get_priors(m::SVC, spec::NamedTuple, arch::String, outcome_idx, M)::String
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
    get_updates(m::SVC, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates Turing code for the SVC component's effect. It delegates the construction
of the spatially varying coefficient field to the inner model and then multiplies
this field by the specified covariate value for each observation.
"""
function get_updates(
    m::SVC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
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
    # as the SVC component applies the effect in its own way.
    effect_app_regex = Regex("$(eta_target) \\.\\+= .*")
    update_inner_cleaned = replace(
        inner_updates_code, effect_app_regex => "# (eta update handled by SVC)"
    )

    is_intercept = (cov_var == Symbol("1") || cov_var == :intercept)
    
    application_code = if is_intercept
        "$(eta_target) .+= view($(inner_latent_var), M.s_idx)"
    else
        "$(eta_target) .+= M.data[!, :$(cov_var)] .* view($(inner_latent_var), M.s_idx)"
    end
    
    return """
        # --- Spatially Varying Coefficient (SVC) for: $(cov_var) ---
        # 1. Generate the latent spatial field for the coefficient.
        $(update_inner_cleaned)

        # 2. Apply the spatially varying coefficient to the linear predictor.
        $(application_code)
    """
end

"""
    get_effects(m::SVC, chain, M::NamedTuple, ...)

Reconstructs the SVC component's effect from posteriors. It reconstructs the inner
spatial field for each posterior sample and then multiplies it by the covariate
values to get the final SVC effect for each observation.
"""
function get_effects(
    m::SVC, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
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

    # Recursively call get_effects on the inner spatial model
    inner_effects_result = get_effects(
        m.model, chain, M, n_samples, outcomes_N, inner_spec, PS, N_total
    )
    
    is_intercept = (cov_var == Symbol("1") || cov_var == :intercept)
    cov_data_full = if is_intercept
        ones(Float64, N_total)
    else
        train_data = M.data[!, cov_var]
        if !isnothing(PS) && hasproperty(PS.data, cov_var)
            vcat(train_data, PS.data[!, cov_var])
        else
            train_data
        end
    end

    # The inner effect is already indexed to the observation level, so we just
    # need to multiply it by the covariate.
    structured_effects = Vector{Matrix{Float64}}()
    for k in 1:outcomes_N
        spatial_effect_k = inner_effects_result.structured[k]
        
        # Element-wise multiplication of the covariate and the spatial effect
        final_effect_k = spatial_effect_k .* cov_data_full
        
        push!(structured_effects, final_effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
