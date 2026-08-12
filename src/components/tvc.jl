"""
    TVC <: ComponentModel

A component model for Temporally Varying Coefficients (TVC), allowing the effect of a
covariate to vary smoothly over time. It acts as an orchestrator, applying an inner
temporal `ComponentModel` to a specified covariate.

# Version
v1.1.0 (2026-08-11)

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

# Inputs
- **Required**:
  - A covariate piped (`|>`) into a temporal `random()` module, e.g., `covariate |> random(year, model=ar1)`.
  - The inner `random()` call must specify a temporal model.
- **Optional**:
  - Priors for the inner temporal model are passed within the inner `random()` call.

# Outputs (Parameter Names)
- Parameter names are inherited from the inner temporal model and are prefixed with
  `_<key>_inner`. For example, if the main component key is `tvc_gdd`, the inner
  sigma would be `sigma_tvc_gdd_inner`.

# Key References
- Gelfand, A. E., Kim, H. J., Sirmans, C. F., & Banerjee, S. (2003). *Spatial
  modeling with spatially varying coefficient processes*. Journal of the American
  Statistical Association, 98(462), 387-396. (Conceptual analogue for space).
"""
struct TVC <: ComponentModel
    covariate::Symbol
    model::ComponentModel
end

COMPONENT_TYPE_REGISTRY[:tvc] = TVC

COMPONENT_CONSTRUCTORS[:tvc] = (p, params) -> begin
    covariate = get(params, :covariate, error("TVC constructor requires a `covariate` parameter."))
    inner_model_obj = get(params, :inner_model_obj, error("TVC constructor requires an `inner_model_obj` parameter."))
    
    TVC(covariate, inner_model_obj)
end

MODEL_TO_STRUCTURE_MAP[:tvc] = :temporal

function get_datastructures!(m_type::Type{<:TVC}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    cov_var = get(params, :covariate, nothing)

    if isnothing(cov_var)
        error("TVC model '$(mod_data[:key])' requires a `covariate` parameter.")
    end

    if !hasproperty(M[:data], cov_var)
        error("Covariate ':$cov_var' for TVC model '$(mod_data[:key])' not found in data.")
    end

    inner_model_spec_node = get(params, :temporal_model_spec, nothing)
    if isnothing(inner_model_spec_node)
        error("TVC model '$(mod_data[:key])' is missing the inner `temporal_model_spec`.")
    end

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

function get_priors(
    m::TVC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
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

    # Generate the code for the inner model. This code will contain an incorrect
    # reference to the spec_registry, as it doesn't know it's being wrapped.
    inner_updates_code = get_updates(m.model, inner_spec, arch, outcome_idx, M)
    inner_p_names = generate_full_variable_names(inner_spec, arch, outcome_idx)
    inner_latent_var = inner_p_names.latent
    
    # The generated code will try to access `spec_registry[:..._inner].hyper`.
    # The correct path is `spec_registry[:...].hyper.inner_precomputes`.
    # We perform a string replacement to fix this.
    incorrect_access = "spec_registry[:$(inner_spec_key)].hyper"
    correct_access = "spec_registry[:$(spec.key)].hyper.inner_precomputes"
    inner_updates_code_fixed = replace(inner_updates_code, incorrect_access => correct_access)

    effect_app_regex = Regex("$(eta_target) \\.\\+= .*")
    update_inner_cleaned = replace(
        inner_updates_code_fixed, effect_app_regex => "# (eta update handled by TVC)"
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

    inner_effects_result = get_effects(
        m.model, chain, M, n_samples, outcomes_N, inner_spec, PS, N_total
    )
    
    cov_data_full = if !isnothing(PS) && hasproperty(PS.data, cov_var)
        vcat(M.data[!, cov_var], PS.data[!, cov_var])
    else
        M.data[!, cov_var]
    end
    
    t_idx_full = if !isnothing(PS) && haskey(PS, :t_idx)
        vcat(M.t_idx, PS.t_idx)
    else
        M.t_idx
    end

    structured_effects = Vector{Matrix{Float64}}()
    for k in 1:outcomes_N
        # inner_effects_result.structured[k] is the latent temporal field, size [t_N_full x n_samples]
        temporal_field_k = inner_effects_result.structured[k]
        
        # Map the temporal field to the observation level using the time index
        temporal_effect_at_obs = temporal_field_k[t_idx_full, :]
        
        # Element-wise multiplication of the covariate and the observation-level temporal effect
        final_effect_k = temporal_effect_at_obs .* cov_data_full
        
        push!(structured_effects, final_effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
