"""
    SVC <: ComponentModel

A component model for Spatially Varying Coefficients (SVC), allowing the effect of a
covariate to vary smoothly across space. It acts as an orchestrator, applying an inner
spatial `ComponentModel` to a specified covariate.

# Version
v1.2.0 (2026-08-11)

# Mathematical Summary
An SVC model replaces a fixed regression coefficient \$\\beta\$ with a spatially-indexed
coefficient \$\\beta(s)\$. The contribution to the linear predictor \$\\eta\$ for an
observation at location \$s_i\$ with covariate value \$x_i\$ is given by:

\$\\eta_i = \\dots + \\beta(s_i) x_i\$

The spatially varying coefficient \$\\boldsymbol{\\beta} = (\\beta(s_1), \\dots, \\beta(s_{s_N}))\$
is itself modeled as a latent Gaussian Process or GMRF, governed by the `inner_model`.
For example, if the inner model is an ICAR process, then the prior on \$\\boldsymbol{\\beta}\$ is:
\$\\boldsymbol{\\beta} \\sim \\mathcal{N}(\\mathbf{0}, (\\sigma^2_{\\beta} \\mathbf{Q}_{ICAR})^{-1})\$

# Computational Methods
The `SVC` component does not have its own methods. The computational method is
determined by the `method` parameter of the inner spatial model. For example, to
use a spectral decomposition for the spatially varying coefficient, you would specify:
`... |> random(s_idx, model=icar, method=:spectral)`

# Inputs
- **Required**:
  - A covariate piped (`|>`) into a spatial `random()` module, e.g., `covariate |> random(s_idx, model=icar)`.
  - The inner `random()` call must specify a spatial model.
- **Optional**:
  - Priors for the inner spatial model are passed within the inner `random()` call.

# Outputs (Parameter Names)
- Parameter names are inherited from the inner spatial model and are prefixed with
  `_<key>_inner`. For example, if the main component key is `svc_temp`, the inner
  sigma would be `sigma_svc_temp_inner`.

# Key References
- Gelfand, A. E., Kim, H. J., Sirmans, C. F., & Banerjee, S. (2003). *Spatial
  modeling with spatially varying coefficient processes*. Journal of the American
  Statistical Association, 98(462), 387-396.
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

function get_datastructures!(m_type::Type{<:SVC}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    cov_var = get(params, :covariate, nothing)

    if isnothing(cov_var)
        error("SVC model '$(mod_data[:key])' requires a `covariate` parameter.")
    end

    if cov_var != Symbol("1") && cov_var != :intercept && !hasproperty(M[:data], cov_var)
        error("Covariate ':$cov_var' for SVC model '$(mod_data[:key])' not found in data.")
    end

    inner_model_spec_node = get(params, :spatial_model_spec, nothing)
    if isnothing(inner_model_spec_node)
        error("SVC model '$(mod_data[:key])' is missing the inner `spatial_model_spec`.")
    end

    inner_model_name = get(inner_model_spec_node.args, :model, :icar)
    if !haskey(COMPONENT_TYPE_REGISTRY, inner_model_name)
        error("Inner model ':$inner_model_name' for SVC not found in COMPONENT_TYPE_REGISTRY.")
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

function get_priors(
    m::SVC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
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
        inner_updates_code_fixed, effect_app_regex => "# (eta update handled by SVC)"
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

    structured_effects = Vector{Matrix{Float64}}()
    for k in 1:outcomes_N
        spatial_effect_k = inner_effects_result.structured[k]
        final_effect_k = spatial_effect_k .* cov_data_full
        push!(structured_effects, final_effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
