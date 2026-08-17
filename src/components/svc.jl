"""
    SVC <: ComponentModel

A component model for Spatially Varying Coefficients (SVC), allowing the effect of a
covariate to vary smoothly across space. It acts as an orchestrator, applying an inner
spatial `ComponentModel` to a specified covariate.

# Version
v1.2.1 (2026-08-14)

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

function get_precomputes(m::SVC, M::NamedTuple, mod_data::Dict)::NamedTuple
    # Data validation moved from get_datastructures!
    cov_var = m.covariate
    if cov_var != Symbol("1") && cov_var != :intercept && !hasproperty(M.data, cov_var)
        error("Covariate ':$cov_var' for SVC model '$(mod_data[:key])' not found in data.")
    end

    # The inner model's variables are the main variables of the SVC component
    inner_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_inner"),
        :variables => mod_data[:variables],
        :params => mod_data[:params]
    )
    
    inner_precomputes = get_precomputes(m.model, M, inner_mod_data)
    
    return (
        inner_precomputes=inner_precomputes,
        covariate_name=m.covariate
    )
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
        hyper = spec.hyper.inner_precomputes
    )
    
    # Generate the code for the inner model's priors
    inner_priors_code = get_priors(m.model, inner_spec, arch, outcome_idx, M)
    
    # The generated code will try to access `spec_registry[:..._inner].hyper`.
    # The correct path is `spec_registry[:...].hyper.inner_precomputes`.
    # We perform a string replacement to fix this.
    incorrect_access = "spec_registry[:$(inner_spec_key)].hyper"
    correct_access = "spec_registry[:$(spec.key)].hyper.inner_precomputes"
    
    return replace(inner_priors_code, incorrect_access => correct_access)
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
        hyper = spec.hyper.inner_precomputes
    )

    # Generate the code for the inner model.
    inner_updates_code = get_updates(m.model, inner_spec, arch, outcome_idx, M)
    inner_p_names = generate_full_variable_names(inner_spec, arch, outcome_idx)
    inner_latent_var = inner_p_names.latent
    
    # Fix the spec_registry path in the generated code.
    incorrect_access = "spec_registry[:$(inner_spec_key)].hyper"
    correct_access = "spec_registry[:$(spec.key)].hyper.inner_precomputes"
    inner_updates_code_fixed = replace(inner_updates_code, incorrect_access => correct_access)

    # Strip the eta update from the inner model's code, as the SVC wrapper handles it.
    effect_app_regex = Regex("$(eta_target) \\.\\+= .*")
    update_inner_cleaned = replace(
        inner_updates_code_fixed, effect_app_regex => "# (eta update handled by SVC wrapper)"
    )

    is_intercept = (cov_var == Symbol("1") || cov_var == :intercept)
    
    covariate_access = is_intercept ? "1.0" : "M.data[!, :$(cov_var)]"
    
    application_code = "$(eta_target) .+= $(covariate_access) .* view($(inner_latent_var), M.s_idx)"
    
    return """
        # --- Spatially Varying Coefficient (SVC) for: $(cov_var) ---
        # 1. Generate the latent spatial field for the coefficient.
        $(update_inner_cleaned)

        # 2. Apply the spatially varying coefficient to the linear predictor.
        $(application_code)
    """
end


function get_effects(
    m::SVC, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    outcomes_N = M.outcomes_N
    cov_var = m.covariate
    
    # --- Construct inner specification ---
    # This creates the specification required to call the inner model's methods.
    inner_spec_key = Symbol("$(spec.key)_inner")
    inner_spec = (
        key = inner_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.model)],
        var = spec.var,
        component_obj = m.model,
        params = spec.params,
        hyper = spec.hyper.inner_precomputes
    )

    # --- Get effects from the inner spatial model ---
    # This call will return effects on the CPU as per the ComponentModel interface contract.
    # The heavy computation (and GPU usage) happens inside this call.
    inner_effects_result = get_effects(m.model, chain, inner_spec, M, PS)
    
    # --- Prepare covariate data on the CPU ---
    is_intercept = (cov_var == Symbol("1") || cov_var == :intercept)
    
    # This data is constructed on the CPU.
    cov_data_full_cpu = if is_intercept
        ones(Float64, size(inner_effects_result.structured[1], 1))
    else
        # The data in M.data might be a CuArray, so we need to bring it to the CPU with Array().
        train_data = Array(M.data[!, cov_var])
        if !isnothing(PS)
            if hasproperty(PS.data, cov_var)
                vcat(train_data, PS.data[!, cov_var])
            else
                error("SVC prediction requires covariate '$(cov_var)' in the prediction dataset.")
            end
        else
            train_data
        end
    end

    # --- Apply covariate to the inner spatial effect on the CPU ---
    structured_effects = Vector{Matrix{Float64}}()
    for k in 1:outcomes_N
        # inner_effects_result.structured[k] is already a CPU matrix.
        spatial_effect_k = inner_effects_result.structured[k]
        
        if size(spatial_effect_k, 1) != length(cov_data_full_cpu)
             @warn "SVC effect reconstruction: dimension mismatch. Expected $(length(cov_data_full_cpu)) rows, but inner effect has $(size(spatial_effect_k, 1)). This may indicate an issue with prediction data handling."
        end

        # Perform element-wise multiplication on the CPU. This is computationally cheap.
        final_effect_k = spatial_effect_k .* cov_data_full_cpu
        push!(structured_effects, final_effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end

