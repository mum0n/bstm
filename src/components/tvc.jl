"""
    TVC <: ComponentModel

A component model for Temporally Varying Coefficients (TVC), allowing the effect of a
covariate to vary smoothly over time. It acts as an orchestrator, applying an inner
temporal `ComponentModel` to a specified covariate.

# Version
v1.0.0

# Mathematical Summary
A TVC model replaces a fixed regression coefficient \$\\beta\$ with a time-indexed
coefficient \$\\beta(t)\$. The contribution to the linear predictor \$\\eta\$ for an
observation at time \$t_i\$ with covariate value \$x_i\$ is given by:

\$\\eta_i = \\dots + \\beta(t_i) x_i\$

The time-varying coefficient \$\\boldsymbol{\\beta} = (\\beta(t_1), \\dots, \\beta(t_{t_N}))\$
is itself modeled as a latent temporal process, governed by the `inner_model`. For
example, if the inner model is a second-order random walk (RW2), then:

\$\\beta(t) = 2\\beta(t-1) - \\beta(t-2) + \\omega_t, \\quad \\omega_t \\sim \\mathcal{N}(0,
  \\sigma^2_{\\beta})\$

This allows the model to learn how the influence of a covariate changes over time.

# Computational Methods
The `TVC` component does not have its own methods. The computational method is
determined by the `method` parameter of the inner temporal model. For example, to
use a spectral decomposition for the time-varying coefficient, you would specify:
`... |> random(year, model=rw2, method=:spectral)`

# Inputs
- **Required**:
  - A covariate piped (`|>`) into a temporal `random()` module, e.g., `covariate |>
    random(year, model=ar1)`.
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
    inner_model_obj = get(params, :inner_model_obj,
        error("TVC constructor requires an `inner_model_obj` parameter."))
    
    TVC(covariate, inner_model_obj)
end

MODEL_TO_STRUCTURE_MAP[:tvc] = :temporal

function get_precomputes(m::TVC, M::NamedTuple, mod_data::Dict)::NamedTuple
    # Data validation moved from get_datastructures!
    cov_var = m.covariate
    if !hasproperty(M.data, cov_var)
        error("Covariate ':$cov_var' for TVC model '$(mod_data[:key])' not found in data.")
    end

    # The inner model's variables are the main variables of the TVC component
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
    m::TVC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    inner_spec_key = Symbol("$(spec.key)_inner")
    inner_spec = (
        key = inner_spec_key,
        structure = get_component_structure(m.model),
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
    m::TVC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    cov_var = m.covariate

    inner_spec_key = Symbol("$(spec.key)_inner")
    inner_spec = (
        key = inner_spec_key,
        structure = get_component_structure(m.model),
        var = spec.var,
        component_obj = m.model,
        params = spec.params,
        hyper = spec.hyper.inner_precomputes
    )

    # Generate the code for the inner model.
    inner_updates_code = get_updates(m.model, inner_spec, arch, outcome_idx, M)
    inner_p_names = generate_full_variable_names(inner_spec, arch, outcome_idx)
    inner_latent_var = inner_p_names.sre
    
    # Fix the spec_registry path in the generated code.
    incorrect_access = "spec_registry[:$(inner_spec_key)].hyper"
    correct_access = "spec_registry[:$(spec.key)].hyper.inner_precomputes"
    inner_updates_code_fixed = replace(inner_updates_code, incorrect_access => correct_access)

    # Strip the eta update from the inner model's code, returning the inner latent field.
    effect_app_regex = Regex("$(eta_target) (\\.\\+=|=|\\.\\=) .*")
    update_inner_cleaned = replace(
        inner_updates_code_fixed, effect_app_regex => "$(inner_latent_var)"
    )

    application_code = """
        $(inner_latent_var) = begin
            $(update_inner_cleaned)
        end
        $(eta_target) = $(eta_target) .+ M.data[!, :$(cov_var)] .* view($(inner_latent_var),
          M.t_idx)
    """
    
    return """
        # --- Temporally Varying Coefficient (TVC) for: $(cov_var) ---
        $(application_code)
    """
end


function get_effects(
    m::TVC, chain, spec::NamedTuple, M::NamedTuple,
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
        structure = get_component_structure(m.model),
        var = spec.var,
        component_obj = m.model,
        params = spec.params,
        hyper = spec.hyper.inner_precomputes
    )

    # --- Get effects from the inner temporal model ---
    inner_effects_result = get_effects(m.model, chain, inner_spec, M, PS)
    
    # --- Prepare covariate and index data on the CPU ---
    # The data in M.data might be a CuArray, so we need to bring it to the CPU with Array().
    cov_data_train_cpu = Array(M.data[!, cov_var])
    cov_data_full_cpu = if !isnothing(PS) && hasproperty(PS.data, cov_var)
        vcat(cov_data_train_cpu, PS.data[!, cov_var])
    else
        cov_data_train_cpu
    end
    
    t_idx_train_cpu = Array(M.t_idx)
    t_idx_full_cpu = if !isnothing(PS) && haskey(PS, :t_idx)
        vcat(t_idx_train_cpu, get(PS, :t_idx, []))
    else
        t_idx_train_cpu
    end

    structured_effects = Vector{Matrix{Float64}}()
    for k in 1:outcomes_N
        # inner_effects_result.structured[k] is the latent temporal field, size [t_N_full x
        #   n_samples]
        # This is already a CPU matrix.
        temporal_field_k = inner_effects_result.structured[k]
        
        # Map the temporal field to the observation level using the time index (on CPU)
        temporal_effect_at_obs = temporal_field_k[t_idx_full_cpu, :]
        
        # Element-wise multiplication of the covariate and the observation-level temporal
        #   effect (on CPU)
        final_effect_k = temporal_effect_at_obs .* cov_data_full_cpu
        
        push!(structured_effects, final_effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end 
