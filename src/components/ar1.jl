
struct AR1 <: ComponentModel
    rho::UnivariateDistribution
    sigma::UnivariateDistribution
end

COMPONENT_CONSTRUCTORS[:ar1] = (p, params) -> AR1(p.rho, p.sigma)
 

# helper to map to classes of methods (data structures), :any mean it can be used in many approaches
MODEL_TO_STRUCTURE_MAP[:ar1] = :temporal


"""
    get_datastructures!(m_type::Type{AR1}, M::Dict, mod_data::Dict)::Bool

v1.9.1 (2026-08-07) - Data-dependent setup for the AR1 component.
This method establishes the temporal context for the model. It identifies the time
variable from the `random()` module call, uses `assign_time_units` to create
discrete time indices (`t_idx`) and determines the total number of time steps (`t_N`).
This information is added to the main model configuration `M`.
"""
function get_datastructures!(m_type::Type{AR1}, M::Dict, mod_data::Dict)::Bool
    data = M[:data]
    params = mod_data[:params]
    variables = mod_data[:variables]

    if isempty(variables)
        error("The AR1 model requires a time index variable, e.g., `random(year, model=:ar1)`.")
    end

    time_var_sym = Symbol(variables[1])
    if !hasproperty(data, time_var_sym)
        error("Time index variable ':$time_var_sym' for AR1 model not found in data.")
    end

    time_opts = Dict(:time_method => get(params, :time_method, "regular"))
    tu_meta = assign_time_units(data[!, time_var_sym]; time_opts...)
    
    M[:t_idx] = tu_meta.idx
    M[:t_N] = tu_meta.N_cat
    M[:t_idx_var] = time_var_sym
    
    return true
end

"""
    get_precomputes(m::AR1, M::NamedTuple, mod_data::Dict)::NamedTuple

v1.9.1 (2026-08-07) - Pre-computes the structure matrix for the AR1 component.
For an AR1 model, the precision matrix template defines the first-order dependencies.
This function calls the central `build_structure_template` utility to generate this
matrix and its spectral decomposition.
"""
function get_precomputes(m::AR1, M::NamedTuple, mod_data::Dict)::NamedTuple
    t_N = get(M, :t_N, 0)
    if t_N == 0
        @warn "Could not determine number of time steps for AR1 component '$(mod_data[:key])'. The component will have no effect."
    end
    template = build_structure_template(:ar1, t_N)
    return (Q_template=template.matrix, U=template.U, L=template.L, scaling_factor=template.scaling_factor)
end

"""
    get_priors(m::AR1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

v1.9.1 (2026-08-07) - Generates the Turing code for the AR1 component's priors.
This function creates the code strings for sampling the `sigma` hyperparameter, the
unconstrained `rho_raw` parameter, and the standard normal innovations `innov`.
The `rho` parameter is constrained to `(-1, 1)` via a `tanh` transform in the `get_updates`
method, which is a robust and AD-friendly approach.
"""
function get_priors(m::AR1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = size(spec.Q_template, 1)
    is_multivariate = (arch == "multivariate")
    is_shared = get(spec.params, :shared, false)
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ $(_distribution_to_string(m.sigma))")
        # Define prior on an unconstrained variable for rho.
        # The tanh transformation will be applied in the updates step.
        push!(priors_acc, "$(v.rho)_raw ~ Normal(0, 1.5)")
    end
    push!(priors_acc, "$(v.innov) ~ MvNormal(zeros(T, $(n_latent)), I)")
    return join(priors_acc, "\n    ")
end

"""
    get_updates(m::AR1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

v1.9.1 (2026-08-07) - Generates the Turing code for the AR1 component's update logic.
This function generates the code to:
1.  Transform the unconstrained `rho_raw` parameter to the `(-1, 1)` range using `tanh`.
2.  Call the `ar1_statespace` helper function to compute the latent AR1 field.
3.  Add the resulting effect to the linear predictor `eta`.
"""
function get_updates(m::AR1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = size(spec.Q_template, 1)
    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx"

    return """
    # --- AR1 Component: $(spec.key) ---
    # Transform the unconstrained parameter to the (-1, 1) range for rho
    local $(v.rho) = tanh($(v.rho)_raw)
    
    # Evolve the AR1 state-space
    $(v.latent) = ar1_statespace($(v.rho), $(v.sigma), $(v.innov), $(n_latent), noise)
    
    # Add the effect to the linear predictor
    $(eta_target) .+= view($(v.latent), M.$(index_var))
    """
end

"""
    get_effects(m::AR1, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

v1.9.1 (2026-08-07) - Reconstructs the AR1 component's effect from posterior samples.
This function extracts the posterior samples for `rho_raw`, `sigma`, and `innov` from
the MCMC chain. It then reconstructs the full posterior distribution of the latent AR1
time series by applying the `tanh` transform to `rho_raw` and re-running the
`ar1_statespace` evolution for each posterior sample.
"""
function get_effects(m::AR1, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    structured_effects = []
    is_multivariate = outcomes_N > 1
    is_shared = get(spec.params, :shared, false)
    n_latent = size(spec.Q_template, 1)

    for k in 1:outcomes_N
        outcome_idx = is_multivariate ? k : nothing
        v = generate_full_variable_names(spec, M.model_arch, outcome_idx)
        
        sigma_var_name = (is_multivariate && is_shared) ? string(generate_full_variable_names(spec, M.model_arch, nothing).sigma) : string(v.sigma)
        rho_raw_var_name = (is_multivariate && is_shared) ? string(generate_full_variable_names(spec, M.model_arch, nothing).rho, "_raw") : string(v.rho, "_raw")

        sigma_samples = get_params_vector(chain, sigma_var_name, 1)
        rho_raw_samples = get_params_vector(chain, rho_raw_var_name, 1)
        innov_samples = get_params_vector(chain, string(v.innov), n_latent)
        
        T = eltype(chain.value)
        effect_k = Matrix{T}(undef, n_latent, n_samples)

        for s in 1:n_samples
            rho_s = tanh(rho_raw_samples[s, 1])
            sigma_s = sigma_samples[s, 1]
            innov_s = innov_samples[s, :]
            
            effect_k[:, s] = ar1_statespace(rho_s, sigma_s, innov_s, n_latent, M.noise)
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end