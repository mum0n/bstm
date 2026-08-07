# This file contains the proposed new and updated functions for the bstm refactoring.
 
  # This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    TAR <: ComponentModel

A component model for Threshold Autoregressive (TAR) effects, capturing regime-switching
temporal dynamics based on a threshold variable.

# Fields
- `threshold_var::Symbol`: The variable used for thresholding.
- `log_rho_regimes::Vector{<:UnivariateDistribution}`: Priors for the AR(1) parameter `rho` in each regime,
  assumed to be on an unbounded scale (e.g., `Normal`). A `tanh` transformation is applied internally.
- `log_sigma_regimes::Vector{<:UnivariateDistribution}`: Priors for the innovation standard deviation `sigma`
  in each regime, assumed to be on an unbounded scale (e.g., `Normal`). An `exp` transformation is applied internally.
"""
struct TAR <: ComponentModel
    threshold_var::Symbol
    log_rho_regimes::Vector{<:UnivariateDistribution}
    log_sigma_regimes::Vector{<:UnivariateDistribution}
end

# Add to the central component constructor registry.
# This logic validates the parameters and constructs the TAR object.
COMPONENT_CONSTRUCTORS[:tar] = (p, params) -> begin
    threshold_var = get(params, :threshold_var, error("TAR model requires a `threshold_var` parameter."))
    
    # Default to Normal(0, 1.5) for log_rho_regimes (unbounded for tanh)
    log_rho_regimes = get(params, :log_rho_regimes, [Normal(0, 1.5), Normal(0, 1.5)])
    # Default to Normal(0, 1.0) for log_sigma_regimes (unbounded for exp)
    log_sigma_regimes = get(params, :log_sigma_regimes, [Normal(0, 1.0), Normal(0, 1.0)])
    
    if !(log_rho_regimes isa Vector && length(log_rho_regimes) == 2)
        error("`log_rho_regimes` for TAR model must be a Vector of two Distributions.")
    end
    if !(log_sigma_regimes isa Vector && length(log_sigma_regimes) == 2)
        error("`log_sigma_regimes` for TAR model must be a Vector of two Distributions.")
    end

    TAR(threshold_var, log_rho_regimes, log_sigma_regimes)
end

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[TAR] = :temporal

"""
    get_datastructures!(m_type::Type{<:TAR}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `TAR` component.
It ensures that the `threshold_var` is present in the data and that the temporal
context (`t_idx`, `t_N`) is set up in the main model configuration `M`.
"""
function get_datastructures!(m_type::Type{<:TAR}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    threshold_var = get(params, :threshold_var, nothing)

    if isnothing(threshold_var)
        error("TAR model requires a `threshold_var` parameter.")
    end

    if !hasproperty(M[:data], threshold_var)
        error("Threshold variable ':$threshold_var' for TAR model not found in data.")
    end

    # Ensure temporal context is set up
    if !haskey(M, :t_idx) || M[:t_N] == 0
        # Assuming the first variable in `mod_data[:variables]` is the time index
        # If no variables are provided, it might need to infer from the data itself or throw an error.
        if isempty(mod_data[:variables])
            error("TAR model requires a temporal index variable, e.g., `random(year, model=:tar, threshold_var=...)`.")
        end
        time_var = Symbol(mod_data[:variables][1])
        if !hasproperty(M[:data], time_var)
            error("Temporal index variable ':$time_var' for TAR model not found in data.")
        end
        tu = assign_time_units(M[:data][!, time_var])
        M[:t_idx] = tu.idx
        M[:t_N] = tu.N_cat
        @info "Inferred temporal context for TAR model: t_N = $(M[:t_N])."
    end

    return true
end

"""
    get_precomputes(m::TAR, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes and stores the threshold data, indexed by unique temporal units.
"""
function get_precomputes(m::TAR, M::NamedTuple, mod_data::Dict)::NamedTuple
    threshold_var = m.threshold_var
    
    # Extract threshold data from the original data frame, indexed by M.t_idx
    # This ensures we have one value per unique time point (1 to M.t_N)
    # If multiple observations share a time point, we'll take the mean of the threshold variable for that time point.
    
    threshold_data_full = M.data[!, threshold_var]
    
    threshold_data_per_t = zeros(eltype(threshold_data_full), M.t_N)
    counts_per_t = zeros(Int, M.t_N)

    for i in 1:M.y_N
        t_val = M.t_idx[i]
        threshold_data_per_t[t_val] += threshold_data_full[i]
        counts_per_t[t_val] += 1
    end

    threshold_data_per_t ./= max.(1, counts_per_t) # Avoid division by zero

    return (threshold_data=threshold_data_per_t,)
end

"""
    get_priors(m::TAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code for the priors on the TAR component's parameters.
This includes priors for the log-transformed `rho` and `sigma` for each regime,
the raw threshold parameter, and the standard normal innovations.
"""
function get_priors(m::TAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    log_rho1_name = Symbol("$(p_names.rho)_1_log")
    log_rho2_name = Symbol("$(p_names.rho)_2_log")
    log_sigma1_name = Symbol("$(p_names.sigma)_1_log")
    log_sigma2_name = Symbol("$(p_names.sigma)_2_log")

    priors = String[]
    # Priors for log-transformed rho (unbounded)
    push!(priors, "$(log_rho1_name) ~ NamedDist($(_distribution_to_string(m.log_rho_regimes[1])), :$(log_rho1_name))")
    push!(priors, "$(log_rho2_name) ~ NamedDist($(_distribution_to_string(m.log_rho_regimes[2])), :$(log_rho2_name))")
    
    # Priors for log-transformed sigma (unbounded for log-sigma)
    push!(priors, "$(log_sigma1_name) ~ NamedDist($(_distribution_to_string(m.log_sigma_regimes[1])), :$(log_sigma1_name))")
    push!(priors, "$(log_sigma2_name) ~ NamedDist($(_distribution_to_string(m.log_sigma_regimes[2])), :$(log_sigma2_name))")
    
    push!(priors, "$(p_names.thresh_raw) ~ NamedDist(Normal(T(0), T(1)), :$(p_names.thresh_raw))")
    push!(priors, "$(p_names.innov) ~ NamedDist(MvNormal(zeros(T, M.t_N), I), :$(p_names.innov))")

    return join(priors, "\n    ")
end

"""
    get_updates(m::TAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code to construct the TAR component's effect by applying
regime-switching AR(1) logic and adds the result to the linear predictor `eta`.
"""
function get_updates(m::TAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    log_rho1_name = Symbol("$(p_names.rho)_1_log")
    log_rho2_name = Symbol("$(p_names.rho)_2_log")
    log_sigma1_name = Symbol("$(p_names.sigma)_1_log")
    log_sigma2_name = Symbol("$(p_names.sigma)_2_log")

    return """
        # --- TAR Component: $(spec.key) ---
        local threshold_level = T(mean(spec_registry[:$(spec.key)].precomputes.threshold_data)) + $(p_names.thresh_raw)
        local innovations = $(p_names.innov)
        
        local $(p_names.latent) = zeros(T, M.t_N)
        
        for t in 1:M.t_N
            local regime_indicator = spec_registry[:$(spec.key)].precomputes.threshold_data[t] > threshold_level
            
            # Apply tanh transformation to ensure rho is within (-1, 1) for stationarity
            local curr_rho = tanh(regime_indicator ? $(log_rho2_name) : $(log_rho1_name))
            
            # Ensure sigma is positive by exponentiating
            local curr_sigma = exp(regime_indicator ? $(log_sigma2_name) : $(log_sigma1_name))

            if t == 1
                # For the first time point, assume it's drawn from the stationary distribution
                $(p_names.latent)[t] = (innovations[t] * curr_sigma) / sqrt(T(1.0) - curr_rho^2 + T(M.noise))
            else
                $(p_names.latent)[t] = curr_rho * $(p_names.latent)[t-1] + innovations[t] * curr_sigma
            end
        end
        eta .+= $(p_names.latent)[M.t_idx]
    """
end

"""
    get_effects(m::TAR, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the TAR component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::TAR, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    log_rho1_name = Symbol("$(p_names.rho)_1_log")
    log_rho2_name = Symbol("$(p_names.rho)_2_log")
    log_sigma1_name = Symbol("$(p_names.sigma)_1_log")
    log_sigma2_name = Symbol("$(p_names.sigma)_2_log")

    thresh_raw_samples = get(chain, p_names.thresh_raw)
    innov_samples = get(chain, p_names.innov)
    log_rho1_samples = get(chain, log_rho1_name)
    log_rho2_samples = get(chain, log_rho2_name)
    log_sigma1_samples = get(chain, log_sigma1_name)
    log_sigma2_samples = get(chain, log_sigma2_name)

    threshold_data_per_t = spec.precomputes.threshold_data
    t_N = M.t_N
    noise = M.noise

    idx_to_use = isnothing(PS) ? M.t_idx : PS.t_idx
    
    reconstructed_effects = zeros(n_samples, t_N)

    for s in 1:n_samples
        current_thresh_raw = thresh_raw_samples[s]
        current_innov = innov_samples[s, :]
        current_log_rho1 = log_rho1_samples[s]
        current_log_rho2 = log_rho2_samples[s]
        current_log_sigma1 = log_sigma1_samples[s]
        current_log_sigma2 = log_sigma2_samples[s]

        threshold_level = mean(threshold_data_per_t) + current_thresh_raw
        
        latent_field_s = zeros(eltype(current_log_rho1), t_N)

        for t in 1:t_N
            regime_indicator = threshold_data_per_t[t] > threshold_level
            
            curr_rho = tanh(regime_indicator ? current_log_rho2 : current_log_rho1)
            curr_sigma = exp(regime_indicator ? current_log_sigma2 : current_log_sigma1)

            if t == 1
                latent_field_s[t] = (current_innov[t] * curr_sigma) / sqrt(1.0 - curr_rho^2 + noise)
            else
                latent_field_s[t] = curr_rho * latent_field_s[t-1] + current_innov[t] * curr_sigma
            end
        end
        reconstructed_effects[s, :] = latent_field_s
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    indexed_mean = mean_effect[idx_to_use]
    indexed_lower = lower_ci[idx_to_use]
    indexed_upper = upper_ci[idx_to_use]

    return (structured=(mean=indexed_mean, lower=indexed_lower, upper=indexed_upper),)
end
