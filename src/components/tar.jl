"""
    TAR <: ComponentModel

A component for a Threshold Autoregressive (TAR) model, which captures
regime-switching temporal dynamics. The model switches between two different AR(1)
processes based on whether an external `threshold_var` is above or below a
learned threshold.

# Version
v1.0.1 (2026-08-09)

# Mathematical Summary
A TAR model defines a time series where the parameters of the autoregressive
process change depending on the value of a threshold variable \$z_t\$. For a
simple two-regime TAR(1) model, the process \$y_t\$ is defined as:

\$y_t = \\begin{cases} \\rho_1 y_{t-1} + \\epsilon_{1,t} & \\text{if } z_t \\le c \\\\ \\rho_2 y_{t-1} + \\epsilon_{2,t} & \\text{if } z_t > c \\end{cases}\$

where:
- \$\\rho_1, \\rho_2\$ are the autoregressive coefficients for the two regimes.
- \$\\epsilon_{1,t} \\sim \\mathcal{N}(0, \\sigma_1^2)\$ and \$\\epsilon_{2,t} \\sim \\mathcal{N}(0, \\sigma_2^2)\$ are the innovation terms for each regime.
- \$c\$ is the threshold value, which is learned from the data.

This implementation uses a non-centered parameterization. The `rho` parameters are
sampled on an unbounded scale and transformed with `tanh` to ensure stationarity.
The `sigma` parameters are sampled on the log scale and transformed with `exp` to
ensure positivity.

# Best Use Case
Modeling time series that exhibit different dynamic behaviors under different
conditions, such as economic data that behaves differently during expansions vs.
recessions, or ecological populations that have different dynamics in high vs.
low resource environments.

# Key References
- Tong, H. (1978). On a threshold model. In *Pattern Recognition and Signal
  Processing* (pp. 575-586). Sijthoff & Noordhoff.
- Wikipedia: Threshold autoregressive model.

# Fields
- `threshold_var::Symbol`: The variable used for thresholding.
- `log_rho_regimes::Vector{<:UnivariateDistribution}`: Priors for the AR(1)
  parameter `rho` in each regime, on an unbounded scale (for `tanh` transform).
- `log_sigma_regimes::Vector{<:UnivariateDistribution}`: Priors for the innovation
  std. dev. `sigma` in each regime, on the log scale (for `exp` transform).
"""
struct TAR <: ComponentModel
    threshold_var::Symbol
    log_rho_regimes::Vector{<:UnivariateDistribution}
    log_sigma_regimes::Vector{<:UnivariateDistribution}
end

COMPONENT_TYPE_REGISTRY[:tar] = TAR

COMPONENT_CONSTRUCTORS[:tar] = (p, params) -> begin
    threshold_var = get(params, :threshold_var, error("TAR model requires a `threshold_var` parameter."))
    
    log_rho_regimes = get(params, :log_rho_regimes, [Normal(0, 1.5), Normal(0, 1.5)])
    log_sigma_regimes = get(params, :log_sigma_regimes, [Normal(0, 1.0), Normal(0, 1.0)])
    
    if !(log_rho_regimes isa Vector && length(log_rho_regimes) == 2)
        error("`log_rho_regimes` for TAR model must be a Vector of two Distributions.")
    end
    if !(log_sigma_regimes isa Vector && length(log_sigma_regimes) == 2)
        error("`log_sigma_regimes` for TAR model must be a Vector of two Distributions.")
    end

    TAR(threshold_var, log_rho_regimes, log_sigma_regimes)
end

MODEL_TO_STRUCTURE_MAP[:tar] = :temporal

"""
    get_datastructures!(m_type::Type{<:TAR}, M::Dict, mod_data::Dict)::Bool

Ensures the `threshold_var` is present in the data and that the temporal context
(`t_idx`, `t_N`) is set up in the main model configuration `M`.
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

    if !haskey(M, :t_idx) || !haskey(M, :t_N) || M[:t_N] == 0
        if isempty(mod_data[:variables])
            error("TAR model requires a temporal index variable, e.g., `random(year, model=:tar, ...)`.")
        end
        time_var = Symbol(mod_data[:variables][1])
        if !hasproperty(M[:data], time_var)
            error("Temporal index variable ':$time_var' for TAR model not found in data.")
        end
        tu = assign_time_units(M[:data][!, time_var])
        M[:t_idx] = tu.idx
        M[:t_N] = tu.N_cat
    end

    return true
end

"""
    get_precomputes(m::TAR, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes and stores the threshold data, averaged for each unique temporal unit.
"""
function get_precomputes(m::TAR, M::NamedTuple, mod_data::Dict)::NamedTuple
    threshold_var = m.threshold_var
    threshold_data_full = M.data[!, threshold_var]
    
    threshold_data_per_t = zeros(eltype(threshold_data_full), M.t_N)
    counts_per_t = zeros(Int, M.t_N)

    for i in 1:M.y_N
        t_val = M.t_idx[i]
        threshold_data_per_t[t_val] += threshold_data_full[i]
        counts_per_t[t_val] += 1
    end

    threshold_data_per_t ./= max.(1, counts_per_t)

    return (threshold_data=threshold_data_per_t,)
end

"""
    get_priors(m::TAR, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for the log-transformed `rho` and `sigma` for each regime,
the raw threshold parameter, and the standard normal innovations.
"""
function get_priors(m::TAR, spec::NamedTuple, arch::String, outcome_idx, M)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    log_rho1_name = Symbol("$(p_names.rho)_1_log")
    log_rho2_name = Symbol("$(p_names.rho)_2_log")
    log_sigma1_name = Symbol("$(p_names.sigma)_1_log")
    log_sigma2_name = Symbol("$(p_names.sigma)_2_log")

    priors = String[]
    push!(priors, "$(log_rho1_name) ~ $(_distribution_to_string(m.log_rho_regimes[1]))")
    push!(priors, "$(log_rho2_name) ~ $(_distribution_to_string(m.log_rho_regimes[2]))")
    push!(priors, "$(log_sigma1_name) ~ $(_distribution_to_string(m.log_sigma_regimes[1]))")
    push!(priors, "$(log_sigma2_name) ~ $(_distribution_to_string(m.log_sigma_regimes[2]))")
    push!(priors, "$(p_names.thresh_raw) ~ Normal(0.0, 1.0)")
    push!(priors, "$(p_names.innov) ~ MvNormal(zeros(M.t_N), I)")

    return join(priors, "\n    ")
end

"""
    get_updates(m::TAR, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to construct the TAR effect by applying regime-switching AR(1)
logic and adds the result to the linear predictor `eta`.
"""
function get_updates(m::TAR, spec::NamedTuple, arch::String, outcome_idx, M)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    log_rho1_name = Symbol("$(p_names.rho)_1_log")
    log_rho2_name = Symbol("$(p_names.rho)_2_log")
    log_sigma1_name = Symbol("$(p_names.sigma)_1_log")
    log_sigma2_name = Symbol("$(p_names.sigma)_2_log")

    return """
        # --- TAR Component: $(spec.key) ---
        local threshold_level = mean(spec.hyper.threshold_data) + $(p_names.thresh_raw)
        local innovations = $(p_names.innov)
        
        local latent_field = Vector{eltype(innovations)}(undef, M.t_N)
        
        for t in 1:M.t_N
            local regime_indicator = spec.hyper.threshold_data[t] > threshold_level
            local curr_rho = tanh(regime_indicator ? $(log_rho2_name) : $(log_rho1_name))
            local curr_sigma = exp(regime_indicator ? $(log_sigma2_name) : $(log_sigma1_name))

            if t == 1
                latent_field[t] = (innovations[t] * curr_sigma) / sqrt(1.0 - curr_rho^2 + M.noise)
            else
                latent_field[t] = curr_rho * latent_field[t-1] + innovations[t] * curr_sigma
            end
        end
        $(p_names.latent) = latent_field
        $(eta_target) .+= view($(p_names.latent), M.t_idx)
    """
end

"""
    get_effects(m::TAR, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_total)::NamedTuple

Reconstructs the TAR component's effect from posterior samples.
"""
function get_effects(m::TAR, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_total)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    t_N_train = M.t_N
    t_idx_full = isnothing(PS) ? M.t_idx : vcat(M.t_idx, PS.t_idx)
    t_N_full = maximum(t_idx_full)

    threshold_data_train = spec.hyper.threshold_data
    threshold_data_full = if !isnothing(PS) && hasproperty(PS.data, m.threshold_var)
        # This assumes prediction data has the same temporal resolution as training data.
        # A more robust implementation might require interpolation.
        pred_threshold_data = PS.data[!, m.threshold_var]
        # This simple vcat assumes PS.data is ordered by time.
        vcat(threshold_data_train, pred_threshold_data[1:(t_N_full - t_N_train)])
    else
        threshold_data_train
    end

    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        log_rho1_name = Symbol("$(v.rho)_1_log")
        log_rho2_name = Symbol("$(v.rho)_2_log")
        log_sigma1_name = Symbol("$(v.sigma)_1_log")
        log_sigma2_name = Symbol("$(v.sigma)_2_log")

        thresh_raw_samples = get_params_vector(chain, string(v.thresh_raw), 1)
        innov_samples = get_params_vector(chain, string(v.innov), t_N_train)
        log_rho1_samples = get_params_vector(chain, string(log_rho1_name), 1)
        log_rho2_samples = get_params_vector(chain, string(log_rho2_name), 1)
        log_sigma1_samples = get_params_vector(chain, string(log_sigma1_name), 1)
        log_sigma2_samples = get_params_vector(chain, string(log_sigma2_name), 1)

        effect_k = zeros(Float64, N_total, n_samples)
        mean_thresh_data = mean(threshold_data_train)
        noise = M.noise

        for s in 1:n_samples
            threshold_level = mean_thresh_data + thresh_raw_samples[s, 1]
            
            # Combine training innovations with new random innovations for prediction period
            innov_full = vcat(innov_samples[s, :], randn(t_N_full - t_N_train))
            
            latent_field_s = zeros(Float64, t_N_full)

            for t in 1:t_N_full
                regime_indicator = threshold_data_full[t] > threshold_level
                
                curr_rho = tanh(regime_indicator ? log_rho2_samples[s, 1] : log_rho1_samples[s, 1])
                curr_sigma = exp(regime_indicator ? log_sigma2_samples[s, 1] : log_sigma1_samples[s, 1])

                if t == 1
                    latent_field_s[t] = (innov_full[t] * curr_sigma) / sqrt(1.0 - curr_rho^2 + noise)
                else
                    latent_field_s[t] = curr_rho * latent_field_s[t-1] + innov_full[t] * curr_sigma
                end
            end
            effect_k[:, s] = view(latent_field_s, t_idx_full)
        end
        push!(structured_effects, effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
