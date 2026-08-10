"""
    TAR <: ComponentModel

A component for a Threshold Autoregressive (TAR) model, which captures
regime-switching temporal dynamics. The model switches between two different AR(1)
processes based on whether an external `threshold_var` is above or below a
learned threshold.

# Version
v1.0.2 (2026-08-10)

# Mathematical Summary
A TAR model defines a time series where the parameters of the autoregressive
process change depending on the value of a threshold variable \$z_t\$. For a
simple two-regime TAR(1) model, the process \$y_t\$ is defined as:

\$y_t = \\begin{cases} \\rho_1 y_{t-1} + \\epsilon_{1,t} & \\text{if } z_t \\le c \\\\ \\rho_2 y_{t-1} + \\epsilon_{2,t} & \\text{if } z_t > c \\end{cases}\$

where:
- \$\\rho_1, \\rho_2\$ are the autoregressive coefficients for the two regimes.
- \$\\epsilon_{1,t} \\sim \\mathcal{N}(0, \\sigma_1^2)\$ and \$\\epsilon_{2,t} \\sim \\mathcal{N}(0, \\sigma_2^2)\$ are the innovation terms for each regime.
- \$c\$ is the threshold value, which is learned from the data.

# Computational Methods
- `:statespace` (default): An efficient, non-centered parameterization that samples
  unconstrained parameters and transforms them to ensure stationarity (`tanh` for
  `rho`) and positivity (`exp` for `sigma`). Recommended for AD.
- `:statespace_constrained` (didactic): A more direct parameterization that samples
  `rho` and `sigma` from their specified (and appropriately constrained) priors.

# Fields
- `threshold_var::Symbol`: The variable used for thresholding.
- `rho_regimes::Vector{<:UnivariateDistribution}`: Priors for the AR(1) parameter
  `rho` in each regime. Priors should be constrained to `(-1, 1)`.
- `sigma_regimes::Vector{<:UnivariateDistribution}`: Priors for the innovation
  std. dev. `sigma` in each regime. Priors should be positive.
- `method::Symbol`: The computational method, `:statespace` or `:statespace_constrained`.
"""
struct TAR <: ComponentModel
    threshold_var::Symbol
    rho_regimes::Vector{<:UnivariateDistribution}
    sigma_regimes::Vector{<:UnivariateDistribution}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:tar] = TAR

COMPONENT_CONSTRUCTORS[:tar] = (p, params) -> begin
    threshold_var = get(params, :threshold_var, error("TAR model requires a `threshold_var` parameter."))
    
    # User provides priors on the natural scale.
    rho_regimes = get(params, :rho_regimes, [truncated(Normal(0, 0.5), -1, 1), truncated(Normal(0, 0.5), -1, 1)])
    sigma_regimes = get(params, :sigma_regimes, [Exponential(1.0), Exponential(1.0)])
    method = get(params, :method, :statespace)
    
    if !(rho_regimes isa Vector && length(rho_regimes) == 2)
        error("`rho_regimes` for TAR model must be a Vector of two Distributions.")
    end
    if !(sigma_regimes isa Vector && length(sigma_regimes) == 2)
        error("`sigma_regimes` for TAR model must be a Vector of two Distributions.")
    end

    TAR(threshold_var, rho_regimes, sigma_regimes, method)
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

Generates priors for the TAR component, dispatching on the chosen method.
"""
function get_priors(
    m::TAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = String[]
    push!(priors, "$(p_names.thresh_raw) ~ Normal(0.0, 1.0)")
    push!(priors, "$(p_names.innov) ~ MvNormal(zeros(M.t_N), I)")

    if m.method == :statespace
        # Priors on unconstrained "raw" parameters for the AD-friendly method.
        push!(priors, "$(p_names.rho)_1_raw ~ Normal(0, 1.5)")
        push!(priors, "$(p_names.rho)_2_raw ~ Normal(0, 1.5)")
        push!(priors, "$(p_names.sigma)_1_raw ~ Normal(0, 1.0)")
        push!(priors, "$(p_names.sigma)_2_raw ~ Normal(0, 1.0)")
    else # :statespace_constrained
        # Priors directly on the constrained parameters for the didactic method.
        push!(priors, "$(p_names.rho)_1 ~ $(_distribution_to_string(m.rho_regimes[1]))")
        push!(priors, "$(p_names.rho)_2 ~ $(_distribution_to_string(m.rho_regimes[2]))")
        push!(priors, "$(p_names.sigma)_1 ~ $(_distribution_to_string(m.sigma_regimes[1]))")
        push!(priors, "$(p_names.sigma)_2 ~ $(_distribution_to_string(m.sigma_regimes[2]))")
    end

    return join(priors, "\n    ")
end


"""
    get_updates(m::TAR, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to construct the TAR effect, dispatching on the chosen method.
"""
function get_updates(
    m::TAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    local param_definitions
    if m.method == :statespace
        param_definitions = """
            local rho1 = tanh($(p_names.rho)_1_raw)
            local rho2 = tanh($(p_names.rho)_2_raw)
            local sigma1 = exp($(p_names.sigma)_1_raw)
            local sigma2 = exp($(p_names.sigma)_2_raw)
        """
    else # :statespace_constrained
        param_definitions = """
            local rho1 = $(p_names.rho)_1
            local rho2 = $(p_names.rho)_2
            local sigma1 = $(p_names.sigma)_1
            local sigma2 = $(p_names.sigma)_2
        """
    end

    return """
        # --- TAR Component: $(spec.key) ($(m.method)) ---
        let
            $(param_definitions)
            local threshold_level = mean(spec.hyper.threshold_data) + $(p_names.thresh_raw)
            local innovations = $(p_names.innov)
            
            local latent_field = Vector{eltype(innovations)}(undef, M.t_N)
            
            for t in 1:M.t_N
                local regime_indicator = spec.hyper.threshold_data[t] > threshold_level
                local curr_rho = regime_indicator ? rho2 : rho1
                local curr_sigma = regime_indicator ? sigma2 : sigma1

                if t == 1
                    latent_field[t] = (innovations[t] * curr_sigma) / sqrt(1.0 - curr_rho^2 + M.noise)
                else
                    latent_field[t] = curr_rho * latent_field[t-1] + innovations[t] * curr_sigma
                end
            end
            $(p_names.latent) = latent_field
            $(eta_target) .+= view($(p_names.latent), M.t_idx)
        end
    """
end

"""
    get_effects(m::TAR, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_total)::NamedTuple

Reconstructs the TAR component's effect from posterior samples, dispatching on
the method used during sampling.
"""
function get_effects(
    m::TAR, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    t_N_train = M.t_N
    t_idx_full = isnothing(PS) ? M.t_idx : vcat(M.t_idx, PS.t_idx)
    t_N_full = maximum(t_idx_full)

    threshold_data_train = spec.hyper.threshold_data
    threshold_data_full = if !isnothing(PS) && hasproperty(PS.data, m.threshold_var)
        pred_threshold_data = PS.data[!, m.threshold_var]
        vcat(threshold_data_train, pred_threshold_data[1:min(length(pred_threshold_data), t_N_full - t_N_train)])
    else
        threshold_data_train
    end
    if length(threshold_data_full) < t_N_full
        threshold_data_full = vcat(threshold_data_full, fill(mean(threshold_data_train), t_N_full - length(threshold_data_full)))
    end

    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        
        thresh_raw_samples = get_params_vector(chain, string(v.thresh_raw), 1)
        innov_samples = get_params_vector(chain, string(v.innov), t_N_train)
        
        local rho1_samples, rho2_samples, sigma1_samples, sigma2_samples
        if m.method == :statespace
            rho1_samples = tanh.(get_params_vector(chain, string(v.rho, "_1_raw"), 1))
            rho2_samples = tanh.(get_params_vector(chain, string(v.rho, "_2_raw"), 1))
            sigma1_samples = exp.(get_params_vector(chain, string(v.sigma, "_1_raw"), 1))
            sigma2_samples = exp.(get_params_vector(chain, string(v.sigma, "_2_raw"), 1))
        else # :statespace_constrained
            rho1_samples = get_params_vector(chain, string(v.rho, "_1"), 1)
            rho2_samples = get_params_vector(chain, string(v.rho, "_2"), 1)
            sigma1_samples = get_params_vector(chain, string(v.sigma, "_1"), 1)
            sigma2_samples = get_params_vector(chain, string(v.sigma, "_2"), 1)
        end

        effect_k = zeros(Float64, N_total, n_samples)
        mean_thresh_data = mean(threshold_data_train)
        noise = M.noise

        for s in 1:n_samples
            threshold_level = mean_thresh_data + thresh_raw_samples[s, 1]
            innov_full = vcat(innov_samples[s, :], randn(t_N_full - t_N_train))
            latent_field_s = zeros(Float64, t_N_full)

            for t in 1:t_N_full
                regime_indicator = threshold_data_full[t] > threshold_level
                curr_rho = regime_indicator ? rho2_samples[s, 1] : rho1_samples[s, 1]
                curr_sigma = regime_indicator ? sigma2_samples[s, 1] : sigma1_samples[s, 1]

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
