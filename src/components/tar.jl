"""
    TAR <: ComponentModel

A component for a Threshold Autoregressive (TAR) model, which captures
regime-switching temporal dynamics. The model switches between two different AR(1)
processes based on whether an external `threshold_var` is above or below a
learned threshold.

# Version
v1.1.1 (2026-08-14)

# Mathematical Summary
A TAR model defines a time series where the parameters of the autoregressive
process change depending on the value of a threshold variable \$z_t\$. For a
simple two-regime TAR(1) model, the process \$\\phi_t\$ is defined as:

\$\\phi_t = \\begin{cases} \\rho_1 \\phi_{t-1} + \\epsilon_{1,t} & \\text{if } z_t \\le c \\\\ \\rho_2 \\phi_{t-1} + \\epsilon_{2,t} & \\text{if } z_t > c \\end{cases}\$

where:
- \$\\rho_1, \\rho_2\$ are the autoregressive coefficients for the two regimes.
- \$\\epsilon_{1,t} \\sim \\mathcal{N}(0, \\sigma_1^2)\$ and \$\\epsilon_{2,t} \\sim \\mathcal{N}(0, \\sigma_2^2)\$ are the innovation terms for each regime.
- \$c\$ is the threshold value, which is learned from the data.

# Computational Methods
- `:statespace` (Default, AD-friendly): An efficient, non-centered parameterization that samples
  unconstrained parameters and transforms them to ensure stationarity (`tanh` for
  `rho`) and positivity (`exp` for `sigma`). Recommended for gradient-based samplers.
- `:statespace_constrained` (Didactic): A more direct parameterization that samples
  `rho` and `sigma` from their specified (and appropriately constrained) priors.

# Inputs
- **Required**:
  - A temporal index variable (e.g., `year`) passed to `random()`.
  - `threshold_var`: `Symbol`, the name of the column in the data to use for thresholding.
- **Optional (in `random()` call)**:
  - `rho_regimes`: `Vector{<:UnivariateDistribution}`, priors for the `rho` parameter in each of the two regimes. Priors should be constrained to `(-1, 1)`. Default: `[truncated(Normal(0, 0.5), -1, 1), truncated(Normal(0, 0.5), -1, 1)]`.
  - `sigma_regimes`: `Vector{<:UnivariateDistribution}`, priors for the innovation standard deviation `sigma` in each regime. Priors should be positive. Default: `[Exponential(1.0), Exponential(1.0)]`.
  - `method`: `Symbol`, computational method (`:statespace` or `:statespace_constrained`). Default: `:statespace`.

# Outputs (Parameter Names)
- `threshold_unconstrained_<key>`: The unconstrained parameter for the threshold level.
- `innovations_<key>`: The raw standard normal innovations for the AR(1) processes.
- **For `:statespace` method**:
  - `unconstrained_rho1_<key>`, `unconstrained_rho2_<key>`: Unconstrained `rho` parameters.
  - `unconstrained_sigma1_<key>`, `unconstrained_sigma2_<key>`: Unconstrained `sigma` parameters.
- **For `:statespace_constrained` method**:
  - `rho1_<key>`, `rho2_<key>`: The `rho` parameters for each regime.
  - `sigma1_<key>`, `sigma2_<key>`: The `sigma` parameters for each regime.
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

function get_precomputes(m::TAR, M::NamedTuple, mod_data::Dict)::NamedTuple
    # --- Validation from get_datastructures! ---
    threshold_var = m.threshold_var
    if !hasproperty(M.data, threshold_var)
        error("Threshold variable ':$threshold_var' for TAR model not found in data.")
    end

    if !haskey(M, :t_idx) || !haskey(M, :t_N) || M.t_N == 0
        error("TAR model requires temporal indices (t_idx, t_N) to be set by the model processor.")
    end
    
    # --- Original precompute logic ---
    threshold_data_full = M.data[!, threshold_var]
    
    threshold_data_per_t = zeros(eltype(threshold_data_full), M.t_N)
    counts_per_t = zeros(Int, M.t_N)

    for i in 1:M.y_N
        t_val = M.t_idx[i]
        threshold_data_per_t[t_val] += threshold_data_full[i]
        counts_per_t[t_val] += 1
    end

    threshold_data_per_t ./= max.(1, counts_per_t)

    return (
        threshold_data=threshold_data_per_t,
        n_latent=M.t_N
    )
end

function get_priors(
    m::TAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    key = spec.key
    
    priors = String[]
    push!(priors, "$(p_names.threshold_unconstrained) ~ Normal(0.0, 1.0)")
    push!(priors, "$(p_names.innovations) ~ MvNormal(zeros(T, spec_registry[:$(key)].hyper.n_latent), I)")

    if m.method == :statespace
        push!(priors, "$(p_names.unconstrained_rho1) ~ Normal(0, 1.5)")
        push!(priors, "$(p_names.unconstrained_rho2) ~ Normal(0, 1.5)")
        push!(priors, "$(p_names.unconstrained_sigma1) ~ Normal(0, 1.0)")
        push!(priors, "$(p_names.unconstrained_sigma2) ~ Normal(0, 1.0)")
    else # :statespace_constrained
        push!(priors, "$(p_names.rho1) ~ $(_distribution_to_string(m.rho_regimes[1]))")
        push!(priors, "$(p_names.rho2) ~ $(_distribution_to_string(m.rho_regimes[2]))")
        push!(priors, "$(p_names.sigma1) ~ $(_distribution_to_string(m.sigma_regimes[1]))")
        push!(priors, "$(p_names.sigma2) ~ $(_distribution_to_string(m.sigma_regimes[2]))")
    end

    return join(priors, "\n    ")
end

function get_updates(
    m::TAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    
    local param_definitions
    if m.method == :statespace
        param_definitions = """
            local rho1 = tanh($(p_names.unconstrained_rho1))
            local rho2 = tanh($(p_names.unconstrained_rho2))
            local sigma1 = exp($(p_names.unconstrained_sigma1))
            local sigma2 = exp($(p_names.unconstrained_sigma2))
        """
    else # :statespace_constrained
        param_definitions = """
            local rho1 = $(p_names.rho1)
            local rho2 = $(p_names.rho2)
            local sigma1 = $(p_names.sigma1)
            local sigma2 = $(p_names.sigma2)
        """
    end

    return """
        # --- TAR Component: $(key) ($(m.method)) ---
        let
            $(param_definitions)
            local hyper = spec_registry[:$(key)].hyper
            local threshold_level = mean(hyper.threshold_data) + $(p_names.threshold_unconstrained)
            local innovations = $(p_names.innovations)
            
            local latent_field = Vector{eltype(innovations)}(undef, M.t_N)
            
            for t in 1:M.t_N
                local regime_indicator = hyper.threshold_data[t] > threshold_level
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

function get_effects(
    m::TAR, chain, M::NamedTuple, n_samples::Int, is_multivariate_model::Bool,
    outcomes_N::Int, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    t_N_train = M.t_N
    t_idx_full = isnothing(PS) ? M.t_idx : vcat(M.t_idx, get(PS, :t_idx, []))
    t_N_full = isempty(t_idx_full) ? 0 : maximum(t_idx_full)

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

    p_names_vec = string.(FlexiChains.parameters(chain))

    for k_outcome in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k_outcome)
        
        thresh_unconstrained_name = _find_parameter(p_names_vec, v.threshold_unconstrained, k_outcome, is_multivariate_model)
        innovations_name = _find_parameter(p_names_vec, v.innovations, k_outcome, is_multivariate_model)

        if isempty(thresh_unconstrained_name) || isempty(innovations_name)
            @warn "Base parameters for TAR component $(spec.key) (outcome $k_outcome) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        thresh_unconstrained_samples = get_params_vector(chain, thresh_unconstrained_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innovations_name, t_N_train)
        
        local rho1_samples, rho2_samples, sigma1_samples, sigma2_samples
        if m.method == :statespace
            rho1_raw_name = _find_parameter(p_names_vec, v.unconstrained_rho1, k_outcome, is_multivariate_model)
            rho2_raw_name = _find_parameter(p_names_vec, v.unconstrained_rho2, k_outcome, is_multivariate_model)
            sigma1_raw_name = _find_parameter(p_names_vec, v.unconstrained_sigma1, k_outcome, is_multivariate_model)
            sigma2_raw_name = _find_parameter(p_names_vec, v.unconstrained_sigma2, k_outcome, is_multivariate_model)
            if isempty(rho1_raw_name) || isempty(rho2_raw_name) || isempty(sigma1_raw_name) || isempty(sigma2_raw_name)
                @warn "Regime parameters for TAR component $(spec.key) (outcome $k_outcome) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            rho1_samples = tanh.(get_params_vector(chain, rho1_raw_name, 1)[:, 1])
            rho2_samples = tanh.(get_params_vector(chain, rho2_raw_name, 1)[:, 1])
            sigma1_samples = exp.(get_params_vector(chain, sigma1_raw_name, 1)[:, 1])
            sigma2_samples = exp.(get_params_vector(chain, sigma2_raw_name, 1)[:, 1])
        else # :statespace_constrained
            rho1_name = _find_parameter(p_names_vec, v.rho1, k_outcome, is_multivariate_model)
            rho2_name = _find_parameter(p_names_vec, v.rho2, k_outcome, is_multivariate_model)
            sigma1_name = _find_parameter(p_names_vec, v.sigma1, k_outcome, is_multivariate_model)
            sigma2_name = _find_parameter(p_names_vec, v.sigma2, k_outcome, is_multivariate_model)
            if isempty(rho1_name) || isempty(rho2_name) || isempty(sigma1_name) || isempty(sigma2_name)
                @warn "Regime parameters for TAR component $(spec.key) (outcome $k_outcome) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            rho1_samples = get_params_vector(chain, rho1_name, 1)[:, 1]
            rho2_samples = get_params_vector(chain, rho2_name, 1)[:, 1]
            sigma1_samples = get_params_vector(chain, sigma1_name, 1)[:, 1]
            sigma2_samples = get_params_vector(chain, sigma2_name, 1)[:, 1]
        end

        effect_k = zeros(Float64, N_total, n_samples)
        mean_thresh_data = mean(threshold_data_train)
        noise = M.noise
        
        for s in 1:n_samples
            threshold_level = mean_thresh_data + thresh_unconstrained_samples[s]
            innov_full = vcat(innovations_samples[s, :], randn(t_N_full - t_N_train))
            latent_field_s = zeros(Float64, t_N_full)

            for t in 1:t_N_full
                regime_indicator = threshold_data_full[t] > threshold_level
                curr_rho = regime_indicator ? rho2_samples[s] : rho1_samples[s]
                curr_sigma = regime_indicator ? sigma2_samples[s] : sigma1_samples[s]

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
