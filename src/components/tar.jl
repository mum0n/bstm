"""
    TAR <: ComponentModel

A component for a Threshold Autoregressive (TAR) model, which captures
regime-switching temporal dynamics. The model switches between two different AR(1)
processes based on whether an external `threshold_var` is above or below a
learned threshold.

# Version
v1.1.2 (2026-08-19)

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
    # --- Validation ---
    threshold_var = m.threshold_var
    if !hasproperty(M.data, threshold_var)
        error("Threshold variable ':$threshold_var' for TAR model not found in data.")
    end

    if !haskey(M, :t_idx) || !haskey(M, :t_N) || M.t_N == 0
        error("TAR model requires temporal indices (t_idx, t_N) to be set by the model processor.")
    end
    
    # --- Precompute logic on CPU ---
    threshold_data_full_cpu = M.data[!, threshold_var]
    t_idx_cpu = M.t_idx

    threshold_data_per_t = zeros(eltype(threshold_data_full_cpu), M.t_N)
    counts_per_t = zeros(Int, M.t_N)

    for i in 1:M.y_N
        t_val = t_idx_cpu[i]
        threshold_data_per_t[t_val] += threshold_data_full_cpu[i]
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
    push!(priors, "$(p_names.ure) ~ MvNormal(zeros(T, spec_registry[:$(key)].hyper.n_latent), I)")

    if m.method == :statespace
        push!(priors, "$(p_names.rho1_unconstrained) ~ Normal(0, 1.5)")
        push!(priors, "$(p_names.rho2_unconstrained) ~ Normal(0, 1.5)")
        push!(priors, "$(p_names.sigma1_unconstrained) ~ Normal(0, 1.0)")
        push!(priors, "$(p_names.sigma2_unconstrained) ~ Normal(0, 1.0)")
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
            local rho1 = tanh($(p_names.rho1_unconstrained))
            local rho2 = tanh($(p_names.rho2_unconstrained))
            local sigma1 = exp($(p_names.sigma1_unconstrained))
            local sigma2 = exp($(p_names.sigma2_unconstrained))
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
            local innovations = $(p_names.ure)
            
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
            $(p_names.sre) = latent_field
            $(eta_target) .+= view($(p_names.sre), M.t_idx)
        end
    """
end


function get_effects(
    m::TAR, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = if occursin("FlexiChain", string(typeof(chain)))
        size(chain, 1) * FlexiChains.nchains(chain)
    else
        size(chain, 1) * size(chain, 3)
    end
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    # --- Get precomputed data ---
    hyper = spec.hyper
    t_N_train = hyper.n_latent
    threshold_data_train_cpu = hyper.threshold_data

    # --- Index and Threshold Data Handling for Training and Prediction ---
    t_idx_train_cpu = M.t_idx
    t_idx_full_cpu = if !isnothing(PS) && haskey(PS.data, :t_idx)
        vcat(t_idx_train_cpu, PS.data.t_idx)
    else
        t_idx_train_cpu
    end
    t_N_full = isempty(t_idx_full_cpu) ? 0 : maximum(t_idx_full_cpu)
    N_total = length(t_idx_full_cpu)

    threshold_data_full_cpu = if !isnothing(PS) && hasproperty(PS.data, m.threshold_var)
        pred_threshold_data = PS.data[!, m.threshold_var]
        len_pred = t_N_full - t_N_train
        if len_pred > 0
            pred_data_agg = pred_threshold_data[1:min(length(pred_threshold_data), len_pred)]
            vcat(threshold_data_train_cpu, pred_data_agg)
        else
            threshold_data_train_cpu
        end
    else
        threshold_data_train_cpu
    end
    if length(threshold_data_full_cpu) < t_N_full
        padding = fill(mean(threshold_data_train_cpu), t_N_full - length(threshold_data_full_cpu))
        threshold_data_full_cpu = vcat(threshold_data_full_cpu, padding)
    end

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop ---
    for k_outcome in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k_outcome)
        
        thresh_unconstrained_name = _find_parameter(p_names, string(v.threshold_unconstrained), k_outcome, is_multivariate_model)
        ure_name = _find_parameter(p_names, string(v.ure), k_outcome, is_multivariate_model)

        if isempty(thresh_unconstrained_name) || isempty(ure_name)
            @warn "Base parameters for TAR component $(spec.key) (outcome $k_outcome) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (CPU)
        thresh_unconstrained_samples_cpu = get_params_vector(chain, thresh_unconstrained_name, 1)[:, 1]
        ure_samples_cpu = get_params_matrix(chain, ure_name, t_N_train)
        
        local rho1_samples_cpu, rho2_samples_cpu, sigma1_samples_cpu, sigma2_samples_cpu
        if m.method == :statespace
            rho1_unconstrained_name = _find_parameter(p_names, string(v.rho1_unconstrained), k_outcome, is_multivariate_model)
            rho2_unconstrained_name = _find_parameter(p_names, string(v.rho2_unconstrained), k_outcome, is_multivariate_model)
            sigma1_unconstrained_name = _find_parameter(p_names, string(v.sigma1_unconstrained), k_outcome, is_multivariate_model)
            sigma2_unconstrained_name = _find_parameter(p_names, string(v.sigma2_unconstrained), k_outcome, is_multivariate_model)
            if isempty(rho1_unconstrained_name) || isempty(rho2_unconstrained_name) || isempty(sigma1_unconstrained_name) || isempty(sigma2_unconstrained_name)
                @warn "Regime parameters for TAR component $(spec.key) (outcome $k_outcome) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            rho1_samples_cpu = tanh.(get_params_vector(chain, rho1_unconstrained_name, 1)[:, 1])
            rho2_samples_cpu = tanh.(get_params_vector(chain, rho2_unconstrained_name, 1)[:, 1])
            sigma1_samples_cpu = exp.(get_params_vector(chain, sigma1_unconstrained_name, 1)[:, 1])
            sigma2_samples_cpu = exp.(get_params_vector(chain, sigma2_unconstrained_name, 1)[:, 1])
        else # :statespace_constrained
            rho1_name = _find_parameter(p_names, string(v.rho1), k_outcome, is_multivariate_model)
            rho2_name = _find_parameter(p_names, string(v.rho2), k_outcome, is_multivariate_model)
            sigma1_name = _find_parameter(p_names, string(v.sigma1), k_outcome, is_multivariate_model)
            sigma2_name = _find_parameter(p_names, string(v.sigma2), k_outcome, is_multivariate_model)
            if isempty(rho1_name) || isempty(rho2_name) || isempty(sigma1_name) || isempty(sigma2_name)
                @warn "Regime parameters for TAR component $(spec.key) (outcome $k_outcome) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            rho1_samples_cpu = get_params_vector(chain, rho1_name, 1)[:, 1]
            rho2_samples_cpu = get_params_vector(chain, rho2_name, 1)[:, 1]
            sigma1_samples_cpu = get_params_vector(chain, sigma1_name, 1)[:, 1]
            sigma2_samples_cpu = get_params_vector(chain, sigma2_name, 1)[:, 1]
        end

        # Initialize the output matrix for the full effect on the CPU
        effect_k_cpu = zeros(Float64, N_total, n_samples)
        mean_thresh_data_cpu = mean(threshold_data_train_cpu)
        noise = M.noise
        
        # --- Sample-wise Reconstruction on the CPU ---
        for s in 1:n_samples
            threshold_level = mean_thresh_data_cpu + thresh_unconstrained_samples_cpu[s]
            
            innov_train_cpu = ure_samples_cpu[s, :]
            innov_pred_cpu = randn(Float32, t_N_full - t_N_train)
            innov_full_cpu = vcat(innov_train_cpu, innov_pred_cpu)
            
            latent_field_s_cpu = zeros(Float64, t_N_full)

            rho1_s = rho1_samples_cpu[s]
            rho2_s = rho2_samples_cpu[s]
            sigma1_s = sigma1_samples_cpu[s]
            sigma2_s = sigma2_samples_cpu[s]

            for t in 1:t_N_full
                regime_indicator = threshold_data_full_cpu[t] > threshold_level
                curr_rho = regime_indicator ? rho2_s : rho1_s
                curr_sigma = regime_indicator ? sigma2_s : sigma1_s

                if t == 1
                    latent_field_s_cpu[t] = (innov_full_cpu[t] * curr_sigma) / sqrt(1.0 - curr_rho^2 + noise)
                else
                    latent_field_s_cpu[t] = curr_rho * latent_field_s_cpu[t-1] + innov_full_cpu[t] * curr_sigma
                end
            end
            effect_k_cpu[:, s] = latent_field_s_cpu[t_idx_full_cpu]
        end
        
        push!(structured_effects, effect_k_cpu)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
 