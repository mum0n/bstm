function _dynamics_logistic_f!(spec, eta, M, priors)
    """
    BSTM Dynamics Model v1.1.0
    Timestamp: 2026-06-26 18:45:00
    Synopsis: Implements a logistic growth model with a latent, time-varying fishing mortality (F).
    Rationale for v1.1.0:
        - Corrected `UndefVarError` by explicitly declaring all sampled parameters
          (`log_r_base`, `log_K_base`, etc.) as local variables before use.
    """
    key = spec.key
    params = spec.params
    T = eltype(eta)

    # 1. Define Priors with User Overrides
    r_prior = get(priors, "r_prior", LogNormal(0, 1))
    K_prior = get(priors, "K_prior", Normal(150, 50))
    sig_pop_prior = get(priors, "sig_pop_prior", Exponential(1.0))
    sig_F_prior = get(priors, "sig_F_prior", Exponential(0.5))

    # 2. Declare and Sample Core Dynamics Parameters
    log_r_base::T = 0.0
    log_K_base::T = 0.0
    sig_pop::T = 0.0
    sig_F::T = 0.0
    log_r_base ~ NamedDist(r_prior, Symbol("log_r_base_", key))
    log_K_base ~ NamedDist(K_prior, Symbol("log_K_base_", key))
    sig_pop ~ NamedDist(sig_pop_prior, Symbol("sig_pop_", key))
    sig_F ~ NamedDist(sig_F_prior, Symbol("sig_F_", key))

    # 3. Initialize State-Space Vectors
    log_pop_state = Vector{T}(undef, M.t_N)
    log_F_state = Vector{T}(undef, M.t_N)
    log_pop_state[1] ~ Normal(log_K_base - log(2.0), 1.0)
    log_F_state[1] ~ Normal(-2.0, 1.0)

    # 4. Handle Environmental Covariates on r and K
    r_cov_effect::T = 0.0
    r_cov_data = nothing
    if haskey(params, :r_covariate)
        r_cov_effect ~ NamedDist(Normal(0, 0.1), Symbol("r_cov_effect_", key))
        r_cov_data = M.data[!, params[:r_covariate]]
    end

    K_cov_effect::T = 0.0
    K_cov_data = nothing
    if haskey(params, :K_covariate)
        K_cov_effect ~ NamedDist(Normal(0, 0.1), Symbol("K_cov_effect_", key))
        K_cov_data = M.data[!, params[:K_covariate]]
    end

    # 5. State Transition Loop
    for t in 2:M.t_N
        current_r = isnothing(r_cov_data) ? exp(log_r_base) : exp(log_r_base + r_cov_effect * r_cov_data[t-1])
        current_K = isnothing(K_cov_data) ? exp(log_K_base) : exp(log_K_base + K_cov_effect * K_cov_data[t-1])
        prev_pop = exp(log_pop_state[t-1])
        prev_F = exp(log_F_state[t-1])
        growth = current_r * (1.0 - prev_pop / current_K) - prev_F
        log_pop_state[t] ~ Normal(log_pop_state[t-1] + growth, sig_pop)
        log_F_state[t] ~ Normal(log_F_state[t-1], sig_F)
    end

    # 6. Link Latent Population State to Observations
    for i in 1:M.y_N
        eta[i] += log_pop_state[M.t_idx[i]]
    end
end

function _dynamics_gompertz!(spec, eta, M, priors)
    """
    BSTM Dynamics Model v1.1.0
    Timestamp: 2026-06-26 18:45:00
    Synopsis: Implements a Gompertz growth model as a state-space process.
    Rationale for v1.1.0:
        - Corrected `UndefVarError` by explicitly declaring `r_dyn`, `K_dyn`, and `sig_dyn`.
    """
    key = spec.key
    params = spec.params
    T = eltype(eta)

    r_prior = get(priors, "r_prior", LogNormal(-1.5, 0.5))
    K_prior = get(priors, "K_prior", Normal(150, 50))
    sig_dyn_prior = get(priors, "sig_dyn_prior", Exponential(1.0))

    r_dyn::T = 0.0
    K_dyn::T = 0.0
    sig_dyn::T = 0.0
    r_dyn ~ NamedDist(r_prior, Symbol("r_dyn_", key))
    K_dyn ~ NamedDist(K_prior, Symbol("K_dyn_", key))
    sig_dyn ~ NamedDist(sig_dyn_prior, Symbol("sig_dyn_", key))

    log_pop_state = Vector{T}(undef, M.t_N)
    log_pop_state[1] ~ Normal(log(K_dyn / 2.0), 1.0)

    for t in 2:M.t_N
        growth = r_dyn * (log(K_dyn) - log_pop_state[t-1])
        log_pop_state[t] ~ Normal(log_pop_state[t-1] + growth, sig_dyn)
    end

    for i in 1:M.y_N
        eta[i] += log_pop_state[M.t_idx[i]]
    end
end

function _dynamics_advection!(spec, eta, M, priors)
    """
    BSTM Dynamics Model v1.1.0
    Timestamp: 2026-06-26 18:45:00
    Synopsis: Implements a simple advection (transport) model on a graph.
    Rationale for v1.1.0:
        - Corrected `UndefVarError` by explicitly declaring `velocity` and `sig_transport`.
    """
    key = spec.key
    T = eltype(eta)

    velocity_prior = get(priors, "velocity_prior", Normal(0, 0.5))
    sigma_prior = get(priors, "sigma_prior", Exponential(1.0))

    velocity::T = 0.0
    sig_transport::T = 0.0
    velocity ~ NamedDist(velocity_prior, Symbol("velocity_", key))
    sig_transport ~ NamedDist(sigma_prior, Symbol("sig_transport_", key))

    L = M.s_Q_template.matrix
    n_s = M.s_N
    n_t = M.t_N

    transport_field = Matrix{T}(undef, n_s, n_t)
    transport_field[:, 1] .~ Normal(0, 1)

    for t in 2:n_t
        drift = -velocity .* (L * transport_field[:, t-1])
        transport_field[:, t] ~ MvNormal(transport_field[:, t-1] .+ drift, sig_transport^2 * I)
    end

    for i in 1:M.y_N
        eta[i] += transport_field[M.s_idx[i], M.t_idx[i]]
    end
end

function _dynamics_diffusion!(spec, eta, M, priors)
    """
    BSTM Dynamics Model v1.1.0
    Timestamp: 2026-06-26 18:45:00
    Synopsis: Implements a simple diffusion model on a graph.
    Rationale for v1.1.0:
        - Corrected `UndefVarError` by explicitly declaring `diffusion_coeff` and `sig_transport`.
    """
    key = spec.key
    T = eltype(eta)
    
    diffusion_prior = get(priors, "diffusion_prior", LogNormal(-1, 1))
    sigma_prior = get(priors, "sigma_prior", Exponential(1.0))

    diffusion_coeff::T = 0.0
    sig_transport::T = 0.0
    diffusion_coeff ~ NamedDist(diffusion_prior, Symbol("diffusion_", key))
    sig_transport ~ NamedDist(sigma_prior, Symbol("sig_transport_", key))

    L = M.s_Q_template.matrix
    n_s = M.s_N
    n_t = M.t_N

    transport_field = Matrix{T}(undef, n_s, n_t)
    transport_field[:, 1] .~ Normal(0, 1)

    for t in 2:n_t
        drift = -diffusion_coeff .* (L * transport_field[:, t-1])
        transport_field[:, t] ~ MvNormal(transport_field[:, t-1] .+ drift, sig_transport^2 * I)
    end

    for i in 1:M.y_N
        eta[i] += transport_field[M.s_idx[i], M.t_idx[i]]
    end
end

function _dynamics_linked_K!(spec, eta, M, priors)
    """
    BSTM Dynamics Model v1.1.0
    Timestamp: 2026-06-26 18:45:00
    Synopsis: Implements a logistic growth model where K is linked to the statistical environment.
    Rationale for v1.1.0:
        - Corrected `UndefVarError` by explicitly declaring `r_dyn`, `sig_pop`, and `K_slope`.
    """
    key = spec.key
    params = spec.params
    T = eltype(eta)

    # 1. Define Priors
    r_prior = get(priors, "r_prior", LogNormal(0, 1))
    sig_pop_prior = get(priors, "sig_pop_prior", Exponential(1.0))
    K_slope_prior = get(priors, "K_slope_prior", Normal(1, 0.5))

    # 2. Declare and Sample Core Dynamics Parameters
    r_dyn::T = 0.0
    sig_pop::T = 0.0
    K_slope::T = 0.0
    r_dyn ~ NamedDist(r_prior, Symbol("r_dyn_", key))
    sig_pop ~ NamedDist(sig_pop_prior, Symbol("sig_pop_", key))
    K_slope ~ NamedDist(K_slope_prior, Symbol("K_slope_", key))

    # 3. Pre-compute time-step specific statistical effects from eta
    eta_statistical_by_time = [mean(eta[M.t_idx .== t]) for t in 1:M.t_N]

    # 4. Initialize State-Space Vector
    log_pop_state = Vector{T}(undef, M.t_N)
    log_K_initial = eta_statistical_by_time[1] * K_slope
    log_pop_state[1] ~ Normal(log_K_initial - log(2.0), 1.0)

    # 5. State Transition Loop
    for t in 2:M.t_N
        log_K_current = eta_statistical_by_time[t-1] * K_slope
        current_K = exp(log_K_current)
        prev_pop = exp(log_pop_state[t-1])
        growth = r_dyn * (1.0 - prev_pop / current_K)
        log_pop_state[t] ~ Normal(log_pop_state[t-1] + growth, sig_pop)
    end

    # 6. Link Latent Population State back to the Linear Predictor
    for i in 1:M.y_N
        eta[i] += log_pop_state[M.t_idx[i]]
    end
end

function _apply_dynamics_model!(spec, eta, M, priors)
    """
    BSTM Dynamics Model v1.0.0
    Timestamp: 2026-06-26 10:21:21
    Synopsis: A dispatcher function that selects and executes the appropriate dynamics model.
    Inputs:
        - spec, eta, M, priors: Standard dynamics model inputs.
    Details:
        - Reads the `model` parameter from the `spec` to determine which dynamics function to call.
        - Supports a set of built-in models (`logistic_f`, `gompertz`, etc.).
        - Allows for user-defined custom models by specifying `model="custom"` and providing a function symbol via the `func` parameter.
    """

    params = spec.params
    model_name = get(params, :model, "logistic_f")

    # # 1. Dispatch to Built-in Dynamics Models
    if model_name == "logistic_f" || model_name == "ricker" || model_name == "basic_logistic"
        _dynamics_logistic_f!(spec, eta, M, priors)
    elseif model_name == "gompertz"
        _dynamics_gompertz!(spec, eta, M, priors)
    elseif model_name == "advection"
        _dynamics_advection!(spec, eta, M, priors)
    elseif model_name == "diffusion"
        _dynamics_diffusion!(spec, eta, M, priors)
    elseif model_name == "linked_K_logistic"
        _dynamics_linked_K!(spec, eta, M, priors)
    # 2. Dispatch to User-Defined Custom Model
    elseif model_name == "custom"
        func_sym = get(params, :func, nothing)
        if !isnothing(func_sym) && isdefined(Main, func_sym)
            user_func = getfield(Main, func_sym)
            user_func(spec, eta, M, priors)
        else
            @warn "Custom dynamics function ':$func_sym' not found in Main scope. No dynamics applied."
        end
    # 3. Fallback for Unknown Models
    else
        @warn "Unsupported dynamics model: '$model_name'. No dynamics applied."
    end
end
