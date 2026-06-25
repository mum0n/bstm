# BSTM Dynamics Model Library v24.1.0
# Timestamp: 2026-06-25 19:00:00

function _dynamics_logistic_f!(spec, eta, M, priors)
    # # BSTM State-Space Dynamics: Logistic Growth with Fishing
    # Implements a logistic growth model with a latent, time-varying fishing mortality (F).
    # Allows for environmental covariates on growth rate (r) and carrying capacity (K).

    key = spec.key
    params = spec.params
    T = eltype(eta)

    # # 1. Define Priors with User Overrides
    # Rationale: Uses priors from the hyperprior registry or falls back to sensible defaults.
    r_prior = get(priors, "r_prior", LogNormal(0, 1))
    K_prior = get(priors, "K_prior", Normal(150, 50))
    sig_pop_prior = get(priors, "sig_pop_prior", Exponential(1.0))
    sig_F_prior = get(priors, "sig_F_prior", Exponential(0.5))

    # # 2. Sample Core Dynamics Parameters
    log_r_base ~ NamedDist(r_prior, Symbol("log_r_base_", key))
    log_K_base ~ NamedDist(K_prior, Symbol("log_K_base_", key))
    sig_pop ~ NamedDist(sig_pop_prior, Symbol("sig_pop_", key))
    sig_F ~ NamedDist(sig_F_prior, Symbol("sig_F_", key))

    # # 3. Initialize State-Space Vectors
    log_pop_state = Vector{T}(undef, M.t_N)
    log_F_state = Vector{T}(undef, M.t_N)

    # Initial states for the population and fishing mortality processes
    log_pop_state[1] ~ Normal(log_K_base - log(2.0), 1.0) # Start near K/2
    log_F_state[1] ~ Normal(-2.0, 1.0) # Start with low F

    # # 4. Handle Environmental Covariates on r and K
    # Rationale: This allows r and K to be time-varying based on external data.
    r_cov_effect = 0.0
    r_cov_data = nothing
    if haskey(params, :r_covariate)
        r_cov_effect ~ NamedDist(Normal(0, 0.1), Symbol("r_cov_effect_", key))
        r_cov_data = M.data[!, params[:r_covariate]]
    end

    K_cov_effect = 0.0
    K_cov_data = nothing
    if haskey(params, :K_covariate)
        K_cov_effect ~ NamedDist(Normal(0, 0.1), Symbol("K_cov_effect_", key))
        K_cov_data = M.data[!, params[:K_covariate]]
    end

    # # 5. State Transition Loop (Euler-Maruyama Discretization)
    for t in 2:M.t_N
        # Calculate time-varying r and K for the previous time step
        current_r = isnothing(r_cov_data) ? exp(log_r_base) : exp(log_r_base + r_cov_effect * r_cov_data[t-1])
        current_K = isnothing(K_cov_data) ? exp(log_K_base) : exp(log_K_base + K_cov_effect * K_cov_data[t-1])

        prev_pop = exp(log_pop_state[t-1])
        prev_F = exp(log_F_state[t-1])
        
        # Logistic growth minus harvest
        growth = current_r * (1.0 - prev_pop / current_K) - prev_F
        
        # Population state follows a random walk with the calculated drift
        log_pop_state[t] ~ Normal(log_pop_state[t-1] + growth, sig_pop)
        # Fishing mortality follows a random walk on the log scale
        log_F_state[t] ~ Normal(log_F_state[t-1], sig_F)
    end

    # # 6. Link Latent Population State to Observations
    for i in 1:M.y_N
        eta[i] += log_pop_state[M.t_idx[i]]
    end
end

function _dynamics_gompertz!(spec, eta, M, priors)
    # # BSTM State-Space Dynamics: Gompertz Growth Model
    key = spec.key
    params = spec.params
    T = eltype(eta)

    r_prior = get(priors, "r_prior", LogNormal(-1.5, 0.5)) # Gompertz r is different
    K_prior = get(priors, "K_prior", Normal(150, 50))
    sig_dyn_prior = get(priors, "sig_dyn_prior", Exponential(1.0))

    r_dyn ~ NamedDist(r_prior, Symbol("r_dyn_", key))
    K_dyn ~ NamedDist(K_prior, Symbol("K_dyn_", key))
    sig_dyn ~ NamedDist(sig_dyn_prior, Symbol("sig_dyn_", key))

    log_pop_state = Vector{T}(undef, M.t_N)
    log_pop_state[1] ~ Normal(log(K_dyn / 2.0), 1.0)

    for t in 2:M.t_N
        # Gompertz growth equation for log-population
        growth = r_dyn * (log(K_dyn) - log_pop_state[t-1])
        log_pop_state[t] ~ Normal(log_pop_state[t-1] + growth, sig_dyn)
    end

    for i in 1:M.y_N
        eta[i] += log_pop_state[M.t_idx[i]]
    end
end

function _dynamics_advection!(spec, eta, M, priors)
    key = spec.key
    T = eltype(eta)

    velocity_prior = get(priors, "velocity_prior", Normal(0, 0.5))
    sigma_prior = get(priors, "sigma_prior", Exponential(1.0))

    velocity ~ NamedDist(velocity_prior, Symbol("velocity_", key))
    sig_transport ~ NamedDist(sigma_prior, Symbol("sig_transport_", key))

    L = M.s_Q_template.matrix # Graph Laplacian
    n_s = M.s_N
    n_t = M.t_N

    transport_field = Matrix{T}(undef, n_s, n_t)
    transport_field[:, 1] .~ Normal(0, 1)

    # State transition
    for t in 2:n_t
        drift = -velocity .* (L * transport_field[:, t-1])
        transport_field[:, t] ~ MvNormal(transport_field[:, t-1] .+ drift, sig_transport^2 * I)
    end

    # Link to observations
    for i in 1:M.y_N
        eta[i] += transport_field[M.s_idx[i], M.t_idx[i]]
    end
end

function _dynamics_diffusion!(spec, eta, M, priors)
    key = spec.key
    T = eltype(eta)
    
    diffusion_prior = get(priors, "diffusion_prior", LogNormal(-1, 1))
    sigma_prior = get(priors, "sigma_prior", Exponential(1.0))

    diffusion_coeff ~ NamedDist(diffusion_prior, Symbol("diffusion_", key))
    sig_transport ~ NamedDist(sigma_prior, Symbol("sig_transport_", key))

    L = M.s_Q_template.matrix # Graph Laplacian
    n_s = M.s_N
    n_t = M.t_N

    transport_field = Matrix{T}(undef, n_s, n_t)
    transport_field[:, 1] .~ Normal(0, 1)

    # State transition
    for t in 2:n_t
        drift = -diffusion_coeff .* (L * transport_field[:, t-1])
        transport_field[:, t] ~ MvNormal(transport_field[:, t-1] .+ drift, sig_transport^2 * I)
    end

    # Link to observations
    for i in 1:M.y_N
        eta[i] += transport_field[M.s_idx[i], M.t_idx[i]]
    end
end

function _dynamics_linked_K!(spec, eta, M, priors)
    # # BSTM State-Space Dynamics: Logistic Growth with eta-dependent Carrying Capacity
    # This model links the carrying capacity (K) to the other latent statistical
    # processes (spatial, temporal, covariate effects) aggregated in `eta`.
    # IMPORTANT: This assumes the dynamics() module is called *after* all other
    # statistical manifolds have been applied to `eta`.
    # formula = "y ~ 1 + spatial(s_idx, model='bym2') + dynamics(t_idx, model=\"linked_K_logistic\")"


    key = spec.key
    params = spec.params
    T = eltype(eta)

    # # 1. Define Priors
    r_prior = get(priors, "r_prior", LogNormal(0, 1))
    sig_pop_prior = get(priors, "sig_pop_prior", Exponential(1.0))
    K_slope_prior = get(priors, "K_slope_prior", Normal(1, 0.5)) # Prior on the scaling effect of eta on K

    # # 2. Sample Core Dynamics Parameters
    r_dyn ~ NamedDist(r_prior, Symbol("r_dyn_", key))
    sig_pop ~ NamedDist(sig_pop_prior, Symbol("sig_pop_", key))
    K_slope ~ NamedDist(K_slope_prior, Symbol("K_slope_", key))

    # # 3. Pre-compute time-step specific statistical effects from eta
    # This averages the current `eta` values for each time step to create a
    # time-series of the underlying statistical environment.
    eta_statistical_by_time = [mean(eta[M.t_idx .== t]) for t in 1:M.t_N]

    # # 4. Initialize State-Space Vector
    log_pop_state = Vector{T}(undef, M.t_N)
    
    # Initial population state depends on the environment at t=1
    log_K_initial = eta_statistical_by_time[1] * K_slope
    log_pop_state[1] ~ Normal(log_K_initial - log(2.0), 1.0)

    # # 5. State Transition Loop
    for t in 2:M.t_N
        # Carrying capacity at the previous time step is a function of the statistical environment
        log_K_current = eta_statistical_by_time[t-1] * K_slope
        current_K = exp(log_K_current)

        prev_pop = exp(log_pop_state[t-1])
        
        # Logistic growth equation
        growth = r_dyn * (1.0 - prev_pop / current_K)
        
        # Population state follows a random walk with the calculated drift
        log_pop_state[t] ~ Normal(log_pop_state[t-1] + growth, sig_pop)
    end

    # # 6. Link Latent Population State back to the Linear Predictor
    # This adds the population dynamics effect on top of the existing statistical effects.
    for i in 1:M.y_N
        eta[i] += log_pop_state[M.t_idx[i]]
    end
end

function _apply_dynamics_model!(spec, eta, M, priors)
    # # BSTM Dynamics Model Dispatcher v24.1.0
    # Timestamp: 2026-06-25 19:00:00
    # Rationale: Centralizes the selection and execution of population dynamics models.
    # Supports built-in models and allows for user-defined custom models.

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
    # # 2. Dispatch to User-Defined Custom Model
    elseif model_name == "custom"
        func_sym = get(params, :func, nothing)
        if !isnothing(func_sym) && isdefined(Main, func_sym)
            user_func = getfield(Main, func_sym)
            user_func(spec, eta, M, priors)
        else
            @warn "Custom dynamics function ':$func_sym' not found in Main scope. No dynamics applied."
        end
    # # 3. Fallback for Unknown Models
    else
        @warn "Unsupported dynamics model: '$model_name'. No dynamics applied."
    end
end