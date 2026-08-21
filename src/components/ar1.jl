"""
    AR1 <: ComponentModel

A component model for a first-order autoregressive (AR1) process, fundamental for
modeling time series data with serial correlation.

# Version
v1.0.0

# Mathematical Summary
The AR1 process models the value of a latent field \$\\phi_t\$ at time \$t\$ as a
fraction of its value at the previous time step, plus an independent innovation
term \$\\epsilon_t\$:
\$\\phi_t = \\rho \\phi_{t-1} + \\epsilon_t\$, where \$\\epsilon_t \\sim \\mathcal{N}(0,
  \\sigma^2)\$

To ensure stationarity (\$-1 < \\rho < 1\$), the autocorrelation parameter \$\\rho\$ is
parameterized via a `tanh` transformation of an unconstrained parameter:
\$\\rho = \\tanh(\\rho_{\\text{unconstrained}})\$

# Computational Methods
This component supports three numerical methods for temporal evolution,
controlled by the `method` parameter in the `random()` call:
1.  **:statespace** (Default, AD-friendly): Implemented using a numerically stable
    state-space formulation. This is the recommended method for most applications.
2.  **:spectral** (AD-friendly): Implemented using a spectral decomposition of the
    AR(1) precision matrix. This method is fully differentiable and can be
    advantageous for Hamiltonian Monte Carlo samplers.
3.  **:centered** (Didactic, Not AD-friendly): Explicitly constructs the dense
    Toeplitz covariance matrix and samples the latent field directly. This is a
    didactic alternative that is less efficient for MCMC and not AD-compatible.

# Inputs
- **Required**:
  - A temporal index variable (e.g., `year`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `rho_unconstrained`: A `Distribution` for the prior on the unconstrained
    autocorrelation parameter. Default: `Normal(0, 1.5)`.
  - `sigma`: A `Distribution` for the prior on the innovations' standard deviation.
    Default: `Exponential(1.0)`.
  - `method`: A `Symbol` specifying the computational method (`:statespace`,
    `:spectral`, or `:centered`). Default: `:statespace`.

# Outputs (Parameter Names)
- `rho_unconstrained_<key>`: The unconstrained parameter sampled by Turing.
- `rho_<key>`: The transformed autocorrelation parameter, `tanh(rho_unconstrained_<key>)`.
- `sigma_<key>`: The standard deviation of the AR1 innovations.
- `ure_<key>`: Standard normal innovations driving the process.
- `sre_<key>`: Reconstructed temporal latent field.

# Key References
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Rue, H., Martino, S., & Chopin, N. (2009). Approximate Bayesian inference for latent
  Gaussian models by using integrated nested Laplace approximations. *Journal of the Royal
  Statistical Society: Series B (Statistical Methodology)*, 71(2), 319-392.
"""
struct AR1 <: ComponentModel
    rho_unconstrained::Distribution
    sigma::Distribution
    method::Symbol # :statespace, :spectral, :centered, :marginalized
end

COMPONENT_TYPE_REGISTRY[:ar1] = AR1
COMPONENT_CONSTRUCTORS[:ar1] = (p, params) -> AR1(
    get(p, :rho_unconstrained, Normal(0, 1.5)),
    get(p, :sigma, Exponential(1.0)),
    get(params, :method, :statespace) # Default method
)
MODEL_TO_STRUCTURE_MAP[:ar1] = :temporal

"""
    get_precomputes(m::AR1, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-dependent setup for the AR1 model. This version is CPU-only.
"""
function get_precomputes(m::AR1, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.t_N
    
    template = build_structure_template(:ar1, n)
    return (
        Q_template=template.matrix, 
        n_latent=n, 
        U=template.U, 
        L=template.L
    )
end

"""
    _ar1_covariance_matrix(rho, sigma, n, noise)

Helper function to construct the dense Toeplitz covariance matrix for a stationary
AR(1) process. Used by the `:centered` method. This version is CPU-only.
"""
function _ar1_covariance_matrix(rho, sigma, n, noise)
    T_num = promote_type(typeof(rho), typeof(sigma), typeof(noise))
    var = sigma^2 / (one(T_num) - rho^2 + T_num(noise))
    
    # Vectorized approach for creating the Toeplitz matrix
    time_diffs = abs.((1:n) .- (1:n)')
    C = var .* (rho .^ time_diffs)
    
    return C
end

"""
    _ar1_precision_matrix(rho, sigma, n, noise)

Helper function to construct the sparse precision matrix for a stationary AR(1) process.
This is used by the `:marginalized` method.
"""
function _ar1_precision_matrix(rho, sigma, n, noise)
    T_num = promote_type(typeof(rho), typeof(sigma), typeof(noise))
    
    # For a stationary AR(1) process, the precision matrix Q is tridiagonal.
    # Q_tt = (1 + rho^2) / (sigma^2 * (1 - rho^2)) for t=2,...,n-1
    # Q_11 = 1 / (sigma^2 * (1 - rho^2))
    # Q_nn = 1 / (sigma^2 * (1 - rho^2))
    # Q_t,t-1 = Q_t-1,t = -rho / (sigma^2 * (1 - rho^2))
    
    # To avoid division by (1 - rho^2) in each element, we can factor it out.
    # Let Q_base be the precision matrix without the (1 - rho^2) factor.
    # Then Q = Q_base / (sigma^2 * (1 - rho^2)).
    
    # Construct the tridiagonal matrix
    diag_val = fill(one(T_num) + rho^2, n)
    diag_val[1] = one(T_num)
    diag_val[n] = one(T_num)
    
    off_diag_val = fill(-rho, n - 1)
    
    Q_base = Tridiagonal(off_diag_val, diag_val, off_diag_val)
    
    # Add jitter for numerical stability and scale by sigma^2 * (1 - rho^2)
    # The (1 - rho^2) factor ensures stationarity and is part of the variance.
    return Symmetric(Q_base ./ (sigma^2 * (one(T_num) - rho^2 + T_num(noise))))
end

"""
    _ar1_log_marginal_likelihood(y_residual, t_idx, t_N, rho, sigma, y_sigma, noise=1e-6)

Computes the exact log marginal likelihood for an AR(1) Gaussian Markov Random Field
  integrated out analytically.
Runs in linear time O(N + T) using sparse tridiagonal Cholesky factorization.
"""
function _ar1_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    t_idx::AbstractVector{Int},
    t_N::Int,
    rho::T,
    sigma::T,
    y_sigma::T,
    noise::Real=1e-6
) where {T}
    N = length(y_residual)
    T_num = promote_type(T, typeof(noise))
    
    # Pre-accumulate observation counts and sums per time index
    N_t = zeros(T_num, t_N)
    S_t = zeros(T_num, t_N)
    for i in 1:N
        t = t_idx[i]
        if 1 <= t <= t_N
            N_t[t] += one(T_num)
            S_t[t] += y_residual[i]
        end
    end
    
    inv_sigma_y2 = one(T_num) / (y_sigma^2 + T_num(noise))
    
    # Construct base tridiagonal prior precision
    diag_val = fill(one(T_num) + rho^2, t_N)
    diag_val[1] = one(T_num)
    diag_val[t_N] = one(T_num)
    off_diag_val = fill(-rho, t_N - 1)
    
    scale = sigma^2 * (one(T_num) - rho^2 + T_num(noise))
    
    diag_post = diag_val .+ (N_t .* (inv_sigma_y2 * scale))
    Q_post_scaled = Tridiagonal(off_diag_val, diag_post, off_diag_val)
    
    F = cholesky(Symmetric(Q_post_scaled))
    
    # Determinant term: log |Q_prior| - log |Q_post| = log(1 - rho^2) - 2 * sum(log, diag(F.U))
    log_det_diff = log(max(one(T_num) - rho^2, T_num(noise))) - 2 * sum(log.(diag(F.U)))
    
    # Quadratic term: scale * (b^T Q_post_scaled^{-1} b)
    b = S_t .* inv_sigma_y2
    v = F.L \ b
    quad_term = scale * dot(v, v)
    
    # Total log marginal likelihood
    log_lik = - (N / 2) * log(2 * T_num(pi) * (y_sigma^2 + T_num(noise))) -
              (inv_sigma_y2 / 2) * dot(y_residual, y_residual) +
              (1 / 2) * log_det_diff +
              (1 / 2) * quad_term
              
    return log_lik
end

"""
    get_priors(m::AR1, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates Turing code for the priors of the `AR1` component.
"""
function get_priors(
    m::AR1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    is_multivariate = (arch == "multivariate")
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))
    is_shared = get(spec.params, :shared, false)
    priors_acc = String[]

    # These hyperparameters can be shared in multivariate models.
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")
        push!(
            priors_acc,
            "$(p_names.rho_unconstrained) ~ " *
            "$(_distribution_to_string(m.rho_unconstrained))"
        )
    end

    # The innovations (ure) are only sampled if the method is not marginalized.
    if m.method in [:statespace, :spectral]
        push!(
            priors_acc,
            "$(p_names.ure) ~ MvNormal(zeros(T, $(n_latent)), I)"
        )
    end
    
    return join(priors_acc, "\n    ")
end

"""
    get_updates(m::AR1, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for constructing the AR1 effect. This version is CPU-only.
"""
function get_updates(
    m::AR1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx"
    n_latent = spec.hyper.n_latent
    key = spec.key

    statespace_code = """
        # --- AR1 Component (State-Space): $(key) ---
        rho = tanh($(p_names.rho_unconstrained))
        $(p_names.sre) = ar1_statespace(
            rho, $(p_names.sigma), $(p_names.ure), $(n_latent), M.noise
        )
        $(eta_target) = $(eta_target) .+ view($(p_names.sre), M.$(index_var))
    """

    spectral_code = """
        # --- AR1 Component (Spectral): $(key) ---
        rho = tanh($(p_names.rho_unconstrained))
        let
            hyper = spec_registry[:$(key)].hyper
            U = hyper.U
            L_base = hyper.L
            lambda_vals = (one(T) + rho^2) .+ rho .* L_base
            diag_D = $(p_names.sigma) ./ sqrt.(lambda_vals .+ M.noise)
            $(p_names.sre) = U * (diag_D .* $(p_names.ure))
            $(eta_target) = $(eta_target) .+ view($(p_names.sre), M.$(index_var))
        end
    """

    centered_code = """
        # --- AR1 Component (Centered, Didactic): $(key) ---
        rho = tanh($(p_names.rho_unconstrained))
        let
            K = _ar1_covariance_matrix(
                rho, $(p_names.sigma), $(n_latent), M.noise
            )
            $(p_names.sre) ~ MvNormal(zeros(T, $(n_latent)), Symmetric(K))
            $(eta_target) = $(eta_target) .+ view($(p_names.sre), M.$(index_var))
        end
    """

    marginalized_code = """
        # --- AR1 Component (Marginalized): $(key) ---
        let
            rho = tanh($(p_names.rho_unconstrained))
            y_residual = M.y_obs .- $(eta_target)
            log_lik_marginalized_$(key) = _ar1_log_marginal_likelihood(
                y_residual,
                M.$(index_var),
                $(n_latent),
                rho,
                $(p_names.sigma),
                y_sigma,
                M.noise
            )
            Turing.@addlogprob! log_lik_marginalized_$(key)
        end
    """

    if m.method == :statespace
        return statespace_code
    elseif m.method == :spectral
        return spectral_code
    elseif m.method == :centered
        return centered_code
    elseif m.method == :marginalized
        return marginalized_code
    end
    
    error(
        "Unsupported method '$(m.method)' for AR1 component. " *
        "Use `:statespace`, `:spectral`, `:centered`, or `:marginalized`."
    )
end

function get_effects(
    m::AR1, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = _get_chain_n_samples(chain)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    noise_val = get(M, :noise, 1e-6)

    # --- Index Handling: Combine training and prediction sets on CPU ---
    t_idx_train = M.t_idx
    t_idx_full = if !isnothing(PS) && hasproperty(PS.data, :t_idx)
        vcat(t_idx_train, PS.data.t_idx)
    else
        t_idx_train
    end
    
    t_N_train = M.t_N
    t_N_full = isempty(t_idx_full) ? 0 : maximum(t_idx_full)
    N_total = length(t_idx_full)

    structured_effects = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        rho_name = _find_parameter(p_names, string(p_names_k.rho_unconstrained), k,
            is_multivariate_model)
        
        if isempty(sigma_name) || isempty(rho_name)
            @warn "Base parameters for AR1 component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # All samples are extracted to CPU arrays
        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        rho_samples = tanh.(get_params_vector(chain, rho_name, 1)[:, 1])
        
        latent_field_samples = zeros(Float64, t_N_full, n_samples)
        
        if m.method in [:statespace, :spectral]
            ure_name = _find_parameter(p_names, string(p_names_k.ure), k, is_multivariate_model)
            if isempty(ure_name)
                @warn "Innovations (ure) for AR1 component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            ure_samples = get_params_matrix(chain, ure_name, t_N_train)
            
            if m.method == :statespace
                for j in 1:n_samples
                    latent_field_train_j = ar1_statespace(
                        rho_samples[j], sigma_samples[j],
                        ure_samples[j, :], t_N_train, noise_val
                    )
                    latent_field_samples[1:t_N_train, j] = latent_field_train_j
                end
            else # :spectral
                U = spec.hyper.U
                L_base = spec.hyper.L
                
                for j in 1:n_samples
                    lambda_vals = (1.0 + rho_samples[j]^2) .+ rho_samples[j] .* L_base
                    diag_D = sigma_samples[j] ./ sqrt.(lambda_vals .+ noise_val)
                    latent_field_samples[1:t_N_train, j] = U * (diag_D .* ure_samples[j, :])
                end
            end

        elseif m.method == :centered
            sre_name = _find_parameter(p_names, string(p_names_k.sre), k, is_multivariate_model)
            if isempty(sre_name)
                @warn "Structured field (sre) for AR1 component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            sre_samples = get_params_matrix(chain, sre_name, t_N_train)
            latent_field_samples[1:t_N_train, :] = sre_samples'

        elseif m.method == :marginalized
            # Exact conditional Gaussian simulation for marginalized GMRF
            y_sigma_name = _find_parameter(p_names, "y_sigma", k, is_multivariate_model)
            y_sigma_samples = if !isempty(y_sigma_name)
                get_params_vector(chain, y_sigma_name, 1)[:, 1]
            else
                fill(1.0, n_samples)
            end
            
            y_vec = M.y_obs isa AbstractMatrix ? M.y_obs[:, k] : M.y_obs
            
            N_t = zeros(Float64, t_N_train)
            S_t = zeros(Float64, t_N_train)
            for i in 1:length(t_idx_train)
                t = t_idx_train[i]
                if 1 <= t <= t_N_train
                    N_t[t] += 1.0
                    S_t[t] += y_vec[i]
                end
            end
            
            for j in 1:n_samples
                rho = rho_samples[j]
                sigma = sigma_samples[j]
                y_sig = y_sigma_samples[j]
                
                diag_val = fill(1.0 + rho^2, t_N_train)
                diag_val[1] = 1.0
                diag_val[t_N_train] = 1.0
                off_diag_val = fill(-rho, t_N_train - 1)
                
                scale = sigma^2 * (1.0 - rho^2 + noise_val)
                inv_sigma_y2 = 1.0 / (y_sig^2 + noise_val)
                
                diag_post = diag_val .+ (N_t .* (inv_sigma_y2 * scale))
                Q_post_scaled = Tridiagonal(off_diag_val, diag_post, off_diag_val)
                
                F = cholesky(Symmetric(Q_post_scaled))
                b = S_t .* inv_sigma_y2
                mu = scale .* (F \ b)
                
                z = randn(t_N_train)
                x_train = mu .+ sqrt(max(scale, 1e-12)) .* (F.U \ z)
                latent_field_samples[1:t_N_train, j] = x_train
            end
        end
        
        # Forecasting step (on CPU)
        if t_N_full > t_N_train
            for j in 1:n_samples
                for t in (t_N_train + 1):t_N_full
                    pred_innov = randn()
                    latent_field_samples[t, j] = rho_samples[j] * latent_field_samples[t-1, j] + pred_innov * sigma_samples[j]
                end
            end
        end
        
        # Indexing on the CPU
        effect_k = latent_field_samples[t_idx_full, :]
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end

"""
    ar1_statespace(rho, sigma, ure, n_latent, noise)

Computes the state-space evolution of a stationary AR(1) process. This is a CPU-only
  implementation.
"""
function ar1_statespace(
    rho, sigma, ure::AbstractVector, n_latent::Int, noise
)
    T_num = promote_type(
        typeof(rho), typeof(sigma), eltype(ure), typeof(noise)
    )
    latent = Vector{T_num}(undef, n_latent)
    if n_latent == 0
        return latent
    end

    if n_latent > 0
        # The denominator is protected from being zero or negative by the `noise` term.
        latent[1] = ure[1] / sqrt(one(T_num) - rho^2 + T_num(noise))
        for t in 2:n_latent
            latent[t] = rho * latent[t-1] + ure[t]
        end
        latent .*= sigma
    end
    
    return latent
end 
