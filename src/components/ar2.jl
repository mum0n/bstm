"""
    AR2 <: ComponentModel

A component model for a second-order autoregressive (AR2) process, fundamental for
modeling time series data with serial correlation and pseudo-periodic behavior.

# Version
v1.4.0 (2026-08-19)

# Mathematical Summary
The AR2 process models the value of a latent field \$\\phi_t\$ at time \$t\$ as a linear
combination of its values at the two previous time steps, plus an independent
innovation term \$\\epsilon_t\$:
\$\\phi_t = \\rho_1 \\phi_{t-1} + \\rho_2 \\phi_{t-2} + \\epsilon_t\$, where \$\\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2)\$

To ensure stationarity, the parameters \$(\\rho_1, \\rho_2)\$ must lie within a
triangular region defined by:
1. \$\\rho_1 + \\rho_2 < 1\$
2. \$\\rho_2 - \\rho_1 < 1\$
3. \$|\\rho_2| < 1\$

This implementation enforces this constraint by reparameterizing the model in terms
of its partial autocorrelations (\$\\pi_1, \\pi_2\$), which are constrained to be in
`(-1, 1)`. We sample unconstrained parameters `unconstrained_rho1` and
`unconstrained_rho2` and transform them using `tanh`. The AR coefficients are then
recovered via the stable transformation:
\$\\rho_1 = \\pi_1 (1 - \\pi_2)\$
\$\\rho_2 = \\pi_2\$

This ensures that the sampled `rho1` and `rho2` always correspond to a stationary process,
improving MCMC efficiency and stability.

# Computational Methods
This component supports multiple numerical methods for temporal evolution,
controlled by the `random()` call:
1.  **:statespace** (Default, AD-friendly): Implemented using a numerically stable
    state-space formulation. This is the recommended method for most applications.
2.  **:centered** (Didactic, Not AD-friendly): Explicitly constructs the dense
    Toeplitz covariance matrix and samples the latent field directly. This is a
    didactic alternative that is less efficient for MCMC and not AD-compatible.

# Inputs
- **Required**:
  - A temporal index variable (e.g., `year`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `unconstrained_rho1`: A `Distribution` for the first unconstrained partial
    autocorrelation coefficient. Default: `Normal(0, 1.5)`.
  - `unconstrained_rho2`: A `Distribution` for the second unconstrained partial
    autocorrelation coefficient. Default: `Normal(0, 1.5)`.
  - `sigma`: A `Distribution` for the prior on the innovations' standard deviation.
    Default: `Exponential(1.0)`.
  - `method`: A `Symbol` specifying the computational method (`:statespace` or
    `:centered`). Default: `:statespace`.

# Outputs (Parameter Names)
- `unconstrained_rho1_<key>`: The first unconstrained parameter sampled by Turing.
- `unconstrained_rho2_<key>`: The second unconstrained parameter sampled by Turing.
- `rho1_<key>`: The first transformed AR coefficient.
- `rho2_<key>`: The second transformed AR coefficient.
- `sigma_<key>`: The standard deviation of the AR2 innovations.
- `innovations_<key>`: The latent standard normal innovations driving the process (for `:statespace`).
- `latent_<key>`: The latent field (for `:centered`).

# Key References
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
"""
struct AR2 <: ComponentModel
    rho1_unconstrained::Distribution
    rho2_unconstrained::Distribution
    sigma::Distribution
    method::Symbol
end

Base.getproperty(m::AR2, s::Symbol) = (
    s === :unconstrained_rho1 ? getfield(m, :rho1_unconstrained) :
    s === :unconstrained_rho2 ? getfield(m, :rho2_unconstrained) :
    getfield(m, s)
)

COMPONENT_TYPE_REGISTRY[:ar2] = AR2
COMPONENT_CONSTRUCTORS[:ar2] = (p, params) -> AR2(
    get(p, :rho1_unconstrained, get(p, :unconstrained_rho1, Normal(0, 1.5))),
    get(p, :rho2_unconstrained, get(p, :unconstrained_rho2, Normal(0, 1.5))),
    get(p, :sigma, Exponential(1.0)),
    get(params, :method, :statespace)
)
MODEL_TO_STRUCTURE_MAP[:ar2] = :temporal

function get_precomputes(m::AR2, M::NamedTuple, mod_data::Dict)::NamedTuple
    t_N = get(M, :t_N, 0)
    if t_N == 0
        @warn "Could not determine number of time steps for AR2 component " *
              "'$(mod_data[:key])'. The component will have no effect."
    end
    return (n_latent=t_N,)
end
 
"""
    get_priors(m::AR2, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates Turing code for the priors of the `AR2` component.
"""
function get_priors(
    m::AR2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent

    is_multivariate = arch == "multivariate"
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]

    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")
        push!(priors_acc, "$(p_names.rho1_unconstrained) ~ " *
                          "$(_distribution_to_string(m.rho1_unconstrained))")
        push!(priors_acc, "$(p_names.rho2_unconstrained) ~ " *
                          "$(_distribution_to_string(m.rho2_unconstrained))")
    end

    # For the :statespace method, we define priors on the innovations (ure).
    if m.method == :statespace
        push!(priors_acc, "$(p_names.ure) ~ MvNormal(zeros(T, $(n_latent)), I)")
    end
    
    return join(priors_acc, "\n    ")
end

"""
    get_updates(m::AR2, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates Turing code to construct the `AR2` effect, dispatching on the chosen method.
"""
function get_updates(
    m::AR2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx"
    n_latent = spec.hyper.n_latent
    key = spec.key

    statespace_code = """
        # --- AR2 Component (State-Space, Stationarity-Enforced): $(spec.key) ---
        pi1 = tanh($(p_names.rho1_unconstrained))
        pi2 = tanh($(p_names.rho2_unconstrained))
        rho1 = pi1 * (1 - pi2)
        rho2 = pi2
        
        $(p_names.sre) = ar2_statespace(
            rho1, rho2, $(p_names.sigma), $(p_names.ure), $(n_latent), M.noise
        )
        $(eta_target) .+= view($(p_names.sre), M.$(index_var))
    """

    centered_code = """
        # --- AR2 Component (Centered, Didactic): $(spec.key) ---
        pi1 = tanh($(p_names.rho1_unconstrained))
        pi2 = tanh($(p_names.rho2_unconstrained))
        rho1 = pi1 * (1 - pi2)
        rho2 = pi2
        
        let
            K = _ar2_covariance_matrix(
                rho1, rho2, $(p_names.sigma), $(n_latent), M.noise
            )
            $(p_names.sre) ~ MvNormal(zeros(T, $(n_latent)), Symmetric(K))
            $(eta_target) .+= view($(p_names.sre), M.$(index_var))
        end
    """

    marginalized_code = """
        # --- AR2 Component (Marginalized): $(spec.key) ---
        let
            pi1 = tanh($(p_names.rho1_unconstrained))
            pi2 = tanh($(p_names.rho2_unconstrained))
            rho1 = pi1 * (1 - pi2)
            rho2 = pi2
            y_residual = M.y_obs .- $(eta_target)
            log_lik_marginalized_$(spec.key) = _ar2_log_marginal_likelihood(
                y_residual,
                M.$(index_var),
                $(n_latent),
                rho1,
                rho2,
                $(p_names.sigma),
                y_sigma,
                M.noise
            )
            Turing.@addlogprob! log_lik_marginalized_$(spec.key)
        end
    """

    if m.method == :statespace
        return statespace_code
    elseif m.method == :centered
        return centered_code
    elseif m.method == :marginalized
        return marginalized_code
    else
        error("Unsupported method '$(m.method)' for AR2 component. Use `:statespace`, `:centered`, or `:marginalized`.")
    end
end

"""
    _ar2_covariance_matrix(rho1, rho2, sigma, n, noise)

Helper function to construct the dense Toeplitz covariance matrix for a stationary
AR(2) process. Used by the `:centered` method. This version is CPU-only.
"""
function _ar2_covariance_matrix(rho1, rho2, sigma, n, noise)
    T_num = promote_type(typeof(rho1), typeof(rho2), typeof(sigma), typeof(noise))
    
    if rho1 + rho2 >= one(T_num) || rho2 - rho1 >= one(T_num) || abs(rho2) >= one(T_num)
        # Return a high-variance diagonal matrix to penalize non-stationary parameters
        return Diagonal(fill(T_num(1e12), n))
    end

    var_innov = sigma^2
    gamma_0 = var_innov * (one(T_num) - rho2) / 
              ((one(T_num) + rho2) * ((one(T_num) - rho2)^2 - rho1^2) + T_num(noise))
    gamma_1 = (rho1 / (one(T_num) - rho2)) * gamma_0

    # Calculate the first row of the Toeplitz matrix (the autocovariance function)
    acf = Vector{T_num}(undef, n)
    if n > 0; acf[1] = gamma_0; end
    if n > 1; acf[2] = gamma_1; end
    for i in 3:n
        acf[i] = rho1 * acf[i-1] + rho2 * acf[i-2]
    end

    # Construct the Toeplitz matrix from the ACF.
    C = Matrix{T_num}(undef, n, n)
    for i in 1:n
        for j in 1:n
            C[i, j] = acf[abs(i - j) + 1]
        end
    end
    
    return Symmetric(C)
end

"""
    _ar2_log_marginal_likelihood(y_residual, t_idx, t_N, rho1, rho2, sigma, y_sigma, noise=1e-6)

Computes the exact log marginal likelihood for an AR(2) process integrated out analytically.
"""
function _ar2_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    t_idx::AbstractVector{Int},
    t_N::Int,
    rho1::T,
    rho2::T,
    sigma::T,
    y_sigma::T,
    noise::Real=1e-6
) where {T}
    N = length(y_residual)
    T_num = promote_type(T, typeof(noise))
    
    if rho1 + rho2 >= one(T_num) || rho2 - rho1 >= one(T_num) || abs(rho2) >= one(T_num)
        return -T_num(1e12)
    end
    
    K = _ar2_covariance_matrix(rho1, rho2, sigma, t_N, noise)
    
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
    
    # Q_prior = inv(K)
    F_K = cholesky(Symmetric(K + T_num(noise) * I))
    Q_prior = inv(F_K)
    
    # Q_post = Q_prior + diag(N_t * inv_sigma_y2)
    Q_post = Matrix(Q_prior)
    for t in 1:t_N
        Q_post[t, t] += N_t[t] * inv_sigma_y2
    end
    
    F_post = cholesky(Symmetric(Q_post))
    
    log_det_diff = - 2 * sum(log.(diag(F_K.U))) - 2 * sum(log.(diag(F_post.U)))
    
    b = S_t .* inv_sigma_y2
    v = F_post.L \ b
    quad_term = dot(v, v)
    
    log_lik = - (N / 2) * log(2 * T_num(pi) * (y_sigma^2 + T_num(noise))) -
              (inv_sigma_y2 / 2) * dot(y_residual, y_residual) +
              (1 / 2) * log_det_diff +
              (1 / 2) * quad_term
              
    return log_lik
end

"""
    ar2_statespace(rho1, rho2, sigma, ure, n_latent, noise)

Computes the state-space evolution of a stationary AR(2) process. This version is CPU-only.
"""
function ar2_statespace(
    rho1, rho2, sigma, ure::AbstractVector, n_latent::Int, noise
)
    T_num = promote_type(
        typeof(rho1), typeof(rho2), typeof(sigma), eltype(ure), typeof(noise)
    )
    latent = Vector{T_num}(undef, n_latent)
    if n_latent == 0
        return latent
    end

    # Check for stationarity; return NaN if parameters are invalid
    if rho1 + rho2 >= one(T_num) || rho2 - rho1 >= one(T_num) || abs(rho2) >= one(T_num)
        fill!(latent, T_num(NaN))
        return latent
    end

    var_innov = sigma^2
    gamma_0 = var_innov * (one(T_num) - rho2) / 
              ((one(T_num) + rho2) * ((one(T_num) - rho2)^2 - rho1^2) + T_num(noise))
    gamma_1 = (rho1 / (one(T_num) - rho2)) * gamma_0

    # Ensure the small 2x2 covariance matrix is created on the CPU
    cov_12 = Matrix{T_num}(undef, 2, 2)
    cov_12[1,1] = gamma_0; cov_12[2,2] = gamma_0;
    cov_12[1,2] = gamma_1; cov_12[2,1] = gamma_1;
    
    L_12 = cholesky(Symmetric(cov_12 + T_num(noise) * I)).L

    # Initialize the first two states
    if n_latent >= 2
        latent[1:2] = L_12 * view(ure, 1:2)
    elseif n_latent == 1
        latent[1] = sqrt(gamma_0) * ure[1]
    end

    # Evolve the process for the remaining time steps
    for t in 3:n_latent
        latent[t] = rho1 * latent[t-1] + rho2 * latent[t-2] + ure[t] * sigma
    end

    return latent
end

"""
    get_effects(m::AR2, chain, spec, M, PS)

Reconstructs the `AR2` component's effect from posterior samples. This version is CPU-only
and uses modern chain accessors.
"""
function get_effects(
    m::AR2, chain, spec::NamedTuple, M::NamedTuple,
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
    
    noise_val = get(M, :noise, 1e-6)
    
    # --- Index Handling: Combine training and prediction sets ---
    t_idx_train = M.t_idx
    t_idx_full = if !isnothing(PS) && haskey(PS.data, :t_idx)
        vcat(t_idx_train, get(PS.data, :t_idx, []))
    else
        t_idx_train
    end
    
    t_N_train = M.t_N
    t_N_full = isempty(t_idx_full) ? 0 : maximum(t_idx_full)
    N_total_obs = length(t_idx_full)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        rho1_name = _find_parameter(p_names, string(p_names_k.rho1_unconstrained), k, is_multivariate_model)
        rho2_name = _find_parameter(p_names, string(p_names_k.rho2_unconstrained), k, is_multivariate_model)
        
        if isempty(sigma_name) || isempty(rho1_name) || isempty(rho2_name)
            @warn "Base parameters for AR2 component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total_obs, n_samples))
            continue
        end

        # Extract posterior samples
        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        pi1_samples = tanh.(get_params_vector(chain, rho1_name, 1)[:, 1])
        pi2_samples = tanh.(get_params_vector(chain, rho2_name, 1)[:, 1])
        rho1_samples = pi1_samples .* (1 .- pi2_samples)
        rho2_samples = pi2_samples
        
        # Initialize output matrix for the full latent field
        latent_field_samples = zeros(Float64, t_N_full, n_samples)
        
        if m.method == :statespace
            ure_name = _find_parameter(p_names, string(p_names_k.ure), k, is_multivariate_model)
            if isempty(ure_name)
                @warn "Innovations (ure) for AR2 component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total_obs, n_samples))
                continue
            end
            ure_samples = get_params_vector(chain, ure_name, t_N_train)
            
            # Reconstruction for training period
            for j in 1:n_samples
                latent_field_train_j = ar2_statespace(
                    rho1_samples[j], rho2_samples[j], sigma_samples[j],
                    ure_samples[j, :], t_N_train, noise_val
                )
                latent_field_samples[1:t_N_train, j] = latent_field_train_j
            end

        elseif m.method == :centered
            sre_name = _find_parameter(p_names, string(p_names_k.sre), k, is_multivariate_model)
            if isempty(sre_name)
                @warn "Structured field (sre) for AR2 component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total_obs, n_samples))
                continue
            end
            sre_samples = get_params_vector(chain, sre_name, t_N_train)
            latent_field_samples[1:t_N_train, :] = sre_samples'
            if t_N_full > t_N_train
                @warn "Forecasting for the AR2 component with the ':centered' method is not supported. Returning zeros for prediction time steps."
            end

        elseif m.method == :marginalized
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
                r1 = rho1_samples[j]
                r2 = rho2_samples[j]
                sig = sigma_samples[j]
                y_sig = y_sigma_samples[j]
                
                K = _ar2_covariance_matrix(r1, r2, sig, t_N_train, noise_val)
                F_K = cholesky(Symmetric(K + noise_val * I))
                Q_prior = inv(F_K)
                
                inv_sigma_y2 = 1.0 / (y_sig^2 + noise_val)
                Q_post = Matrix(Q_prior)
                for t in 1:t_N_train
                    Q_post[t, t] += N_t[t] * inv_sigma_y2
                end
                
                F_post = cholesky(Symmetric(Q_post))
                b = S_t .* inv_sigma_y2
                mu = F_post \ b
                
                z = randn(t_N_train)
                x_train = mu + F_post.U \ z
                latent_field_samples[1:t_N_train, j] = x_train
            end
        end
        
        # Forecasting step for statespace and marginalized methods
        if (m.method in [:statespace, :marginalized]) && t_N_full > t_N_train
            for j in 1:n_samples
                for t in (t_N_train + 1):t_N_full
                    pred_innov = randn()
                    val_t1 = t - 1 >= 1 ? latent_field_samples[t-1, j] : 0.0
                    val_t2 = t - 2 >= 1 ? latent_field_samples[t-2, j] : 0.0
                    latent_field_samples[t, j] = rho1_samples[j] * val_t1 + rho2_samples[j] * val_t2 + pred_innov * sigma_samples[j]
                end
            end
        end
        
        # Indexing to get final effect
        effect_k = latent_field_samples[t_idx_full, :]
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
