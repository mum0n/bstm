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
    unconstrained_rho1::Distribution
    unconstrained_rho2::Distribution
    sigma::Distribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:ar2] = AR2
COMPONENT_CONSTRUCTORS[:ar2] = (p, params) -> AR2(
    get(p, :unconstrained_rho1, Normal(0, 1.5)),
    get(p, :unconstrained_rho2, Normal(0, 1.5)),
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
        push!(priors_acc, "$(p_names.unconstrained_rho1) ~ " *
                          "$(_distribution_to_string(m.unconstrained_rho1))")
        push!(priors_acc, "$(p_names.unconstrained_rho2) ~ " *
                          "$(_distribution_to_string(m.unconstrained_rho2))")
    end

    # For the :statespace method, we define priors on the innovations.
    # For the :centered method, the latent field is sampled directly in get_updates,
    # so no prior is needed here for the latent variable itself.
    if m.method == :statespace
        push!(priors_acc, "$(p_names.innovations) ~ MvNormal(zeros(T, $(n_latent)), I)")
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
        pi1 = tanh($(p_names.unconstrained_rho1))
        pi2 = tanh($(p_names.unconstrained_rho2))
        rho1 = pi1 * (1 - pi2)
        rho2 = pi2
        
        latent_field = ar2_statespace(
            rho1, rho2, $(p_names.sigma), $(p_names.innovations), $(n_latent), M.noise
        )
        $(eta_target) .+= view(latent_field, M.$(index_var))
    """

    centered_code = """
        # --- AR2 Component (Centered, Didactic): $(spec.key) ---
        pi1 = tanh($(p_names.unconstrained_rho1))
        pi2 = tanh($(p_names.unconstrained_rho2))
        rho1 = pi1 * (1 - pi2)
        rho2 = pi2
        
        let
            K = _ar2_covariance_matrix(
                rho1, rho2, $(p_names.sigma), $(n_latent), M.noise
            )
            $(p_names.latent) ~ MvNormal(zeros(T, $(n_latent)), Symmetric(K))
            $(eta_target) .+= view($(p_names.latent), M.$(index_var))
        end
    """

    if m.method == :statespace
        return statespace_code
    elseif m.method == :centered
        return centered_code
    else
        error("Unsupported method '$(m.method)' for AR2 component. Use `:statespace` " *
              "or `:centered`.")
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
    ar2_statespace(rho1, rho2, sigma, innovations, n_latent, noise)

Computes the state-space evolution of a stationary AR(2) process. This version is CPU-only.
"""
function ar2_statespace(
    rho1, rho2, sigma, innovations::AbstractVector, n_latent::Int, noise
)
    T_num = promote_type(
        typeof(rho1), typeof(rho2), typeof(sigma), eltype(innovations), typeof(noise)
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
        latent[1:2] = L_12 * view(innovations, 1:2)
    elseif n_latent == 1
        latent[1] = sqrt(gamma_0) * innovations[1]
    end

    # Evolve the process for the remaining time steps
    for t in 3:n_latent
        latent[t] = rho1 * latent[t-1] + rho2 * latent[t-2] + innovations[t] * sigma
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
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(names(DataFrame(chain)))
    
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
        rho1_name = _find_parameter(p_names, string(p_names_k.unconstrained_rho1), k, is_multivariate_model)
        rho2_name = _find_parameter(p_names, string(p_names_k.unconstrained_rho2), k, is_multivariate_model)
        
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
            innov_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            if isempty(innov_name)
                @warn "Innovations for AR2 component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total_obs, n_samples))
                continue
            end
            innov_samples = get_params_vector(chain, innov_name, t_N_train)
            
            # Reconstruction for training period
            for j in 1:n_samples
                latent_field_train_j = ar2_statespace(
                    rho1_samples[j], rho2_samples[j], sigma_samples[j],
                    innov_samples[j, :], t_N_train, noise_val
                )
                latent_field_samples[1:t_N_train, j] = latent_field_train_j
            end

        elseif m.method == :centered
            latent_name = _find_parameter(p_names, string(p_names_k.latent), k, is_multivariate_model)
            if isempty(latent_name)
                @warn "Latent field for AR2 component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total_obs, n_samples))
                continue
            end
            latent_samples = get_params_vector(chain, latent_name, t_N_train)
            latent_field_samples[1:t_N_train, :] = latent_samples'
            if t_N_full > t_N_train
                @warn "Forecasting for the AR2 component with the ':centered' method is not supported. Returning zeros for prediction time steps."
            end
        end
        
        # Forecasting step for the statespace method
        if m.method == :statespace && t_N_full > t_N_train
            for j in 1:n_samples
                for t in (t_N_train + 1):t_N_full
                    pred_innov = randn()
                    latent_field_samples[t, j] = 
                        rho1_samples[j] * latent_field_samples[t-1, j] +
                        rho2_samples[j] * latent_field_samples[t-2, j] +
                        pred_innov * sigma_samples[j]
                end
            end
        end
        
        # Indexing to get final effect
        effect_k = latent_field_samples[t_idx_full, :]
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
