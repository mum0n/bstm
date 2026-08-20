"""
    AR1 <: ComponentModel

A component model for a first-order autoregressive (AR1) process, fundamental for
modeling time series data with serial correlation.

# Version
v1.4.2 (2026-08-19)

# Mathematical Summary
The AR1 process models the value of a latent field \$\\phi_t\$ at time \$t\$ as a
fraction of its value at the previous time step, plus an independent innovation
term \$\\epsilon_t\$:
\$\\phi_t = \\rho \\phi_{t-1} + \\epsilon_t\$, where \$\\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2)\$

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
  - `unconstrained_rho`: A `Distribution` for the prior on the unconstrained
    autocorrelation parameter. Default: `Normal(0, 1.5)`.
  - `sigma`: A `Distribution` for the prior on the innovations' standard deviation.
    Default: `Exponential(1.0)`.
  - `method`: A `Symbol` specifying the computational method (`:statespace`,
    `:spectral`, or `:centered`). Default: `:statespace`.

# Outputs (Parameter Names)
- `unconstrained_rho_<key>`: The unconstrained parameter sampled by Turing.
- `rho_<key>`: The transformed autocorrelation parameter, `tanh(unconstrained_rho_<key>)`.
- `sigma_<key>`: The standard deviation of the AR1 innovations.
- `innovations_<key>`: The latent standard normal innovations driving the process (for `:statespace` and `:spectral`).
- `latent_<key>`: The latent field (for `:centered`).

# Key References
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
"""
struct AR1 <: ComponentModel
    unconstrained_rho::Distribution
    sigma::Distribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:ar1] = AR1
COMPONENT_CONSTRUCTORS[:ar1] = (p, params) -> AR1(
    get(p, :unconstrained_rho, Normal(0, 1.5)),
    get(p, :sigma, Exponential(1.0)),
    get(params, :method, :statespace)
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
            "$(p_names.unconstrained_rho) ~ " *
            "$(_distribution_to_string(m.unconstrained_rho))"
        )
    end

    # The innovations are always per-outcome in multivariate models.
    if m.method in [:statespace, :spectral]
        push!(
            priors_acc,
            "$(p_names.innovations) ~ MvNormal(zeros(T, $(n_latent)), I)"
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
        rho = tanh($(p_names.unconstrained_rho))
        latent_field = ar1_statespace(
            rho, $(p_names.sigma), $(p_names.innovations), $(n_latent), M.noise
        )
        $(eta_target) .+= view(latent_field, M.$(index_var))
    """

    spectral_code = """
        # --- AR1 Component (Spectral): $(key) ---
        rho = tanh($(p_names.unconstrained_rho))
        let
            hyper = spec_registry[:$(key)].hyper
            U = hyper.U
            L_base = hyper.L
            lambda_vals = (one(T) + rho^2) .+ rho .* L_base
            diag_D = $(p_names.sigma) ./ sqrt.(lambda_vals .+ M.noise)
            latent_field = U * (diag_D .* $(p_names.innovations))
            $(eta_target) .+= view(latent_field, M.$(index_var))
        end
    """

    centered_code = """
        # --- AR1 Component (Centered, Didactic): $(key) ---
        rho = tanh($(p_names.unconstrained_rho))
        let
            K = _ar1_covariance_matrix(
                rho, $(p_names.sigma), $(n_latent), M.noise
            )
            $(p_names.latent) ~ MvNormal(zeros(T, $(n_latent)), Symmetric(K))
            $(eta_target) .+= view($(p_names.latent), M.$(index_var))
        end
    """

    if m.method == :statespace
        return statespace_code
    elseif m.method == :spectral
        return spectral_code
    elseif m.method == :centered
        return centered_code
    else
        error(
            "Unsupported method '$(m.method)' for AR1 component. " *
            "Use `:statespace`, `:spectral`, or `:centered`."
        )
    end
end

function get_effects(
    m::AR1, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = size(chain, 1) * FlexiChains.nchains(chain)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    noise_val = get(M, :noise, 1e-6)

    # --- Index Handling: Combine training and prediction sets on CPU ---
    t_idx_train = M.t_idx
    t_idx_full = if !isnothing(PS) && haskey(PS.data, :t_idx)
        vcat(t_idx_train, PS.data.t_idx)
    else
        t_idx_train
    end
    
    t_N_train = M.t_N
    t_N_full = isempty(t_idx_full) ? 0 : maximum(t_idx_full)
    N_total = length(t_idx_full)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop ---
    for k in 1:outcomes_N
        sigma_name = _find_parameter(p_names, string(spec.key), "sigma", k, is_multivariate_model)
        rho_name = _find_parameter(p_names, string(spec.key), "unconstrained_rho", k, is_multivariate_model)
        
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
            innov_name = _find_parameter(p_names, string(spec.key), "innovations", k, is_multivariate_model)
            if isempty(innov_name)
                @warn "Innovations for AR1 component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            innov_samples = get_params_matrix(chain, innov_name, t_N_train)
            
            if m.method == :statespace
                for j in 1:n_samples
                    latent_field_train_j = ar1_statespace(
                        rho_samples[j], sigma_samples[j],
                        innov_samples[j, :], t_N_train, noise_val
                    )
                    latent_field_samples[1:t_N_train, j] = latent_field_train_j
                end
            else # :spectral
                U = spec.hyper.U
                L_base = spec.hyper.L
                
                for j in 1:n_samples
                    lambda_vals = (1.0 + rho_samples[j]^2) .+ rho_samples[j] .* L_base
                    diag_D = sigma_samples[j] ./ sqrt.(lambda_vals .+ noise_val)
                    latent_field_samples[1:t_N_train, j] = U * (diag_D .* innov_samples[j, :])
                end
            end

        elseif m.method == :centered
            latent_name = _find_parameter(p_names, string(spec.key), "latent", k, is_multivariate_model)
            if isempty(latent_name)
                @warn "Latent field for AR1 component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            latent_samples = get_params_matrix(chain, latent_name, t_N_train)
            latent_field_samples[1:t_N_train, :] = latent_samples'
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
    ar1_statespace(rho, sigma, innovations, n_latent, noise)

Computes the state-space evolution of a stationary AR(1) process. This is a CPU-only implementation.
"""
function ar1_statespace(
    rho, sigma, innovations::AbstractVector, n_latent::Int, noise
)
    T_num = promote_type(
        typeof(rho), typeof(sigma), eltype(innovations), typeof(noise)
    )
    latent = Vector{T_num}(undef, n_latent)
    if n_latent == 0
        return latent
    end

    # The stationarity check `abs(rho) >= one(T_num)` is removed as it can cause
    # issues with AD when rho is very close to 1 or -1. The `tanh` transformation
    # in `get_updates` already ensures |rho| < 1 mathematically. Numerical
    # stability at the boundaries is handled by the `noise` term in the denominator.

    if n_latent > 0
        # The denominator is protected from being zero or negative by the `noise` term.
        latent[1] = innovations[1] / sqrt(one(T_num) - rho^2 + T_num(noise))
        for t in 2:n_latent
            latent[t] = rho * latent[t-1] + innovations[t]
        end
        latent .*= sigma
    end
    
    return latent
end 