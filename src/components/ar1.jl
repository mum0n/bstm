"""
    AR1 <: ComponentModel

A component model for a first-order autoregressive (AR1) process, fundamental for
modeling time series data with serial correlation.

# Version
v1.2.7 (2026-08-15)

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

function get_precomputes(m::AR1, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.t_N
    template = build_structure_template(:ar1, n)
    return (Q_template=template.matrix, n_latent=n, U=template.U, L=template.L)
end

function _ar1_covariance_matrix(rho, sigma, n, noise)
    T_num = promote_type(typeof(rho), typeof(sigma), typeof(noise))
    var = sigma^2 / (one(T_num) - rho^2 + T_num(noise))
    
    C = Matrix{T_num}(undef, n, n)
    for i in 1:n, j in 1:n
        C[i, j] = var * rho^abs(i - j)
    end
    return C
end

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

function ar1_statespace(rho, sigma, innov, n_latent, noise)
    T_num = promote_type(typeof(rho), typeof(sigma), eltype(innov), typeof(noise))
    latent = Vector{T_num}(undef, n_latent)
    
    if n_latent > 0
        latent[1] = innov[1] / sqrt(one(T_num) - rho^2 + T_num(noise))
        for t in 2:n_latent
            latent[t] = rho * latent[t-1] + innov[t]
        end
        latent .*= sigma
    end
    
    return latent
end

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
    m::AR1, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    p_names::Vector{String}, spec::NamedTuple, PS::Union{NamedTuple, Nothing},
    N_total::Int, is_multivariate_model::Bool
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    noise_val = get(M, :noise, 1e-6)
    
    t_idx_full = if !isnothing(PS) && haskey(PS, :t_idx)
        vcat(M.t_idx, PS.t_idx)
    else
        M.t_idx
    end
    
    t_N_full = maximum(t_idx_full)

    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_name = _find_parameter(
            p_names, string(p_names_k.sigma), k, is_multivariate_model
        )
        rho_name = _find_parameter(
            p_names, string(p_names_k.unconstrained_rho), k, is_multivariate_model
        )
        innov_name = _find_parameter(
            p_names, string(p_names_k.innovations), k, is_multivariate_model
        )
        latent_name = _find_parameter(
            p_names, string(p_names_k.latent), k, is_multivariate_model
        )

        if isempty(sigma_name) || isempty(rho_name) || 
           (m.method in [:statespace, :spectral] && isempty(innov_name)) ||
           (m.method == :centered && isempty(latent_name))
            @warn "Parameters for AR1 component $(spec.key) (outcome $k, " *
                  "method $(m.method)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        rho_samples = tanh.(get_params_vector(chain, rho_name, 1)[:, 1])
        
        latent_field_samples = zeros(Float64, t_N_full, n_samples)
        
        if m.method in [:statespace, :spectral]
            innov_samples = get_params_matrix(chain, innov_name, M.t_N)
            
            for j in 1:n_samples
                local latent_field_train
                if m.method == :statespace
                    latent_field_train = ar1_statespace(
                        rho_samples[j], sigma_samples[j],
                        innov_samples[j, :], M.t_N, noise_val
                    )
                else # :spectral
                    U = spec.hyper.U
                    L_base = spec.hyper.L
                    lambda_vals = (1.0 + rho_samples[j]^2) .+ rho_samples[j] .* L_base
                    diag_D = sigma_samples[j] ./ sqrt.(lambda_vals .+ M.noise)
                    latent_field_train = U * (diag_D .* innov_samples[j, :])
                end
                latent_field_samples[1:M.t_N, j] = latent_field_train
            end
        elseif m.method == :centered
            latent_samples = get_params_matrix(chain, latent_name, M.t_N)
            latent_field_samples[1:M.t_N, :] = latent_samples'
        end
        
        if t_N_full > M.t_N
            for j in 1:n_samples
                for t in (M.t_N + 1):t_N_full
                    latent_field_samples[t, j] = rho_samples[j] * 
                        latent_field_samples[t-1, j] + randn() * sigma_samples[j]
                end
            end
        end
        
        effect_k = latent_field_samples[t_idx_full, :]
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
