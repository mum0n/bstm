"""
    AR1 <: ComponentModel

A component model for a first-order autoregressive (AR1) process, fundamental for
modeling time series data with serial correlation.

# Version
v1.3.0 (2026-08-17)

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
    to_device = M.to_device
    template = build_structure_template(:ar1, n)
    return (
        Q_template=to_device(template.matrix), 
        n_latent=n, 
        U=to_device(template.U), 
        L=to_device(template.L)
    )
end

function _ar1_covariance_matrix(rho, sigma, n, noise)
    T_num = promote_type(typeof(rho), typeof(sigma), typeof(noise))
    var = sigma^2 / (one(T_num) - rho^2 + T_num(noise))
    
    # Vectorized approach for creating the Toeplitz matrix
    time_diffs = abs.((1:n) .- (1:n)')
    C = var .* (rho .^ time_diffs)
    
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



function ar1_statespace(rho, sigma, innov, n_latent, noise)
    T_num = promote_type(typeof(rho), typeof(sigma), eltype(innov), typeof(noise))
    latent = similar(innov, T_num, n_latent) # Use similar for device awareness
    
    if n_latent > 0
        latent[1] = innov[1] / sqrt(one(T_num) - rho^2 + T_num(noise))
        for t in 2:n_latent
            latent[t] = rho * latent[t-1] + innov[t]
        end
        latent .*= sigma
    end
    
    return latent
end

function get_effects(
    m::AR1, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device
    noise_val = get(M, :noise, 1e-6)

    # --- Index Handling: Combine training and prediction sets ---
    t_idx_train_cpu = Array(M.t_idx)
    t_idx_full_cpu = if !isnothing(PS) && haskey(PS.data, :t_idx)
        vcat(t_idx_train_cpu, get(PS.data, :t_idx, []))
    else
        t_idx_train_cpu
    end
    
    t_N_train = M.t_N
    t_N_full = isempty(t_idx_full_cpu) ? 0 : maximum(t_idx_full_cpu)
    N_total = length(t_idx_full_cpu)
    t_idx_full_device = to_device(t_idx_full_cpu)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        rho_name = _find_parameter(p_names, string(p_names_k.unconstrained_rho), k, is_multivariate_model)
        
        if isempty(sigma_name) || isempty(rho_name)
            @warn "Base parameters for AR1 component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        rho_samples_cpu = tanh.(get_params_vector(chain, rho_name, 1)[:, 1])
        
        latent_field_samples_device = to_device(zeros(Float64, t_N_full, n_samples))
        
        if m.method in [:statespace, :spectral]
            innov_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            if isempty(innov_name)
                @warn "Innovations for AR1 component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            innov_samples_cpu = get_params_matrix(chain, innov_name, t_N_train)
            innov_samples_device = to_device(innov_samples_cpu') # Transpose to [n_latent x n_samples]
            
            if m.method == :statespace
                for j in 1:n_samples
                    latent_field_train_j = ar1_statespace(
                        rho_samples_cpu[j], sigma_samples_cpu[j],
                        view(innov_samples_device, :, j), t_N_train, noise_val
                    )
                    latent_field_samples_device[1:t_N_train, j] = latent_field_train_j
                end
            else # :spectral
                U_device = spec.hyper.U
                L_base_device = spec.hyper.L
                rho_samples_device = to_device(rho_samples_cpu)
                sigma_samples_device = to_device(sigma_samples_cpu)
                
                lambda_vals = (1.0 .+ rho_samples_device'.^2) .+ rho_samples_device' .* L_base_device
                diag_D = sigma_samples_device' ./ sqrt.(lambda_vals .+ noise_val)
                latent_field_train_device = U_device * (diag_D .* innov_samples_device)
                latent_field_samples_device[1:t_N_train, :] = latent_field_train_device
            end

        elseif m.method == :centered
            latent_name = _find_parameter(p_names, string(p_names_k.latent), k, is_multivariate_model)
            if isempty(latent_name)
                @warn "Latent field for AR1 component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            latent_samples_cpu = get_params_matrix(chain, latent_name, t_N_train)
            latent_field_samples_device[1:t_N_train, :] = to_device(latent_samples_cpu')
        end
        
        # Forecasting step (vectorized over samples)
        if t_N_full > t_N_train
            rho_samples_device = to_device(rho_samples_cpu)
            sigma_samples_device = to_device(sigma_samples_cpu)
            
            for t in (t_N_train + 1):t_N_full
                pred_innov_device = to_device(randn(Float32, n_samples))
                latent_field_samples_device[t, :] = rho_samples_device' .* latent_field_samples_device[t-1, :] .+ pred_innov_device' .* sigma_samples_device'
            end
        end
        
        # Indexing on the device and moving the final result to CPU
        effect_k_device = latent_field_samples_device[t_idx_full_device, :]
        push!(structured_effects, Array(effect_k_device))
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end


