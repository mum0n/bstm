"""
    AR1 <: ComponentModel

A component model for a first-order autoregressive (AR1) process, which is fundamental for
modeling time series data with serial correlation.

# Version
v1.1.3 (2026-08-10)

# Mathematical Summary
The AR1 process models the value of a latent field \$\\phi\$ at time \$t\$ as a fraction of its
value at the previous time step, plus an independent innovation term \$\\epsilon_t\$.
The model is defined as:
\$\\phi_t = \\rho \\phi_{t-1} + \\epsilon_t\$, where \$\\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2)\$

This component supports three computational methods:
1.  **:statespace** (Default): Implemented using a numerically stable state-space formulation. 
    The autocorrelation parameter \$\\rho\$ is constrained to `(-1, 1)` via a `tanh` transformation 
    to ensure stationarity. This is the recommended method for most applications.
2.  **:spectral**: Implemented using a spectral decomposition of the AR(1) precision matrix. This 
    method is fully differentiable and can be advantageous for Hamiltonian Monte Carlo samplers.
3.  **:centered**: Explicitly constructs the dense Toeplitz covariance matrix and samples the
    latent field directly. This is a didactic alternative that can be less efficient for MCMC.

# Fields
- `rho::Distribution`: The prior distribution for the temporal autocorrelation parameter `rho`.
- `sigma::Distribution`: The prior distribution for the standard deviation of the AR1 innovations.
- `method::Symbol`: The computational method, one of `:statespace`, `:spectral`, or `:centered`.
"""
struct AR1 <: ComponentModel
    rho::Distribution
    sigma::Distribution
    method::Symbol
end


COMPONENT_TYPE_REGISTRY[:ar1] = AR1
COMPONENT_CONSTRUCTORS[:ar1] = (p, params) -> AR1(
    p.rho, p.sigma, get(params, :method, :statespace)
)
 
# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[:ar1] = :temporal

"""
    get_datastructures!(m_type::Type{<:AR1}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `AR1` component. It ensures a time index
variable is provided and sets up the temporal context (`t_idx`, `t_N`) in the
main model configuration `M`.

# Assumptions
- The input data column specified in the `random()` call contains discrete, integer-like
  indices representing time units (e.g., year numbers).
"""
function get_datastructures!(m_type::Type{<:AR1}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]
    if isempty(variables)
        error("The AR1 model requires a time index variable, e.g., `random(year, model=:ar1)`.")
    end
    t_var_sym = Symbol(variables[1])
    if !hasproperty(M[:data], t_var_sym)
        error("Time index variable ':$t_var_sym' for AR1 model not found in data.")
    end
    
    tu_meta = assign_time_units(M[:data][!, t_var_sym])
    M[:t_idx] = tu_meta.idx
    M[:t_N] = tu_meta.N_cat
    M[:t_idx_var] = t_var_sym
    
    return true
end

"""
    get_precomputes(m::AR1, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the AR1 structure matrix and its spectral decomposition (`U`, `L`) for use
by the `:spectral` method.
"""
function get_precomputes(m::AR1, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.t_N
    template = build_structure_template(:ar1, n)
    return (Q_template=template.matrix, n_latent=n, U=template.U, L=template.L)
end

"""
    _ar1_covariance_matrix(rho, sigma, n, noise)

Helper function to construct the dense Toeplitz covariance matrix for a stationary
AR(1) process. Used by the `:centered` method.
"""
function _ar1_covariance_matrix(rho, sigma, n, noise)
    T_num = promote_type(typeof(rho), typeof(sigma), typeof(noise))
    var = sigma^2 / (one(T_num) - rho^2 + T_num(noise))
    
    C = Matrix{T_num}(undef, n, n)
    for i in 1:n, j in 1:n
        C[i, j] = var * rho^abs(i - j)
    end
    return C
end

"""
    get_priors(m::AR1, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `sigma` and `rho`. For non-centered methods (`:statespace`,
`:spectral`), it also defines a prior for the `innov` innovations.
"""
function get_priors(
    m::AR1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
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
        push!(priors_acc, "$(p_names.rho)_raw ~ Normal(0, 1.5)")
    end

    if m.method in [:statespace, :spectral]
        push!(priors_acc, "$(p_names.innov) ~ MvNormal(zeros($(n_latent)), I)")
    end
    
    return join(priors_acc, "\n    ")
end


"""
    ar1_statespace(rho, sigma, innov, n_latent, noise)

Computes the state-space evolution of an AR(1) process in a type-stable manner.

This function is designed to be compatible with Automatic Differentiation by
inferring the numeric type (`T_num`) from its arguments. This ensures that if
parameters like `rho` or `sigma` are `Dual` numbers, the entire calculation
is performed with `Dual` numbers, avoiding type errors.
"""
function ar1_statespace(rho, sigma, innov, n_latent, noise)
    # Promote the types of all numeric inputs to ensure type stability.
    T_num = promote_type(typeof(rho), typeof(sigma), eltype(innov), typeof(noise))
    latent = Vector{T_num}(undef, n_latent)
    
    if n_latent > 0
        # Initialize the first state using the stationary variance of the AR(1) process.
        # `one(T_num)` and `T_num(noise)` ensure all terms have the correct type.
        latent[1] = innov[1] / sqrt(one(T_num) - rho^2 + T_num(noise))
        
        # Evolve the process for subsequent time steps.
        for t in 2:n_latent
            latent[t] = rho * latent[t-1] + innov[t]
        end
        
        # Scale the entire latent field by the standard deviation.
        latent .*= sigma
    end
    
    return latent
end



"""
    get_updates(m::AR1, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code to construct the `AR1` effect, dispatching on the chosen method.
"""
function get_updates(
    m::AR1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx"
    n_latent = spec.hyper.n_latent

    statespace_code = """
        # --- AR1 Component (State-Space): $(spec.key) ---
        $(p_names.rho) = tanh($(p_names.rho)_raw)
        local latent_field = ar1_statespace($(p_names.rho), $(p_names.sigma), $(p_names.innov), $(n_latent), M.noise)
        $(eta_target) .+= view(latent_field, M.$(index_var))
    """

    spectral_code = """
        # --- AR1 Component (Spectral): $(spec.key) ---
        $(p_names.rho) = tanh($(p_names.rho)_raw)
        let
            local U = spec_registry[:$(spec.key)].hyper.U
            local L_base = spec_registry[:$(spec.key)].hyper.L
            local lambda_vals = (1.0 + $(p_names.rho)^2) .+ $(p_names.rho) .* L_base
            local diag_D = $(p_names.sigma) ./ sqrt.(lambda_vals .+ M.noise)
            local latent_field = U * (diag_D .* $(p_names.innov))
            $(eta_target) .+= view(latent_field, M.$(index_var))
        end
    """

    centered_code = """
        # --- AR1 Component (Centered, Didactic): $(spec.key) ---
        $(p_names.rho) = tanh($(p_names.rho)_raw)
        let
            local K = _ar1_covariance_matrix($(p_names.rho), $(p_names.sigma), $(n_latent), M.noise)
            $(p_names.latent) ~ MvNormal(zeros($(n_latent)), Symmetric(K))
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
        error("Unsupported method '$(m.method)' for AR1 component.")
    end
end

"""
    get_effects(m::AR1, chain, M::NamedTuple, ...)::NamedTuple

Reconstructs the `AR1` component's effect from posterior samples, dispatching on the method
used during model fitting.
"""
function get_effects(
    m::AR1, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    noise_val = get(M, :noise, 1e-6)
    p_names_vec = string.(FlexiChains.parameters(chain))
    n_latent = spec.hyper.n_latent

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_name = _find_parameter(p_names_vec, string(spec.key), "sigma", k)
        rho_raw_name = _find_parameter(p_names_vec, string(spec.key), "rho_raw", k)
        
        if isempty(sigma_name) || isempty(rho_raw_name)
            @warn "Parameters for AR1 component $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_latent, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        rho_raw_samples = get_params_vector(chain, rho_raw_name, 1)[:, 1]
        rho_samples = tanh.(rho_raw_samples)
        
        temporal_effect_k = zeros(Float64, n_latent, n_samples)
        
        if m.method in [:statespace, :spectral]
            innov_name = _find_parameter(p_names_vec, string(spec.key), "innov", k)
            if isempty(innov_name)
                @warn "Innovations for AR1 component $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, n_latent, n_samples))
                continue
            end
            innovations_samples = get_params_vector(chain, innov_name, n_latent)

            if m.method == :spectral
                U = spec.hyper.U
                L_base = spec.hyper.L
                for j in 1:n_samples
                    lambda_vals = (1.0 + rho_samples[j]^2) .+ rho_samples[j] .* L_base
                    diag_D = sigma_samples[j] ./ sqrt.(lambda_vals .+ noise_val)
                    temporal_effect_k[:, j] = U * (diag_D .* innovations_samples[j, :])
                end
            else # :statespace
                for j in 1:n_samples
                    temporal_effect_k[:, j] = ar1_statespace(rho_samples[j], sigma_samples[j], innovations_samples[j, :], n_latent, noise_val)
                end
            end
        else # :centered
            latent_name = string(p_names.latent)
            latent_samples = get_params_vector(chain, latent_name, n_latent)
            temporal_effect_k = latent_samples'
        end
        push!(structured_effects, temporal_effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
