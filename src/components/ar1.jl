"""
    AR1 <: ComponentModel

A component model for a first-order autoregressive (AR1) process, which is fundamental for
modeling time series data with serial correlation.

# Version
v1.1.2 (2026-08-08)

# Mathematical Summary
The AR1 process models the value of a latent field \$\\phi\$ at time \$t\$ as a fraction of its
value at the previous time step, plus an independent innovation term \$\\epsilon_t\$.
The model is defined as:
\$\\phi_t = \\rho \\phi_{t-1} + \\epsilon_t\$, where \$\\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2)\$

This component supports two computational methods:
1.  **:statespace** (Default): Implemented using a numerically stable state-space formulation. 
    The autocorrelation parameter \$\\rho\$ is constrained to `(-1, 1)` via a `tanh` transformation 
    to ensure stationarity. This is the recommended method for most applications.
2.  **:spectral**: Implemented using a spectral decomposition of the AR(1) precision matrix. This 
    method is fully differentiable and can be advantageous for Hamiltonian Monte Carlo samplers, 
    but may be less stable if `rho` approaches the boundaries of `(-1, 1)`.

# Assumptions
- The temporal process is stationary (i.e., `|rho| < 1`).
- The innovations are Gaussian and independent over time.
- The effect is additive on the scale of the linear predictor.

# Best Use Case
Modeling temporal dependencies where the influence of past events decays geometrically over time. It is
a standard choice for capturing short-term persistence in time series data, such as annual trends
in ecological monitoring, stock prices, or environmental measurements.

# Key References
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. (For a comprehensive overview of ARMA models).
- Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and Applications*. CRC Press. (For the GMRF interpretation of AR1 processes).
- Wikipedia: Autoregressive model

# Fields
- `rho::Distribution`: The prior distribution for the temporal autocorrelation parameter `rho`.
- `sigma::Distribution`: The prior distribution for the standard deviation of the AR1 innovations.
- `method::Symbol`: The computational method, either `:statespace` (default) or `:spectral`.
"""
struct AR1 <: ComponentModel
    rho::Distribution
    sigma::Distribution
    method::Symbol
end

# Add to the central component constructor registry.
COMPONENT_TYPE_REGISTRY[:ar1] = AR1
COMPONENT_CONSTRUCTORS[:ar1] = (p, params) -> AR1(p.rho, p.sigma, get(params, :method, :statespace))

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
    get_priors(m::AR1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code for the `AR1` component's priors. It defines priors for `sigma`,
the unconstrained `rho_raw`, and the latent innovations `innov`.

# Assumptions
- The `tanh` transformation on `rho_raw ~ Normal(0, 1.5)` provides a reasonable prior
  on `rho` that covers the `(-1, 1)` range while softly shrinking towards zero.
"""
function get_priors(m::AR1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent

    is_multivariate = arch == "multivariate"
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]

    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(p_names.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(p_names.sigma))")
        push!(priors_acc, "$(p_names.rho)_raw ~ NamedDist(Normal(0, 1.5), :$(Symbol(string(p_names.rho, "_raw"))))")
    end

    push!(priors_acc, "$(p_names.innov) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(p_names.innov))")
    
    return join(priors_acc, "\n    ")
end



# Purpose: Implements a state-space evolution for an AR(1) process.
# Rationale: This version is simplified by removing the explicit `T_model` argument and
#            replacing the final scaling loop with more efficient broadcasting.
function ar1_statespace(rho, sigma, innov, n_latent, noise)
    # This function computes the state-space evolution of an AR(1) process.
    # It is designed to be type-stable and work with different numeric types.
    
    T_num = promote_type(typeof(rho), typeof(sigma), eltype(innov), typeof(noise))
    latent = Vector{T_num}(undef, n_latent)
    
    if n_latent > 0
        # Initialize the first state using the stationary variance of the AR(1) process.
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
    get_updates(m::AR1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code to construct the `AR1` effect, dispatching on the chosen method
(`:statespace` or `:spectral`).

# Assumptions
- The `ar1_statespace` helper function is available for the `:statespace` method.
- The effect is additive on the scale of the linear predictor.
"""
function get_updates(m::AR1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx"
    n_latent = spec.hyper.n_latent

    if m.method == :spectral
        return """
            # --- AR1 Component: $(spec.key) (Spectral Method) ---
            $(p_names.rho) = tanh($(p_names.rho)_raw)
            let
                local U = spec.hyper.U
                local L_base = spec.hyper.L
                local lambda_vals = (1.0 + $(p_names.rho)^2) .+ $(p_names.rho) .* L_base
                local diag_D = $(p_names.sigma) ./ sqrt.(lambda_vals .+ M.noise)
                local latent_field = U * (diag_D .* $(p_names.innov))
                $(eta_target) .+= view(latent_field, M.$(index_var))
            end
        """
    else # :statespace
        return """
            # --- AR1 Component: $(spec.key) (State-Space Method) ---
            $(p_names.rho) = tanh($(p_names.rho)_raw)
            local latent_field = ar1_statespace($(p_names.rho), $(p_names.sigma), $(p_names.innov), $(n_latent), M.noise)
            $(eta_target) .+= view(latent_field, M.$(index_var))
        """
    end
end

"""
    get_effects(m::AR1, chain, M::NamedTuple, ...)::NamedTuple

Reconstructs the `AR1` component's effect from posterior samples, dispatching on the method
used during model fitting.
"""
function get_effects(m::AR1, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    noise_val = get(M, :noise, 1e-6)
    p_names_vec = string.(FlexiChains.parameters(chain))
    n_latent = spec.hyper.n_latent

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_name = _find_parameter_new(p_names_vec, string(spec.key), "sigma", k)
        rho_raw_name = _find_parameter_new(p_names_vec, string(spec.key), "rho_raw", k)
        innov_name = _find_parameter_new(p_names_vec, string(spec.key), "innov", k)

        if isempty(sigma_name) || isempty(rho_raw_name) || isempty(innov_name)
            @warn "Parameters for AR1 component $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_latent, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        rho_raw_samples = get_params_vector(chain, rho_raw_name, 1)[:, 1]
        rho_samples = tanh.(rho_raw_samples)
        innovations_samples = get_params_vector(chain, innov_name, n_latent)

        temporal_effect_k = zeros(Float64, n_latent, n_samples)
        
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
        push!(structured_effects, temporal_effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
