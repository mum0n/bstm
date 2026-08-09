"""
    AR2 <: ComponentModel

A component model for a second-order autoregressive (AR2) process. This model
extends the AR1 process by including a second lag, allowing it to capture more
complex temporal dynamics, such as damped oscillations or pseudo-periodic behavior.

# Version
v1.0.1 (2026-08-08)

# Mathematical Summary
The AR2 process models the value of a latent field \$\\phi\$ at time \$t\$ as a linear
combination of its values at the two previous time steps, plus an independent
innovation term \$\\epsilon_t\$. The model is defined as:
\$\\phi_t = \\rho_1 \\phi_{t-1} + \\rho_2 \\phi_{t-2} + \\epsilon_t\$,
where \$\\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2)\$

For stability and efficient sampling, the model is implemented using a state-space
formulation. Stationarity of the process is ensured by enforcing constraints on the
autocorrelation parameters \$\\rho_1\$ and \$\\rho_2\$ within the `ar2_statespace`
helper function:
1. \$\\rho_1 + \\rho_2 < 1\$
2. \$\\rho_2 - \\rho_1 < 1\$
3. \$|\\rho_2| < 1\$

If these conditions are violated during sampling, the model's log-probability is set
to `-Inf`, effectively rejecting the proposal.

# Assumptions
- The temporal process is stationary.
- The innovations are Gaussian and independent over time.
- The effect is additive on the scale of the linear predictor.

# Best Use Case
Modeling time series with more complex dynamics than simple exponential decay, such as
business cycles, ecological population fluctuations with oscillatory patterns, or any
process where momentum from two previous time steps is relevant.

# Key References
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
  (For a comprehensive overview of ARMA models).
- Wikipedia: Autoregressive model

# Fields
- `rho1::Distribution`: The prior for the first temporal autocorrelation parameter,
  \$\\rho_1\$.
- `rho2::Distribution`: The prior for the second temporal autocorrelation parameter,
  \$\\rho_2\$.
- `sigma::Distribution`: The prior for the standard deviation of the AR2 innovations.
"""
struct AR2 <: ComponentModel
    rho1::Distribution
    rho2::Distribution
    sigma::Distribution
end

# Add to the central component constructor registry.
COMPONENT_TYPE_REGISTRY[:ar2] = AR2
COMPONENT_CONSTRUCTORS[:ar2] = (p, params) -> AR2(p.rho1, p.rho2, p.sigma)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[:ar2] = :temporal

"""
    get_datastructures!(m_type::Type{<:AR2}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `AR2` component. It ensures a time index
variable is provided and sets up the temporal context (`t_idx`, `t_N`) in the
main model configuration `M`.

# Assumptions
- The input data column specified in the `random()` call contains discrete,
  integer-like indices representing time units (e.g., year numbers).
"""
function get_datastructures!(m_type::Type{<:AR2}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]
    if isempty(variables)
        error("The AR2 model requires a time index variable, e.g., `random(year, model=:ar2)`.")
    end
    t_var_sym = Symbol(variables[1])
    if !hasproperty(M[:data], t_var_sym)
        error("Time index variable ':$t_var_sym' for AR2 model not found in data.")
    end
    
    tu_meta = assign_time_units(M[:data][!, t_var_sym])
    M[:t_idx] = tu_meta.idx
    M[:t_N] = tu_meta.N_cat
    M[:t_idx_var] = t_var_sym
    
    return true
end

"""
    get_precomputes(m::AR2, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations. For AR2, this method is not strictly
necessary as the model is implemented via a state-space formulation, but it returns
the number of latent variables (`n_latent`) for consistency with the component
interface.
"""
function get_precomputes(m::AR2, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.t_N
    template = build_structure_template(:ar1, n) # An AR1 template is sufficient
    return (Q_template=template.matrix, n_latent=n)
end


# Purpose: Implements a stationary state-space evolution for an AR(2) process.
# Rationale: This version is updated to be explicitly AD-aware. All numeric literals
#            are promoted to the generic numeric type `T_num`, preventing type errors
#            when the function is called with `ForwardDiff.Dual` numbers.
function ar2_statespace(rho1, rho2, sigma, innov::AbstractVector, n_latent::Int, noise)
    T_num = promote_type(typeof(rho1), typeof(rho2), typeof(sigma), eltype(innov), typeof(noise))
    latent = Vector{T_num}(undef, n_latent)
    if n_latent == 0
        return latent
    end

    if rho1 + rho2 >= T_num(1.0) || rho2 - rho1 >= T_num(1.0) || abs(rho2) >= T_num(1.0)
        return fill(T_num(1e12), n_latent)
    end

    var_innov = sigma^2
    gamma_0 = var_innov * (T_num(1.0) - rho2) / ((T_num(1.0) + rho2) * ((T_num(1.0) - rho2)^2 - rho1^2) + T_num(noise))
    gamma_1 = (rho1 / (T_num(1.0) - rho2)) * gamma_0

    cov_12 = [gamma_0 gamma_1; gamma_1 gamma_0]
    L_12 = cholesky(Symmetric(cov_12 + T_num(noise) * I)).L

    if n_latent >= 2
        latent[1:2] = L_12 * innov[1:2]
    elseif n_latent == 1
        latent[1] = sqrt(gamma_0) * innov[1]
    end

    for t in 3:n_latent
        latent[t] = rho1 * latent[t-1] + rho2 * latent[t-2] + innov[t] * sigma
    end

    return latent
end


"""
    get_priors(m::AR2, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the `AR2` component's priors. It defines priors for
`sigma`, `rho1`, `rho2`, and the latent innovations `innov`.

# Assumptions
- The stationarity conditions for `rho1` and `rho2` are checked within the
  `ar2_statespace` helper function, not via truncated priors, which is more
  efficient for sampling.
"""
function get_priors(
    m::AR2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent

    is_multivariate = arch == "multivariate"
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]

    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")
        push!(priors_acc, "$(p_names.rho1) ~ $(_distribution_to_string(m.rho1))")
        push!(priors_acc, "$(p_names.rho2) ~ $(_distribution_to_string(m.rho2))")
    end

    push!(priors_acc, "$(p_names.innov) ~ MvNormal(zeros(T, $(n_latent)), I)")
    
    return join(priors_acc, "\n    ")
end

"""
    get_updates(m::AR2, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code to construct the `AR2` effect and add it to the linear
predictor. It uses a state-space formulation for efficiency, calling the
`ar2_statespace` helper.

# Assumptions
- The `ar2_statespace` helper function is available in the model's execution scope.
- The effect is additive on the scale of the linear predictor.
"""
function get_updates(
    m::AR2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx"
    n_latent = spec.hyper.n_latent

    return """
        # --- AR2 Component: $(spec.key) ---
        local latent_field = ar2_statespace($(p_names.rho1), $(p_names.rho2), $(p_names.sigma), $(p_names.innov), $(n_latent), M.noise)
        $(eta_target) .+= view(latent_field, M.$(index_var))
    """
end

"""
    get_effects(m::AR2, chain, M::NamedTuple, n_samples, outcomes_N, spec, PS, N_total)::NamedTuple

Reconstructs the `AR2` component's effect from the MCMC chain's posterior samples.
It mirrors the generative logic by calling `ar2_statespace` for each posterior sample.

# Assumptions
- The MCMC `chain` contains posterior samples for `sigma`, `rho1`, `rho2`, and `innov`.
"""
function get_effects(
    m::AR2, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, spec::NamedTuple,
    PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    noise_val = get(M, :noise, 1e-6)
    p_names_vec = string.(FlexiChains.parameters(chain))
    n_latent = spec.hyper.n_latent

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_name = _find_parameter_new(p_names_vec, string(spec.key), "sigma", k)
        rho1_name = _find_parameter_new(p_names_vec, string(spec.key), "rho1", k)
        rho2_name = _find_parameter_new(p_names_vec, string(spec.key), "rho2", k)
        innov_name = _find_parameter_new(p_names_vec, string(spec.key), "innov", k)

        if isempty(sigma_name) || isempty(rho1_name) || isempty(rho2_name) || isempty(innov_name)
            @warn "Parameters for AR2 component $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_latent, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        rho1_samples = get_params_vector(chain, rho1_name, 1)[:, 1]
        rho2_samples = get_params_vector(chain, rho2_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innov_name, n_latent)

        temporal_effect_k = zeros(Float64, n_latent, n_samples)
        for j in 1:n_samples
            temporal_effect_k[:, j] = ar2_statespace(
                rho1_samples[j], rho2_samples[j], sigma_samples[j],
                innovations_samples[j, :], n_latent, noise_val
            )
        end
        push!(structured_effects, temporal_effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
