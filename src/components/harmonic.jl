
"""
    Harmonic <: ComponentModel

A component model for harmonic temporal effects, capturing periodic patterns using
sine and cosine waves. This component can model one or more harmonics, each with its
own amplitude, phase, and potentially its own period.

# Version
v1.2.2 (2026-08-08)

# Mathematical Summary
The component models a function f(t) as a sum of sinusoids:
f(t) = \\sum_{k=1}^{N_{harmonics}} A_k \\cos\\left(\\frac{2\\pi k}{P_k} t + \\phi_k\\right)
where A_k is the amplitude, P_k is the period, and \\phi_k is the phase shift for
the k-th harmonic. For improved MCMC sampling, this is internally reparameterized
into a two-coefficient form:
f(t) = \\sum_{k=1}^{N_{harmonics}} \\left( \\beta_{\\cos,k} \\cos\\left(\\frac{2\\pi k}{P_k} t\\right) + \\beta_{\\sin,k} \\sin\\left(\\frac{2\\pi k}{P_k} t\\right) \\right)

# Assumptions
- The underlying process has a periodic component that can be well-approximated by a
  sum of sinusoids.
- The seasonal index provided is discrete and regularly spaced.

# Best Use Case
Modeling seasonal effects (e.g., monthly, quarterly) or other known periodic
phenomena in time series data where the periodicity is stable over time. It is
particularly useful for capturing cyclical patterns in environmental data,
economics, and epidemiology.

# Key References
- General concept of Fourier Series: [Wikipedia: Fourier Series](https://en.wikipedia.org/wiki/Fourier_series)

# Fields
- `nharmonics::Int`: The number of harmonic terms to include in the sum (e.g., 1
  for annual, 2 for annual and semi-annual).
- `amplitude::Distribution`: The prior distribution for the amplitude of the
  harmonic(s).
- `phase::Distribution`: The prior distribution for the phase shift of the
  harmonic(s), typically on `[0, 1]`.
- `period::Union{Real, UnivariateDistribution, Vector{<:UnivariateDistribution}}`: The
  period of the cycle(s).
  - If a `Real`, the period is fixed.
  - If a `UnivariateDistribution`, the period is estimated (only for `nharmonics=1`).
  - If a `Vector{<:UnivariateDistribution}`, each harmonic gets its own estimated period.
"""
struct Harmonic <: ComponentModel
    nharmonics::Int
    amplitude::Distribution
    phase::Distribution
    period::Union{Real, UnivariateDistribution, Vector{<:UnivariateDistribution}}
end

COMPONENT_TYPE_REGISTRY[:harmonic] = Harmonic

COMPONENT_CONSTRUCTORS[:harmonic] = (p, params) -> begin
    nharmonics = get(params, :nharmonics, 1)
    period_param = get(params, :period, 12.0)
    
    if nharmonics > 1
        if period_param isa Real
            period_param = fill(period_param, nharmonics)
        elseif period_param isa UnivariateDistribution
            period_param = [period_param for _ in 1:nharmonics]
        end
        if period_param isa Vector && length(period_param) != nharmonics
            error(
                "Length of `period` vector ($(length(period_param))) must match " *
                "`nharmonics` ($(nharmonics))."
            )
        end
    else # nharmonics == 1
        if period_param isa Vector
            error(
                "For `nharmonics = 1`, `period` must be a Real or a single " *
                "UnivariateDistribution, not a Vector."
            )
        end
    end
    
    Harmonic(nharmonics, p.amplitude, p.phase, period_param)
end

MODEL_TO_STRUCTURE_MAP[:harmonic] = :temporal

"""
    get_datastructures!(m_type::Type{<:Harmonic}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Harmonic` component. It extracts the
observation-level seasonal index vector (e.g., a vector of month numbers) and
determines the number of unique seasonal units, `u_N`, which defines the
dimensionality of the latent harmonic field.

# Assumptions
- The input data column specified in the `random()` call contains discrete,
  integer-like indices representing seasonal units (e.g., month 1-12).
"""
function get_datastructures!(
    m_type::Type{<:Harmonic}, M::Dict, mod_data::Dict
)::Bool
    variables = mod_data[:variables]
    if isempty(variables)
        error(
            "The Harmonic model requires a seasonal index variable, e.g., " *
            "`random(month, model=:harmonic)`."
        )
    end

    u_var_sym = Symbol(variables[1])
    if !hasproperty(M[:data], u_var_sym)
        error(
            "Seasonal index variable ':$u_var_sym' for Harmonic model not found " *
            "in data."
        )
    end
    
    M[:u_idx] = M[:data][!, u_var_sym]
    M[:u_N] = length(unique(M[:u_idx]))
    M[:u_idx_var] = u_var_sym
    
    return true
end

"""
    get_precomputes(m::Harmonic, M::NamedTuple, mod_data::Dict)::NamedTuple

Generates the discrete time vector `t = [1, 2, ..., u_N]` over which the harmonic
basis functions will be evaluated to construct the latent seasonal effect.

# Assumptions
- The seasonal units are regularly spaced and can be represented by a simple
  integer sequence.
"""
function get_precomputes(
    m::Harmonic, M::NamedTuple, mod_data::Dict
)::NamedTuple
    u_coords = collect(1.0:M.u_N)
    return (u_coords=u_coords,)
end

"""
    get_priors(m::Harmonic, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the priors on the harmonic component's parameters.
For improved MCMC sampling stability, this method samples the `beta_cos` and
`beta_sin` coefficients of the two-coefficient form directly.

# Assumptions
- A `Normal(0,1)` prior on the beta coefficients is a reasonable default.
- If the period is estimated, the user has provided a suitable prior distribution.
"""
function get_priors(
    m::Harmonic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = String[]
    
    if m.nharmonics > 1
        push!(priors, "$(p_names.beta_cos) ~ filldist(Normal(0, 1), $(m.nharmonics))")
        push!(priors, "$(p_names.beta_sin) ~ filldist(Normal(0, 1), $(m.nharmonics))")
    else
        push!(priors, "$(p_names.beta_cos) ~ Normal(0, 1)")
        push!(priors, "$(p_names.beta_sin) ~ Normal(0, 1)")
    end

    if m.period isa UnivariateDistribution
        period_prior_str = _distribution_to_string(m.period)
        push!(priors, "$(p_names.period) ~ $(period_prior_str)")
    elseif m.period isa Vector{<:UnivariateDistribution}
        period_prior_str = _distribution_to_string(m.period[1])
        push!(
            priors,
            "$(p_names.period) ~ filldist($(period_prior_str), $(m.nharmonics))"
        )
    end

    return join(priors, "\n    ")
end

"""
    get_updates(m::Harmonic, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code to construct the harmonic effect by summing the cosine
and sine waves and adds the result to the linear predictor `eta`.

# Assumptions
- The effect of the harmonic component is additive on the scale of the linear predictor.
"""
function get_updates(
    m::Harmonic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    u_coords = "spec_registry[:$(spec.key)].precomputes.u_coords"
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"

    period_expr = if m.period isa Real
        string(m.period)
    elseif m.period isa Vector{<:Real}
        string(m.period)
    else
        string(p_names.period)
    end

    loop_body = """
        local b_cos = $(m.nharmonics > 1 ? "$(p_names.beta_cos)[k]" : string(p_names.beta_cos))
        local b_sin = $(m.nharmonics > 1 ? "$(p_names.beta_sin)[k]" : string(p_names.beta_sin))
        local period = $(m.period isa Vector ? "$(period_expr)[k]" : period_expr)
        
        local angle = (2 * pi * k ./ period) .* $(u_coords)
        
        $(p_names.latent) .+= b_cos .* cos.(angle) .+ b_sin .* sin.(angle)
    """

    return """
        # --- Harmonic Component: $(spec.key) ---
        local $(p_names.latent) = zeros(M.u_N)
        for k in 1:$(m.nharmonics)
            $(loop_body)
        end
        $(eta_target) .+= view($(p_names.latent), M.u_idx)
    """
end

"""
    get_effects(m::Harmonic, chain, M, n_samples, outcomes_N, spec, PS, N_total)::NamedTuple

Reconstructs the harmonic component's effect from the MCMC chain's posterior samples.
It retrieves the sampled coefficients and period, reconstructs the latent harmonic
field, and maps it to the observation-level indices.

# Assumptions
- The MCMC `chain` contains posterior samples for the `beta_cos`, `beta_sin`, and
  `period` parameters.
"""
function get_effects(
    m::Harmonic, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    u_coords = spec.precomputes.u_coords
    u_idx_full = isnothing(PS) ? M.u_idx : PS.u_idx

    for k_outcome in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k_outcome)

        beta_cos_samples = get_params_vector(
            chain, string(p_names.beta_cos), m.nharmonics
        )
        beta_sin_samples = get_params_vector(
            chain, string(p_names.beta_sin), m.nharmonics
        )
        
        period_samples = if m.period isa Real
            fill(m.period, n_samples, m.nharmonics)
        elseif m.period isa Vector{<:Real}
            repeat(m.period', n_samples, 1)
        else
            get_params_vector(chain, string(p_names.period), m.nharmonics)
        end

        reconstructed_effects_k = zeros(Float64, M.u_N, n_samples)

        for i in 1:n_samples
            sample_effect = zeros(M.u_N)
            for k_harmonic in 1:m.nharmonics
                b_cos_ik = m.nharmonics > 1 ?
                    beta_cos_samples[i, k_harmonic] : beta_cos_samples[i]
                b_sin_ik = m.nharmonics > 1 ?
                    beta_sin_samples[i, k_harmonic] : beta_sin_samples[i]
                period_ik = m.period isa Vector ?
                    period_samples[i, k_harmonic] : period_samples[i]
                
                angle = (2 * pi * k_harmonic / period_ik) .* u_coords
                sample_effect .+= b_cos_ik .* cos.(angle) .+ b_sin_ik .* sin.(angle)
            end
            reconstructed_effects_k[:, i] = sample_effect
        end

        indexed_effects = reconstructed_effects_k[u_idx_full, :]
        push!(structured_effects, indexed_effects)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
