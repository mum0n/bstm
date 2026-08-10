"""
    Harmonic <: ComponentModel

A component model for harmonic temporal effects, capturing periodic patterns using
sine and cosine waves. This component can model one or more harmonics, each with its
own amplitude, phase, and potentially its own period.

# Version
v1.2.3 (2026-08-10)

# Mathematical Summary
The component models a function f(t) as a sum of sinusoids. It supports two
parameterizations controlled by the `method` field:

1.  **:twocoefficient (default, AD-friendly)**:
    f(t) = \\sum_{k=1}^{N_{harmonics}} \\left( \\beta_{\\cos,k} \\cos\\left(\\frac{2\\pi k}{P_k} t\\right) + \\beta_{\\sin,k} \\sin\\left(\\frac{2\\pi k}{P_k} t\\right) \\right)
    This is the recommended method as it is more efficient for gradient-based MCMC.

2.  **:ampphase (didactic)**:
    f(t) = \\sum_{k=1}^{N_{harmonics}} A_k \\cos\\left(\\frac{2\\pi k}{P_k} t + \\phi_k\\right)
    where A_k is the amplitude, P_k is the period, and \\phi_k is the phase shift.
    This is retained as a more intuitive, didactic alternative.

# Fields
- `nharmonics::Int`: The number of harmonic terms.
- `amplitude::Distribution`: Prior for the amplitude (used in `:ampphase` method).
- `phase::Distribution`: Prior for the phase shift (used in `:ampphase` method).
- `period::Union{Real, UnivariateDistribution, Vector}`: The period(s) of the cycle(s).
- `method::Symbol`: The parameterization method, `:twocoefficient` or `:ampphase`.
"""
struct Harmonic <: ComponentModel
    nharmonics::Int
    amplitude::Distribution
    phase::Distribution
    period::Union{Real, UnivariateDistribution, Vector}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:harmonic] = Harmonic

COMPONENT_CONSTRUCTORS[:harmonic] = (p, params) -> begin
    nharmonics = get(params, :nharmonics, 1)
    period_param = get(params, :period, 12.0)
    method = get(params, :method, :twocoefficient)

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
    
    Harmonic(nharmonics, p.amplitude, p.phase, period_param, method)
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

Generates priors for the harmonic component, dispatching on the chosen method.
"""
function get_priors(
    m::Harmonic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    priors = String[]

    if m.method == :twocoefficient
        if m.nharmonics > 1
            push!(priors, "$(p_names.beta_cos) ~ filldist(Normal(0, 1), $(m.nharmonics))")
            push!(priors, "$(p_names.beta_sin) ~ filldist(Normal(0, 1), $(m.nharmonics))")
        else
            push!(priors, "$(p_names.beta_cos) ~ Normal(0, 1)")
            push!(priors, "$(p_names.beta_sin) ~ Normal(0, 1)")
        end
    elseif m.method == :ampphase
        amp_prior_str = _distribution_to_string(m.amplitude)
        phase_prior_str = _distribution_to_string(m.phase)
        if m.nharmonics > 1
            push!(priors, "$(p_names.amplitude) ~ filldist($(amp_prior_str), $(m.nharmonics))")
            push!(priors, "$(p_names.phase) ~ filldist($(phase_prior_str), $(m.nharmonics))")
        else
            push!(priors, "$(p_names.amplitude) ~ $(amp_prior_str)")
            push!(priors, "$(p_names.phase) ~ $(phase_prior_str)")
        end
    end

    if m.period isa UnivariateDistribution
        push!(priors, "$(p_names.period) ~ $(_distribution_to_string(m.period))")
    elseif m.period isa Vector{<:UnivariateDistribution}
        period_prior_str = _distribution_to_string(m.period[1])
        push!(priors, "$(p_names.period) ~ filldist($(period_prior_str), $(m.nharmonics))")
    end

    return join(priors, "\n    ")
end

"""
    get_updates(m::Harmonic, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates Turing code to construct the harmonic effect, dispatching on the method.
"""
function get_updates(
    m::Harmonic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    u_coords = "spec_registry[:$(spec.key)].precomputes.u_coords"
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"

    period_expr = if m.period isa Real || m.period isa Vector{<:Real}
        string(m.period)
    else
        string(p_names.period)
    end

    local loop_body
    if m.method == :twocoefficient
        loop_body = """
            local b_cos = $(m.nharmonics > 1 ? "$(p_names.beta_cos)[k]" : string(p_names.beta_cos))
            local b_sin = $(m.nharmonics > 1 ? "$(p_names.beta_sin)[k]" : string(p_names.beta_sin))
            local period = $(m.period isa Vector ? "$(period_expr)[k]" : period_expr)
            
            local angle = (2 * pi * k ./ period) .* $(u_coords)
            $(p_names.latent) .+= b_cos .* cos.(angle) .+ b_sin .* sin.(angle)
        """
    else # :ampphase
        loop_body = """
            local amp = $(m.nharmonics > 1 ? "$(p_names.amplitude)[k]" : string(p_names.amplitude))
            local phase = $(m.nharmonics > 1 ? "$(p_names.phase)[k]" : string(p_names.phase))
            local period = $(m.period isa Vector ? "$(period_expr)[k]" : period_expr)
            
            local phase_rad = 2 * pi * phase
            local angle = (2 * pi * k ./ period) .* $(u_coords)
            
            $(p_names.latent) .+= amp .* cos.(angle .+ phase_rad)
        """
    end

    return """
        # --- Harmonic Component: $(spec.key) ($(m.method)) ---
        local $(p_names.latent) = zeros(M.u_N)
        for k in 1:$(m.nharmonics)
            $(loop_body)
        end
        $(eta_target) .+= view($(p_names.latent), M.u_idx)
    """
end

"""
    get_effects(m::Harmonic, chain, M, n_samples, outcomes_N, spec, PS, N_total)::NamedTuple

Reconstructs the harmonic effect from posterior samples, dispatching on method.
"""
function get_effects(
    m::Harmonic, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    u_coords = spec.precomputes.u_coords
    u_idx_full = isnothing(PS) ? M.u_idx : vcat(M.u_idx, PS.u_idx)

    for k_outcome in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k_outcome)
        
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
                period_ik = m.period isa Vector ? period_samples[i, k_harmonic] : period_samples[i]
                angle = (2 * pi * k_harmonic / period_ik) .* u_coords
                
                if m.method == :twocoefficient
                    b_cos_samps = get_params_vector(chain, string(p_names.beta_cos), m.nharmonics)
                    b_sin_samps = get_params_vector(chain, string(p_names.beta_sin), m.nharmonics)
                    b_cos_ik = m.nharmonics > 1 ? b_cos_samps[i, k_harmonic] : b_cos_samps[i]
                    b_sin_ik = m.nharmonics > 1 ? b_sin_samps[i, k_harmonic] : b_sin_samps[i]
                    sample_effect .+= b_cos_ik .* cos.(angle) .+ b_sin_ik .* sin.(angle)
                else # :ampphase
                    amp_samps = get_params_vector(chain, string(p_names.amplitude), m.nharmonics)
                    phase_samps = get_params_vector(chain, string(p_names.phase), m.nharmonics)
                    amp_ik = m.nharmonics > 1 ? amp_samps[i, k_harmonic] : amp_samps[i]
                    phase_ik = m.nharmonics > 1 ? phase_samps[i, k_harmonic] : phase_samps[i]
                    sample_effect .+= amp_ik .* cos.(angle .+ (2 * pi * phase_ik))
                end
            end
            reconstructed_effects_k[:, i] = sample_effect
        end

        indexed_effects = reconstructed_effects_k[u_idx_full, :]
        push!(structured_effects, indexed_effects)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
