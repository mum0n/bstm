"""
    Harmonic <: ComponentModel

A component model for harmonic temporal effects, capturing periodic patterns using
sine and cosine waves. This component can model one or more harmonics, each with its
own amplitude, phase, and potentially its own period.

# Version
v1.3.3 (2026-08-14)

# Mathematical Summary
The component models a function \$f(t)\$ as a sum of sinusoids. It supports two
parameterizations controlled by the `method` field:

1.  **:twocoefficient (default, AD-friendly)**:
    \$f(t) = \\sum_{k=1}^{N_{harmonics}} \\left( \\beta_{\\cos,k} \\cos\\left(\\frac{2\\pi k t}{P_k}\\right) + \\beta_{\\sin,k} \\sin\\left(\\frac{2\\pi k t}{P_k}\\right) \\right)\$
    This is the recommended method as it is more efficient for gradient-based MCMC.

2.  **:ampphase (didactic)**:
    \$f(t) = \\sum_{k=1}^{N_{harmonics}} A_k \\cos\\left(\\frac{2\\pi k t}{P_k} + \\phi_k\\right)\$
    where \$A_k\$ is the amplitude, \$P_k\$ is the period, and \$\\phi_k\$ is the phase shift.
    This is retained as a more intuitive, didactic alternative.

# Computational Methods
- `:twocoefficient` (Default, AD-friendly): A two-coefficient (sine and cosine)
  parameterization that is efficient for gradient-based samplers.
- `:ampphase` (Didactic, Not AD-friendly): An amplitude-phase parameterization that
  is more intuitive but can be less efficient for MCMC. Retained for didactic purposes.

# Inputs
- **Required**:
  - A seasonal index variable (e.g., `month`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `nharmonics`: `Int`, the number of harmonic terms to include. Default: `1`.
  - `period`: `Real`, `UnivariateDistribution`, or `Vector`. The period(s) of the
    cycle(s). If a single value, it applies to all harmonics. If a vector, its
    length must match `nharmonics`. Default: `12.0`.
  - `amplitude`: `UnivariateDistribution`, prior for the amplitude (for `:ampphase`).
    Default: `Exponential(1.0)`.
  - `phase`: `UnivariateDistribution`, prior for the phase shift (for `:ampphase`).
    Default: `Beta(1,1)`.
  - `method`: `Symbol`, computational method (`:twocoefficient` or `:ampphase`).
    Default: `:twocoefficient`.

# Outputs (Parameter Names)
- `beta_cos_<key>`: Coefficients for the cosine terms (for `:twocoefficient`).
- `beta_sin_<key>`: Coefficients for the sine terms (for `:twocoefficient`).
- `amplitude_<key>`: Amplitudes of the harmonics (for `:ampphase`).
- `phase_<key>`: Phase shifts of the harmonics (for `:ampphase`).
- `period_<key>`: The period of the cycle(s), if estimated.
- `latent_<key>`: The reconstructed latent harmonic effect.
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

function get_precomputes(
    m::Harmonic, M::NamedTuple, mod_data::Dict
)::NamedTuple
    # Validate that a seasonal index variable is provided.
    variables = mod_data[:variables]
    if isempty(variables)
        error(
            "The Harmonic model requires a seasonal index variable, e.g., " *
            "`random(month, model=:harmonic)`."
        )
    end

    # Extract the seasonal index variable from the data.
    u_var_sym = Symbol(variables[1])
    if !hasproperty(M.data, u_var_sym)
        error(
            "Seasonal index variable ':$u_var_sym' for Harmonic model not found " *
            "in data."
        )
    end
    
    u_idx = M.data[!, u_var_sym]
    u_N = length(unique(u_idx))
    u_idx_var = u_var_sym # Store the symbol for the variable name

    u_coords = collect(1.0:u_N) # Generate coordinates for the unique levels.
    return (u_coords=u_coords, u_idx=u_idx, u_N=u_N, u_idx_var=u_idx_var)
end

function get_priors(
    m::Harmonic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    priors = String[]

    if m.method == :twocoefficient
        # Priors for cosine and sine coefficients
        beta_cos_prior = _distribution_to_string(Normal(0, 1))
        beta_sin_prior = _distribution_to_string(Normal(0, 1))
        push!(priors, "$(p_names.beta_cos) ~ DynamicPPL.NamedDist(filldist($(beta_cos_prior), $(m.nharmonics)), :$(p_names.beta_cos))")
        push!(priors, "$(p_names.beta_sin) ~ DynamicPPL.NamedDist(filldist($(beta_sin_prior), $(m.nharmonics)), :$(p_names.beta_sin))")
    elseif m.method == :ampphase
        # Priors for amplitude and phase
        amplitude_prior_str = _distribution_to_string(m.amplitude)
        phase_prior_str = _distribution_to_string(m.phase)
        push!(priors, "$(p_names.amplitude) ~ DynamicPPL.NamedDist(filldist($(amplitude_prior_str), $(m.nharmonics)), :$(p_names.amplitude))")
        push!(priors, "$(p_names.phase) ~ DynamicPPL.NamedDist(filldist($(phase_prior_str), $(m.nharmonics)), :$(p_names.phase))")
    end

    if m.period isa UnivariateDistribution
        push!(priors, "$(p_names.period) ~ $(_distribution_to_string(m.period))")
    elseif m.period isa Vector{<:UnivariateDistribution}
        period_prior_str = _distribution_to_string(m.period[1])
        push!(priors, "$(p_names.period) ~ filldist($(period_prior_str), $(m.nharmonics))")
    end

    # Prior for the innovations (raw coefficients)
    return join(priors, "\n    ")
end

function get_updates(
    m::Harmonic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    u_coords_access = "spec_registry[:$(spec.key)].hyper.u_coords" # Access pre-computed coordinates
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"

    period_access_code = if m.period isa Real
        string(m.period)
    elseif m.period isa UnivariateDistribution
        string(p_names.period)
    elseif m.period isa Vector{<:Real}
        "($(string(m.period)))[k]"
    elseif m.period isa Vector{<:UnivariateDistribution}
        "$(string(p_names.period))[k]"
    else
        error("Unsupported type for m.period in get_updates.")
    end

    loop_body = ""
    if m.method == :twocoefficient
        loop_body = """ # Use local for variables within the let block
            local b_cos = $(m.nharmonics > 1 ? "$(p_names.beta_cos)[k]" : p_names.beta_cos)
            local b_sin = $(m.nharmonics > 1 ? "$(p_names.beta_sin)[k]" : p_names.beta_sin)
            local period_val = $(period_access_code)
            
            local angle = (2 * pi * k ./ period_val) .* $(u_coords_access)
            $(p_names.latent) .+= b_cos .* cos.(angle) .+ b_sin .* sin.(angle) # Accumulate effect
        """
    else # :ampphase
        loop_body = """ # Use local for variables within the let block
            local amp = $(m.nharmonics > 1 ? "$(p_names.amplitude)[k]" : p_names.amplitude)
            local phase = $(m.nharmonics > 1 ? "$(p_names.phase)[k]" : p_names.phase)
            local period_val = $(period_access_code)
            
            local angle = (2 * pi * k ./ period_val) .* $(u_coords_access)
            $(p_names.latent) .+= amp .* cos.(angle .+ (2 * pi * phase)) # Accumulate effect
        """
    end

    # Initialize latent field with a type inferred from a sampled parameter.
    init_param = if m.method == :twocoefficient; p_names.beta_cos; else; p_names.amplitude; end
    
    return """
        # --- Harmonic Component: $(spec.key) ($(m.method)) ---
        let
            local T_num = eltype($(init_param)) # Promote numeric type for AD compatibility
            local u_N_val = spec_registry[:$(spec.key)].hyper.u_N # Access pre-computed u_N
            local u_idx_val = spec_registry[:$(spec.key)].hyper.u_idx # Access pre-computed u_idx
            $(p_names.latent) = zeros(T_num, u_N_val) # Initialize latent field
            for k in 1:$(m.nharmonics)
                $(loop_body)
            end
            $(eta_target) .+= view($(p_names.latent), u_idx_val) # Add to linear predictor
        end
    """
end

function get_effects(
    m::Harmonic, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}() # Initialize container for effects
    hyper = spec.hyper # Access pre-computed data from spec.hyper
    u_coords = hyper.u_coords # Coordinates for the harmonic basis
    u_idx_full = isnothing(PS) ? hyper.u_idx : vcat(hyper.u_idx, PS.u_idx) # Full index for train + prediction
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))

    for k_outcome in 1:outcomes_N
        reconstructed_effects_k = zeros(Float64, M.u_N, n_samples)
        
        p_names_k = generate_full_variable_names(spec, M.model_arch, k_outcome) # Generate outcome-specific names
        period_name = _find_parameter(p_names_vec, string(p_names_k.period), k_outcome, is_multivariate_model)
        
        period_samples = if m.period isa Real # If period is a fixed real value
            fill(m.period, n_samples, m.nharmonics)
        elseif m.period isa Vector{<:Real}
            repeat(m.period', n_samples, 1)
        else
            if isempty(period_name)
                @warn "Period parameter for Harmonic component $(spec.key) not found. Using prior mean."
                mean_period = m.period isa Vector ? mean.(m.period) : mean(m.period)
                repeat(mean_period', n_samples, 1)
            else
                get_params_vector(chain, period_name, m.nharmonics)
            end
        end

        for i in 1:n_samples
            sample_effect = zeros(hyper.u_N) # Initialize effect for current sample
            for k_harmonic in 1:m.nharmonics
                period_ik = m.period isa Vector ? period_samples[i, k_harmonic] : period_samples[i]
                angle = (2 * pi * k_harmonic / period_ik) .* u_coords
                
                if m.method == :twocoefficient
                    b_cos_name = _find_parameter(p_names_vec, string(p_names_k.beta_cos), k_outcome, is_multivariate_model)
                    b_sin_name = _find_parameter(p_names_vec, string(p_names_k.beta_sin), k_outcome, is_multivariate_model)
                    if isempty(b_cos_name) || isempty(b_sin_name)
                        @warn "beta_cos or beta_sin parameters for Harmonic component $(spec.key) (outcome $k_outcome) not found. Skipping reconstruction for this sample."
                        continue
                    end
                    
                    # Extract samples for current harmonic
                    b_cos_samps = get_params_vector(chain, b_cos_name, m.nharmonics)
                    b_sin_samps = get_params_vector(chain, b_sin_name, m.nharmonics)
                    b_cos_ik = m.nharmonics > 1 ? b_cos_samps[i, k_harmonic] : b_cos_samps[i]
                    b_sin_ik = m.nharmonics > 1 ? b_sin_samps[i, k_harmonic] : b_sin_samps[i]
                    
                    sample_effect .+= b_cos_ik .* cos.(angle) .+ b_sin_ik .* sin.(angle) # Accumulate effect
                else # :ampphase
                    amp_name = _find_parameter(p_names_vec, string(p_names_k.amplitude), k_outcome, is_multivariate_model)
                    phase_name = _find_parameter(p_names_vec, string(p_names_k.phase), k_outcome, is_multivariate_model)
                    if isempty(amp_name) || isempty(phase_name)
                        @warn "amplitude or phase parameters for Harmonic component $(spec.key) (outcome $k_outcome) not found. Skipping reconstruction for this sample."
                        continue
                    end

                    # Extract samples for current harmonic
                    amp_samps = get_params_vector(chain, amp_name, m.nharmonics)
                    phase_samps = get_params_vector(chain, phase_name, m.nharmonics)
                    amp_ik = m.nharmonics > 1 ? amp_samps[i, k_harmonic] : amp_samps[i]
                    phase_ik = m.nharmonics > 1 ? phase_samps[i, k_harmonic] : phase_samps[i]
                    
                    sample_effect .+= amp_ik .* cos.(angle .+ (2 * pi * phase_ik)) # Accumulate effect
                end
            end
            reconstructed_effects_k[:, i] = sample_effect
        end # End of samples loop

        indexed_effects = reconstructed_effects_k[u_idx_full, :] # Index to full observation set
        push!(structured_effects, indexed_effects)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
