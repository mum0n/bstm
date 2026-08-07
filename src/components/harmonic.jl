# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    Harmonic <: ComponentModel

A component model for harmonic temporal effects, capturing periodic patterns using sine and cosine waves.
This component can model one or more harmonics, each with its own amplitude, phase, and potentially its own period.

# Fields
- `nharmonics::Int`: The number of harmonic terms to include in the sum (e.g., 1 for annual, 2 for annual and semi-annual).
- `amplitude::Distribution`: The prior distribution for the amplitude of the harmonic(s).
- `phase::Distribution`: The prior distribution for the phase shift of the harmonic(s), typically on `[0, 1]`.
- `period::Union{Real, UnivariateDistribution, Vector{<:UnivariateDistribution}}`: The period of the cycle(s).
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

# Add to the central component constructor registry.
# This logic validates the parameters and constructs the Harmonic object.
COMPONENT_CONSTRUCTORS[:harmonic] = (p, params) -> begin
    nharmonics = get(params, :nharmonics, 1)
    period_param = get(params, :period, 12.0)
    
    if nharmonics > 1
        if period_param isa Real
            error("For `nharmonics > 1`, `period` must be a `Distribution` or a `Vector{<:Distribution}` to be estimated, not a fixed Real value.")
        end
        if period_param isa UnivariateDistribution
            period_param = [period_param for _ in 1:nharmonics]
        end
        if period_param isa Vector && length(period_param) != nharmonics
            error("Length of `period` vector ($(length(period_param))) must match `nharmonics` ($(nharmonics)).")
        end
    else # nharmonics == 1
        if period_param isa Vector
            error("For `nharmonics = 1`, `period` must be a Real or a single UnivariateDistribution, not a Vector.")
        end
    end
    
    Harmonic(nharmonics, p.amplitude, p.phase, period_param)
end

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[Harmonic] = :temporal

"""
    get_datastructures!(m_type::Type{<:Harmonic}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Harmonic` component. It ensures that a temporal
index variable is provided and that the temporal context (`t_idx`, `t_N`) is set up in the
main model configuration `M`.
"""
function get_datastructures!(m_type::Type{<:Harmonic}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    variables = mod_data[:variables]

    if isempty(variables)
        error("The Harmonic model requires a temporal index variable, e.g., `random(month, model=:harmonic)`.")
    end

    if !haskey(M, :t_idx)
        time_var = Symbol(variables[1])
        if !hasproperty(M[:data], time_var)
            error("Temporal variable ':$time_var' for Harmonic model not found in data.")
        end
        
        time_opts = Dict(:time_method => get(params, :time_method, "regular"))
        period = get(params, :period, 12.0)
        if period isa Real
            time_opts[:t_N] = Int(period)
        end

        tu = assign_time_units(M[:data][!, time_var]; time_opts...)
        M[:t_idx] = tu.idx
        M[:t_N] = tu.N_cat
        @info "Inferred temporal context for Harmonic model: t_N = $(M[:t_N])."
    end

    return true
end

"""
    get_precomputes(m::Harmonic, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the time coordinates `1:M.t_N` for constructing the harmonic basis functions.
"""
function get_precomputes(m::Harmonic, M::NamedTuple, mod_data::Dict)::NamedTuple
    t_coords = collect(1.0:M.t_N)
    return (t_coords=t_coords,)
end

"""
    get_priors(m::Harmonic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code for the priors on amplitude, phase, and period (if estimated).
"""
function get_priors(m::Harmonic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    amp_prior_str = _distribution_to_string(m.amplitude)
    phase_prior_str = _distribution_to_string(m.phase)

    priors = String[]
    # Amplitude and Phase priors
    if m.nharmonics > 1
        push!(priors, "$(p_names.amplitude) ~ filldist($(amp_prior_str), $(m.nharmonics))")
        push!(priors, "$(p_names.phase) ~ filldist($(phase_prior_str), $(m.nharmonics))")
    else
        push!(priors, "$(p_names.amplitude) ~ $(amp_prior_str)")
        push!(priors, "$(p_names.phase) ~ $(phase_prior_str)")
    end

    # Period prior (if it's a distribution)
    if m.period isa UnivariateDistribution
        period_prior_str = _distribution_to_string(m.period)
        push!(priors, "$(p_names.period) ~ $(period_prior_str)")
    elseif m.period isa Vector
        # The constructor ensures all distributions in the vector are the same.
        period_prior_str = _distribution_to_string(m.period[1])
        push!(priors, "$(p_names.period) ~ filldist($(period_prior_str), $(m.nharmonics))")
    end

    return join(priors, "\n    ")
end

"""
    get_updates(m::Harmonic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code to construct the harmonic effect by summing the cosine waves
and adds the result to the linear predictor `eta`.
"""
function get_updates(m::Harmonic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    t_coords = "spec_registry[:$(spec.key)].precomputes.t_coords"

    # Define how to access the period value (fixed or sampled)
    period_expr = if m.period isa Real
        string(m.period)
    else
        string(p_names.period)
    end

    # Construct the loop to sum harmonic components
    loop_body = """
        local amp = $(m.nharmonics > 1 ? "$(p_names.amplitude)[k]" : string(p_names.amplitude))
        local phase = $(m.nharmonics > 1 ? "$(p_names.phase)[k]" : string(p_names.phase))
        local period = $(m.period isa Vector ? "$(period_expr)[k]" : period_expr)
        
        # The full harmonic effect is constructed for each of the t_N time points.
        # The phase is scaled by 2π to map from [0,1] to a full cycle.
        $(p_names.latent) .+= amp .* cos.( (2*pi*k ./ period) .* $(t_coords) .+ (2*pi .* phase) )
    """

    # Assemble the final code block
    return """
        # --- Harmonic Component: $(spec.key) ---
        local $(p_names.latent) = zeros(T, M.t_N)
        for k in 1:$(m.nharmonics)
            $(loop_body)
        end
        # Add the effect to the linear predictor, indexed by the observation's time unit.
        eta .+= $(p_names.latent)[M.t_idx]
    """
end

"""
    get_effects(m::Harmonic, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the harmonic component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::Harmonic, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    # Extract posterior samples
    amp_samples = get(chain, p_names.amplitude)
    phase_samples = get(chain, p_names.phase)
    
    period_samples = if m.period isa Real
        fill(m.period, n_samples)
    else
        get(chain, p_names.period)
    end

    t_coords = spec.precomputes.t_coords
    idx_to_use = isnothing(PS) ? M.t_idx : PS.t_idx
    
    # Initialize array for reconstructed effects over the unique time points
    reconstructed_effects = zeros(n_samples, M.t_N)

    # Reconstruct for each sample
    for i in 1:n_samples
        sample_effect = zeros(M.t_N)
        for k in 1:m.nharmonics
            amp_ik = m.nharmonics > 1 ? amp_samples[i, k] : amp_samples[i]
            phase_ik = m.nharmonics > 1 ? phase_samples[i, k] : phase_samples[i]
            
            period_ik = if m.period isa Real
                m.period
            elseif m.period isa Vector
                period_samples[i, k]
            else # UnivariateDistribution
                period_samples[i]
            end
            
            sample_effect .+= amp_ik .* cos.((2*pi*k ./ period_ik) .* t_coords .+ (2*pi .* phase_ik))
        end
        reconstructed_effects[i, :] = sample_effect
    end

    # Summarize effects
    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    # Index the effects to match the observation data points
    indexed_mean = mean_effect[idx_to_use]
    indexed_lower = lower_ci[idx_to_use]
    indexed_upper = upper_ci[idx_to_use]

    return (structured=(mean=indexed_mean, lower=indexed_lower, upper=indexed_upper),)
end
