
"""
    Cyclic <: ComponentModel

A component model for cyclic temporal effects, typically used for seasonal patterns.
It implements a cyclic random walk of order 1, where the last point connects to the first.

# Fields
- `period::Int`: The length of the cycle (e.g., 12 for months, 7 for days).
- `sigma::Distribution`: The prior distribution for the standard deviation of the cyclic effect.
"""
struct Cyclic <: ComponentModel
    period::Int
    sigma::Distribution
end


COMPONENT_CONSTRUCTORS[:cyclic] = (p, params) -> Cyclic(get(params, :period, 12), p.sigma),

# helper to map to classes of methods (data structures), :any mean it can be used in many approaches
MODEL_TO_STRUCTURE_MAP[:cyclic] = :temporal


"""
    get_datastructures!(m_type::Type{<:Cyclic}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Cyclic` component.
It ensures that the `period` is valid and sets up the temporal context (`t_N`, `t_idx`)
if not already present.

# Arguments
- `m_type`: The `Type` of the `Cyclic` component.
- `M`: The main model configuration dictionary.
- `mod_data`: A dictionary containing parsed module data.

# Returns
- `true` if a component object should be created.
"""
function get_datastructures!(m_type::Type{<:Cyclic}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    period = get(params, :period, 12)

    if !haskey(M, :t_N) || M[:t_N] == 0
        # If temporal context is not set, try to infer from period
        M[:t_N] = period
        M[:t_idx] = repeat(1:period, outer=ceil(Int, M[:y_N] / period))[1:M[:y_N]]
        @warn "Temporal context (t_N, t_idx) not explicitly set. Inferred from Cyclic period: $(period)."
    end

    if period <= 0
        error("Cyclic period must be a positive integer. Got $period.")
    end

    if period != M[:t_N]
        @warn "Cyclic period ($period) does not match inferred or provided temporal units (t_N=$(M[:t_N])). Using t_N for cyclic effect dimension."
        params[:period] = M[:t_N]
    end

    return true
end

"""
    get_precomputes(m::Cyclic, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `Cyclic` component,
specifically building the `Q_template` for the cyclic random walk.

# Arguments
- `m`: The `Cyclic` component instance.
- `M`: The main model configuration `NamedTuple`.
- `mod_data`: A dictionary containing parsed module data.

# Returns
- A `NamedTuple` containing `Q_template`, `scaling_factor`, `U`, and `L`.
"""
function get_precomputes(m::Cyclic, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.t_N # The number of temporal units
    
    # Build the Q_template for a cyclic random walk of order 1
    # This is a circulant matrix:
    # 2 -1  0 ... 0 -1
    # -1 2 -1 ... 0  0
    # ...
    # -1 0  0 ... -1 2
    Q_template = spzeros(Float64, n, n)
    if n > 0
        Q_template = spdiagm(0 => fill(2.0, n), 1 => fill(-1.0, n-1), -1 => fill(-1.0, n-1))
        Q_template[1, n] = -1.0
        Q_template[n, 1] = -1.0
    end
    
    rank_deficiency = 1 # Cyclic RW1 has rank deficiency of 1

    # Compute eigendecomposition for spectral sampling
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values

    # Compute scaling factor (geometric mean of non-zero eigenvalues)
    scaling_factor = _compute_scaling_factor(L, rank_deficiency)
    
    # Rescale Q_template and eigenvalues
    Q_template_scaled = Q_template ./ scaling_factor
    L_scaled = L ./ scaling_factor

    return (Q_template=Q_template_scaled, scaling_factor=scaling_factor, U=U, L=L_scaled)
end

"""
    get_priors(m::Cyclic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `Cyclic` component's priors.
It defines the prior for `sigma` and the latent field `raw`.

# Arguments
- `m`: The `Cyclic` component instance.
- `spec`: A `NamedTuple` containing the component's full specification.
- `arch`: The model architecture (`"univariate"`, `"multivariate"`).
- `outcome_idx`: The index of the outcome for multivariate models, `nothing` otherwise.
- `M`: The main model configuration `NamedTuple`.

# Returns
- A `String` containing the generated Turing code for priors.
"""
function get_priors(m::Cyclic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    # Prior for sigma
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    # Latent field prior (non-centered parameterization using spectral decomposition)
    # raw ~ MvNormal(zeros(T, M.t_N), I)
    # latent = U * Diagonal(1 ./ sqrt.(L .+ M.noise)) * raw
    
    return """
        $(p_names.sigma) ~ NamedDist($(sigma_prior_str), :$(p_names.sigma))
        $(p_names.raw) ~ NamedDist(MvNormal(zeros(T, M.t_N), I), :$(p_names.raw))
    """
end

"""
    get_updates(m::Cyclic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `Cyclic` component's effect
and adding it to the linear predictor (`eta`).

# Arguments
- `m`: The `Cyclic` component instance.
- `spec`: A `NamedTuple` containing the component's full specification.
- `arch`: The model architecture (`"univariate"`, `"multivariate"`).
- `outcome_idx`: The index of the outcome for multivariate models, `nothing` otherwise.
- `M`: The main model configuration `NamedTuple`.

# Returns
- A `String` containing the generated Turing code for the component's update logic.
"""
function get_updates(m::Cyclic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    # Reconstruct latent field using spectral decomposition
    # latent = sigma * U * Diagonal(1 ./ sqrt.(L .+ M.noise)) * raw
    # The effect is then indexed by the temporal index.
    
    return """
        local $(p_names.latent) = $(p_names.sigma) .* spec_registry[:$(spec.key)].precomputes.U * Diagonal(1 ./ sqrt.(spec_registry[:$(spec.key)].precomputes.L .+ M.noise)) * $(p_names.raw)
        eta .+= $(p_names.latent)[M.t_idx]
    """
end

"""
    get_effects(m::Cyclic, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `Cyclic` component's effect from the MCMC chain's posterior samples.

# Arguments
- `m`: The `Cyclic` component instance.
- `chain`: The MCMC chain object.
- `M`: The main model configuration `NamedTuple`.
- `n_samples`: The total number of posterior samples.
- `outcomes_N`: The total number of outcomes in the model.
- `p_names`: A `NamedTuple` of parameter names.
- `spec`: A `NamedTuple` containing the component's full specification.
- `PS`: A `NamedTuple` containing prediction set data, or `nothing`.
- `N_total`: The total number of observations (training + prediction).

# Returns
- A `NamedTuple` containing the reconstructed effects (`structured`).
"""
function get_effects(m::Cyclic, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    # Extract posterior samples for sigma and raw latent field
    sigma_samples = get(chain, p_names.sigma)
    raw_samples = get(chain, p_names.raw)

    # Precomputed values
    U = spec.precomputes.U
    L = spec.precomputes.L
    noise = M.noise
    t_N = M.t_N

    # Determine indices for reconstruction (training or prediction)
    idx_to_use = isnothing(PS) ? M.t_idx : PS.t_idx
    
    # Initialize array for reconstructed effects
    reconstructed_effects = zeros(n_samples, t_N)

    # Reconstruct for each sample
    for i in 1:n_samples
        current_sigma = sigma_samples[i]
        current_raw = raw_samples[i, :]
        
        # Reconstruct latent field: sigma * U * Diagonal(1 ./ sqrt.(L .+ noise)) * raw
        reconstructed_effects[i, :] = current_sigma .* U * Diagonal(1 ./ sqrt.(L .+ noise)) * current_raw
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
