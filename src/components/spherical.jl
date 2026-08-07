# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    Spherical <: ComponentModel

A component model for a full Gaussian Process with a spherical covariance function.
The spherical kernel has compact support, meaning correlation drops to zero beyond a
specified `range`.

# Fields
- `sigma::Distribution`: The prior distribution for the marginal standard deviation of the GP.
- `range::Distribution`: The prior distribution for the effective range of spatial correlation.
"""
struct Spherical <: ComponentModel
    sigma::Distribution
    range::Distribution
end

# Add to the central component constructor registry.
COMPONENT_CONSTRUCTORS[:spherical] = (p, params) -> Spherical(p.sigma, p.range)

# Add to the model-to-structure map.
# Spherical is a continuous-space model, typically used as a smoother.
MODEL_TO_STRUCTURE_MAP[Spherical] = :smooth

"""
    get_datastructures!(m_type::Type{<:Spherical}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Spherical` component.
It ensures that coordinate variables are provided and stores them in the module data.
"""
function get_datastructures!(m_type::Type{<:Spherical}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error("The Spherical GP model requires coordinate variables, e.g., `random(x, y, model=:spherical)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Spherical GP not found in data.")
        end
    end

    # Store the coordinates matrix in the module's parameters for later use.
    mod_data[:params][:coords] = Matrix{Float64}(M[:data][!, Symbol.(variables)])

    return true
end

"""
    get_precomputes(m::Spherical, M::NamedTuple, mod_data::Dict)::NamedTuple

For the `Spherical` component, this function stores the coordinate matrix.
The full covariance matrix is constructed dynamically within the model.
"""
function get_precomputes(m::Spherical, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("Spherical component precomputes failed: coordinates not found in module data.")
    end
    
    n_latent = size(coords, 1)

    # For a full GP, the "template" is the coordinate matrix itself.
    # There is no parameter-independent precision matrix.
    return (coords=coords, n_latent=n_latent)
end

"""
    get_priors(m::Spherical, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `Spherical` component's priors.
It defines the priors for `sigma`, `range`, and the latent field `raw`.
"""
function get_priors(m::Spherical, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    sigma_prior_str = _distribution_to_string(m.sigma)
    range_prior_str = _distribution_to_string(m.range)
    
    # Latent field prior (non-centered parameterization)
    # raw ~ MvNormal(zeros(T, n_latent), I)
    
    return """
        $(p_names.sigma) ~ NamedDist($(sigma_prior_str), :$(p_names.sigma))
        $(p_names.range) ~ NamedDist($(range_prior_str), :$(p_names.range))
        $(p_names.raw) ~ NamedDist(MvNormal(zeros(T, spec_registry[:$(spec.key)].precomputes.n_latent), I), :$(p_names.raw))
    """
end

"""
    get_updates(m::Spherical, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `Spherical` GP effect.
It computes the spherical kernel matrix, performs a Cholesky decomposition, and
transforms the raw innovations to generate the latent field.
"""
function get_updates(m::Spherical, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- Spherical GP Component: $(spec.key) ---
        local coords = T.(spec_registry[:$(spec.key)].precomputes.coords)
        
        # Compute pairwise Euclidean distances
        local dist_matrix = pairwise(Euclidean(), coords, dims=1)
        
        # Compute spherical kernel matrix
        local h = dist_matrix ./ $(p_names.range)
        local K = zeros(T, size(h))
        local mask = h .< one(T)
        K[mask] = ($(p_names.sigma)^2) .* (one(T) .- T(1.5) .* h[mask] .+ T(0.5) .* h[mask].^3)
        K += (M.noise * I)
        
        local F = cholesky(Symmetric(K))
        $(p_names.latent) = F.L * $(p_names.raw)
        
        $(eta_target) .+= $(p_names.latent)
    """
end

"""
    get_effects(m::Spherical, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `Spherical` GP effect from the MCMC chain's posterior samples.
"""
function get_effects(m::Spherical, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    sigma_samples = get(chain, p_names.sigma)
    range_samples = get(chain, p_names.range)
    raw_samples = get(chain, p_names.raw)

    # Use prediction set coordinates if available, otherwise use training coordinates.
    coords_full = if isnothing(PS)
        spec.precomputes.coords
    else
        # This assumes the component struct has a field `variables` storing the coordinate column names
        # which is not standard. A better approach would be to get it from `spec.var` or `spec.params`.
        # For now, we assume `spec.var` holds a string like "s_x_s_y"
        coord_vars = Symbol.(split(spec.var, "_"))
        Matrix{Float64}(PS.data[!, coord_vars])
    end
    
    n_obs_full = size(coords_full, 1)
    noise = M.noise

    reconstructed_effects = zeros(n_samples, n_obs_full)

    dist_matrix = pairwise(Euclidean(), coords_full, dims=1)

    for i in 1:n_samples
        current_sigma = sigma_samples[i]
        current_range = range_samples[i]
        current_raw = raw_samples[i, 1:n_obs_full] # Ensure raw samples match obs dimension
        
        # Reconstruct the kernel matrix for the current sample
        h = dist_matrix ./ current_range
        K = zeros(eltype(h), size(h))
        mask = h .< 1.0
        K[mask] = (current_sigma^2) .* (1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3)
        K += (noise * I)
        
        # Perform Cholesky decomposition
        F = cholesky(Symmetric(K))
        
        # Reconstruct latent field
        reconstructed_effects[i, :] = F.L * current_raw
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    # For full GP models, the effect is already at the observation level.
    return (structured=(mean=mean_effect, lower=lower_ci, upper=upper_ci),)
end
