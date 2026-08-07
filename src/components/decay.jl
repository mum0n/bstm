# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    ExponentialDecay <: ComponentModel

A component model for a full Gaussian Process with an exponential covariance function.
The exponential kernel models correlation that decays exponentially with distance.

# Fields
- `sigma::Distribution`: The prior distribution for the marginal standard deviation of the GP.
- `lengthscale::Distribution`: The prior distribution for the characteristic lengthscale of the decay.
"""
struct ExponentialDecay <: ComponentModel
    sigma::Distribution
    lengthscale::Distribution
end

# Add to the central component constructor registry.
# Includes an alias for :decay.
COMPONENT_CONSTRUCTORS[:exponentialdecay] = (p, params) -> ExponentialDecay(p.sigma, p.lengthscale)
COMPONENT_CONSTRUCTORS[:decay] = (p, params) -> ExponentialDecay(p.sigma, p.lengthscale)

# Add to the model-to-structure map.
# ExponentialDecay is a continuous-space model, typically used as a smoother.
MODEL_TO_STRUCTURE_MAP[ExponentialDecay] = :smooth

"""
    get_datastructures!(m_type::Type{<:ExponentialDecay}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `ExponentialDecay` component.
It ensures that coordinate variables are provided and stores them in the module data.
"""
function get_datastructures!(m_type::Type{<:ExponentialDecay}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error("The ExponentialDecay model requires coordinate variables, e.g., `random(x, y, model=:decay)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for ExponentialDecay model not found in data.")
        end
    end

    # Store the coordinates matrix in the module's parameters for later use.
    mod_data[:params][:coords] = Matrix{Float64}(M[:data][!, Symbol.(variables)])

    return true
end

"""
    get_precomputes(m::ExponentialDecay, M::NamedTuple, mod_data::Dict)::NamedTuple

For the `ExponentialDecay` component, this function stores the coordinate matrix.
The full covariance matrix is constructed dynamically within the model.
"""
function get_precomputes(m::ExponentialDecay, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("ExponentialDecay component precomputes failed: coordinates not found in module data.")
    end
    
    n_latent = size(coords, 1)

    # For a full GP, the "template" is the coordinate matrix itself.
    return (coords=coords, n_latent=n_latent)
end

"""
    get_priors(m::ExponentialDecay, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `ExponentialDecay` component's priors.
It defines the priors for `sigma`, `lengthscale`, and the latent field `raw`.
"""
function get_priors(m::ExponentialDecay, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    sigma_prior_str = _distribution_to_string(m.sigma)
    ls_prior_str = _distribution_to_string(m.lengthscale)
    
    return """
        $(p_names.sigma) ~ NamedDist($(sigma_prior_str), :$(p_names.sigma))
        $(p_names.ls) ~ NamedDist($(ls_prior_str), :$(p_names.ls))
        $(p_names.raw) ~ NamedDist(MvNormal(zeros(T, spec_registry[:$(spec.key)].precomputes.n_latent), I), :$(p_names.raw))
    """
end

"""
    get_updates(m::ExponentialDecay, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `ExponentialDecay` GP effect.
It computes the exponential kernel matrix, performs a Cholesky decomposition, and
transforms the raw innovations to generate the latent field.
"""
function get_updates(m::ExponentialDecay, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- Exponential Decay GP Component: $(spec.key) ---
        local coords = T.(spec_registry[:$(spec.key)].precomputes.coords)
        
        # Compute pairwise Euclidean distances
        local dist_matrix = pairwise(Euclidean(), coords, dims=1)
        
        # Compute exponential decay kernel matrix
        local K = ($(p_names.sigma)^2) .* exp.(-dist_matrix ./ $(p_names.ls)) .+ (M.noise * I)
        
        local F = cholesky(Symmetric(K))
        local $(p_names.latent) = F.L * $(p_names.raw)
        
        $(eta_target) .+= $(p_names.latent)
    """
end

"""
    get_effects(m::ExponentialDecay, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `ExponentialDecay` GP effect from the MCMC chain's posterior samples.
"""
function get_effects(m::ExponentialDecay, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    sigma_samples = get(chain, p_names.sigma)
    ls_samples = get(chain, p_names.ls)
    raw_samples = get(chain, p_names.raw)

    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if isnothing(PS)
        spec.precomputes.coords
    else
        vcat(spec.precomputes.coords, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    end
    
    n_obs_full = size(coords_full, 1)
    noise = M.noise

    reconstructed_effects = zeros(n_samples, n_obs_full)
    dist_matrix = pairwise(Euclidean(), coords_full, dims=1)

    for i in 1:n_samples
        current_sigma = sigma_samples[i]
        current_ls = ls_samples[i]
        current_raw = raw_samples[i, 1:n_obs_full]
        
        # Reconstruct the kernel matrix for the current sample
        K = (current_sigma^2) .* exp.(-dist_matrix ./ current_ls) .+ (noise * I)
        
        F = cholesky(Symmetric(K))
        reconstructed_effects[i, :] = F.L * current_raw
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    return (structured=(mean=mean_effect, lower=lower_ci, upper=upper_ci),)
end
