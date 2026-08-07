# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    Hyperbolic <: ComponentModel

A component model for a full Gaussian Process on a hyperbolic space (Poincaré disk).
This is useful for modeling data with hierarchical or tree-like structures where
Euclidean distance is not appropriate.

# Fields
- `curvature::Float64`: The curvature of the hyperbolic space (must be negative).
- `sigma::Distribution`: The prior distribution for the marginal standard deviation of the GP.
"""
struct Hyperbolic <: ComponentModel
    curvature::Float64
    sigma::Distribution
end

# Add to the central component constructor registry.
COMPONENT_CONSTRUCTORS[:hyperbolic] = (p, params) -> Hyperbolic(get(params, :curvature, -1.0), p.sigma)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[Hyperbolic] = :smooth

"""
    get_datastructures!(m_type::Type{<:Hyperbolic}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Hyperbolic` component. It ensures that
coordinate variables are provided and scales them to fit within the unit disk,
which is a requirement for the Poincaré disk model.
"""
function get_datastructures!(m_type::Type{<:Hyperbolic}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error("The Hyperbolic GP model requires coordinate variables, e.g., `random(x, y, model=:hyperbolic)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Hyperbolic GP not found in data.")
        end
    end

    # Store the coordinates matrix in the module's parameters for later use.
    coords = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    
    # The coordinates must be within the unit disk for the Poincaré model.
    # We scale them if they are not.
    norms_sq = sum(coords.^2, dims=2)
    if any(norms_sq .>= 1.0)
        @warn "Some coordinates for the Hyperbolic GP are outside the unit disk. They will be scaled to fit."
        max_norm = maximum(sqrt.(norms_sq))
        coords ./= (max_norm + 1e-6)
    end
    mod_data[:params][:coords] = coords

    return true
end

"""
    get_precomputes(m::Hyperbolic, M::NamedTuple, mod_data::Dict)::NamedTuple

For the `Hyperbolic` component, this function stores the (potentially scaled)
coordinate matrix for use by the code generator.
"""
function get_precomputes(m::Hyperbolic, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("Hyperbolic component precomputes failed: coordinates not found in module data.")
    end
    
    n_latent = size(coords, 1)

    return (coords=coords, n_latent=n_latent)
end

"""
    get_priors(m::Hyperbolic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `Hyperbolic` component's priors.
It defines the priors for `sigma` and the `raw` latent field.
"""
function get_priors(m::Hyperbolic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    return """
        $(p_names.sigma) ~ NamedDist($(sigma_prior_str), :$(p_names.sigma))
        $(p_names.raw) ~ NamedDist(MvNormal(zeros(T, spec_registry[:$(spec.key)].precomputes.n_latent), I), :$(p_names.raw))
    """
end

"""
    evaluate_hyperbolic_kernel_matrix(coords::AbstractMatrix{T}, sigma::Real, curvature::Real, noise::Real) where T

A helper function to compute the kernel matrix based on hyperbolic distances in the Poincaré disk.
This function is intended to be available in the model's execution scope.
"""
function evaluate_hyperbolic_kernel_matrix(coords::AbstractMatrix{T}, sigma::Real, curvature::Real, noise::Real) where T
    n = size(coords, 1)
    K = zeros(T, n, n)
    
    # Precompute norms squared for efficiency
    norms_sq = sum(coords.^2, dims=2)

    one_minus_norms_sq = one(T) .- norms_sq

    for i in 1:n
        for j in i:n
            dist_sq_euclidean = sum((coords[i,:] .- coords[j,:]).^2)
            
            # Poincaré distance formula
            denominator = one_minus_norms_sq[i] * one_minus_norms_sq[j]
            arg_acosh = one(T) + (2 * dist_sq_euclidean) / (denominator + T(1e-9))
            
            dist_poincare = acosh(arg_acosh)
            
            # Scale distance by curvature
            dist_hyperbolic = dist_poincare / sqrt(max(-T(curvature), T(1e-9)))
            
            # Squared exponential kernel on the hyperbolic distance
            kernel_val = sigma^2 * exp(-T(0.5) * dist_hyperbolic^2)
            
            K[i, j] = kernel_val
            if i != j
                K[j, i] = kernel_val
            end
        end
    end
    
    K += (T(noise) * I)
    return K
end

"""
    get_updates(m::Hyperbolic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `Hyperbolic` GP effect.
It calls a helper function to compute the hyperbolic kernel matrix.
"""
function get_updates(m::Hyperbolic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- Hyperbolic GP Component: $(spec.key) ---
        local coords = T.(spec_registry[:$(spec.key)].precomputes.coords)
        local curvature = T($(m.curvature))
        
        # This helper function is assumed to be available in the model's scope
        local K = evaluate_hyperbolic_kernel_matrix(coords, $(p_names.sigma), curvature, M.noise)
        
        local F = cholesky(Symmetric(K))
        local $(p_names.latent) = F.L * $(p_names.raw)
        
        $(eta_target) .+= $(p_names.latent)
    """
end

"""
    get_effects(m::Hyperbolic, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `Hyperbolic` GP effect from the MCMC chain's posterior samples.
"""
function get_effects(m::Hyperbolic, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    sigma_samples = get(chain, p_names.sigma)
    raw_samples = get(chain, p_names.raw)

    coords_train = spec.precomputes.coords
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        coords_train
    end
    
    n_obs_full = size(coords_full, 1)
    noise = M.noise
    curvature = m.curvature

    reconstructed_effects = zeros(n_samples, n_obs_full)

    for i in 1:n_samples
        current_sigma = sigma_samples[i]
        current_raw = raw_samples[i, 1:n_obs_full]
        
        K = evaluate_hyperbolic_kernel_matrix(coords_full, current_sigma, curvature, noise)
        
        F = cholesky(Symmetric(K))
        reconstructed_effects[i, :] = F.L * current_raw
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    return (structured=(mean=mean_effect, lower=lower_ci, upper=upper_ci),)
end
