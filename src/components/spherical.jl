
"""
    Spherical <: ComponentModel

A component model for a full Gaussian Process with a spherical covariance function.
The spherical kernel has compact support, meaning correlation drops to zero beyond a
specified `range`.

# Version
v1.0.0 (2026-08-08)

# Mathematical Summary
The component models a latent field \$f(s)\$ as a draw from a Gaussian Process with
a zero mean and a spherical covariance function:
\$f(s) \\sim \\mathcal{GP}(0, k(h))\$
where \$h = \\|s - s'\\|\$ is the Euclidean distance between points. The spherical
kernel is defined as:
\$k(h) = \\sigma^2 \\left( 1 - \\frac{3h}{2r} + \\frac{h^3}{2r^3} \\right)\$ for \$0 \\le h \\le r\$
\$k(h) = 0\$ for \$h > r\$
where:
- \$\\sigma^2\$ is the marginal variance (sill).
- \$r\$ is the range parameter.

# Assumptions
- The underlying process is stationary.
- The spatial correlation ceases to exist beyond a finite distance.

# Best Use Case
Modeling spatial processes where influence is strictly local, such as the effect
of a point source of pollution that dissipates completely after a certain distance,
or certain ecological processes with a defined territorial range.

# Key References
- Cressie, N. A. C. (1993). *Statistics for spatial data*. Wiley.
- Wikipedia: Variogram (which discusses the spherical model).

# Fields
- `sigma::Distribution`: The prior for the marginal standard deviation of the GP.
- `range::Distribution`: The prior for the effective range of spatial correlation.
"""
struct Spherical <: ComponentModel
    sigma::Distribution
    range::Distribution
end

COMPONENT_TYPE_REGISTRY[:spherical] = Spherical
COMPONENT_CONSTRUCTORS[:spherical] = (p, params) -> Spherical(p.sigma, p.range)
MODEL_TO_STRUCTURE_MAP[:spherical] = :smooth

"""
    get_datastructures!(m_type::Type{<:Spherical}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Spherical` component.
It ensures that coordinate variables are provided and stores them in the module data.
"""
function get_datastructures!(
    m_type::Type{<:Spherical}, M::Dict, mod_data::Dict
)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error(
            "The Spherical GP model requires coordinate variables, e.g., " *
            "`random(x, y, model=:spherical)`."
        )
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Spherical GP not found in data.")
        end
    end

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
        error("Spherical component precomputes failed: coordinates not found.")
    end
    
    n_latent = size(coords, 1)
    return (coords=coords, n_latent=n_latent)
end

"""
    get_priors(m::Spherical, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `sigma`, `range`, and the `raw` innovations.
"""
function get_priors(
    m::Spherical, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    sigma_prior_str = _distribution_to_string(m.sigma)
    range_prior_str = _distribution_to_string(m.range)
    
    return """
        $(p_names.sigma) ~ $(sigma_prior_str)
        $(p_names.range) ~ $(range_prior_str)
        $(p_names.raw) ~ MvNormal(zeros(spec.precomputes.n_latent), I)
    """
end

"""
    evaluate_spherical_kernel_matrix(coords, sigma, range_param, noise)

A helper function to compute the spherical kernel matrix. This function is
intended to be available in the model's execution scope.
"""
function evaluate_spherical_kernel_matrix(
    coords::AbstractMatrix, sigma::Real, range_param::Real, noise::Real
)
    T = promote_type(eltype(coords), typeof(sigma), typeof(range_param), typeof(noise))
    dist_matrix = pairwise(Euclidean(), coords', dims=2)
    
    h = dist_matrix ./ range_param
    K = zeros(T, size(h))
    mask = h .< 1.0
    
    K[mask] = (sigma^2) .* (1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3)
    K += (noise * I)
    
    return K
end

"""
    get_updates(m::Spherical, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to compute the spherical kernel matrix, perform a Cholesky
decomposition, and sample the latent field using a non-centered parameterization.
"""
function get_updates(
    m::Spherical, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- Spherical GP Component: $(spec.key) ---
        let
            local coords = spec_registry[:$(spec.key)].precomputes.coords
            
            local K = evaluate_spherical_kernel_matrix(
                coords, $(p_names.sigma), $(p_names.range), M.noise
            )
            
            local F = cholesky(Symmetric(K))
            local $(p_names.latent) = F.L * $(p_names.raw)
            
            $(eta_target) .+= $(p_names.latent)
        end
    """
end

"""
    get_effects(m::Spherical, chain, M::NamedTuple, ...)

Reconstructs the `Spherical` GP effect from the MCMC chain's posterior samples.
"""
function get_effects(
    m::Spherical, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    coord_vars = get(spec.params, :positional_args, [])
    coords_train = spec.precomputes.coords
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        coords_train
    end
    n_obs_full = size(coords_full, 1)
    noise = M.noise

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        range_samples = get_params_vector(chain, string(p_names.range), 1)[:, 1]
        raw_samples = get_params_vector(
            chain, string(p_names.raw), spec.precomputes.n_latent
        )

        effect_k = zeros(Float64, n_obs_full, n_samples)
        dist_matrix = pairwise(Euclidean(), coords_full', dims=2)

        for i in 1:n_samples
            current_sigma = sigma_samples[i]
            current_range = range_samples[i]
            
            h = dist_matrix ./ current_range
            K = zeros(Float64, size(h))
            mask = h .< 1.0
            K[mask] = (current_sigma^2) .* (1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3)
            K += (noise * I)
            
            F = cholesky(Symmetric(K))
            
            raw_i = if size(raw_samples, 2) == n_obs_full
                raw_samples[i, :]
            else
                vcat(raw_samples[i, :], randn(n_obs_full - size(raw_samples, 2)))
            end
            
            effect_k[:, i] = F.L * raw_i
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
