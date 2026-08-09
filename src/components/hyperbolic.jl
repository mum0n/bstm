
"""
    Hyperbolic <: ComponentModel

A component model for a full Gaussian Process on a hyperbolic space (the Poincaré
disk). This is useful for modeling data with hierarchical or tree-like structures
where Euclidean distance is not an appropriate metric.

# Version
v1.0.0 (2026-08-08)

# Mathematical Summary
The component models a latent field \$f(x)\$ as a draw from a Gaussian Process with
a zero mean and a kernel defined on the hyperbolic distance:
\$f(x) \\sim \\mathcal{GP}(0, k(d_h(x, x')))\$
where \$d_h(x, x')\$ is the geodesic distance between points \$x\$ and \$x'\$ in the
Poincaré disk model of hyperbolic space. The distance is given by:
\$d_h(u, v) = \\text{arccosh} \\left( 1 + 2 \\frac{\\|u-v\\|^2}{(1-\\|u\\|^2)(1-\\|v\\|^2)} \\right)\$
The kernel is a standard squared exponential kernel applied to this distance:
\$k(d_h) = \\sigma^2 \\exp\\left(-\frac{d_h^2}{2\\ell^2}\\right)\$
In this implementation, the lengthscale \$\\ell\$ is implicitly tied to the space's
curvature, \$\\ell = \\sqrt{-\\text{curvature}}\$.

# Assumptions
- The input coordinates are 2-dimensional and lie within the unit disk. The
  component will automatically scale coordinates to fit if they are outside.
- The underlying process is stationary in the hyperbolic space.

# Best Use Case
Modeling hierarchical data, such as taxonomies, social networks, or phylogenetic
trees, where the notion of distance is non-Euclidean. It is also useful for data
where correlation decays based on path-like or tree-like structures rather than
straight-line distance.

# Key References
- Nickel, M., & Kiela, D. (2017). *Poincaré Embeddings for Learning Hierarchical
  Representations*. In NIPS.
- Borovitskiy, V., Azangulov, A., & Mostowsky, P. (2020). *Hyperbolic Gaussian
  Processes*. In NeurIPS.
- Wikipedia: Poincaré disk model

# Fields
- `curvature::Float64`: The curvature of the hyperbolic space (must be negative).
- `sigma::Distribution`: The prior for the marginal standard deviation of the GP.
"""
struct Hyperbolic <: ComponentModel
    curvature::Float64
    sigma::Distribution
end

COMPONENT_TYPE_REGISTRY[:hyperbolic] = Hyperbolic

COMPONENT_CONSTRUCTORS[:hyperbolic] = (p, params) -> Hyperbolic(
    get(params, :curvature, -1.0), p.sigma
)

MODEL_TO_STRUCTURE_MAP[:hyperbolic] = :smooth

"""
    get_datastructures!(m_type::Type{<:Hyperbolic}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Hyperbolic` component. It ensures that
coordinate variables are provided and scales them to fit within the unit disk,
which is a requirement for the Poincaré disk model.
"""
function get_datastructures!(
    m_type::Type{<:Hyperbolic}, M::Dict, mod_data::Dict
)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error(
            "The Hyperbolic GP model requires coordinate variables, e.g., " *
            "`random(x, y, model=:hyperbolic)`."
        )
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Hyperbolic GP not found in data.")
        end
    end

    coords = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    
    norms_sq = sum(coords.^2, dims=2)
    if any(norms_sq .>= 1.0)
        @warn "Some coordinates for the Hyperbolic GP are outside the unit disk. " *
              "They will be scaled to fit."
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
        error(
            "Hyperbolic component precomputes failed: coordinates not found in " *
            "module data."
        )
    end
    
    n_latent = size(coords, 1)

    return (coords=coords, n_latent=n_latent)
end

"""
    get_priors(m::Hyperbolic, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code string for the `Hyperbolic` component's priors.
It defines the priors for `sigma` and the `raw` latent field.
"""
function get_priors(
    m::Hyperbolic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    return """
        $(p_names.sigma) ~ $(sigma_prior_str)
        $(p_names.raw) ~ MvNormal(
            zeros(T, spec_registry[:$(spec.key)].precomputes.n_latent), I
        )
    """
end

"""
    evaluate_hyperbolic_kernel_matrix(coords, sigma, curvature, noise)

A helper function to compute the kernel matrix based on hyperbolic distances in
the Poincaré disk. This function is intended to be available in the model's
execution scope.
"""
function evaluate_hyperbolic_kernel_matrix(
    coords::AbstractMatrix, sigma::Real, curvature::Real, noise::Real
)
    n = size(coords, 1)
    T = promote_type(eltype(coords), typeof(sigma), typeof(curvature), typeof(noise))
    K = zeros(T, n, n)
    
    norms_sq = sum(coords.^2, dims=2)
    one_minus_norms_sq = one(T) .- norms_sq

    for i in 1:n
        for j in i:n
            dist_sq_euclidean = sum((coords[i,:] .- coords[j,:]).^2)
            
            denominator = one_minus_norms_sq[i] * one_minus_norms_sq[j]
            arg_acosh = one(T) + (2 * dist_sq_euclidean) / (denominator + 1e-9)
            
            dist_poincare = acosh(arg_acosh)
            
            dist_hyperbolic = dist_poincare / sqrt(max(-curvature, 1e-9))
            
            kernel_val = sigma^2 * exp(-0.5 * dist_hyperbolic^2)
            
            K[i, j] = kernel_val
            if i != j
                K[j, i] = kernel_val
            end
        end
    end
    
    K += (noise * I)
    return K
end

"""
    get_updates(m::Hyperbolic, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code string for constructing the `Hyperbolic` GP effect.
It calls a helper function to compute the hyperbolic kernel matrix.
"""
function get_updates(
    m::Hyperbolic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- Hyperbolic GP Component: $(spec.key) ---
        let
            local coords = spec_registry[:$(spec.key)].precomputes.coords
            local curvature = $(m.curvature)
            
            local K = evaluate_hyperbolic_kernel_matrix(
                coords, $(p_names.sigma), curvature, M.noise
            )
            
            local F = cholesky(Symmetric(K))
            local $(p_names.latent) = F.L * $(p_names.raw)
            
            $(eta_target) .+= $(p_names.latent)
        end
    """
end

"""
    get_effects(m::Hyperbolic, chain, M::NamedTuple, ...)

Reconstructs the `Hyperbolic` GP effect from the MCMC chain's posterior samples.
"""
function get_effects(
    m::Hyperbolic, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
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

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        raw_samples = get_params_vector(
            chain, string(p_names.raw), spec.precomputes.n_latent
        )

        effect_k = zeros(Float64, n_obs_full, n_samples)

        for i in 1:n_samples
            current_sigma = sigma_samples[i]
            
            K = evaluate_hyperbolic_kernel_matrix(
                coords_full, current_sigma, curvature, noise
            )
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
