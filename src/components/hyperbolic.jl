"""
    Hyperbolic <: ComponentModel

A component for a non-Euclidean Gaussian Process that operates in hyperbolic space,
specifically the Poincaré disk model. This allows for modeling data with hierarchical
or tree-like structures, where the notion of distance is warped.

# Version
v1.0.0 (2026-08-12)

# Mathematical Summary
This component models a latent field \$f(s)\$ as a draw from a Gaussian Process where
the covariance is a function of the hyperbolic distance between points, not the
Euclidean distance.

1.  **Poincaré Disk**: Input coordinates \$\\mathbf{s}\$ are assumed to lie within the
    unit disk.
2.  **Hyperbolic Distance**: The distance \$d_c(u, v)\$ between two points \$u, v\$ in
    the Poincaré disk with curvature \$c < 0\$ is given by:
    \$d_c(u, v) = \\frac{1}{\\sqrt{-c}} \\text{arccosh}\\left(1 + 2 \\frac{\\|u-v\\|^2}{(1-\\|u\\|^2)(1-\\|v\\|^2)}\\right)\$
3.  **Kernel**: A standard kernel, such as the Squared Exponential, is applied to this
    hyperbolic distance:
    \$k(u, v) = \\sigma^2 \\exp\\left(-\\frac{d_c(u,v)^2}{2}\\right)\$
    (assuming a fixed lengthscale of 1).
4.  **Latent Field**: The latent field \$\\boldsymbol{\\phi}\$ is then sampled from the resulting
    multivariate normal distribution:
    \$\\boldsymbol{\\phi} \\sim \\mathcal{N}(\\mathbf{0}, K)\$
    where \$K\$ is the dense covariance matrix evaluated at all data points using the
    hyperbolic kernel.

# Computational Methods
- `:noncentered` (Default, AD-friendly): Samples standard normal innovations and
  transforms them using the Cholesky factor of the covariance matrix. Recommended
  for gradient-based samplers like NUTS.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`, `y`) passed to `random()`. The coordinates
    should be scaled to lie within the unit disk (i.e., \$\\|s\\| < 1\$).
- **Optional (in `random()` call)**:
  - `sigma`: `UnivariateDistribution`, prior for the marginal standard deviation of the GP.
    Default: `Exponential(1.0)`.
  - `curvature`: `Real`, a fixed negative value for the space's curvature. Default: `-1.0`.

# Outputs (Parameter Names)
- `sigma_<key>`: The marginal standard deviation of the GP.
- `innovations_<key>`: The raw standard normal innovations for the latent field.
- `latent_<key>`: The reconstructed latent hyperbolic GP effect.
"""
struct Hyperbolic <: ComponentModel
    curvature::Real
    sigma::UnivariateDistribution
end

COMPONENT_TYPE_REGISTRY[:hyperbolic] = Hyperbolic

COMPONENT_CONSTRUCTORS[:hyperbolic] = (p, params) -> Hyperbolic(
    get(params, :curvature, -1.0),
    p.sigma
)

MODEL_TO_STRUCTURE_MAP[:hyperbolic] = :smooth

function get_datastructures!(m_type::Type{<:Hyperbolic}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]
    if isempty(variables)
        error("The Hyperbolic model requires coordinate variables, e.g., `random(x, y, model=:hyperbolic)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Hyperbolic model not found in data.")
        end
    end

    coords = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    
    # Validate that coordinates are within the unit disk
    if any(sum(coords.^2, dims=2) .>= 1.0)
        @warn "Some coordinates for the Hyperbolic model lie on or outside the unit disk. The model assumes coordinates are strictly inside the disk (||s|| < 1). Results may be unstable."
    end

    mod_data[:params][:coords] = coords
    return true
end

function get_precomputes(m::Hyperbolic, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("Hyperbolic component precomputes failed: coordinates not found.")
    end
    
    n_latent = size(coords, 1)
    return (coords=coords, n_latent=n_latent)
end

function get_priors(
    m::Hyperbolic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = String[]
    push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")
    push!(priors, "$(p_names.innovations) ~ MvNormal(zeros(T, spec.hyper.n_latent), I)")

    return join(priors, "\n    ")
end

# These helper functions must be available in the scope where the Turing model is evaluated.
# They are defined here and assumed to be loaded into the `bstm` module.

function _poincare_dist_sq(u, v, curvature)
    c = sqrt(max(0.0, -curvature))
    if c == 0.0; return sum((u .- v).^2); end # Fallback to Euclidean
    
    norm_u_sq = sum(u.^2)
    norm_v_sq = sum(v.^2)
    
    # Add a small epsilon to prevent division by zero if points are on the boundary
    denom = (1.0 - norm_u_sq + 1e-9) * (1.0 - norm_v_sq + 1e-9)
    diff_sq = sum((u .- v).^2)
    
    arg = 1.0 + 2.0 * diff_sq / denom
    
    # Hyperbolic distance d(u,v)
    dist = acosh(max(1.0, arg)) / c
    
    return dist^2
end

function _evaluate_hyperbolic_kernel_matrix(coords, sigma, curvature, noise)
    n = size(coords, 1)
    T = eltype(coords)
    K = zeros(T, n, n)
    
    for i in 1:n
        for j in i:n
            dist_sq = _poincare_dist_sq(view(coords, i, :), view(coords, j, :), curvature)
            # Using a squared exponential kernel with fixed lengthscale of 1
            K[i, j] = K[j, i] = sigma^2 * exp(-0.5 * dist_sq)
        end
    end
    return K + (noise * I)
end

function get_updates(
    m::Hyperbolic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    
    return """
        # --- Hyperbolic GP Component: $(key) ---
        let
            coords = spec_registry[:$(key)].hyper.coords
            curvature = $(m.curvature)
            
            K_mat = _evaluate_hyperbolic_kernel_matrix(
                coords, $(p_names.sigma), curvature, M.noise
            )
            
            F_gp = cholesky(Symmetric(K_mat))
            $(p_names.latent) = F_gp.L * $(p_names.innovations)
            
            $(eta_target) .+= $(p_names.latent)
        end
    """
end

function get_effects(
    m::Hyperbolic, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    coord_vars = get(spec.params, :positional_args, [])
    coords_train = spec.hyper.coords
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        coords_train
    end
    n_obs_train = size(coords_train, 1)
    n_obs_full = size(coords_full, 1)
    
    noise = M.noise
    curvature = m.curvature
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))

    for k in 1:outcomes_N
        sigma_name = _find_parameter(p_names_vec, string(spec.key), "sigma", k, is_multivariate_model)
        innovations_name = _find_parameter(p_names_vec, string(spec.key), "innovations", k, is_multivariate_model)

        if isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for Hyperbolic component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innovations_name, n_obs_train)

        effect_k = zeros(Float64, n_obs_full, n_samples)

        for i in 1:n_samples
            K_mat = _evaluate_hyperbolic_kernel_matrix(coords_full, sigma_samples[i], curvature, noise)
            F = cholesky(Symmetric(K_mat))
            innov_i = vcat(innovations_samples[i, :], randn(n_obs_full - n_obs_train))
            effect_k[:, i] = F.L * innov_i
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
