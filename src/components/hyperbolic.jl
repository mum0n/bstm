"""
    Hyperbolic <: ComponentModel

A component for a non-Euclidean Gaussian Process that operates in hyperbolic space,
specifically the Poincaré disk model. This allows for modeling data with hierarchical
or tree-like structures, where the notion of distance is warped.

# Version
v1.0.0

# Mathematical Summary
This component models a latent field \\(f(s)\\) as a draw from a Gaussian Process where
the covariance is a function of the hyperbolic distance between points, not the
Euclidean distance.

1.  **Poincaré Disk**: Input coordinates \$\\mathbf{s}\$ are assumed to lie within the
    unit disk.
2.  **Hyperbolic Distance**: The distance \$d_c(u, v)\$ between two points \$u, v\$ in
    the Poincaré disk with curvature \$c < 0\$ is given by:
    \$d_c(u, v) = \\frac{1}{\\sqrt{-c}} \\text{arccosh}\\left(1 + 2
      \\frac{\\|u-v\\|^2}{(1-\\|u\\|^2)(1-\\|v\\|^2)}\\right)\$
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

# Key References
- Nickel, M., & Kiela, D. (2017). *Poincaré embeddings for learning hierarchical
  representations*. Advances in neural information processing systems, 30.
- Wikipedia: Poincaré disk model
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

function get_precomputes(m::Hyperbolic, M::NamedTuple, mod_data::Dict)::NamedTuple
    variables = mod_data[:variables]
    if isempty(variables)
        error("The Hyperbolic model requires coordinate variables, e.g., " *
              "`random(x, y, model=:hyperbolic)`.")
    end

    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Hyperbolic model not found in data.")
        end
    end

    coords = Matrix{Float64}(M.data[!, Symbol.(variables)])
    
    # Validate that coordinates are within the unit disk
    if any(sum(coords.^2, dims=2) .>= 1.0)
        @warn "Some coordinates for the Hyperbolic model lie on or outside the unit " *
              "disk. The model assumes coordinates are strictly inside the disk " *
              "(||s|| < 1). Results may be unstable."
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
    push!(priors, "$(p_names.ure) ~ MvNormal(zeros(T, spec.hyper.n_latent), I)")
    
    return join(priors, "\n    ")
end
 

function _poincare_dist_sq(u, v, curvature)
    c = sqrt(max(0.0, -curvature))
    if c == 0.0
        return sum((u .- v).^2)
    end # Fallback to Euclidean
    
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
    T = eltype(coords)
    n = size(coords, 1)
    c = sqrt(max(zero(T), -T(curvature)))

    if c == 0.0 # Fallback to Euclidean distance with SE kernel
        sq_dists = sum(coords.^2, dims=2) .+ sum(coords.^2, dims=2)' .- 2 * (coords * coords')
        K = sigma^2 .* exp.(-0.5 .* sq_dists)
        return K + T(noise) * I
    end

    # Calculate squared norms for all points
    norm_sq = sum(coords.^2, dims=2)

    # Pairwise squared Euclidean distances
    diff_sq = norm_sq .+ norm_sq' .- 2 * (coords * coords')
    
    # Denominator for hyperbolic distance formula, with epsilon for stability
    denom = (1.0 .- norm_sq .+ T(1e-9)) * (1.0 .- norm_sq' .+ T(1e-9))

    # Argument for acosh, computed element-wise
    arg = 1.0 .+ 2.0 .* diff_sq ./ denom

    # Compute squared hyperbolic distance matrix
    # acosh.(max.(one(T), arg)) ensures the argument is >= 1
    dist_sq = (acosh.(max.(one(T), arg)) ./ c).^2

    # Squared exponential kernel on hyperbolic distances
    K = sigma^2 .* exp.(-0.5 .* dist_sq)
    
    return K + T(noise) * I
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
            $(p_names.sre) = F_gp.L * $(p_names.ure)
            
            $(eta_target) = $(eta_target) .+ $(p_names.sre)
        end
    """
end


function get_effects(
    m::Hyperbolic, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = if occursin("FlexiChain", string(typeof(chain)))
        size(chain, 1) * FlexiChains.nchains(chain)
    else
        size(chain, 1) * size(chain, 3)
    end
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    # --- Coordinate Handling: Combine training and prediction sets on CPU ---
    coord_vars = get(spec.params, :positional_args, [])
    coords_train = spec.hyper.coords
    # Combine training and prediction coordinates
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data,
        Symbol(v)) for v in coord_vars) # If prediction set is provided
        coords_pred = Matrix{Float64}(PS.data[!,
            Symbol.(coord_vars)]) # Extract prediction coordinates
        vcat(coords_train, coords_pred) # Combine training and prediction coordinates
    else
        coords_train # Otherwise, use only training coordinates
    end
    n_obs_train = size(coords_train, 1) # Number of observations in training data
    n_obs_full = size(coords_full, 1) # Total number of observations (training + prediction)

    noise = M.noise
    curvature = m.curvature # Curvature parameter
    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        ure_name = _find_parameter(p_names, string(p_names_k.ure), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(ure_name)
            @warn "Parameters for Hyperbolic component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        sigma_samples = get_params_vector(chain, sigma_name, 1) # (n_samples, 1)
        ure_samples = get_params_matrix(chain, ure_name, n_obs_train) # (n_samples, n_obs_train)

        # Initialize the output matrix for the full effect
        effect_k_matrix = zeros(Float64, n_obs_full, n_samples)

        # --- Sample-wise Reconstruction ---
        for i in 1:n_samples # Iterate over each posterior sample
            # Kernel evaluation and Cholesky
            K_mat = _evaluate_hyperbolic_kernel_matrix(coords_full, sigma_samples[i, 1],
                curvature, noise)
            F = cholesky(Symmetric(K_mat))

            # Combine training innovations with new innovations for prediction points (if any)
            innov_train = ure_samples[i, :]
            innov_i = if n_obs_full > n_obs_train
                innov_pred = randn(Float64, n_obs_full - n_obs_train)
                vcat(innov_train, innov_pred)
            else
                innov_train
            end

            effect_k_matrix[:, i] = F.L * innov_i
        end
        
        push!(structured_effects, effect_k_matrix)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end 
