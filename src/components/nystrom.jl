"""
    Nystrom <: ComponentModel

A component model for the Nyström sparse Gaussian Process approximation. This method
approximates the full GP kernel matrix with a low-rank version based on a small set
of `n_inducing` points, making it scalable for larger datasets.

# Version
v1.0.0

# Mathematical Summary
The Nyström method approximates the \$N \\times N\$ kernel matrix \$K_{XX}\$ of the data
points \$X\$ using a smaller set of \$M\$ inducing points \$Z\$. The approximation is:
\$\\tilde{K}_{XX} = K_{XZ} K_{ZZ}^{-1} K_{ZX}\$
where \$K_{XZ}\$ is the cross-covariance between data and inducing points, and \$K_{ZZ}\$
is the covariance of the inducing points.

The latent GP values \$f\$ are then modeled as a draw from a GP with this low-rank
covariance:
\$f \\sim \\mathcal{N}(0, \\tilde{K}_{XX})\$
For efficient sampling, this is implemented using a non-centered parameterization.
We first sample the latent values at the inducing points, \$u \\sim \\mathcal{N}(0, K_{ZZ})\$,
and then project them to the data points:
\$f = K_{XZ} K_{ZZ}^{-1} u\$
With a non-centered approach, we sample \$v \\sim \\mathcal{N}(0, I)\$ and set \$u = L_{ZZ}v\$,
where \$K_{ZZ} = L_{ZZ}L_{ZZ}^T\$. The final effect is computed as:
\$f = K_{XZ} (K_{ZZ}^{-1} (L_{ZZ} v)) = K_{XZ} (L_{ZZ}^T \\backslash v)\$

# Computational Methods
- `:noncentered` (Default, AD-friendly): A non-centered parameterization where the latent values
  at inducing points are constructed from standard normal innovations. Recommended
  for efficient MCMC sampling.
- `:centered` (Didactic, Not AD-friendly): A centered parameterization where the latent
  values at inducing points
  are sampled directly from their `MvNormal` distribution. This can be less efficient for MCMC.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`, `y`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `n_inducing`: `Int`, the number of inducing points. Default: `20`.
  - `kernel`: `String`, the name of the kernel function (e.g., `"se"`, `"matern32"`).
    Default: `"se"`.
  - `sigma`: `UnivariateDistribution`, prior for the marginal standard deviation of the GP.
    Default: `Exponential(1.0)`.
  - `lengthscale`: `UnivariateDistribution` or `Vector{<:UnivariateDistribution}`, prior for
    the kernel lengthscale(s). Default: `Gamma(2, 0.5)`.
  - `method`: `Symbol`, computational method (`:noncentered` or `:centered`). Default:
    `:noncentered`.
  - `knot_method`: `Symbol`, method for placing inducing points (`:kmeans`, `:random`,
    `:quantile`, `:range`). Default: `:kmeans`.

# Outputs (Parameter Names)
- `sigma_<key>`: The marginal standard deviation of the GP.
- `ls_<key>`: The kernel lengthscale(s).
- `innovations_<key>`: Raw standard normal innovations for the inducing points (for
  `:noncentered`).
- `latent_<key>`: The latent values at the inducing points (for `:centered`). The final
  effect is derived from these.
"""
struct Nystrom <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    n_inducing::Int
    kernel::String
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:nystrom] = Nystrom

COMPONENT_CONSTRUCTORS[:nystrom] = (p, params) -> Nystrom(
    p.lengthscale,
    p.sigma,
    get(params, :n_inducing, 20),
    string(get(params, :kernel, "se")),
    get(params, :method, :noncentered)
)

MODEL_TO_STRUCTURE_MAP[:nystrom] = :smooth

function get_precomputes(m::Nystrom, M::NamedTuple, mod_data::Dict)::NamedTuple
    variables = mod_data[:variables]
    params = mod_data[:params]

    if isempty(variables)
        error(
            "The Nystrom model requires coordinate variables, e.g., " *
            "`random(x, y, model=:nystrom)`."
        )
    end

    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Nystrom model not found in data.")
        end
    end

    coords = Matrix{Float64}(M.data[!, Symbol.(variables)])
    
    n_inducing = m.n_inducing
    knot_method = string(get(params, :knot_method, "kmeans"))
    Z_inducing = generate_inducing_points(coords, n_inducing; method=knot_method)

    return (
        coords=coords,
        Z_inducing=Z_inducing,
        n_latent=m.n_inducing
    )
end

function get_priors(
    m::Nystrom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = String[]
    push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")

    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors, "$(p_names.ls) ~ Product([$(ls_priors_str)])")
    else
        ls_prior_str = _distribution_to_string(m.lengthscale)
        push!(priors, "$(p_names.ls) ~ $(ls_prior_str)")
    end
    
    if m.method == :noncentered
        push!(priors, "$(p_names.ure) ~ DynamicPPL.NamedDist(MvNormal(zeros(T, m.n_inducing), I), :$(p_names.ure))")
    end

    return join(priors, "\n    ")
end

function get_updates(
    m::Nystrom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    
    common_code = """
        let
            hyper = spec_registry[:$(key)].hyper
            X_coords = hyper.coords
            Z_coords = hyper.Z_inducing
            kernel_type = Symbol("$(m.kernel)")
            
            K_UU = evaluate_kernel_matrix(
                Z_coords, $(p_names.sigma), $(p_names.ls), kernel_type, M.noise
            )
            K_XU = evaluate_cross_kernel_matrix(
                X_coords, Z_coords, $(p_names.sigma), $(p_names.ls), kernel_type
            )
    """

    noncentered_code = """
        # --- Nystrom Sparse GP (Non-Centered): $(key) ---
        $(common_code)
            L_UU = cholesky(Symmetric(K_UU)).L
            $(p_names.sre) = K_XU * (L_UU' \\ $(p_names.ure))
            $(eta_target) = $(eta_target) .+ $(p_names.sre)
        end
    """

    centered_code = """
        # --- Nystrom Sparse GP (Centered): $(key) ---
        $(common_code)
            $(p_names.sre) ~ MvNormal(zeros(T, $(m.n_inducing)), Symmetric(K_UU))
            nystrom_effect = K_XU * (K_UU \\ $(p_names.sre))
            $(eta_target) = $(eta_target) .+ nystrom_effect
        end
    """

    if m.method == :noncentered
        return noncentered_code
    elseif m.method == :centered
        return centered_code
    else
        error("Unsupported method '$(m.method)' for Nystrom component.")
    end
end

"""
    get_effects(m::Nystrom, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the Nyström sparse GP effect from posterior samples. This version is
CPU-only and uses modern chain accessors.
"""
function get_effects(
    m::Nystrom, chain, spec::NamedTuple, M::NamedTuple,
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
    
    hyper = spec.hyper
    noise = M.noise
    kernel_type = Symbol(m.kernel)

    # --- Coordinate and Inducing Point Handling ---
    coords_train = hyper.coords # Training coordinates
    Z_inducing = hyper.Z_inducing # Inducing points
    
    # Combine training and prediction coordinates
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data,
        Symbol(v)) for v in coord_vars) # If prediction set is provided
        coords_pred = Matrix{Float64}(PS.data[!,
            Symbol.(coord_vars)]) # Extract prediction coordinates
        vcat(coords_train, coords_pred) # Combine training and prediction coordinates
    else
        coords_train # Otherwise, use only training coordinates
    end
    n_obs_full = size(coords_full, 1) # Total number of observations (training + prediction)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        
        # Find parameter names in the MCMC chain
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        ls_name = _find_parameter(p_names, string(p_names_k.ls), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(ls_name)
            @warn "Parameters for Nystrom component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
            continue
        end

        # Extract posterior samples (CPU)
        sigma_samples = get_params_vector(chain, sigma_name, 1) # (n_samples, 1)
        ls_dim = m.lengthscale isa Vector ? length(m.lengthscale) : 1 # Dimension of lengthscale parameter
        ls_samples = get_params_matrix(chain, ls_name, ls_dim) # (n_samples, ls_dim)

        # Initialize the output matrix for the full effect
        effect_k_matrix = zeros(Float64, n_obs_full, n_samples)

        # --- Sample-wise Reconstruction ---
        if m.method == :noncentered
            ure_name = _find_parameter(p_names, string(p_names_k.ure), k, is_multivariate_model)
            if isempty(ure_name)
                @warn "ure for Nystrom component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
                continue
            end
            ure_samples = get_params_matrix(chain, ure_name, m.n_inducing) # (n_samples,
                n_inducing)

            for i in 1:n_samples
                current_sigma = sigma_samples[i, 1] # Sigma for current sample
                current_ls = ls_dim > 1 ? ls_samples[i, :] : ls_samples[i, 1] # Lengthscale for current sample
                
                # Kernel evaluations and linear algebra
                K_UU = evaluate_kernel_matrix(Z_inducing, current_sigma, current_ls,
                    kernel_type, noise)
                K_XU = evaluate_cross_kernel_matrix(coords_full, Z_inducing, current_sigma,
                    current_ls, kernel_type)
                
                L_UU = cholesky(Symmetric(K_UU)).L
                u_latent = L_UU * ure_samples[i, :]
                effect_k_matrix[:, i] = K_XU * (K_UU \ u_latent)
            end
        else # :centered
            sre_name = _find_parameter(p_names, string(p_names_k.sre), k, is_multivariate_model)
            if isempty(sre_name)
                @warn "sre for Nystrom component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
                continue
            end
            u_latent_samples = get_params_matrix(chain, sre_name, m.n_inducing) # (n_samples,
                n_inducing)

            for i in 1:n_samples # Iterate over each posterior sample
                current_sigma = sigma_samples[i, 1] # Sigma for current sample
                current_ls = ls_dim > 1 ? ls_samples[i, :] : ls_samples[i, 1] # Lengthscale for current sample
                
                # Kernel evaluations and linear algebra
                K_UU = evaluate_kernel_matrix(Z_inducing, current_sigma, current_ls,
                    kernel_type, noise)
                K_XU = evaluate_cross_kernel_matrix(coords_full, Z_inducing, current_sigma,
                    current_ls, kernel_type)
                
                effect_k_matrix[:, i] = K_XU * (K_UU \ u_latent_samples[i, :])
            end
        end
        
        push!(structured_effects, effect_k_matrix)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end 
