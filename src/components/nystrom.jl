"""
    Nystrom <: ComponentModel

A component model for the Nyström sparse Gaussian Process approximation. This method
approximates the full GP kernel matrix with a low-rank version based on a small set
of `n_inducing` points, making it scalable for larger datasets.

# Version
v1.1.1 (2026-08-14)

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
- `:centered` (Didactic, Not AD-friendly): A centered parameterization where the latent values at inducing points
  are sampled directly from their `MvNormal` distribution. This can be less efficient for MCMC.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`, `y`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `n_inducing`: `Int`, the number of inducing points. Default: `20`.
  - `kernel`: `String`, the name of the kernel function (e.g., `"se"`, `"matern32"`). Default: `"se"`.
  - `sigma`: `UnivariateDistribution`, prior for the marginal standard deviation of the GP. Default: `Exponential(1.0)`.
  - `lengthscale`: `UnivariateDistribution` or `Vector{<:UnivariateDistribution}`, prior for the kernel lengthscale(s). Default: `Gamma(2, 0.5)`.
  - `method`: `Symbol`, computational method (`:noncentered` or `:centered`). Default: `:noncentered`.
  - `knot_method`: `Symbol`, method for placing inducing points (`:kmeans`, `:random`, `:quantile`, `:range`). Default: `:kmeans`.

# Outputs (Parameter Names)
- `sigma_<key>`: The marginal standard deviation of the GP.
- `ls_<key>`: The kernel lengthscale(s).
- `innovations_<key>`: Raw standard normal innovations for the inducing points (for `:noncentered`).
- `latent_<key>`: The latent values at the inducing points (for `:centered`). The final effect is derived from these.
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
        push!(priors, "$(p_names.innovations) ~ DynamicPPL.NamedDist(MvNormal(zeros(T, m.n_inducing), I), :$(p_names.innovations))")
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
        local hyper = spec_registry[:$(key)].hyper
        local X_coords = hyper.coords
        local Z_coords = hyper.Z_inducing
        local kernel_type = Symbol("$(m.kernel)")
        
        local K_UU = evaluate_kernel_matrix(
            Z_coords, $(p_names.sigma), $(p_names.ls), kernel_type, M.noise
        )
        local K_XU = evaluate_cross_kernel_matrix(
            X_coords, Z_coords, $(p_names.sigma), $(p_names.ls), kernel_type
        )
    """

    noncentered_code = """
        # --- Nystrom Sparse GP (Non-Centered): $(key) ---
        let
            $(common_code)
            local L_UU = cholesky(Symmetric(K_UU)).L
            local u_latent = L_UU * $(p_names.innovations)
            local nystrom_effect = K_XU * (K_UU \\ u_latent)
            $(eta_target) .+= nystrom_effect
        end
    """

    centered_code = """
        # --- Nystrom Sparse GP (Centered): $(key) ---
        let
            $(common_code)
            $(p_names.latent) ~ MvNormal(zeros(T, $(m.n_inducing)), Symmetric(K_UU))
            local nystrom_effect = K_XU * (K_UU \\ $(p_names.latent))
            $(eta_target) .+= nystrom_effect
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


function get_effects(
    m::Nystrom, chain::Chains, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device
    
    hyper = spec.hyper
    noise = M.noise
    kernel_type = Symbol(m.kernel)

    # --- Coordinate and Inducing Point Handling ---
    coords_train = hyper.coords # Already on device
    Z_inducing = hyper.Z_inducing # Already on device
    
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        coords_pred_cpu = Matrix{Float64}(PS.data[!, Symbol.(coord_vars)])
        vcat(coords_train, to_device(coords_pred_cpu))
    else
        coords_train
    end
    n_obs_full = size(coords_full, 1)

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

        # Extract posterior samples (these are on the CPU)
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        ls_samples_cpu = get_params_matrix(chain, ls_name, m.lengthscale isa Vector ? length(m.lengthscale) : 1)

        # Initialize the output matrix for the full effect on the target device
        effect_k_device = to_device(zeros(Float64, n_obs_full, n_samples))

        # --- Sample-wise Reconstruction on the Target Device ---
        if m.method == :noncentered
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            if isempty(innovations_name)
                @warn "Innovations for Nystrom component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
                continue
            end
            innovations_samples_cpu = get_params_matrix(chain, innovations_name, m.n_inducing)

            for i in 1:n_samples
                current_sigma = sigma_samples_cpu[i]
                current_ls = m.lengthscale isa Vector ? ls_samples_cpu[i, :] : ls_samples_cpu[i, 1]
                
                # Kernel evaluations and linear algebra happen on the device
                K_UU = evaluate_kernel_matrix(Z_inducing, current_sigma, current_ls, kernel_type, noise)
                K_XU = evaluate_cross_kernel_matrix(coords_full, Z_inducing, current_sigma, current_ls, kernel_type)
                
                L_UU = cholesky(Symmetric(K_UU)).L
                u_latent = L_UU * to_device(innovations_samples_cpu[i, :])
                effect_k_device[:, i] = K_XU * (K_UU \ u_latent)
            end
        else # :centered
            latent_name = _find_parameter(p_names, string(p_names_k.latent), k, is_multivariate_model)
            if isempty(latent_name)
                @warn "Latent values for Nystrom component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
                continue
            end
            u_latent_samples_cpu = get_params_matrix(chain, latent_name, m.n_inducing)

            for i in 1:n_samples
                current_sigma = sigma_samples_cpu[i]
                current_ls = m.lengthscale isa Vector ? ls_samples_cpu[i, :] : ls_samples_cpu[i, 1]
                
                # Kernel evaluations and linear algebra happen on the device
                K_UU = evaluate_kernel_matrix(Z_inducing, current_sigma, current_ls, kernel_type, noise)
                K_XU = evaluate_cross_kernel_matrix(coords_full, Z_inducing, current_sigma, current_ls, kernel_type)
                
                effect_k_device[:, i] = K_XU * (K_UU \ to_device(u_latent_samples_cpu[i, :]))
            end
        end
        
        # Move the final result for this outcome back to the CPU
        push!(structured_effects, Array(effect_k_device))
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
