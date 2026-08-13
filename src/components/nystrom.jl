"""
    Nystrom <: ComponentModel

A component model for the Nyström sparse Gaussian Process approximation. This method
approximates the full GP kernel matrix with a low-rank version based on a small set
of `n_inducing` points, making it scalable for larger datasets.

# Version
v1.1.0 (2026-08-11)

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
\$f = K_{XZ} (L_{ZZ}^T \\backslash v)\$

# Computational Methods
- `:noncentered` (default): A non-centered parameterization where the latent values
  at inducing points are constructed from standard normal innovations. Recommended
  for efficient MCMC sampling.
- `:centered` (didactic): A centered parameterization where the latent values at inducing points
  are sampled directly from their `MvNormal` distribution. This can be less efficient.

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
- `latent_<key>`: The latent values at the inducing points (for `:centered`).
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

function get_datastructures!(m_type::Type{<:Nystrom}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]
    params = mod_data[:params]

    if isempty(variables)
        error(
            "The Nystrom model requires coordinate variables, e.g., " *
            "`random(x, y, model=:nystrom)`."
        )
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Nystrom model not found in data.")
        end
    end

    coords = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    mod_data[:params][:coords] = coords

    n_inducing = get(params, :n_inducing, 20)
    knot_method = get(params, :knot_method, :kmeans)
    Z_inducing = generate_inducing_points(coords, n_inducing; method=string(knot_method))
    mod_data[:params][:Z_inducing] = Z_inducing

    return true
end

function get_precomputes(m::Nystrom, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("Nystrom component precomputes failed: coordinates not found.")
    end
    
    Z_inducing = get(mod_data[:params], :Z_inducing, nothing)
    if isnothing(Z_inducing)
        error("Nystrom component precomputes failed: inducing points not found.")
    end

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
    
    push!(priors, "$(p_names.innovations) ~ DynamicPPL.NamedDist(MvNormal(zeros(T, $(m.n_inducing)), I), :$(p_names.innovations))") # Raw standard normal innovations for inducing points

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
        let
            $(common_code)
            L_UU = cholesky(Symmetric(K_UU)).L
            u_latent = L_UU * $(p_names.innovations)
            $(p_names.latent) = K_XU * (K_UU \\ u_latent)
            $(eta_target) .+= $(p_names.latent)
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
    m::Nystrom, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    coords_train = spec.hyper.coords
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        coords_train
    end
    n_obs_full = size(coords_full, 1)

    Z_inducing = spec.hyper.Z_inducing
    kernel_type = Symbol(m.kernel)
    noise = M.noise
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))

    for k in 1:outcomes_N
        sigma_name = _find_parameter(p_names_vec, string(spec.key), "sigma", k, is_multivariate_model)
        ls_name = _find_parameter(p_names_vec, string(spec.key), "ls", k, is_multivariate_model)

        if isempty(sigma_name) || isempty(ls_name)
            @warn "Parameters for Nystrom component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        ls_samples = get_params_vector(chain, ls_name, m.lengthscale isa Vector ? length(m.lengthscale) : 1)

        effect_k = zeros(Float64, n_obs_full, n_samples)

        if m.method == :noncentered
            innovations_name = _find_parameter(p_names_vec, string(spec.key), "innovations", k, is_multivariate_model)
            if isempty(innovations_name)
                @warn "Innovations for Nystrom component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
                continue
            end
            innovations_samples = get_params_vector(chain, innovations_name, m.n_inducing)
            for i in 1:n_samples
                current_sigma = sigma_samples[i]
                current_ls = m.lengthscale isa Vector ? ls_samples[i, :] : ls_samples[i, 1]
                
                K_UU = evaluate_kernel_matrix(Z_inducing, current_sigma, current_ls, kernel_type, noise)
                K_XU = evaluate_cross_kernel_matrix(coords_full, Z_inducing, current_sigma, current_ls, kernel_type)
                
                L_UU = cholesky(Symmetric(K_UU)).L
                u_latent = L_UU * innovations_samples[i, :]
                effect_k[:, i] = K_XU * (K_UU \ u_latent)
            end
        else # :centered
            latent_name = _find_parameter(p_names_vec, string(spec.key), "latent", k, is_multivariate_model)
            if isempty(latent_name)
                @warn "Latent values for Nystrom component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
                continue
            end
            u_latent_samples = get_params_vector(chain, latent_name, m.n_inducing)
            for i in 1:n_samples
                current_sigma = sigma_samples[i]
                current_ls = m.lengthscale isa Vector ? ls_samples[i, :] : ls_samples[i, 1]
                
                K_UU = evaluate_kernel_matrix(Z_inducing, current_sigma, current_ls, kernel_type, noise)
                K_XU = evaluate_cross_kernel_matrix(coords_full, Z_inducing, current_sigma, current_ls, kernel_type)
                
                effect_k[:, i] = K_XU * (K_UU \ u_latent_samples[i, :])
            end
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
