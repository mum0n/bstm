"""
    GP <: ComponentModel

A component model for a full Gaussian Process (GP), also known as Kriging in
geostatistics. It models a latent field by computing a dense covariance
matrix based on a specified kernel function and coordinate inputs.

# Version
v1.4.1 (2026-08-19)

# Mathematical Summary
The component models a latent field \$f(x)\$ as a draw from a Gaussian Process with
a zero mean and a specified covariance function (kernel):
\$f(x) \\sim \\mathcal{GP}(0, k(x, x'))\$

The kernel \$k(x, x')\$ defines the covariance between any two points. For example,
the Squared Exponential (SE) kernel is:
\$k(x, x') = \\sigma^2 \\exp\\left(-\\frac{\\|x - x'\\|^2}{2\\ell^2}\\right)\$
where:
- \$\\sigma^2\$ is the marginal variance.
- \$\\ell\$ is the characteristic lengthscale.

For **anisotropic** models (Automatic Relevance Determination), the squared distance
is weighted by a vector of lengthscales \$\\boldsymbol{\\ell} = [\\ell_1, \\dots, \\ell_D]\$:
\$k(x, x') = \\sigma^2 \\exp\\left(-\\frac{1}{2} \\sum_{d=1}^D \\frac{(x_d - x'_d)^2}{\\ell_d^2}\\right)\$

The model samples the latent field \$f\$ from the resulting multivariate normal
distribution \$f \\sim \\mathcal{N}(0, K)\$, where \$K\$ is the dense covariance matrix
evaluated at all data points.

# Computational Methods
- `:noncentered` (Default, AD-friendly): Samples standard normal innovations and
  transforms them using the Cholesky factor of the covariance matrix. Recommended
  for gradient-based samplers like NUTS.
- `:centered` (Didactic, Not AD-friendly): Samples the latent field directly from
  the `MvNormal` distribution defined by the covariance matrix. This can be less
  efficient for MCMC due to posterior correlations.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`, `y`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `kernel`: `String`, the name of the kernel function (e.g., `"se"`, `"matern32"`). Default: `"se"`.
  - `sigma`: `UnivariateDistribution`, prior for the marginal standard deviation of the GP. Default: `Exponential(1.0)`.
  - `lengthscale`: `UnivariateDistribution` or `Vector{<:UnivariateDistribution}`, prior for the kernel lengthscale(s). Default: `Gamma(2, 0.5)`.
  - `anisotropic`: `Bool`, if `true`, a separate lengthscale is estimated for each input dimension (ARD). Default: `false`.
  - `method`: `Symbol`, computational method (`:noncentered` or `:centered`). Default: `:noncentered`.

# Outputs (Parameter Names)
- `sigma_<key>`: The marginal standard deviation of the GP.
- `ls_<key>`: The kernel lengthscale(s). A vector if anisotropic.
- `innovations_<key>`: The raw standard normal innovations for the latent field (for `:noncentered`).
- `latent_<key>`: The latent field (for `:centered`).

# Key References
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
"""
struct GP <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    kernel::String
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:gp] = GP
COMPONENT_CONSTRUCTORS[:gp] = (p, params) -> GP(
    p.lengthscale, p.sigma, string(get(params, :kernel, "se")),
    get(params, :method, :noncentered)
)

MODEL_TO_STRUCTURE_MAP[:gp] = :smooth

function get_precomputes(m::GP, M::NamedTuple, mod_data::Dict)::NamedTuple
    variables = mod_data[:variables]
    if isempty(variables)
        error("The GP model requires coordinate variables, e.g., `random(x, y, model=:gp)`.")
    end

    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for GP model not found in data.")
        end
    end

    # Extract coordinates to a CPU matrix.
    coords = Matrix{Float64}(M.data[!, Symbol.(variables)])
    n_latent = size(coords, 1)
    
    # Return CPU arrays.
    return (coords=coords, n_latent=n_latent)
end

"""
    _gp_log_marginal_likelihood(y_residual, coords, sigma, ls, kernel_type, y_sigma, noise=1e-6)

Computes the exact log marginal likelihood for a full Gaussian Process with latent field integrated out analytically.
"""
function _gp_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    coords::AbstractMatrix,
    sigma::T,
    ls::Union{T, AbstractVector{T}},
    kernel_type::Symbol,
    y_sigma::T,
    noise::Real=1e-6
) where {T}
    N = length(y_residual)
    T_num = promote_type(T, typeof(noise))
    
    K_mat = evaluate_kernel_matrix(coords, sigma, ls, kernel_type, noise)
    total_noise = y_sigma^2 + T_num(noise)
    
    Ky = Matrix{T_num}(K_mat)
    for i in 1:N
        Ky[i, i] += total_noise
    end
    
    F = cholesky(Symmetric(Ky))
    log_det_Ky = 2 * sum(log.(diag(F.U)))
    
    v = F.L \ y_residual
    quad_term = dot(v, v)
    
    log_lik = - (N / 2) * log(2 * T_num(pi)) -
              (1 / 2) * log_det_Ky -
              (1 / 2) * quad_term
              
    return log_lik
end

function get_priors(
    m::GP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = String[]
    push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")

    # This logic correctly handles both isotropic (single Distribution) and
    # anisotropic (Vector of Distributions) cases for the lengthscale.
    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors, "$(p_names.ls) ~ Product([$(ls_priors_str)])")
    else
        ls_prior_str = _distribution_to_string(m.lengthscale)
        push!(priors, "$(p_names.ls) ~ $(ls_prior_str)")
    end
    
    if m.method == :noncentered
        push!(priors, "$(p_names.ure) ~ MvNormal(zeros(T, spec.hyper.n_latent), I)")
    end

    return join(priors, "\n    ")
end

function get_updates(
    m::GP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    
    # The `evaluate_kernel_matrix` function is designed to handle both scalar and vector
    # lengthscale parameters (`ls`), correctly implementing isotropic and ARD kernels.
    common_code = """
        let
            coords = spec_registry[:$(key)].hyper.coords
            kernel_type = Symbol("$(m.kernel)")

            K_mat = evaluate_kernel_matrix(
                coords, $(p_names.sigma), $(p_names.ls), kernel_type, M.noise
            )
    """

    noncentered_code = """
        # --- GP (Non-Centered): $(key) ---
        $(common_code)
            F_gp = cholesky(Symmetric(K_mat))
            $(p_names.sre) = F_gp.L * $(p_names.ure)
            $(eta_target) .+= $(p_names.sre)
        end
    """

    centered_code = """
        # --- GP (Centered): $(key) ---
        $(common_code)
            $(p_names.sre) ~ MvNormal(zeros(T, size(K_mat, 1)), Symmetric(K_mat))
            $(eta_target) .+= $(p_names.sre)
        end
    """

    marginalized_code = """
        # --- GP (Marginalized): $(key) ---
        let
            coords = spec_registry[:$(key)].hyper.coords
            kernel_type = Symbol("$(m.kernel)")
            y_residual = M.y_obs .- $(eta_target)
            log_lik_marginalized_$(key) = _gp_log_marginal_likelihood(
                y_residual,
                coords,
                $(p_names.sigma),
                $(p_names.ls),
                kernel_type,
                y_sigma,
                M.noise
            )
            Turing.@addlogprob! log_lik_marginalized_$(key)
        end
    """

    if m.method == :noncentered
        return noncentered_code
    elseif m.method == :centered
        return centered_code
    elseif m.method == :marginalized
        return marginalized_code
    else
        error("Unsupported method '$(m.method)' for GP component. Use :noncentered, :centered, or :marginalized.")
    end
end

"""
    get_effects(m::GP, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the `GP` component's effect from posterior samples. This version is
CPU-only and uses modern chain accessors.
"""
function get_effects(
    m::GP, chain, spec::NamedTuple, M::NamedTuple,
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
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        coords_pred = Matrix{Float64}(PS.data[!, Symbol.(coord_vars)])
        vcat(coords_train, coords_pred)
    else
        coords_train
    end
    n_obs_train = size(coords_train, 1)
    n_obs_full = size(coords_full, 1)
    
    noise = M.noise
    kernel_type = Symbol(m.kernel)
    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        ls_name = _find_parameter(p_names, string(p_names_k.ls), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(ls_name)
            @warn "Parameters for GP component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        ls_dim = m.lengthscale isa Vector ? length(m.lengthscale) : 1
        ls_samples = get_params_matrix(chain, ls_name, ls_dim)

        effect_k_matrix = zeros(Float64, n_obs_full, n_samples)

        # --- Sample-wise Reconstruction ---
        if m.method == :marginalized
            y_sigma_name = _find_parameter(p_names, "y_sigma", k, is_multivariate_model)
            y_sigma_samples = if !isempty(y_sigma_name)
                get_params_vector(chain, y_sigma_name, 1)[:, 1]
            else
                fill(1.0, n_samples)
            end
            
            y_vec = M.y_obs isa AbstractMatrix ? M.y_obs[:, k] : M.y_obs
            
            for i in 1:n_samples
                current_ls = ls_dim > 1 ? ls_samples[i, :] : ls_samples[i, 1]
                sig = sigma_samples[i, 1]
                y_sig = y_sigma_samples[i]
                
                K_ff = evaluate_kernel_matrix(coords_train, sig, current_ls, kernel_type, noise)
                Ky = Matrix{Float64}(K_ff)
                for j in 1:n_obs_train
                    Ky[j, j] += y_sig^2 + noise
                end
                
                F_y = cholesky(Symmetric(Ky))
                alpha = F_y \ y_vec
                
                if n_obs_full == n_obs_train
                    # Training conditional posterior: f | y ~ N(K_ff Ky^{-1} y, K_ff - K_ff Ky^{-1} K_ff)
                    mu_train = K_ff * alpha
                    Sigma_train = K_ff - K_ff * (F_y \ K_ff)
                    for j in 1:n_obs_train
                        Sigma_train[j, j] = max(Sigma_train[j, j], noise)
                    end
                    F_cov = cholesky(Symmetric(Sigma_train + noise * I))
                    effect_k_matrix[:, i] = mu_train + F_cov.L * randn(n_obs_train)
                else
                    coords_pred = coords_full[(n_obs_train+1):end, :]
                    n_pred = size(coords_pred, 1)
                    
                    K_star_f = evaluate_cross_kernel_matrix(coords_pred, coords_train, sig, current_ls, kernel_type)
                    K_star_star = evaluate_kernel_matrix(coords_pred, sig, current_ls, kernel_type, noise)
                    
                    # Full joint covariance
                    K_full = evaluate_kernel_matrix(coords_full, sig, current_ls, kernel_type, noise)
                    K_full_train = evaluate_cross_kernel_matrix(coords_full, coords_train, sig, current_ls, kernel_type)
                    
                    mu_full = K_full_train * alpha
                    Sigma_full = K_full - K_full_train * (F_y \ K_full_train')
                    for j in 1:n_obs_full
                        Sigma_full[j, j] = max(Sigma_full[j, j], noise)
                    end
                    F_full = cholesky(Symmetric(Sigma_full + noise * I))
                    effect_k_matrix[:, i] = mu_full + F_full.L * randn(n_obs_full)
                end
            end
        elseif m.method == :noncentered
            ure_name = _find_parameter(p_names, string(p_names_k.ure), k, is_multivariate_model)
            if isempty(ure_name)
                @warn "Innovations for GP component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
                continue
            end
            ure_samples = get_params_matrix(chain, ure_name, n_obs_train)

            for i in 1:n_samples
                current_ls = ls_dim > 1 ? ls_samples[i, :] : ls_samples[i, 1]
                
                K_mat = evaluate_kernel_matrix(coords_full, sigma_samples[i, 1], current_ls, kernel_type, noise)
                F = cholesky(Symmetric(K_mat))
                
                innov_train = ure_samples[i, :]
                innov_i = if n_obs_full > n_obs_train
                    innov_pred = randn(Float64, n_obs_full - n_obs_train)
                    vcat(innov_train, innov_pred)
                else
                    innov_train
                end
                effect_k_matrix[:, i] = F.L * innov_i
            end
        elseif m.method == :centered
            sre_name = _find_parameter(p_names, string(p_names_k.sre), k, is_multivariate_model)
            if isempty(sre_name)
                @warn "Latent field for GP component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
                continue
            end
            sre_samples = get_params_matrix(chain, sre_name, n_obs_train)

            for i in 1:n_samples
                effect_k_matrix[1:n_obs_train, i] = sre_samples[i, :]
                
                if n_obs_full > n_obs_train
                    coords_pred = coords_full[(n_obs_train+1):end, :]
                    current_ls = ls_dim > 1 ? ls_samples[i, :] : ls_samples[i, 1]
                    
                    K_ff = evaluate_kernel_matrix(coords_train, sigma_samples[i, 1], current_ls, kernel_type, noise)
                    K_star_f = evaluate_cross_kernel_matrix(coords_pred, coords_train, sigma_samples[i, 1], current_ls, kernel_type)
                    K_star_star = evaluate_kernel_matrix(coords_pred, sigma_samples[i, 1], current_ls, kernel_type, noise)
                    
                    L_ff = cholesky(Symmetric(K_ff)).L
                    A = L_ff' \ (L_ff \ K_star_f')
                    mu_pred = A' * sre_samples[i, :]
                    Sigma_pred = K_star_star - K_star_f * A
                    
                    pred_innov = randn(Float64, size(Sigma_pred, 1))
                    effect_k_matrix[(n_obs_train+1):end, i] = mu_pred + cholesky(Symmetric(Sigma_pred + noise * I)).L * pred_innov
                end
            end
        end
        
        push!(structured_effects, effect_k_matrix)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end 