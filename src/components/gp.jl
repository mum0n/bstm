"""
    GP <: ComponentModel

A component model for a full Gaussian Process (GP), also known as Kriging in
geostatistics. It models a latent field by computing a dense covariance
matrix based on a specified kernel function and coordinate inputs.

# Version
v1.3.0 (2026-08-17)

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

    # Get the device transfer function
    to_device = M.to_device

    # Extract coordinates to a CPU matrix first to handle potential GPU arrays in M.data
    coords_cpu = Matrix{Float64}(M.data[!, Symbol.(variables)])
    n_latent = size(coords_cpu, 1)
    
    # Move coordinates to the target device
    return (coords=to_device(coords_cpu), n_latent=n_latent)
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
        push!(priors, "$(p_names.innovations) ~ MvNormal(zeros(T, spec.hyper.n_latent), I)")
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
            $(p_names.latent) = F_gp.L * $(p_names.innovations)
            $(eta_target) .+= $(p_names.latent)
        end
    """

    centered_code = """
        # --- GP (Centered): $(key) ---
        $(common_code)
            $(p_names.latent) ~ MvNormal(zeros(T, size(K_mat, 1)), Symmetric(K_mat))
            $(eta_target) .+= $(p_names.latent)
        end
    """

    if m.method == :noncentered
        return noncentered_code
    elseif m.method == :centered
        return centered_code
    else
        error("Unsupported method '$(m.method)' for GP component.")
    end
end

"""
    get_effects(m::GP, chain, spec, M, PS)

Reconstructs the `GP` component's effect from posterior samples. This version is
updated to handle GPU arrays by moving sampled parameters and prediction data to
the device for computation, and moving the final results back to the CPU.
"""
function get_effects(
    m::GP, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device
    
    # --- Coordinate Handling: Combine training and prediction sets on device ---
    coord_vars = get(spec.params, :positional_args, [])
    coords_train_device = spec.hyper.coords # Already on device
    
    coords_full_device = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        coords_pred_cpu = Matrix{Float64}(PS.data[!, Symbol.(coord_vars)])
        vcat(coords_train_device, to_device(coords_pred_cpu))
    else
        coords_train_device
    end
    n_obs_train = size(coords_train_device, 1)
    n_obs_full = size(coords_full_device, 1)
    
    noise = M.noise
    kernel_type = Symbol(m.kernel)
    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        
        # Find parameter names in the MCMC chain
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        ls_name = _find_parameter(p_names, string(p_names_k.ls), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(ls_name)
            @warn "Parameters for GP component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        ls_dim = m.lengthscale isa Vector ? length(m.lengthscale) : 1
        ls_samples_cpu = get_params_matrix(chain, ls_name, ls_dim)

        # Initialize the output matrix for the full effect on the target device
        effect_k_device = to_device(zeros(Float64, n_obs_full, n_samples))

        # --- Sample-wise Reconstruction on the Target Device ---
        if m.method == :noncentered
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            if isempty(innovations_name)
                @warn "Innovations for GP component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
                continue
            end
            innovations_samples_cpu = get_params_matrix(chain, innovations_name, n_obs_train)

            for i in 1:n_samples
                current_ls = ls_dim > 1 ? ls_samples_cpu[i, :] : ls_samples_cpu[i, 1]
                
                # Kernel evaluation and Cholesky happen on the device
                K_mat = evaluate_kernel_matrix(coords_full_device, sigma_samples_cpu[i], current_ls, kernel_type, noise)
                F = cholesky(Symmetric(K_mat))
                
                # Combine training innovations with new innovations for prediction points on the device
                innov_train_device = to_device(innovations_samples_cpu[i, :])
                innov_i_device = if n_obs_full > n_obs_train
                    innov_pred_device = to_device(randn(Float32, n_obs_full - n_obs_train))
                    vcat(innov_train_device, innov_pred_device)
                else
                    innov_train_device
                end
                
                effect_k_device[:, i] = F.L * innov_i_device
            end
        elseif m.method == :centered
            latent_name = _find_parameter(p_names, string(p_names_k.latent), k, is_multivariate_model)
            if isempty(latent_name)
                @warn "Latent field for GP component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
                continue
            end
            latent_samples_cpu = get_params_matrix(chain, latent_name, n_obs_train)

            for i in 1:n_samples
                # Move training samples to device
                effect_k_device[1:n_obs_train, i] = to_device(latent_samples_cpu[i, :])
                
                if n_obs_full > n_obs_train
                    coords_pred_device = coords_full_device[(n_obs_train+1):end, :]
                    current_ls = ls_dim > 1 ? ls_samples_cpu[i, :] : ls_samples_cpu[i, 1]
                    
                    # All kernel evaluations and linear algebra happen on the device
                    K_ff = evaluate_kernel_matrix(coords_train_device, sigma_samples_cpu[i], current_ls, kernel_type, noise)
                    K_star_f = evaluate_cross_kernel_matrix(coords_pred_device, coords_train_device, sigma_samples_cpu[i], current_ls, kernel_type)
                    K_star_star = evaluate_kernel_matrix(coords_pred_device, sigma_samples_cpu[i], current_ls, kernel_type, noise)
                    
                    L_ff = cholesky(Symmetric(K_ff)).L
                    A = L_ff' \ (L_ff \ K_star_f')
                    mu_pred = A' * to_device(latent_samples_cpu[i, :])
                    Sigma_pred = K_star_star - K_star_f * A
                    
                    # Sample from the conditional posterior on the device
                    pred_innov = to_device(randn(Float32, size(Sigma_pred, 1)))
                    effect_k_device[(n_obs_train+1):end, i] = mu_pred + cholesky(Symmetric(Sigma_pred)).L * pred_innov
                end
            end
        end
        
        # Move the final result for this outcome back to the CPU
        push!(structured_effects, Array(effect_k_device))
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end

