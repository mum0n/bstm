"""
    GP <: ComponentModel

A component model for a full Gaussian Process (GP), also known as Kriging in
geostatistics. It models a latent field by computing a dense covariance
matrix based on a specified kernel function and coordinate inputs.

# Version
v1.2.1 (2026-08-12)

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

function get_datastructures!(m_type::Type{<:GP}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]
    if isempty(variables)
        error("The GP model requires coordinate variables, e.g., `random(x, y, model=:gp)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for GP not found in data.")
        end
    end

    coords = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    mod_data[:params][:coords] = coords
    # Pass the number of input dimensions to the parameter dictionary so that
    # `resolve_hyperpriors` can correctly construct anisotropic priors.
    mod_data[:params][:in_dims] = size(coords, 2)
    return true
end

function get_precomputes(m::GP, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("GP component precomputes failed: coordinates not found in module data.")
    end
    
    n_latent = size(coords, 1)
    return (coords=coords, n_latent=n_latent)
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
    
    # Removed explicit `T` from `zeros` for better AD compatibility.
    push!(priors, "$(p_names.innovations) ~ DynamicPPL.NamedDist(MvNormal(zeros(spec.hyper.n_latent), I), :$(p_names.innovations))")

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
        coords = spec_registry[:$(key)].hyper.coords
        kernel_type = Symbol("$(m.kernel)")

        K_mat = evaluate_kernel_matrix(
            coords, $(p_names.sigma), $(p_names.ls), kernel_type, M.noise
        )
    """

    noncentered_code = """
        # --- GP (Non-Centered): $(key) ---
        let
            $(common_code)
            F_gp = cholesky(Symmetric(K_mat))
            $(p_names.latent) = F_gp.L * $(p_names.innovations)
            $(eta_target) .+= $(p_names.latent)
        end
    """

    centered_code = """
        # --- GP (Centered): $(key) ---
        let
            $(common_code)
            # Removed explicit `T` from `zeros` for better AD compatibility.
            $(p_names.latent) ~ MvNormal(zeros(size(K_mat, 1)), Symmetric(K_mat))
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

function get_effects(
    m::GP, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
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
    kernel_type = Symbol(m.kernel)
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))

    for k in 1:outcomes_N
        sigma_name = _find_parameter(p_names_vec, string(spec.key), "sigma", k, is_multivariate_model)
        ls_name = _find_parameter(p_names_vec, string(spec.key), "ls", k, is_multivariate_model)

        if isempty(sigma_name) || isempty(ls_name)
            @warn "Parameters for GP component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        # This correctly fetches a vector of lengthscales for each sample if the model is anisotropic.
        ls_samples = get_params_vector(chain, ls_name, m.lengthscale isa Vector ? length(m.lengthscale) : 1)

        effect_k = zeros(Float64, n_obs_full, n_samples)

        if m.method == :noncentered
            innovations_name = _find_parameter(p_names_vec, string(spec.key), "innovations", k, is_multivariate_model)
            if isempty(innovations_name)
                @warn "Innovations for GP component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
                continue
            end
            innovations_samples = get_params_vector(chain, innovations_name, n_obs_train)
            for i in 1:n_samples
                # Pass the vector of lengthscales for this sample to the kernel matrix function.
                K_mat = evaluate_kernel_matrix(coords_full, sigma_samples[i], ls_samples[i,:], kernel_type, noise)
                F = cholesky(Symmetric(K_mat))
                innov_i = vcat(innovations_samples[i, :], randn(n_obs_full - n_obs_train))
                effect_k[:, i] = F.L * innov_i
            end
        elseif m.method == :centered
            latent_name = _find_parameter(p_names_vec, string(spec.key), "latent", k, is_multivariate_model)
            if isempty(latent_name)
                @warn "Latent field for GP component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
                continue
            end
            latent_samples = get_params_vector(chain, latent_name, n_obs_train)
            for i in 1:n_samples
                effect_k[1:n_obs_train, i] = latent_samples[i, :]
                if n_obs_full > n_obs_train
                    coords_pred = coords_full[(n_obs_train+1):end, :]
                    # Pass the vector of lengthscales for this sample to the kernel matrix functions.
                    K_ff = evaluate_kernel_matrix(coords_train, sigma_samples[i], ls_samples[i,:], kernel_type, noise)
                    K_star_f = evaluate_cross_kernel_matrix(coords_pred, coords_train, sigma_samples[i], ls_samples[i,:], kernel_type)
                    K_star_star = evaluate_kernel_matrix(coords_pred, sigma_samples[i], ls_samples[i,:], kernel_type, noise)
                    
                    L_ff = cholesky(Symmetric(K_ff)).L
                    A = L_ff' \ (L_ff \ K_star_f')
                    mu_pred = A' * latent_samples[i, :]
                    Sigma_pred = K_star_star - K_star_f * A
                    
                    effect_k[(n_obs_train+1):end, i] = rand(MvNormal(mu_pred, Symmetric(Sigma_pred)))
                end
            end
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
