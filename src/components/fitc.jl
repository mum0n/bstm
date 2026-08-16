"""
    FITC <: ComponentModel

A component model for sparse Gaussian Processes. It supports two common
approximations: FITC (Fully Independent Training Conditional) and VFE
(Variational Free Energy), also known as DTC (Deterministic Training Conditional).

# Version
v1.2.2 (2026-08-15)

# Mathematical Summary
Both methods approximate a full GP using a small set of \$M\$ inducing points \$Z\$.
The latent GP values \$f\$ are modeled as:
\$f \\sim \\mathcal{N}(\\mu_f, \\Sigma_f)\$
where the conditional mean is \$\\mu_f = K_{XZ} K_{ZZ}^{-1} u\$, with \$u \\sim \\mathcal{N}(0, K_{ZZ})\$.

The methods differ in their covariance approximation:
- **`:fitc` (default)**: Includes a diagonal correction to account for the variance
  of data points not captured by the inducing points.
  \$\\Sigma_f = \\text{diag}(K_{XX} - Q_{XX}) + \\sigma_n^2 I\$, where \$Q_{XX} = K_{XZ} K_{ZZ}^{-1} K_{ZX}\$.
- **`:vfe` (didactic)**: A pure low-rank approximation, equivalent to DTC.
  \$\\Sigma_f = Q_{XX}\$. This is simpler but can underestimate variance.

# Computational Methods
- `:fitc` (Default, AD-friendly): The Fully Independent Training Conditional approximation.
  It is generally preferred for its more accurate variance estimates.
- `:vfe` (Didactic, AD-friendly): The Variational Free Energy approximation, also known
  as DTC. It is a pure low-rank approximation that can be faster but may
  underestimate variance. Retained for didactic purposes.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`, `y`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `n_inducing`: `Int`, the number of inducing points. Default: `20`.
  - `kernel`: `String`, the name of the kernel function (e.g., `"se"`, `"matern32"`). Default: `"se"`.
  - `sigma`: `UnivariateDistribution`, prior for the marginal standard deviation of the GP. Default: `Exponential(1.0)`.
  - `lengthscale`: `UnivariateDistribution` or `Vector{<:UnivariateDistribution}`, prior for the kernel lengthscale(s). Default: `Gamma(2, 0.5)`.
  - `method`: `Symbol`, approximation method (`:fitc` or `:vfe`). Default: `:fitc`.
  - `knot_method`: `Symbol`, method for placing inducing points (`:kmeans`, `:random`, `:quantile`, `:range`). Default: `:kmeans`.

# Outputs (Parameter Names)
- `sigma_<key>`: The marginal standard deviation of the GP.
- `ls_<key>`: The kernel lengthscale(s).
- `inducing_innovations_<key>`: Raw standard normal innovations for the inducing points.
- `diag_innovations_<key>`: Raw standard normal innovations for the diagonal correction (for `:fitc` method).
- `latent_<key>`: The reconstructed latent GP effect.
"""
struct FITC <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    n_inducing::Int
    kernel::String
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:fitc] = FITC

COMPONENT_CONSTRUCTORS[:fitc] = (p, params) -> FITC(
    p.lengthscale,
    p.sigma,
    get(params, :n_inducing, 20),
    string(get(params, :kernel, "se")),
    get(params, :method, :fitc)
)

MODEL_TO_STRUCTURE_MAP[:fitc] = :smooth

function get_precomputes(m::FITC, M::NamedTuple, mod_data::Dict)::NamedTuple
    variables = mod_data[:variables]
    params = mod_data[:params]

    if isempty(variables)
        error("The FITC model requires coordinate variables, e.g., `random(x, y, model=:fitc)`.")
    end

    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for FITC model not found in data.")
        end
    end

    coords = Matrix{Float64}(M.data[!, Symbol.(variables)])
    
    n_inducing = m.n_inducing
    knot_method = string(get(params, :knot_method, "kmeans"))
    Z_inducing = generate_inducing_points(coords, n_inducing; method=knot_method)

    return (
        coords=coords,
        Z_inducing=Z_inducing,
        n_latent=size(coords, 1)
    )
end

function get_priors(
    m::FITC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
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
    
    push!(priors, "$(p_names.inducing_innovations) ~ MvNormal(zeros(T, $(m.n_inducing)), I)")
    
    if m.method == :fitc
        push!(priors, "$(p_names.diag_innovations) ~ MvNormal(zeros(T, spec.hyper.n_latent), I)")
    end

    return join(priors, "\n    ")
end

function get_updates(
    m::FITC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
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
            
            L_UU = cholesky(Symmetric(K_UU)).L
            u_latent = L_UU * $(p_names.inducing_innovations)
    """

    fitc_code = """
        # --- FITC Sparse GP Component: $(key) ---
        $(common_code)
            
            K_UU_inv_u = K_UU \\ u_latent
            mean_f = K_XU * K_UU_inv_u
            
            diag_K_XX = fill($(p_names.sigma)^2, hyper.n_latent)
            tmp = (L_UU' \\ K_XU')'
            diag_Q_ff = sum(tmp.^2, dims=2)
            lambda_diag = diag_K_XX - vec(diag_Q_ff)
            
            $(p_names.latent) = mean_f .+
                sqrt.(max.(lambda_diag, 0.0) .+ M.noise) .* $(p_names.diag_innovations)
            
            $(eta_target) .+= $(p_names.latent)
        end
    """

    vfe_code = """
        # --- VFE/DTC Sparse GP Component: $(key) ---
        $(common_code)
            
            K_UU_inv_u = K_UU \\ u_latent
            $(p_names.latent) = K_XU * K_UU_inv_u
            
            $(eta_target) .+= $(p_names.latent)
        end
    """

    if m.method == :fitc
        return fitc_code
    elseif m.method == :vfe
        return vfe_code
    else
        error("Unsupported method '$(m.method)' for FITC component. Supported methods are :fitc and :vfe.")
    end
end

function get_effects(
    m::FITC, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    p_names::Vector{String}, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, 
    N_total::Int, is_multivariate_model::Bool
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    hyper = spec.hyper
    coords_train = hyper.coords
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        coords_train
    end
    n_obs_full = size(coords_full, 1)

    Z_inducing = hyper.Z_inducing
    kernel_type = Symbol(m.kernel)
    noise = M.noise

    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        ls_name = _find_parameter(p_names, string(p_names_k.ls), k, is_multivariate_model)
        inducing_innov_name = _find_parameter(p_names, string(p_names_k.inducing_innovations), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(ls_name) || isempty(inducing_innov_name)
            @warn "Parameters for FITC component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        ls_samples = get_params_matrix(chain, ls_name, m.lengthscale isa Vector ? length(m.lengthscale) : 1)
        inducing_innov_samples = get_params_matrix(chain, inducing_innov_name, m.n_inducing)

        effect_k = zeros(Float64, n_obs_full, n_samples)

        for i in 1:n_samples
            current_sigma = sigma_samples[i]
            current_ls = m.lengthscale isa Vector ? ls_samples[i, :] : ls_samples[i, 1]
            current_u_raw = inducing_innov_samples[i, :]
            
            K_UU = evaluate_kernel_matrix(Z_inducing, current_sigma, current_ls, kernel_type, noise)
            K_XU = evaluate_cross_kernel_matrix(coords_full, Z_inducing, current_sigma, current_ls, kernel_type)
            
            L_UU = cholesky(Symmetric(K_UU)).L
            u_latent = L_UU * current_u_raw
            K_UU_inv_u = K_UU \ u_latent
            mean_f = K_XU * K_UU_inv_u

            if m.method == :fitc
                diag_innov_name = _find_parameter(p_names, string(p_names_k.diag_innovations), k, is_multivariate_model)
                if isempty(diag_innov_name)
                    @warn "Diagonal innovations for FITC component $(spec.key) (outcome $k) not found. Using zero for correction."
                    effect_k[:, i] = mean_f
                    continue
                end
                
                diag_innov_samples = get_params_matrix(chain, diag_innov_name, hyper.n_latent)
                diag_innov_i = if size(diag_innov_samples, 2) == n_obs_full
                    diag_innov_samples[i, :]
                else
                    vcat(diag_innov_samples[i, :], randn(n_obs_full - size(diag_innov_samples, 2)))
                end

                diag_K_XX = fill(current_sigma^2, n_obs_full)
                tmp = (L_UU' \ K_XU')'
                diag_Q_ff = sum(tmp.^2, dims=2)
                lambda_diag = diag_K_XX - vec(diag_Q_ff)
                
                effect_k[:, i] = mean_f .+ sqrt.(max.(lambda_diag, 0.0) .+ noise) .* diag_innov_i
            else # :vfe
                effect_k[:, i] = mean_f
            end
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
