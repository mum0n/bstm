# File: c:\home\jae\projects\bstm\src\components\fitc.jl
"""
    FITC <: ComponentModel

A component model for sparse Gaussian Processes. It supports two common
approximations: FITC (Fully Independent Training Conditional) and VFE
(Variational Free Energy), also known as DTC (Deterministic Training Conditional).

# Version
v1.4.1 (2026-08-19)

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

    # Perform data processing on the CPU
    coords_cpu = Matrix{Float64}(M.data[!, Symbol.(variables)])
    
    n_inducing = m.n_inducing
    knot_method = string(get(params, :knot_method, "kmeans"))
    Z_inducing_cpu = generate_inducing_points(coords_cpu, n_inducing; method=knot_method)

    return (
        coords=coords_cpu,
        Z_inducing=Z_inducing_cpu,
        n_latent=size(coords_cpu, 1)
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
    
    push!(priors, "$(p_names.ure_inducing) ~ MvNormal(zeros(T, $(m.n_inducing)), I)")
    
    if m.method == :fitc
        push!(priors, "$(p_names.ure_diag) ~ MvNormal(zeros(T, spec.hyper.n_latent), I)")
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
            u_latent = L_UU * $(p_names.ure_inducing)
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
            
            $(p_names.sre) = mean_f .+
                sqrt.(max.(lambda_diag, 0.0) .+ M.noise) .* $(p_names.ure_diag)
            
            $(eta_target) .+= $(p_names.sre)
        end
    """

    vfe_code = """
        # --- VFE/DTC Sparse GP Component: $(key) ---
        $(common_code)
            
            K_UU_inv_u = K_UU \\ u_latent
            $(p_names.sre) = K_XU * K_UU_inv_u
            
            $(eta_target) .+= $(p_names.sre)
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

"""
    get_effects(m::FITC, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the sparse GP effect from posterior samples, dispatching on the
method used during sampling. This version is CPU-only and uses modern chain accessors.
"""
function get_effects(
    m::FITC, chain, spec::NamedTuple, M::NamedTuple,
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
    n_latent_train = hyper.n_latent
    kernel_type = Symbol(m.kernel)
    
    # --- Coordinate and Inducing Point Handling ---
    coords_train = hyper.coords # Training coordinates
    Z_inducing = hyper.Z_inducing # Inducing points
    
    # Combine training and prediction coordinates
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars) # If prediction set is provided
        coords_pred = Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]) # Extract prediction coordinates
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
        inducing_ure_name = _find_parameter(p_names, string(p_names_k.ure_inducing), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(ls_name) || isempty(inducing_ure_name)
            @warn "Parameters for FITC component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        sigma_samples = get_params_vector(chain, sigma_name, 1) # (n_samples, 1)
        ls_dim = m.lengthscale isa Vector ? length(m.lengthscale) : 1 # Dimension of lengthscale parameter
        ls_samples = get_params_matrix(chain, ls_name, ls_dim) # (n_samples, ls_dim)
        inducing_ure_samples = get_params_matrix(chain, inducing_ure_name, m.n_inducing) # (n_samples, n_inducing)

        # Initialize the output matrix for the full effect
        effect_k_matrix = zeros(Float64, n_obs_full, n_samples)

        # --- Sample-wise Reconstruction ---
        for i in 1:n_samples # Iterate over each posterior sample
            current_sigma = sigma_samples[i, 1] # Sigma for current sample
            current_ls = ls_dim > 1 ? ls_samples[i, :] : ls_samples[i, 1] # Lengthscale for current sample
            current_u_unscaled = inducing_ure_samples[i, :] # Unscaled innovations for inducing points for current sample
            
            # Kernel evaluations happen on the CPU
            K_UU = evaluate_kernel_matrix(Z_inducing, current_sigma, current_ls, kernel_type, noise)
            K_XU = evaluate_cross_kernel_matrix(coords_full, Z_inducing, current_sigma, current_ls, kernel_type)
            
            # Cholesky and linear solves happen on the CPU
            L_UU = cholesky(Symmetric(K_UU)).L
            u_latent = L_UU * current_u_unscaled
            K_UU_inv_u = K_UU \ u_latent
            mean_f = K_XU * K_UU_inv_u

            if m.method == :fitc
                diag_ure_name = _find_parameter(p_names, string(p_names_k.ure_diag), k, is_multivariate_model)
                if isempty(diag_ure_name)
                    @warn "Diagonal innovations for FITC component $(spec.key) (outcome $k) not found. Using zero for correction."
                    effect_k_matrix[:, i] = mean_f
                    continue
                end
                
                diag_ure_samples = get_params_matrix(chain, diag_ure_name, n_latent_train) # (n_samples, n_latent_train)
                
                # Handle prediction set by generating new innovations
                diag_ure_i = if n_obs_full > n_latent_train
                    vcat(
                        diag_ure_samples[i, :],
                        randn(Float64, n_obs_full - n_latent_train) # Generate new innovations for prediction points
                    )
                else
                    diag_ure_samples[i, :]
                end

                # Diagonal correction calculations
                diag_K_XX = fill(current_sigma^2, n_obs_full)
                tmp = (L_UU' \ K_XU')'
                diag_Q_ff = sum(tmp.^2, dims=2)
                lambda_diag = diag_K_XX - vec(diag_Q_ff)
                
                effect_k_matrix[:, i] = mean_f .+ sqrt.(max.(lambda_diag, 0.0) .+ noise) .* diag_ure_i
            else # :vfe
                effect_k_matrix[:, i] = mean_f
            end
        end
        
        push!(structured_effects, effect_k_matrix)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
