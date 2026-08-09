"""
    SVGP <: ComponentModel

A component for a Sparse Variational Gaussian Process (SVGP). This method
approximates a full Gaussian Process by using a small set of `n_inducing` points
to summarize the data, making it scalable for larger datasets. It typically uses
a non-centered parameterization for efficient sampling.

# Version
v1.0.1 (2026-08-09)

# Mathematical Summary
The SVGP model approximates a full Gaussian Process posterior by introducing a set
of `M` inducing points, \$\\mathbf{Z} = \\{z_1, \\dots, z_M\\}\$. Instead of directly
modeling the full GP over `N` data points, SVGP defines a variational distribution
over the latent function values at these inducing points, \$q(\\mathbf{u}) = \\mathcal{N}(\\mathbf{m}, \\mathbf{S})\$,
where \$\\mathbf{u} = f(\\mathbf{Z})\$. The full latent field \$f(\\mathbf{X})\$ at data
locations \$\\mathbf{X}\$ is then approximated by conditioning on this variational distribution.

In a sampling-based context, as implemented here, the SVGP can be viewed as a
non-centered parameterization of a sparse GP, often equivalent to the Fully
Independent Training Conditional (FITC) approximation. The latent function values
at inducing points, \$\\mathbf{u}\$, are sampled from a distribution whose covariance
is \$K_{UU}\$, the kernel matrix between inducing points. The full latent field
\$f(\\mathbf{X})\$ is then reconstructed as:

\$f(\\mathbf{X}) = K_{XU} K_{UU}^{-1} \\mathbf{u} + \\text{diag}(K_{XX} - K_{XU} K_{UU}^{-1} K_{UX})^{1/2} \\boldsymbol{\\epsilon}\$

where \$\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(0, I)\$, \$K_{UU}\$ is the kernel matrix
between inducing points, \$K_{XU}\$ is the cross-kernel between data and inducing points,
and \$K_{XX}\$ is the kernel matrix between data points. The diagonal term accounts
for the uncertainty not captured by the inducing points.

# Distinction from other GP approximations
- **Nystrom**: Approximates the full kernel matrix \$K_{XX}\$ with a low-rank version
  \$\\tilde{K}_{XX} = K_{XU} K_{UU}^{-1} K_{UX}\$. It's a low-rank approximation of the
  covariance matrix itself.
- **FITC (Fully Independent Training Conditional)**: A sparse GP method that assumes
  data points are conditionally independent given the values at a set of `M` inducing
  points. It approximates the covariance with a low-rank term plus a diagonal
  correction: \$K_{FITC} = K_{XU} K_{UU}^{-1} K_{UX} + \\text{diag}(K_{XX} - Q_{XX})\$,
  where \$Q_{XX}\$ is the Nystrom approximation. This diagonal correction accounts
  for the variance of the data points not captured by the inducing points.
- **SVGP (Sparse Variational Gaussian Process)**: A variational inference method that
  introduces inducing points and optimizes a variational distribution over the GP
  values at these points to approximate the true posterior. In a sampling context,
  the `SVGP` component in `bstm` is implemented similarly to FITC, using a
  non-centered parameterization.
- **RFF (Random Fourier Features)**: Approximates the kernel *function* \$k(x, x')\$
  with a finite-dimensional feature map \$\\phi(x)^T \\phi(x')\$. It transforms the problem
  into a linear model in a high-dimensional feature space.
- **Full GP**: Computes the exact kernel matrix \$K_{XX}\$ and performs inference directly,
  which is \$O(N^3)\$ and memory-intensive (\$O(N^2)\$). SVGP (and FITC) reduce this to
  \$O(NM^2 + M^3)\$ for computation and \$O(NM)\$ for memory.

# Assumptions
- The chosen kernel function is appropriate for the data.
- The `n_inducing` points effectively summarize the GP.

# Best Use Case
Scalable Gaussian Process regression for large datasets where a full GP is
computationally infeasible, offering a more flexible approximation than Nystrom/RFF
by modeling the uncertainty at inducing points.

# Key References
- Titsias, M. (2009). *Variational Learning of Inducing Variables in Sparse Gaussian
  Processes*. PMLR.
- Snelson, E., & Ghahramani, Z. (2006). *Sparse Gaussian Processes using Pseudo-inputs*. NIPS.
- Wikipedia: Gaussian process.

# Fields
- `lengthscale::Union{Distribution, Vector{<:Distribution}}`: Prior for the kernel lengthscale(s).
- `sigma::Distribution`: Prior for the marginal standard deviation of the GP.
- `n_inducing::Int`: Number of inducing points for the approximation.
- `kernel::String`: Name of the kernel function (e.g., "se", "matern32").
"""
struct SVGP <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    n_inducing::Int
    kernel::String
end

COMPONENT_TYPE_REGISTRY[:svgp] = SVGP

COMPONENT_CONSTRUCTORS[:svgp] = (p, params) -> SVGP(
    p.lengthscale,
    p.sigma,
    get(params, :n_inducing, 20),
    string(get(params, :kernel, "se"))
)

MODEL_TO_STRUCTURE_MAP[:svgp] = :smooth

"""
    get_datastructures!(m_type::Type{<:SVGP}, M::Dict, mod_data::Dict)::Bool

Ensures coordinate variables are provided, stores them, and generates the inducing
point locations.
"""
function get_datastructures!(m_type::Type{<:SVGP}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]
    params = mod_data[:params]

    if isempty(variables)
        error("The SVGP model requires coordinate variables, e.g., `random(x, y, model=:svgp)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for SVGP model not found in data.")
        end
    end

    coords = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    mod_data[:params][:coords] = coords

    n_inducing = get(params, :n_inducing, 20)
    knot_method = get(params, :knot_method, :kmeans)
    Z_inducing = generate_inducing_points(coords, n_inducing; method=knot_method)
    mod_data[:params][:Z_inducing] = Z_inducing

    return true
end

"""
    get_precomputes(m::SVGP, M::NamedTuple, mod_data::Dict)::NamedTuple

Stores the coordinate matrix and the inducing point locations for use by the
code generator.
"""
function get_precomputes(m::SVGP, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("SVGP component precomputes failed: coordinates not found.")
    end
    
    Z_inducing = get(mod_data[:params], :Z_inducing, nothing)
    if isnothing(Z_inducing)
        error("SVGP component precomputes failed: inducing points not found.")
    end

    return (
        coords=coords,
        Z_inducing=Z_inducing,
        n_latent=size(coords, 1) # The latent variable `f_raw` is of size N
    )
end
 
"""
    get_priors(m::SVGP, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `sigma`, `lengthscale`, and the raw innovations for both the
inducing points (`u_raw`) and the diagonal correction (`f_raw`).
"""
function get_priors(m::SVGP, spec::NamedTuple, arch::String, outcome_idx, M::NamedTuple)::String
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
    
    # Priors for the latent values at inducing points (`u_raw`) and the final field innovations (`f_raw`).
    push!(priors, "$(p_names.raw) ~ MvNormal(zeros($(m.n_inducing)), I)") # u_raw
    push!(priors, "$(p_names.innov) ~ MvNormal(zeros(spec.precomputes.n_latent), I)") # f_raw

    return join(priors, "\n    ")
end

"""
    get_updates(m::SVGP, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code string for constructing the `SVGP` sparse GP effect.
"""
function get_updates(m::SVGP, spec::NamedTuple, arch::String, outcome_idx, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- SVGP Sparse GP Component: $(spec.key) ---
        local precomputes = spec.hyper
        local X_coords = precomputes.coords
        local Z_coords = precomputes.Z_inducing
        local kernel_type = Symbol("$(m.kernel)")
        
        # 1. Compute kernel matrices
        local K_UU = evaluate_kernel_matrix(Z_coords, $(p_names.sigma), $(p_names.ls), kernel_type, M.noise) # K_UU is M x M
        local K_XU = evaluate_cross_kernel_matrix(X_coords, Z_coords, $(p_names.sigma), $(p_names.ls), kernel_type) # K_XU is N x M
        
        # 2. Sample latent values at inducing points (non-centered)
        local L_UU = cholesky(Symmetric(K_UU)).L # Cholesky of K_UU
        local u_latent = L_UU * $(p_names.raw) # u_latent is M x 1 (p_names.raw corresponds to u_raw)
        
        # 3. Compute conditional mean and variance for SVGP (similar to FITC)
        #    μ_f = K_XU * inv(K_UU) * u_latent
        #    diag_cov_f = diag(K_XX - K_XU * inv(K_UU) * K_XU')
        
        local K_UU_inv_u = K_UU \\ u_latent # K_UU_inv_u is M x 1
        local mean_f = K_XU * K_UU_inv_u # mean_f is N x 1
        
        # Compute diagonal of K_XX - Q_ff efficiently
        # diag(K_XX) is sigma^2 for stationary kernels.
        local diag_K_XX = fill($(p_names.sigma)^2, precomputes.n_latent)
        
        # diag(K_XU * inv(K_UU) * K_XU') = sum((L_UU' \\ K_XU').^2, dims=1)
        local tmp_K_XU_scaled = L_UU' \\ K_XU' # (M x N)
        local diag_Q_ff = sum(tmp_K_XU_scaled.^2, dims=1) # (1 x N)
        
        local lambda_diag = diag_K_XX - vec(diag_Q_ff) # (N x 1)
        
        # 4. Sample final latent field (non-centered)
        $(p_names.latent) = mean_f + sqrt.(max.(lambda_diag, 0.0) .+ M.noise) .* $(p_names.innov) # (N x 1) (p_names.innov corresponds to f_raw)
        
        $(eta_target) .+= $(p_names.latent)
    """
end

"""
    get_effects(m::SVGP, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_total)::NamedTuple

Reconstructs the `SVGP` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::SVGP, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_total)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    # Prepare coordinates for full dataset (training + prediction)
    coords_train = spec.precomputes.coords
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        coords_train
    end
    n_obs_full = size(coords_full, 1)

    Z_inducing = spec.precomputes.Z_inducing
    kernel_type = Symbol(m.kernel)
    noise = M.noise

    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        
        # Extract posterior samples
        sigma_samples = get_params_vector(chain, string(v.sigma), 1)
        ls_samples = get_params_vector(chain, string(v.ls), length(m.lengthscale))
        u_raw_samples = get_params_vector(chain, string(v.raw), m.n_inducing)
        f_innov_samples = get_params_vector(chain, string(v.innov), n_obs_full)

        effect_k = Matrix{Float64}(undef, n_obs_full, n_samples)

        for i in 1:n_samples
            current_sigma = sigma_samples[i, 1]
            current_ls = if m.lengthscale isa Vector
                ls_samples[i, :]
            else
                ls_samples[i, 1]
            end
            current_u_raw = u_raw_samples[i, :]
            current_f_innov = f_innov_samples[i, :]

            # Reconstruct kernel matrices for the current sample
            K_UU = evaluate_kernel_matrix(Z_inducing, current_sigma, current_ls, kernel_type, noise)
            K_XU = evaluate_cross_kernel_matrix(coords_full, Z_inducing, current_sigma, current_ls, kernel_type)
            
            L_UU = cholesky(Symmetric(K_UU)).L
            u_latent = L_UU * current_u_raw
            
            K_UU_inv_u = K_UU \ u_latent
            mean_f = K_XU * K_UU_inv_u
            
            diag_K_XX = fill(current_sigma^2, n_obs_full)
            tmp_K_XU_scaled = L_UU' \ K_XU'
            diag_Q_ff = sum(tmp_K_XU_scaled.^2, dims=1)
            lambda_diag = diag_K_XX - vec(diag_Q_ff)
            
            effect_k[:, i] = mean_f + sqrt.(max.(lambda_diag, 0.0) .+ noise) .* current_f_innov
        end
        push!(structured_effects, effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
