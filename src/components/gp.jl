"""
    GP <: ComponentModel

A component model for a full Gaussian Process, typically used for smooth effects.
It computes a dense covariance matrix based on a specified kernel function. This
component is powerful but computationally intensive (\$O(N^3)\$), making it suitable
for datasets of small to moderate size. For larger datasets, consider scalable
approximations like `random(..., model=:rff)` or `random(..., model=:fitc)`.

# Version
v1.0.2 (2026-08-08)

# Mathematical Summary
The component models a latent field \$f(x)\$ as a draw from a Gaussian Process with
a zero mean and a specified covariance function (kernel):
\$f(x) \\sim \\mathcal{GP}(0, k(x, x'))\$

The kernel \$k(x, x')\$ defines the covariance between any two points. For example,
the Squared Exponential (SE) kernel is:
\$k(x, x') = \\sigma^2 \\exp\\left(-\frac{\\|x - x'\\|^2}{2\\ell^2}\\right)\$
where:
- \$\\sigma^2\$ is the marginal variance.
- \$\\ell\$ is the characteristic lengthscale.
- \$\\|x - x'\\|\$ is the Euclidean distance between points.

The model samples the latent field \$f\$ from the resulting multivariate normal
distribution \$f \\sim \\mathcal{N}(0, K)\$, where \$K\$ is the dense covariance matrix
evaluated at all data points.

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
- The underlying process is stationary (correlation depends only on distance).
- The chosen kernel correctly reflects the smoothness properties of the process.

# Best Use Case
Modeling smooth, non-linear effects of continuous covariates when the number of
observations is not prohibitively large (e.g., < 2000). It is the gold standard
for non-parametric regression. For larger datasets, a spectral approximation like
`random(..., model=:rff)` is recommended.

# Key References
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for
  Machine Learning*. MIT Press.
- Wikipedia: Gaussian process

# Fields
- `lengthscale::Union{Distribution, Vector{<:Distribution}}`: The prior for the
  kernel lengthscale(s). Can be a single distribution for isotropic kernels or a
  vector for ARD kernels.
- `sigma::Distribution`: The prior for the marginal standard deviation of the GP.
- `kernel::String`: The name of the kernel function (e.g., "se", "matern32").
"""
struct GP <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    kernel::String
end

COMPONENT_TYPE_REGISTRY[:gp] = GP

COMPONENT_CONSTRUCTORS[:gp] = (p, params) -> GP(
    p.lengthscale, p.sigma, string(get(params, :kernel, "se"))
)

MODEL_TO_STRUCTURE_MAP[:gp] = :smooth

"""
    get_datastructures!(m_type::Type{<:GP}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup. It ensures that coordinate variables are provided
and stores them in the module's parameter dictionary.
"""
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

    mod_data[:params][:coords] = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    return true
end

"""
    get_precomputes(m::GP, M::NamedTuple, mod_data::Dict)::NamedTuple

Stores the coordinate matrix and latent dimension in the component's `hyper`
registry. The full covariance matrix is constructed dynamically within the model.
"""
function get_precomputes(m::GP, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("GP component precomputes failed: coordinates not found in module data.")
    end
    
    n_latent = size(coords, 1)
    return (coords=coords, n_latent=n_latent)
end

"""
    get_priors(m::GP, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `sigma`, `lengthscale` (`ls`), and the `raw` innovations.
"""
function get_priors(
    m::GP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    priors = String[]
    push!(priors, "$(p_names.sigma) ~ $(sigma_prior_str)")

    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors, "$(p_names.ls) ~ Product([$(ls_priors_str)])")
    else
        ls_prior_str = _distribution_to_string(m.lengthscale)
        push!(priors, "$(p_names.ls) ~ $(ls_prior_str)")
    end
    
    push!(
        priors,
        "$(p_names.raw) ~ MvNormal(zeros(spec.precomputes.n_latent), I)"
    )

    return join(priors, "\n    ")
end

"""
    get_updates(m::GP, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to compute the kernel matrix, perform a Cholesky decomposition,
and sample the latent field using a non-centered parameterization.
"""
function get_updates(
    m::GP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- Full Gaussian Process (GP) Component: $(spec.key) ---
        let
            local coords = spec.hyper.coords
            local kernel_type = Symbol("$(m.kernel)")

            local K_mat = evaluate_kernel_matrix(
                coords, $(p_names.sigma), $(p_names.ls), kernel_type, M.noise
            )
            
            local F_gp = cholesky(Symmetric(K_mat))
            local $(p_names.latent) = F_gp.L * $(p_names.raw)
            
            $(eta_target) .+= $(p_names.latent)
        end
    """
end

"""
    get_effects(m::GP, chain, M::NamedTuple, ...)::NamedTuple

Reconstructs the `GP` component's effect from the MCMC chain's posterior samples
by re-evaluating the kernel for each sample.
"""
function get_effects(
    m::GP, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    coord_vars = get(spec.params, :positional_args, [])
    coords_train = spec.precomputes.coords
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        coords_train
    end
    n_obs_full = size(coords_full, 1)
    
    noise = M.noise
    kernel_type = Symbol(m.kernel)

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        ls_samples = get_params_vector(
            chain, string(p_names.ls), length(m.lengthscale)
        )
        raw_samples = get_params_vector(
            chain, string(p_names.raw), spec.precomputes.n_latent
        )

        effect_k = zeros(Float64, n_obs_full, n_samples)

        for i in 1:n_samples
            current_sigma = sigma_samples[i]
            current_ls = m.lengthscale isa Vector ? ls_samples[i, :] : ls_samples[i]
            
            K_mat = evaluate_kernel_matrix(
                coords_full, current_sigma, current_ls, kernel_type, noise
            )
            F = cholesky(Symmetric(K_mat))
            
            raw_i = if size(raw_samples, 2) == n_obs_full
                raw_samples[i, :]
            else
                vcat(raw_samples[i, :], randn(n_obs_full - size(raw_samples, 2)))
            end
            
            effect_k[:, i] = F.L * raw_i
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
