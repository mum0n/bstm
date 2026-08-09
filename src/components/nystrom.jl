# src/tmp.jl

"""
    Nystrom <: ComponentModel

A component model for the Nyström sparse Gaussian Process approximation. This method
approximates the full GP kernel matrix with a low-rank version based on a small set
of `n_inducing` points, making it scalable for larger datasets.

# Version
v1.0.0 (2026-08-08)

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
- The number of inducing points `n_inducing` is much smaller than the number of
  data points.
- The chosen kernel function is appropriate for the data.

# Best Use Case
Scalable Gaussian Process regression for large datasets where a full GP is
computationally infeasible. It is particularly effective when the underlying
function is smooth and can be well-represented by a low-rank approximation.

# Key References
- Williams, C. K. I., & Seeger, M. (2001). *Using the Nyström method to speed up
  kernel machines*. In NIPS.
- Wikipedia: Nyström method

# Fields
- `lengthscale::Union{Distribution, Vector{<:Distribution}}`: The prior for the
  kernel lengthscale(s).
- `sigma::Distribution`: The prior for the marginal standard deviation of the GP.
- `n_inducing::Int`: The number of inducing points to use for the approximation.
- `kernel::String`: The name of the kernel function (e.g., "se", "matern32").
"""
struct Nystrom <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    n_inducing::Int
    kernel::String
end

COMPONENT_TYPE_REGISTRY[:nystrom] = Nystrom

COMPONENT_CONSTRUCTORS[:nystrom] = (p, params) -> Nystrom(
    p.lengthscale,
    p.sigma,
    get(params, :n_inducing, 20),
    string(get(params, :kernel, "se"))
)

MODEL_TO_STRUCTURE_MAP[:nystrom] = :smooth

"""
    get_datastructures!(m_type::Type{<:Nystrom}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Nystrom` component. It ensures coordinate
variables are provided, stores them, and generates the inducing point locations.
"""
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
    Z_inducing = generate_inducing_points(coords, n_inducing; method=knot_method)
    mod_data[:params][:Z_inducing] = Z_inducing

    return true
end

"""
    get_precomputes(m::Nystrom, M::NamedTuple, mod_data::Dict)::NamedTuple

For the `Nystrom` component, this function stores the coordinate matrix and the
inducing point locations. The number of latent variables is `n_inducing`.
"""
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
        n_latent=m.n_inducing # The latent variable `u` is of size n_inducing
    )
end

"""
    get_priors(m::Nystrom, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code string for the `Nystrom` component's priors. This includes
priors for `sigma`, `lengthscale`, and the `raw` innovations for the inducing points.
"""
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
    
    push!(priors, "$(p_names.raw) ~ MvNormal(zeros($(m.n_inducing)), I)")

    return join(priors, "\n    ")
end

"""
    get_updates(m::Nystrom, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code string for constructing the `Nystrom` sparse GP effect.
"""
function get_updates(
    m::Nystrom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- Nystrom Sparse GP Component: $(spec.key) ---
        let
            local precomputes = spec_registry[:$(spec.key)].precomputes
            local X_coords = precomputes.coords
            local Z_coords = precomputes.Z_inducing
            local kernel_type = Symbol("$(m.kernel)")
            
            local K_UU = evaluate_kernel_matrix(
                Z_coords, $(p_names.sigma), $(p_names.ls), kernel_type, M.noise
            )
            local K_XU = evaluate_cross_kernel_matrix(
                X_coords, Z_coords, $(p_names.sigma), $(p_names.ls), kernel_type
            )
            
            local L_UU = cholesky(Symmetric(K_UU)).L
            
            local $(p_names.latent) = K_XU * (L_UU' \\ $(p_names.raw))
            
            $(eta_target) .+= $(p_names.latent)
        end
    """
end

"""
    get_effects(m::Nystrom, chain, M::NamedTuple, ...)

Reconstructs the `Nystrom` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(
    m::Nystrom, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
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
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        ls_samples = get_params_vector(
            chain, string(p_names.ls), length(m.lengthscale)
        )
        raw_samples = get_params_vector(chain, string(p_names.raw), m.n_inducing)

        effect_k = zeros(Float64, n_obs_full, n_samples)

        for i in 1:n_samples
            current_sigma = sigma_samples[i]
            current_ls = m.lengthscale isa Vector ? ls_samples[i, :] : ls_samples[i]
            current_raw = raw_samples[i, :]

            K_UU = evaluate_kernel_matrix(
                Z_inducing, current_sigma, current_ls, kernel_type, noise
            )
            K_XU = evaluate_cross_kernel_matrix(
                coords_full, Z_inducing, current_sigma, current_ls, kernel_type
            )
            
            L_UU = cholesky(Symmetric(K_UU)).L
            
            effect_k[:, i] = K_XU * (L_UU' \ current_raw)
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
