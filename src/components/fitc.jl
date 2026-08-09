
"""
    FITC <: ComponentModel

A component model for the Fully Independent Training Conditional (FITC) sparse
Gaussian Process. This method approximates a full GP by using a small set of
`n_inducing` points to summarize the data, making it scalable for larger datasets.

# Version
v1.0.0 (2026-08-08)

# Mathematical Summary
The FITC approximation assumes that, conditional on the GP's values at a set of
\$M\$ inducing points \$Z\$, the values at the \$N\$ data points \$X\$ are independent.
The latent GP values \$f\$ are modeled as:
\$f \\sim \\mathcal{N}(\\mu_f, \\Sigma_f)\$
where the conditional mean and covariance are:
- \$\\mu_f = K_{XZ} K_{ZZ}^{-1} u\$
- \$\\Sigma_f = \\text{diag}(K_{XX} - Q_{XX}) + \\sigma_n^2 I\$
and
- \$u \\sim \\mathcal{N}(0, K_{ZZ})\$ are the latent values at the inducing points.
- \$Q_{XX} = K_{XZ} K_{ZZ}^{-1} K_{ZX}\$.
- \$K_{XZ}\$ is the cross-covariance between data and inducing points.
- \$K_{ZZ}\$ is the covariance of the inducing points.

This implementation uses a non-centered parameterization for efficient sampling.

# Assumptions
- The data points are conditionally independent given the inducing points.
- The number of inducing points `n_inducing` is much smaller than the number of
  data points.

# Best Use Case
Scalable Gaussian Process regression for large datasets where a full GP is
computationally infeasible. It is a good general-purpose sparse GP method.

# Key References
- Snelson, E., & Ghahramani, Z. (2006). *Sparse Gaussian Processes using
  Pseudo-inputs*. In Advances in neural information processing systems, 18.
- Wikipedia: Sparse Gaussian process

# Fields
- `lengthscale::Union{Distribution, Vector{<:Distribution}}`: The prior for the
  kernel lengthscale(s).
- `sigma::Distribution`: The prior for the marginal standard deviation of the GP.
- `n_inducing::Int`: The number of inducing points to use for the approximation.
- `kernel::String`: The name of the kernel function (e.g., "se", "matern32").
"""
struct FITC <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    n_inducing::Int
    kernel::String
end

COMPONENT_TYPE_REGISTRY[:fitc] = FITC

COMPONENT_CONSTRUCTORS[:fitc] = (p, params) -> FITC(
    p.lengthscale,
    p.sigma,
    get(params, :n_inducing, 20),
    string(get(params, :kernel, "se"))
)

MODEL_TO_STRUCTURE_MAP[:fitc] = :smooth

"""
    get_datastructures!(m_type::Type{<:FITC}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `FITC` component. It ensures coordinate
variables are provided, stores them, and generates the inducing point locations.

# Assumptions
- The `random()` call provides one or more variables representing the coordinates.
"""
function get_datastructures!(m_type::Type{<:FITC}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]
    params = mod_data[:params]

    if isempty(variables)
        error("The FITC model requires coordinate variables, e.g., `random(x, y, model=:fitc)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for FITC model not found in data.")
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
    get_precomputes(m::FITC, M::NamedTuple, mod_data::Dict)::NamedTuple

For the `FITC` component, this function stores the coordinate matrix and the
inducing point locations for use by the code generator.
"""
function get_precomputes(m::FITC, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("FITC component precomputes failed: coordinates not found.")
    end
    
    Z_inducing = get(mod_data[:params], :Z_inducing, nothing)
    if isnothing(Z_inducing)
        error("FITC component precomputes failed: inducing points not found.")
    end

    return (
        coords=coords,
        Z_inducing=Z_inducing,
        n_latent=size(coords, 1)
    )
end

"""
    get_priors(m::FITC, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `sigma`, `lengthscale`, and the raw innovations for both the
inducing points (`raw`) and the final latent field (`innov`).
"""
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
    
    push!(priors, "$(p_names.raw) ~ MvNormal(zeros(T, $(m.n_inducing)), I)")
    push!(
        priors,
        "$(p_names.innov) ~ MvNormal(zeros(T, spec.precomputes.n_latent), I)"
    )

    return join(priors, "\n    ")
end

"""
    get_updates(m::FITC, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for constructing the `FITC` sparse GP effect.
"""
function get_updates(
    m::FITC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- FITC Sparse GP Component: $(spec.key) ---
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
            local u_latent = L_UU * $(p_names.raw)
            
            local K_UU_inv_u = K_UU \\ u_latent
            local mean_f = K_XU * K_UU_inv_u
            
            local diag_K_XX = fill($(p_names.sigma)^2, precomputes.n_latent)
            local tmp = (L_UU' \\ K_XU')'
            local diag_Q_ff = sum(tmp.^2, dims=2)
            local lambda_diag = diag_K_XX - vec(diag_Q_ff)
            
            local $(p_names.latent) = mean_f .+
                sqrt.(max.(lambda_diag, 0.0) .+ M.noise) .* $(p_names.innov)
            
            $(eta_target) .+= $(p_names.latent)
        end
    """
end

"""
    get_effects(m::FITC, chain, M::NamedTuple, ...)::NamedTuple

Reconstructs the `FITC` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(
    m::FITC, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
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
        u_raw_samples = get_params_vector(chain, string(p_names.raw), m.n_inducing)
        f_innov_samples = get_params_vector(
            chain, string(p_names.innov), spec.precomputes.n_latent
        )

        effect_k = zeros(Float64, n_obs_full, n_samples)

        for i in 1:n_samples
            current_sigma = sigma_samples[i]
            current_ls = m.lengthscale isa Vector ? ls_samples[i, :] : ls_samples[i]
            current_u_raw = u_raw_samples[i, :]
            
            f_innov_i = if size(f_innov_samples, 2) == n_obs_full
                f_innov_samples[i, :]
            else
                vcat(
                    f_innov_samples[i, :],
                    randn(n_obs_full - size(f_innov_samples, 2))
                )
            end

            K_UU = evaluate_kernel_matrix(
                Z_inducing, current_sigma, current_ls, kernel_type, noise
            )
            K_XU = evaluate_cross_kernel_matrix(
                coords_full, Z_inducing, current_sigma, current_ls, kernel_type
            )
            
            L_UU = cholesky(Symmetric(K_UU)).L
            u_latent = L_UU * current_u_raw
            
            K_UU_inv_u = K_UU \ u_latent
            mean_f = K_XU * K_UU_inv_u
            
            diag_K_XX = fill(current_sigma^2, n_obs_full)
            tmp = (L_UU' \ K_XU')'
            diag_Q_ff = sum(tmp.^2, dims=2)
            lambda_diag = diag_K_XX - vec(diag_Q_ff)
            
            effect_k[:, i] = mean_f .+
                sqrt.(max.(lambda_diag, 0.0) .+ noise) .* f_innov_i
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
