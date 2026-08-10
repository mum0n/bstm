"""
    FITC <: ComponentModel

A component model for sparse Gaussian Processes. It supports two common
approximations: FITC (Fully Independent Training Conditional) and VFE
(Variational Free Energy), also known as DTC.

# Version
v1.0.1 (2026-08-10)

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

# Best Use Case
Scalable Gaussian Process regression for large datasets where a full GP is
computationally infeasible. `:fitc` is generally preferred for its more accurate
variance estimates.

# Key References
- Snelson, E., & Ghahramani, Z. (2006). *Sparse Gaussian Processes using
  Pseudo-inputs*. In Advances in neural information processing systems, 18.
- Titsias, M. (2009). *Variational learning of inducing variables in sparse
  Gaussian processes*. In AISTATS.

# Fields
- `lengthscale::Union{Distribution, Vector{<:Distribution}}`: Prior for the kernel lengthscale(s).
- `sigma::Distribution`: Prior for the marginal standard deviation of the GP.
- `n_inducing::Int`: The number of inducing points.
- `kernel::String`: The name of the kernel function (e.g., "se", "matern32").
- `method::Symbol`: The approximation method, `:fitc` (default) or `:vfe`.
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

Generates priors for `sigma`, `lengthscale`, and raw innovations. The `innov`
prior (for diagonal correction) is only included for the `:fitc` method.
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
    
    # Prior for innovations at inducing points
    push!(priors, "$(p_names.raw) ~ MvNormal(zeros($(m.n_inducing)), I)")
    
    # Prior for diagonal correction innovations (only for FITC)
    if m.method == :fitc
        push!(
            priors,
            "$(p_names.innov) ~ MvNormal(zeros(spec.precomputes.n_latent), I)"
        )
    end

    return join(priors, "\n    ")
end

"""
    get_updates(m::FITC, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates Turing code for the sparse GP effect, dispatching on the chosen method.
"""
function get_updates(
    m::FITC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    common_code = """
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
    """

    fitc_code = """
        # --- FITC Sparse GP Component: $(spec.key) ---
        let
            $(common_code)
            
            local K_UU_inv_u = K_UU \\ u_latent
            local mean_f = K_XU * K_UU_inv_u
            
            local diag_K_XX = fill($(p_names.sigma)^2, precomputes.n_latent)
            local tmp = (L_UU' \\ K_XU')'
            local diag_Q_ff = sum(tmp.^2, dims=2)
            local lambda_diag = diag_K_XX - vec(diag_Q_ff)
            
            $(p_names.latent) = mean_f .+
                sqrt.(max.(lambda_diag, 0.0) .+ M.noise) .* $(p_names.innov)
            
            $(eta_target) .+= $(p_names.latent)
        end
    """

    vfe_code = """
        # --- VFE/DTC Sparse GP Component: $(spec.key) ---
        # This is a didactic alternative to FITC that uses a pure low-rank approximation.
        let
            $(common_code)
            
            # The VFE approximation is the conditional mean of the GP given the inducing points.
            # f ≈ K_XU * inv(K_UU) * u
            # This is a low-rank approximation of the full GP.
            local K_UU_inv_u = K_UU \\ u_latent
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

"""
    get_effects(m::FITC, chain, M::NamedTuple, ...)

Reconstructs the `FITC` component's effect from posterior samples, dispatching
on the method used during sampling.
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
            chain, string(p_names.ls), m.lengthscale isa Vector ? length(m.lengthscale) : 1
        )
        u_raw_samples = get_params_vector(chain, string(p_names.raw), m.n_inducing)

        effect_k = zeros(Float64, n_obs_full, n_samples)

        for i in 1:n_samples
            current_sigma = sigma_samples[i]
            current_ls = m.lengthscale isa Vector ? ls_samples[i, :] : ls_samples[i, 1]
            current_u_raw = u_raw_samples[i, :]
            
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

            if m.method == :fitc
                f_innov_samples = get_params_vector(
                    chain, string(p_names.innov), spec.precomputes.n_latent
                )
                f_innov_i = if size(f_innov_samples, 2) == n_obs_full
                    f_innov_samples[i, :]
                else
                    vcat(f_innov_samples[i, :], randn(n_obs_full - size(f_innov_samples, 2)))
                end

                diag_K_XX = fill(current_sigma^2, n_obs_full)
                tmp = (L_UU' \ K_XU')'
                diag_Q_ff = sum(tmp.^2, dims=2)
                lambda_diag = diag_K_XX - vec(diag_Q_ff)
                
                effect_k[:, i] = mean_f .+ sqrt.(max.(lambda_diag, 0.0) .+ noise) .* f_innov_i
            else # :vfe
                effect_k[:, i] = mean_f
            end
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
