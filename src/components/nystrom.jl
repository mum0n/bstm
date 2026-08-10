"""
    Nystrom <: ComponentModel

A component model for the Nyström sparse Gaussian Process approximation. This method
approximates the full GP kernel matrix with a low-rank version based on a small set
of `n_inducing` points, making it scalable for larger datasets.

# Version
v1.0.1 (2026-08-10)

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

# Computational Methods
- `:noncentered` (default): A non-centered parameterization where the latent values
  at inducing points are constructed from standard normal innovations. Recommended
  for efficient MCMC sampling.
- `:centered`: A centered parameterization where the latent values at inducing points
  are sampled directly from their `MvNormal` distribution. This is a didactic
  alternative that can be less efficient.

# Fields
- `lengthscale::Union{Distribution, Vector{<:Distribution}}`: The prior for the
  kernel lengthscale(s).
- `sigma::Distribution`: The prior for the marginal standard deviation of the GP.
- `n_inducing::Int`: The number of inducing points to use for the approximation.
- `kernel::String`: The name of the kernel function (e.g., "se", "matern32").
- `method::Symbol`: The parameterization method, `:noncentered` or `:centered`.
"""
struct Nystrom <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    n_inducing::Int
    kernel::String
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:nystrom] = Nystrom

COMPONENT_CONSTRUCTORS[:nystrom] = (p, params) -> Nystrom(
    p.lengthscale,
    p.sigma,
    get(params, :n_inducing, 20),
    string(get(params, :kernel, "se")),
    get(params, :method, :noncentered)
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

Generates priors for `sigma` and `lengthscale`. For the `:noncentered` method,
it also defines a prior for the `raw` innovations for the inducing points.
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
    
    if m.method == :noncentered
        push!(priors, "$(p_names.raw) ~ MvNormal(zeros($(m.n_inducing)), I)")
    end

    return join(priors, "\n    ")
end


"""
    get_updates(m::Nystrom, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for constructing the `Nystrom` sparse GP effect,
dispatching on the chosen method.
"""
function get_updates(
    m::Nystrom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
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
    """

    noncentered_code = """
        # --- Nystrom Sparse GP (Non-Centered): $(spec.key) ---
        let
            $(common_code)
            local L_UU = cholesky(Symmetric(K_UU)).L
            local u_latent = L_UU * $(p_names.raw)
            $(p_names.latent) = K_XU * (K_UU \\ u_latent)
            $(eta_target) .+= $(p_names.latent)
        end
    """

    centered_code = """
        # --- Nystrom Sparse GP (Centered): $(spec.key) ---
        let
            $(common_code)
            local u_latent ~ MvNormal(zeros($(m.n_inducing)), Symmetric(K_UU))
            $(p_names.latent) = K_XU * (K_UU \\ u_latent)
            $(eta_target) .+= $(p_names.latent)
        end
    """

    if m.method == :noncentered
        return noncentered_code
    elseif m.method == :centered
        return centered_code
    else
        error("Unsupported method '$(m.method)' for Nystrom component.")
    end
end

"""
    get_effects(m::Nystrom, chain, M::NamedTuple, ...)

Reconstructs the `Nystrom` component's effect from posterior samples, dispatching
on the method used during sampling.
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
            chain, string(p_names.ls), m.lengthscale isa Vector ? length(m.lengthscale) : 1
        )

        effect_k = zeros(Float64, n_obs_full, n_samples)

        if m.method == :noncentered
            raw_samples = get_params_vector(chain, string(p_names.raw), m.n_inducing)
            for i in 1:n_samples
                current_sigma = sigma_samples[i]
                current_ls = m.lengthscale isa Vector ? ls_samples[i, :] : ls_samples[i, 1]
                
                K_UU = evaluate_kernel_matrix(Z_inducing, current_sigma, current_ls, kernel_type, noise)
                K_XU = evaluate_cross_kernel_matrix(coords_full, Z_inducing, current_sigma, current_ls, kernel_type)
                
                L_UU = cholesky(Symmetric(K_UU)).L
                u_latent = L_UU * raw_samples[i, :]
                effect_k[:, i] = K_XU * (K_UU \ u_latent)
            end
        else # :centered
            # For centered, the latent values at inducing points are sampled directly.
            u_latent_samples = get_params_vector(chain, string(p_names.latent), m.n_inducing)
            for i in 1:n_samples
                current_sigma = sigma_samples[i]
                current_ls = m.lengthscale isa Vector ? ls_samples[i, :] : ls_samples[i, 1]
                
                K_UU = evaluate_kernel_matrix(Z_inducing, current_sigma, current_ls, kernel_type, noise)
                K_XU = evaluate_cross_kernel_matrix(coords_full, Z_inducing, current_sigma, current_ls, kernel_type)
                
                effect_k[:, i] = K_XU * (K_UU \ u_latent_samples[i, :])
            end
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
