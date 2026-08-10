"""
    Kriging <: ComponentModel

A component model for Kriging, which is the geostatistical term for Gaussian
Process (GP) regression. It models a latent field by computing a dense covariance
matrix based on a specified kernel function and coordinate inputs.

# Version
v1.0.1 (2026-08-10)

# Mathematical Summary
The component models a latent field \$f(x)\$ as a draw from a Gaussian Process with
a zero mean and a specified covariance function (kernel):
\$f(x) \\sim \\mathcal{GP}(0, k(x, x'))\$

The kernel \$k(x, x')\$ defines the covariance between any two points. For example,
the Squared Exponential (SE) kernel is:
\$k(x, x') = \\sigma^2 \\exp\\left(-\\frac{\\|x - x'\\|^2}{2\\ell^2}\\right)\$
where:
- \$\\sigma^2\$ is the marginal variance (sill).
- \$\\ell\$ is the characteristic lengthscale (range).
- \$\\|x - x'\\|\$ is the Euclidean distance between points.

The model samples the latent field \$f\$ from the resulting multivariate normal
distribution \$f \\sim \\mathcal{N}(0, K)\$, where \$K\$ is the dense covariance matrix
evaluated at all data points.

# Assumptions
- The underlying process is stationary (correlation depends only on distance).
- The chosen kernel correctly reflects the smoothness properties of the process.

# Best Use Case
Standard non-parametric regression and spatial interpolation for small to moderate
datasets where the computational cost of a full GP is acceptable. For larger
datasets, consider scalable approximations like `random(..., model=:rff)` or
`random(..., model=:fitc)`.

# Key References
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for
  Machine Learning*. MIT Press.
- Wikipedia: Kriging

# Fields
- `lengthscale::Union{Distribution, Vector{<:Distribution}}`: The prior for the
  kernel lengthscale(s). Can be a single distribution for isotropic kernels or a
  vector for ARD kernels.
- `sigma::Distribution`: The prior for the marginal standard deviation of the GP.
- `kernel::String`: The name of the kernel function (e.g., "se", "matern32").
- `method::Symbol`: The parameterization method. Can be `:noncentered` (default,
  recommended) or `:centered` (didactic alternative).
"""
struct Kriging <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    kernel::String
    method::Symbol
end

# Add to the central component constructor registry.
COMPONENT_TYPE_REGISTRY[:kriging] = Kriging
COMPONENT_CONSTRUCTORS[:kriging] = (p, params) -> Kriging(
    p.lengthscale, p.sigma, string(get(params, :kernel, "se")),
    get(params, :method, :noncentered)
)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[:kriging] = :smooth

"""
    get_datastructures!(m_type::Type{<:Kriging}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Kriging` component.
It ensures that coordinate variables are provided and stores them in the module data.
"""
function get_datastructures!(
    m_type::Type{<:Kriging}, M::Dict, mod_data::Dict
)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error(
            "The Kriging model requires coordinate variables, e.g., " *
            "`random(x, y, model=:kriging)`."
        )
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Kriging model not found in data.")
        end
    end

    mod_data[:params][:coords] = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    return true
end

"""
    get_precomputes(m::Kriging, M::NamedTuple, mod_data::Dict)::NamedTuple

For the `Kriging` component, this function stores the coordinate matrix.
The full covariance matrix is constructed dynamically within the model.
"""
function get_precomputes(m::Kriging, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("Kriging precomputes failed: coordinates not found in module data.")
    end
    
    n_latent = size(coords, 1)
    return (coords=coords, n_latent=n_latent)
end

"""
    get_priors(m::Kriging, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `sigma` and `lengthscale` (`ls`). For the `:noncentered`
method, it also defines a prior for the `raw` innovations.
"""
function get_priors(
    m::Kriging, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
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
        push!(
            priors,
            "$(p_names.raw) ~ MvNormal(zeros(spec.precomputes.n_latent), I)"
        )
    end

    return join(priors, "\n    ")
end

"""
    get_updates(m::Kriging, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to construct the Kriging effect. Supports two methods:
- `:noncentered` (default): Samples standard normal noise and transforms it. This
  is generally more efficient for MCMC.
- `:centered`: Samples the latent field directly from the `MvNormal` distribution.
  This can be less efficient due to strong posterior correlations but is a useful
  didactic alternative.
"""
function get_updates(
    m::Kriging, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    common_code = """
        local coords = spec_registry[:$(spec.key)].precomputes.coords
        local kernel_type = Symbol("$(m.kernel)")

        local K_mat = evaluate_kernel_matrix(
            coords, $(p_names.sigma), $(p_names.ls), kernel_type, M.noise
        )
    """

    noncentered_code = """
        # --- Kriging (Non-Centered): $(spec.key) ---
        let
            $(common_code)
            local F_krig = cholesky(Symmetric(K_mat))
            $(p_names.latent) = F_krig.L * $(p_names.raw)
            $(eta_target) .+= $(p_names.latent)
        end
    """

    centered_code = """
        # --- Kriging (Centered): $(spec.key) ---
        let
            $(common_code)
            $(p_names.latent) ~ MvNormal(zeros(size(K_mat, 1)), Symmetric(K_mat))
            $(eta_target) .+= $(p_names.latent)
        end
    """

    if m.method == :noncentered
        return noncentered_code
    elseif m.method == :centered
        return centered_code
    else
        error("Unsupported method '$(m.method)' for Kriging component.")
    end
end

"""
    get_effects(m::Kriging, chain, M::NamedTuple, ...)

Reconstructs the `Kriging` component's effect from posterior samples,
dispatching on the `method` used during sampling.
"""
function get_effects(
    m::Kriging, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
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
    n_obs_train = size(coords_train, 1)
    n_obs_full = size(coords_full, 1)
    
    noise = M.noise
    kernel_type = Symbol(m.kernel)

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        ls_samples = get_params_vector(
            chain, string(p_names.ls), m.lengthscale isa Vector ? length(m.lengthscale) : 1
        )

        effect_k = zeros(Float64, n_obs_full, n_samples)

        if m.method == :noncentered
            raw_samples = get_params_vector(chain, string(p_names.raw), n_obs_train)
            for i in 1:n_samples
                K_mat = evaluate_kernel_matrix(coords_full, sigma_samples[i], ls_samples[i,:], kernel_type, noise)
                F = cholesky(Symmetric(K_mat))
                raw_i = vcat(raw_samples[i, :], randn(n_obs_full - n_obs_train))
                effect_k[:, i] = F.L * raw_i
            end
        elseif m.method == :centered
            latent_samples = get_params_vector(chain, string(p_names.latent), n_obs_train)
            for i in 1:n_samples
                effect_k[1:n_obs_train, i] = latent_samples[i, :]
                if n_obs_full > n_obs_train
                    coords_pred = coords_full[(n_obs_train+1):end, :]
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
