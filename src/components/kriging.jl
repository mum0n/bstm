
"""
    Kriging <: ComponentModel

A component model for Kriging, which is the geostatistical term for Gaussian
Process (GP) regression. It models a latent field by computing a dense covariance
matrix based on a specified kernel function and coordinate inputs.

# Version
v1.0.0 (2026-08-08)

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
"""
struct Kriging <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    kernel::String
end

# Add to the central component constructor registry.
COMPONENT_TYPE_REGISTRY[:kriging] = Kriging
COMPONENT_CONSTRUCTORS[:kriging] = (p, params) -> Kriging(
    p.lengthscale, p.sigma, string(get(params, :kernel, "se"))
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

Generates priors for `sigma`, `lengthscale` (`ls`), and the `raw` innovations.
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
    
    push!(
        priors,
        "$(p_names.raw) ~ MvNormal(zeros(spec.precomputes.n_latent), I)"
    )

    return join(priors, "\n    ")
end

"""
    get_updates(m::Kriging, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to compute the kernel matrix, perform a Cholesky decomposition,
and sample the latent field using a non-centered parameterization.
"""
function get_updates(
    m::Kriging, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- Kriging (Full GP) Component: $(spec.key) ---
        let
            local coords = spec_registry[:$(spec.key)].precomputes.coords
            local kernel_type = Symbol("$(m.kernel)")

            local K_mat = evaluate_kernel_matrix(
                coords, $(p_names.sigma), $(p_names.ls), kernel_type, M.noise
            )
            
            local F_krig = cholesky(Symmetric(K_mat))
            local $(p_names.latent) = F_krig.L * $(p_names.raw)
            
            $(eta_target) .+= $(p_names.latent)
        end
    """
end

"""
    get_effects(m::Kriging, chain, M::NamedTuple, ...)

Reconstructs the `Kriging` component's effect from the MCMC chain's posterior
samples by re-evaluating the kernel for each sample.
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
