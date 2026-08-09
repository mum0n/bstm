
"""
    ExponentialDecay <: ComponentModel

A component model for a full Gaussian Process with an exponential covariance function.
The exponential kernel models correlation that decays exponentially with distance,
representing a continuous but not smooth (non-differentiable) process.

# Version
v1.0.0 (2026-08-08)

# Mathematical Summary
The component models a latent field \$f(x)\$ as a draw from a Gaussian Process with
a zero mean and an exponential covariance function:
\$f(x) \\sim \\mathcal{GP}(0, k(x, x'))\$
The exponential kernel is defined as:
\$k(x, x') = \\sigma^2 \\exp(-\\frac{\\|x - x'\\|}{\\ell})\$
where:
- \$\\sigma^2\$ is the marginal variance.
- \$\\ell\$ is the characteristic lengthscale.
- \$\\|x - x'\\|\$ is the Euclidean distance between points.

This kernel is a special case of the Matérn kernel with smoothness parameter
\$\\nu = 1/2\$.

# Assumptions
- The underlying process is stationary (correlation depends only on distance).
- The process is continuous but not smooth (mean-square non-differentiable).

# Best Use Case
Modeling processes with short-range correlation where the field is expected to be
continuous but rough, such as certain environmental phenomena or financial time
series.

# Key References
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for
  Machine Learning*. MIT Press.
- Wikipedia: Matérn covariance function

# Fields
- `sigma::Distribution`: The prior for the marginal standard deviation of the GP.
- `lengthscale::Distribution`: The prior for the characteristic lengthscale of the
  decay.
"""
struct ExponentialDecay <: ComponentModel
    sigma::Distribution
    lengthscale::Distribution
end

COMPONENT_TYPE_REGISTRY[:exponentialdecay] = ExponentialDecay
COMPONENT_CONSTRUCTORS[:exponentialdecay] = (p, params) -> ExponentialDecay(
    p.sigma, p.lengthscale
)
MODEL_TO_STRUCTURE_MAP[:exponentialdecay] = :smooth

"""
    get_datastructures!(m_type::Type{<:ExponentialDecay}, M::Dict, mod_data::Dict)

Performs data-dependent setup. It ensures that coordinate variables are provided
and stores them in the module's parameter dictionary.

# Assumptions
- The `random()` call provides one or more variables representing the coordinates.
"""
function get_datastructures!(
    m_type::Type{<:ExponentialDecay}, M::Dict, mod_data::Dict
)::Bool
    variables = mod_data[:variables]
    if isempty(variables)
        error(
            "The ExponentialDecay model requires coordinate variables, e.g., " *
            "`random(x, y, model=:exponentialdecay)`."
        )
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error(
                "Coordinate variable ':$var_sym' for ExponentialDecay model not " *
                "found in data."
            )
        end
    end

    mod_data[:params][:coords] = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    return true
end

"""
    get_precomputes(m::ExponentialDecay, M::NamedTuple, mod_data::Dict)::NamedTuple

Stores the coordinate matrix and latent dimension in the component's `hyper`
registry. The full covariance matrix is constructed dynamically within the model.
"""
function get_precomputes(
    m::ExponentialDecay, M::NamedTuple, mod_data::Dict
)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error(
            "ExponentialDecay precomputes failed: coordinates not found in module data."
        )
    end
    
    n_latent = size(coords, 1)
    return (coords=coords, n_latent=n_latent)
end

"""
    get_priors(m::ExponentialDecay, spec::NamedTuple, arch::String, outcome_idx, M)

Generates priors for `sigma`, `lengthscale` (`ls`), and the `raw` innovations.
"""
function get_priors(
    m::ExponentialDecay, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    sigma_prior_str = _distribution_to_string(m.sigma)
    ls_prior_str = _distribution_to_string(m.lengthscale)
    
    return """
        $(p_names.sigma) ~ $(sigma_prior_str)
        $(p_names.ls) ~ $(ls_prior_str)
        $(p_names.raw) ~ MvNormal(
            zeros(T, spec_registry[:$(spec.key)].precomputes.n_latent), I
        )
    """
end

"""
    get_updates(m::ExponentialDecay, spec::NamedTuple, arch::String, outcome_idx, M)

Generates code to compute the exponential kernel matrix, perform a Cholesky
decomposition, and sample the latent field using a non-centered parameterization.
"""
function get_updates(
    m::ExponentialDecay, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- Exponential Decay GP Component: $(spec.key) ---
        let
            local coords = spec_registry[:$(spec.key)].precomputes.coords
            
            # Compute pairwise Euclidean distances
            local dist_matrix = pairwise(Euclidean(), coords', dims=2)
            
            # Compute exponential decay kernel matrix
            local K = ($(p_names.sigma)^2) .* exp.(-dist_matrix ./ $(p_names.ls))
            local K_stable = K + Diagonal(fill(M.noise, size(K, 1)))
            
            local F = cholesky(Symmetric(K_stable))
            local $(p_names.latent) = F.L * $(p_names.raw)
            
            $(eta_target) .+= $(p_names.latent)
        end
    """
end

"""
    get_effects(m::ExponentialDecay, chain, M::NamedTuple, ...)

Reconstructs the `ExponentialDecay` GP effect from the MCMC chain's posterior
samples by re-evaluating the kernel for each sample.
"""
function get_effects(
    m::ExponentialDecay, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
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
    dist_matrix = pairwise(Euclidean(), coords_full', dims=2)

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        ls_samples = get_params_vector(chain, string(p_names.ls), 1)[:, 1]
        raw_samples = get_params_vector(
            chain, string(p_names.raw), spec.precomputes.n_latent
        )

        effect_k = zeros(Float64, n_obs_full, n_samples)

        for i in 1:n_samples
            K = (sigma_samples[i]^2) .* exp.(-dist_matrix ./ ls_samples[i])
            K_stable = K + Diagonal(fill(M.noise, n_obs_full))
            
            F = cholesky(Symmetric(K_stable))
            
            # For prediction, we need to handle the size of raw_samples
            raw_i = if size(raw_samples, 2) == n_obs_full
                raw_samples[i, :]
            else
                # Pad with new random samples for prediction points
                vcat(raw_samples[i, :], randn(n_obs_full - size(raw_samples, 2)))
            end
            
            effect_k[:, i] = F.L * raw_i
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
