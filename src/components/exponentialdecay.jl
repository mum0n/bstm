"""
    ExponentialDecay <: ComponentModel

A component model for a full Gaussian Process with an exponential covariance function.
The exponential kernel models correlation that decays exponentially with distance,
representing a continuous but not smooth (non-differentiable) process.

# Version
v1.0.1 (2026-08-10)

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
- `method::Symbol`: The parameterization method. Can be `:noncentered` (default,
  recommended) or `:centered` (didactic alternative).
"""
struct ExponentialDecay <: ComponentModel
    sigma::Distribution
    lengthscale::Distribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:exponentialdecay] = ExponentialDecay
COMPONENT_CONSTRUCTORS[:exponentialdecay] = (p, params) -> ExponentialDecay(
    p.sigma, p.lengthscale, get(params, :method, :noncentered)
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

Generates priors for `sigma` and `lengthscale`. For the `:noncentered` method,
it also defines a prior for the `raw` innovations.
"""
function get_priors(
    m::ExponentialDecay, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = [
        "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))",
        "$(p_names.ls) ~ $(_distribution_to_string(m.lengthscale))"
    ]

    if m.method == :noncentered
        push!(priors, "$(p_names.raw) ~ MvNormal(zeros(spec.precomputes.n_latent), I)")
    end

    return join(priors, "\n    ")
end



"""
    get_updates(m::ExponentialDecay, spec::NamedTuple, arch::String, outcome_idx, M)

Generates code to construct the Exponential Decay GP effect. Supports two methods:
- `:noncentered` (default): Samples standard normal noise and transforms it.
- `:centered`: Samples the latent field directly from the `MvNormal` distribution.
"""
function get_updates(
    m::ExponentialDecay, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    common_code = """
        local coords = spec_registry[:$(spec.key)].precomputes.coords
        local dist_matrix = pairwise(Euclidean(), coords', dims=2)
        local K = ($(p_names.sigma)^2) .* exp.(-dist_matrix ./ $(p_names.ls))
        local K_stable = K + Diagonal(fill(M.noise, size(K, 1)))
    """

    noncentered_code = """
        # --- Exponential Decay GP (Non-Centered): $(spec.key) ---
        let
            $(common_code)
            local F = cholesky(Symmetric(K_stable))
            $(p_names.latent) = F.L * $(p_names.raw)
            $(eta_target) .+= $(p_names.latent)
        end
    """

    centered_code = """
        # --- Exponential Decay GP (Centered): $(spec.key) ---
        let
            $(common_code)
            $(p_names.latent) ~ MvNormal(zeros(size(K_stable, 1)), Symmetric(K_stable))
            $(eta_target) .+= $(p_names.latent)
        end
    """

    if m.method == :noncentered
        return noncentered_code
    elseif m.method == :centered
        return centered_code
    else
        error("Unsupported method '$(m.method)' for ExponentialDecay component.")
    end
end

"""
    get_effects(m::ExponentialDecay, chain, M::NamedTuple, ...)

Reconstructs the `ExponentialDecay` GP effect from posterior samples,
dispatching on the method used during sampling.
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
    n_obs_train = size(coords_train, 1)
    n_obs_full = size(coords_full, 1)
    noise = M.noise

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        ls_samples = get_params_vector(chain, string(p_names.ls), 1)[:, 1]

        effect_k = zeros(Float64, n_obs_full, n_samples)

        if m.method == :noncentered
            raw_samples = get_params_vector(chain, string(p_names.raw), n_obs_train)
            dist_matrix_full = pairwise(Euclidean(), coords_full', dims=2)
            for i in 1:n_samples
                K = (sigma_samples[i]^2) .* exp.(-dist_matrix_full ./ ls_samples[i])
                K_stable = K + Diagonal(fill(noise, n_obs_full))
                F = cholesky(Symmetric(K_stable))
                raw_i = vcat(raw_samples[i, :], randn(n_obs_full - n_obs_train))
                effect_k[:, i] = F.L * raw_i
            end
        else # :centered
            latent_samples = get_params_vector(chain, string(p_names.latent), n_obs_train)
            dist_matrix_train = pairwise(Euclidean(), coords_train', dims=2)
            for i in 1:n_samples
                effect_k[1:n_obs_train, i] = latent_samples[i, :]
                if n_obs_full > n_obs_train
                    coords_pred = coords_full[(n_obs_train+1):end, :]
                    dist_pred_train = pairwise(Euclidean(), coords_pred', coords_train', dims=2)
                    dist_pred_pred = pairwise(Euclidean(), coords_pred', dims=2)

                    K_ff = (sigma_samples[i]^2) .* exp.(-dist_matrix_train ./ ls_samples[i]) .+ Diagonal(fill(noise, n_obs_train))
                    K_star_f = (sigma_samples[i]^2) .* exp.(-dist_pred_train ./ ls_samples[i])
                    K_star_star = (sigma_samples[i]^2) .* exp.(-dist_pred_pred ./ ls_samples[i]) .+ Diagonal(fill(noise, size(coords_pred, 1)))
                    
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
