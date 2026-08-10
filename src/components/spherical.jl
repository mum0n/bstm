"""
    Spherical <: ComponentModel

A component model for a full Gaussian Process with a spherical covariance function.
The spherical kernel has compact support, meaning correlation drops to zero beyond a
specified `range`.

# Version
v1.0.1 (2026-08-10)

# Mathematical Summary
The component models a latent field \$f(s)\$ as a draw from a Gaussian Process with
a zero mean and a spherical covariance function:
\$f(s) \\sim \\mathcal{GP}(0, k(h))\$
where \$h = \\|s - s'\\|\$ is the Euclidean distance between points. The spherical
kernel is defined as:
\$k(h) = \\sigma^2 \\left( 1 - \\frac{3h}{2r} + \\frac{h^3}{2r^3} \\right)\$ for \$0 \\le h \\le r\$
\$k(h) = 0\$ for \$h > r\$
where:
- \$\\sigma^2\$ is the marginal variance (sill).
- \$r\$ is the range parameter.

# Computational Methods
- `:noncentered` (default): A non-centered parameterization where the latent field
  is constructed from standard normal innovations. Recommended for AD.
- `:centered`: A centered parameterization where the latent field is sampled
  directly from the `MvNormal` distribution. Didactic, can be less efficient.

# Fields
- `sigma::Distribution`: The prior for the marginal standard deviation of the GP.
- `range::Distribution`: The prior for the effective range of spatial correlation.
- `method::Symbol`: The computational method, `:noncentered` or `:centered`.
"""
struct Spherical <: ComponentModel
    sigma::Distribution
    range::Distribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:spherical] = Spherical
COMPONENT_CONSTRUCTORS[:spherical] = (p, params) -> Spherical(
    p.sigma, p.range, get(params, :method, :noncentered)
)

MODEL_TO_STRUCTURE_MAP[:spherical] = :smooth

"""
    get_datastructures!(m_type::Type{<:Spherical}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Spherical` component.
It ensures that coordinate variables are provided and stores them in the module data.
"""
function get_datastructures!(
    m_type::Type{<:Spherical}, M::Dict, mod_data::Dict
)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error(
            "The Spherical GP model requires coordinate variables, e.g., " *
            "`random(x, y, model=:spherical)`."
        )
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Spherical GP not found in data.")
        end
    end

    mod_data[:params][:coords] = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    return true
end

"""
    get_precomputes(m::Spherical, M::NamedTuple, mod_data::Dict)::NamedTuple

For the `Spherical` component, this function stores the coordinate matrix.
The full covariance matrix is constructed dynamically within the model.
"""
function get_precomputes(m::Spherical, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("Spherical component precomputes failed: coordinates not found.")
    end
    
    n_latent = size(coords, 1)
    return (coords=coords, n_latent=n_latent)
end
 
"""
    get_priors(m::Spherical, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `sigma` and `range`. For the `:noncentered` method, it also
defines a prior for the `raw` innovations.
"""
function get_priors(
    m::Spherical, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = [
        "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))",
        "$(p_names.range) ~ $(_distribution_to_string(m.range))"
    ]

    if m.method == :noncentered
        push!(
            priors,
            "$(p_names.raw) ~ MvNormal(zeros(spec.precomputes.n_latent), I)"
        )
    end

    return join(priors, "\n    ")
end

"""
    get_updates(m::Spherical, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to construct the Spherical GP effect, dispatching on the chosen method.
"""
function get_updates(
    m::Spherical, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    common_code = """
        local coords = spec_registry[:$(spec.key)].precomputes.coords
        local K = evaluate_spherical_kernel_matrix(
            coords, $(p_names.sigma), $(p_names.range), M.noise
        )
    """

    noncentered_code = """
        # --- Spherical GP Component (Non-Centered): $(spec.key) ---
        let
            $(common_code)
            local F = cholesky(Symmetric(K))
            $(p_names.latent) = F.L * $(p_names.raw)
            $(eta_target) .+= $(p_names.latent)
        end
    """

    centered_code = """
        # --- Spherical GP Component (Centered): $(spec.key) ---
        let
            $(common_code)
            $(p_names.latent) ~ MvNormal(zeros(size(K, 1)), Symmetric(K))
            $(eta_target) .+= $(p_names.latent)
        end
    """

    if m.method == :noncentered
        return noncentered_code
    elseif m.method == :centered
        return centered_code
    else
        error("Unsupported method '$(m.method)' for Spherical component.")
    end
end

"""
    get_effects(m::Spherical, chain, M::NamedTuple, ...)

Reconstructs the `Spherical` GP effect from posterior samples, dispatching on the
method used during sampling.
"""
function get_effects(
    m::Spherical, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
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
        range_samples = get_params_vector(chain, string(p_names.range), 1)[:, 1]

        effect_k = zeros(Float64, n_obs_full, n_samples)

        if m.method == :noncentered
            raw_samples = get_params_vector(chain, string(p_names.raw), n_obs_train)
            for i in 1:n_samples
                K = evaluate_spherical_kernel_matrix(
                    coords_full, sigma_samples[i], range_samples[i], noise
                )
                F = cholesky(Symmetric(K))
                raw_i = vcat(raw_samples[i, :], randn(n_obs_full - n_obs_train))
                effect_k[:, i] = F.L * raw_i
            end
        else # :centered
            latent_samples = get_params_vector(chain, string(p_names.latent), n_obs_train)
            for i in 1:n_samples
                effect_k[1:n_obs_train, i] = latent_samples[i, :]
                if n_obs_full > n_obs_train
                    coords_pred = coords_full[(n_obs_train+1):end, :]
                    K_ff = evaluate_spherical_kernel_matrix(
                        coords_train, sigma_samples[i], range_samples[i], noise
                    )
                    K_star_f = evaluate_cross_spherical_kernel_matrix(
                        coords_pred, coords_train, sigma_samples[i], range_samples[i]
                    )
                    K_star_star = evaluate_spherical_kernel_matrix(
                        coords_pred, sigma_samples[i], range_samples[i], noise
                    )
                    
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

"""
    evaluate_cross_spherical_kernel_matrix(coords1, coords2, sigma, range_param)

A helper function to compute the cross-covariance matrix for a spherical kernel.
"""
function evaluate_cross_spherical_kernel_matrix(
    coords1::AbstractMatrix, coords2::AbstractMatrix, sigma::Real, range_param::Real
)
    T = promote_type(eltype(coords1), eltype(coords2), typeof(sigma), typeof(range_param))
    dist_matrix = pairwise(Euclidean(), coords1', coords2', dims=2)
    
    h = dist_matrix ./ range_param
    K = zeros(T, size(h))
    mask = h .< 1.0
    
    K[mask] = (sigma^2) .* (1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3)
    
    return K
end



"""
    evaluate_spherical_kernel_matrix(coords, sigma, range_param, noise)

A helper function to compute the spherical kernel matrix. This function is
intended to be available in the model's execution scope.
"""
function evaluate_spherical_kernel_matrix(
    coords::AbstractMatrix, sigma::Real, range_param::Real, noise::Real
)
    T = promote_type(eltype(coords), typeof(sigma), typeof(range_param), typeof(noise))
    dist_matrix = pairwise(Euclidean(), coords', dims=2)
    
    h = dist_matrix ./ range_param
    K = zeros(T, size(h))
    mask = h .< 1.0
    
    K[mask] = (sigma^2) .* (1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3)
    K += (noise * I)
    
    return K
end
