# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    RFF <: ComponentModel

A component model for a Random Fourier Features (RFF) smoother. This component
approximates a stationary kernel (like Squared Exponential or Matérn) by projecting
the input coordinates into a randomized feature space. This transforms the GP into a
more scalable Bayesian linear regression problem.

# Fields
- `lengthscale::Union{Distribution, Vector{<:Distribution}}`: The prior for the lengthscale(s) of the kernel.
- `sigma::Distribution`: The prior for the standard deviation of the RFF coefficients.
- `n_features::Int`: The number of random features to use for the approximation.
- `kernel::String`: The name of the kernel to approximate (e.g., "se", "matern32").
"""
struct RFF <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    n_features::Int
    kernel::String
end

# Add to the central component constructor registry.
COMPONENT_CONSTRUCTORS[:rff] = (p, params) -> RFF(p.lengthscale, p.sigma, get(params, :n_features, 20), string(get(params, :kernel, "se")))

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[RFF] = :smooth

"""
    get_datastructures!(m_type::Type{<:RFF}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `RFF` component.
It ensures that coordinate variables are provided and stores them in the module data.
"""
function get_datastructures!(m_type::Type{<:RFF}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error("The RFF model requires coordinate variables, e.g., `random(x, y, model=:rff)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for RFF model not found in data.")
        end
    end

    # Store the coordinates matrix in the module's parameters for later use.
    mod_data[:params][:coords] = Matrix{Float64}(M[:data][!, Symbol.(variables)])

    return true
end

"""
    _generate_rff_fixed_params(in_dims::Int, n_features::Int, lengthscale::Union{Real, AbstractVector}, kernel_name::String)

Generates fixed random projection weights (W) and biases (b) for RFF approximation.
This helper function is used in the precomputation stage.
"""
function _generate_rff_fixed_params(in_dims::Int, n_features::Int, lengthscale::Union{Real, AbstractVector}, kernel_name::String)
    b = rand(Uniform(0, 2 * pi), n_features)
    W = Matrix{Float64}(undef, in_dims, n_features)
    k_name = lowercase(kernel_name)

    if k_name in ["se", "gaussian", "rbf"]
        if lengthscale isa Real
            W .= rand(Normal(0, 1.0 / lengthscale), in_dims, n_features)
        else
            if length(lengthscale) != in_dims; error("ARD lengthscale vector length mismatch."); end
            for d in 1:in_dims; W[d, :] = rand(Normal(0, 1.0 / lengthscale[d]), n_features); end
        end
    elseif occursin("matern", k_name)
        nu = if k_name == "matern12"; 0.5; elseif k_name == "matern32"; 1.5; else 2.5; end
        df = 2 * nu
        if lengthscale isa Real
            W .= (sqrt(df) / lengthscale) .* rand(TDist(df), in_dims, n_features)
        else
            if length(lengthscale) != in_dims; error("ARD lengthscale vector length mismatch."); end
            for d in 1:in_dims; W[d, :] = (sqrt(df) / lengthscale[d]) .* rand(TDist(df), n_features); end
        end
    else
        @warn "Kernel '$kernel_name' not recognized for RFF. Defaulting to SE."
        return _generate_rff_fixed_params(in_dims, n_features, lengthscale, "se")
    end
    return W, b
end

"""
    get_precomputes(m::RFF, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `RFF` component.
It generates the fixed random features (`W_fixed`, `b_fixed`) that serve as the
means for the priors on the adaptive feature parameters.
"""
function get_precomputes(m::RFF, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("RFF component precomputes failed: coordinates not found in module data.")
    end
    
    in_dims = size(coords, 2)
    
    # Get initial lengthscale from the prior's mean for generating fixed features.
    ls_prior = m.lengthscale
    local ls_initial
    if ls_prior isa Vector
        ls_initial = [mean(p isa Truncated ? untruncated(p) : p) for p in ls_prior]
    else
        ls_initial = mean(ls_prior isa Truncated ? untruncated(ls_prior) : ls_prior)
    end

    W_fixed, b_fixed = _generate_rff_fixed_params(in_dims, m.n_features, ls_initial, m.kernel)

    return (
        coords=coords,
        W_fixed=W_fixed,
        b_fixed=b_fixed,
        n_latent=m.n_features,
        in_dims=in_dims
    )
end

"""
    get_priors(m::RFF, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `RFF` component's priors.
This defines priors for `sigma`, `lengthscale`, the adaptive feature parameters `W` and `b`,
and the `raw` coefficients.
"""
function get_priors(m::RFF, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = String[]
    push!(priors, "$(p_names.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(p_names.sigma))")

    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors, "$(p_names.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(p_names.ls))")
    else
        ls_prior_str = _distribution_to_string(m.lengthscale)
        push!(priors, "$(p_names.ls) ~ NamedDist($(ls_prior_str), :$(p_names.ls))")
    end
    
    # Priors for adaptive features, centered on the fixed random features.
    push!(priors, "$(p_names.W) ~ NamedDist(MvNormal(vec(spec_registry[:$(spec.key)].precomputes.W_fixed), T(0.1)), :$(p_names.W))")
    push!(priors, "$(p_names.b) ~ NamedDist(MvNormal(spec_registry[:$(spec.key)].precomputes.b_fixed, T(0.1)), :$(p_names.b))")
    
    # Prior for the raw coefficients.
    push!(priors, "$(p_names.raw) ~ NamedDist(MvNormal(zeros(T, spec.precomputes.n_latent), I), :$(p_names.raw))")

    return join(priors, "\n    ")
end

"""
    get_updates(m::RFF, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `RFF` smooth effect.
"""
function get_updates(m::RFF, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    precomputes = "spec_registry[:$(spec.key)].precomputes"
    
    return """
        # --- RFF Smoother Component: $(spec.key) ---
        local X_coords = T.($precomputes.coords)
        local W_matrix = reshape($(p_names.W), $precomputes.in_dims, $precomputes.n_latent)
        
        # Compute the feature matrix Phi
        local Phi = sqrt(T(2.0) / $precomputes.n_latent) .* cos.((X_coords * W_matrix) .+ $(p_names.b)')
        
        # Scale the raw coefficients and compute the final effect
        local scaled_coeffs = $(p_names.raw) .* $(p_names.sigma)
        local $(p_names.latent) = Phi * scaled_coeffs
        
        $(eta_target) .+= $(p_names.latent)
    """
end

"""
    get_effects(m::RFF, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `RFF` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::RFF, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    # Extract posterior samples
    sigma_samples = get(chain, p_names.sigma)
    raw_samples = get(chain, p_names.raw)
    W_samples = get(chain, p_names.W)
    b_samples = get(chain, p_names.b)

    precomputes = spec.precomputes
    in_dims = precomputes.in_dims
    n_features = precomputes.n_latent

    # Use prediction set coordinates if available, otherwise use training coordinates.
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(precomputes.coords, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        precomputes.coords
    end
    
    reconstructed_effects = zeros(n_samples, size(coords_full, 1))

    for i in 1:n_samples
        current_sigma = sigma_samples[i]
        current_raw = raw_samples[i, :]
        current_W = reshape(W_samples[i, :], in_dims, n_features)
        current_b = b_samples[i, :]
        
        # Reconstruct the feature matrix Phi for the current sample
        Phi = sqrt(2.0 / n_features) .* cos.((coords_full * current_W) .+ current_b')
        
        # Reconstruct the scaled coefficients and the final effect
        scaled_coeffs = current_raw .* current_sigma
        reconstructed_effects[i, :] = Phi * scaled_coeffs
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    return (structured=(mean=mean_effect, lower=lower_ci, upper=upper_ci),)
end
