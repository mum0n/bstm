# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    Warp <: ComponentModel

A component model for a non-stationary Gaussian Process using a warping function.
This component implements a simple Deep GP structure with one hidden layer. The input
coordinates are first "warped" by a non-linear function (approximated by RFFs), and
then a standard RFF-based GP is applied to these warped coordinates.

# Fields
- `lengthscale::Union{Distribution, Vector{<:Distribution}}`: The prior for the kernel lengthscale(s) of the main GP.
- `sigma::Distribution`: The prior for the standard deviation of the main GP's coefficients.
- `n_features::Int`: The number of random features to use for both the warping and main RFF layers.
- `kernel::String`: The name of the kernel to approximate (e.g., "se", "matern32").
"""
struct Warp <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    n_features::Int
    kernel::String
end

# Add to the central component constructor registry.
COMPONENT_CONSTRUCTORS[:warp] = (p, params) -> Warp(p.lengthscale, p.sigma, get(params, :n_features, 20), string(get(params, :kernel, "se")))

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[Warp] = :smooth

"""
    get_datastructures!(m_type::Type{<:Warp}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Warp` component.
It ensures that coordinate variables are provided and stores them in the module data.
"""
function get_datastructures!(m_type::Type{<:Warp}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error("The Warp model requires coordinate variables, e.g., `random(x, y, model=:warp)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Warp model not found in data.")
        end
    end

    # Store the coordinates matrix in the module's parameters for later use.
    mod_data[:params][:coords] = Matrix{Float64}(M[:data][!, Symbol.(variables)])

    return true
end

"""
    get_precomputes(m::Warp, M::NamedTuple, mod_data::Dict)::NamedTuple

For the `Warp` component, this function stores the coordinate matrix and its dimensions.
All RFF parameters are learned, so no fixed features are pre-generated.
"""
function get_precomputes(m::Warp, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("Warp component precomputes failed: coordinates not found in module data.")
    end
    
    in_dims = size(coords, 2)
    n_latent = size(coords, 1) # The effect is at the observation level

    return (
        coords=coords,
        in_dims=in_dims,
        n_latent=n_latent
    )
end

"""
    get_priors(m::Warp, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `Warp` component's priors.
This defines priors for `sigma`, `lengthscale`, and all RFF parameters for both
the warping and main GP layers.
"""
function get_priors(m::Warp, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    # Manually create names for the many parameters of this component
    W_warp_name = Symbol("$(p_names.latent)_W_warp")
    b_warp_name = Symbol("$(p_names.latent)_b_warp")
    beta_warp_name = Symbol("$(p_names.latent)_beta_warp")
    W_main_name = Symbol("$(p_names.latent)_W_main")
    b_main_name = Symbol("$(p_names.latent)_b_main")
    beta_main_raw_name = p_names.raw # Use the standard 'raw' for the main coefficients

    in_dims = spec.precomputes.in_dims
    n_features = m.n_features

    priors = String[]
    push!(priors, "$(p_names.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(p_names.sigma))")

    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors, "$(p_names.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(p_names.ls))")
    else
        ls_prior_str = _distribution_to_string(m.lengthscale)
        push!(priors, "$(p_names.ls) ~ NamedDist($(ls_prior_str), :$(p_names.ls))")
    end

    # Priors for the warping function's RFF parameters
    push!(priors, "$(W_warp_name) ~ NamedDist(MvNormal(zeros(T, $(in_dims * n_features)), I), :$(W_warp_name))")
    push!(priors, "$(b_warp_name) ~ NamedDist(MvNormal(zeros(T, $(n_features)), I), :$(b_warp_name))")
    push!(priors, "$(beta_warp_name) ~ NamedDist(MvNormal(zeros(T, $(n_features)), I), :$(beta_warp_name))")

    # Priors for the main GP's RFF parameters
    push!(priors, "$(W_main_name) ~ NamedDist(MvNormal(zeros(T, $(in_dims * n_features)), I), :$(W_main_name))")
    push!(priors, "$(b_main_name) ~ NamedDist(MvNormal(zeros(T, $(n_features)), I), :$(b_main_name))")
    push!(priors, "$(beta_main_raw_name) ~ NamedDist(MvNormal(zeros(T, $(n_features)), I), :$(beta_main_raw_name))")
    
    return join(priors, "\n    ")
end

"""
    get_updates(m::Warp, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `Warp` smooth effect.
"""
function get_updates(m::Warp, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    W_warp_name = Symbol("$(p_names.latent)_W_warp")
    b_warp_name = Symbol("$(p_names.latent)_b_warp")
    beta_warp_name = Symbol("$(p_names.latent)_beta_warp")
    W_main_name = Symbol("$(p_names.latent)_W_main")
    b_main_name = Symbol("$(p_names.latent)_b_main")
    beta_main_raw_name = p_names.raw

    precomputes = "spec_registry[:$(spec.key)].precomputes"
    in_dims = spec.precomputes.in_dims
    n_features = m.n_features

    return """
        # --- Warp (Deep GP) Component: $(spec.key) ---
        local coords = T.($precomputes.coords)
        
        # 1. Construct and apply the warping function
        local W_warp_matrix = reshape($(W_warp_name), $(in_dims), $(n_features))
        local Phi_warp = sqrt(T(2.0) / $(n_features)) .* cos.((coords * W_warp_matrix) .+ $(b_warp_name)')
        local warping_effect = Phi_warp * $(beta_warp_name)
        local coords_warped = coords .+ warping_effect

        # 2. Construct the main GP on the warped coordinates
        local W_main_matrix = reshape($(W_main_name), $(in_dims), $(n_features)) ./ $(p_names.ls)
        local Phi_main = sqrt(T(2.0) / $(n_features)) .* cos.((coords_warped * W_main_matrix) .+ $(b_main_name)')
        
        # 3. Scale coefficients and compute final effect
        local scaled_beta_main = $(beta_main_raw_name) .* $(p_names.sigma)
        local $(p_names.latent) = Phi_main * scaled_beta_main
        
        $(eta_target) .+= $(p_names.latent)
    """
end

"""
    get_effects(m::Warp, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `Warp` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::Warp, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    # Manually create names for the many parameters of this component
    W_warp_name = Symbol("$(p_names.latent)_W_warp")
    b_warp_name = Symbol("$(p_names.latent)_b_warp")
    beta_warp_name = Symbol("$(p_names.latent)_beta_warp")
    W_main_name = Symbol("$(p_names.latent)_W_main")
    b_main_name = Symbol("$(p_names.latent)_b_main")
    beta_main_raw_name = p_names.raw

    # Extract posterior samples
    sigma_samples = get(chain, p_names.sigma)
    ls_samples = get(chain, p_names.ls)
    W_warp_samples = get(chain, W_warp_name)
    b_warp_samples = get(chain, b_warp_name)
    beta_warp_samples = get(chain, beta_warp_name)
    W_main_samples = get(chain, W_main_name)
    b_main_samples = get(chain, b_main_name)
    beta_main_raw_samples = get(chain, beta_main_raw_name)

    precomputes = spec.precomputes
    in_dims = precomputes.in_dims
    n_features = m.n_features

    # Use prediction set coordinates if available, otherwise use training coordinates.
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(precomputes.coords, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        precomputes.coords
    end
    
    reconstructed_effects = zeros(n_samples, size(coords_full, 1))

    for i in 1:n_samples
        # Reconstruct warping function
        W_warp_i = reshape(W_warp_samples[i, :], in_dims, n_features)
        b_warp_i = b_warp_samples[i, :]
        beta_warp_i = beta_warp_samples[i, :]
        Phi_warp_i = sqrt(2.0 / n_features) .* cos.((coords_full * W_warp_i) .+ b_warp_i')
        warping_effect_i = Phi_warp_i * beta_warp_i
        coords_warped_i = coords_full .+ warping_effect_i

        # Reconstruct main GP
        W_main_i = reshape(W_main_samples[i, :], in_dims, n_features) ./ ls_samples[i]
        b_main_i = b_main_samples[i, :]
        beta_main_raw_i = beta_main_raw_samples[i, :]
        Phi_main_i = sqrt(2.0 / n_features) .* cos.((coords_warped_i * W_main_i) .+ b_main_i')
        
        scaled_beta_main_i = beta_main_raw_i .* sigma_samples[i]
        reconstructed_effects[i, :] = Phi_main_i * scaled_beta_main_i
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    return (structured=(mean=mean_effect, lower=lower_ci, upper=upper_ci),)
end
