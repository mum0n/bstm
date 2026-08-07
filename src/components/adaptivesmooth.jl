# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    AdaptiveSmooth <: ComponentModel

A component model for an adaptive smoother. This component creates basis functions
dynamically using a small, single-hidden-layer neural network (MLP). The effect is a
linear combination of these learned basis functions, with coefficients regularized by
a simple prior. This allows the model to learn complex, non-linear relationships
without pre-specifying the basis functions.

# Fields
- `hidden_dim::Int`: The number of neurons in the hidden layer of the MLP.
- `nbins::Int`: The number of basis functions to generate (output dimension of the MLP).
- `sigma::Distribution`: The prior for the standard deviation of the basis function coefficients.
"""
struct AdaptiveSmooth <: ComponentModel
    hidden_dim::Int
    nbins::Int
    sigma::Distribution
end

# Add to the central component constructor registry.
COMPONENT_CONSTRUCTORS[:adaptivesmooth] = (p, params) -> AdaptiveSmooth(get(params, :hidden_dim, 10), get(params, :nbins, 20), p.sigma)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[AdaptiveSmooth] = :smooth

"""
    get_datastructures!(m_type::Type{<:AdaptiveSmooth}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `AdaptiveSmooth` component.
It ensures that coordinate variables are provided and stores them in the module data.
"""
function get_datastructures!(m_type::Type{<:AdaptiveSmooth}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error("The AdaptiveSmooth model requires coordinate variables, e.g., `random(x, y, model=:adaptivesmooth)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for AdaptiveSmooth model not found in data.")
        end
    end

    # Store the coordinates matrix in the module's parameters for later use.
    mod_data[:params][:coords] = Matrix{Float64}(M[:data][!, Symbol.(variables)])

    return true
end

"""
    get_precomputes(m::AdaptiveSmooth, M::NamedTuple, mod_data::Dict)::NamedTuple

For the `AdaptiveSmooth` component, this function stores the coordinate matrix and
the dimensions of the MLP. The basis matrix itself is not pre-computed as it depends
on learned parameters.
"""
function get_precomputes(m::AdaptiveSmooth, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("AdaptiveSmooth component precomputes failed: coordinates not found in module data.")
    end
    
    in_dims = size(coords, 2)
    n_latent = m.nbins # The number of latent variables is the number of basis functions.

    return (
        coords=coords,
        in_dims=in_dims,
        n_latent=n_latent
    )
end

"""
    get_priors(m::AdaptiveSmooth, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `AdaptiveSmooth` component's priors.
This defines priors for the MLP weights (`W1`, `b1`, `W2`), the basis coefficients (`raw`),
and the overall standard deviation (`sigma`).
"""
function get_priors(m::AdaptiveSmooth, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    in_dims = spec.precomputes.in_dims
    hidden_dim = m.hidden_dim
    n_bins = m.nbins

    return """
        # Priors for AdaptiveSmooth component: $(spec.key)
        $(p_names.W1) ~ NamedDist(MvNormal(zeros(T, $(in_dims * hidden_dim)), I), :$(p_names.W1))
        $(p_names.b1) ~ NamedDist(MvNormal(zeros(T, $(hidden_dim)), I), :$(p_names.b1))
        $(p_names.W2) ~ NamedDist(MvNormal(zeros(T, $(hidden_dim * n_bins)), I), :$(p_names.W2))
        $(p_names.raw) ~ NamedDist(MvNormal(zeros(T, $(n_bins)), I), :$(p_names.raw))
        $(p_names.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(p_names.sigma))
    """
end

"""
    get_updates(m::AdaptiveSmooth, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `AdaptiveSmooth` effect.
"""
function get_updates(m::AdaptiveSmooth, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    precomputes = "spec_registry[:$(spec.key)].precomputes"
    in_dims = spec.precomputes.in_dims
    hidden_dim = m.hidden_dim
    n_bins = m.nbins

    return """
        # --- AdaptiveSmooth Component: $(spec.key) ---
        local X_coords = T.($precomputes.coords)
        
        # 1. Reshape MLP weights and construct the adaptive basis matrix
        local W1_mat = reshape($(p_names.W1), $(in_dims), $(hidden_dim))
        local W2_mat = reshape($(p_names.W2), $(hidden_dim), $(n_bins))
        local H = tanh.((X_coords * W1_mat) .+ $(p_names.b1)')
        local B_adaptive = H * W2_mat
        
        # 2. Scale the raw coefficients and compute the final effect
        local scaled_coeffs = $(p_names.raw) .* $(p_names.sigma)
        local $(p_names.latent) = B_adaptive * scaled_coeffs
        
        $(eta_target) .+= $(p_names.latent)
    """
end

"""
    get_effects(m::AdaptiveSmooth, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `AdaptiveSmooth` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::AdaptiveSmooth, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    # Extract posterior samples
    W1_samples = get(chain, p_names.W1)
    b1_samples = get(chain, p_names.b1)
    W2_samples = get(chain, p_names.W2)
    raw_samples = get(chain, p_names.raw)
    sigma_samples = get(chain, p_names.sigma)

    precomputes = spec.precomputes
    in_dims = precomputes.in_dims
    hidden_dim = m.hidden_dim
    n_bins = m.nbins

    # Use prediction set coordinates if available, otherwise use training coordinates.
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(precomputes.coords, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        precomputes.coords
    end
    
    reconstructed_effects = zeros(n_samples, size(coords_full, 1))

    for i in 1:n_samples
        # Reconstruct MLP weights for this sample
        W1_i = reshape(W1_samples[i, :], in_dims, hidden_dim)
        b1_i = b1_samples[i, :]
        W2_i = reshape(W2_samples[i, :], hidden_dim, n_bins)
        
        # Reconstruct basis coefficients for this sample
        scaled_coeffs_i = raw_samples[i, :] .* sigma_samples[i]
        
        # Compute the adaptive basis matrix B for the full coordinate set
        H_i = tanh.((coords_full * W1_i) .+ b1_i')
        B_adaptive_i = H_i * W2_i
        
        # Compute the final effect
        reconstructed_effects[i, :] = B_adaptive_i * scaled_coeffs_i
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    return (structured=(mean=mean_effect, lower=lower_ci, upper=upper_ci),)
end
