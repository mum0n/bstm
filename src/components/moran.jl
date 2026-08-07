# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    Moran <: ComponentModel

A component model for Moran's I Eigenvector Maps. This component decomposes spatial
autocorrelation into a set of orthogonal spatial patterns (eigenvectors) derived
from the Moran operator `(I - 11'/n)W(I - 11'/n)`. The effect is a linear combination
of these eigenvectors, providing a spectral basis for modeling spatial processes.

# Fields
- `sigma::Distribution`: The prior distribution for the standard deviation of the coefficients
  of the Moran eigenvectors.
"""
struct Moran <: ComponentModel
    sigma::Distribution
end

# Add to the central component constructor registry.
COMPONENT_CONSTRUCTORS[:moran] = (p, params) -> Moran(p.sigma)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[Moran] = :spatial

"""
    get_datastructures!(m_type::Type{<:Moran}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Moran` component.
It ensures that an adjacency matrix `W` is provided and sets up the spatial context
(`s_idx`, `s_N`) in the main model configuration `M`.
"""
function get_datastructures!(m_type::Type{<:Moran}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    variables = mod_data[:variables]

    # Ensure W is available, either directly in params or in M
    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr || w_val isa Symbol
            calling_mod = get(M, :calling_module, Main)
            try
                M[:W] = Core.eval(calling_mod, w_val)
            catch e
                error("Could not evaluate `W` argument `$(w_val)` for Moran component. Error: $e")
            end
        else
            M[:W] = w_val
        end
    end

    if !haskey(M, :W)
        error("Moran model requires an adjacency matrix `W` to be provided.")
    end

    if !isa(M[:W], AbstractMatrix) || isempty(M[:W])
        error("Provided `W` for Moran model is not a valid non-empty matrix.")
    end

    M[:s_N] = size(M[:W], 1)

    if isempty(variables)
        # If no variable is provided, assume s_idx is 1:s_N
        M[:s_idx] = collect(1:M[:s_N])
        @warn "Spatial index variable not provided for Moran component. Assuming `s_idx = 1:s_N`."
    else
        s_var_sym = Symbol(variables[1])
        if !hasproperty(M[:data], s_var_sym)
            error("Spatial index variable ':$s_var_sym' for Moran component not found in data.")
        end
        M[:s_idx] = M[:data][!, s_var_sym]
    end

    return true
end

"""
    get_precomputes(m::Moran, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `Moran` component.
It computes the Moran operator `M = (I - 11'/n)W(I - 11'/n)` and calculates its
eigenvectors, which serve as the spatial basis functions.
"""
function get_precomputes(m::Moran, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.s_N
    W = M.W

    # Create the centering matrix H = I - (1/n) * 1*1'
    H = I - (1/n) * ones(n, n)
    
    # Compute the Moran operator M = HWH
    W_mat = Matrix(W)
    moran_operator = H * W_mat * H
    
    # Compute the eigenvectors of the symmetric Moran operator
    eig_result = eigen(Symmetric(moran_operator))
    moran_eigenvectors = eig_result.vectors
    
    n_latent = size(moran_eigenvectors, 2)

    return (moran_eigenvectors=moran_eigenvectors, n_latent=n_latent)
end

"""
    get_priors(m::Moran, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `Moran` component's priors.
It defines the prior for `sigma` and the coefficients (`raw`) for the eigenvectors.
"""
function get_priors(m::Moran, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    # The 'raw' parameters are the coefficients for the eigenvectors.
    # raw ~ MvNormal(zeros(T, n_latent), I)
    
    return """
        $(p_names.sigma) ~ NamedDist($(sigma_prior_str), :$(p_names.sigma))
        $(p_names.raw) ~ NamedDist(MvNormal(zeros(T, spec_registry[:$(spec.key)].precomputes.n_latent), I), :$(p_names.raw))
    """
end

"""
    get_updates(m::Moran, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `Moran` component's effect.
The effect is a linear combination of the pre-computed Moran eigenvectors.
"""
function get_updates(m::Moran, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- Moran Eigenvector Component: $(spec.key) ---
        local moran_eigenvectors = T.(spec_registry[:$(spec.key)].precomputes.moran_eigenvectors)
        
        # The latent effect is a linear combination of the eigenvectors,
        # with coefficients ('raw') scaled by sigma.
        local $(p_names.latent) = moran_eigenvectors * ($(p_names.raw) .* $(p_names.sigma))
        
        $(eta_target) .+= $(p_names.latent)[M.s_idx]
    """
end

"""
    get_effects(m::Moran, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `Moran` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::Moran, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    sigma_samples = get(chain, p_names.sigma)
    raw_samples = get(chain, p_names.raw) # These are the eigenvector coefficients

    moran_eigenvectors = spec.precomputes.moran_eigenvectors
    n_latent = spec.precomputes.n_latent

    # Determine indices for reconstruction (training or prediction)
    idx_to_use = isnothing(PS) ? M.s_idx : PS.s_idx
    
    reconstructed_effects = zeros(n_samples, n_latent)

    for i in 1:n_samples
        current_sigma = sigma_samples[i]
        current_coeffs = raw_samples[i, :]
        
        # Reconstruct latent field: latent = eigenvectors * (coeffs * sigma)
        reconstructed_effects[i, :] = moran_eigenvectors * (current_coeffs .* current_sigma)
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    indexed_mean = mean_effect[idx_to_use]
    indexed_lower = lower_ci[idx_to_use]
    indexed_upper = upper_ci[idx_to_use]

    return (structured=(mean=indexed_mean, lower=indexed_lower, upper=indexed_upper),)
end
