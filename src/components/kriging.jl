# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    Kriging <: ComponentModel

A component model for a full Gaussian Process, often referred to as Kriging in geostatistics.
It computes a dense covariance matrix based on a specified kernel function and coordinate inputs.

# Fields
- `lengthscale::Union{Distribution, Vector{<:Distribution}}`: The prior distribution(s) for the lengthscale(s) of the kernel. Can be a single distribution for isotropic kernels or a vector for ARD kernels.
- `sigma::Distribution`: The prior distribution for the marginal standard deviation (amplitude) of the GP.
- `kernel::String`: The name of the kernel function to use (e.g., "se" for Squared Exponential, "matern32").
"""
struct Kriging <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    kernel::String
end

# Add to the central component constructor registry.
# This constructor allows specifying the kernel type.
COMPONENT_CONSTRUCTORS[:kriging] = (p, params) -> Kriging(p.lengthscale, p.sigma, string(get(params, :kernel, "se")))

# Add to the model-to-structure map.
# Kriging is a continuous-space model, typically used as a smoother.
MODEL_TO_STRUCTURE_MAP[Kriging] = :smooth

"""
    get_datastructures!(m_type::Type{<:Kriging}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Kriging` component.
It ensures that coordinate variables are provided and stores them in the module data.
"""
function get_datastructures!(m_type::Type{<:Kriging}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error("The Kriging model requires coordinate variables, e.g., `random(x, y, model=:kriging)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Kriging model not found in data.")
        end
    end

    # Store the coordinates matrix in the module's parameters for later use.
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
        error("Kriging component precomputes failed: coordinates not found in module data.")
    end
    
    n_latent = size(coords, 1)

    # For a full GP like Kriging, the "template" is the coordinate matrix itself.
    # There is no parameter-independent precision matrix.
    return (coords=coords, n_latent=n_latent)
end

"""
    get_priors(m::Kriging, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `Kriging` component's priors.
It defines the priors for `sigma`, `lengthscale`, and the latent field `raw`.
"""
function get_priors(m::Kriging, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    priors = String[]
    push!(priors, "$(p_names.sigma) ~ NamedDist($(sigma_prior_str), :$(p_names.sigma))")

    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors, "$(p_names.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(p_names.ls))")
    else
        ls_prior_str = _distribution_to_string(m.lengthscale)
        push!(priors, "$(p_names.ls) ~ NamedDist($(ls_prior_str), :$(p_names.ls))")
    end
    
    # Latent field prior (non-centered parameterization)
    # raw ~ MvNormal(zeros(T, n_latent), I)
    push!(priors, "$(p_names.raw) ~ NamedDist(MvNormal(zeros(T, spec_registry[:$(spec.key)].precomputes.n_latent), I), :$(p_names.raw))")

    return join(priors, "\n    ")
end

"""
    get_updates(m::Kriging, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `Kriging` effect.
It computes the kernel matrix, performs a Cholesky decomposition, and
transforms the raw innovations to generate the latent field.
"""
function get_updates(m::Kriging, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- Kriging (Full GP) Component: $(spec.key) ---
        local X_coords = T.(spec_registry[:$(spec.key)].precomputes.coords)
        local kernel_type = Symbol("$(m.kernel)")

        # Compute the kernel matrix K_XX
        local K_mat = evaluate_kernel_matrix(X_coords, $(p_names.sigma), $(p_names.ls), kernel_type, M.noise)
        
        # Perform Cholesky decomposition for non-centered parameterization
        local F_krig = cholesky(Symmetric(K_mat))
        
        # Sample latent field: latent = L * raw
        local $(p_names.latent) = F_krig.L * $(p_names.raw)
        
        $(eta_target) .+= $(p_names.latent)
    """
end

"""
    get_effects(m::Kriging, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `Kriging` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::Kriging, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    sigma_samples = get(chain, p_names.sigma)
    ls_samples = get(chain, p_names.ls)
    raw_samples = get(chain, p_names.raw)

    # Use prediction set coordinates if available, otherwise use training coordinates.
    coords_full = if isnothing(PS)
        spec.precomputes.coords
    else
        # This assumes the component struct has a field `variables` storing the coordinate column names
        # which is not standard. A better approach would be to get it from `spec.var` or `spec.params`.
        # For now, we assume `spec.var` holds a string like "s_x_s_y"
        coord_vars = Symbol.(split(spec.var, "_"))
        Matrix{Float64}(PS.data[!, coord_vars])
    end
    
    n_obs_full = size(coords_full, 1)
    noise = M.noise
    kernel_type = Symbol(m.kernel)

    reconstructed_effects = zeros(n_samples, n_obs_full)

    for i in 1:n_samples
        current_sigma = sigma_samples[i]
        current_ls = if m.lengthscale isa Vector
            ls_samples[i, :] # For ARD kernels, ls_samples will be a row vector
        else
            ls_samples[i]
        end
        current_raw = raw_samples[i, :]
        
        # Reconstruct the kernel matrix for the current sample
        K_mat = evaluate_kernel_matrix(coords_full, current_sigma, current_ls, kernel_type, noise)
        
        # Perform Cholesky decomposition
        F = cholesky(Symmetric(K_mat))
        
        # Reconstruct latent field
        reconstructed_effects[i, :] = F.L * current_raw
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    # For full GP models like Kriging, the effect is already at the observation level.
    return (structured=(mean=mean_effect, lower=lower_ci, upper=upper_ci),)
end
