# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    Nystrom <: ComponentModel

A component model for the Nyström sparse Gaussian Process approximation. This method
approximates the full GP kernel matrix with a low-rank version based on a small set
of `n_inducing` points, making it scalable for larger datasets.

# Fields
- `lengthscale::Union{Distribution, Vector{<:Distribution}}`: The prior for the kernel lengthscale(s).
- `sigma::Distribution`: The prior for the marginal standard deviation of the GP.
- `n_inducing::Int`: The number of inducing points to use for the approximation.
- `kernel::String`: The name of the kernel function (e.g., "se", "matern32").
"""
struct Nystrom <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    n_inducing::Int
    kernel::String
end

# Add to the central component constructor registry.
COMPONENT_CONSTRUCTORS[:nystrom] = (p, params) -> Nystrom(p.lengthscale, p.sigma, get(params, :n_inducing, 20), string(get(params, :kernel, "se")))

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[Nystrom] = :smooth

"""
    get_datastructures!(m_type::Type{<:Nystrom}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Nystrom` component. It ensures coordinate
variables are provided, stores them, and generates the inducing point locations.
"""
function get_datastructures!(m_type::Type{<:Nystrom}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]
    params = mod_data[:params]

    if isempty(variables)
        error("The Nystrom model requires coordinate variables, e.g., `random(x, y, model=:nystrom)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Nystrom model not found in data.")
        end
    end

    # Store the coordinates matrix in the module's parameters for later use.
    coords = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    mod_data[:params][:coords] = coords

    # Generate and store inducing points.
    n_inducing = get(params, :n_inducing, 20)
    knot_method = get(params, :knot_method, :kmeans)
    Z_inducing = generate_inducing_points(coords, n_inducing; method=knot_method)
    mod_data[:params][:Z_inducing] = Z_inducing

    return true
end

"""
    get_precomputes(m::Nystrom, M::NamedTuple, mod_data::Dict)::NamedTuple

For the `Nystrom` component, this function stores the coordinate matrix and the
inducing point locations. The number of latent variables is `n_inducing`.
"""
function get_precomputes(m::Nystrom, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("Nystrom component precomputes failed: coordinates not found.")
    end
    
    Z_inducing = get(mod_data[:params], :Z_inducing, nothing)
    if isnothing(Z_inducing)
        error("Nystrom component precomputes failed: inducing points not found.")
    end

    return (
        coords=coords,
        Z_inducing=Z_inducing,
        n_latent=m.n_inducing # The latent variable `u` is of size n_inducing
    )
end

"""
    get_priors(m::Nystrom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `Nystrom` component's priors. This includes
priors for `sigma`, `lengthscale`, and the `raw` innovations for the inducing points.
"""
function get_priors(m::Nystrom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
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
    
    # Raw innovations for inducing points (u_raw)
    push!(priors, "$(p_names.raw) ~ NamedDist(MvNormal(zeros(T, $(m.n_inducing)), I), :$(p_names.raw))")

    return join(priors, "\n    ")
end

"""
    get_updates(m::Nystrom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `Nystrom` sparse GP effect.
"""
function get_updates(m::Nystrom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- Nystrom Sparse GP Component: $(spec.key) ---
        local precomputes = spec_registry[:$(spec.key)].precomputes
        local X_coords = T.(precomputes.coords)
        local Z_coords = T.(precomputes.Z_inducing)
        local kernel_type = Symbol("$(m.kernel)")
        
        # 1. Compute kernel matrices
        local K_UU = evaluate_kernel_matrix(Z_coords, $(p_names.sigma), $(p_names.ls), kernel_type, M.noise)
        local K_XU = evaluate_cross_kernel_matrix(X_coords, Z_coords, $(p_names.sigma), $(p_names.ls), kernel_type)
        
        # 2. Cholesky decomposition of K_UU
        local L_UU = cholesky(Symmetric(K_UU)).L
        
        # 3. Project standard normal noise through the Nystrom approximation
        # f(X) ≈ K_XU * inv(K_UU) * u, where u ~ N(0, K_UU)
        # Using non-centered parameterization: u = L_UU * raw, where raw ~ N(0, I)
        # f(X) ≈ K_XU * inv(L_UU * L_UU') * L_UU * raw = K_XU * (L_UU' \\ raw)
        local $(p_names.latent) = K_XU * (L_UU' \\ $(p_names.raw))
        
        $(eta_target) .+= $(p_names.latent)
    """
end

"""
    get_effects(m::Nystrom, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `Nystrom` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::Nystrom, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    # Extract posterior samples
    sigma_samples = get(chain, p_names.sigma)
    ls_samples = get(chain, p_names.ls)
    raw_samples = get(chain, p_names.raw) # u_raw

    # Prepare coordinates for training and prediction
    coords_train = spec.precomputes.coords
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        coords_train
    end
    n_obs_full = size(coords_full, 1)

    Z_inducing = spec.precomputes.Z_inducing
    kernel_type = Symbol(m.kernel)
    noise = M.noise

    reconstructed_effects = zeros(n_samples, n_obs_full)

    for i in 1:n_samples
        current_sigma = sigma_samples[i]
        current_ls = if m.lengthscale isa Vector; ls_samples[i, :]; else ls_samples[i]; end
        current_raw = raw_samples[i, :]

        # Reconstruct kernel matrices for the current sample
        K_UU = evaluate_kernel_matrix(Z_inducing, current_sigma, current_ls, kernel_type, noise)
        K_XU = evaluate_cross_kernel_matrix(coords_full, Z_inducing, current_sigma, current_ls, kernel_type)
        
        L_UU = cholesky(Symmetric(K_UU)).L
        
        # Reconstruct latent field
        reconstructed_effects[i, :] = K_XU * (L_UU' \ current_raw)
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    return (structured=(mean=mean_effect, lower=lower_ci, upper=upper_ci),)
end
