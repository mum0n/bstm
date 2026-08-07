# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    SVGP <: ComponentModel

A component model for the Sparse Variational Gaussian Process (SVGP). This method
approximates a full GP by using a small set of `n_inducing` points to summarize
the data, making it scalable for larger datasets. It uses a non-centered
parameterization for efficient sampling.

# Fields
- `lengthscale::Union{Distribution, Vector{<:Distribution}}`: The prior for the kernel lengthscale(s).
- `sigma::Distribution`: The prior for the marginal standard deviation of the GP.
- `n_inducing::Int`: The number of inducing points to use for the approximation.
- `kernel::String`: The name of the kernel function (e.g., "se", "matern32").
"""
struct SVGP <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    n_inducing::Int
    kernel::String
end

# Add to the central component constructor registry.
COMPONENT_CONSTRUCTORS[:svgp] = (p, params) -> SVGP(p.lengthscale, p.sigma, get(params, :n_inducing, 20), string(get(params, :kernel, "se")))

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[SVGP] = :smooth

"""
    get_datastructures!(m_type::Type{<:SVGP}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `SVGP` component. It ensures coordinate
variables are provided, stores them, and generates the inducing point locations.
"""
function get_datastructures!(m_type::Type{<:SVGP}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]
    params = mod_data[:params]

    if isempty(variables)
        error("The SVGP model requires coordinate variables, e.g., `random(x, y, model=:svgp)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for SVGP model not found in data.")
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
    get_precomputes(m::SVGP, M::NamedTuple, mod_data::Dict)::NamedTuple

For the `SVGP` component, this function stores the coordinate matrix and the
inducing point locations for use by the code generator.
"""
function get_precomputes(m::SVGP, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("SVGP component precomputes failed: coordinates not found.")
    end
    
    Z_inducing = get(mod_data[:params], :Z_inducing, nothing)
    if isnothing(Z_inducing)
        error("SVGP component precomputes failed: inducing points not found.")
    end

    return (
        coords=coords,
        Z_inducing=Z_inducing,
        n_latent=size(coords, 1) # Number of data points
    )
end

"""
    get_priors(m::SVGP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `SVGP` component's priors. This includes
priors for `sigma`, `lengthscale`, and the raw innovations for both the inducing
points (`u_raw`) and the diagonal correction (`f_raw`).
"""
function get_priors(m::SVGP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
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
    
    # Priors for the latent values at inducing points (u_raw) and the final field innovations (f_raw)
    push!(priors, "$(p_names.raw) ~ NamedDist(MvNormal(zeros(T, $(m.n_inducing)), I), :$(p_names.raw))") # u_raw
    push!(priors, "$(p_names.innov) ~ NamedDist(MvNormal(zeros(T, spec.precomputes.n_latent), I), :$(p_names.innov))") # f_raw

    return join(priors, "\n    ")
end

"""
    get_updates(m::SVGP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `SVGP` sparse GP effect.
"""
function get_updates(m::SVGP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- SVGP Sparse GP Component: $(spec.key) ---
        local precomputes = spec_registry[:$(spec.key)].precomputes
        local X_coords = T.(precomputes.coords)
        local Z_coords = T.(precomputes.Z_inducing)
        local kernel_type = Symbol("$(m.kernel)")
        
        # 1. Compute kernel matrices
        local K_UU = evaluate_kernel_matrix(Z_coords, $(p_names.sigma), $(p_names.ls), kernel_type, M.noise)
        local K_XU = evaluate_cross_kernel_matrix(X_coords, Z_coords, $(p_names.sigma), $(p_names.ls), kernel_type)
        
        # 2. Sample latent values at inducing points (non-centered)
        local L_UU = cholesky(Symmetric(K_UU)).L
        local u_latent = L_UU * $(p_names.raw) # p_names.raw corresponds to u_raw
        
        # 3. Compute conditional mean and variance for SVGP (similar to FITC)
        #    μ_f = K_XU * inv(K_UU) * u_latent
        #    diag_cov_f = diag(K_XX - K_XU * inv(K_UU) * K_XU')
        
        local K_UU_inv_u = K_UU \\ u_latent
        local mean_f = K_XU * K_UU_inv_u
        
        # Compute diagonal of K_XX - Q_ff efficiently
        # diag(K_XX) is sigma^2 for stationary kernels.
        local diag_K_XX = fill($(p_names.sigma)^2, precomputes.n_latent)
        
        # diag(K_XU * inv(K_UU) * K_XU') = sum((L_UU' \\ K_XU').^2, dims=2)
        local tmp = (L_UU' \\ K_XU')'
        local diag_Q_ff = sum(tmp.^2, dims=2)
        
        local lambda_diag = diag_K_XX - vec(diag_Q_ff)
        
        # 4. Sample final latent field (non-centered)
        $(p_names.latent) = mean_f + sqrt.(max.(lambda_diag, T(0.0)) .+ M.noise) .* $(p_names.innov) # p_names.innov corresponds to f_raw
        
        $(eta_target) .+= $(p_names.latent)
    """
end

"""
    get_effects(m::SVGP, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `SVGP` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::SVGP, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    # Extract posterior samples
    sigma_samples = get(chain, p_names.sigma)
    ls_samples = get(chain, p_names.ls)
    u_raw_samples = get(chain, p_names.raw) # u_raw
    f_innov_samples = get(chain, p_names.innov) # f_raw

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
        current_ls = if m.lengthscale isa Vector
            ls_samples[i, :] # For ARD kernels, ls_samples will be a row vector
        else
            ls_samples[i]
        end
        current_u_raw = u_raw_samples[i, :]
        current_f_innov = f_innov_samples[i, 1:n_obs_full]

        # Reconstruct kernel matrices for the current sample
        K_UU = evaluate_kernel_matrix(Z_inducing, current_sigma, current_ls, kernel_type, noise)
        K_XU = evaluate_cross_kernel_matrix(coords_full, Z_inducing, current_sigma, current_ls, kernel_type)
        
        L_UU = cholesky(Symmetric(K_UU)).L
        u_latent = L_UU * current_u_raw
        
        K_UU_inv_u = K_UU \ u_latent
        mean_f = K_XU * K_UU_inv_u
        
        diag_K_XX = fill(current_sigma^2, n_obs_full)
        tmp = (L_UU' \ K_XU')'
        diag_Q_ff = sum(tmp.^2, dims=2)
        lambda_diag = diag_K_XX - vec(diag_Q_ff)
        
        reconstructed_effects[i, :] = mean_f + sqrt.(max.(lambda_diag, 0.0) .+ noise) .* current_f_innov
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    return (structured=(mean=indexed_mean, lower=indexed_lower, upper=indexed_upper),)
end
