# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    TPS <: ComponentModel

A component model for a Thin Plate Spline (TPS) smoother. This component creates a basis
of radial basis functions centered at knots distributed across the covariate space. The
effect is a linear combination of these basis functions, with coefficients regularized by
a random walk prior to ensure smoothness.

# Fields
- `nbins::Int`: The number of knots (and basis functions) to use.
- `sigma::Distribution`: The prior for the standard deviation of the TPS coefficients.
"""
struct TPS <: ComponentModel
    nbins::Int
    sigma::Distribution
end

# Add to the central component constructor registry.
COMPONENT_CONSTRUCTORS[:tps] = (p, params) -> TPS(get(params, :nbins, 20), p.sigma)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[TPS] = :smooth

"""
    get_datastructures!(m_type::Type{<:TPS}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `TPS` component.
It ensures that coordinate variables are provided and stores them in the module data.
"""
function get_datastructures!(m_type::Type{<:TPS}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error("The TPS model requires coordinate variables, e.g., `random(x, y, model=:tps)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for TPS model not found in data.")
        end
    end

    # Store the coordinates matrix in the module's parameters for later use.
    mod_data[:params][:coords] = Matrix{Float64}(M[:data][!, Symbol.(variables)])

    return true
end

"""
    get_precomputes(m::TPS, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `TPS` component.
This involves creating the knot locations, the radial basis function matrix, and
the precision matrix template for the random walk penalty on the coefficients.
"""
function get_precomputes(m::TPS, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("TPS component precomputes failed: coordinates not found in module data.")
    end
    
    n_obs, n_dims = size(coords)
    n_latent = m.nbins

    # Generate knot locations using k-means for good spatial coverage.
    knot_method = get(mod_data[:params], :knot_method, :kmeans)
    knots = generate_inducing_points(coords, n_latent; method=knot_method)
    actual_n_knots = size(knots, 1)
    if actual_n_knots < n_latent
        @warn "TPS: Could only generate $(actual_n_knots) unique knots, requested $(n_latent). Using $(actual_n_knots)."
        n_latent = actual_n_knots
    end

    # --- Create basis matrix B ---
    # The radial basis function for TPS is r^2 * log(r) for 2D, and |r|^3 for 1D.
    B = zeros(Float64, n_obs, n_latent)
    if n_dims == 1
        for i in 1:n_latent
            r = abs.(coords[:, 1] .- knots[i, 1])
            B[:, i] .= r.^3
        end
    elseif n_dims == 2
        for i in 1:n_latent
            dist_sq = (coords[:, 1] .- knots[i, 1]).^2 .+ (coords[:, 2] .- knots[i, 2]).^2
            r = sqrt.(dist_sq)
            B[:, i] .= (r.^2) .* log.(r .+ 1e-9)
        end
    else # Higher dimensions
        for i in 1:n_latent
            dist_sq = sum((coords .- knots[i, :]').^2, dims=2)
            r = sqrt.(dist_sq)
            # General form for odd dimensions d is r^(2m-d), here m=2, so r^(4-d)
            # For even dimensions, it's r^(2m-d) * log(r)
            if isodd(n_dims)
                B[:, i] .= r.^(4 - n_dims)
            else
                B[:, i] .= (r.^(4 - n_dims)) .* log.(r .+ 1e-9)
            end
        end
    end
    
    # --- Create penalty matrix Q ---
    # The penalty for TPS coefficients is often modeled as an RW2 process.
    template = build_structure_template(:rw2, n_latent)
    Q_template = template.matrix
    
    # --- Spectral decomposition for AD-friendly sampling ---
    rank_deficiency = 2 # RW2
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values
    scaling_factor = _compute_scaling_factor(L, rank_deficiency)
    
    Q_template_scaled = Q_template ./ scaling_factor
    L_scaled = L ./ scaling_factor

    return (
        basis_matrix=B,
        Q_template=Q_template_scaled,
        scaling_factor=scaling_factor,
        U=U,
        L=L_scaled,
        n_latent=n_latent,
        knots=knots # Store knots for prediction
    )
end

"""
    get_priors(m::TPS, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `TPS` component's priors.
It defines the prior for `sigma` and the `raw` coefficients for the basis functions.
"""
function get_priors(m::TPS, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    return """
        $(p_names.sigma) ~ NamedDist($(sigma_prior_str), :$(p_names.sigma))
        $(p_names.raw) ~ NamedDist(MvNormal(zeros(T, spec_registry[:$(spec.key)].precomputes.n_latent), I), :$(p_names.raw))
    """
end

"""
    get_updates(m::TPS, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `TPS` smooth effect.
"""
function get_updates(m::TPS, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- Thin Plate Spline (TPS) Smoother Component: $(spec.key) ---
        local precomputes = spec_registry[:$(spec.key)].precomputes
        
        # Reconstruct latent coefficients using spectral decomposition of the penalty matrix
        local diag_D = $(p_names.sigma) ./ sqrt.(precomputes.L .+ M.noise)
        # Enforce sum-to-zero constraints for RW2 penalty
        diag_D[1] = zero(T)
        diag_D[2] = zero(T)
        
        local coeffs = precomputes.U * (diag_D .* $(p_names.raw))
        
        # Compute final effect by multiplying basis matrix with coefficients
        local $(p_names.latent) = T.(precomputes.basis_matrix) * coeffs
        
        $(eta_target) .+= $(p_names.latent)
    """
end

"""
    get_effects(m::TPS, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `TPS` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::TPS, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    sigma_samples = get(chain, p_names.sigma)
    raw_samples = get(chain, p_names.raw)

    precomputes = spec.precomputes
    U = precomputes.U
    L = precomputes.L
    noise = M.noise
    n_latent = precomputes.n_latent
    knots = precomputes.knots
    n_dims = size(knots, 2)

    # The `predict` function is responsible for creating the basis matrix for the prediction set.
    # Here, we assume it is available in PS.basis_matrices if a prediction set is provided.
    B_train = precomputes.basis_matrix
    B_full = if !isnothing(PS) && haskey(PS, :basis_matrices) && haskey(PS.basis_matrices, spec.key)
        vcat(B_train, PS.basis_matrices[spec.key])
    else
        B_train
    end
    
    if size(B_full, 1) != N_total
        @warn "TPS effect reconstruction: dimension mismatch. Using in-sample effects only."
        B_full = B_train
    end

    reconstructed_effects = zeros(n_samples, size(B_full, 1))

    for i in 1:n_samples
        current_sigma = sigma_samples[i]
        current_raw = raw_samples[i, :]
        
        diag_D = current_sigma ./ sqrt.(L .+ noise)
        diag_D[1] = 0.0
        diag_D[2] = 0.0
        
        coeffs = U * (diag_D .* current_raw)
        reconstructed_effects[i, :] = B_full * coeffs
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    return (structured=(mean=mean_effect, lower=lower_ci, upper=upper_ci),)
end
