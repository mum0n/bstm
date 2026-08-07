# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    PSpline <: ComponentModel

A component model for a P-spline (Penalized B-spline) smoother. This component
creates a basis of B-spline functions and applies a discrete penalty (typically a
random walk) to the coefficients to ensure smoothness and prevent overfitting.

# Fields
- `nbins::Int`: The number of basis functions to generate.
- `degree::Int`: The polynomial degree of the B-spline (e.g., 1 for linear, 3 for cubic).
- `diff_order::Int`: The order of the random walk penalty on the coefficients (1 for RW1, 2 for RW2).
- `sigma::Distribution`: The prior for the standard deviation of the spline coefficients, which controls the smoothness.
"""
struct PSpline <: ComponentModel
    nbins::Int
    degree::Int
    diff_order::Int
    sigma::Distribution
end

# Add to the central component constructor registry.
COMPONENT_CONSTRUCTORS[:pspline] = (p, params) -> PSpline(get(params, :nbins, 20), get(params, :degree, 3), get(params, :diff_order, 2), p.sigma)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[PSpline] = :smooth

"""
    get_datastructures!(m_type::Type{<:PSpline}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `PSpline` component.
It ensures that coordinate variables are provided and stores them in the module data.
"""
function get_datastructures!(m_type::Type{<:PSpline}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error("The PSpline model requires at least one coordinate variable, e.g., `random(x, model=:pspline)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for PSpline model not found in data.")
        end
    end

    # Store the coordinates matrix in the module's parameters for later use.
    mod_data[:params][:coords] = Matrix{Float64}(M[:data][!, Symbol.(variables)])

    return true
end

"""
    get_precomputes(m::PSpline, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `PSpline` component.
This involves creating the B-spline basis matrix and the precision matrix template
for the random walk penalty on the coefficients.
"""
function get_precomputes(m::PSpline, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("PSpline component precomputes failed: coordinates not found in module data.")
    end
    
    if size(coords, 2) > 1
        @warn "PSpline is designed for 1D smooths. For multi-dimensional smoothing, consider `tps` or creating tensor products manually."
    end
    
    # bstm_bspline_basis is a helper function that should be available.
    # It returns the basis matrix and the actual number of basis functions.
    B, actual_nbins = bstm_bspline_basis(coords[:, 1], m.nbins, m.degree)
    n_latent = actual_nbins

    # Create penalty matrix Q based on the difference order.
    penalty_type = m.diff_order == 1 ? :rw1 : :rw2
    template = build_structure_template(penalty_type, n_latent)
    Q_template = template.matrix
    
    # Spectral decomposition for AD-friendly sampling of coefficients.
    rank_deficiency = m.diff_order
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
        n_latent=n_latent
    )
end

"""
    get_priors(m::PSpline, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `PSpline` component's priors.
It defines the prior for `sigma` and the `raw` coefficients for the basis functions.
"""
function get_priors(m::PSpline, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    return """
        $(p_names.sigma) ~ NamedDist($(sigma_prior_str), :$(p_names.sigma))
        $(p_names.raw) ~ NamedDist(MvNormal(zeros(T, spec_registry[:$(spec.key)].precomputes.n_latent), I), :$(p_names.raw))
    """
end

"""
    get_updates(m::PSpline, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `PSpline` smooth effect.
"""
function get_updates(m::PSpline, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- P-Spline Smoother Component: $(spec.key) ---
        local precomputes = spec_registry[:$(spec.key)].precomputes
        
        # Reconstruct latent coefficients using spectral decomposition of the penalty matrix
        local diag_D = $(p_names.sigma) ./ sqrt.(precomputes.L .+ M.noise)
        # Enforce sum-to-zero constraint(s)
        for i in 1:$(m.diff_order); diag_D[i] = zero(T); end
        
        local coeffs = precomputes.U * (diag_D .* $(p_names.raw))
        
        # Compute final effect by multiplying basis matrix with coefficients
        local $(p_names.latent) = T.(precomputes.basis_matrix) * coeffs
        
        $(eta_target) .+= $(p_names.latent)
    """
end

"""
    get_effects(m::PSpline, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `PSpline` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::PSpline, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    sigma_samples = get(chain, p_names.sigma)
    raw_samples = get(chain, p_names.raw)

    precomputes = spec.precomputes
    U = precomputes.U
    L = precomputes.L
    noise = M.noise
    
    # The `predict` function is responsible for creating the basis matrix for the prediction set.
    # Here, we assume it is available in PS.basis_matrices if a prediction set is provided.
    B_train = precomputes.basis_matrix
    B_full = if !isnothing(PS) && haskey(PS, :basis_matrices) && haskey(PS.basis_matrices, spec.key)
        vcat(B_train, PS.basis_matrices[spec.key])
    else
        B_train
    end
    
    if size(B_full, 1) != N_total
        @warn "PSpline effect reconstruction: dimension mismatch. Using in-sample effects only."
        B_full = B_train
    end

    reconstructed_effects = zeros(n_samples, size(B_full, 1))

    for i in 1:n_samples
        current_sigma = sigma_samples[i]
        current_raw = raw_samples[i, :]
        
        diag_D = current_sigma ./ sqrt.(L .+ noise)
        for j in 1:m.diff_order; diag_D[j] = 0.0; end
        
        coeffs = U * (diag_D .* current_raw)
        reconstructed_effects[i, :] = B_full * coeffs
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    return (structured=(mean=mean_effect, lower=lower_ci, upper=upper_ci),)
end
