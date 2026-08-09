"""
    BSpline <: ComponentModel

A component model for a B-spline smoother, often referred to as a P-spline
(Penalized B-spline). This component creates a basis of B-spline functions of a
specified degree, centered at knots distributed across the covariate space. The
effect is a linear combination of these basis functions, with coefficients
regularized by a random walk prior to ensure smoothness.

# Version
v1.0.0 (2026-08-08)

# Mathematical Summary
The component models a smooth function \$f(x)\$ as a linear combination of B-spline
basis functions:
\$f(x) = \\sum_{j=1}^{N_{bins}} \\beta_j B_{j,p}(x)\$
where \$B_{j,p}(x)\$ is the j-th B-spline basis function of degree \$p\$, and \$\\beta_j\$
are the coefficients.

To prevent overfitting and ensure smoothness, a penalty is applied to the
coefficients. This is equivalent to placing a Gaussian Markov Random Field (GMRF)
prior on the coefficients. A common choice is a second-order random walk (RW2)
penalty on the differences between adjacent coefficients:
\$\\Delta^2 \\beta_j = \\beta_j - 2\\beta_{j-1} + \\beta_{j-2}\$
This penalizes deviations from a linear trend, encouraging a smooth function.

# Assumptions
- The relationship between the covariate and the outcome is smooth.
- The number of bins (`nbins`) is large enough to capture the underlying trend,
  with smoothness being controlled by the penalty rather than the number of knots.

# Best Use Case
Modeling flexible, non-linear effects of continuous covariates in a computationally
efficient manner. It is a robust and widely used alternative to Gaussian Processes
for 1D smoothing.

# Key References
- Eilers, P. H., & Marx, B. D. (1996). Flexible smoothing with B-splines and
  penalties. *Statistical Science*, 11(2), 89-102.
- De Boor, C. (1978). *A Practical Guide to Splines*. Springer-Verlag.
- Wikipedia: B-spline

# Fields
- `nbins::Int`: The number of basis functions to generate.
- `degree::Int`: The polynomial degree of the B-spline (e.g., 1 for linear, 3 for
  cubic).
- `sigma::UnivariateDistribution`: The prior for the standard deviation of the
  B-spline coefficients.
"""
struct BSpline <: ComponentModel
    nbins::Int
    degree::Int
    sigma::Distribution
end

COMPONENT_TYPE_REGISTRY[:bspline] = BSpline

COMPONENT_CONSTRUCTORS[:bspline] = (p, params) -> BSpline(
    get(params, :nbins, 10), get(params, :degree, 3), p.sigma
)

MODEL_TO_STRUCTURE_MAP[:bspline] = :smooth

"""
    get_datastructures!(m_type::Type{<:BSpline}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `BSpline` component. It ensures that
coordinate variables are provided and stores them in the module data.

# Assumptions
- The `random()` call provides one or more variables representing the coordinates
  to be smoothed.
"""
function get_datastructures!(
    m_type::Type{<:BSpline}, M::Dict, mod_data::Dict
)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error(
            "The BSpline model requires at least one coordinate variable, e.g., " *
            "`random(x, model=:bspline)`."
        )
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for BSpline model not found in data.")
        end
    end

    mod_data[:params][:coords] = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    return true
end

"""
    get_precomputes(m::BSpline, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the B-spline basis matrix and the spectral decomposition of the RW2
penalty matrix for the coefficients.

# Assumptions
- The `bstm_bspline_basis` helper function is available in the execution scope.
"""
function get_precomputes(m::BSpline, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("BSpline precomputes failed: coordinates not found in module data.")
    end
    
    n_obs, n_dims = size(coords)
    
    if n_dims > 1
        @warn "BSpline component is best for 1D smooths. For multi-dimensional " *
              "smoothing, consider `model=:tps` or `model=:tensorproductsmooth`."
    end
    
    B, actual_nbins = bstm_bspline_basis(coords[:, 1], m.nbins, m.degree)
    n_latent = actual_nbins

    # A second-order random walk (RW2) is a common penalty for spline coefficients.
    template = build_structure_template(:rw2, n_latent)
    Q_template = template.matrix
    
    # Spectral decomposition for AD-friendly sampling
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values
    scaling_factor = _compute_scaling_factor(L, 2) # RW2 has rank deficiency 2
    
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
    get_priors(m::BSpline, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `sigma` and the `raw` innovations for the basis coefficients.
"""
function get_priors(
    m::BSpline, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    return """
        $(p_names.sigma) ~ $(sigma_prior_str)
        $(p_names.raw) ~ MvNormal(
            zeros(T, spec_registry[:$(spec.key)].hyper.n_latent), I
        )
    """
end

"""
    get_updates(m::BSpline, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code to construct the B-spline smooth effect using a spectral
decomposition of the penalty matrix.
"""
function get_updates(
    m::BSpline, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- B-Spline Smoother Component: $(spec.key) ---
        let
            local hyper = spec_registry[:$(spec.key)].hyper
            
            # Reconstruct latent coefficients using spectral decomposition
            local diag_D = $(p_names.sigma) ./ sqrt.(hyper.L .+ M.noise)
            # Enforce sum-to-zero constraints for RW2 penalty
            diag_D[1] = 0.0
            diag_D[2] = 0.0
            
            local coeffs = hyper.U * (diag_D .* $(p_names.raw))
            
            # Compute final effect by multiplying basis matrix with coefficients
            local $(p_names.latent) = hyper.basis_matrix * coeffs
            
            $(eta_target) .+= $(p_names.latent)
        end
    """
end

"""
    get_effects(m::BSpline, chain, M::NamedTuple, ...)::NamedTuple

Reconstructs the `BSpline` component's effect from the MCMC chain's posterior
samples.
"""
function get_effects(
    m::BSpline, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)
        raw_samples = get_params_vector(
            chain, string(p_names.raw), spec.hyper.n_latent
        )

        hyper = spec.hyper
        U = hyper.U
        L = hyper.L
        noise = M.noise
        
        B_train = hyper.basis_matrix
        B_full = if !isnothing(PS) && haskey(PS, :basis_matrices) &&
                    haskey(PS.basis_matrices, spec.key)
            vcat(B_train, PS.basis_matrices[spec.key])
        else
            B_train
        end
        
        if size(B_full, 1) != N_total
            @warn "BSpline effect reconstruction: dimension mismatch. Using in-sample."
            B_full = B_train
        end

        reconstructed_effects_k = zeros(size(B_full, 1), n_samples)

        for i in 1:n_samples
            diag_D = sigma_samples[i] ./ sqrt.(L .+ noise)
            diag_D[1] = 0.0
            diag_D[2] = 0.0
            
            coeffs = U * (diag_D .* raw_samples[i, :])
            reconstructed_effects_k[:, i] = B_full * coeffs
        end
        push!(structured_effects, reconstructed_effects_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
