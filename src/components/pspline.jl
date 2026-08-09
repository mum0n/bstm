
"""
    PSpline <: ComponentModel

A component model for a P-spline (Penalized B-spline) smoother. This component
creates a basis of B-spline functions and applies a discrete penalty (typically a
random walk) to the coefficients to ensure smoothness and prevent overfitting.

# Version
v1.0.0 (2026-08-08)

# Mathematical Summary
The P-spline models a smooth function \$f(x)\$ as a linear combination of \$K\$ B-spline
basis functions \$B_k(x)\$:
\$f(x) = \\sum_{k=1}^{K} \\beta_k B_k(x)\$
To enforce smoothness, a penalty is applied to the coefficients \$\\boldsymbol{\\beta}\$.
This is achieved by assuming the coefficients follow a Gaussian Markov Random Field
(GMRF) structure. A common choice is a second-order random walk (RW2), which
penalizes deviations from a local linear trend:
\$\\Delta^2 \\beta_k = \\beta_k - 2\\beta_{k-1} + \\beta_{k-2} \\sim \\mathcal{N}(0, \\sigma^{-2})\$
The precision matrix \$\\mathbf{Q}\$ for the coefficients is derived from this random
walk structure. The model then samples the coefficients from
\$\\boldsymbol{\\beta} \\sim \\mathcal{N}(\\mathbf{0}, (\\sigma^2 \\mathbf{Q})^{-1})\$.

# Assumptions
- The underlying function to be smoothed is continuous and smooth.
- The number of basis functions (`nbins`) is large enough to capture the function's
  curvature, as smoothness is enforced by the penalty, not the basis itself.

# Best Use Case
Flexible non-linear smoothing of continuous covariates. It is a powerful and widely
used alternative to Gaussian Processes for 1D smoothing, often with better
computational performance for large datasets.

# Key References
- Eilers, P. H., & Marx, B. D. (1996). Flexible smoothing with B-splines and
  penalties. *Statistical Science*, 11(2), 89-102.
- Wikipedia: P-spline

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

COMPONENT_TYPE_REGISTRY[:pspline] = PSpline

COMPONENT_CONSTRUCTORS[:pspline] = (p, params) -> PSpline(
    get(params, :nbins, 20),
    get(params, :degree, 3),
    get(params, :diff_order, 2),
    p.sigma
)

MODEL_TO_STRUCTURE_MAP[:pspline] = :smooth

"""
    get_datastructures!(m_type::Type{<:PSpline}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `PSpline` component.
It ensures that coordinate variables are provided and stores them in the module data.
"""
function get_datastructures!(m_type::Type{<:PSpline}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error(
            "The PSpline model requires at least one coordinate variable, e.g., " *
            "`random(x, model=:pspline)`."
        )
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
        error("PSpline precomputes failed: coordinates not found in module data.")
    end
    
    if size(coords, 2) > 1
        @warn "PSpline is designed for 1D smooths. For multi-dimensional smoothing, " *
              "consider `tps` or creating tensor products manually."
    end
    
    B, actual_nbins = bstm_bspline_basis(coords[:, 1], m.nbins, m.degree)
    n_latent = actual_nbins

    penalty_type = m.diff_order == 1 ? :rw1 : :rw2
    template = build_structure_template(penalty_type, n_latent)
    Q_template = template.matrix
    
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
    get_priors(m::PSpline, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code string for the `PSpline` component's priors.
It defines the prior for `sigma` and the `raw` coefficients for the basis functions.
"""
function get_priors(
    m::PSpline, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    sigma_prior_str = _distribution_to_string(m.sigma)
    n_latent = spec.hyper.n_latent
    
    return """
        $(p_names.sigma) ~ $(sigma_prior_str)
        $(p_names.raw) ~ MvNormal(zeros($(n_latent)), I)
    """
end

"""
    get_updates(m::PSpline, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code string for constructing the `PSpline` smooth effect.
"""
function get_updates(
    m::PSpline, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- P-Spline Smoother Component: $(spec.key) ---
        let
            local precomputes = spec_registry[:$(spec.key)].precomputes
            
            # Reconstruct latent coefficients using spectral decomposition of the penalty matrix
            local diag_D = $(p_names.sigma) ./ sqrt.(precomputes.L .+ M.noise)
            # Enforce sum-to-zero constraint(s)
            for i in 1:$(m.diff_order); diag_D[i] = 0.0; end
            
            local coeffs = precomputes.U * (diag_D .* $(p_names.raw))
            
            # Compute final effect by multiplying basis matrix with coefficients
            local $(p_names.latent) = precomputes.basis_matrix * coeffs
            
            $(eta_target) .+= $(p_names.latent)
        end
    """
end

"""
    get_effects(m::PSpline, chain, M::NamedTuple, ...)

Reconstructs the `PSpline` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(
    m::PSpline, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    coord_vars = get(spec.params, :positional_args, [])
    if isempty(coord_vars)
        error("PSpline effect reconstruction failed: coordinate variable not found.")
    end
    coord_var_sym = Symbol(coord_vars[1])

    B_train = spec.hyper.basis_matrix
    
    B_full = if !isnothing(PS) && hasproperty(PS.data, coord_var_sym)
        coords_pred = PS.data[!, coord_var_sym]
        B_pred, _ = bstm_bspline_basis(coords_pred, m.nbins, m.degree)
        vcat(B_train, B_pred)
    else
        B_train
    end

    if size(B_full, 1) != N_total
        @warn "PSpline effect reconstruction: dimension mismatch. Using in-sample effects only."
        B_full = B_train
    end

    precomputes = spec.hyper
    U = precomputes.U
    L = precomputes.L
    noise = M.noise

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        raw_samples = get_params_vector(
            chain, string(p_names.raw), precomputes.n_latent
        )

        effect_k = zeros(Float64, size(B_full, 1), n_samples)

        for i in 1:n_samples
            current_sigma = sigma_samples[i]
            current_raw = raw_samples[i, :]
            
            diag_D = current_sigma ./ sqrt.(L .+ noise)
            for j in 1:m.diff_order; diag_D[j] = 0.0; end
            
            coeffs = U * (diag_D .* current_raw)
            effect_k[:, i] = B_full * coeffs
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
