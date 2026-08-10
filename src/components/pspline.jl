"""
    PSpline <: ComponentModel

A component model for a P-spline (Penalized B-spline) smoother. This component
creates a basis of B-spline functions and applies a discrete penalty (typically a
random walk) to the coefficients to ensure smoothness and prevent overfitting.

# Version
v1.0.1 (2026-08-10)

# Mathematical Summary
The P-spline models a smooth function \$f(x)\$ as a linear combination of \$K\$ B-spline
basis functions \$B_k(x)\$:
\$f(x) = \\sum_{k=1}^{K} \\beta_k B_k(x)\$
To enforce smoothness, a penalty is applied to the coefficients \$\\boldsymbol{\\beta}\$.
This is achieved by assuming the coefficients follow a Gaussian Markov Random Field
(GMRF) structure. A common choice is a second-order random walk (RW2), which
penalizes deviations from a local linear trend:
\$\\Delta^2 \\beta_k = \\beta_k - 2\\beta_{j-1} + \\beta_{j-2} \\sim \\mathcal{N}(0, \\sigma^{-2})\$
The precision matrix \$\\mathbf{Q}\$ for the coefficients is derived from this random
walk structure. The model then samples the coefficients from
\$\\boldsymbol{\\beta} \\sim \\mathcal{N}(\\mathbf{0}, (\\sigma^2 \\mathbf{Q})^{-1})\$.

# Computational Methods
- `:spectral` (default): An efficient, AD-safe method using spectral decomposition.
- `:cholesky`: An AD-safe didactic alternative using dense Cholesky factorization.
- `:cholesky_sparse`: A non-AD-safe didactic method using sparse Cholesky
  factorization, suitable for gradient-free samplers.

# Fields
- `nbins::Int`: The number of basis functions to generate.
- `degree::Int`: The polynomial degree of the B-spline (e.g., 1 for linear, 3 for cubic).
- `diff_order::Int`: The order of the random walk penalty on the coefficients.
- `sigma::Distribution`: The prior for the standard deviation of the coefficients.
- `method::Symbol`: The computational method for regularizing coefficients.
"""
struct PSpline <: ComponentModel
    nbins::Int
    degree::Int
    diff_order::Int
    sigma::Distribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:pspline] = PSpline

COMPONENT_CONSTRUCTORS[:pspline] = (p, params) -> PSpline(
    get(params, :nbins, 20),
    get(params, :degree, 3),
    get(params, :diff_order, 2),
    p.sigma,
    get(params, :method, :spectral)
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

Pre-computes the B-spline basis matrix, the penalty matrix, and its spectral
decomposition and dense Cholesky factorization.
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

    F = cholesky(Symmetric(Matrix(Q_template_scaled) + M.noise * I))

    return (
        basis_matrix=B,
        Q_template=Q_template_scaled,
        scaling_factor=scaling_factor,
        U=U,
        L=L_scaled,
        n_latent=n_latent,
        cholesky_factor=F
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

Generates the Turing code to construct the P-spline smooth effect, dispatching
on the chosen method.
"""
function get_updates(
    m::PSpline, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent
    
    common_code = """
        local hyper = spec_registry[:$(key)].hyper
        local B_basis = hyper.basis_matrix
    """

    spectral_code = """
        # --- P-Spline Smoother Component (Spectral): $(key) ---
        let
            $(common_code)
            local diag_D = $(p_names.sigma) ./ sqrt.(hyper.L .+ M.noise)
            for i in 1:$(m.diff_order); diag_D[i] = 0.0; end
            
            local coeffs = hyper.U * (diag_D .* $(p_names.raw))
            local $(p_names.latent) = B_basis * coeffs
            $(eta_target) .+= $(p_names.latent)
        end
    """

    cholesky_code = """
        # --- P-Spline Smoother Component (Cholesky, AD-Safe): $(key) ---
        let
            $(common_code)
            local F = hyper.cholesky_factor
            local coeffs_raw = F.L' \\ $(p_names.raw)
            
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_latent)), sum(coeffs_raw))
            
            local coeffs = $(p_names.sigma) .* coeffs_raw
            local $(p_names.latent) = B_basis * coeffs
            $(eta_target) .+= $(p_names.latent)
        end
    """

    cholesky_sparse_code = """
        # --- P-Spline Smoother Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            $(common_code)
            local Q_penalty = hyper.Q_template
            local F = cholesky(Symmetric(Q_penalty + M.noise * I))
            local coeffs_raw = F.L' \\ $(p_names.raw)
            
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_latent)), sum(coeffs_raw))
            
            local coeffs = $(p_names.sigma) .* coeffs_raw
            local $(p_names.latent) = B_basis * coeffs
            $(eta_target) .+= $(p_names.latent)
        end
    """

    if m.method == :spectral
        return spectral_code
    elseif m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    else
        error("Unsupported method '$(m.method)' for PSpline component.")
    end
end

"""
    get_effects(m::PSpline, chain, M::NamedTuple, ...)

Reconstructs the `PSpline` component's effect from posterior samples, dispatching
on the method used during sampling.
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

    hyper = spec.hyper
    noise = M.noise

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        raw_samples = get_params_vector(chain, string(p_names.raw), hyper.n_latent)

        effect_k = zeros(Float64, size(B_full, 1), n_samples)

        for i in 1:n_samples
            local coeffs
            if m.method == :spectral
                U, L = hyper.U, hyper.L
                diag_D = sigma_samples[i] ./ sqrt.(L .+ noise)
                for j in 1:m.diff_order; diag_D[j] = 0.0; end
                coeffs = U * (diag_D .* raw_samples[i, :])
            else # :cholesky or :cholesky_sparse
                F = hyper.cholesky_factor
                coeffs_raw = F.L' \ raw_samples[i, :]
                coeffs_centered = coeffs_raw .- mean(coeffs_raw)
                coeffs = sigma_samples[i] .* coeffs_centered
            end
            effect_k[:, i] = B_full * coeffs
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
