"""
    PSpline <: ComponentModel

A component model for a P-spline (Penalized B-spline) smoother. This component
creates a basis of B-spline functions and applies a discrete penalty (typically a
random walk) to the coefficients to ensure smoothness and prevent overfitting.

# Version
v1.2.0 (2026-08-14)

# Mathematical Summary
The P-spline models a smooth function \$f(x)\$ as a linear combination of \$K\$ B-spline
basis functions \$B_k(x)\$:
\$f(x) = \\sum_{k=1}^{K} \\beta_k B_k(x)\$
To enforce smoothness, a penalty is applied to the coefficients \$\\boldsymbol{\\beta}\$.
This is achieved by assuming the coefficients follow a Gaussian Markov Random Field
(GMRF) structure. A common choice is a second-order random walk (RW2), which
penalizes deviations from a local linear trend:
\$\\Delta^d \\beta_k = \\sum_{j=0}^d (-1)^j \\binom{d}{j} \\beta_{k-j} \\sim \\mathcal{N}(0, \\sigma^{-2})\$
where \$d\$ is the `diff_order`. The precision matrix \$\\mathbf{Q}\$ for the coefficients
is derived from this random walk structure. The model then samples the coefficients from
\$\\boldsymbol{\\beta} \\sim \\mathcal{N}(\\mathbf{0}, (\\sigma^2 \\mathbf{Q})^{-1})\$.

# Computational Methods
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the penalty matrix. Recommended for gradient-based samplers.
- `:cholesky` (AD-friendly): Uses a pre-computed dense Cholesky factorization of the
  penalty matrix.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization,
  which is not compatible with most AD backends.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `nbins`: `Int`, the number of basis functions. Default: `20`.
  - `degree`: `Int`, the polynomial degree of the B-spline. Default: `3`.
  - `diff_order`: `Int`, the order of the random walk penalty (1 or 2). Default: `2`.
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the
    coefficients. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`).
    Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the B-spline coefficients.
- `innovations_<key>`: The raw standard normal innovations for the coefficients.
- `latent_<key>`: The final smooth effect vector.
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

function get_precomputes(m::PSpline, M::NamedTuple, mod_data::Dict)::NamedTuple
    variables = mod_data[:variables]

    if isempty(variables)
        error(
            "The PSpline model requires at least one coordinate variable, e.g., " *
            "`random(x, model=:pspline)`."
        )
    end

    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for PSpline model not found in data.")
        end
    end

    coords = Matrix{Float64}(M.data[!, Symbol.(variables)])

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

function get_priors(
    m::PSpline, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    sigma_prior_str = _distribution_to_string(m.sigma)
    key = spec.key

    return """
        $(p_names.sigma) ~ $(sigma_prior_str)
        $(p_names.innovations) ~ MvNormal(
            zeros(T, spec_registry[:$(key)].hyper.n_latent), I
        )
    """
end

function get_updates(
    m::PSpline, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

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

            coeffs = hyper.U * (diag_D .* $(p_names.innovations))
            $(p_names.latent) = B_basis * coeffs
            $(eta_target) .+= $(p_names.latent)
        end
    """

    cholesky_code = """
        # --- P-Spline Smoother Component (Cholesky, AD-Safe): $(key) ---
        let
            $(common_code)
            local F = hyper.cholesky_factor
            local coeffs_raw = F.L' \\ $(p_names.innovations)

            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * hyper.n_latent), sum(coeffs_raw)
            )

            local coeffs = $(p_names.sigma) .* coeffs_raw
            $(p_names.latent) = B_basis * coeffs
            $(eta_target) .+= $(p_names.latent)
        end
    """

    cholesky_sparse_code = """
        # --- P-Spline Smoother Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            $(common_code)
            local Q_penalty = hyper.Q_template
            local F = cholesky(Symmetric(Q_penalty + M.noise * I))
            local coeffs_raw = F.L' \\ $(p_names.innovations)

            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * hyper.n_latent), sum(coeffs_raw)
            )

            local coeffs = $(p_names.sigma) .* coeffs_raw
            $(p_names.latent) = B_basis * coeffs
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

function get_effects(
    m::PSpline, chain, M::NamedTuple, n_samples::Int, is_multivariate_model::Bool,
    outcomes_N::Int, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
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
    p_names_vec = string.(FlexiChains.parameters(chain))

    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)

        sigma_name = _find_parameter(p_names_vec, v.sigma, k, is_multivariate_model)
        innovations_name = _find_parameter(
            p_names_vec, v.innovations, k, is_multivariate_model
        )

        if isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for PSpline component $(spec.key) (outcome $k) not found. " *
                  "Returning zero-matrix."
            push!(structured_effects, zeros(Float64, size(B_full, 1), n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innovations_name, hyper.n_latent)

        effect_k = zeros(Float64, size(B_full, 1), n_samples)

        for i in 1:n_samples
            local coeffs
            if m.method == :spectral
                U, L = hyper.U, hyper.L
                diag_D = sigma_samples[i] ./ sqrt.(L .+ noise)
                for j in 1:m.diff_order; diag_D[j] = 0.0; end
                coeffs = U * (diag_D .* innovations_samples[i, :])
            else # :cholesky or :cholesky_sparse
                F = hyper.cholesky_factor
                coeffs_raw = F.L' \ innovations_samples[i, :]
                coeffs_centered = coeffs_raw .- mean(coeffs_raw)
                coeffs = sigma_samples[i] .* coeffs_centered
            end
            effect_k[:, i] = B_full * coeffs
        end
        push!(structured_effects, effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
