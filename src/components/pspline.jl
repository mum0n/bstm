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
    m::PSpline, chain::Chains, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device
    
    hyper = spec.hyper
    noise = M.noise
    n_latent = hyper.n_latent

    # --- Basis Matrix Handling for Training and Prediction Sets ---
    B_train = hyper.basis_matrix # This is already on the correct device

    coord_vars = get(spec.params, :positional_args, [])
    coord_var_sym = isempty(coord_vars) ? :none : Symbol(coord_vars[1])

    B_full = if !isnothing(PS) && hasproperty(PS.data, coord_var_sym)
        coords_pred_cpu = PS.data[!, coord_var_sym]
        # Move prediction coordinates to the device before creating the basis
        coords_pred_device = to_device(coords_pred_cpu)
        
        # bstm_bspline_basis is device-aware
        B_pred, _ = bstm_bspline_basis(coords_pred_device, m.nbins, m.degree)
        vcat(B_train, B_pred)
    else
        B_train
    end
    N_total = size(B_full, 1)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        
        # Find parameter names in the MCMC chain
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for PSpline component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples_cpu = get_params_matrix(chain, innovations_name, n_latent)

        # Initialize the output matrix for coefficients on the target device
        coeffs_samples_matrix_device = to_device(zeros(Float64, n_latent, n_samples))

        # --- Sample-wise Reconstruction on the Target Device ---
        if m.method == :spectral
            U = hyper.U # Already on device
            L = hyper.L # Already on device
            
            # Vectorized reconstruction
            sigma_samples_device = to_device(sigma_samples_cpu)
            innovations_device_T = to_device(innovations_samples_cpu') # Transpose to [n_latent x n_samples]
            
            diag_D_matrix = sigma_samples_device' ./ sqrt.(L .+ noise)
            for i in 1:m.diff_order
                diag_D_matrix[i, :] .= 0.0
            end
            
            coeffs_samples_matrix_device = U * (diag_D_matrix .* innovations_device_T)

        else # :cholesky or :cholesky_sparse
            F = hyper.cholesky_factor # Already on device
            for i in 1:n_samples
                innov_i_device = to_device(innovations_samples_cpu[i, :])
                coeffs_raw = F.L' \ innov_i_device
                coeffs_centered = coeffs_raw .- mean(coeffs_raw)
                coeffs_samples_matrix_device[:, i] = sigma_samples_cpu[i] .* coeffs_centered
            end
        end
        
        # Perform final matrix multiplication on the device
        effect_k_device = B_full * coeffs_samples_matrix_device
        
        # Move the final result for this outcome back to the CPU
        push!(structured_effects, Array(effect_k_device))
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
