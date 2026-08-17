"""
    BSpline <: ComponentModel

A component model for a B-spline smoother, often referred to as a P-spline
(Penalized B-spline). This component creates a basis of B-spline functions of a
specified degree, centered at knots distributed across the covariate space. The
effect is a linear combination of these basis functions, with coefficients
regularized by a random walk prior to ensure smoothness.

# Version
v1.1.4 (2026-08-15)

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

# Computational Methods
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the RW2 penalty matrix. Recommended for gradient-based samplers.
- `:cholesky` (AD-friendly): Uses a dense Cholesky factorization of the penalty
  matrix. Can be less efficient for a large number of basis functions.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses a sparse Cholesky
  factorization. Not compatible with AD but retained as a didactic alternative.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `nbins`: `Int`, the number of basis functions. Default: `10`.
  - `degree`: `Int`, the polynomial degree of the B-spline. Default: `3`.
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the
    coefficients. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`).
    Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the B-spline coefficients.
- `innovations_<key>`: The raw standard normal innovations for the coefficients.
- `latent_<key>`: The final smooth effect vector.

# Key References
- Eilers, P. H., & Marx, B. D. (1996). *Flexible smoothing with B-splines and
  penalties*. Statistical Science, 11(2), 89-121.
- de Boor, C. (1978). *A Practical Guide to Splines*. Springer-Verlag.
"""
struct BSpline <: ComponentModel
    nbins::Int
    degree::Int
    sigma::Distribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:bspline] = BSpline

COMPONENT_CONSTRUCTORS[:bspline] = (p, params) -> BSpline(
    get(params, :nbins, 10),
    get(params, :degree, 3),
    p.sigma,
    get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:bspline] = :smooth

"""
    get_precomputes(m::BSpline, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the B-spline basis matrix and the RW2 penalty matrix (and its
spectral/Cholesky decompositions) for the spline coefficients.
"""
function get_precomputes(m::BSpline, M::NamedTuple, mod_data::Dict)::NamedTuple
    variables = mod_data[:variables]
    if isempty(variables)
        error("The BSpline model requires at least one coordinate variable.")
    end

    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for BSpline model not found in data.")
        end
    end

    coords = Matrix{Float64}(M.data[!, Symbol.(variables)])
    
    n_obs, n_dims = size(coords)
    
    if n_dims > 1
        @warn "BSpline component is best for 1D smooths. For multi-dimensional " *
              "smoothing, consider `model=:tps` or `model=:tensorproductsmooth`."
    end
    
    # Generate the B-spline basis matrix.
    B, actual_nbins = bstm_bspline_basis(coords[:, 1], m.nbins, m.degree)
    n_latent = actual_nbins

    # Build the RW2 penalty matrix template.
    template = build_structure_template(:rw2, n_latent)
    Q_template = template.matrix
    
    # Perform eigendecomposition for spectral method.
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values
    scaling_factor = _compute_scaling_factor(L, 2) # RW2 has rank deficiency of 2
    
    Q_template_scaled = Q_template ./ scaling_factor
    L_scaled = L ./ scaling_factor

    # Pre-compute dense Cholesky factor for the :cholesky method.
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
    get_priors(m::BSpline, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for the standard deviation `sigma` and the raw innovations for
the B-spline coefficients.
"""
function get_priors(
    m::BSpline, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    return """
    # Priors for BSpline component: $(spec.key)
    $(p_names.sigma) ~ $(sigma_prior_str)
    $(p_names.innovations) ~ MvNormal(zeros($(spec.hyper.n_latent)), I)
    """
end

"""
    get_updates(m::BSpline, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates Turing code to construct the B-spline smooth effect and add it to the
linear predictor `eta`, dispatching on the chosen computational method.
"""
function get_updates(
    m::BSpline, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent
    
    common_code = """
        hyper = spec_registry[:$(key)].hyper
        B_basis = hyper.basis_matrix
    """

    spectral_code = """
        # --- B-Spline Smoother Component (Spectral): $(key) ---
        let
            $(common_code)
            
            diag_D = $(p_names.sigma) ./ sqrt.(hyper.L .+ M.noise)
            # Enforce sum-to-zero constraints for RW2 penalty
            diag_D[1] = 0.0
            diag_D[2] = 0.0
            
            coeffs = hyper.U * (diag_D .* $(p_names.innovations))
            
            $(p_names.latent) = B_basis * coeffs
            
            $(eta_target) .+= $(p_names.latent)
        end
    """

    cholesky_code = """
        # --- B-Spline Smoother Component (Cholesky, AD-Safe): $(key) ---
        let
            $(common_code)
            
            F = hyper.cholesky_factor
            
            coeffs_raw = F.L' \\ $(p_names.innovations)
            
            # Apply soft sum-to-zero constraints for RW2 penalty
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(coeffs_raw[1:2])
            )
            
            coeffs = $(p_names.sigma) .* coeffs_raw
            $(p_names.latent) = B_basis * coeffs
            
            $(eta_target) .+= $(p_names.latent)
        end
    """

    cholesky_sparse_code = """
        # --- B-Spline Smoother Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        # WARNING: This method is for didactic purposes and is NOT compatible with
        # automatic differentiation (e.g., NUTS sampler).
        let
            $(common_code)
            
            Q_penalty = hyper.Q_template
            F = cholesky(Symmetric(Q_penalty + M.noise * I))
            
            L_sparse = sparse(F.L)
            coeffs_raw = L_sparse' \\ $(p_names.innovations)
            
            # Apply soft sum-to-zero constraints for RW2 penalty
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(coeffs_raw[1:2])
            )
            
            coeffs = $(p_names.sigma) .* coeffs_raw
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
        error("Unsupported method '$(m.method)' for BSpline component.")
    end
end

"""
    get_effects(m::BSpline, chain, spec, M, PS)

Reconstructs the posterior distribution of the B-spline smooth effect from the
MCMC chain.   handle GPU arrays by moving sampled
parameters to the device for computation and moving the final results back to the CPU.
"""
function get_effects(
    m::BSpline, chain::Chains, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device
    noise = M.noise

    # --- Basis Matrix Construction for Training and Prediction ---
    B_train = spec.hyper.basis_matrix # Already on device if use_gpu=true

    B_full = if !isnothing(PS)
        coord_vars = get(spec.params, :positional_args, [])
        if !isempty(coord_vars) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
            # Get prediction coordinates from CPU DataFrame
            coords_pred_cpu = Matrix{Float64}(PS.data[!, Symbol.(coord_vars)])
            # Move coordinates to device
            coords_pred_device = to_device(coords_pred_cpu)
            # bstm_bspline_basis is GPU-aware, so it will return a GPU matrix
            B_pred, _ = bstm_bspline_basis(coords_pred_device[:, 1], m.nbins, m.degree)
            vcat(B_train, B_pred)
        else
            B_train
        end
    else
        B_train
    end
    N_total = size(B_full, 1)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for BSpline component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples_cpu = get_params_matrix(chain, innovations_name, spec.hyper.n_latent)

        # Initialize the output matrix for coefficients on the target device
        coeffs_matrix_device = to_device(zeros(Float64, spec.hyper.n_latent, n_samples))

        hyper = spec.hyper

        # --- Sample-wise Coefficient Reconstruction on the Target Device ---
        for i in 1:n_samples
            innov_i_device = to_device(innovations_samples_cpu[i, :])
            sigma_i = sigma_samples_cpu[i] # CPU scalar

            local coeffs_device
            if m.method == :spectral
                U = hyper.U # Already on device
                L = hyper.L # Already on device
                diag_D = sigma_i ./ sqrt.(L .+ noise)
                diag_D[1] = 0.0 # Enforce sum-to-zero constraints for RW2 penalty
                diag_D[2] = 0.0
                coeffs_device = U * (diag_D .* innov_i_device)
            else # :cholesky or :cholesky_sparse
                F = hyper.cholesky_factor # Already on device
                coeffs_raw = F.L' \ innov_i_device
                # Apply sum-to-zero constraints for RW2 penalty
                coeffs_centered = coeffs_raw .- mean(view(coeffs_raw, 1:2)) # Centering based on first two elements
                coeffs_device = sigma_i .* coeffs_centered
            end
            coeffs_matrix_device[:, i] = coeffs_device
        end

        # Perform matrix multiplication on the device
        effect_k_device = B_full * coeffs_matrix_device
        # Move the final result for this outcome back to the CPU
        push!(structured_effects, Array(effect_k_device))
    end

    return (structured=structured_effects, noisy=structured_effects)
end
