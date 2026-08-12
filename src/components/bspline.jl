"""
    BSpline <: ComponentModel

A component model for a B-spline smoother, often referred to as a P-spline
(Penalized B-spline). This component creates a basis of B-spline functions of a
specified degree, centered at knots distributed across the covariate space. The
effect is a linear combination of these basis functions, with coefficients
regularized by a random walk prior to ensure smoothness.

# Version
v1.1.0 (2026-08-11)

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

    template = build_structure_template(:rw2, n_latent)
    Q_template = template.matrix
    
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values
    scaling_factor = _compute_scaling_factor(L, 2)
    
    Q_template_scaled = Q_template ./ scaling_factor
    L_scaled = L ./ scaling_factor

    # Pre-compute dense Cholesky factor for the :cholesky method
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
    m::BSpline, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    return """ # Priors for sigma and raw innovations
        $(p_names.sigma) ~ $(sigma_prior_str)
        $(p_names.innovations) ~ MvNormal(
            zeros(T, spec.hyper.n_latent), I
        )
    """
end

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
        let
            $(common_code)
            
            Q_penalty = hyper.Q_template
            F = cholesky(Symmetric(Q_penalty + M.noise * I))
            
            coeffs_raw = F.L' \\ $(p_names.innovations)
            
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

function get_effects(
    m::BSpline, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))

    for k in 1:outcomes_N
        sigma_name = _find_parameter(p_names_vec, string(spec.key), "sigma", k, is_multivariate_model)
        innovations_name = _find_parameter(p_names_vec, string(spec.key), "innovations", k, is_multivariate_model)

        if isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for BSpline component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innovations_name, spec.hyper.n_latent)

        hyper = spec.hyper
        noise = M.noise
        
        B_train = hyper.basis_matrix
        B_full = if !isnothing(PS)
            coord_vars = get(spec.params, :positional_args, [])
            if !isempty(coord_vars) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
                coords_pred = Matrix{Float64}(PS.data[!, Symbol.(coord_vars)])
                B_pred, _ = bstm_bspline_basis(coords_pred[:, 1], m.nbins, m.degree)
                vcat(B_train, B_pred)
            else
                B_train
            end
        else
            B_train
        end
        
        if size(B_full, 1) != N_total
            @warn "BSpline effect reconstruction: dimension mismatch. Using in-sample basis."
            B_full = B_train
        end

        reconstructed_effects_k = zeros(size(B_full, 1), n_samples)

        for i in 1:n_samples
            local coeffs
            if m.method == :spectral
                U = hyper.U
                L = hyper.L
                diag_D = sigma_samples[i] ./ sqrt.(L .+ noise)
                diag_D[1] = 0.0
                diag_D[2] = 0.0
                coeffs = U * (diag_D .* innovations_samples[i, :])
            else # :cholesky or :cholesky_sparse
                F = hyper.cholesky_factor
                coeffs_raw = F.L' \ innovations_samples[i, :]
                coeffs_centered = coeffs_raw .- mean(coeffs_raw[1:2])
                coeffs = sigma_samples[i] .* coeffs_centered
            end
            reconstructed_effects_k[:, i] = B_full * coeffs
        end
        push!(structured_effects, reconstructed_effects_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
