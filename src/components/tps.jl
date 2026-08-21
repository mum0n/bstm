"""
    TPS <: ComponentModel

A component model for a Thin Plate Spline (TPS) smoother. This component creates a
basis of radial basis functions centered at knots distributed across the covariate
space. The effect is a linear combination of these basis functions, with coefficients
regularized by a random walk prior to ensure smoothness.

# Version
v1.0.0

# Mathematical Summary
A Thin Plate Spline models a function \$f(\\mathbf{x})\$ as a linear combination of
radial basis functions \$\\phi\$, centered at a set of \$M\$ knots \$\\mathbf{c}_k\$:

\$f(\\mathbf{x}) = \\sum_{k=1}^{M} \\beta_k \\phi(\\|\\mathbf{x} - \\mathbf{c}_k\\|)\$

The radial basis function \$\\phi(r)\$ depends on the dimensionality \$d\$ of the input
space \$\\mathbf{x}\$:
- For \$d=1\$, \$\\phi(r) = r^3\$.
- For \$d=2\$, \$\\phi(r) = r^2 \\log(r)\$.
- For odd \$d > 2\$, \$\\phi(r) = r^{2m-d}\$ (with \$m=2\$, this is \$r^{4-d}\$).
- For even \$d > 2\$, \$\\phi(r) = r^{2m-d} \\log(r)\$.

The coefficients \$\\boldsymbol{\\beta}\$ are given a smoothing prior, typically a
second-order random walk (RW2) prior:
\$\\boldsymbol{\\beta} \\sim \\mathcal{N}(\\mathbf{0}, (\\tau \\mathbf{Q}_{RW2})^{-1})\$

# Computational Methods
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the penalty matrix. Recommended for gradient-based samplers.
- `:cholesky` (AD-friendly): Uses a pre-computed dense Cholesky factorization of the
  penalty matrix.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization,
  which is not compatible with most AD backends.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`, `y`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `nbins`: `Int`, the number of knots (and basis functions) to use. Default: `20`.
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the TPS
    coefficients. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`).
    Default: `:spectral`.
  - `knot_method`: `Symbol`, method for placing knots (`:kmeans`, `:random`, `:quantile`,
    `:range`). Default: `:kmeans`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the TPS coefficients.
- `innovations_<key>`: The raw standard normal innovations for the coefficients.
- `latent_<key>`: The final smooth effect vector.
"""
struct TPS <: ComponentModel
    nbins::Int
    sigma::Distribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:tps] = TPS

COMPONENT_CONSTRUCTORS[:tps] = (p, params) -> TPS(
    get(params, :nbins, 20),
    p.sigma,
    get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:tps] = :smooth

function get_precomputes(m::TPS, M::NamedTuple, mod_data::Dict)::NamedTuple
    variables = mod_data[:variables]

    if isempty(variables)
        error("The TPS model requires coordinate variables, e.g., `random(x, y, model=:tps)`.")
    end

    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for TPS model not found in data.")
        end
    end

    # Ensure data is on CPU for initial processing
    coords_cpu = Matrix{Float64}(M.data[!, Symbol.(variables)])
    
    n_obs, n_dims = size(coords_cpu)
    n_latent = m.nbins

    knot_method = get(mod_data[:params], :knot_method, :kmeans)
    knots_cpu = generate_inducing_points(coords_cpu, n_latent; method=string(knot_method))
    actual_n_knots = size(knots_cpu, 1)
    if actual_n_knots < n_latent
        @warn "TPS: Could only generate $(actual_n_knots) unique knots, requested $(n_latent). Using $(actual_n_knots)."
        n_latent = actual_n_knots
    end

    # Basis matrix calculation on CPU
    B_cpu = zeros(Float64, n_obs, n_latent)
    if n_dims == 1
        for i in 1:n_latent
            r = abs.(coords_cpu[:, 1] .- knots_cpu[i, 1])
            B_cpu[:, i] .= r.^3
        end
    elseif n_dims == 2
        for i in 1:n_latent
            dist_sq = (coords_cpu[:, 1] .- knots_cpu[i, 1]).^2 .+ (coords_cpu[:,
                2] .- knots_cpu[i, 2]).^2
            r = sqrt.(dist_sq)
            B_cpu[:, i] .= (r.^2) .* log.(r .+ 1e-9)
        end
    else
        for i in 1:n_latent
            dist_sq = sum((coords_cpu .- knots_cpu[i, :]').^2, dims=2)
            r = sqrt.(dist_sq)
            if isodd(n_dims)
                B_cpu[:, i] .= r.^(4 - n_dims)
            else
                B_cpu[:, i] .= (r.^(4 - n_dims)) .* log.(r .+ 1e-9)
            end
        end
    end
    
    # Penalty matrix and spectral decomposition on CPU
    template = build_structure_template(:rw2, n_latent)
    Q_template_cpu = template.matrix
    
    rank_deficiency = 2
    eig_decomp = eigen(Symmetric(Matrix(Q_template_cpu)))
    U_cpu = eig_decomp.vectors
    L_cpu = eig_decomp.values
    scaling_factor = _compute_scaling_factor(L_cpu, rank_deficiency)
    
    Q_template_scaled_cpu = Q_template_cpu ./ scaling_factor
    L_scaled_cpu = L_cpu ./ scaling_factor

    # Pre-compute dense Cholesky factor for the :cholesky method on the CPU.
    F_cpu = cholesky(Symmetric(Matrix(Q_template_scaled_cpu) + M.noise * I))

    # All precomputed structures remain on the CPU.
    return (
        basis_matrix = B_cpu,
        Q_template = Q_template_scaled_cpu,
        scaling_factor = scaling_factor,
        U = U_cpu,
        L = L_scaled_cpu,
        n_latent = n_latent,
        knots = knots_cpu,
        cholesky_factor = F_cpu
    )
end

"""
    _tps_log_marginal_likelihood(y_residual, B_basis, Q_penalty, L_eig, sigma, y_sigma, noise=1e-6)

Computes the exact log marginal likelihood for a TPS component with basis coefficients
  integrated out analytically.
"""
function _tps_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    B_basis::AbstractMatrix,
    Q_penalty::AbstractMatrix,
    L_eig::AbstractVector,
    sigma::T,
    y_sigma::T,
    noise::Real=1e-6
) where {T}
    N = length(y_residual)
    K = size(B_basis, 2)
    T_num = promote_type(T, typeof(noise))
    
    inv_sigma_y2 = one(T_num) / (y_sigma^2 + T_num(noise))
    scale = sigma^2 + T_num(noise)
    
    BTB = Matrix{T_num}(B_basis' * B_basis)
    BTy = Vector{T_num}(B_basis' * y_residual)
    
    Q_base = Matrix{T_num}(Q_penalty) .+ (scale * inv_sigma_y2) .* BTB
    for k in 1:K
        Q_base[k, k] += T_num(noise)
    end
    
    F = cholesky(Symmetric(Q_base))
    
    # RW2 has rank deficiency of 2
    valid_eigs = L_eig[3:end]
    log_det_prior = isempty(valid_eigs) ? zero(T_num) : sum(log.(valid_eigs .+ T_num(noise)))
    log_det_diff = - max(K - 2, 1) * log(scale) + log_det_prior - 2 * sum(log.(diag(F.U)))
    
    b = BTy .* inv_sigma_y2
    v = F.L \ b
    quad_term = scale * dot(v, v)
    
    log_lik = - (N / 2) * log(2 * T_num(pi) * (y_sigma^2 + T_num(noise))) -
              (inv_sigma_y2 / 2) * dot(y_residual, y_residual) +
              (1 / 2) * log_det_diff +
              (1 / 2) * quad_term
              
    return log_lik
end

function get_priors(
    m::TPS, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    sigma_prior_str = _distribution_to_string(m.sigma)
    key = spec.key
    
    if m.method == :marginalized
        return """
            $(p_names.sigma) ~ $(sigma_prior_str)
        """
    else
        return """
            $(p_names.sigma) ~ $(sigma_prior_str)
            $(p_names.ure) ~ MvNormal(zeros(T, spec_registry[:$(key)].hyper.n_latent), I)
        """
    end
end

function get_updates(
    m::TPS, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
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
        # --- Thin Plate Spline (TPS) Smoother (Spectral): $(key) ---
        let
            $(common_code)
            local diag_D = $(p_names.sigma) ./ sqrt.(hyper.L .+ M.noise)
            diag_D[1] = 0.0; diag_D[2] = 0.0
            local coeffs = hyper.U * (diag_D .* $(p_names.ure))
            $(p_names.sre) = B_basis * coeffs
            $(eta_target) = $(eta_target) .+ $(p_names.sre)
        end
    """

    cholesky_code = """
        # --- Thin Plate Spline (TPS) Smoother (Cholesky, AD-Safe): $(key) ---
        let
            $(common_code)
            local F = hyper.cholesky_factor
            local coeffs_unscaled = F.L' \\ $(p_names.ure)
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * hyper.n_latent), sum(coeffs_unscaled))
            local coeffs = $(p_names.sigma) .* (coeffs_unscaled .- mean(coeffs_unscaled))
            $(p_names.sre) = B_basis * coeffs
            $(eta_target) = $(eta_target) .+ $(p_names.sre)
        end
    """

    cholesky_sparse_code = """
        # --- Thin Plate Spline (TPS) Smoother (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            $(common_code)
            local Q_penalty = hyper.Q_template
            local F = cholesky(Symmetric(Q_penalty + M.noise * I))
            local coeffs_unscaled = F.L' \\ $(p_names.ure)
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * hyper.n_latent), sum(coeffs_unscaled))
            local coeffs = $(p_names.sigma) .* coeffs_unscaled
            $(p_names.sre) = B_basis * coeffs
            $(eta_target) = $(eta_target) .+ $(p_names.sre)
        end
    """

    marginalized_code = """
        # --- Thin Plate Spline (TPS) Smoother (Marginalized): $(key) ---
        let
            $(common_code)
            y_residual = M.y_obs .- $(eta_target)
            log_lik_marginalized_$(key) = _tps_log_marginal_likelihood(
                y_residual,
                B_basis,
                hyper.Q_template,
                hyper.L,
                $(p_names.sigma),
                y_sigma,
                M.noise
            )
            Turing.@addlogprob! log_lik_marginalized_$(key)
        end
    """

    if m.method == :spectral
        return spectral_code
    elseif m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    elseif m.method == :marginalized
        return marginalized_code
    else; error("Unsupported method '$(m.method)' for TPS component. Use :spectral, :cholesky, :cholesky_sparse, or :marginalized."); end
end


function get_effects(
    m::TPS, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = _get_chain_n_samples(chain)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    hyper = spec.hyper
    noise = M.noise
    n_latent = hyper.n_latent
    knots_cpu = hyper.knots
    n_dims = size(knots_cpu, 2)
    B_train_cpu = hyper.basis_matrix

    # --- Coordinate and Basis Matrix Handling for Prediction ---
    coord_vars = get(spec.params, :positional_args, [])
    has_ps = !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
    
    N_train = size(B_train_cpu, 1)
    N_total = has_ps ? N_train + size(PS.data, 1) : N_train

    B_full_cpu = if has_ps
        coords_pred_cpu = Matrix{Float64}(PS.data[!, Symbol.(coord_vars)])
        
        B_pred_cpu = zeros(Float64, size(coords_pred_cpu, 1), n_latent)
        
        if n_dims == 1
            r = abs.(coords_pred_cpu[:, 1] .- knots_cpu[:, 1]')
            B_pred_cpu .= r.^3
        elseif n_dims == 2
            dist_sq = (coords_pred_cpu[:, 1] .- knots_cpu[:,
                1]').^2 .+ (coords_pred_cpu[:, 2] .- knots_cpu[:, 2]').^2
            r = sqrt.(dist_sq)
            B_pred_cpu .= (r.^2) .* log.(r .+ 1e-9)
        else
            for i in 1:n_latent
                dist_sq = sum((coords_pred_cpu .- knots_cpu[i, :]').^2, dims=2)
                r = sqrt.(dist_sq)
                if isodd(n_dims)
                    B_pred_cpu[:, i] .= r.^(4 - n_dims)
                else
                    B_pred_cpu[:, i] .= (r.^(4 - n_dims)) .* log.(r .+ 1e-9)
                end
            end
        end
        vcat(B_train_cpu, B_pred_cpu)
    else
        B_train_cpu
    end

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop ---
    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names, string(v.sigma), k, is_multivariate_model)

        if isempty(sigma_name)
            @warn "Parameters for TPS component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, size(B_full_cpu, 1), n_samples))
            continue
        end

        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        effect_k_cpu = zeros(Float64, size(B_full_cpu, 1), n_samples)

        # --- Sample-wise Reconstruction ---
        if m.method == :marginalized
            y_sigma_name = _find_parameter(p_names, "y_sigma", k, is_multivariate_model)
            y_sigma_samples = if !isempty(y_sigma_name)
                get_params_vector(chain, y_sigma_name, 1)[:, 1]
            else
                fill(1.0, n_samples)
            end
            
            y_vec = M.y_obs isa AbstractMatrix ? M.y_obs[:, k] : M.y_obs
            BTB = Matrix{Float64}(B_train_cpu' * B_train_cpu)
            BTy = Vector{Float64}(B_train_cpu' * y_vec)
            
            coeffs_samples_matrix = zeros(Float64, n_latent, n_samples)
            for i in 1:n_samples
                sig = sigma_samples_cpu[i]
                y_sig = y_sigma_samples[i]
                
                scale = sig^2 + noise
                inv_sigma_y2 = 1.0 / (y_sig^2 + noise)
                
                Q_base = Matrix{Float64}(hyper.Q_template) .+ (scale * inv_sigma_y2) .* BTB
                for j in 1:n_latent
                    Q_base[j, j] += noise
                end
                
                F = cholesky(Symmetric(Q_base))
                b = BTy .* inv_sigma_y2
                mu = scale .* (F \ b)
                
                z = randn(n_latent)
                coeffs_samples_matrix[:, i] = mu .+ sqrt(max(scale, 1e-12)) .* (F.U \ z)
            end
            effect_k_cpu = B_full_cpu * coeffs_samples_matrix
        else
            ure_name = _find_parameter(p_names, string(v.ure), k, is_multivariate_model)
            if isempty(ure_name)
                @warn "ure for TPS component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, size(B_full_cpu, 1), n_samples))
                continue
            end
            ure_samples_cpu = get_params_matrix(chain, ure_name, n_latent)

            for i in 1:n_samples
                sigma_i = sigma_samples_cpu[i]
                innovations_i = ure_samples_cpu[i, :]
                
                local coeffs_cpu
                if m.method == :spectral
                    U_cpu, L_cpu = hyper.U, hyper.L
                    diag_D = sigma_i ./ sqrt.(L_cpu .+ noise)
                    diag_D[1] = 0.0; diag_D[2] = 0.0
                    coeffs_cpu = U_cpu * (diag_D .* innovations_i)
                else # :cholesky or :cholesky_sparse
                    F_cpu = hyper.cholesky_factor
                    coeffs_unscaled = F_cpu.L' \ innovations_i
                    coeffs_centered = coeffs_unscaled .- mean(coeffs_unscaled)
                    coeffs_cpu = sigma_i .* coeffs_centered
                end
                effect_k_cpu[:, i] = B_full_cpu * coeffs_cpu
            end
        end
        
        push!(structured_effects, effect_k_cpu)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
 
