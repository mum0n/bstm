"""
    TPS <: ComponentModel

A component model for a Thin Plate Spline (TPS) smoother. This component creates a
basis of radial basis functions centered at knots distributed across the covariate
space. The effect is a linear combination of these basis functions, with coefficients
regularized by a random walk prior to ensure smoothness.

# Version
v1.1.1 (2026-08-14)

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
  - `knot_method`: `Symbol`, method for placing knots (`:kmeans`, `:random`, `:quantile`, `:range`). Default: `:kmeans`.

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

    coords = Matrix{Float64}(M.data[!, Symbol.(variables)])
    
    n_obs, n_dims = size(coords)
    n_latent = m.nbins

    knot_method = get(mod_data[:params], :knot_method, :kmeans)
    knots = generate_inducing_points(coords, n_latent; method=string(knot_method))
    actual_n_knots = size(knots, 1)
    if actual_n_knots < n_latent
        @warn "TPS: Could only generate $(actual_n_knots) unique knots, requested $(n_latent). Using $(actual_n_knots)."
        n_latent = actual_n_knots
    end

    B = zeros(Float64, n_obs, n_latent)
    if n_dims == 1
        for i in 1:n_latent; r = abs.(coords[:, 1] .- knots[i, 1]); B[:, i] .= r.^3; end
    elseif n_dims == 2
        for i in 1:n_latent; dist_sq = (coords[:, 1] .- knots[i, 1]).^2 .+ (coords[:, 2] .- knots[i, 2]).^2; r = sqrt.(dist_sq); B[:, i] .= (r.^2) .* log.(r .+ 1e-9); end
    else
        for i in 1:n_latent; dist_sq = sum((coords .- knots[i, :]').^2, dims=2); r = sqrt.(dist_sq); if isodd(n_dims); B[:, i] .= r.^(4 - n_dims); else; B[:, i] .= (r.^(4 - n_dims)) .* log.(r .+ 1e-9); end; end
    end
    
    template = build_structure_template(:rw2, n_latent)
    Q_template = template.matrix
    
    rank_deficiency = 2
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
        knots=knots,
        cholesky_factor=F
    )
end

function get_priors(
    m::TPS, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    sigma_prior_str = _distribution_to_string(m.sigma)
    key = spec.key
    
    return """
        $(p_names.sigma) ~ $(sigma_prior_str)
        $(p_names.innovations) ~ MvNormal(zeros(T, spec_registry[:$(key)].hyper.n_latent), I)
    """
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
            local coeffs = hyper.U * (diag_D .* $(p_names.innovations))
            $(p_names.latent) = B_basis * coeffs
            $(eta_target) .+= $(p_names.latent)
        end
    """

    cholesky_code = """
        # --- Thin Plate Spline (TPS) Smoother (Cholesky, AD-Safe): $(key) ---
        let
            $(common_code)
            local F = hyper.cholesky_factor
            local coeffs_raw = F.L' \\ $(p_names.innovations)
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * hyper.n_latent), sum(coeffs_raw))
            local coeffs = $(p_names.sigma) .* coeffs_raw
            $(p_names.latent) = B_basis * coeffs
            $(eta_target) .+= $(p_names.latent)
        end
    """

    cholesky_sparse_code = """
        # --- Thin Plate Spline (TPS) Smoother (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            $(common_code)
            local Q_penalty = hyper.Q_template
            local F = cholesky(Symmetric(Q_penalty + M.noise * I))
            local coeffs_raw = F.L' \\ $(p_names.innovations)
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * hyper.n_latent), sum(coeffs_raw))
            local coeffs = $(p_names.sigma) .* coeffs_raw
            $(p_names.latent) = B_basis * coeffs
            $(eta_target) .+= $(p_names.latent)
        end
    """

    if m.method == :spectral; return spectral_code;
    elseif m.method == :cholesky; return cholesky_code;
    elseif m.method == :cholesky_sparse; return cholesky_sparse_code;
    else; error("Unsupported method '$(m.method)' for TPS component."); end
end

function get_effects(
    m::TPS, chain, M::NamedTuple, n_samples::Int, is_multivariate_model::Bool,
    outcomes_N::Int, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    hyper = spec.hyper
    noise = M.noise
    n_latent = hyper.n_latent
    knots = hyper.knots
    n_dims = size(knots, 2)

    B_train = hyper.basis_matrix
    B_full = if !isnothing(PS)
        coord_vars = get(spec.params, :positional_args, [])
        if all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
            coords_pred = Matrix{Float64}(PS.data[!, Symbol.(coord_vars)])
            B_pred = zeros(Float64, size(coords_pred, 1), n_latent)
            if n_dims == 1
                for i in 1:n_latent; r = abs.(coords_pred[:, 1] .- knots[i, 1]); B_pred[:, i] .= r.^3; end
            elseif n_dims == 2
                for i in 1:n_latent; dist_sq = (coords_pred[:, 1] .- knots[i, 1]).^2 .+ (coords_pred[:, 2] .- knots[i, 2]).^2; r = sqrt.(dist_sq); B_pred[:, i] .= (r.^2) .* log.(r .+ 1e-9); end
            else
                for i in 1:n_latent; dist_sq = sum((coords_pred .- knots[i, :]').^2, dims=2); r = sqrt.(dist_sq); if isodd(n_dims); B_pred[:, i] .= r.^(4 - n_dims); else; B_pred[:, i] .= (r.^(4 - n_dims)) .* log.(r .+ 1e-9); end; end
            end
            vcat(B_train, B_pred)
        else
            B_train
        end
    else
        B_train
    end
    
    if size(B_full, 1) != N_total
        @warn "TPS effect reconstruction: dimension mismatch. Using in-sample effects only."
        B_full = B_train
    end

    p_names_vec = string.(FlexiChains.parameters(chain))

    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names_vec, v.sigma, k, is_multivariate_model)
        innovations_name = _find_parameter(p_names_vec, v.innovations, k, is_multivariate_model)

        if isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for TPS component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, size(B_full, 1), n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innovations_name, n_latent)

        effect_k = zeros(Float64, size(B_full, 1), n_samples)

        for i in 1:n_samples
            local coeffs
            if m.method == :spectral
                U, L = hyper.U, hyper.L
                diag_D = sigma_samples[i] ./ sqrt.(L .+ noise)
                diag_D[1] = 0.0; diag_D[2] = 0.0
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
