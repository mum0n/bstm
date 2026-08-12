"""
    SAR <: ComponentModel

A component model for the Simultaneous Autoregressive (SAR) effect, also known as a
proper CAR model. The value at each location is modeled as a linear combination of
its neighbors plus an independent innovation term, leading to a precision matrix of
the form `(I - ρW)'(I - ρW)`.

# Version
v1.1.1 (2026-08-12)

# Mathematical Summary
The Simultaneous Autoregressive (SAR) model defines a spatial random effect
\$\\boldsymbol{\\phi}\$ where the value at each location is a linear combination of its
neighbors plus an independent innovation term. The model is typically expressed as:
\$\\boldsymbol{\\phi} = \\rho \\mathbf{W} \\boldsymbol{\\phi} + \\boldsymbol{\\epsilon}\$
where:
- \$\\rho\$ is the spatial autoregressive parameter.
- \$\\mathbf{W}\$ is a row-standardized adjacency matrix.
- \$\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(\\mathbf{0}, \\sigma^2 \\mathbf{I})\$ are independent innovations.

The precision matrix \$\\mathbf{Q}\$ for the SAR model is then given by:
\$\\mathbf{Q} = \\frac{1}{\\sigma^2} (\\mathbf{I} - \\rho \\mathbf{W})^T (\\mathbf{I} - \\rho \\mathbf{W})\$

# Computational Methods
- `:cholesky` (Default, AD-friendly): An AD-safe method using dense Cholesky factorization.
- `:cholesky_sparse` (Didactic, Not AD-friendly): A more memory-efficient method using
  sparse Cholesky factorization, suitable for gradient-free samplers.

**Note on Spectral Method**: A direct spectral method (using pre-computed eigenvectors
and eigenvalues) is not provided for the SAR model because its precision matrix
depends on the sampled parameter `rho` in a way that prevents pre-computation of
its spectral decomposition.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@bstm`.
- **Optional (in `random()` call)**:
  - `rho`: `UnivariateDistribution`, prior for the spatial autoregressive parameter.
    Default: `Normal(0, 0.5)`. Should be constrained to ensure stationarity.
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the innovations.
    Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:cholesky` or `:cholesky_sparse`).
    Default: `:cholesky`.

# Outputs (Parameter Names)
- `rho_<key>`: The spatial autoregressive parameter.
- `sigma_<key>`: The standard deviation of the innovations.
- `innovations_<key>`: The raw standard normal innovations for the latent field.
- `latent_<key>`: The reconstructed latent SAR effect.

# Key References
- Cliff, A. D., & Ord, J. K. (1973). *Spatial Autocorrelation*. Pion.
- Wikipedia: Simultaneous autoregressive model
"""
struct SAR <: ComponentModel
    rho::Distribution
    sigma::Distribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:sar] = SAR
COMPONENT_CONSTRUCTORS[:sar] = (p, params) -> SAR(
    p.rho, p.sigma, get(params, :method, :cholesky)
)

MODEL_TO_STRUCTURE_MAP[:sar] = :spatial

function get_datastructures!(m_type::Type{<:SAR}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    variables = mod_data[:variables]

    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr || w_val isa Symbol
            calling_mod = get(M, :calling_module, Main)
            try
                M[:W] = Core.eval(calling_mod, w_val)
            catch e
                error("Could not evaluate `W` argument `$(w_val)` for SAR component. Error: $e")
            end
        else
            M[:W] = w_val
        end
    end

    if !haskey(M, :W)
        error("SAR model requires an adjacency matrix `W` to be provided.")
    end

    if !isa(M[:W], AbstractMatrix) || isempty(M[:W])
        error("Provided `W` for SAR model is not a valid non-empty matrix.")
    end

    M[:s_N] = size(M[:W], 1)

    if isempty(variables)
        M[:s_idx] = collect(1:M[:s_N])
        @warn "Spatial index variable not provided for SAR. Assuming `s_idx = 1:s_N`."
    else
        s_var_sym = Symbol(variables[1])
        if !hasproperty(M[:data], s_var_sym)
            error("Spatial index variable ':$s_var_sym' for SAR model not found in data.")
        end
        M[:s_idx] = M[:data][!, s_var_sym]
    end

    return true
end

function get_precomputes(m::SAR, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.s_N
    W = sparse(M.W)

    row_sums = sum(W, dims=2)
    non_zero_rows = findall(x -> x > 0, vec(row_sums))
    
    W_std = spzeros(Float64, n, n)
    if !isempty(non_zero_rows)
        D_inv = spdiagm(0 => 1.0 ./ row_sums[non_zero_rows])
        W_std[non_zero_rows, :] = D_inv * W[non_zero_rows, :]
    end

    return (Q_template=W_std, n_latent=n)
end

function get_priors(
    m::SAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent

    rho_prior_str = _distribution_to_string(m.rho)
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    return """
        $(p_names.rho) ~ $(rho_prior_str)
        $(p_names.sigma) ~ $(sigma_prior_str)
        $(p_names.innovations) ~ MvNormal(zeros(T, $(n_latent)), I)
    """
end

function get_updates(
    m::SAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent

    common_code = """
        local W_std = spec_registry[:$(key)].hyper.Q_template
        local L_op = I - $(p_names.rho) * W_std
        local A_sar = L_op' * L_op
        # Enforce numerical symmetry for AD compatibility. This is crucial when
        # L_op contains Dual numbers, as floating-point errors can break symmetry.
        local Q_sar = (A_sar + A_sar') / 2.0
        local Q_final = Symmetric(Q_sar / ($(p_names.sigma)^2) + M.noise * I)
    """

    cholesky_code = """
        # --- SAR Component (Cholesky, AD-Safe): $(key) ---
        let
            $(common_code)
            # The precision matrix Q_final must be converted to a dense Matrix
            # for the cholesky factorization to be AD-compatible. Add a small nugget
            # for numerical stability, especially with AD.
            F = cholesky(Matrix(Q_final + I * 1e-9))
            $(p_names.latent) = F.L' \\ $(p_names.innovations)
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """

    cholesky_sparse_code = """
        # --- SAR Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            $(common_code)
            # Sparse cholesky is not AD-compatible but is more memory-efficient.
            # Add a small nugget for numerical stability.
            F = cholesky(Q_final + I * 1e-9)
            $(p_names.latent) = F.L' \\ $(p_names.innovations)
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """

    if m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    else
        error("Unsupported method '$(m.method)' for SAR component.")
    end
end

function get_effects(
    m::SAR, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    n_latent = spec.hyper.n_latent
    noise = M.noise
    s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, get(PS, :s_idx, []))
    W_std = spec.hyper.Q_template
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))

    for k in 1:outcomes_N
        rho_name = _find_parameter(p_names_vec, string(spec.key), "rho", k, is_multivariate_model)
        sigma_name = _find_parameter(p_names_vec, string(spec.key), "sigma", k, is_multivariate_model)
        innovations_name = _find_parameter(p_names_vec, string(spec.key), "innovations", k, is_multivariate_model)

        if isempty(rho_name) || isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for SAR component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        rho_samples = get_params_vector(chain, rho_name, 1)[:, 1]
        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innovations_name, n_latent)

        effect_k = zeros(Float64, n_latent, n_samples)

        for i in 1:n_samples
            L_op = I(n_latent) - rho_samples[i] * W_std
            A_sar = L_op' * L_op
            # Enforce numerical symmetry.
            Q_sar = (A_sar + A_sar') / 2.0
            Q_final = Symmetric(Q_sar / (sigma_samples[i]^2) + noise * I)
            
            # Add a small nugget for numerical stability before factorization.
            F = cholesky(Matrix(Q_final + I * 1e-9))
            
            effect_k[:, i] = F.L' \ innovations_samples[i, :]
        end
        push!(structured_effects, effect_k[s_idx_full, :])
    end

    return (structured=structured_effects, noisy=structured_effects)
end
