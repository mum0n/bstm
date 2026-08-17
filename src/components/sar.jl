"""
    SAR <: ComponentModel

A component model for the Simultaneous Autoregressive (SAR) effect, also known as a
proper CAR model. The value at each location is modeled as a linear combination of
its neighbors plus an independent innovation term, leading to a precision matrix of
the form `(I - ρW)'(I - ρW)`.

# Version
v1.1.2 (2026-08-14)

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

function get_precomputes(m::SAR, M::NamedTuple, mod_data::Dict)::NamedTuple
    # Data validation moved from get_datastructures!
    if !hasproperty(M, :W)
        error("SAR model requires an adjacency matrix `W` to be provided via keyword.")
    end

    if !isa(M.W, AbstractMatrix) || isempty(M.W)
        error("Provided `W` for SAR model is not a valid non-empty matrix.")
    end

    s_N = size(M.W, 1)

    # The processor is now responsible for creating s_idx.
    if !hasproperty(M, :s_idx)
        error(
            "SAR component '$(mod_data[:key])' failed: s_idx not found in model " *
            "configuration. This should have been set by the model processor."
        )
    end

    W = sparse(M.W)
    row_sums = sum(W, dims=2)
    non_zero_rows = findall(x -> x > 0, vec(row_sums))
    
    W_std = spzeros(Float64, s_N, s_N)
    if !isempty(non_zero_rows)
        D_inv_vals = 1.0 ./ row_sums[non_zero_rows]
        D_inv = spdiagm(0 => vec(D_inv_vals))
        W_std[non_zero_rows, :] = D_inv * W[non_zero_rows, :]
    end

    return (Q_template=W_std, n_latent=s_N)
end

function get_priors(
    m::SAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    key = spec.key

    rho_prior_str = _distribution_to_string(m.rho)
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    return """
        $(p_names.rho) ~ $(rho_prior_str)
        $(p_names.sigma) ~ $(sigma_prior_str)
        $(p_names.innovations) ~ MvNormal(
            zeros(T, spec_registry[:$(key)].hyper.n_latent), I
        )
    """
end

function get_updates(
    m::SAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

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
            F = cholesky(Matrix(Q_final) + I * 1e-9)
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
"""
    get_effects(m::SAR, chain, spec, M, PS)

Reconstructs the SAR effect from posterior samples. This version is updated to
handle GPU arrays by moving sampled parameters to the device for computation and
moving the final results back to the CPU.
"""
function get_effects(
    m::SAR, chain::Chains, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device

    structured_effects = Vector{Matrix{Float64}}()
    n_latent = spec.hyper.n_latent
    noise = M.noise

    # --- Index Handling: Combine training and prediction sets on device ---
    s_idx_train = M.s_idx # Already on device
    s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx)
        s_idx_pred_cpu = get(PS.data, :s_idx, [])
        vcat(s_idx_train, to_device(s_idx_pred_cpu))
    else
        s_idx_train
    end
    N_total = length(s_idx_full)

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        rho_name = _find_parameter(p_names, string(p_names_k.rho), k, is_multivariate_model)
        innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(rho_name) || isempty(innovations_name)
            @warn "Parameters for SAR component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        rho_samples_cpu = get_params_vector(chain, rho_name, 1)[:, 1]
        innovations_samples_cpu = get_params_vector(chain, innovations_name, n_latent)
        
        # Initialize the output matrix for the full latent field on the target device
        effect_k_latent_device = to_device(zeros(Float64, n_latent, n_samples))
        
        W_std_device = spec.hyper.Q_template # Already on device
        I_device = to_device(Matrix(I, n_latent, n_latent))

        # --- Sample-wise Reconstruction on the Target Device ---
        for s in 1:n_samples
            rho_s_device = to_device(rho_samples_cpu[s])
            sigma_s_device = to_device(sigma_samples_cpu[s])
            innov_s_device = to_device(innovations_samples_cpu[s, :])

            L_op = I_device - rho_s_device * W_std_device
            A_sar = L_op' * L_op
            Q_sar = (A_sar + A_sar') / 2.0
            Q_final = Symmetric(Q_sar / (sigma_s_device^2) + noise * I_device)
            
            # Use dense Cholesky for AD-safety and GPU compatibility
            F = cholesky(Matrix(Q_final) + I_device * 1e-9)
            effect_k_latent_device[:, s] = F.L' \ innov_s_device
        end
        
        # Index the reconstructed effects for the full observation set and move back to CPU
        indexed_effects_device = effect_k_latent_device[s_idx_full, :]
        push!(structured_effects, Array(indexed_effects_device))
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
