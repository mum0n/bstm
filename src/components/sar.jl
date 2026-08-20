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

Reconstructs the SAR effect from posterior samples. 
"""
function get_effects(
    m::SAR, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = size(chain, 1) * FlexiChains.nchains(chain)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    noise = M.noise
    
    n_latent = spec.hyper.n_latent
    W_dag = spec.hyper.Q_template

    # --- Coordinate/Index Handling: Combine training and prediction sets ---
    s_idx_train = M.s_idx # Spatial indices for training data
    s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx) # If prediction set is provided
        vcat(s_idx_train, PS.data.s_idx) # Combine training and prediction indices
    else
        s_idx_train # Otherwise, use only training indices
    end
    N_total = length(s_idx_full) # Total number of observations (training + prediction)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        
        # Find parameter names in the MCMC chain
        rho_name = _find_parameter(p_names, string(p_names_k.rho), k, is_multivariate_model)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)

        if isempty(rho_name) || isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for SAR component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        rho_samples = get_params_vector(chain, rho_name, 1) # (n_samples, 1)
        sigma_samples = get_params_vector(chain, sigma_name, 1) # (n_samples, 1)
        innovations_samples = get_params_matrix(chain, innovations_name, n_latent) # (n_samples, n_latent)
        
        # Initialize the output matrix for latent effects
        latent_field_matrix = zeros(Float64, n_latent, n_samples)

        # --- Sample-wise Reconstruction ---
        if m.method == :forward_substitution
            for i in 1:n_samples
                innov_i = innovations_samples[i, :]
                latent_field_i = zeros(Float64, n_latent)
                
                for j in 1:n_latent
                    parent_effect = 0.0
                    for j_ptr in nzrange(W_dag, j)
                        parent_idx = W_dag.rowval[j_ptr]
                        parent_effect += W_dag.nzval[j_ptr] * latent_field_i[parent_idx]
                    end
                    latent_field_i[j] = rho_samples[i, 1] * parent_effect + innov_i[j]
                end
                latent_field_matrix[:, i] = latent_field_i .* sigma_samples[i, 1]
            end
        else # :precision
            for i in 1:n_samples
                innov_i = innovations_samples[i, :]
                
                L_op = I - rho_samples[i, 1] * W_dag
                Q = L_op' * L_op
                
                F = cholesky(Symmetric(Matrix(Q) + noise * I))
                latent_field_i = sigma_samples[i, 1] .* (F.U \ innov_i)
                latent_field_matrix[:, i] = latent_field_i
            end
        end
        # Index the reconstructed latent effects to match the observation indices
        indexed_effects = latent_field_matrix[s_idx_full, :]
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
 