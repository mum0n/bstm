"""
    ICAR <: ComponentModel

A component for an Intrinsic Conditional Autoregressive (ICAR) model, also known as
a Besag model. This is a fundamental model for spatial data on a lattice or graph,
where the value at a location is assumed to be conditionally dependent on the
average of its neighbors.

# Version
v1.3.0 (2026-08-17)

# Mathematical Summary
The ICAR model defines a Gaussian Markov Random Field (GMRF) with a singular
precision matrix (the graph Laplacian), making it an "intrinsic" GMRF. The
conditional distribution of the spatial effect \$\\phi_i\$ at location \$i\$, given all
other locations, is:
\$\\phi_i | \\phi_{j \\ne i} \\sim \\mathcal{N}\\left( \\frac{1}{d_i} \\sum_{j \\sim i} \\phi_j, \\frac{\\sigma^2}{d_i} \\right)\$
where \$j \\sim i\$ denotes that \$j\$ is a neighbor of \$i\$, and \$d_i\$ is the number of
neighbors.

The joint precision matrix is the graph Laplacian, \$Q = D - W\$, where \$D\$ is the
diagonal degree matrix and \$W\$ is the adjacency matrix. Because \$Q\$ is
rank-deficient (its rows sum to zero), a sum-to-zero constraint
(\$\\sum_i \\phi_i = 0\$) is imposed on the latent field to ensure identifiability
from the global intercept.

# Computational Methods
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the precision matrix. Recommended for gradient-based samplers.
- `:cholesky` (AD-friendly): Uses a pre-computed dense Cholesky factorization.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization,
  which is not compatible with most AD backends.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `s_idx`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@bstm`.
- **Optional (in `random()` call)**:
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the
    ICAR effect. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`).
    Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the ICAR effect.
- `innovations_<key>`: The raw standard normal innovations for the effect.
- `latent_<key>`: The reconstructed latent spatial field.

# Key References
- Besag, J. (1974). Spatial interaction and the statistical analysis of lattice
  systems. *Journal of the Royal Statistical Society: Series B (Methodological)*,
  36(2), 192-225.
- Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and
  Applications*. CRC Press.
- Wikipedia: Conditional autoregressive model
"""
struct ICAR <: ComponentModel
    sigma::Distribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:icar] = ICAR
COMPONENT_CONSTRUCTORS[:icar] = (p, params) -> ICAR(
    p.sigma, get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:icar] = :spatial

function get_precomputes(m::ICAR, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.s_N
    W = M.W
    to_device = M.to_device

    # build_structure_template returns CPU arrays
    template = build_structure_template(:icar, n; W=W)
    
    # Move precomputed structures to the target device
    Q_template_device = to_device(template.matrix)
    U_device = to_device(template.U)
    L_device = to_device(template.L)
    
    # Pre-compute the dense Cholesky factor for the :cholesky method on the target device
    F_device = cholesky(Symmetric(Matrix(Q_template_device) + M.noise * I))
    
    return (
        Q_template=Q_template_device, 
        scaling_factor=template.scaling_factor, 
        U=U_device, 
        L=L_device, 
        n_latent=n, 
        cholesky_factor=F_device
    )
end

function get_priors(
    m::ICAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    return """ # Priors for sigma and raw innovations
    $(p_names.sigma) ~ $(_distribution_to_string(m.sigma))
    $(p_names.innovations) ~ MvNormal(zeros(T, $(n_latent)), I)
    """
end

function get_updates(
    m::ICAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent

    spectral_code = """
        # --- ICAR Component: $(key) (Spectral Method) ---
        let
            hyper = spec_registry[:$(key)].hyper
            U = hyper.U
            L = hyper.L
            diag_D = $(p_names.sigma) ./ sqrt.(L .+ M.noise)
            diag_D[1] = 0.0 # Enforce sum-to-zero constraint
            $(p_names.latent) = U * (diag_D .* $(p_names.innovations))
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """

    cholesky_code = """
        # --- ICAR Component: $(key) (Cholesky Method, AD-Safe) ---
        let
            hyper = spec_registry[:$(key)].hyper
            F = hyper.cholesky_factor
            latent_field_raw = F.L' \\ $(p_names.innovations)
            
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_raw)
            )
            
            $(p_names.latent) = latent_field_raw .* $(p_names.sigma)
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """

    cholesky_sparse_code = """
        # --- ICAR Component: $(key) (Sparse Cholesky, Not AD-Safe) ---
        let
            hyper = spec_registry[:$(key)].hyper
            Q = hyper.Q_template
            F = cholesky(Symmetric(Q + M.noise * I))
            latent_field_raw = F.L' \\ $(p_names.innovations)
            
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_raw)
            )
            
            $(p_names.latent) = latent_field_raw .* $(p_names.sigma)
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """

    if m.method == :spectral
        return spectral_code
    elseif m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    else
        error("Unsupported method '$(m.method)' for ICAR component.")
    end
end

function get_effects(
    m::ICAR, chain, spec::NamedTuple, M::NamedTuple,
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

    # --- Coordinate/Index Handling: Combine training and prediction sets on device ---
    s_idx_train_device = M.s_idx # Already on device from main config
    s_idx_full_device = if !isnothing(PS) && hasproperty(PS.data, :s_idx)
        s_idx_pred_cpu = get(PS.data, :s_idx, [])
        vcat(s_idx_train_device, to_device(s_idx_pred_cpu))
    else
        s_idx_train_device
    end
    N_total = length(s_idx_full_device)

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for ICAR component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples_cpu = get_params_matrix(chain, innovations_name, n_latent)

        # Initialize the output matrix for the full latent field on the target device
        effect_k_latent_device = to_device(zeros(Float64, n_latent, n_samples))

        # --- Sample-wise Reconstruction on the Target Device ---
        if m.method == :spectral
            U_device = spec.hyper.U # Already on device
            L_device = spec.hyper.L # Already on device
            for j in 1:n_samples
                diag_D = sigma_samples_cpu[j] ./ sqrt.(L_device .+ noise)
                diag_D[1] = 0.0 # Enforce sum-to-zero
                innov_j_device = to_device(innovations_samples_cpu[j, :])
                effect_k_latent_device[:, j] = U_device * (diag_D .* innov_j_device)
            end
        else # :cholesky or :cholesky_sparse
            F_device = spec.hyper.cholesky_factor # Already on device
            for j in 1:n_samples
                innov_j_device = to_device(innovations_samples_cpu[j, :])
                latent_field_raw = F_device.L' \ innov_j_device
                latent_field_centered = latent_field_raw .- mean(latent_field_raw)
                effect_k_latent_device[:, j] = latent_field_centered .* sigma_samples_cpu[j]
            end
        end
        
        # Indexing on the device and moving the final result to CPU
        indexed_effects_device = effect_k_latent_device[s_idx_full_device, :]
        push!(structured_effects, Array(indexed_effects_device))
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end

