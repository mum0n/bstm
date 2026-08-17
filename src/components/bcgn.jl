"""
    BCGN <: ComponentModel

A component for a Bipartite Graph Convolutional Network (BCGN), which models a
latent spatial field on one partition of a bipartite graph using a Gaussian
Markov Random Field (GMRF) approach. This component is designed for data
structured on a bipartite graph, where nodes are divided into two disjoint sets,
and edges only connect nodes from different sets.

# Version
v1.0.8 (2026-08-15)

# Mathematical Summary
The BCGN component models a latent spatial field on one partition of a bipartite
graph. Given an adjacency matrix \$W\$ for a unipartite graph, the model first
attempts to find a 2-coloring to partition the graph into two sets, \$V_1\$ and
\$V_2\$. This results in a bipartite adjacency matrix \$B\$ representing connections
between the two sets.

The spatial correlation for nodes in one partition (e.g., \$V_1\$) is then induced
by their shared connections in the other partition (\$V_2\$). This is achieved by
creating a **one-mode projection** of the bipartite graph onto \$V_1\$. The adjacency
matrix for this projected graph is given by \$W_{\\text{proj}} = B B^T\$.

From this projected adjacency matrix, a standard graph Laplacian is constructed:
\$Q = D_{\\text{proj}} - W_{\\text{proj}}\$
where \$D_{\\text{proj}}\$ is the diagonal degree matrix of \$W_{\\text{proj}}\$. The latent field
\$\\phi\$ is then modeled as a GMRF with this precision matrix, scaled by \$\\sigma^2\$:
\$\\phi \\sim \\mathcal{N}(0, (\\sigma^2 Q)^{-1})\$.

To ensure identifiability against a global intercept, a soft sum-to-zero constraint
is applied to the latent field.

# Computational Methods
- `:spectral` (Default, AD-friendly): Uses spectral decomposition of the precision
  matrix for efficient and AD-compatible sampling.
- `:cholesky` (AD-friendly): Uses dense Cholesky factorization of the precision
  matrix. AD-compatible but less efficient than spectral for large graphs.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization.
  Not AD-compatible for gradient-based samplers but retained as a didactic alternative.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `s_idx`).
  - An adjacency matrix `W` passed as a keyword argument to `@bstm`.
- **Optional (in `random()` call)**:
  - `sigma`: `UnivariateDistribution`, prior for the marginal standard deviation. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`). Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The marginal standard deviation of the latent field.
- `innovations_<key>`: The raw standard normal innovations for the latent field.

# Key References
- Kipf, T. N., & Welling, M. (2016). *Semi-supervised classification with graph convolutional networks*. arXiv preprint arXiv:1609.02907.
- Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and Applications*. CRC Press.
"""
struct BCGN <: ComponentModel
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:bcgn] = BCGN
COMPONENT_CONSTRUCTORS[:bcgn] = (p, params) -> BCGN(
    p.sigma, get(params, :method, :spectral)
)
MODEL_TO_STRUCTURE_MAP[:bcgn] = :spatial


"""
    get_precomputes(m::BCGN, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the precision matrix `Q_template` from the one-mode projection of the
bipartite graph, along with its spectral decomposition, Cholesky factorization,
and a mapping matrix to link latent effects to observations.
"""
function get_precomputes(m::BCGN, M::NamedTuple, mod_data::Dict)::NamedTuple
    W = get(M, :W, nothing)
    if isnothing(W)
        error("The `bcgn` model requires an adjacency matrix `W`.")
    end

    bipartite_info = adjacency_to_bipartite(W)
    B = bipartite_info.bipartite_adj
    set1_indices = bipartite_info.set1

    if isempty(B) || all(iszero, B)
        error("BCGN component requires a non-empty bipartite adjacency matrix.")
    end

    W_proj = B * B'
    W_proj[diagind(W_proj)] .= 0
    W_proj = dropzeros(W_proj)

    D_proj = spdiagm(0 => vec(sum(W_proj, dims=2)))
    Q_template = D_proj - W_proj
    
    n_latent = size(Q_template, 1)

    set1_map = Dict(original_idx => new_idx for (new_idx, original_idx) in enumerate(set1_indices))
    
    N_obs = M.y_N
    mapping_matrix = spzeros(Float64, N_obs, n_latent)
    for i in 1:N_obs
        original_s_idx = M.s_idx[i]
        if haskey(set1_map, original_s_idx)
            latent_idx = set1_map[original_s_idx]
            mapping_matrix[i, latent_idx] = 1.0
        end
    end

    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values
    
    F = cholesky(Symmetric(Matrix(Q_template) + M.noise * I))

    return (
        Q_template=Q_template,
        U=U,
        L=L,
        n_latent=n_latent,
        cholesky_factor=F,
        mapping_matrix=mapping_matrix,
        set1_indices=set1_indices
    )
end


"""
    get_priors(m::BCGN, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for the innovations and overall scale (`sigma`).
"""
function get_priors(
    m::BCGN, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    return """
    # Priors for BCGN component: $(spec.key)
    $(p_names.sigma) ~ $(_distribution_to_string(m.sigma))
    $(p_names.innovations) ~ MvNormal(zeros($(n_latent)), I)
    """
end

"""
    get_updates(m::BCGN, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to compute the BCGN effect and add it to the linear predictor `eta`.
Supports three methods: `:spectral`, `:cholesky`, and `:cholesky_sparse`.
"""
function get_updates(
    m::BCGN, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent

    spectral_code = """
        # --- BCGN Component (Spectral): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            U = hyper.U
            L = hyper.L
            diag_D = $(p_names.sigma) ./ sqrt.(L .+ M.noise)
            diag_D[L .< 1e-6] .= 0.0
            
            latent_field = U * (diag_D .* $(p_names.innovations))
            $(eta_target) .+= hyper.mapping_matrix * latent_field
        end
    """

    cholesky_code = """
        # --- BCGN Component (Cholesky, AD-Safe): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            F = hyper.cholesky_factor
            latent_field_raw = F.L' \\ $(p_names.innovations)
            
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_raw))
            
            latent_field = latent_field_raw .* $(p_names.sigma)
            $(eta_target) .+= hyper.mapping_matrix * latent_field
        end
    """

    cholesky_sparse_code = """
        # --- BCGN Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            Q = hyper.Q_template
            F = cholesky(Symmetric(Q + M.noise * I))
            latent_field_raw = F.L' \\ $(p_names.innovations)
            
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_raw))
            
            latent_field = latent_field_raw .* $(p_names.sigma)
            $(eta_target) .+= hyper.mapping_matrix * latent_field
        end
    """

    if m.method == :spectral
        return spectral_code
    elseif m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    else
        error("Unsupported method '$(m.method)' for BCGN component. Use `:spectral`, `:cholesky`, or `:cholesky_sparse`.")
    end
end

"""
    get_effects(m::BCGN, chain, spec, M, PS)

Reconstructs the `BCGN` component's effect from the MCMC chain's posterior samples.
  handle GPU arrays by moving sampled parameters to the
device for computation and moving the final results back to the CPU. It also uses
the modern, simplified function signature.
"""
function get_effects(
    m::BCGN, chain::Chains, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device
    noise = M.noise
    n_latent = spec.hyper.n_latent

    # --- Coordinate/Index Handling: Combine training and prediction sets on CPU ---
    s_idx_train_cpu = M.s_idx isa AbstractArray ? Array(M.s_idx) : M.s_idx
    
    s_idx_full_cpu = if !isnothing(PS) && hasproperty(PS.data, :s_idx)
        vcat(s_idx_train_cpu, PS.data.s_idx)
    else
        s_idx_train_cpu
    end
    N_total = length(s_idx_full_cpu)

    # --- Mapping Matrix Construction: Build on CPU, then move to device ---
    set1_map = Dict(original_idx => new_idx for (new_idx, original_idx) in enumerate(spec.hyper.set1_indices))
    
    mapping_matrix_cpu = spzeros(Float64, N_total, n_latent)
    for i in 1:N_total
        original_s_idx = s_idx_full_cpu[i]
        if haskey(set1_map, original_s_idx)
            latent_idx = set1_map[original_s_idx]
            mapping_matrix_cpu[i, latent_idx] = 1.0
        end
    end
    # Move to device as a dense matrix for efficient multiplication
    mapping_matrix_full = to_device(Matrix(mapping_matrix_cpu))

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for BCGN component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples from the chain (these are on the CPU)
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples_cpu = get_params_matrix(chain, innovations_name, n_latent)

        # Initialize the output matrix for latent effects on the target device
        effect_k_device = to_device(zeros(Float64, n_latent, n_samples))

        # --- Sample-wise Reconstruction on the Target Device ---
        if m.method == :spectral
            U = spec.hyper.U # Already on device
            L = spec.hyper.L # Already on device
            innov_samples_device = to_device(innovations_samples_cpu') # Move all innovations at once

            for j in 1:n_samples
                sigma_j = sigma_samples_cpu[j] # CPU scalar
                
                diag_D = sigma_j ./ sqrt.(L .+ noise)
                diag_D[L .< 1e-6] .= 0.0
                effect_k_device[:, j] = U * (diag_D .* innov_samples_device[:, j])
            end
        else # :cholesky or :cholesky_sparse
            F = spec.hyper.cholesky_factor # Already on device
            innov_samples_device = to_device(innovations_samples_cpu')

            for j in 1:n_samples
                sigma_j = sigma_samples_cpu[j]

                latent_field_raw = F.L' \ innov_samples_device[:, j]
                latent_field_raw .-= mean(latent_field_raw)
                effect_k_device[:, j] = latent_field_raw .* sigma_j
            end
        end
        
        # Apply mapping to get observation-level effects and move the final result to CPU
        indexed_effects_device = mapping_matrix_full * effect_k_device
        push!(structured_effects, Array(indexed_effects_device))
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
