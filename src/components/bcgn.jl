"""
    BCGN <: ComponentModel

A component for a Bipartite Graph Convolutional Network (BCGN), which models a
latent spatial field on one partition of a bipartite graph using a Gaussian
Markov Random Field (GMRF) approach. This component is designed for data
structured on a bipartite graph, where nodes are divided into two disjoint sets,
and edges only connect nodes from different sets.

# Version
v1.0.4 (2026-08-11)

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
    get_datastructures!(m_type::Type{<:BCGN}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `BCGN` component. It establishes the spatial
context (s_idx, s_N, W) and then converts the unipartite adjacency matrix `W` into
a bipartite representation `B`, which is stored for the pre-computation step.

# Assumptions
- A base adjacency matrix `W` must be provided.
"""
function get_datastructures!(
    m_type::Type{<:BCGN}, M::Dict, mod_data::Dict
)::Bool
    process_spatial_module!(M, mod_data, Dict(), Dict())
    
    W = get(M, :W, nothing)
    if isnothing(W)
        error("The `bcgn` model requires an adjacency matrix `W`.")
    end

    bipartite_info = adjacency_to_bipartite(W)
    
    mod_data[:params][:bipartite_adj] = bipartite_info.bipartite_adj
    mod_data[:params][:set1_indices] = bipartite_info.set1
    mod_data[:params][:set2_indices] = bipartite_info.set2
    
    M[:s_N] = length(bipartite_info.set1)
    
    return true
end

"""
    get_precomputes(m::BCGN, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the precision matrix `Q_template` from the one-mode projection of the
bipartite graph, along with its spectral decomposition and Cholesky factorization.
"""
function get_precomputes(m::BCGN, M::NamedTuple, mod_data::Dict)::NamedTuple
    B = mod_data[:params][:bipartite_adj]
    if isempty(B) || all(iszero, B)
        error("BCGN component requires a non-empty bipartite adjacency matrix.")
    end

    W_proj = B * B'
    W_proj[diagind(W_proj)] .= 0
    W_proj = dropzeros(W_proj)

    D_proj = spdiagm(0 => vec(sum(W_proj, dims=2)))
    Q_template = D_proj - W_proj
    
    n_latent = size(Q_template, 1)

    # Spectral decomposition for AD-friendly sampling
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values
    
    # Dense Cholesky factor for the :cholesky method
    F = cholesky(Symmetric(Matrix(Q_template) + M.noise * I))

    return (
        Q_template=Q_template,
        U=U,
        L=L,
        n_latent=n_latent,
        cholesky_factor=F
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
    return """ # Priors for sigma and raw innovations
    $(p_names.sigma) ~ $(_distribution_to_string(m.sigma))
    $(p_names.innovations) ~ MvNormal(zeros(T, $(n_latent)), I)
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
        # This method uses spectral decomposition for AD-friendly sampling.
        U = spec_registry[:$(key)].hyper.U
        L = spec_registry[:$(key)].hyper.L
        diag_D = $(p_names.sigma) ./ sqrt.(L .+ M.noise)
        # Enforce sum-to-zero constraint by setting components corresponding to the null space to zero.
        diag_D[L .< 1e-6] .= 0.0
        
        latent_field = U * (diag_D .* $(p_names.innovations))
        $(eta_target) .+= view(latent_field, M.s_idx)
    """

    cholesky_code = """
        # --- BCGN Component (Cholesky, AD-Safe): $(key) ---
        # This method uses a dense Cholesky factorization for AD-safe sampling.
        F = spec_registry[:$(key)].hyper.cholesky_factor
        latent_field_raw = F.L' \\ $(p_names.innovations)
        
        # Apply soft sum-to-zero constraint for identifiability against the global intercept.
        Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_raw))
        
        latent_field = latent_field_raw .* $(p_names.sigma)
        $(eta_target) .+= view(latent_field, M.s_idx)
    """

    cholesky_sparse_code = """
        # --- BCGN Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        # This method uses sparse Cholesky factorization, which is generally not AD-safe
        # for gradient-based samplers but is retained as a didactic alternative.
        Q = spec_registry[:$(key)].hyper.Q_template
        F = cholesky(Symmetric(Q + M.noise * I))
        latent_field_raw = F.L' \\ $(p_names.innovations)
        
        # Apply soft sum-to-zero constraint for identifiability against the global intercept.
        Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_raw))
        
        latent_field = latent_field_raw .* $(p_names.sigma)
        $(eta_target) .+= view(latent_field, M.s_idx)
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
    get_effects(m::BCGN, chain, M::NamedTuple, n_samples, outcomes_N, spec, PS, N_total)::NamedTuple

Reconstructs the `BCGN` component's effect from the MCMC chain's posterior samples,
dispatching on the method used during sampling.
"""
function get_effects(
    m::BCGN, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    n_latent = spec.hyper.n_latent
    noise = M.noise
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))

    for k in 1:outcomes_N
        sigma_name = _find_parameter(p_names_vec, string(spec.key), "sigma", k, is_multivariate_model)
        innovations_name = _find_parameter(p_names_vec, string(spec.key), "innovations", k, is_multivariate_model)

        if isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for BCGN component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innovations_name, n_latent)

        effect_k = zeros(Float64, n_latent, n_samples)

        if m.method == :spectral
            U = spec.hyper.U
            L = spec.hyper.L
            for j in 1:n_samples
                diag_D = sigma_samples[j] ./ sqrt.(L .+ noise)
                diag_D[L .< 1e-6] .= 0.0
                effect_k[:, j] = U * (diag_D .* innovations_samples[j, :])
            end
        else # :cholesky or :cholesky_sparse
            F = spec.hyper.cholesky_factor
            for j in 1:n_samples
                latent_field_raw = F.L' \ innovations_samples[j, :]
                latent_field_raw .-= mean(latent_field_raw)
                effect_k[:, j] = latent_field_raw .* sigma_samples[j]
            end
        end
        
        s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)
        indexed_effects = effect_k[s_idx_full, :]
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
