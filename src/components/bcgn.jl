"""
    BCGN <: ComponentModel

A component for a Bipartite Graph Convolutional Network (BCGN), modeled as a
Gaussian Markov Random Field (GMRF). This component is designed for data structured
on a bipartite graph, where nodes are divided into two disjoint sets, and edges
only connect nodes from different sets.

# Version
v1.0.1 (2026-08-08)

# Mathematical Summary
The BCGN component models a latent spatial field on one partition of a bipartite
graph. Given an adjacency matrix \$W\$ for a unipartite graph, the model first
attempts to find a 2-coloring to partition the graph into two sets, \$V_1\$ and
\$V_2\$. This results in a bipartite adjacency matrix \$B\$ representing connections
between the two sets.

The spatial correlation for nodes in one partition (e.g., \$V_1\$) is then induced
by their shared connections in the other partition (\$V_2\$). This is achieved by
creating a **one-mode projection** of the bipartite graph onto \$V_1\$. The adjacency
matrix for this projected graph is given by \$W_{proj} = B B^T\$.

From this projected adjacency matrix, a standard graph Laplacian is constructed:
\$Q = D_{proj} - W_{proj}\$
where \$D_{proj}\$ is the diagonal degree matrix of \$W_{proj}\$. The latent field
\$\\phi\$ is then modeled as a GMRF with this precision matrix:
\$\\phi \\sim \\mathcal{N}(0, (\\sigma^2 Q)^{-1})\$.

# Assumptions
- The underlying graph structure is bipartite or can be reasonably approximated as
  such.
- The spatial effect is smooth with respect to the one-mode projected graph
  structure.

# Best Use Case
Modeling spatial or network effects where there are two distinct types of entities,
and interactions only occur between types (e.g., users and products, genes and
diseases, locations and events). It is useful for understanding the similarity
between nodes of one type based on the other types of nodes they connect to.

# Key References
- **Bipartite Graphs**: [Wikipedia: Bipartite Graph](https://en.wikipedia.org/wiki/Bipartite_graph)
- **Graph Convolutions**: Kipf, T. N., & Welling, M. (2016). *Semi-supervised
  classification with graph convolutional networks*. arXiv preprint arXiv:1609.02907.

# Fields
- `sigma::UnivariateDistribution`: The prior for the marginal standard deviation of
  the latent field.
"""
struct BCGN <: ComponentModel
    sigma::UnivariateDistribution
end

COMPONENT_TYPE_REGISTRY[:bcgn] = BCGN
COMPONENT_CONSTRUCTORS[:bcgn] = (p, params) -> BCGN(p.sigma)
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
bipartite graph.
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

    return (Q_template=Q_template, n_latent=n_latent)
end

"""
    get_priors(m::BCGN, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for the basis coefficients (`raw`) and overall scale (`sigma`).
"""
function get_priors(
    m::BCGN, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    return """
    $(p_names.sigma) ~ $(_distribution_to_string(m.sigma))
    $(p_names.raw) ~ MvNormal(zeros(T, $(n_latent)), I)
    """
end

"""
    get_updates(m::BCGN, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to compute the BCGN effect and add it to the linear predictor `eta`.
"""
function get_updates(
    m::BCGN, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    
    return """
        # --- BCGN Component: $(key) ---
        let
            local Q = spec_registry[:$(key)].hyper.Q_template
            local F = cholesky(Symmetric(Matrix(Q) + M.noise * I))
            local latent_field_raw = F.L' \\ $(p_names.raw)
            
            # Apply sum-to-zero constraint for identifiability
            Turing.@addlogprob! logpdf(
                Normal(0, 0.001 * $(spec.hyper.n_latent)), sum(latent_field_raw)
            )
            
            local latent_field = latent_field_raw .* $(p_names.sigma)
            
            # Note: The indexing assumes the user's s_idx corresponds to the
            # first partition of the graph.
            $(eta_target) .+= view(latent_field, M.s_idx)
        end
    """
end

"""
    get_effects(m::BCGN, chain, M::NamedTuple, ...)::NamedTuple

Reconstructs the `BCGN` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(
    m::BCGN, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    n_latent = spec.hyper.n_latent
    Q_template = spec.hyper.Q_template
    F = cholesky(Symmetric(Matrix(Q_template) + M.noise * I))

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        raw_samples = get_params_vector(chain, string(p_names.raw), n_latent)

        effect_k = zeros(Float64, n_latent, n_samples)
        for j in 1:n_samples
            latent_field_raw = F.L' \ raw_samples[j, :]
            latent_field_raw .-= mean(latent_field_raw) # Enforce sum-to-zero
            effect_k[:, j] = latent_field_raw .* sigma_samples[j]
        end
        
        s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)
        indexed_effects = effect_k[s_idx_full, :]
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
