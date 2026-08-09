
"""
    LocalAdaptive <: ComponentModel

A component model for a Local Adaptive spatial effect. This model combines a global
smoothing structure (based on a Leroux-style precision matrix) with local,
cluster-specific mean effects. This allows the model to capture both smooth spatial
trends and abrupt shifts between distinct spatial regions.

# Version
v1.0.0 (2026-08-08)

# Mathematical Summary
The `LocalAdaptive` component models a latent spatial field \$\\phi\$ as a non-zero
mean Gaussian Markov Random Field (GMRF). The mean of the field, \$\\boldsymbol{\\mu}\$,
is not constant but varies by spatial cluster, while the precision matrix,
\$\\mathbf{Q}\$, captures global spatial correlation.

\$\\boldsymbol{\\phi} \\sim \\mathcal{N}(\\boldsymbol{\\mu}, (\\sigma^2 \\mathbf{Q})^{-1})\$

1.  **Mean Structure (\$\\boldsymbol{\\mu}\$)**: The spatial domain is partitioned into \$k\$
    clusters using k-means on the area centroids. A separate mean effect, \$\\mu_g\$,
    is estimated for each cluster \$g\$. For any spatial unit \$i\$ belonging to
    cluster \$g\$, its mean is \$\\mu_i = \\mu_g\$. A sum-to-zero constraint is applied
    to the cluster means for identifiability.

2.  **Precision Structure (\$\\mathbf{Q}\$)**: The precision matrix is a proper CAR model
    (Leroux-style), defined as a convex combination of an identity matrix
    \$\\mathbf{I}\$ and a scaled ICAR precision matrix \$\\mathbf{Q}_{ICAR}\$:
    \$\\mathbf{Q} = (1-\\rho)\\mathbf{I} + \\rho\\mathbf{Q}_{ICAR}\$
    This allows the model to smoothly interpolate between unstructured random
    effects (\$\\rho=0\$) and a fully structured ICAR model (\$\\rho=1\$).

# Assumptions
- The provided adjacency matrix `W` represents a single connected graph.
- Spatial coordinates are available to perform clustering.

# Best Use Case
Modeling spatial processes with distinct sub-regions or regimes that exhibit
different baseline levels, while still sharing a common spatial correlation
structure. Examples include modeling species abundance across different ecoregions
or disease rates across areas with different public health policies.

# Key References
- Gelfand, A. E., Schmidt, A. M., Banerjee, S., & Sirmans, C. F. (2005).
  *Nonstationary multivariate process modeling through spatially varying
  coregionalization*. Test, 14(2), 263-312. (For concepts on non-stationary
  spatial modeling).
- Wikipedia: Cluster analysis

# Fields
- `rho::Distribution`: The prior for the spatial correlation parameter `rho`.
- `sigma::Distribution`: The prior for the overall standard deviation of the effect.
"""
struct LocalAdaptive <: ComponentModel
    rho::Distribution
    sigma::Distribution
end

# Add to the central component constructor registry.
COMPONENT_TYPE_REGISTRY[:localadaptive] = LocalAdaptive
COMPONENT_CONSTRUCTORS[:localadaptive] = (p, params) -> LocalAdaptive(p.rho, p.sigma)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[:localadaptive] = :spatial

"""
    get_datastructures!(m_type::Type{<:LocalAdaptive}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `LocalAdaptive` component. It ensures a
spatial context (`W`, `s_idx`, `s_N`) is established, computes centroids for all
spatial units, and performs k-means clustering to assign each spatial unit to a
cluster.
"""
function get_datastructures!(
    m_type::Type{<:LocalAdaptive}, M::Dict, mod_data::Dict
)::Bool
    process_spatial_module!(M, mod_data, Dict(), Dict())
    s_N = M[:s_N]
    data = M[:data]

    if !haskey(M, :centroids)
        @info "Centroids not found for localadaptive model. Attempting to compute from s_x and s_y coordinates."
        if hasproperty(data, :s_x) && hasproperty(data, :s_y) && hasproperty(data, :s_idx)
            coord_map = Dict{Int, Point2D}()
            for i in 1:nrow(data)
                idx = data.s_idx[i]
                if !haskey(coord_map, idx)
                    coord_map[idx] = Point2D(data.s_x[i], data.s_y[i])
                end
            end

            if length(coord_map) < s_N
                error("The `localadaptive` model requires coordinates for all $(s_N) spatial units, but only found coordinates for $(length(coord_map)) unique units in the data. Please provide a complete `centroids` vector as a keyword argument.")
            end

            centroids = [coord_map[i] for i in 1:s_N]
            M[:centroids] = centroids
        else
            error("The `localadaptive()` model requires centroids for clustering. Provide them via the `centroids` keyword argument, or ensure spatial coordinates (s_x, s_y) and indices (s_idx) are in the data frame.")
        end
    end
    
    centroids = M[:centroids]
    if length(centroids) != s_N
        error("The number of provided centroids ($(length(centroids))) does not match the number of spatial units s_N ($(s_N)).")
    end

    params = mod_data[:params]
    n_clusters = get(params, :n_clusters, 5)
    
    if length(centroids) < n_clusters
        @warn "Number of spatial units ($(length(centroids))) is less than the requested number of clusters ($n_clusters). Adjusting n_clusters to $(length(centroids))."
        n_clusters = length(centroids)
    end
    
    centroids_matrix = hcat([c.x for c in centroids], [c.y for c in centroids])'
    kmeans_result = kmeans(centroids_matrix, n_clusters; maxiter=200, display=:none)
    
    M[:cluster_assignments] = assignments(kmeans_result)
    M[:n_clusters] = nclusters(kmeans_result)
    
    return true
end

"""
    get_precomputes(m::LocalAdaptive, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the ICAR precision matrix template and its spectral decomposition.
It also retrieves the cluster assignments and count from the main configuration `M`
and stores them in the `hyper` registry.
"""
function get_precomputes(m::LocalAdaptive, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.s_N
    W = M.W

    W_sym = sparse((W + W') .> 0)
    D = spdiagm(0 => vec(sum(W_sym, dims=2)))
    Q_template = D - W_sym

    rank_deficiency = 1
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values
    scaling_factor = _compute_scaling_factor(L, rank_deficiency)
    
    Q_template_scaled = Q_template ./ scaling_factor
    L_scaled = L ./ scaling_factor

    n_clusters = get(M, :n_clusters, 0)
    cluster_assignments = get(M, :cluster_assignments, Int[])
    if n_clusters == 0 || isempty(cluster_assignments)
        error("LocalAdaptive precomputes failed: cluster information not found.")
    end

    return (
        Q_template=Q_template_scaled, 
        scaling_factor=scaling_factor, 
        U=U, 
        L=L_scaled, 
        n_latent=n,
        n_clusters=n_clusters,
        cluster_assignments=cluster_assignments
    )
end

"""
    get_priors(m::LocalAdaptive, spec::NamedTuple, arch::String, outcome_idx, M)

Generates priors for `rho`, `sigma`, the spatial innovations `raw`, and the
cluster mean innovations `mu_clusters_raw`.
"""
function get_priors(
    m::LocalAdaptive, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    rho_prior_str = _distribution_to_string(m.rho)
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    n_clusters = spec.precomputes.n_clusters
    
    return """
        $(p_names.rho) ~ $(rho_prior_str)
        $(p_names.sigma) ~ $(sigma_prior_str)
        $(p_names.raw) ~ MvNormal(zeros($(spec.precomputes.n_latent)), I)
        $(p_names.mu_clusters_raw) ~ MvNormal(zeros($(n_clusters)), I)
    """
end

"""
    get_updates(m::LocalAdaptive, spec::NamedTuple, arch::String, outcome_idx, M)

Generates code to construct the `LocalAdaptive` effect. It builds the non-zero
mean vector from cluster assignments and samples the latent field using a
non-centered parameterization.
"""
function get_updates(
    m::LocalAdaptive, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    n_clusters = spec.precomputes.n_clusters
    cluster_assignments = "spec_registry[:$(spec.key)].precomputes.cluster_assignments"
    
    return """
        # --- LocalAdaptive Component: $(spec.key) ---
        let
            local mu_clusters_raw = $(p_names.mu_clusters_raw)
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_clusters)), sum(mu_clusters_raw))
            local mean_vector = mu_clusters_raw[$(cluster_assignments)]

            local Q_template = spec_registry[:$(spec.key)].precomputes.Q_template
            local rho_val = $(p_names.rho)
            local Q_final = (rho_val .* Q_template) .+ ((1.0 - rho_val) .* I)
            
            local F = cholesky(Symmetric(Matrix(Q_final) + M.noise * I))
            local latent_centered = F.L' \\ $(p_names.raw)
            
            local final_latent_field = (mean_vector .+ latent_centered) .* $(p_names.sigma)
            
            $(eta_target) .+= final_latent_field[M.s_idx]
        end
    """
end

"""
    get_effects(m::LocalAdaptive, chain, M::NamedTuple, ...)

Reconstructs the `LocalAdaptive` component's effect from posterior samples.
"""
function get_effects(
    m::LocalAdaptive, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    n_latent = spec.precomputes.n_latent
    cluster_assignments = spec.precomputes.cluster_assignments
    Q_template = spec.precomputes.Q_template
    noise = M.noise
    n_clusters = spec.precomputes.n_clusters

    s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        rho_samples = get_params_vector(chain, string(p_names.rho), 1)[:, 1]
        raw_samples = get_params_vector(chain, string(p_names.raw), n_latent)
        mu_clusters_raw_samples = get_params_vector(
            chain, string(p_names.mu_clusters_raw), n_clusters
        )

        effect_k = zeros(Float64, N_total, n_samples)

        for i in 1:n_samples
            mu_clusters_centered = mu_clusters_raw_samples[i, :] .- mean(mu_clusters_raw_samples[i, :])
            mean_vector = mu_clusters_centered[cluster_assignments]
            
            Q_final = (rho_samples[i] .* Q_template) .+ ((1.0 - rho_samples[i]) .* I)
            F = cholesky(Symmetric(Matrix(Q_final) + noise * I))
            latent_centered = F.L' \ raw_samples[i, :]
            
            final_latent_field = (mean_vector .+ latent_centered) .* sigma_samples[i]
            
            effect_k[:, i] = view(final_latent_field, s_idx_full)
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
