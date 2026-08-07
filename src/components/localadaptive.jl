# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    LocalAdaptive <: ComponentModel

A component model for a Local Adaptive spatial effect. This model combines a global
smoothing structure (based on a Leroux-style precision matrix) with local,
cluster-specific mean effects. This allows the model to capture both smooth spatial
trends and abrupt shifts between distinct spatial regions.

# Fields
- `rho::Distribution`: The prior for the spatial correlation parameter `rho`, which
  balances between an IID field and a structured ICAR field.
- `sigma::Distribution`: The prior for the overall standard deviation of the spatial effect.
"""
struct LocalAdaptive <: ComponentModel
    rho::Distribution
    sigma::Distribution
end

# Add to the central component constructor registry.
COMPONENT_CONSTRUCTORS[:localadaptive] = (p, params) -> LocalAdaptive(p.rho, p.sigma)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[LocalAdaptive] = :spatial

"""
    get_datastructures!(m_type::Type{<:LocalAdaptive}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `LocalAdaptive` component. It ensures a spatial
context (`W`, `s_idx`, `s_N`) is established, computes centroids for all spatial units,
and performs k-means clustering to assign each spatial unit to a cluster.
"""
function get_datastructures!(m_type::Type{<:LocalAdaptive}, M::Dict, mod_data::Dict)::Bool
    # First, run the standard spatial processor to set up W, s_idx, s_N, etc.
    process_spatial_module!(M, mod_data, Dict(), Dict())
    s_N = M[:s_N]
    data = M[:data]

    # The localadaptive model requires centroids for clustering.
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
    
    # Clustering.jl expects a [dims x n_points] matrix
    centroids_matrix = hcat([c.x for c in centroids], [c.y for c in centroids])'
    
    kmeans_result = kmeans(centroids_matrix, n_clusters; maxiter=200, display=:none)
    
    # Store cluster assignments and count in the main model config for precomputes.
    M[:cluster_assignments] = assignments(kmeans_result)
    M[:n_clusters] = nclusters(kmeans_result)
    
    return true
end

"""
    get_precomputes(m::LocalAdaptive, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations. It builds the ICAR precision matrix template
and computes its spectral decomposition. It also retrieves the cluster assignments and
count from the main configuration `M` and stores them in the `hyper` registry.
"""
function get_precomputes(m::LocalAdaptive, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.s_N
    W = M.W

    # Build the ICAR Q_template, which is the basis for the Leroux structure.
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

    # Retrieve cluster info from the main config where get_datastructures! stored it.
    n_clusters = get(M, :n_clusters, 0)
    cluster_assignments = get(M, :cluster_assignments, Int[])
    if n_clusters == 0 || isempty(cluster_assignments)
        error("LocalAdaptive precomputes failed: cluster information not found in model configuration.")
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
    get_priors(m::LocalAdaptive, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code for the priors on the `LocalAdaptive` component's parameters.
"""
function get_priors(m::LocalAdaptive, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    rho_prior_str = _distribution_to_string(m.rho)
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    n_clusters = spec.precomputes.n_clusters
    
    return """
        $(p_names.rho) ~ NamedDist($(rho_prior_str), :$(p_names.rho))
        $(p_names.sigma) ~ NamedDist($(sigma_prior_str), :$(p_names.sigma))
        $(p_names.innov) ~ NamedDist(MvNormal(zeros(T, $(spec.precomputes.n_latent)), I), :$(p_names.raw))
        $(p_names.latent) ~ NamedDist(MvNormal(zeros(T, $(n_clusters)), I), :$(p_names.latent)) # Using 'latent' for cluster means innovations
    """
end

"""
    get_updates(m::LocalAdaptive, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code to construct the `LocalAdaptive` component's effect.
"""
function get_updates(m::LocalAdaptive, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    n_clusters = spec.precomputes.n_clusters
    cluster_assignments = "spec_registry[:$(spec.key)].precomputes.cluster_assignments"
    
    return """
        # --- LocalAdaptive Component: $(spec.key) ---
        local mu_clusters_raw = $(p_names.latent)
        Turing.@addlogprob! logpdf(Normal(T(0), T(0.001) * $(n_clusters)), sum(mu_clusters_raw))
        local mean_vector = mu_clusters_raw[$(cluster_assignments)]

        local Q_template = spec_registry[:$(spec.key)].precomputes.Q_template
        local rho_val = $(p_names.rho)
        local Q_final = (rho_val .* Q_template) .+ ((1.0 - rho_val) .* I)
        
        local F = cholesky(Symmetric(Matrix(Q_final) + M.noise * I))
        local latent_centered = F.L' \\ $(p_names.innov)
        
        local final_latent_field = (mean_vector .+ latent_centered) .* $(p_names.sigma)
        
        $(eta_target) .+= final_latent_field[M.s_idx]
    """
end

"""
    get_effects(m::LocalAdaptive, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `LocalAdaptive` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::LocalAdaptive, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    sigma_samples = get(chain, p_names.sigma)
    rho_samples = get(chain, p_names.rho)
    raw_samples = get(chain, p_names.innov)
    cluster_innov_samples = get(chain, p_names.latent)

    n_latent = spec.precomputes.n_latent
    cluster_assignments = spec.precomputes.cluster_assignments
    Q_template = spec.precomputes.Q_template
    noise = M.noise

    idx_to_use = isnothing(PS) ? M.s_idx : PS.s_idx
    
    reconstructed_effects = zeros(n_samples, n_latent)

    for i in 1:n_samples
        current_sigma = sigma_samples[i]
        current_rho = rho_samples[i]
        current_raw = raw_samples[i, :]
        current_cluster_innov = cluster_innov_samples[i, :]
        
        mu_clusters_centered = current_cluster_innov .- mean(current_cluster_innov)
        mean_vector = mu_clusters_centered[cluster_assignments]
        
        Q_final = (current_rho .* Q_template) .+ ((1.0 - current_rho) .* I)
        F = cholesky(Symmetric(Matrix(Q_final) + noise * I))
        latent_centered = F.L' \ current_raw
        
        reconstructed_effects[i, :] = (mean_vector .+ latent_centered) .* current_sigma
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    indexed_mean = mean_effect[idx_to_use]
    indexed_lower = lower_ci[idx_to_use]
    indexed_upper = upper_ci[idx_to_use]

    return (structured=(mean=indexed_mean, lower=indexed_lower, upper=indexed_upper),)
end
