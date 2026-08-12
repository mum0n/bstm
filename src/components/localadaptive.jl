"""
    LocalAdaptive <: ComponentModel

A component for a Local Adaptive spatial effect. This model combines a global
smoothing structure (based on a Leroux-style precision matrix) with local,
cluster-specific mean effects. This allows the model to capture both smooth spatial
trends and abrupt shifts between distinct spatial regions.

# Version
v1.1.0 (2026-08-11)

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

# Computational Methods
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the ICAR precision matrix. Recommended for gradient-based samplers.
- `:cholesky` (AD-friendly): Uses a pre-computed dense Cholesky factorization of the
  full Leroux precision matrix.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization,
  which is not compatible with most AD backends.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@bstm`.
  - Spatial coordinates (`s_x`, `s_y`) in the data frame for clustering.
- **Optional (in `random()` call)**:
  - `n_clusters`: `Int`, the number of spatial clusters to identify. Default: `5`.
  - `rho`: A `UnivariateDistribution` for the prior on the mixing parameter. Default: `Beta(1,1)`.
  - `sigma`: A `UnivariateDistribution` for the prior on the overall standard deviation. Default: `Exponential(1.0)`.
  - `method`: A `Symbol` specifying the computational method. Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The overall marginal standard deviation.
- `rho_<key>`: The mixing parameter.
- `innovations_<key>`: The raw standard normal innovations for the centered spatial effect.
- `cluster_innovations_<key>`: The raw standard normal innovations for the cluster means.
- `latent_<key>`: The reconstructed latent spatial field.

# Key References
- Gelfand, A. E., Schmidt, A. M., Banerjee, S., & Sirmans, C. F. (2005).
  *Nonstationary multivariate process modeling through spatially varying
  coregionalization*. Test, 14(2), 263-312.
"""
struct LocalAdaptive <: ComponentModel
    rho::Distribution
    sigma::Distribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:localadaptive] = LocalAdaptive
COMPONENT_CONSTRUCTORS[:localadaptive] = (p, params) -> LocalAdaptive(
    p.rho, p.sigma, get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:localadaptive] = :spatial

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

    F = cholesky(Symmetric(Matrix(Q_template_scaled) + M.noise * I))

    return (
        Q_template=Q_template_scaled, 
        scaling_factor=scaling_factor, 
        U=U, 
        L=L_scaled, 
        n_latent=n,
        n_clusters=n_clusters,
        cluster_assignments=cluster_assignments,
        cholesky_factor=F
    )
end

function get_priors(
    m::LocalAdaptive, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    rho_prior_str = _distribution_to_string(m.rho)
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    n_clusters = spec.hyper.n_clusters
    
    return """
        $(p_names.rho) ~ $(rho_prior_str)
        $(p_names.sigma) ~ $(sigma_prior_str)
        $(p_names.innovations) ~ MvNormal(zeros(T, $(spec.hyper.n_latent)), I)
        $(p_names.cluster_innovations) ~ MvNormal(zeros(T, $(n_clusters)), I)
    """
end

function get_updates(
    m::LocalAdaptive, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    # The `hyper` variable is removed as it's not available in the Turing model scope.
    # All pre-computed data must be accessed via `spec_registry`.
    n_clusters = spec.hyper.n_clusters
    # Correctly access cluster_assignments through the spec_registry at runtime.
    cluster_assignments_access = "spec_registry[:$(key)].hyper.cluster_assignments"

    spectral_code = """
        # --- LocalAdaptive Component (Spectral): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper # Access precomputed data
            mu_clusters_raw = $(p_names.cluster_innovations)
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_clusters)), sum(mu_clusters_raw))
            mean_vector = mu_clusters_raw[$(cluster_assignments_access)]

            diag_D = $(p_names.sigma) ./ sqrt.((1.0 .- $(p_names.rho)) .+ $(p_names.rho) .* hyper.L .+ M.noise)
            latent_centered = hyper.U * (diag_D .* $(p_names.innovations))
            
            $(p_names.latent) = mean_vector .+ latent_centered
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """

    cholesky_code = """
        # --- LocalAdaptive Component (Cholesky): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper # Access precomputed data
            mu_clusters_raw = $(p_names.cluster_innovations)
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_clusters)), sum(mu_clusters_raw))
            mean_vector = mu_clusters_raw[$(cluster_assignments_access)]

            Q_template = hyper.Q_template
            rho_val = $(p_names.rho)
            Q_final = (1.0 - rho_val) .* I(size(Q_template, 1)) .+ rho_val .* Q_template
            
            F = cholesky(Symmetric(Matrix(Q_final) + M.noise * I))
            
            latent_centered = $(p_names.sigma) .* (F.U \\ $(p_names.innovations))
            
            $(p_names.latent) = mean_vector .+ latent_centered
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """

    cholesky_sparse_code = """
        # --- LocalAdaptive Component (Sparse Cholesky): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper # Access precomputed data
            mu_clusters_raw = $(p_names.cluster_innovations)
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_clusters)), sum(mu_clusters_raw))
            mean_vector = mu_clusters_raw[$(cluster_assignments_access)]

            Q_template = hyper.Q_template
            rho_val = $(p_names.rho)
            Q_final = (1.0 - rho_val) .* sparse(I, size(Q_template, 1), size(Q_template, 1)) .+ rho_val .* Q_template
            
            F = cholesky(Symmetric(Q_final + M.noise * I))
            
            latent_centered = $(p_names.sigma) .* (F.U \\ $(p_names.innovations))
            
            $(p_names.latent) = mean_vector .+ latent_centered
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
        error("Unsupported method '$(m.method)' for LocalAdaptive component.")
    end
end


function get_effects(
    m::LocalAdaptive, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    hyper = spec.hyper
    n_latent = hyper.n_latent
    cluster_assignments = hyper.cluster_assignments
    noise = M.noise
    n_clusters = hyper.n_clusters

    s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))

    for k in 1:outcomes_N
        sigma_samples_name = _find_parameter(p_names_vec, string(spec.key), "sigma", k, is_multivariate_model)
        rho_samples_name = _find_parameter(p_names_vec, string(spec.key), "rho", k, is_multivariate_model)
        innovations_samples_name = _find_parameter(p_names_vec, string(spec.key), "innovations", k, is_multivariate_model)
        cluster_innovations_samples_name = _find_parameter(p_names_vec, string(spec.key), "cluster_innovations", k, is_multivariate_model)

        if isempty(sigma_samples_name) || isempty(rho_samples_name) || isempty(innovations_samples_name) || isempty(cluster_innovations_samples_name)
            @warn "Parameters for LocalAdaptive component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_samples_name, 1)[:, 1]
        rho_samples = get_params_vector(chain, rho_samples_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innovations_samples_name, n_latent)
        cluster_innov_samples = get_params_vector(chain, cluster_innovations_samples_name, n_clusters)

        effect_k = zeros(Float64, N_total, n_samples)

        for i in 1:n_samples
            mu_clusters_centered = cluster_innov_samples[i, :] .- mean(cluster_innov_samples[i, :])
            mean_vector = mu_clusters_centered[cluster_assignments]
            
            sigma_s = sigma_samples[i]
            rho_s = rho_samples[i]
            innov_s = innovations_samples[i, :]

            local latent_centered
            if m.method == :spectral
                U = hyper.U
                L_eig = hyper.L
                diag_D_s = sigma_s ./ sqrt.((1.0 - rho_s) .+ rho_s .* L_eig .+ noise)
                latent_centered = U * (diag_D_s .* innov_s)
            else # :cholesky or :cholesky_sparse
                Q_template = hyper.Q_template
                Q_final = (1.0 - rho_s) .* sparse(I, n_latent, n_latent) .+ rho_s .* Q_template
                F = cholesky(Symmetric(Matrix(Q_final) + noise * I))
                latent_centered = sigma_s .* (F.U \ innov_s)
            end
            
            final_latent_field = mean_vector .+ latent_centered
            effect_k[:, i] = view(final_latent_field, s_idx_full)
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
