"""
    NNGP <: ComponentModel

The Nearest Neighbor Gaussian Process (NNGP) component for scalable continuous geostatistics
and point-referenced spatial modeling on large datasets (Datta et al., 2016).

# Version
v1.0.0

# Mathematical Summary
For a continuous Gaussian process \$w(s) \\sim \\mathcal{GP}(0, C(s, s'))\$ observed at
  \$N\$ locations
\$\\mathbf{S} = \\{s_1, \\dots, s_N\\}\$, NNGP approximates the joint multivariate normal
  distribution
by conditioning each point only on its \$m\$ nearest neighbors among the preceding locations
  in a spatial ordering:

\$p(w_1, \\dots, w_N) \\approx p(w_1) \\prod_{i=2}^N p(w_i \\mid \\mathbf{w}_{N(s_i)})\$

where \$N(s_i) \\subset \\{s_1, \\dots, s_{i-1}\\}\$ with size \$\\min(i-1, m)\$.

The conditional distribution is Gaussian:
\$w_i \\mid \\mathbf{w}_{N(s_i)} \\sim \\mathcal{N}\\left(\\mathbf{B}_i
  \\mathbf{w}_{N(s_i)}, F_i\\right)\$

where the sparse Kriging weights \$\\mathbf{B}_i\$ and conditional variance \$F_i\$ are:
\$\\mathbf{B}_i = C(s_i, N(s_i)) \\left[ C(N(s_i), N(s_i)) + \\epsilon \\mathbf{I} \\right]^{-1}\$
\$F_i = C(s_i, s_i) - \\mathbf{B}_i C(N(s_i), s_i)\$

This induces a sparse lower-triangular Cholesky factor \$(\\mathbf{I} -
  \\mathbf{B})\\mathbf{w} = \\mathbf{F}^{1/2} \\mathbf{u}\$,
allowing the latent spatial field to be sampled via fast forward substitution in
  \$\\mathcal{O}(N m^3)\$ time
and \$\\mathcal{O}(N m)\$ memory, scaling to \$N > 10^5\$ without inducing points or meshes.

# Computational Methods
- `:sequential` (Default, AD-friendly): Forward-substitution recursion computing neighbor
  solves dynamically.
- `:ordered` (AD-friendly): Pre-ordered coordinate traversal.

# Inputs
- **Required**:
  - Spatial coordinates in data: `:s_x` and `:s_y` (or specified via `coord_vars=[:lon, :lat]`).
- **Optional**:
  - `sigma`: `UnivariateDistribution`, prior for marginal standard deviation. Default:
    `Exponential(1.0)`.
  - `lengthscale`: `UnivariateDistribution`, prior for spatial correlation lengthscale.
    Default: `LogNormal(0.0, 1.0)`.
  - `m`: `Int`, number of nearest neighbors (default: 10).
  - `kernel`: `Symbol`, covariance kernel (`:exponential`, `:matern32`, `:matern52`, `:se`).
    Default: `:exponential`.
  - `order`: `Symbol`, spatial ordering scheme (`:x`, `:y`, `:sum`, `:none`). Default: `:x`.

# Outputs (Parameter Names)
- `sigma_<key>`: Marginal spatial standard deviation.
- `ls_<key>`: Spatial correlation lengthscale.
- `ure_<key>`: Standard normal innovations vector (\$N\$ elements).
- `sre_<key>`: Realized NNGP spatial field.

# Key References
- Datta, A., Banerjee, S., Finley, A. O., & Gelfand, A. E. (2016). *Hierarchical
  nearest-neighbor Gaussian process models for large geostatistical datasets*. Journal of
  the American Statistical Association, 111(514), 800-812.
- Finley, A. O., Datta, A., Cook, B. D., & Banerjee, S. (2019). *spNNGP: Spatial Regression
  Models for Large Datasets using Nearest Neighbor Gaussian Processes*. Journal of
  Statistical Software, 90(4), 1-26.
"""
struct NNGP <: ComponentModel
    sigma::UnivariateDistribution
    lengthscale::UnivariateDistribution
    m::Int
    kernel::Symbol
    order::Symbol
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:nngp] = NNGP

COMPONENT_CONSTRUCTORS[:nngp] = (p, params) -> NNGP(
    get(p, :sigma, Exponential(1.0)),
    get(p, :lengthscale, get(p, :ls, LogNormal(0.0, 1.0))),
    get(params, :m, 10),
    get(params, :kernel, :exponential),
    get(params, :order, :x),
    get(params, :method, :sequential)
)

MODEL_TO_STRUCTURE_MAP[:nngp] = :spatial

function get_precomputes(m::NNGP, M::NamedTuple, mod_data::Dict)::NamedTuple
    data = M.data
    params = mod_data[:params]
    
    # 1. Extract 2D Spatial Coordinates
    x_col = haskey(params, :x) ? params[:x] : (hasproperty(data,
        :s_x) ? :s_x : (hasproperty(data, :lon) ? :lon : (hasproperty(data,
        :x) ? :x : nothing)))
    y_col = haskey(params, :y) ? params[:y] : (hasproperty(data,
        :s_y) ? :s_y : (hasproperty(data, :lat) ? :lat : (hasproperty(data,
        :y) ? :y : nothing)))
    
    if isnothing(x_col) || isnothing(y_col)
        error("NNGP component '$(mod_data[:key])' requires 2D coordinates (:s_x, :s_y) or (:lon, :lat) in dataset.")
    end

    raw_x = Float64.(data[!, x_col])
    raw_y = Float64.(data[!, y_col])
    N = length(raw_x)

    # 2. Determine Spatial Ordering
    order_idx = if m.order == :x
        sortperm(raw_x)
    elseif m.order == :y
        sortperm(raw_y)
    elseif m.order == :sum
        sortperm(raw_x .+ raw_y)
    else
        collect(1:N)
    end

    inv_order = invperm(order_idx)
    sorted_x = raw_x[order_idx]
    sorted_y = raw_y[order_idx]

    # 3. Find m-Nearest Neighbors among preceding locations
    m_val = min(m.m, N - 1)
    nn_indices = Vector{Vector{Int}}(undef, N)
    nn_dists_target = Vector{Vector{Float64}}(undef, N)
    nn_dists_mat = Vector{Matrix{Float64}}(undef, N)

    nn_indices[1] = Int[]
    nn_dists_target[1] = Float64[]
    nn_dists_mat[1] = Matrix{Float64}(undef, 0, 0)

    for i in 2:N
        n_candidates = i - 1
        cand_x = view(sorted_x, 1:n_candidates)
        cand_y = view(sorted_y, 1:n_candidates)

        # Distances to all preceding points
        dists = sqrt.((cand_x .- sorted_x[i]).^2 .+ (cand_y .- sorted_y[i]).^2)
        
        k_neighbors = min(m_val, n_candidates)
        nearest_p = partialsortperm(dists, 1:k_neighbors)
        
        nn_indices[i] = nearest_p
        nn_dists_target[i] = dists[nearest_p]
        
        # Inter-neighbor distance matrix (k_neighbors x k_neighbors)
        sub_x = cand_x[nearest_p]
        sub_y = cand_y[nearest_p]
        d_mat = zeros(Float64, k_neighbors, k_neighbors)
        for r in 1:k_neighbors, c in 1:k_neighbors
            d_mat[r, c] = sqrt((sub_x[r] - sub_x[c])^2 + (sub_y[r] - sub_y[c])^2)
        end
        nn_dists_mat[i] = d_mat
    end

    return (
        N = N,
        m = m_val,
        order_idx = order_idx,
        inv_order = inv_order,
        nn_indices = nn_indices,
        nn_dists_target = nn_dists_target,
        nn_dists_mat = nn_dists_mat,
        kernel = m.kernel,
        n_latent = N
    )
end

function get_priors(
    m::NNGP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent

    return """
    $(p_names.sigma) ~ $(_distribution_to_string(m.sigma))
    $(p_names.ls) ~ $(_distribution_to_string(m.lengthscale))
    $(p_names.ure) ~ MvNormal(zeros(T, $(n_latent)), I)
    """
end

function get_updates(
    m::NNGP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    hyper = spec.hyper
    N = hyper.N
    kernel = hyper.kernel

    # Kernel function inline generator
    cov_expr = if kernel == :exponential
        "exp.(-dists ./ max(ls_val, 1e-4))"
    elseif kernel == :se
        "exp.(-0.5 .* (dists ./ max(ls_val, 1e-4)).^2)"
    elseif kernel == :matern32
        "let r = sqrt(3.0) .* dists ./ max(ls_val, 1e-4); (1.0 .+ r) .* exp.(-r); end"
    elseif kernel == :matern52
        "let r = sqrt(5.0) .* dists ./ max(ls_val, 1e-4); (1.0 .+ r .+ (r.^2)./3.0) .* exp.(-r); end"
    else
        "exp.(-dists ./ max(ls_val, 1e-4))"
    end

    return """
    # --- Nearest Neighbor Gaussian Process (NNGP): $(key) ---
    $(p_names.sre) = let
        N_nngp = $(N)
        sig = $(p_names.sigma)
        ls_val = $(p_names.ls)
        u_raw = $(p_names.ure)
        
        nn_idx = spec_registry[:$(key)].hyper.nn_indices
        nn_d_tgt = spec_registry[:$(key)].hyper.nn_dists_target
        nn_d_mat = spec_registry[:$(key)].hyper.nn_dists_mat
        inv_ord = spec_registry[:$(key)].hyper.inv_order

        T_elem = eltype(sig)
        w_sorted = zeros(T_elem, N_nngp)
        
        # Point 1 (unconditional base)
        w_sorted[1] = sig * u_raw[1]

        # Sequential m-nearest neighbor conditional solves
        for i in 2:N_nngp
            dists_tgt = nn_d_tgt[i]
            dists_mat = nn_d_mat[i]
            k_n = length(dists_tgt)
            
            # Cross-covariance C(s_i, N(s_i))
            dists = dists_tgt
            c_vec = $(cov_expr)
            
            # Neighbor covariance C(N(s_i), N(s_i))
            dists = dists_mat
            C_nn = $(cov_expr)
            
            # B_i = c_vec' * inv(C_nn + 1e-6*I)
            C_nn_reg = C_nn + (1e-5 * I(k_n))
            B_i = C_nn_reg \\ c_vec
            
            # Conditional variance F_i = sig^2 * (1 - B_i' * c_vec)
            F_i = max(sig^2 * (1.0 - dot(B_i, c_vec)), 1e-8)
            
            # Latent neighbor history
            w_neighbors = w_sorted[nn_idx[i]]
            mu_cond = dot(B_i, w_neighbors)
            
            w_sorted[i] = mu_cond + sqrt(F_i) * u_raw[i]
        end

        # Map back to original dataset ordering
        w_sorted[inv_ord]
    end

    $(eta_target) = $(eta_target) .+ $(p_names.sre)
    """
end

function get_effects(
    m::NNGP, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    v = generate_full_variable_names(spec, M.model_arch, 1)
    sig_samples = get_param_samples(chain, M.param_registry, Symbol(v.sigma))
    ls_samples = get_param_samples(chain, M.param_registry, Symbol(v.ls))
    ure_samples = get_param_samples(chain, M.param_registry, Symbol(v.ure))

    N = spec.hyper.N
    effect_matrix = zeros(Float64, N, n_samples)
    nn_idx = spec.hyper.nn_indices
    nn_d_tgt = spec.hyper.nn_dists_target
    nn_d_mat = spec.hyper.nn_dists_mat
    inv_ord = spec.hyper.inv_order
    kernel = spec.hyper.kernel

    for s in 1:n_samples
        sig = sig_samples[s]
        ls_val = ls_samples[s]
        u_raw = ure_samples[:, s]

        w_s = zeros(Float64, N)
        w_s[1] = sig * u_raw[1]

        for i in 2:N
            dists_tgt = nn_d_tgt[i]
            dists_mat = nn_d_mat[i]
            k_n = length(dists_tgt)

            c_vec = if kernel == :se
                exp.(-0.5 .* (dists_tgt ./ max(ls_val, 1e-4)).^2)
            else
                exp.(-dists_tgt ./ max(ls_val, 1e-4))
            end

            C_nn = if kernel == :se
                exp.(-0.5 .* (dists_mat ./ max(ls_val, 1e-4)).^2)
            else
                exp.(-dists_mat ./ max(ls_val, 1e-4))
            end

            C_nn_reg = C_nn + (1e-5 * I(k_n))
            B_i = C_nn_reg \ c_vec
            F_i = max(sig^2 * (1.0 - dot(B_i, c_vec)), 1e-8)
            
            mu_cond = dot(B_i, w_s[nn_idx[i]])
            w_s[i] = mu_cond + sqrt(F_i) * u_raw[i]
        end

        effect_matrix[:, s] = w_s[inv_ord]
    end

    return (structured=effect_matrix, noisy=effect_matrix)
end
