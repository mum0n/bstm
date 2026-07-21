
    

    
function generate_sim_data(s_N=25, t_N=10; rndseed=42)
    # Purpose: A utility function for generating a standardized simulated spatiotemporal dataset with 
    #          known underlying trends, seasonal effects, and covariate relationships. 
    # v1.2.1 (2026-07-16)
    # Inputs: s_N (number of spatial units), t_N (number of time units), rndseed.
    # Outputs: A NamedTuple containing the simulated DataFrame and metadata.
    # Note: This function is crucial for testing and validating model implementations.
    Random.seed!(rndseed)
    n_total = s_N * t_N

    # 1. Spatial Coordinates (Unit Level)
    unique_pts = [(rand() * 100.0, rand() * 100.0) for _ in 1:s_N]
    s_coord_tuple = repeat(unique_pts, inner=t_N)
    s_x = getindex.(s_coord_tuple, 1)
    s_y = getindex.(s_coord_tuple, 2)

    # 2. Temporal/Seasonal Indices
    t_v = repeat(collect(1:t_N), outer=s_N) .+ (rand(n_total) .* 0.05)
    t_idx = repeat(1:t_N, outer=s_N)
    u_N = 12
    u_idx = mod1.(1:n_total, u_N)

    # 3. Latent Fields
    period = 12.0
    trend = 0.05 .* t_v
    seasonal = 0.8 .* cos.(2π .* t_v ./ period)

        # Covariate Generation (W1, W2, W3)
    # These simulate continuous predictors with some shared latent signal Z
    Z = randn(n_total)
    W1_obs = 0.5 .* sin.(t_v ./ 5.0) .+ 0.5 .* Z .+ (randn(n_total) .* 0.1)
    W2_obs = 0.5 .* cos.(t_v ./ 5.0) .- 0.3 .* Z .+ (randn(n_total) .* 0.2)
    W3_obs = 0.2 .* (t_v ./ t_N) .+ 0.1 .* Z .+ (randn(n_total) .* 0.3)

    # Mosaic/Cluster Effects
    s_clusters = mod1.(1:s_N, 5)
    cluster_effects = [-2.5, -1.0, 0.0, 1.0, 2.5]
    cluster_assignments_full = repeat(s_clusters, inner=t_N)
    spatial_effect = cluster_effects[cluster_assignments_full]

    # 4. Response Construction
    sigma_y = 0.15
    observation_error = sigma_y .* randn(n_total)
    eta = 1.0 .+ spatial_effect .+ trend .+ seasonal .+ observation_error

    y_binary = [v > mean(eta) ? 1 : 0 for v in eta]
    # Sample from a Poisson distribution for more realistic count data
    y_counts = [rand(Poisson(exp(v))) for v in eta]

    weights = ones(Float64, n_total)
    trials = ones(Int, n_total)

    # Fixed Effects Design Matrix (Standard Intercept-only approach)
    Xfixed = ones(Float64, n_total, 1)

    # a factorial variable
    reg_indices = mod1.(1:n_total, 4)
    reg_levels = ["North", "South", "East", "West"]
    reg = reg_levels[reg_indices]

    # reformat simulated data into a rectangular dataframe (or namedarray the internal default):
    data_df = DataFrame(
        y = y_counts,  # y-variable
        y_obs = eta,
        # Ensuring coordinates are aligned with the flattened observation vector
        s_idx = repeat(1:s_N, inner=t_N),
        s_coord = s_coord_tuple,
        s_x = s_x,
        s_y = s_y,
        t_v = t_v,
        t_coord = vec(t_idx),   # time index 
        u_idx = u_idx,
        u_v = seasonal,
        log_offsets = zeros(n_total),
        region = categorical(reg),  # would make sure it is used as a factorial variable or in the model statement: Fixed(reg)
        z = Z,  # continuous covariate
        w1 = W1_obs, # more covariates 
        w2 = W2_obs,
        w3 = W3_obs,
        cluster_assignments = cluster_assignments_full,

        y_binary = y_binary,
        y_counts = y_counts,

        Xfixed = Xfixed,
        weights = weights,
        trials = trials     
    )

    return (
        data_df = data_df,
        s_coord_tuple = s_coord_tuple, # Corrected key to match expected usage
        metadata = (
            s_N = s_N,
            t_N = t_N,
            u_N = u_N,
            n_total = n_total 
        )
    )   

end




function scottish_lip_cancer_data_spacetime(n_years::Int=10, spatial_expansion::Float64=1.5, temporal_expansion::Float64=1.5; rndseed::Int=42, recreate::Bool=false)
    # Purpose: A data factory that generates a spatiotemporal version of the classic Scottish Lip 
    #          Cancer dataset. It also creates an expanded "nested" dataset for testing 
    #          multi-fidelity models.
    # v1.2.1 (2026-07-16)
    # Inputs: n_years, spatial_expansion, temporal_expansion, rndseed, recreate.
    # Outputs: A tuple containing the primary and nested datasets.
    # Rationale: Resolving symmetry errors in adjacency matrix construction and scoping errors for derived variables.

    cache_path = "data/scottish_lip_cancer_cache.jld2"

    # Check for existing cache and bypass logic unless recreate is explicitly true
    if isfile(cache_path) && !recreate
        println("Loading cached dataset from: ", cache_path)
        data_bundle = JLD2.load(cache_path)
        return (data_bundle["primary"], data_bundle["nested"])
    end

    println("Generating new spatiotemporal dataset...")
    Random.seed!(rndseed)

    # ##########################################################################
    # PRIMARY DATASET CONSTRUCTION (56 Districts)
    # scottish lip cancer data to a space-time version
    # ##########################################################################

    n_districts = 56

    # Canonical neighbor list (undirected counties)
    neighbor_list = [
        [5, 9, 11, 19], [7, 10], [6, 12], [18, 20, 28], [1, 11, 12, 13, 19],
        [3, 8], [2, 10, 13, 16, 17], [6], [1, 11, 17, 19, 23, 29], [2, 7, 16, 22],
        [1, 5, 9, 12], [3, 5, 11], [5, 7, 17, 19], [31, 32, 35], [25, 29, 50],
        [7, 10, 17, 21, 22, 29], [7, 9, 13, 16, 19, 29], [4, 20, 28, 33, 55, 56], [1, 5, 9, 13, 17], [4, 18, 55],
        [16, 29, 50], [10, 16], [9, 29, 34, 36, 37, 39], [27, 30, 31, 44, 47, 48, 55, 56], [15, 26, 29],
        [26, 29, 42, 43], [24, 31, 32, 55], [4, 18, 33, 45], [9, 15, 16, 17, 21, 23, 25, 26, 34, 43, 50], [24, 38, 42, 44, 45, 56],
        [14, 24, 27, 32, 35, 46, 47], [14, 27, 31, 35], [18, 28, 45, 56], [23, 29, 39, 40, 42, 43, 51, 52, 54], [14, 31, 32, 37, 46],
        [23, 37, 39, 41], [23, 35, 36, 41, 46], [30, 42, 44, 49, 51, 54], [23, 34, 36, 40, 41], [34, 39, 41, 49, 52],
        [36, 37, 39, 40, 46, 49, 53], [26, 30, 34, 38, 43, 51], [26, 29, 34, 42], [24, 30, 38, 48, 49], [28, 30, 33, 56],
        [31, 35, 37, 41, 47, 53], [24, 31, 46, 48, 49, 53], [24, 44, 47, 49], [38, 40, 41, 44, 47, 48, 52, 53, 54], [15, 21, 29],
        [34, 38, 42, 54], [34, 40, 49, 54], [41, 46, 47, 49], [34, 38, 49, 51, 52], [18, 20, 24, 27, 56], [18, 24, 30, 33, 45, 55]
    ]

    # Construct and enforce symmetry for adjacency matrix W
    W_raw = spzeros(Int, n_districts, n_districts)
    for i in 1:n_districts
        for nb in neighbor_list[i]
            W_raw[i, nb] = 1
        end
    end
    # Symmetric enforcement: W_{ij} = W_{ji}
    W = sparse(Symmetric(Matrix(W_raw + W_raw')) .> 0)

    # Inferred spatial geometry using force-directed layout
    au_primary = assign_spatial_units_inferred(W)
    p_centroids = au_primary.centroids
    p_hull = au_primary.hull_coords

    # Clayton & Kaldor Reference Values
    y_orig = [9,39,11,9,15,8,26,7,6,20,13,5,3,8,17,9,2,7,9,7,16,31,11,7,19,15,7,10,16,11,5,3,7,8,11,9,11,8,6,4,10,8,2,6,19,3,2,3,28,6,1,1,1,1,0,0]
    E_orig = [1.4,8.7,3.0,2.5,4.3,2.4,8.1,2.3,2.0,6.6,4.4,1.8,1.1,3.3,7.8,4.6,1.1,4.2,5.5,4.4,10.5,22.7,8.8,5.6,15.5,12.5,6.0,9.0,14.4,10.2,4.8,2.9,7.0,8.5,12.3,10.1,12.7,9.4,7.2,5.3,18.8,15.8,4.3,14.6,50.7,8.2,5.6,9.3,88.7,19.6,3.4,3.6,5.7,7.0,4.2,1.8]
    x_orig = [16,16,10,24,10,24,10,7,7,16,7,16,10,24,7,16,10,7,7,10,7,16,10,7,1,1,7,7,10,10,7,24,10,7,7,0,10,1,16,0,1,16,16,0,1,7,1,1,0,1,1,0,1,1,16,10]

    data_primary = DataFrame()
    for i in 1:n_districts
        log_off = log.(fill(E_orig[i], n_years))
        innov = cumsum(randn(n_years) .* 0.1)
        y_p = floor.(Int, abs.(fill(y_orig[i], n_years) .+ (innov .* 4.0)))

        d_df = DataFrame(
            district = i,
            year = 1:n_years,
            y = y_p,
            log_offsets = log_off,
            cov1 = fill(x_orig[i], n_years)
        )
        # Calculate rate within block to avoid scoping errors
        d_df.y_rate = d_df.y ./ exp.(d_df.log_offsets)
        append!(data_primary, d_df)
    end

    # Assign binary response based on grand mean rate
    data_primary.y_bin = [v > mean(data_primary.y_rate) ? 1 : 0 for v in data_primary.y_rate]

    # Generate correlated covariates
    data_primary.cov2 = 0.5 .* data_primary.cov1 .+ randn(nrow(data_primary))
    # Correcting scoping by using the data frame columns
    data_primary.cov3 = randn(nrow(data_primary)) .* (data_primary.y_rate .^ 2)
    data_primary.cov4 = randn(nrow(data_primary)) .* log.(data_primary.y_rate .+ 1.0)
    data_primary.cov5 = randn(nrow(data_primary)) .* exp.(data_primary.y_rate) .* 2.0
    data_primary.cov6 = randn(nrow(data_primary))
    
    data_primary.day = rand(1:365,size(data_primary, 1)) 
    data_primary.month = Int.(round.(data_primary.day ./365 * 12)) .+ 1

    data_primary.f1 = rand(["A", "B"], nrow(data_primary))
    data_primary.s_idx = data_primary.district
    data_primary.s_x =  [c[1] for c in p_centroids[data_primary.s_idx]]
    data_primary.s_y =  [c[2] for c in p_centroids[data_primary.s_idx]]
    
    n_total = length(data_primary.y_bin)

    reg_indices = mod1.(1:n_total, 4)
    reg_levels = ["North", "South", "East", "West"]
    reg = reg_levels[reg_indices]

    data_primary.region = categorical(reg)  # as a "factor"

    au_primary = merge( au_primary, (
        s_idx = data_primary.s_idx,
        s_x = [c[1] for c in p_centroids[data_primary.s_idx]],
        s_y = [c[2] for c in p_centroids[data_primary.s_idx]],
        s_vals = collect(1:n_districts)
    ))

    # ##########################################################################
    # NESTED DATASET CONSTRUCTION (User-Controlled Expansion)
    # ##########################################################################

    # Spatial domain boundary from primary centroids
    px = [c[1] for c in p_centroids]
    py = [c[2] for c in p_centroids]
    x_min, x_max = minimum(px), maximum(px)
    y_min, y_max = minimum(py), maximum(py)
    x_rng, y_rng = x_max - x_min, y_max - y_min

    # Expansion buffer calculations
    s_buff = (spatial_expansion - 1.0) / 2.0
    nx_min, nx_max = x_min - s_buff * x_rng, x_max + s_buff * x_rng
    ny_min, ny_max = y_min - s_buff * y_rng, y_max + s_buff * y_rng

    nt_max = Int(round(n_years * temporal_expansion))
    n_obs_nested = Int(round(nrow(data_primary) * spatial_expansion * temporal_expansion))

    sx_nested = rand(Uniform(nx_min, nx_max), n_obs_nested)
    sy_nested = rand(Uniform(ny_min, ny_max), n_obs_nested)
    time_nested = rand(1:nt_max, n_obs_nested)

    # Spatial Unit Assignment for expanded domain
    au_nested = assign_spatial_units(sx_nested, sy_nested; target_units=100)

    data_nested = DataFrame(
        s_x = sx_nested,
        s_y = sy_nested,
        year = time_nested,
        district = au_nested.s_idx
    )

    # Latent signal generation for nested grid
    s_lat_n = cumsum(randn(length(au_nested.centroids))) .* 0.3
    t_lat_n = sin.(collect(1:nt_max) .* (2π/nt_max))

    eta_n = [1.5 + s_lat_n[data_nested.district[i]] + t_lat_n[data_nested.year[i]] for i in 1:n_obs_nested]

    data_nested.y = [rand(Poisson(exp(v))) for v in eta_n]
    data_nested.y_rate = exp.(eta_n) .+ randn(n_obs_nested) .* 0.2
    data_nested.y_bin = [v > mean(data_nested.y_rate) ? 1 : 0 for v in data_nested.y_rate]

    data_nested.ncov1 = 0.6 .* eta_n .+ randn(n_obs_nested)
    data_nested.ncov2 = randn(n_obs_nested) .* exp.(data_nested.y_rate)
    data_nested.ncov3 = randn(n_obs_nested)

    primary_out = (data=data_primary, au=au_primary )

    nested_out = (data=data_nested, au=au_nested ) # Return the full au_nested object for consistency

    # Directory check and caching
    if !isdir("data"); mkdir("data"); end
    JLD2.save(cache_path, "primary", primary_out, "nested", nested_out)
    println("Dataset successfully cached at: ", cache_path)

    return (primary_out, nested_out)
    # (p_set, n_set) = scottish_lip_cancer_data_spacetime();
end
