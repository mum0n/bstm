
    

function generate_ordinal_data(;
    n_obs::Int=500, 
    n_groups::Int=10, 
    seed::Int=42
)
    # Purpose: Generates a synthetic dataset for a non-proportional odds ordinal regression model.
    # Rationale: This function simulates data that reflects the complexity of the specified model,
    #            including a mix of proportional and non-proportional covariate effects, and a
    #            random intercept structure. This provides a valid dataset for testing and fitting
    #            the advanced ordinal model.
    # v1.0.0 (2026-07-31)
    # Inputs:
    #   - n_obs: The number of observations to generate.
    #   - n_groups: The number of unique groups for the random intercept.
    #   - seed: A random seed for reproducibility.
    # Outputs: A DataFrame containing the simulated data.

    Random.seed!(seed)

    # 1. Generate covariates and grouping variable
    cov1 = randn(n_obs)
    cov2 = rand(n_obs) .* 2
    cov3 = rand(-2:2, n_obs)
    group_id = rand(1:n_groups, n_obs)

    # 2. Define true underlying parameters for the data generating process
    # Cut-points for a 3-category outcome. These act as intercepts for each cumulative logit/probit.
    alpha_1 = -1.0  # Threshold for P(Y<=1)
    alpha_2 = 1.5   # Threshold for P(Y<=2)

    # Proportional effects (same effect across all categories)
    beta_cov2 = 0.5
    beta_cov3 = -0.4
    
    # Non-proportional effects for cov1 (different effect for each category boundary)
    beta_cov1_cat1 = 1.2  # Effect of cov1 on the logit for P(Y<=1)
    beta_cov1_cat2 = -0.8 # Effect of cov1 on the logit for P(Y<=2)

    # Random intercept parameters
    sigma_random_intercept = 0.7
    random_intercepts = rand(Normal(0, sigma_random_intercept), n_groups)

    # 3. Generate the ordinal outcome
    ordinal_y = Vector{Int}(undef, n_obs)
    
    for i in 1:n_obs
        # Calculate the proportional part of the linear predictor
        eta_proportional = (beta_cov2 * cov2[i]) + (beta_cov3 * cov3[i]) + random_intercepts[group_id[i]]

        # Calculate the linear predictor for each cumulative probability, including the non-proportional effect
        # Note: The model is P(Y<=j) = F(alpha_j - eta_j). We simulate based on this structure.
        linear_pred_1 = alpha_1 - (eta_proportional + cov1[i] * beta_cov1_cat1)
        linear_pred_2 = alpha_2 - (eta_proportional + cov1[i] * beta_cov1_cat2)

        # Apply the inverse link function (Normal CDF for probit link)
        # P(Y <= 1)
        cum_prob_1 = cdf(Normal(), linear_pred_1)
        # P(Y <= 2)
        cum_prob_2 = cdf(Normal(), linear_pred_2)

        # Derive the probability for each category
        # P(Y = 1) = P(Y <= 1)
        prob_1 = cum_prob_1
        # P(Y = 2) = P(Y <= 2) - P(Y <= 1)
        prob_2 = max(0.0, cum_prob_2 - cum_prob_1)
        # P(Y = 3) = 1 - P(Y <= 2)
        prob_3 = max(0.0, 1.0 - cum_prob_2)
        
        # Normalize probabilities to ensure they sum to 1, handling potential floating point inaccuracies
        probs = [prob_1, prob_2, prob_3]
        probs ./= sum(probs)

        # Sample the ordinal outcome from the categorical distribution
        ordinal_y[i] = rand(Categorical(probs))
    end

    # 4. Assemble the DataFrame
    inp_df = DataFrame(
        ordinal_y = ordinal_y,
        cov1 = cov1,
        cov2 = cov2,
        cov3 = cov3,
        group_id = categorical(group_id) # Ensure group_id is a categorical variable
    )

    return inp_df
end

    
function generate_sim_data(s_N=25, t_N=10; rndseed=42)
    # Purpose: A utility function for generating a standardized simulated spatiotemporal dataset.
    # Rationale for Change (v1.1.0):
    # This version removes the use of the ternary operator (`? :`) for generating `y_binary`.
    # This was done to adhere to a coding style preference for more explicit control-flow statements.
    # The logic is now an explicit `for` loop with an `if/else` block, which is more verbose but clearer.
    #
    # Assumptions:
    #   - The output is a standardized DataFrame for testing and examples.
    #
    # Inputs:
    #   - s_N: The number of spatial units.
    #   - t_N: The number of time units.
    #   - rndseed: A random seed for reproducibility.
    #
    # Outputs:
    #   - A DataFrame containing simulated spatiotemporal data.

    Random.seed!(rndseed)
    
    # Spatial setup
    s_coord_tuple = tuple.([(mod(i-1, 5), div(i-1, 5)) for i in 1:s_N]...)
    s_x = [pt[1] for pt in s_coord_tuple]
    s_y = [pt[2] for pt in s_coord_tuple]
    W = adjacency_matrix(s_x, s_y)
    
    # Temporal setup
    t_idx = 1:t_N
    
    # True latent fields
    phi_s = rand(MvNormal(zeros(s_N), I))
    phi_t = sin.(2 * pi .* (t_idx./t_N)) .+ rand(Normal(0,0.1), t_N)
    
    # Spatiotemporal interaction
    phi_st = rand(MvNormal(zeros(s_N*t_N), I))
    
    # Covariates
    s_cov = rand(Normal(0,1), s_N)
    t_cov = rand(Normal(0,1), t_N)
    
    # Data assembly
    n_total = s_N * t_N
    s_coord = repeat(1:s_N, inner=t_N)
    t_coord = repeat(t_idx, outer=s_N)
    s_v = vcat([fill(i, t_N) for i in 1:s_N]...)
    t_v = vcat([t_idx for i in 1:s_N]...)
    
    # Linear predictor
    eta = (
        0.5 .* phi_s[s_v] .+
        0.8 .* phi_t[t_v] .+
        0.2 .* s_cov[s_v] .+
        0.3 .* t_cov[t_v]
    )
    
    # Generate outcomes
    y1 = eta .+ rand(Normal(0, 0.25), n_total) # Gaussian
    y2 = [rand(Poisson(exp(v))) for v in eta] # Poisson
    
    # Binary outcome generation - MODIFIED BLOCK
    y_binary = Vector{Int}(undef, n_total)
    mean_eta = mean(eta)
    for i in 1:n_total
        if eta[i] > mean_eta
            y_binary[i] = 1
        else
            y_binary[i] = 0
        end
    end

    # Multinomial outcome
    eta_mult = hcat(eta, -eta, sin.(eta))
    y_mult = zeros(Int, n_total, 3)
    for i in 1:n_total
        p = softmax(eta_mult[i, :])
        y_mult[i, :] = rand(Multinomial(20, p))
    end
    
    # Proxy data for other models
    proxy_y = rand(Normal(0,1), n_total)
    
    # Create DataFrame
    df = DataFrame(
        s_idx = s_v,
        t_idx = t_v,
        s_x = s_x[s_v],
        s_y = s_y[s_v],
        year = t_v,
        month = mod1.(t_v, 12),
        y_gauss = y1,
        y_pois = y2,
        y_bin = y_binary, # Use the new vector
        y_cat1 = y_mult[:, 1],
        y_cat2 = y_mult[:, 2],
        y_cat3 = y_mult[:, 3],
        proxy_val = proxy_y
    )

    return df
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


function generate_lgcp_synthetic_data_regular(grid_side=15)
    Random.seed!(42)
    s_N = grid_side^2
    
    # Create a spatial grid
    x_coords = repeat(1:grid_side, inner=grid_side)
    y_coords = repeat(1:grid_side, outer=grid_side)
    
    # Generate a smooth latent intensity field Z(s)
    # Simulating a simple spatial trend + noise
    Z_true = [2.0 * sin(x/3.0) * cos(y/3.0) for (x, y) in zip(x_coords, y_coords)]
    
    # Sample Poisson counts (y_obs) per cell
    # lambda = exp(Z)
    y_counts = [rand(Poisson(exp(z))) for z in Z_true]
    
    # Prepare the DataFrame
    # Note: LGCP builder expects y_obs to be pre-aggregated to grid units
    df = DataFrame(
        s_idx = 1:s_N,
        s_x = Float64.(x_coords),
        s_y = Float64.(y_coords),
        counts = y_counts
    )
    
    # Adjacency matrix for the grid (Queen contiguity)
    W = libgeos_lattice_adjacency_matrix(grid_side, grid_side)
    
    return df, W, s_N
end


function generate_irregular_lgcp_data(grid_side=10)
    
# Executable Demonstration - Irregular Spatial Grid
# Rationale: Verifies that the model correctly processes varying areas for intensity estimation.

# # 1. Generate Data on a grid where cell sizes vary
# # Simulate a larger region divided into cells of different sizes

    Random.seed!(99)
    s_N = grid_side^2
    
    x_coords = repeat(1:grid_side, inner=grid_side)
    y_coords = repeat(1:grid_side, outer=grid_side)
    
    # # Heterogeneous Areas: Cells in the center are larger
    areas = [1.0 + 2.0 * exp(-((x - 5)^2 + (y - 5)^2) / 10.0) for (x, y) in zip(x_coords, y_coords)]
    
    # # Smooth latent intensity lambda(s) = 5.0 (constant for this test)
    Z_true = fill(log(5.0), s_N)
    
    # # Observed counts y_s ~ Poisson(lambda_s * Area_s)
    y_counts = [rand(Poisson(exp(z) * a)) for (z, a) in zip(Z_true, areas)]
    
    df = DataFrame(
        s_idx = 1:s_N,
        counts = y_counts,
        cell_area = areas
    )
    
    W = libgeos_lattice_adjacency_matrix(grid_side, grid_side)
    return df, W, s_N, areas
end




function prepare_advanced_bstm_data()
    Random.seed!(123)
    n_s = 30
    n_t = 12
    n_obs = n_s * n_t

    # # Spatial and Temporal coordinates
    unique_coords = [(rand()*10, rand()*10) for _ in 1:n_s]
    s_coords = repeat(unique_coords, inner=n_t)
    s_x = [c[1] for c in s_coords]
    s_y = [c[2] for c in s_coords]
    t_v = repeat(collect(1:n_t), outer=n_s)
    s_idx = repeat(collect(1:n_s), inner=n_t)

    # # Latent signal generation (2D Sine wave + Trend)
    latent_field = [sin(x/2) * cos(y/2) + 0.1*t for (x, y, t) in zip(s_x, s_y, t_v)]

    # # 1. Multivariate Data (Gaussian + Poisson)
    y1 = latent_field .+ randn(n_obs) .* 0.2
    y2 = [rand(Poisson(exp(v))) for v in latent_field]

    # # 2. Multinomial Data (3 Categories)
    eta_mult = hcat(latent_field, -0.5 .* latent_field, zeros(n_obs))
    y_mult = zeros(Int, n_obs, 3)
    for i in 1:n_obs
        p = softmax(eta_mult[i, :])
        y_mult[i, :] = rand(Multinomial(20, p))
    end

    # # 3. Multifidelity Data (Proxy)
    # # Low-fidelity proxy covers same spatial area but with high noise
    proxy_y = latent_field .* 0.8 .+ randn(n_obs) .* 0.5

    df = DataFrame(
        s_idx = s_idx,
        s_x = s_x,
        s_y = s_y,
        year = t_v,
        month = mod1.(t_v, 12),
        y_gauss = y1,
        y_pois = y2,
        y_cat1 = y_mult[:, 1],
        y_cat2 = y_mult[:, 2],
        y_cat3 = y_mult[:, 3],
        proxy_val = proxy_y
    )

    return df
end

 

function generate_logistic_data(; s_N=10, t_N=5, n_obs_per_st_unit=1, seed=123, use_effort::Bool=false, use_removal::Bool=false)
    # Purpose: Generates synthetic data for a logistic population dynamics model, with optional exploitation.
    # Rationale: Consolidates `generate_logistic_basic_data` and `generate_logistic_exploitation_data`
    #            into a single flexible function.
    # v1.0.0 (2026-08-01)
    df, W, grid_areas = create_base_st_data(s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=seed)
    
    # True parameters
    r_true = 0.5
    K_true = 100.0
    q_true = 0.01 # Catchability
    
    # Simulate exploitation data if requested
    local effort_sim, removal_sim
    if use_effort
        effort_sim = rand(s_N, t_N) .* 0.5 .+ 0.1 # Random effort between 0.1 and 0.6
    else
        effort_sim = zeros(s_N, t_N)
    end

    if use_removal
        removal_sim = rand(s_N, t_N) .* 5.0 # Random removal
    else
        removal_sim = zeros(s_N, t_N)
    end

    # Simulate population trajectory
    initial_pop = rand(s_N) * 10.0 .+ 5.0 # Initial population per spatial unit
    y_sim = zeros(s_N, t_N)
    y_sim[:, 1] = initial_pop
    
    for t in 2:t_N
        for s in 1:s_N
            N_prev = y_sim[s, t-1]
            D_prev = N_prev / grid_areas[s] # Convert to density
            K_density = K_true / grid_areas[s] # Carrying capacity in density terms
            growth = r_true * D_prev * (1.0 - D_prev / K_density)
            
            exploitation = 0.0
            if use_effort
                exploitation += q_true * effort_sim[s, t] * N_prev
            end
            if use_removal
                exploitation += removal_sim[s, t]
            end

            y_sim[s, t] = max(0.0, N_prev + growth * grid_areas[s] - exploitation + randn() * 2.0) # Convert growth back to total population, add noise
        end
    end
    
    df.y = repeat(vec(y_sim'), inner=n_obs_per_st_unit)

    if use_effort
        df.effort = repeat(vec(effort_sim'), inner=n_obs_per_st_unit)
    end
    if use_removal
        df.removal = repeat(vec(removal_sim'), inner=n_obs_per_st_unit)
    end

    return df, W, grid_areas
end



function generate_delay_difference_data(; s_N=10, t_N=10, n_obs_per_st_unit=1, seed=123, use_effort::Bool=false, use_removal::Bool=false)
    # Purpose: Generates synthetic data for a multivariate delay-difference model.
    # Rationale: This version is updated to generate `effort` and `removal` covariates,
    #            aligning it with the new model formulation.
    # v1.0.1 (2026-07-31)
    df, W, grid_areas = create_base_st_data(s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=seed)
    
    # True parameters
    r_true = 0.6
    K_true = 150.0
    M_nat_true = 0.2
    sigma_rec_true = 0.2
    sigma_pop_true = 0.1

    # Simulate data
    population_sim = zeros(s_N, t_N)
    recruitment_sim = zeros(s_N, t_N)
    
    local effort_sim, removal_sim
    if use_effort
        q_true = 0.01
        effort_sim = rand(s_N, t_N) .* 10.0
    else
        effort_sim = zeros(s_N, t_N)
    end

    if use_removal
        removal_sim = rand(s_N, t_N) .* 5.0
    else
        removal_sim = zeros(s_N, t_N)
    end

    initial_pop = rand(s_N) .* 20.0 .+ 10.0
    population_sim[:, 1] = initial_pop
    recruitment_sim[:, 1] = initial_pop .* 0.2 # Initial recruitment as a fraction of pop

    for t in 2:t_N
        for s in 1:s_N
            N_prev = population_sim[s, t-1]
            D_prev = N_prev / grid_areas[s]
            K_density = K_true / grid_areas[s]
            
            mean_rec = r_true * D_prev * (1.0 - D_prev / K_density) * grid_areas[s]
            recruitment_sim[s, t] = exp(log(mean_rec + 1e-6) + randn() * sigma_rec_true)
            
            C_prev = 0.0
            if use_effort
                C_prev += q_true * effort_sim[s, t-1] * N_prev
            end
            if use_removal
                C_prev += removal_sim[s, t-1]
            end

            N_survived = (N_prev - C_prev) * exp(-M_nat_true)
            population_sim[s, t] = max(0.0, N_survived + recruitment_sim[s, t] + randn() * sigma_pop_true)
        end
    end
    
    df.y = repeat(vec(population_sim'), inner=n_obs_per_st_unit)
    df.recruitment = repeat(vec(recruitment_sim'), inner=n_obs_per_st_unit)
    
    if use_effort
        df.effort = repeat(vec(effort_sim'), inner=n_obs_per_st_unit)
    end
    if use_removal
        df.removal = repeat(vec(removal_sim'), inner=n_obs_per_st_unit)
    end
    
    return df, W, grid_areas
end



function generate_glv_data(; s_N=10, t_N=10, n_species=3, n_obs_per_st_unit=1, seed=123)
    # Purpose: Generates synthetic data for a multivariate generalized Lotka-Volterra model.
    # Rationale: Provides a test case for the `dynamics(model=generalized_lotka_volterra)` model.
    # v1.0.0 (2026-07-31)
    df, W, grid_areas = create_base_st_data(s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=seed)
    
    # True parameters
    r_true = [0.5, 0.6, 0.7]
    K_true = [100.0, 120.0, 150.0]
    
    # Interaction matrix (alpha): effect of column j on row i
    alpha_true = [1.0 0.5 0.2; 
                  0.3 1.0 0.6; 
                  0.1 0.4 1.0]

    sigma_process_true = [0.1, 0.1, 0.1]

    # Simulate population dynamics for each species
    pop_sim = zeros(s_N, t_N, n_species)
    
    # Initial population distribution
    initial_total_pop = rand(s_N) .* 30.0 .+ 10.0
    for s in 1:s_N
        pop_sim[s, 1, :] = initial_total_pop[s] .* softmax(randn(n_species))
    end

    for t in 2:t_N
        for s in 1:s_N
            N_prev = pop_sim[s, t-1, :]
            D_prev = N_prev ./ grid_areas[s]
            K_density = K_true ./ grid_areas[s]
            
            N_intermediate = zeros(n_species)
            for i in 1:n_species
                interaction_sum_density = dot(alpha_true[i, :], D_prev)
                growth_density = r_true[i] * D_prev[i] * (1.0 - interaction_sum_density / K_density[i])
                N_intermediate[i] = N_prev[i] + growth_density * grid_areas[s]
            end
            
            pop_sim[s, t, :] = max.(0.0, N_intermediate .+ randn(n_species) .* sigma_process_true)
        end
    end
    
    # Add species columns to the DataFrame
    for a in 1:n_species
        species_col_name = Symbol("species_$(a)")
        species_data_flat = vec(pop_sim[:, :, a]')
        df[!, species_col_name] = repeat(species_data_flat, inner=n_obs_per_st_unit)
    end
    
    df.y = df.species_1
    
    return df, W, grid_areas, n_species
end

function generate_lotka_volterra_data(; s_N=10, t_N=5, n_obs_per_st_unit=1, seed=123)
    # Purpose: Generates synthetic data for a Lotka-Volterra prey-predator dynamics model.
    # Rationale: This version corrects a `MethodError` by changing the call to `create_base_st_data`
    #            to use keyword arguments instead of positional arguments, aligning it with the
    #            function's definition.
    # v1.0.1 (2026-07-31)
    df, W, grid_areas = create_base_st_data(s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=seed)
    
    # Simulate coupled prey-predator dynamics
    prey_sim = zeros(s_N, t_N)
    predator_sim = zeros(s_N, t_N)

    initial_prey = rand(s_N) * 10.0 .+ 5.0
    initial_predator = rand(s_N) * 2.0 .+ 1.0
    prey_sim[:, 1] = initial_prey
    predator_sim[:, 1] = initial_predator

    alpha_true = 0.5 # Prey growth rate
    beta_true = 0.01 # Predation rate
    gamma_true = 0.005 # Predator growth from predation
    delta_true = 0.2 # Predator mortality rate
    
    for t in 2:t_N
        for s in 1:s_N
            N_prey_prev = prey_sim[s, t-1]
            N_pred_prev = predator_sim[s, t-1]

            d_prey = alpha_true * N_prey_prev - beta_true * N_prey_prev * N_pred_prev
            d_pred = gamma_true * N_prey_prev * N_pred_prev - delta_true * N_pred_prev
            
            prey_sim[s, t] = max(0.0, N_prey_prev + d_prey + randn() * 0.5)
            predator_sim[s, t] = max(0.0, N_pred_prev + d_pred + randn() * 0.1)
        end
    end
    
    df.y = repeat(vec(prey_sim'), inner=n_obs_per_st_unit) # Prey is the observed outcome
    df.predator_pop = repeat(vec(predator_sim'), inner=n_obs_per_st_unit) # Predator is the interaction covariate
    return df, W, grid_areas
end

function generate_leslie_logistic_data(; s_N=10, t_N=5, n_obs_per_st_unit=1, n_age_classes=3, seed=123)
    # Purpose: Generates synthetic data for a Leslie-Logistic population dynamics model.
    # Rationale: This version corrects a `MethodError` by changing the call to `create_base_st_data`
    #            to use keyword arguments instead of positional arguments, aligning it with the
    #            function's definition.
    # v1.0.1 (2026-07-31)
    df, W, grid_areas = create_base_st_data(s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=seed)
    
    # Simulate population based on Leslie matrix-informed logistic growth
    y_sim = zeros(s_N, t_N)
    initial_pop = rand(s_N) * 10.0 .+ 5.0
    y_sim[:, 1] = initial_pop

    K_true = 100.0 # Carrying capacity
    
    # True Leslie matrix parameters (simplified for simulation)
    survival_true = fill(0.8, n_age_classes - 1) # Survival from age i to i+1
    fecundity_true = [0.0, 1.5, 2.0] # Fecundity for each age class

    # Construct Leslie matrix
    L_true = zeros(n_age_classes, n_age_classes)
    for i in 1:(n_age_classes - 1); L_true[i+1, i] = survival_true[i]; end
    L_true[1, :] = fecundity_true

    # Calculate intrinsic growth rate from dominant eigenvalue
    r_leslie_true = log(maximum(abs.(eigen(L_true).values)))
    
    for t in 2:t_N
        for s in 1:s_N
            N_prev = y_sim[s, t-1]
            D_prev = N_prev / grid_areas[s]
            K_density = K_true / grid_areas[s]
            
            growth = r_leslie_true * D_prev * (1.0 - D_prev / K_density)
            y_sim[s, t] = max(0.0, N_prev + growth * grid_areas[s] + randn() * 2.0)
        end
    end
    
    df.y = repeat(vec(y_sim'), inner=n_obs_per_st_unit)
    return df, W, grid_areas, n_age_classes
end

function create_base_st_data(;
    s_N::Int=10, t_N::Int=5, n_obs_per_st_unit::Int=1, seed::Int=123
)
    Random.seed!(seed)
    s_x = rand(s_N) * 10.0
    s_y = rand(s_N) * 10.0
    W = spzeros(Bool, s_N, s_N)
    for i in 1:s_N
        if i > 1; W[i, i-1] = true; end
        if i < s_N; W[i, i+1] = true; end
    end
    W = max.(W, W')
    grid_areas = rand(s_N) * 5.0 .+ 1.0
    s_idx_flat = repeat(1:s_N, inner=t_N * n_obs_per_st_unit)
    t_idx_flat = repeat(repeat(1:t_N, inner=n_obs_per_st_unit), s_N)
    s_x_flat = repeat(s_x, inner=t_N * n_obs_per_st_unit)
    s_y_flat = repeat(s_y, inner=t_N * n_obs_per_st_unit)
    df = DataFrame(s_idx=s_idx_flat, year=t_idx_flat, s_x=s_x_flat, s_y=s_y_flat, grid_area_col=repeat(grid_areas, inner=t_N * n_obs_per_st_unit))
    return df, W, grid_areas
end


function generate_logistic_spatial_K_data(; s_N=10, t_N=5, n_obs_per_st_unit=1, seed=123)
    # Purpose: Generates synthetic data for a logistic growth model with spatially varying carrying capacity (K).
    # Rationale: Provides a test case for the `dynamics(..., spatially_varying_K=true)` flag.
    # v1.0.0 (2026-07-31)
    df, W, grid_areas = create_base_st_data(s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=seed)
    
    # Simulate a spatially varying K based on the x-coordinate
    s_coords = unique(df[!, [:s_idx, :s_x]])
    sort!(s_coords, :s_idx)
    K_spatial_true = 50.0 .+ 150.0 * (s_coords.s_x ./ maximum(s_coords.s_x)) # K varies from 50 to 200

    y_sim = zeros(s_N, t_N)
    initial_pop = rand(s_N) * 20.0 .+ 10.0
    y_sim[:, 1] = initial_pop

    r_true = 0.6
    
    for t in 2:t_N
        for s in 1:s_N
            N_prev = y_sim[s, t-1]
            D_prev = N_prev / grid_areas[s]
            K_density = K_spatial_true[s] / grid_areas[s]
            growth = r_true * D_prev * (1.0 - D_prev / K_density)
            y_sim[s, t] = max(0.0, N_prev + growth * grid_areas[s] + randn() * 2.5)
        end
    end
    df.y = repeat(vec(y_sim'), inner=n_obs_per_st_unit)
    return df, W, grid_areas
end


function generate_logistic_spatial_r_data(; s_N=10, t_N=5, n_obs_per_st_unit=1, seed=123)
    # Purpose: Generates synthetic data for a logistic growth model with spatially varying growth rate (r).
    # Rationale: Provides a test case for the `dynamics(..., spatially_varying_r=true)` flag.
    # v1.0.0 (2026-07-31)
    df, W, grid_areas = create_base_st_data(s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=seed)
    
    # Simulate a spatially varying r based on the y-coordinate
    s_coords = unique(df[!, [:s_idx, :s_y]])
    sort!(s_coords, :s_idx)
    r_spatial_true = 0.2 .+ 0.8 * (s_coords.s_y ./ maximum(s_coords.s_y)) # r varies from 0.2 to 1.0

    y_sim = zeros(s_N, t_N)
    initial_pop = rand(s_N) * 20.0 .+ 10.0
    y_sim[:, 1] = initial_pop

    K_true = 100.0
    
    for t in 2:t_N
        for s in 1:s_N
            N_prev = y_sim[s, t-1]
            D_prev = N_prev / grid_areas[s]
            K_density = K_true / grid_areas[s]
            growth = r_spatial_true[s] * D_prev * (1.0 - D_prev / K_density)
            y_sim[s, t] = max(0.0, N_prev + growth * grid_areas[s] + randn() * 2.5)
        end
    end
    df.y = repeat(vec(y_sim'), inner=n_obs_per_st_unit)
    return df, W, grid_areas
end




function generate_leslie_matrix_data(; s_N=10, t_N=5, n_age_classes=3, n_obs_per_st_unit=1, seed=123, use_effort::Bool=false, use_removal::Bool=false)
    # Purpose: Generates synthetic data for a multivariate Leslie matrix dynamics model.
    # Rationale: Provides a test case for the `dynamics(model=leslie_matrix)` feature,
    #            now including optional exploitation.
    # v1.0.1 (2026-08-01)
    df, W, grid_areas = create_base_st_data(s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=seed)
    
    # True Leslie matrix parameters
    survival_true = [0.5, 0.8] # Survival for age 1 and 2
    fecundity_true = [0.0, 1.5, 3.0] # Fecundity for each age class

    L_true = zeros(n_age_classes, n_age_classes)
    for i in 1:(n_age_classes - 1); L_true[i+1, i] = survival_true[i]; end
    L_true[1, :] = fecundity_true

    # Exploitation parameters
    q_true = fill(0.005, n_age_classes) # Age-specific catchability
    
    local effort_sim, removal_sim
    if use_effort
        effort_sim = rand(s_N, t_N) .* 10.0
    else
        effort_sim = zeros(s_N, t_N)
    end

    if use_removal
        removal_sim = rand(s_N, t_N, n_age_classes) .* 2.0
    else
        removal_sim = zeros(s_N, t_N, n_age_classes)
    end

    # Simulate population dynamics for each age class
    pop_sim = zeros(s_N, t_N, n_age_classes)
    
    # Initial population distribution
    initial_total_pop = rand(s_N) * 50.0 .+ 20.0
    for s in 1:s_N
        pop_sim[s, 1, :] = initial_total_pop[s] .* softmax(randn(n_age_classes))
    end

    for t in 2:t_N
        for s in 1:s_N
            N_prev = pop_sim[s, t-1, :]
            
            C_prev = zeros(n_age_classes)
            if use_effort
                C_prev .+= q_true .* effort_sim[s, t-1] .* N_prev
            end
            if use_removal
                C_prev .+= removal_sim[s, t-1, :]
            end
            N_after_removal = max.(0.0, N_prev - C_prev)

            N_projected = L_true * N_after_removal
            pop_sim[s, t, :] = max.(0.0, N_projected .+ randn(n_age_classes) .* 0.5)
        end
    end
    
    # Add age class columns to the DataFrame
    for a in 1:n_age_classes
        age_col_name = Symbol("age_$(a)")
        age_data_flat = vec(pop_sim[:, :, a]')
        df[!, age_col_name] = repeat(age_data_flat, inner=n_obs_per_st_unit)
    end
    
    df.y = df.age_1 # Default outcome for univariate likelihoods

    if use_effort
        df.effort = repeat(vec(effort_sim'), inner=n_obs_per_st_unit)
    end
    if use_removal
        # For removal, if it's age-specific, it needs to be stored as a matrix or multiple columns.
        # For simplicity in this data generator, we'll store total removal if only one source.
        # If multiple age-specific removal sources are needed, this would need to be expanded.
        df.removal_total = repeat(vec(sum(removal_sim, dims=3)[:,:,1]'), inner=n_obs_per_st_unit)
    end
    
    return df, W, grid_areas, n_age_classes
end


"""
    generate_dirichlet_multinomial_data(;
        n_obs_per_unit::Int=10,
        n_units::Int=25,
        n_categories::Int=3,
        seed::Int=42
    )

Generates synthetic spatial data suitable for a Dirichlet-Multinomial model.

The function creates a spatial domain with `n_units` and simulates compositional data
(counts across `n_categories`) for `n_obs_per_unit` observations within each unit.
The underlying proportions of the categories vary smoothly across space, controlled
by latent Gaussian Processes.

# Returns
- `DataFrame`: A dataframe containing the simulated data, including category counts,
  spatial coordinates (`s_x`, `s_y`), and the spatial unit index (`s_idx`).
- `SparseMatrixCSC`: The adjacency matrix `W` for the spatial units.
"""
function generate_dirichlet_multinomial_data(;
    n_obs_per_unit::Int=10,
    n_units::Int=25,
    n_categories::Int=3,
    seed::Int=42
)
    # Purpose: Generates synthetic spatial data for a Dirichlet-Multinomial model.
    # Rationale for Change (v1.0.1):
    # Corrected a `MethodError` in the call to `assign_spatial_units`. The original
    # implementation passed a single matrix of coordinates, whereas the function expects
    # two separate vectors for x and y coordinates. The call has been updated to
    # `assign_spatial_units(points_for_partition[:, 1], points_for_partition[:, 2], ...)`
    # to correctly pass the coordinate columns.
    #
    # Inputs:
    #   - n_obs_per_unit: Number of observations per spatial unit.
    #   - n_units: The number of distinct spatial units.
    #   - n_categories: The number of categories for the multinomial outcome.
    #   - seed: A random seed for reproducibility.
    #
    # Outputs:
    #   - A DataFrame containing the simulated data.
    #   - The adjacency matrix `W` for the spatial units.

    Random.seed!(seed)
    
    # 1. Create spatial structure by generating random centroids for the units.
    centroids = rand(n_units, 2) .* 10.0
    
    # Use `assign_spatial_units` to get an adjacency matrix `W`.
    # We simulate points around the centroids to give the function something to partition.
    points_for_partition = vcat([centroids[i,:]' .+ randn(n_obs_per_unit, 2) for i in 1:n_units]...)
    
    # Corrected call to assign_spatial_units, passing x and y columns separately.
    au = assign_spatial_units(points_for_partition[:, 1], points_for_partition[:, 2]; target_units=n_units, area_method=:kvt)
    W = au.W
    
    # Create the final observation locations and assign them to the correct spatial unit.
    s_coords = vcat([centroids[i,:]' .+ randn(n_obs_per_unit, 2) for i in 1:n_units]...)
    s_idx = vcat([fill(i, n_obs_per_unit) for i in 1:n_units]...)

    # 2. Define true latent spatial fields for the Dirichlet concentration parameters.
    # We use a simple Gaussian Process to generate smooth spatial fields.
    dist_matrix = pairwise(Euclidean(), centroids, dims=1)
    
    # One spatial field per category to control its prevalence.
    alpha_fields = zeros(n_units, n_categories)
    for k in 1:n_categories
        ls = rand(1.5:0.1:3.0) # Lengthscale for the spatial field
        sigma_f = rand(0.8:0.1:1.2) # Signal variance
        K = sigma_f^2 .* exp.(-0.5 .* (dist_matrix ./ ls).^2) + I * 1e-6
        alpha_fields[:, k] = rand(MvNormal(zeros(n_units), K))
    end
    
    # 3. Generate observations from the latent fields.
    total_counts_per_obs = rand(80:150, n_units * n_obs_per_unit)
    category_counts = zeros(Int, n_units * n_obs_per_unit, n_categories)
    
    for i in 1:(n_units * n_obs_per_unit)
        unit_idx = s_idx[i]
        
        # Get the concentration parameters (`alphas`) for the observation's spatial unit.
        # The exp() transform ensures the alphas are positive.
        alphas = exp.(alpha_fields[unit_idx, :])
        
        # Sample proportions for this observation from a Dirichlet distribution.
        proportions = rand(Dirichlet(alphas))
        
        # Sample the final counts from a Multinomial distribution.
        category_counts[i, :] = rand(Multinomial(total_counts_per_obs[i], proportions))
    end
    
    # 4. Assemble the final DataFrame.
    df = DataFrame(
        s_x = s_coords[:, 1],
        s_y = s_coords[:, 2],
        s_idx = s_idx
    )
    
    for k in 1:n_categories
        df[!, Symbol("cat_", k)] = category_counts[:, k]
    end
    
    return df, W
end



"""
    generate_generalized_leslie_matrix_data(; s_N=10, t_N=10, n_classes=4, n_obs_per_st_unit=1, seed=123, use_effort::Bool=false, use_removal::Bool=false)

Generates synthetic data for a multivariate generalized Leslie/Lefkovitch matrix model.

This function simulates a stage-structured population with `n_classes` across a spatiotemporal domain.
The population dynamics are governed by a transition matrix `A`, where `A[i, j]` represents the
per-capita rate of production of individuals of class `i` from individuals of class `j`. This allows
for modeling complex life cycles beyond simple age progression, including stasis (staying in the same class)
and variable fecundity.

The simulation includes options for:
- Spatially varying carrying capacity (`K`).
- Exploitation modeled via fishing `effort` and catchability `q`.
- Direct `removal` of individuals from each class.

# Arguments
- `s_N`: Number of spatial units.
- `t_N`: Number of time steps.
- `n_classes`: The number of stages or classes in the population model.
- `n_obs_per_st_unit`: Number of observations to generate per space-time cell.
- `seed`: Random seed for reproducibility.
- `use_effort`: If true, generates an `effort` covariate.
- `use_removal`: If true, generates `removal` data for each class.

# Returns
- `DataFrame`: A dataframe containing the simulated data.
- `SparseMatrixCSC`: The adjacency matrix `W` for the spatial units.
- `Vector{Float64}`: A vector of grid cell areas.
- `Int`: The number of classes.
"""
function generate_generalized_leslie_matrix_data(; s_N=10, t_N=10, n_classes=4, n_obs_per_st_unit=1, seed=123, use_effort::Bool=false, use_removal::Bool=false)
    
    df, W, grid_areas = create_base_st_data(s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=seed)

    # True transition matrix (example for 4 classes)
    # A[i, j] is the rate of transition from class j to class i.
    A_true = zeros(n_classes, n_classes)
    if n_classes == 4
        A_true = [
            0.0  0.1  1.5  2.0;  # Fecundity from classes 2, 3, 4 into class 1
            0.5  0.2  0.0  0.0;  # Survival/growth from 1->2, and stasis in 2
            0.0  0.6  0.3  0.0;  # Survival/growth from 2->3, and stasis in 3
            0.0  0.0  0.7  0.4   # Survival/growth from 3->4, and stasis in 4
        ]
    else # Generic fallback
        for i in 1:n_classes
            A_true[1, i] = rand() * 0.5 # Fecundity
            if i > 1; A_true[i, i-1] = rand(0.4:0.1:0.8); end # Survival to next stage
            A_true[i, i] = rand(0.1:0.1:0.4) # Stasis
        end
    end


    # Spatially varying K
    s_coords = unique(df[!, [:s_idx, :s_x]])
    sort!(s_coords, :s_idx)
    K_spatial_true = 50.0 .+ 150.0 * (s_coords.s_x ./ maximum(s_coords.s_x))

    # Exploitation
    q_true = fill(0.01, n_classes) # Class-specific catchability
    effort_sim = use_effort ? rand(s_N, t_N) .* 5.0 : zeros(s_N, t_N)
    removal_sim = use_removal ? rand(s_N, t_N, n_classes) .* 1.0 : zeros(s_N, t_N, n_classes)

    # Simulate population dynamics
    pop_sim = zeros(s_N, t_N, n_classes)
    initial_total_pop = rand(s_N) .* 40.0 .+ 10.0
    for s in 1:s_N
        pop_sim[s, 1, :] = initial_total_pop[s] .* softmax(randn(n_classes))
    end

    for t in 2:t_N
        for s in 1:s_N
            N_prev = pop_sim[s, t-1, :]
            
            C_prev = zeros(n_classes)
            if use_effort
                C_prev .+= q_true .* effort_sim[s, t-1] .* N_prev
            end
            if use_removal
                C_prev .+= removal_sim[s, t-1, :]
            end
            N_after_removal = max.(0.0, N_prev - C_prev)
            
            L_effective = copy(A_true)
            total_pop_prev = sum(N_after_removal)
            K_s = K_spatial_true[s]
            dd_factor = max(0.0, 1.0 - total_pop_prev / K_s)
            L_effective[1, :] .*= dd_factor # Density dependence on fecundity

            N_projected = L_effective * N_after_removal
            pop_sim[s, t, :] = max.(0.0, N_projected .+ randn(n_classes) .* 0.5)
        end
    end

    # Add class columns to the DataFrame
    for a in 1:n_classes
        class_col_name = Symbol("class_$(a)")
        class_data_flat = vec(pop_sim[:, :, a]')
        df[!, class_col_name] = repeat(class_data_flat, inner=n_obs_per_st_unit)
    end
    df.y = df.class_1

    if use_effort
        df.effort = repeat(vec(effort_sim'), inner=n_obs_per_st_unit)
    end
    if use_removal
        for a in 1:n_classes
            df[!, Symbol("removal_class_$(a)")] = repeat(vec(removal_sim[:, :, a]'), inner=n_obs_per_st_unit)
        end
    end
    
    return df, W, grid_areas, n_classes
end
 