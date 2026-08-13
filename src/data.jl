# ==============================================================================
# Monolithic Data Generator and Synthetic Data Utilities for BSTM
# ==============================================================================

"""
    bstm_data(type="scottish_lip"; kwargs...)

Consolidated synthetic data generator for BSTM models.

# Supported Dataset Types (`type`):
- `"scottish_lip"` (Default): Scottish Lip Cancer spatiotemporal dataset (primary & nested) enriched
  with comprehensive covariates (`y`, `y_rate`, `y_bin`, `y_gauss`, `y_pois`, `ordinal_y`,
  `y_cat1..3`, `counts`, `t_idx`, `group`, `group_id`, `group_var`, `cell_area`, `effort`,
  `removal`, `removal_total`, `proxy_val`, `predator_pop`, `recruitment`, `habitat`,
  `species_1..3`, `age_1..3`, `class_1..4`).
- `"ordinal"`: Non-proportional odds ordinal dataset.
- `"sim"` / `"spatiotemporal"`: Standardized spatiotemporal dataset.
- `"lgcp_regular"`: Regular grid Log-Gaussian Cox Process synthetic data.
- `"lgcp_irregular"`: Irregular grid Log-Gaussian Cox Process synthetic data.
- `"advanced"`: Multi-response (Gaussian, Poisson, Multinomial, Proxy) spatiotemporal dataset.
- `"logistic"`: Logistic population dynamics synthetic dataset.
- `"delay_difference"`: Delay-difference multivariate population dynamics dataset.
- `"glv"` / `"generalized_lotka_volterra"`: Generalized Lotka-Volterra dynamics dataset.
- `"lotka_volterra"`: Lotka-Volterra prey-predator dynamics dataset.
- `"leslie_logistic"`: Leslie-Logistic population dataset.
- `"logistic_spatial_k"`: Logistic growth with spatially varying carrying capacity K.
- `"logistic_spatial_r"`: Logistic growth with spatially varying growth rate r.
- `"leslie_matrix"`: Multivariate Leslie matrix dynamics dataset.
- `"dirichlet_multinomial"`: Dirichlet-Multinomial spatial composition dataset.
- `"generalized_leslie_matrix"`: Generalized Leslie stage-structured dataset.

# Common Keyword Arguments:
- `s_N::Int = 10`: Number of spatial units.
- `t_N::Int = 5`: Number of temporal units.
- `n_years::Int = 10`: Number of years for Scottish Lip Cancer dataset.
- `spatial_expansion::Float64 = 1.5`: Spatial expansion factor for nested Scottish Lip dataset.
- `temporal_expansion::Float64 = 1.5`: Temporal expansion factor for nested Scottish Lip dataset.
- `rndseed::Int = 42`: Random seed.
- `seed::Union{Int, Nothing} = nothing`: Alternative random seed keyword.
- `recreate::Bool = false`: Force recreation of cached datasets.
- `grid_side::Int = 15`: Grid side length for LGCP datasets.
- `n_species::Int = 3`: Number of species for multi-species models.
- `n_age_classes::Int = 3`: Number of age classes for Leslie models.
- `n_classes::Int = 4`: Number of classes for generalized Leslie models.
- `n_obs::Int = 500`: Total observations for ordinal data.
- `n_groups::Int = 10`: Group count for random intercepts.
- `n_obs_per_st_unit::Int = 1`: Observations per space-time unit.
- `n_obs_per_unit::Int = 10`: Observations per unit for Dirichlet-Multinomial.
- `n_units::Int = 25`: Spatial units for Dirichlet-Multinomial.
- `n_categories::Int = 3`: Category count for multinomial models.
- `use_effort::Bool = false`: Include effort covariate.
- `use_removal::Bool = false`: Include removal covariate.
"""
function bstm_data(
    type::Union{String, Symbol} = "scottish_lip";
    s_N::Int = 10,
    t_N::Int = 5,
    n_years::Int = 10,
    spatial_expansion::Float64 = 1.5,
    temporal_expansion::Float64 = 1.5,
    rndseed::Int = 42,
    seed::Union{Int, Nothing} = nothing,
    recreate::Bool = false,
    grid_side::Int = 15,
    n_species::Int = 3,
    n_age_classes::Int = 3,
    n_classes::Int = 4,
    n_obs::Int = 500,
    n_groups::Int = 10,
    n_obs_per_st_unit::Int = 1,
    n_obs_per_unit::Int = 10,
    n_units::Int = 25,
    n_categories::Int = 3,
    use_effort::Bool = false,
    use_removal::Bool = false
)
    actual_seed = seed !== nothing ? seed : rndseed
    type_str = lowercase(string(type))

    if type_str in ["scottish_lip", "scottish"]
        cache_path = "data/scottish_lip_cancer_cache.jld2"

        if isfile(cache_path) && !recreate
            try
                println("Loading cached dataset from: ", cache_path)
                data_bundle = JLD2.load(cache_path)
                p_out = data_bundle["primary"]
                n_out = data_bundle["nested"]
                if hasproperty(p_out.data, :y_gauss) && hasproperty(p_out.data, :t_idx)
                    return (p_out, n_out)
                else
                    println("Cached dataset missing consolidated covariates; regenerating...")
                end
            catch e
                println("Failed to load cache ($e); regenerating dataset...")
            end
        end

        println("Generating new spatiotemporal dataset...")
        Random.seed!(actual_seed)

        n_districts = 56

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

        W_raw = spzeros(Int, n_districts, n_districts)
        for i in 1:n_districts
            for nb in neighbor_list[i]
                W_raw[i, nb] = 1
            end
        end
        W = sparse(Symmetric(Matrix(W_raw + W_raw')) .> 0)

        au_primary = assign_spatial_units_inferred(W)
        p_centroids = au_primary.centroids

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
                t_idx = 1:n_years,
                y = y_p,
                log_offsets = log_off,
                cov1 = fill(Float64(x_orig[i]), n_years)
            )
            d_df.y_rate = d_df.y ./ exp.(d_df.log_offsets)
            append!(data_primary, d_df)
        end

        n_total = nrow(data_primary)

        data_primary.y_bin = [v > mean(data_primary.y_rate) ? 1 : 0 for v in data_primary.y_rate]

        data_primary.cov2 = 0.5 .* data_primary.cov1 .+ randn(n_total)
        data_primary.cov3 = randn(n_total) .* (data_primary.y_rate .^ 2)
        data_primary.cov4 = randn(n_total) .* log.(data_primary.y_rate .+ 1.0)
        data_primary.cov5 = randn(n_total) .* exp.(data_primary.y_rate) .* 2.0
        data_primary.cov6 = randn(n_total)

        data_primary.day = rand(1:365, n_total)
        data_primary.month = Int.(round.(data_primary.day ./ 365 .* 12)) .+ 1

        data_primary.f1 = rand(["A", "B"], n_total)
        data_primary.s_idx = data_primary.district
        data_primary.s_x = [c[1] for c in p_centroids[data_primary.s_idx]]
        data_primary.s_y = [c[2] for c in p_centroids[data_primary.s_idx]]

        reg_indices = mod1.(1:n_total, 4)
        reg_levels = ["North", "South", "East", "West"]
        reg = reg_levels[reg_indices]
        data_primary.region = categorical(reg)

        data_primary.group = categorical(data_primary.district)
        data_primary.group_id = categorical(data_primary.district)
        data_primary.group_var = categorical(data_primary.region)

        # Comprehensive additional response types and covariates for component testing
        data_primary.y_gauss = Float64.(data_primary.y_rate) .+ randn(n_total) .* 0.1
        data_primary.y_pois = data_primary.y
        data_primary.counts = data_primary.y
        data_primary.cell_area = fill(1.0, n_total)

        data_primary.ordinal_y = [v == 0 ? 1 : (v < 5 ? 2 : 3) for v in data_primary.y]

        eta_m = hcat(data_primary.cov1, data_primary.cov2, zeros(n_total))
        y_mult = zeros(Int, n_total, 3)
        for i in 1:n_total
            p = NNlib.softmax(eta_m[i, :])
            y_mult[i, :] = rand(Multinomial(20, p))
        end
        data_primary.y_cat1 = y_mult[:, 1]
        data_primary.y_cat2 = y_mult[:, 2]
        data_primary.y_cat3 = y_mult[:, 3]

        data_primary.effort = rand(n_total) .* 0.5 .+ 0.1
        data_primary.removal = rand(n_total) .* 2.0
        data_primary.removal_total = data_primary.removal
        data_primary.proxy_val = data_primary.y_gauss .+ randn(n_total) .* 0.3
        data_primary.predator_pop = rand(n_total) .* 5.0 .+ 1.0
        data_primary.recruitment = rand(n_total) .* 10.0 .+ 2.0
        data_primary.habitat = rand(n_total)
        data_primary.species_1 = rand(n_total) .* 10.0
        data_primary.species_2 = rand(n_total) .* 10.0
        data_primary.species_3 = rand(n_total) .* 10.0
        data_primary.age_1 = rand(n_total) .* 10.0
        data_primary.age_2 = rand(n_total) .* 10.0
        data_primary.age_3 = rand(n_total) .* 10.0
        data_primary.class_1 = rand(n_total) .* 10.0
        data_primary.class_2 = rand(n_total) .* 10.0
        data_primary.class_3 = rand(n_total) .* 10.0
        data_primary.class_4 = rand(n_total) .* 10.0

        au_primary = merge(au_primary, (
            s_idx = data_primary.s_idx,
            s_x = [c[1] for c in p_centroids[data_primary.s_idx]],
            s_y = [c[2] for c in p_centroids[data_primary.s_idx]],
            s_vals = collect(1:n_districts)
        ))

        # Nested dataset construction
        px = [c[1] for c in p_centroids]
        py = [c[2] for c in p_centroids]
        x_min, x_max = minimum(px), maximum(px)
        y_min, y_max = minimum(py), maximum(py)
        x_rng, y_rng = x_max - x_min, y_max - y_min

        s_buff = (spatial_expansion - 1.0) / 2.0
        nx_min, nx_max = x_min - s_buff * x_rng, x_max + s_buff * x_rng
        ny_min, ny_max = y_min - s_buff * y_rng, y_max + s_buff * y_rng

        nt_max = Int(round(n_years * temporal_expansion))
        n_obs_nested = Int(round(n_total * spatial_expansion * temporal_expansion))

        sx_nested = rand(Uniform(nx_min, nx_max), n_obs_nested)
        sy_nested = rand(Uniform(ny_min, ny_max), n_obs_nested)
        time_nested = rand(1:nt_max, n_obs_nested)

        au_nested = assign_spatial_units(sx_nested, sy_nested; target_units=100)

        data_nested = DataFrame(
            s_x = sx_nested,
            s_y = sy_nested,
            year = time_nested,
            t_idx = time_nested,
            district = au_nested.s_idx,
            s_idx = au_nested.s_idx
        )

        s_lat_n = cumsum(randn(length(au_nested.centroids))) .* 0.3
        t_lat_n = sin.(collect(1:nt_max) .* (2π/nt_max))

        eta_n = [1.5 + s_lat_n[data_nested.district[i]] + t_lat_n[data_nested.year[i]] for i in 1:n_obs_nested]

        data_nested.y = [rand(Poisson(exp(v))) for v in eta_n]
        data_nested.y_rate = exp.(eta_n) .+ randn(n_obs_nested) .* 0.2
        data_nested.y_bin = [v > mean(data_nested.y_rate) ? 1 : 0 for v in data_nested.y_rate]

        data_nested.cov1 = 0.6 .* eta_n .+ randn(n_obs_nested)
        data_nested.cov2 = randn(n_obs_nested) .* exp.(data_nested.y_rate)
        data_nested.cov3 = randn(n_obs_nested)
        data_nested.ncov1 = data_nested.cov1
        data_nested.ncov2 = data_nested.cov2
        data_nested.ncov3 = data_nested.cov3
        data_nested.group = categorical(data_nested.district)
        data_nested.group_id = categorical(data_nested.district)
        data_nested.month = mod1.(data_nested.year, 12)
        data_nested.day = rand(1:365, n_obs_nested)

        primary_out = (data=data_primary, au=au_primary)
        nested_out = (data=data_nested, au=au_nested)

        if !isdir("data"); mkdir("data"); end
        JLD2.save(cache_path, "primary", primary_out, "nested", nested_out)
        println("Dataset successfully cached at: ", cache_path)

        return (primary_out, nested_out)

    elseif type_str == "ordinal"
        Random.seed!(actual_seed)
        cov1 = randn(n_obs)
        cov2 = rand(n_obs) .* 2
        cov3 = rand(-2:2, n_obs)
        group_id = rand(1:n_groups, n_obs)

        alpha_1 = -1.0
        alpha_2 = 1.5
        beta_cov2 = 0.5
        beta_cov3 = -0.4
        beta_cov1_cat1 = 1.2
        beta_cov1_cat2 = -0.8

        sigma_random_intercept = 0.7
        random_intercepts = rand(Normal(0, sigma_random_intercept), n_groups)

        ordinal_y = Vector{Int}(undef, n_obs)
        for i in 1:n_obs
            eta_proportional = (beta_cov2 * cov2[i]) + (beta_cov3 * cov3[i]) + random_intercepts[group_id[i]]
            linear_pred_1 = alpha_1 - (eta_proportional + cov1[i] * beta_cov1_cat1)
            linear_pred_2 = alpha_2 - (eta_proportional + cov1[i] * beta_cov1_cat2)

            cum_prob_1 = cdf(Normal(), linear_pred_1)
            cum_prob_2 = cdf(Normal(), linear_pred_2)

            prob_1 = cum_prob_1
            prob_2 = max(0.0, cum_prob_2 - cum_prob_1)
            prob_3 = max(0.0, 1.0 - cum_prob_2)

            probs = [prob_1, prob_2, prob_3]
            probs ./= sum(probs)
            ordinal_y[i] = rand(Categorical(probs))
        end

        return DataFrame(
            ordinal_y = ordinal_y,
            cov1 = cov1,
            cov2 = cov2,
            cov3 = cov3,
            group_id = categorical(group_id)
        )

    elseif type_str in ["sim", "spatiotemporal"]
        Random.seed!(actual_seed)
        s_coord_tuple = tuple.([(mod(i-1, 5), div(i-1, 5)) for i in 1:s_N]...)
        s_x = [pt[1] for pt in s_coord_tuple]
        s_y = [pt[2] for pt in s_coord_tuple]
        W = adjacency_matrix(s_x, s_y)

        t_idx = 1:t_N
        phi_s = rand(MvNormal(zeros(s_N), I))
        phi_t = sin.(2 * pi .* (t_idx./t_N)) .+ rand(Normal(0,0.1), t_N)
        phi_st = rand(MvNormal(zeros(s_N*t_N), I))

        s_cov = rand(Normal(0,1), s_N)
        t_cov = rand(Normal(0,1), t_N)

        n_total = s_N * t_N
        s_v = vcat([fill(i, t_N) for i in 1:s_N]...)
        t_v = vcat([t_idx for i in 1:s_N]...)

        eta = (
            0.5 .* phi_s[s_v] .+
            0.8 .* phi_t[t_v] .+
            0.2 .* s_cov[s_v] .+
            0.3 .* t_cov[t_v]
        )

        y1 = eta .+ rand(Normal(0, 0.25), n_total)
        y2 = [rand(Poisson(exp(v))) for v in eta]

        y_binary = Vector{Int}(undef, n_total)
        mean_eta = mean(eta)
        for i in 1:n_total
            y_binary[i] = eta[i] > mean_eta ? 1 : 0
        end

        eta_mult = hcat(eta, -eta, sin.(eta))
        y_mult = zeros(Int, n_total, 3)
        for i in 1:n_total
            p = NNlib.softmax(eta_mult[i, :])
            y_mult[i, :] = rand(Multinomial(20, p))
        end

        proxy_y = rand(Normal(0,1), n_total)

        return DataFrame(
            s_idx = s_v,
            t_idx = t_v,
            s_x = s_x[s_v],
            s_y = s_y[s_v],
            year = t_v,
            month = mod1.(t_v, 12),
            y_gauss = y1,
            y_pois = y2,
            y_bin = y_binary,
            y_cat1 = y_mult[:, 1],
            y_cat2 = y_mult[:, 2],
            y_cat3 = y_mult[:, 3],
            proxy_val = proxy_y
        )

    elseif type_str in ["lgcp_regular", "lgcp"]
        Random.seed!(actual_seed)
        s_N_eff = grid_side^2
        x_coords = repeat(1:grid_side, inner=grid_side)
        y_coords = repeat(1:grid_side, outer=grid_side)

        Z_true = [2.0 * sin(x/3.0) * cos(y/3.0) for (x, y) in zip(x_coords, y_coords)]
        y_counts = [rand(Poisson(exp(z))) for z in Z_true]

        df = DataFrame(
            s_idx = 1:s_N_eff,
            s_x = Float64.(x_coords),
            s_y = Float64.(y_coords),
            counts = y_counts
        )
        W = libgeos_lattice_adjacency_matrix(grid_side, grid_side)
        return df, W, s_N_eff

    elseif type_str == "lgcp_irregular"
        Random.seed!(actual_seed)
        s_N_eff = grid_side^2
        x_coords = repeat(1:grid_side, inner=grid_side)
        y_coords = repeat(1:grid_side, outer=grid_side)

        areas = [1.0 + 2.0 * exp(-((x - 5)^2 + (y - 5)^2) / 10.0) for (x, y) in zip(x_coords, y_coords)]
        Z_true = fill(log(5.0), s_N_eff)
        y_counts = [rand(Poisson(exp(z) * a)) for (z, a) in zip(Z_true, areas)]

        df = DataFrame(
            s_idx = 1:s_N_eff,
            counts = y_counts,
            cell_area = areas
        )
        W = libgeos_lattice_adjacency_matrix(grid_side, grid_side)
        return df, W, s_N_eff, areas

    elseif type_str == "advanced"
        Random.seed!(actual_seed)
        n_s = 30
        n_t = 12
        n_obs_tot = n_s * n_t

        unique_coords = [(rand()*10, rand()*10) for _ in 1:n_s]
        s_coords = repeat(unique_coords, inner=n_t)
        s_x = [c[1] for c in s_coords]
        s_y = [c[2] for c in s_coords]
        t_v = repeat(collect(1:n_t), outer=n_s)
        s_idx = repeat(collect(1:n_s), inner=n_t)

        latent_field = [sin(x/2) * cos(y/2) + 0.1*t for (x, y, t) in zip(s_x, s_y, t_v)]

        y1 = latent_field .+ randn(n_obs_tot) .* 0.2
        y2 = [rand(Poisson(exp(v))) for v in latent_field]

        eta_mult = hcat(latent_field, -0.5 .* latent_field, zeros(n_obs_tot))
        y_mult = zeros(Int, n_obs_tot, 3)
        for i in 1:n_obs_tot
            p = NNlib.softmax(eta_mult[i, :])
            y_mult[i, :] = rand(Multinomial(20, p))
        end

        proxy_y = latent_field .* 0.8 .+ randn(n_obs_tot) .* 0.5

        return DataFrame(
            s_idx = s_idx,
            s_x = s_x,
            s_y = s_y,
            year = t_v,
            t_idx = t_v,
            month = mod1.(t_v, 12),
            y_gauss = y1,
            y_pois = y2,
            y_cat1 = y_mult[:, 1],
            y_cat2 = y_mult[:, 2],
            y_cat3 = y_mult[:, 3],
            proxy_val = proxy_y
        )

    elseif type_str == "logistic"
        df, W, grid_areas = create_base_st_data(s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=actual_seed)
        r_true = 0.5
        K_true = 100.0
        q_true = 0.01

        effort_sim = use_effort ? rand(s_N, t_N) .* 0.5 .+ 0.1 : zeros(s_N, t_N)
        removal_sim = use_removal ? rand(s_N, t_N) .* 5.0 : zeros(s_N, t_N)

        initial_pop = rand(s_N) * 10.0 .+ 5.0
        y_sim = zeros(s_N, t_N)
        y_sim[:, 1] = initial_pop

        for t in 2:t_N
            for s in 1:s_N
                N_prev = y_sim[s, t-1]
                D_prev = N_prev / grid_areas[s]
                K_density = K_true / grid_areas[s]
                growth = r_true * D_prev * (1.0 - D_prev / K_density)

                exploitation = 0.0
                if use_effort; exploitation += q_true * effort_sim[s, t] * N_prev; end
                if use_removal; exploitation += removal_sim[s, t]; end

                y_sim[s, t] = max(0.0, N_prev + growth * grid_areas[s] - exploitation + randn() * 2.0)
            end
        end

        df.y = repeat(vec(y_sim'), inner=n_obs_per_st_unit)
        if use_effort; df.effort = repeat(vec(effort_sim'), inner=n_obs_per_st_unit); end
        if use_removal; df.removal = repeat(vec(removal_sim'), inner=n_obs_per_st_unit); end

        return df, W, grid_areas

    elseif type_str == "delay_difference"
        df, W, grid_areas = create_base_st_data(s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=actual_seed)
        r_true = 0.6
        K_true = 150.0
        M_nat_true = 0.2
        sigma_rec_true = 0.2
        sigma_pop_true = 0.1

        population_sim = zeros(s_N, t_N)
        recruitment_sim = zeros(s_N, t_N)

        q_true = 0.01
        effort_sim = use_effort ? rand(s_N, t_N) .* 10.0 : zeros(s_N, t_N)
        removal_sim = use_removal ? rand(s_N, t_N) .* 5.0 : zeros(s_N, t_N)

        initial_pop = rand(s_N) .* 20.0 .+ 10.0
        population_sim[:, 1] = initial_pop
        recruitment_sim[:, 1] = initial_pop .* 0.2

        for t in 2:t_N
            for s in 1:s_N
                N_prev = population_sim[s, t-1]
                D_prev = N_prev / grid_areas[s]
                K_density = K_true / grid_areas[s]

                mean_rec = r_true * D_prev * (1.0 - D_prev / K_density) * grid_areas[s]
                recruitment_sim[s, t] = exp(log(mean_rec + 1e-6) + randn() * sigma_rec_true)

                C_prev = 0.0
                if use_effort; C_prev += q_true * effort_sim[s, t-1] * N_prev; end
                if use_removal; C_prev += removal_sim[s, t-1]; end

                N_survived = (N_prev - C_prev) * exp(-M_nat_true)
                population_sim[s, t] = max(0.0, N_survived + recruitment_sim[s, t] + randn() * sigma_pop_true)
            end
        end

        df.y = repeat(vec(population_sim'), inner=n_obs_per_st_unit)
        df.recruitment = repeat(vec(recruitment_sim'), inner=n_obs_per_st_unit)
        if use_effort; df.effort = repeat(vec(effort_sim'), inner=n_obs_per_st_unit); end
        if use_removal; df.removal = repeat(vec(removal_sim'), inner=n_obs_per_st_unit); end

        return df, W, grid_areas

    elseif type_str in ["glv", "generalized_lotka_volterra"]
        df, W, grid_areas = create_base_st_data(s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=actual_seed)
        r_true = [0.5, 0.6, 0.7]
        K_true = [100.0, 120.0, 150.0]
        alpha_true = [1.0 0.5 0.2; 0.3 1.0 0.6; 0.1 0.4 1.0]
        sigma_process_true = [0.1, 0.1, 0.1]

        pop_sim = zeros(s_N, t_N, n_species)
        initial_total_pop = rand(s_N) .* 30.0 .+ 10.0
        for s in 1:s_N
            pop_sim[s, 1, :] = initial_total_pop[s] .* NNlib.softmax(randn(n_species))
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

        for a in 1:n_species
            species_col_name = Symbol("species_$(a)")
            species_data_flat = vec(pop_sim[:, :, a]')
            df[!, species_col_name] = repeat(species_data_flat, inner=n_obs_per_st_unit)
        end
        df.y = df.species_1

        return df, W, grid_areas, n_species

    elseif type_str == "lotka_volterra"
        df, W, grid_areas = create_base_st_data(s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=actual_seed)
        prey_sim = zeros(s_N, t_N)
        predator_sim = zeros(s_N, t_N)

        initial_prey = rand(s_N) * 10.0 .+ 5.0
        initial_predator = rand(s_N) * 2.0 .+ 1.0
        prey_sim[:, 1] = initial_prey
        predator_sim[:, 1] = initial_predator

        alpha_true = 0.5
        beta_true = 0.01
        gamma_true = 0.005
        delta_true = 0.2

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

        df.y = repeat(vec(prey_sim'), inner=n_obs_per_st_unit)
        df.predator_pop = repeat(vec(predator_sim'), inner=n_obs_per_st_unit)
        return df, W, grid_areas

    elseif type_str == "leslie_logistic"
        df, W, grid_areas = create_base_st_data(s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=actual_seed)
        y_sim = zeros(s_N, t_N)
        initial_pop = rand(s_N) * 10.0 .+ 5.0
        y_sim[:, 1] = initial_pop

        K_true = 100.0
        survival_true = fill(0.8, n_age_classes - 1)
        fecundity_true = [0.0, 1.5, 2.0]

        L_true = zeros(n_age_classes, n_age_classes)
        for i in 1:(n_age_classes - 1); L_true[i+1, i] = survival_true[i]; end
        L_true[1, :] = fecundity_true

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

    elseif type_str == "logistic_spatial_k"
        df, W, grid_areas = create_base_st_data(s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=actual_seed)
        s_coords = unique(df[!, [:s_idx, :s_x]])
        sort!(s_coords, :s_idx)
        K_spatial_true = 50.0 .+ 150.0 * (s_coords.s_x ./ maximum(s_coords.s_x))

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

    elseif type_str == "logistic_spatial_r"
        df, W, grid_areas = create_base_st_data(s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=actual_seed)
        s_coords = unique(df[!, [:s_idx, :s_y]])
        sort!(s_coords, :s_idx)
        r_spatial_true = 0.2 .+ 0.8 * (s_coords.s_y ./ maximum(s_coords.s_y))

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

    elseif type_str == "leslie_matrix"
        df, W, grid_areas = create_base_st_data(s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=actual_seed)
        survival_true = [0.5, 0.8]
        fecundity_true = [0.0, 1.5, 3.0]

        L_true = zeros(n_age_classes, n_age_classes)
        for i in 1:(n_age_classes - 1); L_true[i+1, i] = survival_true[i]; end
        L_true[1, :] = fecundity_true

        q_true = fill(0.005, n_age_classes)
        effort_sim = use_effort ? rand(s_N, t_N) .* 10.0 : zeros(s_N, t_N)
        removal_sim = use_removal ? rand(s_N, t_N, n_age_classes) .* 2.0 : zeros(s_N, t_N, n_age_classes)

        pop_sim = zeros(s_N, t_N, n_age_classes)
        initial_total_pop = rand(s_N) * 50.0 .+ 20.0
        for s in 1:s_N
            pop_sim[s, 1, :] = initial_total_pop[s] .* NNlib.softmax(randn(n_age_classes))
        end

        for t in 2:t_N
            for s in 1:s_N
                N_prev = pop_sim[s, t-1, :]
                C_prev = zeros(n_age_classes)
                if use_effort; C_prev .+= q_true .* effort_sim[s, t-1] .* N_prev; end
                if use_removal; C_prev .+= removal_sim[s, t-1, :]; end
                N_after_removal = max.(0.0, N_prev - C_prev)

                N_projected = L_true * N_after_removal
                pop_sim[s, t, :] = max.(0.0, N_projected .+ randn(n_age_classes) .* 0.5)
            end
        end

        for a in 1:n_age_classes
            age_col_name = Symbol("age_$(a)")
            age_data_flat = vec(pop_sim[:, :, a]')
            df[!, age_col_name] = repeat(age_data_flat, inner=n_obs_per_st_unit)
        end
        df.y = df.age_1

        if use_effort; df.effort = repeat(vec(effort_sim'), inner=n_obs_per_st_unit); end
        if use_removal; df.removal_total = repeat(vec(sum(removal_sim, dims=3)[:,:,1]'), inner=n_obs_per_st_unit); end

        return df, W, grid_areas, n_age_classes

    elseif type_str == "dirichlet_multinomial"
        Random.seed!(actual_seed)
        centroids = rand(n_units, 2) .* 10.0
        points_for_partition = vcat([centroids[i,:]' .+ randn(n_obs_per_unit, 2) for i in 1:n_units]...)

        au = assign_spatial_units(points_for_partition[:, 1], points_for_partition[:, 2]; target_units=n_units, area_method=:kvt)
        W = au.W

        s_coords = vcat([centroids[i,:]' .+ randn(n_obs_per_unit, 2) for i in 1:n_units]...)
        s_idx = vcat([fill(i, n_obs_per_unit) for i in 1:n_units]...)

        dist_matrix = pairwise(Euclidean(), centroids, dims=1)
        alpha_fields = zeros(n_units, n_categories)
        for k in 1:n_categories
            ls = rand(1.5:0.1:3.0)
            sigma_f = rand(0.8:0.1:1.2)
            K = sigma_f^2 .* exp.(-0.5 .* (dist_matrix ./ ls).^2) + I * 1e-6
            alpha_fields[:, k] = rand(MvNormal(zeros(n_units), K))
        end

        total_counts_per_obs = rand(80:150, n_units * n_obs_per_unit)
        category_counts = zeros(Int, n_units * n_obs_per_unit, n_categories)

        for i in 1:(n_units * n_obs_per_unit)
            unit_idx = s_idx[i]
            alphas = exp.(alpha_fields[unit_idx, :])
            proportions = rand(Dirichlet(alphas))
            category_counts[i, :] = rand(Multinomial(total_counts_per_obs[i], proportions))
        end

        df = DataFrame(
            s_x = s_coords[:, 1],
            s_y = s_coords[:, 2],
            s_idx = s_idx
        )
        for k in 1:n_categories
            df[!, Symbol("cat_", k)] = category_counts[:, k]
        end

        return df, W

    elseif type_str == "generalized_leslie_matrix"
        df, W, grid_areas = create_base_st_data(s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=actual_seed)

        A_true = zeros(n_classes, n_classes)
        if n_classes == 4
            A_true = [
                0.0  0.1  1.5  2.0;
                0.5  0.2  0.0  0.0;
                0.0  0.6  0.3  0.0;
                0.0  0.0  0.7  0.4
            ]
        else
            for i in 1:n_classes
                A_true[1, i] = rand() * 0.5
                if i > 1; A_true[i, i-1] = rand(0.4:0.1:0.8); end
                A_true[i, i] = rand(0.1:0.1:0.4)
            end
        end

        s_coords = unique(df[!, [:s_idx, :s_x]])
        sort!(s_coords, :s_idx)
        K_spatial_true = 50.0 .+ 150.0 * (s_coords.s_x ./ maximum(s_coords.s_x))

        q_true = fill(0.01, n_classes)
        effort_sim = use_effort ? rand(s_N, t_N) .* 5.0 : zeros(s_N, t_N)
        removal_sim = use_removal ? rand(s_N, t_N, n_classes) .* 1.0 : zeros(s_N, t_N, n_classes)

        pop_sim = zeros(s_N, t_N, n_classes)
        initial_total_pop = rand(s_N) .* 40.0 .+ 10.0
        for s in 1:s_N
            pop_sim[s, 1, :] = initial_total_pop[s] .* NNlib.softmax(randn(n_classes))
        end

        for t in 2:t_N
            for s in 1:s_N
                N_prev = pop_sim[s, t-1, :]
                C_prev = zeros(n_classes)
                if use_effort; C_prev .+= q_true .* effort_sim[s, t-1] .* N_prev; end
                if use_removal; C_prev .+= removal_sim[s, t-1, :]; end
                N_after_removal = max.(0.0, N_prev - C_prev)

                L_effective = copy(A_true)
                total_pop_prev = sum(N_after_removal)
                K_s = K_spatial_true[s]
                dd_factor = max(0.0, 1.0 - total_pop_prev / K_s)
                L_effective[1, :] .*= dd_factor

                N_projected = L_effective * N_after_removal
                pop_sim[s, t, :] = max.(0.0, N_projected .+ randn(n_classes) .* 0.5)
            end
        end

        for a in 1:n_classes
            class_col_name = Symbol("class_$(a)")
            class_data_flat = vec(pop_sim[:, :, a]')
            df[!, class_col_name] = repeat(class_data_flat, inner=n_obs_per_st_unit)
        end
        df.y = df.class_1

        if use_effort; df.effort = repeat(vec(effort_sim'), inner=n_obs_per_st_unit); end
        if use_removal
            for a in 1:n_classes
                df[!, Symbol("removal_class_$(a)")] = repeat(vec(removal_sim[:, :, a]'), inner=n_obs_per_st_unit)
            end
        end

        return df, W, grid_areas, n_classes

    else
        throw(ArgumentError("Unknown dataset type '$type'. Supported types are: " *
            "scottish_lip, ordinal, sim, lgcp_regular, lgcp_irregular, advanced, " *
            "logistic, delay_difference, glv, lotka_volterra, leslie_logistic, " *
            "logistic_spatial_k, logistic_spatial_r, leslie_matrix, " *
            "dirichlet_multinomial, generalized_leslie_matrix."))
    end
end


# ==============================================================================
# Internal Helper Functions
# ==============================================================================

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
    df = DataFrame(s_idx=s_idx_flat, year=t_idx_flat, t_idx=t_idx_flat, s_x=s_x_flat, s_y=s_y_flat, grid_area_col=repeat(grid_areas, inner=t_N * n_obs_per_st_unit))
    return df, W, grid_areas
end


# ==============================================================================
# Legacy Wrappers for Backward Compatibility
# ==============================================================================

scottish_lip_cancer_data_spacetime(n_years::Int=10, spatial_expansion::Float64=1.5, temporal_expansion::Float64=1.5; rndseed::Int=42, recreate::Bool=false) =
    bstm_data("scottish_lip"; n_years=n_years, spatial_expansion=spatial_expansion, temporal_expansion=temporal_expansion, rndseed=rndseed, recreate=recreate)

generate_ordinal_data(; n_obs::Int=500, n_groups::Int=10, seed::Int=42) =
    bstm_data("ordinal"; n_obs=n_obs, n_groups=n_groups, seed=seed)

generate_sim_data(s_N=25, t_N=10; rndseed=42) =
    bstm_data("sim"; s_N=s_N, t_N=t_N, rndseed=rndseed)

generate_lgcp_synthetic_data_regular(grid_side=15) =
    bstm_data("lgcp_regular"; grid_side=grid_side)

generate_irregular_lgcp_data(grid_side=10) =
    bstm_data("lgcp_irregular"; grid_side=grid_side)

prepare_advanced_bstm_data() =
    bstm_data("advanced")

generate_logistic_data(; s_N=10, t_N=5, n_obs_per_st_unit=1, seed=123, use_effort::Bool=false, use_removal::Bool=false) =
    bstm_data("logistic"; s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=seed, use_effort=use_effort, use_removal=use_removal)

generate_delay_difference_data(; s_N=10, t_N=10, n_obs_per_st_unit=1, seed=123, use_effort::Bool=false, use_removal::Bool=false) =
    bstm_data("delay_difference"; s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=seed, use_effort=use_effort, use_removal=use_removal)

generate_glv_data(; s_N=10, t_N=10, n_species=3, n_obs_per_st_unit=1, seed=123) =
    bstm_data("glv"; s_N=s_N, t_N=t_N, n_species=n_species, n_obs_per_st_unit=n_obs_per_st_unit, seed=seed)

generate_lotka_volterra_data(; s_N=10, t_N=5, n_obs_per_st_unit=1, seed=123) =
    bstm_data("lotka_volterra"; s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=seed)

generate_leslie_logistic_data(; s_N=10, t_N=5, n_obs_per_st_unit=1, n_age_classes=3, seed=123) =
    bstm_data("leslie_logistic"; s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, n_age_classes=n_age_classes, seed=seed)

generate_logistic_spatial_K_data(; s_N=10, t_N=5, n_obs_per_st_unit=1, seed=123) =
    bstm_data("logistic_spatial_k"; s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=seed)

generate_logistic_spatial_r_data(; s_N=10, t_N=5, n_obs_per_st_unit=1, seed=123) =
    bstm_data("logistic_spatial_r"; s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=seed)

generate_leslie_matrix_data(; s_N=10, t_N=5, n_age_classes=3, n_obs_per_st_unit=1, seed=123, use_effort::Bool=false, use_removal::Bool=false) =
    bstm_data("leslie_matrix"; s_N=s_N, t_N=t_N, n_age_classes=n_age_classes, n_obs_per_st_unit=n_obs_per_st_unit, seed=seed, use_effort=use_effort, use_removal=use_removal)

generate_dirichlet_multinomial_data(; n_obs_per_unit::Int=10, n_units::Int=25, n_categories::Int=3, seed::Int=42) =
    bstm_data("dirichlet_multinomial"; n_obs_per_unit=n_obs_per_unit, n_units=n_units, n_categories=n_categories, seed=seed)

generate_generalized_leslie_matrix_data(; s_N=10, t_N=10, n_classes=4, n_obs_per_st_unit=1, seed=123, use_effort::Bool=false, use_removal::Bool=false) =
    bstm_data("generalized_leslie_matrix"; s_N=s_N, t_N=t_N, n_classes=n_classes, n_obs_per_st_unit=n_obs_per_st_unit, seed=seed, use_effort=use_effort, use_removal=use_removal)