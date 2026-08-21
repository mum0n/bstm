"""
    movement.jl

Advection-Diffusion-Reaction (ADR) spatiotemporal movement, spatial telemetry,
and particle trajectory simulation engine for Bayesian Spatio-Temporal Models (BSTM).

Version: v1.0.0
"""

"""
    simulate_correlated_density_vector(habitat_prob, rho_target, log_mu, sigma_resid;
                                        rng=Random.GLOBAL_RNG)

Simulates a spatial population density vector correlated with habitat suitability index (HSI).

# Arguments
- `habitat_prob::AbstractVector{<:Real}`: Habitat suitability probabilities for spatial units.
- `rho_target::Real`: Desired correlation between habitat suitability and log-density.
- `log_mu::Real`: Mean of the log-density field.
- `sigma_resid::Real`: Marginal residual standard deviation of log-density.
- `rng::AbstractRNG`: Random number generator.

# Returns
- `Vector{Float64}`: Simulated positive population density values.
"""
function simulate_correlated_density_vector(
    habitat_prob::AbstractVector{<:Real},
    rho_target::Real,
    log_mu::Real,
    sigma_resid::Real;
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
)
    n = length(habitat_prob)
    p_std = (habitat_prob .- mean(habitat_prob)) ./ (std(habitat_prob) + 1e-9)
    epsilon = randn(rng, n)
    log_n_signal = (rho_target .* p_std) .+ (sqrt(max(0.0, 1.0 - rho_target^2)) .* epsilon)
    log_n = log_mu .+ (log_n_signal .* sigma_resid)
    return exp.(log_n)
end

"""
    generate_ADR_simulation_bundle(domain_size, n_units, n_years, n_marks;
                                   area_method=:hexagonal, rng=Random.GLOBAL_RNG)

Generates a complete synthetic simulation bundle for joint Advection-Diffusion-Reaction (ADR)
population density surveys and individual mark-recapture telemetry.

# Arguments
- `domain_size::Real`: Spatial bounding box size (e.g. 1000.0 km).
- `n_units::Int`: Target number of spatial partitioning units.
- `n_years::Int`: Number of discrete temporal observation years.
- `n_marks::Int`: Number of tagged individuals released in mark-recapture telemetry.
- `area_method::Symbol`: Spatial tessellation method (`:hexagonal`, `:cvt`, `:voronoi`, `:grid`).
- `rng::AbstractRNG`: Random number generator instance.

# Returns
- `NamedTuple`:
  - `data::DataFrame`: Long-format observation table with columns `(:density, :unit_id, ...)`.
  - `telemetry_data::DataFrame`: Long-format telemetry observations with columns `(:tagid, ...)`.
  - `au::NamedTuple`: Spatial areal units object containing boundaries, centroids, and `W`.
  - `n_spatial::Int`: Number of spatial units.
  - `n_years::Int`: Number of temporal survey periods.
"""
function generate_ADR_simulation_bundle(
    domain_size::Real,
    n_units::Int,
    n_years::Int,
    n_marks::Int;
    area_method::Symbol = :cvt,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
)
    # 1. Generate spatial partitioning using BSTM's partitioning engine
    s_x_init = rand(rng, 1000) .* domain_size
    s_y_init = rand(rng, 1000) .* domain_size
    
    au = assign_spatial_units(
        s_x_init, s_y_init;
        area_method = area_method,
        target_units = n_units,
        exact_units = false
    )
    
    centroids = au.centroids
    n_spatial = length(centroids)
    cent_x = [c[1] for c in centroids]
    cent_y = [c[2] for c in centroids]

    # 2. Dynamic Spatiotemporal Habitat Suitability Index (HSI)
    habitat_p = zeros(Float64, n_spatial, n_years)
    center_x = domain_size / 2.0
    center_y = domain_size / 2.0
    
    p_init = [exp(-sqrt((cent_x[i] - center_x)^2 + (cent_y[i] - center_y)^2) / (domain_size
      / 3.0)) for i in 1:n_spatial]
    habitat_p[:, 1] = p_init ./ (maximum(p_init) + 1e-9)
    
    for t in 2:n_years
        noise = randn(rng, n_spatial) .* 0.05
        habitat_p[:, t] = min.(1.0, max.(0.01, habitat_p[:, t-1] .+ noise))
    end

    # 3. Simulate Spatiotemporal Density Observations
    density_n = zeros(Float64, n_spatial, n_years)
    for t in 1:n_years
        density_n[:, t] = simulate_correlated_density_vector(habitat_p[:, t], 0.75, 3.5,
          0.6; rng=rng)
    end

    # 4. Simulate Individual Mark-Recapture Telemetry Transitions
    telemetry_df = DataFrame(
        tagid = Int[],
        s_idx = Int[],
        time = Float64[],
        tag = Int[],
        individual_covariate = Float64[]
    )
    
    for i in 1:n_marks
        release_unit = rand(rng, 1:n_spatial)
        year_start = rand(rng, 1:max(1, n_years - 2))
        time_steps = rand(rng, 1:2)
        
        # Simulate physical dispersal distance
        d_travel = rand(rng, Uniform(domain_size / 20.0, domain_size / 2.5)) * time_steps
        dists = [abs(sqrt((cent_x[release_unit] - cent_x[j])^2 + (cent_y[release_unit] -
          cent_y[j])^2) - d_travel) for j in 1:n_spatial]
        recapture_unit = argmin(dists)
        
        ind_cov = randn(rng)

        # Release event (tag = 0)
        push!(telemetry_df, (tagid=i, s_idx=release_unit, time=Float64(year_start), tag=0,
          individual_covariate=ind_cov))
        # Recapture event (tag = 1)
        push!(telemetry_df, (tagid=i, s_idx=recapture_unit, time=Float64(year_start +
          time_steps), tag=1, individual_covariate=ind_cov))
    end

    # 5. Build Long-Format Survey DataFrame
    df = DataFrame()
    for t in 1:n_years
        temp_df = DataFrame(
            density = density_n[:, t],
            unit_id = 1:n_spatial,
            s_idx = 1:n_spatial,
            year = t,
            time_idx = t,
            s_x = cent_x,
            s_y = cent_y,
            habitat_p = habitat_p[:, t]
        )
        append!(df, temp_df)
    end

    return (
        data = df,
        telemetry_data = telemetry_df,
        au = au,
        n_spatial = n_spatial,
        n_years = n_years
    )
end

"""
    compute_velocity_field(prob_vec, grid_dim, strength; mode=:exponential)

Computes an advection velocity vector field from a spatial gradient of habitat suitability.

# Arguments
- `prob_vec::AbstractVector{<:Real}`: Spatial habitat suitability values.
- `grid_dim::Int`: Dimension of regular grid (for lattice geometries).
- `strength::Real`: Scaling factor for advection velocity.
- `mode::Symbol`: `:exponential` (relative gradient) or `:linear` (absolute gradient).

# Returns
- `NamedTuple`: `(vx = vec(vx), vy = vec(vy))` velocity components.
"""
function compute_velocity_field(
    prob_vec::AbstractVector{<:Real},
    grid_dim::Int,
    strength::Real;
    mode::Symbol = :exponential
)
    grid = reshape(prob_vec, grid_dim, grid_dim)
    rows, cols = size(grid)
    vx = zeros(Float64, rows, cols)
    vy = zeros(Float64, rows, cols)
    eps_val = 1e-6

    for r in 1:rows, c in 1:cols
        gx = (c == 1) ? (grid[r, 2] - grid[r, 1]) : (c == cols ? (grid[r, cols] - grid[r,
          cols-1]) : (grid[r, c+1] - grid[r, c-1]) / 2.0)
        gy = (r == 1) ? (grid[2, c] - grid[1, c]) : (r == rows ? (grid[rows, c] -
          grid[rows-1, c]) : (grid[r+1, c] - grid[r-1, c]) / 2.0)

        if mode == :exponential
            denom = max(grid[r, c], eps_val)
            vx[r, c] = (gx / denom) * strength
            vy[r, c] = (gy / denom) * strength
        else
            vx[r, c] = gx * strength
            vy[r, c] = gy * strength
        end
    end
    return (vx = vec(vx), vy = vec(vy))
end

"""
    calculate_multistep_transition(Gamma_base::AbstractMatrix{T}, steps::Int) where T <: Real

Calculates the multi-step dispersal transition matrix via Markov matrix exponentiation:
\$\\Gamma^{(k)} = \\Gamma^k\$.
"""
function calculate_multistep_transition(Gamma_base::AbstractMatrix{T}, steps::Int) where T <: Real
    n_spatial = size(Gamma_base, 1)
    if steps < 1
        return Matrix{T}(I, n_spatial, n_spatial)
    end
    
    G_step = copy(Gamma_base)
    for i in 1:n_spatial
        row_sum = sum(G_step[i, :])
        if row_sum > 0
            G_step[i, :] ./= row_sum
        end
    end
    
    Gamma_k = G_step ^ steps
    
    for i in 1:n_spatial
        final_sum = sum(Gamma_k[i, :])
        if final_sum > 0
            Gamma_k[i, :] ./= final_sum
        end
    end
    
    return Gamma_k
end

"""
    simulate_posterior_trajectories(Gamma_base, start_units, n_steps, au_context;
                                   rho_persistence=0.0, rng=Random.GLOBAL_RNG)

Simulates individual movement trajectories from a stationary transition matrix \$\\Gamma\$,
with optional directional persistence (Correlated Random Walk / CRW).

# Arguments
- `Gamma_base::AbstractMatrix`: Transition probability matrix (\$S \\times S\$).
- `start_units::Vector{Int}`: Starting spatial unit indices for each tracked individual.
- `n_steps::Int`: Number of discrete forward movement steps.
- `au_context::NamedTuple`: Spatial units object containing `centroids`.
- `rho_persistence::Real`: Directional persistence parameter (\$\\rho \\ge 0\$).
- `rng::AbstractRNG`: Random number generator.

# Returns
- `Matrix{Int}`: Matrix of shape `(n_indiv, n_steps + 1)` with unit indices visited over time.
"""
function simulate_posterior_trajectories(
    Gamma_base::AbstractMatrix{T},
    start_units::Vector{Int},
    n_steps::Int,
    au_context::NamedTuple;
    rho_persistence::Real = 0.0,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
) where T <: Real
    
    n_indiv = length(start_units)
    n_spatial = size(Gamma_base, 1)
    centroids = au_context.centroids
    paths = zeros(Int, n_indiv, n_steps + 1)
    paths[:, 1] = start_units
    
    G_sampling = copy(Gamma_base)
    for i in 1:n_spatial
        rs = sum(G_sampling[i, :])
        if rs > 0
            G_sampling[i, :] ./= rs
        else
            G_sampling[i, i] = 1.0
        end
    end
    
    for i in 1:n_indiv
        curr_node = start_units[i]
        prev_node = 0
        for t in 2:(n_steps + 1)
            p_row = vec(G_sampling[curr_node, :])
            if rho_persistence > 0.0 && prev_node != 0
                v_prev = [centroids[curr_node][d] - centroids[prev_node][d] for d in 1:2]
                if norm(v_prev) > 1e-9
                    persistence_weights = ones(T, n_spatial)
                    for j in 1:n_spatial
                        if j == curr_node
                            continue
                        end
                        v_cand = [centroids[j][d] - centroids[curr_node][d] for d in 1:2]
                        n_cand = norm(v_cand)
                        if n_cand > 1e-9
                            cos_theta = dot(v_prev, v_cand) / (norm(v_prev) * n_cand)
                            persistence_weights[j] = exp(rho_persistence * cos_theta)
                        end
                    end
                    p_row = p_row .* persistence_weights
                    row_sum = sum(p_row)
                    if row_sum > 0
                        p_row ./= row_sum
                    else
                        p_row = vec(G_sampling[curr_node, :])
                    end
                end
            end
            next_node = rand(rng, Categorical(p_row))
            paths[i, t] = next_node
            prev_node = curr_node
            curr_node = next_node
        end
    end
    return paths
end

"""
    simulate_mechanistic_trajectories(Gamma_sequence, start_units, t_start, au_context;
                                      rho_persistence=0.0, n_years_sim=1,
                                      rng=Random.GLOBAL_RNG)

Simulates individual movement paths through a dynamic, non-stationary environment where transition
kernels \$\\Gamma_t\$ vary over time.
"""
function simulate_mechanistic_trajectories(
    Gamma_sequence::Vector{<:AbstractMatrix{T}},
    start_units::Vector{Int},
    t_start::Int,
    au_context::NamedTuple;
    rho_persistence::Real = 0.0,
    n_years_sim::Int = 1,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
) where T <: Real

    n_indiv = length(start_units)
    n_spatial = size(Gamma_sequence[1], 1)
    centroids = au_context.centroids
    max_available_time = length(Gamma_sequence)
    
    actual_steps = n_years_sim
    if t_start + n_years_sim > max_available_time
        actual_steps = max_available_time - t_start
    end

    paths = zeros(Int, n_indiv, actual_steps + 1)
    paths[:, 1] = start_units
    
    for i in 1:n_indiv
        curr_node = start_units[i]
        prev_node = 0
        for step in 1:actual_steps
            t_current = t_start + step - 1
            if t_current > max_available_time
                break
            end
            Gamma_t = Gamma_sequence[t_current]
            p_row = vec(Gamma_t[curr_node, :])
            if rho_persistence > 0.0 && prev_node != 0
                v_prev = [centroids[curr_node][d] - centroids[prev_node][d] for d in 1:2]
                if norm(v_prev) > 1e-9
                    persistence_weights = ones(T, n_spatial)
                    for j in 1:n_spatial
                        if j == curr_node
                            continue
                        end
                        v_cand = [centroids[j][d] - centroids[curr_node][d] for d in 1:2]
                        n_cand = norm(v_cand)
                        if n_cand > 1e-9
                            cos_theta = dot(v_prev, v_cand) / (norm(v_prev) * n_cand)
                            persistence_weights[j] = exp(rho_persistence * cos_theta)
                        end
                    end
                    p_row = p_row .* persistence_weights
                    row_sum = sum(p_row)
                    if row_sum > 0
                        p_row ./= row_sum
                    else
                        p_row = vec(Gamma_t[curr_node, :])
                    end
                end
            end
            next_node = rand(rng, Categorical(p_row))
            paths[i, step + 1] = next_node
            prev_node = curr_node
            curr_node = next_node
        end
    end
    return paths
end

"""
    compute_suitability_transition_kernel(suitability_vec, W;
                                          sensitivity=1.0, diffusion_weight=0.1)

Generates a spatial Markov transition kernel based on local habitat suitability differences:
\$\\Gamma_{ij} \\propto \\exp(\\beta \\cdot \\text{suitability}_j) + D_{\\text{weight}}\$.
"""
function compute_suitability_transition_kernel(
    suitability_vec::AbstractVector{T},
    W::AbstractMatrix;
    sensitivity::Real = 1.0,
    diffusion_weight::Real = 0.1
) where T <: Real

    n_spatial = length(suitability_vec)
    Gamma = spzeros(T, n_spatial, n_spatial)
    rows = rowvals(W)
    vals = nonzeros(W)
    
    for i in 1:n_spatial
        Gamma[i, i] = exp(sensitivity * suitability_vec[i])
        for j_idx in nzrange(W, i)
            j = rows[j_idx]
            if i == j
                continue
            end
            suitability_bias = exp(sensitivity * suitability_vec[j])
            edge_weight = vals[j_idx]
            Gamma[i, j] = (suitability_bias + diffusion_weight) * edge_weight
        end
    end
    
    for i in 1:n_spatial
        row_sum = sum(Gamma[i, :])
        if row_sum > 1e-12
            Gamma[i, :] ./= row_sum
        else
            Gamma[i, i] = 1.0
        end
    end
    return Gamma
end

"""
    calculate_regional_connectivity(Gamma, strata_definition)

Aggregates fine-scale spatial unit transition probabilities into a macro-regional
connectivity matrix.
"""
function calculate_regional_connectivity(Gamma::AbstractMatrix, strata_definition::AbstractVector)
    n_units = size(Gamma, 1)
    unique_strata = unique(strata_definition)
    n_strata = length(unique_strata)
    strata_map = Dict(s => i for (i, s) in enumerate(unique_strata))
    
    C = zeros(Float64, n_strata, n_strata)

    for i in 1:n_units
        from_stratum_idx = strata_map[strata_definition[i]]
        for j in 1:n_units
            to_stratum_idx = strata_map[strata_definition[j]]
            C[from_stratum_idx, to_stratum_idx] += Gamma[i, j]
        end
    end

    for r in 1:n_strata
        row_sum = sum(C[r, :])
        if row_sum > 0
            C[r, :] ./= row_sum
        end
    end
    return C
end

"""
    plot_ad_ratio_distribution(advection_field, diffusion_field)

Generates a diagnostic histogram of the Advection-to-Diffusion ratio across spatial units.
"""
function plot_ad_ratio_distribution(advection_field::AbstractVector{<:Real},
  diffusion_field::AbstractVector{<:Real})
    ratios = advection_field ./ (mean(diffusion_field) .+ 1e-6)
    plt = Plots.histogram(
        ratios, bins=25, title="Advection-to-Diffusion Ratio (Péclet-like)",
        xlabel="Ratio (Advection / Diffusion)", ylabel="Frequency",
        label="Spatial Units", color=:plum, linecolor=:white
    )
    Plots.vline!(plt, [1.0], color=:red, linestyle=:dash, linewidth=2.0, label="Equilibrium
      Threshold")
    return plt
end

"""
    synthesize_adr_results(chain_or_res, sim_data, vel_vectors; au=sim_data.au)

Performs comprehensive post-processing, parameter extraction, and diagnostic visualization
for a fitted Advection-Diffusion-Reaction movement model.

# Returns
- `NamedTuple`: Containing:
  - `parameters::NamedTuple`: Posterior mean estimates for velocity, diffusion, sigma.
  - `propagator_matrix::Matrix{Float64}`: Reconstructed forward propagator \$M_{\\text{prop}}\$.
  - `transition_matrix::Matrix{Float64}`: Reconstructed one-step Markov transition matrix.
  - `plots::NamedTuple`: Rendered diagnostic plots (`regional_connectivity`, etc.).
"""
function synthesize_adr_results(
    chain_or_res,
    sim_data::NamedTuple,
    vel_vectors::NamedTuple;
    au = sim_data.au
)
    # 1. Extract Posterior Parameter Estimates
    S_strength_mean = if hasproperty(chain_or_res, :effects) &&
      hasproperty(chain_or_res.effects, :velocity)
        chain_or_res.effects.velocity.mean
    elseif hasproperty(chain_or_res, :value) && :velocity_movement in names(chain_or_res,
      :parameters)
        mean(chain_or_res[:velocity_movement])
    else
        1.0
    end

    D_coeff_mean = if hasproperty(chain_or_res, :effects) &&
      hasproperty(chain_or_res.effects, :diffusion)
        chain_or_res.effects.diffusion.mean
    elseif hasproperty(chain_or_res, :value) && :diffusion_movement in names(chain_or_res,
      :parameters)
        mean(chain_or_res[:diffusion_movement])
    else
        0.5
    end

    # 2. Reconstruct Spatial Graph Operators
    W = au.W
    n_spatial = size(W, 1)
    
    # Graph Laplacian L = D_deg - W
    deg = vec(sum(W, dims=2))
    L = spdiagm(0 => deg) - W
    
    # Directed Advection Operator A
    W_dir = tril(W, -1)
    out_deg = vec(sum(W_dir, dims=2))
    D_inv = spdiagm(0 => 1.0 ./ (out_deg .+ 1e-9))
    A = D_inv * W_dir
    
    # Propagator M_prop = I - v*A - D*L
    M_prop_mean = Matrix(I(n_spatial) - (S_strength_mean .* A) - (D_coeff_mean .* L))
    Gamma_mean = inv(M_prop_mean)
    
    # Normalize rows of Gamma
    for i in 1:n_spatial
        rs = sum(Gamma_mean[i, :])
        if rs > 0
            Gamma_mean[i, :] ./= rs
        else
            Gamma_mean[i, i] = 1.0
        end
    end

    # 3. Visualization 1: Regional Connectivity Matrix
    cents_x = [c[1] for c in au.centroids]
    mid_x = (minimum(cents_x) + maximum(cents_x)) / 2.0
    strata = [x > mid_x ? "East" : "West" for x in cents_x]
    conn_mat = calculate_regional_connectivity(Gamma_mean, strata)
    
    plt_conn = Plots.heatmap(
        ["West", "East"], ["West", "East"], conn_mat,
        title = "Regional Transfer Probability Matrix",
        xlabel = "To Region", ylabel = "From Region", color = :viridis, clims = (0, 1)
    )

    # 4. Visualization 2: Advection-to-Diffusion Ratio Distribution
    advection_magnitude = sqrt.(vel_vectors.vx.^2 .+ vel_vectors.vy.^2) .* S_strength_mean
    diffusion_magnitude = fill(D_coeff_mean, n_spatial)
    plt_ad = plot_ad_ratio_distribution(advection_magnitude, diffusion_magnitude)

    # 5. Visualization 3: Path Simulation with & without Persistence
    random_starts = rand(1:n_spatial, min(4, n_spatial))
    n_sim_steps = 15
    paths_standard = simulate_posterior_trajectories(Gamma_mean, random_starts, n_sim_steps,
      au; rho_persistence=0.0)
    paths_persistent = simulate_posterior_trajectories(Gamma_mean, random_starts,
      n_sim_steps, au; rho_persistence=2.0)
    
    plt_comp = Plots.plot(layout=(1, 2), size=(900, 450), aspect_ratio=:equal)
    render_paths!(plt_comp[1], paths_standard; au=au, color=:crimson, lw=1.5, labels=["Mark
      $i" for i in 1:length(random_starts)])
    Plots.plot!(plt_comp[1], title="Standard Markovian Random Walk")
    
    render_paths!(plt_comp[2], paths_persistent; au=au, color=:navy, lw=1.5, labels=["Mark
      $i" for i in 1:length(random_starts)])
    Plots.plot!(plt_comp[2], title="Persistent Movement (\\rho=2.0)")

    # 6. Visualization 4: Dynamic Multi-Year Projections
    gamma_seq = [Gamma_mean for _ in 1:sim_data.n_years]
    dynamic_paths = simulate_mechanistic_trajectories(gamma_seq, random_starts, 1, au;
      rho_persistence=1.5, n_years_sim=min(4, sim_data.n_years - 1))
    
    plt_dyn = Plots.plot(aspect_ratio=:equal, title="Multi-Year Mechanistic Path
      Projections", legend=:outerright)
    render_paths!(plt_dyn, dynamic_paths; au=au, color=:darkgreen, lw=1.5)

    plots_bundle = (
        regional_connectivity = plt_conn,
        ad_ratio = plt_ad,
        paths_comparison = plt_comp,
        dynamic_paths = plt_dyn
    )

    params_bundle = (
        velocity_mean = S_strength_mean,
        diffusion_mean = D_coeff_mean
    )

    return (
        parameters = params_bundle,
        propagator_matrix = M_prop_mean,
        transition_matrix = Gamma_mean,
        plots = plots_bundle
    )
end
