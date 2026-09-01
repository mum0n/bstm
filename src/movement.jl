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
        gx = (c == 1) ? (grid[r, 2] - grid[r, 1]) :
             (c == cols ? (grid[r, cols] - grid[r, cols-1]) : ((grid[r, c+1] - grid[r, c-1]) / 2.0))
        gy = (r == 1) ? (grid[2, c] - grid[1, c]) :
             (r == rows ? (grid[rows, c] - grid[rows-1, c]) : ((grid[r+1, c] - grid[r-1, c]) / 2.0))

        if mode == :exponential
            denom_x = sqrt(grid[r, c] * (grid[r, c] + eps_val))
            denom_y = sqrt(grid[r, c] * (grid[r, c] + eps_val))
            vx[r, c] = (gx / denom_x) * strength
            vy[r, c] = (gy / denom_y) * strength
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
        G_sampling[i, :] .= max.(0.0, G_sampling[i, :])
        rs = sum(G_sampling[i, :])
        if rs > 1e-12
            G_sampling[i, :] ./= rs
        else
            G_sampling[i, :] .= 0.0
            G_sampling[i, i] = 1.0
        end
    end
    
    for i in 1:n_indiv
        curr_node = start_units[i]
        prev_node = 0
        for t in 2:(n_steps + 1)
            p_row = max.(0.0, vec(G_sampling[curr_node, :]))
            if rho_persistence > 0.0 && prev_node != 0
                v_prev = [centroids[curr_node][d] - centroids[prev_node][d] for d in 1:2]
                if norm(v_prev) > 1e-9
                    persistence_weights = ones(promote_type(T, typeof(rho_persistence)), n_spatial)
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
                    p_row = max.(0.0, p_row .* persistence_weights)
                end
            end
            row_sum = sum(p_row)
            if row_sum > 1e-12
                p_row ./= row_sum
            else
                p_row .= 0.0
                p_row[curr_node] = 1.0
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
            p_row = max.(0.0, vec(Gamma_t[curr_node, :]))
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
                    p_row = max.(0.0, p_row .* persistence_weights)
                end
            end
            row_sum = sum(p_row)
            if row_sum > 1e-12
                p_row ./= row_sum
            else
                p_row .= 0.0
                p_row[curr_node] = 1.0
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
    compute_suitability_transition_kernel(
        suitability_vec, W;
        sensitivity=1.0, diffusion_weight=0.1, relationship=:exponential
    )

Generates a spatial Markov transition probability kernel based on local habitat
suitability index (HSI) differences and network topology:
- `:exponential` (default): \$\\Gamma_{ij} \\propto \\exp(\\beta \\cdot \\text{HSI}_j) + D_{\\text{weight}}\$
- `:linear`: \$\\Gamma_{ij} \\propto (1 + \\beta \\cdot \\text{HSI}_j) + D_{\\text{weight}}\$
- `:logistic`: \$\\Gamma_{ij} \\propto \\frac{1}{1 + \\exp(-\\beta \\cdot \\text{HSI}_j)} + D_{\\text{weight}}\$

# Arguments
- `suitability_vec::AbstractVector{<:Real}`: Habitat Suitability Index (HSI) values (\$S\$).
- `W::AbstractMatrix`: Spatial adjacency weights matrix (\$S \\times S\$).
- `sensitivity::Real`: Advective response sensitivity (\$\\beta \\ge 0\$).
- `diffusion_weight::Real`: Baseline isotropic dispersal weight (\$D_{\\text{weight}} \\ge 0\$).
- `relationship::Symbol`: Functional relationship (`:exponential`, `:linear`, `:logistic`).

# Returns
- `SparseMatrixCSC{Float64, Int}`: Row-stochastic Markov transition probability matrix.
"""
function compute_suitability_transition_kernel(
    suitability_vec::AbstractVector{T},
    W::AbstractMatrix;
    sensitivity::Real = 1.0,
    diffusion_weight::Real = 0.1,
    relationship::Symbol = :exponential
) where T <: Real

    n_spatial = length(suitability_vec)
    Gamma = spzeros(T, n_spatial, n_spatial)
    rows = rowvals(W)
    vals = nonzeros(W)
    
    for i in 1:n_spatial
        h_i = suitability_vec[i]
        bias_i = if relationship == :exponential
            exp(sensitivity * h_i)
        elseif relationship == :logistic
            1.0 / (1.0 + exp(-sensitivity * h_i))
        else # :linear
            max(0.01, 1.0 + sensitivity * h_i)
        end
        Gamma[i, i] = bias_i

        for j_idx in nzrange(W, i)
            j = rows[j_idx]
            if i == j
                continue
            end
            h_j = suitability_vec[j]
            suitability_bias = if relationship == :exponential
                exp(sensitivity * h_j)
            elseif relationship == :logistic
                1.0 / (1.0 + exp(-sensitivity * h_j))
            else # :linear
                max(0.01, 1.0 + sensitivity * h_j)
            end
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
    plot_ad_ratio_distribution(advection_field, diffusion_field; mode=:plots)

Generates a diagnostic histogram of the Advection-to-Diffusion ratio across spatial units.
Supports `mode=:plots` (default Plots.Plot) or `mode=:leaflet` / `mode=:html` (interactive HTML).
"""
function plot_ad_ratio_distribution(advection_field::AbstractVector{<:Real},
  diffusion_field::AbstractVector{<:Real}; mode::Symbol=:plots)
    if mode == :leaflet || mode == :html
        return leaflet_ad_ratio_distribution(advection_field, diffusion_field)
    end
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
    # NOTE: Inverting M_prop to get a transition matrix assumes that the propagator 
    # defines a solvable linear system. However, for advection-diffusion on bounded domains, 
    # this may not be the intended transition kernel. The transition matrix should emerge 
    # directly from discretization, not from inverting a propagator.
    # Fix: Verify this is the intended mathematical model, or construct 
    # Gamma_mean directly from the advection/diffusion operators without inversion.

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

    # 7. Visualization 5: Comprehensive Interactive Leaflet Dashboard
    dash_html = leaflet_movement_dashboard(
        (au = au, transition_matrix = Gamma_mean, opts = (hsi = nothing,),
         d_val = D_coeff_mean, v_val = S_strength_mean),
        paths_persistent;
        strata = strata,
        title = "ADR Movement Estimation Dashboard"
    )

    plots_bundle = (
        regional_connectivity = plt_conn,
        ad_ratio = plt_ad,
        paths_comparison = plt_comp,
        dynamic_paths = plt_dyn,
        leaflet_dashboard = dash_html
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

# ==============================================================================
# SECTION: GENERAL TELEMETRY & POSTERIOR TRANSITION KERNEL UTILITIES
# ==============================================================================

"""
    validate_telemetry(df::DataFrame) -> nothing

Validates that a telemetry / mark-recapture DataFrame conforms to BSTM movement
schema requirements and logical consistency constraints.

# Validation Checks
1. Presence of required identifier, coordinate, and detection type columns:
   `:tagid`, `:lon`, `:lat`, `:tag`, and either `:timestamp` or `:time`.
2. Valid detection codes: `:tag ≥ 0` (where `0` indicates initial release/marking,
   and `n ≥ 1` indicates the \$n\$-th recapture event).
3. Individual trajectory consistency: exactly one release event (`tag == 0`) per
   `:tagid`, and all recapture events chronologically follow the initial release.

Throws an informative `ArgumentError` if any check fails.
"""
function validate_telemetry(df::DataFrame)
    time_col = if hasproperty(df, :time)
        :time
    elseif hasproperty(df, :timestamp)
        :timestamp
    else
        nothing
    end

    if isnothing(time_col)
        throw(ArgumentError("Telemetry DataFrame must contain either :time or :timestamp."))
    end

    required = [:tagid, :lon, :lat, :tag]
    missing_cols = filter(c -> !hasproperty(df, c), required)
    if !isempty(missing_cols)
        throw(ArgumentError("Telemetry DataFrame missing required columns: $(missing_cols)"))
    end

    if any(df.tag .< 0)
        throw(ArgumentError("Column :tag must be ≥ 0 (0 = initial mark, n ≥ 1 = recaptures)."))
    end

    for sub in groupby(df, :tagid)
        release_rows = filter(r -> r.tag == 0, sub)
        if nrow(release_rows) != 1
            tid = first(sub.tagid)
            throw(ArgumentError("tagid=$(tid): expected exactly one release event (tag=0), " *
                                "found $(nrow(release_rows))."))
        end
        release_time = Float64(release_rows[1, time_col])
        recapture_times = Float64.(filter(r -> r.tag > 0, sub)[!, time_col])
        if !all(recapture_times .> release_time)
            tid = first(sub.tagid)
            throw(ArgumentError("tagid=$(tid): one or more recaptures precede or coincide with release."))
        end
    end
    return nothing
end

"""
    map_telemetry_to_units(df::DataFrame, au::NamedTuple)::DataFrame

Projects observation coordinates `(:lon, :lat)` in a telemetry DataFrame to their nearest
spatial unit centroids, returning a copy of `df` with an appended `:s_idx` column.

# Mathematical Formulation
For each observation \$\\mathbf{x}_i = (\\text{lon}_i, \\text{lat}_i)\$:
\$s_i = \\arg\\min_{j \\in \\{1, \\dots, S\\}} \\|\\mathbf{x}_i - \\mathbf{c}_j\\|_2\$
where \$\\mathbf{c}_j\$ is the geographic centroid of spatial unit \$j\$.
"""
function map_telemetry_to_units(df::DataFrame, au::NamedTuple)::DataFrame
    out = copy(df)
    out.s_idx = map_to_units(df.lon, df.lat, au.centroids)
    return out
end

"""
    time_steps_between(t_release::Real, t_recapture::Real)::Int

Converts continuous decimal-year temporal difference to discrete integer time steps.
Enforces a minimum interval of 1 step: \$\\Delta t = \\max(1, \\lfloor t_{\\text{rec}} - t_{\\text{rel}} \\rceil)\$.
"""
function time_steps_between(t_release::Real, t_recapture::Real)::Int
    return max(1, round(Int, t_recapture - t_release))
end

"""
    extract_scalar_param(chain, param_prefix::String)::Vector{Float64}

Extracts posterior samples for a scalar parameter identified by `param_prefix`
(e.g., `"velocity"` or `"diffusion"`), resolving variable name suffixes dynamically
across MCMC chains, FlexiChains, or DataFrames.

# Suffix Resolution Precedence
1. Suffix matching formula indices: `"\$(param_prefix)_s_idx_t_idx"`
2. Suffix matching module name: `"\$(param_prefix)_movement"`
3. Exact parameter name: `param_prefix`
4. Any sampled parameter starting with `param_prefix`
"""
function extract_scalar_param(chain, param_prefix::String)::Vector{Float64}
    if chain isa NamedTuple || chain isa AbstractDict
        for k in keys(chain)
            sk = string(k)
            if sk == param_prefix || sk == "$(param_prefix)_s_idx_t_idx" ||
               sk == "$(param_prefix)_movement" || startswith(sk, param_prefix)
                val = chain[k]
                return isa(val, AbstractVector) ? Float64.(val) : [Float64(val)]
            end
        end
    end

    p_names = if occursin("FlexiChain", string(typeof(chain)))
        string.(keys(chain))
    elseif hasmethod(names, Tuple{typeof(chain), Symbol})
        string.(names(chain, :parameters))
    elseif hasmethod(names, Tuple{typeof(chain)})
        string.(names(chain))
    elseif hasmethod(keys, Tuple{typeof(chain)})
        string.(keys(chain))
    else
        String[]
    end

    target = ""
    for candidate in [
        "$(param_prefix)_s_idx_t_idx",
        "$(param_prefix)_movement",
        param_prefix
    ]
        matched = _find_parameter(p_names, candidate, 1, false)
        if !isempty(matched)
            target = matched
            break
        end
    end

    if isempty(target)
        idx = findfirst(n -> startswith(n, param_prefix), p_names)
        if !isnothing(idx)
            target = p_names[idx]
        end
    end

    if isempty(target)
        error("Could not find parameter matching prefix '$(param_prefix)' in MCMC chain.")
    end

    return get_params_vector(chain, target, 1)[:, 1]
end

const _extract_scalar_param = extract_scalar_param

"""
    reconstruct_posterior_kernel(
        chain,
        W::AbstractMatrix;
        hsi::Union{Nothing, AbstractVector}=nothing,
        relationship::Symbol=:exponential
    )::Matrix{Float64}

Reconstructs the posterior-mean row-stochastic Markov transition matrix \$\\bar{\\mathbf{\\Gamma}}\$
across all MCMC samples.

# Mathematical Formulation
For each posterior sample \$s \\in \\{1, \\dots, N_s\\}\$:
\$\\mathbf{M}_s = \\mathbf{I} - v_s \\mathbf{A} - D_s \\mathbf{L}\$
\$\\mathbf{\\Gamma}_s = \\text{row\\_normalize}\\left(\\mathbf{M}_s^{-1}\\right)\$
\$\\bar{\\mathbf{\\Gamma}} = \\frac{1}{N_s} \\sum_{s=1}^{N_s} \\mathbf{\\Gamma}_s\$

where \$\\mathbf{L} = \\text{diag}(\\mathbf{W} \\mathbf{1}) - \\mathbf{W}\$ is the graph Laplacian,
and \$\\mathbf{A}\$ is the directed advection operator derived from \$\\nabla \\text{HSI}\$
(or spatial topology).

# Arguments
- `chain`: MCMC posterior chain (Turing `Chains` or `FlexiChain`).
- `W::AbstractMatrix`: Adjacency / spatial weights matrix (\$S \\times S\$).
- `hsi::Union{Nothing, AbstractVector}`: Optional Habitat Suitability Index vector of length \$S\$.
- `relationship::Symbol`: Functional form for HSI gradient: `:exponential`, `:logistic`, `:linear`.

# Returns
- `Matrix{Float64}`: Row-stochastic \$S \\times S\$ Markov transition probability matrix.
"""
function reconstruct_posterior_kernel(
    chain,
    W::AbstractMatrix;
    hsi::Union{Nothing, AbstractVector}=nothing,
    relationship::Symbol=:exponential
)::Matrix{Float64}
    S = size(W, 1)

    deg = vec(sum(W, dims=2))
    L = spdiagm(0 => deg) - W

    if !isnothing(hsi)
        hsi_vec = Float64.(hsi)
        W_dir = spzeros(Float64, S, S)
        rows = rowvals(W)
        vals = nonzeros(W)
        for i in 1:S, j_idx in nzrange(W, i)
            j = rows[j_idx]
            i == j && continue
            dh = hsi_vec[j] - hsi_vec[i]
            dh > 0.0 || continue
            W_dir[i, j] = if relationship == :exponential
                vals[j_idx] * exp(dh)
            elseif relationship == :logistic
                vals[j_idx] / (1.0 + exp(-4.0 * dh))
            else
                vals[j_idx] * dh
            end
        end
        out_deg = vec(sum(W_dir, dims=2))
        D_inv = spdiagm(0 => [od > 1e-12 ? 1.0 / od : 0.0 for od in out_deg])
        A_base = Matrix(D_inv * W_dir)
    else
        W_dir = tril(W, -1)
        out_deg = vec(sum(W_dir, dims=2))
        D_inv = spdiagm(0 => 1.0 ./ (out_deg .+ 1e-9))
        A_base = Matrix(D_inv * W_dir)
    end

    L_dense = Matrix(L)
    I_S = Matrix{Float64}(I, S, S)

    v_samps = extract_scalar_param(chain, "velocity")
    d_samps = extract_scalar_param(chain, "diffusion")
    n_s = length(v_samps)

    Gamma_acc = zeros(Float64, S, S)
    for s in 1:n_s
        M_prop = I_S .- (v_samps[s] .* A_base) .- (d_samps[s] .* L_dense)
        Gamma_s = inv(M_prop)
        for i in 1:S
            Gamma_s[i, :] .= max.(0.0, Gamma_s[i, :])
            rs = sum(Gamma_s[i, :])
            if rs > 1e-12
                Gamma_s[i, :] ./= rs
            else
                Gamma_s[i, :] .= 0.0
                Gamma_s[i, i] = 1.0
            end
        end
        Gamma_acc .+= Gamma_s
    end
    Gamma_out = Gamma_acc ./ max(1, n_s)
    for i in 1:S
        Gamma_out[i, :] .= max.(0.0, Gamma_out[i, :])
        rs = sum(Gamma_out[i, :])
        if rs > 1e-12
            Gamma_out[i, :] ./= rs
        else
            Gamma_out[i, :] .= 0.0
            Gamma_out[i, i] = 1.0
        end
    end
    return Gamma_out
end

const _reconstruct_posterior_kernel = reconstruct_posterior_kernel

"""
    reshard_hsi_field(
        hsi_raw::AbstractVector{<:Real},
        au_dest::NamedTuple;
        au_src::Union{Nothing, NamedTuple} = nothing,
        hsi_coords::Union{Nothing, Vector{Tuple{Float64, Float64}}} = nothing,
        hsi_area_method::Symbol = :grid,
        domain_bbox::Union{Nothing, Tuple{Float64, Float64, Float64, Float64}} = nothing
    )::Vector{Float64}

Reshards an input Habitat Suitability Index (HSI) vector or surface from an
arbitrary source spatial geometry onto the destination spatial tessellation
`au_dest` (defaulting to `:hexagonal`).

# Mathematical Formulation
Let ``\\mathbf{h}_{\\text{src}} \\in [0, 1]^{S_{\\text{src}}}`` be the source HSI values
and ``\\mathbf{P} \\in \\mathbb{R}^{S_{\\text{dest}} \\times S_{\\text{src}}}`` be the
spatial interpolation / area-overlap transfer matrix between `au_src` and `au_dest`:
```math
\\mathbf{h}_{\\text{dest}} = \\text{clamp}\\left(\\mathbf{P} \\, \\mathbf{h}_{\\text{src}}, \\, 0.0, \\, 1.0\\right)
```

# Arguments
- `hsi_raw::AbstractVector{<:Real}`: Raw source HSI values of length ``S_{\\text{src}}``.
- `au_dest::NamedTuple`: Destination spatial tessellation (e.g., hexagonal or CVT units).
- `au_src::Union{Nothing, NamedTuple}`: Explicit source spatial tessellation with `:centroids` and `:polygons`.
- `hsi_coords::Union{Nothing, Vector{Tuple{Float64, Float64}}}`: Explicit coordinate points for each source HSI unit.
- `hsi_area_method::Symbol`: Source tessellation geometry when constructing from coordinates/bbox (`:grid`, `:cvt`, `:voronoi`, `:hexagonal`). Default: `:grid`.
- `domain_bbox::Union{Nothing, Tuple{Float64, Float64, Float64, Float64}}`: Optional domain bounding box `(min_lon, max_lon, min_lat, max_lat)`.

# Returns
- `Vector{Float64}`: Resharded HSI vector aligned with `au_dest.centroids` (length ``S_{\\text{dest}}``).
"""
function reshard_hsi_field(
    hsi_raw         :: AbstractVector{<:Real},
    au_dest         :: NamedTuple;
    au_src          :: Union{Nothing, NamedTuple} = nothing,
    hsi_coords      :: Union{Nothing, Vector{Tuple{Float64, Float64}}} = nothing,
    hsi_area_method :: Symbol = :grid,
    domain_bbox     :: Union{Nothing, Tuple{Float64, Float64, Float64, Float64}} = nothing
)::Vector{Float64}
    S_dest = length(au_dest.centroids)
    S_src = length(hsi_raw)

    # 1. If explicit source tessellation is provided
    if !isnothing(au_src) && length(au_src.centroids) == S_src
        hsi_dest = reshard_spatial_field(hsi_raw, au_src, au_dest)
        return clamp.(Vector{Float64}(hsi_dest), 0.0, 1.0)
    end

    # 2. If source and destination sizes match and no distinct geometry/coords specified
    if S_src == S_dest && isnothing(hsi_coords)
        return clamp.(Float64.(hsi_raw), 0.0, 1.0)
    end

    # 3. Determine source points for interpolation
    src_points = if !isnothing(hsi_coords) && length(hsi_coords) == S_src
        hsi_coords
    else
        # Construct synthetic regular grid over destination domain bounding box
        dest_xs = [c[1] for c in au_dest.centroids]
        dest_ys = [c[2] for c in au_dest.centroids]
        min_x = !isnothing(domain_bbox) ? domain_bbox[1] : minimum(dest_xs)
        max_x = !isnothing(domain_bbox) ? domain_bbox[2] : maximum(dest_xs)
        min_y = !isnothing(domain_bbox) ? domain_bbox[3] : minimum(dest_ys)
        max_y = !isnothing(domain_bbox) ? domain_bbox[4] : maximum(dest_ys)

        pad_x = (max_x - min_x) * 0.05
        pad_y = (max_y - min_y) * 0.05
        grid_side = ceil(Int, sqrt(S_src))
        gx = range(min_x - pad_x, max_x + pad_x, length=grid_side)
        gy = range(min_y - pad_y, max_y + pad_y, length=grid_side)
        all_pts = Tuple{Float64, Float64}[(x, y) for x in gx for y in gy]
        all_pts[1:S_src]
    end

    # 4. Inverse Distance Weighting (k-d Tree) from source points to destination centroids
    src_coords_mat = hcat([[c[1], c[2]] for c in src_points]...)
    kdtree = KDTree(src_coords_mat)
    k_nn = min(4, S_src)

    hsi_dest = zeros(Float64, S_dest)
    for j in 1:S_dest
        dest_pt = [au_dest.centroids[j][1], au_dest.centroids[j][2]]
        idxs, dists = knn(kdtree, dest_pt, k_nn, true)
        if any(dists .< 1e-9)
            exact_idx = idxs[findfirst(dists .< 1e-9)]
            hsi_dest[j] = Float64(hsi_raw[exact_idx])
        else
            inv_d = 1.0 ./ dists
            w = inv_d ./ sum(inv_d)
            hsi_dest[j] = sum(w .* hsi_raw[idxs])
        end
    end

    return clamp.(hsi_dest, 0.0, 1.0)
end

# ==============================================================================
# SECTION: SNOW CRAB TELEMETRY & MARK-RECAPTURE DATA INGESTION & PROCESSING
# ==============================================================================

"""
    haversine_distance(lon1::Real, lat1::Real, lon2::Real, lat2::Real; radius::Real=6378137.0)::Float64

Computes the great-circle geodesic distance in metres between two points `(lon1, lat1)`
and `(lon2, lat2)` in decimal degrees on a spherical Earth of radius `radius` (WGS84 mean radius).

# Mathematical Formula
\$\\Delta\\phi = \\text{deg2rad}(\\text{lat}_2 - \\text{lat}_1), \\quad \\Delta\\lambda = \\text{deg2rad}(\\text{lon}_2 - \\text{lon}_1)\$
\$a = \\sin^2\\left(\\frac{\\Delta\\phi}{2}\\right) + \\cos(\\text{lat}_1) \\cos(\\text{lat}_2) \\sin^2\\left(\\frac{\\Delta\\lambda}{2}\\right)\$
\$d = 2 R \\operatorname{atan2}\\left(\\sqrt{a}, \\sqrt{1-a}\\right)\$
"""
function haversine_distance(
    lon1::Real, lat1::Real, lon2::Real, lat2::Real;
    radius::Real=6378137.0
)::Float64
    if !isfinite(lon1) || !isfinite(lat1) || !isfinite(lon2) || !isfinite(lat2)
        return NaN
    end
    phi1 = deg2rad(Float64(lat1))
    phi2 = deg2rad(Float64(lat2))
    dphi = deg2rad(Float64(lat2 - lat1))
    dlam = deg2rad(Float64(lon2 - lon1))

    a = sin(dphi / 2.0)^2 + cos(phi1) * cos(phi2) * sin(dlam / 2.0)^2
    a = clamp(a, 0.0, 1.0)
    return 2.0 * radius * atan(sqrt(a), sqrt(1.0 - a))
end

"""
    tag_to_study_id(tag_id)::Union{Int, Nothing}

Maps a numeric or alphanumeric snow crab tag identifier to its corresponding historical
study index (1 to 57) based on the Scotian Shelf and Gulf historical tagging metadata.

# Arguments
- `tag_id`: Integer, Float, or String representation of the tag number.

# Returns
- `Int`: Study identifier between 1 and 57, or `nothing` if unmapped.
"""
function tag_to_study_id(tag_id)::Union{Int, Nothing}
    if ismissing(tag_id) || isnothing(tag_id)
        return nothing
    end
    raw_str = string(tag_id)
    # Strip leading 'G' or 'g' (e.g. "G1234" -> "1234")
    clean_str = replace(raw_str, r"^[Gg]" => "")

    # Handle prefixed t-tags (e.g. "t1605")
    if startswith(clean_str, "t") || startswith(clean_str, "T")
        t_num = tryparse(Int, clean_str[2:end])
        if !isnothing(t_num)
            if 1605 <= t_num <= 1659
                return 49
            elseif (1217 <= t_num <= 1350) || (1601 <= t_num <= 1603)
                return 52
            elseif 1676 <= t_num <= 1694
                return 56
            elseif 3026 <= t_num <= 3175
                return 57
            end
        end
    end

    # Handle prefixed s-tags (e.g. "s99102")
    if startswith(clean_str, "s") || startswith(clean_str, "S")
        s_num = tryparse(Int, clean_str[2:end])
        if !isnothing(s_num) && 99102 <= s_num <= 99196
            return 57
        end
    end

    # Parse numeric integer tag ID
    id = tryparse(Int, clean_str)
    if isnothing(id)
        f_num = tryparse(Float64, clean_str)
        if !isnothing(f_num)
            id = round(Int, f_num)
        else
            return nothing
        end
    end

    if 0 <= id <= 600
        return 1
    elseif 2350 <= id <= 2399
        return 5
    elseif 1000 <= id <= 1600
        return 6
    elseif (2401 <= id <= 2403) || (2411 <= id <= 2450)
        return 7
    elseif 1601 <= id <= 2349
        return 8
    elseif 6000 <= id <= 6349
        return 9
    elseif 2716 <= id <= 2850
        return 10
    elseif 2456 <= id <= 2715
        return 11
    elseif 5050 <= id <= 5099
        return 12
    elseif 5100 <= id <= 5149
        return 13
    elseif (2851 <= id <= 2900) || (3051 <= id <= 3262) || (3446 <= id <= 3545)
        return 14
    elseif (3263 <= id <= 3444) || (3546 <= id <= 3600)
        return 15
    elseif 4000 <= id <= 4246
        return 16
    elseif 7450 <= id <= 7542
        return 17
    elseif 7543 <= id <= 7699
        return 18
    elseif (4798 <= id <= 4999) || (5250 <= id <= 5520) || (6350 <= id <= 6999)
        return 19
    elseif 5521 <= id <= 5808
        return 20
    elseif 4250 <= id <= 4480
        return 21
    elseif 4482 <= id <= 4797
        return 22
    elseif (7000 <= id <= 7449) || (7700 <= id <= 7999)
        return 23
    elseif (5809 <= id <= 5999) || (8501 <= id <= 8570) || (9229 <= id <= 9649)
        return 24
    elseif 8571 <= id <= 8828
        return 25
    elseif 8829 <= id <= 9228
        return 26
    elseif (10482 <= id <= 10499) || (11082 <= id <= 11334) || (11350 <= id <= 11387)
        return 27
    elseif id == 4301 || (8040 <= id <= 8050) || (8137 <= id <= 8150) ||
           (8201 <= id <= 8298) || (9684 <= id <= 9741) || (9748 <= id <= 9784) ||
           (10254 <= id <= 10406) || (10468 <= id <= 10481) || (11039 <= id <= 11081)
        return 28
    elseif (4248 <= id <= 4249) || (8000 <= id <= 8039) || (8051 <= id <= 8068) ||
           (8070 <= id <= 8136) || (8151 <= id <= 8200) || (10407 <= id <= 10467) ||
           (11000 <= id <= 11036)
        return 29
    elseif (8299 <= id <= 8500) || (9650 <= id <= 9683)
        return 30
    elseif 10032 <= id <= 10253
        return 31
    elseif (9742 <= id <= 9747) || (9785 <= id <= 10031)
        return 32
    elseif (10501 <= id <= 10530) || (10609 <= id <= 10716)
        return 33
    elseif (10531 <= id <= 10608) || (10718 <= id <= 10798)
        return 34
    elseif (10800 <= id <= 10949) || (11400 <= id <= 11449)
        return 35
    elseif (10950 <= id <= 10999) || (11335 <= id <= 11349) || (11388 <= id <= 11399) ||
           (11450 <= id <= 11499) || (13000 <= id <= 13149)
        return 36
    elseif 13150 <= id <= 13490
        return 37
    elseif (13491 <= id <= 13649) || (15650 <= id <= 15673)
        return 38
    elseif 13650 <= id <= 13834
        return 39
    elseif 13835 <= id <= 14109
        return 40
    elseif 14110 <= id <= 14226
        return 41
    elseif (14227 <= id <= 14306) || (14308 <= id <= 14311) || (14313 <= id <= 14314) ||
           id == 14316 || (14318 <= id <= 14345) || (14347 <= id <= 14430)
        return 42
    elseif 14431 <= id <= 14880
        return 43
    elseif (15750 <= id <= 15779) || (15800 <= id <= 15899) || (15950 <= id <= 15999)
        return 44
    elseif (15780 <= id <= 15799) || (15900 <= id <= 15949)
        return 45
    elseif 14881 <= id <= 14951
        return 46
    elseif 14952 <= id <= 15001
        return 47
    elseif 15002 <= id <= 15199
        return 48
    elseif 15400 <= id <= 15499
        return 49
    elseif (15500 <= id <= 15649) || (15674 <= id <= 15749) || (16000 <= id <= 16711)
        return 50
    elseif (16712 <= id <= 17199) || (17300 <= id <= 17399) || (17500 <= id <= 17607)
        return 51
    elseif 15200 <= id <= 15399
        return 52
    elseif (17608 <= id <= 17849) || id == 17953
        return 53
    elseif (17200 <= id <= 17299) || (17850 <= id <= 17999)
        return 54
    elseif 18043 <= id <= 18099
        return 55
    elseif (17401 <= id <= 17499) || (18000 <= id <= 18042) || (18100 <= id <= 18158)
        return 56
    elseif 18226 <= id <= 18326
        return 57
    end

    return nothing
end

"""
    filter_dead_tags(df::DataFrame; time_threshold_days::Real=30.0,
                     dist_threshold_meters::Real=50.0)::DataFrame

Identifies and flags acoustic telemetry records belonging to a terminal stationary period,
which indicates a deceased animal or shed tag (e.g. tag stationary within `dist_threshold_meters`
for \$\\ge\$ `time_threshold_days` continuously until the end of detections).

# Arguments
- `df::DataFrame`: Telemetry table with columns `:tagid`, `:lon`, `:lat`, and `:timestamp`
  (or `:time`).
- `time_threshold_days::Real`: Minimum duration in days of terminal non-movement (default 30.0).
- `dist_threshold_meters::Real`: Maximum radius in metres from terminal location (default 50.0).

# Returns
- `DataFrame`: A copy of `df` with an added boolean column `:is_dead`.
"""
function filter_dead_tags(
    df::DataFrame;
    time_threshold_days::Real=30.0,
    dist_threshold_meters::Real=50.0
)::DataFrame
    out = copy(df)
    time_col = hasproperty(out, :timestamp) ? :timestamp : :time
    if !hasproperty(out, time_col) || !hasproperty(out, :tagid) ||
       !hasproperty(out, :lon) || !hasproperty(out, :lat)
        error("DataFrame must contain :tagid, :lon, :lat, and :timestamp or :time.")
    end

    t_days = if eltype(out[!, time_col]) <: Dates.TimeType
        [Dates.value(Dates.Millisecond(Dates.DateTime(t) - Dates.DateTime(1970, 1, 1))) / 8.64e7 for t in out[!, time_col]]
    else
        Float64.(out[!, time_col]) .* 365.25
    end
    out._t_days = t_days

    sort!(out, [:tagid, :_t_days])

    is_dead_flags = zeros(Bool, nrow(out))
    gdf = groupby(out, :tagid)

    for sub in gdf
        n_obs = nrow(sub)
        if n_obs == 0
            continue
        end

        row_indices = parentindices(sub)[1]
        last_lon = Float64(sub.lon[end])
        last_lat = Float64(sub.lat[end])
        last_t = sub._t_days[end]

        dists_to_last = [haversine_distance(Float64(sub.lon[i]), Float64(sub.lat[i]), last_lon, last_lat) for i in 1:n_obs]
        days_to_last = [last_t - sub._t_days[i] for i in 1:n_obs]

        cummax_dist = zeros(Float64, n_obs)
        curr_max = 0.0
        for i in n_obs:-1:1
            d = isfinite(dists_to_last[i]) ? dists_to_last[i] : Inf
            curr_max = max(curr_max, d)
            cummax_dist[i] = curr_max
        end

        in_terminal_cluster = [cummax_dist[i] <= dist_threshold_meters for i in 1:n_obs]
        cluster_durations = [in_terminal_cluster[i] ? days_to_last[i] : 0.0 for i in 1:n_obs]
        max_cluster_dur = maximum(cluster_durations)

        for i in 1:n_obs
            if in_terminal_cluster[i] && max_cluster_dur >= time_threshold_days
                is_dead_flags[row_indices[i]] = true
            end
        end
    end

    out.is_dead = is_dead_flags
    select!(out, Not(:_t_days))
    return out
end

"""
    _parse_flexible_date(date_val)::Union{Date, Nothing}

Parses date strings, DateTime, Date, or numeric timestamp representations safely.
"""
function _parse_flexible_date(date_val)::Union{Date, Nothing}
    if ismissing(date_val) || isnothing(date_val)
        return nothing
    end
    if date_val isa Date
        return date_val
    elseif date_val isa Dates.DateTime
        return Date(date_val)
    elseif date_val isa Real
        yr = round(Int, date_val)
        if 1950 <= yr <= 2050
            return Date(yr, 6, 1)
        end
    elseif date_val isa AbstractString
        s = strip(date_val)
        isempty(s) && return nothing
        m_iso = match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
        if !isnothing(m_iso)
            y, m, d = parse(Int, m_iso.captures[1]), parse(Int, m_iso.captures[2]), parse(Int, m_iso.captures[3])
            return Date(y, clamp(m, 1, 12), clamp(d, 1, 31))
        end
        m_us = match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})", s)
        if !isnothing(m_us)
            m, d, y = parse(Int, m_us.captures[1]), parse(Int, m_us.captures[2]), parse(Int, m_us.captures[3])
            return Date(y, clamp(m, 1, 12), clamp(d, 1, 31))
        end
        dt = tryparse(DateTime, s)
        if !isnothing(dt)
            return Date(dt)
        end
    end
    return nothing
end

"""
    _date_to_decimal_year(d::Date)::Float64

Converts a `Date` to a continuous decimal year representation.
"""
function _date_to_decimal_year(d::Date)::Float64
    yr = Dates.year(d)
    doy = Dates.dayofyear(d)
    days_in_yr = Dates.isleapyear(yr) ? 366.0 : 365.0
    return Float64(yr) + (Float64(doy) - 1.0) / days_in_yr
end

_date_to_decimal_year(dt::Dates.TimeType)::Float64 = _date_to_decimal_year(Date(dt))



"""
    summarize_tag_activity(df::DataFrame)::DataFrame

Calculates summary statistics per individual tag from a telemetry / mark-recapture DataFrame.

# Computed Metrics per Tag
- `duration_days::Float64`: Elapsed days from first to last observation.
- `cw_change::Float64`: Net carapace width growth in mm (`last_cw - first_cw`).
- `cc_change::Float64`: Net carapace condition progression.
- `total_dist_m::Float64`: Cumulative physical track distance in metres across consecutive pings.
- `n_points::Int`: Total number of detection records.
"""
function summarize_tag_activity(df::DataFrame)::DataFrame
    time_col = hasproperty(df, :timestamp) ? :timestamp : (hasproperty(df, :time) ? :time : nothing)
    if isnothing(time_col) || !hasproperty(df, :tagid)
        error("Input DataFrame must have :tagid and either :timestamp or :time.")
    end

    gdf = groupby(df, :tagid)
    summary_rows = []

    for sub in gdf
        tid = first(sub.tagid)
        n_pts = nrow(sub)

        times_parsed = [_parse_flexible_date(t) for t in sub[!, time_col]]
        valid_times = filter(!isnothing, times_parsed)
        dur_days = if length(valid_times) > 1
            Float64(Dates.value(maximum(valid_times) - minimum(valid_times)))
        else
            0.0
        end

        cw_raw = hasproperty(sub, :cw) ? [tryparse(Float64, string(x)) for x in sub.cw] : nothing
        cw_vals = !isnothing(cw_raw) ? Float64[x for x in cw_raw if !isnothing(x) && isfinite(x)] : Float64[]
        cw_chg = length(cw_vals) > 1 ? cw_vals[end] - cw_vals[1] : NaN

        cc_raw = hasproperty(sub, :cc) ? [tryparse(Float64, string(c)) for c in sub.cc] : nothing
        cc_vals = !isnothing(cc_raw) ? Float64[x for x in cc_raw if !isnothing(x) && isfinite(x)] : Float64[]
        cc_chg = length(cc_vals) > 1 ? cc_vals[end] - cc_vals[1] : NaN

        tot_dist = 0.0
        if hasproperty(sub, :lon) && hasproperty(sub, :lat) && n_pts > 1
            for i in 1:(n_pts - 1)
                lon1, lat1 = Float64(sub.lon[i]), Float64(sub.lat[i])
                lon2, lat2 = Float64(sub.lon[i+1]), Float64(sub.lat[i+1])
                d = haversine_distance(lon1, lat1, lon2, lat2)
                if isfinite(d)
                    tot_dist += d
                end
            end
        end

        push!(summary_rows, (
            tagid = tid,
            duration_days = dur_days,
            cw_change = cw_chg,
            cc_change = cc_chg,
            total_dist_m = tot_dist,
            n_points = n_pts
        ))
    end

    return DataFrame(summary_rows)
end

"""
    sample_markov_bridge(Gamma::AbstractMatrix{<:Real},
                         u_start::Integer, u_end::Integer,
                         n_steps::Integer;
                         graph::Union{Nothing, SimpleGraph} = nothing,
                         W::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
                         hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
                         powers::Union{Nothing, Vector{<:AbstractMatrix}} = nothing,
                         rng::Random.AbstractRNG = Random.GLOBAL_RNG)::Vector{Int}

Samples a discrete Markov state-space trajectory ``z_0 = u_{\\text{start}} \\to z_1 \\to
\\dots \\to z_T = u_{\\text{end}}`` conditioned on fixed start and terminal endpoints under
the transition kernel ``\\boldsymbol{\\Gamma}`` while strictly respecting the marine domain
and coastline movement boundaries of adjacency graph ``G(V, E)``.

# Mathematical Formulation:
Let ``G(V, E)`` be the topological water graph where edges ``(i, j) \\in E`` connect
only physically adjacent water units across marine boundaries.
1. The shortest geodesic graph distance ``d_G(u_{\\text{start}}, u_{\\text{end}})`` is computed via
   ``A^*`` search. To prevent artificial jumps across land barriers or peninsulas (e.g. Cape Breton),
   the effective trajectory step count satisfies:
   ```math
   T_{\\text{eff}} = \\max(n_{\\text{steps}}, d_G(u_{\\text{start}}, u_{\\text{end}}))
   ```
2. For each intermediate step ``t \\in \\{1, \\dots, T_{\\text{eff}}-1\\}``, given previous state
   ``z_{t-1} = i`` and destination ``u_{\\text{end}}``, transitions are sampled strictly from
   valid marine graph neighbors ``j \\in \\mathcal{N}(i) \\cup \\{i\\}`` via Chapman-Kolmogorov:
   ```math
   P(z_t = j \\mid z_{t-1} = i, z_T = u_{\\text{end}}) = \\frac{T_{ij} \\, (\\mathbf{T}^{T_{\\text{eff}}-t})_{j, u_{\\text{end}}}}{(\\mathbf{T}^{T_{\\text{eff}}-t+1})_{i, u_{\\text{end}}}}
   ```
   where ``T_{ij} > 0`` only if ``(i, j) \\in E`` or ``i = j``.

# Inputs:
- `Gamma`: ``S \\times S`` transition matrix.
- `u_start`: Initial spatial unit index ``1 \\le u_{\\text{start}} \\le S``.
- `u_end`: Terminal spatial unit index ``1 \\le u_{\\text{end}} \\le S``.
- `n_steps`: Requested discrete time steps ``T \\ge 1``.
- `graph`: Optional `SimpleGraph` topology for marine coastline path routing.
- `W`: Optional ``S \\times S`` sparse adjacency matrix.
- `hsi`: Optional spatial HSI suitability vector for directional advective weighting.
- `powers`: Precomputed matrix powers of the 1-step kernel.
- `rng`: Random number generator.

# Outputs:
- `path::Vector{Int}`: Sequence of visited spatial unit indices strictly on the marine graph.
"""
function sample_markov_bridge(
    Gamma::AbstractMatrix{<:Real},
    u_start::Integer,
    u_end::Integer,
    n_steps::Integer;
    graph::Union{Nothing, SimpleGraph} = nothing,
    W::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
    powers::Union{Nothing, Vector{<:AbstractMatrix}} = nothing,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
)::Vector{Int}
    u_s = Int(u_start)
    u_e = Int(u_end)
    S = size(Gamma, 1)

    if u_s == u_e
        return fill(u_s, max(2, n_steps + 1))
    end

    # If graph is provided, ensure path stays strictly on valid water edges
    if !isnothing(graph)
        sp = a_star(graph, u_s, u_e)
        sp_nodes = isempty(sp) ? [u_s, u_e] : vcat([src(e) for e in sp], [dst(last(sp))])
        d_min = length(sp_nodes) - 1

        effective_steps = max(n_steps, d_min)
        if effective_steps <= d_min
            return sp_nodes
        end

        # Build 1-step water transition matrix T if W is provided
        T_step = if !isnothing(W)
            T_mat = zeros(Float64, S, S)
            for i in 1:S
                nbrs = findall(>(0.0), W[i, :])
                if isempty(nbrs)
                    T_mat[i, i] = 1.0
                    continue
                end
                w_vec = if !isnothing(hsi)
                    [exp(0.5 * (Float64(hsi[j]) - Float64(hsi[i]))) for j in nbrs]
                else
                    ones(Float64, length(nbrs))
                end
                w_sum = sum(w_vec)
                w_norm = w_sum > 0 ? (w_vec ./ w_sum) : fill(1.0 / length(nbrs), length(nbrs))
                T_mat[i, i] = 0.35 # probability of remaining in cell during unit step
                for (idx, j) in enumerate(nbrs)
                    T_mat[i, j] = 0.65 * w_norm[idx]
                end
            end
            T_mat
        else
            Matrix{Float64}(Gamma)
        end

        # Precompute powers up to effective_steps
        P_mats = Vector{Matrix{Float64}}(undef, effective_steps)
        P_mats[1] = copy(T_step)
        for k in 2:effective_steps
            P_mats[k] = P_mats[k-1] * T_step
        end

        path = zeros(Int, effective_steps + 1)
        path[1] = u_s
        path[end] = u_e
        curr = u_s

        for t in 1:(effective_steps - 1)
            rem = effective_steps - t
            cand_nodes = findall(>(0.0), T_step[curr, :])
            weights = Float64[]
            for j in cand_nodes
                t_ij = T_step[curr, j]
                p_j_end = (rem == 1) ? T_step[j, u_e] : P_mats[rem][j, u_e]
                push!(weights, t_ij * p_j_end)
            end
            w_sum = sum(weights)
            if w_sum <= 1e-15
                sp_curr = a_star(graph, curr, u_e)
                next_node = !isempty(sp_curr) ? dst(first(sp_curr)) : curr
                path[t + 1] = next_node
                curr = next_node
            else
                probs = weights ./ w_sum
                r = rand(rng)
                cum = 0.0
                chosen = cand_nodes[end]
                for (idx, j) in enumerate(cand_nodes)
                    cum += probs[idx]
                    if r <= cum
                        chosen = j
                        break
                    end
                end
                path[t + 1] = chosen
                curr = chosen
            end
        end

        return path
    end

    # Fallback when no graph topology is supplied
    if n_steps <= 1
        return [u_s, u_e]
    end

    local P_mats_dense::Vector{Matrix{Float64}}
    if !isnothing(powers) && length(powers) >= n_steps
        P_mats_dense = [Matrix{Float64}(p) for p in powers[1:n_steps]]
    else
        P_mats_dense = Vector{Matrix{Float64}}(undef, n_steps)
        P_mats_dense[1] = Matrix{Float64}(Gamma)
        for k in 2:n_steps
            P_mats_dense[k] = P_mats_dense[k-1] * P_mats_dense[1]
        end
    end

    path = zeros(Int, n_steps + 1)
    path[1] = u_s
    path[end] = u_e
    curr = u_s

    for t in 1:(n_steps - 1)
        rem_steps = n_steps - t
        weights = zeros(Float64, S)
        for j in 1:S
            g_ij = Float64(Gamma[curr, j])
            p_j_end = (rem_steps == 1) ? Float64(Gamma[j, u_e]) : Float64(P_mats_dense[rem_steps][j, u_e])
            weights[j] = g_ij * p_j_end
        end
        w_sum = sum(weights)
        if w_sum <= 1e-15
            weights = Float64[Float64(Gamma[curr, j]) for j in 1:S]
            w_sum = sum(weights)
        end
        probs = weights ./ (w_sum > 0.0 ? w_sum : 1.0)

        r = rand(rng)
        cum = 0.0
        next_s = S
        for j in 1:S
            cum += probs[j]
            if r <= cum
                next_s = j
                break
            end
        end
        path[t + 1] = next_s
        curr = next_s
    end

    return path
end

"""
    reconstruct_mark_recapture_paths(tagging::DataFrame, result::NamedTuple;
                                     time_interval::Symbol = :monthly,
                                     max_paths::Union{Nothing, Int} = nothing,
                                     smooth_jitter::Bool = true,
                                     rng::Random.AbstractRNG = Random.GLOBAL_RNG)::Vector{NamedTuple}

Reconstructs the latent continuous state-space trajectory for every individual
mark-recapture event in `tagging` using an exact discrete Markov Bridge filter conditioned
on the true observed release and recapture coordinates under the fitted transition matrix
and HSI field. Trajectories strictly follow the marine water graph and avoid land crossing.

# Mathematical Formulation:
For an individual crab ``k`` with release observation at ``(\\text{lon}_0, \\text{lat}_0)``
at time ``t_0`` and recapture at ``(\\text{lon}_T, \\text{lat}_T)`` at time ``t_T``:
1. Coordinates are mapped to nearest discrete marine areal units ``u_0, u_T \\in \\mathcal{U}``.
   When observations lie beyond the domain, the last encountered boundary unit HSI probability is extended.
2. The elapsed time interval ``\\Delta t = \\max(1, \\text{round}(\\text{Int}, |t_T - t_0| / \\Delta t_{\\text{interval}}))``.
3. Conditional latent intermediate units ``(z_0 = u_0, z_1, \\dots, z_T = u_T)`` are sampled
   along connected marine graph edges.
4. Each discrete state ``z_t`` is projected to continuous geographic coordinates ``(\\text{lon}_t, \\text{lat}_t)``
   with boundary pinning at true release and recapture points.

# Inputs:
- `tagging`: Observation DataFrame containing `:tagid`, `:lon`, `:lat`, and `:time` or `:timestamp`.
- `result`: Model bundle containing `au`, `transition_matrix`, and `opts`.
- `time_interval`: Temporal step resolution (`:monthly`, `:weekly`, `:daily`).
- `max_paths`: Optional integer limit on number of individuals to reconstruct (default `nothing` for all).
- `smooth_jitter`: Whether to apply subtle spatial smoothing inside marine polygons.

# Outputs:
- `trajectories::Vector{NamedTuple}`: Reconstructed trajectories for all individuals.
"""
function reconstruct_mark_recapture_paths(
    tagging::DataFrame,
    result::NamedTuple;
    time_interval::Symbol = :monthly,
    max_paths::Union{Nothing, Int} = nothing,
    smooth_jitter::Bool = true,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
)::Vector{NamedTuple}
    au = result.au
    Gamma = Matrix{Float64}(result.transition_matrix)
    S = au.n_units
    g = hasproperty(au, :graph) ? au.graph : nothing
    W = hasproperty(au, :W) ? au.W : nothing
    hsi_vec = hasproperty(result, :hsi) && !isnothing(result.hsi) ? result.hsi : nothing

    # Pre-extract geographic centroids of au in WGS84 degrees
    cents_deg = Tuple{Float64, Float64}[]
    wkt_str = hasproperty(au, :wkt) ? string(au.wkt) : ""
    is_utm = contains(lowercase(wkt_str), "utm") || (!isempty(au.centroids) && au.centroids[1][2] > 1000.0)

    for c in au.centroids
        if is_utm
            push!(cents_deg, utm_to_lonlat(c[1], c[2]; zone=20, is_km=true))
        else
            push!(cents_deg, (Float64(c[1]), Float64(c[2])))
        end
    end

    # Group observations by tag
    time_col = hasproperty(tagging, :time) ? :time : (hasproperty(tagging, :timestamp) ? :timestamp : nothing)
    isnothing(time_col) && error("tagging DataFrame must have :time or :timestamp column.")

    gdf = groupby(tagging, :tagid)
    tag_keys = collect(keys(gdf))
    n_tags_total = length(tag_keys)
    n_to_process = isnothing(max_paths) ? n_tags_total : min(n_tags_total, max_paths)

    # Unit step duration in decimal years
    dt_step = (time_interval == :weekly) ? (1.0 / 52.0) : ((time_interval == :daily) ? (1.0 / 365.0) : (1.0 / 12.0))

    trajectories = NamedTuple[]

    for i in 1:n_to_process
        sub = sort(gdf[tag_keys[i]], time_col)
        nrow(sub) < 2 && continue

        tid = string(first(sub.tagid))
        all_path_units = Int[]
        all_path_coords = Tuple{Float64, Float64}[]

        # Reconstruct piecewise between consecutive observation records
        for seg in 1:(nrow(sub) - 1)
            r1 = sub[seg, :]
            r2 = sub[seg + 1, :]

            lon1, lat1 = Float64(r1.lon), Float64(r1.lat)
            lon2, lat2 = Float64(r2.lon), Float64(r2.lat)
            t1 = Float64(r1[time_col])
            t2 = Float64(r2[time_col])

            # Find nearest spatial units in au
            u_start = _nearest_unit_index(lon1, lat1, cents_deg)
            u_end = _nearest_unit_index(lon2, lat2, cents_deg)

            elapsed_t = max(0.0, t2 - t1)
            n_steps = clamp(round(Int, elapsed_t / dt_step), 1, 60)

            # Sample water-constrained Markov bridge
            bridge = sample_markov_bridge(
                Gamma, u_start, u_end, n_steps;
                graph = g, W = W, hsi = hsi_vec, rng = rng
            )

            # Convert bridge units to continuous geographic coordinates
            for (step_idx, u_id) in enumerate(bridge)
                if seg > 1 && step_idx == 1
                    continue # Avoid duplicate point at segment boundary
                end

                push!(all_path_units, u_id)

                if step_idx == 1
                    push!(all_path_coords, (lon1, lat1))
                elseif step_idx == length(bridge)
                    push!(all_path_coords, (lon2, lat2))
                else
                    c_pt = cents_deg[u_id]
                    pt_lon, pt_lat = c_pt[1], c_pt[2]
                    if smooth_jitter
                        # Subtle perturbation inside marine polygon (approx 0.003 deg ~ 300 m)
                        jit_r = 0.003 * sqrt(rand(rng))
                        jit_theta = 2.0 * pi * rand(rng)
                        pt_lon += jit_r * cos(jit_theta)
                        pt_lat += jit_r * sin(jit_theta)
                    end
                    push!(all_path_coords, (pt_lon, pt_lat))
                end
            end
        end

        length(all_path_coords) < 2 && continue

        # Compute trajectory distance and net displacement (km)
        total_dist_km = 0.0
        for k in 2:length(all_path_coords)
            total_dist_km += haversine_distance(
                all_path_coords[k-1][1], all_path_coords[k-1][2],
                all_path_coords[k][1], all_path_coords[k][2]
            ) / 1000.0
        end

        net_disp_km = haversine_distance(
            all_path_coords[1][1], all_path_coords[1][2],
            all_path_coords[end][1], all_path_coords[end][2]
        ) / 1000.0

        dur_days = Float64((sub[end, time_col] - sub[1, time_col]) * 365.25)

        push!(trajectories, (
            tagid           = tid,
            coords          = all_path_coords,
            units           = all_path_units,
            n_steps         = length(all_path_coords) - 1,
            start_date      = string(sub[1, time_col]),
            end_date        = string(sub[end, time_col]),
            duration_days   = dur_days,
            total_dist_km   = total_dist_km,
            displacement_km = net_disp_km
        ))
    end

    return trajectories
end

function _nearest_unit_index(lon::Float64, lat::Float64, cents_deg::Vector{Tuple{Float64, Float64}})::Int
    best_idx = 1
    best_dist = Inf
    for (i, c) in enumerate(cents_deg)
        d = (c[1] - lon)^2 + (c[2] - lat)^2
        if d < best_dist
            best_dist = d
            best_idx = i
        end
    end
    return best_idx
end



"""
    _build_A_ad(adj_rows, hsi, gamma_g, S)

AD-compatible directed adjacency build. Optimized for ForwardDiff.
"""
function _build_A_ad(adj_rows, hsi, gamma_g, S)
    # 1. Mathematical simplification & Vectorization:
    # exp(γ(HSI_j - HSI_i)) / Σ exp(γ(HSI_k - HSI_i)) mathematically simplifies to
    # exp(γ HSI_j) / Σ exp(γ HSI_k). The HSI_i term cancels out completely!
    # We compute this once for all units, which AD engines handle highly efficiently.
    exp_hsi = exp.(gamma_g .* hsi)
    
    # Extract the promoted AD type (e.g., ForwardDiff.Dual) directly from the math
    T = eltype(exp_hsi)
    A = zeros(T, S, S)
    
    @inbounds for i in 1:S
        nbrs = adj_rows[i]
        isempty(nbrs) && continue
        
        # 2. Allocation-free denominator accumulation
        sw = zero(T)
        for j in nbrs
            sw += exp_hsi[j]
        end
        
        sw <= 0 && continue
        
        # 3. Allocation-free assignment
        for j in nbrs
            A[i, j] = exp_hsi[j] / sw
        end
    end
    
    return A
end


function _row_normalise(M, S)
    T = eltype(M)
    M_rect = max.(zero(T), M)
    s = sum(M_rect, dims=2)
    
    # Non-mutating division-by-zero protection
    s_safe = s .+ (s .== zero(T))
    
    # Broadcast normalization and uniform distribution injection for zero-rows
    return (M_rect ./ s_safe) .+ ((s .== zero(T)) ./ T(S))
end

# ── Directed adjacency and resolvent operator ──────────────────────────────────
 

"""
    compute_directed_adjacency(hsi, W; gamma=1.0) -> Matrix{Float64}

Construct the directed adjacency matrix A from the symmetric adjacency W and
habitat suitability values HSI. Row i has non-zero entries only at W-neighbours
of i, weighted by exp(γ HSI_j) and row-normalised:

    A[i,j] = exp(γ HSI_j) / Σ_k W_ik exp(γ HSI_k)

γ > 0 biases movement towards higher HSI; γ = 0 gives the uniform random walk.

# Arguments
- `hsi::Vector{Float64}`: Habitat suitability per spatial unit (length S).
- `W::SparseMatrixCSC`: Symmetric binary adjacency matrix (S × S).
- `gamma::Real`: Advection sensitivity to HSI gradient (default 1.0).

# Returns
- Dense S × S matrix A with row-stochastic structure over W-neighbours.
"""
function compute_directed_adjacency(
    hsi::AbstractVector{<:Real},
    W::SparseMatrixCSC;
    gamma::Real = 1.0
)::Matrix{Float64}
    
    S = size(W, 1)
    A = zeros(Float64, S, S)
    
    # 1. Mathematically simplify and precompute exponents.
    # exp(γ(HSI_j - HSI_i)) / Σ exp(γ(HSI_k - HSI_i)) simplifies exactly to
    # exp(γ HSI_j) / Σ exp(γ HSI_k). This saves thousands of exp() calls.
    exp_hsi = exp.(gamma .* hsi)
    
    for i in 1:S
        # 2. Because W is symmetric, neighbors of row `i` are neighbors of col `i`.
        # Accessing CSC columns via internal pointers is allocation-free and O(1).
        col_start = W.colptr[i]
        col_end   = W.colptr[i+1] - 1
        
        # Skip if no neighbors
        col_start > col_end && continue 
        
        # 3. Calculate denominator
        sw = 0.0
        @inbounds for ptr in col_start:col_end
            j = W.rowval[ptr]
            sw += exp_hsi[j]
        end
        
        sw <= 0.0 && continue
        
        # 4. Populate dense transition matrix A
        @inbounds for ptr in col_start:col_end
            j = W.rowval[ptr]
            A[i, j] = exp_hsi[j] / sw
        end
    end
    
    return A
end

 
"""
    resolvent_transition(beta, D_diff, A, L, S) -> Matrix{Float64}

Compute the resolvent transition operator:

    Γ̄ = (I − β A − D L)^{-1}

where L is the symmetric graph Laplacian. Rows of Γ̄ are rectified (negative
entries set to zero) and row-normalised to form valid probability distributions.

Note: the inverse exists when ‖β A + D L‖ < 1 in an appropriate operator norm.
The NUTS priors (β < 0.95, D ≥ 0) are chosen to help ensure this, but
near-boundary samples may produce poorly conditioned M; the try/catch below
falls back to `M \\ I` in those cases.

# Arguments
- `beta`: Advective weight (scalar, 0 ≤ β < 1).
- `D_diff`: Diffusion coefficient (scalar, ≥ 0).
- `A::Matrix{Float64}`: Directed adjacency matrix (S × S, row-stochastic over nbrs).
- `L::Matrix{Float64}`: Graph Laplacian (S × S, positive semidefinite).
- `S::Int`: Number of spatial units.

# Returns
- Dense S × S matrix Γ̄ with row-stochastic rows.
"""
function resolvent_transition(
    beta::Real,
    D_diff::Real,
    A::AbstractMatrix{<:Real},
    L::AbstractMatrix{<:Real},
    S::Int
)::Matrix{Float64}

    # Lock types upfront for type-stability
    b = Float64(beta)
    d = Float64(D_diff)

    # 1. Construct M in a single allocation-free, column-major pass
    M = Matrix{Float64}(undef, S, S)
    @inbounds for j in 1:S
        for i in 1:S
            diag = (i == j) ? 1.0 : 0.0
            M[i, j] = diag - b * A[i, j] - d * L[i, j]
        end
    end

    # 2. Invert M
    Gamma = try
        inv(M)
    catch e
        @warn "resolvent_transition: inv failed, falling back to M\\I: $(e)"
        M \ Matrix{Float64}(I, S, S)
    end

    # 3. Rectify and accumulate row sums (Column-Major for peak cache efficiency)
    row_sums = zeros(Float64, S)
    @inbounds for j in 1:S
        for i in 1:S
            v = max(0.0, Gamma[i, j])
            Gamma[i, j] = v
            row_sums[i] += v
        end
    end

    # 4. Row-normalise using the accumulated sums (Column-Major)
    @inbounds for j in 1:S
        for i in 1:S
            if row_sums[i] > 0.0
                Gamma[i, j] /= row_sums[i]
            end
        end
    end

    return Gamma
end



"""
    _powerm(M, k) -> Matrix

Compute M^k using iterative binary (repeated squaring) exponentiation.
Optimized to perform all multiplications in-place, reducing memory allocations 
to \$O(1)\$ regardless of \$k\$.
"""
function _powerm(M::AbstractMatrix{T}, k::Integer) where {T <: Real}
    n = size(M, 1)
    
    # Ensure type stability by matching the precision of M (e.g., Float64)
    OutType = float(T) 
    
    k <= 0 && return Matrix{OutType}(I, n, n)
    k == 1 && return Matrix{OutType}(M)

    # Preallocate active matrices
    R = Matrix{OutType}(I, n, n)
    B = Matrix{OutType}(M)
    
    # Preallocate temporary buffers for in-place multiplication
    R_tmp = similar(R)
    B_tmp = similar(B)

    # Iterative repeated squaring
    while k > 0
        if isodd(k)
            # R = R * B (in-place)
            mul!(R_tmp, R, B)
            R, R_tmp = R_tmp, R  # Swap references (zero cost)
        end
        
        k >>= 1 # Fast bitwise division by 2  ..  isodd(k) and k >>= 1 is faster than k % 2 != 0 and k ÷ 2.
        
        if k > 0
            # B = B * B (in-place)
            mul!(B_tmp, B, B)
            B, B_tmp = B_tmp, B  # Swap references (zero cost)
        end
    end
    
    return R
end

"""
    power_transition(Gamma, k) -> Matrix{Float64}

Compute Γ̄^k and apply a final row rectification and normalisation to correct 
any numerical drift.

# Arguments
- `Gamma::Matrix{Float64}`: Row-stochastic transition matrix.
- `k::Integer`: Number of discrete steps (≥ 1).

# Returns
- S × S matrix Γ̄^k with row-stochastic rows.
"""
function power_transition(Gamma::AbstractMatrix{Float64}, k::Integer)::Matrix{Float64}
    # Uses the optimized O(1) allocation _powerm we built previously
    Gk = _powerm(Gamma, max(1, k))
    S = size(Gk, 1)
    
    # 1. Rectify negative entries and accumulate row sums (Column-Major)
    row_sums = zeros(Float64, S)
    @inbounds for j in 1:S
        for i in 1:S
            v = max(0.0, Gk[i, j])
            Gk[i, j] = v
            row_sums[i] += v
        end
    end
    
    # 2. Row-normalise using the accumulated sums (Column-Major)
    @inbounds for j in 1:S
        for i in 1:S
            if row_sums[i] > 0.0
                Gk[i, j] /= row_sums[i]
            end
        end
    end
    
    return Gk
end




"""
    generate_simulated_data(;
        domain_km  = 200.0,
        n_tags     = 100,
        n_steps    = 5,
        seed       = 42,
        radius_km  = 5.0,
        center_lon = -60.0,
        center_lat = 46.0,
        crs        = nothing,
        datum      = WGS84Latest
    ) -> NamedTuple

Generate synthetic mark-recapture telemetry in a square planar domain of
side `domain_km` km, centred at `(center_lon, center_lat)`.

# Data generation
- `n_tags` individuals released at random hex units.
- Movement simulated with a uniform random walk on the hex adjacency graph
  (all neighbours equally likely).
- `n_steps` monthly recaptures per individual.
- Sex (`"M"` / `"F"`) and maturity (`"mature"` / `"immature"`) assigned
  randomly in equal proportions.
- Timestamps start 2020-01-01 and increment monthly.

# Arguments
- `domain_km`: Side length of the square domain in km (default 200.0).
- `n_tags`: Number of tagged individuals (default 100).
- `n_steps`: Recapture events per individual (default 5).
- `seed`: Random seed (default 42).
- `radius_km`: Hex circumradius in km (default 5.0).
- `center_lon, center_lat`: Geographic centre for projection.
- `crs`: Target Coordinate Reference System (default nothing, local tangent plane).
- `datum`: Source geographic datum (default: WGS84Latest).

# Returns
`NamedTuple`:
- `tagging::DataFrame`: Mark-recapture table.
- `mesh`: Hex mesh from `build_hex_mesh_planar`.
- `true_kernel::Matrix{Float64}`: Uniform random-walk kernel used.
"""
function generate_simulated_data(;
    domain_km  :: Real = 200.0,
    n_tags     :: Int  = 100,
    n_steps    :: Int  = 5,
    seed       :: Int  = 42,
    radius_km  :: Real = 5.0,
    center_lon :: Real = -60.0,
    center_lat :: Real = 46.0,
    crs        = nothing,
    datum      = WGS84Latest
)::NamedTuple

    # Internal categorical draw — avoids importing Distributions in this file
    function _sample_categorical(p::AbstractVector{Float64}, rng::AbstractRNG)::Int
        u    = rand(rng)
        csum = 0.0
        for (i, pi) in enumerate(p)
            csum += pi
            csum >= u && return i
        end
        return length(p)
    end

    rng  = MersenneTwister(seed)
    half = Float64(domain_km) / 2.0

    n_grid = max(10, round(Int, domain_km / radius_km * 2))
    xs_g   = range(-half, half, length=n_grid)
    ys_g   = range(-half, half, length=n_grid)
    
    # Pre-allocate grid coordinate vectors
    n_pts = length(xs_g) * length(ys_g)
    grid_lon = Vector{Float64}(undef, n_pts)
    grid_lat = Vector{Float64}(undef, n_pts)
    
    idx = 1
    for y in ys_g, x in xs_g
        lon, lat = xy_km_to_lonlat(x, y; 
                        center_lon=center_lon, center_lat=center_lat, 
                        crs=crs, datum=datum)
        grid_lon[idx] = lon
        grid_lat[idx] = lat
        idx += 1
    end

    # Pass the CRS formatting down to the mesh generator
    mesh = build_hex_mesh_planar(grid_lon, grid_lat; 
               radius_km=radius_km, crs=crs, datum=datum)
    S = mesh.n_units

    # Construct the true kernel directly using row sums of the sparse matrix
    row_sums = sum(mesh.W, dims=2)
    kernel   = zeros(Float64, S, S)
    for i in 1:S
        rs = row_sums[i]
        if rs > 0
            # Broadcast the sparse row directly into the dense matrix
            @views kernel[i, :] .= mesh.W[i, :] ./ rs
        else
            kernel[i, i] = 1.0
        end
    end

    sexes = [rand(rng, ["M", "F"])            for _ in 1:n_tags]
    mats  = [rand(rng, ["mature", "immature"]) for _ in 1:n_tags]
    t0_dt = Date(2020, 1, 1)
    
    # Use an array of NamedTuples to store rows (drastically faster than DataFrame push!)
    total_records = n_tags * (n_steps + 1)
    records = Vector{NamedTuple{
        (:tagid, :lon, :lat, :tag, :timestamp, :time, :sex, :mat, :is_dead, :s_idx),
        Tuple{String, Float64, Float64, Int, DateTime, Float64, String, String, Bool, Int}
    }}(undef, total_records)
    
    row_idx = 1
    for i in 1:n_tags
        s_cur   = rand(rng, 1:S)
        tid_str = string(i)
        
        (lon_r, lat_r) = mesh.centroids_lonlat[s_cur]
        records[row_idx] = (
            tagid     = tid_str,
            lon       = lon_r,
            lat       = lat_r,
            tag       = 0,
            timestamp = DateTime(t0_dt),
            time      = _to_decimal_year(t0_dt),
            sex       = sexes[i],
            mat       = mats[i],
            is_dead   = false,
            s_idx     = s_cur
        )
        row_idx += 1

        for step in 1:n_steps
            p_row = kernel[s_cur, :]
            s_cur = _sample_categorical(p_row, rng)
            t_dt  = t0_dt + Month(step)
            
            (lon_r, lat_r) = mesh.centroids_lonlat[s_cur]
            records[row_idx] = (
                tagid     = tid_str,
                lon       = lon_r,
                lat       = lat_r,
                tag       = step,
                timestamp = DateTime(t_dt),
                time      = _to_decimal_year(t_dt),
                sex       = sexes[i],
                mat       = mats[i],
                is_dead   = false,
                s_idx     = s_cur
            )
            row_idx += 1
        end
    end

    # Construct the DataFrame once at the very end
    return (
        tagging     = DataFrame(records),
        mesh        = mesh,
        true_kernel = kernel
    )
end



# ── Mark-recapture event extraction ───────────────────────────────────────────
 
"""
    _extract_mark_recapture_events(tagging; time_interval=:monthly) -> DataFrame

Extract consecutive (release, recapture) pairs from a telemetry DataFrame.

For each individual, consecutive observation pairs are converted to events.
The number of discrete time steps `k` is computed per pair from the elapsed
decimal-year difference:

    dt_unit = 1/12 (:monthly), 1/52 (:weekly), 1/26 (:biweekly), 1/365.25 (:daily)
    k       = max(1, round(Int, Δt / dt_unit))

Sex and maturity are taken from the first record of each individual.  Absent
group columns default to `"unknown"`.

# Required columns
`:tagid`, `:s_idx`, `:time`.

# Optional columns
`:sex`, `:mat`.

# Arguments
- `tagging::DataFrame`: Aggregated telemetry with unit assignments.
- `time_interval::Symbol`: Sets `dt_unit` (default `:monthly`).

# Returns
`DataFrame` with columns: `tagid`, `release`, `recapture`, `k`, `sex`, `mat`.
"""
function _extract_mark_recapture_events(
    tagging::DataFrame;
    time_interval::Symbol = :monthly
)::DataFrame

    # NamedTuple avoids the heap allocation of a Dict
    dt_map = (monthly=1.0/12.0, weekly=1.0/52.0, biweekly=1.0/26.0, daily=1.0/365.25, raw=1.0)
    dt = hasproperty(dt_map, time_interval) ? getproperty(dt_map, time_interval) : 1.0 / 12.0

    has_sex = hasproperty(tagging, :sex)
    has_mat = hasproperty(tagging, :mat)

    # 1. Sort globally upfront
    sorted_df = sort(tagging, [:tagid, :time])
    n_rows = nrow(sorted_df)

    # 2. Extract columns to local vectors. 
    # This guarantees type stability and peak access speed inside the loop.
    tagids = sorted_df.tagid
    times  = sorted_df.time
    s_idxs = sorted_df.s_idx
    sexes  = has_sex ? sorted_df.sex : nothing
    mats   = has_mat ? sorted_df.mat : nothing

    RecordType = NamedTuple{
        (:tagid, :release, :recapture, :k, :sex, :mat), 
        Tuple{String, Int, Int, Int, String, String}
    }
    
    records = Vector{RecordType}(undef, 0)
    
    # Return empty DataFrame immediately if not enough rows to form a pair
    if n_rows < 2
        return DataFrame(tagid=String[], release=Int[], recapture=Int[], 
                         k=Int[], sex=String[], mat=String[])
    end
    
    sizehint!(records, n_rows) # Max possible pairs is n_rows - 1

    # 3. Single-pass flat loop
    for i in 2:n_rows
        # If the tag is the same as the previous row, it's a consecutive pair
        if tagids[i] == tagids[i-1]
            Δt = times[i] - times[i-1]
            k  = max(1, round(Int, Δt / dt))
            
            push!(records, (
                tagid     = string(tagids[i-1]),
                release   = s_idxs[i-1],
                recapture = s_idxs[i],
                k         = k,
                sex       = has_sex ? string(sexes[i-1]) : "unknown",
                mat       = has_mat ? string(mats[i-1]) : "unknown"
            ))
        end
    end

    return isempty(records) ? 
           DataFrame(tagid=String[], release=Int[], recapture=Int[], 
                     k=Int[], sex=String[], mat=String[]) : 
           DataFrame(records)
end


# ── Main data preparation entry point ─────────────────────────────────────────
 
"""
    prepare_movement_data(;
        data_source   = :snowcrab,
        data_dir      = joinpath(@__DIR__, "data"),
        tagging_file  = nothing,
        hsi_path      = nothing,
        telemetry_csv = nothing,
        radius_km     = 5.0,
        time_interval = :monthly,
        filter_dead   = true,
        crs           = nothing,
        datum         = WGS84Latest,
        domain_km     = 200.0,
        n_tags        = 100,
        n_steps       = 5,
        seed          = 42,
        center_lon    = -60.0,
        center_lat    = 46.0,
        ref_doy       = 244.0,
        verbose       = false
    ) -> NamedTuple

Unified data preparation pipeline. Loads or generates mark-recapture telemetry,
builds a planar hexagonal mesh, reshards the HSI field, maps observations to
spatial units, and returns a NamedTuple ready for `fit_categorical_movement`.

# Data sources
- `:snowcrab` — snow crab telemetry + HSI from JLD2 files.
- `:simulate` — synthetic square-domain data with forward-simulated tracks.
- `:csv`      — generic telemetry from CSV (requires `telemetry_csv` path).

# Returns
`NamedTuple`:
- `tagging::DataFrame`: Filtered/aggregated telemetry with `:s_idx` and `:hsi`.
- `mesh`: Hex mesh NamedTuple.
- `hsi_vec::Vector{Float64}`: Climatological mean HSI per unit (length S).
- `monthly_hsi::Matrix{Float64}`: (S × 12T), or empty when unavailable.
- `month_lookup::Dict{Tuple{Int,Int}, Int}`: (year, month) → column index.
- `years::Vector{Int}`: Annual year labels.
- `obs::DataFrame`: Mark-recapture pairs (release, recapture, k, sex, mat).
"""
function prepare_movement_data(;
    data_source   :: Symbol  = :snowcrab,
    data_dir      :: AbstractString = joinpath(@__DIR__, "data"),
    tagging_file  :: Union{Nothing, AbstractString} = nothing,
    hsi_path      :: Union{Nothing, AbstractString} = nothing,
    telemetry_csv :: Union{Nothing, AbstractString} = nothing,
    radius_km     :: Real    = 5.0,
    time_interval :: Symbol  = :monthly,
    filter_dead   :: Bool    = true,
    crs                      = nothing,
    datum                    = WGS84Latest,
    domain_km     :: Real    = 200.0,
    n_tags        :: Int     = 100,
    n_steps       :: Int     = 5,
    seed          :: Int     = 42,
    center_lon    :: Real    = -60.0,
    center_lat    :: Real    = 46.0,
    ref_doy       :: Real    = 244.0,
    verbose       :: Bool    = false
)::NamedTuple

    local tagging, mesh, pre_mapped

    # ── Load / generate raw telemetry ─────────────────────────────────────────
 
    if data_source == :user
        if !isfile(tagging_file) 
            error("User specified tagging file not found: $tagging_file")
        else
            verbose && println("  [data] Loading user specified telemetry file …")
            tagging = JLD2.load(tagging_file)
        end

        # 1. Filter dead (using column-based syntax avoids DataFrameRow overhead)
        if filter_dead && hasproperty(tagging, :is_dead)
            filter!(:is_dead => d -> !coalesce(d, false), tagging)
        end

        # 2. Fast vectorized type coercion via broadcasting
        tagging[!, :lon]   = Float64.(tagging.lon)
        tagging[!, :lat]   = Float64.(tagging.lat)
        tagging[!, :tag]   = Int.(tagging.tag)
        tagging[!, :tagid] = string.(tagging.tagid)
        if hasproperty(tagging, :time)
            tagging[!, :time] = Float64.(tagging.time)
        end

        # 3. Zero-allocation mean longitude check
        sum_lon = 0.0
        n_lon   = 0
        for x in tagging.lon
            if isfinite(x)
                sum_lon += x
                n_lon += 1
            end
        end
        
        if n_lon > 0 && (sum_lon / n_lon) > 0
            @warn "Mean longitude is positive ($(round(sum_lon / n_lon, digits=2))). " *
                "Verify sign convention — western Atlantic data should be negative."
        end

        # 4. Require both events using ultra-fast SubDataFrame column views
        if require_both_events
            valid_set = Set{String}()
            
            for sub in groupby(tagging, :tagid)
                tags = sub.tag # direct view into the column
                
                # Use fast short-circuiting checks
                if any(==(0), tags) && any(>(0), tags)
                    push!(valid_set, sub.tagid[1])
                end
            end
            
            # ∈(valid_set) is highly optimized in Julia for Set lookups
            filter!(:tagid => ∈(valid_set), tagging)
        end

        sort!(tagging, [:tagid, :time])

        pre_mapped = nothing
        verbose && println(
            "    $(length(unique(tagging.tagid))) individuals, $(nrow(tagging)) obs.")
    
    elseif data_source == :simulate
        verbose && println("  [data] Generating simulated data …")
        sim        = generate_simulated_data(;
            domain_km=domain_km, n_tags=n_tags, n_steps=n_steps, seed=seed,
            radius_km=radius_km, center_lon=center_lon, center_lat=center_lat,
            crs=crs, datum=datum)
        tagging    = sim.tagging
        pre_mapped = sim.mesh
        verbose && println("    $(n_tags) individuals, $(sim.mesh.n_units) units.")
 
    elseif data_source == :csv
        (!isnothing(telemetry_csv) && isfile(telemetry_csv)) ||
            error("--telemetry-path must be a readable CSV for :csv data source.")
        _HAS_CSV || error("CSV.jl not available; add it to the project.")
        verbose && println("  [data] Loading CSV: $(telemetry_csv) …")
        
        tagging = CSV.read(telemetry_csv, DataFrame)
        if hasproperty(tagging, :timestamp) && eltype(tagging.timestamp) <: AbstractString
            tagging[!, :timestamp] = DateTime.(tagging.timestamp)
        end
        if !hasproperty(tagging, :time)
            tagging[!, :time] = _to_decimal_year.(tagging.timestamp)
        end
        pre_mapped = nothing
        verbose && println("    $(nrow(tagging)) rows loaded.")

    else
        error("Unknown data_source: $(data_source). Use :snowcrab, :simulate, or :csv.")
    end

    # ── Temporal aggregation (skip for simulate — already at target resolution) ─
    if data_source != :simulate
        verbose && println("  [data] Aggregating to :$(time_interval) …")
        tagging = aggregate_telemetry_time(tagging; time_interval=time_interval)
        verbose && println(
            "    $(nrow(tagging)) rows, $(length(unique(tagging.tagid))) individuals.")
    end

    # ── Build planar hex mesh ──────────────────────────────────────────────────
    if isnothing(pre_mapped)
        verbose && println("  [data] Building hex mesh (r=$(radius_km) km) …")
        lon_v = Float64.(tagging.lon)
        lat_v = Float64.(tagging.lat)
        mesh  = build_hex_mesh_planar(lon_v, lat_v;
                    radius_km=radius_km, crs=crs, datum=datum)
    else
        mesh = pre_mapped
    end
    verbose && println("    Mesh: $(mesh.n_units) units.")

    # ── Map telemetry to hex units ─────────────────────────────────────────────
    verbose && println("  [data] Mapping observations to hex units …")
    tagging = map_telemetry_to_units(tagging, mesh.centroids_km,
                  mesh.center_lon, mesh.center_lat; crs=crs, datum=datum)

    # ── Load HSI and reshard to destination mesh ───────────────────────────────
    hsi_vec      = fill(0.5, mesh.n_units)
    monthly_hsi  = Matrix{Float64}(undef, 0, 0)
    month_lookup = Dict{Tuple{Int, Int}, Int}()
    years_vec    = Int[]

    hsi_jld2 = !isnothing(hsi_path) ? hsi_path : joinpath(data_dir, "hsi.jld2")

    if isfile(hsi_jld2)
        verbose && println("  [data] Loading HSI: $(hsi_jld2) …")
        h            = load_hsi_jld2(hsi_jld2; ref_doy=ref_doy)
        years_vec    = h.years
        month_lookup = h.month_lookup
        S_src        = size(h.monthly_hsi, 1)

        if S_src == mesh.n_units
            monthly_hsi = h.monthly_hsi
            hsi_vec     = h.hsi_spatial_mean
        else
            verbose && println("    Resharding HSI $(S_src) → $(mesh.n_units) units …")
            sppoly_jld2 = joinpath(data_dir, "sppoly.jld2")

            if isfile(sppoly_jld2)
                sp     = JLD2.load(sppoly_jld2)
                au_src = sp["au"]
                n_au   = length(au_src.lon)
                
                # Preallocate array for massive speedup over push!()
                src_coords_km = Vector{Tuple{Float64, Float64}}(undef, n_au)
                for i in 1:n_au
                    src_coords_km[i] = lonlat_to_xy_km(
                        au_src.lon[i], au_src.lat[i];
                        center_lon=mesh.center_lon, center_lat=mesh.center_lat,
                        crs=crs, datum=datum)
                end
            else
                src_coords_km = Tuple{Float64, Float64}[]
                @warn "sppoly.jld2 not found; cannot reshard HSI. Using domain mean."
            end

            if !isempty(src_coords_km)
                n_mo        = size(h.monthly_hsi, 2)
                monthly_hsi = zeros(Float64, mesh.n_units, n_mo)
                for m in 1:n_mo
                    monthly_hsi[:, m] = reshard_hsi_field(
                        h.monthly_hsi[:, m], src_coords_km, mesh.centroids_km)
                end
                hsi_vec = vec(mean(monthly_hsi, dims=2))
            else
                hsi_vec = fill(mean(h.hsi_spatial_mean), mesh.n_units)
            end
        end

        if !isempty(month_lookup) && size(monthly_hsi, 1) == mesh.n_units
            tagging[!, :hsi] = match_telemetry_closest_month_hsi(
                tagging, monthly_hsi, month_lookup, years_vec)
        else
            tagging[!, :hsi] = fill(0.5, nrow(tagging))
        end
        verbose && begin
            hsi_vals = filter(isfinite, tagging.hsi)
            isempty(hsi_vals) ||
                println("    HSI matched (mean=$(round(mean(hsi_vals), digits=4))).")
        end
    else
        tagging[!, :hsi] = fill(0.5, nrow(tagging))
        !isnothing(hsi_path) &&
            @warn "HSI file not found: $(hsi_jld2); using uniform HSI = 0.5."
    end

    # ── Extract mark-recapture event pairs ─────────────────────────────────────
    verbose && println("  [data] Extracting mark-recapture event pairs …")
    obs = _extract_mark_recapture_events(tagging; time_interval=time_interval)
    verbose && println("    $(nrow(obs)) event pairs.")

    return (
        tagging      = tagging,
        mesh         = mesh,
        hsi_vec      = hsi_vec,
        monthly_hsi  = monthly_hsi,
        month_lookup = month_lookup,
        years        = years_vec,
        obs          = obs
    )
end

# ── Group stratification ───────────────────────────────────────────────────────


"""
    build_group_indices(obs; sex_col=:sex, mat_col=:mat) 
    -> (DataFrame, Dict{String,Int})

Map each observation to an integer group index based on a 3-tier categorization:
1. `"immature"`: Any individual where maturity is "immature".
2. `"male"`: Any mature male (maturity "mature", sex "M").
3. `"female"`: Any mature female (maturity "mature", sex "F").
(Unmatched combinations default to `"unknown"`).

# Arguments
- `obs::DataFrame`: Mark-recapture events; must contain `sex_col` and `mat_col`.
- `sex_col`: Column name for sex (default `:sex`).
- `mat_col`: Column name for maturity (default `:mat`).

# Returns
- `obs` copy with `:group::Int` column added (1-based group index).
- `group_lookup::Dict{String,Int}`: group label → index.
"""
function build_group_indices(
    obs::DataFrame;
    sex_col::Symbol = :sex,
    mat_col::Symbol = :mat
)::Tuple{DataFrame, Dict{String, Int}}

    n_rows = nrow(obs)
    has_sex = hasproperty(obs, sex_col)
    has_mat = hasproperty(obs, mat_col)

    # 1. Preallocate the labels array
    labels = Vector{String}(undef, n_rows)

    if has_sex && has_mat
        # 2. Extract columns for type-stable, zero-overhead loop access
        sexes = obs[!, sex_col]
        mats  = obs[!, mat_col]

        @inbounds for i in 1:n_rows
            sx = string(sexes[i])
            mt = string(mats[i])

            # Apply the specific 3-tier biological categorization
            if mt == "immature"
                labels[i] = "immature"
            elseif mt == "mature" && sx == "M"
                labels[i] = "male"
            elseif mt == "mature" && sx == "F"
                labels[i] = "female"
            else
                labels[i] = "unknown"
            end
        end
    else
        # Fallback if required columns are missing
        fill!(labels, "unknown")
    end

    # 3. Determine unique groups present in the data
    unique_labels = sort!(unique(labels))
    group_lookup  = Dict{String, Int}(lbl => i for (i, lbl) in enumerate(unique_labels))
    
    # 4. Map labels to integer IDs efficiently
    group_ids = Vector{Int}(undef, n_rows)
    @inbounds for i in 1:n_rows
        group_ids[i] = group_lookup[labels[i]]
    end

    out = copy(obs)
    out[!, :group] = group_ids
    
    return out, group_lookup
end

