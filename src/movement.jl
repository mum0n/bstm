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
    centroids = if hasproperty(au_context, :centroids)
        au_context.centroids
    elseif hasproperty(au_context, :centroids_km)
        au_context.centroids_km
    elseif hasproperty(au_context, :centroids_lonlat)
        au_context.centroids_lonlat
    else
        throw(ArgumentError(
            "au_context must contain :centroids, :centroids_km, or :centroids_lonlat."
        ))
    end
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
                norm_prev = norm(v_prev)
                if norm_prev > 1e-9
                    for j in 1:n_spatial
                        if j != curr_node && p_row[j] > 0.0
                            v_cand = [centroids[j][d] - centroids[curr_node][d] for d in 1:2]
                            norm_cand = norm(v_cand)
                            if norm_cand > 1e-9
                                cos_theta = dot(v_prev, v_cand) / (norm_prev * norm_cand)
                                p_row[j] *= exp(rho_persistence * cos_theta)
                            end
                        end
                    end
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
    centroids = if hasproperty(au_context, :centroids)
        au_context.centroids
    elseif hasproperty(au_context, :centroids_km)
        au_context.centroids_km
    elseif hasproperty(au_context, :centroids_lonlat)
        au_context.centroids_lonlat
    else
        throw(ArgumentError(
            "au_context must contain :centroids, :centroids_km, or :centroids_lonlat."
        ))
    end
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
                norm_prev = norm(v_prev)
                if norm_prev > 1e-9
                    for j in 1:n_spatial
                        if j != curr_node && p_row[j] > 0.0
                            v_cand = [centroids[j][d] - centroids[curr_node][d] for d in 1:2]
                            norm_cand = norm(v_cand)
                            if norm_cand > 1e-9
                                cos_theta = dot(v_prev, v_cand) / (norm_prev * norm_cand)
                                p_row[j] *= exp(rho_persistence * cos_theta)
                            end
                        end
                    end
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
    # exp(γ(HSI_j - HSI_i)) / Σ exp(γ(HSI_k - HSI_i)) simplifies to
    # exp(γ HSI_j) / Σ exp(γ HSI_k). The HSI_i term cancels out 
    # We compute this once for all units 
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

```math
A[i,j] = \\frac{\\exp(\\gamma_i \\cdot \\text{HSI}_j)}{\\sum_{k \\in \\mathcal{N}(i)} \\exp(\\gamma_i \\cdot \\text{HSI}_k)}
```

γ > 0 biases movement towards higher HSI; γ = 0 gives the uniform random walk.

# Arguments
- `hsi::AbstractVector{<:Real}`: Habitat suitability per spatial unit (length S).
- `W::SparseMatrixCSC`: Symmetric binary adjacency matrix (S × S).
- `gamma::Union{Real, AbstractVector{<:Real}}`: Advection sensitivity to HSI gradient
  (default 1.0). May be a scalar or a spatially varying vector of length S.

# Returns
- `Matrix{Float64}`: Dense S × S matrix A with row-stochastic structure over W-neighbours.
"""
function compute_directed_adjacency(
    hsi::AbstractVector{<:Real},
    W::SparseMatrixCSC;
    gamma::Union{Real, AbstractVector{<:Real}} = 1.0
)::Matrix{Float64}
    S = size(W, 1)
    A = zeros(Float64, S, S)
    
    is_spatial = gamma isa AbstractVector
    if is_spatial
        n_g = length(gamma)
        if n_g != S && n_g != 1
            throw(DimensionMismatch(
                "gamma vector has length $n_g, but must be scalar or match spatial units S=$S."
            ))
        end
        if n_g == 1
            gamma_scalar = Float64(gamma[1])
            is_spatial = false
        end
    else
        gamma_scalar = Float64(gamma)
    end

    if !is_spatial
        # Fast path: precompute exponents once for uniform scalar gamma
        exp_hsi = exp.(gamma_scalar .* hsi)
        for i in 1:S
            col_start = W.colptr[i]
            col_end   = W.colptr[i+1] - 1
            col_start > col_end && continue 
            
            sw = 0.0
            @inbounds for ptr in col_start:col_end
                j = W.rowval[ptr]
                sw += exp_hsi[j]
            end
            sw <= 0.0 && continue
            
            @inbounds for ptr in col_start:col_end
                j = W.rowval[ptr]
                A[i, j] = exp_hsi[j] / sw
            end
        end
    else
        # Spatially-varying gamma[i] per unit i
        for i in 1:S
            col_start = W.colptr[i]
            col_end   = W.colptr[i+1] - 1
            col_start > col_end && continue 
            
            g_i = Float64(gamma[i])
            sw = 0.0
            @inbounds for ptr in col_start:col_end
                j = W.rowval[ptr]
                sw += exp(g_i * Float64(hsi[j]))
            end
            sw <= 0.0 && continue
            
            @inbounds for ptr in col_start:col_end
                j = W.rowval[ptr]
                A[i, j] = exp(g_i * Float64(hsi[j])) / sw
            end
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
    construct_stochastic_transition_kernel(W, hsi; gamma=1.0, residence=0.2, advection=0.5, spatial=false)

Constructs strictly row-stochastic, unconditionally non-negative discrete movement
transition matrices ``P`` over spatial graph ``W``:

```math
P = (1 - \\rho) \\left[ (1 - \\alpha) T_{\\text{diff}} + \\alpha A \\right] + \\rho I
```

# Arguments
- `W::SparseMatrixCSC`: Spatial adjacency matrix (size ``S \\times S``).
- `hsi::AbstractVector{<:Real}`: Habitat suitability values per unit (length ``S``).
- `gamma::Union{Real, AbstractVector{<:Real}}`: Sensitivity of directional advection to
  the habitat gradient (default 1.0). May be scalar or vector across groups or units.
- `residence::Union{Real, AbstractVector{<:Real}}`: Probability ``\\rho \\in [0, 1)`` of
  remaining in current unit (default 0.2). May be scalar or vector across groups or units.
- `advection::Union{Real, AbstractVector{<:Real}}`: Fraction ``\\alpha \\in [0, 1]`` of
  directed movement vs random diffusion (default 0.5). May be scalar or vector.
- `spatial::Bool`: When `true`, vector parameters of length ``S`` are interpreted as
  spatially varying per unit ``s \\in 1:S``, returning a single ``S \\times S`` matrix.
  When `false` (default), vector parameters of length ``G`` represent group-level
  parameters across ``G`` biological groups, returning a `Vector{Matrix{Float64}}`.

# Returns
- `Matrix{Float64}`: If all parameters are scalars (or `spatial=true`), dense ``S \\times S``
  row-stochastic transition probability matrix.
- `Vector{Matrix{Float64}}`: If any parameter is an `AbstractVector` (and `spatial=false`),
  vector of ``G`` dense ``S \\times S`` row-stochastic transition matrices.
"""
function construct_stochastic_transition_kernel(
    W::SparseMatrixCSC,
    hsi::AbstractVector{<:Real};
    gamma::Union{Real, AbstractVector{<:Real}} = 1.0,
    residence::Union{Real, AbstractVector{<:Real}} = 0.2,
    advection::Union{Real, AbstractVector{<:Real}} = 0.5,
    spatial::Bool = false,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Union{Matrix{Float64}, Vector{Matrix{Float64}}}
    S = size(W, 1)

    # 1. Unbiased topological random walk matrix T_diff (row-stochastic, independent of params)
    T_diff = zeros(Float64, S, S)
    for i in 1:S
        col_start = W.colptr[i]
        col_end   = W.colptr[i+1] - 1
        deg_i = col_end - col_start + 1
        if deg_i > 0 && col_start <= col_end
            inv_deg = 1.0 / deg_i
            @inbounds for ptr in col_start:col_end
                j = W.rowval[ptr]
                T_diff[i, j] = inv_deg
            end
        else
            T_diff[i, i] = 1.0
        end
    end

    # Helper to enforce land zero-transition barrier and row stochasticity
    function _postprocess_kernel!(P_mat::Matrix{Float64})
        if land_mask !== nothing
            for i in 1:S
                if land_mask[i]
                    P_mat[i, :] .= 0.0
                    P_mat[i, i] = 1.0
                else
                    for j in 1:S
                        if land_mask[j]
                            P_mat[i, j] = 0.0
                        end
                    end
                    rs = sum(view(P_mat, i, :))
                    if rs > 0.0
                        P_mat[i, :] ./= rs
                    else
                        P_mat[i, i] = 1.0
                    end
                end
            end
        else
            for i in 1:S
                rs = sum(view(P_mat, i, :))
                if rs > 0.0
                    P_mat[i, :] ./= rs
                else
                    P_mat[i, i] = 1.0
                end
            end
        end
        return P_mat
    end

    # 2. Check for vector parameters
    any_vector = (gamma isa AbstractVector) ||
                 (residence isa AbstractVector) ||
                 (advection isa AbstractVector)

    if !any_vector
        # Standard scalar parameter execution (backward-compatible)
        rho = clamp(Float64(residence), 0.0, 0.9999)
        alpha = clamp(Float64(advection), 0.0, 1.0)
        A = compute_directed_adjacency(hsi, W; gamma=Float64(gamma))

        P = Matrix{Float64}(undef, S, S)
        w_move = 1.0 - rho
        w_adv  = w_move * alpha
        w_diff = w_move * (1.0 - alpha)

        @inbounds for j in 1:S
            for i in 1:S
                val = w_adv * A[i, j] + w_diff * T_diff[i, j]
                if i == j
                    val += rho
                end
                P[i, j] = val
            end
        end

        return _postprocess_kernel!(P)

    elseif spatial
        # Spatially-varying parameters across S spatial units
        for (name, p) in (("gamma", gamma), ("residence", residence), ("advection", advection))
            if p isa AbstractVector && length(p) != S && length(p) != 1
                throw(DimensionMismatch(
                    "In spatial mode, parameter `$name` has length $(length(p)), " *
                    "but must match spatial units S=$S or be a scalar."
                ))
            end
        end

        rho_vec = residence isa AbstractVector ?
            (length(residence) == 1 ? fill(Float64(residence[1]), S) : Float64.(residence)) :
            fill(Float64(residence), S)
        adv_vec = advection isa AbstractVector ?
            (length(advection) == 1 ? fill(Float64(advection[1]), S) : Float64.(advection)) :
            fill(Float64(advection), S)
        g_vec = gamma isa AbstractVector ?
            (length(gamma) == 1 ? fill(Float64(gamma[1]), S) : Float64.(gamma)) :
            fill(Float64(gamma), S)

        clamp!(rho_vec, 0.0, 0.9999)
        clamp!(adv_vec, 0.0, 1.0)

        A = compute_directed_adjacency(hsi, W; gamma=g_vec)

        P = Matrix{Float64}(undef, S, S)
        @inbounds for i in 1:S
            rho_i    = rho_vec[i]
            alpha_i  = adv_vec[i]
            w_move_i = 1.0 - rho_i
            w_adv_i  = w_move_i * alpha_i
            w_diff_i = w_move_i * (1.0 - alpha_i)

            for j in 1:S
                val = w_adv_i * A[i, j] + w_diff_i * T_diff[i, j]
                if i == j
                    val += rho_i
                end
                P[i, j] = val
            end
        end

        return _postprocess_kernel!(P)

    else
        # Group vector mode: construct distinct transition matrix per group g in 1:G
        v_lengths = [length(p) for p in (gamma, residence, advection) if p isa AbstractVector]
        G = maximum(v_lengths)

        for (name, p) in (("gamma", gamma), ("residence", residence), ("advection", advection))
            if p isa AbstractVector && length(p) != G && length(p) != 1
                throw(DimensionMismatch(
                    "Parameter `$name` has length $(length(p)), but expected length G=$G or 1."
                ))
            end
        end

        g_vec = gamma isa AbstractVector ?
            (length(gamma) == 1 ? fill(Float64(gamma[1]), G) : Float64.(gamma)) :
            fill(Float64(gamma), G)
        rho_vec = residence isa AbstractVector ?
            (length(residence) == 1 ? fill(Float64(residence[1]), G) : Float64.(residence)) :
            fill(Float64(residence), G)
        adv_vec = advection isa AbstractVector ?
            (length(advection) == 1 ? fill(Float64(advection[1]), G) : Float64.(advection)) :
            fill(Float64(advection), G)

        kernels = Vector{Matrix{Float64}}(undef, G)
        for g in 1:G
            rho_g   = clamp(rho_vec[g], 0.0, 0.9999)
            alpha_g = clamp(adv_vec[g], 0.0, 1.0)
            A_g     = compute_directed_adjacency(hsi, W; gamma=g_vec[g])

            P_g = Matrix{Float64}(undef, S, S)
            w_move = 1.0 - rho_g
            w_adv  = w_move * alpha_g
            w_diff = w_move * (1.0 - alpha_g)

            @inbounds for j in 1:S
                for i in 1:S
                    val = w_adv * A_g[i, j] + w_diff * T_diff[i, j]
                    if i == j
                        val += rho_g
                    end
                    P_g[i, j] = val
                end
            end

            kernels[g] = _postprocess_kernel!(P_g)
        end
        return kernels
    end
end


function _spatial_node_distance(c1, c2)::Float64
    x1, y1 = Float64(c1[1]), Float64(c1[2])
    x2, y2 = Float64(c2[1]), Float64(c2[2])
    if abs(x1) <= 180.0 && abs(x2) <= 180.0 && abs(y1) <= 90.0 && abs(y2) <= 90.0
        return haversine_distance(x1, y1, x2, y2)
    else
        return sqrt((x1 - x2)^2 + (y1 - y2)^2)
    end
end

"""
    astar_predict_path(
        P::AbstractMatrix{<:Real},
        release::Int,
        recapture::Int;
        centroids = nothing,
        k::Union{Nothing, Int} = nothing,
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
        p_min::Real = 1e-12
    ) -> Vector{Int}

Reconstructs the most likely individual movement trajectory between `release` and
`recapture` locations using the goal-directed \$A^*\$ heuristic search algorithm on the
negative log-likelihood transition graph.

# Mathematical Formulation
Given row-stochastic transition matrix \$\\mathbf{P} \\in [0, 1]^{S \\times S}\$, the
probability of a discrete trajectory \$\\pi = (u_0=u_{\\text{rel}}, \\dots, u_m=u_{\\text{rec}})\$ is:
```math
\\mathbb{P}(\\pi \\mid u_{\\text{rel}}, u_{\\text{rec}}) = \\prod_{\\tau=0}^{m-1} P_{u_\\tau, u_{\\tau+1}}
```
Maximizing path probability is equivalent to finding the shortest path with additive non-negative costs:
```math
c(u, v) = -\\ln P_{u, v} \\ge 0
```
When node spatial centroids \$\\mathbf{c}_u\$ are provided, the distance \$D(u, u_{\\text{rec}})\$
gives an admissible and consistent heuristic:
```math
h(u) = \\left\\lceil \\frac{D(u, u_{\\text{rec}})}{\\Delta x_{\\max}} \\right\\rceil \\cdot \\min_{i \\neq j} (-\\ln P_{i, j})
```
guaranteeing that \$A^*\$ identifies the exact global maximum-likelihood trajectory while
expanding an order of magnitude fewer nodes than a full trellis search.

# Arguments
- `P`: Row-stochastic transition probability matrix (dense or sparse \$S \\times S\$).
- `release`: Source spatial unit index (1-indexed).
- `recapture`: Destination spatial unit index (1-indexed).
- `centroids`: Optional collection of centroid coordinate tuples `(lon, lat)` or `(x, y)`.
- `k`: Optional target step duration (integer).
- `land_mask`: Optional boolean vector of length \$S\$ (`true` for impermeable land).
- `p_min`: Numerical cutoff below which transitions are treated as zero probability (default `1e-12`).

# Returns
- `Vector{Int}`: Ordered sequence of spatial unit indices connecting `release` to `recapture`.
"""
function astar_predict_path(
    P::AbstractMatrix{<:Real},
    release::Int,
    recapture::Int;
    centroids = nothing,
    k::Union{Nothing, Int} = nothing,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    p_min::Real = 1e-12
)::Vector{Int}
    S = size(P, 1)
    if !(1 <= release <= S) || !(1 <= recapture <= S)
        throw(ArgumentError("Release unit ($release) and recapture unit ($recapture) must be within 1:$S."))
    end
    if release == recapture
        return k !== nothing ? fill(release, max(1, k + 1)) : [release]
    end

    # Build directed graph and sparse cost matrix
    g = SimpleDiGraph(S)
    rows_c = Int[]
    cols_c = Int[]
    vals_c = Float64[]

    max_p_trans = 0.0

    if P isa SparseMatrixCSC
        rows = rowvals(P)
        vals = nonzeros(P)
        for j in 1:S
            if land_mask !== nothing && land_mask[j]
                continue
            end
            for ptr in nzrange(P, j)
                i = rows[ptr]
                if i == j || (land_mask !== nothing && land_mask[i])
                    continue
                end
                p_val = Float64(vals[ptr])
                if p_val > p_min
                    add_edge!(g, i, j)
                    push!(rows_c, i)
                    push!(cols_c, j)
                    cost = -log(p_val)
                    push!(vals_c, cost)
                    if p_val > max_p_trans
                        max_p_trans = p_val
                    end
                end
            end
        end
    else
        for i in 1:S
            if land_mask !== nothing && land_mask[i]
                continue
            end
            for j in 1:S
                if i == j || (land_mask !== nothing && land_mask[j])
                    continue
                end
                p_val = Float64(P[i, j])
                if p_val > p_min
                    add_edge!(g, i, j)
                    push!(rows_c, i)
                    push!(cols_c, j)
                    cost = -log(p_val)
                    push!(vals_c, cost)
                    if p_val > max_p_trans
                        max_p_trans = p_val
                    end
                end
            end
        end
    end

    cents_vec = if centroids !== nothing
        hasproperty(centroids, :centroids_lonlat) ? centroids.centroids_lonlat :
        (hasproperty(centroids, :centroids) ? centroids.centroids : centroids)
    else
        nothing
    end

    heuristic = if cents_vec !== nothing && length(cents_vec) == S && !isempty(rows_c)
        max_d = 1e-6
        for k_idx in 1:length(rows_c)
            u = rows_c[k_idx]
            v = cols_c[k_idx]
            d = _spatial_node_distance(cents_vec[u], cents_vec[v])
            if d > max_d
                max_d = d
            end
        end
        min_cost_hop = max_p_trans > 0.0 ? -log(max_p_trans) : 0.01

        v -> begin
            if v == recapture
                return 0.0
            end
            d_v = _spatial_node_distance(cents_vec[v], cents_vec[recapture])
            hops = ceil(d_v / max_d)
            return hops * min_cost_hop
        end
    else
        v -> 0.0
    end

    distmx = sparse(rows_c, cols_c, vals_c, S, S)
    sp = a_star(g, release, recapture, distmx, heuristic)

    if isempty(sp)
        return [release, recapture]
    end

    raw_path = vcat([src(e) for e in sp], [dst(last(sp))])

    if k !== nothing && k >= 1
        m = length(raw_path) - 1
        if m < k
            p_self = [Float64(P[u, u]) for u in raw_path]
            expanded_path = copy(raw_path)
            while length(expanded_path) < k + 1
                best_idx = argmax(p_self)
                insert!(expanded_path, best_idx, expanded_path[best_idx])
                p_self[best_idx] *= 0.90
            end
            return expanded_path
        end
    end

    return raw_path
end

"""
    astar_least_cost_path(
        centroids::AbstractVector,
        W::AbstractMatrix{<:Real},
        release::Int,
        recapture::Int;
        resistance::Union{Nothing, AbstractVector{<:Real}} = nothing,
        hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
        land_polygons = nothing
    ) -> Vector{Int}

Computes the ecological least-cost migration corridor between `release` and `recapture`
spatial units across an environmental resistance/friction surface using \$A^*\$ graph search.

# Mathematical Formulation
Given node centroids \$\\mathbf{c}_u\$ and network adjacency \$W\$, the physical distance
between adjacent nodes is \$d(u, v) = \\text{dist}(\\mathbf{c}_u, \\mathbf{c}_v)\$.
Traversing node \$v\$ incurs an environmental friction/resistance \$\\Phi(v) \\ge 1.0\$.
The directed edge traversal cost is:
```math
c(u, v) = d(u, v) \\times \\frac{\\Phi(u) + \\Phi(v)}{2}
```
where \$\\Phi(v)\$ can be parameterized from habitat suitability:
```math
\\Phi(v) = 1.0 + 3.0 \\times (1.0 - \\text{HSI}_v)^2
```
With admissible Euclidean / Haversine heuristic \$h(u) = d(u, u_{\\text{rec}}) \\times \\min_w \\Phi(w)\$,
\$A^*\$ identifies the optimal least-resistance corridor avoiding environmental barriers.

# Arguments
- `centroids`: Spatial centroids coordinate vector (length \$S\$).
- `W`: Spatial adjacency matrix (\$S \\times S\$).
- `release`: Starting spatial unit index (1-indexed).
- `recapture`: Destination spatial unit index (1-indexed).
- `resistance`: Optional explicit resistance vector of length \$S\$ (\$\\Phi \\ge 1.0\$).
- `hsi`: Optional habitat suitability index vector (used if `resistance` is not provided).
- `land_mask`: Optional boolean vector denoting impermeable land units.
- `land_polygons`: Optional land barrier polygons for topological edge severing.

# Returns
- `Vector{Int}`: Sequence of spatial unit indices tracing the least-cost path.
"""
function astar_least_cost_path(
    centroids::AbstractVector,
    W::AbstractMatrix{<:Real},
    release::Int,
    recapture::Int;
    resistance::Union{Nothing, AbstractVector{<:Real}} = nothing,
    hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    land_polygons = nothing
)::Vector{Int}
    S = size(W, 1)
    if !(1 <= release <= S) || !(1 <= recapture <= S)
        throw(ArgumentError("Release unit ($release) and recapture unit ($recapture) must be within 1:$S."))
    end
    if release == recapture
        return [release]
    end

    phi = if resistance !== nothing
        Float64.(resistance)
    elseif hsi !== nothing
        [1.0 + 3.0 * (1.0 - clamp(Float64(h), 0.0, 1.0))^2 for h in hsi]
    else
        ones(Float64, S)
    end
    min_phi = minimum(phi)

    W_active = copy(sparse(Float64.(W)))
    if land_mask !== nothing
        for l in findall(land_mask)
            W_active[l, :] .= 0.0
            W_active[:, l] .= 0.0
        end
        dropzeros!(W_active)
    end
    if land_polygons !== nothing
        sever_land_crossing_edges!(W_active, centroids; land_polygons=land_polygons)
    end

    g = SimpleGraph(S)
    rows_c = Int[]
    cols_c = Int[]
    vals_c = Float64[]

    rows = rowvals(W_active)
    for col in 1:S
        for ptr in nzrange(W_active, col)
            row = rows[ptr]
            if row > col
                add_edge!(g, col, row)
                d_ij = _spatial_node_distance(centroids[col], centroids[row])
                cost = d_ij * (phi[col] + phi[row]) / 2.0
                push!(rows_c, col)
                push!(cols_c, row)
                push!(vals_c, cost)
                push!(rows_c, row)
                push!(cols_c, col)
                push!(vals_c, cost)
            end
        end
    end

    distmx = sparse(rows_c, cols_c, vals_c, S, S)
    heuristic = v -> begin
        if v == recapture
            return 0.0
        end
        return _spatial_node_distance(centroids[v], centroids[recapture]) * min_phi
    end

    sp = a_star(g, release, recapture, distmx, heuristic)
    if isempty(sp)
        return [release, recapture]
    end
    return vcat([src(e) for e in sp], [dst(last(sp))])
end

"""
    smooth_marine_path(
        path::Vector{Int},
        centroids::AbstractVector;
        land_polygons = nothing
    ) -> Vector{Int}

Applies line-of-sight shortcutting ("string-pulling") to an animal trajectory on a discrete
mesh, removing artificial cell-to-cell hexagonal zig-zagging while strictly preserving
clearance around terrestrial land barriers (islands, peninsulas, headlands).

# Mathematical Formulation
For path waypoints \$\\mathbf{w} = [u_1, u_2, \\dots, u_m]\$, the algorithm casts a line
of sight between non-adjacent waypoints \$u_i\$ and \$u_j\$ (\$j > i + 1\$).
If the line segment:
```math
L(u_i, u_j) = \\{ (1 - t) \\mathbf{c}_{u_i} + t \\mathbf{c}_{u_j} \\mid t \\in [0, 1] \\}
```
does not intersect any terrestrial barrier polygon (evaluated via `_line_crosses_polygon_or_in`),
all intermediate waypoints \$u_{i+1}, \\dots, u_{j-1}\$ are removed. If an intersection
occurs, waypoints around the headland are retained.

# Arguments
- `path`: Ordered sequence of spatial unit indices.
- `centroids`: Spatial centroids coordinate vector matching unit indices.
- `land_polygons`: Terrestrial boundary polygons (defaults to `:maritimes`).

# Returns
- `Vector{Int}`: Smoothed subset of waypoints with direct lines of sight across open water.
"""
function smooth_marine_path(
    path::Vector{Int},
    centroids::AbstractVector;
    land_polygons = nothing
)::Vector{Int}
    if length(path) <= 2
        return path
    end

    polys = if land_polygons === nothing
        _DEFAULT_MARITIMES_LAND_POLYGONS
    elseif land_polygons in (:none, :false, false)
        nothing
    elseif land_polygons in (:maritimes, :default)
        _DEFAULT_MARITIMES_LAND_POLYGONS
    else
        land_polygons
    end

    smoothed = Int[path[1]]
    curr_idx = 1
    n_pts = length(path)

    while curr_idx < n_pts
        furthest_idx = curr_idx + 1
        for look_idx in n_pts:-1:(curr_idx + 2)
            p_curr = (Float64(centroids[path[curr_idx]][1]), Float64(centroids[path[curr_idx]][2]))
            p_look = (Float64(centroids[path[look_idx]][1]), Float64(centroids[path[look_idx]][2]))
            crosses = polys !== nothing ? _line_crosses_polygon_or_in(p_curr, p_look, polys) : false
            if !crosses
                furthest_idx = look_idx
                break
            end
        end
        push!(smoothed, path[furthest_idx])
        curr_idx = furthest_idx
    end

    return smoothed
end

"""
    StochasticAStarResult

Container holding posterior inference results for stochastic A* pathfinding
propagating uncertainty in habitat suitability, friction parameters, or observation
error terms.

# Fields
- `corridor_prob::Vector{Float64}`: Posterior probability of node inclusion in corridor.
- `edge_prob::SparseMatrixCSC{Float64, Int}`: Posterior traversal probability of edge.
- `medoid_path::Vector{Int}`: Most representative trajectory across posterior draws.
- `all_paths::Vector{Vector{Int}}`: Vector of individual trajectory realizations.
- `path_costs::Vector{Float64}`: Realized path costs across all draws.
- `path_distances::Vector{Float64}`: Realized physical path distances (km).
- `mean_distance::Float64`: Posterior expected physical distance.
- `ci_distance::Tuple{Float64, Float64}`: 95% credible interval for travel distance.
- `release::Int`: Origin node.
- `recapture::Int`: Destination node.
- `n_draws::Int`: Number of stochastic draws evaluated.
"""
struct StochasticAStarResult
    corridor_prob::Vector{Float64}
    edge_prob::SparseMatrixCSC{Float64, Int}
    medoid_path::Vector{Int}
    all_paths::Vector{Vector{Int}}
    path_costs::Vector{Float64}
    path_distances::Vector{Float64}
    mean_distance::Float64
    ci_distance::Tuple{Float64, Float64}
    release::Int
    recapture::Int
    n_draws::Int
end

function Base.show(io::IO, res::StochasticAStarResult)
    S = length(res.corridor_prob)
    n_corridor = count(p -> p > 0.0, res.corridor_prob)
    n_bottleneck = count(p -> p >= 0.80, res.corridor_prob)
    println(io, "StochasticAStarResult:")
    println(io, "  Release -> Recapture:        $(res.release) -> $(res.recapture)")
    println(io, "  Stochastic Draws (M):        $(res.n_draws)")
    println(io, "  Corridor Envelope Units:     $n_corridor / $S units")
    println(io, "  Consensus Bottlenecks (P>=0.8): $n_bottleneck units")
    println(io, "  Medoid Path Waypoints:       $(length(res.medoid_path))")
    print(io,   "  Expected Path Distance:      $(round(res.mean_distance, digits=2)) km " *
                "(95% CI: $(round(res.ci_distance[1], digits=2)) - " *
                "$(round(res.ci_distance[2], digits=2)) km)")
end

"""
    astar_stochastic_least_cost_path(
        centroids::AbstractVector,
        W::AbstractMatrix{<:Real},
        release::Int,
        recapture::Int;
        hsi_samples::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        hsi_mean::Union{Nothing, AbstractVector{<:Real}} = nothing,
        hsi_se::Union{Nothing, AbstractVector{<:Real}} = nothing,
        resistance_samples::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        resistance_mean::Union{Nothing, AbstractVector{<:Real}} = nothing,
        resistance_se::Union{Nothing, AbstractVector{<:Real}} = nothing,
        n_draws::Int = 50,
        friction_power::Real = 2.0,
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
        land_polygons = nothing,
        smooth::Bool = false,
        ci_alpha::Real = 0.05,
        seed::Union{Nothing, Int} = nothing
    ) -> StochasticAStarResult

Performs Bayesian stochastic A* least-cost pathfinding by propagating posterior
uncertainty in habitat suitability (HSI), environmental friction, or observation error
terms across the marine graph.

# Mathematical Formulation
Deterministic A* yields a single trajectory ``\\pi^* = \\arg\\min_{\\pi} \\sum c(u, v)``
conditioned on a fixed point-estimate resistance surface ``\\hat{\\boldsymbol{\\Phi}}``.
In stochastic A*, environmental friction varies across posterior draws:
```math
H_i^{(m)} = \\text{clamp}(H_i^{\\text{obs}} + \\sigma_{H, i} Z_i^{(m)}, 0, 1)
```
or ``\\mathbf{H}^{(m)} \\sim \\pi(\\mathbf{H} \\mid \\mathbf{y})``.
Nodal friction is parameterized as:
```math
\\Phi_i^{(m)} = 1.0 + 3.0 \\left(1.0 - H_i^{(m)}\\right)^\\gamma
```
For each draw ``m = 1, \\dots, M``, A* identifies the optimal trajectory ``\\pi^{(m)}``.
The posterior corridor utilization probability across the graph is computed:
```math
P(i \\in \\text{Corridor} \\mid \\text{data}) =
  \\frac{1}{M} \\sum_{m=1}^M \\mathbb{I}(i \\in \\pi^{(m)})
```
Nodes with ``P(i) \\to 1.0`` indicate mandatory migratory bottleneck pinch-points that
must be traversed regardless of habitat uncertainty, while nodes with intermediate
``0 < P(i) < 1.0`` reveal viable alternative corridors.

# Arguments
- `centroids`: Vector of spatial unit coordinates (lon/lat tuples or planar points).
- `W`: Adjacency matrix of the spatial graph (size ``S \\times S``).
- `release`: Starting spatial unit index.
- `recapture`: Destination spatial unit index.
- `hsi_samples`: Optional ``S \\times M`` matrix of posterior HSI MCMC draws.
- `hsi_mean`: Optional posterior mean / estimated HSI vector (length ``S``).
- `hsi_se`: Optional HSI standard error / observation error vector (length ``S``).
- `resistance_samples`: Optional ``S \\times M`` matrix of friction/resistance draws.
- `resistance_mean`: Optional mean resistance vector.
- `resistance_se`: Optional resistance standard error vector.
- `n_draws`: Number of stochastic realizations (default: 50).
- `friction_power`: Exponent ``\\gamma`` for converting HSI into friction (default: 2.0).
- `land_mask`: Optional boolean vector (`true` for land units).
- `land_polygons`: Optional land boundary geometries for raycasting line-of-sight checks.
- `smooth`: If `true`, applies line-of-sight raycasting (`smooth_marine_path`) to each draw.
- `ci_alpha`: Credible interval significance level (default: 0.05 for 95% CI).
- `seed`: Optional random seed for reproducible sampling.

# Returns
- `StochasticAStarResult`: Posterior corridor probabilities, edge traversal frequencies,
  medoid path, path distance and cost credible intervals.

# References
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic
  determination of minimum cost paths. IEEE Transactions on Systems Science and Cybernetics.
- Adriaensen, F., et al. (2003). The application of least-cost modelling as a functional
  landscape model. Landscape and Urban Planning, 64(4), 233-247.
"""
function astar_stochastic_least_cost_path(
    centroids::AbstractVector,
    W::AbstractMatrix{<:Real},
    release::Int,
    recapture::Int;
    hsi_samples::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    hsi_mean::Union{Nothing, AbstractVector{<:Real}} = nothing,
    hsi_se::Union{Nothing, AbstractVector{<:Real}} = nothing,
    resistance_samples::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    resistance_mean::Union{Nothing, AbstractVector{<:Real}} = nothing,
    resistance_se::Union{Nothing, AbstractVector{<:Real}} = nothing,
    n_draws::Int = 50,
    friction_power = 2.0,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    land_polygons = nothing,
    smooth::Bool = false,
    ci_alpha::Real = 0.05,
    seed::Union{Nothing, Int} = nothing
)::StochasticAStarResult
    S = size(W, 1)
    if !(1 <= release <= S) || !(1 <= recapture <= S)
        throw(ArgumentError(
            "Release ($release) and recapture ($recapture) must be within 1:$S."
        ))
    end

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    _get_power_vec(f_arg, n_sim) = if f_arg isa AbstractVector{<:Real}
        [max(0.05, Float64(f_arg[min(i, length(f_arg))])) for i in 1:n_sim]
    elseif f_arg isa Distribution
        [max(0.05, Float64(rand(rng, f_arg))) for _ in 1:n_sim]
    elseif f_arg isa Real
        fill(max(0.05, Float64(f_arg)), n_sim)
    else
        fill(2.0, n_sim)
    end

    # 1. Assemble resistance realizations
    r_draws::Matrix{Float64} = if !isnothing(resistance_samples)
        if size(resistance_samples, 1) != S
            throw(DimensionMismatch(
                "resistance_samples rows ($(size(resistance_samples, 1))) != units S ($S)"
            ))
        end
        Float64.(resistance_samples)
    elseif !isnothing(resistance_mean)
        if length(resistance_mean) != S
            throw(DimensionMismatch(
                "resistance_mean length ($(length(resistance_mean))) != units S ($S)"
            ))
        end
        M_sim = max(1, n_draws)
        draws = zeros(Float64, S, M_sim)
        mean_vec = Float64.(resistance_mean)
        if !isnothing(resistance_se)
            se_vec = Float64.(resistance_se)
            for m in 1:M_sim
                z = randn(rng, S)
                draws[:, m] = max.(1.0, mean_vec .+ se_vec .* z)
            end
        else
            for m in 1:M_sim
                draws[:, m] = max.(1.0, mean_vec)
            end
        end
        draws
    elseif !isnothing(hsi_samples)
        if size(hsi_samples, 1) != S
            throw(DimensionMismatch(
                "hsi_samples rows ($(size(hsi_samples, 1))) != units S ($S)"
            ))
        end
        M_sim = size(hsi_samples, 2)
        draws = zeros(Float64, S, M_sim)
        p_vec = _get_power_vec(friction_power, M_sim)
        for m in 1:M_sim
            p_m = p_vec[m]
            h_col = clamp.(Float64.(hsi_samples[:, m]), 0.0, 1.0)
            draws[:, m] = [1.0 + 3.0 * (1.0 - h)^p_m for h in h_col]
        end
        draws
    elseif !isnothing(hsi_mean)
        if length(hsi_mean) != S
            throw(DimensionMismatch(
                "hsi_mean length ($(length(hsi_mean))) != units S ($S)"
            ))
        end
        M_sim = max(1, n_draws)
        draws = zeros(Float64, S, M_sim)
        mean_h = Float64.(hsi_mean)
        p_vec = _get_power_vec(friction_power, M_sim)
        if !isnothing(hsi_se)
            se_h = Float64.(hsi_se)
            for m in 1:M_sim
                p_m = p_vec[m]
                z = randn(rng, S)
                h_m = clamp.(mean_h .+ se_h .* z, 0.0, 1.0)
                draws[:, m] = [1.0 + 3.0 * (1.0 - h)^p_m for h in h_m]
            end
        else
            for m in 1:M_sim
                p_m = p_vec[m]
                h_m = clamp.(mean_h, 0.0, 1.0)
                draws[:, m] = [1.0 + 3.0 * (1.0 - h)^p_m for h in h_m]
            end
        end
        draws
    else
        throw(ArgumentError(
            "Either hsi_mean/se, hsi_samples, or resistance_mean/se must be provided."
        ))
    end

    M_total = size(r_draws, 2)
    all_paths = Vector{Vector{Int}}(undef, M_total)
    path_costs = zeros(Float64, M_total)
    path_distances = zeros(Float64, M_total)
    node_counts = zeros(Int, S)
    edge_counts = spzeros(Float64, S, S)

    # 2. Iterate across stochastic draws
    for m in 1:M_total
        r_m = r_draws[:, m]

        # Solve A* least-cost trajectory
        p_m = astar_least_cost_path(
            centroids,
            W,
            release,
            recapture;
            resistance = r_m,
            land_mask = land_mask,
            land_polygons = land_polygons
        )

        if smooth && length(p_m) > 2
            p_m = smooth_marine_path(
                p_m,
                centroids;
                land_polygons = land_polygons
            )
        end

        all_paths[m] = p_m

        # Accumulate node visits
        for u in p_m
            node_counts[u] += 1
        end

        # Accumulate edge visits and metrics
        dist_m = 0.0
        cost_m = 0.0
        for idx in 1:(length(p_m) - 1)
            u = p_m[idx]
            v = p_m[idx + 1]
            edge_counts[u, v] += 1.0
            d_uv = _spatial_node_distance(centroids[u], centroids[v])
            dist_m += d_uv
            cost_m += d_uv * (r_m[u] + r_m[v]) / 2.0
        end

        path_distances[m] = dist_m
        path_costs[m] = cost_m
    end

    # 3. Compute posterior corridor probabilities
    corridor_prob = node_counts ./ Float64(M_total)
    edge_prob = edge_counts ./ Float64(M_total)

    # 4. Identify medoid trajectory (highest mean corridor confidence)
    best_score = -Inf
    best_idx = 1
    for m in 1:M_total
        p_len = length(all_paths[m])
        score_m = p_len > 0 ? sum(corridor_prob[u] for u in all_paths[m]) / p_len : 0.0
        if score_m > best_score
            best_score = score_m
            best_idx = m
        end
    end
    medoid = all_paths[best_idx]

    # 5. Compute distance summary statistics
    mean_dist = mean(path_distances)
    alpha_lo = clamp(Float64(ci_alpha) / 2.0, 0.0, 0.5)
    alpha_hi = 1.0 - alpha_lo
    ci_dist = (quantile(path_distances, alpha_lo), quantile(path_distances, alpha_hi))

    return StochasticAStarResult(
        corridor_prob,
        edge_prob,
        medoid,
        all_paths,
        path_costs,
        path_distances,
        mean_dist,
        ci_dist,
        release,
        recapture,
        M_total
    )
end

"""
    astar_stochastic_predict_path(
        P::AbstractMatrix{<:Real},
        release::Int,
        recapture::Int;
        P_samples::Union{Nothing, Vector{<:AbstractMatrix{<:Real}}} = nothing,
        temperature::Real = 0.05,
        n_draws::Int = 50,
        centroids = nothing,
        k::Union{Nothing, Int} = nothing,
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
        ci_alpha::Real = 0.05,
        seed::Union{Nothing, Int} = nothing
    ) -> StochasticAStarResult

Performs stochastic A* path prediction across uncertain transition probability matrices.
"""
function astar_stochastic_predict_path(
    P::AbstractMatrix{<:Real},
    release::Int,
    recapture::Int;
    P_samples::Union{Nothing, Vector{<:AbstractMatrix{<:Real}}} = nothing,
    temperature::Real = 0.05,
    n_draws::Int = 50,
    centroids = nothing,
    k::Union{Nothing, Int} = nothing,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    ci_alpha::Real = 0.05,
    seed::Union{Nothing, Int} = nothing
)::StochasticAStarResult
    S = size(P, 1)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    M_total = !isnothing(P_samples) ? length(P_samples) : max(1, n_draws)

    all_paths = Vector{Vector{Int}}(undef, M_total)
    path_costs = zeros(Float64, M_total)
    path_distances = zeros(Float64, M_total)
    node_counts = zeros(Int, S)
    edge_counts = spzeros(Float64, S, S)
    temp = max(1e-6, Float64(temperature))

    for m in 1:M_total
        P_m = if !isnothing(P_samples)
            P_samples[m]
        else
            P_pert = copy(P)
            I_nz, J_nz, V_nz = findnz(sparse(P))
            V_new = zeros(Float64, length(V_nz))
            for idx in eachindex(V_nz)
                v = max(1e-12, Float64(V_nz[idx]))
                log_v = log(v) + temp * randn(rng)
                V_new[idx] = exp(log_v)
            end
            P_sparse = sparse(I_nz, J_nz, V_new, S, S)
            row_sums = vec(sum(P_sparse, dims=2))
            D_inv = Diagonal([rs > 0.0 ? 1.0 / rs : 1.0 for rs in row_sums])
            D_inv * P_sparse
        end

        p_m = astar_predict_path(
            P_m,
            release,
            recapture;
            centroids = centroids,
            k = k,
            land_mask = land_mask
        )

        all_paths[m] = p_m
        for u in p_m
            node_counts[u] += 1
        end

        dist_m = 0.0
        cost_m = 0.0
        for idx in 1:(length(p_m) - 1)
            u = p_m[idx]
            v = p_m[idx + 1]
            edge_counts[u, v] += 1.0
            if !isnothing(centroids)
                dist_m += _spatial_node_distance(centroids[u], centroids[v])
            else
                dist_m += 1.0
            end
            p_uv = P_m[u, v]
            cost_m += p_uv > 0.0 ? -log(p_uv) : 25.0
        end
        path_distances[m] = dist_m
        path_costs[m] = cost_m
    end

    corridor_prob = node_counts ./ Float64(M_total)
    edge_prob = edge_counts ./ Float64(M_total)

    best_score = -Inf
    best_idx = 1
    for m in 1:M_total
        p_len = length(all_paths[m])
        score_m = p_len > 0 ? sum(corridor_prob[u] for u in all_paths[m]) / p_len : 0.0
        if score_m > best_score
            best_score = score_m
            best_idx = m
        end
    end
    medoid = all_paths[best_idx]

    mean_dist = mean(path_distances)
    alpha_lo = clamp(Float64(ci_alpha) / 2.0, 0.0, 0.5)
    alpha_hi = 1.0 - alpha_lo
    ci_dist = (quantile(path_distances, alpha_lo), quantile(path_distances, alpha_hi))

    return StochasticAStarResult(
        corridor_prob,
        edge_prob,
        medoid,
        all_paths,
        path_costs,
        path_distances,
        mean_dist,
        ci_dist,
        release,
        recapture,
        M_total
    )
end

"""
    predict_path(P::AbstractMatrix{<:Real}, release::Int, recapture::Int,
                 k::Union{Nothing, Int}=nothing;
                 centroids=nothing, method=:astar, land_mask=nothing) -> Vector{Int}

Computes the single most likely sequence of spatial units visited by an individual
between `release` and `recapture` using either goal-directed \$A^*\$ heuristic search
(`method=:astar`, default) or the classic fixed-horizon Viterbi trellis (`method=:viterbi`).

# Arguments
- `P`: Row-stochastic transition matrix (size ``S \\times S``).
- `release`: Starting spatial unit index (1-indexed).
- `recapture`: Destination spatial unit index (1-indexed).
- `k`: Optional discrete time steps elapsed between release and recapture (``k \\ge 1``).
- `centroids`: Optional spatial centroids coordinate vector for \$A^*\$ distance heuristic.
- `method`: Algorithm selector (`:astar` for high-performance \$A^*\$,
  `:viterbi` for classic trellis).
- `land_mask`: Optional boolean vector of length ``S`` (`true` for land units).

# Returns
- `Vector{Int}`: Sequence of spatial unit indices from `release` to `recapture`.
"""
function predict_path(
    P::AbstractMatrix{<:Real},
    release::Int,
    recapture::Int,
    k::Union{Nothing, Int} = nothing;
    centroids = nothing,
    method::Symbol = :astar,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Vector{Int}
    if k === nothing || method == :astar
        return astar_predict_path(
            P, release, recapture;
            centroids = centroids,
            k = k,
            land_mask = land_mask
        )
    elseif method == :viterbi
        S = size(P, 1)
        if !(1 <= release <= S) || !(1 <= recapture <= S)
            throw(ArgumentError(
                "Release unit ($release) and recapture unit ($recapture) must be within 1:$S."
            ))
        end
        if k < 1
            return [release]
        elseif k == 1
            return [release, recapture]
        end

        logP = Matrix{Float64}(undef, S, S)
        @inbounds for j in 1:S
            for i in 1:S
                p_ij = Float64(P[i, j])
                logP[i, j] = p_ij > 1e-15 ? log(p_ij) : -1e12
            end
        end

        if land_mask !== nothing
            for l in 1:S
                if land_mask[l]
                    logP[:, l] .= -1e12
                    logP[l, :] .= -1e12
                end
            end
        end

        delta = fill(-1e12, S, k + 1)
        psi   = zeros(Int, S, k + 1)
        delta[release, 1] = 0.0

        for tau in 2:(k + 1)
            prev_tau = tau - 1
            for j in 1:S
                best_val = -Inf
                best_prev = 1
                for i in 1:S
                    score = delta[i, prev_tau] + logP[i, j]
                    if score > best_val
                        best_val = score
                        best_prev = i
                    end
                end
                delta[j, tau] = best_val
                psi[j, tau]   = best_prev
            end
        end

        path = zeros(Int, k + 1)
        path[k + 1] = recapture
        for tau in (k + 1):-1:2
            path[tau - 1] = psi[path[tau], tau]
        end
        return path
    else
        throw(ArgumentError("Unknown path method '$method'. Expected :astar or :viterbi."))
    end
end


"""
    predict_corridor(P::AbstractMatrix{<:Real}, release::Int, recapture::Int, k::Int;
                     land_mask=nothing) -> Matrix{Float64}

Computes the Markov bridge probability distribution across all spatial units at each
intermediate time step ``\\tau \\in \\{0, 1, \\dots, k\\}``:

```math
\\mathbb{P}(X_\\tau = j \\mid X_0 = u_{\\text{rel}}, X_k = u_{\\text{rec}}) = 
\\frac{[P^\\tau]_{u_{\\text{rel}}, j} \\cdot [P^{k - \\tau}]_{j, u_{\\text{rec}}}}{[P^k]_{u_{\\text{rel}}, u_{\\text{rec}}}}
```

# Arguments
- `P`: Row-stochastic transition matrix (size ``S \\times S``).
- `release`: Starting spatial unit index (1-indexed).
- `recapture`: Destination spatial unit index at step `k` (1-indexed).
- `k`: Total discrete time steps elapsed between release and recapture (``k \\ge 1``).
- `land_mask`: Optional boolean vector of length ``S`` (`true` for land units). When provided,
  all probability mass on land units is strictly zeroed out and columns are renormalized
  over navigable marine units.

# Returns
- `Matrix{Float64}`: Array of size ``(S, k + 1)`` where column ``\\tau + 1`` gives the spatial
  probability distribution over all ``S`` units at time step ``\\tau``. Each column sums to 1.0.
"""
function predict_corridor(
    P::AbstractMatrix{<:Real},
    release::Int,
    recapture::Int,
    k::Int;
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Matrix{Float64}
    S = size(P, 1)
    if !(1 <= release <= S) || !(1 <= recapture <= S)
        error("Release unit ($release) and recapture unit ($recapture) must be within 1:$S.")
    end

    P_mat = Matrix{Float64}(P)
    corridor = zeros(Float64, S, k + 1)

    if k <= 0
        corridor[release, 1] = 1.0
        return corridor
    end

    # Precompute powers P^tau for tau = 0 to k
    P_powers = Vector{Matrix{Float64}}(undef, k + 1)
    P_powers[1] = Matrix{Float64}(I, S, S)
    for tau in 1:k
        P_powers[tau + 1] = P_powers[tau] * P_mat
    end

    # Total likelihood of transitioning from release to recapture in k steps
    P_total = P_powers[k + 1][release, recapture]

    if P_total <= 1e-15
        @warn "Recapture unit $recapture has near-zero reachability from release $release in $k steps."
        if land_mask !== nothing
            n_water = count(!, land_mask)
            w_val = n_water > 0 ? 1.0 / n_water : 1.0 / S
            for j in 1:S
                corridor[j, :] .= land_mask[j] ? 0.0 : w_val
            end
        else
            corridor[:, :] .= 1.0 / S
        end
        corridor[release, 1] = 1.0
        corridor[recapture, k + 1] = 1.0
        return corridor
    end

    # Markov bridge formula for each intermediate step tau = 0 .. k
    for tau in 0:k
        tau_idx = tau + 1
        rem_idx = (k - tau) + 1
        for j in 1:S
            prob_fwd = P_powers[tau_idx][release, j]
            prob_bwd = P_powers[rem_idx][j, recapture]
            corridor[j, tau_idx] = (prob_fwd * prob_bwd) / P_total
        end
        # Enforce land barrier: strictly zero probability on land
        if land_mask !== nothing
            for l in 1:S
                if land_mask[l]
                    corridor[l, tau_idx] = 0.0
                end
            end
        end
        # Normalize column
        col_sum = sum(view(corridor, :, tau_idx))
        if col_sum > 0.0
            corridor[:, tau_idx] ./= col_sum
        end
    end

    return corridor
end


# Convenience overloads for result dictionaries / named tuples
function predict_path(
    res::NamedTuple,
    release::Int,
    recapture::Int,
    k::Union{Nothing, Int} = nothing;
    group::Union{String, Symbol, Int} = 1,
    centroids = nothing,
    method::Symbol = :astar,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Vector{Int}
    mask = land_mask !== nothing ? land_mask :
        (hasproperty(res, :land_mask) ? res.land_mask : nothing)
    cents = centroids !== nothing ? centroids :
        (hasproperty(res, :mesh) ? res.mesh : nothing)
    if hasproperty(res, :transition_matrices)
        tm = res.transition_matrices
        key = group isa Int ?
            (hasproperty(res, :group_lookup) ? res.group_lookup[group] : first(keys(tm))) :
            string(group)
        P = tm[key]
        return predict_path(
            P, release, recapture, k;
            centroids = cents,
            method = method,
            land_mask = mask
        )
    else
        throw(ArgumentError("Expected a result NamedTuple with field `:transition_matrices`."))
    end
end

function predict_corridor(
    res::NamedTuple, release::Int, recapture::Int, k::Int;
    group::Union{String, Symbol, Int} = 1,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)
    mask = land_mask !== nothing ? land_mask :
        (hasproperty(res, :land_mask) ? res.land_mask : nothing)
    if hasproperty(res, :transition_matrices)
        tm = res.transition_matrices
        key = group isa Int ?
            (hasproperty(res, :group_lookup) ? res.group_lookup[group] : first(keys(tm))) :
            string(group)
        P = tm[key]
        return predict_corridor(P, release, recapture, k; land_mask=mask)
    else
        error("Expected a result NamedTuple with field `:transition_matrices`.")
    end
end

"""
    predict_path(P_vec::AbstractVector{<:AbstractMatrix{<:Real}}, release::Int, recapture::Int, k::Int;
                 group::Union{Integer, Symbol, AbstractString} = 1,
                 land_mask::Union{Nothing, AbstractVector{Bool}} = nothing) -> Vector{Int}

Group-aware overload for `predict_path` when given a vector of group-specific transition matrices.
Reconstructs the most likely sequence of spatial units (Viterbi path) for the specified group.

# Mathematical Formulation
Given group index ``g``, the dynamic programming Viterbi trellis identifies:
```math
\\mathbf{s}^* = \\arg\\max_{\\mathbf{s}} \\prod_{\\tau=1}^k [P_g]_{s_{\\tau-1}, s_\\tau}
```
subject to ``s_0 = u_{\\text{rel}}`` and ``s_k = u_{\\text{rec}}``.

# Arguments
- `P_vec`: Collection of row-stochastic transition matrices for each biological group.
- `release`: Starting spatial unit index (1-indexed).
- `recapture`: Destination spatial unit index at step `k` (1-indexed).
- `k`: Number of discrete time intervals elapsed.
- `group`: 1-based integer index or group label.
- `land_mask`: Optional boolean mask of impermeable barrier units.

# Returns
- `Vector{Int}`: Sequence of ``k + 1`` spatial unit indices from `release` to `recapture`.
"""
function predict_path(
    P_vec::AbstractVector{<:AbstractMatrix{<:Real}},
    release::Int,
    recapture::Int,
    k::Union{Nothing, Int} = nothing;
    group::Union{Integer, Symbol, AbstractString} = 1,
    centroids = nothing,
    method::Symbol = :astar,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Vector{Int}
    if isempty(P_vec)
        throw(ArgumentError("P_vec transition kernel vector cannot be empty."))
    end
    g_idx = if group isa Integer
        Int(group)
    else
        parsed = tryparse(Int, string(group))
        parsed !== nothing ? parsed : 1
    end
    if !(1 <= g_idx <= length(P_vec))
        throw(ArgumentError("Group index $g_idx is out of bounds (1:$(length(P_vec)))."))
    end
    return predict_path(
        P_vec[g_idx], release, recapture, k;
        centroids = centroids,
        method = method,
        land_mask = land_mask
    )
end

"""
    predict_corridor(P_vec::AbstractVector{<:AbstractMatrix{<:Real}}, release::Int, recapture::Int, k::Int;
                     group::Union{Integer, Symbol, AbstractString} = 1,
                     land_mask::Union{Nothing, AbstractVector{Bool}} = nothing) -> Matrix{Float64}

Group-aware overload for `predict_corridor` when given a vector of group-specific transition matrices.
Computes the Markov bridge probability distribution across spatial units for the specified group.

# Mathematical Formulation
Given group index ``g``, the Markov bridge probability at intermediate step ``\\tau`` is:
```math
\\mathbb{P}(X_\\tau = j \\mid X_0 = u_{\\text{rel}}, X_k = u_{\\text{rec}}) = 
\\frac{[P_g^\\tau]_{u_{\\text{rel}}, j} \\cdot [P_g^{k - \\tau}]_{j, u_{\\text{rec}}}}{[P_g^k]_{u_{\\text{rel}}, u_{\\text{rec}}}}
```

# Arguments
- `P_vec`: Collection of row-stochastic transition matrices for each biological group.
- `release`: Starting spatial unit index (1-indexed).
- `recapture`: Destination spatial unit index at step `k` (1-indexed).
- `k`: Number of discrete time intervals elapsed.
- `group`: 1-based integer index or group label.
- `land_mask`: Optional boolean mask of impermeable barrier units.

# Returns
- `Matrix{Float64}`: Array of size ``(S, k + 1)`` where column ``\\tau + 1`` gives the spatial
  probability distribution over all units at time step ``\\tau``.
"""
function predict_corridor(
    P_vec::AbstractVector{<:AbstractMatrix{<:Real}},
    release::Int,
    recapture::Int,
    k::Int;
    group::Union{Integer, Symbol, AbstractString} = 1,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Matrix{Float64}
    if isempty(P_vec)
        throw(ArgumentError("P_vec transition kernel vector cannot be empty."))
    end
    g_idx = if group isa Integer
        Int(group)
    else
        parsed = tryparse(Int, string(group))
        parsed !== nothing ? parsed : 1
    end
    if !(1 <= g_idx <= length(P_vec))
        throw(ArgumentError("Group index $g_idx is out of bounds (1:$(length(P_vec)))."))
    end
    return predict_corridor(P_vec[g_idx], release, recapture, k; land_mask=land_mask)
end


# ==============================================================================
# Domain Tessellation, Autocorrelation Infilling & Land Barrier Engine
# ==============================================================================

"""
    point_in_polygon(x::Real, y::Real, poly) -> Bool

Tests whether planar or geographic coordinate `(x, y)` lies inside a closed 2D polygon
or MultiPolygon using the classical ray-casting (even-odd crossing) algorithm.

# Mathematical Formulation
A horizontal ray is cast from point ``(x, y)`` along the positive ``x``-axis to ``+\\infty``.
For each polygon edge connecting vertices ``(x_i, y_i)`` and ``(x_j, y_j)``:
```math
\\text{crosses}(e) = ((y_i > y) \\ne (y_j > y)) \\land 
\\left( x < \\frac{(x_j - x_i)(y - y_i)}{y_j - y_i} + x_i \\right)
```
The query point is inside if and only if the total edge crossing count is odd:
```math
\\text{inside} = \\left( \\sum_{e \\in E} \\mathbb{I}[\\text{crosses}(e)] \\right) \\pmod 2 = 1
```

# Arguments
- `x::Real`: Longitude or planar x-coordinate of query point.
- `y::Real`: Latitude or planar y-coordinate of query point.
- `poly`: Polygon geometry representation. Supports:
  - `Vector{Tuple{<:Real, <:Real}}` or `Vector{<:AbstractVector{<:Real}}`: Single polygon ring.
  - `Vector{Vector{...}}`: Collection of polygon rings (MultiPolygon / archipelago).
  - `LibGEOS.AbstractGeometry`: Geometric polygon object via LibGEOS.

# Returns
- `Bool`: `true` if `(x, y)` is inside the polygon interior, `false` otherwise.
"""
function point_in_polygon(x::Real, y::Real, poly)::Bool
    if poly isa AbstractVector && !isempty(poly) && first(poly) isa AbstractVector
        for sub_poly in poly
            if point_in_polygon(x, y, sub_poly)
                return true
            end
        end
        return false
    end

    n = length(poly)
    n < 3 && return false
    inside = false
    j = n
    px = Float64(x)
    py = Float64(y)
    @inbounds for i in 1:n
        pt_i = poly[i]
        pt_j = poly[j]
        xi = Float64(pt_i[1])
        yi = Float64(pt_i[2])
        xj = Float64(pt_j[1])
        yj = Float64(pt_j[2])

        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
            inside = !inside
        end
        j = i
    end
    return inside
end

function point_in_polygon(x::Real, y::Real, geom::LibGEOS.AbstractGeometry)::Bool
    pt = LibGEOS.Point(Float64(x), Float64(y))
    return LibGEOS.contains(geom, pt)
end


"""
Curated geographic boundary coordinates for the Canadian Maritime region, enclosing
major terrestrial landmasses while preserving marine straits (Cabot Strait, Northumberland
Strait, Bay of Fundy, Chedabucto Bay, and Scotian Shelf coastal waters).
"""
const _DEFAULT_MARITIMES_LAND_POLYGONS = [
    # 1. Cape Breton Island
    [
        (-60.45, 47.05), (-60.38, 46.90), (-60.38, 46.65), (-60.45, 46.30),
        (-60.15, 46.25), (-59.75, 46.15), (-59.85, 45.85), (-60.50, 45.60),
        (-61.00, 45.48), (-61.40, 45.65), (-61.50, 45.90), (-61.25, 46.25),
        (-61.05, 46.65), (-60.75, 46.90), (-60.55, 47.05), (-60.45, 47.05)
    ],
    # 2. Nova Scotia Mainland
    [
        (-61.40, 45.60), (-61.50, 45.45), (-61.20, 45.35), (-60.99, 45.33),
        (-61.50, 45.10), (-62.50, 44.80), (-63.50, 44.55), (-64.20, 44.40),
        (-64.70, 44.00), (-65.30, 43.50), (-65.65, 43.40), (-66.15, 43.80),
        (-66.25, 44.20), (-65.75, 44.65), (-65.00, 45.15), (-64.35, 45.35),
        (-64.40, 45.85), (-63.70, 45.85), (-62.60, 45.75), (-61.90, 45.75),
        (-61.40, 45.60)
    ],
    # 3. Prince Edward Island
    [
        (-64.40, 46.95), (-64.00, 46.60), (-63.50, 46.50), (-62.50, 46.45),
        (-61.95, 46.40), (-62.40, 46.00), (-63.00, 45.95), (-63.80, 46.20),
        (-64.40, 46.60), (-64.40, 46.95)
    ],
    # 4. New Brunswick & Quebec / Gaspe coast
    [
        (-64.40, 45.85), (-64.50, 45.80), (-64.70, 45.50), (-65.50, 45.30),
        (-67.00, 45.00), (-68.50, 45.50), (-69.50, 48.00), (-68.00, 49.00),
        (-65.00, 49.20), (-64.20, 48.80), (-64.80, 48.00), (-64.80, 47.00),
        (-64.50, 46.20), (-64.40, 45.85)
    ]
]


"""
    identify_land_units(
        centroids::AbstractVector;
        land_polygons::Union{Nothing, Symbol, AbstractVector} = nothing,
        depth::Union{Nothing, AbstractVector{<:Real}} = nothing,
        depth_threshold::Real = 0.0
    ) -> Vector{Bool}

Classifies spatial partitioning units as terrestrial land (`true`) or navigable marine water
(`false`) using boundary polygons, bathymetric depth thresholds, or regional defaults.

# Mathematical Formulation
A spatial unit ``s \\in \\{1, \\dots, S\\}`` with centroid ``\\mathbf{c}_s = (x_s, y_s)`` is classified
as land if it satisfies either geometric boundary polygon inclusion or the bathymetric threshold:
```math
\\text{is\\_land}(s) = (\\exists p \\in \\mathcal{P} : \\mathbf{c}_s \\in p) \\lor (d_s \\le d_{\\text{thresh}})
```
where ``\\mathcal{P}`` is the set of land boundary polygons and ``d_s`` is the bathymetric depth.

# Arguments
- `centroids`: Vector of coordinate tuples `(x, y)` or `(lon, lat)` for all ``S`` units.
- `land_polygons`: Optional polygon or collection of polygons.
  - `nothing`: If `depth` is also `nothing`, defaults to `:maritimes` (curated NW Atlantic coastline).
  - `:maritimes` or `:default`: Uses built-in Nova Scotia, Cape Breton, PEI, and NB/QC polygons.
  - `Vector{...}`: User-provided polygon rings or MultiPolygon coordinate vectors.
- `depth`: Optional bathymetric depth vector of length ``S`` (positive values indicate water depth).
- `depth_threshold`: Scalar threshold below which a unit is classified as land (default `0.0`).

# Returns
- `Vector{Bool}`: Boolean mask of length ``S`` where `true` indicates a terrestrial land unit.
"""
function identify_land_units(
    centroids::AbstractVector;
    polygons::Union{Nothing, AbstractVector} = nothing,
    land_polygons::Union{Nothing, Symbol, AbstractVector} = nothing,
    depth::Union{Nothing, AbstractVector{<:Real}} = nothing,
    depth_threshold::Real = 0.0,
    land_fraction_threshold::Real = 0.5
)::Vector{Bool}
    S = length(centroids)
    land_mask = falses(S)

    # 1. Depth thresholding
    if depth !== nothing
        if length(depth) != S
            throw(DimensionMismatch(
                "Depth vector length ($(length(depth))) must match centroids length ($S)."
            ))
        end
        for i in 1:S
            if isfinite(depth[i]) && depth[i] <= depth_threshold
                land_mask[i] = true
            end
        end
    end

    # 2. Polygon boundaries
    polys = if land_polygons === nothing
        _DEFAULT_MARITIMES_LAND_POLYGONS
    elseif land_polygons in (:none, :false, false)
        nothing
    elseif land_polygons in (:maritimes, :default)
        _DEFAULT_MARITIMES_LAND_POLYGONS
    else
        land_polygons
    end

    if polys !== nothing
        for i in 1:S
            land_mask[i] && continue
            c = centroids[i]
            x, y = Float64(c[1]), Float64(c[2])
            # Check centroid inclusion
            if point_in_polygon(x, y, polys)
                land_mask[i] = true
                continue
            end
            # Check unit polygon vertices if provided
            if polygons !== nothing && i <= length(polygons)
                u_poly = polygons[i]
                if length(u_poly) >= 3
                    n_verts = length(u_poly) - (u_poly[1] == u_poly[end] ? 1 : 0)
                    n_in = count(k -> point_in_polygon(u_poly[k][1], u_poly[k][2], polys), 1:n_verts)
                    if n_in >= ceil(Int, n_verts * land_fraction_threshold)
                        land_mask[i] = true
                    end
                end
            end
        end
    end

    return land_mask
end


"""
    apply_land_barrier(
        W::AbstractMatrix{<:Real},
        hsi::AbstractVector{<:Real},
        land_mask::AbstractVector{Bool}
    ) -> Tuple{SparseMatrixCSC{Float64, Int}, Vector{Float64}}

Enforces impassable terrestrial movement barriers on the spatial adjacency graph ``W`` and
habitat suitability vector ``\\mathbf{h}``, severing all topological connections across land.

# Mathematical Formulation
Let ``\\mathcal{L} = \\{l \\mid \\text{land\\_mask}[l] = \\text{true}\\}`` be the set of land units,
and ``\\mathcal{W} = \\{w \\mid \\text{land\\_mask}[w] = \\text{false}\\}`` be marine water units.
All directed and undirected edges incident to any land unit are strictly eliminated:
```math
W^{\\text{water}}_{ij} = \\begin{cases} 
W_{ij} & \\text{if } i \\in \\mathcal{W} \\land j \\in \\mathcal{W} \\\\
0 & \\text{otherwise}
\\end{cases}
```
Habitat suitability values on land are set to zero:
```math
h^{\\text{water}}_i = \\begin{cases}
h_i & \\text{if } i \\in \\mathcal{W} \\\\
0.0 & \\text{if } i \\in \\mathcal{L}
\\end{cases}
```
This guarantees that topological random walks (``T_{\\text{diff}}``) and directed advection (``A``)
have zero probability of transitioning onto or through land masses:
```math
P(X_{t+1} \\in \\mathcal{L} \\mid X_t \\in \\mathcal{W}) = 0
```

# Arguments
- `W`: Spatial graph adjacency matrix (size ``S \\times S``).
- `hsi`: Habitat suitability vector (length ``S``).
- `land_mask`: Boolean vector of length ``S`` (`true` for land units).

# Returns
- `Tuple{SparseMatrixCSC{Float64, Int}, Vector{Float64}}`:
  - `W_water`: Severed adjacency matrix with all land connections removed and zeros dropped.
  - `hsi_water`: Habitat suitability vector with land units zeroed out.
"""
function apply_land_barrier(
    W::AbstractMatrix{<:Real},
    hsi::AbstractVector{<:Real},
    land_mask::AbstractVector{Bool}
)::Tuple{SparseMatrixCSC{Float64, Int}, Vector{Float64}}
    S = size(W, 1)
    if length(hsi) != S || length(land_mask) != S
        throw(DimensionMismatch(
            "Dimension mismatch: W is $(S)x$(S), hsi has length $(length(hsi)), " *
            "and land_mask has length $(length(land_mask))."
        ))
    end

    W_water = copy(sparse(Float64.(W)))
    for l in findall(land_mask)
        W_water[l, :] .= 0.0
        W_water[:, l] .= 0.0
    end
    dropzeros!(W_water)

    hsi_water = copy(Float64.(hsi))
    hsi_water[land_mask] .= 0.0

    return (W_water, hsi_water)
end


function _extract_polygon_rings(polys)
    if polys isa AbstractVector && !isempty(polys)
        first_elem = first(polys)
        if first_elem isa Tuple || (first_elem isa AbstractVector && length(first_elem) == 2 && first_elem[1] isa Real)
            return [polys]
        elseif first_elem isa AbstractVector
            return polys
        end
    end
    return [polys]
end

function _line_crosses_polygon_or_in(p1, p2, polys)::Bool
    # Check midpoint inclusion in land polygon
    mid_x = (p1[1] + p2[1]) / 2.0
    mid_y = (p1[2] + p2[2]) / 2.0
    if point_in_polygon(mid_x, mid_y, polys)
        return true
    end

    # Orientation test for 2D line segment intersection
    function _ccw(A, B, C)
        return (C[2] - A[2]) * (B[1] - A[1]) > (B[2] - A[2]) * (C[1] - A[1])
    end

    function _seg_intersect(a1, a2, b1, b2)
        return (_ccw(a1, b1, b2) != _ccw(a2, b1, b2)) && (_ccw(a1, a2, b1) != _ccw(a1, a2, b2))
    end

    rings = _extract_polygon_rings(polys)
    for ring in rings
        if ring isa AbstractVector && length(ring) >= 3
            n_pts = length(ring)
            for k in 1:(n_pts - 1)
                e1 = (Float64(ring[k][1]), Float64(ring[k][2]))
                e2 = (Float64(ring[k + 1][1]), Float64(ring[k + 1][2]))
                if _seg_intersect(p1, p2, e1, e2)
                    return true
                end
            end
        end
    end
    return false
end

"""
    sever_land_crossing_edges!(
        W::AbstractMatrix{<:Real},
        centroids::AbstractVector;
        land_polygons = nothing
    ) -> Int

Inspects all non-zero edges ``(i, j)`` in the spatial adjacency matrix ``W`` and
severs any edge whose straight-line trajectory intersects a terrestrial land barrier polygon.

# Mathematical Formulation
For nodes ``i, j \\in \\mathcal{S}``, the trajectory line segment is:
```math
L_{ij} = \\{ (1 - t) \\mathbf{c}_i + t \\mathbf{c}_j \\mid t \\in [0, 1] \\}
```
If ``L_{ij} \\cap \\mathcal{P}_{\\text{land}} \\neq \\emptyset``, the edge is severed:
```math
W_{ij} \\leftarrow 0, \\quad W_{ji} \\leftarrow 0
```
preventing movement models from allowing transitions that cross overland barriers.

# Arguments
- `W`: Spatial graph adjacency matrix (size ``S \\times S``). Modified in-place if mutable.
- `centroids`: Centroids coordinate vector (length ``S``).
- `land_polygons`: Boundary polygons (defaults to `:maritimes`).

# Returns
- `Int`: Total number of directed edges severed.
"""
function sever_land_crossing_edges!(
    W::AbstractMatrix{<:Real},
    centroids::AbstractVector;
    land_polygons = nothing
)::Int
    polys = if land_polygons === nothing
        _DEFAULT_MARITIMES_LAND_POLYGONS
    elseif land_polygons in (:none, :false, false)
        return 0
    elseif land_polygons in (:maritimes, :default)
        _DEFAULT_MARITIMES_LAND_POLYGONS
    else
        land_polygons
    end

    polys === nothing && return 0

    S = size(W, 1)
    if length(centroids) != S
        throw(DimensionMismatch(
            "centroids length ($(length(centroids))) must match W dimensions ($S)."
        ))
    end

    severed_count = 0
    if W isa SparseMatrixCSC
        rows = rowvals(W)
        for col in 1:S
            c_col = centroids[col]
            p_col = (Float64(c_col[1]), Float64(c_col[2]))
            for k in nzrange(W, col)
                row = rows[k]
                if row > col
                    c_row = centroids[row]
                    p_row = (Float64(c_row[1]), Float64(c_row[2]))
                    if _line_crosses_polygon_or_in(p_col, p_row, polys)
                        W[row, col] = 0.0
                        W[col, row] = 0.0
                        severed_count += 2
                    end
                end
            end
        end
        dropzeros!(W)
    else
        for i in 1:S
            c_i = centroids[i]
            p_i = (Float64(c_i[1]), Float64(c_i[2]))
            for j in (i + 1):S
                if W[i, j] != 0.0 || W[j, i] != 0.0
                    c_j = centroids[j]
                    p_j = (Float64(c_j[1]), Float64(c_j[2]))
                    if _line_crosses_polygon_or_in(p_i, p_j, polys)
                        W[i, j] = 0.0
                        W[j, i] = 0.0
                        severed_count += 2
                    end
                end
            end
        end
    end

    return severed_count
end

"""
    compact_marine_mesh(mesh, land_mask::AbstractVector{Bool}) -> NamedTuple

Extracts and compacts a spatial mesh into an active marine water sub-domain by
filtering out all terrestrial land units and reindexing spatial units from ``1`` to
``S_{\\text{water}}``.

# Mathematical Formulation
Given spatial unit set ``\\mathcal{S} = \\{1, \\dots, S\\}`` and boolean indicator
``\\text{land\\_mask}``, the navigable marine sub-domain is defined by:
```math
\\mathcal{M} = \\{ i \\in \\mathcal{S} \\mid \\neg \\text{land\\_mask}[i] \\}
```
The adjacency graph is subsetted to the induced sub-graph:
```math
W_{\\text{marine}} = W[\\mathcal{M}, \\mathcal{M}]
```
preserving all marine graph topology while reducing state dimensionality.

# Arguments
- `mesh`: Spatial mesh NamedTuple containing `:centroids`, `:polygons`, `:n_units`, and `:W`.
- `land_mask`: Boolean vector of length ``S`` (`true` for land units).

# Returns
- `NamedTuple`: Compacted marine mesh with updated `:n_units`, `:W`, `:centroids_lonlat`,
  `:polygons_lonlat`, and the mapping vector `:water_indices`.
"""
function compact_marine_mesh(mesh, land_mask::AbstractVector{Bool})::NamedTuple
    water_idx = findall(!, land_mask)
    S_water = length(water_idx)

    cents_ll = hasproperty(mesh, :centroids_lonlat) ? mesh.centroids_lonlat[water_idx] :
               (hasproperty(mesh, :centroids) ? mesh.centroids[water_idx] : Tuple{Float64, Float64}[])
    polys_ll = hasproperty(mesh, :polygons_lonlat) ? mesh.polygons_lonlat[water_idx] :
               (hasproperty(mesh, :polygons) ? mesh.polygons[water_idx] : Vector{Vector{Tuple{Float64, Float64}}}())
    cents_km = hasproperty(mesh, :centroids_km) ? mesh.centroids_km[water_idx] : nothing
    polys_km = hasproperty(mesh, :polygons_km) ? mesh.polygons_km[water_idx] : nothing

    W_sub = copy(mesh.W[water_idx, water_idx])
    if W_sub isa SparseMatrixCSC
        dropzeros!(W_sub)
    end

    res = Dict{Symbol, Any}(
        :n_units          => S_water,
        :centroids        => cents_ll,
        :centroids_lonlat => cents_ll,
        :polygons         => polys_ll,
        :polygons_lonlat  => polys_ll,
        :W                => W_sub,
        :water_indices    => water_idx
    )
    if cents_km !== nothing
        res[:centroids_km] = cents_km
    end
    if polys_km !== nothing
        res[:polygons_km] = polys_km
    end
    if hasproperty(mesh, :radius_km)
        res[:radius_km] = mesh.radius_km
    end
    if hasproperty(mesh, :areas_km2)
        res[:areas_km2] = mesh.areas_km2[water_idx]
    end

    return NamedTuple(res)
end


"""
    infill_spatial_hsi(
        hsi_raw::AbstractVector{<:Real},
        W::AbstractMatrix{<:Real},
        known_mask::AbstractVector{Bool};
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
        centroids::Union{Nothing, AbstractVector} = nothing,
        rho::Real = 0.95,
        baseline_quantile::Real = 0.1,
        fallback_decay_km::Real = 50.0
    ) -> Vector{Float64}

Infills missing Habitat Suitability Index (HSI) values outside the primary survey domain
using a data-driven graph-Laplacian screened Dirichlet system constrained to marine water channels.

# Mathematical Formulation
The spatial domain is partitioned into known survey units ``\\mathcal{K}``, unknown marine units
``\\mathcal{U}``, and terrestrial land units ``\\mathcal{L}``. On the severed water graph
``W^{\\text{water}}``, the conditional expectation of unknown HSI values satisfies:
```math
(D_{\\mathcal{UU}} - \\rho W_{\\mathcal{UU}}) \\mathbf{h}_{\\mathcal{U}} = 
\\rho W_{\\mathcal{UK}} \\mathbf{h}_{\\mathcal{K}} + (1 - \\rho) D_{\\mathcal{UU}} \\mu_0 \\mathbf{1}
```
where:
- ``D_{\\mathcal{UU}} = \\operatorname{diag}\\left( \\sum_{j \\in \\mathcal{W}} W_{ij} \\right)_{i \\in \\mathcal{U}}``
  is the total degree of each unknown node in the water graph.
- ``W_{\\mathcal{UU}}`` is the internal adjacency submatrix among unknown water nodes.
- ``W_{\\mathcal{UK}}`` connects unknown water nodes to adjacent known survey nodes.
- ``\\rho \\in (0, 1)`` controls spatial autocorrelation strength (default 0.95).
- ``\\mu_0 = \\operatorname{quantile}(\\mathbf{h}_{\\mathcal{K}}, \\text{baseline\\_quantile})``
  is the empirical conservative baseline HSI for distant unsampled marine waters.

Because ``W^{\\text{water}}`` has all land edges severed, the Dirichlet diffusion flows strictly
around peninsulas and through marine straits (e.g. Cabot Strait), never jumping across land.
Isolated water components without graph connections to known units decay smoothly to ``\\mu_0``
or utilize an exponential spatial covariance fallback ``K(d) = \\exp(-d / \\ell)``.

# Arguments
- `hsi_raw`: Vector of observed HSI values (length ``S``).
- `W`: Spatial graph adjacency matrix (size ``S \\times S``).
- `known_mask`: Boolean vector of length ``S`` (`true` where HSI is observed).
- `land_mask`: Optional boolean vector of length ``S`` (`true` for land units).
- `centroids`: Optional coordinates vector for spatial distance fallback on disconnected units.
- `rho`: Spatial autocorrelation strength parameter in ``(0, 1)`` (default 0.95).
- `baseline`: Optional scalar baseline prior mean for unobserved regions. If `nothing`,
  defaults to empirical quantile `baseline_quantile` of known values.
- `baseline_quantile`: Empirical quantile of known HSI used as prior baseline (default 0.1).
- `fallback_decay_km`: Spatial correlation length scale in km for disconnected units (default 50.0).

# Returns
- `Vector{Float64}`: Infilled HSI vector of length ``S`` bounded in ``[0, 1]`` with land units set to 0.0.
"""
function infill_spatial_hsi(
    hsi_raw::AbstractVector{<:Real},
    W::AbstractMatrix{<:Real},
    known_mask::AbstractVector{Bool};
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    centroids::Union{Nothing, AbstractVector} = nothing,
    rho::Real = 0.95,
    baseline::Union{Nothing, Real} = nothing,
    baseline_quantile::Real = 0.1,
    fallback_decay_km::Real = 50.0
)::Vector{Float64}
    S = size(W, 1)
    if length(hsi_raw) != S || length(known_mask) != S
        throw(DimensionMismatch("hsi_raw, W, and known_mask dimensions must match."))
    end

    mask_l = land_mask === nothing ? falses(S) : land_mask
    hsi_out = copy(Float64.(hsi_raw))

    # Land units are set to 0.0
    hsi_out[mask_l] .= 0.0

    # Unknown water units
    unknown_idx = findall(.!known_mask .& .!mask_l)
    known_idx   = findall(known_mask .& .!mask_l)

    if isempty(unknown_idx)
        return hsi_out
    end

    # Baseline prior mean from known survey values or explicit baseline
    known_vals = filter(isfinite, hsi_out[known_idx])
    mu0 = if baseline !== nothing
        Float64(baseline)
    elseif isempty(known_vals)
        0.2
    else
        Float64(quantile(known_vals, baseline_quantile))
    end

    # Graph degree vector (summing only water connections)
    W_sp = sparse(W)
    D = vec(sum(W_sp, dims=2))

    W_uu = W_sp[unknown_idx, unknown_idx]
    W_uk = W_sp[unknown_idx, known_idx]
    D_uu = Diagonal(D[unknown_idx])

    rho_val = clamp(Float64(rho), 0.5, 0.999)
    A_mat = D_uu - rho_val * W_uu
    b_vec = rho_val * (W_uk * hsi_out[known_idx]) + (1.0 - rho_val) * (D[unknown_idx] .* mu0)

    # For any isolated nodes (degree == 0), regularize diagonal to yield mu0
    for (k_sub, u) in enumerate(unknown_idx)
        if D[u] < 1e-12
            A_mat[k_sub, k_sub] = 1.0
            b_vec[k_sub] = mu0
        end
    end

    # Solve the screened Dirichlet system
    h_u = try
        A_mat \ b_vec
    catch e
        @warn "infill_spatial_hsi: direct solve failed ($e); using regularized solve."
        (A_mat + 1e-4 * I) \ b_vec
    end

    # Fallback distance decay for units with no direct path to survey area
    if centroids !== nothing && !isempty(known_idx)
        tree_known = KDTree(hcat([[Float64(c[1]), Float64(c[2])] for c in centroids[known_idx]]...))
        for (k_sub, u) in enumerate(unknown_idx)
            has_survey_conn = sum(W_uk[k_sub, :]) > 0.0
            if !has_survey_conn
                c_u = centroids[u]
                idx_nn, dists = knn(tree_known, [Float64(c_u[1]), Float64(c_u[2])], 1)
                nn_unit = known_idx[first(idx_nn)]
                d_val = first(dists)
                w_dist = exp(-d_val / fallback_decay_km)
                h_u[k_sub] = w_dist * hsi_out[nn_unit] + (1.0 - w_dist) * mu0
            end
        end
    end

    # Bound filled values in [0.0, 1.0]
    for (k_sub, u) in enumerate(unknown_idx)
        hsi_out[u] = clamp(h_u[k_sub], 0.0, 1.0)
    end

    return hsi_out
end


"""
    construct_full_movement_domain(
        lon_vec::AbstractVector{<:Real},
        lat_vec::AbstractVector{<:Real};
        radius_km::Real = 15.0,
        land_polygons = nothing,
        depth = nothing,
        depth_threshold::Real = 0.0,
        crs = nothing,
        datum = WGS84Latest
    ) -> NamedTuple

Generates a unified data-driven spatial tessellation covering the complete movement domain
(including areas outside the core survey domain), identifies terrestrial land units, and
severs land edges to form contiguous marine movement channels.

# Mathematical Formulation
A regular planar hexagonal lattice of circumradius ``r`` is generated across the bounding
envelope of all spatial observations:
```math
\\Delta x = \\sqrt{3} r, \\quad \\Delta y = \\frac{3}{2} r
```
Centroids ``\\mathbf{c}_s`` are classified as land via `identify_land_units`. Land-water edges
in graph ``W`` are severed via `apply_land_barrier`, creating a topological state space where
marine passages remain fully connected while landmasses act as impenetrable barriers.

# Arguments
- `lon_vec, lat_vec`: Geographic coordinates encompassing the full observation domain.
- `radius_km`: Hexagon circumradius in km (default 15.0 km).
- `land_polygons`: Optional land boundary polygons (defaults to `:maritimes`).
- `depth`: Optional depth vector matching units.
- `depth_threshold`: Bathymetric cutoff below which units are classified as land (default 0.0).
- `crs`: Coordinate reference system (default local tangent projection).
- `datum`: Reference ellipsoid datum (default `WGS84Latest`).

# Returns
- `NamedTuple`:
  - `centroids_km`, `centroids_lonlat`: Centroid coordinates in km and degrees.
  - `polygons_km`, `polygons_lonlat`: Hexagonal cell boundary vertex rings.
  - `n_units::Int`: Total count of spatial units ``S``.
  - `W`: Cleaned adjacency matrix with land edges severed.
  - `W_raw`: Original geometric adjacency matrix prior to land barrier severing.
  - `land_mask`: Boolean vector of length ``S`` indicating land units.
  - `radius_km`: Hexagon radius in km.
  - `areas_km2`: Cell area per unit.
  - `center_lon`, `center_lat`: Projection origin.
"""
function construct_full_movement_domain(
    lon_vec::AbstractVector{<:Real},
    lat_vec::AbstractVector{<:Real};
    radius_km::Real = 15.0,
    land_polygons = nothing,
    depth = nothing,
    depth_threshold::Real = 0.0,
    crs = nothing,
    datum = WGS84Latest
)::NamedTuple
    lon_min, lon_max = extrema(lon_vec)
    lat_min, lat_max = extrema(lat_vec)

    center_lon = (lon_min + lon_max) / 2.0
    center_lat = (lat_min + lat_max) / 2.0

    # Planar bounding coordinates
    xy_sw = lonlat_to_xy_km(
        lon_min, lat_min;
        center_lon=center_lon, center_lat=center_lat, crs=crs, datum=datum
    )
    xy_ne = lonlat_to_xy_km(
        lon_max, lat_max;
        center_lon=center_lon, center_lat=center_lat, crs=crs, datum=datum
    )

    r = Float64(radius_km)
    dx = sqrt(3.0) * r
    dy = 1.5 * r

    x_min = min(xy_sw[1], xy_ne[1]) - r
    x_max = max(xy_sw[1], xy_ne[1]) + r
    y_min = min(xy_sw[2], xy_ne[2]) - r
    y_max = max(xy_sw[2], xy_ne[2]) + r

    row_min = floor(Int, y_min / dy)
    row_max = ceil(Int, y_max / dy)

    centroids_km = Tuple{Float64, Float64}[]
    for row in row_min:row_max
        yk = row * dy
        xoff = isodd(row) ? (dx / 2.0) : 0.0
        col_min = floor(Int, (x_min - xoff) / dx)
        col_max = ceil(Int, (x_max - xoff) / dx)
        for col in col_min:col_max
            xk = col * dx + xoff
            push!(centroids_km, (xk, yk))
        end
    end

    S = length(centroids_km)
    centroids_lonlat = [
        xy_km_to_lonlat(
            c[1], c[2];
            center_lon=center_lon, center_lat=center_lat, crs=crs, datum=datum
        )
        for c in centroids_km
    ]

    # Flat-top hexagon vertices
    hex_angles = (30.0 .+ 60.0 .* (0:5)) .* (π / 180.0)
    polygons_km     = Vector{Vector{Tuple{Float64, Float64}}}(undef, S)
    polygons_lonlat = Vector{Vector{Tuple{Float64, Float64}}}(undef, S)

    for i in 1:S
        cx, cy = centroids_km[i]
        verts_km = [(cx + r * cos(a), cy + r * sin(a)) for a in hex_angles]
        push!(verts_km, verts_km[1])
        polygons_km[i] = verts_km
        polygons_lonlat[i] = [
            xy_km_to_lonlat(
                v[1], v[2];
                center_lon=center_lon, center_lat=center_lat, crs=crs, datum=datum
            )
            for v in verts_km
        ]
    end

    # Adjacency graph W via KDTree
    c_mat = Matrix{Float64}(undef, 2, S)
    for i in 1:S
        c_mat[1, i] = centroids_km[i][1]
        c_mat[2, i] = centroids_km[i][2]
    end
    tree = KDTree(c_mat)
    adj_thresh = sqrt(3.0) * r * 1.05

    rows_idx = Int[]
    cols_idx = Int[]
    for i in 1:S
        nbrs = inrange(tree, [centroids_km[i][1], centroids_km[i][2]], adj_thresh)
        for j in nbrs
            if j != i
                push!(rows_idx, i)
                push!(cols_idx, j)
            end
        end
    end
    W_init = sparse(rows_idx, cols_idx, ones(Float64, length(rows_idx)), S, S)
    W_init = max.(W_init, W_init')

    # Identify land units with area fraction check
    land_mask = identify_land_units(
        centroids_lonlat;
        polygons=polygons_lonlat,
        land_polygons=land_polygons,
        depth=depth,
        depth_threshold=depth_threshold
    )

    # Sever land edges in W and topological land-crossing links
    W_water, _ = apply_land_barrier(W_init, zeros(Float64, S), land_mask)
    sever_land_crossing_edges!(W_water, centroids_lonlat; land_polygons=land_polygons)

    area_km2 = (3.0 * sqrt(3.0) / 2.0) * r^2

    return (
        centroids_km     = centroids_km,
        centroids_lonlat = centroids_lonlat,
        polygons_km      = polygons_km,
        polygons_lonlat  = polygons_lonlat,
        n_units          = S,
        W                = W_water,
        W_raw            = W_init,
        land_mask        = land_mask,
        radius_km        = r,
        areas_km2        = fill(area_km2, S),
        center_lon       = center_lon,
        center_lat       = center_lat
    )
end


"""
    prepare_movement_data(
        tagging::DataFrame;
        hsi_file::Union{Nothing, AbstractString} = nothing,
        sppoly_file::Union{Nothing, AbstractString} = nothing,
        radius_km::Real = 15.0,
        time_interval::Symbol = :monthly,
        land_polygons = nothing,
        depth = nothing,
        depth_threshold::Real = 0.0,
        crs = nothing,
        datum = WGS84Latest,
        ref_doy::Int = 182,
        verbose::Bool = true
    ) -> NamedTuple

High-level end-to-end data preparation pipeline for individual animal movement and telemetry.
Builds the full-domain tessellation, classifies and blocks terrestrial land barriers, reshards
and autocorrelates HSI into external marine areas, maps telemetry observations, and extracts
mark-recapture transition events with biological group stratifications.

# Arguments
- `tagging::DataFrame`: Raw telemetry records with `:tagid`, `:lon`, `:lat`, `:time`, `:tag`.
- `hsi_file`: Path to environmental Habitat Suitability Index JLD2 file.
- `sppoly_file`: Path to survey spatial polygons JLD2 file defining known HSI units.
- `radius_km`: Spatial hexagon circumradius in km (default 15.0 km).
- `time_interval`: Discrete transition interval (`:monthly`, `:weekly`, `:biweekly`, `:daily`).
- `land_polygons`: Terrestrial boundary polygons (defaults to `:maritimes`).
- `depth`: Optional depth vector matching units.
- `depth_threshold`: Bathymetric cutoff below which units are classified as land (default 0.0).
- `crs`: Target coordinate reference system (default local tangent projection).
- `datum`: Reference ellipsoid datum (default `WGS84Latest`).
- `ref_doy`: Annual reference survey day of year (default 182).
- `verbose`: Toggle progress logging to console.

# Returns
- `NamedTuple`:
  - `tagging`: Processed telemetry DataFrame with `:s_idx` and `:hsi`.
  - `mesh`: Full-domain tessellation NamedTuple.
  - `W`: Adjacency matrix with severed land boundaries.
  - `hsi_vec`: Infilled spatial HSI vector (length ``S``).
  - `monthly_hsi`: Monthly HSI matrix (size ``S \\times T``).
  - `month_lookup`: Mapping of `(year, month)` to column index.
  - `years`: Vector of survey years.
  - `obs`: Extracted mark-recapture event pairs DataFrame.
  - `group_lookup`: Biological stratum dictionary mapping.
  - `land_mask`: Boolean vector of length ``S`` denoting terrestrial barrier units.
"""
function prepare_movement_data(
    tagging::DataFrame;
    hsi_file::Union{Nothing, AbstractString} = nothing,
    sppoly_file::Union{Nothing, AbstractString} = nothing,
    radius_km::Real = 15.0,
    time_interval::Symbol = :daily,
    land_polygons = nothing,
    depth = nothing,
    depth_threshold::Real = 0.0,
    crs = nothing,
    datum = WGS84Latest,
    ref_doy::Int = 182,
    verbose::Bool = true
)::NamedTuple
    tag_df = copy(tagging)

    # Filter dead records
    if hasproperty(tag_df, :is_dead)
        filter!(:is_dead => d -> !coalesce(d, false), tag_df)
    end

    tag_df[!, :lon]   = Float64.(tag_df.lon)
    tag_df[!, :lat]   = Float64.(tag_df.lat)
    tag_df[!, :tag]   = Int.(tag_df.tag)
    tag_df[!, :tagid] = string.(tag_df.tagid)
    tag_df[!, :time]  = Float64.(tag_df.time)

    # Require both release and recapture events
    valid_set = Set{String}()
    for sub in groupby(tag_df, :tagid)
        tags = sub.tag
        if any(==(0), tags) && any(>(0), tags)
            push!(valid_set, sub.tagid[1])
        end
    end
    filter!(:tagid => ∈(valid_set), tag_df)
    sort!(tag_df, [:tagid, :time])

    # 1. Full-domain tessellation with land barriers
    verbose && println("  [prepare] Constructing full movement domain (r=$(radius_km) km) …")
    mesh = construct_full_movement_domain(
        tag_df.lon, tag_df.lat;
        radius_km=radius_km, land_polygons=land_polygons,
        depth=depth, depth_threshold=depth_threshold,
        crs=crs, datum=datum
    )
    verbose && println("    Total mesh units: $(mesh.n_units) (water: $(count(!, mesh.land_mask)), land: $(sum(mesh.land_mask)))")

    # 2. Map telemetry observations to hex units
    verbose && println("  [prepare] Mapping observations to domain units …")
    tag_df = map_telemetry_to_units(
        tag_df, mesh.centroids_km,
        mesh.center_lon, mesh.center_lat; crs=crs, datum=datum
    )

    # 3. Load HSI and perform autocorrelation infilling outside core domain
    hsi_vec      = zeros(Float64, mesh.n_units)
    monthly_hsi  = Matrix{Float64}(undef, 0, 0)
    month_lookup = Dict{Tuple{Int, Int}, Int}()
    years_vec    = Int[]

    if hsi_file !== nothing && isfile(hsi_file)
        verbose && println("  [prepare] Loading and infilling HSI field …")
        h = load_hsi_jld2(hsi_file; ref_doy=ref_doy)
        years_vec    = h.years
        month_lookup = h.month_lookup
        S_src        = size(h.monthly_hsi, 1)

        src_coords_km = Tuple{Float64, Float64}[]
        known_mask = falses(mesh.n_units)

        if sppoly_file !== nothing && isfile(sppoly_file)
            sp = JLD2.load(sppoly_file)
            au_src = sp["au"]
            n_au = length(au_src.lon)
            src_coords_km = Vector{Tuple{Float64, Float64}}(undef, n_au)
            for i in 1:n_au
                src_coords_km[i] = lonlat_to_xy_km(
                    au_src.lon[i], au_src.lat[i];
                    center_lon=mesh.center_lon, center_lat=mesh.center_lat,
                    crs=crs, datum=datum
                )
            end

            # Identify mesh units lying within the known HSI survey domain
            tree_src = KDTree(hcat([[Float64(c[1]), Float64(c[2])] for c in src_coords_km]...))
            survey_radius_thresh = 1.5 * radius_km
            for i in 1:mesh.n_units
                if !mesh.land_mask[i]
                    _, dists = knn(tree_src, [mesh.centroids_km[i][1], mesh.centroids_km[i][2]], 1)
                    if first(dists) <= survey_radius_thresh
                        known_mask[i] = true
                    end
                end
            end
        end

        n_mo = size(h.monthly_hsi, 2)
        if !isempty(src_coords_km) && n_mo > 0
            monthly_hsi = zeros(Float64, mesh.n_units, n_mo)
            for m in 1:n_mo
                raw_m = reshard_hsi_field(
                    h.monthly_hsi[:, m], src_coords_km, mesh.centroids_km
                )
                monthly_hsi[:, m] = infill_spatial_hsi(
                    raw_m, mesh.W, known_mask;
                    land_mask=mesh.land_mask, centroids=mesh.centroids_km
                )
            end
            hsi_vec = vec(mean(monthly_hsi, dims=2))
        else
            base_vec = fill(0.2, mesh.n_units)
            hsi_vec = infill_spatial_hsi(
                base_vec, mesh.W, known_mask;
                land_mask=mesh.land_mask, centroids=mesh.centroids_km
            )
        end

        if !isempty(month_lookup) && size(monthly_hsi, 1) == mesh.n_units
            tag_df[!, :hsi] = match_telemetry_closest_month_hsi(
                tag_df, monthly_hsi, month_lookup, years_vec
            )
        else
            tag_df[!, :hsi] = fill(0.5, nrow(tag_df))
        end
    else
        verbose && println("  [prepare] No HSI file provided; using uniform marine HSI = 0.5 …")
        hsi_vec = [mesh.land_mask[i] ? 0.0 : 0.5 for i in 1:mesh.n_units]
        tag_df[!, :hsi] = fill(0.5, nrow(tag_df))
    end

    # 4. Extract mark-recapture event pairs
    dt_map = (monthly=1.0/12.0, weekly=1.0/52.0, biweekly=1.0/26.0, daily=1.0/365.25, raw=1.0)
    dt = hasproperty(dt_map, time_interval) ? getproperty(dt_map, time_interval) : 1.0 / 12.0

    has_sex = hasproperty(tag_df, :sex)
    has_mat = hasproperty(tag_df, :mat)

    sorted_df = sort(tag_df, [:tagid, :time])
    n_rows = nrow(sorted_df)

    tagids    = sorted_df.tagid
    times     = sorted_df.time
    s_idxs    = sorted_df.s_idx
    sexes_col = has_sex ? sorted_df.sex : nothing
    mats_col  = has_mat ? sorted_df.mat : nothing

    RecordType = NamedTuple{
        (:tagid, :release, :recapture, :k, :sex, :mat), 
        Tuple{String, Int, Int, Int, String, String}
    }
    pair_records = Vector{RecordType}(undef, 0)

    if n_rows >= 2
        sizehint!(pair_records, n_rows)
        for i in 2:n_rows
            if tagids[i] == tagids[i-1]
                Δt = times[i] - times[i-1]
                k  = max(1, round(Int, Δt / dt))
                push!(pair_records, (
                    tagid     = string(tagids[i-1]),
                    release   = s_idxs[i-1],
                    recapture = s_idxs[i],
                    k         = k,
                    sex       = has_sex ? string(sexes_col[i-1]) : "unknown",
                    mat       = has_mat ? string(mats_col[i-1])  : "unknown"
                ))
            end
        end
    end

    obs = DataFrame(pair_records)

    # 5. Assign biological groupings
    n_obs = nrow(obs)
    labels = Vector{String}(undef, n_obs)
    if n_obs > 0
        obs_sexes = obs[!, :sex]
        obs_mats  = obs[!, :mat]
        @inbounds for i in 1:n_obs
            sx = string(obs_sexes[i])
            mt = string(obs_mats[i])
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
    end

    unique_labels = sort!(unique(labels))
    group_lookup  = Dict{String, Int}(lbl => i for (i, lbl) in enumerate(unique_labels))
    group_ids = Vector{Int}(undef, n_obs)
    @inbounds for i in 1:n_obs
        group_ids[i] = group_lookup[labels[i]]
    end
    obs[!, :group] = group_ids

    return (
        tagging      = tag_df,
        mesh         = mesh,
        W            = mesh.W,
        hsi_vec      = hsi_vec,
        monthly_hsi  = monthly_hsi,
        month_lookup = month_lookup,
        years        = years_vec,
        obs          = obs,
        group_lookup = group_lookup,
        land_mask    = mesh.land_mask
    )
end
