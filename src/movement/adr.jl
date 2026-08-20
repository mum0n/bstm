# ##############################################################################
# # Section 1: Utility and Placeholder Functions
# ##############################################################################

# Provides a concrete implementation for the plot function using the Plots.jl library.
function plot(args...; kwargs...)
    return Plots.plot(args...; kwargs...)
end

# Provides a concrete implementation for the plot! function to modify existing plots.
function plot!(args...; kwargs...)
    return Plots.plot!(args...; kwargs...)
end

# Provides a concrete implementation for the heatmap function.
function heatmap(args...; kwargs...)
    return Plots.heatmap(args...; kwargs...)
end

# Provides a concrete implementation for the quiver function to plot vector fields.
function quiver(args...; kwargs...)
    return Plots.quiver(args...; kwargs...)
end

# Provides a concrete implementation for the histogram function.
function histogram(args...; kwargs...)
    return Plots.histogram(args...; kwargs...)
end

# Provides a concrete implementation for the vline! function to add vertical lines to plots.
function vline!(args...; kwargs...)
    return Plots.vline!(args...; kwargs...)
end


# Creates a choropleth map by plotting a series of polygons,
# where each polygon's fill color is determined by a corresponding value.
# This replaces the placeholder and enables actual visualization of spatial data fields.
function plot_choropleth(vals, polys; title="", cmap=:viridis)
    # Create a color gradient from the specified colormap.
    c_gradient = cgrad(cmap)
    
    # Normalize the values to the range [0, 1] to map them to colors.
    v_min, v_max = extrema(vals)
    # Add a small epsilon to the denominator to avoid division by zero if all values are the same.
    normalized_vals = (vals .- v_min) ./ (v_max - v_min + 1e-9)

    # Initialize a new plot with the given title and an equal aspect ratio.
    p = plot(title=title, aspect_ratio=:equal, legend=false)

    # Iterate through each polygon and its corresponding normalized value.
    for (i, poly) in enumerate(polys)
        if length(poly) > 2
            # Extract x and y coordinates from the polygon's points.
            px = [pt[1] for pt in poly]
            py = [pt[2] for pt in poly]
            
            # Determine the fill color from the normalized value and color gradient.
            fill_color = c_gradient[normalized_vals[i]]
            
            # Plot the polygon with the specified fill color and a subtle border.
            plot!(p, px, py, seriestype=:shape, fillcolor=fill_color, linecolor=:black, linewidth=0.5, label=nothing)
        end
    end
    
    return p
end

# Visualizes the location of a single individual at a specific time step
# by highlighting the corresponding spatial unit on a map. This replaces the
# placeholder and provides a concrete tool for tracking simulated or inferred movement.
function map_individual_occupancy(tracts, id, step, au_context)
    # Initialize a new plot with a title and equal aspect ratio.
    p = plot(title="Occupancy of Mark $id at Step $step", aspect_ratio=:equal, legend=false)

    # Draw all spatial units in the background for context.
    for poly in au_context.polygons
        if length(poly) > 2
            px = [pt[1] for pt in poly]
            py = [pt[2] for pt in poly]
            plot!(p, px, py, seriestype=:shape, fillcolor=:lightgray, fillalpha=0.3, linecolor=:white, linewidth=0.5, label=nothing)
        end
    end

    # Get the spatial unit index for the specified individual and time step.
    unit_idx = tracts[id, step]

    # Highlight the occupied polygon.
    if 1 <= unit_idx <= length(au_context.polygons)
        occupied_poly = au_context.polygons[unit_idx]
        if length(occupied_poly) > 2
            px_occ = [pt[1] for pt in occupied_poly]
            py_occ = [pt[2] for pt in occupied_poly]
            
            # Plot the highlighted polygon with a distinct color.
            plot!(p, px_occ, py_occ, seriestype=:shape, fillcolor=:red, fillalpha=0.8, linecolor=:black, linewidth=1.0, label="Occupied Unit")
        end
    else
        @warn "Invalid unit index $unit_idx for individual $id at step $step."
    end
    
    return p
end

# Rationale: The original script assumes the existence of `assign_spatial_units`.
# This placeholder provides the necessary data structure (a regular grid) for the
# script to run, in place of a more complex Centroidal Voronoi Tesselation.
function assign_spatial_units(s_x, s_y; area_method=:cvt, target_units=100, buffer_dist=0.0)
    n_pts = target_units
    grid_dim = Int(floor(sqrt(n_pts)))
    n_spatial = grid_dim * grid_dim
    
    x_min, x_max = extrema(s_x)
    y_min, y_max = extrema(s_y)
    
    xs = range(x_min, stop=x_max, length=grid_dim)
    ys = range(y_min, stop=y_max, length=grid_dim)
    
    centroids = [(x, y) for y in ys for x in xs]
    
    dx = (x_max - x_min) / (grid_dim - 1)
    dy = (y_max - y_min) / (grid_dim - 1)
    polygons = []
    for (cx, cy) in centroids
        push!(polygons, [
            (cx - dx/2, cy - dy/2), (cx + dx/2, cy - dy/2),
            (cx + dx/2, cy + dy/2), (cx - dx/2, cy + dy/2),
            (cx - dx/2, cy - dy/2)
        ])
    end

    W = spzeros(Int, n_spatial, n_spatial)
    for r in 1:grid_dim, c in 1:grid_dim
        i = (r-1)*grid_dim + c
        if r > 1; W[i, i-grid_dim] = 1; end # Up
        if r < grid_dim; W[i, i+grid_dim] = 1; end # Down
        if c > 1; W[i, i-1] = 1; end # Left
        if c < grid_dim; W[i, i+1] = 1; end # Right
    end

    return (centroids=centroids, polygons=polygons, W=W)
end

# The graph Laplacian is a standard operator in spatial statistics,
# representing diffusion on a graph. This function computes it from an adjacency matrix W.
function laplacian_matrix(W::AbstractMatrix)
    D = spdiagm(0 => vec(sum(W, dims=2)))
    return D - W
end


# ##############################################################################
# # Section 2: Data Simulation
# ##############################################################################

# Simulates a density vector correlated with a habitat probability vector.
# This is a core component for generating realistic ecological data.
function simulate_correlated_density_vector(habitat_prob, rho_target, log_mu, sigma_resid)
    n = length(habitat_prob)
    p_std = (habitat_prob .- mean(habitat_prob)) ./ (std(habitat_prob) + 1e-9)
    epsilon = randn(n)
    log_n_signal = (rho_target .* p_std) .+ (sqrt(1.0 - rho_target^2) .* epsilon)
    log_n = log_mu .+ (log_n_signal .* sigma_resid)
    return exp.(log_n)
end

# Generates a complete simulation bundle for the ADR model, including
# spatiotemporal density, habitat suitability, and mark-recapture data.
#  produce telemetry data in a long format DataFrame.
function generate_ADR_simulation_bundle(domain_size, n_units, n_years, n_marks)
    Random.seed!(42)
    s_x_init = rand(Uniform(0, domain_size), 1000)
    s_y_init = rand(Uniform(0, domain_size), 1000)
    au = assign_spatial_units(s_x_init, s_y_init, area_method=:cvt, target_units=n_units, buffer_dist=0.0)
    
    centroids = au.centroids
    n_spatial = length(centroids)
    cent_x = [c[1] for c in centroids]
    cent_y = [c[2] for c in centroids]

    habitat_p = zeros(n_spatial, n_years)
    p_init = [exp(-sqrt((cent_x[i]-domain_size/2)^2 + (cent_y[i]-domain_size/2)^2)/(domain_size/3)) for i in 1:n_spatial]
    habitat_p[:, 1] = p_init ./ (maximum(p_init) + 1e-9)
    for t in 2:n_years
        noise = randn(n_spatial) .* 0.05
        habitat_p[:, t] = min.(1.0, max.(0.01, habitat_p[:, t-1] .+ noise))
    end

    density_n = zeros(n_spatial, n_years)
    for t in 1:n_years
        density_n[:, t] = simulate_correlated_density_vector(habitat_p[:, t], 0.75, 3.5, 0.6)
    end

    # Create telemetry data in a long format
    telemetry_df = DataFrame(tagid=Int[], s_idx=Int[], time=Float64[], tag=Int[], individual_covariate=Float64[])
    for i in 1:n_marks
        release_unit = rand(1:n_spatial)
        year_start = rand(1:(n_years-2))
        time_steps = rand(1:2)
        
        # Simulate recapture location
        d_travel = rand(Uniform(domain_size/20, domain_size/2.5)) * time_steps
        dists = [abs(sqrt((cent_x[release_unit]-cent_x[j])^2 + (cent_y[release_unit]-cent_y[j])^2) - d_travel) for j in 1:n_spatial]
        recapture_unit = argmin(dists)
        
        ind_cov = randn()

        # Add release event
        push!(telemetry_df, (tagid=i, s_idx=release_unit, time=Float64(year_start), tag=0, individual_covariate=ind_cov))
        # Add recapture event
        push!(telemetry_df, (tagid=i, s_idx=recapture_unit, time=Float64(year_start + time_steps), tag=1, individual_covariate=ind_cov))
    end

    df = DataFrame()
    for t in 1:n_years
        temp_df = DataFrame(
            density=density_n[:, t],
            unit_id=1:n_spatial, 
            year=t, 
            s_x=cent_x, 
            s_y=cent_y, 
            habitat_p=habitat_p[:, t]
        )
        append!(df, temp_df)
    end

    return (data=df, telemetry_data=telemetry_df, au=au, n_spatial=n_spatial, n_years=n_years)
end

# Computes a velocity field from a habitat probability grid, representing
# the direction of advective movement.
function compute_velocity_field(prob_vec, grid_dim, strength, mode=:exponential)
    grid = reshape(prob_vec, grid_dim, grid_dim)
    rows, cols = size(grid)
    vx = zeros(Float64, rows, cols)
    vy = zeros(Float64, rows, cols)
    eps = 1e-6

    for r in 1:rows, c in 1:cols
        gx = (c == 1) ? (grid[r, 2]-grid[r, 1]) : (c == cols ? grid[r, cols]-grid[r, cols-1] : (grid[r, c+1]-grid[r, c-1])/2.0)
        gy = (r == 1) ? (grid[2, c]-grid[1, c]) : (r == rows ? grid[rows, c]-grid[rows-1, c] : (grid[r+1, c]-grid[r-1, c])/2.0)

        if mode == :exponential
            denom = max(grid[r, c], eps)
            vx[r, c] = (gx / denom) * strength
            vy[r, c] = (gy / denom) * strength
        else
            vx[r, c] = gx * strength
            vy[r, c] = gy * strength
        end
    end
    return (vx = vec(vx), vy = vec(vy))
end

# ##############################################################################
# # Section 4: Post-Hoc Simulation and Analysis
# ##############################################################################

# Calculates the k-step transition matrix by matrix exponentiation.
# This is useful for predicting distributions after multiple time steps.
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

# Simulates individual movement paths from a static transition matrix,
# with an option for directional persistence (autocorrelated movement).
function simulate_posterior_trajectories(
    Gamma_base::AbstractMatrix{T}, 
    start_units::Vector{Int}, 
    n_steps::Int, 
    au_context::NamedTuple; 
    rho_persistence::Real = 0.0
) where T <: Real
    
    n_indiv = length(start_units)
    n_spatial = size(Gamma_base, 1)
    centroids = au_context.centroids
    paths = zeros(Int, n_indiv, n_steps + 1)
    paths[:, 1] = start_units
    
    G_sampling = copy(Gamma_base)
    for i in 1:n_spatial
        rs = sum(G_sampling[i, :])
        if rs > 0; G_sampling[i, :] ./= rs; else; G_sampling[i, i] = 1.0; end
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
                        if j == curr_node; continue; end
                        v_cand = [centroids[j][d] - centroids[curr_node][d] for d in 1:2]
                        n_cand = norm(v_cand)
                        if n_cand > 1e-9
                            cos_theta = dot(v_prev, v_cand) / (norm(v_prev) * n_cand)
                            persistence_weights[j] = exp(rho_persistence * cos_theta)
                        end
                    end
                    p_row = p_row .* persistence_weights
                    row_sum = sum(p_row)
                    if row_sum > 0; p_row ./= row_sum; else; p_row = vec(G_sampling[curr_node, :]); end
                end
            end
            next_node = rand(Categorical(p_row))
            paths[i, t] = next_node
            prev_node = curr_node
            curr_node = next_node
        end
    end
    return paths
end

# Simulates individual movement paths through a dynamic environment where
# the transition matrix changes at each time step.
function simulate_mechanistic_trajectories(
    Gamma_sequence::Vector{<:AbstractMatrix{T}},
    start_units::Vector{Int},
    t_start::Int,
    au_context::NamedTuple;
    rho_persistence::Real = 0.0,
    n_years_sim::Int = 1
) where T <: Real

    n_indiv = length(start_units)
    n_spatial = size(Gamma_sequence[1], 1)
    centroids = au_context.centroids
    max_available_time = length(Gamma_sequence)
    
    # Ensure simulation does not exceed available time steps
    actual_steps = n_years_sim
    if t_start + n_years_sim > max_available_time
        actual_steps = max_available_time - t_start
        println("Warning: n_years_sim reduced to $actual_steps to match available Gamma sequence.")
    end

    paths = zeros(Int, n_indiv, actual_steps + 1)
    paths[:, 1] = start_units
    
    G_stochastic = [copy(G) for G in Gamma_sequence]
    for t in 1:length(G_stochastic)
        for i in 1:n_spatial
            rs = sum(G_stochastic[t][i, :])
            if rs > 0; G_stochastic[t][i, :] ./= rs; else; G_stochastic[t][i, i] = 1.0; end
        end
    end

    for i in 1:n_indiv
        curr_node = start_units[i]
        prev_node = 0
        for step in 1:actual_steps
            current_t = t_start + step - 1
            p_row = vec(G_stochastic[current_t][curr_node, :])
            if rho_persistence > 0.0 && prev_node != 0
                v_prev = [centroids[curr_node][d] - centroids[prev_node][d] for d in 1:2]
                if norm(v_prev) > 1e-9
                    persistence_weights = ones(T, n_spatial)
                    for j in 1:n_spatial
                        if j == curr_node; continue; end
                        v_cand = [centroids[j][d] - centroids[curr_node][d] for d in 1:2]
                        n_cand = norm(v_cand)
                        if n_cand > 1e-9
                            cos_theta = dot(v_prev, v_cand) / (norm(v_prev) * n_cand)
                            persistence_weights[j] = exp(rho_persistence * cos_theta)
                        end
                    end
                    p_row = p_row .* persistence_weights
                    row_sum = sum(p_row)
                    if row_sum > 0; p_row ./= row_sum; else; p_row = vec(G_stochastic[current_t][curr_node, :]); end
                end
            end
            next_node = rand(Categorical(p_row))
            paths[i, step + 1] = next_node
            prev_node = curr_node
            curr_node = next_node
        end
    end
    return paths
end

# Generates a movement transition matrix based on a habitat suitability
# field, representing an alternative to the physics-based ADR propagator.
function compute_suitability_transition_kernel(
    suitability_vec::AbstractVector{T},
    W::AbstractMatrix,
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
            if i == j; continue; end
            suitability_bias = exp(sensitivity * suitability_vec[j])
            edge_weight = vals[j_idx]
            Gamma[i, j] = (suitability_bias + diffusion_weight) * edge_weight
        end
    end
    
    for i in 1:n_spatial
        row_sum = sum(Gamma[i, :])
        if row_sum > 1e-12; Gamma[i, :] ./= row_sum; else; Gamma[i, i] = 1.0; end
    end
    return Gamma
end

# A helper function for plotting simulated trajectories. It is refactored
# to accept `au_context` as an argument, removing reliance on global state.
function render_paths!(p_obj, trajects, title_str, au_context)
    for poly in au_context.polygons
        if length(poly) > 2
            plot!(p_obj, [pt[1] for pt in poly], [pt[2] for pt in poly], fillalpha=0.02, color=:black, lw=0.1, label=nothing)
        end
    end
    for i in 1:size(trajects, 1)
        idx_list = trajects[i, :]
        px = [au_context.centroids[id][1] for id in idx_list]
        py = [au_context.centroids[id][2] for id in idx_list]
        plot!(p_obj, px, py, marker=:circle, markersize=1.5, lw=1.2, label="Mark $i", title=title_str)
    end
end

# Aggregates unit-level transition probabilities into a regional
# connectivity matrix, providing a coarse-grained view of movement.
function calculate_regional_connectivity(Gamma, strata_definition)
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

# Creates a histogram of the Advection-to-Diffusion ratio, a diagnostic
# for identifying where movement is dominated by directed vs. random processes.
function plot_ad_ratio_distribution(advection_field, diffusion_field)
    ratios = advection_field ./ (mean(diffusion_field) .+ 1e-6)
    plt = histogram(ratios, bins=30, title="Advection-Diffusion Ratio (A/D)",
        xlabel="Ratio (Magnitude/Coefficient)", ylabel="Frequency",
        label="Spatial Units", color=:plum, linecolor=:white)
    vline!(plt, [1.0], color=:red, linestyle=:dash, label="Critical Threshold")
    return plt
end

# ##############################################################################
# # Section 5: Main Analysis and Visualization Workflow
# ##############################################################################

# This function replaces the original `predict_and_plot_movement_dynamics`,
# which was dependent on an external `bstm` library. This version is self-contained,
# working directly with the Turing chain and simulated data to produce key
# diagnostic plots and analyses shown in the markdown file.
function synthesize_adr_results(chain, sim_data, vel_vectors)
    println("\n--- Synthesizing ADR Model Results ---")
    
    # Extract posterior means for key parameters
    S_strength_mean = mean(chain[:velocity_movement])
    D_coeff_mean = mean(chain[:diffusion_movement])
    
    # Reconstruct the mean propagator and transition matrices
    L = laplacian_matrix(sim_data.au.W)
    # The advection operator A defines the direction of flow on the graph.
    # Here we construct a simple version based on graph topology.
    W_dir = tril(sim_data.au.W, -1)
    out_degree = sum(W_dir, dims=2)[:]
    D_inv = spdiagm(0 => 1.0 ./ (out_degree .+ 1e-9))
    A = D_inv * W_dir
    
    M_prop_mean = I(sim_data.n_spatial) - (S_strength_mean * A) - (D_coeff_mean * L)
    Gamma_mean = inv(Matrix(M_prop_mean))

    # --- Visualization 1: Regional Connectivity ---
    println("Generating Regional Connectivity Matrix...")
    cents_x = [c[1] for c in sim_data.au.centroids]
    strata = [x > 500.0 ? 2 : 1 for x in cents_x] # East/West split
    conn_mat = calculate_regional_connectivity(Gamma_mean, strata)
    plt_conn = heatmap(["West", "East"], ["West", "East"], conn_mat,
        title = "Regional Transfer Probability Matrix",
        xlabel = "To Region", ylabel = "From Region", color = :viridis, clims = (0, 1))
    display(plt_conn)

    # --- Visualization 2: Advection-Diffusion Ratio ---
    println("Generating A/D Ratio Histogram...")
    advection_magnitude = sqrt.(vel_vectors.vx.^2 .+ vel_vectors.vy.^2) .* S_strength_mean
    diffusion_magnitude = fill(D_coeff_mean, sim_data.n_spatial)
    plt_ad = plot_ad_ratio_distribution(advection_magnitude, diffusion_magnitude)
    display(plt_ad)

    # --- Visualization 3: Population Flow Vector Field ---
    println("Generating Population Flow Vector Field...")
    cents_y = [c[2] for c in sim_data.au.centroids]
    v_scale = 5.0
    plt_vectors = quiver(cents_x, cents_y,
        quiver = (advection_magnitude .* v_scale, zeros(length(cents_x))),
        title = "Recovered Population Flow Vectors", aspect_ratio = :equal,
        color = :red, xlabel = "X (km)", ylabel = "Y (km)")
    display(plt_vectors)

    # --- Visualization 4: Path Simulation with Persistence ---
    println("Simulating Paths with and without Persistence...")
    random_starts = rand(1:sim_data.n_spatial, 3)
    n_sim_steps = 20
    paths_standard = simulate_posterior_trajectories(Gamma_mean, random_starts, n_sim_steps, sim_data.au, rho_persistence=0.0)
    paths_persistent = simulate_posterior_trajectories(Gamma_mean, random_starts, n_sim_steps, sim_data.au, rho_persistence=2.0)
    
    plt_comp = plot(layout=(1, 2), size=(1000, 500), aspect_ratio=:equal)
    render_paths!(plt_comp[1], paths_standard, "Standard Markovian Path", sim_data.au)
    render_paths!(plt_comp[2], paths_persistent, "Persistent Path (rho=2.0)", sim_data.au)
    display(plt_comp)

    # --- Visualization 5: Multi-Year Dynamic Projection ---
    println("Initiating Multi-Year Mechanistic Path Projection...")
    gamma_seq = [Gamma_mean for _ in 1:sim_data.n_years]
    high_density_units = sortperm(sim_data.data.density[sim_data.data.year .== 1], rev=true)[1:10]
    starts = rand(high_density_units, 5)
    dynamic_paths = simulate_mechanistic_trajectories(gamma_seq, starts, 1, sim_data.au, rho_persistence=1.5, n_years_sim=4)
    
    plt_dyn = plot(aspect_ratio=:equal, title="Mechanistic 5-Year Path Projections", legend=:outerright)
    render_paths!(plt_dyn, dynamic_paths, "5-Year Paths", sim_data.au)
    display(plt_dyn)
    
    # --- Visualization 6: Raster-Driven Trajectories ---
    println("Generating Multi-Year Transition Sequence from Dynamic Suitability...")
    dynamic_gamma_seq = Vector{SparseMatrixCSC{Float64, Int}}(undef, sim_data.n_years)
    for t in 1:sim_data.n_years
        s_vec = sim_data.data.habitat_p[sim_data.data.year .== t]
        dynamic_gamma_seq[t] = compute_suitability_transition_kernel(s_vec, sim_data.au.W, 1.5, 0.05)
    end
    
    starts_raster = rand(1:sim_data.n_spatial, 5)
    raster_driven_paths = simulate_mechanistic_trajectories(dynamic_gamma_seq, starts_raster, 1, sim_data.au, rho_persistence=1.2, n_years_sim = sim_data.n_years - 1)
    
    plt_raster = plot(aspect_ratio=:equal, title="Raster-Driven Mechanistic Trajectories", legend=:outerright)
    
    # Background rendering would go here if not a dummy plot

    render_paths!(plt_raster, raster_driven_paths, "Raster-Driven Paths", sim_data.au)
    display(plt_raster)

    println("--- Synthesis Complete ---")
end


# Main execution block to run the entire ADR simulation and analysis workflow.
# This makes the script self-contained and demonstrates the intended use of the functions.
function main()
    # --- 1. Setup and Data Simulation ---
    println("--- Generating Simulation Data ---")
    sim_data = generate_ADR_simulation_bundle(1000.0, 100, 5, 100)
    println("Total density records: ", nrow(sim_data.data))

    # --- 2. Pre-computation for Model ---
    println("--- Computing Velocity and Operator Templates ---")
    avg_p = [mean(sim_data.data.habitat_p[sim_data.data.unit_id .== i]) for i in 1:sim_data.n_spatial]
    g_side = Int(sqrt(sim_data.n_spatial))
    vel_vectors = compute_velocity_field(avg_p, g_side, 1.5, :exponential)
    
    # --- 3. Model Definition and Sampling using bstm ---
    println("--- Defining and Sampling Joint ADR-Telemetry Model ---")
    
    # The ADR process and telemetry likelihood are handled by the `movement` component.
    model_instance = @bstm(
        likelihood(density, family=:poisson) ~
            intercept(prior=Normal(0, 5)) +
            movement(unit_id, year,
                velocity = Normal(1.0, 0.5),
                diffusion = LogNormal(-1.0, 1.0),
                sigma = Exponential(1.0),
                beta_het = Normal(0.0, 0.5),
                mark_recapture_data = sim_data.telemetry_data
            ),
        sim_data.data,
        W = sim_data.au.W
    )
    
    println("Starting posterior sampling...")
    chain_results = sample(model_instance, NUTS(0.65), 500; progress=true)

    println("\n--- Recovered Mechanistic Parameters ---")
    display(chain_results)

    # --- 4. Post-Hoc Analysis and Visualization ---
    synthesize_adr_results(chain_results, sim_data, vel_vectors)
end
 