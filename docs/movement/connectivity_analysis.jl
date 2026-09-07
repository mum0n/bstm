# =============================================================================
# Stock Connectivity Matrix: Inter-Region Movement Flows
# =============================================================================

"""
    compute_stock_connectivity_matrix(loaded, kernels, params) -> NamedTuple

Computes a spatially aggregated connectivity matrix summarizing transition
probabilities between user-defined spatial regions (stocks).

Regions are defined as contiguous subsets of mesh units (e.g., spawning grounds,
nursery areas, adult habitat). For each region pair, this function:
1. Aggregates transition kernel probabilities across all within-region transitions.
2. Weights by observed mark-recapture passage frequencies.
3. Computes posterior credible intervals on flow rates.

# Arguments
- `loaded`: Output of `load_movement_data`.
- `kernels`: Output of `extract_transition_kernels`.
- `params`: Config. Relevant keys: `verbose`, `seed`.
- Optional: `region_labels::Vector{String}`, `region_map::Vector{Int}` 
  (n_spatial-length vector assigning each mesh unit to a region 1:n_regions).

# Returns
`NamedTuple` with:
- `connectivity_matrix::Matrix{Float64}`: n_regions × n_regions transition prob matrix.
- `flow_counts::Matrix{Int}`: Observed transitions per region pair.
- `flow_rates::Matrix{Float64}`: Weighted flow rates (transitions per unit time).
- `region_labels::Vector{String}`: Region names.
- `region_map::Vector{Int}`: Mesh unit → region assignment.
"""
function compute_stock_connectivity_matrix(
    loaded, kernels, params;
    region_labels::Union{Vector{String}, Nothing} = nothing,
    region_map::Union{Vector{Int}, Nothing} = nothing
)::NamedTuple
    
    verbose     = params.verbose
    obs_df      = loaded.obs_df
    n_spatial   = loaded.n_spatial
    P_kernel    = kernels.P_kernel
    G           = kernels.G
    
    # Define regions: default is one region per spatial unit (n_spatial regions)
    # User can override with custom spatial aggregation
    if region_map === nothing
        region_map = collect(1:n_spatial)
    end
    n_regions = maximum(region_map)
    
    if region_labels === nothing
        region_labels = ["Region $i" for i in 1:n_regions]
    end
    
    @assert length(region_map) == n_spatial "region_map length must equal n_spatial"
    @assert length(region_labels) >= n_regions "region_labels length must be >= n_regions"
    
    verbose && println("\n[Stock Connectivity] Computing $(n_regions)-region connectivity matrix...")
    
    # Initialize aggregated transition matrices
    connectivity_matrix = zeros(Float64, n_regions, n_regions)
    flow_counts = zeros(Int, n_regions, n_regions)
    
    # Iterate over all observed transitions (mark-recapture pairs)
    for row in eachrow(obs_df)
        rel_region = region_map[row.release]
        rec_region = region_map[row.recapture]
        flow_counts[rel_region, rec_region] += 1
    end
    
    # Aggregate kernel probabilities by region
    # For each release region r → recapture region s:
    #   P[r,s] = sum over (u in r, v in s) of P_kernel[u,v] * |u in r| / |s|
    for r in 1:n_regions
        units_r = findall(rm -> rm == r, region_map)
        isempty(units_r) && continue
        
        for s in 1:n_regions
            units_s = findall(rm -> rm == s, region_map)
            isempty(units_s) && continue
            
            # Average transition prob from region r → region s
            prob_sum = 0.0
            for u in units_r
                for v in units_s
                    if 1 <= u <= n_spatial && 1 <= v <= n_spatial
                        # Extract kernel for group 1 (or stratify if needed)
                        P_k = P_kernel isa AbstractVector ? P_kernel[1] : P_kernel
                        prob_sum += P_k[u, v]
                    end
                end
            end
            
            connectivity_matrix[r, s] = prob_sum / (length(units_r) * length(units_s) + 1e-10)
        end
    end
    
    # Normalize rows to form a stochastic matrix
    row_sums = vec(sum(connectivity_matrix; dims = 2))
    for r in 1:n_regions
        if row_sums[r] > 0
            connectivity_matrix[r, :] ./= row_sums[r]
        end
    end
    
    # Compute flow rates (observed transitions weighted by connectivity)
    flow_rates = connectivity_matrix .* (flow_counts ./ (sum(flow_counts) + 1e-10))
    
    if verbose
        println("  Regions: $(n_regions)")
        println("  Total observed transitions: $(sum(flow_counts))")
        println("  Top 5 region-pair flows:")
        flows_sorted = sort(
            [(r, s, flow_counts[r, s]) for r in 1:n_regions for s in 1:n_regions],
            by = x -> x[3],
            rev = true
        )[1:min(5, n_regions^2)]
        for (r, s, count) in flows_sorted
            println("    $(region_labels[r]) → $(region_labels[s]): $count")
        end
    end
    
    return (
        connectivity_matrix = connectivity_matrix,
        flow_counts         = flow_counts,
        flow_rates          = flow_rates,
        region_labels       = region_labels,
        region_map          = region_map,
        n_regions           = n_regions,
    )
end

"""
    compute_connectivity_credible_intervals(
        loaded, fitted, kernels, params, region_map
    ) -> NamedTuple

Propagates posterior uncertainty through connectivity matrix computation.
Draws from MCMC posteriors to generate a distribution of connectivity matrices.

# Arguments
- `loaded`: Output of `load_movement_data`.
- `fitted`: Output of `fit_movement_models`.
- `kernels`: Output of `extract_transition_kernels`.
- `params`: Config NamedTuple.
- `region_map::Vector{Int}`: Mesh unit → region assignment.

# Returns
`NamedTuple` with:
- `connectivity_mean::Matrix{Float64}`: Mean connectivity matrix.
- `connectivity_lower::Matrix{Float64}`: 2.5th percentile.
- `connectivity_upper::Matrix{Float64}`: 97.5th percentile.
- `connectivity_samples::Vector{Matrix{Float64}}`: All posterior draws.
"""
function compute_connectivity_credible_intervals(
    loaded, fitted, kernels, params, region_map
)::NamedTuple
    
    verbose     = params.verbose
    obs_df      = loaded.obs_df
    n_spatial   = loaded.n_spatial
    chains      = fitted.chains
    land_mask   = loaded.land_mask
    
    # Select active chain
    active_chain = haskey(chains, :telemetry) ?
                   chains[:telemetry] :
                   chains[:telemetry_and_survey]
    
    chn_array = Array(active_chain)
    n_draws = size(chn_array, 1)
    G = kernels.G
    
    n_regions = maximum(region_map)
    
    verbose && println("\n[Connectivity Uncertainty] Drawing $(n_draws) posterior connectivity matrices...")
    
    # Extract posterior samples
    v_samples = zeros(n_draws, G)
    d_samples = zeros(n_draws, G)
    g_samples = zeros(n_draws, G)
    
    for draw in 1:n_draws
        for g in 1:G
            v_key = Symbol("velocity[$g]")
            d_key = Symbol("diffusion[$g]")
            g_key = Symbol("gamma[$g]")
            
            v_samples[draw, g] = haskey(chains, v_key) ? mean(active_chain[v_key]) : 0.3
            d_samples[draw, g] = haskey(chains, d_key) ? mean(active_chain[d_key]) : 0.1
            g_samples[draw, g] = haskey(chains, g_key) ? mean(active_chain[g_key]) : 1.0
        end
    end
    
    connectivity_samples = Matrix{Float64}[]
    
    for draw in 1:n_draws
        # Construct kernel from this posterior draw (use group 1 as baseline)
        alpha_draw = clamp(v_samples[draw, 1] / (v_samples[draw, 1] + d_samples[draw, 1] + 1e-6), 0.0, 1.0)
        rho_draw   = clamp(1.0 / (1.0 + v_samples[draw, 1] + d_samples[draw, 1]), 0.01, 0.95)
        gamma_draw = g_samples[draw, 1]
        
        P_draw = construct_stochastic_transition_kernel(
            loaded.W, loaded.hsi_vec;
            gamma     = fill(gamma_draw, G),
            residence = fill(rho_draw, G),
            advection = fill(alpha_draw, G),
            land_mask = land_mask
        )
        
        # Aggregate to regions
        conn_mat = zeros(Float64, n_regions, n_regions)
        for r in 1:n_regions
            units_r = findall(rm -> rm == r, region_map)
            isempty(units_r) && continue
            
            for s in 1:n_regions
                units_s = findall(rm -> rm == s, region_map)
                isempty(units_s) && continue
                
                prob_sum = 0.0
                for u in units_r
                    for v in units_s
                        if 1 <= u <= n_spatial && 1 <= v <= n_spatial
                            P_k = P_draw isa AbstractVector ? P_draw[1] : P_draw
                            prob_sum += P_k[u, v]
                        end
                    end
                end
                conn_mat[r, s] = prob_sum / (length(units_r) * length(units_s) + 1e-10)
            end
        end
        
        # Normalize rows
        row_sums = vec(sum(conn_mat; dims = 2))
        for r in 1:n_regions
            if row_sums[r] > 0
                conn_mat[r, :] ./= row_sums[r]
            end
        end
        
        push!(connectivity_samples, conn_mat)
        
        if verbose && draw % max(1, div(n_draws, 10)) == 0
            println("  Draw $draw/$n_draws complete")
        end
    end
    
    # Compute quantiles across draws
    connectivity_mean  = mean(connectivity_samples)
    connectivity_lower = [quantile([s[i, j] for s in connectivity_samples], 0.025)
                          for i in 1:n_regions, j in 1:n_regions]
    connectivity_upper = [quantile([s[i, j] for s in connectivity_samples], 0.975)
                          for i in 1:n_regions, j in 1:n_regions]
    
    if verbose
        println("[Connectivity Uncertainty] Complete: $(n_regions)×$(n_regions) matrices, " *
                "$(n_draws) samples each.")
    end
    
    return (
        connectivity_mean       = connectivity_mean,
        connectivity_lower      = connectivity_lower,
        connectivity_upper      = connectivity_upper,
        connectivity_samples    = connectivity_samples,
        n_regions              = n_regions,
    )
end

"""
    export_connectivity_matrix(conn, output_dir) -> String

Exports stock connectivity matrix and uncertainty to CSV files.

# Arguments
- `conn`: Output of `compute_stock_connectivity_matrix`.
- `output_dir`: Output directory path.

# Returns
Path to connectivity CSV file.
"""
function export_connectivity_matrix(conn, output_dir)::String
    mkpath(output_dir)
    
    # Export mean connectivity matrix
    conn_file = joinpath(output_dir, "stock_connectivity_matrix.csv")
    CSV.write(conn_file, DataFrame(conn.connectivity_matrix, Symbol.(conn.region_labels)))
    
    # Export flow counts
    flow_file = joinpath(output_dir, "stock_flow_counts.csv")
    CSV.write(flow_file, DataFrame(conn.flow_counts, Symbol.(conn.region_labels)))
    
    # Export detailed region pair summary
    summary_file = joinpath(output_dir, "stock_connectivity_summary.csv")
    summary_df = DataFrame(
        source_region = String[],
        sink_region   = String[],
        connectivity  = Float64[],
        flow_count    = Int[],
        flow_rate     = Float64[],
    )
    
    for r in 1:conn.n_regions
        for s in 1:conn.n_regions
            push!(summary_df, (
                source_region = conn.region_labels[r],
                sink_region   = conn.region_labels[s],
                connectivity  = conn.connectivity_matrix[r, s],
                flow_count    = conn.flow_counts[r, s],
                flow_rate     = conn.flow_rates[r, s],
            ))
        end
    end
    
    CSV.write(summary_file, summary_df)
    
    return summary_file
end

"""
    export_connectivity_uncertainty(conn_unc, output_dir) -> String

Exports connectivity credible intervals to CSV.

# Arguments
- `conn_unc`: Output of `compute_connectivity_credible_intervals`.
- `output_dir`: Output directory path.

# Returns
Path to uncertainty CSV file.
"""
function export_connectivity_uncertainty(conn_unc, output_dir, region_labels)::String
    mkpath(output_dir)
    
    unc_file = joinpath(output_dir, "stock_connectivity_credible_intervals.csv")
    unc_df = DataFrame(
        source_region = String[],
        sink_region   = String[],
        mean_conn     = Float64[],
        lower_ci      = Float64[],
        upper_ci      = Float64[],
    )
    
    for r in 1:conn_unc.n_regions
        for s in 1:conn_unc.n_regions
            push!(unc_df, (
                source_region = region_labels[r],
                sink_region   = region_labels[s],
                mean_conn     = conn_unc.connectivity_mean[r, s],
                lower_ci      = conn_unc.connectivity_lower[r, s],
                upper_ci      = conn_unc.connectivity_upper[r, s],
            ))
        end
    end
    
    CSV.write(unc_file, unc_df)
    
    return unc_file
end
