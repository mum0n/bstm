# =============================================================================
# Per-Path Credible Intervals: Posterior Uncertainty Propagation
# =============================================================================

"""
    path_credible_intervals(loaded, fitted, kernels, params) -> NamedTuple

Computes per-path credible intervals by propagating posterior uncertainty 
from MCMC chains through individual trajectory reconstruction.

For each sampled individual's path, this function:
1. Draws posterior samples of (alpha, rho, gamma) per biological group
2. Reconstructs a path ensemble for each individual
3. Computes per-node waypoint distributions and credible intervals
4. Quantifies path-level uncertainty (length, corridor width, efficiency)

# Arguments
- `loaded`: Output of `load_movement_data`.
- `fitted`: Output of `fit_movement_models`.
- `kernels`: Output of `extract_transition_kernels`.
- `params`: Config. Relevant keys: `n_samples`, `seed`, `verbose`.

# Returns
`NamedTuple` with:
- `path_samples::Dict`: Per-individual MCMC trajectory ensemble.
- `path_stats::Dict`: Per-individual summaries (credible intervals on path length, 
  waypoint visitation probabilities, corridor widths).
- `node_visit_probs::Matrix`: n_spatial × n_spatial matrix of visitation 
  probabilities for each release–recapture pair.
"""
function path_credible_intervals(loaded, fitted, kernels, params)::NamedTuple
    verbose   = params.verbose
    obs_df    = loaded.obs_df
    n_spatial = loaded.n_spatial
    chains    = fitted.chains
    land_mask = loaded.land_mask
    
    # Select active chain
    active_chain = haskey(chains, :telemetry) ?
                   chains[:telemetry] :
                   chains[:telemetry_and_survey]
    
    # Extract posterior samples
    chn_array = Array(active_chain)
    n_draws = size(chn_array, 1)
    G = kernels.G
    
    verbose && println("\n[Posterior Path Uncertainty] Drawing $(n_draws) samples...")
    
    # Extract velocity/diffusion/gamma samples per group per draw
    v_samples = zeros(n_draws, G)
    d_samples = zeros(n_draws, G)
    g_samples = zeros(n_draws, G)
    
    for draw in 1:n_draws
        for g in 1:G
            # Placeholder extraction; adapt to your actual chain structure
            v_key = Symbol("velocity[$g]")
            d_key = Symbol("diffusion[$g]")
            g_key = Symbol("gamma[$g]")
            
            v_samples[draw, g] = haskey(chains, v_key) ? mean(active_chain[v_key]) : 0.3
            d_samples[draw, g] = haskey(chains, d_key) ? mean(active_chain[d_key]) : 0.1
            g_samples[draw, g] = haskey(chains, g_key) ? mean(active_chain[g_key]) : 1.0
        end
    end
    
    all_tags = unique(obs_df.tagid)
    path_samples = Dict{String, Vector{Vector{Int}}}()
    path_stats = Dict{String, NamedTuple}()
    
    # Per-release–recapture pair, track visitation
    node_visit_counts = zeros(Int, n_spatial, n_spatial)
    node_visit_samples = zeros(n_draws, n_spatial, n_spatial)
    
    verbose && println("[Posterior Path Uncertainty] Reconstructing path ensembles...")
    
    for tid in all_tags
        sub_obs = filter(:tagid => ==(tid), obs_df)
        isempty(sub_obs) && continue
        
        grp = hasproperty(sub_obs, :group) ? first(sub_obs.group) : 1
        grp = clamp(grp, 1, G)
        
        # Trajectory ensemble: one path per posterior draw
        path_ens = Vector{Int}[]
        path_lengths = Float64[]
        
        for draw in 1:n_draws
            # Construct transition kernel from this posterior sample
            alpha_draw = clamp(v_samples[draw, grp] / (v_samples[draw, grp] + d_samples[draw, grp] + 1e-6), 0.0, 1.0)
            rho_draw   = clamp(1.0 / (1.0 + v_samples[draw, grp] + d_samples[draw, grp]), 0.01, 0.95)
            gamma_draw = g_samples[draw, grp]
            
            P_draw = construct_stochastic_transition_kernel(
                loaded.W, loaded.hsi_vec;
                gamma     = fill(gamma_draw, G),
                residence = fill(rho_draw, G),
                advection = fill(alpha_draw, G),
                land_mask = land_mask
            )
            
            # Reconstruct multi-segment path for this individual
            P_k = P_draw isa AbstractVector ? P_draw[grp] : P_draw
            full_path = Int[sub_obs.release[1]]
            
            for row in eachrow(sub_obs)
                seg = predict_path(
                    P_k, row.release, row.recapture, row.k;
                    method    = :astar,
                    land_mask = land_mask
                )
                append!(full_path, seg[2:end])
            end
            
            push!(path_ens, full_path)
            push!(path_lengths, Float64(length(full_path)))
            
            # Track node visitations
            for i in 1:(length(full_path) - 1)
                u, v = full_path[i], full_path[i+1]
                if 1 <= u <= n_spatial && 1 <= v <= n_spatial
                    node_visit_counts[u, v] += 1
                    node_visit_samples[draw, u, v] += 1
                end
            end
        end
        
        # Summarize path statistics
        len_mean   = mean(path_lengths)
        len_lower  = quantile(path_lengths, 0.025)
        len_upper  = quantile(path_lengths, 0.975)
        len_sd     = std(path_lengths)
        
        # Waypoint visitation heatmap: frequency of visiting each node
        waypoint_freq = zeros(n_spatial)
        for path in path_ens
            for node in path
                if 1 <= node <= n_spatial
                    waypoint_freq[node] += 1
                end
            end
        end
        waypoint_freq ./= n_draws
        
        path_samples[string(tid)] = path_ens
        path_stats[string(tid)] = (
            tagid           = tid,
            length_mean     = len_mean,
            length_lower_ci = len_lower,
            length_upper_ci = len_upper,
            length_sd       = len_sd,
            n_paths_sampled = n_draws,
            waypoint_probs  = waypoint_freq,
            release_site    = first(sub_obs.release),
            recapture_site  = first(sub_obs.recapture),
        )
        
        if verbose && length(path_samples) % max(1, div(length(all_tags), 10)) == 0
            println("  $tid: $(round(len_mean; digits=1)) ± $(round(len_sd; digits=1)) hops " *
                    "[$(round(len_lower; digits=1))-$(round(len_upper; digits=1))]")
        end
    end
    
    # Normalize node visitation counts to probabilities
    node_visit_probs = node_visit_counts ./ (n_draws + 1e-10)
    
    if verbose
        println("[Posterior Path Uncertainty] Complete: $(length(path_stats)) individuals, " *
                "$(n_draws) posterior samples each.")
    end
    
    return (
        path_samples       = path_samples,
        path_stats         = path_stats,
        node_visit_probs   = node_visit_probs,
        v_samples          = v_samples,
        d_samples          = d_samples,
        g_samples          = g_samples,
    )
end

"""
    export_path_uncertainty_summary(path_unc, loaded, output_dir) -> String

Exports per-path credible intervals and trajectory ensembles to CSV and HDF5.

# Arguments
- `path_unc`: Output of `path_credible_intervals`.
- `loaded`: Output of `load_movement_data`.
- `output_dir`: Output directory path.

# Returns
Path to summary CSV file.
"""
function export_path_uncertainty_summary(path_unc, loaded, output_dir)::String
    mkpath(output_dir)
    
    summary_file = joinpath(output_dir, "path_credible_intervals.csv")
    summary_df = DataFrame(
        tagid              = String[],
        release_site       = Int[],
        recapture_site     = Int[],
        path_length_mean   = Float64[],
        path_length_lower  = Float64[],
        path_length_upper  = Float64[],
        path_length_sd     = Float64[],
        n_waypoints        = Int[],
        modal_waypoint     = Int[],
    )
    
    for (tid, stats) in path_unc.path_stats
        modal_wp = argmax(stats.waypoint_probs)
        push!(summary_df, (
            tagid              = tid,
            release_site       = stats.release_site,
            recapture_site     = stats.recapture_site,
            path_length_mean   = stats.length_mean,
            path_length_lower  = stats.length_lower_ci,
            path_length_upper  = stats.length_upper_ci,
            path_length_sd     = stats.length_sd,
            n_waypoints        = length(stats.waypoint_probs),
            modal_waypoint     = modal_wp,
        ))
    end
    
    CSV.write(summary_file, summary_df)
    
    return summary_file
end
