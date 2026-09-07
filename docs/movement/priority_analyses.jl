# =============================================================================
# Integration Module: Priority Analyses Pipeline
# =============================================================================

"""
    run_priority_analyses(loaded, fitted, kernels, params, output_dir) -> NamedTuple

Orchestrates the three priority post-processing analyses:
1. Per-path credible intervals (posterior uncertainty propagation)
2. Stock connectivity matrix (inter-region movement flows)
3. Posterior predictive check (model validation)

Integrates outputs and generates consolidated exports.

# Arguments
- `loaded`: Output of `load_movement_data`.
- `fitted`: Output of `fit_movement_models`.
- `kernels`: Output of `extract_transition_kernels`.
- `params`: Configuration NamedTuple.
- `output_dir`: Root output directory.

# Returns
`NamedTuple` with all three analysis outputs plus unified summary.
"""
function run_priority_analyses(loaded, fitted, kernels, params, output_dir)::NamedTuple
    verbose = params.verbose
    
    verbose && println("\n" * "=" ^ 72)
    verbose && println("  PRIORITY POST-PROCESSING ANALYSES")
    verbose && println("=" ^ 72)
    
    mkpath(output_dir)
    
    # --- Analysis 1: Per-Path Credible Intervals ---
    path_unc = path_credible_intervals(loaded, fitted, kernels, params)
    path_unc_file = export_path_uncertainty_summary(path_unc, loaded, output_dir)
    verbose && println("\n✓ Path uncertainty summary: $path_unc_file")
    
    # --- Analysis 2: Stock Connectivity Matrix ---
    # Option 1: Use default (per-mesh-unit regions)
    conn = compute_stock_connectivity_matrix(loaded, kernels, params)
    conn_file = export_connectivity_matrix(conn, output_dir)
    verbose && println("✓ Stock connectivity matrix: $conn_file")
    
    # Option 2: Compute with posterior uncertainty
    region_map = haskey(params, :region_map) ? params.region_map : collect(1:loaded.n_spatial)
    conn_unc = compute_connectivity_credible_intervals(
        loaded, fitted, kernels, params, region_map
    )
    conn_unc_file = export_connectivity_uncertainty(
        conn_unc, output_dir, conn.region_labels
    )
    verbose && println("✓ Connectivity uncertainty: $conn_unc_file")
    
    # --- Analysis 3: Posterior Predictive Check ---
    ppc = posterior_predictive_check(loaded, fitted, kernels, params)
    ppc_file = export_posterior_predictive_check(ppc, output_dir)
    verbose && println("✓ Posterior predictive check: $ppc_file")
    
    # Attempt to generate diagnostic plots (conditional on Plots.jl availability)
    try
        plot_ppc_file = plot_posterior_predictive_check(ppc, output_dir)
        verbose && println("✓ PPC diagnostic plots: $plot_ppc_file")
    catch e
        verbose && @warn "Could not generate PPC plots (Plots.jl may be unavailable): $e"
    end
    
    # --- Unified Summary Report ---
    summary_file = joinpath(output_dir, "PRIORITY_ANALYSES_SUMMARY.txt")
    open(summary_file, "w") do f
        write(f, "PRIORITY ANALYSES SUMMARY\n")
        write(f, "=" ^ 70 * "\n\n")
        
        write(f, "1. PER-PATH CREDIBLE INTERVALS\n")
        write(f, "-" ^ 70 * "\n")
        write(f, "Individuals analyzed: $(length(path_unc.path_stats))\n")
        write(f, "Posterior draws per individual: $(first(values(path_unc.path_stats)).n_paths_sampled)\n\n")
        
        path_lengths = [s.length_mean for s in values(path_unc.path_stats)]
        write(f, "Path length summary:\n")
        write(f, "  Mean:    $(round(mean(path_lengths); digits=2)) hops\n")
        write(f, "  Range:   [$(round(minimum(path_lengths); digits=2)), " *
                  "$(round(maximum(path_lengths); digits=2))]\n")
        write(f, "  Std Dev: $(round(std(path_lengths); digits=2))\n\n")
        
        write(f, "Output file: $path_unc_file\n\n")
        
        write(f, "2. STOCK CONNECTIVITY MATRIX\n")
        write(f, "-" ^ 70 * "\n")
        write(f, "Regions: $(conn.n_regions)\n")
        write(f, "Total observed transitions: $(sum(conn.flow_counts))\n")
        write(f, "Connectivity matrix shape: $(size(conn.connectivity_matrix))\n\n")
        
        # Find top connectivity pairs
        top_pairs = sort(
            [(r, s, conn.connectivity_matrix[r, s]) 
             for r in 1:conn.n_regions for s in 1:conn.n_regions],
            by = x -> x[3],
            rev = true
        )[1:min(5, conn.n_regions^2)]
        
        write(f, "Top 5 strongest connectivity pathways:\n")
        for (r, s, prob) in top_pairs
            write(f, "  $(conn.region_labels[r]) → $(conn.region_labels[s]): " *
                     "$(round(prob; digits=4))\n")
        end
        write(f, "\n")
        write(f, "Output files: $conn_file, $conn_unc_file\n\n")
        
        write(f, "3. POSTERIOR PREDICTIVE CHECK\n")
        write(f, "-" ^ 70 * "\n")
        write(f, "Observations validated: $(ppc.summary.n_observations)\n")
        write(f, "Posterior draws: $(ppc.summary.n_draws)\n\n")
        
        write(f, "Model fit diagnostics:\n")
        write(f, "  Brier Score (MSE):  $(round(ppc.summary.brier_mean; digits=6))\n")
        write(f, "    95% CI: [$(round(ppc.summary.brier_lower_ci; digits=6)), " *
                  "$(round(ppc.summary.brier_upper_ci; digits=6))]\n")
        write(f, "  KL Divergence:      $(round(ppc.summary.kl_mean; digits=6))\n")
        write(f, "    95% CI: [$(round(ppc.summary.kl_lower_ci; digits=6)), " *
                  "$(round(ppc.summary.kl_upper_ci; digits=6))]\n")
        write(f, "  Recapture Entropy:  $(round(ppc.summary.observed_dist_entropy; digits=6))\n\n")
        
        # Interpretation
        if ppc.summary.kl_mean < 0.5
            fit_interp = "EXCELLENT: Model closely reproduces observed distribution."
        elseif ppc.summary.kl_mean < 1.0
            fit_interp = "GOOD: Model adequately reproduces observed distribution."
        elseif ppc.summary.kl_mean < 2.0
            fit_interp = "MODERATE: Model shows notable discrepancy from observations."
        else
            fit_interp = "POOR: Model significantly mismatches observations."
        end
        write(f, "Interpretation: $fit_interp\n\n")
        
        write(f, "Output file: $ppc_file\n\n")
        
        write(f, "=" ^ 70 * "\n")
        write(f, "Generated: $(Dates.now())\n")
    end
    
    verbose && println("\n✓ Unified summary: $summary_file\n")
    
    return (
        path_uncertainty        = path_unc,
        connectivity_matrix     = conn,
        connectivity_uncertainty = conn_unc,
        posterior_predictive    = ppc,
        summary_file            = summary_file,
    )
end
