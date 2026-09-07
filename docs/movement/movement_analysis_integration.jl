# =============================================================================
# INTEGRATION: Load Priority Analysis Modules
# =============================================================================

# Include the three priority analysis modules
include(joinpath(@__DIR__, "path_uncertainty.jl"))
include(joinpath(@__DIR__, "connectivity_analysis.jl"))
include(joinpath(@__DIR__, "posterior_predictive_check.jl"))
include(joinpath(@__DIR__, "priority_analyses.jl"))

# =============================================================================
# Main Execution: run_movement_analysis
# =============================================================================

"""
    run_movement_analysis(params = movement_parameters_default())

Main entry point for the complete BSTM movement analysis pipeline.

Orchestrates all phases:
1. Data ingestion & validation
2. Bayesian model fitting
3. Transition kernel extraction
4. Individual path reconstruction
5. Priority post-processing analyses:
   - Per-path credible intervals
   - Stock connectivity matrix
   - Posterior predictive checks

# Arguments
- `params`: Configuration NamedTuple (default: `movement_parameters_default()`).

# Returns
`NamedTuple` containing outputs from all pipeline phases.

# Example
```julia
# Run with defaults
result = run_movement_analysis()

# Run with custom species config
p = merge(movement_parameters_snowcrab(), (n_samples = 500,))
result = run_movement_analysis(p)
```
"""
function run_movement_analysis(params = movement_parameters_default())::NamedTuple
    verbose = params.verbose
    output_dir = params.output_dir
    
    # =========================================================================
    # Phase 1: Data Ingestion
    # =========================================================================
    loaded = load_movement_data(params)
    
    # =========================================================================
    # Phase 2: Model Fitting
    # =========================================================================
    fitted = fit_movement_models(loaded, params)
    
    # =========================================================================
    # Phase 3: Kernel Construction
    # =========================================================================
    kernels = extract_transition_kernels(loaded, fitted, params)
    
    # =========================================================================
    # Phase 4: Path Reconstruction & Basic Diagnostics
    # =========================================================================
    reconstructed = reconstruct_paths_and_diagnostics(loaded, kernels, params)
    
    # =========================================================================
    # PRIORITY ANALYSES: Uncertainty + Connectivity + Validation
    # =========================================================================
    priority_output_dir = joinpath(output_dir, "priority_analyses")
    priority_results = run_priority_analyses(
        loaded, fitted, kernels, params, priority_output_dir
    )
    
    # =========================================================================
    # Consolidated Summary
    # =========================================================================
    if verbose
        println("\n" * "=" ^ 72)
        println("  PIPELINE COMPLETE")
        println("=" ^ 72)
        println("\nOutput directory: $output_dir")
        println("\nKey Results:")
        println("  ✓ Individuals analyzed: $(length(reconstructed.paths))")
        println("  ✓ Path uncertainty: $(length(priority_results.path_uncertainty.path_stats)) individuals")
        println("  ✓ Connectivity regions: $(priority_results.connectivity_matrix.n_regions)")
        println("  ✓ Model validation: Brier=$(round(priority_results.posterior_predictive.summary.brier_mean; digits=4))")
        println("\nSummary: $(priority_results.summary_file)")
        println("=" ^ 72 * "\n")
    end
    
    return (
        loaded            = loaded,
        fitted            = fitted,
        kernels           = kernels,
        reconstructed     = reconstructed,
        priority_analyses = priority_results,
        output_dir        = output_dir,
    )
end

# =============================================================================
# CLI Argument Parsing (Optional)
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    using ArgParse
    
    function parse_cli_args()
        s = ArgParseSettings(
            description = "Bayesian Spatio-Temporal Movement (BSTM) Analysis Pipeline"
        )
        @add_arg_table! s begin
            "--species"
                help = "Species preset: 'snowcrab' or 'generic'"
                arg_type = String
                default = "generic"
            "--n-samples"
                help = "Number of MCMC samples per chain"
                arg_type = Int
                default = 200
            "--n-warmup"
                help = "Number of MCMC warmup iterations"
                arg_type = Int
                default = 100
            "--output-dir"
                help = "Output directory path"
                arg_type = String
                default = joinpath(pwd(), "output")
            "--verbose"
                help = "Enable verbose logging"
                action = :store_true
            "--help"
                help = "Show this help message"
                action = :show_help
        end
        return parse_args(s)
    end
    
    args = parse_cli_args()
    
    # Select parameter preset
    params = if args["species"] == "snowcrab"
        movement_parameters_snowcrab()
    else
        movement_parameters_default()
    end
    
    # Override with CLI args
    params = merge(params, (
        n_samples  = args["n-samples"],
        n_warmup   = args["n-warmup"],
        output_dir = args["output-dir"],
        verbose    = args["verbose"],
    ))
    
    # Run analysis
    result = run_movement_analysis(params)
end
