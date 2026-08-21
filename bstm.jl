module bstm

    # Use Reexport for its macro, but be selective about what is re-exported.

    # List of packages to be re-exported by bstm.jl.
    # This provides a unified namespace for the user. 
    # Only re-export Turing, as bstm is a framework built on Turing.
    # This makes Turing's @model macro and other core functionalities
    # directly available when 'using bstm'.
    using Reexport

    @reexport using Distributions
    @reexport using Turing
    # @reexport using DynamicPPL

    # Packages that are fundamental to bstm's internal operation
    # but whose entire API is NOT intended to be part of bstm's public API.
    # Users should explicitly 'using' these if they need their full API.
    using AbstractGPs, AbstractMCMC, ADTypes, AdvancedVI, KernelAbstractions,  
          Bijectors, CategoricalArrays, Clustering, ColorSchemes, DataFrames,
          DelaunayTriangulation, DimensionalData, DuckDB, DynamicPPL, LogExpFunctions,
          Distances, FFTW, FillArrays, FlexiChains,
          NNlib, GLM, Graphs, HypothesisTests, Interpolations, JLD2,
          KernelFunctions, LibGEOS, LinearAlgebra, NamedArrays,
          NearestNeighbors, Optim, Optimisers, OrderedCollections, PDMats,
          Plots, PosteriorStats, Random, Requires, SparseArrays,
          SpecialFunctions, StaticArrays, Statistics, StatsBase, StatsModels,
          StatsPlots, Wavelets, WaveletsExt, ForwardDiff, ReverseDiff, Enzyme
    
    rootdir = @__DIR__

    srcdir = joinpath(rootdir, "src")

    # Core framework files
    include(joinpath(srcdir, "definitions.jl"))  # must be first
    include(joinpath(srcdir, "data.jl"))
    include(joinpath(srcdir, "partitioning.jl"))
    include(joinpath(srcdir, "parameters.jl"))
    include(joinpath(srcdir, "model.jl"))
    include(joinpath(srcdir, "likelihoods.jl"))
    include(joinpath(srcdir, "reconstruction.jl")) 
    include(joinpath(srcdir, "plotting.jl")) 
    include(joinpath(srcdir, "input_output.jl"))
    include(joinpath(srcdir, "movement.jl"))
      
    # component definitions
    components_dir = joinpath(srcdir, "components")
    
    for f in readdir(components_dir)
        if endswith(f, ".jl")
            include(joinpath(components_dir, f))
        end
    end

    # User-facing API exports
    export @bstm, model_results_comprehensive, get_optimal_sampler
    export precompute_step_sizes, predict, show_model
    export bstm_cv_orchestrator, bstm_plots, bstm_sample, save_plots
    export assign_spatial_units_inferred, plot_kde_simple
    export assign_spatial_units, assign_time_units, assign_spatiotemporal_units
    export discretize_data, bstm_data
    export spatial_block_cv, spatial_weights_matrix, spatial_knn_graph
    export spatial_radius_graph, scaling_factor_bym2
    export ParamRegistry, ParamDescriptor, build_param_registry
    export calibrate_param_registry, get_samples, get_param_samples
 
    export create_theme, choropleth, timeseries_ci, spatial_graph_plot
    export render_paths!, map_point_occupancy, save_plot, model_results_plots

    # Movement & ADR Telemetry exports
    export generate_ADR_simulation_bundle, simulate_correlated_density_vector
    export compute_velocity_field, calculate_multistep_transition
    export simulate_posterior_trajectories, simulate_mechanistic_trajectories
    export compute_suitability_transition_kernel, calculate_regional_connectivity
    export plot_ad_ratio_distribution, synthesize_adr_results

    # Input / Output & Persistence exports
    export save_bstm_model, load_bstm_model
    export save_bstm_results, load_bstm_results, query_duckdb
    export export_posterior_samples_to_duckdb, import_posterior_samples_from_duckdb
    export append_posterior_samples, extend_sampling
    export save_bstm_bundle, load_bstm_bundle
    export export_spatial_results_to_geojson, extract_posterior_priors
    export save_model_ensemble, bma_weighted_predictions, save_out_of_sample_predictions
    export export_results_to_parquet, export_results_to_csv, compact_duckdb


    # Module initialization function
    function __init__()
      Random.seed!(42) # Set a seed for reproducibility.
      # @info "bstm module loaded from $(@__DIR__)."
    end
 

end # module bstm
