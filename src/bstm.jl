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
          Dates, DelaunayTriangulation, DimensionalData, DuckDB, DynamicPPL, LogExpFunctions,
          Distances, FFTW, FillArrays, FlexiChains,
          NNlib, GLM, Graphs, HypothesisTests, Interpolations, JLD2,
          KernelFunctions, LibGEOS, LinearAlgebra, NamedArrays,
          NearestNeighbors, Optim, Optimisers, OrderedCollections, PDMats,
          Printf, Plots, PosteriorStats, Random, Requires, SHA, SparseArrays,
          SpecialFunctions, StaticArrays, Statistics, StatsBase, StatsModels,
          StatsPlots, Wavelets, WaveletsExt, ForwardDiff, ReverseDiff, Enzyme
    
    rootdir = @__DIR__

    srcdir = joinpath(rootdir, "src")

    # Core framework files
    include( "definitions.jl")  # must be first
    include( "data.jl")
    include( "partitioning.jl")
    include( "parameters.jl")
    include( "model.jl")
    include( "likelihoods.jl")
    include( "reconstruction.jl") 
    include( "plotting.jl") 
    include( "input_output.jl")
    include( "movement.jl")
    include( "derivatives.jl")
    include( "pipeline.jl")
    include( "hierarchical.jl")
      
    # component definitions
    components_dir = joinpath(@__DIR__, "components")

    for f in readdir(components_dir)
        if endswith(f, ".jl")
            include(joinpath(components_dir, f))
        end
    end

    # User-facing API exports
    export @bstm, model_results_comprehensive, get_optimal_sampler
    export precompute_step_sizes, predict, show_model
    export bstm_surface_derivatives, compute_topographic_metrics
    export bstm_pipeline, compute_network_transfer_matrix, reshard_spatial_field
    export summarize_sample_matrix, PipelineResult, PipelineTierSpec
    export bstm_cv_orchestrator, bstm_plots, bstm_sample, save_plots
    export assign_spatial_units_inferred, plot_kde_simple
    export assign_spatial_units, assign_time_units, assign_spatiotemporal_units
    export discretize_data, bstm_data, expand_hull, generate_mock_hierarchical_datasets
    export map_to_units
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

    # Hierarchical Pipeline & Provenance exports
    export compute_data_traits, init_pipeline_manifest!
    export write_tier_table!, read_tier_table, has_tier_table
    export get_manifest_entry, update_manifest_entry!, is_tier_up_to_date
    export check_pipeline_status


    # Module initialization function
    function __init__()
      Random.seed!(42) # Set a seed for reproducibility.
      # @info "bstm module loaded from $(@__DIR__)."
    end
 

end # module bstm
