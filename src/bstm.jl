module bstm

    # Use Reexport for its macro, but be selective about what is re-exported.

    # List of packages to be re-exported by bstm.jl.
    # This provides a unified namespace for the user. 
    # Only re-export Turing, as bstm is a framework built on Turing.
    # This makes Turing's @model macro and other core functionalities
    # directly available when 'using bstm'.
    using Reexport

    @reexport using Random 
    @reexport using Distributions
    @reexport using Turing
    @reexport using AbstractGPs
    @reexport using AbstractMCMC
    @reexport using DynamicPPL
    @reexport using AdvancedVI
    
    @reexport using GLM
    @reexport using StatsBase
    @reexport using Statistics 
    @reexport using DataFrames
    @reexport using ArgParse
    @reexport using ADTypes
    @reexport using KernelAbstractions
    @reexport using Bijectors
    @reexport using CategoricalArrays
    @reexport using ColorSchemes
    @reexport using Dates
    @reexport using DelaunayTriangulation
    @reexport using DimensionalData
    @reexport using DuckDB
    @reexport using LogExpFunctions
    @reexport using Distances
    @reexport using FFTW
    @reexport using FillArrays
    @reexport using FlexiChains
    @reexport using Graphs
    @reexport using HypothesisTests
    @reexport using Interpolations
    @reexport using JLD2
    @reexport using KernelFunctions
    @reexport using LibGEOS
    @reexport using LinearAlgebra
    @reexport using NamedArrays
    @reexport using NearestNeighbors
    @reexport using Optim
    @reexport using Optimisers
    @reexport using OrderedCollections
    @reexport using PDMats
    @reexport using Printf
    @reexport using Plots
    @reexport using PosteriorStats
    @reexport using Requires
    @reexport using SHA
    @reexport using SparseArrays
    @reexport using SpecialFunctions
    @reexport using StaticArrays
    @reexport using StatsModels
    @reexport using StatsPlots
    @reexport using ForwardDiff
    @reexport using ReverseDiff
    @reexport using Enzyme

 


    # Packages that are fundamental to bstm's internal operation
    # but whose entire API is NOT intended to be part of bstm's public API.
    # Users should explicitly 'using' these if they need their full API.
    using Wavelets, WaveletsExt, NNlib 
        
    srcdir = @__DIR__  # bstm/src
    rootdir = dirname(srcdir) # bstm
    docsdir = joinpath(rootdir, "docs") # bstm/docs

    # Core framework files 
    include( "definitions.jl")  # must be first
    
    include( "data.jl")
    include( "derivatives.jl")
    include( "hierarchical.jl") 
    include( "input_output.jl")
    include( "likelihoods.jl")
    include( "model.jl")
    include( "movement.jl")
    include( "parameters.jl")
    include( "partitioning.jl")
    include( "pipeline.jl")
    include( "plotting.jl") 
    include( "reconstruction.jl")  
 
    # component definitions
    components_dir = joinpath(srcdir, "components")

    for f in readdir(components_dir)
        if endswith(f, ".jl")
            include(joinpath(components_dir, f))
        end
    end

    # work in progress: 
    # include( joinpath(docsdir, "hierarchical_workflow", "hierarchical_workflow.jl") )
    include( joinpath(docsdir, "movement", "movement_simple.jl") )


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
