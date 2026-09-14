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
    
    @reexport using 
        AbstractGPs, 
        AbstractMCMC,
        DynamicPPL,
        AdvancedVI,
        GLM,
        StatsBase,
        Statistics, 
        DataFrames,
        ADTypes,
        KernelAbstractions,
        Bijectors,
        CategoricalArrays,
        Clustering,
        ColorSchemes,
        Dates,
        DelaunayTriangulation,
        DimensionalData,
        DuckDB,
        LogExpFunctions,
        Distances,
        FFTW,
        FillArrays,
        FlexiChains,
        Graphs,
        HypothesisTests,
        Interpolations,
        JLD2,
        KernelFunctions,
        LibGEOS,
        LinearAlgebra,
        NamedArrays,
        NearestNeighbors,
        Optim,
        Optimisers,
        OrderedCollections,
        PDMats,
        Printf,
        Plots,
        PosteriorStats,
        Requires,
        SHA,
        SparseArrays,
        SpecialFunctions,
        StaticArrays,
        StatsModels,
        StatsPlots,
        ForwardDiff,
        ReverseDiff,
        Enzyme,     
        Wavelets, 
        WaveletsExt, 
        NNlib, 
        CoordRefSystems, 
        Unitful
    # using Unitful: ustrip, @u_str
    # using NearestNeighbors: KDTree, inrange, knn
    # using SparseArrays: sparse, max.
    # using Statistics: mean
    # using LinearAlgebra: mul!, I

    srcdir = @__DIR__  # bstm/src
    rootdir = dirname(srcdir) # bstm
    docsdir = joinpath(rootdir, "docs") # bstm/docs

    # Core framework files 
    include( "definitions.jl")  # must be first
    
    # order is important:
    include( "data.jl")
    include( "derivatives.jl")
    include( "hierarchical.jl") 
    include( "input_output.jl")
    include( "circuit.jl")
    include( "graph_wavelets.jl")
    include( "leaflet.jl")
    include( "likelihoods.jl")
    include( "model.jl")
    include( "parameters.jl")
    include( "partitioning.jl")
    include( "pipeline.jl")
    include( "plotting.jl") 
    include( "reconstruction.jl")  
    include("par.jl")

    # component definitions
    components_dir = joinpath(srcdir, "components")

    for f in readdir(components_dir)
        if endswith(f, ".jl")
            include(joinpath(components_dir, f))
        end
    end

    # work in progress: 
    # include( joinpath(docsdir, "hierarchical_workflow", "hierarchical_workflow.jl") )


    # User-facing API exports
    export 
        @bstm,
        bstm_config,
        model_results_comprehensive,
        convert_to_chains,
        reconstruct,
        get_optimal_sampler,
        precompute_step_sizes,
        predict,
        show_model,
        bstm_surface_derivatives,
        compute_topographic_metrics,
        generate_rff_params,
        build_structure_template,
        bstm_pipeline,
        compute_network_transfer_matrix,
        reshard_spatial_field,
        summarize_sample_matrix,
        PipelineResult,
        PipelineTierSpec,
        bstm_cv_orchestrator,
        bstm_plots,
        plot_spatial_surface,
        bstm_sample,
        save_plots,
        _detect_xy_columns,
        _detect_time_column,
        _detect_response_column,
        _detect_spatial_unit_column,
        _detect_seasonal_column,
        _detect_group_column,
        _resolve_nested_strata,
        NESTED_COUPLING_MODES,
        STANDARD_SPATIAL_COORDINATE_PAIRS,
        STANDARD_TEMPORAL_CANDIDATES,
        STANDARD_SPATIAL_UNIT_CANDIDATES,
        STANDARD_SEASONAL_CANDIDATES,
        STANDARD_GROUP_CANDIDATES,

        assign_spatial_units_inferred, 
        plot_kde_simple,
        assign_spatial_units, 
        assign_time_units, 
        assign_spatiotemporal_units,
        discretize_data, 
        bstm_data, 
        expand_hull, 
        generate_mock_hierarchical_datasets,
        map_to_units,
        spatial_block_cv, 
        spatial_weights_matrix, 
        spatial_knn_graph,
        spatial_radius_graph, 
        scaling_factor_bym2,
        load_open_bathymetry,
        extract_hydrodynamic_dataset,
        build_hex_mesh_planar,
        map_point_to_units,
        lonlat_to_xy_km,
        xy_km_to_lonlat,

        ParamRegistry, 
        ParamDescriptor, 
        build_param_registry,
        calibrate_param_registry, 
        get_samples, 
        get_param_samples,
        get_descriptors_by_role,
        get_effects,
 
        create_theme, 
        choropleth, 
        timeseries_ci, 
        spatial_graph_plot,
        render_paths!, 
        map_point_occupancy, 
        save_plot, 
        model_results_plots,
        plot_hsi_choropleth, 
        plot_diffusion_map, 
        plot_residence_time_map,
        plot_advection_arrows, 
        plot_velocity_field, 
        plot_hexagonal_field,
        plot_hydrodynamic_stratification,
        plot_hydrodynamic_diffusion,
        plot_hydrodynamic_section,
        householder_to_eigenvector, 
        eigenvector_to_householder,

        # Leaflet Interactive HTML System exports
        LeafletMap, 
        save_html, 
        utm_to_lonlat, 
        lonlat_to_utm,
        leaflet_choropleth, 
        leaflet_spatial_map, 
        leaflet_spatial_graph,
        leaflet_hsi_map, 
        leaflet_diffusion_map, 
        leaflet_residence_time_map,
        leaflet_advection_arrows, 
        leaflet_velocity_field,
        leaflet_spacetime_map,

        # Circuit Theory & Ecological Connectivity exports
        Circuit,
        PosteriorCircuitResult,
        build_circuit_laplacian,
        effective_resistance_matrix,
        solve_circuit_voltage,
        pairwise_effective_resistance,
        current_density_map,
        identify_ecological_pinchpoints,
        identify_stochastic_pinchpoints,
        posterior_circuit_inference,
        get_circuit_paths,
        resistance_covariance_matrix,
        leaflet_current_density_map,

        # Spectral Graph Wavelets (SGWT) exports
        GraphWavelet,
        SpectralGraphWaveletResult,
        build_normalized_laplacian,
        compute_laplacian_spectral_bounds,
        chebyshev_polynomial_coefficients,
        apply_graph_spectral_filter,
        spectral_graph_wavelet_transform,
        inverse_spectral_graph_wavelet_transform,
        denoise_spatial_signal_wavelet,
        graph_wavelet_basis_matrix,
        leaflet_graph_wavelet_dashboard,

        # Input / Output & Persistence exports
        save_bstm_model, 
        load_bstm_model,
        save_bstm_results, 
        load_bstm_results, 
        query_duckdb,   
        export_posterior_samples_to_duckdb, 
        import_posterior_samples_from_duckdb,
        append_posterior_samples, 
        extend_sampling,
        save_bstm_bundle, 
        load_bstm_bundle,
        export_spatial_results_to_geojson, 
        extract_posterior_priors,
        extract_prior_posterior,
        PriorPosteriorBundle,
        save_model_ensemble, 
        bma_weighted_predictions, 
        save_out_of_sample_predictions,
        export_results_to_parquet, 
        export_results_to_csv, 
        compact_duckdb,

        # Hierarchical Pipeline & Provenance exports
        compute_data_traits, 
        init_pipeline_manifest!, 
        write_tier_table!, 
        read_tier_table, 
        has_tier_table,
        get_manifest_entry, 
        update_manifest_entry!, 
        is_tier_up_to_date,
        check_pipeline_status,
        
        par_from_posterior, 
        summarize_par_effects, 
        par_counterfactual,
        export_par_to_table,
        par_credible_interval_plot, 
        par_forest_plot  


    # Module initialization function
    function __init__()
        Random.seed!(42) # Set a seed for reproducibility.
    end
 

end # module bstm
