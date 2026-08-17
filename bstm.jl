module bstm

    # Use Reexport for its macro, but be selective about what is re-exported.
    using Reexport

    # List of packages to be re-exported by bstm.jl.
    # This provides a unified namespace for the user. 
    # Only re-export Turing, as bstm is a framework built on Turing.
    # This makes Turing's @model macro and other core functionalities
    # directly available when 'using bstm'.
    @reexport using Turing
    @reexport using Distributions

    # Packages that are fundamental to bstm's internal operation
    # but whose entire API is NOT intended to be part of bstm's public API.
    # Users should explicitly 'using' these if they need their full API.
    using AbstractGPs, AbstractMCMC, AbstractPPL, ADTypes, AdvancedVI, KernelAbstractions, GPUArrays,
          Bijectors, CategoricalArrays, Clustering, DataFrames,
          DelaunayTriangulation, DimensionalData, LogExpFunctions,
          Distances, DynamicPPL, FFTW, FillArrays, FlexiChains,
          NNlib, GLM, Graphs, HypothesisTests, Interpolations, JLD2,
          KernelFunctions, LibGEOS, LinearAlgebra, MCMCChains, NamedArrays,
          NearestNeighbors, Optim, Optimisers, OrderedCollections, PDMats,
          Plots, PosteriorStats, Random, Requires, SparseArrays,
          SpecialFunctions, StaticArrays, Statistics, StatsBase, StatsModels,
          StatsPlots, Wavelets, WaveletsExt, ForwardDiff, ReverseDiff, Enzyme
    
    rootdir = @__DIR__

    srcdir = joinpath(rootdir, "src")

    # Core framework files
    includet(joinpath(srcdir, "definitions.jl"))  # must be first
    includet(joinpath(srcdir, "data.jl"))
    includet(joinpath(srcdir, "partitioning.jl"))
    includet(joinpath(srcdir, "model.jl"))
    includet(joinpath(srcdir, "likelihoods.jl"))
    includet(joinpath(srcdir, "reconstruction.jl")) 
      
    # component definitions
    components_dir = joinpath(srcdir, "components")
    
    for f in readdir(components_dir)
        if endswith(f, ".jl")
            includet(joinpath(components_dir, f))
        end
    end

    # User-facing API exports
    export @bstm, model_results_comprehensive, get_optimal_sampler, predict, show_model
    export bstm_cv_orchestrator, bstm_plots, bstm_sample
    export assign_spatial_units_inferred, plot_spatial_graph, plot_kde_simple, plot_choropleth
    export assign_spatial_units, assign_time_units, bstm_data

    function load_project_functions( src_dir )
      
      fns_main = [
        "definitions.jl",
        "data.jl", 
        "partitioning.jl", 
        "model.jl", 
        "likelihoods.jl", 
        "reconstruction.jl"
      ]

      fns = joinpath.( src_dir, fns_main )

      fnc = readdir( joinpath( src_dir, "components" ) )
      for fn in fnc 
        push!(fns, joinpath(src_dir, "components", fn) )
      end

      
      for fn in fns 
        if endswith(fn, ".jl")
          try
            println(fn)  
            include(fn)
          catch e
              @error "Error including file '$fn':" e
          end
        end
      end

    end


    # Module initialization function
    function __init__()
      Random.seed!(42) # Set a seed for reproducibility.
      # @info "bstm module loaded from $(@__DIR__)."
    end


        
    if false 
        
        # for debugging: 

        # to help track variables, add something like this inside of a function:  
        # Main.DEBUG[] = y,p,t  # this stores y, p, t into Main.DEBUG 
        DEBUG = Ref{Any}()  # initiate

        print( "\nTo Debug a variable, place something like the following into your function: \n
          Main.DEBUG[] = y,p,t  # this stores y, p, t into Main.DEBUG \n
        which means, you can see what these values are by typing: DEBUG.y, etc... \n")


        using Pkg
        Pkg.add( ["Reexport", "DrWatson", "Revise"  ] )
        using Reexport, DrWatson, Revise 

        if !@isdefined project_directory 
            project_directory = joinpath( "C:\\home\\jae", "projects", "bstm" )
        end

        cd(project_directory)
        quickactivate(project_directory) 

            # This provides a unified namespace for the user.
        pkgs_bstm = [ 
            "AbstractGPs", "AbstractMCMC", "AbstractPPL", "ADTypes", "AdvancedVI", 
            "Bijectors", "CategoricalArrays", "Clustering", "DataFrames", "KernelAbstractions", "GPUArrays",
            "DelaunayTriangulation", "DimensionalData", "LogExpFunctions", "Turing", 
            "Distances", "Distributions", "DynamicPPL", "FFTW", "FillArrays", "FlexiChains", 
            "NNlib", "GLM", "Graphs", "HypothesisTests", "Interpolations", "JLD2", 
            "KernelFunctions", "LibGEOS", "LinearAlgebra", "MCMCChains", "NamedArrays", 
            "NearestNeighbors", "Optim", "Optimisers", "OrderedCollections", "PDMats", 
            "Plots", "PosteriorStats", "Random", "Requires", "SparseArrays", 
            "SpecialFunctions", "StaticArrays", "Statistics", "StatsBase", "StatsModels", 
            "StatsPlots", "Wavelets", "WaveletsExt", "ForwardDiff", "ReverseDiff", "Enzyme"
        ]

        # load them all:
        try
          for pk in pkgs_bstm;  @eval using $(Symbol(pk)); end
        catch e
          # force install all (if in amn incomplete state or first run):
          Pkg.add(pkgs_bstm);
          for pk in pkgs_bstm;  @eval using $(Symbol(pk)); end
          print( "\nInstall not complete or inconsistent, installing required packages. This might require multiple restarts and a bit of time...hours? \n\n" ) 
          Pkg.instantiate()
          Pkg.precompile()
          Pkg.gc() # tidy loose ends:
        end
    
        load_project_functions( "C:\\home\\jae\\projects\\bstm\\src" )
    
        if false

          cd( "/home/jae/projects/bstm" ) # where you saved it
          include("bstm.jl")
          using .bstm

          data_scot, _ = bstm_data(); # Default is "scottish_lip"
          inp_df = data_scot.data;
          W = data_scot.au.W

          m = @bstm(
              likelihood(y, family=poisson, log_offsets=log_offsets) ~
                  intercept() +
                  fixed(cov1) +
                  random(s_idx, model=bym2) +
                  random(year, model=ar1),
              inp_df,
              W = W,
              use_gpu = true,
              verbose = false # Suppress verbose output
          ); 

          os = get_optimal_sampler(m, 5)

          chn = sample(m, MH(), 100; progress=false)
          chn = sample(m, NUTS(), 10; progress=false)
          chn = sample(m, os, 10; progress=false)
          
          res = model_results_comprehensive( m, chn; au=data_scot.au);

          println(res.summary_stats)

        end

    end

end # module bstm
