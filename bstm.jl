module bstm

using Pkg
Pkg.add( ["Reexport", "DrWatson", "Revise"  ] )

using Reexport, DrWatson, Revise 

current_directory =  @__DIR__() 
print( "Current directory is: ", current_directory, "\n\n" )


pkgs_bstm = [
    "AbstractGPs", "AbstractMCMC", "AbstractPPL", "ADTypes", "AdvancedVI", "Bijectors", 
    "CategoricalArrays", "Clustering", "DataFrames", "DelaunayTriangulation", "DimensionalData", "LogExpFunctions",
    "Distances", "Distributions", "DynamicPPL", "FFTW", "FillArrays", "FlexiChains", "NNlib", 
    "GLM", "Graphs", "HypothesisTests", "Interpolations", "JLD2", "KernelFunctions",
    "LibGEOS", "LinearAlgebra", "MCMCChains", "NamedArrays", "NearestNeighbors",
    "Optim", "Optimisers", "OrderedCollections", "PDMats", "Plots", "PosteriorStats", "Random", 
    "Requires", "Revise", "SparseArrays", "SpecialFunctions", "StaticArrays", "Statistics", 
    "StatsBase", "StatsModels", "StatsPlots", "Wavelets", "WaveletsExt",
    "ForwardDiff", "ReverseDiff", "Enzyme"   # , "Zygote"
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

# Re-export core functionality from Turing.jl for convenience
@reexport using Turing

include(joinpath("src", "definitions.jl"))  # must be first
include(joinpath("src", "data.jl"))
include(joinpath("src", "partitioning.jl"))
include(joinpath("src", "model.jl"))
include(joinpath("src", "likelihoods.jl"))
include(joinpath("src", "reconstruction.jl")) 
   
# component definitions
for f in readdir(joinpath(@__DIR__, "src", "components"))
    if endswith(f, ".jl")
        include(joinpath("src", "components", f))
    end
end

export @bstm, model_results_comprehensive, get_optimal_sampler, predict, show_model
export bstm_cv_orchestrator, bstm_plots, bstm_sample
export assign_spatial_units_inferred, plot_spatial_graph, plot_kde_simple, plot_choropleth
export assign_spatial_units, assign_time_units, bstm_data

end # module bstm
