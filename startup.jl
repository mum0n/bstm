# automatic load of things related to all projects go here

using Pkg
Pkg.add( ["DrWatson", "Revise", "Requires", "PrecompileTools", "PackageCompiler"] )

using DrWatson, Revise, Requires, PrecompileTools, PackageCompiler

if !@isdefined project_directory 
    project_directory = joinpath( homedir(), "projects", "bstm" )
end

cd(project_directory)
quickactivate(project_directory) 

# the following are now handled by DrWatson.quickactivate()
# Pkg.activate(project_directory)  # so now you activate the package
# Base.active_project()  
# push!( LOAD_PATH, project_directory )  # add the directory to the load path, so it can be found

current_directory =  @__DIR__() 
print( "Current directory is: ", current_directory, "\n\n" )


pkgs_bstm = [
  "DrWatson", "Revise", "Requires", "PrecompileTools", "PackageCompiler", "SpecialFunctions", "DimensionalData",
  "Random", "Plots", "StatsPlots", "LibGEOS", "Graphs", "DelaunayTriangulation", "OrderedCollections",  
  "Distributions", "Statistics", "MCMCChains", "DataFrames",  "GLM", "FlexiChains", "AbstractPPL",
  "LinearAlgebra", "Clustering", "StatsBase", "HypothesisTests", "KernelFunctions",
  "JLD2", "FFTW",  "SparseArrays", "StaticArrays", "FillArrays", "AbstractGPs", 
  "Bijectors", "DynamicPPL", "AdvancedVI", "Optimisers", "Optim", "PosteriorStats",  "Turing",  
  "Distances", "NamedArrays" , "CategoricalArrays", "StatsModels", "AbstractMCMC", "ForwardDiff", "PDMats"
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



# Pkg.update()


# Pinning LibGEOS to the latest available package version to resolve API inconsistencies
# Pkg.add(name="LibGEOS", version="0.9.7")
  
  
if false
  # For RCall:  (not needed in this project .. but in case you need it)
  if Sys.iswindows()
    # ENV["R_HOME"] = "C:\Program Files\R\R-4.5.2\bin\x64\Rgui.exe"
    ENV["R_HOME"] = "C:\\Program Files\\R\\R-4.5.2"
    ENV["path"] = string( ENV["R_HOME"], "\\bin\\x64; ", ENV["path"] )
    #    using Pkg; Pkg.build("RCall")
  elseif Sys.islinux()
  elseif Sys.isapple()
  else
  end
end

  

print( "\nTo (re)-install required packages, run:  install_required_packages() or Pkg.instantiate() \n\n" ) 
  

# support functions
# include( srcdir( "data_prep.jl") );
# include( srcdir( "unit_test_functions.jl" ))   ;
# include( srcdir( "legacy.jl" ))   ;
# include( srcdir( "example_turing_models.jl" ))   ;

include( srcdir( "utility_functions.jl" ))   ;
include( srcdir( "structs.jl" ))   ;

include( srcdir( "spatiotemporal_partitioning_functions.jl" ))   ;
include( srcdir( "spatiotemporal_functions.jl" ))   ;
include( srcdir( "build_model_dispatch.jl" ))   ;
include( srcdir( "bstm_model_supervisors.jl" ))   ;
include( srcdir( "reconstruction_engine.jl" ))   ;
include( srcdir( "visualization_engine.jl" ))   ;


   
allfiles = unique( pushfirst!( readdir(srcdir() ), "structs.jl" ) )

for filename in allfiles 
  if endswith(filename, ".jl")
    filepath = joinpath(directory_path, filename)
    try
        include(filepath)
    catch e
        @error "Error including file '$filepath':" e
    end
end
 

Random.seed!(42) # Set a seed for reproducibility.

import MCMCChains
import DynamicPPL
import StatsPlots

using Statistics: mean, std, median, quantile, var, cor, Diagonal, eigen
using StatsBase: Weights, sample, midpoints

using AbstractMCMC: logdensity # Explicitly import logdensity

import SpecialFunctions: logfactorial
import LogExpFunctions: logdiffexp, logistic, logsumexp, log1mexp
import Distributions: logpdf, _logpdf, pdf, cdf, logcdf, logccdf, rand, sampler

using LogExpFunctions: logistic, logsumexp, log1mexp


# Extend base names check for ADVI pseudo-chain
using Turing: Variational

# MCMCChains.names(chain::NamedTuple) = collect(keys(chain.data))  # USED? 

# to help track variables, add something like this inside of a function:  
# Main.DEBUG[] = y,p,t  # this stores y, p, t into Main.DEBUG 
DEBUG = Ref{Any}()  # initiate

print( "\nTo Debug a variable, place some like the following into your function: \n
  Main.DEBUG[] = y,p,t  # this stores y, p, t into Main.DEBUG \n
which means, you can see what these values are by typing: DEBUG.y, etc... \n")
