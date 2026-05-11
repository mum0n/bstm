# automatic load of things related to all projects go here

using Pkg
Pkg.add( ["DrWatson", "Revise", "Requires", "PrecompileTools", "PackageCompiler"] )

using DrWatson, Revise, Requires, PrecompileTools, PackageCompiler

if !@isdefined project_directory 
    project_directory = joinpath( homedir(), "projects", "bstm" )
end

quickactivate(project_directory) 

# the following are now handled by DrWatson.quickactivate()
# Pkg.activate(project_directory)  # so now you activate the package
# Base.active_project()  
# push!( LOAD_PATH, project_directory )  # add the directory to the load path, so it can be found

current_directory =  @__DIR__() 
print( "Current directory is: ", current_directory, "\n\n" )

  
pkgs_bstm = [
  "DrWatson", "Revise", "Requires", "PrecompileTools", "PackageCompiler", 
  "Random", "Plots", "StatsPlots", "LibGEOS", "Graphs", "DelaunayTriangulation", "OrderedCollections",  
  "Distributions", "Statistics", "MCMCChains", "DataFrames",  "GLM", "FlexiChains", "AbstractPPL",
  "LinearAlgebra", "Clustering", "StatsBase", "HypothesisTests", "KernelFunctions",
  "JLD2", "FFTW",  "SparseArrays", "StaticArrays", "FillArrays", "AbstractGPs",
  "Bijectors", "DynamicPPL", "AdvancedVI", "Optimisers", "Optim", "PosteriorStats",  "Turing" 
]

# force install all:
Pkg.add(pkgs_bstm)

# load them all:
for pk in pkgs_bstm;  @eval using $(Symbol(pk)) end
 
# tidy loose ends:
Pkg.gc()

# Pkg.precompile()
# Pkg.instantiate()
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
include( srcdir( "data_prep.jl") )
include( srcdir( "spatiotemporal_functions.jl" ))   
include( srcdir( "spatiotemporal_turing_models.jl" ))   


Random.seed!(42) # Set a seed for reproducibility.


# required for a waic: 
import LogExpFunctions: logistic
import LogExpFunctions: logsumexp

# Extend base names check for ADVI pseudo-chain
using Turing: Variational

# MCMCChains.names(chain::NamedTuple) = collect(keys(chain.data))  # USED? 

# to help track variables, add something like this inside of a function:  
# Main.DEBUG[] = y,p,t  # this stores y, p, t into Main.DEBUG 
DEBUG = Ref{Any}()  # initiate

print( "\nTo Debug a variable, place some like the following into your function: \n
  Main.DEBUG[] = y,p,t  # this stores y, p, t into Main.DEBUG \n
which means, you can see what these values are by typing: DEBUG.y, etc... \n")
