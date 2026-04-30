# automatic load of things related to all projects go here

using Pkg
Pkg.add( ["DrWatson", "Revise", "Requires", "PrecompileTools", "PackageCompiler"] )

using DrWatson, Revise, Requires, PrecompileTools, PackageCompiler

if !@isdefined project_directory 
    project_directory = joinpath( homedir(), "projects", "model_covariance" )
end

quickactivate(project_directory) 

# the following are now handled by DrWatson.quickactivate()
# Pkg.activate(project_directory)  # so now you activate the package
# Base.active_project()  
# push!( LOAD_PATH, project_directory )  # add the directory to the load path, so it can be found

current_directory =  @__DIR__() 
print( "Current directory is: ", current_directory, "\n\n" )



# For Areal Units
using Pkg
pkgs_au = ["Random", "Statistics", "LinearAlgebra", "DataFrames",
       "StatsBase", "SparseArrays", "Plots", "StatsPlots", "StaticArrays",
        "JLD2", "LibGEOS", "Graphs", "DelaunayTriangulation" ]


# For CARSTM 
using Pkg
pkgs_carstm = ["Random",   "Distributions", "Statistics", "MCMCChains", "DataFrames",
        "LinearAlgebra", "Clustering", "StatsBase", "HypothesisTests",
        "JLD2", "FFTW",  "SparseArrays", "StaticArrays", "FillArrays",
         "Bijectors", "DynamicPPL", "AdvancedVI", "Optimisers", "PosteriorStats",  "Turing" ]
 

pkgs = unique( [  "DrWatson", "Revise", "Requires", pkgs_au, pkgs_carstm ] )

Pkg.add(pkgs)
for pk in pkgs_au; @eval using $(Symbol(pk)); end

# Pkg.precompile()
# Pkg.instantiate()
# Pkg.gc()
 
  
if false
  # For RCall:
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
  
# to help track variables, add something like this inside of a function:  
# Main.DEBUG[] = y,p,t  # this stores y, p, t into Main.DEBUG 
DEBUG = Ref{Any}()  # initiate

# support functions
include( srcdir( "example_data.jl" ))     
include( srcdir( "shared_functions.jl") )
  
include( srcdir( "car_functions.jl" ))   
include( srcdir( "carstm_functions.jl" ))   

include( srcdir( "spatial_partitioning_functions.jl" ))   
include( srcdir( "spatiotemporal_functions.jl" ))   
include( srcdir( "spatiotemporal_turing_models.jl" ))   


# Set a seed for reproducibility.
Random.seed!(42)

using LogExpFunctions: logistic

