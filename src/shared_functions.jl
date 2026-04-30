
function packages_used(model_variation) 

    pkgs_shared = [
        "DrWatson", "Revise", "Test",  "OhMyREPL", "Logging", 
        "StatsBase", "Statistics", "Distributions", "Random", "Setfield", "Memoization", 
        "MCMCChains", 
        "DataFrames", "JLD2", "CSV", "PlotThemes", "Colors", "ColorSchemes", "RData",  
        "Plots",  "StatsPlots", "MultivariateStats", 
        "ForwardDiff", "ReverseDiff", "Enzyme", "ADTypes",
        "StaticArrays", "LazyArrays", "FillArrays", "LinearAlgebra", "MKL", "Turing"
    ]

    if occursin( r"logistic_discrete.*", model_variation )
        pkgs = []
    elseif occursin( r"size_structured_dde.*", model_variation )
        pkgs = [
            "QuadGK", "ModelingToolkit", "DifferentialEquations", "Interpolations",
        ]
    end
    
    pkgs = unique!( [pkgs_shared; pkgs] )
  
    return pkgs  

end
 

function install_required_packages(pkgs)    # to install packages
    for pk in pkgs; 
        if Base.find_package(pk) === nothing
            Pkg.add(pk)
        end
    end   # Pkg.add( pkgs ) # add required packages
    print( "Pkg.add( \"Bijectors\" , version => \"0.3.16\") # may be required \n" )
end
 
function init_params_extract(X)
  XS = summarize(X)
  vns = XS.nt.parameters  # var names
  init_params = FillArrays.Fill( XS.nt[2] ) # means
  return init_params, vns
end

 
function discretize_decimal( x, delta=0.01 ) 
    num_digits = Int(ceil( log10(1.0 / delta)) )   # time floating point rounding
    out = round.( round.( x ./ delta; digits=0 ) .* delta; digits=num_digits)
    return out
end
 

function expand_grid(; kws...)
    names, vals = keys(kws), values(kws)
    return DataFrame(NamedTuple{names}(t) for t in Iterators.product(vals...))
end
   

function showall( x )
    # print everything to console
    show(stdout, "text/plain", x) # display all estimates
end 
 

function firstindexin(a::AbstractArray, b::AbstractArray)
    bdict = Dict{eltype(b), Int}()
    for i=length(b):-1:1
        bdict[b[i]] = i
    end
    [get(bdict, i, 0) for i in a]
end
   
  
function β( mode, conc )
    # alternate parameterization of beta distribution 
    # conc = α + β     https://en.wikipedia.org/wiki/Beta_distribution
    beta1 = mode *( conc - 2  ) + 1.0
    beta2 = (1.0 - mode) * ( conc - 2  ) + 1.0
    Beta( beta1, beta2 ) 
end 
  
function modelruntime(o)
    dt = ( o.info.stop_time- o.info.start_time )/ 60
    showall( summarize(o) )
    print( dt )
end
 
function code_show(x)
   # printstyled( CodeTracking.@code_string x() )
end
  

function install_required_packages()    # to install packages
    for pk in pkgs; 
        if Base.find_package(pk) === nothing
            Pkg.add(pk)
        end
    end   # Pkg.add( pkgs ) # add required packages

    print( "Pkg.add( \"Bijectors\" , version => \"0.3.16\") # may be required \n" )

end
 


function turingindex( indices, sym=nothing, dims=nothing  ) 
     
    if isa(indices, DynamicPPL.Model)
        _, indices = bijector(turing_model, Val(true));
    end

    if isnothing(sym)
      out = enumerate(keys(indices))
    elseif sym=="varnames"
      out = keys(indices)
    else
      out = union(indices[sym]...)
    end
    
    if !isnothing(dims)
        out = reshape(out, dims)
    end

    return out 
  end





function discretize_data(x; dx=0.5, nx=13, method="regular")   

  if method=="regular"    

    xd = round.(Int, x ./ dx ) .* dx   # resolution to 0.1 units
    xd_breaks = collect( minimum(xd):dx:maximum(xd) + dx  ) 
    xd_mid = midpoints(xd_breaks)
    nx = length(xd_mid)
    
    xd_cut = cut(x, xd_breaks, extend=true)  # from CategoricalArrays
    xi = levelcode.(xd_cut)  # integer index
  
  elseif method=="quantile"
  
    xd_breaks = quantile(x, range(0, 1, length=nx+1))
    xd_mid = midpoints(xd_breaks)
    xd_cut = cut(x, xd_breaks, extend=true)  # from CategoricalArrays
    xi = levelcode.(xd_cut)  # integer index
    dx = diff(xd_mid)[1]
    xd = xd_mid[xi] 

  end

  return xd, xi, xd_mid, nx, dx

end




function random_correlation_matrix(d=3, eta=1)

# etas = [1 10 100 1000 1e+4 1e+5];
# d = size of matrix

# EXTENDED ONION METHOD to generate random correlation matrices
# distributed ~ det(S)^eta [or maybe det(S)^(eta-1), not sure]
# https://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices

# LKJ modify this method slightly, in order to be able to sample correlation matrices C from a distribution proportional to [detC]η−1. The larger the η, the larger will be the determinant, meaning that generated correlation matrices will more and more approach the identity matrix. The value η=1 corresponds to uniform distribution. On the figure below the matrices are generated with η=1,10,100,1000,10000,100000. 

    beta = eta + (d-2)/2;
    u = rand( Beta(beta, beta) );
    r12 = 2*u - 1;
    S = [1 r12; r12 1];  

    for k = 3:d
        beta = beta - 1/2;
        y = rand( Beta((k-1)/2, beta) );  # sample from beta
        r = sqrt(y);
        theta = randn(k-1,1);
        theta = theta/norm(theta);
        w = r*theta;
        U, E = eigen(S);
        U = hcat(U)
        R = U' * sqrt(E) * U; # R is a square root of S
        q = R[].re * w;
        S = [S q; q' 1];
    end
    return S
end




function build_st_inputs(time_indices, space_indices, spatial_coords)
  # Space-Time Input Construction
  # Space and Time as continuous coordinates.
  # Inputs: 
  #   spatial_coords: Matrix (2 x N_nodes) -> [Lat, Lon]
  #   time_coords: Vector (T_steps)
  # Returns:
  #   ColVecs of 3D points (Time, Lat, Lon)

  # Map indices to actual coordinates
  # This assumes spatial_coords is 2xN

  # Extract coords for every observation
  coords = spatial_coords[:, space_indices] # 2 x N_obs
  times = time_indices' # 1 x N_obs

  # Stack to create 3D input: [Time; Lat; Lon]
  return ColVecs(vcat(times, coords))
end
