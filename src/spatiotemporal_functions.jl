


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


function showtuples(X)
    for k in keys(X)
        val = getproperty(X, k)
        # Skip displaying keys with NaN values
        # Check if value is numeric before rounding to avoid errors
        display_val = val isa Number ? round(val, digits=3) : val
        println("$k: $display_val")
    end
end



function showparams(X, keywords=["rho", "phi", "sigma",  "mu_", "l_", "ls_"]; limit=10 )
    # Create a regex pattern by joining keywords with the pipe '|' operator
    pattern = Regex(join(keywords, "|"))

    # Filter the parameter list
    matched_params = filter(p -> occursin(pattern, string(p)), FlexiChains.parameters(X))

    # Display the filtered slice
    if isempty(matched_params)
        println("No parameters matched keywords: $keywords")
    else
        out = X[matched_params[1:min(limit, end)]]
        # display(out)
        return out
    end
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

 

function expand_hull(pts, buffer_dist)
    """
    Synopsis: Computes the convex hull of points and expands it by a buffer distance.
    Inputs:
    - pts: Vector of (x, y) tuples.
    - buffer_dist: Distance to buffer the convex hull.
    Outputs:
    - A LibGEOS Polygon geometry representing the buffered convex hull.
    """

    if isempty(pts) return LibGEOS.Polygon([[ (0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0) ]]) end
    coords_vec = [[Float64(p[1]), Float64(p[2])] for p in pts]
    points_geom = LibGEOS.MultiPoint(coords_vec)
    hull = LibGEOS.convexhull(points_geom)
    buffered_hull = LibGEOS.buffer(hull, buffer_dist)
    return buffered_hull
end



function get_kde_seeds(pts, target_u)
    # Basic KDE-based seeding using StatsBase weights based on local density
    if isempty(pts) return [] end
    n = length(pts)
    dists = [sum((p1 .- p2).^2) for p1 in pts, p2 in pts]
    # Inverse of mean distance as a density proxy
    weights = 1.0 ./ (mean(dists, dims=2)[:] .+ 1e-6)
    idx = sample(1:n, Weights(weights), min(target_u, n), replace=false)
    return pts[idx]
end

 




function get_cvt_centroids(pts, cfg, hull_geom)
    """
    Synopsis: Centroidal Voronoi Tessellation (CVT) with diagnostic termination tracking.
    """
    u_pts = unique(pts)
    idx = StatsBase.sample(1:length(u_pts), min(cfg.target, length(u_pts)), replace=false)
    curr_centroids = [u_pts[i] for i in idx]
    termination_reason = "max_iterations"
    last_mean_density = 0.0
    last_cv = 0.0

    for iter in 1:100
        polys, _ = get_voronoi_polygons_and_edges(curr_centroids, hull_geom)
        new_centroids = Tuple{Float64, Float64}[]
        shifts = Float64[]

        for i in 1:length(polys)
            poly_coords = polys[i]
            area = get_polygon_area(poly_coords)

            if length(poly_coords) > 2 && area >= cfg.min_a && area <= cfg.max_a
                lg_poly = LibGEOS.Polygon([[ [p[1], p[2]] for p in poly_coords ]])
                cent_geom = LibGEOS.centroid(lg_poly)
                seq = LibGEOS.getCoordSeq(cent_geom)
                new_c = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))

                dist = sqrt(sum((new_c .- curr_centroids[i]).^2))
                push!(shifts, dist)
                push!(new_centroids, new_c)
            else
                push!(new_centroids, curr_centroids[i])
            end
        end

        if isempty(shifts) || mean(shifts) < cfg.tol
            termination_reason = "convergence"
            break
        end

        assigns = [argmin([sum((p .- c).^2) for c in new_centroids]) for p in pts]
        counts = [count(==(i), assigns) for i in 1:length(new_centroids)]

        if mean(counts) < cfg.min_p
            termination_reason = "min_points_violation"
            break
        end

        # New Density Convergence Check
        curr_mean_density = mean(counts)
        if abs(curr_mean_density - last_mean_density) < cfg.tol && iter > 1
            termination_reason = "density_convergence"
            break
        end
        last_mean_density = curr_mean_density

        cv_val = std(counts) / (mean(counts) + 1e-9)
        # CV Convergence Check
        if abs(cv_val - last_cv) < cfg.tol && iter > 1
            termination_reason = "cv_convergence"
            break
        end
        last_cv = cv_val

        


        curr_centroids = new_centroids
    end

    return curr_centroids, termination_reason
end

function get_kvt_centroids(pts, cfg, hull_geom)
    """
    Synopsis: K-means Voronoi Tessellation (KVT) with diagnostic termination tracking.
    """
    u_pts = unique(pts)
    idx_init = sample(1:length(u_pts), min(cfg.target, length(u_pts)), replace=false)
    c_iter = [u_pts[i] for i in idx_init]
    data = collect(zip(pts, cfg.t_idx))
    damping = 0.7
    termination_reason = "max_iterations"
    last_mean_density = 0.0
    last_cv = 0.0

    for iter in 1:100
        old_centroids = copy(c_iter)
        assigns = [argmin([sum((p[1] .- sj).^2) for sj in c_iter]) for p in data]

        polys_coords, _ = get_voronoi_polygons_and_edges(c_iter, hull_geom)

        for k in 1:length(c_iter)
            idx_cluster = findall(==(k), assigns)
            ts_count = length(unique([data[j][2] for j in idx_cluster]))

            area = 0.0
            if k <= length(polys_coords)
                area = get_polygon_area(polys_coords[k])
            end

            area_ok = (area > 0) ? (area >= cfg.min_a && area <= cfg.max_a) : true

            if !isempty(idx_cluster) && length(idx_cluster) >= cfg.min_p && ts_count >= cfg.min_ts && area_ok
                mean_x = mean(data[j][1][1] for j in idx_cluster)
                mean_y = mean(data[j][1][2] for j in idx_cluster)

                c_iter[k] = ((1.0 - damping) * old_centroids[k][1] + damping * mean_x,
                             (1.0 - damping) * old_centroids[k][2] + damping * mean_y)
            end
        end

        counts = [count(==(k), assigns) for k in 1:length(c_iter)]
        cv_val = std(counts) / (mean(counts) + 1e-9)

        # New Density Convergence Check
        curr_mean_density = mean(counts)
        if abs(curr_mean_density - last_mean_density) < cfg.tol && iter > 1
            termination_reason = "density_convergence"
            break
        end
        last_mean_density = curr_mean_density

        # CV Convergence Check
        if abs(cv_val - last_cv) < cfg.tol && iter > 1
            termination_reason = "cv_convergence"
            break
        end
        last_cv = cv_val

        


        if mean(counts) < cfg.min_p
            termination_reason = "min_points_violation"
            break
        end

        damping *= 0.99
    end

    return c_iter, termination_reason
end

function is_valid_polygon_coords(poly_coords)
    # Filters out NaN/Inf values and checks for a minimum of 3 valid points for a polygon.
    valid_pts = [p for p in poly_coords if !isnan(p[1]) && !isinf(p[1]) && !isnan(p[2]) && !isinf(p[2])]
    return length(valid_pts) >= 3
end


function get_qvt_centroids(pts, cfg, hull_geom)
    """
    Synopsis: Quadtree Voronoi Tessellation (QVT) with expanded formatting for readability.
    """
    data = collect(zip(pts, cfg.t_idx))
    regions = [data]
    termination_reason = "max_units_reached"

    while length(regions) < cfg.max_u
        yi = argmax([length(r) for r in regions])
        target = regions[yi]

        if length(target) < 2 * cfg.min_p
            termination_reason = "min_points_limit"
            break
        end

        xs = [p[1][1] for p in target]
        ys = [p[1][2] for p in target]
        mx, my = median(xs), median(ys)

        r_splits = [
            filter(p -> p[1][1] <= mx && p[1][2] <= my, target),
            filter(p -> p[1][1] > mx && p[1][2] <= my, target),
            filter(p -> p[1][1] <= mx && p[1][2] > my, target),
            filter(p -> p[1][1] > mx && p[1][2] > my, target)
        ]

        valid_splits = filter(
            r -> length(r) >= cfg.min_p && length(unique([p[2] for p in r])) >= cfg.min_ts, 
            r_splits
        )

        if length(valid_splits) < 2
            termination_reason = "cannot_split_further"
            break
        end

        # Process splitting
        splice!(regions, yi, valid_splits)
        
        candidate_centroids = [
            (mean(p[1][1] for p in r), mean(p[1][2] for p in r)) 
            for r in regions
        ]
        
        polys_coords, _ = get_voronoi_polygons_and_edges(candidate_centroids, hull_geom)

        # Enforcement: Check area violations
        area_violation = any(
            p_coords -> !is_valid_polygon_coords(p_coords) || get_polygon_area(p_coords) < cfg.min_a, 
            polys_coords
        )

        if area_violation
            if length(regions) >= cfg.min_u
                termination_reason = "min_area_violation"
                break
            end
        end
    end

    final_centroids = [
        (mean(p[1][1] for p in r), mean(p[1][2] for p in r)) 
        for r in regions
    ]

    final_status = length(final_centroids) < cfg.min_u ? "insufficient_units_error" : termination_reason

    return final_centroids, final_status
end



function get_qvt_centroids(pts, cfg, hull_geom)
    """
    Synopsis: Quadtree Voronoi Tessellation (QVT) with corrected recursive splitting logic.
    """
    data = collect(zip(pts, cfg.t_idx))
    regions = [data]
    termination_reason = "max_units_reached"

    while length(regions) < cfg.max_u
        # Find the region with the most points to split
        yi = argmax([length(r) for r in regions])
        target = regions[yi]

        if length(target) < 2 * cfg.min_p
            termination_reason = "min_points_limit"
            break
        end

        xs = [p[1][1] for p in target]
        ys = [p[1][2] for p in target]
        mx, my = median(xs), median(ys)

        r_splits = [
            filter(p -> p[1][1] <= mx && p[1][2] <= my, target),
            filter(p -> p[1][1] > mx && p[1][2] <= my, target),
            filter(p -> p[1][1] <= mx && p[1][2] > my, target),
            filter(p -> p[1][1] > mx && p[1][2] > my, target)
        ]

        valid_splits = filter(
            r -> length(r) >= cfg.min_p && length(unique([p[2] for p in r])) >= cfg.min_ts,
            r_splits
        )

        if length(valid_splits) < 2
            # If this specific region can't be split into at least 2 valid parts, 
            # we try to split the next largest, or stop if no others are viable.
            # For simplicity in this logic, we mark as finished for this branch.
            termination_reason = "cannot_split_further"
            break
        end

        # Correct splice: Replace the parent with its valid children
        deleteat!(regions, yi)
        for child in valid_splits
            push!(regions, child)
        end

        # Area check: Only halt if we have already satisfied the minimum unit count
        candidate_centroids = [(mean(p[1][1] for p in r), mean(p[1][2] for p in r)) for r in regions]
        polys_coords, _ = get_voronoi_polygons_and_edges(candidate_centroids, hull_geom)

        area_violation = any(
            p_coords -> !is_valid_polygon_coords(p_coords) || get_polygon_area(p_coords) < cfg.min_a,
            polys_coords
        )

        if area_violation && length(regions) >= cfg.min_u
            termination_reason = "min_area_violation"
            break
        end
    end

    final_centroids = [(mean(p[1][1] for p in r), mean(p[1][2] for p in r)) for r in regions]
    return final_centroids, length(final_centroids) < cfg.min_u ? "insufficient_units_error" : termination_reason
end

function get_bvt_centroids(pts, cfg, hull_geom)
    """
    Synopsis: Binary Voronoi Tessellation (BVT) with corrected recursive splitting logic.
    """
    data = collect(zip(pts, cfg.t_idx))
    regions = [data]
    termination_reason = "max_units_reached"

    while length(regions) < cfg.max_u
        yi = argmax([length(r) for r in regions])
        target = regions[yi]

        if length(target) < 2 * cfg.min_p
            termination_reason = "min_points_limit"
            break
        end

        xs = [p[1][1] for p in target]
        ys = [p[1][2] for p in target]
        dim = std(xs) > std(ys) ? 1 : 2
        vals = [p[1][dim] for p in target]
        med = median(vals)

        r1 = filter(p -> p[1][dim] <= med, target)
        r2 = filter(p -> p[1][dim] > med, target)

        # Validate children
        v1 = length(r1) >= cfg.min_p && length(unique([p[2] for p in r1])) >= cfg.min_ts
        v2 = length(r2) >= cfg.min_p && length(unique([p[2] for p in r2])) >= cfg.min_ts

        if !v1 || !v2
             termination_reason = "statistical_constraints"
             break
        end

        # Correct update
        deleteat!(regions, yi)
        push!(regions, r1)
        push!(regions, r2)

        candidate_centroids = [(mean(p[1][1] for p in r), mean(p[1][2] for p in r)) for r in regions]
        polys_coords, _ = get_voronoi_polygons_and_edges(candidate_centroids, hull_geom)

        area_violation = any(
            p_coords -> !is_valid_polygon_coords(p_coords) || get_polygon_area(p_coords) < cfg.min_a,
            polys_coords
        )

        if area_violation && length(regions) >= cfg.min_u
            termination_reason = "min_area_violation"
            break
        end
    end

    final_centroids = [(mean(p[1][1] for p in r), mean(p[1][2] for p in r)) for r in regions]
    return final_centroids, length(final_centroids) < cfg.min_u ? "insufficient_units_error" : termination_reason
end



function get_avt_centroids(pts, cfg, hull_geom)
    """
    Synopsis: Agglomerative Voronoi Tessellation (AVT) with diagnostic termination tracking.
    """
    u_pts = unique(pts)
    c_init = get_kde_seeds(u_pts, cfg.target)
    data = collect(zip(pts, cfg.t_idx))
    curr_c = [SVector{2, Float64}(c) for c in c_init]
    termination_reason = "min_units_reached"
    last_mean_density = 0.0
    last_cv = 0.0

    while length(curr_c) > cfg.min_u
        assigns = [Int[] for _ in 1:length(curr_c)]
        for i in 1:length(data)
            d = data[i]
            dist_idx = argmin([sum((d[1] .- c).^2) for c in curr_c])
            push!(assigns[dist_idx], i)
        end

        counts = length.(assigns)

        polys_coords, _ = get_voronoi_polygons_and_edges([Tuple(c) for c in curr_c], hull_geom)

        areas = fill(0.0, length(curr_c))
        for i in 1:min(length(curr_c), length(polys_coords))
            areas[i] = get_polygon_area(polys_coords[i])
        end

        violators = []
        for k in 1:length(curr_c)
            ts_count = length(unique([data[idx][2] for idx in assigns[k]]))
            if (counts[k] < cfg.min_p || counts[k] > cfg.max_p ||
                ts_count < cfg.min_ts ||
                (areas[k] > 0 && (areas[k] < cfg.min_a || areas[k] > cfg.max_a)))
                push!(violators, k)
            end
        end

        cv_val = std(counts) / (mean(counts) + 1e-9)

        # New Density Convergence Check
        curr_mean_density = mean(counts)
        if abs(curr_mean_density - last_mean_density) < cfg.tol
            termination_reason = "density_convergence"
            break
        end
        last_mean_density = curr_mean_density

        # CV Convergence Check
        if abs(cv_val - last_cv) < cfg.tol && length(curr_c) < cfg.target
            termination_reason = "cv_convergence"
            break
        end
        last_cv = cv_val

        


        if isempty(violators)
             termination_reason = "no_violators"
             break
        end

        target_idx = violators[argmin(counts[violators])]
        dists = [sum((curr_c[target_idx] .- curr_c[j]).^2) for j in 1:length(curr_c)]
        dists[target_idx] = Inf
        neighbor_idx = argmin(dists)

        total_n = counts[target_idx] + counts[neighbor_idx]
        curr_c[neighbor_idx] = (curr_c[target_idx] .* counts[target_idx] .+ curr_c[neighbor_idx] .* counts[neighbor_idx]) ./ (total_n + 1e-9)

        deleteat!(curr_c, target_idx)
    end

    return [Tuple(c) for c in curr_c], termination_reason
end
 


function assign_spatial_units(input_data, area_method=nothing; target_units=10, kwargs...)
    # Overload to handle adjacency matrices directly
    if input_data isa AbstractMatrix
        
        reason = :inferred
        W = input_data

        au_inferred = assign_spatial_units_inferred(W;
            iterations=get(kwargs, :iterations, 50),
            learning_rate=get(kwargs, :learning_rate, 0.1),
            buffer_dist=get(kwargs, :buffer_dist, 0.5),
            input_polygons=get(kwargs, :input_polygons, nothing))
 
        pts = au_inferred.centroids
        final_centroids = au_inferred.centroids
        new_assigns = [argmin([sum((p .- sj).^2) for sj in final_centroids]) for p in pts]
        polys_coords = au_inferred.polygons        
        v_edges = au_inferred.adjacency_edges
        g = au_inferred.graph
        hull_coords = au_inferred.hull_coords


    else

        cfg = (target=target_units, min_u=get(kwargs, :min_total_arealunits, 3), 
            max_u=get(kwargs, :max_total_arealunits, target_units*2), 
            min_ts=get(kwargs, :min_time_slices, 1), min_p=get(kwargs, :min_points, 1), 
            max_p=get(kwargs, :max_points, length(input_data)), min_a=get(kwargs, :min_area, 0.0), 
            max_a=get(kwargs, :max_area, Inf), cv_min=get(kwargs, :cv_min, 1.0), 
            tol=get(kwargs, :tolerance, 0.1), buff=get(kwargs, :buffer_dist, 0.5), 
            t_idx=get(kwargs, :t_idx, ones(Int, length(input_data))))

        hull_geom = expand_hull(input_data, cfg.buff)

        c_mid, reason = if area_method == :cvt get_cvt_centroids(input_data, cfg, hull_geom)
        elseif area_method == :kvt get_kvt_centroids(input_data, cfg, hull_geom)
        elseif area_method == :qvt get_qvt_centroids(input_data, cfg, hull_geom)
        elseif area_method == :bvt get_bvt_centroids(input_data, cfg, hull_geom)
        elseif area_method == :avt get_avt_centroids(input_data, cfg, hull_geom)
        else error("Unknown partitioning method: $area_method") end

        polys_coords, v_edges = get_voronoi_polygons_and_edges(c_mid, hull_geom)

        final_centroids = Tuple{Float64, Float64}[]
        lg_polys = []
        for p_coords in polys_coords
            if isempty(p_coords); continue; end
            if p_coords[1] != p_coords[end]; push!(p_coords, p_coords[1]); end
            lg_p = LibGEOS.Polygon([[ [pt[1], pt[2]] for pt in p_coords ]])
            push!(lg_polys, lg_p)
            cent_g = LibGEOS.centroid(lg_p)
            seq = LibGEOS.getCoordSeq(cent_g)
            push!(final_centroids, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
        end

        new_assigns = [argmin([sum((p .- sj).^2) for sj in final_centroids]) for p in input_data]
        n_units = length(final_centroids)
        g = SimpleGraph(n_units)
        for i in 1:n_units, j in (i+1):n_units
            # Use robust check here too
            if LibGEOS.touches(lg_polys[i], lg_polys[j]) || LibGEOS.intersects(LibGEOS.buffer(lg_polys[i], 1e-7), lg_polys[j])
                add_edge!(g, i, j)
            end
        end
        g = ensure_connected!(g, final_centroids)

        hull_coords = get_coords_from_geom(hull_geom)
        pts = input_data

        W = Float64.( Graphs.adjacency_matrix(g) )
 
    end

    return (centroids=final_centroids, assignments=new_assigns, polygons=polys_coords, 
            adjacency_edges=v_edges, graph=g, hull_coords=hull_coords, 
            termination_reason=reason, pts=pts, W=W, s_vals=collect(1:size(W,1)))
end
 

function assign_spatial_units_inferred(adjacency_matrix; input_polygons=nothing, iterations=50, learning_rate=0.1, buffer_dist=0.5)
    """
    Synopsis: Replacement for assign_spatial_units_inferred using the refactored workflow semantics.
    Handles spatial inference from a connectivity matrix (W) or extracts structure from provided polygons.
    """
    # 1. Consolidate constraints
    nAU = size(adjacency_matrix, 1)
    cfg = (
        iters = iterations,
        lr    = learning_rate,
        buff  = buffer_dist
    )

    local final_centroids
    local polys_output
    local hull_coords_output

    if input_polygons !== nothing && !isempty(input_polygons)
        # Case A: Polygons provided
        # Extract centroids from geometries
        final_centroids_geoms = [LibGEOS.centroid(p) for p in input_polygons]
        final_centroids = map(final_centroids_geoms) do g_pt
            seq = LibGEOS.getCoordSeq(g_pt)
            (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
        end

        # Determine hull and polygons
        united_geom = LibGEOS.unaryunion(input_polygons)
        hull_coords_output = get_coords_from_geom(united_geom)
        polys_output = [get_coords_from_geom(p) for p in input_polygons]

    else
        # Case B: Infer structure from adjacency matrix via force-directed layout
        g_layout = SimpleGraph(adjacency_matrix)
        side = ceil(Int, sqrt(nAU))
        centroids_vec = [SVector{2, Float64}(Float64(i % side), Float64(i ÷ side)) for i in 0:(nAU-1)]

        for iter in 1:cfg.iters
            new_centroids_vec = copy(centroids_vec)
            for i in 1:nAU
                nb = Graphs.neighbors(g_layout, i)
                if !isempty(nb)
                    avg_pos = sum(centroids_vec[n] for n in nb) / length(nb)
                    new_centroids_vec[i] = centroids_vec[i] + cfg.lr * (avg_pos - centroids_vec[i])
                end
            end
            centroids_vec = new_centroids_vec
        end
        
        inferred_pts = [(p[1], p[2]) for p in centroids_vec]
        hull_geom = expand_hull(inferred_pts, cfg.buff)
        hull_coords_output = get_coords_from_geom(hull_geom)

        # Generate tessellation based on inferred positions
        polys_output, _ = get_voronoi_polygons_and_edges(inferred_pts, hull_geom)

        # Refine centroids based on clipped polygons
        final_centroids = Tuple{Float64, Float64}[]
        for p_coords in polys_output
            if p_coords[1] != p_coords[end] push!(p_coords, p_coords[1]) end
            lg_p = LibGEOS.Polygon([[ [pt[1], pt[2]] for pt in p_coords ]])
            cent_g = LibGEOS.centroid(lg_p)
            seq = LibGEOS.getCoordSeq(cent_g)
            push!(final_centroids, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
        end
    end

    # 2. Finalize Adjacency and Connectivity (Standardized with assign_spatial_units)
    n_final = length(final_centroids)
    lg_polys = []
    for p_coords in polys_output
        if p_coords[1] != p_coords[end] push!(p_coords, p_coords[1]) end
        push!(lg_polys, LibGEOS.Polygon([[ [pt[1], pt[2]] for pt in p_coords ]]))
    end

    g_final = SimpleGraph(n_final)
    v_edges = []
    for i in 1:n_final, j in (i+1):n_final
        if LibGEOS.touches(lg_polys[i], lg_polys[j])
            add_edge!(g_final, i, j)
            push!(v_edges, (final_centroids[i], final_centroids[j]))
        end
    end
    g_final = ensure_connected!(g_final, final_centroids)

    return (
        centroids = final_centroids, 
        adjacency_edges = v_edges, 
        graph = g_final, 
        polygons = polys_output, 
        hull_coords = hull_coords_output
    )
end


 



function get_polygon_area(poly_coords)
    # Calculates the area of a polygon using the Shoelace formula.
    valid_pts = [p for p in poly_coords if !isnan(p[1])]
    if length(valid_pts) > 1 && valid_pts[1] == valid_pts[end]
        pop!(valid_pts)
    end
    if length(valid_pts) < 3 return 0.0 end
    x = [p[1] for p in valid_pts]
    y = [p[2] for p in valid_pts]
    return 0.5 * abs(dot(x, circshift(y, 1)) - dot(y, circshift(x, 1)))
end

 
function get_coords_from_geom(geom)
    """
    Synopsis: Extracts coordinates from various LibGEOS geometry types.
    Inputs:
    - geom: A LibGEOS geometry object.
    Outputs:
    - A vector of (x, y) coordinates.
    """

    coords = Tuple{Float64, Float64}[]
    local type_id = -1
    try
        type_id = LibGEOS.geomTypeId(geom)
        if type_id == LibGEOS.GEOS_POINT
             # Access coordinate sequence directly for point types
             seq = LibGEOS.getCoordSeq(geom)
             push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
             return coords
        elseif type_id == LibGEOS.GEOS_POLYGON
            ring = LibGEOS.exteriorRing(geom)
            n = LibGEOS.numPoints(ring)
            for i in 1:n
                p = LibGEOS.getPoint(ring, i)
                seq = LibGEOS.getCoordSeq(p)
                push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
            end
        elseif type_id == LibGEOS.GEOS_MULTIPOLYGON
            for i in 1:LibGEOS.numGeometries(geom)
                poly = LibGEOS.getGeometryN(geom, i)
                ring = LibGEOS.exteriorRing(poly)
                n = LibGEOS.numPoints(ring)
                for j in 1:n
                    p = LibGEOS.getPoint(ring, j)
                    seq = LibGEOS.getCoordSeq(p)
                    push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
                end
                if i < LibGEOS.numGeometries(geom); push!(coords, (NaN, NaN)); end
            end
        elseif type_id in [LibGEOS.GEOS_LINESTRING, LibGEOS.GEOS_LINEARRING]
            n = LibGEOS.numPoints(geom)
            for i in 1:n
                p = LibGEOS.getPoint(geom, i)
                seq = LibGEOS.getCoordSeq(p)
                push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
            end
        end
    catch e
        @warn "Coordinate extraction failed for type $type_id: $e"
    end
    return coords
end




function get_voronoi_polygons_and_edges(centroids, hull_geom, tol=1e-7)
    """
    Synopsis: Generates clipped Voronoi polygons with robust adjacency detection.
    Uses a small buffer fallback to handle floating-point misalignment in LibGEOS.
    """
    n_c = length(centroids)
    if n_c == 0
        return [], []
    elseif n_c == 1
        return [get_coords_from_geom(hull_geom)], []
    elseif n_c == 2
        # Standard 2-point bisection logic
        p1, p2 = centroids[1], centroids[2]
        mid = ((p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)
        dx, dy = p2[1] - p1[1], p2[2] - p1[2]
        px, py = -dy, dx
        L = 1e7
        pt1 = (mid[1] + L*px, mid[2] + L*py)
        pt2 = (mid[1] - L*px, mid[2] - L*py)
        side1_pts = [pt1, pt2, (pt2[1] - L*dx, pt2[2] - L*dy), (pt1[1] - L*dx, pt1[2] - L*dy), pt1]
        poly1_box = LibGEOS.Polygon([[[p[1], p[2]] for p in side1_pts]])
        side2_pts = [pt1, pt2, (pt2[1] + L*dx, pt2[2] + L*dy), (pt1[1] + L*dx, pt1[2] + L*dy), pt1]
        poly2_box = LibGEOS.Polygon([[[p[1], p[2]] for p in side2_pts]])
        res1 = LibGEOS.intersection(hull_geom, poly1_box)
        res2 = LibGEOS.intersection(hull_geom, poly2_box)
        return [get_coords_from_geom(res1), get_coords_from_geom(res2)], [(p1, p2)]
    end

    # 3+ points logic
    pts_dt = [(Float64(c[1]), Float64(c[2])) for c in centroids]
    tri = triangulate(pts_dt)
    hull_coords = get_coords_from_geom(hull_geom)
    xs = [p[1] for p in hull_coords if !isnan(p[1])]
    ys = [p[2] for p in hull_coords if !isnan(p[2])]
    if isempty(xs) || isempty(ys) return [Tuple{Float64, Float64}[] for _ in 1:length(centroids)], [] end
    
    bbox = (minimum(xs), maximum(xs), minimum(ys), maximum(ys))
    vorn = voronoi(tri)
    final_coords = [Tuple{Float64, Float64}[] for _ in 1:length(centroids)]
    valid_geoms = Dict{Int, Any}()

    for i in each_generator(vorn)
        if i < 1 || i > length(centroids) continue end
        vertices = get_polygon_coordinates(vorn, i, bbox)
        if !isempty(vertices)
            poly_pts = [[v[1], v[2]] for v in vertices]
            if poly_pts[1] != poly_pts[end] push!(poly_pts, poly_pts[1]) end
            try
                lg_poly = LibGEOS.Polygon([poly_pts])
                clipped = LibGEOS.intersection(lg_poly, hull_geom)
                if !LibGEOS.isEmpty(clipped) && LibGEOS.geomTypeId(clipped) in [LibGEOS.GEOS_POLYGON, LibGEOS.GEOS_MULTIPOLYGON]
                    final_coords[i] = get_coords_from_geom(clipped)
                    valid_geoms[i] = clipped
                end
            catch e end
        end
    end

    v_edges = []
    active_ids = sort(collect(keys(valid_geoms)))
    for idx in 1:length(active_ids)
        i = active_ids[idx]
        g1 = valid_geoms[i]
        for jdx in idx+1:length(active_ids)
            j = active_ids[jdx]
            g2 = valid_geoms[j]
            # Primary check: direct contact
            if LibGEOS.touches(g1, g2)
                push!(v_edges, (centroids[i], centroids[j]))
            else
                # Fallback check: microscopic overlap/buffer
                g1_b = LibGEOS.buffer(g1, tol)
                if LibGEOS.intersects(g1_b, g2)
                    push!(v_edges, (centroids[i], centroids[j]))
                end
            end
        end
    end
    return final_coords, v_edges
end

function check_connectivity(g)
    """
    Synopsis: Evaluates the connectivity of a spatial graph.
    Inputs:
    - g: A SimpleGraph.
    Outputs:
    - NamedTuple showing connection status and components.
    """
    comps = connected_components(g)
    return (is_connected = length(comps) == 1, n_components = length(comps), components = comps)
end


function ensure_connected!(g, centroids)
    # Ensures the spatial graph is connected by adding edges between the nearest 
    # components based on the provided centroid coordinates.
    while !is_connected(g)
        comps = connected_components(g)
        best_dist = Inf
        best_pair = (0, 0)
        
        # Find the two closest nodes belonging to different components
        for i in 1:length(comps), j in (i+1):length(comps)
            for u in comps[i], v in comps[j]
                d = sum((centroids[u] .- centroids[v]).^2)
                if d < best_dist
                    best_dist = d
                    best_pair = (u, v)
                end
            end
        end
        
        if best_pair != (0, 0)
            add_edge!(g, best_pair[1], best_pair[2])
        else
            break
        end
    end
    return g
end


 
function plot_spatial_graph(au; title="Spatial Partitioning", domain_boundary=nothing)
    # 1. Base Plot with Polygons
    plt = Plots.plot(aspect_ratio=:equal, title=title, legend=false)
    
    # Plot Polygons
    for poly_coords in au.polygons
        if length(poly_coords) > 2
            px = [p[1] for p in poly_coords if !isnan(p[1])]
            py = [p[2] for p in poly_coords if !isnan(p[2])]
            if !isempty(px) && (px[1], py[1]) != (px[end], py[end])
                push!(px, px[1]); push!(py, py[1])
            end
            Plots.plot!(plt, px, py, seriestype=:shape, fillalpha=0.1, linecolor=:black, lw=0.5)
        end
    end

    # 2. Plot Adjacency Graph Edges (Using au.centroids directly for nodes)
    for edge in Graphs.edges(au.graph)
        u, v = src(edge), dst(edge)
        p1, p2 = au.centroids[u], au.centroids[v]
        Plots.plot!(plt, [p1[1], p2[1]], [p1[2], p2[2]], color=:red, lw=1.5, alpha=0.6)
    end

    # 3. Plot Centroids and Raw Points
    Plots.scatter!(plt, [p[1] for p in au.pts], [p[2] for p in au.pts], 
        markersize=1, color=:gray, alpha=0.3, label="Points")
    Plots.scatter!(plt, [c[1] for c in au.centroids], [c[2] for c in au.centroids], 
        markersize=4, color=:blue, markerstrokecolor=:white, label="Centroids")

    if !isnothing(domain_boundary)
        bx = [p[1] for p in domain_boundary if !isnan(p[1])]
        by = [p[2] for p in domain_boundary if !isnan(p[2])]
        Plots.plot!(plt, bx, by, color=:black, lw=2, ls=:dash)
    end

    return plt
end




function adjacency_matrix_to_nb( W )
    nau = size(W)[1]
    # W = LowerTriangular(W)  # using LinearAlgebra
    nb = [Int[] for _ in 1:nau]
    Threads.@threads for i in 1:nau
        nb[i] = findall( isone, W[i,:] )
    end
    return nb
end


function nb_to_adjacency_matrix( nb )
    nau = Integer( length( unique( reduce(vcat, nb) )) )
    W = zeros( Int8, nau, nau )
    Threads.@threads for i in 1:nau
        for j in 1:length( nb[i] )
            k = nb[i][j]
            W[i, k] = 1
        end
    end
    return(W)
end


function nodes( adj )
    nau = length(adj)
    N_edges = Integer( length( reduce(vcat, adj) )/2 )
    node1 =  fill(0, N_edges); 
    node2 =  fill(0, N_edges); 
    i_edge = 0;
    for i in 1:nau
        u = adj[i]
        num = length(u)
        for j in 1:num
            k = u[j]
            if i < k
                i_edge = i_edge + 1;
                node1[i_edge] = i;
                node2[i_edge] = k;
            end
        end
    end

    e = Edge.(node1, node2)
    g = Graph(e)
    W = Graphs.adjacency_matrix(g)
    
    # D = diagm(vec( sum(W, dims=2) ))
    scalefactor = scaling_factor_bym2(W)

    return node1, node2, scalefactor
end




function scaling_factor_bym2( adjacency_mat )
    # re-scaling variance using Reibler's solution and 
    # Buerkner's implementation: https://codesti.com/issue/paul-buerkner/brms/1241)  
    # Compute the diagonal elements of the covariance matrix subject to the 
    # constraint that the entries of the ICAR sum to zero.
    # See the inla.qinv function help for further details.
    # Q_inv = inla.qinv(Q, constr=list(A = matrix(1,1,nbs$N),e=0))  # sum to zero constraint
    # Compute the geometric mean of the variances, which are on the diagonal of Q.inv
    # scaling_factor = exp(mean(log(diag(Q_inv))))
    N = size(adjacency_mat)[1]
    asum = vec( sum(adjacency_mat, dims=2)) 
    asum = float(asum) + N .* max.(asum) .* sqrt(1e-15)  # small perturbation
    Q = Diagonal(asum) - adjacency_mat
    A = ones(N)   # constraint (sum to zero)
    S = Q \ Diagonal(A)  # == inv(Q)
    V = S * A
    S = S - V * inv(A' * V) * V'
    # equivalent form as inv is scalar
    # S = S - V / (A' * V) * V'
    scale_factor = exp(mean(log.(diag(S))))
    return scale_factor
 
end



function scaling_factor_bym2(node1, node2, groups=ones(length(node1))) 
    ## calculate the scale factor for each of k connected group of nodes, 
    ## copied from the scale_c function from M. Morris
    gr = unique( groups )
    n_groups = length(gr)
    scale_factor = ones(n_groups)
    Threads.@threads for j in 1:n_groups 
      k = findall( x -> x==j, groups)
      if length(k) > 1 
        e = Edge.(node1[k], node2[k])
        g = Graph(e)
        adjacency_mat = adjacency_matrix(g)
        scale_factor[j] = scaling_factor_bym2( adjacency_mat )
      end
    end
    return scale_factor
end
  

 

function icar_form(theta, phi, sigma, rho)
    # https://sites.stat.columbia.edu/gelman/research/published/bym_article_SSTEproof.pdf
    # Reibler parameterization: https://pubmed.ncbi.nlm.nih.gov/27566770/
    # https://www.jstatsoft.org/index.php/jss/article/view/v063c01/841
    sigma .* ( sqrt.(1 .- rho) .* theta .+ sqrt.(rho ./ scaling_factor) .* phi )  
end
   

function sample_gaussian_process( ; GPmethod="cholesky", returntype="default",
    fkernal=nothing, kerneltype="default", kvar=nothing, kscale=nothing, gpc=GPC(),
    Yobs, Xobs, Xinducing=nothing, lambda=0.0001 )
    
    if isnothing(fkernal)
        if kerneltype=="default" || kerneltype=="squared_exponential"
            fkernal = kvar * SqExponentialKernel() ∘ ScaleTransform( kscale) # ∘ ARDTransform(α)
        end
        if kerneltype=="matern32"
            fkernal = kvar * Matern32Kernel() ∘ ScaleTransform( kscale) # ∘ ARDTransform(α)
        end
    end


    if GPmethod=="textbook"
        # mean process at predictons Xobs
        Ko = kernelmatrix( fkernal, vec(Xobs) ) 
        Kcommon = inv(Ko + lambda*I)   # Note already inversed taken

        if !isnothing(Xinducing)

            Ki = kernelmatrix( fkernal, vec(Xinducing) )   
            Kio = kernelmatrix( fkernal, vec(Xinducing), vec(Xobs) )   # transfer to inducing points
            Yinducing_mean_process = Kio * Kcommon * Yobs   # mean process at inducing points
            # covariance at predictions Covp:
            # Covp = Ki - Kio * inv(Ko + lambda*I ) * Kio' 
            Covi = Symmetric( Ki - Kio * Kcommon * Kio'  + lambda*I ) # note Ccommon is already inverted 
            MVNi = MvNormal( Yinducing_mean_process, Covi )

            Yinducing_sample  = rand( MVNi )
            Li =  cholesky(Symmetric( Ki + lambda*I)).L   # cholesky on inducing locations  

            Yobs_mean_process =  Kio' * ( Li' \ (Li \ Yinducing_mean_process  ) )  # back to original locations
            Covo = Symmetric(cov(kernelmatrix( fkernal,  vec(Xobs) )) + lambda*I)
            MVN = MvNormal(Yobs_mean_process, Covo)  # of observations

            if returntype=="fcovariance"  
                return MVN
            end

            Yobs_sample =  Kio' * ( Li' \ (Li \ Yinducing_sample  ) )  # back to original locations

            if returntype=="sample"
                return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
            end

            LogLik = logpdf(MVN, Yobs)

            if returntype=="sample_loglik"
                return ( Yobs_sample=Yobs_sample, loglik=LogLik, GPmethod=GPmethod)
            end

            return (MVN=MVN, MVNi=MVNi, Li=Li, loglik=LogLik,
                Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)

        else
             
            mean_process = Ko * Kcommon * Yobs   # mean process            
            MVN = MvNormal(mean_process, Ko + lambda*I  ) # lambda*I creates a diagonal matrix

            if returntype=="fcovariance"  
                return MVN  # of observations
            end

            Yobs_sample = rand( MVN ) # sample
            
            if returntype=="sample"
                return ( Yobs_sample=Yobs_sample, GPmethod=GPmethod )
            end
            
            LogLik = logpdf(MVN,Yobs)

            if returntype=="sample_loglik"
                return ( Yobs_sample=Yobs_sample, loglik=LogLik, GPmethod=GPmethod)
            end

            return ( MVN=MVN, loglik=LogLik, Yobs_sample=Yobs_sample, GPmethod=GPmethod)

        end 
 
    end

    if GPmethod=="cholesky"
        # this avoids inversion of the big covariance and re-uses cholesky factors 
        if !isnothing(Xinducing)
            Ko = kernelmatrix( fkernal, vec(Xobs) ) 

            Ki = kernelmatrix( fkernal, vec(Xinducing) ) 
            Kio = kernelmatrix( fkernal, vec(Xinducing), vec(Xobs) ) # transfer to inducing points
            Lo = cholesky(Symmetric( Ko + lambda*I)).L 
            Li = cholesky(Symmetric( Ki + lambda*I)).L   # cholesky on inducing locations  
            Yinducing_mean_process  = Kio * ( Lo' \ (Lo \ Yobs ) )  # == mean_process mean latent process

            Covi = Symmetric( cov(Ki) + lambda*I)  
            MVN = MvNormal( Yinducing_mean_process, Covi )

            if returntype=="fcovariance" 
                return MVN
            end

            Yobs_mean_process = Kio' * ( Li' \ (Li \ Yinducing_mean_process ))  # mean process from inducing pts
            Yinducing_sample  = Yinducing_mean_process + Li * rand(Normal(0, 1), size(Li,1))   # faster sampling without covariance
            Yobs_sample = Yobs_mean_process + Lo * rand(Normal(0, 1), size(Lo,2)) # error process 

            if returntype=="sample"
                return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
            end

            LogLik = logpdf(MVN, Yinducing_mean_process)
            
            if returntype=="sample_loglik"
                return ( Yobs_sample=Yobs_sample, loglik=LogLik, GPmethod=GPmethod)
            end
            
            return (MVN=MVN, Li=Li, Lo=Lo, loglik=LogLik,
                    Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)

        else

            Ko = kernelmatrix( fkernal, vec(Xobs) ) 
            Lo = cholesky(Symmetric( Ko + lambda*I)).L 
            Yobs_mean_process = Ko' * ( Lo' \ (Lo \ Yobs ))  # mean process from inducing pts
            
            Covo = Symmetric( cov(Ko) + lambda*I)  
            MVN = MvNormal( Yobs_mean_process, Covo ) # of observations

            if returntype=="fcovariance" 
                return MVN
            end

            Yobs_sample = Yobs_mean_process + Lo * rand(Normal(0, 1), size(Lo,2)) # error process 

            if returntype=="sample"
                return (Yobs_sample=Yobs_sample, GPmethod=GPmethod)
            end

            LogLik = logpdf(MVN, Yobs)
       
            if returntype=="sample_loglik"
                return (Yobs_sample=Yobs_sample, loglik=LogLik,  GPmethod=GPmethod )
            end

            return (MVN=MVN, Lo=Lo, loglik=LogLik, Yobs_sample=Yobs_sample, GPmethod=GPmethod)

        end
    end
 

    if GPmethod=="GPexact"

        fgp = atomic(AbstractGPs.GP(fkernal), gpc)
        fobs = fgp(Xobs, lambda)

        if returntype=="fcovariance"
            return fobs
        end 

        fposterior = posterior(fobs, Yobs) 
        
        if returntype=="posterior"
            return fposterior
        end

        Yobs_sample =  rand(fposterior(Xobs, lambda) )   

        if returntype=="sample"
            return ( Yobs_sample=Yobs_sample, GPmethod=GPmethod)
        end

        LogLik = logpdf(fobs, Yobs)
       
        if returntype=="sample_loglik"
            return (Yobs_sample=Yobs_sample, loglik=LogLik,  GPmethod=GPmethod )
        end

        return ( fgp=fgp, fobs=fobs, fposterior=fposterior, Yobs_sample=Yobs_sample, loglik=LogLik, GPmethod=GPmethod)
    end
 
    if GPmethod=="GPsparse"
        fgp = atomic(AbstractGPs.GP(fkernal), gpc)
        fobs = fgp( Xobs, lambda )
        finducing = fgp( Xinducing, lambda ) 
        fsparse = SparseFiniteGP(fobs, finducing)

        if returntype=="fcovariance"
            return fsparse
        end 

        fposterior = posterior(fsparse, Yobs)

        if returntype=="posterior"
            return fposterior
        end
        
        Yobs_sample =  rand(fposterior(Xobs, lambda) )  
        Yinducing_sample =   rand(fposterior(Xinducing, lambda))

        if returntype=="sample"
            return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
        end

        LogLik = logpdf(fsparse, Yobs)
       
        if returntype=="sample_loglik"
            return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, loglik=LogLik,  GPmethod=GPmethod )
        end

        return ( fgp=fgp, fobs=fobs, finducing=finducing, fsparse=fsparse, fposterior=fposterior, loglik=LogLik, 
                Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)

    end

    if GPmethod=="GPvfe" # Variational Free Energy
        fgp = atomic(AbstractGPs.GP(fkernal), gpc)
        fobs = fgp( Xobs, lambda )
        finducing = fgp(Xinducing, lambda )
        fsparse = VFE( finducing )

        if returntype=="fcovariance"
            return fsparse
        end 
        
        fposterior = posterior(fsparse, fobs, Yobs)  # Distribution is MvNormal  

        if returntype=="posterior"
            return fposterior
        end
        
        Yobs_sample =  rand(fposterior(Xobs, lambda) )  
        Yinducing_sample =   rand(fposterior(Xinducing, lambda))

        if returntype=="sample"
            return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
        end
        
        LogLik = AbstractGPs.elbo(fsparse, fobs, Yobs)  # to a constant
      
        if returntype=="sample_loglik"
            return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, loglik=LogLik,  GPmethod=GPmethod )
        end

        return ( fgp=fgp, fobs=fobs, finducing=finducing, fsparse=fsparse, fposterior=fposterior, loglik=LogLik, 
                Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
    end
      
end
 
 


function turing_glm_icar_summary( method="mcmc"; 
    GPmethod="cholesky", family="poisson", 
    Y=nothing, YG=nothing,  
    msol=nothing, model=nothing,
    X=nothing, G=nothing, Gp=nothing, nInducing=nothing, log_offset=nothing, good=nothing,
    scaling_factor=nothing, n_sample=nothing, nAU=nothing, auid=nothing, tuid=nothing, 
    kerneltype="squared_exponential"
)
 
    fixed_effects = nothing
    sp_re_structured = nothing
    sp_re_unstructured = nothing
    Gymu = nothing

    # Main.DEBUG = Ymu
  
    if !isnothing(good)
                     
        if !isnothing(Y)
            Y = Y[good] 
        end
        
        if !isnothing(YG)
            YG = YG[good] 
        end
        if !isnothing(X)
            X = X[good,:] 
        end
        if !isnothing(G)
            G = G[good,:] 
        end
        if !isnothing(log_offset)
            log_offset = log_offset[good] 
        end
        if !isnothing(auid)
            auid = auid[good] 
        end
        if !isnothing(tuid)
            tuid = tuid[good] 
        end

    end

    if !isnothing(X)
        nX = size(X,2)
        nData = size(X,1)
        fixed_effects = zeros(nX, n_sample)
    end

    if method=="mcmc"
        nchains = size(msol)[3]
        nsims = size(msol)[1]
    end

    if method=="variational_inference"
        res = rand(msol, n_sample)
        nsims = size(res)[2]
    end

    if method=="optim"
        res = hcat( vec(msol.values) )
        nsims = size(res)[2]
    end

    if isnothing(n_sample)
        n_sample = nsims         # do all
    end

    if !isnothing(G)
        nG = size(G)[2]
        if isnothing(X) # in case no fixed effects
            nData = size(G,1)
        end
        if isnothing(nInducing)
            nInducing = size(G,2)
        end
        Gymu =zeros(nInducing, nG, n_sample)
    end

    if !isnothing(auid)
        if isnothing(nAU)
            nAU = length(auid)
        end
        sp_re_structured = zeros(nAU, n_sample) 
        sp_re_unstructured = zeros(nAU, n_sample) 
    end


    ntries_mult=2
    ntries = 0
    z = 0

    Ypred = zeros(nData, n_sample) 
    
    if method=="mcmc"

        while z <= n_sample 
            ntries += 1
            ntries > ntries_mult * n_sample && break 
            z >= n_sample && break

            j = rand(1:nsims)  # nsims
            l = rand(1:nchains) # nchains

            Ymu = zeros( nData ) 

            if !isnothing(X)
                # fixed effects
                # beta = Array(msol[:, turingindex( model, :beta), :] )
                beta  = [ msol[j, Symbol("beta[$k]"), l]  for k in 1:nX]
                Ymu += X * beta
                # Main.DEBUG = beta
            end

            if !isnothing(auid)
                theta = [ msol[j, Symbol("theta[$k]"), l] for k in 1:nAU]
                phi   = [ msol[j, Symbol("phi[$k]"), l]   for k in 1:nAU]
                sigma = msol[j, Symbol("sigma"), l] 
                rho   = msol[j, Symbol("rho"), l] 
                sp_re_besag = sigma .* (sqrt.(rho ./ scaling_factor) .* phi )  
                sp_re_iid   = sigma .* (sqrt.(1 .- rho) .* theta )
                sp_re = sp_re_besag + sp_re_iid  
                Ymu += sp_re[auid] 
            end
            
            if !isnothing(log_offset)
                Ymu .+= log_offset 
            end

            if !isnothing(G)

                # gaussian process for covariates G
                kernel_var = [ msol[j, Symbol("kernel_var[$k]"), l]  for k in 1:nG] 
                kernel_scale = [ msol[j, Symbol("kernel_scale[$k]"), l]  for k in 1:nG] 
                
                if any( occursin.("l2reg", String.(names(msol))) )
                    l2reg = [ msol[j, Symbol("l2reg[$k]"), l]  for k in 1:nG]  
                else
                    l2reg = fill(1.0e-4, nG)                    
                end
                
                Gymu_s = zeros( nInducing, nG)
                YGcurr = YG - Ymu
                for i in 1:nG
                    # Kfn = fkernal( kernfunctype, (kernel_var[i], kernel_scale[i]) ) 
                    ys = sample_gaussian_process( GPmethod=GPmethod, returntype="sample",
                        kerneltype=kerneltype, kvar=kernel_var[i], kscale=kernel_scale[i],
                        Yobs=YGcurr, Xobs=G[:,i], Xinducing=Gp[:,i], lambda=l2reg[i], 
                    )
                    
                    # Main.DEBUG = ys
                    Ymu += ys.Yobs_sample
                    Gymu_s[:,i] = ys.Yinducing_sample 
                end
            end
  
            z += 1

            if family=="poisson"
                ineg = findall(x->x<0, Ymu)
                if length(ineg)>0
                    Ymu[ineg] .= 0.0
                end
                Ypred[:,z] = rand.(LogPoisson.(Ymu));   
            elseif family=="bernoulli"
                Ypred[:,z] = rand.(Bernoulli.( logistic.(Ymu) ) ) 
            elseif family=="gaussian"
                Ysd = msol[j, Symbol("Ysd"), l] 
                Ypred[:,z] = rand.(Normal.( Ymu, Ysd ) ) 
            end

            if !isnothing(X)
                fixed_effects[:,z] = beta
            end

            if !isnothing(auid)
                if !isnothing(sp_re_structured)
                    sp_re_structured[:,z]   = sp_re_besag
                    sp_re_unstructured[:,z] = sp_re_iid
                end
            end

            if !isnothing(G)
                Gymu[:,:,z] = Gymu_s
            end

        end  # while
    
        if z < n_sample 
            @warn  "Insufficient number of solutions" 
        end
        res = Array(msol)
    end


    if method=="variational_inference"
        # variational inference method

        # this in case some samples provide failed predictions (e.g., not PD, etc)
        while z <= n_sample 
            ntries += 1
            ntries > ntries_mult * n_sample && break 
            z >= n_sample && break

            l = rand(1:nsims) # nchains

            Ymu = zeros( nData ) 

            if !isnothing(X)
                # fixed effects
                beta  = [ msol[j, Symbol("beta[$k]"), l]  for k in 1:nX]
                beta = res[ turingindex( model, :beta ), l ]
                Ymu += X * beta
            end

            if !isnothing(auid)
                theta = res[ turingindex( model, :theta ), l ]
                phi   = res[ turingindex( model, :phi ), l ]
                sigma = res[ turingindex( model, :sigma ), l ] 
                rho   = res[ turingindex( model, :rho ), l ] 
                sp_re_besag = sigma .* (sqrt.(rho ./ scaling_factor) .* phi )  
                sp_re_iid   = sigma .* ( sqrt.(1 .- rho) .* theta )
                sp_re = sp_re_besag + sp_re_iid  
                Ymu += sp_re[auid] 
            end
            
            if !isnothing(log_offset)
                Ymu .+= log_offset 
            end

            if !isnothing(G)
                # gaussian process for covariates G

                kernel_var = res[ turingindex( model, :kernel_var )] 
                kernel_scale = res[ turingindex( model, :kernel_scale )]
                if any( occursin.("l2reg", String.(names(msol))) )
                    l2reg = res[ turingindex( model, :l2reg )]
                else
                    l2reg = fill(1.0e-4, nG)                
                end
                Gymu_s = zeros( nInducing, nG)
                YGcurr = YG - Ymu
                for i in 1:nG
                    # Kfn = fkernal( kernfunctype, (kernel_var[i], kernel_scale[i]) ) 
                    ys = sample_gaussian_process( GPmethod=GPmethod, returntype="sample",
                        kerneltype=kerneltype, kvar=kernel_var[i], kscale=kernel_scale[i],
                        Yobs=YGcurr, Xobs=G[:,i], Xinducing=Gp[:,i], lambda=l2reg[i], 
                    ) 
                    # Main.DEBUG = ys
                    Ymu += ys.Yobs_sample
                    Gymu_s[:,i] = ys.Yinducing_sample 
                end
            end
        
    
            z += 1

            if family=="poisson"
                Ypred[:,z] = rand.(LogPoisson.(Ymu));   
            elseif family=="bernoulli"
                Ypred[:,z] = rand.(Bernoulli.( logistic.(Ymu) ) ) 
            elseif family=="gaussian"
                Ysd = res[j, Symbol("Ysd"), l] 
                Ypred[:,z] = rand.(Normal.( Ymu, Ysd ) ) 
            end
    
            if !isnothing(X)
                fixed_effects[:,z] = beta
            end

            if !isnothing(auid)
                sp_re_structured[:,z]   = sp_re_besag
                sp_re_unstructured[:,z] = sp_re_iid
            end

            if !isnothing(G)
                Gymu[:,:,z] = Gymu_s
            end
            
        end  # while
    
        if z < n_sample 
            @warn  "Insufficient number of solutions" 
        end

    end

    if method =="optim"
        # optim method 

        # this in case some samples provide failed predictions (e.g., not PD, etc)
        while z <= n_sample 
            ntries += 1
            ntries > ntries_mult * n_sample && break 
            z >= n_sample && break

            l = rand(1:nsims) # nchains

            Ymu = zeros( nData ) 

            if !isnothing(X)
                # fixed effects
                beta = res[ turingindex( model, :beta ), l ]
                Ymu += X * beta
            end

            if !isnothing(auid)
                theta = res[ turingindex( model, :theta ), l ]
                phi   = res[ turingindex( model, :phi ), l ]
                sigma = res[ turingindex( model, :sigma ), l ] 
                rho   = res[ turingindex( model, :rho ), l ] 
                sp_re_besag = sigma .* (sqrt.(rho ./ scaling_factor) .* phi )  
                sp_re_iid   = sigma .* ( sqrt.(1 .- rho) .* theta )
                sp_re = sp_re_besag + sp_re_iid  
                Ymu += sp_re[auid] 
            end
            
            if !isnothing(log_offset)
                Ymu .+= log_offset 
            end

            if !isnothing(G)
                # gaussian process for covariates G

                kernel_var = res[ turingindex( model, :kernel_var )] 
                kernel_scale = res[ turingindex( model, :kernel_scale )]

                if any( occursin.("l2reg", String.(names(msol.values)[1] )) )
                    l2reg = res[ turingindex( model, :l2reg )]
                else
                    l2reg = fill(1.0e-4, nG)             
                end
                Gymu_s = zeros( nInducing, nG)
                YGcurr = YG - Ymu
                for i in 1:nG
                    # Kfn = fkernal( kernfunctype, (kernel_var[i], kernel_scale[i]) ) 
                    ys = sample_gaussian_process( GPmethod=GPmethod, returntype="sample",
                        kerneltype=kerneltype, kvar=kernel_var[i], kscale=kernel_scale[i],
                        Yobs=YGcurr, Xobs=G[:,i], Xinducing=Gp[:,i], lambda=l2reg[i], 
                    )
                    # Main.DEBUG = ys
                    Ymu += ys.Yobs_sample
                    Gymu_s[:,i] = ys.Yinducing_sample 
                end

            end    
            
            z += 1

            if family=="poisson"
                Ypred[:,z] = rand.(LogPoisson.(Ymu));   
            elseif family=="bernoulli"
                Ypred[:,z] = rand.(Bernoulli.( logistic.(Ymu) ) ) 
            elseif family=="gaussian"
                Ysd = res[j, Symbol("Ysd"), l] 
                Ypred[:,z] = rand.(Normal.( Ymu, Ysd ) ) 
            end
    
            if !isnothing(X)
                fixed_effects[:,z] = beta
            end

            if !isnothing(auid)
                sp_re_structured[:,z]   = sp_re_besag
                sp_re_unstructured[:,z] = sp_re_iid
            end

            if !isnothing(G)
                Gymu[:,:,z] = Gymu_s
            end
            
        end  # while
    
        if z < n_sample 
            @warn  "Insufficient number of solutions" 
        end
    end

    return ( 
        Ypred=Ypred, 
        fixed_effects=fixed_effects,
        sp_re_unstructured=sp_re_unstructured, 
        sp_re_structured=sp_re_structured, 
        res=res, 
        Gymu=Gymu
    )

end

 

function plot_variational_marginals(z, sym2range)
    # copied straight from https://turinglang.org/docs/tutorials/variational-inference/
    ps = []

    for (i, sym) in enumerate(keys(sym2range))
        indices = union(sym2range[sym]...)  # <= array of ranges
        if sum(length.(indices)) > 1
            k = 1
            for r in indices
                p = density(
                    z[r, :];
                    title="$(sym)[$k]",
                    titlefontsize=10,
                    label="",
                    ylabel="Density",
                    margin=1.5mm,
                )
                push!(ps, p)
                k += 1
            end
        else
            p = density(
                z[first(indices), :];
                title="$(sym)",
                titlefontsize=10,
                label="",
                ylabel="Density",
                margin=1.5mm,
            )
            push!(ps, p)
        end
    end

    return plot(ps...; layout=(length(ps), 1), size=(500, 2000), margin=4.0mm)
end


function fkernal( kernfunctype="squared_exp", params=nothing )

    if kernfunctype=="squared_exp"
        out = params[1] * SqExponentialKernel() ∘ ScaleTransform(params[2])  
    end

    if kernfunctype=="matern12"
        out = params[1] * Matern12Kernel() ∘ ScaleTransform(params[2])  
    end

    if kernfunctype=="matern32"
        out = params[1] * Matern32Kernel() ∘ ScaleTransform(params[2])  
    end

    if kernfunctype=="matern52"
        out = params[1] * Matern52Kernel() ∘ ScaleTransform(params[2])  
    end

    # ∘ ARDTransform(α)

    return out
end


sekernel2(v, s) = v * SqExponentialKernel() ∘ ScaleTransform(s) # ∘ ARDTransform(a);

sekernel(v, s) = v * SqExponentialKernel() ∘ ScaleTransform(s) # ∘ ARDTransform(a);

    
    
function assign_time_units(t_obs; method="regular", N_time=nothing, N_season=12)

    if method=="regular"

        tint = Int.(floor.(t_obs))
        t0, t1 = minimum(tint), maximum(tint)
        t_idx = tint .- t0 .+ 1
        t_vals = collect(t0:t1) .- t0 .+ 1
        t_yr = collect(t0:t1)
        t_brks = (t_yr, t1+1)
        t_mids = t_yr .+ 0.5

        if !isnothing(N_time)
            t_disc = discretize_data( t_yr, N_cat=N_time, method="regular" )
            t_idx = t_disc.idx
            t_brks = t_disc.brks
            t_mids = t_disc.mids
        end

        u_obs = t_obs - tint
        u_disc = discretize_data( u_obs, N_cat=N_season, method="regular" )  # seasonality discretized

        return (
            t_obs=t_obs, t_idx = t_idx, t0=t0, t1=t1, t_vals, t_yr=t_yr, tn=length(t_vals), 
            t_mids=t_mids, t_brks=t_brks,
            u_obs=u_obs, u_idx=u_disc.idx, 
            u_brks=u_disc.brks, u_mids=u_disc.mids, u_vals=collect(1:N_season) )
    end

end


function discretize_data(X; method="quantile", N_cat=9, brks=nothing, probs=nothing, dx=nothing, minv = 0, maxv=1)
    


    if method=="quantile" 
        # simpler solutions
        probs = isnothing(probs) ? collect(range(0.0, stop=1.0, length=N_cat + 1)) : probs
        brks = isnothing(brks) ? quantile(X, probs) : brks
        mids = brks[1:N_cat] + diff(brks) 
        idx = map(x -> clamp(searchsortedfirst(brks, x) - 1, 1, N_cat), X)
        return ( idx=idx, brks=brks, mids=mids, probs )

    elseif method == "regular"
        dx = isnothing(dx) ? (maxv - minv) / N_cat : dx
        probs = nothing
        brks = collect(minv:dx:maxv)
        mids = brks[1:N_cat] + diff(brks) 
        idx = map(x -> clamp(searchsortedfirst(brks, x) - 1, 1, N_cat), X)
        return ( idx=idx, brks=brks, mids=mids, probs, dx=dx )
    
    elseif method=="regular_resolution"    

        xd = round.(Int, X ./ dx ) .* dx   # resolution to  units of dx
        brks = collect( minimum(xd):dx:maximum(xd) + dx  ) 
        mids = midpoints(brks)
        N_cats = length(mids)
        
        xd_cut = cut(X, brks, extend=true)  # from CategoricalArrays
        xi = levelcode.(xd_cut)  # integer index
        return xd, xi, mids, N_cats, dx

    elseif method=="quantile_resolution"
    
        brks = quantile(X, range(0, 1, length=N_cats+1))
        mids = midpoints(brks)
        xd_cut = cut(X, brks, extend=true)  # from CategoricalArrays
        xi = levelcode.(xd_cut)  # integer index
        dx = diff(mids)[1]
        xd = mids[xi] 
        return xd, xi, mids, N_cats, dx

    end


end
 
 

function assign_covariate_levels( X; N_cat=9, brks=nothing, probs=nothing )
    X = Array(X)
    U = [ discretize_data( X[:,i], N_cat=N_cat, brks=brks, probs=probs ) for i in 1:size(X, 2) ]
    V = hcat([t[:idx] for t in  U]...)  # as matrix
    B = hcat([t[:brks] for t in  U]...)
    M = hcat([t[:mids] for t in  U]...)
    P = hcat([t[:probs] for t in  U]...)
    return( idx=V, brks=B, mids=M, probs=P )
end



function estimate_local_kde_with_extrapolation(pts, t_idx, target_ts; grid_res=600, sd_extension_factor=0.25)
    """
    Synopsis: Estimates 2D KDE for a specific time slice with extrapolation.
    Inputs:
    - pts: Vector of (x, y) coordinates for all time points.
    - t_idx: Vector of time indices corresponding to pts.
    - target_ts: The specific time slice to estimate KDE for.
    - grid_res: Resolution of the output grid (e.g., 100 for 100x100 grid).
    - sd_extension_factor: Multiplier for standard deviation to define the bandwidth.
    Outputs:
    - Tuple (x_grid, y_grid, intensity) where intensity is a matrix.
    """
    # Filter points for the target time slice
    filtered_pts = [p for (i, p) in enumerate(pts) if t_idx[i] == target_ts]
    if isempty(filtered_pts)
        error("No points found for the target time slice $target_ts")
    end
    xs, ys = [p[1] for p in filtered_pts], [p[2] for p in filtered_pts]
    # Calculate bandwidth based on standard deviation of points
    bw_x = std(xs) * sd_extension_factor
    bw_y = std(ys) * sd_extension_factor
    # Define grid boundaries extending slightly beyond the data range
    x_min, x_max = minimum(xs) - bw_x, maximum(xs) + bw_x
    y_min, y_max = minimum(ys) - bw_y, maximum(ys) + bw_y
    x_grid = collect(range(x_min, stop=x_max, length=grid_res))
    y_grid = collect(range(y_min, stop=y_max, length=grid_res))
    intensity = zeros(grid_res, grid_res)
    # Gaussian KDE implementation
    for i in 1:grid_res
        for j in 1:grid_res
            x_val, y_val = x_grid[i], y_grid[j]
            for (px, py) in filtered_pts
                dx = (x_val - px) / bw_x
                dy = (y_val - py) / bw_y
                intensity[i, j] += exp(-0.5 * (dx^2 + dy^2))
            end
        end
    end
    # Normalize intensity to sum to 1 (optional, depending on desired output)
    intensity ./= sum(intensity)
    return x_grid, y_grid, intensity
end

function calculate_metrics(au)
    # Restoration: Calculate assignments and counts based on the actual centroids in the au object
    assigns = [argmin([sum((p .- c).^2) for c in au.centroids]) for p in au.pts]
    counts = [count(==(i), assigns) for i in 1:length(au.centroids)]

    # Safety: Filter valid counts to prevent downstream NaN propagation
    valid_counts = filter(x -> !isnan(x) && !ismissing(x), counts)

    if isempty(valid_counts)
        return (mean_density=NaN, sd_density=NaN, cv_density=NaN)
    end

    m_dens = mean(valid_counts)
    s_dens = std(valid_counts)
    cv_dens = s_dens / (m_dens + 1e-9)

    return (mean_density=m_dens, sd_density=s_dens, cv_density=cv_dens)
end


function get_spatial_graph( centroids, adjacency_edges )
    """
    Synopsis: Converts partitioning results into a formal SimpleGraph. 
    Outputs: A SimpleGraph object.
    """
    n = length(centroids)
    g = SimpleGraph(n)
    centroid_map = Dict(c => i for (i, c) in enumerate(centroids))
    for edge in adjacency_edges
        xi, yi = get(centroid_map, edge[1], 0), get(centroid_map, edge[2], 0)
        if xi > 0 && yi > 0 add_edge!(g, xi, yi) end
    end
    return g
end



function plot_kde_simple(pts; grid_res=600, sd_extension_factor=0.25, title="Spatial Intensity (KDE)")
    # Internal wrapper for estimate_local_kde_with_extrapolation
    # Description: Generates a simple 2D Heatmap of spatial intensity using Kernel Density Estimation.
    # Inputs:
    #   - pts: Vector of (x, y) coordinate tuples.
    #   - grid_res: Resolution of the output grid.
    #   - sd_extension_factor: Factor to extend the bandwidth standard deviation.
    #   - title: Title for the generated plot.
    # Outputs:
    #   - A Plots.Plot object (Heatmap with scatter overlay).
    # Using a dummy t_idx of 1s since we are plotting a static slice
    t_idx_dummy = ones(Int, length(pts))
    x_g, y_g, intensity = estimate_local_kde_with_extrapolation(pts, t_idx_dummy, 1; grid_res=grid_res, sd_extension_factor=sd_extension_factor)

    plt = Plots.heatmap(x_g, y_g, intensity',
                  title=title,
                  c=:viridis,
                  aspect_ratio=:equal,
                  xlabel="X", ylabel="Y")
    Plots.scatter!(plt, [p[1] for p in pts], [p[2] for p in pts],
                   markersize=2, markercolor=:white, markeralpha=0.5, label="Points")
    return plt
end




function scottish_lip_cancer_data_spacetime(n_years::Int=10; rndseed::Int=42)
    # "expand" scottish lip cancer data to a space-time version
    # original data source:  https://mc-stan.org/users/documentation/case-studies/icar_stan.html

    Random.seed!(rndseed)

    # Load base spatial data
 # Base Spatial Data for 56 Counties
    # data source:  https://mc-stan.org/users/documentation/case-studies/icar_stan.html

    nAU = 56

    y_base = [ 9, 39, 11, 9, 15, 8, 26, 7, 6, 20, 13, 5, 3, 8, 17, 9, 2, 7, 9, 7,
    16, 31, 11, 7, 19, 15, 7, 10, 16, 11, 5, 3, 7, 8, 11, 9, 11, 8, 6, 4,
    10, 8, 2, 6, 19, 3, 2, 3, 28, 6, 1, 1, 1, 1, 0, 0]

    E_base = [1.4, 8.7, 3.0, 2.5, 4.3, 2.4, 8.1, 2.3, 2.0, 6.6, 4.4, 1.8, 1.1, 3.3, 7.8, 4.6,
    1.1, 4.2, 5.5, 4.4, 10.5,22.7, 8.8, 5.6,15.5,12.5, 6.0, 9.0,14.4,10.2, 4.8, 2.9, 7.0,
    8.5, 12.3, 10.1, 12.7, 9.4, 7.2, 5.3,  18.8,15.8, 4.3,14.6,50.7, 8.2, 5.6, 9.3, 88.7,
    19.6, 3.4, 3.6, 5.7, 7.0, 4.2, 1.8]

    x_base = [16,16,10,24,10,24,10, 7, 7,16, 7,16,10,24, 7,16,10, 7, 7,10,
    7,16,10, 7, 1, 1, 7, 7,10,10, 7,24,10, 7, 7, 0,10, 1,16, 0,
    1,16,16, 0, 1, 7, 1, 1, 0, 1, 1, 0, 1, 1,16,10]

    adjacency = [ 5, 9,11,19, 7,10, 6,12, 18,20,28, 1,11,12,13,19,
    3, 8, 2,10,13,16,17, 6, 1,11,17,19,23,29, 2, 7,16,22, 1, 5, 9,12,
    3, 5,11, 5, 7,17,19, 31,32,35, 25,29,50, 7,10,17,21,22,29,
    7, 9,13,16,19,29, 4,20,28,33,55,56, 1, 5, 9,13,17, 4,18,55,
    16,29,50, 10,16, 9,29,34,36,37,39, 27,30,31,44,47,48,55,56,
    15,26,29, 25,29,42,43, 24,31,32,55, 4,18,33,45, 9,15,16,17,21,23,25,
    26,34,43,50, 24,38,42,44,45,56, 14,24,27,32,35,46,47, 14,27,31,35,
    18,28,45,56, 23,29,39,40,42,43,51,52,54, 14,31,32,37,46,
    23,37,39,41, 23,35,36,41,46, 30,42,44,49,51,54, 23,34,36,40,41,
    34,39,41,49,52, 36,37,39,40,46,49,53, 26,30,34,38,43,51, 26,29,34,42,
    24,30,38,48,49, 28,30,33,56, 31,35,37,41,47,53, 24,31,46,48,49,53,
    24,44,47,49, 38,40,41,44,47,48,52,53,54, 15,21,29, 34,38,42,54,
    34,40,49,54, 41,46,47,49, 34,38,49,51,52, 18,20,24,27,56,
    18,24,30,33,45,55]

    number_neighbours = [4, 2, 2, 3, 5, 2, 5, 1,  6,  4, 4, 3, 4, 3, 3, 6, 6, 6 ,5,
    3, 3, 2, 6, 8, 3, 4, 4, 4,11,  6, 7, 4, 4, 9, 5, 4, 5, 6, 5,
    5, 7, 6, 4, 5, 4, 6, 6, 4, 9, 3, 4, 4, 4, 5, 5, 6]
 
    # Build graph from adjacency info

    N_edges = Integer(length(adjacency) / 2)
    node1 = fill(0, N_edges)
    node2 = fill(0, N_edges)
    i_adjacency = 0
    i_edge = 0
    for i in 1:nAU
        for j in 1:number_neighbours[i]
            i_adjacency += 1
            if i < adjacency[i_adjacency]
                i_edge += 1
                node1[i_edge] = i
                node2[i_edge] = adjacency[i_adjacency]
            end
        end
    end

    e = Edge.(node1, node2)
    g = Graph(e)
    W = adjacency_matrix(g)
    # D = diagm(vec(sum(W, dims=2)))
 
    au = assign_spatial_units( W ) # "infer" from the adjacency network (W)
    pts_base = au.centroids
    
    N_total = nAU * n_years

    # 1. Random Walk Trend
    rw_trend = cumsum(randn(n_years) .* 0.5)

    # 2. Expand Data Vectors
    y_expanded = repeat(y_base, n_years)
    E_expanded = repeat(E_base, n_years)
    x_expanded = repeat(x_base, n_years)
    t_idx = repeat(1:n_years, inner=nAU)
    pts = repeat(pts_base, n_years)
    # The s_idx is the spatial unit identifier (1 to 56)
    s_idx = repeat(1:nAU, n_years)
 
    # 3. Add Random Walk + Noise to Response
    # Broadcast rw_trend across years
    trend_component = repeat(rw_trend, inner=nAU)
    noise = randn(N_total) .* 0.2

    # Final response: base_y + trend + noise (ensuring positive counts)
    y_final = floor.(Int, abs.(y_expanded .+ trend_component .+ noise))

    # 4. Final covariate matrix and offsets
    x_scaled = (x_expanded .- mean(x_expanded)) ./ std(x_expanded)
    X = Matrix(DataFrame(AFF=x_scaled))
    log_offset = log.(E_expanded)
   
    return (
        y=y_final, X=X, log_offset=log_offset, t_idx=t_idx,
        s_idx=s_idx, n_years=n_years, pts=pts, W=W, au=au
    )
end
 
function get_optimal_sampler(model::DynamicPPL.Model;
    nuts_adapt=50,
    target_acc=0.65,
    pg_particles=20,
    kwargs...)

    # Draw multiple samples from the prior to detect fixed parameters (e.g., Dirac)
    init_samples = [Dict(pairs(rand(model))) for _ in 1:3]
    full_init_dict = init_samples[1]

    # 0. Identify Fixed/Degenerate Parameters (identical across prior samples)
    fixed_params = filter(k -> all(s -> s[k] == full_init_dict[k], init_samples), keys(full_init_dict))

    # 1. Stochastic Discrete Parameters (Integer/Bool and NOT fixed) -> PG
    discrete_params = filter(k -> k ∉ fixed_params && (full_init_dict[k] isa Integer || full_init_dict[k] isa Bool), keys(full_init_dict))

    # 2. Stochastic Continuous Latent Fields (Vectors and NOT fixed) -> ESS
    latent_fields = filter(k -> begin
        val = full_init_dict[k]
        k ∉ fixed_params && k ∉ discrete_params && val isa AbstractVector
    end, keys(full_init_dict))

    # 3. Stochastic Continuous Hyperparameters (Scalars and NOT fixed) -> NUTS
    active_hypers = filter(k -> begin
        val = full_init_dict[k]
        k ∉ fixed_params && k ∉ discrete_params && k ∉ latent_fields && val isa Real
    end, keys(full_init_dict))

    # 4. Fixed / Dirac fallback -> MH (effectively static)
    safe_params = fixed_params

    sampler_blocks = []

    if !isempty(discrete_params)
        push!(sampler_blocks, Tuple(discrete_params) => PG(pg_particles))
        println("Discrete detected (PG): ", discrete_params)
    end

    if !isempty(fixed_params)
        println("Parameters fixed via Dirac() or similar (MH): ", fixed_params)
    end

    if !isempty(latent_fields)
        push!(sampler_blocks, Tuple(latent_fields) => ESS())
    end

    if !isempty(active_hypers)
        push!(sampler_blocks, Tuple(active_hypers) => Turing.NUTS(nuts_adapt, target_acc))
    end

    if !isempty(safe_params)
        push!(sampler_blocks, Tuple(safe_params) => MH())
    end

    # println("--- Optimized Gibbs Sampler Initialized ---")
    # println("ESS (Fields): ", latent_fields)
    # println("NUTS (Hypers): ", active_hypers)

    return Gibbs(sampler_blocks...)
end

function get_initial_parameters(model::DynamicPPL.Model; n_samples=100)
    # 1. Take multiple prior samples to find a robust center. 
    # This avoids starting in extreme tails for diffuse priors.
    samples = [Dict(pairs(rand(model))) for _ in 1:n_samples]
    
    init_dict = Dict{Symbol, Any}()
    for k in keys(samples[1])
        ks = Symbol(k) # Ensure the key is coerced to a Symbol for the dictionary
        vals = [s[k] for s in samples]
        
        # Detect if it's a fixed parameter (Dirac) - variance is zero
        if all(v -> v == vals[1], vals)
            init_dict[ks] = vals[1]
            continue
        end

        mu = mean(vals)
        s_name = string(ks)

        # 2. Apply expert heuristics to refine stochastic parameters
        if vals[1] isa AbstractVector
            # Latent fields: Start at exactly zero (the mathematical mean)
            # This prevents initial energy spikes in gradient-based samplers.
            init_dict[ks] = zeros(eltype(mu), length(mu))
        elseif occursin(r"sigma|ls_|lengthscale", s_name)
            # Scales: Ensure a healthy positive start
            init_dict[ks] = max(0.1, mu) 
        elseif occursin("rho", s_name)
            # Correlations: Clamp to avoid boundary issues (e.g., rho=1.0)
            init_dict[ks] = clamp(mu, -0.9, 0.9)
        elseif occursin("phi_zi", s_name)
            # Zero-inflation: Typically low
            init_dict[ks] = clamp(mu, 0.01, 0.2)
        else
            init_dict[ks] = mu
        end
    end
    return init_dict
end

"""
    parameter_inits(model; n_samples=10, optimizer=LBFGS())

Refines heuristic initial parameters using Maximum A Posteriori (MAP) estimation.
Starts from the robust center found by `get_initial_parameters` and moves 
toward the peak of the posterior density. 

This is highly recommended for complex spatiotemporal models to prevent 
initial energy spikes that can crash the NUTS sampler.
"""
function parameter_inits(model::DynamicPPL.Model; n_samples=10, 
    optimizer=SimulatedAnnealing(), #  
    maxiters = 500,     # Max optimization steps
    maxtime  = 60.0,    # Max allowed time in seconds
    show_trace = true,   # Print progress to the console
    kwargs...)

    # 1. Get the robust heuristic starting point
    init_guess = get_initial_parameters(model; n_samples=n_samples)
    
    # 2. Use MAP to find the local mode (peak density)
    try
        println("Trying MAP for an even better starting point...")
        # Set link=false to prevent Dirac bijector crashes and fix initial_params keyword
        map_res = maximum_a_posteriori(model, optimizer; link=false, 
            initial_params=DynamicPPL.InitFromParams(init_guess), 
            show_trace=show_trace, iterations=maxiters, maxtime=maxtime, kwargs...) 

        return map_res.values
    catch e
        @warn "MAP refinement failed: $e. Falling back to heuristic guess."
        return init_guess
    end

end


 
function kron_matern_sample(Ns, Nt, unique_s, unique_t, ls_s, sigma_s, ls_t, sigma_t, noise_vec)
    # Spatial Precision
    k_s = Matern32Kernel() ∘ ScaleTransform(inv(ls_s))
    K_s = Symmetric(sigma_s^2 * kernelmatrix(k_s, RowVecs(unique_s)) + 1e-4*I)
    Q_s = inv(K_s)

    # Temporal Precision
    k_t = Matern32Kernel() ∘ ScaleTransform(inv(ls_t))
    K_t = Symmetric(sigma_t^2 * kernelmatrix(k_t, unique_t) + 1e-4*I)
    Q_t = inv(K_t)

    # Full Kronecker Precision (Dense for AD compatibility)
    Q_full = Symmetric(Matrix(kron(Q_t, Q_s)) + 1e-4*I)
    L_q = cholesky(Q_full)

    # Sample: f = (L')^-1 * noise
    return L_q.U \ noise_vec
end


function get_posterior_means(ch, param_base, N)
    # Description:
    #   Extracts and averages posterior samples for a specific vector parameter.
    # Inputs:
    #   ch: MCMC sample chain.
    #   param_base: String prefix of the parameter (e.g., "s_eff").
    #   N: Length of the vector parameter.
    # Outputs:
    #   Vector of posterior means.

    means = zeros(N)
    
    for i in 1:N
        p_symbol = Symbol("$param_base[$i]")
        if p_symbol in names(ch, :parameters)
            means[i] = mean(ch[p_symbol])
        else
            @warn "Parameter $p_symbol not found in chain."
        end
    end
    
    return means
end



function generate_inducing_points(coords_st, M_inducing, seed=42)
    # Helper function to generate inducing points (simple random sampling for now)
    Random.seed!(seed)
    N_data = size(coords_st, 1)
    if M_inducing >= N_data
        return coords_st # If M >= N, just use all data points (becomes exact GP)
    end
    indices = sample(1:N_data, M_inducing, replace=false)
    return coords_st[indices, :]
end


function kmeans_inducing_points(coords_st, M_inducing; seed=42)
    # Description:
    #   Identifies optimal inducing point locations using K-Means clustering.
    #   Essential for Sparse/FITC Gaussian Process models.
    # Inputs:
    #   coords_st: N x D matrix of spatiotemporal coordinates.
    #   M_inducing: Target number of inducing points.
    # Outputs:
    #   M x D matrix of cluster centroids.

    Random.seed!(seed)
    
    # Handle cases where data is smaller than requested inducing points
    n_data = size(coords_st, 1)
    if M_inducing >= n_data
        return coords_st
    end

    # Transpose for Clustering.jl compatibility
    data_matrix = Matrix(coords_st')
    
    # Execute K-Means
    clustering_result = kmeans(data_matrix, M_inducing, maxiter=200)
    
    # Extract centroids and transpose back
    inducing_points = clustering_result.centers'
    
    return inducing_points
end



# Squared-exponential covariance function
sqexp_cov_fn(D, phi, noise=1e-6) = exp.(-D^2 / phi) + LinearAlgebra.I * noise

# Exponential covariance function
exp_cov_fn(D, phi) = exp.(-D / phi)



# generic kernel functions 

# lenscale = -1 / log(ρ)
# σ_ar1^2 / (1 - ρ^2) = marginal variance
# kernel_ar1(σ, ρ) = σ^2 * with_lengthscale(Matern12Kernel(), -1/log(ρ)) 
# the softplus should not be necessary ... 

kernel_ar1(σ, ρ) = σ^2 / softplus(1 - ρ^2) * with_lengthscale(Matern12Kernel(), softplus(-1 / log(ρ)) )

# RW2 is equivalent to a Spline kernel or an Integrated Wiener Process
# For simplicity, we often use a high-order Matern or a custom structure

kernel_rw2(σ) = σ^2 * Matern52Kernel() # Matern32 is a common smooth approximation for RW2



function ar1_covariance(n, rho, var, ::Type{T}=Float64) where {T}
    # Description:
    #   Generates a full AR1 covariance matrix.
    # Inputs:
    #   n: Number of time points.
    #   rho: Correlation coefficient.
    #   var: Marginal variance.
    # Outputs:
    #   n x n Covariance matrix.

    vcv = zeros(T, n, n) .+ I(n)
    
    Threads.@threads for r in 1:n
        for c in 1:n
            if r >= c
                vcv[r, c] = var * rho^(r - c)
            end
        end
    end
    
    return Symmetric(vcv)
end

 


function ar1_covariance_local( n, rho, var,  ::Type{T}=Float64 )  where {T} 
    vcv = zeros( T, n, n) .+ I(n) 
    Threads.@threads for r in 1:n
    for c in 1:n
        d = r-c
        if d == 0 | d == 1
            vcv[r,c] = var * rho^d  
        end
    end
    end
    return vcv
end


Turing.@model function gaussian_process_basic(; Y, D, cov_fn=exp_cov_fn, nData=length(Y), noise=1e-6 )
    mu ~ Normal(0.0, 1.0); # mean process
    sig2 ~ LogNormal(0, 1) # "nugget" variance
    phi ~ LogNormal(0, 1) # phi is ~ lengthscale along Xstar (range parameter)
    # sigma = cov_fn(D, phi) + sig2 * LinearAlgebra.I(nData) # Realized covariance function + nugget variance
    vcov = cov_fn(D, phi) + sig2 .* LinearAlgebra.I(nData ) .+ noise * LinearAlgebra.I(nData )
    Y ~ MvNormal(mu * ones(nData), Symmetric(vcov) )     # likelihood
end


Turing.@model function gaussian_process_covars(; Y, X, D, cov_fn=exp_cov_fn, nData=length(Y), nF=size(X,2), noise=1e-6 )
    # model matrix for fixed effects (X)
    beta ~ filldist( Normal(0.0, 1.0), nF); 
    sig2 ~ LogNormal(0, 1) # "nugget" variance
    phi ~ LogNormal(0, 1) # phi is ~ lengthscale along Xstar (range parameter)
    # sigma = cov_fn(D, phi) + sig2 * LinearAlgebra.I(nData) # Realized covariance function + nugget variance
    mu = X * beta # mean process
    vcov = cov_fn(D, phi) + sig2 .* LinearAlgebra.I(nData ) + noise * LinearAlgebra.I(nData )   
    Y ~ MvNormal(mu, Symmetric(vcov) )     # likelihood
end



Turing.@model function gaussian_process_ar1( ::Type{T}=Float64; Y, X, D, ar1, cov_fn=exp_cov_fn, 
    nData=length(Y), nF=size(X,2), nT=maximum(ar1)-minimum(ar1)+1, noise=1e-6 ) where {T} 
 
    rho ~ truncated(Normal(0,1), -1, 1)
    ar1_process_error ~ LogNormal(0, 1) 
    var_ar1 =  ar1_process_error^2 / (1 - rho^2)
    vcv = ar1_covariance_local(nT, rho, var_ar1, T)  # -- covariance by time
    ymean_ar1 ~ MvNormal(Symmetric(vcv) );  # -- means by time 
     
    # # mean process model matrix  
    beta ~ filldist( Normal(0.0, 1.0), nF); 
 
    sig2 ~ LogNormal(0, 1) # "nugget" variance
    phi ~ LogNormal(0, 1) # phi is ~ lengthscale along Xstar (range parameter)
    # sigma = cov_fn(D, phi) + sig2 * LinearAlgebra.I(nData) # Realized covariance function + nugget variance
    # vcov = cov_fn(D, phi) + sig2 .* LinearAlgebra.I(nData )
    
    Y ~ MvNormal( X * beta .+ ymean_ar1[ar1[1:nData]], Symmetric(cov_fn(D, phi) .+ sig2 * I(nData) + noise *I(ndata)  ) )     # likelihood
end

 


function gp_predictions(; Y, D, mu, sig2, phi, cov_fn=exp_cov_fn, nN=length(Xnew), nP=size(res, 1) ) 
    ynew = Vector{Float64}()
    # Threads.@threads -- to add 
    for i in sample(1:size(res,1), nP, replace=true)
        K = cov_fn(D, phi[i])
        Koo_inv = inv(K[(nN+1):end, (nN+1):end])
        Knn = K[1:nN, 1:nN]
        Kno = K[1:nN, (nN+1):end]
        C = Kno * Koo_inv
        mvn = MvNormal( 
            C * (Y .- mu[i]) .+ mu[i], 
            Matrix(LinearAlgebra.Symmetric(Knn - C * Kno')) + sig2[i] * LinearAlgebra.I 
        ) 
        ynew = vcat(ynew, [rand(mvn) ] )
    end
    ynew = stack(ynew, dims=1)  # rehape to matrix   
    return ynew
end




function variational_inference_solution(m; max_iters=100, nsamps=max_iters,  nelbo=3 )

    # Fit via ADVI. minor speed benefit vs NUTS
    _, indices = Bijectors.bijector(m, Val(true));
    vars = keys(indices)

    q0 = Variational.meanfield(m)     # initialize variational distribution (optional)
    advi = ADVI(nelbo, max_iters)    # num_elbo_samples, max_iters
    msol = Turing.vi(m, advi, q0) #, optimizer=Flux.ADAM(1e-1));
    msamples = DataFrame( rand(msol, nsamps )', :auto ) 

    # vectorize variable names ... needs more conditions if 2-D or higher ..
    vns = []
    for (i, sym) in enumerate(vars) 
        j = union(indices[sym]...)  # <= array of ranges
        nj = sum(length.(j)) 
        if  nj > 1
            k = 1
            for r in j
                push!(vns, "$(sym)[$k]")
                k += 1
            end
        else
            push!(vns, "$(sym)") 
        end
    end
    
    vns = Symbol.(vns)

    msamples = rename(msamples, vns)

    mmean = combine( msamples, [ n => (x -> mean(x)) => n for n in names(msamples)  ] )
    mstd  = combine( msamples, [ n => (x -> std(x)) => n for n in names(msamples)  ] )

    out = (
        msol = msol,
        msamples = msamples, 
        mmean = mmean,
        mstd = mstd
    )
    
    return out
 
end

  


function rff_map(coords, W, b)
    # Description:
    #   Maps input coordinates into a Random Fourier Feature (RFF) space
    #   to approximate a kernel function (usually Squared Exponential).
    # Inputs:
    #   coords: N x D matrix of input features (space/time).
    #   W: D x M weight matrix sampled from spectral density.
    #   b: Vector of M random phases.
    # Outputs:
    #   N x M feature matrix.

    # Project coordinates into higher dimensional space
    projection = (coords * W) .+ b'
    
    # Apply cosine transformation with scaling factor
    m = size(W, 2)
    feature_map = sqrt(2 / m) .* cos.(projection)
    
    return feature_map
end

function generate_informed_rff_params(coords, M_rff_count)
    D_in = size(coords, 2)
    std_coords = vec(std(coords, dims=1)) .+ 1e-6
    W_fixed = randn(D_in, M_rff_count) ./ std_coords
    b_fixed = rand(M_rff_count) .* 2pi
    return W_fixed, b_fixed
end

function generate_rff_params_for_se_kernel(D_in, M_rff, lengthscale)
    # Helper function to generate RFF parameters for a Squared Exponential kernel
    # For a Squared Exponential kernel, the spectral density is Gaussian: N(0, (1/l)^2 * I)
    sigma_spectral = 1.0 / lengthscale
    W_matrix = randn(D_in, M_rff) .* sigma_spectral # D_in x M_rff matrix
    b_vector = rand(Uniform(0, 2pi), M_rff)
    return W_matrix, b_vector
end 


function generate_inducing_points(coords_st, M_inducing, seed=42)
    # Helper function to generate inducing points (simple random sampling for now)
    Random.seed!(seed)
    N_data = size(coords_st, 1)
    if M_inducing >= N_data
        return coords_st # If M >= N, just use all data points (becomes exact GP)
    end
    indices = sample(1:N_data, M_inducing, replace=false)
    return coords_st[indices, :]
end


macro save_carstm_state(file_to_save_name_sym)
  quote
    try
      # Evaluate the input symbol (e.g., :state_filename) to its value (e.g., "carstm_state.jld2")
      local filename_val = $(esc(file_to_save_name_sym))
      @info "Saving CARSTM state to $(filename_val)..."
      # JLD2.@save expects variable names as symbols, not their values.
      # The variables themselves should be directly passed.
      JLD2.@save "$(filename_val)" areal_units mod chain pts y_sim y_binary t_idx weights trials cov_indices cov_indices trials_sim  weights_sim adj_matrix_numeric n_pts n_time area_method
      @info "CARSTM state saved successfully."
    catch e
      @error "Error saving CARSTM state: $e"
    end
  end
end

macro load_carstm_state(filename_sym)
  quote
    # Evaluate the input symbol (e.g., :state_filename) to its value (e.g., "carstm_state.jld2")
    local filename_val = $(esc(filename_sym))
    if !isfile(filename_val)
      @error "File $(filename_val) not found."
      return nothing
    end
    try
      @info "Loading CARSTM state from $(filename_val)..."
      # JLD2.@load expects variable names as symbols, not their values.
      # The variables themselves should be directly passed.
      JLD2.@load "$(filename_val)" areal_units mod chain pts y_sim y_binary t_idx weights trials cov_indices cov_indices trials_sim  weights_sim adj_matrix_numeric n_pts n_time area_method
      @info "CARSTM state loaded successfully."
      # Variables are loaded directly into the calling scope by JLD2.@load
      # No explicit return value from the macro itself, as it injects variables
    catch e
      @error "Error loading CARSTM state: $e"
      return nothing
    end
  end
end


function init_params_extract( res=NaN; load_from_file=false, override_means=false, fn_inits = "init_params.jl2"  )
  # Description: Extracts initial parameter values from a model result summary or loads them from a file.
  # Inputs:
  #   - res: Model result object (default: NaN).
  #   - load_from_file: Boolean, if true loads params from fn_inits.
  #   - override_means: Boolean, if true applies custom overrides for specific parameter patterns.
  #   - fn_inits: String, filename for storage.
  # Outputs:
  #   - A FillArray containing the extracted or loaded mean parameter values.

  if load_from_file
    init_params = load(fn_inits )
    return(init_params)
  end

  ressumm = summarize(res)
  vns = ressumm.nt.parameters
  means = ressumm.nt[2]  # means

  if  override_means
    u = findall(x-> occursin(r"^t_period\[", String(x)), vns ); vns[u]
    if length(u) > 0 
      means[u] = [ 1.0, 1.0, 5.0, 5.0]  # (sin, cos) X annual, 5-year (el nino)
    end

    u = findall(x-> occursin(r"^pca_sd\[", String(x)), vns ); vns[u]
    if length(u) > 0  
      means[u] = sigma_prior  # from basic pca
    end

    u = findall(x-> occursin(r"^v\[", String(x)), vns ); vns[u]
    if length(u) > 0 
      means[u] = v_prior  # from basic pca
    end
  end

  init_params = FillArrays.Fill( means )
  jldsave( fn_inits; init_params )

  return(init_params)
end


function init_params_copy( res=NaN, res0=NaN; load_from_file=false, override_means=false, fn_inits = "init_params.jl2"  )
  # using spatial parts of res0 
  # Description: Copies parameter values from a reference result (res0) to a target result structure (res).
  # Inputs:
  #   - res: Target model result object.
  #   - res0: Reference model result object.
  #   - load_from_file: Boolean to load from fn_inits instead.
  #   - override_means: Boolean to apply custom pattern-based overrides.
  #   - fn_inits: String, filename for storage.
  # Outputs:
  #   - A FillArray containing the merged mean parameter values.
  if load_from_file
    init_params = load(fn_inits )
    return(init_params)
  end

  ressumm = summarize(res)
  vns = ressumm.nt.parameters
  means = ressumm.nt[2]  # means

  ressumm0 = summarize(res0)
  vns0 = ressumm0.nt.parameters
  means0 = ressumm0.nt[2]  # means

  if  override_means
    u = findall(x-> occursin(r"^t_period\[", String(x)), vns );  
    if length(u) > 0 
      means[u] = [ 1.0, 1.0, 5.0, 5.0, 10.0, 10.0][1:length(u)]  # (sin, cos) X annual, 5-year (el nino)
    end

    u = findall(x-> occursin(r"^pca_sd\[", String(x)), vns );  
    if length(u) > 0  
      u0 = findall(x-> occursin(r"^pca_sd\[", String(x)), vns0 );  
      if length(u0) > 0  && length(u) == length(u0)
        means[u] = means0[u0]  # from basic pca
      end
    end

    u = findall(x-> occursin(r"^v\[", String(x)), vns );  
    if length(u) > 0  
      u0 = findall(x-> occursin(r"^pca_sd\[", String(x)), vns0 );  
      if length(u0) > 0  && length(u) == length(u0)
        means[u] = means0[u0]  # from basic pca
      end
    end
  end
  
  init_params = FillArrays.Fill( means )
  jldsave( fn_inits; init_params )

  return(init_params)
end


function libgeos_lattice_adjacency_matrix(rows::Int, cols::Int)
    """
    libgeos_lattice_adjacency_matrix(rows, cols)

    Description:
    Generates a sparse adjacency matrix for a regular 2D lattice using LibGEOS for spatial geometry operations.
    Constructs unit square polygons for each cell and identifies neighbors based on Queen contiguity
    (any shared boundary point or edge).

    Inputs:
    - rows (Int): Number of rows in the lattice grid.
    - cols (Int): Number of columns in the lattice grid.

    Output:
    - W (SparseMatrixCSC{Int, Int}): A binary sparse adjacency matrix of size (rows*cols) x (rows*cols).
    """
    # Create polygons for each cell in the lattice
    polygons = []
    for r in 1:rows, c in 1:cols
        # Define unit square coordinates as nested vectors for LibGEOS compatibility
        coords = [
            [Float64(c-1), Float64(r-1)],
            [Float64(c),   Float64(r-1)],
            [Float64(c),   Float64(r)],
            [Float64(c-1), Float64(r)],
            [Float64(c-1), Float64(r-1)]
        ]
        # Construct LinearRing and then Polygon
        ring = LibGEOS.LinearRing(coords)
        push!(polygons, LibGEOS.Polygon(ring))
    end

    n = length(polygons)
    W = spzeros(Int, n, n)

    # Queen contiguity check
    for i in 1:n
        poly_i = polygons[i]
        for j in (i+1):n
            if LibGEOS.intersects(poly_i, polygons[j])
                W[i, j] = W[j, i] = 1
            end
        end
    end
    return W
end


function summarize_array(samples::AbstractArray; alpha=0.05)
    # Return nothing if empty, all-missing, or all-zero (no data/component)
    if isempty(samples) || all(isnan, samples) || all(==(0), samples)
        return nothing
    end

    # Return as scalar if all values across all units and samples are identical (fixed)
    if all(==(first(samples)), samples)
        val = first(samples)
        return val == 0 ? nothing : val
    end

    dims = size(samples)
    n_dims = length(dims)

    post_mean = mean(samples, dims=n_dims)
    post_median = median(samples, dims=n_dims)
    
    # Quantile mapping across the samples dimension
    low_bound = mapslices(x -> quantile(x, alpha/2), samples, dims=n_dims)
    high_bound = mapslices(x -> quantile(x, 1 - alpha/2), samples, dims=n_dims)

    return (
        mean = dropdims(post_mean, dims=n_dims),
        median = dropdims(post_median, dims=n_dims),
        lower = dropdims(low_bound, dims=n_dims),
        upper = dropdims(high_bound, dims=n_dims)
    )
end
 


# --- 1. Custom PC Priors ---
struct PCPriorSigma <: ContinuousUnivariateDistribution
    U::Float64
    alpha::Float64
    lambda::Float64
    function PCPriorSigma(U, alpha)
        return new(U, alpha, -log(alpha) / U)
    end
end

function Distributions.logpdf(d::PCPriorSigma, x::Real)
    x > 0 ? log(d.lambda) - d.lambda * x : -Inf
end

Distributions.rand(rng::AbstractRNG, d::PCPriorSigma) = rand(rng, Exponential(1 / d.lambda))
Distributions.minimum(d::PCPriorSigma) = 0.0
Distributions.maximum(d::PCPriorSigma) = Inf
Bijectors.bijector(d::PCPriorSigma) = Bijectors.exp


function build_laplacian_precision(adj_matrix)
    # Description:
    #   Constructs a GMRF precision matrix (Graph Laplacian) from an adjacency matrix.
    # Inputs:
    #   adj_matrix: Sparse adjacency matrix (W).
    # Outputs:
    #   Sparse precision matrix (Q).

    # D is the diagonal matrix of node degrees
    D_diag = Diagonal(vec(sum(adj_matrix, dims=2)))
    Q_mat = D_diag - adj_matrix
    
    return Q_mat
end
 




function scale_precision!(Q)
    # Description:
    #   Scales a precision matrix using the geometric mean of non-zero eigenvalues.
    #   Essential for ensuring sigma_sp represents marginal variance in BYM2.
    # Inputs:
    #   Q: Precision matrix to be modified in-place.

    eig_vals = eigvals(Matrix(Q))
    # Filter out near-zero eigenvalues associated with the null space
    valid_eigs = filter(x -> x > 1e-6, eig_vals)
    
    scaling_factor = exp(mean(log.(valid_eigs)))
    
    if Q isa Symmetric
        Q.data ./= scaling_factor
    else
        Q ./= scaling_factor
    end
    
    return Q
end





function build_rw2_precision(n)
    # Description: Builds a second-order random walk (RW2) precision matrix for smoothing.
    # Inputs:
    #   - n: Number of categories or time points.
    # Outputs:
    #   - Sparse precision matrix of size n x n.
    D = spzeros(n - 2, n)
    for i in 1:(n - 2)
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    end
    return D' * D
end

function build_ar1_precision(n, rho, tau)
    # Description: Builds a first-order autoregressive (AR1) precision matrix.
    # Inputs:
    #   - n: Number of time points.
    #   - rho: Correlation coefficient.
    #   - tau: Precision scale.
    # Outputs:
    #   - Sparse precision matrix.
    T = promote_type(typeof(rho), typeof(tau))
    diag_vals = [one(T); fill(one(T) + rho^2, n - 2); one(T)]
    off_diag = fill(-rho, n - 1)
    Q = spdiagm(0 => diag_vals, 1 => off_diag, -1 => off_diag)
    return (tau / (one(T) - rho^2)) * Q
end

function build_cyclic_ar1_precision(n, rho, tau)
    # Description: Builds a cyclic AR1 precision matrix (wrapping last to first).
    # Inputs:
    #   - n: Number of time points.
    #   - rho: Correlation coefficient.
    #   - tau: Precision scale.
    # Outputs:
    #   - Sparse precision matrix.
    T = promote_type(typeof(rho), typeof(tau))
    Q = zeros(T, n, n)
    for i in 1:n
        Q[i, i] = one(T) + rho^2
        prev, nxt = (i == 1 ? n : i - 1), (i == n ? 1 : i + 1)
        Q[i, prev] = -rho
        Q[i, nxt] = -rho
    end
    return (tau / (one(T) - rho^2)) * Q
end

function logpdf_gmrf(x, Q)
    # Description: Calculates the log-probability of a Gaussian Markov Random Field.
    # Inputs:
    #   - x: Vector of values.
    #   - Q: Precision matrix.
    # Outputs:
    #   - Log-likelihood value.
    Q_stable = Matrix(Q) + I * 1e-5
    F = cholesky(Symmetric(Q_stable))
    return 0.5 * (logdet(F) - dot(x, Q, x) - length(x) * log(2pi))
end


 
 

function plot_posterior_results(stats, M=nothing, areal_units=nothing; pts=nothing, time_slice=nothing, effect=:spatial, cov_idx=1, show_pts=false)
    # Description: Comprehensive posterior visualization for CARSTM and Deep GP models.

    # Extract target stats and guard against nothing or scalar values
    st = getproperty(stats, effect)
    isnothing(st) && return nothing
    if st isa Real
        return Plots.plot(title="$effect (Fixed: $st)")
    end

    if !isnothing(M)
        pts = M.pts
    end
 
    # 1. Handle Categorical/Class Bar Plots
    if effect == :beta_cov
        b_list = get(stats, :beta_cov, nothing)
        isnothing(b_list) && return nothing
        b_stats = b_list isa AbstractVector ? b_list[cov_idx] : b_list
        (isnothing(b_stats) || b_stats isa Real) && return nothing
        n_levels = size(b_stats.mean, 1)
        return StatsPlots.bar(1:n_levels, b_stats.mean[:,1],
                  yerror=(b_stats.mean[:,1] .- b_stats.lower[:,1], b_stats.upper[:,1] .- b_stats.mean[:,1]),
                  title="Covariate $cov_idx Effects", xlabel="Level", ylabel="Effect Size", legend=false)

    elseif effect == :b_class1 || effect == :b_class2
        b_stats = st
        if isnothing(b_stats) || b_stats isa Real; return nothing; end
        n_levels = size(b_stats.mean, 1)
        return StatsPlots.bar(1:n_levels, b_stats.mean[:,1],
                  yerror=(b_stats.mean[:,1] .- b_stats.lower[:,1], b_stats.upper[:,1] .- b_stats.mean[:,1]),
                  title="$effect Levels", xlabel="Class Index", ylabel="Effect Size", legend=false)

    # 2. Handle Temporal Main Effects
    elseif effect == :temporal
        t_stats = st
        if isnothing(t_stats) || t_stats isa Real; return nothing; end
        n_times = length(t_stats.mean)
        return StatsPlots.plot(1:n_times, t_stats.mean,
                   ribbon=(t_stats.mean .- t_stats.lower, t_stats.upper .- t_stats.mean),
                   fillalpha=0.2, lw=2, title="Temporal Main Effect", xlabel="Time Index", ylabel="Effect (Latent Scale)", legend=false)

    # 2a. Handle Seasonal Effects
    elseif effect == :seasonal
        t_stats = st
        if isnothing(t_stats) || t_stats isa Real; return nothing; end
        n_times = length(t_stats.mean)
        return StatsPlots.plot(1:n_times, t_stats.mean,
                   ribbon=(t_stats.mean .- t_stats.lower, t_stats.upper .- t_stats.mean),
                   fillalpha=0.2, lw=2, title="Seasonal Effect", xlabel="Time Index", ylabel="Effect (Latent Scale)", legend=false)


    # 3. Handle Spatial, ST, and Deep GP Mean Fields
    elseif effect in [:spatial, :spatial_structured, :spatial_unstructured, :predictions_denoised, :predictions_noisy, :residuals, :eta_gp, :hidden_layer]
        plt = StatsPlots.plot(aspect_ratio=:equal, title="$effect (T=$(time_slice))", legend=true)

        # Determine the values to map to colors
        values = if hasproperty(st, :mean)
            st.mean
        elseif effect == :spatial_structured
            stats.spatial_structured.mean
        elseif effect == :spatial_unstructured
            stats.spatial_unstructured.mean
        elseif effect == :eta_gp
            haskey(stats, :eta_gp) ? stats.eta_gp.mean : error("eta_gp not found in stats")
        elseif effect == :hidden_layer
            haskey(stats, :h1) ? stats.h1.mean : error("hidden layer h1 not found in stats")
        elseif effect == :predictions_denoised && !isnothing(time_slice)
            stats.predictions_denoised.mean[:, time_slice]
        elseif effect == :predictions_noisy && !isnothing(time_slice)
            stats.predictions_noisy.mean[:, time_slice]
        # elseif effect == :residuals
        #    stats.predictions_noisy.mean[:, isnothing(time_slice) ? 1 : time_slice]
        else
            error("Effect $effect requires specific keys in stats or time_slice index")
        end

        # SAFETY FIX: Plot only as many polygons as we have results for to avoid BoundsError
        n_to_plot = min(length(areal_units.polygons), length(values))

        for i in 1:n_to_plot
            poly_coords = areal_units.polygons[i]
            if length(poly_coords) > 2
                px = [pt[1] for pt in poly_coords if !isnan(pt[1])]
                py = [pt[2] for pt in poly_coords if !isnan(pt[2])]

                if !isempty(px)
                    if (px[1], py[1]) != (px[end], py[end])
                        push!(px, px[1]); push!(py, py[1])
                    end

                    val = values[i]
                    StatsPlots.plot!(plt, px, py,
                        seriestype=:shape,
                        fill_z=val,
                        c=:RdYlBu,
                        linecolor=:black,
                        linewidth=0.5,
                        fillalpha=0.8,
                        legend=false
                    )
                end
            end
        end

        if show_pts
            StatsPlots.scatter!(plt, [p[1] for p in pts], [p[2] for p in pts],
                markersize=1, markercolor=:gray, alpha=0.2, label="Observations")
        end

        StatsPlots.scatter!(plt, [c[1] for c in areal_units.centroids], [c[2] for c in areal_units.centroids],
            markersize=2, markercolor=:white, markerstrokecolor=:black, alpha=0.5, label="Centroids")

        return plt
    else
        error("Effect $effect not recognized.")
    end
end




function plot_posterior_vs_prior(model::DynamicPPL.Model, chain::MCMCChains.Chains, param_sym::Symbol; n_prior_samples=1000, title="Posterior vs Prior")
    # Description: Overlays posterior and prior densities for a specific parameter to check learning/shrinkage.
    # Inputs:
    #   - model: Turing model object.
    #   - chain: MCMC sample chain.
    #   - param_sym: Symbol of the parameter to check.
    # Outputs:
    #   - A Plots.Plot object.

    # 1. Extract posterior samples using .data for AxisArray compatibility
    post_samples = vec(chain[param_sym].data)

    # 2. Automated Prior Sampling via Turing
    prior_chain = sample(model, Prior(), n_prior_samples, progress=false)
    prior_samples = vec(prior_chain[param_sym].data)

    # 3. Visualization
    plt = StatsPlots.density(post_samples, label="Posterior: $param_sym", lw=3, color=:blue, fill=(0, 0.2, :blue))
    StatsPlots.density!(plt, prior_samples, label="Prior (sampled)", lw=2, ls=:dash, color=:red)

    title!(plt, title)
    xlabel!(plt, "Value")
    ylabel!(plt, "Density")

    return plt
end




# --- 1. MODEL UTILITIES ---

function NegativeBinomial2(μ, r)
    # Description: Alternative parametrization of Negative Binomial using mean (μ) and dispersion (r).
    # Inputs:
    #   - μ: Mean.
    #   - r: Size/dispersion parameter.
    # Outputs:
    #   - Distributions.NegativeBinomial object.

    p = r / (r + μ)
    return NegativeBinomial(r, p)
end
  

function get_rff_deep2D_basis(X, m, lengthscale)
    # Description: Generates Random Fourier Feature (RFF) basis for 2D inputs (Spatial/Temporal).
    # Inputs:
    #   - X: Input matrix (N x D).
    #   - m: Number of features.
    #   - lengthscale: Gaussian kernel lengthscale.
    # Outputs:
    #   - N x m feature matrix.
    N, D = size(X)
    Random.seed!(42)
    Omega_samples = randn(m, D) ./ lengthscale
    Phi_phases = rand(m) .*  2pi
    return sqrt(2/m) .* cos.(X * Omega_samples' .+ Phi_phases')
end


function get_rff_trend_basis(t, m, lengthscale, ::Type{T}=Float64) where {T}
    N = length(t)
    # Generate random parameters for RFFs.
    # Using a seed ensures consistency within the AD pass.
    Random.seed!(42)
    Omega_samples_float = randn(m)
    Phi_phases_float = rand(m)

    Omega_samples = Omega_samples_float ./ lengthscale
    Phi_phases = Phi_phases_float .* convert(T,  2pi)

    Z = zeros(T, N, m)
    for j in 1:m
        Z[:, j] = convert.(T, sqrt(2/m)) .* cos.(Omega_samples[j] .* t .+ Phi_phases[j])
    end
    return Z
end


function get_rff_seasonal_basis(t, m, freq, lengthscale)
    # Description: Generates RFF-style basis for periodic seasonal components.
    # Inputs:
    #   - t: Time vector.
    #   - m: Number of harmonics.
    #   - freq: Base frequency.
    #   - lengthscale: Smoothness scale.
    # Outputs:
    #   - N x (2*m) feature matrix.
    N = length(t)
    Z = zeros(N, 2*m)
    for j in 1:m
        omega_j =  2pi * j * freq
        Z[:, 2j-1] = cos.(omega_j .* t)
        Z[:, 2j] = sin.(omega_j .* t)
    end
    return Z
end

function bstm_options(M_previous::NamedTuple; kwargs...)
    # Convert NamedTuple to Dict to allow merging
    M_new = Dict{Symbol, Any}(pairs(M_previous))
    # Update with new keyword arguments
    for (key, val) in kwargs
        if isnothing(val)
            delete!(M_new, key)
            M_new[key] = nothing
        else
            M_new[key] = val
        end
    end
    # Re-run original bstm_options to ensure all computed fields are correctly updated
    return bstm_options(; M_new...)
end


function bstm_options(; kwargs...)

    M_base = Dict{Symbol, Any}(kwargs)
    
    model_arch =  get(  M_base, :model_arch, "univariate")
    model_family = get(  M_base, :model_family, "gaussian")  # "poisson", "gaussian", "binomial", "negbin", "lognormal"
    model_space =  get(  M_base, :model_space, "bym2") # 
    model_time = get(  M_base, :model_time, "ar1")  
    model_season = get(  M_base, :model_time, "none")

    model_st = get(  M_base, :model_st, 1)  # type 1 separable
        # Type of Space-time Interaction Effect:
        #- Type I:  (IID) ε_{at} ~ N(0, σ²)
        #- Type II: (Time-Structured) Interaction structured by AR1 in time, independent in space.
        #- Type III: (Space-Structured) Interaction structured by ICAR in space, independent in time.
        #- Type IV: (Fully-Structured) Interaction via Kronecker Product (s_Q ⊗ Q_tm).
        # st_interaction_type = 1    default
     

    y_obs = get(M_base, :y_obs, nothing)
    N_obs = get(M_base, :N_obs, size(y_obs, 1) )  # y_obs can be a matrix
    
    y_has_data = findall(!ismissing, y_obs)


    s_idx = get(M_base, :s_idx, 1)  # force to carry to make dimensional expectations simpler ..
    t_idx = get(M_base, :t_idx, 1)  # force to carry to make dimensional expectations simpler ..
    u_idx = get(M_base, :u_idx, 1)  # force to carry to make dimensional expectations simpler ..

    W = get(M_base, :W, I(1))
    
    N_time = get(M_base, :N_time, (t_idx === nothing ? 1 : maximum(t_idx)))
    N_season = get(M_base, :N_season, 12)
    N_areas = size(W, 1)

    # actual values (directly from observations)
    s_obs = get(M_base, :s_obs, nothing)
    t_obs = get(M_base, :t_obs, nothing)
    u_obs = get(M_base, :u_obs, nothing)

    period = get(M_base, :period, 12)
    weights = get(M_base, :weights, ones(Float64, N_obs))
    log_offset = get(M_base, :log_offset, zeros(Float64, N_obs))
    trials = get(M_base, :trials, ones(Int, N_obs))

    
    rnd_seed = get(M_base, :rnd_seed, 42)
    Random.seed!(rnd_seed)
    
    noise = get(M_base, :noise, 1e-6)   # jitter for Pos Def
    

    # Fixed weights for multi-fidelity/A-series models (A13-A25)
    # --- 2. Initialize Fixed Projection Matrices (DFRFF) ---
    # These are static weights for the functional mapping layers
    # M_base = 40
    # M_sig = 20

    # Layer Z: Maps Space (Dim 2) to Latent Z
    # W_z_fix = randn(2, M_base); b_z_fix = rand(M_base) .* 2π

    # Layer W: Maps [Space, Time, Z] (Dim 4) to Latent U
    # W_w_fix = randn(4, M_base); b_w_fix = rand(M_base) .* 2π

    # Layer Y: Maps [Space, Time, Z, U] (Dim 5) to Output Y
    # W_y_fix = randn(5, M_base); b_y_fix = rand(M_base) .* 2π

    # Layer Sigma: Maps [Space, Time] (Dim 3) to Volatility
    
    # --- RFF Spectral Dimensions ---
    M_rff = get(M_base, :M_rff, 40)
    M_inducing_val = get( M_base, :M_inducing_val, 15)
    
    D_in = get(M_base, :D_in, 2)

    W_fixed = randn(D_in, M_rff)
    b_fixed = rand(M_rff) .* 2 * pi
 
    y_coords_st =  get(M_base, :y_coords_st, nothing )
    use_sv = get(  M_base, :use_sv, false)  # stochastic volatitity
    
    if use_sv
        y_coords_st = !isnothing(y_coords_st) ? y_coords_st :  hcat(s_obs, t_obs ./ N_time) 
    end
    M_rff_sigma = get(M_base, :M_rff_sigma, 20)
    W_sigma_fixed = randn(3, M_rff_sigma); 
 
    M_rff_w = get( M_base, :M_rff_w, 30)

    W_z_fixed = randn(2, M_rff); 
    b_z_fixed = rand(M_rff) .* 2pi

    W_w_fixed = randn(4, M_rff); 
    b_w_fixed = rand(M_rff) .* 2pi

    W_y_gp_fixed = randn(3, M_rff); 
    b_y_gp_fixed = rand(M_rff) .* 2pi

    b_sigma_fixed = rand(M_rff_sigma) .* 2pi


    # 5. Spatial Geometric Indices
    # Coordinates and Inducing points for FITC Sparse GPs
    st_obs = !isnothing(s_obs) && !isnothing(t_obs) ? hcat( (s_obs, t_obs)... ) : nothing
    
    Z_inducing = !isnothing(st_obs) ? kmeans_inducing_points(st_obs, M_inducing_count) : nothing
     
    # Subsets for varied fidelity resolutions
    
    z_obs = get(M_base, :z_obs, nothing)   # spatial only
    w_obs = get(M_base, :w_obs, nothing)   # space-time 
    
    N_z = !isnothing(z_obs) ? size(z_obs, 2) : 0
    N_w = !isnothing(w_obs) ? size(w_obs, 2) : 0 
     
    z_coords_s = get(M_base, :z_coords_s, nothing)  
    w_coords_st = get(M_base, :w_coords_st, nothing)  # consider scaling time to range from 0 to 1
 
    # Interaction & Covariate Mapping Vectors
    # interaction_idx: Maps (area, time) to a unique linear index for Type IV interactions
    # interaction_idx = (t_idx .- 1) .* N_areas .+ s_idx

    # linear design matrix (beta)
    fixed = get( M_base, :fixed, ones(N_obs) )  # intercept only by default
    N_fixed = !isnothing(fixed) ? size(fixed, 2) : 0  

    # random effects covariates that will be smoothed outside of the design matrix (fixed) 
    # e.g. discretizes covariates for RW2 smoothing models, etc
    N_cat = get(M_base, :N_cat, 9)
    
    cov = get(M_base, :cov, nothing )
    N_cov = !isnothing(cov) ? size(cov,2) : 0

    model_cov = get(M_base, :model_cov, nothing )
    cov_indices = get(M_base, :cov_indices, nothing )
    
    c_period = nothing
    c_angle = nothing
    c_steps = nothing

    c_Q = nothing 

    if !isnothing(model_cov) 

        if isnothing(cov_indices)    
            cov_indices = zeros(Int, N_obs, N_cov)
            if N_cov > 0
                probs = collect(range(0.0, stop=1.0, length=N_cat + 1))
                for i in 1:N_cov
                    cov_indices[:,i] = map(x -> clamp(searchsortedfirst(quantile(cov[:,i], probs), x) - 1, 1, N_cat), cov[:,i])
                end
            end
        end

        if model_cov == "ar1"
            c_Q =  build_bstm_ar1_template(N_cat)
        
        elseif model_cov == "rw2"
            c_Q =  build_bstm_rw2_template(N_cat)

        elseif model_cov == "harmonic"
            # c_eff_mat[:, k] = (c_α_k .* sin.(2π .* c_steps ./ 12.0) .+ c_β_k .* cos.(2π .* c_steps ./ 12.0)) .* c_sigma[k]
            c_steps =   get(M_base, :c_steps, 1 )
            c_period =   get(M_base, :c_period, 1 )
            c_angle = 2π * c_steps / c_period
            c_Q = Matrix(build_bstm_harmonic_template(N_cat).matrix)
        end



    end
    
    # Templates for timeseries effects

    t_Z_trend = nothing
    t_Z_seas = nothing
 
    # 3. Seasonal Precision (u_Q)
    u_Q = nothing
    if !isnothing(N_season) 
        if model_season == "ar1"
            u_Q = build_bstm_ar1_template(N_season).matrix
        elseif model_season == "rw2"
            u_Q = build_bstm_rw2_template(N_season).matrix
            
        elseif model_season =="harmonic"
            # Simple seasonal cycle: alpha*sin(2pi*t/P) + beta*cos(2pi*t/P)
            # t_eff = (h_α .* sin.(2π .* t_steps ./ period) .+ h_β .* cos.(2π .* t_steps ./ period)) .* t_sigma
            u_period = 12.0 # Default period (annual)
            u_steps = 1:N_season
            u_angle = 2π * t_steps / t_period
            u_Q = build_bstm_harmonic_template(N_season).matrix

        elseif model_season =="rw2"
            u_Q = build_bstm_rw2_template(N_season).matrix
        else 
            u_Q = I(N_season)
        end
    end

    t_period = nothing
    t_steps = nothing
    t_angle = nothing

    # 2. Temporal Precision (t_Q)
    t_Q = nothing
    if !isnothing(N_time)
        if model_time == "ar1"
            t_Q = build_bstm_ar1_template(N_time).matrix
        elseif model_time == "rw2"
            t_Q = build_bstm_rw2_template(N_time).matrix
        
        elseif model_time == "harmonic"
            # Simple annual cycle: alpha*sin(2pi*t/P) + beta*cos(2pi*t/P)
            # t_eff = (h_α .* sin.(2π .* t_steps ./ period) .+ h_β .* cos.(2π .* t_steps ./ period)) .* t_sigma
            t_period = 1.0 # Default period (annual)
            t_steps = 1:N_time
            t_angle = 2π * t_steps / t_period
            
            t_Q = build_bstm_harmonic_template(N_time).matrix

        elseif model_time == "rff"
            # Fixed Projections & RFF Bases
            # Generating fixed weights and phases ensures stability across iterations for deep functional RFFs
            # Trend & Seasonality Static Projection Vectors
            m_trend = get(M_base, :m_trend, 10) 
            m_seas = get(M_base, :m_seas, 5)
            t_vec = get(M_base, :t_vec,  collect(1:N_time) ./ N_time )
            Om_tr = randn(Float64, m_trend) ./ 0.5; 
            Ph_tr = rand(Float64, m_trend) .* 2pi
            t_Z_trend = sqrt(2/m_trend) .* cos.(t_vec * Om_tr' .+ Ph_tr')
            t_Z_seas = zeros(Float64, N_time, 2 * m_seas)
    #        for j in 1:m_seas
    #            om_j = 2*pi * j; t_Z_seas[:, 2j-1] = cos.(om_j .* t_vec); t_Z_seas[:, 2j] = sin.(om_j .* t_vec)
    #        end
        else 
            t_Q = I(N_time)
        end
    end

    # spatial effects pre-computes 
    cluster_assignments=nothing
    n_clusters = nothing
    n_layers = nothing
    m_layers = nothing

    M_inducing_count = !isnothing(st_obs) ? get(M_base, :M_inducing_count, 15) : nothing
    K_nystrom_proj = nothing
    kernel_nystrom =  get( M_base, :kernel_nystrom,  Matern32Kernel() ∘ ScaleTransform(1.0) )
    inducing_points = get( M_base, :inducing_points, nothing)


    s_Q = I(N_areas)
 
    if model_space in [ "bym2", "besag", "leroux", "svc", "local" ] 
        # Standardized Precision Scaling (BYM2 & RW2)
        # Pre-scaling ensures that priors on sigma_sp and sigma_rw2 are interpretable as marginal scales
        # A. Spatial Scaling (BYM2 ICAR) .. aka : Standard Graph Laplacian
        D_sp = Diagonal(vec(sum(W, dims=2)))
        Q_raw = D_sp - W
        eigs = filter(x -> x > 1e-6, eigvals(Matrix(Q_raw))) 
        # FIX: Explicitly cast to dense Matrix to prevent PDMats MethodError
        sf = isempty(eigs) ? 1.0 : exp(mean(log.(eigs)))
        s_Q = Symmetric(Matrix(Q_raw ./ sf))
    
    elseif model_space == "deepgp"
    
        # --- 2. Hierarchical Deep GP Construction ---
        # Initial input: [Lon, Lat, Time]
        
        current_input = hcat([p[1] for p in M.pts], [p[2] for p in M.pts], Float64.(M.t_idx))
        m_layers = get( M_base, :m_layers, m_layers=[10, 5]) 
        n_layers = length(m_layers)

    elseif model_space == "local"
        
        n_clusters = get(M_base, :n_clusters, 4)
        cl_res = kmeans( M.s_coords[1:N_areas, :]' , n_clusters)  # fix to regular grid
        cluster_assignments = cl_res.assignments

    elseif model_spatial == "nystrom" 
        # Nystrom low-rank approximation
        inducing_points = !isnothing(inducing_points) ? inducing_points : sobs[1:10,]
        K_nystrom_proj = precompute_nystrom_projection( s_obs, inducing_points, kernel_nystrom; jitter=1e-6)
    end
  

    updates = (
        y_obs = y_obs,
        
        s_idx = s_idx, 
        s_obs = s_obs, 
        
        t_idx = t_idx, 
        t_obs = t_obs, 
        
        u_idx =u_idx,
        u_obs = u_obs, 
        
        W = W, 
        trials = trials, 
        weights = weights, 
        log_offset = log_offset,
        
        y_has_data = y_has_data,
        y_coords_st = y_coords_st,

        z_obs = z_obs, 
        z_coords_s = z_coords_s,   
        N_z=N_z,

        w_obs = w_obs, 
        w_coords_st = w_coords_st,
        N_w=N_w,
        
        s_Q = s_Q, 
        t_Q = t_Q,
        u_Q = u_Q,

        t_Z_trend = t_Z_trend, 
        t_Z_seas = t_Z_seas, 
        
        t_period = t_period,
        t_steps = t_steps,
        t_angle = t_angle,
  
        c_period = c_period, 
        c_steps = c_steps,
        c_angle = c_angle, 
    
        Z_inducing = Z_inducing,

        M_inducing_val = M_inducing_val,
        K_nystrom_proj = K_nystrom_proj,
        kernel_nystrom = kernel_nystrom,
        M_inducing_count = M_inducing_count, 
        use_sv = use_sv,

        M_rff = M_rff,
        M_rff_w = M_rff_w,
        M_rff_sigma = M_rff_sigma,
        W_z_fixed = W_z_fixed, 
        b_z_fixed = b_z_fixed, 
        W_w_fixed = W_w_fixed, 
        b_w_fixed = b_w_fixed,
        W_y_gp_fixed = W_y_gp_fixed, 
        b_y_gp_fixed = b_y_gp_fixed, 

        W_sigma_fixed = W_sigma_fixed, 
        b_sigma_fixed = b_sigma_fixed,

        cluster_assignments = cluster_assignments,
        # interaction_idx = interaction_idx,   
        W_fixed = W_fixed,
        b_fixed = b_fixed,
        rff_scale = sqrt(2.0 / M_rff),  

        model_arch = model_arch,
        model_family = model_family,
        model_space = model_space,
        model_time = model_time,
        model_season = model_season,
        model_st = model_st,
        model_cov = model_cov,
        
        fixed = fixed, 
        cov_indices = cov_indices,
        cov = cov,  

        n_mosaics = get(M_base, :n_mosaics, 5),
        m_rff = get(M_base, :m_rff, 20),
        D_st =  get(M_base, :D_st, 3 ),  # svgp
        period = period,  # svgp
        
        N_obs = N_obs, 
        N_areas = N_areas, 
        N_time = N_time, 
        N_season = N_season,
        N_fixed = N_fixed,
        N_cat = N_cat,
        N_cov = N_cov,
        
        
        m_feat = 5,
        m_warp = 10, 
        m_spatial=50,
        grid_res = get(M_base, :grid_res,  64), 
        pad_factor = get(M_base, :pad_factor, 2),

        n_clusters = n_clusters, # "localised"
        noise=noise,
        st_interaction_type =  get(M_base, :st_interaction_type, 0),  # default is no interaction      

        use_zi = get(M_base, :use_zi, false)
    )

    for (k, v) in pairs(updates); M_base[k] = v; end

    return NamedTuple(M_base)
end
 

 


# -----------------------



function assign_spatial_units_inferred(adjacency_matrix; iterations=50, learning_rate=0.1, buffer_dist=0.5, input_polygons = nothing)
    """
    Synopsis: Manually constructs a areal_units object for areal data like the Lip Cancer dataset.
              Centroid locations are spatially inferred from connectivity using a rudimentary force-directed layout.
    Inputs:
    - adjacency_matrix: The adjacency matrix (W) of the areal units.
    - iterations: Number of iterations for the force-directed layout.
    - learning_rate: Step size for moving centroids in the layout algorithm.
    - buffer_dist: Distance to buffer the convex hull when polygons are inferred.
    - input_polygons: Optional. A vector of LibGEOS Polygons. If provided, centroids and hull are derived from these.
    """

    local final_centroids
    local adjacency_edges_output
    local polys_output
    local hull_coords_output
    local g_final # The final graph that will be in the result

    nAU = size(adjacency_matrix, 1)


    if input_polygons !== nothing && !isempty(input_polygons)
        # Case 1: Polygons are provided
        # 1. Extract centroids from input_polygons
        final_centroids_geoms = [LibGEOS.centroid(p) for p in input_polygons]
        final_centroids = map(final_centroids_geoms) do g_pt
            seq = LibGEOS.getCoordSeq(g_pt)
            (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
        end

        # 2. Determine hull by dissolving all internal edges
        united_geom = LibGEOS.unaryunion(input_polygons)
        hull_coords_output = get_coords_from_geom(united_geom)

        # 3. Determine adjacency from input_polygons (using LibGEOS.touches)
        adjacency_edges_output = []
        for i in 1:nAU
            g1 = input_polygons[i]
            for j in (i+1):nAU
                g2 = input_polygons[j]
                if LibGEOS.touches(g1, g2)
                    push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                else
                    # Fallback robust check, similar to get_voronoi_polygons_and_edges
                    g1_buffered = LibGEOS.buffer(g1, 1e-6)
                    if LibGEOS.intersects(g1_buffered, g2)
                        inter = LibGEOS.intersection(g1_buffered, g2)
                        if !LibGEOS.isEmpty(inter) && (LibGEOS.area(inter) > 1e-9 || LibGEOS.geomTypeId(inter) in [LibGEOS.GEOS_LINESTRING, LibGEOS.GEOS_MULTILINESTRING])
                            push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                        end
                    end
                end
            end
        end

        polys_output = [get_coords_from_geom(p) for p in input_polygons]

        # Build graph from the determined adjacency edges and ensure connectivity
        g_final = SimpleGraph(nAU)
        centroid_map = Dict(c => i for (i, c) in enumerate(final_centroids))
        for (c1, c2) in adjacency_edges_output
            xi = get(centroid_map, c1, 0)
            yi = get(centroid_map, c2, 0)
            if xi > 0 && yi > 0 && !has_edge(g_final, xi, yi)
                add_edge!(g_final, xi, yi)
            end
        end
        g_final = ensure_connected!(g_final, final_centroids) # Ensure connectivity if necessary

    else
        # Case 2: Polygons are not provided, infer centroids and use tessellation
        # 1. Build initial graph from adjacency_matrix for force-directed layout
        g_initial_for_layout = SimpleGraph(adjacency_matrix)

        # 2. Infer initial centroids using force-directed layout
        side = ceil(Int, sqrt(nAU))
        initial_centroids_fd = [(Float64(i % side), Float64(i ÷ side)) for i in 0:(nAU-1)]
        centroids_vec = [SVector{2, Float64}(c) for c in initial_centroids_fd]

        for iter in 1:iterations
            new_centroids_vec = copy(centroids_vec)
            for i in 1:nAU
                neighbors_i = Graphs.neighbors(g_initial_for_layout, i)
                if !isempty(neighbors_i)
                    avg_neighbor_pos = sum(centroids_vec[n] for n in neighbors_i) / length(neighbors_i)
                    new_centroids_vec[i] = centroids_vec[i] + learning_rate * (avg_neighbor_pos - centroids_vec[i])
                end
            end
            centroids_vec = new_centroids_vec
        end
        # Centroids after force-directed layout
        forced_layout_centroids = [(p[1], p[2]) for p in centroids_vec]

        # 3. Determine hull_geom from inferred centroids for clipping
        hull_geom = expand_hull(forced_layout_centroids, buffer_dist)
        hull_coords_output = get_coords_from_geom(hull_geom)

        # 4. Use tessellation to determine polygon coordinates and initial adjacency (based on forced_layout_centroids)
        polys_coords_raw, _ = get_voronoi_polygons_and_edges(forced_layout_centroids, hull_geom) # Discard initial edges as they refer to old centroids

        # 5. RECOMPUTE CENTROIDS from the generated (clipped) polygons and prepare for adjacency
        final_centroids = Vector{Tuple{Float64, Float64}}(undef, length(polys_coords_raw))
        lg_polygons_for_adjacency = Vector{Union{LibGEOS.Polygon, Nothing}}(undef, length(polys_coords_raw)) # Allow Nothing
        polys_output = polys_coords_raw # Keep the raw coordinates for output

        for (idx, poly_coord_list) in enumerate(polys_coords_raw)
            if !isempty(poly_coord_list) && length(poly_coord_list) >= 3 # Ensure it's a valid polygon
                # Make sure the polygon is closed for LibGEOS
                if poly_coord_list[1] != poly_coord_list[end]
                    push!(poly_coord_list, poly_coord_list[1])
                end
                lg_poly = LibGEOS.Polygon([ [Float64[p[1], p[2]] for p in poly_coord_list] ])
                centroid_geom = LibGEOS.centroid(lg_poly)
                seq = LibGEOS.getCoordSeq(centroid_geom)
                final_centroids[idx] = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
                lg_polygons_for_adjacency[idx] = lg_poly
            else
                @warn "Invalid or empty polygon encountered in Voronoi tessellation at index $idx. Using original centroid as fallback, and skipping polygon for adjacency checks."
                final_centroids[idx] = forced_layout_centroids[idx]
                lg_polygons_for_adjacency[idx] = nothing # Mark as invalid for adjacency checks
            end
        end

        # 6. Re-build adjacency based on the newly derived centroids and polygons
        adjacency_edges_output = []
        if !isempty(lg_polygons_for_adjacency)
            for i in 1:length(lg_polygons_for_adjacency)
                g1 = lg_polygons_for_adjacency[i]
                if g1 === nothing continue end # Skip if polygon is invalid
                for j in (i+1):length(lg_polygons_for_adjacency)
                    g2 = lg_polygons_for_adjacency[j]
                    if g2 === nothing continue end # Skip if polygon is invalid
                    # Check for adjacency using LibGEOS predicates
                    if LibGEOS.touches(g1, g2)
                        push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                    else
                        # Fallback: Robust check using a tiny buffer for floating-point misalignments
                        g1_buffered = LibGEOS.buffer(g1, 1e-6)
                        if LibGEOS.intersects(g1_buffered, g2)
                            inter = LibGEOS.intersection(g1_buffered, g2)
                            # Check if intersection is a line or has significant area
                            if !LibGEOS.isEmpty(inter) && (LibGEOS.area(inter) > 1e-9 || LibGEOS.geomTypeId(inter) in [LibGEOS.GEOS_LINESTRING, LibGEOS.GEOS_MULTILINESTRING])
                                push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                            end
                        end
                    end
                end
            end
        end

        # 7. Build final graph from the re-derived adjacency edges and ensure connectivity
        g_final = SimpleGraph(nAU) # nAU is the count of regions
        centroid_map = Dict(c => i for (i, c) in enumerate(final_centroids)) # Map new centroids to indices
        for (c1, c2) in adjacency_edges_output
            xi = get(centroid_map, c1, 0)
            yi = get(centroid_map, c2, 0)
            if xi > 0 && yi > 0 && !has_edge(g_final, xi, yi)
                add_edge!(g_final, xi, yi)
            end
        end
        g_final = ensure_connected!(g_final, final_centroids)
    end

    return (
        centroids = final_centroids,
        adjacency_edges = adjacency_edges_output,
        graph = g_final,
        polygons = polys_output,
        hull_coords = hull_coords_output
    )
    
    """
    # Generate the spatial metadata for dataset when only W is known
        areal_units = assign_spatial_units_inferred(W, nAU)
        println("Spatial metadata created for Scottish Lip Cancer dataset.")
        println("Number of units: ", length(areal_units.centroids))
        println("Graph connectivity: ", is_connected(areal_units.graph))
        
        # Quick test run: model 2 using explicit unpacking of required fields
        plt = plot_spatial_graph(lip_inputs.pts, areal_units; title="Lip Cancer Spatial Graph", domain_boundary=lip_inputs.pts)
        display(plt)
        
        println("First few centroids from areal_units: ", areal_units.centroids[1:min(5, length(areal_units.centroids))])
    """
end

 


function get_params_vector(chain, base_name, len)
    # Use the robust helper to get names from either standard MCMCChains or FlexiChains
    names_ch = get_chain_names(chain)
    
    # Use a case-insensitive or exact regex match for indexing
    regex = Regex("^$base_name\\\\[(\\\\d+)\\\\]")
    matched_names = filter(n -> occursin(regex, n), names_ch)

    if isempty(matched_names)
        # Fallback for scalar/missing params: return zero matrix
        return zeros(size(chain, 1), len)
    end

    # Sort matched names by the integer index to ensure [1], [2], ... order
    sort!(matched_names, by = n -> parse(Int, match(regex, n).captures[1]))
    
    # Extract data using Symbol indexing (supported by both types)
    return hcat([vec(chain[Symbol(n)].data) for n in matched_names]...)
end 



function generate_spectral_w_from_magnitude(freqs_x, freqs_y, magnitude_spectrum, M_rff_count)
"""
    generate_spectral_w_from_magnitude(freqs_x, freqs_y, magnitude_spectrum, M_rff_count)

Generates 2D RFF weights W by sampling frequencies from the provided 2D magnitude spectrum.

Args:
    freqs_x: Vector of x-dimension frequencies.
    freqs_y: Vector of y-dimension frequencies.
    magnitude_spectrum: 2D array of magnitude values corresponding to freqs_x, freqs_y.
    M_rff_count: Number of RFF features to generate.

Returns:
    A 2 x M_rff_count matrix for W_fixed.
"""
    # Flatten frequency grids and magnitude spectrum into 1D arrays for sampling
    all_freqs_x = repeat(freqs_x, inner=length(freqs_y))
    all_freqs_y = repeat(freqs_y, outer=length(freqs_x))
    all_magnitudes = vec(magnitude_spectrum)

    # Normalize magnitudes to form a probability distribution
    # Add a small constant to magnitudes before normalization to prevent division by zero for zero probabilities.
    probabilities = (all_magnitudes .+ 1e-9) ./ sum(all_magnitudes .+ 1e-9)

    # Sample M_rff_count indices based on probabilities
    # StatsBase.sample expects Weights from non-negative numbers
    sampled_indices = sample(1:length(probabilities), Weights(probabilities), M_rff_count, replace=true)

    W_fixed = Matrix{Float64}(undef, 2, M_rff_count)
    for i in 1:M_rff_count
        idx = sampled_indices[i]
        W_fixed[1, i] = all_freqs_x[idx] *  2pi # Scale by  2pi to match RFF convention (often ω'x)
        W_fixed[2, i] = all_freqs_y[idx] *  2pi
    end

    return W_fixed
end
 

# Helper to create AR1 precision matrix
function ar1_precision(n, rho, sigma_e)
    # Get the type of the parameters, which will be ForwardDiff.Dual during AD
    # or Float64 when not differentiating
    T = typeof(rho)
    Q = spzeros(T, n, n) # Explicitly create a sparse matrix that can hold Dual numbers

    # Main diagonal
    Q[1, 1] = one(T) # Use one(T) to ensure type compatibility
    for i in 2:(n - 1)
        Q[i, i] = one(T) + rho^2
    end
    Q[n, n] = one(T)
    # Off-diagonals
    for i in 1:(n - 1)
        Q[i, i + 1] = -rho
        Q[i + 1, i] = -rho
    end
    # Ensure division also uses the correct type, and result type of `inv(sigma_e^2)` matches T
    return (one(T) / sigma_e^2) .* Q
end

# Helper to create AR1 covariance matrix
function ar1_covariance_matrix(times::Vector{<:Real}, rho::Real, sigma_e::Real)
    n = length(times)
    T = typeof(rho) # Get the type of the parameters
    C = Matrix{T}(undef, n, n) # Initialize matrix with this type
    for i in 1:n
        for j in 1:n
            C[i, j] = sigma_e^2 * rho^abs(times[i] - times[j])
        end
    end
    return C
end

# Helper to create AR1 cross-covariance matrix
function ar1_cross_covariance_matrix(times_a::Vector{<:Real}, times_b::Vector{<:Real}, rho::Real, sigma_e::Real)
    na = length(times_a)
    nb = length(times_b)
    T = typeof(rho) # Get the type of the parameters
    C = Matrix{T}(undef, na, nb) # Initialize matrix with this type
    for i in 1:na
        for j in 1:nb
            C[i, j] = sigma_e^2 * rho^abs(times_a[i] - times_b[j])
        end
    end
    return C
end

# Helper for Kronecker AR1 x Matern Sampling
function kron_ar1_matern_sample(Ns, Nt, unique_s, ls_s, sigma_s, rho_t, sigma_t_noise, noise_vec, noise=1e-4)
    # Spatial Matern 3/2 Precision
    k_s = Matern32Kernel() ∘ ScaleTransform(inv(ls_s))
    K_s = Symmetric(sigma_s^2 * kernelmatrix(k_s, RowVecs(unique_s)) + noise*I )
    Q_s = sparse(inv(K_s))

    # Temporal AR1 Precision
    Q_t = ar1_precision(Nt, rho_t, sigma_t_noise)

    # Kronecker Product Q = Qt ⊗ Qs
    Q_full = Symmetric( kron(Q_t, Q_s) + noise*I)

    # Explicitly ensure symmetry for sparse matrix before Cholesky decomposition
    # Convert to dense Matrix to avoid SparseArrays.CHOLMOD incompatibility with ForwardDiff.Dual
    L_q = cholesky(Symmetric(Matrix(Q_full) + noise*I)) # Increased noise

    # Correctly extract the lower triangular factor for dense Cholesky
    return L_q.L' \ noise_vec
end
 

function prepare_fft_grid(pts, values; grid_res=64, pad_factor=2)
    # 1. Define the bounding box
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    xmin, xmax = minimum(xs), maximum(xs)
    ymin, ymax = minimum(ys), maximum(ys)

    # 2. Map points to a grid
    # Use the length of the shorter input to prevent BoundsError
    n_limit = min(length(pts), length(values))
    grid = zeros(grid_res, grid_res)

    for i in 1:n_limit
        p = pts[i]
        ix = Int(floor((p[1] - xmin) / (xmax - xmin + 1e-6) * (grid_res - 1))) + 1
        iy = Int(floor((p[2] - ymin) / (ymax - ymin + 1e-6) * (grid_res - 1))) + 1
        grid[ix, iy] = values[i]
    end

    # 3. Apply Zero-Padding
    padded_res = grid_res * pad_factor
    padded_grid = zeros(padded_res, padded_res)

    start_idx = Int(grid_res / 2)
    padded_grid[start_idx:start_idx+grid_res-1, start_idx:start_idx+grid_res-1] .= grid

    return padded_grid, (xmin, xmax, ymin, ymax)
end
 



#####



abstract type AbstractArchitecture end
struct UnivariateArchitecture <: AbstractArchitecture end
struct MultivariateArchitecture <: AbstractArchitecture end
struct MultioutcomeArchitecture <: AbstractArchitecture end
struct MultifidelityArchitecture <: AbstractArchitecture end
struct ComplexArchitecture <: AbstractArchitecture end
struct UnknownArchitecture <: AbstractArchitecture end

function get_architecture(model_arch::String)
    # Map specific manifold structures to consolidated architectural categories
    if model_arch in ["univariate"]
        return UnivariateArchitecture()
    elseif model_arch in ["multivariate", "joint"]
        return MultivariateArchitecture()
    elseif model_arch in ["multioutcome"]
        return MultioutcomeArchitecture()
    elseif model_arch in ["multifidelity"] # , "A13", "A25"
        return MultifidelityArchitecture()
    elseif model_arch in ["complex"]  # "deepGP", "warping", "mosaic", "fft", "svgp", "nystrom"
        return ComplexArchitecture()
    else
        return UnknownArchitecture()
    end
end

abstract type ModelFamily end
struct PoissonFamily <: ModelFamily end
struct GaussianFamily <: ModelFamily end
struct LogNormalFamily <: ModelFamily end
struct BernoulliFamily <: ModelFamily end
struct NegativeBinomialFamily <: ModelFamily end

function get_model_family(model_family::String)
    if model_family == "poisson"
        return PoissonFamily()
    elseif model_family == "gaussian"
        return GaussianFamily()
    elseif model_family == "lognormal"
        return LogNormalFamily()
    elseif model_family == "bernoulli"
        return BernoulliFamily()
    elseif model_family == "negbin"
        return NegativeBinomialFamily()
    else
        error("Unknown model_family: $model_family")
    end
end

function reconstruct_posteriors( chain, M, PS; alpha=0.05)
    arch = get_architecture(M.model_arch)
    fam = get_model_family(M.model_family)
    return _reconstruct(arch, fam, chain, M, PS, alpha)
end

 
function get_chain_names(chain)
    try
        # Primary method: using FlexiChains metadata if available
        return string.(FlexiChains.parameters(chain))
    catch
        # Fallback: Standard MCMCChains names conversion
        return string.(names(chain))
    end
end

function get_model_parameters(m::DynamicPPL.Model)
    # Directly extract names from a model instance sample as a fallback discovery method
    try
        raw_keys = keys(rand(m))
        return map(raw_keys) do k
            replace(string(k), r"[\(\)\"\:]" => "")
        end
    catch
        return String[]
    end
end

function check_has_parameter(m::DynamicPPL.Model, param_name::String)
    all_p = get_model_parameters(m)
    return any(n -> n == param_name || startswith(n, "$param_name["), all_p)
end
 

function _process_ll_and_predictions(fam, eta, chain, M, N_tot, N_samples, y_sigma_samples=nothing, y_obs_custom=nothing)
    denoised = zeros(N_tot, N_samples)
    noisy = zeros(N_tot, N_samples)
    log_lik = zeros(N_samples, M.N_obs)

    name_strs = string.(FlexiChains.parameters(chain))

    for j in 1:N_samples
        # Extract volatility (Heteroskedastic vs Homoskedastic)
        local sig_y
        if !isnothing(y_sigma_samples)
            sig_y = y_sigma_samples[:, j]
        elseif "y_sigma" in name_strs
            sig_y = vec(chain[:y_sigma].data[j])
        elseif "y_sigma_const" in name_strs
            sig_val = Float64(chain[:y_sigma_const].data[j])
            sig_y = fill(sig_val, N_tot)
        else
            sig_y = fill(1.0, N_tot)
        end

        for i in 1:N_tot
            mu_eta = eta[i, j]

            # --- Link Functions ---
            mu = if fam isa PoissonFamily || fam isa NegativeBinomialFamily
                clamp(exp(mu_eta), 1e-10, 1e9) # Prevent Inf/NaN for Poisson sampling
            elseif fam isa BinomialFamily
                logistic(mu_eta)
            elseif fam isa LogNormalFamily
                mu_eta # Latent is mean of log(y)
            else
                mu_eta
            end

            denoised[i, j] = mu

            # --- Likelihood Calculation (Audit Training Only) ---
            if i <= M.N_obs
                y_vals_src = isnothing(y_obs_custom) ? M.y_obs : y_obs_custom
                y_val = y_vals_src[i]
                log_lik[j, i] = if fam isa PoissonFamily; logpdf(Poisson(mu), y_val)
                               elseif fam isa GaussianFamily; logpdf(Normal(mu, sig_y[i]), y_val)
                               elseif fam isa BinomialFamily; logpdf(Binomial(M.trials[i], mu), y_val)
                               elseif fam isa LogNormalFamily; logpdf(LogNormal(mu, sig_y[i]), y_val)
                               elseif fam isa NegativeBinomialFamily
                                   r_val = "r_nb" in name_strs ? chain[:r_nb].data[j] : 1.0
                                   prob = r_val / (r_val + mu)
                                   logpdf(NegativeBinomial(r_val, prob), y_val)
                               else 0.0 end
            end

            # --- Posterior Predictive Sampling ---
            noisy[i, j] = if fam isa GaussianFamily; mu + randn() * sig_y[i]
                          elseif fam isa LogNormalFamily; rand(LogNormal(mu, sig_y[i]))
                          elseif fam isa PoissonFamily; rand(Poisson(mu))
                          elseif fam isa BinomialFamily; rand(Binomial(M.trials[is_obs ? i : 1], mu))
                          elseif fam isa NegativeBinomialFamily
                               r_val = "r_nb" in name_strs ? chain[:r_nb].data[j] : 1.0 # r_nb is the parameter name from the chain
                               rand(NegativeBinomial(r_val, r_val / (r_val + mu)))
                          else mu end
        end
    end
    return denoised, noisy, log_lik
end


 
function _extract_volatility(chain, name_strs, N_tot, N_samples, outcome_idx=nothing)
    """
    Shared utility to extract volatility surfaces (y_sigma).
    Supports both univariate (y_sigma / y_sigma_const) 
    and multivariate (y_sigma_k[k] / y_sigma_const_k[k]) conventions.
    """
    y_sig_samples = zeros(N_tot, N_samples)
    
    for j in 1:N_samples
        local sig_y
        if isnothing(outcome_idx)
            # Univariate Logic
            if "y_sigma" in name_strs
                sig_y = vec(chain[:y_sigma].data[j])
            elseif "y_sigma_const" in name_strs
                sig_val = Float64(chain[:y_sigma_const].data[j])
                sig_y = fill(sig_val, N_tot)
            else
                sig_y = fill(1.0, N_tot)
            end
        else
            # Multivariate Logic for outcome k
            v_key = "y_sigma_k[$outcome_idx]"
            c_key = "y_sigma_const_k[$outcome_idx]"
            
            if v_key in name_strs
                # Vectorized SV: Extracted for training indices, extended to PS via mean
                sig_vec = vec(chain[Symbol(v_key)].data[j])
                sig_y = vcat(sig_vec, fill(mean(sig_vec), N_tot - length(sig_vec)))
            elseif c_key in name_strs
                sig_val = Float64(chain[Symbol(c_key)].data[j])
                sig_y = fill(sig_val, N_tot)
            else
                sig_y = fill(1.0, N_tot)
            end
        end
        y_sig_samples[:, j] = sig_y
    end
    return y_sig_samples
end



function _compute_waic(log_lik)
    nsamples, nobs = size(log_lik)
    lppd = sum(logsumexp(log_lik[:, i]) - log(nsamples) for i in 1:nobs)
    p_waic = sum(var(log_lik[:, i]) for i in 1:nobs)
    return -2 * (lppd - p_waic)
end

function _extract_beta_cov(all_names, chain, M, N_samples, alpha)
    # Identify all categorical covariate groups present in the chain
    # Matches patterns like "beta_cov[1]", "beta_cov[2]", etc.
    cov_matches = unique(map(m -> m.captures[1], filter(!isnothing, match.(r"beta_cov\[(\d+)\]", all_names))))
    
    if isempty(cov_matches)
        return nothing
    end

    # Parse indices and sort to ensure sequential processing
    cov_indices = sort(parse.(Int, cov_matches))
    
    results = []
    for k in cov_indices
        base_name = "beta_cov[$k]"
        # Use the robust get_params_vector helper to extract the full vector for this covariate group
        raw_vals = get_params_vector(chain, base_name, M.N_cat)
        
        if !all(raw_vals .== 0) # Only process if data was actually found
            # Reshape for summarize_array (N_categories x 1 x N_samples)
            summ = summarize_array(reshape(raw_vals', M.N_cat, 1, N_samples); alpha=alpha)
            push!(results, summ)
        end
    end

    return isempty(results) ? nothing : results
end



function create_prediction_surface(basis_df, observations_df, au; fixed_df=nothing, lambda=2.0, st_covs=[:u1, :u2, :u3], z_cov=[:z])

    # 1. Initialization and Join with Observations
    if hasproperty(basis_df, :u_idx) & hasproperty(observations_df, :u_idx) 
        mergeon = [:s_idx, :t_idx, :u_idx]
    else 
        mergeon = [:s_idx, :t_idx]
    end
        
    surface = leftjoin(basis_df, observations_df, on =mergeon)

    # 2. Dynamic Merge of Fixed Effects
    # This identifies present indices (Space, Time, Season) in fixed_df and merges accordingly
    if !isnothing(fixed_df)
        join_cols = intersect( mergeon, propertynames(fixed_df))
        if !isempty(join_cols)
            # Merge fixed effects; 'on' is determined by indices available in fixed_df
            surface = leftjoin(surface, fixed_df, on = join_cols, makeunique=true)
        end
    end

    centroids = au.centroids
    n_units = length(centroids)
    all_covs = [z_cov; st_covs]

    # Helper for safe column-wise mean
    get_col_mean(col) = begin
        v = filter(x -> !ismissing(x) && !isnan(x), col)
        return isempty(v) ? NaN : mean(v)
    end

    # 3. Precompute Global Means and Distance Weights
    global_means = nothing
    if all(in(mergeon, propertynames(surface)) ) 
        global_means = Dict(c => get_col_mean(surface[!, c]) for c in all_covs)
    end

    dist_mat = [sqrt(sum((c1 .- c2).^2)) for c1 in centroids, c2 in centroids]
    weight_mat = exp.(-dist_mat.^2 ./ (2 * lambda^2))
    for i in 1:n_units
        weight_mat[i, :] ./= (sum(weight_mat[i, :]) + 1e-9)
    end

    # Spatiotemporal Imputation via Weighted Distance (Gaussian Kernel)
    if all(in(mergeon, propertynames(surface)) ) 
        for t in unique(surface.t_idx), u_type in unique(surface.u_idx)
            mask = (surface.t_idx .== t) .& (surface.u_idx .== u_type)
            if !any(mask); continue; end

            slice_indices = findall(mask)
            sort!(slice_indices, by = idx -> surface.s_idx[idx])
            current_s_ids = surface.s_idx[slice_indices]

            for c in st_covs
                raw_vals = surface[slice_indices, c]
                missing_mask = ismissing.(raw_vals) .| isnan.(raw_vals)
                if !any(missing_mask) || all(missing_mask); continue; end

                non_missing_idx = findall(.!missing_mask)
                for m_idx in findall(missing_mask)
                    target_s = current_s_ids[m_idx]
                    weights = weight_mat[target_s, current_s_ids[non_missing_idx]]
                    w_sum = sum(weights)
                    if w_sum > 1e-6
                        surface[slice_indices[m_idx], c] = sum(weights .* raw_vals[non_missing_idx]) / w_sum
                    end
                end
            end
        end

        # Temporal Persistence (LOCF Fallback for remaining ST gaps)
        sort!(surface, mergeon)

        for c in st_covs
            for unit_df in groupby(surface, :s_idx)
                for i in 1:nrow(unit_df)
                    if ismissing(unit_df[i, c]) || isnan(unit_df[i, c])
                        prev_idx = i - 1
                        while prev_idx >= 1
                            if !ismissing(unit_df[prev_idx, c]) && !isnan(unit_df[prev_idx, c])
                                unit_df[i, c] = unit_df[prev_idx, c]
                                break
                            end
                            prev_idx -= 1
                        end
                    end
                end
            end
        end
    end

    #  Final Global Fallback
    if !isnothing(global_means)
        for c in all_covs
            surface[!, c] = map(x -> (ismissing(x) || isnan(x)) ? global_means[c] : x, surface[!, c])
        end
    end

    return surface
end


function _reconstruct(arch::MultioutcomeArchitecture, fam::ModelFamily, chain, M, PS, alpha)
    N_obs_points = M.N_obs
    N_outcomes = M.N_outcomes
    N_areas, N_time = M.N_areas, M.N_time
    N_season = M.N_season
    N_PS = isnothing(PS) ? 0 : size(PS, 1)
    N_tot_points = N_obs_points + N_PS
    
    N_samples = size(chain, 1)
    name_strs = string.(FlexiChains.parameters(chain))

    # Storage for outcome-wise summaries aligned with _reconstruct_template output
    outcome_results = Vector{Any}(undef, N_outcomes)

    # Global components
    phi_zi_samples = "phi_zi" in name_strs ? vec(chain[:phi_zi].data) : zeros(N_samples)
    
    for k in 1:N_outcomes
        # --- 1. MANIFOLD RECOVERY FOR OUTCOME K ---
        y_sigma_samples_k = _extract_volatility(chain, name_strs, N_tot_points, N_samples, k)

        s_eta_tot = zeros(N_tot_points, N_samples)
        s_eta_obs_denoised = zeros(N_obs_points, N_samples)
        t_eta_tot = zeros(N_time, N_samples)
        u_eta_tot = zeros(N_season, N_samples)
        st_eta_tot = zeros(N_areas, N_time, N_samples)
        fixed_effects_samples_k = zeros(M.N_fixed, N_samples)
        
        # Extract outcome-specific scales
        s_sigma_k = vec(chain[Symbol("s_sigma[$k]")].data)
        t_sigma_k = vec(chain[Symbol("t_sigma[$k]")].data)
        st_sigma_k = "st_sigma" in name_strs ? vec(chain[Symbol("st_sigma[$k]")].data) : ones(N_samples)
        u_sigma_k = "u_sigma" in name_strs ? vec(chain[Symbol("u_sigma[$k]")].data) : zeros(N_samples)
        
        for s in 1:N_samples
            # --- A. SPATIAL MANIFOLD AUDIT (All 14 Structures) ---
            local field_structured_k = zeros(N_areas)
            local field_noisy_k = zeros(N_areas)

            if M.model_space == "bym2"
                rho_k = "s_rho" in name_strs ? clamp(chain[Symbol("s_rho[$k]")].data[s], 0.0, 1.0) : 1.0
                icar_k = vec(chain[Symbol("s_icar_k[$k]")].data[s])
                iid_k = vec(chain[Symbol("s_iid_k[$k]")].data[s])
                field_structured_k = sqrt(rho_k) .* icar_k .* s_sigma_k[s]
                field_noisy_k = field_structured_k .+ (sqrt(1 - rho_k) .* iid_k .* s_sigma_k[s])
            elseif M.model_space in ["besag", "icar"]
                field_structured_k = vec(chain[Symbol("s_icar_k[$k]")].data[s]) .* s_sigma_k[s]
                field_noisy_k = field_structured_k
            elseif M.model_space == "sar"
                field_noisy_k = vec(chain[Symbol("s_sar_raw_k[$k]")].data[s]) .* s_sigma_k[s]
                field_structured_k = field_noisy_k
            elseif M.model_space == "bgcn"
                D_inv_sqrt = Diagonal(1.0 ./ sqrt.(vec(sum(M.W, dims=2)) .+ M.noise))
                W_norm = D_inv_sqrt * M.W * D_inv_sqrt
                field_noisy_k = (W_norm * vec(chain[Symbol("s_icar_k[$k]")].data[s])) .* s_sigma_k[s]
                field_structured_k = field_noisy_k
            elseif M.model_space == "dag"
                field_noisy_k = vec(chain[Symbol("s_dag_raw_k[$k]")].data[s]) .* s_sigma_k[s]
                field_structured_k = field_noisy_k
            elseif M.model_space == "local"
                mu_cl = vec(chain[Symbol("mu_clusters_k[$k]")].data[s])
                field_structured_k = mu_cl[M.cluster_assignments]
                field_noisy_k = field_structured_k .+ (vec(chain[Symbol("s_eta_raw_k[$k]")].data[s]) .* s_sigma_k[s])
            elseif M.model_space == "mosaic"
                mu_loc = vec(chain[Symbol("mu_local_k[$k]")].data[s])
                field_structured_k = mu_loc[M.cluster_assignments] .* s_sigma_k[s]
                field_noisy_k = field_structured_k
            elseif M.model_space == "warping"
                w_warp = vec(chain[Symbol("w_warp_k[$k]")].data[s])
                warp_proj = (M.s_obs * M.W_fixed) .+ M.b_fixed'
                field_structured_k = (sqrt(2.0 / M.M_rff) .* cos.(warp_proj) * w_warp) .* s_sigma_k[s]
                field_noisy_k = field_structured_k
            elseif M.model_space == "fft"
                field_noisy_k = (M.s_Q \ vec(chain[Symbol("s_spectral_raw_k[$k]")].data[s])) .* s_sigma_k[s]
                field_structured_k = field_noisy_k
            elseif M.model_space == "iid"
                field_noisy_k = vec(chain[Symbol("s_iid_k[$k]")].data[s]) .* s_sigma_k[s]
                field_structured_k = zeros(N_areas)
            else # Catch-all for RFF/GP/SVC which map directly to s_eta_scaled if indexed
                field_noisy_k = zeros(N_areas)
                field_structured_k = zeros(N_areas)
            end

            # Map to Obs + PS
            s_idx_obs = M.s_idx isa AbstractVector ? M.s_idx : fill(M.s_idx[1], N_obs_points)
            s_eta_tot[1:N_obs_points, s] = field_noisy_k[s_idx_obs]
            s_eta_obs_denoised[:, s] = field_structured_k[s_idx_obs]
            if N_PS > 0
                s_idx_ps = PS.s_idx isa AbstractVector ? PS.s_idx : fill(PS.s_idx[1], N_PS)
                s_eta_tot[(N_obs_points+1):end, s] = field_noisy_k[s_idx_ps]
            end

            # --- B. TEMPORAL FIELD ---
            if M.model_time == "ar1"
                t_eta_tot[:, s] = vec(chain[Symbol("t_raw_k[$k]")].data[s]) .* t_sigma_k[s]
            elseif M.model_time == "rw2"
                t_eta_tot[:, s] = vec(chain[Symbol("t_raw_k[$k]")].data[s]) # Sigma built into precision for RW2
            elseif M.model_time == "gp"
                t_eta_tot[:, s] = vec(chain[Symbol("t_gp_k[$k]")].data[s])
            elseif M.model_time == "harmonic"
                t_a, t_b = chain[Symbol("t_alpha_k[$k]")].data[s], chain[Symbol("t_beta_k[$k]")].data[s]
                t_eta_tot[:, s] = (t_a .* sin.(M.t_angle) .+ t_b .* cos.(M.t_angle)) .* t_sigma_k[s]
            end

            # --- C. SEASONAL FIELD ---
            if M.model_season != "none"
                if M.model_season == "harmonic"
                    u_a, u_b = chain[Symbol("u_alpha_k[$k]")].data[s], chain[Symbol("u_beta_k[$k]")].data[s]
                    u_steps = 1:N_season
                    u_eta_tot[:, s] = (u_a .* sin.(2π .* u_steps ./ 12.0) .+ u_b .* cos.(2π .* u_steps ./ 12.0)) .* u_sigma_k[s]
                else
                    u_eta_tot[:, s] = vec(chain[Symbol("u_raw_k[$k]")].data[s]) .* u_sigma_k[s]
                end
            end

            # --- D. SPACE-TIME INTERACTION ---
            if M.model_st > 0
                st_raw_k = vec(chain[Symbol("st_raw_k[$k]")].data[s])
                st_eta_tot[:, :, s] = reshape(st_raw_k .* st_sigma_k[s], N_areas, N_time)
            end
        end

        # --- 2. PREDICTOR ASSEMBLY ---
        eta_k = zeros(N_tot_points, N_samples)
        for s in 1:N_samples
            st_mat_k = st_eta_tot[:, :, s]
            for i in 1:N_tot_points
                is_obs = i <= N_obs_points
                src = is_obs ? M : PS
                idx_adj = is_obs ? i : i - N_obs_points
                
                a = src.s_idx isa AbstractVector ? src.s_idx[idx_adj] : src.s_idx
                t = src.t_idx isa AbstractVector ? src.t_idx[idx_adj] : src.t_idx
                u = src.u_idx isa AbstractVector ? src.u_idx[idx_adj] : src.u_idx
                
                val = (is_obs ? M.log_offset[i, k] : 0.0) + s_eta_tot[i, s] + t_eta_tot[t, s] + st_mat_k[a, t]
                if M.model_season != "none"; val += u_eta_tot[u, s]; end
                
                # Fixed Effects Logic for Multivariate
                if M.N_fixed > 0
                    p_beta_k = [chain[Symbol("d_beta_k[$k][$d]")].data[s] for d in 1:M.N_fixed]
                    fixed_effects_samples_k[:, s] = p_beta_k
                    val += dot(p_beta_k, is_obs ? M.fixed[i, :] : PS.fixed[idx_adj, :])
                end
                
                eta_k[i, s] = val
            end
        end

        # --- 3. SUMMARIES PER OUTCOME ---
        denoised_mu, noisy_y, log_lik = _process_ll_and_predictions(fam, eta_k, chain, M, N_tot_points, N_samples, y_sigma_samples_k, M.y_obs[:, k])
        
        outcome_results[k] = (
            spatial_structured = summarize_array(reshape(s_eta_obs_denoised, N_obs_points, 1, N_samples); alpha=alpha),
            spatial_noisy = summarize_array(reshape(s_eta_tot[1:N_obs_points, :], N_obs_points, 1, N_samples); alpha=alpha),
            temporal = summarize_array(reshape(t_eta_tot, N_time, 1, N_samples); alpha=alpha),
            seasonal = summarize_array(reshape(u_eta_tot, N_season, 1, N_samples); alpha=alpha),
            volatility = summarize_array(reshape(y_sigma_samples_k, N_tot_points, 1, N_samples); alpha=alpha),
            fixed_effects = M.N_fixed > 0 ? summarize_array(reshape(fixed_effects_samples_k, M.N_fixed, 1, N_samples); alpha=alpha) : nothing,
            predictions_observed = summarize_array(reshape(denoised_mu[1:N_obs_points, :], N_obs_points, 1, N_samples); alpha=alpha),
            predictions_denoised = N_PS > 0 ? summarize_array(reshape(denoised_mu[(N_obs_points+1):end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
            predictions_noisy = N_PS > 0 ? summarize_array(reshape(noisy_y[(N_obs_points+1):end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
            waic = _compute_waic(log_lik[:, 1:N_obs_points]),
            family = fam
        )
    end

    return (
        outcomes = outcome_results,
        phi_zi = summarize_array(reshape(phi_zi_samples, 1, 1, N_samples); alpha=alpha),
        architecture = arch
    )
end


function _reconstruct(arch::MultivariateArchitecture, fam::ModelFamily, chain, M, PS, alpha)
    # Description: Reconstructs factor-based multivariate spatiotemporal fields.
    # Maps latent factor scores back to outcome space via the learned PCA loadings (Householder).

    N_obs, N_outcomes = size(M.y_obs)
    N_factors = get(M, :N_factors, 2)
    N_areas, N_time = M.N_areas, M.N_time
    N_PS = isnothing(PS) ? 0 : size(PS, 1)
    N_tot = N_obs + N_PS
    N_samples = size(chain, 1)
    name_strs = string.(FlexiChains.parameters(chain))

    # 1. PCA & Loadings Recovery
    U_samples = zeros(N_outcomes, N_factors, N_samples)
    Kmat_samples = zeros(N_outcomes, N_outcomes, N_samples)
    nvh = Int(N_outcomes * N_factors - N_factors * (N_factors - 1) / 2)
    v_all = get_params_vector(chain, "v", nvh)

    for s in 1:N_samples
        pca_sd = [chain[Symbol("pca_sd[$k]")].data[s] for k in 1:N_factors]
        pca_pdef_sd = chain[:pca_pdef_sd].data[s]
        Kmat, _, U = householder_transform(v_all[s, :], N_outcomes, N_factors, M.ltri, pca_sd, pca_pdef_sd, M.noise)
        U_samples[:, :, s] = U
        Kmat_samples[:, :, s] = Kmat
    end

    # 2. Factor-wise Field Recovery
    factor_fields = zeros(N_tot, N_factors, N_samples)
    for k in 1:N_factors
        for s in 1:N_samples
            # Spatial Component
            s_sigma_k = chain[Symbol("s_sigma_k[$k]")].data[s]
            local field_k = zeros(N_areas)
            if M.model_space == "bym2"
                rho_k = clamp(chain[Symbol("s_rho_k[$k]")].data[s], 0.0, 1.0)
                field_k = s_sigma_k .* (sqrt(rho_k) .* vec(chain[Symbol("s_icar_k[$k]")].data[s]) .+ 
                          sqrt(1 - rho_k) .* vec(chain[Symbol("s_iid_k[$k]")].data[s]))
            elseif M.model_space == "besag"
                field_k = vec(chain[Symbol("s_icar_k[$k]")].data[s]) .* s_sigma_k
            elseif M.model_space == "sar"
                field_k = vec(chain[Symbol("s_eta_k[$k]")].data[s])
            end

            # Temporal Component
            t_sigma_k = chain[Symbol("t_sigma_k[$k]")].data[s]
            local trend_k = zeros(N_time)
            if M.model_time == "ar1"
                trend_k = vec(chain[Symbol("t_raw_k[$k]")].data[s]) .* t_sigma_k
            elseif M.model_time == "rw2"
                trend_k = vec(chain[Symbol("t_eta_k[$k]")].data[s])
            end

            # Assembly
            for i in 1:N_tot
                is_obs = i <= N_obs
                src = is_obs ? M : PS
                idx_adj = is_obs ? i : i - N_obs
                a = src.s_idx isa AbstractVector ? src.s_idx[idx_adj] : src.s_idx
                t = src.t_idx isa AbstractVector ? src.t_idx[idx_adj] : src.t_idx
                factor_fields[i, k, s] = field_k[a] + trend_k[t]
            end
        end
    end

    # 3. Outcome Predictor Synthesis
    eta = zeros(N_tot, N_outcomes, N_samples)
    c_eta_all = "c_eta" in name_strs ? get_params_vector(chain, "c_eta", M.N_cat)' : zeros(M.N_cat, N_samples)
    
    fixed_effects_samples = zeros(M.N_fixed, N_outcomes, N_samples)
    if M.N_fixed > 0
        for k in 1:N_outcomes, d in 1:M.N_fixed
            p_name = Symbol("d_beta[$d,$k]")
            if p_name in Symbol.(name_strs)
                for s in 1:N_samples; fixed_effects_samples[d, k, s] = chain[p_name].data[s]; end
            end
        end
    end

    for s in 1:N_samples
        mu_proj = factor_fields[:, :, s] * U_samples[:, :, s]'
        for i in 1:N_tot
            is_obs = i <= N_obs
            idx_adj = is_obs ? i : i - N_obs
            off = is_obs ? M.log_offset[idx_adj, :] : zeros(N_outcomes)
            for k in 1:N_outcomes
                val = off[k] + mu_proj[i, k]
                if "c_eta" in name_strs
                    val += c_eta_all[is_obs ? M.cov_indices[idx_adj, 1] : PS.cov_indices[idx_adj, 1], s]
                end
                if M.N_fixed > 0
                    fixed_mat = is_obs ? M.fixed : PS.fixed
                    for d in 1:M.N_fixed
                        dm_val = fixed_mat isa AbstractMatrix ? fixed_mat[idx_adj, d] : (fixed_mat isa AbstractVector ? fixed_mat[idx_adj] : fixed_mat)
                        val += fixed_effects_samples[d, k, s] * dm_val
                    end
                end
                eta[i, k, s] = val
            end
        end
    end

    # 4. Summaries
    outcome_results = Vector{Any}(undef, N_outcomes)
    for k in 1:N_outcomes
        y_sig_k = sqrt.(Kmat_samples[k, k, :])
        y_sigma_samples_k = repeat(y_sig_k', N_tot, 1)
        denoised_k, noisy_k, log_lik_k = _process_ll_and_predictions(fam, eta[:, k, :], chain, M, N_tot, N_samples, y_sigma_samples_k, M.y_obs[:, k])
        outcome_results[k] = (
            spatial_noisy = summarize_array(reshape(eta[1:N_obs, k, :], N_obs, 1, N_samples); alpha=alpha),
            volatility = summarize_array(reshape(y_sigma_samples_k, N_tot, 1, N_samples); alpha=alpha),
            fixed_effects = M.N_fixed > 0 ? summarize_array(reshape(fixed_effects_samples[:, k, :], M.N_fixed, 1, N_samples); alpha=alpha) : nothing,
            predictions_observed = summarize_array(reshape(denoised_k[1:N_obs, :], N_obs, 1, N_samples); alpha=alpha),
            predictions_denoised = N_PS > 0 ? summarize_array(reshape(denoised_k[(N_obs+1):end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
            waic = _compute_waic(log_lik_k), family = fam
        )
    end
    return (outcomes = outcome_results, loadings = summarize_array(U_samples; alpha=alpha), architecture = arch)
end




function _reconstruct(arch::MultifidelityArchitecture, fam::ModelFamily, chain, M, PS, alpha)
    N_obs = M.N_obs
    N_PS = isnothing(PS) ? 0 : size(PS, 1)
    N_areas, N_time = M.N_areas, M.N_time
    N_tot = N_obs + N_PS
    N_samples = size(chain, 1)
    name_strs = string.(FlexiChains.parameters(chain))

    # 1. MANIFOLD RECOVERY (Latent Field Synthesis)
    s_eta_tot = zeros(N_tot, N_samples)
    s_eta_obs_denoised = zeros(N_obs, N_samples)
    s_eta_obs_noisy = zeros(N_obs, N_samples)
    t_eta_tot = zeros(N_time, N_samples)
    u_eta_tot = zeros(M.N_season, N_samples)

    # Multi-fidelity specific latent fields
    z_latent_samples = zeros(size(M.z_obs, 1), N_samples)
    w_latent_samples = zeros(size(M.w_obs, 1), 3, N_samples)

    # Extract Volatility Surface
    y_sigma_samples = _extract_volatility(chain, name_strs, N_tot, N_samples)

    # Extract covariate effects
    c_eta_all = zeros(M.N_cat, N_samples)
    if "c_eta" in name_strs
        for s in 1:N_samples
            c_eta_all[:, s] = vec(chain[:c_eta].data[s])
        end
    end

    # Extract fixed effects
    fixed_effects_samples = zeros(M.N_fixed, N_samples)
    if M.N_fixed > 0
        for d in 1:M.N_fixed
            p_name = Symbol("d_beta[$d]")
            if p_name in Symbol.(name_strs)
                for s in 1:N_samples
                    fixed_effects_samples[d, s] = chain[p_name].data[s]
                end
            end
        end
    end

    for s in 1:N_samples
        # Spatial/Temporal Recovery
        s_sigma = "s_sigma" in name_strs ? chain[:s_sigma].data[s] : 1.0
        s_rho = "s_rho" in name_strs ? clamp(chain[:s_rho].data[s], 0.0, 1.0) : 1.0
        s_icar = vec(chain[:s_icar].data[s])
        s_iid = vec(chain[:s_iid].data[s])
        field_structured = s_sigma .* sqrt(s_rho) .* s_icar
        field_noisy = field_structured .+ (s_sigma .* sqrt(1 - s_rho) .* s_iid)

        t_sigma = "t_sigma" in name_strs ? chain[:t_sigma].data[s] : 1.0
        t_raw = vec(chain[:t_raw].data[s])
        field_temporal = t_raw .* t_sigma

        # RFF Latent Fields
        z_latent_samples[:, s] = vec(chain[:z_latent].data[s])
        for k in 1:3
            w_latent_samples[:, k, s] = vec(chain[Symbol("w_latent_k[$k]")].data[s])
        end

        # Map to total grid
        s_eta_tot[1:N_obs, s] = field_noisy[M.s_idx]
        if N_PS > 0 && !isnothing(PS)
            s_eta_tot[(N_obs+1):end, s] = field_noisy[PS.s_idx]
        end
        s_eta_obs_denoised[:, s] = field_structured[M.s_idx]
        s_eta_obs_noisy[:, s] = field_noisy[M.s_idx]
        t_eta_tot[:, s] = field_temporal

        # Seasonal (if present)
        if "u_raw" in name_strs
            u_eta_tot[:, s] = vec(chain[:u_raw].data[s]) .* ("u_sigma" in name_strs ? chain[:u_sigma].data[s] : 1.0)
        end
    end

    # 2. PREDICTOR ASSEMBLY
    eta = zeros(N_tot, N_samples)
    z_beta_eta = vec(chain[:z_beta_eta].data)
    w_beta_eta = [vec(chain[Symbol("w_beta_eta[$k]")].data) for k in 1:3]

    for s in 1:N_samples
        for i in 1:N_tot
            is_obs = i <= N_obs
            idx_adj = is_obs ? i : i - N_obs
            src = is_obs ? M : PS

            # src can be nothing if i > N_obs and PS is nothing, though N_tot implies PS exists if i > N_obs
            a, t, u = src.s_idx[idx_adj], src.t_idx[idx_adj], src.u_idx[idx_adj]
            off = is_obs ? M.log_offset[i] : 0.0

            val = off + s_eta_tot[i, s] + t_eta_tot[t, s] + u_eta_tot[u, s]
            val += z_latent_samples[idx_adj, s] * z_beta_eta[s]
            for k in 1:3
                val += w_latent_samples[idx_adj, k, s] * w_beta_eta[k][s]
            end
            eta[i, s] = val
        end
    end

    denoised_mu, noisy_y, log_lik_mat = _process_ll_and_predictions(fam, eta, chain, M, N_tot, N_samples, y_sigma_samples, M.y_obs)

    return (
        spatial_structured = summarize_array(reshape(s_eta_obs_denoised, N_obs, 1, N_samples); alpha=alpha),
        spatial_noisy = summarize_array(reshape(s_eta_obs_noisy, N_obs, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(t_eta_tot, N_time, 1, N_samples); alpha=alpha),
        seasonal = summarize_array(reshape(u_eta_tot, M.N_season, 1, N_samples); alpha=alpha),
        volatility = summarize_array(reshape(y_sigma_samples, N_tot, 1, N_samples); alpha=alpha),
        fixed_effects = summarize_array(reshape(fixed_effects_samples, M.N_fixed, 1, N_samples); alpha=alpha),
        covariate_effects = summarize_array(reshape(c_eta_all, M.N_cat, 1, N_samples); alpha=alpha),
        predictions_observed = summarize_array(reshape(denoised_mu[1:N_obs, :], N_obs, 1, N_samples); alpha=alpha),
        z_latent = summarize_array(reshape(z_latent_samples, size(M.z_obs, 1), 1, N_samples); alpha=alpha),
        w_latent = [summarize_array(reshape(w_latent_samples[:, k, :], size(M.w_obs, 1), 1, N_samples); alpha=alpha) for k in 1:3],
        predictions_denoised = N_PS > 0 ? summarize_array(reshape(denoised_mu[(N_obs+1):end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        predictions_noisy = N_PS > 0 ? summarize_array(reshape(noisy_y[(N_obs+1):end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        waic = _compute_waic(log_lik_mat[:, 1:N_obs]),
        family = fam, architecture = arch
    )
end


function _reconstruct(arch::UnivariateArchitecture, fam::ModelFamily, chain, M, PS, alpha)

    N_obs = M.N_obs
    N_PS = isnothing(PS) ? 0 : size(PS, 1)
    N_areas, N_time = M.N_areas, M.N_time
    N_tot = N_obs + N_PS
    N_samples = size(chain, 1)
    name_strs = string.(FlexiChains.parameters(chain))

    # 1. MANIFOLD RECOVERY
    s_eta_tot = zeros(N_tot, N_samples)
    s_eta_obs_denoised = zeros(N_obs, N_samples)
    s_eta_obs_noisy = zeros(N_obs, N_samples)
    t_eta_tot = zeros(N_time, N_samples)
    u_eta_tot = zeros(M.N_season, N_samples)
    st_eta_tot = zeros(N_areas, N_time, N_samples)

    y_sigma_samples = _extract_volatility(chain, name_strs, N_tot, N_samples)
    c_eta_all = zeros(M.N_cat, N_samples)
    if "c_eta" in name_strs
        for s in 1:N_samples
            c_eta_all[:, s] = vec(chain[:c_eta].data[s])
        end
    end

    fixed_effects_samples = zeros(M.N_fixed, N_samples)
    if M.N_fixed > 0
        for d in 1:M.N_fixed
            p_name = Symbol("d_beta[$d]")
            if p_name in Symbol.(name_strs)
                for s in 1:N_samples
                    fixed_effects_samples[d, s] = chain[p_name].data[s]
                end
            end
        end
    end

    for s in 1:N_samples
        s_sigma_sample = "s_sigma" in name_strs ? chain[:s_sigma].data[s] : 1.0
        local field_structured_sample = zeros(N_areas)
        local field_noisy_sample = zeros(N_areas)

        if M.model_space == "bym2"
            rho_sample = "s_rho" in name_strs ? clamp(chain[:s_rho].data[s], 0.0, 1.0) : 1.0
            icar_sample = vec(chain[:s_icar].data[s])
            iid_sample = vec(chain[:s_iid].data[s])
            field_structured_sample = s_sigma_sample .* sqrt(rho_sample) .* icar_sample
            field_noisy_sample = field_structured_sample .+ (s_sigma_sample .* sqrt(1.0 - rho_sample) .* iid_sample)
        elseif M.model_space in ["besag", "icar"]
            field_structured_sample = vec(chain[:s_icar].data[s]) .* s_sigma_sample
            field_noisy_sample = field_structured_sample

        elseif M.model_space == "leroux"
            rho_sample = "s_rho" in name_strs ? clamp(chain[:s_rho].data[s], 0.0, 1.0) : 1.0
            # Leroux uses a specific precision form, but for visualization we extract the latent field
            field_structured_sample = vec(chain[:s_raw].data[s]) .* s_sigma_sample
            field_noisy_sample = field_structured_sample
 
        elseif M.model_space == "sar"
            field_noisy_sample = vec(chain[:s_sar_raw].data[s]) .* s_sigma_sample
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "bgcn"
            D_inv_sqrt = Diagonal(1.0 ./ sqrt.(vec(sum(M.W, dims=2)) .+ M.noise))
            W_norm = D_inv_sqrt * M.W * D_inv_sqrt
            field_noisy_sample = (W_norm * vec(chain[:gcn_weight_raw].data[s])) .* s_sigma_sample
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "fft"
            field_noisy_sample = (M.s_Q \ vec(chain[:s_spectral_raw].data[s])) .* s_sigma_sample
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "svgp"
            field_noisy_sample = (M.Z_inducing_proj * vec(chain[:u_latent].data[s])) .* s_sigma_sample
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "mosaic"
            field_noisy_sample = vec(chain[:mu_local].data[s])[M.cluster_assignments] .* s_sigma_sample
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "warping"
            w_warp = vec(chain[:w_warp].data[s])
            warp_proj = (M.s_obs * M.W_fixed) .+ M.b_fixed'
            field_noisy_sample = (sqrt(2.0 / M.M_rff) .* cos.(warp_proj) * w_warp) .* s_sigma_sample
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "fitc"
            u_ind = vec(chain[:s_inducing].data[s])
            field_noisy_sample = (M.K_XZ * (M.K_ZZ \ u_ind))
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "fitcGP"
            field_noisy_sample = (M.Z_inducing_proj * vec(chain[:u_inducing].data[s])) .* s_sigma_sample
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "nystrom"
            field_noisy_sample = (M.K_nystrom_proj * vec(chain[:v_latent].data[s])) .* s_sigma_sample
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "deepGP"
            field_noisy_sample = vec(chain[:eta_gp].data[s])
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "dag"
            field_noisy_sample = vec(chain[:s_eta].data[s])
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "local"
            mu_cl = vec(chain[:mu_clusters].data[s])
            field_structured_sample = mu_cl[M.cluster_assignments] .+ (vec(chain[:s_eta_raw].data[s]) .* s_sigma_sample)
            field_noisy_sample = field_structured_sample
        elseif M.model_space == "svc"
            field_structured_sample = vec(chain[:svc_raw].data[s]) .* s_sigma_sample
            field_noisy_sample = field_structured_sample
        elseif M.model_space == "iid"
            field_noisy_sample = vec(chain[:s_iid].data[s]) .* s_sigma_sample
            field_structured_sample = zeros(N_areas)
        else
            field_noisy_sample = "s_eta" in name_strs ? vec(chain[:s_eta].data[s]) : zeros(N_areas)
            field_structured_sample = field_noisy_sample
        end

        s_idx_obs = M.s_idx isa AbstractVector ? M.s_idx : fill(M.s_idx[1], N_obs)
        s_eta_tot[1:N_obs, s] = field_noisy_sample[s_idx_obs]
        if N_PS > 0  && !isnothing(PS)
            s_idx_ps = PS.s_idx isa AbstractVector ? PS.s_idx : fill(PS.s_idx[1], N_PS)
            s_eta_tot[(N_obs+1):end, s] = field_noisy_sample[s_idx_ps]
        end
        s_eta_obs_denoised[:, s] = field_structured_sample[s_idx_obs]
        s_eta_obs_noisy[:, s] = field_noisy_sample[s_idx_obs]

        t_sigma_sample = "t_sigma" in name_strs ? chain[:t_sigma].data[s] : 1.0
        if "t_eta" in name_strs; t_eta_tot[:, s] = vec(chain[:t_eta].data[s])
        elseif "t_raw" in name_strs; t_eta_tot[:, s] = vec(chain[:t_raw].data[s]) .* t_sigma_sample
        elseif "t_alpha" in name_strs; t_eta_tot[:, s] = (chain[:t_alpha].data[s] .* sin.(M.t_angle) .+ chain[:t_beta].data[s] .* cos.(M.t_angle)) .* t_sigma_sample
        end

        u_sigma_sample = "u_sigma" in name_strs ? chain[:u_sigma].data[s] : 1.0
        if "u_eta" in name_strs; u_eta_tot[:, s] = vec(chain[:u_eta].data[s])
        elseif "u_raw" in name_strs; u_eta_tot[:, s] = vec(chain[:u_raw].data[s]) .* u_sigma_sample
        elseif "u_alpha" in name_strs
            u_steps = 1:M.N_season
            u_eta_tot[:, s] = (chain[:u_alpha].data[s] .* sin.( 2pi .* u_steps ./ 12.0) .+ chain[:u_beta].data[s] .* cos.(2pi .* u_steps ./ 12.0)) .* u_sigma_sample
        end

        if M.model_st > 0 && "st_raw" in name_strs
            st_sigma = "st_sigma" in name_strs ? chain[:st_sigma].data[s] : 1.0
            st_eta_tot[:, :, s] = reshape(vec(chain[:st_raw].data[s]) .* st_sigma, N_areas, N_time)
        end
    end

    eta = zeros(N_tot, N_samples)
    for s in 1:N_samples
        st_mat = M.model_st > 0 ? st_eta_tot[:, :, s] : zeros(N_areas, N_time)
        for i in 1:N_tot
            is_obs = i <= N_obs
            src = is_obs ? M : PS
            idx_adj = is_obs ? i : i - N_obs
            a = src.s_idx isa AbstractVector ? src.s_idx[idx_adj] : src.s_idx
            t = src.t_idx isa AbstractVector ? src.t_idx[idx_adj] : src.t_idx
            u = src.u_idx isa AbstractVector ? src.u_idx[idx_adj] : src.u_idx
            val = (is_obs ? M.log_offset[i] : 0.0) + s_eta_tot[i, s] + t_eta_tot[t, s] + u_eta_tot[u, s] + st_mat[a, t]
            if M.N_fixed > 0
                fixed_mat = is_obs ? M.fixed : PS.fixed
                for d in 1:M.N_fixed
                    dm_val = fixed_mat isa AbstractMatrix ? fixed_mat[idx_adj, d] : (fixed_mat isa AbstractVector ? fixed_mat[idx_adj] : fixed_mat)
                    val += fixed_effects_samples[d, s] * dm_val
                end

            end
            if "c_eta" in name_strs
                c_level = is_obs ? M.cov_indices[idx_adj, 1] : PS.cov_indices[idx_adj, 1]
                val += c_eta_all[c_level, s]
            end
            eta[i, s] = val
        end
    end

    denoised_mu, noisy_y, log_lik_mat = _process_ll_and_predictions(fam, eta, chain, M, N_tot, N_samples, y_sigma_samples, M.y_obs)
    return (
        spatial_structured = summarize_array(reshape(s_eta_obs_denoised, N_obs, 1, N_samples); alpha=alpha),
        spatial_noisy = summarize_array(reshape(s_eta_obs_noisy, N_obs, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(t_eta_tot, N_time, 1, N_samples); alpha=alpha),
        seasonal = summarize_array(reshape(u_eta_tot, M.N_season, 1, N_samples); alpha=alpha),
        volatility = summarize_array(reshape(y_sigma_samples, N_tot, 1, N_samples); alpha=alpha),
        fixed_effects = M.N_fixed > 0 ? summarize_array(reshape(fixed_effects_samples, M.N_fixed, 1, N_samples); alpha=alpha) : nothing,
        covariate_effects = summarize_array(reshape(c_eta_all, M.N_cat, 1, N_samples); alpha=alpha),
        predictions_observed = summarize_array(reshape(denoised_mu[1:N_obs, :], N_obs, 1, N_samples); alpha=alpha),
        predictions_denoised = N_PS > 0 ? summarize_array(reshape(denoised_mu[(N_obs+1):end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        predictions_noisy = N_PS > 0 ? summarize_array(reshape(noisy_y[(N_obs+1):end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        waic = _compute_waic(log_lik_mat[:, 1:N_obs]), family = fam, architecture = arch
    )
end



function _reconstruct(arch::ComplexArchitecture, fam::ModelFamily, chain, M, PS, alpha)
    """
    _reconstruct(arch, fam, chain, M, alpha)

    Post-processing for models using spectral bases (RFF, FFT) or SPDE mesh approximations.
    
    Key Distinction:
    Unlike GMRF models where effects are area-indexed, spectral models typically provide
    realized latent fields (s_eff/f_gp) directly at the observation level (N_obs).
    """
    N_obs, N_areas, N_time = length(M.y_obs), size(M.W, 1), Int(maximum(M.t_idx))
    N_samples = size(chain, 1)
    name_strs = string.(FlexiChains.parameters(chain))

    spatial = zeros(N_obs, N_samples)
    temporal = zeros(N_time, N_samples)
    eta = zeros(N_obs, N_samples)

    # 1. Realized Spatial Field Extraction
    target_field = "s_eff" in name_strs ? :s_eff : ("f_gp" in name_strs ? :f_gp : nothing)
    if !isnothing(target_field)
        for s in 1:N_samples; spatial[:, s] = vec(chain[target_field].data[s]); end
    end

    # 2. Temporal Trend and Interaction
    if "f_tm_raw" in name_strs
        sig_tm = "sigma_tm" in name_strs ? vec(chain[:sigma_tm].data) : ones(N_samples)
        for s in 1:N_samples; temporal[:, s] = vec(chain[:f_tm_raw].data[s]) .* sig_tm[s]; end
    end

    st_int_present = "st_int_raw" in name_strs
    sig_int = st_int_present ? vec(chain[:sigma_int].data) : zeros(N_samples)

    # 3. Predictor Assembly (N_obs level projection)
    beta_z = "beta_z" in name_strs ? vec(chain[:beta_z].data) : zeros(N_samples)
    beta_cov_eff = _extract_beta_cov(name_strs, chain, M, N_samples, alpha)

    for s in 1:N_samples
        st_i = st_int_present ? reshape(vec(chain[:st_int_raw].data[s]) .* sig_int[s], N_areas, N_time) : zeros(N_areas, N_time)
        for i in 1:N_obs
            a, t = M.s_idx[i], M.t_idx[i]
            eta[i, s] = M.log_offset[i] + spatial[i, s] + temporal[t, s] + st_i[a, t] + beta_z[s] * M.z_obs[i]
            for k in 1:4
                p = Symbol("beta_cov[$k][$(M.cov_indices[i, k])]")
                if p in Symbol.(name_strs); eta[i, s] += chain[p].data[s]; end
            end
        end
    end

    actual_fam = (fam isa UnknownFamily) ? PoissonFamily() : fam
    denoised, noisy, log_lik = _process_ll_and_predictions(actual_fam, eta, chain, M, N_obs, N_samples)

    return (spatial_structured = summarize_array(reshape(spatial, N_obs, 1, N_samples); alpha=alpha),
            spatial_unstructured = nothing,
            spatial = summarize_array(reshape(spatial, N_obs, 1, N_samples); alpha=alpha),
            temporal = summarize_array(reshape(temporal, N_time, 1, N_samples); alpha=alpha),
            seasonal = nothing,
            beta_cov = beta_cov_eff,
            predictions_denoised = summarize_array(reshape(denoised, N_obs, 1, N_samples); alpha=alpha),
            predictions_noisy = summarize_array(reshape(noisy, N_obs, 1, N_samples); alpha=alpha),
            waic = _compute_waic(log_lik),
            family = actual_fam, architecture = arch)
end

 

function _reconstruct(arch::UnknownArchitecture, fam::ModelFamily, chain, M, PS, alpha)
    """
    _reconstruct(arch, fam, chain, M, alpha)

    Post-processing for Deep Gaussian Processes, Mosaic Experts 

    Mathematical Logic:
    1. Manifold Extraction: Extracts the realized latent field from the GP manifold (f_latent, eta_gp, etc.).
    2. Temporal Trend: Supports additive AR1 trends if defined outside the GP kernel.
    3. Seasonal Effects: Reconstructs harmonic components (sin/cos) for periodic models.
    4. Prediction: Maps latent states through inverse-link functions with offset scaling.
    """
    N_obs, N_time = length(M.y_obs), Int(maximum(M.t_idx))
    N_samples = size(chain, 1)
    name_strs = string.(FlexiChains.parameters(chain))

    spatial = zeros(N_obs, N_samples)
    temporal = zeros(N_time, N_samples)
    eta = zeros(N_obs, N_samples)

    # 1. Latent Manifold Extraction
    target = "f_latent" in name_strs ? :f_latent : ("eta_gp" in name_strs ? :eta_gp : ("eta_spatial_time" in name_strs ? :eta_spatial_time : nothing))
    if !isnothing(target)
        for s in 1:N_samples; spatial[:, s] = vec(chain[target].data[s]); end
    end

    # 2. Temporal (if explicitly defined outside the GP manifold)
    if "f_tm_raw" in name_strs
        sig_tm = "sigma_tm" in name_strs ? vec(chain[:sigma_tm].data) : ones(N_samples)
        for s in 1:N_samples; temporal[:, s] = vec(chain[:f_tm_raw].data[s]) .* sig_tm[s]; end
    end

    # 3. Covariates and Seasonal Effects
    beta_cov_eff = _extract_beta_cov(name_strs, chain, M, N_samples, alpha)

    # Check for seasonal harmonic coefficients if present in Adaptive RFF models
    has_seasonal = "beta_cos" in name_strs && "beta_sin" in name_strs
    seasonal_post = zeros(N_time, N_samples)
    if has_seasonal
        bc, bs = vec(chain[:beta_cos].data), vec(chain[:beta_sin].data)
        for s in 1:N_samples
            t_vals = collect(1:N_time)
            seasonal_post[:, s] = bc[s] .* cos.(2pi .* t_vals ./ 12.0) .+ bs[s] .* sin.(2pi .* t_vals ./ 12.0)
        end
    end

    # 4. Assembly
    for s in 1:N_samples
        for i in 1:N_obs
            t = M.t_idx[i]
            eta[i, s] = M.log_offset[i] + spatial[i, s] + temporal[t, s] + (has_seasonal ? seasonal_post[t, s] : 0.0)
            for k in 1:4
                p = Symbol("beta_cov[$k][$(M.cov_indices[i, k])]")
                if p in Symbol.(name_strs); eta[i, s] += chain[p].data[s]; end
            end
        end
    end

    actual_fam = (fam isa UnknownFamily) ? GaussianFamily() : fam
    denoised, noisy, log_lik = _process_ll_and_predictions(actual_fam, eta, chain, M, N_obs, N_samples)

    return (spatial_structured = summarize_array(reshape(spatial, N_obs, 1, N_samples); alpha=alpha),
            spatial_unstructured = nothing,
            spatial = summarize_array(reshape(spatial, N_obs, 1, N_samples); alpha=alpha),
            temporal = summarize_array(reshape(temporal, N_time, 1, N_samples); alpha=alpha),
            seasonal = has_seasonal ? summarize_array(reshape(seasonal_post, N_time, 1, N_samples); alpha=alpha) : nothing,
            beta_cov = beta_cov_eff,
            predictions_denoised = summarize_array(reshape(denoised, N_obs, 1, N_samples); alpha=alpha),
            predictions_noisy = summarize_array(reshape(noisy, N_obs, 1, N_samples); alpha=alpha),
            waic = _compute_waic(log_lik),
            family = actual_fam, architecture = arch)
end

 

function model_results_comprehensive(model, result, M, areal_units; PS=nothing, alpha=0.05, time_slice=1)

    # Unpack 
    if result isa Turing.Variational.VIResult 
        # Specialized method for ADVI results produced by convert_advi_to_reconstruct_format
        bundle = convert_advi_to_reconstruct_format(result, model, n_samples)
        chain = bundle.chain
    elseif result isa VNChain
        chain = result
    elseif result isa Turing.Optimisation.ModeResult 
        chain = convert_optim_to_reconstruct_format(result, model, n_samples; use_hessian=true)
    end
 
    pstats = reconstruct_posteriors( chain, M, PS; alpha=alpha)
 
    # 3. Handle PPC Data Synthesis
    # Skip missing values for metric and plot synthesis
    y_obs_all = M.y_obs
    good_idx = findall(!ismissing, y_obs_all)
    y_obs = Float64.(y_obs_all[good_idx])
    
    # Use denoised observed predictions for training PPC (observation scale)
    y_pred = vec(pstats.predictions_observed.mean)[good_idx]

    # Metrics
    rmse = sqrt(mean((y_obs .- y_pred).^2))
    r2 = length(unique(y_pred)) > 1 ? cor(y_obs, y_pred)^2 : 0.0
    waic_val = pstats.waic

    ss = summarystats(chain) 
  
    # Robust diagnostic extraction - filter out NaN values before computing means
    ess_bulk_vals = filter(!isnan, Array(ss[:, stat=At(:ess_bulk)]))
    mean_ess_bulk = isempty(ess_bulk_vals) ? NaN : mean(ess_bulk_vals)

    ess_tail_vals = filter(!isnan, Array(ss[:, stat=At(:ess_tail)]))
    mean_ess_tail = isempty(ess_tail_vals) ? NaN : mean(ess_tail_vals)

    rhat_vals = filter(!isnan, Array(ss[:, stat=At(:rhat)]))
    mean_rhat = isempty(rhat_vals) ? NaN : mean(rhat_vals)
    
    pms = string.(FlexiChains.parameters(chain))

    compute_time_seconds = hasproperty(chain, :_metadata) && "sampling_time" in pms ? only(round.(chain._metadata.sampling_time, digits=1)) : 1.0
    modelname = string(model.f)
    modelarch = string(M.model_arch)
    modelspace = string(M.model_space)
    modeltime = string(M.model_time)
    modelseason = string(M.model_season)
    modelspacetime = string(M.model_st)
    modelcov = string(M.model_cov)
    
    metrics = ( modelname=modelname, modelarch=modelarch,
        modelspace=modelspace, modeltime=modeltime, modelseason=modelseason, 
        modelspacetime=modelspacetime, modelcov=modelcov, 
        compute_time_seconds=compute_time_seconds,
        rmse=rmse, r2=r2, waic=waic_val, mean_rhat=mean_rhat,
        mean_ess_bulk=mean_ess_bulk, mean_ess_tail=mean_ess_tail, 
        ess_per_second=mean_ess_bulk/compute_time_seconds )

    showtuples( metrics ) 

    # Plots
    plt_ppc = scatter(y_pred, y_obs, alpha=0.4, label="Obs vs Pred", title="PPC (RMSE: $(round(rmse, digits=3)), WAIC: $(round(waic_val, digits=1)))", xlabel="Pred", ylabel="Obs")
    plot!(plt_ppc, [minimum(y_obs), maximum(y_obs)], [minimum(y_obs), maximum(y_obs)], ls=:dash, lc=:red, label="Identity")

    plt_tm = nothing
    if haskey(pstats, :temporal) && !isnothing(pstats.temporal) && !(pstats.temporal isa Real)
        plt_tm = plot(pstats.temporal.mean, 
            ribbon=(pstats.temporal.mean .- pstats.temporal.lower, pstats.temporal.upper .- pstats.temporal.mean), 
            title="Temporal Trend", xlabel="Time", ylabel="Effect", legend=false, fillalpha=0.3)
    end

    plt_seas = nothing
    if haskey(pstats, :seasonal) && !isnothing(pstats.seasonal) && !(pstats.seasonal isa Real)
        plt_seas = plot(pstats.seasonal.mean, 
            ribbon=(pstats.seasonal.mean .- pstats.seasonal.lower, pstats.seasonal.upper .- pstats.seasonal.mean), 
            title="Seasonal Trend", xlabel="Time", ylabel="Effect", legend=false, fillalpha=0.3)
    end
 
    plt_sp = nothing 
    if haskey(pstats, :spatial) && !isnothing(pstats.spatial) && !(pstats.spatial isa Real)
        plt_sp = plot_posterior_results(pstats, M, areal_units; effect=:spatial)
        title!(plt_sp, "Main Spatial Effect")
    end

    plt_sp_struct = nothing
    if haskey(pstats, :spatial_structured) && !isnothing(pstats.spatial_structured) && !(pstats.spatial_structured isa Real)
        plt_sp_struct = plot_posterior_results(pstats, M, areal_units; effect=:spatial_structured )
        title!(plt_sp_struct, "Structured Spatial Effect")
    end

    plt_sp_unstruct = nothing
    if haskey(pstats, :spatial_unstructured) && !isnothing(pstats.spatial_unstructured) && !(pstats.spatial_unstructured isa Real)
        plt_sp_unstruct = plot_posterior_results(pstats, M, areal_units; effect=:spatial_unstructured )
        title!(plt_sp_unstruct, "Unstructured Spatial Effect")
    end
 
    
    plt_st_denoised = nothing
    if haskey(pstats, :predictions_denoised) && !isnothing(pstats.predictions_denoised) && !(pstats.predictions_denoised isa Real)
        plt_st_denoised = plot_posterior_results(pstats, M, areal_units; effect=:predictions_denoised, time_slice=time_slice)
        title!(plt_st_denoised, "Denoised Predictions (T=$time_slice)")
    end


    plt_st_noisy = nothing
    if haskey(pstats, :predictions_noisy) && !isnothing(pstats.predictions_noisy) && !(pstats.predictions_noisy isa Real)
        plt_st_noisy = plot_posterior_results(pstats, M, areal_units; effect=:predictions_noisy, time_slice=time_slice)
        title!(plt_st_noisy, "Noisy Predictions (T=$time_slice)")
    end

    # Combine only non-nothing plots
    plot_list = filter(!isnothing, [plt_ppc, plt_tm, plt_seas, plt_sp, plt_st_denoised])
    
    n_plots = length(plot_list)
    n_rows = ceil(Int, n_plots / 2)
    display(plot(plot_list..., layout=(n_rows, min(n_plots, 2)), size=(1000, 350*n_rows)))

    return (
        metrics=metrics, 
        pstats=pstats,
        summarystats=ss, 
        plots=(ppc=plt_ppc, tm=plt_tm, seas=plt_seas, spatial=plt_sp, spatial_structured=plt_sp_struct, spatial_unstructured=plt_sp_unstruct,
            denoised=plt_st_denoised, noisy=plt_st_noisy)
    )
end

 

function convert_advi_to_reconstruct_format(msol, model::DynamicPPL.Model, n_samples::Int=500)
    """
    Synopsis: Converts ADVI variational solutions into a format compatible with _reconstruct logic.
    Inputs:
    - msol: The solution object from Turing.vi()
    - model: The Turing model instance used for VI.
    - n_samples: Number of posterior samples to draw from the variational distribution.
    Outputs:
    - A FlexiChains-like object or standard Chain that _reconstruct can process.
    """
  
    # 1. Sample from the variational solution
    # Turing's rand(msol, n_samples) returns a Vector of VarNamedTuples
    samples_vec = rand(msol, n_samples)

    # 2. Extract unique base parameter names for reconstruction logic
    # We peek at the first sample to find the keys
    all_keys = keys(samples_vec[1])
    unique_bases = Set{Symbol}()
    for k in all_keys
        # Regex handles both scalar 'x' and vector 'x[1]' patterns
        m = match(r"^([^\\\\[]+)", string(k))
        if m !== nothing
            push!(unique_bases, Symbol(m.captures[1]))
        end
    end

    # 3. Create the reconstruct_samples (Vector of NamedTuples)
    # Required for DynamicPPL.reconstruct(model, sample)
    reconstruct_samples = map(samples_vec) do samp
        sample_params = Dict{Symbol, Any}()
        for base_sym in unique_bases
            base_str = string(base_sym)
            # Filter keys belonging to this base parameter group
            col_keys = filter(k -> string(k) == base_str || startswith(string(k), "$base_str["), all_keys)

            if length(col_keys) == 1 && string(first(col_keys)) == base_str
                sample_params[base_sym] = samp[first(col_keys)]
            else
                # Sort indexed keys like x[1], x[10], x[2] into numerical order
                sorted_keys = sort(collect(col_keys), by = k -> begin
                    m_idx = match(r"\\\\[(\\d+)\\\\]", string(k))
                    m_idx !== nothing ? parse(Int, m_idx.captures[1]) : 0
                end)
                sample_params[base_sym] = [samp[k] for k in sorted_keys]
            end
        end
        return (; sample_params...)
    end

    # 4. Construct FlexiChain for diagnostics
    # FIXED: Explicitly convert VarName keys to Symbol for FlexiChains compatibility
    formatted_dicts = map(samples_vec) do samp
        # Use Symbol(k) to ensure keys are Parameter{Symbol} not Parameter{VarName}
        Dict(FlexiChains.Parameter(Symbol(k)) => v for (k, v) in pairs(samp))
    end

    chn = FlexiChains.FlexiChain{Symbol}(n_samples, 1, formatted_dicts)

    return (chain=chn, reconstruct_samples=reconstruct_samples)
end


function convert_optim_to_reconstruct_format(optim_result, model, n_samples::Int=500; use_hessian=true)
    """
    Synopsis: Converts a point estimate (MAP/ML) from Optim/Optimisers into a distribution of samples.
    If a Hessian is available, it samples from the Multivariate Normal Laplace approximation.
    Otherwise, it creates a narrow Gaussian around the point estimate.
    """
    
    # 1. Extract the point estimate (the 'values' from Turing's Optim solution)
    # Turing's optimize() returns a result where .values is a VarNamedTuple
    point_est = optim_result.values
    all_keys = keys(point_est)
    
    # 2. Extract unique base parameter names
    unique_bases = Set{Symbol}()
    for k in all_keys
        m = match(r"^([^\\[]+)", string(k))
        if m !== nothing
            push!(unique_bases, Symbol(m.captures[1]))
        end
    end

    # 3. Create synthetic samples
    # In a real MAP context, we'd use inv(-Hessian). For this helper, we'll 
    # default to a very tight distribution if no Hessian logic is passed.
    reconstruct_samples = map(1:n_samples) do _
        sample_params = Dict{Symbol, Any}()
        for base_sym in unique_bases
            base_str = string(base_sym)
            col_keys = filter(k -> string(k) == base_str || startswith(string(k), "$base_str["), all_keys)

            # Map specific values and add a tiny bit of noise (1e-4) to simulate 'width'
            # if we are not doing a formal Laplace approximation
            if length(col_keys) == 1 && string(first(col_keys)) == base_str
                sample_params[base_sym] = point_est[first(col_keys)] + randn() * 1e-4
            else
                sorted_keys = sort(collect(col_keys), by = k -> begin
                    m_idx = match(r"\\\[(\\d+)\\\]", string(k))
                    m_idx !== nothing ? parse(Int, m_idx.captures[1]) : 0
                end)
                sample_params[base_sym] = [point_est[k] + randn() * 1e-4 for k in sorted_keys]
            end
        end
        return (; sample_params...)
    end

    # 4. Format into a FlexiChain for standard diagnostics
    formatted_dicts = map(1:n_samples) do i
        samp = reconstruct_samples[i]
        # We need to flatten the dictionary back to VarName format for FlexiChains
        d = Dict{FlexiChains.Parameter, Any}()
        for k in keys(samp)
            val = samp[k]
            if val isa AbstractVector
                for (idx, v) in enumerate(val)
                    d[FlexiChains.Parameter(Symbol("$k[$idx]"))] = v
                end
            else
                d[FlexiChains.Parameter(k)] = val
            end
        end
        return d
    end

    chn = FlexiChains.FlexiChain{Symbol}(n_samples, 1, formatted_dicts)

    return (chain=chn, reconstruct_samples=reconstruct_samples)
end

struct bstm_Likelihood{F, Z, W, P, R, S, T, TR} <: ContinuousMultivariateDistribution
    family::F
    use_zi::Z
    weights::W
    phi_zi::P
    r_nb::R
    sigma_y::S
    trials::T
    y_obs::TR
end

Base.length(d::bstm_Likelihood) = length(d.y_obs)

function Distributions.logpdf(d::bstm_Likelihood, eta::AbstractVector)
    
    total_lp = 0.0
    trials_scalar = d.trials isa Number    # size(d.trials,1) == 1 ? true : false
    sig_scalar = d.sigma_y isa Number # size(d.sigma_y, 1) == 1
     
    for i in 1:length(eta)
        y_val = d.y_obs[i]
        lin_pred = eta[i]
        lp = 0.0

        if d.family == "poisson"
            mu = exp(lin_pred)
            if d.use_zi
                if y_val == 0
                    lp = log(d.phi_zi + (1 - d.phi_zi) * exp(-mu) + 1e-10)
                else
                    lp = log(1 - d.phi_zi + 1e-10) + logpdf(Poisson(mu), y_val)
                end
            else
                lp = logpdf(Poisson(mu), y_val)
            end
        elseif d.family == "binomial"
            p = logistic(lin_pred)
            ntrials = trials_scalar ? d.trials : d.trials[i]
            if d.use_zi
                if y_val == 0
                    lp = log(d.phi_zi + (1 - d.phi_zi) * pdf(Binomial(ntrials, p), 0) + 1e-10)
                else
                    lp = log(1 - d.phi_zi + 1e-10) + logpdf(Binomial(ntrials, p), y_val)
                end
            else
                lp = logpdf(Binomial(ntrials, p), y_val)
            end
        elseif d.family == "negbin"
            mu = exp(lin_pred)
            prob = d.r_nb / (d.r_nb + mu)
            if d.use_zi
                if y_val == 0
                    lp = log(d.phi_zi + (1 - d.phi_zi) * pdf(NegativeBinomial(d.r_nb, prob), 0) + 1e-10)
                else
                    lp = log(1 - d.phi_zi + 1e-10) + logpdf(NegativeBinomial(d.r_nb, prob), y_val)
                end
            else
                lp = logpdf(NegativeBinomial(d.r_nb, prob), y_val)
            end
        elseif d.family == "gaussian"
            sig = sig_scalar ? d.sigma_y : d.sigma_y[i]
            lp = logpdf(Normal(lin_pred, sig), y_val)
        elseif d.family == "lognormal"
            sig = sig_scalar ? d.sigma_y : d.sigma_y[i]
            lp = logpdf(LogNormal(lin_pred, sig), y_val)
        end
        total_lp += lp * d.weights[i]
    end
    return total_lp
end




function generate_sim_data(n_pts=10, n_time=5; rndseed=42)
    Random.seed!(rndseed)
    n_total = n_pts * n_time
    unique_pts = [(rand() * 100, rand() * 100) for _ in 1:n_pts]
    pts_full_dataset = repeat(unique_pts, n_time)
    s_obs = vcat([collect(t)' for t in pts_full_dataset]...)
    t_obs = rand(n_total) .* (n_time+1) .+ 2000
    weights = ones(n_total)
    trials  = ones(Int, n_total)
    N_time = n_time
    N_season = 12
    period=12.0
    trend = 0.05 .* t_obs[:,1]
    seasonal = 1.0 .* cos.(2pi .* t_obs[:,1] ./ period)
    temporal_effect =  1.0 .* ( trend .+ seasonal )
    spatial_effect = 1.5 .* sin.(s_obs[:,1] .*  2pi) .* cos.(s_obs[:,2] .*  2pi)
    sigma_y = 0.2
    observation_error = sigma_y .* randn(n_total)
    Z = randn(n_total)
    W1_true = 0.5 .* sin.(t_obs[:,1] ./ 5.0) .+ 0.5 .* Z
    W2_true = 0.5 .* cos.(t_obs[:,1] ./ 5.0) .- 0.3 .* Z
    W3_true = 0.2 .* (t_obs[:,1] ./ n_total) .+ 0.1 .* Z
    sigma_w1, sigma_w2, sigma_w3 = 0.1, 0.2, 0.3
    W1_obs = W1_true .+ randn(n_total) .* sigma_w1
    W2_obs = W2_true .+ randn(n_total) .* sigma_w2
    W3_obs = W3_true .+ randn(n_total) .* sigma_w3
    cov_continuous = hcat(W1_obs, W2_obs, W3_obs)

    y_obs = 1.0 .+  spatial_effect + temporal_effect .+  observation_error .+ W1_obs .+ W2_obs .+ W3_obs
    y_binary = y_obs .> (mean(y_obs) + 0.5)
    y_counts = abs.(Int.(round.(y_obs))) * 100
    
    # fixed (covariate) .. user must make orthogonal for factors .. GLM.jl has a function I think
    fixed = [ones(n_total) ]   # column of ones is the intercept
     
    return (
        pts=pts_full_dataset, s_obs=s_obs, t_obs=t_obs, weights=weights, trials=trials, N_time=N_time, N_season=N_season,
        y_obs=y_obs, y_binary=y_binary, y_counts=y_counts, fixed=fixed,
        z_obs=Z, w_obs=cov_continuous
    )
end
 


function build_bstm_ar1_template(n::Int; method="time", scale=true)
    if n < 1
        return (matrix = zeros(0, 0), scaling_factor = 1.0)
    elseif n == 1
        return (matrix = ones(1, 1), scaling_factor = 1.0)
    end

    Q = spzeros(Float64, n, n)
    m_str = lowercase(string(method))

    if m_str in ["season", "cyclic", "periodic"]
        # Periodic boundary conditions (Ring Graph)
        for i in 1:n
            Q[i, i] = 2.0
            Q[i, mod1(i - 1, n)] -= 1.0
            Q[i, mod1(i + 1, n)] -= 1.0
        end
    else
        # Standard Linear boundary conditions (Path Graph)
        Q[1, 1] = 1.0
        for i in 2:(n - 1); Q[i, i] = 2.0; end
        Q[n, n] = 1.0
        for i in 1:(n - 1)
            Q[i, i + 1] = -1.0
            Q[i + 1, i] = -1.0
        end
    end

    sf = 1.0
    if scale
        # Filter for non-zero eigenvalues (Null space rank 1 for connected graphs)
        e_vals = eigvals(Matrix(Q))
        eigs = filter(x -> x > 1e-7, e_vals)
        if !isempty(eigs)
            sf = exp(mean(log.(eigs)))
        end
    end

    return (matrix = Matrix(Symmetric(Q) ./ sf), scaling_factor = sf)
end


# Helper for AD-compatible precision recomposition
function build_bstm_ar1_precision(template_mat::AbstractMatrix, rho::Real; noise=1e-6)
    T = typeof(rho)
    n = size(template_mat, 1)
    # The AR1 precision is defined as (I + rho^2*I - rho*W) / (1-rho^2)
    # Our template represents (D - W). We transform it back:
    # Since template = (D - W)/sf, and D=2I (approx), we recompose:
    Q_rho = (1.0 / (1.0 - rho^2 + T(noise))) .* ( (1.0 + rho^2) .* I(n) .+ (rho) .* template_mat )
    return Symmetric(Q_rho + (T(noise) * I))
end



function build_bstm_rw2_template(n::Int; noise=1e-6, scale=true)
    # 1. Construct the Second-Order Difference Matrix (D)
    # For RW2, D is an (n-2) x n matrix
    D = spzeros(Float64, n - 2, n)
    for i in 1:(n - 2)
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    end

    # 2. Form the precision matrix Q = D' * D
    Q = D' * D

    # 3. Geometric Mean Scaling
    # Ensures that the variance parameter represents the marginal standard deviation
    sf = 1.0
    if scale
        # Filter for non-zero eigenvalues (RW2 has a null space of rank 2)
        eigs = filter(x -> x > 1e-6, eigvals(Matrix(Q)))
        if !isempty(eigs)
            sf = exp(mean(log.(eigs)))
        end
    end

    return (matrix =Matrix(Symmetric( Q ./ sf)), scaling_factor = sf)
end

function build_bstm_rw2_precision(template_mat::AbstractMatrix, sigma::Real; noise=1e-6)
    # Recompose precision using the marginal variance sigma^2
    # Q_final = (1 / sigma^2) * Q_template
    T = eltype(sigma)
    Q_final = (1.0 / (sigma^2 + noise)) .* template_mat
    
    # Add jitter for numerical stability
    return Symmetric(Matrix(Q_final) + (noise * I))
end

function build_bstm_harmonic_template(n::Int; noise=1e-6, scale=true)
    # Construct a cyclic RW2 matrix for harmonic smoothing
    # This maintains the periodic constraint where the last node wraps to the first
    Q = spzeros(Float64, n, n)
    
    for i in 1:n
        # Difference operator: (x_{i-1} - 2x_i + x_{i+1})
        # Using mod1 for cyclic wrapping
        prev = mod1(i - 1, n)
        curr = i
        nxt  = mod1(i + 1, n)
        
        # Add the squared difference components to the precision matrix
        # This represents the structure of (D'D) where D is cyclic second-order differences
        Q[curr, curr] += 4.0
        Q[prev, prev] += 1.0
        Q[nxt, nxt]   += 1.0
        
        Q[curr, prev] -= 2.0
        Q[prev, curr] -= 2.0
        Q[curr, nxt]  -= 2.0
        Q[nxt, curr]  -= 2.0
        
        Q[prev, nxt]  += 1.0
        Q[nxt, prev]  += 1.0
    end

    # Geometric Mean Scaling
    sf = 1.0
    if scale
        eigs = filter(x -> x > 1e-6, eigvals(Matrix(Q)))
        if !isempty(eigs)
            sf = exp(mean(log.(eigs)))
        end
    end

    return (matrix = Matrix(Symmetric(Q ./ sf)), scaling_factor = sf)
end

 
function build_bstm_harmonic_precision(template_mat::AbstractMatrix, sigma::Real; noise=1e-6)
    # Note this is identical to RW2 ... as they are very similar. ..
    # Recompose precision using the marginal variance sigma^2
    # Q_final = (1 / sigma^2) * Q_template
    T = eltype(sigma)
    Q_final = (1.0 / (sigma^2 + noise)) .* template_mat
    
    # Add jitter for numerical stability
    return Symmetric(Matrix(Q_final) + (noise * I))
end


 
function precompute_nystrom_projection(spatial_coords, inducing_points, kernel_func; jitter=1e-6)

    # Example Usage:
    # kernel = Matern32Kernel() ∘ ScaleTransform(1.0)
    # modinputs_reference.K_nystrom_proj = precompute_nystrom_projection(areal_units.centroids, Z_inducing, kernel)

    # println("Precomputing Nystrom Projection Matrix...")
    
    # 1. K_mm: Kernel matrix between inducing points (M x M)
    K_mm = kernelmatrix(kernel_func, RowVecs(inducing_points))
    K_mm_stable = Symmetric(K_mm + jitter * I)
    
    # 2. K_nm: Kernel matrix between all spatial units and inducing points (N x M)
    K_nm = kernelmatrix(kernel_func, RowVecs(spatial_coords), RowVecs(inducing_points))
    
    # 3. Projection: K_nm * inv(K_mm)
    # We use the backslash operator for better numerical stability than direct inversion
    K_nystrom_proj = K_nm / K_mm_stable
    
    # println("Generated projection matrix of size: ", size(K_nystrom_proj))
    return K_nystrom_proj
end



# --- Optimized Householder PCA Helper Functions ---

function householder_to_eigenvector(v_mat::AbstractMatrix{T}, n_outcomes, n_factors) where {T}
    # Initializes the Identity matrix to be transformed
    U = Matrix{T}(I, n_outcomes, n_outcomes)

    for k in 1:n_factors
        # Extract the k-th Householder vector
        vk = v_mat[:, k]
        norm_v = norm(vk)
        
        if norm_v > 1e-9
            vk = vk / norm_v
            
            # --- O(K * N^2) Optimization ---
            # Naive: U = (I - 2vv') * U  => O(N^3)
            # Optimized: U = U - 2v * (v' * U) => O(N^2)
            # We first compute the row vector (v' * U), then perform an outer product update.
            
            v_transpose_U = vk' * U
            U = U - 2.0 .* vk * v_transpose_U
        end
    end

    # Return only the first n_factors columns as the orthonormal loadings matrix
    return U[:, 1:n_factors]
end

function householder_transform(v, n_outcomes, n_factors, ltri_indices, pca_sd, pdef_sd, noise)
    T = eltype(v)
    v_mat = zeros(T, n_outcomes, n_factors)
    v_mat[ltri_indices] .= v

    # Generate Orthonormal Loadings using optimized transformation
    U = householder_to_eigenvector(v_mat, n_outcomes, n_factors)

    # Reconstruct Covariance Components
    # W = Loadings * Scaled Eigenvalues
    W = U * Diagonal(pca_sd)

    # Kmat is the full covariance matrix: WW' + Residual_Variance
    # We add a small noise term for numerical stability
    Kmat = W * W' + (pdef_sd^2 + noise) * I(n_outcomes)

    return Kmat, pca_sd, U
end



function eigenvector_to_householder(U_in::AbstractMatrix{T}, n_factors) where {T}
# --- Optimal Vector Extraction (Orthonormal to Householder) ---

# eigenvector_to_householder(U, n_factors)
#
# Description:
#   Extracts the Householder reflector vectors (v) from an orthonormal loadings matrix U.
#   This allows for initializing the Bayesian model from a frequentist PCA result.
#
# Complexity: O(K * N^2)

    n_outcomes = size(U_in, 1)
    # We work on a copy to avoid modifying the input
    U = copy(U_in)
    
    # Storage for the lower-triangular part of the v_mat
    # Each column k corresponds to the k-th Householder vector
    v_mat = zeros(T, n_outcomes, n_factors)

    for k in 1:n_factors
        # 1. Target vector is the k-th column of the current transformation
        # For the identity, we want U[k,k] to be 1 and others 0
        x = U[k:end, k]
        
        # 2. Standard Householder Reflection Math
        # v = x + sign(x[1]) * ||x|| * e1
        norm_x = norm(x)
        vk = copy(x)
        
        sign_x1 = x[1] >= 0 ? one(T) : -one(T)
        vk[1] += sign_x1 * norm_x
        
        norm_vk = norm(vk)
        if norm_vk > 1e-9
            vk = vk ./ norm_vk
            
            # 3. Apply the reflection to the rest of the matrix (Rank-1 update)
            # U[k:end, k:end] = (I - 2vv') * U[k:end, k:end]
            # Using the O(N^2) update trick
            v_transpose_U = vk' * U[k:end, k:end]
            U[k:end, k:end] -= 2.0 .* vk * v_transpose_U
            
            # 4. Store the reflector
            v_mat[k:end, k] .= vk
        end
    end

    return v_mat
end


function extract_v_parameters(v_mat, ltri_indices)
    # extract_v_parameters(v_mat, ltri_indices)
    #
    # Utility to extract only the free parameters (lower triangular) from the v_mat
    # for use as initial values in Turing (matching the 'v' parameter vector).
    return v_mat[ltri_indices]
end

;;