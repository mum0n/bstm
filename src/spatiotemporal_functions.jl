
 
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


################




function expand_hull(s_coord_tuple, buffer_dist)
    """
    Synopsis: Computes the convex hull of points and expands it by a buffer distance.
    Inputs:
    - s_coord_tuple: Vector of (x, y) tuples.
    - buffer_dist: Distance to buffer the convex hull.
    Outputs:
    - A LibGEOS Polygon geometry representing the buffered convex hull.
    """

    if isempty(s_coord_tuple) return LibGEOS.Polygon([[ (0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0) ]]) end
    coords_vec = [[Float64(p[1]), Float64(p[2])] for p in s_coord_tuple]
    points_geom = LibGEOS.MultiPoint(coords_vec)
    hull = LibGEOS.convexhull(points_geom)
    buffered_hull = LibGEOS.buffer(hull, buffer_dist)
    return buffered_hull
end


function get_kde_seeds(s_coord_tuple, target_u)
    # Basic KDE-based seeding using StatsBase weights based on local density
    u_pts = unique(s_coord_tuple)
    if isempty(u_pts) return [] end
    n = length(u_pts)
    dists = [sum((p1 .- p2).^2) for p1 in u_pts, p2 in u_pts]
    # Inverse of mean distance as a density proxy
    weights = 1.0 ./ (mean(dists, dims=2)[:] .+ 1e-6)
    idx = sample(1:n, Weights(weights), min(target_u, n), replace=false)
    return u_pts[idx]
end

 

 
function is_valid_polygon_coords(poly_coords)
    # Filters out NaN/Inf values and checks for a minimum of 3 valid points for a polygon.
    valid_pts = [p for p in poly_coords if !isnan(p[1]) && !isinf(p[1]) && !isnan(p[2]) && !isinf(p[2])]
    return length(valid_pts) >= 3
end
 
function get_cvt_centroids(s_coord_tuple, cfg, hull_geom)
    """
    Synopsis: Centroidal Voronoi Tessellation (CVT) with diagnostic termination tracking.
    """

    if isempty(s_coord_tuple)
        return [], "no_points_provided"
    end

    if length(s_coord_tuple) <= cfg.min_total_arealunits
        return [ (mean(p[1] for p in s_coord_tuple), mean(p[2] for p in s_coord_tuple)) ], "not_enough_points_to_tessellate"
    end

    u_pts = unique(s_coord_tuple)
    idx = StatsBase.sample(1:length(u_pts), min(cfg.target, length(u_pts)), replace=false)
    curr_centroids = [u_pts[i] for i in idx]
    termination_reason = "max_iterations"

    # Initialize convergence tracking variables
    last_mean_density = 0.0
    last_cv = 0.0

    for iter in 1:100
        polys, _ = get_voronoi_polygons_and_edges(curr_centroids, hull_geom)
        new_centroids = Tuple{Float64, Float64}[]
        shifts = Float64[]

        for i in 1:length(polys)
            poly_coords = polys[i]
            area = get_polygon_area(poly_coords)

            if length(poly_coords) > 2 && area >= cfg.min_area && area <= cfg.max_area
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

        if isempty(shifts) || mean(shifts) < cfg.tolerance
            termination_reason = "convergence"
            break
        end

        assigns = [argmin([sum((p .- c).^2) for c in new_centroids]) for p in s_coord_tuple]
        counts = [count(==(i), assigns) for i in 1:length(new_centroids)]

        if isempty(counts)
            termination_reason = "no_units_formed"
            break
        end

        # New Density Convergence Check
        curr_mean_density = mean(counts)
        if abs(curr_mean_density - last_mean_density) < cfg.tolerance && iter > 1
            termination_reason = "density_convergence"
            break
        end
        last_mean_density = curr_mean_density

        cv_val = std(counts) / (mean(counts) + 1e-9)
        # CV Convergence Check
        if abs(cv_val - last_cv) < cfg.tolerance && iter > 1
            termination_reason = "cv_convergence"
            break
        end
        last_cv = cv_val

        if mean(counts) < cfg.min_points
            termination_reason = "min_points_violation"
            break
        end

        curr_centroids = new_centroids
    end

    return curr_centroids, termination_reason
end

function get_kvt_centroids(s_coord_tuple, cfg, hull_geom)
    """
    Synopsis: K-means Voronoi Tessellation (KVT) with diagnostic termination tracking.
    """

    if isempty(s_coord_tuple)
        return [], "no_points_provided"
    end

    if length(s_coord_tuple) <= cfg.min_total_arealunits
        return [ (mean(p[1] for p in s_coord_tuple), mean(p[2] for p in s_coord_tuple)) ], "not_enough_points_to_tessellate"
    end

    u_pts = unique(s_coord_tuple)
    idx_init = sample(1:length(u_pts), min(cfg.target, length(u_pts)), replace=false)
    c_iter = [u_pts[i] for i in idx_init]
    data = collect(zip(s_coord_tuple, cfg.t_idx))
    damping = 0.7
    termination_reason = "max_iterations"

    # Initialize convergence tracking variables
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

            # Modified area_ok condition: require positive area
            area_ok = (area > 0) && area >= cfg.min_area && area <= cfg.max_area

            if !isempty(idx_cluster) && length(idx_cluster) >= cfg.min_points && ts_count >= cfg.min_time_slices && area_ok
                mean_x = mean(data[j][1][1] for j in idx_cluster)
                mean_y = mean(data[j][1][2] for j in idx_cluster)

                c_iter[k] = ((1.0 - damping) * old_centroids[k][1] + damping * mean_x,
                             (1.0 - damping) * old_centroids[k][2] + damping * mean_y)
            end
        end

        counts = [count(==(k), assigns) for k in 1:length(c_iter)]
        if isempty(counts)
            termination_reason = "no_units_formed"
            break
        end

        # New Density Convergence Check
        curr_mean_density = mean(counts)
        if abs(curr_mean_density - last_mean_density) < cfg.tolerance && iter > 1
            termination_reason = "density_convergence"
            break
        end
        last_mean_density = curr_mean_density

        cv_val = std(counts) / (mean(counts) + 1e-9)
        # CV Convergence Check
        if abs(cv_val - last_cv) < cfg.tolerance && iter > 1
            termination_reason = "cv_convergence"
            break
        end
        last_cv = cv_val

        if mean(counts) < cfg.min_points
            termination_reason = "min_points_violation"
            break
        end

        damping *= 0.99
    end

    return c_iter, termination_reason
end

function get_qvt_centroids(s_coord_tuple, cfg, hull_geom)
    """
    Synopsis: Quadtree Voronoi Tessellation (QVT) with corrected recursive splitting logic.
    """
    if isempty(s_coord_tuple); return [], "no_points_provided"; end

    if length(s_coord_tuple) <= cfg.min_total_arealunits
        return [ (mean(p[1] for p in s_coord_tuple), mean(p[2] for p in s_coord_tuple)) ], "not_enough_points_to_tessellate"
    end

    data = collect(zip(s_coord_tuple, cfg.t_idx))
    regions = [data]
    # Track specific region objects that failed to split to avoid redundant attempts
    unsplittable = Set{UInt64}()

    effective_min_p = max(1, cfg.min_points)

    if length(data) < 2 * effective_min_p # Initial check: if the whole dataset is too small to split
        return [(mean(p[1][1] for p in data), mean(p[1][2] for p in data))], "initial_data_too_small_to_tessellate"
    end

    termination_reason = "max_units_reached"

    # Initialize convergence tracking variables
    last_mean_density = 0.0
    last_cv = 0.0

    cnt = 0

    while length(regions) < cfg.max_total_arealunits
        cnt += 1

        counts = length.(regions)
        curr_mean_density = mean(counts)
        cv_val = std(counts) / (curr_mean_density + 1e-9)

        # Early breaking based on statistical stabilization and target resolution
        if cnt > 3
            if last_mean_density > 0.0 && (abs(curr_mean_density - last_mean_density) < cfg.tolerance || abs(cv_val - last_cv) < cfg.tolerance)
                if length(regions) >= cfg.target && all(c -> c <= cfg.max_points, counts)
                    termination_reason = "converged_constraints_satisfied"
                    break
                elseif abs(cv_val - cfg.target_cv) < cfg.tolerance
                    termination_reason = "converged_target_cv"
                    break
                elseif (count(>(cfg.max_points), counts) / length(regions)) < cfg.tolerance/10
                    termination_reason = "converged_minor_violations"
                    break
                end
            end
        end

        last_mean_density = curr_mean_density
        last_cv = cv_val

        # Candidacy: regions that can be split
        viable_indices = findall(r -> length(r) >= max(2, effective_min_p) && objectid(r) ∉ unsplittable, regions)
        if cnt > 3
            if isempty(viable_indices); termination_reason = "cannot_split_further"; break; end
        end

        # Split if (below min_total_arealunits) OR (below target OR has max_points violators)
        violators = filter(i -> length(regions[i]) > cfg.max_points, viable_indices)
        must_split = length(regions) < cfg.min_total_arealunits
        want_split = length(regions) < cfg.target || !isempty(violators)
        candidates = (must_split || want_split) ? (isempty(violators) ? viable_indices : violators) : []

        if isempty(candidates); termination_reason = "constraints_satisfied"; break; end

        # Attempt splitting the largest available candidate
        target_idx = candidates[argmax([length(regions[i]) for i in candidates])]
        target_region = regions[target_idx]

        xs = [p[1][1] for p in target_region]; ys = [p[1][2] for p in target_region]

        # Robust splitting: handle datasets with zero variance in one or more dimensions
        if length(unique(xs)) > 1 || length(unique(ys)) > 1
            mx = length(unique(xs)) > 1 ? median(xs) : xs[1]
            my = length(unique(ys)) > 1 ? median(ys) : ys[1]
            r_splits = [
                filter(p -> p[1][1] <= mx && p[1][2] <= my, target_region),
                filter(p -> p[1][1] > mx && p[1][2] <= my, target_region),
                filter(p -> p[1][1] <= mx && p[1][2] > my, target_region),
                filter(p -> p[1][1] > mx && p[1][2] > my, target_region)
            ]
        else
            # All points collocated spatially: split by index to progress toward target unit count
            mid = length(target_region) ÷ 2
            r_splits = [target_region[1:mid], target_region[mid+1:end], [], []]
        end

        valid_splits = filter(r -> length(r) >= effective_min_p, r_splits)

        if length(valid_splits) < 2
            # This specific region is locally unsplittable; mark it and continue with others
            push!(unsplittable, objectid(target_region))
            continue
        end

        deleteat!(regions, target_idx)
        append!(regions, valid_splits)
    end

    # Post-audit: filter by final constraints (minimum time slices and point counts)
    final_filtered_regions = filter(regions) do r
        length(r) >= effective_min_p && length(unique([p[2] for p in r])) >= cfg.min_time_slices
    end

    if isempty(final_filtered_regions)
        return [], "no_valid_units_after_filter"
    end

    final_centroids = [(mean(p[1][1] for p in r), mean(p[1][2] for p in r)) for r in final_filtered_regions]

    polys_coords, _ = get_voronoi_polygons_and_edges(final_centroids, hull_geom)
    area_violation = any(
        p_coords -> !is_valid_polygon_coords(p_coords) || get_polygon_area(p_coords) < cfg.min_area,
        polys_coords
    )

    final_status = termination_reason
    if length(final_centroids) < cfg.min_total_arealunits
        final_status = "insufficient_units_error"
    elseif area_violation
        final_status = "min_area_violation"
    end

    return final_centroids, final_status
end

function get_bvt_centroids(s_coord_tuple, cfg, hull_geom)
    """
    Synopsis: Binary Voronoi Tessellation (BVT) with corrected recursive splitting logic.
    """
    if isempty(s_coord_tuple)
        return [], "no_points_provided"
    end

    if length(s_coord_tuple) <= cfg.min_total_arealunits
        return [ (mean(p[1] for p in s_coord_tuple), mean(p[2] for p in s_coord_tuple)) ], "not_enough_points_to_tessellate"
    end

    data = collect(zip(s_coord_tuple, cfg.t_idx))
    regions = [data]
    # Track specific region objects that failed to split to avoid redundant attempts

    effective_min_p = max(1, cfg.min_points)
    if length(data) < 2 * effective_min_p # Initial check: if the whole dataset is too small to split
        return [(mean(p[1][1] for p in data), mean(p[1][2] for p in data))], "initial_data_too_small_to_tessellate"
    end

    unsplittable = Set{UInt64}()
    termination_reason = "max_units_reached"
    last_mean_density = 0.0
    last_cv = 0.0
    cnt =0

    while length(regions) < cfg.max_total_arealunits
        cnt += 1

        counts = length.(regions)
        curr_mean_density = mean(counts)
        cv_val = std(counts) / (curr_mean_density + 1e-9)

        # Early breaking based on statistical stabilization and target resolution
        if cnt > 3
            if last_mean_density > 0.0 && (abs(curr_mean_density - last_mean_density) < cfg.tolerance || abs(cv_val - last_cv) < cfg.tolerance)
                if length(regions) >= cfg.target && all(c -> c <= cfg.max_points, counts)
                    termination_reason = "converged_constraints_satisfied"
                    break
                elseif (abs(cv_val - cfg.target_cv) < cfg.tolerance)
                    termination_reason = "converged_target_cv"
                    break
                elseif (count(>(cfg.max_points), counts) / length(regions)) < cfg.tolerance/10
                    termination_reason = "converged_minor_violations"
                    break
                end
            end
        end

        last_mean_density = curr_mean_density
        last_cv = cv_val

        # Candidacy: regions that can be split
        viable_indices = findall(r -> length(r) >= max(2, effective_min_p) && objectid(r) ∉ unsplittable, regions)
        if cnt > 3
            if isempty(viable_indices); termination_reason = "cannot_split_further"; break; end
        end

        # Split if (below min_total_arealunits) OR (below target OR has max_points violators)
        violators = filter(i -> length(regions[i]) > cfg.max_points, viable_indices)
        must_split = length(regions) < cfg.min_total_arealunits
        want_split = length(regions) < cfg.target || !isempty(violators)
        candidates = (must_split || want_split) ? (isempty(violators) ? viable_indices : violators) : []

        if isempty(candidates); termination_reason = "constraints_satisfied"; break; end

        # Attempt splitting the largest available candidate
        target_idx = candidates[argmax([length(regions[i]) for i in candidates])]
        target = regions[target_idx]

        xs = [p[1][1] for p in target]; ys = [p[1][2] for p in target]
        var_x = length(xs) > 1 ? var(xs) : 0.0
        var_y = length(ys) > 1 ? var(ys) : 0.0
        dim = var_x > var_y ? 1 : 2

        if var_x > 1e-9 || var_y > 1e-9
            vals = [p[1][dim] for p in target]
            med = length(unique(vals)) > 1 ? median(vals) : vals[1]
            r1 = filter(p -> p[1][dim] <= med, target)
            r2 = filter(p -> p[1][dim] > med, target)
        else
            # Handle collocated points
            mid = length(target) ÷ 2
            r1, r2 = target[1:mid], target[mid+1:end]
        end

        # Validate children for point count and temporal diversity
        v1 = length(r1) >= effective_min_p && length(unique([p[2] for p in r1])) >= cfg.min_time_slices
        v2 = length(r2) >= effective_min_p && length(unique([p[2] for p in r2])) >= cfg.min_time_slices

        if !v1 || !v2
             push!(unsplittable, objectid(target))
             continue
        end

        # Tentative update to check global area constraints
        tentative_regions = copy(regions)
        deleteat!(tentative_regions, target_idx)
        push!(tentative_regions, r1, r2)

        candidate_centroids = [(mean(p[1][1] for p in r), mean(p[1][2] for p in r)) for r in tentative_regions]
        polys_coords, _ = get_voronoi_polygons_and_edges(candidate_centroids, hull_geom)

        area_violation = any(
            p_coords -> !is_valid_polygon_coords(p_coords) || get_polygon_area(p_coords) < cfg.min_area,
            polys_coords
        )

        if area_violation && length(tentative_regions) > cfg.min_total_arealunits
             push!(unsplittable, objectid(target))
             continue
        end

        regions = tentative_regions
    end

    final_centroids_candidate = [(mean(p[1][1] for p in r), mean(p[1][2] for p in r)) for r in regions]

    # if length(final_centroids_candidate) < cfg.min_total_arealunits
    #     # Aggregate all original points (from 'data') into a single centroid
    #     all_pts_x = [p[1][1] for p in data]
    #     all_pts_y = [p[1][2] for p in data]
    #     return [ (mean(all_pts_x), mean(all_pts_y)) ], "insufficient_units_error"
    # else
        return final_centroids_candidate, termination_reason
    # end
end

 
 
function get_hvt_centroids(s_coord_tuple, cfg, hull_geom; max_iter=500)
    if isempty(s_coord_tuple); return [], "no_points_provided"; end

    # Internal utility for point-to-centroid distance
    dist(p1, p2) = sqrt(sum((p1 .- p2).^2))

    # Standardized refinement loop (Lloyd's update)
    function refine(pts_in, centroids, iters)
        curr = deepcopy(centroids)
        for _ in 1:iters
            groups = [Int[] for _ in 1:length(curr)]
            for (i, p) in enumerate(pts_in)
                dists = [dist(p, c) for c in curr]
                push!(groups[argmin(dists)], i)
            end
            new_c = [!isempty(idx) ? 
                     (mean(pts_in[j][1] for j in idx), mean(pts_in[j][2] for j in idx)) : 
                     curr[k] for (k, idx) in enumerate(groups)]
            if all(dist(new_c[j], curr[j]) < cfg.tolerance for j in 1:length(curr))
                return new_c
            end
            curr = new_c
        end
        return curr
    end

    # Initial Seed generation using k-means
    # Convert s_coord_tuple to matrix for Clustering.jl
    pts_matrix = hcat([p[1] for p in s_coord_tuple], [p[2] for p in s_coord_tuple])'
    k_target = max(1, cfg.min_total_arealunits)
    
    # Run k-means to find initial centers
    R = kmeans(pts_matrix, k_target)
    curr_centroids = [(R.centers[1, i], R.centers[2, i]) for i in 1:size(R.centers, 2)]
    
    # Iterative HVT process with advanced stopping conditions
    last_mean_density = 0.0
    last_cv = 0.0
    status = "max_iterations_reached"

    for i in 1:max_iter
        # 1. Assignment step to calculate metrics
        assignments = [argmin([dist(p, c) for c in curr_centroids]) for p in s_coord_tuple]
        counts = [count(==(k), assignments) for k in 1:length(curr_centroids)]
        
        curr_mean_density = mean(counts)
        cv_val = std(counts) / (curr_mean_density + 1e-9)

        # Convergence logic aligned with QVT/BVT
        if i > 5
            if abs(curr_mean_density - last_mean_density) < cfg.tolerance || abs(cv_val - last_cv) < cfg.tolerance
                if length(curr_centroids) >= cfg.target && all(c -> c <= cfg.max_points, counts)
                    status = "converged_constraints_satisfied"
                    break
                elseif abs(cv_val - cfg.target_cv) < cfg.tolerance
                    status = "converged_target_cv"
                    break
                elseif (count(>(cfg.max_points), counts) / length(curr_centroids)) < cfg.tolerance/10
                    status = "converged_minor_violations"
                    break
                end
            end
        end

        last_mean_density = curr_mean_density
        last_cv = cv_val

        # 2. Refinement step (Lloyd's update)
        new_centroids = refine(s_coord_tuple, curr_centroids, 3)
        
        # Check for centroid position stabilization
        if all(dist(new_centroids[j], curr_centroids[j]) < cfg.tolerance for j in 1:length(curr_centroids))
             # If positions stabilized but constraints aren't met, check if we should add a unit
             if length(curr_centroids) < cfg.max_total_arealunits && (length(curr_centroids) < cfg.target || any(counts .> cfg.max_points))
                 # Split the largest group to improve density balance
                 idx_to_split = argmax(counts)
                 group_pts = s_coord_tuple[assignments .== idx_to_split]
                 if length(group_pts) >= 2 * cfg.min_points
                     new_seeds = [(mean(p[1] for p in group_pts) * 0.99, mean(p[2] for p in group_pts) * 0.99), 
                                  (mean(p[1] for p in group_pts) * 1.01, mean(p[2] for p in group_pts) * 1.01)]
                     deleteat!(curr_centroids, idx_to_split)
                     append!(curr_centroids, new_seeds)
                     continue 
                 end
             end
             status = "converged_stable_positions"
             break
        end
        
        curr_centroids = new_centroids
    end

    return curr_centroids, status
end





function get_avt_centroids(s_coord_tuple, cfg, hull_geom)
    """
    Synopsis: Agglomerative Voronoi Tessellation (AVT) with diagnostic termination tracking.
    """
    if isempty(s_coord_tuple)
        return [], "no_points_provided"
    end

    if length(s_coord_tuple) <= cfg.min_total_arealunits
        return [ (mean(p[1] for p in s_coord_tuple), mean(p[2] for p in s_coord_tuple)) ], "not_enough_points_to_tessellate"
    end

    u_pts = unique(s_coord_tuple)
    # Start with maximum allowed units to give agglomeration room to satisfy constraints
    c_init = get_kde_seeds(u_pts, min(length(u_pts), cfg.max_total_arealunits))
    data = collect(zip(s_coord_tuple, cfg.t_idx))
    curr_c = [SVector{2, Float64}(c) for c in c_init]
    termination_reason = "min_units_reached"
    last_mean_density = 0.0
    last_cv = 0.0

    while length(curr_c) > cfg.min_total_arealunits
        # 1. Assignment Phase: Re-map points to current centroids
        assigns = [Int[] for _ in 1:length(curr_c)]
        for i in 1:length(data)
            # Find nearest centroid index
            d_pt = data[i][1]
            dist_idx = argmin([sum((d_pt .- c).^2) for c in curr_c])
            push!(assigns[dist_idx], i)
        end

        counts = length.(assigns)
        if isempty(counts); termination_reason = "no_units_formed"; break; end

        # 2. Geometry Audit
        polys_coords, _ = get_voronoi_polygons_and_edges([Tuple(c) for c in curr_c], hull_geom)
        areas = fill(0.0, length(curr_c))
        for i in 1:min(length(curr_c), length(polys_coords))
            if !is_valid_polygon_coords(polys_coords[i]); areas[i] = 0.0; continue; end
            areas[i] = get_polygon_area(polys_coords[i])
        end

        # 3. Violation Detection: Focus on "Too Small" or "Too Sparse" (Data Starvation)
        violators = []
        for k in 1:length(curr_c)
            ts_count = length(unique([data[idx][2] for idx in assigns[k]]))
            
            # min_points: If a unit has fewer points than required.
            # min_time_slices: If a unit spans too few distinct time slices.
            # min_area: If a unit's polygon area is below the threshold (only for areas > 0).
            # max_area: If a unit's polygon area exceeds the threshold.
            if counts[k] < cfg.min_points || 
               ts_count < cfg.min_time_slices || 
               (areas[k] > 0 && areas[k] < cfg.min_area) || 
               (areas[k] > cfg.max_area)
                push!(violators, k)
            end
        end

        # 4. Exit Conditions
        curr_mean_density = mean(counts)
        cv_val = std(counts) / (mean(counts) + 1e-9)

        # Increase likelihood of stopping early if system statistics stabilize.
        # Check for convergence in either density or uniformity (CV).
        if last_mean_density > 0.0 && (abs(curr_mean_density - last_mean_density) < cfg.tolerance || abs(cv_val - last_cv) < cfg.tolerance)
            if isempty(violators) && length(curr_c) <= cfg.target
                 termination_reason = "converged_target_reached"
                 break
            elseif ( cv_val - cfg.target_cv) < cfg.tolerance
                 termination_reason = "converged_target_cv"
                 break
            elseif (length(violators) / length(curr_c)) < cfg.tolerance/10 # Looser exit: stop if violations are minor
                 termination_reason = "converged_minor_violations"
                 break
            end
        end
        last_mean_density = curr_mean_density
        last_cv = cv_val

        # Logic for determining if we should continue merging
        must_merge = length(curr_c) > cfg.max_total_arealunits
        want_merge = length(curr_c) > cfg.target || !isempty(violators)

        if !must_merge && !want_merge
             termination_reason = "constraints_satisfied"
             break
        end

        # Agglomeration Phase: Select candidates for merging, prioritizing violators
        candidates_indices = isempty(violators) ? collect(1:length(curr_c)) : violators

        # Merge the unit with fewest points among candidates
        v_counts = [counts[k] for k in candidates_indices]
        target_idx = candidates_indices[argmin(v_counts)]

        if length(curr_c) <= cfg.min_total_arealunits ; break; end # Cannot merge further

        dists = [sum((curr_c[target_idx] .- curr_c[j]).^2) for j in 1:length(curr_c)]
        dists[target_idx] = Inf

        # Utility: find neighbor that doesn't violate max_points
        neighbor_indices = sortperm(dists)
        neighbor_idx = neighbor_indices[1] # Default to nearest
        for idx in neighbor_indices
            if counts[target_idx] + counts[idx] <= cfg.max_points
                neighbor_idx = idx
                break
            end
        end

        total_n = counts[target_idx] + counts[neighbor_idx]
        curr_c[neighbor_idx] = (curr_c[target_idx] .* counts[target_idx] .+ curr_c[neighbor_idx] .* counts[neighbor_idx]) ./ (total_n + 1e-9)

        deleteat!(curr_c, target_idx)
    end

    return [Tuple(c) for c in curr_c], termination_reason
end


function get_lattice_centroids(s_coord_tuple, lengthscale)
    """
    Synopsis: Generates centroids for a regular 2D lattice (grid) based on a lengthscale.
    """
    if isempty(s_coord_tuple); return [], 0, 0, (0.0, 0.0, 0.0, 0.0); end

    xs = [p[1] for p in s_coord_tuple]
    ys = [p[2] for p in s_coord_tuple]

    xmin, xmax = minimum(xs), maximum(xs)
    ymin, ymax = minimum(ys), maximum(ys)

    # Generate grid ranges
    x_range = collect(xmin:lengthscale:xmax)
    y_range = collect(ymin:lengthscale:ymax)

    # Ensure at least one cell if the range is smaller than lengthscale
    if isempty(x_range); x_range = [xmin]; end
    if isempty(y_range); y_range = [ymin]; end

    rows = length(y_range)
    cols = length(x_range)

    # Create meshgrid of centroids
    centroids = [(x, y) for y in y_range, x in x_range][:]

    return centroids, rows, cols, (xmin, xmax, ymin, ymax)
end



function load_shapefile_to_libgeos(filepath::String)
    # Read the shapefile
    # import Shapefile  << --- install this if you need it
    # import LibGEOS
    # import GeoInterface

    table = Shapefile.Table(filepath)
    
    # Extract geometries and convert to LibGEOS
    # GeoInterface allows LibGEOS to understand Shapefile objects automatically
    geoms = [LibGEOS.read_geom(row.geometry) for row in table]
    
    return geoms, table
end

function get_user_centroids(input_polygons)
    # Convert input to a concrete vector of LibGEOS Polygons
    geoms = LibGEOS.Polygon[p for p in input_polygons]
    n = length(geoms)
    centroids = Vector{Tuple{Float64, Float64}}(undef, n)
    polys_coords = Vector{Vector{Tuple{Float64, Float64}}}(undef, n)

    for i in 1:n
        poly = geoms[i]
        cent_geom = LibGEOS.centroid(poly)
        seq = LibGEOS.getCoordSeq(cent_geom)
        centroids[i] = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
        polys_coords[i] = get_coords_from_geom(poly)
    end

    # Wrap the vector in a GeometryCollection so GeoInterface traits are recognized
    collection = LibGEOS.GeometryCollection(geoms)
    # Perform unaryUnion on the collection instead of the vector
    united = LibGEOS.unaryUnion(collection)
    hull_coords = get_coords_from_geom(united)

    return centroids, polys_coords, hull_coords
end




function assign_spatial_units(input_data; area_method=:avt, target_units=10, lengthscale=nothing, input_polygons=nothing, geom_hull=nothing, kwargs...)
    # 1. Overload to handle adjacency matrices directly
    if input_data isa AbstractMatrix
        reason = :inferred
        W = input_data
        au_inferred = assign_spatial_units_inferred(W;
            iterations=get(kwargs, :iterations, 50),
            learning_rate=get(kwargs, :learning_rate, 0.1),
            buffer_dist=get(kwargs, :buffer_dist, 0.5),
            input_polygons=input_polygons)

        s_coord_tuple = au_inferred.centroids
        final_centroids = au_inferred.centroids
        new_assigns = [argmin([sum((p .- sj).^2) for sj in final_centroids]) for p in s_coord_tuple]
        polys_coords = au_inferred.polygons
        v_edges = au_inferred.adjacency_edges
        g = au_inferred.graph
        hull_coords = au_inferred.hull_coords

    # 2. Handle User-Defined Polygons
    elseif !isnothing(input_polygons)
        # If geom_hull is provided, we intersect the input polygons with it
        processed_polys = isnothing(geom_hull) ? input_polygons : [LibGEOS.intersection(p, geom_hull) for p in input_polygons]
        
        final_centroids, polys_coords, hull_coords = get_user_centroids(processed_polys)
        reason = :user_polygons
        n_units = length(final_centroids)

        g = SimpleGraph(n_units)
        for i in 1:n_units, j in (i+1):n_units
            if LibGEOS.touches(processed_polys[i], processed_polys[j]) || LibGEOS.intersects(LibGEOS.buffer(processed_polys[i], 1e-7), processed_polys[j])
                add_edge!(g, i, j)
            end
        end
        g = ensure_connected!(g, final_centroids)
        W = Float64.(Graphs.adjacency_matrix(g))

        new_assigns = [argmin([sum((p .- sj).^2) for sj in final_centroids]) for p in input_data]
        v_edges = []
        s_coord_tuple = input_data

    # 3. Handle Lattice Method
    elseif area_method == :lattice
        ls = isnothing(lengthscale) ? sqrt(get_polygon_area(get_coords_from_geom(expand_hull(input_data, 0.0))) / target_units) : lengthscale
        final_centroids_raw, rows, cols, bbox = get_lattice_centroids(input_data, ls)
        reason = :lattice_grid
        
        # Generate square polygons and clip them if geom_hull is provided
        polys_coords = Vector{Vector{Tuple{Float64, Float64}}}()
        lg_polys = LibGEOS.Polygon[]
        final_centroids = Tuple{Float64, Float64}[]
        half = ls / 2.0
        
        for c in final_centroids_raw
            coords = [[(c[1]-half, c[2]-half), (c[1]+half, c[2]-half), (c[1]+half, c[2]+half), (c[1]-half, c[2]+half), (c[1]-half, c[2]-half)]]
            p_geom = LibGEOS.Polygon(coords)
            if !isnothing(geom_hull)
                p_geom = LibGEOS.intersection(p_geom, geom_hull)
            end
            
            if !LibGEOS.isEmpty(p_geom)
                push!(lg_polys, p_geom)
                p_c = LibGEOS.centroid(p_geom)
                seq = LibGEOS.getCoordSeq(p_c)
                push!(final_centroids, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
                push!(polys_coords, get_coords_from_geom(p_geom))
            end
        end

        n_units = length(final_centroids)
        g = SimpleGraph(n_units)
        for i in 1:n_units, j in (i+1):n_units
            if LibGEOS.touches(lg_polys[i], lg_polys[j]) || LibGEOS.intersects(LibGEOS.buffer(lg_polys[i], 1e-7), lg_polys[j])
                add_edge!(g, i, j)
            end
        end
        g = ensure_connected!(g, final_centroids)
        W = Float64.(Graphs.adjacency_matrix(g))
        
        new_assigns = [argmin([sum((p .- sj).^2) for sj in final_centroids]) for p in input_data]
        v_edges = []
        hull_coords = isnothing(geom_hull) ? [(bbox[1], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[4]), (bbox[1], bbox[4]), (bbox[1], bbox[3])] : get_coords_from_geom(geom_hull)
        s_coord_tuple = input_data

    # 4. Standard Tessellation Methods
    else
        cfg = (
            target=Int(target_units),
            min_total_arealunits=Int(get(kwargs, :min_total_arealunits, 3)),
            max_total_arealunits=Int(get(kwargs, :max_total_arealunits, target_units*2)),
            min_time_slices=Int(get(kwargs, :min_time_slices, 1)),
            min_points=Int(get(kwargs, :min_points, 1)),
            max_points=Int(get(kwargs, :max_points, length(input_data))),
            min_area=get(kwargs, :min_area, 0.0),
            max_area=get(kwargs, :max_area, Inf),
            target_cv=get(kwargs, :target_cv, 1.0),
            tolerance=get(kwargs, :tolerance, 0.1),
            buffer_dist=get(kwargs, :buffer_dist, 0.5),
            t_idx=get(kwargs, :t_idx, ones(Int, length(input_data))))

        # Use provided geom_hull or fallback to expanded convex hull
        hull_geom = !isnothing(geom_hull) ? geom_hull : expand_hull(input_data, cfg.buffer_dist)
        
        c_mid, reason = if area_method == :cvt get_cvt_centroids(input_data, cfg, hull_geom)
        elseif area_method == :kvt get_kvt_centroids(input_data, cfg, hull_geom)
        elseif area_method == :qvt get_qvt_centroids(input_data, cfg, hull_geom)
        elseif area_method == :bvt get_bvt_centroids(input_data, cfg, hull_geom)
        elseif area_method == :hvt get_hvt_centroids(input_data, cfg, hull_geom)
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
            if LibGEOS.touches(lg_polys[i], lg_polys[j]) || LibGEOS.intersects(LibGEOS.buffer(lg_polys[i], 1e-7), lg_polys[j])
                add_edge!(g, i, j)
            end
        end
        g = ensure_connected!(g, final_centroids)
        hull_coords = get_coords_from_geom(hull_geom)
        s_coord_tuple = input_data
        W = Float64.(Graphs.adjacency_matrix(g))
    end

    return (centroids=final_centroids, assignments=new_assigns, polygons=polys_coords, 
            adjacency_edges=v_edges, graph=g, hull_coords=hull_coords, 
            termination_reason=reason, s_coord_tuple=s_coord_tuple, W=W, s_vals=collect(1:size(W,1)))
end




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
        plt = plot_spatial_graph(lip_inputs.s_coord_tuple, areal_units; title="Lip Cancer Spatial Graph", domain_boundary=lip_inputs.s_coord_tuple)
        display(plt)
        
        println("First few centroids from areal_units: ", areal_units.centroids[1:min(5, length(areal_units.centroids))])
    """
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

    # Deduplicate centroids before triangulation to suppress package warnings 
    # and ensure the output polygon array matches the input centroid array in length.
    u_centroids = unique(centroids)
    if length(u_centroids) < n_c
        u_polys, u_edges = get_voronoi_polygons_and_edges(u_centroids, hull_geom, tol)
        return [u_polys[findfirst(==(c), u_centroids)] for c in centroids], u_edges
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


 function plot_spatial_graph(au; plot_title="Spatial Partitioning", domain_boundary=nothing)
    # 1. Base Plot - Use qualified Plots.plot and Plots.title!
    plt = Plots.plot(aspect_ratio=:equal, legend=false)
    Plots.title!(plt, plot_title)

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

    # 2. Plot Adjacency Graph Edges
    for edge in Graphs.edges(au.graph)
        u, v = Graphs.src(edge), Graphs.dst(edge)
        p1, p2 = au.centroids[u], au.centroids[v]
        Plots.plot!(plt, [p1[1], p2[1]], [p1[2], p2[2]], color=:red, lw=1.5, alpha=0.6)
    end

    # 3. Plot Centroids and Raw Points
    Plots.scatter!(plt, [p[1] for p in au.s_coord_tuple], [p[2] for p in au.s_coord_tuple], 
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

    

################
 

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
  coords = spatial_coords[:, space_indices] # 2 x y_N
  times = time_indices' # 1 x y_N

  # Stack to create 3D input: [Time; Lat; Lon]
  return ColVecs(vcat(times, coords))
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
        # Dimensional fix: Removed vec() to allow multi-feature inputs (N x D)
        Ko = kernelmatrix( fkernal, Xobs )
        Kcommon = inv(Ko + lambda*I) 

        if !isnothing(Xinducing)
            Ki = kernelmatrix( fkernal, Xinducing )
            Kio = kernelmatrix( fkernal, Xinducing, Xobs ) 
            Yinducing_mean_process = Kio * Kcommon * Yobs 
            Covi = Symmetric( Ki - Kio * Kcommon * Kio'  + lambda*I )
            MVNi = MvNormal( Yinducing_mean_process, Covi )

            Yinducing_sample  = rand( MVNi )
            Li =  cholesky(Symmetric( Ki + lambda*I)).L 

            Yobs_mean_process =  Kio' * ( Li' \ (Li \ Yinducing_mean_process  ) ) 
            Covo = Symmetric(kernelmatrix( fkernal, Xobs ) + lambda*I)
            MVN = MvNormal(Yobs_mean_process, Covo)

            if returntype=="fcovariance"
                return MVN
            end

            Yobs_sample =  Kio' * ( Li' \ (Li \ Yinducing_sample  ) ) 

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
            mean_process = Ko * Kcommon * Yobs   
            MVN = MvNormal(mean_process, Ko + lambda*I  ) 

            if returntype=="fcovariance"
                return MVN
            end

            Yobs_sample = rand( MVN ) 

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
        if !isnothing(Xinducing)
            Ko = kernelmatrix( fkernal, Xobs )
            Ki = kernelmatrix( fkernal, Xinducing )
            Kio = kernelmatrix( fkernal, Xinducing, Xobs )
            Lo = cholesky(Symmetric( Ko + lambda*I)).L
            Li = cholesky(Symmetric( Ki + lambda*I)).L
            Yinducing_mean_process  = Kio * ( Lo' \ (Lo \ Yobs ) ) 

            Covi = Symmetric( Ki + lambda*I)
            MVN = MvNormal( Yinducing_mean_process, Covi )

            if returntype=="fcovariance"
                return MVN
            end

            Yobs_mean_process = Kio' * ( Li' \ (Li \ Yinducing_mean_process )) 
            Yinducing_sample  = Yinducing_mean_process + Li * rand(Normal(0, 1), size(Li,1))
            Yobs_sample = Yobs_mean_process + Lo * rand(Normal(0, 1), size(Lo,2))

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
            Ko = kernelmatrix( fkernal, Xobs )
            Lo = cholesky(Symmetric( Ko + lambda*I)).L
            Yobs_mean_process = Ko' * ( Lo' \ (Lo \ Yobs )) 
            Covo = Symmetric( Ko + lambda*I)
            MVN = MvNormal( Yobs_mean_process, Covo ) 

            if returntype=="fcovariance"
                return MVN
            end

            Yobs_sample = Yobs_mean_process + Lo * rand(Normal(0, 1), size(Lo,2))

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


   
    
function assign_time_units(t_coord; time_method="regular", t_N=nothing, u_N=12, kwargs...)

    if time_method=="regular"

        tint = Int.(floor.(t_coord))
        t0, t1 = minimum(tint), maximum(tint)
        t_n = t1-t0
        if !isnothing(t_N) 
            if t_n != t_N
                print("warning: time range and unique years do not match")
            end
        end

        t_idx = tint .- t0 .+ 1
        t_vals = collect(t0:t1) .- t0 .+ 1
        t_yr = collect(t0:t1)
        t_brks = (t_yr, t1+1)
        t_mids = t_yr .+ 0.5
        
        u_coord = t_coord - tint

        u_disc = discretize_data( u_coord, N_cat=u_N, method="regular" )  # seasonality discretized

        return (
            t_coord = t_coord, 
            t_idx = t_idx, 
            t0=t0, 
            t1=t1, 
            t_vals, 
            t_yr=t_yr, 
            t_mids=t_mids, 
            t_brks=t_brks,
            tn=length(t_vals),
            t_N= length(t_vals),
            u_coord=u_coord, 
            u_idx=u_disc.idx, 
            u_brks=u_disc.brks,
            u_mids=u_disc.mids, 
            u_N=u_N,
            u_vals=collect(1:u_N) 
        )
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



function estimate_local_kde_with_extrapolation(s_coord_tuple, t_idx, target_ts; grid_res=600, sd_extension_factor=0.25)
    """
    Synopsis: Estimates 2D KDE for a specific time slice with extrapolation.
    Inputs:
    - s_coord_tuple: Vector of (x, y) coordinates for all time points.
    - t_idx: Vector of time indices corresponding to s_coord_tuple.
    - target_ts: The specific time slice to estimate KDE for.
    - grid_res: Resolution of the output grid (e.g., 100 for 100x100 grid).
    - sd_extension_factor: Multiplier for standard deviation to define the bandwidth.
    Outputs:
    - Tuple (x_grid, y_grid, intensity) where intensity is a matrix.
    """
    # Filter points for the target time slice
    filtered_pts = [p for (i, p) in enumerate(s_coord_tuple) if t_idx[i] == target_ts]
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
    assigns = [argmin([sum((p .- c).^2) for c in au.centroids]) for p in au.s_coord_tuple]
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



function plot_kde_simple(s_coord_tuple; grid_res=600, sd_extension_factor=0.25, title="Spatial Intensity (KDE)")
    # Internal wrapper for estimate_local_kde_with_extrapolation
    # Description: Generates a simple 2D Heatmap of spatial intensity using Kernel Density Estimation.
    # Inputs:
    #   - s_coord_tuple: Vector of (x, y) coordinate tuples.
    #   - grid_res: Resolution of the output grid.
    #   - sd_extension_factor: Factor to extend the bandwidth standard deviation.
    #   - title: Title for the generated plot.
    # Outputs:
    #   - A Plots.Plot object (Heatmap with scatter overlay).
    # Using a dummy t_idx of 1s since we are plotting a static slice
    t_idx_dummy = ones(Int, length(s_coord_tuple))
    x_g, y_g, intensity = estimate_local_kde_with_extrapolation(s_coord_tuple, t_idx_dummy, 1; grid_res=grid_res, sd_extension_factor=sd_extension_factor)

    plt = Plots.heatmap(x_g, y_g, intensity',
                  title=title,
                  c=:viridis,
                  aspect_ratio=:equal,
                  xlabel="X", ylabel="Y")
    Plots.scatter!(plt, [p[1] for p in s_coord_tuple], [p[2] for p in s_coord_tuple],
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
    s_coord_tuple = repeat(pts_base, n_years)
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
        s_idx=s_idx, n_years=n_years, s_coord_tuple=s_coord_tuple, W=W, au=au
    )
end
  


function icar_form(theta, phi, sigma, rho)
    # https://sites.stat.columbia.edu/gelman/research/published/bym_article_SSTEproof.pdf
    # Reibler parameterization: https://pubmed.ncbi.nlm.nih.gov/27566770/
    # https://www.jstatsoft.org/index.php/jss/article/view/v063c01/841
    sigma .* ( sqrt.(1 .- rho) .* theta .+ sqrt.(rho ./ scaling_factor) .* phi )  
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



function generate_inducing_points(coords, M_inducing, seed=42; method="kmeans")

    # Helper function to generate inducing points (simple random sampling for now)

    Random.seed!(seed)
    n_data = size(coords, 1)
    if M_inducing >= n_data
        return coords # If M >= N, just use all data points (becomes exact GP)
    end

    
    if method=="random"
        indices = sample(1:n_data, M_inducing, replace=false)
        return coords[indices, :]
    end

    if method=="kmeans"
        #   Identifies optimal inducing point locations using K-Means clustering.
        #   Essential for Sparse/FITC Gaussian Process models.
        # Inputs:
        #   coords: N x D matrix of spatiotemporal coordinates.
        #   M_inducing: Target number of inducing points.
        # Outputs:
        #   M x D matrix of cluster centroids.

        # Transpose for Clustering.jl compatibility
        data_matrix = Matrix(coords')
        
        # Execute K-Means
        clustering_result = kmeans(data_matrix, M_inducing, maxiter=200)
        
        # Extract centroids and transpose back
        inducing_points = clustering_result.centers'
        
        return inducing_points
    end

    if method=="furthest_point"

        # Initialize with a random point
        inducing_points_idx = [rand(1:n_data)]
        distances = fill(Inf, n_data)

        # Convert coords to the expected format for pairwise
        coords_matrix = permutedims(coords)

        for _ in 2:M_inducing
            # Calculate distances from all points to the newest inducing point
            last_added_idx = inducing_points_idx[end]
            new_distances = colwise(Euclidean(), coords_matrix, coords_matrix[:, last_added_idx])

            # Update minimum distances to any inducing point found so far
            distances = min.(distances, new_distances)

            # Find the point farthest from any existing inducing point
            farthest_idx = argmax(distances)
            push!(inducing_points_idx, farthest_idx)
        end

        return coords[inducing_points_idx, :], inducing_points_idx

    end

end

 
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
 

function generate_informed_rff_params(coords, M_rff)
    # Ensure coords is treated as a matrix for variance calculation
    # Fix for the RowVecs/SubArray abs2 issue: convert to standard Matrix{Float64}
    mat = coords isa AbstractMatrix ? Matrix{Float64}(coords) : 
          reduce(hcat, [collect(p) for p in coords])'
    
    d = size(mat, 2)
    # Calculate standard deviation per dimension
    coord_scales = std(mat, dims=1) .+ 1e-6
    
    # Scale random frequencies W by the inverse of coordinate scales
    W = randn(d, M_rff) ./ coord_scales'
    
    # Random phase shifts, uniformly distributed in [0, 2π]
    b = rand(M_rff) .* (2 * pi)
    return W, b
end


function generate_rff_params_for_se_kernel(D_in, M_rff, lengthscale)
    # Helper function to generate RFF parameters for a Squared Exponential kernel
    # For a Squared Exponential kernel, the spectral density is Gaussian: N(0, (1/l)^2 * I)
    sigma_spectral = 1.0 / lengthscale
    W_matrix = randn(D_in, M_rff) .* sigma_spectral # D_in x M_rff matrix
    b_vector = rand(Uniform(0, 2pi), M_rff)
    return W_matrix, b_vector
end 
 

macro save_carstm_state(file_to_save_name_sym)
  quote
    try
      # Evaluate the input symbol (e.g., :state_filename) to its value (e.g., "carstm_state.jld2")
      local filename_val = $(esc(file_to_save_name_sym))
      @info "Saving CARSTM state to $(filename_val)..."
      # JLD2.@save expects variable names as symbols, not their values.
      # The variables themselves should be directly passed.
      JLD2.@save "$(filename_val)" areal_units mod chain s_coord_tuple y_sim y_binary t_idx weights trials cov_indices cov_indices trials_sim  weights_sim adj_matrix_numeric s_N t_N area_method
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
      JLD2.@load "$(filename_val)" areal_units mod chain s_coord_tuple y_sim y_binary t_idx weights trials cov_indices cov_indices trials_sim  weights_sim adj_matrix_numeric s_N t_N area_method
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


 
 

function plot_posterior_results(stats, M=nothing, areal_units=nothing; s_coord_tuple=nothing, time_slice=nothing, effect=:spatial, cov_idx=1, show_pts=false)
    # Description: Comprehensive posterior visualization for CARSTM and Deep GP models.

    # Extract target stats and guard against nothing or scalar values
    st = getproperty(stats, effect)
    isnothing(st) && return nothing
    if st isa Real
        return Plots.plot(title="$effect (Fixed: $st)")
    end

    if !isnothing(M)
        s_coord_tuple = M.s_coord_tuple
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
            StatsPlots.scatter!(plt, [p[1] for p in s_coord_tuple], [p[2] for p in s_coord_tuple],
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


function bstm( inp::NamedTuple )
    arch = inp[:model_arch]
    if arch == "multivariate" return bstm_multivariate(inp)
    elseif arch == "multifidelity" return bstm_multifidelity(inp)
    else return bstm_univariate(inp) end
end


function bstm(formula::Union{String, StatsModels.FormulaTerm}, data_input::Union{DataFrame, NamedArray};
    model_family="gaussian",
    model_arch="univariate",
    multifidelity_data=nothing,
    return_data=false,
    contrasts=Dict{Symbol, Any}(),
    kwargs...)

    # 1. Initialize and Standardize Input
    data = data_input isa DataFrame ? copy(data_input) : DataFrame(data_input, :auto)
    opt_kwargs = Dict{Symbol, Any}(kwargs)
    internal_contrasts = copy(contrasts)

    # 2. Formula Breakdown
    f_str = string(formula)
    sides = Base.split(f_str, "~")
    lhs_side = strip(sides[1])
    rhs = strip(sides[2])

    # Multivariate Routing: Detect multiple response variables
    lhs_vars = Symbol.(strip.(Base.split(lhs_side, "+")))
    if length(lhs_vars) > 1
        opt_kwargs[:model_arch] = "multivariate"
        opt_kwargs[:y_obs] = Matrix(data[!, lhs_vars])
        opt_kwargs[:outcomes_N] = length(lhs_vars)
    else
        opt_kwargs[:model_arch] = model_arch
    end

    # 3. Manifold Discovery (Capturing all term variants)
    re_rules = Dict{String, Dict{Symbol, Any}}()
    fixed_parts = String[]
    mixed_terms = []
    interaction_terms = []
    eigen_terms = []
    svc_covs = Symbol[]
    svc_model = get(opt_kwargs, :svc_model, "rff")

    terms = strip.(Base.split(rhs, "+"))

    for term in terms
        # Interaction Effects (ie): 2D Smooths with RFF generation
        if occursin("ie(", term)
            m_ie = match(r"ie\(([^,]+),\s*([^,)]+)(?:,\s*(.*))?\)", term)
            if !isnothing(m_ie)
                v1, v2 = strip(m_ie.captures[1]), strip(m_ie.captures[2])
                m_type = "gp"
                if !isnothing(m_ie.captures[3])
                    pm = match(r"model=['\"]([^'\"]+)['\"]", m_ie.captures[3])
                    if !isnothing(pm) m_type = pm.captures[1] end
                end
                coords_ie = hcat(Vector{Float64}(data[!, Symbol(v1)]), Vector{Float64}(data[!, Symbol(v2)]))
                M_rff_ie = get(opt_kwargs, :M_rff_ie, 20)
                W_ie, b_ie = generate_informed_rff_params(coords_ie, M_rff_ie)
                push!(interaction_terms, (v1=v1, v2=v2, model=m_type, coords=coords_ie, W_ie=W_ie, b_ie=b_ie, M_rff=M_rff_ie))
            end
        # Eigen-effects (ee): Multivariate/Factor manifolds
        elseif occursin("ee(", term)
            m_ee = match(r"ee\((.*?)(?:,\s*model=['\"]([^'\"]+)['\"])?.*\)", term)
            if !isnothing(m_ee)
                vars_str = m_ee.captures[1]
                model_type = m_ee.captures[2]
                ee_vars = Symbol.(strip.(Base.split(vars_str, ",")))
                ee_data = Matrix{Float64}(data[!, ee_vars])
                n_dims = size(ee_data, 2)
                ltri = [i + (k-1)*n_dims for k in 1:n_dims for i in k:n_dims]
                push!(eigen_terms, (data=ee_data, n_dims=n_dims, ltri_indices=ltri, model=!isnothing(model_type) ? model_type : "householder", indices_for_y = 1:size(data,1)))
            end

        # Mixed Effects (me): Random Slopes
        elseif occursin("me(", term)
            m = match(r"me\((.*)\|(.*)\)", term)
            if !isnothing(m)
                lhs_parts = strip.(Base.split(m.captures[1], "+"))
                group_var = Symbol(strip(m.captures[2]))
                lvls = unique(data[!, group_var])
                g_map = Dict(v => i for (i, v) in enumerate(lvls))
                indices = [g_map[v] for v in data[!, group_var]]
                for lp in lhs_parts
                    cov_vals = (lp == "1") ? ones(size(data, 1)) : Vector{Float64}(data[!, Symbol(lp)])
                    push!(mixed_terms, (indices=indices, n_cat=length(lvls), covariate_vals=cov_vals, name=group_var))
                end
            end

        # Spatially Varying Coefficients (svc)
        elseif occursin("svc(", term)
            m_svc = match(r"svc\(([^,)]+)(?:,\s*model=['\"]([^'\"]+?)['\"])?.*\)", term)
            if !isnothing(m_svc)
                var_name = Symbol(strip(m_svc.captures[1]))
                push!(svc_covs, var_name)
                if !isnothing(m_svc.captures[2]); svc_model = m_svc.captures[2]; end
                if !(string(var_name) in fixed_parts); push!(fixed_parts, string(var_name)); end
            end

        # Categorical/Structural Random Effects (ce, re, st)
        elseif occursin("re(", term) || occursin("st(", term) || occursin("ce(", term)
            m_re = match(r"(?:re|st|ce)\(([^,)]+)(?:,\s*(.*))?\)", term)
            if !isnothing(m_re)
                var_name = strip(m_re.captures[1])
                params = Dict{Symbol, Any}()
                if !isnothing(m_re.captures[2])
                    p_parts = strip.(Base.split(m_re.captures[2], ","))
                    for p in p_parts
                        kv = strip.(Base.split(p, "="))
                        if length(kv) == 2; params[Symbol(kv[1])] = replace(kv[2], r"['\"]" => ""); end
                    end
                end
                if !haskey(params, :model); params[:model] = occursin("st(", term) ? "IV" : "bym2"; end
                re_rules[var_name] = params
            end
        else
            push!(fixed_parts, term)
        end
    end

    # 4. Packaging and Delegation to bstm_options
    opt_kwargs[:data] = data
    opt_kwargs[:fixed_parts] = fixed_parts
    opt_kwargs[:contrasts] = internal_contrasts
    opt_kwargs[:re_rules] = re_rules
    opt_kwargs[:mixed_terms] = mixed_terms
    opt_kwargs[:interaction_terms] = interaction_terms
    opt_kwargs[:eigen_terms] = eigen_terms
    opt_kwargs[:svc_covariates] = svc_covs
    opt_kwargs[:svc_model] = svc_model
    opt_kwargs[:model_family] = model_family

    inp = bstm_options(; opt_kwargs...)
    if return_data == "inputs" return inp end

    arch = inp[:model_arch]
    if arch == "multivariate" return bstm_multivariate(inp)
    elseif arch == "multifidelity" return bstm_multifidelity(inp)
    else return bstm_univariate(inp) end
end

function bstm_options(; kwargs...)
    # --- 1. INITIALIZATION & DICT SETUP ---
    # Convert NamedTuple/kwargs to Dict for flexible manipulation
    M = Dict{Symbol, Any}(kwargs)
    data = get(M, :data, nothing)

    # --- 2. AUTOMATED KEYWORD DISCOVERY ---
    # Maps common dataset naming conventions to internal BSTM parameters
    if !isnothing(data)
        v_names = Symbol.(names(data))
        keyword_map = Dict(
            :y_obs => [:y, :y_obs, :response],
            :s_coord_tuple => [:s_coord, :coords, :spatial_coords],
            :t_coord => [:t_coord, :time, :time_coords],
            :s_idx => [:s_idx, :space_idx],
            :t_idx => [:t_idx, :time_idx, :ti],
            :u_idx => [:u_idx, :season_idx, :ui],
            :log_offset => [:log_offset, :offset],
            :weights => [:weights, :wts],
            :trials => [:trials, :n_trials]
        )
        for (key, candidates) in keyword_map
            if !haskey(M, key)
                for cand in candidates
                    if cand in v_names
                        M[key] = data[!, cand]
                        break
                    end
                end
            end
        end
    end

    # --- 3. ARCHITECTURAL DEFAULTS ---
    get!(M, :model_arch, "univariate")
    get!(M, :model_family, "gaussian")
    get!(M, :model_space, "bym2")
    get!(M, :model_time, "ar1")
    get!(M, :model_season, "none")
    get!(M, :model_st, "none")
    get!(M, :noise, 1e-6)
    get!(M, :use_zi, false)
    get!(M, :use_sv, false)

    # --- 4. HARDENED VALIDATION: Field Existence ---
    if isnothing(get(M, :y_obs, nothing))
        @error "bstm_options: :y_obs (response variable) is missing. Model cannot be initialized."
        throw(ArgumentError("y_obs is required"))
    end

    # --- 5. DIMENSIONALITY & INDEX SAFETY ---
    M[:y_N] = size(M[:y_obs], 1)
    
    # Adjacency matrix W validation and fallback
    if !haskey(M, :W) || isnothing(M[:W])
        # Infer spatial units if W is missing, else default to 1
        M[:W] = sparse(LinearAlgebra.I(get(M, :s_N, 1)))
    elseif size(M[:W], 1) != size(M[:W], 2)
        @error "bstm_options: Adjacency matrix W must be square."
        throw(DimensionMismatch("W matrix dimensions $(size(M[:W])) must be square."))
    end
    M[:s_N] = size(M[:W], 1)

    # Chronological and Seasonal indexing
    get!(M, :t_idx, ones(Int, M[:y_N]))
    M[:t_N] = get(M, :t_N, isempty(M[:t_idx]) ? 1 : maximum(M[:t_idx]))
    get!(M, :u_idx, ones(Int, M[:y_N]))
    get!(M, :u_N, 12)
    get!(M, :s_idx, ones(Int, M[:y_N])) 
    
    # Derived Spatiotemporal Indexing
    M[:st_idx] = (M[:t_idx] .- 1) .* M[:s_N] .+ M[:s_idx]

    # --- 6. LIKELIHOOD VECTORIZATION SAFETY ---
    # Ensure weights and trials match y_N length to avoid BoundsErrors in Turing likelihood
    get!(M, :weights, ones(M[:y_N]))
    if length(M[:weights]) != M[:y_N]
        throw(DimensionMismatch("Weights length $(length(M[:weights])) != y_N ($(M[:y_N]))"))
    end
    
    get!(M, :trials, ones(Int, M[:y_N]))
    if length(M[:trials]) != M[:y_N]
        throw(DimensionMismatch("Trials length $(length(M[:trials])) != y_N ($(M[:y_N]))"))
    end

    # --- 7. COORDINATE MATRICES & RFF CACHING ---
    if !haskey(M, :s_coord_mat) && haskey(M, :s_coord_tuple)
        try 
            M[:s_coord_mat] = Matrix{Float64}(reduce(hcat, [collect(p) for p in M[:s_coord_tuple]])')
        catch e
            @error "bstm_options: Failed to convert s_coord_tuple to Matrix."
            rethrow(e)
        end
    end
    M[:s_coord] = haskey(M, :s_coord_mat) && !isnothing(M[:s_coord_mat]) ? RowVecs(M[:s_coord_mat]) : nothing

    # Spatially Varying Coefficients (SVC) Basis Caching
    if !isempty(get(M, :svc_covariates, Symbol[]))
        get!(M, :svc_M_rff, 20)
        if isnothing(get(M, :W_fixed_svc, nothing)) && !isnothing(M[:s_coord_mat])
            W_s, b_s = generate_informed_rff_params(M[:s_coord_mat], M[:svc_M_rff])
            M[:W_fixed_svc] = W_s; M[:b_fixed_svc] = b_s
        end
        if haskey(M, :W_fixed_svc) && !isnothing(M[:W_fixed_svc])
            M[:svc_basis_cached] = sqrt(2.0 / M[:svc_M_rff]) .* cos.((M[:s_coord_mat] * M[:W_fixed_svc]) .+ M[:b_fixed_svc]')
        end
    end

    # Interaction Terms RFF parameters generation
    iterms = get(M, :interaction_terms, [])
    if !isempty(iterms)
        M_rff_ie = get(M, :M_rff_ie, 20)
        new_iterms = []
        for iterm in iterms
            if !haskey(iterm, :W_ie) || isnothing(iterm.W_ie)
                if !haskey(iterm, :coords)
                    @error "bstm_options: Interaction term missing 'coords' for spectral generation."
                    continue
                end
                W_ie, b_ie = generate_informed_rff_params(iterm.coords, M_rff_ie)
                push!(new_iterms, merge(iterm, (W_ie=W_ie, b_ie=b_ie, M_rff=M_rff_ie)))
            else
                push!(new_iterms, iterm)
            end
        end
        M[:interaction_terms] = new_iterms
    end

    # --- 8. DESIGN MATRIX CONSTRUCTION (CRITICAL METADATA) ---
    if !isnothing(data) && haskey(M, :fixed_parts)
        fixed_expr = isempty(M[:fixed_parts]) ? "1" : join(M[:fixed_parts], " + ")
        M[:Xfixed] = Matrix{Float64}(create_fixed_design(fixed_expr, data; contrasts=get(M, :contrasts, Dict())))
    else
        get!(M, :Xfixed, NamedArray(ones(M[:y_N], 1), (1:M[:y_N], [:Intercept])))
    end
    # CRITICAL: Store Xfixed_N to ensure compatibility with get_optimal_sampler and bstm_univariate
    M[:Xfixed_N] = size(M[:Xfixed], 2)

    # --- 9. HIERARCHICAL & CATEGORICAL MANIFOLDS ---
    M[:spatial_hierarchy] = Dict{Symbol, Any}()
    M[:c_groups] = Dict{Symbol, Vector{Int}}()
    M[:c_re_templates] = Dict{Symbol, Any}()
    re_rules = get(M, :re_rules, Dict{String, Any}())

    if !isnothing(data)
        for (c_name, params) in re_rules
            c_sym = Symbol(c_name)
            if c_sym in Symbol.(names(data))
                lvls = unique(data[!, c_sym])
                g_map = Dict(v => i for (i, v) in enumerate(lvls))
                M[:c_groups][c_sym] = [g_map[v] for v in data[!, c_sym]]
                m_type = params isa Dict ? get(params, :model, "bym2") : "bym2"
                M[:spatial_hierarchy][c_sym] = (
                    indices = M[:c_groups][c_sym],
                    n_units = length(lvls),
                    model = m_type,
                    template = build_structure_template(Symbol(m_type), length(lvls); W=sparse(I(length(lvls))))
                )
                M[:c_re_templates][c_sym] = M[:spatial_hierarchy][c_sym].template
            end
        end
    end

    # --- 10. PRECISION TEMPLATES & CLEANUP ---
    M[:s_Q_template] = build_structure_template(Symbol(M[:model_space]), M[:s_N]; W=M[:W])
    M[:t_Q_template] = build_structure_template(Symbol(M[:model_time]), M[:t_N])
    M[:u_Q_template] = (M[:model_season] != "none") ? build_structure_template(Symbol(M[:model_season]), M[:u_N]) : nothing

    get!(M, :mixed_terms, [])
    get!(M, :eigen_terms, [])
    get!(M, :basis_matrices, Dict())

    # Return as NamedTuple for drop-in compatibility with Turing models
    return NamedTuple(M)
end


function apply_discretization_logic(vals, rules)
    groups = nothing
    new_vals = nothing

    if rules == 0 || isnothing(rules)
        groups = collect(1:length(vals))
    elseif rules == "unit"
        min_v, max_v = minimum(vals), maximum(vals)
        new_vals = (min_v == max_v) ? zeros(length(vals)) : (vals .- min_v) ./ (max_v - min_v)
        groups = collect(1:length(vals))
    elseif rules == "zscore"
        m, s = Statistics.mean(vals), Statistics.std(vals)
        new_vals = (s ≈ 0.0) ? zeros(length(vals)) : (vals .- m) ./ s
        groups = collect(1:length(vals))
    elseif rules == "log"
        new_vals = log.(vals .+ 1.0 .- minimum(vals))
        groups = collect(1:length(vals))
    elseif rules isa Int
        if rules > 1
            qs = unique(sort(quantile(vals, (0:rules) ./ rules)))
            groups = (length(qs) < 2) ? ones(Int, length(vals)) : clamp.(map(x -> searchsortedlast(qs, x), vals), 1, length(qs)-1)
        else
            groups = ones(Int, length(vals))
        end
    elseif rules isa AbstractString && startswith(rules, "regular:")
        n = parse(Int, Base.split(rules, ":")[2])
        q025, q975 = quantile(vals, 0.025), quantile(vals, 0.975)
        bins = unique(sort(collect(range(q025, stop=q975, length=n+1))))
        groups = (length(bins) < 2) ? ones(Int, length(vals)) : clamp.(map(x -> searchsortedlast(bins, x), vals), 1, length(bins)-1)
    elseif rules isa AbstractVector
        groups = clamp.(map(x -> searchsortedlast(rules, x) + 1, vals), 1, length(rules) + 1)
    end
    return new_vals, groups
end

function assign_covariate_units(cov_data_base, cov_discretization, re_rules, cov_interactions)
    cov_data_for_processing = deepcopy(cov_data_base)
    cov_groups = Dict{Symbol, Vector{Int}}()
    cov_re_structures = Dict{Symbol, Any}()

    if !isnothing(cov_discretization)
        for cov_name_sym in names(cov_data_base, 1)
            if haskey(cov_discretization, cov_name_sym)
                rules = cov_discretization[cov_name_sym]
                vals = cov_data_for_processing[cov_name_sym, :]
                new_vals, groups = apply_discretization_logic(vals, rules)
                if !isnothing(new_vals); cov_data_for_processing[cov_name_sym, :] = new_vals; end
                if !isnothing(groups)
                    cov_groups[cov_name_sym] = groups
                    if haskey(re_rules, cov_name_sym) && length(unique(groups)) > 1
                        n_bins = length(unique(groups))
                        if re_rules[cov_name_sym] == "rw2"; cov_re_structures[cov_name_sym] = build_bstm_rw2_template(n_bins).matrix
                        elseif re_rules[cov_name_sym] == "ar1"; cov_re_structures[cov_name_sym] = build_bstm_ar1_template(n_bins).matrix
                        end
                    end
                end
            end
        end
    end

    for inter_str in cov_interactions
        parts = Base.split(inter_str, "*")
        if length(parts) == 2
            n1, n2 = Symbol(parts[1]), Symbol(parts[2])
            if n1 in names(cov_data_for_processing, 1) && n2 in names(cov_data_for_processing, 1)
                inter_val = cov_data_for_processing[n1, :] .* cov_data_for_processing[n2, :]
                new_row = NamedArray(inter_val', (Symbol[Symbol(inter_str)], names(cov_data_for_processing, 2)))
                cov_data_for_processing = vcat(cov_data_for_processing, new_row)
                if !isnothing(cov_discretization) && haskey(cov_discretization, Symbol(inter_str))
                    rule_int = cov_discretization[Symbol(inter_str)]
                    iv_vals, iv_groups = apply_discretization_logic(inter_val, rule_int)
                    if !isnothing(iv_vals); cov_data_for_processing[Symbol(inter_str), :] = iv_vals; end
                    if !isnothing(iv_groups)
                        cov_groups[Symbol(inter_str)] = iv_groups
                        if haskey(re_rules, Symbol(inter_str)) && length(unique(iv_groups)) > 1
                            n_bins = length(unique(iv_groups))
                            if re_rules[Symbol(inter_str)] == "rw2"; cov_re_structures[Symbol(inter_str)] = build_bstm_rw2_template(n_bins).matrix
                            elseif re_rules[Symbol(inter_str)] == "ar1"; cov_re_structures[Symbol(inter_str)] = build_bstm_ar1_template(n_bins).matrix
                            end
                        end
                    end
                end
            end
        end
    end
    return cov_data_for_processing, cov_groups, cov_re_structures
end


# -----------------------


function get_chain_names(chain)
    try
        return string.(FlexiChains.parameters(chain))
    catch
        return string.(names(chain))
    end
end


function _extract_volatility(chain, name_strs, N_tot, N_samples, outcome_idx=nothing)
    y_sig_samples = zeros(N_tot, N_samples)
    for j in 1:N_samples
        local sig_y
        if isnothing(outcome_idx)
            if "y_sigma" in name_strs
                val = chain[:y_sigma][j]
                sig_y = val isa AbstractVector ? vec(collect(val)) : fill(Float64(val), N_tot)
            elseif "y_sigma_const" in name_strs
                sig_val = Float64(chain[:y_sigma_const][j])
                sig_y = fill(sig_val, N_tot)
            else
                sig_y = fill(1.0, N_tot)
            end
        else
            v_key = "y_sigma_k[$outcome_idx]"
            c_key = "y_sigma_const_k[$outcome_idx]"
            if v_key in name_strs
                val = chain[Symbol(v_key)][j]
                sig_y = val isa AbstractVector ? vec(collect(val)) : fill(Float64(val), N_tot)
            elseif c_key in name_strs
                sig_val = Float64(chain[Symbol(c_key)][j])
                sig_y = fill(sig_val, N_tot)
            else
                sig_y = fill(1.0, N_tot)
            end
        end

        # Final flattening and type enforcement to prevent MethodError
        flat_sig = vec(Float64.(collect(sig_y)))
        
        if length(flat_sig) >= N_tot
            y_sig_samples[:, j] = flat_sig[1:N_tot]
        else
            y_sig_samples[1:length(flat_sig), j] = flat_sig
            y_sig_samples[length(flat_sig)+1:end, j] .= flat_sig[end]
        end
    end
    return y_sig_samples
end



function get_params_vector(chain, base_name, len)
    # Robust parameter extraction for FlexiChains/MCMCChains
    N_samples = size(chain, 1)

    # Use FlexiChains.parameters to get names
    names_ch = string.(FlexiChains.parameters(chain))

    # Tier 1: Indexed names [k]
    regex = Regex("^" * base_name * "\\[(\\d+)\\]")
    matched_names = filter(n -> occursin(regex, n), names_ch)

    if !isempty(matched_names)
        sort!(matched_names, by = n -> parse(Int, match(regex, n).captures[1]))
        res_mat = zeros(Float64, N_samples, length(matched_names))

        for (idx, n) in enumerate(matched_names)
            val_obj = chain[Symbol(n)]
            raw = hasproperty(val_obj, :data) ? val_obj.data : Array(val_obj)
            # Flatten nested structures if they appear at the index level
            data_fixed = raw isa AbstractVector && eltype(raw) <: AbstractVector ? reduce(vcat, raw) : raw
            res_mat[:, idx] = vec(Float64.(collect(data_fixed)))
        end

        if size(res_mat, 2) == 1 && len > 1
            return repeat(res_mat, 1, len)
        end
        return res_mat
    end

    # Tier 2: Single entity fallback
    if base_name in names_ch
        val_obj = chain[Symbol(base_name)]
        raw_data = hasproperty(val_obj, :data) ? val_obj.data : Array(val_obj)
        if ndims(raw_data) == 3; raw_data = raw_data[:, :, 1]; end

        # Standardize to Matrix [Samples x Params]
        # Robustly handle Matrix{Vector{Float64}} or Vector{Vector{Float64}} by flattening
        if raw_data isa AbstractMatrix && eltype(raw_data) <: AbstractVector
            # Flatten each row of vectors into a single parameter row
            mat_data = reduce(hcat, [reduce(vcat, row) for row in eachrow(raw_data)])'
        elseif raw_data isa AbstractArray && eltype(raw_data) <: AbstractVector
            mat_data = reduce(hcat, vec(raw_data))'
        else
            mat_data = Matrix{Float64}(raw_data)
        end

        if size(mat_data, 2) == len
            return mat_data
        elseif size(mat_data, 1) == len
            return mat_data'
        elseif size(mat_data, 2) == 1
            return repeat(mat_data, 1, len)
        end
    end

    @warn "Parameter '$base_name' not found in chain. Returning zeros for length $len."
    return zeros(Float64, N_samples, len)
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
    A 2 x M_rff_count matrix for Wfixed.
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

    Wfixed = Matrix{Float64}(undef, 2, M_rff_count)
    for i in 1:M_rff_count
        idx = sampled_indices[i]
        Wfixed[1, i] = all_freqs_x[idx] *  2pi # Scale by  2pi to match RFF convention (often ω'x)
        Wfixed[2, i] = all_freqs_y[idx] *  2pi
    end

    return Wfixed
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
 

function prepare_fft_grid(s_coord_tuple, values; grid_res=64, pad_factor=2)
    # 1. Define the bounding box
    xs = [p[1] for p in s_coord_tuple]
    ys = [p[2] for p in s_coord_tuple]
    xmin, xmax = minimum(xs), maximum(xs)
    ymin, ymax = minimum(ys), maximum(ys)

    # 2. Map points to a grid
    # Use the length of the shorter input to prevent BoundsError
    n_limit = min(length(s_coord_tuple), length(values))
    grid = zeros(grid_res, grid_res)

    for i in 1:n_limit
        p = s_coord_tuple[i]
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

 
abstract type AbstractModelArchitecture end

struct UnivariateArchitecture <: AbstractModelArchitecture end
struct MultivariateArchitecture <: AbstractModelArchitecture end
struct MultifidelityArchitecture <: AbstractModelArchitecture end
struct ExampleArchitecture <: AbstractModelArchitecture end
struct UnknownArchitecture <: AbstractModelArchitecture end

function get_architecture(model_arch::String)
    if startswith(model_arch, "example_")
        return ExampleArchitecture() 
    elseif model_arch in ["univariate"]
        return UnivariateArchitecture()
    elseif model_arch in ["multivariate"]
        return MultivariateArchitecture()
    elseif model_arch in ["multifidelity"]
        return MultifidelityArchitecture()
    else
        return UnknownArchitecture()
    end
end

 
abstract type ModelFamily end
struct PoissonFamily <: ModelFamily end
struct GaussianFamily <: ModelFamily end
struct LogNormalFamily <: ModelFamily end
struct BinomialFamily <: ModelFamily end
struct NegativeBinomialFamily <: ModelFamily end

function get_model_family(model_family::String)
    if model_family == "poisson"
        return PoissonFamily()
    elseif model_family == "gaussian"
        return GaussianFamily()
    elseif model_family == "lognormal"
        return LogNormalFamily()
    elseif model_family in ["bernoulli", "binomial"]
        return BinomialFamily()
    elseif model_family == "negbin"
        return NegativeBinomialFamily()
    else
        error("Unknown model_family: $model_family")
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
    log_lik = zeros(N_samples, M.y_N)

    # Helper to handle parameter access across different chain types
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
            is_obs = i <= M.y_N
            mu_eta = eta[i, j]

            # --- Link Functions ---
            mu = if fam isa PoissonFamily || fam isa NegativeBinomialFamily
                clamp(exp(mu_eta), 1e-10, 1e9) 
            elseif fam isa BinomialFamily
                logistic(mu_eta)
            elseif fam isa LogNormalFamily
                mu_eta 
            else
                mu_eta
            end

            denoised[i, j] = mu

            # --- Likelihood Calculation (Training Data Only) ---
            if is_obs
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
                          elseif fam isa BinomialFamily
                               n_trials = is_obs ? M.trials[i] : 1 
                               rand(Binomial(n_trials, mu))
                          elseif fam isa NegativeBinomialFamily
                               r_val = "r_nb" in name_strs ? chain[:r_nb].data[j] : 1.0
                               rand(NegativeBinomial(r_val, r_val / (r_val + mu)))
                          else mu end
        end
    end
    return denoised, noisy, log_lik
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



function create_prediction_surface(basis_df::DataFrame, observations_df::DataFrame, au; lambda_s=2.0, lambda_t=1.0, max_iters=5)
    # 1. Initialization and Automatic Identification of Fixed-Effect Columns
    mergeon = hasproperty(basis_df, :u_idx) && hasproperty(observations_df, :u_idx) ? [:s_idx, :t_idx, :u_idx] : [:s_idx, :t_idx]

    # Identify non-merge, non-outcome columns (the design matrix variables)
    # observations_df should already contain M.y_obs if applicable, so we can exclude it.
    fixed_variable_names = setdiff(propertynames(observations_df), vcat(mergeon, [:y_obs]))

    # Join basis (grid) with observations
    surface = leftjoin(basis_df, observations_df, on = mergeon, makeunique=true)

    # Standardize types to Float64/Missing for imputation math
    # Apply to the combined 'surface' DataFrame
    for c in fixed_variable_names
        surface[!, c] = convert(Vector{Union{Float64, Missing}}, collect(surface[!, c]))
    end

    centroids = au.centroids
    n_units = length(centroids)
    W = au.W

    # 2. Precompute Spatial Adjacency Weights
    dist_mat_s = [sqrt(sum((c1 .- c2).^2)) for c1 in centroids, c2 in centroids]
    weight_mat_s = exp.(-dist_mat_s.^2 ./ (2 * lambda_s^2)) .* (W + I)
    for i in 1:n_units
        s_row = sum(weight_mat_s[i, :])
        if s_row > 1e-9; weight_mat_s[i, :] ./= s_row; end
    end

    # 3. Iterative Spatiotemporal Imputation (Handling Continuous & Binned Factors)
    for iter in 1:max_iters
        # Check if there are any missing values in the relevant columns
        has_missing = false
        for c in fixed_variable_names
            if any(ismissing, surface[!, c])
                has_missing = true
                break
            end
        end
        if !has_missing; break; end # Break if no missing values remain

        for c in fixed_variable_names
            # If the variable is likely a factor (only integers), we store levels to snap back later
            unique_vals = filter(!ismissing, unique(surface[!, c]))
            is_factor = all(v -> v == floor(v), unique_vals) # Check if all unique valid values are integers

            group_cols = hasproperty(surface, :u_idx) ? [:u_idx] : Symbol[] # Convert to Symbol[] for consistency
            # Iterate over unique combinations of grouping columns (e.g., u_idx if present)
            for grp in groupby(surface, group_cols)
                # Iterate over rows within each group
                for i in 1:nrow(grp)
                    current_row_in_grp = grp[i, :]
                    if ismissing(current_row_in_grp[c]) || isnan(current_row_in_grp[c]) # Check for isnan too
                        curr_s = Int(round(current_row_in_grp[:s_idx]))
                        curr_t = Int(round(current_row_in_grp[:t_idx])) # Ensure t_idx is also Int for comparison
                        curr_u = hasproperty(grp, :u_idx) ? Int(round(current_row_in_grp[:u_idx])) : nothing

                        # Spatial Influence: Look for values in the same time slice (and u_idx if applicable)
                        spatial_mask_df = filter(row -> row[:t_idx] == curr_t && (isnothing(curr_u) || row[:u_idx] == curr_u), surface)
                        s_vals_filtered = filter(row -> !ismissing(row[c]), eachrow(spatial_mask_df))

                        val_s, w_s_sum = 0.0, 0.0
                        for row in s_vals_filtered
                            nb_s = Int(round(row[:s_idx]))
                            w = nb_s != curr_s ? weight_mat_s[curr_s, nb_s] : 0.0 # Exclude self-influence
                            val_s += w * Float64(row[c])
                            w_s_sum += w
                        end

                        # Temporal Influence: Look for values in the same spatial unit (and u_idx if applicable)
                        temporal_mask_df = filter(row -> row[:s_idx] == curr_s && (isnothing(curr_u) || row[:u_idx] == curr_u), surface)
                        t_vals_filtered = filter(row -> !ismissing(row[c]), eachrow(temporal_mask_df))

                        val_t, w_t_sum = 0.0, 0.0
                        for row in t_vals_filtered
                            nb_t = Int(round(row[:t_idx]))
                            w = nb_t != curr_t ? exp(-(curr_t - nb_t)^2 / (2 * lambda_t^2)) : 0.0 # Exclude self-influence
                            val_t += w * Float64(row[c])
                            w_t_sum += w
                        end

                        if (w_s_sum + w_t_sum) > 1e-9 # Use a small epsilon to avoid division by zero
                            imputed_val = (val_s + val_t) / (w_s_sum + w_t_sum)
                            # Find the global index of the current row in 'surface' to update
                            # This is a potentially slow lookup, but necessary given DataFrame's groupby behavior
                            global_row_idx = findfirst(r -> r[:s_idx] == curr_s && r[:t_idx] == curr_t && (isnothing(curr_u) || r[:u_idx] == curr_u), eachrow(surface))
                            if !isnothing(global_row_idx)
                                surface[global_row_idx, c] = is_factor ? round(imputed_val) : imputed_val
                            end
                        end
                    end
                end
            end
        end
    end

    # 4. Final Global Fallback for any remaining missing values if iterative imputation couldn't fill them
    for c in fixed_variable_names
        valid_entries = filter(!ismissing, surface[!, c])
        if !isempty(valid_entries)
            m_val = median(valid_entries)
            surface[!, c] = map(x -> ismissing(x) ? m_val : x, surface[!, c])
        end
    end

    return surface
end



function convert_advi_to_reconstruct_format(msol, model::DynamicPPL.Model, n_samples::Int=500)
    # Sample from the variational distribution
    samples_vec = rand(msol, n_samples)

    # Safety check for extraction: ADVI samples often wrap data in .nt, .data, or are ParamsWithStats objects
    function get_data_obj(s)
        if hasproperty(s, :nt) return s.nt
        elseif hasproperty(s, :data) return s.data
        elseif hasproperty(s, :params) return s.params # Handle ParamsWithStats internal field
        else return s end
    end

    # Peek at the first sample to discover parameter keys
    first_samp = get_data_obj(samples_vec[1])
    all_keys = keys(first_samp)

    unique_bases = Set{Symbol}()
    for k in all_keys
        m = match(r"^([^\\\\[]+)", string(k))
        if m !== nothing
            push!(unique_bases, Symbol(m.captures[1]))
        end
    end

    reconstruct_samples = map(samples_vec) do samp
        nt = get_data_obj(samp)
        sample_params = Dict{Symbol, Any}()
        for base_sym in unique_bases
            base_str = string(base_sym)
            col_keys = filter(k -> string(k) == base_str || startswith(string(k), "$base_str["), all_keys)
            if length(col_keys) == 1 && string(first(col_keys)) == base_str
                sample_params[base_sym] = nt[first(col_keys)]
            else
                # Sort indexed keys like x[1], x[10], x[2] into numerical order
                sorted_keys = sort(collect(col_keys), by = k -> begin
                    m_idx = match(r"\\\\[(\d+)\\\\]", string(k))
                    m_idx !== nothing ? parse(Int, m_idx.captures[1]) : 0
                end)
                sample_params[base_sym] = [nt[k] for k in sorted_keys]
            end
        end
        return (; sample_params...)
    end

    # Create a FlexiChain for standard diagnostics
    formatted_dicts = map(samples_vec) do samp
        nt = get_data_obj(samp)
        Dict(FlexiChains.Parameter(Symbol(k)) => v for (k, v) in pairs(nt))
    end
    chn = FlexiChains.FlexiChain{Symbol}(n_samples, 1, formatted_dicts)

    return (chain=chn, reconstruct_samples=reconstruct_samples)
end



function convert_optim_to_reconstruct_format(optim_result, model, n_samples::Int=500; use_hessian=true, external_hessian=nothing)
    """
    Synopsis: Converts a point estimate (MAP/ML) from Optim/Optimisers into a distribution of samples.
    If a Hessian is available, it samples from the Multivariate Normal Laplace approximation.
    Otherwise, it creates a narrow Gaussian around the point estimate.
    """

    point_est_constrained = optim_result.params # This is the named tuple of constrained parameters

    reconstruct_samples_namedtuple = [] # Store NamedTuple for easier handling

    # Determine which Hessian to use
    H_to_use = nothing
    if external_hessian !== nothing
        H_to_use = external_hessian
    elseif hasproperty(optim_result, :hessian)
        H_to_use = optim_result.hessian
    end

    # Attempt Hessian-based sampling if enabled and a Hessian is available from either source
    if use_hessian && H_to_use !== nothing
        try
            # Get the unconstrained minimizer (mu_unconstrained)
            mu_unconstrained = optim_result.minimizer
            H = H_to_use

            # Ensure H is symmetric and compute its inverse for the covariance matrix
            Sigma = inv(Symmetric(Matrix(H) + Diagonal(fill(1e-6, size(H, 1)))))
            
            dist = MvNormal(mu_unconstrained, Sigma)

            # Generate n_samples of unconstrained parameters
            unconstrained_samples_matrix = rand(dist, n_samples)

            # Prepare template for conversion
            vi_template = DynamicPPL.VarInfo(model)

            for i in 1:n_samples
                sample_unconstrained_vec = unconstrained_samples_matrix[:, i]
                vi_current_sample = deepcopy(vi_template)
                DynamicPPL.setlink!(vi_current_sample, sample_unconstrained_vec)
                
                # Convert back to constrained named tuple
                constrained_sample_params = DynamicPPL.vi_to_params(vi_current_sample, model)
                push!(reconstruct_samples_namedtuple, constrained_sample_params)
            end

        catch e
            @warn "Failed to compute covariance from Hessian or convert samples, falling back to adding noise: $e"
            use_hessian = false 
        end
    end

    # Fallback to noise-based sampling
    if !use_hessian || isempty(reconstruct_samples_namedtuple)
        all_keys = keys(point_est_constrained)
        unique_bases = Set{Symbol}()
        for k in all_keys
            m = match(r"^([^\\[]+)", string(k))
            if m !== nothing
                push!(unique_bases, Symbol(m.captures[1]))
            end
        end

        reconstruct_samples_namedtuple = map(1:n_samples) do _
            sample_params_dict = Dict{Symbol, Any}() 
            for base_sym in unique_bases
                base_str = string(base_sym)
                col_keys = filter(k -> string(k) == base_str || startswith(string(k), "$base_str["), all_keys)

                if length(col_keys) == 1 && string(first(col_keys)) == base_str
                    val = point_est_constrained[first(col_keys)]
                    if val isa AbstractVector
                        sample_params_dict[base_sym] = val .+ randn() * 1e-4
                    else
                        sample_params_dict[base_sym] = val + randn() * 1e-4
                    end
                else
                    sorted_keys = sort(collect(col_keys), by = k -> begin
                        m_idx = match(r"\\[(\\d+)\\]", string(k))
                        m_idx !== nothing ? parse(Int, m_idx.captures[1]) : 0
                    end)
                    sample_params_dict[base_sym] = [point_est_constrained[k] + randn() * 1e-4 for k in sorted_keys]
                end
            end
            return (; sample_params_dict...)
        end
    end

    # 4. Format into a FlexiChain
    # IMPORTANT: Store ONLY the base symbols (e.g., :s_icar as a vector).
    # FlexiChains will automatically expand these into indexed names for summary statistics,
    # avoiding duplicate key errors while allowing _reconstruct to find the full vector.
    formatted_dicts = map(1:n_samples) do i
        samp = reconstruct_samples_namedtuple[i]
        d = Dict{FlexiChains.Parameter, Any}()
        for k in keys(samp)
            d[FlexiChains.Parameter(k)] = samp[k]
        end
        return d
    end

    chn = FlexiChains.FlexiChain{Symbol}(n_samples, 1, formatted_dicts)

    return (chain=chn, reconstruct_samples=reconstruct_samples_namedtuple)
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




function generate_sim_data(s_N=10, t_N=5; rndseed=42)
    Random.seed!(rndseed)
    n_total = s_N * t_N
    unique_pts = [(rand() * 100, rand() * 100) for _ in 1:s_N]
    s_coord_tuple = repeat(unique_pts, t_N)
    s_coord = vcat([collect(t)' for t in s_coord_tuple]...)
    t_coord = rand(n_total) .* (t_N+1) .+ 2000
    weights = ones(n_total)
    trials  = ones(Int, n_total)
    u_N = 12
    period=u_N
    trend = 0.05 .* t_coord[:,1]
    seasonal = 1.0 .* cos.(2pi .* t_coord[:,1] ./ period)
    temporal_effect =  1.0 .* ( trend .+ seasonal )
    spatial_effect = 1.5 .* sin.(s_coord[:,1] .*  2pi) .* cos.(s_coord[:,2] .*  2pi)
    sigma_y = 0.2
    observation_error = sigma_y .* randn(n_total)
    Z = randn(n_total)
    W1_true = 0.5 .* sin.(t_coord[:,1] ./ 5.0) .+ 0.5 .* Z
    W2_true = 0.5 .* cos.(t_coord[:,1] ./ 5.0) .- 0.3 .* Z
    W3_true = 0.2 .* (t_coord[:,1] ./ n_total) .+ 0.1 .* Z
    sigma_w1, sigma_w2, sigma_w3 = 0.1, 0.2, 0.3
    W1_obs = W1_true .+ randn(n_total) .* sigma_w1
    W2_obs = W2_true .+ randn(n_total) .* sigma_w2
    W3_obs = W3_true .+ randn(n_total) .* sigma_w3
    cov_continuous = hcat(W1_obs, W2_obs, W3_obs)

    y_obs = 1.0 .+  spatial_effect + temporal_effect .+  observation_error .+ W1_obs .+ W2_obs .+ W3_obs
    y_binary = y_obs .> (mean(y_obs) + 0.5)
    y_counts = abs.(Int.(round.(y_obs))) * 100
    
    # fixed (covariate) ..  
    Xfixed = [ones(n_total) ]   # column of ones is the intercept
     
    return (
        s_coord_tuple=s_coord_tuple, s_coord=s_coord, t_coord=t_coord, weights=weights, trials=trials, 
        s_N=s_N, t_N=t_N, u_N=u_N,
        y_obs=y_obs, y_binary=y_binary, y_counts=y_counts, Xfixed=Xfixed,
        z_obs=Z, w_obs=cov_continuous
    )
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

function householder_to_eigenvector(v_mat::AbstractMatrix{T}, nU, n_factors) where {T}
    # Initializes the Identity matrix to be transformed
    U = Matrix{T}(I, nU, nU)

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

function householder_transform(v, nU, n_factors, ltri_indices, pca_sd, pdef_sd, noise)
    T = eltype(v)
    v_mat = zeros(T, nU, n_factors)
    v_mat[ltri_indices] .= v

    # Generate Orthonormal Loadings using optimized transformation
    U = householder_to_eigenvector(v_mat, nU, n_factors)

    # Reconstruct Covariance Components
    # W = Loadings * Scaled Eigenvalues
    W = U * Diagonal(pca_sd)

    # Kmat is the full covariance matrix: WW' + Residual_Variance
    # We add a small noise term for numerical stability
    Kmat = W * W' + (pdef_sd^2 + noise) * I(nU)

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

    nU = size(U_in, 1)
    # We work on a copy to avoid modifying the input
    U = copy(U_in)
    
    # Storage for the lower-triangular part of the v_mat
    # Each column k corresponds to the k-th Householder vector
    v_mat = zeros(T, nU, n_factors)

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

function extract_v_parameters(v_mat, ltri_indices)
    # extract_v_parameters(v_mat, ltri_indices)
    #
    # Utility to extract only the free parameters (lower triangular) from the v_mat
    # for use as initial values in Turing (matching the 'v' parameter vector).
    return v_mat[ltri_indices]
end 

function get_params_matrix_sizestructured(chain, base_name, dims)
    # Optimized for chn[:param].data[sample][row, col] access pattern
    n_rows, n_cols = dims
    N_samples = size(chain, 1)

    if Symbol(base_name) in names(chain, :parameters)
        res = zeros(Float64, n_rows, n_cols, N_samples)
        data_container = chain[Symbol(base_name)].data

        for s in 1:N_samples
            samp_mat = data_container[s]
            if size(samp_mat) == (n_rows, n_cols)
                res[:, :, s] = samp_mat
            elseif size(samp_mat) == (n_cols, n_rows)
                res[:, :, s] = samp_mat'
            end
        end
        return res
    end
    return nothing
end



 
function summarize_array(samples::AbstractArray; alpha=0.05)
    if isempty(samples) || all(isnan, samples)
        return (mean = [NaN], median = [NaN], lower = [NaN], upper = [NaN])
    end

    dims = size(samples)
    n_dims = length(dims)

    # Calculate statistics across the sample dimension
    post_mean = dropdims(mean(samples, dims=n_dims), dims=n_dims)
    post_median = dropdims(median(samples, dims=n_dims), dims=n_dims)
    low_bound = dropdims(mapslices(x -> quantile(x, alpha/2), samples, dims=n_dims), dims=n_dims)
    high_bound = dropdims(mapslices(x -> quantile(x, 1 - alpha/2), samples, dims=n_dims), dims=n_dims)

    # FIX: Ensure outputs are ALWAYS Vectors, even if they contain only one element.
    # This prevents MethodError: vec(::Float64) in downstream code.
    force_vector(x) = begin
        if x isa AbstractArray
            if ndims(x) == 0 # It's a 0-dimensional array (e.g., Array{Float64, 0})
                return [Float64(x[])] # Extract scalar and wrap in vector
            else
                return vec(collect(x)) # For higher-dimensional arrays, flatten to vector
            end
        else # It's already a scalar (e.g., Float64)
            return [Float64(x)] # Wrap scalar in vector
        end
    end

    return (
        mean = force_vector(post_mean),
        median = force_vector(post_median),
        lower = force_vector(low_bound),
        upper = force_vector(high_bound)
    )
end


 
function model_results_comprehensive(model, result, M=nothing, areal_units=nothing; PS="quick_approximation", alpha=0.05, n_samples=500, kwargs...)
    M = isnothing(M) ? model.args.M : M
    actual_areal_units_data = isnothing(areal_units) ? M.areal_units : areal_units

    actual_PS = if PS == "quick_approximation"
        s_N, t_N = M.s_N, M.t_N
        u_N = hasproperty(M, :u_N) ? M.u_N : 1

        # Create basis_df (DataFrame) with s_idx varying fastest
        basis_df_cols = hasproperty(M, :u_idx) ? [:s_idx, :t_idx, :u_idx] : [:s_idx, :t_idx]
        basis_data = Vector{Vector{Float64}}()
        if hasproperty(M, :u_idx)
            for u in 1:u_N
                for t in 1:t_N
                    for s in 1:s_N
                        push!(basis_data, [Float64(s), Float64(t), Float64(u)])
                    end
                end
            end
        else
            for t in 1:t_N
                for s in 1:s_N
                    push!(basis_data, [Float64(s), Float64(t)])
                end
            end
        end
        basis_df = DataFrame(reduce(hcat, basis_data)', basis_df_cols)

        # Create observations_df (DataFrame) with explicit conversions to avoid NamedArray issues
        obs_df_cols = Symbol[:s_idx, :t_idx]
        obs_raw = hcat(Float64.(M.s_idx), Float64.(M.t_idx))
        if hasproperty(M, :u_idx) && !isnothing(M.u_idx)
            obs_raw = hcat(obs_raw, Float64.(M.u_idx))
            push!(obs_df_cols, :u_idx)
        end
        observations_df = DataFrame(obs_raw, obs_df_cols)

        # Explicitly convert M.Xfixed to a plain Matrix and get Symbol names
        xfixed_matrix = Matrix{Float64}(M.Xfixed)
        xfixed_colnames = Symbol.(names(M.Xfixed, 2))

        # Add Xfixed columns to observations_df, ensuring unique names
        for (col_idx, col_name) in enumerate(xfixed_colnames)
            # Ensure uniqueness for column names in observations_df
            final_col_name = col_name
            k = 1
            while final_col_name in names(observations_df) && !(final_col_name in obs_df_cols)
                final_col_name = Symbol(string(col_name, "_", k))
                k += 1
            end
            observations_df[!, final_col_name] = xfixed_matrix[:, col_idx]
        end

        # Add y_obs to observations_df if it's univariate, so create_prediction_surface knows to ignore it for fixed_variable_names
        if !isa(M.y_obs, AbstractMatrix) # Check if it's not multivariate
            observations_df[!, :y_obs] = M.y_obs
        end
 
        full_surface = create_prediction_surface(basis_df, observations_df, actual_areal_units_data;
            lambda_s = Float64(get(kwargs, :lambda_s, 2.0)),
            lambda_t = Float64(get(kwargs, :lambda_t, 1.0)))

        (s_idx = Int.(full_surface[!, :s_idx]),
         t_idx = Int.(full_surface[!, :t_idx]),
         u_idx = hasproperty(full_surface, :u_idx) ? Int.(full_surface[!, :u_idx]) : nothing,
         surface_df = full_surface)
    else
        PS
    end


    # 2. Extract Chain
    chain = if result isa Turing.Variational.VIResult
        convert_advi_to_reconstruct_format(result, model, n_samples).chain
    elseif result isa Turing.Optimisation.ModeResult
        convert_optim_to_reconstruct_format(result, model, n_samples; use_hessian=true).chain
    else
        total_s = size(result, 1)
        (total_s > n_samples) ? result[iter = (total_s - n_samples + 1):total_s] : result
    end
 

    # 3. Trait-based Reconstruction
    arch = get_architecture(M.model_arch)
    fam = get_model_family(M.model_family)

    modelname = string(model.f)

    pstats = _reconstruct(arch, fam, modelname, chain, M, actual_PS, alpha)

    # 4. Metric Calculations
    y_obs = Float64.(collect(skipmissing(M.y_obs)))
    y_pred = pstats.predictions_observed_denoised.mean
    rmse = sqrt(mean((y_obs .- y_pred).^2))
    r2 = length(unique(y_pred)) > 1 ? cor(y_obs, y_pred)^2 : 0.0
    waic_val = hasproperty(pstats, :waic) ? first(pstats.waic) : 0.0
    ss_df = DataFrame(MCMCChains.summarize(MCMCChains.Chains(chain)))
    mean_rhat = mean(filter(!isnan, ss_df[!, :rhat]))
    mean_ess = mean(filter(!isnan, ss_df[!, :ess_bulk]))

    showtuples((rmse=rmse, r2=r2, waic=waic_val, rhat=mean_rhat, ess=mean_ess))

    # 5. Visualizations
    p_ppc = scatter(y_pred, y_obs, title="PPC (Denoised)", xlabel="Predicted", ylabel="Observed", alpha=0.5, legend=false)
    p_tm = nothing
    if hasproperty(pstats, :temporal) && !all(isnan(pstats.temporal.mean))
        m_vals = pstats.temporal.mean
        p_tm = plot(m_vals, title="Temporal Effect", lw=2, color=:blue, legend=false)
    end
    p_sp = nothing
    try
        p_sp = plot_posterior_results(pstats, M, actual_areal_units_data; effect=:spatial_structured)
    catch; end
    p_fe = nothing
    if hasproperty(pstats, :fixed_effects) && !isnothing(pstats.fixed_effects)
        fe = pstats.fixed_effects
        p_fe = bar(fe.mean, yerror=(fe.mean .- fe.lower, fe.upper .- fe.mean), title="Fixed Effects", xlabel="Index", ylabel="Value", color=:grey, legend=false)
    end
    p_re = nothing
    if hasproperty(pstats, :random_effects) && !isnothing(pstats.random_effects)
        re = pstats.random_effects
        p_re = boxplot(ones(length(re.mean)), re.mean, title="Random Effects Dist.", ylabel="Effect Size", color=:purple, legend=false)
    end

    # New: Mixed Effects Plots
    mixed_plots = []
    if hasproperty(pstats, :mixed_effects) && !isnothing(pstats.mixed_effects)
        for (idx, me_stats) in enumerate(pstats.mixed_effects)
            if !isnothing(me_stats)
                p_me = bar(me_stats.mean, yerror=(me_stats.mean .- me_stats.lower, me_stats.upper .- me_stats.mean),
                           title="Mixed Effect $(idx)", xlabel="Category", ylabel="Effect Size", legend=false)
                push!(mixed_plots, p_me)
            end
        end
    end

    # New: Interaction Effects Plots
    interaction_plots = []
    if hasproperty(pstats, :interaction_effects) && !isnothing(pstats.interaction_effects)
        for (idx, ie_stats) in enumerate(pstats.interaction_effects)
            if !isnothing(ie_stats)
                p_ie = plot(ie_stats.mean, ribbon=(ie_stats.mean .- ie_stats.lower, ie_stats.upper .- ie_stats.mean),
                            title="Interaction Effect $(idx)", xlabel="Index", ylabel="Effect Size", legend=false)
                push!(interaction_plots, p_ie)
            end
        end
    end

    # New: Eigen Effects Plots (e.g., PCA loadings or latent fields)
    eigen_plots = []
    if hasproperty(pstats, :eigen_effects) && !isnothing(pstats.eigen_effects)
        for (idx, ee_stats) in enumerate(pstats.eigen_effects)
            if !isnothing(ee_stats)
                p_ee = plot(ee_stats.mean, ribbon=(ee_stats.mean .- ee_stats.lower, ee_stats.upper .- ee_stats.mean),
                            title="Eigen Effect $(idx)", xlabel="Dimension", ylabel="Effect Size", legend=false)
                push!(eigen_plots, p_ee)
            end
        end
    end


 
    plot_list = filter(!isnothing, [p_ppc, p_tm, p_sp, p_fe, p_re, mixed_plots..., interaction_plots..., eigen_plots...])
    n_plots = length(plot_list)
    layout = n_plots <= 4 ? (2, 2) : (ceil(Int, n_plots/2), 2)
    display(plot(plot_list..., layout=layout, size=(1000, 400 * ceil(Int, n_plots/2)))) # Adjust size dynamically
    return (metrics=(rmse=rmse, r2=r2, waic=waic_val, rhat=mean_rhat, ess=mean_ess), pstats=pstats )
end



function get_vec(obj, key)  
    # Helper for robust indexing
    val = hasproperty(obj, key) ? getproperty(obj, key) : nothing
    val isa AbstractVector ? val : [val]
end

 

function plot_binned_covariates(res)
    # Check if covariate effects exist in pstats
    cov_effects = get(res.pstats, :covariate_effects, Dict())
    if isempty(cov_effects)
        println("No covariate effects found to plot.")
        return nothing
    end

    plts = []
    for (key, samples) in cov_effects
        # Key format: :raw_rw2_cwd
        name = replace(string(key), "raw_rw2_" => "")
        m = vec(mean(samples, dims=2))
        l = vec(quantile.(eachrow(samples), 0.025))
        u = vec(quantile.(eachrow(samples), 0.975))
        
        p = plot(m, ribbon=(m .- l, u .- m), title="Effect: $name", 
                 xlabel="Bin", ylabel="Value", legend=false, fillalpha=0.2)
        push!(plts, p)
    end
    
    n = length(plts)
    if n > 0
        display(plot(plts..., layout=(ceil(Int, n/2), 2), size=(900, 300*ceil(Int, n/2))))
    end
end

function plot_seasonal_cycle(res)
    if !hasproperty(res.pstats, :seasonal) || all(isnan, res.pstats.seasonal.mean)
        println("No seasonal component found to plot.")
        return nothing
    end
    
    m = res.pstats.seasonal.mean
    s = res.pstats.seasonal.samples
    l = vec(quantile.(eachrow(s), 0.025))
    u = vec(quantile.(eachrow(s), 0.975))
    
    plt = plot(m, ribbon=(m .- l, u .- m), title="Seasonal Cycle (Binned/Harmonic)", 
               xlabel="Season Bin", ylabel="Effect", lw=2, fillalpha=0.3, legend=false)
    display(plt)
    return plt
end

 
function dataframe_to_named_array(df::DataFrame)
    # Converts a DataFrame to a NamedArray for internal model processing
    mat = Matrix(df)
    return NamedArray(mat, (1:size(mat, 1), Symbol.(names(df))))
end

##############


function _reconstruct(arch::UnivariateArchitecture, fam::ModelFamily, modelname, chain, M, PS, alpha)
    # --- 1. Dimensions and Metadata Discovery ---
    # Determine the number of post-stratification units and total target size
    N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    N_tot = M.y_N + N_PS
    N_samples = size(chain, 1)
    
    # Map parameter names to strings for feature detection
    p_names_str = string.(FlexiChains.parameters(chain))

    # --- 2. Global Hyperparameter Extraction ---
    # Extract scaling factors (sigmas) for each manifold; default to 1.0 or 0.0 if not present
    s_sigmas = "s_sigma" in p_names_str ? vec(get_params_vector(chain, "s_sigma", 1)) : ones(N_samples)
    t_sigmas = "t_sigma" in p_names_str ? vec(get_params_vector(chain, "t_sigma", 1)) : ones(N_samples)
    u_sigmas = "u_sigma" in p_names_str ? vec(get_params_vector(chain, "u_sigma", 1)) : zeros(N_samples)
    st_sigmas = "st_sigma" in p_names_str ? vec(get_params_vector(chain, "st_sigma", 1)) : zeros(N_samples)

    # --- 3. Field & Effect Storage Initialization ---
    s_eff_struct = zeros(M.s_N, N_samples)
    s_eff_noisy = zeros(M.s_N, N_samples)
    t_eff = zeros(M.t_N, N_samples)
    u_eff = zeros(get(M, :u_N, 1), N_samples)
    st_eff_map = zeros(M.s_N, M.t_N, N_samples)

    # Hierarchical Spatial storage (Region -> County -> Unit)
    hier_fields = Dict{Symbol, Matrix{Float64}}()
    if haskey(M, :spatial_hierarchy)
        for scale_sym in keys(M.spatial_hierarchy)
            hier_fields[scale_sym] = zeros(M.spatial_hierarchy[scale_sym].n_units, N_samples)
        end
    end

    # Regression base storage
    Xfixed_betas = (M.Xfixed_N > 0) ? zeros(M.Xfixed_N, N_samples) : nothing

    # Categorical/Group effect storage (ce / re)
    categorical_eff = Dict{Symbol, Matrix{Float64}}()
    if haskey(M, :c_groups)
        for c_sym in keys(M.c_groups)
            n_lvls = size(M.c_re_templates[c_sym].matrix, 1)
            categorical_eff[c_sym] = zeros(n_lvls, N_samples)
        end
    end

    # Advanced structure storage (Mixed, Interaction, Eigen, SVC)
    mixed_eff = !isempty(get(M, :mixed_terms, [])) ? [zeros(term.n_cat, N_samples) for term in M.mixed_terms] : nothing
    interaction_eff = !isempty(get(M, :interaction_terms, [])) ? [zeros(N_tot, N_samples) for _ in M.interaction_terms] : nothing
    eigen_eff = !isempty(get(M, :eigen_terms, [])) ? [zeros(N_tot, N_samples) for _ in M.eigen_terms] : nothing
    svc_eff = !isempty(get(M, :svc_covariates, [])) ? [zeros(N_tot, N_samples) for _ in M.svc_covariates] : nothing
    basis_eff = !isempty(get(M, :basis_matrices, Dict())) ? Dict(v => zeros(N_tot, N_samples) for v in keys(M.basis_matrices)) : nothing

    # --- 4. Posterior Parameter Reconstruction Loop ---
    for j in 1:N_samples
        # 4.1 Core Manifolds via Trait-based dispatch
        fs, fn = extract_manifold(SpatialTrait(), chain, M, j, s_sigmas[j])
        s_eff_struct[:, j] .= fs
        s_eff_noisy[:, j] .= fn
        t_eff[:, j] .= extract_manifold(TemporalTrait(), chain, M, j, t_sigmas[j])
        if M.model_season != "none"
            u_eff[:, j] .= extract_manifold(SeasonalTrait(), chain, M, j, u_sigmas[j])
        end

        # 4.2 Multi-resolution hierarchical scales
        for scale_sym in keys(hier_fields)
            p_key = Symbol("s_latent_", scale_sym)
            if p_key in FlexiChains.parameters(chain)
                hier_fields[scale_sym][:, j] .= vec(chain[p_key].data[j])
            end
        end

        # 4.3 Spatiotemporal map (Transport or KH)
        if "st_latent" in p_names_str
            st_eff_map[:, :, j] .= reshape(vec(chain[:st_latent].data[j]), M.s_N, M.t_N) .* st_sigmas[j]
        end

        # 4.4 Fixed and Categorical Effects
        if !isnothing(Xfixed_betas); Xfixed_betas[:, j] .= extract_manifold(FixedEffectTrait(), chain, M, j); end
        for c_sym in keys(categorical_eff)
            p_name = Symbol("c_latent_$(c_sym)")
            if p_name in FlexiChains.parameters(chain); categorical_eff[c_sym][:, j] .= vec(chain[p_name].data[j]); end
        end

        # 4.5 Advanced Basis Components (Splines/FFT)
        if !isnothing(basis_eff)
            for (v_sym, B) in M.basis_matrices
                beta_b = get_params_vector(chain, Symbol("beta_basis_", v_sym), size(B, 2))[j, :]
                basis_eff[v_sym][1:M.y_N, j] .= B * beta_b
            end
        end

        # 4.6 Mixed Effects (Varying Slopes)
        if !isnothing(mixed_eff)
            for (idx, m_term) in enumerate(M.mixed_terms)
                mixed_eff[idx][:, j] .= get_params_vector(chain, Symbol("beta_group_$(idx)"), m_term.n_cat)[j, :]
            end
        end

        # 4.7 Interaction Surfaces (RFF Projections)
        if !isnothing(interaction_eff)
            for (idx, i_term) in enumerate(M.interaction_terms)
                beta_ie = get_params_vector(chain, Symbol("beta_ie_$(idx)"), i_term.M_rff)[j, :]
                # Incorporate post-stratification coordinates for full surface prediction
                all_coords = (N_PS > 0) ? vcat(i_term.coords, hcat(PS.surface_df.lat, PS.surface_df.lon)) : i_term.coords
                proj = (all_coords * i_term.W_ie) .+ i_term.b_ie'
                interaction_eff[idx][:, j] .= sqrt(2.0/i_term.M_rff) .* cos.(proj) * beta_ie
            end
        end

        # 4.8 Spatially Varying Coefficients (SVC via RFF basis caching)
        if !isnothing(svc_eff) && get(M, :svc_model, "none") == "rff"
            for (idx, c_sym) in enumerate(M.svc_covariates)
                sig_svc = get_params_vector(chain, Symbol("sig_svc_$(c_sym)"), 1)[j]
                beta_svc = get_params_vector(chain, Symbol("beta_svc_$(c_sym)"), M.svc_M_rff)[j, :]
                z_k = haskey(M, :svc_basis_cached) ? M.svc_basis_cached : (sqrt(2.0 / M.svc_M_rff) .* cos.((M.s_coord * M.W_fixed_svc) .+ M.b_fixed_svc'))
                beta_s = z_k[M.s_idx, :] * beta_svc
                col_idx = findfirst(==(c_sym), Symbol.(names(M.Xfixed, 2)))
                if !isnothing(col_idx); svc_eff[idx][1:M.y_N, j] .= (beta_s .* sig_svc) .* M.Xfixed[:, col_idx]; end
            end
        end

        # 4.9 Eigen-Effects (PCA Reconstruction via Householder)
        if !isnothing(eigen_eff)
            for (idx, e_term) in enumerate(M.eigen_terms)
                v_pca = get_params_vector(chain, Symbol("v_pca_$(idx)"), length(e_term.ltri_indices))[j, :]
                p_sd = get_params_vector(chain, Symbol("pca_sd_$(idx)"), e_term.n_dims)[j, :]
                pdef = get_params_vector(chain, Symbol("pdef_sd_$(idx)"), 1)[j]
                K_pca, _, _ = householder_transform(v_pca, e_term.n_dims, e_term.n_dims, e_term.ltri_indices, p_sd, pdef, M.noise)
                eigen_eff[idx][1:M.y_N, j] .= vec(e_term.data * K_pca)
            end
        end
    end

    # --- 5. Linear Predictor Assembly (Eta) ---
    eta_noisy = zeros(N_tot, N_samples)
    eta_denoised = zeros(N_tot, N_samples)
    fixed_names = Symbol.(names(M.Xfixed, 2))

    for j in 1:N_samples, i in 1:N_tot
        is_obs = i <= M.y_N; idx = is_obs ? i : i - M.y_N; src = is_obs ? M : PS
        s_id = Int(src.s_idx[idx]); t_id = Int(src.t_idx[idx])
        val = is_obs ? get(M, :log_offset, zeros(M.y_N))[i] : 0.0

        # Regression base assembly
        if !isnothing(Xfixed_betas); val += dot(Xfixed_betas[j, :], is_obs ? M.Xfixed[idx, :] : collect(Vector(PS.surface_df[idx, fixed_names]))); end
        for c_sym in keys(categorical_eff); val += categorical_eff[c_sym][src.c_groups[c_sym][idx], j]; end
        for scale_sym in keys(hier_fields); val += hier_fields[scale_sym][M.spatial_hierarchy[scale_sym].indices[idx], j]; end

        # Manifold main effects
        val += t_eff[t_id, j]
        if M.model_season != "none"; val += u_eff[Int(src.u_idx[idx]), j]; end
        
        # Sum advanced structural components
        if !isnothing(mixed_eff); for (m_idx, m_term) in enumerate(M.mixed_terms); val += mixed_eff[m_idx][m_term.indices[idx], j] * m_term.covariate_vals[idx]; end; end
        if !isnothing(interaction_eff); for ie in interaction_eff; val += ie[i, j]; end; end
        if !isnothing(eigen_eff); for ee in eigen_eff; val += ee[i, j]; end; end
        if !isnothing(svc_eff); for se in svc_eff; val += se[i, j]; end; end
        if !isnothing(basis_eff); for be in values(basis_eff); val += be[i, j]; end; end

        st_val = st_eff_map[s_id, t_id, j]
        eta_denoised[i, j] = val + s_eff_struct[s_id, j] + st_val
        eta_noisy[i, j] = val + s_eff_noisy[s_id, j] + st_val
    end

    # --- 6. Post-Processing & Strata Weighting ---
    y_sigma = _extract_volatility(chain, p_names_str, N_tot, N_samples)
    preds_denoised, preds_noisy, log_lik = _process_ll_and_predictions(fam, eta_noisy, chain, M, N_tot, N_samples, y_sigma, M.y_obs)
    field_denoised, _, _ = _process_ll_and_predictions(fam, eta_denoised, chain, M, N_tot, N_samples, y_sigma, M.y_obs)

    # Post-stratification weights (strata-to-unit ratios)
    weights = nothing
    if N_PS > 0
        stratum_map = [(Int(get(M, :u_idx, ones(M.y_N))[i])-1)*M.s_N*M.t_N + (Int(M.t_idx[i])-1)*M.s_N + Int(M.s_idx[i]) for i in 1:M.y_N]
        weights = field_denoised[M.y_N .+ stratum_map, :] ./ (field_denoised[1:M.y_N, :] .+ 1e-9)
    end

    return (
        spatial_structured = summarize_array(reshape(s_eff_struct, M.s_N, 1, N_samples); alpha=alpha),
        spatial_unstructured = summarize_array(reshape(s_eff_noisy .- s_eff_struct, M.s_N, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(t_eff, M.t_N, 1, N_samples); alpha=alpha),
        seasonal = (M.model_season != "none") ? summarize_array(reshape(u_eff, get(M, :u_N, 1), 1, N_samples); alpha=alpha) : nothing,
        hierarchical_spatial = isempty(hier_fields) ? nothing : Dict(k => summarize_array(reshape(v, size(v,1), 1, N_samples); alpha=alpha) for (k,v) in hier_fields),
        spatiotemporal = summarize_array(reshape(st_eff_map, M.s_N * M.t_N, 1, N_samples); alpha=alpha),
        volatility = summarize_array(reshape(y_sigma, N_tot, 1, N_samples); alpha=alpha),
        fixed_effects = !isnothing(Xfixed_betas) ? summarize_array(reshape(Xfixed_betas, M.Xfixed_N, 1, N_samples); alpha=alpha) : nothing,
        mixed_effects = !isnothing(mixed_eff) ? [summarize_array(reshape(me, size(me,1), 1, N_samples); alpha=alpha) for me in mixed_eff] : nothing,
        interaction_effects = !isnothing(interaction_eff) ? [summarize_array(reshape(ie, size(ie,1), 1, N_samples); alpha=alpha) for ie in interaction_eff] : nothing,
        eigen_effects = !isnothing(eigen_eff) ? [summarize_array(reshape(ee, size(ee,1), 1, N_samples); alpha=alpha) for ee in eigen_eff] : nothing,
        predictions_observed_denoised = summarize_array(reshape(preds_denoised[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        post_strat_weights = isnothing(weights) ? nothing : (mean=vec(mean(weights, dims=2)), samples=weights),
        waic = _compute_waic(log_lik[:, 1:M.y_N]),
        family = fam, arch = arch
    )
end


function _reconstruct(arch::MultivariateArchitecture, fam::ModelFamily, modelname,  chain, M, PS, alpha)
    N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    N_tot = M.y_N + N_PS
    N_samples = size(chain, 1)
    p_names_str = string.(FlexiChains.parameters(chain))

    outcome_results = Vector{Any}(undef, M.outcomes_N)
    phi_zi = "phi_zi" in p_names_str ? summarize_array(reshape(get_params_vector(chain, "phi_zi", 1), 1, 1, N_samples); alpha=alpha) : nothing

    # Identify fixed-effects column names from the observation model
    fixed_names = names(M.Xfixed, 2)
    # fixed_names = M.Xfixed isa NamedArray ? names(M.Xfixed, 2) : [Symbol("x$i") for i in 1:size(M.Xfixed, 2)]

    for k in 1:M.outcomes_N
        # 1. Parameter Extraction for outcome k
        s_sigmas = vec(get_params_vector(chain, Symbol("s_sigma_arr[", k, "]"), 1))
        t_sigmas = vec(get_params_vector(chain, Symbol("t_sigma_arr[", k, "]"), 1))
        u_sigmas = Symbol("u_sigma_arr[", k, "]") in FlexiChains.parameters(chain) ? vec(get_params_vector(chain, Symbol("u_sigma_arr[", k, "]"), 1)) : zeros(N_samples)

        s_eff_struct = zeros(M.s_N, N_samples)
        s_eff_noisy = zeros(M.s_N, N_samples)
        t_eff = zeros(M.t_N, N_samples)
        u_eff = zeros(get(M, :u_N, 1), N_samples)

        # Features
        Xfixed_betas = (M.Xfixed_N > 0) ? zeros(M.Xfixed_N, N_samples) : nothing
        random_eff = Symbol("c_eta_", k) in FlexiChains.parameters(chain) ? zeros(M.N_cat, N_samples) : nothing

        for j in 1:N_samples
            fs, fn = extract_manifold_k(SpatialTrait(), chain, M, j, k, s_sigmas[j])
            s_eff_struct[:, j] .= fs
            s_eff_noisy[:, j] .= fn

            t_key = Symbol("t_latent_", k)
            if t_key in FlexiChains.parameters(chain)
                t_eff[:, j] .= vec(chain[t_key].data[j])
            end

            if M.model_season != "none"
                u_key = Symbol("u_latent_", k)
                if u_key in FlexiChains.parameters(chain)
                    u_eff[:, j] .= vec(chain[u_key].data[j])
                end
            end

            if !isnothing(Xfixed_betas)
                Xfixed_betas[:, j] .= get_params_vector(chain, "Xfixed_beta", M.Xfixed_N * M.outcomes_N)[j, (k-1)*M.Xfixed_N+1 : k*M.Xfixed_N]
            end

            if !isnothing(random_eff)
                random_eff[:, j] .= vec(chain[Symbol("c_eta_", k)].data[j])
            end
        end

        # 2. Linear Predictor Construction
        eta_noisy = zeros(N_tot, N_samples)
        eta_denoised = zeros(N_tot, N_samples)

        for j in 1:N_samples, i in 1:N_tot
            is_obs = i <= M.y_N; idx = is_obs ? i : i - M.y_N; src = is_obs ? M : PS
            s_idx = Int(src.s_idx[idx]); t_idx = Int(src.t_idx[idx])

            base = (is_obs ? M.log_offset[i, k] : 0.0) + t_eff[t_idx, j]
            if !isnothing(Xfixed_betas)
                X_row = is_obs ? M.Xfixed[idx, :] : Matrix(PS.surface_df[idx, fixed_names])
                base += dot(Xfixed_betas[:, j], X_row)
            end
            if !isnothing(random_eff)
                # Assuming M.cov_indices[idx, 1] is the correct category for random effects
                c_level = Int(is_obs ? M.c_groups[first(keys(M.c_groups))][idx] : PS.s_idx[idx]) # Placeholder, adjust as needed
                base += random_eff[c_level, j]
            end

            eta_denoised[i, j] = base + s_eff_struct[s_idx, j]
            eta_noisy[i, j] = base + s_eff_noisy[s_idx, j]

            if M.model_season != "none"
                eta_denoised[i, j] += u_eff[Int(src.u_idx[idx]), j]
                eta_noisy[i, j] += u_eff[Int(src.u_idx[idx]), j]
            end
        end

        y_sigma_k = _extract_volatility(chain, p_names_str, N_tot, N_samples, k)
        preds_denoised, preds_noisy, log_lik = _process_ll_and_predictions(fam, eta_noisy, chain, M, N_tot, N_samples, y_sigma_k, M.y_obs[:, k])
        field_denoised, _, _ = _process_ll_and_predictions(fam, eta_denoised, chain, M, N_tot, N_samples, y_sigma_k, M.y_obs[:, k])

        weights = nothing
        if N_PS > 0
            s_idx_obs = M.s_idx; t_idx_obs = M.t_idx; s_N = M.s_N; t_N = M.t_N
            stratum_map = [(Int(t_idx_obs[i])-1)*s_N + Int(s_idx_obs[i]) for i in 1:M.y_N]
            weights = field_denoised[M.y_N .+ stratum_map, :] ./ (field_denoised[1:M.y_N, :] .+ 1e-9)
        end

        outcome_results[k] = (
            spatial_structured = summarize_array(reshape(s_eff_struct, M.s_N, 1, N_samples); alpha=alpha),
            spatial_unstructured = summarize_array(reshape(s_eff_noisy .- s_eff_struct, M.s_N, 1, N_samples); alpha=alpha),
            temporal = summarize_array(reshape(t_eff, M.t_N, 1, N_samples); alpha=alpha),
            seasonal = M.model_season != "none" ? summarize_array(reshape(u_eff, get(M, :u_N, 1), 1, N_samples); alpha=alpha) : nothing,
            volatility = summarize_array(reshape(y_sigma_k, N_tot, 1, N_samples); alpha=alpha),
            fixed_effects = !isnothing(Xfixed_betas) ? summarize_array(reshape(Xfixed_betas, M.Xfixed_N, 1, N_samples); alpha=alpha) : nothing,
            random_effects = !isnothing(random_eff) ? summarize_array(reshape(random_eff, M.N_cat, 1, N_samples); alpha=alpha) : nothing,
            predictions_observed_denoised = summarize_array(reshape(preds_denoised[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
            predictions_observed_noisy = summarize_array(reshape(preds_noisy[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
            predictions_strata_denoised = N_PS > 0 ? summarize_array(reshape(preds_denoised[M.y_N+1:end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
            predictions_strata_noisy = N_PS > 0 ? summarize_array(reshape(preds_noisy[M.y_N+1:end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
            post_strat_weights = isnothing(weights) ? nothing : (mean=vec(mean(weights, dims=2)), samples=weights),
            waic = _compute_waic(log_lik[:, 1:M.y_N])
        )
    end
    return (outcomes = outcome_results, phi_zi = phi_zi, arch = arch)
end


function _reconstruct(arch::MultifidelityArchitecture, fam::ModelFamily,  modelname, chain, M, PS, alpha)
    N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    N_tot = M.y_N + N_PS
    N_samples = size(chain, 1)
    p_names_str = string.(FlexiChains.parameters(chain))

    s_sigmas = "s_sigma" in p_names_str ? vec(get_params_vector(chain, "s_sigma", 1)) : ones(N_samples)
    t_sigmas = "t_sigma" in p_names_str ? vec(get_params_vector(chain, "t_sigma", 1)) : ones(N_samples)
    u_sigmas = "u_sigma" in p_names_str ? vec(get_params_vector(chain, "u_sigma", 1)) : zeros(N_samples)

    s_eff_struct = zeros(M.s_N, N_samples); s_eff_noisy = zeros(M.s_N, N_samples)
    t_eff = zeros(M.t_N, N_samples); u_eff = zeros(get(M, :u_N, 1), N_samples)
    z_latent = zeros(N_tot, N_samples)
    Xfixed_betas = (M.Xfixed_N > 0) ? zeros(M.Xfixed_N, N_samples) : nothing
    random_eff = "c_latent_s_idx" in p_names_str ? zeros(length(M.c_re_templates[:s_idx].matrix), N_samples) : nothing # Adjust based on your random effect setup

    # Identify fixed-effects column names from the observation model
    fixed_names = names(M.Xfixed, 2)
    # fixed_names = M.Xfixed isa NamedArray ? names(M.Xfixed, 2) : [Symbol("x$i") for i in 1:size(M.Xfixed, 2)]

    for j in 1:N_samples
        fs, fn = extract_manifold(SpatialTrait(), chain, M, j, s_sigmas[j])
        s_eff_struct[:, j] .= fs; s_eff_noisy[:, j] .= fn
        t_eff[:, j] .= extract_manifold(TemporalTrait(), chain, M, j, t_sigmas[j])
        if M.model_season != "none"
            u_eff[:, j] .= extract_manifold(SeasonalTrait(), chain, M, j, u_sigmas[j])
        end
        if !isnothing(Xfixed_betas)
            Xfixed_betas[:, j] .= extract_manifold(FixedEffectTrait(), chain, M, j)
        end
        if !isnothing(random_eff)
             # Assuming 's_idx' is the random effect group for multifidelity
            random_eff[:, j] .= get_params_vector(chain, "c_latent_s_idx", size(M.c_re_templates[:s_idx].matrix, 1))[j, :]
        end
        z_latent[1:M.y_N, j] .= vec(chain[:z_latent].data[j])
    end

    eta_noisy = zeros(N_tot, N_samples); eta_denoised = zeros(N_tot, N_samples)
    z_beta = vec(get_params_vector(chain, "fidelity_rho", 1))

    for j in 1:N_samples, i in 1:N_tot
        is_obs = i <= M.y_N; idx = is_obs ? i : i - M.y_N; src = is_obs ? M : PS
        s_idx = Int(src.s_idx[idx]); t_idx = Int(src.t_idx[idx])

        base = (is_obs ? M.log_offset[i] : 0.0) + t_eff[t_idx, j] + z_latent[i, j] * z_beta[j]
        if !isnothing(Xfixed_betas)
            X_row = is_obs ? M.Xfixed[idx, :] : Matrix(PS.surface_df[idx, fixed_names])
            base += dot(Xfixed_betas[:, j], X_row)
        end
        if !isnothing(random_eff)
            # Assuming 's_idx' is the grouping for random effects, map observation index to group index
            c_level_val = is_obs ? M.c_groups[:s_idx][idx] : 1 # Placeholder for PS if needed
            base += random_eff[c_level_val, j]
        end

        eta_denoised[i, j] = base + s_eff_struct[s_idx, j]
        eta_noisy[i, j] = base + s_eff_noisy[s_idx, j]
        if M.model_season != "none"
            u_idx = Int(src.u_idx[idx])
            eta_denoised[i, j] += u_eff[u_idx, j]
            eta_noisy[i, j] += u_eff[u_idx, j]
        end
    end

    y_sigma = _extract_volatility(chain, p_names_str, N_tot, N_samples)
    preds_denoised, preds_noisy, log_lik = _process_ll_and_predictions(fam, eta_noisy, chain, M, N_tot, N_samples, y_sigma, M.y_obs)
    field_denoised, _, _ = _process_ll_and_predictions(fam, eta_denoised, chain, M, N_tot, N_samples, y_sigma, M.y_obs)

    weights = nothing
    if N_PS > 0
        s_idx_obs = M.s_idx; t_idx_obs = M.t_idx; s_N = M.s_N
        stratum_map = [(Int(t_idx_obs[i])-1)*s_N + Int(s_idx_obs[i]) for i in 1:M.y_N]
        weights = field_denoised[M.y_N .+ stratum_map, :] ./ (field_denoised[1:M.y_N, :] .+ 1e-9)
    end

    return (
        spatial_structured = summarize_array(reshape(s_eff_struct, M.s_N, 1, N_samples); alpha=alpha),
        spatial_unstructured = summarize_array(reshape(s_eff_noisy .- s_eff_struct, M.s_N, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(t_eff, M.t_N, 1, N_samples); alpha=alpha),
        seasonal = M.model_season != "none" ? summarize_array(reshape(u_eff, get(M, :u_N, 1), 1, N_samples); alpha=alpha) : nothing,
        volatility = summarize_array(reshape(y_sigma, N_tot, 1, N_samples); alpha=alpha),
        fixed_effects = !isnothing(Xfixed_betas) ? summarize_array(reshape(Xfixed_betas, M.Xfixed_N, 1, N_samples); alpha=alpha) : nothing,
        random_effects = !isnothing(random_eff) ? summarize_array(reshape(random_eff, length(M.c_re_templates[:s_idx].matrix), 1, N_samples); alpha=alpha) : nothing, # Adjust based on random effect setup
        predictions_observed_denoised = summarize_array(reshape(preds_denoised[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_observed_noisy = summarize_array(reshape(preds_noisy[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_strata_denoised = N_PS > 0 ? summarize_array(reshape(preds_denoised[M.y_N+1:end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        predictions_strata_noisy = N_PS > 0 ? summarize_array(reshape(preds_noisy[M.y_N+1:end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        post_strat_weights = isnothing(weights) ? nothing : (mean=vec(mean(weights, dims=2)), samples=weights),
        waic = _compute_waic(log_lik[:, 1:M.y_N]),
        family = fam, arch = arch
    )
end


function _reconstruct(arch::ExampleArchitecture, fam::ModelFamily,  modelname, chain, M, PS, alpha)
    N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    N_tot = M.y_N + N_PS
    N_samples = size(chain, 1)
    p_names_str = string.(FlexiChains.parameters(chain))

    # 1. Parameter Extraction & Model Type Detection
    is_kriging = "signal_std_dev[1]" in p_names_str
    is_spatiotemporal_gp = "ls_s" in p_names_str && "ls_t" in p_names_str

    # 2. Field Storage
    f_latent_samples = zeros(N_tot, N_samples)
    Xfixed_betas = (M.Xfixed_N > 0) ? zeros(M.Xfixed_N, N_samples) : nothing

    for j in 1:N_samples
        # Fixed Effects
        if !isnothing(Xfixed_betas)
            Xfixed_betas[:, j] .= get_params_vector(chain, "Xfixed_beta", M.Xfixed_N)[j, :]
        end

        if is_kriging
            # --- Kriging Reconstruction Logic ---
            # Placeholder extraction: f_latent_all is expected to hold mapped spatial field
            if "f_latent_all" in p_names_str
                 f_latent_samples[1:M.y_N, j] .= vec(chain[Symbol("f_latent_all")].data[j])
            end
            # Note: N_PS Kriging math would go here if PS indices were defined

        elseif is_spatiotemporal_gp
            # --- Spatiotemporal GP (example_gp_gaussian) ---
            u_latent = vec(chain[:s_latent].data[j])
            sig_st = chain[:st_sigma].data[j]
            l_s = chain[:ls_s].data[j]
            l_t = chain[:ls_t].data[j]
            
            k = (sig_st^2) * (SqExponentialKernel() ∘ ScaleTransform(inv(l_s))) ⊗ (SqExponentialKernel() ∘ ScaleTransform(inv(l_t)))
            f_gp = GP(k)
            
            Ns, Nt = size(M.s_inducing, 2), size(M.t_inducing, 2)
            s_grid = repeat(M.s_inducing, outer=(1, Nt))
            t_grid = repeat(M.t_inducing, inner=(1, Ns))
            X_ind = ColVecs(vcat(s_grid, t_grid))
            
            f_cond = condition(f_gp(X_ind, 1e-6), u_latent)
            X_obs = ColVecs(vcat(M.s_coord.X', reshape(M.t_coord, 1, :)))
            f_latent_samples[1:M.y_N, j] .= mean(f_cond(X_obs[M.st_idx]))
        else
            # Fallback
            fs, fn = extract_manifold(SpatialTrait(), chain, M, j, 1.0)
            for i in 1:M.y_N; f_latent_samples[i, j] = fs[M.s_idx[i]]; end
        end
    end

    # 3. Predictor Assembly
    eta_denoised = zeros(N_tot, N_samples)
    eta_noisy = zeros(N_tot, N_samples)
    fixed_names = names(M.Xfixed, 2)

    for j in 1:N_samples, i in 1:N_tot
        is_obs = i <= M.y_N; idx = is_obs ? i : i - M.y_N
        base = (is_obs ? M.log_offset[i] : 0.0)
        if !isnothing(Xfixed_betas)
            X_row = is_obs ? M.Xfixed[idx, :] : Matrix(PS.surface_df[idx, fixed_names])
            base += dot(Xfixed_betas[:, j], X_row)
        end
        eta_denoised[i, j] = base + f_latent_samples[i, j]
        eta_noisy[i, j] = base + f_latent_samples[i, j]
    end

    # 4. Final Family-Aware Transformation (Standardizing outputs to match Univariate)
    y_sigma = _extract_volatility(chain, p_names_str, N_tot, N_samples)
    preds_denoised, preds_noisy, log_lik = _process_ll_and_predictions(fam, eta_noisy, chain, M, N_tot, N_samples, y_sigma, M.y_obs)

    return (
        spatial_structured = summarize_array(reshape(f_latent_samples[1:M.s_N, :], M.s_N, 1, N_samples); alpha=alpha),
        spatial_unstructured = summarize_array(zeros(M.s_N, 1, N_samples); alpha=alpha), # Placeholder for parity
        temporal = nothing,
        seasonal = nothing,
        volatility = summarize_array(reshape(y_sigma, N_tot, 1, N_samples); alpha=alpha),
        fixed_effects = !isnothing(Xfixed_betas) ? summarize_array(reshape(Xfixed_betas, M.Xfixed_N, 1, N_samples); alpha=alpha) : nothing,
        random_effects = nothing,
        predictions_observed_denoised = summarize_array(reshape(preds_denoised[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_observed_noisy = summarize_array(reshape(preds_noisy[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_strata_denoised = N_PS > 0 ? summarize_array(reshape(preds_denoised[M.y_N+1:end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        predictions_strata_noisy = N_PS > 0 ? summarize_array(reshape(preds_noisy[M.y_N+1:end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        waic = _compute_waic(log_lik[:, 1:M.y_N]),
        family = fam, arch = arch
    )
end



@model function bstm_univariate(M, ::Type{T}=Float64) where {T}
    # --- 1. Global Likelihood Hyperparameters ---
    # Handles dispersion for Negative Binomial and probability for Zero-Inflation
    local lik_r = 1.0
    local lik_phi = 0.0
    if M.model_family == "negbin"
        # dispersion parameter r for NB likelihood
        lik_r ~ NamedDist(Exponential(1.0), :lik_r)
    end
    if M.use_zi
        # Probability of structural zero in zero-inflated models
        lik_phi ~ NamedDist(Beta(1, 1), :lik_phi)
    end

    # --- 2. Stochastic Volatility (SV) / Heteroskedasticity ---
    # Maps latent log-variance to a spatiotemporal surface using Random Fourier Features (RFF)
    local y_sigma = 1.0
    if get(M, :use_sv, false)
        sigma_log_var ~ NamedDist(Exponential(1.0), :sigma_log_var)
        beta_vol_latent ~ NamedDist(filldist(Normal(0, 1), M.M_rff_sigma), :beta_vol_latent)
        # Projecting coordinates using cached spectral basis (vol_proj) from bstm_options
        y_sigma = exp.((sigma_log_var .* (sqrt(2.0 / M.M_rff_sigma) .* cos.(M.vol_proj) * beta_vol_latent)) ./ 2.0)
    elseif M.model_family in ["gaussian", "lognormal"]
        # Homoskedastic observation error
        y_sigma ~ NamedDist(Exponential(1.0), :y_sigma)
    end

    # --- 3. Base Predictor Initialization (Fixed Effects) ---
    # Initialize with global intercept and fixed covariates defined in Xfixed
    Xfixed_beta ~ NamedDist(MvNormal(zeros(M.Xfixed_N), 5.0 * I), :Xfixed_beta)
    Xfixed_eff = M.Xfixed * Xfixed_beta
    
    # Initialize linear predictor (eta) with log offsets (exposure) and fixed effects
    eta = (haskey(M, :log_offset) ? M.log_offset : zeros(T, M.y_N)) .+ Xfixed_eff

    # --- 4. Hierarchical Multi-Resolution Spatial Manifolds ---
    # Allows variance partitioning across nested scales (e.g., Region -> County -> Unit)
    if haskey(M, :spatial_hierarchy) && !isempty(M.spatial_hierarchy)
        for scale_sym in keys(M.spatial_hierarchy)
            scale = M.spatial_hierarchy[scale_sym]
            
            # Resolution-specific variance and spatial correlation parameters
            sig_scale ~ NamedDist(Exponential(1.0), Symbol("s_sigma_", scale_sym))
            rho_scale ~ NamedDist(Beta(1, 1), Symbol("s_rho_", scale_sym))

            # Recompose precision (bym2, icar, leroux, etc.) for this specific resolution
            Q_scale = recompose_precision(Symbol(scale.model), scale.template.matrix, sig_scale; extra_param=rho_scale, noise=M.noise)

            # Sample resolution-level latent field and map to observation indices
            f_scale ~ NamedDist(MvNormalCanon(zeros(scale.n_units), Q_scale), Symbol("s_latent_", scale_sym))
            eta .+= f_scale[scale.indices]
        end
    end

    # --- 5. Categorical Covariate Manifolds (ce / re) ---
    # Handles group-level random effects with structural priors (RW2, AR1, or IID)
    if haskey(M, :c_groups) && !isempty(M.c_groups)
        c_names = collect(keys(M.c_groups))
        sig_c ~ NamedDist(filldist(Exponential(1.0), length(c_names)), :sig_c)
        for (i, c_sym) in enumerate(c_names)
            temp = M.c_re_templates[c_sym]
            rule = Symbol(get(M.re_rules, string(c_sym), "iid"))
            c_Q = recompose_precision(rule, temp.matrix, sig_c[i]; noise=M.noise)
            c_latent_i ~ NamedDist(MvNormalCanon(zeros(size(temp.matrix, 1)), c_Q), Symbol("c_latent_", c_sym))
            eta .+= c_latent_i[M.c_groups[c_sym]]
        end
    end

    # --- 6. Advanced Basis Effects (P-splines, FFT, Wavelets) ---
    # Incorporates pre-computed basis matrices for smooth non-linear components
    if haskey(M, :basis_matrices) && !isempty(M.basis_matrices)
        for v_sym in keys(M.basis_matrices)
            B = M.basis_matrices[v_sym]
            M_basis = size(B, 2)
            sig_basis ~ NamedDist(Exponential(1.0), Symbol("sig_basis_", v_sym))
            beta_basis ~ NamedDist(MvNormal(zeros(M_basis), sig_basis^2 * I), Symbol("beta_basis_", v_sym))
            eta .+= B * beta_basis
        end
    end

    # --- 7. Mixed Effects (me: Varying Slopes) ---
    # Allows regression coefficients to vary across categorical group levels
    if haskey(M, :mixed_terms) && !isempty(M.mixed_terms)
        for (i, m_term) in enumerate(M.mixed_terms)
            sig_me_i ~ NamedDist(Exponential(1.0), Symbol("sig_me_", i))
            beta_group_i ~ NamedDist(filldist(Normal(0, sig_me_i), m_term.n_cat), Symbol("beta_group_", i))
            eta .+= beta_group_i[m_term.indices] .* m_term.covariate_vals
        end
    end

    # --- 8. Interaction Smooths (ie: 2D surfaces) ---
    # Uses Random Fourier Features (RFF) to approximate non-linear interaction surfaces f(x,y)
    if haskey(M, :interaction_terms) && !isempty(M.interaction_terms)
        for (i, i_term) in enumerate(M.interaction_terms)
            sig_ie_i ~ NamedDist(Exponential(1.0), Symbol("sig_ie_", i))
            # Projection logic: sqrt(2/M) * cos(Coords*W + b)
            projection = (i_term.coords * i_term.W_ie) .+ i_term.b_ie'
            beta_ie_i ~ NamedDist(filldist(Normal(0, sig_ie_i), i_term.M_rff), Symbol("beta_ie_", i))
            eta .+= sqrt(2.0/i_term.M_rff) .* cos.(projection) * beta_ie_i
        end
    end

    # --- 9. Eigen-Effects (ee: PCA Outcome Manifolds) ---
    # Maps high-dimensional auxiliary manifolds back to the univariate predictor
    if haskey(M, :eigen_terms) && !isempty(M.eigen_terms)
        for (i, e_term) in enumerate(M.eigen_terms)
            v_pca_i ~ NamedDist(filldist(Normal(0, 1.0), length(e_term.ltri_indices)), Symbol("v_pca_", i))
            pca_sd_i ~ NamedDist(filldist(Exponential(1.0), e_term.n_dims), Symbol("pca_sd_", i))
            pdef_sd_i ~ NamedDist(Exponential(0.1), Symbol("pdef_sd_", i))
            K_pca, _, _ = householder_transform(v_pca_i, e_term.n_dims, e_term.n_dims, e_term.ltri_indices, pca_sd_i, pdef_sd_i, M.noise)
            eta .+= vec(e_term.data * K_pca)
        end
    end

    # --- 10. Spatially Varying Coefficients (SVC via RFF basis caching) ---
    # Non-stationary relationships beta(s) * X
    if !isempty(M.svc_covariates) && M.svc_model == "rff"
        for (k, c_sym) in enumerate(M.svc_covariates)
            sig_svc_k ~ NamedDist(Exponential(1.0), Symbol("sig_svc_", c_sym))
            beta_svc_k ~ NamedDist(filldist(Normal(0, 1), M.svc_M_rff), Symbol("beta_svc_", c_sym))
            # Use cached basis from bstm_options to avoid trigonometric overhead in sampler
            z_k = haskey(M, :svc_basis_cached) ? M.svc_basis_cached : (sqrt(2.0 / M.svc_M_rff) .* cos.((M.s_coord * M.W_fixed_svc) .+ M.b_fixed_svc'))
            beta_s_k = z_k[M.s_idx, :] * beta_svc_k
            col_idx = findfirst(==(c_sym), Symbol.(names(M.Xfixed, 2)))
            if !isnothing(col_idx)
                eta .+= (beta_s_k .* sig_svc_k) .* M.Xfixed[:, col_idx]
            end
        end
    end

    # --- 11. Primary Spatial Manifold ---
    local s_eta = zeros(T, M.s_N)
    if M.model_space != "none"
        s_sigma ~ NamedDist(Exponential(1.0), :s_sigma)
        m_space = Symbol(M.model_space)
        # Detect if the model requires a correlation/rho parameter
        s_rho = (m_space in [:bym2, :leroux, :sar, :dag]) ? (~ NamedDist(Beta(1, 1), :s_rho)) : nothing
        s_Q = recompose_precision(m_space, M.s_Q_template.matrix, s_sigma; extra_param=s_rho, noise=M.noise)
        s_latent ~ NamedDist(MvNormalCanon(zeros(M.s_N), s_Q), :s_latent)
        s_eta = s_latent
        eta .+= s_eta[M.s_idx]
    end

    # --- 12. Primary Temporal Manifold ---
    if M.model_time != "none"
        t_sigma ~ NamedDist(Exponential(1.0), :t_sigma)
        m_time = Symbol(M.model_time)
        t_rho = m_time == :ar1 ? (~ NamedDist(Beta(2, 2), :t_rho)) : nothing
        t_Q = recompose_precision(m_time, M.t_Q_template.matrix, t_sigma; extra_param=t_rho, noise=M.noise)
        t_latent ~ NamedDist(MvNormalCanon(zeros(M.t_N), t_Q), :t_latent)
        eta .+= t_latent[M.t_idx]
    end

    # --- 13. Seasonal Manifold ---
    if M.model_season != "none"
        u_sigma ~ NamedDist(Exponential(1.0), :u_sigma)
        m_season = Symbol(M.model_season)
        if m_season == :harmonic
            u_steps = 1:M.u_N; u_alpha ~ NamedDist(Normal(0, 1), :u_alpha); u_beta ~ NamedDist(Normal(0, 1), :u_beta)
            eta .+= (u_alpha .* sin.(2π .* u_steps[M.u_idx] ./ 12.0) .+ u_beta .* cos.(2π .* u_steps[M.u_idx] ./ 12.0)) .* u_sigma
        else
            u_rho = m_season == :ar1 ? (~ NamedDist(Beta(2, 2), :u_rho)) : nothing
            u_Q = recompose_precision(m_season, M.u_Q_template.matrix, u_sigma; extra_param=u_rho, noise=M.noise)
            u_latent ~ NamedDist(MvNormalCanon(zeros(M.u_N), u_Q), :u_latent)
            eta .+= u_latent[M.u_idx]
        end
    end

    # --- 14. Spatiotemporal Interaction & Transport ---
    if M.model_st != "none"
        st_sigma ~ NamedDist(Exponential(1.0), :st_sigma)
        m_st = Symbol(M.model_st)
        if m_st in [:diffusion, :advection, :advection_diffusion]
            # Physics-informed transport manifolds
            st_diff ~ NamedDist(Exponential(0.5), :st_diff); st_adv ~ NamedDist(Exponential(0.5), :st_adv); st_pers ~ NamedDist(MvNormal(zeros(M.s_N), I), :st_pers)
            L_phys = Matrix(Diagonal(vec(sum(M.W, dims=2))) - M.W)
            st_innov ~ filldist(Normal(0, st_sigma), M.s_N, M.t_N)
            st_eta_map = zeros(T, M.s_N, M.t_N)
            st_eta_map[:, 1] .= st_innov[:, 1]
            for t in 2:M.t_N
                # Numerical integration for spread: μ_t = persistence * (μ_{t-1} - Laplacian effects)
                mu_p = st_eta_map[:, t-1] .- (m_st != :advection ? st_diff .* (L_phys * st_eta_map[:, t-1]) : 0.0) .- (m_st != :diffusion ? st_adv .* (L_phys * s_eta) : 0.0)
                st_eta_map[:, t] .= logistic.(st_pers) .* mu_p .+ st_innov[:, t]
            end
            for i in 1:M.y_N; eta[i] += st_eta_map[M.s_idx[i], M.t_idx[i]]; end
        else
            # Standard Kronecker-structured Interactions (Knorr-Held Types I-IV)
            Q_s_st = (m_st in [:III, :IV]) ? M.s_Q_template.matrix : I(M.s_N)
            Q_t_st = (m_st in [:II, :IV]) ? M.t_Q_template.matrix : I(M.t_N)
            st_Q_kron = Symmetric(kron(Q_t_st, Q_s_st) .* (1.0 / (st_sigma^2 + M.noise)) + M.noise * I)
            st_latent ~ NamedDist(MvNormalCanon(zeros(M.s_N * M.t_N), st_Q_kron), :st_latent)
            eta .+= st_latent[(M.t_idx .- 1) .* M.s_N .+ M.s_idx]
        end
    end

    # --- 15. Likelihood Execution ---
    # Ensure linear predictor stays within stable bounds for link functions
    eta = clamp.(eta, -20.0, 20.0)
    # Mask missing observations (Likelihood is conditional on observed data)
    good_idx = findall(!ismissing, M.y_obs)
    Turing.@addlogprob! logpdf(bstm_Likelihood(M.model_family, M.use_zi, M.weights[good_idx], lik_phi, lik_r, y_sigma, M.trials[good_idx], M.y_obs[good_idx]), eta[good_idx])
end


@model function bstm_multivariate(M, ::Type{T}=Float64) where {T}
    # --- 1. SETUP & DIMENSIONS ---
    y_N = M[:y_N]
    outcomes_N = M[:outcomes_N]
    model_type = get(M, :model_type, "component_wise")
    noise = get(M, :noise, 1e-6)
    n_latent = (model_type == "pca_factor") ? get(M, :N_factors, 2) : outcomes_N

    # Global Likelihood Hyperpriors
    local lik_r = fill(1.0, outcomes_N)
    if M[:model_family] == "negbin"
        lik_r_raw ~ filldist(Exponential(1.0), outcomes_N)
        lik_r = lik_r_raw
    end
    local phi_zi = 0.0
    if get(M, :use_zi, false); phi_zi ~ Beta(1, 1); end

    # --- 2. VECTORIZED HYPERPRIORS ---
    s_sigma_arr ~ filldist(Exponential(1.0), n_latent)
    t_sigma_arr ~ filldist(Exponential(1.0), n_latent)
    s_rho_arr ~ filldist(Beta(1, 1), n_latent)
    t_rho_arr ~ filldist(Beta(2, 2), n_latent)

    m_season = Symbol(get(M, :model_season, "none"))
    u_sigma_arr = (m_season != :none) ? (~ filldist(Exponential(1.0), n_latent)) : zeros(T, n_latent)

    m_st = Symbol(get(M, :model_st, "none"))
    st_sigma_arr = (m_st != :none) ? (~ filldist(Exponential(0.5), n_latent)) : zeros(T, n_latent)

    # Pre-allocate latent matrices outside the loop for rank consistency
    s_eta = zeros(T, M[:s_N], n_latent)
    t_eta = zeros(T, M[:t_N], n_latent)
    u_eta = zeros(T, M[:u_N], n_latent)
    st_eta = zeros(T, M[:s_N], M[:t_N], n_latent)

    # --- 3. STOCHASTIC VOLATILITY ---
    local y_sigma = ones(T, y_N, outcomes_N)
    if get(M, :use_sv, false)
        sigma_log_var ~ filldist(Exponential(1.0), outcomes_N)
        for k in 1:outcomes_N
            v_idx = Symbol("beta_vol_", k)
            v_latent ~ NamedDist(filldist(Normal(0, 1), M[:M_rff_sigma]), v_idx)
            y_sigma[:, k] .= exp.((sigma_log_var[k] .* (sqrt(2.0 / M[:M_rff_sigma]) .* cos.(M[:vol_proj]) * v_latent)) ./ 2.0)
        end
    elseif M[:model_family] in ["gaussian", "lognormal"]
        y_sigma_const ~ filldist(Exponential(1.0), outcomes_N)
        for k in 1:outcomes_N; y_sigma[:, k] .= y_sigma_const[k]; end
    end

    # --- 4. LATENT MANIFOLD LOOP (with Unique Indexing) ---
    for k in 1:n_latent
        # A. Spatial
        m_space = Symbol(get(M, :model_space, "none"))
        if m_space != :none
            s_Q_k = recompose_precision(m_space, M[:s_Q_template][:matrix], s_sigma_arr[k]; extra_param=s_rho_arr[k], noise=noise)
            s_eta[:, k] .= (~ NamedDist(MvNormalCanon(zeros(M[:s_N]), s_Q_k), Symbol("s_latent_", k)))
        end

        # B. Temporal
        m_time = Symbol(get(M, :model_time, "none"))
        if m_time != :none
            t_Q_k = recompose_precision(m_time, M[:t_Q_template][:matrix], t_sigma_arr[k]; extra_param=t_rho_arr[k], noise=noise)
            t_eta[:, k] .= (~ NamedDist(MvNormalCanon(zeros(M[:t_N]), t_Q_k), Symbol("t_latent_", k)))
        end

        # C. Seasonal
        if m_season != :none
            u_eta[:, k] .= (~ NamedDist(MvNormalCanon(zeros(M[:u_N]), recompose_precision(m_season, M[:u_Q_template][:matrix], u_sigma_arr[k]; noise=noise)), Symbol("u_latent_", k)))
        end

        # D. Spatiotemporal Interaction (KH Type 1-4 & Transport Supported)
        if m_st != :none
            if m_st in [:diffusion, :advection, :advection_diffusion]
                diff_k ~ NamedDist(Exponential(0.5), Symbol("diff_", k))
                adv_k ~ NamedDist(Exponential(0.5), Symbol("adv_", k))
                pers_k ~ NamedDist(MvNormal(zeros(M[:s_N]), I), Symbol("pers_", k))
                st_z_k ~ NamedDist(MvNormal(zeros(M[:s_N] * M[:t_N]), I), Symbol("st_z_", k))
                st_innov = reshape(st_z_k .* st_sigma_arr[k], M[:s_N], M[:t_N])
                L_phys = Matrix(Diagonal(vec(sum(M[:W], dims=2))) - M[:W])
                st_eta[:, 1, k] .= st_innov[:, 1]
                for t in 2:M[:t_N]
                    mu_p = st_eta[:, t-1, k] .- (m_st != :advection ? diff_k .* (L_phys * st_eta[:, t-1, k]) : 0.0) .- (m_st != :diffusion ? adv_k .* (L_phys * s_eta[:, k]) : 0.0)
                    st_eta[:, t, k] .= logistic.(pers_k) .* mu_p .+ st_innov[:, t]
                end
            else
                # Knorr-Held Interactions
                Q_s = (m_st in [:III, :IV]) ? M[:s_Q_template][:matrix] : I(M[:s_N])
                Q_t = (m_st in [:II, :IV]) ? M[:t_Q_template][:matrix] : I(M[:t_N])
                st_Q_k = Symmetric(kron(Q_t, Q_s) .* (1.0/st_sigma_arr[k]^2) + noise*I)
                st_eta[:, :, k] .= reshape((~ NamedDist(MvNormalCanon(zeros(M[:s_N]*M[:t_N]), st_Q_k), Symbol("st_latent_", k))), M[:s_N], M[:t_N])
            end
        end
    end

    # --- 5. ASSEMBLER & PROJECTION ---
    latent_scores = zeros(T, y_N, n_latent)
    for k in 1:n_latent
        latent_scores[:, k] .= s_eta[M[:s_idx], k] .+ t_eta[M[:t_idx], k] .+ u_eta[M[:u_idx], k]
        if m_st != :none
            for i in 1:y_N
                latent_scores[i, k] += st_eta[M[:s_idx][i], M[:t_idx][i], k]
            end
        end
    end

    local projection_mat = I(outcomes_N)
    if model_type == "component_wise" && outcomes_N > 1
        L_corr ~ LKJCholesky(outcomes_N, 1.0, :L)
        projection_mat = L_corr.L
    end

    if model_type == "pca_factor"
        v_pca ~ filldist(Normal(0, 1), Int(outcomes_N*n_latent - n_latent*(n_latent-1)/2))
        pca_sd ~ filldist(Exponential(1.0), n_latent)
        _, _, U_loadings = householder_transform(v_pca, outcomes_N, n_latent, M[:ltri], pca_sd, 0.1, noise)
        mu_mat = latent_scores * U_loadings'
    else
        mu_mat = latent_scores * projection_mat
    end

    Xfixed_beta ~ MvNormal(zeros(M[:Xfixed_N]*outcomes_N), 2.0*I)
    mu_mat .+= reshape(M[:Xfixed] * reshape(Xfixed_beta, M[:Xfixed_N], outcomes_N), y_N, outcomes_N)

    if haskey(M, :log_offset); mu_mat .+= M[:log_offset]; end

    # --- 6. LIKELIHOOD ---
    mu_mat = clamp.(mu_mat, -20.0, 20.0)
    for k in 1:outcomes_N
        good_idx = findall(!ismissing, M[:y_obs][:, k])
        if isempty(good_idx); continue; end
        Turing.@addlogprob! logpdf(
            bstm_Likelihood(M[:model_family], get(M, :use_zi, false), M[:weights][good_idx], phi_zi, lik_r[k], y_sigma[good_idx, k], M[:trials][good_idx], M[:y_obs][good_idx, k]),
            mu_mat[good_idx, k]
        )
    end
end

@model function bstm_multifidelity(M, ::Type{T}=Float64) where {T}
    # 1. Fidelity Coupling Logic
    fidelity_rho ~ Normal(1.0, 0.5)
    fidelity_bias ~ Normal(0, 1)

    # 2. Likelihood & Global Parameters
    local lik_r = (M.model_family == "negbin") ? (~ Exponential(1.0)) : 1.0
    local lik_phi = (M.use_zi) ? (~ Beta(1, 1)) : 0.0
    local y_sigma = (M.model_family in ["gaussian", "lognormal"]) ? (~ Exponential(1.0)) : 1.0

    # 3. Fixed Effects
    local Xfixed_eff = zeros(T, M.y_N)
    if M.Xfixed_N > 0
        Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), 5.0 * I)
        Xfixed_eff = M.Xfixed * Xfixed_beta
    end

    # 4. Covariate Random Effects (Shared across fidelities)
    c_effects_sum = zeros(T, M.y_N)
    if haskey(M, :c_groups) && !isempty(M.c_groups)
        c_names = collect(keys(M.c_groups))
        sig_c ~ filldist(Exponential(1.0), length(c_names))
        for (i, c_sym) in enumerate(c_names)
            temp = M.c_re_templates[c_sym]
            c_Q = recompose_precision(Symbol(M.re_rules[c_sym]), temp.matrix, sig_c[i]; noise=M.noise)
            c_latent_i ~ NamedDist(MvNormalCanon(zeros(size(temp.matrix, 1)), c_Q), Symbol("c_latent_", c_sym))
            c_effects_sum .+= c_latent_i[M.c_groups[c_sym]]
        end
    end

    # 5. Shared Latent Field Manifolds (Pure Spatial & Temporal)
    local s_eta = zeros(T, M.s_N)
    if M.model_space != "none"
        s_sigma ~ Exponential(1.0)
        m_space = Symbol(M.model_space)
        if m_space in [:bym2, :leroux, :sar, :dag]
            s_rho ~ Beta(1, 1)
            s_Q = recompose_precision(m_space, M.s_Q_template.matrix, s_sigma; extra_param=s_rho, noise=M.noise)
        elseif m_space in [:gp, :dense_gp, :fitc]
            s_ls ~ Exponential(1.0)
            s_Q = recompose_precision(m_space, M.s_Q_template.matrix, s_sigma; extra_param=s_ls, noise=M.noise)
        else
            s_Q = recompose_precision(m_space, M.s_Q_template.matrix, s_sigma; noise=M.noise)
        end
        s_latent ~ NamedDist(MvNormalCanon(zeros(M.s_N), s_Q), :s_latent)
        s_eta = s_latent
    end

    local t_eta = zeros(T, M.t_N)
    if M.model_time != "none"
        t_sigma ~ Exponential(1.0)
        if M.model_time == "ar1"
            t_rho ~ Beta(2, 2)
            t_Q = recompose_precision(:ar1, M.t_Q_template.matrix, t_sigma; extra_param=t_rho, noise=M.noise)
        else
            t_Q = recompose_precision(Symbol(M.model_time), M.t_Q_template.matrix, t_sigma; noise=M.noise)
        end
        t_latent ~ NamedDist(MvNormalCanon(zeros(M.t_N), t_Q), :t_latent)
        t_eta = t_latent
    end

    # 6. Spatiotemporal (ST) Manifold - Now hosting Transport logic
    local st_eta = zeros(T, M.s_N * M.t_N)
    if M.model_st != "none"
        st_sigma ~ Exponential(1.0)
        m_st = Symbol(M.model_st)

        # Dispatch for Transport (Diffusion/Advection) vs standard Interactions
        if m_st in [:diffusion, :advection, :advection_diffusion]
            st_rho ~ Beta(1, 1)
            # Transport precision is reconstructed over the full ST grid
            st_Q = recompose_precision(m_st, M.st_template.matrix, st_sigma; extra_param=st_rho, noise=M.noise)
        else
            # Standard KH Types I-IV
            Q_s_st = (M.model_st in ["III", "IV"]) ? M.s_Q_template.matrix : I(M.s_N)
            Q_t_st = (M.model_st in ["II", "IV"]) ? M.t_Q_template.matrix : I(M.t_N)
            st_Q = Symmetric(Matrix(kron(Q_t_st, Q_s_st)) .* (1.0/st_sigma^2) + M.noise*I)
        end

        st_latent ~ NamedDist(MvNormalCanon(zeros(M.s_N * M.t_N), st_Q), :st_latent)
        st_eta = st_latent
    end

    # 7. Predictor Assembly
    shared_latent = s_eta[M.s_idx] .+ t_eta[M.t_idx]
    if M.model_st != "none"; shared_latent .+= st_eta[M.st_idx]; end

    eta_high = Xfixed_eff .+ c_effects_sum .+ shared_latent
    eta_low  = fidelity_bias .+ (fidelity_rho .* shared_latent) .+ c_effects_sum .+ Xfixed_eff

    # 8. Multifidelity Likelihood Evaluation
    for (fid, eta_f) in [(1, eta_high), (2, eta_low)]
        idx = findall(x -> x == fid, M.fidelity_idx)
        if !isempty(idx)
            good_idx = idx[.!ismissing.(M.y_obs[idx])]
            if !isempty(good_idx)
                Turing.@addlogprob! logpdf(bstm_Likelihood(M.model_family, M.use_zi, M.weights[good_idx], lik_phi, lik_r, y_sigma, M.trials[good_idx], M.y_obs[good_idx]), eta_f[good_idx])
            end
        end
    end
end





# --- Trait-based Manifold Extraction ---
# Standardizes how we pull spatial (s), temporal (t), and seasonal (u) effects from chains.

struct ManifoldTrait{T} end
const SpatialManifold = ManifoldTrait{:spatial}()
const TemporalManifold = ManifoldTrait{:temporal}()
const SeasonalManifold = ManifoldTrait{:seasonal}()

# Map traits to the corresponding index vectors in the model options
get_manifold_indices(::ManifoldTrait{:spatial}, M) = M.s_idx
get_manifold_indices(::ManifoldTrait{:temporal}, M) = M.t_idx
get_manifold_indices(::ManifoldTrait{:seasonal}, M) = M.u_idx

# --- Centralized Prediction & Likelihood Logic ---

"""
    _apply_link_and_lik(family, eta, use_zi, [phi], [r])

Applies the appropriate link function (Inverse Link) to the linear predictor `eta` 
and accounts for zero-inflation or dispersion to return the expected value (mu).
"""
function _apply_link_and_lik(family::String, eta::AbstractArray, use_zi::Bool, phi=0.0, r=1.0)
    if family == "gaussian"
        return eta
    elseif family == "lognormal" || family == "poisson"
        return exp.(eta)
    elseif family == "bernoulli"
        return logistic.(eta)
    elseif family == "negbin"
        # Expected value for NB (mu). If ZI is active, scale by (1-phi).
        mu = exp.(eta)
        return use_zi ? (1.0 .- phi) .* mu : mu
    else
        # Fallback to identity link
        return eta
    end
end
 

  


####


 
;;

function build_structure_template(type::Symbol, n::Int; scale=true, coords=nothing, W=nothing)
    # Comprehensive Template Factory for all supported BSTM Manifolds
    
    if type == :ar1
        # AR1 template: Basic first-order differencing structure
        Q = Matrix(1.0I, n, n)
        for i in 1:(n-1); Q[i, i+1] = Q[i+1, i] = -0.5; end
        sf = 1.0 
        return (matrix = Q, scaling_factor = sf)

    elseif type == :rw2
        # RW2 template: Second-order random walk (Intrinsic GMRF)
        Q = zeros(n, n)
        for i in 1:n
            if i > 2 && i < n-1
                Q[i,i]=6; Q[i,i-1]=Q[i,i+1]=-4; Q[i,i-2]=Q[i,i+2]=1
            elseif i == 1
                Q[i,i]=1; Q[i,i+1]=-2; Q[i,i+2]=1
            elseif i == 2
                Q[i,i]=5; Q[i,i-1]=-2; Q[i,i+1]=-4; Q[i,i+2]=1
            elseif i == n-1
                Q[i,i]=5; Q[i,i+1]=-2; Q[i,i-1]=-4; Q[i,i-2]=1
            elseif i == n
                Q[i,i]=1; Q[i,i-1]=-2; Q[i,i-2]=1
            end
        end
        sf = scale ? exp(mean(log.(filter(x -> x > 1e-6, eigvals(Q))))) : 1.0
        return (matrix = Matrix(Q ./ sf), scaling_factor = sf)

    elseif type in [:icar, :besag, :bym2, :leroux, :sar, :dag, :transport_diffusion, :transport_advection]
        # Graph Laplacian based manifolds
        isnothing(W) && error("Spatial adjacency matrix W required for $type")
        D_sp = Diagonal(vec(sum(W, dims=2)))
        Q_raw = Matrix(D_sp - W)
        sf = scale ? exp(mean(log.(filter(x -> x > 1e-6, eigvals(Q_raw))))) : 1.0
        return (matrix = Matrix(Q_raw ./ sf), scaling_factor = sf)

    elseif type in [:sar, :dag, :transport_advection]
        # Adjacency for SAR/DAG/Advection
        isnothing(W) && error("Adjacency matrix W required for $type")
        row_sums = vec(sum(W, dims=2))
        W_norm = W ./ (row_sums .+ 1e-9)
        return (matrix = Matrix(W_norm), scaling_factor = 1.0)

    elseif type == :gp || type == :nystrom || type == :denseGP
        # Distance matrix for kernel-based manifolds
        mat = isnothing(coords) ? Matrix(1.0I, n, n) : [sqrt(sum((c1 .- c2).^2)) for c1 in coords, c2 in coords]
        return (matrix = mat, scaling_factor = 1.0)

    elseif type == :seasonal || type == :harmonic
        # Cyclic/Seasonal template with sum-to-zero structure
        Q = Matrix(1.0I, n, n) .* (n-1)
        for i in 1:n, j in 1:n
            if i != j; Q[i, j] = -1.0; end
        end
        sf = scale ? exp(mean(log.(filter(x -> x > 1e-6, eigvals(Q))))) : 1.0
        return (matrix = Matrix(Q ./ sf), scaling_factor = sf)

    elseif type == :iid || type == :householder || type == :bgcn
        # Identity bases for unstructured or spectral weights
        return (matrix = Matrix(1.0I, n, n), scaling_factor = 1.0)

    else
        print("Unknown structure template type (defaulting to identity): $type")
        # Identity bases for unstructured or spectral weights
        return (matrix = Matrix(1.0I, n, n), scaling_factor = 1.0)

    end
end
  
 
 

function recompose_precision(type::Symbol, template_mat::AbstractMatrix, param::Real; extra_param=nothing, noise=1e-4)
    # Consolidated precision construction for the full manifold suite
    T = typeof(param)
    n = size(template_mat, 1)

    Q = if type == :iid
        (1.0 / (param^2 + noise)) * I
    elseif type in [:besag, :icar, :transport_diffusion]
        # Standard structural precision (e.g., Laplacian)
        (1.0 / (param^2 + noise)) .* template_mat
    elseif type == :bym2
        # BYM2: rho * Structured + (1-rho) * I
        rho = isnothing(extra_param) ? 0.5 : extra_param
        (1.0 / (param^2 + noise)) .* (rho .* template_mat + (1.0 - rho) .* I)
    elseif type == :leroux
        # Leroux: lambda * Structured + (1-lambda) * I
        lambda = isnothing(extra_param) ? 0.5 : extra_param
        (1.0 / (param^2 + noise)) .* (lambda .* template_mat + (1.0 - lambda) .* I)
    elseif type == :sar || type == :transport_advection || type == :dag
        # Simultaneous Autoregressive or Directed Acyclic Graph structures
        rho = isnothing(extra_param) ? 0.5 : extra_param
        L = I(n) - rho .* template_mat
        (1.0 / (param^2 + noise)) .* (L' * L)
    elseif type == :gp
        # GP: Template is distance matrix, extra_param is lengthscale
        ls = isnothing(extra_param) ? 1.0 : extra_param
        K = (param^2) .* exp.(-template_mat.^2 ./ (2 * ls^2 + noise))
        inv(Symmetric(K + noise * I))
    elseif type == :ar1
        # Template mat contains the AR1 structure; extra_param = rho
        rho = isnothing(extra_param) ? 0.5 : extra_param
        (1.0 / (param^2 + noise)) .* template_mat
    elseif type == :householder
        # Applies a Householder rotation to the precision structure
        v = isnothing(extra_param) ? zeros(T, n) : extra_param
        H = I - 2.0 * (v * v') / (v' * v + noise)
        (1.0 / (param^2 + noise)) .* (H' * template_mat * H)
    else
        # Default fallback
        (1.0 / (param^2 + noise)) .* template_mat
    end

    # Force dense Symmetric result to assist Cholesky stability in AD
    return Symmetric(Matrix(Q) + noise * I)
end



function get_optimal_sampler(model::DynamicPPL.Model;
    nuts_n_samples_adaptation=100,
    nuts_target_acceptance_ratio=0.65,
    pg_particles=20,
    kwargs...)

    # 1. Discovery: Inspect parameter structures
    # Use a small number of samples to identify parameter types
    init_samples = [Dict(pairs(rand(model))) for _ in 1:3]
    full_init_dict = init_samples[1]

    fixed_params = filter(k -> all(s -> s[k] == full_init_dict[k], init_samples), keys(full_init_dict))
    discrete_params = filter(k -> k ∉ fixed_params && (full_init_dict[k] isa Integer || full_init_dict[k] isa Bool), keys(full_init_dict))

    # 2. Strict Gaussian/Latent Field Filter for ESS
    # We target vectors that are typically GMRF/GP realizations
    latent_fields = filter(k -> begin
        val = full_init_dict[k]
        k_str = string(k)
        k ∉ fixed_params &&
        k ∉ discrete_params &&
        val isa AbstractVector &&
        # Exclude hyperparameters and regression coefficients from ESS
        !occursin(r"sigma|sig_|rho|phi|ls_|alpha|beta|Xfixed", k_str)
    end, keys(full_init_dict))

    # 3. NUTS for non-Gaussian/Positive continuous parameters and Fixed Effects
    active_hypers = filter(k -> k ∉ fixed_params && k ∉ discrete_params && k ∉ latent_fields, keys(full_init_dict))

    sampler_blocks = []
    if !isempty(discrete_params) push!(sampler_blocks, Tuple(discrete_params) => PG(pg_particles)) end
    if !isempty(latent_fields) push!(sampler_blocks, Tuple(latent_fields) => ESS()) end
    if !isempty(active_hypers) push!(sampler_blocks, Tuple(active_hypers) => Turing.NUTS(nuts_n_samples_adaptation, nuts_target_acceptance_ratio)) end
    if !isempty(fixed_params) push!(sampler_blocks, Tuple(fixed_params) => MH()) end

    return Gibbs(sampler_blocks...)
end


function get_inits(model::DynamicPPL.Model; refine="map", n_samples=100, optimizer=LBFGS(), max_iters=500, maxtime=60.0, noise=nothing)
    println("--- Generating Initial Parameters ---")

    # 1. Heuristic Initialization from Prior Samples
    samples = [Dict(pairs(rand(model))) for _ in 1:n_samples]
    init_dict = Dict{Symbol, Any}()

    if !isnothing(noise)
        init_dict[:noise] = noise
    end

    for k in keys(samples[1])
        ks = Symbol(k)
        vals = [s[k] for s in samples]

        # Dirac/Fixed parameter check
        if all(v -> v == vals[1], vals)
            init_dict[ks] = vals[1]
            continue
        end

        # FIX: Skip averaging if the elements do not support standard arithmetic (e.g. Cholesky)
        if vals[1] isa Cholesky || vals[1] isa LKJCholesky
            init_dict[ks] = vals[1]
            continue
        end

        mu = mean(vals)
        s_name = string(ks)

        if vals[1] isa AbstractVector
            # Latent fields are centered at zero for stability
            init_dict[ks] = zeros(eltype(mu), length(mu))
        elseif occursin(r"sigma|ls_|lengthscale|sig_", s_name)
            init_dict[ks] = max(0.1, mu)
        elseif occursin("rho", s_name)
            init_dict[ks] = clamp(mu, -0.9, 0.9)
        elseif occursin("phi", s_name)
            init_dict[ks] = clamp(mu, 0.01, 0.5)
        else
            init_dict[ks] = mu
        end
    end

    # 2. Optimization Refinement
    if refine == "map"
        try
            println("Refining inits with Maximum A Posteriori (MAP)...")
            map_res = maximum_a_posteriori(model, optimizer;
                initial_params=DynamicPPL.InitFromParams(NamedTuple(init_dict)),
                iterations=max_iters, maxtime=maxtime)
            return DynamicPPL.InitFromParams(NamedTuple(map_res.params))
        catch e
            @warn "MAP refinement failed ($e). Using heuristic inits."
        end
    end

    return DynamicPPL.InitFromParams(NamedTuple(init_dict))
end



 
abstract type ManifoldTrait end
struct SpatialTrait <: ManifoldTrait end
struct TemporalTrait <: ManifoldTrait end
struct SeasonalTrait <: ManifoldTrait end
struct FixedEffectTrait <: ManifoldTrait end
struct MixedEffectTrait <: ManifoldTrait end
struct RandomEffectTrait <: ManifoldTrait end

# --- Trait-based Manifold Extractors ---
function extract_manifold(::SpatialTrait, chain, M, j, s_sigma)
    s_N = M.s_N
    field_struct = zeros(s_N)
    field_noisy = zeros(s_N)
    p_syms = FlexiChains.parameters(chain)
    spatial_key = :s_icar in p_syms ? :s_icar : (:s_latent in p_syms ? :s_latent : nothing)

    if !isnothing(spatial_key)
        icar = vec(chain[spatial_key].data[j])
        if M.model_space == "bym2" && :s_rho in p_syms
            rho = clamp(chain[:s_rho].data[j], 0.0, 1.0)
            iid = vec(chain[:s_iid].data[j])
            field_struct = s_sigma .* sqrt(rho) .* icar
            field_noisy = field_struct .+ (s_sigma .* sqrt(1.0 - rho) .* iid)
        else
            field_struct = icar .* s_sigma
            field_noisy = field_struct
        end
    end
    return field_struct, field_noisy
end

function extract_manifold(::TemporalTrait, chain, M, j, t_sigma)
    t_N = M.t_N
    field = zeros(t_N)
    p_syms = FlexiChains.parameters(chain)
    t_key = :t_latent in p_syms ? :t_latent : (:t_raw in p_syms ? :t_raw : nothing)
    if !isnothing(t_key)
        field = vec(chain[t_key].data[j]) .* t_sigma
    end
    return field
end

function extract_manifold(::SeasonalTrait, chain, M, j, u_sigma)
    u_N = get(M, :u_N, 1)
    field = zeros(u_N)
    p_syms = FlexiChains.parameters(chain)
    u_key = :u_latent in p_syms ? :u_latent : (:u_raw in p_syms ? :u_raw : nothing)
    if !isnothing(u_key)
        field = vec(chain[u_key].data[j]) .* u_sigma
    end
    return field
end

function extract_manifold(::FixedEffectTrait, chain, M, j)
    if M.Xfixed_N > 0
        return get_params_vector(chain, "Xfixed_beta", M.Xfixed_N)[j, :]
    end
    return Float64[]
end

function extract_manifold(::RandomEffectTrait, chain, M, j)
    p_names = string.(FlexiChains.parameters(chain))
    if "c_eta" in p_names
        return vec(chain[:c_eta].data[j])
    end
    return Float64[]
end
 


function _reconstruct(arch::UnivariateArchitecture, fam::ModelFamily, modelname, chain, M, PS, alpha)
    # --- 1. Dimensions and Metadata ---
    N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    N_tot = M.y_N + N_PS
    N_samples = size(chain, 1)
    p_names_str = string.(FlexiChains.parameters(chain))

    # --- 2. Global Hyperparameter Extraction ---
    # Discover scaling sigmas for all latent manifolds
    s_sigmas = "s_sigma" in p_names_str ? vec(get_params_vector(chain, "s_sigma", 1)) : ones(N_samples)
    t_sigmas = "t_sigma" in p_names_str ? vec(get_params_vector(chain, "t_sigma", 1)) : ones(N_samples)
    u_sigmas = "u_sigma" in p_names_str ? vec(get_params_vector(chain, "u_sigma", 1)) : zeros(N_samples)
    st_sigmas = "st_sigma" in p_names_str ? vec(get_params_vector(chain, "st_sigma", 1)) : zeros(N_samples)

    # --- 3. Latent Field Initialization ---
    s_eff_struct = zeros(M.s_N, N_samples)
    s_eff_noisy = zeros(M.s_N, N_samples)
    t_eff = zeros(M.t_N, N_samples)
    u_eff = zeros(get(M, :u_N, 1), N_samples)
    st_eff_map = zeros(M.s_N, M.t_N, N_samples)

    # Feature-based regression storage
    Xfixed_betas = (M.Xfixed_N > 0) ? zeros(M.Xfixed_N, N_samples) : nothing

    # Hierarchical and Categorical storage (Dictionary-driven for stability)
    hierarchical_fields = Dict{Symbol, Matrix{Float64}}()
    if haskey(M, :spatial_hierarchy)
        for scale_sym in keys(M.spatial_hierarchy)
            hierarchical_fields[scale_sym] = zeros(M.spatial_hierarchy[scale_sym].n_units, N_samples)
        end
    end

    categorical_eff = Dict{Symbol, Matrix{Float64}}()
    if haskey(M, :c_groups) && haskey(M, :c_re_templates)
        for c_sym in keys(M.c_groups)
            n_lvls = size(M.c_re_templates[c_sym].matrix, 1)
            categorical_eff[c_sym] = zeros(n_lvls, N_samples)
        end
    end

    # Effect-specific storage for advanced manifolds
    mixed_eff = !isempty(get(M, :mixed_terms, [])) ? [zeros(N_tot, N_samples) for _ in M.mixed_terms] : nothing
    interaction_eff = !isempty(get(M, :interaction_terms, [])) ? [zeros(N_tot, N_samples) for _ in M.interaction_terms] : nothing
    eigen_eff = !isempty(get(M, :eigen_terms, [])) ? [zeros(N_tot, N_samples) for _ in M.eigen_terms] : nothing
    svc_eff = !isempty(get(M, :svc_covariates, [])) ? [zeros(N_tot, N_samples) for _ in M.svc_covariates] : nothing
    basis_eff = !isempty(get(M, :basis_matrices, Dict())) ? Dict(v => zeros(N_tot, N_samples) for v in keys(M.basis_matrices)) : nothing

    # --- 4. Iterative Sample Reconstruction ---
    for j in 1:N_samples
        # 4.1 Base Manifolds (Trait-based)
        fs, fn = extract_manifold(SpatialTrait(), chain, M, j, s_sigmas[j])
        s_eff_struct[:, j] .= fs
        s_eff_noisy[:, j] .= fn
        t_eff[:, j] .= extract_manifold(TemporalTrait(), chain, M, j, t_sigmas[j])
        if M.model_season != "none"
            u_eff[:, j] .= extract_manifold(SeasonalTrait(), chain, M, j, u_sigmas[j])
        end

        # 4.2 Spatiotemporal & Hierarchical Map
        if "st_latent" in p_names_str
            st_eff_map[:, :, j] .= reshape(vec(chain[:st_latent].data[j]), M.s_N, M.t_N) .* st_sigmas[j]
        end
        for scale_sym in keys(hierarchical_fields)
            p_key = Symbol("s_latent_", scale_sym)
            if p_key in FlexiChains.parameters(chain); hierarchical_fields[scale_sym][:, j] .= vec(chain[p_key].data[j]); end
        end

        # 4.3 Fixed & Categorical Effects
        if !isnothing(Xfixed_betas); Xfixed_betas[:, j] .= extract_manifold(FixedEffectTrait(), chain, M, j); end
        for c_sym in keys(categorical_eff)
            c_name = Symbol("c_latent_", c_sym)
            if c_name in FlexiChains.parameters(chain); categorical_eff[c_sym][:, j] .= vec(chain[c_name].data[j]); end
        end

        # 4.4 Advanced Manifolds (RFF Interaction & SVC)
        if !isnothing(interaction_eff)
            for (idx, i_term) in enumerate(M.interaction_terms)
                beta_ie = get_params_vector(chain, Symbol("beta_ie_", idx), i_term.M_rff)[j, :]
                all_coords = (N_PS > 0) ? vcat(i_term.coords, Matrix(PS.surface_df[:, [:lat, :lon]])) : i_term.coords
                proj = (all_coords * i_term.W_ie) .+ i_term.b_ie'
                interaction_eff[idx][:, j] .= sqrt(2.0/i_term.M_rff) .* cos.(proj) * beta_ie
            end
        end

        if !isnothing(svc_eff) && get(M, :svc_model, "none") == "rff"
            for (idx, c_sym) in enumerate(M.svc_covariates)
                sig_svc = get_params_vector(chain, Symbol("sig_svc_", c_sym), 1)[j]
                beta_svc = get_params_vector(chain, Symbol("beta_svc_", c_sym), M.svc_M_rff)[j, :]
                all_s_coord = (N_PS > 0) ? vcat(M.s_coord_mat, Matrix(PS.surface_df[:, [:lat, :lon]])) : M.s_coord_mat
                proj_svc = (all_s_coord * M.W_fixed_svc) .+ M.b_fixed_svc'
                z_k = sqrt(2.0 / M.svc_M_rff) .* cos.(proj_svc)
                all_s_idx = (N_PS > 0) ? vcat(M.s_idx, PS.s_idx) : M.s_idx
                cov_vals = (N_PS > 0) ? vcat(M.Xfixed[:, c_sym], PS.surface_df[:, c_sym]) : M.Xfixed[:, c_sym]
                svc_eff[idx][:, j] .= (z_k[all_s_idx, :] * beta_svc .* sig_svc) .* cov_vals
            end
        end

        # 4.5 Eigen & Basis Mapping
        if !isnothing(eigen_eff)
            for (idx, e_term) in enumerate(M.eigen_terms)
                v_pca = get_params_vector(chain, Symbol("v_pca_", idx), length(e_term.ltri_indices))[j, :]
                p_sd = get_params_vector(chain, Symbol("pca_sd_", idx), e_term.n_dims)[j, :]
                pdef = get_params_vector(chain, Symbol("pdef_sd_", idx), 1)[j]
                K_pca, _, _ = householder_transform(v_pca, e_term.n_dims, e_term.n_dims, e_term.ltri_indices, p_sd, pdef, M.noise)
                eigen_eff[idx][1:M.y_N, j] .= vec(e_term.data * K_pca)
            end
        end
        if !isnothing(basis_eff)
            for (v_sym, B) in M.basis_matrices
                beta_b = get_params_vector(chain, Symbol("beta_basis_", v_sym), size(B, 2))[j, :]
                basis_eff[v_sym][1:M.y_N, j] .= B * beta_b
            end
        end
    end

    # --- 5. Predictor Assembler (Corrected Full Summation) ---
    eta_noisy = zeros(N_tot, N_samples)
    eta_denoised = zeros(N_tot, N_samples)
    fixed_names = Symbol.(names(M.Xfixed, 2))

    for j in 1:N_samples
        for i in 1:N_tot
            is_obs = i <= M.y_N; idx = is_obs ? i : i - M.y_N; src = is_obs ? M : PS
            s_id, t_id = Int(src.s_idx[idx]), Int(src.t_idx[idx])
            val = is_obs ? get(M, :log_offset, zeros(M.y_N))[i] : 0.0

            # Fixed Part
            if !isnothing(Xfixed_betas)
                X_row = is_obs ? M.Xfixed[idx, :] : collect(Vector(PS.surface_df[idx, fixed_names]))
                val += dot(Xfixed_betas[:, j], X_row)
            end

            # Hierarchical & Categorical Manifolds
            for scale_sym in keys(hierarchical_fields); val += hierarchical_fields[scale_sym][M.spatial_hierarchy[scale_sym].indices[idx], j]; end
            for c_sym in keys(categorical_eff); val += categorical_eff[c_sym][src.c_groups[c_sym][idx], j]; end

            # Trend & Season
            val += t_eff[t_id, j]
            if M.model_season != "none"; val += u_eff[Int(src.u_idx[idx]), j]; end

            # Advanced Component Summation
            if !isnothing(mixed_eff); for e in mixed_eff; val += e[i, j]; end; end
            if !isnothing(interaction_eff); for e in interaction_eff; val += e[i, j]; end; end
            if !isnothing(eigen_eff); for e in eigen_eff; val += e[i, j]; end; end
            if !isnothing(svc_eff); for e in svc_eff; val += e[i, j]; end; end
            if !isnothing(basis_eff); for e in values(basis_eff); val += e[i, j]; end; end

            st_val = st_eff_map[s_id, t_id, j]
            eta_denoised[i, j] = val + s_eff_struct[s_id, j] + st_val
            eta_noisy[i, j] = val + s_eff_noisy[s_id, j] + st_val
        end
    end

    # --- 6. Family Transformation & Denoising Engine ---
    y_sigma = _extract_volatility(chain, p_names_str, N_tot, N_samples)
    preds_denoised, preds_noisy, log_lik = _process_ll_and_predictions(fam, eta_noisy, chain, M, N_tot, N_samples, y_sigma, M.y_obs)
    field_denoised, _, _ = _process_ll_and_predictions(fam, eta_denoised, chain, M, N_tot, N_samples, y_sigma, M.y_obs)

    # Post-stratification Weight logic
    weights = nothing
    if N_PS > 0
        s_idx_obs = M.s_idx; t_idx_obs = M.t_idx; u_idx_obs = get(M, :u_idx, ones(Int, M.y_N))
        s_N = M.s_N; t_N = M.t_N
        stratum_map = [(Int(u_idx_obs[i])-1)*s_N*t_N + (Int(t_idx_obs[i])-1)*s_N + Int(s_idx_obs[i]) for i in 1:M.y_N]
        weights = field_denoised[M.y_N .+ stratum_map, :] ./ (field_denoised[1:M.y_N, :] .+ 1e-9)
    end

    return (
        spatial_structured = summarize_array(reshape(s_eff_struct, M.s_N, 1, N_samples); alpha=alpha),
        spatial_unstructured = summarize_array(reshape(s_eff_noisy .- s_eff_struct, M.s_N, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(t_eff, M.t_N, 1, N_samples); alpha=alpha),
        seasonal = (M.model_season != "none") ? summarize_array(reshape(u_eff, get(M, :u_N, 1), 1, N_samples); alpha=alpha) : nothing,
        spatiotemporal = summarize_array(reshape(st_eff_map, M.s_N * M.t_N, 1, N_samples); alpha=alpha),
        volatility = summarize_array(reshape(y_sigma, N_tot, 1, N_samples); alpha=alpha),
        fixed_effects = !isnothing(Xfixed_betas) ? summarize_array(reshape(Xfixed_betas, M.Xfixed_N, 1, N_samples); alpha=alpha) : nothing,
        hierarchical_spatial = isempty(hierarchical_fields) ? nothing : Dict(k => summarize_array(reshape(v, size(v,1), 1, N_samples); alpha=alpha) for (k,v) in hierarchical_fields),
        categorical_effects = isempty(categorical_eff) ? nothing : Dict(k => summarize_array(reshape(v, size(v,1), 1, N_samples); alpha=alpha) for (k,v) in categorical_eff),
        mixed_effects = !isnothing(mixed_eff) ? [summarize_array(reshape(me, size(me,1), 1, N_samples); alpha=alpha) for me in mixed_eff] : nothing,
        interaction_effects = !isnothing(interaction_eff) ? [summarize_array(reshape(ie, size(ie,1), 1, N_samples); alpha=alpha) for ie in interaction_eff] : nothing,
        eigen_effects = !isnothing(eigen_eff) ? [summarize_array(reshape(ee, size(ee,1), 1, N_samples); alpha=alpha) for ee in eigen_eff] : nothing,
        svc_effects = !isnothing(svc_eff) ? [summarize_array(reshape(svc, size(svc,1), 1, N_samples); alpha=alpha) for svc in svc_eff] : nothing,
        basis_effects = isempty(basis_eff) ? nothing : Dict(k => summarize_array(reshape(v, size(v,1), 1, N_samples); alpha=alpha) for (k,v) in basis_eff),
        predictions_observed_denoised = summarize_array(reshape(preds_denoised[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_observed_noisy = summarize_array(reshape(preds_noisy[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_strata_denoised = N_PS > 0 ? summarize_array(reshape(field_denoised[M.y_N+1:end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        post_strat_weights = isnothing(weights) ? nothing : (mean=vec(mean(weights, dims=2)), samples=weights),
        waic = _compute_waic(log_lik[:, 1:M.y_N]),
        family = fam, arch = arch
    )
end



function _reconstruct(arch::MultivariateArchitecture, fam::ModelFamily, chain, M, PS, alpha)
    N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    N_tot = M.y_N + N_PS
    N_samples = size(chain, 1)
    p_names_str = string.(FlexiChains.parameters(chain))

    outcome_results = Vector{Any}(undef, M.outcomes_N)
    phi_zi = "phi_zi" in p_names_str ? summarize_array(reshape(get_params_vector(chain, "phi_zi", 1), 1, 1, N_samples); alpha=alpha) : nothing

    # Identify fixed-effects column names from the observation model
    fixed_names = names(M.Xfixed, 2)

    for k in 1:M.outcomes_N
        # 1. Parameter Extraction for outcome k
        s_sigmas = vec(get_params_vector(chain, Symbol("s_sigma[", k, "]"), 1))
        t_sigmas = vec(get_params_vector(chain, Symbol("t_sigma[", k, "]"), 1))
        u_sigmas = "u_sigma[", k, "]" in p_names_str ? vec(get_params_vector(chain, Symbol("u_sigma[", k, "]"), 1)) : zeros(N_samples)

        s_eff_struct = zeros(M.s_N, N_samples)
        s_eff_noisy = zeros(M.s_N, N_samples)
        t_eff = zeros(M.t_N, N_samples)
        u_eff = zeros(get(M, :u_N, 1), N_samples)

        # Features
        Xfixed_betas = (M.Xfixed_N > 0) ? zeros(M.Xfixed_N, N_samples) : nothing
        random_eff = Symbol("c_eta_k[", k, "]") in FlexiChains.parameters(chain) ? zeros(M.N_cat, N_samples) : nothing

        for j in 1:N_samples
            fs, fn = extract_manifold_k(SpatialTrait(), chain, M, j, k, s_sigmas[j])
            s_eff_struct[:, j] .= fs
            s_eff_noisy[:, j] .= fn

            t_key = Symbol("t_raw_k[", k, "]")
            if t_key in FlexiChains.parameters(chain)
                t_eff[:, j] .= vec(chain[t_key].data[j]) .* t_sigmas[j]
            end

            if M.model_season != "none"
                u_key = Symbol("u_raw_k[", k, "]")
                if u_key in FlexiChains.parameters(chain)
                    u_eff[:, j] .= vec(chain[u_key].data[j]) .* u_sigmas[j]
                end
            end

            if !isnothing(Xfixed_betas)
                Xfixed_betas[:, j] .= [chain[Symbol("d_beta_k[", k, "][", d, "]")].data[j] for d in 1:M.Xfixed_N]
            end

            if !isnothing(random_eff)
                random_eff[:, j] .= vec(chain[Symbol("c_eta_k[", k, "]")].data[j])
            end
        end

        # 2. Linear Predictor Construction
        eta_noisy = zeros(N_tot, N_samples)
        eta_denoised = zeros(N_tot, N_samples)

        for j in 1:N_samples, i in 1:N_tot
            is_obs = i <= M.y_N; idx = is_obs ? i : i - M.y_N; src = is_obs ? M : PS
            s_idx = Int(src.s_idx[idx]); t_idx = Int(src.t_idx[idx])

            base = (is_obs ? M.log_offset[i, k] : 0.0) + t_eff[t_idx, j]
            if !isnothing(Xfixed_betas)
                X_row = is_obs ? M.Xfixed[idx, :] : Matrix(PS.surface_df[idx, fixed_names])
                base += dot(Xfixed_betas[:, j], X_row)
            end
            if !isnothing(random_eff)
                c_level = Int(is_obs ? M.cov_indices[idx, 1] : PS.z_indices[idx, 1])
                base += random_eff[c_level, j]
            end

            eta_denoised[i, j] = base + s_eff_struct[s_idx, j]
            eta_noisy[i, j] = base + s_eff_noisy[s_idx, j]

            if M.model_season != "none"
                base += u_eff[Int(src.u_idx[idx]), j]
            end
        end

        y_sigma_k = _extract_volatility(chain, p_names_str, N_tot, N_samples, k)
        preds_denoised, preds_noisy, log_lik = _process_ll_and_predictions(fam, eta_noisy, chain, M, N_tot, N_samples, y_sigma_k, M.y_obs[:, k])
        field_denoised, _, _ = _process_ll_and_predictions(fam, eta_denoised, chain, M, N_tot, N_samples, y_sigma_k, M.y_obs[:, k])

        weights = nothing
        if N_PS > 0
            s_idx_obs = M.s_idx; t_idx_obs = M.t_idx; s_N = M.s_N; t_N = M.t_N
            stratum_map = [(Int(t_idx_obs[i])-1)*s_N + Int(s_idx_obs[i]) for i in 1:M.y_N]
            weights = field_denoised[M.y_N .+ stratum_map, :] ./ (field_denoised[1:M.y_N, :] .+ 1e-9)
        end

        outcome_results[k] = (
            spatial_structured = summarize_array(reshape(s_eff_struct, M.s_N, 1, N_samples); alpha=alpha),
            spatial_unstructured = summarize_array(reshape(s_eff_noisy .- s_eff_struct, M.s_N, 1, N_samples); alpha=alpha),
            temporal = summarize_array(reshape(t_eff, M.t_N, 1, N_samples); alpha=alpha),
            seasonal = M.model_season != "none" ? summarize_array(reshape(u_eff, get(M, :u_N, 1), 1, N_samples); alpha=alpha) : nothing,
            volatility = summarize_array(reshape(y_sigma_k, N_tot, 1, N_samples); alpha=alpha),
            fixed_effects = !isnothing(Xfixed_betas) ? summarize_array(reshape(Xfixed_betas, M.Xfixed_N, 1, N_samples); alpha=alpha) : nothing,
            random_effects = !isnothing(random_eff) ? summarize_array(reshape(random_eff, M.N_cat, 1, N_samples); alpha=alpha) : nothing,
            predictions_observed_denoised = summarize_array(reshape(preds_denoised[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
            predictions_observed_noisy = summarize_array(reshape(preds_noisy[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
            predictions_strata_denoised = N_PS > 0 ? summarize_array(reshape(preds_denoised[M.y_N+1:end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
            predictions_strata_noisy = N_PS > 0 ? summarize_array(reshape(preds_noisy[M.y_N+1:end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
            post_strat_weights = isnothing(weights) ? nothing : (mean=vec(mean(weights, dims=2)), samples=weights),
            waic = _compute_waic(log_lik[:, 1:M.y_N])
        )
    end
    return (outcomes = outcome_results, phi_zi = phi_zi, arch = arch)
end


function _reconstruct(arch::MultifidelityArchitecture, fam::ModelFamily, chain, M, PS, alpha)
    N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    N_tot = M.y_N + N_PS
    N_samples = size(chain, 1)
    p_names_str = string.(FlexiChains.parameters(chain))

    s_sigmas = "s_sigma" in p_names_str ? vec(get_params_vector(chain, "s_sigma", 1)) : ones(N_samples)
    t_sigmas = "t_sigma" in p_names_str ? vec(get_params_vector(chain, "t_sigma", 1)) : ones(N_samples)
    u_sigmas = "u_sigma" in p_names_str ? vec(get_params_vector(chain, "u_sigma", 1)) : zeros(N_samples)

    s_eff_struct = zeros(M.s_N, N_samples); s_eff_noisy = zeros(M.s_N, N_samples)
    t_eff = zeros(M.t_N, N_samples); u_eff = zeros(get(M, :u_N, 1), N_samples)
    z_latent = zeros(N_tot, N_samples)
    Xfixed_betas = (M.Xfixed_N > 0) ? zeros(M.Xfixed_N, N_samples) : nothing
    random_eff = "c_eta" in p_names_str ? zeros(M.N_cat, N_samples) : nothing

    # Identify fixed-effects column names from the observation model
    fixed_names = names(M.Xfixed, 2)

    for j in 1:N_samples
        fs, fn = extract_manifold(SpatialTrait(), chain, M, j, s_sigmas[j])
        s_eff_struct[:, j] .= fs; s_eff_noisy[:, j] .= fn
        t_eff[:, j] .= extract_manifold(TemporalTrait(), chain, M, j, t_sigmas[j])
        if M.model_season != "none"
            u_eff[:, j] .= extract_manifold(SeasonalTrait(), chain, M, j, u_sigmas[j])
        end
        if !isnothing(Xfixed_betas)
            Xfixed_betas[:, j] .= extract_manifold(FixedEffectTrait(), chain, M, j)
        end
        if !isnothing(random_eff)
            random_eff[:, j] .= extract_manifold(RandomEffectTrait(), chain, M, j)
        end
        z_latent[1:M.y_N, j] .= vec(chain[:z_latent].data[j])
    end

    eta_noisy = zeros(N_tot, N_samples); eta_denoised = zeros(N_tot, N_samples)
    z_beta = vec(get_params_vector(chain, "z_beta_eta", 1))

    for j in 1:N_samples, i in 1:N_tot
        is_obs = i <= M.y_N; idx = is_obs ? i : i - M.y_N; src = is_obs ? M : PS
        s_idx = Int(src.s_idx[idx]); t_idx = Int(src.t_idx[idx])

        base = (is_obs ? M.log_offset[i] : 0.0) + t_eff[t_idx, j] + z_latent[i, j] * z_beta[j]
        if !isnothing(Xfixed_betas)
            X_row = is_obs ? M.Xfixed[idx, :] : Matrix(PS.surface_df[idx, fixed_names])
            base += dot(Xfixed_betas[:, j], X_row)
        end
        if !isnothing(random_eff)
            c_level = Int(is_obs ? M.cov_indices[idx, 1] : PS.z_indices[idx, 1])
            base += random_eff[c_level, j]
        end

        eta_denoised[i, j] = base + s_eff_struct[s_idx, j]
        eta_noisy[i, j] = base + s_eff_noisy[s_idx, j]
        if M.model_season != "none"
            u_idx = Int(src.u_idx[idx])
            eta_denoised[i, j] += u_eff[u_idx, j]
            eta_noisy[i, j] += u_eff[u_idx, j]
        end
    end

    y_sigma = _extract_volatility(chain, p_names_str, N_tot, N_samples)
    preds_denoised, preds_noisy, log_lik = _process_ll_and_predictions(fam, eta_noisy, chain, M, N_tot, N_samples, y_sigma, M.y_obs)
    field_denoised, _, _ = _process_ll_and_predictions(fam, eta_denoised, chain, M, N_tot, N_samples, y_sigma, M.y_obs)

    weights = nothing
    if N_PS > 0
        s_idx_obs = M.s_idx; t_idx_obs = M.t_idx; s_N = M.s_N
        stratum_map = [(Int(t_idx_obs[i])-1)*s_N + Int(s_idx_obs[i]) for i in 1:M.y_N]
        weights = field_denoised[M.y_N .+ stratum_map, :] ./ (field_denoised[1:M.y_N, :] .+ 1e-9)
    end

    return (
        spatial_structured = summarize_array(reshape(s_eff_struct, M.s_N, 1, N_samples); alpha=alpha),
        spatial_unstructured = summarize_array(reshape(s_eff_noisy .- s_eff_struct, M.s_N, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(t_eff, M.t_N, 1, N_samples); alpha=alpha),
        seasonal = M.model_season != "none" ? summarize_array(reshape(u_eff, get(M, :u_N, 1), 1, N_samples); alpha=alpha) : nothing,
        volatility = summarize_array(reshape(y_sigma, N_tot, 1, N_samples); alpha=alpha),
        fixed_effects = !isnothing(Xfixed_betas) ? summarize_array(reshape(Xfixed_betas, M.Xfixed_N, 1, N_samples); alpha=alpha) : nothing,
        random_effects = !isnothing(random_eff) ? summarize_array(reshape(random_eff, M.N_cat, 1, N_samples); alpha=alpha) : nothing,
        predictions_observed_denoised = summarize_array(reshape(preds_denoised[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_observed_noisy = summarize_array(reshape(preds_noisy[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_strata_denoised = N_PS > 0 ? summarize_array(reshape(preds_denoised[M.y_N+1:end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        predictions_strata_noisy = N_PS > 0 ? summarize_array(reshape(preds_noisy[M.y_N+1:end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        post_strat_weights = isnothing(weights) ? nothing : (mean=vec(mean(weights, dims=2)), samples=weights),
        waic = _compute_waic(log_lik[:, 1:M.y_N]),
        family = fam, arch = arch
    )
end


function _reconstruct(arch::ExampleArchitecture, fam::ModelFamily, modelname, chain, M, PS, alpha)
    N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    N_tot = M.y_N + N_PS
    N_samples = size(chain, 1)
    p_names_str = string.(FlexiChains.parameters(chain))

    # 1. Model Type Detection using explicit modelname
    is_kriging   = modelname == "example_kriging_simple"
    is_hurdle    = modelname == "example_hurdle_bernoulli_poisson"
    is_rff_2d    = modelname == "example_rff_2D"
    is_rff_3d    = modelname == "example_rff_3D"
    is_warping   = modelname == "example_warping_2D"
    is_mosaic_2d = modelname == "example_rff_2D_mosaic"
    is_mosaic_3d = modelname == "example_rff_3D_mosaic"
    is_gp_gaussian = modelname == "example_gp_gaussian"

    # 2. Field Storage (aligned with UnivariateArchitecture)
    # f_latent_samples for the combined linear predictor before link function
    f_latent_samples = zeros(N_tot, N_samples)
    s_eff_struct = zeros(M.s_N, N_samples)
    s_eff_noisy = zeros(M.s_N, N_samples)
    t_eff = zeros(M.t_N, N_samples)
    u_eff = zeros(get(M, :u_N, 1), N_samples)
    st_eff_map = zeros(M.s_N, M.t_N, N_samples)

    Xfixed_betas = (M.Xfixed_N > 0) ? zeros(M.Xfixed_N, N_samples) : nothing
    random_eff = "c_eta" in p_names_str ? zeros(M.N_cat, N_samples) : nothing

    mixed_effects_samples = haskey(M, :mixed_terms) && !isempty(M.mixed_terms) ?
        [zeros(term.n_cat, N_samples) for term in M.mixed_terms] : nothing
    interaction_effects_samples = haskey(M, :interaction_terms) && !isempty(M.interaction_terms) ?
        [zeros(N_tot, N_samples) for _ in M.interaction_terms] : nothing
    eigen_effects_samples = haskey(M, :eigen_terms) && !isempty(M.eigen_terms) ?
        [zeros(term.n_dims, N_samples) for term in M.eigen_terms] : nothing

    # Precompute Coordinates for Mosaic/Warping/RFF methods
    # Adjust for PS data if available
    all_s_coords = N_PS > 0 ? vcat(M.s_coord_tuple, [Tuple(r) for r in eachrow(Matrix(PS.surface_df[:, [:x, :y]]))]) : M.s_coord_tuple
    xs = [p[1] for p in all_s_coords]; ys = [p[2] for p in all_s_coords]
    pts_2d = hcat(xs, ys)

    st_sigmas = "st_sigma" in p_names_str ? vec(get_params_vector(chain, "st_sigma", 1)) : zeros(N_samples)

    for j in 1:N_samples
        # Extract common parameters
        if !isnothing(Xfixed_betas); Xfixed_betas[:, j] .= get_params_vector(chain, "Xfixed_beta", M.Xfixed_N)[j, :]; end

        if is_hurdle # example_hurdle_bernoulli_poisson
            s_h = vec(get_params_vector(chain, "s_latent_h", M.s_N)[j, :])
            t_h = vec(get_params_vector(chain, "t_latent_h", M.t_N)[j, :])
            s_c = vec(get_params_vector(chain, "s_latent_c", M.s_N)[j, :])
            t_c = vec(get_params_vector(chain, "t_latent_c", M.t_N)[j, :])
            for i in 1:N_tot
                is_obs = i <= M.y_N; idx = is_obs ? i : i - M.y_N; src = is_obs ? M : PS
                a, t = Int(src.s_idx[idx]), Int(src.t_idx[idx])
                h_l = s_h[a] + t_h[t]
                c_l = (is_obs ? M.log_offset[i] : 0.0) + s_c[a] + t_c[t]
                f_latent_samples[i, j] = logistic(h_l) * (exp(c_l) / (1.0 - exp(-exp(c_l))))
            end
            s_eff_struct[:,j] .= s_h # use hurdle's spatial for structured spatial
            t_eff[:,j] .= t_h # use hurdle's temporal for temporal

        elseif is_mosaic_2d || is_mosaic_3d # example_rff_2D_mosaic / example_rff_3D_mosaic
            mu_loc = vec(get_params_vector(chain, "mu_local", M.n_mosaics)[j, :])
            sig_loc = vec(get_params_vector(chain, "sigma_local", M.n_mosaics)[j, :])
            # Recalculate centroids (Note: ideally these would be in M to avoid shifts)
            coords_for_clustering = is_mosaic_3d ? hcat((xs .- minimum(xs))./(maximum(xs)-minimum(xs)+1e-6), (ys .- minimum(ys))./(maximum(ys)-minimum(ys)+1e-6)) : pts_2d
            R = kmeans(coords_for_clustering[1:M.y_N, :]' |> Matrix, M.n_mosaics)
            centroids = R.centers

            for i in 1:N_tot
                pt_s_coord = coords_for_clustering[i, :]
                dists = [sum((pt_s_coord .- centroids[:, m]).^2) for m in 1:M.n_mosaics]
                w_st = exp.(-dists) ./ sum(exp.(-dists))
                val_combined = 0.0
                for m in 1:M.n_mosaics
                    s_l_m = vec(get_params_vector(chain, "s_latent[", m, "]", M.m_rff)[j, :])
                    Random.seed!(42) # Ensure consistent RFF projections
                    if is_mosaic_3d
                        l_j = vec(get_params_vector(chain, "l_joint", 3 * M.n_mosaics)[j, :])
                        l_m = l_j[(m-1)*3+1 : m*3]
                        Om = randn(M.m_rff, 3) .* (1.0 ./ (l_m' .+ 1e-6))
                        t_val = (i <= M.y_N ? Float64(M.t_idx[i]) : Float64(PS.t_idx[i-M.y_N]))
                        pt_3d = [pt_s_coord[1], pt_s_coord[2], (t_val - minimum(M.t_idx))/(maximum(M.t_idx)-minimum(M.t_idx)+1e-6)]
                        z_proj = sqrt(2/M.m_rff) * cos.( (Om * pt_3d) .+ (rand(M.m_rff) .* 2pi) )
                    else # is_mosaic_2d
                        l_m = get_params_vector(chain, "l_local", M.n_mosaics)[j, m]
                        Om = randn(M.m_rff, 2) ./ l_m
                        z_proj = sqrt(2/M.m_rff) .* cos.( (Om * pt_s_coord) .+ (rand(M.m_rff) .* 2pi) )
                    end
                    val_combined += w_st[m] * (mu_loc[m] + dot(z_proj, s_l_m .* sig_loc[m]))
                end
                f_latent_samples[i, j] = val_combined
            end
            # For mosaic, spatial_structured is the mean of local mu's, no simple single s_eff_struct
            # We can approximate with global mean of local fields
            s_eff_struct[:, j] .= mean(mu_loc)

        elseif is_warping # example_warping_2D
            s_l = vec(get_params_vector(chain, "s_latent", M.m_spatial)[j, :])
            w_w = vec(get_params_vector(chain, "w_warp", M.m_warp)[j, :])
            t_l = vec(get_params_vector(chain, "t_latent", M.t_N)[j, :]) .* chain[:t_sigma].data[j]
            s_sig = chain[:s_sigma].data[j]; l_w = chain[:l_warp].data[j]; l_s = chain[:l_spatial].data[j]
            coords_norm = hcat((xs .- mean(xs)) ./ (std(xs)+1e-6), (ys .- mean(ys)) ./ (std(ys)+1e-6))
            Random.seed!(44); Om_w = randn(M.m_warp, 2) ./ l_w; Ph_w = rand(M.m_warp) .* 2pi
            warped = (sqrt(2/M.m_warp) .* cos.(coords_norm * Om_w' .+ Ph_w')) * w_w
            Random.seed!(45); Om_s = randn(M.m_spatial, 1) ./ l_s; Ph_s = rand(M.m_spatial) .* 2pi
            Z_sp = sqrt(2/M.m_spatial) .* cos.(reshape(warped, :, 1) * Om_s' .+ Ph_s')
            f_latent_samples[:, j] .= (Z_sp * (s_l .* s_sig)) .+ t_l[N_PS > 0 ? vcat(M.t_idx, PS.t_idx) : M.t_idx]
            # For warping, s_eff_struct is the warped spatial field, and t_eff is the temporal
            s_eff_struct[:, j] .= (Z_sp * (s_l .* s_sig))[1:M.s_N] # Assuming first M.s_N map to original spatial units
            t_eff[:, j] .= t_l

        elseif is_rff_2d # example_rff_2D
            s_l = vec(get_params_vector(chain, "s_latent", M.m_spatial)[j, :])
            t_l_raw = vec(get_params_vector(chain, "t_raw", M.t_N)[j, :])
            t_sig = chain[:t_sigma].data[j]
            s_sig = chain[:s_sigma].data[j]
            kappa_sp = chain[:kappa_sp].data[j]

            coords_norm = hcat(
                (xs .- mean(xs)) ./ (std(xs) + 1e-6),
                (ys .- mean(ys)) ./ (std(ys) + 1e-6)
            )

            Random.seed!(42)
            Om = randn(M.m_spatial, 2) .* kappa_sp
            Ph = rand(M.m_spatial) .* convert(Float64, 2pi)

            Z_sp = convert(Float64, sqrt(2/M.m_spatial)) .* cos.(coords_norm * Om' .+ Ph')
            spatial_field = Z_sp * (s_l .* s_sig)
            temporal_field = t_l_raw .* t_sig

            f_latent_samples[:, j] .= spatial_field .+ temporal_field[N_PS > 0 ? vcat(M.t_idx, PS.t_idx) : M.t_idx]
            s_eff_struct[:, j] .= spatial_field[1:M.s_N]
            t_eff[:, j] .= temporal_field

        elseif is_rff_3d # example_rff_3D
            s_l = vec(get_params_vector(chain, "s_latent", M.m_joint)[j, :])
            sig_joint = chain[:sigma_joint].data[j]
            l_joint = vec(get_params_vector(chain, "l_joint", 3)[j, :])

            ts_all = N_PS > 0 ? vcat(M.t_idx, PS.t_idx) : M.t_idx

            X_joint = hcat(
                (xs .- minimum(xs)) ./ (maximum(xs) - minimum(xs) + 1e-6),
                (ys .- minimum(ys)) ./ (maximum(ys) - minimum(ys) + 1e-6),
                (Float64.(ts_all) .- minimum(M.t_idx)) ./ (maximum(M.t_idx) - minimum(M.t_idx) + 1e-6)
            )

            Random.seed!(42)
            Om = randn(M.m_joint, 3) .* (1.0 ./ l_joint')
            Ph = rand(M.m_joint) .* convert(Float64, 2pi)

            Z_joint = convert(Float64, sqrt(2/M.m_joint)) .* cos.(X_joint * Om' .+ Ph')
            eta_joint = Z_joint * (s_l .* sig_joint)

            f_latent_samples[:, j] .= eta_joint
            s_eff_struct[:, j] .= eta_joint[1:M.s_N] # Approximate spatial as first part of joint effect

        elseif is_kriging # example_kriging_simple
            # In Kriging, f_latent_all contains the entire spatial field
            f_latent_samples[1:M.y_N, j] .= vec(chain[Symbol("f_latent_all")].data[j])
            # No easy way to extract s_eff_struct, so use first part of f_latent_samples
            s_eff_struct[:, j] .= f_latent_samples[1:M.s_N, j]

        elseif is_gp_gaussian # example_gp_gaussian
            s_l = vec(get_params_vector(chain, "s_latent", M.M_inducing_s_N*M.M_inducing_t_N)[j, :]) # Number of inducing points
            st_sig = chain[:st_sigma].data[j]
            l_s = chain[:ls_s].data[j]
            l_t = chain[:ls_t].data[j]

            k = (st_sig^2) * (SqExponentialKernel(ScaleTransform(inv(l_s))) * SqExponentialKernel(ScaleTransform(inv(l_t))))
            f_gp = GP(k)

            Ns = size(M.s_inducing, 2) # Ns and Nt here refer to dimensions of inducing points
            Nt = size(M.t_inducing, 2)
            s_grid = repeat(M.s_inducing, outer=(1, Nt))
            t_grid = repeat(M.t_inducing, inner=(1, Ns))
            X_ind = ColVecs(vcat(s_grid, t_grid))

            f_cond = condition(f_gp(X_ind, 1e-6), s_l)

            s_obs_full = M.s_coord.X'
            t_obs_full = reshape(M.t_coord, 1, :)
            X_obs_all_pts = ColVecs(vcat(s_obs_full, t_obs_full)) # M.y_N observations
            
            # If PS exists, include its coordinates too
            if N_PS > 0
                PS_s_coord_mat = reduce(hcat, [collect(p) for p in PS.s_coord_tuple])'
                PS_t_coord_vec = PS.t_idx
                X_ps_all_pts = ColVecs(vcat(PS_s_coord_mat, reshape(PS_t_coord_vec, 1, :)))
                all_X_coords = ColVecs(hcat(X_obs_all_pts.X, X_ps_all_pts.X))
            else
                all_X_coords = X_obs_all_pts
            end
            
            f_latent_samples[:, j] .= mean(f_cond(all_X_coords[N_PS > 0 ? vcat(M.st_idx, PS.st_idx) : M.st_idx]))
            s_eff_struct[:, j] .= f_latent_samples[1:M.s_N, j] # Approximate spatial as first part

        else # Default / Fallback for other example models (e.g., bym2_ar1_poisson)
            s_sigmas = "s_sigma" in p_names_str ? vec(get_params_vector(chain, "s_sigma", 1)) : ones(N_samples)
            t_sigmas = "t_sigma" in p_names_str ? vec(get_params_vector(chain, "t_sigma", 1)) : ones(N_samples)
            u_sigmas = "u_sigma" in p_names_str ? vec(get_params_vector(chain, "u_sigma", 1)) : zeros(N_samples)

            fs, fn = extract_manifold(SpatialTrait(), chain, M, j, s_sigmas[j])
            s_eff_struct[:, j] .= fs
            s_eff_noisy[:, j] .= fn
            t_eff[:, j] .= extract_manifold(TemporalTrait(), chain, M, j, t_sigmas[j])
            if M.model_season != "none"
                u_eff[:, j] .= extract_manifold(SeasonalTrait(), chain, M, j, u_sigmas[j])
            end
            # Spatiotemporal / Transport Extraction (Scaling by st_sigma)
            if "st_latent" in p_names_str
                st_eff_map[:, :, j] .= reshape(vec(chain[:st_latent].data[j]), M.s_N, M.t_N) .* st_sigmas[j]
            end

            # Combine generic manifolds for f_latent_samples for the rest of the loop
            for i in 1:N_tot
                is_obs = i <= M.y_N; idx = is_obs ? i : i - M.y_N; src = is_obs ? M : PS
                s_idx = Int(src.s_idx[idx]); t_idx = Int(src.t_idx[idx])
                u_idx_val = M.model_season != "none" ? Int(src.u_idx[idx]) : 1
                f_latent_samples[i,j] = s_eff_struct[s_idx,j] + t_eff[t_idx,j] + u_eff[u_idx_val,j] + st_eff_map[s_idx, t_idx, j]
            end
        end

        # Extract new effects (mixed, interaction, eigen) - common to all architectures
        if !isnothing(mixed_effects_samples)
            for (idx, m_term) in enumerate(M.mixed_terms)
                mixed_effects_samples[idx][:, j] .= get_params_vector(chain, "beta_group_$(idx)", m_term.n_cat)[j, :]
            end
        end
        if !isnothing(interaction_effects_samples)
            for (idx, i_term) in enumerate(M.interaction_terms)
                sig_ie = get_params_vector(chain, "sig_ie_$(idx)", 1)[j]
                ls_ie = get_params_vector(chain, "ls_ie_$(idx)", 1)[j]
                beta_ie = get_params_vector(chain, "beta_ie_$(idx)", i_term.M_rff)[j, :]
                all_coords_ie = if N_PS > 0
                    # Assume interaction terms are defined on spatial coords similar to general prediction surface
                    vcat(i_term.coords, hcat(PS.surface_df.x, PS.surface_df.y)) # Need 'x', 'y' from PS.surface_df
                else
                    i_term.coords
                end
                projection = (all_coords_ie * i_term.W_ie) .+ i_term.b_ie'
                interaction_effects_samples[idx][:, j] .= sqrt(2.0/i_term.M_rff) .* cos.(projection) * beta_ie
            end
        end
        if !isnothing(eigen_effects_samples)
            for (idx, e_term) in enumerate(M.eigen_terms)
                v_pca = get_params_vector(chain, "v_pca_$(idx)", length(e_term.ltri_indices))[j, :]
                pca_sd = get_params_vector(chain, "pca_sd_$(idx)", e_term.n_dims)[j, :]
                pdef_sd = get_params_vector(chain, "pdef_sd_$(idx)", 1)[j]
                K_pca, _, _ = householder_transform(v_pca, e_term.n_dims, e_term.n_dims, e_term.ltri_indices, pca_sd, pdef_sd, M.noise)
                eigen_effects_samples[idx][:, j] .= vec(e_term.data * K_pca)
            end
        end
    end

    # 3. Predictor Reconstruction (Noisy vs Denoised) - General Assembly
    eta_noisy = zeros(N_tot, N_samples)
    eta_denoised = zeros(N_tot, N_samples)

    fixed_names = names(M.Xfixed, 2)

    for j in 1:N_samples, i in 1:N_tot
        is_obs = i <= M.y_N; idx = is_obs ? i : i - M.y_N; src = is_obs ? M : PS
        
        base_val = (is_obs ? get(M, :log_offset, zeros(M.y_N))[i] : 0.0)

        if !isnothing(Xfixed_betas)
            X_row = is_obs ? M.Xfixed[idx, :] : collect(Vector(PS.surface_df[idx, fixed_names]))
            base_val += dot(Xfixed_betas[:, j], X_row)
        end

        # Add other effects as in UnivariateArchitecture
        if haskey(M, :c_groups) && !isempty(M.c_groups)
            for c_sym in keys(M.c_groups)
                c_latent_name = Symbol("c_latent_$(c_sym)")
                if c_latent_name in FlexiChains.parameters(chain)
                    c_latent_samples = get_params_vector(chain, string(c_latent_name), size(M.c_re_templates[c_sym].matrix, 1))[j, :]
                    group_val = is_obs ? M.c_groups[c_sym][idx] : 1 # Placeholder for PS
                    base_val += c_latent_samples[group_val]
                end
            end
        end

        if !isnothing(mixed_effects_samples)
            for (midx, m_term) in enumerate(M.mixed_terms)
                beta_group_val = mixed_effects_samples[midx][m_term.indices[idx], j]
                base_val += beta_group_val * m_term.covariate_vals[idx]
            end
        end

        if !isnothing(interaction_effects_samples)
            for (iidx, _) in enumerate(M.interaction_terms)
                base_val += interaction_effects_samples[iidx][i, j]
            end
        end

        if !isnothing(eigen_effects_samples)
            for (eidx, e_term) in enumerate(M.eigen_terms)
                base_val += eigen_effects_samples[eidx][e_term.indices_for_y[idx], j] # Assuming indices_for_y maps to correct data row
            end
        end

        # Final field values from specific models are in f_latent_samples
        eta_denoised[i, j] = base_val + f_latent_samples[i, j]
        eta_noisy[i, j] = base_val + f_latent_samples[i, j] # Assuming for these models, noisy == denoised for the latent field
    end

    y_sigma = _extract_volatility(chain, p_names_str, N_tot, N_samples)
    preds_denoised, preds_noisy, log_lik = _process_ll_and_predictions(fam, eta_noisy, chain, M, N_tot, N_samples, y_sigma, M.y_obs)

    # Recalculate denoised field for output consistency with UnivariateArchitecture
    field_denoised_for_output = zeros(N_tot, N_samples)
    for j in 1:N_samples, i in 1:N_tot
        val = eta_denoised[i, j]
        field_denoised_for_output[i, j] = (fam isa PoissonFamily || fam isa NegativeBinomialFamily) ? exp(val) : (fam isa BinomialFamily ? logistic(val) : val)
    end

    # Post-stratification weights (if PS is active)
    weights = nothing
    if N_PS > 0
        s_idx_obs = M.s_idx; t_idx_obs = M.t_idx; u_idx_obs = get(M, :u_idx, ones(Int, M.y_N))
        s_N = M.s_N; t_N = M.t_N
        stratum_map = [(Int(u_idx_obs[i])-1)*s_N*t_N + (Int(t_idx_obs[i])-1)*s_N + Int(s_idx_obs[i]) for i in 1:M.y_N]
        weights = field_denoised_for_output[M.y_N .+ stratum_map, :] ./ (field_denoised_for_output[1:M.y_N, :] .+ 1e-9)
    end

    return (
        spatial_structured = summarize_array(reshape(s_eff_struct, M.s_N, 1, N_samples); alpha=alpha),
        spatial_unstructured = summarize_array(reshape(s_eff_noisy .- s_eff_struct, M.s_N, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(t_eff, M.t_N, 1, N_samples); alpha=alpha),
        seasonal = M.model_season != "none" ? summarize_array(reshape(u_eff, get(M, :u_N, 1), 1, N_samples); alpha=alpha) : nothing,
        spatiotemporal = summarize_array(reshape(st_eff_map, M.s_N * M.t_N, 1, N_samples); alpha=alpha),
        volatility = summarize_array(reshape(y_sigma, N_tot, 1, N_samples); alpha=alpha),
        fixed_effects = !isnothing(Xfixed_betas) ? summarize_array(reshape(Xfixed_betas, M.Xfixed_N, 1, N_samples); alpha=alpha) : nothing,
        random_effects = !isnothing(random_eff) ? summarize_array(reshape(random_eff, M.N_cat, 1, N_samples); alpha=alpha) : nothing,
        mixed_effects = !isnothing(mixed_effects_samples) ? [summarize_array(reshape(me, size(me,1), 1, N_samples); alpha=alpha) for me in mixed_effects_samples] : nothing,
        interaction_effects = !isnothing(interaction_effects_samples) ? [summarize_array(reshape(ie, size(ie,1), 1, N_samples); alpha=alpha) for ie in interaction_effects_samples] : nothing,
        eigen_effects = !isnothing(eigen_effects_samples) ? [summarize_array(reshape(ee, size(ee,1), 1, N_samples); alpha=alpha) for ee in eigen_effects_samples] : nothing,
        predictions_observed_denoised = summarize_array(reshape(preds_denoised[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_observed_noisy = summarize_array(reshape(preds_noisy[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_strata_denoised = N_PS > 0 ? summarize_array(reshape(field_denoised_for_output[M.y_N+1:end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        predictions_strata_noisy = N_PS > 0 ? summarize_array(reshape(preds_noisy[M.y_N+1:end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        post_strat_weights = isnothing(weights) ? nothing : (mean=vec(mean(weights, dims=2)), samples=weights),
        waic = _compute_waic(log_lik[:, 1:M.y_N]),
        family = fam, arch = arch
    )
end



# Helper for outcome-specific manifold extraction in MultivariateArchitectures
function extract_manifold_k(::SpatialTrait, chain, M, j, k, s_sigma)
    s_N = M.s_N
    field_struct = zeros(s_N)
    field_noisy = zeros(s_N)
    p_syms = FlexiChains.parameters(chain)
    
    # Check for outcome-specific spatial keys
    spatial_key = Symbol("s_icar_k[$k]") in p_syms ? Symbol("s_icar_k[$k]") : (Symbol("s_latent_k[$k]") in p_syms ? Symbol("s_latent_k[$k]") : nothing)

    if !isnothing(spatial_key)
        icar = vec(chain[spatial_key].data[j])
        if M.model_space == "bym2" && Symbol("s_rho_k[$k]") in p_syms
            rho = clamp(chain[Symbol("s_rho_k[$k]")].data[j], 0.0, 1.0)
            iid = vec(chain[Symbol("s_iid_k[$k]")].data[j])
            field_struct = s_sigma .* sqrt(rho) .* icar
            field_noisy = field_struct .+ (s_sigma .* sqrt(1.0 - rho) .* iid)
        else
            field_struct = icar .* s_sigma
            field_noisy = field_struct
        end
    end
    return field_struct, field_noisy
end



function create_mixed_indices(term::String, data::DataFrame)
    # Parses terms like 'me(1 | group)' or 'me(x | group)'
    m = match(r"me\((.*)\|(.*)\)", term)
    isnothing(m) && return nothing
    
    lhs = strip(m.captures[1])
    group_var = Symbol(strip(m.captures[2]))
    
    group_data = data[!, group_var]
    levels = unique(group_data)
    group_map = Dict(v => i for (i, v) in enumerate(levels))
    indices = [group_map[v] for v in group_data]
    
    return (indices = indices, n_cat = length(levels), name = group_var)
end


function create_fixed_design(formula_rhs::AbstractString, data::Union{DataFrame, NamedArray}; contrasts=Dict{Symbol, Any}())
    df_tmp = data isa NamedArray ? DataFrame(data, :auto) : data
    if data isa NamedArray
        rename!(df_tmp, names(data, 2))
    end

    clean_rhs = replace(formula_rhs, r"(re|st|cv|int|me|fe)\\(.*?\\)" => "")
    clean_rhs = strip(replace(clean_rhs, r"\\+\\s*\\+" => "+"))
    clean_rhs = replace(clean_rhs, r"^\\+|\\+$" => "")

    if isempty(strip(clean_rhs)) || clean_rhs == "1"
        return NamedArray(ones(size(df_tmp, 1), 1), (1:size(df_tmp, 1), [:Intercept]))
    end

    try
        f = StatsModels.FormulaTerm(StatsModels.Term(:y), StatsModels.Term(Symbol(clean_rhs)))
        sch = StatsModels.schema(f, df_tmp, contrasts)
        f_applied = StatsModels.apply_schema(f, sch, StatsModels.RegressionModel)
        _, mm = StatsModels.modelcols(f_applied, df_tmp)
        return NamedArray(mm, (1:size(mm, 1), Symbol.(names(mm))))
    catch e
        return NamedArray(ones(size(df_tmp, 1), 1), (1:size(df_tmp, 1), [:Intercept]))
    end
end





;;


