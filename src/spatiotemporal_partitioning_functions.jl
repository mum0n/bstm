# spatial processing/partitioning support


function expand_hull(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, buffer_dist)
    """
    Synopsis: Computes the convex hull of points and expands it by a buffer distance.
    Inputs:
    - s_x: Vector of x-coordinates.
    - s_y: Vector of y-coordinates.
    - buffer_dist: Distance to buffer the convex hull.
    Outputs:
    - A LibGEOS Polygon geometry representing the buffered convex hull.
    """

    s_coord_tuple_local = tuple.(s_x, s_y)

    if isempty(s_coord_tuple_local) return LibGEOS.Polygon([[ (0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0) ]]) end
    coords_vec = [[Float64(p[1]), Float64(p[2])] for p in s_coord_tuple_local]
    points_geom = LibGEOS.MultiPoint(coords_vec)
    hull = LibGEOS.convexhull(points_geom)
    buffered_hull = LibGEOS.buffer(hull, buffer_dist)
    return buffered_hull
end

 

function get_kde_seeds(s_coord_tuple_local, target_u)
 
    # Basic KDE-based seeding using StatsBase weights based on local density
    u_pts = unique(s_coord_tuple_local)
    if isempty(u_pts) return [] end
    n = length(u_pts)
    dists = [sum((p1 .- p2).^2) for p1 in u_pts, p2 in u_pts]
    # Inverse of mean distance as a density proxy
    weights = 1.0 ./ (mean(dists, dims=2)[:] .+ 1e-6)
    idx = StatsBase.sample(1:n, Weights(weights), min(target_u, n), replace=false)
    return u_pts[idx]
end

 
function is_valid_polygon_coords(poly_coords)
    # Filters out NaN/Inf values and checks for a minimum of 3 valid points for a polygon.

    valid_pts = [p for p in poly_coords if !isnan(p[1]) && !isinf(p[1]) && !isnan(p[2]) && !isinf(p[2])]
    return length(valid_pts) >= 3
end
 

function get_cvt_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom)
    """
    Synopsis: Centroidal Voronoi Tessellation (CVT) with diagnostic termination tracking.
    """

    s_coord_tuple_local = tuple.(s_x, s_y)

    if length(s_coord_tuple_local) <= cfg.min_total_arealunits
        return [ (mean(s_x), mean(s_y)) ], "not_enough_points_to_tessellate"
    end

    u_pts = unique(s_coord_tuple_local)
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
            area = get_polygon_area(poly_coords) # Using refactored area func

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

        # Use original s_coord_tuple_local for assignment to clusters
        assigns = [argmin([sum((p .- c).^2) for c in new_centroids]) for p in s_coord_tuple_local]
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


function get_kvt_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom)
    """
    Synopsis: K-means Voronoi Tessellation (KVT) with diagnostic termination tracking.
    """

    s_coord_tuple_local = tuple.(s_x, s_y)


    if length(s_coord_tuple_local) <= cfg.min_total_arealunits
        return [ (mean(s_x), mean(s_y)) ], "not_enough_points_to_tessellate"
    end

    u_pts = unique(s_coord_tuple_local)
    idx_init = StatsBase.sample(1:length(u_pts), min(cfg.target, length(u_pts)), replace=false)
    c_iter = [u_pts[i] for i in idx_init]
    data = tuple.(s_coord_tuple_local, cfg.t_idx)

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



function get_qvt_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom)
    # Local helper to convert flat s_x, s_y to a vector of tuples

    s_coord_tuple_local = tuple.(s_x, s_y)

    if length(s_coord_tuple_local) <= cfg.min_total_arealunits
        return [ (mean(s_x), mean(s_y)) ], "not_enough_points_to_tessellate"
    end
 
    data = tuple.(s_coord_tuple_local, cfg.t_idx)

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

        xs_r = [p[1][1] for p in target_region]; ys_r = [p[1][2] for p in target_region]

        # Robust splitting: handle datasets with zero variance in one or more dimensions
        if length(unique(xs_r)) > 1 || length(unique(ys_r)) > 1
            mx = length(unique(xs_r)) > 1 ? median(xs_r) : xs_r[1]
            my = length(unique(ys_r)) > 1 ? median(ys_r) : ys_r[1]
            r_splits = [
                filter(p -> p[1][1] <= mx && p[1][2] <= my, target_region),
                filter(p -> p[1][1] > mx && p[1][2] <= my, target_region),
                filter(p -> p[1][1] <= mx && p[1][2] > my, target_region),
                filter(p -> p[1][1] > mx && p[1][2] > my, target_region)
            ]
        else
            # All points collocated spatially: split by index to progress toward target unit count
            mid = length(target_region) ÷ 2  # Corrected from ∈ 2 to ÷ 2
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



function get_bvt_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom)
    """
    Synopsis: Binary Voronoi Tessellation (BVT) with corrected recursive splitting logic.
    """

    s_coord_tuple_local = tuple.(s_x, s_y)

    if length(s_coord_tuple_local) <= cfg.min_total_arealunits
        return [ (mean(s_x), mean(s_y)) ], "not_enough_points_to_tessellate"
    end

    data = tuple.(s_coord_tuple_local, cfg.t_idx)
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

 
 
function get_hvt_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom; max_iter=500)

    s_coord_tuple_local = tuple.(s_x, s_y)

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
    # Convert s_coord_tuple_local to matrix for Clustering.jl
    pts_matrix = hcat([p[1] for p in s_coord_tuple_local], [p[2] for p in s_coord_tuple_local])'
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
        s_idx = [argmin([dist(p, c) for c in curr_centroids]) for p in s_coord_tuple_local]
        counts = [count(==(k), s_idx) for k in 1:length(curr_centroids)]

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
        new_centroids = refine(s_coord_tuple_local, curr_centroids, 3)

        # Check for centroid position stabilization
        if all(dist(new_centroids[j], curr_centroids[j]) < cfg.tolerance for j in 1:length(curr_centroids))
             # If positions stabilized but constraints aren't met, check if add a unit
             if length(curr_centroids) < cfg.max_total_arealunits && (length(curr_centroids) < cfg.target || any(counts .> cfg.max_points))
                 # Split the largest group to improve density balance
                 idx_to_split = argmax(counts)
                 group_pts = s_coord_tuple_local[s_idx .== idx_to_split]
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





"""
    get_avt_centroids(s_x, s_y, cfg, hull_geom)

Iterative Adaptive Voronoi Tessellation. Merges units that violate constraints
on point counts, time-slice representation, or geometric area.
"""
function get_avt_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom)
    s_coord_tuple = tuple.(s_x, s_y)
    
    if length(s_coord_tuple) <= cfg.min_total_arealunits
        return [ (mean(p[1] for p in s_coord_tuple), mean(p[2] for p in s_coord_tuple)) ], "not_enough_points_to_tessellate"
    end

    u_pts = unique(s_coord_tuple)
    # Seeding centroids via KDE-based logic
    c_init = get_kde_seeds(u_pts, min(length(u_pts), cfg.max_total_arealunits))
    data = tuple.(s_coord_tuple, cfg.t_idx)
    curr_c = [SVector{2, Float64}(c) for c in c_init]

    termination_reason = "min_units_reached"
    last_mean_density = 0.0
    last_cv = 0.0

    while length(curr_c) > cfg.min_total_arealunits
        # 1. Assignment
        assigns = [Int[] for _ in 1:length(curr_c)]
        for i in 1:length(data)
            d_pt = data[i][1]
            # Finding closest centroid (Explicit loop for clarity)
            dist_idx = argmin([sum((d_pt .- c).^2) for c in curr_c])
            push!(assigns[dist_idx], i)
        end
        
        counts = length.(assigns)
        
        # 2. Geometry Calculation
        # get_voronoi_polygons_and_edges returns Vector{Vector{Tuple{Float64, Float64}}}
        polys_coords, _ = get_voronoi_polygons_and_edges([Tuple(c) for c in curr_c], hull_geom)
        
        areas = fill(0.0, length(curr_c))
        for i in 1:min(length(curr_c), length(polys_coords))
            # Fixed Call: Passes the Vector of Tuples directly to the new method
            areas[i] = get_polygon_area(polys_coords[i])
        end

        # 3. Violation Audit
        violators = Int[]
        for k in 1:length(curr_c)
            ts_count = length(unique([data[idx][2] for idx in assigns[k]]))
            
            # Logic for merging: too few points, too few time slices, or area outside bounds
            is_invalid_count = counts[k] < cfg.min_points
            is_invalid_time = ts_count < cfg.min_time_slices
            is_invalid_area = (areas[k] > 0 && areas[k] < cfg.min_area) || (areas[k] > cfg.max_area)
            
            if is_invalid_count || is_invalid_time || is_invalid_area
                push!(violators, k)
            end
        end

        # 4. Convergence Check
        curr_mean_density = mean(counts)
        cv_val = std(counts) / (mean(counts) + 1e-9)
        
        if last_mean_density > 0.0 && (abs(curr_mean_density - last_mean_density) < cfg.tolerance || abs(cv_val - last_cv) < cfg.tolerance)
            termination_reason = "tolerance_reached"
            break
        end
        
        last_mean_density = curr_mean_density
        last_cv = cv_val

        # 5. Merging Step
        # Identify target unit to merge (the one with the lowest count among violators or overall)
        candidates_indices = isempty(violators) ? collect(1:length(curr_c)) : violators
        v_counts = [counts[k] for k in candidates_indices]
        target_idx = candidates_indices[argmin(v_counts)]

        # Find nearest neighbor for the target centroid
        dists = [sum((curr_c[target_idx] .- curr_c[j]).^2) for j in 1:length(curr_c)]
        dists[target_idx] = Inf
        neighbor_idx = argmin(dists)

        # Weighted update for the merged centroid location
        total_n = counts[target_idx] + counts[neighbor_idx]
        curr_c[neighbor_idx] = (curr_c[target_idx] .* counts[target_idx] .+ curr_c[neighbor_idx] .* counts[neighbor_idx]) ./ (total_n + 1e-9)
        
        # Explicit removal (No clamp used)
        deleteat!(curr_c, target_idx)
    end
    
    return [Tuple(c) for c in curr_c], termination_reason
end


function get_lattice_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, lengthscale)
    """
    Synopsis: Generates centroids for a regular 2D lattice (grid) based on a lengthscale.
    """

    s_coord_tuple = tuple.(s_x, s_y)

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


function assign_spatial_units(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}; area_method=:avt, target_units=10, lengthscale=nothing, input_polygons=nothing, geom_hull=nothing, kwargs...)
    # s_coord_tuple_local will be used for calculations that still expect a collection of points

    s_coord_tuple_local = tuple.(s_x, s_y) # Using the globally defined helper

    # The branch for `input_data isa AbstractMatrix` is removed, as this refactored function
    # is specifically for coordinate-based spatial unit assignment. `bstm_options`
    # or a similar function will call `assign_spatial_units_inferred` (or its refactored version)
    # directly if an adjacency matrix is provided as primary input.

    # 1. Handle User-Defined Polygons
    if !isnothing(input_polygons)
        # If geom_hull is provided, intersect the input polygons with it
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

        # Use s_coord_tuple_local for assignments, as it represents the original observation points
        new_assigns = [argmin([sum((p .- sj).^2) for sj in final_centroids]) for p in s_coord_tuple_local]
        v_edges = []

    # 2. Handle Lattice Method
    elseif area_method == :lattice
        # `expand_hull` and `get_lattice_centroids` will be refactored to take s_x, s_y
        ls = isnothing(lengthscale) ? sqrt(get_polygon_area(get_coords_from_geom(expand_hull( 0.0))) / target_units) : lengthscale # Updated call
        final_centroids_raw, rows, cols, bbox = get_lattice_centroids(s_x, s_y, ls) # Updated call
        reason = :lattice_grid

        # Generate square polygons and clip them if geom_hull is provided
        polys_coords = Vector{Vector{Tuple{Float64, Float64}}}()
        lg_polys = LibGEOS.Polygon[]
        final_centroids = Tuple{Float64, Float64}[]
        half = ls / 2.0

        for c in final_centroids_raw
            coords = [[(c[1]-half, c[2]-half), (c[1]+half, c[2]-half), (c[1]+half, c[2]+half), (c[1]-half, c[2]+half), (c[1]-half, c[2]+half)]]
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

        # Use s_coord_tuple_local for assignments
        new_assigns = [argmin([sum((p .- sj).^2) for sj in final_centroids]) for p in s_coord_tuple_local]
        v_edges = []
        hull_coords = isnothing(geom_hull) ? [(bbox[1], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[4]), (bbox[1], bbox[4]), (bbox[1], bbox[3])] : get_coords_from_geom(geom_hull)

    # 3. Standard Tessellation Methods
    else
        cfg = (
            target=Int(target_units),
            min_total_arealunits=Int(get(kwargs, :min_total_arealunits, 3)),
            max_total_arealunits=Int(get(kwargs, :max_total_arealunits, target_units*2)),
            min_time_slices=Int(get(kwargs, :min_time_slices, 1)),
            min_points=Int(get(kwargs, :min_points, 1)),
            max_points=Int(get(kwargs, :max_points, length(s_x))), # Use length of s_x
            min_area=get(kwargs, :min_area, 0.0),
            max_area=get(kwargs, :max_area, Inf),
            target_cv=get(kwargs, :target_cv, 1.0),
            tolerance=get(kwargs, :tolerance, 0.1),
            buffer_dist=get(kwargs, :buffer_dist, 0.5),
            t_idx=get(kwargs, :t_idx, ones(Int, length(s_x)))) # Use length of s_x

        # `expand_hull` will be refactored to take s_x, s_y
        hull_geom = !isnothing(geom_hull) ? geom_hull : expand_hull(s_x, s_y, cfg.buffer_dist) # Updated call

        # Centroid functions will be refactored to take s_x, s_y
        c_mid, reason = if area_method == :cvt get_cvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :kvt get_kvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :qvt get_qvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :bvt get_bvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :hvt get_hvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :avt get_avt_centroids(s_x, s_y, cfg, hull_geom) # Updated call
        else error("Unknown partitioning method: $area_method") end

        polys_coords, v_edges = get_voronoi_polygons_and_edges(c_mid, hull_geom)
        final_centroids = Tuple{Float64, Float64}[]
        lg_polys = []
        for p_coords in polys_coords
            if isempty(p_coords); continue; end
            # Ensure polygon is closed for LibGEOS if it's not already
            if p_coords[1] != p_coords[end]; push!(p_coords, p_coords[1]); end
            lg_p = LibGEOS.Polygon([[ [pt[1], pt[2]] for pt in p_coords ]])
            push!(lg_polys, lg_p)
            cent_g = LibGEOS.centroid(lg_p)
            seq = LibGEOS.getCoordSeq(cent_g)
            push!(final_centroids, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
        end

        # Use s_coord_tuple_local for assignments
        new_assigns = [argmin([sum((p .- sj).^2) for sj in final_centroids]) for p in s_coord_tuple_local]
        n_units = length(final_centroids)
        g = SimpleGraph(n_units)
        for i in 1:n_units, j in (i+1):n_units
            if LibGEOS.touches(lg_polys[i], lg_polys[j]) || LibGEOS.intersects(LibGEOS.buffer(lg_polys[i], 1e-7), lg_polys[j])
                add_edge!(g, i, j)
            end
        end
        g = ensure_connected!(g, final_centroids)
        hull_coords = get_coords_from_geom(hull_geom)
        W = Float64.(Graphs.adjacency_matrix(g))
    end

    # Update the returned NamedTuple to store s_x and s_y
    return (centroids=final_centroids, polygons=polys_coords,
            adjacency_edges=v_edges, graph=g, W=W, hull_coords=hull_coords,
            s_idx=new_assigns, s_x=s_x, s_y=s_y, s_vals=collect(1:size(W,1)), termination_reason=reason)
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
        fx = getindex.(forced_layout_centroids, 1)
        fy = getindex.(forced_layout_centroids, 2)
        hull_geom = expand_hull(fx, fy, buffer_dist)
        hull_coords_output = get_coords_from_geom(hull_geom)

        # 4. Use tessellation to determine polygon coordinates and initial adjacency (based on forced_layout_centroids)
        polys_coords_raw, _ = get_voronoi_polygons_and_edges(forced_layout_centroids, hull_geom)

        # 5. RECOMPUTE CENTROIDS from the generated (clipped) polygons and prepare for adjacency
        final_centroids = Vector{Tuple{Float64, Float64}}(undef, length(polys_coords_raw))
        lg_polygons_for_adjacency = Vector{Union{LibGEOS.Polygon, Nothing}}(undef, length(polys_coords_raw))
        polys_output = polys_coords_raw

        for (idx, poly_coord_list) in enumerate(polys_coords_raw)
            if !isempty(poly_coord_list) && length(poly_coord_list) >= 3
                if poly_coord_list[1] != poly_coord_list[end]
                    push!(poly_coord_list, poly_coord_list[1])
                end
                lg_poly = LibGEOS.Polygon([ [Float64[p[1], p[2]] for p in poly_coord_list] ])
                centroid_geom = LibGEOS.centroid(lg_poly)
                seq = LibGEOS.getCoordSeq(centroid_geom)
                final_centroids[idx] = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
                lg_polygons_for_adjacency[idx] = lg_poly
            else
                @warn "Invalid or empty polygon encountered in Voronoi tessellation at index $idx. Using original centroid as fallback."
                final_centroids[idx] = forced_layout_centroids[idx]
                lg_polygons_for_adjacency[idx] = nothing
            end
        end

        # 6. Re-build adjacency based on the newly derived centroids and polygons
        adjacency_edges_output = []
        if !isempty(lg_polygons_for_adjacency)
            for i in 1:length(lg_polygons_for_adjacency)
                g1 = lg_polygons_for_adjacency[i]
                if g1 === nothing continue end
                for j in (i+1):length(lg_polygons_for_adjacency)
                    g2 = lg_polygons_for_adjacency[j]
                    if g2 === nothing continue end
                    if LibGEOS.touches(g1, g2)
                        push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                    else
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
        end

        # 7. Build final graph from the re-derived adjacency edges and ensure connectivity
        g_final = SimpleGraph(nAU)
        centroid_map = Dict(c => i for (i, c) in enumerate(final_centroids))
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
        polygons = polys_output,
        adjacency_edges = adjacency_edges_output,
        graph = g_final,
        W = adjacency_matrix,
        hull_coords = hull_coords_output,
        s_idx = collect(1:nAU ),
        s_x = [c[1] for c in final_centroids[1:nAU]],
        s_y = [c[2] for c in final_centroids[1:nAU]],
        s_vals = collect(1:nAU ),
        termination_reason = "positions inferred from adjacency matrix"
    )
end 

 
# Verbatim copy and refinement of get_polygon_area and get_avt_centroids
# This version fixes the MethodError by providing explicit dispatch for tuple vectors.

"""
    get_polygon_area(poly_coords::AbstractVector)

Calculates the area of a polygon defined by a vector of (x, y) tuples using the Shoelace formula.
Includes data cleaning for NaNs, Infs, and duplicate vertices.
"""
function get_polygon_area(poly_coords::AbstractVector)
    # Filter invalid points
    valid_pts = [p for p in poly_coords if !isnan(p[1]) && !isinf(p[1]) && !isnan(p[2]) && !isinf(p[2])]
    
    # Remove trailing duplicate if it matches the start
    if length(valid_pts) > 1 && valid_pts[1] == valid_pts[end]
        pop!(valid_pts)
    end
    
    # A polygon must have at least 3 vertices
    if length(valid_pts) < 3 
        return 0.0 
    end
    
    x = [p[1] for p in valid_pts]
    y = [p[2] for p in valid_pts]
    
    # Shoelace Formula using LinearAlgebra utilities
    return 0.5 * abs(dot(x, circshift(y, 1)) - dot(y, circshift(x, 1)))
end

function get_polygon_area(s_x, s_y)
    # Wrapper for legacy three-argument calls
    poly_coords = tuple.(s_x, s_y)
    return get_polygon_area(poly_coords)
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



function ensure_connected!(W::AbstractMatrix, centroids::Vector{<:Tuple{Real, Real}})
    # Convert the adjacency matrix to a Graph object for component analysis
    g = Graph(W)
    comps = connected_components(g)
    
    # If the graph is already connected, no further action is required
    if length(comps) <= 1
        return W
    end

    # Number of components to bridge
    n_comps = length(comps)
    
    # 1. Calculate Centroids for each component to reduce search space
    # This simplifies the problem from comparing all nodes to comparing component centers
    comp_centroids = Vector{Vector{Float64}}(undef, n_comps)
    for i in 1:n_comps
        pts = [ [centroids[node][1], centroids[node][2]] for node in comps[i] ]
        comp_centroids[i] = mean(pts)
    end

    # 2. Build a KD-Tree of the component centroids for efficient spatial lookup
    # We treat each component as a single point in this high-level search
    centroid_matrix = hcat(comp_centroids...)
    tree = KDTree(centroid_matrix)

    # 3. Iteratively bridge components using the nearest-neighbor logic
    # We connect the most isolated components first to ensure global connectivity
    for i in 1:n_comps
        # Find the nearest component other than itself
        # knn returns indices and distances; we look for the 2 nearest (itself + neighbor)
        idxs, dists = knn(tree, comp_centroids[i], 2)
        target_comp_idx = idxs[2]
        
        # Now find the absolute closest node pair between the two identified components
        # While this sub-step is O(Ni * Nj), the components are typically small sub-graphs
        min_dist = Inf
        best_pair = (0, 0)
        
        for u in comps[i]
            for v in comps[target_comp_idx]
                d = sqdist([centroids[u][1], centroids[u][2]], [centroids[v][1], centroids[v][2]])
                if d < min_dist
                    min_dist = d
                    best_pair = (u, v)
                end
            end
        end
        
        # 4. Update the sparse adjacency matrix with the new bridging edge
        u_node, v_node = best_pair
        if u_node != 0 && v_node != 0
            W[u_node, v_node] = true
            W[v_node, u_node] = true
        end
    end

    # Verify final connectivity to ensure no orphans remain
    final_g = Graph(W)
    if length(connected_components(final_g)) > 1
        # Recursive fallback for complex edge cases (e.g., non-convex boundaries)
        return ensure_connected!(W, centroids)
    end

    return W
end

function sqdist(p1, p2)
    return (p1[1]-p2[1])^2 + (p1[2]-p2[2])^2
end


function plot_spatial_graph(; au=nothing, pts=nothing, plot_title="Spatial Partitioning")
 
    # 2. Base Plot Initialization
    plt = Plots.plot(aspect_ratio=:equal, legend=false)
    Plots.title!(plt, plot_title)

    # 3. Polygon Geometry Rendering
        for poly_coords in au[:polygons]
            if length(poly_coords) > 2
                px = [p[1] for p in poly_coords if !isnan(p[1])]
                py = [p[2] for p in poly_coords if !isnan(p[2])]
                if !isempty(px) && (px[1], py[1]) != (px[end], py[end])
                    push!(px, px[1])
                    push!(py, py[1])
                end
                Plots.plot!(plt, px, py, seriestype=:shape, fillalpha=0.1, linecolor=:black, lw=0.5)
            end
        end 

    # 4. Adjacency Graph Edge Rendering 
        for edge in Graphs.edges(au[:graph])
            u, v = Graphs.src(edge), Graphs.dst(edge)
            p1, p2 = au[:centroids][u], au[:centroids][v]
            Plots.plot!(plt, [p1[1], p2[1]], [p1[2], p2[2]], color=:red, lw=1.5, alpha=0.6)
        end 

    # 5. Scatter Plotting: Data Points and Polygon Centroids
    if !isnothing(pts)
        Plots.scatter!(plt, [p[1] for p in pts], [p[2] for p in pts],
            markersize=1, color=:gray, alpha=0.3, label="Points")
    end
 
        Plots.scatter!(plt, [c[1] for c in au[:centroids]], [c[2] for c in au[:centroids]],
            markersize=4, color=:blue, markerstrokecolor=:white, label="Centroids")

    # 6. Boundary Constraints
        bx = [p[1] for p in au[:hull_coords] if !isnan(p[1])]
        by = [p[2] for p in au[:hull_coords] if !isnan(p[2])]
        Plots.plot!(plt, bx, by, color=:black, lw=2, ls=:dash)

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




   
    
function assign_time_units(t_v; time_method="regular", t_N=nothing, u_N=12, kwargs...)

    if time_method=="regular"

        tint = Int.(floor.(t_v))
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
        
        u_v = t_v - tint

        u_disc = discretize_data( u_v, N_cat=u_N, method="regular" )  # seasonality discretized

        return (
            t_v = t_v, 
            t_idx = t_idx, 
            t0=t0, 
            t1=t1, 
            t_vals, 
            t_yr=t_yr, 
            t_mids=t_mids, 
            t_brks=t_brks,
            tn=length(t_vals),
            t_N= length(t_vals),
            u_v=u_v, 
            u_idx=u_disc.idx, 
            u_brks=u_disc.brks,
            u_mids=u_disc.mids, 
            u_N=u_N,
            u_vals=collect(1:u_N) 
        )
    end

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



function calculate_metrics(au_obj)
    # Map coordinates from constituent vectors to avoid FieldError
    local observation_points = tuple.(au_obj.s_x, au_obj.s_y)
    
    # Re-calculate assignments based on nearest centroid
    local assignments = [argmin([sum((p .- c).^2) for c in au_obj.centroids]) for p in observation_points]
    
    # Compute frequency counts per spatial unit
    local unit_counts = [count(==(i), assignments) for i in 1:length(au_obj.centroids)]

    # Filter valid numerical entries to prevent NaN propagation
    local valid_entries = filter(x -> !isnan(x) && !ismissing(x), unit_counts)

    if isempty(valid_entries)
        return (mean_density=NaN, sd_density=NaN, cv_density=NaN)
    end

    local m_val = mean(valid_entries)
    local s_val = std(valid_entries)
    local cv_val = s_val / (m_val + 1e-9)

    return (mean_density=m_val, sd_density=s_val, cv_density=cv_val)
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
    # Using a dummy t_idx of 1s since plotting a static slice
 
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


