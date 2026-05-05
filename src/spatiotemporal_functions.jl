
 

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
        v_idx = argmax([length(r) for r in regions])
        target = regions[v_idx]

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
        splice!(regions, v_idx, valid_splits)
        
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
        v_idx = argmax([length(r) for r in regions])
        target = regions[v_idx]

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
        deleteat!(regions, v_idx)
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
        v_idx = argmax([length(r) for r in regions])
        target = regions[v_idx]

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
        deleteat!(regions, v_idx)
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
            t_idx=get(kwargs, :time_idx, ones(Int, length(input_data))))

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
            termination_reason=reason, pts=pts, W=W)
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

    
    

 

function generate_sim_data(n_pts=10, n_time=5; rndseed=42)
    # Generates synthetic data across discrete and continuous frameworks.
    # Maintained existing structure; appended nested/joint variables (u, z).
    
    Random.seed!(rndseed)
    
    n_total = n_pts * n_time
    
    # 1. Coordinates: Space (Xlon, Xlat) and Time (T)   
    unique_pts = [(rand() * 100, rand() * 100) for _ in 1:n_pts]
    pts_full_dataset = repeat(unique_pts, n_time) # as tuple
    coords_space = vcat([collect(t)' for t in pts_full_dataset]...)  # as matrix
    
    coords_time = rand(n_total) .* (n_time+1) .+ 2000
    time_years = Int.(floor.(coords_time))
    time_idx = time_years .- 2000 .+ 1
    
    weights = ones(n_total) 
    trials  = ones(Int, n_total)

    # Components: Linear Trend + Seasonal Harmonic + Latent Process + Noise
    period=12.0
    trend = 0.05 .* coords_time[:,1] # linear trend
    seasonal = 1.0 .* cos.(2pi .* coords_time[:,1] ./ period)
    temporal_effect =  1.0 .* ( trend .+ seasonal )
    
    spatial_effect = 1.5 .* sin.(coords_space[:,1] .*  2pi) .* cos.(coords_space[:,2] .*  2pi)
    
    sigma_y = 0.2
    observation_error = sigma_y .* randn(n_total) 

    # 2. Covariates
    # Z: Purely spatial covariate
    Z = randn(n_total)

    # Latent (True) Spatiotemporal Covariates + error
    U1_true = 0.5 .* sin.(coords_time[:,1] ./ 5.0) .+ 0.5 .* Z  
    U2_true = 0.5 .* cos.(coords_time[:,1] ./ 5.0) .- 0.3 .* Z  
    U3_true = 0.2 .* (coords_time[:,1] ./ n_total) .+ 0.1 .* Z  

    # 3. Add measurement error to covariates (observed version)
    sigma_u1 = 0.1
    sigma_u2 = 0.2
    sigma_u3 = 0.3
    
    U1_obs = U1_true .+ randn(n_total) .* sigma_u1
    U2_obs = U2_true .+ randn(n_total) .* sigma_u2
    U3_obs = U3_true .+ randn(n_total) .* sigma_u3

    cov_continuous = hcat(U1_obs, U2_obs, U3_obs)

    N_cat = 7
    probs = collect(range(0.0, stop=1.0, length=N_cat + 1))
    breaks1 = quantile(U1_obs, probs)
    breaks2 = quantile(U2_obs, probs)
    breaks3 = quantile(U3_obs, probs)
    time_idx_quantiles1 = map(x -> clamp(searchsortedfirst(breaks1, x) - 1, 1, N_cat), U1_obs)
    time_idx_quantiles2 = map(x -> clamp(searchsortedfirst(breaks2, x) - 1, 1, N_cat), U2_obs)
    time_idx_quantiles3 = map(x -> clamp(searchsortedfirst(breaks3, x) - 1, 1, N_cat), U3_obs)
    cov_indices_mat = hcat(time_idx_quantiles1, time_idx_quantiles2, time_idx_quantiles3)
    
    # 4. Generate Dependent Variable Y
    y_obs = 1.0 .+  spatial_effect + temporal_effect .+  observation_error .+ U1_obs .+ U2_obs .+ U3_obs
    
    y_binary = y_obs .> (mean(y_obs) + 0.5)
    y_counts = abs.(Int.(round.(y_obs))) * 100
    
    # fixed effects
    class1_sim = rand(1:13, n_total)
    class2_sim = rand(1:2, n_total)
      
    u_obs = randn(n_total, 3)
    
    return (
        pts=pts_full_dataset,
        coords_space=coords_space, 
        coords_time=coords_time, 
        time_years=time_years, 
        time_idx=time_idx, 
        weights=weights, 
        trials=trials,
        y_obs=y_obs, 
        y_binary=y_binary, 
        y_counts=y_counts,
        class1_sim=class1_sim, 
        class2_sim=class2_sim, 
        z_obs=Z, 
        u_obs=cov_continuous,
        cov_continuous=cov_continuous, 
        cov_indices_mat=cov_indices_mat
    )
end



function estimate_local_kde_with_extrapolation(pts, time_idx, target_ts; grid_res=600, sd_extension_factor=0.25)
    """
    Synopsis: Estimates 2D KDE for a specific time slice with extrapolation.
    Inputs:
    - pts: Vector of (x, y) coordinates for all time points.
    - time_idx: Vector of time indices corresponding to pts.
    - target_ts: The specific time slice to estimate KDE for.
    - grid_res: Resolution of the output grid (e.g., 100 for 100x100 grid).
    - sd_extension_factor: Multiplier for standard deviation to define the bandwidth.
    Outputs:
    - Tuple (x_grid, y_grid, intensity) where intensity is a matrix.
    """
    # Filter points for the target time slice
    filtered_pts = [p for (i, p) in enumerate(pts) if time_idx[i] == target_ts]
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
        u_idx, v_idx = get(centroid_map, edge[1], 0), get(centroid_map, edge[2], 0)
        if u_idx > 0 && v_idx > 0 add_edge!(g, u_idx, v_idx) end
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
    # Using a dummy time_idx of 1s since we are plotting a static slice
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
    time_idx = repeat(1:n_years, inner=nAU)
    pts = repeat(pts_base, n_years)
    # The area_idx is the spatial unit identifier (1 to 56)
    area_idx = repeat(1:nAU, n_years)
 
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
        y=y_final, X=X, log_offset=log_offset, time_idx=time_idx,
        area_idx=area_idx, n_years=n_years, pts=pts, W=W, au=au
    )
end


 
function get_optimal_sampler(model_key::String; nuts_adapt=10, target_acc=0.8, pg_particles=40)
    # Optimized Gibbs configurations aligned exactly with model_list registry keys

    samplers = Dict(
        # --- D-Series: Discrete Spatial Models (GMRF/BYM2) ---
        "model_D00_poisson_simple"      => Gibbs((:u_icar, :u_iid, :f_tm_raw) => ESS(), (:sigma_sp, :phi_sp, :sigma_tm, :rho_tm) => MH()),
        "model_D01_poisson"             => Gibbs((:u_icar, :u_iid, :f_tm_raw, :st_int_raw) => ESS(), (:phi_zi) => PG(pg_particles), (:beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_sp, :phi_sp, :sigma_tm, :rho_tm, :sigma_int, :sigma_rw2) => MH()),
        "model_D02_poisson_leroux"      => Gibbs((:phi_leroux, :f_tm_raw, :st_int_raw) => ESS(),  (:phi_zi) => PG(pg_particles), (:beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:tau_leroux, :rho_leroux, :sigma_tm, :rho_tm, :sigma_int, :sigma_rw2) => MH()),
        "model_D03_poisson_localised"   => Gibbs((:s_eff, :f_tm_raw, :st_int_raw) => ESS(),  (:phi_zi) => PG(pg_particles), (:mu_global, :sigma_cluster, :mu_clusters, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:tau_sp, :rho_sp, :sigma_tm, :rho_tm, :sigma_int, :sigma_rw2) => MH()),
        "model_D04_poisson_sar"         => Gibbs((:s_eff, :f_tm_raw, :st_int_raw) => ESS(), (:phi_zi) => PG(pg_particles), (:beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_sp, :rho_sar, :sigma_tm, :rho_tm, :sigma_int, :sigma_rw2) => MH()),
        "model_D05_poisson_mcar"        => Gibbs((:u_raw, :f_tm_raw, :st_int_raw) => ESS(), (:phi_zi) => PG(40), (:L_corr, :beta_cov) => Turing.NUTS(10, 0.8), (:sigma_outcome, :sigma_tm, :rho_tm, :sigma_int, :sigma_rw2) => MH()),
        "model_D06_poisson_svc"         => Gibbs((:W_svc_raw, :u_icar, :u_iid, :f_tm_raw, :st_int_raw) => ESS(), (:phi_zi) => PG(pg_particles), (:beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_svc, :corr_svc, :sigma_sp, :phi_sp, :sigma_tm, :rho_tm, :sigma_int, :sigma_rw2) => MH()),
        "model_D07_poisson_dag"         => Gibbs((:s_raw, :f_tm_raw, :st_int_raw) => ESS(), (:phi_zi) => PG(pg_particles), (:beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_sp, :rho_dag, :sigma_tm, :rho_tm, :sigma_int, :sigma_rw2) => MH()),
        "model_D08_poisson_hurdle"      => Gibbs((:u_icar_h, :u_iid_h, :f_tm_h, :u_icar_c, :u_iid_c, :f_tm_c) => ESS(), (:phi_zi) => PG(pg_particles), (:beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_sp_h, :phi_sp_h, :sigma_tm_h, :rho_tm_h, :sigma_sp_c, :phi_sp_c, :sigma_tm_c, :rho_tm_c, :sigma_rw2) => MH()),
        "model_D09_poisson_ei"          => Gibbs((:u_icar, :u_iid, :f_tm_raw, :st_int_raw, :beta_rff) => ESS(), (:phi_zi) => PG(pg_particles), (:W_rff, :b_rff, :ls_rff, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_sp, :phi_sp, :sigma_tm, :rho_tm, :sigma_f_rff, :sigma_int, :sigma_rw2) => MH()),
        "model_D10_gaussian"            => Gibbs((:u_icar, :u_iid, :f_tm_raw, :st_int_raw) => ESS(), (:phi_zi) => PG(pg_particles), (:beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_sp, :phi_sp, :sigma_tm, :rho_tm, :sigma_int, :sigma_rw2) => MH()),
        "model_D11_gaussian_rff"        => Gibbs((:u_icar, :u_iid, :w_trend, :w_seas, :st_int_raw) => ESS(), (:phi_zi) => PG(pg_particles), (:beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_sp, :phi_sp, :sigma_trend, :sigma_seas, :sigma_int, :sigma_rw2) => MH()),
        "model_D12_lognormal"           => Gibbs((:u_icar, :u_iid, :f_tm_raw, :st_int_raw) => ESS(), (:phi_zi) => PG(pg_particles), (:beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_sp, :phi_sp, :sigma_tm, :rho_tm, :sigma_int, :sigma_rw2) => MH()),
        "model_D13_binomial"            => Gibbs((:u_icar, :u_iid, :f_tm_raw, :st_int_raw) => ESS(), (:phi_zi) => PG(pg_particles), (:beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_sp, :phi_sp, :sigma_tm, :rho_tm, :sigma_int, :sigma_rw2) => MH()),
        "model_D14_negbin"              => Gibbs((:u_icar, :u_iid, :f_tm_raw, :st_int_raw) => ESS(), (:phi_zi) => PG(pg_particles), (:beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:r_nb, :sigma_sp, :phi_sp, :sigma_tm, :rho_tm, :sigma_int, :sigma_rw2) => MH()),
        "model_D15_gaussian_rff_cov"    => Gibbs((:u_icar, :u_iid, :f_tm_raw, :st_int_raw, :W_cov_raw) => ESS(), (:lengthscale_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_sp, :phi_sp, :sigma_tm, :rho_tm, :sigma_int, :sigma_cov) => MH()),
        "model_D16_gaussian_fft"        => Gibbs((:u_spectral_raw, :u_iid, :f_tm_raw) => ESS(), (:phi_zi) => PG(pg_particles), (:beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_sp, :phi_sp, :sigma_tm, :rho_tm, :sigma_rw2) => MH()),
        "model_D17_adaptive_rff_bym2"   => Gibbs((:beta_rff, :u_icar, :u_iid) => ESS(), (:phi_zi) => PG(pg_particles), (:W_matrix, :b_phases, :beta_z, :beta_u, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_f, :sigma_sp, :phi_sp, :sigma_rw2) => MH()),
        "model_D18_nested_multifid_rff" => Gibbs((:beta_rff, :u_icar, :u_iid) => ESS(), (:phi_zi) => PG(pg_particles), (:W_matrix, :b_phases, :beta_z, :beta_u_main, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_u, :sigma_f, :sigma_sp, :phi_sp, :sigma_rw2) => MH()),
        "model_D19_tv_intercept_bym2"   => Gibbs((:beta_rff, :u_icar, :u_iid) => ESS(), (:phi_zi) => PG(pg_particles), (:alpha_rw, :beta_z, :beta_u_main, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_alpha, :sigma_f, :sigma_sp, :sigma_rw2) => MH()),
        "model_D20_stochastic_vol"      => Gibbs((:beta_rff_mean, :beta_rff_vol, :u_icar, :u_iid) => ESS(), (:phi_zi) => PG(pg_particles), (:alpha_rw, :beta_z, :beta_u_main, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_log_var, :sigma_f, :sigma_sp, :sigma_rw2) => MH()),
        "model_D21_fitc_bym2"           => Gibbs((:u_inducing, :u_icar, :u_iid, :f_tm_raw, :st_int_raw) => ESS(), (:phi_zi) => PG(pg_particles), (:ls_st, :beta_cos, :beta_sin, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_f, :sigma_sp, :phi_sp, :sigma_tm, :rho_tm, :sigma_int, :sigma_rw2) => MH()),
        "model_D22_fitc_nonlinear"      => Gibbs((:u_inducing, :f_gp, :beta_rff_u1, :beta_rff_u2, :beta_rff_u3, :beta_rff_vol, :u_icar, :u_iid) => ESS(), (:phi_zi) => PG(pg_particles), (:Z_inducing, :ls_st, :W_vol, :b_vol, :beta_z, :beta_u_main, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_f, :sigma_sp, :phi_sp, :sigma_rw2) => MH()),
        "model_D23_gptime_fitc"         => Gibbs((:alpha_gp, :u_inducing, :f_gp, :beta_rff_u1, :beta_rff_u2, :beta_rff_u3, :beta_rff_vol, :u_icar, :u_iid) => ESS(), (:phi_zi) => PG(pg_particles), (:ls_trend, :ls_st, :W_vol, :b_vol, :beta_z, :beta_u_main, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_trend, :sigma_f, :sigma_sp, :phi_sp, :sigma_rw2) => MH()),

        # --- C-Series: Continuous Spatial Models (Deep GP/RFF/SPDE) ---
        "model_C01_dense_gp"            => Gibbs((:f_latent) => ESS(), (:ls_s, :ls_t, :beta_cos, :beta_sin, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_f, :sigma_rw2) => MH()),
        "model_C02_deep_gp"             => Gibbs((:w1, :w2) => ESS(), (:l1, :l2, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_rw2) => MH()),
        "model_C03_binomial_deep_gp"    => Gibbs((:w1, :w2) => ESS(), (:lengthscale1, :lengthscale2, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_rw2) => MH()),
        "model_C04_deep_gp_3layer"      => Gibbs((:w1, :w2, :w3) => ESS(), (:l1, :l2, :l3, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_rw2) => MH()),
        "model_C05_non_separable_rff"   => Gibbs((:w_joint) => ESS(), (:l_joint, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_joint, :sigma_rw2) => MH()),
        "model_C06_spde_rff"            => Gibbs((:w_sp, :f_tm_raw) => ESS(), (:kappa_sp, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_sp, :sigma_tm, :rho_tm, :sigma_rw2) => MH()),
        "model_C07_nonstationary_warp"  => Gibbs((:w_warp, :w_sp, :f_tm_raw) => ESS(), (:l_warp, :l_spatial, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_sp, :sigma_tm, :rho_tm, :sigma_rw2) => MH()),
        "model_C08_refined_mosaic"      => Gibbs((:w_local) => ESS(), (:mu_local, :l_local, :mu_global, :sigma_mu_local, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_rw2, :sigma_local, :sigma_y_local) => MH()),
        "model_C09_integrated_mosaic"   => Gibbs((:w_local) => ESS(), (:mu_local, :l_joint, :mu_global, :sigma_mu_local, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_rw2, :sigma_local, :sigma_y_local) => MH()),
        "model_C10_fitc_gmrf_hybrid"    => Gibbs((:alpha_gp, :u_inducing, :f_gp) => ESS(), (:ls_st, :ls_trend, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_trend, :sigma_sp, :sigma_rw2) => MH()),
        "model_C11_svgp_learned"        => Gibbs((:u_inducing, :f_gp, :alpha_gp, :beta_rff_u1, :beta_rff_u2, :beta_rff_u3, :beta_rff_vol, :u_icar, :u_iid) => ESS(), (:Z_inducing, :ls_st, :ls_trend, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_f, :sigma_sp, :sigma_rw2) => MH()),
        "model_C12_svgp_full"           => Gibbs((:u_latent, :alpha_gp, :beta_rff_u1, :beta_rff_u2, :beta_rff_u3, :beta_rff_vol, :u_icar, :u_iid) => ESS(), (:Z_inducing, :m_u, :s_u_diag, :ls_st, :ls_trend, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_f, :sigma_sp, :sigma_y, :sigma_rw2) => MH()),
        "model_C13_multifidelity_gp"    => Gibbs((:u_icar, :u_iid, :f_tm_raw, :st_int_raw, :beta_z_rff, :beta_u_rff) => ESS(), (:W_z, :b_z, :W_u, :b_u, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_sp, :phi_sp, :sigma_tm, :rho_tm, :sigma_int, :sigma_rw2, :sigma_z, :sigma_u) => MH()),
        "model_C14_minibatch_mfgp"      => Gibbs((:u_icar, :u_iid, :f_tm_raw, :st_int_raw) => ESS(), (:beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_sp, :phi_sp, :sigma_tm, :rho_tm, :sigma_int, :sigma_rw2) => MH()),
        "model_C15_deep_gp_rff"         => Gibbs((:beta_z, :beta_u_mat, :beta_y_gp, :u_icar, :u_iid) => ESS(), (:W_z, :b_z, :W_u, :b_u, :W_y, :b_y, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_sp, :phi_sp, :sigma_z, :sigma_u, :sigma_rw2) => MH()),
        "model_C16_nystrom_sv"          => Gibbs((:v_latent, :beta_rff_sigma, :u_icar, :u_iid, :u1_coeff) => ESS(), (:ls_st, :W_sigma, :b_sigma, :W_u1, :b_u1, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_f, :sigma_sp, :phi_sp, :sigma_u, :sigma_rw2) => MH()),
        "model_C17_spde_trend"          => Gibbs((:alpha, :u_icar, :u_iid, :beta_rff_sigma) => ESS(), (:ls_trend, :W_sigma, :b_sigma, :W_u1, :b_u1, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_trend, :sigma_sp, :phi_sp, :sigma_u, :sigma_rw2) => MH()),
        "model_C18_kron_spde"           => Gibbs((:u_icar, :u_iid, :y_noise, :z_noise, :u1_noise) => ESS(), (:ls_s_y, :ls_t_y, :ls_s_cov, :ls_t_cov, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_sp, :phi_sp, :sigma_s_y, :sigma_t_y, :sigma_s_cov, :sigma_t_cov, :sigma_rw2) => MH()),
        "model_C19_svgp_matern"         => Gibbs((:u_icar, :u_iid, :y_noise) => ESS(), (:ls_s_y, :ls_t_y, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_sp, :phi_sp, :sigma_s_y, :sigma_t_y, :sigma_rw2) => MH()),
        "model_C20_mf_gp_matern"        => Gibbs((:z_latent, :u1_noise, :y_noise, :u_icar, :u_iid) => ESS(), (:ls_z, :ls_s_u, :ls_s_y, :beta_uz, :beta_y, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_sp, :phi_sp, :sigma_rw2) => MH()),
        "model_C21_mf_sv_seasonal"      => Gibbs((:z_latent, :u1_noise, :y_noise, :u_icar, :u_iid, :beta_rff_sigma) => ESS(), (:ls_z, :ls_s_u, :ls_s_y, :W_sigma, :b_sigma, :beta_y_covs, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_y, :sigma_sp, :phi_sp, :sigma_rw2) => MH()),
        "model_C22_rff_mf_gp"           => Gibbs((:beta_z, :beta_u1, :beta_y_gp, :u_icar, :u_iid, :beta_rff_sigma) => ESS(), (:W_z, :b_z, :W_u, :b_u, :W_y_gp, :b_y_gp, :W_sigma, :b_sigma, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_z_f, :sigma_u_f, :sigma_y_gp_f, :sigma_sp, :phi_sp, :sigma_rw2) => MH()),
        "model_C23_dfrff_mf_gp"         => Gibbs((:beta_z, :beta_u, :beta_y_gp, :u_icar, :u_iid) => ESS(), (:beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_z_f, :sigma_u_f, :sigma_y, :sigma_sp, :phi_sp, :sigma_rw2) => MH()),
        "model_C24_semi_adaptive_rff"   => Gibbs((:u_icar, :u_iid) => ESS(), (:W_z, :W_u, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_W_z, :sigma_W_u, :sigma_y, :sigma_sp, :phi_sp, :sigma_rw2) => MH()),
        "model_C25_hybrid_fitc_rff"     => Gibbs((:beta_z, :u_inducing, :u_icar, :u_iid) => ESS(), (:ls_st, :beta_cov) => Turing.NUTS(nuts_adapt, target_acc), (:sigma_z_f, :sigma_f, :sigma_y, :sigma_sp, :phi_sp, :sigma_rw2) => MH())
    )

    return get(samplers, model_key, Turing.NUTS(nuts_adapt, target_acc))
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
sqexp_cov_fn(D, phi, eps=1e-3) = exp.(-D^2 / phi) + LinearAlgebra.I * eps

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


Turing.@model function gaussian_process_basic(; Y, D, cov_fn=exp_cov_fn, nData=length(Y) )
    mu ~ Normal(0.0, 1.0); # mean process
    sig2 ~ LogNormal(0, 1) # "nugget" variance
    phi ~ LogNormal(0, 1) # phi is ~ lengthscale along Xstar (range parameter)
    # sigma = cov_fn(D, phi) + sig2 * LinearAlgebra.I(nData) # Realized covariance function + nugget variance
    vcov = cov_fn(D, phi) + sig2 .* LinearAlgebra.I(nData ) .+ eps() * 1000.0
    Y ~ MvNormal(mu * ones(nData), Symmetric(vcov) )     # likelihood
end


Turing.@model function gaussian_process_covars(; Y, X, D, cov_fn=exp_cov_fn, nData=length(Y), nF=size(X,2) )
    # model matrix for fixed effects (X)
    beta ~ filldist( Normal(0.0, 1.0), nF); 
    sig2 ~ LogNormal(0, 1) # "nugget" variance
    phi ~ LogNormal(0, 1) # phi is ~ lengthscale along Xstar (range parameter)
    # sigma = cov_fn(D, phi) + sig2 * LinearAlgebra.I(nData) # Realized covariance function + nugget variance
    mu = X * beta # mean process
    vcov = cov_fn(D, phi) + sig2 .* LinearAlgebra.I(nData ) .+ eps() .* 1000.0
    Y ~ MvNormal(mu, Symmetric(vcov) )     # likelihood
end



Turing.@model function gaussian_process_ar1( ::Type{T}=Float64; Y, X, D, ar1, cov_fn=exp_cov_fn, 
    nData=length(Y), nF=size(X,2), nT=maximum(ar1)-minimum(ar1)+1 ) where {T} 
 
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
    
    Y ~ MvNormal( X * beta .+ ymean_ar1[ar1[1:nData]], Symmetric(cov_fn(D, phi) .+ sig2 * I(nData) .+ eps() ) )     # likelihood
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
      JLD2.@save "$(filename_val)" areal_units mod chain pts y_sim y_binary time_idx weights trials cov_indices cov_indices_mat trials_sim class1_sim class2_sim weights_sim adj_matrix_numeric n_pts n_time area_method
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
      JLD2.@load "$(filename_val)" areal_units mod chain pts y_sim y_binary time_idx weights trials cov_indices cov_indices_mat trials_sim class1_sim class2_sim weights_sim adj_matrix_numeric n_pts n_time area_method
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
    # Description:
    #   Calculates summary statistics (mean, median, CIs) across the last dimension.
    # Inputs:
    #   samples: N-dimensional array where the last dimension is MCMC iterations.
    #   alpha: Significance level for credible intervals.
    # Outputs:
    #   NamedTuple: (mean, median, lower, upper).

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


 


function detect_model_family(model::DynamicPPL.Model)
    # Description: Infers the likelihood family (e.g. :poisson) from the Turing model function name.
    # Inputs:
    #   - model: Turing model object.
    #   - Outputs:
    #   - Symbol indicating the family.

    name = lowercase(string(model.f))
    if occursin("gaussian", name) return :gaussian end
    if occursin("poisson", name) return :poisson end
    if occursin("binomial", name) return :binomial end
    if occursin("negativebinomial", name) return :negbinomial end
    if occursin("lognormal", name) return :lognormal end
    return :gaussian # Fallback
end




function plot_posterior_results(stats, modinputs=nothing, areal_units=nothing; pts=nothing, time_slice=nothing, effect=:spatial, cov_idx=1, show_pts=false)
    # Description: Comprehensive posterior visualization for CARSTM and Deep GP models.

    if !isnothing(modinputs)
        pts = modinputs.pts
    end
 
    # 1. Handle Categorical/Class Bar Plots
    if effect == :beta_cov
        b_stats = stats.beta_cov[cov_idx]
        n_levels = size(b_stats.mean, 1)
        return StatsPlots.bar(1:n_levels, b_stats.mean[:,1],
                  yerror=(b_stats.mean[:,1] .- b_stats.lower[:,1], b_stats.upper[:,1] .- b_stats.mean[:,1]),
                  title="Covariate $cov_idx Effects", xlabel="Level", ylabel="Effect Size", legend=false)

    elseif effect == :b_class1 || effect == :b_class2
        b_stats = effect == :b_class1 ? stats.b_class1 : stats.b_class2
        if isnothing(b_stats); error("Effect $effect not found in stats"); end
        n_levels = size(b_stats.mean, 1)
        return StatsPlots.bar(1:n_levels, b_stats.mean[:,1],
                  yerror=(b_stats.mean[:,1] .- b_stats.lower[:,1], b_stats.upper[:,1] .- b_stats.mean[:,1]),
                  title="$effect Levels", xlabel="Class Index", ylabel="Effect Size", legend=false)

    # 2. Handle Temporal Main Effects
    elseif effect == :temporal
        t_stats = stats.temporal
        n_times = length(t_stats.mean)
        return StatsPlots.plot(1:n_times, t_stats.mean,
                   ribbon=(t_stats.mean .- t_stats.lower, t_stats.upper .- t_stats.mean),
                   fillalpha=0.2, lw=2, title="Temporal Main Effect", xlabel="Time Index", ylabel="Effect (Latent Scale)", legend=false)

    # 2a. Handle Seasonal Effects
    elseif effect == :seasonal
        t_stats = stats.seasonal
        n_times = length(t_stats.mean)
        return StatsPlots.plot(1:n_times, t_stats.mean,
                   ribbon=(t_stats.mean .- t_stats.lower, t_stats.upper .- t_stats.mean),
                   fillalpha=0.2, lw=2, title="Seasonal Effect", xlabel="Time Index", ylabel="Effect (Latent Scale)", legend=false)


    # 3. Handle Spatial, ST, and Deep GP Mean Fields
    elseif effect in [:spatial, :spatial_structured, :spatial_unstructured, :predictions_denoised, :predictions_noisy, :residuals, :eta_gp, :hidden_layer]
        plt = StatsPlots.plot(aspect_ratio=:equal, title="$effect (T=$(time_slice))", legend=true)

        # Determine the values to map to colors
        values = if effect == :spatial
            stats.spatial.mean
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



function prepare_model_inputs(; kwargs...)
     
    y = get(kwargs, :y, nothing) # better to throw an error as it is required

    N_obs = get(kwargs, :N_obs,  length(y))
 
    area_idx = get(kwargs, :area_idx, nothing)  # better to throw an error as it is required
    if isnothing(area_idx)
        error("area_idx not provided: It is required. ")
    end

    time_idx = get(kwargs, :time_idx, nothing)  # better to throw an error as it is required
    if isnothing(time_idx)
        error("time_idx not provided: It is required.")
    end

    N_time = get(kwargs, :N_time,  maximum(time_idx) )

    W = get(kwargs, :W, I(1)) # Adjacency matrix

    N_areas = size(W, 1)

    pts = get(kwargs, :pts, nothing)
    if isnothing(pts)
        error("pts not provided: It is required. ")
    end

    coords_space = isnothing(pts) ? nothing : get(kwargs, :coords_space, vcat([collect(t)' for t in pts]...) ) # as a matrix 
    
    coords_time = isnothing(time_idx) ? nothing : get(kwargs, :coords_time, time_idx)
    period = get(kwargs, :period, 12 )

    weights = get(kwargs, :weights, ones(Float64, N_obs) )

    log_offset = get(kwargs, :log_offset, zeros(Float64, N_obs))
    
    trials = get(kwargs, :trials, ones(Int, N_obs))
    
    rnd_seed = get(kwargs, :rnd_seed, 42)
    Random.seed!(rnd_seed)
    
    # Ensures that multi-fidelity and weighted variables exist even for simple models
     
    # 3. Standardized Precision Scaling (BYM2 & RW2)
    # Pre-scaling ensures that priors on sigma_sp and sigma_rw2 are interpretable as marginal scales
    
    # A. Spatial Scaling (BYM2 ICAR)
    D_sp = Diagonal(vec(sum(W, dims=2)))
    Q_sp_raw = D_sp - W
    eigs_sp = filter(x -> x > 1e-6, eigvals(Matrix(Q_sp_raw)))
    scaling_sp_const = exp(mean(log.(eigs_sp)))
    Q_spatial_scaled = sparse(Q_sp_raw ./ scaling_sp_const)
    
    # B. Smoothing Scaling (RW2)
    N_cat = get(kwargs, :N_cat, 9)

    D_rw2_mat = spzeros(Float64, N_cat - 2, N_cat)
    for i in 1:(N_cat - 2)
        D_rw2_mat[i, i] = 1.0; D_rw2_mat[i, i+1] = -2.0; D_rw2_mat[i, i+2] = 1.0
    end
    Q_rw2_raw = D_rw2_mat' * D_rw2_mat
    eigs_rw2 = filter(x -> x > 1e-6, eigvals(Matrix(Q_rw2_raw)))
    scaling_rw2_const = exp(mean(log.(eigs_rw2)))
    Q_rw2_scaled = sparse(Q_rw2_raw ./ scaling_rw2_const)

    # 4. Fixed Projections & RFF Bases
    # Generating fixed weights and phases ensures stability across iterations for deep functional RFFs
    
    # Trend & Seasonality Static Projection Vectors
    m_trend = get(kwargs, :m_trend, 10)
    m_seas = get(kwargs, :m_seas, 5)
    t_vec = get(kwargs, :t_vec,  collect(1:N_time) ./ N_time )

    Om_tr = randn(Float64, m_trend) ./ 0.5; 
    Ph_tr = rand(Float64, m_trend) .* 2pi
    Z_trend = sqrt(2/m_trend) .* cos.(t_vec * Om_tr' .+ Ph_tr')
    
    Z_seas = zeros(Float64, N_time, 2 * m_seas)
    for j in 1:m_seas
        om_j = 2*pi * j; Z_seas[:, 2j-1] = cos.(om_j .* t_vec); Z_seas[:, 2j] = sin.(om_j .* t_vec)
    end

    # Fixed weights for multi-fidelity/A-series models (A13-A25)
    # --- 2. Initialize Fixed Projection Matrices (DFRFF) ---
    # These are static weights for the functional mapping layers
    # M_base = 40
    # M_sig = 20

    # Layer Z: Maps Space (Dim 2) to Latent Z
    # W_z_fix = randn(2, M_base); b_z_fix = rand(M_base) .* 2π

    # Layer U: Maps [Space, Time, Z] (Dim 4) to Latent U
    # W_u_fix = randn(4, M_base); b_u_fix = rand(M_base) .* 2π

    # Layer Y: Maps [Space, Time, Z, U] (Dim 5) to Output Y
    # W_y_fix = randn(5, M_base); b_y_fix = rand(M_base) .* 2π

    # Layer Sigma: Maps [Space, Time] (Dim 3) to Volatility
    
    M_rff = M_rff_base = get(kwargs, :M_rff_base, 40)
    M_rff_sigma = get(kwargs, :M_rff_sigma, 20)
 
    W_z_fixed = randn(2, M_rff_base); 
    b_z_fixed = rand(M_rff_base) .* 2pi

    W_u_fixed = randn(4, M_rff_base); 
    b_u_fixed = rand(M_rff_base) .* 2pi

    W_y_gp_fixed = randn(3, M_rff_base); 
    b_y_gp_fixed = rand(M_rff_base) .* 2pi

    W_sigma_fixed = randn(3, M_rff_sigma); 
    b_sigma_fixed = rand(M_rff_sigma) .* 2pi

    # 5. Spatial Geometric Indices
    # Coordinates and Inducing points for FITC Sparse GPs
    coords_space_y = hcat([p[1] for p in pts], [p[2] for p in pts])
    coords_st_full = hcat(coords_space_y, time_idx ./ N_time)
    
    M_inducing_count = get(kwargs, :M_inducing_count, 15)
    Z_inducing = kmeans_inducing_points(coords_st_full, M_inducing_count)
     
    # Subsets for varied fidelity resolutions
    z_obs = get(kwargs, :z_obs, zeros(Float64, N_obs))
    u_obs = get(kwargs, :u_obs, zeros(Float64, N_obs))
    
    coords_z_spatial = coords_space_y[1:min(length(z_obs), size(coords_space_y,1)), :]
    coords_u_st = hcat(
      coords_space_y[1:min(size(u_obs,1), size(coords_space_y,1)), :], 
      (time_idx ./ N_time)[1:min(size(u_obs,1), length(time_idx))]
    )

    # 6. Interaction & Covariate Mapping Vectors
    # interaction_idx: Maps (area, time) to a unique linear index for Type IV interactions
    interaction_idx = (time_idx .- 1) .* N_areas .+ area_idx
    
    # cov_mapping: Discretizes covariates for RW2 smoothing models
    cov_mapping = zeros(Int, N_obs, 4)
    for k in 1:4; cov_mapping[:, k] .= mod1.(1:N_obs, N_cat); end
    
    # 7. Templates for AR1 Temporal Effects
    Q_ar1_template = spdiagm(0 => ones(N_time), 1 => fill(-1.0, N_time-1), -1 => fill(-1.0, N_time-1))

    # Return centralized named tuple used by the model registry
    return (
        y = y, 
        pts = pts, 
        area_idx = area_idx, 
        time_idx = time_idx, 
        coords_space = coords_space, 
        coords_time = coords_time, 
        W = W, 
        trials = trials, 
        z_obs = z_obs, 
        u_obs = u_obs, 
        weights = weights, 
        log_offset = log_offset,
        coords_z_spatial = coords_z_spatial,   
        coords_u_st = coords_u_st,
        Q_sp = Q_spatial_scaled, 
        Q_rw2 = Q_rw2_scaled, 
        Q_ar1_template = Q_ar1_template,
        Z_trend = Z_trend, 
        Z_seas = Z_seas, 
        Z_inducing = Z_inducing,
        M_inducing_val = get( kwargs, :M_inducing_val, 15),
        M_rff = M_rff,
        M_rff_u = get( kwargs, :M_rff_u, 30),
        M_rff_sigma = M_rff_sigma,
        M_rff_base = M_rff_base,
        W_z_fixed = W_z_fixed, 
        b_z_fixed = b_z_fixed, 
        W_u_fixed = W_u_fixed, 
        b_u_fixed = b_u_fixed,
        W_y_gp_fixed = W_y_gp_fixed, 
        b_y_gp_fixed = b_y_gp_fixed, 
        W_sigma_fixed = W_sigma_fixed, 
        b_sigma_fixed = b_sigma_fixed,
        interaction_idx = interaction_idx, 
        cov_indices = cov_mapping,
        n_mosaics = get(kwargs, :n_mosaics, 5),
        m_rff = get(kwargs, :m_rff, 20),
        D_st =  get(kwargs, :D_st, 3 ),  # svgp
        period = period,  # svgp
        N_obs = N_obs, 
        N_areas = N_areas, 
        N_time = N_time, 
        N_cat = N_cat,
        m_feat = 5,
        m_warp = 10, 
        m_spatial=50,
        grid_res = get(kwargs, :grid_res,  64), 
        pad_factor = get(kwargs, :pad_factor, 2),
        n_clusters = get(kwargs, :n_clusters, 4), # model_D03_poisson_localised
        jitter=1e-4,
        use_zi = get(kwargs, :use_zi, false) 
    )
end



# Helper to create AR1 precision matrix
function ar1_precision(n, rho, sigma_e)
    Q = spzeros(n, n)
    # Main diagonal
    Q[1, 1] = 1.0
    for i in 2:(n - 1)
        Q[i, i] = 1.0 + rho^2
    end
    Q[n, n] = 1.0
    # Off-diagonals
    for i in 1:(n - 1)
        Q[i, i + 1] = -rho
        Q[i + 1, i] = -rho
    end
    return (1.0 / sigma_e^2) .* Q
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
            u_idx = get(centroid_map, c1, 0)
            v_idx = get(centroid_map, c2, 0)
            if u_idx > 0 && v_idx > 0 && !has_edge(g_final, u_idx, v_idx)
                add_edge!(g_final, u_idx, v_idx)
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
            u_idx = get(centroid_map, c1, 0)
            v_idx = get(centroid_map, c2, 0)
            if u_idx > 0 && v_idx > 0 && !has_edge(g_final, u_idx, v_idx)
                add_edge!(g_final, u_idx, v_idx)
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

function waic_compute(model::DynamicPPL.Model, chain::MCMCChains.Chains, modinputs; use_weights=true)
    """
    Consolidated WAIC calculation function.
    1. Attempts native Turing pointwise log-likelihood extraction.
    2. Falls back to manual reconstruction if native fails.
    3. Dynamically handles BYM2 (u_icar), Leroux (phi_leroux), and SAR/DAG (s_eff) spatial effects.
    """
    N_obs = length(modinputs.y)
    N_samples = size(chain, 1)
    family = detect_model_family(model)
    weights = use_weights ? modinputs.weights : ones(N_obs)
    chain_names = string.(names(chain))
    model_type = identify_model_type(chain)

    # Strategy 1: Native Turing extraction
    try
        pointwise_ll = Turing.pointwise_loglikelihoods(model, chain)
        y_keys = [k for k in keys(pointwise_ll) if occursin("y", string(k))]

        if !isempty(y_keys)
            log_lik_mat = Float64.(copy(pointwise_ll[y_keys[1]]))
            for i in 2:length(y_keys)
                log_lik_mat .+= pointwise_ll[y_keys[i]]
            end

            if use_weights
                for i in 1:N_obs; log_lik_mat[:, i] .*= weights[i]; end
            end

            lppd = sum(logsumexp(log_lik_mat[:, i]) - log(N_samples) for i in 1:N_obs)
            p_waic = sum(var(log_lik_mat, dims=1))
            return -2 * (lppd - p_waic)
        end
    catch e
        @info "Native pointwise LL failed for $(model.f), attempting manual reconstruction..."
    end

    # Strategy 2: Manual Reconstruction Fallback
    log_lik_mat = zeros(N_samples, N_obs)
    N_areas = size(modinputs.Q_sp, 1)
    N_time = maximum(modinputs.time_idx)

    for s in 1:N_samples
        # --- Robust Spatial Effect Reconstruction ---
        s_eff = zeros(N_areas)
        
        if any(occursin.("phi_leroux", chain_names))
            # Case: Leroux Models (D02, D03)
            s_eff = [chain[Symbol("phi_leroux[$i]")].data[s] for i in 1:N_areas]
        elseif any(occursin.("u_icar", chain_names))
            # Case: BYM2 Models (D01, D10+)
            sig_sp = :sigma_sp in names(chain) ? chain[:sigma_sp].data[s] : 1.0
            phi_sp = :phi_sp in names(chain) ? chain[:phi_sp].data[s] : 0.5
            u_icar = [chain[Symbol("u_icar[$i]")].data[s] for i in 1:N_areas]
            u_iid = [chain[Symbol("u_iid[$i]")].data[s] for i in 1:N_areas]
            s_eff = sig_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)
        elseif any(occursin.("s_eff", chain_names))
            # Case: SAR/DAG Models (D04, D07)
            s_eff = [chain[Symbol("s_eff[$i]")].data[s] for i in 1:N_areas]
        end

        # --- Temporal Effect Reconstruction ---
        f_time = zeros(N_time)
        if model_type == :ar1
            sig_tm = :sigma_tm in names(chain) ? chain[:sigma_tm].data[s] : 1.0
            f_time = [chain[Symbol("f_tm_raw[$i]")].data[s] for i in 1:N_time] .* sig_tm
        elseif model_type == :rff
            w_tr = get_params_vector(chain, "w_trend", size(modinputs.Z_trend, 2))[s, :]
            w_se = get_params_vector(chain, "w_seas", size(modinputs.Z_seas, 2))[s, :]
            f_time = modinputs.Z_trend * (w_tr .* chain[:sigma_trend].data[s]) + modinputs.Z_seas * (w_se .* chain[:sigma_seas].data[s])
        end

        # --- Likelihood Calculation ---
        for i in 1:N_obs
            a, t = modinputs.area_idx[i], modinputs.time_idx[i]
            eta = modinputs.log_offset[i] + s_eff[a] + f_time[t]

            # Add categorical covariate effects
            for k in 1:4
                if Symbol("beta_cov[$k][1]") in names(chain)
                    eta += chain[Symbol("beta_cov[$k][$(modinputs.cov_indices[i,k])]")].data[s]
                end
            end

            ll = if family == :gaussian
                logpdf(Normal(eta, chain[:sigma_y].data[s]), modinputs.y[i])
            elseif family == :poisson
                logpdf(Poisson(exp(eta)), modinputs.y[i])
            elseif family == :binomial
                logpdf(BinomialLogit(1, eta), modinputs.y[i])
            elseif family == :lognormal
                logpdf(LogNormal(eta, chain[:sigma_y].data[s]), modinputs.y[i])
            else 0.0 end
            log_lik_mat[s, i] = weights[i] * ll
        end
    end

    lppd = sum(logsumexp(log_lik_mat[:, i]) - log(N_samples) for i in 1:N_obs)
    p_waic = sum(var(log_lik_mat, dims=1))
    return -2 * (lppd - p_waic)
end


function identify_model_type(chain::MCMCChains.Chains)
    """
    Synopsis: Infers the model architecture by inspecting parameter names in the MCMC chain.
    Returns: Symbol (:ar1, :rff, :deep_gp, or :unknown)
    """
    vns = string.(names(chain))
    
    if any(occursin.(r"w1\[", vns)) && any(occursin.(r"l1", vns))
        return :deep_gp
    elseif any(occursin.(r"w_trend\[", vns)) || any(occursin.(r"Z_trend", vns))
        return :rff
    elseif any(occursin.(r"f_tm_raw\[", vns)) || any(occursin.(r"rho_tm", vns))
        return :ar1
    else
        return :unknown
    end
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
function kron_ar1_matern_sample(Ns, Nt, unique_s, ls_s, sigma_s, rho_t, sigma_t_noise, noise_vec, jitter=1e-4)
    # Spatial Matern 3/2 Precision
    k_s = Matern32Kernel() ∘ ScaleTransform(inv(ls_s))
    K_s = Symmetric(sigma_s^2 * kernelmatrix(k_s, RowVecs(unique_s)) + jitter*I )
    Q_s = sparse(inv(K_s))

    # Temporal AR1 Precision
    Q_t = ar1_precision(Nt, rho_t, sigma_t_noise)

    # Kronecker Product Q = Qt ⊗ Qs
    Q_full = Symmetric( kron(Q_t, Q_s) + jitter*I)

    # Explicitly ensure symmetry for sparse matrix before Cholesky decomposition
    # Convert to dense Matrix to avoid SparseArrays.CHOLMOD incompatibility with ForwardDiff.Dual
    L_q = cholesky(Symmetric(Matrix(Q_full) + jitter*I)) # Increased jitter

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



# --- 1. ARCHITECTURE & FAMILY TRAITS ---

abstract type ModelArchitecture end
struct GMRFArchitecture     <: ModelArchitecture end
struct RFFArchitecture      <: ModelArchitecture end
struct DeepArchitecture     <: ModelArchitecture end
struct LerouxArchitecture   <: ModelArchitecture end
struct SARArchitecture      <: ModelArchitecture end
struct SVCArchitecture      <: ModelArchitecture end
struct MosaicArchitecture   <: ModelArchitecture end
struct SPDEArchitecture     <: ModelArchitecture end
struct SpectralArchitecture <: ModelArchitecture end
struct DenseGPArchitecture  <: ModelArchitecture end
struct UnknownArchitecture  <: ModelArchitecture end

abstract type ModelFamily end
struct PoissonFamily          <: ModelFamily end
struct GaussianFamily         <: ModelFamily end
struct BinomialFamily         <: ModelFamily end
struct NegativeBinomialFamily <: ModelFamily end
struct LogNormalFamily        <: ModelFamily end
struct HurdleFamily           <: ModelFamily end
struct UnknownFamily          <: ModelFamily end


# --- 2. TRAIT IDENTIFICATION ---

function identify_traits(model_name::String)
    name = lowercase(model_name)
    
    # Identify Architecture
    arch = if occursin("simple", name) || occursin("bym2", name) || occursin("d00", name) || occursin("d01", name)
        GMRFArchitecture()
    elseif occursin("leroux", name) || occursin("d02", name)
        LerouxArchitecture()
    elseif occursin("localised", name) || occursin("d03", name)
        LerouxArchitecture() # centered Leroux variant
    elseif occursin("sar", name) || occursin("d04", name)
        SARArchitecture()
    elseif occursin("svc", name) || occursin("d06", name)
        SVCArchitecture()
    elseif occursin("mosaic", name) || occursin("c08", name) || occursin("c09", name)
        MosaicArchitecture()
    elseif occursin("deep", name) || occursin("c02", name) || occursin("c03", name) || occursin("c04", name)
        DeepArchitecture()
    elseif occursin("rff", name) || occursin("adaptive", name) || occursin("d17", name)
        RFFArchitecture()
    elseif occursin("spde", name) || occursin("c06", name) || occursin("c17", name)
        SPDEArchitecture()
    elseif occursin("fft", name) || occursin("spectral", name) || occursin("d16", name)
        SpectralArchitecture()
    elseif occursin("dense", name) || occursin("c00", name) || occursin("c01", name)
        DenseGPArchitecture()
    else
        UnknownArchitecture()
    end

    # Identify Family
    fam = if occursin("poisson", name)
        PoissonFamily()
    elseif occursin("gaussian", name)
        GaussianFamily()
    elseif occursin("binomial", name)
        BinomialFamily()
    elseif occursin("negbin", name)
        NegativeBinomialFamily()
    elseif occursin("lognormal", name)
        LogNormalFamily()
    elseif occursin("hurdle", name)
        HurdleFamily()
    else
        UnknownFamily()
    end

    return arch, fam
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


# --- 3. UNIFIED RECONSTRUCTION INTERFACE ---
function reconstruct_posteriors(model::DynamicPPL.Model, chain, modinputs; alpha=0.05)
    arch, fam = identify_traits(string.(model.f))
    return _reconstruct(arch, fam, chain, modinputs, alpha)
end

# Helper for linear predictor and LL calculation
function _process_ll_and_predictions(fam, eta, chain, modinputs, N_obs, N_samples)
    denoised = zeros(N_obs, N_samples)
    noisy = zeros(N_obs, N_samples)
    log_lik = zeros(N_samples, N_obs)
    
    # Use the model trait check logic or check the chain properties directly
    # Note: chain indexing remains standard for data extraction
    has_sig_y = :sigma_y in propertynames(chain) || any(n -> startswith(string(n), "sigma_y"), propertynames(chain))

    for j in 1:N_samples
        sig_y = has_sig_y ? Float64(chain[:sigma_y].data[j]) : 1.0
        for i in 1:N_obs
            mu_eta = eta[i, j]
            mu = if fam isa PoissonFamily; exp(mu_eta)
                 elseif fam isa BinomialFamily; 1.0 / (1.0 + exp(-mu_eta))
                 else mu_eta end
            denoised[i, j] = mu
            log_lik[j, i] = if fam isa PoissonFamily; logpdf(Poisson(mu), modinputs.y[i])
                           elseif fam isa BinomialFamily; logpdf(Bernoulli(mu), modinputs.y[i])
                           elseif fam isa GaussianFamily; logpdf(Normal(mu, sig_y), modinputs.y[i])
                           else 0.0 end
            noisy[i, j] = if fam isa GaussianFamily; mu + randn() * sig_y else mu end
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

function _extract_beta_cov(all_names, chain, modinputs, N_samples, alpha)
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
        raw_vals = get_params_vector(chain, base_name, modinputs.N_cat)
        
        if !all(raw_vals .== 0) # Only process if data was actually found
            # Reshape for summarize_array (N_categories x 1 x N_samples)
            summ = summarize_array(reshape(raw_vals', modinputs.N_cat, 1, N_samples); alpha=alpha)
            push!(results, summ)
        end
    end

    return isempty(results) ? nothing : results
end


# --- ARCHITECTURE RECONSTRUCTIONS ---
function _reconstruct(arch::GMRFArchitecture, fam::ModelFamily, chain, modinputs, alpha)
    N_obs, N_areas, N_time = length(modinputs.y), size(modinputs.Q_sp, 1), Int(maximum(modinputs.time_idx))
    N_samples = size(chain, 1)
    all_names = FlexiChains.parameters(chain)
    name_strs = string.(all_names)

    sp_struct = zeros(N_areas, N_samples)
    sp_unstruct = zeros(N_areas, N_samples)
    temporal = zeros(N_time, N_samples)
    seasonal = zeros(N_time, N_samples)
    eta = zeros(N_obs, N_samples)

    if any(occursin.("u_icar", name_strs))
        sig_sp = "sigma_sp" in name_strs ? vec(chain[:sigma_sp].data) : ones(N_samples)
        phi_sp = "phi_sp" in name_strs ? vec(chain[:phi_sp].data) : fill(0.5, N_samples)
        u_icar_data = chain[:u_icar].data
        u_iid_data = chain[:u_iid].data
        for s in 1:N_samples
            sp_struct[:, s] = sig_sp[s] .* sqrt.(phi_sp[s]) .* vec(u_icar_data[s])
            sp_unstruct[:, s] = sig_sp[s] .* sqrt.(1 - phi_sp[s]) .* vec(u_iid_data[s])
        end
    end

    if any(occursin.("f_tm_raw", name_strs))
        sig_tm = "sigma_tm" in name_strs ? vec(chain[:sigma_tm].data) : ones(N_samples)
        f_tm_data = chain[:f_tm_raw].data
        for s in 1:N_samples; temporal[:, s] = vec(f_tm_data[s]) .* sig_tm[s]; end
    end

    beta_cov_eff = _extract_beta_cov(name_strs, chain, modinputs, N_samples, alpha)

    for s in 1:N_samples
        for i in 1:N_obs
            a, t = modinputs.area_idx[i], modinputs.time_idx[i]
            eta[i, s] = modinputs.log_offset[i] + sp_struct[a, s] + sp_unstruct[a, s] + temporal[t, s]
        end
    end

    denoised, noisy, log_lik = _process_ll_and_predictions(fam, eta, chain, modinputs, N_obs, N_samples)

    return (
        spatial_structured = summarize_array(reshape(sp_struct, N_areas, 1, N_samples); alpha=alpha),
        spatial_unstructured = summarize_array(reshape(sp_unstruct, N_areas, 1, N_samples); alpha=alpha),
        spatial = summarize_array(reshape(sp_struct .+ sp_unstruct, N_areas, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(temporal, N_time, 1, N_samples); alpha=alpha),
        seasonal = summarize_array(reshape(seasonal, N_time, 1, N_samples); alpha=alpha),
        beta_cov = beta_cov_eff,
        predictions_denoised = summarize_array(reshape(denoised, N_obs, 1, N_samples); alpha=alpha),
        predictions_noisy = summarize_array(reshape(noisy, N_obs, 1, N_samples); alpha=alpha),
        waic = _compute_waic(log_lik),
        family = fam, architecture = arch
    )
end


function _reconstruct(arch::RFFArchitecture, fam::ModelFamily, chain, modinputs, alpha)
    N_obs, N_areas = length(modinputs.y), size(modinputs.W, 1)
    N_samples = size(chain, 1)
    all_names = FlexiChains.parameters(chain)
    name_strs = string.(all_names)

    # Containers
    spatial = zeros(N_areas, N_samples)
    temporal = zeros(Int(maximum(modinputs.time_idx)), N_samples)
    seasonal = zeros(Int(maximum(modinputs.time_idx)), N_samples)
    eta = zeros(N_obs, N_samples)

    # 1. RFF Spatial Component
    if any(occursin.("beta_rff", name_strs))
        # RFF logic would ideally project from frequencies
        # For summary, we check if s_eff or similar is pre-calculated or needs mapping
    end

    # 2. BYM2 if present in hybrid models
    if any(occursin.("u_icar", name_strs))
        sig_sp = "sigma_sp" in name_strs ? vec(chain[:sigma_sp].data) : ones(N_samples)
        phi_sp = "phi_sp" in name_strs ? vec(chain[:phi_sp].data) : fill(0.5, N_samples)
        u_icar_data = chain[:u_icar].data
        u_iid_data = chain[:u_iid].data

        for s in 1:N_samples
            u_icar_vec = vec(u_icar_data[s])
            u_iid_vec = vec(u_iid_data[s])
            spatial[:, s] = sig_sp[s] .* (sqrt(phi_sp[s]) .* u_icar_vec .+ sqrt(1 - phi_sp[s]) .* u_iid_vec)
        end
    end

    # 3. Categorical Covariates
    beta_cov_eff = _extract_beta_cov(name_strs, chain, modinputs, N_samples, alpha)

    # 4. Predictor Assembly
    for s in 1:N_samples
        for i in 1:N_obs
            a = modinputs.area_idx[i]
            eta[i, s] = modinputs.log_offset[i] + spatial[a, s]
        end
    end

    denoised, noisy, log_lik = _process_ll_and_predictions(fam, eta, chain, modinputs, N_obs, N_samples)

    return (
        spatial_structured = summarize_array(reshape(spatial, N_areas, 1, N_samples); alpha=alpha),
        spatial_unstructured = nothing,
        spatial = summarize_array(reshape(spatial, N_areas, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(temporal, size(temporal, 1), 1, N_samples); alpha=alpha),
        seasonal = summarize_array(reshape(seasonal, size(seasonal, 1), 1, N_samples); alpha=alpha),
        beta_cov = beta_cov_eff,
        predictions_denoised = summarize_array(reshape(denoised, N_obs, 1, N_samples); alpha=alpha),
        waic = _compute_waic(log_lik),
        family = fam,
        architecture = arch
    )
end

function _reconstruct(arch::DeepArchitecture, fam::ModelFamily, chain, modinputs, alpha)
    N_obs = length(modinputs.y)
    N_samples = size(chain, 1)
    all_names = FlexiChains.parameters(chain)
    name_strs = string.(all_names)

    eta = zeros(N_obs, N_samples)

    for s in 1:N_samples
        eta[:, s] .= modinputs.log_offset
    end

    denoised, noisy, log_lik = _process_ll_and_predictions(fam, eta, chain, modinputs, N_obs, N_samples)

    return (
        spatial_structured = nothing,
        spatial_unstructured = nothing,
        spatial = nothing,
        temporal = nothing,
        seasonal = nothing,
        predictions_denoised = summarize_array(reshape(denoised, N_obs, 1, N_samples); alpha=alpha),
        waic = _compute_waic(log_lik),
        family = fam,
        architecture = arch
    )
end



function _reconstruct(arch::LerouxArchitecture, fam::ModelFamily, chain, modinputs, alpha)
    N_obs, N_areas, N_time = length(modinputs.y), size(modinputs.Q_sp, 1), Int(maximum(modinputs.time_idx))
    N_samples = size(chain, 1)
    name_strs = string.(FlexiChains.parameters(chain))

    spatial = zeros(N_areas, N_samples)
    temporal = zeros(N_time, N_samples)
    eta = zeros(N_obs, N_samples)

    if "phi_leroux" in name_strs
        phi_data = chain[:phi_leroux].data
        for s in 1:N_samples; spatial[:, s] = vec(phi_data[s]); end
    end

    if "f_tm_raw" in name_strs
        sig_tm = "sigma_tm" in name_strs ? vec(chain[:sigma_tm].data) : ones(N_samples)
        f_tm_data = chain[:f_tm_raw].data
        for s in 1:N_samples; temporal[:, s] = vec(f_tm_data[s]) .* sig_tm[s]; end
    end

    beta_cov_eff = _extract_beta_cov(name_strs, chain, modinputs, N_samples, alpha)

    for s in 1:N_samples
        for i in 1:N_obs
            a, t = modinputs.area_idx[i], modinputs.time_idx[i]
            eta[i, s] = modinputs.log_offset[i] + spatial[a, s] + temporal[t, s]
        end
    end

    denoised, noisy, log_lik = _process_ll_and_predictions(fam, eta, chain, modinputs, N_obs, N_samples)

    return (
        spatial_structured = summarize_array(reshape(spatial, N_areas, 1, N_samples); alpha=alpha),
        spatial_unstructured = nothing, 
        spatial = summarize_array(reshape(spatial, N_areas, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(temporal, N_time, 1, N_samples); alpha=alpha),
        seasonal = nothing, 
        beta_cov = beta_cov_eff,
        predictions_denoised = summarize_array(reshape(denoised, N_obs, 1, N_samples); alpha=alpha),
        predictions_noisy = summarize_array(reshape(noisy, N_obs, 1, N_samples); alpha=alpha),
        waic = _compute_waic(log_lik), 
        family = fam, architecture = arch
    )
end


function _reconstruct(arch::SARArchitecture, fam::ModelFamily, chain, modinputs, alpha)
    N_obs, N_areas = length(modinputs.y), size(modinputs.Q_sp, 1)
    N_samples = size(chain, 1)
    all_names = FlexiChains.parameters(chain)
    name_strs = string.(all_names)

    spatial = zeros(N_areas, N_samples)
    temporal = zeros(Int(maximum(modinputs.time_idx)), N_samples)
    seasonal = zeros(Int(maximum(modinputs.time_idx)), N_samples)
    eta = zeros(N_obs, N_samples)

    if any(occursin.("s_eff", name_strs))
        s_eff_data = chain[:s_eff].data
        for s in 1:N_samples; spatial[:, s] = vec(s_eff_data[s]); end
    end

    if any(occursin.("f_tm_raw", name_strs))
        sig_tm = "sigma_tm" in name_strs ? vec(chain[:sigma_tm].data) : ones(N_samples)
        f_tm_data = chain[:f_tm_raw].data
        for s in 1:N_samples; temporal[:, s] = vec(f_tm_data[s]) .* sig_tm[s]; end
    end

    beta_cov_eff = _extract_beta_cov(name_strs, chain, modinputs, N_samples, alpha)

    for s in 1:N_samples
        for i in 1:N_obs
            a, t = modinputs.area_idx[i], modinputs.time_idx[i]
            eta[i, s] = modinputs.log_offset[i] + spatial[a, s] + temporal[t, s]
        end
    end

    denoised, noisy, log_lik = _process_ll_and_predictions(fam, eta, chain, modinputs, N_obs, N_samples)

    return (
        spatial_structured = summarize_array(reshape(spatial, N_areas, 1, N_samples); alpha=alpha),
        spatial_unstructured = nothing,
        spatial = summarize_array(reshape(spatial, N_areas, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(temporal, size(temporal, 1), 1, N_samples); alpha=alpha),
        seasonal = summarize_array(reshape(seasonal, size(seasonal, 1), 1, N_samples); alpha=alpha),
        beta_cov = beta_cov_eff,
        predictions_denoised = summarize_array(reshape(denoised, N_obs, 1, N_samples); alpha=alpha),
        predictions_noisy = summarize_array(reshape(noisy, N_obs, 1, N_samples); alpha=alpha),
        waic = _compute_waic(log_lik),
        family = fam,
        architecture = arch
    )
end

function _reconstruct(arch::SVCArchitecture, fam::ModelFamily, chain, modinputs, alpha)
    N_obs, N_areas = length(modinputs.y), size(modinputs.Q_sp, 1)
    N_samples = size(chain, 1)
    name_strs = string.(FlexiChains.parameters(chain))
    
    spatial = zeros(N_areas, N_samples)
    eta = zeros(N_obs, N_samples)

    if "s_eff" in name_strs
        s_data = chain[:s_eff].data
        for s in 1:N_samples; spatial[:, s] = vec(s_data[s]); end
    end

    beta_cov_eff = _extract_beta_cov(name_strs, chain, modinputs, N_samples, alpha)
    for s in 1:N_samples
        eta[:, s] .= modinputs.log_offset .+ spatial[modinputs.area_idx, s]
    end

    denoised, noisy, log_lik = _process_ll_and_predictions(fam, eta, chain, modinputs, N_obs, N_samples)

    return (
        spatial_structured = summarize_array(reshape(spatial, N_areas, 1, N_samples); alpha=alpha),
        spatial_unstructured = nothing,
        spatial = summarize_array(reshape(spatial, N_areas, 1, N_samples); alpha=alpha),
        temporal = nothing,
        seasonal = nothing,
        beta_cov = beta_cov_eff,
        predictions_denoised = summarize_array(reshape(denoised, N_obs, 1, N_samples); alpha=alpha),
        predictions_noisy = summarize_array(reshape(noisy, N_obs, 1, N_samples); alpha=alpha),
        waic = _compute_waic(log_lik), 
        family = fam, architecture = arch
    )
end


function _reconstruct(arch::MosaicArchitecture, fam::ModelFamily, chain, modinputs, alpha)
    N_obs, N_areas = length(modinputs.y), size(modinputs.W, 1)
    N_samples = size(chain, 1)
    all_names = FlexiChains.parameters(chain)
    name_strs = string.(all_names)
    eta = zeros(N_obs, N_samples)

    beta_cov_eff = _extract_beta_cov(name_strs, chain, modinputs, N_samples, alpha)

    for s in 1:N_samples; eta[:, s] .= modinputs.log_offset; end

    denoised, noisy, log_lik = _process_ll_and_predictions(fam, eta, chain, modinputs, N_obs, N_samples)

    return (
        beta_cov = beta_cov_eff,
        predictions_denoised = summarize_array(reshape(denoised, N_obs, 1, N_samples); alpha=alpha),
        predictions_noisy = summarize_array(reshape(noisy, N_obs, 1, N_samples); alpha=alpha),
        waic = _compute_waic(log_lik),
        family = fam,
        architecture = arch
    )
end


function _reconstruct(arch::Union{SPDEArchitecture, SpectralArchitecture, DenseGPArchitecture}, fam::ModelFamily, chain, modinputs, alpha)
    N_obs, N_samples = length(modinputs.y), size(chain, 1)
    name_strs = string.(FlexiChains.parameters(chain))
    eta = zeros(N_obs, N_samples)
    temporal = zeros(Int(maximum(modinputs.time_idx)), N_samples)

    target = "f_latent" in name_strs ? :f_latent : "f_gp" in name_strs ? :f_gp : "s_eff" in name_strs ? :s_eff : nothing
    
    if !isnothing(target)
        data = chain[target].data
        for s in 1:N_samples; eta[:, s] = modinputs.log_offset .+ vec(data[s]); end
    else
        for s in 1:N_samples; eta[:, s] .= modinputs.log_offset; end
    end

    if "f_tm_raw" in name_strs
        sig_tm = "sigma_tm" in name_strs ? vec(chain[:sigma_tm].data) : ones(N_samples)
        f_tm_data = chain[:f_tm_raw].data
        for s in 1:N_samples; temporal[:, s] = vec(f_tm_data[s]) .* sig_tm[s]; end
    end

    beta_cov_eff = _extract_beta_cov(name_strs, chain, modinputs, N_samples, alpha)
    denoised, noisy, log_lik = _process_ll_and_predictions(fam, eta, chain, modinputs, N_obs, N_samples)

    return (
        spatial_structured = nothing,
        spatial_unstructured = nothing,
        spatial = nothing, 
        temporal = summarize_array(reshape(temporal, size(temporal,1), 1, N_samples); alpha=alpha),
        seasonal = nothing,
        beta_cov = beta_cov_eff,
        predictions_denoised = summarize_array(reshape(denoised, N_obs, 1, N_samples); alpha=alpha),
        predictions_noisy = summarize_array(reshape(noisy, N_obs, 1, N_samples); alpha=alpha),
        waic = _compute_waic(log_lik), 
        family = fam, architecture = arch
    )
end



function model_results_comprehensive(model, chain, modinputs, areal_units; alpha=0.05, time_slice=1)
    pstats = reconstruct_posteriors(model, chain, modinputs; alpha=alpha)
    y_obs = Float64.(modinputs.y)
    y_pred = vec(pstats.predictions_denoised.mean)

    # Metrics
    rmse = sqrt(mean((y_obs .- y_pred).^2))
    r2 = length(unique(y_pred)) > 1 ? cor(y_obs, y_pred)^2 : 0.0
    waic_val = pstats.waic

    ss = summarystats(chain) 
    mean_ess_bulk = mean( x for x in Array(ss[:,stat=At(:ess_bulk)] ) if !isnan(x)  )
    mean_ess_tail = mean( x for x in Array(ss[:,stat=At(:ess_tail)] ) if !isnan(x)  )
    mean_rhat = mean( x for x in Array(ss[:,stat=At(:rhat)] ) if !isnan(x)   )

    metrics = (rmse=rmse, r2=r2, waic=waic_val, 
        mean_ess_bulk=mean_ess_bulk, mean_ess_tail=mean_ess_tail, mean_rhat=mean_rhat)
    display(metrics)

    # Plots
    plt_ppc = scatter(y_pred, y_obs, alpha=0.4, label="Obs vs Pred", title="PPC (RMSE: $(round(rmse, digits=3)), WAIC: $(round(waic_val, digits=1)))", xlabel="Pred", ylabel="Obs")
    plot!(plt_ppc, [minimum(y_obs), maximum(y_obs)], [minimum(y_obs), maximum(y_obs)], ls=:dash, lc=:red, label="Identity")

    plt_tm = nothing
    if haskey(pstats, :temporal) && !isnothing(pstats.temporal)
        plt_tm = plot(pstats.temporal.mean, 
            ribbon=(pstats.temporal.mean .- pstats.temporal.lower, pstats.temporal.upper .- pstats.temporal.mean), 
            title="Temporal Trend", xlabel="Time", ylabel="Effect", legend=false, fillalpha=0.3)
    end

    plt_seas = nothing
    if haskey(pstats, :seasonal) && !isnothing(pstats.seasonal)
        plt_seas = plot(pstats.seasonal.mean, 
            ribbon=(pstats.seasonal.mean .- pstats.seasonal.lower, pstats.seasonal.upper .- pstats.seasonal.mean), 
            title="Seasonal Trend", xlabel="Time", ylabel="Effect", legend=false, fillalpha=0.3)
    end
 
    plt_sp = plot_posterior_results(pstats, modinputs, areal_units; effect=:spatial)
    title!(plt_sp, "Main Spatial Effect")

    plt_sp_struct = nothing
    if haskey(pstats, :spatial_structured) && !isnothing(pstats.spatial_structured)
        plt_sp_struct = plot_posterior_results(pstats, modinputs, areal_units; effect=:spatial_structured )
        title!(plt_sp_struct, "Structured Spatial Effect")
    end

    plt_sp_unstruct = nothing
    if haskey(pstats, :spatial_unstructured) && !isnothing(pstats.spatial_unstructured)
        plt_sp_unstruct = plot_posterior_results(pstats, modinputs, areal_units; effect=:spatial_unstructured )
        title!(plt_sp_unstruct, "Unstructured Spatial Effect")
    end


    plt_st_denoised = plot_posterior_results(pstats, modinputs, areal_units; effect=:predictions_denoised, time_slice=time_slice)
    title!(plt_st_denoised, "Denoised Predictions (T=$time_slice)")

    plt_st_noisy = plot_posterior_results(pstats, modinputs, areal_units; effect=:predictions_noisy, time_slice=time_slice)
    title!(plt_st_noisy, "Noisy Predictions (T=$time_slice)")

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


