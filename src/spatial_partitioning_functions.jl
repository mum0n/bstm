

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





function generate_sim_data(n_pts, n_time; rndseed=42)
    # Generates synthetic data across discrete and continuous frameworks.
    # Maintained existing structure; appended nested/joint variables (u, z).
    Random.seed!(rndseed)
    unique_pts = [(rand() * 10, rand() * 10) for _ in 1:n_pts]
    pts_full_dataset = repeat(unique_pts, n_time)
    n_total = n_pts * n_time
    time_idx = repeat(1:n_time, inner=n_pts)
    weights = ones(n_total)
    trials = ones(Int, n_total)
    cov_indices = rand(1:3, n_total)
    spatial_effect = [sin(p[1]/2) + cos(p[2]/2) for p in unique_pts]
    spatial_effect_long = repeat(spatial_effect, n_time)
    temporal_effect = sin.(time_idx)
    y_sim = 1.5 .* spatial_effect_long + 1.0 .* temporal_effect + randn(n_total) * 0.5
    y_binary = y_sim .> (mean(y_sim) + 0.5)
    y_counts = abs.(Int.(round.(y_sim))) * 100
    cov_continuous = randn(length(y_sim), 3)
    cov_indices_mat = hcat(cov_indices, cov_indices, cov_indices, cov_indices)
    trials_sim = ones(Int, length(y_binary))
    class1_sim = rand(1:13, length(y_binary))
    class2_sim = rand(1:2, length(y_binary))
    weights_sim = ones(Float64, length(y_binary))
    # New GP suite variables
    z_obs = randn(n_total)
    u_obs = randn(n_total, 3)
    return (
        pts=pts_full_dataset, y_sim=y_sim, y_binary=y_binary, y_counts=y_counts,
        time_idx=time_idx, weights=weights, trials=trials,
        cov_indices=cov_indices, cov_continuous=cov_continuous, cov_indices_mat=cov_indices_mat,
        trials_sim=trials_sim, class1_sim=class1_sim, class2_sim=class2_sim, weights_sim=weights_sim,
        z_obs=z_obs, u_obs=u_obs
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
    D = diagm(vec(sum(W, dims=2)))
 
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

