"""
    partitioning.jl

Spatial and spatiotemporal domain decomposition, Voronoi/hexagonal tessellation,
areal unit assignment, graph connectivity, and spatial block cross-validation for BSTM.

Version: v1.0.0
"""

# -----------------------------------------------------------------------------
# Section 1: Geometric Primitives & Boundary Utilities
# -----------------------------------------------------------------------------

"""
    expand_hull(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, buffer_dist::Real)

Computes the convex hull of 2D points `(s_x, s_y)` and expands it by `buffer_dist`
using `LibGEOS.buffer`. Returns a `LibGEOS` polygon geometry.
"""
function expand_hull(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, buffer_dist::Real)
    if isempty(s_x) || isempty(s_y)
        return LibGEOS.Polygon([[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]])
    end
    coords_vec = [[Float64(s_x[i]), Float64(s_y[i])] for i in 1:length(s_x)]
    points_geom = LibGEOS.MultiPoint(coords_vec)
    hull = LibGEOS.convexhull(points_geom)
    if buffer_dist > 0.0
        return LibGEOS.buffer(hull, Float64(buffer_dist))
    else
        return hull
    end
end

expand_hull(coords::Vector{<:Tuple{Real, Real}}, buffer_dist::Real) =
    expand_hull([p[1] for p in coords], [p[2] for p in coords], buffer_dist)


"""
    get_polygon_area(poly_coords::AbstractVector)

Calculates the 2D surface area of a polygon using the Shoelace formula.
Coordinates are expected as a vector of `(x, y)` tuples or vectors.
"""
function get_polygon_area(poly_coords::AbstractVector)
    valid_pts = Tuple{Float64, Float64}[]
    for p in poly_coords
        x, y = float(p[1]), float(p[2])
        if !isnan(x) && !isinf(x) && !isnan(y) && !isinf(y)
            push!(valid_pts, (x, y))
        end
    end

    if length(valid_pts) > 1 && valid_pts[1] == valid_pts[end]
        valid_pts = valid_pts[1:end-1]
    end

    if length(valid_pts) < 3
        return 0.0
    end

    x = [p[1] for p in valid_pts]
    y = [p[2] for p in valid_pts]

    # Shoelace formula
    return 0.5 * abs(dot(x, circshift(y, 1)) - dot(y, circshift(x, 1)))
end

get_polygon_area(s_x::AbstractVector, s_y::AbstractVector) = get_polygon_area(tuple.(s_x, s_y))


"""
    is_valid_polygon_coords(poly_coords)

Validates whether a collection of vertices contains at least 3 non-NaN/non-Inf points.
"""
function is_valid_polygon_coords(poly_coords)
    valid_pts = [
        p for p in poly_coords
        if !isnan(p[1]) && !isinf(p[1]) && !isnan(p[2]) && !isinf(p[2])
    ]
    return length(valid_pts) >= 3
end


"""
    get_coords_from_geom(geom)

Extracts coordinate tuples `Vector{Tuple{Float64, Float64}}` from `LibGEOS` geometries
(`Point`, `Polygon`, `MultiPolygon`, `LineString`, `LinearRing`).
"""
function get_coords_from_geom(geom)
    coords = Tuple{Float64, Float64}[]
    if geom === nothing
        return coords
    end
    local type_id = -1
    try
        type_id = LibGEOS.geomTypeId(geom)
        if type_id == LibGEOS.GEOS_POINT
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
            n_geoms = LibGEOS.numGeometries(geom)
            for i in 1:n_geoms
                poly = LibGEOS.getGeometryN(geom, i)
                ring = LibGEOS.exteriorRing(poly)
                n = LibGEOS.numPoints(ring)
                for j in 1:n
                    p = LibGEOS.getPoint(ring, j)
                    seq = LibGEOS.getCoordSeq(p)
                    push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
                end
                if i < n_geoms
                    push!(coords, (NaN, NaN))
                end
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
        @warn "Coordinate extraction failed for LibGEOS type $type_id: $e"
    end
    return coords
end


"""
    get_kde_seeds(coords, target_u)

Fast density-weighted seeding for spatial tessellation. Uses KDTree to estimate local
point densities in O(N log N) without allocating large pairwise distance matrices.
"""
function get_kde_seeds(coords::Vector{<:Tuple{Real, Real}}, target_u::Integer)
    u_pts = unique(coords)
    n = length(u_pts)
    if n == 0
        return Tuple{Float64, Float64}[]
    elseif n <= target_u
        return [(Float64(p[1]), Float64(p[2])) for p in u_pts]
    end

    # Fast KD-Tree 5-nearest-neighbors density proxy
    pts_mat = hcat([[Float64(p[1]), Float64(p[2])] for p in u_pts]...)
    tree = KDTree(pts_mat)
    k_eval = min(6, n)
    _, dists = knn(tree, pts_mat, k_eval)

    # Inverse mean distance as density weight
    weights = [1.0 / (mean(d[2:end]) + 1e-6) for d in dists]
    idx = StatsBase.sample(1:n, Weights(weights), min(target_u, n), replace=false)
    return [(Float64(u_pts[i][1]), Float64(u_pts[i][2])) for i in idx]
end


"""
    get_voronoi_polygons_and_edges(centroids, hull_geom, tol=1e-7)

Generates Voronoi polygons for `centroids`, clips them to `hull_geom` via `LibGEOS`,
and extracts contiguous neighbor edges. Handles edge cases for 0, 1, and 2 centroids.
"""
function get_voronoi_polygons_and_edges(
    centroids::Vector{<:Tuple{Real, Real}}, hull_geom, tol::Real=1e-7
)
    n_c = length(centroids)
    if n_c == 0
        return Tuple{Float64, Float64}[][],
               Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}[]
    elseif n_c == 1
        return [get_coords_from_geom(hull_geom)],
               Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}[]
    elseif n_c == 2
        p1 = (Float64(centroids[1][1]), Float64(centroids[1][2]))
        p2 = (Float64(centroids[2][1]), Float64(centroids[2][2]))
        mid = ((p1[1] + p2[1]) / 2.0, (p1[2] + p2[2]) / 2.0)
        dx, dy = p2[1] - p1[1], p2[2] - p1[2]
        px, py = -dy, dx

        env = LibGEOS.envelope(hull_geom)
        hull_bbox_coords = get_coords_from_geom(env)
        L = if !isempty(hull_bbox_coords)
            min_x, max_x = extrema(c[1] for c in hull_bbox_coords)
            min_y, max_y = extrema(c[2] for c in hull_bbox_coords)
            2.0 * sqrt((max_x - min_x)^2 + (max_y - min_y)^2) + 1.0
        else
            1e7
        end

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

    # Deduplicate centroids
    u_centroids = unique(centroids)
    if length(u_centroids) < n_c
        u_polys, u_edges = get_voronoi_polygons_and_edges(u_centroids, hull_geom, tol)
        return [u_polys[findfirst(==(c), u_centroids)] for c in centroids], u_edges
    end

    # 3+ centroids: Delaunay -> Voronoi
    pts_dt = [(Float64(c[1]), Float64(c[2])) for c in centroids]
    tri = triangulate(pts_dt)
    hull_coords = get_coords_from_geom(hull_geom)
    xs = [p[1] for p in hull_coords if !isnan(p[1])]
    ys = [p[2] for p in hull_coords if !isnan(p[2])]
    if isempty(xs) || isempty(ys)
        return [Tuple{Float64, Float64}[] for _ in 1:length(centroids)], []
    end
    
    bbox = (minimum(xs), maximum(xs), minimum(ys), maximum(ys))
    vorn = voronoi(tri)
    final_coords = [Tuple{Float64, Float64}[] for _ in 1:length(centroids)]
    valid_geoms = Dict{Int, Any}()

    for i in each_generator(vorn)
        if i < 1 || i > length(centroids)
            continue
        end
        vertices = get_polygon_coordinates(vorn, i, bbox)
        if !isempty(vertices)
            poly_pts = [[v[1], v[2]] for v in vertices]
            if poly_pts[1] != poly_pts[end]
                push!(poly_pts, poly_pts[1])
            end
            try
                lg_poly = LibGEOS.Polygon([poly_pts])
                clipped = LibGEOS.intersection(lg_poly, hull_geom)
                if !LibGEOS.isEmpty(clipped) &&
   LibGEOS.geomTypeId(clipped) in [LibGEOS.GEOS_POLYGON, LibGEOS.GEOS_MULTIPOLYGON]
                    final_coords[i] = get_coords_from_geom(clipped)
                    valid_geoms[i] = clipped
                end
            catch
            end
        end
    end

    v_edges = Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}[]
    active_ids = sort(collect(keys(valid_geoms)))
    for idx in 1:length(active_ids)
        i = active_ids[idx]
        g1 = valid_geoms[i]
        for jdx in idx+1:length(active_ids)
            j = active_ids[jdx]
            g2 = valid_geoms[j]
            if LibGEOS.touches(g1, g2)
                push!(v_edges, ((Float64(centroids[i][1]), Float64(centroids[i][2])),
                                (Float64(centroids[j][1]), Float64(centroids[j][2]))))
            else
                g1_b = LibGEOS.buffer(g1, Float64(tol))
                if LibGEOS.intersects(g1_b, g2)
                    push!(v_edges, ((Float64(centroids[i][1]), Float64(centroids[i][2])),
                                    (Float64(centroids[j][1]), Float64(centroids[j][2]))))
                end
            end
        end
    end

    return final_coords, v_edges
end


# -----------------------------------------------------------------------------
# Section 2: Spatial Tessellation & Centroid Algorithms
# -----------------------------------------------------------------------------

"""
    get_cvt_centroids(s_x, s_y, cfg, hull_geom)

Centroidal Voronoi Tessellation (CVT) via Lloyd's relaxation algorithm.
Iteratively relaxes centroids to the geometric center of their Voronoi polygons.
"""
function get_cvt_centroids(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom
)
    coords = tuple.(Float64.(s_x), Float64.(s_y))
    if length(coords) <= cfg.min_total_arealunits
        return [(mean(s_x), mean(s_y))], "not_enough_points_to_tessellate"
    end

    u_pts = unique(coords)
    idx = StatsBase.sample(1:length(u_pts), min(cfg.target, length(u_pts)), replace=false)
    curr_centroids = [u_pts[i] for i in idx]
    termination_reason = "max_iterations"
    last_mean_density, last_cv = 0.0, 0.0

    pts_mat = hcat([[p[1], p[2]] for p in coords]...)

    for iter in 1:100
        polys, _ = get_voronoi_polygons_and_edges(curr_centroids, hull_geom)
        new_centroids = Tuple{Float64, Float64}[]
        shifts = Float64[]

        for i in 1:length(polys)
            poly_coords = polys[i]
            area = get_polygon_area(poly_coords)

            if length(poly_coords) > 2 && area >= cfg.min_area && area <= cfg.max_area
                lg_poly = LibGEOS.Polygon([[[p[1], p[2]] for p in poly_coords]])
                cent_geom = LibGEOS.centroid(lg_poly)
                seq = LibGEOS.getCoordSeq(cent_geom)
                new_c = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
                push!(shifts, sqrt(sum((new_c .- curr_centroids[i]).^2)))
                push!(new_centroids, new_c)
            else
                push!(new_centroids, curr_centroids[i])
            end
        end

        if isempty(shifts) || mean(shifts) < cfg.tolerance
            termination_reason = "convergence"
            break
        end

        # Fast KDTree assignment
        c_mat = hcat([[c[1], c[2]] for c in new_centroids]...)
        tree = KDTree(c_mat)
        assigns, _ = knn(tree, pts_mat, 1)
        assign_indices = [a[1] for a in assigns]
        counts = [count(==(i), assign_indices) for i in 1:length(new_centroids)]

        if isempty(counts)
            termination_reason = "no_units_formed"
            break
        end

        curr_mean_density = mean(counts)
        if abs(curr_mean_density - last_mean_density) < cfg.tolerance && iter > 1
            termination_reason = "density_convergence"
            break
        end
        last_mean_density = curr_mean_density

        cv_val = std(counts) / (mean(counts) + 1e-9)
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


"""
    get_kvt_centroids(s_x, s_y, cfg, hull_geom)

K-means Voronoi Tessellation (KVT) moves centroids toward the center of mass
of data points inside each cell, balancing point density.
"""
function get_kvt_centroids(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom
)
    coords = tuple.(Float64.(s_x), Float64.(s_y))
    if length(coords) <= cfg.min_total_arealunits
        return [(mean(s_x), mean(s_y))], "not_enough_points_to_tessellate"
    end

    u_pts = unique(coords)
    idx_init = StatsBase.sample(1:length(u_pts), min(cfg.target, length(u_pts)), replace=false)
    c_iter = [u_pts[i] for i in idx_init]
    data = tuple.(coords, cfg.t_idx)
    pts_mat = hcat([[p[1], p[2]] for p in coords]...)

    damping = 0.7
    termination_reason = "max_iterations"
    last_mean_density, last_cv = 0.0, 0.0

    for iter in 1:100
        old_centroids = copy(c_iter)
        c_mat = hcat([[c[1], c[2]] for c in c_iter]...)
        tree = KDTree(c_mat)
        assigns_knn, _ = knn(tree, pts_mat, 1)
        assigns = [a[1] for a in assigns_knn]

        polys_coords, _ = get_voronoi_polygons_and_edges(c_iter, hull_geom)

        for k in 1:length(c_iter)
            idx_cluster = findall(==(k), assigns)
            ts_count = length(unique([data[j][2] for j in idx_cluster]))
            area = k <= length(polys_coords) ? get_polygon_area(polys_coords[k]) : 0.0
            area_ok = (area > 0) && area >= cfg.min_area && area <= cfg.max_area

            if !isempty(idx_cluster) && length(idx_cluster) >= cfg.min_points &&
   ts_count >= cfg.min_time_slices && area_ok
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

        curr_mean_density = mean(counts)
        if abs(curr_mean_density - last_mean_density) < cfg.tolerance && iter > 1
            termination_reason = "density_convergence"
            break
        end
        last_mean_density = curr_mean_density

        cv_val = std(counts) / (mean(counts) + 1e-9)
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


"""
    get_qvt_centroids(s_x, s_y, cfg, hull_geom)

Quadtree Voronoi Tessellation (QVT) recursively subdivides regions into quadrants
based on local coordinate medians, adapting to high-density clusters.
"""
function get_qvt_centroids(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom
)
    coords = tuple.(Float64.(s_x), Float64.(s_y))
    if length(coords) <= cfg.min_total_arealunits
        return [(mean(s_x), mean(s_y))], "not_enough_points_to_tessellate"
    end

    data = tuple.(coords, cfg.t_idx)
    regions = [data]
    unsplittable = Set{UInt64}()
    effective_min_p = max(1, cfg.min_points)

    if length(data) < 2 * effective_min_p
        return [(mean(p[1][1] for p in data), mean(p[1][2] for p in data))],
       "initial_data_too_small_to_tessellate"
    end

    termination_reason = "max_units_reached"
    last_mean_density, last_cv = 0.0, 0.0
    cnt = 0

    while length(regions) < cfg.max_total_arealunits
        cnt += 1
        counts = length.(regions)
        curr_mean_density = mean(counts)
        cv_val = std(counts) / (curr_mean_density + 1e-9)

        if cnt > 3
            if last_mean_density > 0.0 &&
   (abs(curr_mean_density - last_mean_density) < cfg.tolerance ||
    abs(cv_val - last_cv) < cfg.tolerance)
                if length(regions) >= cfg.target && all(c -> c <= cfg.max_points, counts)
                    termination_reason = "converged_constraints_satisfied"
                    break
                elseif abs(cv_val - cfg.target_cv) < cfg.tolerance
                    termination_reason = "converged_target_cv"
                    break
                end
            end
        end

        last_mean_density = curr_mean_density
        last_cv = cv_val

        viable_indices = findall(
    r -> length(r) >= max(2, effective_min_p) && objectid(r) ∉ unsplittable, regions
)
        if isempty(viable_indices)
            break
        end

        target_idx = viable_indices[argmax([length(regions[i]) for i in viable_indices])]
        target_region = regions[target_idx]
        xs_r = [p[1][1] for p in target_region]
        ys_r = [p[1][2] for p in target_region]

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
            mid = length(target_region) ÷ 2
            r_splits = [target_region[1:mid], target_region[mid+1:end], [], []]
        end

        valid_splits = filter(r -> length(r) >= effective_min_p, r_splits)
        if length(valid_splits) < 2
            push!(unsplittable, objectid(target_region))
            continue
        end

        deleteat!(regions, target_idx)
        append!(regions, valid_splits)
    end

    final_centroids = [(mean(p[1][1] for p in r), mean(p[1][2] for p in r)) for r in regions]
    return final_centroids, termination_reason
end


"""
    get_bvt_centroids(s_x, s_y, cfg, hull_geom)

Binary Voronoi Tessellation (BVT) recursively bisects regions along their principal axis of
  maximum variance.
"""
function get_bvt_centroids(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom
)
    coords = tuple.(Float64.(s_x), Float64.(s_y))
    if length(coords) <= cfg.min_total_arealunits
        return [(mean(s_x), mean(s_y))], "not_enough_points_to_tessellate"
    end

    data = tuple.(coords, cfg.t_idx)
    regions = [data]
    effective_min_p = max(1, cfg.min_points)

    if length(data) < 2 * effective_min_p
        return [(mean(p[1][1] for p in data), mean(p[1][2] for p in data))],
       "initial_data_too_small_to_tessellate"
    end

    unsplittable = Set{UInt64}()
    termination_reason = "max_units_reached"
    last_mean_density, last_cv = 0.0, 0.0
    cnt = 0

    while length(regions) < cfg.max_total_arealunits
        cnt += 1
        counts = length.(regions)
        curr_mean_density = mean(counts)
        cv_val = std(counts) / (curr_mean_density + 1e-9)

        if cnt > 3
            if last_mean_density > 0.0 &&
   (abs(curr_mean_density - last_mean_density) < cfg.tolerance ||
    abs(cv_val - last_cv) < cfg.tolerance)
                if length(regions) >= cfg.target && all(c -> c <= cfg.max_points, counts)
                    termination_reason = "converged_constraints_satisfied"
                    break
                elseif (abs(cv_val - cfg.target_cv) < cfg.tolerance)
                    termination_reason = "converged_target_cv"
                    break
                end
            end
        end

        last_mean_density = curr_mean_density
        last_cv = cv_val

        viable_indices = findall(
    r -> length(r) >= max(2, effective_min_p) && objectid(r) ∉ unsplittable, regions
)
        if cnt > 3 && isempty(viable_indices)
            termination_reason = "cannot_split_further"
            break
        end

        violators = filter(i -> length(regions[i]) > cfg.max_points, viable_indices)
        must_split = length(regions) < cfg.min_total_arealunits
        want_split = length(regions) < cfg.target || !isempty(violators)
        candidates = if must_split || want_split
    isempty(violators) ? viable_indices : violators
else
    []
end

        if isempty(candidates)
            termination_reason = "constraints_satisfied"; break
        end

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
            mid = length(target) ÷ 2
            r1, r2 = target[1:mid], target[mid+1:end]
        end

        v1 = length(r1) >= effective_min_p &&
     length(unique([p[2] for p in r1])) >= cfg.min_time_slices
        v2 = length(r2) >= effective_min_p &&
     length(unique([p[2] for p in r2])) >= cfg.min_time_slices

        if !v1 || !v2
             push!(unsplittable, objectid(target))
             continue
        end

        tentative_regions = copy(regions)
        deleteat!(tentative_regions, target_idx)
        push!(tentative_regions, r1, r2)

        candidate_centroids = [
    (mean(p[1][1] for p in r), mean(p[1][2] for p in r))
    for r in tentative_regions
]
        polys_coords, _ = get_voronoi_polygons_and_edges(candidate_centroids, hull_geom)

        area_violation = any(
            p_coords -> !is_valid_polygon_coords(p_coords) ||
            get_polygon_area(p_coords) < cfg.min_area,
            polys_coords
        )

        if area_violation && length(tentative_regions) > cfg.min_total_arealunits
             push!(unsplittable, objectid(target))
             continue
        end

        regions = tentative_regions
    end

    final_centroids_candidate = [
    (mean(p[1][1] for p in r), mean(p[1][2] for p in r))
    for r in regions
]

    if length(final_centroids_candidate) < cfg.min_total_arealunits
        all_pts_x = [p[1][1] for p in data]
        all_pts_y = [p[1][2] for p in data]
        return [(mean(all_pts_x), mean(all_pts_y))], "insufficient_units_error"
    else
        return final_centroids_candidate, termination_reason
    end
end


"""
    get_hvt_centroids(s_x, s_y, cfg, hull_geom; max_iter=500)

Hierarchical Voronoi Tessellation (HVT) combines initial k-means seeding
with adaptive splitting of overcrowded partitions.
"""
function get_hvt_centroids(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom; max_iter=500
)
    coords = tuple.(Float64.(s_x), Float64.(s_y))
    dist(p1, p2) = sqrt((p1[1]-p2[1])^2 + (p1[2]-p2[2])^2)

    pts_matrix = hcat([[p[1], p[2]] for p in coords]...)
    k_target = max(1, cfg.min_total_arealunits)

    R = Clustering.kmeans(pts_matrix, k_target)
    curr_centroids = [(R.centers[1, i], R.centers[2, i]) for i in 1:size(R.centers, 2)]

    last_mean_density, last_cv = 0.0, 0.0
    status = "max_iterations_reached"

    for i in 1:max_iter
        c_mat = hcat([[c[1], c[2]] for c in curr_centroids]...)
        tree = KDTree(c_mat)
        assigns_knn, _ = knn(tree, pts_matrix, 1)
        s_idx = [a[1] for a in assigns_knn]
        counts = [count(==(k), s_idx) for k in 1:length(curr_centroids)]

        curr_mean_density = mean(counts)
        cv_val = std(counts) / (curr_mean_density + 1e-9)

        if i > 5
            if abs(curr_mean_density - last_mean_density) < cfg.tolerance ||
   abs(cv_val - last_cv) < cfg.tolerance
                if length(curr_centroids) >= cfg.target && all(c -> c <= cfg.max_points, counts)
                    status = "converged_constraints_satisfied"
                    break
                elseif abs(cv_val - cfg.target_cv) < cfg.tolerance
                    status = "converged_target_cv"
                    break
                end
            end
        end

        last_mean_density = curr_mean_density
        last_cv = cv_val

        # Refinement step (Lloyd's update)
        new_centroids = Tuple{Float64, Float64}[]
        for k in 1:length(curr_centroids)
            group_pts = coords[s_idx .== k]
            if !isempty(group_pts)
                push!(new_centroids, (
    mean(p[1] for p in group_pts), mean(p[2] for p in group_pts)
))
            else
                push!(new_centroids, curr_centroids[k])
            end
        end

        if all(dist(new_centroids[j], curr_centroids[j]) < cfg.tolerance
       for j in 1:length(curr_centroids))
             if length(curr_centroids) < cfg.max_total_arealunits &&
   (length(curr_centroids) < cfg.target || any(counts .> cfg.max_points))
                 idx_to_split = argmax(counts)
                 group_pts = coords[s_idx .== idx_to_split]
                 if length(group_pts) >= 2 * cfg.min_points
                    new_seeds = [
                        (mean(p[1] for p in group_pts) * 0.99,
                         mean(p[2] for p in group_pts) * 0.99),
                        (mean(p[1] for p in group_pts) * 1.01,
                         mean(p[2] for p in group_pts) * 1.01)
                    ]
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

Agglomerative Voronoi Tessellation (AVT) begins with fine-grained partitions and merges
underpopulated or starved units bottom-up until area and point count constraints are satisfied.
"""
function get_avt_centroids(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom
)
    coords = tuple.(Float64.(s_x), Float64.(s_y))
    if length(coords) <= cfg.min_total_arealunits
        return [(mean(p[1] for p in coords), mean(p[2] for p in coords))],
       "not_enough_points_to_tessellate"
    end

    u_pts = unique(coords)
    c_init = get_kde_seeds(u_pts, min(length(u_pts), cfg.max_total_arealunits))
    data = tuple.(coords, cfg.t_idx)
    curr_c = [SVector{2, Float64}(c) for c in c_init]

    termination_reason = "min_units_reached"
    last_mean_density, last_cv = 0.0, 0.0

    while length(curr_c) > cfg.min_total_arealunits
        assigns = [Int[] for _ in 1:length(curr_c)]
        for i in 1:length(data)
            d_pt = data[i][1]
            dist_idx = argmin([sum((d_pt .- c).^2) for c in curr_c])
            push!(assigns[dist_idx], i)
        end
        counts = length.(assigns)

        polys_coords, _ = get_voronoi_polygons_and_edges([Tuple(c) for c in curr_c], hull_geom)
        areas = fill(0.0, length(curr_c))
        for i in 1:min(length(curr_c), length(polys_coords))
            areas[i] = get_polygon_area(polys_coords[i])
        end

        violators = Int[]
        for k in 1:length(curr_c)
            ts_count = length(unique([data[idx][2] for idx in assigns[k]]))
            is_invalid_count = counts[k] < cfg.min_points
            is_invalid_time = ts_count < cfg.min_time_slices
            is_invalid_area = (areas[k] > 0 && areas[k] < cfg.min_area) ||
                  (areas[k] > cfg.max_area)
            if is_invalid_count || is_invalid_time || is_invalid_area
                push!(violators, k)
            end
        end

        curr_mean_density = mean(counts)
        cv_val = std(counts) / (mean(counts) + 1e-9)
        if last_mean_density > 0.0 &&
   (abs(curr_mean_density - last_mean_density) < cfg.tolerance ||
    abs(cv_val - last_cv) < cfg.tolerance)
            termination_reason = "tolerance_reached"
            break
        end
        last_mean_density = curr_mean_density
        last_cv = cv_val

        candidates_indices = isempty(violators) ? collect(1:length(curr_c)) : violators
        v_counts = [counts[k] for k in candidates_indices]
        target_idx = candidates_indices[argmin(v_counts)]

        dists = [sum((curr_c[target_idx] .- curr_c[j]).^2) for j in 1:length(curr_c)]
        dists[target_idx] = Inf
        neighbor_idx = argmin(dists)

        total_n = counts[target_idx] + counts[neighbor_idx]
        curr_c[neighbor_idx] = (curr_c[target_idx] .* counts[target_idx] .+
                        curr_c[neighbor_idx] .* counts[neighbor_idx]) ./ (total_n + 1e-9)
        deleteat!(curr_c, target_idx)
    end

    return [Tuple(c) for c in curr_c], termination_reason
end


"""
    get_lattice_centroids(s_x, s_y, lengthscale)

Generates regular square lattice centroids covering the bounding box of `(s_x, s_y)`.
"""
function get_lattice_centroids(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, lengthscale::Real
)
    if isempty(s_x) || isempty(s_y)
        return Tuple{Float64, Float64}[], 0, 0, (0.0, 0.0, 0.0, 0.0)
    end

    xmin, xmax = extrema(s_x)
    ymin, ymax = extrema(s_y)

    x_range = collect(xmin:lengthscale:xmax)
    y_range = collect(ymin:lengthscale:ymax)

    if isempty(x_range)
        x_range = [xmin]
    end
    if isempty(y_range)
        y_range = [ymin]
    end

    rows = length(y_range)
    cols = length(x_range)
    centroids = [(Float64(x), Float64(y)) for y in y_range, x in x_range][:]

    return centroids, rows, cols, (xmin, xmax, ymin, ymax)
end


"""
    get_hexagonal_centroids(s_x, s_y, radius)

Generates regular hexagonal grid centroids and polygons covering the bounding box of `(s_x, s_y)`.
Hexagonal packing provides uniform distance to all 6 neighbors, eliminating axial orientation bias.
"""
function get_hexagonal_centroids(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, radius::Real
)
    if isempty(s_x) || isempty(s_y)
        return Tuple{Float64, Float64}[], Vector{Tuple{Float64, Float64}}[], (0.0, 0.0, 0.0, 0.0)
    end

    xmin, xmax = extrema(s_x)
    ymin, ymax = extrema(s_y)

    dx = sqrt(3.0) * radius
    dy = 1.5 * radius

    centroids = Tuple{Float64, Float64}[]
    polygons = Vector{Vector{Tuple{Float64, Float64}}}()

    row = 0
    y_curr = ymin
    while y_curr <= ymax + radius
        x_offset = isodd(row) ? (dx / 2.0) : 0.0
        x_curr = xmin - radius + x_offset
        while x_curr <= xmax + radius
            push!(centroids, (x_curr, y_curr))
            # 6 vertices of regular hexagon
            hex_verts = Tuple{Float64, Float64}[]
            for k in 0:5
                angle = deg2rad(30.0 + 60.0 * k)
                vx = x_curr + radius * cos(angle)
                vy = y_curr + radius * sin(angle)
                push!(hex_verts, (vx, vy))
            end
            push!(hex_verts, hex_verts[1]) # closed polygon
            push!(polygons, hex_verts)
            x_curr += dx
        end
        y_curr += dy
        row += 1
    end

    return centroids, polygons, (xmin, xmax, ymin, ymax)
end


"""
    get_user_centroids(input_polygons)

Extracts centroids, closed polygon coordinates, and combined bounding hull from
  user-provided `LibGEOS` polygons.
"""
function get_user_centroids(input_polygons)
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

    collection = LibGEOS.GeometryCollection(geoms)
    united = LibGEOS.unaryUnion(collection)
    hull_coords = get_coords_from_geom(united)

    return centroids, polys_coords, hull_coords
end


"""
    load_shapefile_to_libgeos(filepath::String)

Loads a shapefile (.shp) and converts its geometries into `LibGEOS` objects.
"""
function load_shapefile_to_libgeos(filepath::String)
    table = Shapefile.Table(filepath)
    geoms = [LibGEOS.read_geom(row.geometry) for row in table]
    return geoms, table
end


# -----------------------------------------------------------------------------
# Section 3: High-Level Spatial Partitioning Dispatcher & Granular Sizing
# -----------------------------------------------------------------------------

"""
    _merge_undersized_polygons!(lg_polys, polys_coords, final_centroids, min_area)

Merges any polygon with area strictly less than `min_area` into its most adjacent neighbor
(sharing the longest boundary). Updates the arrays in-place.
"""
function _merge_undersized_polygons!(
    lg_polys::Vector{LibGEOS.Polygon},
    polys_coords::Vector{Vector{Tuple{Float64, Float64}}},
    final_centroids::Vector{Tuple{Float64, Float64}},
    min_area::Real
)
    if min_area <= 0.0 || length(lg_polys) <= 1
        return lg_polys, polys_coords, final_centroids
    end

    modified = true
    while modified && length(lg_polys) > 1
        modified = false
        areas = [get_polygon_area(polys_coords[i]) for i in 1:length(polys_coords)]
        under_idx = findfirst(a -> a > 0 && a < min_area, areas)

        if under_idx !== nothing
            # Find best adjacent neighbor to merge with
            p_small = lg_polys[under_idx]
            best_neighbor = 0
            best_inter_len = -1.0

            for j in 1:length(lg_polys)
                if j != under_idx
                    if LibGEOS.touches(p_small, lg_polys[j]) ||
   LibGEOS.intersects(LibGEOS.buffer(p_small, 1e-6), lg_polys[j])
                        inter = LibGEOS.intersection(LibGEOS.buffer(p_small, 1e-6), lg_polys[j])
                        inter_len = LibGEOS.area(inter)
                        if inter_len > best_inter_len
                            best_inter_len = inter_len
                            best_neighbor = j
                        end
                    end
                end
            end

            # If no touching neighbor found, find the geometrically closest one
            if best_neighbor == 0
                dists = [
                    sqrt((final_centroids[under_idx][1]-final_centroids[j][1])^2 +
                         (final_centroids[under_idx][2]-final_centroids[j][2])^2)
                    for j in 1:length(lg_polys)
                ]
                dists[under_idx] = Inf
                best_neighbor = argmin(dists)
            end

            # Merge under_idx into best_neighbor
            try
                merged_geom = LibGEOS.union(lg_polys[best_neighbor], p_small)
                if !LibGEOS.isEmpty(merged_geom)
                    if LibGEOS.geomTypeId(merged_geom) == LibGEOS.GEOS_MULTIPOLYGON
                        merged_geom = LibGEOS.convexhull(merged_geom)
                    end
                    lg_polys[best_neighbor] = merged_geom
                    polys_coords[best_neighbor] = get_coords_from_geom(merged_geom)
                    cent_g = LibGEOS.centroid(merged_geom)
                    seq = LibGEOS.getCoordSeq(cent_g)
                    final_centroids[best_neighbor] = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))

                    deleteat!(lg_polys, under_idx)
                    deleteat!(polys_coords, under_idx)
                    deleteat!(final_centroids, under_idx)
                    modified = true
                end
            catch
                # Fallback if union fails
                deleteat!(lg_polys, under_idx)
                deleteat!(polys_coords, under_idx)
                deleteat!(final_centroids, under_idx)
                modified = true
            end
        end
    end

    return lg_polys, polys_coords, final_centroids
end


"""
    _adjust_to_exact_units!(lg_polys, polys_coords, final_centroids, target_units, hull_geom)

Strictly enforces an exact number of final polygons (`target_units`).
If `K > target_units`, repeatedly merges the smallest adjacent polygon pair.
If `K < target_units`, repeatedly bisects the largest polygon along its bounding axis.
"""
function _adjust_to_exact_units!(
    lg_polys::Vector{LibGEOS.Polygon},
    polys_coords::Vector{Vector{Tuple{Float64, Float64}}},
    final_centroids::Vector{Tuple{Float64, Float64}},
    target_units::Integer,
    hull_geom
)
    if target_units <= 0 || length(lg_polys) == target_units
        return lg_polys, polys_coords, final_centroids
    end

    # Case A: Too many polygons -> iteratively merge smallest pairs
    while length(lg_polys) > target_units
        areas = [get_polygon_area(polys_coords[i]) for i in 1:length(polys_coords)]
        smallest_idx = argmin(areas)

        # Find closest/touching neighbor
        p_small = lg_polys[smallest_idx]
        best_neighbor = 0
        min_dist = Inf

        for j in 1:length(lg_polys)
            if j != smallest_idx
                d = (final_centroids[smallest_idx][1]-final_centroids[j][1])^2 +
    (final_centroids[smallest_idx][2]-final_centroids[j][2])^2
                if d < min_dist
                    min_dist = d
                    best_neighbor = j
                end
            end
        end

        if best_neighbor == 0
            break
        end

        try
            merged_geom = LibGEOS.union(lg_polys[best_neighbor], p_small)
            if !LibGEOS.isEmpty(merged_geom)
                if LibGEOS.geomTypeId(merged_geom) == LibGEOS.GEOS_MULTIPOLYGON
                    merged_geom = LibGEOS.convexhull(merged_geom)
                end
                lg_polys[best_neighbor] = merged_geom
                polys_coords[best_neighbor] = get_coords_from_geom(merged_geom)
                cent_g = LibGEOS.centroid(merged_geom)
                seq = LibGEOS.getCoordSeq(cent_g)
                final_centroids[best_neighbor] = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))

                deleteat!(lg_polys, smallest_idx)
                deleteat!(polys_coords, smallest_idx)
                deleteat!(final_centroids, smallest_idx)
            else
                deleteat!(lg_polys, smallest_idx)
                deleteat!(polys_coords, smallest_idx)
                deleteat!(final_centroids, smallest_idx)
            end
        catch
            deleteat!(lg_polys, smallest_idx)
            deleteat!(polys_coords, smallest_idx)
            deleteat!(final_centroids, smallest_idx)
        end
    end

    # Case B: Too few polygons -> iteratively bisect the largest polygon
    while length(lg_polys) < target_units
        areas = [get_polygon_area(polys_coords[i]) for i in 1:length(polys_coords)]
        largest_idx = argmax(areas)
        p_large = lg_polys[largest_idx]
        p_coords = polys_coords[largest_idx]

        if length(p_coords) < 3
            break
        end

        xs = [pt[1] for pt in p_coords if !isnan(pt[1])]
        ys = [pt[2] for pt in p_coords if !isnan(pt[2])]
        xmin, xmax = extrema(xs)
        ymin, ymax = extrema(ys)
        c_x, c_y = final_centroids[largest_idx]

        # Bisect along longer dimension
        pad = 2.0 * max(xmax - xmin, ymax - ymin, 1.0)
        box1_pts, box2_pts = if (xmax - xmin) >= (ymax - ymin)
            # Vertical split
            (
                [[(xmin - pad, ymin - pad), (c_x, ymin - pad), (c_x, ymax + pad),
                  (xmin - pad, ymax + pad), (xmin - pad, ymin - pad)]],
                [[(c_x, ymin - pad), (xmax + pad, ymin - pad), (xmax + pad, ymax + pad),
                  (c_x, ymax + pad), (c_x, ymin - pad)]]
            )
        else
            # Horizontal split
            (
                [[(xmin - pad, ymin - pad), (xmax + pad, ymin - pad), (xmax + pad, c_y),
                  (xmin - pad, c_y), (xmin - pad, ymin - pad)]],
                [[(xmin - pad, c_y), (xmax + pad, c_y), (xmax + pad, ymax + pad),
                  (xmin - pad, ymax + pad), (xmin - pad, c_y)]]
            )
        end

        try
            b1 = LibGEOS.Polygon(box1_pts)
            b2 = LibGEOS.Polygon(box2_pts)
            res1 = LibGEOS.intersection(p_large, b1)
            res2 = LibGEOS.intersection(p_large, b2)

            if !LibGEOS.isEmpty(res1) && !LibGEOS.isEmpty(res2) &&
   get_polygon_area(get_coords_from_geom(res1)) > 1e-6 &&
   get_polygon_area(get_coords_from_geom(res2)) > 1e-6
                # Replace largest with res1 and append res2
                lg_polys[largest_idx] = res1
                polys_coords[largest_idx] = get_coords_from_geom(res1)
                seq1 = LibGEOS.getCoordSeq(LibGEOS.centroid(res1))
                final_centroids[largest_idx] = (LibGEOS.getX(seq1, 1), LibGEOS.getY(seq1, 1))

                push!(lg_polys, res2)
                push!(polys_coords, get_coords_from_geom(res2))
                seq2 = LibGEOS.getCoordSeq(LibGEOS.centroid(res2))
                push!(final_centroids, (LibGEOS.getX(seq2, 1), LibGEOS.getY(seq2, 1)))
            else
                break # Cannot split further cleanly
            end
        catch
            break
        end
    end

    return lg_polys, polys_coords, final_centroids
end


"""
    assign_spatial_units(s_x, s_y; area_method=:avt, target_units=10, exact_units=false,
                         target_area=nothing, min_area=0.0, max_area=Inf, min_points=1,
                         max_points=nothing, lengthscale=nothing, radius=nothing,
                         grid_resolution=nothing, aspect_ratio=1.0, prune_empty=false,
                         merge_small_polygons=false, input_polygons=nothing,
                           geom_hull=nothing, kwargs...)

Primary spatial discretization engine in `bstm`. Partitions continuous spatial domains
into discrete areal units with granular control over polygon size and unit count.

### Sizing & Count Control:
- `target_units::Integer`: Desired number of final polygons (default: 10).
- `exact_units::Bool`: If `true`, dynamically enforces exactly `target_units` final polygons
  (default: false).
- `target_area::Union{Nothing, Real}`: Desired mean area per polygon in coordinate units.
  Auto-computes `target_units` and scales cell geometry.
- `min_area::Real`, `max_area::Real`: Minimum and maximum allowed polygon area.
- `merge_small_polygons::Bool`: Merges sliver/boundary-clipped polygons below `min_area`
  into neighbors.
- `prune_empty::Bool`: Removes areal units that contain 0 data observations (default: false).
- `lengthscale::Real`, `radius::Real`: Explicit cell side length (`:lattice`) or hexagon
  radius (`:hexagonal`).
- `grid_resolution`: Explicit grid resolution for `:lattice` (e.g., `20` or `(rows, cols)`).
- `aspect_ratio::Real`: Rectangular cell aspect ratio for `:lattice` (`dy / dx`).

### Methods (`area_method`):
- `:avt` (default): Agglomerative Voronoi Tessellation (bottom-up merge).
- `:cvt`: Centroidal Voronoi Tessellation (Lloyd relaxation).
- `:kvt`: K-means Voronoi Tessellation (density-balanced).
- `:qvt`: Quadtree Voronoi Tessellation.
- `:bvt`: Binary Voronoi Tessellation.
- `:hvt`: Hierarchical Voronoi Tessellation.
- `:hexagonal` / `:hexbin`: Regular hexagonal cell grid.
- `:lattice`: Regular square/rectangular grid.
- `input_polygons`: User-supplied custom geometries.

Returns a `NamedTuple`:
`(centroids, polygons, adjacency_edges, graph, W, hull_coords, s_idx, s_x, s_y, s_vals,
  areas, point_counts, n_units, metrics, termination_reason)`.
"""
function assign_spatial_units(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real};
    area_method::Symbol=:avt, target_units::Union{Nothing, Integer}=nothing,
    exact_units::Bool=false, target_area::Union{Nothing, Real}=nothing,
    min_area::Real=0.0, max_area::Real=Inf, min_points::Integer=1,
    max_points::Union{Nothing, Integer}=nothing, lengthscale=nothing,
    radius=nothing, grid_resolution=nothing, aspect_ratio::Real=1.0,
    prune_empty::Bool=false, merge_small_polygons::Bool=false,
    input_polygons=nothing, geom_hull=nothing, kwargs...
)
    s_coord_tuple_local = tuple.(Float64.(s_x), Float64.(s_y))
    pts_mat = hcat([[p[1], p[2]] for p in s_coord_tuple_local]...)
    n_pts = length(s_x)

    # 0. Resolve target_units and target_area
    hull_geom_raw = !isnothing(geom_hull) ? geom_hull : expand_hull(s_x, s_y, 0.0)
    domain_total_area = get_polygon_area(get_coords_from_geom(hull_geom_raw))

    eff_target_units = if !isnothing(target_units)
        Int(target_units)
    elseif !isnothing(target_area) && target_area > 0.0 && domain_total_area > 0.0
        max(1, round(Int, domain_total_area / target_area))
    else
        10
    end

    eff_max_points = isnothing(max_points) ? n_pts : Int(max_points)

    # 1. User-Supplied Custom Polygons
    if !isnothing(input_polygons)
        processed_polys = if isnothing(geom_hull)
    input_polygons
else
    [LibGEOS.intersection(p, geom_hull) for p in input_polygons]
end
        final_centroids, polys_coords, hull_coords = get_user_centroids(processed_polys)
        reason = :user_polygons
        lg_polys = LibGEOS.Polygon[p for p in processed_polys]

    # 2. Hexagonal Grid Method
    elseif area_method in [:hexagonal, :hexbin]
        hex_rad = if !isnothing(radius)
            Float64(radius)
        elseif !isnothing(target_area) && target_area > 0.0
            sqrt((2.0 * target_area) / (3.0 * sqrt(3.0)))
        elseif !isnothing(lengthscale)
            Float64(lengthscale)
        else
            sqrt(domain_total_area / (1.5 * sqrt(3.0) * max(1, eff_target_units)))
        end

        raw_cents, raw_polys, bbox = get_hexagonal_centroids(s_x, s_y, hex_rad)
        reason = :hexagonal_grid

        hull_geom = !isnothing(geom_hull) ? geom_hull : expand_hull(s_x, s_y, 0.1 * hex_rad)
        hull_coords = get_coords_from_geom(hull_geom)

        polys_coords = Vector{Vector{Tuple{Float64, Float64}}}()
        lg_polys = LibGEOS.Polygon[]
        final_centroids = Tuple{Float64, Float64}[]

        for (i, c) in enumerate(raw_cents)
            hex_coords = raw_polys[i]
            p_geom = LibGEOS.Polygon([[[pt[1], pt[2]] for pt in hex_coords]])
            if !isnothing(geom_hull)
                p_geom = LibGEOS.intersection(p_geom, geom_hull)
            end
            if !LibGEOS.isEmpty(p_geom) && get_polygon_area(get_coords_from_geom(p_geom)) > 1e-6
                push!(lg_polys, p_geom)
                cent_geom = LibGEOS.centroid(p_geom)
                seq = LibGEOS.getCoordSeq(cent_geom)
                push!(final_centroids, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
                push!(polys_coords, get_coords_from_geom(p_geom))
            end
        end

    # 3. Regular Lattice/Grid Partitioning
    elseif area_method == :lattice || area_method == :grid
        ls_x, ls_y = if !isnothing(grid_resolution)
            rows, cols = if grid_resolution isa Integer
                (Int(grid_resolution), Int(grid_resolution))
            else
                (Int(grid_resolution[1]), Int(grid_resolution[2]))
            end
            xmin, xmax = extrema(s_x); ymin, ymax = extrema(s_y)
            ((xmax - xmin) / max(1, cols), (ymax - ymin) / max(1, rows))
        elseif !isnothing(lengthscale)
            (Float64(lengthscale), Float64(lengthscale) * aspect_ratio)
        elseif !isnothing(target_area) && target_area > 0.0
            side = sqrt(target_area)
            (side, side * aspect_ratio)
        else
            side = sqrt(domain_total_area / max(1, eff_target_units))
            (side, side * aspect_ratio)
        end

        xmin, xmax = extrema(s_x)
        ymin, ymax = extrema(s_y)
        x_range = collect(xmin:ls_x:xmax)
        if isempty(x_range)
            x_range = [xmin]
        end
        y_range = collect(ymin:ls_y:ymax)
        if isempty(y_range)
            y_range = [ymin]
        end
        final_centroids_raw = [(Float64(x), Float64(y)) for y in y_range, x in x_range][:]
        reason = :lattice_grid

        polys_coords = Vector{Vector{Tuple{Float64, Float64}}}()
        lg_polys = LibGEOS.Polygon[]
        final_centroids = Tuple{Float64, Float64}[]
        half_x, half_y = ls_x / 2.0, ls_y / 2.0

        for c in final_centroids_raw
            coords = [[
                [Float64(c[1]-half_x), Float64(c[2]-half_y)],
                [Float64(c[1]+half_x), Float64(c[2]-half_y)],
                [Float64(c[1]+half_x), Float64(c[2]+half_y)],
                [Float64(c[1]-half_x), Float64(c[2]+half_y)],
                [Float64(c[1]-half_x), Float64(c[2]-half_y)]
            ]]
            p_geom = LibGEOS.Polygon(coords)
            if !isnothing(geom_hull)
                p_geom = LibGEOS.intersection(p_geom, geom_hull)
            end

            if !LibGEOS.isEmpty(p_geom) && get_polygon_area(get_coords_from_geom(p_geom)) > 1e-6
                push!(lg_polys, p_geom)
                p_c = LibGEOS.centroid(p_geom)
                seq = LibGEOS.getCoordSeq(p_c)
                push!(final_centroids, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
                push!(polys_coords, get_coords_from_geom(p_geom))
            end
        end
        hull_coords = isnothing(geom_hull) ?
    [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)] :
    get_coords_from_geom(geom_hull)

    # 4. Voronoi Tessellation Methods (:avt, :cvt, :kvt, :qvt, :bvt, :hvt)
    else
        cfg = (
            target=eff_target_units,
            min_total_arealunits=Int(get(kwargs, :min_total_arealunits, 3)),
            max_total_arealunits=Int(get(kwargs, :max_total_arealunits, eff_target_units * 2)),
            min_time_slices=Int(get(kwargs, :min_time_slices, 1)),
            min_points=Int(min_points),
            max_points=eff_max_points,
            min_area=Float64(min_area),
            max_area=Float64(max_area),
            target_cv=get(kwargs, :target_cv, 1.0),
            tolerance=get(kwargs, :tolerance, 0.1),
            buffer_dist=get(kwargs, :buffer_dist, 0.5),
            t_idx=get(kwargs, :t_idx, ones(Int, length(s_x)))
        )

        hull_geom = !isnothing(geom_hull) ? geom_hull : expand_hull(s_x, s_y, cfg.buffer_dist)

        c_mid, reason = if area_method == :cvt
            get_cvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :kvt
            get_kvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :qvt
            get_qvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :bvt
            get_bvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :hvt
            get_hvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :avt
            get_avt_centroids(s_x, s_y, cfg, hull_geom)
        else
            error("Unknown partitioning method: $area_method. " *
      "Choose from :avt, :cvt, :kvt, :qvt, :bvt, :hvt, :hexagonal, :lattice.")
        end

        polys_coords, v_edges = get_voronoi_polygons_and_edges(c_mid, hull_geom)
        final_centroids = Tuple{Float64, Float64}[]
        lg_polys = LibGEOS.Polygon[]
        for p_coords in polys_coords
            if isempty(p_coords)
                continue
            end
            if p_coords[1] != p_coords[end]
                push!(p_coords, p_coords[1])
            end
            lg_p = LibGEOS.Polygon([[ [pt[1], pt[2]] for pt in p_coords ]])
            push!(lg_polys, lg_p)
            cent_g = LibGEOS.centroid(lg_p)
            seq = LibGEOS.getCoordSeq(cent_g)
            push!(final_centroids, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
        end
        hull_coords = get_coords_from_geom(hull_geom)
    end

    # -------------------------------------------------------------------------
    # Granular Post-Processing: Merging, Pruning, and Exact Units Enforcement
    # -------------------------------------------------------------------------

    is_regular_grid = area_method in [:hexagonal, :hexbin, :lattice, :grid]

    # A. Merge small polygons below min_area (irregular Voronoi tessellations only)
    # Regular grids (hexagons/lattices) are uniform geometry; merging cells creates
    # irregular multi-polygon aggregations and destroys regular grid properties.
    if !is_regular_grid && (merge_small_polygons || (min_area > 0.0))
        _merge_undersized_polygons!(lg_polys, polys_coords, final_centroids, min_area)
    end

    # B. Strict exact units enforcement
    if exact_units && eff_target_units > 0
        if is_regular_grid
            # For regular grids, rank cells by observation density and proximity
            # to data center without merging or deforming hexagonal geometries.
            if length(final_centroids) > eff_target_units
                cand_mat = hcat([[c[1], c[2]] for c in final_centroids]...)
                cand_tree = KDTree(cand_mat)
                cand_assigns, _ = knn(cand_tree, pts_mat, 1)
                pt_counts = zeros(Int, length(final_centroids))
                for a in cand_assigns
                    pt_counts[a[1]] += 1
                end

                mean_x = mean(s_x)
                mean_y = mean(s_y)
                dists_to_center = [
                    (c[1] - mean_x)^2 + (c[2] - mean_y)^2 for c in final_centroids
                ]
                ranked_indices = sort(
                    1:length(final_centroids),
                    by = i -> (-pt_counts[i], dists_to_center[i])
                )
                keep_indices = sort(ranked_indices[1:eff_target_units])

                lg_polys = lg_polys[keep_indices]
                polys_coords = polys_coords[keep_indices]
                final_centroids = final_centroids[keep_indices]
            end
        else
            hull_for_adjust = !isnothing(geom_hull) ? geom_hull : expand_hull(s_x, s_y, 0.1)
            _adjust_to_exact_units!(
                lg_polys, polys_coords, final_centroids, eff_target_units, hull_for_adjust
            )
        end
    end

    # C. Point Assignments
    c_mat = hcat([[c[1], c[2]] for c in final_centroids]...)
    tree = KDTree(c_mat)
    assigns_knn, _ = knn(tree, pts_mat, 1)
    new_assigns = [a[1] for a in assigns_knn]

    # D. Prune empty units if requested
    if prune_empty && length(final_centroids) > 1
        active_units = unique(new_assigns)
        if length(active_units) < length(final_centroids)
            lg_polys = lg_polys[active_units]
            polys_coords = polys_coords[active_units]
            final_centroids = final_centroids[active_units]

            # Reassign points
            c_mat = hcat([[c[1], c[2]] for c in final_centroids]...)
            tree = KDTree(c_mat)
            assigns_knn, _ = knn(tree, pts_mat, 1)
            new_assigns = [a[1] for a in assigns_knn]
        end
    end

    # E. Construct Final Adjacency Graph W
    n_units = length(final_centroids)
    g = SimpleGraph(n_units)
    for i in 1:n_units, j in (i+1):n_units
        if LibGEOS.touches(lg_polys[i], lg_polys[j]) ||
   LibGEOS.intersects(LibGEOS.buffer(lg_polys[i], 1e-7), lg_polys[j])
            add_edge!(g, i, j)
        end
    end
    g = ensure_connected!(g, final_centroids)
    W = Float64.(Graphs.adjacency_matrix(g))

    v_edges = Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}[]
    for e in Graphs.edges(g)
        push!(v_edges, (final_centroids[src(e)], final_centroids[dst(e)]))
    end

    # F. Compute Detailed Area and Point Count Metrics
    areas = [get_polygon_area(p) for p in polys_coords]
    point_counts = [count(==(i), new_assigns) for i in 1:n_units]
    metrics = (
        mean_density = mean(point_counts),
        sd_density = length(point_counts) > 1 ? std(point_counts) : 0.0,
        cv_density = length(point_counts) > 1 ?
        std(point_counts) / (mean(point_counts) + 1e-9) : 0.0,
        min_density = isempty(point_counts) ? 0 : minimum(point_counts),
        max_density = isempty(point_counts) ? 0 : maximum(point_counts),
        mean_area = mean(areas),
        sd_area = length(areas) > 1 ? std(areas) : 0.0,
        cv_area = length(areas) > 1 ? std(areas) / (mean(areas) + 1e-9) : 0.0,
        min_area = isempty(areas) ? 0.0 : minimum(areas),
        max_area = isempty(areas) ? 0.0 : maximum(areas),
        total_area = sum(areas)
    )

    wkt_val = if haskey(kwargs, :wkt) && !isnothing(kwargs[:wkt])
        string(kwargs[:wkt])
    elseif haskey(kwargs, :crs) && !isnothing(kwargs[:crs])
        string(kwargs[:crs])
    elseif haskey(kwargs, :proj) && !isnothing(kwargs[:proj])
        string(kwargs[:proj])
    else
        nothing
    end

    return (
        centroids = final_centroids,
        polygons = polys_coords,
        adjacency_edges = v_edges,
        graph = g,
        W = W,
        hull_coords = hull_coords,
        s_idx = new_assigns,
        assignments = new_assigns,
        s_x = s_x,
        s_y = s_y,
        s_vals = collect(1:size(W, 1)),
        areas = areas,
        point_counts = point_counts,
        n_units = n_units,
        metrics = metrics,
        termination_reason = reason,
        wkt = wkt_val
    )
end

# Overloaded signatures for Tuples & DataFrames
assign_spatial_units(coords::Vector{<:Tuple{Real, Real}}; kwargs...) =
    assign_spatial_units([p[1] for p in coords], [p[2] for p in coords]; kwargs...)

function assign_spatial_units(df::DataFrame; x::Symbol=:s_x, y::Symbol=:s_y, kwargs...)
    if !hasproperty(df, x) || !hasproperty(df, y)
        error("DataFrame missing required spatial coordinate columns `:$x` and `:$y`.")
    end
    return assign_spatial_units(df[!, x], df[!, y]; kwargs...)
end


"""
    assign_spatial_units_inferred(adjacency_matrix; iterations=50, learning_rate=0.1,
      buffer_dist=0.5, input_polygons=nothing, wkt=nothing)

Infers spatial node positions and creates Voronoi boundaries when only a neighborhood
adjacency matrix `W` is available (e.g. Scottish Lip Cancer dataset). Uses a force-directed
spring layout to determine relative spatial coordinates.
"""
function assign_spatial_units_inferred(
    adjacency_matrix; iterations=50, learning_rate=0.1, buffer_dist=0.5, input_polygons=nothing,
    wkt=nothing, crs=nothing, proj=nothing
)
    nAU = size(adjacency_matrix, 1)

    wkt_val = if !isnothing(wkt)
        string(wkt)
    elseif !isnothing(crs)
        string(crs)
    elseif !isnothing(proj)
        string(proj)
    else
        nothing
    end

    if input_polygons !== nothing && !isempty(input_polygons)
        final_centroids_geoms = [LibGEOS.centroid(p) for p in input_polygons]
        final_centroids = map(final_centroids_geoms) do g_pt
            seq = LibGEOS.getCoordSeq(g_pt)
            (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
        end

        collection = LibGEOS.GeometryCollection(LibGEOS.Polygon[p for p in input_polygons])
        united_geom = LibGEOS.unaryUnion(collection)
        hull_coords_output = get_coords_from_geom(united_geom)

        adjacency_edges_output = Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}[]
        for i in 1:nAU
            g1 = input_polygons[i]
            for j in (i+1):nAU
                g2 = input_polygons[j]
                if LibGEOS.touches(g1, g2)
                    push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                else
                    g1_buffered = LibGEOS.buffer(g1, 1e-6)
                    if LibGEOS.intersects(g1_buffered, g2)
                        push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                    end
                end
            end
        end

        polys_output = [get_coords_from_geom(p) for p in input_polygons]
        g_final = SimpleGraph(nAU)
        for i in 1:nAU, j in (i+1):nAU
            if adjacency_matrix[i, j] != 0
                add_edge!(g_final, i, j)
            end
        end
        g_final = ensure_connected!(g_final, final_centroids)
    else
        g_initial = SimpleGraph(adjacency_matrix)
        side = ceil(Int, sqrt(nAU))
        initial_centroids = [(Float64(i % side), Float64(i ÷ side)) for i in 0:(nAU-1)]
        centroids_vec = [SVector{2, Float64}(c) for c in initial_centroids]

        for _ in 1:iterations
            new_centroids_vec = copy(centroids_vec)
            for i in 1:nAU
                neighbors_i = Graphs.neighbors(g_initial, i)
                if !isempty(neighbors_i)
                    avg_pos = sum(centroids_vec[n] for n in neighbors_i) / length(neighbors_i)
                    new_centroids_vec[i] = centroids_vec[i] +
            learning_rate * (avg_pos - centroids_vec[i])
                end
            end
            centroids_vec = new_centroids_vec
        end

        forced_centroids = [(p[1], p[2]) for p in centroids_vec]
        fx = [c[1] for c in forced_centroids]
        fy = [c[2] for c in forced_centroids]
        hull_geom = expand_hull(fx, fy, buffer_dist)
        hull_coords_output = get_coords_from_geom(hull_geom)

        polys_coords_raw, _ = get_voronoi_polygons_and_edges(forced_centroids, hull_geom)
        final_centroids = Vector{Tuple{Float64, Float64}}(undef, length(polys_coords_raw))
        polys_output = polys_coords_raw

        for (idx, poly_coord_list) in enumerate(polys_coords_raw)
            if length(poly_coord_list) >= 3
                if poly_coord_list[1] != poly_coord_list[end]
                    push!(poly_coord_list, poly_coord_list[1])
                end
                lg_poly = LibGEOS.Polygon([[ [p[1], p[2]] for p in poly_coord_list ]])
                seq = LibGEOS.getCoordSeq(LibGEOS.centroid(lg_poly))
                final_centroids[idx] = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
            else
                final_centroids[idx] = forced_centroids[idx]
            end
        end

        g_final = SimpleGraph(nAU)
        for i in 1:nAU, j in (i+1):nAU
            if adjacency_matrix[i, j] != 0
                add_edge!(g_final, i, j)
            end
        end
        g_final = ensure_connected!(g_final, final_centroids)
        adjacency_edges_output = Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}[]
        for e in Graphs.edges(g_final)
            push!(adjacency_edges_output, (final_centroids[src(e)], final_centroids[dst(e)]))
        end
    end

    return (
        centroids = final_centroids,
        polygons = polys_output,
        adjacency_edges = adjacency_edges_output,
        graph = g_final,
        W = Float64.(adjacency_matrix),
        hull_coords = hull_coords_output,
        s_idx = collect(1:nAU),
        s_x = [c[1] for c in final_centroids[1:nAU]],
        s_y = [c[2] for c in final_centroids[1:nAU]],
        s_vals = collect(1:nAU),
        termination_reason = "positions inferred from adjacency matrix",
        wkt = wkt_val
    )
end


# -----------------------------------------------------------------------------
# Section 4: Spatial Graph Topology & Weighting Matrix Utilities
# -----------------------------------------------------------------------------

"""
    spatial_weights_matrix(W::AbstractMatrix; style::Symbol=:binary)

Transforms an adjacency matrix `W` into standard spatial econometric / statistical weight schemes:
- `:binary` (default): Unweighted 0/1 adjacency.
- `:row_standardized` / `:row_norm`: Row-normalized matrix where rows sum to 1 (`W[i,:] /
  sum(W[i,:])`).
- `:variance_stabilized`: Normalized by `sqrt(row_sum)`.
"""
function spatial_weights_matrix(W::AbstractMatrix; style::Symbol=:binary)
    if style == :binary
        return Float64.(W)
    elseif style in [:row_standardized, :row_norm]
        row_sums = vec(sum(W, dims=2))
        inv_sums = [s > 0 ? 1.0 / s : 0.0 for s in row_sums]
        return Diagonal(inv_sums) * Float64.(W)
    elseif style == :variance_stabilized
        row_sums = vec(sum(W, dims=2))
        inv_sqrt = [s > 0 ? 1.0 / sqrt(s) : 0.0 for s in row_sums]
        return Diagonal(inv_sqrt) * Float64.(W)
    else
        error("Unknown spatial weighting style: $style. " *
      "Choose from :binary, :row_standardized, :variance_stabilized.")
    end
end


"""
    spatial_knn_graph(coords, k::Integer)

Constructs a k-nearest neighbor spatial graph and binary adjacency matrix directly from
  coordinate points.
"""
function spatial_knn_graph(coords::Vector{<:Tuple{Real, Real}}, k::Integer)
    n = length(coords)
    k_adj = min(k, n - 1)
    pts_mat = hcat([[Float64(p[1]), Float64(p[2])] for p in coords]...)
    tree = KDTree(pts_mat)
    idxs, _ = knn(tree, pts_mat, k_adj + 1)

    g = SimpleGraph(n)
    for i in 1:n
        for neighbor in idxs[i][2:end]
            add_edge!(g, i, neighbor)
        end
    end
    W = Float64.(Graphs.adjacency_matrix(g))
    return g, W
end


"""
    spatial_radius_graph(coords, radius::Real)

Constructs a distance-threshold spatial graph connecting all points within distance `radius`.
"""
function spatial_radius_graph(coords::Vector{<:Tuple{Real, Real}}, radius::Real)
    n = length(coords)
    pts_mat = hcat([[Float64(p[1]), Float64(p[2])] for p in coords]...)
    tree = KDTree(pts_mat)
    idxs = inrange(tree, pts_mat, Float64(radius))

    g = SimpleGraph(n)
    for i in 1:n
        for neighbor in idxs[i]
            if neighbor != i
                add_edge!(g, i, neighbor)
            end
        end
    end
    W = Float64.(Graphs.adjacency_matrix(g))
    return g, W
end


"""
    ensure_connected!(g::SimpleGraph, centroids::Vector{<:Tuple{Real, Real}})

Ensures a spatial graph `g` is fully connected by identifying disjoint components
and adding minimal bridging edges between their closest centroid pairs.
"""
function ensure_connected!(g::SimpleGraph, centroids::Vector{<:Tuple{Real, Real}})
    comps = connected_components(g)
    if length(comps) <= 1
        return g
    end

    n_nodes = length(centroids)
    c_mat = hcat([[Float64(c[1]), Float64(c[2])] for c in centroids]...)
    tree = KDTree(c_mat)

    # 1. Add minimal bridging edge from each component to nearest external node
    for comp in comps
        comp_set = Set(comp)
        best_d = Inf
        best_u = 0
        best_v = 0
        k_search = min(n_nodes, max(4, ceil(Int, sqrt(n_nodes))))
        for u in comp
            idxs, dists = knn(tree, [centroids[u][1], centroids[u][2]], k_search)
            for (k_idx, v) in enumerate(idxs)
                if !(v in comp_set)
                    if dists[k_idx] < best_d
                        best_d = dists[k_idx]
                        best_u = u
                        best_v = v
                        break
                    end
                end
            end
        end
        if best_u != 0 && best_v != 0 && !has_edge(g, best_u, best_v)
            add_edge!(g, best_u, best_v)
        end
    end

    # 2. Iteratively bridge any remaining disjoint component centroids
    comps = connected_components(g)
    while length(comps) > 1
        n_c = length(comps)
        comp_centroids = [
            mean([[Float64(centroids[n][1]), Float64(centroids[n][2])] for n in c])
            for c in comps
        ]
        tree_c = KDTree(hcat(comp_centroids...))
        idxs, _ = knn(tree_c, comp_centroids[1], min(2, n_c))
        target_idx = idxs[min(2, length(idxs))]
        u = comps[1][1]
        v = comps[target_idx][1]
        add_edge!(g, u, v)
        comps = connected_components(g)
    end

    return g
end




"""
    scaling_factor_bym2(W::AbstractMatrix)

Computes the BYM2 structured GMRF scaling factor from adjacency matrix `W` by
constructing the singular ICAR precision matrix `Q = Diagonal(row_sums) - W`
and evaluating the geometric mean of non-zero eigenvalues (Riebler et al., 2016).
"""
function scaling_factor_bym2(W::AbstractMatrix)
    if size(W, 1) != size(W, 2)
        error("Adjacency matrix W must be square.")
    end
    Q = Diagonal(vec(sum(W, dims=2))) - W
    return _compute_scaling_factor(Q)
end

function _compute_scaling_factor(Q_template)
    eigenvalues = eigvals(Symmetric(Matrix(Q_template)))
    non_zero_eigenvalues = eigenvalues[eigenvalues .> 1e-12]
    if isempty(non_zero_eigenvalues)
        return 1.0
    end
    return exp(-mean(log.(non_zero_eigenvalues)))
end


"""
    adjacency_matrix_to_nb(W)

Converts a binary adjacency matrix `W` into a neighbor-list representation `Vector{Vector{Int}}`.
"""
function adjacency_matrix_to_nb(W::AbstractMatrix)
    nau = size(W, 1)
    nb = [Int[] for _ in 1:nau]
    for i in 1:nau
        nb[i] = findall(!iszero, W[i, :])
    end
    return nb
end


"""
    nb_to_adjacency_matrix(nb)

Converts a neighbor-list `nb` (`Vector{Vector{Int}}`) into a dense binary adjacency matrix.
"""
function nb_to_adjacency_matrix(nb)
    nau = length(nb)
    W = zeros(Float64, nau, nau)
    for i in 1:nau
        for neighbor in nb[i]
            if neighbor <= nau
                W[i, neighbor] = 1.0
            end
        end
    end
    return W
end


"""
    nodes(adj)

Extracts undirected edge endpoints `(node1, node2)` and computes the BYM2 scaling factor.
"""
function nodes(adj)
    nau = length(adj)
    node1 = Int[]
    node2 = Int[]
    for i in 1:nau
        for k in adj[i]
            if i < k
                push!(node1, i)
                push!(node2, k)
            end
        end
    end
    e = Edge.(node1, node2)
    g = Graph(e)
    W = Float64.(Graphs.adjacency_matrix(g))
    scalefactor = scaling_factor_bym2(W)
    return node1, node2, scalefactor
end


"""
    get_spatial_graph(centroids, adjacency_edges)

Constructs a `SimpleGraph` from a list of centroids and edge tuples.
"""
function get_spatial_graph(centroids, adjacency_edges)
    n = length(centroids)
    g = SimpleGraph(n)
    centroid_map = Dict(c => i for (i, c) in enumerate(centroids))
    for edge in adjacency_edges
        xi, yi = get(centroid_map, edge[1], 0), get(centroid_map, edge[2], 0)
        if xi > 0 && yi > 0
            add_edge!(g, xi, yi)
        end
    end
    return g
end


"""
    libgeos_lattice_adjacency_matrix(rows::Int, cols::Int; contiguity::Symbol=:queen)

Fast analytical lattice adjacency matrix generation in O(N).
Supports `:queen` (8-neighbor, default) and `:rook` (4-neighbor) contiguity.
"""
function libgeos_lattice_adjacency_matrix(rows::Int, cols::Int; contiguity::Symbol=:queen)
    n = rows * cols
    W = spzeros(Int, n, n)
    idx(r, c) = (r - 1) * cols + c

    for r in 1:rows, c in 1:cols
        u = idx(r, c)
        # Rook neighbors (4-connectivity)
        if r > 1
            W[u, idx(r-1, c)] = 1
        end
        if r < rows
            W[u, idx(r+1, c)] = 1
        end
        if c > 1
            W[u, idx(r, c-1)] = 1
        end
        if c < cols
            W[u, idx(r, c+1)] = 1
        end

        # Queen diagonals (8-connectivity)
        if contiguity == :queen
            if r > 1 && c > 1
                W[u, idx(r-1, c-1)] = 1
            end
            if r > 1 && c < cols
                W[u, idx(r-1, c+1)] = 1
            end
            if r < rows && c > 1
                W[u, idx(r+1, c-1)] = 1
            end
            if r < rows && c < cols
                W[u, idx(r+1, c+1)] = 1
            end
        end
    end
    return W
end


"""
    check_connectivity(g)

Evaluates graph connectivity and returns `(is_connected, n_components, components)`.
"""
function check_connectivity(g)
    comps = connected_components(g)
    return (is_connected = length(comps) == 1, n_components = length(comps), components = comps)
end


"""
    calculate_metrics(au_obj)

Calculates summary density metrics for an areal units object:
`(mean_density, sd_density, cv_density)`.
"""
function calculate_metrics(au_obj)
    pts_mat = hcat([
        [Float64(au_obj.s_x[i]), Float64(au_obj.s_y[i])]
        for i in 1:length(au_obj.s_x)
    ]...)
    c_mat = hcat([[Float64(c[1]), Float64(c[2])] for c in au_obj.centroids]...)
    tree = KDTree(c_mat)
    assigns_knn, _ = knn(tree, pts_mat, 1)
    assignments = [a[1] for a in assigns_knn]
    unit_counts = [count(==(i), assignments) for i in 1:length(au_obj.centroids)]

    valid_entries = filter(x -> !isnan(x) && !ismissing(x), unit_counts)
    if isempty(valid_entries)
        return (mean_density=NaN, sd_density=NaN, cv_density=NaN)
    end
    m_val = mean(valid_entries)
    s_val = std(valid_entries)
    cv_val = s_val / (m_val + 1e-9)

    return (mean_density=m_val, sd_density=s_val, cv_density=cv_val)
end


# -----------------------------------------------------------------------------
# Section 5: Temporal Discretization, Spatiotemporal Units & Cross-Validation
# -----------------------------------------------------------------------------

"""
    discretize_data(X; method="quantile", N_cat=9, brks=nothing, ...)

Discretizes a continuous variable `X` into `N_cat` bins using:
- `"quantile"`: Empirical quantiles.
- `"regular"`: Equal-width bins across the data extent.
- `"quantile_regular"`: Equal-width bins between lower/upper quantiles.
- `"kmeans"`: 1D k-means clustering.
- `"jenks"`: Jenks Natural Breaks optimization.
- `"provided"`: Precomputed explicit break bounds.

Returns `(idx = idx, brks = collect(brks), mids = mids, N_cat = N_cat)`.
"""
function discretize_data(
    X; method="quantile", N_cat=9, brks=nothing, probs=nothing, dx=nothing,
    minv=nothing, maxv=nothing, quantile_bounds=[0.025, 0.975]
)
    if method == "quantile"
        probs = isnothing(probs) ? range(0, stop=1, length=N_cat+1) : probs
        brks = quantile(X, probs)
        brks[end] = brks[end] + 1e-6
    elseif method == "regular"
        minv = isnothing(minv) ? minimum(X) : minv
        maxv = isnothing(maxv) ? maximum(X) : maxv
        dx = isnothing(dx) ? (maxv - minv) / N_cat : dx
        brks = minv:dx:maxv
    elseif method == "quantile_regular"
        q = quantile(X, quantile_bounds)
        brks = range(q[1], q[2], length=N_cat + 1)
    elseif method == "kmeans"
        X_mat = reshape(X, 1, :)
        R = kmeans(X_mat, N_cat)
        centers = sort(vec(R.centers))
        brks = vcat(minimum(X), (centers[1:end-1] + centers[2:end]) / 2.0, maximum(X))
    elseif method == "jenks"
        brks = _jenks_breaks(X, N_cat)
    elseif method == "provided"
        if isnothing(brks)
            error("Method 'provided' requires the 'brks' argument.")
        end
        N_cat = length(brks) - 1
    else
        error("Discretization method '$method' not recognized.")
    end

    function get_idx(x::Real, breaks, n_categories)
        raw_idx = searchsortedfirst(breaks, x) - 1
        return clamp(raw_idx, 1, n_categories)
    end

    idx = map(x -> get_idx(x, brks, N_cat), X)
    mids = brks[1:end-1] .+ diff(brks) ./ 2.0

    return (idx = idx, brks = collect(brks), mids = mids, N_cat = N_cat)
end


function _jenks_breaks(data::AbstractVector{<:Real}, n_classes::Int)
    s_data = sort(data)
    n_data = length(s_data)
    lower_class_limits = zeros(n_data, n_classes)
    variance_combinations = zeros(n_data, n_classes)

    for i in 1:n_data
        lower_class_limits[i, 1] = 1
        variance_combinations[i, 1] = sum((s_data[1:i] .- mean(s_data[1:i])).^2)
    end

    for k in 2:n_classes
        for i in k:n_data
            min_variance = Inf
            best_break = 0
            for j in k:i
                ssd = sum((s_data[j:i] .- mean(s_data[j:i])).^2)
                total_variance = ssd + variance_combinations[j-1, k-1]
                if total_variance < min_variance
                    min_variance = total_variance
                    best_break = j
                end
            end
            lower_class_limits[i, k] = best_break
            variance_combinations[i, k] = min_variance
        end
    end

    breaks = zeros(n_classes + 1)
    breaks[n_classes + 1] = s_data[end]
    breaks[1] = s_data[1]
    for k in n_classes:-1:2
        break_index = Int(lower_class_limits[n_data, k])
        breaks[k] = s_data[break_index]
        n_data = break_index - 1
    end

    return breaks
end


"""
    assign_time_units(t_v; time_method="quantile_regular", t_N=nothing, u_N=nothing, kwargs...)

Discretizes a time vector into categorical unit indices. Handles continuous floats and
  discrete integers.
"""
function assign_time_units(
    t_v::AbstractVector{<:Real};
    time_method="quantile_regular", t_N=nothing, u_N=nothing, kwargs...
)
    local_t_N = isnothing(t_N) ? (isnothing(u_N) ? 10 : u_N) : t_N
    return discretize_data(t_v; method=time_method, N_cat=local_t_N, kwargs...)
end

function assign_time_units(
    t_v::AbstractVector{<:Integer}; time_method="unique", t_N=nothing, u_N=nothing, kwargs...
)
    unique_times = sort(unique(t_v))
    N_cat = length(unique_times)
    brks = if N_cat > 1
        vcat(
    [unique_times[1] - 0.5],
    (unique_times[1:end-1] + unique_times[2:end]) / 2.0,
    [unique_times[end] + 0.5]
)
    elseif N_cat == 1
        [unique_times[1] - 0.5, unique_times[1] + 0.5]
    else
        Float64[]
    end
    val_to_idx = Dict(val => i for (i, val) in enumerate(unique_times))
    idx = [val_to_idx[v] for v in t_v]
    mids = Float64.(unique_times)

    return (idx=idx, brks=brks, mids=mids, N_cat=N_cat)
end


"""
    assign_spatiotemporal_units(df::DataFrame; space_x=:s_x, space_y=:s_y, time_var=:t_idx,
                                area_method=:avt, target_units=10, time_method="unique",
                                  t_N=nothing, kwargs...)

Simultaneously discretizes space and time, producing synchronized `(s_idx, t_idx, st_idx)`
and dimensional metadata for spatiotemporal modeling.
"""
function assign_spatiotemporal_units(
    df::DataFrame; space_x::Symbol=:s_x, space_y::Symbol=:s_y, time_var::Symbol=:t_idx,
    area_method::Symbol=:avt, target_units::Integer=10,
    time_method="unique", t_N=nothing, kwargs...
)
    au_space = assign_spatial_units(
        df; x=space_x, y=space_y, area_method=area_method,
        target_units=target_units, kwargs...
    )
    au_time = assign_time_units(df[!, time_var]; time_method=time_method, t_N=t_N, kwargs...)

    S = length(au_space.centroids)
    T = au_time.N_cat
    s_idx = au_space.s_idx
    t_idx = au_time.idx
    st_idx = (t_idx .- 1) .* S .+ s_idx

    scalefactor_s = scaling_factor_bym2(au_space.W)

    return (
        au_spatial = au_space,
        au_temporal = au_time,
        s_idx = s_idx,
        t_idx = t_idx,
        st_idx = st_idx,
        S = S,
        T = T,
        ST = S * T,
        W_spatial = au_space.W,
        scaling_factor_spatial = scalefactor_s
    )
end


"""
    spatial_block_cv(s_x, s_y; n_folds::Integer=5, method::Symbol=:kmeans)

Generates spatial cross-validation fold assignments (`1:n_folds`) to assess model generalizability
without spatial autocorrelation leakage across training/test splits.
"""
function spatial_block_cv(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real};
    n_folds::Integer=5, method::Symbol=:kmeans
)
    n = length(s_x)
    if n == 0
        return Int[]
    elseif n < n_folds
        return collect(1:n)
    end

    if method == :kmeans
        pts_mat = hcat([[Float64(s_x[i]), Float64(s_y[i])] for i in 1:n]...)
        R = kmeans(pts_mat, n_folds)
        return R.assignments
    elseif method == :grid
        xmin, xmax = extrema(s_x)
        ymin, ymax = extrema(s_y)
        side = ceil(Int, sqrt(n_folds))
        x_bins = range(xmin, stop=xmax + 1e-6, length=side + 1)
        y_bins = range(ymin, stop=ymax + 1e-6, length=side + 1)
        folds = zeros(Int, n)
        for i in 1:n
            bx = clamp(searchsortedfirst(x_bins, s_x[i]) - 1, 1, side)
            by = clamp(searchsortedfirst(y_bins, s_y[i]) - 1, 1, side)
            cell_id = (by - 1) * side + bx
            folds[i] = ((cell_id - 1) % n_folds) + 1
        end
        return folds
    else
        error("Unknown spatial block CV method: $method. Choose :kmeans or :grid.")
    end
end

spatial_block_cv(coords::Vector{<:Tuple{Real, Real}}; kwargs...) =
    spatial_block_cv([p[1] for p in coords], [p[2] for p in coords]; kwargs...)


"""
    estimate_local_kde_with_extrapolation(s_coord_tuple, t_idx, target_ts; grid_res=600,
      sd_extension_factor=0.25)

Computes 2D Gaussian Kernel Density Estimation (KDE) on a regular grid for a specific time slice.
"""
function estimate_local_kde_with_extrapolation(
    s_coord_tuple, t_idx, target_ts; grid_res=600, sd_extension_factor=0.25
)
    filtered_pts = [p for (i, p) in enumerate(s_coord_tuple) if t_idx[i] == target_ts]
    if isempty(filtered_pts)
        error("No points found for target time slice $target_ts.")
    end
    xs = [p[1] for p in filtered_pts]
    ys = [p[2] for p in filtered_pts]

    bw_x = max(std(xs) * sd_extension_factor, 1e-4)
    bw_y = max(std(ys) * sd_extension_factor, 1e-4)

    x_min, x_max = minimum(xs) - bw_x, maximum(xs) + bw_x
    y_min, y_max = minimum(ys) - bw_y, maximum(ys) + bw_y

    x_grid = collect(range(x_min, stop=x_max, length=grid_res))
    y_grid = collect(range(y_min, stop=y_max, length=grid_res))
    intensity = zeros(grid_res, grid_res)

    for i in 1:grid_res
        x_val = x_grid[i]
        for j in 1:grid_res
            y_val = y_grid[j]
            for (px, py) in filtered_pts
                dx = (x_val - px) / bw_x
                dy = (y_val - py) / bw_y
                intensity[i, j] += exp(-0.5 * (dx^2 + dy^2))
            end
        end
    end

    intensity ./= max(sum(intensity), 1e-9)
    return x_grid, y_grid, intensity
end




"""
    lonlat_to_xy_km(lon, lat; center_lon=nothing, center_lat=nothing, crs=nothing, datum=WGS84Latest)
    -> Tuple{Float64, Float64}

Projects geographic coordinates (longitude, latitude in decimal degrees) into a planar
Cartesian coordinate space in kilometers, optionally relative to a local tangent origin.

# Mathematical Formulation
When `crs` is supplied and `CoordRefSystems` is available, coordinates are converted
via geodesic projection. Otherwise, a local equirectangular tangent plane approximation
is computed:
```math
x = R \\cdot (\\lambda - \\lambda_0) \\cdot \\cos(\\phi_0) \\cdot \\frac{\\pi}{180^\\circ}
```
```math
y = R \\cdot (\\phi - \\phi_0) \\cdot \\frac{\\pi}{180^\\circ}
```
where \$R = 6371.0\\text{ km}\$ is Earth's mean radius, \$(\\lambda_0, \\phi_0)\$ is the
tangent projection origin (`center_lon`, `center_lat`), and \$(\\lambda, \\phi)\$ are
the input longitude and latitude.

# Arguments
- `lon::Real`: Longitude in decimal degrees.
- `lat::Real`: Latitude in decimal degrees.
- `center_lon`: Tangent plane projection center longitude (default `nothing`).
- `center_lat`: Tangent plane projection center latitude (default `nothing`).
- `crs`: Optional target Coordinate Reference System from `CoordRefSystems`.
- `datum`: Geographic datum (default `WGS84Latest`).

# Returns
- `Tuple{Float64, Float64}`: Planar coordinates `(x_km, y_km)` in kilometers.
"""
function lonlat_to_xy_km(
    lon::Real, lat::Real;
    center_lon = nothing,
    center_lat = nothing,
    crs = nothing,
    datum = WGS84Latest
)::Tuple{Float64, Float64}
    if !isnothing(crs) && _HAS_COORDSYS
        try
            source_pt = LatLon{datum}(lat, lon)
            proj_pt = convert(crs, source_pt)

            x_km = Float64(ustrip(u"km", proj_pt.x))
            y_km = Float64(ustrip(u"km", proj_pt.y))

            # If a local custom center is provided, subtract it to get relative offsets
            if !isnothing(center_lon) && !isnothing(center_lat)
                center_pt = LatLon{datum}(center_lat, center_lon)
                proj_center = convert(crs, center_pt)
                x_km -= Float64(ustrip(u"km", proj_center.x))
                y_km -= Float64(ustrip(u"km", proj_center.y))
            end

            return (x_km, y_km)
        catch
        end
    end
    
    # Fallback to local tangent plane (requires a center to function, defaults to 0.0)
    c_lon = isnothing(center_lon) ? 0.0 : Float64(center_lon)
    c_lat = isnothing(center_lat) ? 0.0 : Float64(center_lat)
    
    R  = 6371.0
    ϕ0 = c_lat * (π / 180.0)
    x  = (Float64(lon) - c_lon) * cos(ϕ0) * (π / 180.0) * R
    y  = (Float64(lat) - c_lat) * (π / 180.0) * R
    return (x, y)
end

"""
    xy_km_to_lonlat(x_km, y_km; center_lon=nothing, center_lat=nothing, crs=nothing, datum=WGS84Latest)
    -> Tuple{Float64, Float64}

Inverts planar Cartesian coordinates in kilometers back to geographic coordinates
(longitude, latitude in decimal degrees).

# Mathematical Formulation
For the local equirectangular tangent plane fallback:
```math
\\lambda = \\lambda_0 + \\frac{x}{R \\cdot \\cos(\\phi_0) \\cdot (\\pi / 180^\\circ)}
```
```math
\\phi = \\phi_0 + \\frac{y}{R \\cdot (\\pi / 180^\\circ)}
```
where \$R = 6371.0\\text{ km}\$ is Earth's mean radius and \$(\\lambda_0, \\phi_0)\$ is
the tangent plane projection origin (`center_lon`, `center_lat`).

# Arguments
- `x::Real`: Planar x-coordinate in kilometers.
- `y::Real`: Planar y-coordinate in kilometers.
- `center_lon`: Tangent plane projection center longitude (default `nothing`).
- `center_lat`: Tangent plane projection center latitude (default `nothing`).
- `crs`: Optional target Coordinate Reference System from `CoordRefSystems`.
- `datum`: Geographic datum (default `WGS84Latest`).

# Returns
- `Tuple{Float64, Float64}`: Geographic coordinates `(lon, lat)` in decimal degrees.
"""
function xy_km_to_lonlat(
    x::Real, y::Real;
    center_lon = nothing,
    center_lat = nothing,
    crs = nothing,
    datum = WGS84Latest
)::Tuple{Float64, Float64}
    if !isnothing(crs) && _HAS_COORDSYS
        try
            abs_x = x * 1.0u"km"
            abs_y = y * 1.0u"km"

            # If a local custom center was used, add it back to get absolute CRS coordinates
            if !isnothing(center_lon) && !isnothing(center_lat)
                center_pt = LatLon{datum}(center_lat, center_lon)
                proj_center = convert(crs, center_pt)
                abs_x += proj_center.x
                abs_y += proj_center.y
            end

            proj_pt = crs(abs_x, abs_y)
            lonlat_pt = convert(LatLon{datum}, proj_pt)

            out_lon = Float64(ustrip(u"°", lonlat_pt.lon))
            out_lat = Float64(ustrip(u"°", lonlat_pt.lat))

            return (out_lon, out_lat)
        catch
        end
    end
    
    # Fallback local tangent plane
    c_lon = isnothing(center_lon) ? 0.0 : Float64(center_lon)
    c_lat = isnothing(center_lat) ? 0.0 : Float64(center_lat)
    
    R  = 6371.0
    ϕ0 = c_lat * (π / 180.0)
    lon = c_lon + (Float64(x) / (R * cos(ϕ0))) * (180.0 / π)
    lat = c_lat + (Float64(y) / R) * (180.0 / π)
    return (lon, lat)
end

# ── Internal helpers ───────────────────────────────────────────────────────────

"""
    _to_decimal_year(d) -> Float64

Convert a `Date` or `DateTime` to decimal year, accounting for leap years:

    decimal_year = year + (dayofyear − 1) / days_in_year
"""
function _to_decimal_year(d::Union{Date, DateTime})::Float64
    yr         = Dates.year(d)
    doy        = Float64(Dates.dayofyear(d))
    days_in_yr = Dates.isleapyear(yr) ? 366.0 : 365.0
    return Float64(yr) + (doy - 1.0) / days_in_yr
end

"""
    _safe_mean(v) -> Float64

Mean of `v` after stripping `Missing` and non-finite values.
Returns `NaN` when no valid values remain.
"""
function _safe_mean(v)::Float64
    vals = collect(skipmissing(v))
    vals = filter(x -> (x isa Real) && isfinite(x), vals)
    return isempty(vals) ? NaN : mean(vals)
end

# ── Telemetry aggregation ──────────────────────────────────────────────────────

"""
    aggregate_telemetry_time(tagging; time_interval=:monthly) -> DataFrame

Aggregate high-frequency telemetry pings per individual into regular temporal
bins, with `Missing`-safe numeric aggregation throughout.

# Binning modes
- `:monthly`  — first day of each (year, month).
- `:weekly`   — Monday of each ISO calendar week.
- `:biweekly` — 14-day epochs anchored at 1990-01-01.
- `:daily`    — calendar date.
- `:raw`      — no aggregation; returns a copy.

# Data requirements
`tagging` must contain at minimum: `:tagid`, `:lon`, `:lat`, `:timestamp`, `:tag`.
Optional numeric columns (`:z`, `:cw`, `:chela`, `:wgt`) are aggregated with
`_safe_mean`.  Group columns (`:sex`, `:mat`, `:datasource`) use `first`.
`:is_dead` is propagated via `any(skipmissing)`.

Individuals with < 2 distinct temporal observations after aggregation are
dropped.  `:tag` is re-indexed (0, 1, 2, …) per individual.

# Arguments
- `tagging::DataFrame`: Input telemetry table.
- `time_interval::Symbol`: Binning resolution (default `:monthly`).

# Returns
- `DataFrame`: Aggregated table with `:time` (decimal year) column.
"""
function aggregate_telemetry_time(
    tagging::DataFrame;
    time_interval::Symbol = :monthly
)::DataFrame

    time_interval == :raw && return copy(tagging)

    df = copy(tagging)
    hasproperty(df, :timestamp) || error("DataFrame must contain :timestamp column.")

    if time_interval == :daily
        df[!, :time_bucket] = Date.(df.timestamp)
    elseif time_interval == :weekly
        df[!, :time_bucket] = [Date(t) - Day(dayofweek(Date(t)) - 1)
                                for t in df.timestamp]
    elseif time_interval == :biweekly
        epoch = Date(1990, 1, 1)
        df[!, :time_bucket] = [epoch + Day(fld(Int(Date(t) - epoch), 14) * 14)
                                for t in df.timestamp]
    elseif time_interval == :monthly
        df[!, :time_bucket] = [Date(year(t), month(t), 1) for t in df.timestamp]
    else
        error("Unsupported time_interval: $(time_interval). " *
              "Use :monthly, :weekly, :biweekly, :daily, or :raw.")
    end

    spec = Pair[
        :lon       => _safe_mean => :lon,
        :lat       => _safe_mean => :lat,
        :tag       => minimum    => :tag,
        :timestamp => minimum    => :timestamp,
    ]
    for col in (:z, :cw, :chela, :wgt)
        hasproperty(df, col) && push!(spec, col => _safe_mean => col)
    end
    for col in (:sex, :mat, :datasource)
        hasproperty(df, col) && push!(spec, col => first => col)
    end
    hasproperty(df, :is_dead) &&
        push!(spec, :is_dead => (x -> any(skipmissing(x))) => :is_dead)

    agg = DataFrames.combine(groupby(df, [:tagid, :time_bucket]), spec...)
    agg[!, :time] = [_to_decimal_year(d) for d in agg.timestamp]
    sort!(agg, [:tagid, :time])

    valid = Set{String}()
    for sub in groupby(agg, :tagid)
        nrow(sub) >= 2 && push!(valid, string(first(sub.tagid)))
    end
    filter!(r -> string(r.tagid) in valid, agg)

    parts = DataFrame[]
    for sub in groupby(agg, :tagid)
        sdf     = DataFrame(sub)
        sdf.tag = collect(0:(nrow(sdf) - 1))
        push!(parts, sdf)
    end
    return isempty(parts) ? DataFrame() : vcat(parts...)
end

# ── Planar hexagonal mesh ──────────────────────────────────────────────────────

"""
    build_hex_mesh_planar(lon_vec, lat_vec; radius_km=5.0, crs=nothing, datum=WGS84Latest)
    -> NamedTuple

Construct a regular hexagonal tessellation in planar km coordinates from
observation lon/lat points, then back-project polygon vertices and centroids
to geographic coordinates for reporting and visualisation.

# Hex geometry (flat-top orientation)
For circumradius r (km = side length for a regular hexagon):
- Column spacing: dx = √3 r
- Row spacing:    dy = 1.5 r
- Offset row grid: odd rows shift by dx/2.
- Exact area:     A = (3√3/2) r²  km²

# Adjacency
W_ij = 1 when planar centroid distance < √3 r × 1.05 (5 % snap tolerance).
Symmetrised: W = max(W, Wᵀ).

# Arguments
- `lon_vec, lat_vec`: Geographic coordinates in decimal degrees.
- `radius_km`: Hexagon circumradius = side length in km (default 5.0).
- `crs`: Target Coordinate Reference System (default nothing, uses local tangent plane).
- `datum`: The geographic datum (default: `WGS84Latest`).

# Returns
`NamedTuple`:
- `centroids_km, centroids_lonlat`: Centroid coordinates in km and degrees.
- `polygons_km, polygons_lonlat`: Closed vertex rings per hex cell.
- `n_units::Int`, `W::SparseMatrixCSC`, `radius_km::Float64`.
- `areas_km2::Vector{Float64}`: Identical exact hex area per cell.
- `center_lon, center_lat`: Projection origin in degrees.
"""
function build_hex_mesh_planar(
    lon_vec::AbstractVector{<:Real},
    lat_vec::AbstractVector{<:Real};
    radius_km::Real = 5.0,
    crs = nothing,
    datum = WGS84Latest
)::NamedTuple

    center_lon = mean(lon_vec)
    center_lat = mean(lat_vec)

    # Calculate planar coordinates directly
    pts_km = [lonlat_to_xy_km(Float64(lon), Float64(lat);
                  center_lon=center_lon, center_lat=center_lat, 
                  crs=crs, datum=datum)
              for (lon, lat) in zip(lon_vec, lat_vec)]

    r  = Float64(radius_km)
    dx = sqrt(3.0) * r
    dy = 1.5 * r

    # Determine planar domain bounds (expanded by r to ensure complete boundary coverage)
    x_coords = [p[1] for p in pts_km]
    y_coords = [p[2] for p in pts_km]
    x_min, x_max = extrema(x_coords)
    y_min, y_max = extrema(y_coords)

    x_min -= r
    x_max += r
    y_min -= r
    y_max += r

    row_min = floor(Int, y_min / dy)
    row_max = ceil(Int, y_max / dy)

    centroids_km = Tuple{Float64, Float64}[]
    for row in row_min:row_max
        yk   = row * dy
        xoff = isodd(row) ? (dx / 2.0) : 0.0
        col_min = floor(Int, (x_min - xoff) / dx)
        col_max = ceil(Int, (x_max - xoff) / dx)
        for col in col_min:col_max
            xk = col * dx + xoff
            push!(centroids_km, (xk, yk))
        end
    end

    sort!(centroids_km, by = c -> (c[2], c[1]))
    S = length(centroids_km)

    polygons_km      = Vector{Vector{Tuple{Float64, Float64}}}(undef, S)
    polygons_lonlat  = Vector{Vector{Tuple{Float64, Float64}}}(undef, S)
    centroids_lonlat = Vector{Tuple{Float64, Float64}}(undef, S)

    # Flat-top hexagon: vertices at 30°, 90°, 150°, 210°, 270°, 330°
    hex_angles = (30.0 .+ 60.0 .* (0:5)) .* (π / 180.0)

    for (i, (cx, cy)) in enumerate(centroids_km)
        verts_km = [(cx + r * cos(a), cy + r * sin(a)) for a in hex_angles]
        push!(verts_km, verts_km[1]) # Close the polygon
        
        polygons_km[i]      = verts_km
        polygons_lonlat[i]  = [xy_km_to_lonlat(v[1], v[2];
                                    center_lon=center_lon, center_lat=center_lat,
                                    crs=crs, datum=datum)
                                for v in verts_km]
        
        centroids_lonlat[i] = xy_km_to_lonlat(cx, cy;
                                    center_lon=center_lon, center_lat=center_lat,
                                    crs=crs, datum=datum)
    end

    # Efficient matrix allocation for KDTree
    c_mat = Matrix{Float64}(undef, 2, S)
    for i in 1:S
        c_mat[1, i] = centroids_km[i][1]
        c_mat[2, i] = centroids_km[i][2]
    end
    
    tree       = KDTree(c_mat)
    adj_thresh = sqrt(3.0) * r * 1.05

    rows_idx = Int[]
    cols_idx = Int[]
    for i in 1:S
        nbrs = inrange(tree, [centroids_km[i][1], centroids_km[i][2]], adj_thresh)
        for j in nbrs
            if j != i
                push!(rows_idx, i)
                push!(cols_idx, j)
            end
        end
    end
    
    W = sparse(rows_idx, cols_idx, ones(Float64, length(rows_idx)), S, S)
    W = max.(W, W')

    area_km2 = (3.0 * sqrt(3.0) / 2.0) * r^2

    return (
        centroids        = centroids_lonlat,
        centroids_km     = centroids_km,
        centroids_lonlat = centroids_lonlat,
        polygons         = polygons_lonlat,
        polygons_km      = polygons_km,
        polygons_lonlat  = polygons_lonlat,
        n_units          = S,
        W                = W,
        radius_km        = r,
        areas_km2        = fill(area_km2, S),
        center_lon       = center_lon,
        center_lat       = center_lat
    )
end

# ── Telemetry → unit mapping ───────────────────────────────────────────────────


"""
    map_telemetry_to_units(
        tagging, centroids_km, center_lon, center_lat; crs=nothing, datum=WGS84Latest
    ) -> DataFrame

Assign each telemetry observation to the nearest hexagonal unit via a KDTree on
planar km centroids. Adds `:s_idx` (1-based integer, 1 ≤ s ≤ S).

# Arguments
- `tagging`: Must contain `:lon`, `:lat` (geographic degrees).
- `centroids_km`: Vector of (x, y) km tuples from `build_hex_mesh_planar`.
- `center_lon, center_lat`: Projection origin matching the mesh.
- `crs`: Target Coordinate Reference System (default nothing).
- `datum`: The geographic datum (default: `WGS84Latest`).
- `land_mask`: Optional boolean vector of length `S` (`true` for land units).
- `W`: Optional adjacency matrix (size `S × S`) to identify disconnected units.

# Returns
- Copy of `tagging` with `:s_idx::Int` appended.
"""
function map_telemetry_to_units(
    tagging::DataFrame,
    centroids_km::Vector{Tuple{Float64, Float64}},
    center_lon::Real,
    center_lat::Real;
    crs = nothing,
    datum = WGS84Latest,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    W::Union{Nothing, AbstractMatrix} = nothing
)::DataFrame
    S = length(centroids_km)

    # Filter to active navigable units if masks or topology are provided
    active_mask = trues(S)
    if land_mask !== nothing
        active_mask .&= .!land_mask
    end
    if W !== nothing
        deg = vec(sum(W, dims=2))
        active_mask .&= (deg .> 0.0)
    end
    active_indices = findall(active_mask)
    if isempty(active_indices)
        active_indices = collect(1:S)
    end

    n_act = length(active_indices)
    c_mat = Matrix{Float64}(undef, 2, n_act)
    for (col, idx) in enumerate(active_indices)
        c_mat[1, col] = centroids_km[idx][1]
        c_mat[2, col] = centroids_km[idx][2]
    end

    # Build the KDTree for fast spatial lookups on navigable marine units
    tree = KDTree(c_mat)

    # Pre-allocate and populate the telemetry points matrix
    N = nrow(tagging)
    pts_mat = Matrix{Float64}(undef, 2, N)
    for (i, (lon, lat)) in enumerate(zip(tagging.lon, tagging.lat))
        x, y = lonlat_to_xy_km(
            Float64(lon), Float64(lat);
            center_lon=center_lon, center_lat=center_lat,
            crs=crs, datum=datum
        )
        pts_mat[1, i] = x
        pts_mat[2, i] = y
    end

    # Perform nearest neighbor search for all points simultaneously
    idxs, _ = knn(tree, pts_mat, 1)

    # Append mapped units to a copy of the DataFrame
    out = copy(tagging)
    out[!, :s_idx] = [active_indices[first(idx)] for idx in idxs]

    return out
end


# ── HSI loading and monthly discretization ─────────────────────────────────────

"""
    _interpolate_hsi(hsi_2d, years, s_idx, decimal_year; ref_doy=244.0)
    -> Float64

Linear interpolation of the annual posterior-mean HSI for spatial unit `s_idx`
at continuous time `decimal_year`.

The annual predictions are anchored at `ref_doy` (day of year; 244 ≈ Sept 1):

    t_ref = (ref_doy − 1) / 365.25
    u     = decimal_year − t_ref
    y1    = clamp(⌊u⌋, y_min, y_max − 1)
    α     = u − y1                          ∈ [0, 1)
    HSI   = (1 − α) HSI(s, y1) + α HSI(s, y1+1)

# Arguments
- `hsi_2d`: Posterior-mean HSI matrix (S × T).
- `years`: Annual year labels (e.g. 1999:2025).
- `s_idx`: Spatial unit index (1-based).
- `decimal_year`: Continuous time in decimal years.
- `ref_doy`: Reference survey day of year (default 244.0).

# Returns
- `Float64`: Interpolated HSI clamped to [0, 1].
"""
function _interpolate_hsi(
    hsi_2d::AbstractMatrix{Float64},
    years::AbstractVector{<:Integer},
    s_idx::Integer,
    decimal_year::Real;
    ref_doy::Real = 244.0
)::Float64
    y_min = first(years)
    y_max = last(years)
    t_off = (ref_doy - 1.0) / 365.25
    u     = Float64(decimal_year) - t_off
    u_cl  = clamp(u, Float64(y_min), Float64(y_max))
    y1    = clamp(floor(Int, u_cl), y_min, y_max - 1)
    α     = u_cl - Float64(y1)
    t1    = clamp(y1 - y_min + 1, 1, length(years))
    t2    = clamp(t1 + 1, 1, length(years))
    return clamp((1.0 - α) * hsi_2d[s_idx, t1] + α * hsi_2d[s_idx, t2], 0.0, 1.0)
end


"""
    build_monthly_hsi_matrix(
        hsi_mean::AbstractMatrix{Float64},
        years::AbstractVector{<:Integer};
        ref_doy::Real = 244.0
    ) -> Tuple{Matrix{Float64}, Dict{Tuple{Int, Int}, Int}}

Discretizes multi-year spatial Habitat Suitability Index (HSI) surfaces onto a
regular monthly grid (12 calendar months per survey year) via continuous
piecewise-linear temporal interpolation, evaluating each cell at mid-month (day 15)
with leap-year awareness.

# Mathematical Formulation
Given annual HSI values ``h_s(y)`` for spatial unit ``s`` in year ``y``, the fractional
calendar year at day-of-year ``d`` of year ``y`` is:
```math
t = y + \\frac{d - 1}{D(y)}
```
where ``D(y) \\in \\{365, 366\\}``. Anchored to annual survey reference day
``d_{\\text{ref}}``, the mid-month HSI is linearly interpolated between adjacent survey epochs:
```math
h_s(t) = (1 - \\alpha) h_s(y_1) + \\alpha h_s(y_2)
```
clamped to ``[0, 1]``.

# Arguments
- `hsi_mean`: Spatial HSI posterior mean matrix (size ``S \\times T_{\\text{years}}``).
- `years`: Integer vector of survey years of length ``T_{\\text{years}}``.
- `ref_doy`: Reference survey day of year (default `244.0` = September 1).

# Returns
- `Tuple{Matrix{Float64}, Dict{Tuple{Int, Int}, Int}}`:
  - `mat`: Monthly discretized HSI matrix (size ``S \\times 12 T_{\\text{years}}``).
  - `lookup`: Dictionary mapping `(year, month)` tuples to matrix column indices.
"""
function build_monthly_hsi_matrix(
    hsi_mean::AbstractMatrix{Float64},
    years::AbstractVector{<:Integer};
    ref_doy::Real = 244.0
)::Tuple{Matrix{Float64}, Dict{Tuple{Int, Int}, Int}}

    S      = size(hsi_mean, 1)
    n_cols = length(years) * 12
    
    # Use undef since we overwrite every element (avoids zeroing overhead)
    mat    = Matrix{Float64}(undef, S, n_cols)
    
    # Pre-size the dictionary to avoid reallocations
    lookup = Dict{Tuple{Int, Int}, Int}()
    sizehint!(lookup, n_cols)
    
    col = 1
    for yr in years
        days_in_yr = Dates.isleapyear(yr) ? 366.0 : 365.0
        for m in 1:12
            doy_mid = Float64(Dates.dayofyear(Date(yr, m, 15)))
            dec_yr  = Float64(yr) + (doy_mid - 1.0) / days_in_yr
            
            # @inbounds safely removes bounds checking for peak speed
            @inbounds for s in 1:S
                mat[s, col] = _interpolate_hsi(
                    hsi_mean, years, s, dec_yr; ref_doy=ref_doy)
            end
            
            lookup[(yr, m)] = col
            col += 1
        end
    end
    
    return mat, lookup
end


"""
    load_hsi_jld2(path; hsi_key="hsi", years_key="years",
                  auids_key="auids", ref_doy=244.0) -> NamedTuple

Load a JLD2 bundle containing a 3D posterior HSI array and compute derived
summaries needed for movement modelling.

# Expected JLD2 keys
- `hsi`:   Array{Float64, 3} (S × T × N_draws).
- `years`: Vector{Int} of length T.
- `auids`: (optional) spatial unit identifiers.

# Arguments
- `path`: Path to the JLD2 file.
- `hsi_key`, `years_key`, `auids_key`: JLD2 key names.
- `ref_doy`: Reference day of year (default 244.0 = Sept 1).

# Returns
`NamedTuple`:
- `hsi`, `years`, `auids`.
- `hsi_mean` (S × T): posterior mean.
- `hsi_sd` (S × T): posterior standard deviation.
- `hsi_spatial_mean` (length S): multi-year mean per spatial unit.
- `monthly_hsi` (S × 12T): monthly discretization.
- `month_lookup`: Dict{(year, month) => column_index}.
"""
function load_hsi_jld2(
    path::AbstractString;
    hsi_key::AbstractString   = "hsi",
    years_key::AbstractString = "years",
    auids_key::AbstractString = "auids",
    ref_doy::Real             = 244.0
)::NamedTuple
    isfile(path) || error("HSI JLD2 not found: $(path)")
    bundle = JLD2.load(path)
    actual_hsi_key = if haskey(bundle, hsi_key)
        hsi_key
    elseif haskey(bundle, "sims")
        "sims"
    elseif haskey(bundle, "predictions")
        "predictions"
    else
        hsi_key
    end
    hsi    = bundle[actual_hsi_key]
    years  = bundle[years_key]
    auids  = haskey(bundle, auids_key) ? bundle[auids_key] : collect(1:size(hsi, 1))

    hsi_mean          = dropdims(mean(hsi, dims=3), dims=3)
    hsi_sd            = dropdims(std(hsi, dims=3),  dims=3)
    hsi_spatial_mean  = vec(mean(hsi_mean, dims=2))

    monthly_hsi, month_lookup = build_monthly_hsi_matrix(
        Float64.(hsi_mean), years; ref_doy=ref_doy)

    return (
        hsi              = hsi,
        years            = years,
        auids            = auids,
        hsi_mean         = hsi_mean,
        hsi_sd           = hsi_sd,
        hsi_spatial_mean = hsi_spatial_mean,
        monthly_hsi      = monthly_hsi,
        month_lookup     = month_lookup
    )
end
"""
    match_telemetry_closest_month_hsi(
        telemetry_df, monthly_hsi, month_lookup, years
    ) -> Vector{Float64}

Assign each telemetry observation its HSI value from the nearest (year, month).

# Fallback strategy (applied in order)
1. Clamp year to [y_min, y_max]; look up `(yr_clamped, month)`.
2. Same month in `y_min` (lower boundary).
3. Same month in `y_max` (upper boundary).
4. Assign `NaN` and emit a single summary `@warn` when entries are missing.

# Arguments
- `telemetry_df`: DataFrame with `:timestamp` and `:s_idx`.
- `monthly_hsi`: Matrix (S × 12T).
- `month_lookup`: Dict{(year, month) => column_index}.
- `years`: Valid year range.

# Returns
- `Vector{Float64}` of length `nrow(telemetry_df)`.
"""
function match_telemetry_closest_month_hsi(
    telemetry_df::DataFrame,
    monthly_hsi::AbstractMatrix{<:Real},
    month_lookup::Dict{Tuple{Int, Int}, Int},
    years::AbstractVector{<:Integer}
)::Vector{Float64}

    y_min, y_max = extrema(years)
    n_rows = nrow(telemetry_df)
    
    # Use undef instead of zeros to prevent unnecessary memory writing
    out = Vector{Float64}(undef, n_rows)

    # 1. Extract columns to local variables for type-stable, zero-overhead indexing
    timestamps = telemetry_df.timestamp
    s_idxs     = telemetry_df.s_idx

    missing_count = 0

    @inbounds for i in 1:n_rows
        dt = timestamps[i]
        s  = s_idxs[i]
        
        yr = clamp(Dates.year(dt), y_min, y_max)
        mo = Dates.month(dt)

        # 2. Use 0 as a default instead of `nothing` to keep types strict and fast
        col = get(month_lookup, (yr, mo), 0)
        
        if col == 0
            col = get(month_lookup, (y_min, mo), get(month_lookup, (y_max, mo), 0))
            if col == 0
                out[i] = NaN
                missing_count += 1
                continue
            end
        end
        
        out[i] = monthly_hsi[s, col]
    end
    
    # 3. Emit a single summary warning rather than spamming the console
    if missing_count > 0
        @warn "No month_lookup entry found for $missing_count telemetry observations. Assigned NaN."
    end

    return out
end
"""
    reshard_hsi_field(
        src_vals, src_coords_km, dest_centroids_km; k=6
    ) -> Vector{Float64}

Reshard a scalar field from source centroids to destination centroids using
planar inverse-distance-squared weighting (IDW) among the `k` nearest source
points.

All coordinates must be in the same planar km system.

Weight: w_j = 1/d_j² (d_j < 1e-9 → w = 1e18, i.e. coincident → snap).

# Arguments
- `src_vals`: Values at source centroids (length S_src).
- `src_coords_km`: Planar km coordinates of source centroids.
- `dest_centroids_km`: Planar km coordinates of destination centroids.
- `k`: Nearest neighbours to use (default 6; capped at S_src).

# Returns
- `Vector{Float64}` of length S_dest.
"""
function reshard_hsi_field(
    src_vals::AbstractVector{<:Real},
    src_coords_km::Vector{Tuple{Float64, Float64}},
    dest_centroids_km::Vector{Tuple{Float64, Float64}};
    k::Integer = 6
)::Vector{Float64}

    N_src = length(src_coords_km)
    N_dest = length(dest_centroids_km)
    k_eff = min(Int(k), length(src_vals))

    # 1. Preallocate and fill matrices efficiently (no splatting)
    src_mat = Matrix{Float64}(undef, 2, N_src)
    for i in 1:N_src
        src_mat[1, i] = src_coords_km[i][1]
        src_mat[2, i] = src_coords_km[i][2]
    end

    dest_mat = Matrix{Float64}(undef, 2, N_dest)
    for i in 1:N_dest
        dest_mat[1, i] = dest_centroids_km[i][1]
        dest_mat[2, i] = dest_centroids_km[i][2]
    end

    # 2. Build tree and find neighbors
    tree = KDTree(src_mat)
    idxs, dists = knn(tree, dest_mat, k_eff)

    # 3. Allocation-free inner loop for IDW
    out = Vector{Float64}(undef, N_dest)
    
    @inbounds for j in 1:N_dest
        w_sum = 0.0
        val_sum = 0.0
        
        # Loop over the k neighbors for the j-th destination point
        for i in 1:k_eff
            idx = idxs[j][i]
            d   = dists[j][i]
            
            # d * d is slightly faster than d^2
            w = d < 1e-9 ? 1e18 : 1.0 / (d * d)
            
            val_sum += w * src_vals[idx]
            w_sum   += w
        end
        
        # Protect against division by zero (though mathematically impossible with strictly positive w)
        out[j] = w_sum > 0.0 ? (val_sum / w_sum) : 0.0 
    end
    
    return out
end

