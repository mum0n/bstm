"""
    plotting.jl

Centralized visualization library for Bayesian Spatio-Temporal Models (BSTM),
including spatial maps, temporal lines, smooth curves, PPC checks, and diagnostic plots.

Version: v1.0.0
"""

# -----------------------------------------------------------------------------
# Section 1: Geometric & Theme Helpers (Internal)
# -----------------------------------------------------------------------------

"""
    _poly_xy(poly)

Converts polygon representations (Vector of (x,y) tuples, vectors, or arrays)
into two vectors of `Float64` coordinates `(xs, ys)`, filtering out `NaN` values.
"""
function _poly_xy(poly)
    xs = Float64[]
    ys = Float64[]
    for p in poly
        x = float(p[1])
        y = float(p[2])
        if !isnan(x) && !isnan(y)
            push!(xs, x)
            push!(ys, y)
        end
    end
    return xs, ys
end

"""
    _centroids_xy(centroids)

Safely extracts x and y coordinates from a collection of centroids into two `Float64` vectors.
"""
function _centroids_xy(centroids)
    xs = Float64[]
    ys = Float64[]
    for c in centroids
        push!(xs, float(c[1]))
        push!(ys, float(c[2]))
    end
    return xs, ys
end

"""
    _get_cscheme(cmap)

Resolves a `ColorSchemes.ColorScheme` object from a `Symbol`, `ColorScheme`, or fallback.
"""
function _get_cscheme(cmap)
    if cmap isa ColorSchemes.ColorScheme
        return cmap
    elseif cmap isa Symbol
        try
            return get(ColorSchemes, cmap, ColorSchemes.viridis)
        catch
            return ColorSchemes.viridis
        end
    else
        return ColorSchemes.viridis
    end
end

"""
    _map_color_from_range(cscheme, v, vmin, vmax)

Linearly maps a scalar value `v` in `[vmin, vmax]` to an RGB color from `cscheme`.
Returns light gray for `NaN` values.
"""
function _map_color_from_range(cscheme, v, vmin, vmax)
    if isnan(v)
        return RGB(0.9, 0.9, 0.9)  # NaN -> light gray
    end
    denom = vmax - vmin
    t = abs(denom) > 1e-12 ? (v - vmin) / denom : 0.5
    t = clamp(t, 0.0, 1.0)
    return get(cscheme, t)
end

"""
    create_theme(; fontsize=10, titlefontsize=12, font="DejaVu Sans", size=(900,600))

Returns a `NamedTuple` of standard plotting attributes for pass-through to Plots functions.
"""
function create_theme(; fontsize=10, titlefontsize=12, font="DejaVu Sans", size=(900,600))
    return (; legendfontsize=fontsize, guidefontsize=fontsize,
             tickfontsize=max(8, fontsize-1), titlefontsize=titlefontsize,
             fontfamily=font, size=size)
end

"""
    _resolve_temporal_coordinates(M, input_data, tm_len::Int, comp_key=nothing)

Robustly infers and aligns temporal coordinates (raw per-observation time vector,
sorted unique time units, and axis label) for plotting.
Prevents observation index fallback (e.g. 1:560) when time units range across 1:10 or
  calendar years.
"""
function _resolve_temporal_coordinates(M, input_data, tm_len::Int, comp_key=nothing)
    # 1. Search in input_data if available
    if !isnothing(input_data) && input_data isa DataFrame
        candidates = Symbol[]
        if !isnothing(comp_key)
            push!(candidates, Symbol(comp_key))
        end
        if haskey(M, :t_idx_var) && !isnothing(M.t_idx_var)
            push!(candidates, Symbol(M.t_idx_var))
        end
        append!(candidates, [:year, :time, :date, :month, :t_idx, :t, :day, :period])

        for cand in unique(candidates)
            if hasproperty(input_data, cand)
                col_data = input_data[!, cand]
                if length(col_data) == tm_len
                    unique_t = sort(unique(col_data))
                    t_lbl = (cand == :t_idx) ? "Time Index" : (cand == :year ? "Year" : string(cand))
                    return (raw_t = col_data, unique_t = unique_t, label = t_lbl)
                elseif length(unique(col_data)) == tm_len
                    unique_t = sort(unique(col_data))
                    t_lbl = (cand == :t_idx) ? "Time Index" : (cand == :year ? "Year" : string(cand))
                    return (raw_t = unique_t, unique_t = unique_t, label = t_lbl)
                end
            end
        end
    end

    # 2. Check M.t_idx
    if haskey(M, :t_idx) && !isnothing(M.t_idx) && !isempty(M.t_idx)
        t_vec = M.t_idx
        if length(t_vec) == tm_len
            unique_t = sort(unique(t_vec))
            t_lbl = haskey(M, :t_idx_var) ? string(M.t_idx_var) : "Time Index"
            return (raw_t = t_vec, unique_t = unique_t, label = t_lbl)
        elseif length(unique(t_vec)) == tm_len
            unique_t = sort(unique(t_vec))
            t_lbl = haskey(M, :t_idx_var) ? string(M.t_idx_var) : "Time Index"
            return (raw_t = unique_t, unique_t = unique_t, label = t_lbl)
        end
    end

    # 3. Check M.t_N (number of time steps)
    if haskey(M, :t_N) && M.t_N > 0
        t_N = M.t_N
        unique_t = collect(1:t_N)
        t_lbl = haskey(M, :t_idx_var) ? string(M.t_idx_var) : "Time Index"
        if tm_len == t_N
            return (raw_t = unique_t, unique_t = unique_t, label = t_lbl)
        elseif haskey(M, :s_N) && M.s_N > 0 && tm_len == M.s_N * t_N
            # Scottish lip cancer / regular space-time layout: time index cycles across units
            raw_t = [(i - 1) % t_N + 1 for i in 1:tm_len]
            return (raw_t = raw_t, unique_t = unique_t, label = t_lbl)
        elseif tm_len % t_N == 0
            raw_t = [(i - 1) % t_N + 1 for i in 1:tm_len]
            return (raw_t = raw_t, unique_t = unique_t, label = t_lbl)
        end
    end

    # 4. Fallback: integer sequence
    unique_t = collect(1:tm_len)
    return (raw_t = unique_t, unique_t = unique_t, label = "Time Index")
end


# -----------------------------------------------------------------------------
# Section 2: Spatial Primitives
# -----------------------------------------------------------------------------

"""
    choropleth(polygons::Vector, values::AbstractVector;
               cmap=:viridis, vmin=nothing, vmax=nothing, center_zero=false,
               show_colorbar=true, title="", border_color=:black, border_alpha=0.6,
               lw=0.4, closed=true, theme_kwargs=NamedTuple(), colorbar_label=nothing)

Creates a high-quality choropleth map from spatial `polygons` and aligned numeric `values`.
Supports automated percentile-based scaling, divergent center-zero palettes, and custom themes.
"""
function _choropleth_impl(
    polygons, values;
    cmap=:viridis, vmin=nothing, vmax=nothing, clims=nothing, center_zero::Bool=false,
    show_colorbar::Bool=true, title::String="", border_color=:black,
    border_alpha::Real=0.6, lw::Real=0.4, closed::Bool=true,
    theme_kwargs=NamedTuple(), colorbar_label::Union{Nothing, String}=nothing,
    kwargs...
)
    n = length(polygons)
    @assert length(values) == n "length(values) ($length(values)) must equal length(polygons) ($n)"

    vals = collect(Float64, values)
    mask = .!isnan.(vals)

    # If all values are NaN: render background polygon outlines
    if !any(mask)
        p = Plots.plot(aspect_ratio=:equal, title=title, legend=false; theme_kwargs..., kwargs...)
        for poly in polygons
            if length(poly) < 3
                continue
            end
            xs, ys = _poly_xy(poly)
            if closed && !isempty(xs) && (xs[1], ys[1]) != (xs[end], ys[end])
                push!(xs, xs[1]); push!(ys, ys[1])
            end
            if !isempty(xs)
                Plots.plot!(p, xs, ys, seriestype=:shape, fillcolor=:white,
                    linecolor=border_color, lw=lw, alpha=border_alpha, label=nothing)
            end
        end
        return p
    end

    # Robust vmin/vmax range
    cs_sym = center_zero ? :RdBu : (cmap isa Symbol ? cmap : :viridis)
    cs = _get_cscheme(cs_sym)
    vvals = vals[mask]
    if clims !== nothing
        vmin = clims[1]
        vmax = clims[2]
    end
    if vmin === nothing
        vmin = length(vvals) > 1 ? quantile(vvals, 0.02) : minimum(vvals)
    end
    if vmax === nothing
        vmax = length(vvals) > 1 ? quantile(vvals, 0.98) : maximum(vvals)
    end

    if center_zero
        limit = max(abs(vmin), abs(vmax))
        vmin, vmax = -limit, limit
    end

    p = Plots.plot(aspect_ratio=:equal, title=title, legend=false; theme_kwargs..., kwargs...)

    for (i, poly) in enumerate(polygons)
        if length(poly) < 3
            continue
        end
        xs, ys = _poly_xy(poly)
        if closed && !isempty(xs) && (xs[1], ys[1]) != (xs[end], ys[end])
            push!(xs, xs[1]); push!(ys, ys[1])
        end
        if !isempty(xs)
            fillcolor = mask[i] ? _map_color_from_range(cs, vals[i], vmin, vmax) : RGB(0.9,
                0.9, 0.9)
            Plots.plot!(p, xs, ys, seriestype=:shape, fillcolor=fillcolor,
                linecolor=border_color, lw=lw, alpha=0.95, label=nothing)
        end
    end

    # Colorbar rendering
    if show_colorbar && abs(vmax - vmin) > 1e-12
        Plots.scatter!(p, [NaN], [NaN], zcolor=[vmin], clims=(vmin, vmax),
                       markersize=0, markeralpha=0, c=cs_sym, colorbar=true, label=nothing)
        if colorbar_label !== nothing
            Plots.plot!(p, colorbar_title=colorbar_label)
        end
    end

    return p
end

# Unified, unambiguous choropleth dispatchers
function choropleth(polygons::AbstractVector{<:AbstractVector}, values::AbstractVector{<:Real};
    mode::Symbol=:plots, kwargs...)
    if mode == :leaflet || mode == :html
        return leaflet_choropleth(polygons, values; kwargs...)
    end
    return _choropleth_impl(polygons, values; kwargs...)
end

function choropleth(values::AbstractVector{<:Real}, polygons::AbstractVector{<:AbstractVector};
    mode::Symbol=:plots, kwargs...)
    if mode == :leaflet || mode == :html
        return leaflet_choropleth(polygons, values; kwargs...)
    end
    return _choropleth_impl(polygons, values; kwargs...)
end


"""
    spatial_graph_plot(centroids, g; polygons=nothing, hull_coords=nothing, pts=nothing,
                       node_size=3, node_color=:black, edge_color=:red, edge_alpha=0.6,
                       title="Spatial Partitioning", theme_kwargs=NamedTuple(), mode=:plots)

Plots spatial partitioning structures: polygons, adjacency graph edges, centroids, boundary
  hulls, and raw data points. Supports `mode=:plots` (Plots.jl) or `mode=:leaflet` (interactive HTML).
"""
function spatial_graph_plot(
    centroids, g; polygons=nothing, hull_coords=nothing, pts=nothing,
    node_size=3, node_color=:black, edge_color=:red, edge_alpha=0.6,
    title::String="Spatial Partitioning", theme_kwargs=NamedTuple(), mode::Symbol=:plots, kwargs...
)
    if mode == :leaflet || mode == :html
        return leaflet_spatial_graph(centroids, g; polygons=polygons, hull_coords=hull_coords,
                                     pts=pts, title=title, kwargs...)
    end
    p = Plots.plot(aspect_ratio=:equal, title=title, legend=false; theme_kwargs..., kwargs...)

    # 1. Polygons (if provided)
    if !isnothing(polygons)
        for poly in polygons
            if length(poly) > 2
                xs, ys = _poly_xy(poly)
                if !isempty(xs) && (xs[1], ys[1]) != (xs[end], ys[end])
                    push!(xs, xs[1]); push!(ys, ys[1])
                end
                if !isempty(xs)
                    Plots.plot!(p, xs, ys, seriestype=:shape, fillalpha=0.1, linecolor=:black,
                        lw=0.5, label=nothing)
                end
            end
        end
    end

    # 2. Graph edges
    if !isnothing(g)
        for e in Graphs.edges(g)
            u, v = Graphs.src(e), Graphs.dst(e)
            if u <= length(centroids) && v <= length(centroids)
                p1, p2 = centroids[u], centroids[v]
                Plots.plot!(p, [p1[1], p2[1]], [p1[2], p2[2]], color=edge_color, lw=1.5,
                    alpha=edge_alpha, label=nothing)
            end
        end
    end

    # 3. Raw data points (if provided)
    if !isnothing(pts)
        Plots.scatter!(p, [pt[1] for pt in pts], [pt[2] for pt in pts],
                       markersize=1, color=:gray, alpha=0.3, label="Points")
    end

    # 4. Centroids
    if !isnothing(centroids) && !isempty(centroids)
        xs, ys = _centroids_xy(centroids)
        Plots.scatter!(p, xs, ys, markersize=node_size, color=node_color,
            markerstrokecolor=:white, label="Centroids")
    end

    # 5. Boundary Hull (if provided)
    if !isnothing(hull_coords) && length(hull_coords) > 2
        bx, by = _poly_xy(hull_coords)
        if !isempty(bx)
            Plots.plot!(p, bx, by, color=:black, lw=2, ls=:dash, label=nothing)
        end
    end

    return p
end

# Keyword-based dispatch accepting `au` NamedTuple/Dict or raw coordinates
function spatial_graph_plot(; au=nothing, pts=nothing, plot_title="Spatial Partitioning",
    title=plot_title, mode::Symbol=:plots, kwargs...)
    if !isnothing(au)
        if mode == :leaflet || mode == :html
            return leaflet_spatial_graph(au; pts=pts, title=title, kwargs...)
        end
        polygons = hasproperty(au, :polygons) ? au.polygons : (haskey(au,
            :polygons) ? au[:polygons] : nothing)
        centroids = hasproperty(au, :centroids) ? au.centroids : (haskey(au,
            :centroids) ? au[:centroids] : nothing)
        g = hasproperty(au, :graph) ? au.graph : (haskey(au, :graph) ? au[:graph] : nothing)
        hull = hasproperty(au, :hull_coords) ? au.hull_coords : (haskey(au,
            :hull_coords) ? au[:hull_coords] : nothing)
        return spatial_graph_plot(centroids, g; polygons=polygons, hull_coords=hull, pts=pts,
            title=title, mode=mode, kwargs...)
    else
        error("spatial_graph_plot requires either (centroids, graph) or `au=(...)`.")
    end
end


"""
    plot_kde_simple(s_coord_tuple; grid_res=600, sd_extension_factor=0.25, title="Spatial
      Intensity (KDE)")
    plot_kde_simple(df::DataFrame; x=:s_x, y=:s_y, grid_res=600, sd_extension_factor=0.25,
      title="Spatial Intensity (KDE)")

Generates a 2D heatmap of spatial intensity using Kernel Density Estimation (KDE) with point
  overlay.
"""
function plot_kde_simple(s_coord_tuple; grid_res=600, sd_extension_factor=0.25,
    title="Spatial Intensity (KDE)")
    t_idx_dummy = ones(Int, length(s_coord_tuple))
    x_g, y_g, intensity = estimate_local_kde_with_extrapolation(s_coord_tuple, t_idx_dummy, 1;
        grid_res=grid_res, sd_extension_factor=sd_extension_factor)

    plt = Plots.heatmap(x_g, y_g, intensity',
                        title=title,
                        c=:viridis,
                        aspect_ratio=:equal,
                        xlabel="X", ylabel="Y")
    Plots.scatter!(plt, [p[1] for p in s_coord_tuple], [p[2] for p in s_coord_tuple],
                   markersize=2, markercolor=:white, markeralpha=0.5, label="Points")
    return plt
end

function plot_kde_simple(df::DataFrame; x=:s_x, y=:s_y, grid_res=600, sd_extension_factor=0.25,
    title="Spatial Intensity (KDE)")
    if !hasproperty(df, x) || !hasproperty(df, y)
        error("Input DataFrame for plot_kde_simple expects columns `:$x` and `:$y`. Override with x=... and y=... if using different names.")
    end
    s_coord_tuple = tuple.(df[!, x], df[!, y])
    return plot_kde_simple(s_coord_tuple; grid_res=grid_res,
        sd_extension_factor=sd_extension_factor, title=title)
end


"""
    render_paths!(p::Plots.Plot, paths; au=nothing, centroids=nothing, polygons=nothing,
      labels=nothing, color=:black, lw=1.0, markersize=2.0)

Adds movement trajectories/paths onto an existing plot `p`.
Accepts `paths` as:
1. `Vector` of `Vector`s of `(x, y)` coordinate tuples/vectors.
2. `Matrix{<:Integer}` of spatial unit indices (e.g. `n_indiv × n_steps`) when `au` or
  `centroids` is provided.
"""
function render_paths!(p::Plots.Plot, paths; au=nothing, centroids=nothing, polygons=nothing,
    labels=nothing, color=:black, lw=1.0, markersize=2.0)
    # Extract background polygons if provided
    poly_list = if !isnothing(au) && hasproperty(au, :polygons)
        au.polygons
    elseif !isnothing(polygons)
        polygons
    else
        nothing
    end

    if !isnothing(poly_list)
        for poly in poly_list
            if length(poly) > 2
                px = [pt[1] for pt in poly]
                py = [pt[2] for pt in poly]
                Plots.plot!(p, px, py, seriestype=:shape, fillalpha=0.03, linecolor=:gray,
                    linewidth=0.3, label=nothing)
            end
        end
    end

    # Extract centroids if provided
    cents = if !isnothing(au) && hasproperty(au, :centroids)
        au.centroids
    elseif !isnothing(centroids)
        centroids
    else
        nothing
    end

    # Case 1: Matrix of unit indices
    if isa(paths, AbstractMatrix)
        if isnothing(cents)
            error("When `paths` is an integer unit matrix, `au` or `centroids` must be provided to map unit indices to coordinates.")
        end
        n_indiv, n_steps = size(paths)
        for i in 1:n_indiv
            xs = Float64[cents[paths[i, t]][1] for t in 1:n_steps]
            ys = Float64[cents[paths[i, t]][2] for t in 1:n_steps]
            lab = (labels === nothing || i > length(labels)) ? nothing : labels[i]
            Plots.plot!(p, xs, ys, marker=:circle, markersize=markersize, lw=lw, label=lab,
                color=color)
        end
        return p
    end

    # Case 2: Vector of coordinate sequences
    if isa(paths, AbstractVector) && !isempty(paths) && !isa(paths[1], Number) && isa(paths[1],
        AbstractVector)
        for (i, path) in enumerate(paths)
            xs = [pt[1] for pt in path]
            ys = [pt[2] for pt in path]
            lab = (labels === nothing || i > length(labels)) ? nothing : labels[i]
            Plots.plot!(p, xs, ys, marker=:circle, markersize=markersize, lw=lw, label=lab,
                color=color)
        end
        return p
    end
    error("render_paths! expects paths as Matrix of unit indices or Vector of Vector of (x,y) points.")
end


"""
    map_point_occupancy(polygons, centroids, id::Integer, step::Integer=1;
      highlight_color=:red, title="")

Highlights the specific spatial polygon unit occupied by index `id` on a map.
"""
function map_point_occupancy(polygons, centroids, id::Integer, step::Integer=1;
    highlight_color=:red, title::String="")
    title_str = isempty(title) ? "Occupancy (Unit $id, Step $step)" : title
    p = Plots.plot(aspect_ratio=:equal, legend=false, title=title_str)

    # 1. Background polygons
    if !isnothing(polygons)
        for poly in polygons
            if length(poly) > 2
                xs, ys = _poly_xy(poly)
                if !isempty(xs)
                    Plots.plot!(p, xs, ys, seriestype=:shape, fillcolor=:white,
                        linecolor=:gray, lw=0.3, alpha=0.6, label=nothing)
                end
            end
        end
    end

    # 2. Highlight occupied polygon
    if !isnothing(polygons) && 1 <= id <= length(polygons)
        poly = polygons[id]
        if length(poly) > 2
            xs, ys = _poly_xy(poly)
            if !isempty(xs)
                Plots.plot!(p, xs, ys, seriestype=:shape, fillcolor=highlight_color,
                    linecolor=:black, lw=0.8, alpha=0.5, label=nothing)
            end
        end
    end

    # 3. Centroids
    if !isnothing(centroids) && !isempty(centroids)
        xs, ys = _centroids_xy(centroids)
        Plots.scatter!(p, xs, ys, markersize=3, color=:black, label=nothing)
    end

    return p
end

function map_individual_occupancy(tracts, id::Integer, step::Integer, au_context)
    polys = get(au_context, :polygons, nothing)
    cents = get(au_context, :centroids, nothing)
    unit_idx = tracts[id, step]
    return map_point_occupancy(polys, cents, unit_idx, step;
        title="Occupancy of Mark $id at Step $step")
end


# -----------------------------------------------------------------------------
# Section 2b: Movement & Trajectory Visualization Primitives
# -----------------------------------------------------------------------------

"""
    plot_hsi_choropleth(hsi, au; title="Habitat Suitability Index (HSI)",
                        cmap=:viridis, show_centroids=false,
                        show_hull=true, show_colorbar=true,
                        border_color=:gray, lw=0.4, kwargs...)

Renders a spatial polygon choropleth map of the Habitat Suitability Index (HSI)
over the tessellated spatial domain.

# Mathematical Formulation
Each spatial unit ``i \\in \\{1,\\dots,S\\}`` is colored according to its local
suitability metric ``\\text{HSI}_i \\in [0, 1]``.

# Arguments
- `hsi::AbstractVector{<:Real}`: Habitat suitability values across ``S`` units.
- `au::NamedTuple`: Spatial tessellation containing `:polygons` and `:centroids`.
- `title::String`: Plot title. Default: `"Habitat Suitability Index (HSI)"`.
- `cmap::Symbol`: Colormap scheme (e.g. `:viridis`, `:cividis`, `:plasma`).
- `show_centroids::Bool`: Whether to plot unit centroid points. Default: `false`.
- `show_hull::Bool`: Whether to outline the domain boundary hull. Default: `true`.
- `show_colorbar::Bool`: Whether to display the value colorbar. Default: `true`.
- `border_color`: Color of polygon unit borders. Default: `:gray`.
- `lw::Real`: Line width for polygon borders. Default: `0.4`.

# Returns
- `Plots.Plot`: Rendered choropleth figure.
"""
function plot_hsi_choropleth(
    hsi::AbstractVector{<:Real},
    au::NamedTuple;
    mode::Symbol = :plots,
    title::String = "Habitat Suitability Index (HSI)",
    cmap::Symbol = :viridis,
    show_centroids::Bool = false,
    show_hull::Bool = true,
    show_colorbar::Bool = true,
    border_color = :gray,
    lw::Real = 0.4,
    kwargs...
)
    if mode == :leaflet || mode == :html
        return leaflet_hsi_map(hsi, au; title=title, cmap=cmap, show_centroids=show_centroids,
                               show_hull=show_hull, kwargs...)
    end
    polys = hasproperty(au, :polygons) ? au.polygons : au[:polygons]
    p = _choropleth_impl(
        polys, hsi;
        title = title,
        cmap = cmap,
        show_colorbar = show_colorbar,
        colorbar_label = "HSI",
        border_color = border_color,
        lw = lw,
        kwargs...
    )

    if show_centroids && hasproperty(au, :centroids)
        xs = [c[1] for c in au.centroids]
        ys = [c[2] for c in au.centroids]
        Plots.scatter!(p, xs, ys, markersize=2.5, color=:black,
                       markerstrokecolor=:white, markerstrokewidth=0.5,
                       label=nothing)
    end

    if show_hull && hasproperty(au, :hull_coords) && !isnothing(au.hull_coords)
        bx, by = _poly_xy(au.hull_coords)
        if !isempty(bx)
            Plots.plot!(p, bx, by, color=:black, lw=1.5, ls=:dash, label=nothing)
        end
    end

    return p
end

"""
    plot_diffusion_map(diffusion, au; title="Spatial Diffusion / Dispersal Field",
                       cmap=:plasma, show_colorbar=true, border_color=:gray,
                       lw=0.4, kwargs...)

Renders a polygon choropleth map representing the spatial distribution of
dispersal or diffusion rates ``D`` across spatial units.

# Arguments
- `diffusion::Union{Real, AbstractVector{<:Real}}`: Scalar or ``S``-vector of diffusion rates.
- `au::NamedTuple`: Spatial tessellation object with `:polygons`.
- `title::String`: Plot title. Default: `"Spatial Diffusion / Dispersal Field"`.
- `cmap::Symbol`: Colormap scheme. Default: `:plasma`.
- `show_colorbar::Bool`: Whether to display the value colorbar. Default: `true`.

# Returns
- `Plots.Plot`: Rendered diffusion map figure.
"""
function plot_diffusion_map(
    diffusion::Union{Real, AbstractVector{<:Real}},
    au::NamedTuple;
    mode::Symbol = :plots,
    title::String = "Spatial Diffusion / Dispersal Field",
    cmap::Symbol = :plasma,
    show_colorbar::Bool = true,
    border_color = :gray,
    lw::Real = 0.4,
    kwargs...
)
    if mode == :leaflet || mode == :html
        return leaflet_diffusion_map(diffusion, au; title=title, cmap=cmap, kwargs...)
    end
    polys = hasproperty(au, :polygons) ? au.polygons : au[:polygons]
    S = length(polys)
    diff_vec = if diffusion isa Real
        fill(Float64(diffusion), S)
    else
        Float64.(diffusion)
    end

    return _choropleth_impl(
        polys, diff_vec;
        title = title,
        cmap = cmap,
        show_colorbar = show_colorbar,
        colorbar_label = "Diffusion (D)",
        border_color = border_color,
        lw = lw,
        kwargs...
    )
end

"""
    plot_residence_time_map(Gamma, au; title="Stationary Residence Distribution (π)",
                            cmap=:inferno, show_colorbar=true, kwargs...)

Computes and maps the stationary distribution ``\\pi`` satisfying ``\\pi \\Gamma = \\pi``,
reflecting long-term spatial occupancy and residence probabilities under the fitted
movement kernel.

# Arguments
- `Gamma::AbstractMatrix{<:Real}`: Markov transition probability matrix (``S \\times S``).
- `au::NamedTuple`: Spatial tessellation with `:polygons`.
- `title::String`: Plot title. Default: `"Stationary Residence Distribution (π)"`.
- `cmap::Symbol`: Colormap scheme. Default: `:inferno`.

# Returns
- `Plots.Plot`: Rendered stationary distribution choropleth.
"""
function plot_residence_time_map(
    Gamma::AbstractMatrix{<:Real},
    au::NamedTuple;
    mode::Symbol = :plots,
    title::String = "Stationary Residence Distribution (π)",
    cmap::Symbol = :inferno,
    show_colorbar::Bool = true,
    kwargs...
)
    if mode == :leaflet || mode == :html
        return leaflet_residence_time_map(Gamma, au; title=title, cmap=cmap, kwargs...)
    end
    S = size(Gamma, 1)
    # Power iteration for robust computation of left principal eigenvector
    pi_vec = fill(1.0 / S, S)
    for _ in 1:250
        pi_next = vec(pi_vec' * Gamma)
        s_sum = sum(pi_next)
        s_sum > 1e-12 && (pi_next ./= s_sum)
        pi_vec = pi_next
    end

    polys = hasproperty(au, :polygons) ? au.polygons : au[:polygons]
    return _choropleth_impl(
        polys, pi_vec;
        title = title,
        cmap = cmap,
        show_colorbar = show_colorbar,
        colorbar_label = "Stationary π",
        kwargs...
    )
end

"""
    plot_advection_arrows(au; hsi=nothing, velocity=1.0, Gamma=nothing,
                          relationship=:exponential, arrow_scale=1.0,
                          arrow_color=:white, background=:hsi, cmap=:viridis,
                          title="Advection Drift & Velocity Field",
                          min_speed_quantile=0.05, kwargs...)

Generates a spatial vector field (arrow / quiver plot) depicting the magnitude and
direction of directed movement / advective drift across spatial units.

# Mathematical Formulation
Advection vectors ``\\mathbf{v}_i = (v_x, v_y)_i`` at unit centroids ``\\mathbf{x}_i``
are computed from the local habitat gradient:
```math
\\mathbf{v}_i = v \\cdot \\sum_{j \\in \\mathcal{N}(i)} w_{ij} \\, f(\\text{HSI}_j - \\text{HSI}_i) \\,
\\frac{\\mathbf{x}_j - \\mathbf{x}_i}{\\|\\mathbf{x}_j - \\mathbf{x}_i\\|}
```
where ``f(\\Delta h)`` is the response function (`:exponential`, `:logistic`, or `:linear`),
or directly from the net transition flux ``\\mathbf{J}_i = \\sum_j \\Gamma_{ij}(\\mathbf{x}_j - \\mathbf{x}_i)``.

# Arguments
- `au::NamedTuple`: Spatial tessellation with `:centroids`, `:polygons`, and `:W`.
- `hsi::Union{Nothing, AbstractVector{<:Real}}`: Habitat suitability vector (length ``S``).
- `velocity::Real`: Movement velocity / advection sensitivity multiplier ``v``.
- `Gamma::Union{Nothing, AbstractMatrix{<:Real}}`: Optional transition probability matrix.
- `relationship::Symbol`: Response form (`:exponential`, `:logistic`, `:linear`).
- `arrow_scale::Real`: Scaling factor for visual arrow lengths. Default: `1.0`.
- `arrow_color`: Color of arrow lines/heads. Default: `:white`.
- `background::Symbol`: Background style (`:hsi`, `:polygons`, `:none`). Default: `:hsi`.
- `cmap::Symbol`: Colormap for background or speed. Default: `:viridis`.
- `title::String`: Plot title.
- `min_speed_quantile::Real`: Minimum speed quantile threshold below which arrows are omitted.

# Returns
- `Plots.Plot`: Rendered vector field map.
"""
function plot_advection_arrows(
    au::NamedTuple;
    mode::Symbol = :plots,
    hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
    velocity::Real = 1.0,
    Gamma::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    relationship::Symbol = :exponential,
    arrow_scale::Real = 1.0,
    arrow_color = :white,
    background::Symbol = :hsi,
    cmap::Symbol = :viridis,
    title::String = "Advection Drift & Velocity Field",
    min_speed_quantile::Real = 0.05,
    kwargs...
)
    if mode == :leaflet || mode == :html
        col_str = arrow_color isa Symbol ? string(arrow_color) : string(arrow_color)
        return leaflet_advection_arrows(au; hsi=hsi, velocity=velocity, Gamma=Gamma,
                                        relationship=relationship, arrow_scale=arrow_scale,
                                        arrow_color=col_str, background=background,
                                        cmap=cmap, title=title, min_speed_quantile=min_speed_quantile,
                                        kwargs...)
    end
    S = length(au.centroids)
    vx = zeros(Float64, S)
    vy = zeros(Float64, S)

    if !isnothing(hsi) && hasproperty(au, :W) && !isnothing(au.W)
        W = au.W
        rows = rowvals(W)
        vals = nonzeros(W)
        hsi_vec = Float64.(hsi)

        for i in 1:S
            ci = au.centroids[i]
            for j_idx in nzrange(W, i)
                j = rows[j_idx]
                i == j && continue
                cj = au.centroids[j]
                dh = hsi_vec[j] - hsi_vec[i]
                if dh > 0.0
                    dx = cj[1] - ci[1]
                    dy = cj[2] - ci[2]
                    dist = sqrt(dx^2 + dy^2) + 1e-9
                    weight = if relationship == :exponential
                        vals[j_idx] * exp(dh)
                    elseif relationship == :logistic
                        vals[j_idx] / (1.0 + exp(-4.0 * dh))
                    else
                        vals[j_idx] * dh
                    end
                    vx[i] += weight * (dx / dist)
                    vy[i] += weight * (dy / dist)
                end
            end
        end
        vx .*= Float64(velocity)
        vy .*= Float64(velocity)
    elseif !isnothing(Gamma)
        for i in 1:S
            ci = au.centroids[i]
            for j in 1:S
                i == j && continue
                cj = au.centroids[j]
                p_ij = Gamma[i, j]
                vx[i] += p_ij * (cj[1] - ci[1])
                vy[i] += p_ij * (cj[2] - ci[2])
            end
        end
    end

    # Background plot
    p = if background == :hsi && !isnothing(hsi) && hasproperty(au, :polygons)
        plot_hsi_choropleth(hsi, au; title=title, cmap=cmap, border_color=:gray40,
                            lw=0.3, show_centroids=false, kwargs...)
    elseif background == :polygons && hasproperty(au, :polygons)
        p_base = Plots.plot(aspect_ratio=:equal, title=title, legend=false; kwargs...)
        for poly in au.polygons
            if length(poly) > 2
                xs, ys = _poly_xy(poly)
                Plots.plot!(p_base, xs, ys, seriestype=:shape, fillcolor=:white,
                            linecolor=:gray60, lw=0.4, label=nothing)
            end
        end
        p_base
    else
        Plots.plot(aspect_ratio=:equal, title=title, legend=false; kwargs...)
    end

    # Calculate arrow lengths and scaling
    speeds = sqrt.(vx.^2 .+ vy.^2)
    max_sp = maximum(speeds)

    if max_sp > 1e-12
        # Determine average nearest centroid distance to scale arrows proportionally
        dists = Float64[]
        for i in 1:min(S, 20)
            ci = au.centroids[i]
            nn_d = minimum([sqrt((cj[1]-ci[1])^2 + (cj[2]-ci[2])^2)
                            for (j, cj) in enumerate(au.centroids) if j != i])
            push!(dists, nn_d)
        end
        avg_spacing = !isempty(dists) ? mean(dists) : 10.0
        target_max_arrow = avg_spacing * 0.75 * arrow_scale
        scale_fac = target_max_arrow / max_sp

        speed_cutoff = quantile(speeds, clamp(min_speed_quantile, 0.0, 0.5))

        xs_arr = Float64[]
        ys_arr = Float64[]
        qx = Float64[]
        qy = Float64[]

        for i in 1:S
            if speeds[i] >= speed_cutoff
                push!(xs_arr, au.centroids[i][1])
                push!(ys_arr, au.centroids[i][2])
                push!(qx, vx[i] * scale_fac)
                push!(qy, vy[i] * scale_fac)
            end
        end

        if !isempty(xs_arr)
            Plots.quiver!(p, xs_arr, ys_arr, quiver=(qx, qy),
                          color=arrow_color, lw=1.5, label=nothing)
            Plots.scatter!(p, xs_arr, ys_arr, markersize=1.8,
                           color=arrow_color, label=nothing)
        end
    end

    return p
end

const plot_velocity_field = plot_advection_arrows

"""
    plot_tracks_on_map(paths, au; hsi=nothing, background=:polygons,
                       title="Movement Trajectories",
                       palette=:tab10, show_start_end=true,
                       alpha=0.85, lw=1.8, max_paths=25, kwargs...)

Maps individual movement tracks / dispersal trajectories across the spatial domain,
with distinct track colorings, release (start) and recapture (end) markers.

# Arguments
- `paths`: Trajectory matrix (size ``n_{\\text{indiv}} \\times (n_{\\text{steps}}+1)``)
  of spatial unit indices, or a `Vector` of coordinate sequence tuples.
- `au::NamedTuple`: Spatial tessellation object with `:centroids` and `:polygons`.
- `hsi::Union{Nothing, AbstractVector{<:Real}}`: Optional HSI vector for background choropleth.
- `background::Symbol`: Background style (`:polygons`, `:hsi`, `:none`). Default: `:polygons`.
- `title::String`: Plot title. Default: `"Movement Trajectories"`.
- `palette::Symbol`: Color palette for differentiating individuals. Default: `:tab10`.
- `show_start_end::Bool`: Whether to mark starts (green circles) and ends (red stars).
- `alpha::Real`: Line transparency. Default: `0.85`.
- `lw::Real`: Trajectory line width. Default: `1.8`.
- `max_paths::Int`: Maximum number of paths to render to avoid clutter. Default: `25`.

# Returns
- `Plots.Plot`: Rendered trajectory map.
"""
function plot_tracks_on_map(
    paths,
    au::NamedTuple;
    mode::Symbol = :plots,
    hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
    background::Symbol = :polygons,
    title::String = "Movement Trajectories",
    palette::Symbol = :tab10,
    show_start_end::Bool = true,
    alpha::Real = 0.85,
    lw::Real = 1.8,
    max_paths::Int = 25,
    kwargs...
)
    if mode == :leaflet || mode == :html
        return leaflet_tracks_map(paths, au; hsi=hsi, background=background, title=title,
                                  palette=palette, show_start_end=show_start_end,
                                  max_paths=max_paths, kwargs...)
    end
    # Background setup
    p = if background == :hsi && !isnothing(hsi) && hasproperty(au, :polygons)
        plot_hsi_choropleth(hsi, au; title=title, border_color=:gray40, lw=0.3,
                            show_centroids=false, kwargs...)
    elseif background == :polygons && hasproperty(au, :polygons)
        p_base = Plots.plot(aspect_ratio=:equal, title=title, legend=:outerright; kwargs...)
        for poly in au.polygons
            if length(poly) > 2
                xs, ys = _poly_xy(poly)
                Plots.plot!(p_base, xs, ys, seriestype=:shape, fillcolor=:white,
                            linecolor=:gray75, lw=0.4, alpha=0.8, label=nothing)
            end
        end
        p_base
    else
        Plots.plot(aspect_ratio=:equal, title=title, legend=:outerright; kwargs...)
    end

    cents = au.centroids
    n_paths_total = size(paths, 1)
    n_to_render = min(n_paths_total, max_paths)
    cs = _get_cscheme(palette)

    for i in 1:n_to_render
        t_col = get(cs, n_to_render > 1 ? (i - 1) / (n_to_render - 1) : 0.5)

        xs = Float64[]
        ys = Float64[]

        if isa(paths, AbstractMatrix)
            n_steps = size(paths, 2)
            xs = [cents[paths[i, t]][1] for t in 1:n_steps]
            ys = [cents[paths[i, t]][2] for t in 1:n_steps]
        elseif isa(paths, AbstractVector) && isa(paths[1], AbstractVector)
            xs = [pt[1] for pt in paths[i]]
            ys = [pt[2] for pt in paths[i]]
        end

        if !isempty(xs)
            Plots.plot!(p, xs, ys, color=t_col, lw=lw, alpha=alpha,
                        label="Tag $i", marker=:none)

            if show_start_end
                # Start marker: filled green circle
                Plots.scatter!(p, [xs[1]], [ys[1]], markershape=:circle,
                               markersize=4.5, color=:lime, markerstrokecolor=:black,
                               markerstrokewidth=0.8, label=nothing)
                # End marker: filled red triangle / star
                Plots.scatter!(p, [xs[end]], [ys[end]], markershape=:star5,
                               markersize=6.0, color=:crimson, markerstrokecolor=:black,
                               markerstrokewidth=0.8, label=nothing)
            end
        end
    end

    # Add legend proxy for start/end indicators
    if show_start_end
        Plots.scatter!(p, [NaN], [NaN], markershape=:circle, markersize=4.0,
                       color=:lime, markerstrokecolor=:black, label="Release (Start)")
        Plots.scatter!(p, [NaN], [NaN], markershape=:star5, markersize=5.0,
                       color=:crimson, markerstrokecolor=:black, label="Recapture (End)")
    end

    return p
end

"""
    plot_dispersal_kernel(Gamma, au; title="Dispersal Kernel Distance Decay",
                          n_bins=15, fit_exponential=true,
                          color=:steelblue, kwargs...)

Plots one-step Markov transition probabilities ``\\Gamma_{ij}`` against pairwise spatial
distances ``d_{ij} = \\|\\mathbf{x}_i - \\mathbf{x}_j\\|``, highlighting isotropic
dispersal attenuation and empirical kernel decay.

# Arguments
- `Gamma::AbstractMatrix{<:Real}`: Markov transition probability matrix (``S \\times S``).
- `au::NamedTuple`: Spatial tessellation with `:centroids`.
- `title::String`: Plot title.
- `n_bins::Int`: Number of distance bins for empirical average curve. Default: `15`.
- `fit_exponential::Bool`: Whether to overlay fitted exponential decay line. Default: `true`.
- `color`: Primary color for data points. Default: `:steelblue`.

# Returns
- `Plots.Plot`: Rendered dispersal distance decay curve.
"""
function plot_dispersal_kernel(
    Gamma::AbstractMatrix{<:Real},
    au::NamedTuple;
    mode::Symbol = :plots,
    title::String = "Dispersal Kernel Distance Decay",
    n_bins::Int = 15,
    fit_exponential::Bool = true,
    color = :steelblue,
    kwargs...
)
    if mode == :leaflet || mode == :html
        return leaflet_dispersal_kernel(Gamma, au; title=title, kwargs...)
    end
    S = size(Gamma, 1)
    cents = au.centroids

    distances = Float64[]
    probs = Float64[]

    for i in 1:S
        ci = cents[i]
        for j in 1:S
            i == j && continue
            cj = cents[j]
            d = sqrt((cj[1] - ci[1])^2 + (cj[2] - ci[2])^2)
            p = Gamma[i, j]
            push!(distances, d)
            push!(probs, p)
        end
    end

    p = Plots.plot(
        title = title,
        xlabel = "Pairwise Distance (km)",
        ylabel = "Transition Probability (Γ_ij)",
        legend = :topright;
        kwargs...
    )

    # Scatter points
    Plots.scatter!(p, distances, probs, color=color, alpha=0.35, markersize=2.5,
                   markerstrokewidth=0, label="Pairs (i, j)")

    # Binned mean curve
    if !isempty(distances)
        d_min, d_max = minimum(distances), maximum(distances)
        bin_edges = range(d_min, d_max, length=n_bins+1)
        bin_mids = Float64[]
        bin_means = Float64[]
        bin_se = Float64[]

        for b in 1:n_bins
            idx = findall(d -> bin_edges[b] <= d < bin_edges[b+1], distances)
            if !isempty(idx)
                push!(bin_mids, (bin_edges[b] + bin_edges[b+1]) / 2.0)
                m = mean(probs[idx])
                se = length(idx) > 1 ? std(probs[idx]) / sqrt(length(idx)) : 0.0
                push!(bin_means, m)
                push!(bin_se, se)
            end
        end

        if !isempty(bin_mids)
            Plots.plot!(p, bin_mids, bin_means, yerror=bin_se, color=:darkorange,
                        lw=2.5, marker=:circle, markersize=4.0, label="Binned Mean ± SE")
        end

        # Exponential fit: P(d) = a * exp(-b * d)
        if fit_exponential && length(probs) > 5
            valid_idx = findall(p -> p > 1e-9, probs)
            if length(valid_idx) > 5
                d_v = distances[valid_idx]
                log_p = log.(probs[valid_idx])
                # Linear regression on log scale
                x_mean = mean(d_v)
                y_mean = mean(log_p)
                denom = sum((d_v .- x_mean).^2)
                if denom > 1e-12
                    slope = sum((d_v .- x_mean) .* (log_p .- y_mean)) / denom
                    intercept = y_mean - slope * x_mean

                    if slope < 0
                        d_eval = range(d_min, d_max, length=100)
                        p_fit = exp.(intercept .+ slope .* d_eval)
                        decay_dist = round(-1.0 / slope, digits=1)
                        Plots.plot!(p, d_eval, p_fit, color=:crimson, lw=2.0, ls=:dash,
                                    label="Exp Decay (λ ≈ $(decay_dist) km)")
                    end
                end
            end
        end
    end

    return p
end

"""
    plot_step_length_distribution(paths, au; title="Step Length & Turning Angle Diagnostics",
                                  kwargs...)

Computes and plots empirical step lengths (km) and turning angle distributions
from individual trajectory paths.

# Arguments
- `paths`: Matrix of unit indices or vector of coordinate sequences.
- `au::NamedTuple`: Spatial tessellation with `:centroids`.
- `title::String`: Plot super-title.

# Returns
- `Plots.Plot`: 2-panel composite figure showing step lengths and turning angles.
"""
function plot_step_length_distribution(
    paths,
    au::NamedTuple;
    mode::Symbol = :plots,
    title::String = "Step Length & Turning Angle Diagnostics",
    kwargs...
)
    if mode == :leaflet || mode == :html
        return leaflet_step_diagnostics(paths, au; title=title, kwargs...)
    end
    cents = au.centroids
    n_indiv = size(paths, 1)
    n_steps = size(paths, 2)

    step_lengths = Float64[]
    turning_angles = Float64[]

    for i in 1:n_indiv
        xs = [cents[paths[i, t]][1] for t in 1:n_steps]
        ys = [cents[paths[i, t]][2] for t in 1:n_steps]

        for t in 2:n_steps
            d = sqrt((xs[t] - xs[t-1])^2 + (ys[t] - ys[t-1])^2)
            push!(step_lengths, d)

            if t >= 3
                # Turning angle between step (t-1 -> t) and (t-2 -> t-1)
                dx1, dy1 = xs[t-1] - xs[t-2], ys[t-1] - ys[t-2]
                dx2, dy2 = xs[t] - xs[t-1], ys[t] - ys[t-1]
                a1 = atan(dy1, dx1)
                a2 = atan(dy2, dx2)
                d_ang = a2 - a1
                # Wrap to [-pi, pi]
                d_ang = atan(sin(d_ang), cos(d_ang))
                push!(turning_angles, d_ang)
            end
        end
    end

    p1 = Plots.histogram(
        step_lengths, bins=15, color=:royalblue, linecolor=:white,
        title="Step Length Distribution", xlabel="Step Length (km)",
        ylabel="Frequency", legend=false
    )

    p2 = Plots.histogram(
        turning_angles, bins=15, color=:darkorange, linecolor=:white,
        title="Turning Angle Distribution", xlabel="Turning Angle (rad)",
        ylabel="Frequency", legend=false
    )
    Plots.vline!(p2, [0.0], color=:red, ls=:dash, lw=1.5)

    return Plots.plot(p1, p2, layout=(1, 2), size=(800, 350), plot_title=title; kwargs...)
end

"""
    plot_regional_connectivity_matrix(C_regional; strata_names=nothing,
                                      title="Macro-Regional Transfer Matrix",
                                      cmap=:Blues, show_values=true, kwargs...)

Renders a heatmap of the macro-regional transition probability matrix ``C_{rs}``
with annotated cell percentages.

# Arguments
- `C_regional::AbstractMatrix{<:Real}`: Regional transition matrix (``R \\times R``).
- `strata_names::Union{Nothing, Vector{String}}`: Names of the regional strata.
- `title::String`: Plot title. Default: `"Macro-Regional Transfer Matrix"`.
- `cmap::Symbol`: Heatmap colormap. Default: `:Blues`.
- `show_values::Bool`: Whether to annotate text percentages in cells. Default: `true`.

# Returns
- `Plots.Plot`: Rendered connectivity matrix heatmap.
"""
function plot_regional_connectivity_matrix(
    C_regional::AbstractMatrix{<:Real};
    mode::Symbol = :plots,
    strata_names::Union{Nothing, Vector{String}} = nothing,
    title::String = "Macro-Regional Transfer Matrix",
    cmap::Symbol = :Blues,
    show_values::Bool = true,
    kwargs...
)
    if mode == :leaflet || mode == :html
        return leaflet_regional_connectivity(C_regional; strata_names=strata_names, title=title, kwargs...)
    end
    R = size(C_regional, 1)
    names = if !isnothing(strata_names) && length(strata_names) == R
        strata_names
    elseif R == 2
        ["West", "East"]
    else
        ["Stratum $i" for i in 1:R]
    end

    # Plots.heatmap expects y-axis as rows, x-axis as cols
    # Transpose so that from-stratum is vertical axis and to-stratum is horizontal axis
    p = Plots.heatmap(
        names, names, C_regional',
        color = cmap,
        clims = (0.0, 1.0),
        aspect_ratio = :equal,
        title = title,
        xlabel = "To Region",
        ylabel = "From Region",
        colorbar_title = "Transfer Prob.",
        yflip = true;
        kwargs...
    )

    if show_values
        for i in 1:R
            for j in 1:R
                val = C_regional[i, j]
                txt = @sprintf("%.1f%%", val * 100.0)
                txt_color = val > 0.5 ? :white : :black
                Plots.annotate!(p, [(j, i, Plots.text(txt, 10, txt_color, :center, :bold))])
            end
        end
    end

    return p
end

"""
    plot_movement_dashboard(result, paths; hsi=nothing, strata=nothing,
                            title="BSTM Movement Diagnostics Dashboard", kwargs...)

Creates a comprehensive 4-panel composite dashboard summarizing:
1. (A) Habitat Suitability & Advective Velocity Vectors
2. (B) Movement Tracks / Trajectories on Tessellation
3. (C) Dispersal Kernel Distance Decay Curve
4. (D) Macro-Regional Transfer Connectivity Matrix

# Arguments
- `result::NamedTuple`: Result object from `run_movement_simple` or `synthesize_adr_results`.
- `paths::AbstractMatrix{<:Integer}`: Simulated individual trajectory paths.
- `hsi::Union{Nothing, AbstractVector{<:Real}}`: Optional HSI vector.
- `strata::Union{Nothing, AbstractVector}`: Regional strata definitions.
- `title::String`: Dashboard super-title.

# Returns
- `Plots.Plot`: 4-panel composite diagnostic dashboard.
"""
function plot_movement_dashboard(
    result::NamedTuple,
    paths::AbstractMatrix{<:Integer};
    mode::Symbol = :plots,
    hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
    strata::Union{Nothing, AbstractVector} = nothing,
    title::String = "BSTM Movement Diagnostics Dashboard",
    kwargs...
)
    if mode == :leaflet || mode == :html
        return leaflet_movement_dashboard(result, paths; hsi=hsi, strata=strata, title=title, kwargs...)
    end
    au = result.au
    Gamma = result.transition_matrix
    hsi_vec = !isnothing(hsi) ? hsi : (hasproperty(result.opts, :hsi) ? result.opts.hsi : nothing)

    # Subplot A: Habitat & Advection Arrows
    p_a = plot_advection_arrows(
        au;
        hsi = hsi_vec,
        Gamma = Gamma,
        background = !isnothing(hsi_vec) ? :hsi : :polygons,
        title = "(A) Habitat & Advective Drift",
        arrow_scale = 1.0
    )

    # Subplot B: Movement Tracks
    p_b = plot_tracks_on_map(
        paths, au;
        hsi = hsi_vec,
        background = :polygons,
        title = "(B) Simulated Trajectories",
        max_paths = 15
    )

    # Subplot C: Dispersal Distance Decay
    p_c = plot_dispersal_kernel(
        Gamma, au;
        title = "(C) Dispersal Kernel Decay",
        fit_exponential = true
    )

    # Subplot D: Regional Connectivity
    cents_x = [c[1] for c in au.centroids]
    mid_x = (minimum(cents_x) + maximum(cents_x)) / 2.0
    strata_vec = !isnothing(strata) ? strata : [x > mid_x ? "East" : "West" for x in cents_x]
    C_reg = calculate_regional_connectivity(Gamma, strata_vec)
    strata_unique = unique(strata_vec)
    p_d = plot_regional_connectivity_matrix(
        C_reg;
        strata_names = strata_unique,
        title = "(D) Macro-Regional Connectivity"
    )

    return Plots.plot(p_a, p_b, p_c, p_d, layout=(2, 2), size=(1050, 850),
                      plot_title=title; kwargs...)
end


# -----------------------------------------------------------------------------
# Section 3: Timeseries & Regression Primitives
# -----------------------------------------------------------------------------

"""
    timeseries_ci(x, mean, lower, upper; color=:royalblue, title="", xlabel="", ylabel="",
      theme_kwargs=NamedTuple(), kwargs...)

Plots a timeseries or 1D effect with a shaded credible/confidence interval ribbon.
If `x` is empty or nothing, defaults to integer indices `1:length(mean)`.
"""
function timeseries_ci(
    x, mean, lower, upper;
    color=:royalblue, title::String="", xlabel::String="", ylabel::String="",
    theme_kwargs=NamedTuple(), kwargs...
)
    meanv = collect(Float64, mean)
    lowv = collect(Float64, lower)
    upv = collect(Float64, upper)

    xvals = (isnothing(x) || isempty(x)) ? collect(1:length(meanv)) : x

    return Plots.plot(
        xvals, meanv, ribbon=(meanv .- lowv, upv .- meanv),
        color=color, lw=2, fillalpha=0.2, legend=false,
        title=title, xlabel=xlabel, ylabel=ylabel;
        kwargs..., theme_kwargs...
    )
end


# -----------------------------------------------------------------------------
# Section 4: Export Utilities
# -----------------------------------------------------------------------------

"""
    save_plot(p::Union{Plots.Plot, LeafletMap}, path::AbstractString; fmt=nothing, dpi::Integer=150)

Saves plot `p` (either a `Plots.Plot` figure or an interactive `LeafletMap` HTML document)
to file path `path`. Creates parent directories automatically if needed.
"""
function save_plot(p::Union{Plots.Plot, LeafletMap}, path::AbstractString; fmt=nothing, dpi::Integer=150)
    if p isa LeafletMap
        return save_html(p, path)
    end
    dir = dirname(path)
    if !isempty(dir)
        try; isdir(dir) || mkpath(dir); catch; end
    end
    out_path = path
    if !isnothing(fmt)
        ext = "." * lowercase(string(lstrip(string(fmt), '.')))
        if !endswith(lowercase(out_path), ext)
            out_path = string(out_path, ext)
        end
    end
    Plots.savefig(p, out_path)
    return out_path
end


# -----------------------------------------------------------------------------
# Section 5: High-Level Model Diagnostics & Reconstruction Visualization
# -----------------------------------------------------------------------------

"""
    bstm_plots(model_obj::DynamicPPL.Model, chain, res, M; au=nothing, data=nothing, outcome=1)

Primary diagnostic and effect visualization engine for the `bstm` framework.
Takes model results and generates standard diagnostic plots:
- Posterior Predictive Check (PPC)
- Fixed Effects coefficients
- Conditional / marginal covariate effects
- Component-wise spatial, temporal, seasonal, smooth, and mixed effects
- Spatiotemporal interaction grids
"""
function _bstm_plots_impl(model_obj, chain, res, M; au=nothing, data=nothing, outcome=1)
    plots = Dict{Symbol, Any}()
    plots_data = Dict{Symbol, Any}()
    
    effects = get(res, :effects, nothing)
    arch_obj = get(res, :arch, UnivariateArchitecture())
    is_mv = arch_obj isa MultivariateArchitecture
    pred_denoised = hasproperty(res, :predictions) && hasproperty(res.predictions,
        :denoised) ? res.predictions.denoised : nothing
    
    input_data = !isnothing(data) ? data : get(M, :data, nothing)
    y_obs = if hasproperty(res, :predictions) && hasproperty(res.predictions,
        :observed) && !isnothing(res.predictions.observed)
        res.predictions.observed
    elseif haskey(M, :y_obs) && !isnothing(M.y_obs)
        M.y_obs
    elseif !isnothing(input_data)
        if haskey(M, :outcomes) && !isempty(M.outcomes) && hasproperty(input_data,
            Symbol(M.outcomes[1]))
            Matrix(input_data[!, Symbol.(M.outcomes)])
        elseif hasproperty(input_data, :y)
            input_data.y
        elseif hasproperty(input_data, :y_rate)
            input_data.y_rate
        else
            nothing
        end
    else
        nothing
    end

    function _extract_outcome_vec(y_raw, k_idx)
        if isnothing(y_raw)
            return nothing
        elseif y_raw isa AbstractVector{<:AbstractVector}
            return (k_idx <= length(y_raw)) ? vec(collect(Float64, y_raw[k_idx])) : nothing
        elseif y_raw isa AbstractMatrix
            return (k_idx <= size(y_raw, 2)) ? vec(collect(Float64, y_raw[:, k_idx])) : nothing
        elseif y_raw isa AbstractVector
            return vec(collect(Float64, y_raw))
        else
            return try vec(collect(Float64, y_raw)) catch; nothing end
        end
    end

    polygons = if !isnothing(au) && (au isa NamedTuple || au isa Dict)
        get(au, :polygons, nothing)
    else
        nothing
    end
    
    centroids = if !isnothing(au) && (au isa NamedTuple || au isa Dict)
        get(au, :centroids, nothing)
    else
        nothing
    end

    # --- 1. Posterior Predictive Check ---
    if !isnothing(pred_denoised) && !isnothing(y_obs)
        pred_summary = is_mv ? (outcome <= length(pred_denoised) ? pred_denoised[outcome] : nothing) : pred_denoised
        y_o = _extract_outcome_vec(y_obs, outcome)
        if !isnothing(pred_summary) && hasproperty(pred_summary, :mean) && !isnothing(y_o)
            y_p = vec(collect(Float64, pred_summary.mean))
            if length(y_p) == length(y_o)
                clean_mask = .!isnan.(y_p) .& .!isnan.(y_o)
                clean_p, clean_o = y_p[clean_mask], y_o[clean_mask]
                
                if !isempty(clean_p) && !isempty(clean_o)
                    min_val = min(minimum(clean_p), minimum(clean_o))
                    max_val = max(maximum(clean_p), maximum(clean_o))
                    pad = max((max_val - min_val) * 0.05, 1e-6)
                    lims = (min_val - pad, max_val + pad)

                    # Plot with Credible Intervals if available
                    has_intervals = hasproperty(pred_summary,
                        :lower) && hasproperty(pred_summary, :upper)
                    
                    p_ppc = if has_intervals
                        y_low = vec(collect(Float64, pred_summary.lower))[clean_mask]
                        y_high = vec(collect(Float64, pred_summary.upper))[clean_mask]
                        err_low = max.(clean_p .- y_low, 0.0)
                        err_high = max.(y_high .- clean_p, 0.0)
                        Plots.scatter(
                            clean_o, clean_p, yerror=(err_low, err_high),
                            title="Posterior Predictive Check (PPC)",
                            xlabel="Observed Value (y_obs)",
                            ylabel="Predicted Expected Value (ŷ)",
                            alpha=0.7, markersize=3.5, color=:dodgerblue,
                            markerstrokecolor=:dodgerblue, markerstrokewidth=0.5,
                            xlims=lims, ylims=lims,
                            legend=:topleft, label="Predictions (95% CI)"
                        )
                    else
                        Plots.scatter(
                            clean_o, clean_p,
                            title="Posterior Predictive Check (PPC)",
                            xlabel="Observed Value (y_obs)",
                            ylabel="Predicted Expected Value (ŷ)",
                            alpha=0.7, markersize=3.5, color=:dodgerblue,
                            markerstrokewidth=0,
                            xlims=lims, ylims=lims,
                            legend=:topleft, label="Predictions"
                        )
                    end
                    
                    Plots.plot!(p_ppc, [min_val - pad, max_val + pad], [min_val - pad,
                        max_val + pad],
                                color=:crimson, ls=:dash, lw=2, label="Identity (1:1)")
                    
                    plots[:posterior_predictive_check] = p_ppc
                    plots_data[:posterior_predictive_check] = (observed = y_o, predicted = y_p)
                end
            end
        end
    end

    # --- 2. Helper function for spatial maps (handles both unit-level and observation-level effects) ---
    function _render_spatial_effect(field_data, title_str, polys, cents; kwargs...)
        if isnothing(field_data) || !hasproperty(field_data, :mean)
            return nothing
        end 
        if isnothing(polys) && isnothing(cents)
            return nothing
        end
        s_mean = vec(collect(Float64, field_data.mean))
        if isempty(s_mean) || all(iszero, s_mean)
            return nothing
        end

        n_units = !isnothing(polys) ? length(polys) : length(cents)
        unit_vals = if length(s_mean) == n_units
            s_mean
        elseif length(s_mean) > n_units && haskey(M, :s_idx)
            s_idx_vec = M.s_idx
            [mean(s_mean[s_idx_vec .== i]) for i in 1:n_units]
        elseif length(s_mean) > n_units && !isnothing(input_data) && (hasproperty(input_data,
            :s_idx) || hasproperty(input_data, :district))
            s_idx_vec = hasproperty(input_data, :s_idx) ? input_data.s_idx : input_data.district
            [mean(s_mean[s_idx_vec .== i]) for i in 1:n_units]
        else
            s_mean[1:min(length(s_mean), n_units)]
        end

        if !isnothing(polys) && length(polys) == length(unit_vals)
            return choropleth(polys, unit_vals; title=title_str, kwargs...)
        elseif !isnothing(cents)
            xs, ys = _centroids_xy(cents)
            return Plots.scatter(xs, ys, marker_z=unit_vals, markersize=5, c=:viridis,
                label=nothing, title=title_str, aspect_ratio=:equal, kwargs...)
        end
        return nothing
    end

    if (!isnothing(polygons) || !isnothing(centroids)) && !isnothing(y_obs)
        y_o = _extract_outcome_vec(y_obs, outcome)
        n_units = !isnothing(polygons) ? length(polygons) : length(centroids)
        s_idx_vec = haskey(M,
            :s_idx) ? M.s_idx : (!isnothing(input_data) && hasproperty(input_data,
            :s_idx) ? input_data.s_idx : (!isnothing(input_data) && hasproperty(input_data,
            :district) ? input_data.district : (!isnothing(y_o) ? (1:length(y_o)) : 1:n_units)))

        if !isnothing(y_o) && length(y_o) >= n_units
            family_str = haskey(M, :likelihood_specs) && length(M.likelihood_specs) >= outcome ? string(get(M.likelihood_specs[outcome], :family, "gaussian")) : "gaussian"
            has_count_offset = family_str in ["poisson", "negbin"] && haskey(M,
                :log_offsets) && !isnothing(M.log_offsets) && !all(iszero, M.log_offsets)
            log_off_vec = if has_count_offset
                vec(M.log_offsets isa AbstractVector ? M.log_offsets : M.log_offsets[:, outcome])
            else
                nothing
            end

            y_unit_obs = if !isnothing(log_off_vec)
                [sum(y_o[s_idx_vec .== i]) / max(sum(exp.(log_off_vec[s_idx_vec .== i])), 1e-12) for i in 1:n_units]
            else
                [mean(y_o[s_idx_vec .== i]) for i in 1:n_units]
            end
            obs_title = !isnothing(log_off_vec) ? "Observed Standardized Rate (SIR)" : "Observed Mean by Spatial Unit"
            p_obs_map = _render_spatial_effect((mean=y_unit_obs,), obs_title, polygons, centroids)
            if !isnothing(p_obs_map)
                plots[:spatial_observed] = p_obs_map
                plots_data[:spatial_observed] = (values=y_unit_obs,
                    geometry=isnothing(polygons) ? centroids : polygons)
            end

            if !isnothing(pred_denoised)
                pred_summary = is_mv ? pred_denoised[outcome] : pred_denoised
                if hasproperty(pred_summary, :mean)
                    y_p = vec(pred_summary.mean)
                    y_unit_fit = if !isnothing(log_off_vec)
                        [sum(y_p[s_idx_vec .== i]) / max(sum(exp.(log_off_vec[s_idx_vec .== i])), 1e-12) for i in 1:n_units]
                    else
                        [mean(y_p[s_idx_vec .== i]) for i in 1:n_units]
                    end
                    fit_title = !isnothing(log_off_vec) ? "Fitted Relative Risk (RR)" : "Fitted Mean by Spatial Unit"
                    p_fit_map = _render_spatial_effect((mean=y_unit_fit, ), fit_title,
                        polygons, centroids)
                    if !isnothing(p_fit_map)
                        plots[:spatial_fitted] = p_fit_map
                        plots_data[:spatial_fitted] = (values=y_unit_fit,
                            geometry=isnothing(polygons) ? centroids : polygons)
                    end

                    y_unit_res = y_unit_obs .- y_unit_fit
                    p_res_map = _render_spatial_effect((mean=y_unit_res, ),
                        "Spatial Residuals (Observed - Fitted)", polygons, centroids)
                    if !isnothing(p_res_map)
                        plots[:spatial_residuals] = p_res_map
                        plots_data[:spatial_residuals] = (values=y_unit_res,
                            geometry=isnothing(polygons) ? centroids : polygons)
                    end
                end
            end
        end
    end

    # --- 3. Fixed Effects Plots ---
    if !isnothing(effects) && hasproperty(effects, :fixed) && !isnothing(effects.fixed)
        fe_summary = is_mv ? effects.fixed[outcome] : effects.fixed
        if hasproperty(fe_summary, :mean) && !all(iszero, fe_summary.mean) 
            fm, fl, fu = vec(fe_summary.mean), vec(fe_summary.lower), vec(fe_summary.upper)
            if !isempty(fm)
                coef_names = haskey(M,
                    :Xfixed_names) ? string.(M.Xfixed_names) : ["Beta_$i" for i in 1:length(fm)]
                p_forest = Plots.scatter(fm, 1:length(fm), xerror=(fm .- fl, fu .- fm),
                    yticks=(1:length(fm), coef_names), title="Fixed Effects Coefficients",
                    xlabel="Estimate (95% CI)", markersize=5, color=:royalblue, legend=false)
                Plots.vline!(p_forest, [0], color=:crimson, ls=:dash, lw=1.5)
                plots[:fixed_effects] = p_forest
                plots_data[:fixed_effects] = (names=coef_names, mean=fm, lower=fl, upper=fu)
            end
        end
    end

    # --- 4. Conditional Effects Plots ---
    conditional_plots = Dict{Symbol, Any}()
    conditional_plots_data = Dict{Symbol, Any}()
    all_covariates = String[]
    if haskey(M, :Xfixed_names)
        append!(all_covariates, string.(M.Xfixed_names))
    end
    if haskey(M, :components)
        for spec in M.components
            if spec.structure == :smooth
                vars = get(spec.params, :positional_args, [])
                append!(all_covariates, string.(vars))
            end
        end
    end
    all_covariates = unique(Symbol.(all_covariates))

    if haskey(M, :data) && !isnothing(M.data)
        for cov_sym in all_covariates
            if !hasproperty(M.data, cov_sym)
                continue
            end

            if eltype(M.data[!, cov_sym]) <: Number # Continuous covariate
                cond_preds_res = _generate_conditional_predictions(model_obj, chain, M, cov_sym)
                if !isnothing(cond_preds_res)
                    cond_preds, cov_range = cond_preds_res
                    cond_summary = is_mv ? cond_preds[outcome] : cond_preds
                    
                    cm, cl, cu = vec(cond_summary.mean), vec(cond_summary.lower), vec(cond_summary.upper)
                    p_cond = timeseries_ci(cov_range, cm, cl, cu; color=:blue,
                        title="Conditional Effect: $(cov_sym)", xlabel=string(cov_sym),
                        ylabel="Expected Response")
                    plot_key = Symbol("conditional_$(cov_sym)")
                    conditional_plots[plot_key] = p_cond
                    conditional_plots_data[plot_key] = (covariate_values=cov_range, mean=cm,
                        lower=cl, upper=cu)
                end
            else # Categorical covariate
                cond_preds_res = _generate_conditional_predictions(model_obj, chain, M, cov_sym)
                if !isnothing(cond_preds_res)
                    cond_preds, cov_levels = cond_preds_res
                    cond_summary = is_mv ? cond_preds[outcome] : cond_preds
                    
                    cm, cl, cu = vec(cond_summary.mean), vec(cond_summary.lower), vec(cond_summary.upper)
                    p_cond = Plots.bar(string.(cov_levels), cm, yerror=(cm .- cl, cu .- cm),
                        title="Conditional Effect: $(cov_sym)", xlabel=string(cov_sym),
                        ylabel="Expected Response", color=:blue, legend=false)
                    plot_key = Symbol("conditional_$(cov_sym)")
                    conditional_plots[plot_key] = p_cond
                    conditional_plots_data[plot_key] = (covariate_levels=string.(cov_levels), mean=cm, lower=cl, upper=cu)
                end
            end
        end
    end
    if !isempty(conditional_plots)
        plots[:conditional_effects] = conditional_plots
        plots_data[:conditional_effects] = conditional_plots_data
    end

    # --- 5. Component-wise Plotting ---
    smooth_effects_plots = Dict{Symbol, Any}()
    smooth_effects_plots_data = Dict{Symbol, Any}()
    if !isnothing(effects)
        comp_keys = haskey(M, :components) ? [s.key for s in M.components] : [k for k in keys(effects) if !(k in [:fixed, :intercept, :mixed_effects, :st_interaction])]
        for key in comp_keys
            spec_idx = haskey(M, :components) ? findfirst(s -> s.key == key,
                M.components) : nothing
            spec = !isnothing(spec_idx) ? M.components[spec_idx] : nothing
            if !isnothing(spec) && spec.component_obj isa Mixed
                continue
            end
            if !haskey(effects, key)
                continue
            end
            
            component_effects = effects[key]
            main_effect_summary = if hasproperty(component_effects, :noisy)
                is_mv ? component_effects.noisy[outcome] : component_effects.noisy
            elseif hasproperty(component_effects, :structured)
                is_mv ? component_effects.structured[outcome] : component_effects.structured
            else
                continue
            end

            if isnothing(main_effect_summary) || !hasproperty(main_effect_summary,
                :mean) || isempty(main_effect_summary.mean)
                continue
            end

            struct_type = if !isnothing(spec)
                spec.structure
            elseif key in [:s_idx, :spatial, :district, :region,
                :space] || hasproperty(component_effects, :unstructured)
                :spatial
            elseif key in [:year, :time, :t_idx, :temporal, :t, :day, :month, :date]
                :temporal
            elseif key in [:season, :u_idx, :seasonal]
                :seasonal
            else
                :smooth
            end

            if struct_type == :spatial
                plot_key = Symbol("spatial_$(key)")
                p = _render_spatial_effect(main_effect_summary, "Spatial Effect: $key",
                    polygons, centroids)
                if !isnothing(p)
                    plots[plot_key] = p
                    plots[:spatial] = p
                    plots_data[plot_key] = (values=vec(main_effect_summary.mean), geometry=isnothing(polygons) ? centroids : polygons)
                    plots_data[:spatial] = plots_data[plot_key]
                end

                if hasproperty(component_effects, :structured)
                    struct_summary = is_mv ? component_effects.structured[outcome] : component_effects.structured
                    p_struct = _render_spatial_effect(struct_summary,
                        "Structured Spatial Effect: $key", polygons, centroids)
                    if !isnothing(p_struct)
                        plot_key_struct = Symbol("structured_$(key)")
                        plots[plot_key_struct] = p_struct
                        plots_data[plot_key_struct] = (values=vec(struct_summary.mean), geometry=isnothing(polygons) ? centroids : polygons)
                    end
                end
                if hasproperty(component_effects, :unstructured)
                    unstruct_summary = is_mv ? component_effects.unstructured[outcome] : component_effects.unstructured
                    p_unstruct = _render_spatial_effect(unstruct_summary,
                        "Unstructured Spatial Effect: $key", polygons, centroids)
                    if !isnothing(p_unstruct)
                        plot_key_unstruct = Symbol("unstructured_$(key)")
                        plots[plot_key_unstruct] = p_unstruct
                        plots_data[plot_key_unstruct] = (values=vec(unstruct_summary.mean), geometry=isnothing(polygons) ? centroids : polygons)
                    end
                end

            elseif struct_type == :temporal
                plot_key = Symbol("temporal_$(key)")
                tm, tl, tu = vec(main_effect_summary.mean), vec(main_effect_summary.lower), vec(main_effect_summary.upper)
                
                t_info = _resolve_temporal_coordinates(M, input_data, length(tm), key)
                raw_t = t_info.raw_t
                unique_t = t_info.unique_t
                t_label = t_info.label

                if length(unique_t) < length(tm) && length(unique_t) > 1
                    t_mean = [mean(tm[raw_t .== t]) for t in unique_t]
                    t_low = [mean(tl[raw_t .== t]) for t in unique_t]
                    t_up = [mean(tu[raw_t .== t]) for t in unique_t]
                    p_temp = timeseries_ci(unique_t, t_mean, t_low, t_up; color=:royalblue,
                        title="Temporal Trend: $key", xlabel=t_label)
                    plots[plot_key] = p_temp
                    plots[:temporal] = p_temp
                    plots_data[plot_key] = (time=unique_t, mean=t_mean, lower=t_low, upper=t_up)
                    plots_data[:temporal] = plots_data[plot_key]
                else
                    p_order = sortperm(raw_t)
                    p_temp = timeseries_ci(raw_t[p_order], tm[p_order], tl[p_order],
                        tu[p_order]; color=:royalblue, title="Temporal Trend: $key",
                        xlabel=t_label)
                    plots[plot_key] = p_temp
                    plots[:temporal] = p_temp
                    plots_data[plot_key] = (time=raw_t[p_order], mean=tm[p_order],
                        lower=tl[p_order], upper=tu[p_order])
                    plots_data[:temporal] = plots_data[plot_key]
                end

            elseif struct_type == :seasonal
                plot_key = Symbol("seasonal_$(key)")
                um, ul, uu = vec(main_effect_summary.mean), vec(main_effect_summary.lower), vec(main_effect_summary.upper)
                u_info = _resolve_temporal_coordinates(M, input_data, length(um), key)
                raw_u = u_info.raw_t
                unique_u = u_info.unique_t
                u_label = "Period"

                if length(unique_u) < length(um) && length(unique_u) > 1
                    u_mean = [mean(um[raw_u .== u]) for u in unique_u]
                    u_low = [mean(ul[raw_u .== u]) for u in unique_u]
                    u_up = [mean(uu[raw_u .== u]) for u in unique_u]
                    p_seas = timeseries_ci(unique_u, u_mean, u_low, u_up; color=:forestgreen,
                        title="Seasonal Component: $key", xlabel=u_label)
                    plots[plot_key] = p_seas
                    plots[:seasonal] = p_seas
                    plots_data[plot_key] = (period=unique_u, mean=u_mean, lower=u_low, upper=u_up)
                    plots_data[:seasonal] = plots_data[plot_key]
                else
                    p_order = sortperm(raw_u)
                    p_seas = timeseries_ci(raw_u[p_order], um[p_order], ul[p_order],
                        uu[p_order]; color=:forestgreen, title="Seasonal Component: $key",
                        xlabel=u_label)
                    plots[plot_key] = p_seas
                    plots[:seasonal] = p_seas
                    plots_data[plot_key] = (period=raw_u[p_order], mean=um[p_order],
                        lower=ul[p_order], upper=uu[p_order])
                    plots_data[:seasonal] = plots_data[plot_key]
                end

            elseif struct_type == :smooth
                vars = !isnothing(spec) ? get(spec.params, :positional_args,
                    [string(key)]) : [string(key)]
                cov_df = !isnothing(input_data)&&
                    input_data isa DataFrame ? input_data : (haskey(M, :data)&&
                    !isnothing(M.data) && M.data isa DataFrame ? M.data : nothing)

                if length(vars) == 1 # 1D smooth
                    var_sym = Symbol(vars[1])
                    if !isnothing(cov_df) && hasproperty(cov_df, var_sym)
                        cov_data = cov_df[!, var_sym]
                        sm, sl, su = vec(main_effect_summary.mean), vec(main_effect_summary.lower), vec(main_effect_summary.upper)
                        
                        if length(cov_data) == length(sm)
                            unique_cov = sort(unique(cov_data))
                            
                            # If there are duplicate covariate values (e.g. repeated
                            #   space-time observations)
                            if length(unique_cov) < length(sm) && length(unique_cov) > 1
                                sm_mean = [mean(sm[cov_data .== x]) for x in unique_cov]
                                sm_low  = [mean(sl[cov_data .== x]) for x in unique_cov]
                                sm_up   = [mean(su[cov_data .== x]) for x in unique_cov]
                                p_sm = timeseries_ci(unique_cov, sm_mean, sm_low, sm_up;
                                    color=:darkorange, title="Smooth Effect: $var_sym",
                                    xlabel=string(var_sym))
                                smooth_effects_plots[var_sym] = p_sm
                                smooth_effects_plots_data[var_sym] = (covariate_values=unique_cov,
                                    mean=sm_mean, lower=sm_low, upper=sm_up)
                            else
                                p_order = sortperm(cov_data)
                                p_sm = timeseries_ci(cov_data[p_order], sm[p_order],
                                    sl[p_order], su[p_order]; color=:darkorange,
                                    title="Smooth Effect: $var_sym", xlabel=string(var_sym))
                                smooth_effects_plots[var_sym] = p_sm
                                smooth_effects_plots_data[var_sym] = (covariate_values=cov_data[p_order], mean=sm[p_order], lower=sl[p_order], upper=su[p_order])
                            end
                            
                            plots[Symbol("smooth_$(var_sym)")] = p_sm
                            if !haskey(plots, :smooth)
                                plots[:smooth] = p_sm
                            end
                        end
                    else
                        sm, sl, su = vec(main_effect_summary.mean), vec(main_effect_summary.lower), vec(main_effect_summary.upper)
                        p_order = sortperm(sm)
                        p_sm = timeseries_ci(1:length(sm), sm[p_order], sl[p_order], su[p_order]; color=:darkorange, title="Smooth Effect: $var_sym", xlabel="Covariate Index")
                        smooth_effects_plots[var_sym] = p_sm
                        smooth_effects_plots_data[var_sym] = (covariate_values=1:length(sm), mean=sm[p_order], lower=sl[p_order], upper=su[p_order])
                        plots[Symbol("smooth_$(var_sym)")] = p_sm
                        if !haskey(plots, :smooth)
                            plots[:smooth] = p_sm
                        end
                    end
                elseif length(vars) == 2 # 2D smooth (interaction)
                    var1_sym, var2_sym = Symbol(vars[1]), Symbol(vars[2])
                    if !isnothing(cov_df) && hasproperty(cov_df,
                        var1_sym) && hasproperty(cov_df, var2_sym)
                        cond_preds_res = _generate_conditional_predictions(model_obj, chain, M,
                            var1_sym, second_cov=var2_sym)
                        if !isnothing(cond_preds_res)
                            cond_preds, range1, range2 = cond_preds_res
                            cond_summary = is_mv ? cond_preds[outcome] : cond_preds
                            grid_mean = reshape(vec(cond_summary.mean), length(range1), length(range2))
                            
                            plot_key = Symbol("$(var1_sym)_$(var2_sym)")
                            p_2d = Plots.heatmap(range1, range2, grid_mean', title="2D Smooth Effect: $(var1_sym) & $(var2_sym)", xlabel=string(var1_sym), ylabel=string(var2_sym), c=:viridis, legend=false)
                            smooth_effects_plots[plot_key] = p_2d
                            smooth_effects_plots_data[plot_key] = (x=range1, y=range2, z=grid_mean)
                            plots[Symbol("smooth_$(plot_key)")] = p_2d
                        end
                    end
                end
            end
        end
    end
    if !isempty(smooth_effects_plots)
        plots[:smooth_effects] = smooth_effects_plots
        plots_data[:smooth_effects] = smooth_effects_plots_data
    end

    # --- 6. Spatiotemporal Interaction Effects ---
    if !isnothing(effects) && hasproperty(effects,
        :st_interaction) && !isnothing(effects.st_interaction)
        st_summary = is_mv ? effects.st_interaction[outcome] : effects.st_interaction
        if hasproperty(st_summary, :mean) && !all(iszero, st_summary.mean)
            if haskey(M, :s_N) && haskey(M, :t_N)
                st_mean_grid = reshape(vec(st_summary.mean), M.s_N, M.t_N)
                p_st_heatmap = Plots.heatmap(1:M.t_N, 1:M.s_N, st_mean_grid,
                    title="Spatiotemporal Interaction", xlabel="Time Index",
                    ylabel="Spatial Unit Index", c=:viridis, legend=false)
                plots[:st_interaction_heatmap] = p_st_heatmap
                plots_data[:st_interaction_heatmap] = (time_idx=1:M.t_N, space_idx=1:M.s_N,
                    mean_effect=st_mean_grid)
            end
        end
    end

    # --- 6b. Spatiotemporal Trajectory Curves & Spacetime Map Slice ---
    if !isnothing(pred_denoised)
        pred_summary = is_mv ? pred_denoised[outcome] : pred_denoised
        if hasproperty(pred_summary, :mean)
            y_p_all = vec(pred_summary.mean)
            N_all = length(y_p_all)

            s_vec_st = if haskey(M, :s_idx) && length(M.s_idx) == N_all
                M.s_idx
            elseif !isnothing(input_data) && hasproperty(input_data,
                :s_idx) && length(input_data.s_idx) == N_all
                input_data.s_idx
            elseif !isnothing(input_data) && hasproperty(input_data,
                :district) && length(input_data.district) == N_all
                input_data.district
            elseif haskey(M, :s_N) && haskey(M, :t_N) && N_all == M.s_N * M.t_N
                repeat(1:M.s_N, inner=M.t_N)
            else
                nothing
            end

            t_info_st = _resolve_temporal_coordinates(M, input_data, N_all, haskey(M,
                :t_idx_var) ? M.t_idx_var : nothing)
            t_vec_st = t_info_st.raw_t
            unique_t = t_info_st.unique_t
            t_label = t_info_st.label

            if !isnothing(s_vec_st) && length(unique_t) > 1
                s_vec = s_vec_st
                t_vec = t_vec_st
                unique_s = sort(unique(s_vec))

                if length(unique_s) > 1
                    # 1. Multi-unit Timeseries
                    p_temporal_curves = Plots.plot(title="Temporal Fitted Trajectories across Units", xlabel=t_label, ylabel="Fitted Value", legend=false)
                    for u in unique_s
                        mask_u = s_vec .== u
                        if any(mask_u)
                            t_u = t_vec[mask_u]
                            p_order = sortperm(t_u)
                            Plots.plot!(p_temporal_curves, t_u[p_order],
                                y_p_all[mask_u][p_order], color=:royalblue, alpha=0.35, lw=1.2)
                        end
                    end
                    plots[:temporal_trajectories] = p_temporal_curves
                    plots_data[:temporal_trajectories] = (spatial_units=unique_s,
                        time_points=unique_t)

                    if !haskey(plots, :temporal)
                        plots[:temporal] = p_temporal_curves
                        plots_data[:temporal] = (spatial_units=unique_s, time_points=unique_t)
                    end
                end

                # 2. Spacetime Predictions Maps across ALL time slices
                n_units = !isnothing(polygons) ? length(polygons) : (!isnothing(centroids) ? length(centroids) : length(unique_s))
                
                valid_y_p = filter(!isnan, y_p_all)
                clims_global = !isempty(valid_y_p) ? (minimum(valid_y_p), maximum(valid_y_p)) : nothing

                st_slice_plots = Plots.Plot[]
                st_slice_data = Dict{Any, Vector{Float64}}()

                for t_val in unique_t
                    mask_t = t_vec .== t_val
                    s_t = s_vec[mask_t]
                    y_p_t = y_p_all[mask_t]

                    unit_preds_t = zeros(Float64, n_units)
                    for i in 1:n_units
                        match_idx = findall(s_t .== i)
                        if !isempty(match_idx)
                            unit_preds_t[i] = mean(y_p_t[match_idx])
                        else
                            match_all = findall(s_vec .== i)
                            unit_preds_t[i] = !isempty(match_all) ? mean(y_p_all[match_all]) : 0.0
                        end
                    end
                    st_slice_data[t_val] = unit_preds_t

                    title_slice = "$t_label = $t_val"
                    p_slice = if !isnothing(clims_global)
                        _render_spatial_effect((mean=unit_preds_t, ), title_slice, polygons,
                            centroids; clims=clims_global)
                    else
                        _render_spatial_effect((mean=unit_preds_t, ), title_slice, polygons,
                            centroids)
                    end
                    if !isnothing(p_slice)
                        push!(st_slice_plots, p_slice)
                    end
                end

                if !isempty(st_slice_plots)
                    T = length(st_slice_plots)
                    if T == 1
                        plots[:spacetime_predictions] = st_slice_plots[1]
                    else
                        ncols = min(4, T)
                        nrows = ceil(Int, T / ncols)
                        p_all_st = Plots.plot(st_slice_plots..., layout=(nrows, ncols),
                            size=(360 * ncols, 320 * nrows),
                            plot_title="Spacetime Predictions Across Time Slices")
                        plots[:spacetime_predictions] = p_all_st
                        
                        slice_dict = Dict{Symbol, Any}()
                        for (idx, t_val) in enumerate(unique_t)
                            if idx <= length(st_slice_plots)
                                slice_dict[Symbol("t_$(t_val)")] = st_slice_plots[idx]
                            end
                        end
                        plots[:spacetime_slices] = NamedTuple(slice_dict)
                    end
                    plots_data[:spacetime_predictions] = (time_points=unique_t,
                        predictions_by_time=st_slice_data,
                        geometry=isnothing(polygons) ? centroids : polygons)
                end
            end
        end
    end

    # --- 7. Spatially Varying Coefficients (SVC) Plots ---
    if !isnothing(effects) && haskey(M, :components)
        for spec in M.components
            if spec.structure == :svc
                key = spec.key
                if !haskey(effects, key) || (isnothing(polygons) && isnothing(centroids))
                    continue
                end
                
                svc_effect_summary = is_mv ? effects[key].structured[outcome] : effects[key].structured
                if !isnothing(svc_effect_summary) && hasproperty(svc_effect_summary, :mean)
                    plot_key = Symbol("svc_$(key)")
                    p_svc = _render_spatial_effect(svc_effect_summary, "SVC Effect: $(key)",
                        polygons, centroids)
                    if !isnothing(p_svc)
                        plots[plot_key] = p_svc
                        plots_data[plot_key] = (values=vec(svc_effect_summary.mean), geometry=isnothing(polygons) ? centroids : polygons)
                    end
                end
            end
        end
    end

    # --- 8. Hierarchical Effects (Mixed Effects) Plots ---
    if !isnothing(effects) && hasproperty(effects,
        :mixed_effects) && !isnothing(effects.mixed_effects)
        mixed_plots = Dict{Symbol, Any}()
        mixed_plots_data = Dict{Symbol, Any}()
        for (key, effect_summary) in pairs(effects.mixed_effects)
            group_var = Symbol(effect_summary.group_var)
            group_levels = (haskey(M, :data) && !isnothing(M.data) && hasproperty(M.data,
                group_var)) ? unique(M.data[!, group_var]) : nothing

            summaries_to_plot = is_mv ? effect_summary.summaries[outcome] : effect_summary.summaries

            for (term_name, summary) in pairs(summaries_to_plot)
                if hasproperty(summary, :mean) && !all(iszero, summary.mean)
                    means = vec(summary.mean)
                    lowers = vec(summary.lower)
                    uppers = vec(summary.upper)
                    n_levels = length(means) 
                    
                    y_ticks_labels = isnothing(group_levels) || length(group_levels) != n_levels ? ["Level $i" for i in 1:n_levels] : string.(group_levels)
                    p_title = "Mixed Effect: $(term_name) | $(group_var)"
                    
                    p_forest = Plots.scatter(means, 1:n_levels, xerror=(means .- lowers,
                        uppers .- means), yticks=(1:n_levels, y_ticks_labels), title=p_title,
                        xlabel="Effect Size", markersize=4, color=:black, legend=false,
                        yflip=true)
                    Plots.vline!(p_forest, [0], color=:red, ls=:dash, lw=1)
                    
                    plot_key = Symbol("$(key)_$(term_name)")
                    mixed_plots[plot_key] = p_forest
                    mixed_plots_data[plot_key] = (group_levels=y_ticks_labels, mean=means,
                        lower=lowers, upper=uppers)
                end
            end
        end
        if !isempty(mixed_plots)
            plots[:mixed_effects] = mixed_plots
            plots_data[:mixed_effects] = mixed_plots_data
        end
    end

    # --- 9. Stratified Categorical Movement Parameter Plots ---
    if !isnothing(effects) && hasproperty(effects, :categorical_movement) && !isnothing(effects.categorical_movement)
        cat_mov_summary = effects.categorical_movement
        cat_plots = Dict{Symbol, Any}()
        cat_plots_data = Dict{Symbol, Any}()
        
        group_lookup = haskey(M, :group_lookup) ? M.group_lookup : (haskey(res, :group_lookup) ? res.group_lookup : nothing)
        group_names = !isnothing(group_lookup) ? sort(collect(keys(group_lookup)), by=x->group_lookup[x]) : nothing

        for param_sym in [:beta, :D_g, :gamma]
            if hasproperty(cat_mov_summary, param_sym)
                group_summaries = getproperty(cat_mov_summary, param_sym)
                if group_summaries isa AbstractVector && !isempty(group_summaries)
                    n_groups = length(group_summaries)
                    g_labels = !isnothing(group_names) && length(group_names) == n_groups ? group_names : ["Group $i" for i in 1:n_groups]
                    
                    means = [Float64(s.mean) for s in group_summaries]
                    lowers = [Float64(s.lower) for s in group_summaries]
                    uppers = [Float64(s.upper) for s in group_summaries]
                    
                    p_forest = Plots.scatter(means, 1:n_groups, xerror=(means .- lowers, uppers .- means),
                        yticks=(1:n_groups, g_labels), title="Categorical Movement Param: $(param_sym)",
                        xlabel="Estimate (95% CI)", markersize=5, color=:darkcyan, legend=false, yflip=true)
                    Plots.vline!(p_forest, [0.0], color=:crimson, ls=:dash, lw=1.5)
                    
                    cat_plots[param_sym] = p_forest
                    cat_plots_data[param_sym] = (groups=g_labels, mean=means, lower=lowers, upper=uppers)
                end
            end
        end
        if !isempty(cat_plots)
            plots[:categorical_movement] = cat_plots
            plots_data[:categorical_movement] = cat_plots_data
        end
    end

    return (plots=NamedTuple(plots), plots_data=NamedTuple(plots_data))
end

"""
    save_plots(plots_obj, save_dir::AbstractString; prefix::String="", fmt::String="png",
      dpi::Integer=150)

Saves all plots in a `plots` NamedTuple/Dict or `bstm_plots` result to `save_dir`.
Automatically saves `Plots.Plot` objects as `fmt` (default: `"png"`) and `LeafletMap` objects as `.html`.
"""
function save_plots(plots_obj, save_dir::AbstractString; prefix::String="", fmt::String="png",
    dpi::Integer=150)
    if !isdir(save_dir)
        mkpath(save_dir)
    end
    
    plots_to_save = hasproperty(plots_obj, :plots) ? plots_obj.plots : plots_obj
    saved_paths = String[]
    clean_fmt = lowercase(string(lstrip(fmt, '.')))

    for (k, p) in pairs(plots_to_save)
        if p isa LeafletMap
            fn = isempty(prefix) ? "$(k).html" : "$(prefix)_$(k).html"
            out_path = joinpath(save_dir, fn)
            save_html(p, out_path)
            push!(saved_paths, out_path)
        elseif p isa Plots.Plot
            fn = isempty(prefix) ? "$(k).$(clean_fmt)" : "$(prefix)_$(k).$(clean_fmt)"
            out_path = joinpath(save_dir, fn)
            save_plot(p, out_path; fmt=nothing, dpi=dpi)
            push!(saved_paths, out_path)
        elseif p isa NamedTuple || p isa Dict
            for (sub_k, sub_p) in pairs(p)
                if sub_p isa LeafletMap
                    fn = isempty(prefix) ? "$(k)_$(sub_k).html" : "$(prefix)_$(k)_$(sub_k).html"
                    out_path = joinpath(save_dir, fn)
                    save_html(sub_p, out_path)
                    push!(saved_paths, out_path)
                elseif sub_p isa Plots.Plot
                    fn = isempty(prefix) ? "$(k)_$(sub_k).$(clean_fmt)" : "$(prefix)_$(k)_$(sub_k).$(clean_fmt)"
                    out_path = joinpath(save_dir, fn)
                    save_plot(sub_p, out_path; fmt=nothing, dpi=dpi)
                    push!(saved_paths, out_path)
                end
            end
        end
    end
    @info "Saved $(length(saved_paths)) plot(s) to '$save_dir'."
    return saved_paths
end

"""
    bstm_plots(res::NamedTuple; au=nothing, data=nothing, outcome=1, save_dir=nothing,
      save_prefix="", fmt="png", dpi=150)
    bstm_plots(res::NamedTuple, data::Any; au=nothing, outcome=1, save_dir=nothing,
      save_prefix="", fmt="png", dpi=150)
    bstm_plots(model::DynamicPPL.Model, chain, res; au=nothing, data=nothing, outcome=1,
      save_dir=nothing, save_prefix="", fmt="png", dpi=150)
    bstm_plots(model::DynamicPPL.Model, chain; au=nothing, data=nothing, outcome=1,
      alpha=0.05, save_dir=nothing, save_prefix="", fmt="png", dpi=150)

Generates diagnostic, spatial, temporal, fixed, and conditional effect plots from model results.
Optionally saves all generated figures to `save_dir`.
"""
function bstm_plots(res::NamedTuple; au=nothing, data=nothing, outcome=1, save_dir=nothing,
    save_prefix="", fmt="png", dpi=150)
    model = get(res, :model, nothing)
    chain = get(res, :chain, nothing)
    M = !isnothing(model) && hasproperty(model, :args) && hasproperty(model.args,
        :M) ? model.args.M : NamedTuple()
    au_to_use = !isnothing(au) ? au : get(res, :au, nothing)
    data_to_use = !isnothing(data) ? data : (!isnothing(model) && hasproperty(model,
        :args) && hasproperty(model.args, :M) ? get(model.args.M, :data, nothing) : nothing)
    
    plot_res = _bstm_plots_impl(model, chain, res, M; au=au_to_use, data=data_to_use,
        outcome=outcome)
    
    if !isnothing(save_dir)
        save_plots(plot_res.plots, save_dir; prefix=save_prefix, fmt=fmt, dpi=dpi)
    end
    
    return plot_res
end

function bstm_plots(res::NamedTuple, data::Any; au=nothing, outcome=1, save_dir=nothing,
    save_prefix="", fmt="png", dpi=150)
    return bstm_plots(res; data=data, au=au, outcome=outcome, save_dir=save_dir,
        save_prefix=save_prefix, fmt=fmt, dpi=dpi)
end

function bstm_plots(model::DynamicPPL.Model, chain, res; au=nothing, data=nothing, outcome=1,
    save_dir=nothing, save_prefix="", fmt="png", dpi=150)
    M = model.args.M
    au_to_use = !isnothing(au) ? au : (hasproperty(res, :au) ? res.au : nothing)
    data_to_use = !isnothing(data) ? data : get(M, :data, nothing)
    
    plot_res = _bstm_plots_impl(model, chain, res, M; au=au_to_use, data=data_to_use,
        outcome=outcome)
    
    if !isnothing(save_dir)
        save_plots(plot_res.plots, save_dir; prefix=save_prefix, fmt=fmt, dpi=dpi)
    end
    
    return plot_res
end

function bstm_plots(model::DynamicPPL.Model, chain; au=nothing, data=nothing, outcome=1,
    alpha=0.05, save_dir=nothing, save_prefix="", fmt="png", dpi=150)
    res = model_results_comprehensive(model, chain; data=data, alpha=alpha)
    return bstm_plots(model, chain, res; au=au, data=data, outcome=outcome, save_dir=save_dir,
        save_prefix=save_prefix, fmt=fmt, dpi=dpi)
end

"""
    model_results_plots(res)

Convenience utility to display all plots generated by `bstm_plots`.
"""
function model_results_plots(res)
    plots_obj = hasproperty(res, :plots) ? res.plots : res
    if isempty(plots_obj)
        println("No plots found in the results object.") 
        return
    end

    println("--- Displaying Generated Plots ---")
    for (plot_name, plot_obj) in pairs(plots_obj)
        if plot_obj isa Dict || plot_obj isa NamedTuple
            for (sub_name, sub_plot) in pairs(plot_obj)
                println("--- Plot: $plot_name -> $sub_name ---")
                display(sub_plot)
            end
        else
            println("--- Plot: $plot_name ---")
            display(plot_obj)
        end
    end
    println("--- End of Plots ---")
end
