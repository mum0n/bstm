"""
    sc_tags_data.jl

Snow Crab (*Chionoecetes opilio*) telemetry and mark-recapture data ingestion,
monthly/weekly temporal aggregation, spatial dead-tag filtering, habitat suitability
resharding onto 5 km hexagonal grids, and BSTM movement estimation.

Replicates and extends the historical R-based workflow with native, high-performance Julia.
"""

using bstm
using DataFrames
using Dates
using Plots
using DuckDB
using JLD2
using Random
using CSV
using SparseArrays
using Graphs
using NearestNeighbors
using Statistics
using LinearAlgebra
import bstm: _parse_flexible_date, _date_to_decimal_year

include(joinpath(@__DIR__, "movement_simple.jl"))
include(joinpath(@__DIR__, "sppoly_to_jld2.jl"))
using .SppolyConverter

function _to_decimal_year(d::Union{Date, DateTime})::Float64
    yr = Dates.year(d)
    doy = Dates.dayofyear(d)
    days_in_yr = Dates.isleapyear(yr) ? 366.0 : 365.0
    return Float64(yr) + (Float64(doy) - 1.0) / days_in_yr
end


"""
    aggregate_telemetry_time(
        tagging::DataFrame;
        time_interval::Symbol = :monthly
    )::DataFrame

Aggregates high-frequency telemetry observation pings per individual animal to regular
discrete time intervals (`:monthly`, `:weekly`, `:biweekly`, `:daily`, or `:raw`).

# Process
1. Maps each observation's timestamp to its temporal bucket (e.g. year-month for `:monthly`).
2. Computes centroid coordinates, mean depth, and preserves earliest observation metadata.
3. Retains only trajectories with at least 2 distinct temporal observations, re-indexing `:tag` (0..N).

# Arguments
- `tagging::DataFrame`: Standardized telemetry dataframe containing `:tagid`, `:lon`, `:lat`, `:time`, `:timestamp`, and `:tag`.
- `time_interval::Symbol`: Target aggregation window (`:monthly`, `:weekly`, `:biweekly`, `:daily`, or `:raw`).

# Returns
- `DataFrame`: Aggregated trajectory table with mean centroids and updated decimal year `:time`.
"""
function aggregate_telemetry_time(
    tagging::DataFrame;
    time_interval::Symbol = :monthly
)::DataFrame
    if time_interval == :raw
        return copy(tagging)
    end

    df = copy(tagging)
    if time_interval == :daily
        df[!, :time_bucket] = [Date(t) for t in df.timestamp]
    elseif time_interval == :weekly
        df[!, :time_bucket] = [
            Date(year(t), month(t), day(t)) - Day(dayofweek(t) - 1)
            for t in df.timestamp
        ]
    elseif time_interval == :biweekly
        df[!, :time_bucket] = [
            Date(1990, 1, 1) + Day(floor(Int, (Date(t) - Date(1990, 1, 1)).value / 14) * 14)
            for t in df.timestamp
        ]
    elseif time_interval == :monthly
        df[!, :time_bucket] = [Date(year(t), month(t), 1) for t in df.timestamp]
    else
        error("Unsupported time_interval: $(time_interval). Choose from :monthly, :weekly, :biweekly, :daily, :raw.")
    end

    # Fast single-pass aggregation
    agg = DataFrames.combine(
        groupby(df, [:tagid, :time_bucket]),
        :lon => (x -> mean(filter(isfinite, x))) => :lon,
        :lat => (x -> mean(filter(isfinite, x))) => :lat,
        :z => (x -> let v = filter(isfinite, x); isempty(v) ? NaN : mean(v); end) => :z,
        :cw => (x -> let v = filter(isfinite, x); isempty(v) ? NaN : mean(v); end) => :cw,
        :chela => (x -> let v = filter(isfinite, x); isempty(v) ? NaN : mean(v); end) => :chela,
        :wgt => (x -> let v = filter(isfinite, x); isempty(v) ? NaN : mean(v); end) => :wgt,
        :tag => minimum => :tag,
        :timestamp => minimum => :timestamp,
        :datasource => first => :datasource,
        :is_dead => (x -> any(x)) => :is_dead
    )
    agg[!, :time] = [_to_decimal_year(d) for d in agg.timestamp]
    sort!(agg, [:tagid, :time])

    # Re-index sequential tag (0..N) and keep trajectories with >= 2 observations
    valid_tags = String[]
    for sub in groupby(agg, :tagid)
        if nrow(sub) >= 2
            push!(valid_tags, first(sub.tagid))
        end
    end
    filter!(r -> r.tagid in Set(valid_tags), agg)

    final_dfs = DataFrame[]
    for sub in groupby(agg, :tagid)
        sdf = DataFrame(sub)
        sdf.tag = collect(0:(nrow(sdf) - 1))
        push!(final_dfs, sdf)
    end
    return isempty(final_dfs) ? DataFrame() : vcat(final_dfs...)
end

"""
    load_snowcrab_spatial_units(;
        data_dir::AbstractString = joinpath(@__DIR__, "data"),
        sppoly_file::Union{Nothing, AbstractString} = nothing
    )::NamedTuple

Loads the spatial areal units `au`, adjacency graph `W`, and metadata from `sppoly.jld2`.

# Returns
- `NamedTuple`:
  - `au`: Standard BSTM areal units NamedTuple (707 units, centroids, polygons, graph, W).
  - `sppoly`: Rectangular DataFrame of polygon attributes.
  - `W`: Sparse adjacency matrix.
  - `graph`: SimpleGraph.
  - `centroids_planar`: Vector of `(x, y)` in UTM Zone 20N (km).
  - `centroids_geo`: Vector of `(lon, lat)` in WGS84 degrees.
  - `metadata`: Dictionary of spatial parameters.
"""
function load_snowcrab_spatial_units(;
    data_dir::AbstractString = joinpath(@__DIR__, "data"),
    sppoly_file::Union{Nothing, AbstractString} = nothing
)::NamedTuple
    path = !isnothing(sppoly_file) ? sppoly_file : joinpath(data_dir, "sppoly.jld2")
    if !isfile(path)
        error("Spatial units file not found at: $(path). Run sppoly_to_jld2() to generate it.")
    end
    bundle = JLD2.load(path)
    return (
        au               = bundle["au"],
        sppoly           = bundle["sppoly"],
        W                = bundle["W"],
        graph            = bundle["graph"],
        centroids_planar = bundle["centroids_planar"],
        centroids_geo    = bundle["centroids_geo"],
        metadata         = bundle["metadata"]
    )
end

"""
    load_snowcrab_hsi(;
        data_dir::AbstractString = joinpath(@__DIR__, "data"),
        hsi_file::Union{Nothing, AbstractString} = nothing,
        ref_doy::Real = 244.0
    )::NamedTuple

Loads the 3D habitat suitability simulation array (`sims`) from `hsi.jld2`, computes
posterior summaries, and discretizes the HSI field onto a monthly basis (12 months per year).

# Returns
- `NamedTuple`:
  - `sims`: 3D Array of shape `(707, 27, 5000)`.
  - `years`: Vector of annual time slices (`1999:2025`).
  - `auids`: Vector of spatial unit IDs (`1:707`).
  - `hsi_mean`: Matrix `(707 x 27)` of posterior means across draws.
  - `hsi_sd`: Matrix `(707 x 27)` of posterior standard deviations.
  - `hsi_spatial_mean`: Vector of length 707 of multi-year mean habitat suitability.
  - `monthly_hsi`: Matrix `(707 x 324)` of monthly discretized habitat suitability.
  - `month_lookup`: Map from `(year, month)` to column index in `monthly_hsi`.
"""
function load_snowcrab_hsi(;
    data_dir::AbstractString = joinpath(@__DIR__, "data"),
    hsi_file::Union{Nothing, AbstractString} = nothing,
    ref_doy::Real = 244.0
)::NamedTuple
    path = !isnothing(hsi_file) ? hsi_file : joinpath(data_dir, "hsi.jld2")
    if !isfile(path)
        error("HSI file not found at: $(path). Run sppoly_to_jld2() to generate it.")
    end
    bundle = JLD2.load(path)
    sims = bundle["sims"]
    years = bundle["years"]
    auids = bundle["auids"]

    # Posterior summaries across draws
    hsi_mean = dropdims(mean(sims, dims=3), dims=3) # 707 x 27
    hsi_sd = dropdims(std(sims, dims=3), dims=3)
    hsi_spatial_mean = vec(mean(hsi_mean, dims=2))

    # Monthly discretization
    monthly_hsi, month_lookup = build_monthly_hsi_matrix(
        hsi_mean, years; ref_doy = ref_doy
    )

    return (
        sims             = sims,
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
    map_snowcrab_telemetry_to_units(
        telemetry_df::DataFrame,
        au::NamedTuple
    )::DataFrame

Maps observation coordinates `(:lon, :lat)` in a telemetry DataFrame to their nearest
areal unit in `au` using geodetic centroids `(au.lon, au.lat)`.

# Returns
- `DataFrame`: A copy of `telemetry_df` with appended `:s_idx` column (`1 <= s_i <= S`).
"""
function map_snowcrab_telemetry_to_units(
    telemetry_df::DataFrame,
    au::NamedTuple
)::DataFrame
    out = copy(telemetry_df)
    n_units = au.n_units
    geo_centroids = if hasproperty(au, :lon) && hasproperty(au, :lat)
        hcat([[au.lon[i], au.lat[i]] for i in 1:n_units]...)
    else
        hcat([[au.centroids[i][1], au.centroids[i][2]] for i in 1:n_units]...)
    end
    tree = KDTree(geo_centroids)
    pts_mat = hcat([[out.lon[i], out.lat[i]] for i in 1:nrow(out)]...)
    idxs, _ = knn(tree, pts_mat, 1)
    out.s_idx = [idx[1] for idx in idxs]
    return out
end

"""
    match_telemetry_closest_month_hsi(
        telemetry_df::DataFrame,
        monthly_hsi::AbstractMatrix{<:Real},
        years::AbstractVector{<:Integer},
        month_lookup::Dict{Tuple{Int, Int}, Int}
    )::Vector{Float64}

Matches each telemetry observation to its habitat suitability value in the closest matching month.

# Arguments
- `telemetry_df`: Telemetry DataFrame with `:timestamp` and `:s_idx`.
- `monthly_hsi`: Matrix ``(S \\times 12T)`` of monthly habitat suitability values.
- `years`: Vector of valid annual bounds (e.g. `1999:2025`).
- `month_lookup`: Map from `(year, month)` to column index in `monthly_hsi`.

# Returns
- `Vector{Float64}`: Vector of length `nrow(telemetry_df)` with matching monthly HSI values.
"""
function match_telemetry_closest_month_hsi(
    telemetry_df::DataFrame,
    monthly_hsi::AbstractMatrix{<:Real},
    years::AbstractVector{<:Integer},
    month_lookup::Dict{Tuple{Int, Int}, Int}
)::Vector{Float64}
    y_min, y_max = extrema(years)
    hsi_matched = zeros(Float64, nrow(telemetry_df))
    for i in 1:nrow(telemetry_df)
        dt = telemetry_df.timestamp[i]
        yr = clamp(year(dt), y_min, y_max)
        mo = month(dt)
        col_idx = get(month_lookup, (yr, mo), 1)
        s_unit = telemetry_df.s_idx[i]
        hsi_matched[i] = monthly_hsi[s_unit, col_idx]
    end
    return hsi_matched
end

"""
    build_snowcrab_hexagonal_mesh(
        tagging::DataFrame,
        au_src::NamedTuple;
        radius_km::Real = 5.0
    )::NamedTuple

Constructs a regular hexagonal mesh with circumradius `radius_km` (default 5.0 km) covering
the snow crab telemetry observation domain and computes adjacency graph topology.

# Returns
- `NamedTuple`: Destination hexagonal areal units `au_hex`.
"""
function build_snowcrab_hexagonal_mesh(
    tagging::DataFrame,
    au_src::Union{Nothing, NamedTuple} = nothing;
    radius_km::Real = 5.0,
    wkt::AbstractString = "EPSG:4326"
)::NamedTuple
    s_x = tagging.lon
    s_y = tagging.lat
    mean_lat = mean(s_y)
    deg_per_km_lat = 1.0 / 111.0
    radius = Float64(radius_km) * deg_per_km_lat

    dx = sqrt(3.0) * radius
    dy = 1.5 * radius

    # Identify unique active hexagon centers containing observation points
    center_set = Set{Tuple{Float64, Float64}}()
    for (x, y) in zip(s_x, s_y)
        r0 = round(Int, y / dy)
        x_off = isodd(r0) ? (dx / 2.0) : 0.0
        c0 = round(Int, (x - x_off) / dx)
        cx = c0 * dx + x_off
        cy = r0 * dy
        push!(center_set, (cx, cy))
    end

    centroids = collect(center_set)
    sort!(centroids, by = c -> (c[2], c[1]))
    n_units = length(centroids)

    # 6 vertices for each regular hexagon
    polygons = Vector{Vector{Tuple{Float64, Float64}}}()
    for (cx, cy) in centroids
        verts = Tuple{Float64, Float64}[]
        for k in 0:5
            angle = deg2rad(30.0 + 60.0 * k)
            vx = cx + radius * cos(angle)
            vy = cy + radius * sin(angle)
            push!(verts, (vx, vy))
        end
        push!(verts, verts[1])
        push!(polygons, verts)
    end

    # Adjacency via neighbor distance (adjacent regular hexagons: distance <= sqrt(3)*radius * 1.05)
    g = SimpleGraph(n_units)
    c_mat = hcat([[c[1], c[2]] for c in centroids]...)
    tree = KDTree(c_mat)
    max_adj_dist = sqrt(3.0) * radius * 1.05

    for i in 1:n_units
        idxs = inrange(tree, [centroids[i][1], centroids[i][2]], max_adj_dist)
        for j in idxs
            if i < j
                add_edge!(g, i, j)
            end
        end
    end

    # Ensure fully connected graph across disconnected clusters
    g = ensure_connected!(g, centroids)
    W = Float64.(Graphs.adjacency_matrix(g))

    adj_edges = Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}[]
    for e in Graphs.edges(g)
        push!(adj_edges, (centroids[src(e)], centroids[dst(e)]))
    end

    # Point assignments
    assigns_knn, _ = knn(tree, hcat([[x, y] for (x, y) in zip(s_x, s_y)]...), 1)
    new_assigns = [a[1] for a in assigns_knn]

    # Hexagon area in km2: (3 * sqrt(3) / 2) * radius_km^2
    hex_area_km2 = (1.5 * sqrt(3.0)) * (radius_km^2)
    areas = fill(hex_area_km2, n_units)
    point_counts = [count(==(i), new_assigns) for i in 1:n_units]

    xs = [c[1] for c in centroids]
    ys = [c[2] for c in centroids]
    min_x, max_x = extrema(xs)
    min_y, max_y = extrema(ys)
    hull_box = [
        (min_x, min_y), (max_x, min_y),
        (max_x, max_y), (min_x, max_y),
        (min_x, min_y)
    ]

    return (
        centroids = centroids,
        polygons = polygons,
        adjacency_edges = adj_edges,
        graph = g,
        W = W,
        hull_coords = hull_box,
        s_idx = new_assigns,
        assignments = new_assigns,
        s_x = xs,
        s_y = ys,
        lon = xs,
        lat = ys,
        s_vals = collect(1:n_units),
        areas = areas,
        point_counts = point_counts,
        n_units = n_units,
        wkt = wkt
    )
end


"""
    load_snowcrab_tagging_data(;
        db_con = nothing,
        data_dir::AbstractString = joinpath(@__DIR__, "data"),
        sources::Vector{Symbol} = [:otn, :bio],
        filter_dead::Bool = true,
        time_threshold_days::Real = 30.0,
        dist_threshold_meters::Real = 50.0,
        historical_lookup_file::AbstractString = "tags_summary1993_2005.csv"
    )::DataFrame

Loads and post-processes snow crab mark-recapture and telemetry observation records
from cached JLD2/DuckDB tables or raw CSV tables.
"""
function load_snowcrab_tagging_data(;
    db_con = nothing,
    data_dir::AbstractString = joinpath(@__DIR__, "data"),
    sources::Vector{Symbol} = [:otn, :bio],
    filter_dead::Bool = true,
    time_threshold_days::Real = 30.0,
    dist_threshold_meters::Real = 50.0,
    historical_lookup_file::AbstractString = "tags_summary1993_2005.csv"
)::DataFrame

    if isnothing(db_con)
        jld2_cache = joinpath(data_dir, "tagging.jld2")
        duckdb_cache = joinpath(data_dir, "tagging.duckdb")
        if isfile(jld2_cache)
            try
                tagging = JLD2.load(jld2_cache, "tagging")
                if filter_dead && hasproperty(tagging, :is_dead)
                    tagging = filter(r -> !r.is_dead, tagging)
                end
                valid_tags = String[]
                for sub in groupby(tagging, :tagid)
                    rel = filter(r -> r.tag == 0, sub)
                    nrow(rel) == 1 || continue
                    rel_t = rel[1, :time]
                    recaps = filter(r -> r.tag > 0, sub)
                    !isempty(recaps) || continue
                    all(recaps.time .> rel_t) || continue
                    push!(valid_tags, string(first(sub.tagid)))
                end
                filter!(r -> string(r.tagid) in Set(valid_tags), tagging)
                tagging[!, :lon] = Float64[-abs(Float64(x)) for x in tagging.lon]
                tagging[!, :lat] = Float64[Float64(y) for y in tagging.lat]
                tagging[!, :tag] = Int[Int(t) for t in tagging.tag]
                tagging[!, :time] = Float64[Float64(t) for t in tagging.time]
                tagging[!, :tagid] = String[string(id) for id in tagging.tagid]
                sort!(tagging, [:tagid, :time])
                return tagging
            catch e
                @warn "Failed to load cached tagging.jld2: $e. Proceeding with table parsing."
            end
        elseif isfile(duckdb_cache)
            try
                db = DuckDB.DB(duckdb_cache)
                con = DuckDB.connect(db)
                tagging = try
                    DuckDB.query(con, "SELECT * FROM tagging") |> DataFrame
                finally
                    DuckDB.disconnect(con)
                    DuckDB.close(db)
                end
                if filter_dead && hasproperty(tagging, :is_dead)
                    tagging = filter(r -> !r.is_dead, tagging)
                end
                valid_tags = String[]
                for sub in groupby(tagging, :tagid)
                    rel = filter(r -> r.tag == 0, sub)
                    nrow(rel) == 1 || continue
                    rel_t = rel[1, :time]
                    recaps = filter(r -> r.tag > 0, sub)
                    !isempty(recaps) || continue
                    all(recaps.time .> rel_t) || continue
                    push!(valid_tags, string(first(sub.tagid)))
                end
                filter!(r -> string(r.tagid) in Set(valid_tags), tagging)
                tagging[!, :lon] = Float64[-abs(Float64(x)) for x in tagging.lon]
                tagging[!, :lat] = Float64[Float64(y) for y in tagging.lat]
                tagging[!, :tag] = Int[Int(t) for t in tagging.tag]
                tagging[!, :time] = Float64[Float64(t) for t in tagging.time]
                tagging[!, :tagid] = String[string(id) for id in tagging.tagid]
                sort!(tagging, [:tagid, :time])
                return tagging
            catch e
                @warn "Failed to load cached tagging.duckdb: $e. Proceeding with table parsing."
            end
        end
    end

    error("Could not load tagging data cache from $(data_dir).")
end

"""
    prepare_snowcrab_telemetry(;
        db_con = nothing,
        data_dir::AbstractString = joinpath(@__DIR__, "data"),
        output_dir::AbstractString = joinpath(@__DIR__, "output"),
        sources::Vector{Symbol} = [:otn, :bio],
        filter_dead::Bool = true,
        time_interval::Symbol = :monthly
    )::NamedTuple

Ingests and filters snow crab telemetry records, aggregates observations into regular
monthly or weekly intervals, and generates interactive trajectory visualizations.
"""
function prepare_snowcrab_telemetry(;
    db_con = nothing,
    data_dir::AbstractString = joinpath(@__DIR__, "data"),
    output_dir::AbstractString = joinpath(@__DIR__, "output"),
    sources::Vector{Symbol} = [:otn, :bio],
    filter_dead::Bool = true,
    time_interval::Symbol = :monthly
)::NamedTuple

    mkpath(output_dir)
    println("========================================================")
    println("  Snow Crab Telemetry & Mark-Recapture Ingestion Pipeline")
    println("========================================================")
    println("  Data directory   : $(data_dir)")
    println("  Output directory : $(output_dir)")
    println("  Time interval    : $(time_interval)\n")

    # 1. Ingest observation records
    println("1. Loading raw observation records …")
    tagging_raw = load_snowcrab_tagging_data(;
        db_con = db_con,
        data_dir = data_dir,
        sources = sources,
        filter_dead = filter_dead
    )
    println("   Loaded $(nrow(tagging_raw)) raw records across $(length(unique(tagging_raw.tagid))) individuals.")

    # 2. Temporal aggregation
    println("2. Aggregating observation records to :$(time_interval) intervals …")
    tagging = aggregate_telemetry_time(tagging_raw; time_interval = time_interval)
    println("   Aggregated dataset: $(nrow(tagging)) records across $(length(unique(tagging.tagid))) trajectories.")

    # 3. Activity statistics
    println("3. Computing trajectory statistics …")
    stats = summarize_tag_activity(tagging)

    # 4. Interactive trajectory map
    println("4. Generating Leaflet trajectory map …")
    track_coords = Vector{Vector{Tuple{Float64, Float64}}}()
    gdf = groupby(tagging, :tagid)
    n_sample_tracks = min(30, length(gdf))
    sample_tags = collect(keys(gdf))[1:n_sample_tracks]

    for k in sample_tags
        sub = gdf[k]
        nrow(sub) < 2 && continue
        push!(track_coords, [(Float64(r.lon), Float64(r.lat)) for r in eachrow(sub)])
    end

    m_snowcrab = if !isempty(track_coords)
        leaflet_tracks_map(track_coords, NamedTuple(); title="Snow Crab Movement Trajectories ($(time_interval))", show_start_end=true)
    else
        nothing
    end

    if !isnothing(m_snowcrab)
        traj_html_path = joinpath(output_dir, "snowcrab_tag_trajectories.html")
        save_html(m_snowcrab, traj_html_path)
        println("   Interactive Leaflet trajectory map saved to: $(traj_html_path)")
    end

    return (
        tagging = tagging,
        summary = stats,
        plot_trajectories = m_snowcrab
    )
end

"""
    run_snowcrab_movement(;
        db_con = nothing,
        data_dir::AbstractString = joinpath(@__DIR__, "data"),
        output_dir::AbstractString = joinpath(@__DIR__, "output"),
        sources::Vector{Symbol} = [:otn, :bio],
        filter_dead::Bool = true,
        sppoly_file::Union{Nothing, AbstractString} = joinpath(data_dir, "sppoly.jld2"),
        hsi_file::Union{Nothing, AbstractString} = joinpath(data_dir, "hsi.jld2"),
        time_interval::Symbol = :monthly,
        area_method::Symbol = :hexagonal,
        radius_km::Real = 5.0,
        n_units::Int = 40,
        reshard_hsi::Bool = true,
        relationship::Symbol = :exponential,
        n_samples::Int = 500,
        n_warmup::Int = 200,
        n_chains::Int = 1,
        save_plots::Bool = true,
        rng::Random.AbstractRNG = Random.GLOBAL_RNG
    )::NamedTuple

End-to-end Snow Crab (*Chionoecetes opilio*) movement estimation pipeline.
1. Ingests telemetry observations and aggregates pings to `:monthly` intervals.
2. Discretizes 3D HSI simulation array (`sims`) to monthly slices, accounting for Sept 1 survey baseline.
3. Reshards HSI field onto a 5 km regular hexagonal spatial tessellation (`radius_km = 5.0`).
4. Matches each crab observation to its spatial unit and closest matching month's HSI.
5. Fits the BSTM movement component via NUTS to estimate advective velocity β and diffusion D.
6. Reconstructs transition kernel Γ̄, stationary distribution π, and renders diagnostic plots.
"""
function run_snowcrab_movement(;
    db_con = nothing,
    data_dir::AbstractString = joinpath(@__DIR__, "data"),
    output_dir::AbstractString = joinpath(@__DIR__, "output"),
    sources::Vector{Symbol} = [:otn, :bio],
    filter_dead::Bool = true,
    sppoly_file::Union{Nothing, AbstractString} = joinpath(data_dir, "sppoly.jld2"),
    hsi_file::Union{Nothing, AbstractString} = joinpath(data_dir, "hsi.jld2"),
    time_interval::Symbol = :monthly,
    area_method::Symbol = :sppoly,
    radius_km::Real = 5.0,
    n_units::Int = 40,
    reshard_hsi::Bool = true,
    relationship::Symbol = :exponential,
    n_samples::Int = 500,
    n_warmup::Int = 200,
    n_chains::Int = 1,
    save_plots::Bool = true,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
)::NamedTuple
    mkpath(output_dir)
    println("\n========================================================")
    println("  Snow Crab End-to-End Movement Estimation Pipeline")
    println("========================================================")

    # 1. Ingest and temporally aggregate telemetry records
    tagging_raw = load_snowcrab_tagging_data(;
        db_con = db_con,
        data_dir = data_dir,
        sources = sources,
        filter_dead = filter_dead
    )
    tagging = aggregate_telemetry_time(tagging_raw; time_interval = time_interval)
    println("  Loaded $(nrow(tagging)) detections across $(length(unique(tagging.tagid))) crabs (interval: :$(time_interval)).")

    # 2. Activity statistics
    stats = summarize_tag_activity(tagging)

    # 3. Load Source Spatial Partitioning (707 units)
    au_src = nothing
    if !isnothing(sppoly_file) && isfile(sppoly_file)
        su_bundle = load_snowcrab_spatial_units(sppoly_file=sppoly_file)
        au_src = su_bundle.au
        println("  Loaded source spatial units (707 units, CRS: $(au_src.wkt)).")
    end

    # 4. Use authoritative marine domain (sppoly) or custom tessellation
    au_used = if area_method in (:sppoly, :tesselation) && !isnothing(au_src)
        println("  Using authoritative 707-unit marine domain with coastline boundaries (sppoly)...")
        au_src
    elseif area_method == :hexagonal
        println("  Building 5 km hexagonal tessellation from observation coordinates...")
        build_snowcrab_hexagonal_mesh(tagging, au_src; radius_km = radius_km)
    elseif !isnothing(au_src)
        au_src
    else
        assign_spatial_units(tagging.lon, tagging.lat; area_method = area_method, target_units = n_units)
    end
    println("  Destination mesh: $(au_used.n_units) units.")

    # 5. Load and Discretize HSI on a Monthly Basis
    hsi_vec = nothing
    if !isnothing(hsi_file) && isfile(hsi_file)
        println("  Loading HSI 3D simulations and discretizing on monthly basis (Sept 1 baseline)...")
        h_bundle = load_snowcrab_hsi(hsi_file = hsi_file)
        monthly_hsi_src = h_bundle.monthly_hsi # 707 x 324
        month_lookup = h_bundle.month_lookup
        years_vec = h_bundle.years

        # Reshard monthly HSI from 707 units onto au_used if needed
        monthly_hsi_dest = if au_used.n_units != size(monthly_hsi_src, 1) && !isnothing(au_src)
            println("  Resharding monthly HSI onto $(au_used.n_units)-unit destination mesh...")
            src_coords = Tuple{Float64, Float64}[(au_src.lon[i], au_src.lat[i]) for i in 1:au_src.n_units]
            n_mo = size(monthly_hsi_src, 2)
            res_mat = zeros(Float64, au_used.n_units, n_mo)
            for m in 1:n_mo
                res_mat[:, m] = reshard_hsi_field(
                    monthly_hsi_src[:, m], au_used; hsi_coords = src_coords
                )
            end
            res_mat
        else
            monthly_hsi_src
        end

        # Map telemetry to destination units and match closest month's HSI
        tagging = map_snowcrab_telemetry_to_units(tagging, au_used)
        tagging.hsi = match_telemetry_closest_month_hsi(
            tagging, monthly_hsi_dest, years_vec, month_lookup
        )
        println("  Matched crab observations to closest monthly HSI (mean: $(round(mean(tagging.hsi), digits=4))).")

        # Climatological spatial mean for destination mesh
        hsi_vec = vec(mean(monthly_hsi_dest, dims=2))
    end

    # 6. Configure and Run Movement Estimation
    opts = MovementOptions(
        n_units          = au_used.n_units,
        area_method      = :tesselation,
        hsi              = hsi_vec,
        reshard_hsi      = false,
        relationship     = relationship,
        n_samples        = n_samples,
        n_warmup         = n_warmup,
        n_chains         = n_chains,
        rng              = rng
    )

    result = run_movement_simple(tagging; opts=opts, au=au_used)

    # 7. Reconstruct State-Space Trajectories for Mark-Recapture Events
    println("7. Reconstructing state-space Markov bridge trajectories for all mark-recapture events …")
    paths = reconstruct_mark_recapture_paths(
        tagging, result;
        time_interval = time_interval,
        max_paths     = nothing,
        smooth_jitter = true,
        rng           = rng
    )
    println("   Reconstructed $(length(paths)) individual trajectories.")

    # 8. Macro-Regional Connectivity
    cents_lon = [c[1] for c in result.au.centroids]
    mid_lon = (minimum(cents_lon) + maximum(cents_lon)) / 2.0
    strata = [x > mid_lon ? "East" : "West" for x in cents_lon]
    C_regional = calculate_regional_connectivity(result.transition_matrix, strata)

    # 9. Persistence
    db_path = joinpath(output_dir, "snowcrab_movement.duckdb")
    jld2_path = joinpath(output_dir, "snowcrab_movement.jld2")
    save_movement_bundle(result; db_path=db_path, jld2_path=jld2_path)

    # 10. Render Diagnostics Plots
    plots_bundle = nothing
    if save_plots
        p_dir = joinpath(output_dir, "plots")
        plots_bundle = generate_movement_plots(
            result, paths;
            output_dir = p_dir,
            hsi        = result.hsi,
            strata     = strata
        )
    end

    println("========================================================\n")
    return (
        tagging           = tagging,
        activity_summary  = stats,
        result            = result,
        paths             = paths,
        regional_transfer = C_regional,
        plots             = plots_bundle
    )
end

"""
    visualize_snowcrab_movement(;
        data_dir::AbstractString = joinpath(@__DIR__, "data"),
        output_dir::AbstractString = joinpath(@__DIR__, "output"),
        sppoly_file::Union{Nothing, AbstractString} = joinpath(data_dir, "sppoly.jld2"),
        hsi_file::Union{Nothing, AbstractString} = joinpath(data_dir, "hsi.jld2"),
        time_interval::Symbol = :monthly,
        radius_km::Real = 5.0,
        n_sim_paths::Int = 50,
        n_sim_steps::Int = 40,
        rho_persistence::Real = 0.85,
        max_empirical_paths::Int = 1000,
        rng::Random.AbstractRNG = Random.GLOBAL_RNG
    )::NamedTuple

Generates comprehensive publication-grade interactive Leaflet maps and diagnostics
for snow crab movement using either persisted simulation results or raw data.
"""
function visualize_snowcrab_movement(;
    data_dir::AbstractString = joinpath(@__DIR__, "data"),
    output_dir::AbstractString = joinpath(@__DIR__, "output"),
    sppoly_file::Union{Nothing, AbstractString} = joinpath(data_dir, "sppoly.jld2"),
    hsi_file::Union{Nothing, AbstractString} = joinpath(data_dir, "hsi.jld2"),
    time_interval::Symbol = :monthly,
    radius_km::Real = 5.0,
    n_sim_paths::Int = 50,
    n_sim_steps::Int = 40,
    rho_persistence::Real = 0.85,
    max_empirical_paths::Int = 1000,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
)::NamedTuple
    mkpath(output_dir)
    plots_dir = joinpath(output_dir, "plots")
    mkpath(plots_dir)
    println("\n========================================================")
    println("  Snow Crab Movement & Habitat Visualization Engine")
    println("========================================================")

    # 1. Load tagging records
    println("1. Loading and aggregating observation records (:$(time_interval)) …")
    tagging_raw = load_snowcrab_tagging_data(data_dir=data_dir, sources=[:otn, :bio], filter_dead=true)
    tagging = aggregate_telemetry_time(tagging_raw; time_interval=time_interval)
    println("   Empirical dataset: $(nrow(tagging)) records across $(length(unique(tagging.tagid))) crabs.")

    # Build empirical tracks
    emp_tracks = Vector{Vector{Tuple{Float64, Float64}}}()
    for sub in groupby(tagging, :tagid)
        if nrow(sub) >= 2
            push!(emp_tracks, [(Float64(r.lon), Float64(r.lat)) for r in eachrow(sub)])
        end
    end
    println("   Built $(length(emp_tracks)) empirical trajectory tracks.")

    # 2. Load authoritative marine domain (sppoly) and HSI
    su_bundle = load_snowcrab_spatial_units(sppoly_file=sppoly_file)
    au_src = su_bundle.au
    h_bundle = load_snowcrab_hsi(hsi_file=hsi_file)
    hsi_annual_mean = vec(mean(h_bundle.monthly_hsi, dims=2))

    jld2_path = joinpath(output_dir, "snowcrab_movement.jld2")
    local au_used, hsi_vec, Gamma, result_bundle

    if isfile(jld2_path)
        println("2. Loading fitted movement model from $(jld2_path) …")
        saved = load(jld2_path)
        au_saved = saved["au"]
        if au_saved.n_units == au_src.n_units
            au_used = au_saved
            Gamma = saved["transition_matrix"]
            opts_loaded = haskey(saved, "opts") ? saved["opts"] : nothing
            hsi_vec = !isnothing(opts_loaded) && hasproperty(opts_loaded, :hsi) && !isnothing(opts_loaded.hsi) ? opts_loaded.hsi : hsi_annual_mean
            result_bundle = (
                au = au_used,
                transition_matrix = Gamma,
                chain = haskey(saved, "chain") ? saved["chain"] : nothing,
                opts = opts_loaded,
                hsi = hsi_vec
            )
        else
            au_used = au_src
            hsi_vec = hsi_annual_mean
            W_dir = compute_directed_adjacency_matrix(hsi_vec, au_used.W; relationship=:exponential)
            W_sym = Symmetric(au_used.W)
            L = Diagonal(vec(sum(W_sym, dims=2))) - W_sym
            I_S = Matrix{Float64}(I, au_used.n_units, au_used.n_units)
            Gamma = inv(I_S - 0.75 * W_dir - 0.15 * L)
            for i in 1:au_used.n_units
                Gamma[i, :] .= max.(0.0, Gamma[i, :])
                rs = sum(Gamma[i, :])
                rs > 1e-12 && (Gamma[i, :] ./= rs)
            end
            result_bundle = (
                au = au_used,
                transition_matrix = Gamma,
                chain = nothing,
                opts = MovementOptions(n_units=au_used.n_units, area_method=:sppoly, hsi=hsi_vec),
                hsi = hsi_vec
            )
        end
    else
        println("2. Constructing marine transition matrix on 707-unit Scotian Shelf domain …")
        au_used = au_src
        hsi_vec = hsi_annual_mean
        W_dir = compute_directed_adjacency_matrix(hsi_vec, au_used.W; relationship=:exponential)
        W_sym = Symmetric(au_used.W)
        L = Diagonal(vec(sum(W_sym, dims=2))) - W_sym
        I_S = Matrix{Float64}(I, au_used.n_units, au_used.n_units)
        Gamma = inv(I_S - 0.75 * W_dir - 0.15 * L)
        for i in 1:au_used.n_units
            Gamma[i, :] .= max.(0.0, Gamma[i, :])
            rs = sum(Gamma[i, :])
            rs > 1e-12 && (Gamma[i, :] ./= rs)
        end
        result_bundle = (
            au = au_used,
            transition_matrix = Gamma,
            chain = nothing,
            opts = MovementOptions(n_units=au_used.n_units, area_method=:sppoly, hsi=hsi_vec),
            hsi = hsi_vec
        )
    end

    # 3. Reconstruct State-Space Trajectories for Every Known Mark-Recapture Event
    println("3. Reconstructing state-space Markov bridge trajectories for all mark-recapture events …")
    reconstructed_paths = reconstruct_mark_recapture_paths(
        tagging, result_bundle;
        time_interval = time_interval,
        max_paths     = nothing,
        smooth_jitter = true,
        rng           = rng
    )
    println("   Successfully reconstructed $(length(reconstructed_paths)) individual mark-recapture trajectories.")

    # 4. Generate the full 9-panel diagnostic suite and dashboard
    println("4. Rendering 9-panel Leaflet diagnostic suite in $(plots_dir) …")
    cents_lon = [c[1] for c in au_used.centroids]
    mid_lon = (minimum(cents_lon) + maximum(cents_lon)) / 2.0
    strata = [x > mid_lon ? "East" : "West" for x in cents_lon]

    plots_bundle = generate_movement_plots(
        result_bundle, reconstructed_paths;
        output_dir = plots_dir,
        hsi        = hsi_vec,
        strata     = strata
    )

    # 5. Generate high-density combined multi-layer trajectory map
    println("5. Generating combined multi-layer trajectory map for all mark-recapture events …")
    m_combined = leaflet_tracks_map(
        reconstructed_paths, au_used;
        empirical_paths     = emp_tracks,
        hsi                 = hsi_vec,
        background          = :polygons,
        title               = "Snow Crab Mark-Recapture Trajectories & Habitat Suitability (HSI)",
        show_start_end      = true,
        max_paths           = length(reconstructed_paths),
        max_empirical_paths = length(emp_tracks)
    )
    combined_html_path = joinpath(output_dir, "snowcrab_tag_trajectories.html")
    save_html(m_combined, combined_html_path)
    println("   ✓ Multi-layer trajectory map saved to: $(combined_html_path)")
    println("   ✓ Interactive Movement Dashboard: $(joinpath(plots_dir, "movement_dashboard.html"))")
    println("   ✓ HSI Choropleth Map: $(joinpath(plots_dir, "hsi_choropleth.html"))")
    println("   ✓ Advective Drift Vectors: $(joinpath(plots_dir, "advection_arrows.html"))")
    println("========================================================\n")

    return (
        tagging         = tagging,
        result          = result_bundle,
        paths           = reconstructed_paths,
        empirical_paths = emp_tracks,
        plots           = plots_bundle,
        map_combined    = m_combined
    )
end

function main(args_vec::Vector{String} = ARGS)
    args = Dict{String, Any}(
        "samples"       => 500,
        "warmup"        => 200,
        "chains"        => 1,
        "time-interval" => "monthly",
        "area-method"   => "sppoly",
        "radius"        => 5.0,
        "output-dir"    => joinpath(@__DIR__, "output"),
        "data-dir"      => joinpath(@__DIR__, "data"),
        "sppoly-file"   => joinpath(@__DIR__, "data", "sppoly.jld2"),
        "hsi-file"      => joinpath(@__DIR__, "data", "hsi.jld2"),
        "plot"          => true,
        "run-movement"  => false,
        "prepare-only"  => false,
        "viz"           => false,
        "test"          => false
    )

    for arg in args_vec
        if arg == "--help" || arg == "-h"
            println("Usage: julia snowcrab.jl [OPTIONS]")
            println("  --time-interval=VAL  Temporal aggregation: monthly (default), weekly, daily, raw")
            println("  --area-method=VAL    Spatial tessellation: sppoly (default), hexagonal, tesselation")
            println("  --radius=FLOAT       Hexagonal circumradius in km (default: 5.0)")
            println("  --samples=INT        NUTS samples (default: 500)")
            println("  --warmup=INT         NUTS warmup steps (default: 200)")
            println("  --chains=INT         MCMC chains (default: 1)")
            println("  --sppoly-file=PATH   Path to sppoly.jld2 (default: docs/movement/data/sppoly.jld2)")
            println("  --hsi-file=PATH      Path to hsi.jld2 (default: docs/movement/data/hsi.jld2)")
            println("  --output-dir=DIR     Output directory (default: docs/movement/output)")
            println("  --data-dir=DIR       Data directory (default: docs/movement/data)")
            println("  --plot               Generate Leaflet plots (default: true)")
            println("  --run-movement       Execute full BSTM movement estimation")
            println("  --viz, --visualize   Generate full multi-layer interactive visualizations & dashboard")
            println("  --prepare-only       Only prepare and filter tagging data")
            println("  --test               Run quick 10-sample test")
            return nothing
        elseif arg == "--test"
            args["test"] = true
            args["run-movement"] = true
            args["samples"] = 10
            args["warmup"] = 5
        elseif arg == "--run-movement"
            args["run-movement"] = true
        elseif arg == "--prepare-only"
            args["prepare-only"] = true
        elseif arg == "--viz" || arg == "--visualize" || arg == "--plot-only"
            args["viz"] = true
        elseif arg == "--plot"
            args["plot"] = true
        elseif startswith(arg, "--time-interval=")
            args["time-interval"] = Base.split(arg, "="; limit=2)[2]
        elseif startswith(arg, "--area-method=")
            args["area-method"] = Base.split(arg, "="; limit=2)[2]
        elseif startswith(arg, "--radius=")
            args["radius"] = parse(Float64, Base.split(arg, "="; limit=2)[2])
        elseif startswith(arg, "--samples=")
            args["samples"] = parse(Int, Base.split(arg, "="; limit=2)[2])
            args["run-movement"] = true
        elseif startswith(arg, "--warmup=")
            args["warmup"] = parse(Int, Base.split(arg, "="; limit=2)[2])
        elseif startswith(arg, "--chains=")
            args["chains"] = parse(Int, Base.split(arg, "="; limit=2)[2])
        elseif startswith(arg, "--output-dir=")
            args["output-dir"] = Base.split(arg, "="; limit=2)[2]
        elseif startswith(arg, "--data-dir=")
            args["data-dir"] = Base.split(arg, "="; limit=2)[2]
        elseif startswith(arg, "--sppoly-file=")
            args["sppoly-file"] = Base.split(arg, "="; limit=2)[2]
        elseif startswith(arg, "--hsi-file=")
            args["hsi-file"] = Base.split(arg, "="; limit=2)[2]
        end
    end

    t_interval = Symbol(args["time-interval"])
    a_method = Symbol(args["area-method"])

    if args["viz"]
        return visualize_snowcrab_movement(
            data_dir      = args["data-dir"],
            output_dir    = args["output-dir"],
            sppoly_file   = args["sppoly-file"],
            hsi_file      = args["hsi-file"],
            time_interval = t_interval,
            radius_km     = args["radius"]
        )
    elseif args["run-movement"] || args["test"]
        return run_snowcrab_movement(
            data_dir      = args["data-dir"],
            output_dir    = args["output-dir"],
            sppoly_file   = args["sppoly-file"],
            hsi_file      = args["hsi-file"],
            time_interval = t_interval,
            area_method   = a_method,
            radius_km     = args["radius"],
            n_samples     = args["samples"],
            n_warmup      = args["warmup"],
            n_chains      = args["chains"],
            save_plots    = args["plot"]
        )
    else
        return prepare_snowcrab_telemetry(
            data_dir      = args["data-dir"],
            output_dir    = args["output-dir"],
            time_interval = t_interval
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

