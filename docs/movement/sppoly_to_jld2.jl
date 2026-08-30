"""
    sppoly_to_jld2.jl

Converts spatial areal unit polygons (`sppoly`) and 3D habitat suitability probability
simulation arrays (`sims`) from RData format (`sppoly_hsi.RData`) into native Julia JLD2
containers (`sppoly.jld2` and `hsi.jld2`).

example: 

include("docs/movement/sppoly_to_jld2.jl")
using .SppolyConverter
using JLD2
# 1. Run conversion from RData (if not already done)
res = sppoly_to_jld2(
    rdata_path = "docs/movement/data/sppoly_hsi.RData",
    output_dir = "docs/movement/data"
)
# 2. Load the serialized artifacts
hsi_bundle = JLD2.load("docs/movement/data/hsi.jld2")
sppoly_bundle = JLD2.load("docs/movement/data/sppoly.jld2")
sims = hsi_bundle["sims"]   # Array{Float64, 3}: (707, 27, 5000)
years = hsi_bundle["years"] # 1999:2025
au = sppoly_bundle["au"]    # Standard BSTM areal units NamedTuple
# 3. Continuous interpolation across time slices for polygon 42 on DOY 150 in 2005
t_cont = 2005.0 + (150 - 1) / 365.25
prob_mean = interpolate_hsi(sims, years, 42, t_cont)
prob_draw1 = interpolate_hsi(sims, years, 42, t_cont; draw_idx=1)


Constructs standard BSTM `au` (Areal Units) NamedTuples, sparse adjacency matrices (`W`),
spatial graph topology (`Graphs.SimpleGraph`), and provides continuous linear time-slice
interpolation utilities for mark-recapture telemetry modeling (`snowcrab.jl`).
"""

module SppolyConverter

using JLD2
using DataFrames
using CSV
using SparseArrays
using Graphs
using Statistics
using LibGEOS

export sppoly_to_jld2, interpolate_hsi, build_monthly_hsi_matrix

"""
    sppoly_to_jld2(;
        rdata_path::AbstractString = joinpath(@__DIR__, "data", "sppoly_hsi.RData"),
        output_dir::AbstractString = joinpath(@__DIR__, "data"),
        rscript_bin::AbstractString = "Rscript",
        extract_hsi::Bool = true
    )::NamedTuple

Extracts `sppoly` and `sims` from an `sf` / `RData` file and serializes them into
Julia JLD2 files (`sppoly.jld2` and `hsi.jld2`) without modifying BSTM package source.

# Mathematical and Structural Details:
1. **Coordinate Systems**:
   - Planar coordinates: Projected UTM Zone 20N in kilometers (`+proj=utm +zone=20 +units=km`).
   - Geographic coordinates: Geodetic WGS84 decimal degrees (`EPSG:4326`).
2. **Graph Topology**:
   - Transforms `NB_graph` (`spdep::nb` list) into a symmetric sparse adjacency matrix:
     ```math
     W_{ij} = \\begin{cases} 1 & \\text{if units } i \\text{ and } j \\text{ share a boundary} \\\\ 0 & \\text{otherwise} \\end{cases}
     ```
3. **Habitat Suitability Index (HSI) 3D Array**:
   - `sims`: Array of dimensions \$S \\times T \\times N_{\\text{draws}}\$
     (e.g., \$707 \\text{ areal units} \\times 27 \\text{ years (1999--2025)} \\times 5000 \\text{ posterior draws}\$).

# Arguments
- `rdata_path::AbstractString`: Path to the source `.RData` file.
- `output_dir::AbstractString`: Destination directory for `sppoly.jld2` and `hsi.jld2`.
- `rscript_bin::AbstractString`: System executable path for `Rscript`.
- `extract_hsi::Bool`: Whether to extract and save the 3D HSI simulation array.

# Returns
- `NamedTuple`: Contains `(:sppoly_path, :hsi_path, :au, :n_units, :years, :n_sims)`.
"""
function sppoly_to_jld2(;
    rdata_path::AbstractString = joinpath(@__DIR__, "data", "sppoly_hsi.RData"),
    output_dir::AbstractString = joinpath(@__DIR__, "data"),
    rscript_bin::AbstractString = "Rscript",
    extract_hsi::Bool = true
)::NamedTuple

    if !isfile(rdata_path)
        error("Source RData file not found: $(rdata_path)")
    end
    mkpath(output_dir)

    tmp_dir = mktempdir()
    println("--> Extracting RData objects using $(rscript_bin)...")

    r_script_path = joinpath(tmp_dir, "extract.R")
    sims_bin_path = joinpath(tmp_dir, "sims_raw.bin")
    csv_table_path = joinpath(tmp_dir, "sppoly_table.csv")
    csv_edges_path = joinpath(tmp_dir, "nb_edges.csv")

    # Generate transient extraction script for R
    r_code = """
    suppressPackageStartupMessages({
      library(sf)
      library(spdep)
    })
    env <- new.env()
    load("$(escape_string(rdata_path))", envir=env)
    sppoly <- env\$sppoly
    sims <- env\$sims

    # 1. Export 3D sims array
    if (!is.null(sims)) {
      writeBin(as.numeric(sims), "$(escape_string(sims_bin_path))", size=8)
    }

    # 2. Transform coordinates and extract geometries
    sppoly_geo <- sf::st_transform(sppoly, 4326)
    geom_col <- attr(sppoly, "sf_column")
    cents_planar <- sf::st_coordinates(sf::st_centroid(sppoly[[geom_col]]))
    cents_geo <- sf::st_coordinates(sf::st_centroid(sppoly_geo[[geom_col]]))

    df_attrs <- sf::st_drop_geometry(sppoly)
    df_attrs\$centroid_x_km <- cents_planar[, 1]
    df_attrs\$centroid_y_km <- cents_planar[, 2]
    df_attrs\$centroid_lon <- cents_geo[, 1]
    df_attrs\$centroid_lat <- cents_geo[, 2]
    df_attrs\$wkt_planar <- sf::st_as_text(sppoly[[geom_col]])
    df_attrs\$wkt_geo <- sf::st_as_text(sppoly_geo[[geom_col]])
    write.csv(df_attrs, "$(escape_string(csv_table_path))", row.names=FALSE)

    # 3. Export NB_graph neighborhood edges
    nb_g <- attr(sppoly, "NB_graph")
    edge_from <- integer()
    edge_to <- integer()
    for (i in 1:length(nb_g)) {
      nbrs <- nb_g[[i]]
      if (length(nbrs) > 0 && nbrs[1] != 0) {
        edge_from <- c(edge_from, rep(i, length(nbrs)))
        edge_to <- c(edge_to, nbrs)
      }
    }
    write.csv(data.frame(from_unit=edge_from, to_unit=edge_to),
              "$(escape_string(csv_edges_path))", row.names=FALSE)
    """

    write(r_script_path, r_code)
    run(`$(rscript_bin) $(r_script_path)`)

    # -------------------------------------------------------------------------
    # Parse Spatial Geometry and Graph Topology
    # -------------------------------------------------------------------------
    df_sppoly = CSV.read(csv_table_path, DataFrame)
    df_edges = CSV.read(csv_edges_path, DataFrame)
    S = nrow(df_sppoly)

    safe_float(x) = something(tryparse(Float64, string(x)), NaN)
    safe_int(x) = something(tryparse(Int, string(x)), 0)

    g = SimpleGraph(S)
    for row in eachrow(df_edges)
        u = safe_int(row.from_unit)
        v = safe_int(row.to_unit)
        if u > 0 && v > 0
            add_edge!(g, u, v)
        end
    end
    W_sparse = sparse(Float64.(Graphs.adjacency_matrix(g)))

    cx_km = [safe_float(x) for x in df_sppoly.centroid_x_km]
    cy_km = [safe_float(x) for x in df_sppoly.centroid_y_km]
    c_lon = [safe_float(x) for x in df_sppoly.centroid_lon]
    c_lat = [safe_float(x) for x in df_sppoly.centroid_lat]

    cents_planar = Tuple{Float64, Float64}[(cx_km[i], cy_km[i]) for i in 1:S]
    cents_geo = Tuple{Float64, Float64}[(c_lon[i], c_lat[i]) for i in 1:S]

    polys_planar = Vector{Vector{Tuple{Float64, Float64}}}()
    for row in eachrow(df_sppoly)
        wkt_str = string(row.wkt_planar)
        try
            geom = LibGEOS.readgeom(wkt_str)
            pts = Tuple{Float64, Float64}[]
            if geom isa LibGEOS.Polygon
                seq = LibGEOS.getCoordSeq(LibGEOS.exteriorRing(geom))
                for i in 1:LibGEOS.getSize(seq)
                    push!(pts, (LibGEOS.getX(seq, i), LibGEOS.getY(seq, i)))
                end
            elseif geom isa LibGEOS.MultiPolygon
                n_sub = LibGEOS.numGeometries(geom)
                best_area = -1.0
                best_seq = nothing
                for k in 1:n_sub
                    sub_p = LibGEOS.getGeometry(geom, k)
                    a = LibGEOS.area(sub_p)
                    if a > best_area
                        best_area = a
                        best_seq = LibGEOS.getCoordSeq(LibGEOS.exteriorRing(sub_p))
                    end
                end
                if best_seq !== nothing
                    for i in 1:LibGEOS.getSize(best_seq)
                        push!(pts, (LibGEOS.getX(best_seq, i), LibGEOS.getY(best_seq, i)))
                    end
                end
            end
            push!(polys_planar, pts)
        catch
            push!(polys_planar, Tuple{Float64, Float64}[])
        end
    end

    adjacency_edges = Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}[]
    for e in Graphs.edges(g)
        push!(adjacency_edges, (cents_planar[src(e)], cents_planar[dst(e)]))
    end

    all_x = [c[1] for c in cents_planar]
    all_y = [c[2] for c in cents_planar]
    hull_coords = [
        (minimum(all_x), minimum(all_y)),
        (maximum(all_x), minimum(all_y)),
        (maximum(all_x), maximum(all_y)),
        (minimum(all_x), maximum(all_y)),
        (minimum(all_x), minimum(all_y))
    ]

    # Standard BSTM au NamedTuple
    au = (
        centroids          = cents_planar,
        polygons           = polys_planar,
        adjacency_edges    = adjacency_edges,
        graph              = g,
        W                  = W_sparse,
        hull_coords        = hull_coords,
        s_idx              = collect(1:S),
        s_x                = cx_km,
        s_y                = cy_km,
        lon                = c_lon,
        lat                = c_lat,
        s_vals             = collect(1:S),
        areas              = [safe_float(x) for x in df_sppoly.au_sa_km2],
        point_counts       = [safe_int(x) for x in df_sppoly.npts],
        n_units            = S,
        wkt                = "+proj=utm +ellps=WGS84 +zone=20 +units=km",
        wkt_geo            = "EPSG:4326",
        termination_reason = "bio.snowcrab $(S)-unit tessellation"
    )

    metadata = Dict(
        :project_name => "bio.snowcrab",
        :spatial_domain => "snowcrab",
        :n_units => S,
        :crs_planar => "+proj=utm +ellps=WGS84 +zone=20 +units=km",
        :crs_geo => "EPSG:4326"
    )

    sppoly_path = joinpath(output_dir, "sppoly.jld2")
    JLD2.save(
        sppoly_path,
        "sppoly", df_sppoly,
        "au", au,
        "W", W_sparse,
        "graph", g,
        "centroids_planar", cents_planar,
        "centroids_geo", cents_geo,
        "wkt_planar", String.(df_sppoly.wkt_planar),
        "wkt_geo", String.(df_sppoly.wkt_geo),
        "metadata", metadata
    )
    println("--> Saved spatial units to: $(sppoly_path)")

    # -------------------------------------------------------------------------
    # Parse and Save 3D HSI Simulation Array (if present)
    # -------------------------------------------------------------------------
    hsi_path = joinpath(output_dir, "hsi.jld2")
    years = Int[]
    n_sims = 0

    if extract_hsi && isfile(sims_bin_path)
        file_bytes = filesize(sims_bin_path)
        # 707 units x 27 years x 5000 draws x 8 bytes = 763,560,000 bytes
        T = 27
        N_draws = 5000
        sims = Array{Float64, 3}(undef, S, T, N_draws)
        read!(sims_bin_path, sims)

        years = collect(1999:2025)
        n_sims = N_draws
        auids = collect(1:S)

        JLD2.save(
            hsi_path,
            "sims", sims,
            "years", years,
            "auids", auids
        )
        println("--> Saved 3D HSI array ($(S)x$(T)x$(N_draws)) to: $(hsi_path)")
    end

    # Clean temporary scratch directory
    rm(tmp_dir; recursive=true, force=true)

    return (
        sppoly_path = sppoly_path,
        hsi_path    = hsi_path,
        au          = au,
        n_units     = S,
        years       = years,
        n_sims      = n_sims
    )
end

"""
    interpolate_hsi(
        sims::AbstractArray{<:Real},
        years::AbstractVector{<:Integer},
        s_idx::Integer,
        decimal_year::Real;
        ref_doy::Real = 244.0,
        draw_idx::Union{Nothing, Integer} = nothing
    )::Float64

Performs seasonal linear interpolation of habitat suitability probabilities ``H(s, t)``
between adjacent annual predictions centered on September 1 (`ref_doy = 244.0`):
```math
t_{\\text{ref}} = \\frac{\\text{ref\\_doy} - 1}{365.25} \\approx 0.6653
```
```math
u = t - t_{\\text{ref}}, \\quad Y_1 = \\lfloor u \\rfloor, \\quad \\alpha = u - \\lfloor u \\rfloor
```
```math
H(s, t) = (1 - \\alpha) H(s, Y_1) + \\alpha H(s, Y_1 + 1)
```

# Arguments
- `sims`: 3D array ``(S \\times T \\times N_{\\text{draws}})`` or 2D matrix ``(S \\times T)``.
- `years`: Discrete annual year coordinates (e.g. `1999:2025`).
- `s_idx`: Spatial areal unit index (``1 \\le s \\le S``).
- `decimal_year`: Continuous temporal coordinate (e.g. `2004.45`).
- `ref_doy`: Reference survey day of year (default `244.0` for Sept 1).
- `draw_idx`: Optional posterior draw index (if `nothing`, uses posterior mean).

# Returns
- `Float64`: Interpolated habitat suitability probability in ``[0, 1]``.
"""
function interpolate_hsi(
    sims::AbstractArray{<:Real},
    years::AbstractVector{<:Integer},
    s_idx::Integer,
    decimal_year::Real;
    ref_doy::Real = 244.0,
    draw_idx::Union{Nothing, Integer} = nothing
)::Float64
    y_min = first(years)
    y_max = last(years)
    t_offset = (ref_doy - 1.0) / 365.25
    u = decimal_year - t_offset
    u_clamped = clamp(Float64(u), Float64(y_min), Float64(y_max))

    y_floor = floor(Int, u_clamped)
    frac = u_clamped - y_floor

    t1 = clamp(y_floor - y_min + 1, 1, length(years))
    t2 = clamp(t1 + 1, 1, length(years))

    val1 = if ndims(sims) == 3
        if isnothing(draw_idx)
            mean(@view sims[s_idx, t1, :])
        else
            Float64(sims[s_idx, t1, draw_idx])
        end
    else
        Float64(sims[s_idx, t1])
    end

    val2 = if ndims(sims) == 3
        if isnothing(draw_idx)
            mean(@view sims[s_idx, t2, :])
        else
            Float64(sims[s_idx, t2, draw_idx])
        end
    else
        Float64(sims[s_idx, t2])
    end

    return clamp((1.0 - frac) * val1 + frac * val2, 0.0, 1.0)
end

"""
    build_monthly_hsi_matrix(
        sims::AbstractArray{<:Real},
        years::AbstractVector{<:Integer};
        ref_doy::Real = 244.0,
        draw_idx::Union{Nothing, Integer} = nothing
    )::Tuple{Matrix{Float64}, Dict{Tuple{Int, Int}, Int}}

Discretizes the HSI simulation array onto a regular monthly basis (12 months per year)
centered at the midpoint of each month, accounting for the September 1 survey baseline.

# Returns
- `Tuple`:
  - `Matrix{Float64}`: Array of shape ``(S \\times 12T)``.
  - `Dict{Tuple{Int, Int}, Int}`: Map from `(year, month)` to column index.
"""
function build_monthly_hsi_matrix(
    sims::AbstractArray{<:Real},
    years::AbstractVector{<:Integer};
    ref_doy::Real = 244.0,
    draw_idx::Union{Nothing, Integer} = nothing
)::Tuple{Matrix{Float64}, Dict{Tuple{Int, Int}, Int}}
    S_units = size(sims, 1)
    T_count = length(years)
    n_months = T_count * 12
    monthly_mat = zeros(Float64, S_units, n_months)
    month_lookup = Dict{Tuple{Int, Int}, Int}()

    sims_2d = if ndims(sims) == 3
        if isnothing(draw_idx)
            dropdims(mean(sims, dims=3), dims=3)
        else
            sims[:, :, draw_idx]
        end
    else
        sims
    end

    col = 1
    for (t_idx, yr) in enumerate(years)
        for m in 1:12
            dec_yr = Float64(yr) + (Float64(m) - 0.5) / 12.0
            for s in 1:S_units
                monthly_mat[s, col] = interpolate_hsi(
                    sims_2d, years, s, dec_yr; ref_doy = ref_doy
                )
            end
            month_lookup[(yr, m)] = col
            col += 1
        end
    end
    return monthly_mat, month_lookup
end

end # module SppolyConverter

if abspath(PROGRAM_FILE) == @__FILE__
    SppolyConverter.sppoly_to_jld2()
end

