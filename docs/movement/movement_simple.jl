"""
    movement_simple.jl

Pure mark-recapture movement estimation with BSTM.

Estimates a spatial Markov transition kernel Γ (S × S) from individual
telemetry sequences alone.  No abundance / density dynamics are modelled.

Data requirements
-----------------
Telemetry must contain one row per detection event with the columns:

  tagid   :: Int    – unique individual identifier
  lon     :: Float64 – longitude  (decimal degrees or projected metres)
  lat     :: Float64 – latitude   (decimal degrees or projected metres)
  time    :: Float64 – decimal year (e.g. 2021.5 = mid-2021)
  tag     :: Int    – detection type: 0 = mark / release, 1 = first recapture,
                      2 = second recapture, …, n = n-th recapture

Optional inputs
---------------
  hsi     :: AbstractVector{<:Real}  – Habitat Suitability Index per spatial unit
                                        (length S). When provided, advection is
                                        driven by ∇HSI; otherwise isotropic dispersal
                                        is used.

Usage
-----
  # As a script:
  julia --project=. docs/movement/movement_simple.jl \
        --telemetry=data/telemetry.csv \
        --status

  # As a Julia API:
  include("docs/movement/movement_simple.jl")

  opts = MovementOptions(hsi=my_hsi_vec, n_units=40, relationship=:exponential)
  result = run_movement_simple(telemetry_df; opts=opts)

  # Save outputs:
  save_movement_bundle(result; db_path="movement_results.duckdb",
                               jld2_path="movement_model.jld2")
"""

using bstm
using DataFrames
using Turing
using Random
using LinearAlgebra
using SparseArrays
using Printf
using JLD2
using DuckDB
using Distributions

# ── Options struct ────────────────────────────────────────────────────────────

"""
    MovementOptions

Configuration for the pure movement estimation workflow.

# Fields
- `n_units::Int`: Target number of spatial partitioning units (default 40).
- `area_method::Symbol`: Spatial tessellation method; one of `:cvt`, `:hexagonal`,
  `:voronoi`, `:grid` (default `:cvt`).
- `hsi::Union{Nothing, AbstractVector{<:Real}}`: Optional per-unit Habitat
  Suitability Index vector of length `S`. When `nothing`, a uniform (isotropic)
  dispersal kernel is used. When provided, advection weights are derived from
  `∇HSI` via the `relationship` functional form.
- `relationship::Symbol`: HSI-to-advection functional form: `:exponential` (default),
  `:linear`, or `:logistic`.
- `sensitivity::Real`: Advection sensitivity β ≥ 0 for the pre-computed kernel
  visualisation (default 1.0). Note: in the Turing model the velocity parameter
  serves as the fitted sensitivity.
- `diffusion_weight::Real`: Baseline isotropic dispersal weight D_weight ≥ 0
  used in the pre-computed kernel visualisation (default 0.1).
- `method::Symbol`: Euler scheme for the latent ADR state: `:explicit` (AD-friendly,
  default) or `:implicit` (unconditionally stable, not AD-compatible).
- `n_samples::Int`: Total MCMC samples per chain (default 500).
- `n_warmup::Int`: HMC warm-up steps (default 200).
- `n_chains::Int`: Number of MCMC chains (default 1).
- `rng::AbstractRNG`: Random number generator (default `Random.GLOBAL_RNG`).
"""
struct MovementOptions
    n_units          :: Int
    area_method      :: Symbol
    hsi              :: Union{Nothing, Vector{Float64}}
    reshard_hsi      :: Bool
    au_hsi           :: Union{Nothing, NamedTuple}
    hsi_coords       :: Union{Nothing, Vector{Tuple{Float64, Float64}}}
    hsi_area_method  :: Symbol
    relationship     :: Symbol
    sensitivity      :: Float64
    diffusion_weight :: Float64
    method           :: Symbol
    n_samples        :: Int
    n_warmup         :: Int
    n_chains         :: Int
    rng              :: Random.AbstractRNG
end

function MovementOptions(;
    n_units          :: Int                             = 40,
    area_method      :: Symbol                          = :hexagonal,
    hsi              :: Union{Nothing, AbstractVector}  = nothing,
    reshard_hsi      :: Bool                            = false,
    au_hsi           :: Union{Nothing, NamedTuple}      = nothing,
    hsi_coords       :: Union{Nothing, Vector{Tuple{Float64, Float64}}} = nothing,
    hsi_area_method  :: Symbol                          = :grid,
    relationship     :: Symbol                          = :exponential,
    sensitivity      :: Real                            = 1.0,
    diffusion_weight :: Real                            = 0.1,
    method           :: Symbol                          = :explicit,
    n_samples        :: Int                             = 500,
    n_warmup         :: Int                             = 200,
    n_chains         :: Int                             = 1,
    rng              :: Random.AbstractRNG              = Random.GLOBAL_RNG
)
    hsi_v = isnothing(hsi) ? nothing : convert(Vector{Float64}, hsi)
    MovementOptions(
        n_units, area_method, hsi_v, reshard_hsi, au_hsi, hsi_coords, hsi_area_method,
        relationship, Float64(sensitivity), Float64(diffusion_weight),
        method, n_samples, n_warmup, n_chains, rng
    )
end

# ── Main workflow ─────────────────────────────────────────────────────────────

"""
    run_movement_simple(telemetry_df; opts=MovementOptions()) -> NamedTuple

Fits a pure movement model to individual telemetry data.

# Process

1. Validates the telemetry schema (see `validate_telemetry`).
2. Constructs a spatial tessellation on the `(lon, lat)` domain using
   `assign_spatial_units` with `target_units = opts.n_units`.
3. Projects each detection to the nearest spatial unit (`map_to_units`).
4. If `opts.hsi` is provided (length S), builds a directed advection operator
   `A` from the HSI gradient (∇HSI → W_dir → A); otherwise isotropic dispersal.
5. Fits the BSTM `movement` component via Turing NUTS, with mark-recapture
   telemetry as the sole observation likelihood:

       π_m(u_rec | u_rel, k) ∝ [Γ^k]_{u_rel, u_rec}

6. Post-processes the chain to recover the posterior transition kernel Γ̄.

# Arguments
- `telemetry_df::DataFrame`: Telemetry table with columns `:tagid`, `:lon`,
  `:lat`, `:timestamp`, `:tag` and an optional `:individual_covariate` column.
- `opts::MovementOptions`: Workflow configuration (see `MovementOptions`).

# Returns
`NamedTuple` with fields:
- `chain`: Raw Turing MCMC chain.
- `transition_matrix::Matrix{Float64}`: Posterior-mean row-stochastic kernel Γ̄.
- `au::NamedTuple`: Spatial areal units (centroids, W, boundaries).
- `telemetry_mapped::DataFrame`: Input table with appended `:s_idx` column.
- `opts::MovementOptions`: Options used for this run.
- `kernel_prior::SparseMatrixCSC`: Pre-fitted deterministic kernel from
  `compute_suitability_transition_kernel` for comparison / warm-start.
"""
function run_movement_simple(
    telemetry_df :: DataFrame;
    opts         :: MovementOptions = MovementOptions(),
    au           :: Union{Nothing, NamedTuple} = nothing
)::NamedTuple

    println("\n========================================================")
    println("  BSTM  Pure Movement Estimation")
    println("========================================================")

    # 1. Validate ─────────────────────────────────────────────────────────────
    validate_telemetry(telemetry_df)
    n_tags = length(unique(telemetry_df.tagid))
    println("  Tags : $(n_tags)")
    println("  Rows : $(nrow(telemetry_df))")

    # 2. Spatial tessellation ─────────────────────────────────────────────────
    au_used = if isnothing(au)
        println("  Building spatial tessellation ($(opts.n_units) target units, " *
                "method=:$(opts.area_method)) …")
        lon_vec = Float64[Float64(x) for x in telemetry_df.lon]
        lat_vec = Float64[Float64(y) for y in telemetry_df.lat]
        assign_spatial_units(
            lon_vec, lat_vec;
            area_method  = opts.area_method,
            target_units = opts.n_units,
            exact_units  = false
        )
    else
        println("  Using provided spatial tessellation ($(length(au.centroids)) units) …")
        au
    end
    S = length(au_used.centroids)
    println("  Actual units: $(S)")

    # 3. Process & reshard HSI vector (if provided) ───────────────────────────
    working_hsi = if !isnothing(opts.hsi)
        needs_reshard = opts.reshard_hsi || !isnothing(opts.au_hsi) ||
                        !isnothing(opts.hsi_coords) || length(opts.hsi) != S
        if needs_reshard
            println("  Resharding HSI surface ($(length(opts.hsi)) units ($(opts.hsi_area_method)) → $(S) units :$(opts.area_method)) …")
            reshard_hsi_field(
                opts.hsi, au_used;
                au_src          = opts.au_hsi,
                hsi_coords      = opts.hsi_coords,
                hsi_area_method = opts.hsi_area_method
            )
        else
            opts.hsi
        end
    else
        fill(0.5, S)
    end

    # 4. Project detections to units ──────────────────────────────────────────
    tel = map_telemetry_to_units(telemetry_df, au_used)

    # Compute time index: 1-based discrete year relative to first survey year
    time_col = hasproperty(tel, :time) ? :time : :timestamp
    t_min  = minimum(tel[!, time_col])
    tel.t_idx = max.(1, round.(Int, tel[!, time_col] .- t_min) .+ 1)
    T_N = maximum(tel.t_idx)

    # 5. Pre-compute deterministic kernel (for comparison / diagnostics) ───────
    kernel_prior = compute_suitability_transition_kernel(
        working_hsi, au_used.W;
        sensitivity      = opts.sensitivity,
        diffusion_weight = opts.diffusion_weight,
        relationship     = opts.relationship
    )

    # 6. Build BSTM model formula ─────────────────────────────────────────────
    if !hasproperty(tel, :individual_covariate)
        tel.individual_covariate = zeros(Float64, nrow(tel))
    end

    # Build a unit-level dummy response (mean detection count per unit)
    unit_counts = DataFrames.combine(groupby(tel, :s_idx), nrow => :n_detections)
    df_model = DataFrame(
        s_idx  = 1:S,
        t_idx  = ones(Int, S),        # single dummy time period
        count  = zeros(Float64, S),   # zero response — likelihood from telemetry only
        s_x    = [c[1] for c in au_used.centroids],
        s_y    = [c[2] for c in au_used.centroids]
    )
    for row in eachrow(unit_counts)
        df_model[row.s_idx, :count] = Float64(row.n_detections)
    end

    movement_params = Dict{Symbol, Any}(
        :W                   => au_used.W,
        :mark_recapture_data => tel,
        :method              => opts.method,
        :relationship        => opts.relationship
    )
    if !isnothing(opts.hsi)
        movement_params[:habitat] = working_hsi
    end

    println("  Fitting movement model ($(opts.n_samples) samples, " *
            "$(opts.n_warmup) warmup, chains=$(opts.n_chains)) …")

    m_mov = @bstm(
        likelihood(count, family=gaussian) ~
            intercept() +
            movement(s_idx, t_idx,
                velocity  = truncated(Normal(0.0, 1.0), lower=0.0),
                diffusion = truncated(Normal(0.0, 1.0), lower=0.0),
                sigma     = truncated(Normal(0.0, 0.5), lower=0.0),
                beta_het  = Normal(0.0, 1.0)),
        df_model;
        movement_params...
    )

    sampler = NUTS(opts.n_warmup, 0.65)
    chain = if opts.n_chains == 1
        Base.invokelatest(sample, m_mov, sampler, opts.n_samples;
               progress=true, rng=opts.rng, check_model=false)
    else
        Base.invokelatest(sample, m_mov, sampler, MCMCThreads(), opts.n_samples, opts.n_chains;
               progress=true, rng=opts.rng, check_model=false)
    end

    # 6. Reconstruct posterior-mean transition kernel ─────────────────────────
    println("  Reconstructing posterior-mean transition kernel Γ̄ …")
    Gamma_posterior = reconstruct_posterior_kernel(
        chain, au_used.W;
        hsi = !isnothing(opts.hsi) ? working_hsi : nothing,
        relationship = opts.relationship
    )

    v_samps = extract_scalar_param(chain, "velocity")
    d_samps = extract_scalar_param(chain, "diffusion")

    println("\n  Done.")
    println("  Posterior velocity : " *
            "$(round(mean(v_samps), digits=3))")
    println("  Posterior diffusion: " *
            "$(round(mean(d_samps), digits=3))")
    println("========================================================\n")

    return (
        chain              = chain,
        transition_matrix  = Gamma_posterior,
        au                 = au_used,
        telemetry_mapped   = tel,
        opts               = opts,
        hsi                = working_hsi,
        kernel_prior       = kernel_prior
    )
end

# ── Individual segment functions ──────────────────────────────────────────────

"""
    build_tessellation(telemetry_df; opts=MovementOptions()) -> NamedTuple

Constructs only the spatial tessellation. Useful for inspecting unit placement
before running MCMC. Returns the `au` areal-units object.
"""
function build_tessellation(
    telemetry_df :: DataFrame;
    opts         :: MovementOptions = MovementOptions()
)::NamedTuple
    validate_telemetry(telemetry_df)
    au = assign_spatial_units(
        telemetry_df.lon, telemetry_df.lat;
        area_method  = opts.area_method,
        target_units = opts.n_units,
        exact_units  = false
    )
    println("Tessellation: $(length(au.centroids)) units ($(opts.area_method))")
    return au
end

"""
    fit_movement_kernel(telemetry_mapped, au; opts=MovementOptions()) -> (chain, Gamma)

Runs only the Turing MCMC fitting step given a pre-tessellated and unit-mapped
telemetry DataFrame (output of `map_telemetry_to_units`). Returns the raw chain
and the posterior-mean kernel.
"""
function fit_movement_kernel(
    telemetry_mapped :: DataFrame,
    au               :: NamedTuple;
    opts             :: MovementOptions = MovementOptions()
)
    return run_movement_simple(telemetry_mapped; opts=opts, au=au)
end

"""
    compute_prior_kernel(au; opts=MovementOptions()) -> SparseMatrixCSC

Computes the deterministic HSI-weighted Markov kernel using `opts.hsi`, `opts.sensitivity`,
`opts.diffusion_weight`, and `opts.relationship`, without running MCMC. When `opts.hsi`
is `nothing`, a uniform HSI is assumed.
"""
function compute_prior_kernel(
    au   :: NamedTuple;
    opts :: MovementOptions = MovementOptions()
)::SparseMatrixCSC
    S = length(au.centroids)
    hsi_vec = if !isnothing(opts.hsi)
        if opts.reshard_hsi || !isnothing(opts.au_hsi) || !isnothing(opts.hsi_coords) || length(opts.hsi) != S
            reshard_hsi_field(
                opts.hsi, au;
                au_src          = opts.au_hsi,
                hsi_coords      = opts.hsi_coords,
                hsi_area_method = opts.hsi_area_method
            )
        else
            opts.hsi
        end
    else
        fill(0.5, S)
    end

    return compute_suitability_transition_kernel(
        hsi_vec, au.W;
        sensitivity      = opts.sensitivity,
        diffusion_weight = opts.diffusion_weight,
        relationship     = opts.relationship
    )
end

# ── I/O helpers ───────────────────────────────────────────────────────────────

"""
    save_movement_bundle(result;
                         db_path="movement_results.duckdb",
                         jld2_path="movement_model.jld2")

Persists the movement workflow result to:
- A JLD2 file containing the raw MCMC chain and transition matrix.
- A DuckDB table `movement_transition` with the posterior-mean kernel Γ̄.
"""
function save_movement_bundle(
    result    :: NamedTuple;
    db_path   :: AbstractString = "movement_results.duckdb",
    jld2_path :: AbstractString = "movement_model.jld2"
)
    # JLD2 binary checkpoint
    JLD2.jldsave(jld2_path;
        chain             = result.chain,
        transition_matrix = result.transition_matrix,
        au                = result.au,
        opts              = result.opts
    )
    println("  Saved JLD2: $(jld2_path)")

    # DuckDB flat table of the posterior-mean kernel
    S   = size(result.transition_matrix, 1)
    df  = DataFrame(
        from_unit = repeat(1:S, inner=S),
        to_unit   = repeat(1:S, outer=S),
        prob      = vec(result.transition_matrix')
    )
    write_tier_table!(db_path, df, "movement_transition")
    println("  Saved DuckDB: $(db_path) → movement_transition ($(S*S) rows)")
    return nothing
end

"""
    generate_movement_plots(result, paths; output_dir="plots", hsi=nothing,
                            strata=nothing, mode=:leaflet, save_static=false,
                            fmt="png", dpi=150) -> NamedTuple

Generates and saves the full diagnostic suite of movement visualizations in interactive
Leaflet HTML format (default) and optional static figure formats:
1. `hsi_choropleth.html` — Habitat Suitability Index spatial polygon choropleth.
2. `diffusion_map.html` — Spatial dispersal rate choropleth.
3. `residence_time_map.html` — Stationary residence distribution choropleth.
4. `advection_arrows.html` — Advective drift vector quiver arrows overlaid on HSI.
5. `movement_tracks.html` — Simulated / observed movement trajectories on spatial units.
6. `dispersal_kernel.html` — Dispersal distance decay curve with exponential fit.
7. `step_diagnostics.html` — Step lengths and turning angle distributions.
8. `regional_connectivity.html` — Annotated macro-regional transfer matrix heatmap.
9. `movement_dashboard.html` — Unified multi-panel interactive diagnostics dashboard.
"""
function generate_movement_plots(
    result      :: NamedTuple,
    paths       :: Union{AbstractMatrix, AbstractVector};
    output_dir  :: AbstractString = "plots",
    hsi         :: Union{Nothing, AbstractVector{<:Real}} = nothing,
    strata      :: Union{Nothing, AbstractVector} = nothing,
    mode        :: Symbol = :leaflet,
    save_static :: Bool = false,
    fmt         :: AbstractString = "png",
    dpi         :: Integer = 150
)::NamedTuple
    mkpath(output_dir)
    println("\n  Generating movement visualizations in $(output_dir) (mode=:$(mode)) …")

    au = result.au
    Gamma = result.transition_matrix
    raw_hsi = !isnothing(hsi) ? hsi : (hasproperty(result, :hsi) ? result.hsi : (hasproperty(result.opts, :hsi) ? result.opts.hsi : nothing))
    hsi_vec = if !isnothing(raw_hsi)
        length(raw_hsi) == length(au.centroids) ? raw_hsi : reshard_hsi_field(raw_hsi, au)
    else
        nothing
    end

    d_val = hasproperty(result, :chain) ? mean(extract_scalar_param(result.chain, "diffusion")) : 0.5
    v_val = hasproperty(result, :chain) ? mean(extract_scalar_param(result.chain, "velocity")) : 1.0

    cents_x = [c[1] for c in au.centroids]
    mid_x = (minimum(cents_x) + maximum(cents_x)) / 2.0
    strata_vec = !isnothing(strata) ? strata : [x > mid_x ? "East" : "West" for x in cents_x]
    C_reg = calculate_regional_connectivity(Gamma, strata_vec)
    strata_unique = unique(strata_vec)

    # 1. Interactive Leaflet HTML Maps
    m_hsi = !isnothing(hsi_vec) ? leaflet_hsi_map(hsi_vec, au; title="Habitat Suitability Index (HSI)") : nothing
    if !isnothing(m_hsi)
        save_html(m_hsi, joinpath(output_dir, "hsi_choropleth.html"))
    end

    m_diff = leaflet_diffusion_map(d_val, au; title="Posterior Diffusion Field (D ≈ $(round(d_val, digits=3)))")
    save_html(m_diff, joinpath(output_dir, "diffusion_map.html"))

    m_res = leaflet_residence_time_map(Gamma, au; title="Stationary Residence Distribution (π)")
    save_html(m_res, joinpath(output_dir, "residence_time_map.html"))

    m_arrows = leaflet_advection_arrows(
        au;
        hsi = hsi_vec,
        velocity = v_val,
        Gamma = Gamma,
        background = !isnothing(hsi_vec) ? :hsi : :polygons,
        title = "Advection Drift Field (v ≈ $(round(v_val, digits=3)))",
        arrow_scale = 1.2
    )
    save_html(m_arrows, joinpath(output_dir, "advection_arrows.html"))

    m_tracks = leaflet_tracks_map(
        paths, au;
        hsi = hsi_vec,
        background = :polygons,
        title = "Dispersal Trajectories (n=$(size(paths, 1)))",
        max_paths = 20
    )
    save_html(m_tracks, joinpath(output_dir, "movement_tracks.html"))

    m_kernel = leaflet_dispersal_kernel(
        Gamma, au;
        title = "Dispersal Distance Decay Curve"
    )
    save_html(m_kernel, joinpath(output_dir, "dispersal_kernel.html"))

    m_steps = leaflet_step_diagnostics(
        paths, au;
        title = "Trajectory Step Lengths & Turning Angles"
    )
    save_html(m_steps, joinpath(output_dir, "step_diagnostics.html"))

    m_conn = leaflet_regional_connectivity(
        C_reg;
        strata_names = strata_unique,
        title = "Regional Transfer Probability Matrix"
    )
    save_html(m_conn, joinpath(output_dir, "regional_connectivity.html"))

    m_dash = leaflet_movement_dashboard(
        result, paths;
        hsi = hsi_vec,
        strata = strata_vec,
        title = "BSTM Movement Diagnostics Dashboard"
    )
    save_html(m_dash, joinpath(output_dir, "movement_dashboard.html"))

    # 2. Optional Static Figures
    p_hsi = nothing
    p_diff = nothing
    p_res = nothing
    p_arrows = nothing
    p_tracks = nothing
    p_kernel = nothing
    p_steps = nothing
    p_conn = nothing
    p_dash = nothing

    if save_static || mode in (:plots, :all)
        if !isnothing(hsi_vec)
            p_hsi = plot_hsi_choropleth(hsi_vec, au; title="Habitat Suitability Index (HSI)")
            save_plot(p_hsi, joinpath(output_dir, "hsi_choropleth.$(fmt)"); fmt=fmt, dpi=dpi)
        end
        p_diff = plot_diffusion_map(d_val, au; title="Posterior Diffusion Field (D ≈ $(round(d_val, digits=3)))")
        save_plot(p_diff, joinpath(output_dir, "diffusion_map.$(fmt)"); fmt=fmt, dpi=dpi)

        p_res = plot_residence_time_map(Gamma, au; title="Stationary Residence Distribution (π)")
        save_plot(p_res, joinpath(output_dir, "residence_time_map.$(fmt)"); fmt=fmt, dpi=dpi)

        p_arrows = plot_advection_arrows(
            au;
            hsi = hsi_vec,
            velocity = v_val,
            Gamma = Gamma,
            background = !isnothing(hsi_vec) ? :hsi : :polygons,
            title = "Advection Drift Field (v ≈ $(round(v_val, digits=3)))",
            arrow_scale = 1.2
        )
        save_plot(p_arrows, joinpath(output_dir, "advection_arrows.$(fmt)"); fmt=fmt, dpi=dpi)

        p_tracks = plot_tracks_on_map(
            paths, au;
            hsi = hsi_vec,
            background = :polygons,
            title = "Dispersal Trajectories (n=$(size(paths, 1)))",
            max_paths = 20
        )
        save_plot(p_tracks, joinpath(output_dir, "movement_tracks.$(fmt)"); fmt=fmt, dpi=dpi)

        p_kernel = plot_dispersal_kernel(
            Gamma, au;
            title = "Dispersal Distance Decay Curve",
            fit_exponential = true
        )
        save_plot(p_kernel, joinpath(output_dir, "dispersal_kernel.$(fmt)"); fmt=fmt, dpi=dpi)

        p_steps = plot_step_length_distribution(
            paths, au;
            title = "Trajectory Step Lengths & Turning Angles"
        )
        save_plot(p_steps, joinpath(output_dir, "step_diagnostics.$(fmt)"); fmt=fmt, dpi=dpi)

        p_conn = plot_regional_connectivity_matrix(
            C_reg;
            strata_names = strata_unique,
            title = "Regional Transfer Probability Matrix"
        )
        save_plot(p_conn, joinpath(output_dir, "regional_connectivity.$(fmt)"); fmt=fmt, dpi=dpi)

        p_dash = plot_movement_dashboard(
            result, paths;
            hsi = hsi_vec,
            strata = strata_vec,
            title = "BSTM Movement Diagnostics Dashboard"
        )
        save_plot(p_dash, joinpath(output_dir, "movement_dashboard.$(fmt)"); fmt=fmt, dpi=dpi)
    end

    println("  Saved interactive Leaflet HTML maps and dashboard to $(output_dir)/")

    if mode == :plots
        return (
            hsi                   = p_hsi,
            diffusion             = p_diff,
            residence_time        = p_res,
            advection_arrows      = p_arrows,
            tracks                = p_tracks,
            dispersal_kernel      = p_kernel,
            step_diagnostics      = p_steps,
            regional_connectivity = p_conn,
            dashboard             = p_dash
        )
    else
        return (
            hsi                   = m_hsi,
            diffusion             = m_diff,
            residence_time        = m_res,
            advection_arrows      = m_arrows,
            tracks                = m_tracks,
            dispersal_kernel      = m_kernel,
            step_diagnostics      = m_steps,
            regional_connectivity = m_conn,
            dashboard             = m_dash,
            leaflet_dashboard     = m_dash
        )
    end
end

# ── Status check ──────────────────────────────────────────────────────────────

"""
    check_movement_status(db_path) -> nothing

Prints a short status summary of the DuckDB movement results table, including
the number of spatial units and the range of transition probabilities.
"""
function check_movement_status(db_path::AbstractString)
    isfile(db_path) || (println("  No results database found at $(db_path)."); return)
    try
        df = query_duckdb(db_path, "SELECT COUNT(*) AS n FROM movement_transition")
        n_rows = df[1, :n]
        S_est  = round(Int, sqrt(n_rows))
        println("\n  DuckDB: $(db_path)")
        println("  movement_transition: $(n_rows) rows  →  $(S_est) × $(S_est) kernel")
    catch e
        println("  Could not query $(db_path): $(e)")
    end
    return nothing
end

# ── Simulations ───────────────────────────────────────────────────────────────

"""
    simulate_movement_paths(result; n_paths=10, n_steps=20,
                            rho_persistence=0.0, rng=Random.GLOBAL_RNG)
        -> Matrix{Int}

Simulates individual dispersal paths from the posterior-mean kernel in
`result.transition_matrix`. Returns a `(n_paths × (n_steps+1))` matrix of
spatial unit indices.

# Arguments
- `result`: Output of `run_movement_simple`.
- `n_paths::Int`: Number of simulated individuals (default 10).
- `n_steps::Int`: Forward simulation steps (default 20).
- `rho_persistence::Real`: Directional persistence ρ ≥ 0 (default 0.0).
- `rng`: Random number generator.
"""
function simulate_movement_paths(
    result           :: NamedTuple;
    n_paths          :: Int                    = 10,
    n_steps          :: Int                    = 20,
    rho_persistence  :: Real                   = 0.0,
    rng              :: Random.AbstractRNG     = Random.GLOBAL_RNG
)::Matrix{Int}
    S           = size(result.transition_matrix, 1)
    start_units = rand(rng, 1:S, n_paths)
    return simulate_posterior_trajectories(
        result.transition_matrix, start_units, n_steps, result.au;
        rho_persistence = rho_persistence,
        rng             = rng
    )
end

# ── Mock-data simulation test ─────────────────────────────────────────────────

"""
    run_simulation_test(; seed=42, n_units=36, n_marks=100,
                          n_samples=200, n_warmup=100,
                          relationship=:exponential,
                          with_hsi=true,
                          output_dir=joinpath(@__DIR__, "output")) -> NamedTuple

End-to-end smoke-test using the same synthetic data generator as the unified
hierarchical workflow (`hierarchical_workflow.jl`).

# Data generation
- **Telemetry**: `bstm_data(type="telemetry", ...)` — 100 tagged individuals on a
  600 km domain; each individual has one release and one recapture event. Columns:
  `tagid`, `s_idx`, `time`, `tag`, `individual_covariate`. The spatial index is
  converted to `(lon, lat)` via the simulation bundle's spatial unit centroids.
- **HSI** (optional): a centred Gaussian blob peaked at the domain centre, derived
  analytically from the simulation centroids and rescaled to [0, 1]. This mimics
  the HSI surface that would be produced by the community tier of the hierarchical
  workflow without requiring a full fit.

# Workflow stages exercised
1. Telemetry validation and lon/lat extraction.
2. `build_tessellation` — spatial CVT tessellation.
3. `compute_prior_kernel` — deterministic HSI-weighted kernel.
4. `run_movement_simple` — full NUTS fit (reduced samples for speed).
5. `simulate_movement_paths` — CRW dispersal with persistence.
6. `calculate_regional_connectivity` — East / West regional aggregation.
7. `save_movement_bundle` — DuckDB + JLD2 output.

# Arguments
- `seed::Int`: Random seed (default 42, matching the hierarchical workflow).
- `n_units::Int`: Target spatial units (default 36).
- `n_marks::Int`: Number of tagged individuals (default 100).
- `n_samples::Int`: NUTS samples (default 200 for speed).
- `n_warmup::Int`: NUTS warmup steps (default 100).
- `relationship::Symbol`: HSI advection functional form (default `:exponential`).
- `with_hsi::Bool`: If `true` (default), a simulated HSI surface is provided.
- `output_dir::AbstractString`: Directory for output files.

# Returns
`NamedTuple` with fields:
- `result`: Full output of `run_movement_simple`.
- `paths::Matrix{Int}`: Simulated CRW paths `(n_paths × n_steps+1)`.
- `C_regional::Matrix{Float64}`: 2×2 East/West regional connectivity matrix.
- `kernel_prior`: Deterministic pre-MCMC kernel.
"""
function run_simulation_test(;
    seed         :: Int                                   = 42,
    n_units      :: Int                                   = 36,
    n_marks      :: Int                                   = 100,
    n_samples    :: Int                                   = 200,
    n_warmup     :: Int                                   = 100,
    area_method  :: Symbol                                = :hexagonal,
    reshard_hsi  :: Bool                                  = true,
    relationship :: Symbol                                = :exponential,
    with_hsi     :: Bool                                  = true,
    output_dir   :: AbstractString                        = joinpath(@__DIR__, "output"),
    save_plots   :: Bool                                  = true,
    plot_dir     :: Union{Nothing, AbstractString}        = nothing
)::NamedTuple
    rng = MersenneTwister(seed)
    mkpath(output_dir)

    println("\n========================================================")
    println("  BSTM Movement Simulation Test")
    println("  (matching hierarchical_workflow.jl mock data)")
    println("========================================================")

    # ── 1. Generate synthetic telemetry ──────────────────────────────────────
    domain_km = 600.0
    println("  Generating telemetry data (n_marks=$(n_marks), domain=$(domain_km) km) …")
    sim_bundle = bstm_data(
        type        = "telemetry",
        domain_size = domain_km,
        n_units     = n_units,
        n_years     = 5,
        n_marks     = n_marks,
        area_method = area_method,
        seed        = seed
    )

    df_tel_raw  = sim_bundle.telemetry_data
    au_sim      = sim_bundle.au
    cent_lon    = [c[1] for c in au_sim.centroids]
    cent_lat    = [c[2] for c in au_sim.centroids]

    tel_df = DataFrame(
        tagid                = df_tel_raw.tagid,
        lon                  = cent_lon[df_tel_raw.s_idx],
        lat                  = cent_lat[df_tel_raw.s_idx],
        time                 = df_tel_raw.time,
        tag                  = df_tel_raw.tag,
        individual_covariate = df_tel_raw.individual_covariate
    )

    # Build the destination tessellation (defaults to :hexagonal)
    opts_notess = MovementOptions(
        n_units          = n_units,
        area_method      = area_method,
        hsi              = nothing,
        relationship     = relationship,
        sensitivity      = 1.2,
        diffusion_weight = 0.3,
        n_samples        = n_samples,
        n_warmup         = n_warmup,
        rng              = rng
    )
    au = build_tessellation(tel_df; opts=opts_notess)
    S  = length(au.centroids)
    println("  Telemetry: $(nrow(tel_df)) rows, $(length(unique(tel_df.tagid))) individuals")

    # Simulate HSI surface (optionally on a source grid to test resharding)
    hsi_vec = if with_hsi
        cx = domain_km / 2.0
        cy = domain_km / 2.0
        r  = domain_km / 3.0
        if reshard_hsi
            # Generate on a 50-unit source grid to demonstrate geometric resharding
            src_n = 49
            side = 7
            gx = range(0.0, domain_km, length=side)
            gy = range(0.0, domain_km, length=side)
            raw_src = [exp(-((x - cx)^2 + (y - cy)^2) / r^2) for x in gx for y in gy]
            mn, mx = minimum(raw_src), maximum(raw_src)
            (raw_src .- mn) ./ (mx - mn + 1e-9) .* 0.9 .+ 0.05
        else
            raw = [exp(-((au.centroids[i][1] - cx)^2 + (au.centroids[i][2] - cy)^2) / r^2)
                   for i in 1:S]
            mn, mx = minimum(raw), maximum(raw)
            (raw .- mn) ./ (mx - mn + 1e-9) .* 0.9 .+ 0.05
        end
    else
        nothing
    end

    hsi_label = with_hsi ? (reshard_hsi ? "Gaussian blob (resharded from 49-cell grid → :$(area_method))" : "Gaussian blob (S=$(S))") : "isotropic (no HSI)"
    println("  HSI: $(hsi_label)")

    # Rebuild opts with the HSI vector and resharding flag
    opts = MovementOptions(
        n_units          = S,
        area_method      = area_method,
        hsi              = hsi_vec,
        reshard_hsi      = reshard_hsi,
        hsi_area_method  = :grid,
        relationship     = relationship,
        sensitivity      = 1.2,
        diffusion_weight = 0.3,
        n_samples        = n_samples,
        n_warmup         = n_warmup,
        rng              = rng
    )
    kernel_det = compute_prior_kernel(au; opts=opts)

    # Fit the movement model.
    println("  Running NUTS fit ($(n_samples) samples, $(n_warmup) warmup) ...")
    result = run_movement_simple(tel_df; opts=opts, au=au)

    # ── 5. Simulate CRW paths ─────────────────────────────────────────────────
    S_res       = size(result.transition_matrix, 1)
    paths = simulate_movement_paths(result;
        n_paths         = 12,
        n_steps         = 20,
        rho_persistence = 1.2,
        rng             = rng
    )
    println("  Simulated $(size(paths,1)) CRW paths × $(size(paths,2)-1) steps.")

    # ── 6. Regional connectivity ──────────────────────────────────────────────
    cents_lon   = [c[1] for c in result.au.centroids]
    mid_lon     = (minimum(cents_lon) + maximum(cents_lon)) / 2.0
    strata      = [x > mid_lon ? "East" : "West" for x in cents_lon]
    C_regional  = calculate_regional_connectivity(result.transition_matrix, strata)

    println("\n  Regional connectivity (East / West):")
    @printf("    West → West: %.3f   West → East: %.3f\n", C_regional[2,2], C_regional[2,1])
    @printf("    East → West: %.3f   East → East: %.3f\n", C_regional[1,2], C_regional[1,1])

    # ── 7. Save outputs ───────────────────────────────────────────────────────
    db_path   = joinpath(output_dir, "movement_test.duckdb")
    jld2_path = joinpath(output_dir, "movement_test.jld2")
    save_movement_bundle(result; db_path=db_path, jld2_path=jld2_path)

    # ── 8. Generate diagnostic visualizations ─────────────────────────────────
    plots_bundle = nothing
    if save_plots
        p_dir = !isnothing(plot_dir) ? plot_dir : joinpath(output_dir, "plots")
        plots_bundle = generate_movement_plots(
            result, paths;
            output_dir = p_dir,
            hsi        = result.hsi,
            strata     = strata
        )
    end

    println("\n  Outputs written to: $(output_dir)")
    println("========================================================\n")

    return (
        result       = result,
        paths        = paths,
        C_regional   = C_regional,
        kernel_prior = kernel_det,
        plots        = plots_bundle
    )
end

# ── CLI entry-point ───────────────────────────────────────────────────────────

"""
    main(cli_args=ARGS) -> Any

Command-line interface. Parses arguments and dispatches to the appropriate
workflow step.

# CLI usage

    julia --project=. docs/movement/movement_simple.jl --help

    # Inspect DuckDB status without running MCMC
    julia --project=. docs/movement/movement_simple.jl --status

    # Run full fit
    julia --project=. docs/movement/movement_simple.jl \\
        --telemetry=data/telemetry.csv \\
        --n-units=40 --samples=500 --warmup=200

    # Run fit with HSI from a CSV (one column: hsi)
    julia --project=. docs/movement/movement_simple.jl \\
        --telemetry=data/telemetry.csv \\
        --hsi=data/hsi.csv

    # Simulate paths from existing results
    julia --project=. docs/movement/movement_simple.jl \\
        --simulate --n-paths=20 --n-steps=30

    # Run self-contained test
    julia --project=. docs/movement/movement_simple.jl --test
"""
function main(cli_args::Vector{String}=ARGS)
    args = Dict{String, Any}(
        "telemetry"      => "",
        "hsi"            => "",
        "area-method"    => "hexagonal",
        "reshard-hsi"    => false,
        "hsi-geometry"   => "grid",
        "hsi-coords"     => "",
        "n-units"        => 40,
        "samples"        => 500,
        "warmup"         => 200,
        "chains"         => 1,
        "relationship"   => "exponential",
        "db"             => "movement_results.duckdb",
        "jld2"           => "movement_model.jld2",
        "plot"           => false,
        "plot-dir"       => "",
        "status"         => false,
        "simulate"       => false,
        "test"           => false,
        "n-paths"        => 10,
        "n-steps"        => 20,
        "seed"           => 42
    )

    for arg in cli_args
        if arg in ["--help", "-h"]
            println("Usage: julia --project=. docs/movement/movement_simple.jl [OPTIONS]")
            println("  --telemetry=FILE     Path to telemetry CSV (tagid, lon, lat, time, tag)")
            println("  --hsi=FILE           Path to CSV of HSI values (single column or with lon,lat)")
            println("  --area-method=STR    Spatial tessellation geometry: hexagonal (default), cvt, voronoi, grid")
            println("  --reshard-hsi        Reshard source HSI onto target destination geometry")
            println("  --hsi-geometry=STR   Source geometry for HSI resharding: grid (default), cvt, hexagonal")
            println("  --hsi-coords=FILE    Optional CSV with lon,lat centroids for source HSI")
            println("  --n-units=INT        Target number of spatial units (default: 40)")
            println("  --samples=INT        NUTS samples per chain (default: 500)")
            println("  --warmup=INT         NUTS warmup steps (default: 200)")
            println("  --chains=INT         Number of MCMC chains (default: 1)")
            println("  --relationship=STR   HSI functional form: exponential, linear, logistic")
            println("  --db=FILE            Output DuckDB path (default: movement_results.duckdb)")
            println("  --jld2=FILE          Output JLD2 path (default: movement_model.jld2)")
            println("  --plot               Generate and save full diagnostic plot suite")
            println("  --plot-dir=DIR       Directory to save plots (default: plots/)")
            println("  --status             Print DuckDB status and exit")
            println("  --simulate           Load existing JLD2 results and simulate paths")
            println("  --test               Run self-contained simulation test")
            println("  --n-paths=INT        Number of simulated individuals (default: 10)")
            println("  --n-steps=INT        Forward simulation steps (default: 20)")
            println("  --seed=INT           Random seed (default: 42)")
            return nothing
        elseif arg == "--status"
            args["status"] = true
        elseif arg == "--simulate"
            args["simulate"] = true
        elseif arg == "--test"
            args["test"] = true
        elseif arg == "--plot"
            args["plot"] = true
        elseif arg == "--reshard-hsi"
            args["reshard-hsi"] = true
        elseif startswith(arg, "--plot-dir=")
            args["plot-dir"] = Base.split(arg, "="; limit=2)[2]
            args["plot"] = true
        elseif startswith(arg, "--area-method=")
            args["area-method"] = Base.split(arg, "="; limit=2)[2]
        elseif startswith(arg, "--hsi-geometry=")
            args["hsi-geometry"] = Base.split(arg, "="; limit=2)[2]
        elseif startswith(arg, "--hsi-coords=")
            args["hsi-coords"] = Base.split(arg, "="; limit=2)[2]
            args["reshard-hsi"] = true
        elseif startswith(arg, "--telemetry=")
            args["telemetry"] = Base.split(arg, "="; limit=2)[2]
        elseif startswith(arg, "--hsi=")
            args["hsi"] = Base.split(arg, "="; limit=2)[2]
        elseif startswith(arg, "--n-units=")
            args["n-units"] = parse(Int, Base.split(arg, "="; limit=2)[2])
        elseif startswith(arg, "--samples=")
            args["samples"] = parse(Int, Base.split(arg, "="; limit=2)[2])
        elseif startswith(arg, "--warmup=")
            args["warmup"] = parse(Int, Base.split(arg, "="; limit=2)[2])
        elseif startswith(arg, "--chains=")
            args["chains"] = parse(Int, Base.split(arg, "="; limit=2)[2])
        elseif startswith(arg, "--relationship=")
            args["relationship"] = Base.split(arg, "="; limit=2)[2]
        elseif startswith(arg, "--db=")
            args["db"] = Base.split(arg, "="; limit=2)[2]
        elseif startswith(arg, "--jld2=")
            args["jld2"] = Base.split(arg, "="; limit=2)[2]
        elseif startswith(arg, "--n-paths=")
            args["n-paths"] = parse(Int, Base.split(arg, "="; limit=2)[2])
        elseif startswith(arg, "--n-steps=")
            args["n-steps"] = parse(Int, Base.split(arg, "="; limit=2)[2])
        elseif startswith(arg, "--seed=")
            args["seed"] = parse(Int, Base.split(arg, "="; limit=2)[2])
        end
    end

    rng = MersenneTwister(args["seed"])

    # Status only
    if args["status"]
        check_movement_status(args["db"])
        return nothing
    end

    # Test mode or fallback when no telemetry CSV specified
    if args["test"] || (isempty(args["telemetry"]) && !args["simulate"])
        if !args["test"]
            println("  Note: No --telemetry path provided; defaulting to --test simulation mode.")
        end
        p_dir = isempty(args["plot-dir"]) ? joinpath(@__DIR__, "output", "plots") : args["plot-dir"]
        return run_simulation_test(;
            seed         = args["seed"],
            n_units      = args["n-units"],
            n_samples    = min(args["samples"], 200),
            n_warmup     = min(args["warmup"], 100),
            area_method  = Symbol(args["area-method"]),
            reshard_hsi  = args["reshard-hsi"] || true,
            relationship = Symbol(args["relationship"]),
            save_plots   = args["plot"] || args["test"],
            plot_dir     = p_dir
        )
    end

    # Simulate from existing results
    if args["simulate"]
        isfile(args["jld2"]) || error("JLD2 file not found: $(args["jld2"])")
        saved  = JLD2.load(args["jld2"])
        au_saved = haskey(saved, "au") ? saved["au"] : nothing
        result = (
            transition_matrix = saved["transition_matrix"],
            au                = au_saved,
            opts              = haskey(saved, "opts") ? saved["opts"] : MovementOptions()
        )
        paths = simulate_movement_paths(result;
            n_paths=args["n-paths"], n_steps=args["n-steps"], rng=rng)
        println("Simulated $(size(paths,1)) paths × $(size(paths,2)-1) steps.")

        if args["plot"]
            p_dir = isempty(args["plot-dir"]) ? "plots" : args["plot-dir"]
            generate_movement_plots(result, paths; output_dir=p_dir)
        end
        return paths
    end

    # Full fit from telemetry CSV via DuckDB auto CSV reader
    db_con = DuckDB.DB()
    con = DuckDB.connect(db_con)
    local tel_df, hsi_vec, hsi_coords
    try
        tel_df = DuckDB.query(con, "SELECT * FROM read_csv_auto('$(escape_string(args["telemetry"]))')") |> DataFrame

        hsi_vec = nothing
        hsi_coords = nothing
        if !isempty(args["hsi"])
            hsi_raw = DuckDB.query(con, "SELECT * FROM read_csv_auto('$(escape_string(args["hsi"]))')") |> DataFrame
            if hasproperty(hsi_raw, :hsi)
                hsi_vec = convert(Vector{Float64}, hsi_raw.hsi)
                if hasproperty(hsi_raw, :lon) && hasproperty(hsi_raw, :lat)
                    hsi_coords = Tuple{Float64, Float64}[(row.lon, row.lat) for row in eachrow(hsi_raw)]
                end
            else
                hsi_vec = convert(Vector{Float64}, hsi_raw[:, 1])
            end
        end

        if !isempty(args["hsi-coords"])
            coords_raw = DuckDB.query(con, "SELECT * FROM read_csv_auto('$(escape_string(args["hsi-coords"]))')") |> DataFrame
            hsi_coords = Tuple{Float64, Float64}[(row.lon, row.lat) for row in eachrow(coords_raw)]
        end
    finally
        DuckDB.disconnect(con)
        DuckDB.close(db_con)
    end

    opts = MovementOptions(
        n_units          = args["n-units"],
        area_method      = Symbol(args["area-method"]),
        hsi              = hsi_vec,
        reshard_hsi      = args["reshard-hsi"],
        hsi_coords       = hsi_coords,
        hsi_area_method  = Symbol(args["hsi-geometry"]),
        relationship     = Symbol(args["relationship"]),
        sensitivity      = 1.0,
        diffusion_weight = 0.1,
        method           = :explicit,
        n_samples        = args["samples"],
        n_warmup         = args["warmup"],
        n_chains         = args["chains"],
        rng              = rng
    )

    result = run_movement_simple(tel_df; opts=opts)
    save_movement_bundle(result; db_path=args["db"], jld2_path=args["jld2"])

    if args["plot"]
        p_dir = isempty(args["plot-dir"]) ? "plots" : args["plot-dir"]
        paths_sim = simulate_movement_paths(result; n_paths=args["n-paths"], n_steps=args["n-steps"], rng=rng)
        generate_movement_plots(result, paths_sim; output_dir=p_dir, hsi=hsi_vec)
    end

    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
