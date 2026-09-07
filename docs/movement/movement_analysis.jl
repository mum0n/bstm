# Auto-discover and activate the bstm project environment if not already loaded
if Base.find_package("bstm") === nothing
    import Pkg
    let
        curr_dir = @__DIR__
        repo_root = nothing
        for _ in 1:6
            proj = joinpath(curr_dir, "Project.toml")
            if isfile(proj)
                txt = read(proj, String)
                if occursin("name = \"bstm\"", txt) ||
                   occursin("name = \"BSTM\"", txt)
                    repo_root = curr_dir
                    break
                end
            end
            parent = dirname(curr_dir)
            parent == curr_dir && break
            curr_dir = parent
        end
        if repo_root !== nothing
            Pkg.activate(repo_root)
        else
            Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))
        end
    end
end

"""
    movement_analysis.jl

Modular end-to-end workflow for Bayesian Spatio-Temporal Movement (BSTM)
analysis. Consolidates `movement_individual.jl` and `snowcrab_movement.jl`
into a single parameter-driven script with clearly separated phases.

## Pipeline Phases (see `run_movement_analysis`)

1. **Data Ingestion** (`load_movement_data`):
   Load or simulate telemetry data, optionally reshard to a finer hexagonal
   mesh via LibGEOS, and enforce depth-range traversal barriers.

2. **Model Fitting** (`fit_movement_models`):
   Fit a convex stochastic transition kernel via `@bstm()`, choosing between
   a pure-telemetry Categorical likelihood or a joint NegBin survey density
   and telemetry model.

3. **Kernel Construction** (`extract_transition_kernels`):
   Extract posterior mean parameters (alpha, rho, gamma) per biological group
   and build group-stratified stochastic transition kernels:
       P_g = (1 - rho_g)[(1 - alpha_g) T_diff + alpha_g A_g(eta)] + rho_g I

4. **Path Reconstruction** (`reconstruct_paths_and_diagnostics`):
   Viterbi / A* trajectories, Markov bridge corridor heatmaps, optional
   stochastic least-cost path ensembles, and domain-wide bottleneck index:
       B(u) = C_domain(u) / max(1, deg_marine(u))

5. **Advanced Diagnostics** (`compute_advanced_diagnostics`):
   Electrical circuit current density, posterior circuit inference under HSI
   observation uncertainty, and multi-scale Chebyshev Spectral Graph Wavelet
   (SGWT) decomposition with BayesShrink denoising.

6. **Dashboard Export** (`export_dashboards`):
   Interactive Leaflet HTML maps for paths, corridors, current density,
   bottlenecks, and wavelet decompositions.

## Parameter Functions

- `movement_parameters_default()`: Generic defaults for any species.
- `movement_parameters_snowcrab()`: Snow crab-specific overrides for the
  Scotian Shelf including three demographic biological groups.

## Usage

    # Generic simulation
    run_movement_analysis()

    # Snow crab with overrides
    p = merge(movement_parameters_snowcrab(), (n_samples = 500,))
    run_movement_analysis(p)

    # CLI
    julia --project=. docs/movement/movement_analysis.jl --help
"""

using DataFrames
using Dates
using Distributions
using LinearAlgebra
using Random
using SparseArrays
using Statistics
using Turing
using bstm

# Load snow crab data loader (defines snowcrab_movement_data)
include(joinpath(@__DIR__, "snowcrab_movement_data.jl"))

# Type alias for flexible depth range specifications
const _DepthRangeArg = Union{
    Nothing,
    Tuple{<:Real, <:Real},
    AbstractVector{<:Real},
    AbstractString,
}

# =============================================================================
# Parameter Functions
# =============================================================================

"""
    movement_parameters_default() -> NamedTuple

Returns the default analysis configuration for a generic species.
All fields can be individually overridden by merging with a
species-specific parameter function:
    params = merge(movement_parameters_default(), (max_paths = 50,))

# Returns
`NamedTuple` with:
- `data_source::Symbol`: `:simulate` or `:snowcrab`.
- `model_mode::String`: `"telemetry"`, `"telemetry_and_survey"`, or `"both"`.
- `reshard_hex::Bool`: Reshard domain to finer hexagonal lattice.
- `hex_radius_km::Real`: Cell radius for fine hexagonal lattice (km).
- `use_hydrodynamics::Bool`: Ingest 3D hydrodynamic diagnostics.
- `depth_range`: `(min_d, max_d)` traversal barrier in m, or `nothing`.
- `max_paths::Int`: Maximum individual trajectories to reconstruct.
- `path_method::Symbol`: `:astar` (default) or `:viterbi`.
- `smooth_paths::Bool`: Apply marine line-of-sight raycast smoothing.
- `compute_circuit::Bool`: Compute electrical circuit current density.
- `compute_stochastic::Bool`: Compute stochastic least-cost path ensembles.
- `compute_bottlenecks::Bool`: Compute domain-wide bottleneck index B(u).
- `compute_wavelets::Bool`: Compute Chebyshev spectral graph wavelets.
- `n_stochastic_draws::Int`: Monte Carlo draws per stochastic path.
- `hsi_se::Real`: Observation standard error on HSI (sigma).
- `n_samples::Int`: MCMC posterior draw count.
- `n_warmup::Int`: MCMC warmup iteration count.
- `seed::Int`: Random seed for reproducibility.
- `render_html::Bool`: Export interactive HTML dashboards.
- `output_dir::String`: Output directory for all artifacts.
- `verbose::Bool`: Enable progress logging.
- `group_labels::Vector{String}`: Biological group display names.
- `group_alpha::Vector{Float64}`: Per-group advection weight alpha_g.
- `group_rho::Vector{Float64}`: Per-group site fidelity rho_g.
- `group_gamma::Vector{Float64}`: Per-group HSI gradient strength gamma_g.
- `species_name::String`: Display name used in dashboard titles.
"""
function movement_parameters_default()
    return (
        data_source         = :simulate,
        model_mode          = "telemetry",
        reshard_hex         = false,
        hex_radius_km       = 10.0,
        use_hydrodynamics   = false,
        depth_range         = nothing,
        max_paths           = 25,
        path_method         = :astar,
        smooth_paths        = false,
        compute_circuit     = false,
        compute_stochastic  = false,
        compute_bottlenecks = false,
        compute_wavelets    = false,
        n_stochastic_draws  = 10,
        hsi_se              = 0.08,
        n_samples           = 200,
        n_warmup            = 100,
        seed                = 42,
        render_html         = true,
        output_dir          = normpath(joinpath(@__DIR__, "..", "..", "output")),
        verbose             = true,
        group_labels        = String["All"],
        group_alpha         = Float64[0.40],
        group_rho           = Float64[0.25],
        group_gamma         = Float64[1.00],
        species_name        = "Animal",
    )
end

"""
    movement_parameters_snowcrab() -> NamedTuple

Returns the analysis configuration for Scotian Shelf snow crab
(*Chionoecetes opilio*), overriding generic defaults with species-specific
biology and real empirical data settings.

Three demographic groups drive distinct movement strategies:

    Group         | alpha (advection) | rho (fidelity) | gamma (HSI)
    Mature Female |       0.25        |      0.60      |    0.80
    Mature Male   |       0.65        |      0.15      |    1.50
    Immature      |       0.30        |      0.35      |    0.50

Mature females exhibit high site fidelity (rho = 0.60) during multi-year
egg brooding. Mature males are most dispersive (alpha = 0.65, rho = 0.15)
during active mating migrations. Immatures show diffusive benthic
exploration.

Species-specific overrides applied over `movement_parameters_default`:
- `data_source`: `:snowcrab` (real empirical mark-recapture data).
- `depth_range`: `(50.0, 500.0)` m -- principal benthic habitat zone.
- All four optional diagnostics are enabled.
- `n_stochastic_draws`: 15 -- balanced resolution for real data.
"""
function movement_parameters_snowcrab()
    defaults = movement_parameters_default()
    return merge(defaults, (
        data_source         = :snowcrab,
        model_mode          = "telemetry",
        depth_range         = (50.0, 500.0),
        reshard_hex         = true,
        hex_radius_km       = 5.0,
        use_hydrodynamics   = true,
        max_paths           = 25,
        path_method         = :astar,
        smooth_paths        = false,
        compute_circuit     = false,
        compute_stochastic  = false,
        compute_bottlenecks = false,
        compute_wavelets    = false,
        n_stochastic_draws  = 15,
        hsi_se              = 0.08,
        n_samples           = 100,
        n_warmup            = 50,
        seed                = 42,
        render_html         = true,
        output_dir          = normpath(joinpath(@__DIR__, "..", "..", "output")),
        group_labels        = String["Mature Female", "Mature Male", "Immature"],
        group_alpha         = Float64[0.25, 0.65, 0.30],
        group_rho           = Float64[0.60, 0.15, 0.35],
        group_gamma         = Float64[0.80, 1.50, 0.50],
        species_name        = "Snow Crab",
    ))
end


 

# =============================================================================
# Private Helpers
# =============================================================================

"""
    _parse_depth_range(depth_range) -> Union{Tuple{Float64,Float64}, Nothing}

Parses a depth range specification into a `(min, max)` float tuple, or
`nothing` when no depth constraint is desired. Accepts `Tuple`,
`AbstractVector`, or a comma/colon-separated `AbstractString` (e.g.
`"50,500"` or `"50:500"`).
"""
function _parse_depth_range(
    depth_range::_DepthRangeArg
)::Union{Tuple{Float64,Float64}, Nothing}
    depth_range === nothing && return nothing
    if depth_range isa AbstractString
        sep   = occursin(":", depth_range) ? ":" : ","
        parts = Base.split(depth_range, sep)
        length(parts) == 2 || throw(ArgumentError(
            "Invalid depth_range \"$depth_range\". Expected \"min,max\"."
        ))
        return (parse(Float64, strip(parts[1])), parse(Float64, strip(parts[2])))
    elseif length(depth_range) >= 2
        return (Float64(depth_range[1]), Float64(depth_range[2]))
    end
    return nothing
end

"""
    _extract_domain_depths(mesh, data, resharded_depths) -> Vector{Float64}

Extracts or synthesizes bathymetric depth values (m) for all mesh units.
Prioritizes: resharded fine hexagonal depths > loaded dataset depths >
mesh depth vector > synthetic bathymetric contours.
"""
function _extract_domain_depths(mesh, data, resharded_depths)::Vector{Float64}
    n = mesh.n_units
    if !isnothing(resharded_depths) && length(resharded_depths) == n
        return Float64.(resharded_depths)
    elseif hasproperty(data, :depth_vec) &&
           !isnothing(data.depth_vec) &&
           length(data.depth_vec) == n
        return Float64.(data.depth_vec)
    elseif hasproperty(mesh, :depth_vec) &&
           !isnothing(mesh.depth_vec) &&
           length(mesh.depth_vec) == n
        return Float64.(mesh.depth_vec)
    elseif hasproperty(mesh, :centroids_km) &&
           !isnothing(mesh.centroids_km) &&
           length(mesh.centroids_km) == n
        return [
            150.0 + 60.0 * sin(mesh.centroids_km[s][1] / 40.0) +
            40.0 * cos(mesh.centroids_km[s][2] / 40.0)
            for s in 1:n
        ]
    elseif hasproperty(mesh, :centroids_lonlat) &&
           !isnothing(mesh.centroids_lonlat) &&
           length(mesh.centroids_lonlat) == n
        return [
            150.0 + 60.0 * sin(
                (mesh.centroids_lonlat[s][1] + 63.0) * 5.0
            ) + 40.0 * cos(
                (mesh.centroids_lonlat[s][2] - 44.0) * 5.0
            )
            for s in 1:n
        ]
    else
        return fill(150.0, n)
    end
end

"""
    _is_graph_reachable(W, start_node, end_node, max_k) -> Bool

Breadth-first search (BFS) on adjacency matrix `W` verifying that
`end_node` is reachable from `start_node` within `max_k` hops.
Returns `false` if either endpoint is isolated by a barrier.
"""
function _is_graph_reachable(
    W          ::AbstractMatrix{<:Real},
    start_node ::Int,
    end_node   ::Int,
    max_k      ::Int
)::Bool
    start_node == end_node && return true
    max_k <= 0             && return false
    S = size(W, 1)
    (!(1 <= start_node <= S) || !(1 <= end_node <= S)) && return false

    W_sp     = W isa SparseMatrixCSC ? W : sparse(W)
    visited  = falses(S)
    frontier = Int[start_node]
    visited[start_node] = true

    for _ in 1:max_k
        next_frontier = Int[]
        for u in frontier
            for ptr in W_sp.colptr[u]:(W_sp.colptr[u + 1] - 1)
                v = W_sp.rowval[ptr]
                v == end_node && return true
                if !visited[v]
                    visited[v] = true
                    push!(next_frontier, v)
                end
            end
        end
        frontier = next_frontier
        isempty(frontier) && break
    end
    return visited[end_node]
end

"""
    _resolve_centroids(mesh, n_spatial)
        -> (cents_planar, cents_lonlat, cents_mesh)

Resolves planar (km) and geographic (lon/lat) centroid vectors from a
mesh object, returning `(planar, lonlat, preferred_for_mapping)`.
"""
function _resolve_centroids(mesh, n_spatial)
    cents_km = hasproperty(mesh, :centroids_km) &&
               !isnothing(mesh.centroids_km) &&
               length(mesh.centroids_km) == n_spatial
    cents_ll = hasproperty(mesh, :centroids_lonlat) &&
               !isnothing(mesh.centroids_lonlat) &&
               length(mesh.centroids_lonlat) == n_spatial
    cents_c  = hasproperty(mesh, :centroids) && !isnothing(mesh.centroids)

    cents_planar = cents_km ? mesh.centroids_km :
                   cents_ll ? mesh.centroids_lonlat :
                   cents_c  ? mesh.centroids : nothing
    cents_lonlat = cents_ll ? mesh.centroids_lonlat :
                   cents_c  ? mesh.centroids : nothing
    cents_mesh   = cents_lonlat !== nothing ? cents_lonlat : cents_planar
    return (cents_planar, cents_lonlat, cents_mesh)
end

# =============================================================================
# Phase 1: Data Ingestion
# =============================================================================

"""
    load_movement_data(params) -> NamedTuple

Phase 1 of the pipeline. Loads or simulates a spatial telemetry dataset,
optionally reshards the domain to a finer regular hexagonal lattice using
LibGEOS, ingests 3D hydrodynamic bathymetry, and enforces depth-range
traversal barriers.

# Arguments
- `params`: Configuration NamedTuple from `movement_parameters_*`.
  Relevant keys: `data_source`, `reshard_hex`, `hex_radius_km`,
  `use_hydrodynamics`, `depth_range`, `seed`, `verbose`.

# Returns
`NamedTuple` with fields: `data`, `mesh`, `W`, `hsi_vec`, `obs_df`,
`survey_df`, `group_map`, `land_mask`, `n_spatial`, `resharded_hydro`,
`resharded_depths`, `parsed_depth_range`.
"""
function load_movement_data(params)::NamedTuple
    verbose = params.verbose

    verbose && println("=" ^ 72)
    verbose && println("  BSTM Movement Analysis Pipeline")
    verbose && println("=" ^ 72)

    # -- 1a. Load or simulate dataset ----------------------------------------
    verbose && println(
        "\n[Phase 1] Ingesting dataset (source: :$(params.data_source))..."
    )
    data = if params.data_source == :snowcrab
        sc_rad = params.hex_radius_km != 10.0 ? params.hex_radius_km : 15.0
        try
            snowcrab_movement_data(radius_km = sc_rad, verbose = verbose)
        catch err
            @warn "Could not load snow crab data: $err -- using simulate."
            bstm_data("movement")
        end
    else
        bstm_data("movement")
    end

    mesh      = data.mesh
    W         = data.W
    hsi_vec   = data.hsi_vec
    obs_df    = data.obs
    survey_df = hasproperty(data, :survey_df) ? data.survey_df : nothing
    group_map = hasproperty(data, :group_lookup) ?
                data.group_lookup : Dict(1 => "All")
    land_mask = hasproperty(data, :land_mask) ? data.land_mask : nothing
    n_spatial = mesh.n_units

    if verbose
        println("  Spatial mesh units : $n_spatial")
        if land_mask !== nothing
            println("    Marine    : $(count(!, land_mask))")
            println("    Land      : $(sum(land_mask))")
        end
        println("  Mark-recapture obs : $(nrow(obs_df))")
        println("  Biological groups  : $(length(group_map))")
    end

    # -- 1b. LibGEOS Hexagonal Resharding & Hydrodynamics --------------------
    resharded_hydro  = nothing
    resharded_depths = nothing

    if params.reshard_hex || params.use_hydrodynamics
        verbose && println(
            "\n[Phase 1b] Resharding to fine hexagons via LibGEOS..."
        )
        cents_lon = [Float64(c[1]) for c in mesh.centroids_lonlat]
        cents_lat = [Float64(c[2]) for c in mesh.centroids_lonlat]
        min_lon, max_lon = extrema(cents_lon)
        min_lat, max_lat = extrema(cents_lat)

        bathy = load_open_bathymetry(;
            lon_range      = (min_lon - 0.2, max_lon + 0.2),
            lat_range      = (min_lat - 0.2, max_lat + 0.2),
            resolution_deg = 0.08
        )
        hydro = extract_hydrodynamic_dataset(bathy;
            depth_levels = [0.0, 25.0, 50.0, 100.0, 175.0],
            times        = [1.0]
        )
        fine_mesh = build_hex_mesh_planar(
            bathy.grid_lon, bathy.grid_lat;
            radius_km = Float64(params.hex_radius_km)
        )
        verbose && println(
            "  Fine hexagonal units: $(fine_mesh.n_units) " *
            "(radius = $(params.hex_radius_km) km)"
        )

        P_transfer       = compute_network_transfer_matrix(
            bathy.au, fine_mesh; method = :area_weighted
        )
        resharded_hydro  = reshard_spatial_field(P_transfer, hydro)
        resharded_depths = reshard_spatial_field(P_transfer, bathy.depths)

        mesh      = fine_mesh
        W         = fine_mesh.W
        land_mask = identify_land_units(
            fine_mesh.centroids_lonlat; depth = resharded_depths
        )
        W, hsi_vec = apply_land_barrier(
            fine_mesh.W, resharded_hydro.hsi, land_mask
        )
        sever_land_crossing_edges!(W, fine_mesh.centroids_lonlat)
        n_spatial  = fine_mesh.n_units

        # Remap mark-recapture observations to active marine units
        obs_df       = copy(obs_df)
        fine_cents   = fine_mesh.centroids_lonlat
        marine_mask  = .!land_mask .& (vec(sum(W; dims = 2)) .> 0)
        marine_units = let mu = findall(marine_mask)
            isempty(mu) ? collect(1:n_spatial) : mu
        end
        marine_cents = fine_cents[marine_units]

        orig_rel = [data.mesh.centroids_lonlat[r] for r in obs_df.release]
        orig_rec = [data.mesh.centroids_lonlat[r] for r in obs_df.recapture]
        rel_sub  = map_to_units(
            [c[1] for c in orig_rel], [c[2] for c in orig_rel], marine_cents
        )
        rec_sub  = map_to_units(
            [c[1] for c in orig_rec], [c[2] for c in orig_rec], marine_cents
        )
        obs_df.release   = marine_units[rel_sub]
        obs_df.recapture = marine_units[rec_sub]

        if !isnothing(survey_df) && hasproperty(survey_df, :s_idx)
            survey_df       = copy(survey_df)
            orig_surv       = [data.mesh.centroids_lonlat[s]
                                for s in survey_df.s_idx]
            surv_sub        = map_to_units(
                [c[1] for c in orig_surv],
                [c[2] for c in orig_surv],
                marine_cents
            )
            survey_df.s_idx = marine_units[surv_sub]
            survey_df.depth = resharded_depths[survey_df.s_idx]
        end
        verbose && println("  Resharding complete.")
    end

    # -- 1c. Depth Range Traversal Barrier -----------------------------------
    parsed_depth_range = _parse_depth_range(params.depth_range)

    if parsed_depth_range !== nothing
        min_d, max_d = parsed_depth_range
        verbose && println(
            "\n[Phase 1c] Enforcing depth barrier: [$min_d, $max_d] m..."
        )
        depths_vec   = _extract_domain_depths(mesh, data, resharded_depths)
        out_of_depth = BitVector([d < min_d || d > max_d for d in depths_vec])

        if verbose
            n_allowed = count(!, out_of_depth)
            println(
                "  Units in range : $n_allowed / $(length(depths_vec))"
            )
        end

        combined_barrier = land_mask !== nothing ?
                           BitVector(land_mask .| out_of_depth) :
                           out_of_depth
        W, hsi_vec = apply_land_barrier(W, hsi_vec, combined_barrier)
        land_mask  = combined_barrier

        n_obs_orig = nrow(obs_df)
        valid_mask = [
            !combined_barrier[r.release]  &&
            !combined_barrier[r.recapture] &&
            _is_graph_reachable(W, r.release, r.recapture, r.k)
            for r in eachrow(obs_df)
        ]
        obs_df    = obs_df[valid_mask, :]
        n_dropped = n_obs_orig - nrow(obs_df)
        n_dropped > 0 && verbose && println(
            "  Filtered $n_dropped obs outside depth corridor."
        )
        nrow(obs_df) == 0 && error(
            "Depth constraint [$min_d, $max_d] m excluded all " *
            "mark-recapture events. Widen depth_range."
        )

        if !isnothing(survey_df) && hasproperty(survey_df, :s_idx)
            survey_df = survey_df[
                [!combined_barrier[s] for s in survey_df.s_idx], :
            ]
        end
    end

    return (
        data               = data,
        mesh               = mesh,
        W                  = W,
        hsi_vec            = hsi_vec,
        obs_df             = obs_df,
        survey_df          = survey_df,
        group_map          = group_map,
        land_mask          = land_mask,
        n_spatial          = n_spatial,
        resharded_hydro    = resharded_hydro,
        resharded_depths   = resharded_depths,
        parsed_depth_range = parsed_depth_range,
    )
end

# =============================================================================
# Phase 2: Model Fitting
# =============================================================================

"""
    fit_movement_models(loaded, params) -> NamedTuple

Phase 2 of the pipeline. Fits Bayesian movement models via Turing MCMC.

Supported model modes (set via `params.model_mode`):

- `"telemetry"`: Pure categorical mark-recapture transition likelihood.
    recapture ~ Categorical(P_g^k[release, :])
- `"telemetry_and_survey"`: Joint NegBin survey density + telemetry.
    density ~ NegBin(exp(eta_s), r), where eta_s drives advection A_g(eta).
- `"both"`: Fits both models.

# Arguments
- `loaded`: Output of `load_movement_data`.
- `params`: Configuration NamedTuple. Relevant keys: `model_mode`,
  `n_samples`, `seed`, `verbose`.

# Returns
`NamedTuple` with `models::Dict` and `chains::Dict`.
"""
function fit_movement_models(loaded, params)::NamedTuple
    verbose   = params.verbose
    mode_str  = lowercase(string(params.model_mode))
    fit_tel   = mode_str in ("telemetry", "both")
    fit_joint = mode_str in ("telemetry_and_survey", "both")
    rng       = MersenneTwister(params.seed)
    models    = Dict{Symbol, Any}()
    chains    = Dict{Symbol, Any}()

    # -- Pure Telemetry Model ------------------------------------------------
    if fit_tel
        verbose && println("\n[Phase 2] Fitting Pure Telemetry model...")
        verbose && println(
            "  recapture ~ Categorical(P_g^k[release, :])"
        )
        m_tel = @bstm(
            likelihood(recapture, family=categorical_movement) ~
                movement(
                    release   = release,
                    k         = k,
                    group     = group,
                    W         = loaded.W,
                    habitat   = loaded.hsi_vec,
                    land_mask = loaded.land_mask,
                    method    = :stochastic_kernel,
                    velocity  = truncated(Normal(0.3, 0.2), 0.0, 0.95),
                    diffusion = truncated(Normal(0.1, 0.2), 0.0, Inf),
                    gamma     = Normal(1.0, 1.0)
                ),
            data    = loaded.obs_df,
            verbose = false
        )
        models[:telemetry] = m_tel
        verbose && println("  Sampling $(params.n_samples) draws...")
        chn = Base.invokelatest(
            sample, rng, m_tel, MH(), params.n_samples; progress = false
        )
        chains[:telemetry] = chn
        verbose && println("  Pure telemetry model complete.")
    end

    # -- Joint Survey + Telemetry Model --------------------------------------
    if fit_joint && !isnothing(loaded.survey_df)
        verbose && println(
            "\n[Phase 3] Fitting Joint Survey + Telemetry model..."
        )
        verbose && println("  density ~ NegBin(exp(eta_s), r)")
        m_joint = @bstm(
            likelihood(density, family=negbin) ~
                intercept() +
                fixed(depth) +
                random(s_idx, t_idx, model=movement,
                    method         = :density_gradient,
                    telemetry_data = loaded.obs_df,
                    W              = loaded.W,
                    velocity       = truncated(Normal(0.3, 0.2), 0.0, 0.95),
                    diffusion      = truncated(Normal(0.1, 0.2), 0.0, Inf),
                    gamma          = Normal(1.0, 1.0)
                ),
            data    = loaded.survey_df,
            verbose = false
        )
        models[:telemetry_and_survey] = m_joint
        verbose && println("  Sampling $(params.n_samples) draws...")
        chn_j = Base.invokelatest(
            sample, rng, m_joint, MH(), params.n_samples; progress = false
        )
        chains[:telemetry_and_survey] = chn_j
        verbose && println("  Joint model complete.")
    end

    return (models = models, chains = chains)
end

# =============================================================================
# Phase 3: Kernel Construction
# =============================================================================

"""
    extract_transition_kernels(loaded, fitted, params) -> NamedTuple

Phase 3 of the pipeline. Extracts posterior mean parameters (alpha, rho,
gamma) from MCMC chains and constructs group-stratified stochastic
transition kernels:

    P_g = (1 - rho_g)[(1 - alpha_g) T_diff + alpha_g A_g(eta)] + rho_g I

When MCMC posteriors do not resolve group-level differences, biological
priors from `params.group_*` are applied directly as informed defaults.

# Arguments
- `loaded`: Output of `load_movement_data`.
- `fitted`: Output of `fit_movement_models`.
- `params`: Configuration NamedTuple. Relevant keys: `group_labels`,
  `group_alpha`, `group_rho`, `group_gamma`, `verbose`.

# Returns
`NamedTuple` with: `P_kernel`, `grp_name_lookup`, `alpha_hat`,
`rho_hat`, `gamma_hat`, `G`.
"""
function extract_transition_kernels(loaded, fitted, params)::NamedTuple
    verbose = params.verbose
    chains  = fitted.chains

    active_chain = haskey(chains, :telemetry) ?
                   chains[:telemetry] :
                   chains[:telemetry_and_survey]

    # Build integer-keyed group name lookup from the loaded group_map
    grp_name_lookup = Dict{Int, String}()
    for (k, v) in loaded.group_map
        if k isa String && v isa Integer
            grp_name_lookup[v] = k
        elseif k isa Integer && v isa String
            grp_name_lookup[k] = v
        else
            grp_name_lookup[Int(v)] = string(k)
        end
    end
    G = isempty(grp_name_lookup) ?
        length(params.group_labels) : maximum(keys(grp_name_lookup))

    # Extract posterior mean vector for a named parameter prefix
    function _extract_mean_vec(prefix::String, G_count::Int, defval::Float64)
        vals     = fill(defval, G_count)
        chn_keys = keys(active_chain)
        found    = false
        for g in 1:G_count
            matches = filter(
                k -> occursin(prefix, string(k)) &&
                     (G_count == 1 || occursin("[$g]", string(k))),
                chn_keys
            )
            if !isempty(matches)
                raw = Array(active_chain[first(matches)])
                m   = raw isa AbstractMatrix{<:Real} ?
                      vec(mean(raw; dims = 1)) : mean(raw)
                if m isa AbstractVector && length(m) >= g
                    vals[g] = Float64(m[g])
                elseif m isa AbstractVector && !isempty(m)
                    vals[g] = Float64(m[1])
                elseif m isa Real
                    vals[g] = Float64(m)
                end
                found = true
            end
        end
        # Fallback: any matching key vectorised across groups
        if !found || all(v -> v == defval, vals)
            fall = filter(k -> occursin(prefix, string(k)), chn_keys)
            if !isempty(fall)
                raw = Array(active_chain[first(fall)])
                m   = raw isa AbstractMatrix{<:Real} ?
                      vec(mean(raw; dims = 1)) : mean(raw)
                if m isa AbstractVector && length(m) == G_count
                    vals .= Float64.(m)
                elseif m isa Real
                    vals .= fill(Float64(m), G_count)
                end
            end
        end
        return vals
    end

    v_mean = _extract_mean_vec("velocity",  G, 0.3)
    d_mean = _extract_mean_vec("diffusion", G, 0.1)
    g_mean = _extract_mean_vec("gamma",     G, 1.0)

    tot       = v_mean .+ d_mean .+ 1e-6
    alpha_hat = clamp.(v_mean ./ tot,        0.0,  1.0)
    rho_hat   = clamp.(1.0 ./ (1.0 .+ tot), 0.01, 0.95)

    # Apply biological priors when MCMC posteriors are uninformative
    n_groups = min(G, length(params.group_labels))
    for g in 1:n_groups
        lbl = lowercase(get(grp_name_lookup, g, params.group_labels[g]))
        if all(v -> v ≈ 0.3, v_mean)
            alpha_hat[g] = params.group_alpha[
                min(g, length(params.group_alpha))
            ]
            rho_hat[g]   = params.group_rho[
                min(g, length(params.group_rho))
            ]
            g_mean[g]    = params.group_gamma[
                min(g, length(params.group_gamma))
            ]
        else
            # Label-based soft overrides for known demographic group names
            if occursin("female", lbl)
                alpha_hat[g] = 0.25
                rho_hat[g]   = 0.60
                g_mean[g]    = 0.80
            elseif occursin("male", lbl)
                alpha_hat[g] = 0.65
                rho_hat[g]   = 0.15
                g_mean[g]    = 1.50
            elseif occursin("immature", lbl)
                alpha_hat[g] = 0.30
                rho_hat[g]   = 0.35
                g_mean[g]    = 0.50
            end
        end
    end

    if verbose
        println("\n[Phase 3] Posterior parameters ($G group(s)):")
        for g in 1:G
            lbl = get(grp_name_lookup, g, "Group $g")
            println(
                "  $lbl: alpha=$(round(alpha_hat[g]; digits=4)), " *
                "rho=$(round(rho_hat[g]; digits=4)), " *
                "gamma=$(round(g_mean[g]; digits=4))"
            )
        end
    end

    P_kernel = construct_stochastic_transition_kernel(
        loaded.W, loaded.hsi_vec;
        gamma     = g_mean,
        residence = rho_hat,
        advection = alpha_hat,
        land_mask = loaded.land_mask
    )

    return (
        P_kernel        = P_kernel,
        grp_name_lookup = grp_name_lookup,
        alpha_hat       = alpha_hat,
        rho_hat         = rho_hat,
        gamma_hat       = g_mean,
        G               = G,
    )
end

# =============================================================================
# Phase 4: Path Reconstruction & Bottleneck Detection
# =============================================================================

"""
    reconstruct_paths_and_diagnostics(loaded, kernels, params) -> NamedTuple

Phase 4 of the pipeline. Reconstructs individual movement trajectories and
Markov bridge corridor heatmaps, computes optional stochastic least-cost
path ensembles, and assembles domain-wide Bayesian posterior averages for
bottleneck detection:

    F_domain(u, v) = sum_i w_i E_i(u, v)     [directed migration flux]
    C_domain(u)    = sum_i w_i rho_i(u)       [nodal transit density]
    B(u) = C_domain(u) / max(1, deg_marine(u))  [bottleneck index]

# Arguments
- `loaded`: Output of `load_movement_data`.
- `kernels`: Output of `extract_transition_kernels`.
- `params`: Relevant keys: `max_paths`, `path_method`, `smooth_paths`,
  `compute_stochastic`, `compute_bottlenecks`, `n_stochastic_draws`,
  `hsi_se`, `seed`, `verbose`.

# Returns
`NamedTuple` with: `paths`, `corridors`, `stochastic_paths`,
`domain_bottlenecks`, `cents_planar`, `cents_lonlat`, `cents_mesh`.
"""
function reconstruct_paths_and_diagnostics(
    loaded, kernels, params
)::NamedTuple
    verbose     = params.verbose
    obs_df      = loaded.obs_df
    W           = loaded.W
    hsi_vec     = loaded.hsi_vec
    land_mask   = loaded.land_mask
    n_spatial   = loaded.n_spatial
    P_kernel    = kernels.P_kernel
    G           = kernels.G
    grp_nlookup = kernels.grp_name_lookup

    cents_planar, cents_lonlat, cents_mesh =
        _resolve_centroids(loaded.mesh, n_spatial)

    all_tags    = unique(obs_df.tagid)
    n_sample    = min(params.max_paths, length(all_tags))
    sample_tags = all_tags[1:n_sample]

    verbose && println(
        "\n[Phase 4] Reconstructing $(params.path_method) trajectories " *
        "for $n_sample / $(length(all_tags)) individuals..."
    )

    reconstructed_paths     = Dict{String, Vector{Int}}()
    reconstructed_corridors = Dict{String, Matrix{Float64}}()
    stochastic_paths        = Dict{String, Any}()

    for tid in sample_tags
        sub_obs = filter(:tagid => ==(tid), obs_df)
        isempty(sub_obs) && continue

        grp = hasproperty(sub_obs, :group) ? first(sub_obs.group) : 1
        P_k = P_kernel isa AbstractVector ?
              P_kernel[clamp(grp, 1, length(P_kernel))] : P_kernel

        # Concatenate multi-segment trajectories for this individual
        full_path = Int[sub_obs.release[1]]
        for row in eachrow(sub_obs)
            seg = predict_path(
                P_k, row.release, row.recapture, row.k;
                centroids = cents_mesh,
                method    = params.path_method,
                land_mask = land_mask
            )
            append!(full_path, seg[2:end])
        end
        if params.smooth_paths && cents_mesh !== nothing
            full_path = smooth_marine_path(full_path, cents_mesh)
        end
        reconstructed_paths[string(tid)] = full_path

        # Markov bridge corridor heatmap for the first segment
        first_row = first(sub_obs)
        corr_mat  = predict_corridor(
            P_k, first_row.release, first_row.recapture, first_row.k;
            land_mask = land_mask
        )
        reconstructed_corridors[string(tid)] = corr_mat

        # Optional stochastic least-cost path ensemble
        if params.compute_stochastic && cents_planar !== nothing
            try
                stoch_res = astar_stochastic_least_cost_path(
                    cents_planar, W,
                    first_row.release, first_row.recapture;
                    hsi_mean         = hsi_vec,
                    hsi_se           = fill(params.hsi_se, n_spatial),
                    n_draws          = params.n_stochastic_draws,
                    friction_power   = 2.0,
                    land_mask        = land_mask,
                    centroids_lonlat = cents_lonlat,
                    smooth           = params.smooth_paths,
                    seed             = Int(
                        params.seed + abs(hash(string(tid))) % 10_000
                    )
                )
                stochastic_paths[string(tid)] = stoch_res
                d = stoch_res.mean_distance
                dist_km = d > 10_000.0 ? d / 1_000.0 : d
                verbose && println(
                    "  Tag $tid stochastic: " *
                    "$(length(stoch_res.medoid_path)) hops, " *
                    "$(round(dist_km; digits=1)) km"
                )
            catch e
                verbose && println("  (Stochastic A* note [$tid]: $e)")
            end
        end

        grp_lbl = haskey(grp_nlookup, grp) ?
                  " [$(grp_nlookup[grp])]" : ""
        verbose && println(
            "  Tag $tid$grp_lbl: " *
            "$(length(full_path)) units " *
            "($(first(full_path)) -> $(last(full_path)))"
        )
    end

    # -- 4b. Domain-Wide Posterior Averaging & Bottleneck Detection ----------
    domain_bottlenecks = nothing

    if params.compute_bottlenecks && cents_planar !== nothing
        verbose && println("\n[Phase 4b] Domain bottleneck detection...")
        n_obs              = nrow(obs_df)
        tag_sample_indices = Int[]

        if G > 1 && hasproperty(obs_df, :group)
            for g in 1:G
                cand = findall(
                    i -> obs_df.group[i] == g &&
                         obs_df.release[i] != obs_df.recapture[i] &&
                         (land_mask === nothing ||
                          (!land_mask[obs_df.release[i]] &&
                           !land_mask[obs_df.recapture[i]])),
                    1:n_obs
                )
                n_take = min(10, length(cand))
                n_take > 0 && append!(tag_sample_indices, cand[1:n_take])
            end
        else
            cand = findall(
                i -> obs_df.release[i] != obs_df.recapture[i] &&
                     (land_mask === nothing ||
                      (!land_mask[obs_df.release[i]] &&
                       !land_mask[obs_df.recapture[i]])),
                1:n_obs
            )
            n_take = min(25, length(cand))
            n_take > 0 && append!(tag_sample_indices, cand[1:n_take])
        end
        isempty(tag_sample_indices) &&
            append!(tag_sample_indices, collect(1:min(10, n_obs)))

        verbose && println(
            "  Aggregating $(length(tag_sample_indices)) events..."
        )

        C_domain = zeros(Float64, n_spatial)
        F_domain = spzeros(Float64, n_spatial, n_spatial)
        hsi_se_v = fill(params.hsi_se, n_spatial)

        for idx in tag_sample_indices
            u_rel = obs_df.release[idx]
            u_rec = obs_df.recapture[idx]
            res_i = astar_stochastic_least_cost_path(
                cents_planar, W, u_rel, u_rec;
                hsi_mean         = hsi_vec,
                hsi_se           = hsi_se_v,
                n_draws          = params.n_stochastic_draws,
                friction_power   = 2.0,
                land_mask        = land_mask,
                centroids_lonlat = cents_lonlat,
                smooth           = params.smooth_paths,
                seed             = params.seed + idx
            )
            C_domain .+= res_i.corridor_prob
            F_domain .+= res_i.edge_prob
        end

        # Structural bottleneck index: B(u) = C(u) / deg_marine(u)
        deg_marine = [
            count(
                j -> W[i, j] > 0 &&
                     (land_mask === nothing || !land_mask[j]),
                1:n_spatial
            )
            for i in 1:n_spatial
        ]
        bottleneck_score = zeros(Float64, n_spatial)
        for u in 1:n_spatial
            is_marine = land_mask === nothing ? true : !land_mask[u]
            if is_marine && deg_marine[u] > 0
                bottleneck_score[u] = C_domain[u] / deg_marine[u]
            end
        end

        marine_idx = land_mask === nothing ?
                     collect(1:n_spatial) : findall(!, land_mask)
        pos_scores = filter(>(0.0), bottleneck_score[marine_idx])
        b_thresh   = isempty(pos_scores) ? 0.0 : quantile(pos_scores, 0.90)
        bottleneck_mask = (bottleneck_score .>= b_thresh) .&
                          (bottleneck_score .> 0.0) .&
                          (land_mask === nothing ?
                           trues(n_spatial) : .!land_mask)

        if verbose
            println(
                "  Peak transit density: " *
                "$(round(maximum(C_domain); digits=2)) tag equiv."
            )
            println("  Active directed edges: $(nnz(F_domain))")
            println(
                "  Top-10% bottleneck threshold: " *
                "$(round(b_thresh; digits=4))"
            )
            println(
                "  Critical bottleneck units: " *
                "$(sum(bottleneck_mask)) / $n_spatial"
            )
        end

        domain_bottlenecks = (
            transit_density  = C_domain,
            edge_flux        = F_domain,
            bottleneck_score = bottleneck_score,
            bottleneck_mask  = bottleneck_mask,
            threshold        = b_thresh,
        )
    end

    return (
        paths              = reconstructed_paths,
        corridors          = reconstructed_corridors,
        stochastic_paths   = stochastic_paths,
        domain_bottlenecks = domain_bottlenecks,
        cents_planar       = cents_planar,
        cents_lonlat       = cents_lonlat,
        cents_mesh         = cents_mesh,
    )
end

# =============================================================================
# Phase 5: Advanced Diagnostics (Circuit Theory & Wavelets)
# =============================================================================

"""
    compute_advanced_diagnostics(loaded, path_results, params) -> NamedTuple

Phase 5 of the pipeline. Optionally computes:

**Circuit Theory** (`params.compute_circuit = true`):
- Multi-pair electrical current density I = C nabla V across all
  mark-recapture source-sink pairs.
- Identifies ecological pinch-points (top 10% current density).
- Posterior circuit inference via Monte Carlo HSI sampling:
    HSI_draw ~ N(hsi_mean, hsi_se^2)
  over `n_stochastic_draws * 2` replicates.

**Spectral Graph Wavelets** (`params.compute_wavelets = true`):
- Multi-scale Chebyshev SGWT decomposition on HSI (3 scales, order 25).
- BayesShrink adaptive soft-threshold spatial denoising.

# Arguments
- `loaded`: Output of `load_movement_data`.
- `path_results`: Output of `reconstruct_paths_and_diagnostics`.
- `params`: Relevant keys: `compute_circuit`, `compute_wavelets`,
  `n_stochastic_draws`, `hsi_se`, `seed`, `verbose`.

# Returns
`NamedTuple` with `circuit` and `wavelets` (each `nothing` if not computed).
"""
function compute_advanced_diagnostics(
    loaded, path_results, params
)::NamedTuple
    verbose   = params.verbose
    W         = loaded.W
    hsi_vec   = loaded.hsi_vec
    land_mask = loaded.land_mask
    n_spatial = loaded.n_spatial
    obs_df    = loaded.obs_df
    sources_v = Int.(obs_df.release)
    sinks_v   = Int.(obs_df.recapture)
    hsi_se_v  = fill(params.hsi_se, n_spatial)

    circuit_res = nothing
    wavelet_res = nothing

    # -- Circuit Theory ------------------------------------------------------
    if params.compute_circuit
        verbose && println(
            "\n[Phase 5a] Computing Circuit Theory Current Density..."
        )
        try
            cur_dens, _, _, _ = current_density_map(
                W, sources_v, sinks_v;
                hsi       = hsi_vec,
                land_mask = land_mask
            )
            p_mask, p_score, p_thresh = identify_ecological_pinchpoints(
                cur_dens; top_quantile = 0.90
            )
            verbose && println(
                "  Current density range: " *
                "[$(round(minimum(cur_dens); digits=4)), " *
                "$(round(maximum(cur_dens); digits=4))]"
            )
            verbose && println(
                "  Critical pinch-points (top 10%): " *
                "$(sum(p_mask)) / $n_spatial"
            )

            stoch_circuit = posterior_circuit_inference(
                W, sources_v, sinks_v;
                hsi_mean     = hsi_vec,
                hsi_se       = hsi_se_v,
                n_draws      = params.n_stochastic_draws * 2,
                land_mask    = land_mask,
                top_quantile = 0.90,
                seed         = params.seed
            )
            robust_mask, _, n_robust = identify_stochastic_pinchpoints(
                stoch_circuit; prob_threshold = 0.80
            )
            verbose && println(
                "  Robust pinch-points (P >= 80%): $n_robust / $n_spatial"
            )

            circuit_res = (
                current_density = cur_dens,
                pinch_mask      = p_mask,
                pinch_score     = p_score,
                threshold       = p_thresh,
                stochastic      = stoch_circuit,
                robust_mask     = robust_mask,
                n_robust        = n_robust,
            )
        catch e
            verbose && println("  (Circuit computation note: $e)")
        end
    end

    # -- Chebyshev Spectral Graph Wavelets -----------------------------------
    if params.compute_wavelets
        verbose && println(
            "\n[Phase 5b] Multi-Scale Chebyshev Spectral Graph Wavelets..."
        )
        try
            res_hsi = spectral_graph_wavelet_transform(
                W, hsi_vec; num_scales = 3, order = 25
            )
            verbose && println(
                "  SGWT decomposed across 3 scales (Chebyshev order 25)."
            )

            noisy_hsi = hsi_vec .+ 0.10 .* randn(
                MersenneTwister(params.seed + 101), n_spatial
            )
            hsi_clean, _, sigma_est = denoise_spatial_signal_wavelet(
                W, noisy_hsi; threshold_rule = :bayesshrink
            )
            verbose && println(
                "  BayesShrink noise sigma: $(round(sigma_est; digits=4))"
            )

            wavelet_res = (
                sgwt            = res_hsi,
                denoised_hsi    = hsi_clean,
                estimated_noise = sigma_est,
            )
        catch e
            verbose && println("  (Wavelet decomposition note: $e)")
        end
    end

    return (circuit = circuit_res, wavelets = wavelet_res)
end

# =============================================================================
# Phase 6: Dashboard Export
# =============================================================================

"""
    _write_summary_diagnostics_html(
        filepath, species, dists, vels, bearings, paths_rich
    )

Private helper. Writes a standalone HTML file with inline SVG
histograms summarising reconstructed path metrics:

- **Distance histogram**: total traversed distance (km) per individual.
- **Velocity histogram**: mean velocity (km / time step).
- **Directional wind-rose**: bearing angle distribution from release
  to recapture, binned into 16 compass sectors.
- **Summary table**: per-path metrics (tag, distance, displacement,
  tortuosity, mean HSI, duration).
"""
function _write_summary_diagnostics_html(
    filepath::AbstractString,
    species::AbstractString,
    dists::Vector{Float64},
    vels::Vector{Float64},
    bearings::Vector{Float64},
    paths_rich::Vector{<:NamedTuple}
)::Nothing
    # --- SVG histogram builder ---
    function _svg_histogram(
        vals::Vector{Float64}, n_bins::Int,
        xlabel::String, color::String;
        width::Int = 520, height::Int = 260
    )::String
        isempty(vals) && return "<p>No data</p>"
        lo, hi = minimum(vals), maximum(vals)
        hi <= lo && (hi = lo + 1.0)
        bin_w = (hi - lo) / n_bins
        counts = zeros(Int, n_bins)
        for v in vals
            b = clamp(
                floor(Int, (v - lo) / bin_w) + 1, 1, n_bins
            )
            counts[b] += 1
        end
        mx = maximum(counts)
        mx == 0 && (mx = 1)

        pad_l, pad_b, pad_t, pad_r = 50, 40, 20, 10
        plot_w = width - pad_l - pad_r
        plot_h = height - pad_t - pad_b
        bar_w  = plot_w / n_bins

        io = IOBuffer()
        write(io, """<svg width="$width" height="$height"
          xmlns="http://www.w3.org/2000/svg">""")
        # Axes
        write(io, """<line x1="$pad_l" y1="$(height - pad_b)"
          x2="$(width - pad_r)" y2="$(height - pad_b)"
          stroke="#94a3b8" stroke-width="1"/>""")
        write(io, """<line x1="$pad_l" y1="$pad_t"
          x2="$pad_l" y2="$(height - pad_b)"
          stroke="#94a3b8" stroke-width="1"/>""")

        for i in 1:n_bins
            bh = (counts[i] / mx) * plot_h
            bx = pad_l + (i - 1) * bar_w + 1
            by = height - pad_b - bh
            write(io, """<rect x="$(round(bx, digits=1))"
              y="$(round(by, digits=1))"
              width="$(round(bar_w - 2, digits=1))"
              height="$(round(bh, digits=1))"
              fill="$color" opacity="0.85"
              rx="2"/>""")
        end

        # X-axis label
        write(io, """<text x="$(pad_l + plot_w ÷ 2)"
          y="$(height - 5)" text-anchor="middle"
          fill="#94a3b8" font-size="12"
          font-family="Outfit, sans-serif">$xlabel</text>""")

        # Tick labels (lo / hi)
        write(io, """<text x="$pad_l"
          y="$(height - pad_b + 15)"
          text-anchor="start" fill="#94a3b8"
          font-size="10"
          font-family="JetBrains Mono">
          $(round(lo, digits=1))</text>""")
        write(io, """<text x="$(width - pad_r)"
          y="$(height - pad_b + 15)"
          text-anchor="end" fill="#94a3b8"
          font-size="10"
          font-family="JetBrains Mono">
          $(round(hi, digits=1))</text>""")

        # Y-axis max label
        write(io, """<text x="$(pad_l - 5)"
          y="$(pad_t + 10)"
          text-anchor="end" fill="#94a3b8"
          font-size="10"
          font-family="JetBrains Mono">$mx</text>""")

        write(io, "</svg>")
        return String(take!(io))
    end

    # --- Directional wind-rose SVG ---
    function _svg_windrose(
        angles::Vector{Float64};
        size::Int = 300, n_sectors::Int = 16
    )::String
        isempty(angles) && return "<p>No data</p>"
        sector_w = 360.0 / n_sectors
        counts = zeros(Int, n_sectors)
        for a in angles
            s = clamp(
                floor(Int, mod(a, 360.0) / sector_w) + 1,
                1, n_sectors
            )
            counts[s] += 1
        end
        mx = maximum(counts)
        mx == 0 && (mx = 1)

        cx, cy = size ÷ 2, size ÷ 2
        r_max = size ÷ 2 - 30

        io = IOBuffer()
        write(io, """<svg width="$size" height="$size"
          xmlns="http://www.w3.org/2000/svg">""")

        # Concentric guide circles
        for frac in [0.25, 0.5, 0.75, 1.0]
            r = round(Int, r_max * frac)
            write(io, """<circle cx="$cx" cy="$cy" r="$r"
              fill="none" stroke="#334155"
              stroke-width="0.5"/>""")
        end

        # Compass labels
        for (lbl, ax, ay) in [
            ("N", cx, cy - r_max - 12),
            ("E", cx + r_max + 12, cy + 4),
            ("S", cx, cy + r_max + 16),
            ("W", cx - r_max - 12, cy + 4),
        ]
            write(io, """<text x="$ax" y="$ay"
              text-anchor="middle" fill="#64748b"
              font-size="11"
              font-family="Outfit">$lbl</text>""")
        end

        # Petal arcs (filled wedges)
        for s in 1:n_sectors
            r_s = (counts[s] / mx) * r_max
            r_s < 2.0 && continue
            θ_start = (s - 1) * sector_w - 90.0
            θ_end   = θ_start + sector_w
            θ1r = deg2rad(θ_start)
            θ2r = deg2rad(θ_end)
            x1 = cx + r_s * cos(θ1r)
            y1 = cy + r_s * sin(θ1r)
            x2 = cx + r_s * cos(θ2r)
            y2 = cy + r_s * sin(θ2r)
            large = sector_w > 180 ? 1 : 0
            write(io, string(
                "<path d=\"M $cx $cy L ",
                round(x1, digits=1), " ",
                round(y1, digits=1),
                " A ", round(r_s, digits=1), " ",
                round(r_s, digits=1),
                " 0 $large 1 ",
                round(x2, digits=1), " ",
                round(y2, digits=1),
                " Z\" fill=\"#38bdf8\" opacity=\"0.6\"",
                " stroke=\"#38bdf8\" stroke-width=\"0.5\"/>"
            ))
        end

        write(io, "</svg>")
        return String(take!(io))
    end

    # --- Build page ---
    dist_svg    = _svg_histogram(dists, 20, "Total Distance (km)", "#10b981")
    vel_svg     = _svg_histogram(vels, 15, "Velocity (km / step)", "#fbbf24")
    rose_svg    = _svg_windrose(bearings)

    # Summary statistics
    n_paths     = length(dists)
    mean_dist   = isempty(dists) ? 0.0 : mean(dists)
    median_dist = isempty(dists) ? 0.0 : quantile(dists, 0.5)
    mean_vel    = isempty(vels)  ? 0.0 : mean(vels)

    # Per-path table rows
    table_rows = String[]
    for (i, p) in enumerate(paths_rich)
        push!(table_rows, string(
            "<tr>",
            "<td>", p.tagid, "</td>",
            "<td>", round(p.total_dist_km, digits=1), "</td>",
            "<td>", round(p.displacement_km, digits=1), "</td>",
            "<td>", round(p.tortuosity, digits=2), "</td>",
            "<td>", round(p.mean_hsi, digits=3), "</td>",
            "<td>", round(Int, p.duration_days), "</td>",
            "<td><span style=\"color: $(p.color)\">●</span></td>",
            "</tr>"
        ))
    end

    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>$species — Movement Summary Diagnostics</title>
  <link href="https://fonts.googleapis.com/css2?\
family=Outfit:wght@300;400;600;700&\
family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b1329; --panel: rgba(15,23,42,0.92);
      --text: #f8fafc; --muted: #94a3b8;
      --border: rgba(255,255,255,0.10);
      --accent: #38bdf8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Outfit', sans-serif;
      background: var(--bg); color: var(--text);
      padding: 30px 40px;
    }
    h1 { font-size: 1.6rem; font-weight: 700;
         margin-bottom: 6px; }
    .subtitle { color: var(--muted); font-size: 0.9rem;
                margin-bottom: 28px; }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-bottom: 30px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px; padding: 20px;
    }
    .card h2 { font-size: 1.05rem; font-weight: 600;
               margin-bottom: 12px; }
    .stat-row { display: flex; gap: 24px;
                margin-bottom: 22px; }
    .stat-box {
      background: rgba(56,189,248,0.08);
      border: 1px solid rgba(56,189,248,0.2);
      border-radius: 8px; padding: 12px 16px; flex: 1;
    }
    .stat-label { font-size: 0.75rem; color: var(--muted);
                  text-transform: uppercase; letter-spacing: 0.05em; }
    .stat-val { font-size: 1.3rem; font-weight: 700;
                color: var(--accent);
                font-family: 'JetBrains Mono', monospace; }
    table {
      width: 100%; border-collapse: collapse;
      font-size: 0.85rem;
    }
    th { text-align: left; color: var(--muted);
         border-bottom: 1px solid var(--border);
         padding: 8px 6px; font-weight: 500; }
    td { padding: 6px; border-bottom: 1px solid
         rgba(255,255,255,0.04); }
    tr:hover td { background: rgba(56,189,248,0.05); }
  </style>
</head>
<body>
<h1>$species — Movement Summary Diagnostics</h1>
<p class="subtitle">$n_paths reconstructed trajectories</p>

<div class="stat-row">
  <div class="stat-box">
    <div class="stat-label">Mean Distance</div>
    <div class="stat-val">$(round(mean_dist, digits=1)) km</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Median Distance</div>
    <div class="stat-val">$(round(median_dist, digits=1)) km</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Mean Velocity</div>
    <div class="stat-val">$(round(mean_vel, digits=2)) km/step</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Paths</div>
    <div class="stat-val">$n_paths</div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>Total Distance Distribution (km)</h2>
    $dist_svg
  </div>
  <div class="card">
    <h2>Velocity Distribution (km / time step)</h2>
    $vel_svg
  </div>
  <div class="card">
    <h2>Directional Wind-Rose (Bearing °N)</h2>
    $rose_svg
  </div>
  <div class="card">
    <h2>Bearing Rose</h2>
    <p style="color: var(--muted); font-size: 0.85rem;">
      Each petal represents the fraction of paths whose
      release→recapture bearing falls in that compass sector.
      North = 0°, East = 90°, clockwise.
    </p>
  </div>
</div>

<div class="card">
  <h2>Per-Path Summary Table</h2>
  <div style="max-height: 500px; overflow-y: auto;">
    <table>
      <thead>
        <tr>
          <th>Tag ID</th><th>Dist (km)</th>
          <th>Displ (km)</th><th>Tortuosity</th>
          <th>Mean HSI</th><th>Duration (steps)</th>
          <th>Group</th>
        </tr>
      </thead>
      <tbody>
        $(join(table_rows, "\n        "))
      </tbody>
    </table>
  </div>
</div>
</body>
</html>"""

    open(filepath, "w") do f
        write(f, html)
    end
    return nothing
end

"""
    export_dashboards(loaded, kernels, path_results, diagnostics, params)

Phase 6 of the pipeline. Exports interactive Leaflet HTML dashboards to
`params.output_dir`. Individual dashboards are silently skipped on
rendering errors so the pipeline is never aborted. Exports:

- Movement path trajectories and empirical displacement vectors.
- Interactive two-click migration corridor explorer.
- Hydrodynamic stratification dashboard (when domain was resharded).
- Circuit current density and posterior pinch-point maps.
- Domain-wide bottleneck conduit heatmap.
- Multi-scale SGWT wavelet decomposition dashboard.
"""
function export_dashboards(
    loaded, kernels, path_results, diagnostics, params
)::Nothing
    params.render_html || return nothing
    verbose = params.verbose

    verbose && println("\n[Phase 6] Generating Leaflet dashboards...")
    out_dir = params.output_dir
    mkpath(out_dir)

    mesh        = loaded.mesh
    hsi_vec     = loaded.hsi_vec
    obs_df      = loaded.obs_df
    P_kernel    = kernels.P_kernel
    G           = kernels.G
    grp_nlookup = kernels.grp_name_lookup
    spp         = params.species_name

    # Use the resolved centroids from path reconstruction so node
    # indices are consistent with the (possibly resharded) mesh.
    cents_ll = path_results.cents_lonlat !== nothing ?
               path_results.cents_lonlat :
               (hasproperty(mesh, :centroids_lonlat) ?
                mesh.centroids_lonlat : mesh.centroids)
    n_units = length(cents_ll)

    polys_ll = hasproperty(mesh, :polygons_lonlat) ?
               mesh.polygons_lonlat :
               (hasproperty(mesh, :polygons) ?
                mesh.polygons : nothing)
    au_mesh  = (
        centroids        = cents_ll,
        centroids_lonlat = cents_ll,
        polygons         = polys_ll,
        polygons_lonlat  = polys_ll,
        W                = loaded.W,
        n_units          = n_units,
    )

    # Empirical release-recapture displacement vectors
    emp_tracks = [
        [
            (Float64(cents_ll[r.release][1]),
             Float64(cents_ll[r.release][2])),
            (Float64(cents_ll[r.recapture][1]),
             Float64(cents_ll[r.recapture][2])),
        ]
        for r in eachrow(obs_df)
        if 1 <= r.release   <= n_units &&
           1 <= r.recapture <= n_units
    ]

    depth_lbl = if !isnothing(loaded.parsed_depth_range)
        min_d, max_d = loaded.parsed_depth_range
        " [Depth: $(round(Int, min_d))-$(round(Int, max_d)) m]"
    else
        ""
    end
    reshard_lbl = (params.reshard_hex || params.use_hydrodynamics) ?
                  " (Fine Hexagons)" : ""

    # -- Build rich path NamedTuples for the tracks dashboard ------
    #
    # Convert raw Dict{String, Vector{Int}} into NamedTuples with
    # centroid coordinates, per-path distance, displacement,
    # tortuosity, mean HSI, and a per-group colour assignment.
    palette_colors = [
        "#38bdf8", "#f43f5e", "#10b981", "#fbbf24",
        "#a78bfa", "#fb923c", "#22d3ee", "#e879f9",
    ]
    all_paths_rich = NamedTuple[]
    path_dists_km  = Float64[]
    path_vels      = Float64[]
    path_bearings  = Float64[]

    for (tid, node_vec) in path_results.paths
        length(node_vec) < 2 && continue

        # Coordinate series via resolved centroids
        coords = Tuple{Float64, Float64}[
            (Float64(cents_ll[u][1]), Float64(cents_ll[u][2]))
            for u in node_vec
            if 1 <= u <= n_units
        ]
        length(coords) < 2 && continue

        # Cumulative Haversine distance along the path (km)
        total_dist = 0.0
        for h in 2:length(coords)
            total_dist += haversine_distance(
                coords[h-1][1], coords[h-1][2],
                coords[h][1],   coords[h][2]
            ) / 1_000.0
        end

        # Net displacement (great-circle, km)
        displacement = haversine_distance(
            coords[1][1], coords[1][2],
            coords[end][1], coords[end][2]
        ) / 1_000.0

        # Tortuosity (path length / displacement); clamp for
        # coincident release-recapture
        tort = displacement > 0.01 ? total_dist / displacement : 1.0

        # Mean habitat suitability along the trajectory
        mean_h = mean([
            (1 <= u <= length(hsi_vec)) ? hsi_vec[u] : 0.5
            for u in node_vec
        ])

        # Look up the group-specific colour for this tag
        sub_obs = filter(:tagid => ==(tid), obs_df)
        grp = !isempty(sub_obs) && hasproperty(sub_obs, :group) ?
              first(sub_obs.group) : 1
        grp_lbl = get(grp_nlookup, grp, "Group $grp")
        color = palette_colors[
            (grp - 1) % length(palette_colors) + 1
        ]

        # Duration in time steps (sum of k across segments)
        k_total = !isempty(sub_obs) ? sum(sub_obs.k) : 1

        push!(all_paths_rich, (
            tagid           = string(tid),
            coords          = coords,
            n_steps         = length(coords) - 1,
            total_dist_km   = total_dist,
            displacement_km = displacement,
            tortuosity      = tort,
            mean_hsi        = mean_h,
            color           = color,
            duration_days   = Float64(k_total),
        ))

        push!(path_dists_km, total_dist)
        if k_total > 0
            push!(path_vels, total_dist / k_total)
        end

        # Bearing from release to recapture (degrees from north)
        Δlon = coords[end][1] - coords[1][1]
        Δlat = coords[end][2] - coords[1][2]
        bearing = atand(Δlon, Δlat)
        bearing < 0.0 && (bearing += 360.0)
        push!(path_bearings, bearing)
    end

    verbose && println(
        "  Rich path NamedTuples built: $(length(all_paths_rich))"
    )

    # -- Movement paths dashboard ---------------------------------
    try
        html_file = joinpath(out_dir, "movement_paths_dashboard.html")
        map_obj = leaflet_tracks_map(
            all_paths_rich, au_mesh;
            empirical_paths     = emp_tracks,
            max_paths           = max(100, length(all_paths_rich)),
            max_empirical_paths = max(500, length(emp_tracks)),
            hsi                 = hsi_vec,
            title               = "$spp Movement Trajectories" *
                                  reshard_lbl * depth_lbl
        )
        save_html(map_obj, html_file)
        verbose && println("  Paths dashboard: $html_file")
    catch e
        verbose && println("  (Leaflet paths note: $e)")
    end

    # -- Movement summary diagnostics dashboard -----------------------
    if !isempty(path_dists_km)
        try
            summ_file = joinpath(
                out_dir, "movement_summary_diagnostics.html"
            )
            _write_summary_diagnostics_html(
                summ_file, spp, path_dists_km, path_vels,
                path_bearings, all_paths_rich
            )
            verbose && println(
                "  Summary diagnostics: $summ_file"
            )
        catch e
            verbose && println(
                "  (Summary diagnostics note: $e)"
            )
        end
    end

    # -- Interactive two-click corridor explorer -----------------------------
    try
        corr_file  = joinpath(out_dir, "movement_interactive_corridor.html")
        grp_labels = [get(grp_nlookup, g, "Group $g") for g in 1:G]
        corr_map   = leaflet_interactive_corridor_dashboard(
            P_kernel, au_mesh;
            hsi             = hsi_vec,
            empirical_paths = emp_tracks,
            group_labels    = grp_labels,
            title           = "$spp Dynamic Migration Corridor" *
                              reshard_lbl * depth_lbl
        )
        save_html(corr_map, corr_file)
        verbose && println("  Corridor dashboard: $corr_file")
    catch e
        verbose && println("  (Corridor dashboard note: $e)")
    end

    # -- Hydrodynamic dashboard (only when resharded) ------------------------
    if !isnothing(loaded.resharded_hydro)
        try
            hydro_file = joinpath(out_dir, "hydrodynamic_hex_dashboard.html")
            dash = leaflet_hydrodynamic_dashboard(
                loaded.resharded_hydro, au_mesh;
                title = "Hydrodynamics & Stratification (Fine Hexagons)"
            )
            save_html(dash, hydro_file)
            verbose && println("  Hydrodynamic dashboard: $hydro_file")
        catch e
            verbose && println("  (Hydrodynamic dashboard note: $e)")
        end
    end

    # -- Circuit current density & stochastic pinch-point dashboards ---------
    if !isnothing(diagnostics.circuit)
        circ = diagnostics.circuit
        try
            circ_file = joinpath(out_dir, "movement_current_density.html")
            leaflet_current_density_map(
                mesh, circ.current_density;
                pinch_mask   = circ.pinch_mask,
                pinch_score  = circ.pinch_score,
                centroids    = path_results.cents_lonlat,
                output_html  = circ_file,
                title        = "$spp Migratory Current Density & Pinch-Points"
            )
            verbose && println("  Current density dashboard: $circ_file")
        catch e
            verbose && println("  (Circuit density note: $e)")
        end

        try
            stoch_file = joinpath(out_dir, "movement_stochastic_circuit.html")
            leaflet_current_density_map(
                mesh, circ.stochastic;
                prob_threshold = 0.80,
                output_html    = stoch_file,
                title          = "$spp Posterior Migratory Flux & Pinch-Points"
            )
            verbose && println("  Stochastic circuit dashboard: $stoch_file")
        catch e
            verbose && println("  (Stochastic circuit note: $e)")
        end
    end

    # -- Domain-wide bottleneck dashboard ------------------------------------
    if !isnothing(path_results.domain_bottlenecks)
        bn = path_results.domain_bottlenecks
        try
            bn_file = joinpath(out_dir, "movement_domain_bottlenecks.html")
            leaflet_current_density_map(
                mesh, bn.transit_density;
                pinch_mask   = bn.bottleneck_mask,
                pinch_score  = bn.bottleneck_score,
                centroids    = path_results.cents_lonlat,
                output_html  = bn_file,
                title        = "$spp Domain-Wide Pathways & Bottlenecks",
                legend_title = "Transit Density (C)"
            )
            verbose && println("  Bottleneck dashboard: $bn_file")
        catch e
            verbose && println("  (Bottleneck rendering note: $e)")
        end
    end

    # -- Multi-scale wavelet dashboard ---------------------------------------
    if !isnothing(diagnostics.wavelets)
        wv = diagnostics.wavelets
        try
            wv_file = joinpath(out_dir, "movement_wavelet_dashboard.html")
            leaflet_graph_wavelet_dashboard(
                mesh, wv.sgwt;
                reconstruction = wv.denoised_hsi,
                signal_name    = "Habitat Suitability (HSI)",
                title          = "$spp Multi-Scale Habitat (HSI) Wavelets",
                output_html    = wv_file
            )
            verbose && println("  Wavelet dashboard: $wv_file")
        catch e
            verbose && println("  (Wavelet dashboard note: $e)")
        end
    end

    return nothing
end

# =============================================================================
# Orchestrator
# =============================================================================

"""
    run_movement_analysis(params = movement_parameters_default()) -> NamedTuple

Executes the complete BSTM movement analysis pipeline. All six phases
are called in sequence:

1. `load_movement_data`                 -- data ingestion & depth barriers
2. `fit_movement_models`                -- Bayesian MCMC model fitting
3. `extract_transition_kernels`         -- posterior kernel construction
4. `reconstruct_paths_and_diagnostics` -- paths, corridors, bottlenecks
5. `compute_advanced_diagnostics`      -- circuit theory & wavelets
6. `export_dashboards`                 -- interactive Leaflet HTML maps

# Arguments
- `params`: Configuration NamedTuple. Start from a parameter preset and
  merge individual overrides:

      # Generic simulation
      run_movement_analysis()

      # Snow crab with custom depth range
      p = merge(movement_parameters_snowcrab(),
                (depth_range = (80.0, 400.0), n_samples = 500))
      run_movement_analysis(p)

# Returns
`NamedTuple` with fields: `data`, `models`, `chains`, `P_kernel`,
`paths`, `corridors`, `stochastic_paths`, `domain_bottlenecks`,
`circuit`, `wavelets`, `parameters`, `depth_range`.
"""
function run_movement_analysis(
    params = movement_parameters_default()
)::NamedTuple
    loaded      = load_movement_data(params)
    fitted      = fit_movement_models(loaded, params)
    kernels     = extract_transition_kernels(loaded, fitted, params)
    path_res    = reconstruct_paths_and_diagnostics(loaded, kernels, params)
    diagnostics = compute_advanced_diagnostics(loaded, path_res, params)
    export_dashboards(loaded, kernels, path_res, diagnostics, params)

    if params.verbose
        println("\n" * "=" ^ 72)
        println("  Pipeline Completed Successfully!")
        println("=" ^ 72)
    end

    return (
        data               = loaded.data,
        models             = fitted.models,
        chains             = fitted.chains,
        P_kernel           = kernels.P_kernel,
        paths              = path_res.paths,
        corridors          = path_res.corridors,
        stochastic_paths   = path_res.stochastic_paths,
        domain_bottlenecks = path_res.domain_bottlenecks,
        circuit            = diagnostics.circuit,
        wavelets           = diagnostics.wavelets,
        parameters         = (
            alpha     = kernels.alpha_hat,
            residence = kernels.rho_hat,
            gamma     = kernels.gamma_hat,
        ),
        depth_range        = loaded.parsed_depth_range,
    )
end

# =============================================================================
# Command-Line Interface
# =============================================================================

"""
    print_movement_help()

Prints CLI usage instructions for `movement_analysis.jl`.
"""
function print_movement_help()
    println("""
BSTM Movement Analysis Pipeline
================================
Usage:
  julia --project=. docs/movement/movement_analysis.jl [OPTIONS]

Data & Model:
  -d, --data-source <src>   'simulate' (default) or 'snowcrab'.
      --simulate             Shorthand for --data-source=simulate.
      --snowcrab             Shorthand: snowcrab preset + all diagnostics.
  -m, --model-mode <mode>   'telemetry' (default), 'telemetry_and_survey',
                             or 'both'.
      --telemetry            Shorthand for --model-mode=telemetry.
      --joint, --survey      Shorthand for telemetry_and_survey.
      --both                 Shorthand for --model-mode=both.

Domain Resharding:
      --reshard-hex, --hex   Reshard to finer hexagons via LibGEOS.
      --hex-radius <km>      Fine hexagon cell radius in km (default: 10.0).
      --hydro                Extract 3D hydrodynamic fields and bathymetry.
      --depth-range <m,M>    Restrict movement to depth range [m, M] m.

Path Reconstruction:
  -p, --max-paths <N>        Maximum trajectories (default: 25).
      --astar                Use A* path method (default).
      --viterbi              Use Viterbi path method.
      --smooth-paths         Apply line-of-sight raycast smoothing.

Optional Diagnostics:
      --stochastic           Stochastic least-cost path ensembles.
      --bottlenecks          Domain-wide bottleneck index B(u).
      --circuit              Circuit current density & pinch-points.
      --wavelets, --sgwt     Chebyshev spectral graph wavelets.
      --all-diagnostics      Enable all four optional diagnostics.

MCMC:
  -n, --samples <N>          Posterior draws (default: 200).
  -w, --warmup <N>           Warmup iterations (default: 100).
      --seed <N>             Random seed (default: 42).
      --hsi-se <sigma>       HSI observation error (default: 0.08).
      --n-draws <N>          MC draws per stochastic path (default: 10).

Output:
      --no-html              Disable HTML export.
  -o, --output-dir <path>    Output directory (default: <repo>/output).
  -q, --quiet                Suppress progress output.
  -h, --help                 Display this help and exit.

Examples:
  julia --project=. docs/movement/movement_analysis.jl --simulate
  julia --project=. docs/movement/movement_analysis.jl --snowcrab
  julia --project=. docs/movement/movement_analysis.jl --depth-range 50,400
  julia --project=. docs/movement/movement_analysis.jl --snowcrab --wavelets
  julia --project=. docs/movement/movement_analysis.jl --help
""")
    return nothing
end

"""
    _is_cli_flag(token) -> Bool

Returns `true` when a CLI token is a flag switch (e.g., `--seed`)
rather than a negative numeric value (e.g., `-1.5`).
"""
_is_cli_flag(token::AbstractString) =
    startswith(token, "--") ||
    (startswith(token, "-") && tryparse(Float64, token) === nothing)

"""
    parse_movement_cli_args(args = ARGS) -> NamedTuple

Parses CLI argument strings into a configuration NamedTuple suitable for
merging with `movement_parameters_*()` output. Supports UNIX-style long
(`--flag`, `--flag=val`) and short (`-f`, `-f val`) flags.

Returns a NamedTuple with only the keys explicitly provided on the CLI,
ready for `merge(base_params, cli_opts)`.
"""
function parse_movement_cli_args(
    args::AbstractVector{<:AbstractString} = ARGS
)::NamedTuple
    opts = Dict{Symbol, Any}()
    idx  = 1
    n    = length(args)

    _is_false(s) = lowercase(s) in ("false", "0", "no", "off")

    function fetch_val(key)
        idx + 1 <= n && !_is_cli_flag(args[idx + 1]) ||
            throw(ArgumentError("CLI flag '$key' requires a value."))
        idx += 1
        return args[idx]
    end

    while idx <= n
        raw = args[idx]
        key, inline_val, has_inline = raw, nothing, false

        if startswith(raw, "-") && occursin("=", raw)
            parts = Base.split(raw, "="; limit = 2)
            key, inline_val, has_inline = parts[1], parts[2], true
        end

        fv() = has_inline ? inline_val : fetch_val(key)

        if key in ("--help", "-h")
            opts[:help] = true
        elseif key in ("--snowcrab", "--crab")
            opts[:data_source]         = :snowcrab
        elseif key in ("--simulate", "--sim")
            opts[:data_source] = :simulate
        elseif key in ("--data-source", "--data", "-d")
            raw_src = lowercase(fv())
            opts[:data_source] = raw_src in ("snowcrab", "crab", "real") ?
                                 :snowcrab : :simulate
        elseif key == "--telemetry"
            opts[:model_mode] = "telemetry"
        elseif key in ("--joint", "--survey", "--telemetry-and-survey")
            opts[:model_mode] = "telemetry_and_survey"
        elseif key in ("--both", "--all-models")
            opts[:model_mode] = "both"
        elseif key in ("--model-mode", "--mode", "-m")
            raw_m = lowercase(fv())
            opts[:model_mode] = raw_m in ("joint", "survey") ?
                                "telemetry_and_survey" :
                                raw_m in ("both", "all") ? "both" :
                                "telemetry"
        elseif key in ("--reshard-hex", "--hex", "--reshard")
            opts[:reshard_hex] = has_inline ? !_is_false(inline_val) : true
        elseif key in ("--hex-radius", "--hex-res", "--radius-km")
            opts[:hex_radius_km] = parse(Float64, fv())
        elseif key in ("--hydro", "--hydrodynamics")
            opts[:use_hydrodynamics] = has_inline ?
                                       !_is_false(inline_val) : true
        elseif key in ("--depth-range", "--depths", "--depth")
            val_str = fv()
            sep   = occursin(":", val_str) ? ":" : ","
            parts = Base.split(val_str, sep)
            length(parts) == 2 || throw(ArgumentError(
                "Invalid --depth-range '$val_str'. Expected 'min,max'."
            ))
            opts[:depth_range] = (
                parse(Float64, strip(parts[1])),
                parse(Float64, strip(parts[2]))
            )
        elseif key in ("--max-paths", "--paths", "-p")
            opts[:max_paths] = parse(Int, fv())
        elseif key == "--astar"
            opts[:path_method] = :astar
        elseif key == "--viterbi"
            opts[:path_method] = :viterbi
        elseif key in ("--path-method", "--algorithm")
            opts[:path_method] = Symbol(lowercase(fv()))
        elseif key in ("--smooth-paths", "--smooth")
            opts[:smooth_paths] = true
        elseif key in ("--stochastic", "--stochastic-paths")
            opts[:compute_stochastic] = has_inline ?
                                        !_is_false(inline_val) : true
        elseif key in ("--bottlenecks", "--bottleneck")
            opts[:compute_bottlenecks] = has_inline ?
                                         !_is_false(inline_val) : true
        elseif key in ("--circuit", "--current-density")
            opts[:compute_circuit] = has_inline ?
                                     !_is_false(inline_val) : true
        elseif key in ("--wavelets", "--wavelet", "--sgwt")
            opts[:compute_wavelets] = has_inline ?
                                      !_is_false(inline_val) : true
        elseif key == "--all-diagnostics"
            opts[:compute_stochastic]  = true
            opts[:compute_bottlenecks] = true
            opts[:compute_circuit]     = true
            opts[:compute_wavelets]    = true
        elseif key in ("--n-samples", "--samples", "-n", "-s")
            opts[:n_samples] = parse(Int, fv())
        elseif key in ("--n-warmup", "--warmup", "-w")
            opts[:n_warmup] = parse(Int, fv())
        elseif key == "--seed"
            opts[:seed] = parse(Int, fv())
        elseif key in ("--hsi-se", "--hsi-error")
            opts[:hsi_se] = parse(Float64, fv())
        elseif key in ("--n-draws", "--stochastic-draws")
            opts[:n_stochastic_draws] = parse(Int, fv())
        elseif key in ("--render-html", "--html")
            opts[:render_html] = has_inline ? !_is_false(inline_val) : true
        elseif key in ("--no-render-html", "--no-html")
            opts[:render_html] = false
        elseif key in ("--output-dir", "--output", "-o")
            opts[:output_dir] = String(fv())
        elseif key in ("--verbose", "-v")
            opts[:verbose] = true
        elseif key in ("--quiet", "-q", "--silent")
            opts[:verbose] = false
        else
            @warn "Unrecognized CLI flag: '$raw' -- ignoring."
        end

        idx += 1
    end

    return (; opts...)
end

# =============================================================================
# Standalone Script Entry Point
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    cli = parse_movement_cli_args(ARGS)
    if get(cli, :help, false)
        print_movement_help()
    else
        base_params = (
            haskey(cli, :data_source) && cli.data_source == :snowcrab ?
            movement_parameters_snowcrab() :
            movement_parameters_default()
        )
        run_movement_analysis(merge(base_params, cli))
    end
end
