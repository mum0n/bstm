"""
    movement_individual.jl

End-to-end executable workflow for Bayesian Spatio-Temporal movement modelling in BSTM.

Demonstrates:
1. Pure Telemetry Model (`"telemetry"`, default):
   Direct categorical mark-recapture model without dummy outcomes, using the well-conditioned
   convex stochastic transition kernel: P_g = (1 - ρ_g) [ (1 - α_g) T_diff + α_g A_g ] + ρ_g I
2. Joint Telemetry & Survey Density Model (`"telemetry_and_survey"`):
   Joint hierarchical model where survey numerical density (NegBin) dynamically constructs
   the latent habitat suitability field (HSI = η_s), simultaneously informing individual
   movement advection and survey catches.
3. Post-processing path reconstruction:
   - Viterbi dynamic programming for the single most likely path (predict_path).
   - Markov bridge transition probability heatmaps across space and time (predict_corridor).
4. Spatial visualization of tracks, transition corridors, and habitat gradients.

Supported Data Sources:
- `bstm_data("movement")`: Self-contained synthetic movement and survey dataset.
- `snowcrab_movement_data()`: Real Scotian Shelf snow crab acoustic tagging and trawl surveys.
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

# Include local snow crab dataset processor if present in repository
const _SNOWCRAB_SCRIPT = joinpath(@__DIR__, "snowcrab_movement_data.jl")
if isfile(_SNOWCRAB_SCRIPT)
    include(_SNOWCRAB_SCRIPT)
end

# ─────────────────────────────────────────────────────────────────────────────
# Depth Extraction & Network Topology Helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    _extract_domain_depths(mesh, data, resharded_depths) -> Vector{Float64}

Extracts or synthesizes bathymetric depth values in meters for all units in `mesh`.
Prioritizes resharded fine hexagonal depths, then loaded dataset depths, mesh depths,
and finally synthetic bathymetric contours.
"""
function _extract_domain_depths(mesh, data, resharded_depths)::Vector{Float64}
    n = mesh.n_units
    if !isnothing(resharded_depths) && length(resharded_depths) == n
        return Float64.(resharded_depths)
    elseif hasproperty(data, :depth_vec) && !isnothing(data.depth_vec) &&
           length(data.depth_vec) == n
        return Float64.(data.depth_vec)
    elseif hasproperty(data, :survey_df) && !isnothing(data.survey_df) &&
           hasproperty(data.survey_df, :depth) && length(data.survey_df.depth) == n
        return Float64.(data.survey_df.depth)
    elseif hasproperty(mesh, :depth_vec) && !isnothing(mesh.depth_vec) &&
           length(mesh.depth_vec) == n
        return Float64.(mesh.depth_vec)
    elseif hasproperty(mesh, :centroids_km) && !isnothing(mesh.centroids_km) &&
           length(mesh.centroids_km) == n
        return [150.0 + 60.0 * sin(mesh.centroids_km[s][1] / 40.0) +
                40.0 * cos(mesh.centroids_km[s][2] / 40.0) for s in 1:n]
    elseif hasproperty(mesh, :centroids_lonlat) && !isnothing(mesh.centroids_lonlat) &&
           length(mesh.centroids_lonlat) == n
        return [150.0 + 60.0 * sin((mesh.centroids_lonlat[s][1] + 63.0) * 5.0) +
                40.0 * cos((mesh.centroids_lonlat[s][2] - 44.0) * 5.0) for s in 1:n]
    else
        return fill(150.0, n)
    end
end

"""
    _is_graph_reachable(W::AbstractMatrix{<:Real}, start_node::Int, end_node::Int, max_k::Int) -> Bool

Performs breadth-first search (BFS) on the network adjacency graph `W` with self-transitions
permitted to verify whether `end_node` is reachable from `start_node` within `max_k` hops.
Returns `false` if `start_node` or `end_node` are isolated, severed, or separated by a barrier.
"""
function _is_graph_reachable(
    W::AbstractMatrix{<:Real},
    start_node::Int,
    end_node::Int,
    max_k::Int
)::Bool
    start_node == end_node && return true
    max_k <= 0 && return false
    S = size(W, 1)
    if !(1 <= start_node <= S) || !(1 <= end_node <= S)
        return false
    end

    W_sp = W isa SparseMatrixCSC ? W : sparse(W)
    visited = falses(S)
    frontier = Int[start_node]
    visited[start_node] = true

    for _ in 1:max_k
        next_frontier = Int[]
        for u in frontier
            col_start = W_sp.colptr[u]
            col_end   = W_sp.colptr[u + 1] - 1
            if col_start <= col_end
                for ptr in col_start:col_end
                    v = W_sp.rowval[ptr]
                    if v == end_node
                        return true
                    end
                    if !visited[v]
                        visited[v] = true
                        push!(next_frontier, v)
                    end
                end
            end
        end
        frontier = next_frontier
        isempty(frontier) && break
    end
    return visited[end_node]
end

# ─────────────────────────────────────────────────────────────────────────────
# Workflow Entry Point
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_movement_individual(; kwargs...) -> NamedTuple
    run_movement_individual(args::AbstractVector{<:AbstractString}; kwargs...) -> Union{NamedTuple, Nothing}
    run_movement_individual(arg_str::AbstractString; kwargs...) -> Union{NamedTuple, Nothing}

Executes the end-to-end individual animal movement modeling workflow using BSTM.
Supports invocation via keyword arguments, parsed CLI argument vectors (`ARGS`), or
space-delimited command-line flag strings.

# Mathematical Formulation
Individual animal telemetry positions are modeled as transitions across a discrete
spatial lattice over discrete elapsed time steps \$k\$:
```math
\\text{recapture}_k \\sim \\text{Categorical}\\left( [P_g^k]_{i, :} \\right)
```
where the biological group-specific transition kernel \$P_g\$ is defined by the
well-conditioned convex stochastic mixture:
```math
P_g = (1 - \\rho_g) \\left[ (1 - \\alpha_g) T_{\\text{diff}} + \\alpha_g A_g(\\eta) \\right] + \\rho_g I
```
with:
- \$\\rho_g \\in [0, 1]\$: Group residence (fidelity) probability.
- \$\\alpha_g \\in [0, 1]\$: Directed advection weighting relative to isotropic diffusion.
- \$T_{\\text{diff}} = \\text{row\\_normalize}(W)\$: Isotropic random-walk graph diffusion.
- \$A_g(\\eta)\$: Gradient-directed advection kernel toward higher habitat suitability \$\\eta\$.
- \$I\$: Identity matrix representing immediate local retention.

# Arguments & CLI Flags
- `data_source::Symbol`: `:simulate` (using `bstm_data("movement")`) or `:snowcrab`
  (using `snowcrab_movement_data()` if local data exists). Default `:simulate`.
  CLI: `-d, --data-source <source>`, `--simulate`, `--snowcrab`.
- `model_mode`: `"telemetry"` (pure categorical movement kernel),
  `"telemetry_and_survey"` (joint survey numerical density and movement model),
  or `"both"`. Default `"telemetry"`.
  CLI: `-m, --model-mode <mode>`, `--telemetry`, `--joint`, `--both`.
- `reshard_hex::Bool`: When `true`, reshard domain to a finer hexagonal lattice using
  LibGEOS before the modeling step (default false). CLI: `--reshard-hex`, `--hex`.
- `hex_radius_km::Real`: Cell radius for finer hexagonal lattice in km (default 10.0).
  CLI: `--hex-radius <km>`, `--hex-res <km>`.
- `use_hydrodynamics::Bool`: When `true`, ingests open bathymetry and extracts 3D/4D
  hydrodynamic model diagnostics (temperature, salinity stratification, advection,
  turbulent diffusion) before modeling (default false). CLI: `--hydro`, `--hydrodynamics`.
- `depth_range::Union{Nothing, Tuple{<:Real, <:Real}, AbstractVector{<:Real}, AbstractString}`:
  Optional `(min_depth, max_depth)` bathymetric bounds in meters. When specified,
  all spatial units with bathymetry outside \$[d_{\\min}, d_{\\max}]\$ are designated
  as impermeable barriers. All incoming and outgoing graph links are severed:
  ```math
  W_{il} = 0, \\quad W_{li} = 0 \\quad \\forall i \\in \\mathcal{S}, \\, l \\in \\mathcal{L}_{\\text{depth}}
  ```
  with habitat suitability set to zero (\$hsi_l = 0\$), guaranteeing zero traversal
  probability (\$P_{il} = 0, P_{li} = 0\$) across the out-of-range bathymetric domain.
  CLI: `--depth-range <min,max>`, `--depth-min <min>`, `--depth-max <max>`.
- `n_samples::Int`: Number of MCMC posterior draws (default 200).
  CLI: `-n, -s, --samples <N>`.
- `n_warmup::Int`: Number of MCMC warmup iterations (default 100).
  CLI: `-w, --warmup <N>`.
- `seed::Int`: Random seed for reproducibility (default 42).
  CLI: `--seed <N>`.
- `render_html::Bool`: Whether to export interactive HTML Leaflet maps (default true).
  CLI: `--render-html`, `--no-html`.
- `output_dir::String`: Directory to save visualization artifacts (default "output").
  CLI: `-o, --output-dir <path>`.
- `verbose::Bool`: Enable progress and summary console output (default true).
  CLI: `-v, --verbose`, `-q, --quiet`.

# Returns
A `NamedTuple` containing:
- `data`: Loaded or simulated dataset NamedTuple.
- `models`: Dictionary of compiled BSTM models (`:telemetry`, `:telemetry_and_survey`).
- `chains`: Dictionary of MCMC posterior sample chains.
- `P_kernel`: Group transition kernels.
- `paths`: Reconstructed individual Viterbi trajectories.
- `corridors`: Markov bridge migration corridor probability matrices.
- `parameters`: Posterior parameter summary (\$\\alpha\$, \$\\rho\$, \$\\gamma\$).
- `depth_range`: Parsed `(min_depth, max_depth)` barrier constraint tuple or `nothing`.
Or `nothing` if `--help` was requested.
"""
function run_movement_individual(;
    data_source       :: Symbol = :simulate,
    model_mode        :: Union{Symbol, String} = "telemetry",
    reshard_hex       :: Bool   = false,
    hex_radius_km     :: Real   = 10.0,
    use_hydrodynamics :: Bool   = false,
    depth_range       :: Union{Nothing, Tuple{<:Real, <:Real}, AbstractVector{<:Real}, AbstractString} = nothing,
    n_samples         :: Int    = 200,
    n_warmup          :: Int    = 100,
    seed              :: Int    = 42,
    render_html       :: Bool   = true,
    output_dir        :: String = "output",
    verbose           :: Bool   = true,
    kwargs...
)::NamedTuple

    # Check for depth_min / depth_max keyword arguments if depth_range is not given
    if depth_range === nothing
        d_min = get(kwargs, :depth_min, nothing)
        d_max = get(kwargs, :depth_max, nothing)
        if d_min !== nothing || d_max !== nothing
            depth_range = (d_min === nothing ? -Inf : Float64(d_min),
                           d_max === nothing ? Inf : Float64(d_max))
        end
    end

    rng = MersenneTwister(seed)
    mode_str = lowercase(string(model_mode))
    fit_telemetry = mode_str in ("telemetry", "both")
    fit_joint     = mode_str in ("telemetry_and_survey", "both")

    verbose && println("=========================================================")
    verbose && println("    BSTM Individual Animal Movement & Telemetry Pipeline ")
    verbose && println("=========================================================\n")

    # ── 1. Data Ingestion ────────────────────────────────────────────────────
    verbose && println("[Step 1] Ingesting movement dataset (source: :$data_source)...")

    data = if data_source == :snowcrab
        if isdefined(Main, :snowcrab_movement_data)
            Base.invokelatest(snowcrab_movement_data; verbose=verbose)
        else
            @warn "snowcrab_movement_data.jl not found; falling back to :simulate."
            bstm_data("movement")
        end
    else
        bstm_data("movement")
    end

    mesh        = data.mesh
    W           = data.W
    hsi_vec     = data.hsi_vec
    obs_df      = data.obs
    survey_df   = hasproperty(data, :survey_df) ? data.survey_df : nothing
    group_map   = hasproperty(data, :group_lookup) ? data.group_lookup : Dict(1 => "All")
    land_mask   = hasproperty(data, :land_mask) ? data.land_mask : nothing
    n_spatial   = mesh.n_units

    verbose && println("  - Spatial mesh units: $n_spatial")
    if land_mask !== nothing
        verbose && println("    * Marine water units: $(count(!, land_mask))")
        verbose && println("    * Terrestrial land barrier units: $(sum(land_mask))")
    end
    verbose && println("  - Mark-recapture events: $(nrow(obs_df))")
    verbose && println("  - Biological groups: $(length(group_map)) ($(join(keys(group_map), ", ")))")

    # Variables to hold resharded hydrodynamic dataset and bathymetry depths
    resharded_hydro  = nothing
    resharded_depths = nothing

    # ── 1b. LibGEOS Hexagonal Resharding & Hydrodynamics (Before Modeling) ────
    if reshard_hex || use_hydrodynamics
        verbose && println("\n[Step 1b] Resharding domain to fine hexagonal resolution using LibGEOS...")

        cents_lon = [Float64(c[1]) for c in mesh.centroids_lonlat]
        cents_lat = [Float64(c[2]) for c in mesh.centroids_lonlat]
        min_lon, max_lon = extrema(cents_lon)
        min_lat, max_lat = extrema(cents_lat)

        # 1. Ingest / synthesize bathymetry
        bathy = load_open_bathymetry(;
            lon_range = (min_lon - 0.2, max_lon + 0.2),
            lat_range = (min_lat - 0.2, max_lat + 0.2),
            resolution_deg = 0.08
        )

        # 2. Extract 3D/4D hydrodynamic diagnostics tied to seafloor bathymetry
        hydro = extract_hydrodynamic_dataset(bathy;
            depth_levels = [0.0, 25.0, 50.0, 100.0, 175.0],
            times = [1.0]
        )

        # 3. Construct finer regular hexagonal mesh
        fine_mesh = build_hex_mesh_planar(
            bathy.grid_lon, bathy.grid_lat;
            radius_km = Float64(hex_radius_km)
        )

        verbose && println("  - Source bathymetric units: $(length(bathy.au.polygons))")
        verbose && println("  - Finer regular hexagonal units: $(fine_mesh.n_units) " *
                           "(radius = $(hex_radius_km) km)")

        # 4. Compute LibGEOS polygon area-weighted transfer matrix
        verbose && println("  - Computing LibGEOS polygon intersection transfer matrix...")
        P_transfer = compute_network_transfer_matrix(bathy.au, fine_mesh; method=:area_weighted)

        # 5. Reshard hydrodynamic fields onto fine hexagonal lattice
        resharded_hydro = reshard_spatial_field(P_transfer, hydro)
        resharded_depths = reshard_spatial_field(P_transfer, bathy.depths)

        # 6. Remap mark-recapture / telemetry observations to fine hexagons
        obs_df = copy(obs_df)
        fine_cents = fine_mesh.centroids_lonlat

        orig_rel_cents = [mesh.centroids_lonlat[r] for r in obs_df.release]
        orig_rec_cents = [mesh.centroids_lonlat[r] for r in obs_df.recapture]
        obs_df.release = map_to_units(
            [c[1] for c in orig_rel_cents], [c[2] for c in orig_rel_cents], fine_cents
        )
        obs_df.recapture = map_to_units(
            [c[1] for c in orig_rec_cents], [c[2] for c in orig_rec_cents], fine_cents
        )

        # Remap survey observations if present
        if !isnothing(survey_df) && hasproperty(survey_df, :s_idx)
            survey_df = copy(survey_df)
            orig_surv_cents = [mesh.centroids_lonlat[s] for s in survey_df.s_idx]
            survey_df.s_idx = map_to_units(
                [c[1] for c in orig_surv_cents], [c[2] for c in orig_surv_cents], fine_cents
            )
            survey_df.depth = resharded_depths[survey_df.s_idx]
        end

        # Switch active modeling mesh to the finer hexagonal lattice
        mesh = fine_mesh
        W = fine_mesh.W
        land_mask = identify_land_units(fine_mesh.centroids_lonlat; depth=resharded_depths)
        W, hsi_vec = apply_land_barrier(fine_mesh.W, resharded_hydro.hsi, land_mask)
        n_spatial = fine_mesh.n_units

        verbose && println("  - Resharding complete. Movement model will execute on fine hexagons!")
    end

    # ── 1c. Depth Range Traversal Barrier Enforcement ────────────────────────
    parsed_depth_range = if depth_range isa AbstractString
        parts = occursin(":", depth_range) ? Base.split(depth_range, ":") : Base.split(depth_range, ",")
        if length(parts) == 2
            (parse(Float64, strip(parts[1])), parse(Float64, strip(parts[2])))
        else
            throw(ArgumentError("Invalid depth_range string '$depth_range'. Expected 'min,max' or 'min:max'."))
        end
    elseif depth_range isa Union{Tuple, AbstractVector} && length(depth_range) >= 2
        (Float64(depth_range[1]), Float64(depth_range[2]))
    else
        nothing
    end

    min_d, max_d = 0.0, 0.0
    if parsed_depth_range !== nothing
        min_d, max_d = extrema(parsed_depth_range)
        verbose && println("\n[Step 1c] Enforcing depth barrier constraint: [$(min_d), $(max_d)] m...")

        depths_vec = _extract_domain_depths(mesh, data, resharded_depths)
        out_of_depth = BitVector([d < min_d || d > max_d for d in depths_vec])
        n_barred = count(out_of_depth)
        n_allowed = length(depths_vec) - n_barred

        verbose && println("  - Allowed units within depth range: $(n_allowed) / $(length(depths_vec))")
        verbose && println("  - Barred units outside depth range: $(n_barred) / $(length(depths_vec))")

        # Combine with existing terrestrial land barrier mask
        combined_barrier = land_mask !== nothing ? BitVector(land_mask .| out_of_depth) : out_of_depth

        # Sever network graph links and zero out habitat suitability
        W, hsi_vec = apply_land_barrier(W, hsi_vec, combined_barrier)
        land_mask = combined_barrier

        # Filter telemetry observations: both endpoints must be unbarred and graph-reachable
        n_obs_orig = nrow(obs_df)
        valid_obs = [
            !combined_barrier[r.release] &&
            !combined_barrier[r.recapture] &&
            _is_graph_reachable(W, r.release, r.recapture, r.k)
            for r in eachrow(obs_df)
        ]
        obs_df = obs_df[valid_obs, :]
        n_obs_filtered = nrow(obs_df)
        n_dropped = n_obs_orig - n_obs_filtered

        if n_dropped > 0
            verbose && println("  - Filtered $(n_dropped) telemetry observation(s) outside depth corridor.")
            verbose && println("  - Remaining active mark-recapture events: $(n_obs_filtered)")
        end

        if n_obs_filtered == 0
            error("Depth barrier constraint [$min_d, $max_d] m excluded all mark-recapture telemetry observations. " *
                  "Please specify a wider depth range containing active telemetry tracks.")
        end

        # Filter survey observations if present
        if !isnothing(survey_df) && hasproperty(survey_df, :s_idx)
            valid_surv = [!combined_barrier[s] for s in survey_df.s_idx]
            survey_df = survey_df[valid_surv, :]
        end
    end

    models = Dict{Symbol, Any}()
    chains = Dict{Symbol, Any}()

    # ── 2. Pure Telemetry Model (No Dummy Outcomes) ──────────────────────────
    if fit_telemetry
        verbose && println("\n[Step 2] Fitting Pure Telemetry Model (mode: \"telemetry\")...")
        verbose && println("  Likelihood: recapture ~ Categorical(P_g^k[release, :])")
        verbose && println("  Kernel: P_g = (1-ρ)[(1-α)T_diff + α A_g] + ρ I")

        # No y_dummy, no fake intercept, no artificial Gaussian sigma
        m_telemetry = @bstm(
            likelihood(recapture, family=categorical_movement) ~
                movement(
                    release   = release,
                    k         = k,
                    group     = group,
                    W         = W,
                    habitat   = hsi_vec,
                    land_mask = land_mask,
                    method    = :stochastic_kernel,
                    velocity  = truncated(Normal(0.3, 0.2), 0.0, 0.95),
                    diffusion = truncated(Normal(0.1, 0.2), 0.0, Inf),
                    gamma     = Normal(1.0, 1.0)
                ),
            data = obs_df,
            verbose = false
        )
        models[:telemetry] = m_telemetry

        verbose && println("  Sampling $n_samples posterior draws via Turing MH...")
        chn_telemetry = Base.invokelatest(sample, rng, m_telemetry, MH(), n_samples; progress=false)
        chains[:telemetry] = chn_telemetry
        verbose && println("  Pure telemetry model sampling complete.")
    end

    # ── 3. Joint Survey Density + Telemetry Model ────────────────────────────
    if fit_joint && !isnothing(survey_df)
        verbose && println("\n[Step 3] Fitting Joint Survey Density + Telemetry Model (mode: \"telemetry_and_survey\")...")
        verbose && println("  Survey Likelihood: density ~ NegBin(exp(η_s), r)")
        verbose && println("  Dynamic HSI: η_s directly drives directed advection A_g(η)!")

        m_joint = @bstm(
            likelihood(density, family=negbin) ~
                intercept() +
                fixed(depth) +
                random(s_idx, t_idx, model=movement,
                    method         = :density_gradient,
                    telemetry_data = obs_df,
                    W              = W,
                    velocity       = truncated(Normal(0.3, 0.2), 0.0, 0.95),
                    diffusion      = truncated(Normal(0.1, 0.2), 0.0, Inf),
                    gamma          = Normal(1.0, 1.0)
                ),
            data = survey_df,
            verbose        = false
        )
        models[:telemetry_and_survey] = m_joint

        verbose && println("  Sampling $n_samples posterior draws via Turing MH...")
        chn_joint = Base.invokelatest(sample, rng, m_joint, MH(), n_samples; progress=false)
        chains[:telemetry_and_survey] = chn_joint
        verbose && println("  Joint telemetry and survey model sampling complete.")
    end

    # ── 4. Parameter Extraction & Transition Kernel Construction ─────────────
    verbose && println("\n[Step 4] Extracting posterior parameters and constructing transition kernel...")
    active_chain = haskey(chains, :telemetry) ? chains[:telemetry] : chains[:telemetry_and_survey]

    # Invert group lookup for integer-keyed indexing and display
    grp_name_lookup = Dict{Int, String}()
    for (k, v) in group_map
        if k isa String && v isa Integer
            grp_name_lookup[v] = k
        elseif k isa Integer && v isa String
            grp_name_lookup[k] = v
        else
            grp_name_lookup[Int(v)] = string(k)
        end
    end
    G = isempty(grp_name_lookup) ? 1 : maximum(keys(grp_name_lookup))
    chn_keys = keys(active_chain)

    # Helper function to extract posterior mean per group
    function _extract_mean_vector(prefix::String, G_count::Int, default_val::Float64)
        vals = fill(default_val, G_count)
        # 1. Attempt to find element-indexed parameters like prefix[1], prefix[2], etc.
        found_indexed = false
        for g in 1:G_count
            k_match = filter(
                k -> occursin(prefix, string(k)) && (G_count == 1 || occursin("[$g]", string(k))),
                chn_keys
            )
            if !isempty(k_match)
                raw = Array(active_chain[first(k_match)])
                m = if eltype(raw) <: AbstractVector ||
                       (length(raw) > 0 && first(raw) isa AbstractVector)
                    mean(raw)
                elseif raw isa AbstractMatrix{<:Real}
                    vec(mean(raw, dims=1))
                elseif raw isa AbstractVector{<:Real}
                    mean(raw)
                else
                    mean(raw)
                end
                if m isa AbstractVector && length(m) >= g
                    vals[g] = Float64(m[g])
                elseif m isa AbstractVector && !isempty(m)
                    vals[g] = Float64(m[1])
                elseif m isa Real
                    vals[g] = Float64(m)
                end
                found_indexed = true
            end
        end

        # 2. If per-element keys were not found, search for whole vector parameter
        if !found_indexed || all(v -> v == default_val, vals)
            k_fallback = filter(k -> occursin(prefix, string(k)), chn_keys)
            if !isempty(k_fallback)
                raw = Array(active_chain[first(k_fallback)])
                if eltype(raw) <: AbstractVector || (length(raw) > 0 && first(raw) isa AbstractVector)
                    m = mean(raw)
                    if m isa AbstractVector && length(m) == G_count
                        vals .= Float64.(m)
                    elseif m isa AbstractVector && length(m) == 1
                        vals .= fill(Float64(m[1]), G_count)
                    end
                elseif raw isa AbstractMatrix{<:Real}
                    if size(raw, 2) == G_count
                        vals .= vec(mean(raw, dims=1))
                    elseif size(raw, 1) == G_count
                        vals .= vec(mean(raw, dims=2))
                    else
                        vals .= fill(Float64(mean(raw)), G_count)
                    end
                elseif raw isa AbstractVector{<:Real}
                    vals .= fill(Float64(mean(raw)), G_count)
                else
                    m = mean(raw)
                    if m isa AbstractVector && length(m) == G_count
                        vals .= Float64.(m)
                    elseif m isa Real
                        vals .= fill(Float64(m), G_count)
                    end
                end
            end
        end
        return vals
    end

    v_mean = _extract_mean_vector("velocity", G, 0.3)
    d_mean = _extract_mean_vector("diffusion", G, 0.1)
    g_mean = _extract_mean_vector("gamma", G, 1.0)

    tot = v_mean .+ d_mean .+ 1e-6
    alpha_hat = clamp.(v_mean ./ tot, 0.0, 1.0)
    rho_hat   = clamp.(1.0 ./ (1.0 .+ tot), 0.01, 0.95)

    verbose && println("  Posterior Parameter Summary ($G biological group$(G > 1 ? "s" : "")):")
    for g in 1:G
        grp_lbl = get(grp_name_lookup, g, "Group $g")
        verbose && println("    - Group $g ($grp_lbl): " *
                           "α=$(round(alpha_hat[g], digits=4)), " *
                           "ρ=$(round(rho_hat[g], digits=4)), " *
                           "γ=$(round(g_mean[g], digits=4))")
    end

    P_kernel = construct_stochastic_transition_kernel(
        W, hsi_vec;
        gamma      = g_mean,
        residence  = rho_hat,
        advection  = alpha_hat,
        land_mask  = land_mask
    )
    if P_kernel isa AbstractVector
        verbose && println("  Constructed $(length(P_kernel)) group transition kernels " *
                           "(each $(size(first(P_kernel))), Row-stochastic: true)")
    else
        verbose && println("  Constructed P_kernel size: $(size(P_kernel)) (Row-stochastic: true)")
    end

    # ── 5. Most Likely Path & Migration Corridor Reconstruction ───────────────
    verbose && println("\n[Step 5] Reconstructing individual movement paths and corridors...")
    
    # Select example tracked individuals from observation data
    sample_tags = unique(obs_df.tagid)[1:min(5, length(unique(obs_df.tagid)))]
    reconstructed_paths = Dict{String, Vector{Int}}()
    reconstructed_corridors = Dict{String, Matrix{Float64}}()

    for tid in sample_tags
        sub_obs = filter(:tagid => ==(tid), obs_df)
        if isempty(sub_obs)
            continue
        end
        # Identify biological group for this individual
        grp = hasproperty(sub_obs, :group) ? first(sub_obs.group) : 1
        P_k = P_kernel isa AbstractVector ?
            (1 <= grp <= length(P_kernel) ? P_kernel[grp] : first(P_kernel)) :
            P_kernel

        # Combine consecutive segments
        full_tag_path = Int[sub_obs.release[1]]
        for row in eachrow(sub_obs)
            seg_path = predict_path(
                P_k, row.release, row.recapture, row.k; land_mask=land_mask
            )
            # Append intermediate and destination units
            append!(full_tag_path, seg_path[2:end])
        end
        reconstructed_paths[tid] = full_tag_path

        # First segment corridor heatmap
        first_row = first(sub_obs)
        corridor_mat = predict_corridor(
            P_k, first_row.release, first_row.recapture, first_row.k; land_mask=land_mask
        )
        reconstructed_corridors[tid] = corridor_mat
        
        grp_label = haskey(grp_name_lookup, grp) ? " [group $(grp): $(grp_name_lookup[grp])]" : ""
        verbose && println("  Tag $tid$grp_label: Viterbi trajectory length = $(length(full_tag_path)) units " *
                           "(from $(first(full_tag_path)) to $(last(full_tag_path)))")
    end

    # ── 6. Spatial Visualization ─────────────────────────────────────────────
    if render_html
        verbose && println("\n[Step 6] Generating interactive Leaflet path visualization...")
        mkpath(output_dir)

        if hasproperty(mesh, :centroids_lonlat) || hasproperty(mesh, :centroids)
            cents_ll = hasproperty(mesh, :centroids_lonlat) ? mesh.centroids_lonlat : mesh.centroids
            polys_ll = hasproperty(mesh, :polygons_lonlat) ? mesh.polygons_lonlat :
                       (hasproperty(mesh, :polygons) ? mesh.polygons : nothing)
            au_mesh = (
                centroids = cents_ll,
                centroids_lonlat = cents_ll,
                polygons = polys_ll,
                polygons_lonlat = polys_ll,
                W = W
            )
            all_paths_vec = collect(values(reconstructed_paths))

            html_file = joinpath(output_dir, "movement_paths_dashboard.html")
            try
                # Render tracks on the finer hexagonal mesh
                map_obj = leaflet_tracks_map(
                    all_paths_vec,
                    au_mesh;
                    hsi   = hsi_vec,
                    title = "Individual Movement Trajectories & Migration Corridors" *
                            (reshard_hex || use_hydrodynamics ? " (Fine Hexagonal Mesh)" : "") *
                            (parsed_depth_range !== nothing ?
                             " [Depth: $(round(Int, min_d))-$(round(Int, max_d)) m]" : "")
                )
                save_html(map_obj, html_file)
                verbose && println("  Interactive Leaflet movement dashboard saved to: $html_file")
            catch e
                verbose && println("  (Leaflet rendering note: $e - saving ASCII summary instead)")
            end

            # When hydrodynamic resharding was used, render the hydrodynamic dashboard on fine hexagons
            if !isnothing(resharded_hydro)
                try
                    hydro_html = joinpath(output_dir, "hydrodynamic_hex_dashboard.html")
                    dash = leaflet_hydrodynamic_dashboard(
                        resharded_hydro,
                        au_mesh;
                        title = "Scotian Shelf Hydrodynamics & Stratification (Fine Hexagons)"
                    )
                    save_html(dash, hydro_html)
                    verbose && println("  Interactive Hydrodynamic Hexagonal dashboard saved to: $hydro_html")
                catch e
                    verbose && println("  (Hydrodynamic dashboard rendering note: $e)")
                end
            end
        end
    end

    verbose && println("\n=========================================================")
    verbose && println("    Pipeline Completed Successfully! ")
    verbose && println("=========================================================\n")

    return (
        data        = data,
        models      = models,
        chains      = chains,
        P_kernel    = P_kernel,
        paths       = reconstructed_paths,
        corridors   = reconstructed_corridors,
        parameters  = (alpha=alpha_hat, residence=rho_hat, gamma=g_mean),
        depth_range = parsed_depth_range
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Command-Line Interface (CLI) & Flag Parser
# ─────────────────────────────────────────────────────────────────────────────

"""
    print_movement_help()

Prints command-line usage instructions, available flag options, and invocation
examples for `run_movement_individual`.
"""
function print_movement_help()
    println("""
BSTM Individual Animal Movement & Telemetry Pipeline
=====================================================
Usage:
  julia --project=. docs/movement/movement_individual.jl [OPTIONS]

Options:
  -d, --data-source <source>   Dataset: 'simulate' (synthetic) or 'snowcrab' (real data).
      --simulate, --sim        Shorthand for --data-source=simulate.
      --snowcrab               Shorthand for --data-source=snowcrab.
  -m, --model-mode <mode>      Model mode: 'telemetry' (default), 'telemetry_and_survey',
                               or 'both'.
      --telemetry              Shorthand for --model-mode=telemetry.
      --joint, --survey        Shorthand for --model-mode=telemetry_and_survey.
      --both                   Shorthand for --model-mode=both.
      --reshard-hex, --hex     Reshard domain to finer regular hexagons with LibGEOS.
      --hex-radius <km>        Cell radius in km for fine hexagons (default: 10.0).
      --hydro                  Extract 3D hydrodynamic fields and bathymetry.
      --depth-range <min,max>  Force all movement within depth range in meters (e.g. '50,300').
                               Sever all network graph links outside range (P = 0 traversal).
      --depth-min <min>        Minimum traversal depth in meters.
      --depth-max <max>        Maximum traversal depth in meters.
  -n, -s, --samples <N>        Number of MCMC posterior draws (default: 200).
  -w, --warmup <N>             Number of MCMC warmup iterations (default: 100).
      --seed <N>               Random number generator seed (default: 42).
      --render-html, --html    Export interactive Leaflet HTML dashboard (default: true).
      --no-html                Disable HTML map export.
  -o, --output-dir <path>      Output directory for generated maps (default: 'output').
  -v, --verbose                Enable detailed progress logging (default: true).
  -q, --quiet, --silent        Suppress non-essential progress output.
  -h, --help                   Display this help message and exit.

Examples:
  julia --project=. docs/movement/movement_individual.jl --help
  julia --project=. docs/movement/movement_individual.jl --simulate --samples=100
  julia --project=. docs/movement/movement_individual.jl --reshard-hex --hex-radius=8.0
  julia --project=. docs/movement/movement_individual.jl --depth-range 50,250
  julia --project=. docs/movement/movement_individual.jl --reshard-hex --depth-range 75,300
  julia --project=. docs/movement/movement_individual.jl --snowcrab --mode=telemetry -n 100
  julia --project=. docs/movement/movement_individual.jl -d simulate -m both -o output/sim
""")
    return nothing
end

"""
    _is_cli_flag(token::AbstractString) -> Bool

Determines whether a command-line token represents a flag switch rather than a
negative numerical parameter value (e.g., distinguishing `-m` or `--seed` from `-1.5`).
"""
function _is_cli_flag(token::AbstractString)::Bool
    if startswith(token, "--")
        return true
    elseif startswith(token, "-")
        return tryparse(Float64, token) === nothing
    end
    return false
end

"""
    parse_movement_cli_args(args::AbstractVector{<:AbstractString}=ARGS) -> Dict{Symbol, Any}

Parses command-line argument strings into a typed options dictionary for
`run_movement_individual`. Supports standard UNIX long (`--flag`, `--flag=val`,
`--flag val`) and short (`-f`, `-f val`) syntax.

# Arguments
- `args::AbstractVector{<:AbstractString}`: Collection of CLI token strings.
  Defaults to top-level `ARGS`.

# Returns
- `Dict{Symbol, Any}`: Dictionary mapping keyword option symbols to parsed values.
"""
function parse_movement_cli_args(
    args::AbstractVector{<:AbstractString} = ARGS
)::Dict{Symbol, Any}
    opts = Dict{Symbol, Any}()
    idx = 1
    total_args = length(args)

    while idx <= total_args
        raw_token = args[idx]
        key = raw_token
        inline_val = nothing
        has_inline = false

        if startswith(raw_token, "-") && occursin("=", raw_token)
            token_parts = Base.split(raw_token, "=", limit=2)
            key = token_parts[1]
            inline_val = token_parts[2]
            has_inline = true
        end

        # Helper closure to extract value either from inline '=' or next token
        fetch_val = () -> begin
            if has_inline
                return inline_val
            elseif idx + 1 <= total_args && !_is_cli_flag(args[idx + 1])
                idx += 1
                return args[idx]
            else
                throw(ArgumentError("CLI flag '$key' requires an argument value."))
            end
        end

        if key in ("--help", "-h")
            opts[:help] = true
        elseif key in ("--snowcrab", "--crab")
            opts[:data_source] = :snowcrab
        elseif key in ("--simulate", "--sim")
            opts[:data_source] = :simulate
        elseif key in ("--data-source", "--data", "-d")
            raw_src = lowercase(fetch_val())
            if raw_src in ("snowcrab", "crab", "real")
                opts[:data_source] = :snowcrab
            elseif raw_src in ("simulate", "sim", "synthetic")
                opts[:data_source] = :simulate
            else
                opts[:data_source] = Symbol(raw_src)
            end
        elseif key == "--telemetry"
            opts[:model_mode] = "telemetry"
        elseif key in ("--joint", "--survey", "--telemetry-and-survey")
            opts[:model_mode] = "telemetry_and_survey"
        elseif key in ("--both", "--all")
            opts[:model_mode] = "both"
        elseif key in ("--model-mode", "--mode", "-m")
            raw_mode = lowercase(fetch_val())
            if raw_mode in ("telemetry", "pure", "pure_telemetry")
                opts[:model_mode] = "telemetry"
            elseif raw_mode in ("telemetry_and_survey", "joint", "survey")
                opts[:model_mode] = "telemetry_and_survey"
            elseif raw_mode in ("both", "all")
                opts[:model_mode] = "both"
            else
                opts[:model_mode] = raw_mode
            end
        elseif key in ("--reshard-hex", "--hex", "--reshard")
            if has_inline
                opts[:reshard_hex] = !(lowercase(inline_val) in ("false", "0", "no", "off"))
            else
                opts[:reshard_hex] = true
            end
        elseif key in ("--hex-radius", "--hex-res", "--radius-km")
            opts[:hex_radius_km] = parse(Float64, fetch_val())
        elseif key in ("--hydro", "--hydrodynamics")
            if has_inline
                opts[:use_hydrodynamics] = !(lowercase(inline_val) in ("false", "0", "no", "off"))
            else
                opts[:use_hydrodynamics] = true
            end
        elseif key in ("--depth-range", "--depths", "--depth-corridor", "--depth")
            val_str = fetch_val()
            parts = occursin(":", val_str) ? Base.split(val_str, ":") : Base.split(val_str, ",")
            if length(parts) == 2
                opts[:depth_range] = (parse(Float64, strip(parts[1])),
                                      parse(Float64, strip(parts[2])))
            else
                throw(ArgumentError("Invalid --depth-range format: '$val_str'. Expected 'min,max' or 'min:max'."))
            end
        elseif key in ("--depth-min", "--min-depth")
            opts[:depth_min] = parse(Float64, fetch_val())
        elseif key in ("--depth-max", "--max-depth")
            opts[:depth_max] = parse(Float64, fetch_val())
        elseif key in ("--n-samples", "--samples", "-n", "-s")
            opts[:n_samples] = parse(Int, fetch_val())
        elseif key in ("--n-warmup", "--warmup", "-w")
            opts[:n_warmup] = parse(Int, fetch_val())
        elseif key == "--seed"
            opts[:seed] = parse(Int, fetch_val())
        elseif key in ("--render-html", "--html")
            if has_inline
                opts[:render_html] = !(lowercase(inline_val) in ("false", "0", "no", "off"))
            else
                opts[:render_html] = true
            end
        elseif key in ("--no-render-html", "--no-html")
            opts[:render_html] = false
        elseif key in ("--output-dir", "--output", "-o")
            opts[:output_dir] = String(fetch_val())
        elseif key in ("--verbose", "-v")
            if has_inline
                opts[:verbose] = !(lowercase(inline_val) in ("false", "0", "no", "off"))
            else
                opts[:verbose] = true
            end
        elseif key in ("--quiet", "-q", "--silent")
            opts[:verbose] = false
        else
            @warn "Unrecognized CLI flag: '$raw_token' - ignoring."
        end

        idx += 1
    end

    # Merge individual depth_min / depth_max flags into depth_range if not already specified
    if haskey(opts, :depth_min) || haskey(opts, :depth_max)
        if !haskey(opts, :depth_range)
            d_min = get(opts, :depth_min, -Inf)
            d_max = get(opts, :depth_max, Inf)
            opts[:depth_range] = (d_min, d_max)
        end
        delete!(opts, :depth_min)
        delete!(opts, :depth_max)
    end

    return opts
end

"""
    run_movement_individual(args::AbstractVector{<:AbstractString}; kwargs...) -> Union{NamedTuple, Nothing}
    run_movement_individual(arg_str::AbstractString; kwargs...) -> Union{NamedTuple, Nothing}

Parses command-line arguments and flags, merges them with any keyword argument
overrides, and executes the individual movement modeling pipeline.

If `-h` or `--help` is supplied, displays command-line help text and returns `nothing`.

# Mathematical Formulation
The telemetry likelihood evaluates transition between spatial nodes \$i\$ and \$j\$
over time interval \$k\$:
```math
\\text{recapture}_k \\sim \\text{Categorical}\\left( [P_g^k]_{i, :} \\right)
```
with stochastic transition matrix:
```math
P_g = (1 - \\rho_g) \\left[ (1 - \\alpha_g) T_{\\text{diff}} + \\alpha_g A_g(\\eta) \\right] + \\rho_g I
```
where \$\\rho_g \\in [0, 1]\$ is residency probability, \$\\alpha_g \\in [0, 1]\$ is
habitat advection velocity weighting, and \$A_g(\\eta)\$ is the gradient-directed
advection kernel driven by habitat suitability \$\\eta\$.

# Inputs
- `args`: Vector of command-line argument strings, or a single space-delimited string.
- `kwargs`: Optional keyword overrides that take precedence over CLI flags.

# Outputs
A `NamedTuple` containing:
- `data`: Loaded or simulated dataset.
- `models`: Dictionary of compiled BSTM models.
- `chains`: Dictionary of MCMC posterior sample chains.
- `P_kernel`: Group transition kernels.
- `paths`: Reconstructed individual Viterbi trajectories.
- `corridors`: Markov bridge corridor probability matrices.
- `parameters`: Posterior parameter summary (\$\\alpha\$, \$\\rho\$, \$\\gamma\$).
- `depth_range`: Parsed `(min_depth, max_depth)` barrier constraint tuple or `nothing`.
Or `nothing` if `--help` was requested.
"""
function run_movement_individual(
    args::AbstractVector{<:AbstractString};
    kwargs...
)::Union{NamedTuple, Nothing}
    cli_opts = parse_movement_cli_args(args)
    if get(cli_opts, :help, false)
        print_movement_help()
        return nothing
    end
    # Explicit keyword arguments take precedence over parsed CLI options
    merged_opts = merge(cli_opts, Dict{Symbol, Any}(kwargs))
    return run_movement_individual(; merged_opts...)
end

function run_movement_individual(
    arg_str::AbstractString;
    kwargs...
)::Union{NamedTuple, Nothing}
    return run_movement_individual(Base.split(arg_str); kwargs...)
end

# ─────────────────────────────────────────────────────────────────────────────
# Standalone Script Execution
# ─────────────────────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    run_movement_individual(ARGS)
end
