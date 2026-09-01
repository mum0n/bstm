"""
    movement_individual.jl

Main executable for categorical/multinomial mark-recapture movement modelling.

Consolidates the logic from `movement_simple.jl`, `snowcrab.jl`, and
`snowcrab_movement.jl` into a single, self-contained workflow using the BSTM
framework and Turing.jl.

# Model

For each sex × maturity group g ∈ {F_mature, F_immature, M_mature, M_immature}:

    A_g[i,j] ∝ W_ij exp(γ_g (HSI_j − HSI_i))   [directed adjacency, row-normalised]
    L         = D_deg − W_sym                      [graph Laplacian]
    Γ̄_g       = (I − β_g A_g − D_g L)^{-1}       [resolvent transition operator]
    p_i^{(k)} = row_i( Γ̄_g^k )                   [k-step probabilities]
    recapture  ~ Categorical(p_i^{(k)})            [observation likelihood]

Priors:
    β_g  ~ Truncated(Normal(0.2, 0.2), 0, 0.95)
    D_g  ~ Truncated(Normal(0.1, 0.2), 0, ∞)
    γ_g  ~ Normal(1.0, 1.0)

The `k` for each event is computed from the elapsed time between consecutive
detections and the chosen `time_interval` unit (per event, not globally fixed).

# CLI usage

    julia --project . docs/movement/movement_individual.jl \\
        --hsi-path     docs/movement/hsi.jld2 \\
        --telemetry-path docs/movement/telemetry.csv \\
        --output-path  docs/movement/output.jld2 \\
        --n-samples    1000  --n-warmup    500  --n-chains  4 \\
        --method       hmc   --n-thin      1    --n-burn-in 200 \\
        --n-discrete-steps 10  --rng-seed  1234 \\
        --verbose --no-progress-bars

    # Self-test with simulated data (no input files required):
    julia --project . docs/movement/movement_individual.jl --simulate

# Dependencies (via Project.toml)
DataFrames, Dates, Distributions, JLD2, LinearAlgebra, NearestNeighbors,
Random, SparseArrays, Statistics, Turing, MCMCChains.
"""
 

"""
    fit_categorical_movement(
        obs, S, W, hsi, group_lookup;
        n_samples=1000, n_warmup=500, n_chains=4, n_thin=1,
        target_acceptance=0.65, rng_seed=42, show_progress=true
    ) -> NamedTuple

Fit a hierarchical categorical movement model via NUTS (No-U-Turn Sampler),
stratified by sex × maturity groups.
"""
function fit_categorical_movement(
    obs          :: DataFrame,
    S            :: Int,
    W            :: SparseMatrixCSC,
    hsi          :: AbstractVector{Float64},
    group_lookup :: Dict{String, Int};
    n_samples         :: Int     = 1000,
    n_warmup          :: Int     = 500,
    n_chains          :: Int     = 4,
    n_thin            :: Int     = 1,
    target_acceptance :: Float64 = 0.65,
    rng_seed          :: Int     = 42,
    show_progress     :: Bool    = true
)::NamedTuple

    G        = length(group_lookup)
    releases = Vector{Int}(obs.release)
    recaps   = Vector{Int}(obs.recapture)
    ks       = Vector{Int}(obs.k)
    groups   = Vector{Int}(obs.group)
    
    # Pre-calculate the maximum 'k' needed to bound our caching array
    max_k    = maximum(ks)

    # 1. Instantly extract neighbors using CSC pointers (no findnz slicing)
    adj_rows = [W.rowval[W.colptr[i]:W.colptr[i+1]-1] for i in 1:S]
    
    # 2. Precompute L and I_S as dense Float64 matrices outside the model
    W_sym   = Array(max.(W, W'))
    deg     = vec(sum(W_sym, dims=2))
    L_dense = Matrix{Float64}(Diagonal(deg) - W_sym)
    I_S     = Matrix{Float64}(I, S, S)

    @model function cat_movement(releases, recaps, ks, groups, S, G, adj_rows, L_dense, I_S, hsi, max_k)
        # Group-level priors
        beta  ~ filldist(truncated(Normal(0.2, 0.2), 0.0, 0.95), G)
        D_g   ~ filldist(truncated(Normal(0.1, 0.2), 0.0, Inf),  G)
        gamma ~ filldist(Normal(1.0, 1.0),                       G)

        # 3. Precompute transition matrices (Strictly functional for AD)
        Gk_cache = map(1:G) do g
            A_g  = _build_A_ad(adj_rows, hsi, gamma[g], S)
            
            M    = I_S .- beta[g] .* A_g .- D_g[g] .* L_dense
            Graw = inv(M)
            Gamma_1 = _row_normalise(Graw, S)
            
            # AD-safe (non-mutating) power sequence generation
            # accumulate passes the result of the previous step forward
            if max_k > 1
                higher_powers = accumulate(2:max_k; init=Gamma_1) do prev_Gamma, _
                    _row_normalise(prev_Gamma * Gamma_1, S)
                end
                return vcat([Gamma_1], higher_powers)
            else
                return [Gamma_1]
            end
        end

        # 4. Per-event likelihood 
        N = length(releases)
        for n in 1:N
            rel = releases[n]
            rec = recaps[n]  # MUST be observed data (Integer), not `missing`
            g   = groups[n]
            k_n = ks[n]
            
            p = Gk_cache[g][k_n][rel, :]
            
            ps = sum(p)
            # Ensure fallback type matches eltype of `p` to prevent AD crashes
            T_el = eltype(p)
            p_norm = ps > eps(T_el) ? (p ./ ps) : fill(one(T_el) / S, S)
            
            rec ~ Categorical(p_norm)
        end
    end


    model_obj = cat_movement(releases, recaps, ks, groups, S, G, adj_rows, L_dense, I_S, hsi, max_k)
    sampler   = NUTS(n_warmup, target_acceptance)
    rng       = MersenneTwister(rng_seed)

    chain = if n_chains > 1
        Turing.sample(rng, model_obj, sampler, MCMCThreads(), n_samples, n_chains;
            thinning=n_thin, progress=show_progress)
    else
        Turing.sample(rng, model_obj, sampler, n_samples;
            thinning=n_thin, progress=show_progress)
    end

    # Posterior-mean transition matrices
    inv_lookup = Dict(v => k for (k, v) in group_lookup)
    trans_mats = Dict{String, Matrix{Float64}}()
    
    for g in 1:G
        # Support for both older MCMCChains syntax and newer ones
        beta_mean  = mean(Array(chain["beta[$g]"]))
        D_mean     = mean(Array(chain["D_g[$g]"]))
        gamma_mean = mean(Array(chain["gamma[$g]"]))
        
        A_g = compute_directed_adjacency(hsi, W; gamma=gamma_mean)
        trans_mats[inv_lookup[g]] = resolvent_transition(beta_mean, D_mean, A_g, L_dense, S)
    end

    return (
        chain               = chain,
        group_lookup        = group_lookup,
        transition_matrices = trans_mats
    )
end


# ── Main workflow ──────────────────────────────────────────────────────────────

"""
    run_movement_individual(; data_kwargs..., model_kwargs...) -> NamedTuple

End-to-end workflow: load data, fit model, return results.

Keyword arguments are forwarded to `prepare_movement_data` (data_kwargs) and
`fit_categorical_movement` (model_kwargs).  See those docstrings for the full
argument list.

# Additional keyword arguments
- `output_path::String`: Path to save results as JLD2 (default: no save).
- `verbose::Bool`: Print pipeline progress (default `false`).

# Returns
`NamedTuple` with fields `data`, `result`, and `model_kwargs`.
"""
function run_movement_individual(;
    # data arguments
    tagging = nothing,
    hsi = nothing,
    data_source   :: Symbol  = :snowcrab,
    data_dir      :: AbstractString = "data",
    tagging_file  :: Union{Nothing, AbstractString} = nothing,
    hsi_path      :: Union{Nothing, AbstractString} = nothing,
    telemetry_csv :: Union{Nothing, AbstractString} = nothing,
    radius_km     :: Real    = 5.0,
    time_interval :: Symbol  = :monthly,
    filter_dead   :: Bool    = true,
    proj_zone     :: Int     = 20,
    domain_km     :: Real    = 200.0,
    n_tags_sim    :: Int     = 100,
    n_steps_sim   :: Int     = 5,
    seed          :: Int     = 42,
    ref_doy       :: Real    = 244.0,
    # model arguments
    n_samples         :: Int     = 1000,
    n_warmup          :: Int     = 500,
    n_chains          :: Int     = 4,
    n_thin            :: Int     = 1,
    n_burn_in         :: Int     = 0,
    target_acceptance :: Float64 = 0.65,
    rng_seed          :: Int     = 42,
    show_progress     :: Bool    = true,
    # output
    output_path :: Union{Nothing, AbstractString} = nothing,
    verbose     :: Bool = false
)::NamedTuple

    verbose && println("=== Movement Individual Workflow ===")

    if data_source ==  :simulate
        data = prepare_movement_data(;
            tagging       = tagging,
            hsi           = hsi,
            data_source   = data_source,
            data_dir      = data_dir,
            tagging_file  = tagging_file,
            hsi_path      = hsi_path,
            telemetry_csv = telemetry_csv,
            radius_km     = radius_km,
            time_interval = time_interval,
            filter_dead   = filter_dead,
            proj_zone     = proj_zone,
            domain_km     = domain_km,
            n_tags        = n_tags_sim,
            n_steps       = n_steps_sim,
            seed          = seed,
            ref_doy       = ref_doy,
            verbose       = verbose
        )
    end

    # ── Data ──────────────────────────────────────────────────────────────────
    nrow(data.obs) > 0 || error("No mark-recapture event pairs found after filtering.")

    verbose && println("  Groups: $(length(unique(data.obs.sex .* \"_\" .* data.obs.mat)))")

    # ── Group stratification ───────────────────────────────────────────────────
    obs_grouped, group_lookup = build_group_indices(data.obs; sex_col=:sex, mat_col=:mat)
    verbose && println("  Group lookup: $(group_lookup)")

    # ── Fit model ──────────────────────────────────────────────────────────────
    verbose && println("  Fitting categorical movement model …")

    # NUTS adaptation = warmup + burn_in (both are pre-collection phases)
    adapt_steps = n_warmup + n_burn_in

    result = fit_categorical_movement(
        obs_grouped, data.mesh.n_units, data.mesh.W,
        data.hsi_vec, group_lookup;
        n_samples         = n_samples,
        n_warmup          = adapt_steps,
        n_chains          = n_chains,
        n_thin            = n_thin,
        target_acceptance = target_acceptance,
        rng_seed          = rng_seed,
        show_progress     = show_progress
    )
    verbose && println("  Sampling complete.")

    # ── Save results ───────────────────────────────────────────────────────────
    if !isnothing(output_path)
        
        # save the following into bstm data structures:
        # Chain
        # Group lookup
        # Transition matrices
        # Mesh centroids and W
        # HSI
        # Tagging
        # Observations
        # BSTM format
        # chain                = result.chain,
        # group_lookup         = result.group_lookup,
        # transition_matrices  = result.transition_matrices,
        # mesh_centroids_lonlat = data.mesh.centroids_lonlat,
        # mesh_W               = data.mesh.W,
        # hsi_vec              = data.hsi_vec,
        # tagging              = data.tagging,
        # obs                  = data.obs    
        # etc

    return (data=data, result=result)
end
 


# ── CLI entry point ────────────────────────────────────────────────────────────

"""
    main(args=ARGS) -> Any

Command-line interface for `movement_individual.jl`.

# Flags and options
    --data-source      snowcrab|simulate|csv   (default: snowcrab)
    --data-dir         PATH                    (default: docs/movement/data)
    --tagging-path     PATH                    (JLD2, snowcrab source)
    --hsi-path         PATH                    (JLD2, HSI file)
    --telemetry-path   PATH                    (CSV, for :csv source)
    --output-path      PATH                    (JLD2 output)
    --radius-km        FLOAT                   (hex circumradius, default 5.0)
    --time-interval    monthly|weekly|daily     (default: monthly)
    --n-samples        INT                     (default 1000)
    --n-warmup         INT                     (default 500)
    --n-burn-in        INT                     (extra discard, default 0)
    --n-chains         INT                     (default 4)
    --n-thin           INT                     (thinning, default 1)
    --n-discrete-steps INT                     (minimum k per event, default 1)
    --method           hmc                     (maps to NUTS; only option)
    --n-print-every    INT                     (unused; for compat)
    --n-progress-every INT                     (unused; for compat)
    --rng-seed         INT                     (default 42)
    --verbose          (flag)
    --debug            (flag, enables verbose + extra logging)
    --no-progress-bars (flag, hides Turing progress)
    --simulate         (shorthand for --data-source simulate)
"""
function main(args=ARGS)
    # ── Defaults ──────────────────────────────────────────────────────────────
    opts = Dict{Symbol, Any}(
        :tagging           => nothing,
        :hsi               => nothing,
        :data_source       => :snowcrab,
        :data_dir          => "docs/movement/data",
        :tagging_path      => nothing,
        :hsi_path          => nothing,
        :telemetry_path    => nothing,
        :output_path       => nothing,
        :radius_km         => 5.0,
        :time_interval     => :monthly,
        :n_samples         => 1000,
        :n_warmup          => 500,
        :n_burn_in         => 0,
        :n_chains          => 4,
        :n_thin            => 1,
        :n_discrete_steps  => 1,
        :method            => :hmc,
        :rng_seed          => 42,
        :verbose           => false,
        :debug             => false,
        :show_progress     => true
    )

    # ── Argument parsing ───────────────────────────────────────────────────────
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--simulate"
            opts[:data_source] = :simulate
        elseif a == "--verbose"
            opts[:verbose] = true
        elseif a == "--debug"
            opts[:debug]   = true
            opts[:verbose] = true
        elseif a == "--no-progress-bars"
            opts[:show_progress] = false
        elseif a == "--data-source"
            opts[:data_source] = Symbol(args[i += 1])
        elseif a == "--data-dir"
            opts[:data_dir] = args[i += 1]
        elseif a == "--tagging-path"
            opts[:tagging_path] = args[i += 1]
        elseif a == "--hsi-path"
            opts[:hsi_path] = args[i += 1]
        elseif a == "--telemetry-path"
            opts[:telemetry_path] = args[i += 1]
        elseif a == "--output-path"
            opts[:output_path] = args[i += 1]
        elseif a == "--radius-km"
            opts[:radius_km] = parse(Float64, args[i += 1])
        elseif a == "--time-interval"
            opts[:time_interval] = Symbol(args[i += 1])
        elseif a == "--n-samples"
            opts[:n_samples] = parse(Int, args[i += 1])
        elseif a == "--n-warmup"
            opts[:n_warmup] = parse(Int, args[i += 1])
        elseif a == "--n-burn-in"
            opts[:n_burn_in] = parse(Int, args[i += 1])
        elseif a == "--n-chains"
            opts[:n_chains] = parse(Int, args[i += 1])
        elseif a == "--n-thin"
            opts[:n_thin] = parse(Int, args[i += 1])
        elseif a == "--n-discrete-steps"
            opts[:n_discrete_steps] = parse(Int, args[i += 1])
        elseif a == "--method"
            opts[:method] = Symbol(args[i += 1])   # hmc → NUTS
        elseif a == "--rng-seed"
            opts[:rng_seed] = parse(Int, args[i += 1])
        elseif a in ("--n-print-every", "--n-progress-every")
            i += 1   # accepted but ignored
        else
            @warn "Unknown argument: $(a)"
        end
        i += 1
    end

    verbose = opts[:verbose]
    verbose && begin
        println("movement_individual.jl — options:")
        for (k, v) in sort(collect(opts), by=x->string(x[1]))
            println("  $(k) = $(v)")
        end
    end

    # ── Run workflow ───────────────────────────────────────────────────────────
    run_movement_individual(;
        tagging       = opts[:tagging],
        hsi           = opts[:hsi],
        data_source   = opts[:data_source],
        data_dir      = opts[:data_dir],
        tagging_file  = opts[:tagging_path],
        hsi_path      = opts[:hsi_path],
        telemetry_csv = opts[:telemetry_path],
        radius_km     = opts[:radius_km],
        time_interval = opts[:time_interval],
        n_samples     = opts[:n_samples],
        n_warmup      = opts[:n_warmup],
        n_burn_in     = opts[:n_burn_in],
        n_chains      = opts[:n_chains],
        n_thin        = opts[:n_thin],
        rng_seed      = opts[:rng_seed],
        show_progress = opts[:show_progress],
        output_path   = opts[:output_path],
        verbose       = verbose
    )
end

# ── Script entrypoint ──────────────────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
