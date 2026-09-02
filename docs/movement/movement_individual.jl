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
""""""
    run_movement_individual(; kwargs...)

Executes the end-to-end individual animal movement modeling workflow, handling data loading or simulation,
group stratification, Bayesian model specification, MCMC sampling via Turing.jl, and optional output persistence.
"""
function run_movement_individual(;
    # Data arguments
    tagging           = nothing,
    hsi               = nothing,
    data_source       :: Symbol         = :snowcrab,
    data_dir          :: AbstractString = "data",
    tagging_file      :: Union{Nothing, AbstractString} = nothing,
    hsi_path          :: Union{Nothing, AbstractString} = nothing,
    telemetry_csv     :: Union{Nothing, AbstractString} = nothing,
    radius_km         :: Real           = 5.0,
    time_interval     :: Symbol         = :monthly,
    filter_dead       :: Bool           = true,
    proj_zone         :: Int            = 20,
    domain_km         :: Real           = 200.0,
    n_tags_sim        :: Int            = 100,
    n_steps_sim       :: Int            = 5,
    seed              :: Int            = 42,
    ref_doy           :: Real           = 244.0,
    # Model arguments
    n_samples         :: Int            = 1000,
    n_warmup          :: Int            = 500,
    n_chains          :: Int            = 4,
    n_thin            :: Int            = 1,
    n_burn_in         :: Int            = 0,
    target_acceptance :: Float64        = 0.65,
    rng_seed          :: Int            = 42,
    show_progress     :: Bool           = true,
    output_path       :: Union{Nothing, AbstractString} = nothing,
    verbose           :: Bool           = false
)::NamedTuple

    verbose && println("=== Movement Individual Workflow ===")

    # ── 1. Data Initialization / Simulation ──────────────────────────────────
    data = if data_source == :simulate
        generate_movement_data(
            radius_km     = radius_km,
            time_interval = time_interval,
            domain_km     = domain_km,
            n_tags        = n_tags_sim,
            n_steps       = n_steps_sim,
            seed          = seed
        )
    elseif data_source == :snowcrab
        snowcrab_movement_data(
            radius_km     = radius_km,
            time_interval = time_interval,
            ref_doy       = Int(ref_doy),
            verbose       = verbose
        )
    else
        error("Unsupported data_source: $data_source")
    end

    nrow(data.obs) > 0 || error("No mark-recapture event pairs found after filtering.")

    verbose && println("  Groups: $(length(unique(data.obs.sex .* \"_\" .* data.obs.mat)))")

    # ── 2. Group Stratification ───────────────────────────────────────────────
    n_rows = nrow(data.obs) 
    labels = Vector{String}(undef, n_rows)
    
    sexes = data.obs[!, :sex]
    mats  = data.obs[!, :mat]

    @inbounds for i in 1:n_rows
        sx = string(sexes[i])
        mt = string(mats[i])

        if mt == "immature"
            labels[i] = "immature"
        elseif mt == "mature" && sx == "M"
            labels[i] = "male"
        elseif mt == "mature" && sx == "F"
            labels[i] = "female"
        else
            labels[i] = "unknown"
        end
    end
    
    unique_labels = sort!(unique(labels))
    group_lookup  = Dict{String, Int}(lbl => i for (i, lbl) in enumerate(unique_labels))
        
    group_ids = Vector{Int}(undef, n_rows)
    @inbounds for i in 1:n_rows
        group_ids[i] = group_lookup[labels[i]]
    end

    data.obs[!, :group] = group_ids
    verbose && println("  Group lookup: $(group_lookup)")

    # ── 3. Model Preparation ──────────────────────────────────────────────────
    verbose && println("  Fitting categorical movement model …")

    adapt_steps = n_warmup + n_burn_in
    effective_warmup = adapt_steps

    obs = data.obs
    S   = data.mesh.n_units
    W   = data.mesh.W
    hsi = data.hsi_vec

    G        = length(group_lookup)
    releases = Vector{Int}(obs.release)
    recaps   = Vector{Int}(obs.recapture)
    ks       = Vector{Int}(obs.k)
    groups   = Vector{Int}(obs.group)
    max_k    = maximum(ks)

    adj_rows = [W.rowval[W.colptr[i]:W.colptr[i+1]-1] for i in 1:S]
    
    W_sym   = Array(max.(W, W'))
    deg     = vec(sum(W_sym, dims=2))
    L_dense = Matrix{Float64}(Diagonal(deg) - W_sym)
    I_S     = Matrix{Float64}(I, S, S)

    @model function cat_movement(releases, recaps, ks, groups, S, G, adj_rows, L_dense, I_S, hsi, max_k)
        beta  ~ filldist(truncated(Normal(0.2, 0.2), 0.0, 0.95), G)
        D_g   ~ filldist(truncated(Normal(0.1, 0.2), 0.0, Inf),  G)
        gamma ~ filldist(Normal(1.0, 1.0),                       G)

        Gk_cache = map(1:G) do g
            A_g  = _build_A_ad(adj_rows, hsi, gamma[g], S)
            M    = I_S .- beta[g] .* A_g .- D_g[g] .* L_dense
            Graw = inv(M)
            Gamma_1 = _row_normalise(Graw, S)
            
            if max_k > 1
                higher_powers = accumulate(2:max_k; init=Gamma_1) do prev_Gamma, _
                    _row_normalise(prev_Gamma * Gamma_1, S)
                end
                return vcat([Gamma_1], higher_powers)
            else
                return [Gamma_1]
            end
        end

        N = length(releases)
        for n in 1:N
            rel = releases[n]
            rec = recaps[n]
            g   = groups[n]
            k_n = ks[n]
            
            p = Gk_cache[g][k_n][rel, :]
            ps = sum(p)
            T_el = eltype(p)
            p_norm = ps > eps(T_el) ? (p ./ ps) : fill(one(T_el) / S, S)
            
            rec ~ Categorical(p_norm)
        end
    end

    model_obj = cat_movement(releases, recaps, ks, groups, S, G, adj_rows, L_dense, I_S, hsi, max_k)
    sampler   = NUTS(effective_warmup, target_acceptance)
    rng       = MersenneTwister(rng_seed)

    chain = if n_chains > 1
        Turing.sample(rng, model_obj, sampler, MCMCThreads(), n_samples, n_chains;
            thinning=n_thin, progress=show_progress)
    else
        Turing.sample(rng, model_obj, sampler, n_samples;
            thinning=n_thin, progress=show_progress)
    end

    inv_lookup = Dict(v => k for (k, v) in group_lookup)
    trans_mats = Dict{String, Matrix{Float64}}()
    
    for g in 1:G
        beta_mean  = mean(Array(chain["beta[$g]"]))
        D_mean     = mean(Array(chain["D_g[$g]"]))
        gamma_mean = mean(Array(chain["gamma[$g]"]))
        
        A_g = compute_directed_adjacency(hsi, W; gamma=gamma_mean)
        trans_mats[inv_lookup[g]] = resolvent_transition(beta_mean, D_mean, A_g, L_dense, S)
    end

    result = (
        chain               = chain,
        group_lookup        = group_lookup,
        transition_matrices = trans_mats
    )

    # ── 4. Save Results (Optional) ───────────────────────────────────────────
    if !isnothing(output_path)
        # Bundle persistence layer integration point (e.g. JLD2 or DuckDB)
    end

    return (data = data, result = result)
end