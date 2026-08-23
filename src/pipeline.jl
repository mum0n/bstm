"""
    pipeline.jl

Declarative Modular DAG Pipeline Orchestrator (`bstm_pipeline`), Multi-Tier
Spatial Network Resharding, Uncertainty Propagation (EIV), and Relational
Database Harmonization for Bayesian Spatio-Temporal Models.

# Mathematical Foundation & Multi-Tier Resharding
In hierarchical ecological workflows (e.g. Bathymetry -> Substrate -> Oceanography
-> Multi-Species -> Focal Biomass), different tiers represent physical or biological
processes operating at distinct observational scales, data densities, and geometries:
- **Tier 1 (Continuous Bathymetry)**: Point soundings modeled via continuous `RFF` or `SpectralGP`.
- **Tier 2 (Sediment Properties)**: Point core grabs discretized onto a coarse graph \$G_2 = (V_2, W_2)\$.
- **Tier 3 (Hydrography)**: Space-time CTD ocean casts on regular grid \$G_3 = (V_3, W_3)\$.
- **Tier 4 (Multispecies Assemblage)**: Trawl surveys on stratified stratum mesh \$G_4 = (V_4, W_4)\$.
- **Tier 5 (Target Species / Focal Assessment)**: Management area or fine master mesh \$G_{\\text{master}}\$.

### Information Transfer & Network Resharding:
1. **Continuous-to-Discrete Evaluation**:
   For continuous models (\$f(\\mathbf{s})\$), the posterior mean and variance are evaluated
   directly at the target network's coordinates \$\\mathbf{s}_j \\in V_{\\text{dest}}\$:
   \$\\hat{z}(\\mathbf{s}_j) = \\mathbb{E}[f(\\mathbf{s}_j) \\mid \\mathcal{D}], \\quad \\hat{\\sigma}^2(\\mathbf{s}_j) = \\text{Var}(f(\\mathbf{s}_j) \\mid \\mathcal{D})\$
2. **Discrete Mesh-to-Mesh Transfer (Area / Distance Weighting)**:
   For fields defined on areal units \$V_{\\text{src}} = \\{A_1, \\dots, A_m\\}\$ transferred to
   \$V_{\\text{dest}} = \\{B_1, \\dots, B_n\\}\$, the transfer matrix \$P \\in \\mathbb{R}^{n \\times m}\$ satisfies:
   \$P_{j, i} = \\frac{\\text{Area}(B_j \\cap A_i)}{\\text{Area}(B_j)}, \\quad \\sum_i P_{j, i} = 1\$
   \$\\mathbf{u}_{\\text{dest}} = P \\mathbf{u}_{\\text{src}}, \\quad \\text{Var}(\\mathbf{u}_{\\text{dest}}) = P \\text{Cov}(\\mathbf{u}_{\\text{src}}) P^\\top\$
3. **Master Network Harmonization**:
   Upon pipeline completion, all upstream physical, sediment, oceanographic, and community
   covariates are resharded and joined onto \$G_{\\text{master}}\$, populating DuckDB relational
   tables and exporting GIS GeoJSON files.

Version: v1.0.0
"""

# ==============================================================================
# SECTION 1: SPATIAL NETWORK RESHARDING & TRANSFER OPERATORS
# ==============================================================================

"""
    compute_network_transfer_matrix(au_src::NamedTuple, au_dest::NamedTuple)

Computes the spatial interpolation / resharding matrix \$P \\in \\mathbb{R}^{n_{\\text{dest}} \\times n_{\\text{src}}}\$
between two areal unit spatial partitions.

Uses polygon geometric area-overlap weighting when available, falling back to
k-nearest inverse-distance-weighted (IDW) centroid mapping.
"""
function compute_network_transfer_matrix(au_src::NamedTuple, au_dest::NamedTuple)
    n_src = length(au_src.centroids)
    n_dest = length(au_dest.centroids)

    if n_src == n_dest && au_src.centroids == au_dest.centroids
        return spdiagm(0 => ones(Float64, n_dest))
    end

    # 1. Try Geometric Polygon Overlap if LibGEOS polygons are available
    has_polys = hasproperty(au_src, :polygons) && hasproperty(au_dest, :polygons) &&
                !isempty(au_src.polygons) && !isempty(au_dest.polygons)

    if has_polys
        try
            P = spzeros(Float64, n_dest, n_src)
            src_lg_polys = [LibGEOS.Polygon([[[pt[1], pt[2]] for pt in p]]) for p in au_src.polygons]
            dest_lg_polys = [LibGEOS.Polygon([[[pt[1], pt[2]] for pt in p]]) for p in au_dest.polygons]

            for j in 1:n_dest
                dest_p = dest_lg_polys[j]
                dest_area = get_polygon_area(au_dest.polygons[j])
                if dest_area <= 1e-9
                    continue
                end

                total_overlap = 0.0
                for i in 1:n_src
                    src_p = src_lg_polys[i]
                    if LibGEOS.intersects(dest_p, src_p)
                        inter_geom = LibGEOS.intersection(dest_p, src_p)
                        if !LibGEOS.isEmpty(inter_geom)
                            inter_area = get_polygon_area(get_coords_from_geom(inter_geom))
                            if inter_area > 1e-9
                                w = inter_area / dest_area
                                P[j, i] = w
                                total_overlap += w
                            end
                        end
                    end
                end

                # Normalize row if partial overlap
                if total_overlap > 1e-6
                    P[j, :] ./= total_overlap
                end
            end

            # If all rows have valid weights, return geometric transfer matrix
            if all(sum(P, dims=2) .> 0.5)
                return P
            end
        catch
            # Fall back to centroid IDW if geometry intersection fails
        end
    end

    # 2. Centroid Inverse Distance Weighting (k-d Tree Fallback)
    src_coords = hcat([[c[1], c[2]] for c in au_src.centroids]...)
    dest_coords = hcat([[c[1], c[2]] for c in au_dest.centroids]...)

    kdtree = KDTree(src_coords)
    k_nn = min(4, n_src)
    idxs, dists = knn(kdtree, dest_coords, k_nn, true)

    P = spzeros(Float64, n_dest, n_src)
    for j in 1:n_dest
        cur_idxs = idxs[j]
        cur_dists = dists[j]

        if cur_dists[1] <= 1e-6
            P[j, cur_idxs[1]] = 1.0
        else
            weights = 1.0 ./ (cur_dists .^ 2)
            weights ./= sum(weights)
            for (idx, w) in zip(cur_idxs, weights)
                P[j, idx] = w
            end
        end
    end

    return P
end

"""
    summarize_sample_matrix(samples::AbstractMatrix{<:Real}; alpha::Real=0.05)

Computes posterior empirical summary statistics (mean, median, std, lower/upper quantiles)
across Monte Carlo sample columns for each spatial unit row.
"""
function summarize_sample_matrix(samples::AbstractMatrix{<:Real}; alpha::Real=0.05)
    n_units, n_samples = size(samples)
    means = vec(Statistics.mean(samples, dims=2))
    stds = vec(Statistics.std(samples, dims=2))
    lowers = zeros(Float64, n_units)
    uppers = zeros(Float64, n_units)
    medians = zeros(Float64, n_units)
    for i in 1:n_units
        row_vals = view(samples, i, :)
        lowers[i] = quantile(row_vals, alpha / 2.0)
        uppers[i] = quantile(row_vals, 1.0 - alpha / 2.0)
        medians[i] = median(row_vals)
    end
    return (
        mean = means,
        median = medians,
        std = stds,
        lower = lowers,
        upper = uppers,
        samples = Matrix{Float64}(samples)
    )
end

"""
    reshard_spatial_field(values::AbstractArray, au_src::NamedTuple, au_dest::NamedTuple)
    reshard_spatial_field(summary::NamedTuple, au_src::NamedTuple, au_dest::NamedTuple; mode::Symbol=:samples, alpha::Real=0.05)

Reshards a spatial field from source network `au_src` to destination network `au_dest`.

# Modes
- **Full Monte Carlo Matrix Resharding** (`mode=:samples` or when passing a 2D matrix `[N_src × S]`):
  Transfers all posterior draws directly via linear transfer operator:
  \$U_{\\text{dest}} = P U_{\\text{src}} \\in \\mathbb{R}^{N_{\\text{dest}} \\times S}\$
  and computes empirical non-Gaussian credible intervals without variance distortion.
- **Simplistic Moment-Matching Resharding** (`mode=:moments`):
  Transfers only the first two moments:
  \$\\mathbf{u}_{\\text{dest}} = P \\mathbf{u}_{\\text{src}}, \\quad \\boldsymbol{\\sigma}_{\\text{dest}} = \\sqrt{P \\boldsymbol{\\sigma}^2_{\\text{src}}}\$
"""
function reshard_spatial_field(
    values::AbstractArray{<:Real}, au_src::NamedTuple, au_dest::NamedTuple
)
    P = compute_network_transfer_matrix(au_src, au_dest)
    if ndims(values) == 1
        return Vector{Float64}(P * values)
    else
        return Matrix{Float64}(P * values)
    end
end

function reshard_spatial_field(
    summary::NamedTuple, au_src::NamedTuple, au_dest::NamedTuple;
    mode::Symbol=:samples, alpha::Real=0.05
)
    P = compute_network_transfer_matrix(au_src, au_dest)

    # 1. Full Monte Carlo Matrix Resharding (if sample matrix is available and requested)
    if mode in [:samples, :full_mc, :matrix] && hasproperty(summary, :samples) &&
       !isnothing(summary.samples) && summary.samples isa AbstractMatrix
        U_dest = Matrix{Float64}(P * summary.samples)
        return summarize_sample_matrix(U_dest; alpha=alpha)
    end

    # 2. Simplistic Moment-Matching Resharding (Fallback or Explicit Mode)
    mean_resharded = Vector{Float64}(P * summary.mean)
    sd_resharded = Vector{Float64}(sqrt.(P * (summary.std .^ 2)))
    lower_resharded = Vector{Float64}(P * summary.lower)
    upper_resharded = Vector{Float64}(P * summary.upper)
    median_resharded = hasproperty(summary, :median) ?
        Vector{Float64}(P * summary.median) : mean_resharded

    return (
        mean = mean_resharded,
        median = median_resharded,
        std = sd_resharded,
        lower = lower_resharded,
        upper = upper_resharded,
        samples = nothing
    )
end

# ==============================================================================
# SECTION 2: DECLARATIVE PIPELINE ORCHESTRATOR ENGINE
# ==============================================================================

"""
    PipelineTierSpec

Specification for a single hierarchical modeling tier in `bstm_pipeline`.
"""
struct PipelineTierSpec
    name::Symbol
    formula::String
    data::DataFrame
    au::Union{NamedTuple, Nothing}
    W::Union{AbstractMatrix, Nothing}
    sampler::Any
    n_samples::Int
    derivatives::Vector{Symbol}
    radii::Vector{Float64}
    eiv_inputs::Vector{Symbol}
    options::Dict{Symbol, Any}
end

function PipelineTierSpec(
    name::Symbol, conf::NamedTuple
)
    formula = string(get(conf, :formula, ""))
    data = get(conf, :data, DataFrame())
    au = get(conf, :au, nothing)
    W = get(conf, :W, nothing)
    sampler = get(conf, :sampler, NUTS(20, 0.65))
    n_samples = get(conf, :n_samples, 50)
    derivatives = get(conf, :derivatives, Symbol[])
    radii = get(conf, :radii, Float64[10.0, 20.0])
    eiv_inputs = get(conf, :eiv_inputs, Symbol[])
    
    opts = Dict{Symbol, Any}()
    for (k, v) in pairs(conf)
        if !(k in [:formula, :data, :au, :W, :sampler, :n_samples, :derivatives, :radii, :eiv_inputs])
            opts[k] = v
        end
    end

    return PipelineTierSpec(
        name, formula, data, au, W, sampler, n_samples, derivatives, radii, eiv_inputs, opts
    )
end

"""
    PipelineResult

Complete execution artifacts, multi-tier models, chains, relational DuckDB database,
and harmonized master datasets produced by `bstm_pipeline`.
"""
struct PipelineResult
    tier_names::Vector{Symbol}
    models::Dict{Symbol, DynamicPPL.Model}
    chains::Dict{Symbol, Any}
    results::Dict{Symbol, Any}
    derivatives::Dict{Symbol, Any}
    tier_networks::Dict{Symbol, Union{NamedTuple, Nothing}}
    master_au::NamedTuple
    master_summary::DataFrame
    duckdb_path::String
end

# ==============================================================================
# SECTION 3: PUBLIC API (`bstm_pipeline`)
# ==============================================================================

"""
    bstm_pipeline(tiers...; master_au=nothing, duckdb_path=":memory:", jld2_path=nothing, geojson_path=nothing, verbose=true)

Executes a multi-tier Bayesian Spatio-Temporal Model (BSTM) hierarchical pipeline
with automated network resharding, Errors-in-Variables (EIV) uncertainty propagation,
DuckDB relational persistence, and master network harmonization.

# Arguments
- `tiers...`: Pairs of `:tier_name => (formula=..., data=..., [au=...], [W=...], ...)`
- `master_au::NamedTuple`: The master spatial network / partition for final harmonization.
  If `nothing`, defaults to the network of the final tier.
- `duckdb_path::String`: File path to DuckDB database (or `":memory:"`).
- `jld2_path::Union{String, Nothing}`: Optional file path to persist complete pipeline state.
- `geojson_path::Union{String, Nothing}`: Optional file path to export harmonized master GIS GeoJSON.
- `verbose::Bool`: Whether to print progress diagnostics (default: `true`).

# Returns
A `PipelineResult` object containing all models, chains, derivatives, DuckDB tables, and
the unified master summary DataFrame.
"""
function bstm_pipeline(
    tiers::Pair{Symbol, <:NamedTuple}...;
    master_au::Union{NamedTuple, Nothing}=nothing,
    duckdb_path::AbstractString=":memory:",
    jld2_path::Union{AbstractString, Nothing}=nothing,
    geojson_path::Union{AbstractString, Nothing}=nothing,
    verbose::Bool=true
)
    tier_specs = [PipelineTierSpec(p.first, p.second) for p in tiers]
    n_tiers = length(tier_specs)

    if n_tiers == 0
        error("bstm_pipeline requires at least one modeling tier.")
    end

    models = Dict{Symbol, DynamicPPL.Model}()
    chains = Dict{Symbol, Any}()
    results = Dict{Symbol, Any}()
    derivatives = Dict{Symbol, Any}()
    tier_networks = Dict{Symbol, Union{NamedTuple, Nothing}}()
    tier_predictions = Dict{Symbol, DataFrame}()
    tier_unit_predictions = Dict{Symbol, Any}()

    if verbose
        println("\n==============================================================================")
        println("  BSTM HIERARCHICAL MODULAR DAG PIPELINE ORCHESTRATOR")
        println("  Tiers: $(join([string(t.name) for t in tier_specs], " -> "))")
        println("  DuckDB: $(duckdb_path)")
        println("==============================================================================\n")
    end

    # --- Sequential Execution Across Tiers ---
    for (t_idx, spec) in enumerate(tier_specs)
        t_name = spec.name
        if verbose
            println("--- [Tier $(t_idx)/$(n_tiers)]: $(uppercase(string(t_name))) ---")
            println("  Formula: $(spec.formula)")
        end

        cur_df = copy(spec.data)
        cur_au = spec.au
        cur_W = spec.W

        # Auto-resolve network if W not explicitly supplied but au is available
        if isnothing(cur_W) && !isnothing(cur_au) && hasproperty(cur_au, :W)
            cur_W = cur_au.W
        end

        # 1. Propagate Upstream Covariates (EIV & Predictions)
        for (prev_name, prev_pred_df) in tier_predictions
            # Check if any column or tier name is referenced in formula
            val_col = Symbol("$(prev_name)_mean")
            sd_col = Symbol("$(prev_name)_sd")

            if hasproperty(prev_pred_df, val_col) && (
                occursin(string(prev_name), spec.formula) || 
                occursin(string(val_col), spec.formula)
            )
                # Map upstream prediction to current tier dataframe
                prev_spec = tier_specs[findfirst(s -> s.name == prev_name, tier_specs)]
                prev_au = tier_networks[prev_name]

                # Match coordinates or spatial units
                if hasproperty(cur_df, :s_x) && hasproperty(cur_df, :s_y) && 
                   hasproperty(prev_pred_df, :s_x) && hasproperty(prev_pred_df, :s_y)
                    # Coordinate-based nearest neighbor / interpolation mapping
                    kdt = KDTree(hcat(prev_pred_df.s_x, prev_pred_df.s_y)')
                    query_pts = hcat(cur_df.s_x, cur_df.s_y)'
                    nns, _ = knn(kdt, query_pts, 1)
                    nn_idxs = [n[1] for n in nns]
                    cur_df[!, prev_name] = prev_pred_df[!, val_col][nn_idxs]
                    cur_df[!, Symbol("$(prev_name)_se")] = prev_pred_df[!, sd_col][nn_idxs]
                elseif hasproperty(cur_df, :s_idx) && !isnothing(cur_au) && !isnothing(prev_au)
                    # Mesh-to-mesh resharding transfer
                    if haskey(tier_unit_predictions, prev_name)
                        resharded = reshard_spatial_field(
                            tier_unit_predictions[prev_name], prev_au, cur_au; mode=:samples
                        )
                        cur_df[!, prev_name] = resharded.mean[cur_df.s_idx]
                        cur_df[!, Symbol("$(prev_name)_se")] = resharded.std[cur_df.s_idx]
                    else
                        P_trans = compute_network_transfer_matrix(prev_au, cur_au)
                        u_means = [mean(prev_pred_df[!, val_col][prev_pred_df.s_idx .== i]) for i in 1:prev_au.n_units]
                        u_sds   = [mean(prev_pred_df[!, sd_col][prev_pred_df.s_idx .== i]) for i in 1:prev_au.n_units]
                        unit_means = Vector{Float64}(P_trans * u_means)
                        unit_sds = Vector{Float64}(sqrt.(P_trans * (u_sds .^ 2)))
                        cur_df[!, prev_name] = unit_means[cur_df.s_idx]
                        cur_df[!, Symbol("$(prev_name)_se")] = unit_sds[cur_df.s_idx]
                    end
                end
            end
        end

        # 2. Fit Model & Sample Posterior
        m_tier = if !isnothing(cur_W)
            bstm_core(spec.formula, cur_df, @__MODULE__; W=cur_W, verbose=false)
        else
            bstm_core(spec.formula, cur_df, @__MODULE__; verbose=false)
        end

        chn_tier = Base.invokelatest(sample, m_tier, spec.sampler, spec.n_samples; progress=false)
        res_tier = Base.invokelatest(model_results_comprehensive, m_tier, chn_tier; alpha=0.05)

        models[t_name] = m_tier
        chains[t_name] = chn_tier
        results[t_name] = res_tier
        tier_networks[t_name] = cur_au

        # 3. Compute Surface Derivatives if requested
        if !isempty(spec.derivatives)
            coords_query = if hasproperty(cur_df, :s_x) && hasproperty(cur_df, :s_y)
                cur_df[:, [:s_x, :s_y]]
            elseif !isnothing(cur_au)
                DataFrame(s_x = [c[1] for c in cur_au.centroids], s_y = [c[2] for c in cur_au.centroids])
            else
                nothing
            end

            if !isnothing(coords_query)
                deriv_res = Base.invokelatest(
                    bstm_surface_derivatives,
                    m_tier, chn_tier, coords_query;
                    metrics=spec.derivatives,
                    radii=spec.radii,
                    return_samples=true
                )
                derivatives[t_name] = deriv_res
            end
        end

        # 4. Generate Predictions DataFrame for Downstream Sharing
        pred_df = copy(cur_df)
        pred_res = Base.invokelatest(predict, m_tier, chn_tier, cur_df)
        pred_df[!, Symbol("$(t_name)_mean")] = pred_res.predictions_denoised.mean
        pred_df[!, Symbol("$(t_name)_sd")] = pred_res.predictions_denoised.std
        tier_predictions[t_name] = pred_df

        # Unit-level predictions on the tier's spatial network
        if !isnothing(cur_au)
            df_units = DataFrame(
                s_idx = 1:cur_au.n_units,
                s_x = [c[1] for c in cur_au.centroids],
                s_y = [c[2] for c in cur_au.centroids]
            )
            for col in names(cur_df)
                if !(col in ["s_idx", "s_x", "s_y", string(t_name), "y_obs"])
                    if eltype(cur_df[!, col]) <: Number
                        df_units[!, col] = fill(mean(skipmissing(cur_df[!, col])), cur_au.n_units)
                    else
                        df_units[!, col] = fill(first(cur_df[!, col]), cur_au.n_units)
                    end
                end
            end
            try
                pred_res_u = Base.invokelatest(predict, m_tier, chn_tier, df_units)
                tier_unit_predictions[t_name] = pred_res_u.predictions_denoised
            catch
                u_m = [mean(pred_df[!, Symbol("$(t_name)_mean")][pred_df.s_idx .== i]) for i in 1:cur_au.n_units]
                u_s = [mean(pred_df[!, Symbol("$(t_name)_sd")][pred_df.s_idx .== i]) for i in 1:cur_au.n_units]
                tier_unit_predictions[t_name] = (mean=u_m, std=u_s, samples=nothing)
            end
        end

        # 5. Persist Tier Results to DuckDB Table
        table_name = "tier_$(t_idx)_$(t_name)"
        if duckdb_path != ":memory:" && !isempty(duckdb_path)
            export_posterior_samples_to_duckdb(
                duckdb_path, chn_tier, m_tier;
                table_name = table_name, overwrite = true
            )
            if verbose
                println("  Saved to DuckDB table: $(table_name)")
            end
        end
    end

    # --- Final Master Network Harmonization ---
    final_master_au = !isnothing(master_au) ? master_au : tier_networks[tier_specs[end].name]
    if isnothing(final_master_au)
        # Construct fallback master mesh from bounding coordinates
        all_x = Float64[]
        all_y = Float64[]
        for s in tier_specs
            if hasproperty(s.data, :s_x) && hasproperty(s.data, :s_y)
                append!(all_x, s.data.s_x)
                append!(all_y, s.data.s_y)
            end
        end
        final_master_au = assign_spatial_units(all_x, all_y; target_units=30)
    end

    n_master = final_master_au.n_units
    master_df = DataFrame(
        unit_id = 1:n_master,
        s_x = [c[1] for c in final_master_au.centroids],
        s_y = [c[2] for c in final_master_au.centroids]
    )

    # Add WKT geometry column if polygons exist
    if hasproperty(final_master_au, :polygons)
        wkt_strings = String[]
        for poly in final_master_au.polygons
            poly_closed = copy(poly)
            if !isempty(poly) && poly[1] != poly[end]
                push!(poly_closed, poly[1])
            end
            coords_str = join(["$(pt[1]) $(pt[2])" for pt in poly_closed], ", ")
            push!(wkt_strings, "POLYGON(($(coords_str)))")
        end
        master_df[!, :geom_wkt] = wkt_strings
    end

    # Reshard / evaluate all tiers onto master network
    for (t_name, pred_df) in tier_predictions
        src_au = tier_networks[t_name]
        val_col = Symbol("$(t_name)_mean")
        sd_col = Symbol("$(t_name)_sd")

        if !isnothing(src_au) && haskey(tier_unit_predictions, t_name)
            resharded = reshard_spatial_field(
                tier_unit_predictions[t_name], src_au, final_master_au; mode=:samples
            )
            master_df[!, val_col] = resharded.mean
            master_df[!, sd_col] = resharded.std
        elseif !isnothing(src_au)
            P_master = compute_network_transfer_matrix(src_au, final_master_au)
            u_means = [mean(pred_df[!, val_col][pred_df.s_idx .== i]) for i in 1:src_au.n_units]
            u_sds   = [mean(pred_df[!, sd_col][pred_df.s_idx .== i]) for i in 1:src_au.n_units]
            master_df[!, val_col] = Vector{Float64}(P_master * u_means)
            master_df[!, sd_col] = Vector{Float64}(sqrt.(P_master * (u_sds .^ 2)))
        elseif hasproperty(pred_df, :s_x) && hasproperty(pred_df, :s_y)
            kdt = KDTree(hcat(pred_df.s_x, pred_df.s_y)')
            q_pts = hcat(master_df.s_x, master_df.s_y)'
            nns, _ = knn(kdt, q_pts, 1)
            idxs = [n[1] for n in nns]
            master_df[!, val_col] = pred_df[!, val_col][idxs]
            master_df[!, sd_col] = pred_df[!, sd_col][idxs]
        end
    end

    # Save harmonized master summary to DuckDB
    if duckdb_path != ":memory:" && !isempty(duckdb_path)
        db = DuckDB.DB(duckdb_path)
        con = DuckDB.connect(db)
        try
            _write_df_to_duckdb(con, master_df, "master_harmonized_summary", true)
        finally
            DuckDB.disconnect(con)
            try close(db) catch end
            GC.gc()
        end
    end

    # Export GeoJSON if requested
    if !isnothing(geojson_path)
        export_spatial_results_to_geojson(geojson_path, master_df, final_master_au)
    end

    # Save JLD2 bundle if requested
    if !isnothing(jld2_path)
        JLD2.jldsave(jld2_path;
            models=models, chains=chains, results=results,
            derivatives=derivatives, master_summary=master_df, master_au=final_master_au
        )
    end

    if verbose
        println("\n==============================================================================")
        println("  PIPELINE EXECUTION COMPLETE")
        println("  Master Network Units: $(n_master)")
        println("  Harmonized Summary Table: 'master_harmonized_summary'")
        println("==============================================================================\n")
    end

    return PipelineResult(
        [s.name for s in tier_specs],
        models,
        chains,
        results,
        derivatives,
        tier_networks,
        final_master_au,
        master_df,
        duckdb_path
    )
end
