"""
    input_output.jl

Serialization, analytical persistence (JLD2 and DuckDB), GIS export, and model ensembling
for Bayesian Spatio-Temporal Models (BSTM).

Version: v1.0.0
"""

using Dates, Printf

# ==============================================================================
# SECTION 1: JLD2 MODEL STATE PERSISTENCE
# ==============================================================================

"""
    save_bstm_model(filepath::AbstractString, model::DynamicPPL.Model; 
                    chain=nothing, au=nothing, metadata::Dict=Dict(), compress::Bool=true)

Saves the complete state of a `bstm` model `m` and optional posterior `chain` to a JLD2 file.
"""

# Arguments
- `filepath::AbstractString`: Path to output file (should end with `.jld2` or `.bstm`).
- `model::DynamicPPL.Model`: Instantiated `bstm` Turing model.
- `chain`: (Optional) MCMC chain object (`FlexiChain` or `MCMCChains.Chains`).
- `au`: (Optional) Areal units NamedTuple from `assign_spatial_units`.
- `metadata::Dict`: (Optional) User-defined metadata dictionary (e.g. project name, notes, author).
- `compress::Bool`: Whether to compress data on disk (default: `true`).

# Example
```julia
save_bstm_model("output/my_model.jld2", m; chain=chn, au=st_data.au_spatial)
```
"""
function save_bstm_model(
    filepath::AbstractString, 
    model::DynamicPPL.Model; 
    chain=nothing, 
    au=nothing, 
    metadata::Dict=Dict(), 
    compress::Bool=true
)
    # Ensure parent directory exists
    dir = dirname(filepath)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end

    if !endswith(filepath, ".jld2") && !endswith(filepath, ".bstm")
        filepath = filepath * ".jld2"
    end

    M = model.args.M
    spec_registry = model.args.spec_registry

    # Package clean serializable configuration without redundant handles
    meta_info = merge(Dict(
        "created_at" => string(now()),
        "bstm_version" => "1.0.0",
        "formula" => string(get(M, :formula, "")),
        "model_arch" => string(get(M, :model_arch, "univariate")),
        "family" => string(get(M, :family, :gaussian)),
        "n_obs" => size(get(M, :data, DataFrame()), 1),
        "has_chain" => !isnothing(chain),
        "has_au" => !isnothing(au)
    ), Dict(string(k) => v for (k, v) in metadata))

    # Strip runtime function handles from spec_registry for clean JLD2 serialization
    clean_spec_reg = Dict{Symbol, Any}()
    for (k, v) in pairs(spec_registry)
        if v isa Function
            continue
        elseif v isa Dict
            sub_dict = Dict{Symbol, Any}()
            for (sk, sv) in pairs(v)
                if !(sv isa Function)
                    sub_dict[sk] = sv
                end
            end
            clean_spec_reg[k] = sub_dict
        else
            clean_spec_reg[k] = v
        end
    end

    # Extract W matrix if present
    W_mat = haskey(M, :W) ? M.W : (haskey(M, :technical) && haskey(M.technical, :W) ?
      M.technical[:W] : nothing)

    JLD2.jldsave(filepath; compress=compress,
        formula = string(M.formula),
        data = M.data,
        generated_model_code = get(M, :generated_model_code, ""),
        W = W_mat,
        spec_registry = clean_spec_reg,
        chain = chain,
        au = au,
        metadata = meta_info
    )

    @info "BSTM model successfully saved to '$filepath'."
    return filepath
end

"""
    load_bstm_model(filepath::AbstractString; calling_module::Module=Main)

Loads a saved `bstm` model from a JLD2 file and re-instantiates a live, callable
Turing `@model` object ready for sampling, prediction, or diagnostic post-processing.

# Returns
A `NamedTuple` containing:
- `model`: Instantiated, callable `DynamicPPL.Model`.
- `chain`: MCMC chain object (or `nothing` if not saved).
- `au`: Areal units object (or `nothing`).
- `metadata`: Saved metadata dictionary.

# Example
```julia
bundle = load_bstm_model("output/my_model.jld2")
m = bundle.model
chn = bundle.chain
```
"""
function load_bstm_model(filepath::AbstractString; calling_module::Module=Main)
    if !isfile(filepath)
        if isfile(filepath * ".jld2")
            filepath = filepath * ".jld2"
        else
            error("BSTM model file not found: '$filepath'")
        end
    end

    f = JLD2.jldopen(filepath, "r")
    formula_str = f["formula"]
    data_df = f["data"]
    W_mat = haskey(f, "W") ? f["W"] : nothing
    chain = haskey(f, "chain") ? f["chain"] : nothing
    au = haskey(f, "au") ? f["au"] : nothing
    meta = haskey(f, "metadata") ? f["metadata"] : Dict()
    close(f)

    # Re-instantiate the live Turing Model using bstm_core
    kwargs = Dict{Symbol, Any}()
    if !isnothing(W_mat)
        kwargs[:W] = W_mat
    end
    if !isnothing(au)
        kwargs[:au] = au
    end
    kwargs[:verbose] = false

    # Reconstruct live model
    live_model = bstm_core(formula_str, data_df, calling_module; kwargs...)

    @info "BSTM model successfully loaded and instantiated from '$filepath'."
    return (
        model = live_model,
        chain = chain,
        au = au,
        metadata = meta
    )
end

# ==============================================================================
# SECTION 2: DUCKDB RESULTS & POSTERIOR PERSISTENCE
# ==============================================================================

"""
    save_bstm_results(duckdb_path::AbstractString, res::NamedTuple; 
                      model=nothing, chain=nothing, au=nothing,
                      table_prefix::String="", overwrite::Bool=true)

Persists post-processing results (`res` from `model_results_comprehensive`) into an analytical
DuckDB database without data redundancy.

Stores normalized relational tables:
- `<prefix>model_metadata`: Model formula, family, creation timestamp, observation count.
- `<prefix>metrics`: Summary metrics (RMSE, Pearson r, ESS, Rhat, WAIC, time).
- `<prefix>parameter_stats`: Parameter posterior means, medians, stds, credible intervals.
- `<prefix>predictions`: Denoised observation-level predictions, intervals, and residuals.
- `<prefix>spatial_geometries`: (Optional) Polygon boundaries in standard WKT format.
- `<prefix>plot_data_<key>`: Tidy dataframes for all diagnostic plots.
- `<prefix>posterior_samples`: (Optional) Raw posterior parameter draws if `chain` is passed.

# Example
```julia
save_bstm_results("output/results.duckdb", res; model=m, chain=chn, au=data.au)
```
"""
function save_bstm_results(
    duckdb_path::AbstractString, 
    res::NamedTuple; 
    model=nothing, 
    chain=nothing, 
    au=nothing, 
    table_prefix::String="", 
    overwrite::Bool=true
)
    # Ensure directory exists
    dir = dirname(duckdb_path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end

    if !endswith(duckdb_path, ".duckdb") && !endswith(duckdb_path, ".db")
        duckdb_path = duckdb_path * ".duckdb"
    end

    db = DuckDB.DB(duckdb_path)
    con = DuckDB.connect(db)

    pfx = isempty(table_prefix) ? "" : (endswith(table_prefix, "_") ? table_prefix :
      table_prefix * "_")

    try
        # 1. Save Model Metadata Table
        formula_str = !isnothing(model) && hasproperty(model, :args) &&
          hasproperty(model.args, :M) ? string(model.args.M.formula) : "unknown"
        family_str = !isnothing(model) && hasproperty(model, :args) &&
          hasproperty(model.args, :M) ? string(get(model.args.M, :family, :gaussian)) :
          "unknown"
        
        df_meta = DataFrame(
            property = ["saved_at", "formula", "family", "bstm_version"],
            value = [string(now()), formula_str, family_str, "1.0.0"]
        )
        _write_df_to_duckdb(con, df_meta, "$(pfx)model_metadata", overwrite)

        # 2. Save Metrics Table
        if hasproperty(res, :metrics)
            m_keys = String[]
            m_vals = Float64[]
            for (k, v) in pairs(res.metrics)
                if v isa Number
                    push!(m_keys, string(k))
                    push!(m_vals, Float64(v))
                elseif v isa AbstractVector{<:Number}
                    for (idx, sub_v) in enumerate(v)
                        push!(m_keys, "$(k)_$idx")
                        push!(m_vals, Float64(sub_v))
                    end
                end
            end
            df_metrics = DataFrame(metric = m_keys, value = m_vals)
            _write_df_to_duckdb(con, df_metrics, "$(pfx)metrics", overwrite)
        end

        # 3. Save Parameter Posteriors Table
        df_params = if hasproperty(res, :parameters) && res.parameters isa DataFrame &&
          !isempty(res.parameters)
            res.parameters
        elseif !isnothing(chain)
            try
                _compute_direct_parameter_summary(chain, model)
            catch
                DataFrame()
            end
        else
            DataFrame()
        end
        if !isempty(df_params)
            _write_df_to_duckdb(con, df_params, "$(pfx)parameter_stats", overwrite)
        end

        # 4. Save Denoised Predictions Table
        preds = hasproperty(res, :predictions) && hasproperty(res.predictions, :denoised) ?
          res.predictions.denoised : nothing
        if !isnothing(preds) && preds isa NamedTuple && hasproperty(preds, :mean)
            N = length(preds.mean)
            df_preds = DataFrame(
                obs_id = 1:N,
                pred_mean = preds.mean,
                pred_lower = hasproperty(preds, :lower) ? preds.lower : fill(NaN, N),
                pred_upper = hasproperty(preds, :upper) ? preds.upper : fill(NaN, N)
            )
            y_obs_vec = if hasproperty(res.predictions, :observed) &&
              !isnothing(res.predictions.observed)
                res.predictions.observed
            elseif !isnothing(model) && hasproperty(model, :args) && hasproperty(model.args,
              :M) && hasproperty(model.args.M, :y_obs)
                Array(model.args.M.y_obs)
            else
                nothing
            end
            if !isnothing(y_obs_vec) && length(y_obs_vec) == N
                df_preds.y_obs = y_obs_vec
                df_preds.residual = y_obs_vec .- preds.mean
            end
            _write_df_to_duckdb(con, df_preds, "$(pfx)predictions", overwrite)
        end

        # 5. Save Spatial Geometries in WKT format (if au is available)
        if !isnothing(au) && hasproperty(au, :polygons) && hasproperty(au, :centroids)
            polys = au.polygons
            cents = au.centroids
            S = length(cents)
            wkt_vec = String[_polygon_to_wkt(polys[i]) for i in 1:min(S, length(polys))]
            cx_vec = Float64[cents[i][1] for i in 1:S]
            cy_vec = Float64[cents[i][2] for i in 1:S]
            area_vec = hasproperty(au, :areas) && length(au.areas) == S ? Float64.(au.areas)
              : fill(NaN, S)
            pt_cnt = hasproperty(au, :point_counts) && length(au.point_counts) == S ?
              Int.(au.point_counts) : fill(0, S)

            df_geom = DataFrame(
                unit_id = 1:S,
                centroid_x = cx_vec,
                centroid_y = cy_vec,
                area = area_vec,
                point_count = pt_cnt,
                wkt = wkt_vec
            )
            _write_df_to_duckdb(con, df_geom, "$(pfx)spatial_geometries", overwrite)
        end

        # 6. Save Plots Data Tables
        if hasproperty(res, :plots_data) && !isnothing(res.plots_data)
            for (p_key, p_df) in pairs(res.plots_data)
                if p_df isa DataFrame && !isempty(p_df)
                    tbl_name = "$(pfx)plot_data_$(p_key)"
                    _write_df_to_duckdb(con, p_df, tbl_name, overwrite)
                end
            end
        end

        # 7. Save Posterior Samples Table (if chain is passed)
        if !isnothing(chain)
            df_samples = _chain_to_tidy_df(chain, model)
            _write_df_to_duckdb(con, df_samples, "$(pfx)posterior_samples", overwrite)
        end
    finally
        DuckDB.disconnect(con)
        try close(db) catch end
        GC.gc()
    end

    @info "BSTM analytical results successfully saved to DuckDB database '$duckdb_path'."
    return duckdb_path
end

"""
    load_bstm_results(duckdb_path::AbstractString; table_prefix::String="")::NamedTuple

Loads previously saved model results from a DuckDB database into a structured NamedTuple
matching `model_results_comprehensive`.
"""
function load_bstm_results(duckdb_path::AbstractString; table_prefix::String="")::NamedTuple
    if !isfile(duckdb_path)
        if isfile(duckdb_path * ".duckdb")
            duckdb_path = duckdb_path * ".duckdb"
        else
            error("DuckDB database file not found: '$duckdb_path'")
        end
    end

    db = DuckDB.DB(duckdb_path)
    con = DuckDB.connect(db)

    pfx = isempty(table_prefix) ? "" : (endswith(table_prefix, "_") ? table_prefix :
      table_prefix * "_")

    metrics_nt = (;)
    parameters_df = DataFrame()
    plots_data_dict = Dict{Symbol, DataFrame}()
    preds_dict = Dict{Symbol, Any}()

    try
        # 1. Load Metrics
        try
            df_m = DataFrame(DuckDB.query(con, "SELECT * FROM $(pfx)metrics"))
            m_dict = Dict{Symbol, Float64}()
            for row in eachrow(df_m)
                m_dict[Symbol(row.metric)] = Float64(row.value)
            end
            metrics_nt = NamedTuple(m_dict)
        catch
        end

        # 2. Load Parameter Stats
        try
            parameters_df = DataFrame(DuckDB.query(con, "SELECT * FROM $(pfx)parameter_stats"))
        catch
        end

        # 3. Load Predictions
        try
            df_p = DataFrame(DuckDB.query(con, "SELECT * FROM $(pfx)predictions"))
            preds_dict[:denoised] = (
                mean = df_p.pred_mean,
                lower = df_p.pred_lower,
                upper = df_p.pred_upper
            )
        catch
        end

        # 4. Load Plot Data Tables
        tbl_names = DataFrame(DuckDB.query(con, "SELECT table_name FROM
          information_schema.tables WHERE table_schema='main'"))
        plot_prefix = "$(pfx)plot_data_"
        for row in eachrow(tbl_names)
            t_name = string(row.table_name)
            if startswith(t_name, plot_prefix)
                clean_k = Symbol(replace(t_name, plot_prefix => ""))
                plots_data_dict[clean_k] = DataFrame(DuckDB.query(con, "SELECT * FROM $(t_name)"))
            end
        end

    finally
        DuckDB.disconnect(con)
        try close(db) catch end
        GC.gc()
    end

    predictions_nt = if haskey(preds_dict, :denoised)
        (denoised = preds_dict[:denoised],)
    else
        (;)
    end

    return (
        metrics = metrics_nt,
        parameters = parameters_df,
        predictions = predictions_nt,
        plots_data = NamedTuple(plots_data_dict)
    )
end

"""
    query_duckdb(duckdb_path::AbstractString, sql_query::AbstractString)::DataFrame

Executes an arbitrary SQL query against a BSTM DuckDB database
and returns the result as a DataFrame.

# Example
```julia
df_high_risk = query_duckdb("output/results.duckdb", 
    "SELECT * FROM bstm_plot_data_sre_spatial WHERE sre_mean > 1.5 ORDER BY sre_mean DESC")
```
"""
function query_duckdb(duckdb_path::AbstractString, sql_query::AbstractString)::DataFrame
    if !isfile(duckdb_path) && isfile(duckdb_path * ".duckdb")
        duckdb_path = duckdb_path * ".duckdb"
    end
    db = DuckDB.DB(duckdb_path)
    con = DuckDB.connect(db)
    local df
    try
        df = DataFrame(DuckDB.query(con, sql_query))
    finally
        DuckDB.disconnect(con)
        try close(db) catch end
        GC.gc()
    end
    return df
end

# Helper to write DataFrame into DuckDB table
function _write_df_to_duckdb(con::Any, df::DataFrame, table_name::String, overwrite::Bool)
    if isempty(df)
        return
    end
    df_clean = copy(df)
    for col in names(df_clean)
        if eltype(df_clean[!, col]) <: Symbol
            df_clean[!, col] = string.(df_clean[!, col])
        end
    end
    temp_view = "temp_view_$(rand(10000:99999))"
    DuckDB.register_data_frame(con, df_clean, temp_view)
    create_stmt = overwrite ? "CREATE OR REPLACE TABLE $(table_name) AS SELECT * FROM
      $(temp_view)" :
                              "CREATE TABLE IF NOT EXISTS $(table_name) AS SELECT * FROM
                                $(temp_view)"
    DuckDB.query(con, create_stmt)
    DuckDB.unregister_data_frame(con, temp_view)
end

# Helper to convert polygon coordinates to Well-Known Text (WKT)
function _polygon_to_wkt(poly)
    pts = if poly isa AbstractVector
        if !isempty(poly) && poly[1] isa Tuple
            poly
        elseif !isempty(poly) && poly[1] isa AbstractVector
            [(p[1], p[2]) for p in poly]
        else
            return "POLYGON EMPTY"
        end
    else
        return "POLYGON EMPTY"
    end

    if isempty(pts)
        return "POLYGON EMPTY"
    end
    # Ensure closed ring
    closed_pts = copy(pts)
    if closed_pts[1] != closed_pts[end]
        push!(closed_pts, closed_pts[1])
    end

    coord_strs = ["$(p[1]) $(p[2])" for p in closed_pts]
    return "POLYGON((" * join(coord_strs, ", ") * "))"
end

# ==============================================================================
# SECTION 3: SPATIAL GIS & GEOJSON EXPORT
# ==============================================================================

"""
    export_spatial_results_to_geojson(geojson_path::AbstractString, res::NamedTuple,
                                      au::NamedTuple; property_keys=nothing)

Exports spatial model estimates and polygon geometries to a standard RFC 7946 GeoJSON file
for direct visualization in GIS tools (QGIS, ArcGIS, Mapbox, Leaflet, Kepler.gl).

# Arguments
- `geojson_path::AbstractString`: Output `.geojson` file path.
- `res::NamedTuple`: Result from `model_results_comprehensive`.
- `au::NamedTuple`: Areal units object with `polygons` and `centroids`.
- `property_keys`: (Optional) Vector of column symbols to attach as GeoJSON properties.
"""
function export_spatial_results_to_geojson(
    geojson_path::AbstractString, 
    res::NamedTuple, 
    au::NamedTuple; 
    property_keys = nothing
)
    polys = au.polygons
    cents = au.centroids
    S = length(cents)

    # Extract spatial DataFrame from plots_data or pstats
    df_spatial = if hasproperty(res, :plots_data) && hasproperty(res.plots_data, :sre_spatial)
        res.plots_data.sre_spatial
    else
        DataFrame(unit_id = 1:S)
    end

    props_to_include = if isnothing(property_keys)
        names(df_spatial)
    else
        string.(property_keys)
    end

    features = String[]
    for i in 1:min(S, length(polys))
        p = polys[i]
        if isempty(p)
            continue
        end
        closed_p = copy(p)
        if closed_p[1] != closed_p[end]
            push!(closed_p, closed_p[1])
        end

        coord_json = "[" * join(["[$(pt[1]), $(pt[2])]" for pt in closed_p], ", ") * "]"
        
        # Build properties JSON
        prop_pairs = String[]
        if nrow(df_spatial) >= i
            row = df_spatial[i, :]
            for col in props_to_include
                if hasproperty(row, Symbol(col))
                    val = row[Symbol(col)]
                    if val isa Number
                        push!(prop_pairs, "\"$col\": $(val)")
                    else
                        push!(prop_pairs, "\"$col\": \"$(val)\"")
                    end
                end
            end
        end
        props_json = "{" * join(prop_pairs, ", ") * "}"

        feature_json = """{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [$coord_json]
            },
            "properties": $props_json
        }"""
        push!(features, feature_json)
    end

    geojson_str = """{
        "type": "FeatureCollection",
        "features": [
            $(join(features, ",\n"))
        ]
    }"""

    dir = dirname(geojson_path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    write(geojson_path, geojson_str)

    @info "Spatial model results successfully exported to GeoJSON: '$geojson_path'."
    return geojson_path
end

# ==============================================================================
# SECTION 4: SEQUENTIAL BAYESIAN PRIOR EXTRACTION
# ==============================================================================

"""
    extract_posterior_priors(source; parameter_names=nothing, prior_family=:normal)

Extracts posterior distributions from previous model runs (`res` or a DuckDB database)
and constructs fitted prior distributions (`Normal(mean, std)` or `truncated(Normal(...))`)
ready for sequential Bayesian updating in subsequent `bstm` models.

# Example
```julia
# Extract priors from stage-1 run
priors = extract_posterior_priors("stage1_results.duckdb")

# Use as informative priors in stage-2 model
m2 = @bstm(
    likelihood(y) ~ intercept(prior=priors[:intercept]) +
                    fixed(elev, prior=priors[:beta_elev]),
    df2
)
```
"""
function extract_posterior_priors(
    source::Union{AbstractString, NamedTuple}; 
    parameter_names = nothing, 
    prior_family::Symbol = :normal
)
    df_stats = if source isa AbstractString
        query_duckdb(source, "SELECT * FROM parameter_stats")
    elseif hasproperty(source, :parameters)
        source.parameters
    else
        error("Source must be a DuckDB path or a model results NamedTuple with parameters.")
    end

    if isempty(df_stats)
        @warn "No parameter statistics found in source."
        return Dict{Symbol, Any}()
    end

    p_col = hasproperty(df_stats, :parameters) ? :parameters : (hasproperty(df_stats,
      :parameter) ? :parameter : names(df_stats)[1])
    m_col = :mean
    s_col = :std

    priors_dict = Dict{Symbol, Any}()

    for row in eachrow(df_stats)
        p_name = Symbol(row[p_col])
        if !isnothing(parameter_names) && !(p_name in parameter_names)
            continue
        end

        mu = Float64(row[m_col])
        sigma = max(Float64(row[s_col]), 1e-4) # ensure non-zero variance

        if prior_family == :normal
            priors_dict[p_name] = Normal(mu, sigma)
        elseif prior_family == :truncated_normal
            priors_dict[p_name] = truncated(Normal(mu, sigma), 0.0, Inf)
        else
            priors_dict[p_name] = Normal(mu, sigma)
        end
    end

    @info "Extracted $(length(priors_dict)) informative prior distributions."
    return priors_dict
end

# ==============================================================================
# SECTION 5: MULTI-MODEL ENSEMBLING & BAYESIAN MODEL AVERAGING (BMA)
# ==============================================================================

"""
    save_model_ensemble(duckdb_path::AbstractString, ensemble_dict::Dict; overwrite::Bool=true)

Persists a collection of candidate models into a unified DuckDB database.

Constructs a central `models_registry` table comparing:
- `rmse`, `r_pearson`, `waic`
- `delta_waic`: \\Delta WAIC_k = WAIC_k - min_j WAIC_j
- `waic_weight`: Akaike/WAIC model weights
"""
function save_model_ensemble(
    duckdb_path::AbstractString, 
    ensemble_dict::Dict; 
    overwrite::Bool=true
)
    # Save individual models with table prefixes
    waic_vals = Float64[]
    rmse_vals = Float64[]
    m_names = String[]

    for (m_sym, bundle) in ensemble_dict
        pfx = string(m_sym) * "_"
        m_obj = hasproperty(bundle, :model) ? bundle.model : nothing
        c_obj = hasproperty(bundle, :chain) ? bundle.chain : nothing
        r_obj = hasproperty(bundle, :results) ? bundle.results : bundle

        save_bstm_results(duckdb_path, r_obj; model=m_obj, chain=c_obj, table_prefix=pfx,
          overwrite=overwrite)

        w_val = hasproperty(r_obj.metrics, :waic) ? Float64(r_obj.metrics.waic) : NaN
        rm_val = hasproperty(r_obj.metrics, :rmse) && r_obj.metrics.rmse isa Number ?
          Float64(r_obj.metrics.rmse) : NaN

        push!(m_names, string(m_sym))
        push!(waic_vals, w_val)
        push!(rmse_vals, rm_val)
    end

    # Compute WAIC Weights
    valid_waic = filter(!isnan, waic_vals)
    min_w = isempty(valid_waic) ? 0.0 : minimum(valid_waic)
    delta_w = [isnan(w) ? NaN : w - min_w for w in waic_vals]
    exp_w = [isnan(d) ? 0.0 : exp(-0.5 * d) for d in delta_w]
    sum_exp = sum(exp_w)
    weights = sum_exp > 0 ? exp_w ./ sum_exp : fill(1.0 / length(m_names), length(m_names))

    df_registry = DataFrame(
        model_name = m_names,
        rmse = rmse_vals,
        waic = waic_vals,
        delta_waic = delta_w,
        bma_weight = weights
    )

    db = DuckDB.DB(duckdb_path)
    con = DuckDB.connect(db)
    try
        _write_df_to_duckdb(con, df_registry, "models_registry", overwrite)
    finally
        DuckDB.disconnect(con)
        try close(db) catch end
        GC.gc()
    end

    @info "Model ensemble saved with $(length(m_names)) models in '$duckdb_path'."
    return df_registry
end

"""
    bma_weighted_predictions(duckdb_path::AbstractString) -> DataFrame

Computes Bayesian Model Averaged (BMA) predictions across all candidate models
registered in the DuckDB database using their normalized WAIC weights.

Returns a DataFrame with `(obs_id, bma_pred_mean, bma_pred_sd)`.
"""
function bma_weighted_predictions(duckdb_path::AbstractString)::DataFrame
    df_reg = query_duckdb(
        duckdb_path, "SELECT * FROM models_registry WHERE NOT isnan(bma_weight)"
    )
    if isempty(df_reg)
        error("No models with valid BMA weights found in '$duckdb_path'.")
    end

    # Fetch predictions for each model
    preds_list = DataFrame[]
    weights = Float64[]

    for row in eachrow(df_reg)
        m_name = string(row.model_name)
        w = Float64(row.bma_weight)
        df_p = query_duckdb(duckdb_path, "SELECT obs_id, pred_mean FROM " *
          "$(m_name)_predictions ORDER BY obs_id")
        push!(preds_list, df_p)
        push!(weights, w)
    end

    N = nrow(preds_list[1])
    bma_mean = zeros(Float64, N)
    bma_var = zeros(Float64, N)

    for (df_p, w) in zip(preds_list, weights)
        bma_mean .+= w .* df_p.pred_mean
    end

    # Law of Total Variance
    for (df_p, w) in zip(preds_list, weights)
        bma_var .+= w .* (df_p.pred_mean .- bma_mean).^2
    end

    return DataFrame(
        obs_id = preds_list[1].obs_id,
        bma_pred_mean = bma_mean,
        bma_pred_sd = sqrt.(bma_var)
    )
end

# ==============================================================================
# SECTION 6: OUT-OF-SAMPLE PREDICTIONS, ZERO-COPY EXPORT & COMPACTION
# ==============================================================================

"""
    save_out_of_sample_predictions(duckdb_path::AbstractString, pred_df::DataFrame; 
                                   table_name::String="out_of_sample_predictions",
                                   overwrite::Bool=true)

Saves out-of-sample predictions (e.g. from `predict(model, chain, new_data)`) into DuckDB.
"""
function save_out_of_sample_predictions(
    duckdb_path::AbstractString, 
    pred_df::DataFrame; 
    table_name::String="out_of_sample_predictions", 
    overwrite::Bool=true
)
    db = DuckDB.DB(duckdb_path)
    con = DuckDB.connect(db)
    try
        _write_df_to_duckdb(con, pred_df, table_name, overwrite)
    finally
        DuckDB.disconnect(con)
        try close(db) catch end
        GC.gc()
    end
    @info "Out-of-sample predictions saved to table '$table_name' in '$duckdb_path'."
end

"""
    export_results_to_parquet(duckdb_path::AbstractString, table_name::AbstractString,
                              output_parquet_path::AbstractString)

Exports a DuckDB table to a high-speed compressed Parquet file using DuckDB's export engine.
"""
function export_results_to_parquet(
    duckdb_path::AbstractString, 
    table_name::AbstractString, 
    output_parquet_path::AbstractString
)
    dir = dirname(output_parquet_path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end

    db = DuckDB.DB(duckdb_path)
    con = DuckDB.connect(db)
    try
        # Escape path for SQL
        clean_path = replace(output_parquet_path, "\\" => "/")
        DuckDB.query(con, "COPY (SELECT * FROM $(table_name)) TO '$(clean_path)' (FORMAT " *
          "PARQUET, COMPRESSION ZSTD)")
    finally
        DuckDB.disconnect(con)
        try close(db) catch end
        GC.gc()
    end
    @info "Table '$table_name' exported to Parquet: '$output_parquet_path'."
    return output_parquet_path
end

"""
    export_results_to_csv(duckdb_path::AbstractString, table_name::AbstractString,
                          output_csv_path::AbstractString)

Exports a DuckDB table to a CSV file.
"""
function export_results_to_csv(
    duckdb_path::AbstractString, 
    table_name::AbstractString, 
    output_csv_path::AbstractString
)
    dir = dirname(output_csv_path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end

    db = DuckDB.DB(duckdb_path)
    con = DuckDB.connect(db)
    try
        clean_path = replace(output_csv_path, "\\" => "/")
        DuckDB.query(con, "COPY (SELECT * FROM $(table_name)) TO '$(clean_path)' (HEADER,
          DELIMITER ',')")
    finally
        DuckDB.disconnect(con)
        try close(db) catch end
        GC.gc()
    end
    @info "Table '$table_name' exported to CSV: '$output_csv_path'."
    return output_csv_path
end

"""
    compact_duckdb(duckdb_path::AbstractString)

Performs `VACUUM; ANALYZE;` on the DuckDB database to reclaim unallocated disk space
and optimize index query statistics.
"""
function compact_duckdb(duckdb_path::AbstractString)
    if !isfile(duckdb_path) && isfile(duckdb_path * ".duckdb")
        duckdb_path = duckdb_path * ".duckdb"
    end
    db = DuckDB.DB(duckdb_path)
    con = DuckDB.connect(db)
    try
        DuckDB.query(con, "VACUUM")
        DuckDB.query(con, "ANALYZE")
    finally
        DuckDB.disconnect(con)
        try close(db) catch end
        GC.gc()
    end
    @info "DuckDB database '$duckdb_path' compacted and optimized."
end

# ==============================================================================
# SECTION 7: POSTERIOR SAMPLES EXPORT & MULTI-MODEL INTEGRATION
# ==============================================================================

"""
    export_posterior_samples_to_duckdb(duckdb_path::AbstractString, chain, model=nothing;
                                       table_name::String="bstm_posterior_samples", 
                                       format::Symbol=:tidy, overwrite::Bool=true)

Exports raw MCMC posterior draws into DuckDB for high-performance SQL querying.
"""
function export_posterior_samples_to_duckdb(
    duckdb_path::AbstractString, 
    chain, 
    model=nothing;
    table_name::String="bstm_posterior_samples", 
    format::Symbol=:tidy, 
    overwrite::Bool=true
)
    if isnothing(chain)
        return
    end

    # Convert chain into tidy or wide DataFrame
    df_samples = if format == :wide
        DataFrame(chain)
    else
        _chain_to_tidy_df(chain, model)
    end

    db = DuckDB.DB(duckdb_path)
    con = DuckDB.connect(db)
    try
        _write_df_to_duckdb(con, df_samples, table_name, overwrite)
    finally
        DuckDB.disconnect(con)
        try close(db) catch end
        GC.gc()
    end

    @info "Posterior samples exported to DuckDB table '$table_name' in '$duckdb_path'."
end

function _chain_to_tidy_df(chain, model=nothing)
    if chain isa DataFrame
        return chain
    end
    p_names = _get_clean_chain_param_names(chain)
    if isempty(p_names) && !isnothing(model) && hasproperty(model.args, :M)
        p_names = build_param_registry(model.args.M).names
    end
    
    n_chains_val = _get_chain_n_chains(chain)
    
    rows = []
    for pname in p_names
        try
            samples_mat = extract_param_matrix(chain, pname)
            dim = size(samples_mat, 2)
            n_total = size(samples_mat, 1)
            n_iter_per_chain = n_chains_val >= 1 ? max(1, n_total ÷ n_chains_val) : n_total
            
            for d in 1:dim
                v = samples_mat[:, d]
                param_label = dim == 1 ? string(pname) : "$(pname)[$d]"
                for idx in 1:n_total
                    c = n_chains_val > 1 ? min(n_chains_val, (idx - 1) ÷ n_iter_per_chain + 1) : 1
                    it = n_chains_val > 1 ? ((idx - 1) % n_iter_per_chain + 1) : idx
                    push!(rows, (
                        iteration = it,
                        chain = c,
                        parameter = param_label,
                        value = Float64(v[idx])
                    ))
                end
            end
        catch
        end
    end
    return DataFrame(rows)
end

"""
    import_posterior_samples_from_duckdb(duckdb_path::AbstractString; 
                                         table_name::String="bstm_posterior_samples")::DataFrame

Reads posterior samples stored in DuckDB into a DataFrame.
"""
function import_posterior_samples_from_duckdb(
    duckdb_path::AbstractString; 
    table_name::String="bstm_posterior_samples"
)::DataFrame
    return query_duckdb(duckdb_path, "SELECT * FROM $(table_name)")
end

# ==============================================================================
# SECTION 8: CHAIN EXTENSION & RESUMING SAMPLING
# ==============================================================================

"""
    append_posterior_samples(chain1, chain2)

Concatenates two MCMC chains across sampling iterations.
Supports `FlexiChain`, `VNChain`, `MCMCChains.Chains`, and `DataFrame`.
"""
function append_posterior_samples(chain1, chain2)
    if isnothing(chain1)
        return chain2
    end
    if isnothing(chain2)
        return chain1
    end

    try
        return vcat(chain1, chain2)
    catch
        try
            return [chain1; chain2]
        catch
            return chain2
        end
    end
end

"""
    extend_sampling(model::DynamicPPL.Model, prev_chain, n_additional_samples::Int; 
                    sampler=NUTS(), kwargs...)

Draws `n_additional_samples` from `model` and concatenates them with `prev_chain`.
"""
function extend_sampling(
    model::DynamicPPL.Model, 
    prev_chain, 
    n_additional_samples::Int; 
    sampler=NUTS(), 
    kwargs...
)
    new_chain = sample(model, sampler, n_additional_samples; kwargs...)
    combined = append_posterior_samples(prev_chain, new_chain)
    @info "Successfully extended chain by $n_additional_samples iterations."
    return combined
end

# ==============================================================================
# SECTION 9: UNIFIED MODEL & RESULTS BUNDLE
# ==============================================================================

"""
    save_bstm_bundle(base_path::AbstractString, model::DynamicPPL.Model, chain, res::NamedTuple; 
                     au=nothing, metadata::Dict=Dict(), compress::Bool=true)

Unified one-line persister saving:
1. `<base_path>.jld2`: Live Turing Model state `m`, `M`, and posterior `chain`.
2. `<base_path>.duckdb`: Relational analytical database containing `res`, metrics, and plot tables.
"""
function save_bstm_bundle(
    base_path::AbstractString, 
    model::DynamicPPL.Model, 
    chain, 
    res::NamedTuple; 
    au=nothing, 
    metadata::Dict=Dict(), 
    compress::Bool=true
)
    clean_base = replace(base_path, r"\.(jld2|duckdb|db|bstm)$" => "")
    jld2_file = clean_base * ".jld2"
    duckdb_file = clean_base * ".duckdb"

    save_bstm_model(jld2_file, model; chain=chain, au=au, metadata=metadata, compress=compress)
    save_bstm_results(duckdb_file, res; model=model, chain=chain, au=au, overwrite=true)

    @info "Complete BSTM bundle saved to:\n  - Model:   '$jld2_file'\n  - Results: '$duckdb_file'"
    return (model_file = jld2_file, results_file = duckdb_file)
end

"""
    load_bstm_bundle(base_path::AbstractString; calling_module::Module=Main)

Loads a complete BSTM model bundle (`<base_path>.jld2` and `<base_path>.duckdb`).
"""
function load_bstm_bundle(base_path::AbstractString; calling_module::Module=Main)
    clean_base = replace(base_path, r"\.(jld2|duckdb|db|bstm)$" => "")
    jld2_file = clean_base * ".jld2"
    duckdb_file = clean_base * ".duckdb"

    m_data = load_bstm_model(jld2_file; calling_module=calling_module)
    res_data = isfile(duckdb_file) ? load_bstm_results(duckdb_file) : (;)

    return (
        model = m_data.model,
        chain = m_data.chain,
        results = res_data,
        au = m_data.au,
        metadata = m_data.metadata
    )
end
