---
title: "BSTM Input / Output & Persistence Subsystem: Technical Guide"
format: html
---

# BSTM Input / Output & Persistence Subsystem: Technical Guide

## 1. Executive Summary & Design Philosophy

In Bayesian spatiotemporal modeling, fitted models encompass heterogeneous artifacts:
1. **Dynamic Model Computations**: Instantiated Turing `@model` instances, compiled code, sparse graph precision matrices ($Q$), and basis function expansions ($B$).
2. **MCMC Posterior Chains**: High-dimensional posterior sample arrays spanning iterations, chains, fixed effects, and latent fields.
3. **Structured Diagnostics & Predictions**: Denoised latent surfaces, observation-level credible intervals, and performance metrics (RMSE, $R^2$, WAIC).
4. **Spatial Geometries & GIS Attributes**: Polygon boundaries, spatial adjacency topologies ($W$), centroid coordinates, and shape metrics.
5. **Publication Visualizations**: Tabular datasets backing spatial choropleths, time-series ribbons, and posterior predictive checks.

Storing all of these entities in a single unstructured file leads to massive memory bloat, high data redundancy, and poor interoperability. 

The `bstm` framework solves this with a **Two-Tier Decoupled Persistence Architecture**:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               Two-Tier Persistence Architecture                         │
├───────────────────────────────────────────┬─────────────────────────────────────────────┤
│  Tier 1: JLD2 Model State (.jld2)         │  Tier 2: DuckDB Analytical Engine (.duckdb) │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│  - Live callable Turing @model state      │  - Normalized relational database schema    │
│  - Symbolic formula AST & data snapshot   │  - Denoised predictions & credible bounds   │
│  - Graph topology (W) & spatial metadata  │  - Parameter summaries & diagnostic metrics │
│  - Raw MCMC posterior chains (FlexiChain) │  - Native WKT spatial geometries & GIS maps │
│  - Warm-restart & chain extension engine  │  - Multi-model Bayesian Averaging (BMA)     │
│  - Sequential Bayesian prior derivation   │  - Zero-copy Parquet / CSV export           │
└───────────────────────────────────────────┴─────────────────────────────────────────────┘
```

---

## 2. Architecture & Data Flow

```
                      ┌──────────────────────────────────────────────┐
                      │    Fitted BSTM Model (m) & Chain (chn)       │
                      │  res = model_results_comprehensive(m, chn)   │
                      └──────────────────────┬───────────────────────┘
                                             │
                      ┌──────────────────────┴───────────────────────┐
                      ▼                                              ▼
        ┌────────────────────────────┐                ┌────────────────────────────┐
        │     save_bstm_model()      │                │    save_bstm_results()     │
        │     save_bstm_bundle()     │                │    save_bstm_bundle()      │
        └─────────────┬──────────────┘                └─────────────┬──────────────┘
                      │                                              │
                      ▼                                              ▼
        ┌────────────────────────────┐                ┌────────────────────────────┐
        │      model_file.jld2       │                │     results.duckdb         │
        │  - Formula & Data          │                │  - model_metadata          │
        │  - Dynamic Model Code      │                │  - metrics                 │
        │  - Spatial Graph (W)       │                │  - parameter_stats         │
        │  - MCMC Chain Arrays       │                │  - predictions             │
        │  - User Metadata           │                │  - spatial_geometries      │
        └─────────────┬──────────────┘                │  - plot_data_*             │
                      │                               │  - posterior_samples       │
                      │                               └─────────────┬──────────────┘
                      │                                              │
        ┌─────────────┴──────────────┐                ┌─────────────┴──────────────┐
        │  load_bstm_model()         │                │  query_duckdb()            │
        │  extend_sampling()         │                │  bma_weighted_predictions()│
        │  extract_posterior_priors()│                │  export_to_parquet()       │
        │  (Resume / Prior Transfer) │                │  export_to_geojson()       │
        └────────────────────────────┘                └────────────────────────────┘
```

---

## 3. DuckDB Relational Schema Reference

When `save_bstm_results`, `save_model_ensemble`, or `save_bstm_bundle` persists results into DuckDB, it writes a set of clean, normalized relational tables:

### 3.1. `model_metadata`
Stores core provenance and architectural configurations:
- `property` (`VARCHAR`): Property name (`saved_at`, `formula`, `family`, `bstm_version`).
- `value` (`VARCHAR`): String value.

### 3.2. `metrics`
Stores global model performance indicators:
- `metric` (`VARCHAR`): Metric key (`rmse`, `r_pearson`, `ess`, `rhat`, `waic`, `time`).
- `value` (`DOUBLE`): Numeric value.

### 3.3. `parameter_stats`
Stores parameter-level posterior summaries computed across chains:
- `parameters` (`VARCHAR`): Parameter name (`beta_elev`, `sigma_s_idx`, `rho_year`, etc.).
- `mean` (`DOUBLE`): Posterior mean.
- `std` (`DOUBLE`): Posterior standard deviation.
- `naive_se` (`DOUBLE`): Naive standard error of the mean.
- `mcse` (`DOUBLE`): Monte Carlo standard error.
- `ess` (`DOUBLE`): Effective sample size.
- `rhat` (`DOUBLE`): Gelman-Rubin split-$\hat{R}$ diagnostic.
- `ess_per_sec` (`DOUBLE`): Sampling efficiency.

### 3.4. `predictions`
Observation-level posterior fitted values and intervals:
- `obs_id` (`BIGINT`): 1-indexed observation row identifier.
- `y_obs` (`DOUBLE`): Observed outcome value.
- `pred_mean` (`DOUBLE`): Posterior mean prediction ($\mathbb{E}[y \mid \text{data}]$).
- `pred_lower` (`DOUBLE`): Lower $(1-\alpha)$ credible limit (e.g. 2.5%).
- `pred_upper` (`DOUBLE`): Upper $(1-\alpha)$ credible limit (e.g. 97.5%).
- `residual` (`DOUBLE`): Observation residual ($y_{\text{obs}} - \hat{y}$).

### 3.5. `spatial_geometries`
Standard OGC Well-Known Text (WKT) geometries for direct GIS and spatial SQL analysis:
- `unit_id` (`BIGINT`): Spatial unit identifier.
- `centroid_x` (`DOUBLE`): Centroid X / Longitude coordinate.
- `centroid_y` (`DOUBLE`): Centroid Y / Latitude coordinate.
- `area` (`DOUBLE`): Polygon geometric surface area.
- `point_count` (`BIGINT`): Number of raw observations falling within the unit.
- `wkt` (`VARCHAR`): `POLYGON((x1 y1, x2 y2, ...))` standard geometry string.

### 3.6. `plot_data_<component>`
Separate relational tables for each diagnostic plot generated by `bstm_plots`:
- `plot_data_ppc`: Posterior predictive check distribution data.
- `plot_data_sre_spatial`: Spatial unit IDs, spatial random effects, areas, and centroid coordinates.
- `plot_data_time_series`: Temporal indices, trends, and ribbon bounds.
- `plot_data_fixed_effects`: Covariate names, coefficients, and posterior probabilities ($\mathbb{P}(\beta > 0)$).

### 3.7. `posterior_samples`
Raw MCMC draws formatted for high-speed SQL analytics:
- **Tidy Format** (`format=:tidy`): `iteration` (`BIGINT`), `chain` (`BIGINT`), `parameter` (`VARCHAR`), `value` (`DOUBLE`).
- **Wide Format** (`format=:wide`): `iteration` (`BIGINT`), `chain` (`BIGINT`), followed by a dedicated column for each parameter.

### 3.8. `models_registry` (for Ensembles)
Central comparison table across competing models in `save_model_ensemble`:
- `model_name` (`VARCHAR`): Model identifier (e.g. `"bym2"`, `"leroux"`, `"spde"`).
- `rmse` (`DOUBLE`): Root Mean Squared Error.
- `waic` (`DOUBLE`): Widely Applicable Information Criterion.
- `delta_waic` (`DOUBLE`): $\Delta \text{WAIC}_k = \text{WAIC}_k - \min_j \text{WAIC}_j$.
- `bma_weight` (`DOUBLE`): Normalized Bayesian Model Averaging weight:
  $$w_k = \frac{\exp(-\frac{1}{2}\Delta \text{WAIC}_k)}{\sum_j \exp(-\frac{1}{2}\Delta \text{WAIC}_j)}$$

---

## 4. API Function Reference

### 4.1. JLD2 Model State Persistence

#### `save_bstm_model`
```julia
save_bstm_model(filepath::AbstractString, model::DynamicPPL.Model; 
                chain=nothing, au=nothing, metadata::Dict=Dict(), compress::Bool=true) -> String
```
Serializes the complete Turing `@model` state, configuration `M`, data, and optional MCMC chain to JLD2.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `filepath` | `AbstractString` | *Required* | Path to output file (`.jld2` or `.bstm`). |
| `model` | `DynamicPPL.Model` | *Required* | Instantiated `bstm` Turing model. |
| `chain` | `Any` | `nothing` | Optional MCMC chain (`FlexiChain` or `MCMCChains.Chains`). |
| `au` | `NamedTuple` | `nothing` | Optional spatial areal units object. |
| `metadata` | `Dict` | `Dict()` | Arbitrary user metadata (experiment ID, git commit, notes). |
| `compress` | `Bool` | `true` | Enables zlib HDF5 compression on disk. |

#### `load_bstm_model`
```julia
load_bstm_model(filepath::AbstractString; calling_module::Module=Main) -> NamedTuple
```
Loads a saved model from `.jld2` and re-instantiates a live, callable `DynamicPPL.Model`.

**Returns:**
- `model`: Instantiated, callable `DynamicPPL.Model` ready for `sample()`, `predict()`, or `model_results_comprehensive()`.
- `chain`: Saved MCMC chain object (or `nothing`).
- `au`: Spatial areal units object (or `nothing`).
- `metadata`: Saved metadata dictionary.

---

### 4.2. DuckDB Results & Analytical Database Persistence

#### `save_bstm_results`
```julia
save_bstm_results(duckdb_path::AbstractString, res::NamedTuple; 
                  model=nothing, chain=nothing, au=nothing, table_prefix::String="", overwrite::Bool=true) -> String
```
Persists post-processing results (`res` from `model_results_comprehensive`) into an analytical DuckDB database without data redundancy.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `duckdb_path` | `AbstractString` | *Required* | Path to DuckDB database (`.duckdb` or `.db`). |
| `res` | `NamedTuple` | *Required* | Result object from `model_results_comprehensive`. |
| `model` | `DynamicPPL.Model` | `nothing` | Associated model object (used to extract observed outcomes). |
| `chain` | `Any` | `nothing` | Associated MCMC chain object (used to export raw draws). |
| `au` | `NamedTuple` | `nothing` | Areal units object (writes WKT `spatial_geometries` table). |
| `table_prefix` | `String` | `""` | Optional prefix for table names (e.g. `"m1_"`). |
| `overwrite` | `Bool` | `true` | Overwrites existing tables if present. |

#### `load_bstm_results`
```julia
load_bstm_results(duckdb_path::AbstractString; table_prefix::String="") -> NamedTuple
```
Reads saved tables from DuckDB and reconstructs a structured NamedTuple matching `model_results_comprehensive`.

#### `query_duckdb`
```julia
query_duckdb(duckdb_path::AbstractString, sql_query::AbstractString) -> DataFrame
```
Executes an arbitrary SQL query against a BSTM DuckDB database and returns the result as a DataFrame.

---

### 4.3. Spatial GIS & GeoJSON Export

#### `export_spatial_results_to_geojson`
```julia
export_spatial_results_to_geojson(geojson_path::AbstractString, res::NamedTuple, au::NamedTuple; 
                                   property_keys=nothing) -> String
```
Exports spatial model estimates and polygon geometries into a standard RFC 7946 GeoJSON file for direct visualization in GIS applications (QGIS, ArcGIS, Mapbox, Leaflet, Kepler.gl).

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `geojson_path` | `AbstractString` | *Required* | Output file path (`.geojson`). |
| `res` | `NamedTuple` | *Required* | Results NamedTuple from `model_results_comprehensive`. |
| `au` | `NamedTuple` | *Required* | Areal units object from `assign_spatial_units`. |
| `property_keys` | `Vector{Symbol}` | `nothing` | Optional subset of column names to attach as feature properties. |

---

### 4.4. Sequential Bayesian Prior Extraction

#### `extract_posterior_priors`
```julia
extract_posterior_priors(source::Union{AbstractString, NamedTuple}; 
                        parameter_names=nothing, prior_family::Symbol=:normal) -> Dict{Symbol, Any}
```
Extracts posterior distributions from previous model runs (`res` or a DuckDB database) and constructs fitted prior distributions (`Normal(mean, std)` or `truncated(Normal(...))`) ready for sequential Bayesian updating in subsequent `bstm` models.

---

### 4.5. Multi-Model Ensembling & Bayesian Model Averaging (BMA)

#### `save_model_ensemble`
```julia
save_model_ensemble(duckdb_path::AbstractString, ensemble_dict::Dict; overwrite::Bool=true) -> DataFrame
```
Persists a collection of candidate models (e.g. `Dict(:bym2 => (model=m1, chain=c1, results=r1), :leroux => ...)`) into a unified DuckDB database with an automated `models_registry` table computing $\Delta \text{WAIC}$ and BMA weights.

#### `bma_weighted_predictions`
```julia
bma_weighted_predictions(duckdb_path::AbstractString) -> DataFrame
```
Computes Bayesian Model Averaged (BMA) predictions across all candidate models registered in the DuckDB database using their normalized WAIC weights. Returns `(obs_id, bma_pred_mean, bma_pred_sd)`.

---

### 4.6. Zero-Copy Parquet & CSV Export & Maintenance

#### `export_results_to_parquet`
```julia
export_results_to_parquet(duckdb_path::AbstractString, table_name::AbstractString, output_parquet_path::AbstractString) -> String
```
Exports a DuckDB table to a high-speed compressed Parquet file using DuckDB's zero-copy export engine.

#### `export_results_to_csv`
```julia
export_results_to_csv(duckdb_path::AbstractString, table_name::AbstractString, output_csv_path::AbstractString) -> String
```
Exports a DuckDB table to a CSV file.

#### `compact_duckdb`
```julia
compact_duckdb(duckdb_path::AbstractString)
```
Performs `VACUUM; ANALYZE;` on the DuckDB database to reclaim unallocated disk space and optimize index query statistics.

---

## 5. Practical Workflows & Use Cases

### Workflow 1: Exporting Spatial Risk Maps to QGIS & GeoJSON

Fit a spatial disease model, persist geometries to DuckDB, and export an RFC 7946 GeoJSON map for direct drag-and-drop GIS inspection:

```julia
using bstm, DataFrames, Random

# 1. Fit BYM2 model on Scottish Lip Cancer data
data_scot, _ = bstm_data()
df = data_scot.data
au = data_scot.au

m = @bstm(likelihood(y, family=poisson, log_offsets=log_pop) ~ intercept() + random(s_idx, model=bym2), df, W=au.W, verbose=false)
chn = sample(m, NUTS(), 300; progress=false)
res = model_results_comprehensive(m, chn; au=au)

# 2. Save complete bundle with WKT geometry table in DuckDB
save_bstm_bundle("output/scot_model", m, chn, res; au=au)

# 3. Export to GeoJSON for QGIS / Web Mapping
export_spatial_results_to_geojson("output/scot_risk_map.geojson", res, au)
```

---

### Workflow 2: Sequential Bayesian Updating Across Time (Continual Learning)

Fit a model on Year $T-1$, extract its posterior parameter distributions, and use them as informative empirical priors for Year $T$:

```julia
# 1. Fit Stage 1 Model on Historical Data
df_hist = filter(row -> row.year == 2023, df)
m_hist = @bstm(likelihood(y, family=poisson) ~ intercept() + fixed(elevation), df_hist, verbose=false)
chn_hist = sample(m_hist, NUTS(), 300; progress=false)
res_hist = model_results_comprehensive(m_hist, chn_hist)

# 2. Extract Posterior Priors from Historical Results
stage1_priors = extract_posterior_priors(res_hist)
println("Learned Prior for Elevation: ", stage1_priors[:beta_elevation])

# 3. Fit Stage 2 Model on New Incoming Data using Learned Priors
df_new = filter(row -> row.year == 2024, df)
m_new = @bstm(
    likelihood(y, family=poisson) ~ 
        intercept(prior = stage1_priors[:intercept]) + 
        fixed(elevation, prior = stage1_priors[:beta_elevation]) + 
        random(s_idx, model=bym2),
    df_new, W=au.W, verbose=false
)
chn_new = sample(m_new, NUTS(), 300; progress=false)
```

---

### Workflow 3: Multi-Model Ensembling & Bayesian Model Averaging (BMA)

Compare competing spatial random effect architectures and compute model-averaged predictions:

```julia
# 1. Fit Model A (IID), Model B (BYM2), Model C (Leroux)
mA = @bstm(likelihood(y, family=poisson) ~ intercept() + random(s_idx, model=iid), df, verbose=false)
cA = sample(mA, NUTS(), 300; progress=false); rA = model_results_comprehensive(mA, cA)

mB = @bstm(likelihood(y, family=poisson) ~ intercept() + random(s_idx, model=bym2), df, W=au.W, verbose=false)
cB = sample(mB, NUTS(), 300; progress=false); rB = model_results_comprehensive(mB, cB)

mC = @bstm(likelihood(y, family=poisson) ~ intercept() + random(s_idx, model=leroux), df, W=au.W, verbose=false)
cC = sample(mC, NUTS(), 300; progress=false); rC = model_results_comprehensive(mC, cC)

# 2. Register Ensemble in DuckDB
ensemble = Dict(
    :iid => (model=mA, chain=cA, results=rA),
    :bym2 => (model=mB, chain=cB, results=rB),
    :leroux => (model=mC, chain=cC, results=rC)
)
df_registry = save_model_ensemble("output/spatial_ensemble.duckdb", ensemble)
display(df_registry)

# 3. Compute BMA Model-Averaged Predictions
df_bma = bma_weighted_predictions("output/spatial_ensemble.duckdb")
display(first(df_bma, 5))
```

---

### Workflow 4: Zero-Copy Parquet Export for Python / R Data Pipelines

Export high-volume posterior predictions and spatial effects to compressed Parquet files for downstream processing in Python (Polars / PyArrow) or R (arrow / sf):

```julia
# Export specific tables directly from DuckDB to Parquet
export_results_to_parquet("output/scot_model.duckdb", "predictions", "exports/predictions.parquet")
export_results_to_parquet("output/scot_model.duckdb", "plot_data_sre_spatial", "exports/spatial_effects.parquet")

# Compact and optimize database
compact_duckdb("output/scot_model.duckdb")
```

---

## 6. Summary of Best Practices

1. **Use `save_bstm_bundle` for Complete Checkpoints**:
   Ensures both model reproducibility (`.jld2`) and relational SQL analytics (`.duckdb`) in a single call.
2. **Include Areal Units (`au`) for GIS Workflows**:
   Passing `au` writes WKT spatial geometries directly into DuckDB and allows one-line GeoJSON export via `export_spatial_results_to_geojson`.
3. **Leverage `save_model_ensemble` for Model Selection**:
   Computes WAIC weights and stacked BMA predictions without manually parsing individual chain files.
4. **Use `extract_posterior_priors` for Time-Sequential Studies**:
   Carries parameter uncertainties forward into new time horizons in a principled Bayesian manner.
