# bstm: Bayesian Spatiotemporal Models in Julia

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10+-blue.svg)](https://julialang.org)
[![Turing.jl](https://img.shields.io/badge/Powered%20By-Turing.jl-purple.svg)](https://turinglang.org)
[![DuckDB](https://img.shields.io/badge/Analytics-DuckDB-yellow.svg)](https://duckdb.org)

The **`bstm`** framework provides a composable, formula-driven probabilistic programming interface for hierarchical Bayesian spatiotemporal modeling in Julia. Built on top of **Turing.jl** and Julia's scientific computing ecosystem, `bstm` separates observation likelihood specifications from latent process dynamics. This decoupling allows researchers to flexibly assemble spatial, temporal, non-linear, and mechanistic differential components into an integrated, differentiable probabilistic model.

Inspired by high-level formula interfaces like R's `brms` and `INLA`, `bstm` provides automated code generation, automatic differentiation (ForwardDiff, ReverseDiff, Zygote), adaptive composite block-sampling, full posterior reconstruction, spatial tessellation, analytical SQL querying, and publication-ready diagnostic visualization.

---

## Key Features

- **Intuitive Formula DSL (`@bstm`)**:
  Declarative formula syntax separating observation likelihoods (`likelihood(y, family=...)`) on the LHS from latent processes (`intercept() + fixed() + random()`) on the RHS.
- **Rich Latent Component Library (50+ Components)**:
  - **Discrete Spatial GMRFs**: `BYM2` (Besag-York-Mollié with spectral scaling), `ICAR`, `Besag`, `Leroux`, `MCAR` (Multivariate Conditional Autoregressive), `SAR` (Simultaneous Autoregressive), `LocalAdaptive` (cluster-specific regime shifts), and `BCGN`.
  - **Continuous Geostatistics**: `NNGP` (Nearest Neighbor Gaussian Process scaling to $N > 10^5$), Exact Gaussian Processes (`GP`), `SparseGP` (FITC pseudo-inputs), `RFF` (Random Fourier Features), and `SPDE` (triangulated mesh Matérn fields).
  - **Temporal & Seasonal**: `AR1`, `AR2`, `RW1` (stochastic level), `RW2` (stochastic curvature), `Harmonic` / `Cyclic` (Fourier seasonality), and `TAR` (Threshold Autoregressive).
  - **Nonparametric Smooths**: Penalized B-splines (`PSpline`), `BSpline`, Thin Plate Regression Splines (`TPS`), and `AdaptiveSmooth`.
  - **Mechanistic & Movement Dynamics**: Advection-Diffusion-Reaction (`movement()`), state-space dynamical systems (`dynamics()`), Bayesian factor analysis (`eigen()`), and ODE/SciML integration (`sciml()`).
- **Component Algebra & Compositional Operators**:
  - **Kronecker Interaction (`⊗`)**: Models non-separable space-time interactions (Knorr-Held Types I–IV, $Q_{st} = Q_t \otimes Q_s$).
  - **Pipe Operator (`|>`)**: Constructs Spatially-Varying Coefficients (SVC) and spatially-varying temporal curves (e.g. `covariate |> random(s_idx, model=icar)`).
  - **Composition (`∘`)**: Links intensity fields to point processes (e.g. Log-Gaussian Cox Processes `LGCP`).
- **Comprehensive Spatial Partitioning Subsystem**:
  - 9 automated spatial discretization algorithms: `:hexagonal` (honeycomb packing), `:cvt` (Lloyd's centroidal relaxation), `:kvt` (K-means density balancing), `:qvt`, `:bvt`, `:hvt`, `:avt`, `:lattice` (fast raster), and spring-layout coordinate inference.
  - Granular sizing and polygon count controls: `target_units`, `exact_units`, `target_area`, `min_area`, `merge_small_polygons`, `prune_empty`, and KD-tree island bridging (`ensure_connected!`).
  - Joint space-time synchronization: `assign_spatiotemporal_units` ($st = (t-1)S + s$).
- **Flexible Likelihoods & Observation Models**:
  - 19 distribution families: `:gaussian`, `:poisson`, `:negbin`, `:bernoulli`, `:binomial`, `:beta`, `:gamma`, `:lognormal`, `:studentt`, `:exponential`, `:weibull`, `:gev`, `:zipoisson`, `:zinegbin`, `:ordered_logistic`, `:ordered_probit`, `:categorical`, `:multinomial`, `:dirichlet`.
  - Observation modifiers: `log_offsets` (epidemiological rate modeling), `weights`, `trials`, `zero_inflated`, `hurdle`, `volatility`, `censor_lower`, and `censor_upper`.
- **Adaptive Sampler Optimization (`get_optimal_sampler`)**:
  Introspects parameter supports and constructs composite Gibbs samplers assigning `PG` to discrete variables, `ESS` to Gaussian latent vectors, `Slice` to bounded parameters, and `NUTS` to continuous blocks.
- **Two-Tier Model Persistence & Analytical SQL Engine (`src/input_output.jl`)**:
  - **Tier 1 (JLD2)**: Full binary serialization of live callable Turing models (`m`), configurations, data, and MCMC chains (`chn`).
  - **Tier 2 (DuckDB)**: Embedded relational SQL database storing normalized metrics, parameter statistics, predictions, WKT spatial geometries, and diagnostic plot data for zero-copy querying, multi-model Bayesian Model Averaging (BMA), sequential prior extraction, and Parquet/GeoJSON export.
- **Diagnostics, Post-Processing & Visualization (`src/plotting.jl`)**:
  One-line extraction of posterior credible intervals, predictive error metrics (RMSE, $R^2$, DIC, WAIC), spatial choropleth maps (`choropleth`), spatial adjacency graphs (`spatial_graph_plot`), animal movement paths (`render_paths!`), and timeseries ribbons (`timeseries_ci`).
- **Spatial Block Cross-Validation (`bstm_cv_orchestrator`)**:
  Assesses out-of-sample generalization using `:spatial_block`, `:temporal_block`, `:lolo` (leave-one-location-out), and `:temporal_forward_chain` to prevent spatial autocorrelation leakage.

---

## Installation & Setup

`bstm` is structured as a self-contained Julia project environment. Clone the repository and load the module:

```bash
git clone https://github.com/mum0n/bstm.git
cd bstm
# cd("c:/home/jae/projects/bstm") # where you saved bstm
```

Start Julia within the repository:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
 
include("bstm.jl")
using .bstm
```

## Quick Start Examples

### Example 1: Spatial Disease Mapping (Scottish Lip Cancer Data)

Fit a hierarchical BYM2 spatial model with an AR1 temporal trend, extract diagnostics, and persist to a DuckDB bundle:

```julia

Random.seed!(42)

# 1. Load benchmark dataset (56 Scottish districts across time)
data_scot, _ = bstm_data() # Scottish Lip Cancer
df = data_scot.data
W = data_scot.au.W

# 2. Specify Hierarchical Spatiotemporal Model
m = @bstm(
    likelihood(y, family=poisson, log_offsets=log_offsets) ~
        intercept() +
        fixed(cov1) +
        random(s_idx, model=bym2) +
        random(year, model=ar1),
    df,
    W = W,
    verbose = false
)

# 3. Sample from Posterior with NUTS
chn = sample(m, NUTS(), 30; progress=false)

# 4. Extract Comprehensive Diagnostics & Summaries (Pure Data)
res = model_results_comprehensive(m, chn)
println("Model WAIC: ", res.metrics.waic)
display(res.parameters)

# 5. Generate, Display, and Export Diagnostic Plots
plots_res = bstm_plots(res; au=data_scot.au, save_dir="output/plots")
display(plots_res.plots[:spatial])

# 6. Persist Unified Bundle to DuckDB and JLD2
save_bstm_bundle("output/scot_lip_model", m, chn, res; au=data_scot.au)
```

---

### Example 2: Continuous Point Data Partitioning with Hexagonal Binning

Discretize continuous GPS coordinates into exact regular hexagons, construct neighborhood topology, and fit a spatiotemporal model:

```julia
using bstm, DataFrames, Random, Plots

rng = MersenneTwister(42)
N = 400

# 1. Continuous point observations
df = DataFrame(
    lon = rand(rng, N) .* 100.0,
    lat = rand(rng, N) .* 50.0,
    year = rand(rng, 2020:2024, N),
    elevation = randn(rng, N),
    y = rand(rng, 0:20, N)
)

# 2. Partition space into exactly 16 regular hexagons across 5 years
st_data = assign_spatiotemporal_units(df;
    space_x = :lon,
    space_y = :lat,
    time_var = :year,
    area_method = :hexagonal,
    target_units = 16,
    exact_units = true,
    merge_small_polygons = true
)

df.s_idx = st_data.s_idx
df.year_idx = st_data.t_idx

# 3. Fit Spatiotemporal Model with BYM2 + AR1
m = @bstm(
    likelihood(y, family=poisson) ~
        intercept() +
        fixed(elevation) +
        random(s_idx, model=bym2) +
        random(year_idx, model=ar1),
    df,
    W = st_data.W_spatial,
    verbose = false
)

# 4. Process Results and Generate Plots
chn = sample(m, NUTS(), 300; progress=false)
res = model_results_comprehensive(m, chn)
plots_res = bstm_plots(res; data=df, au=st_data.au_spatial)

p_map = plots_res.plots[:spatial]
p_ppc = plots_res.plots[:posterior_predictive_check]
plot(p_map, p_ppc, layout=(1, 2), size=(1000, 450))
```

---

### Example 3: Analytical SQL Querying, Prior Extraction & Chain Extension

Leverage DuckDB for relational SQL analytics and resume sampling on previously saved models:

```julia
using bstm

# 1. Query previously saved model results directly via SQL
df_high_risk = query_duckdb("output/scot_lip_model.duckdb", """
    SELECT unit_id, sre_mean, sre_lower, sre_upper 
    FROM plot_data_sre_spatial 
    WHERE sre_mean > 1.0 
    ORDER BY sre_mean DESC
""")
display(df_high_risk)

# 2. Extract posterior parameters as informative priors for a new model
priors = extract_posterior_priors("output/scot_lip_model.duckdb")

# 3. Load model state and extend MCMC chain with 500 additional samples
bundle = load_bstm_bundle("output/scot_lip_model")
chn_extended = extend_sampling(bundle.model, bundle.chain, 500; progress=false)
```

---

## Documentation

Comprehensive guides and technical documentation are available in the `docs/` directory:

- [**Architectural & Methodological Overview** (`docs/bstm_overview.md`)](docs/bstm_overview.md):
  Design principles, formula syntax, component algebra, prior systems, and inference engines.
- [**Technical API Reference** (`docs/bstm_api.md`)](docs/bstm_api.md):
  Developer reference for the `ComponentModel` lifecycle, `ParamRegistry`, likelihood distributions, plotting functions, and AST parser.
- [**Spatial & Spatiotemporal Partitioning Guide** (`docs/bstm_spatial_partitioning.md`)](docs/bstm_spatial_partitioning.md):
  Mathematical formulations, Lloyd's relaxation, hexagonal geometry, MAUP mitigation, island bridging, and BYM2 spectral scaling.
- [**Input / Output & Persistence Guide** (`docs/bstm_input_output.md`)](docs/bstm_input_output.md):
  Two-tier persistence architecture, JLD2 model serialization, DuckDB analytical results storage, sample extension, SQL analytics, and GIS export.
- [**Custom Components & Spatial SEIR Modeling Guide** (`docs/bstm_custom.md`)](docs/bstm_custom.md):
  Mechanistic process modeling, raw Turing code injection with `custom()`, first-class `ComponentModel` implementation, and spatial SEIR disease dynamics.
- [**Movement & ADR Telemetry Guide** (`docs/bstm_movement.md`)](docs/bstm_movement.md):
  Advection-Diffusion-Reaction (ADR) population dynamics, Eulerian survey counts, Lagrangian mark-recapture telemetry, and trajectory path simulation.

---

## Citation & References

If you use `bstm` in your research, please cite:

1. **Besag, J.** (1974). Spatial interaction and the statistical analysis of lattice systems. *Journal of the Royal Statistical Society: Series B*, 36(2), 192–225.
2. **Hooten, M. B., & Hefley, T. J.** (2019). *Bringing Bayesian Models to Life*. CRC Press.
3. **Knorr-Held, L.** (2000). Bayesian modelling of inseparable space-time variation in disease risk. *Statistical Methods in Medical Research*, 9(3), 205–220.
4. **Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H.** (2016). An intuitive Bayesian spatial model for disease mapping that accounts for scaling. *Statistical Methods in Medical Research*, 25(4), 1145–1165.
5. **Ge, H., Xu, K., & Ghahramani, Z.** (2018). Turing: A language for flexible probabilistic programming. *International Conference on Artificial Intelligence and Statistics (AISTATS)*.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
