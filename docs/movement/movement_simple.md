---
title: "Pure Movement Estimation from Telemetry in BSTM"
subtitle: "Mark-Recapture Transition Kernel Inference with Optional Habitat Suitability Advection"
author: "BSTM Development Group"
date: "2026-08-24"
format:
  html:
    toc: true
    toc-depth: 3
    number-sections: true
    code-fold: show
    code-summary: "Show Julia Code"
    theme: cosmo
    highlight-style: github
  pdf:
    toc: true
    number-sections: true
    latex-engine: lualatex
bibliography: ../references.bib
csl: ../chicago-author-date.csl
---

# Pure Movement Estimation from Telemetry

## 1. Overview

This subproject provides a **stripped-down, self-contained** workflow for fitting
a spatial animal movement model when **no concurrent abundance or density survey data are
available**. The sole observational signal is a set of individual acoustic or satellite
telemetry detections — the mark-recapture encounter sequences:

$$
(u_{\text{rel}}, t_{\text{rel}}) \to (u_{\text{rec},1}, t_1) \to \cdots \to (u_{\text{rec},n}, t_n)
$$

where $u \in \{1, \dots, S\}$ is a discrete spatial unit and $t$ is a decimal-year timestamp.

**Key design decisions**:

1. **Movement-only**: No abundance, density, occupancy, or resource dynamics are
   modelled. This keeps the model identifiable and fast when the science question is purely
   about dispersal, connectivity, or habitat use.
2. **Optional Habitat Suitability (HSI)**: If an HSI surface is available from a
   prior ecological assessment (e.g., from the hierarchical workflow), it is incorporated
   into the **directed advection operator** $\mathbf{A}$, biasing movement towards
   higher-suitability areas. Without an HSI, the model reduces to a spatially isotropic
   random walk.
3. **Single source of movement truth**: All movement functions are sourced directly from
   [`src/movement.jl`](../../src/movement.jl) and the `movement` component in
   [`src/components/movement.jl`](../../src/components/movement.jl). No duplicated code.
4. **Segmented, restartable**: Each logical step (tessellation, fitting, simulation) can be
   called independently.

The companion executable is:

👉 [`movement_simple.jl`](movement_simple.jl)

### A note on self-contained scripts ###
This workflow is intentionally "self-contained" such that
```julia
include("docs/movement/movement_simple.jl")
res = main()
```
works immediately after cloning, without requiring any prior data preprocessing.

And to change defaults:

```julia
using bstm
include("docs/movement/movement_simple.jl")
main(["--test", "--samples=100", "--warmup=50"])
main(["--status", "--db=docs/movement/output/movement_test.duckdb"])
main(["--simulate", "--jld2=docs/movement/output/movement_test.jld2", "--n-paths=20"])

```


---

## 2. Mathematical Formulation

### 2.1. Telemetry Data & Observation Model

For each tagged individual $m$, the available data are an ordered sequence of $n_m + 1$
detection events:

$$
\mathcal{D}_m = \{(u_{\text{rel}}^{(m)}, t_0^{(m)}), (u_1^{(m)}, t_1^{(m)}), \dots, (u_{n_m}^{(m)}, t_{n_m}^{(m)})\}
$$

The **mark** event has `tag = 0`; recapture events have `tag = 1, 2, …`. The elapsed
time between consecutive events defines a number of discrete movement steps:

$$
k_{i,i+1} = \operatorname{round}\!\left(\frac{t_{i+1} - t_i}{\Delta t_{\text{unit}}}\right) \ge 1
$$

where $\Delta t_{\text{unit}} = 1$ year by convention (the `time` column is in decimal years).

The **observation likelihood** for a single transition is Categorical over the $S$ spatial units:

$$
\pi_m\!\left(u_{\text{rec}} \mid u_{\text{rel}}, k, z_m\right)
= \frac{\left[\boldsymbol{\Gamma}^k\right]_{u_{\text{rel}}, u_{\text{rec}}} \cdot \exp(\beta_{\text{het}}\, z_m)}
       {\sum_{j=1}^{S} \left[\boldsymbol{\Gamma}^k\right]_{u_{\text{rel}}, j} \cdot \exp(\beta_{\text{het}}\, z_m)}
$$

where $z_m$ is the optional individual covariate (column `individual_covariate`) and
$\beta_{\text{het}}$ is its mixing coefficient.

### 2.2. Advection-Diffusion Propagator

The one-step transition kernel $\boldsymbol{\Gamma} \in \mathbb{R}^{S \times S}$ is the
row-normalised inverse of the **ADR propagator matrix**:

$$
\mathbf{M} = \mathbf{I} - v\, \mathbf{A} - \mathbf{D}\, \mathbf{L}
$$

$$
\boldsymbol{\Gamma} = \text{row\_norm}\!\left(\mathbf{M}^{-1}\right)
$$

where:

| Symbol | Meaning |
| :--- | :--- |
| $v \ge 0$ | Fitted advection velocity (posterior parameter) |
| $\mathbf{A} \in \mathbb{R}^{S \times S}$ | Directed advection operator (from HSI or topology) |
| $D \ge 0$ | Fitted isotropic diffusion coefficient (posterior parameter) |
| $\mathbf{L} = \mathbf{D}_{\text{deg}} - \mathbf{W}$ | Discrete graph Laplacian |

**Without HSI** (`opts.hsi = nothing`): $\mathbf{A}$ is derived from the lower triangle of
the spatial adjacency matrix $\mathbf{W}$, yielding spatially isotropic dispersal.

**With HSI** (`opts.hsi = hsi_vec`): $\mathbf{A}$ is constructed from the directed HSI
gradient across adjacent pairs $(i, j)$:

$$
W_{\text{dir},ij} = \begin{cases}
W_{ij} \cdot \exp(\Delta h_{ij}) & \text{if } \Delta h_{ij} > 0 \quad (\text{`:exponential`})\\
W_{ij} \cdot \frac{1}{1 + e^{-4\Delta h_{ij}}} & (\text{`:logistic`})\\
W_{ij} \cdot \Delta h_{ij} & (\text{`:linear`})
\end{cases}, \quad \Delta h_{ij} = \text{HSI}_j - \text{HSI}_i
$$

$\mathbf{A} = \mathbf{D}_{\text{out}}^{-1} \mathbf{W}_{\text{dir}}$ is then row-normalised.

### 2.3. Priors

| Parameter | Prior | Description |
| :--- | :--- | :--- |
| $v$ | $\mathcal{N}^+(0, 1)$ | Advection velocity |
| $D$ | $\mathcal{N}^+(0, 1)$ | Isotropic diffusion |
| $\sigma$ | $\mathcal{N}^+(0, 0.5)$ | Latent field innovation scale |
| $\beta_{\text{het}}$ | $\mathcal{N}(0, 1)$ | Individual covariate heterogeneity |

These are weakly informative defaults. Override via a custom `@bstm` call if needed.

---

## 3. Input Data Format

### 3.1. Required Telemetry Columns

| Column | Type | Description |
| :--- | :--- | :--- |
| `tagid` | `Int` | Unique individual tag identifier |
| `lon` | `Float64` | Longitude (decimal degrees) or easting (metres) |
| `lat` | `Float64` | Latitude (decimal degrees) or northing (metres) |
| `time` | `Float64` | Decimal year (e.g. `2021.5` = July 2021) |
| `tag` | `Int` | Detection type: `0` = mark / release; `1` = 1st recapture; `n` = nth |

### 3.2. Optional Column

| Column | Type | Description |
| :--- | :--- | :--- |
| `individual_covariate` | `Float64` | Individual-level covariate for heterogeneity term $\beta_{\text{het}} z_m$ |

### 3.3. Optional HSI Vector & Automated Resharding

A `Vector{Float64}` of suitability values in $[0, 1]$. By default, `movement_simple` uses a `:hexagonal` spatial tessellation. If the input HSI vector originates from a different spatial resolution or geometry (such as a regular Cartesian grid, CVT mesh, or master survey grid), `movement_simple` automatically reshard/interpolates the HSI field onto the destination hexagonal tessellation using `reshard_hsi_field`.

---

## 4. Configuration: `MovementOptions`

```julia
opts = MovementOptions(
    n_units          = 40,              # target spatial units
    area_method      = :hexagonal,      # tessellation: :hexagonal (default), :cvt, :voronoi, :grid
    hsi              = nothing,         # Vector{Float64} of length S_src or S_dest
    reshard_hsi      = false,           # automatically reshard HSI if geometries differ
    au_hsi           = nothing,         # optional explicit source tessellation
    hsi_coords       = nothing,         # optional source unit coordinates [(lon, lat), ...]
    hsi_area_method  = :grid,           # source geometry type: :grid (default), :cvt, :hexagonal
    relationship     = :exponential,    # HSI-to-advection: :exponential, :linear, :logistic
    sensitivity      = 1.0,             # β for the prior deterministic kernel (diagnostic)
    diffusion_weight = 0.1,             # isotropic baseline for prior kernel
    method           = :explicit,       # Euler scheme: :explicit (AD) or :implicit
    n_samples        = 500,             # NUTS samples
    n_warmup         = 200,             # NUTS warmup
    n_chains         = 1,               # parallel chains
    rng              = MersenneTwister(42)
)
```

---

## 5. Julia API: Step-by-Step

### 5.1. Full One-Call Workflow

```julia
using bstm, DataFrames, CSV

tel = CSV.read("data/telemetry.csv", DataFrame)

# Without HSI: isotropic dispersal
result = run_movement_simple(tel)

# With HSI from a previous ecological assessment
hsi_vec = CSV.read("data/hsi_by_unit.csv", DataFrame).hsi
opts    = MovementOptions(hsi=hsi_vec, n_units=50, relationship=:exponential)
result  = run_movement_simple(tel; opts=opts)

# Persist results
save_movement_bundle(result; db_path="movement.duckdb", jld2_path="movement.jld2")
```

### 5.2. Independent Segment Calls

Each stage can be called and inspected separately:

```julia
# Stage 1: Build tessellation only — inspect unit placement before MCMC
au = build_tessellation(tel; opts=MovementOptions(n_units=40))

# Stage 2: Fit the kernel (detaches from tessellation inspection)
result = run_movement_simple(tel; opts=opts)

# Stage 3: Compute the deterministic prior kernel (no MCMC needed)
kernel_det = compute_prior_kernel(au; opts=opts)

# Stage 4: Simulate dispersal paths from the fitted kernel
paths = simulate_movement_paths(result; n_paths=20, n_steps=30, rho_persistence=0.8)
```

### 5.3. Multi-Recapture Sequences (n > 1)

Multiple recaptures per individual are handled automatically by `_process_telemetry_data`.
Each consecutive pair of detections generates an independent transition entry:

```
tagid=7: release(t=2020.0, unit=3) → recap1(t=2020.5, unit=12) → recap2(t=2021.1, unit=22)
         generates two transitions: (3→12, k=1) and (12→22, k=1)
```

### 5.4. Resuming / Extending a Fitted Model

```julia
# Load a previously saved JLD2
using JLD2
saved  = JLD2.load("movement.jld2")
Gamma_old = saved["transition_matrix"]

# Simulate new paths without re-fitting
result_loaded = (
    transition_matrix = Gamma_old,
    au                = saved["au"],
    opts              = saved["opts"]
)
new_paths = simulate_movement_paths(result_loaded; n_paths=50, n_steps=40)
```

### 5.5. Mock-Data Simulation Test (`run_simulation_test`)

A self-contained end-to-end test is provided using the **same synthetic data generator**
as [`hierarchical_workflow.jl`](../hierarchical_workflow/hierarchical_workflow.jl) — no
real data files are needed.

#### Data sources

| Data | Generator call | Description |
| :--- | :--- | :--- |
| **Telemetry** | `bstm_data(type="telemetry", seed=42, ...)` | 100 tagged individuals, 600 km domain, 5 survey years, CVT tessellation |
| **HSI** | Analytical Gaussian blob on unit centroids | Mimics the Tier 4 HSI surface; peaks at domain centre and decays radially |

The HSI is constructed as:

$$
\text{HSI}_i = \frac{\exp\!\left(-\frac{(x_i - x_c)^2 + (y_i - y_c)^2}{r^2}\right) - \min}{\max - \min} \cdot 0.9 + 0.05
$$

where $r = \text{domain}/3$ and the result is clipped to $[0.05, 0.95]$.

#### Calling sequence

```julia

using bstm

# Include the script
include("docs/movement/movement_simple.jl")

# ── Run with default settings ───────────────────────────────────────────────
# Uses bstm_data telemetry + Gaussian-blob HSI, 200 NUTS samples (fast)
# test = run_simulation_test()

# ── Run with world-age fix ──────────────────────────────────────────────────
# Wrapping in a function or using invokelatest avoids the "method too new" error
# caused by dynamically generated Turing models.

function run_test_safe()
    # Using invokelatest ensures we can call the model generated by the @bstm macro
    # even if it was created in the current world age.
    Base.invokelatest(run_simulation_test)
end

test = run_test_safe()

# ── Inspect results ─────────────────────────────────────────────────────────
# Posterior-mean transition kernel (S × S row-stochastic matrix)
test.result.transition_matrix

# Simulated CRW paths: (12 individuals × 21 time-steps) matrix of unit indices
test.paths

# 2×2 East / West regional connectivity matrix
test.C_regional

# Deterministic HSI-weighted kernel (pre-MCMC comparison)
test.kernel_prior

# ── Run without HSI to compare isotropic vs. directed dispersal ─────────────
test_iso = run_simulation_test(with_hsi=false)

# ── Compare regional structure ───────────────────────────────────────────────
println("Directed (HSI):  East→East = ", round(test.C_regional[1,1], digits=3))
println("Isotropic:       East→East = ", round(test_iso.C_regional[1,1], digits=3))

# ── Resume with more paths from the saved kernel ────────────────────────────
more_paths = simulate_movement_paths(test.result;
    n_paths=50, n_steps=40, rho_persistence=0.8)

# ── Try logistic functional form ─────────────────────────────────────────────
test_log = run_simulation_test(relationship=:logistic, n_samples=200)
```

#### Expected console output (abridged)

```
========================================================
  BSTM Movement Simulation Test
  (matching hierarchical_workflow.jl mock data)
========================================================
  Generating telemetry data (n_marks=100, domain=600.0 km) …
  Tessellation: 35 units (cvt)          # CVT may yield fewer than n_units=36
  Telemetry: 200 rows, 100 individuals
  HSI: Gaussian blob HSI (S=35)

  ========================================================
    BSTM  Pure Movement Estimation
  ========================================================
    Tags : 100
    Rows : 200
    Building spatial tessellation (36 target units, method=:cvt) …
    Actual units: 36
    Fitting movement model (200 samples, 100 warmup, chains=1) …
    Reconstructing posterior-mean transition kernel Γ̄ …
    Posterior velocity : 0.712
    Posterior diffusion: 0.348

  Simulated 12 CRW paths × 20 steps.

  Regional connectivity (East / West):
    West → West: 0.641   West → East: 0.359
    East → West: 0.312   East → East: 0.688

  Outputs written to: docs/movement/output
========================================================
```

#### CLI shortcut

```bash
# Run the simulation test directly from the shell
julia --project=. -e '
include("docs/movement/movement_simple.jl")
run_simulation_test()
'
```

## 6. Movement Visualizations & Leaflet HTML System

The BSTM movement module provides an interactive Leaflet HTML visualization engine by default (with optional static PNG figure export). All maps include panning, multi-scale zooming, dynamic tooltips, coordinate auto-projection, customizable colorbars, layer switcher controls, and interactive Chart.js analytical charts:

```julia
using bstm

# 1. Map individual movement trajectories over Voronoi / hexagonal units
m_tracks = leaflet_tracks_map(paths, result.au; hsi=opts.hsi, max_paths=20)
# Or static fallback: plot_tracks_on_map(paths, result.au; mode=:plots)

# 2. Habitat Suitability Index (HSI) polygon choropleth
m_hsi = leaflet_hsi_map(opts.hsi, result.au; cmap=:viridis)

# 3. Spatial diffusion rate field map
m_diff = leaflet_diffusion_map(0.5, result.au; cmap=:plasma)

# 4. Long-term stationary residence distribution (π Γ = π)
m_res = leaflet_residence_time_map(result.transition_matrix, result.au)

# 5. Advection vector arrow / quiver plot with directional headings & speed
m_arrows = leaflet_advection_arrows(result.au; hsi=opts.hsi, velocity=1.2, arrow_scale=1.2)

# 6. Dispersal distance decay curve with fitted exponential attenuation (Chart.js)
m_kernel = leaflet_dispersal_kernel(result.transition_matrix, result.au)

# 7. Step lengths and turning angle distributions (Chart.js)
m_steps = leaflet_step_diagnostics(paths, result.au)

# 8. Macro-regional connectivity heatmap with transfer probabilities (Chart.js)
m_conn = leaflet_regional_connectivity(C_regional; strata_names=["West", "East"])

# 9. Unified Multi-Panel Movement Diagnostics Dashboard
# Synchronizes interactive Leaflet map with 4 Chart.js summary statistic panels & KPI cards:
m_dash = leaflet_movement_dashboard(result, paths; hsi=opts.hsi)

# Generate and save all 9 interactive HTML maps and dashboard in a single call (default mode=:leaflet):
generate_movement_plots(result, paths; output_dir="output/plots", mode=:leaflet)
```

| Output HTML File | Visualization Description |
| :--- | :--- |
| `movement_dashboard.html` | **Unified composite web app**: Multi-layer Leaflet map (HSI, diffusion, residence time, velocity vectors, paths) integrated with 4 live Chart.js analytics panels and KPI summary cards. |
| `hsi_choropleth.html` | Interactive polygon choropleth of Habitat Suitability Index with value popups. |
| `diffusion_map.html` | Interactive choropleth of spatial dispersal/diffusion rates ($D$). |
| `residence_time_map.html` | Long-term stationary spatial occupancy distribution ($\pi$) under fitted Markov kernel. |
| `advection_arrows.html` | Directional quiver vector field showing local advection velocities $\mathbf{v}_i$. |
| `movement_tracks.html` | Individual animal telemetry dispersal trajectories with release and recapture markers. |
| `dispersal_kernel.html` | Interactive transition probability decay curve vs pairwise geographic distance. |
| `step_diagnostics.html` | Empirical step displacement length and turning angle distribution histograms. |
| `regional_connectivity.html` | Interactive macro-regional transfer probability matrix chart ($C_{rs}$). |

---

## 7. CLI Reference

```bash
# Print status from an existing DuckDB results file
julia --project=. docs/movement/movement_simple.jl --status

# Fit from a telemetry CSV with default hexagonal geometry
julia --project=. docs/movement/movement_simple.jl \
    --telemetry=data/telemetry.csv \
    --area-method=hexagonal \
    --plot --plot-dir=output/plots

# Fit with HSI from regular grid and reshard to hexagonal destination
julia --project=. docs/movement/movement_simple.jl \
    --telemetry=data/telemetry.csv \
    --hsi=data/hsi_grid.csv \
    --hsi-geometry=grid \
    --reshard-hsi \
    --area-method=hexagonal \
    --relationship=exponential \
    --n-units=50 --samples=800 --warmup=300 \
    --plot

# Simulate paths from existing JLD2 and render track plots
julia --project=. docs/movement/movement_simple.jl \
    --simulate --n-paths=20 --n-steps=30 --seed=99 --plot
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--telemetry` | *(required for fit)* | Path to telemetry CSV |
| `--hsi` | *(none)* | Path to HSI CSV (single column or with lon,lat) |
| `--area-method` | `hexagonal` | Destination spatial tessellation (`hexagonal`, `cvt`, `voronoi`, `grid`) |
| `--reshard-hsi` | `false` | Enable automated HSI geometric resharding |
| `--hsi-geometry` | `grid` | Source geometry for HSI resharding (`grid`, `cvt`, `hexagonal`) |
| `--hsi-coords` | *(none)* | Optional CSV with `(lon, lat)` coordinates of source HSI |
| `--n-units` | `40` | Target spatial units |
| `--relationship` | `exponential` | HSI functional form |
| `--samples` | `500` | NUTS total samples |
| `--warmup` | `200` | NUTS warmup steps |
| `--chains` | `1` | Parallel MCMC chains |
| `--db` | `movement_results.duckdb` | Output DuckDB path |
| `--jld2` | `movement_model.jld2` | Output JLD2 checkpoint |
| `--plot` | `false` | Generate and save full diagnostic plot suite |
| `--plot-dir` | `plots/` | Directory to write generated PNG figures |
| `--status` | | Print DuckDB status and exit |
| `--simulate` | | Load JLD2 and simulate; skip MCMC |
| `--n-paths` | `10` | Simulated individuals |
| `--n-steps` | `20` | Forward simulation steps |
| `--seed` | `42` | RNG seed |

---

## 8. API Reference

| Function | Description |
| :--- | :--- |
| `run_movement_simple(tel; opts)` | Full workflow: hexagonal tessellation → HSI resharding → fit → kernel reconstruction. |
| `build_tessellation(tel; opts)` | Spatial tessellation only (defaults to `:hexagonal`). |
| `reshard_hsi_field(hsi, au_dest; ...)` | Geometric resharding of HSI from arbitrary source mesh onto destination. |
| `compute_prior_kernel(au; opts)` | Deterministic HSI-weighted kernel with optional resharding. |
| `simulate_movement_paths(result; ...)` | Forward path simulation from posterior-mean Γ̄. |
| `generate_movement_plots(result, paths; ...)` | Generate and save 9 diagnostic movement plots. |
| `plot_tracks_on_map(paths, au; ...)` | Map individual tracks with start/end markers. |
| `plot_hsi_choropleth(hsi, au; ...)` | Spatial polygon choropleth of HSI suitability. |
| `plot_diffusion_map(diff, au; ...)` | Spatial polygon choropleth of dispersal rate $D$. |
| `plot_residence_time_map(Gamma, au; ...)` | Stationary spatial residence distribution ($\pi$). |
| `plot_advection_arrows(au; ...)` | Quiver / arrow vector field of advective drift. |
| `plot_dispersal_kernel(Gamma, au; ...)` | Dispersal probability vs distance decay curve. |
| `plot_step_length_distribution(paths, au)` | Step length and turning angle histograms. |
| `plot_regional_connectivity_matrix(C_reg)` | Annotated macro-regional transfer heatmap. |
| `plot_movement_dashboard(result, paths; ...)` | 4-panel composite movement diagnostics figure. |
| `save_movement_bundle(result; ...)` | Persist chain and kernel to JLD2 and DuckDB. |
| `check_movement_status(db_path)` | Print DuckDB table status. |
| `validate_telemetry(df)` | Schema and consistency checks. |
| `map_telemetry_to_units(df, au)` | Project (lon, lat) → spatial unit index. |

Underlying movement engine (from `src/`):

| Function | Module | Description |
| :--- | :--- | :--- |
| `compute_suitability_transition_kernel` | `src/movement.jl` | Build Γ from HSI and W. |
| `simulate_posterior_trajectories` | `src/movement.jl` | CRW path simulation. |
| `calculate_multistep_transition` | `src/movement.jl` | Γ^k via matrix exponentiation. |
| `calculate_regional_connectivity` | `src/movement.jl` | Aggregate fine-scale Γ to regions. |
| `movement(...)` | `src/components/movement.jl` | BSTM formula component (ADR + telemetry). |

---

## 8. Relationship to the Hierarchical Workflow

This subproject is intentionally **decoupled** from the full hierarchical pipeline. When a
previously fitted HSI surface is available, it can be passed directly via `opts.hsi`:

```julia
# After running the hierarchical workflow:
hsi_from_pipeline = community_pred_df.hsi_mean  # length S_master

# Pass to the pure movement model:
opts = MovementOptions(hsi=hsi_from_pipeline, n_units=length(hsi_from_pipeline))
result = run_movement_simple(tel; opts=opts)
```

For coupled estimation of HSI and movement simultaneously, use the full Tier 4–5–7 pipeline
documented in [`docs/hierarchical_workflow/hierarchical_workflow.md`](../hierarchical_workflow/hierarchical_workflow.md).

---

## 9. References

::: {#refs}
:::
