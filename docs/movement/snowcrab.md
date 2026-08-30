# Snow Crab (*Chionoecetes opilio*) Movement & Connectivity in Atlantic Canada

Comprehensive guide to snow crab (*Chionoecetes opilio*) movement estimation,
spatial connectivity, and empirical mark-recapture telemetry modeling in the
Northwest Atlantic (Scotian Shelf and Gulf of St. Lawrence) using **BSTM**
(*Bayesian Spatio-Temporal Models*).

---

## 1. Overview & Scientific Rationale

Snow crab (*Chionoecetes opilio*) undergo ontogenetic migrations and
density-dependent dispersal across continental shelf bathymetric channels, cold
intermediate layers (CIL), and soft substrate basins [@Choi_2011;
@Choi_et_al_2022]. Quantifying their transition dynamics, spatial residence
times, and macro-regional connectivity (e.g., between CFA 20–24, NAFO 4VWX, and
the Gulf of St. Lawrence) is critical for sustainable quota allocation, spatial
refugia design, and climate vulnerability assessment.

The BSTM movement module integrates Eulerian continuous-space advection-diffusion
physics with Lagrangian telemetry observations:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Snow Crab Movement Modeling Pipeline                            │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   [ Tagging Data Ingestion ]              [ Spatial & HSI Model Inputs ]               │
│   - docs/movement/data/tagging.jld2       - docs/movement/data/sppoly.jld2 (707 AUIDs) │
│   - docs/movement/data/tagging.duckdb     - docs/movement/data/hsi.jld2 (3D Array)     │
│   - OTN Acoustic + BIO Conventional       - 707 units × 27 years × 5000 posterior MCMC │
│               │                                      │                                 │
│               ▼                                      ▼                                 │
│   [ Temporal Discretization ]             [ 5 km Hexagonal Resharding ]                │
│   - aggregate_tagging_trajectories()      - reshard_to_hex_mesh(radius_km=5.0)         │
│   - :monthly (default) / :weekly          - Isotropic 6-neighbor adjacency graph W     │
│   - Filters high-frequency ping jitter    - Barycentric polygon transfer matrix P      │
│               │                                      │                                 │
│               └──────────────────┬───────────────────┘                                 │
│                                  ▼                                                     │
│         [ Habitat Suitability (HSI) & Seasonal Interpolation ]                         │
│         - HSI reference date: September 1st (τ = 244 / 365.25 ≈ 0.668)                 │
│         - Intra-annual linear interpolation to matching observation day-of-year        │
│         - Directed advection matrix A from ∇HSI(s, t)                                  │
│                                  │                                                     │
│                                  ▼                                                     │
│            [ Bayesian Transition Kernel Estimation (NUTS) ]                            │
│            - Latent ADR: Γ̄ = (I - v A - d L)⁻¹                                        │
│            - Likelihood: π_m(u_rec | u_rel, k) ∝ [Γ^k]_{u_rel, u_rec}                  │
│            - Fitted parameters: velocity β, diffusion D, heterogeneity β_het, σ        │
│                                  │                                                     │
│                                  ▼                                                     │
│       [ Trajectory Simulation, Connectivity & Visualizations ]                         │
│       - simulate_movement_paths(): Forward CRW dispersal trajectories                  │
│       - calculate_regional_connectivity(): Macro-regional transition matrix C_rs       │
│       - generate_movement_plots(): 9-panel publication-grade Leaflet diagnostic suite  │
│       - DuckDB & JLD2 persistence: movement_transition & tag_activity_summary tables   │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Empirical Telemetry Data Ingestion & Formats

The BSTM data engine ingests and cleans real-world snow crab tracking data from
both binary JLD2 archives and columnar DuckDB databases:

### 2.1. Supported Data Sources & Artifacts

1. **Tagging Database (`tagging.jld2` / `tagging.duckdb`)**:
   - Harmonized repository containing Ocean Tracking Network (OTN) continuous
     acoustic receiver detections and historical BIO conventional streamer /
     t-bar tag releases and recaptures (1990–present).
   - Depths are converted from historical fathoms to metres ($1.8288\text{ m/fathom}$).
   - Missing historical release coordinates and timestamps are imputed from
     reference study registries via `tag_to_study_id`.
   - Biological metrics (carapace width `cw`, chela height `chela`, carapace
     condition `cc`, and wet weight `wgt`) are standardized.

2. **Spatial Polygons (`sppoly.jld2`)**:
   - 707 areal unit polygon geometry covering the Scotian Shelf and Gulf of
     St. Lawrence.
   - Contains unit centroids `(lon, lat)`, boundary polygon coordinate rings,
     and the topological neighborhood connectivity graph `NB_graph`.

3. **Habitat Suitability Index (`hsi.jld2`)**:
   - 3D posterior array of dimensions $707\text{ AUIDs} \times 27\text{ years} \times 5000\text{ MCMC samples}$.
   - Evaluated annually at reference date **September 1st** (day of year 244).

### 2.2. Ingestion Example in Julia

```julia
using bstm, DataFrames, Dates, DuckDB, JLD2

# 1. Load tagging records directly from JLD2 or DuckDB
tagging = load_snowcrab_tagging_data(
    data_dir              = "docs/movement/data",
    sources               = [:otn, :bio],
    filter_dead           = true,
    time_threshold_days   = 30.0,
    dist_threshold_meters = 50.0
)

# 2. Compute individual summary metrics
stats = summarize_tag_activity(tagging)
first(stats, 10)
```

---

## 3. Temporal Discretization & Aggregation

Crustacean acoustic telemetry often records hundreds of pings per day at a
single receiver station, introducing high-frequency positional jitter that can
dominate long-term ontogenetic movement dynamics.

BSTM provides automated temporal binning via `aggregate_tagging_trajectories`:

```julia
# Aggregate telemetry pings into monthly or weekly discrete time buckets
tagging_agg = aggregate_tagging_trajectories(
    tagging;
    time_interval = :monthly  # Options: :monthly, :weekly, :biweekly, :daily, :raw
)
```

### Supported Intervals

| Interval | Temporal Bucket Definition | Recommended Use Case |
| :--- | :--- | :--- |
| `:monthly` | Calendar month beginning ($1^{\text{st}}$ of each month) | **Standard / Default**: Macro-regional migration & seasonal advection |
| `:weekly` | Monday of each calendar week | High-resolution seasonal residency & CIL response |
| `:biweekly` | 14-day rolling epochs from epoch zero | Multi-week transition modeling |
| `:daily` | Daily date ($00\text{:}00\text{ UTC}$) | High-density localized acoustic arrays |
| `:raw` | Native ping timestamps | Sub-daily trajectory diagnostics |

Within each `(tagid, time_bucket)`, spatial coordinates (`lon`, `lat`), depths
(`z`), and biological measurements are averaged over finite observations,
and individual trajectories with $\ge 2$ distinct observations are retained.

---

## 4. 5 km Hexagonal Spatial Resharding

To provide isotropic continuous spatial dispersal modeling, BSTM reshardts the
707 irregular assessment polygons onto a regular **5 km radius hexagonal mesh**:

```julia
# Load 707 source polygons
sppoly_data = load("docs/movement/data/sppoly.jld2")
au_source   = sppoly_data["sppoly"]

# Reshard to regular 5 km radius hexagonal mesh
au_hex = reshard_to_hex_mesh(
    au_source;
    radius_km = 5.0
)

println("Generated $(au_hex.n_units) hexagonal units (radius = 5 km)")
```

### Mathematical Formulation of Hexagonal Resharding

1. **Local Planar Projection**:
   Centroids $(\lambda, \phi)$ in longitude/latitude are projected to planar
   Cartesian coordinates $(x, y)$ in kilometres via local transverse Mercator:
   $$
   x = R \cdot (\lambda - \lambda_0) \cos\phi_0 \cdot \frac{\pi}{180}, \quad
   y = R \cdot (\phi - \phi_0) \cdot \frac{\pi}{180}
   $$
   where $R = 6371\text{ km}$ and $(\lambda_0, \phi_0)$ is the bounding box center.

2. **Equilateral Hexagonal Grid**:
   For hex radius $r = 5\text{ km}$, horizontal step $\Delta x = \sqrt{3} \, r$
   and vertical step $\Delta y = \frac{3}{2} \, r$. Alternate rows are shifted by
   $\frac{\Delta x}{2}$.

3. **Pruning & Boundary Filtering**:
   Hexagonal cells whose centroids exceed $1.5 \times r$ distance from all source
   polygon centroids are removed, leaving a compact mesh covering the continental
   shelf and Gulf.

4. **Barycentric Transfer Operator**:
   A spatial projection matrix $\mathbf{P} \in \mathbb{R}^{S_{\text{hex}} \times S_{\text{source}}}$
   is computed using inverse-distance barycentric weights:
   $$
   P_{i, j} = \frac{d(c_i^{\text{hex}}, c_j^{\text{src}})^{-2}}{\sum_{k} d(c_i^{\text{hex}}, c_k^{\text{src}})^{-2}}
   $$

---

## 5. Habitat Suitability (HSI) & Seasonal Interpolation

### 5.1. September 1st Reference Date Alignment

Annual summer research survey predictions in `hsi.jld2` represent habitat
suitability at **September 1st** (day of year 244, $\tau_{\text{ref}} = 244 / 365.25 \approx 0.668$).

When evaluating HSI for an animal tracked at decimal year $t = \text{year} + \tau$:

$$
t_{\text{ref}, y} = y + \frac{244}{365.25}
$$

For an observation date between reference survey epochs $t_{\text{ref}, y_1}$ and
$t_{\text{ref}, y_2}$, the habitat suitability field is linearly interpolated:

$$
\text{HSI}(s, t) = (1 - w) \cdot \text{HSI}(s, y_1) + w \cdot \text{HSI}(s, y_2), \quad
w = \frac{t - t_{\text{ref}, y_1}}{t_{\text{ref}, y_2} - t_{\text{ref}, y_1}}
$$

```julia
# Compute seasonal HSI for each tag observation
hsi_at_obs = interpolate_seasonal_hsi(hsi_array, au_source, tagging_agg.timestamp)
```

### 5.2. Directed Advection Matrix from HSI Gradients

Spatial advection drifts crab toward areas of higher habitat suitability:

$$
A_{ij} = \frac{W_{ij} \cdot \exp\left(\gamma \, \left(\text{HSI}_j - \text{HSI}_i\right)\right)}{\sum_{k \sim i} W_{ik} \cdot \exp\left(\gamma \, \left(\text{HSI}_k - \text{HSI}_i\right)\right)}
$$

where $W_{ij} \in \{0, 1\}$ is the symmetric 6-neighbor adjacency matrix and
$\gamma > 0$ is the advective sensitivity parameter.

---

## 6. Bayesian Markov Resolvent Movement Estimation

### 6.1. Resolvent Advection-Diffusion Operator

The spatial movement process follows continuous-time Eulerian advection-diffusion:

$$
\frac{\partial u}{\partial t} = \nabla \cdot (D \nabla u) - \nabla \cdot (\mathbf{v} u)
$$

On discrete graph $G = (V, E)$, the 1-step transition probability matrix
$\bar{\boldsymbol{\Gamma}}$ is given by the matrix resolvent:

$$
\bar{\boldsymbol{\Gamma}} = \left( \mathbf{I} - \beta \, \mathbf{A} - D \, \mathbf{L} \right)^{-1}
$$

where $\mathbf{L} = \operatorname{diag}(\mathbf{W} \mathbf{1}) - \mathbf{W}$ is
the graph Laplacian, $\beta \in [0, 1)$ is advective velocity, and $D \ge 0$ is
diffusion.

For a transition over elapsed duration $\Delta t = k$ intervals:

$$
\pi(u_{\text{rec}} \mid u_{\text{rel}}, k) = \left[ \bar{\boldsymbol{\Gamma}}^k \right]_{u_{\text{rel}}, u_{\text{rec}}}
$$

### 6.2. MCMC Estimation with Turing NUTS

```julia
# Configure model options
opts = MovementOptions(
    area_method      = :hexagonal,
    hex_radius_km    = 5.0,
    time_interval    = :monthly,
    sensitivity      = 1.5,
    diffusion_weight = 0.15,
    n_samples        = 500,
    n_warmup         = 200
)

# Run full movement estimation
result = run_snowcrab_movement(
    data_dir   = "docs/movement/data",
    output_dir = "docs/movement/output",
    opts       = opts
)
```

---

## 7. Universal Coordinate Transformations & State-Space Markov Bridges

### 7.1. Pure-Julia Coordinate Transformations via `CoordRefSystems.jl`

`bstm` integrates `CoordRefSystems.jl` (pure Julia coordinate reference systems
from the *JuliaEarth* ecosystem) to perform exact geodesic and projected
coordinate transformations without external C/C++ binary dependencies:

```julia
using bstm, CoordRefSystems, Unitful

# Transforms UTM Zone 20N (km) to WGS84 Geographic (lon, lat) degrees
lon, lat = utm_to_lonlat(x_km, y_km; zone=20, is_km=true, northern=true)

# Inverse transform: WGS84 Geographic (lon, lat) to UTM Zone 20N (km)
x_km, y_km = lonlat_to_utm(lon, lat; zone=20, is_km=true, northern=true)
```

In the interactive Leaflet rendering engine (`src/leaflet.jl`), `CoordinateTransformer`
automatically detects CRS signatures (e.g. Scotian Shelf UTM Zone 20N kilometer grids)
and projects spatial units, centroids, and movement paths directly onto real-world
basemaps.

### 7.2. Discrete Markov Bridge Trajectory Reconstruction

For conventional mark-recapture records where only release $(x_0, y_0, t_0)$ and
recapture $(x_T, y_T, t_T)$ are observed, `bstm` reconstructs the non-linear,
HSI-informed latent state-space path for each individual using an exact **discrete
Markov bridge sampler**:

$$
\mathbb{P}(X_t = j \mid X_{t-1} = i, X_T = u_{\text{rec}}) = \frac{\bar{\Gamma}_{ij} \cdot \left[ \bar{\boldsymbol{\Gamma}}^{T - t} \right]_{j, u_{\text{rec}}}}{\left[ \bar{\boldsymbol{\Gamma}}^{T - t + 1} \right]_{i, u_{\text{rec}}}}
$$

where $\bar{\boldsymbol{\Gamma}} = (\mathbf{I} - \beta \mathbf{A} - D \mathbf{L})^{-1}$ is
the posterior transition matrix incorporating directed habitat gradients $\nabla \text{HSI}$.
This avoids linear interpolation across land barriers and deep channels.

```julia
# Reconstruct state-space Markov bridge paths for all mark-recapture events
reconstructed_paths = reconstruct_mark_recapture_paths(
    tagging_agg,
    au,
    result.transition_matrix;
    time_interval = :monthly
)
```

---

## 8. Interactive Diagnostic Suite (Leaflet HTML)

BSTM exports 9 interactive Leaflet HTML maps, analytical dashboards, and a dedicated
multi-layer empirical trajectory explorer:

| File | Type | Diagnostic Purpose |
| :--- | :--- | :--- |
| `snowcrab_tag_trajectories.html` | Map | **Comprehensive Multi-Layer Trajectory Explorer**: Overlays all 3,175 empirical/Markov bridge trajectories, release markers, recapture markers, tessellation boundaries, and HSI choropleths with individual metadata tooltips. |
| `movement_dashboard.html` | Dashboard | **Unified Multi-Panel Dashboard**: Syncs HSI, diffusion, residence distribution, velocity vectors, and tracks with Chart.js distance decay curves. |
| `hsi_choropleth.html` | Map | 5 km hexagonal spatial choropleth of Habitat Suitability Index. |
| `diffusion_map.html` | Map | Spatial variation in isotropic diffusion rate ($D_i$). |
| `residence_time_map.html` | Map | Stationary Markov distribution ($\boldsymbol{\pi} \bar{\boldsymbol{\Gamma}} = \boldsymbol{\pi}$). |
| `advection_arrows.html` | Map | Vector quiver field of advective velocity ($\mathbf{v}_i$) over HSI contours. |
| `movement_tracks.html` | Map | Simulated forward CRW trajectories and empirical tag tracks. |
| `dispersal_kernel.html` | Chart | Empirical binned displacement vs. exponential decay curve ($e^{-d/\lambda}$). |
| `step_diagnostics.html` | Chart | Step length distribution and turning angle histograms. |
| `regional_connectivity.html` | Heatmap | Macro-regional transition matrix ($C_{rs}$) between CFA areas. |

---

## 9. CLI Command Reference

Execute complete runs directly from the command line:

```bash
# 1. Full production run with monthly aggregation & 5 km hexagonal mesh:
julia --project=. docs/movement/snowcrab.jl \
    --time-interval=monthly \
    --hex-radius=5.0 \
    --samples=500 \
    --warmup=200 \
    --plot

# 2. Fast visualization of existing model fits & all mark-recapture tracks:
julia --project=. docs/movement/snowcrab.jl --viz

# 3. Fast weekly discretization test:
julia --project=. docs/movement/snowcrab.jl \
    --time-interval=weekly \
    --hex-radius=5.0 \
    --samples=200 \
    --warmup=100

# 4. Pure movement simulation benchmark with test plots:
julia --project=. docs/movement/movement_simple.jl \
    --test \
    --plot \
    --plot-dir=docs/movement/output/plots
```

---

## 10. Scientific References

- **Choi, J.S.** (2011). Habitat Preferences of the Snow Crab, *Chionoecetes opilio*: Where Stock Assessment and Ecology Intersect. *Biology and Management of Exploited Crab Populations under Climate Change*, Alaska Sea Grant, 361–376.
- **Choi, J.S., Cameron, B., Christie, K., Glass, A., & MacEachern, E.** (2022). Temperature and depth dependence of the spatial distribution of snow crab. *bioRxiv*, doi:10.1101/2022.12.20.520893.
- **Comeau, M., & Conan, G.Y.** (1992). Morphometry and gonad maturity of male snow crab, *Chionoecetes opilio*. *Canadian Journal of Fisheries and Aquatic Sciences*, 49: 2460–2468.
- **Plummer, M.** (2015). Cuts in Bayesian graphical models. *Statistics and Computing*, 25(1): 37–43.
