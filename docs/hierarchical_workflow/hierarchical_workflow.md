---
title: "Integrated Hierarchical Bayesian Workflows: Multi-Tier DAGs, Surface Derivatives, EIV Priors, Habitat Suitability & ADR Telemetry in BSTM"
subtitle: "Multi-Scale Cross-Mesh Resharding, Continuous Surface Differential Geometry, Second-Last Tier HSI Determination, and Lagrangian Telemetry"
author: "BSTM Development Group"
date: "2026-08-23"
format:
  html:
    toc: true
    toc-depth: 4
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

# Integrated Hierarchical Bayesian Workflows: Multi-Tier DAGs, Surface Derivatives, EIV Priors, Habitat Suitability & ADR Telemetry

## 1. Executive Summary & Design Philosophy

Complex ecological and environmental systems are inherently **multi-scale, multi-fidelity,
and multi-tiered**. Rather than attempting to estimate all physical, oceanographic, community,
and population processes inside a single monolithic, fragile joint likelihood, modern
Bayesian computation relies on **Modular Bayesian Inference (Cut-Posterior DAG Pipelines)**
[@Plummer_2015; @Jacob_2017; @Hooten_Hefley_2019].

The **Bayesian Spatio-Temporal Models (`bstm`)** ecosystem establishes a unified, principled
framework that connects:

1. **Continuous Surface Modeling & Differential Geometry**:
   Analytical derivation of slopes, aspects, profile/planform curvatures, and Bessel
   circular Bathymetric Position Indices ($\text{BPI}_r$) across 6 model families (RFF,
   SpectralGP, WaveletGP, SPDE, PSpline, TPS).
2. **Errors-in-Variables (EIV) Uncertainty Propagation**:
   Carrying upstream posterior parameter uncertainties forward as latent Gaussian measurement
   error priors ($x_i \sim \mathcal{N}(\mu_{x, i}, \sigma^2_{x, i})$) without feedback contamination.
3. **Multi-Scale Cross-Mesh Topological Transfer Operators**:
   Linear geometric transfer matrices ($\mathbf{P} \in \mathbb{R}^{N_{\text{dest}} \times N_{\text{src}}}$)
   and **Full Monte Carlo Matrix Resharding** ($\mathbf{U}_{\text{dest}} = \mathbf{P} \mathbf{U}_{\text{src}} \in \mathbb{R}^{N_{\text{dest}} \times S}$)
   to transfer posterior fields across mismatched irregular spatial tessellations.
4. **Second-Last Tier Habitat Suitability Index (HSI) Determination**:
   Deriving spatial habitat suitability probabilities ($\pi_{\text{HSI}}(\mathbf{s}) \in [0, 1]$)
   just prior to final biomass/management tiers using threshold quantile Bernoulli models,
   continuous sigmoidal transformations, or two-part hurdle formulations.
5. **Mechanistic Movement & Lagrangian Mark-Recapture Telemetry**:
   Advection-Diffusion-Reaction (ADR) population dynamics fusing Eulerian abundance surveys
   and Lagrangian individual tag tracking encounters ($u_{\text{rel}} \to u_{\text{rec}}$).
6. **Two-Tier Input-Output Persistence**:
   Out-of-core DuckDB relational analytics alongside JLD2 binary checkpointing and RFC 7946
   GeoJSON GIS export.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               The Complete 5-Tier Hierarchical Marine Ecological DAG                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   [ Tier 1: Continuous Bathymetry (RFF Surface) ]                                      │
│         │                                                                              │
│         ├──────────────────────────────┬──────────────────────────────┐                │
│         ▼ (Depth & Slope)              ▼ (Depth)                      ▼ (Depth)        │
│   [ Tier 2: Substrate ]         [ Tier 3: Oceanography ]       [ Tier 4: Community ]   │
│   (G_2: 25 grab units)          (G_3: 40 CTD units)            (G_4: 50 haul units)    │
│         │                              │                              │                │
│         │ (Full MC Matrix P_2->4)      │ (Full MC Matrix P_3->4)      │                │
│         └──────────────────────────────┴──────────────────────────────┤                │
│                                                                       ▼                │
│                                                          [ Tier 4b: HSI on G_4 ]       │
│                                                          (Quantile Threshold HSI)      │
│                                                                       │                │
│         ┌──────────────────────────────┬──────────────────────────────┤                │
│         │ (Grain Size P_2->M)          │ (Temperature P_3->M)         │ (HSI & PCs)    │
│         ▼                              ▼                              ▼                │
│   [ Tier 5: Target Species Biomass & ADR Movement Telemetry on Master Mesh G_master ]  │
│   - Eulerian Spatiotemporal Abundance Survey: y_{s, t} ~ Gamma / Poisson               │
│   - Lagrangian Mark-Recapture Encounters: π_m(u_rec | u_rel, k) ~ Categorical(Γ^k)     │
│   - Advective Velocity: v(s) ∝ ∇HSI(s)                                                 │
│                                                                                        │
│                                        ▼                                               │
│   [ Master Harmonized DuckDB Relational Summary Table & GeoJSON Refugia Export ]       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Formulations

### 2.1. Multi-Tier Cut-Posterior Theory

For a multi-tiered DAG with observed datasets $\mathbf{Y}_1, \dots, \mathbf{Y}_K$ and latent
parameters $\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_K$, classical joint inference
evaluates:

$$
p(\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_K \mid \mathbf{Y}_1, \dots, \mathbf{Y}_K) \propto \prod_{k=1}^K p(\mathbf{Y}_k \mid \boldsymbol{\theta}_k, \boldsymbol{\theta}_{\text{pa}(k)}) p(\boldsymbol{\theta}_k)
$$

In practice, misspecifications in downstream tiers (e.g., fisheries catchability) can leak
backward and corrupt physical parameters (e.g., bathymetry or ocean temperature). The
**cut-posterior distribution** [@Plummer_2015] severs this feedback loop:

$$
p_{\text{cut}}(\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_K \mid \mathbf{Y}_1, \dots, \mathbf{Y}_K) = p(\boldsymbol{\theta}_1 \mid \mathbf{Y}_1) \prod_{k=2}^K p(\boldsymbol{\theta}_k \mid \mathbf{Y}_k, \boldsymbol{\theta}_{\text{pa}(k)})
$$

---

### 2.2. Continuous Surface Derivatives & Topographic Differential Geometry

For any continuous surface model $z(\mathbf{s}) = \mathbf{B}(\mathbf{s}) \mathbf{c}$, `bstm`
computes exact spatial derivatives without numerical finite-difference grid artifacts:

1. **Gradient & Seabed Slope**:

   $$
   \nabla z(\mathbf{s}) = \begin{bmatrix} \frac{\partial z}{\partial x} \\ \frac{\partial z}{\partial y} \end{bmatrix}, \quad \text{Slope}(\mathbf{s}) = \arctan\left( \sqrt{ \left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2 } \right)
   $$

2. **Aspect (Compass Direction of Steepest Descent)**:

   $$
   \text{Aspect}(\mathbf{s}) = \operatorname{mod}\left( 180 - \operatorname{atan2}\left(\frac{\partial z}{\partial y}, \frac{\partial z}{\partial x}\right) \cdot \frac{180}{\pi}, 360 \right)
   $$

3. **Hessian & Curvatures**:

   $$
   \mathbf{H}(\mathbf{s}) = \begin{bmatrix} z_{xx} & z_{xy} \\ z_{xy} & z_{yy} \end{bmatrix}
   $$

   - **Profile Curvature** (parallel to steepest gradient):

     $$
     \kappa_{\text{profile}} = \frac{z_{xx} z_x^2 + 2 z_{xy} z_x z_y + z_{yy} z_y^2}{(z_x^2 + z_y^2) (1 + z_x^2 + z_y^2)^{3/2}}
     $$

   - **Planform Curvature** (perpendicular to gradient):

     $$
     \kappa_{\text{planform}} = \frac{z_{xx} z_y^2 - 2 z_{xy} z_x z_y + z_{yy} z_x^2}{(z_x^2 + z_y^2)^{3/2}}
     $$

4. **Continuous Bessel Circular Bathymetric Position Index ($\text{BPI}_r$)**:

   $$
   \text{BPI}_r(\mathbf{s}) = z(\mathbf{s}) - \bar{z}_{C_r(\mathbf{s})}
   $$

   For Random Fourier Features $z(\mathbf{s}) = \sum_{j=1}^M c_j \cos(\boldsymbol{\omega}_j^\top \mathbf{s} + b_j)$, the exact circular annulus mean evaluates analytically via the order-zero Bessel function of the first kind $J_0$:

   $$
   \bar{z}_{C_r(\mathbf{s})} = \sum_{j=1}^M c_j J_0(\|\boldsymbol{\omega}_j\| r) \cos(\boldsymbol{\omega}_j^\top \mathbf{s} + b_j)
   $$

---

### 2.3. Errors-in-Variables (EIV) Uncertainty Propagation

When downstream models ingest predictions $\hat{x}_i$ with standard errors $\sigma_{x, i}$ from
upstream tiers, naive point substitution induces **attenuation bias** (underestimating effect sizes).
`bstm` implements latent Gaussian measurement error priors:

$$
x_i \sim \mathcal{N}(\mu_{x, i}, \sigma^2_{x, i}), \quad \eta_i = \beta_0 + \beta_1 x_i + \dots
$$

Declared simply via:
```julia
fixed(depth_mu, error_sd=:depth_sd)
```

---

### 2.4. Cross-Mesh Topological Transfer & Full Monte Carlo Matrix Resharding

When transferring a spatial field from source graph $G_{\text{src}} = (V_{\text{src}}, E_{\text{src}})$
to destination graph $G_{\text{dest}} = (V_{\text{dest}}, E_{\text{dest}})$, the geometric
transfer operator $\mathbf{P} \in \mathbb{R}^{N_{\text{dest}} \times N_{\text{src}}}$ is constructed
via areal overlap intersection:

$$
P_{ij} = \frac{\operatorname{Area}(A_{\text{dest}, i} \cap A_{\text{src}, j})}{\operatorname{Area}(A_{\text{dest}, i})}
$$

```
                Source Mesh G_src                       Destination Mesh G_dest
           ┌───────────┬───────────┐                     ┌───────────────────┐
           │ A_src, 1  │ A_src, 2  │  ── Transfer ──>   │    A_dest, 1      │
           ├───────────┼───────────┤      Matrix P       │  (Overlaps 1,2,3) │
           │ A_src, 3  │ A_src, 4  │                     └───────────────────┘
           └───────────┴───────────┘
```

#### Full Monte Carlo Matrix Resharding (`mode = :samples`)
To preserve complete posterior covariance across non-Gaussian downstream likelihoods,
the entire posterior sample matrix is transformed directly:

$$
\mathbf{U}_{\text{dest}} = \mathbf{P} \mathbf{U}_{\text{src}} \in \mathbb{R}^{N_{\text{dest}} \times S}
$$

Summary statistics (median, mean, standard deviation, and credible intervals) are then
evaluated empirically without assuming normality:

```julia
resharded = reshard_spatial_field(pred_samples, au_src, au_dest; mode=:samples)
```

---

### 2.5. Second-Last Tier Habitat Suitability Index (HSI) Determination

Prior to the final management/biomass tier, the **Habitat Suitability Index ($\text{HSI}$)** is
determined on the haul network $G_4$ and resharded onto the canonical master mesh $G_{\text{master}}$.

#### 1. Quantile Threshold Binomial/Bernoulli Model (Default)
In survey datasets with zero catches or marginal low-density observations, trawl tows with total
community density below an empirical threshold quantile (default $p = 0.05$) are classified as
poor/unsuitable habitat ($z_i = 0$), and tows above as suitable habitat ($z_i = 1$):

$$
q_p = \operatorname{quantile}(\mathbf{y}, p), \quad z_i = \begin{cases} 1 & \text{if } y_i \ge q_p \\ 0 & \text{if } y_i < q_p \end{cases}
$$

$$
z_i \sim \operatorname{Bernoulli}(\pi_i), \quad \operatorname{logit}(\pi_i) = \beta_0 + \sum_{k} \beta_k x_{k, i} + \text{sre}_{\text{spatial}}(s_i) + \text{tre}_{\text{temporal}}(t_i)
$$

The predicted latent logit field $\eta(\mathbf{s})$ is transformed to the probability scale:

$$
\text{HSI}(\mathbf{s}) = \pi_{\text{HSI}}(\mathbf{s}) = \frac{1}{1 + \exp(-\eta(\mathbf{s}))} \in [0, 1]
$$

#### 2. Alternative Formulations & Comparative Trade-offs

| Formulation | Mathematical Definition | Key Advantages | Best Use Case |
| :--- | :--- | :--- | :--- |
| **Quantile Bernoulli** *(Default)* | $z_i = \mathbb{I}(y_i \ge q_p) \sim \operatorname{Bernoulli}(\pi_i)$ | Robust to extreme outliers; intuitive probability of viable habitat. | Standard trawl surveys with low-density sampling noise. |
| **Continuous Soft Sigmoid** | $\text{HSI}_i = \frac{1}{1 + \exp\left(-\kappa \cdot \frac{y_i - q_p}{\text{IQR}(\mathbf{y})}\right)}$ | Differentiable; preserves continuous variance without sharp step cutoffs. | Continuous gradient modeling across ecotones and transition zones. |
| **Two-Part Hurdle / Delta** | $\mathbb{E}[Y_i] = P(Y_i > 0) \cdot \mathbb{E}[Y_i \mid Y_i > 0]$ | Separates true ecological presence/absence from positive density. | Highly zero-inflated datasets with distinct encounter mechanisms. |
| **Environmental Niche Distance** | $\text{HSI}(\mathbf{s}) = \exp\left(-\frac{1}{2} (\mathbf{x}(\mathbf{s}) - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}(\mathbf{s}) - \boldsymbol{\mu})\right)$ | Independent of sampling catchability; pure physiological envelope. | Multi-species bioclimatic envelope & thermal tolerance mapping. |

---

### 2.6. Mechanistic Advection-Diffusion-Reaction (ADR) & Telemetry

On the canonical master mesh $G_{\text{master}}$, the population density state evolves via:

$$
\mathbf{C}_t = \mathbf{C}_{t-1} + \Delta t \left( -v \mathbf{A} \mathbf{C}_{t-1} - \operatorname{diag}(\mathbf{D}) \mathbf{L} \mathbf{C}_{t-1} + r \mathbf{C}_{t-1} \odot \left(1 - \frac{\mathbf{C}_{t-1}}{K}\right) \right) + \sigma \boldsymbol{\epsilon}_t
$$

where:
- $\mathbf{L} = \mathbf{D}_{\text{deg}} - \mathbf{W}$ is the discrete graph Laplacian (diffusion).
- $\mathbf{A}$ is the directed advection operator driven by the derived Habitat Suitability gradient:

  $$
  \mathbf{A} = \mathbf{D}_{\text{in}} - \mathbf{W}_{\text{dir}}^\top
  $$

  where $\mathbf{W}_{\text{dir}, ij}$ depends on the chosen functional relationship:
  - **`:exponential` (Default)**:

    $$
    W_{\text{dir}, ij} = \begin{cases} W_{ij} \cdot \exp(\gamma \cdot (\text{HSI}_j - \text{HSI}_i)) & \text{if } \text{HSI}_j > \text{HSI}_i \\ 0 & \text{otherwise} \end{cases}
    $$

  - **`:linear`**:

    $$
    W_{\text{dir}, ij} = \begin{cases} W_{ij} \cdot (1 + \gamma \cdot (\text{HSI}_j - \text{HSI}_i)) & \text{if } \text{HSI}_j > \text{HSI}_i \\ 0 & \text{otherwise} \end{cases}
    $$

  - **`:logistic`**:

    $$
    W_{\text{dir}, ij} = W_{ij} \cdot \frac{1}{1 + \exp(-\gamma \cdot (\text{HSI}_j - \text{HSI}_i))}
    $$

- For mark-recapture telemetry encounters ($u_{\text{rel}} \to u_{\text{rec}}$ across $k$ time steps), the transition probability kernel evaluates via matrix exponentiation:

  $$
  \pi_m(u_{\text{rec}} \mid u_{\text{rel}}, k, z_m) = \frac{\left[\mathbf{\Gamma}^k\right]_{u_{\text{rel}}, u_{\text{rec}}} \cdot \exp(\beta_{\text{het}} z_m)}{\sum_{j=1}^S \left[\mathbf{\Gamma}^k\right]_{u_{\text{rel}}, j} \cdot \exp(\beta_{\text{het}} z_m)}
  $$

---

## 3. Workflow Architecture & Step-by-Step Implementation

The complete, standalone executable implementation of this workflow is maintained in the companion script:
👉 [**`hierarchical_workflow.jl`**](hierarchical_workflow.jl)

To execute the entire multi-tier pipeline and generate all DuckDB database tables, summary reports, and GIS maps from the command line:

```bash
julia --project=. docs/hierarchical_workflow/hierarchical_workflow.jl
```

### Step-by-Step Modular Pipeline Breakdown

#### Step 0: Multi-Tier Ecosystem & Telemetry Data Generation
Generates synthetic multi-scale datasets via `bstm_data(type="hierarchical")` and `bstm_data(type="telemetry")`:
- **Bathymetry**: $N = 1,000$ continuous soundings.
- **Substrate**: $N = 500$ sediment grab stations.
- **Temperature**: $N = 1,000$ CTD casts across 10 years.
- **Community**: $N = 15,000$ records across 30 species and 500 tows.
- **Target Species**: $N = 400$ snow crab survey tows.
- **Telemetry**: $N = 200$ mark-recapture encounters across 100 individuals.

#### Step 1: Multi-Mesh Topological Graph Construction
Constructs independent, topology-preserving spatial tessellations for each tier clipped against the domain's convex hull:
- $G_2$ (Substrate): $S = 25$ units.
- $G_3$ (Oceanography): $S = 40$ units.
- $G_4$ (Community & HSI): $S = 50$ units.
- $G_{\text{master}}$ (Assessment & ADR): $S = 80$ units.

#### Step 2: Tier 1 Bathymetric Surface & Analytical Differential Geometry
Fits a continuous Random Fourier Features (RFF) GP and evaluates analytical derivatives:

```julia
m_depth = @bstm(likelihood(depth, family=gaussian) ~ intercept() + random(s_x, s_y, model=rff), df_bathy)
chn_depth = sample(m_depth, NUTS(40, 0.65), 80; progress=false)

# Analytical slopes, curvatures, and circular BPI
pred_depth_master = bstm_surface_derivatives(
    m_depth, chn_depth, 
    DataFrame(s_x=[c[1] for c in au_master.centroids], s_y=[c[2] for c in au_master.centroids]);
    metrics = [:slope, :curvature, :bpi], radii = [10.0, 25.0]
)
```

#### Step 3: Tier 2 Sediment Substrate Modeling with EIV
Propagates depth and slope uncertainties to network $G_2$ ($S = 25$) via latent Gaussian EIV priors:

```julia
m_sub = @bstm(
    likelihood(grain_size_phi, family=gaussian) ~ 
        intercept() + 
        fixed(depth_mu, error_sd=:depth_sd) + 
        fixed(slope_mu, error_sd=:slope_sd) + 
        random(s_idx, model=bym2),
    df_sub, W=au_sub.W
)
```

Reshards full posterior sample matrices ($\mathbf{U}_{\text{dest}} = \mathbf{P} \mathbf{U}_{\text{src}}$) to $G_4$ and $G_{\text{master}}$.

#### Step 4: Tier 3 Oceanographic Temperature with Space-Time Kronecker Interactions
Models bottom water temperature on $G_3$ ($S = 40$) using harmonic seasonality and a separable space-time GMRF:

```julia
m_temp = @bstm(
    likelihood(bottom_temperature, family=gaussian) ~ 
        intercept() + 
        fixed(depth_mu, error_sd=:depth_sd) + 
        random(month, model=harmonic, period=12.0) + 
        (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
    df_temp, W=au_temp.W
)
```

#### Step 5: Tier 4 Community Ordination & Second-Last Tier HSI Determination
On haul network $G_4$ ($S = 50$), classifies habitat suitability via the empirical quantile threshold ($p = 0.05$) and extracts major community axes (PC1, PC2) via Hellinger ordination:

```julia
m_hsi = @bstm(
    likelihood(habitat_suitable, family=bernoulli) ~ 
        intercept() + 
        fixed(depth_mu, error_sd=:depth_sd) + 
        fixed(grain_mu, error_sd=:grain_sd) + 
        fixed(temp_mu, error_sd=:temp_sd) + 
        (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
    haul_meta, W=au_comm.W
)
```

#### Step 6: Tier 5 Snow Crab Biomass on Master Network $G_{\text{master}}$
Integrates all upstream resharded environmental predictors, community gradients, and HSI into the final biomass assessment model:

```julia
m_crab = @bstm(
    likelihood(total_biomass_kg, family=gamma, log_offsets=log_effort) ~ 
        intercept() + 
        fixed(hsi_mu, error_sd=:hsi_sd) + 
        fixed(grain_mu, error_sd=:grain_sd) + 
        fixed(pc1_mu, error_sd=:pc1_sd) + 
        fixed(pc2_mu, error_sd=:pc2_sd) + 
        (temp_mu |> random(s_idx, model=icar)) + 
        (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
    df_crab_pos, W=au_master.W
)
```

#### Step 7: Telemetry Transition Kernel, CRW Dispersal & Visualizations
Computes the Markovian transition probability kernel $\mathbf{\Gamma}(\text{HSI}, W)$, simulates individual movement paths with directional persistence (Correlated Random Walk), and constructs the publication dashboard:

```julia
Gamma_master = compute_suitability_transition_kernel(
    community_pred_df.hsi_mean, au_master.W;
    sensitivity = 1.2, diffusion_weight = 0.3, relationship = :exponential
)

sim_paths = simulate_posterior_trajectories(
    Gamma_master, fill(1, 25), 15, au_master;
    rho_persistence = 1.2
)
```

#### Step 8: Master Harmonization Table & Zero-Copy DuckDB SQL Queries
Saves the integrated multi-tier predictions into DuckDB (`master_hierarchical_project.duckdb`), runs high-speed out-of-core SQL habitat queries, and exports RFC 7946 GeoJSON spatial maps:

```julia
df_critical_habitat = query_duckdb(db_path, """
    SELECT unit_id, s_x, s_y, hsi_mean, biomass_mean, temp_mean, substrate_mean, depth_mean
    FROM master_harmonized_summary
    WHERE hsi_mean >= 0.70
    ORDER BY biomass_mean DESC
""")
```

---

## 4. API Reference: Hierarchical Workflows & Movement

| Function | Module | Description |
| :--- | :--- | :--- |
| `bstm_data` | `src/data.jl` | Generates benchmark and synthetic multi-tier hierarchical marine ecosystem datasets. |
| `generate_mock_hierarchical_datasets` | `src/data.jl` | Generates 5-tier mock marine ecological data bundle (bathymetry, substrate, temp, species, crab). |
| `bstm_pipeline` | `src/pipeline.jl` | Declarative multi-tier DAG pipeline orchestrator. |
| `bstm_surface_derivatives` | `src/derivatives.jl` | Computes continuous surface derivatives ($\nabla z, \nabla^2 z$, slope, curvature, BPI). |
| `compute_network_transfer_matrix` | `src/pipeline.jl` | Constructs linear transfer matrix $\mathbf{P}$ between mismatched meshes. |
| `reshard_spatial_field` | `src/pipeline.jl` | Reshards posterior moments or sample matrices ($U_{\text{dest}} = P U_{\text{src}}$). |
| `summarize_sample_matrix` | `src/pipeline.jl` | Evaluates empirical quantiles and credible intervals from sample matrices. |
| `movement(...)` | `src/components/movement.jl` | Declarative ADR movement and mark-recapture telemetry formula module. |
| `compute_suitability_transition_kernel` | `src/movement.jl` | Computes Markov transition kernel $\mathbf{\Gamma}$ from HSI and graph $W$. |
| `simulate_posterior_trajectories` | `src/movement.jl` | Simulates individual paths with directional persistence (Correlated Random Walk). |
| `calculate_regional_connectivity` | `src/movement.jl` | Aggregates fine-scale transitions into macro-regional migration matrices. |
| `save_bstm_bundle` | `src/input_output.jl` | Serializes models (.jld2) and normalized results tables (.duckdb). |
| `query_duckdb` | `src/input_output.jl` | Executes high-performance out-of-core SQL queries directly against DuckDB tables. |
| `export_spatial_results_to_geojson`| `src/input_output.jl` | Exports spatial polygon geometries and predictions to standard RFC 7946 GeoJSON. |

---

## 5. Methodological Criticisms, Limitations & Directions for Improvement

While the modular hierarchical spatiotemporal framework in `bstm` addresses the fragility
and computational intractability of monolithic joint models, several methodological trade-offs
and theoretical limitations warrant critical examination:

### 5.1. Cut-Posterior Modular Inference vs. Full Joint Bayesian Coherence

1. **Severing the Feedback Loop & The Likelihood Principle**:
   - In modular Bayesian inference, the cut-posterior $p_{\text{cut}}(\boldsymbol{\theta}_1, \boldsymbol{\theta}_2 \mid \mathbf{Y}_1, \mathbf{Y}_2) = p(\boldsymbol{\theta}_1 \mid \mathbf{Y}_1) p(\boldsymbol{\theta}_2 \mid \mathbf{Y}_2, \boldsymbol{\theta}_1)$ intentionally blocks backward information flow from $\mathbf{Y}_2$ to $\boldsymbol{\theta}_1$ [@Plummer_2015; @Jacob_2017].
   - While this prevents downstream biological misspecifications (e.g., gear saturation, undocumented catchability shifts) from corrupting upstream physical models (e.g., bathymetry or ocean temperature), it formally violates Bayesian coherency and the strict Likelihood Principle.
   - Information in downstream observations is discarded: for instance, high-density catches of deep-water species along an unmapped submarine trench cannot refine the upstream bathymetric depth surface.
2. **Directions for Improvement — Tempered Power Posteriors**:
   - Introduce a fractional feedback parameter $\lambda \in [0, 1]$ controlling the degree of information feedback:

     $$
     p_\lambda(\boldsymbol{\theta}_1, \boldsymbol{\theta}_2 \mid \mathbf{Y}_1, \mathbf{Y}_2) \propto p(\boldsymbol{\theta}_1 \mid \mathbf{Y}_1) \, p(\boldsymbol{\theta}_2 \mid \boldsymbol{\theta}_1) \, [p(\mathbf{Y}_2 \mid \boldsymbol{\theta}_2, \boldsymbol{\theta}_1)]^\lambda
     $$

     where $\lambda = 0$ yields the pure cut-posterior, $\lambda = 1$ recovers the full joint posterior, and $0 < \lambda < 1$ permits controlled, regularized feedback without unbounded parameter contamination.

---

### 5.2. Errors-in-Variables (EIV) Off-Diagonal Spatial Covariance

1. **Marginal vs. Joint Spatial Measurement Uncertainty**:
   - The current EIV implementation models upstream uncertainty via diagonal Gaussian measurement error priors:

     $$
     x_i \sim \mathcal{N}(\hat{\mu}_{x, i}, \hat{\sigma}^2_{x, i})
     $$

   - This accounts for marginal uncertainty at each unit or sounding location independently, but neglects the **off-diagonal spatial covariance** $\boldsymbol{\Sigma}_{\mathbf{x}}$ across adjacent spatial units.
   - When upstream spatial fields exhibit long-range spatial autocorrelation in their posterior uncertainty, treating errors as conditionally independent can slightly underestimate the joint uncertainty of spatial gradients ($\nabla x$) or contrasts ($\Delta x_{ij} = x_j - x_i$).
2. **Directions for Improvement — Low-Rank GMRF EIV & Direct Draw Indexing**:
   - Propagate full spatial Gaussian Markov Random Field precision matrices $\mathbf{Q}_x$ or low-rank Cholesky factors $\mathbf{L}_x \mathbf{z}$ rather than diagonal variances $\boldsymbol{\sigma}^2_x$.
   - Implement direct Monte Carlo sample indexing during MCMC sampling, where upstream posterior draws $\mathbf{u}_{\text{src}}^{(s)}$ are drawn sequentially without parametric Gaussian approximations.

---

### 5.3. Piecewise-Constant Cross-Mesh Transfer Operators

1. **Areal Averaging across Sharp Environmental Discontinuities**:
   - The linear geometric transfer operator $\mathbf{P} \in \mathbb{R}^{N_{\text{dest}} \times N_{\text{src}}}$ with weights $P_{ji} = \frac{\operatorname{Area}(B_j \cap A_i)}{\operatorname{Area}(B_j)}$ assumes that the latent field is piecewise-constant across each source polygon $A_i$.
   - When source tessellations are coarse relative to local physical gradients (e.g., steep shelf edges, canyons, or frontal boundaries), transferring areal averages to a fine destination mesh inevitably smooths out sub-polygon topographic variation.
2. **Directions for Improvement — Hybrid Basis-Polygon Resharding & Optimal Transport**:
   - **Hybrid Basis Splitting**: Combine polygon transfer operators with continuous spatial basis expansions (e.g. Random Fourier Features or Thin Plate Splines) so that sub-polygon spatial derivatives are preserved during resharding.
   - **Wasserstein Barycentric Resharding**: Use regularized optimal transport to transfer spatial probability measures across mismatched graphs, preserving sharp spatial mass boundaries.

---

### 5.4. Habitat Suitability Index (HSI) Discretization & Advective Dynamics

1. **Threshold Discretization vs. Continuous Ecological Gradients**:
   - The default quantile threshold ($p = 0.05$) Bernoulli classification imposes an arbitrary binary boundary separating "poor" from "suitable" habitat, which discards continuous density variation.
   - The directed advection operator $\mathbf{A} \propto \nabla \text{HSI}$ assumes animals follow the instantaneous local spatial gradient of habitat suitability (chemotaxis/taxis). In nature, animal movement also involves spatial memory, seasonal ontogenetic depth migrations, home range site fidelity, and passive hydrodynamic advection (ocean currents).
2. **Directions for Improvement — Hydrodynamic Coupling & Bioenergetic Suitability**:
   - **Eulerian Current Integration**: Augment the advection operator with observed or oceanographic model velocity fields $\mathbf{u}_{\text{ocean}}(\mathbf{s}, t)$:

     $$
     \mathbf{v}_{\text{total}}(\mathbf{s}, t) = \mathbf{u}_{\text{ocean}}(\mathbf{s}, t) + \mathbf{v}_{\text{active}}(\nabla \text{HSI})
     $$
   - **Dynamic Energy Budget (DEB) Suitability**: Link HSI directly to metabolic thermal performance curves and prey encounter rates rather than empirical catch quantiles.

---

### 5.5. Computational Scalability for High-Dimensional Spatiotemporal Graphs

1. **Propagator Inversion & Kronecker Memory Bottlenecks**:
   - For large-scale territorial meshes ($S > 5,000$ spatial units, $T > 50$ temporal steps), computing dense LU factorizations of the ADR propagator matrix $\mathbf{M} = \mathbf{I} - \Delta t (v \mathbf{A} + \operatorname{diag}(\mathbf{D}) \mathbf{L})$ inside Turing MCMC sampling becomes computationally demanding.
   - High-dimensional spatiotemporal Kronecker products ($\mathbf{Q}_{st} = \mathbf{Q}_t \otimes \mathbf{Q}_s$) can challenge CPU memory limits.
2. **Directions for Improvement — Krylov Exponential Integrators & GPU Kernels**:
   - **Matrix-Free Krylov Solvers**: Use Arnoldi/Lanczos iterations or Chebyshev polynomial approximations to evaluate matrix-vector products $\exp(-\Delta t \mathbf{M}) \mathbf{C}$ without dense matrix factorizations.
   - **GPU Acceleration**: Dispatch sparse graph Laplacian and advection matrix operations directly to GPU hardware via Julia's `KernelAbstractions.jl` and `CUDA.jl`.

---

## 6. References

::: {#refs}
:::
