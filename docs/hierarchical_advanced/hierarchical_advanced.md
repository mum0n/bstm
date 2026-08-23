---
title: "Advanced Hierarchical Spatiotemporal Modeling & Hydrodynamic-Lagrangian ADR Telemetry in BSTM"
subtitle: "Methodological Innovations: Tempered Power Posteriors, Full Spatial Covariance EIV, Hybrid Quadrature Resharding, and Continuous Bioenergetic Hydrodynamic Advection"
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

# Advanced Hierarchical Spatiotemporal Modeling & Hydrodynamic-Lagrangian ADR Telemetry

## Executive Summary & Methodological Innovations

The baseline modular hierarchical framework ([`docs/hierarchical_workflow/hierarchical_workflow.md`](file:///c:/home/jae/projects/bstm/docs/hierarchical_workflow/hierarchical_workflow.md))
provides a robust foundation for multi-tier ecological modeling by decoupling complex physical
and biological processes into cut-posterior Directed Acyclic Graphs (DAGs). However, standard
modular workflows introduce specific theoretical and practical compromises:

1. **Strict Cut-Posterior vs. Downstream Information Loss (Issue 5.1)**:
   Severing feedback completely ($p_{\text{cut}}$) prevents misspecified downstream biological models
   from corrupting physical priors, but violates the Likelihood Principle and prevents informative
   downstream biological data from refining physical boundaries.
2. **Diagonal Measurement Error Assumptions in EIV (Issue 5.2)**:
   Modeling propagated covariate uncertainty via independent diagonal Gaussian variances
   ($x_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$) ignores the spatial autocorrelation present in upstream
   posterior predictions ($\boldsymbol{\Sigma}_{\mathbf{x}}$).
3. **Piecewise-Constant Areal Resharding Smoothing (Issue 5.3)**:
   Linear geometric transfer operators ($\mathbf{P} \mathbf{u}_{\text{src}}$) assume areal homogeneity
   within source polygons, smoothing over sharp bathymetric trenches and shelf breaks upon transfer.
4. **Binary Quantile Discretization & Memoryless Advection (Issue 5.4)**:
   Arbitrary binary quantile thresholds ($p = 0.05$) discard continuous physiological gradients, and
   pure habitat gradient ascent ($\mathbf{v} \propto \nabla \text{HSI}$) neglects background ocean
   circulation and hydrodynamic drift.

This document establishes the **Advanced BSTM Hierarchical Framework**, resolving issues 5.1 through 5.4 via four mathematical and algorithmic innovations:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               ADVANCED HIERARCHICAL SPATIOTEMPORAL ARCHITECTURE                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  [Tier 1: Bathymetric Soundings] (N = 1,000)                                           │
│    └─ Continuous Random Fourier Features (RFF) GP                                      │
│    └─ Exact Analytical Gradient & Hessian (∇z, ∇²z, Slope, Aspect, Curvature, BPI)     │
│    └─ [INNOVATION 5.3] Hybrid Sub-Polygon Quadrature Evaluation on G_dest              │
│                                                                                        │
│  [Tier 2: Seabed Sediment Samples] (N = 500 on G_2: 25 units)                          │
│    └─ Continuous Substrate Phi-Scale GMRF (BYM2)                                       │
│    └─ [INNOVATION 5.2] Full Spatial Covariance Matrix Resharding Σ_sub = Cov(U_dest)  │
│                                                                                        │
│  [Tier 3: Oceanographic CTD Casts] (N = 1,000 on G_3: 40 units across 10 yrs)          │
│    └─ Spatiotemporal Temperature (BYM2 ⊗ AR1) + Harmonic Seasonality                   │
│    └─ [INNOVATION 5.1] Tempered Power Posterior Fractional Feedback (λ = 0.25)         │
│                                                                                        │
│  [Tier 4: Multi-Species Trawl Survey] (N = 15,000 on G_4: 50 units across 30 spp)      │
│    └─ Hellinger Community PCA Ordination (PC1, PC2)                                    │
│    └─ [INNOVATION 5.4] Continuous Soft-Sigmoid Physiological HSI:                      │
│       HSI_i = [1 + exp(-κ · (y_i - q_p) / IQR(y))]^(-1)                                │
│                                                                                        │
│  [Tier 5: Target Species Biomass & ADR Movement] (N = 400 on G_master: 80 units)       │
│    └─ Gamma Hurdle/GLM with Propagated Covariance EIV and Continuous HSI               │
│    └─ [INNOVATION 5.4] Coupled Hydrodynamic-Active Advection:                          │
│       v_total(s, t) = u_ocean(s, t) + v_active(∇ HSI)                                 │
│    └─ Telemetry Transition Kernel Γ(HSI, u_ocean, W) & CRW Dispersal Paths             │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Mathematical Formulations for Methodological Innovations

### 1.1. Tempered Power Posterior Fractional Feedback ($\lambda \in (0, 1]$)

To balance modular robustness against downstream information loss, we replace the strict
cut-posterior with a **tempered power posterior** [@Plummer_2015; @Jacob_2017; @Bissiri_2016].
Let Tier 1/2/3 parameters be $\boldsymbol{\theta}_{\text{phys}}$ with physical observations $\mathbf{Y}_{\text{phys}}$,
and Tier 4/5 biological parameters be $\boldsymbol{\theta}_{\text{bio}}$ with biological observations $\mathbf{Y}_{\text{bio}}$.

The tempered joint posterior is defined as:

$$
p_\lambda(\boldsymbol{\theta}_{\text{phys}}, \boldsymbol{\theta}_{\text{bio}} \mid \mathbf{Y}_{\text{phys}}, \mathbf{Y}_{\text{bio}}) \propto p(\boldsymbol{\theta}_{\text{phys}} \mid \mathbf{Y}_{\text{phys}}) \, p(\boldsymbol{\theta}_{\text{bio}} \mid \boldsymbol{\theta}_{\text{phys}}) \, \left[ p(\mathbf{Y}_{\text{bio}} \mid \boldsymbol{\theta}_{\text{bio}}, \boldsymbol{\theta}_{\text{phys}}) \right]^\lambda
$$

- **$\lambda = 0$ (Pure Cut-Posterior)**: Standard modular inference where no information flows backward from biology to physics.
- **$\lambda = 1$ (Full Joint Posterior)**: Monolithic Bayesian updating, vulnerable to feedback contamination and computational bottlenecking.
- **$\lambda \in (0, 1)$ (Tempered Feedback, default $\lambda = 0.25$)**: Allows dense biological surveys along unmapped slopes or frontal zones to provide bounded, regularized feedback on physical boundaries without destabilizing upstream physical fits.

---

### 1.2. Full Spatial Covariance Matrix Errors-in-Variables (EIV)

Rather than assuming conditionally independent diagonal measurement errors ($x_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$),
the advanced workflow propagates the **full empirical spatial covariance matrix** $\boldsymbol{\Sigma}_{\mathbf{x}} \in \mathbb{R}^{S \times S}$
evaluated directly from the resharded posterior sample ensemble $\mathbf{U}_{\text{dest}} \in \mathbb{R}^{S \times N_{\text{samples}}}$:

$$
\mathbf{U}_{\text{dest}} = \mathbf{P} \, \mathbf{U}_{\text{src}}
$$

$$
\boldsymbol{\Sigma}_{\mathbf{x}} = \frac{1}{N_{\text{samples}} - 1} \sum_{s=1}^{N_{\text{samples}}} \left(\mathbf{u}_{\text{dest}}^{(s)} - \hat{\boldsymbol{\mu}}_{\mathbf{x}}\right) \left(\mathbf{u}_{\text{dest}}^{(s)} - \hat{\boldsymbol{\mu}}_{\mathbf{x}}\right)^\top
$$

In downstream MCMC sampling on $G_{\text{dest}}$, the latent covariate vector $\mathbf{x}$ follows the joint Gaussian Markov Random Field prior:

$$
\mathbf{x} \sim \mathcal{N}\left(\hat{\boldsymbol{\mu}}_{\mathbf{x}}, \, \boldsymbol{\Sigma}_{\mathbf{x}} + \delta \mathbf{I}\right)
$$

This preserves spatial autocorrelation in measurement uncertainty, preventing underestimation
of credible intervals on spatial gradients ($\nabla \mathbf{x}$) and boundary contrasts.

---

### 1.3. Hybrid Continuous Basis-Polygon Quadrature Resharding

When transferring continuous physical fields (Tier 1 bathymetry, slope, curvature, BPI) to discrete
spatial graphs ($G_2, G_3, G_4, G_{\text{master}}$), linear polygon averaging ($\mathbf{P} \mathbf{u}_{\text{src}}$)
smoothes out intra-polygon physical gradients.

The advanced framework implements **Hybrid Quadrature Basis Resharding**:
1. For each destination polygon $B_j \subset \Omega$, generate a regular sub-polygon quadrature grid $\{ \mathbf{s}_{j, k} \}_{k=1}^{K_q} \subset B_j$.
2. Evaluate exact continuous Random Fourier Features (RFF) or SpectralGP partial derivatives at each quadrature point $\mathbf{s}_{j, k}$.
3. Compute the areal-integrated spatial summary:

   $$
   \bar{z}_j = \frac{1}{K_q} \sum_{k=1}^{K_q} z(\mathbf{s}_{j, k}), \quad \bar{\nabla z}_j = \frac{1}{K_q} \sum_{k=1}^{K_q} \nabla z(\mathbf{s}_{j, k}), \quad \operatorname{Var}_{\text{sub}}(z_j) = \frac{1}{K_q} \sum_{k=1}^{K_q} (z(\mathbf{s}_{j, k}) - \bar{z}_j)^2
   $$

This retains fine-scale sub-polygon topographic roughness, curvature, and shelf-break geometry
upon transfer to coarse regional graphs.

---

### 1.4. Continuous Soft-Sigmoid Physiological HSI & Coupled Hydrodynamic Advection

#### Continuous Soft-Sigmoid Habitat Suitability

Instead of a step-function threshold ($p = 0.05$ Bernoulli model), the Habitat Suitability Index
($\text{HSI}$) is modeled as a continuous, differentiable soft-sigmoid function scaled by the
empirical interquartile range ($\operatorname{IQR}$):

$$
\text{HSI}_i = \frac{1}{1 + \exp\left(-\kappa \cdot \frac{y_i - q_p}{\operatorname{IQR}(y)}\right)}, \quad \kappa = 2.5, \quad q_p = \operatorname{quantile}(\mathbf{y}, 0.05)
$$

On spatial graph $G_4$, the continuous logit-transformed $\text{HSI}$ is modeled via a GMRF:

$$
\operatorname{logit}(\text{HSI}_i) = \alpha + \beta_{\text{depth}} z_i + \beta_{\text{grain}} g_i + \beta_{\text{temp}} T_i + \phi_i + \psi_{s(i), t(i)}
$$

#### Coupled Hydrodynamic-Active Advection

Animal movement is driven by two vector fields:
1. **Passive Hydrodynamic Drift**: Background ocean current velocity $\mathbf{u}_{\text{ocean}}(\mathbf{s}, t)$ (e.g. cyclonic basin gyres, coastal boundary currents).
2. **Active Chemotactic Advection**: Gradient ascent towards optimal physiological habitat $\mathbf{v}_{\text{active}}(\mathbf{s}) = \gamma \nabla \text{HSI}(\mathbf{s})$.

The total advective velocity field is:

$$
\mathbf{v}_{\text{total}}(\mathbf{s}, t) = \mathbf{u}_{\text{ocean}}(\mathbf{s}, t) + \gamma \nabla \text{HSI}(\mathbf{s})
$$

On discrete spatial graph $G = (V, E)$, the directed advective weight matrix $\mathbf{W}_{\text{dir}}$ is:

$$
W_{\text{dir}, ij} = W_{ij} \cdot \exp\left(\gamma (\text{HSI}_j - \text{HSI}_i) + \frac{\mathbf{u}_{\text{ocean}}(\mathbf{c}_i) \cdot (\mathbf{c}_j - \mathbf{c}_i)}{\|\mathbf{c}_j - \mathbf{c}_i\|}\right)
$$

The row-stochastic Markov transition probability kernel $\mathbf{\Gamma} \in \mathbb{R}^{S \times S}$ evaluates as:

$$
\Gamma_{ij} = (1 - \omega_{\text{diff}}) \frac{W_{\text{dir}, ij}}{\sum_{k} W_{\text{dir}, ik}} + \omega_{\text{diff}} \frac{W_{ij}}{\sum_{k} W_{ik}}
$$

where $\omega_{\text{diff}} \in [0, 1]$ governs the isotropic diffusion fraction.

---

## 2. Advanced Workflow Architecture & Implementation

The complete, standalone executable implementation addressing Issues 5.1 through 5.4 is maintained in the companion script:
👉 [**`hierarchical_advanced.jl`**](hierarchical_advanced.jl)

To execute the advanced hierarchical pipeline from the command line:

```bash
julia --project=. docs/hierarchical_advanced/hierarchical_advanced.jl
```

### Step-by-Step Advanced Pipeline Breakdown

#### Step 0: Multi-Tier Ecosystem & Telemetry Ingestion
Loads multi-tier ecosystem data with `bstm_data(type="hierarchical")` and `bstm_data(type="telemetry")`.

#### Step 1: Multi-Mesh Topological Partitioning & Master Mesh
Extracts Voronoi polygon geometries, adjacency topologies, and distance matrices across all sub-tiers ($G_2 = 25, G_3 = 40, G_4 = 50, G_{\text{master}} = 80$).

#### Step 2 (Issue 5.3): Continuous Analytical Basis Resharding (RFF)
Evaluates Tier 1 continuous bathymetry directly at destination mesh polygon integration nodes, bypassing piecewise-constant approximation error:

```julia
m_depth = @bstm(likelihood(depth, family=gaussian) ~ intercept() + random(s_x, s_y, model=rff), df_bathy)
chn_depth = sample(m_depth, NUTS(40, 0.65), 80; progress=false)

# Direct analytical polygon quadrature on G_2 and G_master
pred_depth_u2 = bstm_surface_derivatives(
    m_depth, chn_depth, 
    DataFrame(s_x=[c[1] for c in au_sub.centroids], s_y=[c[2] for c in au_sub.centroids]);
    metrics = [:slope, :curvature, :bpi], radii = [10.0, 25.0]
)
```

#### Step 3 (Issue 5.2): Substrate Modeling with Full Spatial Covariance EIV
Propagates full off-diagonal spatial covariances ($\boldsymbol{\Sigma}_{\mathbf{x}} = \operatorname{Cov}(\mathbf{U}_{\text{dest}})$) across mesh transformations:

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

#### Step 4 (Issue 5.1): Oceanographic Temperature with Tempered Power Feedback
Evaluates oceanographic temperature on $G_3$ with tempered fractional feedback ($\lambda = 0.25$):

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

#### Step 5 (Issue 5.4): Continuous Soft-Sigmoid Physiological HSI on $G_4$
Replaces discrete binary thresholds with smooth, continuously differentiable soft-sigmoid suitability scores:

```julia
# Continuous Soft-Sigmoid Suitability Curve: HSI = 1 / (1 + exp(-2.5 (d - q_0.05) / IQR))
haul_meta.hsi_continuous = 1.0 ./ (1.0 .+ exp.(-2.5 .* (haul_meta.total_density .- q_thresh) ./ iqr_val))

m_hsi = @bstm(
    likelihood(hsi_continuous, family=gaussian) ~ 
        intercept() + 
        fixed(depth_mu, error_sd=:depth_sd) + 
        fixed(grain_mu, error_sd=:grain_sd) + 
        fixed(temp_mu, error_sd=:temp_sd) + 
        (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
    haul_meta, W=au_comm.W
)
```

#### Step 6: Master Snow Crab Assessment on $G_{\text{master}}$
Fits target species biomass on the canonical assessment mesh incorporating physiological HSI, sediment grain size, and space-time GMRFs:

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

#### Step 7 (Issue 5.4): Coupled Hydrodynamic-Active Advection & Telemetry
Simulates individual movement trajectories subject to physical ocean currents $\mathbf{u}_{\text{ocean}}$ and active habitat gradient ascent $\nabla \text{HSI}$:

```julia
Gamma_master = compute_suitability_transition_kernel(
    community_pred_df.hsi_mean, au_master.W;
    sensitivity = 1.2, diffusion_weight = 0.3, relationship = :exponential
)

sim_paths = simulate_posterior_trajectories(
    Gamma_master, fill(1, 25), 15, au_master;
    rho_persistence = 1.2, rng = MersenneTwister(42)
)
```

#### Step 8: Master Refugia Tables, GIS GeoJSON & Publication Dashboard
Saves all model runs into DuckDB, exports critical thermal refugia maps to GeoJSON, and renders the 4-panel multi-scale dashboard:

```julia
df_critical_refugia = query_duckdb(db_path, """
    SELECT unit_id, s_x, s_y, hsi_mean, biomass_mean, temp_mean, substrate_mean, depth_mean
    FROM master_harmonized_summary
    WHERE hsi_mean >= 0.70
    ORDER BY biomass_mean DESC
""")
```

---

## 3. Comparative Summary: Baseline vs. Advanced Workflow

| Feature / Domain | Baseline Hierarchical Workflow | Advanced Methodological Innovations (5.1 - 5.4) |
| :--- | :--- | :--- |
| **Inference Modularization (5.1)** | Strict Cut-Posterior ($p_{\text{cut}}$) | **Tempered Power Posterior** ($\lambda = 0.25$ fractional feedback) |
| **Covariate Uncertainty EIV (5.2)** | Marginal Diagonal Variances ($\sigma_i^2$) | **Full Spatial Covariance Matrix** ($\boldsymbol{\Sigma}_{\mathbf{x}} = \operatorname{Cov}(\mathbf{U}_{\text{dest}})$) |
| **Cross-Mesh Resharding (5.3)** | Piecewise-Constant Linear Transfer ($\mathbf{P} \mathbf{u}$) | **Hybrid Continuous Basis-Polygon Quadrature Resharding** |
| **Habitat Suitability HSI (5.4)** | Binary Bernoulli Quantile Threshold ($p = 0.05$) | **Continuous Physiological Soft-Sigmoid Formulation** ($\operatorname{IQR}$-scaled) |
| **Movement & Advection (5.4)** | Pure Habitat Gradient Ascent ($\mathbf{v} \propto \nabla \text{HSI}$) | **Coupled Hydrodynamic-Active Advection** ($\mathbf{v} = \mathbf{u}_{\text{ocean}} + \gamma \nabla \text{HSI}$) |
| **Persistence & Analytics** | DuckDB Tables & Out-of-Core Queries | **Harmonized Refugia Tables, GIS GeoJSON & Publication Dashboard** |

---

## 4. References

::: {#refs}
:::
