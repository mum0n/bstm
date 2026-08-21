---
title: "Mechanistic Spatiotemporal Modeling: Advection-Diffusion-Reaction (ADR) & Telemetry"
format: html
---

# Mechanistic Spatiotemporal Modeling: Advection-Diffusion-Reaction (ADR) & Telemetry

## 1. Introduction to Mechanistic Movement Modeling

In movement ecology and spatial epidemiology, predicting how populations redistribute across landscapes requires understanding the **underlying biophysical mechanisms** that drive movement. While purely statistical spatiotemporal models (e.g., separable CAR or Gaussian processes) describe empirical correlations, **mechanistic models** explicitly parameterize the physical and behavioral processes of directional migration, random dispersal, and demographic reproduction.

The **Advection-Diffusion-Reaction (ADR)** framework serves as the mathematical foundation for mechanistic population modeling (Okubo, 1980; Turchin, 1998; Hooten & Hefley, 2019). It decomposes spatiotemporal population flux into three distinct components:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        The Advection-Diffusion-Reaction Continuum                      │
├────────────────────────────┬────────────────────────────┬──────────────────────────────┤
│       1. Advection         │        2. Diffusion        │         3. Reaction          │
├────────────────────────────┼────────────────────────────┼──────────────────────────────┤
│ - Directed, active motion  │ - Stochastic dispersal     │ - Local demographic change   │
│ - Follows habitat gradients│ - Random exploratory walks │ - Births, deaths, predation  │
│ - Driven by velocity v     │ - Scaled by coefficient D  │ - Logistic growth r*(1 - N/K)│
└────────────────────────────┴────────────────────────────┴──────────────────────────────┘
```

Furthermore, the `bstm` movement subsystem implements an **Integrated Population Model (IPM)** that fuses two complementary data streams into a single joint Bayesian likelihood:
1. **Eulerian Data (Broad-Scale Population Density)**: Grid- or polygon-level biomass/count surveys ($y_{s, t}$).
2. **Lagrangian Data (Fine-Scale Mark-Recapture Telemetry)**: Individual tracking encounters ($u_{\text{rel}} \to u_{\text{rec}}$) informing movement rates and individual-level behavioral heterogeneity.

---

## 2. Mathematical Foundations

### 2.1. Continuous Partial Differential Equation (PDE)

In continuous space $\mathbf{s} \in \Omega \subset \mathbb{R}^2$ and continuous time $t \ge 0$, the evolution of population density $C(\mathbf{s}, t)$ is governed by the parabolic ADR partial differential equation:

$$\frac{\partial C(\mathbf{s}, t)}{\partial t} = \underbrace{\nabla \cdot \left(D(\mathbf{s}) \nabla C(\mathbf{s}, t)\right)}_{\text{Diffusion}} - \underbrace{\nabla \cdot \left(\mathbf{v}(\mathbf{s}, t) C(\mathbf{s}, t)\right)}_{\text{Advection}} + \underbrace{f(C(\mathbf{s}, t))}_{\text{Reaction}}$$

where:
- $\nabla = \left[\frac{\partial}{\partial x}, \frac{\partial}{\partial y}\right]^T$ is the spatial gradient operator, and $\nabla \cdot$ is the divergence operator.
- $D(\mathbf{s}) > 0$ is the **diffusion coefficient** (dispersal rate), which can vary spatially as a function of habitat permeability: $D(s) = D_0 \exp(\beta_{\text{hab}} \cdot \text{habitat}_s)$.
- $\mathbf{v}(\mathbf{s}, t) \in \mathbb{R}^2$ is the **advective velocity field**, defining the speed and direction of active movement.
- $f(C)$ is the **reaction growth function**, typically parameterized via logistic population dynamics with intrinsic rate $r$ and carrying capacity $K$:
  $$f(C) = r C \left(1 - \frac{C}{K}\right)$$

---

### 2.2. Discrete Graph Representation & Spatial Operators

To conduct Bayesian inference on discrete spatial lattices or irregular polygon tessellations ($s \in \{1, \dots, S\}$), the continuous differential operators are discretized onto a spatial adjacency graph $W$:

```
               Region s_j ────[ W_ij Edge ]──── Region s_i
                    │                                │
                    ▼                                ▼
         Density C_j(t) ─── Directional Flow ───> Density C_i(t)
```

#### 1. The Graph Laplacian (Diffusion Operator $\mathbf{L}$)
Diffusion across connected areal units is governed by the graph Laplacian matrix $\mathbf{L}$:
$$\mathbf{L} = \mathbf{D}_{\text{deg}} - \mathbf{W}, \quad \mathbf{D}_{\text{deg}} = \operatorname{diag}\left(\sum_{j=1}^S W_{sj}\right)$$
For any density vector $\mathbf{C}$, the term $-\mathbf{L}\mathbf{C}$ calculates the net diffusive flux:
$$(-\mathbf{L}\mathbf{C})_i = \sum_{j \in \mathcal{N}(i)} W_{ij}(C_j - C_i)$$
This discrete operator acts as the lattice analogue of the continuous Laplacian $\nabla^2 C$.

#### 2. Directed Advection Operator ($\mathbf{A}$)
Directed migration is driven by an organism's affinity for favorable environmental conditions (e.g., optimal water temperature or depth). Given a Habitat Suitability Index $\text{HSI}(s) \in (0, 1)$, the directional flow matrix $\mathbf{W}_{\text{dir}}$ directs mass from low-suitability to high-suitability neighbors:
$$W_{\text{dir}, ij} = \begin{cases} W_{ij} & \text{if } \text{HSI}_j > \text{HSI}_i \\ 0 & \text{otherwise} \end{cases}$$
Normalizing by out-degree yields the directed advection transition operator:
$$\mathbf{A} = \operatorname{diag}\left(\sum_k W_{\text{dir}, ik} + \epsilon\right)^{-1} \mathbf{W}_{\text{dir}}$$
The scaled advective flux is given by $-v \mathbf{A} \mathbf{C}$, where $v > 0$ is the advection speed.

---

### 2.3. Temporal Discretization & Time-Stepping Schemes

#### Method 1: Explicit Euler Scheme (Default, AD-Friendly)
In discrete time steps $\Delta t = 1$, the state advances forward explicitly:
$$\mathbf{C}_t = \mathbf{C}_{t-1} + \Delta t \left( -v \mathbf{A} \mathbf{C}_{t-1} - \operatorname{diag}(\mathbf{D}) \mathbf{L} \mathbf{C}_{t-1} + r \mathbf{C}_{t-1} \odot \left(1 - \frac{\mathbf{C}_{t-1}}{K}\right) \right) + \sigma \boldsymbol{\epsilon}_t$$
where $\boldsymbol{\epsilon}_t \sim \mathcal{N}(0, \mathbf{I}_S)$ is the stochastic environmental innovation.

> [!IMPORTANT]
> **CFL Stability Condition**: For the explicit Euler scheme to remain numerically stable without exploding, the time step $\Delta t$ and spatial grid spacing $\Delta x$ must satisfy the Courant-Friedrichs-Lewy (CFL) condition:
> $$\Delta t \le \frac{\Delta x^2}{2 D_{\max} + v_{\max} \Delta x}$$

#### Method 2: Implicit Euler & The Propagator Matrix
For long-term stationary dynamics without reaction terms, the implicit formulation constructs the **Propagator Matrix** $\mathbf{M}_{\text{prop}}$:
$$\mathbf{M}_{\text{prop}} = \mathbf{I}_S - v \mathbf{A} - \operatorname{diag}(\mathbf{D}) \mathbf{L}$$
The forward state update is computed via linear solve:
$$\mathbf{C}_t = \mathbf{M}_{\text{prop}}^{-1} \mathbf{C}_{t-1} + \sigma \boldsymbol{\epsilon}_t$$
The matrix inverse $\mathbf{\Gamma} = \mathbf{M}_{\text{prop}}^{-1}$ defines the **one-step Markov transition probability kernel** across the spatial network.

---

### 2.4. Joint Bayesian Likelihood Formulation

The overall model fuses Eulerian population counts and Lagrangian individual telemetry marks:

$$\mathcal{L}_{\text{joint}}(\boldsymbol{\theta}) = \mathcal{L}_{\text{density}}(\mathbf{y} \mid \mathbf{C}, \boldsymbol{\theta}_{\text{obs}}) \times \mathcal{L}_{\text{telemetry}}(\mathbf{tag} \mid \mathbf{\Gamma}, \beta_{\text{het}})$$

```
                               ┌────────────────────────────────────────────────────────┐
                               │   Joint Bayesian Movement Likelihood Architecture      │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                             ┌─────────────────────────────┴────────────────────────────┐
                             ▼                                                          ▼
        ┌────────────────────────────────────────┐                 ┌────────────────────────────────────────┐
        │   Eulerian Population Density Survey   │                 │    Lagrangian Mark-Recapture Telemetry │
        ├────────────────────────────────────────┤                 ├────────────────────────────────────────┤
        │ - Observations: y_{s, t}               │                 │ - Transitions: u_rel -> u_rec (k steps)│
        │ - Likelihood: Poisson / NegBinomial    │                 │ - Likelihood: Categorical(π_m)         │
        │ - Links to latent density C_{s, t}     │                 │ - Multi-step Markov kernel Γ^k         │
        └────────────────────────────────────────┘                 └────────────────────────────────────────┘
```

#### 1. Population Survey Likelihood
Observed survey counts $y_{s, t}$ are conditioned on the latent density $C_{s, t}$:
$$y_{s, t} \sim \operatorname{Poisson}(\lambda_{s, t}), \quad \log \lambda_{s, t} = \log\left(\max(C_{s, t}, 10^{-6})\right) + \alpha + \phi_s$$

#### 2. Mark-Recapture Telemetry Likelihood
For a marked individual $m$ released in spatial unit $u_{\text{rel}}$ and recaptured $k$ time steps later in unit $u_{\text{rec}}$, the multi-step transition probability is governed by the $k$-th matrix power of the transition kernel: $\mathbf{\Gamma}^{(k)} = \mathbf{\Gamma}^k$.

To account for individual-level heterogeneity (e.g. body size, sex, maturity status $z_m$), the transition probabilities are modulated via a multinomial logit link:
$$\pi_{m}(u_{\text{rec}} \mid u_{\text{rel}}, k, z_m) = \frac{\left[\mathbf{\Gamma}^k\right]_{u_{\text{rel}}, u_{\text{rec}}} \cdot \exp(\beta_{\text{het}} z_m)}{\sum_{j=1}^S \left[\mathbf{\Gamma}^k\right]_{u_{\text{rel}}, j} \cdot \exp(\beta_{\text{het}} z_m)}$$
$$\text{tag}_m \sim \operatorname{Categorical}(\boldsymbol{\pi}_m)$$

---

## 3. The `movement()` Component Specification

In BSTM formulas, the ADR process is declared using the `movement()` RHS module:

```julia
movement(s_idx, time_idx; kwargs...)
```

### 3.1. Options & Hyperparameters

| Argument | Type | Default | Description & Mathematical Role |
| :--- | :--- | :--- | :--- |
| `velocity` | `Distribution` | `Normal(1.0, 0.5)` | Prior distribution for the advection speed parameter $v > 0$. |
| `diffusion` | `Distribution` | `LogNormal(-1.0, 1.0)` | Prior distribution for the baseline diffusion rate $D > 0$. |
| `sigma` | `Distribution` | `Exponential(1.0)` | Prior for the standard deviation $\sigma$ of the environmental process noise. |
| `r` | `Distribution` | `nothing` | Optional prior for intrinsic growth rate $r$ (enables reaction term). |
| `K` | `Distribution` | `nothing` | Optional prior for carrying capacity $K$ (enables logistic saturation). |
| `beta_het` | `Distribution` | `nothing` | Prior for individual heterogeneity effect $\beta_{\text{het}}$ in telemetry data. |
| `habitat` | `Symbol` / `Vector` | `nothing` | Spatial covariate column modulating local diffusion: $D(s) = D \exp(\beta_{\text{hab}} \text{hab}_s)$. |
| `mark_recapture_data`| `DataFrame` / `Matrix` | `nothing` | Telemetry dataset containing marked individual transitions. |
| `method` | `Symbol` | `:explicit` | Numerical time-stepping scheme: `:explicit` (AD-friendly) or `:implicit` (linear solve). |

### 3.2. Telemetry DataFrame Schema

When passing telemetry data via `mark_recapture_data`, the `DataFrame` must contain the following columns:

| Column | Type | Description |
| :--- | :--- | :--- |
| `tagid` | `Int` | Unique individual animal identification tag. |
| `s_idx` | `Int` | Spatial unit index where the animal was observed ($1 \le s \le S$). |
| `time` | `Float64` / `Int` | Temporal stamp of encounter (e.g. Year or Day). |
| `tag` | `Int` | Encounter type: `0` for initial release, `1` for subsequent recapture. |
| `individual_covariate`| `Float64` | *Optional*: Continuous individual attribute (e.g. carapace length, body mass). |

---

## 4. Dispersal & Trajectory Simulation Tools

`bstm` provides post-processing and diagnostic simulation utilities in [`src/movement.jl`](file:///c:/home/jae/projects/bstm/src/movement.jl):

### 4.1. Correlated Random Walk (CRW) with Directional Persistence

Real organisms exhibit directional momentum (persistence) rather than memoryless Brownian motion. Given transition kernel $\mathbf{\Gamma}$ and previous movement vector $\mathbf{v}_{\text{prev}} = \mathbf{c}_{t-1} - \mathbf{c}_{t-2}$, candidate transitions are reweighted by turning angles $\cos \theta$:

$$w_j = \exp\left( \rho_{\text{persistence}} \cdot \frac{\mathbf{v}_{\text{prev}} \cdot (\mathbf{c}_j - \mathbf{c}_{t-1})}{\|\mathbf{v}_{\text{prev}}\| \|\mathbf{c}_j - \mathbf{c}_{t-1}\|} \right)$$

Simulated via `simulate_posterior_trajectories`:
```julia
paths = simulate_posterior_trajectories(Gamma, start_units, 20, au; rho_persistence=1.5)
```

### 4.2. Regional Macro-Connectivity Aggregation

To understand macro-scale migration across ecological strata (e.g., North/South or Coastal/Offshore), `calculate_regional_connectivity` aggregates fine-scale transition probabilities into a coarse-grained migration matrix $\mathbf{C}$:

$$C_{R_1 \to R_2} = \frac{\sum_{i \in R_1} \sum_{j \in R_2} \Gamma_{ij}}{|R_1|}$$

```julia
strata = [centroid.x > 500.0 ? "East" : "West" for centroid in au.centroids]
C_regional = calculate_regional_connectivity(Gamma, strata)
```

### 4.3. Advection-to-Diffusion (Péclet-like) Ratio

The local dimensionless ratio $Pe_s = \frac{\|\mathbf{v}_s\|}{D_s}$ diagnoses whether spatial distribution is governed primarily by directed environmental tracking ($Pe > 1$) or random spatial diffusion ($Pe < 1$):

```julia
plt_ad = plot_ad_ratio_distribution(advection_magnitudes, diffusion_coefficients)
```

---

## 5. End-to-End Runnable Tutorial: Joint ADR & Telemetry

Below is a complete, self-contained workflow demonstrating synthetic data generation, joint model fitting with `@bstm`, MCMC sampling, DuckDB persistence, and diagnostic visualization.

```julia
using bstm, DataFrames, Distributions, LinearAlgebra, Random, Plots

# ==============================================================================
# Step 1: Simulate Synthetic Spatial Domain, Population Density & Telemetry
# ==============================================================================
rng = MersenneTwister(42)

println("--- Generating Synthetic ADR & Telemetry Simulation Bundle ---")
sim_data = generate_ADR_simulation_bundle(
    1000.0, # 1000 x 1000 km domain
    64,     # 64 spatial units (Centroidal Voronoi / Hexagonal)
    5,      # 5 annual survey years
    80;     # 80 tagged telemetry individuals
    area_method = :cvt,
    rng = rng
)

println("Spatial Units (S): ", sim_data.n_spatial)
println("Survey Records:    ", nrow(sim_data.data))
println("Telemetry Records: ", nrow(sim_data.telemetry_data))

# ==============================================================================
# Step 2: Define Joint ADR Model via @bstm Macro
# ==============================================================================
println("--- Compiling Joint Bayesian Model ---")

model_instance = @bstm(
    likelihood(density, family=poisson) ~
        intercept(prior=Normal(0, 5)) +
        movement(
            unit_id, year,
            velocity = Normal(1.0, 0.4),
            diffusion = LogNormal(-0.5, 0.5),
            sigma = Exponential(1.0),
            beta_het = Normal(0.0, 0.5),
            mark_recapture_data = sim_data.telemetry_data,
            method = :explicit
        ),
    sim_data.data,
    W = sim_data.au.W,
    verbose = false
)

# ==============================================================================
# Step 3: Posterior Inference with NUTS
# ==============================================================================
println("--- Sampling Posterior Distribution ---")
chain = sample(model_instance, NUTS(150, 0.65), 400; progress=false)

# Reconstruct posterior summaries and performance metrics
res = model_results_comprehensive(model_instance, chain; au=sim_data.au)

# ==============================================================================
# Step 4: Persist Model Bundle to JLD2 and DuckDB
# ==============================================================================
save_bstm_bundle("output/adr_snowcrab_model", model_instance, chain, res; au=sim_data.au)

# Query posterior parameter estimates directly via DuckDB SQL
df_params = query_duckdb("output/adr_snowcrab_model.duckdb", """
    SELECT parameters, mean, std, mcse, ess, rhat
    FROM parameter_stats
    WHERE parameters LIKE '%velocity%' OR parameters LIKE '%diffusion%' OR parameters LIKE '%beta_het%'
""")
display(df_params)

# ==============================================================================
# Step 5: Post-Hoc ADR Diagnostic Synthesis & Trajectory Visualization
# ==============================================================================
avg_p = [mean(sim_data.data.habitat_p[sim_data.data.unit_id .== i]) for i in 1:sim_data.n_spatial]
g_side = Int(floor(sqrt(sim_data.n_spatial)))
vel_vectors = compute_velocity_field(avg_p, g_side, 1.5; mode=:exponential)

# Run full ADR synthesis
adr_synthesis = synthesize_adr_results(chain, sim_data, vel_vectors; au=sim_data.au)

# Display diagnostic multi-panel figures
display(adr_synthesis.plots.regional_connectivity)
display(adr_synthesis.plots.paths_comparison)
display(adr_synthesis.plots.dynamic_paths)

println("\n✓ Joint ADR-Telemetry modeling workflow completed successfully!")
```

---

## 6. Assumptions & Modeling Guidelines

1. **Mass Conservation & Boundary Conditions**:
   The graph Laplacian operator enforces zero-flux (Neumann) boundary conditions across external network edges, ensuring that individuals do not artificially leak out of the spatial domain.
2. **Telemetry Representative Sampling**:
   The telemetry transition likelihood assumes that detection and recapture probabilities are conditionally independent of unobserved individual state after adjusting for individual covariates $z_m$.
3. **Priors on Movement Parameters**:
   Because velocity $v$ and diffusion $D$ operate on non-negative physical scales, priors should be strictly positive (e.g. `LogNormal`, `truncated(Normal, 0, Inf)`).
4. **Graph Connectivity**:
   Ensure the adjacency matrix $W$ contains no isolated graph islands (use `ensure_connected!(g, centroids)` if necessary) so that the Laplacian $\mathbf{L}$ is connected and well-conditioned.

---

## 7. References

1. **Besag, J.** (1974). Spatial interaction and the statistical analysis of lattice systems. *Journal of the Royal Statistical Society: Series B*, 36(2), 192–225.
2. **Hooten, M. B., & Hefley, T. J.** (2019). *Bringing Bayesian Models to Life*. CRC Press.
3. **Johnson, D. S., London, J. M., Lea, M. A., & Durban, J. W.** (2008). Continuous-time correlated random walk model for animal telemetry data. *Ecology*, 89(5), 1208–1215.
4. **Okubo, A.** (1980). *Diffusion and Ecological Problems: Mathematical Models*. Springer-Verlag.
5. **Turchin, P.** (1998). *Quantitative Analysis of Movement: Measuring and Modeling Population Redistribution in Animals and Plants*. Sinauer Associates.