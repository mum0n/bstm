## Priority Analyses for Mark-Recapture Path Prediction

This directory contains implementations of three priority post-processing analyses for Bayesian Spatio-Temporal Movement (BSTM) mark-recapture models. These analyses validate model performance and quantify uncertainty in predicted animal movement paths.

### Overview

The BSTM pipeline reconstructs individual movement trajectories between mark-recapture sites (release → recapture locations). The priority analyses address three critical gaps:

1. **Per-Path Credible Intervals** — Propagate posterior MCMC uncertainty through individual trajectory predictions
2. **Stock Connectivity Matrix** — Summarize inter-regional movement flows with posterior credible intervals
3. **Posterior Predictive Check** — Validate that the fitted model reproduces observed recapture distributions

---

## 1. Per-Path Credible Intervals (`path_uncertainty.jl`)

### Purpose
Quantifies uncertainty in individual movement paths by drawing from the posterior MCMC distribution and reconstructing path ensembles. Answers: *"How confident is our prediction of where individual X moved?"*

### Key Outputs

```
path_credible_intervals.csv
├── tagid                 # Individual animal identifier
├── release_site          # Release spatial unit
├── recapture_site        # Recapture spatial unit
├── path_length_mean      # Mean path length (hops)
├── path_length_lower     # 2.5th percentile
├── path_length_upper     # 97.5th percentile
├── path_length_sd        # Standard deviation across draws
├── n_waypoints           # Number of spatial units in domain
└── modal_waypoint        # Most-visited waypoint
```

### Functions

#### `path_credible_intervals(loaded, fitted, kernels, params)`
Main function for per-path uncertainty propagation.

**Algorithm:**
1. Extract MCMC posterior samples of movement parameters (velocity, diffusion, gamma)
2. For each posterior draw:
   - Construct transition kernel from sampled parameters
   - Reconstruct full trajectory from release → recapture
   - Record path length and node visitation frequencies
3. Compute quantiles across all draws

**Returns:**
- `path_samples`: Dictionary of path ensembles (one per individual)
- `path_stats`: Per-individual summaries with credible intervals
- `node_visit_probs`: n_spatial × n_spatial visitation probability matrix

#### `export_path_uncertainty_summary(path_unc, loaded, output_dir)`
Exports credible intervals to CSV format.

### Example Usage

```julia
# Run path uncertainty analysis
path_unc = path_credible_intervals(loaded, fitted, kernels, params)

# Export to CSV
csv_file = export_path_uncertainty_summary(path_unc, loaded, output_dir)

# Interpret results
for (tid, stats) in path_unc.path_stats
    @printf "%s: %d ± %d hops [%.0f-%.0f]\n" tid stats.length_mean 
            stats.length_sd stats.length_lower_ci stats.length_upper_ci
end
```

### Interpretation

- **Wide credible intervals** → High posterior uncertainty (model not confident)
- **Narrow credible intervals** → Strong posterior signal (high confidence)
- **Modal waypoint** → Most probable intermediate location along path
- **Waypoint probabilities** → Spatial distribution of predicted trajectory

---

## 2. Stock Connectivity Matrix (`connectivity_analysis.jl`)

### Purpose
Aggregates individual movement predictions into a population-level connectivity matrix summarizing inter-regional flows. Answers: *"Which regions are strongly connected through animal movement?"*

### Key Outputs

```
stock_connectivity_matrix.csv          # Mean connectivity (rows = sources, cols = sinks)
stock_flow_counts.csv                  # Observed mark-recapture transitions
stock_connectivity_credible_intervals.csv
├── source_region                      # Donor region
├── sink_region                        # Recipient region
├── mean_conn                          # Mean connectivity probability
├── lower_ci                           # 2.5th percentile
└── upper_ci                           # 97.5th percentile
```

### Functions

#### `compute_stock_connectivity_matrix(loaded, kernels, params; region_labels, region_map)`

Computes aggregated connectivity matrix at regional scale.

**Parameters:**
- `region_map::Vector{Int}` — Length n_spatial vector assigning mesh units to regions (1:n_regions)
- `region_labels::Vector{String}` — Human-readable region names

**Algorithm:**
1. Define spatial regions (user-supplied or default: one region per mesh unit)
2. Aggregate transition kernel probabilities:
   ```
   Connectivity[r, s] = (1 / |r| × |s|) × Σ_u∈r Σ_v∈s P_kernel[u, v]
   ```
3. Normalize rows to form stochastic matrix
4. Weight by observed mark-recapture flows

**Returns:**
- `connectivity_matrix`: n_regions × n_regions stochastic transition matrix
- `flow_counts`: Observed transitions per region pair
- `flow_rates`: Weighted flow rates
- `region_labels`, `region_map`: Region definitions

#### `compute_connectivity_credible_intervals(loaded, fitted, kernels, params, region_map)`

Propagates posterior uncertainty through connectivity computation.

**Algorithm:**
1. For each MCMC posterior draw:
   - Sample movement parameters
   - Construct transition kernel
   - Aggregate to regions
   - Normalize rows
2. Compute empirical quantiles across all draws

**Returns:**
- `connectivity_mean`: Point estimate
- `connectivity_lower`, `connectivity_upper`: 95% credible intervals
- `connectivity_samples`: Full posterior draws (for further analysis)

### Example Usage

```julia
# Define regions (e.g., spawning, nursery, adult habitat)
region_labels = ["Spawning Ground", "Nursery", "Adult Habitat"]
region_map = [1, 1, 2, 2, 3, 3, ...]  # Assign each mesh unit

# Compute connectivity
conn = compute_stock_connectivity_matrix(loaded, kernels, params; 
                                        region_labels, region_map)

# Export
export_connectivity_matrix(conn, output_dir)

# Compute uncertainty
conn_unc = compute_connectivity_credible_intervals(
    loaded, fitted, kernels, params, region_map
)
export_connectivity_uncertainty(conn_unc, output_dir, region_labels)

# Interpret top flows
top_flows = sort(
    [(r, s, conn.connectivity_matrix[r, s]) 
     for r in 1:conn.n_regions for s in 1:conn.n_regions],
    by = x -> x[3], rev = true
)[1:5]
```

### Interpretation

- **High connectivity** → Strong population exchange (likely demographically important)
- **Low connectivity** → Isolated populations (genetic drift risk)
- **Asymmetric connectivity** → Directional bias in movement
- **Wide uncertainty intervals** → Data-limited or model-uncertain pathways

---

## 3. Posterior Predictive Check (`posterior_predictive_check.jl`)

### Purpose
Validates model fit by comparing observed vs. predicted recapture location distributions. Answers: *"Does the fitted model produce recaptures that match the observed data?"*

### Key Outputs

```
posterior_predictive_diagnostics.csv
├── draw                   # MCMC draw index
├── brier_score            # Mean squared error (0 = perfect, 1 = worst)
└── kl_divergence          # KL(observed || predicted), nats

posterior_predictive_summary.txt
├── Brier Score            # Mean, SD, 95% CI
├── KL Divergence          # Mean, SD, 95% CI
├── Observed Entropy       # Shannon entropy of recapture distribution
└── Interpretation         # Qualitative fit assessment

posterior_predictive_distributions.csv
├── spatial_unit           # Mesh unit index
├── observed_prob          # Empirical recapture frequency
└── mean_predicted_prob    # Model predictive mean

posterior_predictive_rank_histogram.csv
├── rank_bin               # Rank within posterior predictive distribution
└── count                  # Frequency of observed recaptures at this rank
```

### Functions

#### `posterior_predictive_check(loaded, fitted, kernels, params)`

Performs full posterior predictive validation.

**Algorithm:**
1. For each MCMC posterior draw:
   - Construct transition kernel from sampled parameters
   - Simulate recapture locations for all observed release sites
   - Compute distribution of simulated recaptures
2. Compare observed vs. simulated distributions:
   - **Brier score:** MSE between observed and simulated probabilities
     ```
     Brier = (1/n_spatial) Σ [P_obs(s) - P_sim(s)]²
     ```
   - **KL divergence:** Information-theoretic distance
     ```
     KL(P_obs || P_sim) = Σ P_obs(s) log(P_obs(s) / P_sim(s))
     ```
3. Compute rank histogram (for uniform coverage diagnostics)

**Returns:**
- `brier_scores`, `kl_divergences`: Per-draw diagnostic vectors
- `observed_dist`: Empirical recapture distribution
- `simulated_recapture_dists`: Full posterior predictive samples
- `summary`: NamedTuple with means, credible intervals, entropy

#### `export_posterior_predictive_check(ppc, output_dir)`

Exports diagnostic summaries and plots to files.

#### `plot_posterior_predictive_check(ppc, output_dir)` *(optional, requires Plots.jl)*

Generates diagnostic visualizations:
- Brier score trace plot
- KL divergence trace plot
- Observed vs. predicted recapture CDFs
- Rank histogram (uniform = good calibration)

### Example Usage

```julia
# Run posterior predictive check
ppc = posterior_predictive_check(loaded, fitted, kernels, params)

# Export summaries
export_posterior_predictive_check(ppc, output_dir)

# Diagnose fit
summary = ppc.summary
@printf "Brier: %.6f ± %.6f (95%% CI: [%.6f, %.6f])\n" 
    summary.brier_mean summary.brier_sd 
    summary.brier_lower_ci summary.brier_upper_ci

@printf "KL:    %.6f ± %.6f (95%% CI: [%.6f, %.6f])\n"
    summary.kl_mean summary.kl_sd
    summary.kl_lower_ci summary.kl_upper_ci

# Visualize (if Plots.jl available)
plot_posterior_predictive_check(ppc, output_dir)
```

### Interpretation

| Metric | Good | Moderate | Poor |
|--------|------|----------|------|
| **Brier Score** | < 0.01 | 0.01–0.05 | > 0.05 |
| **KL Divergence** | < 0.5 nats | 0.5–2.0 nats | > 2.0 nats |
| **Rank Histogram** | Uniform | Slight bias | Strong bias |
| **Observed Entropy** | High | Medium | Low |

- **Low Brier + Low KL:** Model accurately reproduces observed distribution ✓
- **High Brier + High KL:** Model significantly mismatches observations ✗
- **Uniform rank histogram:** Calibrated predictive uncertainty
- **Biased rank histogram:** Model over/under-confident

---

## Integration with Main Pipeline

All three analyses are integrated via `priority_analyses.jl`, which orchestrates:

```julia
result = run_priority_analyses(loaded, fitted, kernels, params, output_dir)
```

Returns:
- `path_uncertainty`: Per-path credible intervals
- `connectivity_matrix`: Stock connectivity point estimates
- `connectivity_uncertainty`: Stock connectivity credible intervals
- `posterior_predictive`: Model validation diagnostics
- `summary_file`: Unified HTML/text report

### Full Workflow

```julia
using bstm
include("docs/movement/movement_analysis_integration.jl")

# Run complete analysis
params = movement_parameters_snowcrab()
result = run_movement_analysis(params)

# Access individual outputs
path_unc = result.priority_analyses.path_uncertainty
conn = result.priority_analyses.connectivity_matrix
ppc = result.priority_analyses.posterior_predictive
```

---

## File Structure

```
docs/movement/
├── movement_analysis.jl                # Main pipeline (Phases 1–4)
├── movement_analysis_integration.jl    # Integration & CLI
├── path_uncertainty.jl                 # Priority analysis 1
├── connectivity_analysis.jl            # Priority analysis 2
├── posterior_predictive_check.jl       # Priority analysis 3
├── priority_analyses.jl                # Integration orchestrator
├── snowcrab_movement_data.jl           # Data loader
└── README.md                           # This file
```

---

## References

- **Brier Score:** Brier (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
- **KL Divergence:** Kullback & Leibler (1951). On information and sufficiency. *Annals of Mathematical Statistics*, 22(1), 79–86.
- **Rank Histogram:** Jolliffe & Stephenson (2012). *Forecast verification: A practitioner's guide in atmospheric science*. Wiley.
- **Posterior Predictive Checks:** Gelman et al. (2013). *Bayesian data analysis* (3rd ed.). Chapman & Hall.

---

## Future Extensions

Potential enhancements to the priority analyses:

1. **Cross-validation** — Hold-out test set evaluation
2. **Movement statistics** — Path efficiency, net displacement per individual
3. **Seasonal/temporal patterns** — Intra-annual movement phenology
4. **Habitat association** — Link path characteristics to environmental covariates
5. **Network centrality** — Betweenness, degree, closeness for bottleneck nodes
6. **Bayesian model comparison** — AIC/BIC or Bayesian stacking for model selection

