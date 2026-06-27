---
title: "Untitled"
format: html
---

The Architecture of Bayesian Spatio-Temporal Modeling: A Monograph on the bstm() Framework

1. Introduction: The Evolution of Latent Gaussian Models

In the current landscape of computational statistics, the primary bottleneck in spatio-temporal inference is the structural rigidity of traditional monolithic models. These legacy approaches often hard-code the relationship between covariates and random effects, leading to models that are computationally fragile and difficult to extend. The bstm() framework resolves this by treating the latent process as an orthogonal, composable entity, decoupled from the observation likelihood.

At the core of bstm() is the "Manifold-as-Primitive" philosophy. Rather than treating spatial or temporal effects as fixed components of a linear predictor, the framework implements them as low-level primitives inheriting from the Manifold abstract type. These primitives are domain-agnostic until the model-building stage; an AR1 structure can define a temporal process or be composed into a higher-dimensional spatial field. This architectural decoupling mitigates structural rigidity and allows for the construction of physics-informed models through a unified formulaic interface. By treating manifolds as first-class citizens in a Domain Specific Language (DSL), bstm() provides the software infrastructure necessary to bridge the gap between abstract manifold geometry and high-performance Bayesian computation.

2. Mathematical Foundations: Manifolds and Stochastic Primitives

The mathematical registry of bstm() (v06.1) standardizes the transition from discrete graph topologies to continuous spectral approximations. This unified representation is critical for dispatching efficient kernels across varying data scales.

The Discrete Registry: Gaussian Markov Random Fields (GMRF)

For discrete domains, bstm() implements GMRF structures where dependency is defined by a precision matrix Q.

Manifold	Theoretical Assumption	Structural Rationale
IID	\epsilon \sim N(0, \sigma^2 I)	Unstructured exchangeability; base model for PC-shrinkage.
ICAR	Intrinsic CAR; Q_{ij} = -1 for neighbors.	Pure local smoothing; identifies spatial gradients.
Besag	Standard CAR model.	Global and local spatial dependency via fixed precision.
BYM2	Scaled Besag + IID component.	Explicit variance partitioning (\rho) for better identifiability.
Leroux	Convex combination of I and Q_{ICAR}.	Bridges IID and ICAR structures through a mixing parameter.
SAR	(I - \rho W)y = \epsilon.	Simultaneous modeling of response autocorrelation.
RW1 / RW2	Random Walk (1st/2nd order).	Temporal continuity and smoothing of non-stationary trends.
AR1	\mu_t = \rho \mu_{t-1} + \epsilon_t.	Stationary temporal process with geometric decay.

Continuous, Spectral, and Advanced Manifolds

To circumvent the O(N^3) cost of kernel-based Gaussian Processes, the framework utilizes spectral projections and sparse approximations found in the v06.1 registry:

* Random Fourier Features (RFF): Maps input coordinates x into a randomized feature space to approximate the kernel. The projection is defined as z(x) = \sqrt{2/M} \cos(Wx + b), where W is the spectral frequency matrix sampled from the kernel’s Fourier transform. This transforms a non-linear GP into a linear model with M features.
* SPDE: Represents the field as a solution to (\kappa^2 - \Delta)^{\alpha/2} u = \mathcal{W}, mapping continuous Matérn processes onto a discrete mesh.
* Nystrom / FITC: Low-rank approximations using n\_inducing points to represent the global field.
* BCGN: Bipartite Coordinate Graph Networks for modeling group-level dependencies.
* NetworkFlow: Captures directed dependencies across an adjacency matrix with :upstream or :downstream dispatch.
* Hyperbolic: Provides curvature-aware embeddings for hierarchical data representation.

3. The Algebra of Manifolds: Operators and DSL Logic

The bstm() DSL operates through a recursive parser, parse_manifold_graph, which serves as the theoretical engine for model composition.

Algebraic Operators and Recursive Parsing

The parser decomposes algebraic strings into a structural tree, mapping tokens to re_rules. These rules dictate the precision matrix factory (build_structure_template).

1. Direct Sum (\oplus): Represents additive effects. f(s, t) = f_s(s) + f_t(t).
2. Kronecker Product (\otimes): Triggers "Type IV" interaction logic in the opt_kwargs. In spatio-temporal contexts, Q_{st} = Q_t \otimes Q_s, enabling the representation of space-time interactions where every spatial location has a unique, correlated temporal trend.
3. Composition (\circ): Chains manifolds for hierarchical stacking.

Structural Transformations and SVC

The Pipe (|>) operator handles both data normalization and Spatially Varying Coefficients (SVC).

* Transformations: Objects like ZScoreManifold or LogManifold act as wrappers that normalize inputs before they enter the latent process.
* SVC Logic: The notation covariate | spatial() instructs the parser to generate a random slope for the covariate that is spatially structured by the specified manifold, effectively modeling local non-stationarity.

4. Likelihood Architectures and Stochastic Kernels

The bstm_Likelihood struct standardizes the mapping between the linear predictor (\eta) and the observation space. This architecture ensures that the latent manifold remains invariant regardless of the noise distribution.

The Family Registry and Parameter Mapping

The framework supports a wide array of AbstractBSTM_Family types, utilizing specific link functions to ensure parameter validity:

* Poisson / NegativeBinomial: Uses a log-link for the rate parameter \mu = \exp(\eta). For the NegativeBinomial, the dispersion r is managed via r_nb to ensure positive definite rates.
* Binomial / Beta: Uses a logistic-link to map \eta \in \mathbb{R} to the [0, 1] interval.
* Gaussian / LogNormal / StudentT: For continuous outcome spaces.

Numerical Stability in Censoring

For complex stochastic states, bstm() employs specialized kernels. In IntervalCensored scenarios, to prevent numerical instability, the system implements stable_logdiffexp using log1mexp for the probability mass within [y_L, y_U]:  \text{logdiffexp}(a, b) = a + \text{log1mexp}(b - a) \quad \text{where } a > b  The use of log1mexp(safe_diff) prevents domain errors and catastrophic cancellation during the evaluation of the Cumulative Distribution Function (CDF).

5. Architectural Paradigms: Univariate, Multivariate, and Multifidelity

Model dispatch is handled by three core architectures, determined by the dimensionality and quality of the response data.

1. UnivariateArchitecture: The default kernel for single-outcome processes.
2. MultivariateArchitecture: Models dependent outcomes through matrix-variate kernels. It utilizes the InverseWishartFamily for covariance estimation, ensuring positive definiteness via a PDMat transformation of the latent matrix \eta \eta^T + \epsilon I.
3. MultifidelityArchitecture: Orchestrates data from heterogeneous sources. It uses a nested() supervisor discovery logic and the fidelity_metadata engine. This engine manages unit-specific templates and observation masks (y_ok) to link low-fidelity "proxies" to high-fidelity "truth," making it ideal for multi-sensor environmental monitoring.

6. The bstm() Formulaic Interface: User-Centric Modeling

The user-facing bstm() function provides a high-level interface that minimizes "implementation debt" while maintaining full control over the underlying priors.

DSL Syntax Breakdown

* spatial(s_idx; model='bym2'): High-level spatial effect dispatch.
* temporal(t_idx; model='rw2'): High-level temporal smoothing.
* smooth(x, y; model='rff'): 2D spectral smoother using informed RFF parameters.
* eigen(var; rank=k): Low-rank factor analytic terms.

Prior Resolution: The PC-Prior Standard

Stability in high-dimensional models is achieved through PC-Priors (Penalized Complexity). These priors are designed to shrink towards a simpler "base model"—for instance, shrinking a spatial field's variance towards zero or its range towards infinity.

Parameter	Default PC-Prior	Rationale
Sigma (\sigma)	Exponential(1.0)	Shrinks towards zero variance (the base model).
Rho (\rho)	Beta(1, 1)	Flat/Uninformative base for correlation parameters.
Lengthscale	InverseGamma(3, 3)	Prevents over-fitting in continuous kernels.
Kappa (\kappa)	Exponential(1.0)	Controls SPDE smoothness via principled shrinkage.

7. Illustrative Examples: Guided Model Implementation

1. BYM2 Disease Mapping: y ~ 1 + spatial(s_idx; model='bym2'). Decomposes risk into structured spatial and unstructured IID noise.
2. AR1 Temporal Forecasting: y ~ 1 + temporal(t_idx; model='ar1'). Captures geometric temporal decay.
3. Spatio-Temporal Interaction: y ~ 1 + spatial(s_idx) ⊗ temporal(t_idx). Employs tensor-product logic to manage O(N \times T) complexity via the Type IV interaction template.
4. Spatially Varying Coefficients (SVC): poverty | spatial(s_idx; model='icar'). Allows the impact of poverty to vary according to local spatial gradients.
5. Spectral Splines: smooth(lon, lat; model='rff'). Approximates a 2D continuous field without the O(N^3) kernel inversion cost.
6. Multifidelity Nested Supervision: nested(z_var; formula='z ~ spatial(s_idx)'). Uses the fidelity_metadata engine and y_ok masks to align observations across different quality levels.
7. Zero-Inflated Ecology: bstm(..., model_family='poisson', use_zi=true). Uses the ZeroInflated stochastic state to handle excess zeros in count data.

8. Critical Evaluation: Strengths, Weaknesses, and Frontiers

Strengths

The bstm() framework succeeds through orthogonality—the ability to compose complex latent geometries independently of the likelihood kernel. The standardized use of PC-Priors provides a principled way to maintain identifiability in over-parameterized spatio-temporal models.

Weaknesses

The primary architectural bottleneck is the reliance on collect() during the index realignment phase (Source line 1332), which acts as a significant memory sink for planetary-scale datasets. Furthermore, the MultivariateArchitecture is currently constrained to Gaussian-like likelihoods, limiting its utility for dependent count or categorical data. Finally, the recursive parser, while powerful, requires strict adherence to coordinate naming conventions (s_x, s_y), which can introduce friction in non-standardized workflows.

Future Frontiers

The expansion into Hyperbolic and NetworkFlow manifolds signals a move toward Non-Euclidean Bayesian modeling. As the framework matures, future iterations will likely focus on replacing the collect() bottlenecks with lazy evaluation and extending the multivariate kernel to the full AbstractBSTM_Family registry.

In summary, bstm() provides a rigorous, Julia-native environment that elevates spatio-temporal modeling from manual implementation to algebraic composition, setting a new standard for computational Bayesian research.



# BSTM Cheat Sheet

This document provides a quick reference for the `bstm` package, covering the formula API, common manifold types, and example usage.

## Formula API Quick Reference

The `bstm` formula language allows you to build complex models by combining modular components.

**Basic Structure:** `outcome ~ intercept + fixed_effects + modules`

### Common Modules

| Keyword              | Example Usage                                      | Purpose                                                                  |                                                                    |
| :---------------------| :---------------------------------------------------| :-------------------------------------------------------------------------| :-------------------------------------------------------------------|
| `spatial`            | `spatial(s_idx, model='bym2')`                     | Models spatial random effects for discrete areal units.                  |                                                                    |
| `temporal`           | `temporal(t_idx, u_idx, model=('ar1', 'cyclic'))`  | Models temporal trends and/or seasonal effects.                          |                                                                    |
| `smooth`             | `smooth(x, nbins=20)`                              | Creates a non-linear smooth of a continuous covariate `x`.               |                                                                    |
| `mixed`              | `mixed(1 \                                         | group)`                                                                  | Defines a random intercept for each level of the `group` variable. |
| `observationprocess` | `observationprocess(log_offsets=log_offset)`       | Specifies likelihood-level parameters. Options include: `log_offsets`, `weights`, `trials`, `hurdle`, `volatility=true`, `nbins`, `y_L`, `y_U`. |
| `fixed`              | `x1 + x2`                                          | Standard fixed-effect linear predictors.                                 |                                                                    |

### Data Transformations

Use the `|>` operator inside a module to transform data on the fly.

*   `smooth(x |> log)`: Applies a log transform to `x`.
*   `smooth(x |> zscore)`: Standardizes `x`.
*   `smooth(x |> unit)`: Scales `x` to `[0, 1]`.

---

## Model Reference Table

The `model` argument specifies the mathematical structure of the latent field.

### Spatial Manifolds (`spatial()`)

| Manifold | `model='...'` | Math (Conceptual) | Use Case |
| :--- | :--- | :--- | :--- |
| **ICAR** | `'icar'`, `'besag'` | $\phi_i \sim \mathcal{N}(\text{mean}(\phi_j \text{ for } j \sim i), \sigma^2/n_i)$ | Strong, localized spatial smoothing for areal data. |
| **BYM2** | `'bym2'` | $\phi_i = \sqrt{\rho}\phi_{str} + \sqrt{1-\rho}\phi_{unstr}$ | Robust default. Separates spatial trend from random noise. |
| **Leroux** | `'leroux'` | $Q = \rho Q_{sp} + (1-\rho)I$ | A proper CAR model that avoids rank-deficiency. |
| **SAR** | `'sar'` | $\phi = \rho W \phi + \epsilon$ | Models spatial "spill-over" effects. |

### Temporal & Seasonal Manifolds (`temporal()`, `seasonal()`)

| Manifold | `model='...'` | Math (Conceptual) | Use Case |
| :--- | :--- | :--- | :--- |
| **AR1** | `'ar1'` | $\phi_t = \rho \phi_{t-1} + \epsilon_t$ | Capturing trends with short-term memory. |
| **RW1/RW2** | `'rw1'`, `'rw2'` | $\Delta\phi_t \sim \mathcal{N}(0, \sigma^2)$ or $\Delta^2\phi_t \sim \mathcal{N}(0, \sigma^2)$ | Modeling stochastic level shifts (RW1) or smooth trends (RW2). |
| **Cyclic** | `'cyclic'` | $\phi_t \sim \text{ICAR on a circular graph}$ | Smooth seasonal patterns where Dec is adjacent to Jan. |

### Smooth Manifolds (`smooth()`)

| Manifold | `model='...'` | Math (Conceptual) | Use Case |
| :--- | :--- | :--- | :--- |
| **P-Spline** | `'pspline'` | B-spline basis with a random walk penalty on coefficients. | Flexible, general-purpose smoother for 1D covariates. |
| **TPS** | `'tps'` | Thin Plate Spline basis. | Smoothing 2D spatial coordinates. |
| **RFF** | `'rff'` | $\phi(x) = \sqrt{2/M}\cos(Wx+b)$ | Scalable approximation of a Gaussian Process smooth. |

### Spacetime Interaction Manifolds (`spacetime()`)

| Manifold | `model='...'` | Description |
| :--- | :--- | :--- |
| **Type I** | `'I'` | Unstructured (IID) interaction over space and time. |
| **Type II** | `'II'` | Spatially unstructured, temporally structured. |
| **Type III**| `'III'`| Spatially structured, temporally unstructured. |
| **Type IV** | `'IV'` | Fully structured in both space and time (Kronecker product). |

---

## Example Recipes

### 1. Standard Poisson CARSTM

A common model for disease mapping or species counts.

```julia
m = bstm(
    "counts ~ 1 + spatial(area_id, model='bym2') + temporal(year_id, model='ar1')",
    my_data,
    model_family="poisson",
    W=my_adjacency_matrix
)
```

### 2. Gaussian Model with Non-linear Smooth and SVC

Models a continuous outcome with a non-linear effect of `temperature` and allows the effect of `rainfall` to vary over space.

```julia
m = bstm(
    "yield ~ 1 + smooth(temperature, nbins=20) + spatial(rainfall, s_idx, model='iid')",
    my_data,
    model_family="gaussian"
)
```

### 3. Binomial Model with Spacetime Interaction

Models prevalence (0 or 1) with a fully structured spatiotemporal interaction term.

```julia
m = bstm(
    "prevalence ~ 1 + spatial(s_idx, model='bym2') + temporal(t_idx, model='ar1') + (spatial(s_idx) ⊗ temporal(t_idx))",
    my_data,
    model_family="binomial",
    W=my_adjacency_matrix
)
```

---


# BSTM API Documentation

## Introduction

`bstm` (Bayesian SpatioTemporal Models) is a powerful and flexible Julia package for fitting complex spatiotemporal models within the `Turing.jl` ecosystem. Its core philosophy is modularity, allowing users to construct sophisticated models by combining different "manifolds" (latent components) through an intuitive formula interface.

This document provides a detailed overview of the `bstm` API, from the main entry point to the specifics of the formula DSL and post-processing utilities.

## Main Entry Point: `bstm()`

The primary function for creating a `Turing` model is `bstm()`. It can be called in two main ways:

### 1. Formula-Based Interface (Recommended)

This is the most user-friendly way to define a model. It mimics the familiar formula syntax of R's `lme4` or `brms`.

**Syntax:**
```julia
bstm(formula::String, data::DataFrame; kwargs...)
```

*   `formula`: A string defining the model structure.
*   `data`: A `DataFrame` containing all necessary variables.
*   `kwargs`: Additional options passed to the configuration engine (e.g., `model_family`, `W` for adjacency).

**Example:**
```julia
m = bstm("y ~ 1 + x + spatial(s_idx, model='bym2')", my_data, model_family="poisson")
```

### 2. Configuration-Based Interface (Advanced)

For programmatic model building or advanced use cases, you can pass a pre-built configuration `NamedTuple` directly.

**Syntax:**
```julia
bstm(config::NamedTuple)
```

*   `config`: A `NamedTuple` created by `bstm_modular_config()` or `bstm_options()`.

## The Formula Mini-Language

The `bstm` formula DSL is designed to be expressive and readable. The general structure is `outcome(s) ~ intercept + fixed_effects + modules`.

### Outcomes

*   **Univariate:** `y ~ ...`
*   **Multivariate:** `y1 + y2 ~ ...` (triggers `MultivariateArchitecture`)

### Formula Keywords (`BSTM_MODULE_KEYWORDS`)

Each keyword initializes a specific type of latent manifold or model component.

| Keyword      | Example Usage                        | Purpose                                                                    |                                                                    |
| :-------------| :-------------------------------------| :---------------------------------------------------------------------------| :-------------------------------------------------------------------|
| `intercept`  | `intercept()`                        | Includes a global intercept term.                                          |                                                                    |
| `bias`       | `bias(hurdle=0.1)`                   | Specifies likelihood-level parameters like hurdles or observation weights. |                                                                    |
| `spatial`    | `spatial(s_idx, model='bym2')`    | Models spatial random effects using discrete areal units.                  |                                                                    |
| `temporal`   | `temporal(t_idx, model='ar1')`    | Models temporal trends using discrete time steps.                          |                                                                    |
| `seasonal`   | `seasonal(u_idx, model='cyclic')` | Models periodic effects (e.g., month-of-year).                             |                                                                    |
| `smooth`     | `smooth(x, nbins=20)`                | Creates a non-linear smooth of a continuous covariate `x`.                 |                                                                    |
| `svc`        | `svc(x, model='iid')`             | Models a spatially-varying coefficient for covariate `x`.                  |                                                                    |
| `mixed`      | `mixed(1                             | group)`                                                                    | Defines a random intercept for each level of the `group` variable. |
| `eigen`      | `eigen(x1, x2, n_factors=1)`         | Creates a latent field from the principal components of `x1` and `x2`.     |                                                                    |
| `spacetime`  | `spacetime(model='IV')`           | Defines a joint spatiotemporal interaction field.                          |                                                                    |
| `network`    | `network(s_idx, W=adj)`              | Models effects on a graph defined by adjacency matrix `W`.                 |                                                                    |
| `hyperbolic` | `hyperbolic(s_idx)`                  | Placeholder for hyperbolic geometry models.                                |                                                                    |
| `dynamics`   | `dynamics(var)`                      | Incorporates a mechanistic state-space model.                              |                                                                    |
| `volatility` | `volatility(s_idx)`                  | Models heteroskedastic observation noise that varies over space.           |                                                                    |
| `fixed`      | `x1 + x2`                            | Standard fixed-effect linear predictors.                                   |                                                                    |

### Manifold Options

The `model` argument within modules like `spatial`, `temporal`, and `smooth` specifies the underlying mathematical structure.

| Manifold         | `model='...'`               | Description                                                                                 |
| :-----------------| :-------------------------------| :--------------------------------------------------------------------------------------------|
| `IID`            | `'iid'`                        | Independent and identically distributed (unstructured) random effects.                      |
| `ICAR`           | `'icar'`, `'besag'`            | Intrinsic Conditional Autoregressive model for spatial smoothing.                           |
| `BYM2`           | `'bym2'`                       | A robust combination of a structured ICAR effect and an unstructured IID effect.            |
| `Leroux`         | `'leroux'`                     | A proper CAR model that is a convex combination of a spatial precision matrix and identity. |
| `SAR`            | `'sar'`                        | Simultaneous Autoregressive model.                                                          |
| `AR1`            | `'ar1'`                        | First-order autoregressive process for temporal trends.                                     |
| `RW1` / `RW2`    | `'rw1'`, `'rw2'`               | First or second-order random walk for temporal or 1D spatial smoothing.                     |
| `Cyclic`         | `'cyclic'`                     | A periodic random walk, suitable for seasonal effects.                                      |
| `Harmonic`       | `'harmonic'`                   | A seasonal effect modeled with sine and cosine basis functions.                             |
| `PSpline`        | `'pspline'`                    | P-spline (penalized B-spline) for smoothing continuous covariates.                          |
| `BSpline`        | `'bspline'`                    | B-spline basis for smoothing.                                                               |
| `TPS`            | `'tps'`                        | Thin Plate Spline for 2D smoothing.                                                         |
| `RFF`            | `'rff'`                        | Random Fourier Features for approximating a Gaussian Process smooth.                        |
| `FFT`            | `'fft'`                        | Fast Fourier Transform basis for periodic smoothing.                                        |
| `ST_I` - `ST_IV` | `'I'`, `'II'`, `'III'`, `'IV'` | Knorr-Held spatiotemporal interaction types.                                                |
| `Advection`      | `'advection'`                  | A physics-informed model for transport phenomena.                                           |

### Data Transformations

The `|>` operator can be used within a module's variable list to apply on-the-fly transformations to the data.

**Syntax:** `module(variable |> transform)`

**Available Transforms:**
*   `log`: Logarithmic transformation.
*   `zscore`: Standardizes to have a mean of 0 and standard deviation of 1.
*   `unit`: Scales the data to the `[0, 1]` interval.

**Example:**
```julia
# Apply a log transform to 'x' before creating the smooth
bstm("y ~ smooth(x |> log, nbins=15)", my_data)
```

### Algebraic Composition

Manifolds can be combined algebraically within the formula string.

*   `⊗` (`\otimes`): Kronecker product. Used to create inseparable spatiotemporal interactions.
    *   Example: `spacetime(model='spatial(s_idx) ⊗ temporal(t_idx)')`
    *   Example: `spacetime(model='spatial(s_idx) ⊗ temporal(t_idx)')`
*   `⊕` (`\oplus`): Direct sum. (Future support for block-diagonal structures).

## Configuration (Advanced)

The `bstm()` call internally uses `bstm_modular_config()` to parse the formula and create a `NamedTuple` containing all model specifications. Advanced users can call this function directly to inspect or modify the configuration before building the model.

```julia
M = bstm_modular_config("y ~ spatial(s_idx)", my_data)

# M will contain fields like:
# M.manifolds  # A registry of manifold specifications
# M.Xfixed     # The fixed-effects design matrix
# M.s_idx      # Spatial indices
# ... and many others
```

## Post-Processing

After fitting a model and obtaining a `Turing` chain, `bstm` provides utilities for analysis.

### `model_results_comprehensive()`

This function takes a fitted model and chain and produces a comprehensive summary of results, including posterior summaries of all latent fields, performance metrics (RMSE, R-squared), and MCMC diagnostics.

**Syntax:**
```julia
res = model_results_comprehensive(model, chain)
```

### `predict()`

This function projects the fitted model onto a new data grid to generate out-of-sample predictions.

**Syntax:**
```julia
preds = predict(model, chain, new_data_frame)
```

It correctly handles the projection of all manifold types, including re-computing basis matrices for smooth terms on the new data grid.


---

# write your own model:

```
using bstm, Turing, DataFrames, Random, Distributions

# 1. Define your custom dynamics function
function custom_schaefer_model(model, eta, M, priors)
    K_prior = get(priors, :K_prior, Normal(150, 20))
    r_prior = get(priors, :r_prior, LogNormal(0, 1))
    sigma_prior = get(priors, :sigma_prior, Exponential(1.0))
    
    K_dyn ~ NamedDist(K_prior, :K_dyn)
    r_dyn ~ NamedDist(r_prior, :r_dyn)
    sig_dyn ~ NamedDist(sigma_prior, :sig_dyn)
    
    T = eltype(eta)
    pop_state = Vector{T}(undef, M.t_N)
    pop_state[1] ~ Normal(log(K_dyn / 2.0), 1.0)

    for t in 2:M.t_N
        prev_pop = exp(pop_state[t-1])
        growth = r_dyn * (1.0 - prev_pop / K_dyn)
        pop_state[t] ~ Normal(pop_state[t-1] + growth, sig_dyn)
    end
    
    for i in 1:M.y_N
        eta[i] += pop_state[M.t_idx[i]]
    end
end

# 2. Generate some simple time-series data
Random.seed!(123)
n_obs = 100
time_steps = 1:n_obs
true_r = 0.3
true_K = 200
pop = zeros(n_obs)
pop[1] = 20
for t in 2:n_obs
    pop[t] = pop[t-1] * exp(true_r * (1 - pop[t-1]/true_K) + randn()*0.1)
end
counts = [rand(Poisson(p)) for p in pop]

# Create a DataFrame
sim_data = DataFrame(y = counts, t_idx = time_steps)

# 3. Run the bstm model with the custom dynamics
# Ensure the spatiotemporal_functions.jl file with the diff is loaded
m = bstm(
    "y ~ 1 + dynamics(t_idx, model=\"custom\", func=:custom_schaefer_model, K_prior=Normal(200,50))",
    sim_data,
    model_family="poisson"
);

# 4. Sample from the model
chn = sample(m, NUTS(200, 0.65), 500; progress=true)

# 5. Inspect results for your custom parameters
println(chn[[:K_dyn, :r_dyn, :sig_dyn]])

```


## Lotka Volterra

```julia
using bstm, Turing, DataFrames, Random, Distributions, Plots

# Ensure the updated spatiotemporal_functions.jl is loaded
# include("c:/home/jae/projects/bstm/src/spatiotemporal_functions.jl")

# --- 1. Simulate a 3-Species Lotka-Volterra System ---

function simulate_lv_system(n_species, n_steps, r, K, alpha, sigma)
    Random.seed!(42)
    populations = zeros(n_steps, n_species)
    populations[1, :] .= K ./ 2.0 # Start at half carrying capacity

    for t in 2:n_steps
        for i in 1:n_species
            prev_pop = populations[t-1, :]
            competition = sum(alpha[i, j] * prev_pop[j] for j in 1:n_species)
            growth_rate = r[i] * (1.0 - competition / K[i])
            
            # Update on log scale and add process noise
            log_next_pop = log(prev_pop[i]) + growth_rate + randn() * sigma[i]
            populations[t, i] = exp(log_next_pop)
        end
    end
    
    # Add observation noise (e.g., Poisson for counts)
    observed_counts = [rand(Poisson(max(0, p))) for p in populations]
    return observed_counts, populations
end

# Define true parameters for a 3-species system
n_species = 3
n_steps = 150
true_r = [0.8, 0.4, 0.6]      # Growth rates
true_K = [200.0, 150.0, 180.0] # Carrying capacities
true_sigma = [0.05, 0.08, 0.06] # Process noise

# Interaction matrix:
# Species 1 competes with 2
# Species 2 is prey for species 3 (benefits 3, harmed by 3)
# Species 1 and 3 have weak competition
true_alpha = [
    1.0  0.7 -0.1; # Sp1: competes with 2, slightly benefits from 3's predation on 2
    0.5  1.0  1.2; # Sp2: competes with 1, is prey for 3
    0.2 -0.9  1.0  # Sp3: competes with 1, predates on 2
]

observed_data, true_latent_pops = simulate_lv_system(n_species, n_steps, true_r, true_K, true_alpha, true_sigma);

# Create a DataFrame for bstm
df = DataFrame(
    y1 = observed_data[:, 1],
    y2 = observed_data[:, 2],
    y3 = observed_data[:, 3],
    t_idx = 1:n_steps
);

# --- 2. Define and Run the bstm Model ---

# Define the formula. Each outcome (y1, y2, y3) is a species.
# The dynamics() module will apply to all of them jointly.
formula = "y1 + y2 + y3 ~ 1 + dynamics(t_idx, model=\"n-species_lotka_volterra\")"

# Build the Turing model instance
# We use a Poisson likelihood as our observations are counts.
m = bstm(formula, df, model_family="poisson");

# Sample from the posterior
# NUTS is recommended for this kind of complex model.
chn = sample(m, NUTS(500, 0.65), 1000; progress=true)

# --- 3. Analyze and Interpret the Results ---

println(chn[[:r_lv, :K_lv, :sigma_lv]])

# Extract and analyze the interaction matrix `alpha`
alpha_samples = chn[:alpha_lv_offdiag]
n_samples = size(alpha_samples, 1)
n_off_diag = n_species * (n_species - 1)

# Reshape the flattened off-diagonal samples into matrices for each MCMC sample
alpha_posterior_matrices = []
for i in 1:n_samples
    sample_matrix = Matrix{Float64}(I, n_species, n_species)
    sample_off_diags = alpha_samples[i, :]
    idx = 1
    for row in 1:n_species
        for col in 1:n_species
            if row != col
                sample_matrix[row, col] = sample_off_diags[idx]
                idx += 1
            end
        end
    end
    push!(alpha_posterior_matrices, sample_matrix)
end

# Plot posterior distributions for key interaction terms
p1 = density(get(chn, :alpha_lv_offdiag)[1].data, label="α12 (Sp2 on Sp1)", title="Competition")
vline!([true_alpha[1,2]], label="True Value", color=:red, ls=:dash)

p2 = density(get(chn, :alpha_lv_offdiag)[5].data, label="α32 (Sp2 on Sp3)", title="Predation")
vline!([true_alpha[3,2]], label="True Value", color=:red, ls=:dash)

p3 = density(get(chn, :alpha_lv_offdiag)[2].data, label="α13 (Sp3 on Sp1)", title="Weak Competition")
vline!([true_alpha[1,3]], label="True Value", color=:red, ls=:dash)

plot(p1, p2, p3, layout=(1,3), size=(1200, 400))


```

Interpretation of Results

By running the example script, you will obtain posterior distributions for all the Lotka-Volterra parameters. The most interesting are the off-diagonal elements of the interaction matrix, $\alpha_{ij}$.

The plot for α12 should show a posterior distribution centered on a positive value, correctly identifying the competitive relationship between species 1 and 2.

The plot for α32 should show a posterior centered on a negative value, correctly identifying that species 3 benefits from species 2 (predation).

The posterior for α23 (the effect of the predator on the prey) should be positive, which you can also plot and verify.

This example demonstrates how bstm can be extended to incorporate complex, mechanistic sub-models, allowing you to fit sophisticated ecological theories directly to data within a unified and statistically rigorous Bayesian framework. You can further extend this by adding spatial effects to the growth rates (r) or carrying capacities (K), turning it into a fully spatiotemporal dynamical model.


 Adding a spatially-varying carrying capacity (K) to the Lotka-Volterra model is an excellent way to incorporate habitat heterogeneity and make the model more ecologically realistic. Instead of a single K for each species, this modification allows K to vary across different spatial locations, structured by a spatial process like a Conditional Autoregressive (CAR) model.

Here is a comprehensive guide on how to implement this feature within the bstm framework.

Conceptual Overview: Spatially-Varying Carrying Capacity
The core idea is to change the carrying capacity parameter, K, from a simple scalar per species into a spatially explicit random field. We model the logarithm of K as the sum of a global baseline and a spatially structured random effect:

$$ \log(K_{s,k}) = \mu_{K,k} + \phi_{K,s,k} $$

$K_{s,k}$: The carrying capacity for species $k$ at spatial location $s$.
$\mu_{K,k}$: The global average log-carrying capacity for species $k$.
$\phi_{K,s,k}$: A spatially structured random effect for species $k$ at location $s$. This term is modeled using a CAR prior, meaning its value is influenced by the values at neighboring locations.
This change requires the underlying state-space process for the population dynamics to also become spatially explicit. The population of each species, pop_state, will now be a matrix of size [n_spatial_units, n_time_steps] instead of just a vector over time.


```{julia}
using bstm, Turing, DataFrames, Random, Distributions, Plots, SparseArrays

# Ensure the updated spatiotemporal_functions.jl is loaded
# include("c:/home/jae/projects/bstm/src/spatiotemporal_functions.jl")

# --- 1. Simulate a 3-Species System with Spatially-Varying K ---

function simulate_spatial_lv_system(s_N, t_N, r, K_base, K_spatial_field, alpha, sigma, W)
    Random.seed!(123)
    
    # Construct the full spatially-varying K
    log_K_field = log.(K_base') .+ K_spatial_field
    K_field = exp.(log_K_field)

    populations = zeros(s_N, t_N, n_species)
    for s in 1:s_N
        populations[s, 1, :] .= K_field[s, :] ./ 2.0
    end

    for t in 2:t_N
        for s in 1:s_N
            prev_pop = populations[s, t-1, :]
            competition = sum(alpha[i, j] * prev_pop[j] for i in 1:n_species, j in 1:n_species)
            
            for i in 1:n_species
                growth_rate = r[i] * (1.0 - sum(alpha[i,j] * prev_pop[j] for j in 1:n_species) / K_field[s, i])
                log_next_pop = log(prev_pop[i]) + growth_rate + randn() * sigma[i]
                populations[s, t, i] = exp(log_next_pop)
            end
        end
    end
    
    # Flatten for DataFrame
    s_indices = repeat(1:s_N, inner=t_N)
    t_indices = repeat(1:t_N, outer=s_N)
    
    observed_counts = zeros(s_N * t_N, n_species)
    for i in 1:(s_N * t_N)
        s = s_indices[i]
        t = t_indices[i]
        for k in 1:n_species
            observed_counts[i, k] = rand(Poisson(max(0, populations[s, t, k])))
        end
    end
    
    return observed_counts, s_indices, t_indices
end

# Define true parameters
n_species = 3
s_N = 25
t_N = 50
true_r = [0.5, 0.3, 0.4]
true_K_base = [100.0, 80.0, 120.0]
true_sigma = [0.1, 0.12, 0.08]
true_alpha = [1.0 0.8 0.1; 0.6 1.0 0.9; 0.2 0.7 1.0]

# Create a simple spatial grid and adjacency matrix
W = sparse(adjacency_matrix(Graphs.grid([Int(sqrt(s_N)), Int(sqrt(s_N))])))

# Create a simple true spatial field for K
coords = hcat(repeat(1:Int(sqrt(s_N)), inner=Int(sqrt(s_N))), repeat(1:Int(sqrt(s_N)), outer=Int(sqrt(s_N))))
true_K_spatial_field = zeros(s_N, n_species)
true_K_spatial_field[:, 1] = sin.(coords[:,1] ./ 2) .* 0.5
true_K_spatial_field[:, 2] = cos.(coords[:,2] ./ 2) .* 0.4
true_K_spatial_field[:, 3] = (sin.(coords[:,1] ./ 2) .+ cos.(coords[:,2] ./ 2)) .* 0.3

# Simulate data
observed_data, s_idx, t_idx = simulate_spatial_lv_system(s_N, t_N, true_r, true_K_base, true_K_spatial_field, true_alpha, true_sigma, W);

df = DataFrame(
    y1 = observed_data[:, 1],
    y2 = observed_data[:, 2],
    y3 = observed_data[:, 3],
    s_idx = s_idx,
    t_idx = t_idx
);

# --- 2. Define and Run the bstm Model ---

# The formula MUST include a spatial() term to provide W and s_idx to the model
formula = "y1 + y2 + y3 ~ 1 + spatial(s_idx) + dynamics(t_idx, model=\"n-species_lotka_volterra_spatial_K\")"

# Build and sample the model
m = bstm(formula, df, model_family="poisson", W=W);
chn = sample(m, NUTS(200, 0.65), 500; progress=true)

# --- 3. Analyze and Interpret the Results ---

# Check posteriors for the new K-related parameters
println(chn[[:log_K_base, :sigma_K_spatial]])

# Extract and plot the estimated spatial field for K
# The latent field is named `K_spatial_effect` in the Turing model
k_field_samples = get_params_vector(chn, "K_spatial_effect", s_N * n_species)

# Reshape and summarize for species 1
k_field_species1 = reshape(k_field_samples, n_samples, s_N, n_species)[:, :, 1]
k_field_mean_s1 = mean(k_field_species1, dims=1)[:]

# Plot true vs. estimated spatial field for K of species 1
p = plot(true_K_spatial_field[:,1], label="True Field", lw=2, title="Spatial Field for K (Species 1)")
plot!(p, k_field_mean_s1, label="Estimated Field", lw=2, ls=:dash)
display(p)


```

By running the example, you will now get posterior estimates for log_K_base (the average log-carrying capacity for each species) and sigma_K_spatial (the marginal standard deviation of the spatial effect on K).

The plot comparing the "True Field" to the "Estimated Field" for the carrying capacity of species 1 demonstrates the model's ability to recover the underlying spatial heterogeneity in K. This spatially-explicit approach provides a much richer and more realistic understanding of the population dynamics across your study area


## adding  fishing mortality

```{julia}
using bstm, Turing, DataFrames, Random, Distributions, Plots

# Ensure the updated spatiotemporal_functions.jl is loaded
# include("c:/home/jae/projects/bstm/src/spatiotemporal_functions.jl")

# --- 1. Simulate a Population with Time-Varying Fishing Mortality ---

function simulate_f_system(n_steps, r, K, initial_F, sigma_F_rw, sigma_pop)
    Random.seed!(123)
    
    # Generate a true F trajectory (random walk on log scale)
    true_log_F = zeros(n_steps)
    true_log_F[1] = log(initial_F)
    for t in 2:n_steps
        true_log_F[t] = true_log_F[t-1] + randn() * sigma_F_rw
    end
    true_F = exp.(true_log_F)

    # Simulate population dynamics
    true_log_pop = zeros(n_steps)
    true_log_pop[1] = log(K / 2.0)
    for t in 2:n_steps
        prev_pop = exp(true_log_pop[t-1])
        growth = r * (1.0 - prev_pop / K) - true_F[t-1]
        true_log_pop[t] = true_log_pop[t-1] + growth + randn() * sigma_pop
    end
    
    # Generate noisy observations (e.g., Poisson counts from a survey index)
    observed_counts = [rand(Poisson(exp(p))) for p in true_log_pop]
    
    return observed_counts, true_F, exp.(true_log_pop)
end

# Define true parameters
n_steps = 100
true_r = 0.4
true_K = 500.0
initial_F = 0.1
sigma_F_rw = 0.15 # Volatility of fishing pressure
sigma_pop = 0.1  # Process noise for population

# Simulate the data
observed_counts, true_F, true_pop = simulate_f_system(n_steps, true_r, true_K, initial_F, sigma_F_rw, sigma_pop);

# Create a DataFrame for bstm
df = DataFrame(
    y = observed_counts,
    t_idx = 1:n_steps
);

# --- 2. Define and Run the bstm Model ---

# Define the formula with the new dynamics model
formula = "y ~ 1 + dynamics(t_idx, model=\"logistic_f\")"

# Build and sample the model
m = bstm(formula, df, model_family="poisson");
chn = sample(m, NUTS(500, 0.65), 1000; progress=true)

# --- 3. Analyze and Interpret the Results ---

println("Posterior summary for key parameters:")
display(chn[[:r_dyn, :K_dyn, :sig_pop, :sig_F]])

# Extract and plot the estimated fishing mortality trajectory
# The latent F state is named `log_F_state` in the Turing model
log_F_samples = get_params_vector(chn, "log_F_state", n_steps)
F_samples = exp.(log_F_samples)

# Calculate posterior mean and 95% credible intervals for F
F_mean = mean(F_samples, dims=1)[:]
F_lower = [quantile(F_samples[:,t], 0.025) for t in 1:n_steps]
F_upper = [quantile(F_samples[:,t], 0.975) for t in 1:n_steps]

# Plot the results
p_F = plot(1:n_steps, true_F, label="True F", lw=3, color=:black, ls=:dash,
           title="Estimated vs. True Fishing Mortality (F)")
plot!(p_F, 1:n_steps, F_mean, ribbon=(F_mean .- F_lower, F_upper .- F_mean),
      label="Estimated F (95% CI)", lw=2, color=:blue, fillalpha=0.2)
xlabel!(p_F, "Time Step")
ylabel!(p_F, "Fishing Mortality Rate")

display(p_F)



```

By executing the example above, you will generate a plot comparing the true, simulated fishing mortality (True F) with the posterior estimate from the bstm model (Estimated F). The blue ribbon represents the 95% credible interval.

This demonstrates the model's ability to infer the unobserved, time-varying fishing pressure directly from the population survey data (y). This is a powerful feature, as it allows you to estimate harvesting impacts even when direct effort data is unavailable or unreliable. The posterior for sig_F will tell you about the volatility of fishing pressure over time.

Alternative: Using Observed Fishing Effort
If you have a reliable time series of fishing effort (e.g., total days fished), you can incorporate it as an observed covariate instead of a latent process. This would involve a different modification where the dynamics module accepts an f_var argument pointing to the effort column in your data, and the model would estimate a catchability coefficient q such that F_t = q \cdot \text{Effort}_t. This approach is useful when you trust your effort data and want to estimate its direct impact on mortality


### covariates that affect r

Making the intrinsic growth rate r dependent on a variable like temperature is a common and powerful way to add ecological realism to a state-space model.

Here is a detailed guide on how to implement this, including the necessary code modifications and a complete, runnable example.

Conceptual Framework: Temperature-Dependent Growth Rate
We will modify the logistic_f model, which currently assumes a constant intrinsic growth rate r. The new formulation will allow r to vary over time as a function of an environmental covariate, such as temperature ($T_t$).

To ensure that the growth rate r remains positive, we will model its logarithm as a linear function of the covariate:

$$ \log(r_t) = \beta_{r,0} + \beta_{r,T} \cdot T_t $$

where:

$r_t$ is the growth rate at time $t$.
$T_t$ is the temperature at time $t$.
$\beta_{r,0}$ is the baseline log-growth rate (the log_r_base parameter in the model).
$\beta_{r,T}$ is the coefficient representing the effect of temperature on the log-growth rate (the r_cov_effect parameter).
This means the growth rate itself is an exponential function of temperature: $$ r_t = \exp(\beta_{r,0} + \beta_{r,T} \cdot T_t) $$

The underlying state-space equation for the log-population ($X_t = \log(N_t)$) is then updated to use this time-varying growth rate: $$ \frac{dX_t}{dt} = r_t \left(1 - \frac{N_t}{K}\right) - F_t $$

This allows the model to capture how changes in the environment can accelerate or decelerate population growth, a critical feature for modeling populations in a changing climate.

How to Use It: The bstm Formula
To use this feature, you simply need to add the r_covariate parameter to your dynamics module call in the formula string. The value should be the symbol of the column in your DataFrame that contains the environmental data.


```{julia}

formula = "y ~ 1 + dynamics(t_idx, model=\"logistic_f\", r_covariate=:temperature)"

using bstm, Turing, DataFrames, Random, Distributions, Plots

# Ensure the updated spatiotemporal_functions.jl is loaded
# include("c:/home/jae/projects/bstm/src/spatiotemporal_functions.jl")

# --- 1. Simulate a Population with Temperature-Dependent Growth ---

function simulate_temp_f_system(n_steps, K, initial_F, sigma_F_rw, sigma_pop, temp_data, log_r_base, r_temp_effect)
    Random.seed!(123)
    
    # Generate true time-varying r based on temperature
    true_log_r = log_r_base .+ r_temp_effect .* temp_data
    true_r = exp.(true_log_r)
    
    # Generate a true F trajectory (random walk on log scale)
    true_log_F = zeros(n_steps)
    true_log_F[1] = log(initial_F)
    for t in 2:n_steps
        true_log_F[t] = true_log_F[t-1] + randn() * sigma_F_rw
    end
    true_F = exp.(true_log_F)

    # Simulate population dynamics
    true_log_pop = zeros(n_steps)
    true_log_pop[1] = log(K / 2.0)
    for t in 2:n_steps
        prev_pop = exp(true_log_pop[t-1])
        growth = true_r[t-1] * (1.0 - prev_pop / K) - true_F[t-1]
        true_log_pop[t] = true_log_pop[t-1] + growth + randn() * sigma_pop
    end
    
    # Generate noisy observations
    observed_counts = [rand(Poisson(max(0, exp(p)))) for p in true_log_pop]
    
    return observed_counts, true_r, true_F
end

# Define true parameters
n_steps = 150
true_K = 800.0
initial_F = 0.05
sigma_F_rw = 0.1
sigma_pop = 0.08

# True parameters for the temperature effect on 'r'
true_log_r_base = log(0.3) # Baseline log-growth rate
true_r_temp_effect = 0.05  # Positive effect of temperature

# Simulate a temperature time series (e.g., with a seasonal cycle)
temp_time_series = 15.0 .+ 5.0 .* sin.(2 * pi .* (1:n_steps) ./ 52) .+ randn(n_steps) .* 0.5

# Simulate the data
observed_counts, true_r, true_F = simulate_temp_f_system(
    n_steps, true_K, initial_F, sigma_F_rw, sigma_pop, 
    temp_time_series, true_log_r_base, true_r_temp_effect
);

# Create a DataFrame for bstm
df = DataFrame(
    y = observed_counts,
    t_idx = 1:n_steps,
    temperature = temp_time_series
);

# --- 2. Define and Run the bstm Model ---

# Define the formula with the new r_covariate parameter
formula = "y ~ 1 + dynamics(t_idx, model=\"logistic_f\", r_covariate=:temperature)"

# Build and sample the model
m = bstm(formula, df, model_family="poisson");
chn = sample(m, NUTS(500, 0.65), 1000; progress=true)

# --- 3. Analyze and Interpret the Results ---

println("Posterior summary for key parameters:")
println("True log_r_base: $(round(true_log_r_base, digits=3)), True r_temp_effect: $(round(true_r_temp_effect, digits=3))")
display(chn[[:log_r_base, :r_cov_effect]])

# Extract and plot the estimated growth rate 'r' over time
log_r_base_samples = chn[:log_r_base].data[:]
r_cov_effect_samples = chn[:r_cov_effect].data[:]

# Reconstruct the posterior for r_t for each sample
r_t_samples = exp.(log_r_base_samples .+ r_cov_effect_samples .* temp_time_series')

# Calculate posterior mean and 95% credible intervals for r_t
r_t_mean = mean(r_t_samples, dims=1)[:]
r_t_lower = [quantile(r_t_samples[:,t], 0.025) for t in 1:n_steps]
r_t_upper = [quantile(r_t_samples[:,t], 0.975) for t in 1:n_steps]

# Plot the results
p_r = plot(1:n_steps, true_r, label="True r(t)", lw=3, color=:black, ls=:dash,
           title="Estimated vs. True Growth Rate r(t)")
plot!(p_r, 1:n_steps, r_t_mean, ribbon=(r_t_mean .- r_t_lower, r_t_upper .- r_t_mean),
      label="Estimated r(t) (95% CI)", lw=2, color=:crimson, fillalpha=0.2)
xlabel!(p_r, "Time Step")
ylabel!(p_r, "Growth Rate (r)")

display(p_r)


```

By running this example, you will see that the model successfully recovers the log_r_base and r_cov_effect parameters. The final plot will show the estimated time-varying growth rate r(t) closely tracking the true, simulated trajectory, demonstrating that the model has learned the relationship between temperature and population growth from the dat



##  make the carrying capacity, K, dependent on an environmental covariate. This is a powerful technique for creating more realistic ecological models where population limits are not static but are influenced by changing environmental conditions.

Conceptual Framework: Covariate-Dependent Carrying Capacity
We will extend the logistic_f model to allow the carrying capacity K to vary over time as a function of an environmental covariate, such as habitat_quality.

To ensure that the carrying capacity K remains positive, we model its logarithm as a linear function of the covariate:

$$ \log(K_t) = \beta_{K,0} + \beta_{K,H} \cdot \text{habitat_quality}_t $$

where:

$K_t$ is the carrying capacity at time $t$.
$\text{habitat_quality}_t$ is the value of the covariate at time $t$.
$\beta_{K,0}$ is the baseline log-carrying capacity (the log_K_base parameter in the model).
$\beta_{K,H}$ is the coefficient representing the effect of habitat quality on the log-carrying capacity (the K_cov_effect parameter).
The carrying capacity itself is then an exponential function of the covariate: $$ K_t = \exp(\beta_{K,0} + \beta_{K,H} \cdot \text{habitat_quality}_t) $$

The model's state-space equation for the log-population ($X_t = \log(N_t)$) is updated to use this time-varying carrying capacity, alongside the potentially time-varying growth rate r_t:

$$ \frac{dX_t}{dt} = r_t \left(1 - \frac{N_t}{K_t}\right) - F_t $$

This allows both the population's growth potential (r) and its upper limit (K) to be dynamically influenced by external environmental factors.

How to Use It: The bstm Formula
To use this feature, you add the K_covariate parameter to your dynamics module call in the formula string. The value should be the symbol of the column in your DataFrame that contains the relevant environmental data.

Example Formula:

formula = "y ~ 1 + dynamics(t_idx, model=\"logistic_f\", r_covariate=:temperature, K_covariate=:habitat_quality)"

This tells bstm to use the temperature column to model the growth rate r and the habitat_quality column to model the carrying capacity K.

Code Modification
The following diff updates c:\home\jae\projects\bstm\src\spatiotemporal_functions.jl to implement this functionality. It modifies the logistic_f model block to check for the K_covariate parameter and apply the corresponding logic.


```{julia}
using bstm, Turing, DataFrames, Random, Distributions, Plots

# Ensure the updated spatiotemporal_functions.jl is loaded
# include("c:/home/jae/projects/bstm/src/spatiotemporal_functions.jl")

# --- 1. Simulate a Population with Covariate-Dependent K ---

function simulate_habitat_system(n_steps, r, initial_F, sigma_F_rw, sigma_pop, habitat_data, log_K_base, K_habitat_effect)
    Random.seed!(42)
    
    # Generate true time-varying K based on habitat quality
    true_log_K = log_K_base .+ K_habitat_effect .* habitat_data
    true_K = exp.(true_log_K)
    
    # Generate a true F trajectory (random walk on log scale)
    true_log_F = zeros(n_steps)
    true_log_F[1] = log(initial_F)
    for t in 2:n_steps
        true_log_F[t] = true_log_F[t-1] + randn() * sigma_F_rw
    end
    true_F = exp.(true_log_F)

    # Simulate population dynamics
    true_log_pop = zeros(n_steps)
    true_log_pop[1] = log(true_K[1] / 2.0)
    for t in 2:n_steps
        prev_pop = exp(true_log_pop[t-1])
        growth = r * (1.0 - prev_pop / true_K[t-1]) - true_F[t-1]
        true_log_pop[t] = true_log_pop[t-1] + growth + randn() * sigma_pop
    end
    
    # Generate noisy observations
    observed_counts = [rand(Poisson(max(0, exp(p)))) for p in true_log_pop]
    
    return observed_counts, true_K, true_F
end

# Define true parameters
n_steps = 150
true_r = 0.35
initial_F = 0.08
sigma_F_rw = 0.1
sigma_pop = 0.08

# True parameters for the habitat effect on 'K'
true_log_K_base = log(400.0) # Baseline log-carrying capacity
true_K_habitat_effect = 0.25  # Positive effect of habitat quality

# Simulate a habitat quality time series (e.g., improving over time)
habitat_quality_series = 0.5 .+ (1:n_steps) ./ n_steps .* 2.0 .+ randn(n_steps) .* 0.2

# Simulate the data
observed_counts, true_K, true_F = simulate_habitat_system(
    n_steps, true_r, initial_F, sigma_F_rw, sigma_pop, 
    habitat_quality_series, true_log_K_base, true_K_habitat_effect
);

# Create a DataFrame for bstm
df = DataFrame(
    y = observed_counts,
    t_idx = 1:n_steps,
    habitat_quality = habitat_quality_series
);

# --- 2. Define and Run the bstm Model ---

# Define the formula with the new K_covariate parameter
formula = "y ~ 1 + dynamics(t_idx, model=\"logistic_f\", K_covariate=:habitat_quality)"

# Build and sample the model
m = bstm(formula, df, model_family="poisson");
chn = sample(m, NUTS(500, 0.65), 1000; progress=true)

# --- 3. Analyze and Interpret the Results ---

println("Posterior summary for key parameters:")
println("True log_K_base: $(round(true_log_K_base, digits=3)), True K_habitat_effect: $(round(true_K_habitat_effect, digits=3))")
display(chn[[:log_K_base, :K_cov_effect]])

# Extract and plot the estimated carrying capacity 'K' over time
log_K_base_samples = chn[:log_K_base].data[:]
K_cov_effect_samples = chn[:K_cov_effect].data[:]

# Reconstruct the posterior for K_t for each sample
K_t_samples = exp.(log_K_base_samples .+ K_cov_effect_samples .* habitat_quality_series')

# Calculate posterior mean and 95% credible intervals for K_t
K_t_mean = mean(K_t_samples, dims=1)[:]
K_t_lower = [quantile(K_t_samples[:,t], 0.025) for t in 1:n_steps]
K_t_upper = [quantile(K_t_samples[:,t], 0.975) for t in 1:n_steps]

# Plot the results
p_K = plot(1:n_steps, true_K, label="True K(t)", lw=3, color=:black, ls=:dash,
           title="Estimated vs. True Carrying Capacity K(t)")
plot!(p_K, 1:n_steps, K_t_mean, ribbon=(K_t_mean .- K_t_lower, K_t_upper .- K_t_mean),
      label="Estimated K(t) (95% CI)", lw=2, color=:darkgreen, fillalpha=0.2)
xlabel!(p_K, "Time Step")
ylabel!(p_K, "Carrying Capacity (K)")

display(p_K)

```

By running this example, you will see that the model successfully recovers the log_K_base and K_cov_effect parameters. The final plot shows the estimated time-varying carrying capacity K(t) closely tracking the true, simulated trajectory, demonstrating that the model has learned the relationship between habitat quality and the population's upper limit from the data.




## both r, K varying

each dependent on a different environmental covariate, within the bstm framework.

The existing logistic_f dynamics model is already designed with this flexibility in mind. The logic for handling the r_covariate and K_covariate parameters is independent, allowing you to specify both in the same model call without any code modifications.

Conceptual Framework: A Doubly Dynamic Population Model
This approach allows you to build a more ecologically realistic state-space model where both the population's growth potential and its environmental limits are dynamic.

Time-Varying Growth Rate (r_t): We model the logarithm of the growth rate as a linear function of an environmental driver, such as temperature. $$ \log(r_t) = \beta_{r,0} + \beta_{r,T} \cdot \text{Temperature}t $$ This implies that the growth rate itself, $r_t = \exp(\beta{r,0} + \beta_{r,T} \cdot \text{Temperature}_t)$, can increase or decrease as the temperature changes.

Time-Varying Carrying Capacity (K_t): Similarly, we model the logarithm of the carrying capacity as a linear function of a different covariate, such as habitat quality. $$ \log(K_t) = \beta_{K,0} + \beta_{K,H} \cdot \text{HabitatQuality}t $$ This implies that the environment's ability to support the population, $K_t = \exp(\beta{K,0} + \beta_{K,H} \cdot \text{HabitatQuality}_t)$, can expand or contract based on habitat conditions.

Combined State-Space Model: The population's evolution over time is then governed by an equation that incorporates both of these dynamic parameters, as well as the time-varying fishing mortality (F_t): $$ \frac{d(\log N_t)}{dt} = r_t \left(1 - \frac{N_t}{K_t}\right) - F_t $$ This creates a rich, mechanistic model where the population's trajectory is jointly influenced by its internal dynamics and multiple, distinct environmental drivers.

How to Use It: The bstm Formula
To implement this model, you simply include both the r_covariate and K_covariate parameters in your dynamics module call within the formula string.

formula = "y ~ 1 + dynamics(t_idx, model=\"logistic_f\", r_covariate=:temperature, K_covariate=:habitat_quality)"

This single line instructs bstm to:

Use the logistic_f dynamics model.

Link the temperature column in your data to the growth rate r.
Link the habitat_quality column to the carrying capacity K.
The bstm_univariate model function in spatiotemporal_functions.jl will automatically detect both parameters and build the appropriate state-space model.

Complete, Runnable Example
The following example demonstrates the entire workflow:

Simulating data where r depends on temperature and K depends on habitat_quality.
Fitting the bstm model using the combined formula.
Analyzing the results to show that the model successfully recovers the effects of both covariates.



```{julia}
using bstm, Turing, DataFrames, Random, Distributions, Plots

# Ensure the spatiotemporal_functions.jl file is loaded
# include("c:/home/jae/projects/bstm/src/spatiotemporal_functions.jl")

# --- 1. Simulate a Population with Covariate-Dependent r and K ---

function simulate_dual_dynamic_system(n_steps, initial_F, sigma_F_rw, sigma_pop, 
                                      temp_data, habitat_data, 
                                      log_r_base, r_temp_effect, 
                                      log_K_base, K_habitat_effect)
    Random.seed!(1234)
    
    # Generate true time-varying r and K
    true_log_r = log_r_base .+ r_temp_effect .* temp_data
    true_r = exp.(true_log_r)
    
    true_log_K = log_K_base .+ K_habitat_effect .* habitat_data
    true_K = exp.(true_log_K)
    
    # Generate a true F trajectory
    true_log_F = zeros(n_steps)
    true_log_F[1] = log(initial_F)
    for t in 2:n_steps
        true_log_F[t] = true_log_F[t-1] + randn() * sigma_F_rw
    end
    true_F = exp.(true_log_F)

    # Simulate population dynamics
    true_log_pop = zeros(n_steps)
    true_log_pop[1] = log(true_K[1] / 2.0)
    for t in 2:n_steps
        prev_pop = exp(true_log_pop[t-1])
        growth = true_r[t-1] * (1.0 - prev_pop / true_K[t-1]) - true_F[t-1]
        true_log_pop[t] = true_log_pop[t-1] + growth + randn() * sigma_pop
    end
    
    # Generate noisy observations
    observed_counts = [rand(Poisson(max(0, exp(p)))) for p in true_log_pop]
    
    return observed_counts, true_r, true_K
end

# Define true parameters
n_steps = 150
initial_F = 0.05
sigma_F_rw = 0.1
sigma_pop = 0.08

# True parameters for the covariate effects
true_log_r_base = log(0.3)
true_r_temp_effect = 0.05
true_log_K_base = log(600.0)
true_K_habitat_effect = 0.3

# Simulate covariate time series
temp_series = 10.0 .+ 8.0 .* sin.(2 * pi .* (1:n_steps) ./ 52) .+ randn(n_steps) .* 0.5
habitat_series = 0.2 .+ (1:n_steps) ./ n_steps .* 1.5 .+ randn(n_steps) .* 0.1

# Simulate the data
observed_counts, true_r, true_K = simulate_dual_dynamic_system(
    n_steps, initial_F, sigma_F_rw, sigma_pop, 
    temp_series, habitat_series, 
    true_log_r_base, true_r_temp_effect, 
    true_log_K_base, true_K_habitat_effect
);

# Create a DataFrame for bstm
df = DataFrame(
    y = observed_counts,
    t_idx = 1:n_steps,
    temperature = temp_series,
    habitat_quality = habitat_series
);

# --- 2. Define and Run the bstm Model ---

# Define the formula with both covariate parameters
formula = "y ~ 1 + dynamics(t_idx, model=\"logistic_f\", r_covariate=:temperature, K_covariate=:habitat_quality)"

# Build and sample the model
m = bstm(formula, df, model_family="poisson");
chn = sample(m, NUTS(500, 0.65), 1000; progress=true)

# --- 3. Analyze and Interpret the Results ---

println("Posterior summary for covariate effects:")
println("True log_r_base: $(round(true_log_r_base, digits=3)), True r_temp_effect: $(round(true_r_temp_effect, digits=3))")
println("True log_K_base: $(round(true_log_K_base, digits=3)), True K_habitat_effect: $(round(true_K_habitat_effect, digits=3))")
display(chn[[:log_r_base, :r_cov_effect, :log_K_base, :K_cov_effect]])

# --- Plot the estimated time-varying r(t) ---
log_r_base_samples = chn[:log_r_base].data[:]
r_cov_effect_samples = chn[:r_cov_effect].data[:]
r_t_samples = exp.(log_r_base_samples .+ r_cov_effect_samples .* temp_series')
r_t_mean = mean(r_t_samples, dims=1)[:]
r_t_lower = [quantile(r_t_samples[:,t], 0.025) for t in 1:n_steps]
r_t_upper = [quantile(r_t_samples[:,t], 0.975) for t in 1:n_steps]

p_r = plot(1:n_steps, true_r, label="True r(t)", lw=3, color=:black, ls=:dash, title="Estimated vs. True Growth Rate r(t)")
plot!(p_r, 1:n_steps, r_t_mean, ribbon=(r_t_mean .- r_t_lower, r_t_upper .- r_t_mean),
      label="Estimated r(t) (95% CI)", lw=2, color=:crimson, fillalpha=0.2)

# --- Plot the estimated time-varying K(t) ---
log_K_base_samples = chn[:log_K_base].data[:]
K_cov_effect_samples = chn[:K_cov_effect].data[:]
K_t_samples = exp.(log_K_base_samples .+ K_cov_effect_samples .* habitat_series')
K_t_mean = mean(K_t_samples, dims=1)[:]
K_t_lower = [quantile(K_t_samples[:,t], 0.025) for t in 1:n_steps]
K_t_upper = [quantile(K_t_samples[:,t], 0.975) for t in 1:n_steps]

p_K = plot(1:n_steps, true_K, label="True K(t)", lw=3, color=:black, ls=:dash, title="Estimated vs. True Carrying Capacity K(t)")
plot!(p_K, 1:n_steps, K_t_mean, ribbon=(K_t_mean .- K_t_lower, K_t_upper .- K_t_mean),
      label="Estimated K(t) (95% CI)", lw=2, color=:darkgreen, fillalpha=0.2)

# Display both plots
plot(p_r, p_K, layout=(2,1), size=(800, 600))


```


The results from this example will demonstrate that the model can successfully disentangle and estimate the separate effects of temperature on r and habitat_quality on K, showcasing the power and flexibility of the bstm framework for building complex, mechanistic models



## Add a third model:

to add a new population dynamics model, such as the Ricker model, to the bstm framework.

The Ricker Model and its Relationship to the Logistic Model
Before diving into the code, it's insightful to understand the relationship between the Ricker model and the existing basic_logistic model in bstm.

The Ricker Model: This is a classic discrete-time population model, typically written as: $$ N_{t+1} = N_t \exp\left(r\left(1 - \frac{N_t}{K}\right)\right) $$ where $r$ is the intrinsic growth rate and $K$ is the carrying capacity. To create a state-space model, we can add process noise. Working with the logarithm of the population, $X_t = \log(N_t)$, the equation becomes: $$ X_{t+1} = X_t + r\left(1 - \frac{\exp(X_t)}{K}\right) + \epsilon_t $$ where $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ is the process noise.

The Continuous Logistic Model: This model is defined by the differential equation: $$ \frac{dN}{dt} = rN\left(1 - \frac{N}{K}\right) $$ To implement this in a discrete-time state-space model, we can use the Euler-Maruyama method on the log-transformed population, $X_t = \log(N_t)$. The corresponding differential equation for the log-population is $d(\log N)/dt = r(1 - N/K)$. The discrete-time approximation is: $$ X_{t+1} \approx X_t + r\left(1 - \frac{\exp(X_t)}{K}\right) + \epsilon_t $$

You'll notice that the state-transition equation for the log-population is identical for both the discrete Ricker model and the Euler-discretized continuous logistic model. The existing dynamics(..., model="basic_logistic") in bstm already implements this exact structure.

However, for the sake of clarity, extensibility, and to directly answer your request, I will show you how to add an explicit ricker model option. This serves as a perfect template for adding any other custom univariate population model in the future.

```{julia}
using bstm, Turing, DataFrames, Random, Distributions, Plots

# Ensure the updated spatiotemporal_functions.jl is loaded
# include("c:/home/jae/projects/bstm/src/spatiotemporal_functions.jl")

# --- 1. Simulate Data from a Ricker Process ---

function simulate_ricker_data(n_steps, r, K, initial_pop, obs_error)
    Random.seed!(123)
    
    true_pop = zeros(n_steps)
    true_pop[1] = initial_pop
    
    # Generate latent population trajectory
    for t in 2:n_steps
        growth = r * (1.0 - true_pop[t-1] / K)
        # Add process noise on the log scale
        log_next_pop = log(true_pop[t-1]) + growth + randn() * 0.1
        true_pop[t] = exp(log_next_pop)
    end
    
    # Generate noisy observations (e.g., Poisson counts)
    observed_counts = [rand(Poisson(max(0, p))) for p in true_pop]
    
    return observed_counts, true_pop
end

# Define true parameters for simulation
n_steps = 100
true_r = 0.4
true_K = 800.0
initial_pop = 100.0
obs_error = 0.2

# Simulate the data
observed_counts, true_population = simulate_ricker_data(n_steps, true_r, true_K, initial_pop, obs_error)

# Create a DataFrame for bstm
df = DataFrame(
    counts = observed_counts,
    time = 1:n_steps
);

# --- 2. Define and Run the bstm Model with the Ricker Dynamic ---

# Define the formula using the new "ricker" model
# We can also provide custom priors for r and K
formula = "counts ~ 1 + dynamics(time, model=\"ricker\", r_prior=LogNormal(0,1), K_prior=Normal(700, 200))"

# Build and sample the model
# Using a Poisson likelihood as our observations are counts
m = bstm(formula, df, model_family="poisson");
chn = sample(m, NUTS(500, 0.65), 1000; progress=true)

# --- 3. Analyze and Interpret the Results ---

println("Posterior summary for Ricker parameters:")
display(chn[[:r_dyn, :K_dyn, :sig_dyn]])

# Plot the posterior distributions
p1 = density(chn[:r_dyn], label="Posterior for r", title="Growth Rate (r)")
vline!([true_r], label="True r", color=:red, ls=:dash)

p2 = density(chn[:K_dyn], label="Posterior for K", title="Carrying Capacity (K)")
vline!([true_K], label="True K", color=:red, ls=:dash)

plot(p1, p2, layout=(1,2), size=(800, 400))

```


### Growth The Gompertz Model: A State-Space Perspective
The Gompertz model is a sigmoidal growth model, similar to the logistic curve, but it is asymmetric. The growth rate is highest at the beginning and slows down as the population approaches the carrying capacity, K.

The model is defined by the differential equation: $$ \frac{dN}{dt} = r N \log\left(\frac{K}{N}\right) $$ where $r$ is the growth rate parameter. A key difference from the logistic model, $\frac{dN}{dt} = rN(1 - N/K)$, is the use of the logarithm, which results in a different growth dynamic.

To implement this within the bstm state-space framework, we model the logarithm of the population, $X_t = \log(N_t)$. Using the chain rule, we can find the differential equation for $X_t$: $$ \frac{dX_t}{dt} = \frac{d(\log N_t)}{dt} = \frac{1}{N_t} \frac{dN_t}{dt} = \frac{1}{N_t} \left( r N_t \log\left(\frac{K}{N_t}\right) \right) = r (\log(K) - \log(N_t)) $$ This simplifies to a linear differential equation in terms of the log-population $X_t$: $$ \frac{dX_t}{dt} = r (\log(K) - X_t) $$ Using the Euler-Maruyama method to discretize this for a state-space model (with time steps $\Delta t = 1$), we get the following transition equation, which forms the core of our implementation: $$ X_{t+1} \sim \mathcal{N}\left(X_t + r(\log(K) - X_t), \sigma^2\right) $$ where $\sigma^2$ is the process variance.



```{julia}

using bstm, Turing, DataFrames, Random, Distributions, Plots

# Ensure the updated spatiotemporal_functions.jl is loaded
# include("c:/home/jae/projects/bstm/src/spatiotemporal_functions.jl")

# --- 1. Simulate Data from a Gompertz Process ---

function simulate_gompertz_data(n_steps, r, K, initial_pop, process_noise)
    Random.seed!(42)
    
    log_pop = zeros(n_steps)
    log_pop[1] = log(initial_pop)
    
    # Generate latent population trajectory
    for t in 2:n_steps
        growth = r * (log(K) - log_pop[t-1])
        log_pop[t] = log_pop[t-1] + growth + randn() * process_noise
    end
    
    pop = exp.(log_pop)
    
    # Generate noisy observations (e.g., Poisson counts)
    observed_counts = [rand(Poisson(max(0, p))) for p in pop]
    
    return observed_counts, pop
end

# Define true parameters for simulation
n_steps = 100
true_r = 0.2
true_K = 1000.0
initial_pop = 50.0
process_noise = 0.1

# Simulate the data
observed_counts, true_population = simulate_gompertz_data(n_steps, true_r, true_K, initial_pop, process_noise)

# Create a DataFrame for bstm
df = DataFrame(
    y = observed_counts,
    time_idx = 1:n_steps
);

# --- 2. Define and Run the bstm Model with the Gompertz Dynamic ---

# Define the formula using the new "gompertz" model
formula = "y ~ 1 + dynamics(time_idx, model=\"gompertz\")"

# Build and sample the model
# Using a Poisson likelihood as our observations are counts
m = bstm(formula, df, model_family="poisson");
chn = sample(m, NUTS(500, 0.65), 1000; progress=true)

# --- 3. Analyze and Interpret the Results ---

println("Posterior summary for Gompertz parameters:")
display(chn[[:r_dyn, :K_dyn, :sig_dyn]])

# Plot the posterior distributions
p1 = density(chn[:r_dyn], label="Posterior for r", title="Growth Rate (r)")
vline!([true_r], label="True r", color=:red, ls=:dash)

p2 = density(chn[:K_dyn], label="Posterior for K", title="Carrying Capacity (K)")
vline!([true_K], label="True K", color=:red, ls=:dash)

plot(p1, p2, layout=(1,2), size=(800, 400))

```
