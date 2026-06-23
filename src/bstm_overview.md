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

* spatial(s_idx; manifold='bym2'): High-level spatial effect dispatch.
* temporal(t_idx; manifold='rw2'): High-level temporal smoothing.
* smooth(x, y; manifold='rff'): 2D spectral smoother using informed RFF parameters.
* eigen(var; rank=k): Low-rank factor analytic terms.

Prior Resolution: The PC-Prior Standard

Stability in high-dimensional models is achieved through PC-Priors (Penalized Complexity). These priors are designed to shrink towards a simpler "base model"—for instance, shrinking a spatial field's variance towards zero or its range towards infinity.

Parameter	Default PC-Prior	Rationale
Sigma (\sigma)	Exponential(1.0)	Shrinks towards zero variance (the base model).
Rho (\rho)	Beta(1, 1)	Flat/Uninformative base for correlation parameters.
Lengthscale	InverseGamma(3, 3)	Prevents over-fitting in continuous kernels.
Kappa (\kappa)	Exponential(1.0)	Controls SPDE smoothness via principled shrinkage.

7. Illustrative Examples: Guided Model Implementation

1. BYM2 Disease Mapping: y ~ 1 + spatial(s_idx; manifold='bym2'). Decomposes risk into structured spatial and unstructured IID noise.
2. AR1 Temporal Forecasting: y ~ 1 + temporal(t_idx; manifold='ar1'). Captures geometric temporal decay.
3. Spatio-Temporal Interaction: y ~ 1 + spatial(s_idx) ⊗ temporal(t_idx). Employs tensor-product logic to manage O(N \times T) complexity via the Type IV interaction template.
4. Spatially Varying Coefficients (SVC): poverty | spatial(s_idx; manifold='icar'). Allows the impact of poverty to vary according to local spatial gradients.
5. Spectral Splines: smooth(lon, lat; manifold='rff'). Approximates a 2D continuous field without the O(N^3) kernel inversion cost.
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

| Keyword     | Example Usage                        | Purpose                                                        |                                                                    |
| :------------| :-------------------------------------| :---------------------------------------------------------------| --------------------------------------------------------------------|
| `spatial`   | `spatial(s_idx, manifold='bym2')`    | Models spatial random effects for discrete areal units.        |                                                                    |
| `temporal`  | `temporal(t_idx, manifold='ar1')`    | Models temporal trends using discrete time steps.              |                                                                    |
| `seasonal`  | `seasonal(u_idx, manifold='cyclic')` | Models periodic effects (e.g., month-of-year).                 |                                                                    |
| `smooth`    | `smooth(x, nbins=20)`                | Creates a non-linear smooth of a continuous covariate `x`.     |                                                                    |
| `svc`       | `svc(x, manifold='iid')`             | Models a spatially-varying coefficient for covariate `x`.      |                                                                    |
| `mixed`     | `mixed(1 \                           | group)`                                                        | Defines a random intercept for each level of the `group` variable. |
| `spacetime` | `spacetime(manifold='IV')`           | Defines a joint spatiotemporal interaction field.              |                                                                    |
| `bias`      | `bias(hurdle=0.1)`                   | Specifies likelihood-level parameters like hurdles or weights. |                                                                    |
| `fixed`     | `x1 + x2`                            | Standard fixed-effect linear predictors.                       |                                                                    |

### Data Transformations

Use the `|>` operator inside a module to transform data on the fly.

*   `smooth(x |> log)`: Applies a log transform to `x`.
*   `smooth(x |> zscore)`: Standardizes `x`.
*   `smooth(x |> unit)`: Scales `x` to `[0, 1]`.

---

## Manifold Reference Table

The `manifold` argument specifies the mathematical structure of the latent field.

### Spatial Manifolds (`spatial()`)

| Manifold | `manifold='...'` | Math (Conceptual) | Use Case |
| :--- | :--- | :--- | :--- |
| **ICAR** | `'icar'`, `'besag'` | $\phi_i \sim \mathcal{N}(\text{mean}(\phi_j \text{ for } j \sim i), \sigma^2/n_i)$ | Strong, localized spatial smoothing for areal data. |
| **BYM2** | `'bym2'` | $\phi_i = \sqrt{\rho}\phi_{str} + \sqrt{1-\rho}\phi_{unstr}$ | Robust default. Separates spatial trend from random noise. |
| **Leroux** | `'leroux'` | $Q = \rho Q_{sp} + (1-\rho)I$ | A proper CAR model that avoids rank-deficiency. |
| **SAR** | `'sar'` | $\phi = \rho W \phi + \epsilon$ | Models spatial "spill-over" effects. |

### Temporal & Seasonal Manifolds (`temporal()`, `seasonal()`)

| Manifold | `manifold='...'` | Math (Conceptual) | Use Case |
| :--- | :--- | :--- | :--- |
| **AR1** | `'ar1'` | $\phi_t = \rho \phi_{t-1} + \epsilon_t$ | Capturing trends with short-term memory. |
| **RW1/RW2** | `'rw1'`, `'rw2'` | $\Delta\phi_t \sim \mathcal{N}(0, \sigma^2)$ or $\Delta^2\phi_t \sim \mathcal{N}(0, \sigma^2)$ | Modeling stochastic level shifts (RW1) or smooth trends (RW2). |
| **Cyclic** | `'cyclic'` | $\phi_t \sim \text{ICAR on a circular graph}$ | Smooth seasonal patterns where Dec is adjacent to Jan. |
| **Harmonic** | `'harmonic'` | $\phi_t = \sum a_k \sin(\omega_k t) + b_k \cos(\omega_k t)$ | Decomposing seasonality into sine/cosine waves. |

### Smooth Manifolds (`smooth()`)

| Manifold | `manifold='...'` | Math (Conceptual) | Use Case |
| :--- | :--- | :--- | :--- |
| **P-Spline** | `'pspline'` | B-spline basis with a random walk penalty on coefficients. | Flexible, general-purpose smoother for 1D covariates. |
| **TPS** | `'tps'` | Thin Plate Spline basis. | Smoothing 2D spatial coordinates. |
| **RFF** | `'rff'` | $\phi(x) = \sqrt{2/M}\cos(Wx+b)$ | Scalable approximation of a Gaussian Process smooth. |

### Spacetime Interaction Manifolds (`spacetime()`)

| Manifold | `manifold='...'` | Description |
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
    "counts ~ 1 + spatial(area_id, manifold='bym2') + temporal(year_id, manifold='ar1')",
    my_data,
    model_family="poisson",
    W=my_adjacency_matrix
)
```

### 2. Gaussian Model with Non-linear Smooth and SVC

Models a continuous outcome with a non-linear effect of `temperature` and allows the effect of `rainfall` to vary over space.

```julia
m = bstm(
    "yield ~ 1 + smooth(temperature, nbins=20) + svc(rainfall, manifold='iid')",
    my_data,
    model_family="gaussian"
)
```

### 3. Binomial Model with Spacetime Interaction

Models prevalence (0 or 1) with a fully structured spatiotemporal interaction term.

```julia
m = bstm(
    "prevalence ~ 1 + spatial(s_idx) + temporal(t_idx) + spacetime(manifold='IV')",
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
m = bstm("y ~ 1 + x + spatial(s_idx, manifold='bym2')", my_data, model_family="poisson")
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
| :-------------| :-------------------------------------| :---------------------------------------------------------------------------| --------------------------------------------------------------------|
| `intercept`  | `intercept()`                        | Includes a global intercept term.                                          |                                                                    |
| `bias`       | `bias(hurdle=0.1)`                   | Specifies likelihood-level parameters like hurdles or observation weights. |                                                                    |
| `spatial`    | `spatial(s_idx, manifold='bym2')`    | Models spatial random effects using discrete areal units.                  |                                                                    |
| `temporal`   | `temporal(t_idx, manifold='ar1')`    | Models temporal trends using discrete time steps.                          |                                                                    |
| `seasonal`   | `seasonal(u_idx, manifold='cyclic')` | Models periodic effects (e.g., month-of-year).                             |                                                                    |
| `smooth`     | `smooth(x, nbins=20)`                | Creates a non-linear smooth of a continuous covariate `x`.                 |                                                                    |
| `svc`        | `svc(x, manifold='iid')`             | Models a spatially-varying coefficient for covariate `x`.                  |                                                                    |
| `mixed`      | `mixed(1                             | group)`                                                                    | Defines a random intercept for each level of the `group` variable. |
| `eigen`      | `eigen(x1, x2, n_factors=1)`         | Creates a latent field from the principal components of `x1` and `x2`.     |                                                                    |
| `spacetime`  | `spacetime(manifold='IV')`           | Defines a joint spatiotemporal interaction field.                          |                                                                    |
| `network`    | `network(s_idx, W=adj)`              | Models effects on a graph defined by adjacency matrix `W`.                 |                                                                    |
| `hyperbolic` | `hyperbolic(s_idx)`                  | Placeholder for hyperbolic geometry models.                                |                                                                    |
| `dynamics`   | `dynamics(var)`                      | Incorporates a mechanistic state-space model.                              |                                                                    |
| `volatility` | `volatility(s_idx)`                  | Models heteroskedastic observation noise that varies over space.           |                                                                    |
| `fixed`      | `x1 + x2`                            | Standard fixed-effect linear predictors.                                   |                                                                    |

### Manifold Options

The `manifold` argument within modules like `spatial`, `temporal`, and `smooth` specifies the underlying mathematical structure.

| Manifold         | `manifold='...'`               | Description                                                                                 |
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
    *   Example: `spacetime(manifold='spatial(s_idx) ⊗ temporal(t_idx)')`
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

