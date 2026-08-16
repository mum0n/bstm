---
title: "The `bstm` Framework: An Architectural Overview"
format: html
---

# The `bstm` Framework: An Architectural Overview

## 1. Introduction

The `bstm` framework provides a composable, formula-based interface for Bayesian spatiotemporal modeling in Julia. It is designed to address the challenge of building complex models by separating the observation likelihood from the specification of the latent process. This decoupling allows for flexible construction of models that can include spatial, temporal, and mechanistic components in an additive and extensible manner. It is a Julia library built on the Turing.jl probabilistic programming framework and many, many other Julia libraries and many more scientists. This work really stands upon their giant shoulders. It provides a high-level, formula-based interface/front-end inspired by R's `brms` and `lme4` to simplify the specification of complex hierarchical models. The framework is designed for composability, allowing users to combine spatial, temporal, and mechanistic components to analyze complex datasets, particularly in fields like ecology and epidemiology. 

`bstm` was designed to pursue my research interests and make my work-life simpler, especially as it often takes more time and effort to post-process data than actually set up the model. It is built on the strength and insights of many people and the strength and composability of Julia and Turing, in particular. It can easily be extended by others for their own purposes and provided as is, warts and all. Full disclosure: I have made heavy use of AI LLMs to restructure and expand the code, especially the crazy regex's and provide coherent descriptions and math, etc. Any errors are, of course, my own. Check your results with simulated data where possible.


## 2. The Formula Interface

The framework uses a formula interface inspired by R's `lme4` and `brms`, but with specific modules for  various classes of components. The model is defined with the observation model on the left-hand side (LHS) with the 'likelihood()' and the process models defined within the modules on the right-hand side (RHS). The `@bstm` macro enables an unquoted formula syntax.

### 2.1. Basic Structure

The general structure of a `bstm` model call is:

```julia
m = @bstm(
    likelihood(outcome_variable, family=poisson, ...) ~ 
    intercept() + 
    fixed(...) + 
    random(...) + 
    other_modules(...),
    data_frame,
    keyword_arguments...
)
```

Any keyword arguments provided after the `data_frame` are passed into the model's configuration. A notable general keyword is `verbose`.

*   **`verbose=false`**: Suppresses the printing of the dynamically generated model code and the results of the automatic prior predictive check that runs at instantiation. This is useful for cleaner output in scripts or notebooks. The default is `true`.


### 2.2. The `likelihood()` 

The `likelihood()` on the LHS specifies the observation model and its parameters.

| Parameter                      | Example Usage          | Description                                                                                                         |
| :-------------------------------| :-----------------------| :--------------------------------------------------------------------------------------------------------------------|
| `family`                       | `family=:poisson`      | Sets the likelihood distribution. See table below for options.                                                      |
| `log_offsets`                  | `log_offsets=pop_log`  | Provides a log-scale offset to the linear predictor ($\eta' = \eta + \text{offset}$). Essential for modeling rates. |
| `weights`                      | `weights=sample_w`     | Applies observation-level weights to the log-likelihood.                                                            |
| `trials`                       | `trials=n_patients`    | Specifies the number of trials for each observation in a Binomial model.                                            |
| `zero_inflated`                | `zero_inflated=true`   | Enables a zero-inflation component for count models.                                                                |
| `volatility`                   | `volatility=true`      | Enables a spatiotemporal stochastic volatility model for the observation noise ($\sigma_y$).                        |
| `censor_lower`, `censor_upper` | `censor_lower=lower_b` | Defines lower and upper bounds for censored data.                                                                   |
| `hurdle`                       | `hurdle=0`             | Implements a hurdle model by truncating the likelihood below the specified threshold.                               |

#### 2.2.1. Likelihood Families

| Family                | `family=...`        | Link Function   | Key Parameters & Priors                                                                                         |
| :----------------------| :--------------------| :----------------| :----------------------------------------------------------------------------------------------------------------|
| **Poisson**           | `:poisson`          | `exp(eta)`      | `rate (λ)`: Determined by `exp(eta)`.                                                                           |
| **Gaussian**          | `:gaussian`         | `identity(eta)` | `mean (μ)`: `eta`, `std. dev. (σ)`: `y_sigma ~ Exponential(1.0)`                                                |
| **Log-Normal**        | `:lognormal`        | `identity(eta)` | `log-mean (μ)`: `eta`, `log-std. dev. (σ)`: `y_sigma ~ Exponential(1.0)`                                        |
| **Negative Binomial** | `:negbin`           | `exp(eta)`      | `rate (μ)`: `exp(eta)`, `dispersion (r)`: `r_nb ~ Exponential(1.0)`                                             |
| **Binomial**          | `:binomial`         | `logistic(eta)` | `trials (n)`: From `likelihood(trials=...)`, `probability (p)`: `logistic(eta)`                                 |
| **Gamma**             | `:gamma`            | `exp(eta)`      | `shape (α)`: `extra_params ~ Exponential(1.0)`, `scale (θ)`: `exp(eta)/α`                                       |
| **Beta**              | `:beta`             | `logistic(eta)` | `mean (μ)`: `logistic(eta)`, `precision (φ)`: `extra_params ~ Exponential(1.0)`                                 |
| **Student's T**       | `:student_t`        | `identity(eta)` | `location (μ)`: `eta`, `scale (σ)`: `y_sigma ~ Exponential(1.0)`, `d.f. (ν)`: `extra_params ~ Exponential(1.0)` |
| **Exponential**       | `:exponential`      | `exp(eta)`      | `rate (λ)`: `1 / exp(eta)`.                                                                                     |
| **Inverse Gaussian**  | `:inverse_gaussian` | `exp(eta)`      | `mean (μ)`: `exp(eta)`, `shape (λ)`: `extra_params ~ Exponential(1.0)`                                          |
| **Half-Normal**       | `:half_normal`      | `identity(eta)` | `std. dev. (σ)`: `y_sigma ~ Exponential(1.0)`. Mean is implicitly 0.                                            |
| **Half-Student's T**  | `:half_student_t`   | `identity(eta)` | `scale (σ)`: `y_sigma ~ Exponential(1.0)`, `d.f. (ν)`: `extra_params ~ Exponential(1.0)`                        |
| **Laplace**           | `:laplace`          | `identity(eta)` | `location (μ)`: `eta`, `scale (b)`: `y_sigma ~ Exponential(1.0)`                                                |
| **Pareto**            | `:pareto`           | `exp(eta)`      | `shape (α)`: `extra_params ~ Exponential(1.0)`, `scale (θ)` from mean.                                          |
| **Dirichlet**         | `:dirichlet`        | `exp(eta)`      | `concentration (α)`: `exp.(eta)`. For multivariate compositional data, specified as `likelihood(y1+y2+...)`.    |


### 2.3 Modules

The `bstm` formula interface is built around a series of modules, which are special function-like terms that define the components of the latent model.

| Module        | Purpose                                   | Key Parameters                | Example Usage                                    |         |
| :--------------| :------------------------------------------| :------------------------------| :-------------------------------------------------| ---------|
| `intercept()` | Controls the global intercept.            | `prior`                       | `intercept(prior=Normal(0, 10))`                 |         |
| `fixed()`     | Defines fixed-effect covariates.          | `prior`, `contrast`           | `fixed(urban, contrast=:effects)`                |         |
| `random()`    | The main module for all random effects.   | `structure`, `model`, `prior` | `random(s_idx, model=:bym2)`                     |         |
| `mixed()`     | Defines hierarchical random effects.      | `model`                       | `mixed(1 + cov                                   | group)` |
| `dynamics()`  | Embeds mechanistic, process-based models. | `model`, `priors...`          | `dynamics(t, model=:logistic, K=Normal(100,10))` |         |
| `eigen()`     | Performs Bayesian PCA.                    | `n_factors`                   | `eigen(y1, y2, y3, n_factors=1)`                 |         |
| `nested()`    | Defines a multi-fidelity sub-model.       | `formula`, `data_source`      | `nested(proxy, formula=..., data_source=...)`    |         |
| `sciml()`     | Integrates a SciML differential equation. | `model_func`, `solver`        | `sciml(t, model_func=f, ...)`                    |         |
| `custom()`    | Injects user-defined Turing code.         | `code_fragment`               | `custom(code_fragment="...")`                    |         |


### `intercept()` and `fixed()`

These modules handle standard regression terms.

*   **`intercept()`**: Explicitly includes a global intercept. While an intercept is included by default, this module allows you to specify a custom prior for it. Use `intercept(false)` or `-1` in the formula to remove it.
*   **`fixed()`**: Marks a variable as a fixed effect. This is primarily used to assign a specific prior or to specify contrast coding for categorical variables (e.g., `contrast=:effects` for sum-to-zero contrasts). Bare terms in the formula (e.g., `... + cov1`) are treated as `fixed(cov1)` with default priors.

### `random()`

This is the primary module for specifying structured and unstructured random effects. Its behavior is determined by the `structure` and `model` arguments.

#### `structure=:spatial`

For modeling effects that vary over discrete areal units or continuous space.

| `model` | Description | Key Parameters |
| :--- | :--- | :--- |
| `:bym2` | The standard for disease mapping, mixing structured and unstructured effects. | `rho`, `sigma` |
| `:icar`, `:besag` | Intrinsic CAR models for strong spatial smoothing. | `sigma` |
| `:leroux` | A proper CAR model that mixes spatial and IID effects. | `rho`, `sigma` |
| `:sar` | Simultaneous Autoregressive model for spatial spill-over. | `rho`, `sigma` |
| `:gp` | A full Gaussian Process for continuous coordinates. | `kernel`, `lengthscale`, `sigma` |
| `:rff` | A scalable GP approximation using Random Fourier Features. | `n_features`, `lengthscale`, `sigma` |
| `:spde` | A scalable GP based on a Stochastic Partial Differential Equation. | `kappa`, `sigma` |
| `:localadaptive` | A non-stationary model with cluster-specific means. | `n_clusters`, `rho`, `sigma` |

**Usage**:

```julia
# Areal data model (requires W matrix)
random(area_id, structure=:spatial, model=:bym2)

# Continuous data model
random(lon, lat, structure=:spatial, model=:gp, kernel="matern32")
```

### 2.4 Illustrative Examples

1.  **BYM2 Disease Mapping:**
    `@bstm(likelihood(y, family=poisson) ~ intercept() + random(s_idx, model=:bym2), data, W=W)`
    Decomposes risk into structured spatial and unstructured IID noise.

2.  **AR1 Temporal Forecasting:**
    `@bstm(likelihood(y) ~ intercept() + random(t_idx, model=:ar1), data)`
    Captures geometric temporal decay.

3.  **Spatio-Temporal Interaction:**
    `@bstm(likelihood(y) ~ intercept() + random(s_idx, model=:besag) ⊗ random(t_idx, model=:ar1), data, W=W)`
    Employs the Kronecker product to create a fully structured spatiotemporal interaction field.

4.  **Spatially Varying Coefficients (SVC) and Curves:**
    `@bstm(likelihood(y) ~ intercept() + (poverty |> random(s_idx, model=:icar)), data, W=W)`
    Allows the impact of `poverty` to vary according to local spatial gradients.
    `@bstm(likelihood(y) ~ intercept() + (random(s_idx, model=:icar) |> random(t_idx, model=:pspline)), data, W=W)`
    Models a temporal trend that varies smoothly across space.

## 3. The Algebra of Components: Composition and State-Space Models

The `bstm` domain-specific language operates through a recursive parser that allows for the algebraic composition of different model components.

### 3.1. Algebraic Operators

1.  **Kronecker Product (⊗)**: Used for creating inseparable interaction effects, such as the Knorr-Held Type IV model (`random(...) ⊗ random(...)`). This builds a joint precision matrix $Q_{st} = Q_t \otimes Q_s$, enabling the representation of space-time interactions where every spatial location has a unique, correlated temporal trend.
2.  **Composition (∘)**: Represents the functional composition of two components, where one component modulates the parameters of another. This is a powerful tool for creating non-stationary models, such as in Log-Gaussian Cox Process (LGCP) models.
3.  **Pipe (`|>`):** The pipe operator handles data normalization and state-space evolution.

*   **Transformations**: Objects like `ZScoreComponent` or `LogComponent` act as wrappers that normalize inputs before they enter the latent process.
*   **State-Space Evolution**: The pipe operator defines a state-space model where one component evolves over the structure of another. This supports both discrete-time dynamics (e.g., `random(s_idx, model=icar) |> random(t_idx, model=ar1)`) and the creation of spatially-varying curves (e.g., `random(s_idx, model=icar) |> random(t_idx, model=pspline)`), where the coefficients of the temporal basis functions are modeled as spatial fields.

### 3.2. Formula Operators in Practice

The formula parser's support for algebraic operators allows for the composition of components to create more complex model structures.

*   **Kronecker Product (`⊗`)**: Creates an interaction term between two components. This is most commonly used for spatiotemporal interactions.
    *   **Formula**: `random(s_idx, model=icar) ⊗ random(year, model=ar1)`
    *   **Interpretation**: The parser identifies this as a `ComposedComponent` with a `:kronecker_product` operator. The model configuration engine then sets up a Knorr-Held Type IV interaction, where the spatial field (ICAR) evolves over time according to a temporal process (AR1).

*   **Pipe (`|>`)**: Defines a state-space model or a spatially-varying coefficient (SVC) model.
    *   **Formula**: `poverty |> random(s_idx, model=icar)`
    *   **Interpretation**: The parser creates an `SVCComponent`. In the model, the effect of the `poverty` covariate is no longer a single global coefficient but is instead a spatially-varying field structured by the `icar` component. This allows the impact of poverty to differ across regions.

*   **Composition (`∘`)**: Used for specialized models where one component modulates the parameters of another, such as in Log-Gaussian Cox Process (LGCP) models for point-pattern data.
    *   **Formula**: `pointprocess(model=:lgcp) ∘ random(s_idx, model=icar)`
    *   **Interpretation**: The parser recognizes this as a point process model. The `pointprocess` module modifies the likelihood contribution, while the `random` component defines the latent intensity field. The composition operator links the two, indicating that the `icar` field represents the log-intensity of the point process.

## 4. Core Components: Components and Priors

The `bstm` framework includes a registry of components that range from discrete graph-based models to continuous spectral approximations.

### 4.1. The Discrete Registry: Gaussian Markov Random Fields (GMRF)

For discrete domains, `bstm` implements GMRF structures where dependency is defined by a precision matrix Q.

| Component     | Theoretical Assumption                                  | Structural Rationale                                                   |
| :--------------| :--------------------------------------------------------| :-----------------------------------------------------------------------|
| IID           | $\epsilon \sim N(0, \sigma^2 I)$                        | Unstructured exchangeability; base model for PC-shrinkage.             |
| ICAR          | Intrinsic CAR; $Q_{ij} = -1$ for neighbors.             | Pure local smoothing; identifies spatial gradients.                    |
| Besag         | Standard CAR model.                                     | Global and local spatial dependency via fixed precision.               |
| BYM2          | Scaled Besag + IID component.                           | Explicit variance partitioning ($\rho$) for better identifiability.    |
| Leroux        | Convex combination of I and $Q_{ICAR}$.                 | Bridges IID and ICAR structures through a mixing parameter.            |
| SAR           | $(I - \rho W)y = \epsilon$.                             | Simultaneous modeling of response autocorrelation.                     |
| RW1 / RW2     | Random Walk (1st/2nd order).                            | Temporal continuity and smoothing of non-stationary trends.            |
| AR1           | $\mu_t = \rho \mu_{t-1} + \epsilon_t$.                  | Stationary temporal process with geometric decay.                      |
| LocalAdaptive | Leroux + Cluster Means                                  | Models localized spatial clusters with cluster-specific means.         |
| DAG           | $y_i = \rho \sum_{j \in pa(i)} W_{ij} y_j + \epsilon_i$ | Models non-reciprocal dependencies for efficient forward substitution. |

### 4.2. Continuous, Spectral, and Advanced Components

To address the $O(N^3)$ computational cost of kernel-based Gaussian Processes, the framework utilizes spectral projections and sparse approximations.

| Component                               | `model=...`         | Description                                                                                                                                     |
| :----------------------------------------| :--------------------| :------------------------------------------------------------------------------------------------------------------------------------------------|
| **Random Fourier Features (RFF)**       | `:rff`              | Maps input coordinates `x` into a randomized feature space to approximate the kernel, defined as $z(x) = \sqrt{2/M} \cos(Wx + b)$.              |
| **SPDE (Stochastic Partial Diff. Eq.)** | `:spde`             | Represents the field as a solution to $(\kappa^2 - \Delta)^{\alpha/2} u = \mathcal{W}$, linking continuous Matérn processes to a discrete mesh. |
| **Nystrom / FITC**                      | `:nystrom`, `:fitc` | Employs low-rank approximations using a set of `n_inducing` points to represent the global field.                                               |
| **NetworkFlow**                         | `:networkflow`      | Captures directed dependencies across an adjacency matrix with `:upstream` or `:downstream` dispatch options.                                   |
| **Wavelet**                             | `:wavelet`          | Provides a multi-resolution analysis of a spatial or temporal field, decomposing it into components at different scales.                        |
| **LGCP (Log-Gaussian Cox Process)**     | `:lgcp`             | Models point patterns where the log-intensity is a Gaussian Process. Used with the `pointprocess()` module.                                     |

### 4.3. Priors and Identifiability

Stability in high-dimensional models is achieved through a principled approach to prior specification. The `bstm` framework provides three built-in prior schemes and allows for user-defined overrides.

#### Prior Schemes

1.  **Penalized Complexity Priors (`:pcpriors`)**: This is the default scheme. PC priors are designed to shrink complex models towards simpler "base models" unless there is strong evidence in the data to the contrary. For example, the prior on a variance parameter (`sigma`) shrinks towards zero, and the prior on a correlation parameter (`rho`) shrinks towards zero (no correlation). This is the recommended scheme for most applications as it helps prevent overfitting and improves model identifiability.

    The core idea of PC priors is to translate a user's belief about the scale of a parameter into a prior distribution. This is done by specifying an upper bound `U` for a parameter and the probability `alpha` that the parameter will exceed this bound. The framework then calculates the necessary hyperparameters for the prior distribution (e.g., the rate `λ` for an `Exponential` prior) that satisfy this constraint.

    The general form of the constraint is: `P(param > U) = alpha`

    For a standard deviation parameter `sigma`, which is given an `Exponential(λ)` prior, the relationship is:
    `P(sigma > U) = exp(-λ * U) = alpha`
    From this, the framework solves for the rate parameter:
    `λ = -log(alpha) / U`

    This allows for a more intuitive and principled way to set priors than choosing arbitrary hyperparameter values.

2.  **Informative Priors (`:informative`)**: This scheme uses priors that are still weakly informative but less aggressive in their shrinkage than PC priors. For example, the prior on `rho` is a `Beta(2, 2)`, which is centered at 0.5, reflecting a belief that some correlation is more likely than none. This can be useful when you have prior knowledge that an effect is likely present.

3.  **Uninformative Priors (`:uninformative`)**: This scheme uses very wide, flat priors (e.g., `Normal(0, 1e6)` for `sigma`, `Uniform(0, 1)` for `rho`). While sometimes used to express ignorance, these priors are generally **not recommended** for complex hierarchical models, as they can lead to poor convergence and unidentifiable parameters.

#### Prior Comparison Table

| Parameter            | PC Prior (Default)                                                                                                             | Informative Prior    | Uninformative Prior        | Rationale                                                                                                                               |
| :---------------------| :-------------------------------------------------------------------------------------------------------------------------------| :---------------------| :---------------------------| :----------------------------------------------------------------------------------------------------------------------------------------|
| **Sigma** ($\sigma$) | `Exponential(λ)` where `λ = -log(α)/U` from `P(σ > U) = α`. A typical default might be `(U=1, α=0.05)`.                        | `Exponential(0.5)`   | `Normal(0, 1e6)`           | Controls the marginal standard deviation of a latent field. PC prior shrinks towards zero variance unless data supports a larger scale. |
| **Rho** ($\rho$)     | Transformed `Exponential(λ)` where `λ = log(α)/log(1-U)` from `P(ρ > U) = α`. A typical default might be `(U=0.5, α=0.05)`.    | `Beta(2, 2)`         | `Uniform(0, 1)`            | Controls spatial/temporal correlation. PC prior shrinks towards 0 (no correlation).                                                     |
| **Lengthscale**      | Transformed `Exponential(λ)` where `λ = -U*log(α)` from `P(lengthscale < U) = α`. A typical default might be `(U=10, α=0.05)`. | `InverseGamma(5, 5)` | `InverseGamma(0.01, 0.01)` | Controls the range of correlation in continuous GP models. PC prior prevents overfitting by shrinking towards large lengthscales.       |
| **Kappa** ($\kappa$) | `Exponential(λ)` derived from a quantile constraint.                                                                           | `Exponential(0.1)`   | `Exponential(10.0)`        | Controls the smoothness of an SPDE/Matérn field. PC prior shrinks towards a smoother field.                                             |
| **Amplitude**        | `Normal(0, 1)`                                                                                                                 | `Normal(0, 0.5)`     | `Normal(0, 100)`           | Controls the amplitude of harmonic (seasonal) components.                                                                               |
| **Phase**            | `Beta(1, 1)`                                                                                                                   | `Beta(2, 2)`         | `Uniform(0, 1)`            | Controls the phase shift of harmonic components.                                                                                        |

#### Setting Priors in a Model

You can control prior specification at three levels of precedence:

1.  **Local Override (Highest Precedence)**: Specify a prior directly within a module call. This will always override any global settings.
    This can be done by passing a pre-defined `Distribution` object or by passing a `Tuple` representing a PC prior quantile constraint.

    ```julia
    # Local Override with a pre-defined Distribution. Note the use of `sigma=...`
    m = @bstm( # Spatial effect with custom sigma prior
        likelihood(y) ~ intercept() + random(s_idx, model=:bym2, sigma=Exponential(0.1)),
        data, W=W
    );

    # Local Override with a PC prior quantile constraint
    # This sets P(sigma > 0.5) = 0.01 for this specific spatial component's sigma.
    m = @bstm( # Spatial effect with PC prior for sigma
        likelihood(y) ~ intercept() + random(s_idx, model=:bym2, sigma=(0.5, 0.01)),
        data, W=W
    );

    # Local Override for a correlation parameter 'rho' in an AR1 model.
    # This sets P(rho > 0.8) = 0.05, shrinking it towards zero (no correlation).
    m = @bstm( # Temporal effect with PC prior for rho
        likelihood(y) ~ intercept() + random(t_idx, model=:ar1, rho=(0.8, 0.05)),
        data
    );

    # Local Override for a 'lengthscale' in a GP model.
    # This sets P(lengthscale < 10.0) = 0.05, shrinking it towards larger values.
    m = @bstm( # Smooth effect with PC prior for lengthscale
        likelihood(y) ~ intercept() + random(x, model=:gp, lengthscale=(10.0, 0.05, :lower)),
        data
    );
    ```

## 5. Architectures

### 5.1. Univariate Architecture

The default kernel for single-outcome processes, as described in the preceding sections.

### 5.2. Multivariate Architecture

The `MultivariateArchitecture` is triggered when multiple outcomes are specified on the LHS of the formula (e.g., `likelihood(y1 + y2) ~ ...`). It is designed to jointly model these outcomes, allowing the model to "borrow strength" across related processes and estimate the correlation structure between them.

#### Key Mechanisms:

*   **Outcome-Specific Parameters**: Each component (e.g., `random(structure=:spatial)`) generates a separate latent field for each outcome. This means hyperparameters like `sigma_spatial` or `rho_temporal` are estimated independently for each response variable, providing maximum flexibility.
*   **LKJ Correlation Prior**: The core of the multivariate coupling is the `LKJCholesky` prior on a correlation matrix `L_corr`. The final latent effects are constructed by multiplying the matrix of independent latent fields by the Cholesky factor of this correlation matrix: `eta_final = eta_latent * L_corr.L`. This induces a shared correlation structure across all outcomes for a given component, ensuring that the model captures shared patterns while allowing for outcome-specific variances.
*   **Householder Reflection (Spectral Orientation)**: For advanced use cases, the framework can apply a Householder reflection (`H = I - 2vv'`) to the latent fields. This allows the outcomes to rotate in the latent space, which is useful for aligning signals in models with complex dependencies like transport or advection-diffusion, where the direction of correlation is as important as its magnitude.

### 5.3. Multifidelity Architecture

The `MultifidelityArchitecture` is designed for data fusion, integrating high-volume, low-cost proxy data with sparse, high-quality observations. It is typically invoked using the `nested()` module.

#### Key Mechanisms:

*   **Hierarchical Latent Fields**: The architecture establishes a hierarchy of latent processes. A common setup involves:
    1.  **High-Fidelity (Target)**: The primary outcome variable (`y_hq`).
    2.  **Low-Fidelity (Proxy)**: A secondary, related variable (`y_lq`) with more abundant data.
*   **Nested Supervision**: The `nested()` module defines a complete sub-model for the low-fidelity data. The latent field from this sub-model (`eta_sub`) is then used as a calibrated predictor in the main model for the high-fidelity data.
*   **Calibration Parameters**: The link between the fidelities is modeled with calibration parameters, typically a bias and a scaling factor (`rho`), which are estimated within the model:
    `eta_main = ... + rho_nested * eta_sub`
    The prior on `rho_nested` is often centered around 1.0, assuming the proxy is a reasonably good, if biased, predictor of the main process. This allows the main model to learn from the structural patterns in the low-fidelity data while correcting for systematic bias and scale differences.

## 6. Advanced Topics

### 6.1. Spatial Partitioning

For discrete spatial models (GMRFs), the continuous spatial domain must be discretized into "Areal Units" (AUs). The `assign_spatial_units` function provides several methods for this, balancing geometric compactness with statistical information density.

| Method     | Description                         | Justification                                                                                                                    |
| :-----------| :------------------------------------| :---------------------------------------------------------------------------------------------------------------------------------|
| `:cvt`     | **Centroidal Voronoi Tessellation** | Iteratively minimizes variance to create geometrically regular cells.                                                            |
| `:kvt`     | **K-Means Voronoi Tessellation**    | Uses K-Means to create units with a balanced number of observations.                                                             |
| `:avt`     | **Agglomerative Voronoi**           | A bottom-up approach that merges small units to prevent data starvation.                                                         |
| `:bvt`     | **Binary Vector Tree**              | Employs recursive partitioning along the axis of maximum variance to efficiently handle large datasets and balance point counts. |
| `:qvt`     | **Quadrant Voronoi Tessellation**   | A quadtree-like method that recursively splits regions into four quadrants, adapting to multi-scale spatial clusters.            |
| `:hvt`     | **Hierarchical Voronoi**            | Combines K-Means seeding with geometric refinement for stable, well-behaved polygons.                                            |
| `:lattice` | **Regular Grid**                    | Simple, fast discretization into uniform squares. Assumes stationarity.                                                          |

When a `geom_hull` is provided, the function performs a spatial intersection ($P_{clipped} = P_{tessellated} \cap H_{hull}$) to ensure that generated units do not extend into invalid areas (e.g., water bodies). Connectivity between units is determined by `LibGEOS.touches`, and the resulting graph is used to construct the Graph Laplacian $Q = D - W$ for GMRF models.

### 6.1.1. Partitioning Control Parameters

The behavior of the `assign_spatial_units` function can be fine-tuned with the following parameters:

*   **`target_units`**: The desired number of areal units.
*   **`target_cv`**: The target coefficient of variation for the number of data points per areal unit, used to balance unit sizes.
*   **`min_total_arealunits`, `max_total_arealunits`**: Hard constraints on the total number of areal units created.
*   **`min_points`, `max_points`**: Ensures each spatial unit contains a number of data points within this range.
*   **`min_time_slices`**: Ensures each areal unit has a minimum number of unique time observations.
*   **`min_area`, `max_area`**: Constraints on the geographic area of each areal unit.
*   **`tolerance`**: Defines the convergence criteria for iterative methods like `:cvt` and `:hvt`.
*   **`buffer_dist`**: Used in methods like `:hvt` to define a buffer zone for identifying neighbors.

### 6.1.2. Partitioning Algorithms

*   **Centroidal Voronoi Tessellation (`:cvt`)**: Uses Lloyd's algorithm to create a regular, "honeycomb" mesh where each unit's centroid is the geometric center of its Voronoi cell. It is ideal for achieving uniform spatial coverage.

*   **K-Means Voronoi Tessellation (`:kvt`)**: A data-driven approach where centroids are the arithmetic mean of the observations within each unit. This creates smaller units in high-density areas, naturally preventing data starvation.

*   **Binary Vector Tree (`:bvt`)**: A high-speed hierarchical method that recursively splits the domain along the axis of maximum variance. It is the fastest approach for massive datasets and excels at creating units with balanced point counts.

*   **Quadrant Voronoi Tessellation (`:qvt`)**: A quadtree-like method that recursively splits regions into four quadrants. It is excellent at adapting its resolution to capture multi-scale spatial clusters.

*   **Agglomerative Voronoi Tessellation (`:avt`)**: A bottom-up approach that starts with an over-partitioned grid and iteratively merges the smallest or sparsest units. This is the most robust method for preventing "data-starved" units, which can cause instability in Bayesian samplers.

### 6.2. Mechanistic Models with `dynamics()`

The `dynamics()` module provides a powerful interface for embedding process-based, mechanistic models directly into the spatiotemporal framework. Unlike statistical models like `AR1` or `RW2` which describe correlation, `dynamics()` models describe the *evolution* of a latent field from one time step to the next based on a predefined equation.

This is accomplished by defining a latent spatiotemporal field, `dyn_field[space, time]`, where the state at time `t` is a function of the state at time `t-1`. For example, a simple advection model implements the state transition:

`dyn_field[:, t] ~ MvNormal(dyn_field[:, t-1] - velocity * L * dyn_field[:, t-1], noise)`

where `L` is the graph Laplacian. This allows the model to learn physical parameters like `velocity` within a fully Bayesian context, similar to the hierarchical dynamic models described by **Wikle (2003)**.

#### Example: A Mechanistic Logistic Growth Model

The `dynamics()` module can be used to embed a logistic growth model directly into the `bstm` formula. In this example, we model population counts where the underlying population dynamics follow a logistic growth curve. The model will estimate the intrinsic growth rate `r` and the carrying capacity `K`.

```julia
# Define a model where the population 'y' follows logistic growth over 'time'.
# The 'r' and 'K' parameters are given priors directly in the call.
m = @bstm(
    likelihood(y, family=poisson) ~
        intercept() +
        dynamics(time, model=:logistic, r=LogNormal(0, 0.5), K=LogNormal(log(100.0), 0.5)),
    population_data
)

# The model will now estimate the posterior distributions for 'r' and 'K'.
```

### 6.3. Multi-fidelity and Nested Models: 'nested()'

The `nested()` module is a "supervisor" component for multi-fidelity modeling. It allows you to define a complete sub-model that is fit to a separate (often larger, lower-quality) dataset. The latent effect from this sub-model is then incorporated as a calibrated predictor into the main model, allowing the main model to "learn" from the proxy data. The `nested()` module accepts a full formula string, including a `likelihood()` block, which enables the specification of independent likelihoods for each fidelity level.

```julia
@bstm(
    likelihood(y_hq) ~ intercept() + random(s_idx, model=icar) + 
        nested(proxy_signal, 
               formula="likelihood(y_lq, family=poisson) ~ intercept() + random(x, model=pspline)", 
               data_source=:low_quality_data),
    high_quality_data,
    low_quality_data = df_low_quality
)
```


### 6.4. Handling Censored Covariates via Joint Modeling: `nested()`

A censored covariate is a predictor variable for which the true value is not always known, but is instead confined to an interval (e.g., $x_{true} > c$). The statistically robust approach to this "errors-in-variables" problem is to treat the censored covariate as a latent variable and model it jointly with the primary outcome.

The `bstm` framework facilitates this through the `nested()` module, which allows for the construction of a joint model in a single step. This approach simultaneously estimates the model for the censored covariate and the main outcome model, correctly propagating all sources of uncertainty. The `nested()` module accepts a full formula string, including a `likelihood()` block, which enables the specification of independent likelihoods for each fidelity level.

In this setup, the `nested()` module defines a complete sub-model for the censored covariate. This sub-model has its own `likelihood()` block where the censoring bounds (`censor_lower`, `censor_upper`) are specified. The latent process estimated by this sub-model is then automatically incorporated as a predictor in the main model's linear predictor.

**Example: Using `nested()` for a Censored Covariate**

```julia
# Assume 'x_censored' is the covariate with censoring, and 'x_L' and 'x_U' are columns
# in the data indicating the censoring bounds. 'z1' is another fully observed predictor.

# The main model for 'y' includes a `nested()` term named `x_latent_process`.
# This term defines a sub-model where 'x_censored' is the outcome.
# The sub-model's `likelihood()` handles the censoring of 'x_censored' using `censor_lower` and `censor_upper`.
# The latent effect from this sub-model is then automatically added as a predictor to the main model.

m = @bstm(
    likelihood(y, family=poisson) ~ intercept() + z1 +
        nested(x_latent_process,
            formula="likelihood(x_censored, family=gaussian, censor_lower=x_L, censor_upper=x_U) ~ intercept() + z1"
        ),
    my_data
)

# Sample the joint model to estimate all parameters simultaneously.
joint_chain = sample(m, NUTS(), 1000)
```

### 6.5. Bayesian Factor Analysis with `eigen()`

The `eigen()` module implements a Bayesian Principal Component Analysis (PCA) to perform dimensionality reduction on a set of multivariate outcomes. It decomposes the input variables into a smaller set of orthogonal latent factors. The framework uses a Householder transformation to construct the orthonormal loadings matrix, ensuring numerical stability and efficient sampling.

#### `eigen()` Module Reference

| Keyword / Parameter | Example Usage              | Data Type      | Default            | Meaning & Assumptions                                                                                                                                                               |
| :--------------------| :---------------------------| :---------------| :-------------------| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `eigen()`           | `eigen(y1, y2, y3; ...)`   | Module         | N/A                | Defines a Bayesian PCA factor model. The variables listed (e.g., `y1, y2, y3`) are the multivariate outcomes to be decomposed.                                                      |
| `n_factors`         | `n_factors=1`              | `Int`          | `1`                | The number of latent factors (principal components) to extract. This determines the dimensionality of the reduced latent space.                                                     |
| `pca_sd`            | `pca_sd=Exponential(0.5)`  | `Distribution` | `Exponential(1.0)` | The prior for the standard deviations of the principal components (latent factors). These are the "eigenvalues" of the system, controlling the variance explained by each factor.   |
| `pdef_sd`           | `pdef_sd=Exponential(0.5)` | `Distribution` | `Exponential(1.0)` | The prior for the standard deviation of the residual (uniqueness) noise. This captures the variance in each observed variable that is *not* explained by the shared latent factors. |


## 7. Inference and Post-Processing

### 7.1. Samplers, Initialization, and Optimization

The `bstm` framework leverages `Turing.jl`'s flexible sampling infrastructure. The choice of sampler is critical for efficient and accurate posterior exploration.

#### Sampler Selection with `get_optimal_sampler`

The `get_optimal_sampler` utility constructs an efficient composite `Gibbs` sampler by assigning specialized MCMC algorithms to blocks of parameters based on their prior distributions. This block-updating strategy improves sampling efficiency and convergence.

Its logic is as follows:
1.  **Parameter Introspection**: It examines the model's `VarInfo` to identify all parameters and their prior distributions.
2.  **Parameter Categorization**: It classifies parameters into four groups based on their prior's support and type:
    *   **Discrete**: Parameters with discrete priors (e.g., `Categorical`, `Poisson`).
    *   **Gaussian**: Continuous parameters with `Normal` or `MvNormal` priors.
    *   **Bounded**: Continuous parameters with one or two-sided bounds (e.g., from `Uniform`, `Beta`, `Exponential`, `InverseGamma` priors).
    *   **Other Continuous**: All remaining continuous parameters (typically unbounded and non-Gaussian).
3.  **Composite Sampler Construction**: It builds a `Gibbs` sampler that uses the optimal algorithm for each group:
    *   `PG` (Particle Gibbs) is assigned to **discrete** parameters.
    *   `ESS` (Elliptical Slice Sampler) is assigned to **Gaussian** parameters.
    *   `Slice` is assigned to **bounded** parameters.
    *   `NUTS` (No-U-Turn Sampler) is assigned to all **other continuous** parameters.


The following table summarizes the available samplers:

| Sampler   | Type           | Key Characteristic                                     | Best Use Case                                                                                        |
| :----------| :---------------| :-------------------------------------------------------| :-----------------------------------------------------------------------------------------------------|
| **NUTS**  | Gradient-Based | Adaptively tunes step size and number of steps.        | The state-of-the-art, general-purpose sampler for models with continuous, differentiable parameters. |
| **HMC**   | Gradient-Based | Requires manual tuning of leapfrog steps.              | A powerful alternative to `NUTS` that can be very efficient but may require expert tuning.           |
| **ESS**   | Gradient-Free  | Designed specifically for models with Gaussian priors. | Highly efficient for latent Gaussian models (e.g., CAR, GP models).                                  |
| **Slice** | Gradient-Free  | Adapts its step size to explore the posterior slice.   | A robust, general-purpose gradient-free sampler, useful when gradient-based methods fail.            |
| **MH**    | Gradient-Free  | Proposes moves from a simple proposal distribution.    | A universal sampler for non-differentiable models, but often inefficient in high dimensions.         |
| **PG**    | Particle-Based | Used for discrete parameters within a `Gibbs` sampler. | Automatically employed by `get_optimal_sampler` for any discrete random variables.                   |

#### Initial Values with `get_inits`

Good initial values are crucial for MCMC convergence. The `get_inits` function provides a robust mechanism for their generation:
1.  **Heuristic Initialization**: It draws a number of samples from the model's `Prior()` distribution.
2.  **Parameter Averaging**: It computes the median or mean of these prior samples to create a plausible starting point for each parameter. Heuristics are applied to ensure values are within valid bounds (e.g., `sigma > 0`).
3.  **MAP Refinement**: Optionally (`refine="map"`), it uses this heuristic starting point to run a fast optimization routine (`MAP()`) to find a mode of the posterior, providing a high-density starting location for the MCMC chains.

#### Optimization-Based Inference

For rapid point estimates, `bstm` models can be used with optimization instead of sampling:

*   **Maximum Likelihood (MLE):** `optimize(m, MLE())`
*   **Maximum A-Posteriori (MAP):** `optimize(m, MAP())`
*   **Variational Inference (VI):** `vi(m, ADVI(10, 1000))`

### 7.2. Interpreting Results

The `model_results_comprehensive` function is the primary tool for post-processing. It takes a fitted model and an MCMC chain and returns a `NamedTuple` containing:

*   Posterior summaries (mean, median, CI) of all latent fields (spatial, temporal, etc.).
*   Performance metrics (RMSE, R-squared, WAIC).
*   MCMC diagnostics (R-hat, ESS).
*   A collection of standard plots (e.g., posterior predictive checks, spatial maps, temporal trends).

The `model_results_plots` function can be used to display all generated plots.

### 7.3. Prediction

The `predict()` function projects a fitted model onto a new data grid to generate out-of-sample predictions. It correctly handles the projection of all component types, including re-computing basis matrices for smooth terms on the new data.

```julia
preds = predict(model, chain, new_data_frame)
```

### 7.4. Cross-Validation with `bstm_cv_orchestrator`

Assessing the predictive performance of spatiotemporal models requires careful cross-validation (CV) strategies that respect the inherent dependencies in the data. Standard k-fold cross-validation, which assumes data points are independent, can lead to overly optimistic performance estimates. The `bstm_cv_orchestrator` function provides a suite of specialized CV methods designed for spatiotemporal data.

#### Arguments

| Argument        | Type              | Default           | Description                                                                                                          |
| :----------------| :------------------| :------------------| :---------------------------------------------------------------------------------------------------------------------|
| `formula`       | `String`          |                   | The `bstm` model formula.                                                                                            |
| `data`          | `DataFrame`       |                   | The full dataset.                                                                                                    |
| `method`        | `Symbol`          | `:kfold`          | The CV strategy. Options are: `:kfold`, `:lolo`, `:spatial_block`, `:temporal_block`, `:temporal_forward_chain`.     |
| `cv_var`        | `Symbol`          | `:s_idx`          | The column in `data` used for grouping in `:lolo`, `:temporal_block`, and `:temporal_forward_chain` methods.         |
| `n_folds`       | `Int`             | `5`               | The number of folds or blocks. For `:temporal_forward_chain`, it's the number of time steps to hold out for testing. |
| `sampler`       | `AbstractSampler` | `NUTS(500, 0.65)` | The Turing sampler used to fit the model in each fold.                                                               |
| `n_samples`     | `Int`             | `500`             | The number of posterior samples to draw in each fold.                                                                |
| `cv_space_vars` | `Vector{Symbol}`  | `[:s_x, :s_y]`    | The coordinate columns used for `:spatial_block` clustering.                                                         |
| `kwargs...`     |                   |                   | Additional keyword arguments passed to the underlying `bstm_config` call.                                            |

#### Cross-Validation Methods

*   **`:kfold`**: Standard random k-fold cross-validation. Suitable only when observations can be considered independent.
*   **`:lolo` (Leave-One-Location-Out)**: Each fold consists of all observations from a unique level of the `cv_var` (e.g., a spatial unit `s_idx`). This tests the model's ability to predict at entirely new locations.
*   **`:spatial_block`**: Creates `n_folds` spatial blocks using k-means clustering on the `cv_space_vars`. This tests spatial extrapolation performance.
*   **`:temporal_block`**: Divides the data into `n_folds` contiguous blocks based on the `cv_var` (e.g., `year`). This tests interpolation performance for missing time periods.
*   **`:temporal_forward_chain`**: A forecasting simulation. It iteratively trains on data up to a certain time point and tests on the next time point. This is repeated for the last `n_folds` time points.

#### Example Usage

```julia
# Perform 5-fold spatial block cross-validation
cv_results = bstm_cv_orchestrator(
    "likelihood(y) ~ intercept() + random(s_idx, model=bym2)",
    data,
    method = :spatial_block,
    n_folds = 5,
    n_samples = 1000,
    W = W_matrix
)

# Display the results
println("Mean RMSE across folds: ", cv_results.mean_rmse)
display(cv_results.folds)
```

#### Output

The function returns a `NamedTuple` containing:
*   `folds`: A vector of `NamedTuple`s, with each containing the `rmse` and `r2` for that fold.
*   `mean_rmse`: The average RMSE across all folds.
*   `mean_r2`: The average R-squared across all folds.

## 8. Conclusion

The `bstm` framework provides a powerful and extensible environment for Bayesian spatiotemporal modeling within the Julia ecosystem. By leveraging a modular, component-based architecture and a formula-driven interface, it bridges the gap between high-level model specification and the low-level performance of the Turing.jl probabilistic programming language. The `ComponentModel` interface is central to this design, offering a clear and consistent contract for developers to integrate novel statistical structures, from simple random effects to complex mechanistic models.

This document has detailed the internal machinery of the framework, from the formula parsing and model configuration engines to the code generation and posterior reconstruction pipelines. By exposing these internals, `bstm` aims to empower advanced users and developers to not only use the available components but also to extend the framework to meet new research challenges. The combination of discrete GMRFs, continuous Gaussian Processes, and advanced approximation techniques provides a rich toolbox for tackling a wide array of problems in ecological modeling and beyond.

## 9. References

*   Besag, J. (1974). Spatial interaction and the statistical analysis of lattice systems. *Journal of the Royal Statistical Society: Series B (Methodological)*, 36(2), 192-225.
*   Besag, J., York, J., & Mollié, A. (1991). Bayesian image restoration, with applications in spatial statistics. *Annals of the Institute of Statistical Mathematics*, 43(1), 1-59.
*   Cliff, A. D., & Ord, J. K. (1973). *Spatial autocorrelation*. Pion.
*   Damianou, A., & Lawrence, N. (2013, April). Deep gaussian processes. In *Artificial intelligence and statistics* (pp. 207-215). PMLR.
*   Gelfand, A. E., Kim, H. J., Sirmans, C. F., & Banerjee, S. (2003). Spatial modeling with spatially varying coefficient processes. *Journal of the American Statistical Association*, 98(462), 387-396.
*   Gelfand, A. E., & Vounatsou, P. (2003). Proper multivariate conditional autoregressive models for spatial data analysis. *Biostatistics*, 4(1), 11-15.
*   Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time variation in disease risk. *Statistical Methods in Medical Research*, 9(3), 205-220.
*   Leroux, B. G., Lei, X., & Breslow, N. (2000). Estimation of disease rates in small areas: a new mixed model for spatial dependence. In *Statistical models in epidemiology, the environment, and clinical trials* (pp. 179-191). Springer, New York, NY.
*   Lewandowski, D., Kurowicka, D., & Joe, H. (2009). Generating random correlation matrices based on vines and extended onion method. *Journal of multivariate analysis*, 100(9), 1989-2001.
*   Lindgren, F., Rue, H., & Lindström, J. (2011). An explicit link between Gaussian fields and Gaussian Markov random fields: The SPDE approach. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 73(4), 423-498.
*   Mullahy, J. (1986). Specification and testing of some modified count data models. *Journal of econometrics*, 33(3), 341-365.
*   Rahimi, A., & Recht, B. (2008). Random features for large-scale kernel machines. *Advances in Neural Information Processing Systems*, 20.
*   Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
*   Riebler, A., Sørbye, S. H., & Rue, H. (2016). An intuitive Bayesian spatial model with two hyperparameters. *Statistical Methods in Medical Research*, 25(2), 1145-1160.
*   Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sørbye, S. H. (2017). Penalising model component complexity: A principled, practical approach to constructing priors. *Statistical Science*, 32(1), 1-28.
*   Snelson, E., & Ghahramani, Z. (2006). Sparse Gaussian processes using pseudo-inputs. *Advances in neural information processing systems*, 18.
*   Wikle, C. K. (2003). Hierarchical Bayesian models for predicting the spread of ecological processes. *Ecology*, 84(6), 1382-1394.
*   Williams, C. K., & Seeger, M. (2001). Using the Nyström method to speed up kernel machines. In *Advances in neural information processing systems*, 13. 