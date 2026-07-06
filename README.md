# Bayesian Spatial-temporal Models in Julia/Turing
 
 
## Installation

Clone the repository or copy/download the Julia files (\*.jl)

 
*The following is a copy of docs/bstm_overview.md* 
 

# The `bstm` Framework: An Architectural Overview

## 1. Introduction

In the landscape of computational statistics, a primary bottleneck in spatiotemporal inference is the structural rigidity of traditional models that often hard-code the relationship between covariates and random effects. This can lead to models that are computationally fragile and difficult to extend. The `bstm` framework addresses this by treating the latent process as an orthogonal, composable entity, decoupled from the observation likelihood.

## 2. The Formula Interface

The `bstm` framework separates the observation model from the latent process model. The observation model is defined on the left-hand side (LHS) of the formula using the `likelihood()` module, while the latent process is defined on the right-hand side (RHS). The `@bstm` macro provides an unquoted formula interface for improved readability and ergonomics.

### 2.1. Basic Structure

The general structure of a `bstm` model call is:

```julia
m = @bstm(
    likelihood(outcome_var, family=:poisson, ...) ~ 1 + fixed_effects + modules(...),
    data_frame,
    keyword_arguments...
)
```

### 2.2. The `likelihood()` Module

The `likelihood()` module on the LHS specifies the observation model.

| Parameter    | Example Usage       | Description                                                                    |
| :-------------| :--------------------| :-------------------------------------------------------------------------------|
| `family`     | `family=:poisson`   | Sets the likelihood distribution (e.g., `:poisson`, `:gaussian`, `:binomial`). |
| `offsets`    | `offsets=log_pop`   | Specifies a log-scale offset, typically for exposure in count models.          |
| `weights`    | `weights=sample_w`  | Applies observation-level weights to the log-likelihood.                       |
| `trials`     | `trials=n_patients` | Defines the number of trials for a binomial likelihood.                        |
| `zi`         | `zi=true`           | Enables a zero-inflation component for count models.                           |
| `hurdle`     | `hurdle=0`          | Implements a hurdle model, separating the zero-generating process.             |
| `volatility` | `volatility=true`   | Enables a spatiotemporal stochastic volatility model for observation noise.    |
| `y_L`, `y_U` | `y_L=:lower_b`      | Defines lower and upper bounds for censored data.                              |

### 2.3. Illustrative Examples

1.  **BYM2 Disease Mapping:**
    `@bstm(likelihood(y, family=:poisson) ~ 1 + spatial(s_idx, model=bym2), data, W=W)`
    Decomposes risk into structured spatial and unstructured IID noise.

2.  **AR1 Temporal Forecasting:**
    `@bstm(likelihood(y) ~ 1 + temporal(t_idx, model=ar1), data)`
    Captures geometric temporal decay.

3.  **Spatio-Temporal Interaction:**
    `@bstm(likelihood(y) ~ 1 + spatial(s_idx, model=besag) ⊗ temporal(t_idx, model=ar1), data, W=W)`
    Employs the Kronecker product to create a fully structured spatiotemporal interaction field.

4.  **Spatially Varying Coefficients (SVC):**
    `@bstm(likelihood(y) ~ 1 + (poverty |> spatial(s_idx, model=icar)), data, W=W)`
    Allows the impact of `poverty` to vary according to local spatial gradients.

5.  **Spectral Splines:**
    `@bstm(likelihood(y) ~ 1 + smooth(lon, lat, model=rff), data)`
    Approximates a 2D continuous field without the $O(N^3)$ kernel inversion cost.

## 3. The Algebra of Manifolds: Composition and State-Space Models

The `bstm` DSL operates through a recursive parser that allows for the algebraic composition of manifolds.

### 3.1. Algebraic Operators

1.  **Direct Sum (⊕)**: Creates a block-diagonal precision matrix from its component manifolds, i.e., $Q_{total} = \text{diag}(Q_1, Q_2)$. This is used to model components that are structurally independent but are specified to share a common hyperparameter (e.g., variance). For simple additive effects with separate, unshared hyperparameters, use `+`.
2.  **Kronecker Product (⊗)**: Used for creating inseparable interaction effects, such as the Knorr-Held Type IV model (`spatial() ⊗ temporal()`). This builds a joint precision matrix $Q_{st} = Q_t \otimes Q_s$, enabling the representation of space-time interactions where every spatial location has a unique, correlated temporal trend.
3.  **Composition (∘)**: Represents the composition of two precision structures as a functional transformation, e.g., $Q_{total} = Q_1 * Q_2 * Q_1$. This can be used for basis warping or creating cascading dependencies between manifolds of the same dimension.

### 3.2. Structural Transformations and State-Space Models

The Pipe (`|>`) operator handles data normalization, Spatially Varying Coefficients (SVC), and state-space evolution.

*   **Transformations**: Objects like `ZScoreManifold` or `LogManifold` act as wrappers that normalize inputs before they enter the latent process.
*   **SVC Logic**: The notation `covariate |> spatial()` instructs the parser to generate a random slope for the covariate that is spatially structured by the specified manifold, effectively modeling local non-stationarity.
*   **State-Space Evolution**: The pipe operator defines a state-space model where one manifold evolves over the domain of another. This supports both discrete-time dynamics (e.g., `spatial() |> temporal(model=ar1)`) and the creation of spatially-varying curves (e.g., `spatial() |> smooth(time, model=pspline)`), where the coefficients of the temporal basis functions are modeled as spatial fields.

## 4. Core Components: Manifolds and Priors

The mathematical registry of `bstm` standardizes the transition from discrete graph topologies to continuous spectral approximations.

### 4.1. The Discrete Registry: Gaussian Markov Random Fields (GMRF)

For discrete domains, `bstm` implements GMRF structures where dependency is defined by a precision matrix Q.

| Manifold | Theoretical Assumption | Structural Rationale |
|:---|:---|:---|
| IID | $\epsilon \sim N(0, \sigma^2 I)$ | Unstructured exchangeability; base model for PC-shrinkage. |
| ICAR | Intrinsic CAR; $Q_{ij} = -1$ for neighbors. | Pure local smoothing; identifies spatial gradients. |
| Besag | Standard CAR model. | Global and local spatial dependency via fixed precision. |
| BYM2 | Scaled Besag + IID component. | Explicit variance partitioning ($\rho$) for better identifiability. |
| Leroux | Convex combination of I and $Q_{ICAR}$. | Bridges IID and ICAR structures through a mixing parameter. |
| SAR | $(I - \rho W)y = \epsilon$. | Simultaneous modeling of response autocorrelation. |
| RW1 / RW2 | Random Walk (1st/2nd order). | Temporal continuity and smoothing of non-stationary trends. |
| AR1 | $\mu_t = \rho \mu_{t-1} + \epsilon_t$. | Stationary temporal process with geometric decay. |

### 4.2. Continuous, Spectral, and Advanced Manifolds

To circumvent the $O(N^3)$ cost of kernel-based Gaussian Processes, the framework utilizes spectral projections and sparse approximations:

*   **Random Fourier Features (RFF)**: Maps input coordinates `x` into a randomized feature space to approximate the kernel. The projection is defined as $z(x) = \sqrt{2/M} \cos(Wx + b)$, where `W` is the spectral frequency matrix sampled from the kernel’s Fourier transform.
*   **SPDE**: Represents the field as a solution to $(\kappa^2 - \Delta)^{\alpha/2} u = \mathcal{W}$, mapping continuous Matérn processes onto a discrete mesh.
*   **Nystrom / FITC**: Low-rank approximations using `n_inducing` points to represent the global field.
*   **NetworkFlow**: Captures directed dependencies across an adjacency matrix with `:upstream` or `:downstream` dispatch.

### 4.3. Priors and Identifiability

Stability in high-dimensional models is achieved through a principled approach to prior specification. The `bstm` framework provides three built-in prior schemes and allows for granular, user-defined overrides.

#### Prior Schemes

1.  **Penalized Complexity Priors (`:pcpriors`)**: This is the default scheme. PC priors are designed to shrink complex models towards simpler "base models" unless there is strong evidence in the data to the contrary. For example, the prior on a variance parameter (`sigma`) shrinks towards zero, and the prior on a correlation parameter (`rho`) shrinks towards zero (no correlation). This is the recommended scheme for most applications as it helps prevent overfitting and improves model identifiability.

2.  **Informative Priors (`:informative`)**: This scheme uses priors that are still weakly informative but less aggressive in their shrinkage than PC priors. For example, the prior on `rho` is a `Beta(2, 2)`, which is centered at 0.5, reflecting a belief that some correlation is more likely than none. This can be useful when you have prior knowledge that an effect is likely present.

3.  **Uninformative Priors (`:uninformative`)**: This scheme uses very wide, flat priors (e.g., `Normal(0, 1e6)` for `sigma`, `Uniform(0, 1)` for `rho`). While sometimes used to express ignorance, these priors are generally **not recommended** for complex hierarchical models, as they can lead to poor convergence and unidentifiable parameters.

#### Prior Comparison Table

| Parameter | PC Prior (Default) | Informative Prior | Uninformative Prior | Rationale |
|:---|:---|:---|:---|:---|
| **Sigma** ($\sigma$) | `Exponential(1.0)` | `Exponential(0.5)` | `Normal(0, 1e6)` | Controls the marginal standard deviation of a latent field. PC prior shrinks towards zero variance. |
| **Rho** ($\rho$) | `Beta(1, 1)` | `Beta(2, 2)` | `Uniform(0, 1)` | Controls spatial/temporal correlation. PC prior is uniform, while informative prior favors correlation. |
| **Lengthscale** | `InverseGamma(3, 3)` | `InverseGamma(5, 5)` | `InverseGamma(0.01, 0.01)` | Controls the range of correlation in continuous GP models. PC prior prevents overfitting. |
| **Kappa** ($\kappa$) | `Exponential(1.0)` | `Exponential(0.1)` | `Exponential(10.0)` | Controls the smoothness of an SPDE/Matérn field. PC prior shrinks towards a smoother field. |
| **Amplitude** | `Normal(0, 1)` | `Normal(0, 0.5)` | `Normal(0, 100)` | Controls the amplitude of harmonic (seasonal) components. |
| **Phase** | `Beta(1, 1)` | `Beta(2, 2)` | `Uniform(0, 1)` | Controls the phase shift of harmonic components. |

#### Setting Priors in a Model

You can control prior specification at three levels of precedence:

1.  **Local Override (Highest Precedence)**: Specify a prior directly within a module call. This will always override any global settings.
    ```julia
    @bstm(
        likelihood(y) ~ 1 + spatial(s_idx, model=bym2, sigma_prior=Exponential(0.1)),
        data, W=W
    )
    ```

2.  **Global Override**: Pass a dictionary of hyperpriors to the `bstm` call. This overrides the default scheme for specific parameter types across all relevant modules.
    ```julia
    @bstm(
        likelihood(y) ~ 1 + spatial(s_idx, model=bym2) + temporal(t_idx, model=ar1),
        data, W=W, hyperpriors=Dict("sigma" => Exponential(0.2))
    )
    ```

3.  **Global Scheme Selection (Lowest Precedence)**: Select a global prior scheme.
    ```julia
    @bstm(
        likelihood(y) ~ 1 + spatial(s_idx, model=bym2),
        data, W=W, prior_scheme=:informative
    )
    ```

## 5. Architectural Paradigms

### 5.1. Univariate Architecture

The default kernel for single-outcome processes, as described in the preceding sections.

### 5.2. Multivariate Architecture

The `MultivariateArchitecture` is triggered when multiple outcomes are specified on the LHS of the formula (e.g., `y1 + y2 ~ ...`). It is designed to jointly model these outcomes, allowing the model to "borrow strength" across related processes and estimate the correlation structure between them.

#### Key Mechanisms:

*   **Outcome-Specific Parameters**: Each manifold (e.g., `spatial`, `temporal`) generates a separate latent field for each outcome. This means hyperparameters like `sigma_spatial` or `rho_temporal` are estimated independently for each response variable, providing maximum flexibility.
*   **LKJ Correlation Prior**: The core of the multivariate coupling is the `LKJCholesky` prior on a correlation matrix `L_corr`. The final latent effects are constructed by multiplying the matrix of independent latent fields by the Cholesky factor of this correlation matrix: `eta_final = eta_latent * L_corr.L`. This induces a shared correlation structure across all outcomes for a given manifold, ensuring that the model captures shared patterns while allowing for outcome-specific variances.
*   **Householder Reflection (Spectral Orientation)**: For advanced use cases, the framework can apply a Householder reflection (`H = I - 2vv'`) to the latent fields. This allows the outcomes to rotate in the latent space, which is useful for aligning signals in models with complex dependencies like transport or advection-diffusion, where the direction of correlation is as important as its magnitude.

### 5.3. Multifidelity Architecture

The `MultifidelityArchitecture` is designed for data fusion, integrating high-volume, low-cost proxy data with sparse, high-quality observations. It is typically invoked using the `nested()` module.

#### Key Mechanisms:

*   **Hierarchical Latent Fields**: The architecture establishes a hierarchy of latent processes. A common setup involves:
    1.  **High-Fidelity (Target)**: The primary outcome variable (`y_hq`).
    2.  **Low-Fidelity (Proxy)**: A secondary, related variable (`y_lq`) with more abundant data.
*   **Nested Supervision**: The `nested()` module defines a complete sub-model for the low-fidelity data. The latent field from this sub-model (`eta_sub`) is then used as a calibrated predictor in the main model for the high-fidelity data.
*   **Calibration Parameters**: The link between the fidelities is modeled with calibration parameters, typically a bias and a scaling factor (rho), which are estimated within the model:
    `eta_main = ... + rho_nested * eta_sub`
    The prior on `rho_nested` is often centered around 1.0, assuming the proxy is a reasonably good, if biased, predictor of the main process. This allows the main model to learn from the structural patterns in the low-fidelity data while correcting for systematic bias and scale differences.

## 6. Advanced Topics

### 6.1. Spatial Partitioning

For discrete spatial models (GMRFs), the continuous spatial domain must be discretized into "Areal Units" (AUs). The `assign_spatial_units` function provides several methods for this, balancing geometric compactness with statistical information density.

| Method | Description | Justification |
|:---|:---|:---|
| `:cvt` | **Centroidal Voronoi Tessellation** | Iteratively minimizes variance to create geometrically regular cells. |
| `:kvt` | **K-Means Voronoi Tessellation** | Uses K-Means to create units with a balanced number of observations. |
| `:avt` | **Agglomerative Voronoi** | A bottom-up approach that merges small units to prevent data starvation. |
| `:bvt` | **Binary Vector Tree** | Employs recursive partitioning along the axis of maximum variance to efficiently handle large datasets and balance point counts. |
| `:qvt` | **Quadrant Voronoi Tessellation** | A quadtree-like recursive method that splits regions into four quadrants, adapting to multi-scale spatial clusters. |
| `:hvt` | **Hierarchical Voronoi** | Combines K-Means seeding with geometric refinement for stable, well-behaved polygons. |
| `:lattice`| **Regular Grid** | Simple, fast discretization into uniform squares. Assumes stationarity. |

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

where `L` is the graph Laplacian. This allows the model to learn physical parameters like `velocity` within a fully Bayesian context.

**Example: Logistic Growth Model**

A logistic growth model for a population can be specified as:

```julia
@bstm(
    likelihood(counts, family=:poisson) ~ 1 + dynamics(time, model=logistic_f, r_covariate=:temp),
    data
)
```

This example fits a logistic growth model where the intrinsic growth rate `r` is itself a function of temperature.

### 6.3. Multi-fidelity and Nested Models

The `nested()` module is a "supervisor" component for multi-fidelity modeling. It allows you to define a complete sub-model that is fit to a separate (often larger, lower-quality) dataset. The latent effect from this sub-model is then incorporated as a calibrated predictor into the main model, allowing the main model to "learn" from the proxy data.

```julia
@bstm(
    likelihood(y_hq) ~ 1 + spatial(s_idx) + nested(proxy_signal, formula="y_lq ~ 1 + smooth(x)", data_source=:low_quality_data),
    high_quality_data,
    low_quality_data = df_low_quality
)
```

### 6.4. Bayesian Factor Analysis with `eigen()`

The `eigen()` module implements a Bayesian Principal Component Analysis (PCA) to perform dimensionality reduction on a set of multivariate outcomes. It decomposes the input variables into a smaller set of orthogonal latent factors. The framework uses a Householder transformation to construct the orthonormal loadings matrix, ensuring numerical stability and efficient sampling.

### 6.5. Handling Censored Covariates via Joint Modeling

A censored covariate is a predictor variable for which the true value is not always known, but is instead confined to an interval (e.g., $x_{true} > c$). The statistically robust approach to this "errors-in-variables" problem is to treat the censored covariate as a latent variable and model it jointly with the primary outcome.

The `bstm` framework facilitates this through the `nested()` module, which allows for the construction of a joint model in a single step. This approach simultaneously estimates the model for the censored covariate and the main outcome model, correctly propagating all sources of uncertainty.

#### Implementation with `nested()`

In this setup, the `nested()` module defines a complete sub-model for the censored covariate. This sub-model has its own `likelihood()` block where the censoring bounds (`y_L`, `y_U`) are specified. The latent process estimated by this sub-model is then automatically incorporated as a predictor in the main model's linear predictor.

**Example: Using `nested()` for a Censored Covariate**

```julia
# Assume 'x_censored' is the covariate with censoring, and 'x_L' and 'x_U' are columns
# in the data indicating the censoring bounds. 'z1' is another fully observed predictor.

# The main model for 'y' includes a `nested()` term named `x_latent_process`.
# This term defines a sub-model where 'x_censored' is the outcome.
# The sub-model's `likelihood()` handles the censoring of 'x_censored'.
# The latent effect from this sub-model is then automatically added as a predictor to the main model.

joint_model = @bstm(
    likelihood(y, family=:poisson) ~ 1 + z1 +
        nested(x_latent_process,
            formula="likelihood(x_censored, family=:gaussian, y_L=:x_L, y_U=:x_U) ~ 1 + z1"
        ),
    my_data
)

# Sample the joint model, which will estimate all parameters simultaneously.
joint_chain = sample(joint_model, NUTS(), 1000)
```

**Stage 2: Main Analysis with Imputed Covariates**

Next, run your primary analysis model multiple times, once for each set of imputed values for the censored covariate. The final posterior is the combination of all chains.

```julia
main_model_chains = []
n_imputations = 100 # Number of imputed datasets to use

for i in 1:n_imputations
    # Create a copy of the data and replace the censored covariate column
    # with one of the imputed posterior samples.
    imputed_data = copy(my_data)
    imputed_data.x_censored = imputed_x_samples.predictions_noisy.mean[:, i] # Or use samples directly

    # Run the main model with the imputed data
    main_model = @bstm(
        likelihood(y, family=:poisson) ~ 1 + x_censored + some_other_var,
        imputed_data
    )
    
    chain = sample(main_model, NUTS(), 500)
    push!(main_model_chains, chain)
end

# Combine the chains from all runs to get the final posterior, which now
# correctly propagates the uncertainty from the censored covariate.
final_chain = vcat(main_model_chains...)
```

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

The `predict()` function projects a fitted model onto a new data grid to generate out-of-sample predictions. It correctly handles the projection of all manifold types, including re-computing basis matrices for smooth terms on the new data.

```julia
preds = predict(model, chain, new_data_frame)
```

## 8. API Reference: Cheat Sheet

This section provides a detailed quick-reference guide to the main modules available in the `bstm` formula interface.

### 8.1. `likelihood()` Module

| Parameter | Example Usage | Data Type | Default | Meaning & Assumptions |
|:---|:---|:---|:---|:---|
| `log_offsets` or `offsets` | `offsets=:pop_log` | `Symbol` | None | Provides a log-scale offset to the linear predictor ($\eta' = \eta + \text{offset}$). Essential for modeling rates. |
| `weights` | `weights=:sample_w` | `Symbol` | `1.0` | Multiplies the log-likelihood of each observation by the specified weight. |
| `trials` | `trials=:n_patients` | `Symbol` | `1` | Specifies the number of trials for each observation in a Binomial model. |
| `volatility`| `volatility=true` | `Bool` | `false` | Enables a spatiotemporal stochastic volatility model for the observation noise ($\sigma_y$). |
| `y_L`, `y_U`| `y_L=:lower_b` | `Symbol` | `-Inf`, `+Inf` | Defines the lower (`y_L`) and upper (`y_U`) bounds for censored observations. |
| `hurdle` | `hurdle=0` | `Number` | `-Inf` | Implements a hurdle model by truncating the likelihood below the specified threshold. |

### 8.2. Likelihood Families

| Family | `family=...` | Link Function | Key Parameters & Priors |
|:---|:---|:---|:---|
| **Poisson** | `:poisson` | `exp(eta)` | `rate (λ)`: Determined by `exp(eta)`. |
| **Gaussian** | `:gaussian` | `identity(eta)` | `mean (μ)`: `eta`, `std. dev. (σ)`: `y_sigma ~ Exponential(1.0)` |
| **Log-Normal** | `:lognormal` | `identity(eta)` | `log-mean (μ)`: `eta`, `log-std. dev. (σ)`: `y_sigma ~ Exponential(1.0)` |
| **Negative Binomial** | `:negbin` | `exp(eta)` | `rate (μ)`: `exp(eta)`, `dispersion (r)`: `r_nb ~ Exponential(1.0)` |
| **Binomial** | `:binomial` | `logistic(eta)` | `trials (n)`: From `likelihood(trials=...)`, `probability (p)`: `logistic(eta)` |
| **Gamma** | `:gamma` | `exp(eta)` | `shape (α)`: `extra_params ~ Exponential(1.0)`, `scale (θ)`: `exp(eta)/α` |
| **Beta** | `:beta` | `logistic(eta)` | `mean (μ)`: `logistic(eta)`, `precision (φ)`: `extra_params ~ Exponential(1.0)` |
| **Student's T** | `:student_t` | `identity(eta)` | `location (μ)`: `eta`, `scale (σ)`: `y_sigma ~ Exponential(1.0)`, `d.f. (ν)`: `extra_params ~ Exponential(1.0)` |
| **Exponential** | `:exponential` | `exp(eta)` | `rate (λ)`: `1 / exp(eta)`. |
| **Inverse Gaussian** | `:inverse_gaussian` | `exp(eta)` | `mean (μ)`: `exp(eta)`, `shape (λ)`: `extra_params ~ Exponential(1.0)` |
| **Half-Normal** | `:half_normal` | `identity(eta)` | `std. dev. (σ)`: `y_sigma ~ Exponential(1.0)`. Mean is implicitly 0. |
| **Half-Student's T** | `:half_student_t` | `identity(eta)` | `scale (σ)`: `y_sigma ~ Exponential(1.0)`, `d.f. (ν)`: `extra_params ~ Exponential(1.0)` |
| **Laplace** | `:laplace` | `identity(eta)` | `location (μ)`: `eta`, `scale (b)`: `y_sigma ~ Exponential(1.0)` |
| **Pareto** | `:pareto` | `exp(eta)` | `shape (α)`: `extra_params ~ Exponential(1.0)`, `scale (θ)` from mean. |
| **Dirichlet** | `:dirichlet` | `exp(eta)` | `concentration (α)`: `exp.(eta)`. For multivariate compositional data. |
| **Inverse Wishart** | `:inverse_wishart` | `identity(eta)` | `d.f. (ν)`: `extra_params`, `Scale Matrix (Ψ)` from `eta`. For covariance modeling. |

### 8.3. `spatial()` Module

| Manifold             | `model='...'`       | Key Parameters                                      | Default PC-Priors                                               | Use Case & Utility                                                                                     |
| :---------------------| :--------------------| :----------------------------------------------------| :----------------------------------------------------------------| :-------------------------------------------------------------------------------------------------------|
| **IID**              | `'iid'`             | `sigma_prior`                                       | `Exponential(1.0)`                                              | Models non-spatial overdispersion or heterogeneity.                                                    |
| **ICAR / Besag**     | `'icar'`, `'besag'` | `sigma_prior`                                       | `Exponential(1.0)`                                              | Provides strong, localized spatial smoothing for lattice data.                                         |
| **BYM2**             | `'bym2'`            | `sigma_prior`, `rho_prior`                          | `sigma`: `Exponential(1.0)`, `rho`: `Beta(1,1)`                 | The most robust default for areal data; separates spatial clustering from random noise.                |
| **Leroux**           | `'leroux'`          | `sigma_prior`, `rho_prior`                          | `sigma`: `Exponential(1.0)`, `rho`: `Beta(1,1)`                 | A flexible alternative to BYM2 that avoids the rank-deficiency of the ICAR model.                      |
| **SAR**              | `'sar'`             | `sigma_prior`, `rho_prior`                          | `sigma`: `Exponential(1.0)`, `rho`: `Beta(1,1)`                 | Models spatial "spill-over" effects where the value at one location directly influences its neighbors. |
| **SPDE**             | `'spde'`            | `sigma_prior`, `kappa_prior`                        | `sigma`: `Exponential(1.0)`, `kappa`: `Exponential(1.0)`        | A scalable and principled way to model continuous spatial processes on irregular domains.              |
| **Gaussian Process** | `'gp'`              | `sigma_prior`, `lengthscale_prior`, `kernel`        | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Gold-standard for continuous spatial modeling but computationally expensive ($O(N^3)$).                |
| **RFF**              | `'rff'`             | `sigma_prior`, `lengthscale_prior`, `n_features`    | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | A scalable approximation to a full GP, excellent for large numbers of areal units.                     |
| **FITC**             | `'fitc'`            | `sigma_prior`, `lengthscale_prior`, `n_inducing`    | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Sparse GP using inducing points; good for large N.                                                     |
| **Nystrom**          | `'nystrom'`         | `sigma_prior`, `lengthscale_prior`, `n_inducing`    | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Low-rank GP approximation, similar to FITC.                                                            |
| **SVGP**             | `'svgp'`            | `sigma_prior`, `lengthscale_prior`, `n_inducing`    | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Sparse Variational GP, for use with VI.                                                                |
| **Warp**             | `'warp'`            | `sigma_prior`, `lengthscale_prior`, `n_features`    | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Models non-stationary fields by warping coordinates.                                                   |
| **NetworkFlow**      | `'network'`         | `sigma_prior`, `adjacency_matrix`, `flow_direction` | `Exponential(1.0)`                                              | For directed graphs like river networks or supply chains.                                              |
| **DAG**              | `'dag'`             | `sigma_prior`, `adjacency_matrix`                   | `Exponential(1.0)`                                              | For Directed Acyclic Graphs, useful in causal inference.                                               |
| **Mosaic**           | `'mosaic'`          | `sigma_prior`, `n_regions`                          | `Exponential(1.0)`                                              | Partitions space into locally stationary regions.                                                      |
| **BCGN**             | `'bcgn'`            | `sigma_prior`, `bipartite_adj`                      | `Exponential(1.0)`                                              | For bipartite graphs (e.g., user-item interactions).                                                   |
| **Hyperbolic**       | `'hyperbolic'`      | `sigma_prior`, `curvature`                          | `Exponential(1.0)`                                              | For embedding hierarchical or tree-like spatial data.                                                  |

### 8.4. `temporal()` and `seasonal()` Modules

| Manifold | `model='...'` | Key Parameters | Default PC-Priors | Use Case & Utility |
|:---|:---|:---|:---|:---|
| **IID** | `'iid'` | `sigma_prior` | `Exponential(1.0)` | Models unstructured temporal noise. |
| **AR1** | `'ar1'` | `sigma_prior`, `rho_prior` | `sigma`: `Exponential(1.0)`, `rho`: `Beta(1,1)` | Modeling serially correlated time series where the influence of past events decays geometrically. |
| **Random Walk (RW1)** | `'rw1'` | `sigma_prior` | `Exponential(1.0)` | Capturing abrupt changes or step-like trends. |
| **Random Walk (RW2)** | `'rw2'` | `sigma_prior` | `Exponential(1.0)` | The most common choice for modeling smooth, non-linear temporal trends. |
| **Gaussian Process** | `'gp'` | `sigma_prior`, `lengthscale_prior` | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Flexible, non-parametric trend modeling. |
| **RFF** | `'rff'` | `sigma_prior`, `lengthscale_prior`, `n_features` | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Scalable GP approximation for long time series. |
| **Cyclic** | `'cyclic'` | `sigma_prior`, `period` | `Exponential(1.0)` | Modeling smooth, periodic effects like day-of-week or month-of-year. |
| **Harmonic** | `'harmonic'` | `amplitude_prior`, `phase_prior`, `period` | `amplitude`: `Normal(0,1)`, `phase`: `Beta(1,1)` | Capturing sharp, regular periodic patterns with sine and cosine waves. |

### 8.5. `smooth()` Module

*Note: Direct censoring of covariates in `smooth()` is not supported. See Section 6.5 for the recommended two-stage modeling approach.*

| Manifold / Method | `model='...'` | Key Parameters | Default Priors | Use Case & Utility |
|:---|:---|:---|:---|:---|
| **P-Spline** | `'pspline'` | `nbins`, `degree`, `diff_order` | `sigma_prior`: `Exponential(1.0)` | The most flexible general-purpose smoother for 1D covariates. |
| **B-Spline** | `'bspline'` | `nbins`, `degree` | `sigma_prior`: `Exponential(1.0)` | A simpler spline smoother than P-splines, useful when less regularization is desired. |
| **Thin Plate Spline** | `'tps'` | `nbins` | `sigma_prior`: `Exponential(1.0)` | The classic choice for smoothing 2D spatial coordinates (e.g., `smooth(lon, lat, model=tps)`). |
| **Random Fourier Features** | `'rff'` | `n_features`, `lengthscale_prior` | `sigma_prior`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | A highly scalable method for approximating a full Gaussian Process smooth. |
| **Random Walk (on bins)** | `'rw1'`, `'rw2'` | `nbins` | `sigma_prior`: `Exponential(1.0)` | A powerful way to model a non-linear effect as a structured random effect on discretized bins. |
| **Gaussian Process (on coords)** | `'gp'` | `lengthscale_prior` | `sigma_prior`: `Exponential(1.0)` | Gold-standard continuous smoother, computationally intensive. |
| **FFT Basis** | `'fft'` | `nbins` | `sigma_prior`: `Exponential(1.0)` | For modeling periodic non-linear effects of a covariate. |
| **Moran's I Basis** | `'moran'` | `nbins`, `W` | `sigma_prior`: `Exponential(1.0)` | Uses eigenvectors of a spatial weights matrix as basis functions. |
| **Spherical Basis** | `'spherical'` | `nbins`, `range_prior` | `sigma_prior`: `Exponential(1.0)` | For effects with a strictly local influence (compact support). |
| **Exponential Decay Basis** | `'decay'` | `nbins`, `lengthscale` | `sigma_prior`: `Exponential(1.0)` | For effects with a strong, rapidly decaying influence. |
| **Barycentric Basis** | `'barycentric'` | `nbins` | `sigma_prior`: `Exponential(1.0)` | Simple, interpretable piecewise linear smoother. |

### 8.6. `mixed()` Module

*Note: Direct censoring of covariates in `mixed()` is not supported. See Section 6.5 for the recommended two-stage modeling approach.*

| Syntax               | Example Usage    | Key Parameters | Default Priors | Mathematical Assumption           |                                                                                                                                              |
| :---------------------| :-----------------| :---------------| :---------------| :----------------------------------| ----------------------------------------------------------------------------------------------------------------------------------------------|
| **Random Intercept** | `mixed(1         | group_var)`    | `model`        | `sigma_prior`: `Exponential(1.0)` | Assumes each level $j$ of `group_var` has a unique intercept $\alpha_j \sim \mathcal{N}(0, \sigma^2_{\text{group}})$.                        |
| **Random Slope**     | `mixed(covariate | group_var)`    | `model`        | `sigma_prior`: `Exponential(1.0)` | Assumes the effect (slope) of a `covariate` varies across the levels of `group_var`, $\beta_j \sim \mathcal{N}(0, \sigma^2_{\text{slope}})$. |

### 8.7. `dynamics()` Module

| Model | `model='...'` | Key Parameters | Default Priors |
|:---|:---|:---|:---|
| **Advection** | `'advection'` | `velocity_prior`, `sigma_prior` | `velocity`: `Normal(0,0.5)`, `sigma`: `Exponential(1.0)` |
| **Diffusion** | `'diffusion'` | `diffusion_prior`, `sigma_prior` | `diffusion`: `LogNormal(-1,1)`, `sigma`: `Exponential(1.0)` |
| **Advection-Diffusion** | `'advection_diffusion'` | `velocity_prior`, `diffusion_prior`, `sigma_prior` | `velocity`: `Normal(0,0.5)`, `diffusion`: `LogNormal(-1,1)` |
| **Directed Spatial Process** | `'directed_spatial_process'` | `rho_prior`, `sigma_innov_prior`, `ls_innov_prior`, `kernel` | `rho`: `Beta(1,1)`, `sigma_innov`: `Exponential(1.0)` |
| **Logistic Growth** | `'logistic_f'` | `r_prior`, `K_prior`, `sigma_F_prior` | `r`: `LogNormal(0,1)`, `K`: `Normal(150,50)`, `sigma_F`: `Exponential(0.5)` |
| **Gompertz Growth** | `'gompertz'` | `r_prior`, `K_prior`, `sig_dyn_prior` | `r`: `LogNormal(-1.5,0.5)`, `K`: `Normal(150,50)` |
| **Custom Model** | `'custom'` | `func` | N/A |

### 8.8. `nested()` and `eigen()` Modules

The `nested()` and `eigen()` modules provide advanced capabilities for multi-fidelity modeling and dimensionality reduction, respectively.

#### `nested()` Module Reference

The `nested()` module is a powerful "supervisor" component used for multi-fidelity modeling and model stacking. It allows you to define a complete sub-model that is fit to a separate (often larger, lower-quality) dataset. The latent effect from this sub-model is then incorporated as a calibrated predictor into the main model, allowing the main model to "learn" from the proxy data.

| Keyword / Parameter     | Example Usage                  | Data Type | Default            | Meaning & Assumptions                                                                                                                                                                                                                                 |
| :------------------------| :-------------------------------| :----------| :-------------------| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nested()`              | `nested(z_var; ...)`           | Module    | N/A                | Defines a supervised sub-model whose latent effect is added to the main model's linear predictor. The `z_var` is a symbolic name for this component.                                                                                                  |
| `formula`               | `formula="z ~ 1 + spatial(s)"` | `String`  | `""`               | A complete `bstm` formula string that defines the structure of the sub-model. This sub-model is fit to the specified `data_source`.                                                                                                                   |
| `data_source`           | `data_source=:proxy_data`      | `Symbol`  | `:data`            | A symbol pointing to a `DataFrame` passed as a keyword argument to the main `bstm()` call. This allows the sub-model to use a different dataset.                                                                                                      |
| `rho_nested` (Implicit) | N/A                            | `Float`   | `Normal(1.0, 0.5)` | A scaling coefficient that links the sub-model's latent effect to the main model's linear predictor: $\eta_{\text{main}} = \dots + \rho_{\text{nested}} \cdot \eta_{\text{sub}}$. The prior assumes the sub-model is a good proxy ($\rho \approx 1$). |

#### `eigen()` Module Reference

The `eigen()` module implements a Bayesian Principal Component Analysis (PCA) to perform dimensionality reduction on a set of multivariate outcomes. It decomposes the input variables into a smaller set of orthogonal latent factors (principal components). The first of these factors is then added to the main model's linear predictor, allowing you to use the dominant shared signal from multiple variables as a predictor.

| Keyword / Parameter | Example Usage                    | Data Type      | Default            | Meaning & Assumptions                                                                                                                                                               |
| :--------------------| :---------------------------------| :---------------| :-------------------| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `eigen()`           | `eigen(y1, y2, y3; ...)`         | Module         | N/A                | Defines a Bayesian PCA factor model. The variables listed (e.g., `y1, y2, y3`) are the multivariate outcomes to be decomposed.                                                      |
| `n_factors`         | `n_factors=1`                    | `Int`          | `1`                | The number of latent factors (principal components) to extract. This determines the dimensionality of the reduced latent space.                                                     |
| `pca_sd_prior`      | `pca_sd_prior=Exponential(0.5)`  | `Distribution` | `Exponential(1.0)` | The prior for the standard deviations of the principal components (latent factors). These are the "eigenvalues" of the system, controlling the variance explained by each factor.   |
| `pdef_sd_prior`     | `pdef_sd_prior=Exponential(0.5)` | `Distribution` | `Exponential(1.0)` | The prior for the standard deviation of the residual (uniqueness) noise. This captures the variance in each observed variable that is *not* explained by the shared latent factors. |

##### The Householder PCA Mechanism

The `eigen()` module uses a Householder transformation to construct the orthonormal loadings matrix ($U$) for the PCA. This is a key feature for ensuring numerical stability and efficient sampling in a Bayesian context.

*   **Why Householder?** Directly sampling an orthonormal matrix is difficult. Traditional methods like Gram-Schmidt orthogonalization are not differentiable and can be numerically unstable within an MCMC sampler like NUTS. The Householder method provides a solution by parameterizing the orthonormal matrix indirectly.
*   **How it Works:** Instead of sampling the $N \times K$ loadings matrix $U$ directly, the model samples a much smaller vector of unconstrained parameters, `v`. These parameters define a sequence of Householder reflections. A Householder reflection is a linear transformation that reflects a vector about a hyperplane. By composing a series of these reflections, we can construct any arbitrary orthonormal matrix.
*   **The Math:** A single reflection is defined by a matrix $H = I - 2vv^T$, where $v$ is a unit vector. The full loadings matrix $U$ is built by recursively applying these reflections: $U_{final} = H_K \dots H_2 H_1 I$.
*   **Benefit:** This process is fully differentiable and allows the sampler to explore the space of orthogonal matrices efficiently and without the geometric constraints that would otherwise make sampling intractable. The model learns the optimal rotation of the latent space by learning the optimal set of reflector vectors `v`.

### 8.9. Spacetime Interaction Manifolds

Spatiotemporal interactions are specified using the Kronecker product operator (`⊗`). The `bstm` framework supports the four canonical interaction types defined by Knorr-Held (2000).

| Manifold     | Formula Syntax                                             | Description                                                  |
| :-------------| :-----------------------------------------------------------| :-------------------------------------------------------------|
| **Type I**   | `spatial(s_idx, model=iid) ⊗ temporal(t_idx, model=iid)`   | Unstructured (IID) interaction over space and time.          |
| **Type II**  | `spatial(s_idx, model=iid) ⊗ temporal(t_idx, model=ar1)`   | Spatially unstructured, temporally structured.               |
| **Type III** | `spatial(s_idx, model=besag) ⊗ temporal(t_idx, model=iid)` | Spatially structured, temporally unstructured.               |
| **Type IV**  | `spatial(s_idx, model=besag) ⊗ temporal(t_idx, model=ar1)` | Fully structured in both space and time (Kronecker product). |

## 9. Conclusion

The `bstm` framework provides a rigorous, Julia-native environment that elevates spatiotemporal modeling from manual implementation to algebraic composition. By treating latent geometries as orthogonal, composable entities, it allows for the construction of highly complex and tailored models that remain computationally tractable. The standardized use of PC-Priors provides a principled way to maintain identifiability in over-parameterized models, while the modular formula interface ensures that models are both easy to specify and to interpret. As the framework matures, future iterations will likely focus on expanding the library of non-Euclidean manifolds and further optimizing the underlying computational kernels.

## 10. References

*   Besag, J., York, J., & Mollié, A. (1991). Bayesian image restoration, with applications in spatial statistics. *Annals of the Institute of Statistical Mathematics*, 43(1), 1-20.
*   Riebler, A., Sørbye, S. H., & Rue, H. (2016). An intuitive Bayesian spatial model with two hyperparameters. *Statistical Methods in Medical Research*, 25(2), 1145-1160.
*   Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time variation in disease risk. *Statistical Methods in Medical Research*, 9(3), 205-220.
*   Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
*   Rahimi, A., & Recht, B. (2008). Random features for large-scale kernel machines. *Advances in Neural Information Processing Systems*, 20.
*   Lindgren, F., Rue, H., & Lindström, J. (2011). An explicit link between Gaussian fields and Gaussian Markov random fields: The SPDE approach. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 73(4), 423-498.
 