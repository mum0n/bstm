---
title: "The Architecture of Bayesian Spatio-Temporal Modeling: A Monograph on the bstm() Framework"
format: html
---

The Architecture of Bayesian Spatio-Temporal Modeling: A Monograph on the `bstm()` Framework

1. Introduction: The Evolution of Latent Gaussian Models

In the current landscape of computational statistics, the primary bottleneck in spatio-temporal inference is the structural rigidity of traditional monolithic models. These legacy approaches often hard-code the relationship between covariates and random effects, leading to models that are computationally fragile and difficult to extend. The bstm() framework resolves this by treating the latent process as an orthogonal, composable entity, decoupled from the observation likelihood.
 
2. Mathematical Foundations: Manifolds and Stochastic Primitives

The mathematical registry of `bstm()` standardizes the transition from discrete graph topologies to continuous spectral approximations. This unified representation is critical for dispatching efficient kernels across varying data scales.

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

To circumvent the O(N^3) cost of kernel-based Gaussian Processes, the framework utilizes spectral projections and sparse approximations found in the `` registry:

* Random Fourier Features (RFF): Maps input coordinates x into a randomized feature space to approximate the kernel. The projection is defined as z(x) = \sqrt{2/M} \cos(Wx + b), where W is the spectral frequency matrix sampled from the kernel’s Fourier transform. This transforms a non-linear GP into a linear model with M features.
* SPDE: Represents the field as a solution to (\kappa^2 - \Delta)^{\alpha/2} u = \mathcal{W}, mapping continuous Matérn processes onto a discrete mesh.
* Nystrom / FITC: Low-rank approximations using n\_inducing points to represent the global field.
* BCGN: Bipartite Coordinate Graph Networks for modeling group-level dependencies.
* NetworkFlow: Captures directed dependencies across an adjacency matrix with `:upstream` or `:downstream` dispatch.
* Hyperbolic: Provides curvature-aware embeddings for hierarchical data representation.

### Comparison of Spectral and Basis Approximation Methods: FFT, RFF, and Wavelets

To handle large continuous spatial fields, `bstm` employs several methods that operate in a frequency or feature space. Understanding their differences is key to selecting the appropriate model.

*   **FFT (Fast Fourier Transform) Method**: This is an **exact** computational method for stationary Gaussian Processes (GPs) on a **regular grid**. It leverages the Wiener-Khinchin theorem, which states that the covariance matrix of such a process is circulant and can be diagonalized by the Discrete Fourier Transform (DFT). Instead of performing expensive `O(N³)` matrix operations in the spatial domain, the FFT method transforms the problem to the frequency domain, where computations become element-wise and scale efficiently at `O(N log N)`. Its primary limitation is the strict requirement for gridded data and stationarity.

*   **RFF (Random Fourier Features) Method**: This is a powerful **approximation** technique for stationary kernels that works on any data geometry, including **irregularly spaced points**. Based on Bochner's theorem, it approximates the kernel function by projecting the input data into a randomized feature space of `M` dimensions using sine and cosine basis functions. The frequencies of these functions are sampled from the kernel's spectral density. This transforms the non-linear GP problem into a linear one, with a computational complexity of roughly `O(NM²)`. Its accuracy is dependent on the number of features `M`.

*   **Wavelet Method (as implemented in `bstm`)**: This is a hybrid approach that applies the multi-resolution philosophy of wavelets within the computationally efficient FFT framework. Unlike the smooth, continuous Power Spectral Density (PSD) used in the standard FFT method, the wavelet implementation in `bstm` models the PSD as a blocky, multi-scale function, where energy is assigned to discrete frequency bands (dyadic scales) corresponding to different wavelet levels. This allows the model to better capture processes where variance is concentrated at specific spatial scales. However, because it still relies on the FFT to construct the final precision matrix, it shares the same requirements and limitations as the FFT method: **regular grids** and an assumption of **stationarity**.

| Feature              | FFT Method                                                           | RFF Method                                                                     | Wavelet Method (in `bstm`)                                                |
| :---------------------| :---------------------------------------------------------------------| :-------------------------------------------------------------------------------| :--------------------------------------------------------------------------|
| **Core Principle**   | Wiener-Khinchin Theorem on circulant matrices.                       | Bochner's Theorem; Monte Carlo approximation of the kernel's spectral density. | Multi-resolution analysis; models PSD in dyadic scales.                   |
| **Data Requirement** | **Regular grid** is required.                                        | General; works on **irregularly spaced** data.                                 | **Regular grid** is required (due to FFT-based implementation).           |
| **Nature of Method** | **Exact** computation (assuming stationarity and grid).              | **Approximation** of the kernel. Quality depends on `M`.                       | **Approximation** of the power spectrum with a multi-scale structure.     |
| **Stationarity**     | Assumes **stationarity**.                                            | Assumes **stationarity**.                                                      | Assumes **stationarity** (as implemented).                                |
| **Key Advantage**    | Extremely fast (`O(N log N)`) and memory efficient for gridded data. | Highly flexible, works for any data geometry, scales well with `N`.            | Better at modeling processes with energy concentrated in specific scales. |
| **Key Disadvantage** | Restricted to regular grids and stationary processes.                | Approximation error; can be slow if `M` is large.                              | Restricted to regular grids; less flexible than a full wavelet transform. |
| **Basis Functions**  | Global Sines and Cosines (Implicitly via DFT).                       | Random Sines and Cosines.                                                      | Localized, scaled functions (Implicitly via multi-scale PSD).             |
* Hyperbolic: Provides curvature-aware embeddings for hierarchical data representation.

3. The Algebra of Manifolds: Operators and DSL Logic

The `bstm()` DSL operates through a recursive parser, `parse_manifold_graph`, which serves as the theoretical engine for model composition.

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

The `bstm_Likelihood` struct standardizes the mapping between the linear predictor (\eta) and the observation space. This architecture ensures that the latent manifold remains invariant regardless of the noise distribution.

The Family Registry and Parameter Mapping

The framework supports a wide array of AbstractBSTM_Family types, utilizing specific link functions to ensure parameter validity:

* Poisson / NegativeBinomial: Uses a log-link for the rate parameter \mu = \exp(\eta). For the NegativeBinomial, the dispersion r is managed via r_nb to ensure positive definite rates.
* Binomial / Beta: Uses a logistic-link to map \eta \in \mathbb{R} to the [0, 1] interval.
* Gaussian / LogNormal / StudentT: For continuous outcome spaces.

Numerical Stability in Censoring

For complex stochastic states, `bstm()` employs specialized kernels. In `IntervalCensored` scenarios, to prevent numerical instability, the system implements `stable_logdiffexp` using `log1mexp` for the probability mass within `[y_L, y_U]`:  `\text{logdiffexp}(a, b) = a + \text{log1mexp}(b - a) \quad \text{where } a > b`  The use of `log1mexp(safe_diff)` prevents domain errors and catastrophic cancellation during the evaluation of the Cumulative Distribution Function (CDF).

5. Architectural Paradigms: Univariate, Multivariate, and Multifidelity

Model dispatch is handled by three core architectures, determined by the dimensionality and quality of the response data.

- UnivariateArchitecture: The default kernel for single-outcome processes.
- MultivariateArchitecture: Models dependent outcomes through matrix-variate kernels. It utilizes the InverseWishartFamily for covariance estimation, ensuring positive definiteness via a PDMat transformation of the latent matrix \eta \eta^T + \epsilon I.
 

6. The bstm() Formulaic Interface: User-Centric Modeling

The user-facing `bstm()` function provides a high-level interface that minimizes "implementation debt" while maintaining full control over the underlying priors.

DSL Syntax Breakdown

* spatial(s_idx; model=bym2): High-level spatial effect dispatch.
* temporal(t_idx; model=rw2): High-level temporal smoothing.
* smooth(x, y; model=rff): 2D spectral smoother using informed RFF parameters.
* eigen(var; rank=k): Low-rank factor with k analytic terms.

Prior Resolution: The PC-Prior Standard

Stability in high-dimensional models is achieved through PC-Priors (Penalized Complexity). These priors are designed to shrink towards a simpler "base model"—for instance, shrinking a spatial field's variance towards zero or its range towards infinity.

Parameter	Default PC-Prior	Rationale
Sigma (\sigma)	Exponential(1.0)	Shrinks towards zero variance (the base model).
Rho (\rho)	Beta(1, 1)	Flat/Uninformative base for correlation parameters.
Lengthscale	InverseGamma(3, 3)	Prevents over-fitting in continuous kernels.
Kappa (\kappa)	Exponential(1.0)	Controls SPDE smoothness via principled shrinkage.

7. Illustrative Examples: Guided Model Implementation

1. BYM2 Disease Mapping: `y ~ 1 + spatial(s_idx; model=bym2)`. Decomposes risk into structured spatial and unstructured IID noise.
2. AR1 Temporal Forecasting: `y ~ 1 + temporal(t_idx; model=ar1)`. Captures geometric temporal decay.
3. Spatio-Temporal Interaction: `y ~ 1 + spatial(s_idx) ⊗ temporal(t_idx)`. Employs tensor-product logic to manage O(N \times T) complexity via the Type IV interaction template.
4. Spatially Varying Coefficients (SVC): `poverty | spatial(s_idx; model=icar)`. Allows the impact of poverty to vary according to local spatial gradients.
5. Spectral Splines: `smooth(lon, lat; model=rff)`. Approximates a 2D continuous field without the O(N^3) kernel inversion cost.
6. Multifidelity Nested Supervision: `nested(z_var; formula='z ~ spatial(s_idx)')`. Uses the `fidelity_metadata` engine and `y_ok` masks to align observations across different quality levels.
7. Zero-Inflated Ecology: `bstm(..., model_family='poisson', use_zi=true)`. Uses the `ZeroInflated` stochastic state to handle excess zeros in count data.

8. Critical Evaluation: Strengths, Weaknesses, and Frontiers

Strengths

The bstm() framework succeeds through orthogonality—the ability to compose complex latent geometries independently of the likelihood kernel. The standardized use of PC-Priors provides a principled way to maintain identifiability in over-parameterized spatio-temporal models.

Weaknesses

- ...
 

Future Frontiers

The expansion into `Hyperbolic` and `NetworkFlow` manifolds signals a move toward Non-Euclidean Bayesian modeling. As the framework matures, future iterations will likely focus on replacing the `collect()` bottlenecks with lazy evaluation and extending the multivariate kernel to the full `AbstractBSTM_Family` registry.

In summary, bstm() provides a rigorous, Julia-native environment that elevates spatio-temporal modeling from manual implementation to algebraic composition, setting a new standard for computational Bayesian research.



# BSTM Cheat Sheet

This document provides a quick reference for the `bstm` package, covering the formula API, common manifold types, and example usage. 

## Formula API Quick Reference

The `bstm` formula language allows you to build complex models by combining modular components.

**Basic Structure:** `outcome ~ intercept + fixed_effects + modules`

### Common Modules

| Keyword              | Example Usage                                 | Purpose                                                                                                                                         |                                                                    |
| :---------------------| :----------------------------------------------| :------------------------------------------------------------------------------------------------------------------------------------------------| --------------------------------------------------------------------|
| `spatial`            | `spatial(s_idx, model=bym2)`                  | Models spatial random effects for discrete areal units.                                                                                         |                                                                    |
| `temporal`           | `temporal(t_idx, u_idx, model=(ar1, cyclic))` | Models temporal trends and/or seasonal effects.                                                                                                 |                                                                    |
| `smooth`             | `smooth(x, nbins=20)`                         | Creates a non-linear smooth of a continuous covariate `x`.                                                                                      |                                                                    |
| `mixed`              | `mixed(1                                      | group)`                                                                                                                                         | Defines a random intercept for each level of the `group` variable. |
| `observationprocess` | `observationprocess(log_offsets=log_offset)`  | Specifies likelihood-level parameters. Options include: `log_offsets`, `weights`, `trials`, `hurdle`, `volatility=true`, `nbins`, `y_L`, `y_U`. |                                                                    |
| `fixed`              | `x1 + x2`                                     | Standard fixed-effect linear predictors.                                                                                                        |                                                                    |

### Data Transformations

Use the `|>` operator inside a module to transform data on the fly.

*   `smooth(x |> log)`: Applies a log transform to `x`.
*   `smooth(x |> zscore)`: Standardizes `x`.
*   `smooth(x |> unit)`: Scales `x` to `[0, 1]`.



### Likelihood Family Reference (`model_family`)

The `model_family` argument in the main `bstm()` call specifies the observation likelihood. It determines the statistical distribution of the outcome variable and the link function used to connect it to the linear predictor `eta`.

| Family                | `model_family` String(s)    | Link Function (`eta` to `mu`) | Key Parameters & Priors                                                                                                     | Meaning, Utility, and Assumptions                                                                                                                                       |
| :----------------------| :----------------------------| :------------------------------| :----------------------------------------------------------------------------------------------------------------------------| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Poisson**           | `"poisson"`                 | `exp(eta)`                    | `rate (λ)`: Determined by `exp(eta)`.                                                                                       | For modeling count data (e.g., number of events, individuals). Assumes the mean of the data is equal to its variance.                                                   |
| **Gaussian**          | `"gaussian"`                | `identity(eta)`               | `mean (μ)`: `eta`, `std. dev. (σ)`: `y_sigma ~ Exponential(1.0)`                                                            | For continuous, symmetric data. Assumes constant variance (homoscedasticity) unless `observationprocess(volatility=true)` is used.                                      |
| **Log-Normal**        | `"lognormal"`               | `identity(eta)`               | `log-mean (μ)`: `eta`, `log-std. dev. (σ)`: `y_sigma ~ Exponential(1.0)`                                                    | For continuous, positive, right-skewed data (e.g., biomass, concentrations). Assumes the logarithm of the data is normally distributed.                                 |
| **Negative Binomial** | `"negbin"`                  | `exp(eta)`                    | `rate (μ)`: `exp(eta)`, `dispersion (r)`: `r_nb ~ Exponential(1.0)`                                                         | For overdispersed count data where the variance is greater than the mean. The dispersion parameter `r` controls the degree of overdispersion.                           |
| **Binomial**          | `"binomial"`, `"bernoulli"` | `logistic(eta)`               | `trials (n)`: From `observationprocess(trials=...)`, `probability (p)`: `logistic(eta)`                                     | For data representing the number of successes in a fixed number of trials. `bernoulli` is a special case where `n=1`, used for binary (0/1) outcomes.                   |
| **Gamma**             | `"gamma"`                   | `exp(eta)`                    | `shape (α)`: `extra_params ~ Exponential(1.0)`, `scale (θ)`: `exp(eta)/α`                                                   | For continuous, positive, right-skewed data (e.g., insurance claims, rainfall). The `extra_params` argument can be used to set a fixed shape `α`.                       |
| **Beta**              | `"beta"`                    | `logistic(eta)`               | `mean (μ)`: `logistic(eta)`, `precision (φ)`: `extra_params ~ Exponential(1.0)`                                             | For data on the (0, 1) interval (e.g., proportions, percentages). The `extra_params` argument controls the precision `φ`.                                               |
| **Student's T**       | `"student_t"`               | `identity(eta)`               | `location (μ)`: `eta`, `scale (σ)`: `y_sigma ~ Exponential(1.0)`, `d.f. (ν)`: `extra_params ~ Exponential(1.0)` (default 5) | A robust alternative to the Gaussian family for continuous data with heavy tails (i.e., more prone to outliers). The degrees of freedom `ν` control the tail thickness. |
| **Laplace**           | `"laplace"`                 | `identity(eta)`               | `location (μ)`: `eta`, `scale (b)`: `y_sigma ~ Exponential(1.0)`                                                            | For continuous data with a sharper peak at the mean and heavier tails than a Gaussian distribution. It is equivalent to minimizing the Mean Absolute Error (MAE).       |
| **Dirichlet**         | `"dirichlet"`               | `exp(eta)`                    | `concentration (α)`: `exp(eta)`                                                                                             | For modeling compositional data, where each observation is a vector of proportions that sum to 1. `eta` is a vector of log-scale parameters.                            |
| **Inverse Wishart**   | `"inverse_wishart"`         | `identity(eta)`               | `d.f. (ν)`: `extra_params ~ Exponential(1.0)`, `Scale Matrix (Ψ)`: `PDMat(eta * eta' + jitter)`                             | For modeling covariance matrices in multivariate models. `eta` is a matrix whose outer product helps form the scale matrix `Ψ`.                                         |


---


### Observation Process Reference (`observationprocess()`)

The `observationprocess()` module is a special component that does not model a latent field itself, but rather configures the observation-level properties of the model's likelihood. It allows you to specify offsets, weights, censoring, and other features that directly affect how the linear predictor `eta` is linked to the observed data `y`.

| Parameter                  | Example Usage               | Data Type | Default        | Meaning & Assumptions                                                                                                                                                                                                                                                            | Use Case & Utility                                                                                                                                                                |
| :---------------------------| :----------------------------| :----------| :---------------| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `log_offsets` or `offsets` | `log_offsets=:pop_log`      | `Symbol`  | None           | Provides a log-scale offset to the linear predictor ($\eta' = \eta + \text{offset}$). Assumes the offset is known and fixed. For Poisson models, this is equivalent to modeling a rate: $\log(\mu) = \eta + \log(\text{exposure})$, so $\mu = \exp(\eta) \cdot \text{exposure}$. | Essential for modeling rates in count data models (e.g., disease incidence per 100,000 people, where `log(population/100000)` is the offset).                                     |
| `weights` or `weight`      | `weights=:sample_w`         | `Symbol`  | `1.0`          | Multiplies the log-likelihood of each observation by the specified weight. Assumes weights are known and non-negative.                                                                                                                                                           | Used in survey statistics to account for complex sampling designs where some observations represent more of the population than others. Can also be used to down-weight outliers. |
| `trials` or `trial`        | `trials=:n_patients`        | `Symbol`  | `1`            | Specifies the number of trials for each observation in a Binomial model. Assumes the number of trials is a known integer.                                                                                                                                                        | Required for Binomial regression where the outcome is the number of successes in a known number of trials (e.g., number of recovered patients out of `n_patients`).               |
| `volatility`               | `volatility=true`           | `Bool`    | `false`        | Enables a spatiotemporal stochastic volatility model for the observation noise ($\sigma_y$). Assumes the log-variance of the noise can be modeled as a smooth, continuous field over space and time.                                                                             | For Gaussian or LogNormal models where the observation error is not constant (heteroscedastic). Useful for financial data or sensor data with varying precision.                  |
| `nbins`                    | `volatility=true, nbins=30` | `Int`     | `20`           | When `volatility=true`, `nbins` specifies the number of Random Fourier Features (`M_rff_sigma`) used to approximate the log-variance surface.                                                                                                                                    | Controls the flexibility of the stochastic volatility surface. Higher values allow for more complex patterns but increase computational cost.                                     |
| `y_L`, `y_U`               | `y_L=:lower_bound`          | `Symbol`  | `-Inf`, `+Inf` | Defines the lower (`y_L`) and upper (`y_U`) bounds for censored observations. Assumes the true value lies outside the observed range but the bounds are known.                                                                                                                   | For censored data, such as instrument detection limits (left-censoring) or survival analysis where an event has not occurred by the end of the study (right-censoring).           |
| `hurdle`                   | `hurdle=0`                  | `Number`  | `-Inf`         | Implements a hurdle model by truncating the likelihood below the specified threshold. Assumes a two-part process: one determines whether the outcome is above the hurdle, and the other models the value given it is above the hurdle.                                           | For data with a large number of zeros where the zero-generating process is distinct from the process generating positive values (e.g., number of fish caught).                    |

 

### Intercept Reference (`intercept()`)

The intercept represents the global baseline effect in the model. Its inclusion is controlled at the top level of the formula string, and the `intercept()` module provides a way to specify a custom prior for it.

| Syntax / Concept      | Example Usage                            | Parameters | Default Prior   | Meaning & Assumptions                                                                                                                                                                                                      |
| :----------------------| :-----------------------------------------| :-----------| :----------------| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Include Intercept** | `y ~ 1 + ...` or `y ~ intercept() + ...` | None       | `Normal(0, 10)` | Adds a global intercept term $\beta_0$ to the model's linear predictor. Assumes a constant baseline effect across all observations. The `1` syntax is standard R-style, while `intercept()` is the explicit `bstm` module. |
| **Exclude Intercept** | `y ~ 0 + ...` or `y ~ -1 + ...`          | None       | N/A             | Removes the global intercept term. The model is forced to pass through the origin. Assumes the response is zero when all predictors are zero.                                                                              |
| **Specify Prior**     | `intercept(prior=Normal(0, 5))`          | `prior`    | `Normal(0, 10)` | Uses the `intercept()` module to assign a specific prior distribution to the intercept term $\beta_0$.                                                                                                                     |

**Note on `intercept(0)`, `intercept(1)`, etc.:**

The inclusion or exclusion of the intercept is handled at the top level of the formula using the standard R-style syntax (`1` to include, `0` or `-1` to exclude). The `intercept()` module is specifically designed for one purpose: to allow the user to specify a custom prior for the intercept when it is included.


### Fixed Effects Reference (`+`)

Fixed effects represent the standard linear regression components of a model. They are specified directly in the formula using the `+` operator. `bstm` uses `StatsModels.jl` internally, so it supports standard R-style syntax.

| Syntax Element            | Example              | Meaning & Assumptions                                                                                                                                 | Use Case & Utility                                                                                                                                                          |
| :--------------------------| :---------------------| :------------------------------------------------------------------------------------------------------------------------------------------------------| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Intercept**             | `1` or `intercept()` | Adds a global intercept term ($\beta_0$). Assumes a baseline level for the response when all other predictors are zero.                               | Models the overall mean of the response. Almost always recommended unless the model is specified to pass through the origin.                                                |
| **Continuous Covariate**  | `+ x`                | Adds a linear term for the continuous variable `x` ($\beta_1 x$). Assumes a linear relationship between `x` and the response on the link scale.       | To model the linear effect of a continuous predictor (e.g., temperature, age) on the outcome.                                                                               |
| **Categorical Covariate** | `+ as_factor(c)`     | Treats the variable `c` as a categorical factor, creating dummy variables based on the specified contrast coding (default is dummy/treatment coding). | To model differences between discrete groups (e.g., regions, species). `as_factor()` ensures numeric variables are treated as categorical.                                  |
| **Interaction Term**      | `+ x1:x2`            | Adds an interaction term ($\beta_3 (x1 \cdot x2)$). Assumes the effect of `x1` on the response depends on the level of `x2`, and vice-versa.          | To model synergies or antagonisms where the combined effect of two variables is different from the sum of their individual effects.                                         |
| **Full Interaction**      | `+ x1*x2`            | A shorthand for `+ x1 + x2 + x1:x2`. Includes the main effects of both variables and their interaction term.                                          | The standard way to test for and model interactions while ensuring the main effects are also accounted for (principle of hierarchy).                                        |
| **Removing Terms**        | `- 1` or `- x`       | Removes the intercept or a specific term from the model.                                                                                              | `-1` is used for models that must pass through the origin. `-x` is less common but can be used to exclude a variable that would otherwise be included (e.g., from `x1*x2`). |

#### Priors for Fixed Effects
The coefficients ($\beta$) for all fixed effects are treated as random variables with a weakly informative prior to regularize the model and prevent overfitting.

*   **Default Prior**: `beta ~ MvNormal(0, 5.0 * I)`
*   **Meaning**: This is a multivariate normal distribution centered at zero with a diagonal covariance matrix. The standard deviation of 5.0 is generally considered weakly informative for data that has been standardized. It gently pulls coefficients towards zero unless there is strong evidence in the data for a large effect.
*   **Customization**: Currently, this prior is not directly configurable via the formula interface.



### Mixed Effects Reference (`mixed()`)

The `mixed()` module is used to specify random effects, also known as mixed effects or hierarchical effects. It allows model parameters (intercepts or slopes) to vary across different levels of a grouping variable. This is essential for modeling hierarchical or longitudinal data where observations within the same group are not independent.

| Syntax               | Example Usage    | Key Parameters | Default Priors | Mathematical Assumption           |                                                                                                                                                                                                                                                                                     |
| :---------------------| :-----------------| :---------------| :---------------| :----------------------------------| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Random Intercept** | `mixed(1         | group_var)`    | `model`        | `sigma_prior`: `Exponential(1.0)` | Assumes each level $j$ of `group_var` has a unique intercept $\alpha_j$ drawn from a common distribution, $\alpha_j \sim \mathcal{N}(0, \sigma^2_{\text{group}})$. The linear predictor for an observation $i$ in group $j$ is $\eta_i = \dots + \alpha_j$.                         |
| **Random Slope**     | `mixed(covariate | group_var)`    | `model`        | `sigma_prior`: `Exponential(1.0)` | Assumes the effect (slope) of a `covariate` varies across the levels of `group_var`. The slope for group $j$, $\beta_j$, is drawn from a common distribution, $\beta_j \sim \mathcal{N}(0, \sigma^2_{\text{slope}})$. The linear predictor is $\eta_i = \dots + \beta_j \cdot x_i$. |

### Spatial Manifold Reference (`spatial()`)

| Manifold             | `model='...'`       | Key Parameters                                      | Default PC-Priors                                               | Mathematical Assumption                                                                                                                                                                                                                                 | Use Case & Utility                                                                                                                                                                                 |                                                                                                                                                                                                                |
| :---------------------| :--------------------| :----------------------------------------------------| :----------------------------------------------------------------| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **IID**              | `'iid'`             | `sigma_prior`                                       | `Exponential(1.0)`                                              | Unstructured random effects, $\phi_i \sim \mathcal{N}(0, \sigma^2)$. Assumes no spatial correlation between areal units.                                                                                                                                | Models non-spatial overdispersion or heterogeneity. It serves as a base model for comparison and is a component of the BYM2 model.                                                                 |                                                                                                                                                                                                                |
| **ICAR / Besag**     | `'icar'`, `'besag'` | `sigma_prior`                                       | `Exponential(1.0)`                                              | Intrinsic Conditional Autoregressive model. The value at a location is conditional on the mean of its neighbors: $\phi_i                                                                                                                                | \phi_j, j \sim i \sim \mathcal{N}(\text{mean}(\phi_j), \sigma^2/n_i)$. Assumes a location is conditionally dependent only on its immediate neighbors.                                              | Provides strong, localized spatial smoothing. Ideal for data with clear neighborhood structures like disease maps or lattice data. The precision matrix is rank-deficient, requiring a sum-to-zero constraint. |
| **BYM2**             | `'bym2'`            | `sigma_prior`, `rho_prior`                          | `sigma`: `Exponential(1.0)`, `rho`: `Beta(1,1)`                 | Decomposes the spatial effect into a structured (ICAR) and an unstructured (IID) component: $\phi_i = \sqrt{\rho}\phi_{str} + \sqrt{1-\rho}\phi_{unstr}$. The parameter $\rho$ controls the proportion of variance attributed to the structured effect. | The most robust and recommended default for areal data. It separates true spatial clustering from random noise, improving model identifiability and interpretation.                                |                                                                                                                                                                                                                |
| **Leroux**           | `'leroux'`          | `sigma_prior`, `rho_prior`                          | `sigma`: `Exponential(1.0)`, `rho`: `Beta(1,1)`                 | A proper (non-singular) CAR model with precision $Q = \rho Q_{sp} + (1-\rho)I$. It's a convex combination of a spatial precision matrix ($Q_{sp}$) and an identity matrix, controlled by the mixing parameter $\rho$.                                   | A flexible alternative to BYM2 that avoids the rank-deficiency of the ICAR model. The parameter $\rho$ smoothly interpolates between a fully spatial model ($\rho=1$) and an IID model ($\rho=0$). |                                                                                                                                                                                                                |
| **SAR**              | `'sar'`             | `sigma_prior`, `rho_prior`                          | `sigma`: `Exponential(1.0)`, `rho`: `Beta(1,1)`                 | Simultaneous Autoregressive model. Assumes the response at a location is a linear function of the responses at neighboring locations: $\phi = \rho W \phi + \epsilon$, where $W$ is the row-standardized adjacency matrix.                              | Models spatial "spill-over" effects where the value at one location directly influences its neighbors. Common in econometrics and spatial regression.                                              |                                                                                                                                                                                                                |
| **SPDE**             | `'spde'`            | `sigma_prior`, `kappa_prior`                        | `sigma`: `Exponential(1.0)`, `kappa`: `Exponential(1.0)`        | Approximates a continuous Gaussian field with a Matérn covariance function as a solution to a Stochastic Partial Differential Equation on a mesh derived from the adjacency graph.                                                                      | A scalable and principled way to model continuous spatial processes on irregular domains. The `kappa` parameter is inversely related to the spatial range of the effect.                           |                                                                                                                                                                                                                |
| **Gaussian Process** | `'gp'`              | `sigma_prior`, `lengthscale_prior`, `kernel`        | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Models the spatial effect as a draw from a multivariate normal distribution with a covariance matrix defined by a kernel function, $K_{ij} = k(s_i, s_j)$, where distance is calculated between areal unit centroids.                                   | Gold-standard for continuous spatial modeling but computationally expensive ($O(N^3)$). Best for smaller numbers of areal units where a flexible, non-parametric fit is needed.                    |                                                                                                                                                                                                                |
| **RFF**              | `'rff'`             | `sigma_prior`, `lengthscale_prior`, `n_features`    | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Approximates a GP kernel using Random Fourier Features on the areal unit centroids: $\phi(x) = \sqrt{2/M}\cos(Wx+b)$.                                                                                                                                   | A scalable approximation to a full GP. Excellent for large numbers of areal units where the $O(N^3)$ cost of a full GP is prohibitive.                                                             |                                                                                                                                                                                                                |
| **FITC**             | `'fitc'`            | `sigma_prior`, `lengthscale_prior`, `n_inducing`    | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Fully Independent Training Conditional. A sparse GP method that uses a small set of inducing points to approximate the full GP on the centroids.                                                                                                        | Another scalable GP approximation. Often better at capturing long-range dependencies than RFFs for a given number of features/points.                                                              |                                                                                                                                                                                                                |
| **Nyström**          | `'nystrom'`         | `sigma_prior`, `lengthscale_prior`, `n_inducing`    | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | A low-rank approximation of the kernel matrix using a subset of inducing points (centroids).                                                                                                                                                            | Similar to FITC, provides a scalable GP approximation.                                                                                                                                             |                                                                                                                                                                                                                |
| **DAG**              | `'dag'`             | `sigma_prior`, `adjacency_matrix`                   | `Exponential(1.0)`                                              | Directed Acyclic Graph model. Assumes a causal or directional dependency structure between locations based on a specified ordering.                                                                                                                     | Used for processes with a clear directional flow, like river networks or causal inference problems. Computationally efficient due to its recursive structure.                                      |                                                                                                                                                                                                                |
| **NetworkFlow**      | `'network'`         | `sigma_prior`, `adjacency_matrix`, `flow_direction` | `Exponential(1.0)`                                              | Models processes on a directed graph, accounting for upstream or downstream effects based on the provided directed adjacency matrix.                                                                                                                    | Specifically designed for river networks, supply chains, or other systems with explicit flow dynamics.                                                                                             |                                                                                                                                                                                                                |


### Temporal & Seasonal Manifold Reference (`temporal()`)

The `temporal()` module is used to model trends and periodic patterns over time. It can handle both main temporal effects and seasonal cycles, often within the same module call.

| Manifold              | `model='...'` | Key Parameters                                            | Default PC-Priors                                                             | Mathematical Assumption                                                                                                                                                                                    | Use Case & Utility                                                                                                                                                                            |          |                                                                                                                           |
| :----------------------| :--------------| :----------------------------------------------------------| :------------------------------------------------------------------------------| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ----------| ---------------------------------------------------------------------------------------------------------------------------|
| **IID**               | `'iid'`       | `sigma_prior`                                             | `Exponential(1.0)`                                                            | Unstructured random effects, $\phi_t \sim \mathcal{N}(0, \sigma^2)$. Assumes no temporal correlation between time points.                                                                                  | Models unstructured temporal noise or serves as a baseline for comparison. Rarely used as a primary trend model.                                                                              |          |                                                                                                                           |
| **AR1**               | `'ar1'`       | `sigma_prior`, `rho_prior`                                | `sigma`: `Exponential(1.0)`, `rho`: `Beta(1,1)`                               | First-Order Autoregressive process, $\phi_t = \rho \phi_{t-1} + \epsilon_t$. Assumes the current state is a fraction of the previous state plus noise. Models stationary processes with short-term memory. | Modeling serially correlated time series where the influence of past events decays geometrically. Common for economic data or environmental time series with memory.                          |          |                                                                                                                           |
| **Random Walk (RW1)** | `'rw1'`       | `sigma_prior`                                             | `Exponential(1.0)`                                                            | First-Order Random Walk, $\phi_t = \phi_{t-1} + \epsilon_t$. The change between time steps is white noise. Models non-stationary processes with stochastic level shifts.                                   | Capturing abrupt changes or step-like trends. Useful for modeling processes where the level can change suddenly and unpredictably.                                                            |          |                                                                                                                           |
| **Random Walk (RW2)** | `'rw2'`       | `sigma_prior`                                             | `Exponential(1.0)`                                                            | Second-Order Random Walk, $\Delta^2 \phi_t = \epsilon_t$. The change in the slope is white noise. Models smooth, locally linear trends.                                                                    | The most common and robust choice for modeling smooth, non-linear temporal trends. It penalizes deviations from a straight line, making it ideal for capturing underlying long-term patterns. |          |                                                                                                                           |
| **Gaussian Process**  | `'gp'`        | `sigma_prior`, `lengthscale_prior`, `kernel`              | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)`               | The temporal effect is a draw from a multivariate normal distribution with a covariance matrix defined by a kernel function, $K_{ij} = k(t_i, t_j)$.                                                       | Gold-standard for flexible, non-parametric trend modeling. Computationally expensive ($O(T^3)$) but captures complex patterns without strong structural assumptions.                          |          |                                                                                                                           |
| **RFF**               | `'rff'`       | `sigma_prior`, `lengthscale_prior`, `n_features`          | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)`               | Approximates a GP kernel using a linear model on random Fourier basis functions: $\phi(t) = \sqrt{2/M}\cos(Wt+b)$.                                                                                         | A scalable approximation to a full GP, suitable for very long time series where the cost of a full GP is prohibitive.                                                                         |          |                                                                                                                           |
| **Cyclic**            | `'cyclic'`    | `sigma_prior`, `period`                                   | `Exponential(1.0)`                                                            | A random walk on a circular graph, where the last time point is a neighbor of the first. $\phi_T \sim \mathcal{N}(\text{mean}(\phi_{T-1}, \phi_1), \sigma^2/2)$.                                           | Modeling smooth, periodic effects like day-of-week or month-of-year where the end of the cycle connects back to the beginning.                                                                |          |                                                                                                                           |
| **Harmonic**          | `'harmonic'`  | `amplitude_prior`, `phase_prior`, `sigma_prior`, `period` | `amplitude`: `Normal(0,1)`, `phase`: `Beta(1,1)`, `sigma`: `Exponential(1.0)` | The seasonal effect is a sum of sine and cosine waves (a Fourier series): $\phi(t) = \sum_j A_j \sin(2\pi j t/P + \theta_j)$.                                                                              | Capturing sharp, regular periodic patterns. More rigid than a cyclic random walk but computationally efficient.                                                                               |          |                                                                                                                           |
| **Exponential Decay** | `'decay'`     | `sigma_prior`, `decay_lengthscale_prior`                  | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)`               | Models a process where the correlation between time points decays exponentially with their distance: $K_{ij} = \sigma^2 \exp(-                                                                             | t_i - t_j                                                                                                                                                                                     | /\ell)$. | Useful for processes with a strong, rapidly decaying memory, such as financial time series or certain physical phenomena. |

### Smooth Manifolds Reference (`smooth()`)

The `smooth()` module is used to model non-linear effects of continuous covariates.

| Manifold / Method           | `model='...'`                                        | Key Parameters                                 | Default Priors                                                        | Mathematical Assumption                                                                                                                                                      | Use Case & Utility                                                                                                                                           |
| :----------------------------| :-----------------------------------------------------| :-----------------------------------------------| :----------------------------------------------------------------------| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------| :-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **P-Spline**                | `'pspline'`                                          | `nbins`, `degree`, `diff_order`                | `sigma_prior`: `Exponential(1.0)`                                     | The effect is a linear combination of B-spline basis functions, with a random walk penalty on the basis coefficients to enforce smoothness.                                  | The most common and flexible general-purpose smoother for 1D covariates. `diff_order=2` (the default) penalizes deviations from a straight line.             |
| **B-Spline**                | `'bspline'`                                          | `nbins`, `degree`                              | `sigma_prior`: `Exponential(1.0)`                                     | The effect is a linear combination of B-spline basis functions with IID coefficients. Assumes the basis functions themselves provide sufficient smoothness.                  | A simpler spline smoother than P-splines. Useful when less regularization is desired.                                                                        |
| **Thin Plate Spline**       | `'tps'`                                              | `nbins`                                        | `sigma_prior`: `Exponential(1.0)`                                     | The basis functions are derived from a kernel that minimizes a bending energy penalty, $k(r) = r^2 \log(r)$.                                                                 | The classic choice for smoothing 2D spatial coordinates (e.g., `smooth(lon, lat, model=tps)`). It is isotropic (direction-agnostic).                         |
| **Random Fourier Features** | `'rff'`                                              | `n_features` (or `m_rff`), `lengthscale_prior` | `sigma_prior`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Approximates a stationary Gaussian Process kernel (e.g., Squared Exponential) by projecting the covariate into a randomized feature space: $\phi(x) = \sqrt{2/M}\cos(Wx+b)$. | A highly scalable method for approximating a full Gaussian Process smooth. Excellent for very large datasets where the $O(N^3)$ cost of a GP is prohibitive. |
| **Random Walk (on bins)**   | `'rw1'`, `'rw2'`                                     | `nbins`                                        | `sigma_prior`: `Exponential(1.0)`                                     | Discretizes the continuous covariate into `nbins` ordered categories and applies a first-order (`rw1`) or second-order (`rw2`) random walk GMRF to the bin indices.          | A powerful way to model a non-linear effect as a structured random effect. It is a discrete approximation to a continuous smooth, often used in INLA.        |
| **ICAR (on bins)**          | `'icar'`, `'besag'`                                  | `nbins`                                        | `sigma_prior`: `Exponential(1.0)`                                     | Discretizes the covariate and applies a spatial ICAR model to the bin indices, assuming adjacent bins are correlated.                                                        | Less common for smoothing a single covariate, but can be used to model effects where adjacent bins have a neighborhood-like dependency.                      |
| **Interaction Smooth**      | `'icar ⊗ ar1'`  `NOTE: direct interaction is better` | `nbins` (can be a tuple, e.g., `(10, 12)`)     | `sigma_prior`: `Exponential(1.0)`                                     | Discretizes two covariates into a 2D grid and applies a tensor product of two GMRF precision matrices (e.g., a spatial ICAR and a temporal AR1) to the grid indices.         | Models a smooth, non-linear interaction surface between two continuous covariates. For example, `smooth(temperature, depth, model=rw2 ⊗ rw2)`.               |
| **FFT Basis**               | `'fft'`                                              | `nbins`                                        | `sigma_prior`: `Exponential(1.0)`                                     | The effect is a linear combination of sine and cosine basis functions (a Fourier series).                                                                                    | Best for modeling periodic or cyclical non-linear effects of a covariate.                                                                                    |
| **Moran's I Basis**         | `'moran'`                                            | `nbins`                                        | `sigma_prior`: `Exponential(1.0)`                                     | Uses eigenvectors of a spatial weights matrix as basis functions to capture effects at different spatial scales. (Proxy used in `bstm`).                                     | Advanced use for filtering or modeling effects at specific spatial frequencies.                                                                              |
| **Spherical Basis**         | `'spherical'`                                        | `nbins`, `range`                               | `sigma_prior`: `Exponential(1.0)`                                     | Uses a kernel with compact support, meaning the correlation between points is exactly zero beyond a specified `range`.                                                       | Useful when the influence of a covariate is known to be strictly local.                                                                                      |
| **Exponential Decay Basis** | `'decay'`                                            | `nbins`, `lengthscale`                         | `sigma_prior`: `Exponential(1.0)`                                     | Uses an exponential decay kernel, where correlation drops off exponentially with distance between covariate values.                                                          | Models effects with a strong, rapidly decaying influence.                                                                                                    |
| **Barycentric Basis**       | `'barycentric'`                                      | `nbins`                                        | `sigma_prior`: `Exponential(1.0)`                                     | Creates a piecewise linear (hat function) basis. The effect is interpolated linearly between knots.                                                                          | A simple, interpretable smoother that assumes the effect is linear between specified points.                                                                 |



#### Spacetime Interaction Manifolds 

** Direct expression of Interaction:** 

Spatio-Temporal Interaction are better expressed as: `y ~ 1 + spatial(s_idx; model=besag) ⊗ temporal(t_idx; model=ar1)`. This employs tensor-product logic to manage O(N \times T) complexity via the Type IV interaction template. 

| Manifold     | formula                                                    | Description                                                  |
| :-------------| :-----------------------------------------------------------| :-------------------------------------------------------------|
| **Type I**   | `spatial(s_idx; model=iid) ⊗ temporal(t_idx; model=iid)`   | Unstructured (IID) interaction over space and time.          |
| **Type II**  | `spatial(s_idx; model=iid) ⊗ temporal(t_idx; model=ar1)`   | Spatially unstructured, temporally structured.               |
| **Type III** | `spatial(s_idx; model=besag) ⊗ temporal(t_idx; model=iid)` | Spatially structured, temporally unstructured.               |
| **Type IV**  | `spatial(s_idx; model=besag) ⊗ temporal(t_idx; model=ar1)` | Fully structured in both space and time (Kronecker product). |
 


### Nested & Multi-fidelity Reference (`nested()`)

The `nested()` module is a powerful "supervisor" component used for multi-fidelity modeling and model stacking. It allows you to define a complete sub-model that is fit to a separate (often larger, lower-quality) dataset. The latent effect from this sub-model is then incorporated as a calibrated predictor into the main model, allowing the main model to "learn" from the proxy data.

| Keyword / Parameter     | Example Usage                  | Data Type | Default            | Meaning & Assumptions                                                                                                                                                                                                                                 |
| :------------------------| :-------------------------------| :----------| :-------------------| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nested()`              | `nested(z_var; ...)`           | Module    | N/A                | Defines a supervised sub-model whose latent effect is added to the main model's linear predictor. The `z_var` is a symbolic name for this component.                                                                                                  |
| `formula`               | `formula="z ~ 1 + spatial(s)"` | `String`  | `""`               | A complete `bstm` formula string that defines the structure of the sub-model. This sub-model is fit to the specified `data_source`.                                                                                                                   |
| `data_source`           | `data_source=:proxy_data`      | `Symbol`  | `:data`            | A symbol pointing to a `DataFrame` passed as a keyword argument to the main `bstm()` call. This allows the sub-model to use a different dataset.                                                                                                      |
| `rho_nested` (Implicit) | N/A                            | `Float`   | `Normal(1.0, 0.5)` | A scaling coefficient that links the sub-model's latent effect to the main model's linear predictor: $\eta_{\text{main}} = \dots + \rho_{\text{nested}} \cdot \eta_{\text{sub}}$. The prior assumes the sub-model is a good proxy ($\rho \approx 1$). |


### Eigen & Factor Model Reference (`eigen()`)

The `eigen()` module implements a Bayesian Principal Component Analysis (PCA) to perform dimensionality reduction on a set of multivariate outcomes. It decomposes the input variables into a smaller set of orthogonal latent factors (principal components). The first of these factors is then added to the main model's linear predictor, allowing you to use the dominant shared signal from multiple variables as a predictor.

| Keyword / Parameter | Example Usage                    | Data Type      | Default            | Meaning & Assumptions                                                                                                                                                               |
| :--------------------| :---------------------------------| :---------------| :-------------------| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `eigen()`           | `eigen(y1, y2, y3; ...)`         | Module         | N/A                | Defines a Bayesian PCA factor model. The variables listed (e.g., `y1, y2, y3`) are the multivariate outcomes to be decomposed.                                                      |
| `n_factors`         | `n_factors=1`                    | `Int`          | `1`                | The number of latent factors (principal components) to extract. This determines the dimensionality of the reduced latent space.                                                     |
| `pca_sd_prior`      | `pca_sd_prior=Exponential(0.5)`  | `Distribution` | `Exponential(1.0)` | The prior for the standard deviations of the principal components (latent factors). These are the "eigenvalues" of the system, controlling the variance explained by each factor.   |
| `pdef_sd_prior`     | `pdef_sd_prior=Exponential(0.5)` | `Distribution` | `Exponential(1.0)` | The prior for the standard deviation of the residual (uniqueness) noise. This captures the variance in each observed variable that is *not* explained by the shared latent factors. |



#### The Householder PCA Mechanism

The `eigen()` module uses a Householder transformation to construct the orthonormal loadings matrix ($U$) for the PCA. This is a key feature for ensuring numerical stability and efficient sampling in a Bayesian context.

*   **Why Householder?** Directly sampling an orthonormal matrix is difficult. Traditional methods like Gram-Schmidt orthogonalization are not differentiable and can be numerically unstable within an MCMC sampler like NUTS. The Householder method provides a solution by parameterizing the orthonormal matrix indirectly.
*   **How it Works:** Instead of sampling the $N \times K$ loadings matrix $U$ directly, the model samples a much smaller vector of unconstrained parameters, `v`. These parameters define a sequence of Householder reflections. A Householder reflection is a linear transformation that reflects a vector about a hyperplane. By composing a series of these reflections, we can construct any arbitrary orthonormal matrix.
*   **The Math:** A single reflection is defined by a matrix $H = I - 2vv^T$, where $v$ is a unit vector. The full loadings matrix $U$ is built by recursively applying these reflections: $U_{final} = H_K \dots H_2 H_1 I$.
*   **Benefit:** This process is fully differentiable and allows the sampler to explore the space of orthogonal matrices efficiently and without the geometric constraints that would otherwise make sampling intractable. The model learns the optimal rotation of the latent space by learning the optimal set of reflector vectors `v`.



### Dynamics & State-Space Model Reference (`dynamics()`)

The `dynamics()` module is used to implement mechanistic state-space models that describe how a latent field evolves over time. This is the primary module for encoding process-based knowledge, such as population growth or physical transport, directly into the model.

| Model                            | `model='...'`                    | Key Parameters                                                                     | Default Priors                                                                                           |
| :---------------------------------| :---------------------------------| :-----------------------------------------------------------------------------------| :---------------------------------------------------------------------------------------------------------|
| **Logistic Growth with Fishing** | `'logistic_fishing'`, `'ricker'` | `r_prior`, `K_prior`, `sig_pop_prior`, `sig_F_prior`, `r_covariate`, `K_covariate` | `r`: `LogNormal(0,1)`, `K`: `Normal(150,50)`, `sig_pop`: `Exponential(1.0)`, `sig_F`: `Exponential(0.5)` |
| **Gompertz Growth**              | `'gompertz'`                     | `r_prior`, `K_prior`, `sig_dyn_prior`                                              | `r`: `LogNormal(-1.5,0.5)`, `K`: `Normal(150,50)`, `sig_dyn`: `Exponential(1.0)`                         |
| **Linked-K Logistic Growth**     | `'linked_K_logistic'`            | `r_prior`, `sig_pop_prior`, `K_slope_prior`                                        | `r`: `LogNormal(0,1)`, `sig_pop`: `Exponential(1.0)`, `K_slope`: `Normal(1,0.5)`                         |
| **Advection**                    | `'advection'`                    | `velocity_prior`, `sigma_prior`                                                    | `velocity`: `Normal(0,0.5)`, `sigma`: `Exponential(1.0)`                                                 |
| **Diffusion**                    | `'diffusion'`                    | `diffusion_prior`, `sigma_prior`                                                   | `diffusion`: `LogNormal(-1,1)`, `sigma`: `Exponential(1.0)`                                              |
| **Advection-Diffusion**          | `'advection_diffusion'`          | `velocity_prior`, `diffusion_prior`, `sigma_prior`                                 | `velocity`: `Normal(0,0.5)`, `diffusion`: `LogNormal(-1,1)`, `sigma`: `Exponential(1.0)`                 |
| **Custom Model**                 | `'custom'`                       | `func`                                                                             | N/A                                                                                                      |







 
###  Partitioning the Map: Areal Units and Information Balance

For discrete *bstm*s, we must first discretize the spatial domain into "Areal Units" (AUs).   

Any will do. But in *bstm*, the following are available (:avt is default). 

The `assign_spatial_units` function is a tool for partitioning a study area into discrete spatial units based on point patterns, pre-defined boundaries, or regular lattices. This process is essential for creating structured spatial random effects in spatiotemporal models (e.g., CAR, BYM2).

Available Partitioning Methods (`area_method`):

| Method           | Description                         | Mathematical Assumption / Justification                                                                                                                                                                         |
| :-----------------| :------------------------------------| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `:cvt`           | **Centroidal Voronoi Tessellation** | Assumes the area should be divided into cells where each unit is the geometric centroid of its Voronoi region. Lloyd's algorithm iteratively minimizes the variance of the partition.                           |
| `:kvt`           | **K-Means Voronoi Tessellation**    | Uses a weighted K-Means approach. It justifies boundaries based on point density rather than just geometry, ensuring units have a balanced number of observations.                                              |
| `:qvt` / `:bvt`  | **Quadtree / Binary Split**         | Employs recursive spatial partitioning. Splits a parent region if it violates constraints (e.g., `max_points`). Best for handling extreme density gradients where some areas are sparse and others hyper-dense. |
| `:hvt`           | **Hierarchical Voronoi**            | Combines K-Means seeding with Lloyd's refinement. It justifies unit placement by ensuring stable centroid positions that satisfy geometric and count constraints.                                               |
| `:avt`           | **Agglomerative Voronoi**           | Starts with a high-resolution mesh and merges units. Justified for small datasets where preventing 'starved' units (zero observations) is a priority.                                                           |
| `:lattice`       | **Regular Grid**                    | Discretizes space into uniform squares ($L \times L$). Assumes spatial stationarity across a regular grid, simplifying adjacency calculations to a Queen's contiguity check.                                    |
| `:user_polygons` | **External Shapefiles**             | Uses provided `LibGEOS` polygons. Justified when political or administrative boundaries (e.g., census tracts) define the spatial process.                                                                       |



**Geometric Clipping (`geom_hull`)**:

When a `geom_hull` is provided, the function performs a spatial intersection: 
$$ P_{clipped} = P_{tessellated} \cap H_{hull} $$
This ensures that generated units do not extend into invalid areas (e.g., water bodies or outside the study region).

**Adjacency and Graph Theory**:

Connectivity is determined by a topological check. Two units $i$ and $j$ are adjacent if:
1.  They share a boundary: `LibGEOS.touches(P_i, P_j) == true`.
2.  They are within a numerical tolerance: `LibGEOS.intersects(buffer(P_i, 1e-7), P_j) == true`.

**Justification for Adjacency Transformation:**

The resulting `SimpleGraph` is used to construct the Graph Laplacian $\mathbf{Q}$ for Intrinsic Conditional Autoregressive (ICAR) models:
$$ \mathbf{Q} = \mathbf{D} - \mathbf{W} $$
where $\mathbf{D}$ is the degree matrix and $\mathbf{W}$ is the binary adjacency matrix.

**Global Connectivity Constraint**:

If the tessellation results in disjoint 'islands', the `ensure_connected!` utility adds edges between the nearest centroids of different components. This satisfies the mathematical assumption of a single connected component required for stable ICAR precision matrix inversion.


#### Controlling paramters: assign_spatial_units()

*   **`t_idx` and `u_idx`:** These indices link each observation to its respective time and seasonal unit, allowing for spatiotemporal modeling.
*   **`target_units`:** The desired resolution of the spatial model. A higher number captures finer local variation but increases computational cost and risk of data sparsity.
*   **`target_cv`:** A target coefficient of variation, often for the number of data points per areal unit, used to balance unit sizes.
*   **`min_total_arealunits`, `max_total_arealunits`:** Constraints on the total number of areal units created.
*   **`min_time_slices`:** Ensures each areal unit has a minimum number of time observations.
*   **`buffer_dist`:** Used in methods like `:hvt` to define a buffer zone for identifying neighbors or influencing centroid placement.
*   **`tolerance`:** Defines the convergence criteria for iterative methods (CVT/HVT). Stability is reached when the movement of centroids is negligible.
*   **`min_points`, `max_points`:** Ensures that each spatial unit contains enough information for the likelihood to be 'identifiable'. Units failing this are merged or split depending on the method.
*   **`min_area`, `max_area`:** Constraints on the geographic area of each areal unit.


#### (Adaptive) Centroidal Voronoi Tessellation (CVT)

CVT is a popular method designed for uniform statistical power. It uses Lloyd's algorithm to create a highly regular, "honeycomb" mesh. Data density is rarely uniform and so an Adaptive form of CVT uses Kernel Density Estimation (KDE) to migrate seeds toward density modes (peaks). This shrinks tiles in high-activity areas and stretches them in sparse areas, ensuring every unit is informative and minimizing Boundary Artifacts that occur when standard tiles split high-density clusters.


**Algorithm:** Lloyd's Algorithm
1. **Initialization:** Generate $K$ initial seeds $\{c_1, ..., c_k\}$ (using K-means or random sampling).
2. **Partitioning:** Construct the Voronoi cells $V_i$ such that $V_i = \{x \in \Omega \mid d(x, c_i) \le d(x, c_j) \forall j \}$.
3. **Centroid Update:** For each cell, calculate the geometric centroid $c_i^* = \frac{1}{A} \iint_{V_i} (x, y) dA$.
4. **Refinement:** Move $c_i \to c_i^*$.
5. **Convergence:** Repeat until the L2-norm of the shift vector $\sum ||c_i - c_i^*||$ is less than `tolerance`.
 
**Advantages:**

*   **Information Balance**: Units in sparse areas are allowed to be larger, while units in dense areas are smaller, balancing the sample size per unit.
*   **Rapid Convergence**: Starting with seeds already near the data mass significantly reduces the number of iterations required for the tessellation to stabilize.
*   **Regularity**: Maintains the high geometric compactness of standard Voronoi cells, which minimizes the distance between observations and their assigned centroids.

**Assumptions:**
*   The desired areal units should be as compact and equitably sized as possible.
*   The 'center' of each unit (its centroid) should coincide with the mean position of the points it contains.
*   Often used when optimizing for uniform coverage or resource allocation.



#### Binary Vector Tree (BVT)

BVT is a recursive splitting method along the axis of maximum variance. It is the fastest approach, ideal for datasets with millions of points. The **KDE-Informed Binary Vector Tree (BVT)** is a high-speed hierarchical partitioning method. While standard BVT focuses strictly on variance-based splitting to balance point counts, the KDE-informed variant incorporates the underlying data intensity surface to ensure that partitions are not only balanced in count but also aligned with the statistical topology of the domain.
 
**Algorithm:** Spatial Binary Tree
1. **Evaluation:** Calculate the number of points $N_p$ in the current region $\Omega_{parent}$.
2. **Decision:** If $N_p > \text{max\_points}$ and the split doesn't violate $\text{min\_area}$, divide the region.
3. **Splitting Mechanics:** Split along the dimension with the highest variance (Principal Component split).
4. **Pruning:** Discard child regions that contain fewer than `min_points` or have zero distinct `time_slices`.
5. **Termination:** Continue until all regions satisfy constraints or `max_total_arealunits` is reached.

Advantages:
*   **Computational Efficiency**: BVT is significantly faster than iterative methods like CVT (Lloyd's algorithm), making it the preferred choice for massive datasets.
*   **Perfect Count Balancing**: Because it is based on a tree structure, it is highly effective at creating units with nearly identical sample sizes, which stabilizes the estimation of local variance parameters.
*   **Scalability**: The recursive nature allows for rapid partitioning of millions of points into thousands of units in logarithmic time $O(N \log N)$.

**Assumptions:**
*   The primary goal is to create Voronoi cells that are approximately equal in terms of a specific metric (e.g., number of points, area, population).
*   Ensuring an equitable distribution of data or resources among units is critical.



#### Quadrant Voronoi Tessellation (QVT)

QVT is a quadtree-like decomposition that excels at capturing multi-scale spatial clusters and density transitions.
The **KDE-Informed Quadrant Voronoi Tessellation (QVT)** is a multi-scale hierarchical partitioning method. It combines the structured efficiency of a quadtree decomposition with the statistical sensitivity of Kernel Density Estimation (KDE) to ensure that the recursive subdivision reflects the underlying data topology.
 
**Algorithm:** Spatial Quadtree
1. **Evaluation:** Calculate the number of points $N_p$ in the current region $\Omega_{parent}$.
2. **Decision:** If $N_p > \text{max\_points}$ and the split doesn't violate $\text{min\_area}$, divide the region.
3. **Splitting Mechanics:** Split at the median $(x, y)$ coordinates into four quadrants.
4. **Pruning:** Discard child regions that contain fewer than `min_points` or have zero distinct `time_slices`.
5. **Termination:** Continue until all regions satisfy constraints or `max_total_arealunits` is reached.

Advantages:
*   **Multi-Scale Sensitivity**: QVT is exceptionally good at automatically adjusting its resolution, creating fine-grained units in high-density urban areas and coarse units in sparse rural areas.
*   **Hierarchical Diagnostics**: Because it follows a tree structure, it allows for easy multi-level spatial modeling (e.g., nesting local effects within regional quadrants).
*   **Void Handling**: By guiding splits via the KDE surface, it naturally avoids creating 'empty' units that can lead to singular precision matrices in Bayesian samplers.


**Assumptions:**
*   Spatial data exhibits varying densities, and an adaptive resolution is desired.
*   Computational efficiency for spatial queries and hierarchical indexing is important.
*   Combines the advantages of quadtrees (hierarchical decomposition) with Voronoi diagrams (proximity-based partitioning).



#### Agglomerative Voronoi Tessellation (AVT)

AVT is an iterative merging approach that balances multiple constraints upon the data that iteratively aggregates small areal units until stopping rules are met. It also begins with KDE to identify initial conditions. The **Agglomerative Voronoi Tessellation (AVT)** is a data-driven spatial partitioning strategy designed specifically to solve the 'Data Starvation' problem in Bayesian inference. Unlike recursive splitting methods (BVT/QVT) which start from a global domain, AVT operates from the 'bottom-up'.
  
**Assumptions:**

*   The density or importance of data points varies significantly across the study area.
*   Finer resolution is needed in high-density areas, while coarser resolution is acceptable in low-density areas.
*   Similar to `:hvt` but often with more sophisticated adaptation criteria.

**Algorithm:** Hierarchical Clustering with Voronoi Constraints

1. **Initialization:** Start with a dense set of units (over-partitioned).
2. **Audit:** Identify 'starved' units that violate `min_points` or `min_time_slices`.
3. **Neighbor Selection:** For each violating unit $i$, find the adjacent unit $j \in \text{neighbors}(i)$ that minimizes the change in the coefficient of variation ($CV$) of point counts.
4. **Merge:** Redefine the new centroid $c_{new} = \frac{N_i c_i + N_j c_j}{N_i + N_j}$ and dissolve the shared boundary.
5. **Termination:** Stop when all units satisfy the minimum constraints. 

**Advantages:**

* **Guaranteed Identifiability**: By enforcing `min_pts`, AVT ensures there is enough local information to identify complex latent parameters like the temporal correlation $\rho$.
* **Topological Integrity**: Because it is based on Voronoi geometry, it maintains a valid adjacency graph (W) required for GMRF precision matrices.
* **Robustness**: Ideal for sparse or irregularly distributed point sets where standard grid-based methods would produce 'empty' units that crash the Bayesian sampler.
  

#### Hierarchical Voronoi Tessellation (HVT)

**Philosophy:** Geometry-driven partitioning with data-aware splitting. It seeks to create 'well-behaved' (equilateral/hexagonal-like) polygons to ensure stable Queen's contiguity graphs.

**The Algorithm:**
1.  **Objective Function:** Minimize the variance of the geometric distribution (quantization error) over the continuous study area $\Omega$:
    $$\mathcal{H} = \sum_{i=1}^{k} \int_{V_i} \rho(x) ||x - c_i||^2 dx$$
    where $\rho(x)$ is typically uniform.
2.  **Lloyd’s Update (The Distinguisher):** Unlike KVT, the centroid is moved to the **geometric center of the clipped polygon**:
    $$c_i^{(t+1)} = \text{centroid}(\text{Polygon}_i \cap \text{Hull})$$
3.  **Adaptive Splitting (The Hierarchy):** After geometric convergence, each unit is audited against a density threshold (`max_points`). If a unit violates the threshold:
    - It is split into two or more sub-units.
    - The system re-runs Lloyd's algorithm locally to re-stabilize the new centroids.
4.  **Constraint Enforcement:** It explicitly checks for `min_area` to prevent 'sliver' polygons which can create unstable edges in the adjacency matrix $\mathbf{W}$.
5.  **Result:** A more uniform spatial grid than KVT, but one that adaptively increases resolution only where geometrically and statistically necessary.

**Assumptions:**
*   Spatial data exhibits clustering or non-uniform density.
*   The objective is to create roughly balanced areal units in terms of number of points or other metrics, while preserving spatial contiguity.
*   Distance is a key factor in defining neighborhood and unit boundaries.
 


#### Regular Grid (grid)

**Assumptions:**
*   Spatial effects are relatively uniform across the study area.
*   A simple, regular partitioning is sufficient for the analysis.
*   Computational efficiency is a priority, as grid-based operations are often faster.

**Algorithm:**
1.  **Bounding Box:** Determine the minimum and maximum coordinates (bounding box) of the entire spatial dataset.
2.  **Grid Definition:** Divide the bounding box into a regular grid of cells (e.g., squares or hexagons) of a predefined size or number of rows/columns.
3.  **Point Assignment:** Assign each data point to the grid cell it falls within.
4.  **Areal Unit Creation:** Each grid cell (or a collection of adjacent cells, if further aggregation is desired) forms an areal unit.


#### K-Means based Voronoi Tesselation (KVT)

**Philosophy:** Data-driven partitioning. It treats the spatial units as clusters of observations, ensuring each unit has sufficient data for robust parameter estimation.

**The Algorithm:**
1.  **Objective Function:** Minimize the Within-Cluster Sum of Squares (WCSS):
    $$J = \sum_{i=1}^{k} \sum_{x \in S_i} ||x - \mu_i||^2$$
    where $x$ are the observed point coordinates and $\mu_i$ is the mean of points in cluster $S_i$.
2.  **Assignment:** Each observation point is assigned to the nearest current centroid.
3.  **Update Step (The Distinguisher):** The new centroid is calculated as the **arithmetic mean of the observations** assigned to that unit:
    $$c_{i}^{(t+1)} = \frac{1}{|S_i^{(t)}|} \sum_{x_j \in S_i^{(t)}} x_j$$
4.  **Damping:** A damping factor $\eta$ is often applied to prevent chaotic boundary shifts in sparse areas: $c_{new} = (1-\eta)c_{old} + \eta c_{mean}$.
5.  **Result:** Areas with high point density receive more (and smaller) spatial units, while sparse areas are aggregated into larger units. This naturally prevents 'data starvation' in the resulting ICAR model.


**Assumptions:**
*   k-means clusters are interpreted as initial Voronoi cells.
*   Areal units can be defined by clusters of points that are geometrically close to each other.
*   The number of clusters (areal units) is known or can be estimated beforehand.
*   The clusters are expected to be roughly spherical and of similar size.

 

####  Predefined Areal Units 

**Assumptions:**
*   A set of existing, well-defined administrative or ecological boundaries (e.g., counties, census tracts, watersheds) are appropriate for the analysis.
*   The data points can be accurately assigned to these predefined units.

**Algorithm:**
1.  **Input Units:** The method takes a set of pre-existing polygon geometries (e.g., shapefiles) representing the areal units.
2.  **Spatial Join:** Each data point is spatially joined (e.g., using point-in-polygon test) to determine which predefined areal unit it belongs to.
3.  **W Matrix Construction:** The adjacency matrix `W` is constructed based on the contiguity of these predefined units (e.g., queen contiguity or rook contiguity).

If using proprietary shapefile formats, you can use something like the following to read in the shapefile:

```{julia}
import Shapefile  << --- install this if you need it
import GeoInterface
import LibGEOS

fn = "filename of shapefile"

plygn = load_shapefile_to_libgeos(fn)
```


####  Force-Directed Layout (Inferred Adjacency)

This was used for the Scottish Lip cancer data. It is a means to quickly assess geographical position when we do not have coordinates but only adjacency information. Its current forms simplistic and so should be refined with constraints if used for anything serious. 

**Algorithm:** Fruchterman-Reingold (Simplified)

1. **Repulsion:** All nodes exert an inverse-square force to prevent overlap.
2. **Attraction:** Nodes connected in the adjacency matrix $W$ exert a linear spring force $F = k(d - d_0)$.
3. **Update:** Shift centroids $c_i \to c_i + \eta \sum F_{total}$.
4. **Justification:** This creates a planar representation of a non-spatial graph, allowing Voronoi tessellation to be applied back to purely topological data.



---



## `bstm_multivariate` Model Documentation

### Conceptual Overview: bstm_multivariate - Multivariate Bayesian Space-Time Models

The `bstm_multivariate` model extends the univariate `bstm` framework to simultaneously model multiple related outcomes (e.g., different disease types, crime categories, or economic indicators) that share underlying spatiotemporal dynamics. It allows for flexible modeling of spatial, temporal, seasonal, and interaction effects for each outcome, while explicitly accounting for inter-outcome correlation using a multivariate normal prior on the effects.

This model is particularly useful when analyzing phenomena that are not independent but rather influenced by common or related latent processes across space and time. By modeling outcomes jointly, it can borrow strength across them, potentially leading to more robust estimates and a better understanding of shared and unique drivers of each outcome.

**Key Features:**

1.  **Multivariate Latent Fields:** Unlike univariate models that produce a single set of spatial/temporal effects, `bstm_multivariate` generates a set of effects for *each* observed outcome (e.g., `s_eta` will be a matrix where columns correspond to outcomes).
2.  **LKJ Correlation Prior:** Inter-outcome correlation is introduced through a Cholesky decomposition of an LKJ (Lewandowski, Kurowicka, Joe) prior. This prior is specifically designed for correlation matrices, ensuring positive definiteness and offering flexibility in modeling correlation structures.
3.  **Modular Component Structure:** Retains the modularity of `bstm`, allowing various choices for spatial, temporal, seasonal, and interaction manifolds for each outcome. This means each outcome can have its own `s_sigma`, `t_sigma`, `st_sigma`, `s_rho`, `t_rho`, etc., providing immense flexibility.
4.  **Flexible Likelihoods:** Supports multiple likelihood families (Gaussian, Log-Normal, Poisson, Binomial, Negative Binomial) for each outcome, with options for zero-inflation and stochastic volatility, enabling analysis of diverse data types within a single framework.


### Mathematical Logic of Multivariate & Multi-outcome BSTM

The multivariate framework extends the univariate logic by allowing $K$ outcomes to be correlated either in their magnitude or their spectral orientation.

#### 1. Magnitude Correlation (LKJ)
Outcomes can share information through a Cholesky-factored correlation matrix $L_{corr}$:
$$\eta_{total} = \eta_{latent} \cdot L_{corr}$$
where $L_{corr} \sim LKJ(1.0)$. This couples the variances across outcomes while maintaining distinct spatial structures.

#### 2. Spectral Orientation (Householder)
To allow outcomes to rotate in the latent space (spectral alignment), we apply a Householder reflection $H$:
$$H = I - 2\frac{vv^T}{v^Tv}$$
$$\eta_{rotated} = \eta_{latent} \cdot H$$
This is particularly useful for detecting 'aligned' signals in complex spatiotemporal manifolds like SAR or Transport models.
 

Let $Y_{i,k}$ be the observed data for the $i$-th spatiotemporal observation of the $k$-th outcome, where $k \in \{1, \ldots, N_{\text{obs}}\}$. The model assumes a multivariate structure where the linear predictor for each outcome $k$, denoted $\eta_k$, contributes to the likelihood of $Y_{i,k}$.

The total linear predictor for each outcome $k$ is given by:

$$\eta_{i,k} = \text{log\_offset}_{i,k} + s_{\text{eta}}[M.s_{\text{idx}}[i], k] + t_{\text{eta}}[M.t_{\text{idx}}[i], k] + \text{st_eta}[k][M.s_{\text{idx}}[i], M.t_{\text{idx}}[i]] + \sum_{j} c_{\text{eta}}[M.cov_{\text{indices}}[i, j], k] + u_{\text{eta}}[M.u_{\text{idx}}[i], k] + \sum_{p} \text{fixed_effect}_{i,p} \cdot \text{SVC}_{p}[M.s_{\text{idx}}[i], k] + \text{fixed_effect}_{i,p} \cdot d_{\text{beta}}[p, k]$$

Where components `s_eta`, `t_eta`, `st_eta`, `c_eta`, `u_eta` (and `svc_raw`, `d_beta` for fixed effects) are now matrices or arrays of matrices, with the last dimension corresponding to the outcome `k`. The key difference from the univariate model is the introduction of outcome-specific parameters and the LKJ correlation.

**1. Setup & Hyperpriors:**

-   `s_sigma` $\sim \text{filldist}(\text{Exponential}(1.0), N_{\text{obs}})$:
    -   Outcome-specific marginal standard deviations for spatial effects.
-   `t_sigma` $\sim \text{filldist}(\text{Exponential}(1.0), N_{\text{obs}})$:
    -   Outcome-specific marginal standard deviations for temporal effects.
-   `st_sigma` $\sim \text{filldist}(\text{Exponential}(0.5), N_{\text{obs}})$:
    -   Outcome-specific marginal standard deviations for space-time interaction effects.
-   `L_corr` $\sim \text{LKJCholesky}(N_{\text{obs}}, 1.0, :L)$:
    -   Cholesky factor of the correlation matrix for the latent fields across outcomes. This prior on the Cholesky factor ensures a valid correlation matrix. The parameter 1.0 indicates a uniform prior on the correlation matrix.
-   `r_nb` $\sim (M.model_{\text{family}} == \text{"negbin"}) ? \text{filldist}(\text{Exponential}(1.0), N_{\text{obs}}) : \text{filldist}(\text{Dirac}(1.0), N_{\text{obs}})$:
    -   Outcome-specific Negative Binomial dispersion parameter.
-   `phi_zi` $\sim M.use_{\text{zi}} ? \text{Beta}(1, 1) : \text{Dirac}(0.0)$:
    -   Zero-inflation probability (shared across outcomes if `use_zi` is true).

**2. Stochastic Volatility Manifold (Optional):**

-   `y_sigma_const` $\sim (M.model_{\text{family}} \in [\text{"gaussian"}, \text{"lognormal"}]) ? \text{Exponential}(1.0) : \text{Dirac}(1.0)$:
    -   Constant observation-level standard deviation (if not using stochastic volatility).
-   If `M.use_sv` is true:
    -   `sigma_log_var` $\sim \text{Exponential}(1.0)$:
        -   Variance for the log of the standard deviation in the stochastic volatility model.
    -   `beta_vol` $\sim \text{filldist}(\text{Normal}(0, \text{sigma_log_var}), M.M_{\text{rff_sigma}})$:
        -   Coefficients for the RFF-based stochastic volatility (shared across outcomes).
    -   `y_sigma` is derived as a spatiotemporally varying standard deviation for each observation.
-   Else: `y_sigma = fill(y_sigma_const, M.N_obs)`: Each outcome uses `y_sigma_const`.

**3. Spatial Manifolds (`s_eta`):**

-   `s_rho` $\sim (M.model_{\text{space}} \in [\text{"bym2"}, \text{"leroux"}, \text{"sar"}]) ? \text{filldist}(\text{Beta}(1, 1), N_{\text{obs}}) : \text{filldist}(\text{Dirac}(1.0), N_{\text{obs}})$:
    -   Outcome-specific spatial correlation/mixing parameters.
-   `gcn_weight` $\sim (M.model_{\text{space}} == \text{"bgcn"}) ? \text{filldist}(\text{Beta}(1, 1), N_{\text{obs}}) : \text{filldist}(\text{Dirac}(0.0), N_{\text{obs}})$:
    -   Outcome-specific weights for GCN models.
-   `s_eta_scaled` (matrix `N_areas` x `N_obs`):
    -   For each outcome $k$, a latent ICAR field `s_icar` $\sim \text{MvNormalCanon}(\mathbf{0}, \text{Symmetric}(M.s_Q + M.noise \cdot I))$ is sampled.
    -   A soft sum-to-zero constraint is applied: `s_total_k = sum(s_icar)` $\sim \text{Normal}(0, 0.001 \cdot N_{\text{areas}})$.
    -   Depending on `M.model_space` (e.g., `besag`, `bym2`, `leroux`, `sar`, `iid`), `s_icar` (and potentially `u_iid_k` for BYM2) is transformed to `s_eta_scaled[:, k]`.
-   The final spatial effect is obtained by applying the LKJ correlation and scaling:
    -   `s_eta = (s_eta_scaled * L_corr) .* s_sigma'`
    -   This allows for correlation between the spatial effects of different outcomes.

**4. Temporal Manifolds (`t_eta`):**

-   `t_eta` (matrix `t_N` x `N_obs`):
    -   For each outcome $k$, outcome-specific temporal hyperparameters are drawn (`t_rho_k`, `t_ls_k`, `t_α_k`, `t_β_k`).
    -   A temporal latent field (`t_raw_k`, `t_gp_k`, `t_iid_k`, or direct harmonic calculation) is sampled based on `M.model_time` (e.g., `ar1`, `rw2`, `gp`, `harmonic`, `iid`).
    -   `t_eta[:, k]` is constructed by scaling this latent field with `t_sigma[k]`.
    -   `t_Q[k]` stores the outcome-specific temporal precision matrix, which is used for space-time interactions.

**5. Season Manifolds (`u_eta`):**

-   `u_sigma` $\sim M.model_{\text{season}} \neq \text{"none"} ? \text{filldist}(\text{Exponential}(0.5), N_{\text{obs}}) : \text{filldist}(\text{Dirac}(0.0), N_{\text{obs}})$:
    -   Outcome-specific scale for seasonal effects.
-   `u_eta` (matrix `u_N` x `N_obs`):
    -   Similar to `t_eta`, for each outcome $k$, outcome-specific seasonal hyperparameters are drawn and a seasonal latent field is constructed based on `M.model_season` (`ar1`, `rw2`, `gp`, `harmonic`, `iid`), scaled by `u_sigma[k]`.

**6. Interactions (`st_eta` - Knorr-Held Types):**

-   `st_eta` (array of `N_obs` matrices, each `N_areas` x `t_N`):
    -   For each outcome $k$, `st_raw_k` $\sim \text{MvNormal}(\mathbf{0}, I)$ (for Type I) or $\sim \text{MvNormalCanon}(\mathbf{0}, st_Q_{type_k})$ (for Types II-IV) is sampled.
    -   `st_Q_{type_k}` is constructed using `M.s_Q` and `t_Q[k]`, depending on `M.model_st`.
    -   `st_eta[k]` is the reshaped and scaled interaction field for outcome $k$, scaled by `st_sigma[k]`.

**7. Covariate Smoothing (`c_eta`):**

-   `c_sigma` $\sim M.N_{\text{cov}} > 0 ? \text{filldist}(\text{Exponential}(1.0), N_{\text{obs}}) : \text{filldist}(\text{Dirac}(0.0), N_{\text{obs}})$:
    -   Outcome-specific scale for covariate effects.
-   `c_eta` (matrix `M.N_cat` x `N_obs`):
    -   For each outcome $k$, outcome-specific covariate hyperparameters are drawn and a covariate latent field is constructed based on `M.model_cov` (`ar1`, `rw2`, `gp`, `harmonic`, `rff`, `iid`), scaled by `c_sigma[k]`.

**8. Fixed Effects:**

-   `svc_sigma` $\sim (M.model_{\text{space}} == \text{"svc"}) ? \text{filldist}(\text{Exponential}(0.5), N_{\text{obs}}) : \text{filldist}(\text{Dirac}(0.0), N_{\text{obs}})$:
    -   Outcome-specific scale for spatially varying coefficients.
-   If `M.model_space == "svc" && M.N_fixed > 0`:
    -   `eta_k .+= (s_eta_scaled[M.s_idx, k] .* svc_sigma[k]) .* M.fixed[:, d]` for each fixed effect `d`.
    -   Here, `s_eta_scaled[:, k]` serves as the spatial effect for the coefficients.
-   Else if `M.N_fixed > 0` (non-SVC fixed effects):
    -   `d_beta_k` $\sim \text{filldist}(\text{Normal}(0, 5), M.N_{\text{fixed}})$:
        -   Outcome-specific coefficients for fixed effects.
    -   `eta_k .+= M.fixed * d_beta_k`.

**9. Likelihood Assembly:**

-   For each outcome $k \in \{1, \ldots, N_{\text{obs}}\}$:
    -   An outcome-specific linear predictor `eta_k` is assembled by summing all relevant components (offset, spatial, temporal, seasonal, interaction, covariates, fixed effects).
    -   `Turing.@addlogprob! logpdf(bstm_Likelihood(M.model_family, M.use_zi, M.weights, phi_zi, r_nb[k], y_sigma[k], M.trials, M.y_obs[:, k]), eta_k)`:
        -   The `bstm_Likelihood` function computes the log-likelihood for the $k$-th outcome, using its specific parameters (`r_nb[k]`, `y_sigma[k]`, `M.y_obs[:, k]`), and shared parameters (`M.model_family`, `M.use_zi`, `M.weights`, `phi_zi`, `M.trials`).



### References:

*   Besag, J., York, J., & Mollié, A. (1991). Bayesian image restoration, with applications in spatial statistics. *Annals of the Institute of Statistical Mathematics*, 43(1), 1-20.
*   Riebler, A., Sørbye, S. H., & Rue, H. (2016). An intuitive Bayesian spatial model with two hyperparameters. *Statistical Methods in Medical Research*, 25(2), 1145-1160.
*   Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time variation in disease risk. *Statistical Methods in Medical Research*, 9(3), 205-220.
*   Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
*   Rahimi, A., & Recht, B. (2008). Random features for large-scale kernel machines. *Advances in Neural Information Processing Systems*, 20.
 



## `bstm_multifidelity` Model Documentation

### Fidelity Definitions

| Attribute    | Low-Fidelity (Fidelity 1) | High-Fidelity (Fidelity 2) |
| :-------------| :--------------------------| :---------------------------|
| **Volume**   | High / Dense              | Low / Sparse               |
| **Cost**     | Cheap (Proxy data)        | Expensive (Validated data) |
| **Bias**     | Potentially High          | Negligible                 |
| **Variance** | High Noise                | Low / Instrument Precision |
| **Role**     | Source (Structural Prior) | Target (Objective Truth)   |

The following code demonstrates the simulation of these two distinct streams.

### Conceptual Overview

The `bstm_multifidelity` model is a Bayesian spatio-temporal framework designed to integrate data from multiple fidelity levels (e.g., high-resolution observations and lower-resolution simulations or auxiliary data). It aims to leverage the strengths of each data source to produce a more robust and accurate inference of the underlying latent processes. The model achieves this by: 

1.  **Global Hyperpriors:** Setting up overarching priors for key model parameters like observation noise, dispersion, and zero-inflation.
2.  **Nested Latent Covariate Structure (RFF-based Multi-Fidelity):** This is the core multi-fidelity component. It models latent spatial (`z_latent`) and spatiotemporal (`w_latent`) processes using Random Fourier Features (RFF). Crucially, the medium-fidelity `w_latent` process *depends* on the high-fidelity `z_latent` process, creating a nested structure that allows information flow between fidelity levels. This captures complex, non-linear relationships between covariates and observations.
3.  **Spatio-Temporal Manifolds:** Incorporating traditional Bayesian spatio-temporal components to capture residual variation:
    *   **Spatial (`s_eta`):** Models spatial autocorrelation using a combination of ICAR (Intrinsic Conditional Autoregressive) and IID (Independent and Identically Distributed) effects, allowing for flexible spatial smoothing.
    *   **Temporal (`t_eta_full`):** Models temporal dependence using various structures like AR(1), RW2 (Random Walk of order 2), Gaussian Processes (GP), or IID effects.
    *   **Seasonality (`u_eta_full`):** Captures periodic patterns in data, also offering AR(1), RW2, GP, or IID structures.
4.  **Space-Time Interaction (`st_eta`):** Allows for different types of interactions between spatial and temporal effects, ranging from IID to fully inseparable structures (Knorr-Held Types 0-4), enabling the model to capture non-additive spatio-temporal dynamics.
5.  **Linear Predictor Construction:** Combines all latent components (spatial, temporal, seasonality, multi-fidelity RFFs, and interaction terms) into a single linear predictor (`eta`), which represents the expected value of the response.
6.  **Joint Multi-fidelity Likelihood:** The model specifies likelihoods for both the multi-fidelity latent observations (`z_obs`, `w_obs`) and the primary observations (`y_obs`), linking the latent processes to the observed data. This allows the model to learn from all available data simultaneously.

In essence, `bstm_multifidelity` is designed to be highly flexible, combining established spatio-temporal modeling techniques with modern multi-fidelity approaches to handle complex data structures and improve predictive performance by integrating information from diverse sources.

### Mathematical Logic of Cross-Fidelity Coupling

The `bstm_multifidelity` model fuses observations from two different regimes (High and Low fidelity) by anchoring them to a common unobserved state. 

#### 1. The Shared Latent Process
Both fidelities share a underlying spatiotemporal signal $\eta_{shared}(s, t)$, which represents the 'true' process we wish to estimate:
$$\eta_{shared} = \text{Spatial}_{latent} + \text{Temporal}_{latent} + \text{Seasonal}_{latent}$$

#### 2. High-Fidelity Path (Gold Standard)
The predictor for high-fidelity observations is a direct sum of fixed effects, random covariate effects, and the shared signal:
$$\eta_{High} = \beta X + \sum \text{Covariate}_{latent} + \eta_{shared}$$

#### 3. Low-Fidelity Path (Calibration)
The low-fidelity (proxy) observations are modeled as a linear transformation of the shared signal to account for systematic sensor bias and sensitivity differences:
$$\eta_{Low} = \beta_{fidelity\_bias} + (\rho_{fidelity\_rho} \cdot \eta_{shared}) + \beta X + \sum \text{Covariate}_{latent}$$

*   **$\beta_{fidelity\_bias}$**: Captures additive bias (constant over/under-estimation).
*   **$\rho_{fidelity\_rho}$**: Captures multiplicative scaling (signal compression or expansion in the proxy).

By sharing the $\eta_{shared}$ terms, the model 'borrows strength' from the typically high-volume low-fidelity data to refine the latent field in regions where high-fidelity data is sparse.


The `bstm_multifidelity` model is defined by a series of hierarchical priors and likelihoods, detailed below:

#### 1. Global Hyperpriors

These priors govern the fundamental properties of the observation noise and data distribution:

*   **`y_sigma` (Observation Noise Scale):**
    *   If `M.model_family` is `"gaussian"` or `"lognormal"`, `y_sigma ~ Exponential(1.0)`. This assigns a prior favoring smaller observation noise, with a mean of 1.0.
    *   Otherwise, `y_sigma ~ Dirac(1.0)`, fixing the scale to 1.0 (e.g., for Poisson or Negative Binomial where dispersion is handled differently or not applicable).

*   **`r_nb` (Negative Binomial Dispersion):**
    *   If `M.model_family == "negbin"`, `r_nb ~ Exponential(1.0)`. This provides a prior for the dispersion parameter of the Negative Binomial distribution, influencing its overdispersion.
    *   Otherwise, `r_nb ~ Dirac(1.0)`, making it inactive for other likelihood families.

*   **`phi_zi` (Zero-Inflation Probability):**
    *   If `M.use_zi` is true, `phi_zi ~ Beta(1, 1)`. A uniform prior between 0 and 1 for the probability of excess zeros.
    *   Otherwise, `phi_zi ~ Dirac(0.0)`, setting the zero-inflation probability to zero.

*   **`z_sigma` (High-Fidelity Noise Scale):** `z_sigma ~ Exponential(0.5)`. Prior for the observation noise of the high-fidelity latent process `z_latent`.

*   **`w_sigma` (Medium-Fidelity Noise Scales):** `w_sigma ~ filldist(Exponential(0.5), 3)`. Independent priors for the observation noise of the three medium-fidelity latent processes `w_latent[:, k]`.

#### 2. Nested Latent Covariate Structure (RFF based)

This section models multi-fidelity latent processes using Random Fourier Features (RFF) for flexible, non-linear function approximation. RFFs transform input coordinates into a high-dimensional space where linear models can approximate functions that are non-linear in the original space.

*   **High-Fidelity `Z` (Spatial):**
    *   **`z_ls` (Length Scale):** `z_ls ~ Gamma(2, 2)`. Prior for the length scale governing the smoothness of the `z_latent` spatial process. `Gamma(2,2)` is a common weakly informative prior for scales.
    *   **`z_beta` (RFF Weights):** `z_beta ~ filldist(Normal(0, 1), M.M_rff)`. Weights for the RFF expansion, drawn from a standard normal distribution.
    *   **`z_proj` (Projected Coordinates):** `z_proj = (M.z_coords_s * (M.W_fixed[1:size(M.z_coords_s, 2), :] ./ z_ls)) .+ M.b_fixed'`. This projects the spatial coordinates `M.z_coords_s` into the RFF space. `M.W_fixed` and `M.b_fixed` are precomputed RFF projection matrices and biases.
    *   **`z_latent` (High-Fidelity Latent Process):** `z_latent = M.rff_scale .* (cos.(z_proj) * z_beta)`. The final high-fidelity latent process, obtained by applying cosine activation to the projected coordinates and scaling by `M.rff_scale` and the RFF weights `z_beta`.

*   **Medium-Fidelity `W` (Spatiotemporal, depends on `Z`):**
    *   **`w_ls` (Length Scale):** `w_ls ~ Gamma(2, 2)`. Prior for the length scale of the `w_latent` spatiotemporal process.
    *   **`w_beta` (RFF Weights):** `w_beta ~ filldist(Normal(0, 1), M.M_rff, 3)`. Weights for the RFF expansion of the `w_latent` processes (three of them, hence `3`).
    *   **`w_coords_augmented` (Augmented Coordinates):** `w_coords_augmented = hcat(M.w_coords_st, z_latent[1:size(M.w_coords_st, 1)])`. This is the crucial step for multi-fidelity nesting: the medium-fidelity spatiotemporal coordinates `M.w_coords_st` are augmented with the *high-fidelity latent process* `z_latent`. This allows the `w_latent` to learn non-linear relationships with `z_latent`.
    *   **`w_proj` (Projected Coordinates):** `w_proj = (w_coords_augmented * (M.W_fixed[1:size(w_coords_augmented, 2), :] ./ w_ls)) .+ M.b_fixed'`. Projects the augmented coordinates into the RFF space.
    *   **`w_latent` (Medium-Fidelity Latent Process):** `w_latent = M.rff_scale .* (cos.(w_proj) * beta_w)`. The medium-fidelity latent process, derived from its RFF expansion and dependence on `z_latent`.

#### 3. Spatial, Temporal, and Seasonality Manifolds

These components capture standard spatio-temporal variation independent of the multi-fidelity RFFs.

*   **Spatial Manifold (`s_eta`):**
    *   **`s_sigma` (Spatial Scale):** `s_sigma ~ Exponential(1.0)`. Overall scale for the spatial effect.
    *   **`s_rho` (Spatial Mixing Parameter):** `s_rho ~ Beta(1, 1)`. A parameter for mixing between structured ICAR and unstructured IID spatial effects. A `Beta(1,1)` prior is uniform over `(0,1)`.
    *   **`s_icar` (ICAR Component):** `s_icar ~ MvNormalCanon(zeros(M.N_areas), M.s_Q + M.noise*I)`. This represents the Intrinsic Conditional Autoregressive (ICAR) component. `MvNormalCanon` parameterizes by mean and precision matrix. `M.s_Q` is a precomputed structural precision matrix, and `M.noise*I` adds a small amount of jitter for numerical stability.
    *   **`s_iid` (IID Component):** `s_iid ~ MvNormal(zeros(M.N_areas), I)`. An Independent and Identically Distributed spatial component.
    *   **Sum-to-zero constraint:** `sum_icar = sum(s_icar); sum_icar ~ Normal(0, 0.001 * M.N_areas)`. This soft constraint helps ensure identifiability by preventing the spatial field from absorbing the global intercept. It gently pulls the sum of the ICAR component towards zero.
    *   **`s_eta` (Mapped Spatial Effect):** `s_eta = (s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid))[M.s_idx]`. The final spatial effect, constructed as a weighted average of the `s_icar` and `s_iid` components, scaled by `s_sigma`, and then mapped to the observations via `M.s_idx`.

*   **Temporal Manifold (`t_eta_full`):**
    *   **`t_sigma` (Temporal Scale):** `t_sigma ~ Exponential(1.0)`. Overall scale for the temporal effect.
    *   **`t_rho` (AR(1) Parameter):** If `M.model_time == "ar1"`, `t_rho ~ Beta(2, 2)`. Prior for the autocorrelation parameter in an AR(1) process, favoring values near 0.5. Otherwise, `t_rho ~ Dirac(0.0)`.
    *   **`t_ls` (GP Length Scale):** If `M.model_time == "gp"`, `t_ls ~ InverseGamma(3, 3)`. Prior for the length scale of a Gaussian Process. Otherwise, `t_ls ~ Dirac(1.0)`.
    *   **Conditional Logic for `t_eta_full` and `t_Q`:**
        *   **`"ar1"` (Autoregressive of order 1):**
            *   `t_Q_base = Symmetric((1.0 + t_rho^2) .* I(M.t_N) .+ (t_rho) .* M.t_Q)`. The base precision matrix for the AR(1) process, constructed from `t_rho` and a structural template `M.t_Q`.
            *   `t_Q = Symmetric((1.0 / (1.0 - t_rho^2 + M.noise)) .* t_Q_base )`. The final precision matrix, scaled by the AR(1) variance `(1 - t_rho^2)`.
            *   `t_raw ~ MvNormalCanon(zeros(M.t_N), t_Q)`. Samples the raw temporal effect from a canonical multivariate normal.
            *   `t_eta_full = (t_raw .* t_sigma)[M.t_idx]`. Scales the raw effect and maps it to observations.
        *   **`"rw2"` (Random Walk of order 2):**
            *   `t_Q = Symmetric((1.0 / (t_sigma^2 + M.noise)) .* M.t_Q )`. The precision matrix, scaled by `t_sigma^2` and the precomputed RW2 structural template `M.t_Q`.
            *   `t_raw ~ MvNormalCanon(zeros(M.t_N), t_Q)`. Samples the raw temporal effect.
            *   `t_eta_full = t_raw[M.t_idx]`. Maps the raw effect. Note `t_sigma` is already in `t_Q`.
        *   **`"gp"` (Gaussian Process):**
            *   `K_t = (t_sigma^2) .* kernelmatrix(SqExponentialKernel() ∘ ScaleTransform(inv(t_ls)), 1.0:Float64(M.t_N)) + M.noise * I`. Constructs the covariance matrix `K_t` using a Squared Exponential kernel with length scale `t_ls` and variance `t_sigma^2`.
            *   `t_gp ~ MvNormal(zeros(M.t_N), Symmetric(K_t))`. Samples the temporal effect directly from a multivariate normal with `K_t` as covariance.
            *   `t_eta_full = t_gp[M.t_idx]`. Maps the GP samples.
            *   `t_Q = inv(Symmetric(K_t))`. Derives the precision matrix for interaction logic.
        *   **`"iid"` (Independent and Identically Distributed):**
            *   `t_Q = Symmetric((1.0 / (t_sigma^2 + M.noise)) .* I(M.t_N))`. Precision matrix for IID effects.
            *   `t_iid ~ MvNormal(zeros(M.t_N), I)`. Samples IID effects.
            *   `t_eta_full = (t_iid .* t_sigma)[M.t_idx]`. Scales and maps IID effects.
        *   **Else:** `t_eta_full = zeros(T, M.N_obs)`.

*   **Seasonality Manifold (`u_eta_full`):**
    *   **`u_sigma` (Seasonality Scale):** If `M.model_season != "none"`, `u_sigma ~ Exponential(0.5)`. Otherwise, `u_sigma ~ Dirac(0.0)`.
    *   **`u_rho` (AR(1) Parameter):** If `M.model_season == "ar1"`, `u_rho ~ Beta(2, 2)`. Otherwise, `u_rho ~ Dirac(0.0)`.
    *   **`u_ls` (GP Length Scale):** If `M.model_season == "gp"`, `u_ls ~ InverseGamma(3, 3)`. Otherwise, `u_ls ~ Dirac(1.0)`.
    *   **Conditional Logic for `u_eta_full`:** Similar logic to the temporal manifold, but applied to seasonal cycles using `M.u_N`, `M.u_Q`, and `M.u_idx`.

#### 4. Space-Time Interaction (`st_eta`)

This component captures non-additive interactions between space and time, following Knorr-Held types.

*   **`st_sigma` (Interaction Scale):** If `M.model_st != "none"`, `st_sigma ~ Exponential(0.5)`. Otherwise, `st_sigma ~ Dirac(0.0)`.
*   **Conditional Logic for `st_eta`:**
    *   **`M.model_st == "none"` (No Interaction):** `st_eta = zeros(T, M.N_areas, M.t_N)`.
    *   **`M.model_st == "I"` (Type I: IID Interaction):** `st_raw ~ MvNormal(zeros(M.N_areas * M.t_N), I)`. Samples IID interaction effects, then reshapes and scales them.
    *   **`M.model_st == "II"` (Type II: Temporal Structure):** `st_Q2 = Symmetric(kron(I(M.N_areas), t_Q) )`. Precision for each area having an independent structured temporal trend, using the previously derived `t_Q`.
    *   **`M.model_st == "III"` (Type III: Spatial Structure):** `st_Q3 = Symmetric(kron(M.s_Q, I(M.t_N)) )`. Precision for each time point having an independent structured spatial field, using `M.s_Q`.
    *   **`M.model_st == "IV"` (Type IV: Inseparable):** `st_Q4 = Symmetric(kron(M.s_Q, t_Q) )`. Fully inseparable spatiotemporal structure, using the Kronecker product of `M.s_Q` and `t_Q`.
    *   In types 1-4, `st_raw` is sampled from a canonical multivariate normal with the derived precision, then `st_eta` is reshaped and scaled by `st_sigma`.

#### 5. Linear Predictor Construction

This combines all latent effects into the final linear predictor `eta`.

*   **`z_beta_eta` (High-Fidelity Linkage):** `z_beta_eta ~ Normal(0, 1)`. A coefficient linking the high-fidelity latent process `z_latent` to the overall linear predictor.
*   **`w_beta_eta` (Medium-Fidelity Linkage):** `w_beta_eta ~ MvNormal(zeros(3), I)`. Coefficients linking the three medium-fidelity latent processes `w_latent` to the overall linear predictor.
*   **`eta` (Linear Predictor):** `eta = M.log_offset .+ s_eta .+ t_eta_full .+ u_eta_full .+ (z_latent .* z_beta_eta) .+ (w_latent * w_beta_eta)`. This sums the log offset, spatial, temporal, seasonal, and multi-fidelity RFF effects.
*   **Interaction Term Addition:** `if M.model_st != "none"; for i in 1:M.N_obs; eta[i] += st_eta[M.s_idx[i], M.t_idx[i]]; end; end`. If a space-time interaction is enabled, its components are added to the corresponding observation points in `eta`.

#### 6. Joint Multi-fidelity Likelihood

This defines how the observed data are generated from the latent processes.

*   **High-Fidelity Latent Likelihood:** `Turing.@addlogprob! logpdf(MvNormal(z_latent, z_sigma^2 * I), M.z_obs)`. The high-fidelity observations `M.z_obs` are assumed to be normally distributed around the `z_latent` process with variance `z_sigma^2`.

*   **Medium-Fidelity Latent Likelihoods:** `for k in 1:3; Turing.@addlogprob! logpdf(MvNormal(w_latent[:, k], w_sigma[k]^2 * I), M.w_obs[:, k]); end`. Each of the three medium-fidelity observations `M.w_obs[:, k]` is normally distributed around its respective `w_latent[:, k]` process with variance `w_sigma[k]^2`.

*   **Primary Observation Likelihood:** `Turing.@addlogprob! logpdf(bstm_Likelihood(M.model_family, M.use_zi, M.weights, phi_zi, r_nb, y_sigma, M.trials, M.y_obs), eta)`. The primary observations `M.y_obs` are modeled according to `M.model_family` (e.g., Poisson, Negative Binomial, Gaussian) with potential zero-inflation (`phi_zi`), negative binomial dispersion (`r_nb`), and observation scale (`y_sigma`), all linked via the linear predictor `eta` and observation-specific weights (`M.weights`) and trials (`M.trials`) for count models.


#### References for Multi-Fidelity Models

*   **General Multi-Fidelity Modeling:**
    *   Forrester, A. I. J., Sóbester, A., & Keane, A. J. (2008). *Engineering Design via Surrogate Modelling: A Practical Guide* (pp. 129-152). Chichester: John Wiley & Sons. (General overview of multi-fidelity optimization and modeling)
    *   Kennedy, M. C., & O'Hagan, A. (2001). Bayesian calibration of computer models. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 63(3), 425-450. (Foundational work on calibrating computer models, often related to multi-fidelity)

*   **Random Fourier Features (RFF) and Kernel Methods:**
    *   Rahimi, A., & Recht, B. (2008). Random features for large-scale kernel machines. *Advances in Neural Information Processing Systems*, 20. (Key paper introducing RFFs)
    *   Sutherland, D. J., & Schneider, J. (2015). On the error of random Fourier features. *Proceedings of the Eighteenth International Conference on Artificial Intelligence and Statistics*. (Analysis of RFF performance)

*   **Bayesian Approaches in Multi-Fidelity:**
    *   P. Perdikaris and K. Willcox. (2017). Learning solutions of parametrized partial differential equations from small datasets. *SIAM Journal on Scientific Computing*, 39(4), A2054–A2079. (Bayesian multi-fidelity methods for PDEs)
    *   Cutler, J., & Wild, S. M. (2020). Multi-fidelity Bayesian Optimization with Deep Neural Networks. *Advances in Neural Information Processing Systems*, 33.



## Reaction-Advection-Diffusion  
 

The `bstm` model extends standard spatiotemporal Bayesian models by incorporating a **mechanistic transport layer**. Instead of assuming a purely statistical interaction (like a Knorr-Held Type IV), it can simulate the evolution of a latent field through a discretized **Diffusion-Advection-Reaction PDE**.

#### The Latent State Dynamics 

Let $\eta_{s,t}$ represent the latent spatiotemporal field at location $s$ and time $t$. The model evolves as:

$$\eta_t = \text{logistic}(\rho_s) \odot \mu_{phys} + \epsilon_{innov}$$

Where $\mu_{phys}$ is the physical prediction derived from the previous state:

$$\mu_{phys} = \eta_{t-1} - \underbrace{\delta (L \eta_{t-1})}_{\text{Diffusion}} - \underbrace{\alpha (L \rho_s)}_{\text{Advection}}$$

 
*   **Diffusion ($\delta L \eta_{t-1}$):** 
    The Graph Laplacian $L = D - W$ acts as a discrete approximation of the negative Laplace-Beltrami operator ($-\nabla^2$). Multiplying by the diffusion coefficient $\delta$ (code: `st_diffusion`) smooths the field, causing high-intensity regions to dissipate into neighboring units.
*   **Advection ($\alpha L \rho_s$):** 
    This represents transport driven by a potential gradient. Here, the static spatial field $\rho_s$ (code: `s_rho`) acts as a 'source' or 'pressure' map. The term $\alpha L \rho_s$ simulates the flow of the latent field toward regions of lower spatial risk/intensity.
*   **Spatially-Varying Persistence ($\text{logistic}(\rho_s)$):**
    The model allows the "memory" of the physical process to vary by location. If $\rho_s$ is high, the location retains more of its physical momentum; if low, it is more heavily dominated by new stochastic innovations ($\epsilon_{innov}$).

#### Implementation Details

*   **Precision Handling:** The model uses `MvNormalCanon` for the static spatial field $\rho_s$ to leverage the sparsity of the Graph Laplacian $L$, ensuring computational efficiency even as spatial resolution increases.
*   **Innovation:** `st_eta_z` represents independent white noise innovations that are reshaped and scaled by `st_sigma` to provide the stochastic driving force at each time step.
*   **Clamping & Stability:** In the linear predictor assembly, we use `clamp` and `Int` casting to ensure indices mapping to the areal units remain within the bounds of the precomputed $W$ and $L$ matrices.



## Example Recipes

### 1. Standard Poisson CARSTM

A common model for disease mapping or species counts.

```julia
m = bstm(
    "counts ~ 1 + spatial(area_id, model=bym2) + temporal(year_id, model=ar1)",
    my_data,
    model_family="poisson",
    W=my_adjacency_matrix
)
```

### 2. Gaussian Model with Non-linear Smooth and SVC

Models a continuous outcome with a non-linear effect of `temperature` and allows the effect of `rainfall` to vary over space.

```julia
m = bstm(
    "yield ~ 1 + smooth(temperature, nbins=20) + spatial(rainfall, s_idx, model=iid)",
    my_data,
    model_family="gaussian"
)
```

### 3. Binomial Model with Spacetime Interaction

Models prevalence (0 or 1) with a fully structured spatiotemporal interaction term.

```julia
m = bstm(
    "prevalence ~ 1 + spatial(s_idx, model=bym2) + temporal(t_idx, model=ar1) + (spatial(s_idx) ⊗ temporal(t_idx))",
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
m = bstm("y ~ 1 + x + spatial(s_idx, model=bym2)", my_data, model_family="poisson")
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
 
### Manifold Options

The `model` argument within modules like `spatial`, `temporal`, and `smooth` specifies the underlying mathematical structure.

| Manifold         | `model=...'`                   | Description                                                                                 |
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
    *   Example: `spacetime(model=spatial(s_idx) ⊗ temporal(t_idx))`
    *   Example: `spacetime(model=spatial(s_idx) ⊗ temporal(t_idx))`
*   `⊕` (`\oplus`): Direct sum. (Future support for block-diagonal structures).

## Configuration (Advanced)



## The Mixed-Sampler Strategy: How the Model Learns

Inference in **bstm** requires a Mixed-Sampler Gibbs approach, as no single algorithm is optimal for the entire parameter space. The `get_optimal_sampler` function provides both an informed automatic choice and manual overrides. The following table summarizes the available samplers and their primary use cases.

| Sampler   | `sampler_choice`                | Type           | Key Characteristic                                     | Best Use Case                                                                                                                                                                                                          |
| :----------| :--------------------------------| :---------------| :-------------------------------------------------------| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **NUTS**  | `:nuts`, `:auto` (default)      | Gradient-Based | Adaptively tunes step size and number of steps.        | The state-of-the-art, general-purpose sampler for models with continuous, differentiable parameters. It is robust and requires minimal tuning.                                                                         |
| **HMC**   | `:hmc`                          | Gradient-Based | Requires manual tuning of leapfrog steps.              | A powerful alternative to `NUTS`. It can be very efficient but may require expert tuning to find optimal parameters for a given model.                                                                                 |
| **ESS**   | `:ess`, `:auto` (if applicable) | Gradient-Free  | Designed specifically for models with Gaussian priors. | Highly efficient for latent Gaussian models (e.g., CAR, GP models). It makes large, coherent moves through the posterior without requiring gradient information.                                                       |
| **Slice** | `:slice`                        | Gradient-Free  | Adapts its step size to explore the posterior slice.   | A robust, general-purpose gradient-free sampler. It can be more efficient than `MH` for some problems, especially those with complex unimodal posteriors, but is generally less efficient than gradient-based methods. |
| **MH**    | `:mh`                           | Gradient-Free  | Proposes moves from a simple proposal distribution.    | A universal sampler that can work for non-differentiable models. It is often inefficient in high-dimensional, correlated parameter spaces and should be used when gradient-based methods fail.                         |
| **PG**    | (Internal)                      | Particle-Based | Used for discrete parameters within a `Gibbs` sampler. | Automatically employed by `get_optimal_sampler` to handle any discrete random variables in the model, such as those from a `Categorical` or `Poisson` prior.                                                           |

A note on `SGLD` (Stochastic Gradient Langevin Dynamics): This sampler is designed for "big data" scenarios where the likelihood cannot be evaluated on the full dataset at each step. It uses mini-batches to approximate the gradient. The `bstm` framework is not currently structured for this type of data subsampling, so `SGLD` is not included as a general-purpose option.

For production-grade point estimates, we may utilize ADVI (Automatic Differentiation Variational Inference). In these cases, increasing the n_samples for the ELBO gradient estimation is critical to stabilize convergence against the noise of complex spatial interactions.


## advanced:

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

model_results_plots(res)  # plots are already created (above), this displays them

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

This is still not possible at present, as Turing is not fully modular. All sampling statements must be made in the same macro. One potential solution is to use string manipulation to inject the sampling statements into the model. At present, the following can, after development, be directly copied into a copy of bstm_univariate() and run. This is therefore, a placemarker and a template for how to get it done now.


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
    "y ~ 1 + dynamics(t_idx, model=custom, func=:custom_schaefer_model, K_prior=Normal(200,50))",
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
formula = "y1 + y2 + y3 ~ 1 + dynamics(t_idx, model=n-species_lotka_volterra)"

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
formula = "y1 + y2 + y3 ~ 1 + spatial(s_idx) + dynamics(t_idx, model=n-species_lotka_volterra_spatial_K)"

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
formula = "y ~ 1 + dynamics(t_idx, model=logistic_f)"

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

formula = "y ~ 1 + dynamics(t_idx, model=logistic_f, r_covariate=:temperature)"

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
formula = "y ~ 1 + dynamics(t_idx, model=logistic_f, r_covariate=:temperature)"

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

formula = "y ~ 1 + dynamics(t_idx, model=logistic_f, r_covariate=:temperature, K_covariate=:habitat_quality)"

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
formula = "y ~ 1 + dynamics(t_idx, model=logistic_f, K_covariate=:habitat_quality)"

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

formula = "y ~ 1 + dynamics(t_idx, model=logistic_f, r_covariate=:temperature, K_covariate=:habitat_quality)"

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
formula = "y ~ 1 + dynamics(t_idx, model=logistic_f, r_covariate=:temperature, K_covariate=:habitat_quality)"

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

You'll notice that the state-transition equation for the log-population is identical for both the discrete Ricker model and the Euler-discretized continuous logistic model. The existing dynamics(..., model=basic_logistic) in bstm already implements this exact structure.

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
formula = "counts ~ 1 + dynamics(time, model=ricker, r_prior=LogNormal(0,1), K_prior=Normal(700, 200))"

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
formula = "y ~ 1 + dynamics(time_idx, model=gompertz)"

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
