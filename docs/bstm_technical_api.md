---
title: "bstm Technical API Reference"
format: html
---

# `bstm` Technical API Reference

## 1. Introduction

This document provides a detailed technical reference for the internal components of the `bstm` framework. It is intended for developers and advanced users who wish to understand, extend, or debug the framework's core machinery. It covers the manifold system, formula parsing engine, model configuration pipeline, Turing model definitions, and the posterior reconstruction engine.

## 2. Core Data Structures & Manifold System

The `bstm` framework is built upon a system of `Manifold` types that represent different structural assumptions about latent fields.

### 2.1. Abstract Manifold Types

The manifold system is organized under a hierarchy of abstract types that define the role of each component.

*   **`abstract type Manifold end`**: The root type for all structural components.
*   **`abstract type ManifoldModel <: Manifold end`**: Represents a base-level statistical model for a latent field (e.g., `ICAR`, `AR1`, `GP`). These are the fundamental building blocks.
*   **`abstract type ManifoldOperator <: Manifold end`**: Represents an operation that combines or transforms one or more `Manifold` objects (e.g., `ComposedManifold` for `⊗`, `SVCManifold` for `|>`).
*   **`abstract type ManifoldSupervisor <: Manifold end`**: A special type for modules that manage entire sub-models, such as the `nested()` module.

### 2.2. `ManifoldModel` Structs

These structs define the specific statistical models for latent fields. Their fields primarily store prior distributions for the model's hyperparameters.
*   **`sigma_prior`**: A `UnivariateDistribution` that defines the prior for the marginal standard deviation (scale) of the latent field. It controls the overall magnitude of the effect.
*   **`rho_prior`**: A `UnivariateDistribution` for a correlation or mixing parameter, typically on `[0, 1]`. In `BYM2` or `Leroux` models, it controls the proportion of variance attributed to the structured spatial component. In `AR1` models, it represents the temporal autocorrelation.
*   **`lengthscale_prior`**: A `UnivariateDistribution` for the lengthscale parameter in continuous Gaussian Process models (`GP`, `RFF`, etc.). It controls the distance over which points are correlated; a larger lengthscale implies a smoother, more global process.
*   **`kappa_prior`**: A `UnivariateDistribution` for the `kappa` parameter in `SPDE` (Matérn) models, which is inversely related to the lengthscale and controls the smoothness of the field.
*   **`n_features` / `nbins`**: An `Int` specifying the number of basis functions (for `RFF`) or bins (for `PSpline`) used in approximation methods.
*   **`degree` / `diff_order`**: `Int` parameters for spline models, controlling the polynomial degree of the basis functions and the order of the difference penalty for smoothing.
*   **`pca_sd_prior`**: Prior for the standard deviations of the principal components (latent factors) in `Eigen` models.
*   **`pdef_sd_prior`**: Prior for the standard deviation of the residual (uniqueness) noise in `Eigen` models.

```julia
# Discrete & Graph Primitives
struct IID <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct ICAR <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct Besag <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct BYM2 <: ManifoldModel
    rho_prior::UnivariateDistribution       # Prior for the mixing parameter between structured and unstructured effects.
    sigma_prior::UnivariateDistribution     # Prior for the overall spatial standard deviation.
end
struct Leroux <: ManifoldModel; rho_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end
struct SAR <: ManifoldModel; rho_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end
struct RW1 <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct RW2 <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct AR1 <: ManifoldModel; rho_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end

# Continuous, Spectral, and Advanced Manifolds
struct GP <: ManifoldModel
    lengthscale_prior::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}} # Prior for lengthscale(s). Can be a single value for isotropic kernels or a vector for ARD kernels.
    sigma_prior::UnivariateDistribution     # Prior for the GP's signal variance.
    kernel::String                          # Name of the kernel function (e.g., "se", "matern32").
end
struct RFF <: ManifoldModel
    lengthscale_prior::UnivariateDistribution # Prior for the lengthscale of the approximated kernel.
    sigma_prior::UnivariateDistribution     # Prior for the signal variance.
    n_features::Int                         # Number of random features to use in the approximation.
    kernel::String                          # The kernel being approximated.
end
struct SPDE <: ManifoldModel; sigma_prior::UnivariateDistribution; kappa_prior::UnivariateDistribution; end
struct PSpline <: ManifoldModel; nbins::Int; degree::Int; diff_order::Int; sigma_prior::UnivariateDistribution; end
struct Eigen <: ManifoldModel
    n_factors::Int                          # Number of latent factors to extract.
    pca_sd_prior::UnivariateDistribution    # Prior for the standard deviations of the principal components.
    pdef_sd_prior::UnivariateDistribution   # Prior for the standard deviation of the residual (uniqueness) noise.
    ltri_indices::Vector{Int}               # Indices for the lower-triangular part of the Householder reflector matrix.
end
struct DynamicsManifold <: ManifoldModel; model::String; params::Dict{Symbol, Any}; end
```

### 2.3. `ManifoldOperator` Structs

These structs implement the algebraic composition of manifolds.

*   **`struct ComposedManifold <: ManifoldOperator`**: Represents algebraic compositions like `⊗` (Kronecker product) and `⊕` (direct sum). It holds a vector of component `Manifold` objects and an `operator` symbol.
*   **`struct SVCManifold <: ManifoldOperator`**: Represents a Spatially Varying Coefficient model, created by the `|>` operator (e.g., `poverty |> spatial(...)`). It links a covariate to a spatial manifold.
*   **`struct MixedManifold <: ManifoldOperator`**: Represents a random effect (intercept or slope) for a specified grouping variable.

### 2.4. `bstm_Likelihood` Struct

This struct defines the observation model and uses a trait-based system for dispatching to the correct log-pdf kernel.

```julia
struct bstm_Likelihood{F<:AbstractBSTM_Family, Z<:AbstractZIState, C<:AbstractCensoringState, ...} <: ContinuousMultivariateDistribution
    family::F           # e.g., PoissonFamily()
    y_obs::TR           # Observed data
    zi_state::Z         # e.g., ZeroInflated()
    censoring_state::C  # e.g., Uncensored()
    # ... other parameters like weights, phi_zi, r_nb, etc.
end
```

The `logpdf` function for this struct calls `bstm_kernel`, which dispatches on the `family`, `censoring_state`, and `zi_state` types to compute the correct log-probability for each observation.

## 3. Formula Parsing Engine (`formula_parsing.jl`)

The formula parser translates the user-provided formula string into a structured representation that the configuration engine can process.

*   **`decompose_bstm_formula(formula_str)`**: This is the main entry point. It splits the formula into its Left-Hand Side (LHS) and Right-Hand Side (RHS).
    *   **LHS**: Parsed to identify outcome variables and their likelihood specifications (e.g., `likelihood(y, family=poisson)`).
    *   **RHS**: Pre-processed to handle intercept control (`-1`, `0`, `intercept(false)`) and to normalize all bare terms (e.g., `z`) into explicit `fixed(z)` module calls. It is then parsed by `_parse_rhs_expression` into an Abstract Syntax Tree (AST).
    *   **Output**: Returns a `NamedTuple` containing `:outcomes`, `:modules` (a dictionary of all parsed RHS terms), `:has_intercept`, and `:intercept_prior`. The concept of a separate `fixed_effects` list is removed from this stage.

*   **`_parse_rhs_expression(term_str)`**: A recursive descent parser that respects operator precedence to build the AST for the RHS. The precedence is:
    1.  `+` (Addition of independent effects)
    2.  `-` (Subtraction of terms, internally converted to `+ -term`)
    2.  `⊕` (Direct Sum)
    3.  `|>` (Pipe for SVC/state-space)
    4.  `⊗` (Kronecker Product)
    5.  `∘` (Composition)

*   **`_categorize_rhs_nodes!(nodes, modules, fixed_effects)`**: Traverses the generated AST to populate the `modules` dictionary and the `fixed_effects` list. It correctly identifies composed manifolds (e.g., `spatial() ⊗ temporal()`) as a single interaction module.

*   **`_parse_single_manifold_term` & `_parse_arguments_string`**: Helper functions that parse individual module calls (e.g., `spatial(s_idx, model=:bym2)`) into a dictionary of variables and parameters. These functions now correctly handle Julia `Symbol` literals (e.g., `:besag`) passed as arguments.

## 4. Model Configuration Engine (`modelling.jl`)

The `bstm_config` function is the main engine that transforms the parsed formula and data into a complete configuration object (`M`) for the Turing models.

### 4.1. `bstm_config` Workflow

1.  **Initialization**: `_initialize_config` creates the base `M` dictionary, populating it with the input `data` and keyword arguments.
2.  **LHS Processing**: `_process_lhs!` processes the `outcomes` from the parser, sets the model architecture (`univariate`, `multivariate`), and resolves observation-level parameters like offsets and weights.
3.  **RHS Module Processing**: This is the core loop that iterates over the `modules` dictionary from the parser. For each module, it performs:
    *   **Processor Dispatch**: Calls the appropriate function from the `MODULE_PROCESSORS` dictionary (e.g., `process_spatial_module!`). These functions handle data-dependent setup, such as creating spatial indices or basis matrices.
	    *   **Primitive Resolution (for non-processor-only modules)**: `resolve_technical_primitive` is called to convert the parsed module data (a `Dict`) into a concrete `Manifold` struct instance (e.g., `BYM2(...)`). This step also resolves hyperpriors using `resolve_hyperpriors`. These processors can also evaluate complex arguments passed in the formula (e.g., `W=my_matrix`) by using the `calling_module` context stored in the configuration object `M`.
	    *   **Template Building (for non-processor-only modules)**: `build_model` is called on the `Manifold` object. This function is a factory that generates the technical specifications needed for the model, most importantly the precision matrix template (`Q_template`).
	    *   **Registration (for non-processor-only modules)**: The complete manifold specification (including the `Manifold` object and its `Q_template`) is added to `M[:manifolds]`.
	*   **Fixed Effects Processing**:
	    *   `_process_fixed_effects!`: Consolidates fixed effect variables from both bare terms (parsed directly from the formula) and explicit `fixed()` module calls. It then creates the design matrix `Xfixed` via `create_fixed_design`.
	    *   `_process_fixed_effects_priors!`: Constructs the prior for the fixed effect coefficients, incorporating any custom priors specified in `fixed()` or `intercept()` modules.
	*   **Intercept Resolution**: The final decision on whether to include an intercept (`M[:add_intercept]`) is prioritized from the `intercept()` module. If no `intercept()` module is present, it defaults to the legacy numeric flags (`1`, `0`, `-1`) parsed from the formula string.
5.  **Finalization**: `_finalize_config!` ensures all necessary keys exist in `M`, providing defaults where needed.

### 4.2. Key Configuration Helpers

*   **`resolve_hyperpriors(...)`**: Implements the three-level precedence for prior specification:
    1.  **Local**: A prior specified directly in a module call (e.g., `sigma_prior=...`).
    2.  **Global**: A prior specified in the `hyperpriors` dictionary passed to `bstm_config`.
    3.  **Scheme**: The default prior from the selected scheme (`:pcpriors`, `:informative`, `:uninformative`).
    It also handles the conversion of PC prior quantile constraints (e.g., `(1.0, 0.05)`) into `Distribution` objects via `create_pc_prior`.

*   **`build_structure_template(...)`**: A factory function that returns a precision matrix template (`Q`) for various GMRF models (`:icar`, `:rw2`, etc.), correctly handling scaling factors.

## 5. Turing Model Definitions (`modelling.jl`)

The `bstm` framework uses a set of core Turing `@model` definitions that are dynamically configured by the `M` object.

### 5.1. `bstm_univariate(M, ::Type{T})`

This is the model for single-outcome processes.

*   **Structure**:
    *   Defines global likelihood parameters (e.g., `lik_r` for Negative Binomial).
    *   Initializes the linear predictor `eta`.
    *   Adds the intercept and fixed effects contributions.
    *   Contains the main **manifold realization loop** (`for spec in M.manifolds`), which is the core of the model. Inside this loop:
        *   It dispatches on the type of the `Manifold` object (`m_obj`).
        *   It samples the hyperparameters for that manifold (e.g., `sigma_val`, `rho_val`).
        *   It samples the latent field itself, typically from an `MvNormal` or `MvNormalCanon` distribution, using the pre-computed `Q_template` and the sampled hyperparameters.
        *   It adds the contribution of the latent field to `eta`.
    *   Handles spatiotemporal interaction models (`model_st`).
    *   Handles nested sub-models from `M[:nested_manifolds]`.
    *   Defines the final observation likelihood using `y_obs ~ bstm_Likelihood(...)`.

### 5.2. `bstm_multivariate(M, ::Type{T})`

This model handles multiple, correlated outcomes.

*   **Key Differences**:
    *   **Outcome-Specific Parameters**: Hyperparameters are sampled for each outcome (e.g., `sigma_spatial_1`, `sigma_spatial_2`).
    *   **Latent Predictor Matrix**: The linear predictor `eta_latent` is a matrix of size `[Observations x Outcomes]`.
    *   **LKJ Correlation**: It samples a Cholesky factor `L_corr` from an `LKJCholesky` distribution. This matrix captures the correlation structure between the outcomes.
    *   **Coupling**: The final linear predictor `eta` is computed by transforming the matrix of independent latent fields: `eta = eta_latent * L_corr.L`. This induces the shared correlation structure.

## 6. Posterior Reconstruction Engine (`reconstruction.jl`)

The reconstruction engine is responsible for post-processing the MCMC `chain` to produce interpretable summaries and predictions.

*   **`_reconstruct(arch, ...)`**: The main entry point, which dispatches on the model architecture (`UnivariateArchitecture`, `MultivariateArchitecture`, etc.). It orchestrates the discovery and assembly of posterior effects.

*   **`_discover_manifold_realizations(...)`**: This is the core discovery function.
    *   It initializes containers for all possible latent effects (spatial, temporal, etc.).
    *   It iterates through the `M[:manifolds]` specification. For each manifold, it calls `extract_manifold`.
    *   **`extract_manifold(m_obj, ...)`**: This function dispatches on the `Manifold` type (`m_obj`). Each method knows how to find its parameters in the chain and reconstruct its specific effect. For example, `extract_manifold(m_obj::BYM2, ...)` finds the `sigma`, `rho`, `latent_struct`, and `latent_iid` samples and combines them to produce the structured, unstructured, and total spatial fields.
    *   **`_find_parameter` & `get_params_vector`**: These are the workhorse utilities for robustly extracting parameter samples from the MCMC chain, handling Turing's naming conventions and ensuring correct numerical ordering of indexed parameters.
    *   **Output**: Returns a `registry` `NamedTuple` containing the posterior samples for every latent field discovered in the model.

*   **`_modular_eta_assembly(...)`**: Takes the `registry` of discovered fields and reassembles the full linear predictor `eta` for each posterior sample. This process mirrors the assembly logic within the Turing model itself but operates on the posterior samples. It correctly handles both in-sample (`M`) and out-of-sample (`PS`) data.

*   **`_process_ll_and_predictions(...)`**: Takes the assembled `eta` samples and generates:
    *   **`p_denoised`**: The expected value of the response, obtained by applying the inverse link function to `eta`.
    *   **`p_noisy`**: Predictions that include observation noise, obtained by sampling from the predictive distribution.
    *   **`log_lik`**: The pointwise log-likelihood matrix, used for computing WAIC and LOO.

## 7. Key Utility Functions

*   **`assign_spatial_units(...)`**: Discretizes continuous spatial coordinates into areal units using various methods (`:cvt`, `:kvt`, `:bvt`, `:avt`, `:qvt`, `:hvt`, `:lattice`). It returns an object containing the adjacency matrix `W`, assignments, and centroids.

*   **`create_fixed_design(...)`**: A wrapper around `StatsModels.jl` that takes a formula string and a `DataFrame` and returns the corresponding design matrix `Xfixed`, correctly handling contrasts and factor variables.

*   **`get_optimal_sampler(...)`**: A utility that inspects the model's parameters and their prior distributions to construct an efficient composite `Gibbs` sampler. It assigns specialized samplers (`ESS`, `Slice`, `PG`) to different parameter blocks to improve MCMC efficiency.

*   **`get_inits(...)`**: Generates initial values for MCMC sampling. It can use a heuristic based on prior samples or run a fast MAP optimization to find a high-density starting point.

*   **`predict(...)`**: The main function for out-of-sample prediction. It creates a prediction-set configuration object (`PS`) by inheriting the structure of the training model (`M`) and updating it with the new data. It then calls `_reconstruct` to generate predictions on the new grid.

*   **`bstm_loo(...)` & `compare_manifolds(...)`**: Wrappers around `PosteriorStats.jl` for performing Leave-One-Out Cross-Validation and formal model comparison based on the ELPD metric.

## 8. API Reference: `bstm` Formula Modules

This section provides a quick reference to the main modules available in the `bstm` formula interface.

### 8.1. `likelihood()` Module

| Parameter                  | Example Usage        | Data Type | Default        | Meaning & Assumptions                                                                                               |
| :---------------------------| :---------------------| :----------| :---------------| :--------------------------------------------------------------------------------------------------------------------|
| `log_offsets` or `offsets` | `offsets=pop_log`   | `Symbol`  | None           | Provides a log-scale offset to the linear predictor ($\eta' = \eta + \text{offset}$). Essential for modeling rates. |
| `weights`                  | `weights=sample_w`  | `Symbol`  | `1.0`          | Multiplies the log-likelihood of each observation by the specified weight.                                          |
| `trials`                   | `trials=n_patients` | `Symbol`  | `1`            | Specifies the number of trials for each observation in a Binomial model.                                            |
| `volatility`               | `volatility=true`    | `Bool`    | `false`        | Enables a spatiotemporal stochastic volatility model for the observation noise ($\sigma_y$).                        |
| `y_L`, `y_U`               | `y_L=lower_b`       | `Symbol`  | `-Inf`, `+Inf` | Defines the lower (`y_L`) and upper (`y_U`) bounds for censored observations.                                       |
| `hurdle`                   | `hurdle=0`           | `Number`  | `-Inf`         | Implements a hurdle model by truncating the likelihood below the specified threshold.                               |

### 8.2. `spatial()` Module

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

### 8.3. `temporal()` and `seasonal()` Modules

| Manifold | `model='...'` | Key Parameters | Default PC-Priors | Use Case & Utility |
|:---|:---|:---|:---|:---|
| **IID** | `'iid'` | `sigma_prior` | `Exponential(1.0)` | Models unstructured temporal noise. |
| **AR1** | `'ar1'` | `sigma_prior`, `rho_prior` | `sigma`: `Exponential(1.0)`, `rho`: `Beta(1,1)` | Modeling serially correlated time series where the influence of past events decays geometrically. |
| **Random Walk (RW1)** | `'rw1'` | `sigma_prior` | `Exponential(1.0)` | Capturing abrupt changes or step-like trends. |
| **Random Walk (RW2)** | `'rw2'` | `sigma_prior` | `Exponential(1.0)` | The most common choice for modeling smooth, non-linear temporal trends. |
| **Gaussian Process** | `'gp'` | `sigma_prior`, `lengthscale_prior` | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Flexible, non-parametric trend modeling. |
| **Cyclic** | `'cyclic'` | `sigma_prior`, `period` | `Exponential(1.0)` | Modeling smooth, periodic effects like day-of-week or month-of-year. |
| **Harmonic** | `'harmonic'` | `amplitude_prior`, `phase_prior`, `period` | `amplitude`: `Normal(0,1)`, `phase`: `Beta(1,1)` | Capturing sharp, regular periodic patterns with sine and cosine waves. |

### 8.4. `smooth()` Module

| Manifold / Method | `model='...'` | Key Parameters | Default Priors | Use Case & Utility |
|:---|:---|:---|:---|:---|
| **P-Spline** | `'pspline'` | `nbins`, `degree`, `diff_order` | `sigma_prior`: `Exponential(1.0)` | The most flexible general-purpose smoother for 1D covariates. |
| **B-Spline** | `'bspline'` | `nbins`, `degree` | `sigma_prior`: `Exponential(1.0)` | A simpler spline smoother than P-splines, useful when less regularization is desired. |
| **Thin Plate Spline** | `'tps'` | `nbins` | `sigma_prior`: `Exponential(1.0)` | The classic choice for smoothing 2D spatial coordinates (e.g., `smooth(lon, lat, model=tps)`). |
| **Random Fourier Features** | `'rff'` | `n_features`, `lengthscale_prior` | `sigma_prior`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | A highly scalable method for approximating a full Gaussian Process smooth. |
| **Random Walk (on bins)** | `'rw1'`, `'rw2'` | `nbins` | `sigma_prior`: `Exponential(1.0)` | A powerful way to model a non-linear effect as a structured random effect on discretized bins. |

### 8.5. `mixed()` Module

| Syntax               | Example Usage    | Key Parameters | Default Priors | Mathematical Assumption           |
| :---------------------| :-----------------| :---------------| :---------------| :----------------------------------|
| **Random Intercept** | `mixed(1, group_var)`    | `model`        | `sigma_prior`: `Exponential(1.0)` | Assumes each level $j$ of `group_var` has a unique intercept $\alpha_j \sim \mathcal{N}(0, \sigma^2_{\text{group}})$.                        |
| **Random Slope**     | `mixed(covariate, group_var)`    | `model`        | `sigma_prior`: `Exponential(1.0)` | Assumes the effect (slope) of a `covariate` varies across the levels of `group_var`, $\beta_j \sim \mathcal{N}(0, \sigma^2_{\text{slope}})$. |

### 8.6. `dynamics()` Module

| Model | `model='...'` | Key Parameters | Default Priors |
|:---|:---|:---|:---|
| **Advection** | `'advection'` | `velocity_prior`, `sigma_prior` | `velocity`: `Normal(0,0.5)`, `sigma`: `Exponential(1.0)` |
| **Diffusion** | `'diffusion'` | `diffusion_prior`, `sigma_prior` | `diffusion`: `LogNormal(-1,1)`, `sigma`: `Exponential(1.0)` |
| **Logistic Growth** | `'logistic_f'` | `r_prior`, `K_prior`, `sigma_F_prior` | `r`: `LogNormal(0,1)`, `K`: `Normal(150,50)`, `sigma_F`: `Exponential(0.5)` |

### 8.7. `nested()` and `eigen()` Modules

#### `nested()` Module Reference

| Keyword / Parameter     | Example Usage                  | Data Type | Default            | Meaning & Assumptions                                                                                                                                                                                                                                 |
| :------------------------| :-------------------------------| :----------| :-------------------| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nested()`              | `nested(z_var; ...)`           | Module    | N/A                | Defines a supervised sub-model whose latent effect is added to the main model's linear predictor. The `z_var` is a symbolic name for this component.                                                                                                   |
| `formula`               | `formula="likelihood(z, family=gaussian) ~ 1 + spatial(s)"` | `String`  | `""`               | A complete `bstm` formula string that defines the structure of the sub-model, including its own likelihood. This sub-model is fit to the specified `data_source`.                                                                                                                   |
| `data_source`           | `data_source=proxy_data`      | `Symbol`  | `:data`            | A symbol pointing to a `DataFrame` passed as a keyword argument to the main `bstm()` call. This allows the sub-model to use a different dataset.                                                                                                       |
| `rho_nested` (Implicit) | N/A                            | `Float`   | `Normal(1.0, 0.5)` | A scaling coefficient that links the sub-model's latent effect to the main model's linear predictor: $\eta_{\text{main}} = \dots + \rho_{\text{nested}} \cdot \eta_{\text{sub}}$. The prior assumes the sub-model is a good proxy ($\rho \approx 1$).  |

#### `eigen()` Module Reference

| Keyword / Parameter | Example Usage                    | Data Type      | Default            | Meaning & Assumptions                                                                                                                                                               |
| :--------------------| :---------------------------------| :---------------| :-------------------| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `eigen()`           | `eigen(y1, y2, y3; ...)`         | Module         | N/A                | Defines a Bayesian PCA factor model. The variables listed (e.g., `y1, y2, y3`) are the multivariate outcomes to be decomposed.                                                      |
| `n_factors`         | `n_factors=1`                    | `Int`          | `1`                | The number of latent factors (principal components) to extract. This determines the dimensionality of the reduced latent space.                                                     |
| `pca_sd_prior`      | `pca_sd_prior=Exponential(0.5)`  | `Distribution` | `Exponential(1.0)` | The prior for the standard deviations of the principal components (latent factors). These are the "eigenvalues" of the system, controlling the variance explained by each factor.   |
| `pdef_sd_prior`     | `pdef_sd_prior=Exponential(0.5)` | `Distribution` | `Exponential(1.0)` | The prior for the standard deviation of the residual (uniqueness) noise. This captures the variance in each observed variable that is *not* explained by the shared latent factors. |

### 8.8. `fixed()` and `intercept()` Modules

#### `fixed()` Module Reference

| Keyword / Parameter | Example Usage | Data Type | Default | Meaning & Assumptions |
|:---|:---|:---|:---|:---|
| `fixed()` | `fixed(Region, ...)` | Module | N/A | Explicitly marks a variable as a fixed effect. Primarily used to specify contrasts or priors. |
| `contrast` | `contrast=:effects` | `Symbol` | `DummyCoding` | Specifies the contrast coding for a categorical variable (e.g., `:effects`, `:helmert`). |
| `prior` | `prior=Normal(0, 2)` | `Distribution` or `Tuple` | `Normal(0, 5)` | Sets the prior for the coefficient(s) of this fixed effect. Can be a `Distribution` or a PC prior tuple. |

#### `intercept()` Module Reference

| Keyword / Parameter | Example Usage          | Data Type                 | Default        | Meaning & Assumptions                                                                                                                |
| :--------------------| :-----------------------| :--------------------------| :---------------| :-------------------------------------------------------------------------------------------------------------------------------------|
| `intercept()`       | `intercept(prior=...)` | Module                    | N/A            | Explicitly includes a global intercept. Using `1` in the formula is equivalent. This module is mainly for specifying a custom prior. |
| `prior`             | `prior=Normal(0, 10)`  | `Distribution` or `Tuple` | `Normal(0, 5)` | Sets the prior for the global intercept term. Can be a `Distribution` or a PC prior tuple.                                           |

## 9. Detailed Modeling Examples

### 9.1. Fixed and Mixed Effects with Custom Priors and Contrasts

This example demonstrates how to build a model incorporating both fixed and mixed effects, with a focus on adjusting their parameters using custom contrast coding and Penalized Complexity (PC) priors directly within the `bstm` formula.

The model will estimate a response `y` based on:
*   A global intercept.
*   A continuous fixed effect (`temperature`).
*   A categorical fixed effect (`region`) with custom `effects` contrast coding and a specific prior on its coefficients.
*   A random intercept for each `site`, where sites are nested within regions. The variance of these random intercepts will also be controlled by a PC prior.

#### 1. Data Simulation

First, a synthetic dataset is generated to provide a clear ground truth for the model to recover.

```julia
using bstm
using Turing
using DataFrames
using Random
using Distributions
using CategoricalArrays
using LinearAlgebra

# Set a seed for reproducibility
Random.seed!(123)

# Define the structure of the data
n_regions = 4
n_sites_per_region = 5
n_obs_per_site = 10
n_total = n_regions * n_sites_per_region * n_obs_per_site

# True parameter values
true_intercept = 10.0
true_temp_effect = 1.5
true_region_effects = [-1.5, 0.5, 1.0, 0.0] # Sum to zero for effects coding
true_site_sd = 0.8  # Std dev of random intercepts
true_obs_noise = 0.5

# Generate predictors
regions = repeat(["R1", "R2", "R3", "R4"], inner=n_sites_per_region * n_obs_per_site)
sites = repeat(1:(n_regions * n_sites_per_region), inner=n_obs_per_site)
temperature = rand(Normal(20, 5), n_total)

# Generate true latent effects
region_map = Dict("R1"=>1, "R2"=>2, "R3"=>3, "R4"=>4)
latent_region_effect = [true_region_effects[region_map[r]] for r in regions]

true_site_intercepts = rand(Normal(0, true_site_sd), n_regions * n_sites_per_region)
latent_site_effect = true_site_intercepts[sites]

# Generate the response variable
latent_mean = true_intercept .+ (temperature .* true_temp_effect) .+ latent_region_effect .+ latent_site_effect
y = latent_mean .+ rand(Normal(0, true_obs_noise), n_total)

# Create the DataFrame
sim_data = DataFrame(
    y = y,
    temperature = temperature,
    region = categorical(regions),
    site = categorical(sites)
)
```

#### 2. Model Formulation with Custom Priors and Contrasts

The formula is constructed to explicitly define the model structure, including priors and contrast coding for the fixed and mixed effects.

```julia
# The formula string defines the complete model structure.
# Note: The `fixed()` module is used for categorical variables when specifying
# contrasts or priors. Continuous variables can be included directly.

fx_mx_formula = """
    likelihood(y, family=gaussian) ~
        intercept(prior=Normal(0, 20)) +
        temperature +
        fixed(region, contrast=effects, prior=(3.0, 0.05)) +
        mixed(1 | site, prior=(1.5, 0.01))
"""

# Explanation of the formula components:
#
# - `intercept(prior=Normal(0, 20))`:
#   Specifies a global intercept with a wide Normal prior.
#
# - `temperature`:
#   Includes `temperature` as a standard continuous fixed effect. Its coefficient
#   will receive the default prior (e.g., Normal(0, 5)).
#
# - `fixed(region, contrast=effects, prior=(3.0, 0.05))`:
#   - `fixed(region, ...)`: Explicitly defines `region` as a fixed effect.
#   - `contrast=effects`: Sets the contrast coding to "effects" (or sum-to-zero) coding.
#     This means the coefficients for the regions will represent deviations from the
#     grand mean, and they will be constrained to sum to zero.
#   - `prior=(3.0, 0.05)`: Specifies a Penalized Complexity (PC) prior for the `region`
#     coefficients. This tuple is interpreted as `P(|beta| > 3.0) = 0.05`, providing
#     regularization that shrinks the coefficients towards zero unless the data
#     strongly supports a large effect.
#
# - `mixed(1 | site, prior=(1.5, 0.01))`:
#   - `mixed(1 | site)`: Defines a random intercept model, where each level of the `site`
#     variable gets its own intercept, drawn from a common distribution.
#   - `prior=(1.5, 0.01)`: Specifies a PC prior on the standard deviation of the random
#     intercepts (`sigma_site`). This is interpreted as `P(sigma_site > 1.5) = 0.01`,
#     shrinking the group-level variance towards zero.
```

#### 3. Model Execution and Inference

The model is instantiated and sampled using Turing.jl's NUTS sampler.

```julia
# Instantiate the bstm model
# The formula parser automatically handles the fixed() and mixed() modules.
m = @bstm(
    fx_mx_formula,
    sim_data
);

# Sample from the posterior distribution
# NUTS is recommended for its efficiency with continuous parameters.
chain = sample(m, NUTS(1000, 0.8), 2000; progress=true);
```

#### 4. Results Interpretation

The posterior summary provides estimates for all model parameters. We can inspect the fixed effect coefficients and the variance of the mixed effect.

```julia
# Display the summary statistics of the posterior chain
summary_stats = summarize(chain)
display(summary_stats)

# Key parameters to inspect:
#
# - `intercept`: Should be close to the `true_intercept` of 10.0.
#
# - `Xfixed_beta[1]`: The coefficient for `temperature`. Should be close to `true_temp_effect` of 1.5.
#
# - `Xfixed_beta[2]`, `Xfixed_beta[3]`, `Xfixed_beta[4]`: The coefficients for the `region` levels.
#   Due to effects coding, these represent the deviation of each region from the intercept.
#   For example, the effect for R1 should be near -1.5.
#
# - `sigma_mixed_site`: The standard deviation of the random intercepts for `site`.
#   This should be close to the `true_site_sd` of 0.8.
#
# - `latent_mixed_site`: This vector contains the posterior means for the individual random
#   intercepts for each site.
```

This example illustrates the flexibility of the `bstm` formula interface in specifying complex hierarchical models with fine-grained control over priors and parameterization, all within a readable and self-contained syntax.



## References

*   Besag, J., York, J., & Mollié, A. (1991). Bayesian image restoration, with applications in spatial statistics. *Annals of the Institute of Statistical Mathematics*, 43(1), 1-20.
*   Riebler, A., Sørbye, S. H., & Rue, H. (2016). An intuitive Bayesian spatial model with two hyperparameters. *Statistical Methods in Medical Research*, 25(2), 1145-1160.
*   Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time variation in disease risk. *Statistical Methods in Medical Research*, 9(3), 205-220.
*   Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
*   Rahimi, A., & Recht, B. (2008). Random features for large-scale kernel machines. *Advances in Neural Information Processing Systems*, 20.
*   Lindgren, F., Rue, H., & Lindström, J. (2011). An explicit link between Gaussian fields and Gaussian Markov random fields: The SPDE approach. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 73(4), 423-498.
