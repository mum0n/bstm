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
*   **`sigma`**: A `UnivariateDistribution` that defines the prior for the marginal standard deviation (scale) of the latent field. It controls the overall magnitude of the effect.
*   **`rho`**: A `UnivariateDistribution` for a correlation or mixing parameter, typically on `[0, 1]`. In `BYM2` or `Leroux` models, it controls the proportion of variance attributed to the structured spatial component. In `AR1` models, it represents the temporal autocorrelation.
*   **`lengthscale`**: A `UnivariateDistribution` for the lengthscale parameter in continuous Gaussian Process models (`GP`, `RFF`, etc.). It controls the distance over which points are correlated; a larger lengthscale implies a smoother, more global process.
*   **`kappa`**: A `UnivariateDistribution` for the `kappa` parameter in `SPDE` (Matérn) models, which is inversely related to the lengthscale and controls the smoothness of the field. 
*   **`n_features` / `nbins`**: An `Int` specifying the number of basis functions (for `RFF`) or bins (for `PSpline`) used in approximation methods.
*   **`degree` / `diff_order`**: `Int` parameters for spline models, controlling the polynomial degree of the basis functions and the order of the difference penalty for smoothing.
*   **`pca_sd`**: Prior for the standard deviations of the principal components (latent factors) in `Eigen` models.
*   **`pdef_sd`**: Prior for the standard deviation of the residual (uniqueness) noise in `Eigen` models.

```julia
# --- Discrete & Graph Primitives ---
struct IID <: ManifoldModel; sigma::UnivariateDistribution; end
struct ICAR <: ManifoldModel; sigma::UnivariateDistribution; end
struct Besag <: ManifoldModel; sigma::UnivariateDistribution; end
struct BYM2 <: ManifoldModel
    rho::UnivariateDistribution       # Prior for the mixing parameter between structured and unstructured effects.
    sigma::UnivariateDistribution     # Prior for the overall spatial standard deviation.
end
struct Leroux <: ManifoldModel; rho::UnivariateDistribution; sigma::UnivariateDistribution; end # Proper CAR model
struct SAR <: ManifoldModel; rho::UnivariateDistribution; sigma::UnivariateDistribution; end # Simultaneous Autoregressive model

# --- Temporal Primitives ---
struct RW1 <: ManifoldModel; sigma::UnivariateDistribution; end
struct RW2 <: ManifoldModel; sigma::UnivariateDistribution; end
struct AR1 <: ManifoldModel; rho::UnivariateDistribution; sigma::UnivariateDistribution; end
struct Cyclic <: ManifoldModel; period::Int; sigma::UnivariateDistribution; end
struct Harmonic <: ManifoldModel
    amplitude::UnivariateDistribution
    phase::UnivariateDistribution
    period::Union{Real, UnivariateDistribution}
end

# --- Continuous, Spectral, and Advanced Manifolds ---
struct GP <: ManifoldModel
    lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}} # Prior for lengthscale(s). Can be a single value for isotropic kernels or a vector for ARD kernels.
    sigma::UnivariateDistribution     # Prior for the GP's signal variance.
    kernel::String                          # Name of the kernel function (e.g., "se", "matern32").
end
struct RFF <: ManifoldModel
    lengthscale::UnivariateDistribution # Prior for the lengthscale of the approximated kernel.
    sigma::UnivariateDistribution     # Prior for the signal variance.
    n_features::Int                         # Number of random features to use in the approximation.
    kernel::String                          # The kernel being approximated.
end
struct FITC <: ManifoldModel; lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; sigma::UnivariateDistribution; n_inducing::Int; kernel::String; end
struct SVGP <: ManifoldModel; lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; sigma::UnivariateDistribution; n_inducing::Int; kernel::String; end
struct Nystrom <: ManifoldModel; lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; sigma::UnivariateDistribution; n_inducing::Int; kernel::String; end
struct SPDE <: ManifoldModel; sigma::UnivariateDistribution; kappa::UnivariateDistribution; end
struct FFT <: ManifoldModel; sigma::UnivariateDistribution; nbins::Int; kernel::String; lengthscale::UnivariateDistribution; end
struct Warp <: ManifoldModel; lengthscale::UnivariateDistribution; sigma::UnivariateDistribution; n_features::Int; kernel::String; end
struct Hyperbolic <: ManifoldModel; curvature::Real; sigma::UnivariateDistribution; end
struct ExponentialDecay <: ManifoldModel; sigma::UnivariateDistribution; lengthscale::UnivariateDistribution; end

# --- Spline-based Manifolds ---
struct PSpline <: ManifoldModel; nbins::Int; degree::Int; diff_order::Int; sigma::UnivariateDistribution; end
struct BSpline <: ManifoldModel; nbins::Int; degree::Int; sigma::UnivariateDistribution; end
struct TPS <: ManifoldModel; nbins::Int; sigma::UnivariateDistribution; end

# --- Specialized & Basis Manifolds ---
struct Wavelet <: ManifoldModel; family::Symbol; nbins::Int; sigma::UnivariateDistribution; lengthscale::UnivariateDistribution; end
struct Eigen <: ManifoldModel
    n_vars::Int                             # Number of input variables for PCA.
    n_factors::Int                          # Number of latent factors to extract.
    pca_sd::UnivariateDistribution    # Prior for the standard deviations of the principal components.
    pdef_sd::UnivariateDistribution   # Prior for the standard deviation of the residual (uniqueness) noise.
    ltri_indices::Vector{Int}               # Indices for the lower-triangular part of the Householder reflector matrix.
end
struct Moran <: ManifoldModel; sigma::UnivariateDistribution; end
struct Spherical <: ManifoldModel; sigma::UnivariateDistribution; range::UnivariateDistribution; end
struct Barycentric <: ManifoldModel; sigma::UnivariateDistribution; end
struct BCGN <: ManifoldModel; sigma::UnivariateDistribution; bipartite_adj::AbstractMatrix; end
struct NetworkFlow <: ManifoldModel; sigma::UnivariateDistribution; adjacency_matrix::AbstractMatrix; flow_direction::Symbol; end
struct LocalAdaptive <: ManifoldModel; rho::UnivariateDistribution; sigma::UnivariateDistribution; end
struct Mosaic <: ManifoldModel; sigma::UnivariateDistribution; n_regions::Int; end
struct TensorProductSmooth <: ManifoldModel; sigma::UnivariateDistribution; Q_template::AbstractMatrix; end
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
    *   v1.2.1 (2026-07-16)
    *   **Output**: Returns a `NamedTuple` containing `:outcomes`, `:modules` (a dictionary of all parsed RHS terms), `:fixed_effects` (a list of bare variable names parsed from the RHS), `:has_intercept`, and `:intercept`.

*   **`_parse_rhs_expression(term_str)`**: A recursive descent parser that respects operator precedence to build the AST for the RHS. The precedence is:
    The parser is called on sub-expressions after the formula has been split by the `+` operator (which has the lowest precedence). The parser then handles operators in the following order of precedence (from highest to lowest):
    1.  `|>` (Pipe for state-space models)
    2.  `⊗` (Kronecker Product)
    3.  `∘` (Composition)

*   **`_categorize_rhs_nodes!(nodes, modules, fixed_effects)`**: Traverses the generated AST to populate the `modules` dictionary and the `fixed_effects` list. It correctly identifies composed manifolds (e.g., `spatial() ⊗ temporal()`) as a single interaction module.


*   **`_parse_single_manifold_term` & `_parse_arguments_string`**: Helper functions that parse individual module calls (e.g., `spatial(s_idx, model=:bym2)`) into a dictionary of variables and parameters. These functions correctly handle Julia `Symbol` literals (e.g., `:besag`) passed as arguments.
## 4. Model Configuration Engine (`modelling.jl`)


The `bstm_config` function is the main engine that transforms the parsed formula and data into a complete configuration object (`M`) for the Turing models.

### 4.1. `bstm_config` Workflow

1.  **Initialization**: `_initialize_config` creates the base `M` dictionary, populating it with the input `data` and keyword arguments.
2.  **LHS Processing**: `_process_lhs!` processes the `outcomes` from the parser, sets the model architecture (`univariate`, `multivariate`), and resolves observation-level parameters like offsets and weights.
3.  **RHS Module Processing**: This is the core loop that iterates over the `modules` dictionary from the parser. For each module, it performs:
    *   **Processor Dispatch**: Calls the appropriate function from the `MODULE_PROCESSORS` dictionary (e.g., `process_spatial_module!`). These functions handle data-dependent setup, such as creating spatial indices or basis matrices.
	    *   **Primitive Resolution (for non-processor-only modules)**: `resolve_technical_primitive` is called to convert the parsed module data (a `Dict`) into a concrete `Manifold` struct instance (e.g., `BYM2(...)`). This step also resolves hyperpriors using `resolve_hyperpriors`. These processors can also evaluate complex arguments passed in the formula (e.g., `W=my_matrix`) by using the `calling_module` context stored in the configuration object `M`.
	    *   **Template Building (for non-processor-only modules)**: `build_model` is called on the `Manifold` object. This function is a factory that generates the technical specifications needed for the model, most importantly the precision matrix template (`Q_template`).
	    *   **Registration (for non-processor-only modules)**: The complete manifold specification (including the `Manifold` object and its `Q_template`) is added to `M[:manifolds]`. The `process_interact_module!` for `⊗` also sets the global `M[:model_st]` parameter to ensure the interaction is included in the model.
	*   **Fixed Effects Processing**:
	    *   `_process_fixed_effects!`: Consolidates fixed effect variables from both bare terms (parsed directly from the formula) and explicit `fixed()` module calls. It then create io es
	*   **Intercept Resolution**: The final decision on whether to include an intercept (`M[:add_intercept]`) is prioritized from the `intercept()` module. If no `intercept()` module is present, it defaults to the legacy numeric flags (`1`, `0`, `-1`) parsed from the formula string.
5.  **Finalization**: `_finalize_config!` ensures all necessary keys exist in `M`, providing defaults where needed.

### 4.2. Key Configuration Helpers

v1.2.1 (2026-07-16)
*   **`resolve_hyperpriors(...)`**: Implements the three-level precedence for prior specification:
    1.  **Local**: A prior specified directly in a module call (e.g., `sigma=...`).
    2.  **Global**: A prior specified in the `hyperpriors` dictionary passed to `bstm_config`.
    3.  **Scheme**: The default prior from the selected scheme (`:pcpriors`, `:informative`, `:uninformative`).
    It also handles the conversion of PC prior quantile constraints (e.g., `(1.0, 0.05)`) into `Distribution` objects via `create_pc_prior`.

*   **`build_structure_template(...)`**: A factory function that returns a precision matrix template (`Q`) for various GMRF models (`:icar`, `:rw2`, etc.), correctly handling scaling factors.

## 5. Turing Model Definitions (`modelling.jl`)

The `bstm` framework uses a set of core Turing `@model` definitions that are dynamically configured by the `M` object.

### 5.1. `bstm_univariate(M, ::Type{T})`

v1.2.1 (2026-07-16)
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

v1.2.1 (2026-07-16)
This model handles multiple, correlated outcomes.

*   **Key Differences**:
    *   **Outcome-Specific Parameters**: Hyperparameters are sampled for each outcome (e.g., `sigma_spatial_1`, `sigma_spatial_2`).
    *   **Latent Predictor Matrix**: The linear predictor `eta_latent` is a matrix of size `[Observations x Outcomes]`.
    *   **LKJ Correlation**: It samples a Cholesky factor `L_corr` from an `LKJCholesky` distribution. This matrix captures the correlation structure between the outcomes.
    *   **Coupling**: The final linear predictor `eta` is computed by transforming the matrix of independent latent fields: `eta = eta_latent * L_corr.L`. This induces the shared correlation structure.

## 6. Posterior Reconstruction Engine (`reconstruction.jl`)

The reconstruction engine is responsible for post-processing the MCMC `chain` to produce interpretable summaries and predictions.

*   **`_reconstruct(arch, ...)`**: The main entry point, which dispatches on the model architecture (`UnivariateArchitecture`, `MultivariateArchitecture`, etc.). It orchestrates the discovery and assembly of posterior effects.

v1.2.1 (2026-07-16)
*   **`_discover_manifold_realizations(...)`**: This is the core discovery function.
    *   It initializes containers for all possible latent effects (spatial, temporal, etc.).
    *   It iterates through the `M[:manifolds]` specification. For each manifold, it calls `extract_manifold`.
    *   **`extract_manifold(m_obj, ...)`**: This function dispatches on the `Manifold` type (`m_obj`). Each method knows how to find its parameters in the chain and reconstruct its specific effect.
        v1.2.1 (2026-07-16)
        *   For simple manifolds like `BYM2`, it finds `sigma`, `rho`, `latent_struct`, and `latent_iid` samples and combines them to produce the structured, unstructured, and total spatial fields.
        *   For a `ComposedManifold` with a `kronecker_product` operator, it reconstructs the interaction field by finding the corresponding latent field and hyperparameters in the chain and applying the correct scaling and reshaping.

*   **`_modular_eta_assembly(...)`**: Takes the `registry` of discovered fields and reassembles the full linear predictor `eta` for each posterior sample. This process mirrors the assembly logic within the Turing model itself but operates on the posterior samples. It correctly handles both in-sample (`M`) and out-of-sample (`PS`) data.

*   **`_process_ll_and_predictions(...)`**: Takes the assembled `eta` samples and generates:
    *   **`p_denoised`**: The expected value of the response, obtained by applying the inverse link function to `eta`.
    *   **`p_noisy`**: Predictions that include observation noise, obtained by sampling from the predictive distribution.
    *   **`log_lik`**: The pointwise log-likelihood matrix, used for computing WAIC and LOO.

## 7. Key Utility Functions

*   **`assign_spatial_units(...)`**: Discretizes continuous spatial coordinates into areal units using various methods (`:cvt`, `:kvt`, `:bvt`, `:avt`, `:qvt`, `:hvt`, `:lattice`). It returns an object containing the adjacency matrix `W`, assignments, and centroids.

v1.2.1 (2026-07-16)
*   **`create_fixed_design(...)`**: A wrapper around `StatsModels.jl` that takes a formula string and a `DataFrame` and returns the corresponding design matrix `Xfixed`, correctly handling contrasts and factor variables.

*   **`get_optimal_sampler(...)`**: A utility that inspects the model's parameters and their prior distributions to construct an efficient composite `Gibbs` sampler. It assigns specialized samplers (`ESS`, `Slice`, `PG`) to different parameter blocks to improve MCMC efficiency.
    *   **Purpose**: To automatically construct an efficient composite Gibbs sampler tailored to the `bstm` model structure.
    *   **Rationale**: Different MCMC algorithms exhibit varying performance depending on the characteristics of the target distribution. A composite `Gibbs` sampler, which applies different samplers to different blocks of parameters, can significantly outperform a single, general-purpose sampler. This function automates the construction of such a sampler.
    *   **Workflow**:
        v1.2.1 (2026-07-16)
        1.  **Manual Override**: The function first checks if a specific sampler has been provided via the `sampler_choice` argument or if a `sampler_map` dictionary has been passed to assign specific samplers to certain parameters. These manual assignments take the highest precedence.
        2.  **Manifold Grouping**: If `group_manifolds=true` (the default), the function identifies all parameters that belong to the same manifold instance (e.g., `spatial_main_sigma`, `spatial_main_rho`, and `spatial_main_latent`). It groups these parameters into a single block and assigns a `NUTS` sampler to them. This is a critical step for efficiency, as it allows the sampler to jointly explore the highly correlated posterior geometry of a latent field and its hyperparameters, reducing the "funnel" problems common in hierarchical models.
        3.  **Default Parameter Categorization**: For all remaining parameters that were not part of a manifold group or manual assignment, the function categorizes them based on their prior distributions:
            *   `:discrete`: Parameters with discrete support (e.g., from `Categorical` or `Poisson` priors).
            *   `:gaussian`: Parameters with `Normal` or `MvNormal` priors.
            *   `:bounded`: Continuous parameters with one or two-sided bounds (e.g., from `Uniform`, `Beta`, `Exponential`, `InverseGamma`).
            *   `:other_continuous`: All other unbounded, non-Gaussian continuous parameters.
        4.  **Sampler Assignment**: It assigns an optimal MCMC algorithm to each category:
            *   `PG` (Particle Gibbs) is assigned to `:discrete` parameters.
            *   `ESS` (Elliptical Slice Sampler) is assigned to `:gaussian` parameters. This is highly efficient for latent Gaussian models.
            *   `Slice` sampling is assigned to `:bounded` parameters, as it robustly handles the boundaries without requiring gradient information.
            *   `NUTS` is assigned to all remaining `:other_continuous` parameters.
        5.  **Composite Sampler Construction**: Finally, it combines all the individually assigned samplers (from the manual map, manifold groups, and default categories) into a single `Gibbs(...)` sampler object, which is then returned.

v1.2.1 (2026-07-16)
*   **`get_inits(...)`**: Generates initial values for MCMC sampling. It can use a heuristic based on prior samples or run a fast MAP optimization to find a high-density starting point.

v1.2.1 (2026-07-16)
*   **`predict(...)`**: The main function for out-of-sample prediction. It creates a prediction-set configuration object (`PS`) by inheriting the structure of the training model (`M`) and updating it with the new data. It then calls `_reconstruct` to generate predictions on the new grid.

v1.2.1 (2026-07-16)
*   **`bstm_loo(...)` & `compare_manifolds(...)`**: Wrappers around `PosteriorStats.jl` for performing Leave-One-Out Cross-Validation and formal model comparison based on the ELPD metric.

## 8. API Reference: `bstm` Formula Modules

This section provides a quick reference to the main modules available in the `bstm` formula interface.

### General Keyword Arguments

In addition to the module-specific parameters, the main `@bstm` call accepts general keyword arguments. A key argument is `verbose=false`, which suppresses the printing of the generated model code and the prior predictive check results upon model instantiation.

### 8.1. `likelihood()` Module

| Parameter                      | Example Usage          | Data Type            | Default        | Meaning & Assumptions                                                                                               |
| :-------------------------------| :-----------------------| :---------------------| :---------------| :--------------------------------------------------------------------------------------------------------------------|
| `family`                       | `family=:poisson`      | `Symbol`             | `:gaussian`    | Sets the likelihood distribution. See table below for options.                                                      |
| `log_offsets`                  | `log_offsets=pop_log`  | `Symbol`             | `0.0`          | Provides a log-scale offset to the linear predictor ($\eta' = \eta + \text{offset}$). Essential for modeling rates. |
| `weights`                      | `weights=sample_w`     | `Symbol`             | `1.0`          | Multiplies the log-likelihood of each observation by the specified weight.                                          |
| `trials`                       | `trials=n_patients`    | `Symbol`             | `1`            | Specifies the number of trials for each observation in a Binomial model.                                            |
| `zero_inflated`                | `zero_inflated=true`   | `Bool`               | `false`        | Enables a zero-inflation component for count models.                                                                |
| `volatility`                   | `volatility=true`      | `Bool`               | `false`        | Enables a spatiotemporal stochastic volatility model for the observation noise ($\sigma_y$).                        |
| `censor_lower`, `censor_upper` | `censor_lower=lower_b` | `Symbol` or `Number` | `-Inf`, `+Inf` | Defines the lower (`censor_lower`) and upper (`censor_upper`) bounds for censored data.                             |
| `hurdle`                       | `hurdle=0`             | `Number`             | `-Inf`         | Implements a hurdle model by truncating the likelihood below the specified threshold.                               |

### 8.2. `spatial()` Module

| Manifold             | `model='...'`       | Key Parameters                       | Default PC-Priors                                               | Use Case & Utility                                                                                     |
| :---------------------| :--------------------| :-------------------------------------| :----------------------------------------------------------------| :-------------------------------------------------------------------------------------------------------|
| **IID**              | `'iid'`             | `sigma`                              | `Exponential(1.0)`                                              | Models non-spatial overdispersion or heterogeneity.                                                    |
| **ICAR / Besag**     | `'icar'`, `'besag'` | `sigma`                              | `Exponential(1.0)`                                              | Provides strong, localized spatial smoothing for lattice data.                                         |
| **BYM2**             | `'bym2'`            | `sigma`, `rho`                       | `sigma`: `Exponential(1.0)`, `rho`: `Beta(1,1)`                 | The most robust default for areal data; separates spatial clustering from random noise.                |
| **Leroux**           | `'leroux'`          | `sigma`, `rho`                       | `sigma`: `Exponential(1.0)`, `rho`: `Beta(1,1)`                 | A flexible alternative to BYM2 that avoids the rank-deficiency of the ICAR model.                      |
| **SAR**              | `'sar'`             | `sigma`, `rho`                       | `sigma`: `Exponential(1.0)`, `rho`: `Beta(1,1)`                 | Models spatial "spill-over" effects where the value at one location directly influences its neighbors. |
| **SPDE**             | `'spde'`            | `sigma`, `kappa`                     | `sigma`: `Exponential(1.0)`, `kappa`: `Exponential(1.0)`        | A scalable and principled way to model continuous spatial processes on irregular domains.              |
| **Gaussian Process** | `'gp'`              | `sigma`, `lengthscale`, `kernel`     | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Gold-standard for continuous spatial modeling but computationally expensive ($O(N^3)$).                |
| **RFF**              | `'rff'`             | `sigma`, `lengthscale`, `n_features` | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | A scalable approximation to a full GP, excellent for large numbers of areal units.                     |

### 8.3. `temporal()` and `seasonal()` Modules

| Manifold              | `model='...'` | Key Parameters                 | Default PC-Priors                                               | Use Case & Utility                                                                                |
| :----------------------| :--------------| :-------------------------------| :----------------------------------------------------------------| :--------------------------------------------------------------------------------------------------|
| **IID**               | `'iid'`       | `sigma`                        | `Exponential(1.0)`                                              | Models unstructured temporal noise.                                                               |
| **AR1**               | `'ar1'`       | `sigma`, `rho`                 | `sigma`: `Exponential(1.0)`, `rho`: `Beta(1,1)`                 | Modeling serially correlated time series where the influence of past events decays geometrically. |
| **Random Walk (RW1)** | `'rw1'`       | `sigma`                        | `Exponential(1.0)`                                              | Capturing abrupt changes or step-like trends.                                                     |
| **Random Walk (RW2)** | `'rw2'`       | `sigma`                        | `Exponential(1.0)`                                              | The most common choice for modeling smooth, non-linear temporal trends.                           |
| **Gaussian Process**  | `'gp'`        | `sigma`, `lengthscale`         | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Flexible, non-parametric trend modeling.                                                          |
| **Cyclic**            | `'cyclic'`    | `sigma`, `period`              | `Exponential(1.0)`                                              | Modeling smooth, periodic effects like day-of-week or month-of-year.                              |
| **Harmonic**          | `'harmonic'`  | `amplitude`, `phase`, `period` | `amplitude`: `Normal(0,1)`, `phase`: `Beta(1,1)`                | Capturing sharp, regular periodic patterns with sine and cosine waves.                            |

### 8.4. `smooth()` Module (Covariate Smoothers)

| Manifold / Method           | `model='...'`    | Key Parameters                  | Default Priors                                                  | Use Case & Utility                                                                             |
| :----------------------------| :-----------------| :--------------------------------| :----------------------------------------------------------------| :-----------------------------------------------------------------------------------------------|
| **P-Spline**                | `'pspline'`      | `nbins`, `degree`, `diff_order` | `sigma`: `Exponential(1.0)`                                     | The most flexible general-purpose smoother for 1D covariates.                                  |
| **B-Spline**                | `'bspline'`      | `nbins`, `degree`               | `sigma`: `Exponential(1.0)`                                     | A simpler spline smoother than P-splines, useful when less regularization is desired.          |
| **Thin Plate Spline**       | `'tps'`          | `nbins`                         | `sigma`: `Exponential(1.0)`                                     | The classic choice for smoothing 2D spatial coordinates (e.g., `smooth(lon, lat, model=tps)`). |
| **Random Fourier Features** | `'rff'`          | `n_features`, `lengthscale`     | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | A highly scalable method for approximating a full Gaussian Process smooth.                     |
| **Random Walk (on bins)**   | `'rw1'`, `'rw2'` | `nbins`                         | `sigma`: `Exponential(1.0)`                                     | A powerful way to model a non-linear effect as a structured random effect on discretized bins. |

### 8.5. `mixed()` Module

| Syntax               | Example Usage                 | Key Parameters | Default Priors              | Mathematical Assumption                                                                                                                      |
| :---------------------| :------------------------------| :---------------| :----------------------------| :---------------------------------------------------------------------------------------------------------------------------------------------|
| **Random Intercept** | `mixed(1, group_var)`         | `model`        | `sigma`: `Exponential(1.0)` | Assumes each level $j$ of `group_var` has a unique intercept $\alpha_j \sim \mathcal{N}(0, \sigma^2_{\text{group}})$.                        |
| **Random Slope**     | `mixed(covariate, group_var)` | `model`        | `sigma`: `Exponential(1.0)` | Assumes the effect (slope) of a `covariate` varies across the levels of `group_var`, $\beta_j \sim \mathcal{N}(0, \sigma^2_{\text{slope}})$. |

### 8.6. `dynamics()` Module

| Model               | `model='...'`  | Key Parameters       | Default Priors                                                              |
| :--------------------| :---------------| :---------------------| :----------------------------------------------------------------------------|
| **Advection**       | `'advection'`  | `velocity`, `sigma`  | `velocity`: `Normal(0,0.5)`, `sigma`: `Exponential(1.0)`                    |
| **Diffusion**       | `'diffusion'`  | `diffusion`, `sigma` | `diffusion`: `LogNormal(-1,1)`, `sigma`: `Exponential(1.0)`                 |
| **Logistic Growth** | `'logistic_f'` | `r`, `K`, `sigma_F`  | `r`: `LogNormal(0,1)`, `K`: `Normal(150,50)`, `sigma_F`: `Exponential(0.5)` |

### 8.7. `nested()` and `eigen()` Modules

#### `nested()` Module Reference

| Keyword / Parameter     | Example Usage                                                         | Data Type | Default            | Meaning & Assumptions                                                                                                                                                                                                                                 |
| :------------------------| :----------------------------------------------------------------------| :----------| :-------------------| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nested()`              | `nested(z_var; ...)`                                                  | Module    | N/A                | Defines a supervised sub-model whose latent effect is added to the main model's linear predictor. The `z_var` is a symbolic name for this component.                                                                                                  |
| `formula`               | `formula="likelihood(z, family=gaussian) ~ intercept() + spatial(s)"` | `String`  | `""`               | A complete `bstm` formula string that defines the structure of the sub-model, including its own likelihood. This sub-model is fit to the specified `data_source`.                                                                                     |
| `data_source`           | `data_source=proxy_data`                                              | `Symbol`  | `:data`            | A symbol pointing to a `DataFrame` passed as a keyword argument to the main `bstm()` call. This allows the sub-model to use a different dataset.                                                                                                      |
| `rho_nested` (Implicit) | N/A                                                                   | `Float`   | `Normal(1.0, 0.5)` | A scaling coefficient that links the sub-model's latent effect to the main model's linear predictor: $\eta_{\text{main}} = \dots + \rho_{\text{nested}} \cdot \eta_{\text{sub}}$. The prior assumes the sub-model is a good proxy ($\rho \approx 1$). |

#### `eigen()` Module Reference

| Keyword / Parameter | Example Usage              | Data Type      | Default            | Meaning & Assumptions                                                                                                                                                               |
| :--------------------| :---------------------------| :---------------| :-------------------| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `eigen()`           | `eigen(y1, y2, y3; ...)`   | Module         | N/A                | Defines a Bayesian PCA factor model. The variables listed (e.g., `y1, y2, y3`) are the multivariate outcomes to be decomposed.                                                      |
| `n_factors`         | `n_factors=1`              | `Int`          | `1`                | The number of latent factors (principal components) to extract. This determines the dimensionality of the reduced latent space.                                                     |
| `pca_sd`            | `pca_sd=Exponential(0.5)`  | `Distribution` | `Exponential(1.0)` | The prior for the standard deviations of the principal components (latent factors). These are the "eigenvalues" of the system, controlling the variance explained by each factor.   |
| `pdef_sd`           | `pdef_sd=Exponential(0.5)` | `Distribution` | `Exponential(1.0)` | The prior for the standard deviation of the residual (uniqueness) noise. This captures the variance in each observed variable that is *not* explained by the shared latent factors. |

### 8.8. `fixed()` and `intercept()` Modules

#### `fixed()` Module Reference

| Keyword / Parameter | Example Usage        | Data Type                 | Default        | Meaning & Assumptions                                                                                    |
| :--------------------| :---------------------| :--------------------------| :---------------| :---------------------------------------------------------------------------------------------------------|
| `fixed()`           | `fixed(Region, ...)` | Module                    | N/A            | Explicitly marks a variable as a fixed effect. Primarily used to specify contrasts or priors.            |
| `contrast`          | `contrast=:effects`  | `Symbol`                  | `DummyCoding`  | Specifies the contrast coding for a categorical variable (e.g., `:effects`, `:helmert`).                 |
| `prior`             | `prior=Normal(0, 2)` | `Distribution` or `Tuple` | `Normal(0, 5)` | Sets the prior for the coefficient(s) of this fixed effect. Can be a `Distribution` or a PC prior tuple. |

#### `intercept()` Module Reference

| Keyword / Parameter | Example Usage          | Data Type                 | Default        | Meaning & Assumptions                                                                                                                |
| :--------------------| :-----------------------| :--------------------------| :---------------| :-------------------------------------------------------------------------------------------------------------------------------------|
| `intercept()`       | `intercept(prior=...)` | Module                    | N/A            | Explicitly includes a global intercept. Using `1` in the formula is equivalent. This module is mainly for specifying a custom prior. |
| `prior`             | `prior=Normal(0, 10)`  | `Distribution` or `Tuple` | `Normal(0, 5)` | Sets the prior for the global intercept term. Can be a `Distribution` or a PC prior tuple.                                           |


#### Interaction Effects

Interaction effects between fixed covariates are specified using the standard `*` and `&` operators from `StatsModels.jl`. The `bstm` framework also supports the `:` operator as a synonym for `&`. These operators can be used both as bare terms in the formula and within the `fixed()` module.

*   `cov1 * cov2`: Expands to `cov1 + cov2 + cov1 & cov2` (main effects and interaction).
*   `cov1 & cov2`: Includes only the interaction term.
*   `cov1 : cov2`: Equivalent to `cov1 & cov2`.

**Example:**
```julia
# These formulas are equivalent and include main effects and the interaction.
m1 = @bstm(likelihood(y) ~ intercept() + cov1 * cov2, data)
m2 = @bstm(likelihood(y) ~ intercept() + fixed(cov1 * cov2), data)
m3 = @bstm(likelihood(y) ~ intercept() + fixed(cov1) * fixed(cov2), data)
m4 = @bstm(likelihood(y) ~ intercept() + fixed(cov1) + fixed(cov2) + fixed(cov1 & cov2), data)

# These formulas include only the interaction term.
m5 = @bstm(likelihood(y) ~ intercept() + cov1 & cov2, data)
m6 = @bstm(likelihood(y) ~ intercept() + cov1 : cov2, data)
```

**Note on Priors:** Applying a custom prior to an interaction term (e.g., `fixed(cov1 * cov2, prior=...)`) is not directly supported, as the prior would be ambiguous across the expanded main and interaction effects. To assign a specific prior to an interaction, you must first manually create the interaction term as a new column in your `DataFrame` and then apply the `fixed()` module with a `prior` to that new column.


## 9. References

*   Besag, J., York, J., & Mollié, A. (1991). Bayesian image restoration, with applications in spatial statistics. *Annals of the Institute of Statistical Mathematics*, 43(1), 1-20.
*   Riebler, A., Sørbye, S. H., & Rue, H. (2016). An intuitive Bayesian spatial model with two hyperparameters. *Statistical Methods in Medical Research*, 25(2), 1145-1160.
*   Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time variation in disease risk. *Statistical Methods in Medical Research*, 9(3), 205-220.
*   Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
*   Rahimi, A., & Recht, B. (2008). Random features for large-scale kernel machines. *Advances in Neural Information Processing Systems*, 20.
*   Lindgren, F., Rue, H., & Lindström, J. (2011). An explicit link between Gaussian fields and Gaussian Markov random fields: The SPDE approach. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 73(4), 423-498.
