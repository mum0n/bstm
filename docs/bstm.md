---
title: "*bstm*: Bayesian SpatioTemporal Models in Julia/Turing"
header: "*bstm* in Julia"
keyword: |
	Keywords - Gaussian Process / CAR Spatiotemporal models
abstract: |
	Bayesian SpatioTemporal Models in Julia.

metadata-files:
  - _metadata.yml

format:
  html:
    code-fold: true

engine: julia

execute:
  eval: false
---

## Abstract
 
Bayesian Space-Time Models in Julia (*bstm*) is a Julia library that combines elements of Spatial partitioning methods (for discrete modelling) together with Bayesian spatiotemporal models as well a basic statistical modelling. At its core is a discrete perspective upon space and time, not for philosophical reasons, but rather operational functionality. Spatiotemporal models are resource intensive. This discrete perspective permits useful solutions within the constraints of most currently available computing resources. After developing these discrete approximations, we explore continuous Gaussian Process methods. Though the focus is upon Ecological applications, the framework is sufficiently general that it can be readily adapted to any spatiotemporal process, no matter how large or small. 

Simplistically, *bstm* can be seen as a domain specific language ("DSL") that brings in elements of statistical modelling languages found in other platforms to build Turing-based models and powered by a large number of Julia libraries. It is reasonably complete and importantly, extensible. Using Julia leverages the power and flexibility of the language (especially the Bayesian Turing.jl framework), with a compact, flexible and extensible set of functions and tools. 

Ultimately, here, we are developing a general framework to explore various models of increasing complexity to handle measurement error, periodic dynamics, and spatial dependencies and enable parameter estimation of custom dynamical/process models embedded in these structural domains. Random Fourier Features (RFF), Fast Fourier Transform (FFT), Fully Independent Training Conditional (FITC) and Deep and Inducing Point Gaussian Processes are explored to make Discrete and Continuous Spatiotemporal models computationally tractable for large datasets.

  
## Introduction: The SpatioTemporal Challenge 

Ecological monitoring is a pursuit of moving targets. To usefully model important variables like bottom temperature, species composition, and the population dynamics of species, one must, almost always, deal with incomplete or low density information from expensive surveys with limits to resources and time. The usual recourse is some variation of Random Stratified Sampling to "absorb" unaccounted errors or "externalities" as unstructured, **independent**, random effects. This can of course be fine in simple settings. In really dynamic environments, no matter how good you think the stratification may be, it will induce biased. In *bstm*, we do not ignore these "externalities" and asuming no bias, instead we embrace them as they are also, usually informative. Though *bstm* can be used for the former, it shines as a high-dimensional Bayesian hierarchical framework designed to decompose complex spatiotemporal data into interpretable latent components. This is because, ecological data is inherently **dependent** or structured. 

To address the "SpatioTemporal Challenge," *bstm* utilizes three primary components:

1. Spatial Clustering: Implemented via spatial autocorrelation specifications to account for geographical neighborhoods.
2. Temporal Autocorrelation: Utilizing temporal autocorrelation to capture evolving trends.
3. Non-linear Interactions: Modeling complex interactions where the relationship between space and time is non-stationary and dynamic.

Failing to distinguish between a permanent habitat feature (captured by a spatial component), strong directional change such as global temperature increases (captured by a timeseries component) and a transient environmental anomaly such as a strong el Nino/la Nina events or Gulf stream flows (captured by the space-time interaction), or variability change across space and/or time (volatility) can result in biased hindcasts and forecasts. By isolating these effects, we ensure that our understanding and consequent management decisions are based on the "true" (latent) underlying drivers rather than improperly accounted statistical noise. This decomposition is made possible by transforming a computationally prohibitive problem into a tractable one. 

By using Julia and Turing, you get the state of the art in computation and optimization with Automatic Differentiation built into most heavy lifting operations. Even if you do not use the spatiotemporal modelling components of *bstm*, you can simply use it a bridge to simple Bayesian modelling using an interface that should be comfortable for most people exposed to modern statistical platforms. Do we need yet another statistical front-end? Perhaps not if you are well versed in Turing and computation in general. But for many this will help bridge the technical demands until you no longer need it or use it to template a model and then adapt it further to your needs. Bstm is also a tool written for myself to simplify my computational work and so will have quirks. Apologies in advance.  

As this document has, in part, a didactic purpose, it is structured like a notebook with explanations inter-spread with examples. The architectural and technical underpinnings are touched upon and then a cook-book like set of recipes for different situations to fleshen out the approaches. I have made heavy use of Google's Gemini LLM to help with the formula parsing and double checking of computational logic; it has been a powerful coding aid and also a cause of much exasperation. This is my contribution to Julia and the scientific community. It is fully open and given without condition, except that I ask that if you add or improve upon it that you share your improvements to the community. 

### Computation: Getting started with the environment

First, let us get the Julia environment set up before anything else. Here we use [Julia](https://julialang.org/), as in my experience, it is a clear didactic tool and better for long-term learning and simultaneously use in large projects due to maintainability of the code-base and high performance. It is an open-source platform created by mathematicians, engineers, natural scientists, statisticians, computer scientists and machine learning specialists, each bringing the best from their respective fields and lessons learned from domain-specific software platforms in a coherent and performative fashion. At the time of this writing, there still remain some lingering issues (start up speed, recompilation of code and incompatibility creep when there are updates to any library (we depend upon many well-established libaries to do the heavy lifting in the background), but the speed that is offered and code clarity in exchange is worth it in any serious data manipulation efforts. Your mileage will vary, but the lessons learned are also easily transportable to R, python, matlab, octave, etc., if forced to use those platforms. They each have their own quirks and challenges, but until their eventual convergence into something (that will likely look a lot like Julia), it is still a great platform to learn, teach and operate/develop cutting edge work. Many learning tools exist. [Have a look here for a curated list](https://julialang.org/learning/). See the Appendices for more details.
 
Installing [Julia](https://julialang.org/) is best done with [juliaup](https://github.com/JuliaLang/juliaup). It can make maintenance simpler. Most functions used here that are not part of a standard library are collected together in [Julia](https://julialang.org/) functions at [src](../src/). They can be loaded with supporting standard libraries.

**WARNING**: if this is your first run, this can take on the order of hours to install libraries and dependencies, so let it run in the background. You might need to re-start the Julia session if there are complex/multiple library dependency issues (that require or support different versions). 


```julia
#----------------------------------------------------------------------------------------
# Project Setup
#
# This code sets up the project directory and loads necessary libraries and functions.
# This step is crucial for ensuring the environment is correctly configured.
#----------------------------------------------------------------------------------------
println("Setting up project environment...")
# Replace project_directory with the location of your files
if Sys.iswindows()
    project_directory = joinpath("C:\\", "home", "jae", "projects", "bstm")
elseif Sys.islinux()
    project_directory = joinpath("/home", Sys.username(), "projects", "bstm")
else
    project_directory = joinpath("C:\\", "Users", "choij", "projects", "bstm")
end

# Load libraries and project-specific functions
include(joinpath(project_directory, "startup.jl"))
load_project_functions(srcdir())
println("Project setup complete.")
```

If there continue to be issues with packages breaking, some more lower level package management and digging may be required or a restart of Julia.


### Example data: Scottish lip cancer data  

As a first step towards spatial modelling, we look at a minimal data series: the [Scottish Lip Cancer data](https://mc-stan.org/users/documentation/case-studies/icar_stan.html). It has been thoroughly studied on many platforms over the years. There are 56 areal units and a simulated temporal component. We do not have access to the map positional data, but we do have the adjacency information from which we can infer approximate spatial topology. In these discrete models, we really only need to know which areal units are neighbours (connected graph), encapsulated through the adjacency matrix $W$.  

```julia
#----------------------------------------------------------------------------------------
# Data Loading and Preparation
#
# Load the Scottish Lip Cancer dataset, which has been extended with a simulated
# temporal component for demonstration purposes.
#----------------------------------------------------------------------------------------
println("Loading and preparing data...")
data_scot, _ = scottish_lip_cancer_data_spacetime()
inp_df = data_scot.data
W = data_scot.au.W
println("Data loaded.")


#----------------------------------------------------------------------------------------
# Visualize the spatial structure of the data. Since we only have adjacency information,
# we infer a spatial layout. We also visualize the spatial intensity of the data points.
#----------------------------------------------------------------------------------------
println("Visualizing spatial data structure...")
plot_spatial_graph(au=data_scot.au, plot_title="Lip Cancer Inferred from Adjacency 'Locations'")
plot_kde_simple(inp_df[!, [:s_x, :s_y]], sd_extension_factor=0.25, title="Spatial Intensity (KDE)")
println("Visualization complete.")

```

In the dataset, we have counts (y) of cancer incidence and population size in each area (log_offset). We also simulate a 10-"year" temporal process, a random walk with magnitude 0.5 and a covariate effect (X: an area-specific continuous covariate that represents the proportion of the population employed in agriculture, fishing, or forestry). An overall random uniform observation error of magnitude 0.2 is added with a count then taken as the overall, rounded integer value.


#### Spatiotemporal model: the shape of things to come

Before getting into the nitty gritty of the spatiotemporal models, let us go through our contrived example to see what the overall workflow is like. First, we format the data into a DataFrame with the correct variable names and run a simple separable spatiotemporal model.  

The Standard Separable Spatiotemporal Model with a Poisson-distributed cancer counts `y`, using a log-offset for population exposure. The linear predictor includes:

- An intercept.
- A fixed effect for the covariate `X` (proportion employed in agriculture, etc.).
- A spatial random effect using the BYM2 model to capture structured and unstructured spatial variation.
- A temporal random effect using an AR1 process to capture serial correlation over years.
The spatial and temporal effects are "separable" as no interaction term is included.

Note the semicolon at the end of the call: m = @bstm(); make the contents not print to screen. You can of course remove it to see what it really contains: a simple compiled Turing model.   

```julia
m  = @bstm(
    likelihood(y, family=:poisson, log_offsets=:log_offset) ~
        intercept(prior=Normal(0, 10)) +
        fixed(X, prior=Normal(0, 5)) +
        spatial(s_idx, model=:bym2, W=W) +
        temporal(year, model=:ar1),
    inp_df,
    verbose = false # Suppress model code printing for cleaner execution
);

# For demonstration, we run a short chain with a simple sampler. 
chn_separable = sample( m , MH(), 100; progress=false)

# For a full analysis:
os = get_optimal_sampler( m )
chn = sample( m , os, 100, nchains=4)  # you will need to tweak the number of samples, as this depends upon model and data
res = model_results_comprehensive( m, chn; au=data_scot.au)
model_results_plots(res)
```

The results from a short run of this model can be examined to check for convergence (e.g., r-hat values close to 1) and to interpret the posterior distributions of the parameters. A full analysis requires longer MCMC chains and multiple chains to properly assess convergence and posterior uncertainty.


### Simplified interface

The recommended way to define models is with the `@bstm` macro, which allows for a clean, unquoted formula syntax directly within your Julia code. This approach is more robust and flexible than using strings. By default, all models include an intercept. To explicitly control this, use the `intercept()` module.

**Recommended Macro Usage:**

```julia
#----------------------------------------------------------------------------------------
# Spatiotemporal Interaction Model (Knorr-Held Type IV)
#
# This model extends the separable model by adding a spatiotemporal interaction term.
# The `spatial(...) ⊗ temporal(...)` syntax specifies a Kronecker product interaction,
# allowing the spatial field to evolve over time. This is a Type IV interaction,
# the most complex type, where both space and time are structured.
#----------------------------------------------------------------------------------------
m = @bstm(
    likelihood(y, family=:poisson, log_offsets=:log_offset) ~
        intercept() +
        fixed(X) +
        spatial(s_idx, model=:bym2) +
        temporal(year, model=:ar1) +
        (spatial(s_idx, model=:besag) ⊗ temporal(year, model=:ar1)),
    inp_df,
    W = W,  # note as W is part of spatial() at multiple places, it is shorter to pass as an overall W
    verbose = false
);
```

The notation is similar to other statistical modeling packages but with specific modules for spatiotemporal components. Here `s_idx` (spatial unit index) and `year` (time unit index) are column names in the input DataFrame.

### Special Formula Terms

In `bstm` formulas, specific function-like terms, or "modules," are used to define the model's structure. These modules tell the pre-processor how to construct latent fields, handle covariates, and set priors. NOTE: using """ ... """ makes construction of long text strings simpler.

**Example Formula Usage**
```
formula = """
  likelihood(y, family=:poisson, offsets=log_pop) ~ 
    intercept(prior=Normal(0, 10)) +
    fixed(z, prior=Normal(0, 5)) +
    fixed(Region, contrast=:effects, prior=Normal(0, 2)) +
    (poverty |> spatial(s_idx, model=icar)) +
    (spatial(s_idx, model=besag) ⊗ temporal(year, model=ar1)) +
    smooth(age, model=pspline, nbins=10)
"""

m = @bstm( formula, data, W=W );

```

### Covariate Discretization & Transformation Rules

The `bstm` framework supports several methods for preprocessing covariates and their interactions:
 
1.  **Continuous Transformations (No Binning):** 
    *   `"unit"`: Performs Min-Max scaling, mapping values to the `[0, 1]` interval.
    *   `"zscore"`: Performs Z-score standardization (subtract mean, divide by standard deviation).
    *   `"log"`: Performs a log transformation: `log(x + 1.0 - min(x))`.
    *   `0` or `nothing`: No transformation; treats as a raw continuous covariate.

2.  **Discretization / Binning:**
    *   `Int` (e.g., `9`): Discretizes into N quantiles. Useful for creating non-linear effects via random effect structures (RW2/AR1).
    *   `"regular:XXX"` (e.g., `"regular:10"`): Creates XXX equal-width intervals between the 0.025 and 0.975 quantiles of the data.
    *   `AbstractVector` (e.g., `[0.1, 0.5, 0.9]`): Uses the provided vector as custom bin edges.

3.  **Interactions:**
    *   Interactions are specified as `"var1*var2"`. They are calculated *after* the individual variables have been transformed (scaled/logged), ensuring interactions operate on normalized representations.


## The `bstm` Framework: An Architectural Overview

The `bstm` framework provides a composable, formula-based interface for Bayesian spatiotemporal modeling in Julia. It is designed to address the challenge of building complex models by separating the observation likelihood from the specification of the latent process. This decoupling allows for flexible construction of models that can include spatial, temporal, and mechanistic components in an additive and extensible manner.

### The Formula Interface

The framework uses a formula interface inspired by R's `lme4` and `brms`, but with specific modules for spatiotemporal components. The model is defined with the observation model on the left-hand side (LHS) and the latent process model on the right-hand side (RHS). The `@bstm` macro enables an unquoted formula syntax.

#### Basic Structure

The general structure of a `bstm` model call is:

```julia
m = @bstm(
    likelihood(outcome_var, family=poisson, ...) ~ intercept() + fixed_effects + modules(...),
    data_frame,
    keyword_arguments...
)
```

Any keyword arguments provided after the `data_frame` are passed into the model's configuration. A notable general keyword is `verbose`.

*   **`verbose=false`**: Suppresses the printing of the dynamically generated model code and the results of the automatic prior predictive check that runs at instantiation. This is useful for cleaner output in scripts or notebooks. The default is `true`.

#### The `likelihood()` Module

The `likelihood()` module on the LHS specifies the observation model and its parameters.

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

#### Illustrative Examples

1.  **BYM2 Disease Mapping:**
    `@bstm(likelihood(y, family=poisson) ~ intercept() + spatial(s_idx, model=bym2), data, W=W)`
    Decomposes risk into structured spatial and unstructured IID noise.

2.  **AR1 Temporal Forecasting:**
    `@bstm(likelihood(y) ~ intercept() + temporal(t_idx, model=ar1), data)`
    Captures geometric temporal decay.

3.  **Spatio-Temporal Interaction:**
    `@bstm(likelihood(y) ~ intercept() + spatial(s_idx, model=besag) ⊗ temporal(t_idx, model=ar1), data, W=W)`
    Employs the Kronecker product to create a fully structured spatiotemporal interaction field.

4.  **Spatially Varying Coefficients (SVC) and Curves:**
    `@bstm(likelihood(y) ~ intercept() + (poverty |> spatial(s_idx, model=icar)), data, W=W)`
    Allows the impact of `poverty` to vary according to local spatial gradients.
    `@bstm(likelihood(y) ~ intercept() + (spatial(s_idx, model=icar) |> smooth(time, model=pspline)), data, W=W)`
    Models a temporal trend that varies smoothly across space.

### The Algebra of Manifolds: Composition and State-Space Models

The `bstm` domain-specific language operates through a recursive parser that allows for the algebraic composition of different model components, referred to as manifolds.

#### Algebraic Operators

1.  **Kronecker Product (⊗)**: Used for creating inseparable interaction effects, such as the Knorr-Held Type IV model (`spatial() ⊗ temporal()`). This builds a joint precision matrix $Q_{st} = Q_t \otimes Q_s$, enabling the representation of space-time interactions where every spatial location has a unique, correlated temporal trend.
2.  **Composition (∘)**: Represents the functional composition of two manifolds, where one manifold modulates the parameters of another. This is a powerful tool for creating non-stationary models.
3.  **Pipe (`|>`):** The pipe operator handles data normalization and state-space evolution.

*   **Transformations**: Objects like `ZScoreManifold` or `LogManifold` act as wrappers that normalize inputs before they enter the latent process.
*   **State-Space Evolution**: The pipe operator defines a state-space model where one manifold evolves over the domain of another. This supports both discrete-time dynamics (e.g., `spatial() |> temporal(model=ar1)`) and the creation of spatially-varying curves (e.g., `spatial() |> smooth(time, model=pspline)`), where the coefficients of the temporal basis functions are modeled as spatial fields.

### Core Components: Manifolds and Priors

The `bstm` framework includes a registry of components that range from discrete graph-based models to continuous spectral approximations.

#### The Discrete Registry: Gaussian Markov Random Fields (GMRF)

For discrete domains, `bstm` implements GMRF structures where dependency is defined by a precision matrix Q.

| Manifold  | Theoretical Assumption                      | Structural Rationale                                                |
| :----------| :--------------------------------------------| :--------------------------------------------------------------------|
| IID       | $\epsilon \sim N(0, \sigma^2 I)$            | Unstructured exchangeability; base model for PC-shrinkage.          |
| ICAR      | Intrinsic CAR; $Q_{ij} = -1$ for neighbors. | Pure local smoothing; identifies spatial gradients.                 |
| Besag     | Standard CAR model.                         | Global and local spatial dependency via fixed precision.            |
| BYM2      | Scaled Besag + IID component.               | Explicit variance partitioning ($\rho$) for better identifiability. |
| Leroux    | Convex combination of I and $Q_{ICAR}$.     | Bridges IID and ICAR structures through a mixing parameter.         |
| SAR       | $(I - \rho W)y = \epsilon$.                 | Simultaneous modeling of response autocorrelation.                  |
| RW1 / RW2 | Random Walk (1st/2nd order).                | Temporal continuity and smoothing of non-stationary trends.         |
| AR1       | $\mu_t = \rho \mu_{t-1} + \epsilon_t$.      | Stationary temporal process with geometric decay.                   |

#### Continuous, Spectral, and Advanced Manifolds

To address the $O(N^3)$ computational cost of kernel-based Gaussian Processes, the framework utilizes spectral projections and sparse approximations:

*   **Random Fourier Features (RFF)**: Maps input coordinates `x` into a randomized feature space to approximate the kernel. The projection is defined as $z(x) = \sqrt{2/M} \cos(Wx + b)$, where `W` is the spectral frequency matrix sampled from the kernel’s Fourier transform.
*   **SPDE**: Represents the field as a solution to $(\kappa^2 - \Delta)^{\alpha/2} u = \mathcal{W}$, mapping continuous Matérn processes onto a discrete mesh.
*   **Nystrom / FITC**: Low-rank approximations using `n_inducing` points to represent the global field.
*   **NetworkFlow**: Captures directed dependencies across an adjacency matrix with `:upstream` or `:downstream` dispatch.

#### Priors and Identifiability

Stability in high-dimensional models is achieved through a principled approach to prior specification. The `bstm` framework provides three built-in prior schemes and allows for user-defined overrides.

##### Prior Schemes

1.  **Penalized Complexity Priors (`:pcpriors`)**: This is the default scheme. PC priors are designed to shrink complex models towards simpler "base models" unless there is strong evidence in the data to the contrary. For example, the prior on a variance parameter (`sigma`) shrinks towards zero, and the prior on a correlation parameter (`rho`) shrinks towards zero (no correlation). This is the recommended scheme for most applications as it helps prevent overfitting and improves model identifiability.

##### Specifying PC Priors with Quantile Constraints

The core idea of PC priors is to translate a user's belief about the scale of a parameter into a prior distribution. This is done by specifying an upper bound `U` for a parameter and the probability `alpha` that the parameter will exceed this bound. The framework then calculates the necessary hyperparameters for the prior distribution (e.g., the rate `λ` for an `Exponential` prior) that satisfy this constraint.

The general form of the constraint is: `P(param > U) = alpha`

For a standard deviation parameter `sigma`, which is given an `Exponential(λ)` prior, the relationship is:
`P(sigma > U) = exp(-λ * U) = alpha`
From this, the framework solves for the rate parameter:
`λ = -log(alpha) / U`

This allows for a more intuitive and principled way to set priors than choosing arbitrary hyperparameter values.
 
2.  **Informative Priors (`:informative`)**: This scheme uses priors that are still weakly informative but less aggressive in their shrinkage than PC priors. For example, the prior on `rho` is a `Beta(2, 2)`, which is centered at 0.5, reflecting a belief that some correlation is more likely than none. This can be useful when you have prior knowledge that an effect is likely present.

3.  **Uninformative Priors (`:uninformative`)**: This scheme uses very wide, flat priors (e.g., `Normal(0, 1e6)` for `sigma`, `Uniform(0, 1)` for `rho`). While sometimes used to express ignorance, these priors are generally **not recommended** for complex hierarchical models, as they can lead to poor convergence and unidentifiable parameters.

##### Prior Comparison Table

| Parameter            | PC Prior (Default)                                                                                                             | Informative Prior    | Uninformative Prior        | Rationale                                                                                                                               |
| :---------------------| :-------------------------------------------------------------------------------------------------------------------------------| :---------------------| :---------------------------| :----------------------------------------------------------------------------------------------------------------------------------------|
| **Sigma** ($\sigma$) | `Exponential(λ)` where `λ = -log(α)/U` from `P(σ > U) = α`. A typical default might be `(U=1, α=0.05)`.                        | `Exponential(0.5)`   | `Normal(0, 1e6)`           | Controls the marginal standard deviation of a latent field. PC prior shrinks towards zero variance unless data supports a larger scale. |
| **Rho** ($\rho$)     | Transformed `Exponential(λ)` where `λ = log(α)/log(1-U)` from `P(ρ > U) = α`. A typical default might be `(U=0.5, α=0.05)`.    | `Beta(2, 2)`         | `Uniform(0, 1)`            | Controls spatial/temporal correlation. PC prior shrinks towards 0 (no correlation).                                                     |
| **Lengthscale**      | Transformed `Exponential(λ)` where `λ = -U*log(α)` from `P(lengthscale < U) = α`. A typical default might be `(U=10, α=0.05)`. | `InverseGamma(5, 5)` | `InverseGamma(0.01, 0.01)` | Controls the range of correlation in continuous GP models. PC prior prevents overfitting by shrinking towards large lengthscales.       |
| **Kappa** ($\kappa$) | `Exponential(λ)` derived from a quantile constraint.                                                                           | `Exponential(0.1)`   | `Exponential(10.0)`        | Controls the smoothness of an SPDE/Matérn field. PC prior shrinks towards a smoother field.                                             |
| **Amplitude**        | `Normal(0, 1)`                                                                                                                 | `Normal(0, 0.5)`     | `Normal(0, 100)`           | Controls the amplitude of harmonic (seasonal) components.                                                                               |
| **Phase**            | `Beta(1, 1)`                                                                                                                   | `Beta(2, 2)`         | `Uniform(0, 1)`            | Controls the phase shift of harmonic components.                                                                                        |

##### Setting Priors in a Model

You can control prior specification at three levels of precedence:

1.  **Local Override (Highest Precedence)**: Specify a prior directly within a module call. This will always override any global settings.
    This can be done by passing a pre-defined `Distribution` object or by passing a `Tuple` representing a PC prior quantile constraint.

    ```julia
    # Local Override with a pre-defined Distribution. Note the use of `sigma=...`
    m = @bstm(
        likelihood(y) ~ intercept() + spatial(s_idx, model=bym2, sigma=Exponential(0.1)),
        data, W=W
    );

    # Local Override with a PC prior quantile constraint
    # This sets P(sigma > 0.5) = 0.01 for this specific spatial component's sigma.
    m = @bstm(
        likelihood(y) ~ intercept() + spatial(s_idx, model=bym2, sigma=(0.5, 0.01)),
        data, W=W
    );

    # Local Override for a correlation parameter 'rho' in an AR1 model.
    # This sets P(rho > 0.8) = 0.05, shrinking it towards zero (no correlation).
    m = @bstm(
        likelihood(y) ~ intercept() + temporal(t_idx, model=ar1, rho=(0.8, 0.05)),
        data
    );

    # Local Override for a 'lengthscale' in a GP model.
    # This sets P(lengthscale < 10.0) = 0.05, shrinking it towards larger values.
    m = @bstm(
        likelihood(y) ~ intercept() + smooth(x, model=gp, lengthscale=(10.0, 0.05, :lower)),
        data
    );
    ```

### Architectural Paradigms

#### Univariate Architecture

The default kernel for single-outcome processes, as described in the preceding sections.

#### Multivariate Architecture

The `MultivariateArchitecture` is triggered when multiple outcomes are specified on the LHS of the formula (e.g., `y1 + y2 ~ ...`). It is designed to jointly model these outcomes, allowing the model to "borrow strength" across related processes and estimate the correlation structure between them.

##### Key Mechanisms:

*   **Outcome-Specific Parameters**: Each manifold (e.g., `spatial`, `temporal`) generates a separate latent field for each outcome. This means hyperparameters like `sigma_spatial` or `rho_temporal` are estimated independently for each response variable, providing maximum flexibility.
*   **LKJ Correlation Prior**: The core of the multivariate coupling is the `LKJCholesky` prior on a correlation matrix `L_corr` (**Lewandowski et al., 2009**). The final latent effects are constructed by multiplying the matrix of independent latent fields by the Cholesky factor of this correlation matrix: `eta_final = eta_latent * L_corr.L`. This induces a shared correlation structure across all outcomes for a given manifold, ensuring that the model captures shared patterns while allowing for outcome-specific variances.
*   **Householder Reflection (Spectral Orientation)**: For advanced use cases, the framework can apply a Householder reflection (`H = I - 2vv'`) to the latent fields. This allows the outcomes to rotate in the latent space, which is useful for aligning signals in models with complex dependencies like transport or advection-diffusion, where the direction of correlation is as important as its magnitude.

#### Multifidelity Architecture

The `MultifidelityArchitecture` is designed for data fusion, integrating high-volume, low-cost proxy data with sparse, high-quality observations. It is typically invoked using the `nested()` module.

##### Key Mechanisms:

*   **Hierarchical Latent Fields**: The architecture establishes a hierarchy of latent processes. A common setup involves:
    1.  **High-Fidelity (Target)**: The primary outcome variable (`y_hq`).
    2.  **Low-Fidelity (Proxy)**: A secondary, related variable (`y_lq`) with more abundant data.
*   **Nested Supervision**: The `nested()` module defines a complete sub-model for the low-fidelity data. The latent field from this sub-model (`eta_sub`) is then used as a calibrated predictor in the main model for the high-fidelity data.
*   **Calibration Parameters**: The link between the fidelities is modeled with calibration parameters, typically a bias and a scaling factor (rho), which are estimated within the model:
    `eta_main = ... + rho_nested * eta_sub`
    The prior on `rho_nested` is often centered around 1.0, assuming the proxy is a reasonably good, if biased, predictor of the main process. This allows the main model to learn from the structural patterns in the low-fidelity data while correcting for systematic bias and scale differences.

### Advanced Topics

#### Spatial Partitioning

For discrete spatial models (GMRFs), the continuous spatial domain must be discretized into "Areal Units" (AUs). The `assign_spatial_units` function provides several methods for this, balancing geometric compactness with statistical information density.

| Method | Description | Justification |
| :----- | :---------- | :------------ |
| `:cvt` | **Centroidal Voronoi Tessellation** | Iteratively minimizes variance to create geometrically regular cells. |
| `:kvt` | **K-Means Voronoi Tessellation** | Uses K-Means to create units with a balanced number of observations. |
| `:avt` | **Agglomerative Voronoi** | A bottom-up approach that merges small units to prevent data starvation. |
| `:bvt` | **Binary Vector Tree** | Employs recursive partitioning along the axis of maximum variance to efficiently handle large datasets and balance point counts. |
| `:qvt` | **Quadrant Voronoi Tessellation** | A quadtree-like method that recursively splits regions into four quadrants, adapting to multi-scale spatial clusters. |
| `:hvt` | **Hierarchical Voronoi** | Combines K-Means seeding with geometric refinement for stable, well-behaved polygons. |
| `:lattice` | **Regular Grid** | Simple, fast discretization into uniform squares. Assumes stationarity. |

When a `geom_hull` is provided, the function performs a spatial intersection ($P_{clipped} = P_{tessellated} \cap H_{hull}$) to ensure that generated units do not extend into invalid areas (e.g., water bodies). Connectivity between units is determined by `LibGEOS.touches`, and the resulting graph is used to construct the Graph Laplacian $Q = D - W$ for GMRF models.

##### Partitioning Control Parameters

The behavior of the `assign_spatial_units` function can be fine-tuned with the following parameters:

*   **`target_units`**: The desired number of areal units.
*   **`target_cv`**: The target coefficient of variation for the number of data points per areal unit, used to balance unit sizes.
*   **`min_total_arealunits`, `max_total_arealunits`**: Hard constraints on the total number of areal units created.
*   **`min_points`, `max_points`**: Ensures each spatial unit contains a number of data points within this range.
*   **`min_time_slices`**: Ensures each areal unit has a minimum number of unique time observations.
*   **`min_area`, `max_area`**: Constraints on the geographic area of each areal unit.
*   **`tolerance`**: Defines the convergence criteria for iterative methods like `:cvt` and `:hvt`.
*   **`buffer_dist`**: Used in methods like `:hvt` to define a buffer zone for identifying neighbors.

##### Partitioning Algorithms

*   **Centroidal Voronoi Tessellation (`:cvt`)**: Uses Lloyd's algorithm to create a regular, "honeycomb" mesh where each unit's centroid is the geometric center of its Voronoi cell. It is ideal for achieving uniform spatial coverage.

*   **K-Means Voronoi Tessellation (`:kvt`)**: A data-driven approach where centroids are the arithmetic mean of the observations within each unit. This creates smaller units in high-density areas, naturally preventing data starvation.

*   **Binary Vector Tree (`:bvt`)**: A high-speed hierarchical method that recursively splits the domain along the axis of maximum variance. It is the fastest approach for massive datasets and excels at creating units with balanced point counts.

*   **Quadrant Voronoi Tessellation (`:qvt`)**: A quadtree-like method that recursively splits regions into four quadrants. It is excellent at adapting its resolution to capture multi-scale spatial clusters.

*   **Agglomerative Voronoi Tessellation (`:avt`)**: A bottom-up approach that starts with an over-partitioned grid and iteratively merges the smallest or sparsest units. This is the most robust method for preventing "data-starved" units, which can cause instability in Bayesian samplers.

#### Mechanistic Models with `dynamics()`

The `dynamics()` module provides a powerful interface for embedding process-based, mechanistic models directly into the spatiotemporal framework. Unlike statistical models like `AR1` or `RW2` which describe correlation, `dynamics()` models describe the *evolution* of a latent field from one time step to the next based on a predefined equation.

This is accomplished by defining a latent spatiotemporal field, `dyn_field[space, time]`, where the state at time `t` is a function of the state at time `t-1`. For example, a simple advection model implements the state transition:

`dyn_field[:, t] ~ MvNormal(dyn_field[:, t-1] - velocity * L * dyn_field[:, t-1], noise)`

where `L` is the graph Laplacian. This allows the model to learn physical parameters like `velocity` within a fully Bayesian context, similar to the hierarchical dynamic models described by **Wikle (2003)**.

**Example: Logistic Growth Model**

A logistic growth model for a population can be specified as:

```julia
m = @bstm(
    likelihood(counts, family=poisson) ~ intercept() + dynamics(time, model=logistic_f, r_covariate=temp),
    data
);
```

This example fits a logistic growth model where the intrinsic growth rate `r` is itself a function of temperature.

#### Multi-fidelity and Nested Models

The `nested()` module is a "supervisor" component for multi-fidelity modeling. It allows you to define a complete sub-model that is fit to a separate (often larger, lower-quality) dataset. The latent effect from this sub-model is then incorporated as a calibrated predictor into the main model, allowing the main model to "learn" from the proxy data. The `nested()` module accepts a full formula string, including a `likelihood()` block, which enables the specification of independent likelihoods for each fidelity level.

```julia
m = @bstm(
    likelihood(y_hq) ~ intercept() + 
      spatial(s_idx) + 
        nested(proxy_submodel, formula="likelihood(y_lq, family=poisson) ~ intercept() + smooth(x)", data_source=low_quality_data),
    high_quality_data,
    low_quality_data = df_low_quality
);
```

#### Bayesian Factor Analysis with `eigen()`

The `eigen()` module implements a Bayesian Principal Component Analysis (PCA) to perform dimensionality reduction on a set of multivariate outcomes. It decomposes the input variables into a smaller set of orthogonal latent factors. The framework uses a Householder transformation to construct the orthonormal loadings matrix, ensuring numerical stability and efficient sampling.

#### Handling Censored Covariates via Joint Modeling

A censored covariate is a predictor variable for which the true value is not always known, but is instead confined to an interval (e.g., $x_{true} > c$). The statistically robust approach to this "errors-in-variables" problem is to treat the censored covariate as a latent variable and model it jointly with the primary outcome.

The `bstm` framework facilitates this through the `nested()` module, which allows for the construction of a joint model in a single step. This approach simultaneously estimates the model for the censored covariate and the main outcome model, correctly propagating all sources of uncertainty. The `nested()` module accepts a full formula string, including a `likelihood()` block, which enables the specification of independent likelihoods for each fidelity level.

##### Implementation with `nested()`

In this setup, the `nested()` module defines a complete sub-model for the censored covariate. This sub-model has its own `likelihood()` block where the censoring bounds (`y_L`, `y_U`) are specified. The latent process estimated by this sub-model is then automatically incorporated as a predictor in the main model's linear predictor.

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

### Inference and Post-Processing

#### Samplers, Initialization, and Optimization

The `bstm` framework leverages `Turing.jl`'s flexible sampling infrastructure. The choice of sampler is critical for efficient and accurate posterior exploration.

##### Sampler Selection with `get_optimal_sampler`

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
| :-------- | :------------- | :----------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| **NUTS**  | Gradient-Based | Adaptively tunes step size and number of steps.        | The state-of-the-art, general-purpose sampler for models with continuous, differentiable parameters. |
| **HMC**   | Gradient-Based | Requires manual tuning of leapfrog steps.              | A powerful alternative to `NUTS` that can be very efficient but may require expert tuning.           |
| **ESS**   | Gradient-Free  | Designed specifically for models with Gaussian priors. | Highly efficient for latent Gaussian models (e.g., CAR, GP models).                                  |
| **Slice** | Gradient-Free  | Adapts its step size to explore the posterior slice.   | A robust, general-purpose gradient-free sampler, useful when gradient-based methods fail.            |
| **MH**    | Gradient-Free  | Proposes moves from a simple proposal distribution.    | A universal sampler for non-differentiable models, but often inefficient in high dimensions.         |
| **PG**    | Particle-Based | Used for discrete parameters within a `Gibbs` sampler. | Automatically employed by `get_optimal_sampler` for any discrete random variables.                   |

##### Initial Values with `get_inits`

Good initial values are crucial for MCMC convergence. The `get_inits` function provides a robust mechanism for their generation:
1.  **Heuristic Initialization**: It draws a number of samples from the model's `Prior()` distribution.
2.  **Parameter Averaging**: It computes the median or mean of these prior samples to create a plausible starting point for each parameter. Heuristics are applied to ensure values are within valid bounds (e.g., `sigma > 0`).
3.  **MAP Refinement**: Optionally (`refine="map"`), it uses this heuristic starting point to run a fast optimization routine (`MAP()`) to find a mode of the posterior, providing a high-density starting location for the MCMC chains.

#### Optimization-Based Inference

For rapid point estimates, `bstm` models can be used with optimization instead of sampling:

*   **Maximum Likelihood (MLE):** `optimize(m, MLE())`
*   **Maximum A-Posteriori (MAP):** `optimize(m, MAP())`
*   **Variational Inference (VI):** `vi(m, ADVI(10, 1000))`

### Interpreting Results

The `model_results_comprehensive` function is the primary tool for post-processing. It takes a fitted model and an MCMC chain and returns a `NamedTuple` containing:

*   Posterior summaries (mean, median, CI) of all latent fields (spatial, temporal, etc.).
*   Performance metrics (RMSE, R-squared, WAIC).
*   MCMC diagnostics (R-hat, ESS).
*   A collection of standard plots (e.g., posterior predictive checks, spatial maps, temporal trends).

The `model_results_plots` function can be used to display all generated plots.

### Prediction

The `predict()` function projects a fitted model onto a new data grid to generate out-of-sample predictions. It correctly handles the projection of all manifold types, including re-computing basis matrices for smooth terms on the new data.

```julia
preds = predict(model, chain, new_data_frame)
```

## Discrete Bayesian Spatiotemporal Models

Many *bstm*s treat space as discrete areal units. There are many reasons for this. Well-constructed spatial partitions balances geometric compactness with statistical information density to avoid "Data Starvation." However, more often, one inherits areal management units, often with no structural support/scientific rationale. Though one can simply push on using such area definitions, if the balance of information available to information extractable is poor, often due  due to improper sizes and shapes, one should consider alternative areal units which then can be reconsolidated post-analysis to estimate at the level of the original, unfortunate areal units (*AUs*).   

Another pivotal advantage is speed. By being able to adjust the number of units, one can balance computational resources against information gain, depending upon the system being studied.

Over and above these considerations, computations are still expensive, of the order of $O(N^3)$ as inversion of a spatial covariance matrix is involved. Additional assumptions/constraints are required to make such problems operationally tractabile and bring computations to the order of $O(N)$ or $O(N \log N)$. Some of the main such constraints include:

**Markov Property**: 
- A spatial unit is independent of all non-neighbors given its immediate neighbors ($\mathcal{N}(i)$). 
- GMRF methods take advantage of operating on Sparse Precision Matrices ($Q$) as it makes high-dimensional problems computationally solvable.

**Additivity**:	
- The predictor $\eta$ is a sum of separable parts: $\alpha + \text{Space} + \text{Time} + \text{Interaction} + \text{Covariates}$.	
- Allows independent study of geographic and temporal drivers while still permitting more complex space-time interactions (e.g., Type IV).

**Stationarity**:
- Processes assume constant mean/variance over a standardized [0, 1] interval.	
- Provides structural stability; ensures the "rules" of time-series (AR1) or kernels (RFF) are consistent.
- Non-stationarity is important is real systems and so we will work towards this being relaxed in later methods.  

**Rank-Deficiency and Identifiability**:
- Intrinsic priors (ICAR and RW2) measure differences between units, not absolute levels. This provides the mathematical basis for smoothing, though it requires constraints to achieve identifiability.
- The **Sum-to-Zero Constraint** is used to solve this. When we use intrinsic priors like the ICAR (spatial) or RW2 (temporal), the precision matrix is singular. Adding any constant $c$ to the latent field ($\mathbf{u} + c\mathbf{1}$) results in the same log-density. This Rank-Deficiency Problem means that computations cannot distinguish between a global intercept ($\alpha$) and the mean level of the spatial field.
- Using a Sum-to-Zero Constraint ($\sum u_i = 0$) "pins" the latent field to a mean of zero, so the global intercept is preserved as the true overall mean of the response. This also stabilizes computations by preventing MCMC chains from wandering.

**Spatiotemporal Interactions (Knorr-Held Classes I-IV)**:
- Four classes of space-time interactions (**Knorr-Held (2000)**) are implemented to allow the spatial effect to evolve dynamically over time or the temporal trend to vary across regions. 
- Given a spatial precision matrix $\mathbf{Q}_{sp}$ and a temporal precision matrix $\mathbf{Q}_{tm}$, the interaction effect $\delta_{at}$ is modeled as a Gaussian Markov Random Field (GMRF) with a precision matrix $\mathbf{Q}_{\delta}$ defined by the Kronecker product of the marginal precisions:

$$\mathbf{Q}_{\delta} = \mathbf{Q}_{time} \otimes \mathbf{Q}_{space}$$

- Interaction Class Definitions:
    *   **Class I (Unstructured):** $\mathbf{I} \otimes \mathbf{I}$ (IID noise in space and time).
    *   **Class II (Time-Structured):** $\mathbf{Q}_{tm} \otimes \mathbf{I}$ (Temporally correlated within each region, but independent across regions).
    *   **Class III (Space-Structured):** $\mathbf{I} \otimes \mathbf{Q}_{sp}$ (Spatially correlated within each time slice, but independent over time).
    *   **Class IV (Fully-Structured):** $\mathbf{Q}_{tm} \otimes \mathbf{Q}_{sp}$ (Correlated in both dimensions; a spatial pattern evolves according to a temporal process).

###  Partitioning the Map: Areal Units and Information Balance

For discrete *bstm*s, we must first discretize the spatial domain into "Areal Units" (AUs). While any partitioning will do, the choice can impact model performance. In *bstm*, several methods are available (see `bstm_overview.md` for details).

Using our basic spatiotemporal data, let us try to represent them in a discrete manner across space. This is a necessary step if we wish to use the more speedy discrete models. Depending upon the constraints chosen, the spatial partitioning will change. Here, there is some subjectivity in the choice of constraints. The primary one to pay attention is: do we have enough data to represent spatial and  temporal processes. 

Here is a simple comparison of the methods using random data.

```julia
#----------------------------------------------------------------------------------------
# Comparing Spatial Partitioning Methods
#
# NOTE: This example uses simulated data, as the Scottish Lip Cancer dataset is
# already pre-partitioned into 56 areal units. This code demonstrates how to
# partition continuous spatial data into discrete units for GMRF-based models.
#----------------------------------------------------------------------------------------
println("Demonstrating spatial partitioning methods (using simulated data)...")
s_N_sim = 100
t_N_sim = 15
sim_data = generate_sim_data(s_N_sim, t_N_sim; rndseed=42)
sim_tu = assign_time_units(sim_data.t_coord; time_method="regular", t_N=sim_data.t_N, u_N=sim_data.u_N)

partition_configs = [:cvt, :kvt, :qvt, :bvt, :avt, :hvt]
partition_results = []
partition_plots = []

for method in partition_configs
    println("  Testing partitioning method: $method")
    local au_sim
    try
        au_sim = assign_spatial_units(sim_data.s_coord_tuple;
            area_method=method,
            t_idx=sim_tu.t_idx,
            target_units=20,
            min_time_slices=5
        )
        met = calculate_metrics(au_sim)
        push!(partition_results, (method=method, units=length(au_sim.centroids), cv_dens=met.cv_density))
        push!(partition_plots, plot_spatial_graph(au_sim; plot_title="Method: $method"))
    catch e
        @error "Method $method failed: $e"
    end
end
display(DataFrame(partition_results))
display(plot(partition_plots..., layout=(3, 2), size=(600, 800)))
println("Partitioning demonstration complete.")

```


Conclusion: All methods seem similar and reasonable. Having fewer areal units can make modeling faster, but that may also mean too much homogenization of the pattern, losing the ability to discriminate "hot" and "cold" spots. Note that if the CV of point density approaches 1, it represents a Poisson-like spatial distribution. Higher than 1 is considered clustered and lower than 1 means homogenous. We want some structure/clusters but not so much that we have unreliable data and not so little that everything is the same. 

## Back to modelling

When a naive model is used, inversion of the full/dense spatial and temporal covariance matrices is required at a cost of $O(n^3)$ operations. This naive approach is essentially the same as "Kriging" solutions (here a squared exponential covariance function is used for space). However, in Kriging, Least-squares assumptions are used to speed up computations. This simple model uses a Gaussian form. And, as can be seen the model code is relatively short and straight-forward, closely mimicking the mathematical relationships. 

The effective number of samples per second ("ess_per_sec") which ultimately will constrain total computational time is useful as a benchmark. For a dense GP model, the effective sampling speed can be orders of magnitude lower than for a sparse GMRF model.


```julia
#----------------------------------------------------------------------------------------
# Dense Gaussian Process (Kriging-style) Model
#
# ATTENTION: This model uses a legacy/didactic function call (`example_kriging_simple`)
# and is not based on the standard @bstm formula interface. It is included to demonstrate
# a dense, separable spatiotemporal GP (similar to Kriging) and to highlight the
# computational cost compared to sparse GMRF models. The modern @bstm equivalent
# would be `smooth(s_x, s_y, model=gp) + temporal(year, model=gp)`.
#----------------------------------------------------------------------------------------
println("Demonstrating a legacy dense GP model...")

# The formula specifies a Gaussian Process over the spatial coordinates (s_x, s_y)
# and another GP over the temporal coordinate (year). This creates a dense,
# separable spatiotemporal model.
m_kriging = @bstm(
    likelihood(y, family=:poisson, log_offsets=:log_offset) ~
        intercept() +
        smooth(s_x, s_y, model=:gp, kernel="se") +
        temporal(year, model=:gp, kernel="se"),
    inp_df,
    verbose = false
)

println("Running a very short sample chain for the dense GP model (this is slow)...")
# This is very slow, so only a few samples are taken for demonstration.
chn_kriging = sample(m_kriging, NUTS(), 10; progress=false)
println("Dense GP sampling complete.")

```

The other purpose of the above example was to show that the workflow is simple, once the model form has been chosen. The model structure is important and where the time and resources should be spent: deliberating utility, rather than trying to debug, implement and run.

#### Optimization-based approaches

Though MCMC sampling is our gold-standard, we also have other options that can be worth considering. All of these methods are boosted by Automatic Differentiation, some require smooth differentiable likelihood surfaces, while others are robust and can be range bound.

- **Maximum likelihood (ML)** estimation can be much faster than MCMC as pure optimization of a point mass is considerably simpler as priors are ignored and there is no need to carry posterior samples.
- **Maximum a-posteriori (MAP)** estimation is the same as ML except that prior information is used as well and so a bit closer to MCMC in spirit, though the focus is still upon the point estimates. 
- **Variational Inference (VI)** is also an optimization method. However, it approaches the problem by approximating the posterior distribution $p(z|x)$ with a simpler, flexible distribution $q(z)$ and minimizing the difference between them.
 
All three methods are accessible with the same Turing/Julia model. The following shows how to run them.

```julia
#----------------------------------------------------------------------------------------
# Optimization-based Inference Approaches
#
# Demonstrates how to find point estimates for model parameters using Maximum
# Likelihood (MLE), Maximum a Posteriori (MAP), and Variational Inference (VI)
# as alternatives to full MCMC sampling. We use the standard separable model
# defined in previously.
#----------------------------------------------------------------------------------------
println("Demonstrating optimization-based inference...")

m_for_optim = m_separable

# Maximum a Posteriori (MAP)
println("  Running MAP optimization...")
res_map = maximum_a_posteriori(m_for_optim, LBFGS())
println("  MAP optimization complete. Log-posterior: ", res_map.optim_result.value)
display(res_map.params)

# Variational Inference (VI)
println("  Running Variational Inference (this can be slow)...")
# For demonstration, a simple VI setup. For real use, tuning is required.
q_vi = vi(m_for_optim, ADVI(10, 1000); optimizer=LBFGS(), show_progress=false)
println("  VI complete.")
# To get samples: chn_vi = rand(q_vi, 1000)
```


Any of these point estimates can be used as starting points for further MCMC runs--if you trust the point estimates to have converged to a correct solution, and not a pathological position. 

### Reconstruction of effects and predictions 
 
The `_reconstruct` function (multiple methods, one for each Architecture) is the core post-processing engine of the `bstm` package. It transforms raw MCMC chains into structured summaries of latent fields, effect sizes, and model predictions. Quite often, this can be more of a struggle than the modelling! But if you know what you want and how to process Turing's output, then there is no need to use these convenience extraction functions.

To ensure compatibility across all architectures (GMRF, Spectral, GP, etc.), the function returns a `NamedTuple` with a standardized set of keys for spatial effects, temporal effects, predictions, etc.

The linear predictor $\eta_{i,s}$ is reassembled for every observation $i$ and MCMC sample $s$:

$$\eta_{i,s} = \text{Offset}_i + \text{Spatial}_{a[i],s} + \text{Temporal}_{t[i],s} + \text{Interaction}_{a,t,s} + \sum_{k} \beta_{k, \text{level}[i],s}$$

# Model Compendium
 
These are quick examples to show the breadth of what is possible. 

## Example data

```julia
project_directory = joinpath( "C:\\", "home", "jae", "projects", "bstm")  
include( joinpath( project_directory, "startup.jl" ) ) 
load_project_functions( srcdir() )
data_scot, _ = scottish_lip_cancer_data_spacetime(); # additional noise and "fake" time slices added
data = data_scot[:data];  
W = data_scot[:au][:W]
```

## Basic Regression & Mixed Effects

Standard regression components.

### Fixed Effects Model (Linear Regression)

A simple linear regression model with an intercept and two continuous covariates.

```julia
m = @bstm( likelihood(y) ~ intercept() + fixed(cov1) + fixed(cov2), data );
```
 
### Categorical Fixed Effects with Contrasts

Models the effect of a categorical variable using custom contrast coding.

**Key Features:**
- **Module**: `fixed()`
- **Contrast Coding**: `effects` coding sets the sum of coefficients to zero.


```julia
m = @bstm( likelihood(y) ~ intercept() + fixed(region, contrast=effects, prior=Normal(0, 10)), data );
```

### Random Intercept Model

Models group-level variability in the intercept.

**Key Features:**
- **Module**: `mixed()`
- **Random Intercept**: `mixed( intercept() | group)`


```julia
m = @bstm( likelihood(y) ~ intercept(false) + fixed(cov1) + mixed( intercept() | region), data );
```

### Random Slope and Intercept Model

Models group-level variability for intercepts and the effects of covariates.

**Key Features:**
- **Module**: `mixed()`
- **Random Slope**: `mixed(covariate | group)`


```julia
m = @bstm(
    likelihood(y) ~ intercept(false) + cov1 + 
        mixed( intercept(true) + cov1 | region ), # Correlated random intercept and slope for cov1
    data
);
```


## Likelihood Features

These examples demonstrate how to modify the observation model using parameters within the `likelihood()` module.

### Censored Data Model

Models a continuous outcome where some observations are censored.


```julia
y_lower_bound, y_upper_bound = 1, 100
m = @bstm(
    likelihood(y_rate, family=gaussian, censor_lower=y_lower_bound, censor_upper=y_upper_bound) ~ intercept() + fixed(region),
    data
);
```

### Zero-Inflated Model

Models count data with an excess of zeros.


```julia
m = @bstm(
    likelihood(y, family=poisson, zero_inflated=true) ~ intercept() + cov1,
    data
);
```

### Hurdle model

The **Hurdle Model** (**Mullahy, 1986**) is designed for data with an excess of zeros by modeling the zero-generating process and the positive-count process separately.

A two-part model where the process for generating zeros is separate from the process for generating positive counts.


```julia
m = @bstm( likelihood(y, family=poisson, hurdle=1) ~ intercept() + cov1, data );
```



### Stochastic Volatility Model

Models observation noise that varies over space and time.


```julia
m = @bstm(
    likelihood(y_rate, family=gaussian, volatility=true) ~ intercept() + spatial(s_idx, model=bym2, W=W),
    data
);
```


## Temporal Models

### Temporal Trend Models

#### Smooth Temporal Trend (Random Walk)

Models a smooth, non-linear temporal trend using a second-order random walk.


```julia
m = @bstm( likelihood(y) ~ intercept() + temporal(year, model=rw2), data );
```

#### Autoregressive Model (AR1)

Models a stationary temporal process where the current value depends on the immediately preceding value.


```julia
m = @bstm( likelihood(y) ~ intercept() + temporal(year, model=ar1), data );
```
 
#### Harmonic Seasonality

Captures periodic effects using sine and cosine basis functions.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + temporal(year, month, model=(ar1, cyclic), period=12),
    data
);
```

#### Cyclic Random Walk

Models a smooth, periodic effect where the end of the cycle connects to the beginning.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + temporal(day, model=cyclic, period=7),
    data
);
```

## Covariate Smoothing (`smooth`)

### 1D P-Spline Smoother

Models the non-linear effect of a continuous covariate using penalized splines.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + smooth(cov1, model=pspline, nbins=20),
    data
);
```

### 2D Thin Plate Spline

Models the smooth, non-linear interaction of two continuous covariates (e.g., spatial coordinates).


```julia
m = @bstm(
    likelihood(y) ~ intercept() + smooth(s_x, s_y, model=tps, nbins=50),
    data
);
```


## Spatial Models

### Areal Data Models (GMRFs)

These models are for data aggregated over discrete spatial units (polygons).

#### BYM2 Disease Mapping Model

These examples demonstrate various models for capturing spatial autocorrelation.

The standard for areal disease mapping, decomposing spatial risk into structured and unstructured components.


```julia
m = @bstm( likelihood(y, family=poisson) ~ intercept() + spatial(s_idx, model=bym2, W=W), data );
```

#### ICAR / Besag Model

A model for strong spatial smoothing based on local neighbors.


```julia
m = @bstm( likelihood(y) ~ intercept() + spatial(s_idx, model=icar, W=W), data );
```

#### Leroux Model

Alternatives to BYM2 that offer different parameterizations of spatial correlation.


```julia
m = @bstm( likelihood(y) ~ intercept() + spatial(s_idx, model=leroux, W=W), data );
```

#### SAR Model

Models spatial "spill-over" effects where the value at one location directly influences its neighbors.


```julia
m = @bstm( likelihood(y) ~ intercept() + spatial(s_idx, model=sar, W=W, noise=1e-6), data);
```



### Continuous & Point-Reference Models

In models using **Random Fourier Features (RFF)**, **FITC Sparse GPs**, and **Deep GPs**, space-time dependencies are defined via a continuous covariance kernel $k(\mathbf{x}_i, \mathbf{x}_j)$ where $\mathbf{x} = [s_{lon}, s_{lat}, t]$.

To avoid $O(N^3)$ kernel inversions, we approximate the interaction using $M$ features mapped through a spectral density $p(\boldsymbol{\omega})$:
$$\phi(\mathbf{x}) = \sqrt{\frac{2}{M}} \cos(\mathbf{W}\mathbf{x} + \mathbf{b})$$
Where $\mathbf{W} \sim p(\boldsymbol{\omega})$. The interaction is then the linear product $\eta_{it} = \phi(\mathbf{x}_i)^T \boldsymbol{\beta}_{rff}$.

These models are for data where exact coordinates are available.

#### Gaussian Process (GP)

The gold-standard for continuous spatial modeling, but computationally expensive.


```julia
m = @bstm( likelihood(y) ~ intercept() + smooth(s_x, s_y, model=gp, kernel=matern32), data);
```

#### SPDE Model

Models a continuous spatial process using an approximation to a Stochastic Partial Differential Equation, linked to the Matérn kernel.


```julia
m = @bstm( likelihood(y) ~ intercept() + spatial(s_idx, model=spde, W=W), data );
```
 


## 6. Interaction & Hierarchical Models

### 6.1. Separable Spatiotemporal Model

A standard model where the spatial and temporal effects are assumed to be independent and additive.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + spatial(s_idx, model=bym2, W=W) + temporal(year, model=ar1),
    data
);
```


### 6.2. Spatiotemporal Interaction Model (Knorr-Held Type IV)

A fully structured interaction where a spatial field (e.g., ICAR) evolves over time according to a temporal process (e.g., AR1).

**Formula Equivalent (using `⊗`):**
```julia
m = @bstm(
    likelihood(y) ~ intercept() + spatial(s_idx, model=icar) + temporal(year, model=ar1) +
        spatial(s_idx, model=icar) ⊗ temporal(year, model=ar1),
    data, W=W
);
```

**Formula Equivalent (using `spacetime`):**
```julia
m = @bstm(
    likelihood(y) ~ intercept() + spatial(s_idx, model=icar) + temporal(year, model=ar1) +
        spacetime(s_idx, year, model=(icar, ar1)),
    data, W=W
);
```

### 6.3. Spatially Varying Coefficients (SVC)

Allows the effect of a covariate to vary smoothly across space.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + cov1 |> spatial(s_idx, model=icar, W=W),
    data
);
```

### 6.4. Spatially Varying Curves

Models a non-linear trend of a covariate that varies smoothly across space.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + (smooth(year, model=pspline) |> spatial(s_idx, model=icar, W=W)),
    data
);
```

### Hurdle Model (spatiotemporal):



```julia
m_hurdle = @bstm(
    likelihood(y, family=:poisson, log_offsets=:log_offset, hurdle=0) ~
        intercept() + fixed(X) +
        spatial(s_idx, model=:bym2) +
        temporal(year, model=:ar1),
    inp_df, W = W, verbose = false
)
println("Hurdle model defined.")
```


## 7. Advanced & Mechanistic Models

### 7.1. Mechanistic Dynamics (Advection-Diffusion)

A mechanistic model for a process that is transported (advection) and spreads (diffusion) over a graph.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + dynamics(s_idx, year, model=advection_diffusion, W=W),
    data
);
```

### 7.1. Bayesian PCA (`eigen`)

Performs dimensionality reduction on a set of covariates, using the dominant latent factor as a predictor.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + eigen(cov1, cov2, cov3, n_factors=1),
    data
);
```

### 7.3. Multi-fidelity Model (`nested`)

Integrates a low-fidelity (but data-rich) proxy variable to improve predictions for a high-fidelity (but data-sparse) target.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + 
        nested(
            proxy_model, 
            formula=likelihood(y_bin, family=binomial) ~ intercept() + smooth(cov3, model=pspline)
        ),
    data
);
```

## Multivariate Models

### Multivariate CAR Model

A multivariate CAR model for jointly modeling multiple correlated spatial processes.


```julia
m = @bstm(
    y + y_bin ~ intercept() + spatial(s_idx, model=besag, W=W) + temporal(year, model=ar1),
    data
);
```

### Joint Model with Different Likelihoods

Jointly models multiple outcomes where each has a different likelihood.


```julia
m = @bstm(
    likelihood(y, family=poisson) + likelihood(y_continuous, family=gaussian) ~ 
        intercept() + spatial(s_idx, model=bym2, W=W),
    data
);

 
# 3.1 Multivariate Model
# Jointly modeling Gaussian and Poisson outcomes with spatial correlation
# Uses continuous 2D Thin Plate Spline for space and RW2 for time
model_mv = @bstm(
    likelihood(y_rate, family=gaussian) + likelihood(y, family=poisson) ~
    intercept() + 
    smooth(s_x, s_y, model=tps, nbins=30) + 
    temporal(year, model=rw2),
    data
);

# 3.2 Multinomial (Compositional) Model
# Modeling counts across 3 categories using Dirichlet-Multinomial
# Note: Multi-column LHS targets the Dirichlet-Multinomial kernel
model_multi = @bstm(
    likelihood(y_cat1 + y_cat2 + y_cat3, family=dirichlet_multinomial) ~
    intercept() + 
    smooth(s_x, s_y, model=gp, kernel=matern32, n_inducing=15),
    data
);

```



## Multifidelity (Nested) Model

```julia
# High-fidelity Gaussian outcome aided by a low-fidelity proxy sub-model

model_mf = @bstm(
    likelihood(y_gauss, family=gaussian) ~
    intercept() + 
    temporal(year, model=ar1) + 
    nested(
        proxy_submodel,
        formula = "likelihood(proxy_val, family=gaussian) ~ intercept() + smooth(s_x, s_y, model=tps, nbins=20)"
    ),
    data
);
```


## Year and Seasonal Structure

```julia
# Combining a long-term trend (AR1) with a Harmonic seasonal component
model_season = @bstm(
    likelihood(y_gauss) ~
    intercept() + 
    temporal(year, model=ar1) + 
    temporal(month, model=harmonic, period=12),
    data
);
```

```julia
# Demonstration of SVAR usage in BSTM
# Rationale: Shows how to model point-level dynamics where temporal persistence varies by region.

# Prepare data with spatiotemporal index
data.st_idx = [(t-1)*30 + s for (s, t) in zip(data.s_idx, data.year)]

# Example call: Spatially Varying Autoregressive (SVAR) Model
# This model allows the temporal autoregressive parameter `rho` to vary across space.
# The spatial variation of `rho` is modeled by an `icar` manifold.
model_svar = @bstm(
   likelihood(y_gauss) ~
   intercept() +
   svar(spatial(s_idx, model=icar)),
   data, W=W
);
 
# Demonstration of Threshold Autoregressive (TAR) logic
# This model switches between two AR(1) regimes based on a covariate's value.

# Prepare data for TAR example
data.price_index = 5.0 .+ cumsum(randn(nrow(data)) .* 0.1)

# Example call: A TAR model where the temporal dynamics of y_gauss
# switch based on whether `price_index` is above or below a learned threshold.
model_tar = @bstm(
    likelihood(y_gauss) ~
    intercept() +
    temporal(year, model=tar, threshold_var=price_index),
    data
);

# Synthetic Point Pattern Data Generation
# Rationale: LGCP models aggregated counts. We generate a smooth latent intensity 
# on a grid and sample Poisson counts to simulate point-pattern data.


lgcp_data, grid_W, total_cells = generate_lgcp_synthetic_data_regular(10)
display(first(lgcp_data, 5))

# LGCP Model Call Demonstration
 
# Note: In the LGCP manifold, the counts in M.y_obs are modeled directly via @addlogprob!.
# We pass the 'counts' column as the response to the likelihood module, but the LGCP
# manifold effectively overrides the standard likelihood application for its domain.

model_lgcp = @bstm(
    likelihood(counts, family=poisson) ~ 
    intercept() + 
    spatial(point_process=lgcp, model=icar, sigma=Exponential(0.5)), 
    lgcp_data, 
    W = grid_W, 
    s_N = total_cells,
    grid_areas = ones(total_cells) # Unit areas for the intensity integral
);
 


irreg_df, irr_W, n_units, cell_areas = generate_irregular_lgcp_data(10)

# 2. Instantiate Model specifying the area column
# Note: we pass grid_areas to the lgcp parameters
model_irreg = @bstm(
    likelihood(counts, family=poisson) ~ 
    intercept() + 
    spatial(point_process=lgcp, model=icar, sigma=Exponential(1.0), grid_areas=cell_areas), 
    irreg_df, 
    W = irr_W, 
    s_N = n_units
);


# Demonstration of Kriging implementation

# Prepare spatial data for Kriging
coord_data = DataFrame(
    s_x = rand(100) .* 10.0,
    s_y = rand(100) .* 10.0,
    y_gauss = randn(100)
);

# Example @bstm call (conceptual structure)
m = @bstm(
    likelihood(y_gauss) ~ 
    intercept() + 
    smooth(s_x, s_y, model=kriging, lengthscale=InverseGamma(3, 3), sigma=Exponential(1.0)),
    coord_data
);
```

   

### Standard Separable Spatiotemporal Model
 
This model is perhaps the most basic spatiotemporal model. We have already seen this model in the introductory example with Scottish lip cancers. It decomposes spatiotemporal count data into three main additive latent components, without any spatio-temporal interactions: a fixed offset, a BYM2 spatial field, and an AR1 temporal trend.

**Utility**: 
- Smooth raw count data and identify 'hotspots' that are statistically significant.
- Partitions observation noise to reveal the underlying latent 'signal'.

**Computation**:
- Efficient due to GMRFs and sparse precision matrices.
- Penalized Complexity (PC) priors pull the model toward a simpler form unless the data strongly supports complex structures.

**Mathematical Formulation**:
- **Poisson Log-Link**: Observations $y_i$ follow a Poisson distribution. The log-linear predictor $\eta$ combines a known offset with latent spatial and temporal effects: $$\log(\mu_{it}) = \text{offset}_{it} + \text{Spatial Effect}_a + \text{Temporal Effect}_t$$
- **Spatial Effect (BYM2)**: Decomposes spatial variance into a structured (ICAR) component and an unstructured (IID) component.
- **Temporal Effect (AR1)**: Follows a First-Order Autoregressive process.


```julia
#----------------------------------------------------------------------------------------
# A Standard Separable Spatiotemporal Model
#
# This is a canonical bstm model. It models the Poisson-distributed cancer counts `y`
# using a log-offset for population exposure. The linear predictor includes:
# - An intercept.
# - A fixed effect for the covariate `X` (proportion employed in agriculture, etc.).
# - A spatial random effect using the BYM2 model to capture structured and unstructured spatial variation.
# - A temporal random effect using an AR1 process to capture serial correlation over years.
# The spatial and temporal effects are "separable" as no interaction term is included.
#----------------------------------------------------------------------------------------
println("Defining a standard separable spatiotemporal model...")
m = @bstm(
    likelihood(y, family=:poisson, log_offsets=:log_offset) ~
        intercept(prior=Normal(0, 10)) +
        fixed(X, prior=Normal(0, 5)) +
        spatial(s_idx, model=:bym2) +
        temporal(year, model=:ar1),
    inp_df,
    W = W,
    verbose = false # Suppress model code printing for cleaner execution
)

println("Running a short sample chain for the separable model...")
# For demonstration, we run a short chain with a simple sampler.
# For robust inference, NUTS() with more samples is recommended.
chn_separable = sample(m, MH(), 1000; progress=false)
println("Separable model sampling complete.")
# For a full analysis:
# os = get_optimal_sampler(m)
# chn = sample(m, os, 2000, nchains=4)
# res = model_results_comprehensive(m, chn; au=data_scot.au)
# model_results_plots(res)
```

### Standard In-Separable Spatiotemporal Model with Interaction

This model demonstrates a more realistic workflow. It adds covariates via a RW2 smoothing process and includes a full Space-Time Interaction field (Type IV), allowing for localized hotspots that aren't captured by the main spatial or temporal trends.

```julia
#----------------------------------------------------------------------------------------
# Spatiotemporal Interaction Model (Knorr-Held Type IV)
#
# This model extends the separable model by adding a spatiotemporal interaction term.
# The `spatial(...) ⊗ temporal(...)` syntax specifies a Kronecker product interaction,
# allowing the spatial field to evolve over time. This is a Type IV interaction,
# the most complex type, where both space and time are structured.
#----------------------------------------------------------------------------------------
m = @bstm(
    likelihood(y, family=:poisson, log_offsets=:log_offset) ~
        intercept() +
        fixed(X) +
        spatial(s_idx, model=:bym2) +
        temporal(year, model=:ar1) +
        (spatial(s_idx, model=:besag) ⊗ temporal(year, model=:ar1)),
    inp_df,
    W = W,
    verbose = false
);
```

The notation is similar to other statistical modeling packages but with specific modules for spatiotemporal components. Here `s_idx` (spatial unit index) and `year` (time unit index) are column names in the input DataFrame.


### Spatial (leroux) Temporal (ar1) Models, separable

This model replaces the BYM2 prior with a **Leroux CAR** prior for spatial effects. It is non-intrinsic, meaning it has a non-singular precision matrix obtained from a combination of the identity matrix $I$ and the scaled spatial Laplacian $Q_{sp}$:

$$Q_{Leroux} = \tau [ (1-\rho)I + \rho Q_{sp} ]$$

This specification is robust because it automatically handles both structured spatial clustering and unstructured heterogeneity within a single latent field.

The Leroux Model is a flexible model that mixes spatial and non-spatial variance.

```julia
m = @bstm(
    likelihood(y, family=:poisson, log_offsets=:log_offset) ~
        intercept() + fixed(X) +
        spatial(s_idx, model=:leroux) +  
        temporal(year, model=:ar1),
    inp_df, W = W, verbose = false
);
```


### Simultaneous Autoregressive (SAR) model

The **Simultaneous Autoregressive (SAR)** model specifies the joint dependency directly, where the spatial field is equal to a weighted average of its neighbors plus independent noise. Its precision matrix is $Q_{SAR} = \frac{1}{\sigma^2} (I - \rho W)'(I - \rho W)$. This approach was formalized in early spatial statistics literature (**Cliff & Ord, 1973**).

SAR Models spatial "spill-over" effects directly.


```julia
m = @bstm(
    likelihood(y, family=:poisson, log_offsets=:log_offset) ~
        intercept() + fixed(X) +
        spatial(s_idx, model=:sar) + # Changed from bym2
        temporal(year, model=:ar1),
    inp_df, W = W, verbose = false
);

``` 

### Spatially Varying Coefficient (SVC) Model 

The **Spatially Varying Coefficient (SVC)** (Gelfand et al., 2003) model relaxes the assumption of global stationarity in regression effects. Instead of a single $\beta$ for the whole domain, each area $i$ has its own coefficient $\beta_{i,k}$. The log-intensity for area $i$ at time $t$ becomes:
$$\log(\mu_{it}) = \text{offset}_{it} + \phi_i + \delta_t + \gamma_{it} + \sum_{k=1}^K x_{it,k} \beta_{i,k}$$

An SVC model allows the effect of a covariate to vary across space. Here, the effect of the covariate `X` is modulated by a spatial field (ICAR model).

```julia
m = @bstm(
    likelihood(y, family=:poisson, log_offsets=:log_offset) ~
        intercept() +
        (X |> spatial(s_idx, model=icar)) + # Effect of X varies spatially
        temporal(year, model=:ar1),
    inp_df, W = W, verbose = false
);
```
 

### Multivariate CAR

The **Multivariate CAR (MCAR)** model (**Gelfand & Vounatsou, 2003**) extends CAR models to the multivariate case, where we wish to model $J$ spatial processes (e.g., two different diseases) that are likely correlated. For a bivariate case ($J=2$), the joint spatial random effect $\mathbf{\Phi} = [\mathbf{\phi}_1, \mathbf{\phi}_2]'$ follows a multivariate normal distribution:
$$\mathbf{\Phi} \sim \text{MvNormal}(\mathbf{0}, [\Sigma \otimes Q_{ICAR}]^{-1})$$
Where $\Sigma$ is a $J \times J$ covariance matrix capturing the correlation between the outcomes.

This model jointly analyzes multiple correlated spatial processes. NOTE: This requires simulating a second response variable, as the Scottish Lip Cancer data is univariate. We create `y2` based on `y` for demonstration.

```julia
inp_df_multi = deepcopy(inp_df)
inp_df_multi.y2 = round.(Int, max.(0, inp_df.y .* rand(Normal(0.8, 0.2), nrow(inp_df))))

m = @bstm(
    y + y2 ~ # Jointly model y and y2
        intercept() +
        fixed(X) +
        spatial(s_idx, model=:besag) + # A shared spatial effect
        temporal(year, model=:ar1),    # A shared temporal effect
    inp_df_multi, W = W, verbose = false
);
```


### Models Requiring Attention

Several models in the original document (`localadaptive`, `dag`, `size_structured`) use custom or legacy APIs that do not map directly to the modern `@bstm` formula interface. These require further development to be integrated.
   
Model: localadaptive
ATTENTION: The `localadaptive` model is not a standard manifold in the bstm formula
API. It appears to be a custom model for detecting localized spatial clusters.
Implementing this would require creating a custom manifold processor or a direct
Turing model definition.

m = @bstm( ... spatial(s_idx, model=:localadaptive) ... ) # This will not work.

Model: dag
ATTENTION: A Directed Acyclic Graph (DAG) model for spatial processes is a specialized
model for non-reciprocal dependencies. It is not a standard manifold in the bstm
formula API. Implementation would require a custom processor.

m = @bstm( ... spatial(s_idx, model=:dag) ... ) # This will not work.

Model: size_structured
ATTENTION: This appears to be a highly specialized application-specific model from
the original document that uses a complex, non-standard API (`bstm_size_structured`).
Converting this to the formula interface would require a significant refactoring
of its internal logic into custom manifolds or a nested model structure.




## References

*   Besag, J. (1974). Spatial interaction and the statistical analysis of lattice systems. *Journal of the Royal Statistical Society: Series B (Methodological)*, 36(2), 192-225.
*   Besag, J., York, J., & Mollié, A. (1991). Bayesian image restoration, with applications in spatial statistics. *Annals of the Institute of Statistical Mathematics*, 43(1), 1-59.
*   Cliff, A. D., & Ord, J. K. (1973). *Spatial autocorrelation*. Pion.
*   Damianou, A., & Lawrence, N. (2013, April). Deep gaussian processes. In *Artificial intelligence and statistics* (pp. 207-215). PMLR.
*   Gelfand, A. E., Kim, H. J., Sirmans, C. F., & Banerjee, S. (2003). Spatial modeling with spatially varying coefficient processes. *Journal of the American Statistical Association*, 98(462), 387-396.
*   Gelfand, A. E., & Vounatsou, P. (2003). Proper multivariate conditional autoregressive models for spatial data analysis. *Biostatistics*, 4(1), 11-15.
*   Riebler, A., Sørbye, S. H., & Rue, H. (2016). An intuitive Bayesian spatial model with two hyperparameters. *Statistical Methods in Medical Research*, 25(2), 1145-1160.
*   Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time variation in disease risk. *Statistical Methods in Medical Research*, 9(3), 205-220.
*   Leroux, B. G., Lei, X., & Breslow, N. (2000). Estimation of disease rates in small areas: a new mixed model for spatial dependence. In *Statistical models in epidemiology, the environment, and clinical trials* (pp. 179-191). Springer, New York, NY.
*   Lewandowski, D., Kurowicka, D., & Joe, H. (2009). Generating random correlation matrices based on vines and extended onion method. *Journal of multivariate analysis*, 100(9), 1989-2001.
*   Mullahy, J. (1986). Specification and testing of some modified count data models. *Journal of econometrics*, 33(3), 341-365.
*   Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
*   Rahimi, A., & Recht, B. (2008). Random features for large-scale kernel machines. *Advances in Neural Information Processing Systems*, 20.
*   Lindgren, F., Rue, H., & Lindström, J. (2011). An explicit link between Gaussian fields and Gaussian Markov random fields: The SPDE approach. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 73(4), 423-498.
*   Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sørbye, S. H. (2017). Penalising model component complexity: A principled, practical approach to constructing priors. *Statistical Science*, 32(1), 1-28.
*   Snelson, E., & Ghahramani, Z. (2006). Sparse Gaussian processes using pseudo-inputs. *Advances in neural information processing systems*, 18.
*   Wikle, C. K. (2003). Hierarchical Bayesian models for predicting the spread of ecological processes. *Ecology*, 84(6), 1382-1394.
*   Williams, C. K., & Seeger, M. (2001). Using the Nyström method to speed up kernel machines. In *Advances in neural information processing systems*, 13.

---

## Appendix 1: Technical API Reference

### 1. Introduction

This document provides a detailed technical reference for the internal components of the `bstm` framework. It is intended for developers and advanced users who wish to understand, extend, or debug the framework's core machinery. It covers the manifold system, formula parsing engine, model configuration pipeline, Turing model definitions, and the posterior reconstruction engine.

### 2. Core Data Structures & Manifold System

The `bstm` framework is built upon a system of `Manifold` types that represent different structural assumptions about latent fields.

#### 2.1. Abstract Manifold Types

The manifold system is organized under a hierarchy of abstract types that define the role of each component.

*   **`abstract type Manifold end`**: The root type for all structural components.
*   **`abstract type ManifoldModel <: Manifold end`**: Represents a base-level statistical model for a latent field (e.g., `ICAR`, `AR1`, `GP`). These are the fundamental building blocks.
*   **`abstract type ManifoldOperator <: Manifold end`**: Represents an operation that combines or transforms one or more `Manifold` objects (e.g., `ComposedManifold` for `⊗`, `SVCManifold` for `|>`).
*   **`abstract type ManifoldSupervisor <: Manifold end`**: A special type for modules that manage entire sub-models, such as the `nested()` module.

#### 2.2. `ManifoldModel` Structs

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
struct bstm_Likelihood{F<:AbstractBSTM_Family, Z<:AbstractZIState, C<:AbstractCensoringState, W, P, R, S, T, TR, TL, TU, HT, EX} <: ContinuousMultivariateDistribution
    family::F           # e.g., PoissonFamily()
    y_obs::TR           # Observed data
    zi_state::Z         # e.g., ZeroInflated()
    censoring_state::C  # e.g., Uncensored()
    weight::W
    phi_zi::P
    phi_hurdle::P
    r_nb::R
    sigma_y::S
    trial::T
    censor_lower::TL
    censor_upper::TU
    hurdle::HT
    extra_params::EX
end
```

The `logpdf` function for this struct calls `bstm_kernel`, which dispatches on the `family`, `censoring_state`, and `zi_state` types to compute the correct log-probability for each observation.

## 3. Formula Parsing Engine (`formula_parsing.jl`)

The formula parser translates the user-provided formula string into a structured representation that the configuration engine can process.

*   **`decompose_bstm_formula(formula_str)`**: This is the main entry point. It splits the formula into its Left-Hand Side (LHS) and Right-Hand Side (RHS).
    *   **LHS**: Parsed to identify outcome variables and their likelihood specifications (e.g., `likelihood(y, family=poisson)`).
    *   **RHS**: Pre-processed to handle intercept control (`-1`, `0`, `intercept(false)`) and to normalize all bare terms (e.g., `z`) into explicit `fixed(z)` module calls. It is then parsed by `_parse_rhs_expression` into an Abstract Syntax Tree (AST).
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

*   **`resolve_hyperpriors(...)`**: Implements the three-level precedence for prior specification:
    1.  **Local**: A prior specified directly in a module call (e.g., `sigma=...`).
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
    *   **`extract_manifold(m_obj, ...)`**: This function dispatches on the `Manifold` type (`m_obj`). Each method knows how to find its parameters in the chain and reconstruct its specific effect.
        *   For simple manifolds like `BYM2`, it finds `sigma`, `rho`, `latent_struct`, and `latent_iid` samples and combines them to produce the structured, unstructured, and total spatial fields.
        *   For a `ComposedManifold` with a `kronecker_product` operator, it reconstructs the interaction field by finding the corresponding latent field and hyperparameters in the chain and applying the correct scaling and reshaping.

*   **`_modular_eta_assembly(...)`**: Takes the `registry` of discovered fields and reassembles the full linear predictor `eta` for each posterior sample. This process mirrors the assembly logic within the Turing model itself but operates on the posterior samples. It correctly handles both in-sample (`M`) and out-of-sample (`PS`) data.

*   **`_process_ll_and_predictions(...)`**: Takes the assembled `eta` samples and generates:
    *   **`p_denoised`**: The expected value of the response, obtained by applying the inverse link function to `eta`.
    *   **`p_noisy`**: Predictions that include observation noise, obtained by sampling from the predictive distribution.
    *   **`log_lik`**: The pointwise log-likelihood matrix, used for computing WAIC and LOO.

## 7. Key Utility Functions

*   **`assign_spatial_units(...)`**: Discretizes continuous spatial coordinates into areal units using various methods (`:cvt`, `:kvt`, `:bvt`, `:avt`, `:qvt`, `:hvt`, `:lattice`). It returns an object containing the adjacency matrix `W`, assignments, and centroids.

*   **`create_fixed_design(...)`**: A wrapper around `StatsModels.jl` that takes a formula string and a `DataFrame` and returns the corresponding design matrix `Xfixed`, correctly handling contrasts and factor variables.

*   **`get_optimal_sampler(...)`**: A utility that inspects the model's parameters and their prior distributions to construct an efficient composite `Gibbs` sampler. It assigns specialized samplers (`ESS`, `Slice`, `PG`) to blocks of parameters based on their prior distributions.
    *   **Purpose**: To automatically construct an efficient composite Gibbs sampler tailored to the `bstm` model structure.
    *   **Rationale**: Different MCMC algorithms exhibit varying performance depending on the characteristics of the target distribution. A composite `Gibbs` sampler, which applies different samplers to different blocks of parameters, can significantly outperform a single, general-purpose sampler. This function automates the construction of such a sampler.
    *   **Workflow**:
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

*   **`get_inits(...)`**: Generates initial values for MCMC sampling. It can use a heuristic based on prior samples or run a fast MAP optimization to find a high-density starting point.

*   **`predict(...)`**: The main function for out-of-sample prediction. It creates a prediction-set configuration object (`PS`) by inheriting the structure of the training model (`M`) and updating it with the new data. It then calls `_reconstruct` to generate predictions on the new grid.

*   **`bstm_loo(...)` & `compare_manifolds(...)`**: Wrappers around `PosteriorStats.jl` for performing Leave-One-Out Cross-Validation and formal model comparison based on the ELPD metric.

---

## Appendix 2: Fast Lookup Cheatsheet

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

### 8.2. Likelihood Families

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
| **Dirichlet**         | `:dirichlet`        | `exp(eta)`      | `concentration (α)`: `exp.(eta)`. For multivariate compositional data.                                          |
| **Inverse Wishart**   | `:inverse_wishart`  | `identity(eta)` | `d.f. (ν)`: `extra_params`, `Scale Matrix (Ψ)` from `eta`. For covariance modeling.                             |

### 8.3. `spatial()` Module

The adjacency matrix `W` can be passed as a keyword argument to the main `@bstm` call (e.g., `@bstm(..., W=my_matrix)`) or, preferably, directly within the `spatial` module (e.g., `spatial(s_idx, W=my_matrix)`).

| Manifold             | `model='...'`       | Key Parameters                                | Default PC-Priors                                               | Use Case & Utility                                                                                     |
| :---------------------| :--------------------| :----------------------------------------------| :----------------------------------------------------------------| :-------------------------------------------------------------------------------------------------------|
| **IID**              | `'iid'`             | `sigma`                                       | `Exponential(1.0)`                                              | Models non-spatial overdispersion or heterogeneity.                                                    |
| **ICAR / Besag**     | `'icar'`, `'besag'` | `sigma`                                       | `Exponential(1.0)`                                              | Provides strong, localized spatial smoothing for lattice data.                                         |
| **BYM2**             | `'bym2'`            | `sigma`, `rho`                                | `sigma`: `Exponential(1.0)`, `rho`: `Beta(1,1)`                 | The most robust default for areal data; separates spatial clustering from random noise.                |
| **Leroux**           | `'leroux'`          | `sigma`, `rho`                                | `sigma`: `Exponential(1.0)`, `rho`: `Beta(1,1)`                 | A flexible alternative to BYM2 that avoids the rank-deficiency of the ICAR model.                      |
| **SAR**              | `'sar'`             | `sigma`, `rho`                                | `sigma`: `Exponential(1.0)`, `rho`: `Beta(1,1)`                 | Models spatial "spill-over" effects where the value at one location directly influences its neighbors. |
| **SPDE**             | `'spde'`            | `sigma`, `kappa`                              | `sigma`: `Exponential(1.0)`, `kappa`: `Exponential(1.0)`        | A scalable and principled way to model continuous spatial processes on irregular domains.              |
| **Gaussian Process** | `'gp'`              | `sigma`, `lengthscale`, `kernel`              | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Gold-standard for continuous spatial modeling but computationally expensive ($O(N^3)$).                |
| **RFF**              | `'rff'`             | `sigma`, `lengthscale`, `n_features`          | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | A scalable approximation to a full GP, excellent for large numbers of areal units.                     |
| **FITC**             | `'fitc'`            | `sigma`, `lengthscale`, `n_inducing`          | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Sparse GP using inducing points; good for large N.                                                     |
| **Nystrom**          | `'nystrom'`         | `sigma`, `lengthscale`, `n_inducing`          | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Low-rank GP approximation, similar to FITC.                                                            |
| **SVGP**             | `'svgp'`            | `sigma`, `lengthscale`, `n_inducing`          | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Sparse Variational GP, for use with VI.                                                                |
| **Warp**             | `'warp'`            | `sigma`, `lengthscale`, `n_features`          | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Models non-stationary fields by warping coordinates.                                                   |
| **NetworkFlow**      | `'network'`         | `sigma`, `adjacency_matrix`, `flow_direction` | `Exponential(1.0)`                                              | For directed graphs like river networks or supply chains.                                              |
| **DAG**              | `'dag'`             | `sigma`, `adjacency_matrix`                   | `Exponential(1.0)`                                              | For Directed Acyclic Graphs, useful in causal inference.                                               |
| **Mosaic**           | `'mosaic'`          | `sigma`, `n_regions`                          | `Exponential(1.0)`                                              | Partitions space into locally stationary regions.                                                      |
| **BCGN**             | `'bcgn'`            | `sigma`, `bipartite_adj`                      | `Exponential(1.0)`                                              | For bipartite graphs (e.g., user-item interactions).                                                   |
| **Hyperbolic**       | `'hyperbolic'`      | `sigma`, `curvature`                          | `Exponential(1.0)`                                              | For embedding hierarchical or tree-like spatial data.                                                  |

### 8.4. `temporal()` and `seasonal()` Modules

| Manifold              | `model='...'` | Key Parameters                                   | Default PC-Priors                                               | Use Case & Utility                                                                                |
| :----------------------| :--------------| :-------------------------------------------------| :----------------------------------------------------------------| :--------------------------------------------------------------------------------------------------|
| **IID**               | `'iid'`       | `sigma`                                    | `Exponential(1.0)`                                              | Models unstructured temporal noise.                                                               |
| **AR1**               | `'ar1'`       | `sigma`, `rho`                       | `sigma`: `Exponential(1.0)`, `rho`: `Beta(1,1)`                 | Modeling serially correlated time series where the influence of past events decays geometrically. |
| **Random Walk (RW1)** | `'rw1'`       | `sigma`                                    | `Exponential(1.0)`                                              | Capturing abrupt changes or step-like trends.                                                     |
| **Random Walk (RW2)** | `'rw2'`       | `sigma`                                    | `Exponential(1.0)`                                              | The most common choice for modeling smooth, non-linear temporal trends.                           |
| **Gaussian Process**  | `'gp'`        | `sigma`, `lengthscale`               | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Flexible, non-parametric trend modeling.                                                          |
| **RFF**               | `'rff'`       | `sigma`, `lengthscale`, `n_features` | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | Scalable GP approximation for long time series.                                                   |
| **Cyclic**            | `'cyclic'`    | `sigma`, `period`                          | `Exponential(1.0)`                                              | Modeling smooth, periodic effects like day-of-week or month-of-year.                              |
| **Harmonic**          | `'harmonic'`  | `amplitude`, `phase`, `period`                   | `amplitude`: `Normal(0,1)`, `phase`: `Beta(1,1)`                | Capturing sharp, regular periodic patterns with sine and cosine waves.                            |


### 8.5. `smooth()` Module (Covariate Smoothers)

| Manifold / Method           | `model='...'`    | Key Parameters                  | Default Priors                                                  | Use Case & Utility                                                                             |
| :-------------------------- | :--------------- | :------------------------------ | :-------------------------------------------------------------- | :--------------------------------------------------------------------------------------------- |
| **P-Spline**                | `'pspline'`      | `nbins`, `degree`, `diff_order` | `sigma`: `Exponential(1.0)`                                     | The most flexible general-purpose smoother for 1D covariates.                                  |
| **B-Spline**                | `'bspline'`      | `nbins`, `degree`               | `sigma`: `Exponential(1.0)`                                     | A simpler spline smoother than P-splines, useful when less regularization is desired.          |
| **Thin Plate Spline**       | `'tps'`          | `nbins`                         | `sigma`: `Exponential(1.0)`                                     | The classic choice for smoothing 2D spatial coordinates (e.g., `smooth(lon, lat, model=tps)`). |
| **Random Fourier Features** | `'rff'`          | `n_features`, `lengthscale`     | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | A highly scalable method for approximating a full Gaussian Process smooth.                     |
| **Random Walk (on bins)**   | `'rw1'`, `'rw2'` | `nbins`                         | `sigma`: `Exponential(1.0)`                                     | A powerful way to model a non-linear effect as a structured random effect on discretized bins. |
| **Gaussian Process (on coords)** | `'gp'`           | `lengthscale`                   | `sigma`: `Exponential(1.0)`                                     | Gold-standard continuous smoother, computationally intensive.                                  |
| **FFT Basis**               | `'fft'`          | `nbins`, `lengthscale`          | `sigma`: `Exponential(1.0)`, `lengthscale`: `InverseGamma(3,3)` | For modeling periodic non-linear effects of a covariate with a learnable period.               |
| **Moran's I Basis**         | `'moran'`        | `nbins`, `W`                    | `sigma`: `Exponential(1.0)`                                     | Uses eigenvectors of a spatial weights matrix as basis functions.                              |
| **Spherical Basis**         | `'spherical'`    | `nbins`, `range`                | `sigma`: `Exponential(1.0)`                                     | For effects with a strictly local influence (compact support).                                 |
| **Exponential Decay Basis** | `'decay'`        | `nbins`, `lengthscale`          | `sigma`: `Exponential(1.0)`                                     | For effects with a strong, rapidly decaying influence.                                         |
| **Barycentric Basis**       | `'barycentric'`  | `nbins`                         | `sigma`: `Exponential(1.0)`                                     | Simple, interpretable piecewise linear smoother.                                               |

### 8.6. `mixed()` Module

*Note: Direct censoring of covariates in `mixed()` is not supported. See Section 6.5 for the recommended joint modeling approach.*

| Syntax               | Example Usage                 | Key Parameters | Default Priors              | Mathematical Assumption                                                                                                                      |
| :------------------- | :------------------------------ | :------------- | :-------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| **Random Intercept** | `mixed(1, group_var)`         | `model`        | `sigma`: `Exponential(1.0)` | Assumes each level $j$ of `group_var` has a unique intercept $\alpha_j \sim \mathcal{N}(0, \sigma^2_{\text{group}})$.                        |
| **Random Slope**     | `mixed(covariate, group_var)` | `model`        | `sigma`: `Exponential(1.0)` | Assumes the effect (slope) of a `covariate` varies across the levels of `group_var`, $\beta_j \sim \mathcal{N}(0, \sigma^2_{\text{slope}})$. |

### 8.7. `dynamics()` Module

| Model                   | `model='...'`           | Key Parameters                               | Default Priors                                                              |
| :---------------------- | :---------------------- | :------------------------------------------- | :-------------------------------------------------------------------------- |
| **Advection**           | `'advection'`           | `velocity`, `sigma`                          | `velocity`: `Normal(0,0.5)`, `sigma`: `Exponential(1.0)`                    |
| **Diffusion**           | `'diffusion'`           | `diffusion`, `sigma`                         | `diffusion`: `LogNormal(-1,1)`, `sigma`: `Exponential(1.0)`                 |
| **Advection-Diffusion** | `'advection_diffusion'` | `velocity`, `diffusion`, `sigma`             | `velocity`: `Normal(0,0.5)`, `diffusion`: `LogNormal(-1,1)`                 |
| **Gompertz Growth**     | `'gompertz'`            | `r`, `K`, `sig_dyn`                          | `r`: `LogNormal(-1.5,0.5)`, `K`: `Normal(150,50)`                           |
| **Logistic Growth**     | `'logistic_basic'`      | `r`, `K`                                     | `r`: `LogNormal(0,1)`, `K`: `Normal(150,50)`                                |

### 8.8. `nested()` and `eigen()` Modules

The `nested()` and `eigen()` modules provide advanced capabilities for multi-fidelity modeling and dimensionality reduction, respectively.

#### `nested()` Module Reference

The `nested()` module is a powerful "supervisor" component used for multi-fidelity modeling and model stacking. It allows you to define a complete sub-model that is fit to a separate (often larger, lower-quality) dataset. The latent effect from this sub-model is then incorporated as a calibrated predictor into the main model, allowing the main model to "learn" from the proxy data.

| Keyword / Parameter     | Example Usage                                                         | Data Type | Default            | Meaning & Assumptions                                                                                                                                                                                                                                 |
| :---------------------- | :---------------------------------------------------------------------- | :-------- | :----------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nested()`              | `nested(z_var; ...)`                                                  | Module    | N/A                | Defines a supervised sub-model whose latent effect is added to the main model's linear predictor. The `z_var` is a symbolic name for this component.                                                                                                  |
| `formula`               | `formula="likelihood(z, family=gaussian) ~ intercept() + spatial(s)"` | `String`  | `""`               | A complete `bstm` formula string that defines the structure of the sub-model, including its own likelihood. This sub-model is fit to the specified `data_source`.                                                                                     |
| `data_source`           | `data_source=proxy_data`                                              | `Symbol`  | `:data`            | A symbol pointing to a `DataFrame` passed as a keyword argument to the main `bstm()` call. This allows the sub-model to use a different dataset.                                                                                                      |
| `rho_nested` (Implicit) | N/A                                                                   | `Float`   | `Normal(1.0, 0.5)` | A scaling coefficient that links the sub-model's latent effect to the main model's linear predictor: $\eta_{\text{main}} = \dots + \rho_{\text{nested}} \cdot \eta_{\text{sub}}$. The prior assumes the sub-model is a good proxy ($\rho \approx 1$). |

#### `eigen()` Module Reference

The `eigen()` module implements a Bayesian Principal Component Analysis (PCA) to perform dimensionality reduction on a set of multivariate outcomes. It decomposes the input variables into a smaller set of orthogonal latent factors (principal components). The first of these factors is then added to the main model's linear predictor, allowing you to use the dominant shared signal from multiple variables as a predictor.

| Keyword / Parameter | Example Usage              | Data Type      | Default            | Meaning & Assumptions                                                                                                                                                               |
| :------------------ | :------------------------- | :------------- | :----------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `eigen()`           | `eigen(y1, y2, y3; ...)`   | Module         | N/A                | Defines a Bayesian PCA factor model. The variables listed (e.g., `y1, y2, y3`) are the multivariate outcomes to be decomposed.                                                      |
| `n_factors`         | `n_factors=1`              | `Int`          | `1`                | The number of latent factors (principal components) to extract. This determines the dimensionality of the reduced latent space.                                                     |
| `pca_sd`            | `pca_sd=Exponential(0.5)`  | `Distribution` | `Exponential(1.0)` | The prior for the standard deviations of the principal components (latent factors). These are the "eigenvalues" of the system, controlling the variance explained by each factor.   |
| `pdef_sd`           | `pdef_sd=Exponential(0.5)` | `Distribution` | `Exponential(1.0)` | The prior for the standard deviation of the residual (uniqueness) noise. This captures the variance in each observed variable that is *not* explained by the shared latent factors. |

### 8.8. `fixed()` and `intercept()` Modules

These modules provide explicit control over standard regression components.

#### `fixed()` Module Reference

| Keyword / Parameter | Example Usage        | Data Type                 | Default        | Meaning & Assumptions                                                                                    |
| :------------------ | :------------------- | :-------------------------- | :------------- | :------------------------------------------------------------------------------------------------------- |
| `fixed()`           | `fixed(Region, ...)` | Module                    | N/A            | Explicitly marks a variable as a fixed effect. Primarily used to specify contrasts or priors.            |
| `contrast`          | `contrast=:effects`  | `Symbol`                  | `DummyCoding`  | Specifies the contrast coding for a categorical variable (e.g., `:effects`, `:helmert`).                 |
| `prior`             | `prior=Normal(0, 2)` | `Distribution` or `Tuple` | `Normal(0, 5)` | Sets the prior for the coefficient(s) of this fixed effect. Can be a `Distribution` or a PC prior tuple. |

#### `intercept()` Module Reference

| Keyword / Parameter | Example Usage          | Data Type                 | Default        | Meaning & Assumptions                                                                                                                |
| :------------------ | :----------------------- | :-------------------------- | :------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| `intercept()`       | `intercept(prior=...)` | Module                    | N/A            | Explicitly includes a global intercept. Using `1` in the formula is equivalent. This module is mainly for specifying a custom prior. |
| `prior`             | `prior=Normal(0, 10)`  | `Distribution` or `Tuple` | `Normal(0, 5)` | Sets the prior for the global intercept term. Can be a `Distribution` or a PC prior tuple.                                           |

#### Interaction Effects

Interaction effects between fixed covariates are specified using the standard `*` and `&` operators from `StatsModels.jl`. The `bstm` framework also supports the `:` operator as a synonym for `&`. These operators can be used both as bare terms in the formula and within the `fixed()` module.

*   `cov1 * cov2`: Expands to `cov1 + cov2 + cov1 & cov2` (main effects and interaction).
*   `cov1 & cov2`: Includes only the interaction term.
*   `cov1 : cov2`: Equivalent to `cov1 & cov2`.

**Example:**

```julia
# These three formulas are equivalent and include main effects and the interaction.
m1 = @bstm(likelihood(y) ~ intercept() + cov1 * cov2, data)
m2 = @bstm(likelihood(y) ~ intercept() + fixed(cov1 * cov2), data)
m3 = @bstm(likelihood(y) ~ intercept() + fixed(cov1) + fixed(cov2) + fixed(cov1 & cov2), data)

# These formulas include only the interaction term.
m4 = @bstm(likelihood(y) ~ intercept() + cov1 & cov2, data)
m5 = @bstm(likelihood(y) ~ intercept() + cov1 : cov2, data)
```

**Note on Priors:** Applying a custom prior to an interaction term (e.g., `fixed(cov1 * cov2, prior=...)`) is not directly supported, as the prior would be ambiguous across the expanded main and interaction effects. To assign a specific prior to an interaction, you must first manually create the interaction term as a new column in your `DataFrame` and then apply the `fixed()` module with a `prior` to that new column.

## 9. Conclusion

The `bstm` framework provides a Julia-native environment for spatiotemporal modeling that emphasizes composability. By treating latent geometries as distinct, combinable entities, it allows for the construction of complex models that remain computationally tractable. The standardized use of PC-Priors offers a principled way to maintain identifiability, while the modular formula interface facilitates model specification and interpretation.

## 10. References

*   Besag, J. (1974). Spatial interaction and the statistical analysis of lattice systems. *Journal of the Royal Statistical Society: Series B (Methodological)*, 36(2), 192-225.
*   Besag, J., York, J., & Mollié, A. (1991). Bayesian image restoration, with applications in spatial statistics. *Annals of the Institute of Statistical Mathematics*, 43(1), 1-59.
*   Cliff, A. D., & Ord, J. K. (1973). *Spatial autocorrelation*. Pion.
*   Damianou, A., & Lawrence, N. (2013, April). Deep gaussian processes. In *Artificial intelligence and statistics* (pp. 207-215). PMLR.
*   Gelfand, A. E., Kim, H. J., Sirmans, C. F., & Banerjee, S. (2003). Spatial modeling with spatially varying coefficient processes. *Journal of the American Statistical Association*, 98(462), 387-396.
*   Gelfand, A. E., & Vounatsou, P. (2003). Proper multivariate conditional autoregressive models for spatial data analysis. *Biostatistics*, 4(1), 11-15.
*   Riebler, A., Sørbye, S. H., & Rue, H. (2016). An intuitive Bayesian spatial model with two hyperparameters. *Statistical Methods in Medical Research*, 25(2), 1145-1160.
*   Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time variation in disease risk. *Statistical Methods in Medical Research*, 9(3), 205-220.
*   Leroux, B. G., Lei, X., & Breslow, N. (2000). Estimation of disease rates in small areas: a new mixed model for spatial dependence. In *Statistical models in epidemiology, the environment, and clinical trials* (pp. 179-191). Springer, New York, NY.
*   Lewandowski, D., Kurowicka, D., & Joe, H. (2009). Generating random correlation matrices based on vines and extended onion method. *Journal of multivariate analysis*, 100(9), 1989-2001.
*   Mullahy, J. (1986). Specification and testing of some modified count data models. *Journal of econometrics*, 33(3), 341-365.
*   Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
*   Rahimi, A., & Recht, B. (2008). Random features for large-scale kernel machines. *Advances in Neural Information Processing Systems*, 20.
*   Lindgren, F., Rue, H., & Lindström, J. (2011). An explicit link between Gaussian fields and Gaussian Markov random fields: The SPDE approach. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 73(4), 423-498.
*   Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sørbye, S. H. (2017). Penalising model component complexity: A principled, practical approach to constructing priors. *Statistical Science*, 32(1), 1-28.
*   Snelson, E., & Ghahramani, Z. (2006). Sparse Gaussian processes using pseudo-inputs. *Advances in neural information processing systems*, 18.
*   Wikle, C. K. (2003). Hierarchical Bayesian models for predicting the spread of ecological processes. *Ecology*, 84(6), 1382-1394.
*   Williams, C. K., & Seeger, M. (2001). Using the Nyström method to speed up kernel machines. In *Advances in neural information processing systems*, 13.