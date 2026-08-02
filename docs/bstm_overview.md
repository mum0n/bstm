---
title: "bstm: Bayesian Spatiotemporal Models in Julia/Turing"
header: "*bstm* in Julia"
keyword: |
	Keywords - Gaussian Process, CAR, Spatiotemporal models
abstract: |
	Bayesian SpatioTemporal Models in Julia.

metadata-files:
  - "_metadata.yml"

format:
  html:
    code-fold: true

engine: julia

execute:
  eval: false
---

## Abstract
 
Bayesian Space-Time Models in Julia (*bstm*) is a Julia library that combines elements of spatial partitioning methods (for discrete modelling) with Bayesian spatiotemporal models and general statistical modeling. At its core is a discrete perspective upon space and time, not for philosophical reasons, but for operational functionality. Spatiotemporal models are resource-intensive. This discrete perspective permits useful solutions within the constraints of most currently available computing resources. After developing these discrete approximations, we explore continuous Gaussian Process methods. Though the focus is upon Ecological applications, the framework is sufficiently general that it can be readily adapted to any spatiotemporal process, no matter how large or small. 

Simplistically, *bstm* can be seen as a domain-specific language ("DSL") that brings in elements of statistical modelling languages found in other platforms to build Turing-based models, powered by a large number of Julia libraries. It is reasonably complete and, importantly, extensible. Using Julia leverages the power and flexibility of the language (especially the Bayesian Turing.jl framework), with a compact, flexible, and extensible set of functions and tools. 

Ultimately, here, we are developing a general framework to explore various models of increasing complexity to handle measurement error, periodic dynamics, and spatial dependencies, and to enable parameter estimation of custom dynamical/process models embedded in these structural domains. Random Fourier Features (RFF), Fast Fourier Transform (FFT), Fully Independent Training Conditional (FITC), and Deep and Inducing Point Gaussian Processes are explored to make Discrete and Continuous Spatiotemporal models computationally tractable for large datasets.

  
## Introduction: The SpatioTemporal Challenge 

Ecological monitoring is a pursuit of moving targets. To usefully model important variables like bottom temperature, species composition, and the population dynamics of species, one must almost always deal with incomplete or low-density information from expensive surveys with limits to resources and time. The usual recourse is some variation of Random Stratified Sampling to "absorb" unaccounted errors or "externalities" as unstructured, **independent**, random effects. This can be fine in simple settings. In dynamic environments, no matter how good you think the stratification may be, it will induce bias. In *bstm*, we do not ignore these "externalities"; assuming no bng ias, we embrace them as they are also usually informative. Though *bstm* can be used for the former, it shines as a high-dimensional Bayesian hierarchical framework designed to decompose complex spatiotemporal data into interpretable latent components. This is because ecological data is inherently **dependent** or structured. 

To address the "SpatioTemporal Challenge," *bstm* utilizes three primary components:

1. Spatial Clustering: Implemented via spatial autocorrelation specifications to account for geographical neighborhoods.
2. Temporal Autocorrelation: Utilizing temporal autocorrelation to capture evolving trends.
3. Non-linear Interactions: Modeling complex interactions where the relationship between space and time is non-stationary and dynamic.

Failing to distinguish between a permanent habitat feature (captured by a spatial component), a strong directional change such as global temperature increases (captured by a time-series component), and a transient environmental anomaly such as a strong El Niño/La Niña event or Gulf Stream flow (captured by the space-time interaction), or variability change across space and/or time (volatility) can result in biased hindcasts and forecasts. By isolating these effects, we ensure that our understanding and consequent management decisions are based on the "true" (latent) underlying drivers rather than improperly accounted statistical noise. This decomposition is made possible by transforming a computationally prohibitive problem into a tractable one. 

By using Julia and Turing, you get state-of-the-art computation and optimization with Automatic Differentiation built into most heavy-lifting operations. Even if you do not use the spatiotemporal modelling components of *bstm*, you can simply use it as a bridge to simple Bayesian modelling using an interface that should be comfortable for most people exposed to modern statistical platforms. Do we need yet another statistical front-end? Perhaps not if you are well-versed in Turing and computation in general. But for many, this will help bridge the technical demands until you no longer need it or use it to template a model and then adapt it further to your needs. *bstm* is also a tool written for myself to simplify my computational work and so will have quirks. Apologies in advance.  

As this document has, in part, a didactic purpose, it is structured like a notebook with explanations inter-spread with examples. The architectural and technical underpinnings are touched upon, and then a cookbook-like set of recipes for different situations to flesh out the approaches. I have made heavy use of Google's Gemini LLM to help with the formula parsing and double-checking of computational logic; it has been a powerful coding aid and also a cause of much exasperation. This is my contribution to Julia and the scientific community. It is fully open and given without condition, except that I ask that if you add or improve upon it, you share your improvements with the community. 

### Computation: Getting started with the environment

First, let us get the Julia environment set up before anything else. Here we use [Julia](https://julialang.org/), as in my experience, it is a clear didactic tool and better for long-term learning and simultaneously use in large projects due to maintainability of the code-base and high performance. It is an open-source platform created by mathematicians, engineers, natural scientists, statisticians, computer scientists and machine learning specialists, each bringing the best from their respective fields and lessons learned from domain-specific software platforms in a coherent and performative fashion. At the time of this writing, there still remain some lingering issues (start up speed, recompilation of code and incompatibility creep when there are updates to any library (we depend upon many well-established libaries to do the heavy lifting in the background), but the speed that is offered and code clarity in exchange is worth it in any serious data manipulation efforts. Your mileage will vary, but the lessons learned are also easily transportable to R, python, matlab, octave, etc., if forced to use those platforms. They each have their own quirks and challenges, but until their eventual convergence into something (that will likely look a lot like Julia), it is still a great platform to learn, teach and operate/develop cutting edge work. Many learning tools exist. [Have a look here for a curated list](https://julialang.org/learning/). See the Appendices for more details.
 
Installing [Julia](https://julialang.org/) is best done with [juliaup](https://github.com/JuliaLang/juliaup). It can make maintenance simpler. Most functions used here that are not part of a standard library are collected together in [Julia](https://julialang.org/) functions at [src](../src/). They can be loaded with supporting standard libraries.

**WARNING**: if this is your first run, this can take on the order of hours to install libraries and dependencies, so let it run in the background. You might need to re-start the Julia session if there are complex/multiple library dependency issues (that require or support different versions). 


```julia
#----------------------------------------------------------------------------------------
# Code Snippet 1: Project Setup
#
# This code sets up the project directory and loads necessary libraries and functions.
# This step is crucial for ensuring the environment is correctly configured.
#----------------------------------------------------------------------------------------
if Sys.iswindows()
    project_directory = joinpath("C:\\", "home", "jae", "projects", "bstm")
elseif Sys.islinux()
    project_directory = joinpath("/home", Sys.username(), "projects", "bstm")
else
    project_directory = joinpath("C:\\", "Users", "choij", "projects", "bstm")
end

# Load libraries and project-specific functions
include(joinpath(project_directory, "startup.jl"))

# load bstm functions
load_project_functions(srcdir())
 
```

If there continue to be issues with packages breaking, some more lower level package management and digging may be required or a restart of Julia.


### Example data: Scottish lip cancer data  

As a first step towards spatial modelling, we look at a minimal data series: the [Scottish Lip Cancer data](https://mc-stan.org/users/documentation/case-studies/icar_stan.html). It has been thoroughly studied on many platforms over the years. There are 56 areal units and a simulated temporal component. We do not have access to the map positional data, but we do have the adjacency information from which we can infer approximate spatial topology. In these discrete models, we really only need to know which areal units are neighbours (connected graph), encapsulated through the adjacency matrix $W$. The actual observations are formatted into a DataFrame with the correct variable names.  


```julia
#----------------------------------------------------------------------------------------
# Data Loading and Preparation
#
# Load the Scottish Lip Cancer dataset, which has been extended with a simulated
# temporal component for demonstration purposes.
#----------------------------------------------------------------------------------------

data_scot, _ = scottish_lip_cancer_data_spacetime()
inp_df = data_scot.data
W = data_scot.au.W

#----------------------------------------------------------------------------------------
# Visualize the spatial structure of the data. Since we only have adjacency information,
# we infer a spatial layout. We also visualize the spatial intensity of the data points.
#----------------------------------------------------------------------------------------
 
plot_spatial_graph(au=data_scot.au, plot_title="Lip Cancer Inferred from Adjacency 'Locations'")
plot_kde_simple(inp_df[!, [:s_x, :s_y]], sd_extension_factor=0.25, title="Spatial Intensity (KDE)")
 

```

In the dataset, we have counts (y) of cancer incidence and population size in each area (log_offset). We also simulate a 10-"year" temporal process, a random walk with magnitude 0.5 and a covariate effect (X: an area-specific continuous covariate that represents the proportion of the population employed in agriculture, fishing, or forestry). An overall random uniform observation error of magnitude 0.2 is added with a count then taken as the overall, rounded integer value.


#### Spatiotemporal model: the shape of things to come

Before getting into the nitty gritty of the spatiotemporal models, for the impatient, let us go through a contrived example to see what the overall workflow is like. After this section we will study the details of *bstm* options more systematically.

The Standard Separable Spatiotemporal Model with a Poisson-distributed cancer counts `y`, using a log-offset for population exposure. The linear predictor includes:

- An intercept.
- A fixed effect for the covariate `X` (proportion employed in agriculture, etc.).
- A spatial random effect using the BYM2 model to capture structured and unstructured spatial variation.
- A temporal random effect using an AR1 process to capture serial correlation over years.
The spatial and temporal effects are "separable" as no interaction term is included.

Note the semicolon at the end of the call: m = @bstm(); make the contents not print to screen. You can of course remove it to see what it really contains: a simple compiled Turing model. The use of `:` (:poisson, for example) indicates a symbol in Julia, and can be used, though it is not necessary as everything is treated just as strings in in the end.

```julia
m  = @bstm(
    likelihood(y, family=:poisson, log_offsets=:log_offsets) ~
        intercept(prior=Normal(0, 10)) +
        fixed(X, prior=Normal(0, 5)) + # Fixed effect for covariate X
        random(s_idx, model=:bym2) + # Spatial random effect
        random(year, model=:ar1), # Temporal random effect
    inp_df 
);

# For demonstration, we run a short chain with a simple sampler. 
chn_separable = sample( m , MH(), 100; progress=false)

# For a full analysis:
os = get_optimal_sampler(m)
chn = sample(m, os, 1000, nchains=4)  # you will need to tweak the number of samples, as this depends upon model and data
res = model_results_comprehensive( m, chn; au=data_scot.au)
model_results_plots(res)
```

The results from a short run of this model can be examined to check for convergence (e.g., r-hat values close to 1) and to interpret the posterior distributions of the parameters. A full analysis requires longer MCMC chains and multiple chains to properly assess convergence and posterior uncertainty.

## Formula-based Framework

The framework uses a formula interface inspired by R's `glm`, `lme4` and `brms`, but with specific modules for spatiotemporal components. A model is defined with the observation model on the left-hand side (LHS) and the latent process model on the right-hand side (RHS). The `@bstm` macro is used to manipulate formula syntax for model building in Turing syntax.

The general structure of a `bstm` model call is:

```julia
m = @bstm(
    likelihood(outcome_var, family=poisson, ...) ~ 
intercept() + fixed_effects + modules(...),
    data_frame,
    keyword_arguments...
)
```

The notation is similar with specific modules for manipulating spatiotemporal and other components. 

Any keyword arguments provided after the `data_frame` are passed into the model's configuration. A notable general keyword is `verbose`. **`verbose=false`**: Suppresses the printing of the dynamically generated model code and the results of the automatic prior predictive check that runs at instantiation. This is useful for cleaner output in scripts or notebooks. The default is `true`.

In `bstm` formulas, specific function-like terms, or "modules," are used to define the model's structure. These modules tell the pre-processor how to construct latent fields, handle covariates, and set priors. By default, all models include an intercept. To explicitly control this, use the `intercept()` module. 

NOTE: using """ ... """, below, makes construction of long text strings simpler in Julia. Here `s_idx` (spatial unit index) and `year` (time unit index) are column names in the input DataFrame.
 

**Example Formula**
```
formula = """
  likelihood(y, family=:poisson, offsets=log_pop) ~
    intercept(prior=Normal(0, 10)) +
    fixed(z, prior=Normal(0, 5)) +
    fixed(Region, contrast=:effects, prior=Normal(0, 2)) +
    poverty |> random(s_idx, model=:icar) + # Spatially varying coefficient for poverty
    random(s_idx, model=:besag) ⊗ random(year, model=:ar1) + # Spatiotemporal interaction
    random(age, model=:pspline, nbins=10) # Smooth effect for age
"""

m = @bstm( formula, df, W=W );
```
  
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

1.  **BYM2 Disease Mapping (Spatial):**
    `@bstm(likelihood(y, family=poisson) ~ intercept() + random(s_idx, model=:bym2), data, W=W)`
    Decomposes risk into structured spatial and unstructured IID noise.

2.  **AR1 Temporal Forecasting (Temporal):**
    `@bstm(likelihood(y) ~ intercept() + random(t_idx, model=:ar1), data)`
    Captures geometric temporal decay.

3.  **Spatio-Temporal Interaction (Spacetime):**
    `@bstm(likelihood(y) ~ intercept() + random(s_idx, model=:besag) ⊗ random(t_idx, model=:ar1), data, W=W)`
    Employs the Kronecker product to create a fully structured spatiotemporal interaction field.

4.  **Spatially Varying Coefficients (SVC) and Curves (Spatial/Smooth):**
    `@bstm(likelihood(y) ~ intercept() + (poverty |> random(s_idx, model=:icar)), data, W=W)`
    Allows the impact of `poverty` to vary according to local spatial gradients.

    `@bstm(likelihood(y) ~ intercept() + (random(s_idx, model=:icar) |> random(t_idx, model=:pspline)), data, W=W`

    Models a temporal trend that varies smoothly across space. (Note: `random(t_idx, model=:pspline)` implies `structure=:smooth` for `t_idx` as a covariate).
 

More complex examples can be found in the Appendix.


#### Covariate Discretization & Transformation Rules

The `bstm` framework supports several methods for preprocessing covariates and their interactions:
  
1.  **Discretization / Binning:**
    *   `Int` (e.g., `9`): Discretizes into N quantiles. Useful for creating non-linear effects via random effect structures (RW2/AR1).
    *   `"regular:XXX"` (e.g., `"regular:10"`): Creates XXX equal-width intervals between the 0.025 and 0.975 quantiles of the data.
    *   `AbstractVector` (e.g., `[0.1, 0.5, 0.9]`): Uses the provided vector as custom bin edges.

2.  **Interactions:**
    *   Interactions are specified as `"var1*var2"`. They are calculated *after* the individual variables have been transformed (scaled/logged), ensuring interactions operate on normalized representations. Spatiotemporal Interaction Model (Knorr-Held Type IV) extends the separable model by adding a spatiotemporal interaction term. The `random(...) ⊗ random(...)` syntax specifies a Kronecker product interaction, allowing the spatial field to evolve over time. This is also known as a Knorr-Held Type IV interaction, the most complex type, where both space and time are structured.  

    ```julia
    m = @bstm(
        likelihood(y, family=:poisson, log_offsets=:log_offsets) ~ # Poisson likelihood with log offset
            intercept() +
            fixed(X) +
            random(s_idx, model=:bym2) + # Spatial random effect
            random(year, model=:ar1) + # Temporal random effect
            random(s_idx, model=:besag) ⊗ random(year, model=:ar1), # Spatiotemporal interaction
        inp_df,
        W = W # note as W is part of random() at multiple places, it is shorter to pass as an overall W
    );
    ```

The `bstm` structure-specific language operates through a recursive parser that allows for the algebraic composition of different model components, referred to as components.

#### Algebraic Operators

1.  **Kronecker Product (⊗)**: Used for creating inseparable interaction effects, such as the Knorr-Held Type IV model (`random(structure=:spatial) ⊗ random(structure=:temporal)`). This builds a joint precision matrix $Q_{st} = Q_t \otimes Q_s$, enabling the representation of space-time interactions where every spatial location has a unique, correlated temporal trend.
2.  **Composition (∘)**: Represents the functional composition of two components, where one component modulates the parameters of another. This is a powerful tool for creating non-stationary models. The `random()` module replaces the older `spatial()` and `smooth()` syntax.
3.  **Pipe (`|>`):** The pipe operator handles data normalization and state-space evolution.

*   **Transformations**: Objects like `ZScoreComponent` or `LogComponent` act as wrappers that normalize inputs before they enter the latent process.
*   **State-Space Evolution**: The pipe operator defines a state-space model where one component evolves over the structure of another. This supports both discrete-time dynamics (e.g., `random(structure=:spatial) |> random(model=:ar1)`) and the creation of spatially-varying curves (e.g., `random(structure=:spatial) |> random(time, model=:pspline)`), where the coefficients of the temporal basis functions are modeled as spatial fields.

### Core Components 

The `bstm` framework includes a registry of components that range from discrete graph-based models to continuous spectral approximations.

#### The Discrete Registry: Gaussian Markov Random Fields (GMRF)

For discrete structures, `bstm` implements GMRF structures where dependency is defined by a precision matrix Q.

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

#### Continuous, Spectral, and Advanced Components

To address the $O(N^3)$ computational cost of kernel-based Gaussian Processes, the framework utilizes spectral projections and sparse approximations:

- Random Fourier Features (RFF): Approximates a stationary kernel $k(\mathbf{x}, \mathbf{x}')$ by mapping input coordinates x into a randomized feature space. This is based on Bochner's theorem, which states that any stationary kernel is the Fourier transform of a non-negative measure (the spectral density). The projection is defined as $z(x) = \sqrt{2/M} \cos(Wx + b)$, where W is a matrix of frequencies sampled from the kernel’s spectral density. This transforms the GP into a more scalable Bayesian linear regression problem.

Syntax: random(x, y, model=:rff, n_features=100, lengthscale=...)

SPDE (Stochastic Partial Differential Equation): Models a continuous Matérn field by interpreting it as the solution to the SPDE $(\kappa^2 - \Delta)^{\alpha/2} u = \mathcal{W}$, where $\mathcal{W}$ is Gaussian white noise. This approach, popularized by Lindgren et al. (2011), creates an explicit link between the continuous GP and a discrete GMRF, allowing for scalable inference via sparse precision matrices derived from a finite element mesh or regular grid.

Syntax: random(s_idx, model=:spde, kappa=...)

Nyström / FITC (Inducing Point Methods): These methods create a low-rank approximation of the full GP by summarizing the data through a small set of $M$ "inducing points."

- Nyström: Approximates the full kernel matrix $K_{NN}$ with a low-rank version $\tilde{K} = K_{NM} K_{MM}^{-1} K_{MN}$.

- FITC (Fully Independent Training Conditional): Assumes that, conditional on the GP's values at the inducing points, the observations are independent. This is computationally efficient and forms the basis for many sparse GP models.

Syntax: random(x, y, model=:fitc, n_inducing=50)

- FFT (Fast Fourier Transform): For regularly gridded data, the eigenvectors of the graph Laplacian are the discrete Fourier basis functions. This allows the spatial filtering operation (i.e., applying the inverse of the precision matrix) to be performed with the Fast Fourier Transform, reducing the complexity from matrix-vector multiplication to $O(N \log N)$.

Syntax: random(s_idx, model=:fft)

- Wavelet: Provides a multi-resolution analysis of a spatial or temporal field. It decomposes the signal into components at different scales (frequencies), which is useful for capturing both broad trends and localized, high-frequency details. The coefficients of the wavelet basis functions are typically given a shrinkage prior.

Syntax: random(x, model=:wavelet, family=:db4)

NetworkFlow: A specialized GMRF for directed graphs, such as river networks or supply chains. It models dependencies where influence is directional. The flow_direction parameter controls whether the process models influence flowing from upstream to downstream neighbors or vice-versa.

Syntax: random(s_idx, model=:network, flow_direction=:downstream, W=directed_W)

- LGCP (Log-Gaussian Cox Process): A model for spatial or spatiotemporal point patterns. It assumes that the points are a realization of a Poisson point process whose log-intensity is a Gaussian Process. The bstm framework models the aggregated counts in discrete areal units by integrating the latent intensity surface over the area of each unit.

Syntax: (pointprocess(model=:lgcp, grid_areas=...) ∘ random(s_idx, model=:icar))

#### Priors and Identifiability

Stability in high-dimensional models is achieved through a principled approach to prior specification. The `bstm` framework provides three built-in prior schemes and allows for user-defined overrides.

##### Prior Schemes

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

### Architectural Paradigms

#### Univariate Architecture

The default kernel for single-outcome processes, as described in the preceding sections.

#### Multivariate Architecture

The `MultivariateArchitecture` is triggered when multiple outcomes are specified on the LHS of the formula (e.g., `y1 + y2 ~ ...`). It is designed to jointly model these outcomes, allowing the model to "borrow strength" across related processes and estimate the correlation structure between them.

##### Key Mechanisms:

*   **Outcome-Specific Parameters**: Each component (e.g., `random(structure=:spatial)`) generates a separate latent field for each outcome. This means hyperparameters like `sigma_spatial` or `rho_temporal` are estimated independently for each response variable, providing maximum flexibility.
*   **LKJ Correlation Prior**: The core of the multivariate coupling is the `LKJCholesky` prior on a correlation matrix `L_corr`. The final latent effects are constructed by multiplying the matrix of independent latent fields by the Cholesky factor of this correlation matrix: `eta_final = eta_latent * L_corr.L`. This induces a shared correlation structure across all outcomes for a given component, ensuring that the model captures shared patterns while allowing for outcome-specific variances.
*   **Householder Reflection (Spectral Orientation)**: For advanced use cases, the framework can apply a Householder reflection (`H = I - 2vv'`) to the latent fields. This allows the outcomes to rotate in the latent space, which is useful for aligning signals in models with complex dependencies like transport or advection-diffusion, where the direction of correlation is as important as its magnitude.

#### Multifidelity Architecture

The `MultifidelityArchitecture` is designed for data fusion, integrating high-volume, low-cost proxy data with sparse, high-quality observations. It is typically invoked using the `nested()` module.

##### Key Mechanisms:

*   **Hierarchical Latent Fields**: The architecture establishes a hierarchy of latent processes. A common setup involves:
    1.  **High-Fidelity (Target)**: The primary outcome variable (`y_hq`).
    2.  **Low-Fidelity (Proxy)**: A secondary, related variable (`y_lq`) with more abundant data.
*   **Nested Supervision**: The `nested()` module defines a complete sub-model for the low-fidelity data. The latent field from this sub-model (`eta_sub`) is then used as a calibrated predictor in the main model for the high-fidelity data.
*   **Calibration Parameters**: The link between the fidelities is modeled with calibration parameters, typically a bias and a scaling factor (`rho`), which are estimated within the model:
    `eta_main = ... + rho_nested * eta_sub`
    The prior on `rho_nested` is often centered around 1.0, assuming the proxy is a reasonably good, if biased, predictor of the main process. This allows the main model to learn from the structural patterns in the low-fidelity data while correcting for systematic bias and scale differences.

### Advanced Topics

#### Spatial Partitioning

For discrete spatial models (GMRFs), the continuous spatial structure must be discretized into "Areal Units" (AUs). The `assign_spatial_units` function provides several methods for this, balancing geometric compactness with statistical information density.

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

*   **Binary Vector Tree (`:bvt`)**: A high-speed hierarchical method that recursively splits the structure along the axis of maximum variance. It is the fastest approach for massive datasets and excels at creating units with balanced point counts.

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
    likelihood(counts, family=poisson) ~ intercept() + dynamics(time, model=:logistic_f, r_covariate=temp),
    data
);
```

This example fits a logistic growth model where the intrinsic growth rate `r` is itself a function of temperature.

#### Multi-fidelity and Nested Models

The `nested()` module is a "supervisor" component for multi-fidelity modeling. It allows you to define a complete sub-model that is fit to a separate (often larger, lower-quality) dataset. The latent effect from this sub-model is then incorporated as a calibrated predictor into the main model, allowing the main model to "learn" from the proxy data. The `nested()` module accepts a full formula string, including a `likelihood()` block, which enables the specification of independent likelihoods for each fidelity level.

```julia

# Define a true, underlying non-linear function
true_function(x) = sin(x * 2 * pi) .* x.^2

# Generate low-fidelity (LF) data: abundant but noisy
n_lf = 200
x_lf = rand(n_lf) .* 2 .- 0.5 # Spread points across a range
y_lf = true_function.(x_lf) .+ rand(Normal(0, 0.4), n_lf)
df_lf = DataFrame(x_lq = x_lf, y_lq = y_lf)

# Generate high-fidelity (HF) data: sparse but more accurate
# This data is a biased and scaled version of the true function.
n_hf = 30
x_hf = rand(n_hf) .* 2 .- 0.5
bias = 0.5
scale = 1.5
y_hf = bias .+ scale .* true_function.(x_hf) .+ rand(Normal(0, 0.1), n_hf)
df_hf = DataFrame(x_hq = x_hf, y_hq = y_hf)
 
 
# Define the formula for the low-fidelity sub-model inside  nested() .
  
# Define the main model for the high-fidelity data.
# It includes an intercept to learn the 'bias' and the above 'nested' component.
# The framework will automatically create a 'rho_nested_proxy' parameter to learn the 'scale'.
fm = "likelihood(y_lq, family=gaussian) ~ 0 + random(x_lq, model=pspline, nbins=20)"
model_mf = @bstm(
    likelihood(y_hq, family=gaussian) ~
        intercept() +
        nested(
            proxy,
            formula = fm,
            data_source = :low_quality_data # This must match the keyword argument below
        ),
    df_hf, # The main data is the high-fidelity set
    low_quality_data = df_lf # Pass the low-fidelity data as a keyword argument
);
 
 
chain_mf = bstm_sample_nowarn(model_mf, MH(), 1000; progress=false)
 

# --- 4. Results Extraction and Visualization --- 

# Extract comprehensive results, including posterior predictive summaries
# The `predict` function is used here to get predictions over a fine grid for plotting.
x_grid = range(minimum(x_lf), maximum(x_lf), length=200)
pred_grid_df = DataFrame(x_hq = x_grid) # Use the main model's variable name

# The prediction for the nested model requires the sub-model's variable name
# We create a prediction set for the sub-model.
sub_ps_data = DataFrame(x_lq = x_grid)
main_ps = (data=pred_grid_df, nested_prediction_sets=(proxy=sub_ps_data,))

# Reconstruct the model posteriors on the prediction grid
res_pred = model_results_comprehensive(model_mf, chain_mf; PS=main_ps)

# Extract predicted mean and credible intervals
pred_mean = res_pred.pstats.predictions_denoised.mean
pred_lower = res_pred.pstats.predictions_denoised.lower
pred_upper = res_pred.pstats.predictions_denoised.upper

# Extract the learned calibration parameters from the chain summary
summary_stats = res_pred.metrics.summarystats
intercept_est = summary_stats[summary_stats.parameters .== :intercept, :mean][1]
rho_est = summary_stats[summary_stats.parameters .== :rho_nested_proxy, :mean][1]

println("\n--- Model Results ---")
println("True Bias: $bias, Estimated Intercept: $(round(intercept_est, digits=3))")
println("True Scale: $scale, Estimated Rho (Scale): $(round(rho_est, digits=3))")

# Create the final plot
p = plot(title="Multifidelity Model Results", xlabel="x", ylabel="y", legend=:topleft)

# Plot the true underlying function
plot!(p, x_grid, bias .+ scale .* true_function.(x_grid), lw=3, color=:black, ls=:dash, label="True Scaled Function")

# Plot the high-fidelity and low-fidelity data points
scatter!(p, x_lf, y_lf, label="Low-Fidelity Data", markersize=2, alpha=0.3, color=:gray)
scatter!(p, x_hf, y_hf, label="High-Fidelity Data", markersize=4, markerstrokewidth=1.5, color=:red)

# Plot the model's posterior predictive mean and credible interval
plot!(p, x_grid, pred_mean, lw=3, color=:blue, label="Model Prediction (Mean)")
plot!(p, x_grid, pred_lower, fillrange=pred_upper, fillalpha=0.2, color=:blue, lw=0, label="95% Credible Interval")

display(p)

println("\nExample complete. The plot shows the model successfully recovering the true underlying function by leveraging both data sources.")

```

Another nested() example:

```julia

# Define a simple spatial structure (e.g., a 15x15 grid)
s_N = 225 # Total number of spatial locations
grid_dim = 15
s_coords = hcat(repeat(1:grid_dim, inner=grid_dim), repeat(1:grid_dim, outer=grid_dim))

# Create a simple adjacency matrix 'W' from the grid
W = spzeros(Int, s_N, s_N)
for i in 1:s_N
    for j in (i+1):s_N
        dist = sqrt((s_coords[i, 1] - s_coords[j, 1])^2 + (s_coords[i, 2] - s_coords[j, 2])^2)
        if dist <= 1.1 # Queen contiguity
            W[i, j] = 1
            W[j, i] = 1
        end
    end
end

# a. Simulate a true, latent covariate 'x_true' with spatial structure
# This field represents the real, unobserved quantity.
true_spatial_effect_x = rand(MvNormal(zeros(s_N), 0.2 * I + inv(Matrix(W) .+ 1e-6I) ), 1)[:]
x_true = 2.0 .+ 1.5 .* (s_coords[:, 1] ./ grid_dim) .+ true_spatial_effect_x

# b. Censor the covariate 'x_true' to create the observed version 'x_censored'
detection_limit = 2.5
x_censored = copy(x_true)
is_censored = x_true .< detection_limit
x_censored[is_censored] .= detection_limit # Set censored values to the limit

# Create censoring bound columns for the model
x_lower = fill(-Inf, s_N)
x_upper = fill(Inf, s_N)
x_upper[is_censored] .= detection_limit # For censored points, the upper bound is the limit

# c. Simulate another fully observed covariate 'z'
z = randn(s_N)

# d. Simulate the primary outcome 'y' (Poisson counts)
# The outcome depends on the TRUE latent covariate 'x_true'.
true_spatial_effect_y = rand(MvNormal(zeros(s_N), 0.15 * I + inv(Matrix(W) .+ 1e-6I)), 1)[:]
log_rate_y = 0.5 .+ 0.8 .* x_true .- 0.5 .* z .+ true_spatial_effect_y
rate_y = exp.(log_rate_y)
y = [rand(Poisson(r)) for r in rate_y]

# e. Assemble the final DataFrame
df = DataFrame(
    y = y,
    x_censored = x_censored,
    x_lower = x_lower,
    x_upper = x_upper,
    z = z,
    s_idx = 1:s_N
)

println("Data generation complete. $(sum(is_censored)) out of $s_N observations are censored.")


# --- 2. Model Definition ---
println("\nDefining the joint model using nested()...")

# a. Define the formula for the censored covariate sub-model.
# This model learns the distribution of the true 'x' values.
# It uses the 'censor_lower' and 'censor_upper' likelihood arguments.
sub_formula = """
    likelihood(x_censored, family=gaussian, censor_lower=x_lower, censor_upper=x_upper) ~
        intercept() +
        random(s_idx, model=bym2)
"""

# b. Define the main model for the outcome 'y'.
# This model uses the latent effect from the sub-model as a predictor.
# The 'nested()' module creates a component named 'latent_x_process'. The framework
# automatically adds this component's effect to the linear predictor of the main model.
model_joint = @bstm(
    likelihood(y, family=poisson) ~
        intercept() +
        fixed(z) +
        random(s_idx, model=bym2) +
        nested(
            latent_x_process,
            formula = sub_formula
        ),
    df,
    W = W,
    verbose = false
);

println("Model definition complete.")


# --- 3. MCMC Inference ---
println("\nRunning MCMC sampling (this may take a moment)...")

# Use a robust sampler like NUTS for this complex model.
# For a real analysis, more samples would be needed.
chain_joint = bstm_sample_nowarn(model_joint, NUTS(0.65), 1000; progress=true, nchains=2)

println("Sampling complete.")


# --- 4. Results Extraction and Visualization ---
println("\nExtracting and plotting results...")

# To get the reconstructed values of the latent covariate, we need to access
# the posterior predictions from the nested sub-model.
res = model_results_comprehensive(model_joint, chain_joint; W=W)

# The reconstructed values for the nested model are stored within the results object.
# The key matches the name given in the nested() call: :latent_x_process.
nested_results = res.nested_model_results.latent_x_process
x_predicted_mean = nested_results.pstats.predictions_denoised.mean

# Create a parity plot to compare true vs. predicted values of the censored covariate
p_parity = plot(
    title="Censored Covariate Reconstruction",
    xlabel="True Latent Value (x_true)",
    ylabel="Predicted Latent Value (Posterior Mean)",
    legend=:topleft,
    aspect_ratio=:equal
)

# Plot a 1-to-1 line for reference
plot!(p_parity, [minimum(x_true), maximum(x_true)], [minimum(x_true), maximum(x_true)],
      ls=:dash, color=:black, label="1-to-1 Line")

# Plot the points, color-coding the censored ones
scatter!(p_parity, x_true[.!is_censored], x_predicted_mean[.!is_censored],
         label="Observed Points", color=:blue, alpha=0.6)
scatter!(p_parity, x_true[is_censored], x_predicted_mean[is_censored],
         label="Censored Points", color=:red, marker=:xcross, markersize=5)

display(p_parity)

# Also, display the posterior predictive check for the main model
println("\nDisplaying posterior predictive check for the main outcome 'y'...")
display(res.plots.ppc)

println("\nExample complete. The parity plot shows the model's ability to recover the true")
println("underlying values for the censored covariate by borrowing strength from the spatial field and the main outcome.")


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
| :----------| :---------------| :-------------------------------------------------------| :-----------------------------------------------------------------------------------------------------|
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

The `predict()` function projects a fitted model onto a new data grid to generate out-of-sample predictions. It correctly handles the projection of all component types, including re-computing basis matrices for smooth terms on the new data.

```julia
preds = predict(model, chain, new_data_frame)
```

### Cross-Validation with `bstm_cv_orchestrator`

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
- Allows independent study of geographic and temporal drivers while still permitting more complex space-time interactions (e.g., Knorr-Held Type IV).

**Stationarity**:
- Processes assume constant mean/variance over a standardized [0, 1] interval.	
- Provides structural stability; ensures the "rules" of time-series (AR1) or kernels (RFF) are consistent.
- Non-stationarity is important is real systems and so we will work towards this being relaxed in later methods.  
- Non-stationarity is important in real systems and so we will work towards this being relaxed in later methods.  

**Rank-Deficiency and Identifiability**:
- Intrinsic priors (ICAR and RW2) measure differences between units, not absolute levels. This provides the mathematical basis for smoothing, though it requires constraints to achieve identifiability.
- The **Sum-to-Zero Constraint** is used to solve this. When we use intrinsic priors like the ICAR (spatial) or RW2 (temporal), the precision matrix is singular. Adding any constant $c$ to the latent field ($\mathbf{u} + c\mathbf{1}$) results in the same log-density. This Rank-Deficiency Problem means that computations cannot distinguish between a global intercept ($\alpha$) and the mean level of the spatial field.
- Using a Sum-to-Zero Constraint ($\sum u_i = 0$) "pins" the latent field to a mean of zero, so the global intercept is preserved as the true overall mean of the response. This also stabilizes computations by preventing MCMC chains from wandering.

**Spatiotemporal Interactions (Knorr-Held Classes I-IV)**:
- Four classes of space-time interactions (**Knorr-Held (2000)**) are implemented to allow the spatial effect to evolve dynamically over time or the temporal trend to vary across regions. 
- Given a spatial precision matrix $\mathbf{Q}_{sp}$ and a temporal precision matrix $\mathbf{Q}_{tm}$, the interaction effect $\delta_{at}$ is modeled as a Gaussian Markov Random Field (GMRF) with a precision matrix $\mathbf{Q}_{\delta}$ defined by the Kronecker product of the marginal precisions:
- Given a spatial precision matrix $\mathbf{Q}_{s}$ and a temporal precision matrix $\mathbf{Q}_{t}$, the interaction effect $\delta_{st}$ is modeled as a Gaussian Markov Random Field (GMRF) with a precision matrix $\mathbf{Q}_{\delta}$ defined by the Kronecker product of the marginal precisions:

$$\mathbf{Q}_{\delta} = \mathbf{Q}_{time} \otimes \mathbf{Q}_{space}$$
$$\mathbf{Q}_{\delta} = \mathbf{Q}_{t} \otimes \mathbf{Q}_{s}$$

- Interaction Class Definitions:
    *   **Class I (Unstructured):** $\mathbf{I} \otimes \mathbf{I}$ (IID noise in space and time).
    *   **Class II (Time-Structured):** $\mathbf{Q}_{tm} \otimes \mathbf{I}$ (Temporally correlated within each region, but independent across regions).
    *   **Class III (Space-Structured):** $\mathbf{I} \otimes \mathbf{Q}_{sp}$ (Spatially correlated within each time slice, but independent over time).
    *   **Class IV (Fully-Structured):** $\mathbf{Q}_{tm} \otimes \mathbf{Q}_{sp}$ (Correlated in both dimensions; a spatial pattern evolves according to a temporal process).
    *   **Type I (Unstructured):** $\mathbf{I} \otimes \mathbf{I}$ (IID noise in space and time).
    *   **Type II (Time-Structured):** $\mathbf{Q}_{t} \otimes \mathbf{I}$ (Temporally correlated within each region, but independent across regions).
    *   **Type III (Space-Structured):** $\mathbf{I} \otimes \mathbf{Q}_{s}$ (Spatially correlated within each time slice, but independent over time).
    *   **Type IV (Fully-Structured):** $\mathbf{Q}_{t} \otimes \mathbf{Q}_{s}$ (Correlated in both dimensions; a spatial pattern evolves according to a temporal process).

###  Partitioning the Map: Areal Units and Information Balance

For discrete *bstm*s, we must first discretize the spatial structure into "Areal Units" (AUs). While any partitioning will do, the choice can impact model performance. In *bstm*, several methods are available (see `bstm_overview.md` for details).

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
# would be `random(s_x, s_y, model=gp) + random(year, model=gp)`.
#----------------------------------------------------------------------------------------
println("Demonstrating a legacy dense GP model...")

# The formula specifies a Gaussian Process over the spatial coordinates (s_x, s_y)
# and another GP over the temporal coordinate (year). This creates a dense,
# separable spatiotemporal model.
m_kriging = @bstm( # Dense GP model for demonstration
    likelihood(y, family=:poisson, log_offsets=:log_offsets) ~ # Poisson likelihood with log offset
        intercept() +
        random(s_x, s_y, model=:gp, kernel="se") + # Smooth GP over spatial coordinates
        random(year, model=:gp, kernel="se"), # Temporal GP over year
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

## Appendix: Model Compendium
 
These are quick examples to show the breadth of what is possible. 

### Example data

Let us re-create the example data.

```julia
project_directory = joinpath( "C:\\", "home", "jae", "projects", "bstm")  
include( joinpath( project_directory, "startup.jl" ) ) 
load_project_functions( srcdir() )
data_scot, _ = scottish_lip_cancer_data_spacetime(); # additional noise and "fake" time slices added
data = data_scot[:data];  
W = data_scot[:au][:W]
```

### Basic Regression 

Standard regression components.


#### Likelihood Features

These examples demonstrate how to define and modify the observation model using parameters within the `likelihood()` module. Many families are supported and many options.  

#### Censored Data Model

Models a continuous outcome where some observations are censored.

```julia
y_lower_bound, y_upper_bound = 1, 100
m = @bstm(
  likelihood(y_rate, family=gaussian, censor_lower=y_lower_bound, censor_upper=y_upper_bound) ~ 
    intercept() + fixed(region),
  data );
```

#### Zero-Inflated Model

Models count data with an excess of zeros.

```julia
m = @bstm( 
  likelihood(y, family=poisson, zero_inflated=true, log_offsets=:log_offsets) ~ 
    intercept() + fixed(region), 
  data );
```

#### Hurdle model

The **Hurdle Model** (**Mullahy, 1986**) is designed for data with an excess of zeros by modeling the zero-generating process and the positive-count process separately.

A two-part model where the process for generating zeros is separate from the process for generating positive counts.


```julia
m = @bstm( 
  likelihood(y, family=poisson, hurdle=1, log_offsets=:log_offsets) ~ 
    intercept() + fixed(region), 
  data );
```



#### Stochastic Volatility Model

Models observation noise that varies over space and time.


```julia
m = @bstm( 
  likelihood(y, family=poisson, hurdle=1, log_offsets=:log_offsets, volatility=true) ~ 
    intercept() + fixed(region), 
  data );
```


### Fixed Effects Model (Linear Regression)

A simple linear regression model with an intercept and two continuous covariates.

```julia
m = @bstm( likelihood(y) ~ intercept() + fixed(cov1) + fixed(cov2), data );
```
 
#### Categorical Fixed Effects with Contrasts

Models the effect of a categorical variable using custom contrast coding.

**Key Features:**
- **Module**: `fixed()`
- **Contrast Coding**: `effects` coding sets the sum of coefficients to zero.
- **Likelihood Weights**: 'weights'  


```julia
m = @bstm( likelihood(y, weights=cov5) ~ intercept() + fixed(region, contrast=effects, prior=Normal(0, 10)), data );
```

### Mixed Effects

These are simple linear models with random (iid) slopes and or intercepts.

#### Random Intercept (Mixed) Model 

Models group-level variability in the intercept.

**Key Features:**
- **Module**: `mixed()`
- **Random Intercept**: `mixed( intercept() | group)`


```julia
m = @bstm( likelihood(y) ~ intercept(false) + fixed(cov1) + mixed( intercept() | region), data );
```

#### Random Slope and Intercept (Mixed) Model

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


### Temporal Models

 
#### Smooth Temporal Trend (Random Walk)

Models a smooth, non-linear temporal trend using a second-order random walk.


```julia
m = @bstm( likelihood(y) ~ intercept() + random(year, model=rw2), data );
```

#### Autoregressive Model (AR1)

Models a stationary temporal process where the current value depends on the immediately preceding value.


```julia
m = @bstm( likelihood(y) ~ intercept() + random(year, model=ar1), data );
```
 
#### Harmonic Seasonality

Captures periodic effects using sine and cosine basis functions.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + random(year, month, model=(ar1, cyclic), period=12), # Temporal effect with AR1 and cyclic components
    data # Input data
);
```

#### Cyclic Random Walk

Models a smooth, periodic effect where the end of the cycle connects to the beginning.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + random(day, model=harmonic ), # Temporal effect with harmonic component
    data # Input data
);
```

### Covariate Smoothing (`random(structure=:smooth)`)

#### 1D P-Spline Smoother

Models the non-linear effect of a continuous covariate using penalized splines.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + random(cov1, model=pspline, nbins=20), # Smooth effect for covariate 1
    data # Input data
);
```

#### 2D Thin Plate Spline

Models the smooth, non-linear interaction of two continuous covariates (e.g., spatial coordinates).


```julia
m = @bstm(
    likelihood(y) ~ intercept() + random(s_x, s_y, model=tps, nbins=50), # Smooth effect for spatial coordinates
    data # Input data
);
```


### Spatial Models: Areal Unit Models (GMRFs)

These models are for data aggregated over discrete spatial units (polygons).

#### BYM2 Disease Mapping Model

These examples demonstrate various models for capturing spatial autocorrelation.

The standard for areal disease mapping, decomposing spatial risk into structured and unstructured components.


```julia
m = @bstm( likelihood(y, family=poisson) ~ intercept() + random(s_idx, model=bym2, W=W), data );
```

#### ICAR / Besag Model

A model for strong spatial smoothing based on local neighbors.


```julia
m = @bstm( likelihood(y) ~ intercept() + random(s_idx, model=icar, W=W), data );
```

#### Leroux Model

Alternative to BYM2 that offer different parameterizations of spatial correlation.


```julia
m = @bstm( likelihood(y) ~ intercept() + random(s_idx, model=leroux, W=W), data );
```

#### SAR Model

Models spatial "spill-over" effects where the value at one location directly influences its neighbors.


```julia
m = @bstm( likelihood(y) ~ intercept() + random(s_idx, model=sar, W=W, noise=1e-6), data);
```



### Spatial Models: Continuous & Point-Reference Models

In models using **Random Fourier Features (RFF)**, **FITC Sparse GPs**, and **Deep GPs**, space-time dependencies are defined via a continuous covariance kernel $k(\mathbf{x}_i, \mathbf{x}_j)$ where $\mathbf{x} = [s_{lon}, s_{lat}, t]$.

To avoid $O(N^3)$ kernel inversions, we approximate the interaction using $M$ features mapped through a spectral density $p(\boldsymbol{\omega})$:
$$\phi(\mathbf{x}) = \sqrt{\frac{2}{M}} \cos(\mathbf{W}\mathbf{x} + \mathbf{b})$$
Where $\mathbf{W} \sim p(\boldsymbol{\omega})$. The interaction is then the linear product $\eta_{it} = \phi(\mathbf{x}_i)^T \boldsymbol{\beta}_{rff}$.

These models are for data where exact coordinates are available.

#### Gaussian Process (GP)

The gold-standard for continuous spatial modeling, but computationally expensive.


```julia
m = @bstm( likelihood(y) ~ intercept() + random(s_x, s_y, model=gp, kernel=matern32), data);
```

#### SPDE Model

Models a continuous spatial process using an approximation to a Stochastic Partial Differential Equation, linked to the Matérn kernel.


```julia
m = @bstm( likelihood(y) ~ intercept() + random(s_idx, model=spde, W=W), data );
```
 


### Interaction Models

#### Separable (no-interaction) Spatiotemporal Model

A standard model where the spatial and temporal effects are assumed to be independent and additive.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + random(s_idx, model=bym2, W=W) + random(year, model=ar1), # Spatial and temporal random effects
    data # Input data
);
```


#### Spatiotemporal Interaction Model (Knorr-Held Type IV)

A fully structured interaction where a spatial field (e.g., ICAR) evolves over time according to a temporal process (e.g., AR1).

**Formula Equivalent (using `⊗`):**
```julia
m = @bstm( # Spatiotemporal interaction model
    likelihood(y) ~ intercept() + random(s_idx, model=icar) + random(year, model=ar1) + # Main spatial and temporal effects
        random(s_idx, model=icar) ⊗ random(year, model=ar1), # Kronecker product interaction
    data, W=W
);
```

**Formula Equivalent (using `random(structure=:spacetime)`):**
```julia
m = @bstm( # Spatiotemporal interaction model
    likelihood(y) ~ intercept() + random(s_idx, model=icar) + random(year, model=ar1) + # Main spatial and temporal effects
        random(s_idx, year, structure=:spacetime, model=(icar, ar1)), # Spacetime interaction module
    data, W=W
);
```

#### Spatially Varying Coefficients (SVC)

Allows the effect of a covariate to vary smoothly across space.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + cov1 |> random(s_idx, model=icar, W=W), # Spatially varying coefficient for cov1
    data # Input data
);
```

#### Spatially Varying Curves

Models a non-linear trend of a covariate that varies smoothly across space.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + random(year, model=pspline) |> random(s_idx, model=icar, W=W), # Spatially varying curve for year
    data # Input data
);
```


### Spatially Varying Autoregressive (SVAR) Model

This model allows the temporal autoregressive parameter `rho` to vary across space. The spatial variation of `rho` is modeled by an `icar` component. It is similar to the SAR model except it is for point-level dynamics where temporal persistence varies by region.

```julia

# Prepare data with spatiotemporal index
data.st_idx = [(t-1)*30 + s for (s, t) in zip(data.s_idx, data.year)]

m = @bstm(
   likelihood(y_rate) ~ 
   intercept() +
   random(s_idx, structure=:svar, model=icar), # Spatially varying AR parameter 
   data, W=W
);
```

### Threshold Autoregressive (TAR) Models

This model switches between two AR(1) regimes based on a covariate's value, where the temporal dynamics of y_rate switches based on whether `price_index` is above or below a learned threshold.

```julia
# Prepare data for TAR example
data.price_index = 5.0 .+ cumsum(randn(nrow(data)) .* 0.1)

m = @bstm(
    likelihood(y_rate) ~ 
    intercept() +
    random(year, model=tar, threshold_var=price_index), # Temporal TAR effect
    data
);
```


### Log-Gaussian Cox Process (LGCP) Model

LGCP models aggregated counts. We generate a smooth latent intensity on a grid and sample Poisson counts to simulate point-pattern data. The counts are modeled directly via a custom log-probability calculation, overriding the standard likelihood. The syntax uses the composition operator `∘` to combine a `pointprocess` module with a latent `random` field.
A Log-Gaussian Cox Process (LGCP) models point patterns by assuming the *logarithm* of the intensity function is a realization of a Gaussian Process. This is the most common point process model in spatial statistics. The syntax uses the composition operator `∘` to combine a `pointprocess` module with a latent `random` field.

```julia
m = @bstm(
    likelihood(y, family=:poisson) ~
        intercept() +
        (pointprocess(model=:lgcp, grid_areas=1.0) ∘ random(s_idx, model=:icar)),
    data,
    W = W
);
```

#### Regular lattice (surface area is constant) 
```julia
lgcp_data, grid_W, total_cells = generate_lgcp_synthetic_data_regular(10)
display(first(lgcp_data, 5))

m = @bstm(
    likelihood(counts, family=poisson) ~ 
    intercept() +  
    (pointprocess(model=:lgcp, grid_areas=ones(total_cells)) ∘ random(s_idx, model=:icar, sigma=Exponential(0.5))), # LGCP Composition  
    lgcp_data, 
    W = grid_W, 
    s_N = total_cells
);

```

#### Irregular areal units
Instantiate Model specifying the area column when areal representations are variable. Note: we pass grid_areas to the lgcp parameters

```julia
irreg_df, irr_W, n_units, cell_areas = generate_irregular_lgcp_data(10)
 
model_irreg = @bstm(
    likelihood(counts, family=poisson) ~ 
    intercept() + # Intercept
    random(s_idx, point_process=lgcp, model=icar, sigma=Exponential(1.0), grid_areas=cell_areas), # LGCP spatial effect
    irreg_df, 
    W = irr_W, 
    s_N = n_units
);
```


### Demonstration of Kriging implementation

```julia 
m = @bstm(
    likelihood(y_rate) ~  
    intercept() + # Intercept
    random(s_x, s_y, model=kriging, lengthscale=InverseGamma(3, 3), sigma=Exponential(1.0)), # Kriging smooth effect
    inp_df
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

m = @bstm(
    likelihood(y, family=:poisson, log_offsets=:log_offsets) ~
        intercept(prior=Normal(0, 10)) +
        fixed(X, prior=Normal(0, 5)) + # Fixed effect for covariate X
        random(s_idx, model=:bym2) + # Spatial random effect
        random(year, model=:ar1), # Temporal random effect
    inp_df;
    W = W
);

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
# The `random(...) ⊗ random(...)` syntax specifies a Kronecker product interaction,
# allowing the spatial field to evolve over time. This is a Type IV interaction,
# the most complex type, where both space and time are structured.
#----------------------------------------------------------------------------------------
m = @bstm(
    likelihood(y, family=:poisson, log_offsets=:log_offsets) ~
        intercept() +
        fixed(X) + # Fixed effect for covariate X
        random(s_idx, model=:bym2) + # Spatial random effect
        random(year, model=:ar1) + # Temporal random effect
        (random(s_idx, model=:besag) ⊗ random(year, model=:ar1)), # Spatiotemporal interaction
    inp_df;
    W = W
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
    likelihood(y, family=:poisson, log_offsets=:log_offsets) ~
        intercept() + fixed(X) +
        random(s_idx, model=:leroux) + # Leroux spatial effect
        random(year, model=:ar1), # Temporal AR1 effect
    inp_df, W = W
);
```


### Simultaneous Autoregressive (SAR) model

The **Simultaneous Autoregressive (SAR)** model specifies the joint dependency directly, where the spatial field is equal to a weighted average of its neighbors plus independent noise. Its precision matrix is $Q_{SAR} = \frac{1}{\sigma^2} (I - \rho W)'(I - \rho W)$. This approach was formalized in early spatial statistics literature (**Cliff & Ord, 1973**).

SAR Models spatial "spill-over" effects directly.


```julia
m = @bstm(
    likelihood(y, family=:poisson, log_offsets=:log_offsets) ~
        intercept() + fixed(X) +
        random(s_idx, model=:sar) + # SAR spatial effect
        random(year, model=:ar1), # Temporal AR1 effect
    inp_df, W = W
);

``` 

### Spatially Varying Coefficient (SVC) Model 

The **Spatially Varying Coefficient (SVC)** (Gelfand et al., 2003) model relaxes the assumption of global stationarity in regression effects. Instead of a single $\beta$ for the whole domain, each area $i$ has its own coefficient $\beta_{i,k}$. The log-intensity for area $i$ at time $t$ becomes:
$$\log(\mu_{it}) = \text{offset}_{it} + \phi_i + \delta_t + \gamma_{it} + \sum_{k=1}^K x_{it,k} \beta_{i,k}$$

An SVC model allows the effect of a covariate to vary across space. Here, the effect of the covariate `X` is modulated by a spatial field (ICAR model).

```julia
m = @bstm(
    likelihood(y, family=:poisson, log_offsets=:log_offsets) ~
        intercept() +
        cov1 |> random(s_idx, model=icar, W = W) + # Spatially varying coefficient for X
        random(year, model=ar1), # Temporal AR1 effect
    inp_df
);
```
 

#### Hurdle Model (spatiotemporal no-interaction):

```julia
m = @bstm(
    likelihood(y, family=:poisson, log_offsets=:log_offsets, hurdle=0) ~
        intercept() + fixed(cov1) +
        random(s_idx, model=:bym2) +
        random(year, model=:ar1),
    inp_df, W = W
);
```


### Directed Acyclic Graph (DAG) Model

The `dag` model is a spatial component for modeling directed dependencies,
which is useful for causal inference or processes with a clear directional
flow (e.g., river networks). It requires an adjacency matrix `W` that
represents a directed acyclic graph. For a valid DAG, this matrix should be
strictly lower or upper triangular. For this example, we use the symmetric
adjacency matrix for demonstration purposes, but a correctly specified DAG
would require a custom-built `W`.

```julia
m_dag = @bstm(
    likelihood(y, family=poisson, log_offsets=:log_offsets) ~
        intercept() +
        random(s_idx, model=:dag),
    inp_df,
    W = W
);
```


###  Local Adaptive Model 

The `localadaptive` model is a spatial component that combines a Leroux-style
precision structure with cluster-specific means. This allows the model to
capture both smooth spatial correlation and abrupt shifts between regions.
It requires spatial coordinates (e.g., `s_x`, `s_y` in the data) to perform
k-means clustering to define the regions. The number of clusters is controlled
by the `n_clusters` parameter.

Note: This example assumes `inp_df` contains `s_x` and `s_y` columns for clustering.

```julia

data_scot, _ = scottish_lip_cancer_data_spacetime()
inp_df = data_scot.data
W = data_scot.au.W

m_localadaptive = @bstm(
    likelihood(y, family=poisson, log_offsets=:log_offsets) ~
        intercept() +
        random(s_idx, model=:localadaptive, n_clusters=5),
    inp_df,
    W = W
);
```


The fundamental mathematical difference between the localadaptive model and the new hierarchical mosaic model lies in their structural composition and how they achieve non-stationarity.

The localadaptive model is a single, non-stationary random field. It modifies a standard Gaussian Markov Random Field (GMRF) by giving it a non-zero mean that is piecewise constant across different spatial regions.
The hierarchical mosaic model is an additive decomposition into two distinct random fields: one that captures the abrupt, piecewise-constant shifts between regions, and another that captures a smooth, continuous global trend across the entire domain.
Here is a more detailed breakdown of their mathematical differences:

1. Model Structure and Formulation
localadaptive Model
The localadaptive model defines a single latent spatial field, let's call it $\phi$, for all $s_N$ spatial locations. This field is assumed to follow a multivariate normal distribution with a non-zero mean vector $\boldsymbol{\mu}$ and a global precision matrix $\mathbf{Q}$.

$\qquad \boldsymbol{\phi} \sim \text{MvNormal}(\boldsymbol{\mu}, \mathbf{Q}^{-1})$

Mean Structure ($\boldsymbol{\mu}$): The domain is first partitioned into $k$ clusters (regions) using k-means. A separate mean effect, $\mu_g$, is estimated for each region $g \in {1, \dots, k}$. The mean vector $\boldsymbol{\mu}$ is then constructed such that for any spatial unit $i$ belonging to region $g$, its mean is $\mu_i = \mu_g$. This makes the mean of the process non-stationary and discontinuous at the region boundaries.

Precision Structure ($\mathbf{Q}$): The precision matrix $\mathbf{Q}$ is a global, stationary structure that defines the spatial correlation across the entire domain, irrespective of the clusters. It is typically a Leroux-style precision matrix: $\qquad \mathbf{Q} = \tau \left( (1-\rho)\mathbf{I} + \rho\mathbf{Q}{\text{ICAR}} \right)$ where $\mathbf{Q}{\text{ICAR}}$ is the intrinsic CAR precision matrix based on the neighborhood graph. This single precision matrix encourages smoothness across the whole field, including across the boundaries of the mean-shift clusters.

In essence, localadaptive models a single, spatially correlated process whose baseline level jumps at region boundaries.

Hierarchical mosaic Model
The new mosaic model is not a standalone model type but a hierarchical flag that decomposes the total spatial effect, $\boldsymbol{\psi}$, into two independent, additive components: a regional (mosaic) effect $\boldsymbol{\phi}{\text{mosaic}}$ and a global smooth effect $\boldsymbol{\phi}{\text{global}}$.

$\qquad \boldsymbol{\psi} = \boldsymbol{\phi}{\text{mosaic}} + \boldsymbol{\phi}{\text{global}}$

Mosaic Component ($\boldsymbol{\phi}_{\text{mosaic}}$): This component captures the distinct mean level of each region. It is modeled as an IID (independent and identically distributed) random effect on the region assignments. If a spatial unit $i$ belongs to region $g$, its effect is $\phi_{\text{mosaic}, i} = \mu_g$, where each $\mu_g$ is sampled independently: $\qquad \mu_g \sim \text{Normal}(0, \sigma^2_{\text{mosaic}})$ This component is a pure step-function; it is constant within regions and discontinuous at the boundaries, with no inherent spatial correlation.

Global Component ($\boldsymbol{\phi}_{\text{global}}$): This is a standard, zero-centered GMRF (e.g., bym2, icar) that applies to the entire spatial domain. It captures the smooth spatial variation that remains after accounting for the regional mean shifts. $\qquad \boldsymbol{\phi}{\text{global}} \sim \text{MvNormal}(\mathbf{0}, \mathbf{Q}{\text{base}}^{-1})$ where $\mathbf{Q}_{\text{base}}$ is the precision matrix of the user-specified base model (like bym2).

This additive structure explicitly separates the non-stationary mean shifts from the stationary, smooth spatial correlation.

2. Interpretation and Continuity
Feature	localadaptive Model	Hierarchical mosaic Model
Composition	A single random field with a non-stationary mean.	The sum of two independent random fields: one for regional means and one for a global smooth trend.
Effect	The effect at a location is a single value drawn from a non-zero mean GMRF.	The effect at a location is the sum of its regional mean effect and its value from the global smooth field.
Continuity	The mean is discontinuous at region boundaries. The realized field has some smoothness across boundaries due to the global precision matrix $\mathbf{Q}$, but the primary feature is the mean shift.	The regional mean component is explicitly discontinuous. The global trend component is explicitly continuous. The sum of the two creates a surface that has sharp "jumps" at boundaries but is smoothly connected by the underlying global field, directly addressing the goal of having "smoothing between group edges".
Interpretability	Estimates a single, complex spatial field. It is difficult to disentangle how much of the effect is due to the regional mean versus the spatial correlation.	Provides a clear decomposition. The mosaic component estimates the magnitude of the mean difference between regions, while the global component estimates the residual, smooth spatial pattern common to all regions.
In summary, the localadaptive model creates a single process with a spatially varying mean, whereas the hierarchical mosaic approach models the spatial effect as a superposition of a discontinuous regional process and a continuous global process. This makes the mosaic model more interpretable and better aligned with the objective of modeling distinct regions that are part of a larger, smoothly varying spatial system.


### Mosaics

```julia 

# Set a seed for reproducibility
Random.seed!(2025)

# --- 1. Data Generation for a Hierarchical Spatial Problem ---
println("Generating synthetic data with a global trend and regional means...")

# Define a spatial grid
grid_dim = 25
s_N = grid_dim * grid_dim # Total number of spatial locations
s_coords_x = repeat(1:grid_dim, inner=grid_dim)
s_coords_y = repeat(1:grid_dim, outer=grid_dim)

# Create a simple adjacency matrix 'W' for the BYM2 component
W = spzeros(Int, s_N, s_N)
for i in 1:s_N
    for j in (i+1):s_N
        dist = sqrt((s_coords_x[i] - s_coords_x[j])^2 + (s_coords_y[i] - s_coords_y[j])^2)
        if dist <= 1.5 # Queen contiguity
            W[i, j] = 1
            W[j, i] = 1
        end
    end
end

# a. Define a true, smooth global spatial trend
true_global_trend(x, y) = 2.0 * sin(x / grid_dim * 2pi) * cos(y / grid_dim * 2pi)
global_field = [true_global_trend(s_coords_x[i], s_coords_y[i]) for i in 1:s_N]

# b. Define three true, distinct spatial regions (mosaics)
true_regions = ones(Int, s_N)
true_regions[s_coords_x .> 12] .= 2
true_regions[(s_coords_x .<= 12) .& (s_coords_y .> 12)] .= 3
n_regions_true = length(unique(true_regions))

# c. Define a different mean level for each region
region_means = Dict(1 => -1.5, 2 => 2.0, 3 => 0.5)
regional_field = [region_means[r] for r in true_regions]

# d. Combine fields and simulate observations with noise
true_latent_field = global_field .+ regional_field
y_obs = true_latent_field .+ rand(Normal(0, 0.3), s_N)

# e. Assemble the final DataFrame
# The model needs s_x and s_y to perform k-means clustering.
df = DataFrame(
    y = y_obs,
    s_idx = 1:s_N,
    s_x = s_coords_x,
    s_y = s_coords_y
)

println("Data generation complete. $s_N observations across $n_regions_true true regions.")


# --- 2. Model Definition ---
println("\nDefining the hierarchical mosaic model using the new syntax...")

# The formula specifies a `bym2` random effect modified by the `mosaic` keyword.
# This tells the processor to create two components:
# 1. A smooth `bym2` field for the global trend.
# 2. An `iid` field for the regional means identified by k-means.
model_hierarchical = @bstm(
    likelihood(y, family=gaussian) ~
        intercept() +
        random(s_idx, model=bym2, mosaic=:kmeans, n_regions=n_regions_true),
    df,
    W = W,
    verbose = false
);

println("Model definition complete.")
# You can inspect the components via `model_hierarchical.args.M.components`
# It will contain two components: `s_idx` (for bym2) and `mosaic_s_idx` (for the regions).


# --- 3. MCMC Inference ---
println("\nRunning MCMC sampling...")
chain_hierarchical = bstm_sample_nowarn(model_hierarchical, NUTS(0.65), 1000; progress=true)
println("Sampling complete.")


# --- 4. Results Extraction and Visualization ---
println("\nExtracting and plotting decomposed effects...")

# Reconstruct the model posteriors.
res = model_results_comprehensive(model_hierarchical, chain_hierarchical)

# Extract the mean posterior effects for each component
# The mosaic effect is automatically named `mosaic_<original_key>`
bym2_effect = res.pstats.effects.s_idx.mean
mosaic_effect = res.pstats.effects.mosaic_s_idx.mean
intercept_effect = res.pstats.effects.intercept.mean
total_predicted_effect = bym2_effect .+ mosaic_effect .+ intercept_effect

# Reshape fields for heatmap plotting
true_global_grid = reshape(global_field, grid_dim, grid_dim)
true_regional_grid = reshape(regional_field, grid_dim, grid_dim)
est_global_grid = reshape(bym2_effect, grid_dim, grid_dim)
est_regional_grid = reshape(mosaic_effect, grid_dim, grid_dim)
est_total_grid = reshape(total_predicted_effect, grid_dim, grid_dim)

# Create the final plot with 4 panels
p = plot(layout=(2, 3), size=(1500, 900), link=:all)

# Row 1: True Fields
heatmap!(p[1], 1:grid_dim, 1:grid_dim, true_global_grid', c=:viridis, title="True Global Trend")
heatmap!(p[2], 1:grid_dim, 1:grid_dim, true_regional_grid', c=:plasma, title="True Regional Means")
heatmap!(p[3], 1:grid_dim, 1:grid_dim, (true_global_grid .+ true_regional_grid)', c=:inferno, title="True Total Field")

# Row 2: Estimated Fields
heatmap!(p[4], 1:grid_dim, 1:grid_dim, est_global_grid', c=:viridis, title="Estimated Global Trend (BYM2)")
heatmap!(p[5], 1:grid_dim, 1:grid_dim, est_regional_grid', c=:plasma, title="Estimated Regional Means (Mosaic)")
heatmap!(p[6], 1:grid_dim, 1:grid_dim, est_total_grid', c=:inferno, title="Estimated Total Field")

display(p)

println("\nExample complete. The plots show the model successfully decomposing the data into a smooth global trend and discontinuous regional means.")



```



### Multivariate Models

#### Bayesian PCA (`eigen`)

Performs dimensionality reduction on a set of covariates, using the dominant latent factor as a predictor.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + eigen(cov1, cov2, cov3, n_factors=2),
    data
);
```

#### Multivariate CAR

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
        fixed(X) + # Fixed effect for covariate X
        random(s_idx, model=:besag) + # A shared spatial effect
        random(year, model=:ar1),    # A shared temporal effect
    inp_df_multi, W = W
);
```
 
#### Joint Model with Different Likelihoods

Jointly models multiple outcomes where each has a different likelihood.


```julia
m = @bstm(
    likelihood(y, family=poisson) + likelihood(y_rate, family=gaussian) ~ # Joint likelihoods
        intercept() + random(s_idx, model=bym2, W=W), # Shared spatial effect
    data
);
```

```julia
# Jointly modeling Gaussian and Poisson outcomes with spatial correlation
# Uses continuous 2D Thin Plate Spline for space and RW2 for time
m = @bstm(
    likelihood(y_rate, family=gaussian) + likelihood(y, family=poisson, log_offsets=log_offsets) ~ # Joint likelihoods
    intercept() + # Intercept
    random(s_x, s_y, model=tps, nbins=30) + # Smooth effect for spatial coordinates
    random(year, model=rw2), # Temporal random walk
    data
);
```

 
#### Multinomial (Compositional) Model

Modeling counts across multiple categories using Dirichlet-Multinomial. Note: Multi-column LHS targets the Dirichlet-Multinomial kernel

```julia

df_dmm, W_dmm = generate_dirichlet_multinomial_data();

m = @bstm(
    likelihood(cat_1 + cat_2 + cat_3, family=:dirichlet_multinomial) ~
        intercept() + 
        random(s_idx, model=:bym2),
    df_dmm,
    W = W_dmm,
    verbose = false # Suppress model code printing for cleaner output
);

```

#### Proportional odds ratio model

```julia
# Define the Proportional Odds Ratio Model using @bstm
# This model uses a three-level ordinal outcome variable 'ordinal_y'.
# It includes:
# - An intercept
# - A fixed effect for 'cov1'
# - A random intercept for 'group_id'
# The 'family=:ordinal' automatically handles the cut-points and likelihood.
# note: latent_dist: :normal is probit link; :logistic is logit; :student_t/ordinal_df (t degrees of freedom parameter) 

inp_ord = generate_ordinal_data(); # y_ordinal is a categorical()
   
m_ordinal = @bstm(
    likelihood(ordinal_y, family=:ordinal, latent_dist=:normal) ~
        intercept() +
        fixed(cov1) +
        mixed(intercept() | group_id), # Random intercept for group_id
    inp_ord
);
 
```

#### Nonproportional odds ratio model

```julia

inp_ord = generate_ordinal_data();  # y_ordinal is a categorical()

m_ordinal = @bstm(
    likelihood(ordinal_y, family=:ordinal, latent_dist=:normal) ~
        intercept() +
        fixed(cov1, non_proportional_effects=true) +
        fixed(cov2, non_proportional_effects=false) +
        fixed(cov3) +
        mixed(intercept() | group_id), # Random intercept for group_id
    inp_ord
);

```

### Multi-fidelity Model (`nested`)

Integrates a low-fidelity (but data-rich) proxy variable to improve predictions for a high-fidelity (but data-sparse) target.

```julia
m = @bstm(
    likelihood(y) ~ intercept() + 
        nested( # Nested sub-model
            proxy_model, # Name of the proxy sub-model
            formula=likelihood(y_bin, family=binomial) ~ intercept() + random(cov3, model=pspline) # Formula for the sub-model
        ),
    data
);
```
 

## Multifidelity (Nested) Model

```julia
# High-fidelity Gaussian outcome aided by a low-fidelity proxy sub-model

model_mf = @bstm(
    likelihood(y_rate, family=gaussian) ~ 
    intercept() + # Intercept
    random(year, model=ar1) + # Temporal AR1 effect
    nested(
        proxy_submodel,
        formula = "likelihood(cov2, family=gaussian) ~ intercept() + random(s_x, s_y, model=tps, nbins=20)"
    ),
    data
);
```

 


### Mechanistic Models

#### Dynamics (Advection-Diffusion)

A mechanistic model for a process that is transported (advection) and spreads (diffusion) over a graph.

```julia
m = @bstm(
    likelihood(y) ~ intercept() + dynamics(s_idx, year, model=advection_diffusion, W=W),
    data
);
```

#### Biological models
 
```julia

# 1. Logistic Basic

df_logistic, W_lb, ga_lb = generate_logistic_data()
m = @bstm(
    likelihood(y, family=poisson) ~
        intercept() +
        dynamics(s_idx, year, model=logistic, r=LogNormal(0, 0.5), K=LogNormal(log(100.0), 0.5)),
    df_logistic,
    W = W_lb,
    grid_areas = ga_lb
);

# 

# 2. Logistic Exploitation

df_logistic_exploitation, W_le, ga_le = generate_logistic_exploitation_data()
m = @bstm(
    likelihood(y, family=poisson) ~
        intercept() +
        dynamics(s_idx, year, model=logistic_exploitation, r=LogNormal(0, 0.5), K=LogNormal(log(100.0), 0.5), q=LogNormal(-2, 0.5), effort=df_logistic_exploitation.effort),
    df_logistic_exploitation,
    W = W_le,
    grid_areas = ga_le
);

# 
# 3. Delay Difference

df_delay_difference, W_dd, ga_dd = generate_delay_difference_data()
m = @bstm(
    likelihood(y, family=poisson) ~
        intercept() +
        dynamics(s_idx, year, model=delay_difference, r=LogNormal(0, 0.5), K=LogNormal(log(100.0), 0.5), M_nat=LogNormal(-1, 0.2), catch_data_col=:catch_data),
    df_delay_difference,
    W = W_dd,
    grid_areas = ga_dd
);

# 
# 4. Lotka-Volterra

df_lotka_volterra, W_lv, ga_lv = generate_lotka_volterra_data()
m_lotka_volterra = @bstm(
    likelihood(y, family=poisson) ~
        intercept() +
        dynamics(s_idx, year, model=lotka_volterra, alpha=LogNormal(0, 0.5), beta=LogNormal(-2, 0.5), gamma=LogNormal(-2, 0.5), delta=LogNormal(0, 0.5), output_species=:prey, interaction_covariate=:predator_pop),
    df_lotka_volterra,
    W = W_lv,
    grid_areas = ga_lv
);


# 5. Generalized Leslie

# Generate synthetic data for the generalized Leslie matrix model
df_glm, W_glm, ga_glm, n_classes_glm = generate_generalized_leslie_matrix_data(
    s_N=10, 
    t_N=10, 
    n_classes=4, 
    use_effort=true 
); 
 
display(first(df_glm, 5))

m = @bstm(
    likelihood(class_1 + class_2 + class_3 + class_4) ~
        intercept() +
        dynamics(s_idx, year, 
            model=generalized_leslie_matrix, 
            n_classes=n_classes_glm,
            spatially_varying_K=true,
            effort=:effort,
            K=LogNormal(log(100.0), 0.5), # Prior for mean K
            q_effort=filldist(LogNormal(-4, 1), n_classes_glm) # Prior for catchability
        ),
    df_glm,
    W = W_glm,
    grid_areas = ga_glm 
);
 
param_subset = [
    "A_flat_dynamics", "A_flat_dynamics", "A_flat_dynamics", 
    "log_K_mean_dynamics", "sigma_K_dynamics", 
    "q_effort", "q_effort"
]
display(chain_glm[param_subset])

# Further analysis could involve reconstructing the full population trajectories
# for each class using the `model_results_comprehensive` function.
# res_glm = model_results_comprehensive(m_glm, chain_glm);
# 

# 6. Spatially varying K logistic
 
# The model structure is defined using the @bstm macro as follows:
# - likelihood(y, family=poisson): The outcome 'y' (population counts) is modeled
#   with a Poisson distribution.
# - intercept(): A global intercept term representing the baseline log-population level.
# - dynamics(...): The core mechanistic component.
#   - model=logistic, spatially_varying_K=true: Specifies the logistic growth model where K is a spatial field.
#   - r=LogNormal(0, 0.5): A prior on the intrinsic growth rate 'r'.
#   - log_K_mean=Normal(log(150.0), 0.5): A prior on the mean of the logarithm of the
#     spatially varying carrying capacity. This sets our prior belief for the average K
#     across the domain to be around 150.
#   - sigma_K=Exponential(1.0): A prior on the standard deviation of the spatial field for K.
#     This controls how much K is expected to vary across space.

 
df_spatial_K, W_sk, ga_sk = generate_logistic_spatial_K_data();

m_spatial_K = @bstm(
    likelihood(y, family=poisson) ~
        intercept() +
        dynamics(s_idx, year,
            model=logistic,
            spatially_varying_K=true,
            spatially_varying_r=true,
            r=LogNormal(0, 0.5),
            log_K_mean=Normal(log(150.0), 0.5),
            sigma_K=Exponential(1.0)
        ),
    df_spatial_K,
    W = W_sk,
    grid_areas = ga_sk,
    verbose = true # Set to false to suppress detailed model code and prior check output
);

# 
# 7. Leslie matrix

# 1. Generate synthetic data with 3 age classes
df_leslie_matrix, W_lm, ga_lm, n_ac_lm = generate_leslie_matrix_data(n_age_classes=3)

# 2. Define the multivariate model
# The outcomes `age_1`, `age_2`, `age_3` correspond to the age classes.
m_leslie_matrix = @bstm(
    likelihood(age_1 + age_2 + age_3) ~
        intercept() +
        dynamics(s_idx, year, model=leslie_matrix, n_age_classes=n_ac_lm),
    df_leslie_matrix,
    W = W_lm,
    grid_areas = ga_lm
);

# 3. Sample from the model
chain_leslie_matrix = sample(m_leslie_matrix, NUTS(), 500; progress=true)

# 4. Display results
# The chain will contain posterior samples for `survival_rates` and `fecundity_rates`.
display(chain_leslie_matrix)



# Delay difference

# Ensure the project environment is set up and functions are loaded.
# This typically involves running a 'startup.jl' or similar script.
# For this example, we assume the necessary functions like
# `generate_delay_difference_data` are available.

# 1. Generate synthetic data for a multivariate delay-difference model with effort data.
#    The `use_effort=true` flag ensures the function generates an 'effort' column
#    instead of a 'catch_data' column, which is appropriate for this model variant.
println("Generating synthetic data for a multivariate delay-difference model with effort...")
df_dd_effort, W_dd, ga_dd = generate_delay_difference_data(use_effort=true)
println("Data generated successfully.")

# 2. Define and instantiate the bstm model
# This example specifies a multivariate delay-difference model where catch is not directly
# observed but is instead modeled as a function of fishing effort and an unknown catchability
# coefficient 'q'.
#
# The model structure is defined using the @bstm macro as follows:
# - likelihood(y + recruitment, family=poisson): The two outcomes, total population ('y')
#   and new recruits ('recruitment'), are jointly modeled, both assuming a Poisson distribution.
# - intercept(): A global intercept term for both outcomes.
# - dynamics(...): The core mechanistic component.
#   - model=delay_difference: Specifies the delay-difference dynamics. Because two outcomes
#     are provided in the likelihood, the multivariate version is automatically triggered.
#   - effort_col=:effort: This crucial parameter tells the model to use the 'effort'
#     column from the dataframe to calculate catch internally.
#   - q=LogNormal(-4, 0.5): A prior on the catchability coefficient 'q'.
#   - r, K, M_nat: Priors for the intrinsic growth rate, carrying capacity, and natural mortality.

println("\nDefining the multivariate delay-difference model with effort...")
m_dd_effort = @bstm(
    likelihood(y + recruitment, family=poisson) ~
        intercept() +
        dynamics(s_idx, year,
            model=delay_difference,
            effort_col=:effort,
            q=LogNormal(-4, 0.5),
            r=LogNormal(0, 0.5),
            K=LogNormal(log(150.0), 0.5),
            M_nat=LogNormal(-1.5, 0.5)
        ),
    df_dd_effort,
    W = W_dd,
    grid_areas = ga_dd,
    verbose = true # Set to false to suppress detailed model code and prior check output
);
# 
# 3. Sample from the model
# For a quick test, we can use a simple sampler. For a full analysis,
# a more advanced sampler like NUTS is recommended.
println("\nRunning a short MCMC chain for demonstration...")
chain_dd_effort = sample(m_dd_effort, MH(), 500; progress=true)

# 4. Display results
# The resulting chain object can be summarized to inspect the posterior distributions
# of the parameters, including the estimated catchability 'q'.
println("\nSampling complete. Displaying summary of the MCMC chain:")
display(chain_dd_effort)



```





### Custom Model Component

The `custom()` module provides an "escape hatch" for users to inject arbitrary
Turing.jl model code directly into the `bstm` framework. The user provides
this code as a string to the `code_fragment` parameter.

This example demonstrates how to add a simple, non-spatial random effect for
the 'cov1' variable. The code within the string defines the prior for the
effect's standard deviation (`my_effect_sigma`), samples the raw innovations
(`my_effect_raw`), and adds the final scaled effect to the linear predictor `eta`.
# 
The user can access the main model configuration object `M` within the fragment.

```julia
custom_code = """
    # --- Custom Code for 'cov1' Random Effect ---

    # 1. Define the prior for the standard deviation of the custom effect.
    my_effect_sigma ~ Exponential(1.0)

    # 2. Sample the raw (unscaled) innovations from a standard normal.
    #    The length must match the number of observations.
    my_effect_raw ~ MvNormal(zeros(T, M.y_N), 1.0)

    # 3. Construct the final effect and add it to the linear predictor 'eta'.
    #    Here, the effect is the product of the raw innovations, the learned
    #    standard deviation, and the covariate values.
    my_effect = my_effect_raw * my_effect_sigma
    eta .+= my_effect .* M.data[!, :cov1]
"""

m = @bstm(
    likelihood(y) ~ intercept() + custom(code_fragment=custom_code),
    inp_df
);

```


## Conclusion

The `bstm` framework provides a Julia-native environment for spatiotemporal modeling that emphasizes composability and extensibility. By treating latent geometries as distinct, combinable entities, it allows for the construction of complex models that remain computationally tractable. The standardized use of PC-Priors offers a principled way to maintain identifiability, while the modular formula interface facilitates model specification and interpretation.


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
