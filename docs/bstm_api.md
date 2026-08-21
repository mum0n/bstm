---
title: "BSTM Technical API Reference"
format: html
---

# BSTM Technical API Reference

This document provides a comprehensive technical reference for the internal architecture, APIs, components, and extension points of the `bstm` framework. It covers the core modeling macro `@bstm`, formula parsing engine, configuration pipeline, `ComponentModel` interface, Parameter Registry system, mathematical component formulations, inference engines, posterior reconstruction, plotting subsystem, spatial partitioning infrastructure, and two-tier persistence subsystem.

---

## 1. Framework Architecture & End-to-End Workflow

The `bstm` framework translates high-level domain formulas into optimized, differentiable Turing.jl probabilistic programs, executes MCMC sampling or variational inference, and reconstructs structured posterior effects:

```
                               ┌────────────────────────────────────────────────────────┐
                               │  User Model Call: @bstm(formula, data, kwargs...)      │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Formula Parser: decompose_bstm_formula                │
                               │  (LHS Likelihood & RHS AST Module Categorization)      │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Configuration Engine: bstm_config                     │
                               │  (Precomputes, Dimensions, Hyperpriors, Registry)      │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Turing Code Generator: bstm_text_assembler            │
                               │  (Dynamic @model generation with AD-safe constructors) │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Inference Engine: get_optimal_sampler / bstm_sample   │
                               │  (Automatic Gibbs block-partitioning & NUTS sampling)  │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Posterior Reconstruction: model_results_comprehensive│
                               │  (Latent discovery, ParamRegistry lookup, bstm_plots)  │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Persistence & SQL Analytics: save_bstm_bundle / DuckDB│
                               │  (JLD2 live models, DuckDB tables, GeoJSON, Parquet)   │
                               └────────────────────────────────────────────────────────┘
```

---

## 2. The `@bstm` Macro & Configuration Engine

### 2.1. Macro Signature & Top-Level Options

```julia
m = @bstm(formula, data; W=nothing, au=nothing, verbose=false, kwargs...)
```

| Argument / Keyword | Type | Default | Description & Implications |
| :--- | :--- | :--- | :--- |
| `formula` | `Expr` | *Required* | Model formula specifying observation likelihood on the LHS and latent additive components on the RHS (e.g. `likelihood(y) ~ intercept() + fixed(x) + random(s, model=bym2)`). |
| `data` | `DataFrame` | *Required* | Tabular dataset containing outcome variables, covariates, spatial indices, coordinates, and temporal stamps. |
| `W` | `AbstractMatrix` | `nothing` | Spatial adjacency matrix ($S \times S$) required for discrete GMRF models (`bym2`, `icar`, `besag`, `leroux`, `sar`, etc.). |
| `au` | `NamedTuple` / `Any` | `nothing` | Areal units object from `assign_spatial_units` containing polygon boundaries, centroids, and neighborhood graphs. |
| `verbose` | `Bool` | `false` | When `true`, prints generated Turing model code, compilation metadata, and automatic prior predictive check summaries. |
| `prior_scheme` | `Symbol` | `:pcpriors` | Global prior default scheme: `:pcpriors` (Penalized Complexity), `:informative`, `:uninformative`. |
| `priors` | `Dict{Symbol, Any}` | `Dict()` | Explicit prior overrides mapped by parameter name or component key. |
| `init_params` | `Dict{Symbol, Any}` | `nothing` | Custom initial parameter values for MCMC chains or optimization routines. |
| `sampler_choice` | `Symbol` | `:auto` | Default sampler selection: `:auto` (triggers `get_optimal_sampler`), `:nuts`, `:hmc`, `:mh`, `:slice`, `:variational`. |
| `adtype` | `ADTypes.AbstractADType` | `AutoForwardDiff()` | Automatic differentiation backend passed to Turing (`AutoForwardDiff()`, `AutoReverseDiff()`, `AutoZygote()`). |

---

### 2.2. Left-Hand Side (LHS) Likelihood Options (`likelihood()`)

The LHS `likelihood(outcome, ...)` defines the observation likelihood distribution and observational noise structure:

| Parameter | Type | Default | Description & Mathematical Role |
| :--- | :--- | :--- | :--- |
| `family` | `Symbol` | `:gaussian` | Likelihood distribution family (19 families supported). |
| `log_offsets` | `Symbol` / `Vector` | `nothing` | Additive offset on the linear predictor link scale: $\eta' = \eta + \text{offset}$. Essential for modeling rates (e.g., $\log(\text{Expected})$ in Poisson models). |
| `weights` | `Symbol` / `Vector` | `nothing` | Observation-level log-likelihood weighting: $\ell_i(\theta) = w_i \cdot \log p(y_i \mid \eta_i)$. |
| `trials` | `Symbol` / `Vector` | `nothing` | Number of binomial trials $N_i$ for `:binomial` and `:betabinomial` families. |
| `zero_inflated` | `Bool` | `false` | Enables structural zero inflation: $p(y=0) = \pi + (1-\pi)p_0$, $p(y>0) = (1-\pi)p(y)$. |
| `hurdle` | `Real` / `Bool` | `false` | Truncates likelihood below the hurdle threshold and models binary passage independently. |
| `censor_lower` | `Symbol` / `Vector` | `nothing` | Left-censoring bound: observation is known only to satisfy $y_i \le c_{\text{lower}, i}$. |
| `censor_upper` | `Symbol` / `Vector` | `nothing` | Right-censoring bound: observation is known only to satisfy $y_i \ge c_{\text{upper}, i}$. |
| `volatility` | `Bool` | `false` | Enables spatiotemporal stochastic volatility on observation variance $\sigma_{y, i}$. |

#### Supported Likelihood Families

| Family (`family=...`) | Link Function | Output Support | Key Parameters & Priors | Assumptions & Utility |
| :--- | :--- | :--- | :--- | :--- |
| `:gaussian` | $\mu = \eta$ | $y \in \mathbb{R}$ | Residual standard deviation $\sigma_y \sim \operatorname{Exponential}(1.0)$. | Continuous symmetrically distributed residuals with homogeneous or heteroscedastic noise. |
| `:poisson` | $\lambda = \exp(\eta)$ | $y \in \{0, 1, 2, \dots\}$ | Rate parameter $\lambda_i = \exp(\eta_i + \text{offset}_i)$. | Equidispersed count data where $\operatorname{Var}(y) = \mathbb{E}[y]$. Standard for disease rates. |
| `:negbin` | $\mu = \exp(\eta)$ | $y \in \{0, 1, 2, \dots\}$ | Dispersion parameter $r \sim \operatorname{Gamma}(2.0, 0.5)$, $\operatorname{Var}(y) = \mu + \mu^2 / r$. | Overdispersed count data capturing unobserved heterogeneity. |
| `:bernoulli` | $p = \operatorname{logistic}(\eta)$ | $y \in \{0, 1\}$ | Success probability $p_i = 1 / (1 + \exp(-\eta_i))$. | Binary presence/absence and classification outcomes. |
| `:binomial` | $p = \operatorname{logistic}(\eta)$ | $y \in \{0, 1, \dots, N_i\}$ | $y_i \sim \operatorname{Binomial}(N_i, p_i)$. Requires `trials`. | Aggregated binary trials (e.g. positive tests out of total tested). |
| `:beta` | $\mu = \operatorname{logistic}(\eta)$ | $y \in (0, 1)$ | Precision parameter $\kappa \sim \operatorname{Exponential}(1.0)$. | Continuous proportions, percentages, and fractional coverage indices. |
| `:gamma` | $\mu = \exp(\eta)$ | $y > 0$ | Shape $\alpha \sim \operatorname{Exponential}(1.0)$, scale $\theta = \mu / \alpha$. | Right-skewed positive continuous data with constant coefficient of variation. |
| `:lognormal` | $\mu_{\log} = \eta$ | $y > 0$ | Scale $\sigma_{\log} \sim \operatorname{Exponential}(1.0)$. | Multiplicative growth processes and heavy-tailed positive measurements. |
| `:studentt` | $\mu = \eta$ | $y \in \mathbb{R}$ | Degrees of freedom $\nu \sim \operatorname{Gamma}(2.0, 0.1)$, scale $\sigma \sim \operatorname{Exponential}(1.0)$. | Heavy-tailed robust regression resistant to outlier contamination. |
| `:exponential` | $\lambda = \exp(-\eta)$ | $y > 0$ | Rate parameter $\lambda_i = \exp(-\eta_i)$. | Memoryless survival time and inter-arrival event durations. |
| `:weibull` | $\lambda = \exp(\eta)$ | $y > 0$ | Shape parameter $k \sim \operatorname{Exponential}(1.0)$. | Monotonically increasing or decreasing hazard rates in survival analysis. |
| `:gev` | $\mu = \eta$ | $y \in \mathbb{R}$ | Generalized Extreme Value: scale $\sigma > 0$, shape $\xi \in \mathbb{R}$. | Block maxima modeling in environmental hydrology and extreme weather. |
| `:zipoisson` | $\lambda = \exp(\eta)$ | $y \in \{0, 1, \dots\}$ | Zero-inflation probability $\pi \sim \operatorname{Beta}(1, 1)$. | Excess zeros arising from dual generating processes (structural + sampling zeros). |
| `:zinegbin` | $\mu = \exp(\eta)$ | $y \in \{0, 1, \dots\}$ | Zero-inflation $\pi \sim \operatorname{Beta}(1, 1)$, dispersion $r \sim \operatorname{Gamma}(2.0, 0.5)$. | Overdispersed counts with structural zero inflation. |
| `:ordered_logistic` | Cutpoints $c_k$ | $y \in \{1, \dots, K\}$ | Ordered categorical threshold vector $c_1 < c_2 < \dots < c_{K-1}$. | Likert scale survey responses and graded disease severity stages. |
| `:ordered_probit` | Cutpoints $c_k$ | $y \in \{1, \dots, K\}$ | Standard normal CDF link $\Phi(\cdot)$ with ordered cutpoints. | Latent Gaussian threshold crossing models for ordinal ratings. |
| `:categorical` | Softmax $\eta_k$ | $y \in \{1, \dots, K\}$ | Categorical probability vector $p = \operatorname{softmax}(\eta)$. | Unordered multi-class choice and state classifications. |
| `:multinomial` | Softmax $\eta_k$ | Vector counts | Multinomial count vector across $K$ categories. | Compositional count data across competing categorical outcomes. |
| `:dirichlet` | Softmax $\eta_k$ | Simplex $\Delta^{K-1}$ | Concentration parameter vector $\alpha = \exp(\eta)$. | Continuous compositional proportions summing to 1. |

---

### 2.3. Right-Hand Side (RHS) Component Syntax

The RHS formula combines linear fixed effects, structured random fields, and process dynamics:

| Module | Purpose | Key Parameters | Example Usage |
| :--- | :--- | :--- | :--- |
| `intercept()` | Controls global intercept prior. | `prior` | `intercept(prior=Normal(0, 5))` |
| `fixed()` | Fixed-effect regression coefficients. | `prior`, `contrast` | `fixed(elevation, prior=Normal(0, 1))` |
| `random()` | Structured & unstructured random fields. | `model`, `sigma`, `rho`, `lengthscale`, etc. | `random(s_idx, model=bym2)` |
| `mixed()` | Correlated random slopes and intercepts. | `model`, `method` | `mixed(1 + poverty \| region)` |
| `dynamics()` | Mechanistic state-space differential equations. | `model`, `r`, `K`, `velocity`, `diffusion` | `dynamics(time, model=:logistic, r=Normal(0.5, 0.1))` |
| `eigen()` | Bayesian PCA factor analysis. | `n_factors`, `pca_sd` | `eigen(pollutant1, pollutant2, n_factors=1)` |
| `nested()` | Multi-fidelity supervised proxy models. | `formula`, `data_source` | `nested(proxy, formula="...", data_source=df_proxy)` |
| `sciml()` | Scientific Machine Learning ODE/PDE integration. | `model_func`, `solver` | `sciml(t, model_func=my_ode)` |
| `custom()` | User-injected raw Turing code fragments. | `code_fragment` | `custom(code_fragment="...")` |

---

### 2.4. Formula Algebraic Operators

The RHS parser (`decompose_bstm_formula`) supports algebraic composition operators:

- **Addition (`+`)**: Additive combination of linear terms ($\eta = \eta_1 + \eta_2$).
- **Kronecker Product (`⊗`)**: Spatiotemporal or multidimensional interaction ($A \otimes B$).
  - *Example*: `random(s_idx, model=icar) ⊗ random(year, model=ar1)` constructs a Knorr-Held Type IV space-time interaction field ($Q_{st} = Q_t \otimes Q_s$).
- **Pipe (`|>`)**: Spatially-varying or time-varying coefficient models.
  - *Example*: `poverty |> random(s_idx, model=icar)` constructs a spatially-varying slope $\beta(s) \cdot \text{poverty}_i$.
  - *Example*: `random(s_idx, model=icar) |> random(month, model=pspline)` constructs spatially-varying seasonal splines.
- **Composition (`∘`)**: Hierarchical modulation or Log-Gaussian Cox Processes.
  - *Example*: `pointprocess(model=:lgcp) ∘ random(s_idx, model=icar)`.

---

### 2.5. Prior Specification & Penalized Complexity (PC) Priors API

`bstm` provides automated mapping between intuitive quantile constraints and mathematical prior distributions:

```julia
# 1. PC Prior Quantile Constraints in Formula Calls
@bstm(
    likelihood(y, family=poisson) ~
        intercept(prior = Normal(0, 5)) +
        fixed(elevation, prior = (2.0, 0.05)) +       # P(|β| > 2.0) = 0.05 => Normal(0, 1.02)
        random(s_idx, model=bym2, 
            sigma = (1.0, 0.01),                     # P(σ > 1.0) = 0.01   => Exponential(4.605)
            rho = (0.5, 0.05)                        # P(ρ > 0.5) = 0.05   => Exponential on -log(1-ρ)
        ) +
        random(time, model=gp, 
            lengthscale = (0.1, 0.01)                # P(ℓ < 0.1) = 0.01   => Exponential on 1/ℓ
        ),
    df, W=W
)
```

| Parameter Type | Quantile Constraint Syntax | Base Model State | Induced Prior Distribution |
| :--- | :--- | :--- | :--- |
| **Standard Deviation (`sigma`, `kappa`)** | `(U, α)` $\implies P(\sigma > U) = \alpha$ | $\sigma = 0$ (No variation) | $\operatorname{Exponential}(\lambda), \; \lambda = -\log(\alpha)/U$ |
| **Correlation (`rho`)** | `(U, α)` $\implies P(\rho > U) = \alpha$ | $\rho = 0$ (Independent noise) | $\operatorname{Exponential}(\lambda)$ on $\theta = -\log(1-\rho)$ |
| **Lengthscale (`lengthscale`, `ls`)** | `(U, α)` $\implies P(\ell < U) = \alpha$ | $\ell = \infty$ (Flat constant) | $\operatorname{Exponential}(\lambda)$ on $\theta = 1/\ell$ |
| **Fixed Effects / Slopes (`prior`)** | `(U, α)` $\implies P(\|\beta\| > U) = \alpha$ | $\beta = 0$ (Null effect) | $\operatorname{Normal}(0, \sigma_{\beta}), \; \sigma_{\beta} = \frac{-U}{\Phi^{-1}(\alpha/2)}$ |

---

## 3. The `ComponentModel` Interface & Extension Guide

To add a new latent component, create a struct subtyping `ComponentModel` and implement the four core interface methods.

### 3.1. Interface Lifecycle & Methods

```
                        ┌────────────────────────────────────────────────────────┐
                        │  1. get_precomputes(m, M, mod_data)                    │
                        │     (Data validation, basis/precision precomputes)     │
                        └───────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                        ┌────────────────────────────────────────────────────────┐
                        │  2. get_priors(m, spec, arch, outcome_idx, M)          │
                        │     (Generates Turing prior code strings)              │
                        └───────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                        ┌────────────────────────────────────────────────────────┐
                        │  3. get_updates(m, spec, arch, outcome_idx, M)         │
                        │     (Calculates latent effect and updates eta)         │
                        └───────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                        ┌────────────────────────────────────────────────────────┐
                        │  4. get_effects(m, chain, M, n_samples, ...)           │
                        │     (Extracts posterior samples and computes effects)  │
                        └────────────────────────────────────────────────────────┘
```

#### Method Specifications:

1. **`get_precomputes(m::ComponentModel, M::NamedTuple, mod_data::Dict)::NamedTuple`**
   - Validates required columns in `M.data` and generates data structures (e.g., basis matrices $B$, precision matrix templates $Q$, eigenvalue decompositions $U, \Lambda$). Stored in `spec.hyper`.

2. **`get_priors(m::ComponentModel, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String`**
   - Emits Turing `@model` code declaring prior distributions for hyperparameters (e.g., `sigma`, `rho_unconstrained`) and standard normal innovations `ure`.

3. **`get_updates(m::ComponentModel, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String`**
   - Emits Turing code computing the realized structured latent field `sre` from `ure` and hyperparameters, and adds the contribution into the linear predictor `eta`.

4. **`get_effects(m::ComponentModel, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple`**
   - Extracts posterior samples from `chain` via `ParamRegistry` and reconstructs posterior trajectories and credible intervals for post-processing and plotting.

---

### 3.2. Canonical Implementation Example: The `IID` Component

Below is a complete implementation of an unstructured group-level random effect ($\phi_g \sim \mathcal{N}(0, \sigma^2)$):

```julia
# 1. Struct Definition
struct IID <: ComponentModel
    sigma::Distribution
    method::Symbol # :noncentered, :centered, :marginalized
end

# 2. Registration in bstm registries
COMPONENT_TYPE_REGISTRY[:iid] = IID
COMPONENT_CONSTRUCTORS[:iid] = (p, params) -> IID(
    get(p, :sigma, Exponential(1.0)),
    get(params, :method, :noncentered)
)

# 3. Precomputations
function get_precomputes(m::IID, M::NamedTuple, mod_data::Dict)::NamedTuple
    n_latent = if mod_data[:structure] == :spatial
        M.s_N
    elseif mod_data[:structure] == :temporal
        M.t_N
    else
        length(unique(M.data[!, mod_data[:var]]))
    end
    return (n_latent = n_latent,)
end

# 4. Turing Priors Generator
function get_priors(m::IID, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    
    return """
    $(p_names.sigma) ~ $(_distribution_to_string(m.sigma))
    $(p_names.ure) ~ MvNormal(zeros(T, $(n_latent)), I)
    """
end

# 5. Turing Updates Generator
function get_updates(m::IID, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = spec.structure == :spatial ? "s_idx" : (spec.structure == :temporal ? "t_idx" : "mixed_idx_$(spec.var)")
    
    return """
    $(p_names.sre) = $(p_names.sigma) .* $(p_names.ure)
    $(eta_target) .+= $(p_names.sre)[$(index_var)]
    """
end

# 6. Posterior Reconstruction
function get_effects(m::IID, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    v = generate_full_variable_names(spec, M.model_arch, 1)
    sigma_samples = get_param_samples(chain, M.param_registry, Symbol(v.sigma))
    ure_samples = get_param_samples(chain, M.param_registry, Symbol(v.ure))
    
    latent_field = ure_samples .* reshape(sigma_samples, 1, :)
    index_var = spec.structure == :spatial ? M.s_idx : (spec.structure == :temporal ? M.t_idx : M.data[!, spec.var])
    effect = latent_field[index_var, :]
    
    return (structured=effect, noisy=effect)
end
```

---

### 3.3. Canonical Naming Standard

All parameter symbols generated across components follow the strict sequence:

$$\mathbf{\{quantity\}\_\{descriptor\}\_\{key\}[\_\{outcome\}]}$$

- **Quantity**: `beta`, `sigma`, `rho`, `ls`, `ure`, `sre`, `threshold`, `v`, `alpha`, `K`, `r`.
- **Descriptor**: `unconstrained`, `unscaled`, `inducing`, `diag`, `pic`, `predator`, `cluster`, `st_interaction`, `flat`.
- **Key**: Unique component identifier derived from formula term (`s_idx`, `year`, `space`, etc.).
- **Outcome**: `1`, `2` (for multivariate models).

#### Core Token Definitions:
- **`beta` / `beta_flat`**: Fixed effects regression coefficients.
- **`ure_<key>`**: Unstructured Random Error / standard normal innovations driving the stochastic process.
- **`sre_<key>`**: Structured Random Error / realized structured latent field.
- **`rho_unconstrained_<key>`**: Unconstrained transform of correlation parameter $\rho \in (-1, 1)$ or $(0, 1)$.

---

## 4. Parameter Registry Engine (`src/parameters.jl`)

The `ParamRegistry` system provides central parameter discovery, support categorization, alias resolution, and MCMC extraction.

### 4.1. Core Types

```julia
struct ParamDescriptor
    name::Symbol             # Canonical parameter name (e.g. :sigma_year)
    role::Symbol             # :fixed, :hyper, :latent, :innovation, :precision
    component::Symbol        # Component key (:fixed, :year, :s_idx)
    model_type::Symbol       # :fixed, :ar1, :bym2, :gp, etc.
    support::Symbol          # :real, :positive, :unit_interval, :discrete
    dimension::Int           # 1 for scalar, >1 for vector/matrix
    prior_str::String        # String representation of prior distribution
end

struct ParamRegistry
    params::Dict{Symbol, ParamDescriptor}
    aliases::Dict{Symbol, Symbol}
    by_component::Dict{Symbol, Vector{Symbol}}
    by_role::Dict{Symbol, Vector{Symbol}}
end
```

### 4.2. Registry Functions

- **`build_param_registry(M::NamedTuple)::ParamRegistry`**: Scans model specification `M` and compiles an initial parameter registry.
- **`calibrate_param_registry(reg::ParamRegistry, vi::VarInfo)::ParamRegistry`**: Introspects DynamicPPL `VarInfo` to calibrate active variable names, dimensions, and empirical supports.
- **`get_param_samples(chain, reg::ParamRegistry, param_sym::Symbol)`**: Fetches posterior samples across `FlexiChain`, `Chains`, `DataFrame`, or `Dict` containers with automatic alias fallback (`ure` $\leftrightarrow$ `innovations`, `sre` $\leftrightarrow$ `latent`, `beta` $\leftrightarrow$ `Xfixed_beta_prop`).
- **`_find_parameter(reg::ParamRegistry, target_name::Symbol)`**: Resolves parameter names accounting for multivariate suffixes and historical aliases.

---

## 5. Mathematical Formulations, Assumptions & Utility of Components

### 5.1. Spatial Areal GMRF Models

#### 1. Intrinsic Conditional Autoregressive (`ICAR`)
- **Mathematical Formulation**:
  $$u_i \mid u_{-i} \sim \mathcal{N}\left( \frac{1}{d_i} \sum_{j \in N(i)} u_j, \frac{\sigma^2}{d_i} \right)$$
  In matrix notation, $u \sim \mathcal{N}(0, (\tau Q)^{-1})$ where $Q = \operatorname{diag}(W \mathbf{1}) - W$ is the singular graph Laplacian and $\tau = 1/\sigma^2$.
- **Core Assumptions**: First-order intrinsic stationarity; singular precision matrix with sum-to-zero constraint $\sum_{i=1}^S u_i = 0$; spatial Markov property (conditional independence given neighbors).
- **Utility**: Standard spatial smoothing across discrete administrative or geographic regions (Besag, 1974).

#### 2. Besag-York-Mollié (`BYM2`)
- **Mathematical Formulation**:
  $$\phi_s = \sigma \left( \sqrt{1 - \rho} \cdot v_s + \sqrt{\rho / s_{\text{scale}}} \cdot u_s \right)$$
  where $u \sim \operatorname{ICAR}(W)$, $v \sim \mathcal{N}(0, I_S)$, $\rho \in [0, 1]$ is the spatial mixing parameter, and $s_{\text{scale}} = \exp(-\frac{1}{S-1}\sum_{i=1}^{S-1} \log \lambda_i)$ is the Riebler et al. (2016) spectral scaling factor.
- **Core Assumptions**: Additive orthogonal decomposition into spatially structured clustering ($u$) and unstructured white noise ($v$); graph is fully connected.
- **Utility**: The gold standard in spatial epidemiology and disease mapping. Allows direct interpretation of $\rho$ as the proportion of total variance explained by spatial autocorrelation.

#### 3. Leroux CAR (`Leroux`)
- **Mathematical Formulation**:
  $$Q = (1 - \rho) I_S + \rho Q_{\text{ICAR}}, \quad \phi \sim \mathcal{N}\left(0, (\sigma^2 Q)^{-1}\right)$$
- **Core Assumptions**: Full-rank proper precision matrix for all $\rho \in [0, 1)$; smooth convex interpolation between pure independence ($\rho=0$) and ICAR ($\rho=1$).
- **Utility**: Prevents boundary edge singularities and avoids rigid sum-to-zero constraints; highly stable for small sample sizes.

#### 4. Simultaneous Autoregressive (`SAR`)
- **Mathematical Formulation**:
  $$\phi = \rho W_{\text{std}} \phi + \epsilon, \quad \phi = (I - \rho W_{\text{std}})^{-1} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$
- **Core Assumptions**: Spatial lag autoregressive process; stationarity requires $\rho \in (1/\lambda_{\min}, 1/\lambda_{\max})$.
- **Utility**: Spatial econometrics, hedonic pricing, and geographic spillover modeling.

#### 5. Local Adaptive GMRF (`LocalAdaptive`)
- **Mathematical Formulation**:
  $$\phi \sim \mathcal{N}\left( \mu_{\text{cluster}(s)}, (\sigma^2 Q_{\text{Leroux}})^{-1} \right)$$
  where $\mu_g$ is a cluster-specific mean effect estimated across spatial clusters $g \in \{1, \dots, K\}$.
- **Core Assumptions**: Continuous background spatial covariance with piecewise discontinuous mean shifts across geographic regimes.
- **Utility**: Environmental epidemiology across distinct jurisdictions, socioeconomic divides, or natural geographic boundaries.

#### 6. Multivariate Conditional Autoregressive (`MCAR`)
- **Mathematical Formulation**:
  $$\operatorname{vec}(\boldsymbol{\Phi}) \sim \mathcal{N}\left(\mathbf{0}, (\mathbf{\Omega}_{\text{cross}} \otimes \mathbf{Q}_{\text{spatial}})^{-1}\right)$$
  where $\mathbf{Q}_{\text{spatial}} = (1 - \rho)\mathbf{I}_S + \rho \mathbf{Q}_{\text{ICAR}}$ is the proper spatial precision matrix, and $\mathbf{\Sigma}_{\text{cross}} = \mathbf{\Omega}_{\text{cross}}^{-1} = \operatorname{diag}(\boldsymbol{\sigma})\mathbf{L}_{\text{corr}}\mathbf{L}_{\text{corr}}^T \operatorname{diag}(\boldsymbol{\sigma})$ captures cross-outcome correlations via an LKJ prior.
- **Core Assumptions**: Cross-outcome and spatial dependencies factorize into a Kronecker separable precision structure.
- **Utility**: Joint multi-disease spatial mapping and multi-species ecological co-occurrence across geographic regions (Gelfand & Vounatsou, 2003).

---

### 5.2. Continuous Geostatistical & Point-Referenced Models

#### 1. Gaussian Process (`GP`)
- **Mathematical Formulation**:
  $$f(x) \sim \mathcal{GP}(0, k(x, x')), \quad k_{\text{SE}}(d) = \sigma^2 \exp\left( -\frac{d^2}{2\ell^2} \right), \quad k_{\text{Matérn}}(d) = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\sqrt{2\nu}\frac{d}{\ell}\right)^\nu K_\nu\left(\sqrt{2\nu}\frac{d}{\ell}\right)$$
- **Core Assumptions**: Second-order stationarity and isotropy (or anisotropic ARD lengthscales $\ell_d$).
- **Utility**: Exact spatial interpolation (Kriging), continuous environmental field mapping, sensor fusion.

#### 2. Nearest Neighbor Gaussian Process (`NNGP`)
- **Mathematical Formulation**:
  Approximates continuous GP joint densities via $m$-nearest neighbor conditioning sets among preceding points in a spatial ordering (Datta et al., 2016):
  $$p(w_1, \dots, w_N) \approx p(w_1) \prod_{i=2}^N \mathcal{N}\left(\mathbf{B}_i \mathbf{w}_{N(s_i)}, F_i\right)$$
  where $\mathbf{B}_i = C(s_i, N(s_i)) [C(N(s_i), N(s_i))]^{-1}$ and $F_i = C(s_i, s_i) - \mathbf{B}_i C(N(s_i), s_i)$.
- **Core Assumptions**: High-order conditional independence given the $m$ nearest preceding spatial neighbors.
- **Utility**: Highly scalable geostatistics achieving $\mathcal{O}(N m^3)$ runtime and $\mathcal{O}(N m)$ memory for point datasets $N > 10^5$ without pseudo-inputs.

#### 3. Sparse Gaussian Process (`SparseGP` / `FITC` / `PIC`)
- **Mathematical Formulation**:
  Approximates dense covariance $K_{ff}$ via $M \ll N$ pseudo-inputs $u = f(X_u)$ using the Fully Independent Training Conditional (FITC) factorization:
  $$K_{ff} \approx Q_{ff} + \operatorname{diag}(K_{ff} - Q_{ff}), \quad Q_{ff} = K_{fu} K_{uu}^{-1} K_{uf}$$
- **Core Assumptions**: Latent spatial field covariance is low-rank conditionally independent given inducing points.
- **Utility**: Scales Gaussian Process inference from $\mathcal{O}(N^3)$ to $\mathcal{O}(NM^2)$, making continuous GPs tractable for $N > 10^5$ observations.

#### 4. Random Fourier Features (`RFF`)
- **Mathematical Formulation**:
  By Bochner's theorem, stationary kernel $k(x - x') = \int p(\omega) e^{i \omega^T (x - x')} d\omega \approx z(x)^T z(x')$:
  $$z(x) = \sqrt{\frac{2}{D}} \left[ \cos(W x + b) \right], \quad W \sim \mathcal{N}(0, \ell^{-2} I), \quad b \sim \operatorname{Uniform}(0, 2\pi)$$
- **Core Assumptions**: Shift-invariant stationary kernel; projection dimension $D$ is sufficiently large.
- **Utility**: Linear $\mathcal{O}(ND)$ time complexity for continuous spatial fields in high-dimensional settings.

#### 5. Stochastic Partial Differential Equations (`SPDE`)
- **Mathematical Formulation**:
  Represents continuous Matérn fields as solutions to the linear fractional SPDE (Lindgren et al., 2011):
  $$(\kappa^2 - \Delta)^{\alpha/2} u(s) = \mathcal{W}(s)$$
  Discretized on a Constrained Delaunay Triangulation mesh via piecewise linear basis functions $u(s) = \sum_{i=1}^V \psi_i(s) u_i$.
- **Core Assumptions**: Matérn field smoothness $\nu = \alpha - d/2$; finite element discretization accurately captures spatial boundary conditions.
- **Utility**: Bridges continuous Gaussian fields with sparse precision GMRFs ($\mathcal{O}(N^{1.5})$ complexity).

---

### 5.3. Temporal Models

#### 1. Autoregressive Processes (`AR1`, `AR2`)
- **Mathematical Formulation**:
  - $\text{AR}(1)$: $x_t = \rho x_{t-1} + \sigma \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, 1)$
  - $\text{AR}(2)$: $x_t = \rho_1 x_{t-1} + \rho_2 x_{t-2} + \sigma \epsilon_t$
- **Core Assumptions**: Weak stationarity ($|\rho| < 1$ for AR1; triangular stability domain for AR2: $\rho_1 + \rho_2 < 1, \rho_2 - \rho_1 < 1, |\rho_2| < 1$).
- **Utility**: Serial autocorrelation, short-memory momentum, macro-temporal shocks.

#### 2. Random Walks (`RW1`, `RW2`)
- **Mathematical Formulation**:
  - $\text{RW}(1)$: $x_t - x_{t-1} \sim \mathcal{N}(0, \sigma^2)$ (first difference / stochastic level)
  - $\text{RW}(2)$: $x_t - 2x_{t-1} + x_{t-2} \sim \mathcal{N}(0, \sigma^2)$ (second difference / stochastic curvature)
- **Core Assumptions**: Non-stationary stochastic trends; RW2 enforces smooth second-order derivatives.
- **Utility**: Nonparametric time trend estimation without imposing rigid polynomial forms.

#### 3. Harmonic & Periodic Models (`Harmonic`, `Cyclic`)
- **Mathematical Formulation**:
  $$f(t) = \sum_{k=1}^K \left( \beta_{\cos, k} \cos\left(\frac{2\pi k t}{P}\right) + \beta_{\sin, k} \sin\left(\frac{2\pi k t}{P}\right) \right)$$
- **Core Assumptions**: Strict periodicity with fundamental cycle period $P$ (e.g. 12 months, 24 hours).
- **Utility**: Seasonal epidemiological peaks, environmental annual rhythms, tidal cycles.

#### 4. Threshold Autoregressive (`TAR`)
- **Mathematical Formulation**:
  $$x_t = \begin{cases} \rho_1 x_{t-1} + \sigma_1 \epsilon_t & \text{if } x_{t-d} \le c \\ \rho_2 x_{t-1} + \sigma_2 \epsilon_t & \text{if } x_{t-d} > c \end{cases}$$
- **Core Assumptions**: Piecewise linear regime switching governed by threshold variable and delay lag $d$.
- **Utility**: Ecological tipping points, predator-prey collapses, financial market regime shifts.

---

### 5.4. Spatiotemporal Interactions (`Composed`, `⊗`)

Knorr-Held (2000) spatiotemporal interactions are constructed via the Kronecker product operator:

$$\eta_{st} = \alpha + u_s + v_t + \delta_{st}, \quad \delta \sim \mathcal{N}\left(0, \sigma_{\delta}^2 (Q_s \otimes Q_t)^{-1}\right)$$

| Interaction Type | Spatial Structure ($Q_s$) | Temporal Structure ($Q_t$) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Type I** | $I_S$ (Unstructured) | $I_T$ (Unstructured) | Uncorrelated white-noise space-time shocks. |
| **Type II** | $I_S$ (Unstructured) | $Q_{\text{RW1}}$ or $Q_{\text{AR1}}$ (Structured) | Spatial units follow independent, temporally persistent trajectories. |
| **Type III**| $Q_{\text{ICAR}}$ (Structured) | $I_T$ (Unstructured) | Spatially correlated maps that vary independently from year to year. |
| **Type IV** | $Q_{\text{ICAR}}$ (Structured) | $Q_{\text{RW1}}$ / $Q_{\text{AR1}}$ (Structured) | Spatially correlated patterns that evolve smoothly over continuous time. |

---

### 5.5. Mechanistic & Dynamical Systems (`dynamics()`)

The `dynamics()` module fuses physical and ecological differential equations into the Bayesian state-space framework:

1. **Logistic Population Growth**:
   $$N_t = N_{t-1} + r N_{t-1} \left(1 - \frac{N_{t-1}}{K}\right) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)$$
2. **Advection-Diffusion PDE**:
   $$\frac{\partial u}{\partial t} = D \nabla^2 u - v \cdot \nabla u + \epsilon(s, t)$$
   Discretized across graph Laplacian $L = D - W$:
   $$u_{t} = u_{t-1} - \Delta t \left( D L u_{t-1} + v \cdot \nabla u_{t-1} \right) + \epsilon_t$$

- **Core Assumptions**: Known physical/biological differential equations with stochastic environmental noise.
- **Utility**: Physics-Informed Machine Learning (SciML), population viability analysis, pollution plume tracking.

---

### 5.6. Bayesian Factor Analysis (`eigen()`)

Bayesian PCA decomposes $P$ multivariate outcomes into $K \ll P$ orthogonal latent factors:

$$Y = Z \Lambda^{1/2} U^T + E, \quad E \sim \mathcal{N}(0, \operatorname{diag}(\sigma_{\epsilon, 1}^2, \dots, \sigma_{\epsilon, P}^2))$$

To enforce strict orthonormality ($U^T U = I_K$) without identification sign-flipping or unconstrained matrix drift, `bstm` parameterizes $U$ using a product of Householder reflections:

$$H_k = I - 2 \frac{v_k v_k^T}{\|v_k\|^2}, \quad U = \prod_{k=1}^K H_k$$

- **Core Assumptions**: High-dimensional outcomes are generated by a low-dimensional orthogonal latent subspace.
- **Utility**: Multi-pollutant exposure indices, ecological multi-species co-occurrence, multi-phenotype genetics.

---

### 5.7. Supervised Multi-Fidelity Modeling (`nested()`)

The `nested()` supervisor module enables multi-fidelity transfer learning and errors-in-variables covariate modeling:

$$\eta_{\text{main}} = \dots + \rho_{\text{nested}} \cdot \eta_{\text{sub}}(\text{Data}_{\text{aux}})$$

- **Core Assumptions**: Coarse or noisy proxy data shares the latent spatial/temporal functional form up to scaling $\rho$.
- **Utility**: Integrating satellite proxy observations with sparse ground-station monitors; jointly modeling censored or missing covariates.

---

## 6. Sampling, Inference & Sampler Optimization

### 6.1. `get_optimal_sampler` and `precompute_step_sizes`

```julia
# Pre-inspect dimension-scaled initial step sizes, tree depths, and principled adaptation steps
proposal_info = precompute_step_sizes(model_obj; min_ϵ=1e-4, max_ϵ=1.0, max_depth=10, adaptation_steps=:auto)

# Build optimal composite Gibbs sampler with pre-conditioned proposal step sizes
os = get_optimal_sampler(
    model_obj;
    sampler_choice = :auto,
    adaptation_steps = :auto,            # Scaled by metric dimension O(sqrt(D)) and condition number
    init_ϵ = :auto,                      # Dimensional scaling ϵ ~ 0.5 * D^(-1/4)
    # init_ϵ = 0.05,                     # Or fixed scalar step size
    # init_ϵ = Dict(:s_idx => 0.03, :year => 0.08), # Or per-component step size
    max_depth = 10,                      # Capped binary tree depth (e.g. 6-10)
    min_ϵ = 1e-4,                        # Lower bound to avoid step collapse
    max_ϵ = 1.0,                         # Upper bound to prevent divergence
    adtype = AutoForwardDiff(),
    n_chains = 3
)
```

`get_optimal_sampler` parses `model_obj` using `ParamRegistry` and builds a block-partitioned composite Gibbs sampler tailored to parameter supports:

1. **Step-Size ($\epsilon$) Pre-Conditioning & Bounds**: Uses optimal Roberts & Rosenthal (2001) dimensional curvature scaling $\epsilon \approx 0.5 \cdot D^{-1/4}$ clamped within `[min_ϵ, max_ϵ]`. This completely bypasses the unstable `find_good_stepsize` search in high-dimensional GMRF/spatial blocks.
2. **Maximum Tree Depth Limiting (`max_depth`)**: Caps the maximum leapfrog trajectory doubling depth to prevent long tree stalls during warmup.
3. **Discrete Variables** ($y \in \mathbb{Z}$): Assigned `PG` (Particle Gibbs).
4. **Gaussian Latent Vectors** ($\mathcal{N}(0, \Sigma)$): Assigned `ESS` (Elliptical Slice Sampling) for gradient-free sampling of high-dimensional latent fields.
5. **Bounded Scalar Parameters** ($\sigma > 0, \rho \in [0, 1]$): Assigned `Slice` sampling or unconstrained `NUTS`.
6. **Unbounded Differentiable Blocks**: Assigned `NUTS(adaptation_steps, target_acceptance; max_depth=max_depth, init_ϵ=init_ϵ, adtype=adtype)`.

---

### 6.2. Sampler Interface Options

```julia
# MCMC Sampling
chn = sample(m, NUTS(), 1000; progress=false)
chn = sample(m, os, 1000; progress=false)
chn = bstm_sample(m, 1000; n_chains=3, progress=false)

# Optimization Inference
map_res = optimize(m, MAP())
mle_res = optimize(m, MLE())
vi_res = vi(m, ADVI(10, 1000))
```

---

## 7. Posterior Reconstruction, Prediction & Diagnostics Engine

### 7.1. `model_results_comprehensive`

```julia
res = model_results_comprehensive(model, chain; data=nothing, alpha=0.05)
```

Processes fitted models and MCMC chains into a clean, structured analytical data object (pure data, without graphics rendering):

| Return Field | Type | Description |
| :--- | :--- | :--- |
| `metrics` | `NamedTuple` | Model evaluation & convergence: `rmse`, `r_pearson`, `waic`, `rhat`, `ess`, `time`. |
| `parameters` | `DataFrame` | Parameter-level posterior summary table: `parameters`, `mean`, `std`, `median`, `lower_95`, `upper_95`, `rhat`, `ess`. |
| `effects` | `NamedTuple` | Latent component effect summaries (spatial, temporal, fixed, random, interaction). |
| `predictions` | `NamedTuple` | Observation-level fitted values: `denoised` (`mean`, `median`, `std`, `lower`, `upper`) and `noisy`. |
| `draws` | `NamedTuple` | Raw posterior sample matrices: `predictions_denoised`, `predictions_noisy`, `weights`, `log_likelihood`. |
| `arch` | `ModelArchitecture` | Model architecture type (`UnivariateArchitecture`, `MultivariateArchitecture`, `MultifidelityArchitecture`). |

### 7.2. Diagnostic & Component Effect Plotting (`bstm_plots`)

```julia
plots_res = bstm_plots(res; data=inp_df, au=data_scot.au, save_dir=nothing, save_prefix="", fmt="png", dpi=150)
```

Generates the complete suite of diagnostic plots and underlying tidy datasets from `res`:

| Return Field | Type | Description |
| :--- | :--- | :--- |
| `plots` | `NamedTuple` | Rendered `Plots.Plot` objects (`:posterior_predictive_check`, `:spatial`, `:spatial_s_idx`, `:spatial_observed`, `:spatial_fitted`, `:spatial_residuals`, `:temporal`, `:spacetime_predictions`, `:fixed_effects`, `:conditional_effects`, `:st_interaction_heatmap`, `:svc_*`, `:mixed_effects`). |
| `plots_data` | `NamedTuple` | Underlying tidy dataframes used to construct all diagnostic plots. |

If `save_dir !== nothing`, all generated plots are automatically saved to disk.

### 7.3. Out-of-Sample Prediction (`predict`)

```julia
preds = predict(model, chain, new_dataframe; alpha=0.05)
```

Projects fitted posterior fields onto a new spatial, temporal, or covariate data grid, automatically re-computing basis matrices for smooth spline terms and Gaussian Process cross-covariances.

### 7.3. Model Inspection (`show_model`)

```julia
show_model(model)
```

Prints formatted diagnostic information about the model, including the parsed formula, data schema, active components, generated Turing model code, and parameter registry mappings.

### 7.4. Benchmark Datasets (`bstm_data`)

```julia
dataset, metadata = bstm_data("scottish_lip")
```

Loads built-in benchmark spatiotemporal datasets (e.g. Scottish Lip Cancer disease mapping dataset with spatial polygons and adjacency matrix $W$).

---

## 8. Plotting Subsystem API (`src/plotting.jl`)

All visualization methods return standard `Plots.Plot` objects and accept customizable themes (`create_theme`).

### 8.1. Function Reference

| Function | Signature | Description |
| :--- | :--- | :--- |
| `choropleth` | `choropleth(polygons, values; cmap=:viridis, center_zero=false, title="")` | Shaded spatial polygon map with automatic percentile scaling and colorbars. Accepts `(polys, vals)` or `(vals, polys)`. |
| `spatial_graph_plot` | `spatial_graph_plot(centroids, g; polygons=nothing, au=nothing, title="")` | Visualizes spatial partitioning polygons, centroid nodes, graph adjacency edges, and boundary hulls. |
| `plot_kde_simple` | `plot_kde_simple(coords; grid_res=100)` or `plot_kde_simple(df; x=:s_x, y=:s_y)` | 2D Kernel Density Estimation surface heatmap with observation point scatter overlays. |
| `timeseries_ci` | `timeseries_ci(x, mean, lower, upper; ribbon_alpha=0.2, title="")` | Credible interval ribbon timeseries plot for temporal trends and seasonal curves. |
| `render_paths!` | `render_paths!(p, paths; color=:crimson, alpha=0.7)` | Overlays agent/individual movement trajectories onto an existing spatial plot. |
| `map_point_occupancy` | `map_point_occupancy(polygons, centroids, current_unit, previous_unit)` | Highlights occupied spatial units on a background map for discrete tracking. |
| `bstm_plots` | `bstm_plots(model, chain, res, M; au=nothing, data=nothing, outcome=1)` | Comprehensive automated diagnostic plotting suite (PPC, fixed effects, spatial, temporal, SVC, etc.). |
| `model_results_plots` | `model_results_plots(res)` | Iterates and displays all diagnostic plots contained in a `model_results` struct. |
| `save_plot` | `save_plot(p, path; fmt=nothing, dpi=150)` | Exports plot to disk, automatically creating parent directories if needed. |
| `create_theme` | `create_theme(; fontsize=10, fontfamily="sans-serif", size=(900, 600))` | Generates standardized styling dictionaries for publication-ready figures. |

---

## 9. Spatial & Spatiotemporal Partitioning API (`src/partitioning.jl`)

Comprehensive spatial discretization, graph extraction, and spatiotemporal index synchronization.

### 9.1. Function Reference

| Function | Signature | Description |
| :--- | :--- | :--- |
| `assign_spatial_units` | `assign_spatial_units(s_x, s_y; area_method=:avt, target_units=10, exact_units=false, target_area=nothing, min_area=0.0, max_area=Inf, min_points=1, max_points=nothing, lengthscale=nothing, radius=nothing, grid_resolution=nothing, aspect_ratio=1.0, prune_empty=false, merge_small_polygons=false, input_polygons=nothing, geom_hull=nothing, kwargs...)` | Primary spatial partitioner. Discretizes 2D coordinates into polygons and builds neighborhood graph $W$. |
| `assign_spatial_units_inferred` | `assign_spatial_units_inferred(W; iterations=50, learning_rate=0.1, buffer_dist=0.5)` | Reconstructs coordinates and Voronoi polygons from an adjacency matrix via force-directed spring layout. |
| `assign_time_units` | `assign_time_units(t_v; time_method="quantile_regular", t_N=10)` | Discretizes continuous or discrete time vectors into categorical temporal indices. |
| `assign_spatiotemporal_units` | `assign_spatiotemporal_units(df; space_x=:s_x, space_y=:s_y, time_var=:t_idx, area_method=:avt, target_units=10, ...)` | Synchronizes space and time partitions into aligned `(s_idx, t_idx, st_idx)` indices and metadata $(S, T, ST)$. |
| `discretize_data` | `discretize_data(X; method="quantile", N_cat=9, brks=nothing, probs=nothing, dx=nothing, minv=nothing, maxv=nothing, quantile_bounds=[0.025, 0.975])` | General 1D variable discretization: `"quantile"`, `"regular"`, `"quantile_regular"`, `"kmeans"`, `"jenks"`, `"provided"`. |
| `spatial_weights_matrix` | `spatial_weights_matrix(W; style=:binary)` | Transforms adjacency $W$ into `:binary`, `:row_standardized` ($D^{-1}W$), or `:variance_stabilized` ($D^{-1/2}W$). |
| `spatial_knn_graph` | `spatial_knn_graph(coords, k)` | Constructs a $k$-nearest neighbors spatial graph and binary adjacency matrix. |
| `spatial_radius_graph` | `spatial_radius_graph(coords, radius)` | Constructs a distance-threshold spatial graph connecting points within distance `radius`. |
| `spatial_block_cv` | `spatial_block_cv(s_x, s_y; n_folds=5, method=:kmeans)` | Generates spatial cross-validation fold IDs using `:kmeans` clustering or `:grid` spatial blocking. |
| `scaling_factor_bym2` | `scaling_factor_bym2(W)` | Computes the BYM2 ICAR precision matrix scaling factor: $s = \exp(-\frac{1}{S-1}\sum \log \lambda_i)$. |
| `ensure_connected!` | `ensure_connected!(g, centroids)` | Connects disjoint spatial graph components by adding minimal bridging edges between closest centroid pairs. |
| `libgeos_lattice_adjacency_matrix` | `libgeos_lattice_adjacency_matrix(rows, cols; contiguity=:queen)` | Fast $\mathcal{O}(N)$ analytical lattice adjacency matrix constructor supporting `:queen` and `:rook` contiguity. |

---

## 10. Cross-Validation Engine (`bstm_cv_orchestrator`)

The `bstm_cv_orchestrator` orchestrates rigorous cross-validation across spatiotemporal datasets:

```julia
cv_res = bstm_cv_orchestrator(
    formula,
    data;
    method = :spatial_block,
    cv_var = :s_idx,
    n_folds = 5,
    sampler = NUTS(500, 0.65),
    n_samples = 500,
    cv_space_vars = [:s_x, :s_y],
    kwargs...
)
```

### Supported CV Methods:
- **`:kfold`**: Standard random $k$-fold cross-validation.
- **`:lolo` (Leave-One-Location-Out)**: Holds out all observations from a specific areal unit or location.
- **`:spatial_block`**: $k$-means spatial block cross-validation to assess geographic generalization without spatial autocorrelation leakage.
- **`:temporal_block`**: Contiguous temporal block holdout for time-series interpolation.
- **`:temporal_forward_chain`**: Rolling-origin forward forecast simulation testing out-of-sample temporal prediction.

---

## 11. Input / Output & Persistence API (`src/input_output.jl`)

The `bstm` framework provides a high-performance, non-redundant serialization and analytical storage subsystem:
- **JLD2 (`.jld2`)**: Full binary serialization of live Turing model states (`m`), formula configurations, data, and posterior MCMC chains (`chn`).
- **DuckDB (`.duckdb`)**: Embedded relational SQL storage for post-processing results (`res`), diagnostic metrics, and posterior samples without memory redundancy.

### 11.1. Function Reference

| Function | Signature | Description |
| :--- | :--- | :--- |
| `save_bstm_model` | `save_bstm_model(path, model; chain=nothing, au=nothing, metadata=Dict(), compress=true)` | Serializes the complete Turing `@model` state, configuration `M`, data, and optional MCMC chain to JLD2. |
| `load_bstm_model` | `load_bstm_model(path; calling_module=Main)` | Reads JLD2 file and re-instantiates a live, callable `DynamicPPL.Model` ready for sampling or prediction. |
| `save_bstm_results` | `save_bstm_results(duckdb_path, res; model=nothing, chain=nothing, au=nothing, table_prefix="", overwrite=true)` | Normalizes `model_results_comprehensive` output into relational DuckDB tables (`metrics`, `parameter_stats`, `predictions`, `spatial_geometries`, `plots_data_*`). |
| `load_bstm_results` | `load_bstm_results(duckdb_path; table_prefix="")` | Reconstructs the `model_results_comprehensive` NamedTuple from DuckDB tables. |
| `query_duckdb` | `query_duckdb(duckdb_path, sql_query)` | Executes analytical SQL queries directly against a BSTM DuckDB database, returning a DataFrame. |
| `export_posterior_samples_to_duckdb` | `export_posterior_samples_to_duckdb(duckdb_path, chain, model=nothing; table_name="bstm_posterior_samples", format=:tidy)` | Exports MCMC posterior draws in `:tidy` long format `(iteration, chain, parameter, value)` or `:wide` format to DuckDB. |
| `import_posterior_samples_from_duckdb` | `import_posterior_samples_from_duckdb(duckdb_path; table_name="bstm_posterior_samples")` | Reads stored posterior draws from DuckDB into a Julia DataFrame. |
| `append_posterior_samples` | `append_posterior_samples(chain1, chain2)` | Concatenates two MCMC chains across sampling iterations for chain extension. |
| `extend_sampling` | `extend_sampling(model, prev_chain, n_additional_samples; sampler=NUTS(), kwargs...)` | Warms up from previous chain state, samples additional iterations, and returns merged chains. |
| `save_bstm_bundle` | `save_bstm_bundle(base_path, model, chain, res; au=nothing, metadata=Dict())` | Unified one-line persistence creating `<base_path>.jld2` (model & chain) and `<base_path>.duckdb` (results). |
| `load_bstm_bundle` | `load_bstm_bundle(base_path; calling_module=Main)` | Unified one-line loader recovering `(model=m, chain=chn, results=res, au=au, metadata=meta)`. |
| `export_spatial_results_to_geojson` | `export_spatial_results_to_geojson(geojson_path, res, au; property_keys=nothing)` | Serializes spatial model results and polygon boundaries to standard RFC 7946 GeoJSON. |
| `extract_posterior_priors` | `extract_posterior_priors(source; parameter_names=nothing, prior_family=:normal)` | Extracts posterior parameters and builds fitted prior distributions for sequential Bayesian updating. |
| `save_model_ensemble` | `save_model_ensemble(duckdb_path, ensemble_dict; overwrite=true)` | Registers a multi-model ensemble in DuckDB and computes $\Delta \text{WAIC}$ and BMA weights. |
| `bma_weighted_predictions` | `bma_weighted_predictions(duckdb_path)` | Computes Bayesian Model Averaged predictions and total variance across all candidate models. |
| `save_out_of_sample_predictions` | `save_out_of_sample_predictions(duckdb_path, pred_df; table_name="out_of_sample_predictions")` | Stores out-of-sample prediction DataFrames into DuckDB. |
| `export_results_to_parquet` | `export_results_to_parquet(duckdb_path, table_name, output_parquet_path)` | Zero-copy compressed Parquet export using DuckDB `COPY`. |
| `export_results_to_csv` | `export_results_to_csv(duckdb_path, table_name, output_csv_path)` | Exports DuckDB table to CSV. |
| `compact_duckdb` | `compact_duckdb(duckdb_path)` | Executes `VACUUM; ANALYZE;` on DuckDB database to reclaim space and optimize query statistics. |

---

## 12. Movement & ADR Telemetry API (`src/movement.jl`)

The movement subsystem implements biophysical Advection-Diffusion-Reaction (ADR) population dynamics and integrated mark-recapture telemetry modeling:

### 12.1. Function Reference

| Function | Signature | Description |
| :--- | :--- | :--- |
| `generate_ADR_simulation_bundle` | `generate_ADR_simulation_bundle(domain_size, n_units, n_years, n_marks; area_method=:cvt, rng=Random.GLOBAL_RNG)` | Simulates synthetic joint population density surveys and mark-recapture telemetry encounters on spatial units. |
| `simulate_correlated_density_vector` | `simulate_correlated_density_vector(habitat_prob, rho_target, log_mu, sigma_resid; rng=Random.GLOBAL_RNG)` | Simulates spatial density vectors correlated with underlying habitat suitability index. |
| `compute_velocity_field` | `compute_velocity_field(prob_vec, grid_dim, strength; mode=:exponential)` | Computes spatial advection velocity vectors $\mathbf{v} \propto \nabla \text{HSI}$. |
| `calculate_multistep_transition` | `calculate_multistep_transition(Gamma_base, steps)` | Calculates multi-step dispersal transition probabilities via matrix exponentiation $\mathbf{\Gamma}^k$. |
| `simulate_posterior_trajectories` | `simulate_posterior_trajectories(Gamma_base, start_units, n_steps, au_context; rho_persistence=0.0, rng=Random.GLOBAL_RNG)` | Simulates individual animal trajectories with directional persistence (Correlated Random Walk). |
| `simulate_mechanistic_trajectories` | `simulate_mechanistic_trajectories(Gamma_sequence, start_units, t_start, au_context; rho_persistence=0.0, n_years_sim=1, rng=Random.GLOBAL_RNG)` | Simulates individual movement through dynamic, time-varying transition kernels $\mathbf{\Gamma}_t$. |
| `compute_suitability_transition_kernel` | `compute_suitability_transition_kernel(suitability_vec, W; sensitivity=1.0, diffusion_weight=0.1)` | Generates spatial Markov transition kernels biased towards high habitat suitability. |
| `calculate_regional_connectivity` | `calculate_regional_connectivity(Gamma, strata_definition)` | Aggregates fine-scale unit transitions into macro-regional migration probability matrices. |
| `plot_ad_ratio_distribution` | `plot_ad_ratio_distribution(advection_field, diffusion_field)` | Generates diagnostic histogram of local Advection-to-Diffusion (Péclet) ratios. |
| `synthesize_adr_results` | `synthesize_adr_results(chain_or_res, sim_data, vel_vectors; au=sim_data.au)` | Comprehensive post-processing, parameter extraction, propagator reconstruction, and multi-panel visualization. |

---

## 13. References

1. **Besag, J.** (1974). Spatial interaction and the statistical analysis of lattice systems. *Journal of the Royal Statistical Society: Series B*, 36(2), 192–225.
2. **Besag, J., York, J., & Mollié, A.** (1991). Bayesian image restoration, with applications in spatial statistics. *Annals of the Institute of Statistical Mathematics*, 43(1), 1–59.
3. **Du, Q., Faber, V., & Gunzburger, M.** (1999). Centroidal Voronoi tessellations: Applications and algorithms. *SIAM Review*, 41(4), 637–676.
4. **Gelfand, A. E., et al.** (2003). Spatial modeling with spatially varying coefficient processes. *Journal of the American Statistical Association*, 98(462), 387–396.
5. **Hooten, M. B., & Hefley, T. J.** (2019). *Bringing Bayesian Models to Life*. CRC Press.
6. **Jenks, G. F.** (1967). The data model concept in statistical mapping. *International Yearbook of Cartography*, 7, 186–190.
7. **Knorr-Held, L.** (2000). Bayesian modelling of inseparable space-time variation in disease risk. *Statistical Methods in Medical Research*, 9(3), 205–220.
8. **Leroux, B. G., Lei, X., & Breslow, N.** (2000). Estimation of disease rates in small areas: A new mixed model for spatial dependence. In *Statistical Models in Epidemiology, the Environment, and Clinical Trials* (pp. 179–191). Springer.
9. **Lindgren, F., Rue, H., & Lindström, J.** (2011). An explicit link between Gaussian fields and Gaussian Markov random fields: The SPDE approach. *Journal of the Royal Statistical Society: Series B*, 73(4), 423–498.
10. **Lloyd, S.** (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129–137.
11. **Okubo, A.** (1980). *Diffusion and Ecological Problems: Mathematical Models*. Springer-Verlag.
12. **Rasmussen, C. E., & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press.
13. **Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H.** (2016). An intuitive Bayesian spatial model for disease mapping that accounts for scaling. *Statistical Methods in Medical Research*, 25(4), 1145–1165.
14. **Roberts, D. R., et al.** (2017). Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure. *Ecography*, 40(8), 913–929.
15. **Turchin, P.** (1998). *Quantitative Analysis of Movement: Measuring and Modeling Population Redistribution in Animals and Plants*. Sinauer Associates.
