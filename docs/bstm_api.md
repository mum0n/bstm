# bstm API Reference

This document provides a detailed technical reference for the internal components of the `bstm` framework. It is intended for developers and advanced users who wish to understand, extend, or debug the framework's core machinery. It covers the component system, formula parsing engine, model configuration pipeline, Turing model definitions, and the posterior reconstruction engine.

## The `ComponentModel` Interface

To extend `bstm` with a new model, a developer must define a new struct that subtypes `ComponentModel` and implement a set of five interface functions. This interface provides the contract between a component and the main `bstm` engine, allowing for seamless integration into the model building, sampling, and post-processing pipeline.
 
 
### 1. `get_precomputes(m, M, mod_data)`
*   **Purpose**: Performs all data-dependent setup and pre-calculations. This method is now the primary entry point for a component to process data and prepare any necessary structures before model generation. It combines the responsibilities previously held by `get_datastructures!` and its own original purpose.
*   **Arguments**:
    *   `m::ComponentModel`: An instance of the component struct.
    *   `M::NamedTuple`: The model configuration object (now read-only). It contains all data structures prepared by `get_datastructures!`.
    *   `mod_data::Dict`: The component's metadata.
*   **Returns**: `NamedTuple`. The results of the pre-computation (e.g., a precision matrix template `Q_template`, its spectral decomposition `U` and `L`, basis matrices, etc.). This `NamedTuple` is stored and made available to the other interface functions via the `spec.hyper` object.
*   **Example Use**: An `icar` component would use this to validate the spatial index and adjacency matrix, and then generate the `Q_template` and its spectral decomposition. A `pspline` component would validate its covariate, and then generate the B-spline basis matrix.


### 2. `get_priors(m, spec, arch, outcome_idx, M)`
*   **Purpose**: Generates the Turing code string for the component's priors.
*   **Arguments**:
    *   `m::ComponentModel`: The component instance.
    *   `spec::NamedTuple`: The full specification for this component instance, including its unique `key` and the `hyper` object returned by `get_precomputes`.
    *   `arch::String`: The model architecture (`"univariate"` or `"multivariate"`).
    *   `outcome_idx::Union{Int, Nothing}`: The index of the outcome variable in a multivariate model.
    *   `M::NamedTuple`: The main model configuration.
*   **Returns**: `String`. A block of Turing.jl code defining the priors.
*   **Example Use**: An `icar` component would generate `sigma_icar_key ~ Exponential(1.0)` and `innovations_icar_key ~ MvNormal(...)`.

### 3. `get_updates(m, spec, arch, outcome_idx, M)`
*   **Purpose**: Generates the Turing code to construct the latent effect and add it to the linear predictor `eta`.
*   **Arguments**: Same as `get_priors`.
*   **Returns**: `String`. A block of Turing.jl code that calculates the latent field and adds it to `eta`.
*   **Example Use**: An `icar` component would generate code to scale the `innovations` by `sigma` and the spectral components to construct the latent field.

### 4. `get_effects(m, chain, M, ...)`
*   **Purpose**: Reconstructs the posterior distribution of the component's effect from the MCMC chain. This function is called during post-processing (e.g., by `model_results_comprehensive` or `predict`).
*   **Arguments**:
    *   `m::ComponentModel`: The component instance.
    *   `chain`: The `MCMCChains.Chains` object from the fitted model.
    *   `M::NamedTuple`: The main model configuration.
    *   `n_samples::Int`: The total number of posterior samples.
    *   `...`: Other arguments related to multivariate models and prediction sets.
*   **Returns**: `NamedTuple`. Typically `(structured=..., noisy=...)`, where each value is a matrix of size `[N_obs x n_samples]` containing the reconstructed posterior effect for each observation and sample.
*   **Example Use**: A `pspline` component would extract the posterior samples for its coefficients (`beta`) and its basis matrix (`B`), and for each sample, compute the effect as `B * beta`.

### A Minimal Example: The IID Component

To illustrate the interface, here is the complete implementation for a simple `IID` (Independent and Identically Distributed) random effect. This component models an unstructured effect $\phi_i \sim \mathcal{N}(0, \sigma^2)$ for each of the $k$ levels of a grouping variable.

**1. Struct Definition**
The struct holds the prior for the single hyperparameter, `sigma`.

```julia
"""
    IID <: ComponentModel

Models an independent and identically distributed (IID) random effect.
"""
struct IID <: ComponentModel
    sigma::UnivariateDistribution
end
```

**2. `get_datastructures!`**
This function validates that the required grouping variable exists in the data.

```julia
function get_datastructures!(m_type::Type{IID}, M::Dict, mod_data::Dict)::Bool
    var_name = mod_data[:variables]
    if !hasproperty(M[:data], var_name)
        error("Grouping variable ':$var_name' for IID model not found in data.")
    end
    return true # Proceed with component creation
end
```

**3. `get_precomputes`**
The IID model requires no data-independent pre-computation, so this function returns an empty `NamedTuple`.

```julia
function get_precomputes(m::IID, M::NamedTuple, mod_data::Dict)::NamedTuple
    return (;)
end
```

**4. `get_priors`**
This function generates the Turing code to define the prior for `sigma` and the standard normal innovations.

```julia
function get_priors(m::IID, spec::NamedTuple, arch::String, outcome_idx, M)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    prior_sigma_str = _distribution_to_string(m.sigma)
    n_levels = M.technical[:component_levels][spec.key]

    return """
    # Priors for IID component: $(spec.key)
    $(v.sigma) ~ $(prior_sigma_str)
    $(v.innovations) ~ MvNormal(zeros($(n_levels)), 1.0)
    """
end
```

**5. `get_updates`**
This function generates the code to scale the innovations by `sigma` and add the effect to the linear predictor `eta`.

```julia
function get_updates(m::IID, spec::NamedTuple, arch::String, outcome_idx, M)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    idx_var = M.technical[:component_indices][spec.key]

    return """
    # Latent effect for IID component: $(spec.key)
    $(v.latent) = $(v.sigma) .* $(v.innovations)
    $(eta_target) .+= $(v.latent)[$(idx_var)]
    """
end
```

**6. `get_effects`**
This function reconstructs the posterior effect by extracting the samples for `sigma` and the `innovations` from the MCMC chain.

```julia
function get_effects(m::IID, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_total)::NamedTuple
    structured_effects = []
    is_multivariate = outcomes_N > 1

    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        
        # Find parameter names in the chain
        sigma_p_name = _find_parameter(p_names, v.sigma, k, is_multivariate)
        innov_p_name = _find_parameter(p_names, v.innovations, k, is_multivariate)

        # Extract posterior samples
        sigma_samples = get_params_vector(chain, sigma_p_name, 1)
        innov_samples = get_params_matrix(chain, innov_p_name)
        
        # Reconstruct the effect for each posterior sample
        latent_field = innov_samples .* sigma_samples'
        
        idx_var = M.technical[:component_indices][spec.key]
        effect_k = latent_field[idx_var, :]
        push!(structured_effects, effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
```

## Available Components

This section details the components available within the `bstm` framework, organized by their primary domain of application. Each component's docstring provides a mathematical summary, a list of required and optional inputs for the `random()` module, and the names of the posterior parameters it produces.

### Spatial Components (Discrete / Areal)

These components are designed for data aggregated into discrete spatial units (areal units or polygons) and rely on a neighborhood graph (`W`) to define spatial relationships.

```julia
"""
    BYM2 <: ComponentModel

The Besag-York-Mollié (BYM2) model, a standard for spatial disease mapping. It
decomposes the spatial effect into a structured component (capturing spatial
clustering) and an unstructured component (capturing random noise).

# Version
v1.0.0 (2026-08-12)

# Mathematical Summary
The BYM2 model represents the spatial random effect \$\\boldsymbol{\\phi}\$ as a sum of
two components: a spatially structured effect \$\\boldsymbol{\\psi}\$ and an
unstructured (IID) effect \$\\boldsymbol{\\epsilon}\$:
\$\\boldsymbol{\\phi} = \\sigma \\left( \\sqrt{\\rho} \\boldsymbol{\\psi}^* + \\sqrt{1-\\rho} \\boldsymbol{\\epsilon} \\right)\$
where:
- \$\\boldsymbol{\\psi}^*\$ is a scaled version of an intrinsic CAR (ICAR) field.
- \$\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(0, I)\$ is standard normal noise.
- \$\\sigma\$ is the overall marginal standard deviation.
- \$\\rho \\in\$ is a mixing parameter that controls the proportion of
  variance attributed to the structured spatial component.

# Computational Methods
- `:spectral` (Default, AD-friendly): Uses a spectral decomposition of the ICAR
  precision matrix. Recommended for gradient-based samplers.
- `:cholesky` (AD-friendly): Uses a dense Cholesky factorization of the
  precision matrix.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky
  factorization.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region_id`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@bstm`.
- **Optional (in `random()` call)**:
  - `sigma`: `UnivariateDistribution`, prior for the marginal standard deviation. Default: `Exponential(1.0)`.
  - `rho`: `UnivariateDistribution`, prior for the mixing parameter. Default: `Beta(1,1)`.
  - `method`: `Symbol`, computational method. Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The overall marginal standard deviation.
- `rho_<key>`: The mixing parameter.
- `innovations_structured_<key>`: Raw standard normal innovations for the structured component.
- `innovations_unstructured_<key>`: Raw standard normal innovations for the unstructured component.
- `latent_<key>`: The reconstructed total spatial effect.
"""
```

```julia
"""
    Leroux <: ComponentModel

A proper Conditional Autoregressive (CAR) model that provides a flexible way to
model spatial autocorrelation. It acts as a bridge between an unstructured IID
model and a fully structured ICAR model.

# Version
v1.0.0 (2026-08-12)

# Mathematical Summary
The Leroux model defines the precision matrix \$\\mathbf{Q}\$ of a spatial field
\$\\boldsymbol{\\phi}\$ as a convex combination of an identity matrix \$\\mathbf{I}\$ and a
scaled ICAR precision matrix \$\\mathbf{Q}_{ICAR}\$:
\$\\mathbf{Q} = (1-\\rho)\\mathbf{I} + \\rho\\mathbf{Q}_{ICAR}\$
The latent field is then modeled as:
\$\\boldsymbol{\\phi} \\sim \\mathcal{N}(0, (\\sigma^2 \\mathbf{Q})^{-1})\$
where:
- \$\\sigma\$ is the marginal standard deviation.
- \$\\rho \\in\$ is a mixing parameter. \$\\rho=0\$ corresponds to an IID
  model, while \$\\rho=1\$ corresponds to an ICAR model.

# Computational Methods
- `:spectral` (Default, AD-friendly): Uses a spectral decomposition of the ICAR
  precision matrix.
- `:cholesky` (AD-friendly): Uses a dense Cholesky factorization.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region_id`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@bstm`.
- **Optional (in `random()` call)**:
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation. Default: `Exponential(1.0)`.
  - `rho`: `UnivariateDistribution`, prior for the mixing parameter. Default: `Beta(1,1)`.
  - `method`: `Symbol`, computational method. Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The marginal standard deviation.
- `rho_<key>`: The mixing parameter.
- `innovations_<key>`: Raw standard normal innovations for the spatial effect.
- `latent_<key>`: The reconstructed latent spatial effect.
"""
```

```julia
"""
    LocalAdaptive <: ComponentModel

A component for a Local Adaptive spatial effect. This model combines a global
smoothing structure (based on a Leroux-style precision matrix) with local,
cluster-specific mean effects. This allows the model to capture both smooth spatial
trends and abrupt shifts between distinct spatial regions.

# Version
v1.1.1 (2026-08-12)

# Mathematical Summary
The `LocalAdaptive` component models a latent spatial field \$\\phi\$ as a non-zero
mean Gaussian Markov Random Field (GMRF). The mean of the field, \$\\boldsymbol{\\mu}\$,
is not constant but varies by spatial cluster, while the precision matrix,
\$\\mathbf{Q}\$, captures global spatial correlation.

\$\\boldsymbol{\\phi} \\sim \\mathcal{N}(\\boldsymbol{\\mu}, (\\sigma^2 \\mathbf{Q})^{-1})\$

1.  **Mean Structure (\$\\boldsymbol{\\mu}\$)**: The spatial domain is partitioned into \$k\$
    clusters using k-means on the area centroids. A separate mean effect, \$\\mu_g\$,
    is estimated for each cluster \$g\$. For any spatial unit \$i\$ belonging to
    cluster \$g\$, its mean is \$\\mu_i = \\mu_g\$. A sum-to-zero constraint is applied
    to the cluster means for identifiability.

2.  **Precision Structure (\$\\mathbf{Q}\$)**: The precision matrix is a proper CAR model
    (Leroux-style), defined as a convex combination of an identity matrix
    \$\\mathbf{I}\$ and a scaled ICAR precision matrix \$\\mathbf{Q}_{ICAR}\$:
    \$\\mathbf{Q} = (1-\\rho)\\mathbf{I} + \\rho\\mathbf{Q}_{ICAR}\$
    This allows the model to smoothly interpolate between unstructured random
    effects (\$\\rho=0\$) and a fully structured ICAR model (\$\\rho=1\$).

# Computational Methods
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the ICAR precision matrix. Recommended for gradient-based samplers.
- `:cholesky` (AD-friendly): Uses a pre-computed dense Cholesky factorization of the
  full Leroux precision matrix.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization,
  which is not compatible with most AD backends.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@bstm`.
  - Spatial coordinates (`s_x`, `s_y`) in the data frame for clustering.
- **Optional (in `random()` call)**:
  - `n_clusters`: `Int`, the number of spatial clusters to identify. Default: `5`.
  - `rho`: A `UnivariateDistribution` for the prior on the mixing parameter. Default: `Beta(1,1)`.
  - `sigma`: A `UnivariateDistribution` for the prior on the overall standard deviation. Default: `Exponential(1.0)`.
  - `method`: A `Symbol` specifying the computational method. Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The overall marginal standard deviation.
- `rho_<key>`: The mixing parameter.
- `innovations_<key>`: The raw standard normal innovations for the centered spatial effect.
- `cluster_innovations_<key>`: The raw standard normal innovations for the cluster means.
- `latent_<key>`: The reconstructed latent spatial field.

# Key References
- Gelfand, A. E., Schmidt, A. M., Banerjee, S., & Sirmans, C. F. (2005).
  *Nonstationary multivariate process modeling through spatially varying
  coregionalization*. Test, 14(2), 263-312.
"""
```

### Spatial Components (Continuous / Geostatistical)

These components are designed for point-referenced data where exact coordinates are available.

```julia
"""
    GP <: ComponentModel

A component model for a full Gaussian Process (GP), also known as Kriging in
geostatistics. It models a latent field by computing a dense covariance
matrix based on a specified kernel function and coordinate inputs.

# Version
v1.2.1 (2026-08-12)

# Mathematical Summary
The component models a latent field \$f(x)\$ as a draw from a Gaussian Process with
a zero mean and a specified covariance function (kernel):
\$f(x) \\sim \\mathcal{GP}(0, k(x, x'))\$

The kernel \$k(x, x')\$ defines the covariance between any two points. For example,
the Squared Exponential (SE) kernel is:
\$k(x, x') = \\sigma^2 \\exp\\left(-\\frac{\\|x - x'\\|^2}{2\\ell^2}\\right)\$
where:
- \$\\sigma^2\$ is the marginal variance.
- \$\\ell\$ is the characteristic lengthscale.

For **anisotropic** models (Automatic Relevance Determination), the squared distance
is weighted by a vector of lengthscales \$\\boldsymbol{\\ell} = [\\ell_1, \\dots, \\ell_D]\$:
\$k(x, x') = \\sigma^2 \\exp\\left(-\\frac{1}{2} \\sum_{d=1}^D \\frac{(x_d - x'_d)^2}{\\ell_d^2}\\right)\$

The model samples the latent field \$f\$ from the resulting multivariate normal
distribution \$f \\sim \\mathcal{N}(0, K)\$, where \$K\$ is the dense covariance matrix
evaluated at all data points.

# Computational Methods
- `:noncentered` (Default, AD-friendly): Samples standard normal innovations and
  transforms them using the Cholesky factor of the covariance matrix. Recommended
  for gradient-based samplers like NUTS.
- `:centered` (Didactic, Not AD-friendly): Samples the latent field directly from
  the `MvNormal` distribution defined by the covariance matrix. This can be less
  efficient for MCMC due to posterior correlations.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`, `y`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `kernel`: `String`, the name of the kernel function (e.g., `"se"`, `"matern32"`). Default: `"se"`.
  - `sigma`: `UnivariateDistribution`, prior for the marginal standard deviation of the GP. Default: `Exponential(1.0)`.
  - `lengthscale`: `UnivariateDistribution` or `Vector{<:UnivariateDistribution}`, prior for the kernel lengthscale(s). Default: `Gamma(2, 0.5)`.
  - `anisotropic`: `Bool`, if `true`, a separate lengthscale is estimated for each input dimension (ARD). Default: `false`.
  - `method`: `Symbol`, computational method (`:noncentered` or `:centered`). Default: `:noncentered`.

# Outputs (Parameter Names)
- `sigma_<key>`: The marginal standard deviation of the GP.
- `ls_<key>`: The kernel lengthscale(s). A vector if anisotropic.
- `innovations_<key>`: The raw standard normal innovations for the latent field (for `:noncentered`).
- `latent_<key>`: The latent field (for `:centered`).
"""
```

### Temporal Components

These components model trends and patterns over time.

```julia
"""
    Harmonic <: ComponentModel

A component model for harmonic temporal effects, capturing periodic patterns using
sine and cosine waves. This component can model one or more harmonics, each with its
own amplitude, phase, and potentially its own period.

# Version
v1.3.1 (2026-08-12)

# Mathematical Summary
The component models a function \$f(t)\$ as a sum of sinusoids. It supports two
parameterizations controlled by the `method` field:

1.  **:twocoefficient (default, AD-friendly)**:
    \$f(t) = \\sum_{k=1}^{N_{harmonics}} \\left( \\beta_{\\cos,k} \\cos\\left(\\frac{2\\pi k t}{P_k}\\right) + \\beta_{\\sin,k} \\sin\\left(\\frac{2\\pi k t}{P_k}\\right) \\right)\$
    This is the recommended method as it is more efficient for gradient-based MCMC.

2.  **:ampphase (didactic)**:
    \$f(t) = \\sum_{k=1}^{N_{harmonics}} A_k \\cos\\left(\\frac{2\\pi k t}{P_k} + \\phi_k\\right)\$
    where \$A_k\$ is the amplitude, \$P_k\$ is the period, and \$\\phi_k\$ is the phase shift.
    This is retained as a more intuitive, didactic alternative.

# Computational Methods
- `:twocoefficient` (Default, AD-friendly): A two-coefficient (sine and cosine)
  parameterization that is efficient for gradient-based samplers.
- `:ampphase` (Didactic, Not AD-friendly): An amplitude-phase parameterization that
  is more intuitive but can be less efficient for MCMC. Retained for didactic purposes.

# Inputs
- **Required**:
  - A seasonal index variable (e.g., `month`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `nharmonics`: `Int`, the number of harmonic terms to include. Default: `1`.
  - `period`: `Real`, `UnivariateDistribution`, or `Vector`. The period(s) of the
    cycle(s). If a single value, it applies to all harmonics. If a vector, its
    length must match `nharmonics`. Default: `12.0`.
  - `amplitude`: `UnivariateDistribution`, prior for the amplitude (for `:ampphase`).
    Default: `Exponential(1.0)`.
  - `phase`: `UnivariateDistribution`, prior for the phase shift (for `:ampphase`).
    Default: `Beta(1,1)`.
  - `method`: `Symbol`, computational method (`:twocoefficient` or `:ampphase`).
    Default: `:twocoefficient`.

# Outputs (Parameter Names)
- `beta_cos_<key>`: Coefficients for the cosine terms (for `:twocoefficient`).
- `beta_sin_<key>`: Coefficients for the sine terms (for `:twocoefficient`).
- `amplitude_<key>`: Amplitudes of the harmonics (for `:ampphase`).
- `phase_<key>`: Phase shifts of the harmonics (for `:ampphase`).
- `period_<key>`: The period of the cycle(s), if estimated.
- `latent_<key>`: The reconstructed latent harmonic effect.
"""
```

### Operators and Specialized Components

These components perform special functions like modeling random effects or combining other components.

```julia
"""
    Mixed <: ComponentModel

An operator component that models random effects (intercepts and/or slopes) for a
specified grouping variable. The correlation structure of the effects is determined
by an inner `ComponentModel`.

# Version
v1.1.1 (2026-08-12)

# Mathematical Summary
The `Mixed` component models effects that vary across the levels of a grouping
variable. It supports both simple (uncorrelated) and correlated random effects.

1.  **Simple Random Effects** (e.g., `random(1 | group)`):
    A single random effect \$\\phi\$ (e.g., an intercept) is modeled for each of the
    \$G\$ levels of the grouping variable. The structure of these effects is
    determined by the inner model. For an `IID` inner model, this is:
    \$\\phi_g \\sim \\mathcal{N}(0, \\sigma^2)\$ for \$g = 1, \\dots, G\$.

2.  **Correlated Random Effects** (e.g., `random(1 + x | group)`):
    A vector of \$K\$ random effects, \$\\boldsymbol{\\beta}_g = [\\beta_{0g}, \\beta_{1g}, \\dots]^T\$,
    is modeled for each group level \$g\$. These effects are assumed to be drawn from
    a multivariate normal distribution with a shared covariance structure:
    \$\\boldsymbol{\\beta}_g \\sim \\mathcal{N}(\\mathbf{0}, \\Sigma)\$
    The covariance matrix \$\\Sigma\$ is decomposed into a set of standard deviations
    \$\\boldsymbol{\\sigma}\$ and a correlation matrix \$\\mathbf{R}\$:
    \$\\Sigma = \\text{diag}(\\boldsymbol{\\sigma}) \\mathbf{R} \\text{diag}(\\boldsymbol{\\sigma})\$
    A prior is placed on the Cholesky factor of \$\\mathbf{R}\$ using the `LKJCholesky`
    distribution. The structure of the effects across group levels (e.g., IID, spatial)
    is determined by the inner `ComponentModel`.

# Computational Methods (for Correlated Effects)
- `:spectral` (default): An efficient, AD-safe method using spectral decomposition of
  the group-level precision matrix.
- `:cholesky`: An AD-safe didactic alternative using dense Cholesky factorization.
- `:cholesky_sparse`: A non-AD-safe didactic method using sparse Cholesky
  factorization, suitable for gradient-free samplers.

# Inputs
- **Required**:
  - A grouping variable (e.g., `group_id`) passed to `random()`.
  - One or more terms for the random effects (e.g., `1` for intercept, `covariate` for slope).
- **Optional (in `random()` call)**:
  - `model`: An inner `ComponentModel` defining the structure across groups (e.g., `iid()`, `ar1()`). Default: `iid()`.
  - `method`: `Symbol`, computational method for correlated effects (`:spectral`, `:cholesky`, `:cholesky_sparse`). Default: `:spectral`.

# Outputs (Parameter Names)
- **Simple Effects**: Same as the inner model (e.g., `sigma_<key>`, `innovations_<key>`).
- **Correlated Effects**:
  - `L_corr_<key>`: The Cholesky factor of the correlation matrix for the effects.
  - `sigma_effects_<key>`: The standard deviations for each random effect term.
  - `innovations_<key>`: The raw standard normal innovations for the coefficients.
"""
```

## Internal Engines

### Formula Parsing Engine

The formula parser translates the user-provided formula string into a structured representation that the configuration engine can process.

*   **`decompose_bstm_formula(formula_str)`**: This is the main entry point. It splits the formula into its Left-Hand Side (LHS) and Right-Hand Side (RHS).
    *   **LHS**: Parsed to identify outcome variables and their likelihood specifications (e.g., `likelihood(y, family=poisson)`).
    *   **RHS**: Pre-processed to handle intercept control (`-1`, `0`, `intercept(false)`) and to normalize all bare terms (e.g., `z`) into explicit `fixed(z)` module calls. It is then parsed by `_parse_rhs_expression` into an Abstract Syntax Tree (AST).
    *   **Output**: Returns a `NamedTuple` containing `:outcomes`, `:modules` (a dictionary of all parsed RHS terms), `:fixed_effects` (a list of bare variable names parsed from the RHS), `:has_intercept`, and `:intercept`.

*   **`_parse_rhs_expression(term_str)`**: A recursive descent parser that respects operator precedence to build the AST for the RHS. The precedence is:
    The parser is called on sub-expressions after the formula has been split by the `+` operator (which has the lowest precedence). The parser then handles operators in the following order of precedence (from highest to lowest):
    1.  `|>` (Pipe for state-space models)
    2.  `⊗` (Kronecker Product)
    3.  `∘` (Composition, e.g., for point process models)

*   **`_categorize_rhs_nodes!(nodes, modules, fixed_effects)`**: Traverses the generated AST to populate the `modules` dictionary and the `fixed_effects` list. It correctly identifies composed components (e.g., `random(...) ⊗ random(...)`) as a single interaction module.

*   **`_parse_single_component_term` & `_parse_arguments_string`**: Helper functions that parse individual module calls (e.g., `random(s_idx, model=:bym2)`) into a dictionary of variables and parameters. These functions correctly handle Julia `Symbol` literals (e.g., `:besag`) passed as arguments.

#### Formula Operators in Practice

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

### Model Configuration Engine

The `bstm_config` function is the main engine that transforms the parsed formula and data into a complete configuration object (`M`) for the Turing models.

#### `bstm_config` Workflow

1.  **Initialization**: `_initialize_config` creates the base `M` dictionary, populating it with the input `data` and keyword arguments.
2.  **LHS Processing**: `_process_lhs!` processes the `outcomes` from the parser, sets the model architecture (`univariate`, `multivariate`), and resolves observation-level parameters like offsets and weights.
3.  **RHS Module Processing**: This is the core loop that iterates over the `modules` dictionary from the parser. For each module, it performs:
    *   **Processor Dispatch**: Calls the appropriate function from the `MODULE_PROCESSORS` dictionary (e.g., `process_random_module!`). These functions handle data-dependent setup, such as creating spatial indices or basis matrices.
	    *   **Primitive Resolution**: `resolve_technical_primitive` is called to convert the parsed module data (a `Dict`) into a concrete `ComponentModel` struct instance (e.g., `BYM2(...)`). This step also resolves hyperpriors using `resolve_hyperpriors`.
	    *   **Pre-computation**: `get_precomputes` is called on the `ComponentModel` object. This function is a factory that generates the technical specifications needed for the model, such as precision matrix templates or basis matrices.
	    *   **Registration**: The complete component specification (including the `ComponentModel` object and its precomputed `hyper` object) is added to `M[:components]`.
	*   **Fixed Effects Processing**:
	    *   `_process_fixed_effects!`: Consolidates fixed effect variables from both bare terms (parsed directly from the formula) and explicit `fixed()` module calls.
	*   **Intercept Resolution**: The final decision on whether to include an intercept (`M[:add_intercept]`) is prioritized from the `intercept()` module. If no `intercept()` module is present, it defaults to the legacy numeric flags (`1`, `0`, `-1`) parsed from the formula string.
4.  **Finalization**: `_finalize_config!` ensures all necessary keys exist in `M`, providing defaults where needed.

### Turing Model Generation

The `bstm` framework uses a code generation engine (`bstm_text_assembler`) to dynamically construct a Turing `@model` definition from the configuration object `M`. This approach avoids world-age issues and allows for highly flexible model specifications.

#### `bstm_text_assembler(config, model_func_name)`

This function assembles the model code string by iterating through the components and calling their `get_priors` and `get_updates` methods.

*   **Generated Code Structure**:
    *   Defines global likelihood parameters (e.g., `lik_r` for Negative Binomial).
    *   Defines priors for the intercept and fixed effects.
    *   Initializes the linear predictor `eta`.
    *   **Component Priors Loop**: Iterates through `config.components` and calls `get_priors` for each one, appending the prior definitions.
    *   **Component Updates Loop**: Iterates through `config.components` and calls `get_updates` for each one, appending the code that constructs the latent effect and adds it to `eta`.
    *   Defines the final observation likelihood using `y_obs ~ bstm_Likelihood(...)`.

### Posterior Reconstruction Engine

The reconstruction engine is responsible for post-processing the MCMC `chain` to produce interpretable summaries and predictions.

*   **`_discover_effects(...)`**: This is the core discovery function.
    *   It initializes containers for all possible latent effects (spatial, temporal, etc.).
    *   It iterates through the `M[:components]` specification. For each component, it calls `extract_component`.
    *   **`get_effects(m_obj, ...)`**: This function dispatches on the `ComponentModel` type (`m_obj`). Each method knows how to find its parameters in the chain and reconstruct its specific effect.
        *   For simple components like `BYM2`, it finds `sigma`, `rho`, and the raw innovations from the chain and re-runs the logic from `get_updates` to reconstruct the effect for each posterior sample.
        *   For a `Composed` component with a `kronecker_product` operator, it reconstructs the interaction field by finding the corresponding latent field and hyperparameters in the chain and applying the correct scaling and reshaping.

*   **`_modular_eta_assembly(...)`**: Takes the `registry` of discovered fields and reassembles the full linear predictor `eta` for each posterior sample. This process mirrors the assembly logic within the Turing model itself but operates on the posterior samples. It correctly handles both in-sample (`M`) and out-of-sample (`PS`) data.

## Advanced Topics

### Spatial Partitioning

For discrete spatial models (GMRFs), the continuous spatial domain must be discretized into "Areal Units" (AUs). The `assign_spatial_units` function provides several methods for this, balancing geometric compactness with statistical information density.

| Method | Description | Justification |
|:---|:---|:---|
| `:cvt` | **Centroidal Voronoi Tessellation** | Iteratively minimizes variance to create geometrically regular cells. |
| `:kvt` | **K-Means Voronoi Tessellation** | Uses K-Means to create units with a balanced number of observations. |
| `:avt` | **Agglomerative Voronoi** | A bottom-up approach that merges small units to prevent data starvation. |
## Advanced Topics

### Spatial Partitioning

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

### Mechanistic Models with `dynamics()`

The `dynamics()` module provides a powerful interface for embedding process-based, mechanistic models directly into the spatiotemporal framework. Unlike statistical models like `AR1` or `RW2` which describe correlation, `dynamics()` models describe the *evolution* of a latent field from one time step to the next based on a predefined equation.

This is accomplished by defining a latent spatiotemporal field, `dyn_field[space, time]`, where the state at time `t` is a function of the state at time `t-1`. For example, a simple advection model implements the state transition:

`dyn_field[:, t] ~ MvNormal(dyn_field[:, t-1] - velocity * L * dyn_field[:, t-1], noise)`

where `L` is the graph Laplacian. This allows the model to learn physical parameters like `velocity` within a fully Bayesian context.

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

### Multi-fidelity and Nested Models

The `nested()` module is a "supervisor" component for multi-fidelity modeling. It allows you to define a complete sub-model that is fit to a separate (often larger, lower-quality) dataset. The latent effect from this sub-model is then incorporated as a calibrated predictor into the main model, allowing the main model to "learn" from the proxy data. The `nested()` module accepts a full formula string, including a `likelihood()` block, which enables the specification of independent likelihoods for each fidelity level.

```julia
@bstm(
    likelihood(y_hq) ~ intercept() + random(s_idx, model=icar) + nested(proxy_signal, formula="likelihood(y_lq, family=poisson) ~ intercept() + random(x, model=pspline)", data_source=low_quality_data),
    high_quality_data,
    low_quality_data = df_low_quality
)
```

#### `nested()` Module Reference

| Keyword / Parameter     | Example Usage                                                        | Data Type | Default            | Meaning & Assumptions                                                                                                                                                                                                                                 |
| :------------------------| :---------------------------------------------------------------------| :----------| :-------------------| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nested()`              | `nested(z_var; ...)`                                                 | Module    | N/A                | Defines a supervised sub-model whose latent effect is added to the main model's linear predictor. The `z_var` is a symbolic name for this component.                                                                                                  |
| `formula`               | `formula="likelihood(z, family=gaussian) ~ intercept() + random(s)"` | `String`  | `""`               | A complete `bstm` formula string that defines the structure of the sub-model, including its own likelihood. This sub-model is fit to the specified `data_source`.                                                                                     |
| `data_source`           | `data_source=proxy_data`                                             | `Symbol`  | `:data`            | A symbol pointing to a `DataFrame` passed as a keyword argument to the main `bstm()` call. This allows the sub-model to use a different dataset.                                                                                                      |
| `rho_nested` (Implicit) | N/A                                                                  | `Float`   | `Normal(1.0, 0.5)` | A scaling coefficient that links the sub-model's latent effect to the main model's linear predictor: $\eta_{\text{main}} = \dots + \rho_{\text{nested}} \cdot \eta_{\text{sub}}$. The prior assumes the sub-model is a good proxy ($\rho \approx 1$). |

### Bayesian Factor Analysis with `eigen()`

The `eigen()` module implements a Bayesian Principal Component Analysis (PCA) to perform dimensionality reduction on a set of multivariate outcomes. It decomposes the input variables into a smaller set of orthogonal latent factors. The framework uses a Householder transformation to construct the orthonormal loadings matrix, ensuring numerical stability and efficient sampling.

#### `eigen()` Module Reference

| Keyword / Parameter | Example Usage              | Data Type      | Default            | Meaning & Assumptions                                                                                                                                                               |
| :--------------------| :---------------------------| :---------------| :-------------------| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `eigen()`           | `eigen(y1, y2, y3; ...)`   | Module         | N/A                | Defines a Bayesian PCA factor model. The variables listed (e.g., `y1, y2, y3`) are the multivariate outcomes to be decomposed.                                                      |
| `n_factors`         | `n_factors=1`              | `Int`          | `1`                | The number of latent factors (principal components) to extract. This determines the dimensionality of the reduced latent space.                                                     |
| `pca_sd`            | `pca_sd=Exponential(0.5)`  | `Distribution` | `Exponential(1.0)` | The prior for the standard deviations of the principal components (latent factors). These are the "eigenvalues" of the system, controlling the variance explained by each factor.   |
| `pdef_sd`           | `pdef_sd=Exponential(0.5)` | `Distribution` | `Exponential(1.0)` | The prior for the standard deviation of the residual (uniqueness) noise. This captures the variance in each observed variable that is *not* explained by the shared latent factors. |

### Handling Censored Covariates via Joint Modeling

A censored covariate is a predictor variable for which the true value is not always known, but is instead confined to an interval (e.g., $x_{true} > c$). The statistically robust approach to this "errors-in-variables" problem is to treat the censored covariate as a latent variable and model it jointly with the primary outcome.

The `bstm` framework facilitates this through the `nested()` module, which allows for the construction of a joint model in a single step. This approach simultaneously estimates the model for the censored covariate and the main outcome model, correctly propagating all sources of uncertainty. The `nested()` module accepts a full formula string, including a `likelihood()` block, which enables the specification of independent likelihoods for each fidelity level.

#### Implementation with `nested()`

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

### Mechanistic Models with `dynamics()`

The `dynamics()` module provides a powerful interface for embedding process-based, mechanistic models directly into the spatiotemporal framework. Unlike statistical models like `AR1` or `RW2` which describe correlation, `dynamics()` models describe the *evolution* of a latent field from one time step to the next based on a predefined equation.

This is accomplished by defining a latent spatiotemporal field, `dyn_field[space, time]`, where the state at time `t` is a function of the state at time `t-1`. For example, a simple advection model implements the state transition:

`dyn_field[:, t] ~ MvNormal(dyn_field[:, t-1] - velocity * L * dyn_field[:, t-1], noise)`

where `L` is the graph Laplacian. This allows the model to learn physical parameters like `velocity` within a fully Bayesian context.

### Multi-fidelity and Nested Models

The `nested()` module is a "supervisor" component for multi-fidelity modeling. It allows you to define a complete sub-model that is fit to a separate (often larger, lower-quality) dataset. The latent effect from this sub-model is then incorporated as a calibrated predictor into the main model, allowing the main model to "learn" from the proxy data. The `nested()` module accepts a full formula string, including a `likelihood()` block, which enables the specification of independent likelihoods for each fidelity level.

```julia
@bstm(
    likelihood(y_hq) ~ intercept() + random(s_idx, model=icar) + nested(proxy_signal, formula="likelihood(y_lq, family=poisson) ~ intercept() + random(x, model=pspline)", data_source=low_quality_data),
    high_quality_data,
    low_quality_data = df_low_quality
)
```

#### `nested()` Module Reference

| Keyword / Parameter     | Example Usage                                                        | Data Type | Default            | Meaning & Assumptions                                                                                                                                                                                                                                 |
| :------------------------| :---------------------------------------------------------------------| :----------| :-------------------| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nested()`              | `nested(z_var; ...)`                                                 | Module    | N/A                | Defines a supervised sub-model whose latent effect is added to the main model's linear predictor. The `z_var` is a symbolic name for this component.                                                                                                  |
| `formula`               | `formula="likelihood(z, family=gaussian) ~ intercept() + random(s)"` | `String`  | `""`               | A complete `bstm` formula string that defines the structure of the sub-model, including its own likelihood. This sub-model is fit to the specified `data_source`.                                                                                     |
| `data_source`           | `data_source=proxy_data`                                             | `Symbol`  | `:data`            | A symbol pointing to a `DataFrame` passed as a keyword argument to the main `bstm()` call. This allows the sub-model to use a different dataset.                                                                                                      |
| `rho_nested` (Implicit) | N/A                                                                  | `Float`   | `Normal(1.0, 0.5)` | A scaling coefficient that links the sub-model's latent effect to the main model's linear predictor: $\eta_{\text{main}} = \dots + \rho_{\text{nested}} \cdot \eta_{\text{sub}}$. The prior assumes the sub-model is a good proxy ($\rho \approx 1$). |


### Handling Censored Covariates via Joint Modeling

A censored covariate is a predictor variable for which the true value is not always known, but is instead confined to an interval (e.g., $x_{true} > c$). The statistically robust approach to this "errors-in-variables" problem is to treat the censored covariate as a latent variable and model it jointly with the primary outcome.

The `bstm` framework facilitates this through the `nested()` module, which allows for the construction of a joint model in a single step. This approach simultaneously estimates the model for the censored covariate and the main outcome model, correctly propagating all sources of uncertainty. The `nested()` module accepts a full formula string, including a `likelihood()` block, which enables the specification of independent likelihoods for each fidelity level.

#### Implementation with `nested()`

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


### Bayesian Factor Analysis with `eigen()`

The `eigen()` module implements a Bayesian Principal Component Analysis (PCA) to perform dimensionality reduction on a set of multivariate outcomes. It decomposes the input variables into a smaller set of orthogonal latent factors. The framework uses a Householder transformation to construct the orthonormal loadings matrix, ensuring numerical stability and efficient sampling.

#### `eigen()` Module Reference

| Keyword / Parameter | Example Usage              | Data Type      | Default            | Meaning & Assumptions                                                                                                                                                               |
| :--------------------| :---------------------------| :---------------| :-------------------| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `eigen()`           | `eigen(y1, y2, y3; ...)`   | Module         | N/A                | Defines a Bayesian PCA factor model. The variables listed (e.g., `y1, y2, y3`) are the multivariate outcomes to be decomposed.                                                      |
| `n_factors`         | `n_factors=1`              | `Int`          | `1`                | The number of latent factors (principal components) to extract. This determines the dimensionality of the reduced latent space.                                                     |
| `pca_sd`            | `pca_sd=Exponential(0.5)`  | `Distribution` | `Exponential(1.0)` | The prior for the standard deviations of the principal components (latent factors). These are the "eigenvalues" of the system, controlling the variance explained by each factor.   |
| `pdef_sd`           | `pdef_sd=Exponential(0.5)` | `Distribution` | `Exponential(1.0)` | The prior for the standard deviation of the residual (uniqueness) noise. This captures the variance in each observed variable that is *not* explained by the shared latent factors. |



## Conclusion

The `bstm` framework provides a powerful and extensible environment for Bayesian spatiotemporal modeling within the Julia ecosystem. By leveraging a modular, component-based architecture and a formula-driven interface, it bridges the gap between high-level model specification and the low-level performance of the Turing.jl probabilistic programming language. The `ComponentModel` interface is central to this design, offering a clear and consistent contract for developers to integrate novel statistical structures, from simple random effects to complex mechanistic models.

This document has detailed the internal machinery of the framework, from the formula parsing and model configuration engines to the code generation and posterior reconstruction pipelines. By exposing these internals, `bstm` aims to empower advanced users and developers to not only use the available components but also to extend the framework to meet new research challenges. The combination of discrete GMRFs, continuous Gaussian Processes, and advanced approximation techniques provides a rich toolbox for tackling a wide array of problems in ecological modeling and beyond.

## References

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
