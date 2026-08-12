 
# bstm: Bayesian Spatiotemporal Models in Julia

 
`bstm` is a Julia library for Bayesian spatiotemporal modeling, built on the Turing.jl probabilistic programming framework and many, many other Julia libraries. It provides a high-level, formula-based interface inspired by R's `brms` and `lme4` to simplify the specification of complex hierarchical models. The framework is designed for composability, allowing users to combine spatial, temporal, and mechanistic components to analyze complex datasets, particularly in fields like ecology and epidemiology.

## Key Features

*   **Formula-Based Interface**: Use the `@bstm` macro for an intuitive, R-like syntax to define complex models.
*   **Rich Component Library**: A wide range of built-in components for modeling various effects:
    *   **Spatial Effects**: Includes discrete GMRF models (`ICAR`, `BYM2`, `Leroux`, `SAR`) for areal data and continuous-space models (`GP`, `SPDE`) for point-referenced data.
    *   **Temporal Effects**: Supports autoregressive models (`AR1`), random walks (`RW1`, `RW2`), and general-purpose smoothers for capturing trends.
    *   **Spatiotemporal Interactions**: Implements Knorr-Held interaction types (`I` through `IV`) via the Kronecker product (`⊗`) operator for modeling complex dependencies between space and time.
    *   **Smoothers**: Provides P-splines, thin-plate splines, and other basis function expansions for modeling non-linear covariate effects.
*   **Scalable Approximations for Large Data**:
    *   **Random Fourier Features (RFF)**: Approximates stationary kernels to scale Gaussian Processes to large datasets.
    *   **SPDE (Stochastic Partial Differential Equation)**: Links continuous Matérn fields to discrete GMRFs for efficient inference.
    *   **Inducing Point Methods (Nyström / FITC)**: Creates low-rank approximations of a full GP.
    *   **FFT & Wavelet**: For regularly gridded data, leverages spectral methods for highly efficient filtering and multi-resolution analysis.
*   **Advanced Hierarchical Modeling**:
    *   **Spatially-Varying Coefficients (SVC)**: Allows covariate effects to vary across space using the `|>` operator (e.g., `covariate |> random(s_idx, model=icar)`).
    *   **Multi-fidelity Data Fusion**: Integrates low-resolution, data-rich proxy variables to improve high-resolution predictions using the `nested()` module.
    *   **Bayesian PCA**: Performs dimensionality reduction on a set of covariates or outcomes with the `eigen()` module.
    *   **Mechanistic Models**: Embeds process-based differential equation models (e.g., for population dynamics) directly into the statistical framework with the `dynamics()` module.
*   **Flexible Observation Models**: Supports numerous likelihoods (`Poisson`, `Binomial`, `Gaussian`, `NegativeBinomial`, etc.) and advanced features like zero-inflation, hurdle models, and spatiotemporal stochastic volatility.
*   **Principled Prior Specification**: Features Penalized Complexity (PC) priors as the default to ensure model identifiability and prevent overfitting, with an intuitive syntax for setting constraints.

## Installation and Setup

`bstm` is designed as a self-contained project environment rather than a standard Julia package. The recommended way to use it is to clone the repository and activate its environment.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/mum0n/bstm.git
    cd bstm
    ```

2.  **Set up the Julia Environment**
    Start a Julia session from within the `bstm` directory. The following script will activate the project, install all necessary dependencies (which may take some time on the first run), and load the framework's functions.

```julia
# Set the path to your cloned bstm repository
project_directory = pwd() # Assumes you started Julia from the 'bstm' directory

# Activate the project and load all functions
include(joinpath(project_directory, "startup.jl"))
``` 

## Quick start

This example demonstrates a standard spatiotemporal model using the Scottish Lip Cancer dataset, which has been extended with a simulated temporal component.

```julia

# Set a seed for reproducibility
Random.seed!(42)

# 1. Load Data
# The dataset includes cancer counts (y), population offsets (log_offset),
# a covariate (X), and spatial/temporal indices.
data_scot, _ = scottish_lip_cancer_data_spacetime()
df = data_scot.data
W = data_scot.au.W

# 2. Define a Spatiotemporal Model
# This model uses a BYM2 prior for spatial effects and an AR1 process for time.
m = @bstm(
    likelihood(y, family=poisson, log_offsets=log_offset) ~
        intercept() +
        fixed(X) +
        random(s_idx, model=bym2) +
        random(year, model=ar1),
    df,
    W = W,
    verbose = false # Suppress verbose output
)

# 3. Sample from the Posterior
# For a real analysis, use a more robust sampler like NUTS and more samples.
chain = sample(m, MH(), 1000; progress=false)

# 4. View Results
println(chain)
```

## Documentation

For a deeper dive into the framework's architecture, components, and API, please see the detailed documentation:

- Architectural Overview: A high-level guide to the design principles, formula interface, and core concepts of the bstm framework. [bstm_overview.md](docs/bstm_overview.md) 

- Technical API Reference: A detailed reference for internal components, including the component system, formula parser, model configuration engine, and posterior reconstruction tools. bstm_api.md 
 
  
## Contributing

This is a personal project that I use to facilitate my research. There will be errors and issues. You can report bugs and their solutions, etc. and I may address them if I have the time. But forking it and modifying it to your needs might be the fastest route...


## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
