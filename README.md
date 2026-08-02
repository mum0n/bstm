 
# bstm: Bayesian Spatiotemporal Models in Julia

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/mum-n/bstm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`bstm` is a Julia library for Bayesian spatiotemporal modeling, built on the Turing.jl probabilistic programming framework and many many other Julia libraries. It provides a high-level, formula-based interface inspired by R's `brms` and `lme4` to simplify the specification of complex hierarchical models. The framework is designed for composability, allowing users to combine spatial, temporal, and mechanistic components to analyze complex datasets, particularly in fields like ecology and epidemiology.

## Key Features

*   **Formula-Based Interface**: Use the `@bstm` macro for an intuitive, R-like syntax to define complex models.
*   **Rich Component Library**: A wide range of built-in components for modeling:
    *   **Spatial Effects**: Discrete GMRF models (`ICAR`, `BYM2`, `Leroux`, `SAR`) and continuous-space models (`GP`, `SPDE`).
    *   **Temporal Effects**: Autoregressive models (`AR1`), random walks (`RW1`, `RW2`), and smoothers.
    *   **Spatiotemporal Interactions**: Knorr-Held interaction types (`I` through `IV`) via the Kronecker product (`⊗`).
    *   **Smoothers**: P-splines, thin-plate splines, and other basis function expansions for non-linear covariate effects.
*   **Scalable Approximations**: Advanced components for large datasets, including:
    *   **Random Fourier Features (RFF)**: Approximates a stationary kernel by mapping input coordinates into a randomized feature space, transforming the GP into a more scalable Bayesian linear regression problem.
    *   **SPDE (Stochastic Partial Differential Equation)**: Models a continuous Matérn field by linking it to a discrete GMRF, allowing for scalable inference via sparse precision matrices.
    *   **Nyström / FITC (Inducing Point Methods)**: Creates a low-rank approximation of a full GP by summarizing the data through a small set of "inducing points."
    *   **FFT (Fast Fourier Transform)**: For regularly gridded data, leverages the DFT to perform spatial filtering in $O(N \log N)$ time.
    *   **Wavelet**: Provides a multi-resolution analysis of a field, useful for capturing both broad trends and localized, high-frequency details.
*   **Hierarchical & Advanced Modeling**:
    *   **Spatially-Varying Coefficients**: Model covariate effects that change across space using the `|>` operator.
    *   **Multi-fidelity Data Fusion**: Integrate low-resolution, data-rich proxy variables to improve high-resolution predictions using the `nested()` module.
    *   **Bayesian PCA**: Perform dimensionality reduction on outcomes with the `eigen()` module.
    *   **Mechanistic Models**: Embed process-based differential equation models with the `dynamics()` module.
*   **Flexible Likelihoods**: Support for numerous observation models (`Poisson`, `Binomial`, `Gaussian`, `NegativeBinomial`, etc.), including features like zero-inflation, hurdle models, and stochastic volatility.
*   **Principled Priors**: A robust system for prior specification, featuring Penalized Complexity (PC) priors as the default to ensure model identifiability and prevent overfitting.

## Installation

The project can be installed directly from the GitHub repository.

```julia
using Pkg
Pkg.add(url="https://github.com/mum0n/bstm.git")

# Clone the repository first
# git clone https://github.com/mum0n/bstm.git

# Define project directory
project_directory = "path/to/your/cloned/bstm"

# Activate and load
include(joinpath(project_directory, "startup.jl"))
load_project_functions(srcdir())
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