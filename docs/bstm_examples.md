---
title: "bstm Model Examples"
format: html
---

# `bstm` Model Examples

This document provides a comprehensive library of model specifications using the `@bstm()` formula interface. It is organized from simple to complex, covering a wide range of features from basic regression to advanced hierarchical and mechanistic models.

## 1. Basic Regression & Mixed Effects

These examples show how to specify standard regression components.

### 1.1. Fixed Effects Model (Linear Regression)

A simple linear regression model with an intercept and two continuous covariates.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + cov1 + cov2,
    data
)
```

### 1.2. Categorical Fixed Effects with Contrasts

Models the effect of a categorical variable using custom contrast coding.

**Key Features:**
- **Module**: `fixed()`
- **Contrast Coding**: `:effects` coding sets the sum of coefficients to zero.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + fixed(region, contrast=:effects, prior=Normal(0, 10)),
    data
)
```

### 1.3. Random Intercept Model

Models group-level variability in the intercept.

**Key Features:**
- **Module**: `mixed()`
- **Random Intercept**: `mixed(1 | group)`

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + cov1 + mixed(1 | site),
    data
)
```

### 1.4. Random Slope and Intercept Model

Models group-level variability for intercepts and the effects of covariates.

**Key Features:**
- **Module**: `mixed()`
- **Random Slope**: `mixed(covariate | group)`

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + cov1 + 
        mixed(1 + cov1 | site), # Correlated random intercept and slope for cov1
    data
)
```

## 2. Likelihood Features

These examples demonstrate how to modify the observation model using parameters within the `likelihood()` module.

### 2.1. Censored Data Model

Models a continuous outcome where some observations are censored.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y_obs, family=gaussian, y_L=y_lower_bound, y_U=y_upper_bound) ~ 1,
    data
)
```

### 2.2. Zero-Inflated Model

Models count data with an excess of zeros.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y_counts, family=poisson, zi=true) ~ 1 + cov1,
    data
)
```

### 2.3. Hurdle Model

A two-part model where the process for generating zeros is separate from the process for generating positive counts.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y_counts, family=poisson, hurdle=0) ~ 1 + cov1,
    data
)
```

### 2.4. Stochastic Volatility Model

Models observation noise that varies over space and time.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y, family=gaussian, volatility=true) ~ 1 + spatial(s_idx, model=bym2),
    data, W=W
)
```

## 3. Spatial Models

### 3.1. Areal Data Models (GMRFs)

These models are for data aggregated over discrete spatial units (polygons).

#### BYM2 Disease Mapping Model

These examples demonstrate various models for capturing spatial autocorrelation.

The standard for areal disease mapping, decomposing spatial risk into structured and unstructured components.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y_counts, family=poisson) ~ 1 + spatial(s_idx, model=bym2, W=W),
    data
)
```

#### ICAR / Besag Model

A model for strong spatial smoothing based on local neighbors.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + spatial(s_idx, model=icar, W=W),
    data
)
```

#### Leroux Model

Alternatives to BYM2 that offer different parameterizations of spatial correlation.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + spatial(s_idx, model=leroux, W=W),
    data
)
```

#### SAR Model

Models spatial "spill-over" effects where the value at one location directly influences its neighbors.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + spatial(s_idx, model=sar, W=W),
    data
)
```

### 3.2. Continuous & Point-Reference Models

These models are for data where exact coordinates are available.

#### Gaussian Process (GP)

The gold-standard for continuous spatial modeling, but computationally expensive.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + smooth(lon, lat, model=gp, kernel="matern32"),
    data
)
```

#### SPDE Model

Models a continuous spatial process using an approximation to a Stochastic Partial Differential Equation, linked to the Matérn kernel.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + spatial(s_idx, model=spde, mesh=mesh_object),
    data, mesh=mesh_object # Requires a pre-computed mesh
)
```

## 4. Temporal & Seasonal Models

### 4.1. Temporal Trend Models

#### Smooth Temporal Trend (Random Walk)

Models a smooth, non-linear temporal trend using a second-order random walk.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + temporal(time_index, model=rw2),
    data
)
```

#### Autoregressive Model (AR1)

Models a stationary temporal process where the current value depends on the immediately preceding value.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + temporal(time_index, model=ar1),
    data
)
```

### 4.2. Seasonal Models

#### Harmonic Seasonality

Captures periodic effects using sine and cosine basis functions.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + seasonal(month, model=harmonic, period=12),
    data
)
```

#### Cyclic Random Walk

Models a smooth, periodic effect where the end of the cycle connects to the beginning.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + seasonal(day_of_week, model=cyclic, period=7),
    data
)
```

## 5. Covariate Smoothing (`smooth`)

### 5.1. 1D P-Spline Smoother

Models the non-linear effect of a continuous covariate using penalized splines.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + smooth(temperature, model=pspline, nbins=20),
    data
)
```

### 5.2. 2D Thin Plate Spline

Models the smooth, non-linear interaction of two continuous covariates (e.g., spatial coordinates).

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + smooth(longitude, latitude, model=tps, nbins=50),
    data
)
```

## 6. Interaction & Hierarchical Models

### 6.1. Separable Spatiotemporal Model

A standard model where the spatial and temporal effects are assumed to be independent and additive.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + spatial(s_idx, model=bym2, W=W) + temporal(t_idx, model=ar1),
    data
)
```

### 6.2. Spatiotemporal Interaction Model (Knorr-Held Type IV)

A fully structured interaction where a spatial field (e.g., ICAR) evolves over time according to a temporal process (e.g., AR1).

**Formula Equivalent (using `⊗`):**
```julia
m = @bstm(
    likelihood(y) ~ 1 + spatial(s_idx, model=icar) + temporal(t_idx, model=ar1) +
        (spatial(s_idx, model=icar) ⊗ temporal(t_idx, model=ar1)),
    data, W=W
)
```

**Formula Equivalent (using `spacetime`):**
```julia
m = @bstm(
    likelihood(y) ~ 1 + spatial(s_idx, model=icar) + temporal(t_idx, model=ar1) +
        spacetime(s_idx, t_idx, model=(icar, ar1)),
    data, W=W
)
```

### 6.3. Spatially Varying Coefficients (SVC)

Allows the effect of a covariate to vary smoothly across space.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + (poverty |> spatial(s_idx, model=icar, W=W)),
    data
)
```

### 6.4. Spatially Varying Curves

Models a non-linear trend of a covariate that varies smoothly across space.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + (smooth(time, model=pspline) |> spatial(s_idx, model=icar, W=W)),
    data
)
```

## 7. Advanced & Mechanistic Models

### 7.1. Mechanistic Dynamics (Advection-Diffusion)

A mechanistic model for a process that is transported (advection) and spreads (diffusion) over a graph.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(concentration) ~ 1 + dynamics(s_idx, t_idx, model=advection_diffusion, W=W),
    data
)
```

### 7.2. Bayesian PCA (`eigen`)

Performs dimensionality reduction on a set of covariates, using the dominant latent factor as a predictor.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y) ~ 1 + eigen(temp, salinity, depth, n_factors=1),
    data
)
```

### 7.3. Multi-fidelity Model (`nested`)

Integrates a low-fidelity (but data-rich) proxy variable to improve predictions for a high-fidelity (but data-sparse) target.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y_hq) ~ 1 + 
        nested(
            proxy_model, 
            formula="y_lq ~ 1 + smooth(x, model=pspline)"
        ),
    data
)
```

## 8. Multivariate Models

### 8.1. Multivariate CAR Model

A multivariate CAR model for jointly modeling multiple correlated spatial processes.

**Formula Equivalent:**
```julia
m = @bstm(
    y1 + y2 ~ 1 + spatial(s_idx, model=besag, W=W) + temporal(t_idx, model=ar1),
    data
)
```

### 8.2. Joint Model with Different Likelihoods

Jointly models multiple outcomes where each has a different likelihood.

**Formula Equivalent:**
```julia
m = @bstm(
    likelihood(y_counts, family=poisson) + likelihood(y_continuous, family=gaussian) ~ 
        1 + spatial(s_idx, model=bym2, W=W),
    data
)
```