---
title: "bstm Model Examples"
format: html
---

# `bstm` Model Examples

This document provides a comprehensive library of model specifications using the `@bstm()` formula interface. It is organized from simple to complex, covering a wide range of features from basic regression to advanced hierarchical and mechanistic models.

## 0. Example data

```julia
project_directory = joinpath( "C:\\", "home", "jae", "projects", "bstm")  
include( joinpath( project_directory, "startup.jl" ) ) 
load_project_functions( srcdir() )
data_scot, _ = scottish_lip_cancer_data_spacetime(); # additional noise and "fake" time slices added
data = data_scot[:data];  
W = data_scot[:au][:W]
```

## 1. Basic Regression & Mixed Effects

These examples show how to specify standard regression components.

### 1.1. Fixed Effects Model (Linear Regression)

A simple linear regression model with an intercept and two continuous covariates.


```julia
m = @bstm( likelihood(y) ~ intercept() + fixed(cov1) + fixed(cov2), data )
```
 
### 1.2. Categorical Fixed Effects with Contrasts

Models the effect of a categorical variable using custom contrast coding.

**Key Features:**
- **Module**: `fixed()`
- **Contrast Coding**: `effects` coding sets the sum of coefficients to zero.


```julia
m = @bstm( likelihood(y) ~ intercept() + fixed(region, contrast=effects, prior=Normal(0, 10)), data )
```

### 1.3. Random Intercept Model

Models group-level variability in the intercept.

**Key Features:**
- **Module**: `mixed()`
- **Random Intercept**: `mixed(1 | group)`


```julia
m = @bstm( likelihood(y) ~ intercept(false) + fixed(cov1) + mixed( intercept() | region), data )
```

### 1.4. Random Slope and Intercept Model

Models group-level variability for intercepts and the effects of covariates.

**Key Features:**
- **Module**: `mixed()`
- **Random Slope**: `mixed(covariate | group)`


```julia
m = @bstm(
    likelihood(y) ~ intercept(false) + cov1 + 
        mixed( intercept(true) + cov1 | region ), # Correlated random intercept and slope for cov1
    data
)
```

## 2. Likelihood Features

These examples demonstrate how to modify the observation model using parameters within the `likelihood()` module.

### 2.1. Censored Data Model

Models a continuous outcome where some observations are censored.


```julia
y_lower_bound, y_upper_bound = 1, 100
m = @bstm(
    likelihood(y_rate, family=gaussian, censor_lower=y_lower_bound, censor_upper=y_upper_bound) ~ intercept() + fixed(region),
    data
)
```

### 2.2. Zero-Inflated Model

Models count data with an excess of zeros.


```julia
m = @bstm(
    likelihood(y, family=poisson, zero_inflated=true) ~ intercept() + cov1,
    data
)
```

### 2.3. Hurdle Model

A two-part model where the process for generating zeros is separate from the process for generating positive counts.


```julia
m = @bstm( likelihood(y, family=poisson, hurdle=1) ~ intercept() + cov1, data )
```

### 2.4. Stochastic Volatility Model

Models observation noise that varies over space and time.


```julia
m = @bstm(
    likelihood(y_rate, family=gaussian, volatility=true) ~ intercept() + spatial(s_idx, model=bym2, W=W),
    data
)
```

## 3. Spatial Models

### 3.1. Areal Data Models (GMRFs)

These models are for data aggregated over discrete spatial units (polygons).

#### BYM2 Disease Mapping Model

These examples demonstrate various models for capturing spatial autocorrelation.

The standard for areal disease mapping, decomposing spatial risk into structured and unstructured components.


```julia
m = @bstm( likelihood(y, family=poisson) ~ intercept() + spatial(s_idx, model=bym2, W=W), data )
```

#### ICAR / Besag Model

A model for strong spatial smoothing based on local neighbors.


```julia
m = @bstm( likelihood(y) ~ intercept() + spatial(s_idx, model=icar, W=W), data )
```

#### Leroux Model

Alternatives to BYM2 that offer different parameterizations of spatial correlation.


```julia
m = @bstm( likelihood(y) ~ intercept() + spatial(s_idx, model=leroux, W=W), data )
```

#### SAR Model

Models spatial "spill-over" effects where the value at one location directly influences its neighbors.


```julia
m = @bstm( likelihood(y) ~ intercept() + spatial(s_idx, model=sar, W=W, noise=1e-6), data)
```

### 3.2. Continuous & Point-Reference Models

These models are for data where exact coordinates are available.

#### Gaussian Process (GP)

The gold-standard for continuous spatial modeling, but computationally expensive.


```julia
m = @bstm( likelihood(y) ~ intercept() + smooth(s_x, s_y, model=gp, kernel=matern32), data)
```

#### SPDE Model

Models a continuous spatial process using an approximation to a Stochastic Partial Differential Equation, linked to the Matérn kernel.


```julia
m = @bstm( likelihood(y) ~ intercept() + spatial(s_idx, model=spde, W=W), data )
```

## 4. Temporal & Seasonal Models

### 4.1. Temporal Trend Models

#### Smooth Temporal Trend (Random Walk)

Models a smooth, non-linear temporal trend using a second-order random walk.


```julia
m = @bstm( likelihood(y) ~ intercept() + temporal(year, model=rw2), data )
```

#### Autoregressive Model (AR1)

Models a stationary temporal process where the current value depends on the immediately preceding value.


```julia
m = @bstm( likelihood(y) ~ intercept() + temporal(year, model=ar1), data )
```

### 4.2. Seasonal Models

#### Harmonic Seasonality

Captures periodic effects using sine and cosine basis functions.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + temporal(year, month, model=(ar1, cyclic), period=12),
    data
)
```

#### Cyclic Random Walk

Models a smooth, periodic effect where the end of the cycle connects to the beginning.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + temporal(day, model=cyclic, period=7),
    data
)
```

## 5. Covariate Smoothing (`smooth`)

### 5.1. 1D P-Spline Smoother

Models the non-linear effect of a continuous covariate using penalized splines.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + smooth(cov1, model=pspline, nbins=20),
    data
)
```

### 5.2. 2D Thin Plate Spline

Models the smooth, non-linear interaction of two continuous covariates (e.g., spatial coordinates).


```julia
m = @bstm(
    likelihood(y) ~ intercept() + smooth(s_x, s_y, model=tps, nbins=50),
    data
)
```

## 6. Interaction & Hierarchical Models

### 6.1. Separable Spatiotemporal Model

A standard model where the spatial and temporal effects are assumed to be independent and additive.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + spatial(s_idx, model=bym2, W=W) + temporal(year, model=ar1),
    data
)
```

### 6.2. Spatiotemporal Interaction Model (Knorr-Held Type IV)

A fully structured interaction where a spatial field (e.g., ICAR) evolves over time according to a temporal process (e.g., AR1).

**Formula Equivalent (using `⊗`):**
```julia
m = @bstm(
    likelihood(y) ~ intercept() + spatial(s_idx, model=icar) + temporal(year, model=ar1) +
        spatial(s_idx, model=icar) ⊗ temporal(year, model=ar1),
    data, W=W
)
```

**Formula Equivalent (using `spacetime`):**
```julia
m = @bstm(
    likelihood(y) ~ intercept() + spatial(s_idx, model=icar) + temporal(year, model=ar1) +
        spacetime(s_idx, year, model=(icar, ar1)),
    data, W=W
)
```

### 6.3. Spatially Varying Coefficients (SVC)

Allows the effect of a covariate to vary smoothly across space.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + cov1 |> spatial(s_idx, model=icar, W=W),
    data
)
```

### 6.4. Spatially Varying Curves

Models a non-linear trend of a covariate that varies smoothly across space.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + (smooth(year, model=pspline) |> spatial(s_idx, model=icar, W=W)),
    data
)
```

## 7. Advanced & Mechanistic Models

### 7.1. Mechanistic Dynamics (Advection-Diffusion)

A mechanistic model for a process that is transported (advection) and spreads (diffusion) over a graph.


```julia
m = @bstm(
    likelihood(concentration) ~ intercept() + dynamics(s_idx, year, model=advection_diffusion, W=W),
    data
)
```

### 7.1. Bayesian PCA (`eigen`)

Performs dimensionality reduction on a set of covariates, using the dominant latent factor as a predictor.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + eigen(cov1, cov2, cov3, n_factors=1),
    data
)
```

### 7.3. Multi-fidelity Model (`nested`)

Integrates a low-fidelity (but data-rich) proxy variable to improve predictions for a high-fidelity (but data-sparse) target.


```julia
m = @bstm(
    likelihood(y) ~ intercept() + 
        nested(
            proxy_model, 
            formula="likelihood(y_bin, family=binomial) ~ intercept() + smooth(cov3, model=pspline)"
        ),
    data
)
```

## 8. Multivariate Models

### 8.1. Multivariate CAR Model

A multivariate CAR model for jointly modeling multiple correlated spatial processes.


```julia
m = @bstm(
    y + y_bin ~ intercept() + spatial(s_idx, model=besag, W=W) + temporal(year, model=ar1),
    data
)
```

### 8.2. Joint Model with Different Likelihoods

Jointly models multiple outcomes where each has a different likelihood.


```julia
m = @bstm(
    likelihood(y, family=poisson) + likelihood(y_continuous, family=gaussian) ~ 
        intercept() + spatial(s_idx, model=bym2, W=W),
    data
)
```


# Advanced @bstm Call Examples

# 3.1 Multivariate Model
# Jointly modeling Gaussian and Poisson outcomes with spatial correlation
# Uses continuous 2D Thin Plate Spline for space and RW2 for time
println("Example 3.1: Constructing Multivariate Model...")
model_mv = @bstm(
    likelihood(y_gauss, family=gaussian) + likelihood(y_pois, family=poisson) ~
    intercept() + 
    smooth(s_x, s_y, model=tps, nbins=30) + 
    temporal(year, model=rw2),
    adv_data
)

# 3.2 Multinomial (Compositional) Model
# Modeling counts across 3 categories using Dirichlet-Multinomial
println("Example 3.2: Constructing Multinomial Model...")
# Note: Multi-column LHS targets the Dirichlet-Multinomial kernel
model_multi = @bstm(
    likelihood(y_cat1 + y_cat2 + y_cat3, family=dirichlet_multinomial) ~
    intercept() + 
    smooth(s_x, s_y, model=gp, kernel=matern32, n_inducing=15),
    adv_data
)

# 3.3 Multifidelity (Nested) Model
# High-fidelity Gaussian outcome aided by a low-fidelity proxy sub-model
println("Example 3.3: Constructing Multifidelity Model...")
model_mf = @bstm(
    likelihood(y_gauss, family=gaussian) ~
    intercept() + 
    temporal(year, model=ar1) + 
    nested(
        proxy_submodel,
        formula = "likelihood(proxy_val, family=gaussian) ~ intercept() + smooth(s_x, s_y, model=tps, nbins=20)"
    ),
    adv_data
)

# 3.4 Year and Seasonal Structure
# Combining a long-term trend (AR1) with a Harmonic seasonal component
println("Example 3.4: Constructing Year-Seasonal Model...")
model_season = @bstm(
    likelihood(y_gauss) ~
    intercept() + 
    temporal(year, model=ar1) + 
    temporal(month, model=harmonic, period=12),
    adv_data
)

# Demonstration of SVAR usage in BSTM
# Rationale: Shows how to model point-level dynamics where temporal persistence varies by region.

# Prepare data with spatiotemporal index
adv_data.st_idx = [(t-1)*30 + s for (s, t) in zip(adv_data.s_idx, adv_data.year)]

# Example call: Spatially Varying Autoregressive (SVAR) Model
# This model allows the temporal autoregressive parameter `rho` to vary across space.
# The spatial variation of `rho` is modeled by an `icar` manifold.
model_svar = @bstm(
   likelihood(y_gauss) ~
   intercept() +
   svar(spatial(s_idx, model=icar)),
   adv_data, W=W
)

println("The SVAR model allows local temporal dynamics to be influenced by spatial proximity.")


# Demonstration of Threshold Autoregressive (TAR) logic
# This model switches between two AR(1) regimes based on a covariate's value.

# Prepare data for TAR example
adv_data.price_index = 5.0 .+ cumsum(randn(nrow(adv_data)) .* 0.1)

# Example call: A TAR model where the temporal dynamics of y_gauss
# switch based on whether `price_index` is above or below a learned threshold.
model_tar = @bstm(
    likelihood(y_gauss) ~
    intercept() +
    temporal(year, model=tar, threshold_var=price_index),
    adv_data
)

println("The TAR model allows for regime-switching temporal dynamics.")

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
)
 


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
)


# Demonstration of Kriging implementation

# Prepare spatial data for Kriging
coord_data = DataFrame(
    s_x = rand(100) .* 10.0,
    s_y = rand(100) .* 10.0,
    y_gauss = randn(100)
)

# Example @bstm call (conceptual structure)
m = @bstm(
    likelihood(y_gauss) ~ 
    intercept() + 
    smooth(s_x, s_y, model=kriging, lengthscale=InverseGamma(3, 3), sigma=Exponential(1.0)),
    coord_data
)


  
