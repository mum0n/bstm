    

@model function example_bym2_ar1_poisson(M, ::Type{T}=Float64) where {T}
 
#     This model serves as a foundational spatiotemporal baseline for count data, typically used in epidemiology, crime analysis, or other fields where events occur over space and time. It decomposes the observed counts into a structured spatial component, a serially correlated temporal component, and an overall baseline, all within a Poisson likelihood framework.

#     The model explicitly addresses two key challenges in spatiotemporal data analysis:
#     1.  **Spatial Dependence:** Accounts for the fact that neighboring areas often exhibit similar patterns (e.g., disease rates in adjacent regions are correlated) using a BYM2 structure, which balances spatially structured effects with unstructured area-specific variability.
#     2.  **Temporal Dependence:** Captures trends or serial correlation over time (e.g., a high incidence in one period is likely followed by a high incidence in the next) through a first-order autoregressive (AR1) process.
 
    # --- 1. GLOBAL HYPERPRIORS ---
    s_sigma ~ Exponential(1.0) # Standard BSTM name for spatial scale
    s_rho ~ Beta(1, 1)        # Standard BSTM name for BYM2 mixing
    t_sigma ~ Exponential(1.0) # Standard BSTM name for temporal scale
    t_rho ~ Beta(2, 2)         # Standard BSTM name for AR1 persistence

    # --- 2. SPATIAL COMPONENT (BYM2) ---
    s_icar ~ MvNormal(zeros(M.N_areas), I)
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)

    s_iid ~ MvNormal(zeros(M.N_areas), I)

    # Mixture formulation using standard s_eta mapping
    s_eta_field = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # --- 3. TEMPORAL COMPONENT (AR1) ---
    t_Q = (1.0 / (1.0 - t_rho^2 + M.noise)) .* (M.t_Q + (t_rho^2) * I)
    t_raw ~ MvNormal(zeros(M.N_time), I)
    Turing.@addlogprob! -0.5 * dot(t_raw, t_Q * t_raw)

    t_eta_field = t_raw .* t_sigma

    # --- 4. POISSON LIKELIHOOD ---
    for i in 1:M.N_obs
        a = M.s_idx[i]
        t = M.t_idx[i]

        # Linear Predictor Synthesis
        eta = M.log_offset[i] + s_eta_field[a] + t_eta_field[t]

        # Weighted Likelihood Evaluation
        Turing.@addlogprob! M.weights[i] * logpdf(Poisson(exp(eta)), M.y_obs[i])
    end
end



@model function example_besag_ar1_poisson(M, ::Type{T}=Float64) where {T}
    # --- 1. GLOBAL HYPERPRIORS ---
    s_sigma ~ Exponential(1.0)  # Standard spatial scale
    t_sigma ~ Exponential(1.0)  # Standard temporal scale
    t_rho ~ Beta(2, 2)          # AR1 persistence
    st_sigma ~ Exponential(0.5) # Interaction scale
    c_sigma ~ filldist(Exponential(1.0), 4)

    if M.use_zi
        phi_zi ~ Beta(1, 1)
    end
 
    # --- 2. BESAG (ICAR) SPATIAL FIELD ---
    s_icar ~ MvNormal(zeros(M.N_areas), I)
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    
    # Soft sum-to-zero constraint for identifiability
    s_icar_sum ~ Normal(0, 0.001 * M.N_areas)
    Turing.@addlogprob! -0.5 * (sum(s_icar)^2 / (0.001 * M.N_areas))

    # Map to standard s_eta for reconstruction
    s_eta_field = s_icar .* s_sigma

    # --- 3. TEMPORAL FIELD (AR1) ---
    t_Q = (1.0 / (1.0 - t_rho^2 + M.noise)) .* (M.t_Q + (t_rho^2) * I)
    t_raw ~ MvNormal(zeros(M.N_time), I)
    Turing.@addlogprob! -0.5 * dot(t_raw, t_Q * t_raw)
    
    # Map to standard t_eta for reconstruction
    t_eta_field = t_raw .* t_sigma

    # --- 4. SPACE-TIME INTERACTION ---
    st_raw ~ MvNormal(zeros(M.N_areas * M.N_time), I)
    st_eta = reshape(st_raw .* st_sigma, M.N_areas, M.N_time)

    # --- 5. CATEGORICAL SMOOTHING (RW2) ---
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ (c_sigma[k]^2 + M.noise)) * c_beta[k])
    end

    # --- 6. LIKELIHOOD ---
    for i in 1:M.N_obs
        a, t = M.s_idx[i], M.t_idx[i]
        eta = M.log_offset[i] + s_eta_field[a] + t_eta_field[t] + st_eta[a, t]
        for k in 1:M.N_cov; eta += c_beta[k][M.cov_indices[i, k]]; end
        
        mu = exp(eta)
        if M.use_zi
            if M.y_obs[i] == 0
                Turing.@addlogprob! M.weights[i] * log(phi_zi + (1 - phi_zi) * exp(-mu) + 1e-10)
            else
                Turing.@addlogprob! M.weights[i] * (log(1 - phi_zi + 1e-10) + logpdf(Poisson(mu), M.y_obs[i]))
            end
        else
            Turing.@addlogprob! M.weights[i] * logpdf(Poisson(mu), M.y_obs[i])
        end
    end
end



@model function example_gp_gaussian(M, ::Type{T}=Float64 ) where {T}
    # ## `example_gp_gaussian` Model Documentation
    # 
    # ### Conceptual Overview
    # This model implements a dense spatiotemporal Gaussian Process (GP) with a separable kernel, specifically designed for Gaussian likelihood data. It aims to capture complex, non-linear spatiotemporal patterns in data.
    # 
    # 1.  **Priors:** Sets priors for the observation noise (`y_sigma`), the overall amplitude of the GP (`sigma_f`), and the length scales for the spatial (`ls_s`) and temporal (`ls_t`) dimensions.
    # 2.  **Coordinate Extraction:** Prepares the spatial (`xs`, `ys`) and temporal (`ts`) coordinates from the input data structure `M` for kernel computation.
    # 3.  **Kernel Matrix Construction:** Builds a full spatiotemporal covariance matrix `K`. It uses a separable kernel, meaning the spatiotemporal covariance is the product of a spatial kernel and a temporal kernel. Both are Squared Exponential (RBF) kernels with their respective length scales.
    # 4.  **Latent Process:** Models the underlying latent spatiotemporal function (`f_latent`) as a Multivariate Normal distribution with mean zero and covariance `K`. A small `M.noise` is added for numerical stability.
    # 5.  **Likelihood:** Assumes that the observed data (`M.y_obs`) are drawn from a Normal distribution, where the mean is given by a log offset (`M.log_offset`) plus the latent GP process (`f_latent`), and the standard deviation is `y_sigma`.
    # 
    # ### Detailed Mathematical Justification
    # 
    # The `example_gp_gaussian` model defines a Gaussian Process to model spatiotemporal data with a Gaussian likelihood.
    # 
    # #### 1. Priors
    # 
    # *   **`y_sigma` (Observation Noise Standard Deviation):**
    #     *   `y_sigma ~ Exponential(1.0)`
    #     *   This is the standard deviation of the observation noise for the Gaussian likelihood. An `Exponential(1.0)` prior encourages smaller noise values, with a mean of 1.0.
    # 
    # *   **`sigma_f` (GP Amplitude):**
    #     *   `sigma_f ~ Exponential(1.0)`
    #     *   This parameter controls the overall amplitude or vertical scale of the Gaussian Process. It acts as a scaling factor for the covariance function. An `Exponential(1.0)` prior is used.
    # 
    # *   **`ls_s` (Spatial Length Scale):**
    #     *   `ls_s ~ Gamma(2, 2)`
    #     *   This is the length scale for the spatial component of the Squared Exponential kernel. It determines how quickly the correlation decays with increasing spatial distance. A `Gamma(2, 2)` prior is a common weakly informative prior for positive scale parameters.
    # 
    # *   **`ls_t` (Temporal Length Scale):**
    #     *   `ls_t ~ Gamma(2, 2)`
    #     *   This is the length scale for the temporal component of the Squared Exponential kernel. It determines how quickly the correlation decays with increasing temporal distance. A `Gamma(2, 2)` prior is used.
    # 
    # #### 2. Coordinate Extraction
    # 
    # *   `xs = [p[1] for p in M.pts]`
    # *   `ys = [p[2] for p in M.pts]`
    # *   `ts = Float64.(M.t_idx)`
    # *   These lines extract the spatial (x and y coordinates) and temporal (time indices) information from the input data structure `M`. `M.pts` is assumed to be a collection of (x, y) spatial points, and `M.t_idx` are the corresponding time indices for each observation. `RowVecs` is used to correctly format the spatial coordinates for the `kernelmatrix` function.
    # 
    # #### 3. Kernel Matrix Construction
    # 
    # The model uses a separable spatiotemporal kernel, meaning the overall covariance function `k_st((s, t), (s', t'))` can be expressed as a product of a spatial kernel `k_s(s, s')` and a temporal kernel `k_t(t, t')`.
    # 
    # *   **Spatial Kernel `k_s`:**
    #     *   `k_s = SqExponentialKernel() ∘ ScaleTransform(inv(ls_s))`
    #     *   This defines a Squared Exponential (or Radial Basis Function - RBF) kernel for the spatial dimension. The `ScaleTransform(inv(ls_s))` scales the input by `1/ls_s`, so `ls_s` acts as the characteristic length scale. The spatial covariance between two points `s` and `s'` is `exp(-0.5 * ||s - s'||^2 / ls_s^2)`.
    # 
    # *   **Temporal Kernel `k_t`:**
    #     *   `k_t = SqExponentialKernel() ∘ ScaleTransform(inv(ls_t))`
    #     *   Similar to the spatial kernel, this defines a Squared Exponential kernel for the temporal dimension, with `ls_t` as its length scale. The temporal covariance between two time points `t` and `t'` is `exp(-0.5 * ||t - t'||^2 / ls_t^2)`.
    # 
    # *   **Full Covariance Matrix `K`:**
    #     *   `coords_s = RowVecs(hcat(xs, ys))`
    #     *   `K = (sigma_f^2) .* kernelmatrix(k_s, coords_s) .* kernelmatrix(k_t, ts)`
    #     *   `kernelmatrix(k_s, coords_s)` computes the `N_obs x N_obs` spatial covariance matrix.
    #     *   `kernelmatrix(k_t, ts)` computes the `N_obs x N_obs` temporal covariance matrix.
    #     *   The element-wise product (`.*`) of these two matrices, scaled by `sigma_f^2` (which is the overall signal variance), forms the separable spatiotemporal covariance matrix `K`. This means `K[i,j] = sigma_f^2 * k_s(s_i, s_j) * k_t(t_i, t_j)`.
    # 
    # #### 4. Latent Process
    # 
    # *   `f_latent ~ MvNormal(zeros(M.N_obs), K + M.noise*I)`
    # *   The latent spatiotemporal field `f_latent` is modeled as a Multivariate Normal distribution.
    #     *   `zeros(M.N_obs)`: The mean of the GP is assumed to be zero, capturing deviations from an overall mean or offset.
    #     *   `K + M.noise*I`: The covariance matrix is `K`, with `M.noise*I` added for numerical stability (a "nugget" effect, preventing the covariance matrix from being exactly singular). `M.N_obs` is the total number of observations.
    # 
    # #### 5. Likelihood
    # 
    # *   `for i in 1:M.N_obs`
    #     *   `mu = M.log_offset[i] + f_latent[i]`
    #     *   `Turing.@addlogprob! logpdf(Normal(mu, y_sigma), M.y_obs[i])`
    # *   For each observation `i` in `M.N_obs`:
    #     *   The mean `mu` of the Gaussian likelihood is constructed as the sum of a fixed `M.log_offset[i]` (e.g., an intercept or known covariate effect) and the corresponding value from the latent GP `f_latent[i]`. `M.log_offset` can be used to incorporate known trends or covariates.
    #     *   The observation `M.y_obs[i]` is assumed to follow a Normal distribution with this mean `mu` and a standard deviation `y_sigma`. `Turing.@addlogprob!` adds the log-probability density of this observation to the model's total log-likelihood.
    # 
    # This structure allows the model to infer the underlying smooth spatiotemporal function `f_latent` and its uncertainty, while accounting for observation noise and fixed effects.

    # Dense Spatiotemporal Gaussian Process
    # Uses a separable space and time kernel

    # --- 1. Priors ---
    y_sigma ~ Exponential(1.0)
    sigma_f ~ Exponential(1.0)
    ls_s ~ Gamma(2, 2)
    ls_t ~ Gamma(2, 2)

    # --- 2. Coordinate Extraction ---
    xs = [p[1] for p in M.pts]
    ys = [p[2] for p in M.pts]
    ts = Float64.(M.t_idx)

    # --- 3. Kernel Matrix Construction ---
    k_s = SqExponentialKernel() ∘ ScaleTransform(inv(ls_s))  # space
    k_t = SqExponentialKernel() ∘ ScaleTransform(inv(ls_t))  # time
    coords_s = RowVecs(hcat(xs, ys))
    K = (sigma_f^2) .* kernelmatrix(k_s, coords_s) .* kernelmatrix(k_t, ts) # full covariance (separable)

    # --- 4. Latent Process ---
    # Renamed to s_eta_field for compatibility with _reconstruct
    s_eta_field ~ MvNormal(zeros(M.N_obs), K + M.noise*I)

    # --- 5. Likelihood ---
    for i in 1:M.N_obs
        mu = M.log_offset[i] + s_eta_field[i]
        Turing.@addlogprob! logpdf(Normal(mu, y_sigma), M.y_obs[i])
    end
end


@model function example_hurdle_bernoulli_poisson(M, ::Type{T}=Float64) where {T}
    # """
    # **Conceptual Overview: Hurdle Bernoulli-Poisson Model**

    # This model is designed for count data that frequently exhibit an excess of zeros (e.g., disease incidence where many regions have zero cases). It addresses this by explicitly modeling the data generation process in two stages:
    # 1.  **Hurdle Stage (Binary):** Determines whether an event occurs at all (i.e., if the count is zero or positive). This is modeled using a Bernoulli distribution with a logit link function.
    # 2.  **Count Stage (Truncated Poisson):** If an event occurs (i.e., the count is positive), the magnitude of the count is then modeled using a zero-truncated Poisson distribution.

    # This two-part structure allows for a more flexible and accurate representation of data with many zeros compared to standard Poisson or negative binomial models, which often struggle to fit both the zero and positive counts simultaneously. Each stage (hurdle and count) can have its own set of predictors and spatiotemporal random effects, providing a rich framework for understanding complex epidemiological or ecological processes.

    # The model incorporates spatiotemporal random effects for both the hurdle and count components:
    # -   **Spatial (BYM2):** Accounts for spatial correlation using the BYM2 formulation (Riebler et al., 2016), which combines a structured Intrinsic Conditional Autoregressive (ICAR) component and an unstructured Independent and Identically Distributed (IID) component. This helps separate true spatial trends from local heterogeneity.
    # -   **Temporal (AR1):** Captures serial correlation over time through a first-order autoregressive (AR1) process.
    # -   **Shared Covariates (RW2):** Allows for the inclusion of categorical covariates, with effects smoothed using a second-order random walk (RW2) prior, impacting both the hurdle and count stages.

    # **Mathematical Justification:**

    # Let $Y_{it}$ be the observed count for area $i$ at time $t$. The hurdle model operates in two stages:

    # **Stage 1: Binary Process (Hurdle for Zero vs. Non-zero)**

    # Let $Z_{it}$ be an indicator variable where $Z_{it}=0$ if $Y_{it}=0$ and $Z_{it}=1$ if $Y_{it}>0$. We model $Z_{it}$ using a Bernoulli distribution:

    # $$Z_{it} \sim \text{Bernoulli}(p_{it})$$

    # with a logit link function for $p_{it}$:

    # $$\text{logit}(p_{it}) = \eta_{it}^H = \eta_i^{S,H} + \eta_t^{T,H} + \sum_{k=1}^4 \beta_{k, c_{ik}}$$

    # where:
    # -   $\eta_{it}^H$ is the linear predictor for the hurdle (binary) component.
    # -   $\eta_i^{S,H}$ is the spatial random effect for the hurdle component.
    # -   $\eta_t^{T,H}$ is the temporal random effect for the hurdle component.
    # -   $\beta_{k, c_{ik}}$ are the effects of categorical covariate $k$, impacting both components.

    # **Stage 2: Count Process (for Positive Counts)**

    # If $Y_{it} > 0$, we model $Y_{it}$ using a zero-truncated Poisson distribution:

    # $$Y_{it} \sim \text{TruncatedPoisson}(\lambda_{it})$$

    # where the Poisson mean $\lambda_{it}$ is linked to a linear predictor via a log link:

    # $$\log(\lambda_{it}) = \eta_{it}^C = \text{offset}_{it} + \eta_i^{S,C} + \eta_t^{T,C} + \sum_{k=1}^4 \beta_{k, c_{ik}}$$

    # where:
    # -   $\text{offset}_{it}$ is a known log-transformed offset.
    # -   $\eta_i^{S,C}$ is the spatial random effect for the count component.
    # -   $\eta_t^{T,C}$ is the temporal random effect for the count component.
    # -   $\beta_{k, c_{ik}}$ are the shared categorical covariate effects.

    # The probability mass function for the Hurdle-Poisson model is:
    # -   For $Y_{it} = 0$: $P(Y_{it}=0) = 1 - p_{it}$
    # -   For $Y_{it} > 0$: $P(Y_{it}=y) = p_{it} \cdot \frac{\lambda_{it}^y e^{-\lambda_{it}}}{y! \cdot (1 - e^{-\lambda_{it}})}$

    # **1. Priors:**

    # -   **Hurdle (Binary) Component Hyperparameters:**
    #     -   `s_sigma_h` $\sim \text{Exponential}(1.0)$: Spatial scale for the hurdle part.
    #     -   `s_rho_h` $\sim \text{Beta}(1, 1)$: BYM2 mixing parameter for the hurdle part.
    #     -   `t_sigma_h` $\sim \text{Exponential}(1.0)$: Temporal scale for the hurdle part.
    #     -   `t_rho_h` $\sim \text{Beta}(2, 2)$: AR1 persistence for the hurdle part.

    # -   **Count (Positive) Component Hyperparameters:**
    #     -   `s_sigma_c` $\sim \text{Exponential}(1.0)$: Spatial scale for the count part.
    #     -   `s_rho_c` $\sim \text{Beta}(1, 1)$: BYM2 mixing parameter for the count part.
    #     -   `t_sigma_c` $\sim \text{Exponential}(1.0)$: Temporal scale for the count part.
    #     -   `t_rho_c` $\sim \text{Beta}(2, 2)$: AR1 persistence for the count part.

    # -   **Shared Covariate Hyperparameters:**
    #     -   `c_sigma` $\sim \text{filldist}(\text{Exponential}(1.0), 4)$: Scale parameters for the four categorical covariate effects.

    # **2. Latent Fields for Hurdle Part (BYM2 Spatial, AR1 Temporal):**

    # -   **Spatial (BYM2) component $\eta_i^{S,H}$:**
    #     -   `s_icar_h` $\sim \text{MvNormal}(\mathbf{0}, I)$ with precision `M.s_Q` (ICAR component).
    #     -   `s_iid_h` $\sim \text{MvNormal}(\mathbf{0}, I)$ (IID component).
    #     -   `s_eta_h = s_sigma_h .* (sqrt(s_rho_h) .* s_icar_h .+ sqrt(1 - s_rho_h) .* s_iid_h)` (Riebler et al., 2016 formulation).

    # -   **Temporal (AR1) component $\eta_t^{T,H}$:**
    #     -   `t_Q_h = (1.0 / (1.0 - t_rho_h^2)) .* (M.t_Q + (t_rho_h^2) * I)`: AR1 precision matrix.
    #     -   `t_f_h` $\sim \text{MvNormal}(\mathbf{0}, I)$ with precision `t_Q_h`.
    #     -   `t_eta_h = t_f_h .* t_sigma_h`.

    # **3. Latent Fields for Count Part (BYM2 Spatial, AR1 Temporal):**

    # -   **Spatial (BYM2) component $\eta_i^{S,C}$:**
    #     -   `s_icar_c` $\sim \text{MvNormal}(\mathbf{0}, I)$ with precision `M.s_Q`.
    #     -   `s_iid_c` $\sim \text{MvNormal}(\mathbf{0}, I)$.
    #     -   `s_eta_c = s_sigma_c .* (sqrt(s_rho_c) .* s_icar_c .+ sqrt(1 - s_rho_c) .* s_iid_c)`.

    # -   **Temporal (AR1) component $\eta_t^{T,C}$:**
    #     -   `t_Q_c = (1.0 / (1.0 - t_rho_c^2)) .* (M.t_Q + (t_rho_c^2) * I)`: AR1 precision matrix.
    #     -   `t_f_c` $\sim \text{MvNormal}(\mathbf{0}, I)$ with precision `t_Q_c`.
    #     -   `t_eta_f_c = t_f_c .* t_sigma_c`.

    # **4. Shared Categorical Smoothing (RW2):**

    # -   `c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]`: A vector of effects for each of the 4 categorical covariates.
    # -   For each `k` in `1:4`:
    #     -   `c_beta[k]` $\sim \text{MvNormal}(\mathbf{0}, I)$ with precision `(M.Q_rw2 ./ c_sigma[k]^2)`. $M.Q_{rw2}$ is the RW2 precision matrix, and `c_sigma[k]` scales the variance.

    # **5. Joint Hurdle Likelihood:**

    # For each observation $i$ (mapping to area $a=M.s_{idx}[i]$ and time $t=M.t_{idx}[i]$):

    # -   **Linear Predictors:**
    #     -   `eta_h = s_eta_h[a] + t_eta_h[t] + \sum_{k=1}^4 c_{\text{beta}}[k][M.cov_{indices}[i, k]]`
    #     -   `eta_c = M.log_offset[i] + s_eta_c[a] + t_eta_f_c[t] + \sum_{k=1}^4 c_{\text{beta}}[k][M.cov_{indices}[i, k]]`

    # -   **Likelihood Calculation:**
    #     -   If $Y_{it} = 0$: `Turing.@addlogprob! M.weights[i] * logpdf(BernoulliLogit(eta_h), 0)` (log-probability of being in the zero-state).
    #     -   If $Y_{it} > 0$: `mu = exp(eta_c)`. The log-probability is `M.weights[i] * (logpdf(BernoulliLogit(eta_h), 1) + logpdf(Poisson(mu), M.y_obs[i]) - log(1 - exp(-mu)))`. This combines the log-probability of being non-zero from the Bernoulli part and the log-density of the truncated Poisson part.
    # """

    # --- 1. Priors ---
    # Hurdle (Binary) Component Hyperparams
    s_sigma_h ~ Exponential(1.0); s_rho_h ~ Beta(1, 1)
    t_sigma_h ~ Exponential(1.0); t_rho_h ~ Beta(2, 2)
    
    # Count (Positive) Component Hyperparams
    s_sigma_c ~ Exponential(1.0); s_rho_c ~ Beta(1, 1)
    t_sigma_c ~ Exponential(1.0); t_rho_c ~ Beta(2, 2)
    
    c_sigma ~ filldist(Exponential(1.0), 4)

    # --- 2. Latent Fields for Hurdle Part ---
    s_icar_h ~ MvNormal(zeros(M.N_areas), I); Turing.@addlogprob! -0.5 * dot(s_icar_h, M.s_Q * s_icar_h)
    s_iid_h ~ MvNormal(zeros(M.N_areas), I)
    # Renamed to s_eta_field_h for reconstruction compatibility
    s_eta_field_h = s_sigma_h .* (sqrt(s_rho_h) .* s_icar_h .+ sqrt(1 - s_rho_h) .* s_iid_h)

    t_Q_h = (1.0 / (1.0 - t_rho_h^2)) .* (M.t_Q + (t_rho_h^2) * I)
    t_f_h ~ MvNormal(zeros(M.N_time), I); Turing.@addlogprob! -0.5 * dot(t_f_h, t_Q_h * t_f_h)
    # Renamed to t_eta_field_h for reconstruction compatibility
    t_eta_field_h = t_f_h .* t_sigma_h

    # --- 3. Latent Fields for Count Part ---
    s_icar_c ~ MvNormal(zeros(M.N_areas), I); Turing.@addlogprob! -0.5 * dot(s_icar_c, M.s_Q * s_icar_c)
    s_iid_c ~ MvNormal(zeros(M.N_areas), I)
    # Renamed to s_eta_field_c for reconstruction compatibility
    s_eta_field_c = s_sigma_c .* (sqrt(s_rho_c) .* s_icar_c .+ sqrt(1 - s_rho_c) .* s_iid_c)

    t_Q_c = (1.0 / (1.0 - t_rho_c^2)) .* (M.t_Q + (t_rho_c^2) * I)
    t_f_c ~ MvNormal(zeros(M.N_time), I); Turing.@addlogprob! -0.5 * dot(t_f_c, t_Q_c * t_f_c)
    # Renamed to t_eta_field_c for reconstruction compatibility
    t_eta_field_c = t_f_c .* t_sigma_c

    # --- 4. Shared Categorical Smoothing ---
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 5. Joint Hurdle Likelihood ---
    for i in 1:M.N_obs
        a, t = M.s_idx[i], M.t_idx[i]

        # Linear predictors
        eta_h = s_eta_field_h[a] + t_eta_field_h[t]
        eta_c = M.log_offset[i] + s_eta_field_c[a] + t_eta_field_c[t]
        for k in 1:M.N_cov;
            eff_k = c_beta[k][M.cov_indices[i, k]]
            eta_h += eff_k
            eta_c += eff_k
        end

        if M.y_obs[i] == 0
            Turing.@addlogprob! M.weights[i] * logpdf(BernoulliLogit(eta_h), 0)
        else
            mu = exp(eta_c)
            lp_nonzero = logpdf(BernoulliLogit(eta_h), 1)
            lp_count = logpdf(Poisson(mu), M.y_obs[i]) - log(1 - exp(-mu))
            Turing.@addlogprob! M.weights[i] * (lp_nonzero + lp_count)
        end
    end
end


@model function bstm(M, ::Type{T}=Float64) where {T}
    # """
    # **Conceptual Overview: bstm - Generalized Random Markov Field Spatiotemporal Model**

    # The `bstm` (Bayesian Space-Time Model) is a highly modular and flexible framework for inference on spatiotemporal data. It decomposes observed phenomena into various underlying components, such as spatial effects, temporal trends, space-time interactions, covariate effects, and seasonal patterns. Its design emphasizes adaptability, supporting a range of model specifications and likelihood families to accommodate diverse data types and research questions.

    # The core idea is to build complex spatiotemporal models by combining different 'manifolds' (components), each with its own set of prior distributions and mathematical structures. This modularity allows researchers to tailor models precisely to the nuances of their data, from simple IID effects to sophisticated Gaussian Processes or graph-based convolutions.

    # **Architecture Hierarchy & Key Components:**

    # 1.  **Stochastic Volatility Manifold (Optional):** Allows the observation-level variance (for Gaussian/Log-Normal models) to vary over space and time, captured by a Random Fourier Feature (RFF) representation.

    # 2.  **Spatial Manifolds (`s_eta`):** Models spatial dependencies. Options include:
    #     -   **IID:** Independent and identically distributed effects (no spatial structure).
    #     -   **Besag/ICAR:** Intrinsic Conditional Autoregressive model for structured spatial smoothing.
    #     -   **BYM2:** Riebler et al.'s (2016) BYM2 model, combining structured ICAR and unstructured IID spatial effects for improved identifiability.
    #     -   **FITC (Fully Independent Training Conditional) Sparse GP:** A low-rank approximation of a Gaussian Process, efficient for large datasets.
    #     -   **FITC-GP / Nystrom:** Variations of sparse GP approximations.
    #     -   **Mosaic:** Localized experts where spatial effects are clustered.
    #     -   **Warping:** Nonstationary spatial fields using RFF for flexible deformation.
    #     -   **RFF / EI:** Random Fourier Features for continuous risk surfaces.
    #     -   **BGCN (Bayesian Graph Convolutional Network):** Models spatial dependencies via spectral graph convolutions, suitable for irregular graph structures.
    #     -   **SAR (Spatial Autoregressive):** Models spatial dependence directly through a spatial lag operator.
    #     -   **DAG (Directed Acyclic Graph):** For causal inference or flow data, where dependencies are directional.
    #     -   **Local:** Cluster-specific effects with Leroux-like precision structure.
    #     -   **SVGP (Sparse Variational Gaussian Process):** A scalable GP using variational inference.
    #     -   **SVC (Spatially Varying Coefficients):** Allows regression coefficients of fixed effects to vary spatially.

    # 3.  **Temporal Manifolds (`t_eta`):** Models temporal trends. Options include:
    #     -   **AR1:** First-order Autoregressive process for capturing serial correlation.
    #     -   **RW2:** Second-order Random Walk for smooth temporal trends.
    #     -   **GP:** Gaussian Process for flexible non-parametric temporal modeling.
    #     -   **Harmonic:** Seasonal patterns using sine and cosine functions.
    #     -   **IID:** Independent and identically distributed temporal effects.

    # 4.  **Space-Time Interaction (`st_eta`):** Captures how spatial patterns change over time or how temporal trends vary across space. Implements Knorr-Held (2000) interaction types:
    #     -   **Type I:** IID interaction (unstructured).
    #     -   **Type II:** Structured temporal trend varying independently across space.
    #     -   **Type III:** Structured spatial pattern varying independently across time.
    #     -   **Type IV:** Fully inseparable spatiotemporal structure (structured both spatially and temporally).

    # 5.  **Covariate Smoothing (`c_beta`):** Handles effects of covariates, including:
    #     -   **RW2/AR1/GP/Harmonic/IID:** Smoothing for categorical or discretized continuous covariates.
    #     -   **RFF:** For continuous Gaussian Processes with Random Fourier Features.

    # 6.  **Season Manifold (`u_eta`):** (If separate from temporal trend) Models distinct seasonal effects, similar options to temporal manifolds.

    # 7.  **Likelihood Engine:** A unified interface (`bstm_Likelihood`) for various distributions:
    #     -   **Gaussian, Log-Normal, Poisson, Binomial, Negative Binomial.**
    #     -   Optional **Zero-Inflation** for count data (Poisson, Negative Binomial).

 
    # Global Hyperpriors & Toggle Logic ---
    local r_nb = 1.0
    local phi_zi = 0.0

    if M.model_family == "negbin"
        r_nb ~ Exponential(1.0)
    end

    if M.use_zi
        phi_zi ~ Beta(1, 1)
    end

    #  Stochastic Volatility Manifold (Optional) ---

    local y_sigma = 1.0
         
    if get(M, :use_sv, false)
        sigma_log_var ~ Exponential(1.0)
        beta_vol ~ filldist(Normal(0, sigma_log_var), M.M_rff_sigma)
        # Use spatiotemporal coordinates for volatility surface
        coords_st = hcat(M.s_obs, M.t_obs ./ M.N_time)
        vol_proj = (coords_st * M.W_sigma_fixed) .+ M.b_sigma_fixed'
        # Map RFF to positive scale via exponential link
        y_sigma = exp.((sqrt(2.0 / M.M_rff_sigma) .* cos.(vol_proj) * beta_vol) ./ 2.0)
    else
        if M.model_family in ["gaussian", "lognormal"]
            y_sigma ~ Exponential(1.0) 
        end
    end


    # Spatial Manifold (s_eta) ---
    local s_eta = zeros(T, M.N_areas)  # default is none
    local s_Q = I(M.N_areas)

    

    if M.model_space == "iid"
        s_sigma ~ Exponential(1.0)
        s_iid ~ MvNormal(zeros(M.N_areas), I)
        s_eta = (s_iid .* s_sigma)
        s_Q = I(M.N_areas)

    elseif M.model_space in ["besag", "icar"]
        s_sigma ~ Exponential(1.0)
        s_icar ~ MvNormalCanon(zeros(M.N_areas), Symmetric(Matrix(M.s_Q + M.noise*I)) )
        s_eta = (s_icar .* s_sigma)

    elseif M.model_space == "bym2"
        s_sigma ~ Exponential(1.0)
        s_rho ~ Beta(1, 1)
        s_icar ~ MvNormalCanon(zeros(M.N_areas), Symmetric(Matrix(M.s_Q + M.noise*I)) )
        s_iid ~ MvNormal(zeros(M.N_areas), I)
        s_eta = (s_sigma .* (sqrt.(s_rho) .* s_icar .+ sqrt.(1 .- s_rho) .* s_iid))

    elseif M.model_space == "leroux"
        # Leroux Precision: Q = 1/sigma^2 * (rho * Q_icar + (1-rho) * I)
        s_sigma ~ Exponential(1.0)
        s_rho ~ Beta(1, 1)
        s_Q = s_rho .* M.s_Q .+ (1 - s_rho) .* I(M.N_areas)
        s_raw ~ MvNormalCanon(zeros(M.N_areas), s_Q + M.noise * I)
        s_eta = (s_raw .* s_sigma)

    elseif M.model_space == "fitc"
        # --- 1. FITC Sparse GP Implementation ---
        # This logic provides a low-rank approximation of the spatiotemporal field
        # using M.Z_inducing locations.

        # Priors for the ARD (Automatic Relevance Determination) lengthscales
        # Dimensions: 1 for time, 2 for space
        s_sigma ~ Exponential(1.0)
        ls_st ~ filldist(Gamma(2, 2), 3)

        # Kernel construction with ARD and stability jitter
        k_st = (s_sigma^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(ls_st .+ M.noise)))

        # Covariance matrices between inducing points (ZZ) and inducing-to-observations (XZ)
        K_ZZ = kernelmatrix(k_st, RowVecs(M.Z_inducing)) + M.noise * I
        K_XZ = kernelmatrix(k_st, RowVecs(hcat(M.s_obs, M.t_obs)), RowVecs(M.Z_inducing))

        # Sample latent inducing values using the precision-based canonical form for efficiency
        # if M.noise is small, or MvNormal for standard covariance
        s_inducing ~ MvNormal(zeros(size(M.Z_inducing, 1)), K_ZZ)

        # FITC Projection: E[f|u] = K_XZ * inv(K_ZZ) * u
        # Numerically stable solve using backslash
        s_eta = K_XZ * (K_ZZ \ s_inducing)

        # Precision representation for the low-rank component (Optional for audit)
        s_Q = inv(K_ZZ) + M.noise * I

    elseif M.model_space == "fitcGP"
        # Sparse GP inducing points
        s_sigma ~ Exponential(1.0)
        u_inducing ~ filldist(Normal(0, 1), M.M_inducing_count)
        # Projection to observation space (simplification for bstm context)
        s_eta = (M.Z_inducing_proj * u_inducing) .* s_sigma

    elseif M.model_space == "mosaic"
        # Localized experts per region
        s_sigma ~ Exponential(1.0)
        mu_local ~ filldist(Normal(0, 1), M.n_mosaics)
        s_eta = mu_local[M.cluster_assignments] .* s_sigma

    elseif M.model_space == "warping"
        # Nonstationary warping field
        s_sigma ~ Exponential(1.0)
        w_warp ~ filldist(Normal(0, 1), M.M_rff)
        warp_proj = (M.s_obs * M.W_fixed) .+ M.b_fixed'
        s_warp = (sqrt(2.0 / M.M_rff) .* cos.(warp_proj) * w_warp)
        s_eta = vec( s_warp .* s_sigma )

    elseif M.model_space == "rff" || M.model_space == "ei"
        s_sigma ~ Exponential(1.0)
        ls_rff ~ Gamma(2, 2)
        beta_rff ~ filldist(Normal(0, 1), M.M_rff)
        # Spectral mapping for continuous risk surfaces
        projection = (M.s_obs * (M.W_fixed ./ ls_rff)) .+ M.b_fixed'
        s_eta = vec( (sqrt(2.0 / M.M_rff) * s_sigma) .* (cos.(projection) * beta_rff) )
        # In RFF, the latent variables are the iid weights beta_rff
        s_Q = I(M.M_rff)

    elseif M.model_space == "bgcn"
        # 1. Spectral Weight Priors
        # These weights represent the learnable signal amplitude across the graph nodes
        # typically inferred from the graph signal (spectral domain)
        s_sigma ~ Exponential(1.0)
        gcn_weight_raw ~ MvNormal(zeros(M.N_areas), I)

        # 2. Normalized Laplacian / Transition Matrix Construction
        # Symmetric normalization (D^-0.5 * W * D^-0.5) ensures spectral stability
        D_inv_sqrt = Diagonal(1.0 ./ sqrt.(vec(sum(M.W, dims=2)) .+ M.noise))
        W_norm = D_inv_sqrt * M.W * D_inv_sqrt

        # 3. Graph Convolution Operation
        # This approximates a localized spectral filter on the graph signal
        s_eta_raw = W_norm * gcn_weight_raw

        # 4. Observation Mapping and Precision Log
        # Map the smoothed latent signal to observations and apply the marginal scale sigma
        s_eta = (s_eta_raw[M.s_idx] .* s_sigma)


    elseif M.model_space == "deepGP"
            
        # Fixed deepGP logic: iterative RFF mapping across layers
        # Prepare base coordinates [X, Y, T]
        curr_coords = hcat(M.s_obs, M.t_obs ./ M.N_time)
        
        for l in 1:M.n_layers
            m_l = M.m_layers[l]
            d_in = size(curr_coords, 2)

            ls_l ~ Gamma(2, 1)
            w_l ~ filldist(Normal(0, 1), m_l)
            
            # Sample frequencies for this layer
            Random.seed!(42 + l)
            Om = randn(m_l, d_in) ./ ls_l
            Ph = rand(m_l) .* convert(T, 2 * pi)

            # Project and apply non-linearity
            Phi = convert(T, sqrt(2 / m_l)) .* cos.(curr_coords * Om' .+ Ph')
            
            if l < M.n_layers
                # Intermediate layer output becomes next layer input
                curr_coords = reshape(Phi * w_l, :, 1)
            else
                # Final layer output is the spatial effect
                s_eta = Phi * w_l
            end
        end
        s_eta = vec(s_eta)
    
    elseif M.model_space == "sar"
        # 1. SAR Precision Matrix Construction
        # The SAR model is defined by (I - rho*W)x = e, where e ~ N(0, sigma^2 * I)
        # The resulting precision matrix is Q = (I - rho*W)' * (I - rho*W)
        # Standardize the row-normalized adjacency matrix W
        s_sigma ~ Exponential(1.0) 

        s_rho ~ Beta(1, 1)
        W_row_norm = M.W ./ (vec(sum(M.W, dims=2)) .+ M.noise)
        L_sar = I(M.N_areas) - s_rho .* W_row_norm

        # 2. Latent Field Sampling
        # sample the latent spatial field directly in its canonical form
        # Observation Mapping and Scaling
        s_Q = Symmetric(Matrix(L_sar' * L_sar)) + (s_sigma + M.noise) * I
        s_eta ~ MvNormalCanon(zeros(M.N_areas), s_Q)

    elseif M.model_space == "fft"
        s_sigma ~ Exponential(1.0)
        s_raw ~ MvNormal(zeros(M.N_areas), I)
        s_eta = ( M.s_Q \ s_raw ) .* s_sigma

    elseif M.model_space == "nystrom"
        # Nystrom low-rank approximation
        s_sigma ~ Exponential(1.0)
        s_raw ~ filldist(Normal(0, 1), M.M_inducing_count)
        s_eta = (M.K_nystrom_proj * s_raw) .* s_sigma

    elseif M.model_space == "svc"  

        # Logic for Spatially Varying Coefficients
        # We generate a spatial field for each of the N_fixed covariates
        s_svc = Matrix{T}(undef, M.N_areas, M.N_fixed)
        svc_sigma ~ filldist(Exponential(0.5), M.N_fixed)
        svc_raw ~ arraydist([MvNormalCanon(zeros(M.N_areas), M.s_Q + M.noise*I) for _ in 1:M.N_fixed])
        
        for k in 1:M.N_fixed
            # Each covariate gets its own spatial scale and field
            # Use the canonical ICAR or standard spatial template provided in M.s_Q
            s_svc[:, k] = svc_raw[:, k] .* svc_sigma[k]
        end

    elseif M.model_space == "dag"
        s_sigma ~ Exponential(1.0)
        s_rho ~ Beta(1, 1)
        # 1. Construct the transformation matrix L = (I - rho * W_scaled)
        # Where W_scaled accounts for the number of predecessors per node
        W_adj = Matrix(M.W)
        row_sums = vec(sum(W_adj, dims=2))
        # Create the weighted adjacency matrix for the DAG
        W_scaled = zeros(eltype(s_rho), M.N_areas, M.N_areas)
        for i in 1:M.N_areas
            preds = findall(x -> x == 1, W_adj[i, 1:i-1])
            if !isempty(preds)
                W_scaled[i, preds] .= s_rho / length(preds)
            end
        end

        L = I(M.N_areas) - W_scaled

        # 2. Derive the structural precision matrix s_Q_dag = L' * L
        # This represents the internal spatial structure before variance scaling
        s_Q_structural = L' * L

        # 3. Apply marginal scaling (s_sigma) to get the final s_Q
        s_Q = (1.0 / s_sigma^2) .* s_Q_structural + M.noise * I

        # For DAG, Q = (I - rho*W)' * (I - rho*W)
        s_eta ~ MvNormalCanon(zeros(M.N_areas), s_Q)

    elseif M.model_space == "local"
        # 1. Cluster-Specific Hyper-priors
        # mu_clusters represents the mean of each spatial partition (mosaic)
        s_sigma ~ Exponential(1.0)
        mu_clusters ~ filldist(Normal(0, 1), M.n_clusters)

        # 2. Leroux Precision Construction
        # Combines structured ICAR (M.s_Q) and unstructured IID (I) components
        s_rho ~ Beta(1, 1)
        s_Q_base = s_rho .* M.s_Q .+ (1 - s_rho) .* I(M.N_areas)

        # 3. Latent Field Sampling
        # Scale the precision by the inverse variance (1/s_sigma^2)
        # and map cluster means to the full spatial field via M.cluster_assignments
        s_Q = (1.0 / (s_sigma^2 + M.noise)) .* s_Q_base + M.noise * I
        s_eta_raw ~ MvNormalCanon(mu_clusters[M.cluster_assignments], s_Q )

        # 4. Observation Mapping
        s_eta = s_eta_raw

    elseif M.model_space == "svgp"

        sigma_f ~ Exponential(1.0)
        ls_st ~ filldist(Gamma(2, 2), size(M.Z_inducing_coords, 2))
        s_sigma ~ Exponential(1.0)

        # Define the kernel using sigma_f and ls_st
        kernel_svgp = sigma_f * SqExponentialKernel() ∘ ScaleTransform(ls_st)
        # Compute the covariance matrix between inducing points
        Kuu = kernelmatrix(kernel_svgp, RowVecs(M.Z_inducing_coords))

        # Variational parameters for inducing points
        m_u ~ filldist(Normal(0, 1), M.M_inducing_count)
        s_u_diag ~ filldist(Exponential(1.0), M.M_inducing_count)
        # Latent inducing points, with prior covariance based on Kuu and variational diagonal
        u_latent ~ MvNormal(m_u, Symmetric(Kuu + Diagonal(s_u_diag.^2)))
        # Project from inducing points to observation locations
        s_eta = vec( (M.Z_inducing_proj * u_latent) .* s_sigma )

    elseif M.model_space == "none"
        s_eta = zeros(T, M.N_areas)
        s_Q = I(M.N_areas)

    else
        # nothing to do
    end
 
    # --- 3. Temporal Manifold (t_eta) ---
 
    local t_eta = zeros(M.N_time)
    local t_Q = I(M.N_time)

    if haskey(M, :N_time) && M.N_time > 1

        if M.model_time == "ar1"
            t_sigma ~ Exponential(1.0)
            t_rho ~ Beta(2, 2)
            # Construct the standardized AR1 precision matrix
            # The template M.t_Q represents the first-order difference structure
            t_Q_base = Symmetric((1.0 + t_rho^2) .* I(M.N_time) .+ (t_rho) .* M.t_Q)

            # Normalize by the variance scale and add stability jitter
            t_Q = (1.0 / (1.0 - t_rho^2 + M.noise)) .* t_Q_base

            # Use the Canonical form of the Multivariate Normal for precision-based sampling
            t_raw ~ MvNormalCanon(zeros(M.N_time), Symmetric(Matrix(t_Q + M.noise * I)) )
            t_eta = t_raw .* t_sigma

        elseif M.model_time == "rw2"
            t_sigma ~ Exponential(1.0)
            # 1. Structural Template: M.t_Q (the second-order difference matrix)
            # 2. Recompose Precision: Scale the structural template by the inverse variance
            # Q_rw2 = (1 / sigma^2) * Q_template
            t_Q = (1.0 / (t_sigma^2 + M.noise)) .* M.t_Q

            # 3. Sampling: Use the Canonical (Precision) form
            # We add a small noise term to the diagonal to ensure the matrix is PosDef
            t_raw ~ MvNormalCanon(zeros(M.N_time), Matrix(t_Q + M.noise * I))
            t_eta = t_raw .* t_sigma

        elseif M.model_time == "gp"
            t_sigma ~ Exponential(1.0)
            t_ls ~ InverseGamma(3, 3) # GP Lengthscale
            # 1. Construct Kernel Matrix (Covariance Space)
            t_K = (t_sigma^2) .* kernelmatrix(SqExponentialKernel() ∘ ScaleTransform(inv(t_ls)), 1.0:Float64(M.N_time)) + M.noise * I

            # 2. Latent Field Sampling
            # Using MvNormal for covariance-based sampling
            t_eta ~ MvNormal(zeros(M.N_time), Symmetric(t_K))

            # 3. Derive Precision Matrix (t_Q)
            # Required for Type II and Type IV space-time interactions
            t_Q = inv(Symmetric(t_K))

        elseif M.model_time == "harmonic"
            t_sigma ~ Exponential(1.0)
            t_α ~ Normal(0, 1)
            t_β ~ Normal(0, 1) 

            # 1. Linear Predictor Composition (Seasonal Harmonics)
            # Uses precomputed angles from the data container M
            t_eta = (t_α .* sin.(M.t_angle) .+ t_β .* cos.(M.t_angle)) .* t_sigma

            # 2. Derive Structural Precision Matrix (t_Q)
            # Required for space-time interactions (Types II/IV).
            # We use a cyclic RW2 template to represent the periodic dependency structure.
            # Note: t_sigma here acts as the marginal amplitude scale.
            t_Q = (1.0 / (t_sigma^2 + M.noise)) .* M.t_Q

        elseif M.model_time == "iid"
            t_sigma ~ Exponential(1.0)
            t_eta ~ MvNormal(zeros(M.N_time), t_sigma*I)

            t_Q = I(M.N_time)

        elseif M.model_time == "none"
            t_eta = zeros(M.N_time)
            t_Q = I(M.N_time)

        else # Default to no effects
            # nothing to do
        end
    end


    # --- 4. Season Manifold (u_eta) ---
    local u_eta = zeros(M.N_season)

    if haskey(M, :N_season) && M.N_season > 1

        if M.model_season == "ar1"
            u_sigma ~ Exponential(0.5)
            u_rho ~ Beta(2, 2)
            u_Q = Symmetric((1.0 / (1.0 - u_rho^2 + M.noise)) .* (M.u_Q + (u_rho^2) * I))
            u_raw ~ MvNormalCanon(zeros(M.N_season), u_Q)
            u_eta = u_raw .* u_sigma
        elseif M.model_season == "rw2"
            u_raw ~ MvNormalCanon(zeros(M.N_season), M.u_Q + M.noise*I)
            u_eta = u_raw .* u_sigma
        elseif M.model_season == "gp"
            u_ls ~ InverseGamma(3, 3)
            u_K = kernelmatrix(SqExponentialKernel() ∘ ScaleTransform(1.0/u_ls), 1.0:Float64(M.N_season)) + M.noise*I
            u_gp ~ MvNormal(zeros(M.N_season), u_K)
            u_eta = u_gp .* u_sigma
        elseif M.model_season == "harmonic"
            u_α ~ Normal(0, 1)
            u_β ~ Normal(0, 1)
            u_steps = 1:M.N_season # Assuming M.N_season represents the period for seasonal cycles
            u_eta = (u_α .* sin.(2π .* u_steps ./ 12.0) .+ u_β .* cos.(2π .* u_steps ./ 12.0)) .* u_sigma
        elseif M.model_season == "iid"
            u_iid ~ MvNormal(zeros(M.N_season), I)
            u_eta = u_iid .* u_sigma
        else
            # nothing to do
        end

    end

    
    # --- 5. Space-Time Interaction (st_eta) ---
    # Captures localized deviations across both space and time
 
    local st_eta = zeros(M.N_areas, M.N_time)

    if M.model_st == 0 # None
        st_eta = zeros(M.N_areas, M.N_time)

    elseif M.model_st == 1 # Type I: IID Interaction
        st_sigma ~ Exponential(0.5)
        st_raw ~ MvNormal(zeros(M.N_areas * M.N_time), I)
        st_eta = reshape(st_raw .* st_sigma, M.N_areas, M.N_time)

    elseif M.model_st == 2 # Type II: Temporal Structure (I ⊗ Q_t)
        # Each area has its own structured temporal trend
        st_sigma ~ Exponential(0.5)
        st_Q2 = kron(I(M.N_areas), t_Q)
        st_raw ~ MvNormalCanon(zeros(M.N_areas * M.N_time), st_Q2)
        st_Q2 = Matrix(kron(I(M.N_areas), t_Q))
        st_raw ~ MvNormalCanon(zeros(M.N_areas * M.N_time), Matrix(st_Q2))
        st_eta = reshape(st_raw .* st_sigma, M.N_areas, M.N_time)

    elseif M.model_st == 3 # Type III: Spatial Structure (Q_s ⊗ I)
        # Each time point has its own structured spatial field
        st_sigma ~ Exponential(0.5)
        st_Q3 = kron(s_Q, I(M.N_time))
        st_raw ~ MvNormalCanon(zeros(M.N_areas * M.N_time), st_Q3)
        st_Q3 = Matrix(kron(s_Q, I(M.N_time)))
        st_raw ~ MvNormalCanon(zeros(M.N_areas * M.N_time), Matrix(st_Q3))
        st_eta = reshape(st_raw .* st_sigma, M.N_areas, M.N_time)

    elseif M.model_st == 4 # Type IV: Inseparable (Q_s ⊗ Q_t)
        # Fully inseparable spatiotemporal structure
        st_sigma ~ Exponential(0.5)
        st_Q4 = kron(s_Q, t_Q)
        st_raw ~ MvNormalCanon(zeros(M.N_areas * M.N_time), st_Q4)
        st_Q4 = Matrix(kron(s_Q, t_Q))
        st_raw ~ MvNormalCanon(zeros(M.N_areas * M.N_time), Matrix(st_Q4))
        st_eta = reshape(st_raw .* st_sigma, M.N_areas, M.N_time)
    else
        # nothing to do 
    end


    # --- 6. Covariate Smoothing (c_beta) ---
    # c_beta handles discretized continuous covariates (RW2/IID) or RFF-based GP
    
    local c_eta = zeros(M.N_cat)
    
    if M.N_cov > 0
            
        if M.model_cov == "ar1"
            c_rho ~ Beta(2, 2) 
            c_Q = (1.0 / (1.0 - c_rho^2 + M.noise)) .* ( (1.0 + c_rho^2) .* I(M.N_cat) .+ (c_rho) .* M.c_Q )  
            c_raw ~ MvNormalCanon(zeros(M.N_cat), Matrix(c_Q))
            c_eta = c_raw .* c_sigma
        elseif M.model_cov == "rw2"
            c_sigma ~  Exponential(1.0)
            c_Q = (1.0 / (c_sigma^2 + M.noise)) .* M.c_Q
            c_raw ~ MvNormalCanon(zeros(M.N_cat), Matrix(c_Q) )
            c_eta = c_raw .* c_sigma
        elseif M.model_cov == "gp"
            c_sigma ~  Exponential(1.0)
            c_ls ~  InverseGamma(3, 3)
            c_K = kernelmatrix(SqExponentialKernel() ∘ ScaleTransform(1.0/c_ls), 1.0:Float64(M.N_cat)) + M.noise*I
            c_gp ~ MvNormal(zeros(M.N_cat), Matrix(c_K) )
            c_eta = c_gp .* c_sigma
        elseif M.model_cov == "harmonic"
            c_sigma ~  Exponential(1.0)
            h_c_α ~ Normal(0, 1);
            h_c_β ~ Normal(0, 1)
            c_steps = 1:M.N_cat
            c_eta = (h_c_α .* sin.(2π .* c_steps ./ 12.0) .+ h_c_β .* cos.(2π .* c_steps ./ 12.0)) .* c_sigma
        elseif M.model_cov == "rff"
            c_sigma ~  Exponential(1.0)
            W_rff ~ MvNormal(zeros(M.N_rff), I)
            B_rff ~ filldist(Uniform(0, 2π), M.N_rff)
            # Projections are usually precomputed in M for speed, here conceptualized:
            c_eta = (cos.( (1:M.N_cat) * W_rff' .+ B_rff' ) * ones(M.N_rff)) .* c_sigma
        elseif M.model_cov == "iid"
            c_sigma ~  Exponential(1.0)
            c_iid ~ MvNormal(zeros(M.N_cat), I)
            c_eta = c_iid .* c_sigma
        else
            # nothing to do
        end
    end

    # --- 6. Linear Predictor (eta) ---
    eta = zeros(T, M.N_obs) 
    
    if haskey(M, :log_offset) 
        eta .+= M.log_offset 
    end
    
    if !isnothing(s_eta)
        if length(s_eta) == M.N_areas
            eta .+= s_eta[M.s_idx]
        else
            eta .+= s_eta
        end
    end

    if !isnothing(t_eta)
        eta .+= t_eta[M.t_idx]
    end

    if !isnothing(u_eta)
        eta .+= u_eta[M.u_idx]
    end

    if !isnothing(st_eta)
        for i in 1:M.N_obs; 
            eta[i] += st_eta[M.s_idx[i], M.t_idx[i]]; 
        end
    end

    if !isnothing(c_eta) 
        for i in 1:M.N_cov
            eta .+= c_eta[M.cov_indices[:, i]] 
        end
    end

    if M.N_fixed > 0
        if M.model_space == "svc"

            # Assemble eta component: sum(beta_k(s) * x_ik)
            for i in 1:M.N_obs
                a = M.s_idx[i]
                for k in 1:M.N_fixed
                    eta[i] += s_svc[a, k] * M.fixed[i, k]
                end
            end

        else 
            # fixed effects
            d_beta ~ MvNormal(zeros(M.N_fixed), I)
            eta += M.fixed * d_beta'
        end
    end
  
    # --- 7. Likelihood (handling missing observations) ---
    # Filter out missing observations and corresponding linear predictors/weights/trials
    # This ensures logpdf only operates on valid data points.
    good_indices = findall(!ismissing, M.y_obs)
    y_obs_filtered = M.y_obs[good_indices]
    eta_filtered = eta[good_indices]
    weights_filtered = M.weights[good_indices]
    trials_filtered = M.trials[good_indices]

    Turing.@addlogprob! logpdf(bstm_Likelihood(M.model_family, M.use_zi, weights_filtered, phi_zi, r_nb, y_sigma, trials_filtered, y_obs_filtered), eta_filtered)

end




@model function bstm_multioutcome(M, ::Type{T}=Float64) where {T}
    # --- 1. SETUP & HYPERPRIORS ---
    N_areas, N_time, N_outcomes = M.N_areas, M.N_time, M.N_outcomes
    N_obs_points = M.N_obs
    N_season = M.N_season

    # Global Scales & Correlations
    s_sigma ~ filldist(Exponential(1.0), N_outcomes)
    t_sigma ~ filldist(Exponential(1.0), N_outcomes)
    local st_sigma
    if M.model_st > 0
        st_sigma ~ filldist(Exponential(0.5), N_outcomes)
    else
        st_sigma = zeros(N_outcomes)
    end

    local u_sigma
    if M.model_season != "none"
        u_sigma ~ filldist(Exponential(0.5), N_outcomes)
    else
        u_sigma = zeros(N_outcomes)
    end

    local c_sigma
    if M.N_cov > 0
        c_sigma ~ filldist(Exponential(1.0), N_outcomes)
    else
        c_sigma = zeros(N_outcomes)
    end


    # Cross-outcome correlation on spatial manifold
    L_corr ~ LKJCholesky(N_outcomes, 1.0, :L)

    # Distribution specific hyperpriors
    local r_nb
    if M.model_family == "negbin"
        r_nb ~ filldist(Exponential(1.0), N_outcomes)
    else
        r_nb = fill(1.0, N_outcomes)
    end

    local phi_zi
    if M.use_zi
        phi_zi ~ Beta(1, 1)
    else
        phi_zi = 0.0
    end

    local y_sigma = 1.0
    if get(M, :use_sv, false) 
        sigma_log_var ~ Exponential(1.0)
        beta_vol ~ filldist(Normal(0, sigma_log_var), M.M_rff_sigma)
        coords_st = hcat(M.s_obs, M.t_obs ./ N_time)
        vol_proj = (coords_st * M.W_sigma_fixed) .+ M.b_sigma_fixed'
        y_sigma_vec = exp.((sqrt(2.0 / M.M_rff_sigma) .* cos.(vol_proj) * beta_vol) ./ 2.0)
        y_sigma = [y_sigma_vec for _ in 1:N_outcomes]
    else
        if M.model_family in ["gaussian", "lognormal"]
            y_sigma_const ~ filldist(Exponential(1.0), N_outcomes)
            y_sigma = [fill(y_sigma_const[k], N_obs_points) for k in 1:N_outcomes]
        else
            y_sigma = [fill(1.0, N_obs_points) for k in 1:N_outcomes]
        end
    end

    # --- 2. SPATIAL MANIFOLDS ---
    # Mixing/Correlation parameters
    local s_rho
    if M.model_space in ["bym2", "leroux", "sar", "fft", "dag", "local"]
        s_rho ~ filldist(Beta(1, 1), N_outcomes)
    else
        s_rho = fill(1.0, N_outcomes)
    end

    s_eta_scaled = Matrix{T}(undef, N_areas, N_outcomes)
    s_Q = M.s_Q

    for k in 1:N_outcomes
        if M.model_space == "iid"
            s_iid_k ~ MvNormal(zeros(N_areas), I)
            s_eta_scaled[:, k] = s_iid_k
        elseif M.model_space in ["besag", "icar"]
            s_icar_k ~ MvNormalCanon(zeros(N_areas), M.s_Q + M.noise*I)
            s_eta_scaled[:, k] = s_icar_k
        elseif M.model_space == "bym2"
            s_icar_k ~ MvNormalCanon(zeros(N_areas), M.s_Q + M.noise*I)
            s_iid_k ~ MvNormal(zeros(N_areas), I)
            s_eta_scaled[:, k] = sqrt.(s_rho[k]) .* s_icar_k .+ sqrt(1 .- s_rho[k]) .* s_iid_k
        elseif M.model_space == "leroux"
            Q_k = s_rho[k] .* M.s_Q .+ (1 - s_rho[k]) .* I(N_areas)
            s_raw_k ~ MvNormalCanon(zeros(N_areas), Q_k + M.noise * I)
            s_eta_scaled[:, k] = s_raw_k
        elseif M.model_space == "sar"
            W_row_norm = M.W ./ (vec(sum(M.W, dims=2)) .+ M.noise)
            L_sar_k = I(N_areas) - s_rho[k] .* W_row_norm
            s_Q_k = Symmetric(L_sar_k' * L_sar_k) + M.noise * I
            s_sar_raw_k ~ MvNormalCanon(zeros(N_areas), s_Q_k)
            s_eta_scaled[:, k] = s_sar_raw_k
        elseif M.model_space == "bgcn"
            gcn_raw_k ~ MvNormal(zeros(N_areas), I)
            D_inv_sqrt = Diagonal(1.0 ./ sqrt.(vec(sum(M.W, dims=2)) .+ M.noise))
            W_norm = D_inv_sqrt * M.W * D_inv_sqrt
            s_eta_scaled[:, k] = W_norm * gcn_raw_k
        elseif M.model_space == "mosaic"
            mu_local_k ~ filldist(Normal(0, 1), M.n_mosaics)
            s_eta_scaled[:, k] = mu_local_k[M.cluster_assignments]
        elseif M.model_space == "warping"
            w_warp_k ~ filldist(Normal(0, 1), M.M_rff)
            warp_proj = (M.s_obs * M.W_fixed) .+ M.b_fixed'
            s_eta_scaled[:, k] = (sqrt(2.0 / M.M_rff) .* cos.(warp_proj) * w_warp_k)
        elseif M.model_space == "fitcGP" || M.model_space == "svgp"
            u_inducing_k ~ filldist(Normal(0, 1), M.M_inducing_count)
            s_eta_scaled[:, k] = M.Z_inducing_proj * u_inducing_k
        elseif M.model_space == "fft"
            s_spectral_raw_k ~ MvNormal(zeros(N_areas), I)
            s_eta_scaled[:, k] = M.s_Q \ s_spectral_raw_k
        elseif M.model_space == "dag"
            W_adj = Matrix(M.W)
            W_scaled = zeros(T, N_areas, N_areas)
            for i in 1:N_areas
                preds = findall(x -> x == 1, W_adj[i, 1:i-1])
                if !isempty(preds); W_scaled[i, preds] .= s_rho[k] / length(preds); end
            end
            L_dag = I(N_areas) - W_scaled
            s_Q_k = Symmetric(L_dag' * L_dag) + M.noise * I
            s_dag_raw_k ~ MvNormalCanon(zeros(N_areas), s_Q_k)
            s_eta_scaled[:, k] = s_dag_raw_k
        else
            s_eta_scaled[:, k] = zeros(N_areas)
        end
    end

    # Unified spatial field with cross-outcome linkage
    s_eta_field = (s_eta_scaled * L_corr) .* s_sigma'

    # --- 3. TEMPORAL, SEASONAL & COVARIATE MANIFOLDS ---
    t_eta_field = Matrix{T}(undef, N_time, N_outcomes)
    u_eta_field = Matrix{T}(undef, N_season, N_outcomes)
    c_eta_field = Matrix{T}(undef, M.N_cat, N_outcomes)
    t_Q_list = Vector{Any}(undef, N_outcomes)

    for k in 1:N_outcomes
        # --- Temporal Options ---
        if M.model_time == "ar1"
            t_rho_k ~ Beta(2, 2)
            t_Q_k = (1.0 / (1.0 - t_rho_k^2 + M.noise)) .* Symmetric((1.0 + t_rho_k^2) .* I(N_time) .+ (t_rho_k) .* M.t_Q)
            t_raw_k ~ MvNormalCanon(zeros(N_time), t_Q_k + M.noise * I)
            t_eta_field[:, k] = t_raw_k .* t_sigma[k]
            t_Q_list[k] = t_Q_k
        elseif M.model_time == "rw2"
            t_Q_k = (1.0 / (t_sigma[k]^2 + M.noise)) .* M.t_Q
            t_raw_k ~ MvNormalCanon(zeros(N_time), t_Q_k + M.noise * I)
            t_eta_field[:, k] = t_raw_k .* t_sigma[k]
            t_Q_list[k] = t_Q_k
        elseif M.model_time == "gp"
            t_ls_k ~ InverseGamma(3, 3)
            t_K_k = (t_sigma[k]^2) .* kernelmatrix(SqExponentialKernel() ∘ ScaleTransform(inv(t_ls_k)), 1.0:Float64(N_time)) + M.noise * I
            t_gp_k ~ MvNormal(zeros(N_time), Symmetric(t_K_k))
            t_eta_field[:, k] = t_gp_k
            t_Q_list[k] = inv(Symmetric(t_K_k))
        elseif M.model_time == "harmonic"
            t_alpha_k ~ Normal(0, 1); 
            t_beta_k ~ Normal(0, 1)
            t_eta_field[:, k] = (t_alpha_k .* sin.(M.t_angle) .+ t_beta_k .* cos.(M.t_angle)) .* t_sigma[k]
            t_Q_list[k] = (1.0 / (t_sigma[k]^2 + M.noise)) .* M.t_Q
        elseif M.model_time == "iid"
            t_eta_field[:, k] ~ MvNormal(zeros(N_time), t_sigma[k] * I)
            t_Q_list[k] = I(N_time)
        else
            t_Q_list[k] = I(N_time)
        end

        # --- Seasonal Options ---
        if M.model_season == "ar1"
            u_rho_k ~ Beta(2, 2)
            u_Q_k = (1.0 / (1.0 + M.noise)) .* (M.u_Q + M.noise*I)
            u_raw_k ~ MvNormalCanon(zeros(N_season), u_Q_k)
            u_eta_field[:, k] = u_raw_k .* u_sigma[k]
        elseif M.model_season == "rw2"
            u_raw_k ~ MvNormalCanon(zeros(N_season), M.u_Q + M.noise*I)
            u_eta_field[:, k] = u_raw_k .* u_sigma[k]
        elseif M.model_season == "gp"
            u_ls_k ~ InverseGamma(3, 3)
            u_K_k = (u_sigma[k]^2) .* kernelmatrix(SqExponentialKernel() ∘ ScaleTransform(inv(u_ls_k)), 1.0:Float64(N_season)) + M.noise * I
            u_gp_k ~ MvNormal(zeros(N_season), Symmetric(u_K_k))
            u_eta_field[:, k] = u_gp_k
        elseif M.model_season == "harmonic"
            u_alpha_k ~ Normal(0, 1); 
            u_beta_k ~ Normal(0, 1)
            u_steps = 1:N_season
            u_eta_field[:, k] = (u_alpha_k .* sin.(2π .* u_steps ./ 12.0) .+ u_beta_k .* cos.(2π .* u_steps ./ 12.0)) .* u_sigma[k]
        elseif M.model_season == "iid"
            u_iid_k ~ MvNormal(zeros(N_season), I)
            u_eta_field[:, k] = u_iid_k .* u_sigma[k]
        end

        if M.N_cov > 0
            if M.model_cov == "ar1"
                c_rho_k ~ Beta(2, 2)
                c_Q_k = (1.0 / (1.0 - c_rho_k^2 + M.noise)) .* ((1.0 + c_rho_k^2) .* I(M.N_cat) .+ (c_rho_k) .* M.c_Q)
                c_raw_k ~ MvNormalCanon(zeros(M.N_cat), Matrix(c_Q_k))
                c_eta_field[:, k] = c_raw_k .* c_sigma[k]
            elseif M.model_cov == "rw2"
                c_raw_k ~ MvNormalCanon(zeros(M.N_cat), M.c_Q + M.noise * I)
                c_eta_field[:, k] = c_raw_k .* c_sigma[k]
            elseif M.model_cov == "gp"
                c_ls_k ~ InverseGamma(3, 3)
                c_K_k = (c_sigma[k]^2) .* kernelmatrix(SqExponentialKernel() ∘ ScaleTransform(1.0 / c_ls_k), 1.0:Float64(M.N_cat)) + M.noise * I
                c_gp_k ~ MvNormal(zeros(M.N_cat), Matrix(c_K_k))
                c_eta_field[:, k] = c_gp_k
            elseif M.model_cov == "harmonic"
                h_c_alpha_k ~ Normal(0, 1)
                h_c_beta_k ~ Normal(0, 1)
                c_steps = 1:M.N_cat
                c_eta_field[:, k] = (h_c_alpha_k .* sin.(2π .* c_steps ./ 12.0) .+ h_c_beta_k .* cos.(2π .* c_steps ./ 12.0)) .* c_sigma[k]
            elseif M.model_cov == "rff"
                W_rff_k ~ MvNormal(zeros(M.N_rff), I)
                B_rff_k ~ filldist(Uniform(0, 2π), M.N_rff)
                c_eta_field[:, k] = (cos.((1:M.N_cat) * W_rff_k' .+ B_rff_k') * ones(M.N_rff)) .* c_sigma[k]
            elseif M.model_cov == "iid"
                c_iid_k ~ MvNormal(zeros(M.N_cat), I)
                c_eta_field[:, k] = c_iid_k .* c_sigma[k]
            else
                # Default to zeros if no specific model is matched
                c_eta_field[:, k] = zeros(M.N_cat)
            end
        end
    end

    # --- 4. LIKELIHOOD ASSEMBLY ---
    for k in 1:N_outcomes
        eta_k = M.log_offset[:, k] .+ s_eta_field[M.s_idx, k] .+ t_eta_field[M.t_idx, k]
        if M.model_season != "none"; eta_k .+= u_eta_field[M.u_idx, k]; end
        if M.N_cov > 0; eta_k .+= c_eta_field[M.cov_indices[:, 1], k]; end

        if M.model_space == "svc" && M.N_fixed > 0
            # Multivariate SVC logic using shared s_eta_scaled components
            svc_sigma_k ~ Exponential(0.5)
            for d in 1:M.N_fixed; eta_k .+= (s_eta_scaled[M.s_idx, k] .* svc_sigma_k) .* M.fixed[:, d]; end
        elseif M.N_fixed > 0
            d_beta_k ~ filldist(Normal(0, 1), M.N_fixed)
            eta_k .+= M.fixed * d_beta_k
        end

        if M.model_st > 0
            local st_Q
            if M.model_st == 1; st_Q = I(N_areas * N_time)
            elseif M.model_st == 2; st_Q = kron(I(N_areas), Symmetric(t_Q_list[k]))
            elseif M.model_st == 3; st_Q = kron(M.s_Q, I(N_time))
            elseif M.model_st == 4; st_Q = kron(M.s_Q, Symmetric(t_Q_list[k])); end

            st_raw_k ~ MvNormalCanon(zeros(N_areas * N_time), Symmetric(Matrix(st_Q + M.noise * I)))
            st_field_k = reshape(st_raw_k .* st_sigma[k], N_areas, N_time)
            for i in 1:N_obs_points; eta_k[i] += st_field_k[M.s_idx[i], M.t_idx[i]]; end
        end

        # --- Likelihood (handling missing observations for this outcome) ---
        # Filter out missing observations and corresponding linear predictors/weights/trials
        # This ensures logpdf only operates on valid data points.
        
        y_obs_filtered_k = M.y_obs[M.y_has_data, k]
        eta_filtered_k = eta_k[M.y_has_data]
        weights_filtered_k = M.weights[M.y_has_data]
        trials_filtered_k = M.trials[M.y_has_data]

        Turing.@addlogprob! logpdf(bstm_Likelihood(M.model_family, M.use_zi, weights_filtered_k, phi_zi, r_nb[k], y_sigma[k], trials_filtered_k, y_obs_filtered_k), eta_filtered_k)
    end
end


 
@model function bstm_multivariate(M, ::Type{T}=Float64) where {T}
    # --- 1. SETUP & DIMENSIONS ---
    N_obs, N_outcomes = size(M.y_obs)
    N_factors = get(M, :N_factors, 2)
    nvh = Int(N_outcomes * N_factors - N_factors * (N_factors - 1) / 2)

    # --- 2. LATENT PCA MANIFOLD ---
    pca_sd ~ Bijectors.ordered(arraydist(LogNormal.(get(M, :sigma_prior, zeros(N_factors)), 1.0)))
    pca_pdef_sd ~ LogNormal(0.0, 0.5)
    v ~ filldist(Normal(0.0, 1.0), nvh)

    # Householder reconstruction: Kmat is the residual/uniqueness matrix
    Kmat, r_diag, U = householder_transform(v, N_outcomes, N_factors, M.ltri, pca_sd, pca_pdef_sd, M.noise)

    # --- 3. LATENT FACTOR MANIFOLDS ---
    pc_scores = Matrix{T}(undef, N_obs, N_factors)

    for k in 1:N_factors
        local s_eta_k = zeros(T, M.N_areas)
        local t_eta_k = zeros(T, M.N_time)
        local st_inter_k = zeros(T, M.N_areas, M.N_time)

        # A. Spatial Manifold (Factor-Specific)
        s_sigma_k ~ Exponential(1.0)
        if M.model_space == "bym2"
            s_rho_k ~ Beta(1, 1)
            s_icar_k ~ MvNormalCanon(zeros(M.N_areas), Symmetric(Matrix(M.s_Q + M.noise*I)))
            s_iid_k ~ MvNormal(zeros(M.N_areas), I)
            s_eta_k = s_sigma_k .* (sqrt(s_rho_k) .* s_icar_k .+ sqrt(1 - s_rho_k) .* s_iid_k)
        elseif M.model_space == "besag"
            s_icar_k ~ MvNormalCanon(zeros(M.N_areas), Symmetric(Matrix(M.s_Q + M.noise*I)))
            s_eta_k = s_icar_k .* s_sigma_k
        elseif M.model_space == "sar"
            s_rho_k ~ Beta(1, 1)
            W_rn = M.W ./ (vec(sum(M.W, dims=2)) .+ M.noise)
            L_sar = I(M.N_areas) - s_rho_k .* W_rn
            s_eta_k ~ MvNormalCanon(zeros(M.N_areas), Symmetric(Matrix(L_sar' * L_sar) + (s_sigma_k + M.noise)*I))
        end # (Other M.model_space types like DAG/Mosaic follow same pattern as bstm())

        # B. Temporal Manifold (Factor-Specific)
        t_sigma_k ~ Exponential(1.0)
        if M.model_time == "ar1"
            t_rho_k ~ Beta(2, 2)
            t_Q_k = build_bstm_ar1_precision(M.t_Q, t_rho_k; noise=M.noise)
            t_raw_k ~ MvNormalCanon(zeros(M.N_time), t_Q_k)
            t_eta_k = t_raw_k .* t_sigma_k
        elseif M.model_time == "rw2"
            t_Q_k = (1.0 / (t_sigma_k^2 + M.noise)) .* M.t_Q
            t_eta_k ~ MvNormalCanon(zeros(M.N_time), Symmetric(Matrix(t_Q_k + M.noise*I)))
        end

        # C. Space-Time Interaction (Factor-Specific)
        if M.model_st > 0
            st_sigma_k ~ Exponential(0.5)
            # Use factor-specific precision derived from s_Q and t_Q_k
            # Simplified Type I (IID) for example:
            st_raw_k ~ MvNormal(zeros(M.N_areas * M.N_time), I)
            st_inter_k = reshape(st_raw_k .* st_sigma_k, M.N_areas, M.N_time)
        end

        # Assemble scores for factor k
        for i in 1:N_obs
            a, t = M.s_idx[i], M.t_idx[i]
            pc_scores[i, k] = s_eta_k[a] + t_eta_k[t] + st_inter_k[a, t]
        end
    end

    # --- 4. SHARED COVARIATE EFFECTS ---
    # These apply to the predictor matrix AFTER projection to outcome space
    mu_matrix = pc_scores * U'

    if M.N_cov > 0
        c_sigma ~ Exponential(1.0)
        if M.model_cov == "rw2"
            c_Q = (1.0 / (c_sigma^2 + M.noise)) .* M.c_Q
            c_eta ~ MvNormalCanon(zeros(M.N_cat), Symmetric(Matrix(c_Q + M.noise*I)))
            for i in 1:N_obs
                mu_matrix[i, :] .+= c_eta[M.cov_indices[i, 1]]
            end
        end
    end

    if M.N_fixed > 0
        d_beta ~ filldist(Normal(0, 1), M.N_fixed, N_outcomes)
        mu_matrix .+= M.fixed * d_beta
    end

    if haskey(M, :log_offset)
        mu_matrix .+= M.log_offset
    end

    # --- 5. MULTIVARIATE LIKELIHOOD ---
    Kmat_sym = Symmetric(Kmat)
    M.y_obs ~ arraydist([MvNormal(mu_matrix[i, :], Kmat_sym) for i in 1:N_obs])
end




@model function bstm_multifidelity(M, ::Type{T}=Float64) where {T}
    
    
    # The `bstm_multifidelity` model is a sophisticated Bayesian spatio-temporal framework designed to integrate data from multiple fidelity levels (e.g., high-resolution observations and lower-resolution simulations or auxiliary data). It aims to leverage the strengths of each data source to produce a more robust and accurate inference of the underlying latent processes. The model achieves this by:
    #
    # 1.  **Global Hyperpriors:** Setting up overarching priors for key model parameters like observation noise, dispersion, and zero-inflation.
    # 2.  **Nested Latent Covariate Structure (RFF-based Multi-Fidelity):** This is the core multi-fidelity component. It models latent spatial (`z_latent`) and spatiotemporal (`w_latent`) processes using Random Fourier Features (RFF). Crucially, the medium-fidelity `w_latent` process *depends* on the high-fidelity `z_latent` process, creating a nested structure that allows information flow between fidelity levels. This captures complex, non-linear relationships between covariates and observations.
    # 3.  **Spatio-Temporal Manifolds:** Incorporating traditional Bayesian spatio-temporal components to capture residual variation:
    #     *   **Spatial (`s_eta`):** Models spatial autocorrelation using a combination of ICAR (Intrinsic Conditional Autoregressive) and IID (Independent and Identically Distributed) effects, allowing for flexible spatial smoothing.
    #     *   **Temporal (`t_eta_full`):** Models temporal dependence using various structures like AR(1), RW2 (Random Walk of order 2), Gaussian Processes (GP), or IID effects.
    #     *   **Seasonality (`u_eta_full`):** Captures periodic patterns in data, also offering AR(1), RW2, GP, or IID structures.
    # 4.  **Space-Time Interaction (`st_eta`):** Allows for different types of interactions between spatial and temporal effects, ranging from IID to fully inseparable structures (Knorr-Held Types 0-4), enabling the model to capture non-additive spatio-temporal dynamics.
    # 5.  **Linear Predictor Construction:** Combines all latent components (spatial, temporal, seasonality, multi-fidelity RFFs, and interaction terms) into a single linear predictor (`eta`), which represents the expected value of the response. Fixed effects are also included.
    # 6.  **Joint Multi-fidelity Likelihood:** The model specifies likelihoods for both the multi-fidelity latent observations (`z_obs`, `w_obs`) and the primary observations (`y_obs`), linking the latent processes to the observed data. This allows the model to learn from all available data simultaneously.
    #
    # In essence, `bstm_multifidelity` is designed to be highly flexible, combining established spatio-temporal modeling techniques with modern multi-fidelity approaches to handle complex data structures and improve predictive performance by integrating information from diverse sources.
    # 

    # --- 1. Global Hyperpriors ---
    local r_nb
    if M.model_family == "negbin"
        r_nb ~ Exponential(1.0)
    else
        r_nb = 1.0
    end

    local phi_zi
    if M.use_zi
        phi_zi ~ Beta(1, 1)
    else
        phi_zi = 0.0
    end

    local y_sigma_val
    if get(M, :use_sv, false) && M.model_family in ["gaussian", "lognormal"]
        sigma_log_var ~ Exponential(1.0)
        beta_vol ~ filldist(Normal(0, sigma_log_var), M.M_rff_sigma)
        # Use spatiotemporal coordinates for volatility surface
        coords_st = hcat(M.s_obs, M.t_obs ./ M.N_time)
        vol_proj = (coords_st * M.W_sigma_fixed) .+ M.b_sigma_fixed'
        # Map RFF to positive scale via exponential link
        y_sigma_val = exp.((sqrt(2.0 / M.M_rff_sigma) .* cos.(vol_proj) * beta_vol) ./ 2.0)
    else
        if M.model_family in ["gaussian", "lognormal"]
            y_sigma_val ~ Exponential(1.0)
        else
            y_sigma_val = 1.0
        end
    end

    # Multi-fidelity noise scales
    z_sigma ~ filldist(Exponential(0.5), M.N_z)
    w_sigma ~ filldist(Exponential(0.5), M.N_w)

    # --- 2. Nested Latent Covariate Structure (RFF based) ---
    z_ls ~ Gamma(2, 2)
    # High-Fidelity Z (Spatial)
    z_beta ~ filldist(Normal(0, 1), M.M_rff, M.N_z)
    in_dim_z = size(M.z_coords_s, 2)
    z_proj = (M.z_coords_s * (M.W_fixed[1:in_dim_z, :] ./ z_ls)) .+ M.b_fixed'
    z_latent = M.rff_scale .* (cos.(z_proj) * z_beta)

    # Medium-Fidelity U (Spatiotemporal, depends on Z)
    w_ls ~ Gamma(2, 2)
    w_beta ~ filldist(Normal(0, 1), M.M_rff, M.N_w)
    w_coords_augmented = hcat(M.w_coords_st, z_latent[1:size(M.w_coords_st, 1), :])
    in_dim_w = size(w_coords_augmented, 2)
    w_proj = (w_coords_augmented * (M.W_fixed[1:in_dim_w, :] ./ w_ls)) .+ M.b_fixed'
    w_latent = M.rff_scale .* (cos.(w_proj) * w_beta)

    # --- 3. Spatial, Temporal, and Seasonality Manifolds ---
    s_sigma ~ Exponential(1.0)
    s_rho ~ Beta(1, 1)
    s_icar ~ MvNormalCanon(zeros(M.N_areas), M.s_Q + M.noise*I)
    s_iid ~ MvNormal(zeros(M.N_areas), I)

    # Soft sum-to-zero constraint for spatial identifiability
    sum_icar = sum(s_icar)
    sum_icar ~ Normal(0, 0.001 * M.N_areas)

    # Map to observations via index
    s_eta = (s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid))[M.s_idx]

    # --- Temporal Manifold ---
    local t_eta_full
    local t_Q = I(1) # Default for when N_time is not available or model_time is "none"


    t_sigma ~ Exponential(1.0)
    if haskey(M, :N_time) && M.N_time > 1
        if M.model_time == "ar1"
            t_rho ~ Beta(2, 2)
        # Recompose AR1 precision using precomputed structural template
        t_Q_base = Symmetric((1.0 + t_rho^2) .* I(M.N_time) .+ (t_rho) .* M.t_Q)
        t_Q = Symmetric((1.0 / (1.0 - t_rho^2 + M.noise)) .* t_Q_base )

        t_raw ~ MvNormalCanon(zeros(M.N_time), t_Q)
        t_eta_full = (t_raw .* t_sigma)[M.t_idx]

    elseif M.model_time == "rw2"
        # Scale structural RW2 template by inverse variance
        t_Q = Symmetric((1.0 / (t_sigma^2 + M.noise)) .* M.t_Q  )

        t_raw ~ MvNormalCanon(zeros(M.N_time), t_Q)
        t_eta_full = t_raw[M.t_idx] # sigma is already incorporated in the precision scale

    elseif M.model_time == "gp"
        t_ls ~ InverseGamma(3, 3)

        # Covariance-based sampling for GP with lengthscale
        K_t = (t_sigma^2) .* kernelmatrix(SqExponentialKernel() ∘ ScaleTransform(inv(t_ls)), 1.0:Float64(M.N_time)) + M.noise * I
        
        t_gp ~ MvNormal(zeros(M.N_time), Symmetric(K_t))
        t_eta_full = t_gp[M.t_idx]

        # Derive precision for interaction logic fallback
        t_Q = inv(Symmetric(K_t))

    elseif M.model_time == "harmonic"
        t_alpha ~ Normal(0, 1);
        t_beta ~ Normal(0, 1)
        t_eta_full = (t_alpha .* sin.(M.t_angle) .+ t_beta .* cos.(M.t_angle)) .* t_sigma
        t_Q = (1.0 / (t_sigma^2 + M.noise)) .* M.t_Q
    elseif M.model_time == "iid"
        t_Q = Symmetric((1.0 / (t_sigma^2 + M.noise)) .* I(M.N_time))
        t_iid ~ MvNormal(zeros(M.N_time), I)
        t_eta_full = (t_iid .* t_sigma)[M.t_idx]

    else
        t_Q = Symmetric(I(M.N_time))
        t_eta_full = zeros(T, M.N_obs) # Mapped to observations
    end
    else # N_time is not available or 1
        t_Q = I(1)
        t_eta_full = zeros(T, M.N_obs) # Mapped to observations
    end

    # --- Seasonality Manifold ---
    local u_eta_full
    local u_sigma
    if M.model_season != "none"
        u_sigma ~ Exponential(1.0)
    else
        u_sigma = 0.0
    end

    local u_rho
    if M.model_season == "ar1"
        u_rho ~ Beta(1, 1)
    else
        u_rho = 0.0
    end

    local u_ls
    if M.model_season == "gp"
        u_ls ~ InverseGamma(3, 3)
    else
        u_ls = 1.0
    end

    local u_eta_full
    if M.model_season == "ar1" && haskey(M, :N_season) && M.N_season > 1
        u_Q = build_bstm_ar1_precision(M.u_Q, u_rho; noise=M.noise)
        u_raw ~ MvNormalCanon(zeros(M.N_season), u_Q)
        u_eta_full = (u_raw .* u_sigma)[M.u_idx]
    elseif M.model_season == "rw2" && haskey(M, :N_season) && M.N_season > 1
        u_Q = build_bstm_rw2_precision(M.u_Q, u_sigma; noise=M.noise)
        u_raw ~ MvNormalCanon(zeros(M.N_season), u_Q)
        u_eta_full = (u_raw .* u_sigma)[M.u_idx]
    elseif M.model_season == "gp" && haskey(M, :N_season) && M.N_season > 1
        u_ls ~ InverseGamma(3, 3)
        K_s = kernelmatrix(SqExponentialKernel() ∘ ScaleTransform(1.0/u_ls), 1.0:Float64(M.N_season)) + M.noise*I
        u_gp ~ MvNormal(zeros(M.N_season), K_s)
        u_eta_full = (u_gp .* u_sigma)[M.u_idx]
    elseif M.model_season == "iid" && haskey(M, :N_season) && M.N_season > 1
        s_iid ~ MvNormal(zeros(M.N_season), I)
        u_eta_full = (s_iid .* u_sigma)[M.u_idx]
    else
        u_eta_full = zeros(M.N_obs)
    end
    
    # --- 4. Space-Time Interaction (Knorr-Held Types 0-4) ---
    local st_sigma
    if M.model_st > 0
        st_sigma ~ Exponential(0.5)
    else
        st_sigma = 0.0
    end
    # --- 4. Space-Time Interaction (Knorr-Held Types 0-4) ---
    local st_eta = zeros(T, M.N_areas, M.N_time)

    if M.model_st == 0
        st_eta = zeros(T, M.N_areas, M.N_time)

    elseif M.model_st == 1 # Type I: IID Interaction
        st_raw ~ MvNormal(zeros(M.N_areas * M.N_time), I)
        st_eta = reshape(st_raw .* st_sigma, M.N_areas, M.N_time)

    elseif M.model_st == 2 # Type II: Temporal Structure (I ⊗ Q_t)
        # Each area has an independent structured temporal trend
        st_Q2 = Symmetric(kron(I(M.N_areas), t_Q) )
        st_raw ~ MvNormalCanon(zeros(M.N_areas * M.N_time), st_Q2)
        st_eta = reshape(st_raw .* st_sigma, M.N_areas, M.N_time)

    elseif M.model_st == 3 # Type III: Spatial Structure (Q_s ⊗ I)
        # Each time point has an independent structured spatial field
        st_Q3 = Symmetric(kron(M.s_Q, I(M.N_time)) )
        st_raw ~ MvNormalCanon(zeros(M.N_areas * M.N_time), st_Q3)
        st_eta = reshape(st_raw .* st_sigma, M.N_areas, M.N_time)

    elseif M.model_st == 4 # Type IV: Inseparable (Q_s ⊗ Q_t)
        # Fully inseparable spatiotemporal structure
        st_Q4 = Symmetric(kron(M.s_Q, t_Q) )
        st_raw ~ MvNormalCanon(zeros(M.N_areas * M.N_time), st_Q4)
        st_eta = reshape(st_raw .* st_sigma, M.N_areas, M.N_time)
    end

    # --- 5. Linear Predictor Construction ---
    # Fixed effects
    local d_beta
    if M.N_fixed > 0
        d_beta ~ MvNormal(zeros(M.N_fixed), I)
    else
        d_beta = zeros(M.N_fixed)
    end

    z_beta_eta ~ filldist(Normal(0, 1), M.N_z) # Coefficients for each Z latent process
    w_beta_eta ~ MvNormal(zeros(M.N_w), I)

    eta = M.log_offset .+ s_eta .+ t_eta_full .+ u_eta_full .+ (z_latent * z_beta_eta) .+ (w_latent * w_beta_eta) # Summing contributions
    
    if M.N_fixed > 0; eta .+= M.fixed * d_beta; end

    if M.model_st > 0
        for i in 1:M.N_obs; eta[i] += st_eta[M.s_idx[i], M.t_idx[i]]; end
    end

    # --- 6. Joint Multi-fidelity Likelihood ---
    for k in 1:M.N_z # Loop over each high-fidelity latent process
        Turing.@addlogprob! logpdf(MvNormal(z_latent[:, k], z_sigma[k]^2 * I), M.z_obs[:, k]) # Each z_obs column is linked to a z_latent column
    end
    for k in 1:M.N_w # Loop over each medium-fidelity latent process
        Turing.@addlogprob! logpdf(MvNormal(w_latent[:, k], w_sigma[k]^2 * I), M.w_obs[:, k]) # Each w_obs column is linked to a w_latent column
    end

    # --- Likelihood (handling missing observations for primary observations) ---
    # Filter out missing observations and corresponding linear predictors/weights/trials
    # This ensures logpdf only operates on valid data points.
    good_indices_y = findall(!ismissing, M.y_obs)
    y_obs_filtered_y = M.y_obs[good_indices_y]
    eta_filtered_y = eta[good_indices_y]
    weights_filtered_y = M.weights[good_indices_y]
    trials_filtered_y = M.trials[good_indices_y]
    Turing.@addlogprob! logpdf(bstm_Likelihood(M.model_family, M.use_zi, weights_filtered_y, phi_zi, r_nb, y_sigma_val, trials_filtered_y, y_obs_filtered_y), eta_filtered_y) # Primary observations
end


####





@model function model_C05_gaussian_non_separable_rff(M, ::Type{T}=Float64; m_joint=25 ) where {T}
    # Model v11: Non-Separable Spatiotemporal RFF model.
    # Instead of separate spatial and temporal components, this model projects
    # the joint [X, Y, Time] vector into a shared feature space.
 
    
    # --- 1. Priors ---
    y_sigma ~ Exponential(1.0)
    sigma_joint ~ Exponential(1.0)
    # Lengthscales for X, Y, and Time dimensions within the joint kernel
    l_joint ~ filldist(Gamma(2, 1), 3) 
    w_joint ~ MvNormal(zeros(m_joint), I)
    c_sigma ~ filldist(Exponential(1.0), 4)

    # --- 2. Feature Matrix Construction ---
    # X_joint: [normalized_x, normalized_y, normalized_time]
    # We use M.pts and normalize them for numerical stability
    xs = [p[1] for p in M.pts]
    ys = [p[2] for p in M.pts]
    ts = Float64.(M.t_idx)
    
    # Normalize inputs to [0, 1] range
    X_joint = hcat(
        (xs .- minimum(xs)) ./ (maximum(xs) - minimum(xs) + M.noise),
        (ys .- minimum(ys)) ./ (maximum(ys) - minimum(ys) + M.noise),
        (ts .- minimum(ts)) ./ (maximum(ts) - minimum(ts) + M.noise)
    )

    # --- 3. Joint RFF Projection ---
    # This creates the non-separable interaction
    Random.seed!(42)
    # Sample frequencies scaled by dimension-specific lengthscales
    Om = randn(m_joint, 3) .* (1.0 ./ l_joint')
    Ph = rand(m_joint) .* convert(T, 2pi)
    
    Z_joint = convert(T, sqrt(2/m_joint)) .* cos.(X_joint * Om' .+ Ph')
    eta_joint = Z_joint * (w_joint .* sigma_joint)

    # --- 4. Categorical Smoothing (RW2) ---
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 5. Likelihood ---
    for i in 1:M.N_obs
        mu = M.log_offset[i] + eta_joint[i]
        for k in 1:M.N_cov; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), M.y_obs[i])
    end
end


@model function model_C06_gaussian_spde_rff(M, ::Type{T}=Float64 ) where {T}
    # Model v12: SPDE-style continuous spatial field using spectral RFF approximation.

    
    # --- 1. SPDE / Matern Priors ---
    y_sigma ~ Exponential(1.0)
    s_sigma ~ Exponential(1.0)
    kappa_sp ~ Gamma(2, 1)  # Range parameter (1/lengthscale)
    w_sp ~ MvNormal(zeros(M.m_spatial), I)

    # --- 2. Temporal (AR1) & Smoothing Priors ---
    t_sigma ~ Exponential(1.0)
    t_rho ~ Beta(2, 2)
    c_sigma ~ filldist(Exponential(1.0), 4)

    # --- 3. Continuous Spatial Basis (Spectral SPDE) ---
    # Normalize points for spectral projection
    xs = [p[1] for p in M.pts]
    ys = [p[2] for p in M.pts]
    coords = hcat(
        (xs .- mean(xs)) ./ std(xs),
        (ys .- mean(ys)) ./ std(ys)
    )

    Random.seed!(42)
    # Frequencies sampled for a Matern kernel approximation
    # Note: For Matern nu=1.5, we sample from a Student-t distribution spectral density
    Om = randn(M.m_spatial, 2) .* kappa_sp
    Ph = rand(M.m_spatial) .* convert(T, 2pi)
    
    Z_sp = convert(T, sqrt(2/M.m_spatial)) .* cos.(coords * Om' .+ Ph')
    s_eta = Z_sp * (w_sp .* s_sigma)

    # --- 4. Temporal Effect (AR1) ---
    t_Q = (1.0 / (1.0 - t_rho^2)) .* (M.t_Q + (t_rho^2) * I)
    t_raw ~ MvNormal(zeros(M.N_time), I)
    Turing.@addlogprob! -0.5 * dot(t_raw, t_Q * t_raw)
    t_eta = t_raw .* t_sigma

    # --- 5. Categorical & Likelihood ---
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    for i in 1:M.N_obs
        mu = M.log_offset[i] + s_eta[i] + t_eta[M.t_idx[i]]
        for k in 1:M.N_cov; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), M.y_obs[i])
    end
end




@model function model_C07_gaussian_nonstationary_warping(M, ::Type{T}=Float64) where {T}

    # --- 1. Priors ---
    y_sigma ~ Exponential(1.0)
    s_sigma ~ Exponential(1.0)
    l_warp ~ Gamma(2, 1)    # Smoothness of the warping manifold
    l_spatial ~ Gamma(2, 1) # Smoothness of the stationary field in warped space
    
    w_warp ~ MvNormal(zeros(M.m_warp), I)
    w_sp ~ MvNormal(zeros(M.m_spatial), I)
    
    t_sigma ~ Exponential(1.0)
    t_rho ~ Beta(2, 2)
    c_sigma ~ filldist(Exponential(1.0), 4)

    # --- 2. Input Preprocessing ---
    xs = [p[1] for p in M.pts]
    ys = [p[2] for p in M.pts]
    coords = hcat((xs .- mean(xs)) ./ std(xs), (ys .- mean(ys)) ./ std(ys))

    # --- 3. Warping Layer (Non-Stationarity) ---
    # This layer 'warps' the 2D coordinates into a latent space
    Random.seed!(44)
    Om_w = randn(M.m_warp, 2) ./ l_warp
    Ph_w = rand(M.m_warp) .* convert(T, 2pi)
    
    # Warped coordinates: g(s)
    warped_coords = (convert(T, sqrt(2/M.m_warp)) .* cos.(coords * Om_w' .+ Ph_w')) * w_warp

    # --- 4. Spatial Field on Warped Manifold ---
    # We apply a stationary kernel to the warped output
    Random.seed!(45)
    Om_s = randn(M.m_spatial, 1) ./ l_spatial
    Ph_s = rand(M.m_spatial) .* convert(T, 2pi)
    
    Z_sp = convert(T, sqrt(2/M.m_spatial)) .* cos.(reshape(warped_coords, :, 1) * Om_s' .+ Ph_s')
    s_eta = Z_sp * (w_sp .* s_sigma)

    # --- 5. Temporal & Categorical Components ---
    t_Q = (1.0 / (1.0 - t_rho^2)) .* (M.t_Q + (t_rho^2) * I)
    t_raw ~ MvNormal(zeros(M.N_time), I)
    Turing.@addlogprob! -0.5 * dot(t_raw, t_Q * t_raw)
    t_eta = t_raw .* t_sigma

    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 6. Likelihood ---
    for i in 1:M.N_obs
        mu = M.log_offset[i] + s_eta[i] + t_eta[M.t_idx[i]]
        for k in 1:M.N_cov; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), M.y_obs[i])
    end
end




@model function model_C08_gaussian_refined_mosaic(M, ::Type{T}=Float64 ) where {T}
      
    # --- 1. Global & Hierarchical Priors ---
    c_sigma ~ filldist(Exponential(1.0), 4)
    mu_global ~ Normal(0, 1)
    sigma_mu_local ~ Exponential(1.0)

    # Local Parameters per Mosaic
    mu_local ~ filldist(Normal(mu_global, sigma_mu_local), M.n_mosaics)
    l_local ~ filldist(Gamma(2, 1), M.n_mosaics)
    sigma_local ~ filldist(Exponential(1.0), M.n_mosaics)
    y_sigma_local ~ filldist(Exponential(1.0), M.n_mosaics) # Localized noise scale

    # M.Weights for each mosaic's RFF field
    w_local = [Vector{T}(undef, M.m_rff) for _ in 1:M.n_mosaics]
    for m in 1:M.n_mosaics; w_local[m] ~ MvNormal(zeros(M.m_rff), I); end

    # Categorical Covariates (Shared)
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 2. Spatial Indexing & Soft Boundary M.Weights ---
    coords = hcat([p[1] for p in M.pts], [p[2] for p in M.pts])
    R = kmeans(coords', M.n_mosaics)
    centroids = R.centers # 2 x M.n_mosaics

    # Pre-sample RFF frequencies
    Random.seed!(42)
    Om_m = [randn(M.m_rff, 2) for _ in 1:M.n_mosaics]
    Ph_m = [rand(M.m_rff) for _ in 1:M.n_mosaics]

    # --- 3. Likelihood with Soft Integration ---
    for i in 1:M.N_obs
        pt = [coords[i,1], coords[i,2]]
        
        # Calculate Softmax M.weights based on distance to centroids for smooth stitching
        dists = [sum((pt .- centroids[:, m]).^2) for m in 1:M.n_mosaics]
        
        max_d = maximum(-dists)
        weights_stitching = exp.(-dists .- max_d) ./ (sum(exp.(-dists .- max_d)) + 1e-9)
     
        eta_spatial_combined = zero(T)
        y_sigma_combined = zero(T)
        
        for m in 1:M.n_mosaics
            # Local RFF Field Calculation
            z_proj = sqrt(2/M.m_rff) .* cos.( (Om_m[m] * pt ./ l_local[m]) .+ (Ph_m[m] .* 2pi) )
            local_field = mu_local[m] + dot(z_proj, w_local[m] .* sigma_local[m])
            
            # Blend local field and local noise
            eta_spatial_combined += weights_stitching[m] * local_field
            y_sigma_combined += weights_stitching[m] * y_sigma_local[m]
        end

        mu = M.log_offset[i] + eta_spatial_combined
        for k in 1:M.N_cov; 
            mu += c_beta[k][M.cov_indices[i, k]]; 
        end
        
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma_combined + 1e-4), M.y_obs[i])
    end
end


@model function model_C09_gaussian_integrated_mosaic(M, ::Type{T}=Float64 ) where {T}

    # --- 1. Global Hierarchical Priors ---
    c_sigma ~ filldist(Exponential(1.0), 4)
    mu_global ~ Normal(0, 1)
    sigma_mu_local ~ Exponential(0.5)

    # Shared Categorical Effects
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 2. Local Mosaic Hyperparameters ---
    mu_local ~ filldist(Normal(mu_global, sigma_mu_local), M.n_mosaics)

    # Refactored: Use arraydist for joint lengthscales instead of a loop
    l_joint ~ arraydist([filldist(Gamma(2, 1), 3) for _ in 1:M.n_mosaics])

    sigma_local ~ filldist(Exponential(1.0), M.n_mosaics)
    y_sigma_local ~ filldist(Exponential(1.0), M.n_mosaics)

    # Local M.Weights for Non-Separable RFF
    w_local = [Vector{T}(undef, M.m_rff) for _ in 1:M.n_mosaics]
    for m in 1:M.n_mosaics; w_local[m] ~ MvNormal(zeros(M.m_rff), I); end

    # --- 3. Geometric Indexing ---
    xs = [p[1] for p in M.pts]
    ys = [p[2] for p in M.pts]
    ts = Float64.(M.t_idx)

    # Normalize [X, Y, T] to [0, 1] range for RFF stability
    X_joint = hcat(
        (xs .- minimum(xs)) ./ (maximum(xs) - minimum(xs) + M.noise),
        (ys .- minimum(ys)) ./ (maximum(ys) - minimum(ys) + M.noise),
        (ts .- minimum(ts)) ./ (maximum(ts) - minimum(ts) + M.noise)
    )

    # Static centroids for stitching (calculated once)
    coords_2d = X_joint[:, 1:2]
    R = kmeans(coords_2d', M.n_mosaics)
    centroids = R.centers

    # Fixed RFF Frequencies
    Random.seed!(42)
    Om_base = [randn(M.m_rff, 3) for _ in 1:M.n_mosaics]
    Ph_base = [rand(M.m_rff) for _ in 1:M.n_mosaics]

    # --- 4. Predictive Synthesis ---
    for i in 1:M.N_obs
        pt_3d = X_joint[i, :]
        pt_2d = pt_3d[1:2]

        # Soft Boundary M.Weights (Softmax of distance to centroids)
        dists = [sum((pt_2d .- centroids[:, m]).^2) for m in 1:M.n_mosaics]
        weights_st = exp.(-dists) ./ sum(exp.(-dists))

        eta_spatial_time = zero(T)
        y_sigma_total = zero(T)

        for m in 1:M.n_mosaics
            # Scale base frequencies by local lengthscales [Lx, Ly, Lt]
            # l_joint is now a matrix where each column corresponds to a mosaic
            Om = Om_base[m] .* (1.0 ./ (l_joint[:, m] .+ M.noise)')

            # Local Non-Separable Field
            z_proj = sqrt(2/M.m_rff) * cos.( (Om * pt_3d) .+ (Ph_base[m] .* 2pi) )
            local_field = mu_local[m] + dot(z_proj, w_local[m] .* sigma_local[m])

            eta_spatial_time += weights_st[m] * local_field
            y_sigma_total += weights_st[m] * y_sigma_local[m]
        end

        mu = M.log_offset[i] + eta_spatial_time
        for k in 1:M.N_cov; mu += c_beta[k][M.cov_indices[i, k]]; end

        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma_total + M.noise), M.y_obs[i])
    end
end



@model function model_C10_gaussian_fitcxed_fitc_grmf(M, ::Type{T}=Float64) where {T}
    # FITC Sparse GP using fixed inducing point priors for performance optimization.
    # Architecture: [1] BYM2 Spatial, [2] AR1 Temporal, [3] Harmonic Seasonal, [4] GMRF Interaction, [5] RW2 Smoothing.
 
    # Priors
    y_sigma ~ Exponential(1.0)
    sigma_f ~ Exponential(1.0)
    ls_st ~ filldist(Gamma(2, 2), 3)
    s_sigma ~ Exponential(1.0)
    s_rho ~ Beta(1, 1)
    t_sigma ~ Exponential(1.0)
    t_rho ~ Beta(2, 2)
    st_sigma ~ Exponential(0.5)
    c_sigma ~ filldist(Exponential(1.0), 4)
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)

    # Component 1: BYM2
    s_icar ~ MvNormal(zeros(M.N_areas), I)
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.N_areas), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # Component 2: AR1
    t_Q = (1.0 / (1.0 - t_rho^2)) .* (M.t_Q + (t_rho^2) * I)
    t_raw ~ MvNormal(zeros(M.N_time), I)
    Turing.@addlogprob! -0.5 * dot(t_raw, t_Q * t_raw)
    t_eta = t_raw .* t_sigma

    # Component 3: Seasonal
    t_vec = Float64.(M.t_idx)
    seasonal = beta_cos .* cos.(2pi .* t_vec ./ M.period) .+ beta_sin .* sin.(2pi .* t_vec ./ M.period)

    # Component 4: GMRF Interaction
    st_raw ~ MvNormal(zeros(M.N_areas * M.N_time), I)
    st_eta = reshape(st_raw .* st_sigma, M.N_areas, M.N_time)

    # Component 5: Fixed Inducing FITC (Optimized GP Projection)
    t_norm = M.t_idx ./ M.N_time
    coords_st = hcat([p[1] for p in M.pts], [p[2] for p in M.pts], t_norm)
    k_st = (sigma_f^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(ls_st)))
    K_XZ = kernelmatrix(k_st, RowVecs(coords_st), RowVecs(M.Z_inducing))
    u_inducing ~ MvNormal(zeros(size(M.Z_inducing, 1)), I) # Assumes unit variance at inducing points
    f_gp = K_XZ * u_inducing # Linear projection

    # RW2 Categorical
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # Likelihood
    for i in 1:M.N_obs
        idx_a = M.s_idx[i]
        idx_t = M.t_idx[i]
        mu = M.log_offset[i] + f_gp[i] + s_eta[idx_a] + t_eta[idx_t] + seasonal[i] + st_eta[idx_a, idx_t]
        for k in 1:M.N_cov; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), M.y_obs[i])
    end
end




@model function model_C11_svgp(M, ::Type{T}=Float64) where {T}
    # SVGP-like (learned inducing points) with GP Trend and Non-linear Nested Covariates.
  
    # --- 1. Priors ---
    sigma_w ~ filldist(Exponential(0.5), 3)
    sigma_f ~ Exponential(1.0); ls_st ~ filldist(Gamma(2, 2), M.D_st)
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    c_sigma ~ filldist(Exponential(1.0), 4)
    sigma_log_var ~ Exponential(1.0)

    # --- 2. GP Temporal Trend ---
    ls_trend ~ Gamma(2, 2); sigma_trend ~ Exponential(0.5)
    k_trend = (sigma_trend^2) * (SqExponentialKernel() ∘ ScaleTransform(inv(ls_trend)))
    t_unique = collect(1:M.N_time) ./ M.N_time
    alpha_gp ~ MvNormal(zeros(M.N_time), kernelmatrix(k_trend, t_unique) + M.noise*I)

    # --- 3. Non-linear Nested Structure (RFF) ---
    t_norm = M.t_idx ./ M.N_time
    coords_tz = hcat(t_norm, M.z_obs)
    W_u1 ~ filldist(Normal(0, 1), 2, M.M_rff_u); b_u1 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    sigma_f_u1 ~ Exponential(1.0); beta_rff_u1 ~ filldist(Normal(0, sigma_f_u1), M.M_rff_u)
    u1_true = (sqrt(2/M.M_rff_u) .* cos.(coords_tz * W_u1 .+ b_u1')) * beta_rff_u1

    coords_tz_u1 = hcat(t_norm, M.z_obs, u1_true)
    W_u2 ~ filldist(Normal(0, 1), 3, M.M_rff_u); b_u2 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    sigma_f_u2 ~ Exponential(1.0); beta_rff_u2 ~ filldist(Normal(0, sigma_f_u2), M.M_rff_u)
    u2_true = (sqrt(2/M.M_rff_u) .* cos.(coords_tz_u1 * W_u2 .+ b_u2')) * beta_rff_u2

    W_u3 ~ filldist(Normal(0, 1), 3, M.M_rff_u); b_u3 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    sigma_f_u3 ~ Exponential(1.0); beta_rff_u3 ~ filldist(Normal(0, sigma_f_u3), M.M_rff_u)
    u3_true = (sqrt(2/M.M_rff_u) .* cos.(coords_tz_u1 * W_u3 .+ b_u3')) * beta_rff_u3
    u_true_mat = hcat(u1_true, u2_true, u3_true)

    # --- 4. Spatial Effect (BYM2) ---
    s_icar ~ MvNormal(zeros(M.N_areas), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.N_areas), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # --- 5. SVGP Mean (Learned Inducing Points) ---
    coords_st = hcat([p[1] for p in M.pts], [p[2] for p in M.pts], t_norm)
    Z_inducing = Matrix{T}(undef, M.M_inducing_val, M.D_st)
    for j in 1:M.D_st
        # Learn inducing locations via prior-constrained exploration
        Z_inducing[:, j] ~ filldist(Normal(mean(coords_st[:,j]), 2*std(coords_st[:,j])), M.M_inducing_val)
    end

    k_st = (sigma_f^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(ls_st)))
    K_ZZ = kernelmatrix(k_st, RowVecs(Z_inducing)) + M.noise * I
    K_XZ = kernelmatrix(k_st, RowVecs(coords_st), RowVecs(Z_inducing))
    K_XX_diag = diag(kernelmatrix(k_st, RowVecs(coords_st)))

    u_inducing ~ MvNormal(zeros(M.M_inducing_val), K_ZZ)
    f_mean = K_XZ * (K_ZZ \ u_inducing)
    cov_f_diag = K_XX_diag - diag(K_XZ * (K_ZZ \ K_XZ'))
    f_gp ~ MvNormal(f_mean, Diagonal(max.(M.noise, cov_f_diag)))

    # --- 6. Volatility & Categorical Smoothing ---
    W_vol ~ filldist(Normal(0, 1), M.D_st, M.M_rff_sigma); b_vol ~ filldist(Uniform(0, 2pi), M.M_rff_sigma)
    beta_rff_vol ~ filldist(Normal(0, sigma_log_var), M.M_rff_sigma)
    log_y_sigma = (sqrt(2/M.M_rff_sigma) .* cos.(coords_st * W_vol .+ b_vol')) * beta_rff_vol
    y_sigma = exp.(log_y_sigma ./ 2.0)

    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 7. Likelihoods ---
    z_beta ~ Normal(0, 2); beta_w_main ~ MvNormal(zeros(3), 2.0 * I)
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2pi .* t_norm * (M.N_time/M.period)) .+ beta_sin .* sin.(2pi .* t_norm * (M.N_time/M.period))

    for i in 1:M.N_obs
        a, t = M.s_idx[i], M.t_idx[i]
        mu = M.log_offset[i] + alpha_gp[t] + seasonal[i] + z_beta * M.z_obs[i] + dot(beta_w_main, u_true_mat[i, :]) + f_gp[i] + s_eta[a]
        for k in 1:M.N_cov; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma[i]), M.y_obs[i])
        for k in 1:3; Turing.@addlogprob! logpdf(Normal(u_true_mat[i, k], sigma_w[k]), M.w_obs[i, k]); end
    end
end


@model function model_C12_svgp_full(M, ::Type{T}=Float64) where {T}
    # Full SVGP logic (learned inducing locations and latent distribution) with GP Trend.
 
    # --- 1. Priors ---
    sigma_w ~ filldist(Exponential(0.5), M.N_cov)
    sigma_f ~ Exponential(1.0); ls_st ~ filldist(Gamma(2, 2), M.D_st)
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    c_sigma ~ filldist(Exponential(1.0), 4)
    sigma_log_var ~ Exponential(1.0)

    # --- 2. GP Temporal Trend ---
    ls_trend ~ Gamma(2, 2); sigma_trend ~ Exponential(0.5)
    k_trend = (sigma_trend^2) * (SqExponentialKernel() ∘ ScaleTransform(inv(ls_trend)))
    t_unique = collect(1:M.N_time) ./ M.N_time
    alpha_gp ~ MvNormal(zeros(M.N_time), kernelmatrix(k_trend, t_unique) + M.noise*I)

    # --- 3. Non-linear Nested Structure (RFF) ---
    t_norm = M.t_idx ./ M.N_time
    coords_tz = hcat(t_norm, M.z_obs)
    W_u1 ~ filldist(Normal(0, 1), 2, M.M_rff_u); b_u1 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    sigma_f_u1 ~ Exponential(1.0); beta_rff_u1 ~ filldist(Normal(0, sigma_f_u1), M.M_rff_u)
    u1_true = (sqrt(2/M.M_rff_u) .* cos.(coords_tz * W_u1 .+ b_u1')) * beta_rff_u1

    coords_tz_u1 = hcat(t_norm, M.z_obs, u1_true)
    W_u2 ~ filldist(Normal(0, 1), 3, M.M_rff_u); b_u2 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    sigma_f_u2 ~ Exponential(1.0); beta_rff_u2 ~ filldist(Normal(0, sigma_f_u2), M.M_rff_u)
    u2_true = (sqrt(2/M.M_rff_u) .* cos.(coords_tz_u1 * W_u2 .+ b_u2')) * beta_rff_u2

    W_u3 ~ filldist(Normal(0, 1), 3, M.M_rff_u); b_u3 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    sigma_f_u3 ~ Exponential(1.0); beta_rff_u3 ~ filldist(Normal(0, sigma_f_u3), M.M_rff_u)
    u3_true = (sqrt(2/M.M_rff_u) .* cos.(coords_tz_u1 * W_u3 .+ b_u3')) * beta_rff_u3
    u_true_mat = hcat(u1_true, u2_true, u3_true)

    # --- 4. Spatial Effect (BYM2) ---
    s_icar ~ MvNormal(zeros(M.N_areas), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.N_areas), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # --- 5. Full SVGP Mean (Learned Locations and Latent Params) ---
    coords_st = hcat([p[1] for p in M.pts], [p[2] for p in M.pts], t_norm)
    Z_inducing = Matrix{T}(undef, M.M_inducing_val, M.D_st)
    for j in 1:M.D_st
        Z_inducing[:, j] ~ filldist(Normal(mean(coords_st[:,j]), 2*std(coords_st[:,j])), M.M_inducing_val)
    end

    # Variational parameters for the inducing points
    m_u ~ MvNormal(zeros(M.M_inducing_val), 10.0 * I)
    s_u_diag ~ filldist(Exponential(1.0), M.M_inducing_val)
    w_latent ~ MvNormal(m_u, Diagonal(s_u_diag.^2) + M.noise*I)

    k_st = (sigma_f^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(ls_st)))
    K_ZZ = kernelmatrix(k_st, RowVecs(Z_inducing)) + M.noise * I
    K_XZ = kernelmatrix(k_st, RowVecs(coords_st), RowVecs(Z_inducing))
    K_XX_diag = diag(kernelmatrix(k_st, RowVecs(coords_st)))

    f_mean = K_XZ * (K_ZZ \ w_latent)
    cov_f_diag = K_XX_diag - diag(K_XZ * (K_ZZ \ K_XZ'))
    f_gp ~ MvNormal(f_mean, Diagonal(max.(M.noise, cov_f_diag)))

    # --- 6. Volatility & Categorical Smoothing ---
    W_vol ~ filldist(Normal(0, 1), M.D_st, M.M_rff_sigma); b_vol ~ filldist(Uniform(0, 2pi), M.M_rff_sigma)
    beta_rff_vol ~ filldist(Normal(0, sigma_log_var), M.M_rff_sigma)
    log_y_sigma = (sqrt(2/M.M_rff_sigma) .* cos.(coords_st * W_vol .+ b_vol')) * beta_rff_vol
    y_sigma = exp.(log_y_sigma ./ 2.0)

    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 7. Likelihoods ---
    z_beta ~ Normal(0, 2); beta_w_main ~ MvNormal(zeros(3), 2.0 * I)
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2pi .* t_norm * (M.N_time/M.period)) .+ beta_sin .* sin.(2pi .* t_norm * (M.N_time/M.period))

    for i in 1:M.N_obs
        a, t = M.s_idx[i], M.t_idx[i]
        mu = M.log_offset[i] + alpha_gp[t] + seasonal[i] + z_beta * M.z_obs[i] + dot(beta_w_main, u_true_mat[i, :]) + f_gp[i] + s_eta[a]
        for k in 1:M.N_cov; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma[i]), M.y_obs[i])
        for k in 1:3; Turing.@addlogprob! logpdf(Normal(u_true_mat[i, k], sigma_w[k]), M.w_obs[i, k]); end
    end
end


@model function model_C13_multifidelity_gp(M, ::Type{T}=Float64) where {T}
    # --- 1. Data Dimensions & Multi-fidelity Inputs ---
    
    z_obs = M.z_obs
    u_obs = M.w_obs
    coords_z = M.z_coords_s
    coords_u = M.w_coords_st
     
    # --- 2. Hierarchical Priors ---
    y_sigma ~ Exponential(1.0) # Standard noise
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1) # BYM2 params
    t_sigma ~ Exponential(1.0); t_rho ~ Beta(2, 2) # AR1 params
    st_sigma ~ Exponential(0.5); c_sigma ~ filldist(Exponential(1.0), 4) # Smoothing
    sigma_z ~ Exponential(0.5); sigma_w ~ filldist(Exponential(0.5), 3) # Fidelity noise
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1) # Seasonal M.weights

    # --- 3. Component 1: BYM2 Spatial Effect ---
    s_icar ~ MvNormal(zeros(M.N_areas), I)
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.N_areas), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # --- 4. Component 2: AR1 Temporal Effect ---
    t_Q = (1.0 / (1.0 - t_rho^2)) .* (M.t_Q + (t_rho^2) * I)
    t_raw ~ MvNormal(zeros(M.N_time), I)
    Turing.@addlogprob! -0.5 * dot(t_raw, t_Q * t_raw)
    t_eta = t_raw .* t_sigma

    # --- 5. Component 3: Harmonic Seasonality ---
    t_vec = Float64.(M.t_idx)
    seasonal = beta_cos .* cos.(2pi .* t_vec ./ M.period) .+ beta_sin .* sin.(2pi .* t_vec ./ M.period)

    # --- 6. Component 4: Space-Time Interaction (GMRF) ---
    st_raw ~ MvNormal(zeros(M.N_areas * M.N_time), I)
    st_eta = reshape(st_raw .* st_sigma, M.N_areas, M.N_time)

    # --- 7. Component 5: Multi-fidelity RFF Projections ---
    # Latent Z Fidelity
    W_z ~ filldist(Normal(0, 1), size(coords_z, 2), M.M_rff)
    b_z ~ filldist(Uniform(0, 2pi), M.M_rff)
    beta_z_rff ~ filldist(Normal(0, 1), M.M_rff)
    z_latent = rff_map(coords_z, W_z, b_z) * beta_z_rff
    Turing.@addlogprob! logpdf(MvNormal(z_latent, sigma_z^2 * I), z_obs)

    # Latent U Fidelity
    W_u ~ filldist(Normal(0, 1), size(coords_u, 2), M.M_rff)
    b_u ~ filldist(Uniform(0, 2pi), M.M_rff)
    beta_w_rff ~ filldist(Normal(0, 1), M.M_rff, 3)
    w_latent = rff_map(coords_u, W_u, b_u) * beta_w_rff
    for k in 1:3; Turing.@addlogprob! logpdf(MvNormal(w_latent[:, k], sigma_w[k]^2 * I), u_obs[:, k]); end

    # --- 8. Categorical Smoothing (RW2) ---
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 9. Final Likelihood ---
    for i in 1:M.N_obs
        idx_a = M.s_idx[i]
        idx_t = M.t_idx[i]
        mu = M.log_offset[i] + s_eta[idx_a] + t_eta[idx_t] + seasonal[i] + st_eta[idx_a, idx_t]
        for k in 1:M.N_cov; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), M.y_obs[i])
    end
end

@model function model_C14_minibatch_mfgp(M, ::Type{T}=Float64) where {T}
      
    # Priors
    y_sigma ~ Exponential(1.0); s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    t_sigma ~ Exponential(1.0); t_rho ~ Beta(2, 2); st_sigma ~ Exponential(0.5)
    c_sigma ~ filldist(Exponential(1.0), 4); beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)

    # BYM2
    s_icar ~ MvNormal(zeros(M.N_areas), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.N_areas), I); s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # AR1
    t_Q = (1.0 / (1.0 - t_rho^2)) .* (M.t_Q + (t_rho^2) * I)
    t_raw ~ MvNormal(zeros(M.N_time), I); Turing.@addlogprob! -0.5 * dot(t_raw, t_Q * t_raw)
    t_eta = t_raw .* t_sigma

    # Seasonal
    seasonal = beta_cos .* cos.(2pi .* Float64.(M.t_idx) ./ M.period) .+ beta_sin .* sin.(2pi .* Float64.(M.t_idx) ./ M.period)

    # GMRF Interaction
    st_raw ~ MvNormal(zeros(M.N_areas * M.N_time), I); st_eta = reshape(st_raw .* st_sigma, M.N_areas, M.N_time)

    # RW2
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # Likelihood
    for i in 1:M.N_obs
        idx_a = M.s_idx[i]; idx_t = M.t_idx[i]
        mu = M.log_offset[i] + s_eta[idx_a] + t_eta[idx_t] + seasonal[i] + st_eta[idx_a, idx_t]
        for k in 1:M.N_cov; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), M.y_obs[i])
    end
end


@model function model_C15_deep_gp(M, ::Type{T}=Float64) where {T}
    # Deep Spatiotemporal GP.
    # Integrates BYM2 spatial effects and RW2 smoothing into a 3-layer RFF hierarchy.
   
    D_s = size(M.s_obs, 2)
    D_t = size(M.t_obs, 2)

    # --- 1. Priors & Structural Components ---
    y_sigma ~ Exponential(1.0); s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    sigma_z ~ Exponential(0.5); sigma_w ~ filldist(Exponential(0.5), 3)
    c_sigma ~ filldist(Exponential(1.0), 4)

    # BYM2 Spatial Effect
    s_icar ~ MvNormal(zeros(M.N_areas), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.N_areas), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # RW2 Categorical Smoothing
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 2. Layer 1: Latent Spatial GP (Z) ---
    W_z ~ filldist(Normal(0, 1), D_s, M.M_rff)
    b_z ~ filldist(Uniform(0, 2pi), M.M_rff)
    z_beta ~ filldist(Normal(0, 1), M.M_rff)
    z_latent = rff_map(M.s_obs, W_z, b_z) * z_beta

    # --- 3. Layer 2: Latent Spatiotemporal GPs (U1, U2, U3) ---
    coords_l2 = hcat(M.s_obs, M.t_obs, z_latent)
    W_u ~ filldist(Normal(0, 1), size(coords_l2, 2), M.M_rff)
    b_u ~ filldist(Uniform(0, 2pi), M.M_rff)
    beta_w_mat ~ filldist(Normal(0, 1), M.M_rff, 3)
    Phi_u = rff_map(coords_l2, W_u, b_u)
    w_latent = Phi_u * beta_w_mat # Matrix M.N_obs x 3

    # --- 4. Layer 3: Final Output GP (Y) ---
    coords_l3 = hcat(M.s_obs, M.t_obs, w_latent)
    W_y ~ filldist(Normal(0, 1), size(coords_l3, 2), M.M_rff)
    b_y ~ filldist(Uniform(0, 2pi), M.M_rff)
    beta_y_gp ~ filldist(Normal(0, 1), M.M_rff)

    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2pi .* M.t_obs ./ M.period) .+ beta_sin .* sin.(2pi .* M.t_obs ./ M.period)
    f_y = (rff_map(coords_l3, W_y, b_y) * beta_y_gp) .+ vec(seasonal)

    # --- 5. Likelihood ---
    for i in 1:M.N_obs
        a = M.s_idx[i]
        mu = M.log_offset[i] + f_y[i] + s_eta[a]
        for k in 1:M.N_cov; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), M.y_obs[i])
    end

    # Multi-fidelity cross-resolution constraints (Optional prior links)
    M.z_obs ~ MvNormal(z_latent, sigma_z^2 * I)
    for k in 1:3; M.w_obs[:, k] ~ MvNormal(w_latent[:, k], sigma_w[k]^2 * I); end
end


@model function model_C16_nystrom(M, ::Type{T}=Float64) where {T}
    # Model A16 Optimized: Standardized Nyström GP with Stochastic Volatility.
    # Combines low-rank spatiotemporal approximations with heteroskedastic noise modeling.
 
    
    D_s, D_t = size(M.s_obs, 2), size(M.t_obs, 2)
    D_st = D_s + D_t

    # --- 1. Priors & Structural Components ---
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    sigma_w ~ filldist(Exponential(0.5), 3)
    c_sigma ~ filldist(Exponential(1.0), 4)

    # BYM2 Spatial Effect
    s_icar ~ MvNormal(zeros(M.N_areas), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.N_areas), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # RW2 Categorical Smoothing
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 2. Nested Latent Covariates (RFF Mapping) ---
    coords_tz = hcat(M.t_obs, M.z_obs)
    W_u1 ~ filldist(Normal(0, 1), size(coords_tz, 2), M.M_rff_u); b_u1 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    u1_true = rff_map(coords_tz, W_u1, b_u1) * filldist(Normal(0, 1), M.M_rff_u)

    coords_tz_u1 = hcat(M.t_obs, M.z_obs, u1_true)
    W_u2 ~ filldist(Normal(0, 1), size(coords_tz_u1, 2), M.M_rff_u); b_u2 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    u2_true = rff_map(coords_tz_u1, W_u2, b_u2) * filldist(Normal(0, 1), M.M_rff_u)

    # --- 3. Nyström GP (Low Rank Approximation) ---
    coords_st = hcat(M.s_obs, M.t_obs)
    ls_st ~ filldist(Gamma(2, 2), D_st); 
    sigma_f ~ Exponential(1.0)
    k_st = SqExponentialKernel() ∘ ARDTransform(inv.(ls_st + M.eps))

    Z_ind = Matrix{T}(undef, M.M_inducing_val, D_st)
    mu_st, std_st = mean(coords_st, dims=1), std(coords_st, dims=1)
    for j in 1:D_st; Z_ind[:, j] ~ filldist(Normal(mu_st[j], 2.0 * std_st[j]), M.M_inducing_val); end

    K_zz = Symmetric( kernelmatrix(k_st, RowVecs(Z_ind)) + M.eps*I)
    K_xz = kernelmatrix(k_st, RowVecs(coords_st), RowVecs(Z_ind))
    v_latent ~ filldist(Normal(0, 1), M.M_inducing_val)
    f_nystrom = sigma_f .* (K_xz * (cholesky(K_zz).U \ v_latent))

    # --- 4. Spatiotemporal Stochastic Volatility ---
    W_sigma ~ filldist(Normal(0, 1), D_st, M.M_rff_sigma); b_sigma ~ filldist(Uniform(0, 2pi), M.M_rff_sigma)
    beta_rff_sigma ~ filldist(Normal(0, 1), M.M_rff_sigma)
    y_sigma = exp.(rff_map(coords_st, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 5. Likelihood ---
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2pi .* M.t_obs ./ M.period) .+ beta_sin .* sin.(2pi .* M.t_obs ./ M.period)

    for i in 1:M.N_obs
        a = M.s_idx[i]
        mu = M.log_offset[i] + f_nystrom[i] + seasonal[i] + s_eta[a]
        for k in 1:M.N_cov; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma[i]), M.y_obs[i])
    end

    # Prior links for latent variables
    M.w_obs[:, 1] ~ MvNormal(u1_true, sigma_w[1]^2 * I)
    M.w_obs[:, 2] ~ MvNormal(u2_true, sigma_w[2]^2 * I)
end


@model function model_C17_spde(M, ::Type{T}=Float64 ) where {T}
    # SPDE-based Spatiotemporal GP.
    # Employs sparse precision approximations for spatial effects and RFF for volatility.
 
    D_s, D_t = size(M.s_obs, 2), size(M.t_obs, 2)
    D_st = D_s + D_t

    # --- 1. Priors & Structural Components ---
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    sigma_w ~ filldist(Exponential(0.5), 3)
    c_sigma ~ filldist(Exponential(1.0), 4)

    # BYM2 Spatial Effect
    s_icar ~ MvNormal(zeros(M.N_areas), I); 
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.N_areas), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # RW2 Categorical Smoothing
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 2. Latent Trends (GP Trend & Seasonal) ---
    ls_trend ~ Gamma(2, 2); sigma_trend ~ Exponential(0.5)

    k_trend = SqExponentialKernel() ∘ ScaleTransform(inv(ls_trend))
    unique_times = sort(unique(M.t_obs[:,1]))
    
    K_trend = Symmetric(sigma_trend^2 * kernelmatrix(k_trend, unique_times) + 1e-4 * I)
    alpha ~ MvNormal(zeros(length(unique_times)), K_trend)

    # alpha ~ GP(sigma_trend^2 * k_trend)(unique_times, M.noise)
    trend = alpha[indexin(M.t_obs[:,1], unique_times)]

    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2pi .* M.t_obs ./ M.period) .+ beta_sin .* sin.(2pi .* M.t_obs ./ M.period)

    # --- 3. Nested Latent Covariates (RFF) ---
    coords_tz = hcat(M.t_obs, M.z_obs)
    W_u1 ~ filldist(Normal(0, 1), size(coords_tz, 2), M.M_rff_u); b_u1 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    u1_true = rff_map(coords_tz, W_u1, b_u1) * filldist(Normal(0, 1), M.M_rff_u)

    # --- 4. Stochastic Volatility ---
    coords_st = hcat(M.s_obs, M.t_obs)
    W_sigma ~ filldist(Normal(0, 1), D_st, M.M_rff_sigma); b_sigma ~ filldist(Uniform(0, 2pi), M.M_rff_sigma)
    beta_rff_sigma ~ filldist(Normal(0, 1), M.M_rff_sigma)
    y_sigma = exp.(rff_map(coords_st, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 5. Likelihood ---
    for i in 1:M.N_obs
        a = M.s_idx[i]
        mu = M.log_offset[i] + trend[i] + seasonal[i] + s_eta[a] + u1_true[i]
        for k in 1:M.N_cov; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma[i]), M.y_obs[i])
    end

    # multi-fidelity links
    M.w_obs[:, 1] ~ MvNormal(u1_true, sigma_w[1]^2 * I)
end


@model function model_C18_kronecker_spde(M, ::Type{T}=Float64) where {T}
    # Kronecker SPDE Spatiotemporal GP.
    # Utilizes Kronecker-structured precision matrices for efficient spatiotemporal inference.
  
    unique_t = collect(1:M.N_time) ./ M.N_time
    unique_s = M.s_obs 

    # --- 1. Priors & Structural Components ---
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    sigma_w ~ filldist(Exponential(0.5), 3)
    c_sigma ~ filldist(Exponential(1.0), 4)

    # BYM2 Spatial Effect
    s_icar ~ MvNormal(zeros(M.N_areas), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.N_areas), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # RW2 Categorical Smoothing
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 2. Latent Kronecker Fields (Z, U1) ---
    ls_s_cov ~ Gamma(2, 2); sigma_s_cov ~ Exponential(1.0)
    ls_t_cov ~ Gamma(2, 2); sigma_t_cov ~ Exponential(1.0)

    z_noise ~ filldist(Normal(0, 1), M.N_obs)
    z_true = kron_matern_sample(M.N_areas, M.N_time, unique_s, unique_t, ls_s_cov, sigma_s_cov, ls_t_cov, sigma_t_cov, z_noise)

    u1_noise ~ filldist(Normal(0, 1), M.N_obs)
    u1_true = kron_matern_sample(M.N_areas, M.N_time, unique_s, unique_t, ls_s_cov, sigma_s_cov, ls_t_cov, sigma_t_cov, u1_noise)

    # --- 3. Latent Kronecker Field for Y (Main Effect) ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    ls_t_y ~ Gamma(2, 2); sigma_t_y ~ Exponential(1.0)
    y_noise ~ filldist(Normal(0, 1), M.N_obs)
    f_st = kron_matern_sample(M.N_areas, M.N_time, unique_s, unique_t, ls_s_y, sigma_s_y, ls_t_y, sigma_t_y, y_noise)

    # --- 4. Stochastic Volatility (RFF) ---
    coords_st = hcat(M.s_obs, M.t_obs)
    W_sigma ~ filldist(Normal(0, 1), size(coords_st, 2), M.M_rff_sigma); b_sigma ~ filldist(Uniform(0, 2pi), M.M_rff_sigma)
    beta_rff_sigma ~ filldist(Normal(0, 1), M.M_rff_sigma)
    y_sigma = exp.(rff_map(coords_st, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 5. Likelihood ---
    for i in 1:M.N_obs
        a = M.s_idx[i]
        mu = M.log_offset[i] + f_st[i] + s_eta[a] + u1_true[i] + z_true[i]
        for k in 1:M.N_cov; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma[i]), M.y_obs[i])
    end

    # Prior links for multi-fidelity variables
    M.z_obs ~ MvNormal(z_true, 0.1 * I)
    M.w_obs[:, 1] ~ MvNormal(u1_true, sigma_w[1]^2 * I)
end


@model function model_C19_svgp_matern(M, ::Type{T}=Float64) where {T}
    # SVGP with Matern structure.
    # Integrates nested RFF latent covariates and Kronecker spatiotemporal kernels.
   
    unique_t = collect(1:M.N_time) ./ M.N_time
    unique_s = M.s_obs[1:M.N_areas, :]  ## need a better implementation 
    # D_st = size(M.s_obs, 2) + 1

    # --- 1. Priors & Structural Components ---
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    sigma_w ~ filldist(Exponential(0.5), 3); c_sigma ~ filldist(Exponential(1.0), 4)

    # BYM2 Spatial Effect
    s_icar ~ MvNormal(zeros(M.N_areas), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.N_areas), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # RW2 Categorical Smoothing
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 2. Nested Latent Covariates (RFF) ---
    coords_tz = hcat(M.t_obs, M.z_obs)
    W_u1 ~ filldist(Normal(0, 1), size(coords_tz, 2), M.M_rff_u); b_u1 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    u1_true = rff_map(coords_tz, W_u1, b_u1) * filldist(Normal(0, 1), M.M_rff_u)

    # --- 3. Main Spatiotemporal Process (Kronecker) ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    ls_t_y ~ Gamma(2, 2); sigma_t_y ~ Exponential(1.0)
    y_noise ~ filldist(Normal(0, 1), M.N_obs)
    f_st = kron_matern_sample(M.N_areas, M.N_time, unique_s, unique_t, ls_s_y, sigma_s_y, ls_t_y, sigma_t_y, y_noise)

    # --- 4. Stochastic Volatility (RFF) ---
    coords_st = hcat(M.s_obs, M.t_obs)
    W_sigma ~ filldist(Normal(0, 1), size(coords_st, 2), M.M_rff_sigma); b_sigma ~ filldist(Uniform(0, 2pi), M.M_rff_sigma)
    beta_rff_sigma ~ filldist(Normal(0, 1), M.M_rff_sigma)
    y_sigma = exp.(rff_map(coords_st, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 5. Likelihood ---
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2pi .* M.t_obs ./ M.period) .+ beta_sin .* sin.(2pi .* M.t_obs ./ M.period)

    for i in 1:M.N_obs
        a = M.s_idx[i]
        mu = M.log_offset[i] + f_st[i] + seasonal[i] + s_eta[a] + u1_true[i]
        for k in 1:M.N_cov; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma[i]), M.y_obs[i])
    end

    # multi-fidelity links
    M.w_obs[:, 1] ~ MvNormal(u1_true, sigma_w[1]^2 * I)
end



@model function model_C20_multifidelity_gp_matern(M, ::Type{T}=Float64) where {T}
    # Multi-fidelity GP with standardized interface.
    # Combines high-fidelity Z (Matern), medium-fidelity U (Kron AR1 x Matern), 
    # and standard-fidelity Y with standardized BYM2 and RW2 components.

    # Dimensions and Observations from M
    y_obs = M.y
    z_obs = M.z_obs
    u1_obs, u2_obs, u3_obs = M.w_obs[:, 1], M.w_obs[:, 2], M.w_obs[:, 3]
    
    Ny = length(y_obs); Nt_y = maximum(M.t_idx); Ns_y = Ny ÷ Nt_y
    Nu = length(u1_obs); Nt_u = Nt_y; Ns_u = Nu ÷ Nt_u
    Nz = length(z_obs)

    # --- 1. Priors ---
    y_sigma ~ Exponential(1.0); s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    st_sigma ~ Exponential(0.5); c_sigma ~ filldist(Exponential(1.0), 4)
    
    # --- 2. High Fidelity: Latent Spatial Z (Matern 3/2) ---
    z_ls ~ Gamma(2, 2); sigma_z_f ~ Exponential(1.0)
    k_z = Matern32Kernel() ∘ ScaleTransform(inv(z_ls))
    K_z = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(M.pts[1:Nz, :])) + M.noise*I
    z_latent ~ MvNormal(zeros(Nz), K_z)
    z_sigma ~ Exponential(0.5)
    z_obs ~ MvNormal(z_latent, z_sigma^2 * I)

    # Interpolation of Z to U/Y locations
    K_z_u = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(M.pts[1:Ns_u, :]), RowVecs(M.pts[1:Nz, :]))
    z_at_u = (K_z_u * (K_z \ z_latent))
    z_at_u_full = repeat(z_at_u, Nt_u)

    # --- 3. Medium Fidelity: Latent Spatiotemporal U (Kron AR1 x Matern) ---
    ls_s_u ~ Gamma(2, 2); sigma_s_u ~ Exponential(1.0)
    rho_u ~ Uniform(-0.99, 0.99); sigma_t_u ~ Exponential(0.5)
    u1_noise ~ filldist(Normal(0, 1), Nu)
    
    # Sample U1 (Hierarchical dependence on Z)
    u1_true = kron_ar1_matern_sample(Ns_u, Nt_u, M.pts[1:Ns_u, :], ls_s_u, sigma_s_u, rho_u, sigma_t_u, u1_noise)
    beta_wz ~ Normal(0, 1)
    u1_obs ~ MvNormal(u1_true .+ beta_wz .* z_at_u_full, 0.1*I)

    # --- 4. Spatial Effect (BYM2) ---
    s_icar ~ MvNormal(zeros(Ns_y), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(Ns_y), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # --- 5. Categorical Smoothing (RW2) ---
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:M.N_cov]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 6. Likelihood (Standard Fidelity Y) ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    rho_y ~ Uniform(-0.99, 0.99); sigma_t_y ~ Exponential(0.5)
    y_noise ~ filldist(Normal(0, 1), Ny)
    f_st_y = kron_ar1_matern_sample(Ns_y, Nt_y, M.pts[1:Ns_y, :], ls_s_y, sigma_s_y, rho_y, sigma_t_y, y_noise)

    beta_y ~ Normal(0, 1)
    for i in 1:Ny
        a, t = M.s_idx[i], M.t_idx[i]
        mu = M.log_offset[i] + f_st_y[i] + s_eta[a] + beta_y * z_at_u_full[i]
        for k in 1:M.N_cov; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), y_obs[i])
    end
end


@model function model_C21_multifidelity_gp_matern_sv_seasonal(M, ::Type{T}=Float64 ) where {T}
    # Multi-fidelity GP with SV, Seasonal Harmonics, and Standardized Interface.

    # Dimensions and Observations
    y_obs = M.y
    z_obs = M.z_obs
    u1_obs, u2_obs, u3_obs = M.w_obs[:, 1], M.w_obs[:, 2], M.w_obs[:, 3]

    Ny = length(y_obs); Nt_y = maximum(M.t_idx); Ns_y = Ny ÷ Nt_y
    Nu = length(u1_obs); Nt_u = Nt_y; Ns_u = Nu ÷ Nt_u
    Nz = length(z_obs)

    # --- 1. High Fidelity: Latent Spatial Z (Matern 3/2) ---
    z_ls ~ Gamma(2, 2); sigma_z_f ~ Exponential(1.0)
    k_z = Matern32Kernel() ∘ ScaleTransform(inv(z_ls))
    K_z = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(M.pts[1:Nz, :])) + 1e-3*I
    z_latent ~ MvNormal(zeros(Nz), K_z)
    z_sigma ~ Exponential(0.5)
    z_obs ~ MvNormal(z_latent, z_sigma^2 * I)

    # Interpolation of Z to U and Y
    K_z_u = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(M.pts[1:Ns_u, :]), RowVecs(M.pts[1:Nz, :]))
    z_at_u = (K_z_u * (K_z \ z_latent))
    K_z_y = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(M.pts[1:Ns_y, :]), RowVecs(M.pts[1:Nz, :]))
    z_at_y = (K_z_y * (K_z \ z_latent))

    # --- 2. Medium Fidelity: Latent Spatiotemporal U (Kron AR1 x Matern) ---
    ls_s_u ~ Gamma(2, 2); sigma_s_u ~ Exponential(1.0)
    rho_u ~ Uniform(-0.99, 0.99); sigma_t_u ~ Exponential(0.5)
    u1_noise ~ filldist(Normal(0, 1), Nu)
    unique_pts_u = M.pts[1:Ns_u, :]
    u1_true = kron_ar1_matern_sample(Ns_u, Nt_u, unique_pts_u, ls_s_u, sigma_s_u, rho_u, sigma_t_u, u1_noise)
    
    beta_wz ~ Normal(0, 1)
    u1_obs ~ MvNormal(u1_true .+ beta_wz .* repeat(z_at_u, Nt_u), 0.1*I)

    # Interpolation of U to Y coordinates (Kronecker)
    unique_pts_y = M.pts[1:Ns_y, :]
    k_s_u_interp = Matern32Kernel() ∘ ScaleTransform(inv(ls_s_u))
    K_s_uu = sigma_s_u^2 * kernelmatrix(k_s_u_interp, RowVecs(unique_pts_u)) + 1e-3*I
    K_s_yu = sigma_s_u^2 * kernelmatrix(k_s_u_interp, RowVecs(unique_pts_y), RowVecs(unique_pts_u))
    
    t_u = collect(1:Nt_u); t_y = collect(1:Nt_y)
    K_t_uu = ar1_covariance_matrix(t_u, rho_u, sigma_t_u) + 1e-3*I
    K_t_yu = ar1_cross_covariance_matrix(t_y, t_u, rho_u, sigma_t_u)
    u1_at_y = kron(K_t_yu, K_s_yu) * (kron(K_t_uu, K_s_uu) \ u1_true)

    # --- 3. Standard Components (BYM2 & RW2) ---
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    s_icar ~ MvNormal(zeros(Ns_y), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(Ns_y), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    c_sigma ~ filldist(Exponential(1.0), 4)
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 4. Stochastic Volatility & Seasonality ---
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    t_vec = M.t_idx ./ Nt_y
    seasonal_y = beta_cos .* cos.(2pi .* t_vec) .+ beta_sin .* sin.(2pi .* t_vec)

    coords_st_y = hcat(M.pts, M.t_idx)
    W_sigma ~ filldist(Normal(0, 1), size(coords_st_y, 2), M.M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2pi), M.M_rff_sigma)
    sigma_log_var ~ Exponential(1.0); beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M.M_rff_sigma)
    y_sigma_vec = exp.(rff_map(coords_st_y, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 5. Likelihood ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    rho_y ~ Uniform(-0.99, 0.99); sigma_t_y ~ Exponential(0.5); y_noise ~ filldist(Normal(0, 1), Ny)
    f_st_y = kron_ar1_matern_sample(Ns_y, Nt_y, unique_pts_y, ls_s_y, sigma_s_y, rho_y, sigma_t_y, y_noise)

    beta_y_covs ~ filldist(Normal(0, 1), 2)
    for i in 1:Ny
        a = M.s_idx[i]
        mu = M.log_offset[i] + f_st_y[i] + s_eta[a] + seasonal_y[i] + (u1_at_y[i] * beta_y_covs[1]) + (repeat(z_at_y, Nt_y)[i] * beta_y_covs[2])
        for k in 1:M.N_cov; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma_vec[i]), y_obs[i])
    end
end
 
 

Turing.@model function pca_carstm( Y, ::Type{T}=Float64 ) where {T}
  # mulitple likelihood model .. 
  # X, G, M.log_offset, y, z, auid, nData, nX, nG, nAU, node1, node2, scaling_factor 
  # Description: Turing model performing Latent PCA (Householder) followed by a spatial CARSTM (BYM2) on factor scores.
  # Inputs:
  #   - Y: Observation matrix (nData x nVar).
  #   - Implicit globals: nAU, nvar, nz, node1, node2, scaling_factor, sigma_prior.
  # Outputs:
  #   - Bayesian posterior estimates for PCA loadings and spatial components.
  # first pca (latent householder transform) then carstm bym2
  
  # pca_sd ~ Bijectors.ordered( MvLogNormal(MvNormal(ones(nz) )) )  
  pca_sd ~ Bijectors.ordered( arraydist( LogNormal.(sigma_prior, 1.0)) )  
  # minimum(pca_sd) < noise && Turing.@addlogprob! Inf  
  # maximum(pca_sd) > nvar && Turing.@addlogprob! Inf 
  pca_pdef_sd ~ LogNormal(0.0, 0.5)
  v ~ filldist(Normal(0.0, 1.0), nvh )
  Kmat, r, U = householder_transform(v, nvar, nz, ltri, pca_sd, pca_pdef_sd, noise)
  # soft priors for r 
  # new .. Gamma in stan is same as in Distributions
  r ~ filldist(Gamma(0.5, 0.5), nz)
  Turing.@addlogprob! sum(-log.(r) .* iz)
  Turing.@addlogprob! -0.5 * sum(pca_sd.^ 2) + (nvar-nz-1) * sum(log.(pca_sd)) 
  Turing.@addlogprob! sum(log.(pca_sd[hindex[:,1]].^ 2) .- pca_sd[hindex[:,2]].^ 2)
  Turing.@addlogprob! sum(log.(2.0 .* pca_sd))
   
  Y ~ filldist( MvNormal( Symmetric(Kmat)), nData )  # latent factors
  
  pcscores = Y' * U 

  # Fixed (covariate) effects (including intercept)
  f_beta ~ filldist( Normal(0.0, 1.0), nz) ;
  
  # icar (spatial effects)
  s_theta ~ filldist( Normal(0.0, 1.0), nAU, nz)  # unstructured (heterogeneous effect)
  s_rho ~ filldist( Normal(0.0, 1.0), nAU, nz) # spatial effects: stan goes from -Inf to Inf .. 
    
  s_sigma ~ filldist( LogNormal(0.0, 1.0), nz) ; 
  s_rho ~ filldist(Beta(0.5, 0.5), nz);
        
  eta ~ filldist( LogNormal(0.0, 1.0), nz ) # overall observation variance
 
  # spatial effects (without inverting covariance) 
  dphi = phi[node1] - phi[node2]
  dot_phi = dot( dphi, dphi )
  Turing.@addlogprob! -0.5 * dot_phi

  # soft sum-to-zero constraint on phi
  sum_phi_s = sum( dot_phi ) 
  sum_phi_s ~ Normal(0, 0.001 * nAU);      # soft sum-to-zero constraint on s_rho)

  convolved_re_s = icar_form( s_theta[:,z], s_rho[:,z], s_sigma[z], s_rho[z] )
  Turing.@addlogprob! -0.5 * dot_phi
  sum_phi_s ~ Normal(0, 0.001 * nAU_float);      # soft sum-to-zero constraint on s_rho)
  mu = f_beta[z] .+ convolved_re_s[auid] 
  pcscores[:,z] ~ MvNormal( mu, eta[z] *I )
   
  return
end
 


Turing.@model function carstm_pca( Y, ::Type{T}=Float64; nData=size(Y, 1), nvar=size(Y, 2), nz=2, nvh=Int(nvar*nz - nz * (nz-1) / 2), noise=1e-9, log_offset=0.0, hindex=(2,1) ) where {T}

    # Description: Turing model that estimates spatial/temporal effects per variable before a latent PCA reduction.
    # Inputs:
    #   - Y: Data matrix.
    #   - nz: Number of latent factors.
    #   - Implicit globals: nAU, nX, node1, node2, scaling_factor.
    # Outputs:
    #   - Bayesian posterior estimates for individual trends and latent PCA structure.
    # first carstm then pca .. as in msmi . incomplete ... too slow

    # Threads.@threads for f in 1:nvar
    for f in 1:nvar
        # est betas, sp, st eff for each sp

        # Fixed (covariate) effects 
        f_beta ~ filldist( Normal(0.0, 1.0), nX);
        f_effect = X * f_beta .+ log_offset

        # icar (spatial effects)
        beta_s ~ filldist( Normal(0.0, 1.0), nX); 
        s_theta ~ filldist( Normal(0.0, 1.0), nAU)  # unstructured (heterogeneous effect)
        s_rho ~ filldist( Normal(0.0, 1.0), nAU) # spatial effects: stan goes from -Inf to Inf .. 
        dphi_s = s_rho[node1] - s_rho[node2]
        Turing.@addlogprob! (-0.5 * dot( dphi_s, dphi_s ))
        sum_phi_s = sum(s_rho) 
        sum_phi_s ~ Normal(0, 0.001 * nAU);      # soft sum-to-zero constraint on s_rho)
        s_sigma ~ truncated( Normal(0.0, 1.0), 0, Inf) ; 
        s_rho ~ Beta(0.5, 0.5);
        # spatial effects:  nAU
        convolved_re_s = s_sigma .*( sqrt.(1 .- s_rho) .* s_theta .+ sqrt.(s_rho ./ scaling_factor) .* s_rho )
        mp_icar =  mp_pca * beta_s +  convolved_re_s[auid]  # mean process for bym2 / icar
        #  @. y ~ LogPoisson( mp_icar);


        # Fourier process (global, main effect)
        ncf = 4  # 2 for seasonal 2 for interannual ..
        t_period ~ filldist( LogNormal(0.0, 0.5), ncf ) 
        t_beta ~ Normal(0, 1)  # linear trend in time

        t_amp ~ MvNormal(Zeros(ncf), I) #  coefficients of harmonic components
 # ~ MvNormal(Zeros(ncf), I) #  coefficients of harmonic components
     #   t_error ~ LogNormal(0, 1)
    #end
    #    mp_fp = rand( MvNormal( mu_fp, t_error^2 * I ) )  
# = t_beta .* ti + sin.( (2pi / t_period) .* ti ) * betahs + cos.((2pi / t_period) .* ti ) * t_amp


        # space X time

    end

    # latent PCA with householder transform  upon the L (latent Y)   
    pca_pdef_sd ~ LogNormal(0.0, 0.5)

    # currently, Bijectors.ordered is broken, revert for better posteriors once it works again
    # sigma ~ Bijectors.ordered( MvLogNormal(MvNormal(ones(nz) )) )  
    sigma ~ filldist(LogNormal(0.0, 1.0), nz ) 
    v ~ filldist(Normal(0.0, 1.0), nvh )

    v_mat = zeros(T, nvar, nz)
    v_mat[ltri] .= v

    U = householder_to_eigenvector( v_mat, nvar, nz )
    
    W = zeros(T, nvar, nz)
    W += U * Diagonal(sigma)
    Kmat = W * W' + (pca_pdef_sd^2 + noise) * I(nvar)

   # soft priors for r 
    # favour reasonably small r .. new .. Gamma in stan is same as in Distributions
    r = sqrt.(mapslices(norm, v_mat[:,1:nz]; dims=1))
    r ~ filldist(Gamma(2.0, 2.0), nz)
    Turing.@addlogprob! sum(-log.(r) .* iz)

    minimum(sigma) < noise && Turing.@addlogprob! Inf && return
    
    Turing.@addlogprob! -0.5 * sum(sigma.^ 2) + (nvar-nz-1) * sum(log.(sigma)) 
    Turing.@addlogprob! sum(log.(sigma[hindex[:,1]].^ 2) .- sigma[hindex[:,2]].^ 2)
    Turing.@addlogprob! sum(log.(2.0 .* sigma))

    mp_pca = rand( (MvNormal( Symmetric(Kmat)), nData ) )

    # Y ~ filldist(MvNormal( Symmetric(Kmat)), nData )

    # y ~ ....

    return 
end
 


Turing.@model function carstm_temperature( Y, ::Type{T}=Float64; 
  nData=size(Y, 1), nvar=size(Y, 2), nz=2, nvh=Int(nvar*nz - nz * (nz-1) / 2), noise=1e-9 ) where {T}
  
    # Description: Turing model for temperature mapping using ICAR (spatial) and Fourier (temporal) processes.
    # Inputs:
    #   - Y: Temperature observation vector.
    #   - Implicit globals: X, nX, nAU, node1, node2, scaling_factor, ti, ncf, vcv.
    # Outputs:
    #   - Posterior distribution of spatial trends and periodic temperature fluctuations.

    # Fixed (covariate) effects 
    #f_beta ~ filldist( Normal(0.0, 1.0), nX);
    #f_effect = X * f_beta + log_offset

    # icar (spatial effects)
    beta_s ~ filldist( Normal(0.0, 1.0), nX); 
    s_theta ~ filldist( Normal(0.0, 1.0), nAU)  # unstructured (heterogeneous effect)
    s_rho ~ filldist( Normal(0.0, 1.0), nAU) # spatial effects: stan goes from -Inf to Inf .. 
    dphi_s = s_rho[node1] - s_rho[node2]
    Turing.@addlogprob! (-0.5 * dot( dphi_s, dphi_s ))
    sum_phi_s = sum(s_rho) 
    sum_phi_s ~ Normal(0, 0.001 * nAU);      # soft sum-to-zero constraint on s_rho)
    s_sigma ~ truncated( Normal(0.0, 1.0), 0, Inf) ; 
    s_rho ~ Beta(0.5, 0.5);

    # spatial effects:  nAU
    convolved_re_s = s_sigma .*( sqrt.(1 .- s_rho) .* s_theta .+ sqrt.(s_rho ./ scaling_factor) .* s_rho )
    mp_icar =  X * beta_s +  convolved_re_s[auid]  # mean process for bym2 / icar
 
    # GP (higher order terms)
    # kernel_var ~ filldist(LogNormal(0.0, 0.5), nG)
    # kernel_scale ~ filldist(LogNormal(0.0, 0.5), nG)

    # k = ( kernel_var[1] * SqExponentialKernel() ) ∘ ScaleTransform(kernel_scale[1])

    # variance process  
    # gp = atomic( Stheno.GP(k), Stheno.GPC())
    # gpo = gp(Xo, I2reg )
    # gpp = gp(Xp, eps() )
    # sfgp = SparseFiniteGP(gpp, gpp)
    # vcv = cov(sfgp.fobs)

    #    --- add more .. but kind of slow 
#    --- ... looking at AbstractGPs as a possible solution

    # gps = rand( MvNormal( mean_process, Symmetric(kmat) ) ) # faster
    # mp_gp = sum(gps, dims=1)  # mean process



    # Fourier process (global, main effect)
    t_period ~ filldist( LogNormal(0.0, 0.5), ncf ) 
    t_beta ~ Normal(0, 1)  # linear trend in time
    t_amp ~ MvNormal(Zeros(ncf), I) #  coefficients of harmonic components
    t_phase ~ MvNormal(Zeros(ncf), I) #  coefficients of harmonic components
    # t_error ~ LogNormal(0, 1)
 
     # fourier effects
    mu_fp = t_beta .* ti + 
        t_amp[1] .* cos.(t_phase[1]) .* sin.( (2pi / t_period[1]) .* ti )   + 
        t_amp[1] .* sin.(t_phase[1]) .* cos.( (2pi / t_period[1]) .* ti )   +
        t_amp[2] .* cos.(t_phase[2]) .* sin.( (2pi / t_period[2]) .* ti )   + 
        t_amp[2] .* sin.(t_phase[2]) .* cos.( (2pi / t_period[2]) .* ti ) 

    # mp_fp = rand( MvNormal( mu_fp, t_error^2 * I ) )  
  
    # space X time


    Y ~ MvNormal( mu_fp .+ mp_icar, Symmetric(vcv) )   # add mvn noise


    return 
end
 


Turing.@model function turing_glm_icar( ; family="poisson", GPmethod="GPsparse", 
    Y=nothing, YG=nothing, good=nothing, 
    X=nothing, G=nothing, Gp=nothing, nInducing=nothing, log_offset=nothing, 
    auid=nothing, nAU=nothing, node1=nothing, node2=nothing, scaling_factor=nothing, tuid=nothing,
    kerneltype="squared_exponential" )

    # almost a full random effect GLM (poisson, binomial and gaussian)
    # spatial random effects with Reibler parameterization
    # covariates (fixed and GP)
    # use this as a basis and strip out uneeded parts to optimize
 
    if !isnothing(good)
                
        if !isnothing(Y)
            Y = Y[good] 
        end

        if !isnothing(YG)
            YG = YG[good] 
        end
        if !isnothing(X)
            X = X[good,:] 
        end
        if !isnothing(G)
            G = G[good,:] 
        end
        if !isnothing(log_offset)
            log_offset = log_offset[good] 
        end
        if !isnothing(auid)
            auid = auid[good] 
        end
        if !isnothing(tuid)
            tuid = tuid[good] 
        end

    end

    nData=length(Y)
    
    Ymu = zeros( nData ) 
    
    if !isnothing(log_offset)
        Ymu .+= log_offset 
    end

    if !isnothing(X)
        # fixed effects
        nX=size(X,2)
        beta ~ filldist( Normal(0.0, 1.0), nX )
        Ymu += X * beta
    end
 
    if !isnothing(G)
        # gaussian process for covariates G
        nG = size(G, 2)
          
        kernel_var ~ filldist( Exponential(1.0), nG )  # can't be much larger than 1 (already scaled)
        kernel_scale ~ filldist( LogNormal(1.0, 2.0), nG )  

        l2reg ~  filldist(Gamma(1.0, 0.001), nG )    # L2 regularization factor for ridge regression
        
#        Gymu = zeros(nInducing, nG)   # component-specific random effect

        sum_Gy_sample = zeros(nG)
        
        YGcurr = YG - Ymu

        for i in 1:nG

            ys = sample_gaussian_process( GPmethod=GPmethod, returntype="sample_loglik", 
                kerneltype=kerneltype,
                kvar=kernel_var[i], kscale=kernel_scale[i],
                Yobs=YGcurr, Xobs=G[:,i], Xinducing=Gp[:,i], lambda=l2reg[i]
            )
            
            Turing.@addlogprob! -ys.loglik 
            Ymu += ys.Yobs_sample
            # Gymu[:,i] ~  rand(fposterior(Gp[:,i], l2reg[i]))   # a mechanism to store sampled mean process 
               
            sum_Gy_sample[i] = sum(ys.Yobs_sample)  
            sum_Gy_sample[i] ~ Normal(0.0, 0.0001 )   # soft sum-to-zero constraint
        end

    end


    if !isnothing(auid)
 
        # spatial effects (without inverting covariance)  
        if isnothing(nAU)
            nAU = length(auid)
        end

        theta ~ filldist( Normal(0.0, 1.0), nAU )  # unstructured (heterogeneous effect)
        phi ~ filldist( Normal(0.0, 1.0), nAU ) # spatial effects: stan goes from -Inf to Inf .. 
    
        sigma ~ Exponential(1.0)   # == Gamma(1,1)
        rho ~   Beta(0.5, 0.5) 
        
        dphi = phi[node1] - phi[node2]
        dot_phi = dot( dphi, dphi )
        Turing.@addlogprob! -0.5 * dot_phi

        # soft sum-to-zero constraint on phi
        sum_phi = sum( dot_phi ) 
        sum_phi ~ Normal(0.0, 0.01 * nAU);      # soft sum-to-zero constraint on s_rho)
        
        Ymu += icar_form( theta, phi, sigma, rho )[auid] 

    end


    # ------------
    # data likelihood
    # 
    # notes: 
    # a method to truncate safely if needed: Ymu = max.(zero(eltype(Ymu)), Ymu) #
    # equivalent ways of expressing likelihood:
    # @. y ~ LogPoisson( Ymu);
    # y ~ arraydist([LogPoisson( Ymu[i] ) for i in 1:nData ])
    # y ~ arraydist(LazyArray(Base.broadcasted((l) -> LogPoisson(l), Ymu)))
    # y ~ arraydist(LazyArray( @~ LogPoisson.(Ymu) ) )

    if family=="poisson"
        Y ~ arraydist( @. LogPoisson( Ymu ) )   
    elseif family=="bernoulli"
        Y ~ arraydist( @. Bernoulli( logistic.(Ymu) ) ) 
    elseif family=="gaussian"
        Ysd ~ Exponential(1.0)
        Y ~ arraydist( @. Normal.( Ymu, Ysd ) ) 
    end

    return nothing
      
end
 

Turing.@model function turing_glm_icar_optimized( ; family="poisson", GPmethod="cholesky", 
    Y=nothing,  YG=nothing, good=nothing, 
    X=nothing, G=nothing, Gp=nothing, nInducing=nothing, log_offset=nothing, 
    auid=nothing, nAU=nothing, node1=nothing, node2=nothing, scaling_factor=nothing, tuid=nothing,
    kerneltype="squared_exponential", gpc=GPC()
    )

    # fast version  .. optimized as much as possible
    # almost a full random effect GLM (poisson, binomial and gaussian)
    # spatial random effects with Reibler parameterization
    # covariates (fixed and GP)
    # use this as a basis and strip out uneeded parts to optimize
 
    if !isnothing(good)
                
        if !isnothing(Y)
            Y = Y[good] 
        end
        
        if !isnothing(YG)
            YG = YG[good] 
        end
        if !isnothing(X)
            X = X[good,:] 
        end
        if !isnothing(G)
            G = G[good,:] 
        end
        if !isnothing(log_offset)
            log_offset = log_offset[good] 
        end
        if !isnothing(auid)
            auid = auid[good] 
        end
        if !isnothing(tuid)
            tuid = tuid[good] 
        end

    end

    nData=length(Y)
   
    Ymu = zeros( nData ) 

    if !isnothing(log_offset)
        Ymu .+= log_offset 
    end

    # fixed and linear effects
    nX = size(X,2)
    # fixed effects
    beta ~ filldist( Normal(0.0, 1.0), nX )
    Ymu += X * beta

    # must have gaussian process for covariates G
    nG = size(G, 2)
    
    kernel_var ~ filldist( Exponential(1.0), nG )  # can't be much larger than 1 (already scaled)
    kernel_scale ~ filldist( LogNormal(1.0, 2.0), nG )  

    l2reg ~  filldist(Gamma(1.0, 0.01), nG )    # L2 regularization factor for ridge regression
    # l2reg = fill(0.001, nG)
#    Gymu = zeros(nInducing, nG)   # component-specific random effects
    sum_Gy_sample = zeros(nG) 
    # α = MvLogNormal(MvNormal(Zeros(nG), I))
    YGcurr = YG - Ymu

    for i in 1:nG
          
        if kerneltype=="default" || kerneltype=="squared_exponential"
            fkernal = kernel_var[i] * SqExponentialKernel() ∘ ScaleTransform( kernel_scale[i]) # ∘ ARDTransform(α)
        end

        if kerneltype=="matern32"
            fkernal = kernel_var[i] * Matern32Kernel() ∘ ScaleTransform( kernel_scale[i]) # ∘ ARDTransform(α)
        end
 
        fgp = atomic(AbstractGPs.GP(fkernal), gpc)

        fobs = fgp( G[:,i], l2reg[i] )
        finducing = fgp( Gp[:,i], l2reg[i] ) 
        
        # only GPsparse or GPvfe 
        
        if !isnothing(match(r"GPsparse", GPmethod))
            fsparse = SparseFiniteGP(fobs, finducing)
            Turing.@addlogprob!  -AbstractGPs.logpdf(fsparse, YGcurr)
            fposterior = posterior(fsparse, YGcurr)
        end
        
        if !isnothing(match(r"GPvfe", GPmethod))
            fsparse = VFE( finducing ) 
            Turing.@addlogprob! -AbstractGPs.elbo( fsparse, fobs, YGcurr )                  
            fposterior = posterior( fsparse, fobs, YGcurr)  
        end

        Gy_sample = rand( fposterior( G[:,i],  l2reg[i] )  )
        sum_Gy_sample[i] = sum( Gy_sample ) 
        sum_Gy_sample[i] ~ Normal(0.0, 0.0001 * nData)  
        
        Ymu += Gy_sample

#        Gymu[:,i] ~  fposterior(Gp[:,i], l2reg[i]) # a mechanism to store sampled mean process 
    end
    
    # spatial effects (without inverting covariance)  
    theta ~ filldist( Normal(0.0, 1.0), nAU )  # unstructured (heterogeneous effect)
    phi ~ filldist( Normal(0.0, 1.0), nAU ) # spatial effects: stan goes from -Inf to Inf .. 

    sigma ~ Exponential(1.0)   
    rho ~   Beta(0.5, 0.5) 
    
    dphi = phi[node1] - phi[node2]
    dot_phi = dot( dphi, dphi )
    Turing.@addlogprob! -0.5 * dot_phi

    # soft sum-to-zero constraint on phi
    sum_phi = sum( dot_phi ) 
    sum_phi ~ Normal(0.0, 0.001 * nAU);      # soft sum-to-zero constraint on s_rho)
    
    # https://sites.stat.columbia.edu/gelman/research/published/bym_article_SSTEproof.pdf
    Ymu += (sigma .* ( sqrt.(1 .- rho) .* theta .+ sqrt.(rho ./ scaling_factor) .* phi ))[auid] 

  
    # ------------
    # data likelihood
    # 
    # notes: 
    # a method to truncate safely if needed: Ymu = max.(zero(eltype(Ymu)), Ymu) #
    # equivalent ways of expressing likelihood:
    # @. y ~ LogPoisson( Ymu);
    # y ~ arraydist([LogPoisson( Ymu[i] ) for i in 1:nData ])
    # y ~ arraydist(LazyArray(Base.broadcasted((l) -> LogPoisson(l), Ymu)))
    # y ~ arraydist(LazyArray( @~ LogPoisson.(Ymu) ) )

    if family=="poisson"
        Y ~ arraydist( @. LogPoisson( Ymu ) )   
    elseif family=="bernoulli"
        Y ~ arraydist( @. Bernoulli( logistic.(Ymu) ) ) 
    elseif family=="gaussian"
        Ysd ~ Exponential(1.0)
        Y ~ arraydist( @. Normal.( Ymu, Ysd ) ) 
    end

    return nothing
end
  


Turing.@model function test_gp0( Y, YG, G, Gp, nInducing, good, i, gpc=GPC() )
               
    Y = Y[good] 
    YG = YG[good]  
    G = G[good,:] 

    nData=length(Y)
    
    Ymu = zeros( nData )  

    nG = 1 # size(G, 2)
    i = 1
    kernel_var ~ LogNormal(0.0, 1.0)  
    kernel_scale ~  LogNormal(0.0, 1.0)   
    l2reg = 0.001    # L2 regularization factor for ridge regression
    
    Gymu = zeros(nInducing)   # component-specific random effect

    sum_Gy_sample = zeros(nG)

    # Kf = kernel_var  * Matern32Kernel()  #  ∘ ScaleTransform(kernel_scale ) 

    Kf = sekernel(kernel_var, kernel_scale  )

        fkernal= Kf
        Yobs=YG
        
        Xobs=G[:,i]
        Xinducing=Gp[:,i]
        lambda=l2reg 
       
        fgp = atomic(AbstractGPs.GP(fkernal), gpc) 
          
        fobs = fgp( Xobs, lambda )
        finducing = fgp( Xinducing, lambda ) 
        
        fposterior = posterior( SparseFiniteGP(fobs, finducing), Yobs)
        
        Gy_sample = rand(fposterior(Xobs, lambda)) 
        Gymu_sample = rand(fposterior(Xinducing, lambda))
                
        # sum_Gy_sample  = sum( Gymu_sample ) 
        # sum_Gy_sample  ~ Normal(0.0, 0.0001 * nInducing )  
        Ymu += Gy_sample
        # Gymu ~ fposterior(Xinducing, lambda)()  # a mechanism to store sampled mean process 
        
        # Main.DEBUG = fposterior(Xinducing, lambda)

        Y ~ arraydist( @. LogPoisson( Ymu ) )   
        
    return nothing
   
end
 


Turing.@model function test_gp1( Y, YG, G, Gp, nInducing, good, i, gpc=GPC() )
               
    Y = Y[good] 
    YG = YG[good]  
    G = G[good,:] 

    nData=length(Y)
    
    Ymu = zeros( nData )  

    nG = 1 # size(G, 2)
    i = 1
    kernel_var ~ LogNormal(0.0, 1.0)  
    kernel_scale ~  LogNormal(0.0, 1.0)   
    l2reg = 0.001    # L2 regularization factor for ridge regression
    sigma ~ LogNormal(0.0,1.0)

    Gymu = zeros(nInducing)   # component-specific random effect

    sum_Gy_sample = zeros(nG)

    # Kf = kernel_var  * Matern32Kernel()  #  ∘ ScaleTransform(kernel_scale ) 

    Kf = sekernel( kernel_var, kernel_scale  )

        fkernal= Kf
        Yobs=YG
        
        Xobs=G[:,i]
        Xinducing=Gp[:,i]
        lambda=l2reg + sigma^2
       
        fgp = atomic(AbstractGPs.GP(fkernal), gpc) # implicit zero-mean
        
        fobs = fgp( Xobs, lambda )
        finducing = fgp( Xinducing, lambda ) 
        
        # fposterior = posterior( SparseFiniteGP(fobs, finducing), Yobs)
        
        # Distribution is MvNormal  sample from dense cov matrix
        Gy_sample ~ fgp( Xobs, lambda) 
        Gymu_sample ~ fgp(Xinducing, lambda)
                
        # sum_Gy_sample  = sum( Gymu_sample ) 
        # sum_Gy_sample  ~ Normal(0.0, 0.0001 * nInducing )  
        Ymu += Gy_sample
        # Gymu ~ fposterior(Xinducing, lambda)()  # a mechanism to store sampled mean process 
        
        # Main.DEBUG = fposterior(Xinducing, lambda)

        Y ~ arraydist( @. LogPoisson( Ymu ) )   
        
    return nothing
   
    
end
 

Turing.@model function test_gp1sparse( Y, YG, G, Gp, nInducing, good, i, gpc=GPC() )
               
    Y = Y[good] 
    YG = YG[good]  
    G = G[good,:] 

    nData=length(Y)
    
    Ymu = zeros( nData )  

    nG = 1 # size(G, 2)
    i = 1
    kernel_var ~ LogNormal(0.0, 1.0)  
    kernel_scale ~  LogNormal(0.0, 1.0)   
    l2reg = 0.001    # L2 regularization factor for ridge regression
    sigma ~ LogNormal(0.0,1.0)

    Gymu = zeros(nInducing)   # component-specific random effect

    sum_Gy_sample = zeros(nG)
     
    # Kf = kernel_var  * Matern32Kernel()  #  ∘ ScaleTransform(kernel_scale ) 

    Kf = sekernel( kernel_var, kernel_scale  )

        fkernal= Kf
        Yobs=YG
        
        Xobs=G[:,i]
        Xinducing=Gp[:,i]
        lambda=l2reg + sigma^2
       
        fgp = atomic(AbstractGPs.GP(fkernal), gpc) # implicit zero-mean
        
        fobs = fgp( Xobs, lambda )
        finducing = fgp( Xinducing, lambda ) 
        
        fposterior = posterior(SparseFiniteGP(fobs, finducing), Yobs)
        
        Main.DEBUG = fposterior

        # Distribution is MvNormal  sample from sparse cov matrix
        Gy_sample ~ fposterior( Xobs, lambda) 
        Gymu_sample ~ fposterior(Xinducing, lambda)
                
        # sum_Gy_sample  = sum( Gymu_sample ) 
        # sum_Gy_sample  ~ Normal(0.0, 0.0001 * nInducing )  
        Ymu += Gy_sample
        # Gymu ~ fposterior(Xinducing, lambda)()  # a mechanism to store sampled mean process 
        

        Y ~ arraydist( @. LogPoisson( Ymu ) )   
        
    return nothing
   
    
end
 


Turing.@model function test_gp1vfe( Y, YG, G, Gp, nInducing, good, i, gpc=GPC() )
               
    Y = Y[good] 
    YG = YG[good]  
    G = G[good,:] 

    nData=length(Y)
    
    Ymu = zeros( nData )  

    nG = 1 # size(G, 2)
    i = 1
    kernel_var ~ LogNormal(0.0, 1.0)  
    kernel_scale ~  LogNormal(0.0, 1.0)   
    l2reg = 0.001    # L2 regularization factor for ridge regression
    sigma ~ LogNormal(0.0,1.0)

    Gymu = zeros(nInducing)   # component-specific random effect

    sum_Gy_sample = zeros(nG)

    # Kf = kernel_var  * Matern32Kernel()  #  ∘ ScaleTransform(kernel_scale ) 

    Kf = sekernel( kernel_var, kernel_scale  )

        fkernal= Kf
        Yobs=YG
        
        Xobs=G[:,i]
        Xinducing=Gp[:,i]
        lambda=l2reg + sigma^2
       
        fgp = atomic(AbstractGPs.GP(fkernal), gpc) # implicit zero-mean
        
        fobs = fgp( Xobs, lambda )

        vfe = VFE( fgp(Xinducing, lambda ) ) 
        fposterior = posterior(vfe, fobs, Yobs)  # Distribution is MvNormal  
 
        Main.DEBUG = fposterior

        # Distribution is MvNormal  sample from sparse cov matrix
        Gy_sample ~ fposterior( Xobs, lambda) 
        Gymu_sample ~ fposterior(Xinducing, lambda)
                
        # sum_Gy_sample  = sum( Gymu_sample ) 
        # sum_Gy_sample  ~ Normal(0.0, 0.0001 * nInducing )  
        Ymu += Gy_sample
        # Gymu ~ fposterior(Xinducing, lambda)()  # a mechanism to store sampled mean process 

        Y ~ arraydist( @. LogPoisson( Ymu ) )   
        
    return nothing
   
    
end
 


Turing.@model function test_gp2( Y, YG, G, Gp, nInducing, good, i )
               
    Y = Y[good] 
    YG = YG[good]  
    G = G[good,:] 

    nData=length(Y)
    
    Ymu = zeros( nData )  

    nG = 1 # size(G, 2)
    i = 1
    kernel_var ~ LogNormal(0.0, 1.0)  
    kernel_scale ~  LogNormal(0.0, 1.0)   
    l2reg = 0.001    # L2 regularization factor for ridge regression
    
    Gymu = zeros(nInducing)   # component-specific random effect

    sum_Gy_sample = zeros(nG)

    # Kf = kernel_var  * Matern32Kernel()  #  ∘ ScaleTransform(kernel_scale ) 

    Kf = sekernel2(kernel_var, kernel_scale  )

        fkernal= Kf
        Yobs=YG
        
        Xobs=G[:,i]
        Xinducing=Gp[:,i]
        lambda=l2reg 
    
        Ko = kernelmatrix( fkernal, vec(Xobs) ) 
        Ki = kernelmatrix( fkernal, vec(Xinducing) ) 
        Kio = kernelmatrix( fkernal, vec(Xinducing), vec(Xobs) ) # transfer to inducing points
        Lo = cholesky(Symmetric( Ko + lambda*I)).L 
        Li = cholesky(Symmetric( Ki + lambda*I)).L   # cholesky on inducing locations  
        
        Gymu_sample  = Kio * ( Lo' \ (Lo \ Yobs ) )  # == mean_process mean latent process
        Gy_sample = Kio' * ( Li' \ (Li \ Gymu_sample ))  # mean process from inducing pts
        
        Gymu_sample  += Li * rand(Normal(0, 1), size(Li,1))   # faster sampling without covariance
        Gy_sample += Lo * rand(Normal(0, 1), size(Lo,2)) # error process 
    
        # sum_Gy_sample  = sum( Gymu_sample ) 
        # sum_Gy_sample  ~ Normal(0.0, 0.0001 * nInducing )  
        Ymu += Gy_sample
        # Gymu ~ fposterior(Xinducing, lambda)()  # a mechanism to store sampled mean process 
        
        # Main.DEBUG = fposterior(Xinducing, lambda)

        Y ~ arraydist( @. LogPoisson( Ymu ) )   
        
    return nothing
 
end
 
;;
