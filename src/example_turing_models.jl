



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
    s_icar ~ MvNormal(zeros(M.s_N), I)
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)

    s_iid ~ MvNormal(zeros(M.s_N), I)

    # Mixture formulation using standard s_eta mapping
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # --- 3. TEMPORAL COMPONENT (AR1) ---
    t_Q = (1.0 / (1.0 - t_rho^2 + M.noise)) .* (M.t_Q + (t_rho^2) * I)
    t_raw ~ MvNormal(zeros(M.t_N), I)
    Turing.@addlogprob! -0.5 * dot(t_raw, t_Q * t_raw)

    t_eta = t_raw .* t_sigma

    # --- 4. POISSON LIKELIHOOD ---
    for i in 1:M.y_N
        a = M.s_idx[i]
        t = M.t_idx[i]

        # Linear Predictor Synthesis
        eta = M.log_offset[i] + s_eta[a] + t_eta[t]

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
    s_icar ~ MvNormal(zeros(M.s_N), I)
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    
    # Soft sum-to-zero constraint for identifiability
    s_icar_sum ~ Normal(0, 0.001 * M.s_N)
    Turing.@addlogprob! -0.5 * (sum(s_icar)^2 / (0.001 * M.s_N))

    # Map to standard s_eta for reconstruction
    s_eta = s_icar .* s_sigma

    # --- 3. TEMPORAL FIELD (AR1) ---
    t_Q = (1.0 / (1.0 - t_rho^2 + M.noise)) .* (M.t_Q + (t_rho^2) * I)
    t_raw ~ MvNormal(zeros(M.t_N), I)
    Turing.@addlogprob! -0.5 * dot(t_raw, t_Q * t_raw)
    
    # Map to standard t_eta for reconstruction
    t_eta = t_raw .* t_sigma

    # --- 4. SPACE-TIME INTERACTION ---
    st_raw ~ MvNormal(zeros(M.s_N * M.t_N), I)
    st_eta = reshape(st_raw .* st_sigma, M.s_N, M.t_N)

    # --- 5. CATEGORICAL SMOOTHING (RW2) ---
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ (c_sigma[k]^2 + M.noise)) * c_beta[k])
    end

    # --- 6. LIKELIHOOD ---
    for i in 1:M.y_N
        a, t = M.s_idx[i], M.t_idx[i]
        eta = M.log_offset[i] + s_eta[a] + t_eta[t] + st_eta[a, t]
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
   y_sigma ~ Exponential(1.0)
    st_sigma ~ Exponential(1.0)
    ls_s ~ Gamma(2, 2)
    ls_t ~ Gamma(2, 2)

    k_s = SqExponentialKernel() ∘ ScaleTransform(inv(ls_s))
    k_t = SqExponentialKernel() ∘ ScaleTransform(inv(ls_t))

    K_s = kernelmatrix(k_s, M.s_coord_unique)
    K_t = kernelmatrix(k_t, M.t_coord_unique)

    K_full = (st_sigma^2) .* kron(K_t, K_s) + M.noise * I

    n_total_latent = length(M.s_coord_unique) * length(M.t_coord_unique)
    s_eta_raw ~ MvNormal(zeros(n_total_latent), Symmetric(K_full))

    s_eta = s_eta_raw[M.interaction_idx]

    mu = M.log_offset .+ s_eta
    M.y_obs ~ MvNormal(mu, y_sigma^2 * I)
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
    s_icar_h ~ MvNormal(zeros(M.s_N), I); Turing.@addlogprob! -0.5 * dot(s_icar_h, M.s_Q * s_icar_h)
    s_iid_h ~ MvNormal(zeros(M.s_N), I)
    # Renamed to s_eta_h for reconstruction compatibility
    s_eta_h = s_sigma_h .* (sqrt(s_rho_h) .* s_icar_h .+ sqrt(1 - s_rho_h) .* s_iid_h)

    t_Q_h = (1.0 / (1.0 - t_rho_h^2)) .* (M.t_Q + (t_rho_h^2) * I)
    t_f_h ~ MvNormal(zeros(M.t_N), I); Turing.@addlogprob! -0.5 * dot(t_f_h, t_Q_h * t_f_h)
    # Renamed to t_eta_h for reconstruction compatibility
    t_eta_h = t_f_h .* t_sigma_h

    # --- 3. Latent Fields for Count Part ---
    s_icar_c ~ MvNormal(zeros(M.s_N), I); Turing.@addlogprob! -0.5 * dot(s_icar_c, M.s_Q * s_icar_c)
    s_iid_c ~ MvNormal(zeros(M.s_N), I)
    # Renamed to s_eta_c for reconstruction compatibility
    s_eta_c = s_sigma_c .* (sqrt(s_rho_c) .* s_icar_c .+ sqrt(1 - s_rho_c) .* s_iid_c)

    t_Q_c = (1.0 / (1.0 - t_rho_c^2)) .* (M.t_Q + (t_rho_c^2) * I)
    t_f_c ~ MvNormal(zeros(M.t_N), I); Turing.@addlogprob! -0.5 * dot(t_f_c, t_Q_c * t_f_c)
    # Renamed to t_eta_c for reconstruction compatibility
    t_eta_c = t_f_c .* t_sigma_c

    # --- 4. Shared Categorical Smoothing ---
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 5. Joint Hurdle Likelihood ---
    for i in 1:M.y_N
        a, t = M.s_idx[i], M.t_idx[i]

        # Linear predictors
        eta_h = s_eta_h[a] + t_eta_h[t]
        eta_c = M.log_offset[i] + s_eta_c[a] + t_eta_c[t]
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




@model function example_rff_3D(M, ::Type{T}=Float64; m_joint=25 ) where {T}
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
    # We use M.s_coord_tuple and normalize them for numerical stability
    xs = [p[1] for p in M.s_coord_tuple]
    ys = [p[2] for p in M.s_coord_tuple]
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
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 5. Likelihood ---
    for i in 1:M.y_N
        mu = M.log_offset[i] + eta_joint[i]
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), M.y_obs[i])
    end
end


@model function example_rff_2D(M, ::Type{T}=Float64 ) where {T}
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
    xs = [p[1] for p in M.s_coord_tuple]
    ys = [p[2] for p in M.s_coord_tuple]
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
    t_raw ~ MvNormal(zeros(M.t_N), I)
    Turing.@addlogprob! -0.5 * dot(t_raw, t_Q * t_raw)
    t_eta = t_raw .* t_sigma

    # --- 5. Categorical & Likelihood ---
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    for i in 1:M.y_N
        mu = M.log_offset[i] + s_eta[i] + t_eta[M.t_idx[i]]
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), M.y_obs[i])
    end
end




@model function example_warping_2D(M, ::Type{T}=Float64) where {T}

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
    xs = [p[1] for p in M.s_coord_tuple]
    ys = [p[2] for p in M.s_coord_tuple]
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
    t_raw ~ MvNormal(zeros(M.t_N), I)
    Turing.@addlogprob! -0.5 * dot(t_raw, t_Q * t_raw)
    t_eta = t_raw .* t_sigma

    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 6. Likelihood ---
    for i in 1:M.y_N
        mu = M.log_offset[i] + s_eta[i] + t_eta[M.t_idx[i]]
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), M.y_obs[i])
    end
end




@model function example_rff_2D_mosaic(M, ::Type{T}=Float64 ) where {T}
      
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
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 2. Spatial Indexing & Soft Boundary M.Weights ---
    coords = hcat([p[1] for p in M.s_coord_tuple], [p[2] for p in M.s_coord_tuple])
    R = kmeans(coords', M.n_mosaics)
    centroids = R.centers # 2 x M.n_mosaics

    # Pre-sample RFF frequencies
    Random.seed!(42)
    Om_m = [randn(M.m_rff, 2) for _ in 1:M.n_mosaics]
    Ph_m = [rand(M.m_rff) for _ in 1:M.n_mosaics]

    # --- 3. Likelihood with Soft Integration ---
    for i in 1:M.y_N
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
        for k in 1:M.cov_N; 
            mu += c_beta[k][M.cov_indices[i, k]]; 
        end
        
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma_combined + 1e-4), M.y_obs[i])
    end
end


@model function example_rff_3D_mosaic(M, ::Type{T}=Float64 ) where {T}

    # --- 1. Global Hierarchical Priors ---
    c_sigma ~ filldist(Exponential(1.0), 4)
    mu_global ~ Normal(0, 1)
    sigma_mu_local ~ Exponential(0.5)

    # Shared Categorical Effects
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.cov_N
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
    xs = [p[1] for p in M.s_coord_tuple]
    ys = [p[2] for p in M.s_coord_tuple]
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
    for i in 1:M.y_N
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
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end

        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma_total + M.noise), M.y_obs[i])
    end
end



@model function example_fitc_2D(M, ::Type{T}=Float64) where {T}
    # FITC Sparse GP using fixed inducing point priors for performance optimization.
    # Architecture: [1] BYM2 Spatial, [2] AR1 Temporal, [3] Harmonic Seasonal, [4] GMRF Interaction, [5] RW2 Smoothing.
 
    # Priors
    y_sigma ~ Exponential(1.0)
    f_sigma ~ Exponential(1.0)
    s_sigma ~ Exponential(1.0)
    s_rho ~ Beta(1, 1)
    t_sigma ~ Exponential(1.0)
    t_rho ~ Beta(2, 2)
    st_sigma ~ Exponential(0.5)
    st_ls ~ filldist(Gamma(2, 2), 3)
    c_sigma ~ filldist(Exponential(1.0), 4)
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)

    # Component 1: BYM2
    s_icar ~ MvNormal(zeros(M.s_N), I)
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # Component 2: AR1
    t_Q = (1.0 / (1.0 - t_rho^2)) .* (M.t_Q + (t_rho^2) * I)
    t_raw ~ MvNormal(zeros(M.t_N), I)
    Turing.@addlogprob! -0.5 * dot(t_raw, t_Q * t_raw)
    t_eta = t_raw .* t_sigma

    # Component 3: Seasonal
    t_vec = Float64.(M.t_idx)
    seasonal = beta_cos .* cos.(2pi .* t_vec ./ M.period) .+ beta_sin .* sin.(2pi .* t_vec ./ M.period)

    # Component 4: GMRF Interaction
    st_raw ~ MvNormal(zeros(M.s_N * M.t_N), I)
    st_eta = reshape(st_raw .* st_sigma, M.s_N, M.t_N)

    # Component 5: Fixed Inducing FITC (Optimized GP Projection)
    # t_norm = M.t_idx ./ M.t_N
    # coords_st = hcat([p[1] for p in M.s_coord_tuple], [p[2] for p in M.s_coord_tuple], t_norm)
    k_st = (f_sigma^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(st_ls)))
    K_XZ = kernelmatrix(k_st, RowVecs(M.st_coord_normalized), RowVecs(M.Z_inducing))
    u_inducing ~ MvNormal(zeros(size(M.Z_inducing, 1)), I) # Assumes unit variance at inducing points
    f_gp = K_XZ * u_inducing # Linear projection

    # RW2 Categorical
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # Likelihood
    for i in 1:M.y_N
        idx_a = M.s_idx[i]
        idx_t = M.t_idx[i]
        mu = M.log_offset[i] + f_gp[i] + s_eta[idx_a] + t_eta[idx_t] + seasonal[i] + st_eta[idx_a, idx_t]
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), M.y_obs[i])
    end
end




@model function example_svgp(M, ::Type{T}=Float64) where {T}
    # SVGP-like (learned inducing points) with GP Trend and Non-linear Nested Covariates.
  
    # --- 1. Priors ---
    w_sigma ~ filldist(Exponential(0.5), 3)
    f_sigma ~ Exponential(1.0); st_ls ~ filldist(Gamma(2, 2), M.st_D)
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    c_sigma ~ filldist(Exponential(1.0), 4)
    sigma_log_var ~ Exponential(1.0)

    # --- 2. GP Temporal Trend ---
    ls_trend ~ Gamma(2, 2); sigma_trend ~ Exponential(0.5)
    k_trend = (sigma_trend^2) * (SqExponentialKernel() ∘ ScaleTransform(inv(ls_trend)))
    t_unique = collect(1:M.t_N) ./ M.t_N
    alpha_gp ~ MvNormal(zeros(M.t_N), kernelmatrix(k_trend, t_unique) + M.noise*I)

    # --- 3. Non-linear Nested Structure (RFF) ---
    t_norm = M.t_idx ./ M.t_N
    coords_tz = hcat(t_norm, M.z_obs)
    W_u1 ~ filldist(Normal(0, 1), 2, M.M_rff_u); b_u1 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    f_sigma_u1 ~ Exponential(1.0); beta_rff_u1 ~ filldist(Normal(0, f_sigma_u1), M.M_rff_u)
    u1_true = (sqrt(2/M.M_rff_u) .* cos.(coords_tz * W_u1 .+ b_u1')) * beta_rff_u1

    coords_tz_u1 = hcat(t_norm, M.z_obs, u1_true)
    W_u2 ~ filldist(Normal(0, 1), 3, M.M_rff_u); b_u2 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    f_sigma_u2 ~ Exponential(1.0); beta_rff_u2 ~ filldist(Normal(0, f_sigma_u2), M.M_rff_u)
    u2_true = (sqrt(2/M.M_rff_u) .* cos.(coords_tz_u1 * W_u2 .+ b_u2')) * beta_rff_u2

    W_u3 ~ filldist(Normal(0, 1), 3, M.M_rff_u); b_u3 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    f_sigma_u3 ~ Exponential(1.0); beta_rff_u3 ~ filldist(Normal(0, f_sigma_u3), M.M_rff_u)
    u3_true = (sqrt(2/M.M_rff_u) .* cos.(coords_tz_u1 * W_u3 .+ b_u3')) * beta_rff_u3
    u_true_mat = hcat(u1_true, u2_true, u3_true)

    # --- 4. Spatial Effect (BYM2) ---
    s_icar ~ MvNormal(zeros(M.s_N), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # --- 5. SVGP Mean (Learned Inducing Points) ---
    Z_inducing = Matrix{T}(undef, M.M_inducing_val, M.st_D)
    for j in 1:M.st_D
        # Learn inducing locations via prior-constrained exploration
        Z_inducing[:, j] ~ filldist(Normal(mean(M.st_coord_normalized[:,j]), 2*std(M.st_coord_normalized[:,j])), M.M_inducing_val)
    end

    k_st = (f_sigma^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(st_ls)))
    K_ZZ = kernelmatrix(k_st, RowVecs(Z_inducing)) + M.noise * I
    K_XZ = kernelmatrix(k_st, RowVecs(M.st_coord_normalized), RowVecs(Z_inducing))
    K_XX_diag = diag(kernelmatrix(k_st, RowVecs(M.st_coord_normalized)))

    u_inducing ~ MvNormal(zeros(M.M_inducing_val), K_ZZ)
    f_mean = K_XZ * (K_ZZ \ u_inducing)
    cov_f_diag = K_XX_diag - diag(K_XZ * (K_ZZ \ K_XZ'))
    f_gp ~ MvNormal(f_mean, Diagonal(max.(M.noise, cov_f_diag)))

    # --- 6. Volatility & Categorical Smoothing ---
    W_vol ~ filldist(Normal(0, 1), M.st_D, M.M_rff_sigma); b_vol ~ filldist(Uniform(0, 2pi), M.M_rff_sigma)
    beta_rff_vol ~ filldist(Normal(0, sigma_log_var), M.M_rff_sigma)
    log_y_sigma = (sqrt(2/M.M_rff_sigma) .* cos.(M.st_coord_normalized * W_vol .+ b_vol')) * beta_rff_vol
    y_sigma = exp.(log_y_sigma ./ 2.0)

    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 7. Likelihoods ---
    z_beta ~ Normal(0, 2); beta_w_main ~ MvNormal(zeros(3), 2.0 * I)
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2pi .* t_norm * (M.t_N/M.period)) .+ beta_sin .* sin.(2pi .* t_norm * (M.t_N/M.period))

    for i in 1:M.y_N
        a, t = M.s_idx[i], M.t_idx[i]
        mu = M.log_offset[i] + alpha_gp[t] + seasonal[i] + z_beta * M.z_obs[i] + dot(beta_w_main, u_true_mat[i, :]) + f_gp[i] + s_eta[a]
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma[i]), M.y_obs[i])
        for k in 1:3; Turing.@addlogprob! logpdf(Normal(u_true_mat[i, k], w_sigma[k]), M.w_obs[i, k]); end
    end
end


@model function example_svgp_nested(M, ::Type{T}=Float64) where {T}
    # Full SVGP logic (learned inducing locations and latent distribution) with GP Trend.
 
    # --- 1. Priors ---
    w_sigma ~ filldist(Exponential(0.5), M.cov_N)
    f_sigma ~ Exponential(1.0); st_ls ~ filldist(Gamma(2, 2), M.st_D)
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    c_sigma ~ filldist(Exponential(1.0), 4)
    sigma_log_var ~ Exponential(1.0)

    # --- 2. GP Temporal Trend ---
    ls_trend ~ Gamma(2, 2); sigma_trend ~ Exponential(0.5)
    k_trend = (sigma_trend^2) * (SqExponentialKernel() ∘ ScaleTransform(inv(ls_trend)))
    t_unique = collect(1:M.t_N) ./ M.t_N
    alpha_gp ~ MvNormal(zeros(M.t_N), kernelmatrix(k_trend, t_unique) + M.noise*I)

    # --- 3. Non-linear Nested Structure (RFF) ---
    t_norm = M.t_idx ./ M.t_N
    coords_tz = hcat(t_norm, M.z_obs)
    W_u1 ~ filldist(Normal(0, 1), 2, M.M_rff_u); b_u1 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    f_sigma_u1 ~ Exponential(1.0); beta_rff_u1 ~ filldist(Normal(0, f_sigma_u1), M.M_rff_u)
    u1_true = (sqrt(2/M.M_rff_u) .* cos.(coords_tz * W_u1 .+ b_u1')) * beta_rff_u1

    coords_tz_u1 = hcat(t_norm, M.z_obs, u1_true)
    W_u2 ~ filldist(Normal(0, 1), 3, M.M_rff_u); b_u2 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    f_sigma_u2 ~ Exponential(1.0); beta_rff_u2 ~ filldist(Normal(0, f_sigma_u2), M.M_rff_u)
    u2_true = (sqrt(2/M.M_rff_u) .* cos.(coords_tz_u1 * W_u2 .+ b_u2')) * beta_rff_u2

    W_u3 ~ filldist(Normal(0, 1), 3, M.M_rff_u); b_u3 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    f_sigma_u3 ~ Exponential(1.0); beta_rff_u3 ~ filldist(Normal(0, f_sigma_u3), M.M_rff_u)
    u3_true = (sqrt(2/M.M_rff_u) .* cos.(coords_tz_u1 * W_u3 .+ b_u3')) * beta_rff_u3
    u_true_mat = hcat(u1_true, u2_true, u3_true)

    # --- 4. Spatial Effect (BYM2) ---
    s_icar ~ MvNormal(zeros(M.s_N), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # --- 5. Full SVGP Mean (Learned Locations and Latent Params) ---
    Z_inducing = Matrix{T}(undef, M.M_inducing_val, M.st_D)
    for j in 1:M.st_D
        Z_inducing[:, j] ~ filldist(Normal(mean(M.st_coord_normalized[:,j]), 2*std(M.st_coord_normalized[:,j])), M.M_inducing_val)
    end

    # Variational parameters for the inducing points
    m_u ~ MvNormal(zeros(M.M_inducing_val), 10.0 * I)
    s_u_diag ~ filldist(Exponential(1.0), M.M_inducing_val)
    w_latent ~ MvNormal(m_u, Diagonal(s_u_diag.^2) + M.noise*I)

    k_st = (f_sigma^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(st_ls)))
    K_ZZ = kernelmatrix(k_st, RowVecs(Z_inducing)) + M.noise * I
    K_XZ = kernelmatrix(k_st, RowVecs(M.st_coord_normalized), RowVecs(Z_inducing))
    K_XX_diag = diag(kernelmatrix(k_st, RowVecs(M.st_coord_normalized)))

    f_mean = K_XZ * (K_ZZ \ w_latent)
    cov_f_diag = K_XX_diag - diag(K_XZ * (K_ZZ \ K_XZ'))
    f_gp ~ MvNormal(f_mean, Diagonal(max.(M.noise, cov_f_diag)))

    # --- 6. Volatility & Categorical Smoothing ---
    W_vol ~ filldist(Normal(0, 1), M.st_D, M.M_rff_sigma); b_vol ~ filldist(Uniform(0, 2pi), M.M_rff_sigma)
    beta_rff_vol ~ filldist(Normal(0, sigma_log_var), M.M_rff_sigma)
    log_y_sigma = (sqrt(2/M.M_rff_sigma) .* cos.(M.st_coord_normalized * W_vol .+ b_vol')) * beta_rff_vol
    y_sigma = exp.(log_y_sigma ./ 2.0)

    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 7. Likelihoods ---
    z_beta ~ Normal(0, 2); beta_w_main ~ MvNormal(zeros(3), 2.0 * I)
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2pi .* t_norm * (M.t_N/M.period)) .+ beta_sin .* sin.(2pi .* t_norm * (M.t_N/M.period))

    for i in 1:M.y_N
        a, t = M.s_idx[i], M.t_idx[i]
        mu = M.log_offset[i] + alpha_gp[t] + seasonal[i] + z_beta * M.z_obs[i] + dot(beta_w_main, u_true_mat[i, :]) + f_gp[i] + s_eta[a]
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma[i]), M.y_obs[i])
        for k in 1:3; Turing.@addlogprob! logpdf(Normal(u_true_mat[i, k], w_sigma[k]), M.w_obs[i, k]); end
    end
end


@model function example_multifidelity(M, ::Type{T}=Float64) where {T}
    # --- 1. Data Dimensions & Multi-fidelity Inputs ---
    
    z_obs = M.z_obs
    u_obs = M.w_obs
      
    # --- 2. Hierarchical Priors ---
    y_sigma ~ Exponential(1.0) # Standard noise
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1) # BYM2 params
    t_sigma ~ Exponential(1.0); t_rho ~ Beta(2, 2) # AR1 params
    st_sigma ~ Exponential(0.5); c_sigma ~ filldist(Exponential(1.0), 4) # Smoothing
    sigma_z ~ Exponential(0.5); w_sigma ~ filldist(Exponential(0.5), 3) # Fidelity noise
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1) # Seasonal M.weights

    # --- 3. Component 1: BYM2 Spatial Effect ---
    s_icar ~ MvNormal(zeros(M.s_N), I)
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # --- 4. Component 2: AR1 Temporal Effect ---
    t_Q = (1.0 / (1.0 - t_rho^2)) .* (M.t_Q + (t_rho^2) * I)
    t_raw ~ MvNormal(zeros(M.t_N), I)
    Turing.@addlogprob! -0.5 * dot(t_raw, t_Q * t_raw)
    t_eta = t_raw .* t_sigma

    # --- 5. Component 3: Harmonic Seasonality ---
    t_vec = Float64.(M.t_idx)
    seasonal = beta_cos .* cos.(2pi .* t_vec ./ M.period) .+ beta_sin .* sin.(2pi .* t_vec ./ M.period)

    # --- 6. Component 4: Space-Time Interaction (GMRF) ---
    st_raw ~ MvNormal(zeros(M.s_N * M.t_N), I)
    st_eta = reshape(st_raw .* st_sigma, M.s_N, M.t_N)

    # --- 7. Component 5: Multi-fidelity RFF Projections ---
    # Latent Z Fidelity
    W_z ~ filldist(Normal(0, 1), size(M.z_coords_s, 2), M.M_rff)
    b_z ~ filldist(Uniform(0, 2pi), M.M_rff)
    beta_z_rff ~ filldist(Normal(0, 1), M.M_rff)
    z_latent = rff_map(M.z_coords_s, W_z, b_z) * beta_z_rff
    Turing.@addlogprob! logpdf(MvNormal(z_latent, sigma_z^2 * I), M.z_obs)

    # Latent U Fidelity
    W_u ~ filldist(Normal(0, 1), size(M.w_coords_st, 2), M.M_rff)
    b_u ~ filldist(Uniform(0, 2pi), M.M_rff)
    beta_w_rff ~ filldist(Normal(0, 1), M.M_rff, 3)
    w_latent = rff_map(M.w_coords_st, W_u, b_u) * beta_w_rff
    for k in 1:3; Turing.@addlogprob! logpdf(MvNormal(w_latent[:, k], w_sigma[k]^2 * I), u_obs[:, k]); end

    # --- 8. Categorical Smoothing (RW2) ---
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 9. Final Likelihood ---
    for i in 1:M.y_N
        idx_a = M.s_idx[i]
        idx_t = M.t_idx[i]
        mu = M.log_offset[i] + s_eta[idx_a] + t_eta[idx_t] + seasonal[i] + st_eta[idx_a, idx_t]
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), M.y_obs[i])
    end
end

@model function example_minibatch(M, ::Type{T}=Float64) where {T}
      
    # Priors
    y_sigma ~ Exponential(1.0); s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    t_sigma ~ Exponential(1.0); t_rho ~ Beta(2, 2); st_sigma ~ Exponential(0.5)
    c_sigma ~ filldist(Exponential(1.0), 4); beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)

    # BYM2
    s_icar ~ MvNormal(zeros(M.s_N), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I); s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # AR1
    t_Q = (1.0 / (1.0 - t_rho^2)) .* (M.t_Q + (t_rho^2) * I)
    t_raw ~ MvNormal(zeros(M.t_N), I); Turing.@addlogprob! -0.5 * dot(t_raw, t_Q * t_raw)
    t_eta = t_raw .* t_sigma

    # Seasonal
    seasonal = beta_cos .* cos.(2pi .* Float64.(M.t_idx) ./ M.period) .+ beta_sin .* sin.(2pi .* Float64.(M.t_idx) ./ M.period)

    # GMRF Interaction
    st_raw ~ MvNormal(zeros(M.s_N * M.t_N), I); st_eta = reshape(st_raw .* st_sigma, M.s_N, M.t_N)

    # RW2
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # Likelihood
    for i in 1:M.y_N
        idx_a = M.s_idx[i]; idx_t = M.t_idx[i]
        mu = M.log_offset[i] + s_eta[idx_a] + t_eta[idx_t] + seasonal[i] + st_eta[idx_a, idx_t]
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), M.y_obs[i])
    end
end


@model function example_deepgp(M, ::Type{T}=Float64) where {T}
    # Deep Spatiotemporal GP.
    # Integrates BYM2 spatial effects and RW2 smoothing into a 3-layer RFF hierarchy.
   
     
    # --- 1. Priors & Structural Components ---
    y_sigma ~ Exponential(1.0); s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    sigma_z ~ Exponential(0.5); w_sigma ~ filldist(Exponential(0.5), 3)
    c_sigma ~ filldist(Exponential(1.0), 4)

    # BYM2 Spatial Effect
    s_icar ~ MvNormal(zeros(M.s_N), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # RW2 Categorical Smoothing
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 2. Layer 1: Latent Spatial GP (Z) ---
    W_z ~ filldist(Normal(0, 1), M.s_D, M.M_rff)
    b_z ~ filldist(Uniform(0, 2pi), M.M_rff)
    z_beta ~ filldist(Normal(0, 1), M.M_rff)
    z_latent = rff_map(M.s_coord, W_z, b_z) * z_beta

    # --- 3. Layer 2: Latent Spatiotemporal GPs (U1, U2, U3) ---
    coords_l2 = hcat(M.s_coord, M.t_coord, z_latent)
    W_u ~ filldist(Normal(0, 1), size(coords_l2, 2), M.M_rff)
    b_u ~ filldist(Uniform(0, 2pi), M.M_rff)
    beta_w_mat ~ filldist(Normal(0, 1), M.M_rff, 3)
    Phi_u = rff_map(coords_l2, W_u, b_u)
    w_latent = Phi_u * beta_w_mat # Matrix M.y_N x 3

    # --- 4. Layer 3: Final Output GP (Y) ---
    coords_l3 = hcat(M.s_coord, M.t_coord, w_latent)
    W_y ~ filldist(Normal(0, 1), size(coords_l3, 2), M.M_rff)
    b_y ~ filldist(Uniform(0, 2pi), M.M_rff)
    beta_y_gp ~ filldist(Normal(0, 1), M.M_rff)

    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2pi .* M.t_coord ./ M.period) .+ beta_sin .* sin.(2pi .* M.t_coord ./ M.period)
    f_y = (rff_map(coords_l3, W_y, b_y) * beta_y_gp) .+ vec(seasonal)

    # --- 5. Likelihood ---
    for i in 1:M.y_N
        a = M.s_idx[i]
        mu = M.log_offset[i] + f_y[i] + s_eta[a]
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), M.y_obs[i])
    end

    # Multi-fidelity cross-resolution constraints (Optional prior links)
    M.z_obs ~ MvNormal(z_latent, sigma_z^2 * I)
    for k in 1:3; M.w_obs[:, k] ~ MvNormal(w_latent[:, k], w_sigma[k]^2 * I); end
end


@model function example_nystrom(M, ::Type{T}=Float64) where {T}
    # Model A16 Optimized: Standardized Nyström GP with Stochastic Volatility.
    # Combines low-rank spatiotemporal approximations with heteroskedastic noise modeling.
 
     # --- 1. Priors & Structural Components ---
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    w_sigma ~ filldist(Exponential(0.5), 3)
    c_sigma ~ filldist(Exponential(1.0), 4)

    # BYM2 Spatial Effect
    s_icar ~ MvNormal(zeros(M.s_N), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # RW2 Categorical Smoothing
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 2. Nested Latent Covariates (RFF Mapping) ---
    coords_tz = hcat(M.t_coord, M.z_obs)
    W_u1 ~ filldist(Normal(0, 1), size(coords_tz, 2), M.M_rff_u); b_u1 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    u1_true = rff_map(coords_tz, W_u1, b_u1) * filldist(Normal(0, 1), M.M_rff_u)

    coords_tz_u1 = hcat(M.t_coord, M.z_obs, u1_true)
    W_u2 ~ filldist(Normal(0, 1), size(coords_tz_u1, 2), M.M_rff_u); b_u2 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    u2_true = rff_map(coords_tz_u1, W_u2, b_u2) * filldist(Normal(0, 1), M.M_rff_u)

    # --- 3. Nyström GP (Low Rank Approximation) ---
    st_ls ~ filldist(Gamma(2, 2), M.st_D); 
    f_sigma ~ Exponential(1.0)
    k_st = SqExponentialKernel() ∘ ARDTransform(inv.(st_ls + M.eps))

    Z_ind = Matrix{T}(undef, M.M_inducing_val, M.st_D)
    st_mu, st_std = mean(M.st_coord, dims=1), std(M.st_coord, dims=1)
    for j in 1:M.st_D; Z_ind[:, j] ~ filldist(Normal(st_mu[j], 2.0 * st_std[j]), M.M_inducing_val); end

    K_zz = Symmetric( kernelmatrix(k_st, RowVecs(Z_ind)) + M.eps*I)
    K_xz = kernelmatrix(k_st, RowVecs(M.st_coord), RowVecs(Z_ind))
    v_latent ~ filldist(Normal(0, 1), M.M_inducing_val)
    f_nystrom = f_sigma .* (K_xz * (cholesky(K_zz).U \ v_latent))

    # --- 4. Spatiotemporal Stochastic Volatility ---
    W_sigma ~ filldist(Normal(0, 1), M.st_D, M.M_rff_sigma); b_sigma ~ filldist(Uniform(0, 2pi), M.M_rff_sigma)
    beta_rff_sigma ~ filldist(Normal(0, 1), M.M_rff_sigma)
    y_sigma = exp.(rff_map(M.st_coord, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 5. Likelihood ---
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2pi .* M.t_coord ./ M.period) .+ beta_sin .* sin.(2pi .* M.t_coord ./ M.period)

    for i in 1:M.y_N
        a = M.s_idx[i]
        mu = M.log_offset[i] + f_nystrom[i] + seasonal[i] + s_eta[a]
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma[i]), M.y_obs[i])
    end

    # Prior links for latent variables
    M.w_obs[:, 1] ~ MvNormal(u1_true, w_sigma[1]^2 * I)
    M.w_obs[:, 2] ~ MvNormal(u2_true, w_sigma[2]^2 * I)
end


@model function example_spde_nested(M, ::Type{T}=Float64 ) where {T}
    # SPDE-based Spatiotemporal GP.
    # Employs sparse precision approximations for spatial effects and RFF for volatility.
 
    # --- 1. Priors & Structural Components ---
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    w_sigma ~ filldist(Exponential(0.5), 3)
    c_sigma ~ filldist(Exponential(1.0), 4)

    # BYM2 Spatial Effect
    s_icar ~ MvNormal(zeros(M.s_N), I); 
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # RW2 Categorical Smoothing
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 2. Latent Trends (GP Trend & Seasonal) ---
    ls_trend ~ Gamma(2, 2); sigma_trend ~ Exponential(0.5)

    k_trend = SqExponentialKernel() ∘ ScaleTransform(inv(ls_trend))
    unique_times = sort(unique(M.t_coord[:,1]))
    
    K_trend = Symmetric(sigma_trend^2 * kernelmatrix(k_trend, unique_times) + 1e-4 * I)
    alpha ~ MvNormal(zeros(length(unique_times)), K_trend)

    # alpha ~ GP(sigma_trend^2 * k_trend)(unique_times, M.noise)
    trend = alpha[indexin(M.t_coord[:,1], unique_times)]

    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2pi .* M.t_coord ./ M.period) .+ beta_sin .* sin.(2pi .* M.t_coord ./ M.period)

    # --- 3. Nested Latent Covariates (RFF) ---
    coords_tz = hcat(M.t_coord, M.z_obs)
    W_u1 ~ filldist(Normal(0, 1), size(coords_tz, 2), M.M_rff_u); b_u1 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    u1_true = rff_map(coords_tz, W_u1, b_u1) * filldist(Normal(0, 1), M.M_rff_u)

    # --- 4. Stochastic Volatility ---
    W_sigma ~ filldist(Normal(0, 1), M.st_D, M.M_rff_sigma); b_sigma ~ filldist(Uniform(0, 2pi), M.M_rff_sigma)
    beta_rff_sigma ~ filldist(Normal(0, 1), M.M_rff_sigma)
    y_sigma = exp.(rff_map(M.st_coord, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 5. Likelihood ---
    for i in 1:M.y_N
        a = M.s_idx[i]
        mu = M.log_offset[i] + trend[i] + seasonal[i] + s_eta[a] + u1_true[i]
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma[i]), M.y_obs[i])
    end

    # multi-fidelity links
    M.w_obs[:, 1] ~ MvNormal(u1_true, w_sigma[1]^2 * I)
end


@model function example_kronecker_spde_nested(M, ::Type{T}=Float64) where {T}
    # Kronecker SPDE Spatiotemporal GP.
    # Utilizes Kronecker-structured precision matrices for efficient spatiotemporal inference.
  
    unique_t = collect(1:M.t_N) ./ M.t_N
    unique_s = M.s_coord 

    # --- 1. Priors & Structural Components ---
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    w_sigma ~ filldist(Exponential(0.5), 3)
    c_sigma ~ filldist(Exponential(1.0), 4)

    # BYM2 Spatial Effect
    s_icar ~ MvNormal(zeros(M.s_N), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # RW2 Categorical Smoothing
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 2. Latent Kronecker Fields (Z, U1) ---
    ls_s_cov ~ Gamma(2, 2); sigma_s_cov ~ Exponential(1.0)
    ls_t_cov ~ Gamma(2, 2); sigma_t_cov ~ Exponential(1.0)

    z_noise ~ filldist(Normal(0, 1), M.y_N)
    z_true = kron_matern_sample(M.s_N, M.t_N, unique_s, unique_t, ls_s_cov, sigma_s_cov, ls_t_cov, sigma_t_cov, z_noise)

    u1_noise ~ filldist(Normal(0, 1), M.y_N)
    u1_true = kron_matern_sample(M.s_N, M.t_N, unique_s, unique_t, ls_s_cov, sigma_s_cov, ls_t_cov, sigma_t_cov, u1_noise)

    # --- 3. Latent Kronecker Field for Y (Main Effect) ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    ls_t_y ~ Gamma(2, 2); sigma_t_y ~ Exponential(1.0)
    y_noise ~ filldist(Normal(0, 1), M.y_N)
    f_st = kron_matern_sample(M.s_N, M.t_N, unique_s, unique_t, ls_s_y, sigma_s_y, ls_t_y, sigma_t_y, y_noise)

    # --- 4. Stochastic Volatility (RFF) ---
    W_sigma ~ filldist(Normal(0, 1), size(M.st_coord, 2), M.M_rff_sigma); b_sigma ~ filldist(Uniform(0, 2pi), M.M_rff_sigma)
    beta_rff_sigma ~ filldist(Normal(0, 1), M.M_rff_sigma)
    y_sigma = exp.(rff_map(M.st_coord, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 5. Likelihood ---
    for i in 1:M.y_N
        a = M.s_idx[i]
        mu = M.log_offset[i] + f_st[i] + s_eta[a] + u1_true[i] + z_true[i]
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma[i]), M.y_obs[i])
    end

    # Prior links for multi-fidelity variables
    M.z_obs ~ MvNormal(z_true, 0.1 * I)
    M.w_obs[:, 1] ~ MvNormal(u1_true, w_sigma[1]^2 * I)
end


@model function example_svgp_matern_nested(M, ::Type{T}=Float64) where {T}
    # SVGP with Matern structure.
    # Integrates nested RFF latent covariates and Kronecker spatiotemporal kernels.
   
    unique_t = collect(1:M.t_N) ./ M.t_N
    unique_s = M.s_coord[1:M.s_N, :]  ## need a better implementation 
    # M.st_D = size(M.s_coord, 2) + 1

    # --- 1. Priors & Structural Components ---
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    w_sigma ~ filldist(Exponential(0.5), 3); c_sigma ~ filldist(Exponential(1.0), 4)

    # BYM2 Spatial Effect
    s_icar ~ MvNormal(zeros(M.s_N), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # RW2 Categorical Smoothing
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 2. Nested Latent Covariates (RFF) ---
    coords_tz = hcat(M.t_coord, M.z_obs)
    W_u1 ~ filldist(Normal(0, 1), size(coords_tz, 2), M.M_rff_u); b_u1 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    u1_true = rff_map(coords_tz, W_u1, b_u1) * filldist(Normal(0, 1), M.M_rff_u)

    # --- 3. Main Spatiotemporal Process (Kronecker) ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    ls_t_y ~ Gamma(2, 2); sigma_t_y ~ Exponential(1.0)
    y_noise ~ filldist(Normal(0, 1), M.y_N)
    f_st = kron_matern_sample(M.s_N, M.t_N, unique_s, unique_t, ls_s_y, sigma_s_y, ls_t_y, sigma_t_y, y_noise)

    # --- 4. Stochastic Volatility (RFF) ---
    W_sigma ~ filldist(Normal(0, 1), size(M.st_coord, 2), M.M_rff_sigma); b_sigma ~ filldist(Uniform(0, 2pi), M.M_rff_sigma)
    beta_rff_sigma ~ filldist(Normal(0, 1), M.M_rff_sigma)
    y_sigma = exp.(rff_map(M.st_coord, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 5. Likelihood ---
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2pi .* M.t_coord ./ M.period) .+ beta_sin .* sin.(2pi .* M.t_coord ./ M.period)

    for i in 1:M.y_N
        a = M.s_idx[i]
        mu = M.log_offset[i] + f_st[i] + seasonal[i] + s_eta[a] + u1_true[i]
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma[i]), M.y_obs[i])
    end

    # multi-fidelity links
    M.w_obs[:, 1] ~ MvNormal(u1_true, w_sigma[1]^2 * I)
end



@model function example_multifidelity_gp_matern(M, ::Type{T}=Float64) where {T}
    # Multi-fidelity GP with standardized interface.
    # Combines high-fidelity Z (Matern), medium-fidelity U (Kron AR1 x Matern), 
    # and standard-fidelity Y with standardized BYM2 and RW2 components.

    # Dimensions and Observations from M
    u1_obs, u2_obs, u3_obs = M.w_obs[:, 1], M.w_obs[:, 2], M.w_obs[:, 3]
    
    Nu = length(u1_obs); 
    Nt_u = M.t_N; 
    Ns_u = Nu ÷ Nt_u
    Nz = length(z_obs)

    # --- 1. Priors ---
    y_sigma ~ Exponential(1.0); s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    st_sigma ~ Exponential(0.5); c_sigma ~ filldist(Exponential(1.0), 4)
    
    # --- 2. High Fidelity: Latent Spatial Z (Matern 3/2) ---
    z_ls ~ Gamma(2, 2); sigma_z_f ~ Exponential(1.0)
    k_z = Matern32Kernel() ∘ ScaleTransform(inv(z_ls))
    K_z = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(M.s_coord_tuple[1:Nz, :])) + M.noise*I
    z_latent ~ MvNormal(zeros(Nz), K_z)
    z_sigma ~ Exponential(0.5)
    z_obs ~ MvNormal(z_latent, z_sigma^2 * I)

    # Interpolation of Z to U/Y locations
    K_z_u = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(M.s_coord_tuple[1:Ns_u, :]), RowVecs(M.s_coord_tuple[1:Nz, :]))
    z_at_u = (K_z_u * (K_z \ z_latent))
    z_at_u_full = repeat(z_at_u, Nt_u)

    # --- 3. Medium Fidelity: Latent Spatiotemporal U (Kron AR1 x Matern) ---
    ls_s_u ~ Gamma(2, 2); sigma_s_u ~ Exponential(1.0)
    rho_u ~ Uniform(-0.99, 0.99); sigma_t_u ~ Exponential(0.5)
    u1_noise ~ filldist(Normal(0, 1), Nu)
    
    # Sample U1 (Hierarchical dependence on Z)
    u1_true = kron_ar1_matern_sample(Ns_u, Nt_u, M.s_coord_tuple[1:Ns_u, :], ls_s_u, sigma_s_u, rho_u, sigma_t_u, u1_noise)
    beta_wz ~ Normal(0, 1)
    u1_obs ~ MvNormal(u1_true .+ beta_wz .* z_at_u_full, 0.1*I)

    # --- 4. Spatial Effect (BYM2) ---
    s_icar ~ MvNormal(zeros(M.s_N), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # --- 5. Categorical Smoothing (RW2) ---
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:M.cov_N]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 6. Likelihood (Standard Fidelity Y) ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    rho_y ~ Uniform(-0.99, 0.99); sigma_t_y ~ Exponential(0.5)
    y_noise ~ filldist(Normal(0, 1), M.y_N)
    f_st_y = kron_ar1_matern_sample(M.s_N, M.t_N, M.s_coord_tuple[1:M.s_N, :], ls_s_y, sigma_s_y, rho_y, sigma_t_y, y_noise)

    beta_y ~ Normal(0, 1)
    for i in 1:M.y_N
        a, t = M.s_idx[i], M.t_idx[i]
        mu = M.log_offset[i] + f_st_y[i] + s_eta[a] + beta_y * z_at_u_full[i]
        for k in 1:M.N_cov; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), M.y_obs[i])
    end
end


@model function example_multifidelity_gp_matern_sv_seasonal(M, ::Type{T}=Float64 ) where {T}
    # Multi-fidelity GP with SV, Seasonal Harmonics, and Standardized Interface.

    # Dimensions and Observations
    y_obs = M.y_obs
    z_obs = M.z_obs
    u1_obs, u2_obs, u3_obs = M.w_obs[:, 1], M.w_obs[:, 2], M.w_obs[:, 3]

    M.y_N = length(y_obs); M.t_N = maximum(M.t_idx); M.s_N = M.y_N ÷ M.t_N
    Nu = length(u1_obs); Nt_u = M.t_N; Ns_u = Nu ÷ Nt_u
    Nz = length(z_obs)

    # --- 1. High Fidelity: Latent Spatial Z (Matern 3/2) ---
    z_ls ~ Gamma(2, 2); sigma_z_f ~ Exponential(1.0)
    k_z = Matern32Kernel() ∘ ScaleTransform(inv(z_ls))
    K_z = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(M.s_coord_tuple[1:Nz, :])) + 1e-3*I
    z_latent ~ MvNormal(zeros(Nz), K_z)
    z_sigma ~ Exponential(0.5)
    z_obs ~ MvNormal(z_latent, z_sigma^2 * I)

    # Interpolation of Z to U and Y
    K_z_u = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(M.s_coord_tuple[1:Ns_u, :]), RowVecs(M.s_coord_tuple[1:Nz, :]))
    z_at_u = (K_z_u * (K_z \ z_latent))
    K_z_y = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(M.s_coord_tuple[1:M.s_N, :]), RowVecs(M.s_coord_tuple[1:Nz, :]))
    z_at_y = (K_z_y * (K_z \ z_latent))

    # --- 2. Medium Fidelity: Latent Spatiotemporal U (Kron AR1 x Matern) ---
    ls_s_u ~ Gamma(2, 2); sigma_s_u ~ Exponential(1.0)
    rho_u ~ Uniform(-0.99, 0.99); sigma_t_u ~ Exponential(0.5)
    u1_noise ~ filldist(Normal(0, 1), Nu)
    unique_pts_u = M.s_coord_tuple[1:Ns_u, :]
    u1_true = kron_ar1_matern_sample(Ns_u, Nt_u, unique_pts_u, ls_s_u, sigma_s_u, rho_u, sigma_t_u, u1_noise)
    
    beta_wz ~ Normal(0, 1)
    u1_obs ~ MvNormal(u1_true .+ beta_wz .* repeat(z_at_u, Nt_u), 0.1*I)

    # Interpolation of U to Y coordinates (Kronecker)
    unique_pts_y = M.s_coord_tuple[1:M.s_N, :]
    k_s_u_interp = Matern32Kernel() ∘ ScaleTransform(inv(ls_s_u))
    K_s_uu = sigma_s_u^2 * kernelmatrix(k_s_u_interp, RowVecs(unique_pts_u)) + 1e-3*I
    K_s_yu = sigma_s_u^2 * kernelmatrix(k_s_u_interp, RowVecs(unique_pts_y), RowVecs(unique_pts_u))
    
    t_u = collect(1:Nt_u); t_y = collect(1:M.t_N)
    K_t_uu = ar1_covariance_matrix(t_u, rho_u, sigma_t_u) + 1e-3*I
    K_t_yu = ar1_cross_covariance_matrix(t_y, t_u, rho_u, sigma_t_u)
    u1_at_y = kron(K_t_yu, K_s_yu) * (kron(K_t_uu, K_s_uu) \ u1_true)

    # --- 3. Standard Components (BYM2 & RW2) ---
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    s_icar ~ MvNormal(zeros(M.s_N), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_eta = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    c_sigma ~ filldist(Exponential(1.0), 4)
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 4. Stochastic Volatility & Seasonality ---
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    t_vec = M.t_idx ./ M.t_N
    seasonal_y = beta_cos .* cos.(2pi .* t_vec) .+ beta_sin .* sin.(2pi .* t_vec)

    W_sigma ~ filldist(Normal(0, 1), size(M.st_coord, 2), M.M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2pi), M.M_rff_sigma)
    sigma_log_var ~ Exponential(1.0); beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M.M_rff_sigma)
    y_sigma_vec = exp.(rff_map(M.st_coord, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 5. Likelihood ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    rho_y ~ Uniform(-0.99, 0.99); sigma_t_y ~ Exponential(0.5); y_noise ~ filldist(Normal(0, 1), M.y_N)
    f_st_y = kron_ar1_matern_sample(M.s_N, M.t_N, unique_pts_y, ls_s_y, sigma_s_y, rho_y, sigma_t_y, y_noise)

    beta_y_covs ~ filldist(Normal(0, 1), 2)
    for i in 1:M.y_N
        a = M.s_idx[i]
        mu = M.log_offset[i] + f_st_y[i] + s_eta[a] + seasonal_y[i] + (u1_at_y[i] * beta_y_covs[1]) + (repeat(z_at_y, M.t_N)[i] * beta_y_covs[2])
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
        Gy_sample = Kio' * ( Li' \ (Li \ Gymu_sample ))  # mean process from inducing locations
        
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




Turing.@model function gaussian_process_basic(; Y, D, cov_fn=exp_cov_fn, nData=length(Y), noise=1e-6 )
    mu ~ Normal(0.0, 1.0); # mean process
    sig2 ~ LogNormal(0, 1) # "nugget" variance
    phi ~ LogNormal(0, 1) # phi is ~ lengthscale along Xstar (range parameter)
    # sigma = cov_fn(D, phi) + sig2 * LinearAlgebra.I(nData) # Realized covariance function + nugget variance
    vcov = cov_fn(D, phi) + sig2 .* LinearAlgebra.I(nData ) .+ noise * LinearAlgebra.I(nData )
    Y ~ MvNormal(mu * ones(nData), Symmetric(vcov) )     # likelihood
end


Turing.@model function gaussian_process_covars(; Y, X, D, cov_fn=exp_cov_fn, nData=length(Y), nF=size(X,2), noise=1e-6 )
    # model matrix for fixed effects (X)
    beta ~ filldist( Normal(0.0, 1.0), nF); 
    sig2 ~ LogNormal(0, 1) # "nugget" variance
    phi ~ LogNormal(0, 1) # phi is ~ lengthscale along Xstar (range parameter)
    # sigma = cov_fn(D, phi) + sig2 * LinearAlgebra.I(nData) # Realized covariance function + nugget variance
    mu = X * beta # mean process
    vcov = cov_fn(D, phi) + sig2 .* LinearAlgebra.I(nData ) + noise * LinearAlgebra.I(nData )   
    Y ~ MvNormal(mu, Symmetric(vcov) )     # likelihood
end



Turing.@model function gaussian_process_ar1( ::Type{T}=Float64; Y, X, D, ar1, cov_fn=exp_cov_fn, 
    nData=length(Y), nF=size(X,2), nT=maximum(ar1)-minimum(ar1)+1, noise=1e-6 ) where {T} 
 
    rho ~ truncated(Normal(0,1), -1, 1)
    ar1_process_error ~ LogNormal(0, 1) 
    var_ar1 =  ar1_process_error^2 / (1 - rho^2)
    vcv = ar1_covariance_local(nT, rho, var_ar1, T)  # -- covariance by time
    ymean_ar1 ~ MvNormal(Symmetric(vcv) );  # -- means by time 
     
    # # mean process model matrix  
    beta ~ filldist( Normal(0.0, 1.0), nF); 
 
    sig2 ~ LogNormal(0, 1) # "nugget" variance
    phi ~ LogNormal(0, 1) # phi is ~ lengthscale along Xstar (range parameter)
    # sigma = cov_fn(D, phi) + sig2 * LinearAlgebra.I(nData) # Realized covariance function + nugget variance
    # vcov = cov_fn(D, phi) + sig2 .* LinearAlgebra.I(nData )
    
    Y ~ MvNormal( X * beta .+ ymean_ar1[ar1[1:nData]], Symmetric(cov_fn(D, phi) .+ sig2 * I(nData) + noise *I(ndata)  ) )     # likelihood
end

 
;;
