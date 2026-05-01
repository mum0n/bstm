 

@model function model_D00_poisson_simple(modinputs, ::Type{T}=Float64; offset=modinputs.offset ) where {T}
    # based on Model v5 but simplified for demonstration (no covars, no spacetime, no weights nor zeroinflation)
    
    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)

    # --- 1. Priors ---
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0); rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5); sigma_rw2 ~ filldist(Exponential(1.0), 4)
    
    # --- 2. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I); s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Temporal Effect (AR1) ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I); Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm
  
    # --- Poisson Likelihood (Log Link) ---
    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        eta = offset[i] + s_eff[a] + f_time[t]  
        mu = exp(eta)
        Turing.@addlogprob! weights[i] * logpdf(Poisson(mu), y[i])
    end
end



@model function model_D01_poisson(modinputs, ::Type{T}=Float64; use_zi=false, offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v5 Optimized: Poisson Spatiotemporal model with optional Zero-Inflation.
    # Uses a log-link to ensure non-negative intensity (mu).

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)

    # --- 1. Priors ---
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0); rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5); sigma_rw2 ~ filldist(Exponential(1.0), 4)
    phi_zi ~ use_zi ? Beta(1, 1) : Dirac(0.0)

    # --- 2. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I); s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Temporal Effect (AR1) ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I); Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # --- 4. Space-Time Interaction ---
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I); st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # --- 5. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Poisson Likelihood (Log Link) ---
    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        eta = offset[i] + s_eff[a] + f_time[t] + st_interaction[a, t]
        for k in 1:4; eta += beta_cov[k][modinputs.cov_indices[i, k]]; end
        mu = exp(eta)
        if use_zi
            Turing.@addlogprob! weights[i] * (y[i] == 0 ? log(phi_zi + (1 - phi_zi) * exp(-mu)) : log(1 - phi_zi) + logpdf(Poisson(mu), y[i]))
        else
            Turing.@addlogprob! weights[i] * logpdf(Poisson(mu), y[i])
        end
    end
end


@model function model_D02_gaussian(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v1 Optimized: Foundational Gaussian Spatiotemporal model.
    # Decomposes the response into spatial (BYM2), temporal (AR1), and interaction effects.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)

    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0)
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0); rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5); sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 2. Spatial Effect (BYM2) ---
    # Combines ICAR (structured) and IID (unstructured) components.
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Temporal Effect (AR1) ---
    # Models temporal autocorrelation using a first-order autoregressive process.
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I); Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # --- 4. Space-Time Interaction (Type IV) ---
    # Captures localized deviations that vary over both space and time.
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I)
    st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # --- 5. Categorical Covariates (RW2 Smoothing) ---
    # Applies second-order random walk smoothing across categorical levels.
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Likelihood ---
    for i in 1:N_obs
        mu = offset[i] + s_eff[modinputs.area_idx[i]] + f_time[modinputs.time_idx[i]] + st_interaction[modinputs.area_idx[i], modinputs.time_idx[i]]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end


@model function model_D03_gaussian_rff(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v2 Optimized: Gaussian model replacing AR1 with Random Fourier Features (RFF).
    # Captures smooth non-linear trends and seasonality alongside spatial clustering.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)

    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0); sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    w_trend ~ MvNormal(zeros(size(modinputs.Z_trend, 2)), I); sigma_trend ~ Exponential(1.0)
    w_seas ~ MvNormal(zeros(size(modinputs.Z_seas, 2)), I); sigma_seas ~ Exponential(1.0)
    sigma_int ~ Exponential(0.5); sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 2. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Temporal Basis (RFF Trend & Seasonality) ---
    # Projects time into a high-dimensional space for non-linear trend/periodic effects.
    f_trend = modinputs.Z_trend * (w_trend .* sigma_trend)
    f_seas = modinputs.Z_seas * (w_seas .* sigma_seas)

    # --- 4. Space-Time Interaction ---
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I)
    st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # --- 5. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Likelihood ---
    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        mu = offset[i] + f_trend[t] + f_seas[t] + s_eff[a] + st_interaction[a, t]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end


@model function model_D04_lognormal(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v3 Optimized: LogNormal Spatiotemporal model for positive skewed data.
    # Employs a log-link to model the median of the distribution.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)

    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0); sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0); rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5); sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 2. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Temporal Effect (AR1) ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I); Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # --- 4. Space-Time Interaction ---
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I)
    st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # --- 5. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. LogNormal Likelihood ---
    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        mu = offset[i] + s_eff[a] + f_time[t] + st_interaction[a, t]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(LogNormal(mu, sigma_y), y[i])
    end
end


@model function model_D05_binomial(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v4 Optimized: Binomial Spatiotemporal model with Logit link.
    # Suitable for binary outcomes or proportion data across areas/time.

    y = modinputs.y
    trials = modinputs.trials
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)

    # --- 1. Priors ---
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0); rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5); sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 2. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I); s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Temporal Effect (AR1) ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I); Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # --- 4. Space-Time Interaction ---
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I); st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # --- 5. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Binomial Likelihood (Logit Link) ---
    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        eta = offset[i] + s_eff[a] + f_time[t] + st_interaction[a, t]
        for k in 1:4; eta += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(BinomialLogit(trials[i], eta), y[i])
    end
end
 

@model function model_D06_negativebinomial(modinputs, ::Type{T}=Float64; use_zi=false, offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v6 Optimized: Negative Binomial Spatiotemporal model.
    # Suitable for over-dispersed counts, with optional zero-inflation.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)

    # --- 1. Priors ---
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0); rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5); sigma_rw2 ~ filldist(Exponential(1.0), 4)
    r_nb ~ Exponential(1.0); phi_zi ~ use_zi ? Beta(1, 1) : Dirac(0.0)

    # --- 2. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I); s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Temporal Effect (AR1) ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I); Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # --- 4. Space-Time Interaction ---
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I); st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # --- 5. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Negative Binomial Likelihood ---
    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        eta = offset[i] + s_eff[a] + f_time[t] + st_interaction[a, t]
        for k in 1:4; eta += beta_cov[k][modinputs.cov_indices[i, k]]; end
        mu = exp(eta); p_nb = r_nb / (r_nb + mu)
        if use_zi
            Turing.@addlogprob! weights[i] * (y[i] == 0 ? log(phi_zi + (1 - phi_zi) * pdf(NegativeBinomial(r_nb, p_nb), 0)) : log(1 - phi_zi) + logpdf(NegativeBinomial(r_nb, p_nb), y[i]))
        else
            Turing.@addlogprob! weights[i] * logpdf(NegativeBinomial(r_nb, p_nb), y[i])
        end
    end
end

 


@model function model_D07_gaussian_rff_cov(modinputs, ::Type{T}=Float64; m_feat=5, offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v9 Optimized: Spatiotemporal model integrating continuous covariates via RFF-Matern Kernels.
    # Merges traditional BYM2 spatial effects with flexible non-linear covariate trends.

    y = modinputs.y
    continuous_covs = modinputs.u_obs
    N_obs, N_areas, N_time, N_covs = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx), size(continuous_covs, 2)

    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0); sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0); rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5); sigma_cov ~ filldist(Exponential(1.0), N_covs); lengthscale_cov ~ filldist(Gamma(2, 1), N_covs)

    # --- 2. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I); s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Temporal Effect (AR1) ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I); Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # --- 4. Space-Time Interaction ---
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I); st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # --- 5. Continuous Covariates (RFF Approximation) ---
    # Approximates a Matern kernel for each continuous covariate using Random Fourier Features.
    W_cov_raw ~ MvNormal(zeros(N_covs * m_feat), I); W_mat = reshape(W_cov_raw, m_feat, N_covs)
    f_cov_total = zeros(T, N_obs)
    for k in 1:N_covs
        Random.seed!(42 + k); Om = randn(m_feat) ./ lengthscale_cov[k]; Ph = rand(m_feat) .* convert(T, 2 * π)
        Z_k = convert(T, sqrt(2/m_feat)) .* cos.(continuous_covs[:, k] * Om' .+ Ph')
        f_cov_total .+= Z_k * (W_mat[:, k] .* sigma_cov[k])
    end

    # --- 6. Gaussian Likelihood ---
    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        mu = offset[i] + s_eff[a] + f_time[t] + st_interaction[a, t] + f_cov_total[i]
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end

 
@model function model_D08_gaussian_fft(modinputs, ::Type{T}=Float64; grid_res=64, pad_factor=2, offset=modinputs.offset) where {T}
    # Model v14 Refined: FFT-Accelerated GMRF
    y = modinputs.y
    N_obs = length(y)
    N_areas = size(modinputs.Q_sp, 1)
    N_time = maximum(modinputs.time_idx)
    
    padded_res = grid_res * pad_factor
    
    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0)
    sigma_sp ~ Exponential(1.0)
    phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0)
    rho_tm ~ Beta(2, 2)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 2. Spectral Spatial Field (FFT) ---
    # We sample in the frequency domain for the ICAR component
    # A padded grid of white noise
    u_spectral_raw ~ MvNormal(zeros(padded_res^2), I)
    u_iid ~ MvNormal(zeros(N_areas), I)

    # Reshape and perform FFT
    u_fft = fft(reshape(convert.(Complex{T}, u_spectral_raw), padded_res, padded_res))
    
    # Apply a simplified Spectral Matern/Laplacian Filter
    # In a full version, we'd use analytic eigenvalues of the Laplacian
    # For this version, we use the scaled spatial precision for the structured part
    u_icar_raw = modinputs.Q_sp \ u_spectral_raw[1:N_areas]
    
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar_raw .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Temporal (AR1) ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # --- 4. Categorical Effects ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 5. Likelihood ---
    for i in 1:N_obs
        mu = offset[i] + s_eff[modinputs.area_idx[i]] + f_time[modinputs.time_idx[i]]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! modinputs.weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end


 
@model function model_D09_gaussian_adaptive_rff(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, M_rff=50, period=12.0) where {T}
    # Model A02 Standardized: Spatiotemporal GP using Adaptive Random Fourier Features with Seasonality.
    # Combines BYM2 spatial effects, learned RFF spatiotemporal features, explicit periodicity, and high-fidelity covariates (z, u).

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)

    # Dimensions for RFF: [lon, lat, time]
    D_st = 3

    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0)
    sigma_f ~ Exponential(1.0)       # RFF amplitude
    sigma_sp ~ Exponential(1.0)      # Spatial component scaling
    phi_sp ~ Beta(1, 1)              # BYM2 mixing
    sigma_int ~ Exponential(0.5)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # Multi-fidelity / Covariate Priors
    beta_z ~ Normal(0, 2)
    beta_u ~ MvNormal(zeros(3), 2.0 * I) # For the 3-dimensional u_obs
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)

    # --- 2. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Adaptive RFF Spatiotemporal Process ---
    coords_st = hcat([p[1] for p in modinputs.pts], [p[2] for p in modinputs.pts], modinputs.time_idx ./ N_time)

    W_matrix ~ filldist(Normal(0, 1), D_st, M_rff)
    b_phases ~ filldist(Uniform(0, 2 * π), M_rff)
    beta_rff ~ filldist(Normal(0, sigma_f), M_rff)

    projection = (coords_st * W_matrix) .+ b_phases'
    Phi = sqrt(2.0 / M_rff) .* cos.(projection)
    f_gp = Phi * beta_rff

    # --- 4. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 5. Likelihood ---
    t_vals = collect(1:N_time)
    seasonal_trend = beta_cos .* cos.(2 * π .* t_vals ./ period) .+ beta_sin .* sin.(2 * π .* t_vals ./ period)

    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        # Link: offset + seasonality + high-fidelity (z) + latent (u) + space-time GP + spatial BYM2 + RW2
        mu = offset[i] + seasonal_trend[t] + beta_z * modinputs.z_obs[i] + dot(beta_u, modinputs.u_obs[i, :]) + f_gp[i] + s_eff[a]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end


@model function model_D10_nested_multifidelity_rff(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, M_rff=50, period=12.0) where {T}
    # Model A03 Standardized: Nested Multi-fidelity Spatiotemporal Model.
    # Models u latent variables as deterministic/probabilistic functions of z and time.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)
    D_st = 3

    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0)
    sigma_u ~ filldist(Exponential(0.5), 3) # Measurement error for u_obs
    sigma_f ~ Exponential(1.0)       # RFF amplitude
    sigma_sp ~ Exponential(1.0)
    phi_sp ~ Beta(1, 1)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # Nested Relationship Coefficients
    beta_u1_z ~ Normal(0, 1); beta_u1_t ~ Normal(0, 1)
    beta_u2_z ~ Normal(0, 1); beta_u2_t ~ Normal(0, 1); beta_u2_u1 ~ Normal(0, 1)
    beta_u3_z ~ Normal(0, 1); beta_u3_t ~ Normal(0, 1); beta_u3_u1 ~ Normal(0, 1)

    # Seasonal and Main Effects
    beta_z ~ Normal(0, 2)
    beta_u_main ~ MvNormal(zeros(3), 2.0 * I)
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)

    # --- 2. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Latent Nested Structure ---
    # Define true latent states based on nested dependencies
    t_norm = modinputs.time_idx ./ N_time
    u1_true = beta_u1_t .* t_norm .+ beta_u1_z .* modinputs.z_obs
    u2_true = beta_u2_t .* t_norm .+ beta_u2_z .* modinputs.z_obs .+ beta_u2_u1 .* u1_true
    u3_true = beta_u3_t .* t_norm .+ beta_u3_z .* modinputs.z_obs .+ beta_u3_u1 .* u1_true
    u_true_mat = hcat(u1_true, u2_true, u3_true)

    # --- 4. Adaptive RFF Spatiotemporal Process ---
    coords_st = hcat([p[1] for p in modinputs.pts], [p[2] for p in modinputs.pts], t_norm)
    W_matrix ~ filldist(Normal(0, 1), D_st, M_rff)
    b_phases ~ filldist(Uniform(0, 2 * π), M_rff)
    beta_rff ~ filldist(Normal(0, sigma_f), M_rff)
    Phi = sqrt(2.0 / M_rff) .* cos.((coords_st * W_matrix) .+ b_phases')
    f_gp = Phi * beta_rff

    # --- 5. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Likelihoods ---
    t_vals = collect(1:N_time)
    seasonal_trend = beta_cos .* cos.(2 * π .* t_vals ./ period) .+ beta_sin .* sin.(2 * π .* t_vals ./ period)

    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        # Response mean includes nested effects, seasonality, and spatial/GP components
        mu = offset[i] + seasonal_trend[t] + beta_z * modinputs.z_obs[i] + dot(beta_u_main, u_true_mat[i, :]) + f_gp[i] + s_eff[a]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        
        # Standard response likelihood
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
        
        # Observation models for u components (linking obs to nested true states)
        Turing.@addlogprob! logpdf(Normal(u1_true[i], sigma_u[1]), modinputs.u_obs[i, 1])
        Turing.@addlogprob! logpdf(Normal(u2_true[i], sigma_u[2]), modinputs.u_obs[i, 2])
        Turing.@addlogprob! logpdf(Normal(u3_true[i], sigma_u[3]), modinputs.u_obs[i, 3])
    end
end



@model function model_D11_nested_time_varying_intercept(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, M_rff=50, period=12.0) where {T}
    # Time-Varying Intercept with Nested Multi-fidelity Spatiotemporal GP.
    # Features a Random Walk (RW1) for the intercept and deterministic nested covariate relationships.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)
    D_st = 3

    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0)
    sigma_u ~ filldist(Exponential(0.5), 3)
    sigma_f ~ Exponential(1.0)
    sigma_sp ~ Exponential(1.0)
    phi_sp ~ Beta(1, 1)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # Time-varying Intercept (RW1) Prior
    sigma_alpha ~ Exponential(0.1)
    alpha_rw = Vector{T}(undef, N_time)
    alpha_rw[1] ~ Normal(0, 1.0)
    for t in 2:N_time
        alpha_rw[t] ~ Normal(alpha_rw[t-1], sigma_alpha)
    end

    # Nested Relationship Coefficients
    beta_u1_z ~ Normal(0, 1); beta_u1_t ~ Normal(0, 1)
    beta_u2_z ~ Normal(0, 1); beta_u2_t ~ Normal(0, 1); beta_u2_u1 ~ Normal(0, 1)
    beta_u3_z ~ Normal(0, 1); beta_u3_t ~ Normal(0, 1); beta_u3_u1 ~ Normal(0, 1)

    # Seasonal and Main Effects
    beta_z ~ Normal(0, 2)
    beta_u_main ~ MvNormal(zeros(3), 2.0 * I)
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)

    # --- 2. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Latent Nested Structure ---
    t_norm = modinputs.time_idx ./ N_time
    u1_true = beta_u1_t .* t_norm .+ beta_u1_z .* modinputs.z_obs
    u2_true = beta_u2_t .* t_norm .+ beta_u2_z .* modinputs.z_obs .+ beta_u2_u1 .* u1_true
    u3_true = beta_u3_t .* t_norm .+ beta_u3_z .* modinputs.z_obs .+ beta_u3_u1 .* u1_true
    u_true_mat = hcat(u1_true, u2_true, u3_true)

    # --- 4. Adaptive RFF Spatiotemporal Process ---
    coords_st = hcat([p[1] for p in modinputs.pts], [p[2] for p in modinputs.pts], t_norm)
    W_matrix ~ filldist(Normal(0, 1), D_st, M_rff)
    b_phases ~ filldist(Uniform(0, 2 * π), M_rff)
    beta_rff ~ filldist(Normal(0, sigma_f), M_rff)
    Phi = sqrt(2.0 / M_rff) .* cos.((coords_st * W_matrix) .+ b_phases')
    f_gp = Phi * beta_rff

    # --- 5. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Likelihoods ---
    t_vals = collect(1:N_time)
    seasonal_trend = beta_cos .* cos.(2 * π .* t_vals ./ period) .+ beta_sin .* sin.(2 * π .* t_vals ./ period)

    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        # Link: offset + alpha_rw[t] + seasonal + high-fidelity (z, u) + GP + spatial + RW2
        mu = offset[i] + alpha_rw[t] + seasonal_trend[t] + beta_z * modinputs.z_obs[i] + dot(beta_u_main, u_true_mat[i, :]) + f_gp[i] + s_eff[a]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end

        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
        Turing.@addlogprob! logpdf(Normal(u1_true[i], sigma_u[1]), modinputs.u_obs[i, 1])
        Turing.@addlogprob! logpdf(Normal(u2_true[i], sigma_u[2]), modinputs.u_obs[i, 2])
        Turing.@addlogprob! logpdf(Normal(u3_true[i], sigma_u[3]), modinputs.u_obs[i, 3])
    end
end


@model function model_D12_stochastic_volatility(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, M_rff=50, M_rff_sigma=20, period=12.0) where {T}
    # Spatiotemporal Stochastic Volatility Model.
    # Uses secondary RFFs to model a latent log-variance process alongside the mean GP.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)
    D_st = 3

    # --- 1. Priors ---
    sigma_u ~ filldist(Exponential(0.5), 3)
    sigma_f ~ Exponential(1.0)       # Mean RFF amplitude
    sigma_sp ~ Exponential(1.0)
    phi_sp ~ Beta(1, 1)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # Stochastic Volatility Priors
    sigma_log_var ~ Exponential(1.0) # Amplitude for the log-variance GP

    # Time-varying Intercept (RW1)
    sigma_alpha ~ Exponential(0.1)
    alpha_rw = Vector{T}(undef, N_time)
    alpha_rw[1] ~ Normal(0, 1.0)
    for t in 2:N_time; alpha_rw[t] ~ Normal(alpha_rw[t-1], sigma_alpha); end

    # Nested Multi-fidelity Coefficients
    beta_u1_z ~ Normal(0, 1); beta_u1_t ~ Normal(0, 1)
    beta_u2_z ~ Normal(0, 1); beta_u2_t ~ Normal(0, 1); beta_u2_u1 ~ Normal(0, 1)
    beta_u3_z ~ Normal(0, 1); beta_u3_t ~ Normal(0, 1); beta_u3_u1 ~ Normal(0, 1)

    # Main Effects and Seasonality
    beta_z ~ Normal(0, 2)
    beta_u_main ~ MvNormal(zeros(3), 2.0 * I)
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)

    # --- 2. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Latent Nested Structure ---
    t_norm = modinputs.time_idx ./ N_time
    u1_true = beta_u1_t .* t_norm .+ beta_u1_z .* modinputs.z_obs
    u2_true = beta_u2_t .* t_norm .+ beta_u2_z .* modinputs.z_obs .+ beta_u2_u1 .* u1_true
    u3_true = beta_u3_t .* t_norm .+ beta_u3_z .* modinputs.z_obs .+ beta_u3_u1 .* u1_true
    u_true_mat = hcat(u1_true, u2_true, u3_true)

    # --- 4. Adaptive Spatiotemporal Processes (Mean & Volatility) ---
    coords_st = hcat([p[1] for p in modinputs.pts], [p[2] for p in modinputs.pts], t_norm)

    # Mean GP (RFF)
    W_mean ~ filldist(Normal(0, 1), D_st, M_rff)
    b_mean ~ filldist(Uniform(0, 2 * π), M_rff)
    beta_rff_mean ~ filldist(Normal(0, sigma_f), M_rff)
    f_mean = (sqrt(2.0 / M_rff) .* cos.((coords_st * W_mean) .+ b_mean')) * beta_rff_mean

    # Volatility GP (RFF for Log-Variance)
    W_vol ~ filldist(Normal(0, 1), D_st, M_rff_sigma)
    b_vol ~ filldist(Uniform(0, 2 * π), M_rff_sigma)
    beta_rff_vol ~ filldist(Normal(0, sigma_log_var), M_rff_sigma)
    log_sigma_y = (sqrt(2.0 / M_rff_sigma) .* cos.((coords_st * W_vol) .+ b_vol')) * beta_rff_vol
    sigma_y_process = exp.(log_sigma_y ./ 2.0)

    # --- 5. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Likelihoods ---
    t_vals = collect(1:N_time)
    seasonal_trend = beta_cos .* cos.(2 * π .* t_vals ./ period) .+ beta_sin .* sin.(2 * π .* t_vals ./ period)

    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        mu = offset[i] + alpha_rw[t] + seasonal_trend[t] + beta_z * modinputs.z_obs[i] + dot(beta_u_main, u_true_mat[i, :]) + f_mean[i] + s_eff[a]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end

        # Likelihood uses the local standard deviation from the volatility process
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y_process[i]), y[i])
        
        # Fidelity observation models
        Turing.@addlogprob! logpdf(Normal(u1_true[i], sigma_u[1]), modinputs.u_obs[i, 1])
        Turing.@addlogprob! logpdf(Normal(u2_true[i], sigma_u[2]), modinputs.u_obs[i, 2])
        Turing.@addlogprob! logpdf(Normal(u3_true[i], sigma_u[3]), modinputs.u_obs[i, 3])
    end
end

@model function model_D13_fitc(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, period=12.0) where {T}
    # --- 1. Data Dimensions & Inputs ---
    y = modinputs.y
    Z_inducing = modinputs.Z_inducing
    N_obs = length(y)
    N_areas = size(modinputs.Q_sp, 1)
    N_time = maximum(modinputs.time_idx)

    # --- 2. Hierarchical Priors (Variances & Mixing) ---
    sigma_y ~ Exponential(1.0) # Observation noise scale
    sigma_f ~ Exponential(1.0) # GP amplitude scale
    ls_st ~ filldist(Gamma(2, 2), 3) # ARD Lengthscales [Lon, Lat, Time]
    sigma_sp ~ Exponential(1.0) # Spatial effect scaling
    phi_sp ~ Beta(1, 1) # BYM2 mixing parameter
    sigma_tm ~ Exponential(1.0) # Temporal effect scaling
    rho_tm ~ Beta(2, 2) # AR1 correlation coefficient
    sigma_int ~ Exponential(0.5) # Space-time interaction scale
    sigma_rw2 ~ filldist(Exponential(1.0), 4) # Categorical smoothing scales
    beta_cos ~ Normal(0, 1) # Seasonal cosine weight
    beta_sin ~ Normal(0, 1) # Seasonal sine weight

    # --- 3. Component 1: BYM2 Spatial Random Effect ---
    u_icar ~ MvNormal(zeros(N_areas), I) # Structured component prior
    Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar) # ICAR log-density projection
    u_iid ~ MvNormal(zeros(N_areas), I) # Unstructured component prior
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid) # Final spatial effect

    # --- 4. Component 2: AR1 Temporal Random Effect ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I) # Construct AR1 precision
    f_tm_raw ~ MvNormal(zeros(N_time), I) # Latent temporal prior
    Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw) # AR1 log-density projection
    f_time = f_tm_raw .* sigma_tm # Scaled temporal trend

    # --- 5. Component 3: Harmonic Seasonal Effect ---
    t_vec = Float64.(modinputs.time_idx) # Convert time indices to floats
    seasonal = beta_cos .* cos.(2 * pi .* t_vec ./ period) .+ beta_sin .* sin.(2 * pi .* t_vec ./ period) # Harmonic wave

    # --- 6. Component 4: Space-Time Interaction (GMRF) ---
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I) # Latent interaction prior
    st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time) # Reshaped interaction matrix

    # --- 7. Component 5: FITC Sparse GP Approximation ---
    t_norm = modinputs.time_idx ./ N_time # Normalize time for GP stability
    coords_st = hcat([p[1] for p in modinputs.pts], [p[2] for p in modinputs.pts], t_norm) # Spatiotemporal features
    k_st = (sigma_f^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(ls_st))) # Matern/SqExp Kernel definition
    K_ZZ = kernelmatrix(k_st, RowVecs(Z_inducing)) + 1e-6 * I # Inducing point covariance
    K_XZ = kernelmatrix(k_st, RowVecs(coords_st), RowVecs(Z_inducing)) # Cross-covariance matrix
    u_inducing ~ MvNormal(zeros(size(Z_inducing, 1)), K_ZZ) # Sample values at inducing locations
    f_gp = K_XZ * (K_ZZ \ u_inducing) # Project to all data points via FITC logic

    # --- 8. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4] # Initialize covariate vectors
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I) # Standard normal prior for levels
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k]) # Apply RW2 smoothing penalty
    end

    # --- 9. Final Linear Predictor & Likelihood ---
    for i in 1:N_obs
        idx_a = modinputs.area_idx[i] # Current area index
        idx_t = modinputs.time_idx[i] # Current time index
        mu = offset[i] + f_gp[i] + s_eff[idx_a] + f_time[idx_t] + seasonal[i] + st_interaction[idx_a, idx_t] # Sum all components
        # Add categorical effects
        for k in 1:4 
            mu += beta_cov[k][modinputs.cov_indices[i, k]] 
        end 
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i]) # Weighted likelihood evaluation
    end
end
 
@model function model_D14_fitc_nonlinear_nested_rff(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, M_inducing_val=15, M_rff_u=30, M_rff_sigma=20, period=12.0) where {T}
    # FITC Sparse GP with Non-linear Nested Latent Covariates (RFF).

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)
    D_st = 3

    # --- 1. Priors ---
    sigma_u ~ filldist(Exponential(0.5), 3)
    sigma_f ~ Exponential(1.0); ls_st ~ filldist(Gamma(2, 2), D_st)
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)
    sigma_log_var ~ Exponential(1.0)

    # RW1 Trend
    sigma_alpha ~ Exponential(0.1)
    alpha_rw = Vector{T}(undef, N_time)
    alpha_rw[1] ~ Normal(0, 1.0)
    for t in 2:N_time; alpha_rw[t] ~ Normal(alpha_rw[t-1], sigma_alpha); end

    # --- 2. Non-linear Nested Structure (RFF) ---
    t_norm = modinputs.time_idx ./ N_time
    coords_tz = hcat(t_norm, modinputs.z_obs)
    
    # U1 = f1(t, z)
    W_u1 ~ filldist(Normal(0, 1), 2, M_rff_u); b_u1 ~ filldist(Uniform(0, 2 * π), M_rff_u)
    sigma_f_u1 ~ Exponential(1.0); beta_rff_u1 ~ filldist(Normal(0, sigma_f_u1), M_rff_u)
    u1_true = (sqrt(2/M_rff_u) .* cos.(coords_tz * W_u1 .+ b_u1')) * beta_rff_u1

    # U2 = f2(t, z, u1)
    coords_tz_u1 = hcat(t_norm, modinputs.z_obs, u1_true)
    W_u2 ~ filldist(Normal(0, 1), 3, M_rff_u); b_u2 ~ filldist(Uniform(0, 2 * π), M_rff_u)
    sigma_f_u2 ~ Exponential(1.0); beta_rff_u2 ~ filldist(Normal(0, sigma_f_u2), M_rff_u)
    u2_true = (sqrt(2/M_rff_u) .* cos.(coords_tz_u1 * W_u2 .+ b_u2')) * beta_rff_u2

    # U3 = f3(t, z, u1)
    W_u3 ~ filldist(Normal(0, 1), 3, M_rff_u); b_u3 ~ filldist(Uniform(0, 2 * π), M_rff_u)
    sigma_f_u3 ~ Exponential(1.0); beta_rff_u3 ~ filldist(Normal(0, sigma_f_u3), M_rff_u)
    u3_true = (sqrt(2/M_rff_u) .* cos.(coords_tz_u1 * W_u3 .+ b_u3')) * beta_rff_u3
    u_true_mat = hcat(u1_true, u2_true, u3_true)

    # --- 3. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 4. FITC Sparse GP ---
    coords_st = hcat([p[1] for p in modinputs.pts], [p[2] for p in modinputs.pts], t_norm)
    Z_inducing = Matrix{T}(undef, M_inducing_val, D_st)
    for j in 1:D_st; Z_inducing[:, j] ~ filldist(Normal(mean(coords_st[:,j]), 2*std(coords_st[:,j])), M_inducing_val); end

    k_st = (sigma_f^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(ls_st)))
    K_ZZ = kernelmatrix(k_st, RowVecs(Z_inducing)) + 1e-6 * I
    K_XZ = kernelmatrix(k_st, RowVecs(coords_st), RowVecs(Z_inducing))
    K_XX_diag = diag(kernelmatrix(k_st, RowVecs(coords_st)))

    u_inducing ~ MvNormal(zeros(M_inducing_val), K_ZZ)
    f_mean = K_XZ * (K_ZZ \ u_inducing)
    cov_f_diag = K_XX_diag - diag(K_XZ * (K_ZZ \ K_XZ'))
    f_gp ~ MvNormal(f_mean, Diagonal(max.(1e-6, cov_f_diag)))

    # --- 5. Volatility & Categorical Smoothing ---
    W_vol ~ filldist(Normal(0, 1), D_st, M_rff_sigma); b_vol ~ filldist(Uniform(0, 2 * π), M_rff_sigma)
    beta_rff_vol ~ filldist(Normal(0, sigma_log_var), M_rff_sigma)
    log_sigma_y = (sqrt(2/M_rff_sigma) .* cos.(coords_st * W_vol .+ b_vol')) * beta_rff_vol
    sigma_y = exp.(log_sigma_y ./ 2.0)

    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Likelihoods ---
    beta_z ~ Normal(0, 2); beta_u_main ~ MvNormal(zeros(3), 2.0 * I)
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * π .* t_norm * (N_time/period)) .+ beta_sin .* sin.(2 * π .* t_norm * (N_time/period))

    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        mu = offset[i] + alpha_rw[t] + seasonal[i] + beta_z * modinputs.z_obs[i] + dot(beta_u_main, u_true_mat[i, :]) + f_gp[i] + s_eff[a]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y[i]), y[i])
        for k in 1:3; Turing.@addlogprob! logpdf(Normal(u_true_mat[i, k], sigma_u[k]), modinputs.u_obs[i, k]); end
    end
end


@model function model_D15_gptime_fitc(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, M_inducing_val=15, M_rff_u=30, M_rff_sigma=20, period=12.0) where {T}
    # FITC Sparse GP with continuous GP temporal trend and Non-linear Nested Covariates.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)
    D_st = 3

    # --- 1. Priors ---
    sigma_u ~ filldist(Exponential(0.5), 3)
    sigma_f ~ Exponential(1.0); ls_st ~ filldist(Gamma(2, 2), D_st)
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)
    sigma_log_var ~ Exponential(1.0)

    # --- 2. GP Temporal Trend ---
    ls_trend ~ Gamma(2, 2)
    sigma_trend ~ Exponential(0.5)
    k_trend = (sigma_trend^2) * (SqExponentialKernel() ∘ ScaleTransform(inv(ls_trend)))
    t_unique = collect(1:N_time) ./ N_time
    # Latent trend at unique time points
    alpha_gp ~ MvNormal(zeros(N_time), kernelmatrix(k_trend, t_unique) + 1e-6*I)

    # --- 3. Non-linear Nested Structure (RFF) ---
    t_norm = modinputs.time_idx ./ N_time
    coords_tz = hcat(t_norm, modinputs.z_obs)
    
    W_u1 ~ filldist(Normal(0, 1), 2, M_rff_u); b_u1 ~ filldist(Uniform(0, 2 * π), M_rff_u)
    sigma_f_u1 ~ Exponential(1.0); beta_rff_u1 ~ filldist(Normal(0, sigma_f_u1), M_rff_u)
    u1_true = (sqrt(2/M_rff_u) .* cos.(coords_tz * W_u1 .+ b_u1')) * beta_rff_u1

    coords_tz_u1 = hcat(t_norm, modinputs.z_obs, u1_true)
    W_u2 ~ filldist(Normal(0, 1), 3, M_rff_u); b_u2 ~ filldist(Uniform(0, 2 * π), M_rff_u)
    sigma_f_u2 ~ Exponential(1.0); beta_rff_u2 ~ filldist(Normal(0, sigma_f_u2), M_rff_u)
    u2_true = (sqrt(2/M_rff_u) .* cos.(coords_tz_u1 * W_u2 .+ b_u2')) * beta_rff_u2

    W_u3 ~ filldist(Normal(0, 1), 3, M_rff_u); b_u3 ~ filldist(Uniform(0, 2 * π), M_rff_u)
    sigma_f_u3 ~ Exponential(1.0); beta_rff_u3 ~ filldist(Normal(0, sigma_f_u3), M_rff_u)
    u3_true = (sqrt(2/M_rff_u) .* cos.(coords_tz_u1 * W_u3 .+ b_u3')) * beta_rff_u3
    u_true_mat = hcat(u1_true, u2_true, u3_true)

    # --- 4. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 5. FITC Sparse GP Mean ---
    coords_st = hcat([p[1] for p in modinputs.pts], [p[2] for p in modinputs.pts], t_norm)
    Z_inducing = Matrix{T}(undef, M_inducing_val, D_st)
    for j in 1:D_st; Z_inducing[:, j] ~ filldist(Normal(mean(coords_st[:,j]), 2*std(coords_st[:,j])), M_inducing_val); end

    k_st = (sigma_f^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(ls_st)))
    K_ZZ = kernelmatrix(k_st, RowVecs(Z_inducing)) + 1e-6 * I
    K_XZ = kernelmatrix(k_st, RowVecs(coords_st), RowVecs(Z_inducing))
    K_XX_diag = diag(kernelmatrix(k_st, RowVecs(coords_st)))

    u_inducing ~ MvNormal(zeros(M_inducing_val), K_ZZ)
    f_mean = K_XZ * (K_ZZ \ u_inducing)
    cov_f_diag = K_XX_diag - diag(K_XZ * (K_ZZ \ K_XZ'))
    f_gp ~ MvNormal(f_mean, Diagonal(max.(1e-6, cov_f_diag)))

    # --- 6. Volatility & Categorical ---
    W_vol ~ filldist(Normal(0, 1), D_st, M_rff_sigma); b_vol ~ filldist(Uniform(0, 2 * π), M_rff_sigma)
    beta_rff_vol ~ filldist(Normal(0, sigma_log_var), M_rff_sigma)
    log_sigma_y = (sqrt(2/M_rff_sigma) .* cos.(coords_st * W_vol .+ b_vol')) * beta_rff_vol
    sigma_y = exp.(log_sigma_y ./ 2.0)

    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 7. Likelihoods ---
    beta_z ~ Normal(0, 2); beta_u_main ~ MvNormal(zeros(3), 2.0 * I)
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * π .* t_norm * (N_time/period)) .+ beta_sin .* sin.(2 * π .* t_norm * (N_time/period))

    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        mu = offset[i] + alpha_gp[t] + seasonal[i] + beta_z * modinputs.z_obs[i] + dot(beta_u_main, u_true_mat[i, :]) + f_gp[i] + s_eff[a]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y[i]), y[i])
        for k in 1:3; Turing.@addlogprob! logpdf(Normal(u_true_mat[i, k], sigma_u[k]), modinputs.u_obs[i, k]); end
    end
end




@model function model_C01_gaussian_dense_gp(modinputs, ::Type{T}=Float64; trials=ones(Int, length(modinputs.y)), offset=modinputs.offset, weights=modinputs.weights, period=12.0) where {T}
    # Model A00 Optimized: Dense Spatiotemporal Gaussian Process with Seasonality.
    # Uses a separable kernel and seasonal harmonics.

    y = modinputs.y
    N_obs = length(y)

    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0)
    sigma_f ~ Exponential(1.0)
    ls_s ~ Gamma(2, 2)
    ls_t ~ Gamma(2, 2)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # Seasonal priors
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)

    # --- 2. Coordinate Extraction ---
    xs = [p[1] for p in modinputs.pts]
    ys = [p[2] for p in modinputs.pts]
    ts = Float64.(modinputs.time_idx)

    # --- 3. Kernel Matrix Construction ---
    k_s = SqExponentialKernel() ∘ ScaleTransform(inv(ls_s))
    k_t = SqExponentialKernel() ∘ ScaleTransform(inv(ls_t))
    coords_s = RowVecs(hcat(xs, ys))
    K = (sigma_f^2) .* kernelmatrix(k_s, coords_s) .* kernelmatrix(k_t, ts)
    
    # --- 4. Latent Process ---
    f_latent ~ MvNormal(zeros(N_obs), K + 1e-6*I)

    # --- 5. Categorical Covariates (RW2 Smoothing) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Likelihood ---
    for i in 1:N_obs
        # Add Seasonality component
        seasonal = beta_cos * cos(2 * π * ts[i] / period) + beta_sin * sin(2 * π * ts[i] / period)
        
        mu = offset[i] + f_latent[i] + seasonal
        for k in 1:4
            mu += beta_cov[k][modinputs.cov_indices[i, k]]
        end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end
 


@model function model_C02_gaussian_deep_gp(modinputs, ::Type{T}=Float64; m1=10, m2=5, offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v8 Optimized: Refined 2-Layer Deep GP Spatiotemporal model for Gaussian data.
    # Combines deep non-linear interaction with robust modinputsuted covariate smoothing.

    y = modinputs.y
    N_obs = length(y); sigma_y ~ Exponential(1.0); sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 1. Deep GP Priors ---
    l1 ~ Gamma(2, 1); w1 ~ MvNormal(zeros(m1), I)
    l2 ~ Gamma(2, 1); w2 ~ MvNormal(zeros(m2), I)

    # --- 2. Deep GP Layer 1 (Input Transformation) ---
    X = hcat([p[1] for p in modinputs.pts], [p[2] for p in modinputs.pts], Float64.(modinputs.time_idx))
    Random.seed!(42); Om1 = randn(m1, 3) ./ l1; Ph1 = rand(m1) .* convert(T, 2 * π)
    h1 = (convert(T, sqrt(2/m1)) .* cos.(X * Om1' .+ Ph1')) * w1

    # --- 3. Deep GP Layer 2 (Latent Mean Field) ---
    Random.seed!(43); Om2 = randn(m2, 1) ./ l2; Ph2 = rand(m2) .* convert(T, 2 * π)
    eta_gp = (convert(T, sqrt(2/m2)) .* cos.(reshape(h1, :, 1) * Om2' .+ Ph2')) * w2

    # --- 4. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 5. Gaussian Likelihood ---
    for i in 1:N_obs
        mu = offset[i] + eta_gp[i]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end



@model function model_C03_binomial_deep_gp(modinputs, ::Type{T}=Float64; m1=10, m2=5, offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v7 Optimized: Deep Gaussian Process (GP) Spatiotemporal model with Binomial likelihood.
    # Uses Random Fourier Features (RFF) to approximate non-stationary spatio-temporal interactions.

    y = modinputs.y
    trials = modinputs.trials
    N_obs = length(y); sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 1. Deep GP Priors ---
    lengthscale1 ~ Gamma(2, 1); w1 ~ MvNormal(zeros(m1), I)
    lengthscale2 ~ Gamma(2, 1); w2 ~ MvNormal(zeros(m2), I)

    # --- 2. Feature Matrix Construction ---
    X = hcat([p[1] for p in modinputs.pts], [p[2] for p in modinputs.pts], Float64.(modinputs.time_idx))

    # --- 3. Layer 1 (Hidden Warp) ---
    Random.seed!(42); Om1 = randn(m1, 3) ./ lengthscale1; Ph1 = rand(m1) .* convert(T, 2 * π)
    h1 = (convert(T, sqrt(2/m1)) .* cos.(X * Om1' .+ Ph1')) * w1

    # --- 4. Layer 2 (Latent Response) ---
    Random.seed!(43); Om2 = randn(m2, 1) ./ lengthscale2; Ph2 = rand(m2) .* convert(T, 2 * π)
    eta_gp = (convert(T, sqrt(2/m2)) .* cos.(reshape(h1, :, 1) * Om2' .+ Ph2')) * w2

    # --- 5. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Likelihood (Binomial Logit Link) ---
    for i in 1:N_obs
        eta = offset[i] + eta_gp[i]
        for k in 1:4; eta += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(BinomialLogit(trials[i], eta), y[i])
    end
end
 


@model function model_C04_gaussian_deep_gp_3layer(modinputs, ::Type{T}=Float64; m1=10, m2=5, m3=3, offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v10 Optimized: 3-Layer Deep GP with Gaussian likelihood.
    # Hierarchical composition of GPs for capturing extremely complex spatio-temporal dynamics.

    y = modinputs.y
    N_obs = length(y); sigma_y ~ Exponential(1.0); sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 1. 3-Layer Deep GP Priors ---
    l1 ~ Gamma(2, 1); w1 ~ MvNormal(zeros(m1), I)
    l2 ~ Gamma(2, 1); w2 ~ MvNormal(zeros(m2), I)
    l3 ~ Gamma(2, 1); w3 ~ MvNormal(zeros(m3), I)

    # --- 2. Layer 1 (Input Transformation) ---
    X = hcat([p[1] for p in modinputs.pts], [p[2] for p in modinputs.pts], Float64.(modinputs.time_idx))
    Random.seed!(42); Om1 = randn(m1, 3) ./ l1; Ph1 = rand(m1) .* convert(T, 2 * π)
    h1 = (convert(T, sqrt(2/m1)) .* cos.(X * Om1' .+ Ph1')) * w1

    # --- 3. Layer 2 (Non-linear Manifold Transformation) ---
    Random.seed!(43); Om2 = randn(m2, 1) ./ l2; Ph2 = rand(m2) .* convert(T, 2 * π)
    h2 = (convert(T, sqrt(2/m2)) .* cos.(reshape(h1, :, 1) * Om2' .+ Ph2')) * w2

    # --- 4. Layer 3 (Response Surface) ---
    Random.seed!(44); Om3 = randn(m3, 1) ./ l3; Ph3 = rand(m3) .* convert(T, 2 * π)
    eta_gp = (convert(T, sqrt(2/m3)) .* cos.(reshape(h2, :, 1) * Om3' .+ Ph3')) * w3

    # --- 5. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Gaussian Likelihood ---
    for i in 1:N_obs
        mu = offset[i] + eta_gp[i]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end



@model function model_C05_gaussian_non_separable_rff(modinputs, ::Type{T}=Float64; m_joint=25, offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v11: Non-Separable Spatiotemporal RFF model.
    # Instead of separate spatial and temporal components, this model projects
    # the joint [X, Y, Time] vector into a shared feature space.

    y = modinputs.y
    N_obs = length(y)
    
    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0)
    sigma_joint ~ Exponential(1.0)
    # Lengthscales for X, Y, and Time dimensions within the joint kernel
    l_joint ~ filldist(Gamma(2, 1), 3) 
    w_joint ~ MvNormal(zeros(m_joint), I)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 2. Feature Matrix Construction ---
    # X_joint: [normalized_x, normalized_y, normalized_time]
    # We use modinputs.pts and normalize them for numerical stability
    xs = [p[1] for p in modinputs.pts]
    ys = [p[2] for p in modinputs.pts]
    ts = Float64.(modinputs.time_idx)
    
    # Normalize inputs to [0, 1] range
    X_joint = hcat(
        (xs .- minimum(xs)) ./ (maximum(xs) - minimum(xs) + 1e-6),
        (ys .- minimum(ys)) ./ (maximum(ys) - minimum(ys) + 1e-6),
        (ts .- minimum(ts)) ./ (maximum(ts) - minimum(ts) + 1e-6)
    )

    # --- 3. Joint RFF Projection ---
    # This creates the non-separable interaction
    Random.seed!(42)
    # Sample frequencies scaled by dimension-specific lengthscales
    Om = randn(m_joint, 3) .* (1.0 ./ l_joint')
    Ph = rand(m_joint) .* convert(T, 2π)
    
    Z_joint = convert(T, sqrt(2/m_joint)) .* cos.(X_joint * Om' .+ Ph')
    eta_joint = Z_joint * (w_joint .* sigma_joint)

    # --- 4. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 5. Likelihood ---
    for i in 1:N_obs
        mu = offset[i] + eta_joint[i]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end


@model function model_C06_gaussian_spde_rff(modinputs, ::Type{T}=Float64; m_spatial=50, offset=modinputs.offset) where {T}
    # Model v12: SPDE-style continuous spatial field using spectral RFF approximation.
    y = modinputs.y
    N_obs = length(y)
    N_time = maximum(modinputs.time_idx)

    # --- 1. SPDE / Matern Priors ---
    sigma_y ~ Exponential(1.0)
    sigma_sp ~ Exponential(1.0)
    kappa_sp ~ Gamma(2, 1)  # Range parameter (1/lengthscale)
    w_sp ~ MvNormal(zeros(m_spatial), I)

    # --- 2. Temporal (AR1) & Smoothing Priors ---
    sigma_tm ~ Exponential(1.0)
    rho_tm ~ Beta(2, 2)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 3. Continuous Spatial Basis (Spectral SPDE) ---
    # Normalize points for spectral projection
    xs = [p[1] for p in modinputs.pts]
    ys = [p[2] for p in modinputs.pts]
    coords = hcat(
        (xs .- mean(xs)) ./ std(xs),
        (ys .- mean(ys)) ./ std(ys)
    )

    Random.seed!(42)
    # Frequencies sampled for a Matern kernel approximation
    # Note: For Matern nu=1.5, we sample from a Student-t distribution spectral density
    Om = randn(m_spatial, 2) .* kappa_sp
    Ph = rand(m_spatial) .* convert(T, 2π)
    
    Z_sp = convert(T, sqrt(2/m_spatial)) .* cos.(coords * Om' .+ Ph')
    s_eff = Z_sp * (w_sp .* sigma_sp)

    # --- 4. Temporal Effect (AR1) ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # --- 5. Categorical & Likelihood ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    for i in 1:N_obs
        mu = offset[i] + s_eff[i] + f_time[modinputs.time_idx[i]]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! modinputs.weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end




@model function model_C07_gaussian_nonstationary_warping(modinputs, ::Type{T}=Float64; m_warp=10, m_spatial=40, offset=modinputs.offset) where {T}
    y = modinputs.y
    N_obs = length(y)
    N_time = maximum(modinputs.time_idx)

    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0)
    sigma_sp ~ Exponential(1.0)
    l_warp ~ Gamma(2, 1)    # Smoothness of the warping manifold
    l_spatial ~ Gamma(2, 1) # Smoothness of the stationary field in warped space
    
    w_warp ~ MvNormal(zeros(m_warp), I)
    w_sp ~ MvNormal(zeros(m_spatial), I)
    
    sigma_tm ~ Exponential(1.0)
    rho_tm ~ Beta(2, 2)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 2. Input Preprocessing ---
    xs = [p[1] for p in modinputs.pts]
    ys = [p[2] for p in modinputs.pts]
    coords = hcat((xs .- mean(xs)) ./ std(xs), (ys .- mean(ys)) ./ std(ys))

    # --- 3. Warping Layer (Non-Stationarity) ---
    # This layer 'warps' the 2D coordinates into a latent space
    Random.seed!(44)
    Om_w = randn(m_warp, 2) ./ l_warp
    Ph_w = rand(m_warp) .* convert(T, 2π)
    
    # Warped coordinates: g(s)
    warped_coords = (convert(T, sqrt(2/m_warp)) .* cos.(coords * Om_w' .+ Ph_w')) * w_warp

    # --- 4. Spatial Field on Warped Manifold ---
    # We apply a stationary kernel to the warped output
    Random.seed!(45)
    Om_s = randn(m_spatial, 1) ./ l_spatial
    Ph_s = rand(m_spatial) .* convert(T, 2π)
    
    Z_sp = convert(T, sqrt(2/m_spatial)) .* cos.(reshape(warped_coords, :, 1) * Om_s' .+ Ph_s')
    s_eff = Z_sp * (w_sp .* sigma_sp)

    # --- 5. Temporal & Categorical Components ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Likelihood ---
    for i in 1:N_obs
        mu = offset[i] + s_eff[i] + f_time[modinputs.time_idx[i]]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! modinputs.weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end




@model function model_C08_gaussian_refined_mosaic(modinputs, ::Type{T}=Float64; n_mosaics=5, m_rff=20, offset=modinputs.offset) where {T}
    y = modinputs.y
    N_obs = length(y)
    
    # --- 1. Global & Hierarchical Priors ---
    sigma_rw2 ~ filldist(Exponential(1.0), 4)
    mu_global ~ Normal(0, 1)
    sigma_mu_local ~ Exponential(1.0)

    # Local Parameters per Mosaic
    mu_local ~ filldist(Normal(mu_global, sigma_mu_local), n_mosaics)
    l_local ~ filldist(Gamma(2, 1), n_mosaics)
    sigma_local ~ filldist(Exponential(1.0), n_mosaics)
    sigma_y_local ~ filldist(Exponential(1.0), n_mosaics) # Localized noise scale

    # Weights for each mosaic's RFF field
    w_local = [Vector{T}(undef, m_rff) for _ in 1:n_mosaics]
    for m in 1:n_mosaics; w_local[m] ~ MvNormal(zeros(m_rff), I); end

    # Categorical Covariates (Shared)
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 2. Spatial Indexing & Soft Boundary Weights ---
    coords = hcat([p[1] for p in modinputs.pts], [p[2] for p in modinputs.pts])
    R = kmeans(coords', n_mosaics)
    centroids = R.centers # 2 x n_mosaics

    # Pre-sample RFF frequencies
    Random.seed!(42)
    Om_m = [randn(m_rff, 2) for _ in 1:n_mosaics]
    Ph_m = [rand(m_rff) for _ in 1:n_mosaics]

    # --- 3. Likelihood with Soft Integration ---
    for i in 1:N_obs
        pt = [coords[i,1], coords[i,2]]
        
        # Calculate Softmax weights based on distance to centroids for smooth stitching
        dists = [sum((pt .- centroids[:, m]).^2) for m in 1:n_mosaics]
        weights_stitching = exp.(-dists) ./ sum(exp.(-dists))
        
        eta_spatial_combined = zero(T)
        sigma_y_combined = zero(T)
        
        for m in 1:n_mosaics
            # Local RFF Field Calculation
            z_proj = sqrt(2/m_rff) .* cos.( (Om_m[m] * pt ./ l_local[m]) .+ (Ph_m[m] .* 2π) )
            local_field = mu_local[m] + dot(z_proj, w_local[m] .* sigma_local[m])
            
            # Blend local field and local noise
            eta_spatial_combined += weights_stitching[m] * local_field
            sigma_y_combined += weights_stitching[m] * sigma_y_local[m]
        end

        mu = offset[i] + eta_spatial_combined
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        
        Turing.@addlogprob! modinputs.weights[i] * logpdf(Normal(mu, sigma_y_combined + 1e-6), y[i])
    end
end


@model function model_C09_gaussian_integrated_mosaic(modinputs, ::Type{T}=Float64; n_mosaics=4, m_rff=20, offset=modinputs.offset) where {T}
    y = modinputs.y
    N_obs = length(y)

    # --- 1. Global Hierarchical Priors ---
    sigma_rw2 ~ filldist(Exponential(1.0), 4)
    mu_global ~ Normal(0, 1)
    sigma_mu_local ~ Exponential(0.5)

    # Shared Categorical Effects
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 2. Local Mosaic Hyperparameters ---
    mu_local ~ filldist(Normal(mu_global, sigma_mu_local), n_mosaics)

    # Refactored: Use arraydist for joint lengthscales instead of a loop
    l_joint ~ arraydist([filldist(Gamma(2, 1), 3) for _ in 1:n_mosaics])

    sigma_local ~ filldist(Exponential(1.0), n_mosaics)
    sigma_y_local ~ filldist(Exponential(1.0), n_mosaics)

    # Local Weights for Non-Separable RFF
    w_local = [Vector{T}(undef, m_rff) for _ in 1:n_mosaics]
    for m in 1:n_mosaics; w_local[m] ~ MvNormal(zeros(m_rff), I); end

    # --- 3. Geometric Indexing ---
    xs = [p[1] for p in modinputs.pts]
    ys = [p[2] for p in modinputs.pts]
    ts = Float64.(modinputs.time_idx)

    # Normalize [X, Y, T] to [0, 1] range for RFF stability
    X_joint = hcat(
        (xs .- minimum(xs)) ./ (maximum(xs) - minimum(xs) + 1e-6),
        (ys .- minimum(ys)) ./ (maximum(ys) - minimum(ys) + 1e-6),
        (ts .- minimum(ts)) ./ (maximum(ts) - minimum(ts) + 1e-6)
    )

    # Static centroids for stitching (calculated once)
    coords_2d = X_joint[:, 1:2]
    R = kmeans(coords_2d', n_mosaics)
    centroids = R.centers

    # Fixed RFF Frequencies
    Random.seed!(42)
    Om_base = [randn(m_rff, 3) for _ in 1:n_mosaics]
    Ph_base = [rand(m_rff) for _ in 1:n_mosaics]

    # --- 4. Predictive Synthesis ---
    for i in 1:N_obs
        pt_3d = X_joint[i, :]
        pt_2d = pt_3d[1:2]

        # Soft Boundary Weights (Softmax of distance to centroids)
        dists = [sum((pt_2d .- centroids[:, m]).^2) for m in 1:n_mosaics]
        weights_st = exp.(-dists) ./ sum(exp.(-dists))

        eta_spatial_time = zero(T)
        sigma_y_total = zero(T)

        for m in 1:n_mosaics
            # Scale base frequencies by local lengthscales [Lx, Ly, Lt]
            # l_joint is now a matrix where each column corresponds to a mosaic
            Om = Om_base[m] .* (1.0 ./ (l_joint[:, m] .+ 1e-6)')

            # Local Non-Separable Field
            z_proj = sqrt(2/m_rff) * cos.( (Om * pt_3d) .+ (Ph_base[m] .* 2π) )
            local_field = mu_local[m] + dot(z_proj, w_local[m] .* sigma_local[m])

            eta_spatial_time += weights_st[m] * local_field
            sigma_y_total += weights_st[m] * sigma_y_local[m]
        end

        mu = offset[i] + eta_spatial_time
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end

        Turing.@addlogprob! modinputs.weights[i] * logpdf(Normal(mu, sigma_y_total + 1e-6), y[i])
    end
end



@model function model_C10_gaussian_fitcxed_fitc_grmf(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, period=12.0) where {T}
    # FITC Sparse GP using fixed inducing point priors for performance optimization.
    # Architecture: [1] BYM2 Spatial, [2] AR1 Temporal, [3] Harmonic Seasonal, [4] GMRF Interaction, [5] RW2 Smoothing.

    y = modinputs.y
    Z_inducing = modinputs.Z_inducing
    N_obs = length(y)
    N_areas = size(modinputs.Q_sp, 1)
    N_time = maximum(modinputs.time_idx)

    # Priors
    sigma_y ~ Exponential(1.0)
    sigma_f ~ Exponential(1.0)
    ls_st ~ filldist(Gamma(2, 2), 3)
    sigma_sp ~ Exponential(1.0)
    phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0)
    rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)

    # Component 1: BYM2
    u_icar ~ MvNormal(zeros(N_areas), I)
    Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # Component 2: AR1
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # Component 3: Seasonal
    t_vec = Float64.(modinputs.time_idx)
    seasonal = beta_cos .* cos.(2 * pi .* t_vec ./ period) .+ beta_sin .* sin.(2 * pi .* t_vec ./ period)

    # Component 4: GMRF Interaction
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I)
    st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # Component 5: Fixed Inducing FITC (Optimized GP Projection)
    t_norm = modinputs.time_idx ./ N_time
    coords_st = hcat([p[1] for p in modinputs.pts], [p[2] for p in modinputs.pts], t_norm)
    k_st = (sigma_f^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(ls_st)))
    K_XZ = kernelmatrix(k_st, RowVecs(coords_st), RowVecs(Z_inducing))
    u_inducing ~ MvNormal(zeros(size(Z_inducing, 1)), I) # Assumes unit variance at inducing points
    f_gp = K_XZ * u_inducing # Linear projection

    # RW2 Categorical
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # Likelihood
    for i in 1:N_obs
        idx_a = modinputs.area_idx[i]
        idx_t = modinputs.time_idx[i]
        mu = offset[i] + f_gp[i] + s_eff[idx_a] + f_time[idx_t] + seasonal[i] + st_interaction[idx_a, idx_t]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end




@model function model_C11_svgp(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, M_inducing_val=15, M_rff_u=30, M_rff_sigma=20, period=12.0) where {T}
    # SVGP-like (learned inducing points) with GP Trend and Non-linear Nested Covariates.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)
    D_st = 3

    # --- 1. Priors ---
    sigma_u ~ filldist(Exponential(0.5), 3)
    sigma_f ~ Exponential(1.0); ls_st ~ filldist(Gamma(2, 2), D_st)
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)
    sigma_log_var ~ Exponential(1.0)

    # --- 2. GP Temporal Trend ---
    ls_trend ~ Gamma(2, 2); sigma_trend ~ Exponential(0.5)
    k_trend = (sigma_trend^2) * (SqExponentialKernel() ∘ ScaleTransform(inv(ls_trend)))
    t_unique = collect(1:N_time) ./ N_time
    alpha_gp ~ MvNormal(zeros(N_time), kernelmatrix(k_trend, t_unique) + 1e-6*I)

    # --- 3. Non-linear Nested Structure (RFF) ---
    t_norm = modinputs.time_idx ./ N_time
    coords_tz = hcat(t_norm, modinputs.z_obs)
    W_u1 ~ filldist(Normal(0, 1), 2, M_rff_u); b_u1 ~ filldist(Uniform(0, 2 * π), M_rff_u)
    sigma_f_u1 ~ Exponential(1.0); beta_rff_u1 ~ filldist(Normal(0, sigma_f_u1), M_rff_u)
    u1_true = (sqrt(2/M_rff_u) .* cos.(coords_tz * W_u1 .+ b_u1')) * beta_rff_u1

    coords_tz_u1 = hcat(t_norm, modinputs.z_obs, u1_true)
    W_u2 ~ filldist(Normal(0, 1), 3, M_rff_u); b_u2 ~ filldist(Uniform(0, 2 * π), M_rff_u)
    sigma_f_u2 ~ Exponential(1.0); beta_rff_u2 ~ filldist(Normal(0, sigma_f_u2), M_rff_u)
    u2_true = (sqrt(2/M_rff_u) .* cos.(coords_tz_u1 * W_u2 .+ b_u2')) * beta_rff_u2

    W_u3 ~ filldist(Normal(0, 1), 3, M_rff_u); b_u3 ~ filldist(Uniform(0, 2 * π), M_rff_u)
    sigma_f_u3 ~ Exponential(1.0); beta_rff_u3 ~ filldist(Normal(0, sigma_f_u3), M_rff_u)
    u3_true = (sqrt(2/M_rff_u) .* cos.(coords_tz_u1 * W_u3 .+ b_u3')) * beta_rff_u3
    u_true_mat = hcat(u1_true, u2_true, u3_true)

    # --- 4. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 5. SVGP Mean (Learned Inducing Points) ---
    coords_st = hcat([p[1] for p in modinputs.pts], [p[2] for p in modinputs.pts], t_norm)
    Z_inducing = Matrix{T}(undef, M_inducing_val, D_st)
    for j in 1:D_st
        # Learn inducing locations via prior-constrained exploration
        Z_inducing[:, j] ~ filldist(Normal(mean(coords_st[:,j]), 2*std(coords_st[:,j])), M_inducing_val)
    end

    k_st = (sigma_f^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(ls_st)))
    K_ZZ = kernelmatrix(k_st, RowVecs(Z_inducing)) + 1e-6 * I
    K_XZ = kernelmatrix(k_st, RowVecs(coords_st), RowVecs(Z_inducing))
    K_XX_diag = diag(kernelmatrix(k_st, RowVecs(coords_st)))

    u_inducing ~ MvNormal(zeros(M_inducing_val), K_ZZ)
    f_mean = K_XZ * (K_ZZ \ u_inducing)
    cov_f_diag = K_XX_diag - diag(K_XZ * (K_ZZ \ K_XZ'))
    f_gp ~ MvNormal(f_mean, Diagonal(max.(1e-6, cov_f_diag)))

    # --- 6. Volatility & Categorical Smoothing ---
    W_vol ~ filldist(Normal(0, 1), D_st, M_rff_sigma); b_vol ~ filldist(Uniform(0, 2 * π), M_rff_sigma)
    beta_rff_vol ~ filldist(Normal(0, sigma_log_var), M_rff_sigma)
    log_sigma_y = (sqrt(2/M_rff_sigma) .* cos.(coords_st * W_vol .+ b_vol')) * beta_rff_vol
    sigma_y = exp.(log_sigma_y ./ 2.0)

    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 7. Likelihoods ---
    beta_z ~ Normal(0, 2); beta_u_main ~ MvNormal(zeros(3), 2.0 * I)
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * π .* t_norm * (N_time/period)) .+ beta_sin .* sin.(2 * π .* t_norm * (N_time/period))

    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        mu = offset[i] + alpha_gp[t] + seasonal[i] + beta_z * modinputs.z_obs[i] + dot(beta_u_main, u_true_mat[i, :]) + f_gp[i] + s_eff[a]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y[i]), y[i])
        for k in 1:3; Turing.@addlogprob! logpdf(Normal(u_true_mat[i, k], sigma_u[k]), modinputs.u_obs[i, k]); end
    end
end


@model function model_C12_svgp_full(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, M_inducing_val=15, M_rff_u=30, M_rff_sigma=20, period=12.0) where {T}
    # Full SVGP logic (learned inducing locations and latent distribution) with GP Trend.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)
    D_st = 3

    # --- 1. Priors ---
    sigma_u ~ filldist(Exponential(0.5), 3)
    sigma_f ~ Exponential(1.0); ls_st ~ filldist(Gamma(2, 2), D_st)
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)
    sigma_log_var ~ Exponential(1.0)

    # --- 2. GP Temporal Trend ---
    ls_trend ~ Gamma(2, 2); sigma_trend ~ Exponential(0.5)
    k_trend = (sigma_trend^2) * (SqExponentialKernel() ∘ ScaleTransform(inv(ls_trend)))
    t_unique = collect(1:N_time) ./ N_time
    alpha_gp ~ MvNormal(zeros(N_time), kernelmatrix(k_trend, t_unique) + 1e-6*I)

    # --- 3. Non-linear Nested Structure (RFF) ---
    t_norm = modinputs.time_idx ./ N_time
    coords_tz = hcat(t_norm, modinputs.z_obs)
    W_u1 ~ filldist(Normal(0, 1), 2, M_rff_u); b_u1 ~ filldist(Uniform(0, 2 * π), M_rff_u)
    sigma_f_u1 ~ Exponential(1.0); beta_rff_u1 ~ filldist(Normal(0, sigma_f_u1), M_rff_u)
    u1_true = (sqrt(2/M_rff_u) .* cos.(coords_tz * W_u1 .+ b_u1')) * beta_rff_u1

    coords_tz_u1 = hcat(t_norm, modinputs.z_obs, u1_true)
    W_u2 ~ filldist(Normal(0, 1), 3, M_rff_u); b_u2 ~ filldist(Uniform(0, 2 * π), M_rff_u)
    sigma_f_u2 ~ Exponential(1.0); beta_rff_u2 ~ filldist(Normal(0, sigma_f_u2), M_rff_u)
    u2_true = (sqrt(2/M_rff_u) .* cos.(coords_tz_u1 * W_u2 .+ b_u2')) * beta_rff_u2

    W_u3 ~ filldist(Normal(0, 1), 3, M_rff_u); b_u3 ~ filldist(Uniform(0, 2 * π), M_rff_u)
    sigma_f_u3 ~ Exponential(1.0); beta_rff_u3 ~ filldist(Normal(0, sigma_f_u3), M_rff_u)
    u3_true = (sqrt(2/M_rff_u) .* cos.(coords_tz_u1 * W_u3 .+ b_u3')) * beta_rff_u3
    u_true_mat = hcat(u1_true, u2_true, u3_true)

    # --- 4. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 5. Full SVGP Mean (Learned Locations and Latent Params) ---
    coords_st = hcat([p[1] for p in modinputs.pts], [p[2] for p in modinputs.pts], t_norm)
    Z_inducing = Matrix{T}(undef, M_inducing_val, D_st)
    for j in 1:D_st
        Z_inducing[:, j] ~ filldist(Normal(mean(coords_st[:,j]), 2*std(coords_st[:,j])), M_inducing_val)
    end

    # Variational parameters for the inducing points
    m_u ~ MvNormal(zeros(M_inducing_val), 10.0 * I)
    s_u_diag ~ filldist(Exponential(1.0), M_inducing_val)
    u_latent ~ MvNormal(m_u, Diagonal(s_u_diag.^2) + 1e-6*I)

    k_st = (sigma_f^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(ls_st)))
    K_ZZ = kernelmatrix(k_st, RowVecs(Z_inducing)) + 1e-6 * I
    K_XZ = kernelmatrix(k_st, RowVecs(coords_st), RowVecs(Z_inducing))
    K_XX_diag = diag(kernelmatrix(k_st, RowVecs(coords_st)))

    f_mean = K_XZ * (K_ZZ \ u_latent)
    cov_f_diag = K_XX_diag - diag(K_XZ * (K_ZZ \ K_XZ'))
    f_gp ~ MvNormal(f_mean, Diagonal(max.(1e-6, cov_f_diag)))

    # --- 6. Volatility & Categorical Smoothing ---
    W_vol ~ filldist(Normal(0, 1), D_st, M_rff_sigma); b_vol ~ filldist(Uniform(0, 2 * π), M_rff_sigma)
    beta_rff_vol ~ filldist(Normal(0, sigma_log_var), M_rff_sigma)
    log_sigma_y = (sqrt(2/M_rff_sigma) .* cos.(coords_st * W_vol .+ b_vol')) * beta_rff_vol
    sigma_y = exp.(log_sigma_y ./ 2.0)

    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 7. Likelihoods ---
    beta_z ~ Normal(0, 2); beta_u_main ~ MvNormal(zeros(3), 2.0 * I)
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * π .* t_norm * (N_time/period)) .+ beta_sin .* sin.(2 * π .* t_norm * (N_time/period))

    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        mu = offset[i] + alpha_gp[t] + seasonal[i] + beta_z * modinputs.z_obs[i] + dot(beta_u_main, u_true_mat[i, :]) + f_gp[i] + s_eff[a]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y[i]), y[i])
        for k in 1:3; Turing.@addlogprob! logpdf(Normal(u_true_mat[i, k], sigma_u[k]), modinputs.u_obs[i, k]); end
    end
end


@model function model_C13_multifidelity_gp(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, period=12.0) where {T}
    # --- 1. Data Dimensions & Multi-fidelity Inputs ---
    y = modinputs.y
    z_obs = modinputs.z_obs
    u_obs = modinputs.u_obs
    coords_z = modinputs.coords_z_spatial
    coords_u = modinputs.coords_u_st
    N_obs = length(y)
    N_areas = size(modinputs.Q_sp, 1)
    N_time = maximum(modinputs.time_idx)
    M_rff = size(modinputs.W_z_fixed, 2)

    # --- 2. Hierarchical Priors ---
    sigma_y ~ Exponential(1.0) # Standard noise
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1) # BYM2 params
    sigma_tm ~ Exponential(1.0); rho_tm ~ Beta(2, 2) # AR1 params
    sigma_int ~ Exponential(0.5); sigma_rw2 ~ filldist(Exponential(1.0), 4) # Smoothing
    sigma_z ~ Exponential(0.5); sigma_u ~ filldist(Exponential(0.5), 3) # Fidelity noise
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1) # Seasonal weights

    # --- 3. Component 1: BYM2 Spatial Effect ---
    u_icar ~ MvNormal(zeros(N_areas), I)
    Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 4. Component 2: AR1 Temporal Effect ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # --- 5. Component 3: Harmonic Seasonality ---
    t_vec = Float64.(modinputs.time_idx)
    seasonal = beta_cos .* cos.(2 * pi .* t_vec ./ period) .+ beta_sin .* sin.(2 * pi .* t_vec ./ period)

    # --- 6. Component 4: Space-Time Interaction (GMRF) ---
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I)
    st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # --- 7. Component 5: Multi-fidelity RFF Projections ---
    # Latent Z Fidelity
    W_z ~ filldist(Normal(0, 1), size(coords_z, 2), M_rff)
    b_z ~ filldist(Uniform(0, 2pi), M_rff)
    beta_z_rff ~ filldist(Normal(0, 1), M_rff)
    z_latent = rff_map(coords_z, W_z, b_z) * beta_z_rff
    Turing.@addlogprob! logpdf(MvNormal(z_latent, sigma_z^2 * I), z_obs)

    # Latent U Fidelity
    W_u ~ filldist(Normal(0, 1), size(coords_u, 2), M_rff)
    b_u ~ filldist(Uniform(0, 2pi), M_rff)
    beta_u_rff ~ filldist(Normal(0, 1), M_rff, 3)
    u_latent = rff_map(coords_u, W_u, b_u) * beta_u_rff
    for k in 1:3; Turing.@addlogprob! logpdf(MvNormal(u_latent[:, k], sigma_u[k]^2 * I), u_obs[:, k]); end

    # --- 8. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 9. Final Likelihood ---
    for i in 1:N_obs
        idx_a = modinputs.area_idx[i]
        idx_t = modinputs.time_idx[i]
        mu = offset[i] + s_eff[idx_a] + f_time[idx_t] + seasonal[i] + st_interaction[idx_a, idx_t]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end

@model function model_C14_minibatch_mfgp(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, period=12.0) where {T}

    y = modinputs.y
    N_obs = length(y)
    N_areas = size(modinputs.Q_sp, 1)
    N_time = maximum(modinputs.time_idx)
    
    # Priors
    sigma_y ~ Exponential(1.0); sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0); rho_tm ~ Beta(2, 2); sigma_int ~ Exponential(0.5)
    sigma_rw2 ~ filldist(Exponential(1.0), 4); beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)

    # BYM2
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I); s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # AR1
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I); Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # Seasonal
    seasonal = beta_cos .* cos.(2 * pi .* Float64.(modinputs.time_idx) ./ period) .+ beta_sin .* sin.(2 * pi .* Float64.(modinputs.time_idx) ./ period)

    # GMRF Interaction
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I); st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # RW2
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # Likelihood
    for i in 1:N_obs
        idx_a = modinputs.area_idx[i]; idx_t = modinputs.time_idx[i]
        mu = offset[i] + s_eff[idx_a] + f_time[idx_t] + seasonal[i] + st_interaction[idx_a, idx_t]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end


@model function model_C15_deep_gp(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, period=12.0, M_rff=40) where {T}
    # Deep Spatiotemporal GP.
    # Integrates BYM2 spatial effects and RW2 smoothing into a 3-layer RFF hierarchy.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)
    coords_space = modinputs.pts
    coords_time = reshape(modinputs.time_idx ./ N_time, :, 1)
    D_s = size(coords_space, 2)
    D_t = size(coords_time, 2)

    # --- 1. Priors & Structural Components ---
    sigma_y ~ Exponential(1.0); sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_z ~ Exponential(0.5); sigma_u ~ filldist(Exponential(0.5), 3)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # BYM2 Spatial Effect
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # RW2 Categorical Smoothing
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 2. Layer 1: Latent Spatial GP (Z) ---
    W_z ~ filldist(Normal(0, 1), D_s, M_rff)
    b_z ~ filldist(Uniform(0, 2pi), M_rff)
    beta_z ~ filldist(Normal(0, 1), M_rff)
    z_latent = rff_map(coords_space, W_z, b_z) * beta_z

    # --- 3. Layer 2: Latent Spatiotemporal GPs (U1, U2, U3) ---
    coords_l2 = hcat(coords_space, coords_time, z_latent)
    W_u ~ filldist(Normal(0, 1), size(coords_l2, 2), M_rff)
    b_u ~ filldist(Uniform(0, 2pi), M_rff)
    beta_u_mat ~ filldist(Normal(0, 1), M_rff, 3)
    Phi_u = rff_map(coords_l2, W_u, b_u)
    u_latent = Phi_u * beta_u_mat # Matrix N_obs x 3

    # --- 4. Layer 3: Final Output GP (Y) ---
    coords_l3 = hcat(coords_space, coords_time, u_latent)
    W_y ~ filldist(Normal(0, 1), size(coords_l3, 2), M_rff)
    b_y ~ filldist(Uniform(0, 2pi), M_rff)
    beta_y_gp ~ filldist(Normal(0, 1), M_rff)

    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * π .* coords_time ./ period) .+ beta_sin .* sin.(2 * π .* coords_time ./ period)
    f_y = (rff_map(coords_l3, W_y, b_y) * beta_y_gp) .+ vec(seasonal)

    # --- 5. Likelihood ---
    for i in 1:N_obs
        a = modinputs.area_idx[i]
        mu = offset[i] + f_y[i] + s_eff[a]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end

    # Multi-fidelity cross-resolution constraints (Optional prior links)
    modinputs.z_obs ~ MvNormal(z_latent, sigma_z^2 * I)
    for k in 1:3; modinputs.u_obs[:, k] ~ MvNormal(u_latent[:, k], sigma_u[k]^2 * I); end
end


@model function model_C16_nystrom(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, period=12.0, M_rff_sigma=20, M_inducing_val=15, M_rff_u=30) where {T}
    # Model A16 Optimized: Standardized Nyström GP with Stochastic Volatility.
    # Combines low-rank spatiotemporal approximations with heteroskedastic noise modeling.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)
    coords_space = modinputs.pts
    coords_time = reshape(modinputs.time_idx ./ N_time, :, 1)
    D_s, D_t = size(coords_space, 2), size(coords_time, 2)
    D_st = D_s + D_t

    # --- 1. Priors & Structural Components ---
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_u ~ filldist(Exponential(0.5), 3)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # BYM2 Spatial Effect
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # RW2 Categorical Smoothing
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 2. Nested Latent Covariates (RFF Mapping) ---
    coords_tz = hcat(coords_time, modinputs.z_obs)
    W_u1 ~ filldist(Normal(0, 1), size(coords_tz, 2), M_rff_u); b_u1 ~ filldist(Uniform(0, 2 * π), M_rff_u)
    u1_true = rff_map(coords_tz, W_u1, b_u1) * filldist(Normal(0, 1), M_rff_u)

    coords_tz_u1 = hcat(coords_time, modinputs.z_obs, u1_true)
    W_u2 ~ filldist(Normal(0, 1), size(coords_tz_u1, 2), M_rff_u); b_u2 ~ filldist(Uniform(0, 2 * π), M_rff_u)
    u2_true = rff_map(coords_tz_u1, W_u2, b_u2) * filldist(Normal(0, 1), M_rff_u)

    # --- 3. Nyström GP (Low Rank Approximation) ---
    coords_st = hcat(coords_space, coords_time)
    ls_st ~ filldist(Gamma(2, 2), D_st); sigma_f ~ Exponential(1.0)
    k_st = SqExponentialKernel() ∘ ARDTransform(inv.(ls_st))

    Z_ind = Matrix{T}(undef, M_inducing_val, D_st)
    mu_st, std_st = mean(coords_st, dims=1), std(coords_st, dims=1)
    for j in 1:D_st; Z_ind[:, j] ~ filldist(Normal(mu_st[j], 2.0 * std_st[j]), M_inducing_val); end

    K_zz = kernelmatrix(k_st, RowVecs(Z_ind)) + 1e-6*I
    K_xz = kernelmatrix(k_st, RowVecs(coords_st), RowVecs(Z_ind))
    v_latent ~ filldist(Normal(0, 1), M_inducing_val)
    f_nystrom = sigma_f .* (K_xz * (cholesky(K_zz).U \ v_latent))

    # --- 4. Spatiotemporal Stochastic Volatility ---
    W_sigma ~ filldist(Normal(0, 1), D_st, M_rff_sigma); b_sigma ~ filldist(Uniform(0, 2 * π), M_rff_sigma)
    beta_rff_sigma ~ filldist(Normal(0, 1), M_rff_sigma)
    sigma_y = exp.(rff_map(coords_st, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 5. Likelihood ---
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * π .* coords_time ./ period) .+ beta_sin .* sin.(2 * π .* coords_time ./ period)

    for i in 1:N_obs
        a = modinputs.area_idx[i]
        mu = offset[i] + f_nystrom[i] + seasonal[i] + s_eff[a]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y[i]), y[i])
    end

    # Prior links for latent variables
    modinputs.u_obs[:, 1] ~ MvNormal(u1_true, sigma_u[1]^2 * I)
    modinputs.u_obs[:, 2] ~ MvNormal(u2_true, sigma_u[2]^2 * I)
end


@model function model_C17_spde(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, period=12.0, M_rff_sigma=20, M_rff_u=30) where {T}
    # SPDE-based Spatiotemporal GP.
    # Employs sparse precision approximations for spatial effects and RFF for volatility.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)
    coords_space = modinputs.pts
    coords_time = reshape(modinputs.time_idx ./ N_time, :, 1)
    D_s, D_t = size(coords_space, 2), size(coords_time, 2)
    D_st = D_s + D_t

    # --- 1. Priors & Structural Components ---
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_u ~ filldist(Exponential(0.5), 3)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # BYM2 Spatial Effect
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # RW2 Categorical Smoothing
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 2. Latent Trends (GP Trend & Seasonal) ---
    ls_trend ~ Gamma(2, 2); sigma_trend ~ Exponential(0.5)
    k_trend = SqExponentialKernel() ∘ ScaleTransform(inv(ls_trend))
    unique_times = sort(unique(coords_time[:,1]))
    alpha ~ GP(sigma_trend^2 * k_trend)(unique_times, 1e-6)
    trend = alpha[indexin(coords_time[:,1], unique_times)]

    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * π .* coords_time ./ period) .+ beta_sin .* sin.(2 * π .* coords_time ./ period)

    # --- 3. Nested Latent Covariates (RFF) ---
    coords_tz = hcat(coords_time, modinputs.z_obs)
    W_u1 ~ filldist(Normal(0, 1), size(coords_tz, 2), M_rff_u); b_u1 ~ filldist(Uniform(0, 2 * π), M_rff_u)
    u1_true = rff_map(coords_tz, W_u1, b_u1) * filldist(Normal(0, 1), M_rff_u)

    # --- 4. Stochastic Volatility ---
    coords_st = hcat(coords_space, coords_time)
    W_sigma ~ filldist(Normal(0, 1), D_st, M_rff_sigma); b_sigma ~ filldist(Uniform(0, 2 * π), M_rff_sigma)
    beta_rff_sigma ~ filldist(Normal(0, 1), M_rff_sigma)
    sigma_y = exp.(rff_map(coords_st, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 5. Likelihood ---
    for i in 1:N_obs
        a = modinputs.area_idx[i]
        mu = offset[i] + trend[i] + seasonal[i] + s_eff[a] + u1_true[i]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y[i]), y[i])
    end

    # multi-fidelity links
    modinputs.u_obs[:, 1] ~ MvNormal(u1_true, sigma_u[1]^2 * I)
end


@model function model_C18_kronecker_spde(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, M_rff_sigma=20) where {T}
    # Kronecker SPDE Spatiotemporal GP.
    # Utilizes Kronecker-structured precision matrices for efficient spatiotemporal inference.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)
    coords_space = modinputs.pts
    coords_time = reshape(modinputs.time_idx ./ N_time, :, 1)
    unique_t = collect(1:N_time) ./ N_time
    unique_s = coords_space[1:N_areas, :]

    # --- 1. Priors & Structural Components ---
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_u ~ filldist(Exponential(0.5), 3)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # BYM2 Spatial Effect
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # RW2 Categorical Smoothing
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 2. Latent Kronecker Fields (Z, U1) ---
    ls_s_cov ~ Gamma(2, 2); sigma_s_cov ~ Exponential(1.0)
    ls_t_cov ~ Gamma(2, 2); sigma_t_cov ~ Exponential(1.0)

    z_noise ~ filldist(Normal(0, 1), N_obs)
    z_true = kron_matern_sample(N_areas, N_time, unique_s, unique_t, ls_s_cov, sigma_s_cov, ls_t_cov, sigma_t_cov, z_noise)

    u1_noise ~ filldist(Normal(0, 1), N_obs)
    u1_true = kron_matern_sample(N_areas, N_time, unique_s, unique_t, ls_s_cov, sigma_s_cov, ls_t_cov, sigma_t_cov, u1_noise)

    # --- 3. Latent Kronecker Field for Y (Main Effect) ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    ls_t_y ~ Gamma(2, 2); sigma_t_y ~ Exponential(1.0)
    y_noise ~ filldist(Normal(0, 1), N_obs)
    f_st = kron_matern_sample(N_areas, N_time, unique_s, unique_t, ls_s_y, sigma_s_y, ls_t_y, sigma_t_y, y_noise)

    # --- 4. Stochastic Volatility (RFF) ---
    coords_st = hcat(coords_space, coords_time)
    W_sigma ~ filldist(Normal(0, 1), size(coords_st, 2), M_rff_sigma); b_sigma ~ filldist(Uniform(0, 2 * π), M_rff_sigma)
    beta_rff_sigma ~ filldist(Normal(0, 1), M_rff_sigma)
    sigma_y = exp.(rff_map(coords_st, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 5. Likelihood ---
    for i in 1:N_obs
        a = modinputs.area_idx[i]
        mu = offset[i] + f_st[i] + s_eff[a] + u1_true[i] + z_true[i]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y[i]), y[i])
    end

    # Prior links for multi-fidelity variables
    modinputs.z_obs ~ MvNormal(z_true, 0.1 * I)
    modinputs.u_obs[:, 1] ~ MvNormal(u1_true, sigma_u[1]^2 * I)
end


@model function model_C19_svgp_matern(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, period=12.0, M_rff_sigma=20, M_rff_u=30) where {T}
    # SVGP with Matern structure.
    # Integrates nested RFF latent covariates and Kronecker spatiotemporal kernels.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)
    coords_space = modinputs.pts
    coords_time = reshape(modinputs.time_idx ./ N_time, :, 1)
    unique_t = collect(1:N_time) ./ N_time
    unique_s = coords_space[1:N_areas, :]
    D_st = size(coords_space, 2) + 1

    # --- 1. Priors & Structural Components ---
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_u ~ filldist(Exponential(0.5), 3); sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # BYM2 Spatial Effect
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # RW2 Categorical Smoothing
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 2. Nested Latent Covariates (RFF) ---
    coords_tz = hcat(coords_time, modinputs.z_obs)
    W_u1 ~ filldist(Normal(0, 1), size(coords_tz, 2), M_rff_u); b_u1 ~ filldist(Uniform(0, 2 * π), M_rff_u)
    u1_true = rff_map(coords_tz, W_u1, b_u1) * filldist(Normal(0, 1), M_rff_u)

    # --- 3. Main Spatiotemporal Process (Kronecker) ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    ls_t_y ~ Gamma(2, 2); sigma_t_y ~ Exponential(1.0)
    y_noise ~ filldist(Normal(0, 1), N_obs)
    f_st = kron_matern_sample(N_areas, N_time, unique_s, unique_t, ls_s_y, sigma_s_y, ls_t_y, sigma_t_y, y_noise)

    # --- 4. Stochastic Volatility (RFF) ---
    coords_st = hcat(coords_space, coords_time)
    W_sigma ~ filldist(Normal(0, 1), size(coords_st, 2), M_rff_sigma); b_sigma ~ filldist(Uniform(0, 2 * π), M_rff_sigma)
    beta_rff_sigma ~ filldist(Normal(0, 1), M_rff_sigma)
    sigma_y = exp.(rff_map(coords_st, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 5. Likelihood ---
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * π .* coords_time ./ period) .+ beta_sin .* sin.(2 * π .* coords_time ./ period)

    for i in 1:N_obs
        a = modinputs.area_idx[i]
        mu = offset[i] + f_st[i] + seasonal[i] + s_eff[a] + u1_true[i]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y[i]), y[i])
    end

    # multi-fidelity links
    modinputs.u_obs[:, 1] ~ MvNormal(u1_true, sigma_u[1]^2 * I)
end



@model function model_C20_multifidelity_gp_matern(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Multi-fidelity GP with standardized interface.
    # Combines high-fidelity Z (Matern), medium-fidelity U (Kron AR1 x Matern), 
    # and standard-fidelity Y with standardized BYM2 and RW2 components.

    # Dimensions and Observations from modinputs
    y_obs = modinputs.y
    z_obs = modinputs.z_obs
    u1_obs, u2_obs, u3_obs = modinputs.u_obs[:, 1], modinputs.u_obs[:, 2], modinputs.u_obs[:, 3]
    
    Ny = length(y_obs); Nt_y = maximum(modinputs.time_idx); Ns_y = Ny ÷ Nt_y
    Nu = length(u1_obs); Nt_u = Nt_y; Ns_u = Nu ÷ Nt_u
    Nz = length(z_obs)

    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0); sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_int ~ Exponential(0.5); sigma_rw2 ~ filldist(Exponential(1.0), 4)
    
    # --- 2. High Fidelity: Latent Spatial Z (Matern 3/2) ---
    ls_z ~ Gamma(2, 2); sigma_z_f ~ Exponential(1.0)
    k_z = Matern32Kernel() ∘ ScaleTransform(inv(ls_z))
    K_z = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(modinputs.pts[1:Nz, :])) + 1e-6*I
    z_latent ~ MvNormal(zeros(Nz), K_z)
    sigma_z_obs ~ Exponential(0.5)
    z_obs ~ MvNormal(z_latent, sigma_z_obs^2 * I)

    # Interpolation of Z to U/Y locations
    K_z_u = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(modinputs.pts[1:Ns_u, :]), RowVecs(modinputs.pts[1:Nz, :]))
    z_at_u = (K_z_u * (K_z \ z_latent))
    z_at_u_full = repeat(z_at_u, Nt_u)

    # --- 3. Medium Fidelity: Latent Spatiotemporal U (Kron AR1 x Matern) ---
    ls_s_u ~ Gamma(2, 2); sigma_s_u ~ Exponential(1.0)
    rho_u ~ Uniform(-0.99, 0.99); sigma_t_u ~ Exponential(0.5)
    u1_noise ~ filldist(Normal(0, 1), Nu)
    
    # Sample U1 (Hierarchical dependence on Z)
    u1_true = kron_ar1_matern_sample(Ns_u, Nt_u, modinputs.pts[1:Ns_u, :], ls_s_u, sigma_s_u, rho_u, sigma_t_u, u1_noise)
    beta_uz ~ Normal(0, 1)
    u1_obs ~ MvNormal(u1_true .+ beta_uz .* z_at_u_full, 0.1*I)

    # --- 4. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(Ns_y), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(Ns_y), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 5. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Likelihood (Standard Fidelity Y) ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    rho_y ~ Uniform(-0.99, 0.99); sigma_t_y ~ Exponential(0.5)
    y_noise ~ filldist(Normal(0, 1), Ny)
    f_st_y = kron_ar1_matern_sample(Ns_y, Nt_y, modinputs.pts[1:Ns_y, :], ls_s_y, sigma_s_y, rho_y, sigma_t_y, y_noise)

    beta_y ~ Normal(0, 1)
    for i in 1:Ny
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        mu = offset[i] + f_st_y[i] + s_eff[a] + beta_y * z_at_u_full[i]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y_obs[i])
    end
end


@model function model_C21_multifidelity_gp_matern_sv_seasonal(modinputs, ::Type{T}=Float64; period=12.0, M_rff_sigma=20, offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Multi-fidelity GP with SV, Seasonal Harmonics, and Standardized Interface.

    # Dimensions and Observations
    y_obs = modinputs.y
    z_obs = modinputs.z_obs
    u1_obs, u2_obs, u3_obs = modinputs.u_obs[:, 1], modinputs.u_obs[:, 2], modinputs.u_obs[:, 3]

    Ny = length(y_obs); Nt_y = maximum(modinputs.time_idx); Ns_y = Ny ÷ Nt_y
    Nu = length(u1_obs); Nt_u = Nt_y; Ns_u = Nu ÷ Nt_u
    Nz = length(z_obs)

    # --- 1. High Fidelity: Latent Spatial Z (Matern 3/2) ---
    ls_z ~ Gamma(2, 2); sigma_z_f ~ Exponential(1.0)
    k_z = Matern32Kernel() ∘ ScaleTransform(inv(ls_z))
    K_z = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(modinputs.pts[1:Nz, :])) + 1e-3*I
    z_latent ~ MvNormal(zeros(Nz), K_z)
    sigma_z_obs ~ Exponential(0.5)
    z_obs ~ MvNormal(z_latent, sigma_z_obs^2 * I)

    # Interpolation of Z to U and Y
    K_z_u = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(modinputs.pts[1:Ns_u, :]), RowVecs(modinputs.pts[1:Nz, :]))
    z_at_u = (K_z_u * (K_z \ z_latent))
    K_z_y = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(modinputs.pts[1:Ns_y, :]), RowVecs(modinputs.pts[1:Nz, :]))
    z_at_y = (K_z_y * (K_z \ z_latent))

    # --- 2. Medium Fidelity: Latent Spatiotemporal U (Kron AR1 x Matern) ---
    ls_s_u ~ Gamma(2, 2); sigma_s_u ~ Exponential(1.0)
    rho_u ~ Uniform(-0.99, 0.99); sigma_t_u ~ Exponential(0.5)
    u1_noise ~ filldist(Normal(0, 1), Nu)
    unique_pts_u = modinputs.pts[1:Ns_u, :]
    u1_true = kron_ar1_matern_sample(Ns_u, Nt_u, unique_pts_u, ls_s_u, sigma_s_u, rho_u, sigma_t_u, u1_noise)
    
    beta_uz ~ Normal(0, 1)
    u1_obs ~ MvNormal(u1_true .+ beta_uz .* repeat(z_at_u, Nt_u), 0.1*I)

    # Interpolation of U to Y coordinates (Kronecker)
    unique_pts_y = modinputs.pts[1:Ns_y, :]
    k_s_u_interp = Matern32Kernel() ∘ ScaleTransform(inv(ls_s_u))
    K_s_uu = sigma_s_u^2 * kernelmatrix(k_s_u_interp, RowVecs(unique_pts_u)) + 1e-3*I
    K_s_yu = sigma_s_u^2 * kernelmatrix(k_s_u_interp, RowVecs(unique_pts_y), RowVecs(unique_pts_u))
    
    t_u = collect(1:Nt_u); t_y = collect(1:Nt_y)
    K_t_uu = ar1_covariance_matrix(t_u, rho_u, sigma_t_u) + 1e-3*I
    K_t_yu = ar1_cross_covariance_matrix(t_y, t_u, rho_u, sigma_t_u)
    u1_at_y = kron(K_t_yu, K_s_yu) * (kron(K_t_uu, K_s_uu) \ u1_true)

    # --- 3. Standard Components (BYM2 & RW2) ---
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    u_icar ~ MvNormal(zeros(Ns_y), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(Ns_y), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    sigma_rw2 ~ filldist(Exponential(1.0), 4)
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 4. Stochastic Volatility & Seasonality ---
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    t_vec = modinputs.time_idx ./ Nt_y
    seasonal_y = beta_cos .* cos.(2 * π .* t_vec) .+ beta_sin .* sin.(2 * π .* t_vec)

    coords_st_y = hcat(modinputs.pts, modinputs.time_idx)
    W_sigma ~ filldist(Normal(0, 1), size(coords_st_y, 2), M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2 * π), M_rff_sigma)
    sigma_log_var ~ Exponential(1.0); beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M_rff_sigma)
    sigma_y_vec = exp.(rff_map(coords_st_y, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 5. Likelihood ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    rho_y ~ Uniform(-0.99, 0.99); sigma_t_y ~ Exponential(0.5); y_noise ~ filldist(Normal(0, 1), Ny)
    f_st_y = kron_ar1_matern_sample(Ns_y, Nt_y, unique_pts_y, ls_s_y, sigma_s_y, rho_y, sigma_t_y, y_noise)

    beta_y_covs ~ filldist(Normal(0, 1), 2)
    for i in 1:Ny
        a = modinputs.area_idx[i]
        mu = offset[i] + f_st_y[i] + s_eff[a] + seasonal_y[i] + (u1_at_y[i] * beta_y_covs[1]) + (repeat(z_at_y, Nt_y)[i] * beta_y_covs[2])
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y_vec[i]), y_obs[i])
    end
end


@model function model_C22_rff_multifidelity_gp(modinputs, ::Type{T}=Float64; period=12.0, M_rff_base=40, M_rff_sigma=20, offset=modinputs.offset, weights=modinputs.weights) where {T}
    # RFF-based Multi-fidelity GP with standardized interface.
    # Uses RFF mapping for cross-fidelity interpolation and stochastic volatility.

    # Dimensions and Observations from modinputs
    y_obs = modinputs.y
    z_obs = modinputs.z_obs
    u1_obs, u2_obs, u3_obs = modinputs.u_obs[:, 1], modinputs.u_obs[:, 2], modinputs.u_obs[:, 3]

    Ny = length(y_obs); Nt_y = maximum(modinputs.time_idx); Ns_y = Ny ÷ Nt_y
    Nu = length(u1_obs); Nt_u = Nt_y; Ns_u = Nu ÷ Nt_u
    Nz = length(z_obs)

    # --- 1. High Fidelity: Latent Spatial Z (RFF GP) ---
    coords_z_s = modinputs.pts[1:Nz, :]
    D_z_in = size(coords_z_s, 2)
    W_z ~ filldist(Normal(0, 1), D_z_in, M_rff_base)
    b_z ~ filldist(Uniform(0, 2 * π), M_rff_base)
    sigma_z_f ~ Exponential(1.0)
    beta_z ~ filldist(Normal(0, sigma_z_f^2), M_rff_base)

    Phi_z = rff_map(coords_z_s, W_z, b_z)
    z_latent = Phi_z * beta_z
    sigma_z_obs ~ Exponential(0.5)
    z_obs ~ MvNormal(z_latent, sigma_z_obs^2 * I)

    # --- 2. Medium Fidelity: Latent Spatiotemporal U (RFF GP) ---
    # Interpolate Z to U locations using RFF
    coords_u_s = modinputs.pts[1:Nu, :] # Simplified mapping for demo
    coords_u_t = repeat(1:Nt_u, inner=Ns_u)
    z_at_u_s = (rff_map(coords_u_s, W_z, b_z) * beta_z)

    coords_u_rff_in = hcat(coords_u_s, coords_u_t, z_at_u_s)
    W_u ~ filldist(Normal(0, 1), size(coords_u_rff_in, 2), M_rff_base)
    b_u ~ filldist(Uniform(0, 2 * π), M_rff_base)
    sigma_u_f ~ Exponential(1.0)
    beta_u1 ~ filldist(Normal(0, sigma_u_f^2), M_rff_base)

    u1_true = rff_map(coords_u_rff_in, W_u, b_u) * beta_u1
    sigma_u_obs ~ Exponential(0.5)
    u1_obs ~ MvNormal(u1_true, sigma_u_obs^2 * I)

    # --- 3. Standard Components (BYM2 & RW2) ---
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    u_icar ~ MvNormal(zeros(Ns_y), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(Ns_y), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    sigma_rw2 ~ filldist(Exponential(1.0), 4)
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 4. Standard Fidelity: Dependent Y (RFF GP) ---
    # Interpolate Z and U to Y locations
    z_at_y = (rff_map(modinputs.pts, W_z, b_z) * beta_z)
    u1_at_y = rff_map(hcat(modinputs.pts, modinputs.time_idx, z_at_y), W_u, b_u) * beta_u1

    coords_y_rff_in = hcat(modinputs.pts, modinputs.time_idx, z_at_y, u1_at_y)
    W_y_gp ~ filldist(Normal(0, 1), size(coords_y_rff_in, 2), M_rff_base)
    b_y_gp ~ filldist(Uniform(0, 2 * π), M_rff_base)
    sigma_y_gp_f ~ Exponential(1.0)
    beta_y_gp ~ filldist(Normal(0, sigma_y_gp_f^2), M_rff_base)
    f_st_y = rff_map(coords_y_rff_in, W_y_gp, b_y_gp) * beta_y_gp

    # Stochastic Volatility & Seasonality
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal_y = beta_cos .* cos.(2 * π .* modinputs.time_idx ./ Nt_y) .+ beta_sin .* sin.(2 * π .* modinputs.time_idx ./ Nt_y)

    coords_st_y = hcat(modinputs.pts, modinputs.time_idx)
    W_sigma ~ filldist(Normal(0, 1), size(coords_st_y, 2), M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2 * π), M_rff_sigma)
    sigma_log_var ~ Exponential(1.0); beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M_rff_sigma)
    sigma_y_vec = exp.(rff_map(coords_st_y, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 5. Likelihood ---
    for i in 1:Ny
        a = modinputs.area_idx[i]
        mu = offset[i] + f_st_y[i] + s_eff[a] + seasonal_y[i]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y_vec[i] + 1e-3), y_obs[i])
    end
end


@model function model_C23_dfrff_multifidelity_gp(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, period=12.0) where {T}
    # Deep Functional RFF using precomputed fixed bases from modinputs.
    y = modinputs.y
    W_z = modinputs.W_z_fixed; b_z = modinputs.b_z_fixed
    W_u = modinputs.W_u_fixed; b_u = modinputs.b_u_fixed
    W_y = modinputs.W_y_gp_fixed; b_y = modinputs.b_y_gp_fixed
    N_obs = length(y); N_areas = size(modinputs.Q_sp, 1); N_time = maximum(modinputs.time_idx)

    # Priors
    sigma_y ~ Exponential(1.0); sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0); rho_tm ~ Beta(2, 2); sigma_int ~ Exponential(0.5)
    sigma_rw2 ~ filldist(Exponential(1.0), 4); beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)

    # BYM2, AR1, Seasonal, GMRF components consistent with A06 documentation style
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I); s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I); Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm
    seasonal = beta_cos .* cos.(2 * pi .* Float64.(modinputs.time_idx) ./ period) .+ beta_sin .* sin.(2 * pi .* Float64.(modinputs.time_idx) ./ period)
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I); st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # Functional Basis Mapping (Deep Functional Logic)
    beta_z ~ filldist(Normal(0, 1), size(W_z, 2))
    beta_u ~ filldist(Normal(0, 1), size(W_u, 2))
    beta_y_gp ~ filldist(Normal(0, 1), size(W_y, 2))
    f_z = rff_map(modinputs.coords_z_spatial, W_z, b_z) * beta_z
    f_u = rff_map(modinputs.coords_u_st, W_u, b_u) * beta_u
    f_gp = rff_map(hcat(modinputs.pts, modinputs.time_idx), W_y, b_y) * beta_y_gp

    # RW2 & Likelihood
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end
    for i in 1:N_obs
        idx_a = modinputs.area_idx[i]; idx_t = modinputs.time_idx[i]
        mu = offset[i] + f_gp[i] + s_eff[idx_a] + f_time[idx_t] + seasonal[i] + st_interaction[idx_a, idx_t]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end

@model function model_C24_semi_adaptive_dfrff(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, period=12.0) where {T}
    # Semi-Adaptive DFRFF with learned noise on fixed weight bases.
    y = modinputs.y
    N_obs = length(y); N_areas = size(modinputs.Q_sp, 1); N_time = maximum(modinputs.time_idx)
    W_z_base = modinputs.W_z_fixed; W_u_base = modinputs.W_u_fixed

    # Adaptive Priors
    sigma_W_z ~ Exponential(0.1); sigma_W_u ~ Exponential(0.1)
    sigma_y ~ Exponential(1.0); sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0); rho_tm ~ Beta(2, 2); sigma_int ~ Exponential(0.5)
    sigma_rw2 ~ filldist(Exponential(1.0), 4); beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)

    # Components
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I); s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I); Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm
    seasonal = beta_cos .* cos.(2 * pi .* Float64.(modinputs.time_idx) ./ period) .+ beta_sin .* sin.(2 * pi .* Float64.(modinputs.time_idx) ./ period)
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I); st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # Semi-Adaptive Weights
    W_z ~ MvNormal(vec(W_z_base), sigma_W_z^2 * I)
    W_u ~ MvNormal(vec(W_u_base), sigma_W_u^2 * I)
    f_gp = rff_map(hcat(modinputs.pts, modinputs.time_idx), modinputs.W_y_gp_fixed, modinputs.b_y_gp_fixed) * filldist(Normal(0,1), size(modinputs.W_y_gp_fixed, 2))

    # RW2 & Likelihood
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end
    for i in 1:N_obs
        idx_a = modinputs.area_idx[i]; idx_t = modinputs.time_idx[i]
        mu = offset[i] + f_gp[i] + s_eff[idx_a] + f_time[idx_t] + seasonal[i] + st_interaction[idx_a, idx_t]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end

@model function model_C25_hybrid_fitc_rff(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights, period=12.0) where {T}
    # Hybrid Spatiotemporal GP combining RFF and Sparse FITC. 
    # Refactored for modinputs_gaussian variable seeking.
    y = modinputs.y
    Z_inducing = modinputs.Z_inducing
    W_z_fixed = modinputs.W_z_fixed; b_z_fixed = modinputs.b_z_fixed
    N_obs = length(y); N_areas = size(modinputs.Q_sp, 1); N_time = maximum(modinputs.time_idx)

    # Priors
    sigma_y ~ Exponential(1.0); sigma_f ~ Exponential(1.0); sigma_sp ~ Exponential(1.0)
    phi_sp ~ Beta(1, 1); sigma_tm ~ Exponential(1.0); rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5); sigma_rw2 ~ filldist(Exponential(1.0), 4)
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1); sigma_z_f ~ Exponential(1.0)
    ls_st ~ filldist(Gamma(2, 2), 3); beta_z ~ filldist(Normal(0, sigma_z_f), size(W_z_fixed, 2))

    # BYM2, AR1, Seasonal, GMRF components
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I); s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I); Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm
    seasonal = beta_cos .* cos.(2 * pi .* Float64.(modinputs.time_idx) ./ period) .+ beta_sin .* sin.(2 * pi .* Float64.(modinputs.time_idx) ./ period)
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I); st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # Hybrid GP Basis Logic
    z_at_y = vec(rff_map(modinputs.pts, W_z_fixed, b_z_fixed) * beta_z)
    k_y = (sigma_f^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(ls_st)))
    K_XZ = kernelmatrix(k_y, RowVecs(modinputs.pts), RowVecs(Z_inducing))
    u_inducing ~ MvNormal(zeros(size(Z_inducing, 1)), I)
    f_gp_fitc = K_XZ * u_inducing

    # RW2 Smoothing & Likelihood
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end
    for i in 1:N_obs
        idx_a = modinputs.area_idx[i]; idx_t = modinputs.time_idx[i]
        mu = offset[i] + f_gp_fitc[i] + z_at_y[i] + s_eff[idx_a] + f_time[idx_t] + seasonal[i] + st_interaction[idx_a, idx_t]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end


 


###-------------------

###-------------------

###-------------------

###-------------------

###-------------------

###-------------------

###-------------------

###-------------------

###-------------------

###-------------------

###-------------------

###-------------------




Turing.@model function turing_car(D, W, X, log_offset, y, auid )
    # base model .. slow, MVN of var-covariance matrix .. didactic version
    nX=size(X,2)
    A ~ Uniform(0.0, 1.0); # A = 0.9 ; A==1 for BYM / iCAR
    tau ~ Gamma(2.0, 1.0/2.0);  # tau=0.9
    beta ~ filldist( Normal(0.0, 1.0), nX );
    prec = tau .* (D - A .* W)
    if !isposdef(prec)
        # check postive definiteness
        # phi ~ MvNormal( zeros(N) ); 
        Turing.@addlogprob! -Inf
        return nothing
    end
    sigma = inv( Symmetric(prec) )
    # sigma = Symmetric( prec) \ Diagonal(ones(nX))  # alternatively
    phi ~ MvNormal( sigma );  # mean zero
    mu =  X * beta .+ phi[auid] .+ log_offset 
    @. y ~ LogPoisson( mu );
end

   
Turing.@model function turing_car_prec(D, W, X, log_offset, y,  auid  )
    # MVN of precision matrix .. slightly faster
    nX=size(X,2)
    A ~ Uniform(0.0, 1.0); # A = 0.9 ; A==1 for BYM / iCAR
    tau ~ Gamma(2.0, 1.0/2.0);  # tau=0.9
    beta ~ filldist( Normal(0.0, 1.0), nX);
    prec = tau .* (D - A .* W)
    if !isposdef(prec)
        # check postive definiteness
        # phi ~ MvNormal( zeros(N) ); 
        Turing.@addlogprob! -Inf
        return nothing
    end
    phi ~ MvNormalCanon( Symmetric(prec) );  # mean zero .. no inverse
    mu =  X * beta .+ phi[auid] .+ log_offset 
    @. y ~ LogPoisson( mu );
end
 

Turing.@model function turing_icar_test( node1, node2; ysd=std(skipmissing(y)), nData=size(y,1)  )
    # equivalent to Morris' "simple_iar' .. testing pairwise difference formulation
    # see (https://mc-stan.org/users/documentation/case-studies/icar_stan.html)

    phi ~ filldist( Normal(0.0, ysd), nData)   # 10 is std from data: std(y)=7.9 stan goes from U(-Inf,Inf) .. not sure why 
    dphi = phi[node1] - phi[node2]
    lp_phi =  -0.5 * dot( dphi, dphi )
    Turing.@addlogprob! lp_phi
    
    # soft sum-to-zero constraint on phi)
    # equivalent to mean(phi) ~ normal(0,0.001)
    sum_phi = sum(phi)
    sum_phi ~ Normal(0, 0.001 * nData);  
  
    # no data likelihood -- just prior sampling  -- 
end

  
Turing.@model function turing_icar_bym( X, log_offset, y, nX, node1, node2; ysd=std(skipmissing(y)),  nData=size(X,1), auid=1:nData )
    # BYM
    # A ~ Uniform(0.0, 1.0); # A = 0.9 ; A==1 for BYM / iCAR
     # tau ~ Gamma(2.0, 1.0/2.0);  # tau=0.9
     beta ~ filldist( Normal(0.0, 5.0), nX);
     theta ~ filldist( Normal(0.0, 1.0), nData) # unstructured (heterogeneous effect)
     # phi ~ filldist( Laplace(0.0, ysd), nData) # spatial effects: stan goes from -Inf to Inf .. 
     phi ~ filldist( Normal(0.0, ysd), nData) # spatial effects: stan goes from -Inf to Inf .. 
 
     # pairwise difference formulation ::  prior on phi on the unit scale with sd = 1
     # see (https://mc-stan.org/users/documentation/case-studies/icar_stan.html)
     dphi = phi[node1] - phi[node2]
     lp_phi =  -0.5 * dot( dphi, dphi )
     Turing.@addlogprob! lp_phi
     
     # soft sum-to-zero constraint on phi)
     # equivalent to mean(phi) ~ normal(0, 0.001)
     sum_phi = sum(phi)
     sum_phi ~ Normal(0, 0.001 * nData);  

     tau_theta ~ Gamma(3.2761, 1.0/1.81);  # Carlin WinBUGS priors
     tau_phi ~ Gamma(1.0, 1.0);            # Carlin WinBUGS priors

     sigma_theta = inv(sqrt(tau_theta));  # convert precision to sigma
     sigma_phi = inv(sqrt(tau_phi));      # convert precision to sigma

     mu =  X * beta .+ phi[auid] .* sigma_phi .+ theta .* sigma_theta .+ log_offset 
  
     @. y ~ LogPoisson( mu );
end
 

Turing.@model function turing_icar_bym2( X, log_offset, y, auid, nX, nAU, node1, node2, scaling_factor )
    beta ~ filldist( Normal(0.0, 1.0), nX);
    theta ~ filldist( Normal(0.0, 1.0), nAU)  # unstructured (heterogeneous effect)
    phi ~ filldist( Normal(0.0, 1.0), nAU) # spatial effects: stan goes from -Inf to Inf .. 
    # pairwise difference formulation ::  prior on phi on the unit scale with sd = 1
    # see (https://mc-stan.org/users/documentation/case-studies/icar_stan.html)
    dphi = phi[node1] - phi[node2]
    Turing.@addlogprob! -0.5 * dot( dphi, dphi )
    # soft sum-to-zero constraint on phi)
    sum_phi = sum(phi)
    sum_phi ~ Normal(0, 0.001 * nAU);  
    sigma ~ truncated( Normal(0, 1.0), 0, Inf) ; 
    rho ~ Beta(0.5, 0.5);
    # variance of each component should be approximately equal to 1
    convolved_re =  sigma .*  ( sqrt.(1 .- rho) .* theta .+ sqrt.(rho ./ scaling_factor) .* phi );
    mu =  X * beta +  convolved_re[auid] + log_offset 
    @. y ~ LogPoisson( mu );
end
 

Turing.@model function turing_icar_bym2_binomial(y, Ntrial, X, nX, nAU, node1, node2, scaling_factor)
    # poor form to not pass args directly but simpler to use global vars as this is a one-off:
    beta0 ~ Normal(0.0, 1.0)
    betas ~ filldist( Normal(0.0, 1.0), nX); #coeff
    theta ~ filldist( Normal(0.0, 1.0), nAU)  # unstructured (heterogeneous effect)
    phi ~ filldist( Normal(0.0, 1.0), nAU) # spatial effects: stan goes from -Inf to Inf .. 
    dphi = phi[node1] - phi[node2]
    Turing.@addlogprob! (-0.5 * dot( dphi, dphi )) # directly add to logprob
    sum_phi = sum(phi)
    sum_phi ~ Normal(0.0, 0.001 * nAU);      # soft sum-to-zero constraint on phi), equivalent to 
    sigma ~ Gamma(1.0, 1.0)
    rho ~ Beta(0.5, 0.5);
    # variance of each component should be approximately equal to 1
    convolved_re =  sqrt(1 - rho) .* theta .+ sqrt(rho / scaling_factor) .* phi ;
    mu = beta0 .+ X * betas .+ sigma .* convolved_re 
    # y ~ arraydist(LazyArray(@~ BinomialLogit.(Ntrial, v)))  # 100 sec
    @. y ~ BinomialLogit(Ntrial, mu)
end



Turing.@model function turing_icar_bym2_groups( X, log_offset, y, auid, nX, nAU, node1, node2, scaling_factor, groups, gi; ysd=std(skipmissing(y)) )

    ## incomplete group effect ... 
 
    # BYM2
    # A ~ Uniform(0.0, 1.0); # A = 0.9 ; A==1 for BYM / iCAR
     # tau ~ Gamma(2.0, 1.0/2.0);  # tau=0.9
     beta ~ filldist( Normal(0.0, 5.0), nX);
     theta ~ filldist( Normal(0.0, 1.0), nAU)  # unstructured (heterogeneous effect)
     phi ~ filldist(Normal(0.0, ysd), nAU) # spatial effects: stan goes from -Inf to Inf .. 
        
     # pairwise difference formulation ::  prior on phi on the unit scale with sd = 1
     # see (https://mc-stan.org/users/documentation/case-studies/icar_stan.html)
     dphi = phi[node1] - phi[node2]
     lp_phi =  -0.5 * dot( dphi, dphi )
     Turing.@addlogprob! lp_phi
     
     sigma ~ truncated( Normal(0, 1.0), 0, Inf) ; 
     rho ~ Beta(0.5, 0.5);

     convolved_re = zeros(nAU)

     # Threads.@threads add once working
     for j in 1:length(gi)
         ic = gi[j] 
        
         # soft sum-to-zero constraint on phi)
         # equivalent to mean(phi) ~ normal(0, 0.001)
         sum_phi = sum(phi[ic])
         sum_phi ~ Normal(0, 0.001 * nAU);  

         if  length(ic) == 1 
             convolved_re[ ic ] = sigma .* theta[ ic ];
         else  
             convolved_re[ ic ] = sigma .* ( sqrt.(1 .- rho) .* theta[ ic ]  +  sqrt(rho ./ scaling_factor[j] )  .* phi[ ic ] ) ;
         end 
     end
  
     # convolved_re =  sqrt.(1 .- rho) .* theta .+ sqrt.(rho ./ scaling_factor) .* phi;
   
     mu =   X * beta .+  convolved_re[auid] .+ log_offset 
   
     @. y ~ LogPoisson( mu );
  
    # to compute from posteriors
    #  real logit_rho = log(rho / (1.0 - rho));
    #  vector[N] eta = log_E + beta0 + x * betas + convolved_re * sigma; // co-variates
    #  vector[N] mu = exp(eta);
end 





Turing.@model function ar1_gp(  ::Type{T}=Float64; Y, ar1,  nData=length(Y), nT=Integer(maximum(ar1)-minimum(ar1)+1) ) where {T} 
    Ymean = mean(Y)
    rho ~ truncated(Normal(0,1), -1, 1)
    ar1_process_error ~ LogNormal(0, 1) 
    var_ar1 =  ar1_process_error^2 / (1 - rho^2)
    vcv = ar1_covariance(nT, rho, var_ar1, T)  # -- covariance by time
    ymean_ar1 ~ MvNormal(Symmetric(vcv) );  # -- means by time 
    observation_error ~ LogNormal(0, 1) 
    Y ~ MvNormal( ymean_ar1[ar1[1:nData]] .+ Ymean, observation_error )     # likelihood
end
 


Turing.@model function ar1_gp_local(  ::Type{T}=Float64; Y, ar1,  nData=length(Y), nT=Integer(maximum(ar1)-minimum(ar1)+1) ) where {T} 
    Ymean = mean(Y)
    rho ~ truncated(Normal(0,1), -1, 1)
    ar1_process_error ~ LogNormal(0, 1) 
    var_ar1 =  ar1_process_error^2 / (1 - rho^2)
    vcv = ar1_covariance_local(nT, rho, var_ar1, T)  # -- covariance by time
    ymean_ar1 ~ MvNormal(Symmetric(vcv) );  # -- means by time 
    observation_error ~ LogNormal(0, 1) 
    Y ~ MvNormal( ymean_ar1[ar1[1:nData]] .+ Ymean, observation_error )     # likelihood
end
 

Turing.@model function ar1_recursive(; Y, ar1,  nData=length(Y), nT=Integer(maximum(ar1)-minimum(ar1)+1) )
    Ymean = mean(Y)
    alpha_ar1 ~ Normal(0,1)
    rho ~ truncated(Normal(0,1), -1, 1)
    ar1_process_error ~ LogNormal(0, 1) 
    ymean_ar1 = tzeros(nT);  # -- means by time 
    ymean_ar1[1] ~ Normal(Ymean, ar1_process_error) 
    for t in 2:nT
        ymean_ar1[t] ~ Normal(alpha_ar1 + rho * ymean_ar1[t-1], ar1_process_error );
    end
    observation_error ~ LogNormal(0, 1) 
    Y ~ MvNormal( ymean_ar1[ar1[1:nData]] .+ Ymean, observation_error )     # likelihood
end
 


Turing.@model function SparseFinite_example( Yobs, Xobs, Xinducing )
    nInducing = length(Xinducing)
 
    # m ~ filldist( Normal(0, 100), nInducing ) 
    # A ~ filldist( Normal(), nInducing, nInducing ) 
    # S = PDMat(Cholesky(LowerTriangular(A)))
 
    kernel_var ~ Gamma(0.5, 1.0)
    kernel_scale ~ Gamma(2.0, 1.0)
    lambda = 0.001
    
    fkernel = kernel_var * Matern52Kernel() ∘ ScaleTransform(kernel_scale) # ∘ ARDTransform(α)
         
    fgp = atomic(GP(fkernel), GPC())
    fobs = fgp( Xobs, lambda )
    finducing = fgp( Xinducing, lambda ) 
    fsparse = SparseFiniteGP(fobs, finducing)
    Turing.@addlogprob! -Stheno.elbo( fsparse, Yobs ) 
    # GPpost = posterior(fsparse, Yobs)
 
    # m ~ GPpost(Xinducing, lambda) 
  
    # Yobs ~ GPpost(Xobs, lambda)
      
end



Turing.@model function SparseVariationalApproximation_example( Yobs, Xobs, Xinducing, lambda = 0.001)
    nInducing = length(Xinducing)

    m ~ filldist( Normal(0, 100), nInducing ) 
  
    # variance process
    # Efficiently constructs S as A*Aᵀ
    A ~ filldist( Normal(), nInducing, nInducing ) 
    S = PDMat(Cholesky(LowerTriangular(A)))
 
    kernel_var ~ Gamma(0.5, 1.0)
    kernel_scale ~ Gamma(2.0, 1.0)
    
    fkernel = kernel_var * Matern52Kernel() ∘ ScaleTransform(kernel_scale) # ∘ ARDTransform(α)

    fgp = atomic( GP(fkernel), GPC()) 

    finducing = fgp(Xinducing, lambda) # aka "prior" in AbstractGPs
    fsparse = SparseVariationalApproximation(finducing, MvNormal(m, S))
    
    # Turing.@addlogprob! -Stheno.elbo(fsparse, fobs, Yobs )  # failing here, 

    fposterior = posterior(fsparse, finducing, Yobs)
    
    o = fposterior(Xobs)
    Yobs ~ MvNormal( mean(o), Symmetric(cov(o)) + I*lambda )

end



 

Turing.@model function pca_carstm( Y, ::Type{T}=Float64 ) where {T}
  # mulitple likelihood model .. 
  # X, G, log_offset, y, z, auid, nData, nX, nG, nAU, node1, node2, scaling_factor 
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
  s_phi ~ filldist( Normal(0.0, 1.0), nAU, nz) # spatial effects: stan goes from -Inf to Inf .. 
    
  s_sigma ~ filldist( LogNormal(0.0, 1.0), nz) ; 
  s_rho ~ filldist(Beta(0.5, 0.5), nz);
        
  eta ~ filldist( LogNormal(0.0, 1.0), nz ) # overall observation variance
 
  # spatial effects (without inverting covariance) 
  dphi = phi[node1] - phi[node2]
  dot_phi = dot( dphi, dphi )
  Turing.@addlogprob! -0.5 * dot_phi

  # soft sum-to-zero constraint on phi
  sum_phi_s = sum( dot_phi ) 
  sum_phi_s ~ Normal(0, 0.001 * nAU);      # soft sum-to-zero constraint on s_phi)

  convolved_re_s = icar_form( s_theta[:,z], s_phi[:,z], s_sigma[z], s_rho[z] )
  Turing.@addlogprob! -0.5 * dot_phi
  sum_phi_s ~ Normal(0, 0.001 * nAU_float);      # soft sum-to-zero constraint on s_phi)
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
        s_phi ~ filldist( Normal(0.0, 1.0), nAU) # spatial effects: stan goes from -Inf to Inf .. 
        dphi_s = s_phi[node1] - s_phi[node2]
        Turing.@addlogprob! (-0.5 * dot( dphi_s, dphi_s ))
        sum_phi_s = sum(s_phi) 
        sum_phi_s ~ Normal(0, 0.001 * nAU);      # soft sum-to-zero constraint on s_phi)
        s_sigma ~ truncated( Normal(0.0, 1.0), 0, Inf) ; 
        s_rho ~ Beta(0.5, 0.5);
        # spatial effects:  nAU
        convolved_re_s = s_sigma .*( sqrt.(1 .- s_rho) .* s_theta .+ sqrt.(s_rho ./ scaling_factor) .* s_phi )
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
    s_phi ~ filldist( Normal(0.0, 1.0), nAU) # spatial effects: stan goes from -Inf to Inf .. 
    dphi_s = s_phi[node1] - s_phi[node2]
    Turing.@addlogprob! (-0.5 * dot( dphi_s, dphi_s ))
    sum_phi_s = sum(s_phi) 
    sum_phi_s ~ Normal(0, 0.001 * nAU);      # soft sum-to-zero constraint on s_phi)
    s_sigma ~ truncated( Normal(0.0, 1.0), 0, Inf) ; 
    s_rho ~ Beta(0.5, 0.5);

    # spatial effects:  nAU
    convolved_re_s = s_sigma .*( sqrt.(1 .- s_rho) .* s_theta .+ sqrt.(s_rho ./ scaling_factor) .* s_phi )
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
        sum_phi ~ Normal(0.0, 0.01 * nAU);      # soft sum-to-zero constraint on s_phi)
        
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
    sum_phi ~ Normal(0.0, 0.001 * nAU);      # soft sum-to-zero constraint on s_phi)
    
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
 