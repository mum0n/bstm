
@model function example_bym2_ar1_poisson(M, ::Type{T}=Float64) where {T}

    # --- 1. GLOBAL HYPERPRIORS ---
    s_sigma ~ Exponential(1.0)
    s_rho ~ Beta(1, 1)
    t_sigma ~ Exponential(1.0)
    t_rho ~ Beta(2, 2)
    
    # Fixed effects coefficients (aligned with UnivariateArchitecture)
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), 10.0 * I)

    # --- 2. SPATIAL COMPONENT (BYM2) ---
    # Uses s_Q_template from the updated bstm_options structure
    s_icar ~ MvNormal(zeros(M.s_N), I)
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q_template * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    
    s_latent = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # --- 3. TEMPORAL COMPONENT (AR1) ---
    # Uses t_Q_template from the updated bstm_options structure
    t_prec_mat = (1.0 / (1.0 - t_rho^2 + 1e-6)) .* (M.t_Q_template + (t_rho^2) * I)
    t_raw ~ MvNormal(zeros(M.t_N), I)
    Turing.@addlogprob! -0.5 * dot(t_raw, t_prec_mat * t_raw)

    t_latent = t_raw .* t_sigma

    # --- 4. LIKELIHOOD ---
    for i in 1:M.y_N
        s_idx = M.s_idx[i]
        t_idx = M.t_idx[i]

        # Fixed effects contribution
        linpred_fixed = dot(M.Xfixed[i, :], Xfixed_beta)
        
        # Total linear predictor
        eta = M.log_offset[i] + linpred_fixed + s_latent[s_idx] + t_latent[t_idx]

        Turing.@addlogprob! M.weights[i] * logpdf(Poisson(exp(eta)), M.y_obs[i])
    end
end

@model function example_kriging_simple(M, ::Type{T}=Float64) where {T}
    # Aligning global mean with Xfixed_beta (standardized for reconstruction)
    # Usually Xfixed_N=1 for just an intercept
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), 10.0 * I)

    # Hyperparameters for spatial GP per time step
    signal_std_dev ~ filldist(Exponential(1.0), M.t_N)
    spatial_lengthscale ~ filldist(Gamma(2.0, 0.5), M.t_N)
    obs_std_dev ~ filldist(Exponential(1.0), M.t_N)

    # s_latent: standard name for spatial random effects
    s_latent = Vector{T}(undef, M.s_N * M.t_N) 
    
    # For simplicity in reconstruction, we map latent spatial values to s_latent
    # If the model is purely spatial-kriging per slice:
    f_latent_all = Vector{T}(undef, length(M.y_obs))

    for i in 1:M.t_N
        time_mask_i = findall(==(i), M.t_idx)

        if !isempty(time_mask_i)
            # Extract coordinates for the specific time slice
            coords_raw = M.s_coord[M.s_idx[time_mask_i]]
            X_obs_time_i = reduce(hcat, coords_raw)

            k_spatial_i = (signal_std_dev[i]^2) * SqExponentialKernel() ∘ ScaleTransform(1/spatial_lengthscale[i])
            f_spatial_i = GP(ZeroMean(), k_spatial_i)

            # Standard variable name for the spatial latent field
            # We sample them slice by slice
            f_latent_all[time_mask_i] ~ f_spatial_i(ColVecs(X_obs_time_i), 1e-6)
        end
    end

    # Map the realized values to a consistent s_latent vector if needed for the backend,
    # or rely on the combined linear predictor.
    # For reconstruction to work, we'll store the values in s_latent.
    # Here we'll treat f_latent_all as our primary spatial component.

    # Observation Likelihood using standardized fixed effects and spatial latent
    # M.Xfixed typically contains the intercept column (ones)
    linpred_fixed = M.Xfixed * Xfixed_beta
    combined_mean = linpred_fixed .+ f_latent_all

    M.y_obs ~ MvNormal(combined_mean, Diagonal(obs_std_dev[M.t_idx].^2 .+ 1e-6))
end


@model function example_gp_gaussian(M, ::Type{T}=Float64) where {T}
    # 1. Hyperparameters
    y_sigma ~ Exponential(1.0)
    st_sigma ~ Exponential(1.0)
    ls_s ~ Gamma(2, 2)
    ls_t ~ Gamma(2, 2)
    
    # Standardized fixed effects (aligned with UnivariateArchitecture)
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), 10.0 * I)

    # 2. Kernel Definitions
    k = (st_sigma^2) * (SqExponentialKernel() ∘ ScaleTransform(inv(ls_s))) ⊗ (SqExponentialKernel() ∘ ScaleTransform(inv(ls_t)))

    # 3. Define the GP
    f = GP(k)

    # 4. Construct Inducing Points Grid
    Ns = size(M.s_inducing, 2)
    Nt = size(M.t_inducing, 2)
    s_grid = repeat(M.s_inducing, outer=(1, Nt))
    t_grid = repeat(M.t_inducing, inner=(1, Ns))
    X_ind = ColVecs(vcat(s_grid, t_grid))

    # 5. Sparse GP Approximation
    f_u = f(X_ind, 1e-6)
    # Map latent GP values to s_latent for standardized reconstruction
    s_latent ~ f_u

    # 6. Conditional GP and Likelihood
    f_cond = condition(f_u, s_latent)

    # Map to observed space coordinates
    s_obs_full = M.s_coord.X'
    t_obs_full = reshape(M.t_coord, 1, :)
    X_obs_all = ColVecs(vcat(s_obs_full, t_obs_full))

    # 7. Linear Predictor
    linpred_fixed = M.Xfixed * Xfixed_beta
    mu_gp = mean(f_cond(X_obs_all[M.st_idx]))
    mu = M.log_offset .+ linpred_fixed .+ mu_gp

    # 8. Observation Likelihood
    M.y_obs ~ MvNormal(mu, y_sigma^2 * I)
end


@model function example_hurdle_bernoulli_poisson(M, ::Type{T}=Float64) where {T}
    # --- 1. GLOBAL HYPERPRIORS ---
    # Hurdle (Binary) Component Hyperparams
    s_sigma_h ~ Exponential(1.0); s_rho_h ~ Beta(1, 1)
    t_sigma_h ~ Exponential(1.0); t_rho_h ~ Beta(2, 2)
    
    # Count (Positive) Component Hyperparams
    s_sigma_c ~ Exponential(1.0); s_rho_c ~ Beta(1, 1)
    t_sigma_c ~ Exponential(1.0); t_rho_c ~ Beta(2, 2)
    
    # Fixed effects for hurdle and count (if applicable)
    # Assuming M.Xfixed_N applies to both or separate intercepts
    beta_fixed_h ~ MvNormal(zeros(M.Xfixed_N), 10.0 * I)
    beta_fixed_c ~ MvNormal(zeros(M.Xfixed_N), 10.0 * I)

    # --- 2. LATENT FIELDS FOR HURDLE PART ---
    # Standardized spatial component
    s_icar_h ~ MvNormal(zeros(M.s_N), I); Turing.@addlogprob! -0.5 * dot(s_icar_h, M.s_Q_template * s_icar_h)
    s_iid_h ~ MvNormal(zeros(M.s_N), I)
    s_latent_h = s_sigma_h .* (sqrt(s_rho_h) .* s_icar_h .+ sqrt(1 - s_rho_h) .* s_iid_h)

    # Standardized temporal component
    t_prec_h = (1.0 / (1.0 - t_rho_h^2 + 1e-6)) .* (M.t_Q_template + (t_rho_h^2) * I)
    t_raw_h ~ MvNormal(zeros(M.t_N), I); Turing.@addlogprob! -0.5 * dot(t_raw_h, t_prec_h * t_raw_h)
    t_latent_h = t_raw_h .* t_sigma_h

    # --- 3. LATENT FIELDS FOR COUNT PART ---
    s_icar_c ~ MvNormal(zeros(M.s_N), I); Turing.@addlogprob! -0.5 * dot(s_icar_c, M.s_Q_template * s_icar_c)
    s_iid_c ~ MvNormal(zeros(M.s_N), I)
    s_latent_c = s_sigma_c .* (sqrt(s_rho_c) .* s_icar_c .+ sqrt(1 - s_rho_c) .* s_iid_c)

    t_prec_c = (1.0 / (1.0 - t_rho_c^2 + 1e-6)) .* (M.t_Q_template + (t_rho_c^2) * I)
    t_raw_c ~ MvNormal(zeros(M.t_N), I); Turing.@addlogprob! -0.5 * dot(t_raw_c, t_prec_c * t_raw_c)
    t_latent_c = t_raw_c .* t_sigma_c

    # --- 4. JOINT LIKELIHOOD ---
    for i in 1:M.y_N
        a, t = M.s_idx[i], M.t_idx[i]

        # Linear predictors using standardized names
        eta_h = dot(M.Xfixed[i, :], beta_fixed_h) + s_latent_h[a] + t_latent_h[t]
        eta_c = M.log_offset[i] + dot(M.Xfixed[i, :], beta_fixed_c) + s_latent_c[a] + t_latent_c[t]

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


@model function example_rff_2D(M, ::Type{T}=Float64 ) where {T}
    # SPDE-style continuous spatial field using spectral RFF approximation.

    # --- 1. GLOBAL HYPERPRIORS ---
    y_sigma ~ Exponential(1.0)
    s_sigma ~ Exponential(1.0)
    kappa_sp ~ Gamma(2, 1) 
    t_sigma ~ Exponential(1.0)
    t_rho ~ Beta(2, 2)
    
    # Standardized fixed effects (aligned with UnivariateArchitecture)
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), 10.0 * I)

    # --- 2. LATENT FIELDS ---
    # s_latent: spatial RFF weights
    s_latent ~ MvNormal(zeros(M.m_spatial), I)
    
    # t_latent: AR1 temporal field
    t_prec = (1.0 / (1.0 - t_rho^2 + 1e-6)) .* (M.t_Q_template + (t_rho^2) * I)
    t_raw ~ MvNormal(zeros(M.t_N), I)
    Turing.@addlogprob! -0.5 * dot(t_raw, t_prec * t_raw)
    t_latent = t_raw .* t_sigma

    # --- 3. SPATIAL BASIS CONSTRUCTION ---
    xs = [p[1] for p in M.s_coord_tuple]
    ys = [p[2] for p in M.s_coord_tuple]
    coords = hcat(
        (xs .- mean(xs)) ./ (std(xs) + 1e-6),
        (ys .- mean(ys)) ./ (std(ys) + 1e-6)
    )

    Random.seed!(42)
    Om = randn(M.m_spatial, 2) .* kappa_sp
    Ph = rand(M.m_spatial) .* convert(T, 2pi)

    Z_sp = convert(T, sqrt(2/M.m_spatial)) .* cos.(coords * Om' .+ Ph')
    spatial_field = Z_sp * (s_latent .* s_sigma)

    # --- 4. LIKELIHOOD ---
    linpred_fixed = M.Xfixed * Xfixed_beta
    
    # Combine fixed effects, continuous spatial field, and temporal lattice effect
    mu = M.log_offset .+ linpred_fixed .+ spatial_field .+ t_latent[M.t_idx]

    M.y_obs ~ MvNormal(mu, y_sigma^2 * I)
end


@model function example_rff_3D(M, ::Type{T}=Float64; m_joint=25 ) where {T}
    # Non-Separable Spatiotemporal RFF model aligned with UnivariateArchitecture
    
    # --- 1. GLOBAL HYPERPRIORS ---
    y_sigma ~ Exponential(1.0)
    sigma_joint ~ Exponential(1.0)
    # Lengthscales for X, Y, and Time dimensions within the joint kernel
    l_joint ~ filldist(Gamma(2, 1), 3) 
    
    # Standardized fixed effects (aligned with UnivariateArchitecture)
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), 10.0 * I)
    
    # Random features weights - mapped to s_latent for standardized reconstruction
    s_latent ~ MvNormal(zeros(m_joint), I)

    # --- 2. FEATURE MATRIX CONSTRUCTION ---
    # X_joint: [normalized_x, normalized_y, normalized_time]
    xs = [p[1] for p in M.s_coord_tuple]
    ys = [p[2] for p in M.s_coord_tuple]
    ts = Float64.(M.t_idx)
    
    # Normalize inputs to [0, 1] range for numerical stability
    X_joint = hcat(
        (xs .- minimum(xs)) ./ (maximum(xs) - minimum(xs) + 1e-6),
        (ys .- minimum(ys)) ./ (maximum(ys) - minimum(ys) + 1e-6),
        (ts .- minimum(ts)) ./ (maximum(ts) - minimum(ts) + 1e-6)
    )

    # --- 3. JOINT RFF PROJECTION ---
    # Non-separable interaction via random Fourier features
    Random.seed!(42)
    Om = randn(m_joint, 3) .* (1.0 ./ l_joint')
    Ph = rand(m_joint) .* convert(T, 2pi)
    
    Z_joint = convert(T, sqrt(2/m_joint)) .* cos.(X_joint * Om' .+ Ph')
    eta_joint = Z_joint * (s_latent .* sigma_joint)

    # --- 4. LIKELIHOOD ---
    # Using M.Xfixed for intercepts/covariates to satisfy reconstruction logic
    linpred_fixed = M.Xfixed * Xfixed_beta
    
    mu = M.log_offset .+ linpred_fixed .+ eta_joint
    
    # Observation Likelihood
    M.y_obs ~ MvNormal(mu, y_sigma^2 * I)
end




@model function example_warping_2D(M, ::Type{T}=Float64) where {T}

    # --- 1. Priors ---
    y_sigma ~ Exponential(1.0)
    s_sigma ~ Exponential(1.0)
    l_warp ~ Gamma(2, 1)    # Smoothness of the warping manifold
    l_spatial ~ Gamma(2, 1) # Smoothness of the stationary field in warped space

    # Standardized Naming: Xfixed_beta, s_latent, t_latent
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), I)
    w_warp ~ MvNormal(zeros(M.m_warp), I)
    s_latent ~ MvNormal(zeros(M.m_spatial), I)
    
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
    Random.seed!(45)
    Om_s = randn(M.m_spatial, 1) ./ l_spatial
    Ph_s = rand(M.m_spatial) .* convert(T, 2pi)

    Z_sp = convert(T, sqrt(2/M.m_spatial)) .* cos.(reshape(warped_coords, :, 1) * Om_s' .+ Ph_s')
    s_eta = Z_sp * (s_latent .* s_sigma)

    # --- 5. Temporal & Categorical Components ---
    t_Q = (1.0 / (1.0 - t_rho^2)) .* (M.t_Q_template + (t_rho^2) * I)
    t_latent ~ MvNormal(zeros(M.t_N), I)
    Turing.@addlogprob! -0.5 * dot(t_latent, t_Q * t_latent)
    t_eta = t_latent .* t_sigma

    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 6. Likelihood ---
    fixed_effect = M.Xfixed * Xfixed_beta
    for i in 1:M.y_N
        mu = M.log_offset[i] + fixed_effect[i] + s_eta[i] + t_eta[M.t_idx[i]]
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), M.y_obs[i])
    end
end


@model function example_rff_2D_mosaic(M, ::Type{T}=Float64 ) where {T}
      
    # --- 1. Global & Hierarchical Priors ---
    c_sigma ~ filldist(Exponential(1.0), 4)
    
    # Standardized Naming: Xfixed_beta for global intercept/fixed effects
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), I)
    sigma_mu_local ~ Exponential(1.0)

    # Local Parameters per Mosaic
    mu_local ~ filldist(Normal(Xfixed_beta[1], sigma_mu_local), M.n_mosaics)
    l_local ~ filldist(Gamma(2, 1), M.n_mosaics)
    sigma_local ~ filldist(Exponential(1.0), M.n_mosaics)
    y_sigma_local ~ filldist(Exponential(1.0), M.n_mosaics) # Localized noise scale

    # Standardized Naming: s_latent for spatial random effects
    s_latent = [Vector{T}(undef, M.m_rff) for _ in 1:M.n_mosaics]
    for m in 1:M.n_mosaics; s_latent[m] ~ MvNormal(zeros(M.m_rff), I); end

    # Categorical Covariates (Shared)
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 2. Spatial Indexing & Soft Boundary Weights ---
    coords = hcat([p[1] for p in M.s_coord_tuple], [p[2] for p in M.s_coord_tuple])
    # Note: centroids should ideally be pre-calculated and passed in M to keep the model pure
    R = kmeans(coords', M.n_mosaics)
    centroids = R.centers 

    # Pre-sample RFF frequencies
    Random.seed!(42)
    Om_m = [randn(M.m_rff, 2) for _ in 1:M.n_mosaics]
    Ph_m = [rand(M.m_rff) for _ in 1:M.n_mosaics]

    # --- 3. Likelihood with Soft Integration ---
    for i in 1:M.y_N
        pt = [coords[i,1], coords[i,2]]
        
        # Calculate Softmax weights based on distance to centroids for smooth stitching
        dists = [sum((pt .- centroids[:, m]).^2) for m in 1:M.n_mosaics]
        max_d = maximum(-dists)
        weights_stitching = exp.(-dists .- max_d) ./ (sum(exp.(-dists .- max_d)) + 1e-9)
     
        eta_spatial_combined = zero(T)
        y_sigma_combined = zero(T)
        
        for m in 1:M.n_mosaics
            # Local RFF Field Calculation
            z_proj = sqrt(2/M.m_rff) .* cos.( (Om_m[m] * pt ./ l_local[m]) .+ (Ph_m[m] .* 2pi) )
            local_field = mu_local[m] + dot(z_proj, s_latent[m] .* sigma_local[m])
            
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
    
    # Standardized Naming: Xfixed_beta for global intercept/fixed effects
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), I)
    sigma_mu_local ~ Exponential(0.5)

    # Shared Categorical Effects
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 2. Local Mosaic Hyperparameters ---
    mu_local ~ filldist(Normal(Xfixed_beta[1], sigma_mu_local), M.n_mosaics)

    # Refactored: Use arraydist for joint lengthscales instead of a loop
    l_joint ~ arraydist([filldist(Gamma(2, 1), 3) for _ in 1:M.n_mosaics])

    sigma_local ~ filldist(Exponential(1.0), M.n_mosaics)
    y_sigma_local ~ filldist(Exponential(1.0), M.n_mosaics)

    # Standardized Naming: s_latent for local mosaic RFF weights
    s_latent = [Vector{T}(undef, M.m_rff) for _ in 1:M.n_mosaics]
    for m in 1:M.n_mosaics; s_latent[m] ~ MvNormal(zeros(M.m_rff), I); end

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

    # Static centroids for stitching
    coords_2d = X_joint[:, 1:2]
    R = kmeans(coords_2d', M.n_mosaics)
    centroids = R.centers

    # Fixed RFF Frequencies
    Random.seed!(42)
    Om_base = [randn(M.m_rff, 3) for _ in 1:M.n_mosaics]
    Ph_base = [rand(M.m_rff) for _ in 1:M.n_mosaics]

    # --- 4. Predictive Synthesis ---
    fixed_effect = M.Xfixed * Xfixed_beta
    
    for i in 1:M.y_N
        pt_3d = X_joint[i, :]
        pt_2d = pt_3d[1:2]

        # Soft Boundary Weights
        dists = [sum((pt_2d .- centroids[:, m]).^2) for m in 1:M.n_mosaics]
        weights_st = exp.(-dists) ./ sum(exp.(-dists))

        eta_spatial_time = zero(T)
        y_sigma_total = zero(T)

        for m in 1:M.n_mosaics
            # Scale base frequencies by local lengthscales [Lx, Ly, Lt]
            Om = Om_base[m] .* (1.0 ./ (l_joint[:, m] .+ M.noise)')

            # Local Non-Separable Field calculation using s_latent
            z_proj = sqrt(2/M.m_rff) * cos.( (Om * pt_3d) .+ (Ph_base[m] .* 2pi) )
            local_field = mu_local[m] + dot(z_proj, s_latent[m] .* sigma_local[m])

            eta_spatial_time += weights_st[m] * local_field
            y_sigma_total += weights_st[m] * y_sigma_local[m]
        end

        mu = M.log_offset[i] + fixed_effect[i] + eta_spatial_time
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

    # Standardized Naming: Xfixed_beta for fixed effects
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), I)

    # Component 1: BYM2 (Standardized Naming: s_latent part 1 & 2)
    s_icar ~ MvNormal(zeros(M.s_N), I)
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q_template * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_eta_bym2 = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # Component 2: AR1 (Standardized Naming: t_latent)
    t_Q = (1.0 / (1.0 - t_rho^2)) .* (M.t_Q_template + (t_rho^2) * I)
    t_latent ~ MvNormal(zeros(M.t_N), I)
    Turing.@addlogprob! -0.5 * dot(t_latent, t_Q * t_latent)
    t_eta = t_latent .* t_sigma

    # Component 3: Seasonal
    t_vec = Float64.(M.t_idx)
    seasonal = beta_cos .* cos.(2pi .* t_vec ./ M.period) .+ beta_sin .* sin.(2pi .* t_vec ./ M.period)

    # Component 4: GMRF Interaction
    st_raw ~ MvNormal(zeros(M.s_N * M.t_N), I)
    st_eta = reshape(st_raw .* st_sigma, M.s_N, M.t_N)

    # Component 5: Fixed Inducing FITC (Standardized Naming: s_latent part 3)
    k_st = (f_sigma^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(st_ls)))
    K_XZ = kernelmatrix(k_st, RowVecs(M.st_coord_normalized), RowVecs(M.Z_inducing))
    s_latent_gp ~ MvNormal(zeros(size(M.Z_inducing, 1)), I)
    f_gp = K_XZ * s_latent_gp # Linear projection

    # RW2 Categorical
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:4]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # Likelihood
    fixed_effect = M.Xfixed * Xfixed_beta
    for i in 1:M.y_N
        idx_a = M.s_idx[i]
        idx_t = M.t_idx[i]
        mu = M.log_offset[i] + fixed_effect[i] + f_gp[i] + s_eta_bym2[idx_a] + t_eta[idx_t] + seasonal[i] + st_eta[idx_a, idx_t]
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

    # Standardized Naming: Xfixed_beta for fixed effects and intercepts
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), I)
    z_beta = Xfixed_beta[2] # Example mapping if second term is z coefficient

    # --- 2. GP Temporal Trend (Standardized Naming: t_latent) ---
    ls_trend ~ Gamma(2, 2); sigma_trend ~ Exponential(0.5)
    k_trend = (sigma_trend^2) * (SqExponentialKernel() ∘ ScaleTransform(inv(ls_trend)))
    t_unique = collect(1:M.t_N) ./ M.t_N
    t_latent ~ MvNormal(zeros(M.t_N), kernelmatrix(k_trend, t_unique) + M.noise*I)
    alpha_gp = t_latent

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

    # --- 4. Spatial Effect (BYM2) (Standardized Naming: s_latent part 1 & 2) ---
    s_icar ~ MvNormal(zeros(M.s_N), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q_template * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_eta_bym2 = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # --- 5. SVGP Mean (Learned Inducing Points) (Standardized Naming: s_latent part 3) ---
    Z_inducing = Matrix{T}(undef, M.M_inducing_val, M.st_D)
    for j in 1:M.st_D
        Z_inducing[:, j] ~ filldist(Normal(mean(M.st_coord_normalized[:,j]), 2*std(M.st_coord_normalized[:,j])), M.M_inducing_val)
    end

    k_st = (f_sigma^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(st_ls)))
    K_ZZ = kernelmatrix(k_st, RowVecs(Z_inducing)) + M.noise * I
    K_XZ = kernelmatrix(k_st, RowVecs(M.st_coord_normalized), RowVecs(Z_inducing))
    K_XX_diag = diag(kernelmatrix(k_st, RowVecs(M.st_coord_normalized)))

    s_latent_gp ~ MvNormal(zeros(M.M_inducing_val), K_ZZ)
    f_mean = K_XZ * (K_ZZ \ s_latent_gp)
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
    beta_w_main ~ MvNormal(zeros(3), 2.0 * I)
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2pi .* t_norm * (M.t_N/M.period)) .+ beta_sin .* sin.(2pi .* t_norm * (M.t_N/M.period))

    fixed_effect = M.Xfixed * Xfixed_beta

    for i in 1:M.y_N
        a, t = M.s_idx[i], M.t_idx[i]
        mu = M.log_offset[i] + fixed_effect[i] + alpha_gp[t] + seasonal[i] + dot(beta_w_main, u_true_mat[i, :]) + f_gp[i] + s_eta_bym2[a]
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma[i]), M.y_obs[i])
        for k in 1:3; Turing.@addlogprob! logpdf(Normal(u_true_mat[i, k], w_sigma[k]), M.w_obs[i, k]); end
    end
end




@model function example_svgp_nested(M, ::Type{T}=Float64) where {T}
    # --- 1. Priors ---
    # Standardized fixed effects naming
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), 2.0 * I)
    
    # Latent process scales
    f_sigma ~ Exponential(1.0)
    st_ls ~ filldist(Gamma(2, 2), M.st_D)
    s_sigma ~ Exponential(1.0)
    s_rho ~ Beta(1, 1)
    t_sigma ~ Exponential(1.0)
    t_rho ~ Beta(1, 1)
    
    sigma_log_var ~ Exponential(1.0)

    # --- 2. Temporal Latent (t_latent) ---
    # AR1 or GP based on M.model_time configuration
    t_latent ~ MvNormal(zeros(M.t_N), I) # Placeholder for AR(1) innovation
    
    # --- 3. Spatial Latent (s_latent) ---
    s_icar ~ MvNormal(zeros(M.s_N), I)
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q_template * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    # Combined spatial effect (BYM2)
    s_latent = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # --- 4. SVGP Component (Spatiotemporal GP) ---
    # Inducing point locations (Z) and values (w_latent)
    Z_inducing = M.Z_inducing # Usually pre-calculated via Farthest Point Sampling
    
    m_u ~ MvNormal(zeros(M.M_inducing_val), 10.0 * I)
    s_u_diag ~ filldist(Exponential(1.0), M.M_inducing_val)
    w_latent ~ MvNormal(m_u, Diagonal(s_u_diag.^2) + M.noise*I)

    k_st = (f_sigma^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(st_ls)))
    K_ZZ = kernelmatrix(k_st, RowVecs(Z_inducing)) + M.noise * I
    K_XZ = kernelmatrix(k_st, RowVecs(M.st_coord_normalized), RowVecs(Z_inducing))
    
    # Project latent values to observations
    f_gp = K_XZ * (K_ZZ \ w_latent)

    # --- 5. Volatility / Noise ---
    log_y_sigma ~ Normal(0, sigma_log_var)
    y_sigma = exp(log_y_sigma / 2.0)

    # --- 6. Likelihood ---
    # Using standardized reconstruction fields
    Xfixed = M.Xfixed # Design matrix
    mu_fixed = Xfixed * Xfixed_beta

    for i in 1:M.y_N
        s_idx, t_idx = M.s_idx[i], M.t_idx[i]
        
        # Combine effects: Fixed + Spatial + Temporal + GP
        eta = mu_fixed[i] + s_latent[s_idx] + t_latent[t_idx] + f_gp[i] + M.log_offset[i]
        
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(eta, y_sigma), M.y_obs[i])
    end
end

@model function example_multifidelity(M, ::Type{T}=Float64) where {T}
    # --- 1. Priors & Parameters ---
    y_sigma ~ Exponential(1.0)
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    t_sigma ~ Exponential(1.0); t_rho ~ Beta(2, 2)
    st_sigma ~ Exponential(0.5)
    sigma_z ~ Exponential(0.5); w_sigma ~ filldist(Exponential(0.5), 3)
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    
    # Standardized fixed effects naming
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), 2.0 * I)

    # --- 2. Spatial Effect (BYM2) ---
    s_icar ~ MvNormal(zeros(M.s_N), I)
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q_template * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_latent = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # --- 3. Temporal Effect (AR1) ---
    t_Q = (1.0 / (1.0 - t_rho^2)) .* (M.t_Q_template + (t_rho^2) * I)
    t_raw ~ MvNormal(zeros(M.t_N), I)
    Turing.@addlogprob! -0.5 * dot(t_raw, t_Q * t_raw)
    t_latent = t_raw .* t_sigma

    # --- 4. Space-Time Interaction (GMRF Type IV) ---
    st_raw ~ MvNormal(zeros(M.s_N * M.t_N), I)
    st_latent = reshape(st_raw .* st_sigma, M.s_N, M.t_N)

    # --- 5. Multi-fidelity RFF Projections ---
    # Latent Z Fidelity Projection
    W_z ~ filldist(Normal(0, 1), size(M.z_coords_s, 2), M.M_rff)
    b_z ~ filldist(Uniform(0, 2pi), M.M_rff)
    beta_z_rff ~ filldist(Normal(0, 1), M.M_rff)
    z_projection = rff_map(M.z_coords_s, W_z, b_z) * beta_z_rff
    Turing.@addlogprob! logpdf(MvNormal(z_projection, sigma_z^2 * I), M.z_obs)

    # Latent U Fidelity Projection (3-dimensional)
    W_u ~ filldist(Normal(0, 1), size(M.w_coords_st, 2), M.M_rff)
    b_u ~ filldist(Uniform(0, 2pi), M.M_rff)
    beta_w_rff ~ filldist(Normal(0, 1), M.M_rff, 3)
    u_projection = rff_map(M.w_coords_st, W_u, b_u) * beta_w_rff
    for k in 1:3
        Turing.@addlogprob! logpdf(MvNormal(u_projection[:, k], w_sigma[k]^2 * I), M.w_obs[:, k])
    end

    # --- 6. Likelihood ---
    # Using standardized reconstruction fields
    Xfixed = M.intercept
    mu_fixed = Xfixed * Xfixed_beta
    t_vec = Float64.(M.t_idx)
    
    for i in 1:M.y_N
        s_idx, t_idx = M.s_idx[i], M.t_idx[i]
        
        seasonal = beta_cos * cos(2pi * t_vec[i] / M.period) + beta_sin * sin(2pi * t_vec[i] / M.period)
        
        eta = mu_fixed[i] + s_latent[s_idx] + t_latent[t_idx] + st_latent[s_idx, t_idx] + seasonal + M.log_offset[i]
        
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(eta, y_sigma), M.y_obs[i])
    end
end



@model function example_minibatch(M, ::Type{T}=Float64) where {T}
    # --- 1. Priors ---
    y_sigma ~ Exponential(1.0)
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    t_sigma ~ Exponential(1.0); t_rho ~ Beta(2, 2)
    st_sigma ~ Exponential(0.5)
    c_sigma ~ filldist(Exponential(1.0), 4)
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    
    # Standardized fixed effects naming
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), 2.0 * I)

    # --- 2. Spatial Latent (s_latent) ---
    s_icar ~ MvNormal(zeros(M.s_N), I)
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q_template * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_latent = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # --- 3. Temporal Latent (t_latent) ---
    t_Q = (1.0 / (1.0 - t_rho^2)) .* (M.t_Q_template + (t_rho^2) * I)
    t_raw ~ MvNormal(zeros(M.t_N), I)
    Turing.@addlogprob! -0.5 * dot(t_raw, t_Q * t_raw)
    t_latent = t_raw .* t_sigma

    # --- 4. GMRF Interaction (st_latent) ---
    st_raw ~ MvNormal(zeros(M.s_N * M.t_N), I)
    st_latent = reshape(st_raw .* st_sigma, M.s_N, M.t_N)

    # --- 5. Categorical Effects (c_beta) ---
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:M.cov_N]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2 ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 6. Likelihood ---
    Xfixed = M.intercept
    mu_fixed = Xfixed * Xfixed_beta
    t_vec = Float64.(M.t_idx)

    for i in 1:M.y_N
        idx_a = M.s_idx[i]
        idx_t = M.t_idx[i]
        
        seasonal = beta_cos * cos(2pi * t_vec[i] / M.period) + beta_sin * sin(2pi * t_vec[i] / M.period)
        
        eta = mu_fixed[i] + s_latent[idx_a] + t_latent[idx_t] + st_latent[idx_a, idx_t] + seasonal + M.log_offset[i]
        
        for k in 1:M.cov_N
            eta += c_beta[k][M.cov_indices[i, k]]
        end
        
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(eta, y_sigma), M.y_obs[i])
    end
end


@model function example_deepgp(M, ::Type{T}=Float64) where {T}
    # --- 1. Priors & Structural Components ---
    y_sigma ~ Exponential(1.0); s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    sigma_z ~ Exponential(0.5); w_sigma ~ filldist(Exponential(0.5), 3)
    c_sigma ~ filldist(Exponential(1.0), 4)
    
    # Standardized fixed effects
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), 2.0 * I)

    # BYM2 Spatial Effect
    s_icar ~ MvNormal(zeros(M.s_N), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q_template * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_latent = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # RW2 Categorical Smoothing
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:M.cov_N]
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
    w_latent = Phi_u * beta_w_mat 

    # --- 4. Layer 3: Final Output GP (Y) ---
    coords_l3 = hcat(M.s_coord, M.t_coord, w_latent)
    W_y ~ filldist(Normal(0, 1), size(coords_l3, 2), M.M_rff)
    b_y ~ filldist(Uniform(0, 2pi), M.M_rff)
    beta_y_gp ~ filldist(Normal(0, 1), M.M_rff)

    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2pi .* M.t_coord ./ M.period) .+ beta_sin .* sin.(2pi .* M.t_coord ./ M.period)
    f_y = (rff_map(coords_l3, W_y, b_y) * beta_y_gp) .+ vec(seasonal)

    # --- 5. Likelihood ---
    Xfixed = M.intercept
    mu_fixed = Xfixed * Xfixed_beta

    for i in 1:M.y_N
        a = M.s_idx[i]
        mu = M.log_offset[i] + mu_fixed[i] + f_y[i] + s_latent[a]
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), M.y_obs[i])
    end

    # Multi-fidelity cross-resolution constraints
    Turing.@addlogprob! logpdf(MvNormal(z_latent, sigma_z^2 * I), M.z_obs)
    for k in 1:3
        Turing.@addlogprob! logpdf(MvNormal(w_latent[:, k], w_sigma[k]^2 * I), M.w_obs[:, k])
    end
end

@model function example_nystrom(M, ::Type{T}=Float64) where {T}
    # Model A16 Optimized: Standardized Nyström GP with Stochastic Volatility.
    
    # --- 1. Priors & Structural Components ---
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    w_sigma ~ filldist(Exponential(0.5), 3)
    c_sigma ~ filldist(Exponential(1.0), 4)

    # Standardized fixed effects
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), 2.0 * I)

    # BYM2 Spatial Effect using template precision matrix
    s_icar ~ MvNormal(zeros(M.s_N), I)
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q_template * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_latent = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # RW2 Categorical Smoothing
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:M.cov_N]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2_template ./ c_sigma[k]^2) * c_beta[k])
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
    k_st = SqExponentialKernel() ∘ ARDTransform(inv.(st_ls .+ 1e-6))

    Z_ind = Matrix{T}(undef, M.M_inducing_val, M.st_D)
    st_mu, st_std = mean(M.st_coord, dims=1), std(M.st_coord, dims=1)
    for j in 1:M.st_D; Z_ind[:, j] ~ filldist(Normal(st_mu[j], 2.0 * st_std[j]), M.M_inducing_val); end

    K_zz = Symmetric( kernelmatrix(k_st, RowVecs(Z_ind)) + 1e-6*I)
    K_xz = kernelmatrix(k_st, RowVecs(M.st_coord), RowVecs(Z_ind))
    v_latent ~ filldist(Normal(0, 1), M.M_inducing_val)
    f_nystrom = f_sigma .* (K_xz * (cholesky(K_zz).U \ v_latent))

    # --- 4. Spatiotemporal Stochastic Volatility ---
    W_sigma ~ filldist(Normal(0, 1), M.st_D, M.M_rff_sigma); b_sigma ~ filldist(Uniform(0, 2pi), M.M_rff_sigma)
    beta_rff_sigma ~ filldist(Normal(0, 1), M.M_rff_sigma)
    y_sigma = exp.(rff_map(M.st_coord, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 5. Likelihood ---
    Xfixed = M.intercept
    mu_fixed = Xfixed * Xfixed_beta
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2pi .* M.t_coord ./ M.period) .+ beta_sin .* sin.(2pi .* M.t_coord ./ M.period)

    for i in 1:M.y_N
        a = M.s_idx[i]
        mu = M.log_offset[i] + mu_fixed[i] + f_nystrom[i] + seasonal[i] + s_latent[a]
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma[i]), M.y_obs[i])
    end

    # Prior links for latent variables
    Turing.@addlogprob! logpdf(MvNormal(u1_true, w_sigma[1]^2 * I), M.w_obs[:, 1])
    Turing.@addlogprob! logpdf(MvNormal(u2_true, w_sigma[2]^2 * I), M.w_obs[:, 2])
end


@model function example_spde_nested(M, ::Type{T}=Float64) where {T}
    # --- 1. Priors & Structural Components ---
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    w_sigma ~ filldist(Exponential(0.5), 3)
    c_sigma ~ filldist(Exponential(1.0), 4)

    # Standardized fixed effects naming
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), 2.0 * I)

    # BYM2 Spatial Effect using template precision matrix
    s_icar ~ MvNormal(zeros(M.s_N), I)
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q_template * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_latent = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # RW2 Categorical Smoothing using template precision matrix
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:M.cov_N]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2_template ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 2. Latent Trends (GP Trend & Seasonal) ---
    ls_trend ~ Gamma(2, 2); sigma_trend ~ Exponential(0.5)

    k_trend = SqExponentialKernel() ∘ ScaleTransform(inv(ls_trend))
    unique_times = sort(unique(M.t_coord[:,1]))

    K_trend = Symmetric(sigma_trend^2 * kernelmatrix(k_trend, unique_times) + 1e-4 * I)
    alpha ~ MvNormal(zeros(length(unique_times)), K_trend)
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
    Xfixed = M.intercept
    mu_fixed = Xfixed * Xfixed_beta

    for i in 1:M.y_N
        a = M.s_idx[i]
        mu = M.log_offset[i] + mu_fixed[i] + trend[i] + seasonal[i] + s_latent[a] + u1_true[i]
        for k in 1:M.cov_N
            mu += c_beta[k][M.cov_indices[i, k]]
        end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma[i]), M.y_obs[i])
    end

    # multi-fidelity links
    Turing.@addlogprob! logpdf(MvNormal(u1_true, w_sigma[1]^2 * I), M.w_obs[:, 1])
end



@model function example_kronecker_spde_nested(M, ::Type{T}=Float64) where {T}
    # --- 1. Dimensions and Time setup ---
    unique_t = collect(1:M.t_N) ./ M.t_N
    unique_s = M.s_coord 

    # --- 2. Priors & Structural Components ---
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    w_sigma ~ filldist(Exponential(0.5), 3)
    c_sigma ~ filldist(Exponential(1.0), 4)
    
    # Standardized fixed effects naming
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), 2.0 * I)

    # BYM2 Spatial Effect using template precision matrix
    s_icar ~ MvNormal(zeros(M.s_N), I)
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q_template * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_latent = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # RW2 Categorical Smoothing using template precision matrix
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:M.cov_N]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2_template ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 3. Latent Kronecker Fields (Z, U1) ---
    ls_s_cov ~ Gamma(2, 2); sigma_s_cov ~ Exponential(1.0)
    ls_t_cov ~ Gamma(2, 2); sigma_t_cov ~ Exponential(1.0)

    z_noise ~ filldist(Normal(0, 1), M.y_N)
    z_true = kron_matern_sample(M.s_N, M.t_N, unique_s, unique_t, ls_s_cov, sigma_s_cov, ls_t_cov, sigma_t_cov, z_noise)

    u1_noise ~ filldist(Normal(0, 1), M.y_N)
    u1_true = kron_matern_sample(M.s_N, M.t_N, unique_s, unique_t, ls_s_cov, sigma_s_cov, ls_t_cov, sigma_t_cov, u1_noise)

    # --- 4. Main Spatiotemporal Effect (Kronecker) ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    ls_t_y ~ Gamma(2, 2); sigma_t_y ~ Exponential(1.0)
    y_noise ~ filldist(Normal(0, 1), M.y_N)
    f_st = kron_matern_sample(M.s_N, M.t_N, unique_s, unique_t, ls_s_y, sigma_s_y, ls_t_y, sigma_t_y, y_noise)

    # --- 5. Stochastic Volatility (RFF) ---
    W_sigma ~ filldist(Normal(0, 1), size(M.st_coord, 2), M.M_rff_sigma); b_sigma ~ filldist(Uniform(0, 2pi), M.M_rff_sigma)
    beta_rff_sigma ~ filldist(Normal(0, 1), M.M_rff_sigma)
    y_sigma = exp.(rff_map(M.st_coord, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 6. Likelihood ---
    Xfixed = M.intercept
    mu_fixed = Xfixed * Xfixed_beta

    for i in 1:M.y_N
        a = M.s_idx[i]
        # Combined Mean: Fixed + Kronecker + Spatial + Latent terms
        mu = M.log_offset[i] + mu_fixed[i] + f_st[i] + s_latent[a] + u1_true[i] + z_true[i]
        for k in 1:M.cov_N
            mu += c_beta[k][M.cov_indices[i, k]]
        end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma[i]), M.y_obs[i])
    end

    # Multi-fidelity links
    Turing.@addlogprob! logpdf(MvNormal(z_true, 0.1 * I), M.z_obs)
    Turing.@addlogprob! logpdf(MvNormal(u1_true, w_sigma[1]^2 * I), M.w_obs[:, 1])
end



@model function example_multifidelity_gp_matern(M, ::Type{T}=Float64) where {T}
    # Multi-fidelity GP with standardized interface.
    # Combines high-fidelity Z (Matern), medium-fidelity U (Kron AR1 x Matern), 
    # and standard-fidelity Y with standardized BYM2 and RW2 components.

    # Dimensions and Observations from M
    y_obs = M.y_obs
    z_obs = M.z_obs
    u1_obs, u2_obs, u3_obs = M.w_obs[:, 1], M.w_obs[:, 2], M.w_obs[:, 3]
    
    Nu = length(u1_obs); 
    Nt_u = M.t_N; 
    Ns_u = Nu ÷ Nt_u
    Nz = length(z_obs)

    # --- 1. Priors ---
    y_sigma ~ Exponential(1.0); s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    st_sigma ~ Exponential(0.5); c_sigma ~ filldist(Exponential(1.0), M.cov_N)
    
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
    # Using standardized _template suffix
    s_icar ~ MvNormal(zeros(M.s_N), I); Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q_template * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_latent = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # --- 5. Categorical Smoothing (RW2) ---
    # Using standardized _template suffix
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:M.cov_N]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2_template ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 6. Likelihood (Standard Fidelity Y) ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    rho_y ~ Uniform(-0.99, 0.99); sigma_t_y ~ Exponential(0.5)
    y_noise ~ filldist(Normal(0, 1), M.y_N)
    t_latent = kron_ar1_matern_sample(M.s_N, M.t_N, M.s_coord_tuple[1:M.s_N, :], ls_s_y, sigma_s_y, rho_y, sigma_t_y, y_noise)

    Xfixed_beta ~ Normal(0, 1)
    for i in 1:M.y_N
        a = M.s_idx[i]
        mu = M.log_offset[i] + t_latent[i] + s_latent[a] + Xfixed_beta * z_at_u_full[i]
        for k in 1:M.cov_N; mu += c_beta[k][M.cov_indices[i, k]]; end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma), y_obs[i])
    end
end

@model function example_multifidelity_gp_matern_sv_seasonal(M, ::Type{T}=Float64) where {T}
    # --- 1. Dimensions and Observations ---
    y_obs = M.y_obs
    z_obs = M.z_obs
    u1_obs = M.w_obs[:, 1]

    Nu = length(u1_obs); Nt_u = M.t_N; Ns_u = Nu ÷ Nt_u
    Nz = length(z_obs)

    # Standardized fixed effects naming
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), 2.0 * I)

    # --- 2. High Fidelity: Latent Spatial Z (Matern 3/2) ---
    z_ls ~ Gamma(2, 2); sigma_z_f ~ Exponential(1.0)
    k_z = Matern32Kernel() ∘ ScaleTransform(inv(z_ls))
    K_z = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(M.s_coord_tuple[1:Nz, :])) + 1e-3*I
    z_latent ~ MvNormal(zeros(Nz), K_z)
    z_sigma ~ Exponential(0.5)
    Turing.@addlogprob! logpdf(MvNormal(z_latent, z_sigma^2 * I), z_obs)

    # Interpolation of Z to U and Y
    K_z_u = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(M.s_coord_tuple[1:Ns_u, :]), RowVecs(M.s_coord_tuple[1:Nz, :]))
    z_at_u = (K_z_u * (K_z \ z_latent))
    K_z_y = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(M.s_coord_tuple[1:M.s_N, :]), RowVecs(M.s_coord_tuple[1:Nz, :]))
    z_at_y = (K_z_y * (K_z \ z_latent))

    # --- 3. Medium Fidelity: Latent Spatiotemporal U (Kron AR1 x Matern) ---
    ls_s_u ~ Gamma(2, 2); sigma_s_u ~ Exponential(1.0)
    rho_u ~ Uniform(-0.99, 0.99); sigma_t_u ~ Exponential(0.5)
    u1_noise ~ filldist(Normal(0, 1), Nu)
    unique_pts_u = M.s_coord_tuple[1:Ns_u, :]
    u1_true = kron_ar1_matern_sample(Ns_u, Nt_u, unique_pts_u, ls_s_u, sigma_s_u, rho_u, sigma_t_u, u1_noise)
    
    beta_wz ~ Normal(0, 1)
    Turing.@addlogprob! logpdf(MvNormal(u1_true .+ beta_wz .* repeat(z_at_u, Nt_u), 0.1*I), u1_obs)

    # Kronecker Interpolation of U to Y
    unique_pts_y = M.s_coord_tuple[1:M.s_N, :]
    k_s_u_interp = Matern32Kernel() ∘ ScaleTransform(inv(ls_s_u))
    K_s_uu = sigma_s_u^2 * kernelmatrix(k_s_u_interp, RowVecs(unique_pts_u)) + 1e-3*I
    K_s_yu = sigma_s_u^2 * kernelmatrix(k_s_u_interp, RowVecs(unique_pts_y), RowVecs(unique_pts_u))
    
    t_u = collect(1:Nt_u); t_y = collect(1:M.t_N)
    K_t_uu = ar1_covariance_matrix(t_u, rho_u, sigma_t_u) + 1e-3*I
    K_t_yu = ar1_cross_covariance_matrix(t_y, t_u, rho_u, sigma_t_u)
    u1_at_y = kron(K_t_yu, K_s_yu) * (kron(K_t_uu, K_s_uu) \ u1_true)

    # --- 4. Standard Components (BYM2 & RW2) ---
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    s_icar ~ MvNormal(zeros(M.s_N), I)
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q_template * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_latent = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    c_sigma ~ filldist(Exponential(1.0), 4)
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:M.N_cov]
    for k in 1:M.N_cov
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2_template ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 5. Volatility & Seasonality ---
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    t_vec = M.t_idx ./ M.t_N
    seasonal_y = beta_cos .* cos.(2pi .* t_vec) .+ beta_sin .* sin.(2pi .* t_vec)

    W_sigma ~ filldist(Normal(0, 1), size(M.st_coord, 2), M.M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2pi), M.M_rff_sigma)
    sigma_log_var ~ Exponential(1.0); beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M.M_rff_sigma)
    y_sigma_vec = exp.(rff_map(M.st_coord, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 6. Likelihood ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    rho_y ~ Uniform(-0.99, 0.99); sigma_t_y ~ Exponential(0.5); y_noise ~ filldist(Normal(0, 1), M.y_N)
    f_st_y = kron_ar1_matern_sample(M.s_N, M.t_N, unique_pts_y, ls_s_y, sigma_s_y, rho_y, sigma_t_y, y_noise)

    beta_y_covs ~ filldist(Normal(0, 1), 2)
    Xfixed = M.intercept
    mu_fixed = Xfixed * Xfixed_beta

    for i in 1:M.y_N
        a = M.s_idx[i]
        mu = M.log_offset[i] + mu_fixed[i] + f_st_y[i] + s_latent[a] + seasonal_y[i] + (u1_at_y[i] * beta_y_covs[1]) + (repeat(z_at_y, M.t_N)[i] * beta_y_covs[2])
        for k in 1:M.N_cov
            mu += c_beta[k][M.cov_indices[i, k]]
        end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma_vec[i]), y_obs[i])
    end
end
 
@model function example_svgp_matern_nested(M, ::Type{T}=Float64) where {T}
    # --- 1. Dimensions and Time setup ---
    unique_t = collect(1:M.t_N) ./ M.t_N
    # unique_s assumes coordinates are stored for each distinct unit 1:M.s_N
    unique_s = M.s_coord[1:M.s_N, :]

    # --- 2. Priors & Structural Components ---
    s_sigma ~ Exponential(1.0); s_rho ~ Beta(1, 1)
    w_sigma ~ filldist(Exponential(0.5), 3); c_sigma ~ filldist(Exponential(1.0), 4)
    
    # Standardized fixed effects naming
    Xfixed_beta ~ MvNormal(zeros(M.Xfixed_N), 2.0 * I)

    # BYM2 Spatial Effect using template precision matrix
    s_icar ~ MvNormal(zeros(M.s_N), I)
    Turing.@addlogprob! -0.5 * dot(s_icar, M.s_Q_template * s_icar)
    s_iid ~ MvNormal(zeros(M.s_N), I)
    s_latent = s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid)

    # RW2 Categorical Smoothing using template precision matrix
    c_beta = [Vector{T}(undef, M.N_cat) for _ in 1:M.cov_N]
    for k in 1:M.cov_N
        c_beta[k] ~ MvNormal(zeros(M.N_cat), I)
        Turing.@addlogprob! -0.5 * dot(c_beta[k], (M.Q_rw2_template ./ c_sigma[k]^2) * c_beta[k])
    end

    # --- 3. Nested Latent Covariates (RFF) ---
    coords_tz = hcat(M.t_coord, M.z_obs)
    W_u1 ~ filldist(Normal(0, 1), size(coords_tz, 2), M.M_rff_u); b_u1 ~ filldist(Uniform(0, 2pi), M.M_rff_u)
    u1_true = rff_map(coords_tz, W_u1, b_u1) * filldist(Normal(0, 1), M.M_rff_u)

    # --- 4. Main Spatiotemporal Process (Kronecker) ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    ls_t_y ~ Gamma(2, 2); sigma_t_y ~ Exponential(1.0)
    y_noise ~ filldist(Normal(0, 1), M.y_N)
    f_st = kron_matern_sample(M.s_N, M.t_N, unique_s, unique_t, ls_s_y, sigma_s_y, ls_t_y, sigma_t_y, y_noise)

    # --- 5. Stochastic Volatility (RFF) ---
    W_sigma ~ filldist(Normal(0, 1), size(M.st_coord, 2), M.M_rff_sigma); b_sigma ~ filldist(Uniform(0, 2pi), M.M_rff_sigma)
    beta_rff_sigma ~ filldist(Normal(0, 1), M.M_rff_sigma)
    y_sigma = exp.(rff_map(M.st_coord, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 6. Likelihood ---
    Xfixed = M.intercept
    mu_fixed = Xfixed * Xfixed_beta

    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2pi .* M.t_coord ./ M.period) .+ beta_sin .* sin.(2pi .* M.t_coord ./ M.period)

    for i in 1:M.y_N
        a = M.s_idx[i]
        mu = M.log_offset[i] + mu_fixed[i] + f_st[i] + seasonal[i] + s_latent[a] + u1_true[i]
        for k in 1:M.cov_N
            mu += c_beta[k][M.cov_indices[i, k]]
        end
        Turing.@addlogprob! M.weights[i] * logpdf(Normal(mu, y_sigma[i]), M.y_obs[i])
    end

    # Multi-fidelity links
    Turing.@addlogprob! logpdf(MvNormal(u1_true, w_sigma[1]^2 * I), M.w_obs[:, 1])
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
