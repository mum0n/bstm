    


function bstm(modinput::NamedTuple; kwargs...)  
    modinput = bstm_options(modinput; kwargs...)
    if modinput.model_arch == "univariate"
        return bstm_univariate(modinput )
    elseif modinput.model_arch == "multivariate"
        return bstm_multivariate(modinput )
    elseif modinput.model_arch == "multioutcome"
        return bstm_multioutcome(modinput )
    elseif modinput.model_arch == "multifidelity"
        return bstm_multifidelity(modinput )
    else
        error("Unknown model architecture: $(modinput.model_arch)")
    end
end


@model function bstm_univariate(M, ::Type{T}=Float64) where {T}
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
        # Map RFF to positive scale via exponential link
        y_sigma = exp.((sqrt(2.0 / M.M_rff_sigma) .* cos.(M.vol_proj) * beta_vol) ./ 2.0)
    else
        if M.model_family in ["gaussian", "lognormal"]
            y_sigma ~ Exponential(1.0) 
        end
    end


    # Spatial Manifold (s_eta) ---
    local s_eta = zeros(T, M.s_N)  # default is none
    local s_Q = I(M.s_N)

    

    if M.model_space == "iid"
        s_sigma ~ Exponential(1.0)
        s_iid ~ MvNormal(zeros(M.s_N), I)
        s_eta = (s_iid .* s_sigma)
        s_Q = I(M.s_N)

    elseif M.model_space in ["besag", "icar"]
        s_sigma ~ Exponential(1.0)
        s_icar ~ MvNormalCanon(zeros(M.s_N), Symmetric(Matrix(M.s_Q + M.noise*I)) )
        s_eta = (s_icar .* s_sigma)

    elseif M.model_space == "bym2"
        s_sigma ~ Exponential(1.0)
        s_rho ~ Beta(1, 1)
        s_icar ~ MvNormalCanon(zeros(M.s_N), Symmetric(Matrix(M.s_Q + M.noise*I)) )
        s_iid ~ MvNormal(zeros(M.s_N), I)
        s_eta = (s_sigma .* (sqrt.(s_rho) .* s_icar .+ sqrt.(1 .- s_rho) .* s_iid))

    elseif M.model_space == "leroux"
        # Leroux Precision: Q = 1/sigma^2 * (rho * Q_icar + (1-rho) * I)
        s_sigma ~ Exponential(1.0)
        s_rho ~ Beta(1, 1)
        s_Q = s_rho .* M.s_Q .+ (1 - s_rho) .* I(M.s_N)
        s_raw ~ MvNormalCanon(zeros(M.s_N), s_Q + M.noise * I)
        s_eta = (s_raw .* s_sigma)

    elseif M.model_space == "fitc"
        # --- 1. FITC Sparse GP Implementation ---
        # This logic provides a low-rank approximation of the spatiotemporal field
        # using M.Z_inducing locations.

        # Priors for the ARD (Automatic Relevance Determination) lengthscales
        # Dimensions: 1 for time, 2 for space
        s_sigma ~ Exponential(1.0)
        st_ls ~ filldist(Gamma(2, 2), 3)

        # Kernel construction with ARD and stability jitter
        k_st = (s_sigma^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(st_ls .+ M.noise)))

        # Covariance matrices between inducing points (ZZ) and inducing-to-observations (XZ)
        K_ZZ = kernelmatrix(k_st, RowVecs(M.Z_inducing)) + M.noise * I
        K_XZ = kernelmatrix(k_st, RowVecs(hcat(M.s_coord, M.t_coord)), RowVecs(M.Z_inducing))

        # Sample latent inducing values using the precision-based canonical form for efficiency
        # if M.noise is small, or MvNormal for standard covariance
        s_inducing ~ MvNormal(zeros(size(M.Z_inducing, 1)), K_ZZ)

        # FITC Projection: E[f|u] = K_XZ * inv(K_ZZ) * u
        # Numerically stable solve using backslash
        s_eta = K_XZ * (K_ZZ \ s_inducing)

        # Precision representation for the low-rank component (Optional for audit)
        s_Q = inv(K_ZZ) + M.noise * I
            
    elseif M.model_space =="denseGP"
       # --- Dense Gaussian Process Spatial Manifold ---
        s_sigma ~ Exponential(1.0)
        s_ls ~ Gamma(2, 2) 
        k_s_gp = SqExponentialKernel() ∘ ScaleTransform(inv(s_ls))
        K_s_gp = (s_sigma^2) .* kernelmatrix(k_s_gp, M.s_coord_unique) + M.noise * I
        s_eta_raw ~ MvNormal(zeros(M.s_N), Symmetric(K_s_gp))
        s_eta = s_eta_raw
        s_Q = inv(Symmetric(K_s_gp))

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
        warp_proj = (M.s_coord * M.W_fixed) .+ M.b_fixed'
        s_warp = (sqrt(2.0 / M.M_rff) .* cos.(warp_proj) * w_warp)
        s_eta = vec( s_warp .* s_sigma )

    elseif M.model_space == "rff" || M.model_space == "ei"
        s_sigma ~ Exponential(1.0)
        ls_rff ~ Gamma(2, 2)
        beta_rff ~ filldist(Normal(0, 1), M.M_rff)
        # Spectral mapping for continuous risk surfaces
        projection = (M.s_coord * (M.W_fixed ./ ls_rff)) .+ M.b_fixed'
        s_eta = vec( (sqrt(2.0 / M.M_rff) * s_sigma) .* (cos.(projection) * beta_rff) )
        # In RFF, the latent variables are the iid weights beta_rff
        s_Q = I(M.M_rff)

    elseif M.model_space == "bgcn"
        # 1. Spectral Weight Priors
        # These weights represent the learnable signal amplitude across the graph nodes
        # typically inferred from the graph signal (spectral domain)
        s_sigma ~ Exponential(1.0)
        gcn_weight_raw ~ MvNormal(zeros(M.s_N), I)

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
        curr_coords = hcat(M.s_coord, M.t_coord ./ M.t_N)
        
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
        L_sar = I(M.s_N) - s_rho .* W_row_norm

        # 2. Latent Field Sampling
        # sample the latent spatial field directly in its canonical form
        # Observation Mapping and Scaling
        s_Q = Symmetric(Matrix(L_sar' * L_sar)) + (s_sigma + M.noise) * I
        s_eta ~ MvNormalCanon(zeros(M.s_N), s_Q)

    elseif M.model_space == "fft"
        s_sigma ~ Exponential(1.0)
        s_raw ~ MvNormal(zeros(M.s_N), I)
        s_eta = ( M.s_Q \ s_raw ) .* s_sigma

    elseif M.model_space == "nystrom"
        # Nystrom low-rank approximation
        s_sigma ~ Exponential(1.0)
        s_raw ~ filldist(Normal(0, 1), M.M_inducing_count)
        s_eta = (M.K_nystrom_proj * s_raw) .* s_sigma

    elseif M.model_space == "svc"  

        # Logic for Spatially Varying Coefficients
        # We generate a spatial field for each of the fixed_N covariates
        s_svc = Matrix{T}(undef, M.s_N, M.fixed_N)
        svc_sigma ~ filldist(Exponential(0.5), M.fixed_N)
        svc_raw ~ arraydist([MvNormalCanon(zeros(M.s_N), M.s_Q + M.noise*I) for _ in 1:M.fixed_N])
        
        for k in 1:M.fixed_N
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
        W_scaled = zeros(eltype(s_rho), M.s_N, M.s_N)
        for i in 1:M.s_N
            preds = findall(x -> x == 1, W_adj[i, 1:i-1])
            if !isempty(preds)
                W_scaled[i, preds] .= s_rho / length(preds)
            end
        end

        L = I(M.s_N) - W_scaled

        # 2. Derive the structural precision matrix s_Q_dag = L' * L
        # This represents the internal spatial structure before variance scaling
        s_Q_structural = L' * L

        # 3. Apply marginal scaling (s_sigma) to get the final s_Q
        s_Q = (1.0 / s_sigma^2) .* s_Q_structural + M.noise * I

        # For DAG, Q = (I - rho*W)' * (I - rho*W)
        s_eta ~ MvNormalCanon(zeros(M.s_N), s_Q)

    elseif M.model_space == "local"
        # 1. Cluster-Specific Hyper-priors
        # mu_clusters represents the mean of each spatial partition (mosaic)
        s_sigma ~ Exponential(1.0)
        mu_clusters ~ filldist(Normal(0, 1), M.n_clusters)

        # 2. Leroux Precision Construction
        # Combines structured ICAR (M.s_Q) and unstructured IID (I) components
        s_rho ~ Beta(1, 1)
        s_Q_base = s_rho .* M.s_Q .+ (1 - s_rho) .* I(M.s_N)

        # 3. Latent Field Sampling
        # Scale the precision by the inverse variance (1/s_sigma^2)
        # and map cluster means to the full spatial field via M.cluster_assignments
        s_Q = (1.0 / (s_sigma^2 + M.noise)) .* s_Q_base + M.noise * I
        s_eta_raw ~ MvNormalCanon(mu_clusters[M.cluster_assignments], s_Q )

        # 4. Observation Mapping
        s_eta = s_eta_raw

    elseif M.model_space == "svgp"

        f_sigma ~ Exponential(1.0)
        st_ls ~ filldist(Gamma(2, 2), size(M.Z_inducing_coords, 2))
        s_sigma ~ Exponential(1.0)

        # Define the kernel using f_sigma and st_ls
        kernel_svgp = f_sigma * SqExponentialKernel() ∘ ScaleTransform(st_ls)
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
        s_eta = zeros(T, M.s_N)
        s_Q = I(M.s_N)

    else
        # nothing to do
    end
 
    # --- 3. Temporal Manifold (t_eta) ---
 
    local t_eta = zeros(M.t_N)
    local t_Q = I(M.t_N)

    if haskey(M, :t_N) && M.t_N > 1

        if M.model_time == "ar1"
            t_sigma ~ Exponential(1.0)
            t_rho ~ Beta(2, 2)
            # Construct the standardized AR1 precision matrix
            # The template M.t_Q represents the first-order difference structure
            t_Q_base = Symmetric((1.0 + t_rho^2) .* I(M.t_N) .+ (t_rho) .* M.t_Q)

            # Normalize by the variance scale and add stability jitter
            t_Q = (1.0 / (1.0 - t_rho^2 + M.noise)) .* t_Q_base

            # Use the Canonical form of the Multivariate Normal for precision-based sampling
            t_raw ~ MvNormalCanon(zeros(M.t_N), Symmetric(Matrix(t_Q + M.noise * I)) )
            t_eta = t_raw .* t_sigma

        elseif M.model_time == "rw2"
            t_sigma ~ Exponential(1.0)
            # 1. Structural Template: M.t_Q (the second-order difference matrix)
            # 2. Recompose Precision: Scale the structural template by the inverse variance
            # Q_rw2 = (1 / sigma^2) * Q_template
            t_Q = (1.0 / (t_sigma^2 + M.noise)) .* M.t_Q

            # 3. Sampling: Use the Canonical (Precision) form
            # We add a small noise term to the diagonal to ensure the matrix is PosDef
            t_raw ~ MvNormalCanon(zeros(M.t_N), Matrix(t_Q + M.noise * I))
            t_eta = t_raw .* t_sigma

        elseif M.model_time == "gp"
            t_sigma ~ Exponential(1.0)
            t_ls ~ InverseGamma(3, 3) # GP Lengthscale
            # 1. Construct Kernel Matrix (Covariance Space)
            t_K = (t_sigma^2) .* kernelmatrix(SqExponentialKernel() ∘ ScaleTransform(inv(t_ls)), 1.0:Float64(M.t_N)) + M.noise * I

            # 2. Latent Field Sampling
            # Using MvNormal for covariance-based sampling
            t_eta ~ MvNormal(zeros(M.t_N), Symmetric(t_K))

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
            t_eta ~ MvNormal(zeros(M.t_N), t_sigma*I)

            t_Q = I(M.t_N)

        elseif M.model_time == "none"
            t_eta = zeros(M.t_N)
            t_Q = I(M.t_N)

        else # Default to no effects
            # nothing to do
        end
    end


    # --- 4. Season Manifold (u_eta) ---
    local u_eta = zeros(M.u_N)

    if haskey(M, :u_N) && M.u_N > 1

        if M.model_season == "ar1"
            u_sigma ~ Exponential(0.5)
            u_rho ~ Beta(2, 2)
            u_Q = Symmetric((1.0 / (1.0 - u_rho^2 + M.noise)) .* (M.u_Q + (u_rho^2) * I))
            u_raw ~ MvNormalCanon(zeros(M.u_N), u_Q)
            u_eta = u_raw .* u_sigma
        elseif M.model_season == "rw2"
            u_raw ~ MvNormalCanon(zeros(M.u_N), M.u_Q + M.noise*I)
            u_eta = u_raw .* u_sigma
        elseif M.model_season == "gp"
            u_ls ~ InverseGamma(3, 3)
            u_K = kernelmatrix(SqExponentialKernel() ∘ ScaleTransform(1.0/u_ls), 1.0:Float64(M.u_N)) + M.noise*I
            u_gp ~ MvNormal(zeros(M.u_N), u_K)
            u_eta = u_gp .* u_sigma
        elseif M.model_season == "harmonic"
            u_α ~ Normal(0, 1)
            u_β ~ Normal(0, 1)
            u_steps = 1:M.u_N # Assuming M.u_N represents the period for seasonal cycles
            u_eta = (u_α .* sin.(2π .* u_steps ./ 12.0) .+ u_β .* cos.(2π .* u_steps ./ 12.0)) .* u_sigma
        elseif M.model_season == "iid"
            u_iid ~ MvNormal(zeros(M.u_N), I)
            u_eta = u_iid .* u_sigma
        else
            # nothing to do
        end

    end

    
    # --- 5. Space-Time Interaction (st_eta) ---
    # Captures localized deviations across both space and time
 
    local st_eta = zeros(M.s_N, M.t_N)

    if M.model_st == 0 # None
        st_eta = zeros(M.s_N, M.t_N)

    elseif M.model_st == 1 # Type I: IID Interaction
        st_sigma ~ Exponential(0.5)
        st_raw ~ MvNormal(zeros(M.s_N * M.t_N), I)
        st_eta = reshape(st_raw .* st_sigma, M.s_N, M.t_N)

    elseif M.model_st == 2 # Type II: Temporal Structure (I ⊗ Q_t)
        # Each area has its own structured temporal trend
        st_sigma ~ Exponential(0.5)
        st_Q2 = kron(I(M.s_N), t_Q)
        st_raw ~ MvNormalCanon(zeros(M.s_N * M.t_N), st_Q2)
        st_Q2 = Matrix(kron(I(M.s_N), t_Q))
        st_raw ~ MvNormalCanon(zeros(M.s_N * M.t_N), Matrix(st_Q2))
        st_eta = reshape(st_raw .* st_sigma, M.s_N, M.t_N)

    elseif M.model_st == 3 # Type III: Spatial Structure (Q_s ⊗ I)
        # Each time point has its own structured spatial field
        st_sigma ~ Exponential(0.5)
        st_Q3 = kron(s_Q, I(M.t_N))
        st_raw ~ MvNormalCanon(zeros(M.s_N * M.t_N), st_Q3)
        st_Q3 = Matrix(kron(s_Q, I(M.t_N)))
        st_raw ~ MvNormalCanon(zeros(M.s_N * M.t_N), Matrix(st_Q3))
        st_eta = reshape(st_raw .* st_sigma, M.s_N, M.t_N)

    elseif M.model_st == 4 # Type IV: Inseparable (Q_s ⊗ Q_t)
        # Fully inseparable spatiotemporal structure
        st_sigma ~ Exponential(0.5)
        st_Q4 = kron(s_Q, t_Q)
        st_raw ~ MvNormalCanon(zeros(M.s_N * M.t_N), st_Q4)
        st_Q4 = Matrix(kron(s_Q, t_Q))
        st_raw ~ MvNormalCanon(zeros(M.s_N * M.t_N), Matrix(st_Q4))
        st_eta = reshape(st_raw .* st_sigma, M.s_N, M.t_N)
    else
        # nothing to do 
    end


    # --- 6. Covariate Smoothing (c_beta) ---
    # c_beta handles discretized continuous covariates (RW2/IID) or RFF-based GP
    
    local c_eta = zeros(M.N_cat)
    
    if M.cov_N > 0
            
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
    eta = zeros(T, M.y_N) 
    
    if haskey(M, :log_offset) 
        eta .+= M.log_offset 
    end
    
    if !isnothing(s_eta)
        if length(s_eta) == M.s_N
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
        for i in 1:M.y_N; 
            eta[i] += st_eta[M.s_idx[i], M.t_idx[i]]; 
        end
    end

    if !isnothing(c_eta) 
        for i in 1:M.cov_N
            eta .+= c_eta[M.cov_indices[:, i]] 
        end
    end

    if M.fixed_N > 0
        if M.model_space == "svc"

            # Assemble eta component: sum(beta_k(s) * x_ik)
            for i in 1:M.y_N
                a = M.s_idx[i]
                for k in 1:M.fixed_N
                    eta[i] += s_svc[a, k] * M.fixed[i, k]
                end
            end

        else 
            # fixed effects
            d_beta ~ MvNormal(zeros(M.fixed_N), I)
            eta .+= M.fixed * d_beta
            
        end
    end
  
    # --- 7. Likelihood (handling missing observations) ---
    # Filter out missing observations and corresponding linear predictors/weights/trials
    # This ensures logpdf only operates on valid data points.

    eta = clamp.(eta, -20.0, 20.0)

    good_indices = findall(!ismissing, M.y_obs)
    y_obs_filtered = M.y_obs[good_indices]
    eta_filtered = eta[good_indices]
    weights_filtered = M.weights[good_indices]
    trials_filtered = M.trials[good_indices]

    Turing.@addlogprob! logpdf(bstm_Likelihood(M.model_family, M.use_zi, weights_filtered, phi_zi, r_nb, y_sigma, trials_filtered, y_obs_filtered), eta_filtered)

end




@model function bstm_multioutcome(M, ::Type{T}=Float64) where {T}
    # --- 1. SETUP & HYPERPRIORS ---
    s_N, M.t_N, 
    outcomes_N =  M.outcomes_N
    y_N = M.y_N
 
    # Global Scales & Correlations
    s_sigma ~ filldist(Exponential(1.0), outcomes_N)
    t_sigma ~ filldist(Exponential(1.0), outcomes_N)
    local st_sigma
    if M.model_st > 0
        st_sigma ~ filldist(Exponential(0.5), outcomes_N)
    else
        st_sigma = zeros(outcomes_N)
    end

    local u_sigma
    if M.model_season != "none"
        u_sigma ~ filldist(Exponential(0.5), outcomes_N)
    else
        u_sigma = zeros(outcomes_N)
    end

    local c_sigma
    if M.cov_N > 0
        c_sigma ~ filldist(Exponential(1.0), outcomes_N)
    else
        c_sigma = zeros(outcomes_N)
    end


    # Cross-outcome correlation on spatial manifold
    L_corr ~ LKJCholesky(outcomes_N, 1.0, :L)

    # Distribution specific hyperpriors
    local r_nb
    if M.model_family == "negbin"
        r_nb ~ filldist(Exponential(1.0), outcomes_N)
    else
        r_nb = fill(1.0, outcomes_N)
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
        y_sigma_vec = exp.( ( M.stochastic_volatility_factor * beta_vol) ./ 2.0)
        y_sigma = [y_sigma_vec for _ in 1:outcomes_N]
    else
        if M.model_family in ["gaussian", "lognormal"]
            y_sigma_const ~ filldist(Exponential(1.0), outcomes_N)
            y_sigma = [fill(y_sigma_const[k], M.y_N) for k in 1:outcomes_N]
        else
            y_sigma = [fill(1.0, M.y_N) for k in 1:outcomes_N]
        end
    end

    # --- 2. SPATIAL MANIFOLDS ---
    # Mixing/Correlation parameters
    local s_rho
    if M.model_space in ["bym2", "leroux", "sar", "fft", "dag", "local"]
        s_rho ~ filldist(Beta(1, 1), outcomes_N)
    else
        s_rho = fill(1.0, outcomes_N)
    end

    s_eta_scaled = Matrix{T}(undef, M.s_N, outcomes_N)
    s_Q = M.s_Q

    for k in 1:outcomes_N
        if M.model_space == "iid"
            s_iid_k ~ MvNormal(zeros(M.s_N), I)
            s_eta_scaled[:, k] = s_iid_k
        elseif M.model_space in ["besag", "icar"]
            s_icar_k ~ MvNormalCanon(zeros(M.s_N), M.s_Q + M.noise*I)
            s_eta_scaled[:, k] = s_icar_k
        elseif M.model_space == "bym2"
            s_icar_k ~ MvNormalCanon(zeros(M.s_N), M.s_Q + M.noise*I)
            s_iid_k ~ MvNormal(zeros(M.s_N), I)
            s_eta_scaled[:, k] = sqrt.(s_rho[k]) .* s_icar_k .+ sqrt(1 .- s_rho[k]) .* s_iid_k
        elseif M.model_space == "leroux"
            Q_k = s_rho[k] .* M.s_Q .+ (1 - s_rho[k]) .* I(M.s_N)
            s_raw_k ~ MvNormalCanon(zeros(M.s_N), Q_k + M.noise * I)
            s_eta_scaled[:, k] = s_raw_k
        elseif M.model_space == "sar"
            W_row_norm = M.W ./ (vec(sum(M.W, dims=2)) .+ M.noise)
            L_sar_k = I(M.s_N) - s_rho[k] .* W_row_norm
            s_Q_k = Symmetric(L_sar_k' * L_sar_k) + M.noise * I
            s_sar_raw_k ~ MvNormalCanon(zeros(M.s_N), s_Q_k)
            s_eta_scaled[:, k] = s_sar_raw_k
        elseif M.model_space == "bgcn"
            gcn_raw_k ~ MvNormal(zeros(M.s_N), I)
            D_inv_sqrt = Diagonal(1.0 ./ sqrt.(vec(sum(M.W, dims=2)) .+ M.noise))
            W_norm = D_inv_sqrt * M.W * D_inv_sqrt
            s_eta_scaled[:, k] = W_norm * gcn_raw_k
        elseif M.model_space == "mosaic"
            mu_local_k ~ filldist(Normal(0, 1), M.n_mosaics)
            s_eta_scaled[:, k] = mu_local_k[M.cluster_assignments]
        elseif M.model_space == "warping"
            w_warp_k ~ filldist(Normal(0, 1), M.M_rff)
            warp_proj = (M.s_coord * M.W_fixed) .+ M.b_fixed'
            s_eta_scaled[:, k] = (sqrt(2.0 / M.M_rff) .* cos.(warp_proj) * w_warp_k)
        elseif M.model_space == "fitcGP" || M.model_space == "svgp"
            u_inducing_k ~ filldist(Normal(0, 1), M.M_inducing_count)
            s_eta_scaled[:, k] = M.Z_inducing_proj * u_inducing_k
        elseif M.model_space == "fft"
            s_spectral_raw_k ~ MvNormal(zeros(M.s_N), I)
            s_eta_scaled[:, k] = M.s_Q \ s_spectral_raw_k
        elseif M.model_space == "dag"
            W_adj = Matrix(M.W)
            W_scaled = zeros(T, M.s_N, M.s_N)
            for i in 1:M.s_N
                preds = findall(x -> x == 1, W_adj[i, 1:i-1])
                if !isempty(preds); W_scaled[i, preds] .= s_rho[k] / length(preds); end
            end
            L_dag = I(M.s_N) - W_scaled
            s_Q_k = Symmetric(L_dag' * L_dag) + M.noise * I
            s_dag_raw_k ~ MvNormalCanon(zeros(M.s_N), s_Q_k)
            s_eta_scaled[:, k] = s_dag_raw_k
        else
            s_eta_scaled[:, k] = zeros(M.s_N)
        end
    end

    # Unified spatial field with cross-outcome linkage
    s_eta = (s_eta_scaled * L_corr) .* s_sigma'

    # --- 3. TEMPORAL, SEASONAL & COVARIATE MANIFOLDS ---
    t_eta = Matrix{T}(undef, M.t_N, outcomes_N)
    u_eta = Matrix{T}(undef, M.u_N, outcomes_N)
    c_eta = Matrix{T}(undef, M.N_cat, outcomes_N)
    t_Q_list = Vector{Any}(undef, outcomes_N)

    for k in 1:outcomes_N
        # --- Temporal Options ---
        if M.model_time == "ar1"
            t_rho_k ~ Beta(2, 2)
            t_Q_k = (1.0 / (1.0 - t_rho_k^2 + M.noise)) .* Symmetric((1.0 + t_rho_k^2) .* I(M.t_N) .+ (t_rho_k) .* M.t_Q)
            t_raw_k ~ MvNormalCanon(zeros(M.t_N), t_Q_k + M.noise * I)
            t_eta[:, k] = t_raw_k .* t_sigma[k]
            t_Q_list[k] = t_Q_k
        elseif M.model_time == "rw2"
            t_Q_k = (1.0 / (t_sigma[k]^2 + M.noise)) .* M.t_Q
            t_raw_k ~ MvNormalCanon(zeros(M.t_N), t_Q_k + M.noise * I)
            t_eta[:, k] = t_raw_k .* t_sigma[k]
            t_Q_list[k] = t_Q_k
        elseif M.model_time == "gp"
            t_ls_k ~ InverseGamma(3, 3)
            t_K_k = (t_sigma[k]^2) .* kernelmatrix(SqExponentialKernel() ∘ ScaleTransform(inv(t_ls_k)), 1.0:Float64(M.t_N)) + M.noise * I
            t_gp_k ~ MvNormal(zeros(M.t_N), Symmetric(t_K_k))
            t_eta[:, k] = t_gp_k
            t_Q_list[k] = inv(Symmetric(t_K_k))
        elseif M.model_time == "harmonic"
            t_alpha_k ~ Normal(0, 1); 
            t_beta_k ~ Normal(0, 1)
            t_eta[:, k] = (t_alpha_k .* sin.(M.t_angle) .+ t_beta_k .* cos.(M.t_angle)) .* t_sigma[k]
            t_Q_list[k] = (1.0 / (t_sigma[k]^2 + M.noise)) .* M.t_Q
        elseif M.model_time == "iid"
            t_eta[:, k] ~ MvNormal(zeros(M.t_N), t_sigma[k] * I)
            t_Q_list[k] = I(M.t_N)
        else
            t_Q_list[k] = I(M.t_N)
        end

        # --- Seasonal Options ---
        if M.model_season == "ar1"
            u_rho_k ~ Beta(2, 2)
            u_Q_k = (1.0 / (1.0 + M.noise)) .* (M.u_Q + M.noise*I)
            u_raw_k ~ MvNormalCanon(zeros(M.u_N), u_Q_k)
            u_eta[:, k] = u_raw_k .* u_sigma[k]
        elseif M.model_season == "rw2"
            u_raw_k ~ MvNormalCanon(zeros(M.u_N), M.u_Q + M.noise*I)
            u_eta[:, k] = u_raw_k .* u_sigma[k]
        elseif M.model_season == "gp"
            u_ls_k ~ InverseGamma(3, 3)
            u_K_k = (u_sigma[k]^2) .* kernelmatrix(SqExponentialKernel() ∘ ScaleTransform(inv(u_ls_k)), 1.0:Float64(M.u_N)) + M.noise * I
            u_gp_k ~ MvNormal(zeros(M.u_N), Symmetric(u_K_k))
            u_eta[:, k] = u_gp_k
        elseif M.model_season == "harmonic"
            u_alpha_k ~ Normal(0, 1); 
            u_beta_k ~ Normal(0, 1)
            u_steps = 1:M.u_N
            u_eta[:, k] = (u_alpha_k .* sin.(2π .* u_steps ./ 12.0) .+ u_beta_k .* cos.(2π .* u_steps ./ 12.0)) .* u_sigma[k]
        elseif M.model_season == "iid"
            u_iid_k ~ MvNormal(zeros(M.u_N), I)
            u_eta[:, k] = u_iid_k .* u_sigma[k]
        end

        if M.cov_N > 0
            if M.model_cov == "ar1"
                c_rho_k ~ Beta(2, 2)
                c_Q_k = (1.0 / (1.0 - c_rho_k^2 + M.noise)) .* ((1.0 + c_rho_k^2) .* I(M.N_cat) .+ (c_rho_k) .* M.c_Q)
                c_raw_k ~ MvNormalCanon(zeros(M.N_cat), Matrix(c_Q_k))
                c_eta[:, k] = c_raw_k .* c_sigma[k]
            elseif M.model_cov == "rw2"
                c_raw_k ~ MvNormalCanon(zeros(M.N_cat), M.c_Q + M.noise * I)
                c_eta[:, k] = c_raw_k .* c_sigma[k]
            elseif M.model_cov == "gp"
                c_ls_k ~ InverseGamma(3, 3)
                c_K_k = (c_sigma[k]^2) .* kernelmatrix(SqExponentialKernel() ∘ ScaleTransform(1.0 / c_ls_k), 1.0:Float64(M.N_cat)) + M.noise * I
                c_gp_k ~ MvNormal(zeros(M.N_cat), Matrix(c_K_k))
                c_eta[:, k] = c_gp_k
            elseif M.model_cov == "harmonic"
                h_c_alpha_k ~ Normal(0, 1)
                h_c_beta_k ~ Normal(0, 1)
                c_steps = 1:M.N_cat
                c_eta[:, k] = (h_c_alpha_k .* sin.(2π .* c_steps ./ 12.0) .+ h_c_beta_k .* cos.(2π .* c_steps ./ 12.0)) .* c_sigma[k]
            elseif M.model_cov == "rff"
                W_rff_k ~ MvNormal(zeros(M.N_rff), I)
                B_rff_k ~ filldist(Uniform(0, 2π), M.N_rff)
                c_eta[:, k] = (cos.((1:M.N_cat) * W_rff_k' .+ B_rff_k') * ones(M.N_rff)) .* c_sigma[k]
            elseif M.model_cov == "iid"
                c_iid_k ~ MvNormal(zeros(M.N_cat), I)
                c_eta[:, k] = c_iid_k .* c_sigma[k]
            else
                # Default to zeros if no specific model is matched
                c_eta[:, k] = zeros(M.N_cat)
            end
        end
    end

    # --- 4. LIKELIHOOD ASSEMBLY ---
    for k in 1:outcomes_N
        eta_k = M.log_offset[:, k] .+ s_eta[M.s_idx, k] .+ t_eta[M.t_idx, k]
        if M.model_season != "none"; eta_k .+= u_eta[M.u_idx, k]; end
        if M.cov_N > 0; eta_k .+= c_eta[M.cov_indices[:, 1], k]; end

        if M.model_space == "svc" && M.fixed_N > 0
            # Multivariate SVC logic using shared s_eta_scaled components
            svc_sigma_k ~ Exponential(0.5)
            for d in 1:M.fixed_N; eta_k .+= (s_eta_scaled[M.s_idx, k] .* svc_sigma_k) .* M.fixed[:, d]; end
        elseif M.fixed_N > 0
            d_beta_k ~ filldist(Normal(0, 1), M.fixed_N)
            eta_k .+= M.fixed * d_beta_k
        end

        if M.model_st > 0
            local st_Q
            if M.model_st == 1; st_Q = I(M.s_N * M.t_N)
            elseif M.model_st == 2; st_Q = kron(I(M.s_N), Symmetric(t_Q_list[k]))
            elseif M.model_st == 3; st_Q = kron(M.s_Q, I(M.t_N))
            elseif M.model_st == 4; st_Q = kron(M.s_Q, Symmetric(t_Q_list[k])); end

            st_raw_k ~ MvNormalCanon(zeros(M.s_N * M.t_N), Symmetric(Matrix(st_Q + M.noise * I)))
            st_field_k = reshape(st_raw_k .* st_sigma[k], M.s_N, M.t_N)
            for i in 1:M.y_N; eta_k[i] += st_field_k[M.s_idx[i], M.t_idx[i]]; end
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
    outcomes_N = size(M.y_obs)
    N_factors = get(M, :N_factors, 2)
    nvh = Int(outcomes_N * N_factors - N_factors * (N_factors - 1) / 2)

    # --- 2. LATENT PCA MANIFOLD ---
    pca_sd ~ Bijectors.ordered(arraydist(LogNormal.(get(M, :sigma_prior, zeros(N_factors)), 1.0)))
    pca_pdef_sd ~ LogNormal(0.0, 0.5)
    v ~ filldist(Normal(0.0, 1.0), nvh)

    # Householder reconstruction: Kmat is the residual/uniqueness matrix
    Kmat, r_diag, U = householder_transform(v, outcomes_N, N_factors, M.ltri, pca_sd, pca_pdef_sd, M.noise)

    # --- 3. LATENT FACTOR MANIFOLDS ---
    pc_scores = Matrix{T}(undef, M.y_N, N_factors)

    for k in 1:N_factors
        local s_eta_k = zeros(T, M.s_N)
        local t_eta_k = zeros(T, M.t_N)
        local st_inter_k = zeros(T, M.s_N, M.t_N)

        # A. Spatial Manifold (Factor-Specific)
        s_sigma_k ~ Exponential(1.0)
        if M.model_space == "bym2"
            s_rho_k ~ Beta(1, 1)
            s_icar_k ~ MvNormalCanon(zeros(M.s_N), Symmetric(Matrix(M.s_Q + M.noise*I)))
            s_iid_k ~ MvNormal(zeros(M.s_N), I)
            s_eta_k = s_sigma_k .* (sqrt(s_rho_k) .* s_icar_k .+ sqrt(1 - s_rho_k) .* s_iid_k)
        elseif M.model_space == "besag"
            s_icar_k ~ MvNormalCanon(zeros(M.s_N), Symmetric(Matrix(M.s_Q + M.noise*I)))
            s_eta_k = s_icar_k .* s_sigma_k
        elseif M.model_space == "sar"
            s_rho_k ~ Beta(1, 1)
            W_rn = M.W ./ (vec(sum(M.W, dims=2)) .+ M.noise)
            L_sar = I(M.s_N) - s_rho_k .* W_rn
            s_eta_k ~ MvNormalCanon(zeros(M.s_N), Symmetric(Matrix(L_sar' * L_sar) + (s_sigma_k + M.noise)*I))
        end # (Other M.model_space types like DAG/Mosaic follow same pattern as bstm())

        # B. Temporal Manifold (Factor-Specific)
        t_sigma_k ~ Exponential(1.0)
        if M.model_time == "ar1"
            t_rho_k ~ Beta(2, 2)
            t_Q_k = build_bstm_ar1_precision(M.t_Q, t_rho_k; noise=M.noise)
            t_raw_k ~ MvNormalCanon(zeros(M.t_N), t_Q_k)
            t_eta_k = t_raw_k .* t_sigma_k
        elseif M.model_time == "rw2"
            t_Q_k = (1.0 / (t_sigma_k^2 + M.noise)) .* M.t_Q
            t_eta_k ~ MvNormalCanon(zeros(M.t_N), Symmetric(Matrix(t_Q_k + M.noise*I)))
        end

        # C. Space-Time Interaction (Factor-Specific)
        if M.model_st > 0
            st_sigma_k ~ Exponential(0.5)
            # Use factor-specific precision derived from s_Q and t_Q_k
            # Simplified Type I (IID) for example:
            st_raw_k ~ MvNormal(zeros(M.s_N * M.t_N), I)
            st_inter_k = reshape(st_raw_k .* st_sigma_k, M.s_N, M.t_N)
        end

        # Assemble scores for factor k
        for i in 1:M.y_N
            a, t = M.s_idx[i], M.t_idx[i]
            pc_scores[i, k] = s_eta_k[a] + t_eta_k[t] + st_inter_k[a, t]
        end
    end

    # --- 4. SHARED COVARIATE EFFECTS ---
    # These apply to the predictor matrix AFTER projection to outcome space
    mu_matrix = pc_scores * U'

    if M.cov_N > 0
        c_sigma ~ Exponential(1.0)
        if M.model_cov == "rw2"
            c_Q = (1.0 / (c_sigma^2 + M.noise)) .* M.c_Q
            c_eta ~ MvNormalCanon(zeros(M.N_cat), Symmetric(Matrix(c_Q + M.noise*I)))
            for i in 1:M.y_N
                mu_matrix[i, :] .+= c_eta[M.cov_indices[i, 1]]
            end
        end
    end

    if M.fixed_N > 0
        d_beta ~ filldist(Normal(0, 1), M.fixed_N, outcomes_N)
        mu_matrix .+= M.fixed * d_beta
    end

    if haskey(M, :log_offset)
        mu_matrix .+= M.log_offset
    end

    # --- 5. MULTIVARIATE LIKELIHOOD ---
    Kmat_sym = Symmetric(Kmat)
    M.y_obs ~ arraydist([MvNormal(mu_matrix[i, :], Kmat_sym) for i in 1:M.y_N])
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
        # Map RFF to positive scale via exponential link
        y_sigma_val = exp.( ( M.stochastic_volatility_factor * beta_vol) ./ 2.0)
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
    s_icar ~ MvNormalCanon(zeros(M.s_N), M.s_Q + M.noise*I)
    s_iid ~ MvNormal(zeros(M.s_N), I)

    # Soft sum-to-zero constraint for spatial identifiability
    sum_icar = sum(s_icar)
    sum_icar ~ Normal(0, 0.001 * M.s_N)

    # Map to observations via index
    s_eta = (s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid))[M.s_idx]

    # --- Temporal Manifold ---
    local t_eta_full
    local t_Q = I(1) # Default for when t_N is not available or model_time is "none"


    t_sigma ~ Exponential(1.0)
    if haskey(M, :t_N) && M.t_N > 1
        if M.model_time == "ar1"
            t_rho ~ Beta(2, 2)
        # Recompose AR1 precision using precomputed structural template
        t_Q_base = Symmetric((1.0 + t_rho^2) .* I(M.t_N) .+ (t_rho) .* M.t_Q)
        t_Q = Symmetric((1.0 / (1.0 - t_rho^2 + M.noise)) .* t_Q_base )

        t_raw ~ MvNormalCanon(zeros(M.t_N), t_Q)
        t_eta_full = (t_raw .* t_sigma)[M.t_idx]

    elseif M.model_time == "rw2"
        # Scale structural RW2 template by inverse variance
        t_Q = Symmetric((1.0 / (t_sigma^2 + M.noise)) .* M.t_Q  )

        t_raw ~ MvNormalCanon(zeros(M.t_N), t_Q)
        t_eta_full = t_raw[M.t_idx] # sigma is already incorporated in the precision scale

    elseif M.model_time == "gp"
        t_ls ~ InverseGamma(3, 3)

        # Covariance-based sampling for GP with lengthscale
        K_t = (t_sigma^2) .* kernelmatrix(SqExponentialKernel() ∘ ScaleTransform(inv(t_ls)), 1.0:Float64(M.t_N)) + M.noise * I
        
        t_gp ~ MvNormal(zeros(M.t_N), Symmetric(K_t))
        t_eta_full = t_gp[M.t_idx]

        # Derive precision for interaction logic fallback
        t_Q = inv(Symmetric(K_t))

    elseif M.model_time == "harmonic"
        t_alpha ~ Normal(0, 1);
        t_beta ~ Normal(0, 1)
        t_eta_full = (t_alpha .* sin.(M.t_angle) .+ t_beta .* cos.(M.t_angle)) .* t_sigma
        t_Q = (1.0 / (t_sigma^2 + M.noise)) .* M.t_Q
    elseif M.model_time == "iid"
        t_Q = Symmetric((1.0 / (t_sigma^2 + M.noise)) .* I(M.t_N))
        t_iid ~ MvNormal(zeros(M.t_N), I)
        t_eta_full = (t_iid .* t_sigma)[M.t_idx]

    else
        t_Q = Symmetric(I(M.t_N))
        t_eta_full = zeros(T, M.y_N) # Mapped to observations
    end
    else # t_N is not available or 1
        t_Q = I(1)
        t_eta_full = zeros(T, M.y_N) # Mapped to observations
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
    if M.model_season == "ar1" && haskey(M, :u_N) && M.u_N > 1
        u_Q = build_bstm_ar1_precision(M.u_Q, u_rho; noise=M.noise)
        u_raw ~ MvNormalCanon(zeros(M.u_N), u_Q)
        u_eta_full = (u_raw .* u_sigma)[M.u_idx]
    elseif M.model_season == "rw2" && haskey(M, :u_N) && M.u_N > 1
        u_Q = build_bstm_rw2_precision(M.u_Q, u_sigma; noise=M.noise)
        u_raw ~ MvNormalCanon(zeros(M.u_N), u_Q)
        u_eta_full = (u_raw .* u_sigma)[M.u_idx]
    elseif M.model_season == "gp" && haskey(M, :u_N) && M.u_N > 1
        u_ls ~ InverseGamma(3, 3)
        K_s = kernelmatrix(SqExponentialKernel() ∘ ScaleTransform(1.0/u_ls), 1.0:Float64(M.u_N)) + M.noise*I
        u_gp ~ MvNormal(zeros(M.u_N), K_s)
        u_eta_full = (u_gp .* u_sigma)[M.u_idx]
    elseif M.model_season == "iid" && haskey(M, :u_N) && M.u_N > 1
        s_iid ~ MvNormal(zeros(M.u_N), I)
        u_eta_full = (s_iid .* u_sigma)[M.u_idx]
    else
        u_eta_full = zeros(M.y_N)
    end
    
    # --- 4. Space-Time Interaction (Knorr-Held Types 0-4) ---
    local st_sigma
    if M.model_st > 0
        st_sigma ~ Exponential(0.5)
    else
        st_sigma = 0.0
    end
    # --- 4. Space-Time Interaction (Knorr-Held Types 0-4) ---
    local st_eta = zeros(T, M.s_N, M.t_N)

    if M.model_st == 0
        st_eta = zeros(T, M.s_N, M.t_N)

    elseif M.model_st == 1 # Type I: IID Interaction
        st_raw ~ MvNormal(zeros(M.s_N * M.t_N), I)
        st_eta = reshape(st_raw .* st_sigma, M.s_N, M.t_N)

    elseif M.model_st == 2 # Type II: Temporal Structure (I ⊗ Q_t)
        # Each area has an independent structured temporal trend
        st_Q2 = Symmetric(kron(I(M.s_N), t_Q) )
        st_raw ~ MvNormalCanon(zeros(M.s_N * M.t_N), st_Q2)
        st_eta = reshape(st_raw .* st_sigma, M.s_N, M.t_N)

    elseif M.model_st == 3 # Type III: Spatial Structure (Q_s ⊗ I)
        # Each time point has an independent structured spatial field
        st_Q3 = Symmetric(kron(M.s_Q, I(M.t_N)) )
        st_raw ~ MvNormalCanon(zeros(M.s_N * M.t_N), st_Q3)
        st_eta = reshape(st_raw .* st_sigma, M.s_N, M.t_N)

    elseif M.model_st == 4 # Type IV: Inseparable (Q_s ⊗ Q_t)
        # Fully inseparable spatiotemporal structure
        st_Q4 = Symmetric(kron(M.s_Q, t_Q) )
        st_raw ~ MvNormalCanon(zeros(M.s_N * M.t_N), st_Q4)
        st_eta = reshape(st_raw .* st_sigma, M.s_N, M.t_N)
    end

    # --- 5. Linear Predictor Construction ---
    # Fixed effects
    local d_beta
    if M.fixed_N > 0
        d_beta ~ MvNormal(zeros(M.fixed_N), I)
    else
        d_beta = zeros(M.fixed_N)
    end

    z_beta_eta ~ filldist(Normal(0, 1), M.N_z) # Coefficients for each Z latent process
    w_beta_eta ~ MvNormal(zeros(M.N_w), I)

    eta = M.log_offset .+ s_eta .+ t_eta_full .+ u_eta_full .+ (z_latent * z_beta_eta) .+ (w_latent * w_beta_eta) # Summing contributions
    
    if M.fixed_N > 0; eta .+= M.fixed * d_beta; end

    if M.model_st > 0
        for i in 1:M.y_N; eta[i] += st_eta[M.s_idx[i], M.t_idx[i]]; end
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


 
;;
