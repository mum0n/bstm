

function _assemble_linear_predictor(N_tot, N_samples, M, PS, Xfixed_betas, mixed_eff_coeffs, 
                                    mixed_terms_list, basis_eff_accum, st_eff_map, svc_slopes, 
                                    svc_covs, s_eff_struct, s_eff_noisy, t_eff, u_eff)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: The primary internal engine for assembling the final linear predictor `eta`. It
              superimposes all recovered latent fields (spatial, temporal, covariate, etc.) for
              both the observed data and any provided prediction grid. This function appears to be
              an older or duplicated version.
    """
    # Initialize the high-fidelity linear predictor container
    local eta_samples = zeros(Float64, N_tot, N_samples)

    # Phase 1: Dimensional Enforcement with Broadcast-Aware Matrix utility
    # We extract the first element of manifold arrays (standard for univariate) 
    # and ensure they are expanded to [Units x Samples] matrices.
    local S_struct = _ensure_matrix(s_eff_struct[1], M.s_N, N_samples)
    local S_noisy  = _ensure_matrix(s_eff_noisy[1], M.s_N, N_samples)
    local T_field  = _ensure_matrix(t_eff[1], M.t_N, N_samples)
    local U_field  = _ensure_matrix(u_eff, M.u_N, N_samples)
    local B_field  = _ensure_matrix(basis_eff_accum, M.y_N, N_samples)

    for j in 1:N_samples
        # Phase 2: Fixed Effects Superposition
        # We perform a row-wise dot product between the design matrix and sampled coefficients.
        if M.Xfixed_N > 0 && !isnothing(Xfixed_betas)
            # Ensure betas are treated as a flat vector for the dot product
            local beta_j = vec(collect(Xfixed_betas[:, j]))
            for i in 1:N_tot
                local X_mat = i <= M.y_N ? M.Xfixed : PS.Xfixed
                local row_idx = i <= M.y_N ? i : i - M.y_N
                
                # Design Row extraction
                local X_row = vec(collect(X_mat[row_idx, :]))
                
                # Additive contribution of fixed covariates
                eta_samples[i, j] += dot(X_row, beta_j)
            end
        end

        # Phase 3: Structural Manifold Assembly
        # Iterate through every observation (Observed + PS Strata)
        for i in 1:N_tot
            local is_obs = i <= M.y_N
            local src = is_obs ? M : PS
            local idx = is_obs ? i : i - M.y_N

            # Manifold Coordinate Discovery
            local s_ptr = Int(src.s_idx[idx])
            local t_ptr = Int(src.t_idx[idx])
            local u_ptr = Int(src.u_idx[idx])

            # 3.1 Link-Scale Offsets (Training Context Only)
            if is_obs && haskey(M, :log_offset)
                eta_samples[i, j] += M.log_offset[i]
            end

            # 3.2 Additive Spatial Components (Structured + Overdispersion)
            # Rationale: S_struct + (S_noisy - S_struct) = S_noisy total realization.
            eta_samples[i, j] += S_struct[s_ptr, j] + (S_noisy[s_ptr, j] - S_struct[s_ptr, j])

            # 3.3 Temporal Trend Realization
            eta_samples[i, j] += T_field[t_ptr, j]

            # 3.4 Seasonal Periodic Realization
            eta_samples[i, j] += U_field[u_ptr, j]

            # 3.5 Spatiotemporal (ST) Mapping
            # Handles both 3D Sample Tensors [S, T, Sample] and static 2D maps.
            if !isnothing(st_eff_map) && !isempty(st_eff_map)
                if ndims(st_eff_map) == 3
                    eta_samples[i, j] += st_eff_map[s_ptr, t_ptr, j]
                else
                    eta_samples[i, j] += st_eff_map[s_ptr, t_ptr]
                end
            end

            # 3.6 Spatially Varying Coefficients (SVC)
            # If active, we multiply the spatial slope field by the covariate value.
            if !isnothing(svc_slopes) && !isempty(svc_slopes)
                 for (k, c_sym) in enumerate(svc_covs)
                    local col_idx = findfirst(==(c_sym), Symbol.(names(M.Xfixed, 2)))
                    if !isnothing(col_idx)
                        local cov_val = is_obs ? M.Xfixed[idx, col_idx] : PS.Xfixed[idx, col_idx]
                        # Svc_slopes indexed as [Unit, Covariate, Sample]
                        eta_samples[i, j] += svc_slopes[1][s_ptr, k, j] * cov_val
                    end
                end
            end

            # 3.7 Smooth Basis / Spline Components (Observed grid mapping only)
            if is_obs
                eta_samples[i, j] += B_field[i, j]
            end
        end
    end

    # Verification: Matrix dimensions must be [N_tot x N_samples]
    return eta_samples
end



function _discover_manifold_realizations(chain, M, n_samples, outcomes_N, p_names)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: The main internal engine for discovering and extracting all latent manifold
              realizations (spatial, temporal, seasonal, etc.) from a fitted MCMC chain. It
              populates a standardized registry with the posterior samples for each component.
              This function appears to be an older or duplicated version.
    """
    # # 1. Fixed Effect Parameter Hoisting
    # Extracting the design matrix coefficients for all outcomes simultaneously.
    xf_betas = nothing
    if "Xfixed_beta" in p_names || any(occursin.(Ref("Xfixed_beta["), p_names))
        xf_betas = get_params_vector(chain, "Xfixed_beta", M.Xfixed_N * outcomes_N)'
    end

    # # 2. Spatial and Temporal Realization Containers
    # Initializing structures for K outcomes to store primary latent fields.
    s_eff_struct = []
    s_eff_noisy = []
    t_eff = []

    for k in 1:outcomes_N
        # Spatial Field Discovery (Discrete units)
        s_key = outcomes_N > 1 ? "s_latent_$k" : "s_latent"
        if s_key in p_names || any(occursin.(Regex("^$s_key\\["), p_names))
            s_vals = get_params_vector(chain, s_key, M.s_N)'
            push!(s_eff_struct, s_vals)
            push!(s_eff_noisy, s_vals)
        else
            push!(s_eff_struct, zeros(M.s_N, n_samples))
            push!(s_eff_noisy, zeros(M.s_N, n_samples))
        end

        # Temporal Trend Discovery
        t_key = outcomes_N > 1 ? "t_latent_$k" : "t_latent"
        if t_key in p_names || any(occursin.(Regex("^$t_key\\["), p_names))
            push!(t_eff, get_params_vector(chain, t_key, M.t_N)')
        else
            push!(t_eff, zeros(M.t_N, n_samples))
        end
    end

    # # 3. Seasonal Cycle Realization
    # Mapping trigonometric or cyclic random effects based on periodicity metadata.
    u_eff = zeros(M.u_N, n_samples)
    if M.model_season == "harmonic"
        u_alpha = vec(get_params_vector(chain, "u_alpha", 1))
        u_beta = vec(get_params_vector(chain, "u_beta", 1))
        u_sigma = vec(get_params_vector(chain, "u_sigma", 1))
        angles = collect(1:M.u_N) .* (2.0 * pi / M.period)
        for j in 1:n_samples
            u_eff[:, j] .= (u_alpha[j] .* sin.(angles) .+ u_beta[j] .* cos.(angles)) .* u_sigma[j]
        end
    elseif M.model_season == "cyclic" && "u_latent" in p_names
        u_eff .= get_params_vector(chain, "u_latent", M.u_N)'
    end

    # # 4. Spatiotemporal Interaction (Knorr-Held I-IV)
    # Generalizing interaction tensors across all outcomes.
    st_eff_maps = [zeros(M.s_N, M.t_N, n_samples) for _ in 1:outcomes_N]
    for k in 1:outcomes_N
        st_key = outcomes_N > 1 ? "st_latent_$k" : "st_latent"
        if st_key in p_names || any(occursin.(Regex("^$st_key\\["), p_names))
            st_raw = get_params_vector(chain, st_key, M.s_N * M.t_N)'
            for j in 1:n_samples
                st_eff_maps[k][:, :, j] .= reshape(st_raw[:, j], M.s_N, M.t_N)
            end
        end
    end

    # # 5. Spatially Varying Coefficients (SVC)
    # Mapping non-stationary regression slopes for discovered covariates.
    svc_slopes = [zeros(M.s_N, length(M.svc_covariates), n_samples) for _ in 1:outcomes_N]
    for k in 1:outcomes_N
        for (v_idx, c_sym) in enumerate(M.svc_covariates)
            svc_key = outcomes_N > 1 ? "beta_svc_$(c_sym)_$k" : "beta_svc_$c_sym"
            if svc_key in p_names || any(occursin.(Regex("^$svc_key\\["), p_names))
                svc_raw = get_params_vector(chain, svc_key, M.s_N)'
                for j in 1:n_samples
                    svc_slopes[k][:, v_idx, j] .= svc_raw[:, j]
                end
            end
        end
    end

    # # 6. Basis Smooth Accumulation (1D-4D)
    # Projects non-linear basis matrices using recovered weights.
    basis_eff_accum = nothing
    if haskey(M, :basis_matrices) && !isempty(M.basis_matrices)
        basis_eff_accum = zeros(M.y_N, n_samples)
        for v_sym in keys(M.basis_matrices)
            b_sig_key = "sig_basis_$v_sym"
            b_beta_key = "beta_basis_$v_sym"
            if b_sig_key in p_names && (b_beta_key in p_names || any(occursin.(Regex("^$b_beta_key\\["), p_names)))
                b_sig = vec(get_params_vector(chain, b_sig_key, 1))
                b_beta = get_params_vector(chain, b_beta_key, size(M.basis_matrices[v_sym], 2))
                for j in 1:n_samples
                    basis_eff_accum[:, j] .+= (M.basis_matrices[v_sym] * b_beta[j, :]) .* b_sig[j]
                end
            end
        end
    end

    # # 7. Hierarchical Multi-Resolution Scale Recovery
    # Discovery of regional or aggregate spatial latent fields.
    hierarchical_scales = Dict{Symbol, Any}()
    if haskey(M, :spatial_hierarchy) && !isempty(M.spatial_hierarchy)
        for scale_sym in keys(M.spatial_hierarchy)
            h_key = "s_latent_$scale_sym"
            if h_key in p_names || any(occursin.(Regex("^$h_key\\["), p_names))
                n_units_h = M.spatial_hierarchy[scale_sym].n_units
                hierarchical_scales[scale_sym] = get_params_vector(chain, h_key, n_units_h)'
            end
        end
    end

    # # 8. Heteroskedastic Volatility Mapping
    # Extraction of Stochastic Volatility surfaces via RFF projections.
    sv_surface = outcomes_N > 1 ? [ones(M.y_N, n_samples) for _ in 1:outcomes_N] : ones(M.y_N, n_samples)
    if get(M, :use_sv, false)
        for k in 1:outcomes_N
            sig_v_key = outcomes_N > 1 ? "sigma_log_var_k[$k]" : "sigma_log_var"
            beta_v_key = outcomes_N > 1 ? "beta_vol_latent_k[$k]" : "beta_vol_latent"
            
            if sig_v_key in p_names && beta_v_key in p_names
                sig_log_var = vec(get_params_vector(chain, sig_v_key, 1))
                b_vol = get_params_vector(chain, beta_v_key, M.M_rff_sigma)
                for j in 1:n_samples
                    vol_lat = sqrt(2.0 / M.M_rff_sigma) .* cos.(M.vol_proj * b_vol[j, :])
                    target = outcomes_N > 1 ? sv_surface[k] : sv_surface
                    target[:, j] .= exp.((sig_log_var[j] .* vol_lat) ./ 2.0)
                end
            end
        end
    elseif "y_sigma" in p_names
        y_sig = vec(get_params_vector(chain, "y_sigma", 1))
        for j in 1:n_samples
            if outcomes_N > 1
                for k in 1:outcomes_N; sv_surface[k][:, j] .= y_sig[j]; end
            else
                sv_surface[:, j] .= y_sig[j]
            end
        end
    end

    return ( 
        n_samples = n_samples,
        outcomes_N = outcomes_N,
        Xfixed_betas = xf_betas,
        s_eff_struct = s_eff_struct,
        s_eff_noisy = s_eff_noisy,
        t_eff = t_eff,
        u_eff = u_eff,
        st_eff_maps = st_eff_maps,
        svc_slopes = svc_slopes,
        basis_eff_accum = basis_eff_accum,
        sv_surface = sv_surface,
        hierarchical_scales = hierarchical_scales
    )
end




function resolve_technical_primitive_legacy(manifold_name::String, M, priors_dict, scheme::Symbol)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: DEPRECATED. An older version of the function that resolves a parsed module's
              metadata into a concrete `Manifold` struct instance. Use the non-legacy version,
              `resolve_technical_primitive`.
    """
    local resolved_priors = resolve_hyperpriors(manifold_name, priors_dict, scheme)

    # 1. Discrete Spatial Primitives
    if manifold_name == "bym2"
        return BYM2(resolved_priors.rho_prior, resolved_priors.sigma_prior)
    elseif manifold_name == "leroux"
        return Leroux(resolved_priors.rho_prior, resolved_priors.sigma_prior)
    elseif manifold_name == "icar" || manifold_name == "besag"
        # Besag is the mathematical basis for ICAR
        return ICAR(resolved_priors.sigma_prior)

    # 2. Temporal and 1D Processes
    elseif manifold_name == "ar1"
        return AR1(resolved_priors.rho_prior, resolved_priors.sigma_prior)
    elseif manifold_name == "rw1"
        return RW1(resolved_priors.sigma_prior)
    elseif manifold_name == "rw2"
        return RW2(resolved_priors.sigma_prior)

    # 3. Continuous and Spectral Kernels
    elseif manifold_name == "gp"
        kernel_str = string(get(params, :kernel, "squared_exponential"))
        return GP(resolved_priors.lengthscale_prior, resolved_priors.sigma_prior, kernel_str)
    elseif manifold_name == "fitc"
        kernel_str = string(get(params, :kernel, "squared_exponential"))
        n_inducing = get(params, :n_inducing, get(M, :n_inducing, 20))
        return FITC(resolved_priors.lengthscale_prior, resolved_priors.sigma_prior, n_inducing, kernel_str)
    elseif manifold_name == "svgp"
        kernel_str = string(get(params, :kernel, "squared_exponential"))
        n_inducing = get(params, :n_inducing, get(M, :n_inducing, 20))
        return SVGP(resolved_priors.lengthscale_prior, resolved_priors.sigma_prior, n_inducing, kernel_str)
    elseif manifold_name == "warp"
        kernel_str = string(get(params, :kernel, "squared_exponential"))
        n_features = get(params, :n_features, get(M, :nbins, 20))
        return Warp(resolved_priors.lengthscale_prior, resolved_priors.sigma_prior, n_features, kernel_str)
    elseif manifold_name == "nystrom"
        # Low-rank kernel approximation via inducing points
        return Nystrom(resolved_priors.sigma_prior, resolved_priors.lengthscale_prior, get(M, :n_inducing, 10))
    elseif manifold_name == "spde"
        return SPDE(resolved_priors.sigma_prior, resolved_priors.kappa_prior)
    elseif manifold_name == "dag"
        return DAG(get(M, :W, sparse(I(M.s_N))), resolved_priors.sigma_prior)

    # 4. Seasonal and Periodic Structures
    elseif manifold_name == "harmonic"
        return Harmonic(resolved_priors.amplitude_prior, resolved_priors.phase_prior, resolved_priors.sigma_prior, M.period)
    elseif manifold_name == "cyclic"
        return Cyclic(M.period, resolved_priors.sigma_prior)

    # 5. Specialized and Smooth Manifolds
    elseif manifold_name == "mosaic"
        return Mosaic(get(M, :s_coord, zeros(M.s_N, 2)), M.n_regions, true)
    elseif manifold_name == "localadaptive" || manifold_name == "local_adaptive"
        # Riemannian diagonal weighting for non-stationary smoothing
        return LocalAdaptive(resolved_priors.sigma_prior, :local_weights)
    elseif manifold_name == "bcgn"
        # Bipartite Covariate Graph Network
        return BCGN(resolved_priors.sigma_prior, get(M, :bipartite_adj, sparse(I(M.s_N))), get(M, :group_weights, ones(M.s_N)))
    # elseif manifold_name == "KnorrHeld"
        # Space-Time Interaction types I, II, III, IV
    #    return KnorrHeld(:space, :time, 'I', resolved_priors.sigma_prior)
    #elseif manifold_name == "AdvectionDiffusion"
        # Physics-informed transport manifold
    #    return AdvectionDiffusion(:space, :time, nothing, resolved_priors.sigma_prior, 0.1, resolved_priors.sigma_prior)
    
    # 6. Basis-mapped Smooths and Defaults
    elseif manifold_name == "wavelet"
        return Wavelets(:db4, M.nbins, resolved_priors.sigma_prior)
    elseif manifold_name == "network"
        adj_matrix = get(M, :W, sparse(I(M.s_N)))
        flow_direction = get(M, :flow_direction, :bidirectional)
        return NetworkFlow(resolved_priors.sigma_prior, adj_matrix, flow_direction)
    elseif manifold_name == "I"
        return ST_I(resolved_priors.sigma_prior)
    elseif manifold_name == "II"
        return ST_II(resolved_priors.sigma_prior)
    elseif manifold_name == "III"
        return ST_III(resolved_priors.sigma_prior)
    elseif manifold_name == "IV"
        return ST_IV(resolved_priors.sigma_prior)
    elseif manifold_name == "advection"
        # Using some defaults for priors on physical params
        return Advection(Normal(0,1), resolved_priors.sigma_prior)
    elseif manifold_name == "diffusion"
        return Diffusion(Exponential(1.0), resolved_priors.sigma_prior)
    elseif manifold_name == "advection_diffusion"
        return AdvectionDiffusion(Normal(0,1), Exponential(1.0), resolved_priors.sigma_prior)
    elseif manifold_name == "fixed" || manifold_name == "iid"
        return IID(resolved_priors.sigma_prior)
    elseif manifold_name in ["pspline", "bspline", "tps", "rff", "fft"]
        return PSpline(M.nbins, 3, 2, resolved_priors.sigma_prior)
    else
        # Fallback to standard identity overdispersion
        return IID(resolved_priors.sigma_prior)
    end
end


function scale_precision!(Q)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: DEPRECATED. This function is no longer used in favor of the more robust
              `scaling_factor_bym2` which handles disconnected components and uses a more stable
              method for calculating the scaling factor.
    """
    # Description:
    #   Scales a precision matrix using the geometric mean of non-zero eigenvalues.
    #   Essential for ensuring sigma_sp represents marginal variance in BYM2.
    # Inputs:
    #   Q: Precision matrix to be modified in-place.

    eig_vals = eigvals(Matrix(Q))
    # Filter out near-zero eigenvalues associated with the null space
    valid_eigs = filter(x -> x > 1e-6, eig_vals)
    
    scaling_factor = exp(mean(log.(valid_eigs)))
    
    if Q isa Symmetric
        Q.data ./= scaling_factor
    else
        Q ./= scaling_factor
    end
    
    return Q
end


  


function logpdf_gmrf(x, Q)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: DEPRECATED. This function is no longer used. The `MvNormalCanon` distribution from
              `Distributions.jl` is used directly within Turing models for more efficient and
              stable GMRF likelihood calculations.
    """
    # Description: Calculates the log-probability of a Gaussian Markov Random Field.
    # Inputs:
    #   - x: Vector of values.
    #   - Q: Precision matrix.
    # Outputs:
    #   - Log-likelihood value.
    Q_stable = Matrix(Q) + I * 1e-5
    F = cholesky(Symmetric(Q_stable))
    return 0.5 * (logdet(F) - dot(x, Q, x) - length(x) * log(2pi))
end




function predict(model_obj::DynamicPPL.Model, chain, new_data::DataFrame; n_samples::Int=100, alpha=0.05)
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: DEPRECATED. This function is a duplicate of the one defined earlier.
    """
    # Description: Primary engine for projecting recovered latent manifolds onto new spatiotemporal coordinates.
    # Rationale: Standardizing the out-of-sample path to support the full BSTM v06.1 Taxonomy.
    # # BSTM Monolithic Prediction & Projection Engine [v19.38.6 ]
    # Timestamp: 2025-11-06 10:00:00
    # Rationale: Updating out-of-sample projection to support 1D-4D smooths, nested supervisors, and mixed effects.
    # Requirements: 100% parity with v19.38.5 Results Dashboard. Zero-truncation of latent manifolds.

    println("--- Starting BSTM Out-of-Sample Prediction v19.38.6 ---")

    # # 1. Training Metadata Recovery
    # Rationale: M contains the configuration and technical registry established during the modular config pass.
    M_train = model_obj.args.M
    n_samples_total = size(chain, 1)
    n_samps = n_samples_total < n_samples ? n_samples_total : n_samples

    # # 2. Prediction Metadata Configuration (PS)
    # Rationale: PSU utilizes bstm_config to ensure the new data is formatted identically to training data.
    # Note: Dummy responses are used as y_obs is not utilized during the projection phase.
    # We pass the original model's formula and configuration (M_train) to ensure consistency.
    PS = bstm_config(M_train.formula, new_data; pairs(M_train)...)
    # # 3. Manifold Coordinate Alignment
    # Rationale: Aligning out-of-sample points with the training grid for discrete and continuous manifolds.
    
    # Case A: Centroid Alignment for Discrete Manifolds (ICAR/BYM2/Mosaic)
    # If the training model utilized areal units, we map new points to the nearest training centroid.
    if haskey(M_train, :centroids) && !isnothing(M_train.centroids)
        centroids_train = M_train.centroids
        # Standardizing coordinates from new_data
        nx = hasproperty(new_data, :s_x) ? new_data.s_x : zeros(nrow(new_data))
        ny = hasproperty(new_data, :s_y) ? new_data.s_y : zeros(nrow(new_data))
        
        PS_s_idx = Int[]
        for i in 1:nrow(new_data)
            # Find nearest neighbor in the training unit grid
            dists = [sum(((nx[i], ny[i]) .- c).^2) for c in centroids_train]
            push!(PS_s_idx, argmin(dists))
        end
        
        # Injecting aligned indices into the PS metadata
        PS_mut = Dict(pairs(PS))
        PS_mut[:s_idx] = PS_s_idx
        PS = NamedTuple(PS_mut)
    end

    # # 5. Architectural Dispatch for Latent Recovery
    # Rationale: Utilizing the validated _reconstruct engine to assemble the prediction tensor.
    raw_arch = get(M_train, :model_arch, "univariate")
    arch_type = if raw_arch == "univariate"
        UnivariateArchitecture()
    elseif raw_arch == "multivariate"
        MultivariateArchitecture()
    elseif raw_arch == "multifidelity" || raw_arch == "nested"
        MultifidelityArchitecture()
    else
        UnivariateArchitecture()
    end

    # Subset the chain for the requested number of samples
    chain_sub = chain[1:n_samps]

    # Rationale: _reconstruct treats PS as a grid for Post-Stratification, 
    # which is mathematically equivalent to out-of-sample prediction.
    res = _reconstruct(arch_type, "prediction_projection", chain_sub, M_train, PS, alpha)

    println("--- Projection Complete [New Observations: ", nrow(new_data), "] ---")

    return (
        predictions_denoised = res.predictions_denoised,
        predictions_noisy = res.predictions_noisy,
        pstats = res,
        PS = PS,
        centroids = haskey(PS, :s_coord) ? PS.s_coord : nothing
    )
end



function parse_module_params(params_str::String)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: DEPRECATED. This function is a duplicate of the one defined earlier.
    """
    # BSTM Parameter Extractor and Expression Realizer [v20.2.1]
    # Timestamp: 2025-06-27 15:30:00
    # Rationale: Resolving the evaluation gap by identifying and executing Julia code strings found in DSL parameters.

    d = Dict{Symbol, Any}()
    if isempty(strip(params_str))
        return d
    end

    # Split at depth zero to handle nested structures correctly
    function split_params(input::String)
        parts = String[]
        depth = 0
        current_block = ""
        for char in input
            if char == '(' || char == '['
                depth = depth + 1
            elseif char == ')' || char == ']'
                depth = depth - 1
            end
            if char == ',' && depth == 0
                push!(parts, strip(current_block))
                current_block = ""
            else
                current_block = current_block * char
            end
        end
        if !isempty(strip(current_block))
            push!(parts, strip(current_block))
        end
        return parts
    end

    pairs = split_params(params_str)

    for entry in pairs
        if occursin("=", entry)
            elements = Base.split(entry, "=", limit = 2)
            param_key = Symbol(strip(elements[1]))
            param_val_raw = strip(elements[2])

            # Dynamic Expression Realization
            # If the string contains indexing symbols or call symbols, evaluate it in Main scope.
            if occursin(r"\[|:|[()]", param_val_raw) && 
               !(startswith(param_val_raw, "(") && endswith(param_val_raw, ")")) && 
               !(startswith(param_val_raw, "[") && endswith(param_val_raw, "]"))
                try
                    d[param_key] = Main.eval(Meta.parse(param_val_raw))
                catch e
                    @warn "BSTM Parser: Could not evaluate expression $param_val_raw. Keeping as string. Error: $e"
                    d[param_key] = param_val_raw
                end

            # Collection Handling
            elseif (startswith(param_val_raw, "(") && endswith(param_val_raw, ")")) || 
                   (startswith(param_val_raw, "[") && endswith(param_val_raw, "]"))
                content_inner = strip(param_val_raw[2:end-1])
                parts_inner = split_params(content_inner)
                # Map cleaned values or recursively parse if sub-pairs exist
                d[param_key] = [occursin("=", v) ? parse_module_params(v) : strip(v, ['\'', '"']) for v in parts_inner]

            # String Literals
            elseif (startswith(param_val_raw, "'") && endswith(param_val_raw, "'")) || 
                   (startswith(param_val_raw, "\"") && endswith(param_val_raw, "\""))
                d[param_key] = strip(param_val_raw, ['\'', '"'])

            # Numeric Literals
            elseif !isnothing(tryparse(Int, param_val_raw))
                d[param_key] = parse(Int, param_val_raw)
            elseif !isnothing(tryparse(Float64, param_val_raw))
                d[param_key] = parse(Float64, param_val_raw)

            # Boolean Literals
            elseif param_val_raw == "true"
                d[param_key] = true
            elseif param_val_raw == "false"
                d[param_key] = false

            # Symbolic Keywords
            else
                d[param_key] = Symbol(param_val_raw)
            end
        end
    end

    return d
end


function decompose_bstm_formula(formula_str::String)
    # BSTM Formula Parser v2.1.0
    # Timestamp: 2026-06-25 20:00:00
    # Rationale: Consolidating duplicated function definitions and enhancing module key generation
    # to support multiple `dynamics` calls by incorporating the `model` parameter into the key.
    # This version is based on the more modular, recursive parsing approach.

    # Partition the formula into Left-Hand Side (Outcomes) and Right-Hand Side (Predictors)
    parts = Base.split(formula_str, "~")
    lhs = strip(parts[1])
    rhs = strip(parts[2])

    # # Outcome Discovery
    outcomes = String[]
    if startswith(lhs, "[") && endswith(lhs, "]")
        content = lhs[2:end-1]
        outcomes = [strip(s) for s in Base.split(content, ",")]
    else
        outcomes = [lhs]
    end

    # # Module and Fixed Effect Registries
    terms_raw = _split_terms_at_depth(rhs, "+")
    modules = Dict{String, Any}()
    fixed_effects = String[]
    has_intercept = false

    for term in terms_raw
        t_clean = strip(term)
        if t_clean == "1"
            has_intercept = true
            continue
        end

        # Check if it's a module call
        m_mod = match(r"(\w+)\((.*)\)", t_clean)
        if !isnothing(m_mod) && (lowercase(m_mod.captures[1]) in BSTM_MODULE_KEYWORDS)
            module_data = _parse_module_call(t_clean)
            
            # Generate a unique key for the module.
            # This logic is enhanced to prevent collisions when multiple modules of the same type are used.
            key_parts = [string(module_data[:type])]
            
            if haskey(module_data, :variables) && !isempty(module_data[:variables])
                push!(key_parts, join(module_data[:variables], "_"))
            end

            if haskey(module_data, :params) && haskey(module_data[:params], :model)
                model_val = module_data[:params][:model]
                if model_val isa Symbol || model_val isa String
                    push!(key_parts, string(model_val))
                end
            end

            # To ensure uniqueness if all else is identical, add a counter.
            base_key = join(key_parts, "_")
            module_key = base_key
            counter = 1
            while haskey(modules, module_key)
                counter += 1
                module_key = base_key * "_$counter"
            end

            modules[module_key] = module_data
        else
            # It's a fixed effect or algebraic operator
            operator_found = ""
            for op in ["⊗", "otimes", "⊕", "oplus", "∘", "compose", "|>", "pipe"]
                if occursin(op, t_clean)
                    operator_found = op
                    break
                end
            end

            if !isempty(operator_found)
                norm_op = operator_found
                if operator_found == "otimes"; norm_op = "⊗"; end
                if operator_found == "oplus"; norm_op = "⊕"; end
                if operator_found == "compose"; norm_op = "∘"; end
                if operator_found == "pipe"; norm_op = "|>"; end

                sub_terms = _split_terms_at_depth(t_clean, operator_found)

                parsed_components = []
                for st in sub_terms
                    push!(parsed_components, _parse_module_call(st))
                end

                modules["algebraic_" * t_clean] = Dict{Symbol, Any}(
                    :type => :interaction_composition,
                    :components => parsed_components,
                    :operator => Symbol(norm_op)
                )
                continue
            end
            
            # If not a module and not an algebraic operator, it's a fixed effect
            if !isempty(t_clean)
                push!(fixed_effects, t_clean)
            end
        end
    end

    return (outcomes=outcomes, modules=modules, fixed_effects=fixed_effects, has_intercept=has_intercept)
end



function resolve_hyperpriors(m_id::Union{String, Symbol}, user_priors::Dict{String, Any}, scheme::Symbol=:pcpriors)
    # This function maps manifold identifiers to their required hyperpriors,
    # allowing for user overrides and standardized default schemes.

    m_id_str = string(m_id)

    # # 1. Scheme Selection
    # Selects the default prior set based on the chosen scheme.
    defaults = if scheme == :pcpriors
        PC_PRIORS
    elseif scheme == :informative
        INFORMATIVE_PRIORS
    else
        UNINFORMATIVE_PRIORS
    end

    # # 2. Resolved Prior Assembly using a mutable Dictionary
    # This approach is more efficient than repeated merging of NamedTuples.
    res = Dict{Symbol, Any}(
        :sigma_prior => get(user_priors, "sigma", defaults["sigma"]),
        :rho_prior => nothing,
        :lengthscale_prior => nothing,
        :kappa_prior => nothing,
        :amplitude_prior => nothing,
        :phase_prior => nothing
    )

    # # 3. Manifold-Specific Prior Dispatch
    # Assigns default priors for optional parameters based on the manifold type.

    # Manifolds requiring a correlation/mixing parameter (rho)
    if m_id_str in ["bym2", "leroux", "sar", "ar1", "proper_car", "dag", "network"]
        res[:rho_prior] = get(user_priors, "rho", defaults["rho"])
    end

    # Manifolds requiring a lengthscale parameter
    if m_id_str in ["gp", "fitc", "rff", "warp", "nystrom", "decay"]
        res[:lengthscale_prior] = get(user_priors, "lengthscale", defaults["lengthscale"])
    end

    # Manifolds requiring an SPDE-specific kappa parameter
    if m_id_str == "spde"
        res[:kappa_prior] = get(user_priors, "kappa", defaults["kappa"])
    end

    # Manifolds for periodic effects
    if m_id_str == "harmonic"
        res[:amplitude_prior] = get(user_priors, "amplitude", defaults["amplitude"])
        res[:phase_prior] = get(user_priors, "phase", defaults["phase"])
    end

    # # 4. Final Conversion to NamedTuple
    # Returns an immutable, type-stable NamedTuple.
    return NamedTuple(res)
end 


function _reconstruct(arch::MultivariateArchitecture, modelname::String, chain, M, PS, alpha)
    # BSTM Internal Utility v1.1.0
    # Timestamp: 2026-06-27 20:15:00
    # Synopsis: The internal reconstruction engine for multivariate models.
    # Rationale for v1.1.0:
    #     - Implemented the missing logic for out-of-sample prediction of `smooth()` manifolds.
    #       It now correctly generates the basis matrix for the prediction surface (`PS`) and
    #       applies the posterior coefficients to compute the smooth effect on the new data.

    N_samples = size(chain, 1)
    outcomes_N = Int(M.outcomes_N)
    p_names = string.(FlexiChains.parameters(chain))
    N_PS = isnothing(PS) ? 0 : size(PS.Xfixed, 1)
    N_tot = M.y_N + N_PS
    family_str = get(M, :model_family, "gaussian")
    fam_obj = get_model_family(family_str)

    eta = zeros(Float64, N_tot, outcomes_N, N_samples)
    summarized_effects = Dict{Symbol, Any}()

    # Fixed Effects
    if "Xfixed_beta" in p_names
        betas_flat = get_params_vector(chain, "Xfixed_beta", M.Xfixed_N * outcomes_N)'
        for k in 1:outcomes_N
            betas_k = betas_flat[((k-1)*M.Xfixed_N + 1):(k*M.Xfixed_N), :]
            eta[1:M.y_N, k, :] .+= M.Xfixed * betas_k
            if N_PS > 0; eta[(M.y_N+1):end, k, :] .+= PS.Xfixed * betas_k; end
        end
        summarized_effects[:fixed_effects] = summarize_array(betas_flat'; alpha=alpha)
    end

    # Manifold Effects
    latent_innov_samples = zeros(N_tot, outcomes_N, N_samples)
    for spec in M.manifolds
        m_domain = spec.domain
        m_obj = spec.manifold_obj
        var_name = string(spec.var)

        for k in 1:outcomes_N
            latent_name_str = if m_domain == :smooth
                "beta_basis_" * var_name * "_" * string(k)
            else
                "latent_" * string(m_domain) * "_" * var_name * "_" * string(k)
            end

            if latent_name_str in p_names
                n_units = if m_domain == :spatial; M.s_N
                          elseif m_domain == :temporal; M.t_N
                          elseif m_domain == :seasonal; M.u_N
                          elseif m_domain == :smooth; size(M.basis_matrices[spec.var], 2)
                          else 1 end

                latent_samples = get_params_vector(chain, latent_name_str, n_units)'
                summarized_effects[Symbol(latent_name_str)] = summarize_array(latent_samples; alpha=alpha)

                if m_domain in [:spatial, :temporal, :seasonal]
                    indices = if m_domain == :spatial; M.s_idx
                              elseif m_domain == :temporal; M.t_idx
                              else M.u_idx end
                    latent_innov_samples[1:M.y_N, k, :] .+= latent_samples[indices, :]
                    if N_PS > 0
                        ps_indices = if m_domain == :spatial; PS.s_idx
                                     elseif m_domain == :temporal; PS.t_idx
                                     else PS.u_idx end
                        latent_innov_samples[(M.y_N+1):end, k, :] .+= latent_samples[ps_indices, :]
                    end
                elseif m_domain == :smooth
                    B_mat_train = M.basis_matrices[spec.var]
                    eta[1:M.y_N, k, :] .+= B_mat_train * latent_samples
                    if N_PS > 0
                        # --- COMPLETED LOGIC ---
                        # Re-compute basis matrix for the prediction surface (PS)
                        smooth_vars = spec.variables
                        ps_coords = Matrix{Float64}(PS.data[!, Symbol.(smooth_vars)])
                        nbins = hasproperty(m_obj, :nbins) ? m_obj.nbins : 20
                        
                        local B_mat_pred
                        if length(smooth_vars) == 1
                            B_mat_pred = bstm_smooth_basis_1D(string(m_obj.model), ps_coords[:,1], nbins)
                        elseif length(smooth_vars) == 2
                            B_mat_pred = bstm_smooth_basis_2D(string(m_obj.model), ps_coords, nbins)
                        elseif length(smooth_vars) == 3
                            B_mat_pred = bstm_smooth_basis_3D(string(m_obj.model), ps_coords, nbins)
                        else # 4D or more
                            B_mat_pred = bstm_smooth_basis_4D(string(m_obj.model), ps_coords, nbins)
                        end
                        
                        eta[(M.y_N+1):end, k, :] .+= B_mat_pred * latent_samples
                    end
                end
            end
        end
    end

    # Apply LKJ coupling
    if "L_corr" in p_names
        L_corr_samples = get_params_matrix_sizestructured(chain, "L_corr", (outcomes_N, outcomes_N))
        for j in 1:N_samples
            eta[:, :, j] .+= latent_innov_samples[:, :, j] * L_corr_samples[:,:,j]'
        end
    end

    # Predictions
    p_den = zeros(Float64, N_tot, outcomes_N, N_samples)
    y_sigma_samples = zeros(N_tot, outcomes_N, N_samples)
    if "y_sigma" in p_names
        sig_samps = get_params_vector(chain, "y_sigma", outcomes_N)'
        for k in 1:outcomes_N
            y_sigma_samples[:, k, :] .= sig_samps[k, :]'
        end
    end

    for j in 1:N_samples
        for k in 1:outcomes_N
            p_den[:, k, j] .= _apply_link_and_lik(family_str, eta[:, k, j], false)
        end
    end

    summarized_effects[:predictions_denoised] = summarize_array(p_den[1:M.y_N, :, :]; alpha=alpha)
    if N_PS > 0
        summarized_effects[:ps_predictions_denoised] = summarize_array(p_den[(M.y_N+1):end, :, :]; alpha=alpha)
    end
    
    summarized_effects[:family] = family_str
    summarized_effects[:arch] = arch
    return NamedTuple(summarized_effects)
end


 
 



function _reconstruct(arch::MultifidelityArchitecture, modelname::String, chain, M, PS, alpha)
    N_samples = size(chain, 1)
    p_names = string.(FlexiChains.parameters(chain))
    family_str = get(M, :model_family, "gaussian")
    fam_obj = get_model_family(family_str)
    noise = get(M, :noise, 1e-4)

    # --- 1. Parameter Extraction ---
    z_ls_s = get_params_vector(chain, "z_ls", 1)
    z_beta_s = get_params_vector(chain, "z_beta", M.M_rff)
    w_ls_s = get_params_vector(chain, "w_ls", 1)
    w_beta_s = get_params_vector(chain, "w_beta", M.M_rff * 3)
    s_sigma_s = get_params_vector(chain, "s_sigma", 1)
    s_rho_s = get_params_vector(chain, "s_rho", 1)
    s_icar_s = get_params_vector(chain, "s_icar", M.N_areas)
    s_iid_s = get_params_vector(chain, "s_iid", M.N_areas)
    t_sigma_s = get_params_vector(chain, "t_sigma", 1)
    t_rho_s = get_params_vector(chain, "t_rho", 1)
    t_ls_s = get_params_vector(chain, "t_ls", 1)
    t_raw_s = get_params_vector(chain, "t_raw", M.t_N)
    t_gp_s = get_params_vector(chain, "t_gp", M.t_N)
    u_sigma_s = get_params_vector(chain, "u_sigma", 1)
    u_rho_s = get_params_vector(chain, "u_rho", 1)
    u_raw_s = get_params_vector(chain, "u_raw", M.u_N)
    st_sigma_s = get_params_vector(chain, "st_sigma", 1)
    st_raw_s = get_params_vector(chain, "st_raw", M.N_areas * M.t_N)
    z_beta_eta_s = get_params_vector(chain, "z_beta_eta", 1)
    w_beta_eta_s = get_params_vector(chain, "w_beta_eta", 3)

    # --- 2. Latent Field Reconstruction ---
    eta_samples = zeros(M.N_obs, N_samples)
    z_latent_samples = zeros(size(M.z_coords_s, 1), N_samples)
    w_latent_samples = zeros(size(M.w_coords_st, 1), 3, N_samples)
    s_eta_samples = zeros(M.N_obs, N_samples)
    t_eta_samples = zeros(M.N_obs, N_samples)
    u_eta_samples = zeros(M.N_obs, N_samples)
    st_eta_samples = zeros(M.N_obs, N_samples)

    for j in 1:N_samples
        # RFF fields
        z_proj = (M.z_coords_s * (M.W_fixed[1:size(M.z_coords_s, 2), :] ./ z_ls_s[j])) .+ M.b_fixed'
        z_latent_j = M.rff_scale .* (cos.(z_proj) * z_beta_s[j, :])
        z_latent_samples[:, j] = z_latent_j

        w_coords_aug = hcat(M.w_coords_st, z_latent_j[1:size(M.w_coords_st, 1)])
        w_proj = (w_coords_aug * (M.W_fixed[1:size(w_coords_aug, 2), :] ./ w_ls_s[j])) .+ M.b_fixed'
        w_beta_mat = reshape(w_beta_s[j, :], M.M_rff, 3)
        w_latent_j = M.rff_scale .* (cos.(w_proj) * w_beta_mat)
        w_latent_samples[:, :, j] = w_latent_j

        # GMRF fields
        s_eta_samples[:, j] = (s_sigma_s[j] .* (sqrt(s_rho_s[j]) .* s_icar_s[j, :] .+ sqrt(1 - s_rho_s[j]) .* s_iid_s[j, :]))[M.s_idx]

        if M.model_time == "ar1"
            t_eta_samples[:, j] = (t_raw_s[j, :] .* t_sigma_s[j])[M.t_idx]
        elseif M.model_time == "gp"
            t_eta_samples[:, j] = t_gp_s[j, :][M.t_idx]
        else # rw2, iid
            t_eta_samples[:, j] = t_raw_s[j, :][M.t_idx]
        end

        if M.model_season != "none"
            u_eta_samples[:, j] = (u_raw_s[j, :] .* u_sigma_s[j])[M.u_idx]
        end

        if M.model_st != "none"
            st_eta_mat = reshape(st_raw_s[j, :], M.N_areas, M.t_N) .* st_sigma_s[j]
            for i in 1:M.N_obs
                st_eta_samples[i, j] = st_eta_mat[M.s_idx[i], M.t_idx[i]]
            end
        end

        # Assemble final eta
        eta_samples[:, j] = M.log_offset .+ s_eta_samples[:, j] .+ t_eta_samples[:, j] .+ u_eta_samples[:, j] .+ st_eta_samples[:, j] .+
                            (z_latent_j .* z_beta_eta_s[j]) .+ (w_latent_j * w_beta_eta_s[j, :])
    end

    # --- 3. Predictions and WAIC ---
    y_sigma_samples = _extract_volatility(chain, p_names, M.N_obs, N_samples, nothing, M)
    p_denoised, p_noisy, log_lik = _process_ll_and_predictions(fam_obj, eta_samples, chain, M, M.N_obs, N_samples, y_sigma_samples)

    # --- 4. Summarize and Return ---
    summarized_effects[:predictions_denoised] = summarize_array(p_denoised; alpha=alpha)
    summarized_effects[:predictions_noisy] = summarize_array(p_noisy; alpha=alpha)
    summarized_effects[:spatial_structured] = summarize_array(s_eta_samples; alpha=alpha)
    summarized_effects[:temporal] = summarize_array(t_eta_samples; alpha=alpha)
    summarized_effects[:seasonal] = summarize_array(u_eta_samples; alpha=alpha)
    summarized_effects[:spacetime_interaction] = summarize_array(st_eta_samples; alpha=alpha)
    summarized_effects[:z_latent_summary] = summarize_array(z_latent_samples; alpha=alpha)
    summarized_effects[:w_latent_summary] = summarize_array(w_latent_samples; alpha=alpha)
    summarized_effects[:waic] = _compute_waic(log_lik)

    if !isnothing(registry.mixed_eff_coeffs) && !isempty(registry.mixed_eff_coeffs)
        summarized_effects[:mixed_effects] = Dict{Symbol, Any}()
        for (i, term) in enumerate(M.mixed_terms)
            # The samples are [n_cat, n_samples]. summarize_array expects [dims..., samples].
            # It will correctly average over the last dimension.
            summarized_effects[:mixed_effects][term.name] = summarize_array(registry.mixed_eff_coeffs[i]; alpha=alpha)
        end
    end

    if !isnothing(registry.nested_eff) && !all(iszero, registry.nested_eff)
        summarized_effects[:nested_contributions] = summarize_array(registry.nested_eff; alpha=alpha)
    end

    summarized_effects[:log_lik_matrix] = log_lik
    summarized_effects[:family] = family_str
    summarized_effects[:arch] = arch
    
    return NamedTuple(summarized_effects)
end
 


function _reconstruct(arch::UnivariateArchitecture, modelname::String, chain, M, PS, alpha)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: The internal reconstruction engine for univariate models. It discovers all latent
              fields from the MCMC chain, assembles the linear predictor, and generates
              predictions, summaries, and diagnostic metrics.
    """
    # BSTM Modular Reconstruction Engine v22.3.0
    # Timestamp: 2026-06-25 19:25:00
    # Rationale: Integrating post-stratification weight calculation and summarization
    # to provide a complete set of outputs for weighted population estimates.

    n_samples = size(chain, 1)
    p_names = string.(FlexiChains.parameters(chain))
    N_PS = isnothing(PS) ? 0 : size(PS.Xfixed, 1)
    N_tot = M.y_N + N_PS
    family_str = get(M, :model_family, "gaussian")
    fam_obj = get_model_family(family_str)

    # 1. Parameter and Latent Field Discovery
    registry = _discover_manifold_realizations(chain, M, n_samples, 1, p_names)

    # 2. Linear Predictor Assembly for Training and Prediction Grids
    eta_samples = _modular_eta_assembly(
        N_tot, registry, M, PS
    )

    # 3. Summarize Primary Latent Effects
    summarized_effects = Dict{Symbol, Any}()

    if !isnothing(registry.s_eff_struct) && !isempty(registry.s_eff_struct)
        s_denoised_samples = registry.s_eff_struct[1][M.s_idx, :]
        summarized_effects[:spatial_denoised] = summarize_array(s_denoised_samples; alpha=alpha)
    end
    if !isnothing(registry.s_eff_noisy) && !isempty(registry.s_eff_noisy)
        s_noisy_samples = registry.s_eff_noisy[1][M.s_idx, :]
        summarized_effects[:spatial_noisy] = summarize_array(s_noisy_samples; alpha=alpha)
    end
    if !isnothing(registry.t_eff) && !isempty(registry.t_eff)
        t_samples = registry.t_eff[1][M.t_idx, :]
        summarized_effects[:temporal] = summarize_array(t_samples; alpha=alpha)
    end
    if !isnothing(registry.u_eff) && !all(iszero, registry.u_eff)
        u_samples = registry.u_eff[M.u_idx, :]
        summarized_effects[:seasonal] = summarize_array(u_samples; alpha=alpha)
    end
    if !isnothing(registry.st_eff_maps) && !isempty(registry.st_eff_maps)
        st_samples = zeros(M.y_N, n_samples)
        for j in 1:n_samples
            for i in 1:M.y_N
                st_samples[i, j] = registry.st_eff_maps[1][M.s_idx[i], M.t_idx[i], j]
            end
        end
        summarized_effects[:spacetime_interaction] = summarize_array(st_samples; alpha=alpha)
    end
    if !isnothing(registry.basis_eff_accum)
        summarized_effects[:smooth_effects] = summarize_array(registry.basis_eff_accum[1:M.y_N, :]; alpha=alpha)
    end
    if !isnothing(registry.Xfixed_betas)
        summarized_effects[:fixed_effects] = summarize_array(registry.Xfixed_betas'; alpha=alpha)
    end

    if !isnothing(registry.mixed_eff_coeffs) && !isempty(registry.mixed_eff_coeffs)
        summarized_effects[:mixed_effects] = Dict{Symbol, Any}()
        for (i, term) in enumerate(M.mixed_terms)
            # The samples are [n_cat, n_samples]. summarize_array expects [dims..., samples].
            # It will correctly average over the last dimension.
            summarized_effects[:mixed_effects][term.name] = summarize_array(registry.mixed_eff_coeffs[i]; alpha=alpha)
        end
    end

    if !isnothing(registry.nested_eff) && !all(iszero, registry.nested_eff)
        summarized_effects[:nested_contributions] = summarize_array(registry.nested_eff; alpha=alpha)
    end

    # 4. Generate Predictions and Compute Log-Likelihood
    # Reshape eta_samples to 2D for univariate case, as _process_ll_and_predictions expects [obs x samples]
    eta_samples_2d = reshape(eta_samples, N_tot, n_samples)
    p_denoised, p_noisy, log_lik = _process_ll_and_predictions(
        fam_obj, eta_samples_2d, chain, M, N_tot, n_samples, registry.sv_surface
    )

    # 5. Summarize Predictions and Post-Stratification Weights
    summarized_effects[:eta] = summarize_array(eta_samples_2d[1:M.y_N, :]; alpha=alpha)
    summarized_effects[:predictions_denoised] = summarize_array(p_denoised[1:M.y_N, :]; alpha=alpha)
    summarized_effects[:predictions_noisy] = summarize_array(p_noisy[1:M.y_N, :]; alpha=alpha)
    if N_PS > 0
        summarized_effects[:ps_predictions_denoised] = summarize_array(p_denoised[(M.y_N+1):end, :]; alpha=alpha)
        summarized_effects[:ps_predictions_noisy] = summarize_array(p_noisy[(M.y_N+1):end, :]; alpha=alpha)

        # Calculate and summarize post-stratification weights
        ps_weights_samples = _calculate_ps_weights(p_denoised, M, PS, N_PS, n_samples)
        if !isnothing(ps_weights_samples)
            summarized_effects[:ps_weights_raw] = ps_weights_samples
            summarized_effects[:ps_weights] = summarize_array(ps_weights_samples; alpha=alpha)
        end
    end

    # 6. Final Diagnostics and Metadata
    summarized_effects[:waic] = _compute_waic(log_lik)
    summarized_effects[:log_lik_matrix] = log_lik
    summarized_effects[:family] = family_str
    summarized_effects[:arch] = arch

    return NamedTuple(summarized_effects)
end




# # BSTM Monolithic Visualization Engine [v19.38.5 ]
function model_results_plots(res, ts=1; outcome=1, centroids=nothing, polygons=nothing, y_obs=nothing)
    # v1.0.1 (2026-06-29 16:13:05)
    # v1.0.2 (2026-06-30)
    # Purpose: Creates a dashboard of plots for visualizing model results.
    # Change: Corrected variable names for latent fields to match the output of `model_results_comprehensive`.
    #         Added plots for `spatial_unstructured` and `spacetime_interaction`.
    # Inputs: res (results object), and various plotting options.
    # Outputs: A composite Plots.jl plot object.
    println("--- Rendering Dashboard v19.38.6 [Outcome: ", outcome, "] ---")

    pstats = res.pstats
    obs_src = isnothing(y_obs) ? res.y_obs : y_obs
    plots_list = []

    # Panel 1: Posterior Predictive Check (PPC)
    is_mv = (obs_src isa AbstractMatrix && size(obs_src, 2) > 1)
    y_p = is_mv ? pstats.predictions_denoised.mean[:, outcome] : vec(pstats.predictions_denoised.mean)
    y_o = is_mv ? obs_src[:, outcome] : vec(obs_src)

    if length(y_p) == length(y_o)
        p_ppc = Plots.scatter(vec(y_p), vec(y_o), title="PPC Audit", xlabel="Predicted", ylabel="Observed", alpha=0.5, markersize=3, markerstrokewidth=0, legend=false)
        clean_p = filter(x -> !isnan(x), y_p)
        clean_o = filter(x -> !isnan(x), y_o)
        if !isempty(clean_p) && !isempty(clean_o)
            min_val = min(minimum(clean_p), minimum(clean_o))
            max_val = max(maximum(clean_p), maximum(clean_o))
            Plots.plot!(p_ppc, [min_val, max_val], [min_val, max_val], color=:red, ls=:dash, lw=1.5)
        end
        push!(plots_list, p_ppc)
    end

    # Helper function for spatial plots
    function create_spatial_plot(field_data, title_str)
        if !isnothing(field_data) && hasproperty(field_data, :mean)
            s_mean = vec(collect(field_data.mean))
            if !all(iszero, s_mean)
                p_map = Plots.plot(aspect_ratio=:equal, title=title_str, frame=:box, colorbar=true)
                if !isnothing(polygons) && length(polygons) >= length(s_mean)
                    for i in 1:length(s_mean)
                        px = [pt[1] for pt in polygons[i]]
                        py = [pt[2] for pt in polygons[i]]
                        if !isempty(px) && (px[1], py[1]) != (px[end], py[end])
                            push!(px, px[1]); push!(py, py[1])
                        end
                        Plots.plot!(p_map, px, py, seriestype=:shape, fill_z=s_mean[i], c=:viridis, linecolor=:black, lw=0.2, label=nothing)
                    end
                elseif !isnothing(centroids)
                    cx = [c[1] for c in centroids]
                    cy = [c[2] for c in centroids]
                    Plots.scatter!(p_map, cx, cy, marker_z=vec(s_mean), markersize=4, c=:viridis, label=nothing)
                end
                return p_map
            end
        end
        return nothing
    end

    # Spatial Plots
    if !isnothing(polygons) || !isnothing(centroids)
        # Structured Spatial Effect
        s_field_structured = (pstats.arch isa MultivariateArchitecture || pstats.arch isa MultifidelityArchitecture) ? get(pstats, :spatial_structured, [nothing])[outcome] : get(pstats, :spatial_structured, nothing)
        p_spatial_struct = create_spatial_plot(s_field_structured, "Spatial Structured Effect")
        !isnothing(p_spatial_struct) && push!(plots_list, p_spatial_struct)

        # Unstructured Spatial Effect
        s_field_unstructured = (pstats.arch isa MultivariateArchitecture || pstats.arch isa MultifidelityArchitecture) ? get(pstats, :spatial_unstructured, [nothing])[outcome] : get(pstats, :spatial_unstructured, nothing)
        p_spatial_unstruct = create_spatial_plot(s_field_unstructured, "Spatial Unstructured Effect")
        !isnothing(p_spatial_unstruct) && push!(plots_list, p_spatial_unstruct)

        # Spacetime Interaction
        st_field = (pstats.arch isa MultivariateArchitecture || pstats.arch isa MultifidelityArchitecture) ? get(pstats, :spacetime_interaction, [nothing])[outcome] : get(pstats, :spacetime_interaction, nothing)
        p_st_interaction = create_spatial_plot(st_field, "Spacetime Interaction (t=$ts)")
        !isnothing(p_st_interaction) && push!(plots_list, p_st_interaction)
    end

    # Temporal Plots
    # Corrected key from :temporal_effects to :temporal
    if hasproperty(pstats, :temporal)
        raw_t = pstats.temporal
        t_field = (pstats.arch isa MultivariateArchitecture || pstats.arch isa MultifidelityArchitecture) ? raw_t[outcome] : raw_t
        if !isnothing(t_field) && hasproperty(t_field, :mean) && !all(iszero, t_field.mean)
            tm, tl, tu = vec(t_field.mean), vec(t_field.lower), vec(t_field.upper)
            p_trend = Plots.plot(tm, ribbon=(tm .- tl, tu .- tm), title="Temporal Trend", lw=2, fillalpha=0.2, color=:royalblue, legend=false, xlabel="Time Index")
            push!(plots_list, p_trend)
        end
    end

    # Seasonal Plot
    if hasproperty(pstats, :seasonal) && !isnothing(pstats.seasonal) && hasproperty(pstats.seasonal, :mean) && !all(iszero, pstats.seasonal.mean)
        um, ul, uu = vec(pstats.seasonal.mean), vec(pstats.seasonal.lower), vec(pstats.seasonal.upper)
        p_seas = Plots.plot(um, ribbon=(um .- ul, uu .- um), title="Seasonal Cycle", lw=2, fillalpha=0.2, color=:forestgreen, legend=false, xlabel="Season Bin")
        push!(plots_list, p_seas)
    end

    # Smoothed Covariate Effects
    if hasproperty(pstats, :smooth_effects) && !isnothing(pstats.smooth_effects) && hasproperty(pstats.smooth_effects, :mean) && !all(iszero, pstats.smooth_effects.mean)
        bm, bl, bu = vec(pstats.smooth_effects.mean), vec(pstats.smooth_effects.lower), vec(pstats.smooth_effects.upper)
        p_smooth = Plots.plot(bm, ribbon=(bm .- bl, bu .- bm), title="Accumulated Smooth Effects", lw=1.5, fillalpha=0.1, color=:darkorange, legend=false, xlabel="Observation Index")
        push!(plots_list, p_smooth)
    end

    # Fixed Effects
    if hasproperty(pstats, :fixed_effects) && !isnothing(pstats.fixed_effects) && hasproperty(pstats.fixed_effects, :mean)
        fm, fl, fu = vec(pstats.fixed_effects.mean), vec(pstats.fixed_effects.lower), vec(pstats.fixed_effects.upper)
        n_coeffs = length(fm)
        if n_coeffs > 0
            p_forest = Plots.scatter(fm, 1:n_coeffs, xerror=(fm .- fl, fu .- fm), title="Fixed Effects", xlabel="Coefficient", ylabel="Index", markersize=4, color=:black, legend=false, yticks=(1:n_coeffs, ["β$i" for i in 1:n_coeffs]))
            Plots.vline!(p_forest, [0], color=:red, ls=:dash, lw=1)
            push!(plots_list, p_forest)
        end
    end

    # Final plot assembly
    if !isempty(plots_list)
        n_plots = length(plots_list)
        cols = min(n_plots, 2)
        rows = Int(ceil(n_plots / cols))
        final_plt = Plots.plot(plots_list..., layout=(rows, cols), size=(1200, 350 * rows), margin=5Plots.mm)
        return final_plt
    end

    @warn "BSTM Visualization: No active manifolds discovered for outcome $outcome."
    return nothing
end


function get_params_vector(chain, base_name::String, expected_len::Int)
    # Audited and Hardened Parameter Extraction Engine (v14.0.10)
    # Rationale: This utility acts as the primary data-bridge between raw MCMC chains
    # and the posterior reconstruction assembly. It must robustly handle FlexiChains
    # nesting and ensure that indexed parameters are recovered in numerical order.

    local N_samples = size(chain, 1)
    local all_names = string.(FlexiChains.parameters(chain))

    # 1. Tier 1: Regex-based Indexed Recovery (Numerical Sorting)
    # Rationale: Standard string sorting often places 'beta[10]' before 'beta[2]'.
    # This Regex captures the integer index to ensure strictly numerical alignment.
    local regex = Regex("^" * base_name * "\\[(\\d+)\\]")
    local matched_names = filter(n -> occursin(regex, n), all_names)

    if !isempty(matched_names)
        # Sort by the captured integer index to maintain dimensional alignment
        sort!(matched_names, by = n -> parse(Int, match(regex, n).captures[1]))

        local res_mat = zeros(Float64, N_samples, length(matched_names))

        for (idx, n) in enumerate(matched_names)
            local val_obj = chain[Symbol(n)]
            # Extract raw data while handling potential vector wrapping from FlexiChains
            local raw = hasproperty(val_obj, :data) ? val_obj.data : collect(val_obj)

            # Map each sample to a Float64 scalar. If the sample is a 1-element vector, extract index 1.
            for s in 1:N_samples
                local v = raw[s]
                res_mat[s, idx] = (v isa AbstractVector) ? Float64(v[1]) : Float64(v)
            end
        end

        # Standard Scalar Broadcast: If only one index is found but multiple are expected,
        # we broadcast the column to the expected length.
        if size(res_mat, 2) == 1 && expected_len > 1
            return repeat(res_mat, 1, expected_len)
        end

        return res_mat
    end

    # 2. Tier 2: Vectorized/Single Entity Recovery
    # Handles cases where parameters are stored as a single vector (e.g., 's_sigma_arr')
    if base_name in all_names
        local val_obj = chain[Symbol(base_name)]
        local raw_data = hasproperty(val_obj, :data) ? val_obj.data : collect(val_obj)

        # Standardize to Matrix [Samples x Params]
        # We iterate through samples to flatten any nested Matrix{Vector} artifacts.
        local mat_data = if eltype(raw_data) <: AbstractVector
             reduce(hcat, [vec(collect(v)) for v in raw_data])'
        else
             Matrix{Float64}(reshape(collect(raw_data), N_samples, :))
        end

        # Dimensional Realignment and Transpose Logic
        if size(mat_data, 2) == expected_len
            return mat_data
        elseif size(mat_data, 1) == expected_len && size(mat_data, 2) != expected_len
            # Transpose if the chain orientation is [Params x Samples]
            return mat_data'
        elseif size(mat_data, 2) == 1
            # Scalar Broadcast fallback
            return repeat(mat_data, 1, expected_len)
        else
            # Final fallback: return the raw data if it matches expected_len after collect
            return reshape(vec(mat_data), N_samples, :)
        end
    end

    # 3. Null Safety Fallback
    # If the parameter is missing, return a zero-matrix to prevent downstream assembly failure.
    @warn "get_params_vector: Parameter '$base_name' not discovered in chain. Initializing with zeros (len=$expected_len)."
    return zeros(Float64, N_samples, expected_len)
end



function _get_latent_field(chain, p_names, base_name, n_units, n_samples, outcomes_N, k, spec)
    # v1.0.2 (2026-06-30)
    # Purpose: Robustly extracts latent field parameters from an MCMC chain by searching
    #          through a prioritized list of possible parameter names.
    # Inputs: chain, p_names (all param names), base_name (e.g., "latent_spatial_struct"), 
    #         n_units, n_samples, outcomes_N, k (outcome index), spec (manifold spec).
    # Outputs: A matrix of posterior samples for the latent field.

    var_name = string(spec.var)

    # Define a prioritized list of possible parameter names to search for.
    # This handles different naming conventions (e.g., with/without var name, with/without outcome index).
    name_candidates = [
        "$(base_name)_$(var_name)_$(k)", # e.g., latent_spatial_struct_s_idx_1
        "$(base_name)_$(k)",             # e.g., latent_spatial_struct_1
        "$(base_name)_$(var_name)",      # e.g., latent_spatial_struct_s_idx (for univariate models)
        base_name                        # e.g., latent_spatial_struct or latent_temporal
    ]

    # Find the first matching name in the chain's parameter list.
    found_idx = findfirst(name -> name in p_names, name_candidates)
    final_name = isnothing(found_idx) ? base_name : name_candidates[found_idx]
    
    return get_params_vector(chain, final_name, n_units)
end




function extract_manifold(m_obj::Union{ICAR, Besag, RW1, RW2, AR1, Leroux, SAR, Cyclic}, chain, M, n_samples, outcomes_N, p_names, spec)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Extracts posterior samples for standard GMRF manifold types.
    # Inputs: m_obj, chain, M, n_samples, outcomes_N, p_names, spec.
    # Outputs: A NamedTuple containing structured and noisy fields.
    structured_fields = Vector{Matrix{Float64}}()
    
    for k in 1:outcomes_N
        var_name = string(spec.var)
        m_domain = spec.domain

        sigma_name = outcomes_N > 1 ? "sigma_$(m_domain)_$(var_name)_$(k)" : "sigma_$(m_domain)_$(var_name)"
        if !(sigma_name in p_names) && outcomes_N == 1
            sigma_name = "sigma_$(m_domain)_$(k)" # Fallback for generic name
        end
        
        n_units = if m_domain == :spatial; M.s_N; elseif m_domain == :temporal; M.t_N; else M.u_N; end

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        latent_samples = _get_latent_field(chain, p_names, "latent_$(m_domain)", n_units, n_samples, outcomes_N, k, spec)
        
        # Transpose to [n_units, n_samples] and apply scaling
        effect = latent_samples' .* sigma_samples'
        push!(structured_fields, effect)
    end
    
    return (structured=structured_fields, noisy=structured_fields)
end

function extract_manifold(m_obj::BYM2, chain, M, n_samples, outcomes_N, p_names, spec)
    # v1.0.2 (2026-06-30)
    # Purpose: Extracts posterior samples for the BYM2 manifold.
    # Inputs: m_obj, chain, M, n_samples, outcomes_N, p_names, spec.
    # Outputs: A NamedTuple containing structured and noisy fields.
    structured_fields = Vector{Matrix{Float64}}()
    noisy_fields = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        var_name = string(spec.var)
        m_domain = spec.domain

        sigma_name = outcomes_N > 1 ? "sigma_$(m_domain)_$(var_name)_$(k)" : "sigma_$(m_domain)_$(var_name)"
        rho_name = outcomes_N > 1 ? "rho_$(m_domain)_$(var_name)_$(k)" : "rho_$(m_domain)_$(var_name)"
        if !(sigma_name in p_names) && outcomes_N == 1
            sigma_name = "sigma_$(m_domain)_$(k)"
            rho_name = "rho_$(m_domain)_$(k)"
        end
        
        sigma_samples = get_params_vector(chain, sigma_name, 1)
        rho_samples = get_params_vector(chain, rho_name, 1)
        
        struct_samples = _get_latent_field(chain, p_names, "latent_$(m_domain)_struct", M.s_N, n_samples, outcomes_N, k, spec)
        iid_samples = _get_latent_field(chain, p_names, "latent_$(m_domain)_iid", M.s_N, n_samples, outcomes_N, k, spec)
        
        # Transpose to [n_units, n_samples] and apply scaling
        structured_effect = (struct_samples' .* sqrt.(rho_samples')) .* sigma_samples'
        noisy_effect = (iid_samples' .* sqrt.(1.0 .- rho_samples')) .* sigma_samples' .+ structured_effect
        
        push!(structured_fields, structured_effect)
        push!(noisy_fields, noisy_effect)
    end
    
    return (structured=structured_fields, noisy=noisy_fields)
end

function extract_manifold(m_obj::IID, chain, M, n_samples, outcomes_N, p_names, spec)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Extracts posterior samples for the IID manifold.
    # Inputs: m_obj, chain, M, n_samples, outcomes_N, p_names, spec.
    # Outputs: A NamedTuple containing structured and noisy fields.
    structured_fields = Vector{Matrix{Float64}}()
    
    for k in 1:outcomes_N
        var_name = string(spec.var)
        m_domain = spec.domain

        sigma_name = outcomes_N > 1 ? "sigma_$(m_domain)_$(var_name)_$(k)" : "sigma_$(m_domain)_$(var_name)"
        if !(sigma_name in p_names) && outcomes_N == 1
            sigma_name = "sigma_$(m_domain)_$(k)"
        end
        
        n_units = if m_domain == :mixed; spec.params.n_cat; elseif m_domain == :spatial; M.s_N; else M.t_N; end
        
        sigma_samples = get_params_vector(chain, sigma_name, 1)
        latent_samples = _get_latent_field(chain, p_names, "latent_$(m_domain)", n_units, n_samples, outcomes_N, k, spec)
        
        # Transpose to [n_units, n_samples] and apply scaling
        effect = latent_samples' .* sigma_samples'
        push!(structured_fields, effect)
    end
    
    return (structured=structured_fields, noisy=structured_fields)
end

function extract_manifold(m_obj::Union{PSpline, BSpline, TPS, RFF, FFT}, chain, M, n_samples, outcomes_N, p_names, spec)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Extracts posterior samples for basis function manifolds.
    # Inputs: m_obj, chain, M, n_samples, outcomes_N, p_names, spec.
    # Outputs: A NamedTuple containing structured and noisy fields.
    var_sym = spec.var
    B_mat = M.basis_matrices[var_sym]
    n_basis_cols = size(B_mat, 2)
    
    structured_fields = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        beta_name = "beta_basis_$(var_sym)_$(k)"
        coeffs = get_params_vector(chain, beta_name, n_basis_cols)
        
        effect = B_mat * coeffs' # Result is [n_obs, n_samples]
        push!(structured_fields, effect)
    end
    
    return (structured=structured_fields, noisy=structured_fields)
end

function extract_manifold(m_obj::DynamicsManifold, chain, M, n_samples, outcomes_N, p_names, spec)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Extracts posterior samples for dynamics manifolds.
    # Inputs: m_obj, chain, M, n_samples, outcomes_N, p_names, spec.
    # Outputs: A NamedTuple containing structured and noisy fields.
    structured_fields = Vector{Matrix{Float64}}()
    
    for k in 1:outcomes_N
        base_name = "dyn_field_$(spec.key)"
        latent_samples = _get_latent_field(chain, p_names, base_name, M.t_N, n_samples, outcomes_N, k, spec)
        
        # Transpose to [n_units, n_samples]
        effect = latent_samples'
        push!(structured_fields, effect)
    end
    
    return (structured=structured_fields, noisy=structured_fields)
end

function extract_manifold(m_obj::MixedManifold, chain, M, n_samples, outcomes_N, p_names, spec)
    # v1.0.0 (2026-06-30)
    # Purpose: Extracts posterior samples for a Mixed Effects manifold (random intercept/slope).
    #          This separates the extraction logic from the main discovery loop for clarity.
    # Inputs: m_obj, chain, M, n_samples, outcomes_N, p_names, spec.
    # Outputs: A NamedTuple containing the structured effect (the random effect coefficients).

    # A mixed effect is a single outcome effect on a set of categories.
    local_outcomes_N = 1
    structured_fields = Vector{Matrix{Float64}}()

    for k in 1:local_outcomes_N # This loop will run once
        var_name = string(spec.var)
        m_domain = spec.domain # This will be :mixed

        sigma_name = "sigma_$(m_domain)_$(var_name)"
        latent_base_name = "latent_$(m_domain)"
        
        n_units = spec.params.n_cat
        
        sigma_samples = get_params_vector(chain, sigma_name, 1)
        latent_samples = _get_latent_field(chain, p_names, latent_base_name, n_units, n_samples, local_outcomes_N, k, spec)
        
        effect = latent_samples' .* sigma_samples'
        push!(structured_fields, effect)
    end
    
    return (structured=structured_fields, noisy=structured_fields)
end

function extract_manifold(m_obj::Harmonic, chain, M, n_samples, outcomes_N, p_names, spec)
    # v1.0.0 (2026-06-30)
    # Purpose: Extracts posterior samples for a Harmonic manifold (seasonal/periodic effects).
    # Inputs: m_obj, chain, M, n_samples, outcomes_N, p_names, spec.
    # Outputs: A NamedTuple containing the structured effect.

    structured_fields = Vector{Matrix{Float64}}()
    m_domain = spec.domain

    # Determine the number of units and basis coordinates based on domain
    local n_units, basis_coords
    if m_domain == :seasonal
        n_units = M.u_N
        basis_coords = 1.0:Float64(n_units)
    elseif m_domain == :temporal
        n_units = M.t_N
        basis_coords = 1.0:Float64(n_units)
    else
        @warn "Harmonic manifold on domain :$m_domain not supported. Returning zero effect."
        return (structured=[zeros(Float64, 1, n_samples) for _ in 1:outcomes_N], noisy=[zeros(Float64, 1, n_samples) for _ in 1:outcomes_N])
    end

    n_harmonics = get(spec.params, :n_harmonics, 2)
    period = get(spec.params, :period, Float64(n_units))
    n_basis_cols = 2 * n_harmonics
    B_mat = zeros(Float64, n_units, n_basis_cols)
    for j in 1:n_harmonics
        omega_j = 2.0 * pi * j / period
        B_mat[:, 2*j - 1] = sin.(omega_j .* basis_coords)
        B_mat[:, 2*j] = cos.(omega_j .* basis_coords)
    end

    for k in 1:outcomes_N
        coeffs = get_params_vector(chain, "beta_basis_$(spec.var)_$(k)", n_basis_cols)
        effect = B_mat * coeffs'
        push!(structured_fields, effect)
    end
    
    return (structured=structured_fields, noisy=structured_fields)
end

function extract_manifold(m_obj::Eigen, chain, M, n_samples, outcomes_N, p_names, spec)
    # v1.0.0 (2026-06-30)
    # Purpose: Extracts posterior samples for an EigenManifold (Bayesian PCA factor).
    # Inputs: m_obj, chain, M, n_samples, outcomes_N, p_names, spec.
    # Outputs: A NamedTuple containing the structured effect (the first principal component).

    # 1. Get parameters from chain
    # The parameter names are constructed based on the key of the manifold spec
    key = spec.key
    v_samples = get_params_vector(chain, "v_$(key)", length(spec.params.ltri_indices))
    
    # 2. Get data for the factor model
    eigen_vars = spec.variables # e.g., ["y1", "y2", "w1", "w2", "w3"]
    Y_data = Matrix(M.data[!, Symbol.(eigen_vars)])
    n_vars = length(eigen_vars)
    n_factors = spec.params.n_factors

    # 3. Reconstruct the effect for each sample
    eigen_effect = zeros(Float64, M.y_N, n_samples)

    for j in 1:n_samples
        v_vec = v_samples[j, :]
        v_mat = zeros(Float64, n_vars, n_factors)
        v_mat[spec.params.ltri_indices] .= v_vec
        U = householder_to_eigenvector(v_mat, n_vars, n_factors)
        factors = Y_data * U
        eigen_effect[:, j] = factors[:, 1]
    end

    structured_fields = [eigen_effect]
    return (structured=structured_fields, noisy=structured_fields)
end

function _extract_nested_fields(chain, M, n_samples, outcomes_N, p_names, spec)
    # v1.0.0 (2026-06-30)
    # Purpose: Extracts posterior samples for nested/multifidelity manifolds. This logic was
    #          previously inside the main discovery loop and is now modularized.
    # Inputs: chain, M, n_samples, outcomes_N, p_names, spec.
    # Outputs: A NamedTuple containing the reconstructed z_latent and w_latent fields.

    z_ls_s = get_params_vector(chain, "z_ls", 1)
    z_beta_s = get_params_vector(chain, "z_beta", M.M_rff)
    w_ls_s = get_params_vector(chain, "w_ls", 1)
    w_beta_s = get_params_vector(chain, "w_beta", M.M_rff * 3)
    
    z_latent_field = zeros(size(M.z_coords_s, 1), n_samples)
    w_latent_field = zeros(size(M.w_coords_st, 1), 3, n_samples)

    for j in 1:n_samples
        z_proj = (M.z_coords_s * (M.W_fixed[1:size(M.z_coords_s, 2), :] ./ z_ls_s[j])) .+ M.b_fixed'
        z_latent_j = M.rff_scale .* (cos.(z_proj) * z_beta_s[j, :])
        z_latent_field[:, j] = z_latent_j

        w_coords_aug = hcat(M.w_coords_st, z_latent_j[1:size(M.w_coords_st, 1)])
        w_proj = (w_coords_aug * (M.W_fixed[1:size(w_coords_aug, 2), :] ./ w_ls_s[j])) .+ M.b_fixed'
        w_beta_mat = reshape(w_beta_s[j, :], M.M_rff, 3)
        w_latent_field[:, :, j] = M.rff_scale .* (cos.(w_proj) * w_beta_mat)
    end

    return (z_latent=z_latent_field, w_latent=w_latent_field)
end

# Fallback for other manifold types
function extract_manifold(m_obj::ManifoldModel, chain, M, n_samples, outcomes_N, p_names, spec)
    # v1.0.1 (2026-06-29 17:16:00)
    # Purpose: A fallback method for manifold types without a specific extraction rule.
    # Inputs: m_obj, chain, M, n_samples, outcomes_N, p_names, spec.
    # Outputs: A NamedTuple containing zero-filled fields to prevent downstream errors.

    @warn "No specific `extract_manifold` method for $(typeof(m_obj)). Returning zero matrix."
    zero_field = [zeros(Float64, M.y_N, n_samples) for _ in 1:outcomes_N]
    return (structured=zero_field, noisy=zero_field)
end

 
 
function _discover_manifold_realizations(chain, M, n_samples, outcomes_N, p_names)
    # BSTM Manifold Discovery Engine v1.1.0
    # Timestamp: 2026-06-27 12:45:00
    # Synopsis: The main internal engine for discovering and extracting all latent manifold
    #           realizations from a fitted MCMC chain. It iterates through the model's manifold
    #           registry and uses multiple dispatch to call the correct extraction method for each component.
    # Rationale for v1.1.0: Updated versioning for consistency.
    # --- 1. Initialize Manifold Registries ---
    s_eff_struct = [zeros(Float64, M.s_N, n_samples) for _ in 1:outcomes_N]
    s_eff_noisy  = [zeros(Float64, M.s_N, n_samples) for _ in 1:outcomes_N]
    t_eff = [zeros(Float64, M.t_N, n_samples) for _ in 1:outcomes_N]
    u_eff = zeros(Float64, M.u_N, n_samples)
    basis_coeffs = Dict{Symbol, Matrix{Float64}}()
    st_eff_maps = [zeros(Float64, M.s_N, M.t_N, n_samples) for _ in 1:outcomes_N]
    dynamics_eff = [zeros(Float64, M.t_N, n_samples) for _ in 1:outcomes_N]
    eigen_eff = zeros(Float64, M.y_N, n_samples)
    nested_eff = zeros(Float64, M.y_N, n_samples)
    
    # Registries specific to Multifidelity Architecture
    z_latent_field = nothing
    w_latent_field = nothing

    svc_slopes = !isempty(get(M, :svc_covariates, [])) ? [zeros(Float64, M.s_N, length(M.svc_covariates), n_samples) for _ in 1:outcomes_N] : nothing
    mixed_eff_coeffs = !isempty(get(M, :mixed_terms, [])) ? [zeros(Float64, term.n_cat, n_samples) for term in M.mixed_terms] : nothing
    sv_surface = [ones(Float64, M.y_N, n_samples) for _ in 1:outcomes_N]
    
    xf_betas = nothing
    if M.Xfixed_N > 0 && ("Xfixed_beta" in p_names || any(occursin.(Ref("Xfixed_beta["), p_names)))
        xf_betas = get_params_vector(chain, "Xfixed_beta", M.Xfixed_N * outcomes_N)'
    end

    if get(M, :model_arch, "univariate") == "multifidelity"
        # The `spec` argument is not needed for this monolithic extraction.
        nested_fields = _extract_nested_fields(chain, M, n_samples, outcomes_N, p_names, nothing)
        z_latent_field = nested_fields.z_latent
        w_latent_field = nested_fields.w_latent
    end

    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (z_key, z_meta) in M.nested_manifolds
            rho_samples = get_params_vector(chain, "rho_nested_$(z_key)", 1)

            if haskey(z_meta, :model_space) && z_meta.model_space != "none"
                lat_samples = get_params_vector(chain, "lat_nested_spatial_$(z_key)", z_meta.s_N)' # Transpose to [n_units, n_samples]
                z_s_ptr = z_meta.s_idx
                # This assumes the sub-model's observation count matches the main model's
                if length(z_s_ptr) == M.y_N
                    spatial_eff_sub = lat_samples[z_s_ptr, :] # [sub_obs x n_samples]
                    for j in 1:n_samples
                        nested_eff[:, j] .+= rho_samples[j] .* spatial_eff_sub[:, j]
                    end
                end
            end

            if haskey(z_meta, :Xfixed)
                beta_samples = get_params_vector(chain, "beta_nested_fixed_$(z_key)", size(z_meta.Xfixed, 2))
                fixed_eff_sub = z_meta.Xfixed * beta_samples' # [sub_obs x n_samples]
                if size(fixed_eff_sub, 1) == M.y_N
                    nested_eff .+= fixed_eff_sub .* rho_samples'
                end
            end
        end
    end

    if get(M, :use_sv, false)
        sv_surface = _extract_volatility(chain, p_names, M.y_N, n_samples, outcomes_N, M)
    end

    return (
        Xfixed_betas = xf_betas,
        mixed_eff_coeffs = mixed_eff_coeffs,
        s_eff_struct = s_eff_struct, 
        s_eff_noisy = s_eff_noisy,
        t_eff = t_eff, 
        u_eff = u_eff, 
        basis_coeffs = basis_coeffs,
        st_eff_maps = st_eff_maps, 
        svc_slopes = svc_slopes,
        dynamics_eff = dynamics_eff,
        eigen_eff = eigen_eff,
        nested_eff = nested_eff,
        z_latent = z_latent_field,
        w_latent = w_latent_field,
        sv_surface = outcomes_N == 1 ? sv_surface[1] : sv_surface,
        n_samples = n_samples,
        outcomes_N = outcomes_N
    )
end




function _modular_eta_assembly(N_tot_in, registry, M, PS_in)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Assembles the final linear predictor `eta` from all recovered latent fields.
    # Inputs: N_tot_in, registry, M, PS_in.
    # Outputs: A 3D array of eta samples [N_total x N_outcomes x N_samples].
    n_samples = registry.n_samples
    outcomes_n = registry.outcomes_N
    y_n_train = Int(M.y_N)
    actual_limit = isnothing(PS_in) ? y_n_train : Int(N_tot_in)

    svc_vars = get(M, :svc_covariates, Symbol[])
    n_svc = length(svc_vars)
    
    train_col_names = Symbol.(names(M.Xfixed, 2))
    train_svc_indices = Int[findfirst(==(v), train_col_names) for v in svc_vars]
    
    ps_svc_indices = Int[]
    if !isnothing(PS_in)
        ps_col_names = Symbol.(names(PS_in.Xfixed, 2))
        ps_svc_indices = Int[findfirst(==(v), ps_col_names) for v in svc_vars]
    end

    eta_container = zeros(Float64, actual_limit, outcomes_n, n_samples)

    for j in 1:n_samples
        for k in 1:outcomes_n
            s_f = !isnothing(registry.s_eff_noisy) && !isempty(registry.s_eff_noisy) ? registry.s_eff_noisy[k][:, j] : Float64[]
            t_f = !isnothing(registry.t_eff) && !isempty(registry.t_eff) ? registry.t_eff[k][:, j] : Float64[]
            u_f = !isnothing(registry.u_eff) && !isempty(registry.u_eff) ? registry.u_eff[:, j] : Float64[]
            dyn_f = !isnothing(registry.dynamics_eff) && !isempty(registry.dynamics_eff) ? registry.dynamics_eff[k][:, j] : Float64[]
            dyn_f = !isnothing(registry.dynamics_eff) && !isempty(registry.dynamics_eff) ? registry.dynamics_eff[k][:, j] : Float64[]

            st_f = !isnothing(registry.st_eff_maps) && !isempty(registry.st_eff_maps[k]) ? registry.st_eff_maps[k][:, :, j] : zeros(0, 0)
            svc_f = !isnothing(registry.svc_slopes) && !isempty(registry.svc_slopes[k]) ? registry.svc_slopes[k][:, :, j] : zeros(0, 0)

            xf_n = Int(M.Xfixed_N)
            has_fixed = !isnothing(registry.Xfixed_betas) && xf_n > 0
            beta_slice = has_fixed ? registry.Xfixed_betas[((k-1)*xf_n + 1):(k*xf_n), j] : Float64[]

            for i in 1:actual_limit
                is_obs = i <= y_n_train
                src = is_obs ? M : PS_in
                idx = is_obs ? i : i - y_n_train
                
                target_svc_indices = is_obs ? train_svc_indices : ps_svc_indices

                s_ptr = Int(src.s_idx[idx])
                t_ptr = Int(src.t_idx[idx])
                u_ptr = Int(src.u_idx[idx])

                val = 0.0
                if !isempty(s_f); val += s_f[s_ptr]; end
                if !isempty(t_f); val += t_f[t_ptr]; end
                if !isempty(u_f); val += u_f[u_ptr]; end
                if !isempty(dyn_f); val += dyn_f[t_ptr]; end

                if !isempty(st_f); val += st_f[s_ptr, t_ptr]; end

                if !isempty(svc_f)
                    for v_idx in 1:n_svc
                        col_idx = target_svc_indices[v_idx]
                        if !isnothing(col_idx)
                            val += svc_f[s_ptr, v_idx] * src.Xfixed[idx, col_idx]
                        end
                    end
                end

                if is_obs
                    if !isnothing(registry.basis_eff_accum)
                        val += registry.basis_eff_accum[idx, j]
                    end
                    if haskey(M, :log_offset)
                        val += M.log_offset[idx]
                    end
                end

                if has_fixed
                    val += dot(vec(collect(src.Xfixed[idx, :])), beta_slice)
                end

                eta_container[i, k, j] = val
            end
        end
    end
    
    return eta_container
end

 


function _extract_volatility(chain, name_strs, N_tot, N_samples, outcome_idx=nothing, M=nothing)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Reconstructs the observation volatility (noise) surface from MCMC samples.
    # Inputs: chain, name_strs, N_tot, N_samples, outcome_idx, M (model config).
    # Outputs: A matrix of volatility samples [N_tot x N_samples].
    y_sig_samples = zeros(N_tot, N_samples)
    for j in 1:N_samples
        local sig_y
        if !isnothing(outcome_idx)
            v_key = string("y_sigma_k[", outcome_idx, "]")
            c_key = string("y_sigma_const_k[", outcome_idx, "]")

            if v_key in name_strs
                sig_val = get_params_vector(chain, string("sigma_log_var_k[", outcome_idx, "]"), 1)[j]
                beta_vol_latent_val = get_params_vector(chain, string("beta_vol_latent_k[", outcome_idx, "]"), M.M_rff_sigma)[j, :]
                y_vol_proj_k = M.vol_proj
                sig_y = exp.((sig_val .* (sqrt(2.0 / M.M_rff_sigma) .* cos.(y_vol_proj_k) * beta_vol_latent_val)) ./ 2.0)
            elseif c_key in name_strs # Homoskedastic volatility for specific outcome
                sig_val = Float64(chain[Symbol(c_key)][j])
                sig_y = fill(sig_val, N_tot)
            else
                sig_y = fill(1.0, N_tot)
            end
        elseif !isnothing(M) && get(M, :use_sv, false)
            sigma_log_var_val = get_params_vector(chain, "sigma_log_var", 1)[j]
            beta_vol_latent_val = get_params_vector(chain, "beta_vol_latent", M.M_rff_sigma)[j, :]
            if haskey(M, :vol_proj)
                sig_y = exp.((sigma_log_var_val .* (sqrt(2.0 / M.M_rff_sigma) .* cos.(M.vol_proj) * beta_vol_latent_val)) ./ 2.0)
            else
                @warn "M.vol_proj not found for stochastic volatility reconstruction. Defaulting to 1.0."
                sig_y = fill(1.0, N_tot)
            end
        else
            if "y_sigma" in name_strs
                val = get_params_vector(chain, "y_sigma", 1)[j]
                sig_y = val isa AbstractVector ? vec(collect(val)) : fill(Float64(val), N_tot)
            elseif "y_sigma_const" in name_strs
                sig_val = Float64(chain[:y_sigma_const][j])
                sig_y = fill(sig_val, N_tot)
            else
                sig_y = fill(1.0, N_tot)
            end
        end

        flat_sig = vec(Float64.(collect(sig_y)))

        if length(flat_sig) >= N_tot
            y_sig_samples[:, j] = flat_sig[1:N_tot]
        else
            y_sig_samples[1:length(flat_sig), j] = flat_sig
            y_sig_samples[length(flat_sig)+1:end, j] .= flat_sig[end]
        end
    end
    return y_sig_samples
end

 


function _apply_link_and_lik(family::String, eta::AbstractArray, use_zi::Bool, phi=0.0, r=1.0)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Applies the inverse link function to the linear predictor `eta`.
    # Inputs: family, eta, use_zi, phi, r.
    # Outputs: The expected value `mu` on the response scale.
    local mu

    if family in ["poisson", "negbin", "gamma", "exponential", "inverse_gaussian", "pareto"]
        mu = exp.(eta)

    elseif family in ["bernoulli", "binomial", "beta"]
        mu = logistic.(eta)

    elseif family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
        mu = eta

    else
        mu = eta
    end

    if use_zi
        mu = (1.0 .- phi) .* mu
    end

    return mu
end




function _process_ll_and_predictions(fam, eta, chain, M, N_tot, N_samples, y_sigma_samples=nothing, y_obs_custom=nothing)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Generates predictions and pointwise log-likelihood values.
    # Inputs: fam, eta, chain, M, N_tot, N_samples, y_sigma_samples, y_obs_custom.
    # Outputs: A tuple (denoised_predictions, noisy_predictions, log_likelihood_matrix).
    denoised = zeros(N_tot, N_samples)
    noisy = zeros(N_tot, N_samples)
    log_lik = zeros(N_samples, M.y_N)

    name_strs = string.(FlexiChains.parameters(chain))
    use_zi = get(M, :use_zi, false)
    fam_str = hasproperty(M, :model_family) ? M.model_family : "gaussian"

    for j in 1:N_samples

        sig_y = if !isnothing(y_sigma_samples)
            sig_y = y_sigma_samples[:, j]
        else
            sig_y = _extract_volatility(chain, name_strs, N_tot, N_samples, nothing, M)[:, j]
        end

        r_val = "lik_r" in name_strs ? chain[:lik_r].data[j] : 1.0
        phi_val = "lik_phi" in name_strs ? chain[:lik_phi].data[j] : 0.0
        extra = "extra_params" in name_strs ? chain[:extra_params].data[j] : 1.0

        mu_vec = _apply_link_and_lik(fam_str, eta[:, j], use_zi, phi_val, r_val)
        denoised[:, j] .= mu_vec

        for i in 1:N_tot
            is_obs = i <= M.y_N
            eta_val = eta[i, j]

            if is_obs
                y_vals_src = isnothing(y_obs_custom) ? M.y_obs : y_obs_custom
                lik_obj = bstm_Likelihood(
                    fam_str, [y_vals_src[i]]; sigma_y=sig_y[i], weight=M.weights[i],
                    phi_zi=use_zi ? phi_val : -Inf, r_nb=r_val, trial=M.trials[i], extra_params=extra
                )
                log_lik[j, i] = Distributions.logpdf(lik_obj, eta_val)
            end

            if use_zi && rand() < phi_val
                noisy[i, j] = 0.0
            else
                n_t = Int(is_obs ? M.trials[i] : 1)
                
                temp_lik_obj = bstm_Likelihood(
                    fam_str, [0.0];
                    sigma_y=sig_y[i],
                    r_nb=r_val,
                    trial=n_t,
                    extra_params=extra
                )
                
                dist = get_dist_ref(fam, temp_lik_obj, eta_val, sig_y[i])
                
                noisy[i, j] = rand(dist)
            end
        end
    end

    return denoised, noisy, log_lik
end



function _compute_waic(log_lik)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Computes the Widely Applicable Information Criterion (WAIC).
    # Inputs: log_lik - A matrix of pointwise log-likelihoods [N_samples x N_obs].
    # Outputs: The WAIC value.
    nsamples, nobs = size(log_lik)
    lppd = sum(logsumexp(log_lik[:, i]) - log(nsamples) for i in 1:nobs)
    p_waic = sum(var(log_lik[:, i]) for i in 1:nobs)
    return -2 * (lppd - p_waic)
end




function _calculate_ps_weights(p_denoised, M, PS, N_PS, N_samples)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Calculates post-stratification weights.
    # Inputs: p_denoised, M, PS, N_PS, N_samples.
    # Outputs: A matrix of post-stratification weights [N_PS x N_samples].
    if N_PS == 0
        return nothing
    end

    local ps_weights = zeros(N_PS, N_samples)
    
    for k in 1:N_PS
        local s_target = PS.s_idx[k]
        local t_target = PS.t_idx[k]
        local u_target = PS.u_idx[k]

        local obs_match_idx = findfirst(i -> M.s_idx[i] == s_target && M.t_idx[i] == t_target && M.u_idx[i] == u_target, 1:M.y_N)

        if !isnothing(obs_match_idx)
            for j in 1:N_samples
                ps_weights[k, j] = p_denoised[M.y_N + k, j] / (p_denoised[obs_match_idx, j] + 1e-9)
            end
        else
            local sample_mean_obs = mean(p_denoised[1:M.y_N, :], dims=1)
            for j in 1:N_samples
                ps_weights[k, j] = p_denoised[M.y_N + k, j] / (sample_mean_obs[j] + 1e-9)
            end
        end
    end

    return ps_weights
end


