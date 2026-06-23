using Turing, Distributions, LinearAlgebra, SparseArrays, KernelFunctions
# Manifold structs (e.g., PSpline) are assumed to be imported from spatiotemporal_functions.jl

function get_kernel_from_string(kernel_name::String)
    k_name = lowercase(kernel_name)
    if k_name == "constant"
        return ConstantKernel()
    elseif k_name == "linear"
        return LinearKernel()
    elseif k_name == "matern12" || k_name == "exponential"
        return Matern12Kernel()
    elseif k_name == "matern32"
        return Matern32Kernel()
    elseif k_name == "matern52"
        return Matern52Kernel()
    elseif k_name == "spherical"
        return SphericalKernel()
    elseif k_name == "squared_exponential" || k_name == "se" || k_name == "gaussian" || k_name == "rbf"
        return SqExponentialKernel()
    elseif k_name == "periodic"
        return PeriodicKernel()
    else
        @warn "Kernel '$kernel_name' not recognized. Defaulting to SqExponentialKernel."
        return SqExponentialKernel()
    end
end

@model function bstm_modular(M, ::Type{T}=Float64) where {T}
    # # 1. Global Likelihood Hyperparameters
    # Rationale: Standardizing scalars for the target likelihood family.
    family = M.model_family
    noise = get(M, :noise, 1e-4)
    use_zi = get(M, :use_zi, false)

    lik_r = one(T)
    lik_phi = zero(T)
    extra_p = one(T)

    if family == "negbin"; lik_r ~ NamedDist(Exponential(1.0), :lik_r); end
    if use_zi == true; lik_phi ~ NamedDist(Beta(1, 1), :lik_phi); end
    if family in ["gamma", "beta", "student_t", "inverse_gaussian", "pareto"]; extra_p ~ NamedDist(Exponential(1.0), :extra_params); end

    # # 2. Observation Volatility & Stochastic Volatility (SV)
    # Rationale: Reconstructs heteroskedastic error scales via RFF log-variance projection.
    y_sigma = Vector{T}(undef, M.y_N)
    if get(M, :use_sv, false) == true
        sigma_log_var ~ NamedDist(Exponential(1.0), :sigma_log_var)
        beta_vol_latent ~ NamedDist(filldist(Normal(0, 1), M.M_rff_sigma), :beta_vol_latent)
        vol_proj_field = M.vol_proj * beta_vol_latent
        vol_latent_field = sqrt(2.0 / M.M_rff_sigma) .* cos.(vol_proj_field)
        for i in 1:M.y_N; y_sigma[i] = exp((sigma_log_var * vol_latent_field[i]) / 2.0); end
    else
        y_sigma_val ~ NamedDist(Exponential(1.0), :y_sigma)
        for i in 1:M.y_N; y_sigma[i] = y_sigma_val; end
    end

    # # 3. Base Predictor: Fixed Effects & Link-Scale Offsets
    Xfixed_beta ~ NamedDist(MvNormal(zeros(M.Xfixed_N), 5.0 * I), :Xfixed_beta)
    eta = Vector{T}(M.Xfixed * Xfixed_beta)
    if haskey(M, :log_offset)
        if family in ["gaussian", "student_t", "laplace"]; eta .+= exp.(M.log_offset);
        else; eta .+= M.log_offset; end
    end

    # # 4. Modular Manifold Realization
    # This section iterates through the manifold objects created by the config engine.
    for spec in M.manifolds
        m_obj = spec.manifold_obj
        m_domain = spec.domain
        
        # Realize latent field based on manifold type
        if m_obj isa IID
            var_name = string(spec.var)
            is_svc = m_domain == :svc
            sigma_name = is_svc ? Symbol("sig_svc_", var_name) : Symbol("sigma_", m_domain, "_", var_name)
            latent_name = is_svc ? Symbol("beta_svc_", var_name) : Symbol("latent_", m_domain, "_", var_name)
            sigma ~ NamedDist(m_obj.sigma_prior, sigma_name)

            local n_units, indices
            if m_domain == :mixed
                n_units = spec.params.n_cat
                indices = spec.params.indices
            else # :spatial, :temporal, or :svc (which is spatial)
                n_units = is_svc ? M.s_N : ((m_domain == :spatial) ? M.s_N : ((m_domain == :temporal) ? M.t_N : M.u_N))
                indices = is_svc ? M.s_idx : ((m_domain == :spatial) ? M.s_idx : ((m_domain == :temporal) ? M.t_idx : M.u_idx))
            end

            latent ~ NamedDist(filldist(Normal(0, sigma), n_units), latent_name)

            if is_svc
                x_col_idx = findfirst(==(spec.var), Symbol.(names(M.Xfixed, 2)))
                if !isnothing(x_col_idx)
                    x_vals = M.Xfixed[:, x_col_idx]
                    for i in 1:M.y_N; eta[i] += latent[indices[i]] * x_vals[i]; end
                end
            else
                 for i in 1:M.y_N; eta[i] += latent[indices[i]]; end
            end

        elseif m_obj isa ICAR
            var_name = string(spec.var)
            is_svc = m_domain == :svc
            sigma_name = is_svc ? Symbol("sig_svc_", var_name) : Symbol("sigma_", m_domain)
            latent_name = is_svc ? Symbol("beta_svc_", var_name) : Symbol("latent_", m_domain)
            sigma ~ NamedDist(m_obj.sigma_prior, sigma_name)
            template = M.s_Q_template.matrix # SVC is spatial
            n_units = size(template, 1)
            Q = recompose_precision(:icar, template, sigma; noise=noise)
            latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), latent_name)
            
            if is_svc
                x_col_idx = findfirst(==(spec.var), Symbol.(names(M.Xfixed, 2)))
                if !isnothing(x_col_idx)
                    x_vals = M.Xfixed[:, x_col_idx]
                    for i in 1:M.y_N; eta[i] += latent[M.s_idx[i]] * x_vals[i]; end
                end
            else
                indices = (m_domain == :spatial) ? M.s_idx : M.t_idx
                for i in 1:M.y_N; eta[i] += latent[indices[i]]; end
            end

        elseif m_obj isa BYM2
            var_name = string(spec.var)
            is_svc = m_domain == :svc
            sigma_name = is_svc ? Symbol("sig_svc_", var_name) : Symbol("sigma_", m_domain)
            rho_name = is_svc ? Symbol("rho_svc_", var_name) : Symbol("rho_", m_domain)
            latent_name = is_svc ? Symbol("beta_svc_", var_name) : Symbol("latent_", m_domain)

            sigma ~ NamedDist(m_obj.sigma_prior, sigma_name)
            rho ~ NamedDist(m_obj.rho_prior, rho_name)
            template = M.s_Q_template.matrix # SVC is spatial
            n_units = size(template, 1)
            Q = recompose_precision(:bym2, template, sigma; extra_param=rho, noise=noise)
            latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), latent_name)

            if is_svc
                x_col_idx = findfirst(==(spec.var), Symbol.(names(M.Xfixed, 2)))
                if !isnothing(x_col_idx)
                    x_vals = M.Xfixed[:, x_col_idx]
                    for i in 1:M.y_N; eta[i] += latent[M.s_idx[i]] * x_vals[i]; end
                end
            else
                indices = (m_domain == :spatial) ? M.s_idx : M.t_idx
                for i in 1:M.y_N; eta[i] += latent[indices[i]]; end
            end

        elseif m_obj isa Union{Leroux, SAR, DAG}
            var_name = string(spec.var)
            sigma_name = Symbol("sigma_", m_domain, "_", var_name)
            rho_name = Symbol("rho_", m_domain, "_", var_name)
            latent_name = Symbol("latent_", m_domain, "_", var_name)

            sigma ~ NamedDist(m_obj.sigma_prior, sigma_name)
            rho ~ NamedDist(m_obj.rho_prior, rho_name)
            template = M.s_Q_template.matrix
            n_units = size(template, 1)
            
            m_type_sym = m_obj isa Leroux ? :leroux : (m_obj isa SAR ? :sar : :dag)
            Q = recompose_precision(m_type_sym, template, sigma; extra_param=rho, noise=noise)
            latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), latent_name)
            indices = M.s_idx
            for i in 1:M.y_N; eta[i] += latent[indices[i]]; end

        elseif m_obj isa NetworkFlow
            sigma ~ NamedDist(m_obj.sigma_prior, Symbol("sigma_", m_domain))
            rho ~ NamedDist(Beta(1,1), Symbol("rho_", m_domain))
            template = M.s_Q_template.matrix
            n_units = size(template, 1)
            Q = recompose_precision(:network, template, sigma; extra_param=rho, directed_adj=m_obj.adjacency_matrix, flow_direction=m_obj.flow_direction, noise=noise)
            latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), Symbol("latent_", m_domain))
            indices = M.s_idx
            for i in 1:M.y_N; eta[i] += latent[indices[i]]; end

        elseif m_obj isa Hyperbolic
            sigma ~ NamedDist(m_obj.sigma_prior, Symbol("sigma_", m_domain, "_hyperbolic"))
            n_units = M.s_N
            latent ~ NamedDist(filldist(Normal(0, sigma), n_units), Symbol("latent_", m_domain, "_hyperbolic"))
            indices = M.s_idx
            for i in 1:M.y_N; eta[i] += latent[indices[i]]; end
            
        elseif m_obj isa AR1
            sigma ~ NamedDist(m_obj.sigma_prior, Symbol("sigma_", m_domain))
            rho ~ NamedDist(m_obj.rho_prior, Symbol("rho_", m_domain))
            template = (m_domain == :temporal) ? M.t_Q_template.matrix : M.s_Q_template.matrix # Can be spatial
            n_units = size(template, 1)
            Q = recompose_precision(:ar1, template, sigma; extra_param=rho, noise=noise)
            latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), Symbol("latent_", m_domain))
            indices = (m_domain == :temporal) ? M.t_idx : M.s_idx
            for i in 1:M.y_N; eta[i] += latent[indices[i]]; end

        elseif m_obj isa RW1
            sigma ~ NamedDist(m_obj.sigma_prior, Symbol("sigma_", m_domain))
            template = (m_domain == :temporal) ? M.t_Q_template.matrix : M.s_Q_template.matrix
            n_units = size(template, 1)
            Q = recompose_precision(:rw1, template, sigma; noise=noise)
            latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), Symbol("latent_", m_domain))
            indices = (m_domain == :temporal) ? M.t_idx : M.s_idx
            for i in 1:M.y_N; eta[i] += latent[indices[i]]; end

        elseif m_obj isa RW2
            sigma ~ NamedDist(m_obj.sigma_prior, Symbol("sigma_", m_domain))
            template = (m_domain == :temporal) ? M.t_Q_template.matrix : M.s_Q_template.matrix
            n_units = size(template, 1)
            Q = recompose_precision(:rw2, template, sigma; noise=noise)
            latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), Symbol("latent_", m_domain))
            indices = (m_domain == :temporal) ? M.t_idx : M.s_idx
            for i in 1:M.y_N; eta[i] += latent[indices[i]]; end

        elseif m_obj isa Cyclic
            sigma ~ NamedDist(m_obj.sigma_prior, Symbol("sigma_", m_domain))
            template = M.u_Q_template.matrix
            n_units = size(template, 1)
            Q = recompose_precision(:cyclic, template, sigma; noise=noise)
            latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), Symbol("latent_", m_domain))
            indices = M.u_idx
            for i in 1:M.y_N; eta[i] += latent[indices[i]]; end

        elseif m_obj isa Harmonic
            # For now, we treat it like a cyclic IID effect.
            sigma ~ NamedDist(m_obj.sigma_prior, Symbol("sigma_", m_domain))
            latent ~ NamedDist(filldist(Normal(0, sigma), M.u_N), Symbol("latent_", m_domain))
            indices = M.u_idx
            for i in 1:M.y_N; eta[i] += latent[indices[i]]; end

        elseif m_obj isa Union{PSpline, BSpline, TPS, Wavelets, RFF, FFT}
            var_sym = spec.var
            B_mat = M.basis_matrices[var_sym]
            n_basis_cols = size(B_mat, 2)
            
            sigma ~ NamedDist(m_obj.sigma_prior, Symbol("sigma_basis_", var_sym))
            latent_coeffs ~ NamedDist(filldist(Normal(0, sigma), n_basis_cols), Symbol("beta_basis_", var_sym))
            
            eta .+= B_mat * latent_coeffs
        
        elseif m_obj isa Eigen
            var_syms = spec.var
            data_mat = spec.params.data
            n_dims = spec.params.n_dims
            n_factors = m_obj.n_factors
            ltri_indices = m_obj.ltri_indices
            v_pca ~ NamedDist(filldist(Normal(0, 1.0), length(ltri_indices)), Symbol("v_pca_", join(var_syms, "_")))
            sd_pca ~ NamedDist(filldist(Exponential(1.0), n_factors), Symbol("pca_sd_", join(var_syms, "_")))
            pdef_sd ~ NamedDist(Exponential(1.0), Symbol("pdef_sd_", join(var_syms, "_")))
            
            K_pca, _, _ = householder_transform(v_pca, n_dims, n_factors, ltri_indices, sd_pca, pdef_sd, noise)
            latent_eigen ~ NamedDist(MvNormal(zeros(n_dims), K_pca), Symbol("latent_eigen_", join(var_syms, "_")))
            eta .+= data_mat * latent_eigen

        elseif m_obj isa GP
            var_sym = spec.var
            ls ~ NamedDist(m_obj.lengthscale_prior, Symbol("ls_gp_", var_sym))
            sigma ~ NamedDist(m_obj.sigma_prior, Symbol("sigma_gp_", var_sym))
            coords = spec.params.coords

            kernel_base = get_kernel_from_string(m_obj.kernel)
            k_gp = sigma^2 * kernel_base o ScaleTransform(1/ls)
            
            # Full GP - computationally expensive
            K_ff = kernelmatrix(k_gp, coords) + noise*I
            f_latent ~ NamedDist(MvNormal(zeros(size(coords,1)), K_ff), Symbol("latent_smooth_", var_sym))
            eta .+= f_latent

        elseif m_obj isa Union{SVGP, FITC}
            var_sym = spec.var
            ls ~ NamedDist(m_obj.lengthscale_prior, Symbol("ls_svgp_", var_sym))
            sigma ~ NamedDist(m_obj.sigma_prior, Symbol("sigma_svgp_", var_sym))
            n_inducing = m_obj.n_inducing
            coords = spec.params.coords

            mean_coords = mean(coords, dims=1)
            std_coords = std(coords, dims=1)
            inducing_locs ~ MvNormal(vec(mean_coords), Diagonal(vec(std_coords)))

            kernel_base = get_kernel_from_string(m_obj.kernel)
            k_svgp = sigma^2 * kernel_base o ScaleTransform(1/ls)
            K_uu = kernelmatrix(k_svgp, inducing_locs) + noise*I
            
            u_latent ~ MvNormal(zeros(n_inducing), K_uu)
            
            K_fu = kernelmatrix(k_svgp, coords, inducing_locs)
            f_mean = K_fu * (K_uu \\ u_latent)
            eta .+= f_mean

        elseif m_obj isa Warp
            var_sym = spec.var
            ls_warp ~ NamedDist(m_obj.lengthscale_prior, Symbol("ls_warp_", var_sym))
            sigma_warp ~ NamedDist(m_obj.sigma_prior, Symbol("sigma_warp_", var_sym))
            
            coords = spec.params.coords
            # Pass kernel information to the RFF parameter generator
            W_warp, b_warp = generate_informed_rff_params(coords, m_obj.n_features; kernel_name=m_obj.kernel)
            beta_warp ~ MvNormal(zeros(m_obj.n_features), I)
            
            # The warping field is added directly to the linear predictor.
            warp_effect = rff_map(coords, W_warp ./ ls_warp, b_warp) * beta_warp * sigma_warp
            eta .+= warp_effect

        elseif m_obj isa Union{FFT, Wavelets}
            # This block handles FFT/Wavelet as GMRF-like components for temporal/spatial effects.
            var_name = string(spec.var)
            sigma_name = Symbol("sigma_", m_domain, "_", var_name)
            ls_name = Symbol("ls_", m_domain, "_", var_name)
            latent_name = Symbol("latent_", m_domain, "_", var_name)

            sigma ~ NamedDist(m_obj.sigma_prior, sigma_name)
            ls ~ NamedDist(m_obj.lengthscale_prior, ls_name)
            
            Q_spectral = recompose_precision(:spectral, nothing, sigma; extra_param=ls, kernel=Symbol(m_obj.kernel), noise=noise)
            latent ~ NamedDist(MvNormalCanon(zeros(size(Q_spectral,1)), Q_spectral), latent_name)
            indices = (m_domain == :temporal) ? M.t_idx : M.s_idx
            for i in 1:M.y_N; eta[i] += latent[indices[i]]; end

        elseif spec.domain == :spacetime
            st_sigma ~ NamedDist(Exponential(1.0), :st_sigma)
            st_type = Symbol(M.model_st)
            if st_type in [:I, :II, :III, :IV]
                Q_st = recompose_precision(st_type, M.s_Q_template.matrix, st_sigma; template_t = M.t_Q_template.matrix, noise=noise)
                st_latent ~ NamedDist(MvNormalCanon(zeros(M.s_N * M.t_N), Q_st), :st_latent)
                for i in 1:M.y_N
                    obs_st_idx = (M.t_idx[i] - 1) * M.s_N + M.s_idx[i]
                    eta[i] += st_latent[obs_st_idx]
                end
            elseif st_type in [:advection, :diffusion, :advection_diffusion]
                velocity_phys ~ NamedDist(Normal(0, 1), :velocity_phys)
                K_op = build_transport_operator(M.W, velocity_phys, 1.0)
                phys_innov ~ NamedDist(filldist(Normal(0, st_sigma), M.s_N * M.t_N), :phys_innov)
                phys_field = Vector{T}(undef, M.s_N * M.t_N)

                for t in 1:M.t_N
                    idx_t = ((t-1)*M.s_N + 1):(t*M.s_N)
                    if t == 1; phys_field[idx_t] .= phys_innov[idx_t];
                    else
                        idx_prev = ((t-2)*M.s_N + 1):((t-1)*M.s_N)
                        phys_field[idx_t] .= K_op * phys_field[idx_prev] .+ phys_innov[idx_t]
                    end
                end
                for i in 1:M.y_N
                    obs_st_ptr = (M.t_idx[i] - 1) * M.s_N + M.s_idx[i]
                    eta[i] += phys_field[obs_st_ptr]
                end
            end
        end
    end

    # # 4.2 Mechanistic Dynamics
    if get(M, :use_dynamics, false) == true
        K_dyn ~ NamedDist(get(M.hyperpriors, "K", Normal(150, 10)), :K_dyn)
        r_dyn ~ NamedDist(get(M.hyperpriors, "r", Beta(2, 2)), :r_dyn)
        sig_dyn ~ NamedDist(Exponential(1.0), :sig_dyn)
        
        m_states ~ NamedDist(filldist(Normal(0, 1), M.t_N), :m_states)
        for i in 1:M.y_N
            eta[i] += m_states[M.t_idx[i]] * sig_dyn
        end
    end

    # # 4.1 Nested Hierarchical Supervisors
    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (z_key, z_meta) in M.nested_manifolds
            rho_nested ~ NamedDist(Normal(1.0, 0.5), Symbol("rho_nested_", z_key))

            if haskey(z_meta, :model_space) && z_meta.model_space != "none"
                sig_z_s ~ NamedDist(Exponential(1.0), Symbol("sig_nested_spatial_", z_key))
                Q_z_s = recompose_precision(Symbol(z_meta.model_space), z_meta.s_Q_template.matrix, sig_z_s, noise=noise)
                lat_z_s ~ NamedDist(MvNormalCanon(zeros(z_meta.s_N), Q_z_s), Symbol("lat_nested_spatial_", z_key))

                z_s_ptr = z_meta.s_idx
                for i in 1:M.y_N
                    eta[i] += rho_nested * lat_z_s[z_s_ptr[i]]
                end
            end

            if haskey(z_meta, :Xfixed)
                xf_z = z_meta.Xfixed
                beta_z_f ~ NamedDist(MvNormal(zeros(size(xf_z, 2)), 5.0 * I), Symbol("beta_nested_fixed_", z_key))
                eta .+= rho_nested .* (xf_z * beta_z_f)
            end
        end
    end

    # # 5. Final Pointwise Likelihood Dispatch
    yL = T(get(M, :y_lower_bound, -Inf))
    yU = T(get(M, :y_upper_bound, Inf))
    h = T(get(M, :hurdle, -Inf))

    for i in 1:M.y_N
        d_lik = bstm_Likelihood(
            family, [T(M.y_obs[i])]; sigma_y=[y_sigma[i]], weight=[T(M.weights[i])],
            phi_zi=lik_phi, r_nb=lik_r, trial=[Int(M.trials[i])],
            y_L=yL, y_U=yU, hurdle=h, extra_params=extra_p
        )
        Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i])
    end
end


function _reconstruct(arch::UnivariateArchitecture, modelname::String, chain, M, PS, alpha)
    # BSTM Modular Reconstruction Engine v21.0.1
    # This engine is designed for models created with `bstm_modular_config`.
    # It iterates through the `M.manifolds` registry to reconstruct latent effects.

    if !haskey(M, :manifolds)
        @error "This reconstruction engine requires a model configuration with a `:manifolds` registry."
        return nothing
    end

    N_samples = size(chain, 1)
    p_names = string.(FlexiChains.parameters(chain))
    N_PS = isnothing(PS) ? 0 : size(PS.Xfixed, 1)
    N_tot = M.y_N + N_PS
    family_str = get(M, :model_family, "gaussian")
    fam_obj = get_model_family(family_str)

    # --- 1. Initialize Linear Predictor & Summaries ---
    eta = zeros(Float64, N_tot, N_samples)
    summarized_effects = Dict{Symbol, Any}()

    # --- 2. Base Predictor: Fixed Effects & Offsets ---
    if "Xfixed_beta" in p_names
        betas = get_params_vector(chain, "Xfixed_beta", M.Xfixed_N)'
        eta[1:M.y_N, :] .+= M.Xfixed * betas
        if N_PS > 0; eta[(M.y_N+1):end, :] .+= PS.Xfixed * betas; end
        summarized_effects[:fixed_effects] = summarize_array(betas'; alpha=alpha)
    end
    if haskey(M, :log_offset)
        eta[1:M.y_N, :] .+= M.log_offset
    end

    # --- 3. Modular Manifold Reconstruction Loop ---
    for spec in M.manifolds
        m_obj = spec.manifold_obj
        m_domain = spec.domain
        var_name = string(spec.var)
        
        latent_name_str = if m_domain == :svc
            "beta_svc_" * var_name
        elseif m_domain == :smooth
            "beta_basis_" * var_name
        else
            "latent_" * string(m_domain) * "_" * var_name
        end

        if latent_name_str in p_names
            n_units = if m_domain == :mixed
                spec.params.n_cat
            elseif m_domain in [:spatial, :svc, :hyperbolic]
                M.s_N
            elseif m_domain in [:temporal, :dynamics]
                M.t_N
            elseif m_domain == :seasonal
                M.u_N
            elseif m_domain == :smooth
                size(M.basis_matrices[spec.var], 2)
            elseif m_domain == :eigen
                spec.params.n_dims
            else 1 end

            latent_samples = get_params_vector(chain, latent_name_str, n_units)'
            summarized_effects[Symbol(latent_name_str)] = summarize_array(latent_samples; alpha=alpha)

            if m_domain in [:spatial, :temporal, :seasonal, :mixed, :hyperbolic]
                indices = if m_domain == :mixed
                    spec.params.indices
                else
                    (m_domain == :spatial || m_domain == :hyperbolic) ? M.s_idx : ((m_domain == :temporal) ? M.t_idx : M.u_idx)
                end
                eta[1:M.y_N, :] .+= latent_samples[indices, :]
                if N_PS > 0 && m_domain != :mixed
                    ps_indices = (m_domain == :spatial || m_domain == :hyperbolic) ? PS.s_idx : ((m_domain == :temporal) ? PS.t_idx : PS.u_idx)
                    eta[(M.y_N+1):end, :] .+= latent_samples[ps_indices, :]
                end
            elseif m_domain == :smooth
                B_mat_train = M.basis_matrices[spec.var]
                eta[1:M.y_N, :] .+= B_mat_train * latent_samples
                if N_PS > 0
                    vars_list = Symbol.(split(string(spec.var), "_"))
                    n_vars = length(vars_list)
                    nbins = size(B_mat_train, 2)
                    params = spec.params
                    manifold_type_str = lowercase(string(typeof(m_obj)))
                    
                    local B_mat_ps
                    ps_data = PS.data
                    if n_vars == 1
                        v_vec = ps_data[!, vars_list[1]]
                        B_mat_ps = bstm_smooth_basis_1D(manifold_type_str, v_vec, nbins, get(params, :degree, 3); params...)
                    elseif n_vars > 1
                        c_mat = Matrix{Float64}(ps_data[!, vars_list])
                        # The basis function factory dispatches based on the number of columns
                        if n_vars == 2; B_mat_ps = bstm_smooth_basis_2D(manifold_type_str, c_mat, nbins; params...);
                        elseif n_vars == 3; B_mat_ps = bstm_smooth_basis_3D(manifold_type_str, c_mat, nbins; params...);
                        elseif n_vars == 4; B_mat_ps = bstm_smooth_basis_4D(manifold_type_str, c_mat, nbins; params...);
                        end
                    end
                    eta[(M.y_N+1):end, :] .+= B_mat_ps * latent_samples
                end
            elseif m_domain == :svc
                x_col_idx = findfirst(==(spec.var), Symbol.(names(M.Xfixed, 2)))
                if !isnothing(x_col_idx)
                    x_vals_train = M.Xfixed[:, x_col_idx]
                    eta[1:M.y_N, :] .+= latent_samples[M.s_idx, :] .* x_vals_train
                    if N_PS > 0
                        x_vals_ps = PS.Xfixed[:, x_col_idx]
                        eta[(M.y_N+1):end, :] .+= latent_samples[PS.s_idx, :] .* x_vals_ps
                    end
                end
            elseif m_domain == :eigen
                data_mat_train = spec.params.data
                eta[1:M.y_N, :] .+= data_mat_train * latent_samples
                if N_PS > 0
                    data_mat_ps = Matrix(PS.data[!, spec.var])
                    eta[(M.y_N+1):end, :] .+= data_mat_ps * latent_samples
                end
            end
        end
    end

    # --- 4. Nested Supervisors & Dynamics ---
    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (z_key, z_meta) in M.nested_manifolds
            rho_key = Symbol("rho_nested_", z_key)
            if string(rho_key) in p_names
                rho_samples = get_params_vector(chain, string(rho_key), 1)
                lat_z_s_key = Symbol("lat_nested_spatial_", z_key)
                if string(lat_z_s_key) in p_names
                    lat_z_s_samples = get_params_vector(chain, string(lat_z_s_key), z_meta.s_N)'
                    for i in 1:M.y_N
                        eta[i, :] .+= rho_samples' .* lat_z_s_samples[z_meta.s_idx[i], :]
                    end
                end
            end
        end
    end
    if get(M, :use_dynamics, false)
        if "sig_dyn" in p_names && "m_states" in p_names
            sig_dyn_samples = get_params_vector(chain, "sig_dyn", 1)
            m_states_samples = get_params_vector(chain, "m_states", M.t_N)'
            for i in 1:M.y_N
                eta[i, :] .+= m_states_samples[M.t_idx[i], :] .* sig_dyn_samples'
            end
        end
    end

    # --- 5. Link-Scale Realization & Prediction ---
    p_den = zeros(Float64, N_tot, N_samples)
    p_noi = zeros(Float64, N_tot, N_samples)
    log_lik = zeros(Float64, N_samples, M.y_N)
    
    y_sigma_samples = _extract_volatility(chain, p_names, N_tot, N_samples, nothing, M)

    for j in 1:N_samples
        eta_j = eta[:, j]
        phi_val = "lik_phi" in p_names ? chain[:lik_phi].data[j] : 0.0
        r_val = "lik_r" in p_names ? chain[:lik_r].data[j] : 1.0

        p_den[:, j] .= _apply_link_and_lik(family_str, eta_j, get(M, :use_zi, false), phi_val, r_val)

        _, noisy_j, ll_j = _process_ll_and_predictions(
            fam_obj, reshape(eta_j, N_tot, 1), chain, M, N_tot, 1, reshape(y_sigma_samples[:, j], N_tot, 1)
        )
        p_noi[:, j] .= noisy_j[:, 1]
        log_lik[j, :] .= ll_j[1, :]
    end

    # --- 6. Final Summarization ---
    summarized_effects[:predictions_denoised] = summarize_array(p_den[1:M.y_N, :]; alpha=alpha)
    summarized_effects[:predictions_noisy] = summarize_array(p_noi[1:M.y_N, :]; alpha=alpha)
    
    if N_PS > 0
        summarized_effects[:ps_predictions_denoised] = summarize_array(p_den[(M.y_N+1):end, :]; alpha=alpha)
        summarized_effects[:ps_predictions_noisy] = summarize_array(p_noi[(M.y_N+1):end, :]; alpha=alpha)
        summarized_effects[:post_strat_weights] = _calculate_ps_weights(p_den, M, PS, N_PS, N_samples)
    end

    summarized_effects[:waic] = _compute_waic(log_lik)
    summarized_effects[:log_lik_matrix] = log_lik
    summarized_effects[:family] = family_str
    summarized_effects[:arch] = arch

    return summarized_effects
end


@model function bstm_modular_multivariate(M, ::Type{T}=Float64) where {T}
    # # 1. Architectural Scope Discovery
    outcomes_N = M.outcomes_N
    y_N = M.y_N
    family = M.model_family
    noise = get(M, :noise, 1e-4)
    use_zi = get(M, :use_zi, false)

    # # 2. Global Likelihood Hyperparameters
    lik_r = one(T)
    lik_phi = zero(T)
    extra_p = ones(T, outcomes_N)

    if family == "negbin"; lik_r ~ NamedDist(Exponential(1.0), :lik_r); end
    if use_zi == true; lik_phi ~ NamedDist(Beta(1, 1), :lik_phi); end
    if family in ["gamma", "beta", "student_t"]; extra_p ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :extra_params); end

    # # 3. Multivariate Coupling & Latent Scales
    L_corr ~ NamedDist(LKJCholesky(outcomes_N, 1.0, T), :L_corr)

    # # 4. Observation Volatility
    y_sigma = Matrix{T}(undef, y_N, outcomes_N)
    if family in ["gaussian", "lognormal"]
        y_sigma_val ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :y_sigma)
        for k in 1:outcomes_N; y_sigma[:, k] .= y_sigma_val[k]; end
    else
        for k in 1:outcomes_N; y_sigma[:, k] .= one(T); end
    end

    # # 5. Base Predictor
    Xfixed_beta ~ NamedDist(MvNormal(zeros(M.Xfixed_N * outcomes_N), 5.0 * I), :Xfixed_beta)
    eta = M.Xfixed * reshape(Xfixed_beta, M.Xfixed_N, outcomes_N)

    # # 6. Modular Manifold Realization
    latent_innovations = zeros(T, y_N, outcomes_N)
    for spec in M.manifolds
        m_obj = spec.manifold_obj
        m_domain = spec.domain
        var_name = string(spec.var)

        for k in 1:outcomes_N
            sigma_name = Symbol("sigma_", m_domain, "_", var_name, "_", k)
            latent_name = Symbol("latent_", m_domain, "_", var_name, "_", k)
            sigma ~ NamedDist(m_obj.sigma_prior, sigma_name)

            if m_domain == :spatial
                template = M.s_Q_template.matrix
                n_units = M.s_N
                indices = M.s_idx
            elseif m_domain == :temporal
                template = M.t_Q_template.matrix
                n_units = M.t_N
                indices = M.t_idx
            elseif m_domain == :seasonal
                template = M.u_Q_template.matrix
                n_units = M.u_N
                indices = M.u_idx
            end

            Q = recompose_precision(Symbol(typeof(m_obj)), template, sigma; noise=noise)
            latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), latent_name)
            for i in 1:y_N
                latent_innovations[i, k] += latent[indices[i]]
            end
        end
    end

    eta .+= (latent_innovations * L_corr.L)

    # # 7. Pointwise Likelihood
    for k in 1:outcomes_N
        ok_idx = M.y_ok[k]
        for i in ok_idx
            d_lik = bstm_Likelihood(
                family, [T(M.y_obs[i, k])]; sigma_y=[y_sigma[i, k]],
                phi_zi=lik_phi, r_nb=lik_r, extra_params=extra_p[k]
            )
            Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i, k])
        end
    end
end

function _reconstruct(arch::MultivariateArchitecture, modelname::String, chain, M, PS, alpha)
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
                        # Re-compute basis for PS grid - logic similar to univariate _reconstruct
                    end
                end
            end
        end
    end

    # Apply LKJ coupling
    if "L_corr" in p_names
        L_corr_samples = get_params_matrix_sizestructured(chain, "L_corr", (outcomes_N, outcomes_N))
        for j in 1:N_samples
            eta[:, :, j] .+= latent_innov_samples[:, :, j] * L_corr_samples[:,:,j]
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
    return summarized_effects
end


@model function bstm_modular_multifidelity(M, ::Type{T}=Float64) where {T}
    fidelities = M.fidelities
    K_levels = length(fidelities)
    family = M.model_family
    noise = get(M, :noise, 1e-4)

    rho_mf ~ NamedDist(Normal(1.0, 0.5), :rho_mf)

    eta_innovations = Vector{Any}(undef, K_levels)
    y_sigma_tensors = Vector{Any}(undef, K_levels)
    s_latent_fields = Vector{Any}(undef, K_levels)
    t_latent_fields = Vector{Any}(undef, K_levels)

    for k in 1:K_levels
        fk = fidelities[k]
        N_k = fk.y_N
        field_k = zeros(T, N_k)

        # Manifolds for this fidelity level
        for spec in fk.manifolds
            m_obj = spec.manifold_obj
            m_domain = spec.domain
            var_name = string(spec.var)
            sigma_name = Symbol("sigma_", m_domain, "_", var_name, "_", k)
            latent_name = Symbol("latent_", m_domain, "_", var_name, "_", k)
            sigma ~ NamedDist(m_obj.sigma_prior, sigma_name)

            if m_domain == :spatial
                template = fk.s_Q_template.matrix
                n_units = fk.s_N
                indices = fk.s_idx
                Q = recompose_precision(:icar, template, sigma; noise=noise)
                latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), latent_name)
                s_latent_fields[k] = latent
                for i in 1:N_k; field_k[i] += latent[indices[i]]; end
            elseif m_domain == :temporal
                template = fk.t_Q_template.matrix
                n_units = fk.t_N
                indices = fk.t_idx
                Q = recompose_precision(:ar1, template, sigma; noise=noise)
                latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), latent_name)
                t_latent_fields[k] = latent
                for i in 1:N_k; field_k[i] += latent[indices[i]]; end
            end
        end
        eta_innovations[k] = field_k

        # Likelihood scale
        sig_val_k ~ NamedDist(Exponential(1.0), Symbol("y_sigma_", k))
        y_sigma_tensors[k] = fill(sig_val_k, N_k)
    end

    # Coupled Predictor Assembly
    s_source = s_latent_fields[1]
    t_source = t_latent_fields[1]

    for k in 1:K_levels
        fk = fidelities[k]
        N_k = fk.y_N
        mu_k = Vector{T}(eta_innovations[k])

        if fk.Xfixed_N > 0
            Xf_beta ~ NamedDist(MvNormal(zeros(fk.Xfixed_N), 5.0 * I), Symbol("Xfixed_beta_", k))
            mu_k .+= fk.Xfixed * Xf_beta
        end

        if k > 1
            for i in 1:N_k
                source_proj = s_source[fk.s_idx[i]] + t_source[fk.t_idx[i]]
                mu_k[i] += rho_mf * source_proj
            end
        end

        # Pointwise Likelihood
        ok_idx = fk.y_ok[1]
        for i in ok_idx
            d_lik = bstm_Likelihood(family, [T(fk.y_obs[i])]; sigma_y=[y_sigma_tensors[k][i]])
            Turing.@addlogprob! logpdf(d_lik, mu_k[i])
        end
    end
end

function _reconstruct(arch::MultifidelityArchitecture, modelname::String, chain, M, PS, alpha)
    N_samples = size(chain, 1)
    fidelities = M.fidelities
    K_levels = length(fidelities)
    p_names = string.(FlexiChains.parameters(chain))
    family_str = get(M, :model_family, "gaussian")

    summarized_effects = Dict{Symbol, Any}()
    
    # Extract innovations for each fidelity level
    eta_innovations_samples = [zeros(f.y_N, N_samples) for f in fidelities]
    s_latent_samples = [zeros(f.s_N, N_samples) for f in fidelities]
    t_latent_samples = [zeros(f.t_N, N_samples) for f in fidelities]

    for k in 1:K_levels
        fk = fidelities[k]
        for spec in fk.manifolds
            m_domain = spec.domain
            var_name = string(spec.var)
            latent_name = Symbol("latent_", m_domain, "_", var_name, "_", k)
            if string(latent_name) in p_names
                if m_domain == :spatial
                    s_latent_samples[k] = get_params_vector(chain, string(latent_name), fk.s_N)'
                    eta_innovations_samples[k] .+= s_latent_samples[k][fk.s_idx, :]
                elseif m_domain == :temporal
                    t_latent_samples[k] = get_params_vector(chain, string(latent_name), fk.t_N)'
                    eta_innovations_samples[k] .+= t_latent_samples[k][fk.t_idx, :]
                end
            end
        end
    end

    # Extract coupling parameter
    rho_mf_samples = get_params_vector(chain, "rho_mf", 1)'

    # Assemble final predictor
    eta_final = Vector{Any}(undef, K_levels)
    s_source_samples = s_latent_samples[1]
    t_source_samples = t_latent_samples[1]

    for k in 1:K_levels
        fk = fidelities[k]
        mu_k = copy(eta_innovations_samples[k])
        if k > 1
            source_proj = s_source_samples[fk.s_idx, :] .+ t_source_samples[fk.t_idx, :]
            mu_k .+= rho_mf_samples .* source_proj
        end
        eta_final[k] = mu_k
    end

    # Summarize predictions
    predictions_denoised = [summarize_array(eta; alpha=alpha) for eta in eta_final]
    summarized_effects[:predictions_denoised] = predictions_denoised
    summarized_effects[:family] = family_str
    summarized_effects[:arch] = arch

    return summarized_effects
end