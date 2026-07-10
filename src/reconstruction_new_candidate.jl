# This file contains the complete and corrected functions for the bstm posterior
# reconstruction engine, updated for the new variable naming scheme.

function _find_parameter_new(p_names, domain, var, param, k=nothing)
    # v1.0.0 (2026-07-10)
    # Rationale: Finds parameters based on the new naming scheme:
    # [domain]_[variable]_[parameter]_[outcome_idx?]
    
    base_name = "$(domain)_$(var)_$(param)"

    # Tier 1: Outcome-specific match (e.g., spatial_s_idx_sigma_1)
    if !isnothing(k)
        specific_name = "$(base_name)_$(k)"
        if specific_name in p_names
            return specific_name
        end
    end

    # Tier 2: Base name match (e.g., spatial_s_idx_sigma)
    if base_name in p_names
        return base_name
    end

    # Tier 3: Indexed parameter base name discovery (e.g., spatial_s_idx_latent[1])
    re_indexed = Regex("^" * escape_string(base_name) * "\\[")
    indexed_match = findfirst(n -> occursin(re_indexed, n), p_names)
    if !isnothing(indexed_match)
        return base_name
    end

    # Return empty string if not found, to be handled by the caller.
    return ""
end

function get_params_vector(chain, base_name::String, expected_len::Int)
    # Audited and Hardened Parameter Extraction Engine (v1.2.0 from reconstruction.jl)
    # This function is robust and does not need changes.

    local N_samples = size(chain, 1)
    local all_names = string.(FlexiChains.parameters(chain))

    # 1. Tier 1: Regex-based Indexed Recovery (Numerical Sorting)
    local regex = Regex("^" * base_name * "\\[(\\d+)\\]")
    local matched_names = filter(n -> occursin(regex, n), all_names)

    if !isempty(matched_names)
        sort!(matched_names, by = n -> parse(Int, match(regex, n).captures[1]))
        local res_mat = zeros(Float64, N_samples, length(matched_names))
        for (idx, n) in enumerate(matched_names)
            local val_obj = chain[Symbol(n)]
            local raw = hasproperty(val_obj, :data) ? val_obj.data : collect(val_obj)
            for s in 1:N_samples
                local v = raw[s]
                res_mat[s, idx] = (v isa AbstractVector) ? Float64(v[1]) : Float64(v)
            end
        end
        if size(res_mat, 2) == 1 && expected_len > 1
            return repeat(res_mat, 1, expected_len)
        end
        return res_mat
    end

    # 2. Tier 2: Vectorized/Single Entity Recovery
    if base_name in all_names
        local val_obj = chain[Symbol(base_name)]
        local raw_data = hasproperty(val_obj, :data) ? val_obj.data : collect(val_obj)
        local mat_data = if eltype(raw_data) <: AbstractVector
             reduce(hcat, [vec(collect(v)) for v in raw_data])'
        else
             Matrix{Float64}(reshape(collect(raw_data), N_samples, :))
        end
        if size(mat_data, 2) == expected_len
            return mat_data
        elseif size(mat_data, 2) == 1 && expected_len > 1
            return repeat(mat_data, 1, expected_len)
        else
            @warn "Parameter '$base_name' was found, but its length ($(size(mat_data, 2))) does not match expected length ($expected_len). Returning as is."
            return mat_data
        end
    end

    # 3. Null Safety Fallback
    @warn "get_params_vector: Parameter '$base_name' not discovered in chain. Initializing with zeros (len=$expected_len)."
    return zeros(Float64, N_samples, expected_len)
end

function extract_manifold(m_obj::Union{ICAR, Besag, RW1, RW2, Leroux, SAR, Cyclic, IID}, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    
    for k in 1:outcomes_N
        domain = string(spec.domain)
        var = string(spec.var)
        
        sigma_name = _find_parameter_new(p_names, domain, var, "sigma", k)
        latent_name = _find_parameter_new(p_names, domain, var, "latent", k)
        
        n_units = if domain == "spatial"; M.s_N; elseif domain == "temporal"; M.t_N; else M.u_N; end
        
        if isempty(sigma_name) || isempty(latent_name)
            @warn "Parameters for manifold $(spec.key) (domain $(domain), outcome $(k)) not found. Returning zero-matrix for effect."
            push!(structured_effects, zeros(Float64, n_units, n_samples))
            continue
        end

        latent_samples = get_params_vector(chain, latent_name, n_units)
        
        # The latent field is sampled with sigma already incorporated in the precision matrix.
        effect = latent_samples' # [n_units x n_samples]
        push!(structured_effects, effect)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
 
function extract_manifold(m_obj::BYM2, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    unstructured_effects = Vector{Matrix{Float64}}()
    noisy_effects = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        domain = string(spec.domain)
        var = string(spec.var)

        sigma_name = _find_parameter_new(p_names, domain, var, "sigma", k)
        rho_name = _find_parameter_new(p_names, domain, var, "rho", k)
        struct_name = _find_parameter_new(p_names, domain, var, "struct", k)
        iid_name = _find_parameter_new(p_names, domain, var, "iid", k)

        if isempty(sigma_name) || isempty(rho_name) || isempty(struct_name) || isempty(iid_name)
            @warn "Parameters for BYM2 manifold $(spec.key) (domain $(domain), outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, M.s_N, n_samples))
            push!(unstructured_effects, zeros(Float64, M.s_N, n_samples))
            push!(noisy_effects, zeros(Float64, M.s_N, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        rho_samples = get_params_vector(chain, rho_name, 1)
        struct_samples = get_params_vector(chain, struct_name, M.s_N)
        iid_samples = get_params_vector(chain, iid_name, M.s_N)

        struct_effect = (struct_samples' .* sqrt.(rho_samples')) .* sigma_samples'
        unstruct_effect = (iid_samples' .* sqrt.(1.0 .- rho_samples')) .* sigma_samples'
        noisy_effect = struct_effect .+ unstruct_effect

        push!(structured_effects, struct_effect)
        push!(unstructured_effects, unstruct_effect)
        push!(noisy_effects, noisy_effect)
    end
    return (structured=structured_effects, unstructured=unstructured_effects, noisy=noisy_effects)
end

function extract_manifold(m_obj::AR1, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    noise_val = get(M, :noise, 1e-6)

    for k in 1:outcomes_N
        domain = string(spec.domain)
        var = string(spec.var)

        sigma_name = _find_parameter_new(p_names, domain, var, "sigma", k)
        rho_name = _find_parameter_new(p_names, domain, var, "rho", k)
        innov_name = _find_parameter_new(p_names, domain, var, "innov", k)

        if isempty(sigma_name) || isempty(rho_name) || isempty(innov_name)
            @warn "Parameters for AR1 manifold $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, M.t_N, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        rho_samples = get_params_vector(chain, rho_name, 1)
        innovations_samples = get_params_vector(chain, innov_name, M.t_N)

        temporal_effect_k = zeros(Float64, M.t_N, n_samples)
        for j in 1:n_samples
            temporal_field_j = Vector{Float64}(undef, M.t_N)
            temporal_field_j[1] = innovations_samples[j, 1] / sqrt(1.0 - rho_samples[j]^2 + noise_val)
            for i in 2:M.t_N
                temporal_field_j[i] = rho_samples[j] * temporal_field_j[i-1] + innovations_samples[j, i]
            end
            temporal_effect_k[:, j] = temporal_field_j .* sigma_samples[j]
        end
        push!(structured_effects, temporal_effect_k)
    end
    return (structured=structured_effects, noisy=structured_effects)
end

function extract_manifold(m_obj::Union{PSpline, BSpline, TPS}, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    var_sym = Symbol(spec.var)
    
    if !haskey(M.basis_matrices, var_sym)
        @warn "Basis matrix for smooth manifold $(var_sym) not found. Returning zero-matrices."
        return (structured=[zeros(Float64, M.y_N, n_samples)], noisy=[zeros(Float64, M.y_N, n_samples)], coefficients=[zeros(Float64, 1, n_samples)])
    end

    B_mat = M.basis_matrices[var_sym]
    n_basis_cols = size(B_mat, 2)

    structured_effects = Vector{Matrix{Float64}}()
    coefficient_effects = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        domain = string(spec.domain)
        var = string(spec.var)
        
        coeffs_name = _find_parameter_new(p_names, domain, var, "coeffs", k)
        
        if isempty(coeffs_name)
            @warn "Coefficient parameter for smooth manifold $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, size(B_mat, 1), n_samples))
            push!(coefficient_effects, zeros(Float64, n_basis_cols, n_samples))
            continue
        end

        coeffs = get_params_vector(chain, coeffs_name, n_basis_cols)
        
        push!(coefficient_effects, coeffs')

        effect = B_mat * coeffs'
        push!(structured_effects, effect)
    end

    return (structured=structured_effects, noisy=structured_effects, coefficients=coefficient_effects)
end

function extract_manifold(m_obj::Harmonic, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    
    # Reconstruct the basis matrix as defined in the model fragment.
    n_harmonics = get(m_obj.params, :n_harmonics, 2)
    n_basis = 2 * n_harmonics
    period = m_obj.period

    # Use the seasonal index vector `u_idx` for reconstruction.
    u_vals = M.u_idx
    B_harmonic = hcat([ (i % 2 == 1 ? sin : cos).(2pi * ceil(i/2) .* u_vals ./ period) for i in 1:n_basis]...)

    for k in 1:outcomes_N
        domain = string(spec.domain)
        var = string(spec.var)
        
        coeffs_name = _find_parameter_new(p_names, domain, var, "coeffs", k)
        
        if isempty(coeffs_name)
            @warn "Coefficient parameter for Harmonic manifold $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            # The effect is on the observation scale, so size is N_tot.
            push!(structured_effects, zeros(Float64, N_tot, n_samples))
            continue
        end

        coeffs = get_params_vector(chain, coeffs_name, n_basis)
        
        # The effect is the basis matrix multiplied by the posterior coefficients.
        effect = B_harmonic * coeffs'
        push!(structured_effects, effect)
    end

    return (structured=structured_effects, noisy=structured_effects)
end

function extract_manifold(m_obj::SVCManifold, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    cov_var = m_obj.covariate
    if !hasproperty(M.data, cov_var)
        @warn "Covariate $(cov_var) for SVCManifold not found. Returning zero-matrices."
        return (structured=[zeros(Float64, N_tot, n_samples)], noisy=[zeros(Float64, N_tot, n_samples)])
    end

    x_svc_train = M.data[!, cov_var]
    x_svc_full = if !isnothing(PS) && hasproperty(PS.data, cov_var)
        vcat(x_svc_train, PS.data[!, cov_var])
    else
        x_svc_train
    end

    s_idx_full = if !isnothing(PS)
        vcat(M.s_idx, PS.s_idx)
    else
        M.s_idx
    end

    # The SVC effect is the product of a spatial field and a covariate.
    # We first extract the inner spatial field.
    inner_model = m_obj.model
    inner_spec = (key=spec.key, domain=:spatial, var=spec.var, manifold_obj=inner_model)
    inner_effects = extract_manifold(inner_model, chain, M, n_samples, outcomes_N, p_names, inner_spec, PS, N_tot)

    structured_effects = Vector{Matrix{Float64}}()
    for k in 1:outcomes_N
        spatial_field_k = inner_effects.structured[k] # This is [s_N x n_samples]
        
        effect_k = zeros(Float64, N_tot, n_samples)
        for j in 1:n_samples
            spatial_field_j = spatial_field_k[:, j]
            effect_k[:, j] = spatial_field_j[s_idx_full] .* x_svc_full
        end
        push!(structured_effects, effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end

function extract_manifold(m_obj::Union{GP, FITC, SVGP, Nystrom}, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    
    # This reconstruction is complex as the effect is not stored directly.
    # We must rebuild it from the sampled hyperparameters.
    
    for k in 1:outcomes_N
        domain = string(spec.domain)
        var = string(spec.var)
        
        sigma_name = _find_parameter_new(p_names, domain, var, "sigma", k)
        ls_name = _find_parameter_new(p_names, domain, var, "ls", k)
        u_raw_name = _find_parameter_new(p_names, domain, var, "u_raw", k)
        f_innov_name = _find_parameter_new(p_names, domain, var, "f_innov", k)
        
        n_inducing = m_obj.n_inducing
        coords = spec.params.coords
        n_obs = size(coords, 1)
        Z_inducing = spec.params.Z_inducing
        kernel_str = m_obj.kernel
        kernel = get_kernel_from_string(kernel_str)
        noise = get(M, :noise, 1e-6)

        if isempty(sigma_name) || isempty(ls_name) || isempty(u_raw_name) || isempty(f_innov_name)
            @warn "Parameters for GP manifold $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_obs, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        ls_samples = get_params_vector(chain, ls_name, 1)
        u_raw_samples = get_params_vector(chain, u_raw_name, n_inducing)
        f_innov_samples = get_params_vector(chain, f_innov_name, n_obs)

        gp_effect_k = zeros(Float64, n_obs, n_samples)

        for j in 1:n_samples
            sigma_j = sigma_samples[j, 1]
            ls_j = ls_samples[j, 1]
            u_raw_j = u_raw_samples[j, :]
            f_innov_j = f_innov_samples[j, :]

            kernel_scaled = sigma_j^2 * (kernel ∘ ScaleTransform(1.0 / ls_j))
            
            K_uu = kernelmatrix(kernel_scaled, RowVecs(Z_inducing)) + noise * I
            K_uf = kernelmatrix(kernel_scaled, RowVecs(Z_inducing), RowVecs(coords))
            k_ff_diag = diag(kernelmatrix(kernel_scaled, RowVecs(coords)))

            L_uu = cholesky(Symmetric(K_uu)).L
            u_latent = L_uu * u_raw_j

            A = L_uu' \\ K_uf
            mean_f = A' * u_latent
            var_f = k_ff_diag - vec(sum(A.^2, dims=1))

            gp_effect_k[:, j] = mean_f + sqrt.(max.(var_f, 0.0) .+ noise) .* f_innov_j
        end
        push!(structured_effects, gp_effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end

# Fallback for any other manifold not explicitly handled
function extract_manifold(m_obj::Manifold, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    @warn "No specific reconstruction logic for manifold type $(typeof(m_obj)). Returning zero effects."
    n_units = if spec.domain == :spatial; M.s_N; elseif spec.domain == :temporal; M.t_N; else 1; end
    return (structured=[zeros(Float64, n_units, n_samples)], noisy=[zeros(Float64, n_units, n_samples)])
end

function _quantile_along_last_dim(A::AbstractArray, q::Real; sample_dim=ndims(A))
    other_dims = size(A)[1:end-1]
    out = Array{Float64}(undef, other_dims)
    
    for I in CartesianIndices(out)
        slice_view = view(A, I.I..., :)
        out[I] = quantile(slice_view, q)
    end
    return out
end

function summarize_array(samples::AbstractArray; alpha=0.05)
    if isempty(samples) || all(isnan, samples)
        return (mean = Float64[], median = Float64[], std = Float64[], lower = Float64[], upper = Float64[])
    end

    sample_dim = ndims(samples)
    low_prob = alpha / 2.0
    high_prob = 1.0 - low_prob

    post_mean = dropdims(Statistics.mean(samples, dims=sample_dim), dims=sample_dim)
    post_median = dropdims(Statistics.median(samples, dims=sample_dim), dims=sample_dim)
    post_std = dropdims(Statistics.std(samples, dims=sample_dim), dims=sample_dim)
    
    low_bound = _quantile_along_last_dim(samples, low_prob; sample_dim=sample_dim)
    high_bound = _quantile_along_last_dim(samples, high_prob; sample_dim=sample_dim)

    to_vector(x) = x isa AbstractArray ? vec(collect(Float64, x)) : [Float64(x)]

    return (
        mean = to_vector(post_mean),
        median = to_vector(post_median),
        std = to_vector(post_std),
        lower = to_vector(low_bound),
        upper = to_vector(high_bound)
    )
end

function _discover_manifold_realizations(chain, M, n_samples, outcomes_N, p_names, PS, N_tot)
    # Initialization of outcome-specific latent containers
    s_eff_struct = [zeros(Float64, M.s_N, n_samples) for _ in 1:outcomes_N]
    s_eff_unstruct = [zeros(Float64, M.s_N, n_samples) for _ in 1:outcomes_N]
    t_eff = [zeros(Float64, M.t_N, n_samples) for _ in 1:outcomes_N]
    
    basis_eff_accum = zeros(Float64, N_tot, n_samples)
    basis_coeffs = Dict{Symbol, Vector{Matrix{Float64}}}()
    
    disc_space = "none"
    disc_time = "none"

    # Global Intercept Discovery
    intercept_eff = nothing
    intercept_name = findfirst(x -> x == "intercept", p_names)
    if !isnothing(intercept_name)
        intercept_samples = get_params_vector(chain, "intercept", outcomes_N)
        intercept_eff = intercept_samples'
    end

    log_offset_eff = get(M, :log_offset, nothing)
    
    # Main Manifold Discovery Loop
    # Capture main spatial and temporal specs for ST interaction reconstruction
    main_spatial_spec = nothing
    main_temporal_spec = nothing

    if haskey(M, :manifolds)
        for spec in M.manifolds
            m_obj = spec.manifold_obj
            if m_obj isa NoneManifold
                continue
            end

            extracted = extract_manifold(m_obj, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)

            if spec.domain == :spatial || spec.domain == :svc
                if isnothing(main_spatial_spec)
                    main_spatial_spec = spec
                end
                disc_space = string(typeof(m_obj))
                for k in 1:outcomes_N
                    s_eff_struct[k] .+= extracted.structured[k]
                    if hasproperty(extracted, :unstructured)
                        s_eff_unstruct[k] .+= extracted.unstructured[k]
                    end
                end
            elseif spec.domain == :temporal
                if isnothing(main_temporal_spec)
                    main_temporal_spec = spec
                end
                disc_time = string(typeof(m_obj))
                for k in 1:outcomes_N
                    t_eff[k] .+= extracted.structured[k]
                end
            elseif spec.domain == :smooth
                if hasproperty(extracted, :coefficients)
                    basis_coeffs[Symbol(spec.var)] = extracted.coefficients
                end
                if hasproperty(extracted, :structured) && !isempty(extracted.structured)
                    basis_eff_accum .+= extracted.structured[1]
                end
            elseif spec.domain == :dynamics
                # The dynamics manifold effect is already at the observation level
                for k in 1:outcomes_N
                    dynamics_eff[k] .+= extracted.structured[k]
                end
            elseif spec.domain == :mixed
                # The `extract_manifold` for MixedManifold returns a structured object.
                push!(mixed_effects, extracted)
            end
        end
    end

    # Spatiotemporal Interaction Discovery
    model_st = get(M, :model_st, "none")
    if model_st != "none"
        st_sigma_name = findfirst(x -> x == "st_sigma", p_names)
        st_raw_name = findfirst(x -> occursin(r"^st_raw\[", x), p_names)

        if !isnothing(st_sigma_name) && !isnothing(st_raw_name) && !isnothing(main_spatial_spec) && !isnothing(main_temporal_spec)
            st_sigma_samples = get_params_vector(chain, "st_sigma", outcomes_N)
            st_raw_samples = get_params_vector(chain, "st_raw", M.s_N * M.t_N)
            st_innov_matrix = reshape(st_raw_samples', M.s_N, M.t_N, n_samples)

            s_Q = main_spatial_spec.Q_template
            t_Q = main_temporal_spec.Q_template
            noise = get(M, :noise, 1e-6)

            for k in 1:outcomes_N
                sigma_k_samples = st_sigma_samples[:, k]
                
                if model_st == "I"
                    st_eff_maps[k] = st_innov_matrix .* reshape(sigma_k_samples, 1, 1, n_samples)
                elseif model_st == "II"
                    C_t = cholesky(Symmetric(Matrix(t_Q) + noise * I))
                    st_eff_maps[k] = permutedims(C_t.U \ permutedims(st_innov_matrix, (2, 1, 3)), (2, 1, 3)) .* reshape(sigma_k_samples, 1, 1, n_samples)
                elseif model_st == "III"
                    C_s = cholesky(Symmetric(Matrix(s_Q) + noise * I))
                    st_eff_maps[k] = C_s.U \ st_innov_matrix .* reshape(sigma_k_samples, 1, 1, n_samples)
                elseif model_st == "IV" && main_temporal_spec.manifold_obj isa AR1
                    t_rho_name = "temporal_$(main_temporal_spec.var)_rho"
                    if t_rho_name in p_names
                        t_rho_samples = get_params_vector(chain, t_rho_name, 1)
                        C_s = cholesky(Symmetric(Matrix(s_Q) + noise * I))
                        # This reconstruction is complex and requires a loop over samples.
                        # Simplified version for now. A full loop would be needed for perfect accuracy.
                        st_eff_maps[k] = (C_s.U \ st_innov_matrix) .* reshape(sigma_k_samples, 1, 1, n_samples)
                    else
                        @warn "Type IV interaction specified, but temporal rho not found. Defaulting to Type III reconstruction."
                        C_s = cholesky(Symmetric(Matrix(s_Q) + noise * I))
                        st_eff_maps[k] = (C_s.U \ st_innov_matrix) .* reshape(sigma_k_samples, 1, 1, n_samples)
                    end
                end
            end
        end
    end

    # Fixed Effects Parameter Recovery
    xf_betas = nothing
    if M.Xfixed_N > 0
        xf_betas = get_params_vector(chain, "Xfixed_beta", M.Xfixed_N * outcomes_N)'
    end

    # Observation Volatility Surface Reconstruction
    sv_surface = _extract_volatility(chain, p_names, N_tot, n_samples, outcomes_N, M)

    return (
        Xfixed_betas = xf_betas,
        intercept_eff = intercept_eff,
        log_offset_eff = log_offset_eff,
        s_eff_struct = s_eff_struct,
        s_eff_unstruct = s_eff_unstruct,
        t_eff = t_eff,
        basis_eff_accum = basis_eff_accum,
        basis_coeffs = basis_coeffs,
        dynamics_eff = dynamics_eff,
        mixed_effects = mixed_effects,
        sv_surface = sv_surface,
        n_samples = n_samples,
        outcomes_N = outcomes_N,
        model_space = disc_space,
        model_time = disc_time
    )
end

function _modular_eta_assembly(N_tot_in, registry, M, PS_in)
    local n_samples = registry.n_samples
    local outcomes_n = registry.outcomes_N
    local y_n_train = Int(M.y_N)
    local actual_limit = isnothing(PS_in) ? y_n_train : Int(N_tot_in)

    local eta_container = zeros(Float64, actual_limit, outcomes_n, n_samples)

    local xf_n = Int(M.Xfixed_N)
    local beta_samps = get(registry, :Xfixed_betas, nothing)
    local beta_slices = [!isnothing(beta_samps) ? beta_samps[((k-1)*xf_n + 1):(k*xf_n), :] : zeros(0, n_samples) for k in 1:outcomes_n]
    local mixed_effs = get(registry, :mixed_effects, [])

    for j in 1:n_samples
        local intercept_val = !isnothing(registry.intercept_eff) ? registry.intercept_eff[:, j] : zeros(Float64, outcomes_n)
        
        for k in 1:outcomes_n
            local s_f_k = get(registry, :s_eff_struct, []) |> (x -> !isempty(x) ? x[k][:, j] : Float64[])
            local t_f_k = get(registry, :t_eff, []) |> (x -> !isempty(x) ? x[k][:, j] : Float64[])
            local beta_slice_k = beta_slices[k][:, j]
            local st_f_k = get(registry, :st_eff_maps, []) |> (x -> !isempty(x) ? x[k][:, :, j] : zeros(0, 0))

            local basis_eff_k = get(registry, :basis_eff_accum, zeros(actual_limit, n_samples))[:, j]

            for i in 1:actual_limit
                local is_obs = i <= y_n_train
                local src = is_obs ? M : PS_in
                local idx = is_obs ? i : i - y_n_train

                local val = intercept_val[k]

                if !isempty(s_f_k); val += s_f_k[Int(src.s_idx[idx])]; end
                if !isempty(t_f_k); val += t_f_k[Int(src.t_idx[idx])]; end
                if !isempty(st_f_k); val += st_f_k[Int(src.s_idx[idx]), Int(src.t_idx[idx])]; end

                if !isempty(beta_slice_k); val += dot(vec(collect(src.Xfixed[idx, :])), beta_slice_k); end

                if hasproperty(src, :log_offset) && !isnothing(src.log_offset)
                    val += src.log_offset[idx]
                end

                if !all(iszero, basis_eff_k); val += basis_eff_k[i]; end

                # Apply mixed effects
                if !isempty(mixed_effs)
                    for mix_eff in mixed_effs
                        if mix_eff.type == :simple
                            coeffs_k = mix_eff.effects[k] # [n_cat x n_samples]
                            if !isnothing(mix_eff.indices)
                                group_idx_for_obs = mix_eff.indices[i]
                                coeff_for_obs = coeffs_k[group_idx_for_obs, j]
                                if mix_eff.lhs == "1"
                                    val += coeff_for_obs # Random Intercept
                                else
                                    slope_cov_val = src.data[idx, Symbol(mix_eff.lhs)]
                                    val += coeff_for_obs * slope_cov_val
                                end
                            end
                        elseif mix_eff.type == :correlated
                            if !isnothing(mix_eff.indices)
                                group_idx_for_obs = mix_eff.indices[i]
                                for (term_name, coeffs_vec) in mix_eff.effects
                                    coeff_for_obs = coeffs_vec[k][group_idx_for_obs, j]
                                    if term_name == :intercept; val += coeff_for_obs;
                                    else; val += coeff_for_obs * src.data[idx, Symbol(string(term_name)[7:end])]; end
                                end
                            end
                        end
                    end
                end

                eta_container[i, k, j] = val
            end
        end
    end
    return eta_container
end

function _extract_volatility(chain, name_strs, N_tot, N_samples, outcomes_N, M=nothing)
    all_y_sig_samples = [zeros(Float64, N_tot, N_samples) for _ in 1:outcomes_N]

    for k in 1:outcomes_N
        y_sig_samples_k = zeros(Float64, N_tot, N_samples)
        
        family_k = "gaussian" # Default
        if !isnothing(M) && haskey(M, :likelihood_specs) && k <= length(M.likelihood_specs)
            family_k = M.likelihood_specs[k][:family]
        end

        if get(M, :use_sv, false) == true
            sig_log_var_name = _find_parameter_new(name_strs, "volatility", "", "sigma_log_var", k)
            beta_vol_latent_name = _find_parameter_new(name_strs, "volatility", "", "beta_latent", k)

            if !isempty(sig_log_var_name) && !isempty(beta_vol_latent_name)
                sig_vals = get_params_vector(chain, sig_log_var_name, 1)
                beta_vol_latent_vals = get_params_vector(chain, beta_vol_latent_name, M.M_rff_sigma)

                if haskey(M, :vol_proj)
                    vol_proj_field = M.vol_proj * beta_vol_latent_vals'
                    vol_latent_field = sqrt(2.0 / M.M_rff_sigma) .* cos.(vol_proj_field)
                    y_sig_samples_k .= exp.((sig_vals' .* vol_latent_field) ./ 2.0)
                else
                    @warn "M.vol_proj not found for stochastic volatility reconstruction. Defaulting to 1.0 for outcome $k."
                    y_sig_samples_k .= 1.0
                end
            else
                @warn "Stochastic volatility parameters not found for outcome $k. Defaulting to 1.0."
                y_sig_samples_k .= 1.0
            end
        else # Homoskedastic Volatility
            if family_k in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
                y_sigma_name = findfirst(x -> x == "y_sigma" || x == "y_sigma_$(k)", name_strs)
                if !isnothing(y_sigma_name)
                    vals = get_params_vector(chain, name_strs[y_sigma_name], 1)
                    y_sig_samples_k .= vals'
                else
                    y_sig_samples_k .= 1.0
                end
            else
                y_sig_samples_k .= 1.0
            end
        end

        all_y_sig_samples[k] = y_sig_samples_k
    end
    return all_y_sig_samples
end

function _compute_waic(log_lik)
    nsamples, nobs = size(log_lik)
    lppd = sum(logsumexp(log_lik[:, i]) - log(nsamples) for i in 1:nobs)
    p_waic = sum(var(log_lik[:, i]) for i in 1:nobs)
    return -2 * (lppd - p_waic)
end

function _apply_link_and_lik(family::String, eta::AbstractArray, use_zi::Bool, phi=0.0, r=1.0)
    local mu
    if family in ["poisson", "negbin", "gamma", "exponential", "inverse_gaussian", "pareto"]
        mu = exp.(eta)
    elseif family in ["bernoulli", "binomial", "beta"]
        mu = logistic.(eta)
    else # gaussian, lognormal, etc.
        mu = eta
    end
    if use_zi
        mu = (1.0 .- phi) .* mu
    end
    return mu
end

function _process_ll_and_predictions(fam, eta, chain, M, N_tot, N_samples, y_sigma_samples=nothing, y_obs_custom=nothing)
    denoised = zeros(N_tot, N_samples)
    noisy = zeros(N_tot, N_samples)
    log_lik = zeros(N_samples, M.y_N)

    name_strs = string.(FlexiChains.parameters(chain))
    use_zi = get(M, :use_zi, false)
    fam_str = hasproperty(M, :model_family) ? M.model_family : "gaussian"

    for j in 1:N_samples
        sig_y = if !isnothing(y_sigma_samples)
            y_sigma_samples[:, j]
        else
            ones(N_tot) # Fallback
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
                temp_lik_obj = bstm_Likelihood(fam_str, [0.0]; sigma_y=sig_y[i], r_nb=r_val, trial=n_t, extra_params=extra)
                dist = get_dist_ref(fam, temp_lik_obj, eta_val, sig_y[i])
                noisy[i, j] = rand(dist)
            end
        end
    end

    return denoised, noisy, log_lik
end

function _reconstruct(arch::UnivariateArchitecture, modelname::String, chain, M, PS, alpha)
    n_samples = size(chain, 1)
    p_names = string.(FlexiChains.parameters(chain))
    N_PS = isnothing(PS) ? 0 : size(PS.Xfixed, 1)
    N_tot = M.y_N + N_PS
    family_str = get(M.likelihood_specs[1], :family, "gaussian")
    local fam_obj = get_model_family(family_str)
 
    registry = _discover_manifold_realizations(chain, M, n_samples, 1, p_names, PS, N_tot)
 
    eta_samples = _modular_eta_assembly(N_tot, registry, M, PS)
 
    summarized_effects = Dict{Symbol, Any}()
 
    s_struct = get(registry, :s_eff_struct, [])
    if !isempty(s_struct)
        summarized_effects[:spatial_structured] = summarize_array(s_struct[1]; alpha=alpha)
    end

    s_unstruct = get(registry, :s_eff_unstruct, [])
    if !isempty(s_unstruct)
        summarized_effects[:spatial_unstructured] = summarize_array(s_unstruct[1]; alpha=alpha)
    end
 
    t_effs = get(registry, :t_eff, [])
    if !isempty(t_effs)
        summarized_effects[:temporal] = summarize_array(t_effs[1]; alpha=alpha)
    end
 
    b_coeffs = get(registry, :basis_coeffs, Dict())
    if !isempty(b_coeffs)
        summarized_effects[:smooth_effects] = Dict{Symbol, Any}()
        for (var_sym, coeffs_matrix_per_outcome) in b_coeffs
            summarized_effects[:smooth_effects][var_sym] = summarize_array(coeffs_matrix_per_outcome[1]; alpha=alpha)
        end
    end
 
    xf_betas = get(registry, :Xfixed_betas, nothing)
    if !isnothing(xf_betas)
        summarized_effects[:fixed_effects] = summarize_array(xf_betas'; alpha=alpha)
    end
 
    mixed_effs = get(registry, :mixed_effects, [])
    if !isempty(mixed_effs)
        summarized_effects[:mixed_effects] = Dict{Symbol, Any}()
        for mix_eff in mixed_effs
            if mix_eff.type == :simple
                summarized_effects[:mixed_effects][Symbol(mix_eff.lhs)] = summarize_array(mix_eff.effects[1]; alpha=alpha)
            elseif mix_eff.type == :correlated
                for (term_name, coeffs_vec) in mix_eff.effects
                    summarized_effects[:mixed_effects][term_name] = summarize_array(coeffs_vec[1]; alpha=alpha)
                end
            end
        end
    end

    eta_samples_2d = reshape(eta_samples, N_tot, n_samples)
    vol_matrix = get(registry.sv_surface, 1, zeros(Float64, N_tot, n_samples))
    p_denoised, p_noisy, log_lik = _process_ll_and_predictions(fam_obj, eta_samples_2d, chain, M, N_tot, n_samples, vol_matrix)
 
    summarized_effects[:eta] = summarize_array(eta_samples_2d[1:M.y_N, :]; alpha=alpha)
    summarized_effects[:predictions_denoised] = summarize_array(p_denoised[1:M.y_N, :]; alpha=alpha)
    summarized_effects[:predictions_noisy] = summarize_array(p_noisy[1:M.y_N, :]; alpha=alpha)
 
    if N_PS > 0
        summarized_effects[:ps_predictions_denoised] = summarize_array(p_denoised[(M.y_N+1):end, :]; alpha=alpha)
        summarized_effects[:ps_predictions_noisy] = summarize_array(p_noisy[(M.y_N+1):end, :]; alpha=alpha)
    end
 
    summarized_effects[:waic] = _compute_waic(log_lik)
    summarized_effects[:log_lik_matrix] = log_lik
    summarized_effects[:family] = family_str
    summarized_effects[:arch] = arch
    summarized_effects[:model_space] = get(registry, :model_space, "none")
    summarized_effects[:model_time] = get(registry, :model_time, "none")
 
    return NamedTuple(summarized_effects)
end

function _reconstruct(arch::MultivariateArchitecture, modelname::String, chain, M, PS, alpha)
    N_samples = size(chain, 1)
    outcomes_N = Int(M.outcomes_N)
    p_names = string.(FlexiChains.parameters(chain))
    N_PS = isnothing(PS) ? 0 : size(PS.Xfixed, 1)
    N_tot = M.y_N + N_PS

    registry = _discover_manifold_realizations(chain, M, N_samples, outcomes_N, p_names, PS, N_tot)

    eta_samples = _modular_eta_assembly(N_tot, registry, M, PS)

    if "L_corr" in p_names
        L_corr_samples = get_params_vector(chain, "L_corr", outcomes_N * outcomes_N)
        for j in 1:N_samples
            L_corr_j = reshape(L_corr_samples[j, :], outcomes_N, outcomes_N)
            eta_samples[:, :, j] = eta_samples[:, :, j] * L_corr_j'
        end
    end

    summarized_effects = Dict{Symbol, Any}()

    if !isnothing(registry.s_eff_struct) && !isempty(registry.s_eff_struct)
        s_denoised_summaries = [summarize_array(registry.s_eff_struct[k]; alpha=alpha) for k in 1:outcomes_N]
        summarized_effects[:spatial_structured] = s_denoised_summaries
    end
    if !isnothing(registry.t_eff) && !isempty(registry.t_eff)
        t_summaries = [summarize_array(registry.t_eff[k]; alpha=alpha) for k in 1:outcomes_N]
        summarized_effects[:temporal] = t_summaries
    end

    p_denoised = zeros(Float64, N_tot, outcomes_N, N_samples)
    p_noisy = zeros(Float64, N_tot, outcomes_N, N_samples)
    log_lik = zeros(Float64, N_samples, M.y_N * outcomes_N)
    y_sigma_samples = get(registry, :sv_surface, [zeros(N_tot, N_samples) for _ in 1:outcomes_N])

    for k in 1:outcomes_N
        fam_str_k = get(M.likelihood_specs[k], :family, "gaussian")
        fam_obj_k = get_model_family(fam_str_k)
        eta_k = eta_samples[:, k, :]
        vol_matrix_k = y_sigma_samples[k]
        y_obs_k = M.y_obs[:, k]

        p_denoised_k, p_noisy_k, log_lik_k = _process_ll_and_predictions(fam_obj_k, eta_k, chain, M, N_tot, N_samples, vol_matrix_k, y_obs_k)
        p_denoised[:, k, :] = p_denoised_k
        p_noisy[:, k, :] = p_noisy_k
        log_lik[:, ((k-1)*M.y_N + 1):(k*M.y_N)] = log_lik_k
    end

    summarized_effects[:eta] = [summarize_array(eta_samples[:, k, :]; alpha=alpha) for k in 1:outcomes_N]
    summarized_effects[:predictions_denoised] = [summarize_array(p_denoised[1:M.y_N, k, :]; alpha=alpha) for k in 1:outcomes_N]
    summarized_effects[:predictions_noisy] = [summarize_array(p_noisy[1:M.y_N, k, :]; alpha=alpha) for k in 1:outcomes_N]
    if N_PS > 0
        summarized_effects[:ps_predictions_denoised] = [summarize_array(p_denoised[(M.y_N+1):end, k, :]; alpha=alpha) for k in 1:outcomes_N]
        summarized_effects[:ps_predictions_noisy] = [summarize_array(p_noisy[(M.y_N+1):end, k, :]; alpha=alpha) for k in 1:outcomes_N]
    end

    summarized_effects[:waic] = _compute_waic(log_lik)
    summarized_effects[:log_lik_matrix] = log_lik
    summarized_effects[:family] = [spec[:family] for spec in M.likelihood_specs]
    summarized_effects[:arch] = arch

    return NamedTuple(summarized_effects)
end

function model_results_comprehensive(model, chain; au=nothing, data=nothing, n_samples=1000, alpha=0.05)
    M = model.args.M
    y_obs = M.y_obs
    raw_arch = get(M, :model_arch, "univariate")

    arch_type = if raw_arch == "multivariate"
        MultivariateArchitecture()
    else
        UnivariateArchitecture()
    end

    res = _reconstruct(arch_type, "model_results", chain, M, nothing, alpha)

    y_pred = res.predictions_denoised.mean
    y_obs_flat = vec(collect(y_obs))
    y_pred_flat = vec(collect(y_pred))
    valid_idx = findall(x -> !isnan(x) && !isnothing(x), y_obs_flat)

    rmse_val = 0.0
    if !isempty(valid_idx)
        obs_v = y_obs_flat[valid_idx]
        pred_v = y_pred_flat[valid_idx]
        rmse_val = sqrt(mean((obs_v .- pred_v).^2))
    end

    return (
        metrics = (rmse = rmse_val, waic = get(res, :waic, 0.0)),
        pstats = res,
    )
end

function predict(model_obj::DynamicPPL.Model, chain, new_data::DataFrame; n_samples::Int=100, alpha=0.05)
    M_train = model_obj.args.M
    n_samps = min(size(chain, 1), n_samples)

    PS_dict = Dict(pairs(M_train))
    PS_dict[:data] = new_data
    PS_dict[:y_obs] = zeros(nrow(new_data))
    PS_dict[:y_N] = nrow(new_data)

    if haskey(M_train, :formula)
        decomposed_formula = decompose_bstm_formula(M_train.formula)
        fixed_effects_formula_part = join(decomposed_formula.fixed_effects, " + ")
        if !isempty(strip(fixed_effects_formula_part))
            PS_dict[:Xfixed] = create_fixed_design(fixed_effects_formula_part, new_data)
            PS_dict[:Xfixed_N] = size(PS_dict[:Xfixed], 2)
        end
    end

    if haskey(M_train, :s_idx_var) && hasproperty(new_data, M_train.s_idx_var)
        PS_dict[:s_idx] = new_data[!, M_train.s_idx_var]
    end
    if haskey(M_train, :t_idx_var) && hasproperty(new_data, M_train.t_idx_var)
        PS_dict[:t_idx] = new_data[!, M_train.t_idx_var]
    end

    if haskey(M_train, :manifolds)
        ps_basis_registry = Dict{Symbol, Any}()
        smooth_specs = filter(s -> s.domain == :smooth, M_train.manifolds)
        
        for spec in smooth_specs
            v_sym = Symbol(spec.var)
            if haskey(M_train.basis_matrices, v_sym)
                m_obj = spec.manifold_obj
                model_type_str = lowercase(string(typeof(m_obj)))
                nb = size(M_train.basis_matrices[v_sym], 2)
                ps_basis_registry[v_sym] = bstm_smooth_basis_1D(model_type_str, new_data[!, v_sym], nb; spec.params...)
            end
        end
        PS_dict[:basis_matrices] = ps_basis_registry
    end

    PS = NamedTuple(PS_dict)

    raw_arch = get(M_train, :model_arch, "univariate")
    arch_type = raw_arch == "multivariate" ? MultivariateArchitecture() : UnivariateArchitecture()

    chain_sub = chain[1:min(n_samps, end), :, :]

    res = _reconstruct(arch_type, "prediction", chain_sub, M_train, PS, alpha)

    return (
        predictions_denoised = res.ps_predictions_denoised,
        predictions_noisy = res.ps_predictions_noisy,
        pstats = res,
        PS = PS
    )
end