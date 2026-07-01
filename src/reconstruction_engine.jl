#=
File: reconstruction_engine_v3.jl
Version: 3.0.0
Timestamp: 2026-07-01 10:03:04

Description:
This file contains the complete and corrected versions of all functions related
to the `bstm` post-sampling reconstruction pipeline. It resolves critical bugs
in the latent field discovery process and consolidates all necessary helper
functions into a single, maintainable module.

Changes in this version:
- Added a specialized `extract_manifold` method for `AR1` models to correctly
  reconstruct the latent temporal field from its innovations, resolving a
  `parameter not discovered` error.
- Updated the generic `extract_manifold` method to exclude `AR1`, which is now
  handled by its own implementation.
=#


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

function extract_manifold(m_obj::Union{ICAR, Besag, RW1, RW2, Leroux, SAR, Cyclic}, chain, M, n_samples, outcomes_N, p_names, spec)
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

function extract_manifold(m_obj::AR1, chain, M, n_samples, outcomes_N, p_names, spec)
    # v1.0.0 (2026-07-01)
    # Purpose: Extracts posterior samples for AR1 manifold by reconstructing the latent field.
    structured_fields = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        var_name = string(spec.var)
        m_domain = spec.domain

        sigma_name = outcomes_N > 1 ? "sigma_$(m_domain)_$(var_name)_$(k)" : "sigma_$(m_domain)_$(var_name)"
        rho_name = outcomes_N > 1 ? "rho_$(m_domain)_$(var_name)_$(k)" : "rho_$(m_domain)_$(var_name)"
        innov_name = outcomes_N > 1 ? "innovations_$(m_domain)_$(var_name)_$(k)" : "innovations_$(m_domain)_$(var_name)"

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        rho_samples = get_params_vector(chain, rho_name, 1)
        innov_samples = get_params_vector(chain, innov_name, M.t_N)

        t_raw_samples = zeros(Float64, M.t_N, n_samples)
        noise = get(M, :noise, 1e-4)
        for j in 1:n_samples
            t_rho = rho_samples[j]
            t_innov = innov_samples[j, :]
            t_raw_samples[1, j] = t_innov[1] / sqrt(1.0 - t_rho^2 + noise)
            for i in 2:M.t_N
                t_raw_samples[i, j] = t_rho * t_raw_samples[i-1, j] + t_innov[i]
            end
        end

        effect = t_raw_samples .* sigma_samples'
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
    # v2.0.0 (2026-07-01)
    # Purpose: Extracts posterior samples for basis function manifolds.
    # Change: Returns both the final effect and the raw coefficients.
    var_sym = spec.var
    B_mat = M.basis_matrices[var_sym]
    n_basis_cols = size(B_mat, 2)

    structured_fields = Vector{Matrix{Float64}}()
    coefficient_fields = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        beta_name = "beta_basis_$(var_sym)"
        if outcomes_N > 1; beta_name *= "_$(k)"; end

        coeffs = get_params_vector(chain, beta_name, n_basis_cols)
        push!(coefficient_fields, coeffs') # store as [n_basis, n_samples]

        effect = B_mat * coeffs' # Result is [n_obs, n_samples]
        push!(structured_fields, effect)
    end

    return (structured=structured_fields, noisy=structured_fields, coefficients=coefficient_fields)
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
    # BSTM Manifold Discovery Engine v2.0.0
    # Timestamp: 2026-07-01 09:58:17
    # Synopsis: The main internal engine for discovering and extracting all latent manifold
    #           realizations from a fitted MCMC chain. It iterates through the model's manifold
    #           registry and uses multiple dispatch to call the correct extraction method for each component.
    # Rationale for v2.0.0:
    #     - Fully implemented the manifold discovery loop, which was previously a stub.
    #     - Added logic to handle extraction of space-time interactions and SVC effects.
    #     - Differentiated between storing basis coefficients and accumulated basis effects.

    # --- 1. Initialize Manifold Registries ---
    s_eff_struct = [zeros(Float64, M.s_N, n_samples) for _ in 1:outcomes_N]
    s_eff_noisy  = [zeros(Float64, M.s_N, n_samples) for _ in 1:outcomes_N]
    t_eff = [zeros(Float64, M.t_N, n_samples) for _ in 1:outcomes_N]
    u_eff = zeros(Float64, M.u_N, n_samples)
    basis_eff_accum = zeros(Float64, M.y_N, n_samples)
    basis_coeffs = Dict{Symbol, Matrix{Float64}}()
    st_eff_maps = [zeros(Float64, M.s_N, M.t_N, n_samples) for _ in 1:outcomes_N]
    dynamics_eff = [zeros(Float64, M.t_N, n_samples) for _ in 1:outcomes_N]
    eigen_eff = zeros(Float64, M.y_N, n_samples)
    nested_eff = zeros(Float64, M.y_N, n_samples)

    z_latent_field = nothing
    w_latent_field = nothing

    svc_vars = get(M, :svc_covariates, Symbol[])
    svc_slopes = !isempty(svc_vars) ? [zeros(Float64, M.s_N, length(svc_vars), n_samples) for _ in 1:outcomes_N] : nothing
    mixed_eff_coeffs = !isempty(get(M, :mixed_terms, [])) ? [zeros(Float64, term.n_cat, n_samples) for term in M.mixed_terms] : nothing
    sv_surface = [ones(Float64, M.y_N, n_samples) for _ in 1:outcomes_N]

    xf_betas = nothing
    if M.Xfixed_N > 0 && ("Xfixed_beta" in p_names || any(occursin.(Ref("Xfixed_beta["), p_names)))
        xf_betas = get_params_vector(chain, "Xfixed_beta", M.Xfixed_N * outcomes_N)'
    end

    # --- 2. Iterate and Dispatch Manifold Extraction ---
    if haskey(M, :manifolds) && !isempty(M.manifolds)
        for spec in M.manifolds
            m_obj = spec.manifold_obj
            m_domain = spec.domain

            if m_obj isa NoneManifold; continue; end

            extracted = extract_manifold(m_obj, chain, M, n_samples, outcomes_N, p_names, spec)

            if m_domain == :spatial
                for k in 1:outcomes_N; s_eff_struct[k] .+= extracted.structured[k]; s_eff_noisy[k] .+= extracted.noisy[k]; end
            elseif m_domain == :temporal
                for k in 1:outcomes_N; t_eff[k] .+= extracted.structured[k]; end
            elseif m_domain == :seasonal
                u_eff .+= extracted.structured[1]
            elseif m_domain == :smooth
                basis_eff_accum .+= extracted.structured[1]
                if hasproperty(extracted, :coefficients)
                    basis_coeffs[spec.var] = extracted.coefficients[1]
                end
            elseif m_domain == :dynamics
                for k in 1:outcomes_N; dynamics_eff[k] .+= extracted.structured[k]; end
            elseif m_domain == :eigen
                eigen_eff .+= extracted.structured[1]
            elseif m_domain == :mixed
                term_idx = findfirst(t -> t.name == spec.var, M.mixed_terms)
                if !isnothing(term_idx); mixed_eff_coeffs[term_idx] = extracted.structured[1]; end
            end
        end
    end

    # --- 3. Extract Standalone Components (Interactions, SVC, etc.) ---
    if get(M, :model_st, "none") != "none"
        for k in 1:outcomes_N
            sigma_name = outcomes_N > 1 ? "st_sigma_k[$(k)]" : "st_sigma"
            raw_name = outcomes_N > 1 ? "st_raw_k[$(k)]" : "st_raw"
            st_sigma_samples = get_params_vector(chain, sigma_name, 1)
            st_raw_samples = get_params_vector(chain, raw_name, M.s_N * M.t_N)
            for j in 1:n_samples
                st_map = reshape(st_raw_samples[j, :], M.s_N, M.t_N) .* st_sigma_samples[j]
                st_eff_maps[k][:, :, j] = st_map
            end
        end
    end

    if !isnothing(svc_slopes)
        for k in 1:outcomes_N
            for (i_svc, svc_var) in enumerate(svc_vars)
                svc_latent_samples = get_params_vector(chain, "beta_svc_$(svc_var)_$(k)", M.s_N)
                svc_slopes[k][:, i_svc, :] = svc_latent_samples'
            end
        end
    end

    if get(M, :model_arch, "univariate") == "multifidelity"
        nested_fields = _extract_nested_fields(chain, M, n_samples, outcomes_N, p_names, nothing)
        z_latent_field = nested_fields.z_latent
        w_latent_field = nested_fields.w_latent
    end

    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (z_key, z_meta) in M.nested_manifolds
            rho_samples = get_params_vector(chain, "rho_nested_$(z_key)", 1)
            if haskey(z_meta, :model_space) && z_meta.model_space != "none"
                lat_samples = get_params_vector(chain, "lat_nested_spatial_$(z_key)", z_meta.s_N)'
                if length(z_meta.s_idx) == M.y_N
                    spatial_eff_sub = lat_samples[z_meta.s_idx, :]
                    nested_eff .+= spatial_eff_sub .* rho_samples'
                end
            end
            if haskey(z_meta, :Xfixed)
                beta_samples = get_params_vector(chain, "beta_nested_fixed_$(z_key)", size(z_meta.Xfixed, 2))
                fixed_eff_sub = z_meta.Xfixed * beta_samples'
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
        basis_eff_accum = basis_eff_accum,
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
    # v1.0.2 (2026-07-01)
    # Purpose: Assembles the final linear predictor `eta` from all recovered latent fields.
    # Change: Removed redundant line of code.
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

function _reconstruct(arch::UnivariateArchitecture, modelname::String, chain, M, PS, alpha)
    # BSTM Internal Utility v2.0.0
    # Timestamp: 2026-06-30 11:30:00
    # Synopsis: The internal reconstruction engine for univariate models. It discovers all latent
    #           fields from the MCMC chain, assembles the linear predictor, and generates
    #           predictions, summaries, and diagnostic metrics.
    # Rationale for v2.0.0:
    #     - Standardized output to include `:spatial_denoised` and `:spatial_noisy`.
    #     - Re-integrated summarization for `:mixed_effects` and `:nested_contributions`.

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
    if !isnothing(registry.basis_coeffs) && !isempty(registry.basis_coeffs)
        summarized_effects[:smooth_effects] = Dict{Symbol, Any}()
        for (var_sym, coeffs_matrix) in registry.basis_coeffs
            B_mat = M.basis_matrices[var_sym]
            effect_matrix = B_mat * coeffs_matrix'
            summarized_effects[:smooth_effects][var_sym] = summarize_array(effect_matrix; alpha=alpha)
        end
    end
    if !isnothing(registry.Xfixed_betas)
        summarized_effects[:fixed_effects] = summarize_array(registry.Xfixed_betas'; alpha=alpha)
    end

    if !isnothing(registry.mixed_eff_coeffs) && !isempty(registry.mixed_eff_coeffs)
        summarized_effects[:mixed_effects] = Dict{Symbol, Any}()
        for (i, term) in enumerate(M.mixed_terms)
            summarized_effects[:mixed_effects][term.name] = summarize_array(registry.mixed_eff_coeffs[i]; alpha=alpha)
        end
    end

    if !isnothing(registry.nested_eff) && !all(iszero, registry.nested_eff)
        summarized_effects[:nested_contributions] = summarize_array(registry.nested_eff; alpha=alpha)
    end

    # 4. Generate Predictions and Compute Log-Likelihood
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

function _reconstruct(arch::MultivariateArchitecture, modelname::String, chain, M, PS, alpha)
    # BSTM Internal Utility v2.0.0
    # Timestamp: 2026-06-30 11:30:00
    # Synopsis: The internal reconstruction engine for multivariate models.
    # Rationale for v2.0.0:
    #     - Refactored to use the standardized `_discover_manifold_realizations` and
    #       `_modular_eta_assembly` helpers for consistency and robustness.
    #     - Ensures correct summarization of all latent fields, including spatial effects,
    #       mixed effects, and nested contributions for each outcome.

    N_samples = size(chain, 1)
    outcomes_N = Int(M.outcomes_N)
    p_names = string.(FlexiChains.parameters(chain))
    N_PS = isnothing(PS) ? 0 : size(PS.Xfixed, 1)
    N_tot = M.y_N + N_PS
    family_str = get(M, :model_family, "gaussian")
    fam_obj = get_model_family(family_str)

    # 1. Parameter and Latent Field Discovery
    registry = _discover_manifold_realizations(chain, M, N_samples, outcomes_N, p_names)

    # 2. Linear Predictor Assembly
    eta_samples = _modular_eta_assembly(N_tot, registry, M, PS)

    # Apply LKJ coupling if present
    if "L_corr" in p_names
        L_corr_samples = get_params_matrix_sizestructured(chain, "L_corr", (outcomes_N, outcomes_N))
        for j in 1:N_samples
            eta_samples[:, :, j] = eta_samples[:, :, j] * L_corr_samples[:,:,j]'
        end
    end

    # 3. Summarize Primary Latent Effects
    summarized_effects = Dict{Symbol, Any}()

    if !isnothing(registry.s_eff_struct) && !isempty(registry.s_eff_struct)
        s_denoised_summaries = []
        for k in 1:outcomes_N
            s_denoised_samples = registry.s_eff_struct[k][M.s_idx, :]
            push!(s_denoised_summaries, summarize_array(s_denoised_samples; alpha=alpha))
        end
        summarized_effects[:spatial_denoised] = s_denoised_summaries
    end
    if !isnothing(registry.s_eff_noisy) && !isempty(registry.s_eff_noisy)
        s_noisy_summaries = []
        for k in 1:outcomes_N
            s_noisy_samples = registry.s_eff_noisy[k][M.s_idx, :]
            push!(s_noisy_summaries, summarize_array(s_noisy_samples; alpha=alpha))
        end
        summarized_effects[:spatial_noisy] = s_noisy_summaries
    end

    # Predictions
    p_den = zeros(Float64, N_tot, outcomes_N, N_samples)
    y_sigma_samples = zeros(N_tot, outcomes_N, N_samples)
    if "y_sigma" in p_names
        sig_samps = get_params_vector(chain, "y_sigma", outcomes_N)
        for k in 1:outcomes_N
            y_sigma_samples[:, k, :] .= sig_samps[:, k]'
        end
    end

    for j in 1:N_samples
        for k in 1:outcomes_N
            p_den[:, k, j] .= _apply_link_and_lik(family_str, eta_samples[:, k, j], false)
        end
    end

    summarized_effects[:predictions_denoised] = summarize_array(p_den[1:M.y_N, :, :]; alpha=alpha)
    if N_PS > 0
        summarized_effects[:ps_predictions_denoised] = summarize_array(p_den[(M.y_N+1):end, :, :]; alpha=alpha)
    end

    if !isnothing(registry.basis_coeffs) && !isempty(registry.basis_coeffs)
        summarized_effects[:smooth_effects] = Dict{Symbol, Any}()
        # Note: This assumes smooths are shared across outcomes in multivariate models.
        # A more complex implementation would handle outcome-specific smooth coefficients.
        for (var_sym, coeffs_matrix) in registry.basis_coeffs
            effect_matrix = M.basis_matrices[var_sym] * coeffs_matrix'
            summarized_effects[:smooth_effects][var_sym] = summarize_array(effect_matrix; alpha=alpha)
        end
    end

    if !isnothing(registry.mixed_eff_coeffs) && !isempty(registry.mixed_eff_coeffs)
        summarized_effects[:mixed_effects] = Dict{Symbol, Any}()
        for (i, term) in enumerate(M.mixed_terms)
            summarized_effects[:mixed_effects][term.name] = summarize_array(registry.mixed_eff_coeffs[i]; alpha=alpha)
        end
    end

    if !isnothing(registry.nested_eff) && !all(iszero, registry.nested_eff)
        summarized_effects[:nested_contributions] = summarize_array(registry.nested_eff; alpha=alpha)
    end

    summarized_effects[:family] = family_str
    summarized_effects[:arch] = arch
    return NamedTuple(summarized_effects)
end

function _reconstruct(arch::MultifidelityArchitecture, modelname::String, chain, M, PS, alpha)
    # BSTM Internal Utility v2.0.0
    # Timestamp: 2026-06-30 11:30:00
    # Synopsis: The internal reconstruction engine for multifidelity models.
    # Rationale for v2.0.0:
    #     - Refactored to use the standardized `_discover_manifold_realizations` and
    #       `_modular_eta_assembly` helpers for consistency and robustness.
    #     - Ensures correct summarization of all latent fields, including spatial effects
    #       and multifidelity-specific fields (z_latent, w_latent).

    n_samples = size(chain, 1)
    p_names = string.(FlexiChains.parameters(chain))
    N_PS = isnothing(PS) ? 0 : size(PS.Xfixed, 1)
    N_tot = M.y_N + N_PS
    family_str = get(M, :model_family, "gaussian")
    fam_obj = get_model_family(family_str)

    # 1. Parameter and Latent Field Discovery
    registry = _discover_manifold_realizations(chain, M, n_samples, 1, p_names)

    # 2. Linear Predictor Assembly
    eta_samples = _modular_eta_assembly(N_tot, registry, M, PS)

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

    # 4. Predictions and WAIC
    eta_samples_2d = reshape(eta_samples, N_tot, n_samples)
    p_denoised, p_noisy, log_lik = _process_ll_and_predictions(
        fam_obj, eta_samples_2d, chain, M, N_tot, n_samples, registry.sv_surface
    )

    summarized_effects[:predictions_denoised] = summarize_array(p_denoised; alpha=alpha)
    summarized_effects[:predictions_noisy] = summarize_array(p_noisy; alpha=alpha)

    # Summarize multifidelity fields
    if !isnothing(registry.z_latent)
        summarized_effects[:z_latent_summary] = summarize_array(registry.z_latent; alpha=alpha)
    end
    if !isnothing(registry.w_latent)
        summarized_effects[:w_latent_summary] = summarize_array(registry.w_latent; alpha=alpha)
    end

    if !isnothing(registry.basis_coeffs) && !isempty(registry.basis_coeffs)
        summarized_effects[:smooth_effects] = Dict{Symbol, Any}()
        for (var_sym, coeffs_matrix) in registry.basis_coeffs
            B_mat = M.basis_matrices[var_sym]
            effect_matrix = B_mat * coeffs_matrix'
            summarized_effects[:smooth_effects][var_sym] = summarize_array(effect_matrix; alpha=alpha)
        end
    end

    summarized_effects[:waic] = _compute_waic(log_lik)

    if !isnothing(registry.mixed_eff_coeffs) && !isempty(registry.mixed_eff_coeffs)
        summarized_effects[:mixed_effects] = Dict{Symbol, Any}()
        for (i, term) in enumerate(M.mixed_terms)
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