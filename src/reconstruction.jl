# This file contains the complete and corrected functions for the bstm posterior
# reconstruction engine. It is designed to be a self-contained replacement for
# the various post-processing utilities previously distributed across other files.
 
function _find_parameter(p_names, base, domain, key, k=nothing)
    # v2.0.0 (2026-07-13)
    # Rationale: This function was failing to find vector-valued parameters named with
    #            Turing's `[i]` syntax. A new tier has been added to explicitly check for
    #            this pattern, ensuring that base names like `s_icar` are correctly
    #            identified even when they appear in the chain as `s_icar[1]`, `s_icar[2]`, etc.
    #            This prevents the parameter discovery from failing and returning zero-matrices.

    # Tier 1: Literal and Exact Match
    if base in p_names
        return base
    end

    # Tier 2: Indexed Parameter Base Name Discovery
    # Rationale: Handles cases where Turing adds `[i]` to vector/matrix parameters.
    # This finds the base name (e.g., "param" from "param[1]").
    re_indexed = Regex(string("^", base, "\\["))
    indexed_match = findfirst(n -> occursin(re_indexed, n), p_names)
    if !isnothing(indexed_match)
        return base
    end

    # Tier 3: Outcome-Specific Suffixing
    if !isnothing(k)
        k_targets = ["$(base)_$(k)", "$(base)_$(key)_$(k)", "$(base)_$(domain)_$(key)_$(k)"]
        for t in k_targets
            if t in p_names
                return t
            end
        end
    end

    # Tier 4: Key-Based and Domain-Based Discovery
    keyed_targets = ["$(base)_$(key)", "$(base)_$(domain)_$(key)", "$(base)_$(domain)"]
    for t in keyed_targets
        if t in p_names
            return t
        end
    end

    # Tier 5: Suffix Cleanup (Turing Internal Naming)
    re_suffix = Regex(string("^(", base, ")(_\\d*)?\$"))
    matches = filter(n -> occursin(re_suffix, n), p_names)
    if !isempty(matches) && length(matches) == 1 # Ensure unique match for direct return.
        return matches[1]
    end

    # Tier 6: Fuzzy Coordinate Resolution
    re_fuzzy = Regex("^$(base)_.*($(key)|$(domain)).*")
    fuzzy_matches = filter(n -> occursin(re_fuzzy, n), p_names)
    if !isempty(fuzzy_matches)
        # Return the shortest match, assuming it's the most direct.
        # Ensure it's a unique shortest match to avoid ambiguity.
        shortest_matches = filter(n -> length(n) == minimum(length.(fuzzy_matches)), fuzzy_matches)
        if length(shortest_matches) == 1; return shortest_matches[1]; end
    end

    # Return the original base name as a last resort. `get_params_vector` has its own fallback.
    return base
end

function get_params_vector(chain, base_name::String, expected_len::Int)
    # Audited and Hardened Parameter Extraction Engine (v1.2.0)
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
            return repeat(res_mat, 1, expected_len) # This creates a matrix of size N_samples x expected_len
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

        # Dimensional Realignment and Broadcasting
        if size(mat_data, 2) == expected_len
            return mat_data
        elseif size(mat_data, 2) == 1 && expected_len > 1
            # Scalar Broadcast fallback to N_samples x expected_len
            return repeat(mat_data, 1, expected_len)
        else
            @warn "Parameter '$base_name' was found, but its length ($(size(mat_data, 2))) does not match expected length ($expected_len). Returning as is, which may cause downstream errors."
            return mat_data
        end
    end

    # 3. Null Safety Fallback
    # If the parameter is missing, return a zero-matrix of size N_samples x expected_len to prevent downstream assembly failure.
    @warn "get_params_vector: Parameter '$base_name' not discovered in chain. Initializing with zeros (len=$expected_len)."
    return zeros(Float64, N_samples, expected_len)
end

function extract_manifold(m_obj::Union{ICAR, Besag, RW1, RW2, Leroux, SAR, Cyclic}, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_fields = Vector{Matrix{Float64}}()
    noisy_fields = Vector{Matrix{Float64}}()
    
    for k in 1:outcomes_N
        key = spec.key
        m_domain = spec.domain
        
        sigma_name = _find_parameter(p_names, "sigma_" * string(key), string(m_domain), string(key), k)
        latent_name = _find_parameter(p_names, "latent_" * string(key), string(m_domain), string(key), k)
        
        n_units = if m_domain == :spatial; M.s_N; elseif m_domain == :temporal; M.t_N; else M.u_N; end
        
        if isempty(sigma_name) || isempty(latent_name)
            @warn "Parameters for manifold $(key) (domain $(m_domain), outcome $(k)) not found. Returning zero-matrix for effect."
            push!(structured_fields, zeros(Float64, n_units, n_samples))
            push!(noisy_fields, zeros(Float64, n_units, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        latent_samples = get_params_vector(chain, latent_name, n_units)
        
        effect = latent_samples' .* sigma_samples'
        push!(structured_fields, effect)
        push!(noisy_fields, effect)
    end
    
    return (structured=structured_fields, noisy=noisy_fields)
end
 
function extract_manifold(m_obj::BYM2, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_fields = Vector{Matrix{Float64}}()
    unstructured_fields = Vector{Matrix{Float64}}()
    noisy_fields = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        key = string(spec.key)
        m_domain = string(spec.domain)
        sigma_name = _find_parameter(p_names, "sigma_" * key, m_domain, key, k)
        rho_name = _find_parameter(p_names, "rho_" * key, m_domain, key, k)
        struct_name = _find_parameter(p_names, "latent_struct_" * key, m_domain, key, k)
        iid_name = _find_parameter(p_names, "latent_iid_" * key, m_domain, key, k)

        if isempty(sigma_name) || isempty(rho_name) || isempty(struct_name) || isempty(iid_name)
            @warn "Parameters for BYM2 manifold $(key) (domain $(m_domain), outcome $(k)) not found. Returning zero-matrix for effect."
            n_units = M.s_N
            push!(structured_fields, zeros(Float64, n_units, n_samples))
            push!(unstructured_fields, zeros(Float64, n_units, n_samples))
            push!(noisy_fields, zeros(Float64, n_units, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        rho_samples = get_params_vector(chain, rho_name, 1)
        struct_samples = get_params_vector(chain, struct_name, M.s_N)
        iid_samples = get_params_vector(chain, iid_name, M.s_N)

        struct_eff = (struct_samples' .* sqrt.(rho_samples')) .* sigma_samples'
        unstruct_eff = (iid_samples' .* sqrt.(1.0 .- rho_samples')) .* sigma_samples'
        noisy_eff = struct_eff .+ unstruct_eff

        push!(structured_fields, struct_eff)
        push!(unstructured_fields, unstruct_eff)
        push!(noisy_fields, noisy_eff)
    end
    return (structured=structured_fields, unstructured=unstructured_fields, noisy=noisy_fields)
end

function extract_manifold(m_obj::AR1, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_fields = Vector{Matrix{Float64}}()
    noisy_fields = Vector{Matrix{Float64}}()

    noise_val = get(M, :noise, 1e-6)

    for k in 1:outcomes_N
        key = string(spec.key)
        m_domain = string(spec.domain)

        sigma_name = _find_parameter(p_names, "sigma_" * key, m_domain, key, k)
        rho_name = _find_parameter(p_names, "rho_" * key, m_domain, key, k)
        innov_name = _find_parameter(p_names, "innov_" * key, m_domain, key, k)

        n_units = M.t_N

        if isempty(sigma_name) || isempty(rho_name) || isempty(innov_name)
            @warn "Parameters for AR1 manifold $(key) (outcome $(k)) not found. Returning zero-matrix for effect."
            push!(structured_fields, zeros(Float64, n_units, n_samples))
            push!(noisy_fields, zeros(Float64, n_units, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        rho_samples = get_params_vector(chain, rho_name, 1)
        innovations_samples = get_params_vector(chain, innov_name, n_units)

        temporal_effect_k = zeros(Float64, n_units, n_samples)
        for j in 1:n_samples
            t_field_j = Vector{Float64}(undef, n_units)
            t_field_j[1] = innovations_samples[j, 1] / sqrt(1.0 - rho_samples[j]^2 + noise_val)
            for i in 2:n_units
                t_field_j[i] = rho_samples[j] * t_field_j[i-1] + innovations_samples[j, i]
            end
            temporal_effect_k[:, j] = t_field_j .* sigma_samples[j]
        end
        push!(structured_fields, temporal_effect_k)
        push!(noisy_fields, temporal_effect_k)
    end
    return (structured=structured_fields, noisy=noisy_fields)
end

function extract_manifold(m_obj::Union{PSpline, BSpline, TPS, RFF, FFT, Wavelet, Moran, Spherical, ExponentialDecay, Barycentric}, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    var_sym = spec.var
    
    if !haskey(M.basis_matrices, var_sym)
        @warn "Basis matrix for smooth manifold $(var_sym) not found in M.basis_matrices. Returning zero-matrices."
        n_obs_or_units = if hasproperty(M, :y_N); M.y_N; else 1; end
        return (structured=[zeros(Float64, n_obs_or_units, n_samples) for _ in 1:outcomes_N],
                noisy=[zeros(Float64, n_obs_or_units, n_samples) for _ in 1:outcomes_N],
                coefficients=[zeros(Float64, 1, n_samples) for _ in 1:outcomes_N])
    end

    B_mat = M.basis_matrices[var_sym]
    n_basis_cols = size(B_mat, 2)

    structured_fields = Vector{Matrix{Float64}}()
    noisy_fields = Vector{Matrix{Float64}}()
    coefficient_fields = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        key = spec.key
        beta_name = outcomes_N > 1 ? Symbol("beta_basis_", key, "_", k) : Symbol("beta_basis_", key)
        
        if !(string(beta_name) in p_names)
            @warn "Coefficient parameter $(beta_name) for smooth manifold $(key) (outcome $(k)) not found. Returning zero-matrix for effect."
            push!(structured_fields, zeros(Float64, size(B_mat, 1), n_samples))
            push!(noisy_fields, zeros(Float64, size(B_mat, 1), n_samples))
            push!(coefficient_fields, zeros(Float64, n_basis_cols, n_samples))
            continue
        end

        coeffs = get_params_vector(chain, string(beta_name), n_basis_cols)
        
        push!(coefficient_fields, coeffs')

        effect = B_mat * coeffs'
        push!(structured_fields, effect)
        push!(noisy_fields, effect)
    end

    return (structured=structured_fields, noisy=noisy_fields, coefficients=coefficient_fields)
end
 
function extract_manifold(m_obj::DynamicsManifold, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_fields = Vector{Matrix{Float64}}()
    noisy_fields = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        key = spec.key
        latent_name = _find_parameter(p_names, "dyn_field_" * string(key), "dynamics", string(key), k)
        
        if isempty(latent_name)
            @warn "Latent field parameter for DynamicsManifold $(key) (outcome $(k)) not found. Returning zero-matrix for effect."
            push!(structured_fields, zeros(Float64, N_tot, n_samples))
            push!(noisy_fields, zeros(Float64, N_tot, n_samples))
            continue
        end

        latent_samples = get_params_vector(chain, latent_name, M.s_N * M.t_N)

        dyn_field_samples = reshape(latent_samples', M.s_N, M.t_N, n_samples)
        
        effect_k = zeros(Float64, N_tot, n_samples)
        for j in 1:n_samples
            for i in 1:N_tot
                is_obs = i <= M.y_N
                src = is_obs ? M : PS
                idx = is_obs ? i : i - M.y_N
                
                s_ptr = Int(src.s_idx[idx])
                t_ptr = Int(src.t_idx[idx])
                
                if s_ptr > M.s_N || t_ptr > M.t_N || s_ptr < 1 || t_ptr < 1
                    @warn "Index out of bounds for DynamicsManifold effect reconstruction: s_ptr=$(s_ptr), t_ptr=$(t_ptr). Setting effect to zero."
                    effect_k[i, j] = 0.0
                else
                    effect_k[i, j] = dyn_field_samples[s_ptr, t_ptr, j]
                end
            end
        end
        push!(structured_fields, effect_k)
        push!(noisy_fields, effect_k)
    end

    return (structured=structured_fields, noisy=noisy_fields)
end

function extract_manifold(m_obj::SVCManifold, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_fields = Vector{Matrix{Float64}}()
    noisy_fields = Vector{Matrix{Float64}}()
    
    cov_var = m_obj.covariate
    if !hasproperty(M.data, cov_var)
        @warn "Covariate $(cov_var) for SVCManifold not found in training data. Returning zero-matrices."
        return (structured=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N],
                noisy=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N])
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

    for k in 1:outcomes_N
        key = spec.key
        beta_name = _find_parameter(p_names, "beta_svc_" * string(key), "svc", string(cov_var), k)
        
        if isempty(beta_name)
            @warn "Coefficient parameter $(beta_name) for SVCManifold $(key) (outcome $(k)) not found. Returning zero-matrix for effect."
            push!(structured_fields, zeros(Float64, N_tot, n_samples))
            push!(noisy_fields, zeros(Float64, N_tot, n_samples))
            continue
        end

        beta_samples = get_params_vector(chain, beta_name, M.s_N)
        
        effect_k = zeros(Float64, N_tot, n_samples)
        for j in 1:n_samples
            beta_j = beta_samples[j, :]
            if any(idx -> idx > length(beta_j) || idx < 1, s_idx_full)
                @warn "Spatial index out of bounds for SVCManifold effect reconstruction. Setting effect to zero."
                effect_k[:, j] .= 0.0
            else
                effect_k[:, j] = beta_j[s_idx_full] .* x_svc_full
            end
        end
        push!(structured_fields, effect_k)
        push!(noisy_fields, effect_k)
    end

    return (structured=structured_fields, noisy=noisy_fields)
end

function extract_manifold(m_obj::MixedManifold, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_fields = Vector{Matrix{Float64}}()
    noisy_fields = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        n_units = spec.params.n_cat
        latent_name = _find_parameter(p_names, "latent_" * string(spec.key), "mixed", string(spec.var), k)
        
        if isempty(latent_name)
            @warn "Latent field parameter for MixedManifold $(spec.key) (outcome $(k)) not found. Returning zero-matrix for effect."
            push!(structured_fields, zeros(Float64, n_units, n_samples))
            push!(noisy_fields, zeros(Float64, n_units, n_samples))
            continue
        end

        latent_samples = get_params_vector(chain, latent_name, n_units)
        push!(structured_fields, latent_samples')
        push!(noisy_fields, latent_samples')
    end

    return (structured=structured_fields, noisy=noisy_fields)
end

function extract_manifold(m_obj::Harmonic, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_fields = Vector{Matrix{Float64}}()
    noisy_fields = Vector{Matrix{Float64}}()
    
    n_units = M.t_N
    basis_coords = 1.0:Float64(n_units)

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
        beta_name = outcomes_N > 1 ? Symbol("beta_basis_", spec.key, "_", k) : Symbol("beta_basis_", spec.key)
        
        if !(string(beta_name) in p_names)
            @warn "Coefficient parameter $(beta_name) for Harmonic manifold $(spec.key) (outcome $(k)) not found. Returning zero-matrix for effect."
            push!(structured_fields, zeros(Float64, n_units, n_samples))
            push!(noisy_fields, zeros(Float64, n_units, n_samples))
            continue
        end

        coeffs = get_params_vector(chain, string(beta_name), n_basis_cols)
        effect = B_mat * coeffs'
        push!(structured_fields, effect)
        push!(noisy_fields, effect)
    end

    return (structured=structured_fields, noisy=noisy_fields)
end

function extract_manifold(m_obj::Eigen, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_fields = Vector{Matrix{Float64}}()
    noisy_fields = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        key = spec.key
        v_name = _find_parameter(p_names, "v_" * string(key), "eigen", string(key), k)
        
        if isempty(v_name)
            @warn "Reflector parameter $(v_name) for Eigen manifold $(key) (outcome $(k)) not found. Returning zero-matrix for effect."
            push!(structured_fields, zeros(Float64, N_tot, n_samples))
            push!(noisy_fields, zeros(Float64, N_tot, n_samples))
            continue
        end

        v_samples = get_params_vector(chain, v_name, length(spec.params.ltri_indices))

        eigen_vars = spec.variables
        
        if !all(hasproperty(M.data, Symbol(v)) for v in eigen_vars)
            @warn "Eigen variables $(eigen_vars) not found in training data. Returning zero-matrices."
            push!(structured_fields, zeros(Float64, N_tot, n_samples))
            push!(noisy_fields, zeros(Float64, N_tot, n_samples))
            continue
        end

        Y_data_train = Matrix(M.data[!, Symbol.(eigen_vars)])
        Y_data = if !isnothing(PS) && hasproperty(PS, :data) && all(hasproperty(PS.data, Symbol(v)) for v in eigen_vars)
            Matrix(PS.data[!, Symbol.(eigen_vars)])
        else
            Y_data_train
        end

        n_vars = length(eigen_vars)
        n_factors = spec.params.n_factors

        eigen_effect = zeros(Float64, N_tot, n_samples)

        for j in 1:n_samples
            v_vec = v_samples[j, :]
            v_mat = zeros(Float64, n_vars, n_factors)
            
            if any(idx -> idx > length(v_mat) || idx < 1, spec.params.ltri_indices)
                @warn "ltri_indices out of bounds for Eigen manifold effect reconstruction. Setting effect to zero."
                eigen_effect[:, j] .= 0.0
                continue
            end
            v_mat[spec.params.ltri_indices] .= v_vec
            
            U = householder_to_eigenvector(v_mat, n_vars, n_factors)
            
            if size(Y_data, 2) != size(U, 1)
                @warn "Dimension mismatch between Y_data ($(size(Y_data, 2))) and U ($(size(U, 1))) for Eigen manifold effect reconstruction. Setting effect to zero."
                eigen_effect[:, j] .= 0.0
                continue
            end

            factors = Y_data * U
            eigen_effect[:, j] = factors[:, 1]
        end
        push!(structured_fields, eigen_effect)
        push!(noisy_fields, eigen_effect)
    end

    return (structured=structured_fields, noisy=noisy_fields)
end

function extract_manifold(m_obj::ComposedManifold, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_fields = Vector{Matrix{Float64}}()
    noisy_fields = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        key = spec.key
        latent_name = _find_parameter(p_names, "latent_" * string(key), "interaction", string(key), k)

        if isempty(latent_name)
            @warn "Latent field parameter for ComposedManifold $(key) (outcome $(k)) not found. Returning zero-matrix for effect."
            push!(structured_fields, zeros(Float64, N_tot, n_samples))
            push!(noisy_fields, zeros(Float64, N_tot, n_samples))
            continue
        end

        latent_samples_flat = get_params_vector(chain, latent_name, 0)

        if size(latent_samples_flat, 2) == 0
            @warn "Latent field for composed manifold '$key' has zero length. Skipping."
            push!(structured_fields, zeros(Float64, N_tot, n_samples))
            push!(noisy_fields, zeros(Float64, N_tot, n_samples))
            continue
        end

        indices = get(spec, :indices, nothing)

        if isnothing(indices)
             @warn "Composed manifold '$key' is missing indices. Cannot reconstruct effect."
             push!(structured_fields, zeros(Float64, N_tot, n_samples))
             push!(noisy_fields, zeros(Float64, N_tot, n_samples))
             continue
        end

        effect_k = zeros(Float64, N_tot, n_samples)

        if any(idx -> idx > size(latent_samples_flat, 2) || idx < 1, indices)
            @warn "Indices for ComposedManifold effect reconstruction are out of bounds. Setting effect to zero."
            push!(structured_fields, zeros(Float64, N_tot, n_samples))
            push!(noisy_fields, zeros(Float64, N_tot, n_samples))
            continue
        end

        if !isnothing(PS)
            @warn "Prediction for ComposedManifold is not fully supported in this reconstruction path. The effect will be zero for the prediction set. The `predict` function should be used for out-of-sample prediction."
            for j in 1:n_samples
                field_j = latent_samples_flat[j, :]
                effect_k[1:M.y_N, j] = field_j[indices]
            end
        else
            for j in 1:n_samples
                field_j = latent_samples_flat[j, :]
                effect_k[:, j] = field_j[indices]
            end
        end
        push!(structured_fields, effect_k)
        push!(noisy_fields, effect_k)
    end

    return (structured=structured_fields, noisy=noisy_fields)
end

function extract_manifold(m_obj::IID, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_fields = Vector{Matrix{Float64}}()
    noisy_fields = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        key = spec.key
        m_domain = spec.domain
        
        sigma_name = _find_parameter(p_names, "sigma_" * string(key), string(m_domain), string(key), k)
        latent_name = _find_parameter(p_names, "latent_" * string(key), string(m_domain), string(key), k)
        
        n_units = if m_domain == :spatial; M.s_N; elseif m_domain == :temporal; M.t_N; else M.u_N; end
        
        if isempty(sigma_name) || isempty(latent_name)
            @warn "Parameters for IID manifold $(key) (domain $(m_domain), outcome $(k)) not found. Returning zero-matrix for effect."
            push!(structured_fields, zeros(Float64, n_units, n_samples))
            push!(noisy_fields, zeros(Float64, n_units, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        latent_samples = get_params_vector(chain, latent_name, n_units)
        
        effect = latent_samples' .* sigma_samples'
        push!(structured_fields, effect)
        push!(noisy_fields, effect)
    end

    return (structured=structured_fields, noisy=noisy_fields)
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


function _extract_simple_manifold_effect(chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot) # Added PS, N_tot for consistency
    structured_fields = Vector{Matrix{Float64}}()
    noisy_fields = Vector{Matrix{Float64}}()
    
    for k in 1:outcomes_N
        key = spec.key
        m_domain = spec.domain

        sigma_name = _find_parameter(p_names, "sigma_" * string(key), string(m_domain), string(key), k)
        latent_name = _find_parameter(p_names, "latent_" * string(key), string(m_domain), string(key), k)

        n_units = if m_domain == :spatial; M.s_N; elseif m_domain == :temporal; M.t_N; else M.u_N; end

        # Handle cases where parameters might not be found, returning zero-matrices
        if isempty(sigma_name) || isempty(latent_name)
            @warn "Parameters for manifold $(key) (domain $(m_domain), outcome $(k)) not found. Returning zero-matrix for effect."
            push!(structured_fields, zeros(Float64, n_units, n_samples))
            push!(noisy_fields, zeros(Float64, n_units, n_samples))
            continue
        end
        sigma_samples = get_params_vector(chain, sigma_name, 1)
        latent_samples = get_params_vector(chain, latent_name, n_units)
    end        
end


function _discover_manifold_realizations(chain, M, n_samples, outcomes_N, p_names, PS, N_tot)
    # Initialization of outcome-specific latent containers
    # These structures aggregate multiple manifold effects per outcome for assembly
    s_eff_struct = [zeros(Float64, M.s_N, n_samples) for _ in 1:outcomes_N]
    s_eff_unstruct = [zeros(Float64, M.s_N, n_samples) for _ in 1:outcomes_N]
    s_eff_noisy  = [zeros(Float64, M.s_N, n_samples) for _ in 1:outcomes_N]
    t_eff = [zeros(Float64, M.t_N, n_samples) for _ in 1:outcomes_N]
    u_eff = zeros(Float64, M.u_N, n_samples)
    
    # Extended registries for complex manifold types
    basis_eff_accum = zeros(Float64, N_tot, n_samples)
    basis_coeffs = Dict{Symbol, Vector{Matrix{Float64}}}()
    st_eff_maps = [zeros(Float64, M.s_N, M.t_N, n_samples) for _ in 1:outcomes_N]
    dynamics_eff = [zeros(Float64, M.y_N, n_samples) for _ in 1:outcomes_N]
    eigen_eff = [zeros(Float64, M.y_N, n_samples) for _ in 1:outcomes_N]
    

    # Registry for component metadata identification used in reporting wrappers
    disc_space = "none"
    disc_time = "none"

    # Global Intercept Discovery
    intercept_eff = nothing
    intercept_name = _find_parameter(p_names, "intercept", "", "")
    if !isempty(intercept_name) && intercept_name in p_names
        intercept_samples = get_params_vector(chain, intercept_name, outcomes_N)
        intercept_eff = intercept_samples'
    end

    log_offset_eff = get(M, :log_offset, nothing)
    
    # Technical Registry for random effects tracking
    mixed_terms_list = get(M, :mixed_terms, [])
    mixed_eff_coeffs = !isempty(mixed_terms_list) ? [zeros(Float64, term.n_cat, n_samples) for term in mixed_terms_list] : nothing
    
    # Main Manifold Discovery Loop
    if haskey(M, :manifolds)
        for spec in M.manifolds
            m_obj = spec.manifold_obj
            if m_obj isa NoneManifold
                continue
            end

            # Extract posterior realizations based on manifold type trait
            extracted = extract_manifold(m_obj, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)

            # Map extracted fields to their respective domains
            if spec.domain == :spatial
                disc_space = string(typeof(m_obj))
                for k in 1:outcomes_N
                    s_eff_struct[k] .+= extracted.structured[k]
                    if hasproperty(extracted, :unstructured)
                        s_eff_unstruct[k] .+= extracted.unstructured[k]
                    end
                    s_eff_noisy[k] .+= extracted.noisy[k]
                end
            elseif spec.domain == :temporal
                disc_time = string(typeof(m_obj))
                for k in 1:outcomes_N
                    t_eff[k] .+= extracted.structured[k]
                end
            elseif spec.domain == :seasonal
                u_eff .+= extracted.structured[1]
            elseif spec.domain == :smooth
                if hasproperty(extracted, :coefficients)
                    basis_coeffs[spec.var] = extracted.coefficients
                end
                if hasproperty(extracted, :structured) && !isempty(extracted.structured)
                    basis_eff_accum .+= extracted.structured[1]
                end
            elseif m_obj isa DynamicsManifold
                for k in 1:outcomes_N
                    dynamics_eff[k] .+= extracted.structured[k]
                end
            elseif m_obj isa Eigen
                for k in 1:outcomes_N
                    eigen_eff[k] .+= extracted.structured[k]
                end
            elseif m_obj isa MixedManifold
                term_idx = findfirst(t -> t.name == spec.var, mixed_terms_list)
                if !isnothing(term_idx)
                    mixed_eff_coeffs[term_idx] = extracted.structured[1]
                end
            elseif spec.domain == :interaction
                # A ComposedManifold's effect is already at the observation level.
                # Accumulate it directly into the basis accumulator.
                basis_eff_accum .+= extracted.structured[1]
            end
        end
    end

    # Nested Hierarchical Supervisor Realization
    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (z_key, z_meta) in M.nested_manifolds
            rho_name = _find_parameter(p_names, "rho_nested", "", string(z_key))
            rho_samples = get_params_vector(chain, rho_name, 1) # [n_samples, 1]

            if haskey(z_meta, :model_space) && z_meta.model_space != "none"
                lat_base = "latent_nested_spatial"
                lat_name = _find_parameter(p_names, lat_base, "", string(z_key))
                lat_samples = get_params_vector(chain, lat_name, z_meta.s_N)' # [s_N, n_samples]

                for j in 1:n_samples
                    for i in 1:M.y_N
                        s_ptr = M.s_idx[i]
                        nested_eff[1][i, j] += rho_samples[j] * lat_samples[s_ptr, j]
                    end
                end
            end
        end
    end

    # Discovery of Space-Time Interaction components
    if get(M, :model_st, "none") != "none"
        st_sigma_name = _find_parameter(p_names, "st_sigma", "", "")
        st_sigma_samples = get_params_vector(chain, st_sigma_name, outcomes_N)
        st_raw_name = _find_parameter(p_names, "st_raw", "", "")
        st_raw_samples = get_params_vector(chain, st_raw_name, M.s_N * M.t_N)

        for k in 1:outcomes_N
            for j in 1:n_samples
                st_eff_maps[k][:, :, j] = reshape(st_raw_samples[j, :], M.s_N, M.t_N) .* st_sigma_samples[j, k]
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

    # Bundle all discovered realizations for the linear predictor assembly engine
    return (
        Xfixed_betas = xf_betas,
        intercept_eff = intercept_eff,
        log_offset_eff = log_offset_eff,
        mixed_eff_coeffs = mixed_eff_coeffs,
        s_eff_struct = s_eff_struct,
        s_eff_unstruct = s_eff_unstruct,
        s_eff_noisy = s_eff_noisy,
        t_eff = t_eff,
        u_eff = u_eff,
        basis_eff_accum = basis_eff_accum,
        basis_coeffs = basis_coeffs,
        st_eff_maps = st_eff_maps,
        dynamics_eff = dynamics_eff,
        eigen_eff = eigen_eff,
        all_nested_effects = all_nested_effects,
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

    # Pre-slice fixed effects betas for efficiency
    local xf_n = Int(M.Xfixed_N)
    local beta_samps = get(registry, :Xfixed_betas, nothing)
    local beta_slices = [!isnothing(beta_samps) ? beta_samps[((k-1)*xf_n + 1):(k*xf_n), :] : zeros(0, n_samples) for k in 1:outcomes_n]

    # Main assembly loop over posterior samples
    for j in 1:n_samples
        local intercept_val = !isnothing(registry.intercept_eff) ? registry.intercept_eff[:, j] : zeros(Float64, outcomes_n)
        
        for k in 1:outcomes_n
            # Extract single-sample slices for all latent fields for the current outcome
            local s_f_k = get(registry, :s_eff_noisy, []) |> (x -> !isempty(x) ? x[k][:, j] : Float64[])
            local t_f_k = get(registry, :t_eff, []) |> (x -> !isempty(x) ? x[k][:, j] : Float64[])
            local u_f_k = get(registry, :u_eff, zeros(M.u_N, n_samples))[:, j]
            local st_f_k = get(registry, :st_eff_maps, []) |> (x -> !isempty(x) ? x[k][:, :, j] : zeros(0, 0))
            local beta_slice_k = beta_slices[k][:, j]

            # Extract effects from specialized manifolds
            local basis_eff_k = get(registry, :basis_eff_accum, zeros(actual_limit, n_samples))[:, j]
            local dynamics_eff_k = get(registry, :dynamics_eff, []) |> (x -> !isempty(x) ? x[k][:, j] : Float64[])
            local eigen_eff_k = get(registry, :eigen_eff, []) |> (x -> !isempty(x) ? x[k][:, j] : Float64[])
            local nested_eff_k = get(registry, :nested_eff, []) |> (x -> !isempty(x) ? x[k][:, j] : Float64[])

            # Loop over all observations (in-sample and out-of-sample)
            for i in 1:actual_limit
                local is_obs = i <= y_n_train
                local src = is_obs ? M : PS_in
                local idx = is_obs ? i : i - y_n_train

                # Initialize with intercept
                local val = intercept_val[k]

                # Add standard spatiotemporal effects
                if !isempty(s_f_k); val += s_f_k[Int(src.s_idx[idx])]; end
                if !isempty(t_f_k); val += t_f_k[Int(src.t_idx[idx])]; end
                if !all(iszero, u_f_k); val += u_f_k[Int(src.u_idx[idx])]; end
                if !isempty(st_f_k); val += st_f_k[Int(src.s_idx[idx]), Int(src.t_idx[idx])]; end

                # Add fixed effects
                if !isempty(beta_slice_k); val += dot(vec(collect(src.Xfixed[idx, :])), beta_slice_k); end

                # Add offset (for both observed and prediction data)
                if hasproperty(src, :log_offset) && !isnothing(src.log_offset)
                    val += src.log_offset[idx]
                end

                # Add smooth covariate effects
                if !all(iszero, basis_eff_k); val += basis_eff_k[i]; end

                # Add mixed effects
                if haskey(M, :mixed_terms) && !isempty(M.mixed_terms)
                    for (term_idx, term) in enumerate(M.mixed_terms)
                        group_idx = term.indices[i]
                        val += registry.mixed_eff_coeffs[term_idx][group_idx, j]
                    end
                end

                # Add specialized manifold effects
                if !isempty(dynamics_eff_k); val += dynamics_eff_k[i]; end
                if !isempty(eigen_eff_k); val += eigen_eff_k[i]; end
                
                # Add nested effects
                if haskey(registry, :all_nested_effects) && !isempty(registry.all_nested_effects)
                    for (_, nested_effect_samples) in registry.all_nested_effects
                        val += nested_effect_samples[i, j]
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
            sig_log_var_name = _find_parameter(name_strs, "sigma_log_var", "sv", "volatility", k)
            beta_vol_latent_name = _find_parameter(name_strs, "beta_vol_latent", "sv", "volatility", k)

            if sig_log_var_name in name_strs && beta_vol_latent_name in name_strs
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
                y_sigma_name = _find_parameter(name_strs, "y_sigma", "observation", "noise", k)
                vals = get_params_vector(chain, y_sigma_name, 1)
                y_sig_samples_k .= vals'
            else
                y_sig_samples_k .= 1.0
            end
        end

        all_y_sig_samples[k] = y_sig_samples_k
    end
    return all_y_sig_samples
end




function _compute_waic(log_lik)
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: Computes the Widely Applicable Information Criterion (WAIC). 
    # Inputs: log_lik - A matrix of pointwise log-likelihoods [N_samples x N_obs].
    # Outputs: The WAIC value.
    nsamples, nobs = size(log_lik)
    lppd = sum(logsumexp(log_lik[:, i]) - log(nsamples) for i in 1:nobs)
    p_waic = sum(var(log_lik[:, i]) for i in 1:nobs)
    return -2 * (lppd - p_waic)
end

function _process_ll_and_predictions(fam, eta, chain, M, N_tot, N_samples, y_sigma_samples=nothing, y_obs_custom=nothing)
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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

 


function _apply_link_and_lik(family::String, eta::AbstractArray, use_zi::Bool, phi=0.0, r=1.0)
    # v1.2.0 (2026-06-29 16:13:05)
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





function _reconstruct(arch::UnivariateArchitecture, modelname::String, chain, M, PS, alpha)
    n_samples = size(chain, 1)
    p_names = string.(FlexiChains.parameters(chain))
    N_PS = isnothing(PS) ? 0 : size(PS.Xfixed, 1)
    N_tot = M.y_N + N_PS
    family_str = get(M.likelihood_specs[1], :family, "gaussian")
    local fam_obj = get_model_family(family_str)
 
    # 1. Parameter and Latent Field Discovery
    registry = _discover_manifold_realizations(chain, M, n_samples, 1, p_names, PS, N_tot)
 
    # 2. Linear Predictor Assembly for Training and Prediction Grids
    eta_samples = _modular_eta_assembly(N_tot, registry, M, PS)
 
    # 3. Summarize Primary Latent Effects
    summarized_effects = Dict{Symbol, Any}()
 
    s_struct = get(registry, :s_eff_struct, [])
    if !isempty(s_struct)
        summarized_effects[:spatial_denoised] = summarize_array(s_struct[1]; alpha=alpha)
    end

    s_unstruct = get(registry, :s_eff_unstruct, [])
    if !isempty(s_unstruct)
        summarized_effects[:spatial_unstructured] = summarize_array(s_unstruct[1]; alpha=alpha)
    end
 
    s_noisy = get(registry, :s_eff_noisy, [])
    if !isempty(s_noisy)
        summarized_effects[:spatial_noisy] = summarize_array(s_noisy[1]; alpha=alpha)
    end
 
    t_effs = get(registry, :t_eff, [])
    if !isempty(t_effs)
        summarized_effects[:temporal] = summarize_array(t_effs[1]; alpha=alpha)
    end
 
    u_eff = get(registry, :u_eff, zeros(0, 0))
    if !all(iszero, u_eff)
        summarized_effects[:seasonal] = summarize_array(u_eff; alpha=alpha)
    end
 
    st_maps = get(registry, :st_eff_maps, [])
    if !isempty(st_maps)
        summarized_effects[:spacetime_interaction] = summarize_array(st_maps[1]; alpha=alpha)
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

    # Summarize nested effects
    if haskey(registry, :all_nested_effects) && !isempty(registry.all_nested_effects)
        summarized_effects[:nested_effects] = Dict{Symbol, Any}()
        for (var_name, effect_samples) in registry.all_nested_effects
            summarized_effects[:nested_effects][var_name] = summarize_array(effect_samples; alpha=alpha)
        end
    end
 
    # 4. Generate Predictions and Compute Log-Likelihood
    eta_samples_2d = reshape(eta_samples, N_tot, n_samples)
    vol_matrix = get(registry.sv_surface, 1, zeros(Float64, N_tot, n_samples))
    p_denoised, p_noisy, log_lik = _process_ll_and_predictions(fam_obj, eta_samples_2d, chain, M, N_tot, n_samples, vol_matrix)
 
    # 5. Summarize Predictions
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
    family_str = get(M, :model_family, "gaussian")
    fam_obj = get_model_family(family_str)

    # 1. Parameter and Latent Field Discovery
    registry = _discover_manifold_realizations(chain, M, N_samples, outcomes_N, p_names, PS, N_tot)

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
        s_denoised_summaries = [summarize_array(registry.s_eff_struct[k]; alpha=alpha) for k in 1:outcomes_N]
        summarized_effects[:spatial_denoised] = s_denoised_summaries
    end
    if !isnothing(registry.s_eff_noisy) && !isempty(registry.s_eff_noisy)
        s_noisy_summaries = [summarize_array(registry.s_eff_noisy[k]; alpha=alpha) for k in 1:outcomes_N]
        summarized_effects[:spatial_noisy] = s_noisy_summaries
    end
    if !isnothing(registry.t_eff) && !isempty(registry.t_eff)
        t_summaries = [summarize_array(registry.t_eff[k]; alpha=alpha) for k in 1:outcomes_N]
        summarized_effects[:temporal] = t_summaries
    end
    if !isnothing(registry.u_eff) && !all(iszero, registry.u_eff)
        summarized_effects[:seasonal] = summarize_array(registry.u_eff; alpha=alpha)
    end
    if !isnothing(registry.st_eff_maps) && !isempty(registry.st_eff_maps)
        st_summaries = [summarize_array(registry.st_eff_maps[k]; alpha=alpha) for k in 1:outcomes_N]
        summarized_effects[:spacetime_interaction] = st_summaries
    end

    # Summarize nested effects
    if haskey(registry, :all_nested_effects) && !isempty(registry.all_nested_effects)
        summarized_effects[:nested_effects] = Dict{Symbol, Any}()
        for (var_name, effect_samples) in registry.all_nested_effects
            summarized_effects[:nested_effects][var_name] = summarize_array(effect_samples; alpha=alpha)
        end
    end

    # 4. Generate Predictions and Compute Log-Likelihood
    p_denoised = zeros(Float64, N_tot, outcomes_N, N_samples)
    p_noisy = zeros(Float64, N_tot, outcomes_N, N_samples)
    log_lik = zeros(Float64, N_samples, M.y_N * outcomes_N)
    y_sigma_samples = zeros(N_tot, outcomes_N, N_samples)

    if "y_sigma" in p_names
        sig_samps = get_params_vector(chain, "y_sigma", outcomes_N)
        for k in 1:outcomes_N
            y_sigma_samples[:, k, :] .= sig_samps[:, k]'
        end
    end

    use_zi = get(M, :use_zi, false)
    r_nb_samples = "lik_r" in p_names ? get_params_vector(chain, "lik_r", outcomes_N) : fill(1.0, N_samples, outcomes_N)
    phi_zi_samples = "lik_phi" in p_names ? get_params_vector(chain, "lik_phi", 1) : fill(0.0, N_samples, 1)
    extra_p_samples = "extra_params" in p_names ? get_params_vector(chain, "extra_params", outcomes_N) : fill(1.0, N_samples, outcomes_N)

    for j in 1:N_samples
        for k in 1:outcomes_N
            eta_jk = eta_samples[:, k, j]
            r_val = r_nb_samples[j, k]
            phi_val = phi_zi_samples[j]
            extra_val = extra_p_samples[j, k]

            mu_vec = _apply_link_and_lik(family_str, eta_jk, use_zi, phi_val, r_val)
            p_denoised[:, k, j] .= mu_vec

            for i in 1:N_tot
                is_obs = i <= M.y_N
                eta_val = eta_jk[i]
                sig_y = y_sigma_samples[i, k, j]

                if is_obs
                    lik_obj = bstm_Likelihood(
                        fam_obj, [M.y_obs[i, k]]; sigma_y=[sig_y],
                        phi_zi=use_zi ? phi_val : -Inf, r_nb=r_val, extra_params=[extra_val]
                    )
                    log_lik[j, (k-1)*M.y_N + i] = Distributions.logpdf(lik_obj, eta_val)
                end

                if use_zi && rand() < phi_val
                    p_noisy[i, k, j] = 0.0
                else
                    temp_lik_obj = bstm_Likelihood(fam_obj, [0.0]; sigma_y=[sig_y], r_nb=[r_val], extra_params=[extra_val])
                    dist = get_dist_ref(fam_obj, temp_lik_obj, eta_val, sig_y)
                    p_noisy[i, k, j] = rand(dist)
                end
            end
        end
    end

    # 5. Summarize Predictions and other effects
    summarized_effects[:eta] = [summarize_array(eta_samples[:, k, :]; alpha=alpha) for k in 1:outcomes_N]
    summarized_effects[:predictions_denoised] = [summarize_array(p_denoised[1:M.y_N, k, :]; alpha=alpha) for k in 1:outcomes_N]
    summarized_effects[:predictions_noisy] = [summarize_array(p_noisy[1:M.y_N, k, :]; alpha=alpha) for k in 1:outcomes_N]
    if N_PS > 0
        summarized_effects[:ps_predictions_denoised] = [summarize_array(p_denoised[(M.y_N+1):end, k, :]; alpha=alpha) for k in 1:outcomes_N]
        summarized_effects[:ps_predictions_noisy] = [summarize_array(p_noisy[(M.y_N+1):end, k, :]; alpha=alpha) for k in 1:outcomes_N]
    end

    # 6. Final Diagnostics and Metadata
    summarized_effects[:waic] = _compute_waic(log_lik)
    summarized_effects[:log_lik_matrix] = log_lik
    summarized_effects[:family] = [spec[:family] for spec in M.likelihood_specs] # Return all families
    summarized_effects[:arch] = arch

    return NamedTuple(summarized_effects)
end


function _reconstruct(arch::MultifidelityArchitecture, modelname::String, chain, M, PS, alpha)
    # Dimensions and Scope Discovery 
    n_samples = size(chain, 1)
    p_names = string.(FlexiChains.parameters(chain))
    N_PS = isnothing(PS) ? 0 : size(PS.Xfixed, 1)
    N_tot = M.y_N + N_PS
    family_str = get(M.likelihood_specs[1], :family, "gaussian")
    local fam_obj = get_model_family(family_str)

    # #
    # 1. Parameter and Latent Field Discovery
    # Extracts posterior realizations for the main model components, now aware of prediction set
    registry = _discover_manifold_realizations(chain, M, n_samples, 1, p_names, PS, N_tot)

    # #
    # 2. Primary Linear Predictor Assembly
    # Modular assembly of the primary latent predictor, already handles N_tot
    eta_samples = _modular_eta_assembly(N_tot, registry, M, PS)

    # #
    # 3. Multifidelity / Nested Component Integration
    # This section specifically handles the contributions from nested sub-models
    summarized_effects = Dict{Symbol, Any}()

    # RFF-based multi-fidelity fields (z_latent, w_latent) - This part is for specific RFF-based multifidelity models, not the general nested() module.
    # This logic was modularized into _extract_nested_fields but was not being called.
    # We call it here to reconstruct and summarize these core multi-fidelity components.
    if haskey(M, :z_coords_s) # Check if RFF-based multi-fidelity is active
        nested_rff_fields = _extract_nested_fields(chain, M, n_samples, 1, p_names)
        if hasproperty(nested_rff_fields, :z_latent) && !isnothing(nested_rff_fields.z_latent)
            summarized_effects[:z_latent_field] = summarize_array(nested_rff_fields.z_latent; alpha=alpha)
        end
        if hasproperty(nested_rff_fields, :w_latent) && !isnothing(nested_rff_fields.w_latent)
            # Summarize each of the 3 w_latent fields
            summarized_effects[:w_latent_field_1] = summarize_array(nested_rff_fields.w_latent[:, 1, :]; alpha=alpha)
            summarized_effects[:w_latent_field_2] = summarize_array(nested_rff_fields.w_latent[:, 2, :]; alpha=alpha)
            summarized_effects[:w_latent_field_3] = summarize_array(nested_rff_fields.w_latent[:, 3, :]; alpha=alpha)
        end
    end

    summarized_effects[:nested_effects] = Dict{Symbol, Any}()

    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (z_key, z_meta) in M.nested_manifolds
            # Discover linking parameter rho_nested_{z_key}
            rho_name = _find_parameter(p_names, "rho_nested", "", string(z_key))
            rho_samples = get_params_vector(chain, rho_name, 1)

            # Discover nested spatial field realizations
            # Note: This discovery should ideally be centralized in _discover_manifold_realizations
            summarized_effects[:nested_effects][z_key] = Dict{Symbol, Any}()
            summarized_effects[:nested_effects][z_key][:rho_nested] = summarize_array(rho_samples; alpha=alpha)
            if haskey(z_meta, :model_space) && z_meta.model_space != "none"
                lat_base = "latent_nested_spatial"
                lat_name = _find_parameter(p_names, lat_base, "", string(z_key))
                lat_samples = get_params_vector(chain, lat_name, z_meta.s_N)' # [s_N x n_samples]

                # Map nested spatial effects to main model indices
                # Rationale: The low-fidelity field contributes to the main predictor scaled by rho_nested
                for j in 1:n_samples
                    for i in 1:N_tot
                        src = (i <= M.y_N) ? M : PS
                        idx_local = (i <= M.y_N) ? i : i - M.y_N
                        s_ptr = Int(src.s_idx[idx_local])
                        # Accumulate weighted contribution
                        eta_samples[i, 1, j] += rho_samples[j] * lat_samples[s_ptr, j]
                    end
                end
                summarized_effects[:nested_effects][z_key][:spatial_field] = summarize_array(lat_samples; alpha=alpha)
            end
        end
    end


    # Reshape for univariate-compatible predictive functions
    eta_samples_2d = reshape(eta_samples, N_tot, n_samples)
    vol_matrix = registry.sv_surface[1]

    p_denoised, p_noisy, log_lik = _process_ll_and_predictions(
        fam_obj, eta_samples_2d, chain, M, N_tot, n_samples, vol_matrix
    )

    # # 4. Summarization and Metadata Assembly
 
    # Primary effects
    if !isnothing(registry.s_eff_struct) && !isempty(registry.s_eff_struct)
        summarized_effects[:spatial_denoised] = summarize_array(registry.s_eff_struct[1]; alpha=alpha)
    end
    if !isnothing(registry.s_eff_noisy) && !isempty(registry.s_eff_noisy)
        summarized_effects[:spatial_noisy] = summarize_array(registry.s_eff_noisy[1]; alpha=alpha)
    end
    if !isnothing(registry.t_eff) && !isempty(registry.t_eff)
        summarized_effects[:temporal] = summarize_array(registry.t_eff[1]; alpha=alpha)
    end
    if !isnothing(registry.u_eff) && !all(iszero, registry.u_eff)
        summarized_effects[:seasonal] = summarize_array(registry.u_eff; alpha=alpha)
    end
    if !isnothing(registry.st_eff_maps) && !isempty(registry.st_eff_maps)
        summarized_effects[:spacetime_interaction] = summarize_array(registry.st_eff_maps[1]; alpha=alpha)
    end
    if !isnothing(registry.basis_coeffs) && !isempty(registry.basis_coeffs)
        summarized_effects[:smooth_effects] = Dict{Symbol, Any}()
        for (var_sym, coeffs_vec) in registry.basis_coeffs
            B_mat = M.basis_matrices[var_sym]
            effect_matrix = B_mat * coeffs_vec[1]
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
    if !isnothing(registry.dynamics_eff) && !all(iszero, registry.dynamics_eff)
        summarized_effects[:dynamics_eff] = summarize_array(registry.dynamics_eff[1]; alpha=alpha)
    end
    if !isnothing(registry.eigen_eff) && !all(iszero, registry.eigen_eff)
        summarized_effects[:eigen_eff] = summarize_array(registry.eigen_eff[1]; alpha=alpha)
    end

    # Predictive summaries
    summarized_effects[:eta] = summarize_array(eta_samples_2d[1:M.y_N, :]; alpha=alpha)
    summarized_effects[:predictions_denoised] = summarize_array(p_denoised[1:M.y_N, :]; alpha=alpha)
    summarized_effects[:predictions_noisy] = summarize_array(p_noisy[1:M.y_N, :]; alpha=alpha)

    if N_PS > 0
        summarized_effects[:ps_predictions_denoised] = summarize_array(p_denoised[(M.y_N+1):end, :]; alpha=alpha)
        summarized_effects[:ps_predictions_noisy] = summarize_array(p_noisy[(M.y_N+1):end, :]; alpha=alpha)
    end

    # # 5. Diagnostics 
    summarized_effects[:waic] = _compute_waic(log_lik) 
    summarized_effects[:log_lik_matrix] = log_lik
    summarized_effects[:arch] = arch
    summarized_effects[:family] = family_str

    return NamedTuple(summarized_effects)
end

function model_results_comprehensive(model, chain; au=nothing, data=nothing, n_samples=1000, alpha=0.05)
    # Metadata and Architecture Extraction
    M = model.args.M
    y_obs = M.y_obs
    raw_arch = get(M, :model_arch, "univariate")
    model_family = get(M, :model_family, "gaussian")

    # Technical Dispatch Resolution
    arch_type = if raw_arch == "multivariate"
        MultivariateArchitecture()
    else
        UnivariateArchitecture()
    end

    # Latent Manifold Reconstruction
    res = _reconstruct(arch_type, "model_results", chain, M, nothing, alpha)

    # Performance Metric Assessment
    y_pred = res.predictions_denoised.mean
    y_obs_flat = vec(collect(y_obs))
    y_pred_flat = vec(collect(y_pred))
    valid_idx = findall(x -> !isnan(x) && !isnothing(x), y_obs_flat)

    rmse_val = 0.0
    r_pearson = 0.0

    if !isempty(valid_idx)
        obs_v = y_obs_flat[valid_idx]
        pred_v = y_pred_flat[valid_idx]
        rmse_val = sqrt(mean((obs_v .- pred_v).^2))
        try
            r_pearson = cor(obs_v, pred_v)
        catch
            r_pearson = 0.0
        end
    end

    # MCMC Diagnostic Extraction
    mean_rhat = 1.0
    min_ess = 0.0
    try
        chains_obj = MCMCChains.Chains(chain)
        df_stats = DataFrame(MCMCChains.summarize(chains_obj))

        if hasproperty(df_stats, :rhat)
            r_vals = filter(x -> !isnan(x) && x > 0, df_stats.rhat)
            mean_rhat = isempty(r_vals) ? 1.0 : mean(r_vals)
        end

        e_col = hasproperty(df_stats, :ess_bulk) ? :ess_bulk : (hasproperty(df_stats, :ess) ? :ess : nothing)
        if !isnothing(e_col)
            e_vals = filter(x -> !isnan(x) && x >= 0, df_stats[!, e_col])
            min_ess = isempty(e_vals) ? 0.0 : minimum(e_vals)
        end
    catch e
        @warn "Diagnostic extraction failed: $e. Falling back to default values."
    end

    # Compute Time Recovery
    sampling_time = 0.0
    if hasproperty(chain, :info) && haskey(chain.info, :stop_time)
        sampling_time = (chain.info.stop_time - chain.info.start_time)
    elseif hasproperty(chain, :_metadata) && hasproperty(chain._metadata, :sampling_time)
        sampling_time = chain._metadata.sampling_time[]
    end

    # Formatted Reporting
    println("\n--- Model Registry Summary ---")
    println("Architecture:     ", raw_arch)
    println("Family:           ", model_family)
    println("Space Component:  ", get(res, :model_space, "none"))
    println("Time Component:   ", get(res, :model_time, "none"))

    println("\n--- Performance Metrics ---")
    println("RMSE:             ", round(rmse_val, digits=4))
    println("Pearson r:        ", round(r_pearson, digits=4))
    println("WAIC Score:       ", round(get(res, :waic, 0.0), digits=2))

    println("\n--- MCMC Diagnostics ---")
    println("Compute Time:     ", round(sampling_time, digits=2), " seconds")
    println("Mean R-hat:       ", round(mean_rhat, digits=4))
    println("Minimum ESS:      ", round(min_ess, digits=2))

    # Post-Stratification Weight Validation
    if haskey(res, :ps_weights)
        println("\n--- Post-Stratification Audit ---")
        println("PS Weights Found: Yes")
        println("Weight Mean:      ", round(mean(res.ps_weights.mean), digits=4))
    end

    data_for_plots = isnothing(data) ? get(M, :data, nothing) : data
    plots = _generate_plots(res, M; au=au, data=data_for_plots)

    return (
        metrics = (rmse = rmse_val, r_pearson = r_pearson, ess = min_ess, rhat = mean_rhat, waic = get(res, :waic, 0.0), time = sampling_time),
        pstats = res,
        plots = plots
    )
end

function _generate_plots(res, M; au=nothing, data=nothing, ts=1, outcome=1)
    # Initialization of the plot registry
    plots = Dict{Symbol, Any}()
    
    # Metadata Retrieval
    y_obs = get(M, :y_obs, nothing)
    polygons = isnothing(au) ? nothing : get(au, :polygons, nothing)
    centroids = isnothing(au) ? nothing : get(au, :centroids, nothing)

    # 1. Posterior Predictive Check (PPC)
    if hasproperty(res, :predictions_denoised)
        if isnothing(y_obs)
            @info "Skipping PPC plot: Observation data not found in model configuration `M`."
        else
            is_mv = res.arch isa MultivariateArchitecture
            pred_summary = is_mv ? res.predictions_denoised[outcome] : res.predictions_denoised
            
            if !isnothing(pred_summary) && hasproperty(pred_summary, :mean)
                y_p = vec(pred_summary.mean)
                y_o = is_mv ? vec(y_obs[:, outcome]) : vec(y_obs)

                if length(y_p) == length(y_o)
                    p_ppc = scatter(y_p, y_o, title="Posterior Predictive Check", xlabel="Predicted", ylabel="Observed", alpha=0.5, markersize=3, markerstrokewidth=0, legend=false)
                    clean_p = filter(!isnan, y_p)
                    clean_o = filter(!isnan, y_o)
                    if !isempty(clean_p) && !isempty(clean_o)
                        min_val, max_val = min(minimum(clean_p), minimum(clean_o)), max(maximum(clean_p), maximum(clean_o))
                        plot!(p_ppc, [min_val, max_val], [min_val, max_val], color=:red, ls=:dash, lw=1.5)
                    end
                    plots[:ppc] = p_ppc
                end
            end
        end
    end

    # Internal Utility for Spatial Plots
    function _create_choropleth_plot(field_data, title_str, polygons, centroids)
        if isnothing(field_data) || !hasproperty(field_data, :mean)
            @info "Skipping spatial plot '$title_str': The effect data is missing or invalid."
            return nothing
        end
        if isnothing(polygons) && isnothing(centroids)
            @info "Skipping spatial plot '$title_str': No spatial geometry (`polygons` or `centroids`) provided. Pass the `au` object to `model_results_comprehensive`."
            return nothing
        end
        s_mean = vec(collect(field_data.mean))
        if all(iszero, s_mean)
            @info "Skipping spatial plot '$title_str': The mean effect is zero. This may indicate a short or non-converged MCMC chain."
            return nothing
        end
        if !isnothing(polygons) && length(polygons) >= length(s_mean)
            return plot_choropleth(s_mean, polygons; title=title_str)
        elseif !isnothing(centroids)
            return scatter(getindex.(centroids, 1), getindex.(centroids, 2), marker_z=s_mean, markersize=4, c=:viridis, label=nothing, title=title_str, aspect_ratio=:equal)
        end
        return nothing
    end

    # 2. Spatial Latent Fields
    if hasproperty(res, :spatial_denoised)
        s_field = (res.arch isa MultivariateArchitecture) ? res.spatial_denoised[outcome] : res.spatial_denoised
        p = _create_choropleth_plot(s_field, "Spatial Denoised Effect", polygons, centroids)
        if !isnothing(p); plots[:spatial_denoised] = p; end
    end
    if hasproperty(res, :spatial_noisy)
        s_field = (res.arch isa MultivariateArchitecture) ? res.spatial_noisy[outcome] : res.spatial_noisy
        p = _create_choropleth_plot(s_field, "Total Spatial Effect (incl. IID)", polygons, centroids)
        if !isnothing(p); plots[:spatial_noisy] = p; end
    end

    # 3. Temporal Main Trend
    if hasproperty(res, :temporal)
        t_field = (res.arch isa MultivariateArchitecture) ? res.temporal[outcome] : res.temporal
        if !isnothing(t_field) && hasproperty(t_field, :mean)
            if all(iszero, t_field.mean)
                @info "Skipping temporal plot: The mean temporal effect is zero. This may indicate a short or non-converged MCMC chain."
            else
                tm, tl, tu = vec(t_field.mean), vec(t_field.lower), vec(t_field.upper)
                plots[:temporal] = plot(tm, ribbon=(tm .- tl, tu .- tm), title="Temporal Trend", lw=2, fillalpha=0.2, color=:royalblue, legend=false, xlabel="Time Index")
            end
        end
    end

    # 4. Seasonal Dynamics
    if hasproperty(res, :seasonal) && !isnothing(res.seasonal) && hasproperty(res.seasonal, :mean)
        if all(iszero, res.seasonal.mean)
            @info "Skipping seasonal plot: The mean seasonal effect is zero. This may indicate a short or non-converged MCMC chain."
        else
            um, ul, uu = vec(res.seasonal.mean), vec(res.seasonal.lower), vec(res.seasonal.upper)
            plots[:seasonal] = plot(um, ribbon=(um .- ul, uu .- um), title="Seasonal Component", lw=2, fillalpha=0.2, color=:forestgreen, legend=false, xlabel="Period")
        end
    end

    # 5. Smooth Covariate Effects
    if hasproperty(res, :smooth_effects) && res.smooth_effects isa Dict
        if isnothing(data)
            @info "Skipping smooth effects plots: The `data` DataFrame with original covariates was not provided."
        else
            smooth_plots = Dict{Symbol, Any}()
            for (var_sym, smooth_summary) in res.smooth_effects
                if hasproperty(smooth_summary, :mean) && all(iszero, smooth_summary.mean)
                    @info "Skipping smooth effect plot for '$var_sym': The mean effect is zero."
                    continue
                end
                if hasproperty(data, var_sym) && hasproperty(smooth_summary, :mean)
                    cov_data = data[!, var_sym]
                    p_order = sortperm(cov_data)
                    sm, sl, su = vec(smooth_summary.mean), vec(smooth_summary.lower), vec(smooth_summary.upper)
                    smooth_plots[var_sym] = plot(cov_data[p_order], sm[p_order], ribbon=(sm[p_order] .- sl[p_order], su[p_order] .- sm[p_order]), title="Smooth Effect: $var_sym", xlabel=string(var_sym), ylabel="Latent Effect", legend=false, color=:darkorange, fillalpha=0.2)
                end
            end
            if !isempty(smooth_plots); plots[:smooth_effects] = smooth_plots; end
        end
    end

    # 6. Fixed Effects Forest Plot
    if hasproperty(res, :fixed_effects) && !isnothing(res.fixed_effects)
        fe_summary = (res.arch isa MultivariateArchitecture) ? res.fixed_effects[outcome] : res.fixed_effects
        if hasproperty(fe_summary, :mean)
            if all(iszero, fe_summary.mean)
                @info "Skipping fixed effects plot: All mean effects are zero."
            else
                fm, fl, fu = vec(fe_summary.mean), vec(fe_summary.lower), vec(fe_summary.upper)
                n_coeffs = length(fm)
                if n_coeffs > 0
                    coef_names = haskey(M, :Xfixed) ? string.(names(M.Xfixed, 2)) : ["Coef_$i" for i in 1:n_coeffs]
                    p_forest = scatter(fm, 1:n_coeffs, xerror=(fm .- fl, fu .- fm), yticks=(1:n_coeffs, coef_names), title="Fixed Effects Coefficients", xlabel="Estimate", markersize=4, color=:black, legend=false)
                    vline!(p_forest, [0], color=:red, ls=:dash, lw=1)
                    plots[:fixed_effects] = p_forest
                end
            end
        end
    end

    return NamedTuple(plots)
end

function model_results_plots(res)
    # Displays all plots generated by `model_results_comprehensive` and stored
    # in the results object.
    if !hasproperty(res, :plots) || isempty(res.plots)
        println("No plots found in the results object.")
        return
    end

    println("Displaying generated plots...")
    for (plot_name, plot_obj) in pairs(res.plots)
        if plot_obj isa Dict # Handle nested plot dictionaries like for smooth_effects
            for (sub_name, sub_plot) in plot_obj
                println("--- Plot: $plot_name -> $sub_name ---")
                display(sub_plot)
            end
        else
            println("--- Plot: $plot_name ---")
            display(plot_obj)
        end
    end
    println("--- End of plots ---")
end



function plot_choropleth(values::AbstractVector, polygons::Vector; title="Spatial Distribution", cmap=:viridis)
    plt = Plots.plot(aspect_ratio=:equal, title=title, legend=true)

    for i in 1:min(length(polygons), length(values))
        poly_coords = polygons[i]
        if length(poly_coords) > 2
            px = [pt[1] for pt in poly_coords if !isnan(pt[1])]
            py = [pt[2] for pt in poly_coords if !isnan(pt[2])]

            if !isempty(px)
                if (px[1], py[1]) != (px[end], py[end])
                    push!(px, px[1])
                    push!(py, py[1])
                end

                Plots.plot!(plt, px, py,
                    seriestype=:shape,
                    fill_z=values[i],
                    c=cmap,
                    linecolor=:black,
                    lw=0.5,
                    fillalpha=0.8,
                    label=nothing
                )
            end
        end
    end
    return plt
end

 
 

function plot_posterior_results(stats, M=nothing, areal_units=nothing; s_x=nothing, s_y=nothing, time_slice=nothing, effect=:spatial, cov_idx=1, show_pts=false)
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: Comprehensive visualization for posterior results from bstm models. 
    # Inputs: stats (results object), M (model config), areal_units, and various plotting options.
    # Outputs: A Plots.jl plot object.
    st = getproperty(stats, effect)
    isnothing(st) && return nothing
    if st isa Real
        return Plots.plot(title="$effect (Fixed: $st)")
    end
 
    if effect == :beta_cov
        b_list = get(stats, :beta_cov, nothing)
        isnothing(b_list) && return nothing
        b_stats = b_list isa AbstractVector ? b_list[cov_idx] : b_list
        (isnothing(b_stats) || b_stats isa Real) && return nothing
        n_levels = size(b_stats.mean, 1)
        return StatsPlots.bar(1:n_levels, b_stats.mean[:,1],
                  yerror=(b_stats.mean[:,1] .- b_stats.lower[:,1], b_stats.upper[:,1] .- b_stats.mean[:,1]),
                  title="Covariate $cov_idx Effects", xlabel="Level", ylabel="Effect Size", legend=false)

    elseif effect == :b_class1 || effect == :b_class2
        b_stats = st
        if isnothing(b_stats) || b_stats isa Real; return nothing; end
        n_levels = size(b_stats.mean, 1)
        return StatsPlots.bar(1:n_levels, b_stats.mean[:,1],
                  yerror=(b_stats.mean[:,1] .- b_stats.lower[:,1], b_stats.upper[:,1] .- b_stats.mean[:,1]),
                  title="$effect Levels", xlabel="Class Index", ylabel="Effect Size", legend=false)

    elseif effect == :temporal
        t_stats = st
        if isnothing(t_stats) || t_stats isa Real; return nothing; end
        n_times = length(t_stats.mean)
        return StatsPlots.plot(1:n_times, t_stats.mean,
                   ribbon=(t_stats.mean .- t_stats.lower, t_stats.upper .- t_stats.mean),
                   fillalpha=0.2, lw=2, title="Temporal Main Effect", xlabel="Time Index", ylabel="Effect (Latent Scale)", legend=false)

    elseif effect == :seasonal
        t_stats = st
        if isnothing(t_stats) || t_stats isa Real; return nothing; end
        n_times = length(t_stats.mean)
        return StatsPlots.plot(1:n_times, t_stats.mean,
                   ribbon=(t_stats.mean .- t_stats.lower, t_stats.upper .- t_stats.mean),
                   fillalpha=0.2, lw=2, title="Seasonal Effect", xlabel="Time Index", ylabel="Effect (Latent Scale)", legend=false)

    elseif effect in [:spatial, :spatial_structured, :spatial_unstructured, :predictions_denoised, :predictions_noisy, :residuals, :eta_gp, :hidden_layer]
        plt = StatsPlots.plot(aspect_ratio=:equal, title="$effect (T=$(time_slice))", legend=true)

        values = if hasproperty(st, :mean)
            st.mean
        elseif effect == :spatial_structured
            stats.spatial_structured.mean
        elseif effect == :spatial_unstructured
            stats.spatial_unstructured.mean
        elseif effect == :eta_gp
            haskey(stats, :eta_gp) ? stats.eta_gp.mean : error("eta_gp not found in stats")
        elseif effect == :hidden_layer
            haskey(stats, :h1) ? stats.h1.mean : error("hidden layer h1 not found in stats")
        elseif effect == :predictions_denoised && !isnothing(time_slice)
            stats.predictions_denoised.mean[:, time_slice]
        elseif effect == :predictions_noisy && !isnothing(time_slice)
            stats.predictions_noisy.mean[:, time_slice]
        else
            error("Effect $effect requires specific keys in stats or time_slice index")
        end

        n_to_plot = min(length(areal_units.polygons), length(values))

        for i in 1:n_to_plot
            poly_coords = areal_units.polygons[i]
            if length(poly_coords) > 2
                px = [pt[1] for pt in poly_coords if !isnan(pt[1])]
                py = [pt[2] for pt in poly_coords if !isnan(pt[2])]

                if !isempty(px)
                    if (px[1], py[1]) != (px[end], py[end])
                        push!(px, px[1]); push!(py, py[1])
                    end

                    val = values[i]
                    StatsPlots.plot!(plt, px, py,
                        seriestype=:shape,
                        fill_z=val,
                        c=:RdYlBu,
                        linecolor=:black,
                        linewidth=0.5,
                        fillalpha=0.8,
                        legend=false
                    )
                end
            end
        end

        if show_pts
            StatsPlots.scatter!(plt, s_x, s_y,
                markersize=1, markercolor=:gray, alpha=0.2, label="Observations")
        end

        StatsPlots.scatter!(plt, [c[1] for c in areal_units.centroids], [c[2] for c in areal_units.centroids],
            markersize=2, markercolor=:white, markerstrokecolor=:black, alpha=0.5, label="Centroids")

        return plt
    else
        error("Effect $effect not recognized.")
    end
end




function plot_posterior_vs_prior(model::DynamicPPL.Model, chain::MCMCChains.Chains, param_sym::Symbol; n_prior_samples=1000, title="Posterior vs Prior")
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: Overlays the posterior density of a parameter with its prior density. 
    # Inputs: model, chain, param_sym, n_prior_samples, title.
    # Outputs: A Plots.jl plot object.

    post_samples = vec(chain[param_sym].data)

    prior_chain = sample(model, Prior(), n_prior_samples, progress=false)
    prior_samples = vec(prior_chain[param_sym].data)

    plt = StatsPlots.density(post_samples, label="Posterior: $param_sym", lw=3, color=:blue, fill=(0, 0.2, :blue))
    StatsPlots.density!(plt, prior_samples, label="Prior (sampled)", lw=2, ls=:dash, color=:red)

    title!(plt, title)
    xlabel!(plt, "Value")
    ylabel!(plt, "Density")

    return plt
end



function predict(model_obj::DynamicPPL.Model, chain, new_data::DataFrame; n_samples::Int=100, alpha=0.05)
    # Description: Primary engine for projecting recovered latent manifolds onto new spatiotemporal coordinates.
    # Rationale: Standardizing the out-of-sample path to support the full  Taxonomy.
    # Timestamp: 2026-07-03 10:00:00
    # Rationale: Overhauled the prediction configuration (`PS`) generation to be robust. Instead of
    #            re-running `bstm_config`, this version creates a new configuration by inheriting
    #            the training model's structure and selectively updating data-dependent fields.
    #            This resolves a critical flaw where smoother types and parameters were not correctly
    #            propagated to the prediction set.
 
    # # 1. Training Metadata Recovery
    # Rationale: M_train contains the complete configuration and technical registry from the training phase.
    M_train = model_obj.args.M
    n_samples_total = size(chain, 1)
    n_samps = min(n_samples_total, n_samples)

    # # 2. Prediction Metadata Configuration (PS)
    # Rationale: Create a new configuration `PS` for the prediction data. We start by copying the
    #            training configuration and then update only the data-dependent parts. This ensures
    #            that the model structure (manifolds, priors, etc.) is identical.
    
    # Convert NamedTuple to a mutable dictionary for updates
    PS_dict = Dict(pairs(M_train))

    # Update with new data and dimensions
    PS_dict[:data] = new_data
    PS_dict[:y_obs] = zeros(nrow(new_data)) # Dummy response
    PS_dict[:y_N] = nrow(new_data)
    PS_dict[:log_offset] = hasproperty(new_data, :log_offset) ? new_data.log_offset : zeros(nrow(new_data))
    PS_dict[:weights] = hasproperty(new_data, :weights) ? new_data.weights : ones(nrow(new_data))
    PS_dict[:trials] = hasproperty(new_data, :trials) ? new_data.trials : ones(Int, nrow(new_data))

    # Re-create fixed effects design matrix for the new data using the original formula part
    if haskey(M_train, :formula)
        decomposed_formula = decompose_bstm_formula(M_train.formula)
        fixed_effects_formula_part = join(decomposed_formula.fixed_effects, " + ")
        if decomposed_formula.has_intercept
             fixed_effects_formula_part = isempty(strip(fixed_effects_formula_part)) ? "1" : "1 + " * fixed_effects_formula_part
        end
        if !isempty(strip(fixed_effects_formula_part))
            PS_dict[:Xfixed] = create_fixed_design(fixed_effects_formula_part, new_data; contrasts=get(M_train, :contrasts, Dict()))
            PS_dict[:Xfixed_N] = size(PS_dict[:Xfixed], 2)
        end
    end

    # Update indices from new_data
    if haskey(M_train, :s_idx_var) && !isnothing(M_train.s_idx_var) && hasproperty(new_data, M_train.s_idx_var)
        PS_dict[:s_idx] = new_data[!, M_train.s_idx_var]
    end
    if haskey(M_train, :t_idx_var) && !isnothing(M_train.t_idx_var) && hasproperty(new_data, M_train.t_idx_var)
        PS_dict[:t_idx] = new_data[!, M_train.t_idx_var]
    end

    # # 3. Manifold Coordinate Alignment
    # Rationale: Aligning out-of-sample points with the training grid for discrete spatial models.
    if haskey(M_train, :centroids) && !isnothing(M_train.centroids)
        centroids_train = M_train.centroids
        nx = hasproperty(new_data, :s_x) ? new_data.s_x : zeros(nrow(new_data))
        ny = hasproperty(new_data, :s_y) ? new_data.s_y : zeros(nrow(new_data))
        
        PS_s_idx = Vector{Int}(undef, nrow(new_data))
        for i in 1:nrow(new_data)
            # Find nearest neighbor in the training unit grid
            dists = [sum(((nx[i], ny[i]) .- c).^2) for c in centroids_train]
            PS_s_idx[i] = argmin(dists)
        end
        PS_dict[:s_idx] = PS_s_idx
    end

    # # 4. Hyper-Volumetric Basis Projection (1D-4D)
    # Rationale: Reconstructing basis matrices (Splines/RFF/FFT) for new coordinates
    #            using the original model specifications from the training manifolds.
    if haskey(M_train, :manifolds) && !isempty(M_train.manifolds)
        ps_basis_registry = Dict{Symbol, Any}()
        smooth_specs = filter(s -> s.domain == :smooth, M_train.manifolds)
        
        for spec in smooth_specs
            # The key for the basis matrix is derived from the variable names.
            # This logic must match the key generation in `bstm_config`.
            v_sym = Symbol(join(spec.params.variables, "_"))

            if haskey(M_train.basis_matrices, v_sym)
                B_train = M_train.basis_matrices[v_sym]
                m_obj = spec.manifold_obj
                model_type_str = lowercase(string(typeof(m_obj)))
                
                vars = Symbol.(spec.params.variables)
                n_vars = length(vars)
                nb = size(B_train, 2)
                
                # Reconstruct the basis matrix for the new data using the correct model type and parameters
                if n_vars == 1
                    ps_basis_registry[v_sym] = bstm_smooth_basis_1D(model_type_str, new_data[!, vars[1]], nb; spec.params...)
                elseif n_vars == 2
                    coords_new = Matrix{Float64}(new_data[!, vars])
                    ps_basis_registry[v_sym] = bstm_smooth_basis_2D(model_type_str, coords_new, nb; spec.params...)
                elseif n_vars == 3
                    coords_new = Matrix{Float64}(new_data[!, vars])
                    ps_basis_registry[v_sym] = bstm_smooth_basis_3D(model_type_str, coords_new, nb; spec.params...)
                elseif n_vars == 4
                    coords_new = Matrix{Float64}(new_data[!, vars])
                    ps_basis_registry[v_sym] = bstm_smooth_basis_4D(model_type_str, coords_new, nb; spec.params...)
                end
            end
        end
        PS_dict[:basis_matrices] = ps_basis_registry
    end

    # Convert dictionary back to NamedTuple for the reconstruct engine
    PS = NamedTuple(PS_dict)

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
    chain_sub = chain[1:min(n_samps, end), :, :]

    # Rationale: _reconstruct treats PS as a grid for Post-Stratification, 
    # which is mathematically equivalent to out-of-sample prediction.
    res = _reconstruct(arch_type, "prediction_projection", chain_sub, M_train, PS, alpha)

    println("--- Projection Complete [Observations: ", nrow(new_data), "] ---")

    return (
        predictions_denoised = res.predictions_denoised,
        predictions_noisy = res.predictions_noisy,
        pstats = res,
        PS = PS,
        centroids = haskey(PS, :s_coord) ? PS.s_coord : nothing
    )
end
 

# 1. PSIS-LOO Implementation for BSTM
# Rationale: Standardizing the extraction of log-likelihood matrices to provide 
# Expected Log Pointwise Predictive Density (ELPD) estimates. 
function bstm_loo(model_obj::DynamicPPL.Model, chain; alpha=0.05)    
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: A utility for performing Leave-One-Out Cross-Validation using Pareto Smoothed Importance 
    #          Sampling (PSIS-LOO) to assess a model's out-of-sample predictive accuracy.
    # Inputs: model_obj, chain, alpha.
    # Outputs: A NamedTuple containing the LOO object, metrics, log-likelihood matrix, and Pareto k values.
    
    
    # # 1. Metadata and Architecture Extraction
    # Rationale: M contains the configuration and technical registry required for reconstruction.
    M = model_obj.args.M
    raw_arch = get(M, :model_arch, "univariate")

    # # 2. Technical Dispatch Resolution
    # Mapping the configuration string to the architectural dispatch types.
    arch_type = if raw_arch == "univariate"
        UnivariateArchitecture()
    elseif raw_arch == "multivariate"
        MultivariateArchitecture()
    else
        UnivariateArchitecture()
    end

    # # 3. Latent Manifold Reconstruction for Likelihood Registry
    # Rationale: _reconstruct generates the [Samples x Observations] log-likelihood matrix.
    # We utilize alpha for consistent summarization during the recovery phase.
    println("Audit: Recovering pointwise log-likelihood registry...")
    res = _reconstruct(arch_type, "loo_recovery", chain, M, nothing, alpha)

    # # 4. Matrix Extraction and Validation
    # Rationale: Ensuring the log_lik_matrix matches the observation grid dimensions.
    log_lik = res.log_lik_matrix
    n_samples, n_obs = size(log_lik)

    println("Audit: Processing ", n_samples, " samples for ", n_obs, " observations.")

    # # 5. PSIS-LOO Calculation via PosteriorStats
    # Rationale: LOO-CV provides a reliable estimate of out-of-sample predictive performance.
    loo_result = nothing
    try
        loo_result = loo(log_lik)
    catch e
        @error "BSTM Selection Error: PSIS-LOO calculation failed. Error: " * string(e)
        return nothing
    end

    println("\n--- BSTM Model Selection Report ---")
    println("Expected Log Pointwise Predictive Density (ELPD): ", round(loo_result.estimates[:elpd_loo, :estimate], digits=2))
    println("Effective Number of Parameters (p_loo):          ", round(loo_result.estimates[:p_loo, :estimate], digits=2))
    println("LOO Information Criterion:                       ", round(loo_result.estimates[:looic, :estimate], digits=2))

    # Check for influential observations (k > 0.7)
    # Rationale: Identifying data points where the importance weight is unstable.
    pareto_k = loo_result.pointwise[:pareto_k]
    influential_count = count(x -> x > 0.7, pareto_k)
    if influential_count > 0
        @warn "BSTM: " * string(influential_count) * " influential observations detected (Pareto k > 0.7)."
    end

    return (
        loo_obj = loo_result,
        metrics = (
            elpd = loo_result.estimates[:elpd_loo, :estimate],
            p_loo = loo_result.estimates[:p_loo, :estimate],
            looic = loo_result.estimates[:looic, :estimate]
        ),
        log_lik_matrix = log_lik,
        pareto_k = pareto_k
    )
end



# 2. Bayes Factor Suite (Manifold Comparison)
function compare_manifolds(loo_a_report, loo_b_report; model_names=["Model_A", "Model_B"])    
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: A utility for formal model comparison between two fitted `bstm` models. It uses 
    #          their PSIS-LOO results to compute the difference in Expected Log Pointwise 
    #          Predictive Density (ELPD) and provides a statistical basis for model selection.
    # Inputs: loo_a_report, loo_b_report, model_names.
    # Outputs: A NamedTuple containing the comparison table, ELPD difference, and LOO objects.
    # Description: Performs formal model selection between two BSTM manifold candidates.
    # Rationale: Standardizing the assessment of ELPD differences and complexity trade-offs.
    # Requirements: Absolute parity with the PSIS-LOO metrics.

    println("--- Starting BSTM Manifold Comparison ---")

    # # 1. LOO Object Extraction
    # Rationale: Extracting the underlying PosteriorStats LOO objects for comparison.
    loo_a = loo_a_report.loo_obj
    loo_b = loo_b_report.loo_obj

    # # 2. Formal Selection Metric Calculation
    # Rationale: The difference in ELPD is the primary metric for out-of-sample performance.
    # We utilize the compare function from PosteriorStats to compute deltas and standard errors.
    comparison_stats = nothing
    try
        comparison_stats = compare([loo_a, loo_b])
    catch e
        @error "BSTM Comparison Error: Selection suite failed. Error: " * string(e)
        return nothing
    end

    # # 3. Parameter and Diagnostic Extraction
    # Rationale: Collecting effective parameter counts (p_loo) to assess complexity.
    p_loo_a = loo_a_report.metrics.p_loo
    p_loo_b = loo_b_report.metrics.p_loo

    elpd_a = loo_a_report.metrics.elpd
    elpd_b = loo_b_report.metrics.elpd

    # # 4. Report Generation
    println("\n--- BSTM Manifold Selection Registry ---")
    println("Model A (", model_names[1], "): ELPD = ", round(elpd_a, digits=2), " | p_loo = ", round(p_loo_a, digits=2))
    println("Model B (", model_names[2], "): ELPD = ", round(elpd_b, digits=2), " | p_loo = ", round(p_loo_b, digits=2))

    diff_elpd = elpd_a - elpd_b
    println("\nELPD Delta (A - B): ", round(diff_elpd, digits=2))

    # Interpretation Logic
    # Rationale: If |diff_elpd| > 4, the difference is generally considered significant.
    if abs(diff_elpd) > 4.0
        winning_model = diff_elpd > 0 ? model_names[1] : model_names[2]
        println("CONCLUSION: ", winning_model, " is statistically preferred based on predictive density.")
    else
        println("CONCLUSION: Competing manifold structures provide indistinguishable predictive density.")
    end

    # # 5. Table Construction
    comparison_df = DataFrame(
        Metric = ["ELPD (LOO)", "Effective Parameters (p_loo)", "LOO-IC"],
        Model_A = [elpd_a, p_loo_a, loo_a_report.metrics.looic],
        Model_B = [elpd_b, p_loo_b, loo_b_report.metrics.looic]
    )

    comparison_df[!, :Delta] = comparison_df.Model_A .- comparison_df.Model_B

    display(comparison_df)

    return (
        comparison_table = comparison_df,
        elpd_diff = diff_elpd,
        loo_objects = (loo_a, loo_b)
    )
end


function bstm_cv_orchestrator(
    formula::String, 
    data::DataFrame; 
    method::Symbol = :lolo, 
    lolo_var::Symbol = :s_idx, 
    n_folds::Int = 5, 
    n_samples::Int = 500, 
    sampler = NUTS(500, 0.65), 
    alpha = 0.05, 
    kwargs...
)    
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: An orchestration utility for performing cross-validation. It supports standard 
    #          k-fold and Leave-One-Location-Out (LOLO) strategies to assess model performance on 
    #          held-out data.
    # Inputs: formula, data, and optional parameters for CV method, sampler, etc.
    # Outputs: A NamedTuple containing fold results and summary metrics.
    # # 1. Metadata Discovery and Outcome Resolution
    # The formula is decomposed to identify the primary response variable and module requirements.
    meta_discovery = decompose_bstm_formula(formula)
    response_name = Symbol(meta_discovery.outcomes[1])

    # # 2. Partition Strategy Selection
    # Establishing fold indices based on spatiotemporal logic or random sampling.
    folds_indices = Vector{Vector{Int}}()

    if method == :lolo
        # Leave-One-Location-Out: Grouping indices by the spatial unit identifier.
        unique_locs = unique(data[!, lolo_var])
        for loc in unique_locs
            push!(folds_indices, findall(x -> x == loc, data[!, lolo_var]))
        end
    else
        # Standard K-Fold: Random permutation of row indices.
        n_obs = size(data, 1)
        row_indices = Random.randperm(n_obs)
        fold_size = div(n_obs, n_folds)
        for i in 1:n_folds
            idx_start = (i - 1) * fold_size + 1
            idx_end = i == n_folds ? n_obs : i * fold_size
            push!(folds_indices, row_indices[idx_start:idx_end])
        end
    end

    # # 3. Cross-Validation Loop
    fold_results = []
    n_actual_folds = length(folds_indices)

    for (f_idx, test_idx) in enumerate(folds_indices)
        # Splitting dataset into training and testing partitions.
        # Training mask is constructed to exclude the test indices.
        train_mask = trues(size(data, 1))
        train_mask[test_idx] .= false
        
        train_data = data[train_mask, :]
        test_data = data[test_idx, :]

        # # 4. Modular Training Configuration
        # Pre-configuring the model to ensure technical registries (W, s_N, t_N) are consistent.
        # bstm_config resolves the manifold registry M.
        opt_train = bstm_config(formula, train_data; kwargs...)

        # # 5. Model Execution
        # Dispatching to the modular univariate or multivariate supervisor.
        model_train = bstm(opt_train)
        
        # Posterior sampling using the requested sampler configuration.
        chain_train = sample(model_train, sampler, n_samples, progress = false)

        # # 6. Manifold Projection (Out-of-Sample)
        # Using the standardized predict() function which handles reconstruction of S, T, and Smooth basis.
        # This ensures that PS (Prediction Surface) alignment is consistent with the modular BSTM taxonomy.
        res_pred = predict(model_train, chain_train, test_data, n_samples = div(n_samples, 2), alpha = alpha)

        # # 7. Performance Assessment
        # Extracting denoised expectations for the test partition.
        y_test_obs = test_data[!, response_name]
        y_test_pred = res_pred.predictions_denoised.mean

        # Verification of dimensional parity between prediction and observation.
        if length(y_test_obs) == length(y_test_pred)
            residuals = y_test_obs .- y_test_pred
            rmse = sqrt(Statistics.mean(residuals.^2))
            
            # R-Squared calculation with safety floor for variance.
            ss_res = sum(residuals.^2)
            ss_tot = sum((y_test_obs .- Statistics.mean(y_test_obs)).^2)
            r2 = 1.0 - (ss_res / (ss_tot + 1e-15))

            push!(fold_results, (fold=f_idx, rmse=rmse, r2=r2))
        else
            @warn "Fold $f_idx: Prediction length mismatch. Observed: $(length(y_test_obs)), Predicted: $(length(y_test_pred))"
        end
    end

    # # 8. Aggregate Reporting
    mean_rmse = Statistics.mean([r.rmse for r in fold_results])
    mean_r2 = Statistics.mean([r.r2 for r in fold_results])

    return (
        folds = fold_results,
        mean_rmse = mean_rmse,
        mean_r2 = mean_r2,
        response_var = response_name,
        method = method,
        n_folds = n_actual_folds
    )
end

 

 