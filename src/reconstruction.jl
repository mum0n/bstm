

abstract type AbstractModelArchitecture end
struct UnivariateArchitecture <: AbstractModelArchitecture end
struct MultivariateArchitecture <: AbstractModelArchitecture end
struct MultifidelityArchitecture <: AbstractModelArchitecture end
struct ExampleArchitecture <: AbstractModelArchitecture end

# ==============================================================================
# SECTION 1: CORE UTILITIES FOR PARAMETER EXTRACTION
# ==============================================================================

function get_kernel_from_string(kernel_name::String)
    # Purpose: Maps a string identifier to a `KernelFunctions.jl` kernel object.
    # Rationale: Centralizes kernel selection for GP-based models.
    # Inputs:
    #   - kernel_name: The string name of the kernel.
    # Outputs: A `Kernel` object.
    k_name = lowercase(kernel_name)
    if k_name == "constant"; return ConstantKernel();
    elseif k_name == "linear"; return LinearKernel();
    elseif k_name == "matern12" || k_name == "exponential"; return Matern12Kernel();
    elseif k_name == "matern32"; return Matern32Kernel();
    elseif k_name == "matern52"; return Matern52Kernel();
    elseif k_name == "spherical"; return SphericalKernel();
    elseif k_name == "squared_exponential" || k_name == "se" || k_name == "gaussian" || k_name == "rbf"; return SqExponentialKernel();
    elseif k_name == "periodic"; return PeriodicKernel();
    else
        @warn "Kernel '$kernel_name' not recognized. Defaulting to SqExponentialKernel."
        return SqExponentialKernel()
    end
end

function _find_parameter_new(p_names, domain, var, param, k=nothing)
    # Purpose: Finds parameter names in the MCMC chain based on the new naming scheme.
    # Rationale: Provides a robust way to locate parameters, trying outcome-specific names first.
    # Inputs:
    #   - p_names: A vector of all parameter names from the chain.
    #   - domain, var, param: The components of the parameter name.
    #   - k: The optional outcome index for multivariate models.
    # Outputs: The full parameter name string, or an empty string if not found.
    base_name = "$(domain)_$(var)_$(param)"

    if !isnothing(k)
        specific_name = "$(base_name)_$(k)"
        if specific_name in p_names
            return specific_name
        end
    end

    if base_name in p_names
        return base_name
    end

    re_indexed = Regex("^" * escape_string(base_name) * "\\[")
    indexed_match = findfirst(n -> occursin(re_indexed, n), p_names)
    if !isnothing(indexed_match)
        return base_name
    end

    return ""
end

function get_params_vector(chain, base_name::String, expected_len::Int)
    # Purpose: Extracts all posterior samples for a given parameter into a matrix.
    # Rationale: Handles both scalar and vector parameters, correctly parsing indexed names.
    # Inputs:
    #   - chain: The MCMC chain object.
    #   - base_name: The base name of the parameter (e.g., "latent_spatial").
    #   - expected_len: The expected number of elements for this parameter.
    # Outputs: A matrix of size `[n_samples x expected_len]`.
    local N_samples = size(chain, 1)
    local all_names = string.(FlexiChains.parameters(chain))

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

    @warn "get_params_vector: Parameter '$base_name' not discovered in chain. Initializing with zeros (len=$expected_len)."
    return zeros(Float64, N_samples, expected_len)
end


# ==============================================================================
# SECTION 2: MANIFOLD-SPECIFIC EXTRACTION
# ==============================================================================

function extract_manifold(m_obj::Union{ICAR, Besag, RW1, RW2, Leroux, SAR, Cyclic, IID}, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    
    for k in 1:outcomes_N
        domain = string(spec.domain)
        var = string(spec.key)
        
        sigma_name = _find_parameter_new(p_names, domain, var, "sigma", k)
        latent_name = _find_parameter_new(p_names, domain, var, "latent", k)
        
        n_units = if domain == "spatial"; M.s_N; elseif domain == "temporal"; M.t_N; else M.u_N; end
        
        if isempty(sigma_name) || isempty(latent_name)
            @warn "Parameters for manifold $(spec.key) (domain $(domain), outcome $(k)) not found. Returning zero-matrix for effect."
            push!(structured_effects, zeros(Float64, n_units, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        latent_samples = get_params_vector(chain, latent_name, n_units)
        
        # Scale the latent field by sigma for each sample
        effect = latent_samples' .* sigma_samples' # [n_units x n_samples] with broadcasting
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
        var = string(spec.key)

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
        var = string(spec.key)

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


function extract_manifold(m_obj::Union{PSpline, BSpline, TPS, FFT, Wavelet, Moran, Spherical, ExponentialDecay, Barycentric}, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    basis_key = Symbol(spec.var)
    
    if !haskey(M.basis_matrices, basis_key)
        @warn "Basis matrix for smooth manifold $(basis_key) not found. Returning zero-matrices."
        return (structured=[zeros(Float64, M.y_N, n_samples)], noisy=[zeros(Float64, M.y_N, n_samples)], coefficients=[zeros(Float64, 1, n_samples)])
    end

    B_mat_train = M.basis_matrices[basis_key]
    B_mat_full = if !isnothing(PS) && haskey(PS, :basis_matrices) && haskey(PS.basis_matrices, basis_key)
        vcat(B_mat_train, PS.basis_matrices[basis_key])
    else
        B_mat_train
    end
    n_basis_cols = size(B_mat_full, 2)

    structured_effects = Vector{Matrix{Float64}}()
    coefficient_effects = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        domain = string(spec.domain)
        var = string(spec.key)
        
        coeffs_name = _find_parameter_new(p_names, domain, var, "coeffs", k)
        
        if isempty(coeffs_name)
            push!(structured_effects, zeros(Float64, size(B_mat_full, 1), n_samples))
            push!(coefficient_effects, zeros(Float64, n_basis_cols, n_samples))
            continue
        end

        coeffs = get_params_vector(chain, coeffs_name, n_basis_cols)
        
        push!(coefficient_effects, coeffs')

        effect = B_mat_full * coeffs'
        push!(structured_effects, effect)
    end

    return (structured=structured_effects, noisy=structured_effects, coefficients=coefficient_effects)
end

function extract_manifold(m_obj::Harmonic, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    
    n_harmonics = get(m_obj.params, :n_harmonics, 2)
    n_basis = 2 * n_harmonics
    period = m_obj.period
    
    u_idx_full = haskey(M, :u_idx) ? (isnothing(PS) || !haskey(PS, :u_idx) ? M.u_idx : vcat(M.u_idx, PS.u_idx)) : ones(Int, N_tot)
    B_harmonic_full = hcat([ (i % 2 == 1 ? sin : cos).(2pi * ceil(i/2) .* u_idx_full ./ period) for i in 1:n_basis]...)

    for k in 1:outcomes_N
        domain = string(spec.domain)
        var = string(spec.key)
        
        coeffs_name = _find_parameter_new(p_names, domain, var, "coeffs", k)
        
        if isempty(coeffs_name)
            @warn "Coefficient parameter for Harmonic manifold $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_tot, n_samples))
            continue
        end

        coeffs = get_params_vector(chain, coeffs_name, n_basis)
        effect = B_harmonic_full * coeffs'
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
    
    coord_vars = get(spec.params, :positional_args, [])
    coords_train = haskey(spec.params, :coords) ? spec.params.coords : Matrix{Float64}(M.data[!, Symbol.(coord_vars)])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        coords_train
    end
    n_obs_full = size(coords_full, 1)

    for k in 1:outcomes_N
        domain = string(spec.domain)
        var = string(spec.key)
        
        sigma_name = _find_parameter_new(p_names, domain, var, "sigma", k)
        ls_name = _find_parameter_new(p_names, domain, var, "ls", k)
        u_raw_name = _find_parameter_new(p_names, domain, var, "u_raw", k)
        f_innov_name = _find_parameter_new(p_names, domain, var, "f_innov", k)
        
        n_inducing = m_obj.n_inducing
        Z_inducing = spec.params.Z_inducing
        kernel_str = m_obj.kernel
        kernel = get_kernel_from_string(kernel_str)
        noise = get(M, :noise, 1e-6)

        if isempty(sigma_name) || isempty(ls_name) || isempty(u_raw_name) || isempty(f_innov_name)
            @warn "Parameters for GP manifold $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        ls_samples = get_params_vector(chain, ls_name, 1)
        u_raw_samples = get_params_vector(chain, u_raw_name, n_inducing)
        f_innov_samples = get_params_vector(chain, f_innov_name, n_obs_full)

        gp_effect_k = zeros(Float64, n_obs_full, n_samples)

        for j in 1:n_samples
            sigma_j = sigma_samples[j, 1]
            ls_j = ls_samples[j, 1]
            u_raw_j = u_raw_samples[j, :]
            f_innov_j = f_innov_samples[j, 1:n_obs_full]

            kernel_scaled = sigma_j^2 * (kernel ∘ ScaleTransform(1.0 / ls_j))
            
            K_uu = kernelmatrix(kernel_scaled, RowVecs(Z_inducing)) + noise * I
            K_uf = kernelmatrix(kernel_scaled, RowVecs(Z_inducing), RowVecs(coords_full))
            k_ff_diag = diag(kernelmatrix(kernel_scaled, RowVecs(coords_full)))

            L_uu = cholesky(Symmetric(K_uu)).L
            u_latent = L_uu * u_raw_j

            A = (L_uu') \ K_uf
            mean_f = A' * u_latent
            var_f = k_ff_diag - vec(sum(A.^2, dims=1))

            gp_effect_k[:, j] = mean_f + sqrt.(max.(var_f, 0.0) .+ noise) .* f_innov_j
        end
        push!(structured_effects, gp_effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end

function extract_manifold(m_obj::DynamicsManifold, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    model_type = m_obj.model
    key = string(spec.var)
    noise = get(M, :noise, 1e-6)

    for k in 1:outcomes_N
        domain = "dynamics"
        var = key

        if model_type == "advection" || model_type == "diffusion"
            param_name = model_type == "advection" ? "v" : "d"
            rate_name = _find_parameter_new(p_names, domain, key, param_name, k)
            sigma_name = _find_parameter_new(p_names, domain, var, "sigma", k)
            innov_name = _find_parameter_new(p_names, domain, var, "innov", k)

            if isempty(rate_name) || isempty(sigma_name) || isempty(innov_name)
                @warn "Parameters for Dynamics manifold $(key) (outcome $(k)) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_tot, n_samples))
                continue
            end

            rate_samples = get_params_vector(chain, rate_name, 1)[:, 1]
            sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
            innov_samples = get_params_vector(chain, innov_name, M.s_N * M.t_N)

            L = M.s_Q # Graph Laplacian
            I_s = I(M.s_N)
            
            s_idx_full = !isnothing(PS) ? vcat(M.s_idx, PS.s_idx) : M.s_idx
            t_idx_full = !isnothing(PS) ? vcat(M.t_idx, PS.t_idx) : M.t_idx

            effect_k = zeros(Float64, N_tot, n_samples)

            for j in 1:n_samples
                dyn_field = zeros(Float64, M.s_N, M.t_N)
                innov_matrix = reshape(innov_samples[j, :], M.s_N, M.t_N)
                
                # Propagator for the implicit Euler step: (I - v*dt*L)
                L_op = cholesky(Symmetric(I_s - rate_samples[j] * L + noise * I_s)).L

                # Initialize first time step
                dyn_field[:, 1] = innov_matrix[:, 1]

                # Evolve over time by solving the linear system at each step
                for t in 2:M.t_N
                    dyn_field[:, t] = L_op' \ (L_op \ dyn_field[:, t-1]) + innov_matrix[:, t]
                end
                
                dyn_field .*= sigma_samples[j]

                for i in 1:N_tot
                    effect_k[i, j] = dyn_field[s_idx_full[i], t_idx_full[i]]
                end
            end
            push!(structured_effects, effect_k)

        elseif model_type == "advection_diffusion" 
            v_name = _find_parameter_new(p_names, domain, key, "v", k) 
            d_name = _find_parameter_new(p_names, domain, key, "d", k) 
            sigma_name = _find_parameter_new(p_names, domain, key, "sigma", k) 
            innov_name = _find_parameter_new(p_names, domain, key, "innov", k) 
 
            if isempty(v_name) || isempty(d_name) || isempty(sigma_name) || isempty(innov_name) 
                @warn "Parameters for advection-diffusion manifold $(key) not found. Returning zero-matrix." 
                push!(structured_effects, zeros(Float64, N_tot, n_samples)); continue 
            end 
 
            v_samples = get_params_vector(chain, v_name, 1)[:, 1] 
            d_samples = get_params_vector(chain, d_name, 1)[:, 1] 
            sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1] 
            innov_samples = get_params_vector(chain, innov_name, M.s_N * M.t_N) 
 
            L = M.s_Q 
            I_s = I(M.s_N) 
            s_idx_full = !isnothing(PS) ? vcat(M.s_idx, PS.s_idx) : M.s_idx 
            t_idx_full = !isnothing(PS) ? vcat(M.t_idx, PS.t_idx) : M.t_idx 
            effect_k = zeros(Float64, N_tot, n_samples) 
            for j in 1:n_samples 
                dyn_field = zeros(Float64, M.s_N, M.t_N) 
                innov_matrix = reshape(innov_samples[j, :], M.s_N, M.t_N) 
                L_op = cholesky(Symmetric(I_s - v_samples[j] * L - d_samples[j] * L + noise * I_s)).L 
                dyn_field[:, 1] = innov_matrix[:, 1] 
                for t in 2:M.t_N 
                    dyn_field[:, t] = L_op' \ (L_op \ dyn_field[:, t-1]) + innov_matrix[:, t] 
                end 
                dyn_field .*= sigma_samples[j] 
                for i in 1:N_tot 
                    effect_k[i, j] = dyn_field[s_idx_full[i], t_idx_full[i]] 
                end 
            end 
            push!(structured_effects, effect_k) 

        elseif model_type == "gompertz" || model_type == "logistic_basic"
            r_name = _find_parameter_new(p_names, domain, key, "r", k)
            K_name = _find_parameter_new(p_names, domain, key, "K", k)

            if isempty(r_name) || isempty(K_name)
                @warn "Parameters for Dynamics manifold $(key) (outcome $(k)) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_tot, n_samples))
                continue
            end

            r_samples = get_params_vector(chain, r_name, 1)
            K_samples = get_params_vector(chain, K_name, 1)
            
            pop_state_samples = zeros(Float64, M.t_N, n_samples)
            for j in 1:n_samples
                pop_state_j = Vector{Float64}(undef, M.t_N)
                pop_state_j[1] = log(K_samples[j] / 2.0) # Start at half carrying capacity
                for t in 2:M.t_N
                    growth = r_samples[j] * (log(K_samples[j]) - pop_state_j[t-1])
                    pop_state_j[t] = pop_state_j[t-1] + growth
                end
                pop_state_samples[:, j] = pop_state_j
            end
            
            t_idx_full = !isnothing(PS) ? vcat(M.t_idx, PS.t_idx) : M.t_idx
            effect_k = zeros(Float64, N_tot, n_samples)
            for i in 1:N_tot
                effect_k[i, :] = pop_state_samples[t_idx_full[i], :]
            end
            push!(structured_effects, effect_k)
        end
    end
    return (structured=structured_effects, noisy=structured_effects)
end

function extract_manifold(m_obj::MixedManifold, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    lhs_terms = [strip(t) for t in split(m_obj.lhs, '+')]
    n_terms = length(lhs_terms)

    if n_terms == 1
        structured_effects = Vector{Matrix{Float64}}()
        for k in 1:outcomes_N
            domain = string(spec.domain)
            var = string(spec.key)
            coeffs_name = _find_parameter_new(p_names, domain, var, "coeffs", k)
            n_cat = spec.params.n_cat
            
            if isempty(coeffs_name)
                @warn "Parameters for Mixed manifold $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, n_cat, n_samples))
                continue
            end

            coeffs_samples = get_params_vector(chain, coeffs_name, n_cat)
            effect = coeffs_samples'
            push!(structured_effects, effect)
        end
        return (type=:simple, effects=structured_effects, lhs=m_obj.lhs, indices=spec.params.indices)
    else
        correlated_effects = Dict{Symbol, Vector{Matrix{Float64}}}()
        for k in 1:outcomes_N
            domain = string(spec.domain)
            var = string(spec.key)
            coeffs_name = _find_parameter_new(p_names, domain, var, "coeffs_correlated", k)
            n_cat = spec.params.n_cat

            if isempty(coeffs_name)
                @warn "Correlated coefficients for Mixed manifold $(spec.key) (outcome $(k)) not found. Returning zero-matrices."
                continue
            end

            coeffs_flat = get_params_vector(chain, coeffs_name, n_cat * n_terms)
            
            for (i, term) in enumerate(lhs_terms)
                term_name = term == "1" ? :intercept : Symbol("slope_$(term)")
                if !haskey(correlated_effects, term_name) correlated_effects[term_name] = [zeros(0,0) for _ in 1:outcomes_N] end
                term_coeffs = coeffs_flat[:, i:n_terms:end]
                correlated_effects[term_name][k] = term_coeffs'
            end
        end
        return (type=:correlated, effects=correlated_effects, lhs=m_obj.lhs, indices=spec.params.indices)
    end
end

function extract_manifold(m_obj::Eigen, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # Purpose: Reconstructs the effect of the Eigen (Bayesian PCA) manifold.
    # Rationale: This function reconstructs the principal component effects from the
    #            posterior samples of the Householder parameterization.
    # Inputs: Standard `extract_manifold` arguments.
    # Outputs: A NamedTuple with the reconstructed effect of all principal components.
    structured_effects = Vector{Matrix{Float64}}()
    key = string(spec.key)
    domain = string(spec.domain)
    var = string(spec.key)
    
    # Extract parameters from the chain
    v_raw_name = _find_parameter_new(p_names, domain, var, "v_raw", nothing)
    d_raw_name = _find_parameter_new(p_names, domain, var, "d_raw", nothing)
    pca_sds_name = _find_parameter_new(p_names, domain, var, "pca_sds", nothing)
    pdef_sds_name = _find_parameter_new(p_names, domain, var, "pdef_sds", nothing)
    factors_name = _find_parameter_new(p_names, domain, var, "factors", nothing)

    if isempty(v_raw_name) || isempty(d_raw_name) || isempty(pca_sds_name) || isempty(pdef_sds_name) || isempty(factors_name)
        @warn "Parameters for Eigen manifold $(key) not found. Returning zero-matrix."
        push!(structured_effects, zeros(Float64, N_tot, n_samples))
        return (structured=structured_effects, noisy=structured_effects)
    end

    n_vars = m_obj.n_vars
    n_factors = m_obj.n_factors
    ltri_indices = m_obj.ltri_indices

    v_raw_samples = get_params_vector(chain, v_raw_name, length(ltri_indices))
    d_raw_samples = get_params_vector(chain, d_raw_name, n_factors)
    pca_sds_samples = get_params_vector(chain, pca_sds_name, n_factors)
    pdef_sds_samples = get_params_vector(chain, pdef_sds_name, n_factors)
    factors_samples = get_params_vector(chain, factors_name, M.y_N * n_factors)

    total_effect = zeros(Float64, M.y_N, n_samples)

    for j in 1:n_samples
        # Reconstruct U (eigenvectors) from Householder reflectors
        v_mat_j = zeros(n_vars, n_factors)
        v_mat_j[ltri_indices] .= v_raw_samples[j, :]
        U_j = householder_to_eigenvector(v_mat_j, n_vars, n_factors)

        # Reconstruct D (eigenvalues)
        d_trans_j = exp.(d_raw_samples[j, :] .* pdef_sds_samples[j, :])
        D_mat_j = Diagonal(d_trans_j)

        # Reconstruct loadings matrix L
        L_j = U_j * D_mat_j

        # Reconstruct factors matrix F
        F_matrix_j = reshape(factors_samples[j, :], M.y_N, n_factors)

        # Calculate the final eigen effects
        eigen_effects_j = (F_matrix_j * L_j') .* pca_sds_samples[j, :]'
        
        # Sum effects across components
        total_effect[:, j] = sum(eigen_effects_j, dims=2)
    end

    if N_tot > M.y_N
        @warn "Prediction for Eigen manifold is not fully implemented and will be zero for out-of-sample points."
        total_effect = vcat(total_effect, zeros(Float64, N_tot - M.y_N, n_samples))
    end

    # The Eigen effect is univariate; it applies the same effect to all outcomes.
    # We replicate the single reconstructed effect for each outcome.
    for k in 1:outcomes_N
        push!(structured_effects, total_effect)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end

function extract_manifold(m_obj::ComposedManifold, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # Purpose: Reconstructs effects from composed manifolds, with special handling for Kronecker product (spatiotemporal) interactions.
    # Rationale: This function is the designated reconstruction point for effects that are generated by combining multiple manifolds,
    #            such as Knorr-Held spatiotemporal interactions. This logic mirrors the model generation in the assembler.
    # Inputs: Standard `extract_manifold` arguments.
    # Outputs: A NamedTuple with the reconstructed effects.
    op = m_obj.operator 
    key = string(spec.key)

    function _find_spec_by_obj(obj, specs)
        idx = findfirst(s -> s.manifold_obj === obj, specs)
        return isnothing(idx) ? nothing : specs[idx]
    end

    if op == :kronecker_product && haskey(M, :model_st) && M.model_st != "none"
        # This block handles the reconstruction of a global spatiotemporal interaction term.
        spatial_comp_obj = m_obj.components[1] # By convention
        temporal_comp_obj = m_obj.components[2]

        spatial_spec = find_spec_by_obj(spatial_comp_obj, M.manifolds)
        temporal_spec = find_spec_by_obj(temporal_comp_obj, M.manifolds)

        if isnothing(spatial_spec) || isnothing(temporal_spec)
            @warn "Could not resolve components for Kronecker product '$(key)'. ST effect reconstruction skipped."
            return (structured=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N], noisy=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N])
        end

        s_Q = spatial_spec.Q_template
        t_Q = temporal_spec.Q_template
        model_st = M.model_st
        noise = get(M, :noise, 1e-6)
        
        C_s = cholesky(Symmetric(Matrix(s_Q) + noise * I))
        C_t = cholesky(Symmetric(Matrix(t_Q) + noise * I))

        s_idx_full = !isnothing(PS) ? vcat(M.s_idx, PS.s_idx) : M.s_idx
        t_idx_full = !isnothing(PS) ? vcat(M.t_idx, PS.t_idx) : M.t_idx

        all_effects = [zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N]

        for k in 1:outcomes_N
            t_rho_samples = zeros(n_samples)
            if model_st == "IV" && temporal_spec.manifold_obj isa AR1
                t_rho_name = _find_parameter_new(p_names, "temporal", string(temporal_spec.key), "rho", k)
                if !isempty(t_rho_name); t_rho_samples = get_params_vector(chain, t_rho_name, 1)[:, 1]; end
            end

            local st_sigma_samples, st_raw_samples
            if outcomes_N > 1
                st_sigma_samples = get_params_vector(chain, "st_interaction_sigma", outcomes_N)[:, k]
                st_raw_flat = get_params_vector(chain, "st_interaction_raw_flat", M.s_N * M.t_N * outcomes_N)
                st_raw_samples = st_raw_flat[:, (k-1)*M.s_N*M.t_N+1 : k*M.s_N*M.t_N]
            else
                st_sigma_samples = get_params_vector(chain, "st_interaction_sigma", 1)[:, 1]
                st_raw_samples = get_params_vector(chain, "st_interaction_raw", M.s_N * M.t_N)
            end

            st_effect_k = zeros(Float64, N_tot, n_samples)

            if model_st == "I"
                for j in 1:n_samples
                    st_innov_matrix = reshape(st_raw_samples[j, :], M.s_N, M.t_N)
                    st_inter = st_innov_matrix .* st_sigma_samples[j]
                    for i in 1:N_tot; st_effect_k[i, j] = st_inter[s_idx_full[i], t_idx_full[i]]; end
                end
            elseif model_st == "II"
                for j in 1:n_samples
                    st_innov_matrix = reshape(st_raw_samples[j, :], M.s_N, M.t_N)
                    st_inter = (C_t.U \ st_innov_matrix')' .* st_sigma_samples[j]
                    for i in 1:N_tot; st_effect_k[i, j] = st_inter[s_idx_full[i], t_idx_full[i]]; end
                end
            elseif model_st == "III"
                for j in 1:n_samples
                    st_innov_matrix = reshape(st_raw_samples[j, :], M.s_N, M.t_N)
                    st_inter = (C_s.U \ st_innov_matrix) .* st_sigma_samples[j]
                    for i in 1:N_tot; st_effect_k[i, j] = st_inter[s_idx_full[i], t_idx_full[i]]; end
                end
            elseif model_st == "IV"
                for j in 1:n_samples
                    st_innov_matrix = reshape(st_raw_samples[j, :], M.s_N, M.t_N)
                    st_inter = zeros(Float64, M.s_N, M.t_N)
                    t_rho_j = t_rho_samples[j]
                    if abs(t_rho_j) < 1.0
                        st_inter[:, 1] = (C_s.U \ st_innov_matrix[:, 1]) ./ sqrt(1.0 - t_rho_j^2 + noise)
                        for t in 2:M.t_N; st_inter[:, t] = t_rho_j .* st_inter[:, t-1] .+ (C_s.U \ st_innov_matrix[:, t]); end
                        st_inter .*= st_sigma_samples[j]
                    else # Fallback
                        st_inter = (C_s.U \ st_innov_matrix) .* st_sigma_samples[j]
                    end
                    for i in 1:N_tot; st_effect_k[i, j] = st_inter[s_idx_full[i], t_idx_full[i]]; end
                end
            end
            all_effects[k] = st_effect_k
        end
        return (structured=all_effects, noisy=all_effects)

    elseif op == :pipe
        # Handles state-space models like `spatial |> smooth(time)` where the coefficients
        # of the dynamic manifold are themselves structured by the state manifold.
        if length(m_obj.components) != 2; error("Pipe operator reconstruction requires exactly two components: state |> dynamic."); end

        state_manifold_obj = m_obj.components[1] 
        dynamic_manifold_obj = get(spec.params, :dynamic_manifold_obj, nothing)

        if isnothing(dynamic_manifold_obj)
            @warn "Could not resolve dynamic manifold for piped manifold '$(key)'. Skipping."
            return (structured=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N], noisy=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N])
        end

        basis_key = get(spec.params, :dynamic_basis_key, nothing)
        if isnothing(basis_key) || !haskey(M.basis_matrices, basis_key)
            @warn "Could not find basis matrix for dynamic component of piped manifold '$(key)'. Skipping."
            return (structured=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N], noisy=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N])
        end

        B_dynamic_train = M.basis_matrices[basis_key]
        B_dynamic_full = if !isnothing(PS) && haskey(PS, :basis_matrices) && haskey(PS.basis_matrices, basis_key)
            vcat(B_dynamic_train, PS.basis_matrices[basis_key])
        else
            B_dynamic_train
        end
        n_basis = size(B_dynamic_full, 2)
        n_spatial = M.s_N

        all_effects = [zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N]

        for k in 1:outcomes_N
            sigma_name = _find_parameter_new(p_names, "interact", key, "sigma", k)
            rho_name = _find_parameter_new(p_names, "interact", key, "rho", k)
            coeffs_raw_name = _find_parameter_new(p_names, "interact", key, "coeffs_raw", k)

            if isempty(sigma_name) || isempty(coeffs_raw_name); continue; end

            sigma_samples = get_params_vector(chain, sigma_name, 1)
            rho_samples = hasproperty(state_manifold_obj, :rho_prior) ? get_params_vector(chain, rho_name, 1) : nothing
            coeffs_raw_samples = get_params_vector(chain, coeffs_raw_name, n_spatial * n_basis)

            Q_spatial_template = spec.hyper.state_spec.Q_template
            state_m_type = spec.hyper.state_spec.model_type
            s_idx_full = !isnothing(PS) ? vcat(M.s_idx, PS.s_idx) : M.s_idx

            for j in 1:n_samples
                rho_val = isnothing(rho_samples) ? nothing : rho_samples[j, 1]
                Q_spatial = recompose_precision(state_m_type, Q_spatial_template, 1.0; extra_param=rho_val)
                F_spatial = cholesky(Symmetric(Q_spatial + 1e-6 * I))

                coeffs_raw_matrix = reshape(coeffs_raw_samples[j, :], n_spatial, n_basis)
                spatial_coeffs = sigma_samples[j, 1] .* (F_spatial.U \ coeffs_raw_matrix)

                all_effects[k][:, j] = sum(B_dynamic_full .* spatial_coeffs[s_idx_full, :], dims=2)
            end
        end
        return (structured=all_effects, noisy=all_effects)

    else
        @warn "Reconstruction for ComposedManifold with operator ':$op' is not implemented. Returning zero-effect for '$(key)'."
        return (structured=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N], noisy=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N])
    end
end


function extract_manifold(m_obj::CustomManifold, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    reconstruct_func = get(m_obj.params, :reconstruct_func, nothing)

    if !isnothing(reconstruct_func) && isa(reconstruct_func, Function)
        try
            return reconstruct_func(chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
        catch e
            @error "The custom reconstruction function for manifold '$(spec.key)' failed."
            rethrow(e)
        end
    else
        @warn "Reconstruction for custom manifold '$(spec.key)' is not defined. Returning a zero-effect. Please provide a `reconstruct_func` to the `custom()` module to enable posterior reconstruction."
        structured_effects = [zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N]
        return (structured=structured_effects, noisy=structured_effects)
    end
end

function extract_manifold(m_obj::Manifold, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    @warn "No specific reconstruction logic for manifold type $(typeof(m_obj)). Returning zero effects."
    n_units = if spec.domain == :spatial; M.s_N; elseif spec.domain == :temporal; M.t_N; else 1; end
    return (structured=[zeros(Float64, n_units, n_samples)], noisy=[zeros(Float64, n_units, n_samples)])
end


function _apply_multivariate_correlation(eta_latent, chain, outcomes_N)
    # Purpose: Applies the estimated correlation structure to independent latent fields.
    # Rationale: Centralizes the core logic of multivariate models, where independent
    #            latent effects are combined via a learned correlation matrix.
    # Inputs:
    #   - eta_latent: A 3D array of un-correlated effects [n_obs, n_samples, n_outcomes].
    #   - chain: The MCMC chain, to extract the correlation matrix.
    #   - outcomes_N: The number of outcomes.
    # Outputs: A 3D array of correlated effects.
    if outcomes_N == 1
        return eta_latent
    end
    N_tot, n_samples, _ = size(eta_latent)
    L_corr_samples = get_params_vector(chain, "L_corr", outcomes_N * outcomes_N)
    eta_final = zeros(N_tot, n_samples, outcomes_N)
    for s in 1:n_samples
        L_s = reshape(L_corr_samples[s, :], outcomes_N, outcomes_N)
        eta_final[:, s, :] = eta_latent[:, s, :] * L_s'
    end
    return eta_final
end

function _summarize_effects_registry(registry, M, outcomes_N, alpha)
    # Purpose: Summarizes the posterior samples for all discovered manifold effects.
    # Rationale: Consolidates the logic for summarizing simple, mixed, and multivariate
    #            effects into a single, reusable function.
    # Inputs:
    #   - registry: The dictionary of raw posterior effects.
    #   - M: The model configuration object.
    #   - outcomes_N: The number of outcomes.
    #   - alpha: The significance level for credible intervals.
    # Outputs: A NamedTuple containing summarized effects.
    summarized_registry = Dict{Symbol, Any}()
    mixed_effects_summaries = Dict{Symbol, Any}()

    for (key, effects) in pairs(registry)
        if key in [:intercept, :fixed]; continue; end

        spec_idx = findfirst(s -> s.key == key, M.manifolds)
        if !isnothing(spec_idx) && M.manifolds[spec_idx].manifold_obj isa MixedManifold
            summaries_per_outcome = [Dict{Symbol, Any}() for _ in 1:outcomes_N]
            if effects.type == :simple
                for k in 1:outcomes_N
                    summaries_per_outcome[k][Symbol(effects.lhs)] = summarize_array(effects.effects[k], alpha=alpha)
                end
            elseif effects.type == :correlated
                for (term_name, term_effects) in pairs(effects.effects)
                    for k in 1:outcomes_N
                        summaries_per_outcome[k][term_name] = summarize_array(term_effects[k], alpha=alpha)
                    end
                end
            end
            
            summaries_final = outcomes_N > 1 ? [NamedTuple(s) for s in summaries_per_outcome] : NamedTuple(summaries_per_outcome[1])
            mixed_effects_summaries[key] = (group_var=M.manifolds[spec_idx].var, summaries=summaries_final)
        else
            effect_set = hasproperty(effects, :noisy) ? effects.noisy : effects.structured
            if outcomes_N > 1
                summarized_registry[key] = [summarize_array(effect_set[k], alpha=alpha) for k in 1:outcomes_N]
            else
                summarized_registry[key] = summarize_array(effect_set[1], alpha=alpha)
            end
        end
    end
    if !isempty(mixed_effects_summaries); summarized_registry[:mixed_effects] = NamedTuple(mixed_effects_summaries); end
    
    return NamedTuple(summarized_registry)
end

# ==============================================================================
# SECTION 3: CORE RECONSTRUCTION WORKFLOW
# ==============================================================================

function _reconstruct(arch::UnivariateArchitecture, mode::String, chain, M, PS, alpha)
    # Purpose: Main reconstruction entry point for univariate models.
    # Rationale: Orchestrates the discovery, assembly, and summarization of all model effects.
    # Inputs: Standard reconstruction arguments for a univariate model.
    # Outputs: A comprehensive NamedTuple with all summarized posterior statistics.
    n_samples = size(chain, 1)
    p_names = string.(names(chain))
    N_tot = isnothing(PS) ? M.y_N : M.y_N + PS.y_N

    registry = _discover_manifold_realizations(chain, M, PS, n_samples, p_names, 1, N_tot)
    eta_latent = _modular_eta_assembly(registry, M, PS, n_samples, 1)
    eta_final = eta_latent[:,:,1] # Drop the third dimension

    pred_results = _process_ll_and_predictions(eta_final, chain, M, PS, 1, 1)
    
    summarized_effects = _summarize_effects_registry(registry, M, 1, alpha)
    
    p_denoised_summary = summarize_array(pred_results.p_denoised, alpha=alpha)
    p_noisy_summary = summarize_array(pred_results.p_noisy, alpha=alpha)
    waic = _compute_waic(pred_results.log_lik)

    return (
        predictions_denoised = p_denoised_summary,
        predictions_noisy = p_noisy_summary,
        raw_predictions_denoised = pred_results.p_denoised,
        raw_predictions_noisy = pred_results.p_noisy,
        log_likelihood = pred_results.log_lik,
        waic = waic,
        effects = summarized_effects,
        arch = arch
    )
end

function _discover_manifold_realizations(chain, M, PS, n_samples, p_names, outcomes_N, N_tot)
    # Purpose: Extracts all latent effects from the MCMC chain.
    # Rationale: Iterates through all specified manifolds and fixed effects, calling the appropriate
    #            extraction function for each to populate a central registry of posterior samples.
    # Inputs: Standard reconstruction arguments.
    # Outputs: A NamedTuple registry containing posterior samples for each model component.
    registry = Dict{Symbol, Any}()

    # Fixed effects
    if M.Xfixed_N > 0
        Xfixed_train = M.Xfixed
        Xfixed_pred = if isnothing(PS) || !haskey(PS, :Xfixed) || isempty(PS.Xfixed)
            zeros(0, M.Xfixed_N)
        else
            PS.Xfixed
        end
        Xfixed_full = vcat(Xfixed_train, Xfixed_pred)
        
        if outcomes_N > 1
            beta_samples_flat = get_params_vector(chain, "Xfixed_beta", M.Xfixed_N * outcomes_N)
            fixed_effects_all = zeros(N_tot, n_samples, outcomes_N)
            for k in 1:outcomes_N
                beta_k = beta_samples_flat[:, (k-1)*M.Xfixed_N+1 : k*M.Xfixed_N]
                fixed_effects_all[:, :, k] = Xfixed_full * beta_k'
            end
            registry[:fixed] = fixed_effects_all
        else
            beta_samples = get_params_vector(chain, "Xfixed_beta", M.Xfixed_N)
            registry[:fixed] = Xfixed_full * beta_samples'
        end
    else
        registry[:fixed] = zeros(N_tot, n_samples, outcomes_N)
    end

    # Intercept
    if M.add_intercept
        intercept_samples = get_params_vector(chain, "intercept", outcomes_N)
        intercept_effects = zeros(N_tot, n_samples, outcomes_N)
        for k in 1:outcomes_N
            intercept_effects[:, :, k] .= intercept_samples[:, k]'
        end
        registry[:intercept] = intercept_effects
    else
        registry[:intercept] = zeros(N_tot, n_samples, outcomes_N)
    end

    # Manifolds
    for spec in M.manifolds
        effects = extract_manifold(spec.manifold_obj, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
        registry[spec.key] = effects
    end

    return NamedTuple(registry)
end

function _modular_eta_assembly(registry, M, PS, n_samples, outcomes_N)
    # Purpose: Assembles the full linear predictor (eta) from all discovered latent effects.
    # Rationale: Mirrors the model's additive structure, combining all components on the link scale.
    # Inputs: The registry of effects and model configuration.
    # Outputs: A 3D array of eta samples `[n_obs, n_samples, n_outcomes]`.
    N_tot = isnothing(PS) ? M.y_N : M.y_N + PS.y_N
    eta_latent = zeros(Float64, N_tot, n_samples, outcomes_N)

    eta_latent .+= registry.intercept
    eta_latent .+= registry.fixed

    s_idx_full = haskey(M, :s_idx) ? (isnothing(PS) || !haskey(PS, :s_idx) ? M.s_idx : vcat(M.s_idx, PS.s_idx)) : ones(Int, N_tot)
    t_idx_full = haskey(M, :t_idx) ? (isnothing(PS) || !haskey(PS, :t_idx) ? M.t_idx : vcat(M.t_idx, PS.t_idx)) : ones(Int, N_tot)
    u_idx_full = haskey(M, :u_idx) ? (isnothing(PS) || !haskey(PS, :u_idx) ? M.u_idx : vcat(M.u_idx, PS.u_idx)) : ones(Int, N_tot)

    for spec in M.manifolds
        key = spec.key
        if !haskey(registry, key); continue; end
        
        effects = registry[key]
        effect_set = hasproperty(effects, :noisy) ? effects.noisy : effects.structured
        if isempty(effect_set); continue; end

        for k in 1:outcomes_N
            if spec.domain in [:spatial, :temporal]
                effect_to_add = effect_set[k]
                idx_vec = spec.domain == :spatial ? s_idx_full : t_idx_full
                eta_latent[:, :, k] .+= effect_to_add[idx_vec, :]
            elseif spec.domain == :seasonal
                effect_to_add = effect_set[k]
                if spec.manifold_obj isa Harmonic # Harmonic basis is already expanded to N_tot
                    eta_latent[:, :, k] .+= effect_to_add
                else # Assumes Cyclic or other GMRF-like seasonal model
                    idx_vec = u_idx_full
                    eta_latent[:, :, k] .+= effect_to_add[idx_vec, :]
                end
            elseif spec.domain == :smooth || spec.domain == :interact
                eta_latent[:, :, k] .+= effect_set[k]
            elseif spec.domain == :mixed
                group_var_sym = Symbol(spec.var)
                train_indices = effects.indices
                idx_full = if isnothing(PS); train_indices;
                else; train_levels = unique(M.data[!, group_var_sym]); pred_levels = hasproperty(PS.data, group_var_sym) ? unique(PS.data[!, group_var_sym]) : []; all_levels = unique(vcat(train_levels, pred_levels)); level_map = Dict(v => i for (i, v) in enumerate(all_levels)); pred_indices = hasproperty(PS.data, group_var_sym) ? [level_map[v] for v in PS.data[!, group_var_sym]] : Int[]; vcat(train_indices, pred_indices);
                end

                if effects.type == :simple
                    effect_to_add = effects.effects[k]
                    if effects.lhs == "1"; eta_latent[:, :, k] .+= effect_to_add[idx_full, :]; else; cov_vec = isnothing(PS) ? M.data[!, Symbol(effects.lhs)] : vcat(M.data[!, Symbol(effects.lhs)], PS.data[!, Symbol(effects.lhs)]); eta_latent[:, :, k] .+= effect_to_add[idx_full, :] .* cov_vec; end
                elseif effects.type == :correlated
                    for (term_name, term_effects) in pairs(effects.effects)
                        effect_to_add = term_effects[k]
                        if term_name == :intercept; eta_latent[:, :, k] .+= effect_to_add[idx_full, :]; else; cov_name = Symbol(replace(string(term_name), "slope_" => "")); cov_vec = isnothing(PS) ? M.data[!, cov_name] : vcat(M.data[!, cov_name], PS.data[!, cov_name]); eta_latent[:, :, k] .+= effect_to_add[idx_full, :] .* cov_vec; end
                    end
                end
            else
                if size(effect_set[k], 1) == N_tot
                    eta_latent[:, :, k] .+= effect_set[k]
                end
            end
        end
    end

    if haskey(M, :log_offset)
        offset_full = isnothing(PS) ? M.log_offset : vcat(M.log_offset, get(PS, :log_offset, zeros(PS.y_N)))
        for k in 1:outcomes_N
            eta_latent[:, :, k] .+= offset_full
        end
    end

    return eta_latent
end

function _process_ll_and_predictions(eta_samples, chain, M, PS, outcomes_N, k)
    # Purpose: Generates denoised predictions, noisy predictions, and log-likelihood values from eta.
    # Rationale: This function applies the inverse link function and samples from the predictive distribution.
    # Inputs: Eta samples and model configuration.
    # Outputs: A NamedTuple with denoised predictions, noisy predictions, and log-likelihood matrix.
    n_samples = size(eta_samples, 2)
    N_train = M.y_N
    N_pred = isnothing(PS) ? 0 : PS.y_N
    N_tot = N_train + N_pred

    y_obs_k = outcomes_N > 1 ? M.y_obs[:, k] : M.y_obs
    
    lik_spec = M.likelihood_specs[k]
    family = string(get(lik_spec, :family, "gaussian"))
    use_zi = get(M, :use_zi, false)
    phi_zi_samples = use_zi ? get_params_vector(chain, "lik_phi_zi", 1)[:,1] : zeros(n_samples)
    
    # Denoised predictions (on response scale)
    p_denoised_samples = similar(eta_samples)
    for s in 1:n_samples
        p_denoised_samples[:, s] = _apply_link_and_lik(family, eta_samples[:, s], use_zi, phi_zi_samples[s])
    end

    p_noisy_samples = similar(eta_samples)
    log_lik_samples = zeros(Float64, N_train, n_samples)

    # Get likelihood-specific parameters
    y_sigma_samples = get_params_vector(chain, "y_sigma", outcomes_N)
    r_nb_samples = get_params_vector(chain, "r_nb", outcomes_N)
    
    trials_full = haskey(M, :trials) ? (isnothing(PS) ? M.trials : vcat(M.trials, get(PS, :trials, ones(Int, PS.y_N)))) : ones(Int, N_tot)
    
    family_trait = get_model_family(family)

    for s in 1:n_samples
        phi_zi_s = phi_zi_samples[s]
        y_sigma_s = y_sigma_samples[s, k]
        r_nb_s = r_nb_samples[s, k]

        for i in 1:N_tot
            eta_is = eta_samples[i, s]
            
            # For sampling, y_obs in lik_obj doesn't matter.
            lik_obj = bstm_Likelihood(family, [0.0]; phi_zi=phi_zi_s, r_nb=r_nb_s, sigma_y=y_sigma_s, trial=trials_full[i])
            dist = get_dist_ref(lik_obj.family, lik_obj, eta_is, y_sigma_s)
            
            p_noisy_samples[i, s] = rand(dist)

            if i <= N_train
                log_lik_samples[i, s] = logpdf(dist, y_obs_k[i])
            end
        end
    end

    return (p_denoised = p_denoised_samples, p_noisy = p_noisy_samples, log_lik = log_lik_samples)
end

function _reconstruct(arch::MultivariateArchitecture, mode::String, chain, M, PS, alpha)
    # Purpose: Main reconstruction entry point for multivariate models.
    # Rationale: Handles the additional complexity of multiple outcomes and their correlations.
    # Inputs: Standard reconstruction arguments for a multivariate model.
    # Outputs: A comprehensive NamedTuple with all summarized posterior statistics.
    n_samples = size(chain, 1)
    p_names = string.(names(chain))
    outcomes_N = M.outcomes_N
    N_tot = isnothing(PS) ? M.y_N : M.y_N + PS.y_N

    registry = _discover_manifold_realizations(chain, M, PS, n_samples, p_names, outcomes_N, N_tot)
    eta_latent = _modular_eta_assembly(registry, M, PS, n_samples, outcomes_N)
    eta_final = _apply_multivariate_correlation(eta_latent, chain, outcomes_N)

    all_pred_results = [_process_ll_and_predictions(eta_final[:,:,k], chain, M, PS, outcomes_N, k) for k in 1:outcomes_N]
    
    summarized_effects = _summarize_effects_registry(registry, M, outcomes_N, alpha)

    p_denoised_summaries = [summarize_array(res.p_denoised, alpha=alpha) for res in all_pred_results]
    p_noisy_summaries = [summarize_array(res.p_noisy, alpha=alpha) for res in all_pred_results]
    
    all_log_lik = hcat([res.log_lik for res in all_pred_results]...)
    waic = _compute_waic(all_log_lik)

    return (
        predictions_denoised = p_denoised_summaries,
        predictions_noisy = p_noisy_summaries,
        raw_predictions_denoised = [res.p_denoised for res in all_pred_results],
        raw_predictions_noisy = [res.p_noisy for res in all_pred_results],
        log_likelihood = all_log_lik,
        waic = waic,
        effects = NamedTuple(summarized_effects),
        arch = arch
    )
end

function _reconstruct(arch::MultifidelityArchitecture, mode::String, chain, M, PS, alpha)
    # Purpose: Main reconstruction entry point for multi-fidelity models.
    # Rationale: Handles the hierarchical reconstruction of a main model and its nested sub-models.
    # Inputs: Standard reconstruction arguments for a multi-fidelity model.
    # Outputs: A comprehensive NamedTuple with all summarized posterior statistics for the main model and its sub-models.
    n_samples = size(chain, 1)
    p_names = string.(names(chain))
    N_tot = isnothing(PS) ? M.y_N : M.y_N + PS.y_N
    outcomes_N = M.outcomes_N

    # 1. Reconstruct the main model's components (excluding nested effects)
    main_registry = _discover_manifold_realizations(chain, M, PS, n_samples, p_names, outcomes_N, N_tot)
    
    # 2. Assemble the main model's base eta
    eta_main = _modular_eta_assembly(main_registry, M, PS, n_samples, outcomes_N)

    # 3. Reconstruct sub-models' etas and add them to the main eta
    nested_results = Dict{Symbol, Any}()
    if haskey(M, :nested_manifolds)
        for (key, sub_M) in M.nested_manifolds
            sub_PS = if !isnothing(PS) && haskey(PS, :nested_prediction_sets)
                get(PS.nested_prediction_sets, key, nothing)
            else
                nothing
            end

            sub_outcomes_N = get(sub_M, :outcomes_N, 1)
            sub_N_tot = isnothing(sub_PS) ? sub_M.y_N : sub_M.y_N + sub_PS.y_N
            sub_registry = _discover_manifold_realizations(chain, sub_M, sub_PS, n_samples, p_names, sub_outcomes_N, sub_N_tot)
            eta_sub = _modular_eta_assembly(sub_registry, sub_M, sub_PS, n_samples, sub_outcomes_N)

            rho_name = "nested_$(key)_rho"
            rho_samples = get_params_vector(chain, rho_name, 1)[:, 1]

            if size(eta_sub, 1) != N_tot
                @warn "Size mismatch between main model observations ($N_tot) and nested model '$(key)' observations ($(size(eta_sub, 1))). Cannot apply nested effect."
                continue
            end
            
            if outcomes_N > 1 || sub_outcomes_N > 1
                @warn "Multi-fidelity connection between multivariate models is not fully supported. Assuming a 1-to-1 outcome mapping."
            end
            
            eta_main .+= reshape(rho_samples, 1, n_samples, 1) .* eta_sub

            sub_arch_raw = get(sub_M, :model_arch, "univariate")
            sub_arch_type = sub_arch_raw == "multivariate" ? MultivariateArchitecture() : UnivariateArchitecture()
            nested_results[key] = _reconstruct(sub_arch_type, mode, chain, sub_M, sub_PS, alpha)
        end
    end

    # 4. Apply correlation and generate predictions
    eta_final = _apply_multivariate_correlation(eta_main, chain, outcomes_N)

    if outcomes_N > 1
        all_pred_results = [_process_ll_and_predictions(eta_final[:,:,k], chain, M, PS, outcomes_N, k) for k in 1:outcomes_N]
        p_denoised_summaries = [summarize_array(res.p_denoised, alpha=alpha) for res in all_pred_results]
        p_noisy_summaries = [summarize_array(res.p_noisy, alpha=alpha) for res in all_pred_results]
        raw_denoised = [res.p_denoised for res in all_pred_results]
        raw_noisy = [res.p_noisy for res in all_pred_results]
        all_log_lik = hcat([res.log_lik for res in all_pred_results]...)
    else
        pred_results = _process_ll_and_predictions(eta_final[:,:,1], chain, M, PS, 1, 1)
        p_denoised_summaries = summarize_array(pred_results.p_denoised, alpha=alpha)
        p_noisy_summaries = summarize_array(pred_results.p_noisy, alpha=alpha)
        raw_denoised = pred_results.p_denoised
        raw_noisy = pred_results.p_noisy
        all_log_lik = pred_results.log_lik
    end

    summarized_effects = _summarize_effects_registry(main_registry, M, outcomes_N, alpha)
    waic = _compute_waic(all_log_lik)

    return (
        predictions_denoised = p_denoised_summaries, 
        predictions_noisy = p_noisy_summaries, 
        raw_predictions_denoised = raw_denoised,
        raw_predictions_noisy = raw_noisy,
        log_likelihood = all_log_lik, 
        waic = waic, 
        effects = summarized_effects, 
        nested_results = nested_results, 
        arch = arch
    )
end


# ==============================================================================
# SECTION 3: POSTERIOR ASSEMBLY AND SUMMARIZATION
# ==============================================================================

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
    else
        mu = eta
    end
    if use_zi
        mu = (1.0 .- phi) .* mu
    end
    return mu
end


# ==============================================================================
# SECTION 4: MAIN RECONSTRUCTION AND PREDICTION API
# ==============================================================================

function model_results_comprehensive(model::DynamicPPL.Model, chain; au=nothing, data=nothing, alpha=0.05)
    # Purpose: The primary post-processing engine that generates comprehensive summaries,
    #          diagnostics, and plots from a fitted bstm model and MCMC chain.
    # Rationale: This function orchestrates the entire reconstruction workflow, from latent
    #            field discovery to metric calculation and visualization, providing a unified
    #            and standardized output for model assessment.
    # Inputs:
    #   - model: The fitted Turing model object.
    #   - chain: The MCMC chain result.
    #   - au: (Optional) Areal unit object containing spatial geometries for plotting.
    #   - data: (Optional) The original DataFrame, used for plotting covariate effects.
    #   - alpha: The significance level for credible intervals.
    # Outputs: A comprehensive NamedTuple with `:metrics`, `:pstats` (posterior stats), and `:plots`.

    # #
    # 1. Metadata and Architecture Extraction
    M = model.args.M
    y_obs = M.y_obs
    raw_arch = get(M, :model_arch, "univariate")

    arch_type = if raw_arch == "multivariate"; MultivariateArchitecture()
    elseif raw_arch == "multifidelity"; MultifidelityArchitecture()
    else; UnivariateArchitecture(); end

    # #
    # 2. Core Reconstruction
    # This calls the appropriate _reconstruct method based on the model architecture.
    res = _reconstruct(arch_type, "model_results", chain, M, nothing, alpha)

    # #
    # 2.5 Post-Stratification Weight Calculation
    # This is done here because we need the raw denoised prediction samples, which are
    # returned by _reconstruct but not typically stored in the final summary.
    post_strat_weights = nothing
    if hasproperty(res, :raw_predictions_denoised)
        samples_denoised = res.arch isa MultivariateArchitecture ? res.raw_predictions_denoised[1] : res.raw_predictions_denoised
        post_strat_weights = post_stratification_weights(res, M, nothing, samples_denoised)
    end

    # #
    # 3. Performance Metric Calculation
    # Handles both univariate and multivariate cases for RMSE and Pearson R.
    pred_summary = res.predictions_denoised
    y_pred = pred_summary isa AbstractVector ? vcat([ps.mean for ps in pred_summary]...) : (hasproperty(pred_summary, :mean) ? pred_summary.mean : [])
    y_obs_flat = vec(collect(y_obs))
    y_pred_flat = vec(collect(y_pred))
    valid_idx = findall(x -> !isnan(x) && !isnothing(x), y_obs_flat)

    rmse_val = 0.0
    r_pearson = 0.0
    if !isempty(valid_idx)
        obs_v = y_obs_flat[valid_idx]
        pred_v = y_pred_flat[valid_idx]
        rmse_val = sqrt(mean((obs_v .- pred_v).^2))
        try; r_pearson = cor(obs_v, pred_v); catch; r_pearson = 0.0; end
    end

    # #
    # 4. MCMC Diagnostics
    mean_rhat, min_ess, sampling_time = 1.0, 0.0, 0.0
    try
        chains_obj = MCMCChains.Chains(chain)
        df_stats = DataFrame(MCMCChains.summarize(chains_obj))
        if hasproperty(df_stats, :rhat); r_vals = filter(x -> !isnan(x) && x > 0, df_stats.rhat); mean_rhat = isempty(r_vals) ? 1.0 : mean(r_vals); end
        e_col = hasproperty(df_stats, :ess_bulk) ? :ess_bulk : (hasproperty(df_stats, :ess) ? :ess : nothing)
        if !isnothing(e_col); e_vals = filter(x -> !isnan(x) && x >= 0, df_stats[!, e_col]); min_ess = isempty(e_vals) ? 0.0 : minimum(e_vals); end
        if hasproperty(chain, :info) && haskey(chain.info, :stop_time); sampling_time = (chain.info.stop_time - chain.info.start_time); end
    catch e; @warn "MCMC diagnostic extraction failed: $e. Using default values."; end

    # #
    # 5. Plot Generation
    data_for_plots = isnothing(data) ? get(M, :data, nothing) : data
    plots = _generate_plots(res, M; au=au, data=data_for_plots)

    return (
        metrics = (rmse = rmse_val, r_pearson = r_pearson, ess = min_ess, rhat = mean_rhat, waic = get(res, :waic, 0.0), time = sampling_time),
        pstats = res,
        plots = plots,
        post_strat_weights = post_strat_weights
    )
end

function _generate_plots(res, M; au=nothing, data=nothing, outcome=1)
    # Purpose: Generates a standard set of diagnostic and summary plots from the
    #          reconstructed posterior results.
    # Rationale: Centralizes visualization logic, providing a consistent visual output
    #            for different model architectures and components.
    # Inputs:
    #   - res: The main results object from `_reconstruct`.
    #   - M: The model configuration object.
    #   - au: (Optional) Areal unit object with geometries.
    #   - data: (Optional) The original DataFrame for covariate plots.
    #   - outcome: The index of the outcome to plot for multivariate models.
    # Outputs: A dictionary of Plots.jl plot objects.

    plots = Dict{Symbol, Any}()
    effects = res.effects
    
    y_obs = get(M, :y_obs, nothing)
    polygons = isnothing(au) ? nothing : get(au, :polygons, nothing)
    centroids = isnothing(au) ? nothing : get(au, :centroids, nothing)

    if hasproperty(res, :predictions_denoised)
        if isnothing(y_obs); @info "Skipping PPC plot: Observation data not found.";
        else
            is_mv = res.arch isa MultivariateArchitecture
            pred_summary = is_mv ? res.predictions_denoised[outcome] : res.predictions_denoised
            if !isnothing(pred_summary) && hasproperty(pred_summary, :mean)
                y_p, y_o = vec(pred_summary.mean), is_mv ? vec(y_obs[:, outcome]) : vec(y_obs)
                if length(y_p) == length(y_o)
                    p_ppc = scatter(y_p, y_o, title="Posterior Predictive Check", xlabel="Predicted", ylabel="Observed", alpha=0.5, markersize=3, markerstrokewidth=0, legend=false)
                    clean_p, clean_o = filter(!isnan, y_p), filter(!isnan, y_o)
                    if !isempty(clean_p) && !isempty(clean_o); min_val, max_val = min(minimum(clean_p), minimum(clean_o)), max(maximum(clean_p), maximum(clean_o)); plot!(p_ppc, [min_val, max_val], [min_val, max_val], color=:red, ls=:dash, lw=1.5); end
                    plots[:ppc] = p_ppc
                end
            end
        end
    end

    function _create_choropleth_plot(field_data, title_str, polygons, centroids)
        if isnothing(field_data) || !hasproperty(field_data, :mean); @info "Skipping spatial plot '$title_str': Data missing."; return nothing; end
        if isnothing(polygons) && isnothing(centroids); @info "Skipping spatial plot '$title_str': No geometry provided."; return nothing; end
        s_mean = vec(collect(field_data.mean))
        if all(iszero, s_mean); @info "Skipping spatial plot '$title_str': Mean effect is zero."; return nothing; end
        if !isnothing(polygons) && length(polygons) >= length(s_mean); return plot_choropleth(s_mean, polygons; title=title_str);
        elseif !isnothing(centroids); return scatter(getindex.(centroids, 1), getindex.(centroids, 2), marker_z=s_mean, markersize=4, c=:viridis, label=nothing, title=title_str, aspect_ratio=:equal); end
        return nothing
    end

    if hasproperty(effects, :spatial_denoised); s_field = (res.arch isa MultivariateArchitecture) ? effects.spatial_denoised[outcome] : effects.spatial_denoised; p = _create_choropleth_plot(s_field, "Spatial Denoised Effect", polygons, centroids); if !isnothing(p); plots[:spatial_denoised] = p; end; end
    if hasproperty(effects, :spatial_noisy); s_field = (res.arch isa MultivariateArchitecture) ? effects.spatial_noisy[outcome] : effects.spatial_noisy; p = _create_choropleth_plot(s_field, "Total Spatial Effect", polygons, centroids); if !isnothing(p); plots[:spatial_noisy] = p; end; end

    if hasproperty(effects, :temporal); t_field = (res.arch isa MultivariateArchitecture) ? effects.temporal[outcome] : effects.temporal; if !isnothing(t_field) && hasproperty(t_field, :mean) && !all(iszero, t_field.mean); tm, tl, tu = vec(t_field.mean), vec(t_field.lower), vec(t_field.upper); plots[:temporal] = plot(tm, ribbon=(tm .- tl, tu .- tm), title="Temporal Trend", lw=2, fillalpha=0.2, color=:royalblue, legend=false, xlabel="Time Index"); end; end
    if hasproperty(effects, :seasonal) && !isnothing(effects.seasonal) && hasproperty(effects.seasonal, :mean) && !all(iszero, effects.seasonal.mean); um, ul, uu = vec(effects.seasonal.mean), vec(effects.seasonal.lower), vec(effects.seasonal.upper); plots[:seasonal] = plot(um, ribbon=(um .- ul, uu .- um), title="Seasonal Component", lw=2, fillalpha=0.2, color=:forestgreen, legend=false, xlabel="Period"); end

    if hasproperty(effects, :smooth_effects) && effects.smooth_effects isa NamedTuple
        if isnothing(data); @info "Skipping smooth effects plots: `data` not provided.";
        else
            smooth_plots = Dict{Symbol, Any}()
            for (var_sym, smooth_summary) in pairs(effects.smooth_effects)
                if hasproperty(smooth_summary, :mean) && !all(iszero, smooth_summary.mean) && hasproperty(data, var_sym)
                    cov_data = data[!, var_sym]; p_order = sortperm(cov_data); sm, sl, su = vec(smooth_summary.mean), vec(smooth_summary.lower), vec(smooth_summary.upper)
                    smooth_plots[var_sym] = plot(cov_data[p_order], sm[p_order], ribbon=(sm[p_order] .- sl[p_order], su[p_order] .- sm[p_order]), title="Smooth Effect: $var_sym", xlabel=string(var_sym), ylabel="Latent Effect", legend=false, color=:darkorange, fillalpha=0.2)
                end
            end
            if !isempty(smooth_plots); plots[:smooth_effects] = smooth_plots; end
        end
    end

    if hasproperty(effects, :fixed_effects) && !isnothing(effects.fixed_effects)
        fe_summary = (res.arch isa MultivariateArchitecture) ? effects.fixed_effects[outcome] : effects.fixed_effects
        if hasproperty(fe_summary, :mean) && !all(iszero, fe_summary.mean)
            fm, fl, fu = vec(fe_summary.mean), vec(fe_summary.lower), vec(fe_summary.upper)
            if !isempty(fm); coef_names = haskey(M, :Xfixed_names) ? string.(M.Xfixed_names) : ["Coef_$i" for i in 1:length(fm)]; p_forest = scatter(fm, 1:length(fm), xerror=(fm .- fl, fu .- fm), yticks=(1:length(fm), coef_names), title="Fixed Effects Coefficients", xlabel="Estimate", markersize=4, color=:black, legend=false); vline!(p_forest, [0], color=:red, ls=:dash, lw=1); plots[:fixed_effects] = p_forest; end
        end
    end

    if hasproperty(effects, :mixed_effects) && !isnothing(effects.mixed_effects)
        mixed_plots = Dict{Symbol, Any}()
        is_mv = res.arch isa MultivariateArchitecture
        for (key, effect_summary) in pairs(effects.mixed_effects)
            group_var = Symbol(effect_summary.group_var)
            group_levels = hasproperty(data, group_var) ? string.(levels(data[!, group_var])) : nothing

            summaries_to_plot = is_mv ? effect_summary.summaries[outcome] : effect_summary.summaries

            for (term_name, summary) in pairs(summaries_to_plot)
                if hasproperty(summary, :mean) && !all(iszero, summary.mean)
                    means = vec(summary.mean)
                    lowers = vec(summary.lower)
                    uppers = vec(summary.upper)
                    n_levels = length(means)
                    y_ticks_labels = isnothing(group_levels) || length(group_levels) != n_levels ? ["Level $i" for i in 1:n_levels] : group_levels
                    p_title = "Mixed Effect: $(term_name) | $(group_var)"
                    p_forest = scatter(means, 1:n_levels, xerror=(means .- lowers, uppers .- means), yticks=(1:n_levels, y_ticks_labels), title=p_title, xlabel="Effect Size", markersize=4, color=:black, legend=false, yflip=true)
                    vline!(p_forest, [0], color=:red, ls=:dash, lw=1)
                    mixed_plots[Symbol("$(key)_$(term_name)")] = p_forest
                end
            end
        end
        if !isempty(mixed_plots); plots[:mixed_effects] = mixed_plots; end
    end

    return (
        NamedTuple(plots)
    )
end

function predict(model_obj::DynamicPPL.Model, chain, new_data::DataFrame; n_samples::Int=100, alpha=0.05)
    # Purpose: The primary engine for projecting a fitted model onto new data.
    # Rationale: This function constructs a "prediction set" configuration (PS) that mirrors the training configuration (M)
    #            but is adapted for the `new_data`. It correctly handles the projection of fixed effects, smooth basis functions,
    #            and nested models.
    # v1.2.8 (2026-07-17)
    # Inputs:
    #   - model_obj: The fitted Turing model object.
    #   - chain: The MCMC chain result.
    #   - new_data: A DataFrame with the same column names as the training data.
    #   - n_samples: The number of posterior samples to use for prediction.
    #   - alpha: The significance level for credible intervals.
    # Outputs: A NamedTuple containing denoised and noisy predictions, posterior stats, and the PS object.
    M_train = model_obj.args.M
    n_samps = min(size(chain, 1), n_samples)

    PS_dict = Dict(pairs(M_train))
    PS_dict[:data] = new_data
    PS_dict[:y_obs] = zeros(nrow(new_data)) # Placeholder
    PS_dict[:y_N] = nrow(new_data)

    # Re-create fixed effects design matrix for the new data
    if haskey(M_train, :formula)
        decomposed_formula = decompose_bstm_formula(M_train.formula)
        fixed_effects_vars = String[]
        append!(fixed_effects_vars, decomposed_formula.fixed_effects)
        for (_, mod_data_nt) in decomposed_formula.modules
            if mod_data_nt.module_type == :fixed && haskey(mod_data_nt.args, :positional_args)
                append!(fixed_effects_vars, string.(mod_data_nt.args[:positional_args]))
            end
        end
        fixed_effects_vars = unique(fixed_effects_vars)

        if !isempty(fixed_effects_vars)
            rhs = "0 + " * join(fixed_effects_vars, " + ")
            Xfixed_pred, _ = create_fixed_design(rhs, new_data; contrasts=get(M_train, :contrasts, Dict()))
            PS_dict[:Xfixed] = Matrix(Xfixed_pred)
            PS_dict[:Xfixed_N] = size(Xfixed_pred, 2)
            PS_dict[:Xfixed_names] = names(Xfixed_pred, 2)
        end
    end

    # Update indices from new_data
    if haskey(M_train, :s_idx_var) && hasproperty(new_data, M_train.s_idx_var); PS_dict[:s_idx] = new_data[!, M_train.s_idx_var]; end
    if haskey(M_train, :t_idx_var) && hasproperty(new_data, M_train.t_idx_var); PS_dict[:t_idx] = new_data[!, M_train.t_idx_var]; end
    if haskey(M_train, :u_idx_var) && hasproperty(new_data, M_train.u_idx_var); PS_dict[:u_idx] = new_data[!, M_train.u_idx_var]; end

    # Re-create basis matrices for smoothers on the new data
    if haskey(M_train, :manifolds)
        ps_basis_registry = Dict{Symbol, Any}()
        smooth_specs = filter(s -> s.domain == :smooth, M_train.manifolds)
        
        for spec in smooth_specs
            key_sym = Symbol(spec.var)
            vars = get(spec.params, :positional_args, [])
            n_vars = length(vars)
            if haskey(M_train.basis_matrices, key_sym) && all(hasproperty(new_data, Symbol(v)) for v in vars)
                m_obj = spec.manifold_obj
                model_type_str = lowercase(string(typeof(m_obj)))
                nb = size(M_train.basis_matrices[key_sym], 2)
                if n_vars == 1
                    ps_basis_registry[key_sym] = bstm_smooth_basis_1D(model_type_str, new_data[!, Symbol(vars[1])], nb; spec.params...)
                elseif n_vars > 1
                    coords_new = Matrix{Float64}(new_data[!, Symbol.(vars)])
                    if n_vars == 2; ps_basis_registry[key_sym] = bstm_smooth_basis_2D(model_type_str, coords_new, nb; spec.params...);
                    elseif n_vars == 3; ps_basis_registry[key_sym] = bstm_smooth_basis_3D(model_type_str, coords_new, nb; spec.params...);
                    elseif n_vars == 4; ps_basis_registry[key_sym] = bstm_smooth_basis_4D(model_type_str, coords_new, nb; spec.params...);
                    end
                end
            end
        end
        PS_dict[:basis_matrices] = ps_basis_registry
    end

    # Create prediction sets for nested sub-models
    if haskey(M_train, :nested_manifolds) && !isempty(M_train.nested_manifolds)
        PS_dict[:nested_prediction_sets] = Dict{Symbol, Any}()
        for (key, sub_M) in M_train.nested_manifolds
            sub_PS_dict = Dict(pairs(sub_M))
            sub_PS_dict[:data] = new_data
            sub_PS_dict[:y_obs] = zeros(nrow(new_data)) # Placeholder
            sub_PS_dict[:y_N] = nrow(new_data)

            if haskey(sub_M, :formula)
                sub_decomposed = decompose_bstm_formula(sub_M.formula)
                
                sub_fixed_effects_vars = String[]
                append!(sub_fixed_effects_vars, sub_decomposed.fixed_effects)
                for (_, mod_data_nt) in sub_decomposed.modules
                    if mod_data_nt.module_type == :fixed && haskey(mod_data_nt.args, :positional_args)
                        append!(sub_fixed_effects_vars, string.(mod_data_nt.args[:positional_args]))
                    end
                end
                sub_fixed_effects_vars = unique(sub_fixed_effects_vars)

                if !isempty(sub_fixed_effects_vars)
                    rhs = "0 + " * join(sub_fixed_effects_vars, " + ")
                    Xfixed_sub, _ = create_fixed_design(rhs, new_data; contrasts=get(sub_M, :contrasts, Dict()))
                    sub_PS_dict[:Xfixed] = Matrix(Xfixed_sub)
                    sub_PS_dict[:Xfixed_N] = size(Xfixed_sub, 2)
                    sub_PS_dict[:Xfixed_names] = names(Xfixed_sub, 2)
                else
                    sub_PS_dict[:Xfixed] = zeros(nrow(new_data), 0)
                    sub_PS_dict[:Xfixed_N] = 0
                    sub_PS_dict[:Xfixed_names] = Symbol[]
                end
            end

            if haskey(sub_M, :manifolds)
                sub_ps_basis_registry = Dict{Symbol, Any}()
                sub_smooth_specs = filter(s -> s.domain == :smooth, sub_M.manifolds)
                for spec in sub_smooth_specs
                    v_sym = Symbol(spec.var)
                    vars = get(spec.params, :positional_args, [])
                    n_vars = length(vars)
                    if haskey(sub_M.basis_matrices, v_sym) && all(hasproperty(new_data, Symbol(v)) for v in vars)
                        m_obj = spec.manifold_obj
                        model_type_str = lowercase(string(typeof(m_obj)))
                        nb = size(sub_M.basis_matrices[v_sym], 2)
                        if n_vars == 1
                            sub_ps_basis_registry[v_sym] = bstm_smooth_basis_1D(model_type_str, new_data[!, Symbol(vars[1])], nb; spec.params...)
                        elseif n_vars > 1
                            coords_new = Matrix{Float64}(new_data[!, Symbol.(vars)])
                            if n_vars == 2; sub_ps_basis_registry[v_sym] = bstm_smooth_basis_2D(model_type_str, coords_new, nb; spec.params...);
                            elseif n_vars == 3; sub_ps_basis_registry[v_sym] = bstm_smooth_basis_3D(model_type_str, coords_new, nb; spec.params...);
                            elseif n_vars == 4; sub_ps_basis_registry[v_sym] = bstm_smooth_basis_4D(model_type_str, coords_new, nb; spec.params...);
                            end
                        end
                    end
                end
                sub_PS_dict[:basis_matrices] = sub_ps_basis_registry
            end

            if haskey(sub_M, :likelihood_specs) && !isempty(sub_M.likelihood_specs)
                sub_lik_params = sub_M.likelihood_specs[1]
                _resolve_obs_param!(sub_PS_dict, sub_lik_params, new_data, [:offsets, :log_offsets], :log_offset)
                _resolve_obs_param!(sub_PS_dict, sub_lik_params, new_data, [:weights], :weights)
                _resolve_obs_param!(sub_PS_dict, sub_lik_params, new_data, [:trials], :trials)
            end
            _precompute_likelihood_params!(sub_PS_dict)

            PS_dict[:nested_prediction_sets][key] = NamedTuple(sub_PS_dict)
        end
    end

    PS = NamedTuple(PS_dict)

    raw_arch = get(M_train, :model_arch, "univariate")
    arch_type = if raw_arch == "multivariate"; MultivariateArchitecture()
    elseif raw_arch == "multifidelity"; MultifidelityArchitecture()
    else; UnivariateArchitecture(); end

    chain_sub = chain[1:min(n_samps, end), :, :]

    res = _reconstruct(arch_type, "prediction", chain_sub, M_train, PS, alpha)

    # Slice the prediction part from the full summary.
    N_train = M_train.y_N
    
    function slice_summary(summary)
        if summary isa AbstractVector # Multivariate case
            return [(mean=s.mean[(N_train+1):end], median=s.median[(N_train+1):end], std=s.std[(N_train+1):end], lower=s.lower[(N_train+1):end], upper=s.upper[(N_train+1):end]) for s in summary]
        else # Univariate case
            return (mean=summary.mean[(N_train+1):end], median=summary.median[(N_train+1):end], std=summary.std[(N_train+1):end], lower=summary.lower[(N_train+1):end], upper=summary.upper[(N_train+1):end])
        end
    end

    return (
        predictions_denoised = slice_summary(res.predictions_denoised),
        predictions_noisy = slice_summary(res.predictions_noisy),
        pstats = res,
        PS = PS
    )
end

function post_stratification_weights(res, M, PS, samples_denoised)
    # Purpose: Computes post-stratification weights to scale sample-level predictions to population-level estimates.
    # Rationale: This is essential for generating total abundance or biomass indices from survey data.
    #            The weight for an observation `i` in stratum `j` is calculated as `Area(j) / n_obs_in_stratum(j)`.
    #            Multiplying the predicted density at `i` by this weight gives its contribution to the total stratified estimate.
    # Assumptions:
    #   1. `M` contains a `:strata_info` DataFrame with `stratum_id` and `stratum_area` columns.
    #   2. The data (`M.data` and optionally `PS.data`) contains a `stratum_id` column.
    # Inputs:
    #   - res: The main results object (not used in this implementation but kept for API consistency).
    #   - M: The model configuration object for the training data.
    #   - PS: The prediction set configuration object (can be `nothing`).
    #   - samples_denoised: A matrix of posterior predictions [n_obs x n_samples].
    # Outputs: A matrix of weights of the same size as `samples_denoised`.

    # #
    # Input validation
    if !haskey(M, :strata_info) || !("stratum_id" in names(M.strata_info)) || !("stratum_area" in names(M.strata_info))
        @warn "Post-stratification requires `:strata_info` in the model configuration with `stratum_id` and `stratum_area` columns. Returning ones."
        return ones(Float64, size(samples_denoised))
    end
    if !hasproperty(M.data, :stratum_id)
        @warn "Post-stratification requires a `stratum_id` column in the training data. Returning ones."
        return ones(Float64, size(samples_denoised))
    end

    # #
    # Combine stratum IDs from training and prediction sets
    strata_info = M.strata_info
    strata_ids_train = M.data.stratum_id
    
    strata_ids_full = if !isnothing(PS)
        if !hasproperty(PS.data, :stratum_id)
            @warn "Prediction set provided but is missing `stratum_id` column. Post-stratification weights will only be calculated for training data."
            strata_ids_train
        else
            vcat(strata_ids_train, PS.data.stratum_id)
        end
    else
        strata_ids_train
    end
    
    n_obs_total = length(strata_ids_full)
    n_samples = size(samples_denoised, 2)

    # #
    # Calculate the weight for each stratum (Area / N_obs)
    unique_strata = unique(strata_info.stratum_id)
    stratum_area_map = Dict(row.stratum_id => row.stratum_area for row in eachrow(strata_info))
    obs_counts = StatsBase.countmap(strata_ids_full)
    
    stratum_weight_map = Dict{eltype(unique_strata), Float64}()
    for stratum in unique_strata
        area = get(stratum_area_map, stratum, 0.0)
        count = get(obs_counts, stratum, 0)
        stratum_weight_map[stratum] = count > 0 ? area / count : 0.0
    end

    # #
    # Map stratum weights to each observation
    obs_weights = [get(stratum_weight_map, id, 0.0) for id in strata_ids_full]

    # #
    # Return weights matrix, broadcasted across all posterior samples
    return repeat(obs_weights, 1, n_samples)
end

function model_results_plots(res)
    # Purpose: Displays all plots generated by `model_results_comprehensive`.
    # Rationale: A simple convenience function to iterate through and display the
    #            contents of the `plots` object returned by the main results function.
    if !hasproperty(res, :plots) || isempty(res.plots)
        println("No plots found in the results object.")
        return
    end

    println("--- Displaying Generated Plots ---")
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
    println("--- End of Plots ---")
end

function plot_choropleth(values::AbstractVector, polygons::Vector; title="Spatial Distribution", cmap=:viridis)
    # Purpose: A simple choropleth plotting utility.
    # Rationale: Provides a basic visualization for spatial fields on polygonal units.
    plt = plot(aspect_ratio=:equal, title=title, legend=false, grid=false, showaxis=false, xticks=false, yticks=false)
    
    # Determine the color range for normalization
    min_val, max_val = extrema(values)
    
    for i in 1:min(length(polygons), length(values))
        poly_coords = polygons[i]
        
        # A valid polygon requires at least 3 vertices
        if length(poly_coords) > 2
            # Extract x and y coordinates, filtering out any NaN values
            px = [pt[1] for pt in poly_coords if !isnan(pt[1])]
            py = [pt[2] for pt in poly_coords if !isnan(pt[2])]
            
            # Proceed only if there are valid coordinates
            if !isempty(px)
                # Ensure the polygon is closed for plotting
                if (px[1], py[1]) != (px[end], py[end])
                    push!(px, px[1])
                    push!(py, py[1])
                end
                
                plot!(plt, px, py, seriestype=:shape, fill_z=values[i], c=cmap, linecolor=:black, lw=0.5, fillalpha=0.8, label=nothing)
            end
        end
    end
    return plt
end

function bstm_cv_orchestrator(
    formula::String, 
    data::DataFrame; 
    method::Symbol = :kfold, 
    cv_var::Symbol = :s_idx, 
    n_folds::Int = 5, 
    n_samples::Int = 500, 
    sampler = NUTS(500, 0.65), 
    alpha = 0.05, 
    cv_space_vars::Vector{Symbol} = [:s_x, :s_y],
    kwargs...
)    
    # Purpose: An orchestration utility for performing cross-validation. It supports standard 
    #          k-fold, Leave-One-Location-Out (LOLO), spatial blocking, and temporal blocking/forward-chaining
    #          strategies to assess model performance on held-out data.
    # Rationale: Provides a standardized and flexible way to evaluate model predictive performance
    #            while accounting for spatial and temporal data structures.
    # Inputs:
    #   - formula: The bstm model formula.
    #   - data: The input DataFrame.
    #   - method: The CV method. One of `:kfold`, `:lolo`, `:spatial_block`, `:temporal_block`, `:temporal_forward_chain`.
    #   - cv_var: The column name to use for grouping/blocking (for `:lolo`, `:temporal_block`, `:temporal_forward_chain`).
    #   - n_folds: The number of folds for k-fold or blocking methods.
    #   - sampler: The Turing sampler to use.
    #   - cv_space_vars: Columns for spatial coordinates for `:spatial_block`.
    #   - kwargs: Additional arguments passed to `bstm_config`.
    # Outputs: A NamedTuple containing fold-level results and summary metrics.
    
    meta_discovery = decompose_bstm_formula(formula)
    response_name = Symbol(meta_discovery.outcomes[1][:var])

    folds_indices = Vector{Vector{Int}}()
    is_forward_chain = false

    if method == :lolo
        if !hasproperty(data, cv_var); error("LOLO cross-validation requires the specified `cv_var` column ':$cv_var' in the data."); end
        unique_locs = unique(data[!, cv_var])
        for loc in unique_locs
            push!(folds_indices, findall(x -> x == loc, data[!, cv_var]))
        end
    elseif method == :spatial_block
        if !all(hasproperty(data, v) for v in cv_space_vars); error("Spatial block cross-validation requires coordinate columns specified in `cv_space_vars`: $cv_space_vars."); end
        coords = Matrix(data[!, cv_space_vars])' # kmeans expects features in rows
        R = Clustering.kmeans(coords, n_folds; maxiter=200, display=:none)
        assignments = R.assignments
        for k in 1:n_folds
            fold_k_indices = findall(x -> x == k, assignments)
            if !isempty(fold_k_indices); push!(folds_indices, fold_k_indices); end
        end
    elseif method == :temporal_block
        if !hasproperty(data, cv_var); error("Temporal block cross-validation requires the specified `cv_var` column ':$cv_var' in the data."); end
        unique_times = sort(unique(data[!, cv_var]))
        fold_size = cld(length(unique_times), n_folds) # ceiling division
        for i in 1:n_folds
            start_idx = (i - 1) * fold_size + 1
            end_idx = min(i * fold_size, length(unique_times))
            if start_idx > length(unique_times); continue; end
            time_block = unique_times[start_idx:end_idx]
            push!(folds_indices, findall(t -> t in time_block, data[!, cv_var]))
        end
    elseif method == :temporal_forward_chain
        if !hasproperty(data, cv_var); error("Forward-chaining cross-validation requires the specified `cv_var` column ':$cv_var' in the data."); end
        is_forward_chain = true
        unique_times = sort(unique(data[!, cv_var]))
        if length(unique_times) <= n_folds; @warn "Number of unique time points ($(length(unique_times))) is less than or equal to `n_folds` ($n_folds). Consider reducing `n_folds` for forward-chaining."; end
        test_times = unique_times[end-n_folds+1:end]
        for t in test_times
            push!(folds_indices, findall(x -> x == t, data[!, cv_var]))
        end
    else # Default to k-fold
        n_obs = size(data, 1)
        row_indices = Random.randperm(n_obs)
        fold_size = cld(n_obs, n_folds)
        for i in 1:n_folds
            idx_start = (i - 1) * fold_size + 1
            idx_end = min(i * fold_size, n_obs)
            if idx_start > n_obs; continue; end
            push!(folds_indices, row_indices[idx_start:idx_end])
        end
    end

    fold_results = []
    n_actual_folds = length(folds_indices)

    for (f_idx, test_idx) in enumerate(folds_indices)
        test_data = data[test_idx, :]
        
        train_data = if is_forward_chain
            min_test_time = minimum(test_data[!, cv_var])
            train_idx = findall(t -> t < min_test_time, data[!, cv_var])
            data[train_idx, :]
        else
            train_mask = trues(size(data, 1))
            train_mask[test_idx] .= false
            data[train_mask, :]
        end

        if nrow(train_data) == 0; @warn "Fold $f_idx created an empty training set. Skipping."; continue; end

        opt_train = bstm_config(formula, train_data; kwargs...)
        model_train = bstm(opt_train)
        chain_train = sample(model_train, sampler, n_samples; progress=false)
        res_pred = predict(model_train, chain_train, test_data; n_samples=div(n_samples, 2), alpha=alpha)

        y_test_obs = test_data[!, response_name]
        y_test_pred = res_pred.predictions_denoised.mean

        if length(y_test_obs) == length(y_test_pred)
            residuals = y_test_obs .- y_test_pred
            rmse = sqrt(Statistics.mean(residuals.^2))
            ss_res = sum(residuals.^2)
            ss_tot = sum((y_test_obs .- Statistics.mean(y_test_obs)).^2)
            r2 = 1.0 - (ss_res / (ss_tot + 1e-15))
            push!(fold_results, (fold=f_idx, rmse=rmse, r2=r2))
        else
            @warn "Fold $f_idx: Prediction length mismatch. Observed: $(length(y_test_obs)), Predicted: $(length(y_test_pred))"
        end
    end

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

# ==============================================================================
# SECTION 5: MODEL SELECTION AND COMPARISON
# ==============================================================================

function bstm_loo(model_obj::DynamicPPL.Model, chain; alpha=0.05)    
    # Purpose: A utility for performing Leave-One-Out Cross-Validation using Pareto Smoothed Importance 
    #          Sampling (PSIS-LOO) to assess a model's out-of-sample predictive accuracy.
    # Inputs: model_obj, chain, alpha.
    # Outputs: A NamedTuple containing the LOO object, metrics, log-likelihood matrix, and Pareto k values.
    
    # #
    # 1. Metadata and Architecture Extraction
    # Rationale: M contains the configuration and technical registry required for reconstruction.
    M = model_obj.args.M
    raw_arch = get(M, :model_arch, "univariate")

    # #
    # 2. Technical Dispatch Resolution
    # Mapping the configuration string to the architectural dispatch types.
    arch_type = if raw_arch == "univariate"
        UnivariateArchitecture()
    elseif raw_arch == "multivariate"
        MultivariateArchitecture()
    elseif raw_arch == "multifidelity"
        MultifidelityArchitecture()
    else
        UnivariateArchitecture()
    end

    # #
    # 3. Latent Manifold Reconstruction for Likelihood Registry
    # Rationale: _reconstruct generates the [Samples x Observations] log-likelihood matrix.
    # We utilize alpha for consistent summarization during the recovery phase.
    println("Audit: Recovering pointwise log-likelihood registry...")
    res = _reconstruct(arch_type, "loo_recovery", chain, M, nothing, alpha)

    # #
    # 4. Matrix Extraction and Validation
    # Rationale: Ensuring the log_likelihood matches the observation grid dimensions.
    log_lik = res.log_likelihood
    n_samples, n_obs = size(log_lik)

    println("Audit: Processing ", n_samples, " samples for ", n_obs, " observations.")

    # #
    # 5. PSIS-LOO Calculation via PosteriorStats
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
        log_likelihood = log_lik,
        pareto_k = pareto_k
    )
end

function compare_manifolds(loo_a_report, loo_b_report; model_names=["Model_A", "Model_B"])    
    # Purpose: A utility for formal model comparison between two fitted `bstm` models. It uses 
    #          their PSIS-LOO results to compute the difference in Expected Log Pointwise 
    #          Predictive Density (ELPD) and provides a statistical basis for model selection.
    # Inputs: loo_a_report, loo_b_report, model_names.
    # Outputs: A NamedTuple containing the comparison table, ELPD difference, and LOO objects.

    println("--- Starting BSTM Manifold Comparison ---")

    # #
    # 1. LOO Object Extraction
    loo_a = loo_a_report.loo_obj
    loo_b = loo_b_report.loo_obj

    # #
    # 2. Formal Selection Metric Calculation
    comparison_stats = nothing
    try
        comparison_stats = compare([loo_a, loo_b])
    catch e
        @error "BSTM Comparison Error: Selection suite failed. Error: " * string(e)
        return nothing
    end

    # #
    # 3. Parameter and Diagnostic Extraction
    p_loo_a = loo_a_report.metrics.p_loo
    p_loo_b = loo_b_report.metrics.p_loo
    elpd_a = loo_a_report.metrics.elpd
    elpd_b = loo_b_report.metrics.elpd

    # #
    # 4. Report Generation
    println("\n--- BSTM Manifold Selection Registry ---")
    println("Model A (", model_names[1], "): ELPD = ", round(elpd_a, digits=2), " | p_loo = ", round(p_loo_a, digits=2))
    println("Model B (", model_names[2], "): ELPD = ", round(elpd_b, digits=2), " | p_loo = ", round(p_loo_b, digits=2))
    diff_elpd = elpd_a - elpd_b
    println("\nELPD Delta (A - B): ", round(diff_elpd, digits=2))

    if abs(diff_elpd) > 4.0
        winning_model = diff_elpd > 0 ? model_names[1] : model_names[2]
        println("CONCLUSION: ", winning_model, " is statistically preferred based on predictive density.")
    else
        println("CONCLUSION: Competing manifold structures provide indistinguishable predictive density.")
    end

    # #
    # 5. Table Construction
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
