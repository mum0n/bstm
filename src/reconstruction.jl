# reconstruction logic



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

 
# Version 1.1.2 (2026-08-06)
# Purpose: Finds a parameter name in a list based on the `param_name_key` convention.
# Rationale: This version is updated to correctly match the naming convention used by
#            `generate_full_variable_names`, which includes an underscore and outcome
#            index (e.g., `sigma_s_idx_1`) for multivariate/outcome-specific parameters,
#            and handles bracketed indexing (e.g., `sigma_s_idx[1]`) for vector parameters.
#            It prioritizes the most specific match found in the chain's parameter names.
function _find_parameter(p_names, key, param_name, k=nothing)
    # This function searches for a parameter in the MCMC chain's parameter list (`p_names`)
    # by trying several common naming conventions in order of specificity.

    base_name_prefix = "$(param_name)_$(key)" # e.g., "sigma_s_idx"

    # Priority 1: Exact match with outcome index suffix (e.g., "sigma_s_idx_1").
    # This is the most specific pattern, used for outcome-specific parameters in multivariate models.
    if !isnothing(k)
        specific_name_with_k_suffix = "$(base_name_prefix)_$(k)"
        if specific_name_with_k_suffix in p_names
            return specific_name_with_k_suffix
        end
        
        # Also check for bracketed version with a specific index (e.g., "sigma_s_idx[1]").
        indexed_name_with_k_bracket = "$(base_name_prefix)[$(k)]"
        if indexed_name_with_k_bracket in p_names
            return indexed_name_with_k_bracket
        end
    end

    # Priority 2: Exact match for the base name (e.g., "sigma_s_idx").
    # This is used for univariate models or parameters shared across outcomes.
    if base_name_prefix in p_names
        return base_name_prefix
    end

    # Priority 3: Check for any bracketed indexed versions of the base name (e.g., "sigma_s_idx[1]", "sigma_s_idx[2]", ...).
    # This is a fallback to indicate that a parameter of this base name exists as a vector.
    # `get_params_vector` will then handle extracting all its elements.
    re_indexed_any = Regex("^" * escape_string(base_name_prefix) * "\\[\\d+\\]")
    if any(n -> occursin(re_indexed_any, n), p_names)
        return base_name_prefix
    end

    return "" # Return empty string if no match is found.
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
# SECTION 2: COMPONENT-SPECIFIC EXTRACTION
# ==============================================================================
 

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

# Version 1.0.1 (2026-08-06)
# Purpose: Summarizes the posterior samples for all discovered component effects.
# Rationale: This version is updated to create a more informative summary object. Instead of
#            only summarizing one effect per component, it now iterates through all effect
#            (e.g., `structured`, `unstructured`, `noisy`)
#            and summarizes each one. This allows downstream functions like `bstm_plots` to
#            visualize different aspects of a component (e.g., the structured vs. unstructured
#            parts of a BYM2 model) separately.
function _summarize_effects_registry(registry, M, outcomes_N, alpha)
    summarized_registry = Dict{Symbol, Any}()
    mixed_effects_summaries = Dict{Symbol, Any}()

    for (key, effects) in pairs(registry)
        if key in [:intercept, :fixed]; continue; end

        spec_idx = findfirst(s -> s.key == key, M.components)
        if !isnothing(spec_idx) && M.components[spec_idx].component_obj isa Mixed
            # --- Handle Mixed Effects Separately ---
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
            mixed_effects_summaries[key] = (group_var=M.components[spec_idx].var, summaries=summaries_final)
        else
            # --- Handle Standard Components ---
            component_summary = Dict{Symbol, Any}()
            # Iterate over all fields in the effects tuple (e.g., :structured, :unstructured, :noisy)
            for effect_type in keys(effects)
                if !(effects[effect_type] isa AbstractVector); continue; end # Skip non-effect fields like :indices
                
                effect_set = effects[effect_type]
                if isempty(effect_set); continue; end

                if outcomes_N > 1
                    component_summary[effect_type] = [summarize_array(effect_set[k], alpha=alpha) for k in 1:outcomes_N]
                else
                    component_summary[effect_type] = summarize_array(effect_set[1], alpha=alpha)
                end
            end
            if !isempty(component_summary)
                summarized_registry[key] = NamedTuple(component_summary)
            end
        end
    end
    if !isempty(mixed_effects_summaries); summarized_registry[:mixed_effects] = NamedTuple(mixed_effects_summaries); end
    
    return NamedTuple(summarized_registry)
end


# Version 1.0.0 (2026-08-06)
# Purpose: Computes summary statistics (mean, median, std, credible intervals) from a matrix of posterior samples.
# Rationale: This function was recreated to restore lost functionality. It provides a standardized way to summarize
#            posterior predictions from a `[n_observations x n_samples]` matrix into a NamedTuple of vectors,
#            which is the format expected by downstream plotting and analysis functions. It is functionally
#            equivalent to `summarize_array` but is named specifically for summarizing prediction matrices.
# Inputs:
#   - samples: An AbstractArray of posterior samples, where the last dimension is the sample dimension.
#   - alpha: The significance level for the credible intervals (e.g., 0.05 for a 95% CI).
# Outputs: A NamedTuple containing vectors for the mean, median, standard deviation, and lower/upper credible bounds.
function summarize_predictions(samples::AbstractArray; alpha=0.05)
    # Ensure the input array is not empty or full of NaNs to prevent errors.
    if isempty(samples) || all(isnan, samples)
        return (mean = Float64[], median = Float64[], std = Float64[], lower = Float64[], upper = Float64[])
    end

    # The last dimension is assumed to be the dimension over which samples are drawn.
    sample_dim = ndims(samples)
    low_prob = alpha / 2.0
    high_prob = 1.0 - low_prob

    # Calculate standard summary statistics across the sample dimension.
    # `dropdims` removes the singleton dimension after the reduction.
    post_mean = dropdims(Statistics.mean(samples, dims=sample_dim), dims=sample_dim)
    post_median = dropdims(Statistics.median(samples, dims=sample_dim), dims=sample_dim)
    post_std = dropdims(Statistics.std(samples, dims=sample_dim), dims=sample_dim)

    # Calculate quantiles for the credible intervals.
    # `_quantile_along_last_dim` is a helper to compute quantiles over the last dimension.
    low_bound = _quantile_along_last_dim(samples, low_prob; sample_dim=sample_dim)
    high_bound = _quantile_along_last_dim(samples, high_prob; sample_dim=sample_dim)

    # Helper function to ensure all outputs are consistently formatted as vectors.
    to_vector(x) = x isa AbstractArray ? vec(collect(Float64, x)) : [Float64(x)]

    # Return the results as a NamedTuple.
    return (
        mean = to_vector(post_mean),
        median = to_vector(post_median),
        std = to_vector(post_std),
        lower = to_vector(low_bound),
        upper = to_vector(high_bound)
    )
end


"""
    _reconstruct(arch::UnivariateArchitecture, mode::String, chain, M::NamedTuple, PS, alpha::Float64)

Reconstructs posterior summaries for univariate models.

# Rationale for Update
This function is updated to call the refactored `_discover_component_realizations`
without the obsolete `all_vars` (parameter names) argument, aligning it with the
new explicit component interface where components manage their own parameter discovery.
"""
function _reconstruct(arch::UnivariateArchitecture, mode::String, chain, M::NamedTuple, PS, alpha::Float64)
    # --- 1. Metadata and Dimension Discovery ---
    n_samples_val = size(chain, 1)
    N_tot_val = isnothing(PS) ? M.y_N : M.y_N + PS.y_N
    outcomes_N_val = M.outcomes_N # For univariate, this is always 1.

    # --- 2. Latent Field Reconstruction ---
    # The `all_vars` argument is removed from this call.
    registry = _discover_component_realizations(chain, M, PS, n_samples_val, outcomes_N_val, N_tot_val)

    # --- 3. Linear Predictor Assembly ---
    eta_post = _modular_eta_assembly(registry, M, PS, n_samples_val, outcomes_N_val)

    # --- 4. Prediction and Log-Likelihood Calculation ---
    p_denoised, p_noisy, log_lik = _process_ll_and_predictions(eta_post, chain, M, PS, outcomes_N_val, 1)

    # --- 5. Final Result Consolidation ---
    pstats = (
        effects=_summarize_effects_registry(registry, M, outcomes_N_val, alpha),
        predictions_denoised=summarize_predictions(dropdims(p_denoised, dims=3); alpha=alpha),
        predictions_noisy=summarize_predictions(dropdims(p_noisy, dims=3); alpha=alpha),
        log_likelihood=log_lik,
        waic=_compute_waic(log_lik),
        arch=arch
    )

    return pstats
end



"""
    _reconstruct(arch::MultivariateArchitecture, mode::String, chain, M::NamedTuple, PS, alpha::Float64)

Reconstructs posterior summaries for multivariate models.

# Rationale for Update
This function is updated to call the refactored `_discover_component_realizations`
without the obsolete `all_vars` (parameter names) argument, aligning it with the
new explicit component interface.
"""
function _reconstruct(arch::MultivariateArchitecture, mode::String, chain, M::NamedTuple, PS, alpha::Float64)
    # --- 1. Metadata and Dimension Discovery ---
    n_samples_val = size(chain, 1)
    N_tot_val = isnothing(PS) ? M.y_N : M.y_N + PS.y_N
    outcomes_N_val = M.outcomes_N

    # --- 2. Latent Field Reconstruction ---
    # The `all_vars` argument is removed from this call.
    registry = _discover_component_realizations(chain, M, PS, n_samples_val, outcomes_N_val, N_tot_val)
  
    # --- 3. Linear Predictor Assembly ---
    eta_latent_post = _modular_eta_assembly(registry, M, PS, n_samples_val, outcomes_N_val)

    # --- 4. Apply Correlation Structure ---
    L_corr_samples = chain["L_corr"]
    eta_post = similar(eta_latent_post)
    for s in 1:size(eta_latent_post, 3)
        eta_post[:, :, s] = eta_latent_post[:, :, s] * L_corr_samples[s].L
    end

    # --- 5. Prediction and Log-Likelihood Calculation ---
    all_pred_results = [_process_ll_and_predictions(eta_post[:,:,k], chain, M, PS, outcomes_N_val, k) for k in 1:outcomes_N_val]
    
    p_denoised_summaries = [summarize_array(res.p_denoised, alpha=alpha) for res in all_pred_results]
    p_noisy_summaries = [summarize_array(res.p_noisy, alpha=alpha) for res in all_pred_results]
    all_log_lik = hcat([res.log_lik for res in all_pred_results]...)

    # --- 6. Final Result Consolidation ---
    pstats = (
        effects=_summarize_effects_registry(registry, M, outcomes_N_val, alpha),
        predictions_denoised=p_denoised_summaries,
        predictions_noisy=p_noisy_summaries,
        log_likelihood=all_log_lik
    )

    return pstats
end



"""
    _discover_component_realizations(chain, M, PS, n_samples, outcomes_N, N_tot)

Extracts all latent effects from the MCMC chain by dispatching to the `get_effects`
method of each component.

# Rationale for Update
This function is the core of the refactored posterior reconstruction engine. It replaces
the legacy `extract_component` dispatch system with a standardized loop that calls the
`get_effects` method defined in the explicit interface of each `ComponentModel`. This
delegates the responsibility of reconstruction to the components themselves, simplifying
the orchestration logic and making the system more modular and extensible. The `p_names`
argument has been removed, as each `get_effects` method is now responsible for generating
its own required parameter names.
"""
function _discover_component_realizations(chain, M, PS, n_samples, outcomes_N, N_tot)
    registry = Dict{Symbol, Any}()

    # --- Fixed effects and Intercept (logic remains the same) ---
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
            fixed_effects_all = zeros(Float64, N_tot, n_samples, outcomes_N)
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
        registry[:fixed] = zeros(Float64, N_tot, n_samples, outcomes_N)
    end

    if M.add_intercept
        intercept_samples = get_params_vector(chain, "intercept", outcomes_N)
        intercept_effects = zeros(Float64, N_tot, n_samples, outcomes_N)
        for k in 1:outcomes_N
            intercept_effects[:, :, k] .= intercept_samples[:, k]'
        end
        registry[:intercept] = intercept_effects
    else
        registry[:intercept] = zeros(Float64, N_tot, n_samples, outcomes_N)
    end

    # --- NEW: Main Component Loop using get_effects ---
    for spec in M.components
        # Call the get_effects method for the component. This method is now responsible
        # for all logic, including handling multivariate cases and finding its own
        # parameters in the chain.
        effects = get_effects(spec.component_obj, chain, M, n_samples, outcomes_N, spec, PS, N_tot)
        registry[spec.key] = effects
    end

    return NamedTuple(registry)
end




function _modular_eta_assembly(registry, M, PS, n_samples, outcomes_N)
    # Purpose: Assembles the full linear predictor (`eta`) from all discovered latent effects.
    # Rationale: This function iterates through all registered effects (intercept, fixed, components)
    #            and sums their contributions to form the final linear predictor for each observation
    #            and each posterior sample. It correctly handles the mapping of effects from
    #            component-specific units (e.g., spatial units) to observation-level.
    # v1.5.9 (2026-08-06)
    # Inputs:
    #   - registry: A NamedTuple containing raw posterior samples for each model component.
    #   - M: The model configuration NamedTuple (for training data).
    #   - PS: The prediction set configuration NamedTuple (for out-of-sample data), or `nothing`.
    #   - n_samples: The total number of posterior samples.
    #   - outcomes_N: The number of outcome variables.
    # Outputs: A 3D array `eta_latent` of size `[N_total x n_samples x outcomes_N]`.

    N_tot = isnothing(PS) ? M.y_N : M.y_N + PS.y_N
    eta_latent = zeros(Float64, N_tot, n_samples, outcomes_N)

    # Add intercept and fixed effects first.
    # These are already expanded to [N_total x n_samples x outcomes_N] by _discover_component_realizations.
    eta_latent .+= registry.intercept
    eta_latent .+= registry.fixed

    # Pre-compute full index vectors for spatial, temporal, and seasonal structures.
    # These map observations to their corresponding component units (e.g., spatial unit ID).
    s_idx_full = haskey(M, :s_idx) ? (isnothing(PS) || !haskey(PS, :s_idx) ? M.s_idx : vcat(M.s_idx, PS.s_idx)) : ones(Int, N_tot)
    t_idx_full = haskey(M, :t_idx) ? (isnothing(PS) || !haskey(PS, :t_idx) ? M.t_idx : vcat(M.t_idx, PS.t_idx)) : ones(Int, N_tot)
    u_idx_full = haskey(M, :u_idx) ? (isnothing(PS) || !haskey(PS, :u_idx) ? M.u_idx : vcat(M.u_idx, PS.u_idx)) : ones(Int, N_tot)

    # Iterate through all model components and add their effects to the linear predictor.
    for spec in M.components
        key = spec.key # Get the component's unique identifier (e.g., :s_idx, :year)
        if !haskey(registry, key); continue; end # Skip if no effects were reconstructed for this key.
        
        effects = registry[key]
        # Determine which set of effects to use (structured, noisy, or just the default `structured`).
        effect_set = hasproperty(effects, :noisy) ? effects.noisy : effects.structured
        if isempty(effect_set); continue; end # Skip if the effect set is empty.
        
        for k in 1:outcomes_N
            if spec.structure in [:spatial, :temporal]
                # For standard GMRF-like effects (e.g., ICAR, AR1), the effect matrix is [n_units x n_samples].
                # We use the appropriate index vector (s_idx_full or t_idx_full) to map the effect
                # from component units to each observation.
                effect_to_add = effect_set[k]
                idx_vec = spec.structure == :spatial ? s_idx_full : t_idx_full
                eta_latent[:, :, k] .+= effect_to_add[idx_vec, :]
            elseif spec.structure == :seasonal
                # Seasonal effects. Harmonic components are typically expanded to observation-level,
                # while Cyclic (GMRF-like) components are mapped via u_idx_full.
                effect_to_add = effect_set[k]
                if spec.component_obj isa Harmonic # Harmonic basis is already expanded to N_total observations
                    eta_latent[:, :, k] .+= effect_to_add
                else # Assumes Cyclic or other GMRF-like seasonal model mapped via index
                    idx_vec = u_idx_full
                    eta_latent[:, :, k] .+= effect_to_add[idx_vec, :]
                end
            elseif spec.structure in [:smooth, :interact, :nonstationaryvariance]
                # For smoothers, interactions, and non-stationary variance, the effect matrix
                # is typically already expanded to [N_total x n_samples].
                eta_latent[:, :, k] .+= effect_set[k]
            elseif spec.structure == :mixed
                # Mixed effects (random intercepts/slopes).
                # The `extract_component` for Mixed returns `effects.indices` which is
                # the full observation-level index vector for grouping levels.
                idx_full = effects.indices
                
                if effects.type == :simple
                    # Uncorrelated random intercept or slope.
                    effect_to_add = effects.effects[k]
                    if effects.lhs == "1" # Random intercept
                        eta_latent[:, :, k] .+= effect_to_add[idx_full, :]
                    else # Random slope
                        # Covariate vector needs to be combined from training and prediction data.
                        cov_name = Symbol(effects.lhs)
                        cov_vec = if isnothing(PS)
                            M.data[!, cov_name]
                        else
                            vcat(M.data[!, cov_name], PS.data[!, cov_name])
                        end
                        eta_latent[:, :, k] .+= effect_to_add[idx_full, :] .* cov_vec
                    end
                elseif effects.type == :correlated
                    # Correlated random effects (multiple terms in LHS, e.g., `(1 + cov | group)`).
                    for (term_name, term_effects) in pairs(effects.effects)
                        effect_to_add = term_effects[k]
                        if term_name == :intercept
                            eta_latent[:, :, k] .+= effect_to_add[idx_full, :]
                        else
                            # For random slopes, get the corresponding covariate data.
                            cov_name = Symbol(replace(string(term_name), "slope_" => ""))
                            cov_vec = isnothing(PS) ? M.data[!, cov_name] : vcat(M.data[!, cov_name], PS.data[!, cov_name])
                            eta_latent[:, :, k] .+= effect_to_add[idx_full, :] .* cov_vec
                        end
                    end
                end
            else # Fallback for other component types.
                # Assume the effect is already expanded to N_total x n_samples.
                if size(effect_set[k], 1) == N_tot
                    eta_latent[:, :, k] .+= effect_set[k]
                else
                    @warn "Component '$(key)' (structure $(spec.structure)) has an unexpected effect dimension ($(size(effect_set[k], 1))) for outcome $k. Expected $N_tot. Skipping."
                end
            end
        end
    end

    # Add log_offsets at the very end.
    # M.log_offsets is a matrix of [N_obs x outcomes_N].
    if haskey(M, :log_offsets)
        for k in 1:outcomes_N
            # `offset_full` is constructed for the current outcome `k`.
            offset_full = isnothing(PS) ? M.log_offsets[:,k] : vcat(M.log_offsets[:,k], get(PS, :log_offsets, zeros(PS.y_N, outcomes_N))[:,k])
            eta_latent[:, :, k] .+= offset_full
        end
    end

    return eta_latent
end





# Version 1.1.0 (2026-08-06)
# Purpose: Generates denoised predictions, noisy predictions, and log-likelihood values from eta.
# Rationale: This version corrects the parameter names used to extract likelihood-specific
#            parameters from the MCMC chain. The original code used `sigma_y` and `lik_r_nb`,
#            which caused "Parameter not discovered" warnings. The fix updates these to `y_sigma`
#            and `r_nb`, which are the standard names produced by the bstm code generator.
function _process_ll_and_predictions(eta_samples, chain, M, PS, outcomes_N, k)
    # This function applies the inverse link function to the linear predictor (`eta`) to get
    # denoised predictions and then samples from the full predictive distribution to get
    # noisy predictions. It also calculates the pointwise log-likelihood for WAIC.

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

    # FIX: Use the correct parameter names `y_sigma` and `r_nb`.
    y_sigma_samples = get_params_vector(chain, "y_sigma", outcomes_N)
    r_nb_samples = get_params_vector(chain, "r_nb", outcomes_N)

    trials_full = haskey(M, :trials) ? (isnothing(PS) ? M.trials[:,k] : vcat(M.trials[:,k], get(PS, :trials, ones(Int, PS.y_N)))) : ones(Int, N_tot)
    
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



function _process_multinomial_predictions(eta_samples, chain, M, PS)
    # Purpose: Generates predictions and log-likelihood for multinomial models.
    # Rationale: This specialized function handles the vector nature of multinomial outcomes.
    n_samples = size(eta_samples, 2)
    N_train = M.y_N
    N_pred = isnothing(PS) ? 0 : PS.y_N
    N_tot = N_train + N_pred
    K = M.outcomes_N

    y_obs_train = M.y_obs # [N_train, K]

    # Denoised predictions (proportions)
    p_denoised_samples = zeros(Float64, N_tot, K, n_samples)
    for s in 1:n_samples 
        for i in 1:N_tot
            p_denoised_samples[i, :, s] = softmax(eta_samples[i, s, :])
        end
    end

    # Noisy predictions (counts)
    p_noisy_samples = zeros(Int, N_tot, K, n_samples)
    log_lik_samples = zeros(Float64, N_train, n_samples)

    # Get total trials for each observation
    trials_train = sum(y_obs_train, dims=2)
    # For prediction, we might need to assume a total count, or it could be in PS.
    # Assuming 1 for simplicity if not provided.
    trials_pred = haskey(PS, :trials) ? sum(PS.trials, dims=2) : ones(Int, N_pred)
    trials_full = vcat(vec(trials_train), vec(trials_pred))

    for s in 1:n_samples 
        for i in 1:N_tot
            probs = p_denoised_samples[i, :, s]
            dist = Multinomial(Int(trials_full[i]), probs)
            p_noisy_samples[i, :, s] = rand(dist)
            if i <= N_train; log_lik_samples[i, s] = logpdf(dist, y_obs_train[i, :]); end
        end
    end
    return (p_denoised=p_denoised_samples, p_noisy=p_noisy_samples, log_lik=log_lik_samples)
end
 


"""
    _reconstruct(arch::MultifidelityArchitecture, mode::String, chain, M, PS, alpha)

Main reconstruction entry point for multi-fidelity models.

# Rationale for Update
This function is updated to call the refactored `_discover_component_realizations`
without the obsolete `p_names` argument, aligning it with the new explicit
component interface.
"""
function _reconstruct(arch::MultifidelityArchitecture, mode::String, chain, M::NamedTuple, PS, alpha::Float64)
    n_samples = size(chain, 1)
    N_tot = isnothing(PS) ? M.y_N : M.y_N + PS.y_N
    outcomes_N = M.outcomes_N

    # 1. Reconstruct the main model's components (excluding nested effects)
    main_registry = _discover_component_realizations(chain, M, PS, n_samples, outcomes_N, N_tot)
    
    # 2. Assemble the main model's base eta
    eta_main = _modular_eta_assembly(main_registry, M, PS, n_samples, outcomes_N)

    # 3. Reconstruct sub-models' etas and add them to the main eta
    nested_results = Dict{Symbol, Any}()
    if haskey(M, :nested_components)
        for (key, sub_M) in M.nested_components
            sub_PS = if !isnothing(PS) && haskey(PS, :nested_prediction_sets)
                get(PS.nested_prediction_sets, key, nothing)
            else
                nothing
            end

            sub_outcomes_N = get(sub_M, :outcomes_N, 1)
            sub_N_tot = isnothing(sub_PS) ? sub_M.y_N : sub_M.y_N + sub_PS.y_N
            sub_registry = _discover_component_realizations(chain, sub_M, sub_PS, n_samples, sub_outcomes_N, sub_N_tot)
            eta_sub = _modular_eta_assembly(sub_registry, sub_M, sub_PS, n_samples, sub_outcomes_N)

            rho_name = "rho_nested_$(key)"
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

    # 4. Apply correlation and generate predictions for the final main model
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


# Version 1.0.2 (2026-08-06)
# Purpose: Computes the Widely Applicable Information Criterion (WAIC).
# Rationale: This version corrects a `BoundsError` that occurred because the function
#            was incorrectly slicing the log-likelihood matrix. The original code
#            iterated over observations but sliced columns, leading to an out-of-bounds
#            access when the number of observations exceeded the number of samples.
#            The fix changes the slicing from `log_lik[:, i]` to `log_lik[i, :]` to
#            correctly compute statistics over the samples for each observation.
function _compute_waic(log_lik)
    # This function calculates the WAIC from a matrix of pointwise log-likelihoods.
    # The matrix is expected to have dimensions [n_observations, n_samples].

    # Ensure log_lik is a matrix.
    if !(log_lik isa AbstractMatrix)
        @warn "log_lik passed to _compute_waic is not a matrix. Returning NaN."
        return NaN
    end
    if isempty(log_lik)
        return 0.0
    end

    nobs, nsamples = size(log_lik)
    
    # lppd: log pointwise predictive density.
    # This is the sum over observations of the log of the mean likelihood for each observation.
    # The mean is taken over the posterior samples.
    # The original code `log_lik[:, i]` was incorrect as it sliced columns.
    # The correct approach is to slice rows `log_lik[i, :]`.
    lppd = sum(logsumexp(view(log_lik, i, :)) - log(nsamples) for i in 1:nobs)
    
    # p_waic: effective number of parameters.
    # This is the sum over observations of the variance of the log-likelihood for each observation.
    # The variance is taken over the posterior samples.
    p_waic = sum(var(view(log_lik, i, :)) for i in 1:nobs)
    
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



# Version 1.6.1 (2026-08-06)
# Purpose: The primary post-processing engine that generates comprehensive summaries,
#          diagnostics, and plots from a fitted bstm model and MCMC chain.
# Rationale: This version is updated to correctly handle metric calculation for multivariate
#            models. Instead of flattening all predictions, it computes RMSE and Pearson R
#            for each outcome variable separately and returns them as vectors.
#            This version also includes an incremented version number to reflect ongoing updates.
function model_results_comprehensive(model::DynamicPPL.Model, chain; au=nothing, data=nothing, alpha=0.05)
    # --- 1. Metadata and Architecture Extraction ---
    M = model.args.M
    y_obs = M.y_obs
    raw_arch = get(M, :model_arch, "univariate")

    arch_type = if raw_arch == "multivariate"
        MultivariateArchitecture()
    elseif raw_arch == "multifidelity"
        MultifidelityArchitecture()
    else
        UnivariateArchitecture()
    end

    # --- 2. Core Reconstruction ---
    # This calls the appropriate _reconstruct method based on the model architecture.
    # The `nothing` for PS indicates we are doing in-sample reconstruction.
    res = _reconstruct(arch_type, "model_results", chain, M, nothing, alpha)

    # --- 2.5 Post-Stratification Weight Calculation (if applicable) ---
    # This is done here because we need the raw denoised prediction samples, which are
    # returned by _reconstruct but not typically stored in the final summary.
    post_strat_weights = nothing 
    if hasproperty(res, :raw_predictions_denoised)
        samples_denoised = res.arch isa MultivariateArchitecture ? res.raw_predictions_denoised[1] : res.raw_predictions_denoised
        post_strat_weights = post_stratification_weights(res, M, nothing, samples_denoised)
    end

    # --- 3. Performance Metric Calculation ---
    # Correctly handle both univariate and multivariate cases for RMSE and Pearson R.
    pred_summary = res.predictions_denoised
    local rmse_val, r_pearson
    if arch_type isa MultivariateArchitecture
        rmse_val = Float64[]
        r_pearson = Float64[]
        for k in 1:M.outcomes_N
            y_obs_k = y_obs[:, k]
            y_pred_k = pred_summary[k].mean
            valid_idx_k = findall(x -> !isnan(x) && !isnothing(x), y_obs_k)
            if !isempty(valid_idx_k)
                obs_v = y_obs_k[valid_idx_k]
                pred_v = y_pred_k[valid_idx_k]
                push!(rmse_val, sqrt(mean((obs_v .- pred_v).^2)))
                try; push!(r_pearson, cor(obs_v, pred_v)); catch; push!(r_pearson, 0.0); end
            else
                push!(rmse_val, NaN); push!(r_pearson, NaN)
            end
        end
    else # Univariate
        y_pred = hasproperty(pred_summary, :mean) ? pred_summary.mean : []
        valid_idx = findall(x -> !isnan(x) && !isnothing(x), y_obs)
        if !isempty(valid_idx)
            obs_v = y_obs[valid_idx]
            pred_v = y_pred[valid_idx]
            rmse_val = sqrt(mean((obs_v .- pred_v).^2))
            try; r_pearson = cor(obs_v, pred_v); catch; r_pearson = 0.0; end
        else
            rmse_val = NaN; r_pearson = NaN
        end
    end

    # --- 4. MCMC Diagnostics ---
    mean_rhat, min_ess, sampling_time = 1.0, 0.0, 0.0 
    try
        chains_obj = MCMCChains.Chains(chain)
        df_stats = DataFrame(MCMCChains.summarize(chains_obj))
        if hasproperty(df_stats, :rhat); r_vals = filter(x -> !isnan(x) && x > 0, df_stats.rhat); mean_rhat = isempty(r_vals) ? 1.0 : mean(r_vals); end
        e_col = hasproperty(df_stats, :ess_bulk) ? :ess_bulk : (hasproperty(df_stats, :ess) ? :ess : nothing)
        if !isnothing(e_col); e_vals = filter(x -> !isnan(x) && x >= 0, df_stats[!, e_col]); min_ess = isempty(e_vals) ? 0.0 : minimum(e_vals); end
        if hasproperty(chain, :info) && haskey(chain.info, :stop_time); sampling_time = (chain.info.stop_time - chain.info.start_time); end
    catch e; @warn "MCMC diagnostic extraction failed: $e. Using default values."; end
    
    # --- 5. Plot Generation ---
    data_for_plots = isnothing(data) ? get(M, :data, nothing) : data
    plots = bstm_plots(res, M; au=au, data=data_for_plots)

    return (
        metrics = (rmse = rmse_val, r_pearson = r_pearson, ess = min_ess, rhat = mean_rhat, waic = get(res, :waic, 0.0), time = sampling_time),
        pstats = res,
        plots = plots,
        post_strat_weights = post_strat_weights
    )
end



# Version 1.0.1 (2026-08-06)
# Purpose: Generates a standard set of diagnostic and summary plots.
# Rationale: This version is updated to be more robust and general. Instead of relying on
#            hardcoded effect names (e.g., `:spatial_denoised`), it now iterates through the
#            components defined in the model configuration (`M.components`). For each component,
#            it uses the component's `structure` (e.g., `:spatial`, `:temporal`) to determine
#            the appropriate plot type. This ensures that any component included in the model
#            will be plotted correctly, resolving the "missing plots" issue.
function bstm_plots(res, M; au=nothing, data=nothing, outcome=1)
    plots = Dict{Symbol, Any}()
    effects = res.effects
    is_mv = res.arch isa MultivariateArchitecture
    
    y_obs = get(M, :y_obs, nothing)
    polygons = isnothing(au) ? nothing : get(au, :polygons, nothing)
    centroids = isnothing(au) ? nothing : get(au, :centroids, nothing)

    # --- 1. Posterior Predictive Check ---
    if hasproperty(res, :predictions_denoised)
        if isnothing(y_obs); @info "Skipping PPC plot: Observation data not found.";
        else
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

    # --- 2. Helper function for choropleth plots ---
    function _create_choropleth_plot(field_data, title_str, polygons, centroids)
        if isnothing(field_data) || !hasproperty(field_data, :mean); @info "Skipping spatial plot '$title_str': Data missing."; return nothing; end 
        if isnothing(polygons) && isnothing(centroids); @info "Skipping spatial plot '$title_str': No geometry provided."; return nothing; end
        s_mean = vec(collect(field_data.mean))
        if all(iszero, s_mean); @info "Skipping spatial plot '$title_str': Mean effect is zero."; return nothing; end
        if !isnothing(polygons) && length(polygons) >= length(s_mean); return plot_choropleth(s_mean, polygons; title=title_str);
        elseif !isnothing(centroids); return scatter(getindex.(centroids, 1), getindex.(centroids, 2), marker_z=s_mean, markersize=4, c=:viridis, label=nothing, title=title_str, aspect_ratio=:equal); end
        return nothing
    end

    # --- 3. Iterate through model components to generate plots ---
    for spec in M.components
        key = spec.key
        if !haskey(effects, key); continue; end

        component_effects = effects[key]
        
        # Plot the main effect (either 'noisy' or 'structured')
        main_effect_summary = if hasproperty(component_effects, :noisy)
            is_mv ? component_effects.noisy[outcome] : component_effects.noisy
        elseif hasproperty(component_effects, :structured)
            is_mv ? component_effects.structured[outcome] : component_effects.structured
        else
            continue
        end

        if isnothing(main_effect_summary) || !hasproperty(main_effect_summary, :mean) || all(iszero, main_effect_summary.mean)
            @info "Skipping plot for component '$key': Main effect is zero or data is missing."
            continue
        end

        if spec.structure == :spatial
            p = _create_choropleth_plot(main_effect_summary, "Spatial Effect: $key", polygons, centroids)
            if !isnothing(p); plots[Symbol("spatial_$(key)")] = p; end

            # Additionally, plot structured and unstructured parts if available (for BYM2)
            if hasproperty(component_effects, :structured)
                struct_summary = is_mv ? component_effects.structured[outcome] : component_effects.structured
                p_struct = _create_choropleth_plot(struct_summary, "Structured Effect: $key", polygons, centroids)
                if !isnothing(p_struct); plots[Symbol("structured_$(key)")] = p_struct; end
            end
            if hasproperty(component_effects, :unstructured)
                unstruct_summary = is_mv ? component_effects.unstructured[outcome] : component_effects.unstructured
                p_unstruct = _create_choropleth_plot(unstruct_summary, "Unstructured Effect: $key", polygons, centroids)
                if !isnothing(p_unstruct); plots[Symbol("unstructured_$(key)")] = p_unstruct; end
            end

        elseif spec.structure == :temporal
            tm, tl, tu = vec(main_effect_summary.mean), vec(main_effect_summary.lower), vec(main_effect_summary.upper)
            plots[Symbol("temporal_$(key)")] = plot(tm, ribbon=(tm .- tl, tu .- tm), title="Temporal Trend: $key", lw=2, fillalpha=0.2, color=:royalblue, legend=false, xlabel="Time Index")

        elseif spec.structure == :seasonal
            um, ul, uu = vec(main_effect_summary.mean), vec(main_effect_summary.lower), vec(main_effect_summary.upper)
            plots[Symbol("seasonal_$(key)")] = plot(um, ribbon=(um .- ul, uu .- um), title="Seasonal Component: $key", lw=2, fillalpha=0.2, color=:forestgreen, legend=false, xlabel="Period")

        elseif spec.structure == :smooth
            if isnothing(data); @info "Skipping smooth effect plot for '$key': `data` not provided."; continue; end
            
            var_sym = Symbol(spec.var)
            if hasproperty(data, var_sym)
                cov_data = data[!, var_sym]
                p_order = sortperm(cov_data)
                sm, sl, su = vec(main_effect_summary.mean), vec(main_effect_summary.lower), vec(main_effect_summary.upper)
                
                if !haskey(plots, :smooth_effects); plots[:smooth_effects] = Dict{Symbol, Any}(); end
                plots[:smooth_effects][var_sym] = plot(cov_data[p_order], sm[p_order], ribbon=(sm[p_order] .- sl[p_order], su[p_order] .- sm[p_order]), title="Smooth Effect: $var_sym", xlabel=string(var_sym), ylabel="Latent Effect", legend=false, color=:darkorange, fillalpha=0.2)
            end
        end
    end

    # --- 4. Fixed and Mixed Effects Plots ---
    # (This logic can remain similar as it already iterates through the effects object)
    if hasproperty(effects, :fixed_effects) && !isnothing(effects.fixed_effects)
        fe_summary = is_mv ? effects.fixed_effects[outcome] : effects.fixed_effects
        if hasproperty(fe_summary, :mean) && !all(iszero, fe_summary.mean) 
            fm, fl, fu = vec(fe_summary.mean), vec(fe_summary.lower), vec(fe_summary.upper)
            if !isempty(fm); coef_names = haskey(M, :Xfixed_names) ? string.(M.Xfixed_names) : ["Coef_$i" for i in 1:length(fm)]; p_forest = scatter(fm, 1:length(fm), xerror=(fm .- fl, fu .- fm), yticks=(1:length(fm), coef_names), title="Fixed Effects Coefficients", xlabel="Estimate", markersize=4, color=:black, legend=false); vline!(p_forest, [0], color=:red, ls=:dash, lw=1); plots[:fixed_effects] = p_forest; end
        end
    end

    if hasproperty(effects, :mixed_effects) && !isnothing(effects.mixed_effects)
        mixed_plots = Dict{Symbol, Any}()
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

    return NamedTuple(plots)
end



function predict(model_obj::DynamicPPL.Model, chain, new_data::DataFrame; n_samples::Int=100, alpha=0.05)
    # Purpose: The primary engine for projecting a fitted model onto new data.
    # Rationale: This function constructs a "prediction set" configuration (PS) that mirrors the training configuration (M)
    #            but is adapted for the `new_data`. It correctly handles the projection of fixed effects, smooth basis functions,
    #            and nested models.
    # v1.0.0 (2026-07-17)
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
    if haskey(M_train, :components)
        ps_basis_registry = Dict{Symbol, Any}()
        smooth_specs = filter(s -> s.structure == :smooth, M_train.components)
        
        for spec in smooth_specs
            key_sym = Symbol(spec.var)
            vars = get(spec.params, :positional_args, [])
            n_vars = length(vars)
            if haskey(M_train.basis_matrices, key_sym) && all(hasproperty(new_data, Symbol(v)) for v in vars)
                m_obj = spec.component_obj
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
    if haskey(M_train, :nested_components) && !isempty(M_train.nested_components)
        PS_dict[:nested_prediction_sets] = Dict{Symbol, Any}()
        for (key, sub_M) in M_train.nested_components
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

            if haskey(sub_M, :components)
                sub_ps_basis_registry = Dict{Symbol, Any}()
                sub_smooth_specs = filter(s -> s.structure == :smooth, sub_M.components)
                for spec in sub_smooth_specs
                    v_sym = Symbol(spec.var)
                    vars = get(spec.params, :positional_args, [])
                    n_vars = length(vars)
                    if haskey(sub_M.basis_matrices, v_sym) && all(hasproperty(new_data, Symbol(v)) for v in vars)
                        m_obj = spec.component_obj
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
                _resolve_obs_param!(sub_PS_dict, sub_lik_params, new_data, [:log_offsets], :log_offsets)
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
            ss_tot = sum((y_test_obs .- Statistics.mean(y_test_obs)).^2) # This can be zero if all test obs are the same. 
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

# Version 1.0.2 (2026-08-06)
# Purpose: A utility for performing Leave-One-Out Cross-Validation using Pareto Smoothed Importance
#          Sampling (PSIS-LOO) to assess a model's out-of-sample predictive accuracy.
# Rationale: This version corrects two issues related to the log-likelihood matrix dimensions.
#            1. The unpacking of `size(log_lik)` was incorrect, swapping `n_samples` and `n_obs`.
#            2. The `PosteriorStats.loo` function expects a matrix of size `[n_samples, n_obs]`,
#               but the `log_lik` matrix from `_reconstruct` has dimensions `[n_obs, n_samples]`.
#            The fix corrects the size unpacking and transposes the `log_lik` matrix before
#            passing it to `loo()`.
function bstm_loo(model_obj::DynamicPPL.Model, chain; alpha=0.05)    
    # --- 1. Metadata and Architecture Extraction ---
    # Rationale: M contains the configuration and technical registry required for reconstruction.
    M = model_obj.args.M
    raw_arch = get(M, :model_arch, "univariate")

    # --- 2. Technical Dispatch Resolution ---
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

    # --- 3. Latent Component Reconstruction for Likelihood Registry ---
    # Rationale: _reconstruct generates the [n_obs, n_samples] log-likelihood matrix.
    println("Audit: Recovering pointwise log-likelihood registry...")
    res = _reconstruct(arch_type, "loo_recovery", chain, M, nothing, alpha)

    # --- 4. Matrix Extraction and Validation ---
    # Rationale: Ensuring the log_likelihood matches the observation grid dimensions.
    log_lik = res.log_likelihood 
    if isempty(log_lik)
        @warn "Log-likelihood matrix is empty. Cannot compute LOO."
        return nothing
    end
    
    # FIX: The log_lik matrix is [n_obs, n_samples]. Unpack dimensions correctly.
    n_obs, n_samples = size(log_lik)

    println("Audit: Processing ", n_samples, " samples for ", n_obs, " observations.")

    # --- 5. PSIS-LOO Calculation via PosteriorStats ---
    # Rationale: LOO-CV provides a reliable estimate of out-of-sample predictive performance.
    # FIX: Transpose log_lik to [n_samples, n_obs] as expected by PosteriorStats.loo.
    loo_result = nothing
    try
        loo_result = loo(Matrix(log_lik'))
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


function compare_components(loo_a_report, loo_b_report; model_names=["Model_A", "Model_B"])    
    # Purpose: A utility for formal model comparison between two fitted `bstm` models. It uses 
    #          their PSIS-LOO results to compute the difference in Expected Log Pointwise 
    #          Predictive Density (ELPD) and provides a statistical basis for model selection.
    # Inputs: loo_a_report, loo_b_report, model_names.
    # Outputs: A NamedTuple containing the comparison table, ELPD difference, and LOO objects.

    println("--- Starting BSTM Component Comparison ---")

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
    println("\n--- BSTM Component Selection Registry ---")
    println("Model A (", model_names[1], "): ELPD = ", round(elpd_a, digits=2), " | p_loo = ", round(p_loo_a, digits=2))
    println("Model B (", model_names[2], "): ELPD = ", round(elpd_b, digits=2), " | p_loo = ", round(p_loo_b, digits=2))
    diff_elpd = elpd_a - elpd_b
    println("\nELPD Delta (A - B): ", round(diff_elpd, digits=2))

    if abs(diff_elpd) > 4.0
        winning_model = diff_elpd > 0 ? model_names[1] : model_names[2] 
        println("CONCLUSION: ", winning_model, " is statistically preferred based on predictive density.")
    else
        println("CONCLUSION: Competing component structures provide indistinguishable predictive density.")
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


 