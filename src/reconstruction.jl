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

 
# Version 1.3.0 (2026-08-20)
# Purpose: Finds a parameter name in a list based on the canonical target_name_base convention.
# Canonical signature: _find_parameter(p_names, target_name_base, outcome_idx=nothing, is_multivariate_model=false)
function _find_parameter(
    reg::ParamRegistry, target_name_base::Union{String, Symbol}, outcome_idx=nothing, is_multivariate_model::Bool=false
)
    return find_chain_param(reg, string(target_name_base); outcome_idx=outcome_idx)
end

function _find_parameter(
    p_names, target_name_base::Union{String, Symbol}, outcome_idx=nothing, is_multivariate_model::Bool=false
)
    base_str = string(target_name_base)
    p_names_str = string.(p_names)

    # Priority 1: Exact match with outcome index suffix (e.g., "sigma_s_idx_1").
    if is_multivariate_model && !isnothing(outcome_idx)
        specific_name_with_k_suffix = "$(base_str)_$(outcome_idx)"
        if specific_name_with_k_suffix in p_names_str
            return specific_name_with_k_suffix
        end
        
        indexed_name_with_k_bracket = "$(base_str)[$(outcome_idx)]"
        if indexed_name_with_k_bracket in p_names_str
            return indexed_name_with_k_bracket
        end
    end

    # Priority 2: Exact match for the base name
    if base_str in p_names_str
        return base_str
    end

    # Priority 3: Check for any bracketed indexed versions of the base name.
    re_indexed_any = Regex("^" * escape_string(base_str) * "\\[\\d+\\]")
    if any(n -> occursin(re_indexed_any, n), p_names_str)
        return base_str
    end

    # Priority 4: Alias transformations (e.g., ure <-> innovations, sre <-> latent/struct, rho_unconstrained <-> unconstrained_rho)
    alias_candidates = String[]
    if startswith(base_str, "ure_")
        push!(alias_candidates, replace(base_str, r"^ure_" => "innovations_"))
        push!(alias_candidates, replace(base_str, r"^ure_" => "innov_"))
        push!(alias_candidates, replace(base_str, r"^ure_" => "raw_"))
    elseif startswith(base_str, "innovations_")
        push!(alias_candidates, replace(base_str, r"^innovations_" => "ure_"))
    elseif startswith(base_str, "sre_")
        push!(alias_candidates, replace(base_str, r"^sre_" => "latent_"))
        push!(alias_candidates, replace(base_str, r"^sre_" => "struct_"))
    elseif startswith(base_str, "latent_")
        push!(alias_candidates, replace(base_str, r"^latent_" => "sre_"))
    elseif startswith(base_str, "struct_")
        push!(alias_candidates, replace(base_str, r"^struct_" => "sre_"))
    elseif occursin("unconstrained", base_str)
        push!(alias_candidates, replace(base_str, "rho_unconstrained" => "unconstrained_rho"))
        push!(alias_candidates, replace(base_str, "unconstrained_rho" => "rho_unconstrained"))
        push!(alias_candidates, replace(base_str, "sigma_unconstrained" => "unconstrained_sigma"))
        push!(alias_candidates, replace(base_str, "unconstrained_sigma" => "sigma_unconstrained"))
    elseif base_str == "beta"
        push!(alias_candidates, "Xfixed_beta_prop", "beta_prop", "Xfixed_beta")
    elseif base_str == "beta_flat"
        push!(alias_candidates, "Xfixed_beta_prop_flat", "beta_prop_flat")
    elseif base_str == "Xfixed_beta_prop"
        push!(alias_candidates, "beta")
    elseif base_str == "Xfixed_beta_prop_flat"
        push!(alias_candidates, "beta_flat")
    elseif base_str == "sigma_st_interaction"
        push!(alias_candidates, "st_interaction_sigma")
    elseif base_str == "ure_st_interaction"
        push!(alias_candidates, "st_interaction_raw")
    elseif base_str == "v_unscaled_reflection"
        push!(alias_candidates, "v_raw_reflection")
    end

    for alias in alias_candidates
        if is_multivariate_model && !isnothing(outcome_idx)
            s_k = "$(alias)_$(outcome_idx)"
            if s_k in p_names_str; return s_k; end
            s_b = "$(alias)[$(outcome_idx)]"
            if s_b in p_names_str; return s_b; end
        end
        if alias in p_names_str; return alias; end
        re_alias = Regex("^" * escape_string(alias) * "\\[\\d+\\]")
        if any(n -> occursin(re_alias, n), p_names_str); return alias; end
    end

    # Priority 5: Substring fallback
    for n in p_names_str
        if occursin(base_str, n)
            return n
        end
    end

    return "" # Return empty string if no match is found.
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

# Version 1.0.1 (2026-08-06)
# Purpose: Summarizes the posterior samples for all discovered component effects.
# Rationale:   create a more informative summary object. Instead of
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



function _discover_component_realizations(chain, M::NamedTuple, PS::Union{NamedTuple, Nothing}, n_samples::Int, outcomes_N::Int, N_tot::Int)
    p_names = string.(names(DataFrame(chain))) # Use names(DataFrame(chain)) for robustness

    # --- Intercept ---
    intercept_samples = zeros(Float64, n_samples, outcomes_N)
    if M.add_intercept
        for k in 1:outcomes_N
            param_name = (M.model_arch == "multivariate") ? "intercept_$(k)" : "intercept"
            found_name = _find_parameter(p_names, param_name, k, M.model_arch == "multivariate")
            if !isempty(found_name)
                intercept_samples[:, k] = get_params_vector(chain, found_name, 1)[:, 1]
            else
                @warn "Intercept parameter '$param_name' not found in the MCMC chain. Using zero for this effect."
            end
        end
    end

    # --- Fixed Effects ---
    fixed_effects_samples = zeros(Float64, M.Xfixed_N, n_samples, outcomes_N)
    if M.Xfixed_N > 0
        param_name_base = (M.model_arch == "multivariate") ? "beta_flat" : "beta"
        found_base = _find_parameter(p_names, param_name_base, nothing, M.model_arch == "multivariate")
        if !isempty(found_base)
            if M.model_arch == "multivariate"
                all_fixed_beta_flat = get_params_matrix(chain, found_base, M.Xfixed_N * outcomes_N)
                reshaped_fixed_beta = reshape(all_fixed_beta_flat, n_samples, M.Xfixed_N, outcomes_N)
                for k in 1:outcomes_N
                    fixed_effects_samples[:, :, k] = permutedims(reshaped_fixed_beta[:, :, k], (2, 1))
                end
            else # Univariate
                fixed_beta = get_params_matrix(chain, found_base, M.Xfixed_N)
                fixed_effects_samples[:, :, 1] = permutedims(fixed_beta, (2, 1))
            end
        else
            @warn "Fixed effects parameter '$param_name_base' not found in the MCMC chain. Using zero for this effect."
        end
    end

    # --- Component Effects ---
    component_realizations = Dict{Symbol, Any}()
    for spec in M.components
        effects_result = get_effects(spec.component_obj, chain, spec, M, PS)
        component_realizations[spec.key] = effects_result
    end

    # --- Spatiotemporal Interaction Effects ---
    st_interaction_effects_samples = zeros(Float64, M.s_N * M.t_N, n_samples, outcomes_N)
    if get(M, :model_st, "none") != "none"
        param_name_base = any(p -> occursin("ure_st_interaction", string(p)), p_names) ? "ure_st_interaction" : "st_interaction_raw"
        sigma_name_base = any(p -> occursin("sigma_st_interaction", string(p)), p_names) ? "sigma_st_interaction" : "st_interaction_sigma"
        has_param = any(p -> occursin(param_name_base, string(p)), p_names)
        has_sigma = any(p -> occursin(sigma_name_base, string(p)), p_names)
        if has_param && has_sigma
            if M.model_arch == "multivariate"
                sigma_samples = get_params_matrix(chain, sigma_name_base, outcomes_N)
                raw_samples = get_params_matrix(chain, param_name_base, M.s_N * M.t_N * outcomes_N)
                raw_samples_reshaped = reshape(raw_samples, n_samples, M.s_N * M.t_N, outcomes_N)
                for k in 1:outcomes_N
                    s_spec = M.components[findfirst(s -> s.structure == :spatial, M.components)]
                    t_spec = M.components[findfirst(s -> s.structure == :temporal, M.components)]
                    s_chol_factor = s_spec.hyper.cholesky_factor
                    t_chol_factor = t_spec.hyper.cholesky_factor
                    for i in 1:n_samples
                        Z_matrix = reshape(raw_samples_reshaped[i, :, k], M.s_N, M.t_N)
                        tmp_spatial = s_chol_factor.U \ Z_matrix
                        st_field_unscaled = transpose(t_chol_factor.U \ transpose(tmp_spatial))
                        st_field = st_field_unscaled .* sigma_samples[i, k]
                        st_interaction_effects_samples[:, i, k] = vec(st_field)
                    end
                end
            else # Univariate
                sigma_samples = get_params_vector(chain, sigma_name_base, 1)[:, 1]
                raw_samples = get_params_matrix(chain, param_name_base, M.s_N * M.t_N)
                s_spec = M.components[findfirst(s -> s.structure == :spatial, M.components)]
                t_spec = M.components[findfirst(s -> s.structure == :temporal, M.components)]
                s_chol_factor = s_spec.hyper.cholesky_factor
                t_chol_factor = t_spec.hyper.cholesky_factor
                for i in 1:n_samples
                    Z_matrix = reshape(raw_samples[i, :], M.s_N, M.t_N)
                    tmp_spatial = s_chol_factor.U \ Z_matrix
                    st_field_unscaled = transpose(t_chol_factor.U \ transpose(tmp_spatial))
                    st_field = st_field_unscaled .* sigma_samples[i]
                    st_interaction_effects_samples[:, i, 1] = vec(st_field)
                end
            end
        else
            @warn "Spatiotemporal interaction parameters not found. Using zero for this effect."
        end
    end

    # --- Householder Reflection Effects ---
    householder_effects_samples = zeros(Float64, outcomes_N, outcomes_N, n_samples)
    if get(M, :spectral_orientation, false) && M.model_arch == "multivariate"
        param_name = any(p -> occursin("v_unscaled_reflection", string(p)), p_names) ? "v_unscaled_reflection" : "v_raw_reflection"
        if any(p -> occursin(param_name, string(p)), p_names)
            v_reflection_samples = get_params_matrix(chain, param_name, outcomes_N)
            for i in 1:n_samples
                v_reflection = v_reflection_samples[i, :] / (norm(v_reflection_samples[i, :]) + 1e-9)
                householder_effects_samples[:, :, i] = I - 2.0 * v_reflection * v_reflection'
            end
        else
            @warn "Householder reflection parameter '$param_name' not found. Using identity matrix for reflection."
            for i in 1:n_samples
                householder_effects_samples[:, :, i] = Matrix(I, outcomes_N, outcomes_N)
            end
        end
    end

    return (
        intercept=intercept_samples,
        fixed_effects=fixed_effects_samples,
        components=component_realizations,
        st_interaction=st_interaction_effects_samples,
        householder_reflection=householder_effects_samples
    )
end



function _reconstruct(
    arch::UnivariateArchitecture, mode::String, chain, M::NamedTuple, PS,
    alpha::Float64
)
    # --- 1. Metadata and Dimension Discovery ---
    n_samples_val = if occursin("FlexiChain", string(typeof(chain)))
        size(chain, 1) * FlexiChains.nchains(chain)
    else
        size(chain, 1) * size(chain, 3) # For MCMCChains
    end
    N_tot_val = isnothing(PS) ? M.y_N : M.y_N + PS.y_N
    outcomes_N_val = 1 # For univariate, this is always 1.

    # --- 2. Latent Field Reconstruction ---
    registry = _discover_component_realizations(
        chain, M, PS, n_samples_val, outcomes_N_val, N_tot_val
    )

    # --- 3. Linear Predictor Assembly ---
    eta_post_3d = _modular_eta_assembly(registry, M, PS, n_samples_val, outcomes_N_val)
    
    # Extract the 2D matrix for the single outcome
    eta_post_2d = eta_post_3d[:, :, 1]

    # --- 4. Prediction and Log-Likelihood Calculation ---
    pred_results = _process_ll_and_predictions(
        eta_post_2d, chain, M, PS, outcomes_N_val, 1
    )

    # --- 5. Final Result Consolidation ---
    pstats = (
        effects=_summarize_effects_registry(registry, M, outcomes_N_val, alpha),
        predictions_denoised=summarize_predictions(pred_results.p_denoised; alpha=alpha),
        predictions_noisy=summarize_predictions(pred_results.p_noisy; alpha=alpha),
        raw_predictions_denoised=pred_results.p_denoised,
        raw_predictions_noisy=pred_results.p_noisy,
        log_likelihood=pred_results.log_lik,
        waic=_compute_waic(pred_results.log_lik),
        arch=arch
    )

    return pstats
end


function _reconstruct(
    arch::MultivariateArchitecture, mode::String, chain, M::NamedTuple, PS,
    alpha::Float64
)
    # --- 1. Metadata and Dimension Discovery ---
    n_samples_val = if occursin("FlexiChain", string(typeof(chain)))
        size(chain, 1) * FlexiChains.nchains(chain)
    else
        size(chain, 1) * size(chain, 3) # For MCMCChains
    end
    N_tot_val = isnothing(PS) ? M.y_N : M.y_N + PS.y_N
    outcomes_N_val = M.outcomes_N

    # --- 2. Latent Field Reconstruction ---
    registry = _discover_component_realizations(
        chain, M, PS, n_samples_val, outcomes_N_val, N_tot_val
    )
  
    # --- 3. Linear Predictor Assembly ---
    eta_latent_post = _modular_eta_assembly(registry, M, PS, n_samples_val, outcomes_N_val)

    # --- 4. Apply Correlation Structure ---
    L_corr_samples = get_params_matrix(chain, "L_corr", outcomes_N_val * outcomes_N_val)
    eta_post = similar(eta_latent_post)
    for s in 1:n_samples_val
        L_s = reshape(L_corr_samples[s, :], outcomes_N_val, outcomes_N_val)
        eta_post[:, s, :] = eta_latent_post[:, s, :] * L_s'
    end

    # --- 5. Prediction and Log-Likelihood Calculation ---
    all_pred_results = [
        _process_ll_and_predictions(eta_post[:,:,k], chain, M, PS, outcomes_N_val, k)
        for k in 1:outcomes_N_val
    ]
    
    p_denoised_summaries = [summarize_array(res.p_denoised, alpha=alpha) for res in all_pred_results]
    p_noisy_summaries = [summarize_array(res.p_noisy, alpha=alpha) for res in all_pred_results]
    raw_denoised = [res.p_denoised for res in all_pred_results]
    raw_noisy = [res.p_noisy for res in all_pred_results]
    all_log_lik = hcat([res.log_lik for res in all_pred_results]...)

    # --- 6. Final Result Consolidation ---
    pstats = (
        effects=_summarize_effects_registry(registry, M, outcomes_N_val, alpha),
        predictions_denoised=p_denoised_summaries,
        predictions_noisy=p_noisy_summaries,
        raw_predictions_denoised=raw_denoised,
        raw_predictions_noisy=raw_noisy,
        log_likelihood=all_log_lik,
        waic=_compute_waic(all_log_lik),
        arch=arch
    )

    return pstats
end


function _reconstruct(
    arch::MultifidelityArchitecture, mode::String, chain, M::NamedTuple, PS,
    alpha::Float64
)
    n_samples = if occursin("FlexiChain", string(typeof(chain)))
        size(chain, 1) * FlexiChains.nchains(chain)
    else
        size(chain, 1) * size(chain, 3) # For MCMCChains
    end
    N_tot = isnothing(PS) ? M.y_N : M.y_N + PS.y_N
    outcomes_N = M.outcomes_N

    # 1. Reconstruct the main model's components (excluding nested effects)
    main_registry = _discover_component_realizations(
        chain, M, PS, n_samples, outcomes_N, N_tot
    )
    
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
            sub_registry = _discover_component_realizations(
                chain, sub_M, sub_PS, n_samples, sub_outcomes_N, sub_N_tot
            )
            eta_sub = _modular_eta_assembly(sub_registry, sub_M, sub_PS, n_samples, sub_outcomes_N)

            rho_name = "rho_nested_$(key)"
            rho_samples = get_params_vector(chain, rho_name, 1)[:, 1]

            if size(eta_sub, 1) != N_tot
                @warn "Size mismatch between main model observations ($N_tot) and " *
                      "nested model '$(key)' observations ($(size(eta_sub, 1)))." *
                      " Cannot apply nested effect."
                continue
            end
            
            if outcomes_N > 1 || sub_outcomes_N > 1
                @warn "Multi-fidelity connection between multivariate models is not " *
                      "fully supported. Assuming a 1-to-1 outcome mapping." 
            end
            
            eta_main .+= reshape(rho_samples, 1, n_samples, 1) .* eta_sub

            sub_arch_raw = get(sub_M, :model_arch, "univariate")
            sub_arch_type = if sub_arch_raw == "multivariate"
                MultivariateArchitecture()
            else
                UnivariateArchitecture()
            end
            nested_results[key] = _reconstruct(
                sub_arch_type, mode, chain, sub_M, sub_PS, alpha
            )
        end
    end

    # 4. Apply correlation and generate predictions for the final main model
    eta_final = _apply_multivariate_correlation(eta_main, chain, outcomes_N)
    
    if outcomes_N > 1
        all_pred_results = [
            _process_ll_and_predictions(eta_final[:,:,k], chain, M, PS, outcomes_N, k)
            for k in 1:outcomes_N
        ]
        p_denoised_summaries = [summarize_array(res.p_denoised, alpha=alpha) for res in all_pred_results]
        p_noisy_summaries = [summarize_array(res.p_noisy, alpha=alpha) for res in all_pred_results]
        raw_denoised = [res.p_denoised for res in all_pred_results]
        raw_noisy = [res.p_noisy for res in all_pred_results]
        all_log_lik = hcat([res.log_lik for res in all_pred_results]...)
    else
        pred_results = _process_ll_and_predictions(
            eta_final[:,:,1], chain, M, PS, 1, 1
        )
        p_denoised_summaries = summarize_array(pred_results.p_denoised, alpha=alpha)
        p_noisy_summaries = summarize_array(pred_results.p_noisy, alpha=alpha)
        raw_denoised = pred_results.p_denoised
        raw_noisy = pred_results.p_noisy
        all_log_lik = pred_results.log_lik
    end

    summarized_effects = _summarize_effects_registry(
        main_registry, M, outcomes_N, alpha
    )
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





"""
    _modular_eta_assembly(registry, M, PS, n_samples, outcomes_N)

Assembles the full linear predictor (`eta`) from all discovered latent effects.

# Version
v1.6.0 (2026-08-19)

# Arguments
- `registry`: A NamedTuple containing raw posterior samples for each model component.
- `M`: The model configuration NamedTuple (for training data).
- `PS`: The prediction set configuration NamedTuple (for out-of-sample data), or `nothing`.
- `n_samples`: The total number of posterior samples.
- `outcomes_N`: The number of outcome variables.

# Returns
- A 3D array `eta_latent` of size `[N_total x n_samples x outcomes_N]`.
"""
function _modular_eta_assembly(registry, M, PS, n_samples, outcomes_N)
    N_tot = isnothing(PS) ? M.y_N : M.y_N + PS.y_N
    eta_latent = zeros(Float64, N_tot, n_samples, outcomes_N)

    # --- 1. Add Intercept ---
    # registry.intercept is [n_samples x outcomes_N]. Reshape to [1 x n_samples x outcomes_N] for broadcasting.
    eta_latent .+= reshape(registry.intercept, 1, n_samples, outcomes_N)

    # --- 2. Add Fixed Effects ---
    if haskey(registry, :fixed_effects) && size(registry.fixed_effects, 1) > 0
        Xfixed_full = if isnothing(PS)
            M.Xfixed
        else
            vcat(M.Xfixed, get(PS, :Xfixed, zeros(PS.y_N, size(M.Xfixed, 2))))
        end

        for k in 1:outcomes_N
            # Effect = X * beta_k
            eta_latent[:, :, k] .+= Xfixed_full * registry.fixed_effects[:, :, k]
        end
    end

    # Pre-compute full index vectors for spatial, temporal, and seasonal structures.
    # These map observations to their corresponding component units (e.g., spatial unit ID).
    s_idx_full = haskey(M, :s_idx) ? Array(isnothing(PS) || !haskey(PS, :s_idx) ? M.s_idx : vcat(M.s_idx, PS.s_idx)) : ones(Int, N_tot)
    t_idx_full = haskey(M, :t_idx) ? Array(isnothing(PS) || !haskey(PS, :t_idx) ? M.t_idx : vcat(M.t_idx, PS.t_idx)) : ones(Int, N_tot)
    u_idx_full = haskey(M, :u_idx) ? Array(isnothing(PS) || !haskey(PS, :u_idx) ? M.u_idx : vcat(M.u_idx, PS.u_idx)) : ones(Int, N_tot)

    # Iterate through all model components and add their effects to the linear predictor.
    for spec in M.components
        key = spec.key
        if !haskey(registry.components, key); continue; end
        
        effects = registry.components[key]
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
            elseif spec.structure in [:smooth, :interact, :nonstationaryvariance, :svc]
                # For smoothers, interactions, SVC, and non-stationary variance, the effect matrix
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
            offset_full = Array(isnothing(PS) ? M.log_offsets[:,k] : vcat(M.log_offsets[:,k], get(PS, :log_offsets, zeros(PS.y_N, outcomes_N))[:,k]))
            eta_latent[:, :, k] .+= offset_full
        end
    end

    return eta_latent
end




"""
    _process_ll_and_predictions(eta_samples, chain, M, PS, outcomes_N, k)

Generates predictions and log-likelihood values from the posterior `eta` samples.

# Version
v1.2.0 (2026-08-19)

# Arguments
- `eta_samples`: A matrix of posterior linear predictor samples `[N_obs, n_samples]`.
- `chain`: The MCMC chain object.
- `M`: The main model configuration.
- `PS`: The prediction set configuration, or `nothing`.
- `outcomes_N`: The number of outcomes.
- `k`: The index of the current outcome.

# Returns
- A `NamedTuple` containing `p_denoised`, `p_noisy`, and `log_lik` matrices.
"""

function _process_ll_and_predictions(eta_samples, chain, M, PS, outcomes_N, k)
    n_samples = size(eta_samples, 2)
    N_train = M.y_N
    N_pred = isnothing(PS) ? 0 : PS.y_N
    N_tot = N_train + N_pred

    y_obs_k = Array(outcomes_N > 1 ? M.y_obs[:, k] : M.y_obs)
    
    lik_spec = M.likelihood_specs[k]
    family = string(get(lik_spec, :family, "gaussian"))
    use_zi = get(M, :use_zi, false)
    phi_zi_samples = use_zi ? get_params_vector(chain, "lik_phi_zi", 1)[:,1] : zeros(n_samples)
    
    p_denoised_samples = similar(eta_samples)
    for s in 1:n_samples
        p_denoised_samples[:, s] = _apply_link_and_lik(
            family, view(eta_samples, :, s), use_zi, phi_zi_samples[s]
        )
    end

    p_noisy_samples = similar(eta_samples)
    log_lik_samples = zeros(Float64, N_train, n_samples)

    y_sigma_samples = if family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
        outcomes_N > 1 ? get_params_matrix(chain, "y_sigma", outcomes_N) : get_params_vector(chain, "y_sigma", 1)
    else
        ones(Float64, n_samples, outcomes_N)
    end

    r_nb_samples = if family == "negbin"
        outcomes_N > 1 ? get_params_matrix(chain, "r_nb", outcomes_N) : get_params_vector(chain, "r_nb", 1)
    else
        ones(Float64, n_samples, outcomes_N)
    end

    trials_full = Array(haskey(M, :trials) ? (isnothing(PS) ? M.trials[:,k] : vcat(M.trials[:,k], get(PS, :trials, ones(Int, PS.y_N, outcomes_N))[:,k])) : ones(Int, N_tot))
    
    for s in 1:n_samples
        phi_zi_s = phi_zi_samples[s, 1]
        y_sigma_s = y_sigma_samples[s, k]
        r_nb_s = r_nb_samples[s, k]
        
        # Explicitly loop to avoid broadcasting issues with keyword arguments.
        lik_obj_vec = [
            bstm_Likelihood(family, eta_samples[i, s]; phi_zi=phi_zi_s, r_nb=r_nb_s, sigma_y=y_sigma_s, trial=trials_full[i])
            for i in 1:N_tot
        ]
        
        p_noisy_samples[:, s] = rand.(lik_obj_vec)

        if N_train > 0
            lik_obj_train = view(lik_obj_vec, 1:N_train)
            log_lik_samples[:, s] = logpdf.(lik_obj_train, view(y_obs_k, 1:N_train))
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
            p_denoised_samples[i, :, s] = NNlib.softmax(eta_samples[i, s, :])
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
    lppd = sum(LogExpFunctions.logsumexp(view(log_lik, i, :)) - log(nsamples) for i in 1:nobs)
    
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
        mu = LogExpFunctions.logistic.(eta)
    else
        mu = eta 
    end
    if use_zi
        mu = (1.0 .- phi) .* mu
    end
    return mu
end
 
# Version 1.1.0 (2026-08-11)
# Purpose: Computes model-based post-stratification weights.
# Rationale: This version implements a model-based weighting scheme where the weight for
#            an observation is the ratio of the mean prediction within its stratum to the
#            prediction for the observation itself. This method is intended to adjust
#            individual predictions based on their stratum's average behavior.
#            NOTE: This is a departure from traditional post-stratification weights, which
#            are typically calculated as `Area(j) / n_obs_in_stratum(j)` to scale sample
#            densities to population totals. This new implementation directly follows the user's
#            request to base weights on prediction ratios.
function post_stratification_weights(res, M, PS, samples_denoised)
    # Assumptions:
    #   1. The model configuration `M` contains the spatial index vector `:s_idx`.
    # Inputs:
    #   - res: The main results object (not used in this implementation).
    #   - M: The model configuration object for the training data.
    #   - PS: The prediction set configuration object (can be `nothing`).
    #   - samples_denoised: A matrix of posterior predictions [n_obs x n_samples].
    # Outputs: A matrix of weights of the same size as `samples_denoised`.

    # #
    # Input validation
    if !haskey(M, :s_idx)
        @warn "Post-stratification requires a spatial index `:s_idx` in the model configuration. Returning ones."
        return ones(Float64, size(samples_denoised))
    end

    # #
    # Combine stratum IDs from training and prediction sets
    strata_ids_train = M.s_idx

    strata_ids_full = if !isnothing(PS)
        if !haskey(PS, :s_idx)
            @warn "Prediction set provided but is missing spatial index `:s_idx`. Post-stratification weights will only be calculated for training data."
            strata_ids_train
        else
            vcat(strata_ids_train, PS.s_idx)
        end
    else
        strata_ids_train
    end

    n_obs_total, n_samples = size(samples_denoised)
    
    # Ensure the number of observations in samples_denoised matches the number of stratum IDs
    if n_obs_total != length(strata_ids_full)
        @error "Dimension mismatch: `samples_denoised` has $(n_obs_total) observations, but there are $(length(strata_ids_full)) stratum IDs. Cannot compute weights."
        return ones(Float64, size(samples_denoised))
    end

    weights = zeros(Float64, n_obs_total, n_samples)
    unique_strata = unique(strata_ids_full)

    # #
    # Calculate weights based on the ratio of stratum-mean prediction to observation-level prediction
    for stratum in unique_strata
        # Find indices of observations in the current stratum
        obs_indices_in_stratum = findall(x -> x == stratum, strata_ids_full)
        
        if isempty(obs_indices_in_stratum)
            continue
        end

        # Get the predictions for this stratum
        predictions_in_stratum = view(samples_denoised, obs_indices_in_stratum, :)

        # Calculate the mean prediction for the stratum for each posterior sample
        # This results in a row vector of size [1 x n_samples]
        mean_pred_per_sample = mean(predictions_in_stratum, dims=1)

        # Calculate weights for each observation in the stratum.
        # Weight = mean_pred_stratum / pred_observation
        # This uses broadcasting to divide each element in predictions_in_stratum
        # by the corresponding column mean in mean_pred_per_sample.
        # A small epsilon is added to the denominator to prevent division by zero.
        weights[obs_indices_in_stratum, :] = mean_pred_per_sample ./ (predictions_in_stratum .+ 1e-9)
    end

    return weights
end
 

"""
    model_results_comprehensive(model::DynamicPPL.Model, chain; ...)

The primary post-processing engine that generates comprehensive summaries,
diagnostics, and plots from a fitted `bstm` model and MCMC chain.
"""
function model_results_comprehensive(model::DynamicPPL.Model, chain; au=nothing, data=nothing, alpha=0.05)
    # --- No initial chain standardization ---
    
    n_samples = if occursin("FlexiChain", string(typeof(chain)))
        size(chain, 1) * FlexiChains.nchains(chain)
    else
        size(chain, 1) * size(chain, 3)
    end
    
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
    # Pass the original chain object directly to the reconstruction engine.
    res = _reconstruct(arch_type, "model_results", chain, M, nothing, alpha) # n_samples is now calculated inside _reconstruct

    # --- 2.5 Post-Stratification Weight Calculation (if applicable) ---
    post_strat_weights = nothing 

    local M_for_post_strat = M
    if !haskey(M, :strata_info) && !isnothing(au) && hasproperty(au, :strata_info)
        M_for_post_strat = merge(M, (strata_info=au.strata_info,))
    end

    if hasproperty(res, :raw_predictions_denoised)
        samples_denoised = res.arch isa MultivariateArchitecture ? res.raw_predictions_denoised[1] : res.raw_predictions_denoised
        post_strat_weights = post_stratification_weights(res, M_for_post_strat, nothing, samples_denoised)
    end

    # --- 3. Performance Metric Calculation ---
    pred_summary = res.predictions_denoised
    local rmse_val, r_pearson
    if arch_type isa MultivariateArchitecture
        rmse_val = Float64[]
        r_pearson = Float64[]
        for k in 1:M.outcomes_N
            y_obs_k = Array(y_obs[:, k]) # Ensure y_obs is on CPU
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
        y_obs_cpu = Array(y_obs) # Ensure y_obs is on CPU
        valid_idx = findall(x -> !isnan(x) && !isnothing(x), y_obs_cpu)
        if !isempty(valid_idx)
            obs_v = y_obs_cpu[valid_idx]
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
        # This block now handles both FlexiChains and MCMCChains
        if occursin("FlexiChain", string(typeof(chain)))
            df_stats = DataFrame(StatsBase.summarystats(MCMCChains.Chains(chain)))
        else
            df_stats = DataFrame(StatsBase.summarystats(chain))
        end

        if hasproperty(df_stats, :rhat); r_vals = filter(x -> !isnan(x) && x > 0, df_stats.rhat); mean_rhat = isempty(r_vals) ? 1.0 : mean(r_vals); end
        e_col = hasproperty(df_stats, :ess_bulk) ? :ess_bulk : (hasproperty(df_stats, :ess) ? :ess : nothing)
        if !isnothing(e_col); e_vals = filter(x -> !isnan(x) && x >= 0, df_stats[!, e_col]); min_ess = isempty(e_vals) ? 0.0 : minimum(e_vals); end
        if hasproperty(chain, :info) && haskey(chain.info, :stop_time); sampling_time = (chain.info.stop_time - chain.info.start_time); end
    catch e; @warn "MCMC diagnostic extraction failed: $e. Using default values."; end
    
    # --- 5. Plot Generation ---
    data_for_plots = isnothing(data) ? get(M, :data, nothing) : data
    # Pass the original model and chain to bstm_plots
    plot_results = bstm_plots(model, chain, res, M; au=au, data=data_for_plots)

    # --- 6. Final Assembly ---
    pstats_final = merge(res, (post_strat_weights=post_strat_weights,))

    return (
        metrics = (rmse = rmse_val, r_pearson = r_pearson, ess = min_ess, rhat = mean_rhat, waic = get(res, :waic, 0.0), time = sampling_time),
        pstats = pstats_final,
        plots = plot_results.plots,
        plots_data = plot_results.plots_data
    )
end



# New helper function for generating conditional predictions
function _generate_conditional_predictions(model_obj, chain, M, target_cov::Symbol;
                                           second_cov::Union{Symbol, Nothing}=nothing,
                                           n_points::Int=50, alpha::Float64=0.05)
    
    # Create a base DataFrame for prediction by taking the first row of the original data
    # and replicating it. Then, set other covariates to their mean/mode.
    base_df = DataFrame(M.data[1:1, :])
    for col in names(M.data)
        col_sym = Symbol(col)
        if col_sym != target_cov && (isnothing(second_cov) || col_sym != second_cov)
            if eltype(M.data[!, col_sym]) <: Number
                base_df[1, col_sym] = mean(M.data[!, col_sym])
            else # Categorical
                # For categorical, pick the mode or first level
                # Ensure the column type is preserved
                if isempty(levels(M.data[!, col_sym]))
                    @warn "Categorical covariate $(col_sym) has no levels. Skipping setting default value."
                    continue
                end
                base_df[1, col_sym] = first(levels(M.data[!, col_sym]))
            end
        end
    end

    if isnothing(second_cov) # 1D conditional plot
        # Generate a range for the target covariate
        if eltype(M.data[!, target_cov]) <: Number
            min_val, max_val = extrema(M.data[!, target_cov])
            cov_range = collect(range(min_val, stop=max_val, length=n_points))
            pred_df = vcat([base_df for _ in 1:n_points]...)
            pred_df[!, target_cov] = cov_range
        else # Categorical
            unique_levels = unique(M.data[!, target_cov])
            n_levels = length(unique_levels)
            pred_df = vcat([base_df for _ in 1:n_levels]...)
            pred_df[!, target_cov] = unique_levels
        end
        
        # Generate predictions
        preds = predict(model_obj, chain, pred_df; n_samples=size(chain, 1), alpha=alpha)
        return (preds.predictions_denoised, pred_df[!, target_cov])

    else # 2D conditional plot (interaction)
        if !(eltype(M.data[!, target_cov]) <: Number && eltype(M.data[!, second_cov]) <: Number)
            @warn "2D conditional plots are currently only supported for two continuous covariates."
            return nothing
        end

        min_val1, max_val1 = extrema(M.data[!, target_cov])
        min_val2, max_val2 = extrema(M.data[!, second_cov])
        range1 = collect(range(min_val1, stop=max_val1, length=n_points))
        range2 = collect(range(min_val2, stop=max_val2, length=n_points))

        pred_df_rows = DataFrame()
        for val1 in range1
            for val2 in range2
                row = deepcopy(base_df)
                row[1, target_cov] = val1
                row[1, second_cov] = val2
                append!(pred_df_rows, row)
            end
        end
        
        preds = predict(model_obj, chain, pred_df_rows; n_samples=size(chain, 1), alpha=alpha)
        return (preds.predictions_denoised, range1, range2)
    end
end





"""
    predict(model_obj::DynamicPPL.Model, chain, new_data::DataFrame; n_samples::Int=100, alpha=0.05)

The primary engine for projecting a fitted `bstm` model onto a new dataset to
generate out-of-sample predictions.

# Version
v1.1.0 (2026-08-13)

# Rationale
This function constructs a "prediction set" configuration (`PS`) that mirrors the
training configuration (`M`) but is adapted for the `new_data`. It correctly
handles the projection of fixed effects, the re-generation of basis matrices for
smoothers on the new coordinates, and the recursive prediction for nested models.
  consistent with the refactored architecture,
correcting `MethodError` exceptions that arose from changes in the signatures of
`decompose_bstm_formula` and `create_fixed_design`.

# Workflow
1.  **Initialize Prediction Set (PS)**: Creates a copy of the training model's
    configuration (`M`) and updates it with the `new_data`.
2.  **Re-create Fixed Effects**: Parses the original formula and calls
    `create_fixed_design` to generate a new design matrix for the fixed effects
    based on the `new_data`.
3.  **Update Indices**: Populates the spatial, temporal, and seasonal index vectors
    in the `PS` from the corresponding columns in `new_data`.
4.  **Re-create Basis Matrices**: For any smoother components in the original model,
    it regenerates the basis matrices (e.g., for P-splines, thin-plate splines)
    using the coordinate data from `new_data`.
5.  **Handle Nested Models**: If the original model contained `nested()` components,
    it recursively creates a prediction set for each sub-model.
6.  **Call Reconstruction Engine**: Invokes the main `_reconstruct` function, passing
    it both the training configuration `M` and the new prediction set `PS`.
7.  **Slice Predictions**: Extracts and returns only the out-of-sample portion of
    the predictions from the full reconstructed output.

# Arguments
- `model_obj::DynamicPPL.Model`: The fitted Turing model object.
- `chain`: The `MCMCChains.Chains` object from the fitted model.
- `new_data::DataFrame`: A `DataFrame` with the same column names as the training data.
- `n_samples::Int`: The number of posterior samples to use for prediction.
- `alpha::Float64`: The significance level for credible intervals.

# Returns
- A `NamedTuple` containing the summarized `predictions_denoised` and
  `predictions_noisy`, the full posterior statistics object `pstats`, and the
  prediction set configuration `PS`.
"""
function predict(model_obj::DynamicPPL.Model, chain, new_data::DataFrame; n_samples::Int=100, alpha=0.05)
    M_train = model_obj.args.M
    n_samps = min(size(chain, 1), n_samples)

    # 1. Initialize the Prediction Set (PS) configuration
    PS_dict = Dict(pairs(M_train))
    PS_dict[:data] = new_data
    PS_dict[:y_obs] = zeros(nrow(new_data)) # Placeholder
    PS_dict[:y_N] = nrow(new_data)

    # 2. Re-create fixed effects design matrix for the new data
    if haskey(M_train, :formula)
        # Corrected call: Pass training data for context.
        decomposed_formula = decompose_bstm_formula(M_train.formula, M_train.data)
        
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
            # Corrected call: Pass calling_module for scoped evaluation.
            Xfixed_pred, _ = create_fixed_design(
                rhs, 
                new_data, 
                M_train.calling_module; 
                contrasts=get(M_train, :contrasts, Dict())
            )
            PS_dict[:Xfixed] = Matrix(Xfixed_pred)
            PS_dict[:Xfixed_N] = size(Xfixed_pred, 2)
            PS_dict[:Xfixed_names] = names(Xfixed_pred, 2)
        end
    end

    # 3. Update indices from new_data
    if haskey(M_train, :s_idx_var) && hasproperty(new_data, M_train.s_idx_var); PS_dict[:s_idx] = new_data[!, M_train.s_idx_var]; end 
    if haskey(M_train, :t_idx_var) && hasproperty(new_data, M_train.t_idx_var); PS_dict[:t_idx] = new_data[!, M_train.t_idx_var]; end 
    if haskey(M_train, :u_idx_var) && hasproperty(new_data, M_train.u_idx_var); PS_dict[:u_idx] = new_data[!, M_train.u_idx_var]; end 

    # 4. Re-create basis matrices for smoothers on the new data
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
                
                # Correctly pass keyword arguments from the spec's params dictionary.
                local_kwargs = Dict(spec.params)

                B_matrix, _ = if n_vars == 1 
                    bstm_smooth_basis_1D(model_type_str, new_data[!, Symbol(vars[1])], nb; local_kwargs...)
                elseif n_vars > 1
                    coords_new = Matrix{Float64}(new_data[!, Symbol.(vars)])
                    B_matrix_raw = if n_vars == 2; bstm_smooth_basis_2D(model_type_str, coords_new, nb; local_kwargs...);
                    elseif n_vars == 3; bstm_smooth_basis_3D(model_type_str, coords_new, nb; local_kwargs...);
                    elseif n_vars == 4; bstm_smooth_basis_4D(model_type_str, coords_new, nb; local_kwargs...);
                    else; error("Smoothers with more than 4 dimensions are not supported for prediction."); end
                    (B_matrix_raw, size(B_matrix_raw, 2))
                end
                ps_basis_registry[key_sym] = B_matrix
            end
        end
        PS_dict[:basis_matrices] = ps_basis_registry
    end

    # 5. Create prediction sets for nested sub-models
    if haskey(M_train, :nested_components) && !isempty(M_train.nested_components)
        PS_dict[:nested_prediction_sets] = Dict{Symbol, Any}()
        for (key, sub_M) in M_train.nested_components
            sub_PS_dict = Dict(pairs(sub_M))
            sub_PS_dict[:data] = new_data
            sub_PS_dict[:y_obs] = zeros(nrow(new_data))
            sub_PS_dict[:y_N] = nrow(new_data)

            if haskey(sub_M, :formula)
                # Corrected call for sub-model
                sub_decomposed = decompose_bstm_formula(sub_M.formula, sub_M.data) 
                
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
                    # Corrected call for sub-model
                    Xfixed_sub, _ = create_fixed_design(
                        rhs, 
                        new_data, 
                        sub_M.calling_module; 
                        contrasts=get(sub_M, :contrasts, Dict())
                    )
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
                sub_ps_basis_registry = _recreate_basis_matrices_for_prediction(sub_M, new_data)
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

    # 6. Finalize PS and call reconstruction
    PS = NamedTuple(PS_dict)

    raw_arch = get(M_train, :model_arch, "univariate")
    arch_type = if raw_arch == "multivariate"; MultivariateArchitecture()
    elseif raw_arch == "multifidelity"; MultifidelityArchitecture()
    else; UnivariateArchitecture(); end 

    chain_sub = chain[1:min(n_samps, end), :, :]

    res = _reconstruct(arch_type, "prediction", chain_sub, M_train, PS, alpha)

    # 7. Slice the prediction part from the full summary.
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



"""
    bstm_cv_orchestrator(formula::String, data::DataFrame; ...)

An orchestration utility for performing cross-validation on `bstm` models. It
supports several strategies designed to handle the dependent nature of
spatiotemporal data.

# Version
v1.1.0 (2026-08-13)

# Rationale
This function provides a standardized and flexible way to evaluate a model's
out-of-sample predictive performance.   consistent
with the refactored `bstm` architecture, replacing the deprecated model
instantiation logic with a call to the new `bstm_core` function. It also
corrects a bug in the formula parsing step.

# Cross-Validation Methods (`method`)
- **`:kfold`**: Standard random k-fold cross-validation. Suitable only when
  observations can be considered independent.
- **`:lolo` (Leave-One-Location-Out)**: Each fold consists of all observations
  from a unique level of the `cv_var` (e.g., a spatial unit `s_idx`). This
  tests the model's ability to predict at entirely new locations.
- **`:spatial_block`**: Creates `n_folds` spatial blocks using k-means
  clustering on the `cv_space_vars`. This tests spatial extrapolation performance.
- **`:temporal_block`**: Divides the data into `n_folds` contiguous blocks based
  on the `cv_var` (e.g., `year`). This tests interpolation performance for
  missing time periods.
- **`:temporal_forward_chain`**: A forecasting simulation. It iteratively trains
  on data up to a certain time point and tests on the next `n_folds` time points.

# Arguments
- `formula::String`: The `bstm` model formula.
- `data::DataFrame`: The full dataset.
- `method::Symbol`: The CV strategy to use. Default: `:kfold`.
- `cv_var::Symbol`: The column in `data` used for grouping/blocking. Default: `:s_idx`.
- `n_folds::Int`: The number of folds or blocks. Default: `5`.
- `sampler`: The Turing sampler used to fit the model in each fold. Default: `NUTS(500, 0.65)`.
- `n_samples::Int`: The number of posterior samples to draw in each fold. Default: `500`.
- `alpha::Float64`: The significance level for credible intervals in prediction. Default: `0.05`.
- `cv_space_vars::Vector{Symbol}`: Coordinate columns for `:spatial_block`. Default: `[:s_x, :s_y]`.
- `kwargs...`: Additional keyword arguments passed to the underlying `bstm_core` call.

# Returns
- A `NamedTuple` containing:
  - `folds`: A vector of `NamedTuple`s with `rmse` and `r2` for each fold.
  - `mean_rmse`: The average RMSE across all folds.
  - `mean_r2`: The average R-squared across all folds.
  - `response_var`: The name of the response variable.
  - `method`: The CV method used.
  - `n_folds`: The number of folds executed.
"""
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
    # Corrected call to include the data argument for formula parsing.
    meta_discovery = decompose_bstm_formula(formula, data)
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

        # Updated model instantiation to use bstm_core, consistent with refactor.
        # Pass kwargs through, but force verbose=false to avoid excessive output.
        cv_kwargs = Dict(kwargs)
        cv_kwargs[:verbose] = false
        
        model_train = bstm_core(formula, train_data; cv_kwargs...)
        
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
"""
    bstm_loo(model_obj::DynamicPPL.Model, chain; alpha=0.05)

A utility for performing Leave-One-Out Cross-Validation using Pareto Smoothed
Importance Sampling (PSIS-LOO) to assess a model's out-of-sample predictive accuracy.

# Version
v1.0.3 (2026-08-13)

# Rationale
This function serves as a high-level wrapper around the `PosteriorStats.loo`
function. It is a critical tool for model selection, providing a more robust
estimate of out-of-sample predictive performance than simpler metrics like WAIC.

This version is confirmed to be correct and consistent with the refactored `bstm`
architecture. It correctly handles the dimensional requirements of the underlying
`PosteriorStats.loo` function by transposing the log-likelihood matrix.

# Workflow
1.  **Architecture Dispatch**: Determines the model architecture (univariate,
    multivariate, etc.) from the model configuration.
2.  **Log-Likelihood Reconstruction**: Calls the internal `_reconstruct` function
    to generate the pointwise log-likelihood matrix, which has dimensions
    `[n_observations, n_samples]`.
3.  **Matrix Transposition**: Transposes the log-likelihood matrix to the
    `[n_samples, n_observations]` format required by `PosteriorStats.loo`.
4.  **PSIS-LOO Calculation**: Calls `PosteriorStats.loo` to compute the LOO-CV metrics.
5.  **Reporting**: Prints a summary of the key metrics (ELPD, p_loo, LOOIC) and
    warns the user if any influential observations (high Pareto-k values) are detected.

# Arguments
- `model_obj::DynamicPPL.Model`: The fitted Turing model object.
- `chain`: The `MCMCChains.Chains` object from the fitted model.
- `alpha::Float64`: The significance level for credible intervals (not directly
  used in LOO calculation but maintained for API consistency).

# Returns
- A `NamedTuple` containing:
  - `loo_obj`: The full results object from `PosteriorStats.loo`.
  - `metrics`: A `NamedTuple` with the key estimates (`elpd`, `p_loo`, `looic`).
  - `log_likelihood`: The original `[n_obs, n_samples]` log-likelihood matrix.
  - `pareto_k`: A vector of the Pareto-k diagnostic values for each observation.
"""
function bstm_loo(model_obj::DynamicPPL.Model, chain; alpha=0.05)    
    # --- 1. Metadata and Architecture Extraction ---
    M = model_obj.args.M
    raw_arch = get(M, :model_arch, "univariate")

    # --- 2. Technical Dispatch Resolution ---
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
    res = _reconstruct(arch_type, "loo_recovery", chain, M, nothing, alpha)

    # --- 4. Matrix Extraction and Validation ---
    log_lik = res.log_likelihood 
    if isempty(log_lik)
        @warn "Log-likelihood matrix is empty. Cannot compute LOO."
        return nothing
    end
    
    n_obs, n_samples = size(log_lik)

    # --- 5. PSIS-LOO Calculation via PosteriorStats ---
    # PosteriorStats.loo expects a matrix of size [n_samples, n_obs].
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


"""
    compare_models(loo_a_report, loo_b_report; model_names=["Model_A", "Model_B"])

A utility for formal model comparison between two fitted `bstm` models. It uses
their PSIS-LOO results to compute the difference in Expected Log Pointwise
Predictive Density (ELPD) and provides a statistical basis for model selection.

# Rationale
This function is updated to be consistent with the refactored `bstm` framework,
which uses the term "component" instead of the deprecated "manifold". The function
name and internal print statements have been updated accordingly. The core logic,
which relies on `PosteriorStats.compare`, remains unchanged as it is correct.

# Arguments
- `loo_a_report`, `loo_b_report`: The output `NamedTuple` from `bstm_loo` for each model.
- `model_names`: A vector of strings with names for the models being compared.

# Returns
- A `NamedTuple` containing the comparison table, ELPD difference, and the original LOO objects.
"""
function compare_models(loo_a_report, loo_b_report; model_names=["Model_A", "Model_B"])
    println("--- Starting BSTM Model Comparison ---")

    # 1. LOO Object Extraction
    loo_a = loo_a_report.loo_obj
    loo_b = loo_b_report.loo_obj

    # 2. Formal Selection Metric Calculation
    comparison_stats = nothing
    try
        comparison_stats = compare([loo_a, loo_b])
    catch e
        @error "BSTM Comparison Error: Selection suite failed. Error: " * string(e)
        return nothing
    end

    # 3. Parameter and Diagnostic Extraction
    p_loo_a = loo_a_report.metrics.p_loo
    p_loo_b = loo_b_report.metrics.p_loo
    elpd_a = loo_a_report.metrics.elpd
    elpd_b = loo_b_report.metrics.elpd

    # 4. Report Generation
    println("\n--- BSTM Model Selection Registry ---")
    println("Model A (", model_names[1], "): ELPD = ", round(elpd_a, digits=2), " | p_loo = ", round(p_loo_a, digits=2))
    println("Model B (", model_names[2], "): ELPD = ", round(elpd_b, digits=2), " | p_loo = ", round(p_loo_b, digits=2))
    diff_elpd = elpd_a - elpd_b
    println("\nELPD Delta (A - B): ", round(diff_elpd, digits=2))

    if abs(diff_elpd) > 4.0
        winning_model = diff_elpd > 0 ? model_names[1] : model_names[2]
        println("CONCLUSION: ", winning_model, " is statistically preferred based on predictive density.")
    else
        println("CONCLUSION: Competing models provide indistinguishable predictive density.")
    end

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

