#=
File: reconstruction_engine.jl
Version: 1.0.0
Timestamp: 2026-06-30 11:30:00

Description:
This file contains the complete and audited versions of the `_reconstruct` methods
for all model architectures supported by the `bstm` framework. These functions are
responsible for post-processing MCMC chains to produce comprehensive and standardized
result summaries.

This file was created to resolve inconsistencies and failed updates in the previous
implementation, ensuring that all architectures correctly report all latent effects,
including spatial components, mixed effects, and nested contributions.
=#


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
    if !isnothing(registry.basis_eff_accum)
        summarized_effects[:smooth_effects] = summarize_array(registry.basis_eff_accum[1:M.y_N, :]; alpha=alpha)
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