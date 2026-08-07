
 
 
function _generate_component_code_fragments(m::Union{TensorProductSmooth, TPS, BSpline, PSpline}, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    return _generate_component_code_fragments(m, spec, arch, outcome_idx, M, prefix=prefix)
end



# Version 1.0.8 (2026-08-06)
# Purpose: Generates Turing code for multivariate dynamics models.
# Rationale: This version is updated for AD compatibility. It converts the sparse precision
#            matrix `Q_spatial` to a dense `Matrix` before the Cholesky decomposition.
#            This avoids calling the sparse Cholesky factorization from CHOLMOD, which does
#            not support Dual types. It also replaces the incorrect use of `.U` with the
#            correct `.L'` for solving with the Cholesky factor, resolving the `CanonicalIndexError`.
function _generate_multivariate_dynamics_code(m::DynamicsComponent, spec::NamedTuple, M::NamedTuple)
    key_str = string(spec.key)
    params = m.params
    model_type = m.model
    prefixed_key = key_str
    n_species = M.outcomes_N # For generalized_lotka_volterra

    if model_type == "leslie_matrix"
        n_age_classes = get(params, :n_age_classes, M.outcomes_N)
        spatially_varying_K = get(params, :spatially_varying_K, false)
        spatially_varying_rates = get(params, :spatially_varying_rates, false)

        if n_age_classes != M.outcomes_N; error("Number of age classes ($n_age_classes) must match number of outcomes ($(M.outcomes_N))."); end

        priors_acc = []
        if spatially_varying_rates
            push!(priors_acc, "log_fecundity_mean_$(key_str) ~ NamedDist(filldist(Normal(0, 1), $(n_age_classes)), :log_fecundity_mean_$(key_str))")
            push!(priors_acc, "sigma_fecundity_$(key_str) ~ NamedDist(filldist(Exponential(1.0), $(n_age_classes)), :sigma_fecundity_$(key_str))")
            push!(priors_acc, "fecundity_raw_$(key_str) ~ NamedDist(MvNormal(zeros(T, M.s_N * $(n_age_classes)), I), :fecundity_raw_$(key_str))")
            push!(priors_acc, "logit_survival_mean_$(key_str) ~ NamedDist(filldist(Normal(1.5, 1), $(n_age_classes-1)), :logit_survival_mean_$(key_str))")
            push!(priors_acc, "sigma_survival_$(key_str) ~ NamedDist(filldist(Exponential(1.0), $(n_age_classes-1)), :sigma_survival_$(key_str))")
            push!(priors_acc, "survival_raw_$(key_str) ~ NamedDist(MvNormal(zeros(T, M.s_N * ($(n_age_classes)-1)), I), :survival_raw_$(key_str))")
        else
            push!(priors_acc, "survival_rates_$(key_str) ~ NamedDist(filldist(Beta(9, 1), $(n_age_classes - 1)), :survival_rates_$(key_str))")
            push!(priors_acc, "fecundity_rates_$(key_str) ~ NamedDist(filldist(LogNormal(0, 1), $(n_age_classes)), :fecundity_rates_$(key_str))")
        end

        if haskey(params, :K) || spatially_varying_K
            if spatially_varying_K
                sigma_K_prior = get(params, :sigma_K, Exponential(1.0)); log_K_mean_prior = get(params, :log_K_mean, haskey(params, :K) && params[:K] isa LogNormal ? Normal(Distributions.params(params[:K])...) : Normal(log(100.0), 0.5))
                push!(priors_acc, "sigma_K_$(prefixed_key) ~ NamedDist($(_distribution_to_string(sigma_K_prior)), :sigma_K_$(prefixed_key))")
                push!(priors_acc, "log_K_mean_$(prefixed_key) ~ NamedDist($(_distribution_to_string(log_K_mean_prior)), :log_K_mean_$(prefixed_key))")
                push!(priors_acc, "K_raw_$(prefixed_key) ~ NamedDist(MvNormal(zeros(T, M.s_N), I), :K_raw_$(prefixed_key))")
            else
                K_prior = get(params, :K, LogNormal(log(100.0), 1.0)); push!(priors_acc, "K_$(key_str) ~ NamedDist($(_distribution_to_string(K_prior)), :K_$(key_str))")
            end
        end

        effort_keys = get(spec.hyper, :effort_keys, [])
        for key in effort_keys
            q_prior = get(params, Symbol("q_$(key)"), filldist(LogNormal(-4, 1), n_age_classes))
            push!(priors_acc, "q_$(key) ~ NamedDist($(_distribution_to_string(q_prior)), :q_$(key))")
        end

        push!(priors_acc, "sigma_process_$(key_str) ~ NamedDist(filldist(Exponential(1.0), $(n_age_classes)), :sigma_process_$(key_str))")
        push!(priors_acc, "innov_process_$(key_str) ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N * $(n_age_classes)), I), :innov_process_$(key_str))")
        priors_str = join(priors_acc, "\n    ")

        exploitation_block = "local C_prev = zeros(T, $(n_age_classes))"
        for key in effort_keys
            exploitation_block *= "\n            C_prev .+= q_$(key) .* spec_registry[\"$(key_str)\"].hyper.processed_params[:$(key)][s, t-1] .* N_prev"
        end
        removal_keys = get(spec.hyper, :removal_keys, [])
        for key in removal_keys
            exploitation_block *= "\n            C_prev .+= spec_registry[\"$(key_str)\"].hyper.processed_params[:$(key)][s, t-1, :]"
        end

        update_str = """
        begin
            local Q_spatial = spec_registry["$(key_str)"].hyper.L_template; local F_spatial = cholesky(Symmetric(Matrix(Q_spatial) + noise * I)); local areas = spec_registry["$(key_str)"].hyper.areas
            local survival_rates_spatial, fecundity_rates_spatial; if $(spatially_varying_rates); fecundity_raw_matrix = reshape(fecundity_raw_$(key_str), M.s_N, $(n_age_classes)); fecundity_field = F_spatial.L' \\ fecundity_raw_matrix; fecundity_rates_spatial = exp.(log_fecundity_mean_$(key_str)' .+ fecundity_field .* sigma_fecundity_$(key_str)'); survival_raw_matrix = reshape(survival_raw_$(key_str), M.s_N, $(n_age_classes-1)); survival_field = F_spatial.L' \\ survival_raw_matrix; survival_rates_spatial = logistic.(logit_survival_mean_$(key_str)' .+ survival_field .* sigma_survival_$(key_str)'); end
            local K_values_$(key_str); if $(spatially_varying_K); K_field_raw = F_spatial.L' \\ K_raw_$(prefixed_key); Turing.@addlogprob! logpdf(Normal(0, 0.001 * M.s_N), sum(K_field_raw)); K_values_$(key_str) = exp.(log_K_mean_$(prefixed_key) .+ K_field_raw .* sigma_K_$(prefixed_key)); elseif haskey(spec_registry["$(key_str)"].component_obj.params, :K); K_values_$(key_str) = fill(K_$(key_str), M.s_N); end
            local innov_tensor_$(key_str) = reshape(innov_process_$(key_str), M.s_N, M.t_N, $(n_age_classes)); local population_field_$(key_str) = zeros(T, M.s_N, M.t_N, $(n_age_classes))
            for a in 1:$(n_age_classes); population_field_$(key_str)[:, 1, a] = max.(0.0, innov_tensor_$(key_str)[:, 1, a] .* sigma_process_$(key_str)[a]); end
            for s in 1:M.s_N
                local L_s = zeros(T, $(n_age_classes), $(n_age_classes)); if $(spatially_varying_rates); for i in 1:($(n_age_classes)-1); L_s[i+1, i] = survival_rates_spatial[s, i]; end; L_s[1, :] = fecundity_rates_spatial[s, :]; else; for i in 1:($(n_age_classes)-1); L_s[i+1, i] = survival_rates_$(key_str)[i]; end; L_s[1, :] = fecundity_rates_$(key_str); end
                for t in 2:M.t_N
                    local N_prev = view(population_field_$(key_str), s, t-1, :); $(exploitation_block); local N_after_removal = max.(0.0, N_prev - C_prev); local L_effective = copy(L_s)
                    if haskey(spec_registry["$(key_str)"].component_obj.params, :K); local total_pop_prev = sum(N_after_removal); local K_density = K_values_$(key_str)[s] / areas[s]; local dd_factor = max(0.0, 1.0 - (total_pop_prev / areas[s]) / K_density); L_effective[1, :] .*= dd_factor; end
                    local N_projected = L_effective * N_after_removal; local current_innov = view(innov_tensor_$(key_str), s, t, :) .* sigma_process_$(key_str); population_field_$(key_str)[s, t, :] = max.(0.0, N_projected .+ current_innov)
                end
            end
            for k in 1:$(n_age_classes); for i in 1:N; eta_latent[i, k] += log(population_field_$(key_str)[M.s_idx[i], M.t_idx[i], k] + 1e-6); end; end
        end
        """
        return (priors=priors_str, update=update_str)

    elseif model_type == "delay_difference"
        if M.outcomes_N != 2; error("The multivariate `delay_difference` model requires exactly two outcomes: population and recruitment."); end
        priors_acc = []; push!(priors_acc, "r_$(key_str) ~ NamedDist(LogNormal(0, 1), :r_$(key_str))"); push!(priors_acc, "K_$(key_str) ~ NamedDist(LogNormal(log(100.0), 1.0), :K_$(key_str))"); push!(priors_acc, "M_nat_$(key_str) ~ NamedDist(LogNormal(-1, 0.5), :M_nat_$(key_str))"); push!(priors_acc, "sigma_recruitment_$(key_str) ~ NamedDist(Exponential(1.0), :sigma_recruitment_$(key_str))"); push!(priors_acc, "sigma_population_$(key_str) ~ NamedDist(Exponential(1.0), :sigma_population_$(key_str))"); push!(priors_acc, "innov_recruitment_$(key_str) ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :innov_recruitment_$(key_str))"); push!(priors_acc, "innov_population_$(key_str) ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :innov_population_$(key_str))")
        effort_keys = get(spec.hyper, :effort_keys, []); for key in effort_keys; q_prior = get(params, Symbol("q_$(key)"), LogNormal(-2, 1)); push!(priors_acc, "q_$(key) ~ NamedDist($(_distribution_to_string(q_prior)), :q_$(key))"); end
        priors_str = join(priors_acc, "\n    ")

        exploitation_block = "local C_prev = zeros(T, M.s_N)"; for key in effort_keys; exploitation_block *= "\n            C_prev .+= q_$(key) .* spec_registry[\"$(key_str)\"].hyper.processed_params[:$(key)][:, t-1] .* N_prev"; end; removal_keys = get(spec.hyper, :removal_keys, []); for key in removal_keys; exploitation_block *= "\n            C_prev .+= spec_registry[\"$(key_str)\"].hyper.processed_params[:$(key)][:, t-1]"; end

        update_str = """
        begin
            local areas = spec_registry["$(key_str)"].hyper.areas; local population_field = zeros(T, M.s_N, M.t_N); local recruitment_field = zeros(T, M.s_N, M.t_N); local innov_recruitment_matrix = reshape(innov_recruitment_$(key_str), M.s_N, M.t_N); local innov_population_matrix = reshape(innov_population_$(key_str), M.s_N, M.t_N)
            population_field[:, 1] = max.(0.0, innov_population_matrix[:, 1] .* sigma_population_$(key_str)); recruitment_field[:, 1] = max.(0.0, innov_recruitment_matrix[:, 1] .* sigma_recruitment_$(key_str))
            for t in 2:M.t_N
                local N_prev = population_field[:, t-1]; local D_prev = N_prev ./ areas; local K_density = K_$(key_str) ./ areas
                local mean_recruitment = r_$(key_str) .* D_prev .* (1.0 .- D_prev ./ K_density) .* areas; recruitment_field[:, t] = exp.(log.(mean_recruitment .+ 1e-6) .+ innov_recruitment_matrix[:, t] .* sigma_recruitment_$(key_str))
                $(exploitation_block)
                local N_survived = (N_prev .- C_prev) .* exp.(-M_nat_$(key_str)); population_field[:, t] = max.(0.0, N_survived .+ recruitment_field[:, t] .+ innov_population_matrix[:, t] .* sigma_population_$(key_str))
            end
            for i in 1:N; s_i, t_i = M.s_idx[i], M.t_idx[i]; eta_latent[i, 1] += log(population_field[s_i, t_i] + 1e-6); eta_latent[i, 2] += log(recruitment_field[s_i, t_i] + 1e-6); end
        end
        """
        return (priors=priors_str, update=update_str)

     elseif model_type == "generalized_lotka_volterra"
        spatially_varying_K = get(params, :spatially_varying_K, false); priors_acc = []; push!(priors_acc, "r_$(key_str) ~ NamedDist(filldist(LogNormal(0, 1), $(n_species)), :r_$(key_str))"); n_off_diag = n_species * (n_species - 1); push!(priors_acc, "alpha_raw_$(key_str) ~ NamedDist(MvNormal(zeros(T, $(n_off_diag)), I), :alpha_raw_$(key_str))")
        if spatially_varying_K; push!(priors_acc, "log_K_mean_$(key_str) ~ NamedDist(filldist(Normal(log(100.0), 1.0), $(n_species)), :log_K_mean_$(key_str))"); push!(priors_acc, "sigma_K_$(key_str) ~ NamedDist(filldist(Exponential(1.0), $(n_species)), :sigma_K_$(key_str))"); push!(priors_acc, "K_raw_$(key_str) ~ NamedDist(MvNormal(zeros(T, M.s_N * $(n_species)), I), :K_raw_$(key_str))"); else; push!(priors_acc, "K_$(key_str) ~ NamedDist(filldist(LogNormal(log(100.0), 1.0), $(n_species)), :K_$(key_str))"); end
        push!(priors_acc, "sigma_process_$(key_str) ~ NamedDist(filldist(Exponential(1.0), $(n_species)), :sigma_process_$(key_str))"); push!(priors_acc, "innov_process_$(key_str) ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N * $(n_species)), I), :innov_process_$(key_str))"); priors_str = join(priors_acc, "\n    ")
        update_str = """
        begin
            local areas = spec_registry["$(key_str)"].hyper.areas; local alpha_$(key_str) = diagm(0 => ones(T, $(n_species))); local off_diag_indices = [i for i in 1:($(n_species)^2) if mod(i-1, $(n_species)+1) != 0]; alpha_$(key_str)[off_diag_indices] = alpha_raw_$(key_str)
            local K_values_$(key_str); if $(spatially_varying_K); local Q_spatial = spec_registry["$(key_str)"].hyper.L_template; local F_spatial = cholesky(Symmetric(Matrix(Q_spatial) + noise * I)); local K_raw_matrix = reshape(K_raw_$(key_str), M.s_N, $(n_species)); local K_field = F_spatial.L' \\ K_raw_matrix; K_values_$(key_str) = exp.(log_K_mean_$(key_str)' .+ K_field .* sigma_K_$(key_str)'); else; K_values_$(key_str) = repeat(K_$(key_str)', M.s_N, 1); end
            local innov_tensor = reshape(innov_process_$(key_str), M.s_N, M.t_N, $(n_species)); local population_field = zeros(T, M.s_N, M.t_N, $(n_species)); population_field[:, 1, :] = max.(0.0, innov_tensor[:, 1, :] .* sigma_process_$(key_str)')
            for s in 1:M.s_N, t in 2:M.t_N
                local N_prev = view(population_field, s, t-1, :); local D_prev = N_prev ./ areas[s]; local K_density = K_values_$(key_str)[s, :] ./ areas[s]; local N_intermediate = zeros(T, $(n_species))
                for i in 1:$(n_species); local interaction_sum_density = dot(alpha_$(key_str)[i, :], D_prev); local growth_density = r_$(key_str)[i] * D_prev[i] * (1.0 - interaction_sum_density / K_density[i]); N_intermediate[i] = N_prev[i] + growth_density * areas[s]; end
                local current_innov = view(innov_tensor, s, t, :) .* sigma_process_$(key_str); population_field[s, t, :] = max.(0.0, N_intermediate .+ current_innov)
            end
            for k in 1:$(n_species); for i in 1:N; eta_latent[i, k] += log(population_field[M.s_idx[i], M.t_idx[i], k] + 1e-6); end; end
        end
        """
        return (priors=priors_str, update=update_str)
            
    end
    
    return (priors="", update="")
end



# Version 1.5.3 (2026-08-06)
# Purpose: Generates Turing code fragments for the Spatially Varying Autoregressive (SVAR) model.
# Rationale: This version is updated for AD compatibility. It converts the precision matrix
#            to a dense matrix before the Cholesky decomposition when the matrix contains
#            Dual numbers. This avoids calling the sparse Cholesky factorization from CHOLMOD,
#            which does not support Dual types, resolving the `TypeError`.
function _generate_component_code_fragments(m::SVAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="", generate_eta_update::Bool=true)
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    
    rho_spatial_prefix = prefixed_key * "_rho_spatial"
    inner_spec_for_codegen = (
        key = Symbol(rho_spatial_prefix),
        structure = :spatial,
        var = spec.var,
        component_obj = m.rho_spatial,
        params = spec.params,
        Q_template = spec.hyper.rho_spatial_spec.Q_template,
        scaling_factor = spec.hyper.rho_spatial_spec.scaling_factor,
        hyper = spec.hyper.rho_spatial_spec.hyper
    )
    v_rho_spatial = generate_full_variable_names(inner_spec_for_codegen, arch, outcome_idx, prefix="")

    inner_model = m.rho_spatial
    n_latent_inner = size(inner_spec_for_codegen.Q_template, 1)
    n_latent_svar = spec.hyper.s_N * spec.hyper.t_N

    is_multivariate = arch == "multivariate"
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]

    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        if hasproperty(inner_model, :sigma)
            push!(priors_acc, "$(v_rho_spatial.sigma) ~ NamedDist($(_distribution_to_string(inner_model.sigma)), :$(v_rho_spatial.sigma))")
        end
    end
    push!(priors_acc, "$(v_rho_spatial.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent_inner)), I), :$(v_rho_spatial.raw))")

    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
    end
    push!(priors_acc, "$(v.innov) ~ NamedDist(MvNormal(zeros(T, $(n_latent_svar)), I), :$(v.innov))")

    priors_str = join(priors_acc, "\n")
    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"

    update_str = if generate_eta_update
        """
        begin
            # --- SVAR Model: $(key_str) ---
            
            # 1. Compute the spatially varying rho field (inner model logic)
            begin
                local Q_template_inner = spec_registry["$(key_str)"].hyper.rho_spatial_spec.Q_template
                local F_inner = cholesky(Symmetric(Matrix(Q_template_inner) + noise * I))
                local latent_field_raw_inner = F_inner.L' \\ $(v_rho_spatial.raw)
                Turing.@addlogprob! logpdf(Normal(T(0), T(0.001) * $(n_latent_inner)), sum(latent_field_raw_inner))
                $(v_rho_spatial.latent) = latent_field_raw_inner .* $(v_rho_spatial.sigma)
            end
            
            local $(v.rho_field) = tanh.($(v_rho_spatial.latent))
            
            local $(v.latent) = zeros(T, M.s_N, M.t_N)
            local innov_matrix = reshape($(v.innov), M.s_N, M.t_N)
            
            for s in 1:M.s_N
                $(v.latent)[s, :] = ar1_statespace($(v.rho_field)[s], T(1.0), innov_matrix[s, :], T, M.t_N, noise)
            end
            $(v.latent) .*= $(v.sigma)
     
            for i in 1:M.y_N
                $(eta_update_target)[i] += $(v.latent)[M.s_idx[i], M.t_idx[i]]
            end
        end
        """
    else
        ""
    end

    return (priors=priors_str, update=update_str)
end





# Version 1.5.3 (2026-08-06)
# Purpose: Generates Turing code fragments for a Kriging (full Gaussian Process) component.
# Rationale: This version ensures type stability by explicitly casting `X_coords` to `T`
#            before use in `evaluate_kernel_matrix`.
function _generate_component_code_fragments(m::Kriging, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    n_obs = size(spec.Q_template, 1) # For Kriging/GP, Q_template holds the coordinates

    priors_acc = String[]
    push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
    
    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors_acc, "$(v.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(v.ls))")
    else
        push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
    end
    
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_obs)), I), :$(v.raw))")
    priors = join(priors_acc, "\n")

    update = """
    begin
        local X_coords = T.(spec_registry["$(key_str)"].Q_template)
        local kernel_type = Symbol("$(m.kernel)")
        local K_mat = evaluate_kernel_matrix(X_coords, $(v.sigma), $(v.ls), kernel_type, noise)
        local F_krig = cholesky(Symmetric(K_mat))
        $(v.latent) = F_krig.L * $(v.raw)
        $(arch == "multivariate" ? "eta_latent[:, $(outcome_idx)]" : "eta") .+= $(v.latent)
    end
    """
    return (priors=priors, update=update)
end





function _generate_component_code_fragments(m::Any, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; kwargs...)
    # This is a generic fallback. Ideally, every component has a specialized method.
    # We can try to dispatch to the most general `ComponentModel` method if it fits.
    if m isa ComponentModel
        return _generate_component_code_fragments(m, spec, arch, outcome_idx, M; kwargs...)
    else
        @warn "Code generation for component type `$(typeof(m))` is not explicitly implemented. No code will be generated for component `$(spec.key)`."
        return (priors="", update="")
    end
end


# Version 1.5.6 (2026-08-06)
# Purpose: Generates Turing code fragments for the `MixedComponent`.
# Rationale: This version is updated for AD compatibility. It replaces hard-coded
#            `T.(I(...))` in `MvNormal` priors and `cholesky` calls with `I`, which
#            correctly uses `UniformScaling` to allow for automatic type promotion
#            with `Dual` numbers, preventing `MethodError` during AD.
function _generate_component_code_fragments(m::MixedComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    inner_model = m.model
    group_var = m.group_var 
    lhs_effects_raw = m.lhs
    lhs_effects = vcat([Base.split(s, r"\s*\+\s*") for s in lhs_effects_raw]...)
    k_effects = length(lhs_effects)
    n_groups = get(spec.params, :n_cat, 0) 
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "mixed_idx_$(group_var)"

    if k_effects == 1
        inner_frags = _generate_component_code_fragments(inner_model, spec, arch, outcome_idx, M, prefix=prefix)
        effect_app_regex = r"for i in 1:length\($(eta_target)\)\s*$(eta_target)\[i\] \+= .*end"
        update_inner_cleaned = replace(inner_frags.update, effect_app_regex => "")
        
        lhs_str = lhs_effects[1]
        is_intercept = (lhs_str == "1" || lhs_str == "intercept()" || lhs_str == "(Intercept)")

        local application_code
        if is_intercept
            application_code = """
            for i in 1:length($(eta_target))
                $(eta_target)[i] += $(v.latent)[M.$(index_var)[i]]
            end
            """
        else
            application_code = """
            local cov_data = M.data[!, :$(Symbol(lhs_str))]
            for i in 1:length($(eta_target))
                $(eta_target)[i] += cov_data[i] * $(v.latent)[M.$(index_var)[i]]
            end
            """
        end

        update_str = """
        begin
            # Mixed Effect Logic (Single): $(lhs_str) | $(group_var)
            $(update_inner_cleaned)
            $(application_code)
        end
        """
        return (priors=inner_frags.priors, update=update_str)
    else
        priors_acc = String[]
        push!(priors_acc, "$(v.L_corr) ~ NamedDist(LKJCholesky(T($(k_effects)), T(1.0)), :$(v.L_corr))")
        push!(priors_acc, "$(v.sigma_effects) ~ NamedDist(filldist(Exponential{T}(1.0), $(k_effects)), :$(v.sigma_effects))")
        push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_groups * k_effects)), I), :$(v.raw))")

        local group_chol_logic
        if get(spec, :is_static, false)
            group_chol_logic = "local F_groups_$(spec.key) = spec_registry[\"$(spec.key)\"].cholesky_factor\nlocal L_groups_cov_inv_t_$(spec.key) = F_groups_$(spec.key).L'"
        else
            group_chol_logic = "local Q_groups_$(spec.key) = spec_registry[\"$(spec.key)\"].Q_template\nlocal F_groups_$(spec.key) = cholesky(Symmetric(Matrix(Q_groups_$(spec.key)) + noise * I))\nlocal L_groups_cov_inv_t_$(spec.key) = F_groups_$(spec.key).L'"
        end

        application_loop_acc = String[]
        for i in 1:k_effects
            term = lhs_effects[i]
            is_int = (term == "1" || term == "intercept()" || term == "(Intercept)")
            local term_application
            if is_int
                term_application = """
                for j in 1:length($(eta_target))
                    $(eta_target)[j] += effects_matrix_$(spec.key)[M.$(index_var)[j], $(i)]
                end
                """
            else
                term_application = """
                local cov_data = M.data[!, :$(Symbol(term))]
                for j in 1:length($(eta_target))
                    $(eta_target)[j] += cov_data[j] * effects_matrix_$(spec.key)[M.$(index_var)[j], $(i)]
                end
                """
            end
            push!(application_loop_acc, term_application)
        end

        update_str = """
        begin
            # Correlated Mixed Effects Construction for $(group_var)
            local L_effects_t_$(spec.key) = ($(v.L_corr).L' * Diagonal($(v.sigma_effects)))
            $(group_chol_logic)
            local innovations_matrix_$(spec.key) = reshape($(v.raw), $(n_groups), $(k_effects))
            
            local gamma_matrix_$(spec.key)::Matrix{T} = convert(Matrix{T}, L_groups_cov_inv_t_$(spec.key)' \\ innovations_matrix_$(spec.key))
            local effects_matrix_$(spec.key)::Matrix{T} = convert(Matrix{T}, gamma_matrix_$(spec.key) * L_effects_t_$(spec.key))
            
            $(join(application_loop_acc, "\n            "))
        end
        """
        return (priors=join(priors_acc, "\n    "), update=update_str)
    end
end


# Version 1.3.11 (2026-08-05)
# Purpose: Generates Turing code fragments for the `SVCComponent`.
# Rationale: This version replaces all broadcasted assignments (`.+=`) and `view()` calls
#            with explicit `for` loops and direct indexing. This ensures maximum type
#            stability for automatic differentiation by preventing the compiler from making
#            incorrect type inferences about arrays containing `ForwardDiff.Dual` numbers.
#            Covariate data is also explicitly cast to type `T` within the loop.
function _generate_component_code_fragments(m::SVCComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    inner_model = m.model
    cov_var = m.covariate
    
    inner_frags = _generate_component_code_fragments(inner_model, spec, arch, outcome_idx, M, prefix=prefix)
    priors_str = inner_frags.priors
    update_inner = inner_frags.update
    
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    # Remove any existing effect application from the inner fragment's update string.
    effect_app_regex = r"for i in 1:length\($(eta_target)\)\s*$(eta_target)\[i\] \+= .*end"
    update_inner_cleaned = replace(update_inner, effect_app_regex => "")
    
    is_intercept = (string(cov_var) == "1" || string(cov_var) == "intercept()")
    
    local application_code
    if is_intercept
        application_code = """
        for i in 1:length($(eta_target))
            $(eta_target)[i] += $(v.latent)[M.s_idx[i]]
        end
        """
    else
        application_code = """
        local cov_data = T.(M.data[!, :$(cov_var)])
        for i in 1:length($(eta_target))
            $(eta_target)[i] += cov_data[i] * $(v.latent)[M.s_idx[i]]
        end
        """
    end

    update_str = """
    begin
        # Spatially Varying Coefficient (SVC) for: $(cov_var)
        $(update_inner_cleaned)
        $(application_code)
    end
    """
    return (priors=priors_str, update=update_str)
end


 

# Version 1.5.3 (2026-08-06)
# Purpose: Generates Turing code fragments for the `DynamicsComponent`.
# Rationale: This version ensures type stability by explicitly casting numeric literals
#            and `T(1e-6)` to `T` where appropriate.
function _generate_component_code_fragments(m::DynamicsComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = m.params
    model_type = m.model
    is_multivariate = arch == "multivariate"
    
    if model_type == "generalized_leslie_matrix"
        if !is_multivariate; error("The `generalized_leslie_matrix` model requires a multivariate likelihood with an outcome for each class."); end
        
        n_classes = get(params, :n_classes, M.outcomes_N)
        if n_classes != M.outcomes_N; error("Number of classes ($n_classes) must match number of outcomes ($(M.outcomes_N))."); end

        spatially_varying_K = get(params, :spatially_varying_K, false)
        spatially_varying_rates = get(params, :spatially_varying_rates, false)

        priors_acc = String[]
        
        if spatially_varying_rates
            push!(priors_acc, "A_mean_$(key_str) ~ NamedDist(filldist(Normal(T(0), T(1)), $(n_classes^2)), :A_mean_$(key_str))")
            push!(priors_acc, "A_sigma_$(key_str) ~ NamedDist(filldist(Exponential(T(1.0)), $(n_classes^2)), :A_sigma_$(key_str))")
            push!(priors_acc, "A_raw_$(key_str) ~ NamedDist(MvNormal(fill(zero(T), M.s_N * $(n_classes^2)), I), :A_raw_$(key_str))")
        else
            push!(priors_acc, "A_flat_$(key_str) ~ NamedDist(filldist(Normal(T(0), T(1)), $(n_classes^2)), :A_flat_$(key_str))")
        end

        if haskey(params, :K) || spatially_varying_K
            if spatially_varying_K
                sigma_K_prior = get(params, :sigma_K, Exponential(T(1.0))); log_K_mean_prior = get(params, :log_K_mean, haskey(params, :K) && params[:K] isa LogNormal ? Normal(Distributions.params(params[:K])...) : Normal(T(log(100.0)), T(0.5)))
                push!(priors_acc, "sigma_K_$(prefixed_key) ~ NamedDist($(_distribution_to_string(sigma_K_prior)), :sigma_K_$(prefixed_key))")
                push!(priors_acc, "log_K_mean_$(prefixed_key) ~ NamedDist($(_distribution_to_string(log_K_mean_prior)), :log_K_mean_$(prefixed_key))")
                push!(priors_acc, "K_raw_$(prefixed_key) ~ NamedDist(MvNormal(fill(zero(T), M.s_N), I), :K_raw_$(prefixed_key))")
            else
                K_prior = get(params, :K, LogNormal(T(log(100.0)), T(1.0))); push!(priors_acc, "K_$(key_str) ~ NamedDist($(_distribution_to_string(K_prior)), :K_$(key_str))")
            end
        end

        effort_keys = get(spec.hyper, :effort_keys, [])
        for key in effort_keys
            q_prior = get(params, Symbol("q_$(key)"), filldist(LogNormal(T(-4), T(1)), n_classes))
            push!(priors_acc, "q_$(key) ~ NamedDist($(_distribution_to_string(q_prior)), :q_$(key))")
        end

        push!(priors_acc, "sigma_process_$(key_str) ~ NamedDist(filldist(Exponential(T(1.0)), $(n_classes)), :sigma_process_$(key_str))")
        push!(priors_acc, "innov_process_$(key_str) ~ NamedDist(MvNormal(fill(zero(T), M.s_N * M.t_N * $(n_classes)), I), :innov_process_$(key_str))")
        priors_str = join(priors_acc, "\n    ")

        exploitation_block = "local C_prev = fill(zero(T_num), $(n_classes))"
        for key in effort_keys
            exploitation_block *= "\n            C_prev .+= q_$(key) .* spec_registry[\"$(key_str)\"].hyper.processed_params[:$(key)][s, t-1] .* N_prev"
        end
        removal_keys = get(spec.hyper, :removal_keys, [])
        for key in removal_keys
            exploitation_block *= "\n            C_prev .+= spec_registry[\"$(key_str)\"].hyper.processed_params[:$(key)][s, t-1, :]"
        end

        update_str = """
        begin
            local Q_spatial = spec_registry["$(key_str)"].hyper.L_template; local F_spatial = cholesky(Symmetric(Q_spatial + noise * I)); local areas = spec_registry["$(key_str)"].hyper.areas
            local A_spatial; if $(spatially_varying_rates); A_raw_matrix = reshape(A_raw_$(key_str), M.s_N, $(n_classes^2)); A_field = F_spatial.L' \\ A_raw_matrix; A_spatial = exp.(A_mean_$(key_str)' .+ A_field .* A_sigma_$(key_str)'); else; A_s = reshape(A_flat_$(key_str), $(n_classes), $(n_classes)); end
            local K_values_$(key_str); if $(spatially_varying_K); K_field_raw = F_spatial.L' \\ K_raw_$(prefixed_key); Turing.@addlogprob! logpdf(Normal(zero(T), T(0.001) * M.s_N), sum(K_field_raw)); K_values_$(key_str) = exp.(log_K_mean_$(prefixed_key) .+ K_field_raw .* sigma_K_$(prefixed_key)); else; K_values_$(key_str) = fill(K_$(key_str), M.s_N); end
            local innov_tensor = reshape(innov_process_$(key_str), M.s_N, M.t_N, $(n_classes)); 
            local T_num = eltype(innov_tensor);
            local population_field = fill(zero(T_num), M.s_N, M.t_N, $(n_classes));
            for c in 1:$(n_classes); population_field[:, 1, c] = max.(T_num(0.0), innov_tensor[:, 1, c] .* sigma_process_$(key_str)[c]); end
            for s in 1:M.s_N
                local A_s; if $(spatially_varying_rates); A_s = reshape(A_spatial[s, :], $(n_classes), $(n_classes)); else; A_s = A_spatial; end
                for t in 2:M.t_N
                    local N_prev = view(population_field, s, t-1, :); $(exploitation_block); local N_after_removal = max.(T_num(0.0), N_prev - C_prev); local A_effective = copy(A_s)
                    if haskey(spec_registry["$(key_str)"].component_obj.params, :K); local total_pop_prev = sum(N_after_removal); local K_density = K_values_$(key_str)[s] / areas[s]; local dd_factor = max(T_num(0.0), T(1.0) - (total_pop_prev / areas[s]) / K_density); A_effective[1, :] .*= dd_factor; end
                    local N_projected = A_effective * N_after_removal; local current_innov = view(innov_tensor, s, t, :) .* sigma_process_$(key_str); population_field[s, t, :] = max.(T_num(0.0), N_projected .+ current_innov)
                end
            end
            for k in 1:$(n_classes); for i in 1:N; eta_latent[i, k] += log(population_field[M.s_idx[i], M.t_idx[i], k] + T(1e-6)); end; end
        end
        """
        return (priors=priors_str, update=update_str)
    end

    spatially_varying_K = get(params, :spatially_varying_K, false)
    spatially_varying_r = get(params, :spatially_varying_r, false)

    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)

    priors_acc = String[]

    has_propagator = model_type in ["advection", "diffusion", "advection_diffusion"]
    if has_propagator
        if model_type in ["advection", "advection_diffusion"]; vel = get(params, :velocity, Normal(T(0), T(0.5))); push!(priors_acc, "$(v.velocity) ~ NamedDist($(_distribution_to_string(vel)), :$(v.velocity))"); end
        if model_type in ["diffusion", "advection_diffusion"]; diff = get(params, :diffusion, LogNormal(T(-1), T(1))); push!(priors_acc, "$(v.diffusion) ~ NamedDist($(_distribution_to_string(diff)), :$(v.diffusion))"); end
    end
    
    sigma = get(params, :sigma, Exponential(T(1.0)))
    push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(sigma)), :$(v.sigma))")

    if model_type in ["logistic", "delay_difference"]
        if spatially_varying_r
            log_r_mean_prior = get(params, :log_r_mean, haskey(params, :r) && params[:r] isa LogNormal ? Normal(Distributions.params(params[:r])...) : Normal(T(0.0), T(0.5)))
            sigma_r_prior = get(params, :sigma_r, Exponential(T(1.0)))
            push!(priors_acc, "sigma_r_$(prefixed_key) ~ NamedDist($(_distribution_to_string(sigma_r_prior)), :sigma_r_$(prefixed_key))")
            push!(priors_acc, "log_r_mean_$(prefixed_key) ~ NamedDist($(_distribution_to_string(log_r_mean_prior)), :log_r_mean_$(prefixed_key))")
            push!(priors_acc, "r_raw_$(prefixed_key) ~ NamedDist(MvNormal(fill(zero(T), M.s_N), I), :r_raw_$(prefixed_key))")
        else
            r = get(params, :r, LogNormal(T(0), T(1))); push!(priors_acc, "$(v.r) ~ NamedDist($(_distribution_to_string(r)), :$(v.r))")
        end

        if spatially_varying_K
            log_K_mean_prior = get(params, :log_K_mean, haskey(params, :K) && params[:K] isa LogNormal ? Normal(Distributions.params(params[:K])...) : Normal(T(log(100.0)), T(0.5)))
            sigma_K_prior = get(params, :sigma_K, Exponential(T(1.0)))
            push!(priors_acc, "sigma_K_$(prefixed_key) ~ NamedDist($(_distribution_to_string(sigma_K_prior)), :sigma_K_$(prefixed_key))")
            push!(priors_acc, "log_K_mean_$(prefixed_key) ~ NamedDist($(_distribution_to_string(log_K_mean_prior)), :log_K_mean_$(prefixed_key))")
            push!(priors_acc, "K_raw_$(prefixed_key) ~ NamedDist(MvNormal(fill(zero(T), M.s_N), I), :K_raw_$(prefixed_key))")
        else
            K = get(params, :K, LogNormal(T(log(100.0)), T(1.0))); push!(priors_acc, "$(v.K) ~ NamedDist($(_distribution_to_string(K)), :$(v.K))")
        end
    end
    
    if model_type == "logistic" || model_type == "delay_difference"
        effort_keys = get(spec.hyper, :effort_keys, [])
        for key in effort_keys
            q_prior = get(params, Symbol("q_$(key)"), LogNormal(T(-2), T(1)))
            push!(priors_acc, "q_$(key) ~ NamedDist($(_distribution_to_string(q_prior)), :q_$(key))")
        end
    end
    if model_type == "delay_difference"; M_nat = get(params, :M_nat, LogNormal(T(-1), T(0.5))); push!(priors_acc, "$(v.M_nat) ~ NamedDist($(_distribution_to_string(M_nat)), :$(v.M_nat))"); end
    if model_type == "lotka_volterra"; alpha = get(params, :alpha, LogNormal(T(0), T(1))); beta = get(params, :beta, LogNormal(T(-1), T(1))); gamma = get(params, :gamma, LogNormal(T(-1), T(1))); delta = get(params, :delta, LogNormal(T(0), T(1))); push!(priors_acc, "$(v.alpha) ~ NamedDist($(_distribution_to_string(alpha)), :$(v.alpha))"); push!(priors_acc, "$(v.beta) ~ NamedDist($(_distribution_to_string(beta)), :$(v.beta))"); push!(priors_acc, "$(v.gamma) ~ NamedDist($(_distribution_to_string(gamma)), :$(v.gamma))"); push!(priors_acc, "$(v.delta) ~ NamedDist($(_distribution_to_string(delta)), :$(v.delta))"); push!(priors_acc, "$(v.innov)_predator ~ NamedDist(MvNormal(fill(zero(T), M.s_N * M.t_N), I), :$(Symbol(string(v.innov, "_predator"))))"); end

    innov_name = v.innov
    push!(priors_acc, "$(innov_name) ~ NamedDist(MvNormal(fill(zero(T), M.s_N * M.t_N), I), :$(innov_name))")
    priors_str = join(priors_acc, "\n    ")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    L_op = "spec_registry[\"$(key_str)\"].hyper.L_template"
    A_op = "spec_registry[\"$(key_str)\"].hyper.A_template"
    grid_areas_access = "spec_registry[\"$(key_str)\"].hyper.areas"

    local evolution_loop_body, propagator_setup, field_setup, K_setup_block, r_setup_block
    
    propagator_setup = ""
    if has_propagator; if model_type == "advection"; propagator_setup = "local propagator = lu(I(M.s_N) - $(v.velocity) * $(A_op) + noise * I)"; elseif model_type == "diffusion"; propagator_setup = "local propagator = cholesky(Symmetric(I(M.s_N) - $(v.diffusion) * $(L_op) + noise * I))"; elseif model_type == "advection_diffusion"; propagator_setup = "local propagator = lu(I(M.s_N) - $(v.velocity) * $(A_op) - $(v.diffusion) * $(L_op) + noise * I)"; end; end

    field_setup = "local T_num_dyn = eltype($(innov_name)); dyn_field = fill(zero(T_num_dyn), M.s_N, M.t_N)\n    innov_matrix = reshape($(innov_name), M.s_N, M.t_N)\n    dyn_field[:, 1] = innov_matrix[:, 1]"
    
    K_setup_block = ""; K_variable_name = spatially_varying_K ? "K_spatial" : string(v.K)
    if spatially_varying_K; K_setup_block = "local Q_K = spec_registry[\"$(key_str)\"].hyper.L_template\nlocal F_K = cholesky(Symmetric(Q_K + noise * I))\nlocal K_field_raw = F_K.L' \\ K_raw_$(prefixed_key)\nTuring.@addlogprob! logpdf(Normal(zero(T), T(0.001) * M.s_N), sum(K_field_raw))\nlocal K_spatial = exp.(log_K_mean_$(prefixed_key) .+ K_field_raw .* sigma_K_$(prefixed_key))"; end

    r_setup_block = ""; r_variable_name = spatially_varying_r ? "r_spatial" : string(v.r)
    if spatially_varying_r; r_setup_block = "local Q_r = spec_registry[\"$(key_str)\"].hyper.L_template\nlocal F_r = cholesky(Symmetric(Q_r + noise * I))\nlocal r_field_raw = F_r.L' \\ r_raw_$(prefixed_key)\nTuring.@addlogprob! logpdf(Normal(zero(T), T(0.001) * M.s_N), sum(r_field_raw))\nlocal r_spatial = exp.(log_r_mean_$(prefixed_key) .+ r_field_raw .* sigma_r_$(prefixed_key))"; end

    propagator_logic = has_propagator ? "dyn_field[:, t] = (propagator \\ N_intermediate) .+ innov_matrix[:, t]" : "dyn_field[:, t] = N_intermediate .+ innov_matrix[:, t]"

    if model_type == "logistic"
        exploitation_logic = generate_exploitation_block(spec, "t")
        evolution_loop_body = "local areas = $(grid_areas_access)\nfor t in 2:M.t_N\n    local N_prev = dyn_field[:, t-1]\n    local D_prev = N_prev ./ areas\n    local K_density = $(K_variable_name) ./ areas\n    local growth = $(r_variable_name) .* D_prev .* (T(1.0) .- D_prev ./ K_density)\n    $(exploitation_logic)\n    local N_intermediate = N_prev .+ (growth .* areas) .- exploitation\n    $(propagator_logic)\n    dyn_field[:, t] = max.(T_num_dyn(0.0), dyn_field[:, t])\nend"
    elseif model_type == "delay_difference"
        exploitation_logic = generate_exploitation_block(spec, "t-1")
        evolution_loop_body = "local areas = $(grid_areas_access)\nfor t in 2:M.t_N\n    local N_prev = dyn_field[:, t-1]\n    local D_prev = N_prev ./ areas\n    local K_density = $(K_variable_name) ./ areas\n    local growth = $(r_variable_name) .* D_prev .* (T(1.0) .- D_prev ./ K_density)\n    $(exploitation_logic)\n    local N_survived = (N_prev .- exploitation) .* exp.(-$(v.M_nat))\n    local N_intermediate = N_survived .+ (growth .* areas)\n    $(propagator_logic)\n    dyn_field[:, t] = max.(T_num_dyn(0.0), dyn_field[:, t])\nend"
    elseif model_type == "lotka_volterra"; output_species = get(params, :output_species, :prey); interaction_cov_sym = get(params, :interaction_covariate, nothing); field_setup = "local T_num_dyn = eltype($(v.innov)); dyn_field_prey = fill(zero(T_num_dyn), M.s_N, M.t_N)\ndyn_field_predator = fill(zero(T_num_dyn), M.s_N, M.t_N)\ninnov_matrix_prey = reshape($(v.innov), M.s_N, M.t_N)\ninnov_matrix_predator = reshape($(v.innov)_predator, M.s_N, M.t_N)\ndyn_field_prey[:, 1] = innov_matrix_prey[:, 1]\ndyn_field_predator[:, 1] = innov_matrix_predator[:, 1]"; evolution_loop_body = "local predator_pop_matrix = if !isnothing(Symbol(\"$(interaction_cov_sym)\"))\n    spec_registry[\"$(key_str)\"].hyper.processed_params[:$(interaction_cov_sym)]\nelse\n    nothing\nend\nfor t in 2:M.t_N\n    local N_prey_prev = dyn_field_prey[:, t-1]\n    local N_pred_prev = isnothing(predator_pop_matrix) ? dyn_field_predator[:, t-1] : predator_pop_matrix[:, t-1]\n    local d_prey = ($(v.alpha) .* N_prey_prev) .- ($(v.beta) .* N_prey_prev .* N_pred_prev)\n    local d_pred = ($(v.gamma) .* N_prey_prev .* N_pred_prev) .- ($(v.delta) .* N_pred_prev)\n    dyn_field_prey[:, t] = max.(T_num_dyn(0.0), N_prey_prev .+ d_prey .+ innov_matrix_prey[:, t])\n    dyn_field_predator[:, t] = max.(T_num_dyn(0.0), N_pred_prev .+ d_pred .+ innov_matrix_predator[:, t])\nend\nlocal dyn_field = $(output_species == :prey ? "dyn_field_prey" : "dyn_field_predator")"
    else; evolution_loop_body = "for t in 2:M.t_N\n    dyn_field[:, t] = (propagator \\ dyn_field[:, t-1]) + innov_matrix[:, t]\nend"; end

    update_str = "begin\n# Dynamics model: $(model_type) for $(key_str)\n$(K_setup_block)\n$(r_setup_block)\n$(propagator_setup)\n$(field_setup)\n$(evolution_loop_body)\ndyn_field .*= $(v.sigma)\nfor i in 1:N\n    $(eta_update_target)[i] += log(dyn_field[M.s_idx[i], M.t_idx[i]] + T(1e-6))\nend\nend"
    
    return (priors=priors_str, update=update_str)
end





# Version 1.5.6 (2026-08-06)
# Purpose: Generates Turing code for a Random Walk 1 (RW1) process.
# Rationale: This version is updated for AD compatibility. It replaces the hard-coded
#            `T.(I(...))` in the `MvNormal` prior with `I`, which correctly uses
#            `UniformScaling` to allow for automatic type promotion with `Dual` numbers.
function _generate_component_code_fragments(m::RW1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    key_str = string(spec.key)
    v = generate_full_variable_names(spec, arch, outcome_idx; prefix=prefix)
    n_latent = size(spec.Q_template, 1)

    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
    end
    push!(priors_acc, "$(v.innov) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.innov))")
    priors_str = join(priors_acc, "\n")

    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx"

    update_str = """
    begin
        # RW1 state-space evolution for $(key_str)
        local innovations = $(v.innov)
        local T_num = eltype(innovations)
        local latent_field_raw::Vector{T_num} = Vector{T_num}(undef, $(n_latent))
        
        if $(n_latent) > 0
            latent_field_raw[1] = innovations[1]
            for t in 2:$(n_latent)
                latent_field_raw[t] = latent_field_raw[t-1] + innovations[t]
            end
        end
        
        if $(n_latent) > 0
            Turing.@addlogprob! logpdf(Normal(T(0), T(0.001) * $(n_latent)), sum(latent_field_raw))
            local $(v.latent)::Vector{T_num} = Vector{T_num}(undef, $(n_latent))
            for i in 1:$(n_latent)
                $(v.latent)[i] = latent_field_raw[i] * $(v.sigma)
            end
            for i in 1:length($(eta_target))
                $(eta_target)[i] += $(v.latent)[M.$(index_var)[i]]
            end
        end
    end
    """
    
    return (priors=priors_str, update=update_str)
end


# Version 1.5.6 (2026-08-06)
# Purpose: Generates Turing code for a Random Walk 2 (RW2) process.
# Rationale: This version is updated for AD compatibility. It replaces the hard-coded
#            `T.(I(...))` in the `MvNormal` prior with `I`, which correctly uses
#            `UniformScaling` to allow for automatic type promotion with `Dual` numbers.
function _generate_component_code_fragments(m::RW2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    key_str = string(spec.key)
    v = generate_full_variable_names(spec, arch, outcome_idx; prefix=prefix)
    
    n_latent = size(spec.Q_template, 1)

    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
    end
    push!(priors_acc, "$(v.innov) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.innov))")
    priors_str = join(priors_acc, "\n")

    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx"

    update_str = """
    begin
        # RW2 state-space evolution for $(key_str)
        local innovations = $(v.innov)
        local T_num = eltype(innovations)
        local latent_field_raw::Vector{T_num} = Vector{T_num}(undef, $(n_latent))
        
        if $(n_latent) > 0; latent_field_raw[1] = innovations[1]; end
        if $(n_latent) > 1; latent_field_raw[2] = T(2.0)*latent_field_raw[1] + innovations[2]; end

        for t in 3:$(n_latent)
            latent_field_raw[t] = T(2.0)*latent_field_raw[t-1] - latent_field_raw[t-2] + innovations[t]
        end
        
        if $(n_latent) > 0
            Turing.@addlogprob! logpdf(Normal(T(0), T(0.001) * $(n_latent)), sum(latent_field_raw))
            local $(v.latent)::Vector{T_num} = Vector{T_num}(undef, $(n_latent))
            for i in 1:$(n_latent)
                $(v.latent)[i] = latent_field_raw[i] * $(v.sigma)
            end
            for i in 1:length($(eta_target))
                $(eta_target)[i] += $(v.latent)[M.$(index_var)[i]]
            end
        end
    end
    """
    
    return (priors=priors_str, update=update_str)
end



# Version 1.5.3 (2026-08-06)
# Purpose: Generates Turing code fragments for the `RFF` component.
# Rationale: This version is updated for AD compatibility. It ensures that the coordinate
#            data `X_coords` is explicitly converted to the generic model type `T` before
#            being multiplied with the RFF projection weights. This prevents a `MethodError`
#            when a `Matrix{Float64}` is multiplied by a `Matrix{ForwardDiff.Dual}`.
function _generate_component_code_fragments(m::RFF, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    n_features = m.n_features
    
    in_dims = size(spec.hyper.coords, 2)
    
    priors_acc = String[]
    push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors_acc, "$(v.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(v.ls))")
    else
        push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
    end
    
    push!(priors_acc, "$(v.W) ~ NamedDist(MvNormal(vec(spec_registry[\"$(key_str)\"].hyper.W_fixed), T(0.1)), :$(v.W))")
    push!(priors_acc, "$(v.b) ~ NamedDist(MvNormal(spec_registry[\"$(key_str)\"].hyper.b_fixed, T(0.1)), :$(v.b))")
    push!(priors_acc, "$(v.beta) ~ NamedDist(MvNormal(zeros(T, $(n_features)), I), :$(v.beta))")
    
    priors_str = join(priors_acc, "\n")
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"

    update = """
    begin
        # RFF GP model for $(key_str)
        local X_coords = T.(spec_registry["$(key_str)"].hyper.coords)
        local W_matrix = reshape($(v.W), $(in_dims), $(n_features))
        local Phi = sqrt(T(2.0) / $(n_features)) .* cos.((X_coords * W_matrix) .+ $(v.b)')
        local scaled_beta = $(v.beta) .* $(v.sigma)
        local rff_effect = Phi * scaled_beta
        $(eta_target) .+= rff_effect
    end
    """
    return (priors=priors_str, update=update)
end

 

# Version 1.5.3 (2026-08-06)
# Purpose: Generates Turing code fragments for the `TAR` component.
# Rationale: This version is updated for AD compatibility. It ensures that all numeric
#            literals and data-derived values are explicitly converted to the generic
#            model type `T` before being used in operations with model parameters.
function _generate_component_code_fragments(m::TAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"

    rho1_name = Symbol("$(v.rho)_1")
    rho2_name = Symbol("$(v.rho)_2")
    sigma1_name = Symbol("$(v.sigma)_1")
    sigma2_name = Symbol("$(v.sigma)_2")

    priors = """
    # TAR Regime-Switching Priors
    $(rho1_name) ~ NamedDist($(_distribution_to_string(m.rho_regimes[1])), :$(rho1_name))
    $(rho2_name) ~ NamedDist($(_distribution_to_string(m.rho_regimes[2])), :$(rho2_name))
    $(sigma1_name) ~ NamedDist($(_distribution_to_string(m.sigma_regimes[1])), :$(sigma1_name))
    $(sigma2_name) ~ NamedDist($(_distribution_to_string(m.sigma_regimes[2])), :$(sigma2_name))
    $(v.thresh_raw) ~ NamedDist(Normal(T(0), T(1)), :$(v.thresh_raw))
    $(v.innov) ~ NamedDist(MvNormal(zeros(T, M.t_N), I), :$(v.innov))
    """

    update = """
    begin
        threshold_level = T(mean(spec_registry["$(key_str)"].hyper.threshold_data)) + $(v.thresh_raw)
        innovations = $(v.innov)
        
        local T_num = promote_type(typeof($(rho1_name)), typeof($(sigma1_name)), eltype(innovations))
        $(v.latent) = zeros(T_num, M.t_N)
        
        for t in 1:M.t_N
            regime_indicator = spec_registry["$(key_str)"].hyper.threshold_data[t] > threshold_level
            curr_rho = regime_indicator ? $(rho2_name) : $(rho1_name)
            curr_sigma = regime_indicator ? $(sigma2_name) : $(sigma1_name)
            
            if t == 1
                $(v.latent)[t] = (innovations[t] * curr_sigma) / sqrt(T(1.0) - curr_rho^2 + T(noise))
            else
                $(v.latent)[t] = curr_rho * $(v.latent)[t-1] + innovations[t] * curr_sigma
            end
        end
        $(eta_target) .+= view($(v.latent), M.t_idx)
    end
    """
    return (priors=priors, update=update)
end


# Version 1.5.3 (2026-08-06)
# Purpose: Generates Turing code fragments for the `AdaptiveSmooth` component.
# Rationale: This version is updated for AD compatibility. It ensures that the coordinate
#            data `X_orig` is explicitly converted to the generic model type `T` before
#            being multiplied with the MLP weights. This prevents a `MethodError` when
#            a `Matrix{Float64}` is multiplied by a `Matrix{ForwardDiff.Dual}`.
function _generate_component_code_fragments(m::AdaptiveSmooth, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    
    h_dim = m.hidden_dim
    in_dim = spec.hyper.in_dim
    n_bins = m.nbins

    priors = """
    # Adaptive Basis Priors
    $(v.W1) ~ NamedDist(MvNormal(zeros(T, $(in_dim * h_dim)), I), :$(v.W1))
    $(v.b1) ~ NamedDist(MvNormal(zeros(T, $(h_dim)), I), :$(v.b1))
    $(v.W2) ~ NamedDist(MvNormal(zeros(T, $(h_dim * n_bins)), I), :$(v.W2))
    $(v.innov) ~ NamedDist(MvNormal(zeros(T, $(n_bins)), I), :$(v.innov))
    $(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))
    """

    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"

    update = """
    begin
        local X_orig = T.(spec_registry["$(key_str)"].hyper.coords)
        local W1 = reshape($(v.W1), $(in_dim), $(h_dim))
        local b1 = $(v.b1)
        local W2 = reshape($(v.W2), $(h_dim), $(n_bins))
        local H = tanh.((X_orig * W1) .+ b1')
        local B_adaptive = H * W2
        local scaled_coeffs = $(v.innov) .* $(v.sigma)
        local adaptive_effect = B_adaptive * scaled_coeffs
        $(eta_target) .+= adaptive_effect
    end
    """
    return (priors=priors, update=update)
end



# Version 1.5.3 (2026-08-06)
# Purpose: Generates Turing code fragments for the `Harmonic` component.
# Rationale: This version is updated for AD compatibility. It ensures that the `time_points`
#            data vector is explicitly converted to the generic model type `T` before being
#            used in calculations with model parameters. This prevents a `MethodError`
#            when a `Vector{Float64}` is multiplied by a `ForwardDiff.Dual` number.
function _generate_component_code_fragments(m::Harmonic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx; prefix=prefix)
    nharmonics = m.nharmonics

    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]
    
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(v.amplitude_raw) ~ NamedDist(filldist($(_distribution_to_string(m.amplitude)), $(nharmonics)), :$(v.amplitude_raw))")
        push!(priors_acc, "$(v.phase) ~ NamedDist(filldist($(_distribution_to_string(m.phase)), $(nharmonics)), :$(v.phase))")
        
        if m.period isa Vector
            for i in 1:nharmonics; period_var_i = Symbol("period_$(spec.key)_$(i)"); push!(priors_acc, "$(period_var_i) ~ NamedDist($(_distribution_to_string(m.period[i])), :$(period_var_i))"); end
        elseif m.period isa UnivariateDistribution
            push!(priors_acc, "$(v.period) ~ NamedDist($(_distribution_to_string(m.period)), :$(v.period))")
        end
    end
    
    priors_str = join(priors_acc, "\n")
    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "u_idx" 
    
    local period_access_logic
    if m.period isa Real; period_access_logic = "local period_vals = fill(T($(m.period)), $(nharmonics))";
    elseif m.period isa UnivariateDistribution; period_access_logic = "local period_vals = fill($(v.period), $(nharmonics))";
    elseif m.period isa Vector; period_vars = ["period_$(spec.key)_$(i)" for i in 1:nharmonics]; period_access_logic = "local period_vals = [$(join(period_vars, ", "))]"; end

    update_str = """
    begin
        local harmonic_effect = zeros(T, length(M.$(index_var)))
        local amplitudes = abs.($(v.amplitude_raw))
        local phases = $(v.phase)
        local time_points = T.(M.$(index_var))
        $(period_access_logic)
        
        for m_idx in 1:$(nharmonics)
            local phase_rad = T(2.0) * pi * phases[m_idx]
            local angle = (T(2.0) * pi / period_vals[m_idx]) .* time_points
            local beta_cos = amplitudes[m_idx] * cos(phase_rad)
            local beta_sin = amplitudes[m_idx] * sin(phase_rad)
            harmonic_effect .+= beta_cos .* cos.(angle) .+ beta_sin .* sin.(angle)
        end
        $(eta_update_target) .+= harmonic_effect
    end
    """
    return (priors=priors_str, update=update_str)
end



# Version 1.5.3 (2026-08-06)
# Purpose: Generates Turing code for standard GMRF components.
# Rationale: This version is updated for AD compatibility. It converts the precision matrix
#            to a dense matrix before the Cholesky decomposition when the matrix contains
#            Dual numbers (i.e., for dynamic components). This avoids calling the sparse
#            Cholesky factorization from CHOLMOD, which does not support Dual types,
#            resolving the `TypeError`.
function _generate_component_code_fragments(m::Union{Besag, ICAR, BCGN, Cyclic}, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="", generate_eta_update::Bool=true)
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    params = spec.params
    n_latent = size(spec.Q_template, 1)
    is_multivariate = (arch == "multivariate")
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))
    is_shared = get(params, :shared, false)

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        if hasproperty(m, :sigma)
            push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        end
        if hasproperty(m, :rho)
            push!(priors_acc, "$(v.rho) ~ NamedDist($(_distribution_to_string(m.rho)), :$(v.rho))")
        end
    end
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.raw))")
    priors_str = join(priors_acc, "\n    ")

    local index_var
    if spec.structure == :spatial; index_var = "s_idx";
    elseif spec.structure == :temporal; index_var = (typeof(m) <: Cyclic) ? "u_idx" : "t_idx";
    elseif spec.structure == :mixed; index_var = "mixed_idx_$(spec.var)";
    else; index_var = string(spec.structure) * "_idx"; end

    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    local effect_app_str
    if generate_eta_update
        if spec.structure == :smooth
            effect_app_str = "$(eta_target) .+= M.basis_matrices[:$(spec.var)] * $(v.latent)"
        else
            effect_app_str = "$(eta_target) .+= view($(v.latent), M.$(index_var))"
        end
    else
        effect_app_str = ""
    end

    local update_str
    if get(spec, :is_static, false)
        update_str = """
        begin
            # Static Component: $(spec.key) - Using pre-computed Cholesky factor
            local F_static = spec_registry["$(spec.key)"].cholesky_factor
            $(v.latent) = $(v.sigma) .* (F_static.L' \\ $(v.raw))
            $(effect_app_str)
        end
        """
    else
        flow_direction_kwarg = (m isa NetworkFlow) ? ", flow_direction=:$(m.flow_direction)" : "" # Special case for NetworkFlow
        update_str = """
        begin
            # Dynamic Component: $(spec.key)
            local Q_template = spec_registry["$(spec.key)"].Q_template
            local model_type = spec_registry["$(spec.key)"].component_obj |> typeof |> Symbol
            local rho_value = $(hasproperty(m, :rho) ? v.rho : "nothing")
            
            local Q_final = recompose_precision(model_type, Q_template, T(1.0); extra_param=rho_value$(flow_direction_kwarg))
            # Convert sparse Q to dense before cholesky to ensure AD compatibility
            local F_dynamic = cholesky(Symmetric(Matrix(Q_final) + noise * I))
            
            $(v.latent) = $(v.sigma) .* (F_dynamic.L' \\ $(v.raw))
            $(effect_app_str)
        end
        """
    end

    return (priors=priors_str, update=update_str)
end


# Version 1.5.4 (2026-08-06)
# Purpose: Generates Turing code fragments for the `BYM2` component.
# Rationale: This version is updated for AD compatibility. It converts the sparse precision
#            matrix `Q_template` to a dense `Matrix` before the Cholesky decomposition.
#            This avoids calling the sparse Cholesky factorization from CHOLMOD, which does
#            not support Dual types, resolving the `CanonicalIndexError` during sampling.
function _generate_component_code_fragments(m::BYM2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="", generate_eta_update::Bool=true)
    key_str = string(spec.key)
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    params = spec.params
    n_latent = isnothing(spec.Q_template) ? 0 : size(spec.Q_template, 1)
    is_multivariate = (arch == "multivariate")
    is_first_outcome = (outcome_idx == 1)
    is_shared = get(params, :shared, false)

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        push!(priors_acc, "$(v.rho) ~ NamedDist($(_distribution_to_string(m.rho)), :$(v.rho))")
    end
    push!(priors_acc, "$(v.struct) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.struct))")
    push!(priors_acc, "$(v.iid) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.iid))")
    priors_str = join(priors_acc, "\n    ")

    index_var = (spec.structure == :spatial) ? "s_idx" : string(spec.structure) * "_idx"
    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"

    update_str = """
    begin
        # BYM2 model for component: $(key_str)
        local Q_template = spec_registry["$(spec.key)"].Q_template
        
        if !isnothing(Q_template) && $(n_latent) > 0
            # 1. Reconstruct the structured (ICAR) component from its raw innovations.
            #    Convert to dense Matrix to ensure AD-compatible Cholesky factorization.
            local F = cholesky(Symmetric(Matrix(Q_template) + noise * I))
            local struct_latent = F.L' \\ $(v.struct)
            
            # 2. Apply a soft sum-to-zero constraint for identifiability.
            Turing.@addlogprob! logpdf(Normal(T(0), T(0.001) * $(n_latent)), sum(struct_latent))
            
            # 3. Combine structured and unstructured components using the Riebler parameterization.
            local bym2_effect = $(v.sigma) .* (sqrt($(v.rho)) .* struct_latent .+ sqrt(T(1.0) - $(v.rho)) .* $(v.iid))
            
            # 4. Add the final effect to the linear predictor.
            $(eta_target) .+= view(bym2_effect, M.$(index_var))
        end
    end
    """
    return (priors=priors_str, update=update_str)
end




# Version 1.5.3 (2026-08-06)
# Purpose: Generates Turing code fragments for the `SPDE` component.
# Rationale: This version is updated for AD compatibility. It converts the precision matrix
#            to a dense matrix before the Cholesky decomposition when the matrix contains
#            Dual numbers. This avoids calling the sparse Cholesky factorization from CHOLMOD,
#            which does not support Dual types, resolving the `TypeError`.
function _generate_component_code_fragments(m::SPDE, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]

    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        if m.kappa isa Vector
            kappa_priors_str = join([_distribution_to_string(p) for p in m.kappa], ", ")
            push!(priors_acc, "$(v.kappa) ~ NamedDist(Product([$(kappa_priors_str)]), :$(v.kappa))")
        else
            push!(priors_acc, "$(v.kappa) ~ NamedDist($(_distribution_to_string(m.kappa)), :$(v.kappa))")
        end
    end

    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.raw))")
    priors = join(priors_acc, "\n")
    
    index_var = spec.structure == :spatial ? "s_idx" : string(spec.structure) * "_idx"
    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    effect_application_str = spec.structure == :smooth ? "$(eta_target) .+= M.basis_matrices[:$(spec.var)] * $(v.latent)" : "$(eta_target) .+= view($(v.latent), M.$(index_var))"

    update = """
    begin
        local Q_template = spec_registry["$(key_str)"].Q_template
        local m_type = spec_registry["$(key_str)"].component_obj |> typeof |> Symbol
        local kappa_val = $(v.kappa)
        
        local Q_final = recompose_precision(m_type, Q_template, T(1.0); extra_param=kappa_val)
        # Convert sparse Q to dense before cholesky to ensure AD compatibility
        local F = cholesky(Symmetric(Matrix(Q_final) + noise * I))
        $(v.latent) = $(v.sigma) .* (F.L' \\ $(v.raw))
        $(effect_application_str)
    end
    """
    
    return (priors=priors, update=update)
end


# Version 1.5.3 (2026-08-06)
# Purpose: Generates Turing code fragments for the `LocalAdaptive` component.
# Rationale: This version is updated for AD compatibility. It converts the precision matrix
#            to a dense matrix before the Cholesky decomposition when the matrix contains
#            Dual numbers. This avoids calling the sparse Cholesky factorization from CHOLMOD,
#            which does not support Dual types, resolving the `TypeError`.
function _generate_component_code_fragments(m::LocalAdaptive, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    mu_clusters_raw_name = v.innov

    priors_acc = String[]

    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        push!(priors_acc, "$(v.rho) ~ NamedDist($(_distribution_to_string(m.rho)), :$(v.rho))")
        
        n_clusters = spec.hyper.n_clusters
        push!(priors_acc, "$(mu_clusters_raw_name) ~ NamedDist(MvNormal(zeros(T, $(n_clusters)), I), :$(mu_clusters_raw_name))")
    end

    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.raw))")
    priors_str = join(priors_acc, "\n")

    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "s_idx"

    update_str = """
    begin
        # LocalAdaptive model for $(key_str)
        
        local n_clusters = spec_registry["$(key_str)"].hyper.n_clusters
        Turing.@addlogprob! logpdf(Normal(T(0), T(0.001) * n_clusters), sum($(mu_clusters_raw_name)))
        
        local mean_vector = $(mu_clusters_raw_name)[M.cluster_assignments]

        local Q_template = spec_registry["$(key_str)"].hyper.L_template
        local m_type = spec_registry["$(key_str)"].component_obj |> typeof |> Symbol
        local rho_val = $(v.rho)
        local Q_final = recompose_precision(m_type, Q_template, T(1.0); extra_param=rho_val)
        
        # Convert sparse Q to dense before cholesky to ensure AD compatibility
        local F = cholesky(Symmetric(Matrix(Q_final) + noise * I))
        local latent_field_centered_part = F.L' \\ $(v.raw)
        $(v.latent) = mean_vector .+ latent_field_centered_part

        $(v.latent) .*= $(v.sigma)
        $(eta_target) .+= view($(v.latent), M.$(index_var))
    end
    """
    
    return (priors=priors_str, update=update_str)
end

 

function _generate_component_code_fragments(m::Wavelet, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    # Purpose: Generates Turing code fragments for the `Wavelet` component.
    # Rationale: This version is updated for AD compatibility. It ensures that the wavelet
    #            basis matrix `B_wavelet` is explicitly converted to the generic model type `T`
    #            before being multiplied with the latent coefficients. This prevents a `MethodError`
    #            when a `Matrix{Float64}` is multiplied by a `Vector{ForwardDiff.Dual}`.
    # v1.0.2 (2026-08-03)
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    
    n_basis = m.nbins
    
    nbins_per_dim_str = if haskey(spec.hyper, :nbins_per_dim)
        string(spec.hyper.nbins_per_dim)
    elseif haskey(spec.hyper, :coords)
        string([round(Int, n_basis^(1/size(spec.hyper.coords, 2)))])
    else
        string([n_basis])
    end

    priors_acc = String[]
    push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
    
    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors_acc, "$(v.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(v.ls))")
    else
        push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
    end
    
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_basis)), I), :$(v.raw))")
    priors = join(priors_acc, "\n")

    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"

    update = """
    begin
        # --- Wavelet Component: $(key_str) ---
        local coords = spec_registry["$(key_str)"].hyper.coords
        local nbins_per_dim = $(nbins_per_dim_str)
        
        local B_wavelet = bstm_tensor_product_wavelet_basis(coords, nbins_per_dim, Symbol("$(m.family)"), $(v.ls))
        local Q_penalty = build_structure_template(:rw2, $(n_basis)).matrix
        local F_penalty = cholesky(Symmetric(Q_penalty + noise * I))
        
        local wavelet_coeffs = Matrix(sparse(F_penalty.L))' \\ $(v.raw)
        local scaled_coeffs = wavelet_coeffs .* $(v.sigma)
        local wavelet_effect = T.(B_wavelet) * scaled_coeffs
        
        $(eta_target) .+= wavelet_effect
    end
    """
    return (priors=priors, update=update)
end


function _generate_component_code_fragments(m::FFT, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    # Purpose: Generates Turing code fragments for the `FFT` component.
    # Rationale: This version is updated for AD compatibility. It ensures that the coordinate
    #            data is explicitly converted to the generic model type `T` before being
    #            used in the calculation of the Fourier basis matrix. This prevents a `MethodError`
    #            when `Float64` data is used in operations that result in `ForwardDiff.Dual` numbers.
    # v1.0.2 (2026-08-03)
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    n_basis = m.nbins
    
    coords = spec.hyper.coords
    n_dims = size(coords, 2)
    n_obs = size(coords, 1)

    priors_acc = String[]
    push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors_acc, "$(v.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(v.ls))")
    else
        push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
    end
    
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_basis)), I), :$(v.raw))")
    priors = join(priors_acc, "\n")

    local basis_gen_code
    if n_dims == 1
        basis_gen_code = """
        local vals = T.(spec_registry["$(key_str)"].hyper.coords[:, 1])
        local ls_val = $(v.ls) isa Real ? $(v.ls) : $(v.ls)[1]
        local t_coords = vals ./ ls_val
        local B_fft = zeros(T, $(n_obs), $(n_basis))
        for m_fft in 1:div($(n_basis), 2)
            if (2*m_fft) <= $(n_basis)
                local arg = (T(2.0) * pi * m_fft) .* t_coords
                B_fft[:, 2*m_fft-1] = sin.(arg)
                B_fft[:, 2*m_fft]   = cos.(arg)
            end
        end
        """
    elseif n_dims >= 2
        nbins_per_dim = get(spec.hyper, :nbins_per_dim, [round(Int, n_basis^(1/n_dims)) for _ in 1:n_dims])
        n_marginal_x = nbins_per_dim[1]
        n_marginal_y = nbins_per_dim[2]
        basis_gen_code = """
        local coords = T.(spec_registry["$(key_str)"].hyper.coords)
        local ls_vec = $(v.ls) isa Real ? fill($(v.ls), $n_dims) : $(v.ls)
        local nx = coords[:, 1] ./ ls_vec[1]
        local ny = coords[:, 2] ./ ls_vec[2]
        local B_fft = zeros(T, $(n_obs), $(n_basis))
        local idx = 1
        for my in 1:$(n_marginal_y), mx in 1:$(n_marginal_x)
            if idx + 1 <= $(n_basis)
                local arg = mx .* nx .+ my .* ny
                B_fft[:, idx]   = sin.(T(2.0) * pi * arg)
                B_fft[:, idx+1] = cos.(T(2.0) * pi * arg)
                idx += 2
            end
        end
        """
    end

    update = """
    begin
        $(basis_gen_code)
        local Q_penalty = build_structure_template(:rw2, $(n_basis)).matrix # RW2 penalty on coefficients
        local F_penalty = cholesky(Symmetric(Q_penalty + noise * I)) # Cholesky of penalty matrix
        local fft_coeffs = Matrix(sparse(F_penalty.L))' \\ $(v.raw)
        local scaled_coeffs = fft_coeffs .* $(v.sigma)
        local fft_effect = B_fft * scaled_coeffs
        $(arch == "multivariate" ? "eta_latent[:, $(outcome_idx)]" : "eta") .+= fft_effect
    end
    """
    return (priors=priors, update=update)
end


 

function _generate_component_code_fragments(m::Moran, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"
    
    n_latent = size(spec.hyper.moran_eigenvectors, 1)
    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    local sigma_name_str, coeffs_name_str, latent_name_str
    if is_multivariate && !is_shared
        sigma_name_str = "$(prefixed_key)_sigma_$(outcome_idx)"
        coeffs_name_str = "$(prefixed_key)_coeffs_$(outcome_idx)"
        latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
    else
        sigma_name_str = "$(prefixed_key)_sigma"
        if is_multivariate
            coeffs_name_str = "$(prefixed_key)_coeffs_$(outcome_idx)"
            latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
        else
            coeffs_name_str = "$(prefixed_key)_coeffs"
            latent_name_str = "$(prefixed_key)_latent"
        end
    end

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(Symbol(sigma_name_str)) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(Symbol(sigma_name_str)))")
    end
    
    # Priors for the coefficients of the Moran eigenvectors
    push!(priors_acc, "$(Symbol(coeffs_name_str)) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(Symbol(coeffs_name_str)))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "s_idx"

    update_str = """
    begin
        # Moran eigenvector spectral model for $(key_str)
        local moran_eigenvectors = spec_registry["$(key_str)"].hyper.moran_eigenvectors
        
        # The latent effect is a linear combination of the eigenvectors,
        # with coefficients scaled by sigma.
        $(v.latent) = T.(moran_eigenvectors) * ($(coeffs_name) .* $(v.sigma))
        
        $(eta_update_target) .+= view($(v.latent), M.$(index_var))
    end
    """
    
    return (priors=priors_str, update=update_str)
end



function _generate_component_code_fragments(m::Spherical, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    # Specialized implementation for the Spherical Gaussian Process model.
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    # Retrieve centralized variable names
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        push!(priors_acc, "$(v.range) ~ NamedDist($(_distribution_to_string(m.range)), :$(v.range))")
    end

    n_latent = size(spec.hyper.coords, 1)
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.raw))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # Spherical GP model for $(key_str)
        coords = spec_registry["$(key_str)"].hyper.coords
        
        # Compute pairwise Euclidean distances
        dist_matrix = pairwise(Euclidean(), coords, dims=1)
        
        # Compute spherical kernel matrix
        h = T.(dist_matrix) ./ $(v.range)
        K = zeros(T, size(h))
        mask = h .< 1.0
        K[mask] = ($(v.sigma)^2) .* (1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3)
        K += (noise * I)
        
        F = cholesky(Symmetric(K))
        $(v.latent) = F.L * $(v.raw)
        
        $(eta_update_target) .+= $(v.latent)
    end
    """
    
    return (priors=priors_str, update=update_str)
end



# Version 1.5.3 (2026-08-06)
# Purpose: Generates Turing code fragments for the `Warp` component.
# Rationale: This version ensures type stability by explicitly casting `coords` to `T`
#            before use in calculations with `ForwardDiff.Dual` numbers.
function _generate_component_code_fragments(m::Warp, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    # This model has many specific parameters not in the standard naming convention.
    # Manual naming is clearer here.
    beta_main_name = Symbol("$(prefixed_key)_beta_main_$(outcome_idx)")
    W_main_name = Symbol("$(prefixed_key)_W_main_$(outcome_idx)")
    b_main_name = Symbol("$(prefixed_key)_b_main_$(outcome_idx)")
    beta_warp_name = Symbol("$(prefixed_key)_beta_warp_$(outcome_idx)")
    W_warp_name = Symbol("$(prefixed_key)_W_warp_$(outcome_idx)")
    b_warp_name = Symbol("$(prefixed_key)_b_warp_$(outcome_idx)")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)

    priors_acc = String[]

    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
    end

    n_features = m.n_features
    in_dims = size(spec.hyper.coords, 2)

    # Priors for the warping function's RFF parameters
    push!(priors_acc, "$(W_warp_name) ~ NamedDist(MvNormal(zeros(T, $(in_dims * n_features)), I), :$(W_warp_name))")
    push!(priors_acc, "$(b_warp_name) ~ NamedDist(MvNormal(zeros(T, $(n_features)), I), :$(b_warp_name))")
    push!(priors_acc, "$(beta_warp_name) ~ NamedDist(MvNormal(zeros(T, $(n_features)), I), :$(beta_warp_name))")

    # Priors for the main GP's RFF parameters
    push!(priors_acc, "$(W_main_name) ~ NamedDist(MvNormal(zeros(T, $(in_dims * n_features)), I), :$(W_main_name))")
    push!(priors_acc, "$(b_main_name) ~ NamedDist(MvNormal(zeros(T, $(n_features)), I), :$(b_main_name))")
    push!(priors_acc, "$(beta_main_name) ~ NamedDist(MvNormal(zeros(T, $(n_features)), $(v.sigma)^2 * I), :$(beta_main_name))")
    
    priors_str = join(priors_acc, "\n")
    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        local coords = T.(spec_registry["$(key_str)"].hyper.coords)
        
        # 1. Construct and apply the warping function
        local W_warp_matrix = reshape($(W_warp_name), $(in_dims), $(n_features))
        local Phi_warp = sqrt(T(2.0) / $(n_features)) .* cos.((coords * W_warp_matrix) .+ $(b_warp_name)')
        local warping_effect = Phi_warp * $(beta_warp_name)
        local coords_warped = coords .+ warping_effect

        # 2. Construct the main GP on the warped coordinates
        local W_main_matrix = reshape($(W_main_name), $(in_dims), $(n_features)) ./ $(v.ls)
        local Phi_main = sqrt(T(2.0) / $(n_features)) .* cos.((coords_warped * W_main_matrix) .+ $(b_main_name)')
        local main_effect = Phi_main * $(beta_main_name)

        $(eta_update_target) .+= main_effect
    end
    """
    
    return (priors=priors_str, update=update_str)
end
 
 

# Version 1.5.3 (2026-08-06)
# Purpose: Generates Turing code fragments for the `SVGP` (Sparse Variational Gaussian Process) component.
# Rationale: This version ensures type stability by explicitly casting `Z_coords` and `X_coords` to `T`
#            before use in `evaluate_kernel_matrix` and `evaluate_cross_kernel_matrix`.
function _generate_component_code_fragments(m::SVGP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    # Specialized implementation for the SVGP (Sparse Variational Gaussian Process) model.
    # This implementation is functionally similar to FITC for the purpose of MCMC sampling,
    # providing a non-centered parameterization for a sparse GP.
    key_str = string(spec.key)
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    u_raw_name = v.raw
    f_raw_name = v.innov

    priors_acc = String[]

    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        if m.lengthscale isa Vector
            ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
            push!(priors_acc, "$(v.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(v.ls))")
        else
            push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
        end
    end

    n_inducing = m.n_inducing
    n_latent = size(spec.Q_template, 1)
    
    push!(priors_acc, "$(u_raw_name) ~ NamedDist(MvNormal(zeros(T, $(n_inducing)), I), :$(u_raw_name))")
    push!(priors_acc, "$(f_raw_name) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(f_raw_name))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # SVGP sparse GP model for $(key_str)
        local X_coords = T.(spec_registry["$(key_str)"].Q_template)
        local Z_coords = T.(spec_registry["$(key_str)"].hyper.Z_inducing)
        
        local K_UU = evaluate_kernel_matrix(Z_coords, $(v.sigma), $(v.ls), Symbol("$(m.kernel)"), noise)
        local K_XU = evaluate_cross_kernel_matrix(X_coords, Z_coords, $(v.sigma), $(v.ls), Symbol("$(m.kernel)"))
        
        local L_UU = cholesky(Symmetric(K_UU)).L
        local u_latent = L_UU * $(u_raw_name)
        
        local K_UU_inv_u = K_UU \\ u_latent
        local mean_f = K_XU * K_UU_inv_u
        
        local diag_K_XX = fill($(v.sigma)^2, $(n_latent))
        local tmp = (L_UU' \\ K_XU')'
        local diag_Q_ff = sum(tmp.^2, dims=2)
        local lambda_diag = diag_K_XX - vec(diag_Q_ff)
        
        $(v.latent) = mean_f + sqrt.(max.(lambda_diag, T(0.0)) .+ noise) .* $(f_raw_name)
        
        $(eta_update_target) .+= $(v.latent)
    end
    """
    
    return (priors=priors_str, update=update_str)
end



 
# Version 1.5.3 (2026-08-06)
# Purpose: Generates Turing code fragments for the `Nystrom` sparse Gaussian Process model.
# Rationale: This version ensures type stability by explicitly casting `Z_coords` and `X_coords` to `T`
#            before use in `evaluate_kernel_matrix` and `evaluate_cross_kernel_matrix`.
function _generate_component_code_fragments(m::Nystrom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    # Specialized implementation for the Nystrom sparse Gaussian Process model.
    # This method uses a low-rank approximation based on inducing points.
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    # Retrieve centralized variable names
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    # For Nystrom, 'raw' holds the innovations for the inducing points.
    v_latent_name = v.raw

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        if m.lengthscale isa Vector; ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", "); push!(priors_acc, "$(v.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(v.ls))");
        else; push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))"); end
    end

    n_inducing = m.n_inducing
    push!(priors_acc, "$(v_latent_name) ~ NamedDist(MvNormal(zeros(T, $(n_inducing)), I), :$(v_latent_name))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # Nystrom sparse GP model for $(key_str)
        local X_coords = T.(spec_registry["$(key_str)"].hyper.coords)
        local Z_coords = T.(spec_registry["$(key_str)"].hyper.Z_inducing)
        
        local K_UU = evaluate_kernel_matrix(Z_coords, $(v.sigma), $(v.ls), Symbol("$(m.kernel)"), noise)
        local K_XU = evaluate_cross_kernel_matrix(X_coords, Z_coords, $(v.sigma), $(v.ls), Symbol("$(m.kernel)"))
        
        local L_UU = cholesky(Symmetric(K_UU)).L
        
        # Project standard normal noise through the Nystrom approximation
        # f(X) ≈ K_XU * inv(K_UU) * u, where u ~ N(0, K_UU)
        # Using non-centered parameterization: u = L_UU * v, where v ~ N(0, I)
        # f(X) ≈ K_XU * inv(K_UU) * L_UU * v = K_XU * inv(L_UU' * L_UU) * L_UU * v = K_XU * (L_UU' \\ v)
        $(v.latent) = K_XU * (L_UU' \\ $(v_latent_name))
        $(eta_update_target) .+= $(v.latent)
    end
    """
    
    return (priors=priors_str, update=update_str)
end



# Version 1.5.3 (2026-08-06)
# Purpose: Generates Turing code fragments for the `FITC` sparse Gaussian Process model.
# Rationale: This version ensures type stability by explicitly casting `Z_coords` and `X_coords` to `T`
#            before use in `evaluate_kernel_matrix` and `evaluate_cross_kernel_matrix`.
function _generate_component_code_fragments(m::FITC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    key_str = string(spec.key)
    v = generate_full_variable_names(spec, arch, outcome_idx; prefix=prefix)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    # Retrieve centralized variable names
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    # For FITC, 'raw' holds innovations for inducing points, 'innov' for the final field.
    u_raw_name = v.raw
    f_raw_name = v.innov

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        if m.lengthscale isa Vector
            ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
            push!(priors_acc, "$(v.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(v.ls))")
        else
            push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
        end
    end

    n_inducing = m.n_inducing
    n_latent = size(spec.Q_template, 1) # Number of data points
    
    # Priors for the latent values at inducing points and the final field innovations
    push!(priors_acc, "$(u_raw_name) ~ NamedDist(MvNormal(zeros(T, $(n_inducing)), I), :$(u_raw_name))")
    push!(priors_acc, "$(f_raw_name) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(f_raw_name))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # FITC sparse GP model for $(key_str)
        local X_coords = T.(spec_registry["$(key_str)"].Q_template)
        local Z_coords = T.(spec_registry["$(key_str)"].hyper.Z_inducing)
        
        # 1. Compute kernel matrices
        local K_UU = evaluate_kernel_matrix(Z_coords, $(v.sigma), $(v.ls), Symbol("$(m.kernel)"), noise)
        local K_XU = evaluate_cross_kernel_matrix(X_coords, Z_coords, $(v.sigma), $(v.ls), Symbol("$(m.kernel)"))
        
        # 2. Sample latent values at inducing points (non-centered)
        local L_UU = cholesky(Symmetric(K_UU)).L
        local u_latent = L_UU * $(u_raw_name)
        
        # 3. Compute conditional mean and variance for FITC
        #    μ_f = K_XU * inv(K_UU) * u_latent
        #    diag_cov_f = diag(K_XX - K_XU * inv(K_UU) * K_XU')
        
        local K_UU_inv_u = K_UU \\ u_latent
        local mean_f = K_XU * K_UU_inv_u
        
        # Compute diagonal of K_XX - Q_ff efficiently
        # diag(K_XX) is sigma^2 for stationary kernels.
        local diag_K_XX = fill($(v.sigma)^2, $(n_latent))
        
        # diag(K_XU * inv(K_UU) * K_XU') = sum((K_XU / L_UU.U).^2, dims=2)
        local tmp = (L_UU' \\ K_XU')'
        local diag_Q_ff = sum(tmp.^2, dims=2)
        
        local lambda_diag = diag_K_XX - vec(diag_Q_ff)
        
        # 4. Sample final latent field (non-centered)
        $(v.latent) = mean_f + sqrt.(max.(lambda_diag, T(0.0)) .+ noise) .* $(f_raw_name)
        
        $(eta_update_target) .+= $(v.latent)
    end
    """
    
    return (priors=priors_str, update=update_str)
end

 
# Version 1.5.3 (2026-08-06)
# Purpose: Generates Turing code fragments for the `Hyperbolic` component.
# Rationale: This version ensures type stability by explicitly casting `coords` to `T`
#            before use in `evaluate_hyperbolic_kernel_matrix`.
function _generate_component_code_fragments(m::Hyperbolic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    # Specialized implementation for the Hyperbolic Gaussian Process model.
    # This model computes distances in a hyperbolic space (Poincaré disk) before applying a kernel.
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    # Retrieve centralized variable names
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        # Curvature is fixed for now, but could be given a prior.
    end

    n_latent = size(spec.hyper.coords, 1)
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.raw))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # Hyperbolic GP model for $(key_str)
        local coords = T.(spec_registry["$(key_str)"].hyper.coords)
        local curvature = T($(m.curvature)) # Fixed curvature
        
        local K = evaluate_hyperbolic_kernel_matrix(coords, $(v.sigma), curvature, noise)
        local F = cholesky(Symmetric(K))
        $(v.latent) = F.L * $(v.raw)
        
        $(eta_update_target) .+= $(v.latent)
    end
    """
    
    return (priors=priors_str, update=update_str)
end
 
 

# Version 1.5.3 (2026-08-06)
# Purpose: Generates Turing code fragments for the `ExponentialDecay` component.
# Rationale: This version ensures type stability by explicitly casting `coords` to `T`
#            before use in `pairwise(Euclidean(), ...)`.
function _generate_component_code_fragments(m::ExponentialDecay, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    # Specialized implementation for the Exponential Decay GP model.
    # This model uses an exponential kernel based on Euclidean distances between coordinates.
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    # Retrieve centralized variable names
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
    end

    n_latent = size(spec.hyper.coords, 1)
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.raw))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # Exponential Decay GP model for $(key_str)
        local coords = T.(spec_registry["$(key_str)"].hyper.coords)
        
        # Compute pairwise Euclidean distances
        local dist_matrix = pairwise(Euclidean(), coords, dims=1)
        
        # Compute exponential decay kernel matrix
        local K = ($(v.sigma)^2) .* exp.(-dist_matrix ./ $(v.ls)) .+ (noise * I)
        
        local F = cholesky(Symmetric(K))
        $(v.latent) = F.L * $(v.raw)
        
        $(eta_update_target) .+= $(v.latent)
    end
    """
    
    return (priors=priors_str, update=update_str)
end
 
 
# Version 1.5.3 (2026-08-06)
# Purpose: Generates Turing code fragments for the `DAG` component.
# Rationale: This version is updated to be type-stable for automatic differentiation.
#            The latent field vector `$(v.latent)` is now initialized with the promoted
#            numeric type of its inputs (`rho_val`, `innovations`), which could be
#            `ForwardDiff.Dual`. This resolves a `MethodError` that occurred when
#            trying to assign a `Dual` number to a `Vector{Float64}`.
#            `parent_effect` is initialized with `zero(T_num)` for type consistency.
function _generate_component_code_fragments(m::DAG, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        push!(priors_acc, "$(v.rho) ~ NamedDist($(_distribution_to_string(m.rho)), :$(v.rho))")
    end
    push!(priors_acc, "$(v.innov) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.innov))")
    priors_str = join(priors_acc, "\n    ")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "s_idx"

    update_str = """
    begin
        # DAG model for $(key_str) using forward substitution.
        local W_dag = spec_registry["$(key_str)"].Q_template
        local innovations = $(v.innov)
        local rho_val = $(v.rho)
        local sigma_val = $(v.sigma)

        local T_num = promote_type(typeof(rho_val), eltype(innovations))
        local $(v.latent) = zeros(T_num, $(n_latent))

        # Assumes W_dag is lower triangular, representing a valid DAG ordering.
        for i in 1:$(n_latent)
            local parent_effect = zero(T_num)
            # Efficiently iterate over non-zero elements in the row of the sparse matrix
            for j_ptr in nzrange(W_dag, i)
                parent_idx = W_dag.rowval[j_ptr]
                parent_effect += W_dag.nzval[j_ptr] * $(v.latent)[parent_idx]
            end
            $(v.latent)[i] = rho_val * parent_effect + innovations[i]
        end
        $(v.latent) .*= sigma_val
        $(eta_update_target) .+= view($(v.latent), M.$(index_var))
    end
    """
    
    return (priors=priors_str, update=update_str)
end


"""
    _generate_component_code_fragments(m::CustomComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for a user-defined `CustomComponent`.

# Rationale for New Implementation
The previous implementation incorrectly dispatched `CustomComponent` to a generic GMRF
generator, ignoring the user-provided code. This new, specialized function correctly
implements the intended behavior by directly injecting the user's code into the model.

The `code_fragment` provided by the user in the `custom()` module is expected to be a
complete and valid block of Turing model code. This block is inserted directly into the
model's main assembly block. The user is responsible for defining any necessary priors
and update logic within this fragment. The function returns an empty `priors` string
and places the entire user code into the `update` string, as Turing does not
distinguish between these contexts within the `@model` macro.
"""
function _generate_component_code_fragments(m::CustomComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    # The user's code fragment is expected to be a self-contained block
    # that includes both prior definitions and update logic for the linear predictor.
    
    user_code = m.code_fragment
    
    if isempty(strip(user_code))
        @warn "Custom component '$(spec.key)' was specified but the `code_fragment` is empty. This component will have no effect."
        return (priors="", update="")
    end

    # The entire user code is treated as an update block.
    # Turing doesn't distinguish between prior and update sections inside the @model macro,
    # so this is a valid approach. The user must ensure their code is correct and
    # that any new parameter names are unique to avoid collisions.
    update_str = """
    begin
        # --- Custom Code Block for $(spec.key) ---
        $(user_code)
    end
    """
    
    # Return an empty priors string as all logic is contained in the update block.
    return (priors="", update=update_str)
end


# Version 1.5.4 (2026-08-06)
# Purpose: Generates Turing code fragments for the `LGCP` component.
# Rationale: This version removes the explicit cast `T(...)` from the observed data
#            `y_st` and grid areas `A_s` inside the `logpdf` calculation. This prevents a
#            `MethodError` during automatic differentiation when these data-derived values
#            are involved in operations with `Dual` numbers.
function _generate_component_code_fragments(m::LGCP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)

    is_spatiotemporal = hasproperty(spec.hyper, :temporal_spec)
    n_latent_dims = is_spatiotemporal ? "M.s_N * M.t_N" : "M.s_N"

    priors = """
    # LGCP Intensity Field Priors
    $(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))
    $(v.raw) ~ NamedDist(MvNormal(fill(zero(T), $(n_latent_dims)), I), :$(v.raw))
    """

    update = """
    begin
        # LGCP Model: $(key_str)
        local latent_field_st = fill(zero(T), M.s_N, M.t_N)
        
        # 1. Reconstruct the latent spatiotemporal field Z(s,t)
        if $(is_spatiotemporal)
            # Spatiotemporal case with Kronecker solver
            local s_spec = spec_registry["$(key_str)"].hyper.inner_spec
            local t_spec = spec_registry["$(key_str)"].hyper.temporal_spec
            
            local C_s = cholesky(Symmetric(Matrix(s_spec.Q_template) + noise * I))
            
            local t_model_type = t_spec.component_obj |> typeof |> Symbol
            local t_rho_var_name = "rho_" * string(t_spec.key)
            local Q_t_final = recompose_precision(t_model_type, t_spec.Q_template, T(1.0); extra_param=getfield(@__MODULE__, Symbol(t_rho_var_name)))
            local C_t = cholesky(Symmetric(Matrix(Q_t_final) + noise * I))
            
            local Z_matrix = reshape($(v.raw), M.s_N, M.t_N)
            
            local tmp_spatial = C_s.L' \\ Z_matrix
            latent_field_st = (transpose(C_t.L' \\ transpose(tmp_spatial))) .* $(v.sigma)
        else
            # Purely spatial case
            local Q_lgcp = spec_registry["$(key_str)"].hyper.inner_spec.Q_template
            local F_lgcp = cholesky(Symmetric(Matrix(Q_lgcp) + noise * I))
            local spatial_component = $(v.sigma) .* (F_lgcp.L' \\ $(v.raw))
            latent_field_st = repeat(spatial_component, 1, M.t_N)
        end

        # 2. Assemble the full log-intensity surface.
        local log_intensity_surface = fill(zero(T), M.s_N, M.t_N)
        for t in 1:M.t_N, s in 1:M.s_N
            obs_indices = findall(i -> M.s_idx[i] == s && M.t_idx[i] == t, 1:N)
            base_contribution = isempty(obs_indices) ? zero(T) : mean(view(eta, obs_indices))
            log_intensity_surface[s, t] = base_contribution + latent_field_st[s, t]
        end

        # 4. Point Process Likelihood Evaluation
        local grid_areas = spec_registry["$(key_str)"].hyper.areas
        for t in 1:M.t_N, s in 1:M.s_N
            local y_st = M.y_obs[s, t]
            local A_s = grid_areas[s]
            local Z_st = log_intensity_surface[s, t]
            
            # Do NOT cast y_st or A_s to T, as this breaks AD.
            Turing.@addlogprob! (y_st * (Z_st + log(A_s + T(1e-6))) - A_s * exp(Z_st))
        end
    end
    """

    return (priors=priors, update=update)
end



# Version 1.3.11 (2026-08-05)
# Purpose: Generates Turing code fragments for the `TVCComponent`.
# Rationale: This version replaces `view()` with direct indexing to improve type stability for AD.
function _generate_component_code_fragments(m::TVCComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    inner_model = m.model
    cov_var = m.covariate
    
    inner_frags = _generate_component_code_fragments(inner_model, spec, arch, outcome_idx, M, prefix=prefix)
    priors_str = inner_frags.priors
    update_inner = inner_frags.update
    
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    # Remove any existing effect application from the inner fragment's update string.
    effect_app_regex = r"for i in 1:length\($(eta_target)\)\s*$(eta_target)\[i\] \+= .*end"
    update_inner_cleaned = replace(update_inner, effect_app_regex => "")
    
    application_code = """
    local cov_data::Vector{T} = T.(M.data[!, :$(cov_var)])
    for i in 1:length($(eta_target))
        $(eta_target)[i] += cov_data[i] * $(v.latent)[M.t_idx[i]]
    end
    """
    
    update_str = """
    begin
        # Temporally Varying Coefficient (TVC) for: $(cov_var)
        $(update_inner_cleaned)
        $(application_code)
    end
    """
    return (priors=priors_str, update=update_str)
end


 

# Version 1.5.6 (2026-08-06)
# Purpose: A specialized code generator for the `NonStationaryVariance` component.
# Rationale: This version is updated for AD compatibility. It replaces hard-coded
#            `T.(I(...))` in `MvNormal` priors and `cholesky` calls with `I`, which
#            correctly uses `UniformScaling` to allow for automatic type promotion
#            with `Dual` numbers, preventing `MethodError` during AD.
function _generate_component_code_fragments(m::NonStationaryVariance, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    key_str = string(spec.key)
    is_multivariate = arch == "multivariate"
    
    base_spec = spec.hyper.base_spec
    modifier_spec = spec.hyper.modifier_spec
    basis_key = spec.hyper.modifier_basis_key

    v_base = generate_full_variable_names(base_spec, arch, outcome_idx, prefix=key_str)
    v_modifier = generate_full_variable_names(modifier_spec, arch, outcome_idx, prefix=key_str)

    modifier_frags = _generate_component_code_fragments(m.modifier_model, modifier_spec, arch, outcome_idx, M, prefix=key_str)
    
    n_latent_base = size(base_spec.Q_template, 1)
    base_priors = "$(v_base.raw) ~ NamedDist(MvNormal(fill(zero(T), $(n_latent_base)), I), :$(v_base.raw))"
    
    priors_str = """
    # Priors for NonStationaryVariance component: $(key_str)
    $(modifier_frags.priors)
    $(base_priors)
    """

    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    effect_app_regex = r"for i in 1:length\($(eta_target)\)\s*$(eta_target)\[i\] \+= .*end"
    modifier_update_cleaned = replace(modifier_frags.update, effect_app_regex => "")

    update_str = """
    begin
        # --- Non-Stationary Variance Logic for $(key_str) ---
        
        $(modifier_update_cleaned)
        
        local log_sigma_field::Vector{T} = M.basis_matrices[:$(basis_key)] * $(v_modifier.latent)
        local spatially_varying_sigma::Vector{T} = exp.(log_sigma_field)
        
        local Q_base_template = spec_registry["$(spec.key)"].hyper.base_spec.Q_template
        local F_base = cholesky(Symmetric(Matrix(Q_base_template) + noise * I))
        local base_latent_raw::Vector{T} = F_base.L' \\ $(v_base.raw)
        
        if $(m.base_model isa Union{ICAR, Besag})
            Turing.@addlogprob! logpdf(Normal(T(0), T(0.001) * $(n_latent_base)), sum(base_latent_raw))
        end
        
        for i in 1:length($(eta_target))
            local s_idx_i = M.s_idx[i]
            local final_effect_i = base_latent_raw[s_idx_i] * spatially_varying_sigma[s_idx_i]
            $(eta_target)[i] += final_effect_i
        end
    end
    """
    
    return (priors=priors_str, update=update_str)
end

 

# Version 1.5.3 (2026-08-06)
# Purpose: Generates Turing code fragments for the `GP` component.
# Rationale: This version ensures type stability by explicitly casting `X_coords` to `T`
#            before use in `evaluate_kernel_matrix`.
function _generate_component_code_fragments(m::GP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    n_obs = size(spec.Q_template, 1) # For GP, Q_template holds the coordinates

    priors_acc = String[]
    push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
    
    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors_acc, "$(v.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(v.ls))")
    else
        push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
    end
    
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_obs)), I), :$(v.raw))")
    priors = join(priors_acc, "\n")

    update = """
    begin
        # Full Gaussian Process (GP) Logic for $(key_str)
        local X_coords = T.(spec_registry["$(key_str)"].Q_template)
        local kernel_type = Symbol("$(m.kernel)")

        local K_mat = evaluate_kernel_matrix(X_coords, $(v.sigma), $(v.ls), kernel_type, noise)
        local F_gp = cholesky(Symmetric(K_mat))
        $(v.latent) = F_gp.L * $(v.raw)
        $(arch == "multivariate" ? "eta_latent[:, $(outcome_idx)]" : "eta") .+= $(v.latent)
    end
    """
    return (priors=priors, update=update)
end



# Version 2.0.1 (2026-08-06)
# Purpose: Dispatches to the appropriate code generation function for a given model component.
# Rationale: This version is updated to use spectral decomposition for GMRF models like
#            BYM2, ICAR, Leroux, RW1, and RW2 when the `spectral_orientation` flag is true
#            and the necessary `U` and `L` matrices are available in the component specification.
#            This prioritizes AD-friendly sampling methods over the older MvNormalCanon approach.
# Assumptions: The main model configuration `M` contains a `spectral_orientation` flag.
function _generate_component_code_fragments(spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)
    # This function acts as a dispatcher, calling the correct code generator
    # based on the component's type and the model's configuration.

    m_obj = spec.component_obj
    use_spectral = get(M, :spectral_orientation, true) && hasproperty(spec.hyper, :U) && hasproperty(spec.hyper, :L)

    # --- Dispatch Logic ---
    if m_obj isa BYM2 && use_spectral
        return generate_bym2_assembly_spectral(spec, M, arch)
    elseif m_obj isa Union{ICAR, Besag} && use_spectral
        return generate_icar_assembly_spectral(spec, M, arch)
    elseif m_obj isa Leroux && use_spectral
        return generate_leroux_assembly_spectral(spec, M, arch)
    elseif m_obj isa Union{RW1, RW2} && use_spectral
        return generate_rw_assembly_spectral(spec, M, arch)
    elseif m_obj isa SciMLComponent
        return generate_sciml_component_assembly(spec, M, arch)
    else
        # Fallback to the existing, type-specific code generators for other components.
        return _generate_component_code_fragments(m_obj, spec, arch, outcome_idx, M)
    end
end
 


# Version 1.5.6 (2026-08-06)
# Purpose: Generates Turing code for an AR(2) process.
# Rationale: This version is updated for AD compatibility. It replaces the hard-coded
#            `T.(I(...))` in the `MvNormal` prior with `I`, which correctly uses
#            `UniformScaling` to allow for automatic type promotion with `Dual` numbers.
function _generate_component_code_fragments(m::AR2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx; prefix=prefix)
    
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        push!(priors_acc, "$(v.rho1) ~ NamedDist(truncated($(_distribution_to_string(m.rho1)), T(-2.0), T(2.0)), :$(v.rho1))")
        push!(priors_acc, "$(v.rho2) ~ NamedDist(truncated($(_distribution_to_string(m.rho2)), T(-1.0), T(1.0)), :$(v.rho2))")
    end
    push!(priors_acc, "$(v.innov) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.innov))")
    priors_str = join(priors_acc, "\n")

    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx"

    update_str = """
    begin
        # AR2 state-space evolution for $(spec.key)
        local latent_year::Vector{T} = ar2_statespace($(v.rho1), $(v.rho2), $(v.sigma), $(v.innov), T, $(n_latent), noise)
        for i in 1:length($(eta_target))
            $(eta_target)[i] += latent_year[M.$(index_var)[i]]
        end
    end
    """
    
    return (priors=priors_str, update=update_str)
end



# Version 1.5.3 (2026-08-06)
# Purpose: Generates code for the Householder reflection (spectral orientation) feature.
# Rationale: This version ensures type stability by explicitly casting `2.0` to `T`.
function _generate_householder_reflection_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    # Purpose: Generates code for the Householder reflection (spectral orientation) feature.
    # Rationale: This allows for rotating the latent space in multivariate models to better
    # v1.0.0 (2026-07-16)
    #            align signals, which can be useful for processes with directional dependencies.
    #            This is controlled by the `spectral_orientation=true` keyword argument.
    # Inputs:
    #   - M: The model configuration NamedTuple.
    #   - is_multivariate: A boolean indicating if the model is multivariate.
    #   - eta_name: The name of the latent predictor matrix (e.g., "eta_latent").
    # Outputs: A tuple of strings (priors_str, update_str).

    if !is_multivariate || !M.spectral_orientation
        return "", ""
    end

    K = M[:outcomes_N]
    
    priors_str = """
    # Householder reflection for spectral orientation
    v_raw_reflection ~ NamedDist(MvNormal(zeros(T, $(K)), I), :v_raw_reflection)
    """

    update_str = """
    begin
        v_reflection = v_raw_reflection / (norm(v_raw_reflection) + T(1e-9))
        H_reflection = I - T(2.0) * v_reflection * v_reflection'
        $(eta_name) = $(eta_name) * H_reflection
    end
    """
    return priors_str, update_str
end

 

function _generate_likelihood_section(M::NamedTuple, is_multivariate::Bool)
    families = [string(get(spec, :family, "gaussian")) for spec in M.likelihood_specs]
    needs_sigma = any(f -> f in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t", "dirichlet_multinomial"], families)

    zi_prior_block = ""
    hurdle_prior_block = ""
    if get(M, :user_provided_hurdle, false)
        hurdle_prior_block = "lik_phi_hurdle ~ NamedDist(Beta(1,1), :lik_phi_hurdle)"
    elseif get(M, :use_zi, false)
        zi_prior_block = "lik_phi_zi ~ NamedDist(Beta(1,1), :lik_phi_zi)"
    end

    nu_student_t_block = ""
    if any(f -> string(f) == "student_t", families)
        nu_student_t_block = "lik_nu_student_t ~ NamedDist(Exponential(1.0), :lik_nu_student_t)"
    end

    sigma_block = ""
    if needs_sigma
        sigma_y_prior_str = _distribution_to_string(Exponential(1.0))
        if is_multivariate
            sigma_block = "sigma_y ~ NamedDist(filldist($(sigma_y_prior_str), K), :sigma_y)"
        else
            sigma_block = "sigma_y ~ NamedDist($(sigma_y_prior_str), :sigma_y)"
        end
    end
    
    extra_params_block = ""
    if any(f -> string(f) in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"], families)
        extra_params_block = "lik_extra_params ~ NamedDist(Exponential(1.0), :lik_extra_params)"
    end

    corr_block = is_multivariate ? "L_corr ~ NamedDist(LKJCholesky(K, 1.0), :L_corr)" : ""

    ordinal_priors_block = ""
    ordinal_spec_idx = findfirst(s -> string(get(s, :family, "")) == "ordinal", M.likelihood_specs)
    if !isnothing(ordinal_spec_idx)
        K = M.likelihood_specs[ordinal_spec_idx][:K]
        latent_dist = M.likelihood_specs[ordinal_spec_idx][:latent_dist]

        if K > 1
            for j in 1:(K-1)
                # Use `\n` instead of `\\n` for correct line breaks
                ordinal_priors_block *= "ordinal_alpha_$(j) ~ NamedDist(Dirac(T(0.0)), :ordinal_alpha_$(j))\n"
            end
            if K > 2
                ordinal_priors_block *= """
                ordinal_alpha_raw_1 ~ NamedDist(Normal(0, 5), :ordinal_alpha_raw_1)
                ordinal_alpha_diffs ~ NamedDist(filldist(Exponential(1.0), $(K - 2)), :ordinal_alpha_diffs)
                """
            elseif K == 2
                # Use `\n` instead of `\\n`
                ordinal_priors_block *= "ordinal_alpha_raw_1 ~ NamedDist(Normal(0, 5), :ordinal_alpha_raw_1)\n"
            end

            if latent_dist == :student_t
                # Use `\n` instead of `\\n`
                ordinal_priors_block *= "ordinal_df ~ NamedDist(Exponential(1.0), :ordinal_df)\n"
            end
        end
    end

    return """
    $(corr_block)
    $(sigma_block)
    $(hurdle_prior_block)
    $(zi_prior_block)
    $(nu_student_t_block)
    $(extra_params_block)
    $(ordinal_priors_block)
    """
end



function _generate_final_likelihood_block(M::NamedTuple, is_multivariate::Bool)
    # v1.0.3 (2026-07-31)
    # Rationale: This version is updated to source the list of non-proportional effects
    #            from the main model configuration `M[:non_proportional_effects]`, aligning
    #            it with the new `fixed(..., non_proportional_effects=true)` syntax. This
    #            decouples the likelihood generator from the `likelihood()` module's parameters.
    if is_multivariate
        return _generate_multivariate_likelihood_block(M)
    end

    family = string(M.likelihood_specs[1][:family])

    if family == "ordinal"
        K = M.likelihood_specs[1][:K]
        latent_dist_val = M.likelihood_specs[1][:latent_dist]
        if K < 2; return ""; end

        non_prop_terms = get(M, :non_proportional_effects, Symbol[])
        is_npo = !isempty(non_prop_terms)
        
        npo_indices = findall(x -> x in non_prop_terms, M.Xfixed_names)
        n_npo_vars = length(npo_indices)

        assignment_lines = ""
        for j in 1:(K-1)
            assignment_lines *= "ordinal_alpha_$(j) = alphas_computed[$(j)]\n"
        end

        npo_update_block = ""
        if is_npo && n_npo_vars > 0
            npo_update_block = """
            # Non-proportional effects calculation
            local X_npo = M.Xfixed[:, $(npo_indices)]
            local beta_npo_matrix = reshape(beta_npo, $(n_npo_vars), $(K-1))
            local eta_npo = X_npo * beta_npo_matrix
            """
        end

        return """
        # Proportional Odds Likelihood
        let
            local alphas_computed
            if $(K > 2)
                alphas_computed = cumsum([ordinal_alpha_raw_1; ordinal_alpha_diffs])
            else
                alphas_computed = [ordinal_alpha_raw_1]
            end

            $(assignment_lines)
            local latent_dist_symbol = :$(latent_dist_val)
            $(npo_update_block)

            for i in 1:N
                linear_predictor_prop = eta[i]
                
                local linear_predictor_vec
                if $(is_npo && n_npo_vars > 0)
                    # Combine proportional and non-proportional parts for each cut-point
                    linear_predictor_vec = linear_predictor_prop .+ view(eta_npo, i, :)
                else
                    # If fully proportional, broadcast the single predictor
                    linear_predictor_vec = fill(linear_predictor_prop, $(K-1))
                end
                
                local cumulative_probs
                if latent_dist_symbol == :normal
                    cumulative_probs = Distributions.cdf.(Normal(), alphas_computed .- linear_predictor_vec)
                elseif latent_dist_symbol == :logistic
                    cumulative_probs = logistic.(alphas_computed .- linear_predictor_vec)
                elseif latent_dist_symbol == :student_t
                    cumulative_probs = Distributions.cdf.(TDist(ordinal_df), alphas_computed .- linear_predictor_vec)
                else
                    error("Unsupported latent distribution ':\$(latent_dist_symbol)' for ordinal model.")
                end
                
                probs = Vector{T}(undef, $(K))
                if $(K > 1)
                    probs[1] = cumulative_probs[1]
                    for j in 2:($(K-1))
                        probs[j] = max(0.0, cumulative_probs[j] - cumulative_probs[j-1])
                    end
                    probs[$(K)] = max(0.0, 1.0 - cumulative_probs[$(K-1)])
                else
                    probs[1] = 1.0
                end

                probs ./= (sum(probs) + 1e-9)
                Turing.@addlogprob! logpdf(Categorical(probs), M.y_obs[i])
            end
        end
        """
    else
        return _generate_univariate_likelihood_block(M)
    end
end


 

 

# Version 1.1.1 (2026-08-06)
# Purpose: Generates the code block for spatiotemporal interactions.
# Rationale: This version is updated to force the use of dense Cholesky factorization
#            by converting the sparse precision matrices to dense `Matrix` types before
#            the `cholesky` call. This avoids using `CHOLMOD`'s sparse factorization,
#            whose `FactorComponent` objects do not support the indexing required by
#            some generic linear algebra fallbacks, resolving the `CanonicalIndexError`.
function _generate_st_interaction_block(M::NamedTuple, s_spec, t_spec, is_multivariate::Bool, eta_name::String)
    if get(M, :model_st, "none") == "none" 
        return ""
    end

    if isnothing(s_spec) || isnothing(t_spec)
        @warn "Spatiotemporal interaction requested but marginal specifications are missing."
        return ""
    end

    s_key = string(s_spec.key)
    t_key = string(t_spec.key)
    
    s_chol_access = "cholesky(Symmetric(Matrix(spec_registry[\"$s_key\"].Q_template) + noise * I))"
    
    t_model_type = t_spec.component_obj |> typeof |> Symbol
    t_rho_var_name = "rho_$(t_key)"
    t_chol_access = "cholesky(Symmetric(Matrix(recompose_precision(:$(t_model_type), spec_registry[\"$t_key\"].Q_template, 1.0; extra_param=$(t_rho_var_name))) + noise * I))"

    K = get(M, :outcomes_N, 1)

    dummy_interaction_spec = (key = Symbol("st_interaction"), structure = :spacetime, var = "$(s_key)_$(t_key)", component_obj = NoneComponent(), params = Dict{Symbol, Any}(), Q_template = nothing, scaling_factor = 1.0, hyper = nothing)
    v_st_interaction = generate_full_variable_names(dummy_interaction_spec, "univariate", nothing)

    sigma_name = string(v_st_interaction.sigma)
    raw_name = string(v_st_interaction.raw)

    st_sigma_prior_dist_str = haskey(M, :st_interaction_sigma_prior) ? _distribution_to_string(M.st_interaction_sigma_prior) : "Exponential(1.0)"

    if is_multivariate
        interaction_code = """
        # --- Spatiotemporal Interaction Priors ---
        $(sigma_name) ~ NamedDist(filldist($(st_sigma_prior_dist_str), $K), :$(Symbol(sigma_name)))
        $(raw_name) ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N * $K), I), :$(Symbol(raw_name)))

        let
            C_s = $(s_chol_access)
            C_t = $(t_chol_access)
            
            Z_tensor = reshape($(raw_name), M.s_N, M.t_N, $K)
            
            for k in 1:$K
                Z_k = view(Z_tensor, :, :, k)
                
                tmp_spatial = C_s.L' \\ Z_k
                tmp_spatial_T = transpose(tmp_spatial)
                st_field_k_unscaled = transpose(C_t.L' \\ tmp_spatial_T)
                
                Turing.@addlogprob! logpdf(Normal(0, 0.001 * (M.s_N * M.t_N)), sum(st_field_k_unscaled))
                
                st_field_k = st_field_k_unscaled .* $(sigma_name)[k]

                for i in 1:N
                    $(eta_name)[i, k] += st_field_k[M.s_idx[i], M.t_idx[i]]
                end
            end
        end
        """
    else
        interaction_code = """
        # --- Spatiotemporal Interaction Priors ---
        $(sigma_name) ~ NamedDist($(st_sigma_prior_dist_str), :$(Symbol(sigma_name)))
        $(raw_name) ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :$(Symbol(raw_name)))

        let
            C_s = $(s_chol_access)
            C_t = $(t_chol_access)
            
            Z_matrix = reshape($(raw_name), M.s_N, M.t_N)
            
            tmp_spatial = C_s.L' \\ Z_matrix
            tmp_spatial_T = transpose(tmp_spatial)
            st_field_unscaled = transpose(C_t.L' \\ tmp_spatial_T)
            
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * (M.s_N * M.t_N)), sum(st_field_unscaled))
            
            st_field = st_field_unscaled .* $(sigma_name)

            for i in 1:N
                $(eta_name)[i] += st_field[M.s_idx[i], M.t_idx[i]]
            end
        end
        """
    end
    
    return interaction_code
end




# Version 1.5.0 (2026-08-06)
# Purpose: Generates the code block for adding the intercept to the linear predictor.
# Rationale: This version uses efficient broadcasting (`.+=`) instead of explicit loops,
#            resulting in cleaner and more performant generated code.
function _generate_intercept_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    # This function generates the prior and update code for the global intercept.
    if !get(M, :add_intercept, false) return "", "" end
    
    intercept_prior_obj = get(M, :intercept_prior, Normal(0,5))
    local dist_str, update_code, prior_code
    if is_multivariate
        dist_str = "filldist($(_distribution_to_string(intercept_prior_obj)), K)"
        # Use broadcasting with transpose to add the intercept vector to each row of eta_latent
        update_code = "$(eta_name) .+= intercept'"
    else
        dist_str = _distribution_to_string(intercept_prior_obj)
        # Use broadcasting to add the scalar intercept to the eta vector
        update_code = "$(eta_name) .+= intercept"
    end
    
    prior_code = "intercept ~ NamedDist($(dist_str), :intercept)"
    return prior_code, update_code
end

 

# Version 1.5.0 (2026-08-06)
# Purpose: Generates the code block for adding fixed effects to the linear predictor.
# Rationale: This version simplifies the generated code by removing `let` blocks and
#            replacing explicit loops with a single matrix-vector multiplication and
#            broadcasted addition (`.+=`), which is more efficient and readable.
function _generate_fixed_effects_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    # This function generates the priors and update code for fixed effects.
    if get(M, :Xfixed_N, 0) == 0; return "", ""; end

    priors_vec = get(M, :Xfixed_priors_vec, [Normal(0, 5) for _ in 1:M.Xfixed_N])
    
    is_ordinal = any(spec -> string(get(spec, :family, "")) == "ordinal", M.likelihood_specs)
    non_prop_terms = is_ordinal ? get(M, :non_proportional_effects, Symbol[]) : Symbol[]
    
    prop_indices = collect(1:M.Xfixed_N)
    npo_indices = Int[]

    if is_ordinal && !isempty(non_prop_terms)
        npo_indices = findall(x -> x in non_prop_terms, M.Xfixed_names)
        prop_indices = setdiff(prop_indices, npo_indices)
    end

    n_prop = length(prop_indices)
    n_npo = length(npo_indices)
    K_ordinal = is_ordinal ? get(M.likelihood_specs[1], :K, 0) : 0

    prior_parts = String[]
    update_parts = String[]

    if n_prop > 0
        priors_prop = priors_vec[prop_indices]
        all_same_prop = !isempty(priors_prop) && all(p -> p == priors_prop[1], priors_prop)
        
        if is_multivariate
            beta_prop_name = "Xfixed_beta_prop_flat"
            # Use matrix multiplication and broadcasting
            update_code = """
            beta_prop_matrix = reshape($(beta_prop_name), $(n_prop), M.outcomes_N)
            $(eta_name) .+= M.Xfixed[:, $(prop_indices)] * beta_prop_matrix
            """
            push!(update_parts, update_code)
            
            if all_same_prop
                prior_str = _distribution_to_string(priors_prop[1])
                push!(prior_parts, "$(beta_prop_name) ~ NamedDist(filldist($(prior_str), $(n_prop * M.outcomes_N)), :Xfixed_beta_prop)")
            else
                full_priors_list = vcat([priors_prop for _ in 1:M.outcomes_N]...)
                priors_str_list = [_distribution_to_string(p) for p in full_priors_list]
                push!(prior_parts, "$(beta_prop_name) ~ NamedDist(Product([$(join(priors_str_list, ", "))]), :Xfixed_beta_prop)")
            end
        else
            beta_prop_name = "Xfixed_beta_prop"
            # Use matrix-vector multiplication and broadcasting
            update_code = "$(eta_name) .+= M.Xfixed[:, $(prop_indices)] * $(beta_prop_name)"
            push!(update_parts, update_code)
            
            if all_same_prop
                prior_str = _distribution_to_string(priors_prop[1])
                push!(prior_parts, "$(beta_prop_name) ~ NamedDist(filldist($(prior_str), $(n_prop)), :Xfixed_beta_prop)")
            else
                priors_str_list = [_distribution_to_string(p) for p in priors_prop]
                push!(prior_parts, "$(beta_prop_name) ~ NamedDist(Product([$(join(priors_str_list, ", "))]), :Xfixed_beta_prop)")
            end
        end
    end

    if n_npo > 0 && K_ordinal > 1
        priors_npo = priors_vec[npo_indices]
        all_same_npo = !isempty(priors_npo) && all(p -> p == priors_npo[1], priors_npo)
        beta_npo_name = "beta_npo"
        n_npo_params = n_npo * (K_ordinal - 1)

        if all_same_npo
            prior_str = _distribution_to_string(priors_npo[1])
            push!(prior_parts, "$(beta_npo_name) ~ NamedDist(filldist($(prior_str), $(n_npo_params)), :beta_npo)")
        else
            full_priors_list = vcat([priors_npo for _ in 1:(K_ordinal-1)]...)
            priors_str_list = [_distribution_to_string(p) for p in full_priors_list]
            push!(prior_parts, "$(beta_npo_name) ~ NamedDist(Product([$(join(priors_str_list, ", "))]), :beta_npo)")
        end
    end

    prior_code = join(prior_parts, "\n    ")
    update_code = join(update_parts, "\n    ")
    
    return prior_code, update_code
end


# Version 1.5.8 (2026-08-06)
# Purpose: Generates Turing code for standard GMRF components.
# Rationale: This version is updated to be more AD-friendly. For dynamic components
#            (where the precision matrix depends on a sampled parameter like `rho`), it
#            now passes a type-stable `one` value to `recompose_precision` and uses
#            `UniformScaling` (`I`) in the `cholesky` call. This avoids hard-coded `Float64`
#            types that cause `MethodError` with `ForwardDiff.Dual` numbers.
function _generate_component_code_fragments(m::ComponentModel, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="", generate_eta_update::Bool=true)
    # This function generates the Turing model code for a generic GMRF component.
    # It defines priors, constructs the precision matrix, and samples the final effect.

    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    params = spec.params
    n_latent = size(spec.Q_template, 1)
    is_multivariate = (arch == "multivariate")
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))
    is_shared = get(params, :shared, false)

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        if hasproperty(m, :sigma); push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))"); end
        if hasproperty(m, :rho); push!(priors_acc, "$(v.rho) ~ NamedDist($(_distribution_to_string(m.rho)), :$(v.rho))"); end
    end
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.raw))")
    priors_str = join(priors_acc, "\n    ")

    index_var = if spec.structure == :spatial; "s_idx";
    elseif spec.structure == :temporal; (typeof(m) <: Union{Cyclic, Harmonic}) ? "u_idx" : "t_idx";
    elseif spec.structure == :mixed; "mixed_idx_$(spec.var)";
    else; index_var = string(spec.structure) * "_idx"; end

    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"

    effect_app_str = if generate_eta_update
        if spec.structure == :smooth
            "$(eta_target) .+= M.basis_matrices[:$(spec.var)] * $(v.latent)"
        else
            "$(eta_target) .+= view($(v.latent), M.$(index_var))"
        end
    else
        ""
    end

    update_str = if get(spec, :is_static, false)
        """
        # Static component: $(spec.key)
        F_static = spec_registry["$(spec.key)"].cholesky_factor
        unscaled_latent = F_static.L' \\ $(v.raw)
        $(v.latent) = $(v.sigma) .* unscaled_latent
        $(effect_app_str)
        """
    else
        flow_direction_kwarg = (m isa NetworkFlow) ? ", flow_direction=:$(m.flow_direction)" : ""
        """
        # Dynamic component: $(spec.key)
        let
            Q_template = spec_registry["$(spec.key)"].Q_template
            model_type = spec_registry["$(spec.key)"].component_obj |> typeof |> Symbol
            rho_value = $(hasproperty(m, :rho) ? v.rho : "nothing")
            
            # Pass a type-stable 1.0 to ensure promotion with Dual numbers from rho_value
            one_val = isnothing(rho_value) ? one(T) : one(typeof(rho_value))
            Q_final = recompose_precision(model_type, Q_template, one_val; extra_param=rho_value$(flow_direction_kwarg), noise=noise)
            
            # Use UniformScaling 'I' for AD-safety. Note: cholesky on a Dual matrix may still fail if it dispatches to LAPACK.
            F_dynamic = cholesky(Symmetric(Matrix(Q_final) + noise * I))
            
            unscaled_latent = F_dynamic.L' \\ $(v.raw)
            $(v.latent) = $(v.sigma) .* unscaled_latent
            $(effect_app_str)
        end
        """
    end

    return (priors=priors_str, update=update_str)
end


# Version 1.5.9 (2026-08-06)
# Purpose: Generates Turing code for an AR(1) process.
# Rationale: This version corrects the prior for the `rho` parameter. Instead of using a
#            conceptually incorrect `truncated(Beta, ...)` distribution, it now uses a `tanh`
#            transformation on an unconstrained `Normal` variable. This correctly maps the
#            parameter to the `(-1, 1)` range required for a stationary AR(1) process and
#            ensures stable gradient-based sampling.
function _generate_component_code_fragments(m::AR1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx; prefix=prefix)
    
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        # Use a tanh transform for rho to constrain it to (-1, 1)
        push!(priors_acc, "$(v.rho)_raw ~ NamedDist(Normal(0, 1.5), :$(Symbol(string(v.rho, "_raw"))))")
    end
    push!(priors_acc, "$(v.innov) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.innov))")
    priors_str = join(priors_acc, "\n")

    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx"

    update_str = """
    # AR1 state-space evolution for $(spec.key)
    $(v.rho) = tanh($(v.rho)_raw) # Transform to (-1, 1)
    latent_year = ar1_statespace($(v.rho), $(v.sigma), $(v.innov), $(n_latent), noise)
    $(eta_target) .+= view(latent_year, M.$(index_var))
    """
    
    return (priors=priors_str, update=update_str)
end




# Version 1.5.3 (2026-08-06)
# Purpose: Implements a state-space evolution for an AR(1) process.
# Rationale: This version is simplified by removing the explicit `T_model` argument and
#            replacing the final scaling loop with more efficient broadcasting.
function ar1_statespace(rho, sigma, innov, n_latent, noise)
    # This function computes the state-space evolution of an AR(1) process.
    # It is designed to be type-stable and work with different numeric types.
    
    T_num = promote_type(typeof(rho), typeof(sigma), eltype(innov), typeof(noise))
    latent = Vector{T_num}(undef, n_latent)
    
    if n_latent > 0
        # Initialize the first state using the stationary variance of the AR(1) process.
        latent[1] = innov[1] / sqrt(one(T_num) - rho^2 + T_num(noise))
        
        # Evolve the process for subsequent time steps.
        for t in 2:n_latent
            latent[t] = rho * latent[t-1] + innov[t]
        end
        
        # Scale the entire latent field by the standard deviation.
        latent .*= sigma
    end
    
    return latent
end




# Version 1.5.3 (2026-08-06)
# Purpose: Implements a stationary state-space evolution for an AR(2) process.
# Rationale: This version is updated to be explicitly AD-aware. All numeric literals
#            are promoted to the generic numeric type `T_num`, preventing type errors
#            when the function is called with `ForwardDiff.Dual` numbers.
function ar2_statespace(rho1, rho2, sigma, innov::AbstractVector, n_latent::Int, noise)
    T_num = promote_type(typeof(rho1), typeof(rho2), typeof(sigma), eltype(innov), typeof(noise))
    latent = Vector{T_num}(undef, n_latent)
    if n_latent == 0
        return latent
    end

    if rho1 + rho2 >= T_num(1.0) || rho2 - rho1 >= T_num(1.0) || abs(rho2) >= T_num(1.0)
        return fill(T_num(1e12), n_latent)
    end

    var_innov = sigma^2
    gamma_0 = var_innov * (T_num(1.0) - rho2) / ((T_num(1.0) + rho2) * ((T_num(1.0) - rho2)^2 - rho1^2) + T_num(noise))
    gamma_1 = (rho1 / (T_num(1.0) - rho2)) * gamma_0

    cov_12 = [gamma_0 gamma_1; gamma_1 gamma_0]
    L_12 = cholesky(Symmetric(cov_12 + T_num(noise) * I)).L

    if n_latent >= 2
        latent[1:2] = L_12 * innov[1:2]
    elseif n_latent == 1
        latent[1] = sqrt(gamma_0) * innov[1]
    end

    for t in 3:n_latent
        latent[t] = rho1 * latent[t-1] + rho2 * latent[t-2] + innov[t] * sigma
    end

    return latent
end
# Version 1.5.8 (2026-08-06)
# Purpose: Generates the main model string for the Turing model.
# Rationale: This version modifies the initialization of the linear predictor `eta`.
#            Instead of being hardcoded to `zeros(T, ...)` (where T is often Float64),
#            `eta` is now initialized with a type inferred from the `intercept` parameter.
#            During automatic differentiation, `intercept` becomes a `ForwardDiff.Dual` number,
#            so `eta` will be correctly allocated as a `Vector{Dual}`, preventing a `MethodError`
#            when `Dual`-valued parameters are added to it.
function bstm_text_assembler(M::NamedTuple, model_func_name::Symbol)
    # This function assembles the full Turing model string from various code fragments.
    # It orchestrates the inclusion of priors, the linear predictor assembly, and the
    # final likelihood evaluation.

    arch = get(M, :model_arch, "univariate")
    is_multivariate = arch == "multivariate"

    eta_name = is_multivariate ? "eta_latent" : "eta"
    # New `eta` initialization logic to ensure AD compatibility.
    # It infers the element type from the `intercept` parameter, which will be a
    # `ForwardDiff.Dual` during automatic differentiation.
    eta_init = if is_multivariate
        "zeros(typeof(intercept) <: AbstractArray ? eltype(intercept) : typeof(intercept), N, K)"
    else
        "zeros(typeof(intercept) <: AbstractArray ? eltype(intercept) : typeof(intercept), N)"
    end
    outcomes_N = get(M, :outcomes_N, 1)

    spec_registry = Dict{String, Any}()
    priors_acc = String[]
    updates_acc = String[]

    main_spatial_spec = nothing
    main_temporal_spec = nothing
    
    has_custom_likelihood_from_component = any(spec -> any(T -> spec.component_obj isa T, [LGCP, LogGammaCoxProcess, ShotNoiseCoxProcess]), M.components)
    has_custom_likelihood_from_family = any(spec -> string(get(spec, :family, "")) == "ordinal", M.likelihood_specs)
    has_custom_likelihood = has_custom_likelihood_from_component || has_custom_likelihood_from_family

    # Generate all code fragments first
    intercept_priors, intercept_update = _generate_intercept_block(M, is_multivariate, eta_name)
    if !isempty(intercept_priors); push!(priors_acc, intercept_priors); end
    
    offset_block = _generate_offset_block(M, is_multivariate, eta_name)
    
    fixed_effects_priors, fixed_effects_update = _generate_fixed_effects_block(M, is_multivariate, eta_name)
    if !isempty(fixed_effects_priors); push!(priors_acc, fixed_effects_priors); end
    
    # The update blocks must be ordered correctly: intercept, then others.
    push!(updates_acc, intercept_update)
    push!(updates_acc, offset_block)
    push!(updates_acc, fixed_effects_update)


    if get(M, :is_multivariate_dynamics, false)
        mv_dyn_key = M[:multivariate_dynamics_key]
        spec_idx = findfirst(s -> string(s.key) == mv_dyn_key, M.components)
        if !isnothing(spec_idx)
            spec = M.components[spec_idx]
            spec_registry[string(spec.key)] = spec
            frags = _generate_component_code_fragments(spec.component_obj, spec, arch, nothing, M)
            push!(priors_acc, frags.priors)
            push!(updates_acc, frags.update)
        end
    end

    for spec in M.components
        if get(M, :is_multivariate_dynamics, false) && string(spec.key) == M[:multivariate_dynamics_key]
            continue
        end
        spec_registry[string(spec.key)] = spec
        for k in 1:outcomes_N
            outcome_idx = is_multivariate ? k : nothing            
            frag = _generate_component_code_fragments(spec.component_obj, spec, arch, outcome_idx, M)
            if !isempty(Base.strip(frag.priors)); push!(priors_acc, frag.priors); end
            if !isempty(Base.strip(frag.update)); push!(updates_acc, frag.update); end
        end

        if spec.structure == :spatial && isnothing(main_spatial_spec); main_spatial_spec = spec; end
        if spec.structure == :temporal && isnothing(main_temporal_spec); main_temporal_spec = spec; end
    end

    function _indent_block(text::String, level=1)
        if isempty(Base.strip(text)) return "" end
        indent_str = "    " ^ level
        return indent_str * replace(Base.strip(text), "\n" => "\n" * indent_str)
    end

    likelihood_section_priors = _generate_likelihood_section(M, is_multivariate)
    st_interaction_block = _generate_st_interaction_block(M, main_spatial_spec, main_temporal_spec, is_multivariate, eta_name)
    householder_priors, householder_update = _generate_householder_reflection_block(M, is_multivariate, eta_name)
    
    final_likelihood = if has_custom_likelihood
        if has_custom_likelihood_from_family; _generate_final_likelihood_block(M, is_multivariate); else ""; end
    else
        _generate_final_likelihood_block(M, is_multivariate)
    end
    
    nested_priors, nested_updates, nested_likelihoods = _generate_nested_model_block(M, is_multivariate, eta_name)
    
    # Add all remaining priors to the accumulator
    if !isempty(Base.strip(likelihood_section_priors)); push!(priors_acc, likelihood_section_priors); end
    if !isempty(Base.strip(householder_priors)); push!(priors_acc, householder_priors); end
    if !isempty(Base.strip(nested_priors)); push!(priors_acc, nested_priors); end

    priors_code = join(priors_acc, "\n\n")
    updates_code = join(updates_acc, "\n\n")

    model_string = """
@model function $(model_func_name)(M, spec_registry; T::Type=Float64)
    noise = T(M.noise)
    N = M.y_N
    K = $(outcomes_N)

    # --- Priors & Hyperparameters ---
$(_indent_block(priors_code))

    # --- Linear Predictor ---
    # Initialize eta with a type that matches the intercept to support AD.
    $(eta_name) = $(eta_init)

$(_indent_block(updates_code))
$(_indent_block(householder_update))
$(_indent_block(nested_updates))
$(_indent_block(st_interaction_block))

    # --- Likelihood ---
$(_indent_block(final_likelihood))
$(_indent_block(nested_likelihoods))
end
"""
 
    model_string_to_parse = model_string
    
    try
        return model_string_to_parse, Meta.parse(model_string_to_parse), spec_registry
    catch e
        println("BSTM Assembler Error: Failed to parse the generated model string.")
        println(model_string_to_parse)
        rethrow(e)
    end
end

# Version 1.5.8 (2026-08-06)
# Purpose: Generates the code block for adding log-offsets to the linear predictor.
# Rationale: This version removes the explicit cast `T.(...)` from the log_offsets.
#            This prevents a `MethodError` during automatic differentiation when `eta` is a
#            `Vector{ForwardDiff.Dual}` and the code attempts to cast a `Dual` number
#            back to `Float64`. Standard promotion rules for `Dual + Float64` are sufficient.
function _generate_offset_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    # This function generates the code to add log-offsets to the linear predictor.
    if !haskey(M, :log_offsets) || all(iszero, M[:log_offsets])
        return ""
    end
    
    if is_multivariate
        # Broadcast the matrix of offsets to the eta_latent matrix
        return "$(eta_name) .+= M.log_offsets"
    else
        # Broadcast the vector of offsets to the eta vector
        return "$(eta_name) .+= M.log_offsets[:, 1]"
    end
end


# Version 1.5.5 (2026-08-06)
# Purpose: Generates Turing code fragments for the `Eigen` (Bayesian PCA) component.
# Rationale: This version removes the explicit cast `T.(...)` from the observed data
#            `Y_eigen_data` inside the `logpdf` call. This is a critical fix to prevent a
#            `MethodError` during automatic differentiation (e.g., with NUTS), which occurs
#            if `Y_eigen_data` is promoted to a `ForwardDiff.Dual` number and the code
#            attempts to cast it back to `Float64`.
function _generate_component_code_fragments(m::Eigen, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    
    n_vars = m.n_vars
    n_factors = m.n_factors
    n_obs = size(spec.hyper.eigen_data, 1)

    pca_sd_prior_str = _distribution_to_string(m.pca_sd)
    pdef_sd_prior_str = _distribution_to_string(m.pdef_sd)

    priors_str = """
    # Priors for eigen component: $(key_str)
    $(v.v_raw) ~ NamedDist(MvNormal(zeros(T, $(length(m.ltri_indices))), T(1.0)), :$(v.v_raw))
    $(v.pca_sd) ~ NamedDist(filldist($(pca_sd_prior_str), $(n_factors)), :$(v.pca_sd))
    $(v.pdef_sd) ~ NamedDist(filldist($(pdef_sd_prior_str), $(n_vars)), :$(v.pdef_sd))
    $(v.factors_flat) ~ NamedDist(MvNormal(zeros(T, $(n_obs * n_factors)), I), :$(v.factors_flat))
    """

    update_str = """
    begin
        # --- Factor Model for Eigen Component: $(key_str) ---
        local v_mat = zeros(T, $(n_vars), $(n_factors)); v_mat[$(m.ltri_indices)] .= $(v.v_raw)
        local U = householder_to_eigenvector(v_mat, $(n_vars), $(n_factors))
        local L = U * Diagonal($(v.pca_sd))
        local F = reshape($(v.factors_flat), $(n_obs), $(n_factors))
        local Y_hat = F * L'
        local Psi = Diagonal($(v.pdef_sd).^2) + (T(noise) * I)
        
        local Y_eigen_data = spec_registry["$(key_str)"].hyper.eigen_data
        for i in 1:$(n_obs)
            # Do NOT cast Y_eigen_data to T, as this breaks AD.
            Turing.@addlogprob! logpdf(MvNormal(Y_hat[i, :], Psi), Y_eigen_data[i, :])
        end
        eta .+= sum(F, dims=2)
    end
    """
    return (priors=priors_str, update=update_str)
end

# Version 1.5.5 (2026-08-06)
# Purpose: Generates Turing code fragments for the `LogGammaCoxProcess` component.
# Rationale: This version removes the explicit cast `T(...)` from the observed data `y_st`
#            inside the `logpdf` call. This prevents a `MethodError` during automatic
#            differentiation when `y_st` is promoted to a `Dual` number.
function _generate_component_code_fragments(m::LogGammaCoxProcess, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)

    is_spatiotemporal = hasproperty(spec.hyper, :temporal_spec)
    n_latent_dims = is_spatiotemporal ? "M.s_N * M.t_N" : "M.s_N"

    # Priors for the Gamma shape and the raw innovations for the latent field
    priors = """
    # Log-Gamma Cox Process Priors
    $(v.innov) ~ NamedDist($(_distribution_to_string(m.shape)), :$(v.innov)) # Using 'innov' for the shape parameter
    $(v.raw) ~ NamedDist(MvNormal(fill(zero(T), $(n_latent_dims)), I), :$(v.raw))
    """

    update = """
    begin
        # Log-Gamma Cox Process Model: $(key_str)
        local latent_field_st = zeros(T, M.s_N, M.t_N)
        
        # 1. Reconstruct the latent spatiotemporal field Z(s,t)
        if $(is_spatiotemporal)
            local s_spec = spec_registry["$(key_str)"].hyper.inner_spec
            local t_spec = spec_registry["$(key_str)"].hyper.temporal_spec
            local C_s = cholesky(Symmetric(Matrix(s_spec.Q_template) + noise * I))
            local C_t = cholesky(Symmetric(Matrix(t_spec.Q_template) + noise * I))
            local Z_matrix = reshape($(v.raw), M.s_N, M.t_N)
            local tmp_spatial = C_s.U \\ Z_matrix
            latent_field_st = exp.(transpose(C_t.U \\ transpose(tmp_spatial))) # Exponentiate to ensure positivity
        else
            local Q_inner = spec_registry["$(key_str)"].hyper.inner_spec.Q_template
            local F_inner = cholesky(Symmetric(Matrix(Q_inner) + noise * I))
            local spatial_component = exp.(F_inner.U \\ $(v.raw)) # Exponentiate
            latent_field_st = repeat(spatial_component, 1, M.t_N)
        end

        # 2. Assemble the full mean intensity surface.
        local mean_intensity_surface = zeros(T, M.s_N, M.t_N)
        local gamma_shape = $(v.innov) # The learned shape parameter

        for t in 1:M.t_N, s in 1:M.s_N
            obs_indices = findall(i -> M.s_idx[i] == s && M.t_idx[i] == t, 1:N)
            base_contribution = isempty(obs_indices) ? zero(T) : mean(view(eta, obs_indices))
            mean_intensity_surface[s, t] = exp(base_contribution) * latent_field_st[s, t]
        end

        # 3. Point Process Likelihood Evaluation using Negative Binomial
        local grid_areas = spec_registry["$(key_str)"].hyper.areas
        for t in 1:M.t_N, s in 1:M.s_N
            local y_st = M.y_obs[s, t]
            local A_s = grid_areas[s]
            local mu = mean_intensity_surface[s, t] * A_s
            
            local r_nb = gamma_shape
            local p_nb = r_nb / (r_nb + mu)
            local nb_dist = NegativeBinomial(r_nb, p_nb)
            # Do NOT cast y_st to T, as this breaks AD.
            Turing.@addlogprob! logpdf(nb_dist, y_st)
        end

        M[:likelihood_handled] = true
    end
    """

    return (priors=priors, update=update)
end

# Version 1.5.5 (2026-08-06)
# Purpose: Generates Turing code fragments for the `ShotNoiseCoxProcess` component.
# Rationale: This version removes the explicit cast `T(...)` from the observed data `y_s`
#            inside the `logpdf` calculation. This prevents a `MethodError` during automatic
#            differentiation when `y_s` is promoted to a `Dual` number.
function _generate_component_code_fragments(m::ShotNoiseCoxProcess, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)

    n_parents_str = if m.n_parents isa Int
        string(m.n_parents)
    else
        "$(v.innov)_n_parents"
    end

    priors_list = String[]
    if m.n_parents isa UnivariateDistribution
        push!(priors_list, "$(n_parents_str) ~ NamedDist($(_distribution_to_string(m.n_parents)), :$(Symbol(n_parents_str)))")
    end
    
    bounds = spec.hyper.domain_bounds
    push!(priors_list, "$(v.raw)_parent_locs_x ~ NamedDist(filldist(Uniform(T($(bounds.x_min)), T($(bounds.x_max))), $(n_parents_str)), :$(Symbol(string(v.raw, "_parent_locs_x"))))")
    push!(priors_list, "$(v.raw)_parent_locs_y ~ NamedDist(filldist(Uniform(T($(bounds.y_min)), T($(bounds.y_max))), $(n_parents_str)), :$(Symbol(string(v.raw, "_parent_locs_y"))))")
    push!(priors_list, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
    push!(priors_list, "$(v.amplitude) ~ NamedDist(filldist($(_distribution_to_string(m.amplitude)), $(n_parents_str)), :$(Symbol(string(v.amplitude))))")

    priors = join(priors_list, "\n    ")

    update = """
    begin
        # Shot-Noise Cox Process Model: $(key_str)
        local obs_locs = M.centroids 
        local parent_locs = hcat($(v.raw)_parent_locs_x, $(v.raw)_parent_locs_y)
        
        local intensity_at_obs = fill(zero(T), M.s_N)
        for i in 1:M.s_N
            local intensity_i = zero(T)
            for j in 1:$(n_parents_str)
                local dist_sq = (obs_locs[i].x - parent_locs[j, 1])^2 + (obs_locs[i].y - parent_locs[j, 2])^2
                local kernel_val = exp(T(-0.5) * dist_sq / ($(v.ls)^2))
                intensity_i += $(v.amplitude)[j] * kernel_val
            end
            intensity_at_obs[i] = intensity_i
        end

        local grid_areas = spec_registry["$(key_str)"].hyper.areas
        for s in 1:M.s_N
            local y_s = M.y_obs[s] 
            local A_s = grid_areas[s]
            local lambda_s = intensity_at_obs[s] * A_s
            
            # Do NOT cast y_s to T, as this breaks AD.
            Turing.@addlogprob! (y_s * log(lambda_s + T(1e-6)) - lambda_s)
        end

        M[:likelihood_handled] = true
    end
    """
    
    return (priors=priors, update=update)
end

# Version 1.5.5 (2026-08-06)
# Purpose: Generates the full code block for all nested sub-models.
# Rationale: This version removes the explicit cast `T(...)` from the observed data
#            `sub_M.y_obs[i]` and `sub_M.log_offsets` inside the model. This prevents a
#            `MethodError` during automatic differentiation when these data-derived values
#            are involved in operations with `Dual` numbers.
function _generate_nested_model_block(M::NamedTuple, is_multivariate::Bool, main_eta_name::String)
    if !haskey(M, :nested_components) || isempty(M.nested_components)
        return "", "", ""
    end

    all_nested_priors = String[]
    all_nested_updates = String[]
    all_nested_likelihoods = String[]

    for (var_key, sub_config) in pairs(M.nested_components)
        prefix = string(var_key)
        sub_eta_name = "eta_$(prefix)"

        # --- 1. Generate Priors for Sub-Model ---
        if get(sub_config, :add_intercept, false)
            prior_obj = get(sub_config, :intercept_prior, Normal(T(0),T(5)))
            push!(all_nested_priors, "$(prefix)_intercept ~ NamedDist($(_distribution_to_string(prior_obj)), :$(Symbol(prefix, "_intercept")))")
        end
        
        for spec in sub_config.components
            frag = _generate_component_code_fragments(spec.component_obj, spec, "univariate", nothing, sub_config; prefix=prefix)
            push!(all_nested_priors, frag.priors)
        end
        
        sub_lik_spec = sub_config.likelihood_specs[1]
        sub_family_str = string(get(sub_lik_spec, :family, "gaussian"))
        if sub_family_str in ["gaussian", "lognormal", "student_t"]
            push!(all_nested_priors, "$(prefix)_y_sigma ~ NamedDist(Exponential(T(1.0)), :$(Symbol(prefix, "_y_sigma")))")
        end

        # --- 2. Generate Update Block for Sub-Model's Linear Predictor ---
        intercept_block = if get(sub_config, :add_intercept, false)
            """
            for i in 1:sub_M.y_N
                $(sub_eta_name)[i] += $(prefix)_intercept
            end
            """
        else
            ""
        end

        offset_block = if haskey(sub_config, :log_offsets) && !all(iszero, sub_config.log_offsets)
            """
            for i in 1:sub_M.y_N
                $(sub_eta_name)[i] += sub_M.log_offsets[i, 1]
            end
            """
        else
            ""
        end

        sub_eta_initialization_code = """
        $(sub_eta_name) = Vector{T}(undef, sub_M.y_N)
        for n_idx in 1:sub_M.y_N
            $(sub_eta_name)[n_idx] = zero(T)
        end
        """

        update_block = """
        # --- Nested Model: $(prefix) ---
        local $(sub_eta_name)
        let sub_M = M.nested_components[:$(var_key)]
            # Initialize the linear predictor for the sub-model
            \$(sub_eta_initialization_code)
            \$(intercept_block)
            \$(offset_block)
        """
        
        for spec in sub_config.components
            frag = _generate_component_code_fragments(spec.component_obj, spec, "univariate", nothing, sub_config; prefix=prefix)
            update_block *= "\n" * frag.update
        end
        update_block *= "\n        end" # End of the `let` block
        push!(all_nested_updates, update_block)

        # --- 3. Generate Linking Code ---
        rho_name = "rho_nested_$(prefix)"
        push!(all_nested_priors, "$(rho_name) ~ NamedDist(Normal(T(1.0), T(0.5)), :$(rho_name))")
        push!(all_nested_updates, "$(main_eta_name) .+= $(rho_name) .* $(sub_eta_name)")

        # --- 4. Generate Likelihood for Sub-Model ---
        kwargs_parts = String[]
        if sub_family_str in ["gaussian", "lognormal", "student_t"]; push!(kwargs_parts, "sigma_y=$(prefix)_y_sigma"); end
        kwargs_str = join(kwargs_parts, ", ")

        lik_loop = """
        # Likelihood for nested model: $(prefix)
        let sub_M = M.nested_components[:$(var_key)]
            sub_family_str = string(sub_M.likelihood_specs[1][:family])
            for i in 1:sub_M.y_N
                local d_lik_sub = bstm_Likelihood(sub_family_str, $(sub_eta_name)[i]; $(kwargs_str))
                # Do NOT cast sub_M.y_obs to T, as this breaks AD.
                Turing.@addlogprob! Distributions.logpdf(d_lik_sub, sub_M.y_obs[i])
            end
        end
        """
        push!(all_nested_likelihoods, lik_loop)
    end

    return join(all_nested_priors, "\n\n"), join(all_nested_updates, "\n\n"), join(all_nested_likelihoods, "\n\n")
end

# Version 1.5.5 (2026-08-06)
# Purpose: Generates the final likelihood block for univariate models.
# Rationale: This version removes the explicit cast of the observed data `M.y_obs[i]`
#            to the model's float type `T` inside the `logpdf` call. This is a critical
#            fix to prevent a `MethodError` during automatic differentiation (e.g., with NUTS),
#            which occurs if `M.y_obs` is promoted to a `ForwardDiff.Dual` number and the
#            code attempts to cast it back to `Float64`. The `logpdf` function is robust
#            enough to handle a `Dual`-parameterized distribution evaluated at `Float64` data.
function _generate_univariate_likelihood_block(M::NamedTuple)
    family = string(M.likelihood_specs[1][:family])
    any_needs_sigma = family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
    any_needs_nu = family == "student_t"
    any_needs_extra = family in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"]

    kwargs_parts = String[]
    if any_needs_sigma; push!(kwargs_parts, "sigma_y=sigma_y"); end
    if get(M, :user_provided_trials, false); push!(kwargs_parts, "trial=Int(M.trials[i, 1])"); end
    if get(M, :user_provided_weights, false); push!(kwargs_parts, "weight=T(M.weights[i, 1])"); end
    if get(M, :user_provided_censor_lower, false); push!(kwargs_parts, "censor_lower=T(M.censor_lower[i, 1])"); end
    if get(M, :user_provided_censor_upper, false); push!(kwargs_parts, "censor_upper=T(M.censor_upper[i, 1])"); end
    if get(M, :user_provided_hurdle, false); push!(kwargs_parts, "hurdle=T(M.hurdle[i, 1])"); end
    if get(M, :user_provided_hurdle, false); push!(kwargs_parts, "phi_hurdle=lik_phi_hurdle");
    elseif get(M, :use_zi, false); push!(kwargs_parts, "phi_zi=lik_phi_zi"); end

    extra_param_logic = if any_needs_nu && any_needs_extra
        "local extra_p = family == \"student_t\" ? lik_nu_student_t : lik_extra_params"
    elseif any_needs_nu
        "local extra_p = lik_nu_student_t"
    elseif any_needs_extra
        "local extra_p = lik_extra_params"
    else "" end
    if !isempty(extra_param_logic); push!(kwargs_parts, "extra_params=extra_p"); end

    kwargs_str = join(kwargs_parts, ", ")

    return """
    family = M.likelihood_specs[1][:family]
    $(extra_param_logic)
    for i in 1:N
        # Construct distribution parameterized by eta[i]
        local d_lik = bstm_Likelihood(family, eta[i]; $(kwargs_str))
        # Evaluate logpdf at the observed data M.y_obs[i].
        # Do NOT cast M.y_obs[i] to T, as this breaks AD if y_obs is promoted to a Dual.
        Turing.@addlogprob! Distributions.logpdf(d_lik, M.y_obs[i])
    end
    """
end

# Version 1.5.5 (2026-08-06)
# Purpose: Generates the final likelihood block for multivariate models.
# Rationale: This version removes the explicit cast of the observed data `M.y_obs`
#            to the model's float type `T` inside the `logpdf` call. This is a critical
#            fix to prevent a `MethodError` during automatic differentiation (e.g., with NUTS),
#            which occurs if `M.y_obs` is promoted to a `ForwardDiff.Dual` number and the
#            code attempts to cast it back to `Float64`. The `logpdf` function is robust
#            enough to handle a `Dual`-parameterized distribution evaluated at `Float64` data.
function _generate_multivariate_likelihood_block(M::NamedTuple)
    families = [string(get(spec, :family, "gaussian")) for spec in M.likelihood_specs]
    any_needs_sigma = any(f -> f in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t", "dirichlet_multinomial"], families)
    any_needs_nu = any(f -> f == "student_t", families)
    any_needs_extra = any(f -> f in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"], families)
    is_multinomial = any(f -> f == "dirichlet_multinomial", families)

    if is_multinomial
        return """
        local eta_correlated = eta_latent * L_corr.L
        local family = M.likelihood_specs[1][:family]
        for i in 1:N
            # Construct distribution parameterized by eta_correlated
            local d_lik = bstm_Likelihood(family, eta_correlated[i, :]; trial=Int(sum(M.y_obs[i, :])) )
            # Evaluate logpdf at the observed data vector M.y_obs[i, :].
            # Do NOT cast M.y_obs to T, as this breaks AD.
            Turing.@addlogprob! Distributions.logpdf(d_lik, M.y_obs[i, :])
        end
        """
    end

    kwargs_parts = String[]
    if any_needs_sigma; push!(kwargs_parts, "sigma_y=sigma_y[k]"); end
    if get(M, :user_provided_trials, false); push!(kwargs_parts, "trial=Int(M.trials[i, k])"); end
    if get(M, :user_provided_weights, false); push!(kwargs_parts, "weight=T(M.weights[i, k])"); end
    if get(M, :user_provided_censor_lower, false); push!(kwargs_parts, "censor_lower=T(M.censor_lower[i, k])"); end
    if get(M, :user_provided_censor_upper, false); push!(kwargs_parts, "censor_upper=T(M.censor_upper[i, k])"); end
    if get(M, :user_provided_hurdle, false); push!(kwargs_parts, "hurdle=T(M.hurdle[i, k])"); end
    if get(M, :user_provided_hurdle, false); push!(kwargs_parts, "phi_hurdle=lik_phi_hurdle");
    elseif get(M, :use_zi, false); push!(kwargs_parts, "phi_zi=lik_phi_zi"); end

    extra_param_logic = if any_needs_nu && any_needs_extra
        "local extra_p = family_k == \"student_t\" ? lik_nu_student_t : lik_extra_params"
    elseif any_needs_nu
        "local extra_p = lik_nu_student_t"
    elseif any_needs_extra
        "local extra_p = lik_extra_params"
    else "" end
    if !isempty(extra_param_logic); push!(kwargs_parts, "extra_params=extra_p"); end

    kwargs_str = join(kwargs_parts, ", ")

    return """
    local eta_correlated = eta_latent * L_corr.L
    for k in 1:K
        local family_k = M.likelihood_specs[k][:family]
        $(extra_param_logic)
        for i in 1:N
            # Construct distribution parameterized by eta_correlated
            local d_lik = bstm_Likelihood(family_k, eta_correlated[i, k]; $(kwargs_str))
            # Evaluate logpdf at the observed data M.y_obs[i, k].
            # Do NOT cast M.y_obs to T, as this breaks AD.
            Turing.@addlogprob! Distributions.logpdf(d_lik, M.y_obs[i, k])
        end
    end
    """
end

# Version 1.5.9 (2026-08-06)
# Purpose: Generates Turing code for static GMRF components like ICAR and Besag.
# Rationale: This function demonstrates the correct, AD-unsafe but efficient, way to use
#            sparse Cholesky factorization for static components. It solves the `CanonicalIndexError`
#            by explicitly creating a `SparseMatrixCSC` from the Cholesky factor `F.L` before
#            performing the solve. This ensures the backslash operator dispatches to a method
#            that supports sparse triangular solves. This pattern should be used for all
#            pre-computed static components. For dynamic components, spectral decomposition remains
#            the only AD-safe and efficient method.
function _generate_component_code_fragments(m::Union{ICAR, Besag}, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="", generate_eta_update::Bool=true)
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    params = spec.params
    n_latent = size(spec.Q_template, 1)
    is_multivariate = (arch == "multivariate")
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))
    is_shared = get(params, :shared, false)

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
    end
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.raw))")
    priors_str = join(priors_acc, "\n    ")

    index_var = (spec.structure == :spatial) ? "s_idx" : string(spec.structure) * "_idx"
    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    effect_app_str = if generate_eta_update
        if spec.structure == :smooth
            "$(eta_target) .+= M.basis_matrices[:$(spec.var)] * $(v.latent)"
        else
            "$(eta_target) .+= view($(v.latent), M.$(index_var))"
        end
    else
        ""
    end

    # This component is static, so we use the pre-computed Cholesky factor.
    update_str = """
    begin
        # Static ICAR/Besag Component: $(spec.key)
        local F_static = spec_registry["$(spec.key)"].cholesky_factor
        
        # Explicitly create a sparse matrix from the factor component to ensure correct dispatch.
        local L_sparse = sparse(F_static.L)
        local unscaled_latent = L_sparse' \\ $(v.raw)
        
        $(v.latent) = $(v.sigma) .* unscaled_latent
        
        # Apply sum-to-zero constraint for identifiability
        Turing.@addlogprob! logpdf(Normal(T(0), T(0.001) * $(n_latent)), sum($(v.latent)))
        
        $(effect_app_str)
    end
    """

    return (priors=priors_str, update=update_str)
end


# Version 1.0.0 (2026-08-06)
# Purpose: Generates the Turing model code for a BYM2 component using spectral decomposition.
# Rationale: This function implements the "spectral trick" for sampling the latent BYM2 field.
#            Instead of constructing a parameter-dependent precision matrix and using MvNormalCanon
#            (which is not AD-friendly), it samples standard normal noise and transforms it using
#            the pre-computed eigenvectors (U) and a diagonal matrix (D) constructed from the
#            eigenvalues (L) and the sampled hyperparameters (sigma, rho). This approach is
#            fully differentiable and avoids LAPACK errors with ForwardDiff.jl.
# Assumptions: The component specification `spec` contains pre-computed `U` (eigenvectors) and
#              `L` (eigenvalues) from `build_structure_template`.
function generate_bym2_assembly_spectral(spec::NamedTuple, M::NamedTuple, arch::String)
    key = spec.key
    p_names = generate_full_variable_names(spec, arch, get(spec.params, :outcome_idx, nothing))

    # --- 1. Generate Priors for Hyperparameters ---
    priors_str = """
    # --- Priors for BYM2 component: $(key) ---
    $(p_names.sigma) ~ $(_distribution_to_string(spec.component_obj.sigma))
    $(p_names.rho) ~ $(_distribution_to_string(spec.component_obj.rho))
    """

    # --- 2. Generate Spectral Sampling Code for the Latent Field ---
    # This is the core of the non-centered parameterization using spectral decomposition.
    assembly_lines = [
        "# --- BYM2 spectral assembly: $(key) ---",
        "# Sample standard normal noise for the structured and unstructured components.",
        "$(p_names.struct) ~ MvNormal(zeros(T, $(spec.hyper.n_latent)), I)",
        "$(p_names.iid) ~ MvNormal(zeros(T, $(spec.hyper.n_latent)), I)",
        "",
        "# Construct the diagonal of the spectral transformation matrix D.",
        "# D = diag(1 / sqrt( (1-rho) + rho*L_j ) ) where L_j are eigenvalues of Q_template.",
        "# This avoids inverting or decomposing a matrix inside the model.",
        "local diag_D_structured = sqrt.(1.0 ./ ((1.0 .- $(p_names.rho)) .+ $(p_names.rho) .* spec.L .+ 1e-9))",
        "",
        "# Apply the spectral transformation: latent = U * D * z",
        "# This constructs the structured spatial effect in an AD-friendly way.",
        "local structured_effect = spec.U * (diag_D_structured .* $(p_names.struct))",
        "",
        "# Combine structured and unstructured components using the BYM2 parameterization.",
        "local bym2_field = $(p_names.sigma) .* (sqrt($(p_names.rho)) .* structured_effect .+ sqrt(1.0 - $(p_names.rho)) .* $(p_names.iid))",
        "",
        "# Add the final effect to the linear predictor, indexed by the spatial units.",
        "eta .+= bym2_field[M.s_idx]"
    ]
    assembly_str = join(assembly_lines, "\n    ")

    return (priors=priors_str, assembly=assembly_str, post_assembly="")
end




# Version 1.0.0 (2026-08-06)
# Purpose: Generates the Turing model code for an ICAR/Besag component using spectral decomposition.
# Rationale: Implements the "spectral trick" for sampling the latent ICAR field.
#            This approach is fully differentiable and avoids Cholesky decomposition of
#            parameter-dependent matrices, which is incompatible with ForwardDiff.jl.
# Assumptions: The component specification `spec` contains pre-computed `U` (eigenvectors) and
#              `L` (eigenvalues) from `build_structure_template`.
function generate_icar_assembly_spectral(spec::NamedTuple, M::NamedTuple, arch::String)
    key = spec.key
    p_names = generate_full_variable_names(spec, arch, get(spec.params, :outcome_idx, nothing))

    # --- 1. Generate Priors for Hyperparameters ---
    priors_str = """
    # --- Priors for ICAR component: $(key) ---
    $(p_names.sigma) ~ $(_distribution_to_string(spec.component_obj.sigma))
    """

    # --- 2. Generate Spectral Sampling Code for the Latent Field ---
    assembly_lines = [
        "# --- ICAR spectral assembly: $(key) ---",
        "# Sample standard normal noise.",
        "$(p_names.struct) ~ MvNormal(zeros(T, $(spec.hyper.n_latent)), I)",
        "",
        "# Construct the diagonal of the spectral transformation matrix D.",
        "# For ICAR, Cov = sigma^2 * Q_template_inv. In spectral domain, this is sigma^2 / L_j.",
        "# D = diag(sigma / sqrt(L_j) )",
        "local diag_D = $(p_names.sigma) ./ sqrt.(spec.L .+ 1e-9)",
        "# Set the eigenvalue corresponding to the null-space to zero to enforce the sum-to-zero constraint.",
        "# The first eigenvalue of a scaled ICAR precision matrix is zero.",
        "diag_D[1] = 0.0",
        "",
        "# Apply the spectral transformation: latent = U * D * z",
        "local icar_field = spec.U * (diag_D .* $(p_names.struct))",
        "",
        "# The sum-to-zero constraint is implicitly handled by setting the first diagonal element to zero.",
        "",
        "# Add the final effect to the linear predictor, indexed by the spatial units.",
        "eta .+= icar_field[M.s_idx]"
    ]
    assembly_str = join(assembly_lines, "\n    ")

    return (priors=priors_str, assembly=assembly_str, post_assembly="")
end



# Version 2.0.0 (2026-08-06)
# Purpose: Dispatches to the appropriate code generation function for a given model component.
# Rationale: This version is updated to use spectral decomposition for GMRF models like BYM2
#            and ICAR when the `spectral_orientation` flag is true and the necessary `U` and `L` matrices
#            are available in the component specification. This prioritizes AD-friendly sampling
#            methods over the older MvNormalCanon approach.
# Assumptions: The main model configuration `M` contains a `spectral_orientation` flag.
function generate_component_assembly(spec::NamedTuple, M::NamedTuple, arch::String)
    # This function acts as a dispatcher, calling the correct code generator
    # based on the component's type and the model's configuration.

    m_obj = spec.component_obj
    use_spectral = get(M, :spectral_orientation, true) && hasproperty(spec, :U) && hasproperty(spec, :L)

    # --- Dispatch Logic ---
    if m_obj isa BYM2 && use_spectral
        # New path: Use the AD-friendly spectral sampling method.
        return generate_bym2_assembly_spectral(spec, M, arch)
    
    elseif m_obj isa Union{ICAR, Besag} && use_spectral
        # New path: Use the AD-friendly spectral sampling method for ICAR/Besag.
        return generate_icar_assembly_spectral(spec, M, arch)

    elseif m_obj isa SciMLComponent
        return generate_sciml_component_assembly(spec, M, arch)

    # ... other component types would have their own dispatch logic here ...

    else
        # Fallback to legacy or other component generators
        @warn "No spectral generator for $(typeof(m_obj)). Falling back to default assembly."
        # This would call the old MvNormalCanon-based generator, which is not shown here
        # but is assumed to exist in the full codebase.
        # return generate_generic_assembly_legacy(spec, M, arch) 
        return (priors="", assembly="# Legacy assembly for $(spec.key) not shown.", post_assembly="")
    end
end

 


# Version 1.0.0 (2026-08-06)
# Purpose: Evaluates a kernel function between two points with a given lengthscale.
# Rationale: This AD-safe helper function is for use inside the generated Turing model.
#            It computes the kernel value for a single pair of points, which is needed
#            for the explicit loop in the process convolution model. It supports multiple
#            kernel types and ensures all operations are compatible with `ForwardDiff.Dual` types.
function _evaluate_kernel_pointwise(p1::AbstractVector, p2::AbstractVector, sigma::Real, ls::Real, kernel_type::Symbol)
    # Promote types to handle potential Dual numbers from AD
    T = promote_type(eltype(p1), eltype(p2), typeof(sigma), typeof(ls))
    
    # Ensure ls is positive to avoid numerical issues
    ls_safe = ls + convert(T, 1e-9)
    
    dist_sq = sum((p1 .- p2).^2)

    if kernel_type == :gaussian || kernel_type == :se
        return sigma^2 * exp(-dist_sq / (convert(T, 2.0) * ls_safe^2))
    
    elseif kernel_type == :matern32
        d = sqrt(dist_sq)
        val = sqrt(convert(T, 3.0)) * d / ls_safe
        return sigma^2 * (one(T) + val) * exp(-val)
        
    elseif kernel_type == :exponential
        d = sqrt(dist_sq)
        return sigma^2 * exp(-d / ls_safe)
        
    else # Default to squared exponential
        return sigma^2 * exp(-dist_sq / (convert(T, 2.0) * ls_safe^2))
    end
end
 
# Version 1.0.0 (2026-08-06)
# Purpose: Generates the Turing model code for the `ProcessConvolution` component.
# Rationale: This function implements a non-stationary spatial process by constructing
#            a convolution of a base process with a spatially varying kernel.
#            1. It first generates the code to realize a spatially varying `lengthscale_field`
#               by calling the code generator for its nested `lengthscale_model`.
#            2. It defines priors for the `base_process` at a set of knots.
#            3. It then enters a loop over all observation points to dynamically construct
#               a basis matrix `B`, where each entry `B[i, k]` is the kernel evaluation
#               between observation `i` and knot `k`, using the specific lengthscale `ls_i`
#               at that observation's location.
#            4. The final effect is the matrix-vector product of this dynamic basis `B`
#               and the `base_process`, which is then added to the linear predictor.
#            WARNING: The explicit loop over observations and knots makes this component
#            computationally intensive. It is intended for models where non-stationarity
#            is a critical feature and the number of knots and observations is moderate.
function _generate_component_code_fragments(m::ProcessConvolution, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple; prefix::String="", generate_eta_update::Bool=true)
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    hyper = spec.hyper
    n_knots = hyper.n_knots

    # 1. Generate code for the nested lengthscale model
    ls_model_obj = hyper.lengthscale_model_obj
    ls_model_spec_hyper = hyper.lengthscale_model_spec.hyper
    
    # Create a dummy spec for the inner model to pass to its generator
    inner_spec = (
        key = Symbol("$(key_str)_ls"),
        component_obj = ls_model_obj,
        hyper = ls_model_spec_hyper,
        Q_template = hyper.lengthscale_model_spec.Q_template,
        params = Dict() # Not strictly needed here
    )
    ls_frags = _generate_component_code_fragments(ls_model_obj, inner_spec, arch, outcome_idx, M; prefix="$(prefix)_ls", generate_eta_update=false)
    ls_latent_var = generate_full_variable_names(inner_spec, arch, outcome_idx, prefix="$(prefix)_ls").latent

    # 2. Priors for the base process
    priors_acc = [ls_frags.priors]
    push!(priors_acc, "$(v.sigma) ~ $(_distribution_to_string(hyper.base_sigma_prior))")
    push!(priors_acc, "$(v.raw) ~ MvNormal(zeros(T, $(n_knots)), I)")
    priors_str = join(priors_acc, "\n    ")

    # 3. Assembly block
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"

    update_str = """
    begin
        # --- Process Convolution: $(key_str) ---
        # This component models a non-stationary process by convolving a base process
        # with a kernel whose parameters (e.g., lengthscale) vary spatially.
        # WARNING: This involves a loop over all observations and knots, which can be
        # computationally expensive for large datasets.

        # A. Realize the spatially-varying lengthscale field from the nested model.
        $(ls_frags.update)
        local lengthscale_field = exp.($(ls_latent_var)) # Use exp to ensure positivity

        # B. Realize the base process at the knot locations.
        local base_process = $(v.raw) * $(v.sigma)

        # C. Dynamically construct the convolution basis matrix B.
        #    The entry B[i, k] is the kernel evaluation between observation i and knot k,
        #    using the lengthscale specific to observation i's location.
        local coords = spec_registry["$(key_str)"].hyper.coords
        local knots = spec_registry["$(key_str)"].hyper.knots
        local kernel_type = Symbol("$(hyper.kernel)")
        local n_obs = size(coords, 1)
        local B_conv = zeros(T, n_obs, $(n_knots))

        for i in 1:n_obs
            # The lengthscale field is indexed by the spatial unit of the observation.
            local ls_i = lengthscale_field[M.s_idx[i]]
            for k in 1:$(n_knots)
                # This helper function must be defined within the model's scope.
                B_conv[i, k] = _evaluate_kernel_pointwise(
                    view(coords, i, :), 
                    view(knots, k, :), 
                    one(T), # Sigma is applied to the base process, not here.
                    ls_i, 
                    kernel_type
                )
            end
        end

        # D. Compute the final effect and add it to the linear predictor.
        $(v.latent) = B_conv * base_process
        $(eta_target) .+= $(v.latent)
    end
    """
    return (priors=priors_str, update=update_str)
end
 
# Version 1.0.0 (2026-08-06)
# Purpose: Generates the Turing model code for a Leroux component using spectral decomposition.
# Rationale: This function implements the "spectral trick" for sampling the latent Leroux field.
#            It avoids constructing a parameter-dependent precision matrix and using MvNormalCanon,
#            which is not AD-friendly. Instead, it samples standard normal noise and transforms
#            it using the pre-computed eigenvectors (U) and a diagonal matrix (D) constructed
#            from the eigenvalues (L) and the sampled hyperparameters (sigma, rho).
# Assumptions: The component specification `spec` contains pre-computed `U` and `L`.
function generate_leroux_assembly_spectral(spec::NamedTuple, M::NamedTuple, arch::String)
    key = spec.key
    p_names = generate_full_variable_names(spec, arch, get(spec.params, :outcome_idx, nothing))

    priors_str = """
    # --- Priors for Leroux component: $(key) ---
    $(p_names.sigma) ~ $(_distribution_to_string(spec.component_obj.sigma))
    $(p_names.rho) ~ $(_distribution_to_string(spec.component_obj.rho))
    """

    assembly_lines = [
        "# --- Leroux spectral assembly: $(key) ---",
        "$(p_names.struct) ~ MvNormal(zeros(T, $(spec.hyper.n_latent)), I)",
        "",
        "# Construct the diagonal of the spectral transformation matrix D.",
        "# For Leroux, Q = (1-rho)*I + rho*Q_star. Eigenvalues are (1-rho) + rho*L_j.",
        "# D = diag(sigma / sqrt( (1-rho) + rho*L_j ) )",
        "local diag_D = $(p_names.sigma) ./ sqrt.((1.0 .- $(p_names.rho)) .+ $(p_names.rho) .* spec.L .+ 1e-9)",
        "",
        "# Apply the spectral transformation: latent = U * D * z",
        "local leroux_field = spec.U * (diag_D .* $(p_names.struct))",
        "",
        "eta .+= leroux_field[M.s_idx]"
    ]
    assembly_str = join(assembly_lines, "\n    ")

    return (priors=priors_str, assembly=assembly_str, post_assembly="")
end


# Version 1.0.0 (2026-08-06)
# Purpose: Generates the Turing model code for RW1 and RW2 components using spectral decomposition.
# Rationale: Implements the "spectral trick" for sampling intrinsic random walk fields.
#            This approach is fully differentiable and avoids Cholesky decomposition of
#            parameter-dependent matrices. It enforces the necessary sum-to-zero constraints
#            by zeroing out the components of the spectral transformation that correspond
#            to the null space of the precision matrix.
# Assumptions: The component specification `spec` contains pre-computed `U` and `L`.
function generate_rw_assembly_spectral(spec::NamedTuple, M::NamedTuple, arch::String)
    key = spec.key
    m_obj = spec.component_obj
    p_names = generate_full_variable_names(spec, arch, get(spec.params, :outcome_idx, nothing))

    rank_deficiency = if m_obj isa RW1; 1; elseif m_obj isa RW2; 2; else 0; end

    priors_str = """
    # --- Priors for $(typeof(m_obj)) component: $(key) ---
    $(p_names.sigma) ~ $(_distribution_to_string(m_obj.sigma))
    """

    assembly_lines = [
        "# --- $(typeof(m_obj)) spectral assembly: $(key) ---",
        "$(p_names.struct) ~ MvNormal(zeros(T, $(spec.hyper.n_latent)), I)",
        "",
        "# Construct the diagonal of the spectral transformation matrix D.",
        "# D = diag(sigma / sqrt(L_j) )",
        "local diag_D = $(p_names.sigma) ./ sqrt.(spec.L .+ 1e-9)",
        "",
        "# Enforce sum-to-zero constraint(s) by zeroing out components corresponding to the null space.",
    ]
    for i in 1:rank_deficiency
        push!(assembly_lines, "diag_D[$(i)] = 0.0")
    end
    
    push!(assembly_lines, "")
    push!(assembly_lines, "# Apply the spectral transformation: latent = U * D * z")
    push!(assembly_lines, "local rw_field = spec.U * (diag_D .* $(p_names.struct))")
    push!(assembly_lines, "")
    push!(assembly_lines, "eta .+= rw_field[M.t_idx]")

    assembly_str = join(assembly_lines, "\n    ")

    return (priors=priors_str, assembly=assembly_str, post_assembly="")
end


