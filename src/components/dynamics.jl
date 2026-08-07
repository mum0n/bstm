# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    Dynamics <: ComponentModel

A component for embedding mechanistic, process-based models within the `bstm` framework.
It simulates the evolution of a latent field over space and time according to a
user-specified differential or difference equation.

# Fields
- `model::String`: The name of the specific dynamics model to use (e.g., "logistic", "advection_diffusion", "leslie_matrix").
- `params::Dict{Symbol, Any}`: A dictionary of parameters and priors for the specified model.
"""
struct Dynamics <: ComponentModel
    model::String
    params::Dict{Symbol, Any}
end

# Add to the central component constructor registry.
COMPONENT_CONSTRUCTORS[:dynamics] = (p, params) -> Dynamics(string(get(params, :model, "none")), params)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[Dynamics] = :spacetime

"""
    get_datastructures!(m_type::Type{<:Dynamics}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Dynamics`. It establishes the spatiotemporal
context (`s_N`, `t_N`, `W`), processes exploitation data (`effort`, `removal`), and validates
the configuration for the specified dynamic model.
"""
function get_datastructures!(m_type::Type{<:Dynamics}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    data = M[:data]
    variables = mod_data[:variables]

    if length(variables) < 2
        error("Dynamics module requires at least two positional arguments: a spatial index and a temporal index (e.g., dynamics(s_idx, year, ...)).")
    end

    spatial_idx_var_sym = Symbol(variables[1])
    temporal_idx_var_sym = Symbol(variables[2])

    if !hasproperty(data, spatial_idx_var_sym); error("Spatial index variable ':$spatial_idx_var_sym' for dynamics module not found in data."); end
    M[:s_idx] = data[!, spatial_idx_var_sym]
    M[:s_N] = length(unique(M[:s_idx]))

    if !hasproperty(data, temporal_idx_var_sym); error("Temporal index variable ':$temporal_idx_var_sym' for dynamics module not found in data."); end
    M[:t_idx] = data[!, temporal_idx_var_sym]
    M[:t_N] = length(unique(M[:t_idx]))

    # Process adjacency matrix W
    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr || w_val isa Symbol
            calling_mod = get(M, :calling_module, Main)
            try; M[:W] = Core.eval(calling_mod, w_val); catch e; error("Could not evaluate `W` argument `$(w_val)`. Error: $e"); end
        else
            M[:W] = w_val
        end
    end
    if !haskey(M, :W); error("Dynamics models require an adjacency matrix `W`."); end
    if M[:s_N] != size(M[:W], 1); error("Dimension of `W` ($(size(M[:W], 1))) does not match number of spatial units ($(M[:s_N]))."); end

    # Process grid areas
    if !haskey(M, :grid_areas)
        if haskey(params, :grid_areas)
            ga_val = params[:grid_areas]
            if ga_val isa Symbol && hasproperty(data, ga_val); M[:grid_areas] = data[!, ga_val];
            elseif ga_val isa AbstractVector; M[:grid_areas] = ga_val;
            else; try; M[:grid_areas] = Core.eval(get(M, :calling_module, Main), ga_val); catch; M[:grid_areas] = ones(M[:s_N]); end; end
        else
            M[:grid_areas] = ones(M[:s_N])
        end
    end

    return true
end

"""
    get_precomputes(m::Dynamics, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `Dynamics`. This includes
building the graph Laplacian (`L_template`) and advection operator (`A_template`), and
processing exploitation data into a spatiotemporal grid format.
"""
function get_precomputes(m::Dynamics, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.s_N
    W = M.W

    L_template = build_structure_template(:besag, n; W=W).matrix
    A_template = if m.model in ["advection", "advection_diffusion"]
        W_dir = tril(W, -1)
        out_degree = sum(W_dir, dims=2)[:]
        D_inv = spdiagm(0 => 1.0 ./ (out_degree .+ 1e-9))
        D_inv * W_dir
    else
        spzeros(Float64, n, n)
    end

    processed_params = Dict{Symbol, Any}()
    effort_keys = Symbol[]
    removal_keys = Symbol[]

    for param_base_name in [:effort, :removal]
        if haskey(m.params, param_base_name)
            val = m.params[param_base_name]
            target_keys_list = param_base_name == :effort ? effort_keys : removal_keys
            
            vals_to_process = val isa Vector && !(val isa AbstractVector{<:Real}) ? val : [val]

            for (i, v) in enumerate(vals_to_process)
                storage_key = length(vals_to_process) > 1 ? Symbol("$(param_base_name)_$(i)") : param_base_name
                covariate_data = if v isa AbstractArray{<:Real}; v; elseif v isa Symbol && hasproperty(M.data, v); M.data[!, v]; else; nothing; end

                if !isnothing(covariate_data)
                    if ndims(covariate_data) == 1
                        cov_matrix = zeros(M.s_N, M.t_N)
                        counts = zeros(Int, M.s_N, M.t_N)
                        for obs_i in 1:M.y_N
                            si, ti = M.s_idx[obs_i], M.t_idx[obs_i]
                            cov_matrix[si, ti] += covariate_data[obs_i]
                            counts[si, ti] += 1
                        end
                        cov_matrix ./= max.(1, counts)
                        processed_params[storage_key] = cov_matrix
                        push!(target_keys_list, storage_key)
                    else
                        processed_params[storage_key] = covariate_data
                        push!(target_keys_list, storage_key)
                    end
                end
            end
        end
    end

    return (
        L_template=L_template,
        A_template=A_template,
        areas=M.grid_areas,
        effort_keys=effort_keys,
        removal_keys=removal_keys,
        processed_params=processed_params
    )
end

"""
    _generate_exploitation_block(spec, time_var)

A helper function to generate the Turing code block for calculating exploitation
(e.g., catch from fishing effort) within a dynamics model simulation loop.
"""
function _generate_exploitation_block(spec, time_var)
    effort_keys = get(spec.hyper, :effort_keys, [])
    removal_keys = get(spec.hyper, :removal_keys, [])
    
    if isempty(effort_keys) && isempty(removal_keys)
        return "local exploitation = zeros(T_num_dyn, M.s_N)"
    end
    
    lines = ["local exploitation = zeros(T_num_dyn, M.s_N)"]
    for key in effort_keys
        push!(lines, "exploitation .+= q_$(key) .* spec_registry[:$(spec.key)].hyper.processed_params[:$(key)][:, $(time_var)] .* N_prev")
    end
    for key in removal_keys
        push!(lines, "exploitation .+= spec_registry[:$(spec.key)].hyper.processed_params[:$(key)][:, $(time_var)]")
    end
    return join(lines, "\n    ")
end

"""
    get_priors(m::Dynamics, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `Dynamics`'s priors.
"""
function get_priors(m::Dynamics, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    params = m.params
    priors_acc = String[]

    # Priors for univariate dynamics models
    if arch == "univariate"
        if m.model in ["advection", "advection_diffusion"]; push!(priors_acc, "$(p_names.velocity) ~ NamedDist($(_distribution_to_string(get(params, :velocity, Normal(0, 0.5)))), :$(p_names.velocity))"); end
        if m.model in ["diffusion", "advection_diffusion"]; push!(priors_acc, "$(p_names.diffusion) ~ NamedDist($(_distribution_to_string(get(params, :diffusion, LogNormal(-1, 1)))), :$(p_names.diffusion))"); end
        
        push!(priors_acc, "$(p_names.sigma) ~ NamedDist($(_distribution_to_string(get(params, :sigma, Exponential(1.0)))), :$(p_names.sigma))")
        
        if m.model in ["logistic", "delay_difference"]
            if get(params, :spatially_varying_r, false); push!(priors_acc, "sigma_r_$(spec.key) ~ NamedDist(Exponential(1.0), :sigma_r_$(spec.key))"); push!(priors_acc, "log_r_mean_$(spec.key) ~ NamedDist(Normal(0.0, 0.5), :log_r_mean_$(spec.key))"); push!(priors_acc, "r_raw_$(spec.key) ~ NamedDist(MvNormal(zeros(T, M.s_N), I), :r_raw_$(spec.key))"); else; push!(priors_acc, "$(p_names.r) ~ NamedDist($(_distribution_to_string(get(params, :r, LogNormal(0,1)))), :$(p_names.r))"); end
            if get(params, :spatially_varying_K, false); push!(priors_acc, "sigma_K_$(spec.key) ~ NamedDist(Exponential(1.0), :sigma_K_$(spec.key))"); push!(priors_acc, "log_K_mean_$(spec.key) ~ NamedDist(Normal(log(100.0), 0.5), :log_K_mean_$(spec.key))"); push!(priors_acc, "K_raw_$(spec.key) ~ NamedDist(MvNormal(zeros(T, M.s_N), I), :K_raw_$(spec.key))"); else; push!(priors_acc, "$(p_names.K) ~ NamedDist($(_distribution_to_string(get(params, :K, LogNormal(log(100.0),1)))), :$(p_names.K))"); end
            for key in spec.hyper.effort_keys; push!(priors_acc, "q_$(key) ~ NamedDist($(_distribution_to_string(get(params, Symbol("q_$(key)"), LogNormal(-2,1)))), :q_$(key))"); end
        end
        if m.model == "delay_difference"; push!(priors_acc, "$(p_names.M_nat) ~ NamedDist($(_distribution_to_string(get(params, :M_nat, LogNormal(-1, 0.5)))), :$(p_names.M_nat))"); end
        
        if m.model == "lotka_volterra"
            push!(priors_acc, "$(p_names.alpha) ~ NamedDist($(_distribution_to_string(get(params, :alpha, LogNormal(0, 0.5)))), :$(p_names.alpha))")
            push!(priors_acc, "$(p_names.beta) ~ NamedDist($(_distribution_to_string(get(params, :beta, LogNormal(-1, 0.5)))), :$(p_names.beta))")
            push!(priors_acc, "$(p_names.gamma) ~ NamedDist($(_distribution_to_string(get(params, :gamma, LogNormal(-1, 0.5)))), :$(p_names.gamma))")
            push!(priors_acc, "$(p_names.delta) ~ NamedDist($(_distribution_to_string(get(params, :delta, LogNormal(0, 0.5)))), :$(p_names.delta))")
            push!(priors_acc, "$(p_names.innov)_predator ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :$(Symbol(string(p_names.innov, "_predator"))))")
        end

        push!(priors_acc, "$(p_names.innov) ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :$(p_names.innov))")
    
    # Priors for multivariate dynamics models
    elseif arch == "multivariate"
        if m.model == "leslie_matrix"
            n_classes = get(params, :n_age_classes, M.outcomes_N)
            if get(params, :spatially_varying_rates, false)
                push!(priors_acc, "log_fecundity_mean_$(spec.key) ~ NamedDist(filldist(Normal(0, 1), $(n_classes)), :log_fecundity_mean_$(spec.key))")
                push!(priors_acc, "sigma_fecundity_$(spec.key) ~ NamedDist(filldist(Exponential(1.0), $(n_classes)), :sigma_fecundity_$(spec.key))")
                push!(priors_acc, "fecundity_raw_$(spec.key) ~ NamedDist(MvNormal(zeros(T, M.s_N * $(n_classes)), I), :fecundity_raw_$(spec.key))")
                push!(priors_acc, "logit_survival_mean_$(spec.key) ~ NamedDist(filldist(Normal(1.5, 1), $(n_classes-1)), :logit_survival_mean_$(spec.key))")
                push!(priors_acc, "sigma_survival_$(spec.key) ~ NamedDist(filldist(Exponential(1.0), $(n_classes-1)), :sigma_survival_$(spec.key))")
                push!(priors_acc, "survival_raw_$(spec.key) ~ NamedDist(MvNormal(zeros(T, M.s_N * ($(n_classes)-1)), I), :survival_raw_$(spec.key))")
            else
                push!(priors_acc, "survival_rates_$(spec.key) ~ NamedDist(filldist(Beta(9, 1), $(n_classes - 1)), :survival_rates_$(spec.key))")
                push!(priors_acc, "fecundity_rates_$(spec.key) ~ NamedDist(filldist(LogNormal(0, 1), $(n_classes)), :fecundity_rates_$(spec.key))")
            end
            push!(priors_acc, "sigma_process_$(spec.key) ~ NamedDist(filldist(Exponential(1.0), $(n_classes)), :sigma_process_$(spec.key))")
            push!(priors_acc, "innov_process_$(spec.key) ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N * $(n_classes)), I), :innov_process_$(spec.key))")
        
        elseif m.model == "generalized_lotka_volterra"
            n_species = M.outcomes_N
            push!(priors_acc, "r_$(spec.key) ~ NamedDist(filldist(LogNormal(0, 1), $(n_species)), :r_$(spec.key))")
            push!(priors_acc, "alpha_raw_$(spec.key) ~ NamedDist(MvNormal(zeros(T, $(n_species * (n_species - 1))), I), :alpha_raw_$(spec.key))")
            push!(priors_acc, "sigma_process_$(spec.key) ~ NamedDist(filldist(Exponential(1.0), $(n_species)), :sigma_process_$(spec.key))")
            push!(priors_acc, "innov_process_$(spec.key) ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N * $(n_species)), I), :innov_process_$(spec.key))")
        end
    end

    return join(priors_acc, "\n    ")
end

"""
    get_updates(m::Dynamics, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for simulating the `Dynamics`'s effect.
"""
function get_updates(m::Dynamics, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent" : "eta"

    if arch == "univariate"
        propagator_setup = ""
        if m.model in ["advection", "advection_diffusion"]; propagator_setup *= "local propagator = lu(I(M.s_N) - $(p_names.velocity) * spec_registry[:$(spec.key)].hyper.A_template + M.noise * I)\n"; end
        if m.model in ["diffusion", "advection_diffusion"]; propagator_setup *= "local propagator = cholesky(Symmetric(I(M.s_N) - $(p_names.diffusion) * spec_registry[:$(spec.key)].hyper.L_template + M.noise * I))\n"; end
        
        field_setup = "local T_num_dyn = eltype($(p_names.innov)); local dyn_field = zeros(T_num_dyn, M.s_N, M.t_N)\n    local innov_matrix = reshape($(p_names.innov), M.s_N, M.t_N)\n    dyn_field[:, 1] = innov_matrix[:, 1]"
        
        K_setup_block = get(m.params, :spatially_varying_K, false) ? "local F_K = cholesky(Symmetric(Matrix(spec_registry[:$(spec.key)].hyper.L_template) + M.noise * I)); local K_field_raw = F_K.L' \\ K_raw_$(spec.key); Turing.@addlogprob! logpdf(Normal(0, 0.001 * M.s_N), sum(K_field_raw)); local K_spatial = exp.(log_K_mean_$(spec.key) .+ K_field_raw .* sigma_K_$(spec.key))" : ""
        r_setup_block = get(m.params, :spatially_varying_r, false) ? "local F_r = cholesky(Symmetric(Matrix(spec_registry[:$(spec.key)].hyper.L_template) + M.noise * I)); local r_field_raw = F_r.L' \\ r_raw_$(spec.key); Turing.@addlogprob! logpdf(Normal(0, 0.001 * M.s_N), sum(r_field_raw)); local r_spatial = exp.(log_r_mean_$(spec.key) .+ r_field_raw .* sigma_r_$(spec.key))" : ""
        
        K_var = get(m.params, :spatially_varying_K, false) ? "K_spatial" : string(p_names.K)
        r_var = get(m.params, :spatially_varying_r, false) ? "r_spatial" : string(p_names.r)
        
        propagator_logic = hasproperty(m, :velocity) || hasproperty(m, :diffusion) ? "dyn_field[:, t] = (propagator \\ N_intermediate) .+ innov_matrix[:, t]" : "dyn_field[:, t] = N_intermediate .+ innov_matrix[:, t]"

        local evolution_loop_body
        if m.model == "logistic"
            exploitation_logic = _generate_exploitation_block(spec, "t")
            evolution_loop_body = "local areas = spec_registry[:$(spec.key)].hyper.areas\nfor t in 2:M.t_N\n    local N_prev = dyn_field[:, t-1]; local D_prev = N_prev ./ areas; local K_density = $(K_var) ./ areas; local growth = $(r_var) .* D_prev .* (1.0 .- D_prev ./ K_density); $(exploitation_logic); local N_intermediate = N_prev .+ (growth .* areas) .- exploitation; $(propagator_logic); dyn_field[:, t] = max.(T_num_dyn(0.0), dyn_field[:, t]);\nend"
        elseif m.model == "delay_difference"
            exploitation_logic = _generate_exploitation_block(spec, "t-1")
            evolution_loop_body = "local areas = spec_registry[:$(spec.key)].hyper.areas\nfor t in 2:M.t_N\n    local N_prev = dyn_field[:, t-1]; local D_prev = N_prev ./ areas; local K_density = $(K_var) ./ areas; local growth = $(r_var) .* D_prev .* (1.0 .- D_prev ./ K_density); $(exploitation_logic); local N_survived = (N_prev .- exploitation) .* exp.(-$(p_names.M_nat)); local N_intermediate = N_survived .+ (growth .* areas); $(propagator_logic); dyn_field[:, t] = max.(T_num_dyn(0.0), dyn_field[:, t]);\nend"
        elseif m.model == "lotka_volterra"
            output_species = get(m.params, :output_species, :prey)
            field_setup = "local T_num_dyn = eltype($(p_names.innov)); local dyn_field_prey = zeros(T_num_dyn, M.s_N, M.t_N); local dyn_field_predator = zeros(T_num_dyn, M.s_N, M.t_N); local innov_matrix_prey = reshape($(p_names.innov), M.s_N, M.t_N); local innov_matrix_predator = reshape($(p_names.innov)_predator, M.s_N, M.t_N); dyn_field_prey[:, 1] = innov_matrix_prey[:, 1]; dyn_field_predator[:, 1] = innov_matrix_predator[:, 1]"
            evolution_loop_body = "for t in 2:M.t_N\n    local N_prey_prev = dyn_field_prey[:, t-1]; local N_pred_prev = dyn_field_predator[:, t-1]; local d_prey = ($(p_names.alpha) .* N_prey_prev) .- ($(p_names.beta) .* N_prey_prev .* N_pred_prev); local d_pred = ($(p_names.gamma) .* N_prey_prev .* N_pred_prev) .- ($(p_names.delta) .* N_pred_prev); dyn_field_prey[:, t] = max.(T_num_dyn(0.0), N_prey_prev .+ d_prey .+ innov_matrix_prey[:, t]); dyn_field_predator[:, t] = max.(T_num_dyn(0.0), N_pred_prev .+ d_pred .+ innov_matrix_predator[:, t]);\nend\nlocal dyn_field = $(output_species == :prey ? "dyn_field_prey" : "dyn_field_predator")"
        else
            evolution_loop_body = "for t in 2:M.t_N\n    dyn_field[:, t] = (propagator \\ dyn_field[:, t-1]) + innov_matrix[:, t];\nend"
        end

        return """
        begin
            # Dynamics model: $(m.model) for $(spec.key)
            $(K_setup_block)
            $(r_setup_block)
            $(propagator_setup)
            $(field_setup)
            $(evolution_loop_body)
            dyn_field .*= $(p_names.sigma)
            for i in 1:M.y_N
                $(eta_target)[i] += log(dyn_field[M.s_idx[i], M.t_idx[i]] + T(1e-6))
            end
        end
        """
    elseif arch == "multivariate"
        key_str = string(spec.key)
        if m.model == "leslie_matrix"
            n_classes = get(params, :n_age_classes, M.outcomes_N)
            spatially_varying_K = get(params, :spatially_varying_K, false)
            spatially_varying_rates = get(params, :spatially_varying_rates, false)
            exploitation_block = _generate_exploitation_block(spec, "t-1")
            return """
            begin
                local Q_spatial = spec_registry[:$(key_str)].hyper.L_template; local F_spatial = cholesky(Symmetric(Matrix(Q_spatial) + M.noise * I)); local areas = spec_registry[:$(key_str)].hyper.areas
                local survival_rates_spatial, fecundity_rates_spatial; if $(spatially_varying_rates); fecundity_raw_matrix = reshape(fecundity_raw_$(key_str), M.s_N, $(n_classes)); fecundity_field = F_spatial.L' \\ fecundity_raw_matrix; fecundity_rates_spatial = exp.(log_fecundity_mean_$(key_str)' .+ fecundity_field .* sigma_fecundity_$(key_str)'); survival_raw_matrix = reshape(survival_raw_$(key_str), M.s_N, $(n_classes-1)); survival_field = F_spatial.L' \\ survival_raw_matrix; survival_rates_spatial = logistic.(logit_survival_mean_$(key_str)' .+ survival_field .* sigma_survival_$(key_str)'); end
                local K_values_$(key_str); if $(spatially_varying_K); K_field_raw = F_spatial.L' \\ K_raw_$(key_str); Turing.@addlogprob! logpdf(Normal(0, 0.001 * M.s_N), sum(K_field_raw)); K_values_$(key_str) = exp.(log_K_mean_$(key_str) .+ K_field_raw .* sigma_K_$(key_str)); elseif haskey(spec.params, :K); K_values_$(key_str) = fill(K_$(key_str), M.s_N); end
                local innov_tensor_$(key_str) = reshape(innov_process_$(key_str), M.s_N, M.t_N, $(n_classes)); local population_field_$(key_str) = zeros(T, M.s_N, M.t_N, $(n_classes))
                for a in 1:$(n_classes); population_field_$(key_str)[:, 1, a] = max.(0.0, innov_tensor_$(key_str)[:, 1, a] .* sigma_process_$(key_str)[a]); end
                for s in 1:M.s_N
                    local L_s = zeros(T, $(n_classes), $(n_classes)); if $(spatially_varying_rates); for i in 1:($(n_classes)-1); L_s[i+1, i] = survival_rates_spatial[s, i]; end; L_s[1, :] = fecundity_rates_spatial[s, :]; else; for i in 1:($(n_classes)-1); L_s[i+1, i] = survival_rates_$(key_str)[i]; end; L_s[1, :] = fecundity_rates_$(key_str); end
                    for t in 2:M.t_N
                        local N_prev = view(population_field_$(key_str), s, t-1, :); $(exploitation_block); local N_after_removal = max.(0.0, N_prev - C_prev); local L_effective = copy(L_s)
                        if haskey(spec.params, :K); local total_pop_prev = sum(N_after_removal); local K_density = K_values_$(key_str)[s] / areas[s]; local dd_factor = max(0.0, 1.0 - (total_pop_prev / areas[s]) / K_density); L_effective[1, :] .*= dd_factor; end
                        local N_projected = L_effective * N_after_removal; local current_innov = view(innov_tensor_$(key_str), s, t, :) .* sigma_process_$(key_str); population_field_$(key_str)[s, t, :] = max.(0.0, N_projected .+ current_innov)
                    end
                end
                for k in 1:$(n_classes); for i in 1:M.y_N; $(eta_target)[i, k] += log(population_field_$(key_str)[M.s_idx[i], M.t_idx[i], k] + 1e-6); end; end
            end
            """
        end
    end
    return "# Dynamics model $(m.model) not implemented for this architecture."
end

"""
    get_effects(m::Dynamics, chain, M::NamedTuple, ...)::NamedTuple

Reconstructs the `Dynamics`'s effect from the MCMC chain's posterior samples.
"""
function get_effects(m::Dynamics, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)
    t_idx_full = isnothing(PS) ? M.t_idx : vcat(M.t_idx, PS.t_idx)
    t_N_full = maximum(t_idx_full)

    if m.model == "logistic"
        sigma_samples = get(chain, p_names.sigma)
        innov_samples = get(chain, p_names.innov)
        r_samples = get(chain, p_names.r)
        K_samples = get(chain, p_names.K)
        
        areas = spec.hyper.areas
        reconstructed_effects = zeros(n_samples, N_total)

        for i in 1:n_samples
            dyn_field = zeros(eltype(sigma_samples), M.s_N, t_N_full)
            innov_matrix = reshape(innov_samples[i, :], M.s_N, M.t_N)
            if t_N_full > M.t_N; innov_matrix = hcat(innov_matrix, randn(M.s_N, t_N_full - M.t_N)); end
            dyn_field[:, 1] = innov_matrix[:, 1]

            for t in 2:t_N_full
                N_prev = dyn_field[:, t-1]
                D_prev = N_prev ./ areas
                K_density = K_samples[i] ./ areas
                growth = r_samples[i] .* D_prev .* (1.0 .- D_prev ./ K_density)
                N_intermediate = N_prev .+ (growth .* areas)
                dyn_field[:, t] = max.(0.0, N_intermediate .+ innov_matrix[:, t])
            end
            
            dyn_field .*= sigma_samples[i]
            log_dyn_field = log.(dyn_field .+ 1e-6)
            
            for obs_idx in 1:N_total
                reconstructed_effects[i, obs_idx] = log_dyn_field[s_idx_full[obs_idx], t_idx_full[obs_idx]]
            end
        end
        
        mean_effect = mean(reconstructed_effects, dims=1)[:]
        lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
        upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]
        
        return (structured=(mean=mean_effect, lower=lower_ci, upper=upper_ci),)
    end

    @warn "Posterior reconstruction for Dynamics model '$(m.model)' is not implemented. Returning zero effects."
    return (structured=(mean=zeros(N_total), lower=zeros(N_total), upper=zeros(N_total)),)
end
