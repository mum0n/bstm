"""
    Dynamics <: ComponentModel

A component for embedding mechanistic, process-based models within the `bstm`
framework. It simulates the evolution of a latent field over space and time
according to a user-specified differential or difference equation.

# Version
v1.0.2 (2026-08-10)

# Mathematical Summary
Dynamics models describe the evolution of a latent field \$N(s, t)\$ over space \$s\$
and time \$t\$. The state at time \$t\$ is a function of the state at \$t-1\$, often
including process noise \$\\epsilon(s, t)\$.

1.  **General Form (Implicit Euler Scheme)**:
    \$N(s, t) = P^{-1} (N(s, t-1) + \\epsilon(s, t))\$
    where \$P\$ is a propagator matrix (e.g., \$I - \\text{operator} \\cdot \\Delta t\$).

2.  **Advection-Diffusion**:
    Models transport and spreading. The operator combines an advection term
    (e.g., \$v \\cdot A\$) and a diffusion term (e.g., \$D \\cdot L\$), where \$A\$ is an
    advection operator and \$L\$ is the graph Laplacian.
    \$P = I - v \\cdot A - D \\cdot L\$

3.  **Population Dynamics (Logistic, Delay-Difference, Lotka-Volterra, Leslie Matrix)**:
    These models describe population changes based on biological processes (growth,
    mortality, recruitment, competition) and can include exploitation (effort, removal).
    They are typically formulated as difference equations:
    \$N(s, t) = N(s, t-1) + \\text{Growth}(N(s, t-1)) - \\text{Exploitation}(N(s, t-1)) + \\epsilon(s, t)\$

    *   **Logistic**: \$N_{t+1} = N_t + r N_t (1 - N_t/K) - C_t\$
    *   **Delay-Difference**: \$N_{t+1} = (N_t - C_t)e^{-M} + R_{t+1}\$
    *   **Lotka-Volterra**: Predator-prey or competition dynamics.
    *   **Leslie Matrix**: Age-structured population dynamics.

# Assumptions
- The spatial domain is either provided as a graph (`W`) or can be discretized
  from coordinates into a regular grid.
- The evolution is deterministic given parameters, with additive process noise.
- Exploitation data (effort, removal) can be mapped to the spatiotemporal grid.

# Best Use Case
Modeling ecological population dynamics, disease spread, or other processes where
mechanistic understanding of change over space and time is critical. It allows
estimation of physical or biological parameters within a Bayesian framework.

# Key References
- Wikle, C. K. (2003). Hierarchical Bayesian models for predicting the spread of
  ecological processes. *Ecology*, 84(6), 1382-1394.
- Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time variation
  in disease risk. *Statistical Methods in Medical Research*, 9(3), 205-220.
- Wikipedia: Population dynamics

# Fields
- `model::String`: The name of the specific dynamics model to use (e.g., "logistic",
  "advection_diffusion", "leslie_matrix").
- `params::Dict{Symbol, Any}`: A dictionary of parameters and priors for the
  specified model.
- `resolution::Int`: The grid resolution to use when `W` is not provided (continuous mode).
"""
struct Dynamics <: ComponentModel
    model::String
    params::Dict{Symbol, Any}
    resolution::Int
end

COMPONENT_TYPE_REGISTRY[:dynamics] = Dynamics

COMPONENT_CONSTRUCTORS[:dynamics] = (p, params) -> Dynamics(
    string(get(params, :model, "none")), params, get(params, :resolution, 30)
)

MODEL_TO_STRUCTURE_MAP[:dynamics] = :spacetime

"""
    get_datastructures!(m_type::Type{<:Dynamics}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Dynamics` component. It supports two modes:
1.  **Graph Mode (Default)**: If an adjacency matrix `W` is provided, it uses the
    provided graph structure. Requires a spatial index variable.
2.  **Continuous Mode**: If `W` is not provided, it discretizes the domain into a
    regular grid based on coordinate variables and the `resolution` parameter.

In both modes, it establishes the spatiotemporal context (`s_N`, `t_N`), processes
exploitation data (`effort`, `removal`), and validates the configuration.
"""
function get_datastructures!(
    m_type::Type{<:Dynamics}, M::Dict, mod_data::Dict
)::Bool
    params = mod_data[:params]
    data = M[:data]
    variables = mod_data[:variables]
    m = Dynamics(string(get(params, :model, "none")), params, get(params, :resolution, 30))

    W_provided = haskey(params, :W) || haskey(M, :W)
    local temporal_idx_var_sym::Symbol

    if W_provided
        # --- Graph-based method (W is provided) ---
        if length(variables) < 2
            error("Graph-based dynamics requires at least two positional arguments: a spatial index and a temporal index.")
        end
        spatial_idx_var_sym = Symbol(variables[1])
        temporal_idx_var_sym = Symbol(variables[2])

        if haskey(params, :W)
            w_val = params[:W]
            if w_val isa Expr || w_val isa Symbol
                M[:W] = Core.eval(get(M, :calling_module, Main), w_val)
            else
                M[:W] = w_val
            end
        end
        M[:s_N] = size(M[:W], 1)
        
        if !hasproperty(data, spatial_idx_var_sym)
            error("Spatial index variable ':$spatial_idx_var_sym' not found for graph-based dynamics.")
        end
        M[:s_idx] = data[!, spatial_idx_var_sym]
    else
        # --- Continuous/Grid-based method (W is not provided) ---
        @info "Adjacency matrix 'W' not provided for Dynamics component. Creating a regular grid from coordinates."
        
        if length(variables) < 3
            error("Continuous dynamics requires three positional arguments: x-coordinate, y-coordinate, and a temporal index.")
        end
        x_coord_sym = Symbol(variables[1])
        y_coord_sym = Symbol(variables[2])
        temporal_idx_var_sym = Symbol(variables[3])

        if !hasproperty(data, x_coord_sym) || !hasproperty(data, y_coord_sym)
            error("Coordinate variables ':$x_coord_sym' or ':$y_coord_sym' not found for continuous dynamics.")
        end
        
        res = m.resolution
        M[:s_N] = res * res

        x_coords = data[!, x_coord_sym]
        y_coords = data[!, y_coord_sym]
        grid_x = range(minimum(x_coords), maximum(x_coords), length=res)
        grid_y = range(minimum(y_coords), maximum(y_coords), length=res)
        
        W_grid = spzeros(Int, M[:s_N], M[:s_N])
        centroids = Vector{Point2D}(undef, M[:s_N])
        for c in 1:res, r in 1:res
            idx = (c-1)*res + r
            centroids[idx] = Point2D(grid_x[r], grid_y[c])
            for dr in -1:1, dc in -1:1
                if dr == 0 && dc == 0; continue; end
                nr, nc = r + dr, c + dc
                if 1 <= nr <= res && 1 <= nc <= res
                    n_idx = (nc-1)*res + nr
                    W_grid[idx, n_idx] = 1
                end
            end
        end
        M[:W] = W_grid
        M[:centroids] = centroids

        s_idx_new = zeros(Int, nrow(data))
        for i in 1:nrow(data)
            obs_x, obs_y = x_coords[i], y_coords[i]
            best_r = searchsortedfirst(grid_x, obs_x)
            best_c = searchsortedfirst(grid_y, obs_y)
            if best_r > 1 && abs(grid_x[best_r-1] - obs_x) < abs(grid_x[best_r] - obs_x); best_r -= 1; end
            if best_c > 1 && abs(grid_y[best_c-1] - obs_y) < abs(grid_y[best_c] - obs_y); best_c -= 1; end
            s_idx_new[i] = (best_c-1)*res + best_r
        end
        M[:s_idx] = s_idx_new
    end

    # Common temporal setup
    if !hasproperty(data, temporal_idx_var_sym)
        error("Temporal index variable ':$temporal_idx_var_sym' not found.")
    end
    M[:t_idx] = data[!, temporal_idx_var_sym]
    M[:t_N] = length(unique(M[:t_idx]))

    # Process grid areas
    if !haskey(M, :grid_areas)
        if haskey(params, :grid_areas)
            ga_val = params[:grid_areas]
            if ga_val isa Symbol && hasproperty(data, ga_val)
                M[:grid_areas] = data[!, ga_val]
            elseif ga_val isa AbstractVector
                M[:grid_areas] = ga_val
            else
                try; M[:grid_areas] = Core.eval(get(M, :calling_module, Main), ga_val);
                catch; M[:grid_areas] = ones(M[:s_N]); end
            end
        else
            M[:grid_areas] = ones(M[:s_N])
        end
    end

    # Process exploitation data (effort or removal)
    processed_params = get(mod_data, :processed_params, Dict{Symbol, Any}())
    mod_data[:processed_params] = processed_params

    for param_base_name in [:effort, :removal]
        if haskey(params, param_base_name)
            val = params[param_base_name]
            vals_to_process = val isa Vector && !(val isa AbstractVector{<:Real}) ? val : [val]
            for (i, v) in enumerate(vals_to_process)
                storage_key = length(vals_to_process) > 1 ? Symbol("$(param_base_name)_$(i)") : param_base_name
                covariate_data = if v isa AbstractArray{<:Real}; v; elseif v isa Symbol && hasproperty(data, v); data[!, v]; else; nothing; end
                if !isnothing(covariate_data)
                    if ndims(covariate_data) == 1
                        cov_matrix = zeros(M[:s_N], M[:t_N])
                        counts = zeros(Int, M[:s_N], M[:t_N])
                        for obs_i in 1:M[:y_N]
                            si, ti = M[:s_idx][obs_i], M[:t_idx][obs_i]
                            cov_matrix[si, ti] += covariate_data[obs_i]
                            counts[si, ti] += 1
                        end
                        cov_matrix ./= max.(1, counts)
                        processed_params[storage_key] = cov_matrix
                    elseif ndims(covariate_data) >= 2
                        processed_params[storage_key] = covariate_data
                    end
                end
            end
        end
    end

    return true
end

"""
    get_precomputes(m::Dynamics, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `Dynamics` component. This
includes building the graph Laplacian (`L_template`) and advection operator
(`A_template`), and processing exploitation data into a spatiotemporal grid format.
"""
function get_precomputes(
    m::Dynamics, M::NamedTuple, mod_data::Dict
)::NamedTuple
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

    processed_params = get(mod_data, :processed_params, Dict{Symbol, Any}())
    
    effort_keys = [k for k in keys(processed_params) if startswith(string(k), "effort")]
    removal_keys = [k for k in keys(processed_params) if startswith(string(k), "removal")]

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
    generate_exploitation_block(spec, time_var)

Generates a Turing code fragment for calculating exploitation (e.g., catch or removals)
within a `dynamics` component.

# Rationale
This function is a helper for the `get_updates` method of the `Dynamics` component. It
is not deprecated and is consistent with the refactored architecture. Its purpose is
to dynamically construct the code that calculates total removals from a population
based on `effort` and/or `removal` parameters specified in the model formula.

This approach is consistent with the refactor's goals:
- **Modularity**: It encapsulates the specific logic for handling exploitation,
  keeping the main `dynamics` code generator cleaner.
- **Flexibility**: It supports multiple sources of exploitation (e.g., multiple
  fishing fleets with different catchability coefficients `q`) by iterating
  through the `effort_keys` and `removal_keys` stored in the component's
  `hyper` registry.
- **AD Compatibility**: The generated code initializes the `exploitation` variable
  as `zeros(T_num_dyn, M.s_N)`, where `T_num_dyn` is the promoted numeric type
  (e.g., `ForwardDiff.Dual`) within the dynamics loop. This ensures type stability
  and prevents `MethodError` during automatic differentiation, a key requirement
  of the refactor.

# Arguments
- `spec`: The `NamedTuple` specification for the `Dynamics` component.
- `time_var`: A string representing the time index variable in the generated code (e.g., "t" or "t-1").

# Returns
- A `String` containing the generated Turing code for the exploitation block.
"""
function generate_exploitation_block(spec, time_var)
    effort_keys = get(spec.hyper, :effort_keys, [])
    removal_keys = get(spec.hyper, :removal_keys, [])
    
    if isempty(effort_keys) && isempty(removal_keys)
        # If no exploitation is specified, return a zero-initialized vector.
        # This is initialized with the correct numeric type for AD safety.
        return "local exploitation = zeros(T_num_dyn, M.s_N)"
    end
    
    # Initialize the exploitation vector. The `local` keyword is necessary to
    # ensure the variable is scoped correctly within the generated model code.
    lines = ["local exploitation = zeros(T_num_dyn, M.s_N)"]

    # Add exploitation from effort-based removals.
    # Assumes a catchability coefficient `q_...` is defined in the model.
    for key in effort_keys
        push!(lines, "exploitation .+= q_$(key) .* spec_registry[\"$(spec.key)\"].hyper.processed_params[:$(key)][:, $(time_var)] .* N_prev")
    end

    # Add exploitation from direct removals.
    for key in removal_keys
        push!(lines, "exploitation .+= spec_registry[\"$(spec.key)\"].hyper.processed_params[:$(key)][:, $(time_var)]")
    end

    return join(lines, "\n    ")
end




"""
    get_priors(m::Dynamics, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code string for the `Dynamics`'s priors.
"""
function get_priors(
    m::Dynamics, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    params = m.params
    priors_acc = String[]

    if arch == "univariate"
        if m.model in ["advection", "advection_diffusion"]
            vel_prior = get(params, :velocity, Normal(0.0, 0.5))
            push!(priors_acc, "$(p_names.velocity) ~ $(_distribution_to_string(vel_prior))")
        end
        if m.model in ["diffusion", "advection_diffusion"]
            diff_prior = get(params, :diffusion, LogNormal(-1.0, 1.0))
            push!(priors_acc, "$(p_names.diffusion) ~ $(_distribution_to_string(diff_prior))")
        end
        
        sigma_prior = get(params, :sigma, Exponential(1.0))
        push!(priors_acc, "$(p_names.sigma) ~ $(_distribution_to_string(sigma_prior))")
        
        if m.model in ["logistic", "delay_difference"]
            if get(params, :spatially_varying_r, false)
                log_r_mean_prior = get(params, :log_r_mean, Normal(0.0, 0.5))
                sigma_r_prior = get(params, :sigma_r, Exponential(1.0))
                push!(priors_acc, "sigma_r_$(spec.key) ~ $(_distribution_to_string(sigma_r_prior))")
                push!(priors_acc, "log_r_mean_$(spec.key) ~ $(_distribution_to_string(log_r_mean_prior))")
                push!(priors_acc, "r_raw_$(spec.key) ~ MvNormal(zeros(M.s_N), I)")
            else
                r_prior = get(params, :r, LogNormal(0.0, 1.0))
                push!(priors_acc, "$(p_names.r) ~ $(_distribution_to_string(r_prior))")
            end
            if get(params, :spatially_varying_K, false)
                log_K_mean_prior = get(params, :log_K_mean, Normal(log(100.0), 0.5))
                sigma_K_prior = get(params, :sigma_K, Exponential(1.0))
                push!(priors_acc, "sigma_K_$(spec.key) ~ $(_distribution_to_string(sigma_K_prior))")
                push!(priors_acc, "log_K_mean_$(spec.key) ~ $(_distribution_to_string(log_K_mean_prior))")
                push!(priors_acc, "K_raw_$(spec.key) ~ MvNormal(zeros(M.s_N), I)")
            else
                K_prior = get(params, :K, LogNormal(log(100.0), 1.0))
                push!(priors_acc, "$(p_names.K) ~ $(_distribution_to_string(K_prior))")
            end
            for key in spec.hyper.effort_keys
                q_prior = get(params, Symbol("q_$(key)"), LogNormal(-2.0, 1.0))
                push!(priors_acc, "q_$(key) ~ $(_distribution_to_string(q_prior))")
            end
        end
        if m.model == "delay_difference"
            M_nat_prior = get(params, :M_nat, LogNormal(-1.0, 0.5))
            push!(priors_acc, "$(p_names.M_nat) ~ $(_distribution_to_string(M_nat_prior))")
        end
        
        if m.model == "lotka_volterra"
            alpha_prior = get(params, :alpha, LogNormal(0.0, 0.5))
            beta_prior = get(params, :beta, LogNormal(-1.0, 0.5))
            gamma_prior = get(params, :gamma, LogNormal(-1.0, 0.5))
            delta_prior = get(params, :delta, LogNormal(0.0, 0.5))
            push!(priors_acc, "$(p_names.alpha) ~ $(_distribution_to_string(alpha_prior))")
            push!(priors_acc, "$(p_names.beta) ~ $(_distribution_to_string(beta_prior))")
            push!(priors_acc, "$(p_names.gamma) ~ $(_distribution_to_string(gamma_prior))")
            push!(priors_acc, "$(p_names.delta) ~ $(_distribution_to_string(delta_prior))")
            push!(priors_acc, "$(p_names.innov)_predator ~ MvNormal(zeros(M.s_N * M.t_N), I)")
        end

        push!(priors_acc, "$(p_names.innov) ~ MvNormal(zeros(M.s_N * M.t_N), I)")
    
    elseif arch == "multivariate"
        key_str = string(spec.key)
        if m.model == "leslie_matrix"
            n_classes = get(params, :n_age_classes, M.outcomes_N)
            if get(params, :spatially_varying_rates, false)
                push!(priors_acc, "log_fecundity_mean_$(key_str) ~ filldist(Normal(0.0, 1.0), $(n_classes))")
                push!(priors_acc, "sigma_fecundity_$(key_str) ~ filldist(Exponential(1.0), $(n_classes))")
                push!(priors_acc, "fecundity_raw_$(key_str) ~ MvNormal(zeros(M.s_N * $(n_classes)), I)")
                push!(priors_acc, "logit_survival_mean_$(key_str) ~ filldist(Normal(1.5, 1.0), $(n_classes-1))")
                push!(priors_acc, "sigma_survival_$(key_str) ~ filldist(Exponential(1.0), $(n_classes-1))")
                push!(priors_acc, "survival_raw_$(key_str) ~ MvNormal(zeros(M.s_N * ($(n_classes)-1)), I)")
            else
                push!(priors_acc, "survival_rates_$(key_str) ~ filldist(Beta(9.0, 1.0), $(n_classes - 1))")
                push!(priors_acc, "fecundity_rates_$(key_str) ~ filldist(LogNormal(0.0, 1.0), $(n_classes))")
            end
            if get(params, :spatially_varying_K, false)
                push!(priors_acc, "sigma_K_$(key_str) ~ Exponential(1.0)")
                push!(priors_acc, "log_K_mean_$(key_str) ~ Normal(log(100.0), 0.5)")
                push!(priors_acc, "K_raw_$(key_str) ~ MvNormal(zeros(M.s_N), I)")
            else
                push!(priors_acc, "K_$(key_str) ~ LogNormal(log(100.0), 1.0)")
            end
            for key in spec.hyper.effort_keys
                q_prior = get(params, Symbol("q_$(key)"), filldist(LogNormal(-4.0, 1.0), n_classes))
                push!(priors_acc, "q_$(key) ~ $(_distribution_to_string(q_prior))")
            end
            push!(priors_acc, "sigma_process_$(key_str) ~ filldist(Exponential(1.0), $(n_classes))")
            push!(priors_acc, "innov_process_$(key_str) ~ MvNormal(zeros(M.s_N * M.t_N * $(n_classes)), I)")
        
        elseif m.model == "generalized_lotka_volterra"
            n_species = M.outcomes_N
            push!(priors_acc, "r_$(key_str) ~ filldist(LogNormal(0.0, 1.0), $(n_species))")
            push!(priors_acc, "alpha_raw_$(key_str) ~ MvNormal(zeros($(n_species * (n_species - 1))), I)")
            if get(params, :spatially_varying_K, false)
                push!(priors_acc, "log_K_mean_$(key_str) ~ filldist(Normal(log(100.0), 1.0), $(n_species))")
                push!(priors_acc, "sigma_K_$(key_str) ~ filldist(Exponential(1.0), $(n_species))")
                push!(priors_acc, "K_raw_$(key_str) ~ MvNormal(zeros(M.s_N * $(n_species)), I)")
            else
                push!(priors_acc, "K_$(key_str) ~ filldist(LogNormal(log(100.0), 1.0), $(n_species))")
            end
            push!(priors_acc, "sigma_process_$(key_str) ~ filldist(Exponential(1.0), $(n_species))")
            push!(priors_acc, "innov_process_$(key_str) ~ MvNormal(zeros(M.s_N * M.t_N * $(n_species)), I)")
        end
    end

    return join(priors_acc, "\n    ")
end

"""
    get_updates(m::Dynamics, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code string for simulating the `Dynamics`'s effect.
"""
function get_updates(
    m::Dynamics, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent" : "eta"
    key_str = string(spec.key)
    params = m.params

    if arch == "univariate"
        propagator_setup = ""
        if m.model in ["advection", "advection_diffusion"]
            propagator_setup *=
                "local propagator = lu(I(M.s_N) - $(p_names.velocity) * " *
                "spec_registry[:$(spec.key)].hyper.A_template + M.noise * I)\n"
        end
        if m.model in ["diffusion", "advection_diffusion"]
            propagator_setup *=
                "local propagator = cholesky(Symmetric(I(M.s_N) - " *
                "$(p_names.diffusion) * spec_registry[:$(spec.key)].hyper.L_template + M.noise * I))\n"
        end
        
        field_setup =
            "local T_num_dyn = eltype($(p_names.innov)); " *
            "local dyn_field = zeros(T_num_dyn, M.s_N, M.t_N)\n    " *
            "local innov_matrix = reshape($(p_names.innov), M.s_N, M.t_N)\n    " *
            "dyn_field[:, 1] = innov_matrix[:, 1]"
        
        K_setup_block = ""
        K_var = string(p_names.K)
        if get(params, :spatially_varying_K, false)
            K_setup_block =
                "local Q_K = spec_registry[:$(spec.key)].hyper.L_template\n" *
                "local F_K = cholesky(Symmetric(Matrix(Q_K) + M.noise * I))\n" *
                "local K_field_raw = F_K.L' \\ K_raw_$(spec.key)\n" *
                "Turing.@addlogprob! logpdf(Normal(0.0,0.001 * M.s_N), sum(K_field_raw))\n" *
                "local K_spatial = exp.(log_K_mean_$(spec.key) .+ K_field_raw .* sigma_K_$(spec.key))"
            K_var = "K_spatial"
        end

        r_setup_block = ""
        r_var = string(p_names.r)
        if get(params, :spatially_varying_r, false)
            r_setup_block =
                "local Q_r = spec_registry[:$(spec.key)].hyper.L_template\n" *
                "local F_r = cholesky(Symmetric(Matrix(Q_r) + M.noise * I))\n" *
                "local r_field_raw = F_r.L' \\ r_raw_$(spec.key)\n" *
                "Turing.@addlogprob! logpdf(Normal(0.0,0.001 * M.s_N), sum(r_field_raw))\n" *
                "local r_spatial = exp.(log_r_mean_$(spec.key) .+ r_field_raw .* sigma_r_$(spec.key))"
            r_var = "r_spatial"
        end
        
        propagator_logic = if m.model in ["advection", "diffusion", "advection_diffusion"]
            "dyn_field[:, t] = (propagator \\ N_intermediate) .+ innov_matrix[:, t]"
        else
            "dyn_field[:, t] = N_intermediate .+ innov_matrix[:, t]"
        end

        local evolution_loop_body
        if m.model == "logistic"
            exploitation_logic = _generate_exploitation_block(spec, "t")
            evolution_loop_body =
                "local areas = spec_registry[:$(spec.key)].hyper.areas\n" *
                "for t in 2:M.t_N\n    " *
                "local N_prev = dyn_field[:, t-1]; local D_prev = N_prev ./ areas; " *
                "local K_density = $(K_var) ./ areas; " *
                "local growth = $(r_var) .* D_prev .* (1.0 .- D_prev ./ K_density); " *
                "$(exploitation_logic); " *
                "local N_intermediate = N_prev .+ (growth .* areas) .- exploitation; " *
                "$(propagator_logic); " *
                "dyn_field[:, t] = max.(T_num_dyn(0.0), dyn_field[:, t]);\n" *
                "end"
        elseif m.model == "delay_difference"
            exploitation_logic = _generate_exploitation_block(spec, "t-1")
            evolution_loop_body =
                "local areas = spec_registry[:$(spec.key)].hyper.areas\n" *
                "for t in 2:M.t_N\n    " *
                "local N_prev = dyn_field[:, t-1]; local D_prev = N_prev ./ areas; " *
                "local K_density = $(K_var) ./ areas; " *
                "local growth = $(r_var) .* D_prev .* (1.0 .- D_prev ./ K_density); " *
                "$(exploitation_logic); " *
                "local N_survived = (N_prev .- exploitation) .* exp.(-$(p_names.M_nat)); " *
                "local N_intermediate = N_survived .+ (growth .* areas); " *
                "$(propagator_logic); " *
                "dyn_field[:, t] = max.(T_num_dyn(0.0), dyn_field[:, t]);\n" *
                "end"
        elseif m.model == "lotka_volterra"
            output_species = get(params, :output_species, :prey)
            field_setup =
                "local T_num_dyn = eltype($(p_names.innov)); " *
                "local dyn_field_prey = zeros(T_num_dyn, M.s_N, M.t_N); " *
                "local dyn_field_predator = zeros(T_num_dyn, M.s_N, M.t_N); " *
                "local innov_matrix_prey = reshape($(p_names.innov), M.s_N, M.t_N); " *
                "local innov_matrix_predator = reshape($(p_names.innov)_predator, M.s_N, M.t_N); " *
                "dyn_field_prey[:, 1] = innov_matrix_prey[:, 1]; " *
                "dyn_field_predator[:, 1] = innov_matrix_predator[:, 1]"
            evolution_loop_body =
                "for t in 2:M.t_N\n    " *
                "local N_prey_prev = dyn_field_prey[:, t-1]; " *
                "local N_pred_prev = dyn_field_predator[:, t-1]; " *
                "local d_prey = ($(p_names.alpha) .* N_prey_prev) .- ($(p_names.beta) .* N_prey_prev .* N_pred_prev); " *
                "local d_pred = ($(p_names.gamma) .* N_prey_prev .* N_pred_prev) .- ($(p_names.delta) .* N_pred_prev); " *
                "dyn_field_prey[:, t] = max.(T_num_dyn(0.0), N_prey_prev .+ d_prey .+ innov_matrix_prey[:, t]); " *
                "dyn_field_predator[:, t] = max.(T_num_dyn(0.0), N_pred_prev .+ d_pred .+ innov_matrix_predator[:, t]);\n" *
                "end\n" *
                "local dyn_field = $(output_species == :prey ? "dyn_field_prey" : "dyn_field_predator")"
        else
            evolution_loop_body =
                "for t in 2:M.t_N\n    " *
                "dyn_field[:, t] = (propagator \\ dyn_field[:, t-1]) + innov_matrix[:, t];\n" *
                "end"
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
                $(eta_target)[i] += log(dyn_field[M.s_idx[i], M.t_idx[i]] + 1.0e-6)
            end
        end
        """
    elseif arch == "multivariate"
        if m.model == "leslie_matrix"
            n_classes = get(params, :n_age_classes, M.outcomes_N)
            spatially_varying_K = get(params, :spatially_varying_K, false)
            spatially_varying_rates = get(params, :spatially_varying_rates, false)
            exploitation_block = _generate_exploitation_block(spec, "t-1")
            return """
            begin
                local Q_spatial = spec_registry[:$(key_str)].hyper.L_template;
                local F_spatial = cholesky(Symmetric(Matrix(Q_spatial) + M.noise * I));
                local areas = spec_registry[:$(key_str)].hyper.areas
                local survival_rates_spatial, fecundity_rates_spatial;
                if $(spatially_varying_rates)
                    fecundity_raw_matrix = reshape(fecundity_raw_$(key_str), M.s_N, $(n_classes));
                    fecundity_field = F_spatial.L' \\ fecundity_raw_matrix;
                    fecundity_rates_spatial = exp.(log_fecundity_mean_$(key_str)' .+ fecundity_field .* sigma_fecundity_$(key_str)');
                    survival_raw_matrix = reshape(survival_raw_$(key_str), M.s_N, $(n_classes-1));
                    survival_field = F_spatial.L' \\ survival_raw_matrix;
                    survival_rates_spatial = logistic.(logit_survival_mean_$(key_str)' .+ survival_field .* sigma_survival_$(key_str)');
                end
                local K_values_$(key_str);
                if $(spatially_varying_K)
                    K_field_raw = F_spatial.L' \\ K_raw_$(key_str);
                    Turing.@addlogprob! logpdf(Normal(0.0,0.001 * M.s_N), sum(K_field_raw));
                    K_values_$(key_str) = exp.(log_K_mean_$(key_str) .+ K_field_raw .* sigma_K_$(key_str));
                else
                    K_values_$(key_str) = fill(K_$(key_str), M.s_N);
                end
                local innov_tensor_$(key_str) = reshape(innov_process_$(key_str), M.s_N, M.t_N, $(n_classes));
                local population_field_$(key_str) = zeros(T, M.s_N, M.t_N, $(n_classes))
                for a in 1:$(n_classes)
                    population_field_$(key_str)[:, 1, a] = max.(0.0, innov_tensor_$(key_str)[:, 1, a] .* sigma_process_$(key_str)[a]);
                end
                for s in 1:M.s_N
                    local L_s = zeros(T, $(n_classes), $(n_classes));
                    if $(spatially_varying_rates)
                        for i in 1:($(n_classes)-1); L_s[i+1, i] = survival_rates_spatial[s, i]; end;
                        L_s[1, :] = fecundity_rates_spatial[s, :];
                    else
                        for i in 1:($(n_classes)-1); L_s[i+1, i] = survival_rates_$(key_str)[i]; end;
                        L_s[1, :] = fecundity_rates_$(key_str);
                    end
                    for t in 2:M.t_N
                        local N_prev = view(population_field_$(key_str), s, t-1, :);
                        $(exploitation_block);
                        local N_after_removal = max.(0.0, N_prev - C_prev);
                        local L_effective = copy(L_s)
                        if $(spatially_varying_K) || haskey(params, :K)
                            local total_pop_prev = sum(N_after_removal);
                            local K_density = K_values_$(key_str)[s] / areas[s];
                            local dd_factor = max(0.0, 1.0 - (total_pop_prev / areas[s]) / K_density);
                            L_effective[1, :] .*= dd_factor;
                        end
                        local N_projected = L_effective * N_after_removal;
                        local current_innov = view(innov_tensor_$(key_str), s, t, :) .* sigma_process_$(key_str);
                        population_field_$(key_str)[s, t, :] = max.(0.0, N_projected .+ current_innov)
                    end
                end
                for k in 1:$(n_classes)
                    for i in 1:M.y_N
                        $(eta_target)[i, k] += log(population_field_$(key_str)[M.s_idx[i], M.t_idx[i], k] + 1.0e-6);
                    end
                end
            end
            """
        elseif m.model == "generalized_lotka_volterra"
            n_species = M.outcomes_N
            spatially_varying_K = get(params, :spatially_varying_K, false)
            return """
            begin
                local areas = spec_registry[:$(key_str)].hyper.areas;
                local alpha_$(key_str) = diagm(0 => ones(T, $(n_species)));
                local off_diag_indices = [i for i in 1:($(n_species)^2) if mod(i-1, $(n_species)+1) != 0];
                alpha_$(key_str)[off_diag_indices] = alpha_raw_$(key_str)
                local K_values_$(key_str);
                if $(spatially_varying_K)
                    local Q_spatial = spec_registry[:$(key_str)].hyper.L_template;
                    local F_spatial = cholesky(Symmetric(Matrix(Q_spatial) + M.noise * I));
                    local K_raw_matrix = reshape(K_raw_$(key_str), M.s_N, $(n_species));
                    local K_field = F_spatial.L' \\ K_raw_matrix;
                    K_values_$(key_str) = exp.(log_K_mean_$(key_str)' .+ K_field .* sigma_K_$(key_str)');
                else
                    K_values_$(key_str) = repeat(K_$(key_str)', M.s_N, 1);
                end
                local innov_tensor = reshape(innov_process_$(key_str), M.s_N, M.t_N, $(n_species));
                local population_field = zeros(T, M.s_N, M.t_N, $(n_species));
                population_field[:, 1, :] = max.(0.0, innov_tensor[:, 1, :] .* sigma_process_$(key_str)')
                for s in 1:M.s_N, t in 2:M.t_N
                    local N_prev = view(population_field, s, t-1, :);
                    local D_prev = N_prev ./ areas[s];
                    local K_density = K_values_$(key_str)[s, :] ./ areas[s];
                    local N_intermediate = zeros(T, $(n_species))
                    for i in 1:$(n_species)
                        local interaction_sum_density = dot(alpha_$(key_str)[i, :], D_prev);
                        local growth_density = r_$(key_str)[i] * D_prev[i] * (1.0 - interaction_sum_density / K_density[i]);
                        N_intermediate[i] = N_prev[i] + growth_density * areas[s];
                    end
                    local current_innov = view(innov_tensor, s, t, :) .* sigma_process_$(key_str);
                    population_field[s, t, :] = max.(0.0, N_intermediate .+ current_innov)
                end
                for k in 1:$(n_species)
                    for i in 1:M.y_N
                        $(eta_target)[i, k] += log(population_field[M.s_idx[i], M.t_idx[i], k] + 1.0e-6);
                    end
                end
            end
            """
        end
    end
    return "# Dynamics model '$(m.model)' not implemented for this architecture."
end

"""
    get_effects(m::Dynamics, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `Dynamics`'s effect from the MCMC chain's posterior samples.
"""
function get_effects(
    m::Dynamics, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing},
    N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    key_str = string(spec.key)
    model_type = m.model
    params = m.params

    s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)
    t_idx_full = isnothing(PS) ? M.t_idx : vcat(M.t_idx, PS.t_idx)
    t_N_full = maximum(t_idx_full) # Max time index across train and pred

    L_op = spec.hyper.L_template
    A_op = spec.hyper.A_template
    areas = spec.hyper.areas
    noise = M.noise

    for k_outcome in 1:outcomes_N
        # Generate outcome-specific parameter names
        p_names_k = generate_full_variable_names(spec, M.model_arch, k_outcome)

        if model_type in ["advection", "diffusion", "advection_diffusion"]
            sigma_samples = get(chain, p_names_k.sigma)
            innov_samples = get(chain, p_names_k.innov)
            
            rate_samples = if model_type == "advection"
                get(chain, p_names_k.velocity)
            elseif model_type == "diffusion"
                get(chain, p_names_k.diffusion)
            elseif model_type == "advection_diffusion"
                v_samples = get(chain, p_names_k.velocity)
                d_samples = get(chain, p_names_k.diffusion)
                hcat(v_samples, d_samples) # Combine for easier iteration
            else
                nothing
            end

            if isnothing(rate_samples) || isnothing(sigma_samples) || isnothing(innov_samples)
                @warn "Parameters for Dynamics manifold $(key_str) (model: $(model_type), outcome $(k_outcome)) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end

            effect_k = zeros(Float64, N_total, n_samples)
            I_s = Matrix(I, M.s_N, M.s_N)

            for j in 1:n_samples
                current_sigma = sigma_samples[j]
                current_innov_flat = innov_samples[j, :]
                
                # Expand innov_matrix to t_N_full if prediction is beyond training data
                innov_matrix = reshape(current_innov_flat, M.s_N, M.t_N)
                if t_N_full > M.t_N
                    innov_matrix = hcat(innov_matrix, randn(M.s_N, t_N_full - M.t_N))
                end

                dyn_field = zeros(Float64, M.s_N, t_N_full)
                
                local propagator
                if model_type == "advection"
                    current_velocity = rate_samples[j]
                    propagator = lu(I_s - current_velocity * A_op + noise * I_s)
                elseif model_type == "diffusion"
                    current_diffusion = rate_samples[j]
                    propagator = cholesky(Symmetric(I_s - current_diffusion * L_op + noise * I_s))
                elseif model_type == "advection_diffusion"
                    current_velocity = rate_samples[j, 1]
                    current_diffusion = rate_samples[j, 2]
                    propagator = lu(I_s - current_velocity * A_op - current_diffusion * L_op + noise * I_s)
                end

                dyn_field[:, 1] = innov_matrix[:, 1]
                for t in 2:t_N_full
                    dyn_field[:, t] = (propagator \ dyn_field[:, t-1]) + innov_matrix[:, t]
                end
                
                dyn_field .*= current_sigma
                log_dyn_field = log.(dyn_field .+ 1e-6)

                for obs_idx in 1:N_total
                    effect_k[obs_idx, j] = log_dyn_field[s_idx_full[obs_idx], t_idx_full[obs_idx]]
                end
            end
            push!(structured_effects, effect_k)

        elseif model_type == "logistic"
            sigma_samples = get(chain, p_names_k.sigma)
            innov_samples = get(chain, p_names_k.innov)
            r_samples = get(chain, p_names_k.r)
            K_samples = get(chain, p_names_k.K)
            
            effect_k = zeros(Float64, N_total, n_samples)

            for j in 1:n_samples
                current_sigma = sigma_samples[j]
                current_innov_flat = innov_samples[j, :]
                current_r = r_samples[j]
                current_K = K_samples[j]

                innov_matrix = reshape(current_innov_flat, M.s_N, M.t_N)
                if t_N_full > M.t_N
                    innov_matrix = hcat(innov_matrix, randn(M.s_N, t_N_full - M.t_N))
                end

                dyn_field = zeros(Float64, M.s_N, t_N_full)
                dyn_field[:, 1] = innov_matrix[:, 1] # Initial state

                for t in 2:t_N_full
                    N_prev = dyn_field[:, t-1]
                    D_prev = N_prev ./ areas
                    K_density = current_K ./ areas
                    growth = current_r .* D_prev .* (1.0 .- D_prev ./ K_density)
                    N_intermediate = N_prev .+ (growth .* areas)
                    dyn_field[:, t] = max.(0.0, N_intermediate .+ innov_matrix[:, t])
                end
                
                dyn_field .*= current_sigma
                log_dyn_field = log.(dyn_field .+ 1e-6)
                
                for obs_idx in 1:N_total
                    effect_k[obs_idx, j] = log_dyn_field[s_idx_full[obs_idx], t_idx_full[obs_idx]]
                end
            end
            push!(structured_effects, effect_k)

        elseif model_type == "delay_difference"
            sigma_samples = get(chain, p_names_k.sigma)
            innov_samples = get(chain, p_names_k.innov)
            r_samples = get(chain, p_names_k.r)
            K_samples = get(chain, p_names_k.K)
            M_nat_samples = get(chain, p_names_k.M_nat)
            
            effort_keys = spec.hyper.effort_keys
            q_samples_dict = Dict(key => get(chain, Symbol("q_$(key)")) for key in effort_keys)
            removal_keys = spec.hyper.removal_keys

            effect_k = zeros(Float64, N_total, n_samples)

            for j in 1:n_samples
                current_sigma = sigma_samples[j]
                current_innov_flat = innov_samples[j, :]
                current_r = r_samples[j]
                current_K = K_samples[j]
                current_M_nat = M_nat_samples[j]

                innov_matrix = reshape(current_innov_flat, M.s_N, M.t_N)
                if t_N_full > M.t_N
                    innov_matrix = hcat(innov_matrix, randn(M.s_N, t_N_full - M.t_N))
                end

                dyn_field = zeros(Float64, M.s_N, t_N_full)
                dyn_field[:, 1] = innov_matrix[:, 1] # Initial state

                for t in 2:t_N_full
                    N_prev = dyn_field[:, t-1]
                    D_prev = N_prev ./ areas
                    K_density = current_K ./ areas
                    growth = current_r .* D_prev .* (1.0 .- D_prev ./ K_density)
                    
                    C_prev = zeros(M.s_N)
                    for e_key in effort_keys
                        C_prev .+= q_samples_dict[e_key][j] .* spec.hyper.processed_params[e_key][:, t-1] .* N_prev
                    end
                    for r_key in removal_keys
                        C_prev .+= spec.hyper.processed_params[r_key][:, t-1]
                    end

                    N_survived = (N_prev .- C_prev) .* exp.(-current_M_nat)
                    N_intermediate = N_survived .+ (growth .* areas)
                    dyn_field[:, t] = max.(0.0, N_intermediate .+ innov_matrix[:, t])
                end
                
                dyn_field .*= current_sigma
                log_dyn_field = log.(dyn_field .+ 1e-6)
                
                for obs_idx in 1:N_total
                    effect_k[obs_idx, j] = log_dyn_field[s_idx_full[obs_idx], t_idx_full[obs_idx]]
                end
            end
            push!(structured_effects, effect_k)

        elseif model_type == "lotka_volterra"
            alpha_samples = get(chain, p_names_k.alpha)
            beta_samples = get(chain, p_names_k.beta)
            gamma_samples = get(chain, p_names_k.gamma)
            delta_samples = get(chain, p_names_k.delta)
            sigma_samples = get(chain, p_names_k.sigma)
            innov_prey_samples = get(chain, p_names_k.innov)
            innov_predator_samples = get(chain, Symbol(string(p_names_k.innov, "_predator")))
            
            output_species = get(params, :output_species, :prey)
            
            effect_k = zeros(Float64, N_total, n_samples)

            for j in 1:n_samples
                current_alpha = alpha_samples[j]
                current_beta = beta_samples[j]
                current_gamma = gamma_samples[j]
                current_delta = delta_samples[j]
                current_sigma = sigma_samples[j]

                innov_matrix_prey = reshape(innov_prey_samples[j, :], M.s_N, M.t_N)
                innov_matrix_predator = reshape(innov_predator_samples[j, :], M.s_N, M.t_N)
                if t_N_full > M.t_N
                    innov_matrix_prey = hcat(innov_matrix_prey, randn(M.s_N, t_N_full - M.t_N))
                    innov_matrix_predator = hcat(innov_matrix_predator, randn(M.s_N, t_N_full - M.t_N))
                end

                dyn_field_prey = zeros(Float64, M.s_N, t_N_full)
                dyn_field_predator = zeros(Float64, M.s_N, t_N_full)
                
                dyn_field_prey[:, 1] = innov_matrix_prey[:, 1]
                dyn_field_predator[:, 1] = innov_matrix_predator[:, 1]

                for t in 2:t_N_full
                    N_prey_prev = dyn_field_prey[:, t-1]
                    N_pred_prev = dyn_field_predator[:, t-1]
                    
                    d_prey = (current_alpha .* N_prey_prev) .- (current_beta .* N_prey_prev .* N_pred_prev)
                    d_pred = (current_gamma .* N_prey_prev .* N_pred_prev) .- (current_delta .* N_pred_prev)
                    
                    dyn_field_prey[:, t] = max.(0.0, N_prey_prev .+ d_prey .+ innov_matrix_prey[:, t])
                    dyn_field_predator[:, t] = max.(0.0, N_pred_prev .+ d_pred .+ innov_matrix_predator[:, t])
                end
                
                final_dyn_field = if output_species == :prey
                    dyn_field_prey
                else
                    dyn_field_predator
                end
                final_dyn_field .*= current_sigma
                log_final_dyn_field = log.(final_dyn_field .+ 1e-6)

                for obs_idx in 1:N_total
                    effect_k[obs_idx, j] = log_final_dyn_field[s_idx_full[obs_idx], t_idx_full[obs_idx]]
                end
            end
            push!(structured_effects, effect_k)

        elseif model_type == "leslie_matrix"
            n_classes = get(params, :n_age_classes, outcomes_N)
            spatially_varying_K = get(params, :spatially_varying_K, false)
            spatially_varying_rates = get(params, :spatially_varying_rates, false)

            sigma_process_samples = get(chain, Symbol("sigma_process_$(key_str)"))
            innov_process_samples = get(chain, Symbol("innov_process_$(key_str)"))

            K_samples = if spatially_varying_K
                get(chain, Symbol("K_raw_$(key_str)"))
            else
                get(chain, Symbol("K_$(key_str)"))
            end
            log_K_mean_samples = if spatially_varying_K
                get(chain, Symbol("log_K_mean_$(key_str)"))
            else
                nothing
            end
            sigma_K_samples = if spatially_varying_K
                get(chain, Symbol("sigma_K_$(key_str)"))
            else
                nothing
            end

            survival_rates_samples = if spatially_varying_rates
                get(chain, Symbol("survival_raw_$(key_str)"))
            else
                get(chain, Symbol("survival_rates_$(key_str)"))
            end
            fecundity_rates_samples = if spatially_varying_rates
                get(chain, Symbol("fecundity_raw_$(key_str)"))
            else
                get(chain, Symbol("fecundity_rates_$(key_str)"))
            end
            log_fecundity_mean_samples = if spatially_varying_rates
                get(chain, Symbol("log_fecundity_mean_$(key_str)"))
            else
                nothing
            end
            sigma_fecundity_samples = if spatially_varying_rates
                get(chain, Symbol("sigma_fecundity_$(key_str)"))
            else
                nothing
            end
            logit_survival_mean_samples = if spatially_varying_rates
                get(chain, Symbol("logit_survival_mean_$(key_str)"))
            else
                nothing
            end
            sigma_survival_samples = if spatially_varying_rates
                get(chain, Symbol("sigma_survival_$(key_str)"))
            else
                nothing
            end

            effort_keys = spec.hyper.effort_keys
            q_samples_dict = Dict(key => get(chain, Symbol("q_$(key)")) for key in effort_keys)
            removal_keys = spec.hyper.removal_keys

            age_class_effects = [zeros(Float64, N_total, n_samples) for _ in 1:n_classes]
            I_s = Matrix(I, M.s_N, M.s_N)
            F_spatial = cholesky(Symmetric(Matrix(L_op) + noise * I_s))

            for j in 1:n_samples
                current_sigma_process = sigma_process_samples[j, :]
                current_innov_tensor_flat = innov_process_samples[j, :]
                
                innov_tensor = reshape(current_innov_tensor_flat, M.s_N, M.t_N, n_classes)
                if t_N_full > M.t_N
                    innov_tensor = cat(innov_tensor, randn(M.s_N, t_N_full - M.t_N, n_classes), dims=2)
                end

                local K_values_j
                if spatially_varying_K
                    K_raw_j = get_params_vector(chain, Symbol("K_raw_$(key_str)"), M.s_N)[j, :]
                    log_K_mean_j = log_K_mean_samples[j]
                    sigma_K_j = sigma_K_samples[j]
                    K_field_raw = F_spatial.L' \ K_raw_j
                    K_values_j = exp.(log_K_mean_j .+ K_field_raw .* sigma_K_j)
                else
                    K_values_j = fill(K_samples[j], M.s_N)
                end

                population_field_j = zeros(Float64, M.s_N, t_N_full, n_classes)
                for a in 1:n_classes
                    population_field_j[:, 1, a] = max.(0.0, innov_tensor[:, 1, a] .* current_sigma_process[a])
                end

                for s in 1:M.s_N
                    L_s = zeros(Float64, n_classes, n_classes)
                    if spatially_varying_rates
                        fecundity_raw_matrix_j = reshape(get_params_vector(chain, Symbol("fecundity_raw_$(key_str)"), M.s_N * n_classes)[j, :], M.s_N, n_classes)
                        fecundity_field = F_spatial.L' \ fecundity_raw_matrix_j
                        fecundity_rates_spatial = exp.(log_fecundity_mean_samples[j, :]' .+ fecundity_field .* sigma_fecundity_samples[j, :]')
                        
                        survival_raw_matrix_j = reshape(get_params_vector(chain, Symbol("survival_raw_$(key_str)"), M.s_N * (n_classes - 1))[j, :], M.s_N, n_classes - 1)
                        survival_field = F_spatial.L' \ survival_raw_matrix_j
                        survival_rates_spatial = logistic.(logit_survival_mean_samples[j, :]' .+ survival_field .* sigma_survival_samples[j, :]')

                        for i_age in 1:(n_classes-1); L_s[i_age+1, i_age] = survival_rates_spatial[s, i_age]; end
                        L_s[1, :] = fecundity_rates_spatial[s, :]
                    else
                        current_survival_rates = survival_rates_samples[j, :]
                        current_fecundity_rates = fecundity_rates_samples[j, :]
                        for i_age in 1:(n_classes-1); L_s[i_age+1, i_age] = current_survival_rates[i_age]; end
                        L_s[1, :] = current_fecundity_rates
                    end

                    for t in 2:t_N_full
                        N_prev = view(population_field_j, s, t-1, :)
                        C_prev = zeros(n_classes)
                        for e_key in effort_keys
                            C_prev .+= q_samples_dict[e_key][j, :] .* spec.hyper.processed_params[e_key][s, t-1] .* N_prev
                        end
                        for r_key in removal_keys
                            C_prev .+= spec.hyper.processed_params[r_key][s, t-1, :]
                        end
                        N_after_removal = max.(0.0, N_prev - C_prev)
                        
                        L_effective = copy(L_s)
                        if spatially_varying_K || haskey(params, :K)
                            total_pop_prev = sum(N_after_removal)
                            K_density = K_values_j[s] / areas[s]
                            dd_factor = max(0.0, 1.0 - (total_pop_prev / areas[s]) / K_density)
                            L_effective[1, :] .*= dd_factor
                        end
                        N_projected = L_effective * N_after_removal
                        current_innov = view(innov_tensor, s, t, :) .* current_sigma_process
                        population_field_j[s, t, :] = max.(0.0, N_projected .+ current_innov)
                    end
                end
                
                for a in 1:n_classes
                    log_pop_field = log.(view(population_field_j, :, :, a) .+ 1e-6)
                    for obs_idx in 1:N_total
                        age_class_effects[a][obs_idx, j] = log_pop_field[s_idx_full[obs_idx], t_idx_full[obs_idx]]
                    end
                end
            end
            structured_effects = age_class_effects

        elseif model_type == "generalized_lotka_volterra"
            n_species = M.outcomes_N
            spatially_varying_K = get(params, :spatially_varying_K, false)

            r_samples = get(chain, Symbol("r_$(key_str)"))
            alpha_raw_samples = get(chain, Symbol("alpha_raw_$(key_str)"))
            sigma_process_samples = get(chain, Symbol("sigma_process_$(key_str)"))
            innov_process_samples = get(chain, Symbol("innov_process_$(key_str)"))

            K_samples = if spatially_varying_K
                get(chain, Symbol("K_raw_$(key_str)"))
            else
                get(chain, Symbol("K_$(key_str)"))
            end
            log_K_mean_samples = if spatially_varying_K
                get(chain, Symbol("log_K_mean_$(key_str)"))
            else
                nothing
            end
            sigma_K_samples = if spatially_varying_K
                get(chain, Symbol("sigma_K_$(key_str)"))
            else
                nothing
            end

            species_effects = [zeros(Float64, N_total, n_samples) for _ in 1:n_species]
            I_s = Matrix(I, M.s_N, M.s_N)
            F_spatial = cholesky(Symmetric(Matrix(L_op) + noise * I_s))

            for j in 1:n_samples
                alpha_j = diagm(0 => ones(n_species))
                off_diag_indices = [i for i in 1:(n_species^2) if mod(i-1, n_species+1) != 0]
                alpha_j[off_diag_indices] = alpha_raw_samples[j, :]

                local K_values_j
                if spatially_varying_K
                    K_raw_matrix_j = reshape(get_params_vector(chain, Symbol("K_raw_$(key_str)"), M.s_N * n_species)[j, :], M.s_N, n_species)
                    K_field = F_spatial.L' \ K_raw_matrix_j
                    K_values_j = exp.(log_K_mean_samples[j, :]' .+ K_field .* sigma_K_samples[j, :]')
                else
                    K_values_j = repeat(K_samples[j, :]', M.s_N, 1)
                end

                innov_tensor_j = reshape(innov_process_samples[j, :], M.s_N, M.t_N, n_species)
                if t_N_full > M.t_N
                    innov_tensor_j = cat(innov_tensor_j, randn(M.s_N, t_N_full - M.t_N, n_species), dims=2)
                end
                current_sigma_process = sigma_process_samples[j, :]

                population_field_j = zeros(Float64, M.s_N, t_N_full, n_species)
                population_field_j[:, 1, :] = max.(0.0, innov_tensor_j[:, 1, :] .* current_sigma_process')

                for s in 1:M.s_N, t in 2:t_N_full
                    N_prev = view(population_field_j, s, t-1, :)
                    D_prev = N_prev ./ areas[s]
                    K_density = K_values_j[s, :] ./ areas[s]
                    
                    N_intermediate = zeros(Float64, n_species)
                    for i_species in 1:n_species
                        interaction_sum_density = dot(alpha_j[i_species, :], D_prev)
                        growth_density = r_samples[j, i_species] * D_prev[i_species] * (1.0 - interaction_sum_density / K_density[i_species])
                        N_intermediate[i_species] = N_prev[i_species] + growth_density * areas[s]
                    end
                    
                    current_innov = view(innov_tensor_j, s, t, :) .* current_sigma_process
                    population_field_j[s, t, :] = max.(0.0, N_intermediate .+ current_innov)
                end

                for i_species in 1:n_species
                    log_pop_field = log.(view(population_field_j, :, :, i_species) .+ 1e-6)
                    for obs_idx in 1:N_total
                        species_effects[i_species][obs_idx, j] = log_pop_field[s_idx_full[obs_idx], t_idx_full[obs_idx]]
                    end
                end
            end
            structured_effects = species_effects
        end
    end

    if isempty(structured_effects)
        @warn "Reconstruction for Dynamics model '$(model_type)' is not implemented or parameters not found. Returning zero effects."
        structured_effects = [zeros(Float64, N_total, n_samples) for _ in 1:outcomes_N]
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
