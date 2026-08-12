"""
    SciML <: ComponentModel

A component for embedding user-defined differential equation models from the SciML
ecosystem into a Bayesian framework. It allows for the estimation of differential
equation parameters and initial conditions.

# Version
v1.1.0 (2026-08-11)

# Mathematical Summary
This component models an observed process \$y(t)\$ as noisy observations of a latent
state vector \$\\mathbf{u}(t)\$, where \$\\mathbf{u}(t)\$ is the solution to a system of
differential equations:
\$ \\frac{d\\mathbf{u}}{dt} = f(\\mathbf{u}, \\mathbf{p}, t) \$
where \$\\mathbf{p}\$ is a vector of parameters and \$\\mathbf{u}(t_0) = \\mathbf{u}_0\$ are the
initial conditions. The component estimates \$\\mathbf{u}_0\$ and \$\\mathbf{p}\$ by fitting
the model to observed data.

# Likelihood Types
- `:additive` (Default): The solution of the DE is treated as an additive component
  in the model's linear predictor, \$\\eta = \\dots + \\mathbf{u}(t)\$. This is for
  when the DE models one of several influential processes.
- `:direct`: The solution of the DE is assumed to be the mean of the observation
  model directly, \$\\mu = \\mathbf{u}(t)\$. This is for when the DE is the complete
  generative model for the data. In this mode, the component handles its own
  likelihood evaluation, and the main model's likelihood is ignored.

# Inputs
- **Required**:
  - A temporal index variable (e.g., `year`) passed to `sciml()`.
  - `model_func`: `Symbol`, the name of the user-defined function specifying the DE system.
  - `u0_prior`: A `Distribution` for the prior on the initial conditions `u0`.
  - `p_priors`: A `NamedTuple` of priors for the DE parameters (e.g., `(alpha=Normal(0,1), beta=LogNormal(0,1))`).
  - `tspan`: A `Tuple` specifying the integration time span (e.g., `(0.0, 10.0)`).
  - `solver`: A `SciML` solver object (e.g., `Tsit5()`).
- **Optional (in `sciml()` call)**:
  - `de_type`: `Symbol`, the type of differential equation (`:ODE`, `:SDE`, `:DDE`, `:Jump`). Default: `:ODE`.
  - `likelihood_type`: `Symbol`, the likelihood evaluation method (`:additive` or `:direct`). Default: `:additive`.
  - `de_kwargs`: A `Dict` of additional keyword arguments for the `DEProblem` constructor (e.g., `constant_lags` for DDEs).
  - `saveat`: `Float64`, the time step for saving the DE solution. Default: `0.1`.

# Outputs (Parameter Names)
- `u0_<key>`: The initial conditions of the DE system.
- `p_<param_name>_<key>`: The parameters of the DE system (e.g., `p_alpha_<key>`).
- `latent_<key>`: The reconstructed latent effect from the DE solution.

# Key References
- Rackauckas, C., & Nie, Q. (2017). *DifferentialEquations.jl – A Performant and
  Feature-Rich Ecosystem for Solving Differential Equations in Julia*. Journal of
  Open Research Software, 5(1).
- SciML Documentation: https://sciml.ai/
"""
struct SciML <: ComponentModel
    model_func::Symbol
    u0_prior::Any
    p_priors::NamedTuple
    de_type::Symbol
    de_kwargs::Dict{Symbol, Any}
    likelihood_type::Symbol
end

COMPONENT_TYPE_REGISTRY[:sciml] = SciML

COMPONENT_CONSTRUCTORS[:sciml] = (p, params) -> begin
    model_func = get(params, :model_func, error("SciML requires a `model_func` parameter."))
    u0_prior = get(params, :u0_prior, error("SciML requires a `u0_prior` parameter."))
    p_priors = get(params, :p_priors, error("SciML requires a `p_priors` parameter."))
    de_type = get(params, :de_type, :ODE)
    likelihood_type = get(params, :likelihood_type, :additive)
    
    de_kwargs = Dict{Symbol, Any}()
    for key in [:constant_lags, :saveat, :h, :noise_func]
        if haskey(params, key); de_kwargs[key] = params[key]; end
    end

    SciML(model_func, u0_prior, p_priors, de_type, de_kwargs, likelihood_type)
end

MODEL_TO_STRUCTURE_MAP[:sciml] = :temporal

function get_datastructures!(m_type::Type{<:SciML}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    variables = mod_data[:variables]

    if isempty(variables)
        error("The `sciml()` module requires a time index variable, e.g., `sciml(year, ...)`.")
    end

    time_var_sym = Symbol(variables[1])
    if !hasproperty(M[:data], time_var_sym)
        error("Time index variable ':$time_var_sym' for sciml() module not found in data.")
    end

    time_opts = Dict(:time_method => get(params, :time_method, "continuous"))
    tu_meta = assign_time_units(M[:data][!, time_var_sym]; time_opts...)
    M[:t_idx] = tu_meta.idx
    M[:t_N] = tu_meta.N_cat
    M[:t_idx_var] = time_var_sym
    M[:t_coords] = M[:data][!, time_var_sym]
    mod_data[:params][:coords] = M[:data][!, time_var_sym]

    required_args = [:model_func, :u0_prior, :p_priors, :tspan, :solver]
    for arg in required_args
        if !haskey(params, arg)
            error("The `sciml()` module is missing the required keyword argument `:$arg`.")
        end
    end

    if !haskey(M, :sciml_problem_templates)
        M[:sciml_problem_templates] = Dict{Symbol, Any}()
    end
    
    calling_mod = get(M, :calling_module, Main)
    model_func = Core.eval(calling_mod, params[:model_func])
    
    u0_template = mean(params[:u0_prior])
    p_template = Tuple(mean(p) for p in params[:p_priors])
    tspan = params[:tspan]
    de_type = get(params, :de_type, :ODE)
    de_kwargs = get(params, :de_kwargs, Dict())

    local prob_template
    if de_type == :ODE
        prob_template = ODEProblem(model_func, u0_template, tspan, p_template)
    elseif de_type == :SDE
        noise_func = get(params, :noise_func, error("SDE requires a `noise_func`."))
        prob_template = SDEProblem(model_func, noise_func, u0_template, tspan, p_template)
    elseif de_type == :DDE
        h_func = get(params, :h, error("DDE requires a history function `h`."))
        prob_template = DDEProblem(model_func, u0_template, h_func, tspan, p_template; de_kwargs...)
    elseif de_type == :Jump
        prob_template = model_func(u0_template, p_template, tspan)
    else
        error("Unsupported `de_type`: $de_type")
    end

    M[:sciml_problem_templates][mod_data[:key]] = prob_template
    M[:sciml_solver] = params[:solver]
    M[:sciml_saveat] = get(params, :saveat, 0.1)

    return true
end

function get_precomputes(m::SciML, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("SciML component precomputes failed: coordinates not found.")
    end
    
    return (
        coords=coords,
        n_latent=M.y_N,
        param_names=keys(m.p_priors)
    )
end

function get_priors(
    m::SciML, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    key = string(spec.key)
    
    u0_name = "u0_$(key)"
    if arch == "multivariate" && !get(spec.params, :shared, false)
        u0_name *= "_$(outcome_idx)"
    end
    
    prior_lines = ["# --- Priors for SciML component: $(spec.key) ---"]
    push!(prior_lines, "$(u0_name) ~ $(_distribution_to_string(m.u0_prior))")
    
    for p_name in spec.hyper.param_names
        p_prior = m.p_priors[p_name]
        p_var_name = "p_$(p_name)_$(key)"
        if arch == "multivariate" && !get(spec.params, :shared, false)
            p_var_name *= "_$(outcome_idx)"
        end
        push!(prior_lines, "$(p_var_name) ~ $(_distribution_to_string(p_prior))")
    end
    
    return join(prior_lines, "\n    ")
end

function get_updates(
    m::SciML, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = string(spec.key)
    
    u0_name = "u0_$(key)"
    param_names = spec.hyper.param_names
    p_vars = ["p_$(p)_$(key)" for p in param_names]
    
    if arch == "multivariate" && !get(spec.params, :shared, false)
        u0_name *= "_$(outcome_idx)"
        p_vars = [v * "_$(outcome_idx)" for v in p_vars]
    end

    p_tuple_str = "p_$(spec.key) = ($(join(p_vars, ", ")),)"

    # Access pre-computed coordinates via spec_registry for consistency.
    common_solve_code = """
        $(p_tuple_str)
        prob_$(spec.key) = remake(
            M.sciml_problem_templates[:$(spec.key)]; u0=$(u0_name), p=p_$(spec.key)
        )
        sol_$(spec.key) = solve(
            prob_$(spec.key), M.sciml_solver; saveat=M.sciml_saveat
        )
        
        if !SciMLBase.successful_retcode(sol_$(spec.key))
            Turing.@addlogprob! -Inf
            return
        end
        
        sciml_effect_$(spec.key) = sol_$(spec.key)(spec_registry[:$(spec.key)].hyper.coords)
    """

    additive_code = """
        # --- SciML Component (Additive): $(spec.key) ---
        let
            $(common_solve_code)
            $(eta_target) .+= sciml_effect_$(spec.key)[1,:]
        end
    """

    # The invalid `M.likelihood_handled = true` assignment is removed.
    # The model assembler must be configured to check for this component type.
    direct_code = """
        # --- SciML Component (Direct Likelihood): $(spec.key) ---
        let
            $(common_solve_code)
            mu = sciml_effect_$(spec.key)
            y_sigma ~ Exponential(1.0)
            for i in 1:M.y_N
                Turing.@addlogprob! logpdf(Normal(mu[1, i], y_sigma), M.y_obs[i])
            end
            # The line `M.likelihood_handled = true` was removed as it is invalid.
            # The model assembler must be updated to recognize that this component
            # handles its own likelihood when `likelihood_type` is `:direct`.
        end
    """

    if m.likelihood_type == :additive
        return additive_code
    elseif m.likelihood_type == :direct
        return direct_code
    else
        error("Unsupported `likelihood_type`: $(m.likelihood_type)")
    end
end

function get_effects(
    m::SciML, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))
    key = string(spec.key)
    param_names = spec.hyper.param_names

    # Use coordinates from the spec for consistency.
    coords_train = spec.hyper.coords
    t_coords_full = if isnothing(PS)
        coords_train
    else
        vcat(coords_train, PS.data[!, M.t_idx_var])
    end
    tspan_full = (minimum(t_coords_full), maximum(t_coords_full))
    
    for k in 1:outcomes_N
        u0_name = _find_parameter(p_names_vec, key, "u0", k, is_multivariate_model)
        p_var_names = Dict(p_name => _find_parameter(p_names_vec, key, "p_$(p_name)", k, is_multivariate_model) for p_name in param_names)

        if isempty(u0_name) || any(isempty, values(p_var_names))
            @warn "Parameters for SciML component $(key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        u0_samples = get_params_vector(chain, u0_name, length(m.u0_prior))
        p_samples = Dict(p_name => get_params_vector(chain, p_var_name, 1) for (p_name, p_var_name) in p_var_names)

        T = eltype(chain.value)
        trajectories = zeros(T, length(t_coords_full), n_samples)

        prob_template = M.sciml_problem_templates[spec.key]

        for s in 1:n_samples
            u0_s = u0_samples[s, :]
            p_s = Tuple(p_samples[p_name][s, 1] for p_name in param_names)

            prob_s = remake(prob_template; u0=u0_s, p=p_s, tspan=tspan_full)
            
            sol_s = solve(prob_s, M.sciml_solver; saveat=t_coords_full)

            if SciMLBase.successful_retcode(sol_s)
                trajectories[:, s] = sol_s[1, :]
            else
                trajectories[:, s] .= NaN
            end
        end
        push!(structured_effects, trajectories)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
