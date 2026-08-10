"""
    SciML <: ComponentModel

A component for embedding user-defined differential equation models from the SciML
ecosystem into a Bayesian framework. It allows for the estimation of differential
equation parameters and initial conditions.

# Version
v1.0.1 (2026-08-10)

# Mathematical Summary
This component models an observed process \$y(t)\$ as noisy observations of a latent
state vector \$\\mathbf{u}(t)\$, where \$\\mathbf{u}(t)\$ is the solution to a system of
differential equations:
\$ \\frac{d\\mathbf{u}}{dt} = f(\\mathbf{u}, \\mathbf{p}, t) \$
where \$\\mathbf{p}\$ is a vector of parameters and \$\\mathbf{u}(t_0) = \\mathbf{u}_0\$ are the
initial conditions.

# Likelihood Types
- `:additive` (default): The solution of the DE is treated as an additive component
  in the model's linear predictor, \$\\eta = \\dots + \\mathbf{u}(t)\$. This is for
  when the DE models one of several influential processes.
- `:direct`: The solution of the DE is assumed to be the mean of the observation
  model directly, \$\\mu = \\mathbf{u}(t)\$. This is for when the DE is the complete
  generative model for the data.

# Fields
- `model_func::Symbol`: The name of the user-defined function that specifies the
  differential equation system (e.g., `:my_ode!`).
- `u0_prior::Any`: The prior distribution for the initial conditions `u0`.
- `p_priors::NamedTuple`: A `NamedTuple` where keys are parameter names and values
  are their prior distributions.
- `de_type::Symbol`: The type of differential equation: `:ODE`, `:SDE`, `:DDE`, or `:Jump`.
- `de_kwargs::Dict{Symbol, Any}`: Keyword arguments for the `DEProblem` constructor.
- `likelihood_type::Symbol`: The likelihood evaluation method, `:additive` or `:direct`.
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

"""
    get_datastructures!(m_type::Type{<:SciML}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `SciML` component. It validates arguments,
sets up the temporal context, and creates a `DEProblem` template that will be
`remake`d during MCMC sampling.
"""
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

    # Set up temporal context in the main configuration dictionary.
    time_opts = Dict(:time_method => get(params, :time_method, "continuous"))
    tu_meta = assign_time_units(M[:data][!, time_var_sym]; time_opts...)
    M[:t_idx] = tu_meta.idx
    M[:t_N] = tu_meta.N_cat
    M[:t_idx_var] = time_var_sym
    M[:t_coords] = M[:data][!, time_var_sym] # Store original time coordinates

    # Validate required parameters
    required_args = [:model_func, :u0_prior, :p_priors, :tspan, :solver]
    for arg in required_args
        if !haskey(params, arg)
            error("The `sciml()` module is missing the required keyword argument `:$arg`.")
        end
    end

    # Create and store the DEProblem template in the main model configuration.
    if !haskey(M, :sciml_problem_templates)
        M[:sciml_problem_templates] = Dict{Symbol, Any}()
    end
    
    calling_mod = get(M, :calling_module, Main)
    model_func = Core.eval(calling_mod, params[:model_func])
    
    # Use mean of priors for template instantiation
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

"""
    get_precomputes(m::SciML, M::NamedTuple, mod_data::Dict)::NamedTuple

Stores necessary information for code generation, including parameter names.
"""
function get_precomputes(m::SciML, M::NamedTuple, mod_data::Dict)::NamedTuple
    return (
        n_latent = M.y_N, # The effect is at the observation level
        param_names = keys(m.p_priors)
    )
end

"""
    get_priors(m::SciML, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the `SciML` component's priors.
"""
function get_priors(
    m::SciML, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    # Manually construct names as they are specific to this component
    u0_name = Symbol("$(p_names.latent)_u0")
    
    prior_lines = ["# --- Priors for SciML component: $(spec.key) ---"]
    push!(prior_lines, "$(u0_name) ~ $(_distribution_to_string(m.u0_prior))")
    
    for p_name in spec.precomputes.param_names
        p_prior = m.p_priors[p_name]
        p_var_name = Symbol("$(p_names.latent)_p_$(p_name)")
        push!(prior_lines, "$(p_var_name) ~ $(_distribution_to_string(p_prior))")
    end
    
    return join(prior_lines, "\n    ")
end

"""
    get_updates(m::SciML, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for solving the SciML problem, dispatching on the chosen
`likelihood_type`.
"""
function get_updates(
    m::SciML, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    u0_name = Symbol("$(p_names.latent)_u0")
    param_names = spec.precomputes.param_names
    p_vars = [Symbol("$(p_names.latent)_p_$(p)") for p in param_names]
    p_tuple_str = "local p_$(spec.key) = ($(join(p_vars, ", ")),)"

    common_solve_code = """
        $(p_tuple_str)
        local prob_$(spec.key) = remake(
            M.sciml_problem_templates[:$(spec.key)]; u0=$(u0_name), p=p_$(spec.key)
        )
        local sol_$(spec.key) = solve(
            prob_$(spec.key), M.sciml_solver; saveat=M.sciml_saveat
        )
        
        if !SciMLBase.successful_retcode(sol_$(spec.key))
            Turing.@addlogprob! -Inf
            return
        end
        
        # Interpolate solution to observation time points
        local sciml_effect_$(spec.key) = sol_$(spec.key)(M.t_coords)
    """

    additive_code = """
        # --- SciML Component (Additive): $(spec.key) ---
        let
            $(common_solve_code)
            # Add the first state variable's effect to the linear predictor
            $(eta_target) .+= sciml_effect_$(spec.key)[1,:]
        end
    """

    direct_code = """
        # --- SciML Component (Direct Likelihood): $(spec.key) ---
        let
            $(common_solve_code)
            
            # The solution is the mean of the observation model.
            local mu = sciml_effect_$(spec.key)
            
            # Evaluate the likelihood directly.
            # This assumes a Gaussian likelihood for simplicity.
            # A more advanced version could inspect M.likelihood_specs.
            y_sigma ~ Exponential(1.0)
            for i in 1:M.y_N
                Turing.@addlogprob! logpdf(Normal(mu[1, i], y_sigma), M.y_obs[i])
            end
            
            # Signal to the assembler that the likelihood has been handled.
            M[:likelihood_handled] = true
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

"""
    get_effects(m::SciML, chain, M::NamedTuple, ...)

Reconstructs the `SciML` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(
    m::SciML, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    p_names = generate_full_variable_names(spec, M.model_arch, nothing)
    u0_name = Symbol("$(p_names.latent)_u0")
    param_names = spec.precomputes.param_names
    p_var_names = [Symbol("$(p_names.latent)_p_$(p)") for p in param_names]

    # Extract posterior samples
    u0_samples = get_params_vector(chain, string(u0_name), length(m.u0_prior))
    p_samples = Dict(p_name => get_params_vector(chain, string(p_var_name), 1) for (p_name, p_var_name) in zip(param_names, p_var_names))

    # Determine the full time grid for reconstruction
    t_coords_full = if isnothing(PS)
        M.t_coords
    else
        vcat(M.t_coords, PS.data[!, M.t_idx_var])
    end
    tspan_full = (minimum(t_coords_full), maximum(t_coords_full))
    
    T = eltype(chain.value)
    trajectories = zeros(T, length(t_coords_full), n_samples)

    # Re-solve the differential equation for each posterior sample.
    for s in 1:n_samples
        u0_s = u0_samples[s, :]
        p_s = Tuple(p_samples[p_name][s, 1] for p_name in param_names)

        prob_template = M.sciml_problem_templates[spec.key]
        prob_s = remake(prob_template; u0=u0_s, p=p_s, tspan=tspan_full)
        
        sol_s = solve(prob_s, M.sciml_solver; saveat=t_coords_full)

        if SciMLBase.successful_retcode(sol_s)
            trajectories[:, s] = sol_s[1, :]
        else
            trajectories[:, s] .= NaN
        end
    end

    # The effect is assumed to be the same for all outcomes in a multivariate model.
    for k in 1:outcomes_N
        push!(structured_effects, trajectories)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
