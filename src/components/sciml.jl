# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    SciMLComponent <: ComponentModel

Represents a user-defined differential equation model from the SciML ecosystem.

# Fields
- `model_func::Symbol`: The name of the user-defined function that specifies the differential equation system (e.g., `:my_ode!`).
- `u0_prior::Any`: The prior distribution for the initial conditions `u0` of the system.
- `p_priors::NamedTuple`: A `NamedTuple` where keys are parameter names and values are their prior distributions.
- `de_type::Symbol`: The type of differential equation, one of `:ODE`, `:SDE`, `:DDE`, or `:Jump`.
- `de_kwargs::Dict{Symbol, Any}`: A dictionary of additional keyword arguments to be passed to the `DEProblem` constructor (e.g., `constant_lags` for DDEs).
"""
struct SciMLComponent <: ComponentModel
    model_func::Symbol
    u0_prior::Any
    p_priors::NamedTuple
    de_type::Symbol
    de_kwargs::Dict{Symbol, Any}
end

# Add to the central component constructor registry.
# This is called by `resolve_technical_primitive`.
COMPONENT_CONSTRUCTORS[:sciml] = (p, params) -> begin
    # The priors are resolved in `resolve_technical_primitive` and passed in `p`.
    # Other parameters are in `params`.
    model_func = get(params, :model_func, error("SciMLComponent requires a `model_func` parameter."))
    u0_prior = get(params, :u0_prior, error("SciMLComponent requires a `u0_prior` parameter."))
    p_priors = get(params, :p_priors, error("SciMLComponent requires a `p_priors` parameter."))
    de_type = get(params, :de_type, :ODE)
    
    # Collect any extra kwargs for the DEProblem constructor
    de_kwargs = Dict{Symbol, Any}()
    for key in [:constant_lags, :saveat] # Add other relevant kwargs here
        if haskey(params, key)
            de_kwargs[key] = params[key]
        end
    end

    SciMLComponent(model_func, u0_prior, p_priors, de_type, de_kwargs)
end

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[SciMLComponent] = :temporal

"""
    get_datastructures!(m_type::Type{<:SciMLComponent}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `SciMLComponent`. It validates arguments,
sets up the temporal context, and creates a `DEProblem` template.
"""
function get_datastructures!(m_type::Type{<:SciMLComponent}, M::Dict, mod_data::Dict)::Bool
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
    time_opts = Dict(:time_method => get(params, :time_method, "regular"))
    tu_meta = assign_time_units(M[:data][!, time_var_sym]; time_opts...)
    M[:t_idx] = tu_meta.idx
    M[:t_N] = tu_meta.N_cat
    M[:t_idx_var] = time_var_sym
    M[:t_coords] = M[:data][!, time_var_sym] # Store original time coordinates for interpolation.

    # Validate that all required parameters for defining a SciML problem are present.
    required_args = [:model_func, :u0_prior, :p_priors, :tspan, :solver]
    for arg in required_args
        if !haskey(params, arg)
            error("The `sciml()` module is missing the required keyword argument `:$arg`.")
        end
    end

    # Create and store the DEProblem template in the main model configuration.
    # This template will be `remake`d during sampling.
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
        # SDE requires a noise process function
        noise_func = get(params, :noise_func, error("SDE requires a `noise_func` parameter."))
        prob_template = SDEProblem(model_func, noise_func, u0_template, tspan, p_template)
    elseif de_type == :DDE
        h_func = get(params, :h, error("DDE requires a history function `h`."))
        prob_template = DDEProblem(model_func, u0_template, h_func, tspan, p_template; de_kwargs...)
    elseif de_type == :Jump
        # Jump problems are constructed differently, often from a DiscreteProblem
        # The `model_func` for jumps should be a constructor that returns a JumpProblem
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
    get_precomputes(m::SciMLComponent, M::NamedTuple, mod_data::Dict)::NamedTuple

For the `SciMLComponent`, this function stores necessary information for code generation.
"""
function get_precomputes(m::SciMLComponent, M::NamedTuple, mod_data::Dict)::NamedTuple
    # The main work is done in get_datastructures!. Here we just pass info.
    return (
        n_latent = M.y_N, # The effect is at the observation level
        param_names = keys(m.p_priors)
    )
end

"""
    get_priors(m::SciMLComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `SciMLComponent`'s priors.
"""
function get_priors(m::SciMLComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    # Manually construct names as they are specific to this component
    u0_name = Symbol("$(p_names.latent)_u0")
    
    prior_lines = ["# --- Priors for SciML component: $(spec.key) ---"]
    push!(prior_lines, "$(u0_name) ~ NamedDist($(_distribution_to_string(m.u0_prior)), :$(u0_name))")
    
    for p_name in spec.precomputes.param_names
        p_prior = m.p_priors[p_name]
        p_var_name = Symbol("$(p_names.latent)_p_$(p_name)")
        push!(prior_lines, "$(p_var_name) ~ NamedDist($(_distribution_to_string(p_prior)), :$(p_var_name))")
    end
    
    return join(prior_lines, "\n    ")
end

"""
    get_updates(m::SciMLComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for solving the SciML problem and updating eta.
"""
function get_updates(m::SciMLComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    u0_name = Symbol("$(p_names.latent)_u0")
    param_names = spec.precomputes.param_names
    p_vars = [Symbol("$(p_names.latent)_p_$(p)") for p in param_names]
    p_tuple_str = "p_$(spec.key) = ($(join(p_vars, ", ")),)"

    return """
        # --- SciML component assembly: $(spec.key) ---
        $(p_tuple_str)
        local prob_$(spec.key) = remake(M.sciml_problem_templates[:$(spec.key)]; u0=$(u0_name), p=p_$(spec.key))
        local sol_$(spec.key) = solve(prob_$(spec.key), M.sciml_solver; saveat=M.sciml_saveat)
        
        if !SciMLBase.successful_retcode(sol_$(spec.key))
            Turing.@addlogprob! -Inf
            return
        end
        
        # Interpolate solution to observation time points
        local sciml_effect_$(spec.key) = sol_$(spec.key)(M.t_coords)
        
        # Add the first state variable's effect to the linear predictor
        $(eta_target) .+= sciml_effect_$(spec.key)[1,:]
    """
end

"""
    get_effects(m::SciMLComponent, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `SciMLComponent`'s effect from the MCMC chain's posterior samples.
"""
function get_effects(m::SciMLComponent, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    # Manually construct names
    u0_name = Symbol("$(p_names.latent)_u0")
    param_names = spec.precomputes.param_names
    p_var_names = [Symbol("$(p_names.latent)_p_$(p)") for p in param_names]

    # Extract posterior samples
    u0_samples = get(chain, u0_name)
    p_samples = Dict(p_name => get(chain, p_var_name) for (p_name, p_var_name) in zip(param_names, p_var_names))

    # Determine the full time grid for reconstruction
    t_coords_full = if isnothing(PS)
        M.t_coords
    else
        vcat(M.t_coords, PS.data[!, M.t_idx_var])
    end
    tspan_full = (minimum(t_coords_full), maximum(t_coords_full))
    
    # Determine the element type from the chain's values for robust type handling.
    T = eltype(chain.value)
    trajectories = zeros(T, length(t_coords_full), n_samples)

    # Re-solve the differential equation for each posterior sample.
    for s in 1:n_samples
        u0_s = u0_samples[s, :]
        p_s = Tuple(p_samples[p_name][s] for p_name in param_names)

        prob_template = M.sciml_problem_templates[spec.key]
        prob_s = remake(prob_template; u0=u0_s, p=p_s, tspan=tspan_full)
        
        sol_s = solve(prob_s, M.sciml_solver; saveat=t_coords_full)

        if SciMLBase.successful_retcode(sol_s)
            trajectories[:, s] = sol_s[1, :]
        else
            trajectories[:, s] .= T(NaN)
        end
    end

    # The effect is assumed to be the same for all outcomes in a multivariate model unless specified otherwise.
    # For now, we return a single effect matrix.
    mean_effect = mean(trajectories, dims=2)[:]
    lower_ci = mapslices(x -> quantile(x, 0.025), trajectories, dims=2)[:]
    upper_ci = mapslices(x -> quantile(x, 0.975), trajectories, dims=2)[:]

    return (structured=(mean=mean_effect, lower=lower_ci, upper=upper_ci),)
end
