
"""
    SciMLComponent <: ComponentModel

Represents a user-defined differential equation model from the SciML ecosystem.

# Fields
- `model_func::Symbol`: The name of the user-defined function that specifies the differential equation system (e.g., `my_ode!`).
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


# --- SECTION 2: MODULE-SPECIFIC PROCESSOR ---

"""
    process_sciml_module!(opt_dict, mod_data, registries, hyperpriors)

Processes the `sciml()` module call from the formula.

# Rationale
This function validates the arguments provided to the `sciml()` module and establishes
the necessary temporal context (`t_idx`, `t_N`, `t_coords`) in the main model
configuration. It ensures that all required parameters for defining a SciML problem
(`model_func`, `u0_prior`, `p_priors`, `tspan`, `solver`) are present.
"""
function process_sciml_module!(opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict)
    # Purpose: Processes the `sciml()` module call, validating arguments and setting up temporal context.
    # Version: 1.0.0 (2026-08-06)

    data = opt_dict[:data]
    params = mod_data[:params]
    variables = mod_data[:variables]

    if isempty(variables)
        error("The `sciml()` module requires a time index variable, e.g., `sciml(year, ...)`.")
    end

    time_var_sym = Symbol(variables[1])
    if !hasproperty(data, time_var_sym)
        error("Time index variable ':$time_var_sym' for sciml() module not found in data.")
    end

    # Set up temporal context in the main configuration dictionary.
    time_opts = Dict(:time_method => get(params, :time_method, "regular"))
    tu_meta = assign_time_units(data[!, time_var_sym]; time_opts...)
    opt_dict[:t_idx] = tu_meta.idx
    opt_dict[:t_N] = tu_meta.N_cat
    opt_dict[:t_idx_var] = time_var_sym
    opt_dict[:t_coords] = data[!, time_var_sym] # Store original time coordinates for interpolation.

    # Validate that all required parameters for defining a SciML problem are present.
    required_args = [:model_func, :u0_prior, :p_priors, :tspan, :solver]
    for arg in required_args
        if !haskey(params, arg)
            error("The `sciml()` module is missing the required keyword argument `:$arg`.")
        end
    end

    # Store solver and tspan in the main config for the code generator to access.
    # This is necessary because they are not part of the ComponentModel struct itself.
    opt_dict[:sciml_solver] = params[:solver]
    opt_dict[:sciml_tspan] = params[:tspan]
    opt_dict[:sciml_saveat] = get(params, :saveat, 0.1) # Default saveat

    return true # Indicates that a component object should be created.
end


# --- SECTION 3: MODEL BUILDING AND ASSEMBLY ---

# This function should be added to the `resolve_technical_primitive` dispatch in `model.jl`.
# For this proposal, it is included here to show the complete logic.
# function resolve_technical_primitive(module_metadata::Dict{Symbol, Any}, M, priors_dict, scheme::Symbol)
#     if module_metadata[:type] == :sciml
#         ...
#     end
# end

"""
    build_model(m::SciMLComponent, data_inputs::Dict, module_metadata::Dict)

Model builder for the `SciMLComponent`.
"""
function build_model(m::SciMLComponent, data_inputs::Dict, module_metadata::Dict)
    # Purpose: Prepares the SciMLComponent for code generation.
    # Rationale: This builder packages all necessary information—priors, function names,
    #            and problem arguments—into the `hyper` registry. The code generator
    #            will then use this registry to construct the Turing model code.
    # Version: 1.0.0 (2026-08-06)

    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        hyper_dict[fn] = getfield(m, fn)
    end

    # No Q_template is needed for SciML components.
    return (Q_template=nothing, scaling_factor=1.0, model_type=:sciml, hyper=NamedTuple(hyper_dict))
end

"""
    generate_sciml_component_assembly(spec::NamedTuple, M::NamedTuple, arch::String)

Generates the Turing model code for a `SciMLComponent`.
"""
function generate_sciml_component_assembly(spec::NamedTuple, M::NamedTuple, arch::String)
    # Purpose: Generates the Turing code to sample parameters, solve a SciML problem, and add the result to eta.
    # Version: 1.0.0 (2026-08-06)

    key = spec.key
    hyper = spec.hyper
    p_priors = hyper.p_priors
    
    # --- 1. Generate Priors ---
    prior_lines = ["# --- Priors for SciML component: $(key) ---"]
    push!(prior_lines, "u0_$(key) ~ $(_distribution_to_string(hyper.u0_prior))")
    
    param_names = keys(p_priors)
    for p_name in param_names
        push!(prior_lines, "$(p_name)_$(key) ~ $(_distribution_to_string(p_priors[p_name]))")
    end
    priors_str = join(prior_lines, "\n    ")

    # --- 2. Generate Problem Construction and Solver Call ---
    # Collect sampled parameters into a tuple for the DE problem
    p_tuple_str = "p_$(key) = ($(join(["$(p)_$(key)" for p in param_names], ", ")),)"

    # Build the kwargs string for the problem constructor
    kwargs_str = join(["$(k)=$(v)" for (k, v) in hyper.de_kwargs], ", ")

    # The main assembly block
    assembly_lines = [
        "# --- SciML component assembly: $(key) ---",
        p_tuple_str,
        "prob_$(key) = remake(M.sciml_problem_templates[:$(key)]; u0=u0_$(key), p=p_$(key))",
        "sol_$(key) = solve(prob_$(key), M.sciml_solver; saveat=M.sciml_saveat, $(kwargs_str))",
        "",
        "if !SciMLBase.successful_retcode(sol_$(key))",
        "    Turing.@addlogprob! -Inf",
        "    return",
        "end",
        "",
        "# Interpolate solution to observation time points",
        "sciml_effect_$(key) = sol_$(key)(M.t_coords)",
        "",
        "# Add effect to the linear predictor",
        "# Assuming the first state variable is the output of interest.",
        "eta .+= sciml_effect_$(key)[1,:]"
    ]
    assembly_str = join(assembly_lines, "\n    ")

    return (priors=priors_str, assembly=assembly_str, post_assembly="")
end


# --- SECTION 4: POSTERIOR RECONSTRUCTION ---

"""
    extract_component(m_obj::SciMLComponent, chain, M, ...)

Reconstructs the posterior trajectories of a SciML model.
"""
function extract_component(m_obj::SciMLComponent, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # Purpose: Re-solves the differential equation for each posterior sample of its parameters.
    # Version: 1.0.0 (2026-08-06)

    key = spec.key
    hyper = spec.hyper
    p_priors = hyper.p_priors
    param_names = keys(p_priors)

    # Extract posterior samples for u0 and parameters
    u0_samples = get_params_vector(chain, "u0_$(key)", length(m_obj.u0_prior))
    p_samples = Dict(p_name => get_params_vector(chain, "$(p_name)_$(key)", 1) for p_name in param_names)

    # Determine the full time grid for reconstruction (training + prediction)
    t_coords_full = if isnothing(PS)
        M.t_coords
    else
        # This assumes time coordinates are consistent and sorted
        vcat(M.t_coords, PS.data[!, M.t_idx_var])
    end
    tspan_full = (minimum(t_coords_full), maximum(t_coords_full))

    # Initialize storage for the reconstructed trajectories
    # Assuming the first state variable is the output
    trajectories = zeros(Float64, N_tot, n_samples)

    # Re-solve the DE for each posterior sample
    for s in 1:n_samples
        u0_s = u0_samples[s, :]
        p_s = Tuple(p_samples[p_name][s, 1] for p_name in param_names)

        prob_template = M.sciml_problem_templates[key]
        prob_s = remake(prob_template; u0=u0_s, p=p_s, tspan=tspan_full)
        
        sol_s = solve(prob_s, M.sciml_solver; saveat=t_coords_full)

        if SciMLBase.successful_retcode(sol_s)
            # Extract the first state variable at the required time points
            trajectories[:, s] = sol_s[1, :]
        else
            # If solver fails, fill with NaNs to indicate failure
            trajectories[:, s] .= NaN
        end
    end

    # The effect is the same for all outcomes in a multivariate model unless specified otherwise
    structured_effects = [trajectories for _ in 1:outcomes_N]
    
    return (structured=structured_effects, noisy=structured_effects)
end


