"""
    SciML <: ComponentModel

A component for embedding user-defined differential equation models from the SciML
ecosystem into a Bayesian framework. It allows for the estimation of differential
equation parameters and initial conditions.

# Version
v1.3.0 (2026-08-17)

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
- `y_sigma_<key>`: The observation noise standard deviation (for `:direct` likelihood).
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

function get_precomputes(m::SciML, M::NamedTuple, mod_data::Dict)::NamedTuple
    # --- Data and Parameter Validation ---
    variables = mod_data[:variables]
    if isempty(variables)
        error("The `sciml()` module requires a time index variable, e.g., `sciml(year, ...)`.")
    end

    time_var_sym = Symbol(variables[1])
    if !hasproperty(M.data, time_var_sym)
        error("Time index variable ':$time_var_sym' for sciml() module not found in data.")
    end

    # Get device transfer function
    to_device = M.to_device

    # Extract coordinates to CPU first, then move to device
    coords_cpu = M.data[!, time_var_sym]

    params = mod_data[:params]
    required_args = [:model_func, :u0_prior, :p_priors, :tspan, :solver]
    for arg in required_args
        if !haskey(params, arg)
            error("The `sciml()` module is missing the required keyword argument `:$arg`.")
        end
    end

    # --- Problem Template Creation (on CPU) ---
    calling_mod = get(M, :calling_module, Main)
    model_func = Core.eval(calling_mod, params[:model_func])
    
    u0_template = mean(params[:u0_prior])
    p_template = Tuple(mean(p) for p in values(m.p_priors))
    tspan = params[:tspan]

    local prob_template
    if m.de_type == :ODE
        prob_template = ODEProblem(model_func, u0_template, tspan, p_template)
    elseif m.de_type == :SDE
        noise_func_sym = get(params, :noise_func, error("SDE requires a `noise_func`."))
        noise_func = Core.eval(calling_mod, noise_func_sym)
        prob_template = SDEProblem(model_func, noise_func, u0_template, tspan, p_template)
    elseif m.de_type == :DDE
        h_func_sym = get(params, :h, error("DDE requires a history function `h`."))
        h_func = Core.eval(calling_mod, h_func_sym)
        prob_template = DDEProblem(model_func, u0_template, h_func, tspan, p_template; m.de_kwargs...)
    elseif m.de_type == :Jump
        prob_template = model_func(u0_template, p_template, tspan)
    else
        error("Unsupported `de_type`: $(m.de_type)")
    end

    return (
        prob_template=prob_template,
        solver=params[:solver],
        saveat=get(params, :saveat, 0.1),
        coords=to_device(coords_cpu),
        param_names=keys(m.p_priors)
    )
end

function get_priors(
    m::SciML, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    
    prior_lines = ["# --- Priors for SciML component: $(spec.key) ---"]
    push!(prior_lines, "$(v.u0) ~ $(_distribution_to_string(m.u0_prior))")
    
    for p_name in spec.hyper.param_names
        p_prior = m.p_priors[p_name]
        p_var_name = getproperty(v, Symbol("p_$(p_name)"))
        push!(prior_lines, "$(p_var_name) ~ $(_distribution_to_string(p_prior))")
    end

    if m.likelihood_type == :direct
        y_sigma_name = getproperty(v, :y_sigma)
        push!(prior_lines, "$(y_sigma_name) ~ Exponential(1.0)")
    end
    
    return join(prior_lines, "\n    ")
end

function get_updates(
    m::SciML, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    
    v = generate_full_variable_names(spec, arch, outcome_idx)
    param_names = spec.hyper.param_names
    
    u0_name = v.u0
    p_var_names = [getproperty(v, Symbol("p_$(p)")) for p in param_names]
    p_tuple_str = "p_$(key) = ($(join(p_var_names, ", ")),)"

    common_solve_code = """
        $(p_tuple_str)
        hyper = spec_registry[:$(key)].hyper
        prob_$(key) = remake(
            hyper.prob_template; u0=$(u0_name), p=p_$(key)
        )
        sol_$(key) = solve(
            prob_$(key), hyper.solver; saveat=hyper.saveat
        )
        
        if !SciMLBase.successful_retcode(sol_$(key))
            Turing.@addlogprob! -Inf
            return
        end
        
        sciml_effect_$(key) = sol_$(key)(hyper.coords)
    """

    additive_code = """
        # --- SciML Component (Additive): $(key) ---
        let
            $(common_solve_code)
            $(eta_target) .+= sciml_effect_$(key)[1,:]
        end
    """

    direct_code = """
        # --- SciML Component (Direct Likelihood): $(key) ---
        let
            $(common_solve_code)
            # Assuming the first state variable is the one being observed.
            mu = sciml_effect_$(key)[1, :]
            y_sigma = $(getproperty(v, :y_sigma))
            
            # Vectorized likelihood calculation
            Turing.@addlogprob! logpdf(MvNormal(mu, y_sigma), M.y_obs)

            # The model assembler must be configured to check for this component type
            # and skip the main likelihood evaluation if likelihood_type is :direct.
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
    m::SciML, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device
    
    structured_effects = Vector{Matrix{Float64}}()
    key = spec.key
    hyper = spec.hyper
    param_names = hyper.param_names

    # --- Coordinate Handling: Combine training and prediction sets ---
    coords_train_cpu = Array(hyper.coords)
    t_coords_full_cpu = if !isnothing(PS) && hasproperty(PS.data, Symbol(spec.var))
        vcat(coords_train_cpu, Array(PS.data[!, Symbol(spec.var)]))
    else
        coords_train_cpu
    end
    tspan_full = (minimum(t_coords_full_cpu), maximum(t_coords_full_cpu))
    N_total = length(t_coords_full_cpu)

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        
        u0_name = _find_parameter(p_names, string(v.u0), k, is_multivariate_model)
        
        p_var_names = Dict{Symbol, String}()
        all_params_found = true
        for p_name in param_names
            p_sym = Symbol("p_$(p_name)")
            p_full_name = getproperty(v, p_sym)
            found_name = _find_parameter(p_names, string(p_full_name), k, is_multivariate_model)
            if isempty(found_name)
                all_params_found = false
                break
            end
            p_var_names[p_name] = found_name
        end

        if isempty(u0_name) || !all_params_found
            @warn "Parameters for SciML component $(key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (CPU)
        u0_samples_cpu = get_params_vector(chain, u0_name, length(m.u0_prior))
        p_samples_cpu = Dict(p_name => get_params_vector(chain, p_var_name, 1) for (p_name, p_var_name) in p_var_names)

        prob_template = hyper.prob_template

        # --- Ensemble Problem Setup for Parallel GPU Execution ---
        function prob_func(prob, i, repeat)
            u0_s = u0_samples_cpu[i, :]
            p_s_tuple = Tuple(p_samples_cpu[p_name][i, 1] for p_name in param_names)
            remake(prob; u0=u0_s, p=p_s_tuple, tspan=tspan_full)
        end

        ensemble_prob = EnsembleProblem(prob_template, prob_func=prob_func)
        
        # Choose ensemble algorithm based on device
        ensemble_alg = if parent_type(to_device(zeros(1))) <: CuArray
            @info "Using EnsembleGPUKernel for SciML reconstruction."
            EnsembleGPUKernel(0.0) # Auto-batching
        else
            EnsembleThreads()
        end

        # Solve all trajectories in parallel
        sim = solve(ensemble_prob, hyper.solver, ensemble_alg; trajectories=n_samples, saveat=t_coords_full_cpu)

        # --- Process Results ---
        trajectories_cpu = zeros(Float64, N_total, n_samples)
        for i in 1:n_samples
            if SciMLBase.successful_retcode(sim[i])
                # Solution is on CPU since the problem was defined on CPU.
                # We extract the first state variable's trajectory.
                trajectories_cpu[:, i] = sim[i][1, :]
            else
                trajectories_cpu[:, i] .= NaN
            end
        end
        
        push!(structured_effects, trajectories_cpu)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end

