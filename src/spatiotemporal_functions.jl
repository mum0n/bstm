#!Reference

function _generate_model_pseudocode(m::DynamicPPL.Model)
    # v1.0.0 (2026-06-30)
    # Purpose: Reconstructs a pseudo-code representation of the Turing model definition
    #          based on the configuration object. This is for inspection and clarity, as
    #          the actual model is built dynamically.
    # Inputs: m - The Turing model instance.
    # Outputs: A string containing the pseudo-code.
    
    config = m.args[1]
    model_name = get(config, :model_name, nameof(m.f))
    
    lines = ["@model function $model_name(M)"]
    push!(lines, "    # --- Priors & Hyperparameters ---")

    # Global priors
    family = get(config, :model_family, "gaussian")
    if family == "negbin"
        push!(lines, "    r_nb ~ Exponential(1.0)")
    end
    if get(config, :use_zi, false)
        push!(lines, "    phi_zi ~ Beta(1, 1)")
    end
    if family in ["gaussian", "lognormal"] && !get(config, :use_sv, false)
        push!(lines, "    y_sigma ~ Exponential(1.0)")
    end

    # Manifold-specific priors
    if haskey(config, :manifolds) && !isempty(config.manifolds)
        for spec in config.manifolds
            m_obj = spec.manifold_obj
            m_type_str = string(typeof(m_obj))
            key = spec.key
            
            push!(lines, "\n    # Priors for manifold: $(key) ($(m_type_str))")
            
            if hasproperty(m_obj, :sigma_prior) && !isnothing(m_obj.sigma_prior)
                push!(lines, "    sigma_$(key) ~ $(m_obj.sigma_prior)")
            end
            if hasproperty(m_obj, :rho_prior) && !isnothing(m_obj.rho_prior)
                push!(lines, "    rho_$(key) ~ $(m_obj.rho_prior)")
            end
            if hasproperty(m_obj, :lengthscale_prior) && !isnothing(m_obj.lengthscale_prior)
                push!(lines, "    ls_$(key) ~ $(m_obj.lengthscale_prior)")
            end
        end
    end

    # Fixed effects prior
    if get(config, :Xfixed_N, 0) > 0
        push!(lines, "\n    # Prior for fixed effects")
        push!(lines, "    Xfixed_beta ~ MvNormal(0, 5.0 * I)")
    end

    push!(lines, "\n    # --- Latent Field Definitions & Linear Predictor Assembly ---")
    eta_parts = haskey(config, :log_offset) && !all(iszero, get(config, :log_offset, [])) ? ["M.log_offset"] : []

    if haskey(config, :manifolds)
        for spec in config.manifolds
            push!(eta_parts, string(spec.key))
        end
    end
    if get(config, :Xfixed_N, 0) > 0
        push!(eta_parts, "M.Xfixed * Xfixed_beta")
    end
    
    push!(lines, "    eta = " * (isempty(eta_parts) ? "zeros(M.y_N)" : join(eta_parts, " .+ ")))

    push!(lines, "\n    # --- Likelihood ---")
    push!(lines, "    y_obs ~ bstm_Likelihood(\"$family\", eta, ...)")
    push!(lines, "end")

    return join(lines, "\n")
end


function show_model(m::DynamicPPL.Model)
"""
    show_model(m)

Displays a summary of the Turing model, including its name, arguments,
and attempts to draw a single sample from its prior to verify parameter definitions.
This function addresses the `UndefVarError: SampleFromPrior` by ensuring
`using Turing` is present and correctly calling the `SampleFromPrior` sampler.
"""
    println("\n--- Model Summary ---\n")

    # The model configuration object `M` is the first argument passed to the Turing model.
    config = m.args[1]

    # Use `get` with a fallback to `nameof(m.f)` for the model name.
    println("Model Name: ", get(config, :model_name, nameof(m.f)))
    println("Model Architecture: ", get(config, :model_arch, "N/A"))
    println("Model Family: ", get(config, :model_family, "N/A"))
    println("Number of observed data points: ", get(config, :N_obs, get(config, :y_N, "N/A")))
    println("Number of spatial units: ", get(config, :N_areas, get(config, :s_N, "N/A")))
    println("Number of time units: ", get(config, :N_time, get(config, :t_N, "N/A")))
    println("Number of covariates: ", get(config, :N_cov, "N/A"))

    println("\n--- Model Code View ---\n")
    println("Likelihood Family: ", get(config, :model_family, "N/A"))
    println("Zero-Inflated: ", get(config, :use_zi, false) ? "Yes" : "No")
    println("Hurdle Model: ", get(config, :hurdle, -Inf) > -Inf ? "Yes" : "No")

    println("\nFixed Effects:")
    if get(config, :Xfixed_N, 0) > 0
        println("  Variables: ", join(string.(names(config.Xfixed, 2)), ", "))
    else
        println("  None")
    end

    println("\nManifolds:\n")
    if haskey(config, :manifolds) && !isempty(config.manifolds)
        for spec in config.manifolds
            println("  - Key: ", spec.key)
            println("    Domain: ", spec.domain)
            println("    Variable: ", spec.var)
            println("    Manifold Type: ", typeof(spec.manifold_obj))
            println("    Parameters:")
            for (p_key, p_val) in pairs(spec.params)
                println("      ", p_key, ": ", p_val)
            end
        end
    else
        println("  None")
    end

    println("\nObservation Process:\n")
    # Check if log_offset is present and not all zeros
    log_offset_present = haskey(config, :log_offset) && !all(iszero, get(config, :log_offset, []))
    println("  Log Offsets: ", log_offset_present ? "Yes" : "No")

    # Check if weights are present and not all ones
    weights_present = haskey(config, :weights) && !all(isone, get(config, :weights, []))
    println("  Weights: ", weights_present ? "Yes" : "No")

    # Check if trials are present and not all ones
    trials_present = haskey(config, :trials) && !all(isone, get(config, :trials, []))
    println("  Trials: ", trials_present ? "Yes" : "No")

    println("  Stochastic Volatility: ", get(config, :use_sv, false) ? "Yes" : "No")

    censoring_status = "None"
    y_L = get(config, :y_lower_bound, -Inf)
    y_U = get(config, :y_upper_bound, Inf)
    if y_L > -Inf && y_U < Inf
        censoring_status = "Interval"
    elseif y_L > -Inf
        censoring_status = "Right"
    elseif y_U < Inf
        censoring_status = "Left"
    end
    println("  Censoring: ", censoring_status)

    println("\n--- End Model Code View ---")

    println("\n--- Reconstructed Model Source ---\n")

    println(_generate_model_pseudocode(m))
    println("\n--- End Reconstructed Model Source ---")

    
    println("\n--- Prior Sample Check ---")
    try
        prior_sample_chain = sample(m, Prior(), 1)
        println("Successfully drew 1 sample from the model's prior.")
        display(prior_sample_chain)
    catch e
        println("ERROR: Failed to draw a sample from the prior.")
        println("  Reason: ", e)
        println("  This might indicate issues with model parameter definitions,")
        println("  missing `using Turing` statement, or an incompatible Turing.jl version.")
        
        try 
            println( "Trying a simple rand(m) test: passed\n")
            println( "rand(m): sample\n")
            show( rand(m) )

        catch e
            println( "Trying a simple rand(m) test: failed\n")
            println("  Reason: ", e)
        end


    end

    println("\n--- End Model Summary ---")
    return nothing
end


function capitalize(s::String)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Capitalizes the first letter of a string.
    # Inputs: s::String - The input string.
    # Outputs: A string with the first letter capitalized.
    isempty(s) && return s
    return uppercasefirst(s)
end




function split_terms_at_depth(input::AbstractString, sep::AbstractString)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Splits a string by a separator, but respects nested parentheses or brackets.
    # Inputs: input::AbstractString - The string to split.
    #         sep::AbstractString - The separator to split by.
    # Outputs: A Vector{String} of the split terms.
    terms = String[]
    depth = 0
    current = ""
    for char in input
        if char == '(' || char == '['
            depth += 1
        elseif char == ')' || char == ']'
            depth -= 1
        end
        if string(char) == sep && depth == 0
            push!(terms, strip(current))
            current = ""
        else
            current *= char
        end
    end
    if !isempty(strip(current))
        push!(terms, strip(current))
    end
    return terms
end

function parse_module_params(params_str::AbstractString)
    # v1.0.0 (2026-06-29)
    # Purpose: Parses a string of key-value parameters (e.g., "nbins=20, model='bym2'")
    #          into a dictionary of symbols and values.
    # Inputs: params_str - The string of parameters.
    # Outputs: A dictionary of parsed parameters.
    params = Dict{Symbol, Any}()
    if isempty(strip(params_str))
        return params
    end

    # Use the existing utility to split at commas, respecting parentheses
    param_parts = split_terms_at_depth(params_str, ",")

    for part in param_parts
        kv = Base.split(part, "=")
        if length(kv) == 2
            key = Symbol(strip(kv[1]))
            val_str = strip(kv[2])

            # Attempt to parse the value
            # 1. Try parsing as a number
            parsed_val = tryparse(Float64, val_str)
            if !isnothing(parsed_val)
                # Check if it's an integer
                if parsed_val == floor(parsed_val)
                    params[key] = Int(parsed_val)
                else
                    params[key] = parsed_val
                end
            # 2. Check for booleans
            elseif val_str == "true"
                params[key] = true
            elseif val_str == "false"
                params[key] = false
            # 3. Check for strings (single or double quotes)
            elseif (startswith(val_str, "'") && endswith(val_str, "'")) || (startswith(val_str, "\"") && endswith(val_str, "\""))
                params[key] = val_str[2:end-1]
            # If it looks like code, try to parse it as an expression.
            elseif occursin(r"[:\[\]\(\)]", val_str)
                try; params[key] = Meta.parse(val_str); catch; params[key] = Symbol(val_str); end
            else
                params[key] = Symbol(val_str)
            end
        end
    end
    return params
end

function resolve_hyperpriors(m_id::Union{String, Symbol}, global_priors::Dict{String, Any}, module_priors::Dict{Symbol, Any}, scheme::Symbol=:pcpriors)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Resolves and assigns prior distributions to manifold types.
    # Inputs: m_id - The manifold identifier.
    #         global_priors - Priors defined globally for the model.
    #         module_priors - Priors defined within a specific module call.
    #         scheme - The default prior scheme to use.
    # Outputs: A NamedTuple of resolved hyperpriors.
    m_id_str = string(m_id)

    defaults = if scheme == :pcpriors
        PC_PRIORS
    elseif scheme == :informative
        INFORMATIVE_PRIORS
    else
        UNINFORMATIVE_PRIORS
    end

    function get_prior(module_key::Symbol, global_key::String, default_key::String)
        if haskey(module_priors, module_key)
            return module_priors[module_key]
        elseif haskey(global_priors, global_key)
            return global_priors[global_key]
        else
            return get(defaults, default_key, nothing)
        end
    end

    res = Dict{Symbol, Any}()
    res[:sigma_prior] = get_prior(:sigma_prior, "sigma", "sigma")
    res[:rho_prior] = m_id_str in ["bym2", "leroux", "sar", "ar1", "proper_car", "dag", "network"] ? get_prior(:rho_prior, "rho", "rho") : nothing
    res[:lengthscale_prior] = m_id_str in ["gp", "fitc", "rff", "warp", "nystrom", "decay"] ? get_prior(:lengthscale_prior, "lengthscale", "lengthscale") : nothing
    res[:kappa_prior] = m_id_str == "spde" ? get_prior(:kappa_prior, "kappa", "kappa") : nothing
    res[:amplitude_prior] = m_id_str == "harmonic" ? get_prior(:amplitude_prior, "amplitude", "amplitude") : nothing
    res[:phase_prior] = m_id_str == "harmonic" ? get_prior(:phase_prior, "phase", "phase") : nothing
    res[:pca_sd_prior] = m_id_str == "eigen" ? get_prior(:pca_sd_prior, "pca_sd", "pca_sd") : nothing
    res[:pdef_sd_prior] = m_id_str == "eigen" ? get_prior(:pdef_sd_prior, "pdef_sd", "pdef_sd") : nothing
    res[:range_prior] = m_id_str == "spherical" ? get_prior(:range_prior, "range", "range") : nothing

    return NamedTuple(res)
end







function apply_soft_constraint(latent_field::AbstractVector, constraint_type::Symbol, weight::Float64)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Applies a soft constraint to a latent field by adding a penalty to the log-probability.
    # Inputs: latent_field, constraint_type, weight.
    # Outputs: The penalty value (a Float64).
    if isempty(latent_field)
        return zero(eltype(latent_field))
    end

    if constraint_type == :sum_to_zero
        penalty = -weight * sum(latent_field)^2
    elseif constraint_type == :monotonic_increasing
        penalty = -weight * sum(max.(zero(T), -diff(latent_field)).^2)
    elseif constraint_type == :monotonic_decreasing
        penalty = -weight * sum(max.(zero(T), diff(latent_field)).^2)
    elseif constraint_type == :periodicity
        if length(latent_field) > 1
            penalty = -weight * (latent_field[1] - latent_field[end])^2
        end
    elseif constraint_type == :non_negative
        penalty = -weight * sum(max.(zero(T), -latent_field).^2)
    elseif constraint_type == :convex
        if length(latent_field) > 2; penalty = -weight * sum(max.(zero(T), -diff(diff(latent_field))).^2); end
    elseif constraint_type == :concave
        if length(latent_field) > 2; penalty = -weight * sum(max.(zero(T), diff(diff(latent_field))).^2); end
    else
        @warn "Unknown soft constraint type: $constraint_type. No penalty applied."
    end
    return penalty
end

function apply_regularization_penalty(fields::Vector{<:AbstractVector}, penalty_type::Symbol, lambda::Float64, alpha::Float64=0.5)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Applies a regularization penalty (Ridge, Lasso, Elastic Net) to a set of fields.
    # Inputs: fields, penalty_type, lambda, alpha.
    # Outputs: The penalty value (a Float64).
    if isempty(fields)
        return 0.0
    end
    all_coeffs = vcat(fields...)
    T = eltype(all_coeffs)
    penalty = zero(T)

    if penalty_type == :ridge
        penalty = -lambda * sum(all_coeffs.^2)
    elseif penalty_type == :lasso
        penalty = -lambda * sum(abs.(all_coeffs))
    elseif penalty_type == :elastic_net
        penalty = -lambda * (alpha * sum(abs.(all_coeffs)) + (1 - alpha) * sum(all_coeffs.^2))
    else
        @warn "Unknown regularization penalty type: $penalty_type. No penalty applied."
    end
    return penalty
end

function manifold_type(m::Manifold)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Returns the lowercase string representation of a manifold's type.
    # Inputs: m::Manifold.
    # Outputs: A string with the manifold type name.
    return lowercase(string(typeof(m)))
end
 
function _parse_module_call(module_call_str::AbstractString)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Parses a single module call string from the formula into a structured dictionary.
    # Inputs: module_call_str::AbstractString - The string to parse (e.g., "spatial(s_idx, model='bym2')").
    # Outputs: A dictionary containing the module's type, variables, and parameters.
    m_mod = match(r"(\w+)\((.*)\)", module_call_str)
    if isnothing(m_mod)
        return Dict{Symbol, Any}(:type => :literal, :value => module_call_str)
    end

    kw = lowercase(m_mod.captures[1])
    inner_str = m_mod.captures[2]
    
    if !(kw in BSTM_MODULE_KEYWORDS)
        return Dict{Symbol, Any}(:type => :literal, :value => module_call_str)
    end

    v_part_str = ""
    p_dict_raw = Dict{Symbol, Any}()
    
    semicolon_parts = split_terms_at_depth(inner_str, ";")
    if length(semicolon_parts) > 1
        v_part_str = semicolon_parts[1]
        p_dict_raw = parse_module_params(semicolon_parts[2])
    else
        # If no semicolon, it could be all variables, or variables mixed with params
        # We need to distinguish variables from key=value parameters
        all_inner_parts = split_terms_at_depth(inner_str, ",")
        var_only_parts = String[]
        param_only_parts = String[]
        for part in all_inner_parts
            if occursin("=", part)
                push!(param_only_parts, part)
            else
                push!(var_only_parts, part)
            end
        end
        v_part_str = join(var_only_parts, ",")
        p_dict_raw = parse_module_params(join(param_only_parts, ","))
    end

    raw_vars = [strip(v) for v in split_terms_at_depth(v_part_str, ",")]
    final_vars = String[]
    transforms = Dict{String, Vector{String}}()
    for var_expr in raw_vars
        var_name, transform_list = parse_variable_and_transforms(var_expr)
        push!(final_vars, var_name)
        if !isempty(transform_list)
            transforms[var_name] = transform_list
        end
    end

    if kw == "spatial" && length(final_vars) > 1
        cov_var = final_vars[1]
        idx_var = final_vars[2]
        
        p_dict_raw[:index_var] = idx_var
        
        return Dict{Symbol, Any}(
            :type => :svc,
            :variables => [cov_var],
            :params => p_dict_raw
        )
    end

    if kw == "transform"
        inner_module_call_str = final_vars[1]
        inner_module_data = _parse_module_call(inner_module_call_str) # Recursive call
        
        return Dict{Symbol, Any}(
            :type => Symbol(kw),
            :inner_module => inner_module_data,
            :params => p_dict_raw # This should contain 'fn'
        )
    elseif kw == "interaction"
        inner_model_call_str = get(p_dict_raw, :model, "")
        if inner_model_call_str isa String && !isempty(inner_model_call_str)
            inner_model_data = _parse_module_call(inner_model_call_str) # Recursive call
            p_dict_raw[:model] = inner_model_data # Replace string with parsed module Dict
        end
        return Dict{Symbol, Any}(
            :type => Symbol(kw),
            :variables => final_vars,
            :params => p_dict_raw # This should contain 'model'
        )
    else
        return Dict{Symbol, Any}(
            :type => Symbol(kw),
            :variables => final_vars,
            :params => p_dict_raw,
            :transforms => transforms
        )
    end
end

 




function resolve_technical_primitive(module_metadata::Dict{Symbol, Any}, M, priors_dict, scheme::Symbol)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Resolves a parsed module's metadata into a concrete `Manifold` struct instance.
    # Inputs: module_metadata, M (model config), priors_dict, scheme.
    # Outputs: An instance of a `ManifoldModel` or `ManifoldOperator`.
    m_type = module_metadata[:type]
    m_params = module_metadata[:params]
    
    if m_type == :transform
        inner_module_data = module_metadata[:inner_module]
        inner_manifold_obj = resolve_technical_primitive(inner_module_data, M, priors_dict, scheme)
        transform_fn = get(m_params, :fn, :identity)
        return TransformedManifold(inner_manifold_obj, transform_fn)
    elseif m_type == :interaction
        interaction_vars = [Symbol(v) for v in module_metadata[:variables]]
        inner_model_data = get(m_params, :model, Dict{Symbol, Any}())
        inner_manifold_obj = resolve_technical_primitive(inner_model_data, M, priors_dict, scheme)
        return VaryingInteractionManifold(interaction_vars, inner_manifold_obj)
    elseif m_type == :svc
        cov_sym = Symbol(module_metadata[:variables][1])
        model_str = string(get(m_params, :model, "iid"))
        
        inner_priors = resolve_hyperpriors(model_str, priors_dict, m_params, scheme)
        constructor_func = MANIFOLD_CONSTRUCTORS[Symbol(model_str)]
        inner_manifold = constructor_func(inner_priors, m_params)
        
        return SVCManifold(cov_sym, inner_manifold)
    else
        default_model = if m_type in [:spatial, :temporal, :smooth]
                            "none"
                        else string(m_type) end
        
        model_name = string(get(m_params, :model, default_model))
        model_sym = Symbol(model_name)

        resolved_priors = resolve_hyperpriors(model_name, priors_dict, m_params, scheme)

        if haskey(MANIFOLD_CONSTRUCTORS, model_sym)
            constructor_func = MANIFOLD_CONSTRUCTORS[model_sym]
            return constructor_func(resolved_priors, m_params)
        else
            @warn "Unknown manifold model '$model_name' for module '$m_type'. Defaulting to IID."
            return IID(Exponential(1.0))
        end
    end
end


  
function bstm_Likelihood(family_input, y_obs; sigma_y=0.0, weight=1.0, phi_zi=-Inf, r_nb=0, trial=0,
                         y_L=-Inf, y_U=Inf, hurdle=-Inf, extra_params=zeros(1)[])
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Constructor for the unified likelihood structure.
    # Inputs: family_input, y_obs, and various optional parameters for likelihood modifications.
    # Outputs: An instance of the bstm_Likelihood struct with appropriate traits.

    # Map string input to a concrete family type
    local fam = if family_input isa AbstractBSTM_Family; family_input
        elseif family_input == "poisson"; PoissonFamily()
        elseif family_input == "gaussian"; GaussianFamily()
        elseif family_input == "lognormal"; LogNormalFamily()
        elseif family_input == "negbin"; NegativeBinomialFamily()
        elseif family_input == "binomial"; BinomialFamily()
        elseif family_input == "gamma"; GammaFamily()
        elseif family_input == "exponential"; ExponentialFamily()
        elseif family_input == "beta"; BetaFamily()
        elseif family_input == "inverse_gaussian"; InverseGaussianFamily()
        elseif family_input == "student_t"; StudentTFamily()
        elseif family_input == "half_normal"; HalfNormalFamily()
        elseif family_input == "half_student_t"; HalfStudentTFamily()
        elseif family_input == "laplace"; LaplaceFamily()
        elseif family_input == "pareto"; ParetoFamily()
        elseif family_input == "dirichlet"; DirichletFamily()
        else PoissonFamily() end

    # Determine zero-inflation state from phi_zi
    local z_trait = phi_zi >= 0.0 ? ZeroInflated() : NonZeroInflated()

    local h_val = isnothing(hurdle) ? -Inf : hurdle
    local yL_val = isnothing(y_L) ? -Inf : y_L
    local yU_val = isnothing(y_U) ? Inf : y_U

    # Determine censoring state from bounds
    local c_trait = if !isfinite(yL_val) && !isfinite(yU_val); Uncensored()
        elseif !isfinite(yL_val) && isfinite(yU_val); LeftCensored()
        elseif isfinite(yL_val) && !isfinite(yU_val); RightCensored()
        else IntervalCensored() end

    return bstm_Likelihood(fam, y_obs, z_trait, c_trait, weight, phi_zi, r_nb, sigma_y, trial, yL_val, yU_val, h_val, extra_params)
end

# v1.0.1 (2026-06-29 17:16:00)
# Purpose: Returns a `Distributions.jl` object for a given model family and parameters.
# Inputs: family_type, d (likelihood struct), eta (linear predictor), sig (scale/noise).
# Outputs: A concrete distribution object (e.g., `Poisson`, `Normal`).


function get_dist_ref(::PoissonFamily, d, eta, sig); return Poisson(clamp(exp(eta), 1e-9, 1e9)); end
function get_dist_ref(::GaussianFamily, d, eta, sig); return Normal(eta, max(sig, 1e-9)); end
function get_dist_ref(::LogNormalFamily, d, eta, sig); return LogNormal(eta, max(sig, 1e-9)); end
function get_dist_ref(::NegativeBinomialFamily, d, eta, sig); mu = clamp(exp(eta), 1e-9, 1e9); return NegativeBinomial(d.r_nb, d.r_nb/(d.r_nb + mu)); end
function get_dist_ref(::BinomialFamily, d, eta, sig); n = d.trial isa AbstractVector ? d.trial[1] : d.trial; return Binomial(Int(n), logistic(eta)); end
function get_dist_ref(::GammaFamily, d, eta, sig); alpha = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0; return Gamma(alpha, clamp(exp(eta), 1e-9, 1e9)/alpha); end
function get_dist_ref(::ExponentialFamily, d, eta, sig); return Exponential(clamp(exp(eta), 1e-9, 1e9)); end
function get_dist_ref(::BetaFamily, d, eta, sig); mu = logistic(eta); phi = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 10.0; return Beta(clamp(mu*phi, 1e-9, Inf), clamp((1.0-mu)*phi, 1e-9, Inf)); end
function get_dist_ref(::InverseGaussianFamily, d, eta, sig); mu = clamp(exp(eta), 1e-9, 1e9); lambda = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0; return InverseGaussian(mu, lambda); end
function get_dist_ref(::StudentTFamily, d, eta, sig); nu = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 5.0; return LocationScale(eta, max(sig, 1e-9), TDist(nu)); end
function get_dist_ref(::HalfNormalFamily, d, eta, sig); return truncated(Normal(0.0, max(sig, 1e-9)), 0.0, Inf); end
function get_dist_ref(::HalfStudentTFamily, d, eta, sig); nu = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 5.0; return truncated(LocationScale(0.0, max(sig, 1e-9), TDist(nu)), 0.0, Inf); end
function get_dist_ref(::LaplaceFamily, d, eta, sig); return Laplace(eta, max(sig, 1e-9)); end
function get_dist_ref(::ParetoFamily, d, eta, sig)
    shape = d.extra_params isa Number && d.extra_params > 1.0 ? d.extra_params : 1.1
    mean_val = clamp(exp(eta), 1e-9, 1e9)
    scale = mean_val * (shape - 1.0) / shape
    return Pareto(shape, scale)
end
function get_dist_ref(::DirichletFamily, d, eta, sig); return Dirichlet(clamp.(exp.(eta), 1e-9, 1e9)); end

function get_dist_ref(::InverseWishartFamily, d, eta::AbstractMatrix, sig)
    p = size(eta, 1)
    nu = d.extra_params isa Number && d.extra_params > (p - 1) ? d.extra_params : Float64(p + 1)
    jitter = 1e-6
    scale_matrix = Symmetric(eta * eta' + jitter * I)
    return InverseWishart(nu, scale_matrix)
end


function is_discrete_family(::Union{PoissonFamily, NegativeBinomialFamily, BinomialFamily})
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Checks if a model family is discrete.
    # Inputs: A concrete type inheriting from AbstractBSTM_Family.
    # Outputs: true if the family is discrete, false otherwise.
    return true
end

function is_discrete_family(::AbstractBSTM_Family)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Default method for checking if a family is discrete.
    # Inputs: A concrete type inheriting from AbstractBSTM_Family.
    # Outputs: false.
    return false
end

function bstm_kernel(fam::AbstractBSTM_Family, ::Uncensored, zi::AbstractZIState, d, eta, sig, y)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Computes the log-probability for an uncensored observation.
    # Inputs: fam, censoring_state, zi_state, d (likelihood struct), eta, sig, y.
    # Outputs: The log-probability value.
    dist = get_dist_ref(fam, d, eta, sig)
    
    lp_branch = logpdf(dist, y)
    
    lp_final = lp_branch

    if zi isa ZeroInflated
        log_phi = log(d.phi_zi + 1e-15)
        log_one_minus_phi = log(1.0 - d.phi_zi + 1e-15)
        
        if y == 0.0
            if is_discrete_family(fam)
                p_base_zero = pdf(dist, 0.0)
                lp_final = logsumexp(log_phi, log_one_minus_phi + log(p_base_zero + 1e-15))
            else
                lp_final = log_phi
            end
        else
            lp_final = log_one_minus_phi + lp_branch
        end
    end

    if d.hurdle > -Inf
        lp_final = lp_final - logccdf(dist, d.hurdle)
    end

    return lp_final
end

function bstm_kernel(fam::AbstractBSTM_Family, ::LeftCensored, zi::AbstractZIState, d, eta, sig, y)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Computes the log-probability for a left-censored observation.
    # Inputs: fam, censoring_state, zi_state, d (likelihood struct), eta, sig, y.
    # Outputs: The log-probability value.
    dist = get_dist_ref(fam, d, eta, sig)
    
    upper_bound = d.y_U[1]
    
    if d.hurdle > -Inf
        if upper_bound <= d.hurdle
            return -Inf
        end
        lp_base = stable_logdiffexp(logcdf(dist, upper_bound), logcdf(dist, d.hurdle))
    else
        lp_base = logcdf(dist, upper_bound)
    end

    lp_final = lp_base
    if zi isa ZeroInflated
        log_phi = log(d.phi_zi + 1e-15)
        log_one_minus_phi = log(1.0 - d.phi_zi + 1e-15)
        
        if upper_bound >= 0.0 && d.hurdle < 0.0
             lp_final = logsumexp(log_phi, log_one_minus_phi + lp_base)
        else
             lp_final = log_one_minus_phi + lp_base
        end
    end

    # Normalize by the probability of being above the hurdle
    if d.hurdle > -Inf
        lp_final -= logccdf(dist, d.hurdle)
    end

    return lp_final
end


function bstm_kernel(fam::AbstractBSTM_Family, ::RightCensored, zi::AbstractZIState, d, eta, sig, y)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Computes the log-probability for a right-censored observation.
    # Inputs: fam, censoring_state, zi_state, d (likelihood struct), eta, sig, y.
    # Outputs: The log-probability value.
    dist = get_dist_ref(fam, d, eta, sig)
    
    lower_bound = d.y_L[1]
    effective_lower_bound = d.hurdle > -Inf ? max(lower_bound, d.hurdle) : lower_bound
    
    adj_L = is_discrete_family(fam) ? effective_lower_bound - 1.0 : effective_lower_bound
    
    lp_base = logccdf(dist, adj_L)

    lp_final = lp_base
    if zi isa ZeroInflated
        log_phi = log(d.phi_zi + 1e-15)
        log_one_minus_phi = log(1.0 - d.phi_zi + 1e-15)
        
        adj_L_cdf = is_discrete_family(fam) ? lower_bound : lower_bound
        logp_le_L = logcdf(dist, adj_L_cdf)
        
        log_prob_le_bound_zi = if lower_bound < 0.0
             log_one_minus_phi + logp_le_L
        else
            logsumexp(log_phi, log_one_minus_phi + logp_le_L)
        end
        
        lp_final = log1mexp(log_prob_le_bound_zi)
    end

    if d.hurdle > -Inf
        lp_final -= logccdf(dist, d.hurdle)
    end

    return lp_final
end


function bstm_kernel(fam::AbstractBSTM_Family, ::IntervalCensored, zi::AbstractZIState, d, eta, sig, y)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Computes the log-probability for an interval-censored observation.
    # Inputs: fam, censoring_state, zi_state, d (likelihood struct), eta, sig, y.
    # Outputs: The log-probability value.
    function stable_logdiffexp(a, b)
        if a <= b
            return -Inf # Effectively zero probability
        else
            diff = b - a
            safe_diff = min(-eps(typeof(diff)), diff)
            return a + log1mexp(safe_diff)
        end
    end

    dist = get_dist_ref(fam, d, eta, sig)
    
    # Adjust interval for hurdle model
    lower_bound = d.y_L[1]
    upper_bound = d.y_U[1]
    effective_lower_bound = d.hurdle > -Inf ? max(lower_bound, d.hurdle) : lower_bound

    # If the effective interval is invalid, return -Inf
    if upper_bound <= effective_lower_bound
        return -Inf
    end

    # Adjust for discrete distributions
    adj_L = is_discrete_family(fam) ? effective_lower_bound - 1.0 : effective_lower_bound
    
    # Base log-probability: P(adj_L < y <= upper_bound)
    lp_base = stable_logdiffexp(logcdf(dist, upper_bound), logcdf(dist, adj_L))

    # Apply zero-inflation logic
    lp_final = lp_base
    if zi isa ZeroInflated
        log_phi = log(d.phi_zi + 1e-15)
        log_one_minus_phi = log(1.0 - d.phi_zi + 1e-15)
        
        # If the original interval [y_L, y_U] includes zero, add the point mass phi.
        # The hurdle adjustment is already incorporated in lp_base.
        if lower_bound <= 0.0 && upper_bound >= 0.0 && d.hurdle < 0.0
            lp_final = logsumexp(log_phi, log_one_minus_phi + lp_base)
        else
            lp_final = log_one_minus_phi + lp_base
        end
    end

    if d.hurdle > -Inf
        lp_final -= logccdf(dist, d.hurdle)
    end

    return lp_final
end


# BSTM Internal Utility v1.1.0
# Timestamp: 2026-06-27 18:45:00
# Synopsis: Overloads the `Distributions.logpdf` function for the `bstm_Likelihood` struct.
#           This provides the main entry point for calculating the log-likelihood within a
#           Turing model, dispatching to the appropriate `bstm_kernel` based on the data traits.
# Rationale for v1.1.0:
#     - Removed redundant `local` keyword declarations for improved code clarity and style consistency.
#       The underlying logic remains correct and unchanged.

function Distributions.logpdf(d::bstm_Likelihood, eta::Real)
    sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
    w = d.weight isa AbstractVector ? d.weight[1] : d.weight
    return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, d.y_obs[1]) * w
end

function Distributions.logpdf(d::bstm_Likelihood, eta::AbstractVector)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Overloads logpdf for a vector of observations.
    # Inputs: d::bstm_Likelihood, eta::AbstractVector.
    # Outputs: The total log-probability value.
    if d.family isa DirichletFamily
        w = d.weight isa AbstractVector ? d.weight[1] : d.weight
        return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, 1.0, d.y_obs) * w
    end
    
    total_lp = 0.0
    for i in 1:length(eta)
        sig = d.sigma_y isa AbstractVector ? d.sigma_y[i] : d.sigma_y
        w = d.weight isa AbstractVector ? d.weight[i] : d.weight
        total_lp += bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta[i], sig, d.y_obs[i]) * w
    end
    return total_lp
end

function Distributions.logpdf(d::bstm_Likelihood, eta::AbstractMatrix)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Overloads logpdf for matrix-variate observations (e.g., InverseWishart).
    # Inputs: d::bstm_Likelihood, eta::AbstractMatrix.
    # Outputs: The log-probability value.
    w = d.weight isa AbstractVector ? d.weight[1] : d.weight
    return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, 1.0, d.y_obs) * w
end


function scaling_factor_bym2( adjacency_mat )
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Calculates the geometric mean of the variances of an ICAR precision matrix.
    # Inputs: adjacency_mat - The adjacency matrix of the graph.
    # Outputs: The scaling factor.
    N = size(adjacency_mat)[1]
    if N <= 1
        return 1.0
    end

    # 1. Construct the Graph Laplacian
    asum = vec(sum(adjacency_mat, dims=2))
    Q = Diagonal(asum) - adjacency_mat

    jitter = sqrt(1e-15) * mean(asum)
    Q_stable = Q +jitter * I

    A = ones(N)
    try
        S = Q_stable \ Diagonal(A)
        V = S * A
        S = S - V * inv(A' * V) * V'

        diag_s = diag(S)

        valid_diag = filter(x -> x > jitter, diag_s)

        if isempty(valid_diag)
            return 1.0
        end

        scale_factor = exp(mean(log.(valid_diag)))

        return isnan(scale_factor) ? 1.0 : scale_factor
    catch
        return 1.0
    end
end

function scaling_factor_bym2(node1, node2, groups=ones(length(node1)))
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Overloaded method to calculate scaling factors for multiple disconnected graph components.
    # Inputs: node1, node2 (edge list), groups (vector of group identifiers).
    # Outputs: A vector of scaling factors, one for each group.
    gr = unique( groups )
    n_groups = length(gr)
    scale_factor = ones(n_groups)

    Threads.@threads for j in 1:n_groups
        k = findall( x -> x==j, groups)
        if length(k) > 1
            e = Edge.(node1[k], node2[k])
            g = Graph(e)
            adjacency_mat = adjacency_matrix(g)
            scale_factor[j] = scaling_factor_bym2( adjacency_mat )
        end
    end

    return scale_factor
end



function discretize_data(X; method="quantile", N_cat=9, brks=nothing, probs=nothing, dx=nothing, minv = 0, maxv=1)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Discretizes a continuous vector into categories.
    # Inputs: X (vector), method, N_cat, and other optional parameters.
    # Outputs: A NamedTuple with indices, breaks, and midpoints.
     
    if method=="quantile" 
        probs = isnothing(probs) ? collect(range(0.0, stop=1.0, length=N_cat + 1)) : probs
        brks = isnothing(brks) ? quantile(X, probs) : brks
        mids = brks[1:N_cat] + diff(brks) 
        idx = map(x -> clamp(searchsortedfirst(brks, x) - 1, 1, N_cat), X)
        return ( idx=idx, brks=brks, mids=mids, probs )

    elseif method == "regular"
        dx = isnothing(dx) ? (maxv - minv) / N_cat : dx
        probs = nothing
        brks = collect(minv:dx:maxv)
        mids = brks[1:N_cat] + diff(brks) 
        idx = map(x -> clamp(searchsortedfirst(brks, x) - 1, 1, N_cat), X)
        return ( idx=idx, brks=brks, mids=mids, probs, dx=dx )
    
    elseif method=="regular_resolution"    

        xd = round.(Int, X ./ dx ) .* dx
        brks = collect( minimum(xd):dx:maximum(xd) + dx  ) 
        mids = midpoints(brks)
        N_cats = length(mids)
        
        xd_cut = cut(X, brks, extend=true)
        xi = levelcode.(xd_cut)
        return xd, xi, mids, N_cats, dx

    elseif method=="quantile_resolution"
    
        brks = quantile(X, range(0, 1, length=N_cats+1))
        mids = midpoints(brks)
        xd_cut = cut(X, brks, extend=true)  # from CategoricalArrays
        xi = levelcode.(xd_cut)
        dx = diff(mids)[1]
        xd = mids[xi] 
        return xd, xi, mids, N_cats, dx

    end

end
 
 

function rff_map(coords, W, b)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Maps input coordinates into a Random Fourier Feature (RFF) space.
    # Inputs: coords (N x D matrix), W (D x M weight matrix), b (M-element phase vector).
    # Outputs: An N x M feature matrix.

    projection = (coords * W) .+ b'
    
    m = size(W, 2)
    feature_map = sqrt(2 / m) .* cos.(projection)
    
    return feature_map
end
 

function generate_informed_rff_params(coords, M_rff; kernel_name="se", nu=nothing, lengthscale_mult=0.5)
    # v1.0.2 (2026-06-29 18:10:00)
    # Purpose: Generates RFF parameters (W, b) with a lengthscale informed by the input data's standard deviation.
    # Inputs: coords (matrix or vector of tuples), M_rff (number of features).
    #         kernel_name - The kernel to use for the spectral density.
    #         nu - Smoothness parameter for Matern kernels.
    #         lengthscale_mult - Multiplier for the informed lengthscale.
    # Outputs: A tuple (W, b) of RFF parameters.
    mat = if coords isa AbstractMatrix && eltype(coords) <: Real
        Matrix{Float64}(coords)
    elseif coords isa AbstractVector && eltype(coords) <: Tuple
        reduce(hcat, [[Float64(p[1]), Float64(p[2])] for p in coords])'
    else
        Matrix{Float64}(reduce(hcat, collect.(coords))')
    end

    if size(mat, 1) > size(mat, 2) && size(mat, 2) <= 3
        mat = mat'
    end

    d = size(mat, 1)
    
    coord_scales = vec(std(mat, dims=2)) .+ 1e-6

    informed_ls = mean(coord_scales) * lengthscale_mult
    W = sample_spectral_density(kernel_name, d, M_rff, informed_ls; nu=nu)

    b = rand(M_rff) .* (2.0 * pi)
    
    return W, b
end


function generate_rff_params_for_se_kernel(D_in, M_rff, lengthscale)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: A helper function to generate Random Fourier Feature parameters specifically for a Squared Exponential kernel.
    # Inputs: D_in - The input dimension.
    #         M_rff - The number of RFF features.
    #         lengthscale - The lengthscale of the SE kernel.
    # Outputs: A tuple (W, b) of RFF parameters.
    # Helper function to generate RFF parameters for a Squared Exponential kernel
    # For a Squared Exponential kernel, the spectral density is Gaussian: N(0, (1/l)^2 * I)
    sigma_spectral = 1.0 / lengthscale
    W_matrix = randn(D_in, M_rff) .* sigma_spectral # D_in x M_rff matrix
    b_vector = rand(Uniform(0, 2pi), M_rff)
    return W_matrix, b_vector
end 
 





function build_laplacian_precision(adj_matrix)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Constructs a GMRF precision matrix (Graph Laplacian, Q = D - W) from an adjacency matrix.
    # Inputs: adj_matrix - Sparse adjacency matrix (W).
    # Outputs: Sparse precision matrix (Q).
    # Note: This is the fundamental structure for ICAR and other graph-based spatial models.

    # D is the diagonal matrix of node degrees
    D_diag = Diagonal(vec(sum(adj_matrix, dims=2)))
    Q_mat = D_diag - adj_matrix
    
    return Q_mat
end
 





 
 

function plot_posterior_results(stats, M=nothing, areal_units=nothing; s_x=nothing, s_y=nothing, time_slice=nothing, effect=:spatial, cov_idx=1, show_pts=false)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Comprehensive visualization for posterior results from bstm models.
    # Inputs: stats (results object), M (model config), areal_units, and various plotting options.
    # Outputs: A Plots.jl plot object.
    st = getproperty(stats, effect)
    isnothing(st) && return nothing
    if st isa Real
        return Plots.plot(title="$effect (Fixed: $st)")
    end
 
    if effect == :beta_cov
        b_list = get(stats, :beta_cov, nothing)
        isnothing(b_list) && return nothing
        b_stats = b_list isa AbstractVector ? b_list[cov_idx] : b_list
        (isnothing(b_stats) || b_stats isa Real) && return nothing
        n_levels = size(b_stats.mean, 1)
        return StatsPlots.bar(1:n_levels, b_stats.mean[:,1],
                  yerror=(b_stats.mean[:,1] .- b_stats.lower[:,1], b_stats.upper[:,1] .- b_stats.mean[:,1]),
                  title="Covariate $cov_idx Effects", xlabel="Level", ylabel="Effect Size", legend=false)

    elseif effect == :b_class1 || effect == :b_class2
        b_stats = st
        if isnothing(b_stats) || b_stats isa Real; return nothing; end
        n_levels = size(b_stats.mean, 1)
        return StatsPlots.bar(1:n_levels, b_stats.mean[:,1],
                  yerror=(b_stats.mean[:,1] .- b_stats.lower[:,1], b_stats.upper[:,1] .- b_stats.mean[:,1]),
                  title="$effect Levels", xlabel="Class Index", ylabel="Effect Size", legend=false)

    elseif effect == :temporal
        t_stats = st
        if isnothing(t_stats) || t_stats isa Real; return nothing; end
        n_times = length(t_stats.mean)
        return StatsPlots.plot(1:n_times, t_stats.mean,
                   ribbon=(t_stats.mean .- t_stats.lower, t_stats.upper .- t_stats.mean),
                   fillalpha=0.2, lw=2, title="Temporal Main Effect", xlabel="Time Index", ylabel="Effect (Latent Scale)", legend=false)

    elseif effect == :seasonal
        t_stats = st
        if isnothing(t_stats) || t_stats isa Real; return nothing; end
        n_times = length(t_stats.mean)
        return StatsPlots.plot(1:n_times, t_stats.mean,
                   ribbon=(t_stats.mean .- t_stats.lower, t_stats.upper .- t_stats.mean),
                   fillalpha=0.2, lw=2, title="Seasonal Effect", xlabel="Time Index", ylabel="Effect (Latent Scale)", legend=false)

    elseif effect in [:spatial, :spatial_structured, :spatial_unstructured, :predictions_denoised, :predictions_noisy, :residuals, :eta_gp, :hidden_layer]
        plt = StatsPlots.plot(aspect_ratio=:equal, title="$effect (T=$(time_slice))", legend=true)

        values = if hasproperty(st, :mean)
            st.mean
        elseif effect == :spatial_structured
            stats.spatial_structured.mean
        elseif effect == :spatial_unstructured
            stats.spatial_unstructured.mean
        elseif effect == :eta_gp
            haskey(stats, :eta_gp) ? stats.eta_gp.mean : error("eta_gp not found in stats")
        elseif effect == :hidden_layer
            haskey(stats, :h1) ? stats.h1.mean : error("hidden layer h1 not found in stats")
        elseif effect == :predictions_denoised && !isnothing(time_slice)
            stats.predictions_denoised.mean[:, time_slice]
        elseif effect == :predictions_noisy && !isnothing(time_slice)
            stats.predictions_noisy.mean[:, time_slice]
        else
            error("Effect $effect requires specific keys in stats or time_slice index")
        end

        n_to_plot = min(length(areal_units.polygons), length(values))

        for i in 1:n_to_plot
            poly_coords = areal_units.polygons[i]
            if length(poly_coords) > 2
                px = [pt[1] for pt in poly_coords if !isnan(pt[1])]
                py = [pt[2] for pt in poly_coords if !isnan(pt[2])]

                if !isempty(px)
                    if (px[1], py[1]) != (px[end], py[end])
                        push!(px, px[1]); push!(py, py[1])
                    end

                    val = values[i]
                    StatsPlots.plot!(plt, px, py,
                        seriestype=:shape,
                        fill_z=val,
                        c=:RdYlBu,
                        linecolor=:black,
                        linewidth=0.5,
                        fillalpha=0.8,
                        legend=false
                    )
                end
            end
        end

        if show_pts
            StatsPlots.scatter!(plt, s_x, s_y,
                markersize=1, markercolor=:gray, alpha=0.2, label="Observations")
        end

        StatsPlots.scatter!(plt, [c[1] for c in areal_units.centroids], [c[2] for c in areal_units.centroids],
            markersize=2, markercolor=:white, markerstrokecolor=:black, alpha=0.5, label="Centroids")

        return plt
    else
        error("Effect $effect not recognized.")
    end
end




function plot_posterior_vs_prior(model::DynamicPPL.Model, chain::MCMCChains.Chains, param_sym::Symbol; n_prior_samples=1000, title="Posterior vs Prior")
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Overlays the posterior density of a parameter with its prior density.
    # Inputs: model, chain, param_sym, n_prior_samples, title.
    # Outputs: A Plots.jl plot object.

    post_samples = vec(chain[param_sym].data)

    prior_chain = sample(model, Prior(), n_prior_samples, progress=false)
    prior_samples = vec(prior_chain[param_sym].data)

    plt = StatsPlots.density(post_samples, label="Posterior: $param_sym", lw=3, color=:blue, fill=(0, 0.2, :blue))
    StatsPlots.density!(plt, prior_samples, label="Prior (sampled)", lw=2, ls=:dash, color=:red)

    title!(plt, title)
    xlabel!(plt, "Value")
    ylabel!(plt, "Density")

    return plt
end


function get_rff_deep2D_basis(X, m, lengthscale)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Generates a Random Fourier Feature (RFF) basis for 2D inputs.
    # Inputs: X (N x D matrix), m (number of features), lengthscale.
    # Outputs: An N x m feature matrix.
    N, D = size(X)
    Random.seed!(42)
    Omega_samples = randn(m, D) ./ lengthscale
    Phi_phases = rand(m) .*  2pi
    return sqrt(2/m) .* cos.(X * Omega_samples' .+ Phi_phases')
end

function get_rff_trend_basis(t, m, lengthscale, ::Type{T}=Float64) where {T}
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Generates an RFF basis for a 1D temporal trend.
    # Inputs: t (time vector), m (number of features), lengthscale.
    # Outputs: An N x m feature matrix.
    N = length(t)
    Random.seed!(42)
    Omega_samples_float = randn(m)
    Phi_phases_float = rand(m)

    Omega_samples = Omega_samples_float ./ lengthscale
    Phi_phases = Phi_phases_float .* convert(T,  2pi)

    Z = zeros(T, N, m)
    for j in 1:m
        Z[:, j] = convert.(T, sqrt(2/m)) .* cos.(Omega_samples[j] .* t .+ Phi_phases[j])
    end
    return Z
end


function get_rff_seasonal_basis(t, m, freq, lengthscale)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Generates a deterministic Fourier series basis for periodic components.
    # Inputs: t (time vector), m (number of harmonics), freq (base frequency), lengthscale.
    # Outputs: An N x (2*m) feature matrix.
    N = length(t)
    Z = zeros(N, 2*m)
    for j in 1:m
        omega_j =  2pi * j * freq
        Z[:, 2j-1] = cos.(omega_j .* t)
        Z[:, 2j] = sin.(omega_j .* t)
    end
    return Z
end

 

function apply_discretization_logic(vals, rules)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Applies a discretization or transformation rule to a vector of values.
    # Inputs: vals (vector), rules (discretization/transformation rule).
    # Outputs: A tuple (new_vals, groups) where new_vals may be transformed values and groups are integer indices.
    groups = nothing
    new_vals = nothing

    if rules == 0 || isnothing(rules)
        groups = collect(1:length(vals))
    elseif rules == "unit"
        min_v, max_v = minimum(vals), maximum(vals)
        new_vals = (min_v == max_v) ? zeros(length(vals)) : (vals .- min_v) ./ (max_v - min_v)
        groups = collect(1:length(vals))
    elseif rules == "zscore"
        m, s = Statistics.mean(vals), Statistics.std(vals)
        new_vals = (s ≈ 0.0) ? zeros(length(vals)) : (vals .- m) ./ s
        groups = collect(1:length(vals))
    elseif rules == "log"
        new_vals = log.(vals .+ 1.0 .- minimum(vals))
        groups = collect(1:length(vals))
    elseif rules isa Int
        if rules > 1
            qs = unique(sort(quantile(vals, (0:rules) ./ rules)))
            groups = (length(qs) < 2) ? ones(Int, length(vals)) : clamp.(map(x -> searchsortedlast(qs, x), vals), 1, length(qs)-1)
        else
            groups = ones(Int, length(vals))
        end
    elseif rules isa AbstractString && startswith(rules, "regular:")
        n = parse(Int, Base.split(rules, ":")[2])
        q025, q975 = quantile(vals, 0.025), quantile(vals, 0.975)
        bins = unique(sort(collect(range(q025, stop=q975, length=n+1))))
        groups = (length(bins) < 2) ? ones(Int, length(vals)) : clamp.(map(x -> searchsortedlast(bins, x), vals), 1, length(bins)-1)
    elseif rules isa AbstractVector
        groups = clamp.(map(x -> searchsortedlast(rules, x) + 1, vals), 1, length(rules) + 1)
    end
    return new_vals, groups
end



function assign_covariate_units(cov_data_base, cov_discretization, re_rules, cov_interactions)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Pre-processes covariates by handling discretization, transformation, and interaction terms.
    # Inputs: cov_data_base, cov_discretization, re_rules, cov_interactions.
    # Outputs: A tuple of the processed covariate data, group indices, and random effect structures.
    cov_data_for_processing = deepcopy(cov_data_base)
    cov_groups = Dict{Symbol, Vector{Int}}()
    cov_re_structures = Dict{Symbol, Any}()

    if !isnothing(cov_discretization)
        for cov_name_sym in names(cov_data_base, 1)
            if haskey(cov_discretization, cov_name_sym)
                rules = cov_discretization[cov_name_sym]
                vals = cov_data_for_processing[cov_name_sym, :]
                new_vals, groups = apply_discretization_logic(vals, rules)
                if !isnothing(new_vals); cov_data_for_processing[cov_name_sym, :] = new_vals; end
                if !isnothing(groups)
                    cov_groups[cov_name_sym] = groups
                    if haskey(re_rules, cov_name_sym) && length(unique(groups)) > 1
                        n_bins = length(unique(groups))
                        if re_rules[cov_name_sym] == "rw2"; cov_re_structures[cov_name_sym] = build_bstm_rw2_template(n_bins).matrix
                        elseif re_rules[cov_name_sym] == "ar1"; cov_re_structures[cov_name_sym] = build_bstm_ar1_template(n_bins).matrix
                        end
                    end
                end
            end
        end
    end

    for inter_str in cov_interactions
        parts = Base.split(inter_str, "*")
        if length(parts) == 2
            n1, n2 = Symbol(parts[1]), Symbol(parts[2])
            if n1 in names(cov_data_for_processing, 1) && n2 in names(cov_data_for_processing, 1)
                inter_val = cov_data_for_processing[n1, :] .* cov_data_for_processing[n2, :]
                new_row = NamedArray(inter_val', (Symbol[Symbol(inter_str)], names(cov_data_for_processing, 2)))
                cov_data_for_processing = vcat(cov_data_for_processing, new_row)
                if !isnothing(cov_discretization) && haskey(cov_discretization, Symbol(inter_str))
                    rule_int = cov_discretization[Symbol(inter_str)]
                    iv_vals, iv_groups = apply_discretization_logic(inter_val, rule_int)
                    if !isnothing(iv_vals); cov_data_for_processing[Symbol(inter_str), :] = iv_vals; end
                    if !isnothing(iv_groups)
                        cov_groups[Symbol(inter_str)] = iv_groups
                        if haskey(re_rules, Symbol(inter_str)) && length(unique(iv_groups)) > 1
                            n_bins = length(unique(iv_groups))
                            if re_rules[Symbol(inter_str)] == "rw2"; cov_re_structures[Symbol(inter_str)] = build_bstm_rw2_template(n_bins).matrix
                            elseif re_rules[Symbol(inter_str)] == "ar1"; cov_re_structures[Symbol(inter_str)] = build_bstm_ar1_template(n_bins).matrix
                            end
                        end
                    end
                end
            end
        end
    end
    return cov_data_for_processing, cov_groups, cov_re_structures
end




function variational_inference_solution(m; max_iters=100, nsamps=max_iters,  nelbo=3 )
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Runs Automatic Differentiation Variational Inference (ADVI) on a Turing model.
    # Inputs: m (Turing model), max_iters, nsamps, nelbo.
    # Outputs: A NamedTuple containing the VI result, samples, mean, and standard deviation.

    _, indices = Bijectors.bijector(m, Val(true));
    vars = keys(indices)

    q0 = Variational.meanfield(m)     # initialize variational distribution (optional)
    advi = ADVI(nelbo, max_iters)    # num_elbo_samples, max_iters
    msol = Turing.vi(m, advi, q0) #, optimizer=Flux.ADAM(1e-1));
    msamples = DataFrame( rand(msol, nsamps )', :auto ) 

    vns = []
    for (i, sym) in enumerate(vars) 
        j = union(indices[sym]...)  # <= array of ranges
        nj = sum(length.(j)) 
        if  nj > 1
            k = 1
            for r in j
                push!(vns, "$(sym)[$k]")
                k += 1
            end
        else
            push!(vns, "$(sym)") 
        end
    end
    
    vns = Symbol.(vns)

    msamples = rename(msamples, vns)

    mmean = combine( msamples, [ n => (x -> mean(x)) => n for n in names(msamples)  ] )
    mstd  = combine( msamples, [ n => (x -> std(x)) => n for n in names(msamples)  ] )

    out = (
        msol = msol,
        msamples = msamples, 
        mmean = mmean,
        mstd = mstd
    )
    
    return out
 
end


# -----------------------

function get_params_vector(chain, base_name::String, expected_len::Int)
    # Logic: Extracts parameters from FlexiChains, handling indexed names via numerical sorting.
    # Change: Fixed regex escaping for Julia strings to correctly match square brackets.

    N_samples = size(chain, 1)
    all_names = string.(FlexiChains.parameters(chain))

    # Use raw string r"..." to avoid double-backslash requirement for brackets
    regex = Regex("^" * base_name * "\\[(\\d+)\\]")
    matched_names = filter(n -> occursin(regex, n), all_names)

    if !isempty(matched_names)
        # Sort by the captured integer index
        sort!(matched_names, by = n -> parse(Int, match(regex, n).captures[1]))

        res_mat = zeros(Float64, N_samples, length(matched_names))

        for (idx, n) in enumerate(matched_names)
            val_obj = chain[Symbol(n)]
            raw = hasproperty(val_obj, :data) ? val_obj.data : collect(val_obj)

            for s in 1:N_samples
                v = raw[s]
                res_mat[s, idx] = (v isa AbstractVector) ? Float64(v[1]) : Float64(v)
            end
        end

        if size(res_mat, 2) == 1 && expected_len > 1 # Broadcast scalar to vector
            return repeat(res_mat, 1, expected_len)
        end

        return res_mat
    end

    if base_name in all_names
        val_obj = chain[Symbol(base_name)]
        raw_data = hasproperty(val_obj, :data) ? val_obj.data : collect(val_obj)

        mat_data = if eltype(raw_data) <: AbstractVector
             reduce(hcat, [vec(collect(v)) for v in raw_data])'
        else
             Matrix{Float64}(reshape(collect(raw_data), N_samples, :))
        end

        if size(mat_data, 2) == expected_len
            return mat_data
        elseif size(mat_data, 1) == expected_len && size(mat_data, 2) != expected_len
            return mat_data'
        elseif size(mat_data, 2) == 1
            return repeat(mat_data, 1, expected_len)
        else
            return reshape(vec(mat_data), N_samples, :)
        end
    end

    return zeros(Float64, N_samples, expected_len)
end

 


function generate_spectral_w_from_magnitude(freqs_x, freqs_y, magnitude_spectrum, M_rff_count)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Generates RFF weights (W) by sampling from a provided 2D magnitude spectrum.
    # Inputs: freqs_x, freqs_y, magnitude_spectrum, M_rff_count.
    # Outputs: A 2 x M_rff_count matrix for W.
    all_freqs_x = repeat(freqs_x, inner=length(freqs_y))
    all_freqs_y = repeat(freqs_y, outer=length(freqs_x))
    all_magnitudes = vec(magnitude_spectrum)

    probabilities = (all_magnitudes .+ 1e-9) ./ sum(all_magnitudes .+ 1e-9)

    sampled_indices = sample(1:length(probabilities), Weights(probabilities), M_rff_count, replace=true)

    Wfixed = Matrix{Float64}(undef, 2, M_rff_count)
    for i in 1:M_rff_count
        idx = sampled_indices[i]
        Wfixed[1, i] = all_freqs_x[idx] *  2pi
        Wfixed[2, i] = all_freqs_y[idx] *  2pi
    end

    return Wfixed
end
  
function generate_inducing_points(coords, M_inducing, seed=42; method="kmeans")
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Selects inducing point locations for sparse GP models.
    # Inputs: coords, M_inducing, seed, method.
    # Outputs: A matrix of inducing point coordinates.
    Random.seed!(seed)
    n_data = size(coords, 1)
    if M_inducing >= n_data
        return coords # If M >= N, just use all data points (becomes exact GP)
    end

    
    if method=="random"
        indices = sample(1:n_data, M_inducing, replace=false)
        return coords[indices, :]
    end

    if method=="kmeans"
        #   Identifies optimal inducing point locations using K-Means clustering.
        #   Essential for Sparse/FITC Gaussian Process models.
        # Inputs:
        #   coords: N x D matrix of spatiotemporal coordinates.
        #   M_inducing: Target number of inducing points.
        # Outputs:
        #   M x D matrix of cluster centroids.

        # Transpose for Clustering.jl compatibility
        data_matrix = Matrix(coords')
        
        # Execute K-Means
        clustering_result = kmeans(data_matrix, M_inducing, maxiter=200)
        
        # Extract centroids and transpose back
        inducing_points = clustering_result.centers'
        
        return inducing_points
    end

    if method=="furthest_point"

        # Initialize with a random point
        inducing_points_idx = [rand(1:n_data)]
        distances = fill(Inf, n_data)

        # Convert coords to the expected format for pairwise
        coords_matrix = permutedims(coords)

        for _ in 2:M_inducing
            # Calculate distances from all points to the newest inducing point
            last_added_idx = inducing_points_idx[end]
            new_distances = colwise(Euclidean(), coords_matrix, coords_matrix[:, last_added_idx])

            # Update minimum distances to any inducing point found so far
            distances = min.(distances, new_distances)

            # Find the point farthest from any existing inducing point
            farthest_idx = argmax(distances)
            push!(inducing_points_idx, farthest_idx)
        end

        return coords[inducing_points_idx, :], inducing_points_idx

    end

end


# Squared-exponential covariance function
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: A simple helper function to compute a squared-exponential covariance.
    # Inputs: D (distance matrix), phi (lengthscale), noise.
    # Outputs: A covariance matrix.
sqexp_cov_fn(D, phi, noise=1e-6) = exp.(-D^2 / phi) + LinearAlgebra.I * noise

# Exponential covariance function
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: A simple helper function to compute an exponential covariance.
    # Inputs: D (distance matrix), phi (lengthscale).
    # Outputs: A covariance matrix.
exp_cov_fn(D, phi) = exp.(-D / phi)

 


# Helper to create AR1 covariance matrix
function ar1_covariance_matrix(times::Vector{<:Real}, rho::Real, sigma_e::Real)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: A helper function to compute a full AR(1) covariance matrix based on the time differences between observations.
    # Inputs: times, rho, sigma_e.
    # Outputs: An AR(1) covariance matrix.
    n = length(times)
    T = typeof(rho) # Get the type of the parameters
    C = Matrix{T}(undef, n, n) # Initialize matrix with this type
    for i in 1:n
        for j in 1:n
            C[i, j] = sigma_e^2 * rho^abs(times[i] - times[j])
        end
    end
    return C
end

function ar1_cross_covariance_matrix(times_a::Vector{<:Real}, times_b::Vector{<:Real}, rho::Real, sigma_e::Real)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Computes an AR(1) cross-covariance matrix between two sets of time points.
    # Inputs: times_a, times_b, rho, sigma_e.
    # Outputs: The AR(1) cross-covariance matrix.
    na = length(times_a)
    nb = length(times_b)
    T = typeof(rho) # Get the type of the parameters
    C = Matrix{T}(undef, na, nb) # Initialize matrix with this type
    for i in 1:na
        for j in 1:nb
            C[i, j] = sigma_e^2 * rho^abs(times_a[i] - times_b[j])
        end
    end
    return C
end
 

function prepare_fft_grid(s_x, s_y, values; grid_res=64, pad_factor=2)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Prepares data for FFT analysis by gridding and zero-padding.
    # Inputs: s_x, s_y, values, grid_res, pad_factor.
    # Outputs: A tuple of the padded grid and the bounding box.
    xmin, xmax = minimum(s_x), maximum(s_x)
    ymin, ymax = minimum(s_y), maximum(s_y)

    n_limit = min(length(s_x), length(values))
    grid = zeros(grid_res, grid_res)

    for i in 1:n_limit
        ix = Int(floor((s_x[i] - xmin) / (xmax - xmin + 1e-6) * (grid_res - 1))) + 1
        iy = Int(floor((s_y[i] - ymin) / (ymax - ymin + 1e-6) * (grid_res - 1))) + 1
        grid[ix, iy] = values[i]
    end

    padded_res = grid_res * pad_factor
    padded_grid = zeros(padded_res, padded_res)

    start_idx = Int(grid_res / 2)
    padded_grid[start_idx:start_idx+grid_res-1, start_idx:start_idx+grid_res-1] .= grid

    return padded_grid, (xmin, xmax, ymin, ymax)
end
 



#####

 


function get_model_family(model_family::String)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Maps a string identifier to its corresponding concrete `AbstractBSTM_Family` type.
    # Inputs: model_family::String.
    # Outputs: An instance of a concrete subtype of AbstractBSTM_Family.
    family_key = lowercase(model_family)
    if haskey(BSTM_FAMILY_REGISTRY, family_key)
        return BSTM_FAMILY_REGISTRY[family_key]
    else
        error("Unknown model_family: '$model_family'. Supported families are: $(keys(BSTM_FAMILY_REGISTRY))")
    end
end

 

function get_model_parameters(m::DynamicPPL.Model)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Extracts parameter names from a Turing model instance by sampling from it once.
    # Inputs: m - A Turing model instance.
    # Outputs: A vector of parameter name strings.
    # Note: This is a fallback method for parameter discovery.
    # Directly extract names from a model instance sample as a fallback discovery method
    try
        raw_keys = keys(rand(m))
        return map(raw_keys) do k
            replace(string(k), r"[\(\)\"\:]" => "")
        end
    catch
        return String[]
    end
end


function _ensure_matrix(field, n_units, n_samples)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Converts a scalar, vector, or matrix into a matrix of a specified size.
    # Inputs: field, n_units, n_samples.
    # Outputs: A matrix of size [n_units x n_samples].
    if isnothing(field) || isempty(field)
        return zeros(Float64, n_units, n_samples)
    end

    if field isa Number
        return fill(Float64(field), n_units, n_samples)
    end

    if field isa AbstractVector
        if length(field) == n_units
            return repeat(reshape(field, n_units, 1), 1, n_samples)
        elseif length(field) == n_samples
            return repeat(reshape(field, 1, n_samples), n_units, 1)
        else
            return fill(Float64(field[1]), n_units, n_samples)
        end
    end

    if ndims(field) == 2
        local mat = Matrix{Float64}(field)
        local r, c = size(mat)

        if r == n_units && c == n_samples
            return mat
        elseif r == 1 && c == n_samples
            return repeat(mat, n_units, 1)
        elseif r == n_units && c == 1
            return repeat(mat, 1, n_samples)
        else
            return fill(mat[1,1], n_units, n_samples)
        end
    end

    return zeros(Float64, n_units, n_samples)
end

  
 

function _extract_beta_cov(all_names, chain, M, N_samples, alpha)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Extracts and summarizes posterior distributions for smoothed covariate effects.
    # Inputs: all_names, chain, M, N_samples, alpha.
    # Outputs: A vector of summarized effects or nothing.
    cov_matches = unique(map(m -> m.captures[1], filter(!isnothing, match.(r"beta_cov\[(\d+)\]", all_names))))
    
    if isempty(cov_matches)
        return nothing
    end

    cov_indices = sort(parse.(Int, cov_matches))
    
    results = []
    for k in cov_indices
        base_name = "beta_cov[$k]"
        # Use the robust get_params_vector helper to extract the full vector for this covariate group
        raw_vals = get_params_vector(chain, base_name, M.N_cat)
        
        if !all(raw_vals .== 0)
            summ = summarize_array(reshape(raw_vals', M.N_cat, 1, N_samples); alpha=alpha)
            push!(results, summ)
        end
    end

    return isempty(results) ? nothing : results
end


mutable struct ManifoldRegistry
    Xfixed_betas::Union{Nothing, Matrix{Float64}}
    mixed_eff_coeffs::Vector{Matrix{Float64}}
    s_eff_struct::Vector{Matrix{Float64}}
    s_eff_noisy::Vector{Matrix{Float64}}
    t_eff::Vector{Matrix{Float64}}
    u_eff::Matrix{Float64}
    basis_eff_accum::Matrix{Float64}
    st_eff_maps::Vector{Array{Float64, 3}}
    svc_slopes::Vector{Array{Float64, 3}}
    sv_surface::Matrix{Float64}
    n_samples::Int
    outcomes_N::Int
end
 


function summarize_array(samples::AbstractArray; alpha=0.05)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Computes summary statistics for posterior samples.
    # Inputs: samples (AbstractArray), alpha (for credible intervals).
    # Outputs: A NamedTuple of summary statistics.
    if isempty(samples) || all(isnan, samples)
        return (mean = Float64[], median = Float64[], std = Float64[], lower = Float64[], upper = Float64[])
    end

    dims = size(samples)
    sample_dim = length(dims)
    low_prob = alpha / 2.0
    high_prob = 1.0 - low_prob

    post_mean = dropdims(Statistics.mean(samples, dims=sample_dim), dims=sample_dim)
    post_median = dropdims(Statistics.median(samples, dims=sample_dim), dims=sample_dim)
    post_std = dropdims(Statistics.std(samples, dims=sample_dim), dims=sample_dim)
    
    low_bound = dropdims(mapslices(x -> Statistics.quantile(x, low_prob), samples, dims=sample_dim), dims=sample_dim)
    high_bound = dropdims(mapslices(x -> Statistics.quantile(x, high_prob), samples, dims=sample_dim), dims=sample_dim)

    function to_vector(x)
        if x isa AbstractArray
            return vec(collect(Float64, x))
        else
            return [Float64(x)]
        end
    end
    return (
        mean = to_vector(post_mean),
        median = to_vector(post_median),
        std = to_vector(post_std),
        lower = to_vector(low_bound),
        upper = to_vector(high_bound)
    )
end





function decompose_bstm_formula(formula_str::String)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Decomposes a bstm formula string into outcomes, fixed effects, and modules.
    # Inputs: formula_str::String.
    # Outputs: A NamedTuple containing the parsed components.
    parts = Base.split(formula_str, "~")
    lhs = strip(parts[1])
    rhs = strip(parts[2])

    outcomes = String[]
    if startswith(lhs, "[") && endswith(lhs, "]")
        content = lhs[2:end-1]
        outcomes = [strip(s) for s in Base.split(content, ",")]
    else
        outcomes = [lhs]
    end

    terms_raw = split_terms_at_depth(rhs, "+")
    modules = Dict{String, Any}()
    fixed_effects = String[]
    has_intercept = false

    for term in terms_raw
        t_clean = strip(term)
        if t_clean == "1"
            has_intercept = true
            continue
        end

        m_mod = match(r"(\w+)\((.*)\)", t_clean)
        if !isnothing(m_mod) && (lowercase(m_mod.captures[1]) in BSTM_MODULE_KEYWORDS)
            module_data = _parse_module_call(t_clean)
            
            key_parts = [string(module_data[:type])]
            
            if haskey(module_data, :variables) && !isempty(module_data[:variables])
                push!(key_parts, join(module_data[:variables], "_"))
            end

            if haskey(module_data, :params) && haskey(module_data[:params], :model)
                model_val = module_data[:params][:model]
                if model_val isa Symbol || model_val isa String
                    push!(key_parts, string(model_val))
                end
            end

            base_key = join(key_parts, "_")
            module_key = base_key
            counter = 1
            while haskey(modules, module_key)
                counter += 1
                module_key = base_key * "_$counter"
            end

            modules[module_key] = module_data
        else
            operator_found = ""
            for op in ["⊗", "otimes", "⊕", "oplus", "∘", "compose", "|>", "pipe"]
                if occursin(op, t_clean)
                    operator_found = op
                    break
                end
            end

            if !isempty(operator_found)
                norm_op = operator_found
                if operator_found == "otimes"; norm_op = "⊗"; end
                if operator_found == "oplus"; norm_op = "⊕"; end
                if operator_found == "compose"; norm_op = "∘"; end
                if operator_found == "pipe"; norm_op = "|>"; end

                sub_terms_str = _split_terms_at_depth(t_clean, operator_found)

                parsed_components = []
                for st in sub_terms_str
                    push!(parsed_components, _parse_module_call(st))
                end

                comp_keys = map(parsed_components) do comp
                    c_type = string(get(comp, :type, "lit"))
                    c_vars = join(get(comp, :variables, [get(comp, :value, "")]), "_")
                    c_model = string(get(get(comp, :params, Dict()), :model, ""))
                    return join(filter!(!isempty, [c_type, c_vars, c_model]), "_")
                end
                algebraic_key = "interaction_" * join(comp_keys, "_$(norm_op)_")

                modules[algebraic_key] = Dict{Symbol, Any}(
                    :type => :interaction_composition,
                    :components => parsed_components,
                    :operator => Symbol(norm_op)
                )
                continue
            end
            
            if !isempty(t_clean)
                push!(fixed_effects, t_clean)
            end
        end
    end

    return (outcomes=outcomes, modules=modules, fixed_effects=fixed_effects, has_intercept=has_intercept)
end
 


function process_interaction_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Processes algebraic interaction modules (e.g., `spatial() ⊗ temporal()`).
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies opt_dict in place.
    op = mod_data[:operator]
    if op == :⊗ || op == Symbol("otimes")
        has_structured_space = false
        has_structured_time = false

        for comp_data in mod_data[:components]
            if comp_data[:type] == :spatial
                model_name = string(get(comp_data[:params], :model, "iid"))
                if model_name != "iid"; has_structured_space = true; end
            elseif comp_data[:type] == :temporal
                model_name = string(get(comp_data[:params], :model, "iid"))
                if model_name != "iid"; has_structured_time = true; end
            end
        end

        if has_structured_space && has_structured_time; opt_dict[:model_st] = "IV";
        elseif !has_structured_space && has_structured_time; opt_dict[:model_st] = "II";
        elseif has_structured_space && !has_structured_time; opt_dict[:model_st] = "III";
        else; opt_dict[:model_st] = "I"; end
    elseif op == :⊕ || op == Symbol("oplus"); @warn "Direct sum operator '⊕' is not yet fully supported. Treating as additive effects.";
    elseif op == :∘ || op == Symbol("compose"); @warn "Composition operator '∘' is not yet fully supported. Treating as additive effects.";
    end
end

function process_observationprocess_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Processes the `observationprocess()` module to handle offsets, weights, etc.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies opt_dict in place.
    data = opt_dict[:data]
    params = mod_data[:params]

    _resolve_obs_param!(opt_dict, params, data, [:weights, :weight], :weights)
    _resolve_obs_param!(opt_dict, params, data, [:log_offsets, :offsets], :log_offset)
    _resolve_obs_param!(opt_dict, params, data, [:trials, :trial], :trials)
    
    # Process other observation-level parameters
    if get(params, :volatility, false) == true
        opt_dict[:use_sv] = true
        opt_dict[:M_rff_sigma] = get(params, :nbins, 20)
    end
    
    if haskey(params, :y_L); opt_dict[:y_lower_bound] = params[:y_L]; end
    if haskey(params, :y_U); opt_dict[:y_upper_bound] = params[:y_U]; end
    if haskey(params, :hurdle); opt_dict[:hurdle] = params[:hurdle]; end
end


function process_intercept_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Processes the `intercept()` module to handle the global intercept and its prior.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies opt_dict in place.
    opt_dict[:add_intercept] = true
    params = mod_data[:params]
    if haskey(params, :prior)
        prior_val = params[:prior]
        if prior_val isa String
            try
                opt_dict[:intercept_prior] = Main.eval(Meta.parse(prior_val))
            catch e
                @warn "Could not evaluate intercept prior: $prior_val. Using default. Error: $e"
            end
        else
            opt_dict[:intercept_prior] = prior_val
        end
    end
end

function process_spatial_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.0.2 (2026-06-29 18:10:00)
    # Purpose: Processes the `spatial()` module to extract the spatial index and adjacency matrix,
    #          and to prepare coordinates for continuous spatial models.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies opt_dict and mod_data in place.
    data = opt_dict[:data]

    if !isempty(mod_data[:variables])
        var_sym = Symbol(mod_data[:variables][1])
        if hasproperty(data, var_sym)
            opt_dict[:s_idx] = data[!, var_sym]
            opt_dict[:s_idx_var] = var_sym

            opt_dict[:s_N] = length(unique(data[!, var_sym]))
        else
            @warn "Spatial index variable ':$var_sym' not found in data."
        end
    end
 
    if haskey(mod_data[:params], :W)
        opt_dict[:W] = mod_data[:params][:W]
    end

    model_name = string(get(mod_data[:params], :model, "none"))
    if model_name in ["gp", "fitc", "svgp", "nystrom", "rff", "spde"]
        if haskey(opt_dict, :s_coord)
            s_N = get(opt_dict, :s_N, 0)
            if s_N > 0
                # Need unique coordinates per spatial unit
                df_coords = DataFrame(s_idx=opt_dict[:s_idx], s_x=opt_dict[:s_x], s_y=opt_dict[:s_y])
                unique_coords_df = combine(groupby(df_coords, :s_idx), :s_x => first => :s_x, :s_y => first => :s_y)
                sort!(unique_coords_df, :s_idx)
                if nrow(unique_coords_df) == s_N
                    mod_data[:params][:coords] = Matrix(unique_coords_df[!, [:s_x, :s_y]])
                else
                     @warn "Could not determine unique coordinates for spatial GP. Number of unique coordinates does not match s_N."
                end
            end
        elseif haskey(opt_dict, :centroids)
            s_N = get(opt_dict, :s_N, 0)
            if s_N > 0 && length(opt_dict[:centroids]) == s_N
                mod_data[:params][:coords] = reduce(hcat, [collect(c) for c in opt_dict[:centroids]])'
            else
                @warn "Centroids provided but length does not match s_N. Cannot use for spatial GP."
            end
        else
            @warn "Spatial GP model specified, but no coordinates (`s_x`, `s_y` in data or `centroids` kwarg) provided."
        end
    end
end



function bstm_sample(m::DynamicPPL.Model; nsample=10, testing=false)
    # v1.0.0 (2026-06-30)
    # Purpose: A utility function to streamline the process of sampling from a bstm model.
    #          It handles initialization, sampler selection, and execution.
    # Inputs: m - The Turing model instance.
    #         nsample - The number of posterior samples to draw.
    #         testing - If false, uses a fast MH() sampler for quick checks.
    #         sampler_override - Optionally provide a pre-configured sampler.
    # Outputs: A tuple containing the MCMC chain, initial values, the sampler used, and a model summary.

    model_summary = show_model(m)
    
    if !testing
        inits = get_inits(m)
        nadapt = max(Int(round(nsample * 0.25)), 200)
        os = get_optimal_sampler(m; adaptation_steps=nadapt)
    else
        inits =  get_inits(m; refine="none")
        os = MH()
    end

    chn = sample(m, os, nsample; initial_params=inits, progress=true, drop_warmup=true)
    
    plt = StatsPlots.plot(chn, seriestype=:traceplot)

    return chn, inits, os, model_summary, plt
 
end



function process_temporal_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Processes the `temporal()` module to handle time and season indices.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies opt_dict in place.
    data = opt_dict[:data]
    params = mod_data[:params]
    variables = mod_data[:variables]

    model_spec = get(params, :model, "ar1")
    if model_spec isa Tuple && length(model_spec) >= 2 # BSTM v06.1 Syntax: temporal(t_idx, u_idx, model=(ar1, cyclic))
        opt_dict[:model_time] = string(model_spec[1])
        opt_dict[:model_season] = string(model_spec[2])
    elseif model_spec isa String
        opt_dict[:model_time] = model_spec
    end

    if !isempty(mod_data[:variables])
        t_var_sym = Symbol(variables[1])
        if hasproperty(data, t_var_sym)
            time_opts = Dict(:time_method => get(params, :time_method, "regular"), :t_N => get(params, :t_N, nothing), :u_N => get(params, :u_N, nothing))
            filter!(p -> !isnothing(p.second), time_opts)
            tu_meta = assign_time_units(data[!, t_var_sym]; time_opts...)
            opt_dict[:t_idx] = tu_meta.t_idx
            opt_dict[:t_N] = tu_meta.tn
            opt_dict[:t_idx_var] = t_var_sym
            if length(variables) > 1
                u_var_sym = Symbol(variables[2])
                if hasproperty(data, u_var_sym); opt_dict[:u_idx] = data[!, u_var_sym]; opt_dict[:u_N] = length(unique(data[!, u_var_sym])); else; @warn "Seasonal index variable ':$u_var_sym' not found. Using derived seasonal index."; opt_dict[:u_idx] = tu_meta.u_idx; opt_dict[:u_N] = tu_meta.u_N; end
            else
                opt_dict[:u_idx] = tu_meta.u_idx
                opt_dict[:u_N] = tu_meta.u_N
            end
        else; @warn "Temporal index variable ':$t_var_sym' not found in data."; end
    end
end

function process_smooth_module!(opt_dict, mod_data, basis_matrices_registry, manifolds_registry)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Processes `smooth()` modules for non-linear covariate effects.
    # Inputs: opt_dict, mod_data, basis_matrices_registry, manifolds_registry.
    # Outputs: Modifies registries and opt_dict in place.
    data = opt_dict[:data]
    params = mod_data[:params]
    model_param = get(params, :model, "pspline")

    if model_param isa NamedTuple
        vars = mod_data[:variables]
        n_vars = length(vars)
        
        if n_vars >= 2
            B_list, Q_list, nbins_list = [], [], []
            for var_str in vars
                var_sym = Symbol(var_str)
                model_i = string(get(model_param, var_sym, "pspline"))
                nbins_i = get(get(params, :nbins, (;)), var_sym, get(params, :nbins, 20))
                degree_i = get(get(params, :degree, (;)), var_sym, get(params, :degree, 3))
                ls_i = get(get(params, :lengthscale, (;)), var_sym, get(params, :lengthscale, nothing))
                
                kwargs_i = Dict{Symbol, Any}()
                if !isnothing(ls_i) kwargs_i[:lengthscale] = ls_i end

                vals_i = data[!, var_sym]
                push!(B_list, bstm_smooth_basis_1D(model_i, vals_i, nbins_i, degree_i; kwargs_i...))
                
                template_type = if model_i in ["pspline", "rw2"]; :rw2 elseif model_i == "rw1"; :rw1 else :iid end
                push!(Q_list, build_structure_template(template_type, nbins_i).matrix)
                push!(nbins_list, nbins_i)
            end

            B_final = B_list[end]
            for i in (n_vars-1):-1:1; B_final = kron(B_final, B_list[i]); end

            Q_final = Q_list[1]
            n_units_total = nbins_list[1]
            for i in 2:n_vars
                n_i = nbins_list[i]
                Q_i = Q_list[i]
                Q_final = kron(I(n_i), Q_final) + kron(Q_i, I(n_units_total))
                n_units_total *= n_i
            end
            
            reg_key = Symbol(join(vars, "_"))
            basis_matrices_registry[reg_key] = B_final
            mod_data[:params][:model] = "tensor_product_smooth"
            mod_data[:params][:Q_tensor] = Q_final
            return
        end
    end

    model_str = string(model_param)
    
    basis_models = ["pspline", "bspline", "tps", "rff", "fft", "moran", "spherical", "barycentric", "decay", "wavelet", "linear", "invdist", "kriging"]
    continuous_kernel_models = ["gp", "fitc", "svgp", "nystrom", "warp", "spde", "exponentialdecay"]

    if model_str in basis_models
        if !isempty(mod_data[:variables])
            nb = get(mod_data[:params], :nbins, get(mod_data[:params], :m_rff, 20))
            reg_key = Symbol(join(mod_data[:variables], "_"))
            
            if all(hasproperty(data, Symbol(v)) for v in mod_data[:variables])
                n_vars = length(mod_data[:variables])
                if n_vars == 1
                    v_vec = data[!, Symbol(mod_data[:variables][1])]
                    basis_matrices_registry[reg_key] = bstm_smooth_basis_1D(model_str, v_vec, nb, get(mod_data[:params], :degree, 3); mod_data[:params]...)
                else
                    c_mat = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
                    if n_vars == 2; basis_matrices_registry[reg_key] = bstm_smooth_basis_2D(model_str, c_mat, nb; mod_data[:params]...);
                    elseif n_vars == 3; basis_matrices_registry[reg_key] = bstm_smooth_basis_3D(model_str, c_mat, nb; mod_data[:params]...);
                    elseif n_vars == 4; basis_matrices_registry[reg_key] = bstm_smooth_basis_4D(model_str, c_mat, nb; mod_data[:params]...);
                    end
                end
            end
        end
    elseif model_str in continuous_kernel_models
        return
    else
        if !haskey(opt_dict, :structured_smooths); opt_dict[:structured_smooths] = []; end

        vars = mod_data[:variables]
        nbins_param = get(mod_data[:params], :nbins, 20)
        smooth_name = Symbol(join(vars, "_"))

        if occursin("⊗", model_str) # 2D Interaction
            if length(vars) != 2; @warn "Interaction smooth on $(join(vars, ",")) requires exactly 2 variables."; return; end
            
            parts = split_terms_at_depth(model_str, "⊗")
            model1_str, model2_str = strip(parts[1]), strip(parts[2])
            nbins1, nbins2 = nbins_param isa Tuple ? nbins_param : (nbins_param, nbins_param)

            _, idx1 = apply_discretization_logic(data[!, Symbol(vars[1])], nbins1)
            _, idx2 = apply_discretization_logic(data[!, Symbol(vars[2])], nbins2)

            Q1 = build_structure_template(Symbol(model1_str), nbins1).matrix
            Q2 = build_structure_template(Symbol(model2_str), nbins2).matrix
            
            push!(opt_dict[:structured_smooths], (
                name = smooth_name,
                indices = (idx1 .- 1) .* nbins2 .+ idx2,
                Q_template = kron(Q2, Q1),
                n_units = nbins1 * nbins2,
                sigma_prior = Exponential(1.0)
            ))
        else # 1D GMRF Smooth
            if length(vars) != 1; @warn "1D structured smooth on $(join(vars, ",")) requires exactly 1 variable."; return; end

            _, indices = apply_discretization_logic(data[!, Symbol(vars[1])], nbins_param)
            n_units = length(unique(indices))

            push!(opt_dict[:structured_smooths], (
                name = smooth_name,
                indices = indices,
                Q_template = build_structure_template(Symbol(model_str), n_units).matrix,
                n_units = n_units,
                sigma_prior = Exponential(1.0)
            ))
        end
    end
end

 
function process_dynamics_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.0.1 (2026-06-28 21:30:00)
    # Purpose: Processes the `dynamics()` module.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies opt_dict in place.
end
function process_eigen_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.0.1 (2026-06-28 21:30:00)
    # Purpose: Processes the `eigen()` module for PCA-based factor models.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies mod_data in place.
    params = mod_data[:params]
    vars = mod_data[:variables]
    n_vars = length(vars)
    n_factors = get(params, :n_factors, 1)

    if n_factors > n_vars
        @warn "Number of factors ($n_factors) for eigen() module cannot be greater than number of variables ($n_vars). Setting to $n_vars."
        n_factors = n_vars
    end

    ltri_mask = [r >= c for r in 1:n_vars, c in 1:n_factors]
    ltri_indices = findall(vec(ltri_mask))

    mod_data[:params][:ltri_indices] = ltri_indices
    mod_data[:params][:n_factors] = n_factors
end

function process_mixed_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Processes the `mixed()` module for random effects.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies mod_data in place.
    data = opt_dict[:data]
    vars = mod_data[:variables]
    
    if length(vars) < 2
        @warn "The mixed() module requires at least two variables: mixed(effect_var, group_var). Skipping."
        return
    end
    
    effect_var_str = vars[1]
    group_var_str = vars[2]
    group_var_sym = Symbol(group_var_str)
    
    if !hasproperty(data, group_var_sym)
        @warn "Grouping variable ':$group_var_sym' for mixed() module not found in data. Skipping."
        return
    end
    
    group_data = data[!, group_var_sym]
    levels = unique(group_data)
    group_map = Dict(v => i for (i, v) in enumerate(levels))
    indices = [group_map[v] for v in group_data]
    
    mod_data[:params][:indices] = indices
    mod_data[:params][:n_cat] = length(levels)
    
    mod_data[:variables] = [effect_var_str]
end

function process_nested_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Processes the `nested()` supervisor module for multi-fidelity models.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies opt_dict in place.
    opt_dict[:model_arch] = "multifidelity"
    if !haskey(opt_dict, :nested_manifolds); opt_dict[:nested_manifolds] = Dict{Symbol, Any}(); end
    
    var = Symbol(mod_data[:variables][1])
    params = mod_data[:params]
    sub_formula = get(params, :formula, "")
    data_source_sym = get(params, :data_source, :data)
    
    if !haskey(opt_dict, data_source_sym)
        @warn "Data source ':$data_source_sym' for nested module on '$var' not found. Skipping."
        return
    end
    
    sub_data = opt_dict[data_source_sym]
    
    sub_config = Dict{Symbol, Any}()
    sub_config[:data] = sub_data
    sub_metadata = decompose_bstm_formula(sub_formula)
    
    spatial_mod_data_pair = filter(m -> m.second[:type] == :spatial, sub_metadata.modules)
    if !isempty(spatial_mod_data_pair)
        spatial_mod_data = first(spatial_mod_data_pair).second
        process_spatial_module!(sub_config, spatial_mod_data, registries, hyperpriors)
        sub_config[:model_space] = get(spatial_mod_data[:params], :model, "bym2")
        sub_config[:s_Q_template] = build_structure_template(Symbol(sub_config[:model_space]), sub_config[:s_N]; W=get(sub_config, :W, nothing))
    end
    
    fixed_effects_formula_part = join(sub_metadata.fixed_effects, " + ")
    if sub_metadata.has_intercept
        fixed_effects_formula_part = isempty(fixed_effects_formula_part) ? "1" : "1 + " * fixed_effects_formula_part
    end
    if !isempty(fixed_effects_formula_part)
        sub_config[:Xfixed] = create_fixed_design(fixed_effects_formula_part, sub_data)
    end
    
    opt_dict[:nested_manifolds][var] = sub_config
end


# must come aftyer the above
const MODULE_PROCESSORS = Dict{Symbol, Function}(
    :spatial => process_spatial_module!,
    :temporal => process_temporal_module!,
    :smooth => process_smooth_module!,
    :interaction_composition => process_interaction_module!,
    :intercept => process_intercept_module!,
    :observationprocess => process_observationprocess_module!,
    :nested => process_nested_module!,
    :eigen => process_eigen_module!,
    :mixed => process_mixed_module!,
    :dynamics => process_dynamics_module!
)


 
function model_results_comprehensive(model, chain; n_samples=100, alpha=0.05)
    println("--- Starting Comprehensive Model Reporting ---")

    # # Metadata and Architecture Extraction
    # Rationale: Extracting the technical configuration from the model arguments.
    M = model.args.M
    y_obs = M.y_obs
    raw_arch = get(M, :model_arch, "univariate")
    model_family = get(M, :model_family, "gaussian")

    # Mapping the configuration string to the architectural dispatch types for reconstruction
    arch_type = if raw_arch == "univariate"
        UnivariateArchitecture()
    elseif raw_arch == "multivariate"
        MultivariateArchitecture()
    elseif raw_arch == "multifidelity" || raw_arch == "nested"
        MultifidelityArchitecture()
    else
        UnivariateArchitecture()
    end

    # # Latent Manifold Reconstruction
    # Rationale: Executes the architectural-specific reconstruction of every latent component.
    # The alpha parameter controls the width of the credible intervals (default 95%).
    res = _reconstruct(arch_type, "model_results", chain, M, nothing, alpha)

    # # Performance Metric Assessment
    # Rationale: Metrics are calculated using the posterior predictive mean (denoised expectation).
    y_pred = res.predictions_denoised.mean

    # Flatten for metric consistency across Univariate and Multivariate responses
    y_obs_flat = vec(collect(y_obs))
    y_pred_flat = vec(collect(y_pred))

    # Filter valid indices for calculation to handle missing values or NaNs in observed data
    valid_idx = findall(x -> !isnan(x) && !isnothing(x), y_obs_flat)

    rmse_val = 0.0
    r_pearson = 0.0

    if !isempty(valid_idx)
        obs_v = y_obs_flat[valid_idx]
        pred_v = y_pred_flat[valid_idx]

        # Root Mean Square Error (RMSE)
        rmse_val = sqrt(Statistics.mean((obs_v .- pred_v).^2))

        # Pearson Correlation Coefficient (r)
        try
            r_pearson = Statistics.cor(obs_v, pred_v)
        catch
            # Handle cases with zero variance in predictions to prevent domain errors
            r_pearson = 0.0
        end
    end

    # FlexiChains store timing data in _metadata, not info.
    # Rationale: Direct access to _metadata keys via Symbol lookup.
    sampling_time = chain._metadata.sampling_time[]

    # Standardizing Diagnostic Extraction via DataFrames
    # Rationale: summarystats(chain) returns a FlexiSummary which is converted to a DataFrame
    # to ensure column-based indexing for rhat and ess_bulk is stable and type-safe.
    sum_stats_df = DataFrame(MCMCChains.summarystats(chain))
    
    mean_rhat = 1.0
    min_ess = 0.0

    try
        # Extraction of diagnostic vectors
        rhat_vector = sum_stats_df[!, :rhat]
        ess_bulk_vector = sum_stats_df[!, :ess_bulk]
        
        # Robust aggregation ignoring NaNs
        mean_rhat = Statistics.mean(filter(!isnan, rhat_vector))
        min_ess = Statistics.minimum(filter(!isnan, ess_bulk_vector))
    catch
        # Fallback to standard ess if ess_bulk is absent in the DataFrame schema
        try
            ess_alt = sum_stats_df[!, :ess]
            min_ess = Statistics.minimum(filter(!isnan, ess_alt))
        catch
            mean_rhat = 1.0
            min_ess = 0.0
        end
    end

    waic_val = get(res, :waic, 0.0)

    ess_rate = round(min_ess/sampling_time, digits=2)

    # # Displaying the Report
    println("\n--- Model Metadata ---")
    println("Architecture:     ", raw_arch)
    println("Family:           ", model_family)
    println("Space Component:  ", get(M, :model_space, "none"))
    println("Time Component:   ", get(M, :model_time, "none"))
    println("Seasonal Component: ", get(M, :model_season, "none"))

    println("\n--- Performance Metrics ---")
    println("Compute Time:     ", round(sampling_time, digits=2), " seconds")
    println("RMSE:             ", round(rmse_val, digits=4))
    println("Pearson r:        ", round(r_pearson, digits=4))
    println("WAIC Score:       ", round(waic_val, digits=2))

    println("\n--- MCMC Diagnostics ---")
    println("Mean R-hat:       ", round(mean_rhat, digits=4))
    println("Minimum ESS:      ", round(min_ess, digits=2))
    println("ESS per second:    ", ess_rate)

    # # Final Registry Object Assembly
    # Rationale: Returning the full set of metrics and reconstructed latent fields.
    return (
        metrics = (
            rmse = rmse_val, 
            r_pearson = r_pearson, 
            waic = waic_val, 
            rhat = mean_rhat, 
            ess = min_ess, 
            ess_rate = ess_rate,
            sampling_time = sampling_time
        ),
        pstats = res,
        y_obs = y_obs,
        model_family = model_family,
        arch = arch_type
    )
end
 

 

function create_prediction_surface(
    data_df::DataFrame, 
    au_obj::NamedTuple, 
    tu_obj::NamedTuple, 
    covariate_vars::Vector{Symbol}; 
    iterations::Int = 3
)
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: A utility function that creates a prediction surface (a full spatiotemporal grid)
    #          and imputes missing covariate values using an iterative spatiotemporal neighborhood
    #          averaging method.
    # Inputs: data_df, au_obj, tu_obj, covariate_vars, iterations.
    # Outputs: A DataFrame representing the complete prediction surface with imputed covariates.

    # # 1. Grid Construction
    # Establishing a full Cartesian product of spatial centroids and temporal slices
    s_N = length(au_obj.centroids)
    t_N = tu_obj.tn
    
    # Construct initial empty surface
    surface = DataFrame(
        s_idx = repeat(1:s_N, inner = t_N),
        t_idx = repeat(1:t_N, outer = s_N)
    )
    
    # Map coordinates from centroids
    surface.s_x = [au_obj.centroids[i][1] for i in surface.s_idx]
    surface.s_y = [au_obj.centroids[i][2] for i in surface.s_idx]
    
    # # 2. Feature Mapping
    # Transferring existing observed covariates to the prediction grid
    # Observed points are averaged per spatio-temporal cell
    data_agg = combine(
        groupby(data_df, [:s_idx, :t_idx]), 
        covariate_vars .=> mean .=> covariate_vars
    )
    
    surface = leftjoin(surface, data_agg, on = [:s_idx, :t_idx])
    
    # # 3. Safe Spatiotemporal Imputation
    # Rationale: Filling gaps using spatial adjacency and temporal neighbors.
    # Updates are collected in a buffer to prevent mutation of the DataFrame during iteration.
    
    adj_matrix = au_obj.W
    
    for iter_idx in 1:iterations
        # Create a copy of the current state to serve as the source for this pass
        current_state = copy(surface)
        
        for cov in covariate_vars
            # Identify rows needing imputation
            missing_mask = isnothing.(current_state[!, cov]) .| isnan.(current_state[!, cov])
            
            if !any(missing_mask)
                continue
            end
            
            # Initialize update buffer for the current covariate
            updates = copy(current_state[!, cov])
            
            # Group by spatial unit to efficiently access temporal neighbors
            surface_groups = groupby(current_state, :s_idx)
            
            for (row_idx, is_missing) in enumerate(missing_mask)
                if !is_missing
                    continue
                end
                
                s_i = current_state.s_idx[row_idx]
                t_i = current_state.t_idx[row_idx]
                
                # Accumulate neighborhood values
                neighbor_vals = Float64[]
                
                # # A. Temporal Context (Previous and Next slices in same unit)
                # Accessing the group for current spatial unit s_i
                unit_data = surface_groups[(s_idx = s_i,)]
                t_prev = t_i > 1 ? unit_data[t_i - 1, cov] : NaN
                t_next = t_i < t_N ? unit_data[t_i + 1, cov] : NaN
                
                if !isnan(t_prev) push!(neighbor_vals, t_prev) end
                if !isnan(t_next) push!(neighbor_vals, t_next) end
                
                # # B. Spatial Context (Same slice in adjacent units)
                neighbors_idx = findall(x -> x > 0, adj_matrix[s_i, :])
                for nb_s in neighbors_idx
                    # Find the specific row in current_state for (nb_s, t_i)
                    # Rationale: In a full Cartesian grid, index = (nb_s - 1) * t_N + t_i
                    nb_row_idx = (nb_s - 1) * t_N + t_i
                    val = current_state[nb_row_idx, cov]
                    if !isnothing(val) && !isnan(val)
                        push!(neighbor_vals, val)
                    end
                end
                
                # # C. Apply Mean Imputation
                if !isempty(neighbor_vals)
                    imputed_val = mean(neighbor_vals)
                    
                    # Factor Snapping: Rounding if the column appears to be integer-based
                    if all(x -> x == floor(x), filter(!isnan, current_state[!, cov]))
                        imputed_val = round(imputed_val)
                    end
                    
                    updates[row_idx] = imputed_val
                end
            end
            
            # Apply the collected updates to the surface after the row loop
            surface[!, cov] = updates
        end
    end
    
    # # 4. Final Verification
    # Fallback to global means for any remaining orphaned cells
    for cov in covariate_vars
        final_missing = isnothing.(surface[!, cov]) .| isnan.(surface[!, cov])
        if any(final_missing)
            global_avg = mean(filter(!isnan, filter(!isnothing, surface[!, cov])))
            surface[final_missing, cov] .= isnan(global_avg) ? 0.0 : global_avg
        end
    end
    
    return surface
end



function convert_advi_to_reconstruct_format(msol, model::DynamicPPL.Model, n_samples::Int=500)
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: A utility to convert the output of a `Turing.jl` ADVI (variational inference) run
    #          into a format compatible with the standard post-processing engine. It samples from
    #          the variational posterior and formats the output as a `FlexiChain`.
    # Inputs: msol (VI result), model, n_samples.
    # Outputs: A NamedTuple containing the FlexiChain and the raw reconstructed samples.
    # Sample from the variational distribution
    samples_vec = rand(msol, n_samples)

    # Safety check for extraction: ADVI samples often wrap data in .nt, .data, or are ParamsWithStats objects
    function get_data_obj(s)
        if hasproperty(s, :nt) return s.nt
        elseif hasproperty(s, :data) return s.data
        elseif hasproperty(s, :params) return s.params # Handle ParamsWithStats internal field
        else return s end
    end

    # Peek at the first sample to discover parameter keys
    first_samp = get_data_obj(samples_vec[1])
    all_keys = keys(first_samp)

    unique_bases = Set{Symbol}()
    for k in all_keys
        m = match(r"^([^\\\\[]+)", string(k))
        if m !== nothing
            push!(unique_bases, Symbol(m.captures[1]))
        end
    end

    reconstruct_samples = map(samples_vec) do samp
        nt = get_data_obj(samp)
        sample_params = Dict{Symbol, Any}()
        for base_sym in unique_bases
            base_str = string(base_sym)
            col_keys = filter(k -> string(k) == base_str || startswith(string(k), "$base_str["), all_keys)
            if length(col_keys) == 1 && string(first(col_keys)) == base_str
                sample_params[base_sym] = nt[first(col_keys)]
            else
                # Sort indexed keys like x[1], x[10], x[2] into numerical order
                sorted_keys = sort(collect(col_keys), by = k -> begin
                    m_idx = match(r"\\\\[(\d+)\\\\]", string(k))
                    m_idx !== nothing ? parse(Int, m_idx.captures[1]) : 0
                end)
                sample_params[base_sym] = [nt[k] for k in sorted_keys]
            end
        end
        return (; sample_params...)
    end

    # Create a FlexiChain for standard diagnostics
    formatted_dicts = map(samples_vec) do samp
        nt = get_data_obj(samp)
        Dict(FlexiChains.Parameter(Symbol(k)) => v for (k, v) in pairs(nt))
    end
    chn = FlexiChains.FlexiChain{Symbol}(n_samples, 1, formatted_dicts)

    return (chain=chn, reconstruct_samples=reconstruct_samples)
end



function convert_optim_to_reconstruct_format(optim_result, model, n_samples::Int=500; use_hessian=true, external_hessian=nothing)
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: A utility to convert a point estimate from an optimization routine (e.g., MAP, MLE)
    #          into a sample-based format for post-processing. It uses a Laplace approximation
    #          (sampling from a multivariate normal centered at the mode) if the Hessian is available.
    #          If a Hessian is not available, it adds a small amount of Gaussian noise to the point
    #          estimate to create a sample distribution.
    # Inputs: optim_result, model, n_samples, and optional parameters.
    # Outputs: A NamedTuple containing the FlexiChain and the raw reconstructed samples.
    point_est_constrained = optim_result.params # This is the named tuple of constrained parameters

    reconstruct_samples_namedtuple = [] # Store NamedTuple for easier handling

    # Determine which Hessian to use
    H_to_use = nothing
    if external_hessian !== nothing
        H_to_use = external_hessian
    elseif hasproperty(optim_result, :hessian)
        H_to_use = optim_result.hessian
    end

    # Attempt Hessian-based sampling if enabled and a Hessian is available from either source
    if use_hessian && H_to_use !== nothing
        try
            # Get the unconstrained minimizer (mu_unconstrained)
            mu_unconstrained = optim_result.minimizer
            H = H_to_use

            # Ensure H is symmetric and compute its inverse for the covariance matrix
            Sigma = inv(Symmetric(Matrix(H) + Diagonal(fill(1e-6, size(H, 1)))))
            
            dist = MvNormal(mu_unconstrained, Sigma)

            # Generate n_samples of unconstrained parameters
            unconstrained_samples_matrix = rand(dist, n_samples)

            # Prepare template for conversion
            vi_template = DynamicPPL.VarInfo(model)

            for i in 1:n_samples
                sample_unconstrained_vec = unconstrained_samples_matrix[:, i]
                vi_current_sample = deepcopy(vi_template)
                DynamicPPL.setlink!(vi_current_sample, sample_unconstrained_vec)
                
                # Convert back to constrained named tuple
                constrained_sample_params = DynamicPPL.vi_to_params(vi_current_sample, model)
                push!(reconstruct_samples_namedtuple, constrained_sample_params)
            end

        catch e
            @warn "Failed to compute covariance from Hessian or convert samples, falling back to adding noise: $e"
            use_hessian = false 
        end
    end

    # Fallback to noise-based sampling
    if !use_hessian || isempty(reconstruct_samples_namedtuple)
        all_keys = keys(point_est_constrained)
        unique_bases = Set{Symbol}()
        for k in all_keys
            m = match(r"^([^\\[]+)", string(k))
            if m !== nothing
                push!(unique_bases, Symbol(m.captures[1]))
            end
        end

        reconstruct_samples_namedtuple = map(1:n_samples) do _
            sample_params_dict = Dict{Symbol, Any}() 
            for base_sym in unique_bases
                base_str = string(base_sym)
                col_keys = filter(k -> string(k) == base_str || startswith(string(k), "$base_str["), all_keys)

                if length(col_keys) == 1 && string(first(col_keys)) == base_str
                    val = point_est_constrained[first(col_keys)]
                    if val isa AbstractVector
                        sample_params_dict[base_sym] = val .+ randn() * 1e-4
                    else
                        sample_params_dict[base_sym] = val + randn() * 1e-4
                    end
                else
                sorted_keys = sort(collect(col_keys), by = k -> begin # Corrected regex
                    m_idx = match(r"\[(\d+)\]", string(k))
                        m_idx !== nothing ? parse(Int, m_idx.captures[1]) : 0
                    end)
                    sample_params_dict[base_sym] = [point_est_constrained[k] + randn() * 1e-4 for k in sorted_keys]
                end
            end
            return (; sample_params_dict...)
        end
    end

    # 4. Format into a FlexiChain
    # IMPORTANT: Store ONLY the base symbols (e.g., :s_icar as a vector).
    # FlexiChains will automatically expand these into indexed names for summary statistics,
    # avoiding duplicate key errors while allowing _reconstruct to find the full vector.
    formatted_dicts = map(1:n_samples) do i
        samp = reconstruct_samples_namedtuple[i]
        d = Dict{FlexiChains.Parameter, Any}()
        for k in keys(samp)
            d[FlexiChains.Parameter(k)] = samp[k]
        end
        return d
    end

    chn = FlexiChains.FlexiChain{Symbol}(n_samples, 1, formatted_dicts)

    return (chain=chn, reconstruct_samples=reconstruct_samples_namedtuple)
end
 
    



    

    
function generate_sim_data(s_N=25, t_N=10; rndseed=42)
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: A utility function for generating a standardized simulated spatiotemporal dataset with
    #          known underlying trends, seasonal effects, and covariate relationships.
    # Inputs: s_N (number of spatial units), t_N (number of time units), rndseed.
    # Outputs: A NamedTuple containing the simulated DataFrame and metadata.
    # Note: This function is crucial for testing and validating model implementations.
    Random.seed!(rndseed)
    n_total = s_N * t_N

    # 1. Spatial Coordinates (Unit Level)
    unique_pts = [(rand() * 100.0, rand() * 100.0) for _ in 1:s_N]
    s_coord_tuple = repeat(unique_pts, inner=t_N)
    s_x = getindex.(s_coord_tuple, 1)
    s_y = getindex.(s_coord_tuple, 2)

    # 2. Temporal/Seasonal Indices
    t_v = repeat(collect(1:t_N), outer=s_N) .+ (rand(n_total) .* 0.05)
    t_idx = repeat(1:t_N, outer=s_N)
    u_N = 12
    u_idx = mod1.(1:n_total, u_N)

    # 3. Latent Fields
    period = 12.0
    trend = 0.05 .* t_v
    seasonal = 0.8 .* cos.(2π .* t_v ./ period)

        # Covariate Generation (W1, W2, W3)
    # These simulate continuous predictors with some shared latent signal Z
    Z = randn(n_total)
    W1_obs = 0.5 .* sin.(t_v ./ 5.0) .+ 0.5 .* Z .+ (randn(n_total) .* 0.1)
    W2_obs = 0.5 .* cos.(t_v ./ 5.0) .- 0.3 .* Z .+ (randn(n_total) .* 0.2)
    W3_obs = 0.2 .* (t_v ./ t_N) .+ 0.1 .* Z .+ (randn(n_total) .* 0.3)
 
    # Mosaic/Cluster Effects
    s_clusters = mod1.(1:s_N, 5)
    cluster_effects = [-2.5, -1.0, 0.0, 1.0, 2.5]
    cluster_assignments_full = repeat(s_clusters, inner=t_N)
    spatial_effect = cluster_effects[cluster_assignments_full]

    # 4. Response Construction
    sigma_y = 0.15
    observation_error = sigma_y .* randn(n_total)
    eta = 1.0 .+ spatial_effect .+ trend .+ seasonal .+ observation_error

    y_binary = Int.(eta .> (mean(eta) + 0.5))
    y_counts = abs.(Int.(round.(exp.(eta)))) # Poisson-friendly counts

    weights = ones(Float64, n_total)
    trials = ones(Int, n_total)

    # Fixed Effects Design Matrix (Standard Intercept-only approach)
    Xfixed = ones(Float64, n_total, 1)

    # a factorial variable
    reg_indices = mod1.(1:n_total, 4)
    reg_levels = ["North", "South", "East", "West"]
    reg = reg_levels[reg_indices]

    # reformat simulated data into a rectangular dataframe (or namedarray the internal default):
    data_df = DataFrame(
        y = y_counts,  # y-variable
        y_obs = eta,
        # Ensuring coordinates are aligned with the flattened observation vector
        s_idx = repeat(1:s_N, inner=t_N),
        s_coord = s_coord_tuple,
        s_x = s_x,
        s_y = s_y,
        t_v = t_v,
        t_coord = vec(t_idx),   # time index
        u_idx = u_idx,
        u_v = seasonal,
        log_offset = zeros(n_total),
        region = categorical(reg),  # would make sure it is used as a factorial variable or in the model statement: Fixed(reg)
        z = Z,  # continuous covariate
        w1 = W1_obs, # more covariates 
        w2 = W2_obs,
        w3 = W3_obs,
        cluster_assignments = cluster_assignments_full,

        y_binary = y_binary,
        y_counts = y_counts,

        Xfixed = Xfixed,
        weights = weights,
        trials = trials     
    )

    return (
        data_df = data_df,
        s_coord = s_coord_tuple,
        metadata = (
            s_N = s_N,
            t_N = t_N,
            u_N = u_N,
            n_total = n_total 
        )
    )   

end




function scottish_lip_cancer_data_spacetime(n_years::Int=20, spatial_expansion::Float64=1.5, temporal_expansion::Float64=1.5; rndseed::Int=42, recreate::Bool=false)
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: A data factory that generates a spatiotemporal version of the classic Scottish Lip
    #          Cancer dataset. It also creates an expanded "nested" dataset for testing
    #          multi-fidelity models.
    # Inputs: n_years, spatial_expansion, temporal_expansion, rndseed, recreate.
    # Outputs: A tuple containing the primary and nested datasets.
    # BSTM Standard Data Factory v28.3.0
    # Timestamp: 2025-12-01 15:00:00
    # Rationale: Resolving symmetry errors in adjacency matrix construction and scoping errors for derived variables.

    cache_path = "data/scottish_lip_cancer_cache.jld2"

    # Check for existing cache and bypass logic unless recreate is explicitly true
    if isfile(cache_path) && !recreate
        println("Loading cached dataset from: ", cache_path)
        data_bundle = JLD2.load(cache_path)
        return (data_bundle["primary"], data_bundle["nested"])
    end

    println("Generating new spatiotemporal dataset...")
    Random.seed!(rndseed)

    # ##########################################################################
    # PRIMARY DATASET CONSTRUCTION (56 Districts)
    # scottish lip cancer data to a space-time version
    # ##########################################################################

    n_districts = 56

    # Canonical neighbor list (undirected counties)
    neighbor_list = [
        [5, 9, 11, 19], [7, 10], [6, 12], [18, 20, 28], [1, 11, 12, 13, 19],
        [3, 8], [2, 10, 13, 16, 17], [6], [1, 11, 17, 19, 23, 29], [2, 7, 16, 22],
        [1, 5, 9, 12], [3, 5, 11], [5, 7, 17, 19], [31, 32, 35], [25, 29, 50],
        [7, 10, 17, 21, 22, 29], [7, 9, 13, 16, 19, 29], [4, 20, 28, 33, 55, 56], [1, 5, 9, 13, 17], [4, 18, 55],
        [16, 29, 50], [10, 16], [9, 29, 34, 36, 37, 39], [27, 30, 31, 44, 47, 48, 55, 56], [15, 26, 29],
        [26, 29, 42, 43], [24, 31, 32, 55], [4, 18, 33, 45], [9, 15, 16, 17, 21, 23, 25, 26, 34, 43, 50], [24, 38, 42, 44, 45, 56],
        [14, 24, 27, 32, 35, 46, 47], [14, 27, 31, 35], [18, 28, 45, 56], [23, 29, 39, 40, 42, 43, 51, 52, 54], [14, 31, 32, 37, 46],
        [23, 37, 39, 41], [23, 35, 36, 41, 46], [30, 42, 44, 49, 51, 54], [23, 34, 36, 40, 41], [34, 39, 41, 49, 52],
        [36, 37, 39, 40, 46, 49, 53], [26, 30, 34, 38, 43, 51], [26, 29, 34, 42], [24, 30, 38, 48, 49], [28, 30, 33, 56],
        [31, 35, 37, 41, 47, 53], [24, 31, 46, 48, 49, 53], [24, 44, 47, 49], [38, 40, 41, 44, 47, 48, 52, 53, 54], [15, 21, 29],
        [34, 38, 42, 54], [34, 40, 49, 54], [41, 46, 47, 49], [34, 38, 49, 51, 52], [18, 20, 24, 27, 56], [18, 24, 30, 33, 45, 55]
    ]

    # Construct and enforce symmetry for adjacency matrix W
    W_raw = spzeros(Int, n_districts, n_districts)
    for i in 1:n_districts
        for nb in neighbor_list[i]
            W_raw[i, nb] = 1
        end
    end
    # Symmetric enforcement: W_{ij} = W_{ji}
    W = sparse(Symmetric(Matrix(W_raw + W_raw')) .> 0)

    # Inferred spatial geometry using force-directed layout
    au_primary = assign_spatial_units_inferred(W)
    p_centroids = au_primary.centroids
    p_hull = au_primary.hull_coords

    # Clayton & Kaldor Reference Values
    y_orig = [9,39,11,9,15,8,26,7,6,20,13,5,3,8,17,9,2,7,9,7,16,31,11,7,19,15,7,10,16,11,5,3,7,8,11,9,11,8,6,4,10,8,2,6,19,3,2,3,28,6,1,1,1,1,0,0]
    E_orig = [1.4,8.7,3.0,2.5,4.3,2.4,8.1,2.3,2.0,6.6,4.4,1.8,1.1,3.3,7.8,4.6,1.1,4.2,5.5,4.4,10.5,22.7,8.8,5.6,15.5,12.5,6.0,9.0,14.4,10.2,4.8,2.9,7.0,8.5,12.3,10.1,12.7,9.4,7.2,5.3,18.8,15.8,4.3,14.6,50.7,8.2,5.6,9.3,88.7,19.6,3.4,3.6,5.7,7.0,4.2,1.8]
    x_orig = [16,16,10,24,10,24,10,7,7,16,7,16,10,24,7,16,10,7,7,10,7,16,10,7,1,1,7,7,10,10,7,24,10,7,7,0,10,1,16,0,1,16,16,0,1,7,1,1,0,1,1,0,1,1,16,10]

    data_primary = DataFrame()
    for i in 1:n_districts
        log_off = log.(fill(E_orig[i], n_years))
        innov = cumsum(randn(n_years) .* 0.1)
        y_p = floor.(Int, abs.(fill(y_orig[i], n_years) .+ (innov .* 4.0)))

        d_df = DataFrame(
            district = i,
            year = 1:n_years,
            y = y_p,
            log_offset = log_off,
            cov1 = fill(x_orig[i], n_years)
        )
        # Calculate rate within block to avoid scoping errors
        d_df.y_rate = d_df.y ./ exp.(d_df.log_offset)
        append!(data_primary, d_df)
    end

    # Assign binary response based on grand mean rate
    data_primary.y_bin = [v > mean(data_primary.y_rate) ? 1 : 0 for v in data_primary.y_rate]

    # Generate correlated covariates
    data_primary.cov2 = 0.5 .* data_primary.cov1 .+ randn(nrow(data_primary))
    # Correcting scoping by using the data frame columns
    data_primary.cov3 = randn(nrow(data_primary)) .* (data_primary.y_rate .^ 2)
    data_primary.cov4 = randn(nrow(data_primary)) .* log.(data_primary.y_rate .+ 1.0)
    data_primary.cov5 = randn(nrow(data_primary)) .* exp.(data_primary.y_rate) .* 2.0
    data_primary.cov6 = randn(nrow(data_primary))

    data_primary.f1 = rand(["A", "B"], nrow(data_primary))
    data_primary.s_idx = data_primary.district
    data_primary.s_x =  [c[1] for c in p_centroids[data_primary.s_idx]]
    data_primary.s_y =  [c[2] for c in p_centroids[data_primary.s_idx]]
    
    n_total = length(data_primary.y_bin)

    reg_indices = mod1.(1:n_total, 4)
    reg_levels = ["North", "South", "East", "West"]
    reg = reg_levels[reg_indices]

    data_primary.region = categorical(reg)  # as a "factor"

    au_primary = merge( au_primary, (
        s_idx = data_primary.s_idx,
        s_x = [c[1] for c in p_centroids[data_primary.s_idx]],
        s_y = [c[2] for c in p_centroids[data_primary.s_idx]],
        s_vals = collect(1:n_districts)
    ))

    # ##########################################################################
    # NESTED DATASET CONSTRUCTION (User-Controlled Expansion)
    # ##########################################################################

    # Spatial domain boundary from primary centroids
    px = [c[1] for c in p_centroids]
    py = [c[2] for c in p_centroids]
    x_min, x_max = minimum(px), maximum(px)
    y_min, y_max = minimum(py), maximum(py)
    x_rng, y_rng = x_max - x_min, y_max - y_min

    # Expansion buffer calculations
    s_buff = (spatial_expansion - 1.0) / 2.0
    nx_min, nx_max = x_min - s_buff * x_rng, x_max + s_buff * x_rng
    ny_min, ny_max = y_min - s_buff * y_rng, y_max + s_buff * y_rng

    nt_max = Int(round(n_years * temporal_expansion))
    n_obs_nested = Int(round(nrow(data_primary) * spatial_expansion * temporal_expansion))

    sx_nested = rand(Uniform(nx_min, nx_max), n_obs_nested)
    sy_nested = rand(Uniform(ny_min, ny_max), n_obs_nested)
    time_nested = rand(1:nt_max, n_obs_nested)

    # Spatial Unit Assignment for expanded domain
    au_nested = assign_spatial_units(sx_nested, sy_nested; target_units=100)

    data_nested = DataFrame(
        s_x = sx_nested,
        s_y = sy_nested,
        year = time_nested,
        district = au_nested.s_idx
    )

    # Latent signal generation for nested grid
    s_lat_n = cumsum(randn(length(au_nested.centroids))) .* 0.3
    t_lat_n = sin.(collect(1:nt_max) .* (2π/nt_max))

    eta_n = [1.5 + s_lat_n[data_nested.district[i]] + t_lat_n[data_nested.year[i]] for i in 1:n_obs_nested]

    data_nested.y = [rand(Poisson(exp(v))) for v in eta_n]
    data_nested.y_rate = exp.(eta_n) .+ randn(n_obs_nested) .* 0.2
    data_nested.y_bin = [v > mean(data_nested.y_rate) ? 1 : 0 for v in data_nested.y_rate]

    data_nested.ncov1 = 0.6 .* eta_n .+ randn(n_obs_nested)
    data_nested.ncov2 = randn(n_obs_nested) .* exp.(data_nested.y_rate)
    data_nested.ncov3 = randn(n_obs_nested)

    primary_out = (data=data_primary, au=au_primary )

    nested_out = (data=data_nested, au=au_nested.W )

    # Directory check and caching
    if !isdir("data"); mkdir("data"); end
    JLD2.save(cache_path, "primary", primary_out, "nested", nested_out)
    println("Dataset successfully cached at: ", cache_path)

    return (primary_out, nested_out)
    # (p_set, n_set) = scottish_lip_cancer_data_spacetime();
end

  
   
function precompute_nystrom_projection(spatial_coords, inducing_points, kernel_func; jitter=1e-6)

    # Example Usage:
    # kernel = Matern32Kernel() ∘ ScaleTransform(1.0)
    # modinputs_reference.K_nystrom_proj = precompute_nystrom_projection(areal_units.centroids, Z_inducing, kernel)

    # println("Precomputing Nystrom Projection Matrix...")
    
    # 1. K_mm: Kernel matrix between inducing points (M x M)
    K_mm = kernelmatrix(kernel_func, RowVecs(inducing_points))
    K_mm_stable = Symmetric(K_mm + jitter * I)
    
    # 2. K_nm: Kernel matrix between all spatial units and inducing points (N x M)
    K_nm = kernelmatrix(kernel_func, RowVecs(spatial_coords), RowVecs(inducing_points))
    
    # 3. Projection: K_nm * inv(K_mm)
    # use the backslash operator for better numerical stability than direct inversion
    K_nystrom_proj = K_nm / K_mm_stable
    return K_nystrom_proj
end



# --- Optimized Householder PCA Helper Functions ---

function householder_to_eigenvector(v_mat::AbstractMatrix{T}, nU, n_factors) where {T}
    # v1.0.1 (2026-06-29 17:16:00)
    # Purpose: Constructs an orthonormal loadings matrix (eigenvectors) from a matrix of
    #          Householder reflector vectors.
    # Inputs: v_mat (matrix of reflector vectors), nU (number of variables), n_factors (number of factors).
    # Outputs: An orthonormal loadings matrix [nU x n_factors].
    # Note: Uses an efficient O(K*N^2) update.

    U = Matrix{T}(I, nU, nU)

    for k in 1:n_factors
        # Extract the k-th Householder vector
        vk = v_mat[:, k]
        norm_v = norm(vk)
        
        if norm_v > 1e-9
            vk = vk / norm_v
            
            # --- O(K * N^2) Optimization ---
            # Naive: U = (I - 2vv') * U  => O(N^3)
            # Optimized: U = U - 2v * (v' * U) => O(N^2)
            # first compute the row vector (v' * U), then perform an outer product update.
            
            v_transpose_U = vk' * U
            U = U - 2.0 .* vk * v_transpose_U
        end
    end

    # Return only the first n_factors columns as the orthonormal loadings matrix
    return U[:, 1:n_factors]
end

function householder_transform(v, nU, n_factors, ltri_indices, pca_sd, pdef_sd, noise)
    # v1.0.1 (2026-06-29 17:16:00)
    # Purpose: Performs the full Householder transformation to reconstruct the covariance matrix
    #          for a Bayesian PCA model.
    # Inputs: v (vector of free parameters), nU, n_factors, ltri_indices, pca_sd, pdef_sd, noise.
    # Outputs: A tuple containing the reconstructed covariance matrix, PCA standard deviations, and loadings matrix.

    T = eltype(v)
    v_mat = zeros(T, nU, n_factors)
    v_mat[ltri_indices] .= v

    # Generate Orthonormal Loadings using optimized transformation
    U = householder_to_eigenvector(v_mat, nU, n_factors)

    # Reconstruct Covariance Components
    # W = Loadings * Scaled Eigenvalues
    W = U * Diagonal(pca_sd)

    # Kmat is the full covariance matrix: WW' + Residual_Variance
    # add a small noise term for numerical stability
    Kmat = W * W' + (pdef_sd^2 + noise) * I(nU)

    return Kmat, pca_sd, U
end



function eigenvector_to_householder(U_in::AbstractMatrix{T}, n_factors) where {T}
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: An internal helper for PCA-based models that performs the inverse operation:
    #          extracting Householder reflector vectors from a given orthonormal loadings matrix.
    #          This is useful for initializing a Bayesian PCA model from a frequentist result.
    # Inputs: U_in (orthonormal loadings matrix), n_factors.
    # Outputs: A matrix of Householder reflector vectors.
    # Note: Complexity is O(K * N^2).

    nU = size(U_in, 1)
    # work on a copy to avoid modifying the input
    U = copy(U_in)
    
    # Storage for the lower-triangular part of the v_mat
    # Each column k corresponds to the k-th Householder vector
    v_mat = zeros(T, nU, n_factors)

    for k in 1:n_factors
        # 1. Target vector is the k-th column of the current transformation
        # For the identity, want U[k,k] to be 1 and others 0
        x = U[k:end, k]
        
        # 2. Standard Householder Reflection Math
        # v = x + sign(x[1]) * ||x|| * e1
        norm_x = norm(x)
        vk = copy(x)
        
        sign_x1 = x[1] >= 0 ? one(T) : -one(T)
        vk[1] += sign_x1 * norm_x
        
        norm_vk = norm(vk)
        if norm_vk > 1e-9
            vk = vk ./ norm_vk
            
            # 3. Apply the reflection to the rest of the matrix (Rank-1 update)
            # U[k:end, k:end] = (I - 2vv') * U[k:end, k:end]
            # Using the O(N^2) update trick
            v_transpose_U = vk' * U[k:end, k:end]
            U[k:end, k:end] -= 2.0 .* vk * v_transpose_U
            
            # 4. Store the reflector
            v_mat[k:end, k] .= vk
        end
    end

    return v_mat
end


 
;;

function extract_v_parameters(v_mat, ltri_indices)
    # extract_v_parameters(v_mat, ltri_indices)
    #
    # Utility to extract only the free parameters (lower triangular) from the v_mat
    # for use as initial values in Turing (matching the 'v' parameter vector).
    return v_mat[ltri_indices]
end 

function get_params_matrix_sizestructured(chain, base_name, dims)
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A specialized utility for extracting matrix-valued parameters from an MCMC chain,
              specifically for size-structured models where parameters are stored as matrices per
              sample.
    """
    # Optimized for chn[:param].data[sample][row, col] access pattern
    n_rows, n_cols = dims
    N_samples = size(chain, 1)

    if Symbol(base_name) in names(chain, :parameters)
        res = zeros(Float64, n_rows, n_cols, N_samples)
        data_container = chain[Symbol(base_name)].data

        for s in 1:N_samples
            samp_mat = data_container[s]
            if size(samp_mat) == (n_rows, n_cols)
                res[:, :, s] = samp_mat
            elseif size(samp_mat) == (n_cols, n_rows)
                res[:, :, s] = samp_mat'
            end
        end
        return res
    end
    return nothing
end

 



 
function get_kernel_from_string(kernel_name::String)
    k_name = lowercase(kernel_name)
    if k_name == "constant"
        return ConstantKernel()
    elseif k_name == "linear"
        return LinearKernel()
    elseif k_name == "matern12" || k_name == "exponential"
        return Matern12Kernel()
    elseif k_name == "matern32"
        return Matern32Kernel()
    elseif k_name == "matern52"
        return Matern52Kernel()
    elseif k_name == "spherical"
        return SphericalKernel()
    elseif k_name == "squared_exponential" || k_name == "se" || k_name == "gaussian" || k_name == "rbf"
        return SqExponentialKernel()
    elseif k_name == "periodic"
        return PeriodicKernel()
    else
        @warn "Kernel '$kernel_name' not recognized. Defaulting to SqExponentialKernel."
        return SqExponentialKernel()
    end
end

  
 

####


 
;;


function build_structure_template(type::Symbol, n::Int; scale=true, coords=nothing, W=nothing, bipartite_adj=nothing)
    # v1.0.1 (2026-06-29 17:16:00)
    # Purpose: A factory for creating precision matrix templates for various GMRF and other structured models.
    #          It handles different manifold types (e.g., ICAR, RW2, SPDE) and computes scaling factors.
    # Inputs: type (Symbol), n (size), and optional parameters like W (adjacency matrix).
    # Outputs: A NamedTuple containing the precision matrix template and a scaling factor.
    
    Q = Matrix{Float64}(undef, 0, 0)
    sf = 1.0

    # Group 0: Null Manifold Handling
    # If the type is :none, we return a 1x1 identity to satisfy supervisor shapes without overhead
        # 'harmonic' and 'rff' are basis-driven but require a recognized identity entry in the registry.
    if type == :iid || type == :none || type == :identity || type == :harmonic || type == :rff
        return (matrix = sparse(I(n)), scaling_factor = 1.0)
    end

  
    # Group 1: Identity and Spectral Bases
    # Rationale: These manifolds operate in transformed or group-level spaces.
    if type in [:iid, :eigen, :bgcn, :nystrom, :rff, :fft, :bspline]
        Q = Matrix(1.0I, n, n)
        sf = 1.0
   
   # Group 2: Graph-Based & CAR Manifolds
    elseif type in [:icar, :besag, :bym2, :leroux, :sar]
        if isnothing(W)
            # If W is missing but n=1, fallback to identity instead of error
            if n <= 1
                Q = Matrix(1.0I, 1, 1)
                sf = 1.0
            else
                error("BSTM Factory Error: Adjacency matrix W required for manifold :$type with size $n")
            end
        else
            D_sp = Diagonal(vec(sum(W, dims=2)))
            Q_raw = Matrix(D_sp - W)
            if scale
                evals = eigvals(Q_raw)
                nz_ev = filter(x -> x > 1e-6, evals)
                sf = isempty(nz_ev) ? 1.0 : exp(mean(log.(nz_ev)))
                Q = Q_raw ./ sf
            else
                Q = Q_raw
                sf = 1.0
            end
        end


    # Group 3: Temporal & Higher-Order Intrinsic GMRFs
    elseif type == :ar1
        local Q_ar = Matrix(1.0I, n, n)
        for i in 1:(n-1); Q_ar[i, i+1] = Q_ar[i+1, i] = -0.5; end
        Q = Q_ar
        sf = 1.0

    elseif type == :rw1
        local Q_rw1 = zeros(n, n)
        for i in 1:n
            if i == 1 || i == n; Q_rw1[i,i] = 1.0; else Q_rw1[i,i] = 2.0; end
            if i < n; Q_rw1[i, i+1] = Q_rw1[i+1, i] = -1.0; end
        end
        sf = 1.0
        Q = Q_rw1

    elseif type == :rw2
        local Q_rw2 = zeros(n, n)
        for i in 1:n
            if 2 < i < n-1
                Q_rw2[i,i]=6; Q_rw2[i,i-1]=Q_rw2[i,i+1]=-4; Q_rw2[i,i-2]=Q_rw2[i,i+2]=1
            elseif i == 1; Q_rw2[i,i]=1; Q_rw2[i,i+1]=-2; Q_rw2[i,i+2]=1
            elseif i == 2; Q_rw2[i,i]=5; Q_rw2[i,i-1]=-2; Q_rw2[i,i+1]=-4; Q_rw2[i,i+2]=1
            elseif i == n-1; Q_rw2[i,i]=5; Q_rw2[i,i+1]=-2; Q_rw2[i,i-1]=-4; Q_rw2[i,i-2]=1
            elseif i == n; Q_rw2[i,i]=1; Q_rw2[i,i-1]=-2; Q_rw2[i,i-2]=1
            end
        end
        if scale
            local evals_rw2 = eigvals(Q_rw2)
            local nz_ev_rw2 = filter(x -> x > 1e-6, evals_rw2)
            sf = isempty(nz_ev_rw2) ? 1.0 : exp(mean(log.(nz_ev_rw2)))
            Q = Q_rw2 ./ sf
        else
            Q = Q_rw2
            sf = 1.0
        end

    # Group 4: Advanced Topological & SPDE Manifolds
    elseif type == :spde
        if isnothing(W)
            error("BSTM Registry Error: Laplacian structure W required for :spde Finite Element approximation")
        end
        local D_mat = Diagonal(vec(sum(W, dims=2)))
        local L_mat = Matrix(D_mat - W)
        # SPDE Matern Precision approximation: (kappa^2 I + L)^2. Template is L.
        local Q_raw = L_mat
        if scale
            local evals_spde = eigvals(Q_raw)
            local nz_ev_spde = filter(x -> x > 1e-6, evals_spde)
            sf = isempty(nz_ev_spde) ? 1.0 : exp(mean(log.(nz_ev_spde)))
            Q = Q_raw ./ sf
        else
            Q = Q_raw
            sf = 1.0
        end

    elseif type == :dag
        # DAGs utilize a directed adjacency matrix (I - rho*W) directly.
        # Scale must be 1.0 to preserve the unit-diagonal properties of the Vecchia operator.
        Q = isnothing(W) ? Matrix(1.0I, n, n) : Matrix(W)
        sf = 1.0

    # Group 5: Basis & Continuous Kernels
    elseif type == :tps
        # Thin Plate Spline bending energy matrix standardized to RW2 scaling logic
        local template_rw2 = build_structure_template(:rw2, n; scale=scale)
        Q = template_rw2.matrix
        sf = template_rw2.scaling_factor

    elseif type == :pspline
        # Penalized Spline template defaults to second-order differencing (RW2)
        local template_rw2 = build_structure_template(:rw2, n; scale=scale)
        Q = template_rw2.matrix
        sf = template_rw2.scaling_factor

    elseif type == :gp
        Q = isnothing(coords) ? Matrix(1.0I, n, n) : [sqrt(sum((c1 .- c2).^2)) for c1 in coords, c2 in coords]
        sf = 1.0

    # Group 6: Periodic & Seasonal Continuity
    elseif type in [:seasonal, :harmonic]
        local Q_cyc = Matrix(1.0I, n, n) .* 2.0
        for i in 1:n
            Q_cyc[i, mod1(i-1, n)] = -1.0
            Q_cyc[i, mod1(i+1, n)] = -1.0
        end
        if scale
            local evals_cyc = eigvals(Q_cyc)
            local nz_ev_cyc = filter(x -> x > 1e-6, evals_cyc)
            sf = isempty(nz_ev_cyc) ? 1.0 : exp(mean(log.(nz_ev_cyc)))
            Q = Q_cyc ./ sf
        else
            Q = Q_cyc
            sf = 1.0
        end

        
    # # Periodic / Cyclic Manifolds
    # Rationale: Standardizing periodic structures as Besag fields on circulant graphs.
    elseif type == :cyclic
        rows = Int[]
        cols = Int[]
        for i in 1:n
            # Backward connection (previous node in cycle)
            prev_node = i == 1 ? n : i - 1
            push!(rows, i)
            push!(cols, prev_node)
            
            # Forward connection (next node in cycle)
            next_node = i == n ? 1 : i + 1
            push!(rows, i)
            push!(cols, next_node)
        end
        
        W_circ = sparse(rows, cols, ones(length(rows)), n, n)
        
        # Recompute Laplacian and scaling using the generated circulant adjacency
        asum_c = vec(sum(W_circ, dims=2))
        Q_circ = Diagonal(asum_c) - W_circ
        sf_c = scale ? scaling_factor_bym2(W_circ) : 1.0
        return (matrix = Q_circ, scaling_factor = sf_c)
 
 
    elseif type == :fft || type == :spectral || type == :wavelet
        # # Wiener-Khinchin Spectral Path
        # This replaces the previous basis-only FFT modeling       
        return build_spectral_precision(n, ls, sig, kernel_type=kernel)
 
    else
        @warn "BSTM Registry Fallback: Manifold :$type not recognized. Initializing Identity."
        Q = Matrix(1.0I, n, n)
        sf = 1.0
    end

    return (matrix = Q, scaling_factor = sf)
end

 
  
# BSTM Spectral & Wavelet Precision Engine v22.25.0
# Timestamp: 2026-06-22 10:45:00
# Rationale: Extending the spectral engine to support wavelet-domain precision mapping and localized non-stationary kernels.

function build_spectral_precision(n::Int64, ls::Real, sig::Real; kernel_type=:se, periodicity=nothing, noise_floor=1e-6, wavelet_levels=3, wavelet_filter=:db2)
    # #
    # 1. Frequency and Basis Discovery
    freqs = FFTW.fftfreq(n)
    psd = zeros(n)

    # #
    # 2. Analytical PSD Definitions for Stationary Taxonomy

    # Gaussian / Squared Exponential
    if kernel_type == :gaussian || kernel_type == :se
        psd .= (sig^2) .* sqrt(2 * pi * ls^2) .* exp.( -2 * pi^2 * ls^2 .* freqs.^2 )

    # Exponential / Matern 1/2
    elseif kernel_type == :exponential || kernel_type == :matern12
        psd .= (sig^2) .* (2 * ls) ./ (1.0 .+ (2 * pi .* freqs .* ls).^2)

    # Matern 3/2
    elseif kernel_type == :matern32
        psd .= (sig^2) .* (4 * ls) ./ ( (1.0 .+ (2 * pi .* freqs .* ls).^2).^2 )

    # Matern 5/2
    elseif kernel_type == :matern52
        psd .= (sig^2) .* (16 * ls / 3.0) ./ ( (1.0 .+ (2 * pi .* freqs .* ls).^2).^3 )

    # Constant Kernel (DC component)
    elseif kernel_type == :constant
        psd[1] = sig^2 * n

    # Linear Kernel (Non-stationary approximation)
    elseif kernel_type == :linear
        psd .= (sig^2) ./ (abs.(freqs) .+ 1e-8).^2
        psd[1] = sig^2

    # Periodic / Seasonal
    elseif kernel_type == :periodic
        p = isnothing(periodicity) ? n : periodicity
        f0 = 1.0 / p
        for (i, f) in enumerate(freqs)
            dist = abs(f) % f0
            psd[i] = (sig^2) * exp(-dist / 0.01)
        end

    # Rationale: For wavelet kernels, the precision is defined by the energy distribution across scales.
    elseif kernel_type == :wavelet
        # In the wavelet domain, we define eigenvalues based on decomposition levels.
        # This allows for modeling multiscale dependencies.
        wavelet_eigenvalues = zeros(n)
        for scale in 1:wavelet_levels
            # Energy decays with scale to ensure smoothness
            energy = sig^2 * exp(-scale / ls)
            # Map energy to indices corresponding to wavelet levels (approximate mapping)
            idx_start = max(1, floor(Int, n / 2^scale))
            idx_end = max(1, floor(Int, n / 2^(scale-1)))
            wavelet_eigenvalues[idx_start:idx_end] .= energy
        end
        
        precision_eigenvalues = 1.0 ./ (wavelet_eigenvalues .+ noise_floor)
        
        first_row_q = real(FFTW.ifft(precision_eigenvalues))
    end

    if kernel_type != :wavelet
        precision_eigenvalues = 1.0 ./ (psd .+ noise_floor)
        first_row_q = real(FFTW.ifft(precision_eigenvalues))
    end

    Q_rows = Int[]
    Q_cols = Int[]
    Q_vals = Float64[]

    for i in 1:n
        for j in 1:n
            # Circulant index shift
            shift = mod(i - j, n) + 1
            val = first_row_q[shift]
            if abs(val) > 1e-12
                push!(Q_rows, i)
                push!(Q_cols, j)
                push!(Q_vals, val)
            end
        end
    end

    Q_sparse = sparse(Q_rows, Q_cols, Q_vals, n, n)

    return (matrix = Q_sparse, scaling_factor = 1.0, psd = psd, weights = first_row_q)
end

 


 

function evaluate_kernel_matrix(dist_sq::AbstractMatrix, param_val::Real, ls::Real, kernel_type::Symbol, noise::Real; wavelet_levels=3)
    # #
    # Gaussian / Squared Exponential
    if kernel_type == :gaussian || kernel_type == :se
        return (param_val^2) .* exp.(-dist_sq ./ (2 * ls^2 + noise))
    
    # #
    # Exponential / Matern 1/2
    elseif kernel_type == :exponential || kernel_type == :matern12
        d = sqrt.(dist_sq .+ noise)
        return (param_val^2) .* exp.(-d ./ (ls + noise))
    
    # #
    # Matern 3/2
    elseif kernel_type == :matern32
        d = sqrt.(dist_sq .+ noise)
        val = sqrt(3.0) .* d ./ (ls + noise)
        return (param_val^2) .* (1.0 .+ val) .* exp.(-val)
    
    # #
    # Matern 5/2
    elseif kernel_type == :matern52
        d = sqrt.(dist_sq .+ noise)
        val = sqrt(5.0) .* d ./ (ls + noise)
        return (param_val^2) .* (1.0 .+ val .+ (val.^2 ./ 3.0)) .* exp.(-val)

    # #
    # Constant Kernel (Identity innovation)
    elseif kernel_type == :constant
        return fill(param_val^2, size(dist_sq))

    # #
    # Linear Kernel
    elseif kernel_type == :linear
        return (param_val^2) .* dist_sq

    # #
    # Wavelet Multiscale Kernel
    # Rationale: Approximates a wavelet covariance by superposing energy scales. 
    # In real-space, this behaves as a sum of kernels with varying lengthscales corresponding to levels.
    elseif kernel_type == :wavelet
        K_accum = zeros(eltype(dist_sq), size(dist_sq))
        for scale in 1:wavelet_levels
            # Scale-specific lengthscale and energy weight
            ls_scale = ls / (2^(scale-1))
            weight_scale = (param_val^2) * exp(-scale / ls)
            
            # Contribution from current resolution level
            K_accum .+= weight_scale .* exp.(-dist_sq ./ (2 * ls_scale^2 + noise))
        end
        return K_accum

    # #
    # Fallback Dispatch
    else
        return (param_val^2) .* exp.(-dist_sq ./ (2 * ls^2 + noise))
    end
end



function wendland_taper(d::AbstractVector, range::Real)
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: Computes a Wendland compactly supported correlation function. This is used as a
    #          taper to enforce compact support on a covariance function, which helps to
    #          reduce spectral leakage in FFT-based methods.
    # Inputs: d (distance vector), range (the range beyond which the function is zero).
    # Outputs: A vector of taper weights.
    h = d ./ range
    taper = zeros(eltype(h), size(h))
    mask = h .< 1.0
    # Wendland function for k=2, d=1
    taper[mask] .= (1.0 .- h[mask]).^4 .* (1.0 .+ 4.0 .* h[mask])
    return taper
end


function estimate_spectral_precision(n::Int64, ls::Real, sig::Real; kernel_type=:se, periodicity=nothing, noise_floor=1e-6, wavelet_levels=3, taper_range=nothing, lowpass_range=nothing)
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: Constructs a sparse precision matrix for a stationary process by computing the
    #          kernel's power spectral density (PSD), applying optional spectral filters,
    #          and using the Wiener-Khinchin theorem.
    # Inputs: n, ls, sig, and optional parameters for kernel type, tapering, and filtering.
    # Outputs: A NamedTuple containing the sparse precision matrix, scaling factor, PSD, and weights.

    # #
    # Frequency Grid Definition
    # Establish discrete Fourier modes for a signal of length n
    freqs = FFTW.fftfreq(n)
    psd = zeros(n)

    # #
    # Analytical Power Spectral Density (PSD) Definitions

    # Gaussian / Squared Exponential
    if kernel_type == :gaussian || kernel_type == :se
        psd .= (sig^2) .* sqrt(2 * pi * ls^2) .* exp.( -2 * pi^2 * ls^2 .* freqs.^2 )

    # Exponential / Matern 1/2
    elseif kernel_type == :matern12 || kernel_type == :exponential
        psd .= (sig^2) .* (2 * ls) ./ (1.0 .+ (2 * pi .* freqs .* ls).^2)

    # Matern 3/2
    elseif kernel_type == :matern32
        psd .= (sig^2) .* (4 * ls) ./ ( (1.0 .+ (2 * pi .* freqs .* ls).^2).^2 )

    # Matern 5/2
    elseif kernel_type == :matern52
        psd .= (sig^2) .* (16 * ls / 3.0) ./ ( (1.0 .+ (2 * pi .* freqs .* ls).^2).^3 )

    # Constant Kernel (DC Component)
    elseif kernel_type == :constant
        psd[1] = sig^2 * n

    # Linear Kernel (Low-frequency approximation)
    elseif kernel_type == :linear
        psd .= (sig^2) ./ (abs.(freqs) .+ 1e-8).^2
        psd[1] = sig^2

    # Periodic / Seasonal harmonics
    elseif kernel_type == :periodic
        p = isnothing(periodicity) ? n : periodicity
        f0 = 1.0 / p
        for i in 1:n
            dist = abs(freqs[i]) % f0
            psd[i] = (sig^2) * exp(-dist / 0.01)
        end

    # Wavelet Multiscale Approximation
    # Rationale: Maps energy distribution across decomposition scales into frequency modes.
    elseif kernel_type == :wavelet
        # Initialize wavelet energy spectrum
        # In the stationary approximation, energy is assigned to frequency octaves
        for scale in 1:wavelet_levels
            # Energy weight for current scale level
            energy_scale = (sig^2) * exp(-scale / ls)
            
            # Identify frequency indices corresponding to the current resolution level
            # Higher scales correspond to lower frequencies (coarse features)
            # Lower scales correspond to higher frequencies (fine details)
            idx_low = floor(Int, n / 2^scale) + 1
            idx_high = floor(Int, n / 2^(scale - 1))
            
            # Apply energy to the spectral mask
            if idx_low <= idx_high
                psd[idx_low:idx_high] .+= energy_scale
            end
        end

        # Apply low-pass filter as decay on scale energy for wavelets
        if !isnothing(lowpass_range)
            for scale in 1:wavelet_levels
                decay_factor = exp(-scale / lowpass_range)
                idx_start = max(1, floor(Int, n / 2^scale))
                idx_end = max(1, floor(Int, n / 2^(scale - 1)))
                if idx_start <= idx_end
                    psd[idx_start:idx_end] .*= decay_factor
                end
            end
        end
    end

    # # 3. Apply Tapering and Filtering in Spatial Domain
    if !isnothing(taper_range) || !isnothing(lowpass_range)
        first_row_k = real(FFTW.ifft(psd))
        
        # Distances on a circulant grid
        x_grid = 0:(n-1)
        distances = min.(x_grid, n .- x_grid)

        if !isnothing(taper_range); first_row_k .*= wendland_taper(distances, taper_range); end
        if !isnothing(lowpass_range); first_row_k .*= exp.(-(distances.^2) ./ (2 * lowpass_range^2)); end

        # Transform back to get the modified PSD
        psd = real(FFTW.fft(first_row_k))
    end

    # # 4. Remove DC Component
    # This is required to remove the influence of the process's mean from the
    # autocorrelation estimate, focusing on the variance structure.
    psd[1] = 0.0

    # # 5. Precision Recomposition in Spectral Domain
    precision_eigenvalues = 1.0 ./ (psd .+ noise_floor)

    # #
    # Real-Space Mapping via Inverse Fourier Transform
    # The first row of the circulant precision matrix is the IFFT of its eigenvalues
    first_row_q = real(FFTW.ifft(precision_eigenvalues))

    # # 6. Sparse Matrix Construction
    # Generate a sparse circulant operator based on the recovered weights
    Q_rows = Int[]
    Q_cols = Int[]
    Q_vals = Float64[]

    for i in 1:n
        for j in 1:n
            # Periodic index shift mapping
            shift = mod(i - j, n) + 1
            val = first_row_q[shift]
            
            # Sparsification using a tolerance relative to the noise floor
            if abs(val) > 1e-12
                push!(Q_rows, i)
                push!(Q_cols, j)
                push!(Q_vals, val)
            end
        end
    end

    Q_sparse = sparse(Q_rows, Q_cols, Q_vals, n, n)

    return (matrix = Q_sparse, scaling_factor = 1.0, psd = psd, weights = first_row_q)
end


 
 
 
# Helper constructor for common defaults (cubic splines)
"""
BSTM Internal Utility v1.0.0
Timestamp: 2026-06-26 10:22:15
Synopsis: A helper constructor for `BSpline` with default parameters.
"""
BSpline(v::Symbol; nbins=10, degree=3, sigma_prior=Exponential(1.0)) = BSpline(v, nbins, degree, sigma_prior)

 

function recompose_precision(
    m_type::Symbol, 
    template_s::AbstractMatrix, 
    param_val::Real; 
    template_t=nothing, 
    extra_param=nothing, 
    noise=1e-4, 
    directed_adj=nothing, 
    flow_direction=:bidirectional
)
    """
    v1.0.1 (2026-06-29 17:16:00)
    Purpose: An internal factory function that constructs a final precision matrix from a
             template, a scale parameter (`param_val`), and other manifold-specific parameters
             (e.g., correlation `rho`). This function is central to defining GMRF priors within
             the model.
    Inputs: m_type, template_s, param_val, and other optional parameters.
    Outputs: A symmetric precision matrix.
    """
    # Rationale: Standardizing the inversion of the variance scale for precision mapping.
    # Requirement: Avoid ternary if/else during sampling to maintain AD stability.
    
    n_s = size(template_s, 1)
    scale_factor = 1.0 / (param_val^2 + noise)

    # 1. Base Graph & CAR Manifolds
    if m_type == :none || m_type == :fixed
        return Symmetric(scale_factor * I(n_s) + noise * I)
    end

    if m_type == :besag || m_type == :icar || m_type == :cyclic
        return Symmetric(scale_factor .* template_s + noise * I)
    end

    if m_type == :bym2
        rho = isnothing(extra_param) ? 0.5 : extra_param
        return Symmetric(scale_factor .* (rho .* template_s + (1.0 - rho) .* I(n_s)) + noise * I)
    end

    if m_type == :leroux
        lambda_val = isnothing(extra_param) ? 0.5 : extra_param
        return Symmetric(scale_factor .* (lambda_val .* template_s + (1.0 - lambda_val) .* I(n_s)) + noise * I)
    end

    # 2. Knorr-Held Interaction Suite (Types I-IV)
    # Rationale: These require the Kronecker product of temporal and spatial structures.
    # Fix: Ensure template_t is utilized to prevent UndefVarError.
    if m_type == :I || m_type == :II || m_type == :III || m_type == :IV
        if isnothing(template_t)
            # Fallback to temporal identity if template_t was not discovered
            Q_full = template_s
        else
            Q_full = kron(template_t, template_s)
        end
        return Symmetric(scale_factor .* Q_full + noise * I)
    end

    # 3. Directed Network & Physics-Informed Transport
    if m_type == :network
        rho_net = isnothing(extra_param) ? 0.8 : extra_param
        W_net = !isnothing(directed_adj) ? directed_adj : template_s
        
        # Adjoint operator for directed flow
        L_op = if flow_direction == :upstream
            I(n_s) - rho_net .* W_net'
        elseif flow_direction == :downstream
            I(n_s) - rho_net .* W_net
        else
            I(n_s) - rho_net .* (W_net + W_net') ./ 2.0
        end
        return Symmetric(scale_factor .* (L_op' * L_op) + noise * I)
    end

    if m_type == :sar || m_type == :dag || m_type == :proper_car
        rho_p = isnothing(extra_param) ? 0.8 : extra_param
        L_op = I(n_s) - rho_p .* template_s
        return Symmetric(scale_factor .* (L_op' * L_op) + noise * I)
    end

    # Spectral Precision via Wiener-Khinchin
    if m_type == :spectral
        ls = isnothing(extra_param) ? 1.0 : extra_param
        kernel = get(kwargs, :kernel, :se)
        return build_spectral_precision(n_s, ls, param_val; kernel_type=kernel, noise_floor=noise).matrix
    end

    # 4. Continuous Distance & Difference Penalties
    if m_type == :gp
        ls = isnothing(extra_param) ? 1.0 : extra_param
        K = (param_val^2) .* exp.(-(Matrix(template_s).^2) ./ (2 * ls^2 + noise))
        return inv(Symmetric(K + noise * I(n_s)))
    end

    if m_type == :spde
        kappa = isnothing(extra_param) ? 1.0 : extra_param
        L_spde = (kappa^2 .* I(n_s) + template_s)
        return Symmetric(scale_factor .* (L_spde' * L_spde) + noise * I)
    end

    if m_type == :rff || m_type == :fft || m_type == :bspline || m_type == :pspline || m_type == :rw1 || m_type == :rw2 || m_type == :tps
        return Symmetric(scale_factor .* template_s + noise * I)
    end

    # Fallback for unrecognized types
    return Symmetric(scale_factor .* template_s + noise * I)
end

# POSITIONAL OVERLOAD: Dispatches from bstm_univariate metadata objects
function recompose_precision(manifold_metadata::NamedTuple, param_val::Real; template_t=nothing, extra_param=nothing, noise=1e-4)
    # v1.0.1 (2026-06-29 17:16:00)
    # Purpose: An overloaded method for `recompose_precision` that dispatches based on a `NamedTuple`
    #          containing manifold metadata.
    # Inputs: manifold_metadata, param_val, and other optional parameters.
    # Outputs: A symmetric precision matrix.

    return recompose_precision(
        Symbol(manifold_metadata.model_type), 
        manifold_metadata.Q_template, 
        param_val; 
        template_t=template_t, 
        extra_param=extra_param, 
        noise=noise
    )
end

# # Precision Recomposition Factory (Recursive Algebraic Dispatch)
# This function now accepts Manifold structs or algebraic operator results.
function recompose_precision(manifold_node::Any, M, param_sig::Real; noise=1e-4)
    # v1.0.1 (2026-06-29 17:16:00)
    # Purpose: A recursive version of `recompose_precision` that handles algebraic compositions
    #          of manifolds (e.g., Kronecker products, direct sums).
    # Inputs: manifold_node (a Manifold struct or ComposedManifold), M (model config), param_sig (scale).
    # Outputs: A precision matrix or a structure representing the composed precision.

    # # Base Case: Atomic Manifold Structs
    if manifold_node isa Manifold
        m_type = manifold_type(manifold_node)
        # Atomic builders provide the base template Q
        m_meta = build_model(manifold_node, M)
        return recompose_precision(Symbol(m_type), m_meta.Q_template, param_sig; noise=noise)
    
    # # Algebraic Case: Composed Manifolds (⊗, ⊕)
    elseif manifold_node isa ComposedManifold
        # Retrieve components
        comps = manifold_node.components
        op = manifold_node.operator
        
        if op == :kronecker_product
            # Q_total = Q1 ⊗ Q2
            # We split the parameter variance across components or treat sig as global scale
            Q1 = recompose_precision(comps[1], M, 1.0; noise=noise)
            Q2 = recompose_precision(comps[2], M, 1.0; noise=noise)
            Q_full = kron(Q1, Q2)
            return Symmetric(Q_full ./ (param_sig^2 + noise) + noise * I)
            
        elseif op == :direct_sum
            # Q_total is block diagonal if domains are distinct
            # Here we return a list of precisions for the supervisor to iterate
            return [recompose_precision(c, M, param_sig; noise=noise) for c in comps]

        elseif op == :directed_dependency
            # Q_total = (I - ρM1)' Q2 (I - ρM1)
            # Rationale: Representing state-space transitions or advection operators
            Q1 = recompose_precision(comps[1], M, 1.0; noise=noise)
            Q2 = recompose_precision(comps[2], M, 1.0; noise=noise)
            
            # For directed dependencies, we return the tuple of operators for the supervisor
            # as they typically require an explicit coupling parameter (rho)
            return (Q_source=Q1, Q_innovation=Q2, type=:directed)
            
        elseif op == :composition
            # Q_total = Q1 * Q2 * Q1 (Basis warping/Cascading dependency)
            # Rationale: Representing the composition of two precision structures as a functional transformation.
            Q1 = recompose_precision(comps[1], M, 1.0; noise=noise)
            Q2 = recompose_precision(comps[2], M, 1.0; noise=noise)
            
            # Ensure dimensional parity for matrix multiplication in composition
            if size(Q1) == size(Q2)
                Q_comp = Q1 * Q2 * Q1
                return Symmetric(Q_comp ./ (param_sig^2 + noise) + noise * I)
            else
                # Fallback to Kronecker if dimensions are heterogeneous
                return Symmetric(kron(Q1, Q2) ./ (param_sig^2 + noise) + noise * I)
            end

        elseif op == :pipe
            # Stacking: M1 |> M2 (e.g., Spatial |> AR1 for space-time evolution)
            # This implies a state-space transition matrix
            return recompose_precision(comps[1], M, param_sig; noise=noise)
        end
    end
    
    # Fallback for symbol-based legacy calls
    return Matrix(1.0I, 1, 1)
end



function parse_variable_and_transforms(var_str::AbstractString)
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: An internal helper for the formula parser that separates a variable name from any
    #          piped transformation functions (e.g., "x |> log").
    # Inputs: var_str (string).
    # Outputs: A tuple containing the variable name and a vector of transform names.
    parts = Base.split(var_str, "|>")
    var_name = strip(parts[1])
    transforms = [strip(p) for p in parts[2:end]]
    return var_name, transforms
end

function apply_transforms(data_vector, transform_names::Vector{String})
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal utility that applies a series of named transformations (e.g., "log",
              "zscore") to a data vector based on the `BSTM_TRANSFORM_KEYWORDS` dictionary.
    """
    v = copy(data_vector)
    for t_name in transform_names
        if haskey(BSTM_TRANSFORM_KEYWORDS, t_name)
            v = BSTM_TRANSFORM_KEYWORDS[t_name](v)
        else
            @warn "Unknown transformation '$t_name' requested. Ignoring."
        end
    end
    return v
end

function sample_spectral_density(kernel_name::String, D_in::Int, M_rff::Int, lengthscale::Real; nu::Union{Real, Nothing}=nothing)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal helper that samples frequencies from the analytical spectral density of
              a given kernel (e.g., Matern, Squared Exponential). This is used to generate the
              `W` matrix for Random Fourier Features.
    """
    k_name = lowercase(kernel_name)
    
    # Default to Squared Exponential if kernel is not supported for RFF
    if !(k_name in ["squared_exponential", "se", "gaussian", "rbf", "matern12", "exponential", "matern32", "matern52"])
        @warn "RFF sampling for kernel '$kernel_name' is not implemented. Defaulting to 'squared_exponential'."
        k_name = "squared_exponential"
    end

    if startswith(k_name, "matern")
        local effective_nu
        if !isnothing(nu)
            effective_nu = nu
        else
            if k_name == "matern12"; effective_nu = 0.5;
            elseif k_name == "matern32"; effective_nu = 1.5;
            elseif k_name == "matern52"; effective_nu = 2.5;
            else; effective_nu = 1.5; end # Default Matern nu
        end
        # Matern spectral density corresponds to a scaled Student's-t distribution
        df = 2 * effective_nu
        dist = TDist(df)
        scale_factor = sqrt(df) / lengthscale
        return rand(dist, D_in, M_rff) .* scale_factor
    else # Squared Exponential
        sigma_spectral = 1.0 / lengthscale
        return randn(D_in, M_rff) .* sigma_spectral
    end
end


function bstm_smooth_basis_1D(type::String, vals::AbstractVector, nbins::Int, degree::Int; W=nothing, kwargs...)
    # BSTM Smooth Basis Factory v2.2.0
    # Timestamp: 2026-06-27 22:30:00
    # Synopsis: A factory function that generates a 1D basis matrix for various smoothers.
    # Rationale for v2.2.0:
    #     - Added support for 'linear', 'invdist', and 'kriging' basis types.
    #     - 'linear' is an alias for the existing piecewise linear (hat function) basis.
    #     - 'invdist' implements an inverse squared distance weighting basis.
    #     - 'kriging' implements a basis using a Gaussian (Squared Exponential) kernel.

    n_obs = length(vals)
    B = zeros(Float64, n_obs, nbins)
    
    v_min = minimum(vals)
    v_max = maximum(vals)
    v_std = std(vals) + 1e-9

    if type in ["bspline", "pspline", "smooth", "barycentric", "linear"]
        knots = collect(range(v_min, stop=v_max, length=nbins))
        h = (v_max - v_min) / (nbins > 1 ? (nbins - 1) : 1)
        h = h > 0 ? h : 1.0

        for m in 1:nbins
            dist = abs.(vals .- knots[m]) ./ h
            mask = dist .< 1.0
            B[mask, m] .= 1.0 .- dist[mask]
        end

    elseif type == "tps"
        knots = quantile(vals, range(0, 1, length=nbins))
        for m in 1:nbins
            r = abs.(vals .- knots[m])
            B[:, m] .= (r.^2) .* log.(r .+ 1e-6)
        end

    elseif type == "rff"
        m_rff = nbins
        ls = get(kwargs, :lengthscale, v_std)
        Omega = randn(1, m_rff) ./ ls
        Phi_phases = rand(m_rff) .* (2.0 * pi)
        B .= sqrt(2.0 / m_rff) .* cos.((vals * Omega) .+ Phi_phases')

    elseif type == "fft"
        ls = get(kwargs, :lengthscale, v_std)
        t_coords = vals ./ ls
        for m in 1:div(nbins, 2)
            B[:, 2m-1] .= sin.(2.0 * pi * m .* t_coords)
            B[:, 2m]   .= cos.(2.0 * pi * m .* t_coords)
        end

    elseif type == "wavelet"
        for m in 1:nbins
            center = v_min + (m/nbins) * (v_max - v_min)
            width = (v_max - v_min) / nbins
            B[:, m] .= (vals .>= center) .& (vals .< (center + width)) ? 1.0 : 0.0
        end

    elseif type == "moran"
        if isnothing(W)
            @warn "Moran's I basis requires an adjacency matrix 'W'. Falling back to sine proxy."
            for m in 1:nbins
                B[:, m] .= sin.(pi * m * (vals .- v_min) ./ (v_max - v_min))
            end
        else
            n_spatial = size(W, 1)
            if n_spatial != n_obs
                @warn "Moran's I basis requires W matrix size to match number of observations for 1D smooth. Using proxy."
                 for m in 1:nbins; B[:, m] .= sin.(pi * m * (vals .- v_min) ./ (v_max - v_min)); end
            else
                centering_matrix = I - (1/n_spatial) * ones(n_spatial, n_spatial)
                W_centered = centering_matrix * W * centering_matrix
                eigen_decomp = eigen(Symmetric(Matrix(W_centered)))
                B = eigen_decomp.vectors[:, end-nbins+1:end]
            end
        end

    elseif type == "spherical"
        range_r = get(kwargs, :range, v_std * 2.0)
        knots = quantile(vals, range(0, 1, length=nbins))
        for m in 1:nbins
            h = abs.(vals .- knots[m]) ./ range_r
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    elseif type == "decay"
        ls = get(kwargs, :lengthscale, v_std)
        knots = quantile(vals, range(0, 1, length=nbins))
        for m in 1:nbins
            B[:, m] .= exp.(-abs.(vals .- knots[m]) ./ ls)
        end

    elseif type == "invdist"
        knots = quantile(vals, range(0, 1, length=nbins))
        for m in 1:nbins
            dist_sq = (vals .- knots[m]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end

    elseif type == "kriging"
        knots = quantile(vals, range(0, 1, length=nbins))
        ls = get(kwargs, :lengthscale, v_std)
        for m in 1:nbins
            dist_sq = (vals .- knots[m]).^2
            B[:, m] .= exp.(-dist_sq ./ (2 * ls^2))
        end

    else
        B = ones(n_obs, 1)
    end

    return B
end


function bstm_smooth_basis_2D(type::String, coords::AbstractMatrix, nbins::Int; W=nothing, kwargs...)
    # v1.0.1 (2026-06-29 17:16:00)
    # Purpose: A factory function that generates a 2D basis matrix for various smoothers
    #          (e.g., splines, RFF, wavelets) for 2D coordinates.
    # Inputs: type (string), coords (N_obs x 2 matrix), nbins, and optional parameters.
    # Outputs: A basis matrix [N_obs x nbins].
    # Note: Used for `smooth(x, y)` terms.


    n_obs = size(coords, 1)
    
    c_min = [minimum(coords[:, 1]), minimum(coords[:, 2])]
    c_max = [maximum(coords[:, 1]), maximum(coords[:, 2])]
    c_std = [std(coords[:, 1]), std(coords[:, 2])] .+ 1e-9

    ls_x = get(kwargs, :ls_x, c_std[1])
    ls_y = get(kwargs, :ls_y, c_std[2])

    if type in ["pspline", "bspline", "smooth", "barycentric", "linear"]
        n_marginal = Int(floor(sqrt(nbins)))
        m_total = n_marginal^2
        B = zeros(Float64, n_obs, m_total)
        
        kx = collect(range(c_min[1], stop=c_max[1], length=n_marginal))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_marginal))
        hx = (c_max[1] - c_min[1]) / (n_marginal > 1 ? (n_marginal - 1) : 1)
        hy = (c_max[2] - c_min[2]) / (n_marginal > 1 ? (n_marginal - 1) : 1)
        hx = hx > 0 ? hx : 1.0
        hy = hy > 0 ? hy : 1.0

        idx = 1
        for i in 1:n_marginal
            for j in 1:n_marginal
                dist_x = abs.(coords[:, 1] .- kx[i]) ./ hx
                dist_y = abs.(coords[:, 2] .- ky[j]) ./ hy
                mask_x = dist_x .< 1.0
                mask_y = dist_y .< 1.0
                
                b_x = zeros(n_obs)
                b_y = zeros(n_obs)
                b_x[mask_x] .= 1.0 .- dist_x[mask_x]
                b_y[mask_y] .= 1.0 .- dist_y[mask_y]
                
                B[:, idx] .= b_x .* b_y
                idx += 1
            end
        end

    elseif type == "tps"
        B = zeros(Float64, n_obs, nbins)
        n_grid = Int(floor(sqrt(nbins)))
        kx = collect(range(c_min[1], stop=c_max[1], length=n_grid))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_grid))
        centers = [(x, y) for x in kx, y in ky][:]

        for m in 1:min(nbins, length(centers))
            dx = (coords[:, 1] .- centers[m][1]) ./ ls_x
            dy = (coords[:, 2] .- centers[m][2]) ./ ls_y
            r = sqrt.(dx.^2 .+ dy.^2)
            B[:, m] .= (r.^2) .* log.(r .+ 1e-6)
        end

    elseif type == "rff" || type == "anisotropic"
        Omega = randn(2, nbins)
        Omega[1, :] ./= ls_x
        Omega[2, :] ./= ls_y
        Phi_phases = rand(nbins) .* (2.0 * pi)
        B = sqrt(2.0 / nbins) .* cos.((coords * Omega) .+ Phi_phases')

    elseif type == "fft"
        n_marginal = Int(floor(sqrt(nbins / 2)))
        B = zeros(Float64, n_obs, nbins)
        nx = coords[:, 1] ./ ls_x
        ny = coords[:, 2] ./ ls_y

        idx = 1
        for mx in 1:n_marginal, my in 1:n_marginal
            if idx + 1 <= nbins
                arg = mx .* nx .+ my .* ny
                B[:, idx]   .= sin.(2.0 * pi * arg)
                B[:, idx+1] .= cos.(2.0 * pi * arg)
                idx += 2
            end
        end

    elseif type == "wavelet"
        n_marginal = Int(floor(sqrt(nbins)))
        m_total = n_marginal^2
        B = zeros(Float64, n_obs, m_total)
        
        centers_x = c_min[1] .+ ((1:n_marginal) ./ n_marginal) .* (c_max[1] - c_min[1])
        width_x = (c_max[1] - c_min[1]) / n_marginal
        centers_y = c_min[2] .+ ((1:n_marginal) ./ n_marginal) .* (c_max[2] - c_min[2])
        width_y = (c_max[2] - c_min[2]) / n_marginal

        idx = 1
        for i in 1:n_marginal
            for j in 1:n_marginal
                b_x = (coords[:, 1] .>= centers_x[i]) .& (coords[:, 1] .< (centers_x[i] + width_x))
                b_y = (coords[:, 2] .>= centers_y[j]) .& (coords[:, 2] .< (centers_y[j] + width_y))
                B[:, idx] .= b_x .* b_y
                idx += 1
            end
        end

    elseif type == "spherical"
        B = zeros(Float64, n_obs, nbins)
        n_grid = Int(floor(sqrt(nbins)))
        kx = collect(range(c_min[1], stop=c_max[1], length=n_grid))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_grid))
        centers = [(x, y) for x in kx, y in ky][:]
        range_r = get(kwargs, :range, mean(c_std))
        for m in 1:min(nbins, length(centers))
            dx = coords[:, 1] .- centers[m][1]
            dy = coords[:, 2] .- centers[m][2]
            h = sqrt.(dx.^2 .+ dy.^2) ./ range_r
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    elseif type == "moran"
        if isnothing(W)
            @warn "Moran's I basis requires an adjacency matrix 'W'. Falling back to sine proxy."
            B = zeros(Float64, n_obs, nbins)
            for m in 1:nbins
                B[:, m] .= sin.(pi * m .* (coords[:, 1] .+ coords[:, 2]) ./ sum(c_max))
            end
        else
            n = size(W, 1)
            centering_matrix = I - (1/n) * ones(n, n)
            W_centered = centering_matrix * W * centering_matrix
            eigen_decomp = eigen(Symmetric(Matrix(W_centered)))
            B = eigen_decomp.vectors[:, end-nbins+1:end]
        end

    elseif type == "invdist"
        B = zeros(Float64, n_obs, nbins)
        n_grid = Int(floor(sqrt(nbins)))
        kx = collect(range(c_min[1], stop=c_max[1], length=n_grid))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_grid))
        centers = [(x, y) for x in kx, y in ky][:]
        for m in 1:min(nbins, length(centers))
            dist_sq = (coords[:, 1] .- centers[m][1]).^2 .+ (coords[:, 2] .- centers[m][2]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end

    elseif type == "kriging"
        B = zeros(Float64, n_obs, nbins)
        n_grid = Int(floor(sqrt(nbins)))
        kx = collect(range(c_min[1], stop=c_max[1], length=n_grid))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_grid))
        centers = [(x, y) for x in kx, y in ky][:]
        for m in 1:min(nbins, length(centers))
            dist_sq = ((coords[:, 1] .- centers[m][1]).^2 ./ ls_x^2) .+ ((coords[:, 2] .- centers[m][2]).^2 ./ ls_y^2)
            B[:, m] .= exp.(-dist_sq ./ 2.0)
        end

    else
        B = ones(n_obs, 1)
    end

    return B
end


function bstm_smooth_basis_3D(type::String, coords::AbstractMatrix, nbins::Int; W=nothing, kwargs...)
    # v1.0.1 (2026-06-29 17:16:00)
    # Purpose: A factory function that generates a 3D basis matrix for various smoothers
    #          (e.g., splines, RFF, wavelets) for 3D coordinates.
    # Inputs: type (string), coords (N_obs x 3 matrix), nbins, and optional parameters.
    # Outputs: A basis matrix [N_obs x nbins].
    # Note: Used for `smooth(x, y, z)` terms.


    n_obs = size(coords, 1)

    c_min = [minimum(coords[:, 1]), minimum(coords[:, 2]), minimum(coords[:, 3])]
    c_max = [maximum(coords[:, 1]), maximum(coords[:, 2]), maximum(coords[:, 3])]
    c_std = [std(coords[:, 1]), std(coords[:, 2]), std(coords[:, 3])] .+ 1e-9

    ls_x = get(kwargs, :ls_x, c_std[1])
    ls_y = get(kwargs, :ls_y, c_std[2])
    ls_z = get(kwargs, :ls_z, c_std[3])

    if type in ["pspline", "bspline", "smooth", "barycentric", "linear"]
        n_marginal = Int(floor(cbrt(nbins)))
        m_total = n_marginal^3
        B = zeros(Float64, n_obs, m_total)

        kx = collect(range(c_min[1], stop=c_max[1], length=n_marginal))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_marginal))
        kz = collect(range(c_min[3], stop=c_max[3], length=n_marginal))
        hx = (c_max[1] - c_min[1]) / (n_marginal > 1 ? (n_marginal - 1) : 1); hx = hx > 0 ? hx : 1.0
        hy = (c_max[2] - c_min[2]) / (n_marginal > 1 ? (n_marginal - 1) : 1); hy = hy > 0 ? hy : 1.0
        hz = (c_max[3] - c_min[3]) / (n_marginal > 1 ? (n_marginal - 1) : 1); hz = hz > 0 ? hz : 1.0

        idx = 1
        for i in 1:n_marginal, j in 1:n_marginal, k in 1:n_marginal
            b_x = max.(0.0, 1.0 .- abs.(coords[:, 1] .- kx[i]) ./ hx)
            b_y = max.(0.0, 1.0 .- abs.(coords[:, 2] .- ky[j]) ./ hy)
            b_z = max.(0.0, 1.0 .- abs.(coords[:, 3] .- kz[k]) ./ hz)
            B[:, idx] .= b_x .* b_y .* b_z
            idx += 1
        end

    elseif type == "tps"
        n_grid = Int(floor(cbrt(nbins)))
        m_total = n_grid^3
        B = zeros(Float64, n_obs, m_total)
        centers = [(x, y, z) for x in range(c_min[1], c_max[1], length=n_grid), y in range(c_min[2], c_max[2], length=n_grid), z in range(c_min[3], c_max[3], length=n_grid)][:]
        for m in 1:min(m_total, length(centers))
            dx = (coords[:, 1] .- centers[m][1]) ./ ls_x
            dy = (coords[:, 2] .- centers[m][2]) ./ ls_y
            dz = (coords[:, 3] .- centers[m][3]) ./ ls_z
            r = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
            B[:, m] .= (r.^2) .* log.(r .+ 1e-6)
        end

    elseif type == "rff"
        Omega = randn(3, nbins)
        Omega[1, :] ./= ls_x; Omega[2, :] ./= ls_y; Omega[3, :] ./= ls_z
        Phi_phases = rand(nbins) .* (2.0 * pi)
        B = sqrt(2.0 / nbins) .* cos.((coords * Omega) .+ Phi_phases')

    elseif type == "fft"
        n_marginal = Int(floor(cbrt(nbins / 2)))
        B = zeros(Float64, n_obs, nbins)
        nx = coords[:, 1] ./ ls_x
        ny = coords[:, 2] ./ ls_y
        nz = coords[:, 3] ./ ls_z
        idx = 1
        for mx in 1:n_marginal, my in 1:n_marginal, mz in 1:n_marginal
            if idx + 1 <= nbins
                arg = mx .* nx .+ my .* ny .+ mz .* nz
                B[:, idx]   .= sin.(2.0 * pi * arg)
                B[:, idx+1] .= cos.(2.0 * pi * arg)
                idx += 2
            end
        end

    elseif type == "wavelet"
        n_marginal = Int(floor(cbrt(nbins)))
        m_total = n_marginal^3
        B = zeros(Float64, n_obs, m_total)
        
        centers_x = c_min[1] .+ ((1:n_marginal) ./ n_marginal) .* (c_max[1] - c_min[1])
        width_x = (c_max[1] - c_min[1]) / n_marginal
        centers_y = c_min[2] .+ ((1:n_marginal) ./ n_marginal) .* (c_max[2] - c_min[2])
        width_y = (c_max[2] - c_min[2]) / n_marginal
        centers_z = c_min[3] .+ ((1:n_marginal) ./ n_marginal) .* (c_max[3] - c_min[3])
        width_z = (c_max[3] - c_min[3]) / n_marginal

        idx = 1
        for i in 1:n_marginal, j in 1:n_marginal, k in 1:n_marginal
            b_x = (coords[:, 1] .>= centers_x[i]) .& (coords[:, 1] .< (centers_x[i] + width_x))
            b_y = (coords[:, 2] .>= centers_y[j]) .& (coords[:, 2] .< (centers_y[j] + width_y))
            b_z = (coords[:, 3] .>= centers_z[k]) .& (coords[:, 3] .< (centers_z[k] + width_z))
            B[:, idx] .= b_x .* b_y .* b_z
            idx += 1
        end

    elseif type == "spherical"
        n_grid = Int(floor(cbrt(nbins)))
        m_total = n_grid^3
        B = zeros(Float64, n_obs, m_total)
        centers = [(x, y, z) for x in range(c_min[1], c_max[1], length=n_grid), y in range(c_min[2], c_max[2], length=n_grid), z in range(c_min[3], c_max[3], length=n_grid)][:]
        for m in 1:min(m_total, length(centers))
            dx = (coords[:, 1] .- centers[m][1]) ./ ls_x
            dy = (coords[:, 2] .- centers[m][2]) ./ ls_y
            dz = (coords[:, 3] .- centers[m][3]) ./ ls_z
            h = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    elseif type == "moran"
        if isnothing(W)
            @warn "Moran's I basis requires an adjacency matrix 'W'. Using sine proxy."
            B = zeros(Float64, n_obs, nbins)
            for m in 1:nbins
                B[:, m] .= sin.(pi * m .* (coords[:, 1] .+ coords[:, 2] .+ coords[:, 3]) ./ sum(c_max))
            end
        else
            n = size(W, 1)
            centering_matrix = I - (1/n) * ones(n, n)
            W_centered = centering_matrix * W * centering_matrix
            eigen_decomp = eigen(Symmetric(Matrix(W_centered)))
            B = eigen_decomp.vectors[:, end-nbins+1:end]
        end

    elseif type == "invdist"
        B = zeros(Float64, n_obs, nbins)
        n_grid = Int(floor(cbrt(nbins)))
        m_total = n_grid^3
        centers = [(x, y, z) for x in range(c_min[1], c_max[1], length=n_grid), y in range(c_min[2], c_max[2], length=n_grid), z in range(c_min[3], c_max[3], length=n_grid)][:]
        for m in 1:min(m_total, length(centers))
            dist_sq = (coords[:, 1] .- centers[m][1]).^2 .+ (coords[:, 2] .- centers[m][2]).^2 .+ (coords[:, 3] .- centers[m][3]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end

    elseif type == "kriging"
        B = zeros(Float64, n_obs, nbins)
        n_grid = Int(floor(cbrt(nbins)))
        m_total = n_grid^3
        centers = [(x, y, z) for x in range(c_min[1], c_max[1], length=n_grid), y in range(c_min[2], c_max[2], length=n_grid), z in range(c_min[3], c_max[3], length=n_grid)][:]
        for m in 1:min(m_total, length(centers))
            dist_sq = ((coords[:, 1] .- centers[m][1]).^2 ./ ls_x^2) .+ ((coords[:, 2] .- centers[m][2]).^2 ./ ls_y^2) .+ ((coords[:, 3] .- centers[m][3]).^2 ./ ls_z^2)
            B[:, m] .= exp.(-dist_sq ./ 2.0)
        end

    else
        B = ones(n_obs, 1)
    end

    return B
end


function bstm_smooth_basis_4D(type::String, coords::AbstractMatrix, nbins::Int; W=nothing, kwargs...)
    # v1.0.1 (2026-06-29 17:16:00)
    # Purpose: A factory function that generates a 4D basis matrix for various smoothers
    #          (e.g., splines, RFF, wavelets) for 4D coordinates.
    # Inputs: type (string), coords (N_obs x 4 matrix), nbins, and optional parameters.
    # Outputs: A basis matrix [N_obs x nbins].
    # Note: Used for `smooth(v, x, y, z)` terms.


    n_obs = size(coords, 1)

    c_min = [minimum(coords[:, i]) for i in 1:4]
    c_max = [maximum(coords[:, i]) for i in 1:4]
    c_std = [std(coords[:, i]) for i in 1:4] .+ 1e-9

    ls_1 = get(kwargs, :ls_1, c_std[1])
    ls_2 = get(kwargs, :ls_2, c_std[2])
    ls_3 = get(kwargs, :ls_3, c_std[3])
    ls_4 = get(kwargs, :ls_4, c_std[4])

    if type in ["pspline", "bspline", "smooth", "barycentric", "linear"]
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        B = zeros(Float64, n_obs, m_total)

        k1 = collect(range(c_min[1], stop=c_max[1], length=n_marginal))
        k2 = collect(range(c_min[2], stop=c_max[2], length=n_marginal))
        k3 = collect(range(c_min[3], stop=c_max[3], length=n_marginal))
        k4 = collect(range(c_min[4], stop=c_max[4], length=n_marginal))
        h1 = (c_max[1] - c_min[1]) / (n_marginal > 1 ? (n_marginal - 1) : 1); h1 = h1 > 0 ? h1 : 1.0
        h2 = (c_max[2] - c_min[2]) / (n_marginal > 1 ? (n_marginal - 1) : 1); h2 = h2 > 0 ? h2 : 1.0
        h3 = (c_max[3] - c_min[3]) / (n_marginal > 1 ? (n_marginal - 1) : 1); h3 = h3 > 0 ? h3 : 1.0
        h4 = (c_max[4] - c_min[4]) / (n_marginal > 1 ? (n_marginal - 1) : 1); h4 = h4 > 0 ? h4 : 1.0

        idx = 1
        for i in 1:n_marginal, j in 1:n_marginal, k in 1:n_marginal, l in 1:n_marginal
            b1 = max.(0.0, 1.0 .- abs.(coords[:, 1] .- k1[i]) ./ h1)
            b2 = max.(0.0, 1.0 .- abs.(coords[:, 2] .- k2[j]) ./ h2)
            b3 = max.(0.0, 1.0 .- abs.(coords[:, 3] .- k3[k]) ./ h3)
            b4 = max.(0.0, 1.0 .- abs.(coords[:, 4] .- k4[l]) ./ h4)
            B[:, idx] .= b1 .* b2 .* b3 .* b4
            idx += 1
        end

    elseif type == "tps"
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        B = zeros(Float64, n_obs, m_total)
        centers = [(w,x,y,z) for w in range(c_min[1],c_max[1],length=n_marginal), x in range(c_min[2],c_max[2],length=n_marginal), y in range(c_min[3],c_max[3],length=n_marginal), z in range(c_min[4],c_max[4],length=n_marginal)][:]
        for m in 1:min(m_total, length(centers))
            d1 = (coords[:, 1] .- centers[m][1]) ./ ls_1
            d2 = (coords[:, 2] .- centers[m][2]) ./ ls_2
            d3 = (coords[:, 3] .- centers[m][3]) ./ ls_3
            d4 = (coords[:, 4] .- centers[m][4]) ./ ls_4
            r = sqrt.(d1.^2 .+ d2.^2 .+ d3.^2 .+ d4.^2)
            B[:, m] .= (r.^2) .* log.(r .+ 1e-6)
        end

    elseif type == "rff"
        Omega = randn(4, nbins)
        Omega[1, :] ./= ls_1; Omega[2, :] ./= ls_2; Omega[3, :] ./= ls_3; Omega[4, :] ./= ls_4
        Phi_phases = rand(nbins) .* (2.0 * pi)
        B = sqrt(2.0 / nbins) .* cos.((coords * Omega) .+ Phi_phases')

    elseif type == "fft"
        n_marginal = Int(floor(sqrt(sqrt(nbins / 2))))
        B = zeros(Float64, n_obs, nbins)
        nx1 = coords[:, 1] ./ ls_1
        nx2 = coords[:, 2] ./ ls_2
        nx3 = coords[:, 3] ./ ls_3
        nx4 = coords[:, 4] ./ ls_4
        idx = 1
        for m1 in 1:n_marginal, m2 in 1:n_marginal, m3 in 1:n_marginal, m4 in 1:n_marginal
            if idx + 1 <= nbins
                arg = m1 .* nx1 .+ m2 .* nx2 .+ m3 .* nx3 .+ m4 .* nx4
                B[:, idx]   .= sin.(2.0 * pi * arg)
                B[:, idx+1] .= cos.(2.0 * pi * arg)
                idx += 2
            end
        end

    elseif type == "wavelet"
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        B = zeros(Float64, n_obs, m_total)

        centers1 = c_min[1] .+ ((1:n_marginal) ./ n_marginal) .* (c_max[1] - c_min[1])
        width1 = (c_max[1] - c_min[1]) / n_marginal
        centers2 = c_min[2] .+ ((1:n_marginal) ./ n_marginal) .* (c_max[2] - c_min[2])
        width2 = (c_max[2] - c_min[2]) / n_marginal
        centers3 = c_min[3] .+ ((1:n_marginal) ./ n_marginal) .* (c_max[3] - c_min[3])
        width3 = (c_max[3] - c_min[3]) / n_marginal
        centers4 = c_min[4] .+ ((1:n_marginal) ./ n_marginal) .* (c_max[4] - c_min[4])
        width4 = (c_max[4] - c_min[4]) / n_marginal

        idx = 1
        for i in 1:n_marginal, j in 1:n_marginal, k in 1:n_marginal, l in 1:n_marginal
            b1 = (coords[:, 1] .>= centers1[i]) .& (coords[:, 1] .< (centers1[i] + width1))
            b2 = (coords[:, 2] .>= centers2[j]) .& (coords[:, 2] .< (centers2[j] + width2))
            b3 = (coords[:, 3] .>= centers3[k]) .& (coords[:, 3] .< (centers3[k] + width3))
            b4 = (coords[:, 4] .>= centers4[l]) .& (coords[:, 4] .< (centers4[l] + width4))
            B[:, idx] .= b1 .* b2 .* b3 .* b4
            idx += 1
        end

    elseif type == "spherical"
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        B = zeros(Float64, n_obs, m_total)
        centers = [(w,x,y,z) for w in range(c_min[1],c_max[1],length=n_marginal), x in range(c_min[2],c_max[2],length=n_marginal), y in range(c_min[3],c_max[3],length=n_marginal), z in range(c_min[4],c_max[4],length=n_marginal)][:]
        for m in 1:min(m_total, length(centers))
            d1 = (coords[:, 1] .- centers[m][1]) ./ ls_1
            d2 = (coords[:, 2] .- centers[m][2]) ./ ls_2
            d3 = (coords[:, 3] .- centers[m][3]) ./ ls_3
            d4 = (coords[:, 4] .- centers[m][4]) ./ ls_4
            h = sqrt.(d1.^2 .+ d2.^2 .+ d3.^2 .+ d4.^2)
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    elseif type == "moran"
        if isnothing(W)
            @warn "Moran's I basis requires an adjacency matrix 'W'. Using sine proxy."
            B = zeros(Float64, n_obs, nbins)
            for m in 1:nbins
                B[:, m] .= sin.(pi * m .* (coords[:, 1] .+ coords[:, 2]) ./ (c_max[1] + c_max[2]))
            end
        else
            n = size(W, 1)
            centering_matrix = I - (1/n) * ones(n, n)
            W_centered = centering_matrix * W * centering_matrix
            eigen_decomp = eigen(Symmetric(Matrix(W_centered)))
            B = eigen_decomp.vectors[:, end-nbins+1:end]
        end

    elseif type == "invdist"
        B = zeros(Float64, n_obs, nbins)
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        centers = [(w,x,y,z) for w in range(c_min[1],c_max[1],length=n_marginal), x in range(c_min[2],c_max[2],length=n_marginal), y in range(c_min[3],c_max[3],length=n_marginal), z in range(c_min[4],c_max[4],length=n_marginal)][:]
        for m in 1:min(m_total, length(centers))
            dist_sq = (coords[:, 1] .- centers[m][1]).^2 .+ (coords[:, 2] .- centers[m][2]).^2 .+ (coords[:, 3] .- centers[m][3]).^2 .+ (coords[:, 4] .- centers[m][4]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end

    elseif type == "kriging"
        B = zeros(Float64, n_obs, nbins)
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        centers = [(w,x,y,z) for w in range(c_min[1],c_max[1],length=n_marginal), x in range(c_min[2],c_max[2],length=n_marginal), y in range(c_min[3],c_max[3],length=n_marginal), z in range(c_min[4],c_max[4],length=n_marginal)][:]
        for m in 1:min(m_total, length(centers))
            dist_sq = ((coords[:, 1] .- centers[m][1]).^2 ./ ls_1^2) .+ ((coords[:, 2] .- centers[m][2]).^2 ./ ls_2^2) .+ ((coords[:, 3] .- centers[m][3]).^2 ./ ls_3^2) .+ ((coords[:, 4] .- centers[m][4]).^2 ./ ls_4^2)
            B[:, m] .= exp.(-dist_sq ./ 2.0)
        end

    else
        B = ones(n_obs, 1)
    end

    return B
end

 


function get_inits(model::DynamicPPL.Model; refine="map", n_samples=100, optimizer=LBFGS(), max_iters=500, maxtime=60.0, noise=nothing)
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: A utility for generating sensible initial values for MCMC sampling. It uses either a
    #          heuristic based on prior samples or refines these initial values via Maximum A
    #          Posteriori (MAP) estimation to find a good starting point for the sampler.
    # Inputs: model, and optional parameters for refinement method and optimization.
    # Outputs: A `DynamicPPL.InitFromParams` object containing the initial values.
    println("--- Generating Initial Parameters ---")

    # 1. Heuristic Initialization from Prior Samples
    samples = [Dict(pairs(rand(model))) for _ in 1:n_samples]
    init_dict = Dict{Symbol, Any}()

    if !isnothing(noise)
        init_dict[:noise] = noise
    end

    for k in keys(samples[1])
        ks = Symbol(k)
        vals = [s[k] for s in samples]

        # Dirac/Fixed parameter check
        if all(v -> v == vals[1], vals)
            init_dict[ks] = vals[1]
            continue
        end

        # FIX: Skip averaging if the elements do not support standard arithmetic (e.g. Cholesky)
        if vals[1] isa Cholesky || vals[1] isa LKJCholesky
            init_dict[ks] = vals[1]
            continue
        end

        if vals[1] isa AbstractVector
            # Latent fields are centered at zero for stability
            init_dict[ks] = zeros(eltype(vals[1]), length(vals[1]))
        else
            mu = median(vals)
            s_name = string(ks)
            if occursin(r"sigma|ls_|lengthscale|sig_", s_name)
                init_dict[ks] = max(0.1, mu)
            elseif occursin("rho", s_name)
                init_dict[ks] = clamp(mu, -0.99, 0.99)
            elseif occursin("phi", s_name)
                init_dict[ks] = clamp(mu, 0.01, 0.5)
            else
                init_dict[ks] = mu
            end
        end
    end

    # 2. Optimization Refinement
    if refine == "map"
        try
            println("Refining inits with Maximum A Posteriori (MAP)...")
            map_res = maximum_a_posteriori(model, optimizer;
                initial_params=DynamicPPL.InitFromParams(NamedTuple(init_dict)),
                iterations=max_iters, maxtime=maxtime)
            return DynamicPPL.InitFromParams(NamedTuple(map_res.params))
        catch e
            @warn "MAP refinement failed ($e). Using heuristic inits."
        end
    end

    return DynamicPPL.InitFromParams(NamedTuple(init_dict))
end

 
 
# --- 4. Trait Retrieval Helpers ---
# These helpers standardise the coordinate mapping from struct types to the metadata index vectors.

# Maps Spatial manifolds to the 's_idx' column in the data source
# manifold_indices(::SpatialManifold, M) = M.s_idx

# Maps Temporal manifolds to the 't_idx' column (time steps)
# manifold_indices(::TemporalManifold, M) = M.t_idx

# Maps Seasonal manifolds to the 'u_idx' column (periodic bins)
# manifold_indices(::SeasonalManifold, M) = M.u_idx


    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal helper to parse mixed-effects terms from the formula string (e.g.,
              "me(1 | group)") and create the necessary index mappings for group-level random
              effects.
    """

function create_mixed_indices(term::String, data::DataFrame)
    # Parses terms like 'me(1 | group)' or 'me(x | group)'
    m = match(r"me\((.*)\|(.*)\)", term)
    isnothing(m) && return nothing
    
    lhs = strip(m.captures[1])
    group_var = Symbol(strip(m.captures[2]))
    
    group_data = data[!, group_var]
    levels = unique(group_data)
    group_map = Dict(v => i for (i, v) in enumerate(levels))
    indices = [group_map[v] for v in group_data]
    
    return (indices = indices, n_cat = length(levels), name = group_var)
end


 

function create_fixed_design(formula_rhs::AbstractString, data::Union{DataFrame, NamedArray}; contrasts=Dict{Symbol, Any}())
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A utility that creates the fixed-effects design matrix (`X`) from a formula string.
              It handles `as_factor` directives for categorical variables and prunes `bstm`-specific
              module calls before passing the formula to `StatsModels.jl`.
    """
    # # 1. Synchronize Internal Data Frame
    df_internal = data isa NamedArray ? DataFrame(data, :auto) : copy(data)
    if data isa NamedArray
        rename!(df_internal, names(data, 2))
    end

    # # 2. Process as_factor() Directives
    # Identified in v19.19.10 to prevent numerical collapse of categorical predictors
    factor_regex = r"as_factor\(\s*(\w+)\s*\)"
    factor_matches = eachmatch(factor_regex, formula_rhs)

    for m in factor_matches
        var_name_str = m.captures[1]
        var_sym = Symbol(var_name_str)

        if hasproperty(df_internal, var_sym)
            df_internal[!, var_sym] = CategoricalArrays.categorical(df_internal[!, var_sym])
        end
    end

    # # 3. Dynamic BSTM Module Wrapper Pruning
    # Rationale: Instead of hard-coding keywords, we construct the regex from the taxonomy registry.
    # Requirement: Modules like Spatial, Temporal, etc., must be removed before StatsModels parsing.
    keywords_pattern = join(collect(BSTM_MODULE_KEYWORDS), "|")
    clean_rhs = replace(formula_rhs, Regex("(" * keywords_pattern * ")\\(.*?\\)", "i") => "")
    
    # Cleanup remaining artifacts from the stripping process
    final_rhs_string = replace(clean_rhs, "as_factor(" => "(")
    final_rhs_string = isempty(strip(final_rhs_string)) ? formula_rhs : final_rhs_string
    final_rhs_string = strip(replace(final_rhs_string, r"\\+\\s*\\+" => "+"))
    final_rhs_string = strip(replace(final_rhs_string, r"^\\+|\\+$" => ""))

    # # 4. Intercept Only Fallback
    if isempty(strip(final_rhs_string)) || final_rhs_string == "1"
        return NamedArray(ones(size(df_internal, 1), 1), (1:size(df_internal, 1), [:Intercept]))
    end

    # # 5. Technical Design Matrix Expansion
    try
        placeholder_name = :__y_placeholder
        df_internal[!, placeholder_name] = zeros(size(df_internal, 1))

        # Generating the formula expression for Main scope evaluation
        formula_expression = Meta.parse("@formula($placeholder_name ~ $final_rhs_string)")
        dynamic_formula = Main.eval(formula_expression)

        # Applying contrast schema and model matrix expansion
        data_schema = StatsModels.schema(dynamic_formula, df_internal, contrasts)
        applied_formula = StatsModels.apply_schema(dynamic_formula, data_schema, StatsModels.RegressionModel)

        _, model_matrix_numeric = StatsModels.modelcols(applied_formula, df_internal)
        coefficient_labels = StatsModels.coefnames(applied_formula.rhs)

        # Standardizing label types for NamedArray construction
        label_vector = coefficient_labels isa AbstractString ? [Symbol(coefficient_labels)] : Symbol.(coefficient_labels)

        return NamedArray(model_matrix_numeric, (1:size(model_matrix_numeric, 1), label_vector))

    catch design_error
        @warn "BSTM Registry: create_fixed_design expansion failed for: $final_rhs_string. Error: $design_error"
        return NamedArray(ones(size(df_internal, 1), 1), (1:size(df_internal, 1), [:Intercept]))
    end
end

 

# 1. PSIS-LOO Implementation for BSTM
# Rationale: Standardizing the extraction of log-likelihood matrices to provide 
# Expected Log Pointwise Predictive Density (ELPD) estimates.
function bstm_loo(model_obj::DynamicPPL.Model, chain; alpha=0.05)    
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: A utility for performing Leave-One-Out Cross-Validation using Pareto Smoothed Importance
    #          Sampling (PSIS-LOO) to assess a model's out-of-sample predictive accuracy.
    # Inputs: model_obj, chain, alpha.
    # Outputs: A NamedTuple containing the LOO object, metrics, log-likelihood matrix, and Pareto k values.
    println("--- Starting BSTM PSIS-LOO Audit v19.38.6 ---")

    # # 1. Metadata and Architecture Extraction
    # Rationale: M contains the configuration and technical registry required for reconstruction.
    M = model_obj.args.M
    raw_arch = get(M, :model_arch, "univariate")

    # # 2. Technical Dispatch Resolution
    # Mapping the configuration string to the architectural dispatch types.
    arch_type = if raw_arch == "univariate"
        UnivariateArchitecture()
    elseif raw_arch == "multivariate"
        MultivariateArchitecture()
    elseif raw_arch == "multifidelity" || raw_arch == "nested"
        MultifidelityArchitecture()
    else
        UnivariateArchitecture()
    end

    # # 3. Latent Manifold Reconstruction for Likelihood Registry
    # Rationale: _reconstruct generates the [Samples x Observations] log-likelihood matrix.
    # We utilize alpha for consistent summarization during the recovery phase.
    println("Audit: Recovering pointwise log-likelihood registry...")
    res = _reconstruct(arch_type, "loo_recovery", chain, M, nothing, alpha)

    # # 4. Matrix Extraction and Validation
    # Rationale: Ensuring the log_lik_matrix matches the observation grid dimensions.
    log_lik = res.log_lik_matrix
    n_samples, n_obs = size(log_lik)

    println("Audit: Processing ", n_samples, " samples for ", n_obs, " observations.")

    # # 5. PSIS-LOO Calculation via PosteriorStats
    # Rationale: LOO-CV provides a reliable estimate of out-of-sample predictive performance.
    loo_result = nothing
    try
        loo_result = loo(log_lik)
    catch e
        @error "BSTM Selection Error: PSIS-LOO calculation failed. Error: " * string(e)
        return nothing
    end

    println("\n--- BSTM Model Selection Report ---")
    println("Expected Log Pointwise Predictive Density (ELPD): ", round(loo_result.estimates[:elpd_loo, :estimate], digits=2))
    println("Effective Number of Parameters (p_loo):          ", round(loo_result.estimates[:p_loo, :estimate], digits=2))
    println("LOO Information Criterion:                       ", round(loo_result.estimates[:looic, :estimate], digits=2))

    # Check for influential observations (k > 0.7)
    # Rationale: Identifying data points where the importance weight is unstable.
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
        log_lik_matrix = log_lik,
        pareto_k = pareto_k
    )
end



# 2. Bayes Factor Suite (Manifold Comparison)
function compare_manifolds(loo_a_report, loo_b_report; model_names=["Model_A", "Model_B"])    
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: A utility for formal model comparison between two fitted `bstm` models. It uses
    #          their PSIS-LOO results to compute the difference in Expected Log Pointwise
    #          Predictive Density (ELPD) and provides a statistical basis for model selection.
    # Inputs: loo_a_report, loo_b_report, model_names.
    # Outputs: A NamedTuple containing the comparison table, ELPD difference, and LOO objects.
    # Description: Performs formal model selection between two BSTM manifold candidates.
    # Rationale: Standardizing the assessment of ELPD differences and complexity trade-offs.
    # Requirements: Absolute parity with the PSIS-LOO metrics.

    println("--- Starting BSTM Manifold Comparison ---")

    # # 1. LOO Object Extraction
    # Rationale: Extracting the underlying PosteriorStats LOO objects for comparison.
    loo_a = loo_a_report.loo_obj
    loo_b = loo_b_report.loo_obj

    # # 2. Formal Selection Metric Calculation
    # Rationale: The difference in ELPD is the primary metric for out-of-sample performance.
    # We utilize the compare function from PosteriorStats to compute deltas and standard errors.
    comparison_stats = nothing
    try
        comparison_stats = compare([loo_a, loo_b])
    catch e
        @error "BSTM Comparison Error: Selection suite failed. Error: " * string(e)
        return nothing
    end

    # # 3. Parameter and Diagnostic Extraction
    # Rationale: Collecting effective parameter counts (p_loo) to assess complexity.
    p_loo_a = loo_a_report.metrics.p_loo
    p_loo_b = loo_b_report.metrics.p_loo

    elpd_a = loo_a_report.metrics.elpd
    elpd_b = loo_b_report.metrics.elpd

    # # 4. Report Generation
    println("\n--- BSTM Manifold Selection Registry ---")
    println("Model A (", model_names[1], "): ELPD = ", round(elpd_a, digits=2), " | p_loo = ", round(p_loo_a, digits=2))
    println("Model B (", model_names[2], "): ELPD = ", round(elpd_b, digits=2), " | p_loo = ", round(p_loo_b, digits=2))

    diff_elpd = elpd_a - elpd_b
    println("\nELPD Delta (A - B): ", round(diff_elpd, digits=2))

    # Interpretation Logic
    # Rationale: If |diff_elpd| > 4, the difference is generally considered significant.
    if abs(diff_elpd) > 4.0
        winning_model = diff_elpd > 0 ? model_names[1] : model_names[2]
        println("CONCLUSION: ", winning_model, " is statistically preferred based on predictive density.")
    else
        println("CONCLUSION: Competing manifold structures provide indistinguishable predictive density.")
    end

    # # 5. Table Construction
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


function bstm_cv_orchestrator(
    formula::String, 
    data::DataFrame; 
    method::Symbol = :lolo, 
    lolo_var::Symbol = :s_idx, 
    n_folds::Int = 5, 
    n_samples::Int = 500, 
    sampler = NUTS(500, 0.65), 
    alpha = 0.05, 
    kwargs...
)    
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: An orchestration utility for performing cross-validation. It supports standard
    #          k-fold and Leave-One-Location-Out (LOLO) strategies to assess model performance on
    #          held-out data.
    # Inputs: formula, data, and optional parameters for CV method, sampler, etc.
    # Outputs: A NamedTuple containing fold results and summary metrics.
    # # 1. Metadata Discovery and Outcome Resolution
    # The formula is decomposed to identify the primary response variable and module requirements.
    meta_discovery = decompose_bstm_formula(formula)
    response_name = Symbol(meta_discovery.outcomes[1])

    # # 2. Partition Strategy Selection
    # Establishing fold indices based on spatiotemporal logic or random sampling.
    folds_indices = Vector{Vector{Int}}()

    if method == :lolo
        # Leave-One-Location-Out: Grouping indices by the spatial unit identifier.
        unique_locs = unique(data[!, lolo_var])
        for loc in unique_locs
            push!(folds_indices, findall(x -> x == loc, data[!, lolo_var]))
        end
    else
        # Standard K-Fold: Random permutation of row indices.
        n_obs = size(data, 1)
        row_indices = Random.randperm(n_obs)
        fold_size = div(n_obs, n_folds)
        for i in 1:n_folds
            idx_start = (i - 1) * fold_size + 1
            idx_end = i == n_folds ? n_obs : i * fold_size
            push!(folds_indices, row_indices[idx_start:idx_end])
        end
    end

    # # 3. Cross-Validation Loop
    fold_results = []
    n_actual_folds = length(folds_indices)

    for (f_idx, test_idx) in enumerate(folds_indices)
        # Splitting dataset into training and testing partitions.
        # Training mask is constructed to exclude the test indices.
        train_mask = trues(size(data, 1))
        train_mask[test_idx] .= false
        
        train_data = data[train_mask, :]
        test_data = data[test_idx, :]

        # # 4. Modular Training Configuration
        # Pre-configuring the model to ensure technical registries (W, s_N, t_N) are consistent.
        # bstm_config resolves the manifold registry M.
        opt_train = bstm_config(formula, train_data; kwargs...)

        # # 5. Model Execution
        # Dispatching to the modular univariate or multivariate supervisor.
        model_train = bstm(opt_train)
        
        # Posterior sampling using the requested sampler configuration.
        chain_train = sample(model_train, sampler, n_samples, progress = false)

        # # 6. Manifold Projection (Out-of-Sample)
        # Using the standardized predict() function which handles reconstruction of S, T, and Smooth basis.
        # This ensures that PS (Prediction Surface) alignment is consistent with the modular BSTM taxonomy.
        res_pred = predict(model_train, chain_train, test_data, n_samples = div(n_samples, 2), alpha = alpha)

        # # 7. Performance Assessment
        # Extracting denoised expectations for the test partition.
        y_test_obs = test_data[!, response_name]
        y_test_pred = res_pred.predictions_denoised.mean

        # Verification of dimensional parity between prediction and observation.
        if length(y_test_obs) == length(y_test_pred)
            residuals = y_test_obs .- y_test_pred
            rmse = sqrt(Statistics.mean(residuals.^2))
            
            # R-Squared calculation with safety floor for variance.
            ss_res = sum(residuals.^2)
            ss_tot = sum((y_test_obs .- Statistics.mean(y_test_obs)).^2)
            r2 = 1.0 - (ss_res / (ss_tot + 1e-15))

            push!(fold_results, (fold=f_idx, rmse=rmse, r2=r2))
        else
            @warn "Fold $f_idx: Prediction length mismatch. Observed: $(length(y_test_obs)), Predicted: $(length(y_test_pred))"
        end
    end

    # # 8. Aggregate Reporting
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

 


function bstm(config::NamedTuple)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Dispatches a pre-built configuration object to the appropriate model supervisor.
    # Inputs: config::NamedTuple.
    # Outputs: A Turing.jl model instance.
    arch = get(config, :model_arch, "univariate")
    if arch == "multivariate"; return bstm_multivariate(config)
    elseif arch == "multifidelity"; return bstm_multifidelity(config)
    else; return bstm_univariate(config); end
end

function bstm(formula::String, data::DataFrame; kwargs...)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: The primary user-facing function that provides a formula-driven interface.
    # Inputs: formula::String, data::DataFrame, and other keyword arguments.
    # Outputs: A Turing.jl model instance.
    options = bstm_config(formula, data; kwargs...)
    return bstm(options)
end


function bstm_config(formula::String, data::DataFrame; kwargs...)
    # v1.0.3 (2026-06-29 18:10:00)
    # Purpose: Parses the bstm formula and data to create a model configuration NamedTuple.
    # Inputs: formula::String, data::DataFrame, and other keyword arguments.
    # Outputs: A NamedTuple containing the full model specification.    
    metadata = decompose_bstm_formula(formula)

    opt_dict = Dict{Symbol, Any}(kwargs)
    opt_dict[:y_obs_vars] = metadata.outcomes
    opt_dict[:data] = data
    opt_dict[:y_N] = size(data, 1)
    opt_dict[:add_intercept] = metadata.has_intercept

    # Pre-evaluate any expressions passed as keyword arguments or in the formula string.
    # This allows passing variables from the calling scope, e.g., W=my_matrix.
    # This must be done before the main module processing loop.
    for (key, val) in opt_dict
        if val isa Expr
            try
                opt_dict[key] = Core.eval(Main, val)
            catch e; @error "Failed to evaluate keyword argument `$key` with expression `$val`."; rethrow(e); end
        end
    end
    for (key, mod_data) in metadata.modules
        if haskey(mod_data, :params)
            for (param_key, param_val) in mod_data[:params]
                # Evaluate expressions and symbols that represent variables in the Main scope.
                # This allows users to pass variables from their workspace into the formula string.
                # A check for `isdefined` is used for Symbols to avoid attempting to evaluate
                # option keywords (e.g., model=bym2, which is parsed to the symbol :bym2).
                should_eval = param_val isa Expr || (param_val isa Symbol && isdefined(Main, param_val))

                if should_eval
                    try
                        mod_data[:params][param_key] = Core.eval(Main, param_val)
                    catch e; @error "Failed to evaluate formula parameter `$param_key` with expression `$param_val` in module `$key`."; rethrow(e); end
                end
            end
        end
    end

     
    # Pre-process common coordinate columns
    # FIX: Use `hasproperty` for DataFrames instead of `haskey`.
    if hasproperty(data, :s_x) && hasproperty(data, :s_y)
        opt_dict[:s_x] = data.s_x
        opt_dict[:s_y] = data.s_y
        opt_dict[:s_coord] = hcat(data.s_x, data.s_y)
    end

    # Initialize default model components
    opt_dict[:model_space] = get(opt_dict, :model_space, "none")
    opt_dict[:model_time] = get(opt_dict, :model_time, "none")
    opt_dict[:model_season] = get(opt_dict, :model_season, "none")
    opt_dict[:model_st] = get(opt_dict, :model_st, "none")

    initial_hp = get(opt_dict, :hyperpriors, Dict{Any, Any}())
    hyperprior_registry = Dict{String, Any}(string(k) => v for (k, v) in initial_hp)
    opt_dict[:hyperpriors] = hyperprior_registry

    basis_matrices_registry = Dict{Symbol, Any}()
    manifolds_registry = []

    # Process modules to populate opt_dict and registries
    for (key_str, mod_data) in metadata.modules
        m_type = mod_data[:type]

        if haskey(MODULE_PROCESSORS, m_type)
            processor_func = MODULE_PROCESSORS[m_type]
            processor_func(opt_dict, mod_data, (basis_matrices_registry=basis_matrices_registry, manifolds_registry=manifolds_registry), hyperprior_registry)
        end

        # Skip modules that don't create a primary manifold in the main registry
        if m_type in [:interaction_composition, :observationprocess, :intercept, :nested, :fixed]
            continue
        end

        manifold_obj = resolve_technical_primitive(mod_data, opt_dict, hyperprior_registry, :pcpriors)

        # Determine n for build_structure_template
        n_units = 0
        if m_type == :spatial
            n_units = get(opt_dict, :s_N, 0)
        elseif m_type == :temporal
            n_units = get(opt_dict, :t_N, 0)
        elseif m_type == :seasonal
            n_units = get(opt_dict, :u_N, 0)
        elseif m_type == :smooth
            reg_key = Symbol(join(mod_data[:variables], "_"))
            if haskey(basis_matrices_registry, reg_key)
                n_units = size(basis_matrices_registry[reg_key], 2) # number of basis functions
            end
        elseif m_type == :mixed
             n_units = mod_data[:params][:n_cat]
        end

        if n_units == 0 && !(manifold_obj isa GP) # GP doesn't need pre-set n
             @warn "Could not determine size for manifold '$key_str'. Skipping."
             continue
        end

        # Get the template matrix and scaling factor
        template_info = build_structure_template(Symbol(lowercase(string(typeof(manifold_obj)))), n_units; W=get(opt_dict, :W, nothing))

        spec_params = mod_data[:params]
        if manifold_obj isa GP
            if !haskey(spec_params, :coords)
                # This is for smooth(x,y, model='gp') where coords are not set by process_spatial_module
                spec_params[:coords] = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
            end
            spec_params[:coords] = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
        end

        spec = (
            key = Symbol(key_str),
            domain = m_type,
            var = isempty(mod_data[:variables]) ? :none : Symbol(mod_data[:variables][1]),
            manifold_obj = manifold_obj,
            Q_template = template_info.matrix,
            scaling_factor = template_info.scaling_factor,
            params = spec_params
        )
        push!(manifolds_registry, spec)
    end

    opt_dict[:manifolds] = manifolds_registry
    opt_dict[:basis_matrices] = basis_matrices_registry
    opt_dict[:formula] = formula

    # Post-module processing for features requiring computed components
    if get(opt_dict, :use_sv, false)
        # FIX: Use `hasproperty` for DataFrames instead of `haskey`.
        if hasproperty(data, :s_x) && hasproperty(data, :s_y) && haskey(opt_dict, :t_idx)
            m_rff_sigma = get(opt_dict, :M_rff_sigma, 20)
            # Use observation-level coordinates for the volatility surface
            coords_st = hcat(data.s_x, data.s_y, opt_dict[:t_idx])
            W_sigma, b_sigma = generate_informed_rff_params(coords_st, m_rff_sigma)
            opt_dict[:vol_proj] = (coords_st * W_sigma) .+ b_sigma'
        else
             @warn "Stochastic volatility requires spatial coordinates (:s_x, :s_y) in `data` and a temporal index. SV disabled."
             opt_dict[:use_sv] = false
        end
    end

    # Process fixed effects
    fixed_effects_formula_part = join(metadata.fixed_effects, " + ")
    add_intercept_to_formula = get(opt_dict, :add_intercept, false) && !any(m -> m[:type] == :intercept, values(metadata.modules))

    if add_intercept_to_formula
        fixed_effects_formula_part = isempty(strip(fixed_effects_formula_part)) ? "1" : "1 + " * fixed_effects_formula_part
    end

    if !isempty(strip(fixed_effects_formula_part))
        opt_dict[:Xfixed] = create_fixed_design(fixed_effects_formula_part, data; contrasts=get(opt_dict, :contrasts, Dict()))
        opt_dict[:Xfixed_N] = size(opt_dict[:Xfixed], 2)
    else
        opt_dict[:Xfixed] = NamedArray(zeros(size(data, 1), 0), (1:size(data,1), Symbol[]))
        opt_dict[:Xfixed_N] = 0
    end

    # Process outcomes
    opt_dict[:outcomes_N] = length(metadata.outcomes)
    if length(metadata.outcomes) == 1
        opt_dict[:y_obs] = data[!, metadata.outcomes[1]]
        opt_dict[:model_arch] = get(opt_dict, :model_arch, "univariate") # Respect user override
    else
        opt_dict[:y_obs] = Matrix(data[!, metadata.outcomes])
        opt_dict[:model_arch] = "multivariate"
        opt_dict[:outcomes_N] = length(metadata.outcomes)
    end

    # Set defaults for observation-level parameters if not provided
    if !haskey(opt_dict, :weights); opt_dict[:weights] = ones(Float64, opt_dict[:y_N]); end
    if !haskey(opt_dict, :trials); opt_dict[:trials] = ones(Int, opt_dict[:y_N]); end

    opt_dict[:model_family] = get(opt_dict, :model_family, "gaussian")

    return NamedTuple(opt_dict)
end



function get_optimal_sampler(model_obj::DynamicPPL.Model; sampler_choice=:auto, target_acceptance=0.65, adaptation_steps=100, n_particles=20, hmc_leapfrog_steps=10)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Automatically selects an optimal MCMC sampler (or a composite Gibbs sampler).
    # Inputs: model_obj, target_acceptance, adaptation_steps, n_particles.
    # Outputs: A Turing.jl sampler object.
    # Note: This simplified logic only identifies discrete parameters. The NUTS sampler in a Gibbs
    #       composition automatically handles all remaining (continuous) parameters, removing the
    #       need to explicitly partition them. This makes the function more concise and robust.

    # # 1. Parameter Discovery and Type Introspection
    # Inspect the model's VarInfo to get metadata about each parameter's prior distribution.
    # This is a robust method to determine if a parameter's support is discrete or continuous.
    vi = DynamicPPL.VarInfo(model_obj)
    vns = DynamicPPL.keys(vi)
    
    discrete_params = Symbol[]
    all_gaussian_priors = true
    num_continuous_params = 0

    for vn in vns
        try
            # Attempt to get the distribution for the variable using the modern API.
            dist = DynamicPPL.getdist(vi, vn)
            sym = Symbol(vn) # Convert VarName to Symbol

            # Check if the distribution's support is discrete.
            if Distributions.value_support(typeof(dist)) == Distributions.Discrete
                if !(sym in discrete_params)
                    push!(discrete_params, sym)
                end
            else # Continuous parameter
                num_continuous_params += 1
                # Check if the prior is a Gaussian type. If not, set the flag to false.
                if !(dist isa Normal || dist isa MvNormal || dist isa Truncated{<:Normal})
                    all_gaussian_priors = false
                end
            end
        catch e
            # Not all variables in VarInfo have a corresponding distribution
            # (e.g., derived quantities). `getdist` will throw an error for these,
            # which can be safely ignored.
            if !(e isa KeyError)
                # rethrow(e) # Optionally rethrow other unexpected errors
                all_gaussian_priors = false # Assume non-Gaussian if prior is not found
            end
        end
    end

    # # 2. Sampler Construction and Dispatch
    # Based on user choice or an informed automatic selection, construct the most appropriate sampler.
    
    local continuous_sampler
    if sampler_choice == :auto
        if all_gaussian_priors && num_continuous_params > 0
            println("Info: All continuous priors appear Gaussian. Selecting ESS for the continuous part.")
            continuous_sampler = ESS()
        else
            println("Info: Model contains non-Gaussian continuous priors or complex structure. Selecting NUTS for the continuous part.")
            continuous_sampler = NUTS(adaptation_steps, target_acceptance)
        end
    elseif sampler_choice == :nuts
        continuous_sampler = NUTS(adaptation_steps, target_acceptance)
    elseif sampler_choice == :hmc
        # HMC is a good alternative but requires manual tuning of leapfrog steps.
        continuous_sampler = HMC(0.1, hmc_leapfrog_steps) # Default step size 0.1
    elseif sampler_choice == :mh
        # Metropolis-Hastings is a gradient-free option, useful for non-differentiable models.
        continuous_sampler = MH()
    elseif sampler_choice == :ess
        # Elliptical Slice Sampling is efficient for models with Gaussian priors.
        continuous_sampler = ESS()
    elseif sampler_choice == :slice
        # Slice sampling is another robust gradient-free method.
        continuous_sampler = Slice()
    else
        @warn "Unknown sampler_choice ':$sampler_choice'. Defaulting to NUTS."
        continuous_sampler = NUTS(adaptation_steps, target_acceptance)
    end

    if isempty(discrete_params)
        # If no discrete parameters are found, return the chosen sampler for the continuous space.
        println("Info: No discrete parameters found. Using sampler: $(nameof(typeof(continuous_sampler))).")
        return continuous_sampler
    else
        # If discrete parameters exist, a composite Gibbs sampler is required.
        # Particle Gibbs (PG) will handle the discrete parameters.
        # The chosen continuous sampler will handle all other (continuous) parameters.
        println("Info: Discrete parameters found: ", discrete_params, ". Using composite Gibbs sampler (PG + $(nameof(typeof(continuous_sampler)))).")
        return Gibbs(PG(n_particles, discrete_params...), continuous_sampler)
    end
end




function _resolve_obs_param!(opt_dict, params, data, param_keys, target_key)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Resolves observation-level parameters (e.g., offsets, weights) from symbols to data vectors.
    # Inputs: opt_dict, params, data, param_keys, target_key.
    # Outputs: Modifies opt_dict in place.
    # Note: It checks for a symbol in the `params` dict and extracts the corresponding column from the `data` DataFrame.
    for key in param_keys
        if haskey(params, key)
            val = params[key]
            if val isa Symbol && hasproperty(data, val)
                opt_dict[target_key] = data[!, val]
            elseif val isa AbstractVector
                opt_dict[target_key] = val
            else
                @warn "Observation parameter ':$val' for '$target_key' not found in data or is not a vector. Ignoring."
            end
            return # Exit after the first matching key is found and processed
        end
    end
end
 
