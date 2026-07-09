# CORE FORMULA PARSING UTILITIES




# v2.1.0: Expanded keyword registry
const BSTM_MODULE_KEYWORDS = Set([ 
    :intercept, :spatial, :temporal, :seasonal, :smooth, :fixed,
    :nested, :eigen, :mixed, :dynamics, :spacetime, :interaction
]);



# This file contains refactored versions of the core formula parsing and
# configuration functions for the bstm framework. These functions provide
# a more robust and maintainable pipeline for model specification.


function split_terms_at_depth(input::AbstractString, sep::AbstractString)
    # v1.2.0 (2026-07-08)
    # Purpose: Splits a string by a separator, but only at a parenthesis depth of zero.
    #          This is crucial for correctly parsing complex formula terms that may
    #          contain nested function calls.
    # Inputs: input (string), sep (separator string).
    # Outputs: A vector of strings.
    terms = String[]
    current_term = ""
    depth = 0
    i = 1
    sep_len = length(sep)

    while i <= length(input)
        char = input[i]
        if char == '(' || char == '['
            depth += 1
        elseif char == ')' || char == ']'
            depth -= 1
        end

        if depth == 0 && i <= length(input) - sep_len + 1 && SubString(input, i, i + sep_len - 1) == sep
            push!(terms, strip(current_term))
            current_term = ""
            i += sep_len
            continue
        end
        
        current_term *= char
        i += 1
    end
    if !isempty(strip(current_term))
        push!(terms, strip(current_term))
    end
    
    return terms
end


function _parse_value(val_str::String)
    # v1.2.0 (2026-07-08)
    # Purpose: A helper function to parse a string value from a formula argument into
    #          the appropriate Julia type (Tuple, String, Number, or Expression).
    val_str = strip(val_str)
    if startswith(val_str, "(") && endswith(val_str, ")") # Tuple
        inner_val_str = val_str[2:end-1]
        tuple_parts = split_terms_at_depth(inner_val_str, r",\s*")
        parsed_tuple = []
        for tp in tuple_parts
            if !isempty(tp)
                push!(parsed_tuple, _parse_value(tp))
            end
        end
        return Tuple(parsed_tuple)
    elseif startswith(val_str, "\"") && endswith(val_str, "\"") # String
        return val_str[2:end-1]
    elseif startswith(val_str, ":")
        # Handle quoted symbols like :besag by converting them directly to a Symbol.
        # This ensures `model=:besag` is parsed identically to `model=besag`.
        return Symbol(val_str[2:end])
    else
        try_num = tryparse(Float64, val_str)
        if try_num !== nothing
            return try_num
        else
            return Meta.parse(val_str)
        end
    end
end

function _parse_value(val_str::SubString{String})
    _parse_value(String(val_str))
end

function _add_parsed_arg!(args_dict::Dict{Symbol, Any}, positional_args::Vector{Any}, arg_val::String)
    # v1.2.0 (2026-07-08)
    # Purpose: A helper to add a parsed argument to either the keyword argument
    #          dictionary or the positional argument vector.
    if contains(arg_val, "=")
        key_val = Base.split(arg_val, "=", limit=2)
        key = Symbol(strip(key_val[1]))
        val_str = String(strip(key_val[2]))
        args_dict[key] = _parse_value(val_str)
    else
        push!(positional_args, _parse_value(arg_val))
    end
end

function _parse_arguments_string(args_str::String)
    # v1.2.0 (2026-07-08)
    # Purpose: Parses the argument string from within a module call's parentheses.
    args_dict = Dict{Symbol, Any}()
    positional_args = []
    current_arg = ""
    depth = 0

    for char in args_str
        if char == ',' && depth == 0
            arg_val = String(strip(current_arg))
            if !isempty(arg_val)
                _add_parsed_arg!(args_dict, positional_args, arg_val)
            end
            current_arg = ""
        else
            current_arg *= char
            if char == '(' || char == '['
                depth += 1
            elseif char == ')' || char == ']'
                depth -= 1
            end
        end
    end

    arg_val = String(strip(current_arg))
    if !isempty(arg_val)
        _add_parsed_arg!(args_dict, positional_args, arg_val)
    end

    if !isempty(positional_args)
        args_dict[:positional_args] = positional_args
    end

    return args_dict
end

function _parse_single_manifold_term(term_str::String)
    # v1.2.0 (2026-07-08)
    # Purpose: Parses a single module call string (e.g., "spatial(...)") into a structured NamedTuple.
    term_str = strip(term_str)
    m = match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$", term_str)
    
    if m === nothing
        error("Internal Parser Error: Expected a module call (e.g., 'name(...)'), but got non-module term '$term_str'.")
    end

    module_name = Symbol(m.captures[1])
    args_str = String(m.captures[2])
    args_dict = _parse_arguments_string(args_str)

    return (module_type = module_name, args = args_dict)
end 

function _parse_rhs_expression(term_str::String)
    # v1.2.0 (2026-07-08)
    # Purpose: A recursive descent parser for the formula's right-hand side, respecting operator precedence.
    parts = split_terms_at_depth(term_str, " ⊕ ")
    if length(parts) > 1; return (type=:operator, op=:direct_sum, children=[_parse_rhs_expression(p) for p in parts]); end

    parts = split_terms_at_depth(term_str, " |> ")
    if length(parts) > 1; return (type=:operator, op=:pipe, children=[_parse_rhs_expression(parts[1]), _parse_rhs_expression(join(parts[2:end], " |> "))]); end

    parts = split_terms_at_depth(term_str, " ⊗ ")
    if length(parts) > 1; return (type=:operator, op=:kronecker_product, children=[_parse_rhs_expression(p) for p in parts]); end

    parts = split_terms_at_depth(term_str, " ∘ ")
    if length(parts) > 1; return (type=:operator, op=:composition, children=[_parse_rhs_expression(p) for p in parts]); end

    return _parse_single_manifold_term(term_str)
end

 

function _categorize_rhs_nodes!(nodes, modules)
    # v2.6.0 (2026-07-08)
    # Purpose: Traverses the AST from the parser and categorizes the nodes into a
    #          dictionary of modules, giving each a unique key.
    for node in nodes
        if hasproperty(node, :type) && node.type == :operator
            key = "composed_$(length(modules)+1)"
            modules[key] = (module_type = :interaction, args = Dict(:operator => node.op, :components => node.children))

        elseif hasproperty(node, :module_type) && node.module_type in BSTM_MODULE_KEYWORDS
            key_parts = [string(node.module_type)]
            if haskey(node.args, :positional_args) && !isempty(node.args[:positional_args])
                pos_arg_str = string(node.args[:positional_args][1])
                push!(key_parts, pos_arg_str)
            end
            base_key = join(key_parts, "_")
            module_key = base_key
            counter = 1
            while haskey(modules, module_key)
                counter += 1
                module_key = base_key * "_$counter"
            end
            modules[module_key] = node
        else
            @warn "Unrecognized node type '$(node.module_type)' during categorization. Skipping."
        end
    end
end

function bstm_config(formula::String, data::DataFrame; calling_module::Module = Main, kwargs...)
    # v2.7.0 (2026-07-09)
    # Purpose: The main configuration engine. It orchestrates the entire process of
    #          transforming a formula string and data into a complete, model-ready
    #          configuration object.
    # Rationale: This version adds a pre-processing step to handle categorical variable
    #            conversions before the design matrix is created. This version also ensures
    #            that the categorical conversion happens before completecases filtering.

    decomposed_formula = decompose_bstm_formula(formula)

    # --- Variable Discovery and Data Filtering ---
    # Rationale: Identify all variables used in the formula to filter for complete cases
    # before any configuration is built. This prevents dimension mismatches where
    # StatsModels.jl drops rows with missings, but other parts of the model do not.
    all_vars = Set{Symbol}()
    for out_spec in decomposed_formula.outcomes
        push!(all_vars, Symbol(out_spec[:var]))
        # Also consider offsets, weights, etc. from likelihood
        for key in [:offsets, :log_offsets, :weights, :trials, :y_L, :y_U]
            if haskey(out_spec[:params], key)
                val = out_spec[:params][key]
                if val isa Symbol
                    push!(all_vars, val)
                end
            end
        end
    end
    for fe in decomposed_formula.fixed_effects
        push!(all_vars, Symbol(fe))
    end
    for (_, mod_data) in pairs(decomposed_formula.modules)
        if haskey(mod_data.args, :positional_args)
            for arg in mod_data.args[:positional_args]
                if arg isa Symbol
                    push!(all_vars, arg)
                elseif arg isa Expr
                    # Recursively find all symbols in expressions like `cov1*cov2`.
                    _extract_symbols_from_expr!(all_vars, arg)
                end
            end
        end
    end
    
    # --- Categorical Conversion (Pre-filtering) ---
    # Rationale: Convert specified columns to CategoricalArrays *before* completecases filtering.
    # This ensures `completecases` correctly handles missing values in categorical columns.
    df_processed = deepcopy(data) # Work on a copy to avoid modifying original data
    vars_to_categorize = Set{Symbol}() # Collect variables that need to be categorized
    
    # Collect variables from `fixed()` modules that might need categorization
    for (_, mod_data_nt) in decomposed_formula.modules
        if mod_data_nt.module_type == :fixed
            if haskey(mod_data_nt.args, :positional_args)
                for var_sym in mod_data_nt.args[:positional_args]
                    if var_sym isa Symbol
                        # If a contrast is specified or model=:categorical, it's categorical
                        if haskey(mod_data_nt.args, :contrast) || get(mod_data_nt.args, :model, nothing) == :categorical
                            push!(vars_to_categorize, var_sym)
                        end
                    end
                end
            end
        end
    end
    for var_sym in vars_to_categorize
        if hasproperty(df_processed, var_sym) && !(eltype(df_processed[!, var_sym]) <: CategoricalValue)
            df_processed[!, var_sym] = categorical(df_processed[!, var_sym])
        end
    end

    # --- Data Filtering for Complete Cases ---
    # Now, apply completecases on the already-categorized data.
    valid_vars_in_data = filter(v -> hasproperty(df_processed, v), all_vars)
    filtered_data = DataFrame(df_processed[completecases(df_processed, collect(valid_vars_in_data)), :])
    if nrow(filtered_data) < nrow(data)
        @warn "Missing values detected in formula variables. $(nrow(data) - nrow(filtered_data)) rows were removed."
    end

    M = _initialize_config(filtered_data, merge(Dict(kwargs), Dict(:calling_module => calling_module)))
    M[:formula] = formula
    _process_lhs!(M, decomposed_formula.outcomes)
    M[:N_cov] = 0 # Initialize N_cov, will be updated after fixed effects processing
    M[:add_intercept] = decomposed_formula.has_intercept

    if !isnothing(decomposed_formula.intercept_prior)
        M[:intercept_prior] = decomposed_formula.intercept_prior
    end

    M[:hyperpriors] = get(M, :hyperpriors, Dict{String, Any}())
    M[:prior_scheme] = get(M, :prior_scheme, :pcpriors)
    # M[:vars_to_categorize] is now handled earlier

    # Process modules (fixed effects are now handled by _replace_bare_effects and then _process_fixed_effects!)
    for (key, mod_data_nt) in decomposed_formula.modules
        mod_type = mod_data_nt.module_type
 
        # Retrieve the processor function for the current module type.
        processor! = get(MODULE_PROCESSORS, mod_type, nothing)
 
        mod_data_dict = Dict(
            :type => mod_data_nt.module_type,
            :variables => get(mod_data_nt.args, :positional_args, []),
            :params => mod_data_nt.args
        )
 
        # If a processor exists for this module, execute it.
        if !isnothing(processor!)
            processor!(M, mod_data_dict, M, M[:hyperpriors])
        end
 
        # Modules like `fixed` are processor-only and do not create a manifold.
        # After their processor runs, we skip to the next module.
        if mod_type in [:fixed]
            continue
        end
 
        manifold_obj = resolve_technical_primitive(mod_data_dict, M, M[:hyperpriors], M[:prior_scheme])
        manifold_spec_built = build_model(manifold_obj, M)

        spec = (key=Symbol(key), domain=mod_data_dict[:type], var=join(mod_data_dict[:variables], "_"), manifold_obj=manifold_obj, params=mod_data_dict[:params], Q_template=manifold_spec_built.Q_template, scaling_factor=manifold_spec_built.scaling_factor)
        push!(M[:manifolds], spec)
    end

    # Consolidate and process fixed effects after all modules have been parsed
    # The `fixed_effects` list from `decomposed_formula` is now empty due to `_replace_bare_effects`.
    # `M[:fixed_effects]` is populated by `process_fixed_module!`.
    all_fixed_effects = get(M, :fixed_effects, String[]) # This now contains the unique fixed effect names
    _process_fixed_effects!(M, get(M, :fixed_effects, String[]), decomposed_formula.has_intercept)
    _process_fixed_effects_priors!(M)
    _finalize_config!(M)

    return NamedTuple(M)
end



function _parse_single_manifold_term(term_str::SubString{String})
    _parse_single_manifold_term(String(term_str))
end
 
function _parse_rhs_expression(term_str::SubString{String})
    _parse_rhs_expression(String(term_str))
end

function parse_module_params(params_str::AbstractString)
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


"""
    resolve_hyperpriors(m_id, global_priors, module_priors, scheme)

Resolves and assigns prior distributions to manifold types, handling local overrides,
global overrides, and default schemes, including PC prior quantile constraints.
"""
function resolve_hyperpriors(m_id::Union{String, Symbol}, global_priors::Dict{String, Any}, module_priors::Dict{Symbol, Any}, scheme::Symbol=:pcpriors)
    m_id_str = string(m_id)

    defaults = if scheme == :pcpriors
        PC_PRIORS
    elseif scheme == :informative
        INFORMATIVE_PRIORS
    else
        UNINFORMATIVE_PRIORS
    end

    function get_prior(module_key::Symbol, global_key::String, default_key::String)
        # Priority 1: Local override from module parameters (e.g., sigma_prior=(1.0, 0.05))
        if haskey(module_priors, module_key)
            val = module_priors[module_key]
            if val isa Tuple
                return create_pc_prior(module_key, val)
            elseif val isa UnivariateDistribution
                return val
            end
        end

        # Priority 2: Global override from hyperpriors dictionary
        if haskey(global_priors, global_key)
            val = global_priors[global_key]
            if val isa Tuple
                return create_pc_prior(Symbol(global_key), val)
            elseif val isa UnivariateDistribution
                return val
            end
        end
        
        # Priority 3: Default from the selected scheme
        return get(defaults, default_key, nothing)
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




function _parse_module_call(module_call_str::AbstractString)
    # v1.2.0 (2026-06-29 16:13:05)
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



function _resolve_obs_param!(opt_dict, params, data, param_keys, target_key)
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
            return
        end
    end
end

  
 
function _process_fixed_effects!(M::Dict, fixed_effects_vars::Vector{String}, has_intercept::Bool)
    rhs = has_intercept ? "1" : "0"
    if !isempty(fixed_effects_vars)
        rhs_vars = join(fixed_effects_vars, " + ")
        rhs = (rhs == "0") ? rhs_vars : rhs * " + " * rhs_vars
    end
    Xfixed_named = create_fixed_design(rhs, M[:data]; contrasts=get(M, :contrasts, Dict()))
    
    # Robustness check: Ensure Xfixed has the same number of rows as y_N
    if size(Xfixed_named, 1) != M[:y_N]
        @warn "Dimension mismatch in fixed effects design matrix: Expected $(M[:y_N]) rows, but got $(size(Xfixed_named, 1)). Attempting to reconcile."
        if size(Xfixed_named, 1) < M[:y_N]
            # Pad with zeros if Xfixed_named has fewer rows
            padded_Xfixed = zeros(M[:y_N], size(Xfixed_named, 2))
            padded_Xfixed[1:size(Xfixed_named, 1), :] = Matrix(Xfixed_named)
            M[:Xfixed] = padded_Xfixed
        else # If Xfixed_named has more rows (should not happen after completecases)
            M[:Xfixed] = Matrix(Xfixed_named[1:M[:y_N], :])
        end
    else
        M[:Xfixed] = Matrix(Xfixed_named)
    end

    M[:Xfixed_N] = size(M[:Xfixed], 2)
    M[:Xfixed_names] = size(Xfixed_named, 2) > 0 ? names(Xfixed_named, 2) : Symbol[]
end

"""
    create_pc_prior(param_name::Symbol, constraint::Tuple)

Creates a prior distribution from a Penalized Complexity (PC) prior constraint.
This function interprets a tuple like `(U, α)` to generate a prior distribution
based on the type of parameter.
"""
function create_pc_prior(param_name::Symbol, constraint::Tuple)
    # #
    # This function creates a prior distribution from a PC prior constraint.
    # It dispatches based on the parameter name to apply the correct formula.
    #
    
    # # Default to upper tail constraint P(param > U) = alpha
    U, α = constraint[1], constraint[2]
    direction = length(constraint) > 2 ? constraint[3] : :upper

    if U <= 0 || α <= 0 || α >= 1
        error("Invalid PC prior constraint: U > 0 and 0 < α < 1 required.")
    end

    param_str = string(param_name)
    if occursin("sigma", param_str) || occursin("amplitude", param_str)
        # # For standard deviation parameters, use an Exponential prior.
        # # Constraint: P(σ > U) = α
        if direction != :upper; error("PC prior for sigma-like parameters only supports upper tail constraints."); end
        λ = -log(α) / U
        return Exponential(λ)

    # # Other cases for rho, lengthscale, etc., would be added here.

    else # # Assume it's a regression coefficient (beta)
        # # Constraint: P(|beta| > U) = alpha
        # # We solve for the standard deviation of a Normal(0, sigma) prior.
        sigma = -U / quantile(Normal(0, 1), α / 2)
        return Normal(0, sigma)
    end
end


"""
    _process_fixed_effects_priors!(M::Dict)

Constructs the prior distribution for the fixed effects coefficient vector (`Xfixed_beta`).
It combines default priors with any custom priors specified in the formula for the
intercept or other fixed effects, including PC prior tuples.
"""
function _process_fixed_effects_priors!(M::Dict)
    # #
    # This function constructs the prior for the vector of fixed effect coefficients.
    #
    n_fixed = get(M, :Xfixed_N, 0)
    if n_fixed == 0
        M[:Xfixed_priors_vec] = UnivariateDistribution[]
        return
    end
    
    coef_names = get(M, :Xfixed_names, Symbol[])
    custom_priors = get(M, :fixed_effects_priors, Dict{Symbol, Any}())
    intercept_prior_val = get(M, :intercept_prior, nothing)
    
    default_prior = Normal(0, 5)
    priors_vec = Vector{UnivariateDistribution}(undef, n_fixed)
    
    for i in 1:n_fixed
        coef_name_sym = coef_names[i]
        coef_name_str = string(coef_name_sym)
        
        # # Priority 1: Check for a specific intercept prior from an `intercept()` call.
        if coef_name_str == "(Intercept)" && !isnothing(intercept_prior_val)
            prior = intercept_prior_val
            priors_vec[i] = prior isa Tuple ? create_pc_prior(:intercept, prior) : prior
            continue
        end
        
        # # Priority 2: Check for custom priors on other variables from `fixed()` calls.
        found_custom = false
        for (var_sym, prior) in custom_priors
            var_str = string(var_sym)
            if coef_name_str == var_str || startswith(coef_name_str, var_str * ":")
                priors_vec[i] = prior isa Tuple ? create_pc_prior(var_sym, prior) : prior
                found_custom = true
                break
            end
        end
        
        # # Priority 3: Use the default prior if no custom one is found.
        if !found_custom
            priors_vec[i] = default_prior
        end
    end
    
    M[:Xfixed_priors_vec] = priors_vec
end


function process_fixed_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.2.1 (2026-07-08)
    # Purpose: Processes the `fixed()` module. This module explicitly marks terms to be
    #          treated as fixed effects in the regression model. It now handles `contrast`
    #          and `prior` specifications.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies opt_dict in place.
    if !haskey(opt_dict, :fixed_effects)
        opt_dict[:fixed_effects] = String[]
    end
    if !haskey(opt_dict, :contrasts)
        opt_dict[:contrasts] = Dict{Symbol, Any}()
    end
    if !haskey(opt_dict, :fixed_effects_priors)
        opt_dict[:fixed_effects_priors] = Dict{Symbol, Any}()
    end
    if !haskey(opt_dict, :vars_to_categorize)
        opt_dict[:vars_to_categorize] = Set{Symbol}()
    end
    params = mod_data[:params]
    vars = mod_data[:variables]
    append!(opt_dict[:fixed_effects], string.(vars))
    if haskey(params, :contrast)
        if !isempty(vars)
            contrast_sym = params[:contrast]
            if haskey(STATSMODELS_CONTRASTS, contrast_sym)
                opt_dict[:contrasts][Symbol(vars[1])] = STATSMODELS_CONTRASTS[contrast_sym]
            else
                @warn "Unknown contrast coding ':$contrast_sym'. Using default (DummyCoding)."
                opt_dict[:contrasts][Symbol(vars[1])] = StatsModels.DummyCoding()
            end
        else
            @warn "A 'contrast' was specified in a fixed() module with no variable. Ignoring."
        end
    end
    if haskey(params, :prior)
        for var in vars
            opt_dict[:fixed_effects_priors][Symbol(var)] = params[:prior]
        end
    end
    if get(params, :model, nothing) == :categorical || haskey(params, :contrast)
        for var in vars
            push!(opt_dict[:vars_to_categorize], Symbol(var))
        end
    end
end

 
function _initialize_config(data::DataFrame, kwargs)
    M = Dict{Symbol, Any}()
    M[:data] = data
    M[:y_N] = nrow(data)
    for (k, v) in kwargs; M[k] = v; end
    M[:calling_module] = get(kwargs, :calling_module, Main)
    M[:manifolds] = []
    M[:basis_matrices] = Dict{Symbol, Any}()
    return M
end


function _process_lhs!(M::Dict, outcome_specs::Vector{<:Dict})
    outcomes = [Symbol(spec[:var]) for spec in outcome_specs]
    likelihood_specs = [spec[:params] for spec in outcome_specs]
    
    M[:outcomes] = outcomes

    if length(outcomes) > 1
        M[:model_arch] = "multivariate"
        M[:outcomes_N] = length(outcomes)
        M[:y_obs] = Matrix(M[:data][!, M[:outcomes]])
        M[:likelihood_specs] = likelihood_specs
    else
        M[:model_arch] = get(M, :model_arch, "univariate")
        M[:outcomes_N] = 1
        M[:y_obs] = M[:data][!, M[:outcomes][1]]
        M[:likelihood_specs] = likelihood_specs
    end

    representative_params = !isempty(likelihood_specs) ? likelihood_specs[1] : Dict()

    _resolve_obs_param!(M, representative_params, M[:data], [:offsets, :log_offsets], :log_offset)
    _resolve_obs_param!(M, representative_params, M[:data], [:weights], :weights)
    _resolve_obs_param!(M, representative_params, M[:data], [:trials], :trials)
end



function decompose_bstm_formula(formula_str::String) # v2.6.1
    parts = Base.split(formula_str, "~")
    lhs = strip(parts[1])
    rhs = strip(parts[2])

    # # 1. Parse the Left-Hand Side (LHS) for Outcomes and Likelihood Specs
    outcome_specs = Dict{Symbol, Any}[]
    lhs_terms = split_terms_at_depth(lhs, "+")
    for term in lhs_terms
        term = strip(term)
        m = match(r"likelihood\((.*)\)", term)
        if !isnothing(m)
            inner_content = m.captures[1]
            args = split_terms_at_depth(inner_content, ",")
            !isempty(args) || continue
            outcome_var = strip(args[1])
            params_str = join(args[2:end], ",")
            params = _parse_arguments_string(params_str)
            for ov in Base.split(outcome_var, '+')
                push!(outcome_specs, Dict(:var => strip(ov), :params => params))
            end
        else
            push!(outcome_specs, Dict(:var => term, :params => Dict()))
        end
    end

    # # 2. Pre-process and Normalize Right-Hand Side (RHS) Terms
    # This turns 'a - b' into 'a + -b', making it splittable by ' + '.
    # Using `\s*` handles cases with or without spaces around the minus operator.
    rhs_normalized = replace(rhs, r"\s*-\s*" => " + -")
    rhs_terms = split_terms_at_depth(rhs_normalized, " + ")

    has_intercept = true
    intercept_prior = nothing
    fixed_effects = String[]
    module_terms = String[]
    intercept_module_found = false

    for term in rhs_terms
        term_stripped = strip(term)
        
        # New authoritative logic for intercept control via intercept() module
        if startswith(term_stripped, "intercept(")
            if intercept_module_found
                @warn "Multiple intercept() modules found. Only the first will be evaluated for intercept control."
            else
                intercept_module_found = true
                intercept_data = _parse_single_manifold_term(term_stripped)
                pos_args = get(intercept_data.args, :positional_args, [])
                
                # `intercept(false)` disables the intercept.
                if !isempty(pos_args) && pos_args[1] == false
                    has_intercept = false
                else
                    has_intercept = true
                end

                # Extract prior if specified, e.g., intercept(prior=Normal(0,10))
                if haskey(intercept_data.args, :prior)
                    intercept_prior = intercept_data.args[:prior]
                end
            end
            # The intercept module's job is done; it is not passed to the main module loop.
            continue
        
        # Legacy numeric intercept control
        elseif term_stripped == "0" || term_stripped == "-1"
            has_intercept = false
        elseif term_stripped == "1"
            # This term is just for intercept control, do nothing.
        else
            if !occursin('(', term_stripped)
                push!(fixed_effects, term_stripped)
            else
                push!(module_terms, term_stripped)
            end
        end
    end

    top_level_nodes = [_parse_rhs_expression(term) for term in module_terms]

    modules = Dict{String, Any}()
    _categorize_rhs_nodes!(top_level_nodes, modules)

    return (outcomes=outcome_specs, modules=modules, fixed_effects=fixed_effects, has_intercept=has_intercept, intercept_prior=intercept_prior)
end


function _finalize_config!(M::Dict)
    defaults = Dict(
        :s_N => 0, :t_N => 0, :u_N => 0,
        :s_idx => ones(Int, M[:y_N]),
        :t_idx => ones(Int, M[:y_N]),
        :u_idx => ones(Int, M[:y_N]),
        :log_offset => zeros(M[:y_N]),
        :weights => ones(M[:y_N]),
        :trials => ones(Int, M[:y_N]),
        :hyperpriors => Dict(),
        :prior_scheme => :pcpriors
    )

    for (key, val) in defaults
        if !haskey(M, key)
            M[key] = val
        end
    end
end
 

 
function process_spatial_module!(opt_dict, mod_data, registries, hyperpriors)
    data = opt_dict[:data]
    params = mod_data[:params]
    variables = mod_data[:variables]
    
    # Check if W is provided inside the spatial() module call.
    # This has higher precedence than a globally passed W.
    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr
            # Evaluate the expression in the calling module's context
            calling_mod = get(opt_dict, :calling_module, Main)
            try
                opt_dict[:W] = Core.eval(calling_mod, w_val)
            catch e
                error("Could not evaluate `W` argument `$(w_val)` in spatial module. Error: $e")
            end
        else # It's already a value (e.g., if passed via bstm_config directly)
            opt_dict[:W] = w_val
        end
    end
    
    if !haskey(opt_dict, :W)
        @warn "Adjacency matrix 'W' not provided for spatial module. Attempting to infer from coordinates."
        if hasproperty(data, :s_x) && hasproperty(data, :s_y)
            au = assign_spatial_units(data.s_x, data.s_y; target_units=get(params, :target_units, 50))
            opt_dict[:W] = au.W
            opt_dict[:s_idx] = au.s_idx
            opt_dict[:s_N] = size(au.W, 1)
            opt_dict[:centroids] = au.centroids
        else
            error("Cannot infer spatial structure without 'W' or coordinate columns 's_x', 's_y'.")
        end
    else
        opt_dict[:s_N] = size(opt_dict[:W], 1)
        if !isempty(variables)
            s_var_sym = Symbol(variables[1])
            if hasproperty(data, s_var_sym)
                opt_dict[:s_idx] = data[!, s_var_sym]
            else
                @warn "Spatial index variable ':$s_var_sym' not found. Ensure data is aligned with W."
            end
        end
    end
end
 

function process_interaction_module!(opt_dict, mod_data, registries, hyperpriors)
    #
    # Processes algebraic interaction modules (`⊗`, `⊕`, `∘`, `|>`).
    #
    op = get(mod_data, :operator, get(mod_data[:params], :operator, nothing))
    isnothing(op) && return

    components = get(mod_data, :components, get(mod_data[:params], :components, []))
    if isempty(components)
        @warn "Interaction module found with no components. Skipping."
        return
    end

    if op == :⊕ || op == Symbol("oplus")
        opt_dict[:direct_sum_components] = components
    elseif op == :∘ || op == Symbol("compose")
        opt_dict[:composition_components] = components
    elseif op == :|> || op == Symbol("pipe")
        opt_dict[:pipe_components] = components
    end
end

function process_spacetime_module!(opt_dict, mod_data, registries, hyperpriors)
    #
    # Processes the `spacetime()` module, which is a shorthand for a Knorr-Held interaction.
    #
    models = get(mod_data[:params], :model, (:iid, :iid))
    spatial_model = string(models[1])
    temporal_model = string(models[2])

    has_structured_space = spatial_model != "iid"
    has_structured_time = temporal_model != "iid"

    if has_structured_space && has_structured_time; opt_dict[:model_st] = "IV";
    elseif !has_structured_space && has_structured_time; opt_dict[:model_st] = "II";
    elseif has_structured_space && !has_structured_time; opt_dict[:model_st] = "III";
    else opt_dict[:model_st] = "I"; end
end


function process_intercept_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.2.0 (2026-07-08)
    # Purpose: Processes the `intercept()` module to signal the inclusion of a global intercept
    #          and handle any specified prior.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies opt_dict in place.
    opt_dict[:add_intercept] = true

    # Check for a locally specified prior within the intercept() call, e.g., intercept(prior=Normal(0,10))
    if haskey(mod_data[:params], :prior)
        opt_dict[:intercept_prior] = mod_data[:params][:prior]
    end
end

function process_temporal_module!(opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict)
    #
    # bstm Module Processor: Temporal
    #
    # This function processes the `temporal()` module from the formula. It is responsible for
    # identifying the primary time index variable from the data, calling the `assign_time_units`
    # utility to create discrete time steps, and registering the resulting indices and total
    # number of time units into the main model configuration dictionary.
    #

    # #
    # 1. Extract data and module specifications.
    data = opt_dict[:data]
    params = mod_data[:params]
    variables = mod_data[:variables]

    # #
    # 2. Identify and process the time index variable.
    # The first positional argument in `temporal()` is assumed to be the time variable.
    if !isempty(variables)
        t_var_sym = Symbol(variables[1])

        if hasproperty(data, t_var_sym)
            # #
            # 3. Configure and call the time unit assignment utility.
            # Parameters for `assign_time_units` can be passed via the `temporal()` module.
            time_opts = Dict(
                :time_method => get(params, :time_method, "regular"),
                :t_N => get(params, :t_N, nothing),
                :u_N => get(params, :u_N, nothing)
            )
            # Remove any parameters that were not explicitly provided.
            filter!(p -> !isnothing(p.second), time_opts)

            # Call the utility to get time unit metadata.
            tu_meta = assign_time_units(data[!, t_var_sym]; time_opts...)

            # #
            # 4. Register the results into the main configuration dictionary.
            opt_dict[:t_idx] = tu_meta.t_idx
            opt_dict[:t_N] = tu_meta.tn
            opt_dict[:t_idx_var] = t_var_sym
        else
            # Issue a warning if the specified time variable is not found in the data.
            @warn "Temporal index variable ':$t_var_sym' not found in data. Temporal effects may not be correctly applied."
        end
    else
        @warn "The `temporal()` module was called without specifying a time variable. Temporal effects will be ignored."
    end
end


 

function process_seasonal_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: Processes a `seasonal()` module, which is created by the pre-processor 
    #          in `bstm_config` from a dual `temporal()` call. 
    data = opt_dict[:data]
    variables = mod_data[:variables]
    
    # Ensure seasonal indices are initialized to avoid errors if the module is present but data is not.
    if !haskey(opt_dict, :u_idx); opt_dict[:u_idx] = nothing; end
    if !haskey(opt_dict, :u_N); opt_dict[:u_N] = 0; end

    if !isempty(variables)
        u_var_sym = Symbol(variables[1])
        if hasproperty(data, u_var_sym)
            opt_dict[:u_idx] = data[!, u_var_sym]
            opt_dict[:u_N] = length(unique(opt_dict[:u_idx]))
        else
            @warn "Seasonal index variable ':$u_var_sym' not found in data. A default seasonal index may be used if available from `assign_time_units`."
        end
    end
end

function process_smooth_module!(opt_dict, mod_data, basis_matrices_registry, manifolds_registry)
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: Processes `smooth()` modules for non-linear covariate effects. 
    # This version correctly handles tensor products, GMRF-on-bins, continuous kernels, and basis models. 
    # kernel model to support dynamic lengthscale sampling for PC priors.

    data = opt_dict[:data]
    params = mod_data[:params]
    model_param = get(params, :model, "pspline")

    # RFF moved to continuous_kernel_models to support dynamic lengthscale for PC priors.
    basis_models = ["pspline", "bspline", "tps", "fft", "moran", "spherical", "barycentric", "decay", "wavelet", "linear", "invdist", "kriging"]
    continuous_kernel_models = ["gp", "fitc", "svgp", "nystrom", "warp", "spde", "exponentialdecay", "rff"]
    gmrfs_on_bins_models = ["rw1", "rw2", "ar1", "icar", "besag", "cyclic"]

    # --- 1. Tensor Product Smooths ---
    # Handles `smooth(x, y, model=(x=pspline, y=rw2))`
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
            
            mod_data[:params][:model] = "tensor_product_smooth"
            mod_data[:params][:Q_tensor] = Q_final
            return # Exit after handling tensor product
        end
    end

    model_str = string(model_param)
    
    basis_models = ["pspline", "bspline", "tps", "rff", "fft", "moran", "spherical", "barycentric", "decay", "wavelet", "linear", "invdist", "kriging"]
    continuous_kernel_models = ["gp", "fitc", "svgp", "nystrom", "warp", "spde", "exponentialdecay"]
    gmrfs_on_bins_models = ["rw1", "rw2", "ar1", "icar", "besag", "cyclic"]

    # --- 2. Basis Function Models ---
    # Handles `smooth(x, model='pspline')`
    if model_str in basis_models
        if !isempty(mod_data[:variables])
            nb = get(mod_data[:params], :nbins, 20)
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
    # --- 3. Continuous Kernel Models ---
    # Handles `smooth(lon, lat, model='gp')`
    elseif model_str in continuous_kernel_models
        if all(v -> hasproperty(data, Symbol(v)), mod_data[:variables])            
            coords = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
            mod_data[:params][:coords] = coords
            
            # Pre-compute inducing points for sparse GP models to avoid doing it inside the Turing model.
            if model_str in ["fitc", "svgp", "nystrom", "gp"] # Treat 'gp' as sparse for scalability
                n_inducing_default = min(100, size(coords, 1)) # Sensible default
                n_inducing = get(mod_data[:params], :n_inducing, n_inducing_default)
                
                # Using a fixed seed for reproducibility of inducing points across runs.
                # The generate_inducing_points function should be available in the environment.
                Z_inducing = generate_inducing_points(coords, n_inducing; seed=42, method="kmeans")
                mod_data[:params][:Z_inducing] = Z_inducing
            end
        else
            @warn "Continuous kernel smooth specified, but coordinate variables not found in data. Manifold may be misspecified."
        end

    # --- 4. GMRF-on-Bins Models ---
    # Handles `smooth(x, model='rw2')`
    elseif model_str in gmrfs_on_bins_models
        vars = mod_data[:variables]
        if length(vars) != 1
            @warn "GMRF smooth on $(join(vars, ",")) requires exactly 1 variable. Skipping."
            return
        end
        
        var_sym = Symbol(vars[1])
        nbins = get(mod_data[:params], :nbins, 20)
        
        _, indices = apply_discretization_logic(data[!, var_sym], nbins)
        mod_data[:params][:indices] = indices
        mod_data[:params][:n_cat] = length(unique(indices))
        
        # Re-brand this module as a 'mixed' effect so the main loop handles it correctly.
        mod_data[:type] = :mixed
    end
end

 
function process_dynamics_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: Processes the `dynamics()` module. 
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies opt_dict in place.
end
 
function process_eigen_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: Processes the `nested()` supervisor module for multi-fidelity models. This 
    #          function configures a complete sub-model based on a nested formula, which is
    #          then used for joint likelihood evaluation in the main model.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies opt_dict in place.
    # Rationale for v1.2.0:
    #     - Added logic to parse the outcome variable and model family for the nested model,
    #       which was previously missing, causing downstream errors.
    #     - Ensured that all necessary components (y_obs, y_N, model_family, spatial indices,
    #       and fixed effects) are correctly configured for the sub-model.

    if !haskey(opt_dict, :nested_manifolds); opt_dict[:nested_manifolds] = Dict{Symbol, Any}(); end
    
    var = Symbol(mod_data[:variables][1])
    params = mod_data[:params]
    sub_formula = get(params, :formula, "")
    data_source_sym = get(params, :data_source, :data)
    
    # 1. Validate and retrieve the data source for the nested model.
    if !haskey(opt_dict, data_source_sym)
        @warn "Data source ':$data_source_sym' for nested module on '$var' not found. Skipping."
        return
    end
    sub_data = opt_dict[data_source_sym]
    
    # Initialize a configuration dictionary for the sub-model.
    sub_config = _initialize_config(sub_data, Dict(:calling_module => get(opt_dict, :calling_module, Main)))
    
    # Decompose the sub-formula.
    sub_metadata = decompose_bstm_formula(sub_formula)
    
    # 2. Process the LHS of the sub-formula to get outcome and likelihood specs.
    _process_lhs!(sub_config, sub_metadata.outcomes)

    # 3. Process RHS modules for the sub-model.
    sub_config[:manifolds] = [] # Initialize manifolds for the sub-model.
    for (key, mod_data_nt) in sub_metadata.modules
        mod_type = mod_data_nt.module_type
        processor! = get(MODULE_PROCESSORS, mod_type, nothing)
        isnothing(processor!) && continue

        mod_data_dict = Dict(:type => mod_data_nt.module_type, :variables => get(mod_data_nt.args, :positional_args, []), :params => mod_data_nt.args)
        
        # Call the processor for the specific module type on the sub_config.
        processor!(sub_config, mod_data_dict, sub_config, hyperpriors)

        # Resolve the manifold object and build its template.
        manifold_obj = resolve_technical_primitive(mod_data_dict, sub_config, hyperpriors, get(opt_dict, :prior_scheme, :pcpriors))
        manifold_spec_built = build_model(manifold_obj, sub_config)

        spec = (key=Symbol(key), domain=mod_data_dict[:type], var=join(mod_data_dict[:variables], "_"), manifold_obj=manifold_obj, params=mod_data_dict[:params], Q_template=manifold_spec_built.Q_template, scaling_factor=manifold_spec_built.scaling_factor)
        push!(sub_config[:manifolds], spec)
    end

    # 4. Process fixed effects for the sub-model.
    _process_fixed_effects!(sub_config, sub_metadata.fixed_effects, sub_metadata.has_intercept)
    _process_fixed_effects_priors!(sub_config)

    # Finalize the sub-configuration.
    _finalize_config!(sub_config)
    
    # 5. Register the fully configured nested manifold.
    opt_dict[:nested_manifolds][var] = sub_config
end
 


function resolve_technical_primitive(module_metadata::Dict{Symbol, Any}, M, priors_dict, scheme::Symbol)
    # v1.2.0 (2026-06-29 16:13:05)
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
        # This block now handles both algebraic compositions (⊗, ⊕, etc.) and
        # varying interaction manifolds by checking for the presence of the :operator key.
        if haskey(m_params, :operator) && haskey(m_params, :components)
            # #
            # Algebraic composition case (e.g., spatial() ⊗ temporal())
            op = m_params[:operator]
            components_data = m_params[:components]
            
            # The components from the parser are NamedTuples. They must be converted to
            # Dicts to match the expected input type for the recursive call.
            components_metadata = map(components_data) do c_node
                if haskey(c_node, :module_type)
                    # This is a simple manifold module.
                    Dict(:type => c_node.module_type, :params => c_node.args, :variables => get(c_node.args, :positional_args, []))
                else # It's a nested operator node.
                    # Re-wrap it into the structure expected by this function for recursion.
                    Dict(:type => :interaction, :params => Dict(:operator => c_node.op, :components => c_node.children), :variables => [])
                end
            end

            resolved_components = [resolve_technical_primitive(comp_meta, M, priors_dict, scheme) for comp_meta in components_metadata]
            return ComposedManifold(resolved_components, op)
        else
            # Varying interaction manifold case (e.g., interaction(x, y, model=...))
            interaction_vars = [Symbol(v) for v in module_metadata[:variables]]
            inner_model_data = get(m_params, :model, Dict{Symbol, Any}())
            inner_manifold_obj = resolve_technical_primitive(inner_model_data, M, priors_dict, scheme)
            return VaryingInteractionManifold(interaction_vars, inner_manifold_obj)
        end
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
        elseif m_type == :intercept || m_type == :fixed
            "none"
        else
            string(m_type)
        end
                
        model_val = get(m_params, :model, default_model)
        model_name = if model_val isa Symbol
            # Use String() to convert a symbol to a string without the leading colon.
            String(model_val)
        else
            string(model_val)
        end

        model_sym = Symbol(model_name)

        resolved_priors = resolve_hyperpriors(model_name, priors_dict, m_params, scheme)

        if haskey(MANIFOLD_CONSTRUCTORS, model_sym)
            constructor_func = MANIFOLD_CONSTRUCTORS[model_sym]
            return constructor_func(resolved_priors, m_params)
        else
            supported_keys = join(sort(collect(string.(keys(MANIFOLD_CONSTRUCTORS)))), ", ")
            error("Unknown manifold model '$model_name' for module '$m_type'. Supported models are: $supported_keys")
        end
    end
end


"""
    @bstm(formula, data, kwargs...)

A macro to simplify the `bstm` formula syntax. It captures the formula as an
unevaluated expression, converts it to a string, and passes it to the `bstm`
function along with the calling module's context for correct evaluation of
parameters.
"""
macro bstm(formula, data, kwargs...)
    formula_str = string(formula)
    data_esc = esc(data)
    kwargs_esc = [esc(kw) for kw in kwargs]
    # Pass the calling module's context to the bstm function
    return :(bstm($formula_str, $data_esc, @__MODULE__; $(kwargs_esc...)))
end

"""
    bstm(formula::String, data::DataFrame; kwargs...)

The primary user-facing function for creating a `bstm` model. This version is
for direct string-based formula calls and defaults to the `Main` module for
evaluation context.
"""
function bstm(formula::String, data::DataFrame; kwargs...)
    return bstm(formula, data, Main; kwargs...)
end

"""
    bstm(formula::String, data::DataFrame, calling_module::Module; kwargs...)

Internal entry point that receives the calling module's context. It orchestrates
the configuration and construction of the Turing model.
"""
function bstm(formula::String, data::DataFrame, calling_module::Module; kwargs...) 
    options = bstm_config(formula, data; calling_module=calling_module, kwargs...)
    return bstm(options)
end

"""
    bstm(config::NamedTuple)

The core model dispatcher. It takes a fully resolved configuration `NamedTuple`
and calls the appropriate model supervisor (`bstm_univariate`, `bstm_multivariate`, etc.).
"""
function bstm(config::NamedTuple)
    arch = get(config, :model_arch, "univariate")
    
    if arch == "multivariate"
        return bstm_multivariate(config)
    elseif arch == "multifidelity"
        return bstm_multifidelity(config)
    else
        return bstm_univariate(config) 
    end
end
 

const MODULE_PROCESSORS = Dict{Symbol, Function}(
    :spatial => process_spatial_module!,
    :temporal => process_temporal_module!,
    :seasonal => process_seasonal_module!,
    :smooth => process_smooth_module!,
    :dynamics => process_dynamics_module!,
    :eigen => process_eigen_module!,
    :mixed => process_mixed_module!,
    :nested => process_nested_module!,
    :interaction => process_interaction_module!,
    :spacetime => process_spacetime_module!,
    :fixed => process_fixed_module!
);
