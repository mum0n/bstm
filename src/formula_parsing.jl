# CORE FORMULA PARSING UTILITIES


const BSTM_MODULE_KEYWORDS = Set([ # v2.1.0: Expanded keyword registry
    "intercept", "spatial", "temporal", "seasonal", "smooth", "fixed",
    "nested", "eigen", "mixed", "dynamics", "spacetime", "interaction"
]);

function _parse_value(val_str::String)
    val_str = strip(val_str)
    if startswith(val_str, "(") && endswith(val_str, ")") # Tuple
        inner_val_str = val_str[2:end-1]
        tuple_parts = Base.split(inner_val_str, r",\s*")
        parsed_tuple = []
        for tp in tuple_parts
            if !isempty(tp)
                push!(parsed_tuple, _parse_value(tp))
            end
        end
        return Tuple(parsed_tuple)
    elseif startswith(val_str, "\"") && endswith(val_str, "\"") # String
        return val_str[2:end-1]
    else
        try_num = tryparse(Float64, val_str)
        if try_num !== nothing
            return try_num
        else
            # Return as an expression to be evaluated later
            return Meta.parse(val_str)
        end
    end
end

function _parse_value(val_str::SubString{String})
    _parse_value(String(val_str))
end

function _add_parsed_arg!(args_dict::Dict{Symbol, Any}, positional_args::Vector{Any}, arg_val::String)
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
    args_dict = Dict{Symbol, Any}()
    positional_args = []
    current_arg = ""
    depth = 0 # Unified counter for parentheses and brackets

    for char in args_str
        if (char == ',' || char == ';') && depth == 0
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
    term_str = strip(term_str)
    m = match(r"^\s*(\w+)\s*\((.*)\)\s*$", term_str)
    if m === nothing
        return (module_type = :fixed, args = Dict(:name => term_str))
    end

    module_name = Symbol(m.captures[1])
    args_str = String(m.captures[2])
    args_dict = _parse_arguments_string(args_str)

    return (module_type = module_name, args = args_dict)
end

function _parse_single_manifold_term(term_str::SubString{String})
    _parse_single_manifold_term(String(term_str))
end

function _parse_rhs_expression(term_str::SubString{String})
    _parse_rhs_expression(String(term_str))
end

function parse_module_params(params_str::AbstractString)
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
 
    
 

function decompose_bstm_formula(formula_str::String)
    # BSTM Formula Decomposer v2.1.0
    # Timestamp: 2026-07-08
    # Synopsis: This function is the main entry point for formula parsing. It separates the
    #           formula into its Left-Hand Side (LHS) for likelihood specification and
    #           Right-Hand Side (RHS) for the latent process definition. It now uses a
    #           recursive descent parser for the RHS to correctly handle complex algebraic
    #           expressions between manifolds.

    parts = Base.split(formula_str, "~")
    lhs = strip(parts[1])
    rhs = strip(parts[2])

    # --- 1. Parse the Left-Hand Side (LHS) for Outcomes and Likelihood Specs ---
    outcome_specs = []
    lhs_terms = split_terms_at_depth(lhs, "+")
    for term in lhs_terms
        term = strip(term)
        m = match(r"likelihood\((.*)\)", term)
        if !isnothing(m)
            # Handles `likelihood(y1+y2, family=:poisson, ...)`
            inner_content = m.captures[1]
            args = split_terms_at_depth(inner_content, ",")
            !isempty(args) || continue
            outcome_var = strip(args[1])
            params_str = join(args[2:end], ",")
            params = _parse_arguments_string(params_str)
            # Handle multiple outcomes within one likelihood call
            for ov in Base.split(outcome_var, '+')
                push!(outcome_specs, Dict(:var => strip(ov), :params => params))
            end
        else
            # Handles `y1 + y2 ~ ...`
            push!(outcome_specs, Dict(:var => term, :params => Dict()))
        end
    end

    # --- 2. Parse the Right-Hand Side (RHS) into an Abstract Syntax Tree (AST) ---
    rhs_ast = _parse_rhs_expression(rhs)

    # --- 3. Traverse the AST to separate modules from fixed effects ---
    modules = Dict{String, Any}()
    fixed_effects = String[]
    has_intercept = false

    # The top-level operator in a formula is always '+', so we can iterate through its children.
    top_level_nodes = (rhs_ast.type == :operator && rhs_ast.op == :+) ? rhs_ast.children : [rhs_ast]

    has_intercept = _categorize_rhs_nodes!(top_level_nodes, modules, fixed_effects)

    return (outcomes=outcome_specs, modules=modules, fixed_effects=fixed_effects, has_intercept=has_intercept)
end

function _categorize_rhs_nodes!(nodes, modules, fixed_effects)
    # BSTM Internal Parser v2.1.0
    # Traverses the list of top-level nodes from the RHS AST and categorizes them
    # as modules, fixed effects, or an intercept.
    local_has_intercept = false
    for node in nodes
        if node.type == :operator
            # Any expression involving operators (⊗, ⊕, |>, etc.) is treated as a single complex module.
            key = "composed_$(length(modules)+1)"
            modules[key] = (module_type = :interaction, args = Dict(:operator => node.op, :components => node.children))

        elseif node.module_type in BSTM_MODULE_KEYWORDS
            key_parts = [string(node.module_type)]
            if haskey(node.args, :positional_args) && !isempty(node.args[:positional_args])
                push!(key_parts, string(node.args[:positional_args][1]))
            end
            base_key = join(key_parts, "_")
            module_key = base_key
            counter = 1
            while haskey(modules, module_key)
                counter += 1
                module_key = base_key * "_$counter"
            end
            modules[module_key] = node

        elseif node.module_type == :fixed && haskey(node.args, :name)
            # This handles literal terms parsed as fixed effects
            if node.args[:name] == "1"
                local_has_intercept = true
            else
                push!(fixed_effects, node.args[:name])
            end
        end
    end
    return local_has_intercept
end

function split_terms_at_depth(input::AbstractString, sep::AbstractString)
    terms = String[]
    depth = 0
    current = ""
    for char in input
        if char == '(' || char == '['; depth += 1;
        elseif char == ')' || char == ']'; depth -= 1;
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


function _parse_rhs_expression(term_str::String)
    # BSTM Internal Parser v2.1.0
    # Timestamp: 2026-07-08
    # Synopsis: A recursive descent parser to convert the RHS of a formula string into an
    #           Abstract Syntax Tree (AST). It correctly handles operator precedence and
    #           associativity, which is critical for complex manifold compositions.
    #
    # Operator Precedence (from lowest to highest):
    # 1. `+` (Addition of independent effects)
    # 2. `⊕` (Direct Sum - shared hyperparameters)
    # 3. `|>` (Pipe - state-space evolution or SVC)
    # 4. `⊗` (Kronecker Product - interactions)
    # 5. `∘` (Composition)

    term_str = strip(term_str)

    # Define operators with their symbol and associativity (:left or :right)
    operators = [
        (" + ", :+, :left),
        (" ⊕ ", :direct_sum, :left),
        (" |> ", :pipe, :left),
        (" ⊗ ", :kronecker_product, :left),
        (" ∘ ", :composition, :left)
    ]

    for (op_str, op_sym, associativity) in operators
        parts = split_terms_at_depth(term_str, op_str)
        if length(parts) > 1
            if associativity == :left
                # Build the tree from left to right for left-associative operators
                node = _parse_rhs_expression(parts[1])
                for i in 2:length(parts)
                    right_node = _parse_rhs_expression(parts[i])
                    node = (type=:operator, op=op_sym, children=[node, right_node])
                end
                return node
            else # :right
                # Build the tree from right to left for right-associative operators
                node = _parse_rhs_expression(parts[end])
                for i in (length(parts)-1):-1:1
                    left_node = _parse_rhs_expression(parts[i])
                    node = (type=:operator, op=op_sym, children=[left_node, node])
                end
                return node
            end
        end
    end

    # Base case: No operators found, so it's a single manifold term or a fixed effect.
    return _parse_single_manifold_term(term_str)
end

function _categorize_rhs_nodes!(nodes, modules, fixed_effects)
    # BSTM Internal Parser v2.1.0
    # Traverses the list of top-level nodes from the RHS AST and categorizes them
    # as modules, fixed effects, or an intercept.
    local_has_intercept = false
    for node in nodes
        if node.type == :operator
            # Any expression involving operators (⊗, ⊕, |>, etc.) is treated as a single complex module.
            key = "composed_$(length(modules)+1)"
            modules[key] = (module_type = :interaction, args = Dict(:operator => node.op, :components => node.children))

        elseif node.module_type in BSTM_MODULE_KEYWORDS
            key_parts = [string(node.module_type)]
            if haskey(node.args, :positional_args) && !isempty(node.args[:positional_args])
                push!(key_parts, string(node.args[:positional_args][1]))
            end
            base_key = join(key_parts, "_")
            module_key = base_key
            counter = 1
            while haskey(modules, module_key)
                counter += 1
                module_key = base_key * "_$counter"
            end
            modules[module_key] = node

        elseif node.module_type == :fixed && haskey(node.args, :name)
            # This handles literal terms parsed as fixed effects
            if node.args[:name] == "1"
                local_has_intercept = true
            else
                push!(fixed_effects, node.args[:name])
            end
        end
    end
    return local_has_intercept
end

function bstm_config(formula::String, data::DataFrame; kwargs...)
    # #
    # bstm_config v2.1.0 (2026-07-08)
    #
    # This function serves as the primary configuration engine for the bstm framework. It parses
    # a formula string and a dataset to construct a detailed configuration object (`M`) that is
    # passed to the Turing models for inference.
    #
    # Key Change in v2.1.0:
    # Rationale: This function now correctly orchestrates the parsing and processing pipeline.
    #            It iterates over the `modules` dictionary produced by the AST parser
    #            in `decompose_bstm_formula`, ensuring that complex algebraic compositions
    #            of manifolds are handled correctly and only once.

    # # 1. Initialization
    M = _initialize_config(data, kwargs)

    # # 2. Formula Decomposition
    # The formula is decomposed into LHS (outcomes) and RHS (modules, fixed effects).
    decomposed_formula = decompose_bstm_formula(formula)
    M[:formula] = formula

    # # 3. Likelihood and Architecture Processing (LHS)
    _process_lhs!(M, decomposed_formula.outcomes)

    # # 4. Manifold and Module Processing (RHS)
    # This section now correctly processes the pre-parsed modules from the AST.
    M[:hyperpriors] = get(M, :hyperpriors, Dict())
    M[:prior_scheme] = get(M, :prior_scheme, :pcpriors)

    for (key, mod_data_nt) in decomposed_formula.modules
        mod_type = mod_data_nt.module_type

        if haskey(MODULE_PROCESSORS, mod_type)
            processor! = MODULE_PROCESSORS[mod_type]
            
            # Convert the parser's NamedTuple into a mutable Dict for modification by processors.
            mod_data_dict = Dict(
                :type => mod_data_nt.module_type,
                :variables => get(mod_data_nt.args, :positional_args, []),
                :params => mod_data_nt.args
            )

            # The processor function modifies the main config `M` and the module's data dict.
            processor!(M, mod_data_dict, M, M[:hyperpriors])
            
            # Resolve the module into a concrete Manifold object.
            manifold_obj = resolve_technical_primitive(mod_data_dict, M, M[:hyperpriors], M[:prior_scheme])
            
            # Build the technical specification (e.g., Q_template) for the manifold.
            manifold_spec_built = build_model(manifold_obj, M)

            # Add the complete manifold specification to the registry, including the built template.
            spec = (
                key = Symbol(key),
                domain = mod_data_dict[:type], # Use the potentially updated domain from the processor
                var = join(mod_data_dict[:variables], "_"),
                manifold_obj = manifold_obj,
                params = mod_data_dict[:params],
                Q_template = manifold_spec_built.Q_template,
                scaling_factor = manifold_spec_built.scaling_factor
            )
            push!(M[:manifolds], spec)
        end
    end

    # # 5. Fixed Effects Processing
    # Rationale: Consolidating fixed effects from literal terms (e.g., `~ z`) and `fixed()` modules.
    _process_fixed_effects_priors!(M)

    # # 6. Finalization
    _finalize_config!(M)

    return NamedTuple(M)
end

function _process_fixed_effects_priors!(M::Dict)
    if !haskey(M, :Xfixed_N) || M[:Xfixed_N] == 0
        M[:Xfixed_prior] = MvNormal(zeros(0), Diagonal(zeros(0)))
        return
    end

    default_prior_std = 5.0
    priors_map = get(M, :fixed_effects_priors, Dict{Symbol, Any}())
    coef_names = M[:Xfixed_names]
    
    variances = fill(default_prior_std^2, M[:Xfixed_N])

    for (i, coef_name_sym) in enumerate(coef_names)
        coef_name_str = string(coef_name_sym)
        
        # Clean up names like "(Intercept)"
        clean_coef_name = replace(coef_name_str, r"[\(\)]" => "")
        
        # Try to find a matching prior
        found_prior = false
        for (var_sym, prior_dist) in priors_map
            var_str = string(var_sym)
            
            # Match if the coefficient name is the variable name, or starts with "variable:"
            if clean_coef_name == var_str || startswith(clean_coef_name, var_str * ":")
                if prior_dist isa Normal
                    variances[i] = prior_dist.σ^2
                    found_prior = true
                    break # First match wins
                else
                    @warn "Prior for '$coef_name_sym' (from '$var_str') is not a Normal distribution. Using default."
                end
            end
        end
    end
    
    M[:Xfixed_prior] = MvNormal(zeros(M[:Xfixed_N]), Diagonal(variances))
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
    params = mod_data[:params]
    vars = mod_data[:variables]
    append!(opt_dict[:fixed_effects], string.(vars))
    if haskey(params, :contrast)
        if !isempty(vars)
            opt_dict[:contrasts][Symbol(vars[1])] = params[:contrast]
        else
            @warn "A 'contrast' was specified in a fixed() module with no variable. Ignoring."
        end
    end
    if haskey(params, :prior)
        for var in vars
            opt_dict[:fixed_effects_priors][Symbol(var)] = params[:prior]
        end
    end
end


function _initialize_config(data::DataFrame, kwargs)
    # BSTM Internal Utility v2.0.0
    # Initializes the main configuration dictionary `M`.
    M = Dict{Symbol, Any}()
    M[:data] = data
    M[:y_N] = nrow(data)
    for (k, v) in kwargs; M[k] = v; end
    M[:calling_module] = get(kwargs, :calling_module, Main)
    M[:manifolds] = []
    M[:basis_matrices] = Dict{Symbol, Any}()
    return M
end

function _process_lhs!(M::Dict, outcome_specs::Vector{Dict})
    # BSTM Internal Utility v2.1.0
    # Processes the LHS of the formula to set up outcomes and likelihood specifications.
    # It now accepts the `outcome_specs` structure directly from the formula decomposer.

    # Extract outcome names and parameter dictionaries from the specs.
    outcomes = [Symbol(spec[:var]) for spec in outcome_specs]
    likelihood_specs = [spec[:params] for spec in outcome_specs]
    
    M[:outcomes] = outcomes

    if length(outcomes) > 1
        M[:model_arch] = "multivariate"
        M[:outcomes_N] = length(outcomes)
        M[:y_obs] = Matrix(M[:data][!, M[:outcomes]])
        
        # The parser creates a spec for each outcome. If a single likelihood() block
        # was used for multiple outcomes, the parameter dicts will be the same object.
        # The current parser already handles this correctly.
        M[:likelihood_specs] = likelihood_specs

    else
        M[:model_arch] = get(M, :model_arch, "univariate")
        M[:outcomes_N] = 1
        M[:y_obs] = M[:data][!, M[:outcomes][1]]
        M[:likelihood_specs] = likelihood_specs # Should be a vector with one element
    end

    # For resolving obs params (offset, weights, trials), the framework assumes these are
    # shared across outcomes if specified in a single likelihood() block. We use the
    # parameters from the first outcome spec as the reference.
    representative_params = !isempty(likelihood_specs) ? likelihood_specs[1] : Dict()

    # Resolve observation-level parameters
    _resolve_obs_param!(M, representative_params, M[:data], [:offsets, :log_offsets], :log_offset)
    _resolve_obs_param!(M, representative_params, M[:data], [:weights], :weights)
    _resolve_obs_param!(M, representative_params, M[:data], [:trials], :trials)
end

function _process_fixed_effects!(M::Dict, fixed_effects::Vector{String}, has_intercept::Bool)
    # BSTM Internal Utility v2.0.0
    # Handles the creation of the fixed effects design matrix.
    
    M[:add_intercept] = has_intercept
    fixed_effects_formula = join(fixed_effects, " + ")
    
    if !isempty(strip(fixed_effects_formula)) || has_intercept
        # The intercept is handled separately in the model, so we add "0" to the formula
        # if we don't want StatsModels to add one automatically.
        rhs = has_intercept ? "1 + " * fixed_effects_formula : "0 + " * fixed_effects_formula
        
        Xfixed_named = create_fixed_design(rhs, M[:data]; contrasts=get(M, :contrasts, Dict()))
        M[:Xfixed] = Matrix(Xfixed_named)
        M[:Xfixed_N] = size(M[:Xfixed], 2)
        M[:Xfixed_names] = names(Xfixed_named, 2)
    else
        M[:Xfixed] = zeros(M[:y_N], 0)
        M[:Xfixed_N] = 0
        M[:Xfixed_names] = []
    end
end

function _finalize_config!(M::Dict)
    # BSTM Internal Utility v2.0.0
    # Performs final checks and clean-up on the configuration dictionary.
    
    # Ensure essential keys exist with default values
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


const MODULE_PROCESSORS = Dict{Symbol, Function}(
    :spatial => process_spatial_module!,
    :temporal => process_temporal_module!,
    :seasonal => process_seasonal_module!,
    :smooth => process_smooth_module!,
    :intercept => process_intercept_module!,
    :dynamics => process_dynamics_module!,
    :eigen => process_eigen_module!,
    :mixed => process_mixed_module!,
    :nested => process_nested_module!,
    :interaction => process_interaction_module!,
    :spacetime => process_spacetime_module!,
    :fixed => process_fixed_module!
);




function process_temporal_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: Processes the `temporal()` module to handle the primary time index. 
    # Note: This version is simplified, as seasonal components are now handled by 
    #       a pre-processing step in `bstm_config` that creates a `seasonal()` module.
    data = opt_dict[:data]
    
    params = mod_data[:params]
    variables = mod_data[:variables]

    if !isempty(variables)
        t_var_sym = Symbol(variables[1])
        if hasproperty(data, t_var_sym)
            time_opts = Dict(:time_method => get(params, :time_method, "regular"), :t_N => get(params, :t_N, nothing), :u_N => get(params, :u_N, nothing))
            filter!(p -> !isnothing(p.second), time_opts)
            tu_meta = assign_time_units(data[!, t_var_sym]; time_opts...)
            opt_dict[:t_idx] = tu_meta.t_idx
            opt_dict[:t_N] = tu_meta.tn
            opt_dict[:t_idx_var] = t_var_sym
        else
            @warn "Temporal index variable ':$t_var_sym' not found in data."
        end
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
            mod_data[:params][:coords] = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
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
    
    sub_config = Dict{Symbol, Any}()
    sub_config[:data] = sub_data
    sub_metadata = decompose_bstm_formula(sub_formula)
    
    # 2. Extract and configure the outcome variable for the nested model.
    if isempty(sub_metadata.outcomes)
        @warn "No outcome variable specified in nested formula for '$var'. Skipping."
        return
    end
    sub_outcome_sym = Symbol(sub_metadata.outcomes[1])
    if !hasproperty(sub_data, sub_outcome_sym)
        @warn "Outcome variable ':$sub_outcome_sym' for nested module on '$var' not found in data source ':$data_source_sym'. Skipping."
        return
    end
    sub_config[:y_obs] = sub_data[!, sub_outcome_sym]
    sub_config[:y_N] = length(sub_config[:y_obs])
    sub_config[:model_family] = get(params, :model_family, "gaussian")

    # 3. Process spatial components within the nested formula.
    spatial_mod_data_pair = filter(m -> m.second[:type] == :spatial, sub_metadata.modules)
    if !isempty(spatial_mod_data_pair)
        spatial_mod_data = first(spatial_mod_data_pair).second
        process_spatial_module!(sub_config, spatial_mod_data, registries, hyperpriors)
        sub_config[:model_space] = get(spatial_mod_data[:params], :model, "bym2")
        sub_config[:s_Q_template] = build_structure_template(Symbol(sub_config[:model_space]), sub_config[:s_N]; W=get(sub_config, :W, nothing))
    end
    
    # 4. Process fixed effects within the nested formula.
    fixed_effects_formula_part = join(sub_metadata.fixed_effects, " + ")
    if sub_metadata.has_intercept
        fixed_effects_formula_part = isempty(fixed_effects_formula_part) ? "1" : "1 + " * fixed_effects_formula_part
    end
    if !isempty(fixed_effects_formula_part)
        sub_config[:Xfixed] = create_fixed_design(fixed_effects_formula_part, sub_data)
    end
    
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
                        else string(m_type) end
        
        model_name = string(get(m_params, :model, default_model))
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
