


function Base.:|>(m1::Component, m2::Component)
    return Composed([m1, m2], :pipe)
end

composition(m1::Component, m2::Component) = Composed([m1, m2], :composition)
∘(m1::Component, m2::Component) = Composed([m1, m2], :composition)

otimes(m1::Component, m2::Component) = Composed([m1, m2], :kronecker_product)
⊗(m1::Component, m2::Component) = Composed([m1, m2], :kronecker_product)

# ==============================================================================
# SECTION 3: FORMULA PARSING ENGINE
# ==============================================================================

"""
    apply_transformation(func_name::Symbol, data_vector::AbstractVector)

Applies a specified data transformation to a vector.

# Arguments
- `func_name::Symbol`: The name of the transformation (e.g., `:zscore`).
- `data_vector::AbstractVector`: The input data column.

# Returns
- A new vector with the transformation applied.
"""
function apply_transformation(func_name::Symbol, data_vector::AbstractVector)
    if func_name == :zscore
        return standardize(ZScoreTransform, data_vector)
    elseif func_name == :log
        # Add offset to handle non-positive values before log transform.
        offset = 1.0 - minimum(data_vector)
        return log.(data_vector .+ offset)
    elseif func_name == :center
        return data_vector .- mean(data_vector)
    elseif func_name == :scale
        # Scales data to the [0, 1] range.
        return standardize(UnitRangeTransform, data_vector)
    else
        @warn "Unknown transformation function ':$func_name'. Returning original data."
        return data_vector
    end
end


"""
    _rewrite_transformations!(nodes::Vector, data::DataFrame)

Recursively traverses the formula's AST, applies data transformations, and rewrites
the AST nodes to use the newly created data columns.
"""
function _rewrite_transformations!(nodes::Vector, data::DataFrame)
    new_nodes = []
    for node in nodes
        if hasproperty(node, :type) && node.type == :operator && node.op == :pipe && length(node.children) == 2
            lhs, rhs = node.children
            if hasproperty(lhs, :module_type) && lhs.module_type in TRANSFORMATION_FUNCTIONS
                # This is a transformation pipe, e.g., `zscore(temp) |> fixed()`
                transform_func_name = lhs.module_type
                vars_to_transform = get(lhs.args, :positional_args, [])
                if isempty(vars_to_transform)
                    @warn "Transformation module '$(transform_func_name)' called without a variable. Skipping."
                    push!(new_nodes, rhs) # Keep the RHS component
                    continue
                end
                var_sym = vars_to_transform[1]

                if !hasproperty(data, var_sym)
                    error("Variable ':$var_sym' for transformation not found in data.")
                end
                
                # Apply transformation and add new column to the DataFrame
                original_data = data[!, var_sym]
                transformed_data = apply_transformation(transform_func_name, original_data)
                
                new_col_name = Symbol("$(var_sym)_$(transform_func_name)")
                counter = 1
                while hasproperty(data, new_col_name)
                    counter += 1
                    new_col_name = Symbol("$(var_sym)_$(transform_func_name)_$(counter)")
                end
                data[!, new_col_name] = transformed_data

                # Rewrite the RHS node to use the new variable
                new_rhs_args = deepcopy(rhs.args)
                new_rhs_args[:positional_args] = [new_col_name]
                new_rhs = (module_type=rhs.module_type, args=new_rhs_args)
                
                # Recursively process the rewritten node
                rewritten_children = _rewrite_transformations!([new_rhs], data)
                append!(new_nodes, rewritten_children)
            else
                # Not a transformation pipe, so process children recursively
                rewritten_children = _rewrite_transformations!(node.children, data)
                push!(new_nodes, (type=:operator, op=:pipe, children=rewritten_children))
            end
        elseif hasproperty(node, :type) && node.type == :operator
            # Handle other operators like ⊗ and ∘
            rewritten_children = _rewrite_transformations!(node.children, data)
            push!(new_nodes, (type=node.type, op=node.op, children=rewritten_children))
        else
            # It's a terminal node (a standard component)
            push!(new_nodes, node)
        end
    end
    return new_nodes
end

"""
    generate_rff_params(...)

Generates random projection weights (W) and biases (b) for RFF approximation.
This version is included for completeness and supports ARD.
"""
function generate_rff_params(in_dims::Int, n_features::Int, lengthscale::Union{Real, AbstractVector}, kernel_name::String)
    b = rand(Uniform(0, 2 * pi), n_features)
    W = Matrix{Float64}(undef, in_dims, n_features)
    k_name = lowercase(kernel_name)

    if k_name in ["se", "gaussian", "rbf"]
        if lengthscale isa Real
            W .= rand(Normal(0, 1.0 / lengthscale), in_dims, n_features)
        else
            if length(lengthscale) != in_dims; error("ARD lengthscale vector length mismatch."); end
            for d in 1:in_dims; W[d, :] = rand(Normal(0, 1.0 / lengthscale[d]), n_features); end
        end
    elseif occursin("matern", k_name)
        nu = if k_name == "matern12"; 0.5; elseif k_name == "matern32"; 1.5; else 2.5; end
        df = 2 * nu
        if lengthscale isa Real
            W .= (sqrt(df) / lengthscale) .* rand(TDist(df), in_dims, n_features)
        else
            if length(lengthscale) != in_dims; error("ARD lengthscale vector length mismatch."); end
            for d in 1:in_dims; W[d, :] = (sqrt(df) / lengthscale[d]) .* rand(TDist(df), n_features); end
        end
    else
        @warn "Kernel '$kernel_name' not recognized for RFF. Defaulting to SE."
        return generate_rff_params(in_dims, n_features, lengthscale, "se")
    end
    return W, b
end


function split_terms_at_depth(input::AbstractString, sep::AbstractString)
    # Purpose: Splits a string by a separator, but only when the separator is not inside parentheses or brackets.
    # Rationale: This version uses a robust iteration pattern that correctly handles multi-byte characters
    #            (like `∘` or `⊗`) and avoids StringIndexError.
    # v1.1.0 (2026-08-01)
    terms = String[]
    current_term = IOBuffer()
    depth = 0
    
    i = 1
    while i <= ncodeunits(input)
        # Check for separator at current position, but only if not inside parentheses.
        if depth == 0 && startswith(SubString(input, i), sep)
            # Finalize the current term and add it to the list.
            push!(terms, strip(String(take!(current_term))))
            
            # Advance index past the separator.
            i = nextind(input, i, length(sep))
            continue
        end
        
        # Append character to current term and update depth.
        char = input[i]
        if char == '(' || char == '['
            depth += 1
        elseif char == ')' || char == ']'
            depth -= 1
        end
        
        write(current_term, char)
        
        # Move to the next character's starting byte index.
        i = nextind(input, i)
    end
    
    # Add the final term after the last separator.
    push!(terms, strip(String(take!(current_term))))
    
    # Filter out any empty strings that might result from leading/trailing separators.
    return filter!(!isempty, terms)
end



"""
    _infer_structure_from_args(args::Dict)

Infers the `structure` of a `random()` module call based on its arguments.

# Rationale
This function implements a hierarchical inference logic to automatically determine the
structural type of a model component (e.g., :spatial, :temporal, :smooth) when it is
not explicitly provided by the user. This restores a key convenience feature from
previous versions, reducing verbosity in model formulas.

The inference follows a specific order of precedence to ensure robustness:
1.  **Explicit User Input**: If `structure` or the legacy `domain` keyword is provided,
    it is always respected.
2.  **Spacetime Tuple Syntax**: The syntax `model=(m1, m2)` is unambiguously inferred
    as a `:spacetime` interaction.
3.  **Hardcoded Unambiguous Models**: A comprehensive, hardcoded map (`KNOWN_UNAMBIGUOUS_MODELS`)
    is checked. This provides a robust fallback for all standard, built-in models
    (e.g., `:leroux` is always `:spatial`), ensuring inference works even if component
    files have not been loaded or registered correctly at parse time.
4.  **Runtime-Registered Models**: The `MODEL_TO_STRUCTURE_MAP`, which is populated at
    runtime as component files are loaded, is checked. This allows user-defined
    components to integrate with the inference system.
5.  **Ambiguous Models**: For a small set of truly ambiguous models (e.g., `:iid`, `:gp`),
    the function inspects the number and names of the input variables to infer the
    structure (e.g., a variable named `year` implies `:temporal`).
6.  **Special Keywords**: Catches other specific cases, like `point_process=:lgcp`.
7.  **Error**: If no structure can be inferred, an informative error is thrown.

This version prioritizes the hardcoded map over the runtime map to guarantee that
core model types are always resolved correctly, addressing failures where component
files might not be loaded in time for parsing.

# Version
v1.0.3 (2026-08-08)

# Arguments
- `args::Dict`: A dictionary of parsed arguments from the module call.

# Returns
- `Symbol`: The inferred structure (e.g., :spatial, :temporal, :smooth).
"""
function _infer_structure_from_args(args::Dict)
    # Priority 1: Explicit user-provided structure.
    if haskey(args, :structure)
        return args[:structure]
    end
    if haskey(args, :domain) # Support legacy `domain` keyword
        return args[:domain]
    end

    model = get(args, :model, nothing)
    vars = get(args, :vars, [])

    # Priority 2: Spacetime interaction syntax, e.g., `random(s, t, model=(m1, m2))`.
    if model isa Tuple && length(vars) >= 2
        return :spacetime
    end

    # Handle aliases before performing lookups.
    # if model == :proper_car; model = :sar; end
 
    # Priority 3: Check against the hardcoded map of known, unambiguous models.
    # This is the most robust fallback for built-in models.
    if haskey(KNOWN_UNAMBIGUOUS_MODELS, model)
        return KNOWN_UNAMBIGUOUS_MODELS[model]
    end

    # Priority 4: Check the runtime-populated map from loaded component files.
    if haskey(MODEL_TO_STRUCTURE_MAP, model)
        return MODEL_TO_STRUCTURE_MAP[model]
    end

    # Priority 5: Ambiguous models requiring variable inspection.
    # These models can be spatial, temporal, or smooth depending on the input variables.
    if model in AMBIGUOUS_MODELS
        num_vars = length(vars)
        if num_vars >= 2
            # Check for explicit spatiotemporal variable names.
            if num_vars == 2
                var1_name = string(vars[1])
                var2_name = string(vars[2])
                
                is_sp_1 = occursin(r"spatial|space|s_idx|auid|region|area|location"i, var1_name)
                is_t_1 = occursin(r"year|time|t_idx|month|day"i, var1_name)
                
                is_sp_2 = occursin(r"spatial|space|s_idx|auid|region|area|location"i, var2_name)
                is_t_2 = occursin(r"year|time|t_idx|month|day"i, var2_name)

                if (is_sp_1 && is_t_2) || (is_t_1 && is_sp_2)
                    return :spacetime
                end
            end
            return :smooth # Default for 2+ variables if not explicitly spatiotemporal.
        elseif num_vars == 1
            var_name = string(first(vars))
            # If 'iid' model with 'W' matrix, it's spatial.
            if model == :iid && haskey(args, :W)
                return :spatial
            end
            if occursin(r"year|time|t_idx|tuid|month|day"i, var_name) # Variable name suggests temporal.
                return :temporal
            elseif occursin(r"spatial|space|s_idx|auid|region|area|location"i, var_name) # Variable name suggests spatial.
                return :spatial
            else
                return :smooth # Default for a single, non-specific variable.
            end
        else 
            error("Ambiguous model '$model' used with no variables. Please specify a `structure` (e.g., structure=:spatial).")
        end
    end
    
    # Priority 6: Special keyword arguments like `point_process`.
    if haskey(args, :point_process) && args[:point_process] == :lgcp
        return :spatial
    end

    # Fallback: Inference failed.
    error("Could not infer structure for model '$model'. Please specify `structure` explicitly (e.g., structure=:spatial).")
end



"""
    _parse_arguments_from_expr(args::Vector{Any})

Parses the arguments from a Julia expression (specifically, the `args` field of a `:call` expression)
into a dictionary of keyword arguments and a list of positional arguments.

# Rationale
This function is a core part of the formula parsing engine that works directly with
Julia's Abstract Syntax Tree (AST). It correctly distinguishes between positional
arguments (like variable names) and keyword arguments (like `model=:bym2`). It preserves
expressions (e.g., `prior=Normal(0,1)`) for later evaluation, which is crucial for
allowing complex objects as parameters.

# Arguments
- `args`: A vector of arguments from an `Expr` object.

# Returns
- A `Dict{Symbol, Any}` where keyword arguments are stored by their key, and positional
  arguments are stored under the `:vars` key.
"""
function _parse_arguments_from_expr(args::Vector{Any})
    parsed_args = Dict{Symbol, Any}()
    positional_args = []

    for arg in args
        if arg isa Expr && arg.head == :kw
            # This is a keyword argument, e.g., `model=:bym2`.
            # arg.args is the key (e.g., :model)
            # arg.args[1] is the key (e.g., :model)
            # arg.args[2] is the value (e.g., :bym2 as a QuoteNode or `Normal(0,1)` as an Expr)
            key = arg.args[1]
            value = arg.args[2]
            
            # If the value is a QuoteNode, extract the inner symbol.
            # Otherwise, keep it as is (it could be a literal or another expression).
            if value isa QuoteNode
                parsed_args[key] = value.value # Extract the Symbol from QuoteNode
            else
                parsed_args[key] = value
            end
        else
            # This is a positional argument, e.g., `s_idx`.
            push!(positional_args, arg)
        end
    end

    # Store positional arguments under the conventional `:vars` key.
    if !isempty(positional_args)
        parsed_args[:vars] = positional_args
    end

    return parsed_args
end





"""
    _parse_value(val_str::String)

Parses a string value from a formula argument into an appropriate Julia type.

# Rationale for Update
This function has been updated to be more robust in parsing symbol literals. The
original implementation could, in some contexts, misinterpret a symbol like `:foo`
as the string `":foo"`. The new logic explicitly checks if the string starts with
a colon (`:`) and, if so, correctly converts the rest of the string to a `Symbol`.
This ensures that `model=:bym2` is always parsed as the symbol `:bym2`, resolving
the `KeyError`. It also correctly handles bare words (e.g., `log_offsets=my_offsets_col`)
and other value types.
"""
function _parse_value(val_str::String)
    val_str = strip(val_str)

    # New logic: explicitly handle symbol literals first.
    if startswith(val_str, ":")
        # This handles `:foo` -> Symbol("foo") -> :foo
        return Symbol(val_str[2:end])
    elseif (startswith(val_str, "'") && endswith(val_str, "'")) || (startswith(val_str, "\"") && endswith(val_str, "\""))
        # It's a string literal like "foo"
        return String(val_str[2:end-1])
    elseif val_str == "true"
        return true
    elseif val_str == "false"
        return false
    elseif occursin(r"^[a-zA-Z_][a-zA-Z0-9_]*$", val_str)
        # It's a bare word, treat as a symbol (variable name)
        return Symbol(val_str)
    else
        # It could be a number, or a complex expression like Normal(0,1)
        try
            return Meta.parse(val_str)
        catch
            # If all else fails, it's just a string that wasn't quoted
            return String(val_str)
        end
    end
end


_parse_value(val_str::SubString{String}) = _parse_value(String(val_str))


function _add_parsed_arg!(args_dict::Dict{Symbol, Any}, positional_args::Vector{Any}, arg_val::AbstractString)
    # Purpose: A helper to add a parsed argument to either the keyword or positional argument list.
    # Rationale: Centralizes the logic for distinguishing between `key=value` and positional arguments.
    #            This version is updated to accept `AbstractString` to handle both `String` and `SubString`
    #            types, resolving a `MethodError`.
    # v1.0.1 (2026-08-02)
    # Inputs:
    #   - args_dict: Dictionary for keyword arguments.
    #   - positional_args: Vector for positional arguments.
    #   - arg_val: The argument string to parse.
    # Outputs: None (mutates input collections).
    if contains(arg_val, "=")
        key_val = Base.split(arg_val, "=", limit=2)
        key = Symbol(Base.strip(key_val[1]))
        val_str = String(Base.strip(key_val[2]))
        args_dict[key] = _parse_value(val_str)
    else
        push!(positional_args, _parse_value(arg_val))
    end
end



function _parse_arguments_string(args_str::String)
    # Purpose: Parses the inner content of a module call, e.g., "s_idx, model=bym2".
    # Rationale: This version is updated to correctly handle commas within string literals,
    #            preventing incorrect splitting of arguments.
    # v1.1.0 (2026-08-01)
    args_dict = Dict{Symbol, Any}()
    positional_args = []
    current_arg = IOBuffer()
    depth = 0
    in_string = false
    string_char = ' '

    for char in args_str
        if char == '"' || char == '\''
            if !in_string
                in_string = true
                string_char = char
            elseif char == string_char
                in_string = false
            end
        end

        if char == ',' && depth == 0 && !in_string
            arg_val = strip(String(take!(current_arg)))
            if !isempty(arg_val)
                _add_parsed_arg!(args_dict, positional_args, arg_val)
            end
        else
            write(current_arg, char)
            if !in_string
                if char == '(' || char == '['; depth += 1;
                elseif char == ')' || char == ']'; depth -= 1; end
            end
        end
    end

    arg_val = strip(String(take!(current_arg)))
    if !isempty(arg_val)
        _add_parsed_arg!(args_dict, positional_args, arg_val)
    end

    if !isempty(positional_args)
        args_dict[:positional_args] = positional_args
    end

    return args_dict
end



function _sanitize_variablename(name::String)
    # Purpose: Sanitizes a string to be a valid Julia variable name.
    # Rationale: The formula parser can generate module keys with characters (e.g., "|", " ")
    #            that are invalid in variable names. This function cleans those keys before
    #            they are used by the code generator to create variable names for latent effects.
    # Inputs:
    #   - name: The raw string from the formula module key.
    # Outputs: A sanitized version of the name suitable for use as a variable.
    s = name
    # Replace pipe, parentheses, spaces, and other invalid characters with an underscore
    s = replace(s, r"[\s\|()\+\*&:]" => "_")
    # Replace kronecker product symbol
    s = replace(s, "⊗" => "_kron_")
    # Replace composition symbol
    s = replace(s, "∘" => "_comp_")
    # Remove any leading or trailing underscores that may result
    s = Base.strip(s, '_')
    # Consolidate sequences of multiple underscores into a single one
    return replace(s, r"__+" => "_")
end

function _parse_single_component_term(term_str::AbstractString)
    # Purpose: Parses a single module call string like "spatial(s_idx, model=bym2)".
    # Rationale: Extracts the module name and its arguments.
    # Assumptions: The input is a single, well-formed module call.
    # Inputs:
    #   - term_str: The module call string.
    # Outputs: A NamedTuple `(module_type, args)`.
    term_str = Base.strip(term_str)
    m = match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$", term_str)
    
    if m === nothing
        # This error is thrown when a term without parentheses is incorrectly passed to this function.
        error("Internal Parser Error: _parse_single_component_term was called with a non-module term '$term_str'. This indicates a logic error in the parent parser.")
    end

    module_name = Symbol(m.captures[1])
    args_str = String(m.captures[2])
    args_dict = _parse_arguments_string(args_str)

    return (module_type = module_name, args = args_dict)
end


"""
    resolve_hyperpriors(...)

# Rationale for Update
The `possible_priors` list has been updated to include `:rho1` and `:rho2`. This
ensures that the function will correctly search for and resolve priors for the
two coefficients of an `AR2` model, following the standard 3-level precedence
(local, global, scheme).
"""
function resolve_hyperpriors(model_name::String, global_priors::Dict, local_params::Dict, scheme::Symbol, calling_mod::Module)
    prior_defaults = if scheme == :pcpriors
        PC_PRIORS
    elseif scheme == :informative
        INFORMATIVE_PRIORS
    else
        UNINFORMATIVE_PRIORS
    end
    is_anisotropic = get(local_params, :anisotropic, false)
    in_dims = get(local_params, :in_dims, 0)

    possible_priors = [:sigma, :rho, :rho1, :rho2, :lengthscale, :kappa, :amplitude, :phase, :pca_sd, :pdef_sd, :range]

    resolved = Dict{Symbol, Any}()

    for p_sym in possible_priors
        p_base_name = p_sym

        is_ard_param = p_sym in [:lengthscale, :kappa]

        if is_ard_param && is_anisotropic
            if in_dims == 0; error("Cannot resolve anisotropic prior for '$p_sym' because input dimensionality is unknown."); end
            
            prior_val = get(local_params, p_sym, nothing)
            
            if prior_val isa Expr && prior_val.head == :vect
                # Case: lengthscale=[Normal(0,1), Normal(0,1)]
                resolved_priors = [Core.eval(calling_mod, p) for p in prior_val.args]
                if length(resolved_priors) != in_dims; error("Anisotropic prior vector for '$p_sym' has length $(length(resolved_priors)), but expected $in_dims."); end
                resolved[p_sym] = resolved_priors
            else
                # Case: lengthscale=Normal(0,1) or default
                single_prior = if !isnothing(prior_val)
                    prior_val isa Expr ? Core.eval(calling_mod, prior_val) : prior_val
                elseif haskey(global_priors, Symbol(model_name, "_", p_sym)); global_priors[Symbol(model_name, "_", p_sym)]
                elseif haskey(global_priors, p_sym); global_priors[p_sym]
                else; prior_defaults[string(p_sym)]; end
                
                if !(single_prior isa Distribution); error("Resolved prior for '$p_sym' is not a Distribution."); end
                resolved[p_sym] = [single_prior for _ in 1:in_dims]
            end
            continue
        end

        if haskey(local_params, p_sym)
            prior_val = local_params[p_sym]
            if prior_val isa Tuple
                resolved[p_sym] = create_pc_prior(p_base_name, prior_val)
            elseif prior_val isa Expr
                try
                    resolved[p_sym] = Core.eval(calling_mod, prior_val)
                catch e
                    error("Could not evaluate `prior` argument `$(prior_val)` for component '$model_name'. Error: $e")
                end
            else
                resolved[p_sym] = prior_val
            end
            continue
        end

        global_key_model = Symbol(model_name, "_", p_base_name)
        global_key_param = p_base_name
        
        if haskey(global_priors, global_key_model)
            resolved[p_sym] = global_priors[global_key_model]
            continue
        elseif haskey(global_priors, global_key_param)
            resolved[p_sym] = global_priors[global_key_param]
            continue
        elseif haskey(prior_defaults, string(p_base_name))
            resolved[p_sym] = prior_defaults[string(p_base_name)]
        end
    end

    return NamedTuple(resolved)
end


function _is_outermost_grouping_parentheses(s::AbstractString)
    if !startswith(s, "(") || !endswith(s, ")")
        return false
    end
    depth = 0
    # Iterate from the second character to the second-to-last to check the balance
    # of parentheses within the outer pair.
    for i in 2:ncodeunits(s)-1
        char = s[i]
        if char == '('
            depth += 1
        elseif char == ')'
            depth -= 1
        end
        # If depth becomes negative, it means a closing parenthesis appeared before its opening one.
        if depth < 0
            return false
        end
        # If depth returns to 0 before the end of the string, it means the outer
        # parentheses are not grouping the entire expression.
        if depth == 0
            return false
        end
    end
    # The depth should be exactly 0 after checking all internal characters.
    return depth == 0
end



function _parse_rhs_expression(term_str::AbstractString)
    term_str_stripped = Base.strip(term_str)

    # If the expression is wrapped in balanced parentheses, parse the inner content recursively.
    if startswith(term_str_stripped, "(") && endswith(term_str_stripped, ")")
        # Safely get the inner content, respecting multi-byte characters
        inner_content = SubString(term_str_stripped, nextind(term_str_stripped, 1), prevind(term_str_stripped, lastindex(term_str_stripped)))
        
        # Check if the parentheses around the inner content are balanced.
        # This is a simple check to avoid parsing malformed strings.
        depth = 0
        is_balanced = true
        for char in inner_content
            if char == '('; depth += 1; elseif char == ')'; depth -= 1; end
            if depth < 0; is_balanced = false; break; end
        end
        if depth != 0; is_balanced = false; end

        # If the parentheses are balanced, it's a grouped expression.
        # We recurse on the inner content to handle nested parentheses and expressions.
        if is_balanced
            return _parse_rhs_expression(inner_content)
        end
    end

    # Proceed with parsing based on operator precedence.
    parts = split_terms_at_depth(term_str_stripped, " |> ")
    if length(parts) > 1
        return (type=:operator, op=:pipe, children=[_parse_rhs_expression(parts[1]), _parse_rhs_expression(join(parts[2:end], " |> "))])
    end
    parts = split_terms_at_depth(term_str_stripped, " ⊗ ")
    if length(parts) > 1
        return (type=:operator, op=:kronecker_product, children=[_parse_rhs_expression(p) for p in parts])
    end
    parts = split_terms_at_depth(term_str_stripped, " ∘ ")
    if length(parts) > 1
        return (type=:operator, op=:composition, children=[_parse_rhs_expression(p) for p in parts])
    end

    # If no operators are found, parse as a single module or a fixed effect.
    if occursin(r"\(.*\)", term_str_stripped)
        return _parse_single_component_term(term_str_stripped)
    else
        return (module_type = :fixed, args = Dict(:positional_args => [term_str_stripped]))
    end
end




function _categorize_rhs_nodes!(nodes, modules, fixed_effects)
    for node in nodes
        is_pp_composition = false
        if hasproperty(node, :type) && node.type == :operator && node.op == :composition && length(node.children) == 2
            outer_node, inner_node = node.children[1], node.children[2]
            if outer_node.module_type == :pointprocess && inner_node.module_type == :random
                is_pp_composition = true
                
                pp_module_type = get(outer_node.args, :model, :lgcp)
                final_params = copy(inner_node.args)
                for (k, v) in outer_node.args
                    if k != :model; final_params[k] = v; end
                end
                
                vars = get(inner_node.args, :positional_args, [])
                new_module_data = (module_type = pp_module_type, args = final_params)
                
                key_parts = [string(pp_module_type), join(string.(vars), "_")]
                raw_key = join(filter(!isempty, key_parts), "_")
                sanitized_base_key = _sanitize_variablename(raw_key)
                module_key = sanitized_base_key
                counter = 1
                while haskey(modules, module_key)
                    counter += 1
                    module_key = "$(sanitized_base_key)_$(counter)"
                end
                modules[module_key] = new_module_data
            end
        end

        if is_pp_composition
            continue
        end

        if hasproperty(node, :type) && node.type == :operator
            # Simplified key generation for composed nodes.
            function _get_simplified_composed_node_key(n)
                if hasproperty(n, :type) && n.type == :operator
                    op_str = string(n.op)
                    child_keys = [_get_simplified_composed_node_key(child) for child in n.children]
                    return join(child_keys, "_$(op_str)_")
                elseif hasproperty(n, :module_type)
                    pos_args = get(n.args, :positional_args, [])
                    return isempty(pos_args) ? string(n.module_type) : join(string.(pos_args), "_")
                else
                    return "unknown"
                end
            end
            raw_key = _get_simplified_composed_node_key(node)
            sanitized_base_key = _sanitize_variablename(raw_key)
            module_key = isempty(sanitized_base_key) ? "composed_$(length(modules)+1)" : sanitized_base_key
            
            final_key = module_key
            counter = 1
            while haskey(modules, final_key)
                counter += 1
                final_key = "$(module_key)_$(counter)"
            end
            modules[final_key] = (module_type = :interact, args = Dict(:operator => node.op, :components => node.children))

        elseif hasproperty(node, :module_type)
            m_type = node.module_type
            if m_type in BSTM_MODULE_KEYWORDS
                local raw_key
                pos_args = get(node.args, :positional_args, [])
                
                if m_type == :mixed
                    # For `mixed(effect | group)`, the key is based on the group variable.
                    if !isempty(pos_args) && pos_args[1] isa Expr && pos_args[1].head == :call && pos_args[1].args[1] == :|
                        group_var_str = string(pos_args[1].args[3])
                        raw_key = group_var_str
                    else
                        raw_key = isempty(pos_args) ? "mixed_unknown" : string(pos_args[1])
                    end
                elseif !isempty(pos_args)
                    # For most modules, the key is based on the variables it acts on.
                    raw_key = join([string(a) for a in pos_args], "_")
                else
                    # Fallback for modules with no positional args (e.g., intercept).
                    raw_key = string(m_type)
                end

                sanitized_base_key = _sanitize_variablename(raw_key)
                module_key = sanitized_base_key
                counter = 1
                while haskey(modules, module_key)
                    counter += 1
                    module_key = "$(sanitized_base_key)_$(counter)"
                end
                modules[module_key] = node
            else
                push!(fixed_effects, string(m_type))
            end
        end
    end
end





function _extract_symbols_from_expr!(sym_set::Set{Symbol}, ex::Expr)
    # Purpose: Recursively extracts all symbols from a Julia expression.
    # Rationale: Used to find all variable names mentioned in the formula to check for their presence in the data.
    # Assumptions: Input is a valid Julia expression.
    # Inputs:
    #   - sym_set: A set to store the found symbols.
    #   - ex: The expression to traverse.
    # Outputs: None (mutates `sym_set`).
    for arg in ex.args
        if arg isa Symbol
            push!(sym_set, arg)
        elseif arg isa Expr
            _extract_symbols_from_expr!(sym_set, arg)
        end
    end
end

function _parse_lhs_term(term::String)
    # Purpose: Parses a single term from the left-hand side of the formula.
    # Rationale: Handles both `likelihood()` blocks and bare outcome variables,
    #            including multiple outcomes specified with `+`.
    # Inputs:
    #   - term: A string representing one part of the LHS.
    # Outputs: A vector of outcome specification dictionaries.
    term = Base.strip(term)
    m = match(r"likelihood\((.*)\)", term)
    specs = Dict{Symbol, Any}[]
    if !isnothing(m)
        inner_content = m.captures[1]
        args = split_terms_at_depth(inner_content, ",")
        if isempty(args); return specs; end
        outcome_var_str = Base.strip(args[1])
        params_str = join(args[2:end], ",")
        params = _parse_arguments_string(params_str)
        for ov in [Base.strip(s) for s in Base.split(outcome_var_str, '+')]; push!(specs, Dict(:var => ov, :params => params)); end
    else
        for ov in [Base.strip(s) for s in Base.split(term, '+')]; push!(specs, Dict(:var => ov, :params => Dict())); end
    end
    return specs
end


function decompose_bstm_formula(formula_str::String, data::DataFrame)
    # Purpose: Decomposes a formula string into its constituent parts.
    # Rationale: This version is updated to handle formulas that do not contain a right-hand
    #            side (RHS). If the `~` operator is missing, it now defaults the RHS to "1"
    #            (an intercept-only model), preventing a `BoundsError` during parsing.
    # v1.0.1 (2026-08-01)
    # Inputs:
    #   - formula_str: The model formula string.
    #   - data: The input DataFrame, used for AST rewriting.
    # Outputs: A NamedTuple containing the parsed formula components.
    parts = Base.split(formula_str, "~")
    lhs_str = Base.strip(parts[1])
    
    # If there is no '~', default the RHS to "1" (intercept only).
    rhs_str = if length(parts) > 1
        Base.strip(parts[2])
    else
        "1" 
    end

    outcome_specs = vcat([_parse_lhs_term(term) for term in split_terms_at_depth(lhs_str, "+")]...)

    rhs_normalized = replace(rhs_str, r"\s*-\s*" => " + -")
    rhs_terms = split_terms_at_depth(rhs_normalized, " + ")
    
    has_intercept = !("0" in rhs_terms || "-1" in rhs_terms)
    intercept_prior = nothing
    module_terms = String[]
    intercept_module_found = false

    for term in rhs_terms
        term_stripped = Base.strip(term)
        if term_stripped == "0" || term_stripped == "-1" || term_stripped == "1"
            continue
        elseif startswith(term_stripped, "intercept(")
            if !intercept_module_found
                intercept_module_found = true
                intercept_data = _parse_single_component_term(term_stripped)
                has_intercept = get(intercept_data.args, :positional_args, [true])[1] != false
                if haskey(intercept_data.args, :prior); intercept_prior = intercept_data.args[:prior]; end
            end
        else
            push!(module_terms, term_stripped)
        end
    end

    top_level_nodes = [_parse_rhs_expression(term) for term in module_terms]
    
    rewritten_nodes = _rewrite_transformations!(top_level_nodes, data)

    modules = Dict{String, Any}()
    fixed_effects = String[]
    _categorize_rhs_nodes!(rewritten_nodes, modules, fixed_effects)
    
    return (outcomes=outcome_specs, modules=modules, fixed_effects=unique(fixed_effects), has_intercept=has_intercept, intercept_prior=intercept_prior)
end



# ==============================================================================
# SECTION 4: MODEL CONFIGURATION ENGINE
# ==============================================================================


function _precompute_likelihood_params!(M::Dict)
    # Purpose: Ensures all observation-level likelihood parameters are consistently formatted as matrices.
    # Rationale: This function standardizes all likelihood parameters (per-observation and per-outcome)
    #            into matrices of size `(N, K)` (observations x outcomes). This simplifies the downstream
    #            code generator, which can then access these parameters with simple indexing `[i, k]`
    #            inside the model's observation loop, avoiding inefficient runtime conditional checks.
    # v1.0.0 (2026-07-18) - Aligned with new parser logic for scalar-per-outcome parameters.
    # Assumptions: M[:y_N] (number of observations) and M[:outcomes_N] (number of outcomes) are set.
    # Inputs:
    #   - M: The model configuration dictionary, which is mutated by this function.
    # Outputs: None.

    N = M[:y_N]
    K = M[:outcomes_N]

    param_defaults = [
        (key=:censor_lower, default=-Inf, is_scalar_per_outcome=true),
        (key=:censor_upper, default=Inf, is_scalar_per_outcome=true),
        (key=:hurdle, default=-Inf, is_scalar_per_outcome=true),
        (key=:trials, default=1, is_scalar_per_outcome=false),
        (key=:weights, default=1.0, is_scalar_per_outcome=false),
        (key=:log_offsets, default=0.0, is_scalar_per_outcome=false)
    ]

    for spec in param_defaults
        key = spec.key
        default_val = spec.default
        is_scalar_per_outcome = spec.is_scalar_per_outcome

        final_matrix = Matrix{typeof(default_val)}(undef, N, K)

        if !haskey(M, key)
            fill!(final_matrix, default_val)
        else
            val = M[key]
            if is_scalar_per_outcome
                if val isa Real
                    fill!(final_matrix, val)
                elseif val isa AbstractVector && length(val) == K # Multivariate case
                    final_matrix = repeat(val', N, 1)
                else
                    @warn "Scalar-per-outcome parameter `:$key` has unexpected type or dimensions. Using default."
                    fill!(final_matrix, default_val)
                end
            else # Per-observation parameter
                if val isa Real
                    fill!(final_matrix, val)
                elseif val isa AbstractVector && length(val) == N
                    final_matrix = repeat(val, 1, K)
                elseif val isa AbstractMatrix && size(val) == (N, K)
                    final_matrix = val
                else
                    @warn "Per-observation parameter `:$key` has incorrect dimensions. Expected scalar, vector of length $N, or matrix ($N, $K). Using default."
                    fill!(final_matrix, default_val)
                end
            end
        end
        M[key] = final_matrix
    end
end


"""
    bstm_config(formula::String, data::DataFrame; calling_module::Module=Main, kwargs...)

Constructs the complete model configuration by parsing the formula, processing data,
and assembling component specifications using the explicit component interface. This
corrected version includes the main component processing loop that dispatches modules
to their respective processors, enabling features like `nested()` models.
"""
function bstm_config(formula::String, data::DataFrame; calling_module::Module=Main, kwargs...)
    df_processed = deepcopy(data)
    decomposed_formula = decompose_bstm_formula(formula, df_processed)

    M = _initialize_config(df_processed, merge(Dict(kwargs), Dict(:calling_module => calling_module)))
    M[:formula] = formula
    
    _process_lhs!(M, decomposed_formula.outcomes)
    
    is_multivariate = get(M, :model_arch, "univariate") == "multivariate"
    if is_multivariate
        for (key, mod_data_nt) in decomposed_formula.modules
            model_name = get(mod_data_nt.args, :model, :none)
            if mod_data_nt.module_type == :dynamics && model_name in [:leslie_matrix, :delay_difference, :generalized_lotka_volterra, :generalized_leslie_matrix]
                M[:is_multivariate_dynamics] = true
                M[:multivariate_dynamics_key] = key
                break
            end
        end
    end

    _precompute_likelihood_params!(M)

    M[:add_intercept] = decomposed_formula.has_intercept
    if !isnothing(decomposed_formula.intercept_prior)
        prior_val = decomposed_formula.intercept_prior
        if prior_val isa Expr
            try; M[:intercept_prior] = Core.eval(calling_module, prior_val);
            catch e; error("Could not evaluate `prior` argument `$(prior_val)` in intercept() module. Error: $e"); end
        else; M[:intercept_prior] = prior_val; end
    end

    # --- Main Component Processing Loop ---
    # This loop iterates through all parsed modules from the formula's RHS.
    # It dispatches each module to its specific processor function (if one exists)
    # via the `MODULE_PROCESSORS` dictionary. This is the core mechanism that
    # enables data-dependent setup for modules like `nested()`, `random()`, etc.
    for (key, mod_data_nt) in decomposed_formula.modules
        mod_type = mod_data_nt.module_type
        mod_data_dict = Dict(:key => key, :type => mod_type, :variables => get(mod_data_nt.args, :positional_args, []), :params => mod_data_nt.args)

        processor! = get(MODULE_PROCESSORS, mod_type, nothing)
        
        create_component = true
        if !isnothing(processor!)
            # The processor function performs data-dependent setup and returns a boolean
            # indicating whether a component object should be created.
            create_component = processor!(M, mod_data_dict, M, M[:hyperpriors])
        end

        if !create_component
            continue
        end

        # Instantiate the final component object
        component_obj = resolve_technical_primitive(mod_data_dict, M, M[:hyperpriors], M[:prior_scheme])
        mod_data_dict[:component_obj] = component_obj

        # Call get_precomputes to generate templates, basis matrices, etc.
        M_nt = NamedTuple(M)
        precomputes = get_precomputes(component_obj, M_nt, mod_data_dict)

        # Assemble the final component specification and register it.
        spec = (
            key=Symbol(key), 
            structure=mod_data_dict[:type], 
            var=join(string.(mod_data_dict[:variables]), "_"), 
            component_obj=component_obj, 
            params=mod_data_dict[:params], 
            hyper=precomputes
        )
        push!(M[:components], spec)
    end

    all_fixed_effects = copy(decomposed_formula.fixed_effects)
    if haskey(M, :fixed_effects_from_modules)
        append!(all_fixed_effects, M[:fixed_effects_from_modules])
    end
    _process_fixed_effects!(M, unique(all_fixed_effects))
    _process_fixed_effects_priors!(M)
    _finalize_config!(M)
    
    return NamedTuple(M)
end




# Version 1.0.0 (2026-08-06)
# Purpose: Generates a NamedTuple of full variable names for a given component.
# Rationale: This utility function centralizes the naming convention for all parameters
#            associated with a model component, ensuring consistency across the framework.
#            It handles suffixes for multivariate models and prefixes for different parameter types.
# Assumptions: The component specification `spec` contains a unique `key`.
function generate_full_variable_names(spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    base_key = string(spec.key)
    full_key = isempty(prefix) ? base_key : "$(prefix)_$(base_key)"

    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    suffix = (is_multivariate && !is_shared) ? "_$(outcome_idx)" : ""

    names = Dict{Symbol, Symbol}()
    
    all_prefixes = [
        :sigma, :rho, :rho1, :rho2, :kappa, :ls, :range, :period, :amplitude, :phase,
        :velocity, :diffusion, :pca_sd, :pdef_sd, :L_corr, :sigma_effects,
        :r, :K, :q, :M_nat, :alpha, :beta, :gamma, :delta,
        :raw, :innov, :latent, :struct, :iid, :beta_cos, :beta_sin, :rho_field,
        :W, :b, :v_raw, :factors_flat, :thresh_raw, :W1, :b1, :W2, :amplitude_raw,
        :innov_predator
    ]

    for p in all_prefixes
        names[p] = Symbol("$(p)_$(full_key)$(suffix)")
    end

    return NamedTuple(names)
end




"""
    _process_fixed_effects!(M::Dict, fixed_effects_vars::Vector{String})

Processes all fixed effect variables from the formula.

This function is updated to pass the `calling_module` from the main configuration `M`
to the `create_fixed_design` function. This ensures that the formula parsing within
`create_fixed_design` occurs in the correct module context, resolving the `MethodError`
related to world age issues.
"""
function _process_fixed_effects!(M::Dict, fixed_effects_vars::Vector{String})
    if isempty(fixed_effects_vars)
        M[:Xfixed] = zeros(M[:y_N], 0)
        M[:Xfixed_N] = 0
        M[:Xfixed_names] = Symbol[]
        M[:Xfixed_applied_formula] = nothing
        return
    end

    rhs_vars = join(fixed_effects_vars, " + ")
    # Explicitly add "0" to prevent StatsModels from creating its own intercept.
    # The intercept is handled separately by the `intercept()` module.
    rhs = "0 + " * rhs_vars
    
    # Pass the calling_module to create_fixed_design.
    Xfixed_named, applied_formula = create_fixed_design(
        rhs, 
        M[:data], 
        M[:calling_module]; 
        contrasts=get(M, :contrasts, Dict())
    )

    if size(Xfixed_named, 1) != M[:y_N]
        @warn "Dimension mismatch in fixed effects design matrix: Expected $(M[:y_N]) rows, but got $(size(Xfixed_named, 1)). This can happen if there are missing values in the fixed effect covariates. Attempting to reconcile."
        # This is a simple reconciliation; a more robust solution might involve
        # filtering the main data frame based on complete cases for all model variables.
        if size(Xfixed_named, 1) < M[:y_N]
            padded_Xfixed = zeros(M[:y_N], size(Xfixed_named, 2))
            # This assumes the rows align, which might not be safe without row indices.
            # A more robust implementation would use indices from `completecases`.
            padded_Xfixed[1:size(Xfixed_named, 1), :] = Matrix(Xfixed_named)
            M[:Xfixed] = padded_Xfixed
        else
            M[:Xfixed] = Matrix(Xfixed_named[1:M[:y_N], :])
        end
    else
        M[:Xfixed] = Matrix(Xfixed_named)
    end
    
    M[:Xfixed_N] = size(M[:Xfixed], 2)
    M[:Xfixed_names] = size(Xfixed_named, 2) > 0 ? names(Xfixed_named, 2) : Symbol[]
    M[:Xfixed_applied_formula] = applied_formula
end


function _canonical_term_string(term::StatsModels.AbstractTerm)
    if term isa StatsModels.InteractionTerm
        # Sort term names for canonical representation, e.g., "a&b" is the same as "b&a"
        term_names = sort([string(t.sym) for t in term.terms])
        return join(term_names, "&")
    elseif term isa StatsModels.Term
        return string(term.sym)
    elseif term isa StatsModels.ConstantTerm
        return "(Intercept)"
    else
        # Fallback for other term types like FunctionTerm, etc.
        # This might not be perfectly canonical but is a reasonable default.
        return string(term)
    end
end



# Version 1.0.2 (2026-08-06)
# Purpose: Pre-computes a matrix factorization for static components.
# Rationale: This version reverts to using `cholesky` factorization instead of `lu`.
#            The `CanonicalIndexError` that previously prompted the switch to `lu` is now
#            addressed in the code generator by explicitly creating a `SparseMatrixCSC`
#            from the Cholesky factor before solving. This change restores the use of the
#            more efficient and appropriate Cholesky decomposition for symmetric
#            positive-definite precision matrices.
# Assumptions: The input `Q_template` is a sparse matrix representing a valid precision structure.
function _precompute_static_components!(M::Dict)
    noise = M[:noise]
    new_components = []
    # Define component types that do not have dynamic structure parameters like `rho`.
    static_component_types = [IID, ICAR, Besag, RW1, RW2, Cyclic, PSpline, TPS, BSpline, Eigen, Moran, Spherical, Barycentric, TensorProductSmooth]

    for spec_in in M[:components]
        current_spec = spec_in
        m_obj = current_spec.component_obj

        if m_obj isa MixedComponent || m_obj isa SVC
            inner_model = m_obj.model
            is_inner_static = any(T -> inner_model isa T, static_component_types)
            if is_inner_static && !isnothing(current_spec.Q_template) && size(current_spec.Q_template, 1) > 0
                try
                    Q_concrete = sparse(current_spec.Q_template)
                    # Revert to cholesky
                    F = cholesky(Symmetric(Q_concrete + noise * I))
                    final_spec = merge(current_spec, (is_static=true, cholesky_factor=F))
                    push!(new_components, final_spec)
                    continue
                catch e
                    @warn "Cholesky factorization failed for static inner model in $(current_spec.key). Reverting to dynamic computation. Error: $e"
                end
            end
        end

        if m_obj isa Composed && m_obj.operator == :pipe
            state_spec = get(current_spec.hyper, :state_spec, nothing)
            if !isnothing(state_spec)
                state_m_obj = m_obj.components[2]
                is_state_static = any(T -> state_m_obj isa T, static_component_types)

                if is_state_static && !isnothing(state_spec.Q_template) && size(state_spec.Q_template, 1) > 0
                    try
                        Q_concrete = sparse(state_spec.Q_template)
                        # Revert to cholesky
                        F = cholesky(Symmetric(Q_concrete + noise * I))
                        new_state_spec = merge(state_spec, (is_static=true, cholesky_factor=F))
                        new_hyper = merge(current_spec.hyper, (state_spec=new_state_spec,))
                        current_spec = merge(current_spec, (hyper=new_hyper,))
                    catch e
                        @warn "Cholesky factorization failed for static state component in $(current_spec.key). Reverting to dynamic computation. Error: $e"
                    end
                end
            end
        end

        is_main_static = !(current_spec.component_obj isa Composed) && any(T -> current_spec.component_obj isa T, static_component_types)

        if is_main_static && !isnothing(current_spec.Q_template) && size(current_spec.Q_template, 1) > 0
            try
                Q_concrete = sparse(current_spec.Q_template)
                # Revert to cholesky
                F = cholesky(Symmetric(Q_concrete + noise * I))
                final_spec = merge(current_spec, (is_static=true, cholesky_factor=F))
                push!(new_components, final_spec)
            catch e
                @warn "Cholesky factorization failed for static component $(current_spec.key). Reverting to dynamic computation. Error: $e"
                final_spec = merge(current_spec, (is_static=false,))
                push!(new_components, final_spec)
            end
        else
            final_spec = merge(current_spec, (is_static=get(current_spec, :is_static, false),))
            push!(new_components, final_spec)
        end
    end
    M[:components] = new_components
end

 


function _initialize_config(data::DataFrame, kwargs)
    # Purpose: Creates the initial model configuration dictionary.
    # Rationale: Centralizes the creation of the `M` object.
    # Assumptions: `data` is a DataFrame.
    # Inputs:
    #   - data: The input DataFrame.
    #   - kwargs: Keyword arguments passed from the main `bstm` call.
    # Outputs: A dictionary `M`.
    M = Dict{Symbol, Any}()
    M[:data] = data
    M[:y_N] = nrow(data)
    
    # Set defaults that can be overridden by user kwargs
    M[:noise] = 1e-6
    M[:hyperpriors] = Dict{String, Any}()
    M[:prior_scheme] = :pcpriors
    M[:fixed_effects_priors] = Dict{Symbol, Any}()
    M[:spectral_orientation] = true

    for (k, v) in kwargs; M[k] = v; end
    M[:calling_module] = get(kwargs, :calling_module, Main)
    M[:components] = []
    M[:basis_matrices] = Dict{Symbol, Any}()
    return M
end



function _process_lhs!(M::Dict, outcome_specs::Vector{Dict{Symbol, Any}})
    # Purpose: Processes the LHS of the formula, setting up outcomes and likelihoods.
    # Rationale: This version adds special handling for the `dirichlet_multinomial` family,
    #            which treats multiple variables specified with `+` as categories of a single
    #            response, rather than as separate outcome variables. This resolves an error
    #            where the parser would incorrectly try to validate each category as a
    #            standalone outcome.
    # v1.0.1 (2026-07-31)
    # Inputs:
    #   - M: The model configuration dictionary.
    #   - outcome_specs: A vector of parsed outcome specifications from the formula.
    # Outputs: None (mutates `M`).

    outcomes = [Symbol(spec[:var]) for spec in outcome_specs]
    likelihood_specs = [spec[:params] for spec in outcome_specs]
    
    # Check the family from the first spec, assuming it's consistent for a `+` separated group.
    family_type = string(get(likelihood_specs[1], :family, "gaussian"))

    if family_type == "dirichlet_multinomial"
        # --- Special Handling for Dirichlet-Multinomial ---
        # In this case, `y1 + y2 + y3` represents categories of one response.
        
        if length(outcomes) <= 1
            error("The `dirichlet_multinomial` family requires multiple outcome variables specified with `+`, e.g., `likelihood(y1 + y2, ...)`.")
        end
        
        for out_sym in outcomes
            if !hasproperty(M[:data], out_sym)
                error("Outcome category variable ':$out_sym' for dirichlet_multinomial not found in the data frame.")
            end
        end

        M[:outcomes] = outcomes
        M[:outcomes_N] = length(outcomes)
        M[:model_arch] = "multivariate"
        M[:y_obs] = Matrix(M[:data][!, M[:outcomes]])
        
        merged_params = Dict{Symbol, Any}()
        for spec in reverse(likelihood_specs); merge!(merged_params, spec); end
        M[:likelihood_specs] = [merged_params]

    else
        # --- Standard Handling for Other Families ---
        M[:outcomes] = outcomes
        M[:outcomes_N] = length(outcomes)
        M[:likelihood_specs] = likelihood_specs

        for (i, spec) in enumerate(M[:likelihood_specs])
            if !haskey(spec, :family)
                spec[:family] = "gaussian"
                @warn "Likelihood `family` not specified. Defaulting to `family=gaussian`."
            end
            
            if string(get(spec, :family, "")) == "ordinal"
                if M[:outcomes_N] > 1; error("The `ordinal` family is currently only supported for univariate models."); end
                outcome_var = M[:outcomes][i]
                if !hasproperty(M[:data], outcome_var); error("Ordinal outcome variable ':$outcome_var' not found in data."); end
                
                outcome_data = M[:data][!, outcome_var]
                if !(eltype(outcome_data) <: Integer)
                    @warn "Ordinal outcome variable ':$outcome_var' is not of integer type. Attempting to convert."
                    try; outcome_data = round.(Int, outcome_data); catch; error("Could not convert ordinal outcome variable ':$outcome_var' to integers."); end
                end
                
                unique_levels = sort(unique(outcome_data))
                K = length(unique_levels)
                if K < 2; error("Ordinal outcome variable ':$outcome_var' must have at least 2 unique levels."); end
                
                spec[:latent_dist] = get(spec, :latent_dist, :logistic)
                spec[:K] = K
                
                level_map = Dict(level => i for (i, level) in enumerate(unique_levels))
                M[:data][!, outcome_var] = [level_map[val] for val in outcome_data]
                @info "Ordinal outcome '$outcome_var' recoded to integers 1:$K."
            end
        end

        for out_sym in M[:outcomes]
            if !hasproperty(M[:data], out_sym)
                error("Outcome variable ':$out_sym' specified in the formula was not found as a column in the provided data frame. Please check for typos or ensure the column exists.")
            end
        end

        if M[:outcomes_N] > 1
            M[:model_arch] = "multivariate"
            M[:y_obs] = Matrix(M[:data][!, M[:outcomes]])
        else
            M[:model_arch] = get(M, :model_arch, "univariate")
            M[:y_obs] = M[:data][!, M[:outcomes][1]]
        end
    end

    # --- Resolve Observation-Level Parameters ---
    calling_mod = get(M, :calling_module, Main)
    merged_params = Dict{Symbol, Any}()
    for spec_params in M[:likelihood_specs]; merge!(merged_params, spec_params); end

    _resolve_obs_param!(M, merged_params, M[:data], [:log_offsets, :offsets], :log_offsets)
    _resolve_obs_param!(M, merged_params, M[:data], [:weights], :weights)
    _resolve_obs_param!(M, merged_params, M[:data], [:trials], :trials)

    scalar_param_keys = [:censor_lower, :censor_upper, :hurdle]
    for key in scalar_param_keys
        values_per_outcome = []
        any_provided = false
        for spec_params in M[:likelihood_specs]
            val = _resolve_outcome_scalar_param!(spec_params, key, calling_mod)
            if !isnothing(val); any_provided = true; end
            push!(values_per_outcome, val)
        end

        if any_provided
            default_val = if key == :censor_lower || key == :hurdle; -Inf; else Inf; end
            final_values = [isnothing(v) ? default_val : v for v in values_per_outcome]
            M[key] = M[:outcomes_N] == 1 ? final_values[1] : final_values
            M[Symbol("user_provided_", key)] = true
        end
    end

    _resolve_boolean_obs_param!(M, merged_params, :zero_inflated, :use_zi)
    if get(M, :user_provided_hurdle, false) && get(M, :use_zi, false)
        @warn "Both `hurdle` and `zero_inflated` were specified. The hurdle model will be used and zero-inflation will be ignored."
        M[:use_zi] = false
    end
    _resolve_boolean_obs_param!(M, merged_params, :volatility, :volatility)
end





function _resolve_outcome_scalar_param!(params::Dict, key::Symbol, calling_mod::Module)
    # Purpose: Resolves a likelihood parameter that must be a scalar value for a given outcome.
    # Rationale: This enforces that parameters like `censor_lower` cannot be specified as
    #            per-observation vectors from the data, only as single scalar values.
    # v1.0.0 (2026-07-18)
    # Inputs:
    #   - params: The likelihood parameters from the formula.
    #   - key: The symbol for the parameter (e.g., `:censor_lower`).
    #   - calling_mod: The module context for evaluating symbols.
    # Outputs: The resolved scalar value, or `nothing`.
    if !haskey(params, key); return nothing; end

    val = params[key]
    if val isa Number
        return val
    elseif val isa Symbol || val isa Expr
        try
            evaluated_val = Core.eval(calling_mod, val)
            if evaluated_val isa Number
                return evaluated_val
            else
                @warn "Parameter '$val' for '$key' must be a scalar number, but evaluated to type '$(typeof(evaluated_val))'. Ignoring."
                return nothing
            end
        catch
            @warn "Parameter '$val' for '$key' could not be evaluated as a scalar variable in the calling module. Ignoring."
            return nothing
        end
    else
        @warn "Parameter for '$key' has an unsupported type '$(typeof(val))'. It must be a scalar number or a variable that evaluates to one. Ignoring."
        return nothing
    end
end


function _resolve_obs_param!(opt_dict, params, data, param_keys, target_key)
    # Purpose: Finds an observation-level parameter (like offsets or weights) in the data and adds it to the config.
    # Rationale: This version is updated to correctly handle cases where the provided argument is a variable
    #            in the calling scope that itself contains a Symbol or String referring to a data column.
    #            This resolves a common failure point when using the macro non-interactively.
    # v1.0.1 (2026-07-27)
    # Assumptions: `data` is a DataFrame.
    # Inputs:
    #   - opt_dict: The configuration dictionary to update.
    #   - params: The likelihood parameters from the formula.
    #   - data: The input DataFrame.
    #   - param_keys: A list of possible keys for the parameter (e.g., [:log_offsets, :offsets]).
    #   - target_key: The key to use in `opt_dict`.
    # Outputs: None (mutates `opt_dict`).
    for key in param_keys
        if haskey(params, key)
            val = params[key]
            if val isa Symbol
                if hasproperty(data, val)
                    opt_dict[target_key] = data[!, val]
                    opt_dict[Symbol("user_provided_", target_key)] = true
                else
                    # The symbol might refer to a variable in the calling scope.
                    calling_mod = get(opt_dict, :calling_module, Main)
                    try
                        evaluated_val = Core.eval(calling_mod, val)
                        if evaluated_val isa Symbol && hasproperty(data, evaluated_val)
                            # Case: log_offsets = :my_col
                            opt_dict[target_key] = data[!, evaluated_val]
                            opt_dict[Symbol("user_provided_", target_key)] = true
                        elseif evaluated_val isa String && hasproperty(data, Symbol(evaluated_val))
                            # Case: log_offsets = "my_col"
                            opt_dict[target_key] = data[!, Symbol(evaluated_val)]
                            opt_dict[Symbol("user_provided_", target_key)] = true
                        elseif evaluated_val isa Number || evaluated_val isa AbstractVector
                            # Case: log_offsets = my_vector
                            opt_dict[target_key] = evaluated_val
                            opt_dict[Symbol("user_provided_", target_key)] = true
                        else
                            @warn "Parameter '$val' for '$target_key' evaluated to an unsupported type '$(typeof(evaluated_val))'. Ignoring."
                        end
                    catch
                        @warn "Parameter '$val' for '$target_key' is not a valid column name and could not be evaluated as a variable in the calling module '$(calling_mod)'. Ignoring."
                    end
                end
            elseif val isa Number || val isa AbstractVector
                opt_dict[target_key] = val # The value was parsed directly as a number/vector.
                opt_dict[Symbol("user_provided_", target_key)] = true
            else
                @warn "Observation parameter '$val' for '$target_key' is not a valid column name, vector, or scalar. Ignoring."
            end
            return
        end
    end
end

 
function _resolve_boolean_obs_param!(opt_dict, params, param_key, target_key) 
    # Purpose: Resolves a boolean flag from the likelihood parameters.
    # Rationale: Handles boolean flags like `zero_inflated=true`. This version is more robust,
    #            handling symbols that evaluate to booleans and issuing warnings for invalid types.
    # v1.0.0 (2026-07-19)
    # Inputs:
    #   - opt_dict: The configuration dictionary to update.
    #   - params: The likelihood parameters from the formula.
    #   - param_key: The key for the boolean flag.
    #   - target_key: The key to set in `opt_dict`.
    # Outputs: None (mutates `opt_dict`).
    if haskey(params, param_key)
        val = params[param_key]
        if val isa Bool
            opt_dict[target_key] = val
        elseif val isa Symbol || val isa Expr
            calling_mod = get(opt_dict, :calling_module, Main)
            try
                evaluated_val = Core.eval(calling_mod, val)
                if evaluated_val isa Bool
                    opt_dict[target_key] = evaluated_val
                else
                    @warn "Parameter '$val' for '$param_key' evaluated to a non-boolean type '$(typeof(evaluated_val))'. Ignoring."
                end
            catch
                @warn "Parameter '$val' for '$param_key' could not be evaluated as a boolean variable in the calling module. Ignoring."
            end
        else
            @warn "Parameter for '$param_key' has an unsupported type '$(typeof(val))'. It must be a boolean or a variable that evaluates to one. Ignoring."
        end
    end
end



function _process_fixed_effects_priors!(M::Dict) 
    # Purpose: Resolves and stores the prior distributions for each fixed effect coefficient.
    # Rationale: This version is updated to correctly evaluate `Expr` objects for priors,
    #            allowing users to specify distributions directly in the formula string.
    # v1.0.0 (2026-07-17)
    # Assumptions: `M[:Xfixed_applied_formula]` is populated by `_process_fixed_effects!`.
    # Inputs:
    #   - M: The model configuration dictionary.
    # Outputs: None (mutates `M`).
    n_fixed = get(M, :Xfixed_N, 0) 
    if n_fixed == 0
        M[:Xfixed_priors_vec] = UnivariateDistribution[]
        return
    end

    calling_mod = get(M, :calling_module, Main)
    custom_priors = M[:fixed_effects_priors]
    intercept_prior_val = get(M, :intercept_prior, nothing)
    default_prior = Normal(0, 5)
    priors_vec = Vector{Union{UnivariateDistribution, Nothing}}(undef, n_fixed)
    fill!(priors_vec, nothing)

    normalized_priors = Dict{String, Any}()
    for (key, prior) in custom_priors
        norm_key = replace(string(key), r"\s*[\*:]\s*" => "&")
        norm_key = replace(norm_key, r"\s*&\s*" => "&")
        
        if occursin("&", norm_key)
            parts = sort(Base.split(norm_key, '&'))
            norm_key = join(parts, '&')
        end
        normalized_priors[norm_key] = prior
    end

    applied_formula = get(M, :Xfixed_applied_formula, nothing)
    if isnothing(applied_formula)
        @warn "Could not find the applied formula for fixed effects. Prior assignment may be incomplete. This is an internal issue."
        M[:Xfixed_priors_vec] = fill(default_prior, n_fixed)
        return
    end

    all_coef_names = string.(coefnames(applied_formula.rhs))
    coef_name_to_idx = Dict(name => i for (i, name) in enumerate(all_coef_names))
    processed_indices = Set{Int}()

    for term in applied_formula.rhs.terms
        canonical_name = _canonical_term_string(term)
        
        if haskey(normalized_priors, canonical_name)
            prior_val = normalized_priors[canonical_name]
            local prior_obj
            if prior_val isa Expr
                try; prior_obj = Core.eval(calling_mod, prior_val);
                catch e; error("Could not evaluate `prior` argument `$(prior_val)` for fixed effect '$canonical_name'. Error: $e"); end
            else
                prior_obj = prior_val
            end

            term_coef_names = coefnames(term)
            
            term_coef_names_vec = term_coef_names isa AbstractString ? [term_coef_names] : term_coef_names

            for coef_name in term_coef_names_vec
                if haskey(coef_name_to_idx, coef_name)
                    idx = coef_name_to_idx[coef_name]
                    priors_vec[idx] = prior_obj isa Tuple ? create_pc_prior(Symbol(canonical_name), prior_obj) : prior_obj
                    push!(processed_indices, idx)
                else
                    @warn "Coefficient name '$coef_name' for term '$canonical_name' not found in the full coefficient list. Prior may not be applied."
                end
            end
        end
    end

    for i in 1:n_fixed
        if !(i in processed_indices)
            coef_name_str = all_coef_names[i]
            if coef_name_str == "(Intercept)"
                prior = intercept_prior_val
                priors_vec[i] = isnothing(prior) ? default_prior : (prior isa Tuple ? create_pc_prior(:intercept, prior) : prior)
            else
                priors_vec[i] = default_prior
            end
        end
    end

    M[:Xfixed_priors_vec] = convert(Vector{UnivariateDistribution}, priors_vec) 
end 


function _finalize_config!(M::Dict)
    # Purpose: Ensures the configuration dictionary has all necessary keys with default values.
    # Rationale: Prevents `KeyError` exceptions in the model assembler and execution.
    # Assumptions: `M` is a valid configuration dictionary.
    # Inputs:
    #   - M: The model configuration dictionary.
    # Outputs: None (mutates `M`).
    defaults = Dict(
        :s_N => 0, :t_N => 0, :u_N => 0,
        :s_idx => ones(Int, M[:y_N]),
        :t_idx => ones(Int, M[:y_N]),
        :u_idx => ones(Int, M[:y_N]),
        :log_offset => zeros(M[:y_N]),
        :weights => ones(M[:y_N]),
        :trials => ones(Int, M[:y_N]),
        :hyperpriors => Dict(),
        :prior_scheme => :pcpriors,
        :intercept_prior => Normal(0, 5)
    )
    for (key, val) in defaults
        if !haskey(M, key)
            M[key] = val
        end
    end
end


# ==============================================================================
# SECTION 5: MODULE-SPECIFIC PROCESSORS
# ==============================================================================

 

"""
    _replace_bstm_modules_in_expr(ex)

Recursively traverses a Julia expression and replaces `bstm`-specific modules
with their `StatsModels.jl` equivalents for parsing within other modules like `mixed()`.
- `intercept()` becomes `1`.
- `fixed(x)` becomes `x`.

# Arguments
- `ex`: A Julia expression, symbol, or literal.

# Returns
- The modified expression.
"""
function _replace_bstm_modules_in_expr(ex)
    if ex isa Expr && ex.head == :call
        if ex.args[1] == :intercept
            return 1
        elseif ex.args[1] == :fixed && length(ex.args) > 1
            return ex.args[2] # Return the variable inside fixed()
        end
        return Expr(ex.head, _replace_bstm_modules_in_expr.(ex.args)...)
    elseif ex isa Expr
        return Expr(ex.head, _replace_bstm_modules_in_expr.(ex.args)...)
    else
        return ex
    end
end


 

# ==============================================================================
# SECTION 6: MODEL BUILDING AND ASSEMBLY
# ==============================================================================

function _bstm_error_handler(e, model)
    println("\nERROR during prior predictive check (rand(m)):")
    showerror(stdout, e, stacktrace(catch_backtrace()))
    println("\n\n--- bstm Diagnosis ---")

    if e isa DimensionMismatch
        println("A `DimensionMismatch` error occurred. This often points to an issue in the model's structure.")
        println("Potential Causes:")
        println("  1. Latent Field vs. Index Mismatch: The number of latent variables for a spatial, temporal, or mixed effect does not match the number of unique levels in the corresponding index variable.")
        println("     - Check `s_N`, `t_N`, or the number of levels in your `mixed()` effect's grouping variable.")
        println("     - Ensure the data passed to `bstm()` is consistent with the dimensions inferred from the formula.")
        println("  2. Matrix Multiplication: An operation like `X * beta` has incompatible dimensions.")
        println("     - Verify the number of columns in your fixed effects design matrix `X` matches the length of the `beta` vector.")
    elseif e isa BoundsError
        println("A `BoundsError` occurred. This means an index is out of range for an array.")
        println("Potential Causes:")
        println("  - This is very common with `mixed()` or `spatial()` effects. The latent field vector is smaller than the maximum index in the corresponding index vector (e.g., `M.s_idx` or `M.mixed_idx_...`).")
        println("  - Review the configuration of the component that caused the error. The number of latent units (e.g., `n_cat` for a mixed effect, or `s_N` for a spatial effect) might be miscalculated.")
        println("  - Check for off-by-one errors in manual indexing if using a `custom()` component.")
    elseif e isa PosDefException
        println("A `PosDefException` occurred. This means a matrix that needs to be positive definite (e.g., for a Cholesky decomposition) is not.")
        println("Potential Causes:")
        println("  1. GMRF Precision Matrix: The precision matrix `Q` for a spatial or temporal model might be numerically unstable or not positive definite.")
        println("     - For `icar` or `besag` models, ensure your adjacency matrix `W` corresponds to a single connected graph. Disconnected spatial 'islands' will cause this error.")
        println("     - A small amount of diagonal jitter is added (`noise` parameter), but it might be insufficient. Try increasing the `noise` keyword argument in the `@bstm` call (e.g., `noise=1e-5`).")
        println("  2. GP Covariance Matrix: A kernel matrix `K` in a Gaussian Process model is not positive definite.")
        println("     - This can happen with very close data points. A small amount of 'nugget' or jitter is usually added to the diagonal. Check the `noise` parameter.")
    elseif e isa KeyError
        println("A `KeyError` occurred. The model tried to access a parameter in the configuration that does not exist.")
        println("Potential Causes:")
        println("  - A typo in a variable name within the formula string.")
        println("  - A required parameter (e.g., `W` for a spatial model, or a custom index like `s_idx`) was not passed as a keyword argument to the `@bstm` call.")
        println("  - An internal error where a parameter was not correctly propagated during model configuration. Check the generated model code for missing `M.` accesses.")
    else
        println("An unexpected error occurred. Here are some general debugging tips:")
        println("  - Carefully review the generated model code printed above the error for any obvious issues.")
        println("  - Use `show_model(m)` to inspect the full model configuration and ensure all parameters seem correct.")
        println("  - Simplify your model formula by removing components one by one to isolate the source of the error.")
    end
    println("------------------------")

    # Suggest simplified formulas to help the user debug.
    println("\n--- Suggested Debugging Steps ---")
    try
        formula_str = model.args.M.formula
        lhs, rhs_raw = split(formula_str, '~')
        lhs = Base.strip(lhs)

        # Use the same logic as the parser to split terms
        rhs_normalized = replace(Base.strip(rhs_raw), r"\s*-\s*" => " + -")
        all_terms = split_terms_at_depth(rhs_normalized, " + ")

        # Determine if the original model had an intercept
        has_intercept = !any(in.(Base.strip.(all_terms), (["0", "-1"],))) && !any(startswith.(Base.strip.(all_terms), "intercept(false"))

        # Suggest a base model
        base_rhs = has_intercept ? "1" : "0"
        println("1. Start with the simplest possible model to isolate the issue.")
        println("   This helps determine if the error is in your `likelihood()` definition or in the model components.")
        println("\n   Suggested base model:")
        println("   @bstm(\n       $lhs ~ $base_rhs,\n       data, ...\n   )")

        # Filter out intercept-related terms for the incremental build-up
        structural_terms = filter(t -> !in(Base.strip(t), ["1", "0", "-1"]) && !startswith(Base.strip(t), "intercept("), all_terms)

        if !isempty(structural_terms)
            println("\n2. If the base model works, add components back one by one to find the problematic term.")
            println("   For example, try the following formulas in order:")
            
            current_formula_rhs = base_rhs
            for (i, term) in enumerate(structural_terms)
                current_formula_rhs *= " + " * term
                println("\n   Step $i: Add '$term'")
                println("   @bstm(\n       $lhs ~ $current_formula_rhs,\n       data, ...\n   )")
            end
        end
    catch e_sugg
        println("\nCould not automatically generate debugging suggestions. Error: $e_sugg")
    end
    println("---------------------------------\n")
end

"""
    @bstm(exprs...)

The main user-facing macro for defining a `bstm` model. It supports two syntaxes:
1.  `m = @bstm(formula, data, ...)`: Returns the model object.
2.  `@bstm m = formula, data, ...`: Assigns the model to `m`.

# Rationale for Correction
The previous implementation used a simplistic method to parse arguments, which failed
when keyword arguments were passed out of order or with a semicolon, leading to a `MethodError`
or causing keyword arguments to be ignored. This updated version implements a more robust
parser that correctly distinguishes between positional arguments (like `formula` and `data`)
and keyword arguments (like `W=...`), regardless of their order. This resolves the `MethodError`
and the issue of `W` not being found, making the macro's behavior more predictable and
aligned with standard Julia syntax.
"""
macro bstm(exprs...)
    local var_name = nothing
    local formula_expr = nothing
    local data_expr = nothing
    local collected_kwargs = []
    local current_positional_args = []

    # Handle assignment syntax: `@bstm m = formula, data, ...`
    local expressions_to_parse = exprs
    if !isempty(exprs) && exprs[1] isa Expr && exprs[1].head == :(=)
        var_name = exprs[1].args[1]
        expressions_to_parse = (exprs[1].args[2], exprs[2:end]...)
    end

    # Iterate through all expressions to separate positional and keyword arguments
    for ex in expressions_to_parse
        if ex isa Expr && ex.head == :parameters # This is the block after a semicolon
            append!(collected_kwargs, ex.args)
        elseif ex isa Expr && ex.head == :(=) # This is a keyword argument without a semicolon
            # Convert Expr(:(=), key, value) to Expr(:kw, key, value) for consistency
            push!(collected_kwargs, Expr(:kw, ex.args[1], ex.args[2]))
        elseif ex isa Expr && ex.head == :kw # This is a keyword argument within a :parameters block
            push!(collected_kwargs, ex)
        else # Positional argument
            push!(current_positional_args, ex)
        end
    end

    # Extract formula and data from positional arguments
    if length(current_positional_args) < 2
        error("The @bstm macro requires at least a formula and a data frame, e.g., `@bstm(y ~ 1, my_data)`.")
    end
    formula_expr = current_positional_args[1]
    data_expr = current_positional_args[2]

    # Warn if there are unexpected extra positional arguments
    if length(current_positional_args) > 2
        @warn "Ignoring extra positional arguments: $(current_positional_args[3:end])"
    end

    # Convert the formula expression to a string
    formula_str = string(formula_expr)

    # Escape all expressions for interpolation
    data_esc = esc(data_expr)
    kwargs_esc = [esc(kw) for kw in collected_kwargs]

    # Construct the core function call
    core_logic = :(bstm($formula_str, $data_esc, $(__module__); $(kwargs_esc...)))

    # Return the appropriate expression
    if !isnothing(var_name)
        return :($(esc(var_name)) = $core_logic)
    else
        return core_logic
    end
end



function _print_param(name, value, status; indent=4)
    # Purpose: Helper function to print a single parameter with its value and status.
    # Rationale: Ensures consistent formatting and handles truncation of long values safely.
    # v1.0.1 (2026-07-31) - Fixed StringIndexError by using `first(str, N)` for safe truncation.
    indent_str = " " ^ indent
    status_str = status == :user ? "(User-provided)" : "(Default)"
    
    value_str = string(value)
    # Safely truncate long values for cleaner display, handling multi-byte characters.
    if length(value_str) > 70
        value_str = first(value_str, 67) * "..."
    end
    
    println("$indent_str- $(rpad(name, 20)): $(value_str)  $status_str")
end
 

# --- Updated function to print finalized parameters ---
"""
    _print_finalized_parameters(config::NamedTuple)

Prints a comprehensive and well-formatted summary of all finalized parameters for each
module used in the `bstm` model. This function is called when `verbose=true`.

It details the configuration for the likelihood, intercept, fixed effects, and all
random/smooth components. For each parameter, it shows the final value that will be
used in the model and indicates whether this value was explicitly provided by the user
or if it was assigned a system default. This provides clarity on the model's exact
specification before sampling begins.
"""

function _print_finalized_parameters(config::NamedTuple)
    println("\n--- Finalized Model Configuration ---")

    # 1. Likelihood Configuration
    println("\n[ Likelihood ]")
    lik_param_defs = [
        (:family, "gaussian"), (:log_offsets, "0.0"), (:weights, "1.0"),
        (:trials, "1"), (:zero_inflated, false), (:volatility, false),
        (:censor_lower, -Inf), (:censor_upper, Inf), (:hurdle, -Inf)
    ]
    
    # Add latent_dist for ordinal family
    push!(lik_param_defs, (:latent_dist, :logistic))

    for (i, spec) in enumerate(config.likelihood_specs)
        outcome = config.outcomes[i]
        println("  Outcome: $outcome")
        user_params = spec # `spec` itself is already the params dictionary
        
        for (p_name, p_default) in lik_param_defs
            final_val = get(user_params, p_name, p_default)
            status = haskey(user_params, p_name) ? :user : :default
            _print_param(p_name, final_val, status)
        end
    end

    # 2. Intercept Configuration
    println("\n[ Intercept ]")
    if config.add_intercept
        is_user_provided = haskey(config, :intercept_prior) && config.intercept_prior != Normal(0, 5)
        _print_param(:prior, config.intercept_prior, is_user_provided ? :user : :default, indent=2)
    else
        println("  - Intercept removed from model.")
    end

    # 3. Fixed Effects Configuration
    if get(config, :Xfixed_N, 0) > 0
        println("\n[ Fixed Effects ]")
        println("  Formula: ~ $(config.Xfixed_applied_formula.rhs)")
        
        if haskey(config, :contrasts) && !isempty(config.contrasts)
            println("  Contrasts:")
            for (var, cont) in config.contrasts
                println("    - $var: $(typeof(cont))")
            end
        end

        println("  Priors per Coefficient:")
        for (i, name) in enumerate(config.Xfixed_names)
            prior_obj = config.Xfixed_priors_vec[i]
            is_default = prior_obj == Normal(0, 5)
            _print_param(name, prior_obj, is_default ? :default : :user, indent=4)
        end
    end

    # 4. Model Components (random, smooth, etc.)
    if !isempty(config.components)
        println("\n[ Model Components ]")
        for spec in config.components
            println("  --- Component: $(spec.key) ---")
            component_obj = spec.component_obj
            model_type_sym = Symbol(lowercase(string(typeof(component_obj))))
            println("    - Type: $(typeof(component_obj))")
            
            latent_dim_val = 0
            # Corrected access: Q_template is inside spec.hyper
            if hasproperty(spec.hyper, :Q_template) && spec.hyper.Q_template isa AbstractMatrix
                latent_dim_val = size(spec.hyper.Q_template, 1)
            elseif hasproperty(component_obj, :nbins)
                latent_dim_val = component_obj.nbins
            elseif hasproperty(component_obj, :n_features)
                latent_dim_val = component_obj.n_features
            elseif hasproperty(component_obj, :n_inducing)
                latent_dim_val = component_obj.n_inducing
            elseif hasproperty(spec.hyper, :n_latent) # Fallback for components that explicitly define n_latent in hyper
                latent_dim_val = spec.hyper.n_latent
            end
            
            if latent_dim_val > 0
                println("    - Latent Field Dimension: $(latent_dim_val)")
            end

            println("    - Parameters:")
            user_provided_params_raw = spec.params 
            
            # Combine struct fields (priors) and config args
            all_param_names = Set(fieldnames(typeof(component_obj)))
            if haskey(COMPONENT_CONFIG_ARGS, model_type_sym)
                union!(all_param_names, keys(COMPONENT_CONFIG_ARGS[model_type_sym]))
            end

            for param_name in sort(collect(all_param_names))
                local final_val, status
                if param_name in fieldnames(typeof(component_obj))
                    # It's a hyperparameter (prior)
                    final_val = getfield(component_obj, param_name)
                    status = haskey(user_provided_params_raw, param_name) ? :user : :default
                else
                    # It's a configuration argument
                    final_val = get(user_provided_params_raw, param_name, COMPONENT_CONFIG_ARGS[model_type_sym][param_name])
                    status = haskey(user_provided_params_raw, param_name) ? :user : :default
                end
                
                if final_val isa Vector{<:UnivariateDistribution}
                    println("      - $(rpad(param_name, 20)): [")
                    for (idx, p_dist) in enumerate(final_val)
                        println("        $idx: $p_dist")
                    end
                    println("      ] $(status == :user ? "(User-provided)" : "(Default)")")
                else
                    _print_param(param_name, final_val, status, indent=6)
                end
            end
        end
    end
    println("\n-------------------------------------\n")
end

"""
    bstm(formula::String, data::DataFrame, calling_module::Module; kwargs...)

The main entry point for the `@bstm` macro. This function orchestrates the configuration,
code generation, and instantiation of a Turing model.

# Rationale
This function is the core of the `bstm` framework and replaces the deprecated
`bstm_dynamic_model` function. It provides a robust and complete workflow:
1.  **Configuration**: It calls `bstm_config` to translate the user's formula and
    data into a detailed model specification.
2.  **Code Generation**: It calls `bstm_codegen` to dynamically generate a unique
    Turing `@model` definition, preventing "world age" errors.
3.  **Scoped Evaluation**: It evaluates the generated model in the correct `calling_module`,
    ensuring it is accessible in the user's scope.
4.  **Safe Instantiation**: It uses `Base.invokelatest` to instantiate the model,
    which is necessary for dynamically generated functions.
5.  **Validation**: It automatically runs a prior predictive check (`rand(model)`) and
    provides detailed, user-friendly diagnostics if the check fails.

This function is not deprecated and is the correct, unified entry point for all
`bstm` models.
"""
function bstm(formula::String, data::DataFrame, calling_module::Module; kwargs...)
    # Generate model configuration dictionary based on formula syntax and data schema
    options = bstm_config(formula, data; calling_module = calling_module, kwargs...)

    # Invoke the codegen engine to produce the model source string and expression
    model_func_name, expr, new_config, registry = bstm_codegen(options)

    if get(new_config, :verbose, true)
        println("\n--- Dynamically Generated Model Code ---")
        println(new_config.generated_model_code)
        println("----------------------------------------\n")
    end

    # Evaluate the generated @model macro expression in the target module scope
    calling_module.eval(expr)

    # Access the function binding from the module's global scope
    model_func = getfield(calling_module, model_func_name)

    # Instantiation of the Turing Model Object
    model_instance = Base.invokelatest(model_func, new_config, registry)

    if get(new_config, :verbose, true)
        println("\n--- Running prior predictive check ---")
    end

    prior_sample = nothing
    try
        # Prior Predictive Validation
        redirect_stderr(devnull) do
            prior_sample = Base.invokelatest(rand, model_instance)
        end
    
        if get(new_config, :verbose, true) && !isnothing(prior_sample)
            println("Prior sample check successful. Sample values:")
            display(prior_sample)
        end
    catch e 
        # Error handling for structural or parameterization issues
        _bstm_error_handler(e, model_instance)
    end

    if get(new_config, :verbose, true)
        println("--------------------------------------\n")
    end

    # Return the fully configured and validated model object
    return model_instance
end



"""
    bstm(formula::String, data::DataFrame; kwargs...)

A convenience overload for the main `bstm` constructor. This method defaults the
model's evaluation scope to the `Main` module.

# Rationale
This function is not deprecated and serves as a useful entry point for interactive
use, where the user is typically working in the `Main` module. It simplifies the
function call by removing the need to explicitly provide the module context.

For use within packages or other modules, the `@bstm` macro is the recommended
entry point, as it automatically captures the correct calling module (`__module__`),
ensuring that the dynamically generated model code is evaluated in the correct scope.

# Arguments
- `formula::String`: The model formula.
- `data::DataFrame`: The input data.
- `kwargs...`: Additional keyword arguments passed to the main `bstm` constructor.

# Returns
- An instantiated Turing model object.
"""
function bstm(formula::String, data::DataFrame; kwargs...)
    # This overload defaults the calling_module to `Main`, which is a sensible
    # default for interactive use. The `@bstm` macro should be preferred for
    # use inside other modules to ensure correct scope resolution.
    return bstm(formula, data, Main; kwargs...)
end

"""
    bstm_text_assembler(config::NamedTuple, model_func_name::Symbol)

Assembles the full Turing model code as a string and a Julia `Expr`.

This version corrects a type-instability issue that caused errors with
Automatic Differentiation (AD) in samplers like NUTS. The linear predictor `eta`
is now initialized using the type of the `intercept` parameter. This ensures that
`eta` is allocated with the correct `Dual` number type during AD, preventing
`MethodError` when `Dual`-valued latent effects are added to it.
"""
function bstm_text_assembler(config::NamedTuple, model_func_name::Symbol)
    priors_acc = String[]
    updates_acc = String[]
    
    is_multivariate = config.model_arch == "multivariate"
    arch_str = config.model_arch

    # --- 1. Generate all prior definitions ---
    push!(priors_acc, _generate_likelihood_section(config, is_multivariate))
    
    intercept_priors, intercept_update = _generate_intercept_block(config, is_multivariate, is_multivariate ? "eta_latent" : "eta")
    push!(priors_acc, intercept_priors)
    
    fixed_effects_priors, fixed_effects_update = _generate_fixed_effects_block(config, is_multivariate, is_multivariate ? "eta_latent" : "eta")
    push!(priors_acc, fixed_effects_priors)

    # --- 2. Assemble the linear predictor updates ---
    # NOTE: The intercept update is now handled in the initialization block below, not here.
    push!(updates_acc, _generate_offset_block(config, is_multivariate, is_multivariate ? "eta_latent" : "eta"))
    push!(updates_acc, fixed_effects_update)

    # --- 3. Main Component Loop ---
    for spec in config.components
        m_obj = spec.component_obj
        if m_obj isa None; continue; end

        push!(priors_acc, "\n# --- Priors for component: $(spec.key) ---")
        push!(updates_acc, "\n# --- Updates for component: $(spec.key) ---")

        if is_multivariate
            for k in 1:config.outcomes_N
                push!(priors_acc, get_priors(m_obj, spec, arch_str, k, config))
                push!(updates_acc, get_updates(m_obj, spec, arch_str, k, config))
            end
        else
            push!(priors_acc, get_priors(m_obj, spec, arch_str, nothing, config))
            push!(updates_acc, get_updates(m_obj, spec, arch_str, nothing, config))
        end
    end

    # --- 4. Assemble Final Code Blocks ---
    priors_code = join(filter(!isempty, priors_acc), "\n    ")
    updates_code = join(filter(!isempty, updates_acc), "\n")
    
    final_likelihood_code = _generate_final_likelihood_block(config, is_multivariate)

    # --- 5. Construct the final model string ---
    eta_name = is_multivariate ? "eta_latent" : "eta"
    
    # AD-safe initialization block for eta
    eta_initialization_block = if get(config, :add_intercept, false)
        if is_multivariate
            """
            # Initialize eta_latent with the intercept to get the correct AD type
            local $(eta_name) = repeat(intercept', M.y_N, 1)
            """
        else
             """
            # Initialize eta with the intercept to get the correct AD type
            local $(eta_name) = fill(intercept, M.y_N)
            """
        end
    else
        # Fallback for models without an intercept. This may fail AD if the first
        # component added to eta is a Dual number and not handled with promotion.
        if is_multivariate
            "local $(eta_name) = zeros(T, M.y_N, M.outcomes_N)"
        else
            "local $(eta_name) = zeros(T, M.y_N)"
        end
    end

    model_string = """
    @model function $(model_func_name)(M, spec_registry; T=Float64)
        # --- Priors ---
        $(priors_code)

        # --- Model Definition ---
        # --- Linear Predictor Assembly ---
        $(eta_initialization_block)
        
        $(updates_code)
        
        # --- Likelihood ---
        if !get(M, :likelihood_handled, false)
            $(final_likelihood_code)
        end
    end
    """

    # --- 6. Create Spec Registry and Expression ---
    spec_registry = NamedTuple(spec.key => spec for spec in config.components)
    expr = Meta.parse(model_string)

    return model_string, expr, spec_registry
end






function bstm_codegen(config::NamedTuple)
    # Purpose: Generates the necessary components to define and instantiate a Turing model.
    # Rationale: Decouples code generation from evaluation to better handle Julia's world-age issues.
    # v1.0.0 (2026-07-18)
    # Inputs:
    #   - config: The model configuration NamedTuple.
    # Outputs: A tuple containing the model function name, the model definition expression,
    #          the updated configuration, and the specification registry.
    # Generate a unique name for the model function to avoid world age issues
    # when interactively redefining models.
    random_suffix = rand(10000:99999)
    model_func_name = Symbol("bstm_dynamic_model_$(random_suffix)")

    model_string, expr, registry = bstm_text_assembler(config, model_func_name)

    config_dict = Dict(pairs(config))
    config_dict[:generated_model_code] = model_string
    new_config = NamedTuple(config_dict)

    return model_func_name, expr, new_config, registry
end


"""
    resolve_technical_primitive(module_metadata::Dict{Symbol, Any}, M, priors_dict, scheme::Symbol)

Instantiates a `ComponentModel` object from its parsed formula representation. This function
acts as a factory, resolving hyperpriors and calling the appropriate constructor from the
`COMPONENT_CONSTRUCTORS` registry.
"""
function resolve_technical_primitive(module_metadata::Dict{Symbol, Any}, M, priors_dict, scheme::Symbol)
    m_type = module_metadata[:type]
    m_params = module_metadata[:params]
    calling_mod = get(M, :calling_module, Main)

    # Handle composed components recursively
    if m_type == :interact
        op = m_params[:operator]
        components_data = m_params[:components]
        components_metadata = map(c_node -> Dict(:key=>"temp", :type => c_node.module_type, :params => c_node.args, :variables => get(c_node.args, :positional_args, [])), components_data)
        resolved_components = [resolve_technical_primitive(comp_meta, M, priors_dict, scheme) for comp_meta in components_metadata]
        return Composed(resolved_components, op)
    end

    # Handle standard components
    local model_name
    if haskey(m_params, :model)
        model_name = m_params[:model]
    else
        # Infer default model based on the structure if no model is specified.
        model_name = if m_type == :spatial; haskey(M, :W) ? :bym2 : :iid; elseif m_type == :temporal; :rw2; else :iid; end
    end

    model_name_str = string(model_name)
    resolved_priors = resolve_hyperpriors(model_name_str, priors_dict, m_params, scheme, calling_mod)
    
    if !haskey(COMPONENT_CONSTRUCTORS, model_name)
        error("Component model ':$model_name' is not a recognized model type.")
    end
    
    constructor_func = COMPONENT_CONSTRUCTORS[model_name]
    return constructor_func(resolved_priors, m_params)
end



# Version 1.5.3 (2026-08-06)
# Purpose: Creates a precision matrix template and its spectral decomposition for a GMRF model.
# Rationale: This version is updated to pre-compute and return the eigendecomposition (eigenvectors `U`
#            and eigenvalues `L`) of the precision matrix template. This enables the use of AD-friendly
#            spectral sampling methods for latent Gaussian fields, as recommended for improving HMC
#            performance. Instead of sampling from `MvNormalCanon(0, Q)` where `Q` might depend on
#            sampled parameters (requiring a non-differentiable Cholesky decomposition), the model can
#            sample `z ~ N(0,I)` and construct the latent field as `latent = U * D * z`, where `D` is a
#            diagonal matrix constructed from the eigenvalues `L` and other hyperparameters. This avoids
#            all matrix decompositions within the model's execution path.
# Assumptions: The input `model_type` is a recognized GMRF structure.
function build_structure_template(model_type::Symbol, n::Int; W::Union{AbstractMatrix, Nothing}=nothing)
    # This function creates a standardized precision matrix template (Q) and its
    # eigendecomposition for various Gaussian Markov Random Field (GMRF) models.

    Q_template = spzeros(Float64, n, n)
    rank_deficiency = 0

    if n == 0
        return (matrix=Q_template, scaling_factor=1.0, U=spzeros(Float64, 0, 0), L=Float64[])
    end

    if model_type in [:icar, :besag, :bym2, :leroux, :localadaptive]
        if isnothing(W); error("Spatial model '$model_type' requires an adjacency matrix `W`."); end
        if size(W, 1) != n || size(W, 2) != n; error("Adjacency matrix `W` dimensions ($(size(W))) do not match `n` ($n)."); end
        W_sym = sparse((W + W') .> 0)
        D = spdiagm(0 => vec(sum(W_sym, dims=2)))
        Q_template = D - W_sym
        rank_deficiency = 1
    elseif model_type == :rw1
        if n > 1
            Q_template = spdiagm(0 => fill(2.0, n), -1 => fill(-1.0, n-1), 1 => fill(-1.0, n-1))
            Q_template[1, 1] = 1.0
            Q_template[n, n] = 1.0
        elseif n == 1
            Q_template[1,1] = 1.0
        end
        rank_deficiency = 1
    elseif model_type == :rw2
        # Implementation for RW2 precision matrix
        if n > 1
            Q_template = spdiagm(0 => fill(6.0, n), -1 => fill(-4.0, n-1), 1 => fill(-4.0, n-1), -2 => fill(1.0, n-2), 2 => fill(1.0, n-2))
            Q_template[1, 1] = 1.0; Q_template[2, 2] = 5.0
            Q_template[1, 2] = -2.0; Q_template[2, 1] = -2.0
            Q_template[n-1, n-1] = 5.0; Q_template[n, n] = 1.0
            Q_template[n-1, n] = -2.0; Q_template[n, n-1] = -2.0
        elseif n == 1
            Q_template[1,1] = 1.0
        end
        rank_deficiency = 2
    elseif model_type == :cyclic
        if n > 0
            Q_template = spdiagm(0 => fill(2.0, n), 1 => fill(-1.0, n-1), -1 => fill(-1.0, n-1))
            Q_template[1, n] = -1.0
            Q_template[n, 1] = -1.0
        end
        rank_deficiency = 1
    elseif model_type == :ar1
        if n > 1
            Q_template = spdiagm(0 => zeros(n), -1 => fill(-1.0, n-1), 1 => fill(-1.0, n-1))
        end
        rank_deficiency = 0
    elseif model_type == :iid
        Q_template = sparse(I, n, n)
        rank_deficiency = 0
    else
        @warn "Unknown model type '$model_type'. Returning identity matrix as template."
        Q_template = sparse(I, n, n)
        rank_deficiency = 0
    end

    local U, L, scaling_factor
    # Always compute eigendecomposition for potential use by spectral methods.
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values

    if rank_deficiency > 0
        scaling_factor = _compute_scaling_factor(L, rank_deficiency)
        Q_template = Q_template ./ scaling_factor
        # Rescale eigenvalues as well for consistency.
        L = L ./ scaling_factor
    else
        scaling_factor = 1.0
    end

    return (matrix=Q_template, scaling_factor=scaling_factor, U=U, L=L)
end



# Version 1.0.0 (2026-07-21)
# Purpose: Computes a robust scaling factor for a precision matrix from its eigenvalues.
# Rationale: The scaling factor is the geometric mean of the non-zero eigenvalues.
#            This method avoids using a fixed tolerance to identify zero eigenvalues,
#            which can be sensitive to floating-point noise. Instead, it uses the known
#            rank deficiency of the GMRF model to correctly identify the structural zero
#            eigenvalues.
# Assumptions: `evals` is a vector of eigenvalues from a symmetric matrix.
function _compute_scaling_factor(evals::Vector{Float64}, rank_deficiency::Int)
    # Sort eigenvalues in ascending order to easily discard the smallest ones, which correspond to the null space.
    sorted_evals = sort(evals)
    
    n = length(sorted_evals)
    if n <= rank_deficiency
        return 1.0
    end
    
    # Select the eigenvalues that are not part of the null space.
    positive_evals = sorted_evals[(rank_deficiency + 1):end]
    
    if isempty(positive_evals)
        return 1.0
    end
    
    # The scaling factor is the geometric mean of the positive eigenvalues.
    # This is a standard method for ensuring the determinant of the scaled precision matrix is 1.
    return exp(mean(log.(positive_evals)))
end





# Version 1.5.4 (2026-08-06)
# Purpose: Computes the cross-covariance kernel matrix between two sets of coordinates.
# Rationale: This version is updated for improved type stability and AD-friendliness,
#            mirroring the changes in `evaluate_kernel_matrix`. It avoids explicit
#            casts by using `convert` and relying on Julia's promotion system, which is
#            safer for automatic differentiation.
# Assumptions: `coords1` and `coords2` are matrices of data points.
function evaluate_cross_kernel_matrix(coords1::AbstractMatrix, coords2::AbstractMatrix, param_val::Real, ls::Union{Real, AbstractVector}, kernel_type::Symbol)
    T = promote_type(eltype(coords1), eltype(coords2), typeof(param_val), eltype(ls))
    coords1_T = convert(AbstractMatrix{T}, coords1)
    coords2_T = convert(AbstractMatrix{T}, coords2)
    ls_T = convert(typeof(ls) <: Real ? T : AbstractVector{T}, ls)

    local dist_sq
    if ls isa AbstractVector # ARD case
        if size(coords1_T, 2) != length(ls_T) || size(coords2_T, 2) != length(ls_T)
            error("Dimension mismatch for ARD kernel: Number of coordinate dimensions does not match number of lengthscales.")
        end
        # Calculate weighted squared Euclidean distance
        dist_sq = pairwise(SqEuclidean(), coords1_T ./ ls_T', coords2_T ./ ls_T', dims=1)
    else # Isotropic case
        dist_sq = pairwise(SqEuclidean(), coords1_T, coords2_T, dims=1) ./ ls_T^2
    end

    # Gaussian / Squared Exponential
    if kernel_type == :gaussian || kernel_type == :se
        return param_val^2 .* exp.(-one(T)/2 .* dist_sq)
    
    # Exponential / Matern 1/2
    elseif kernel_type == :exponential || kernel_type == :matern12
        d = sqrt.(dist_sq)
        return param_val^2 .* exp.(-d)
    
    # Matern 3/2
    elseif kernel_type == :matern32
        d = sqrt.(dist_sq)
        val = sqrt(convert(T, 3.0)) .* d
        return param_val^2 .* (one(T) .+ val) .* exp.(-val)
    
    # Matern 5/2
    elseif kernel_type == :matern52
        d = sqrt.(dist_sq)
        val = sqrt(convert(T, 5.0)) .* d
        return param_val^2 .* (one(T) .+ val .+ (val.^2 ./ convert(T, 3.0))) .* exp.(-val)

    # Fallback Dispatch
    else
        return param_val^2 .* exp.(-one(T)/2 .* dist_sq)
    end
end




# ==============================================================================
# SECTION 7: DYNAMIC MODEL ASSEMBLER
# ==============================================================================

function observation_volatility(M::NamedTuple)
    # Purpose: Generates code fragments for the observation error variance.
    # Rationale: Handles both constant variance and a spatiotemporal stochastic volatility (SV) model
    # v1.0.0 (2026-07-16)
    #            using Random Fourier Features (RFF), activated by a flag in the model configuration.
    # Inputs:
    #   - M: The model configuration NamedTuple.
    # Outputs: A NamedTuple with code strings for priors and calculations.

    if get(M, :volatility, false)
        # Stochastic Volatility Model using RFF
        required_keys = [:M_rff_sigma, :W_sigma_fixed, :b_sigma_fixed, :coords_st]
        if !all(k -> haskey(M, k), required_keys)
            error("Stochastic volatility is enabled, but one or more required keys are missing from the model configuration: $required_keys. Ensure 'volatility=true' is set in the likelihood() module and necessary data is provided.")
        end

        priors_str = """
        sigma_log_var ~ NamedDist(Exponential(1.0), :sigma_log_var)
        beta_vol ~ NamedDist(MvNormal(zeros(T, M.M_rff_sigma), sigma_log_var^2 * I), :beta_vol)
        """

        # The calculation projects spatiotemporal coordinates through the RFF basis
        # to generate a latent log-variance field, which is then transformed to
        # the standard deviation `y_sigma`.
        calc_str = """
        local vol_proj = (M.coords_st * M.W_sigma_fixed) .+ M.b_sigma_fixed'
        local log_var_latent = sqrt(2.0 / M.M_rff_sigma) .* cos.(vol_proj) * beta_vol
        y_sigma = exp.(log_var_latent ./ 2.0)
        """
    else
        # Default behavior: constant observation variance.
        # The y_sigma_const prior is defined in the assembler. This just uses it.
        priors_str = ""
        calc_str = "y_sigma = fill(y_sigma_const, N)"
    end

    return (priors=priors_str, calculation=calc_str)
end

 
 
# ==============================================================================
# SECTION 9: UTILITY AND HELPER FUNCTIONS
# ==============================================================================
### Version 1.9.8 - 2025-05-22 17:00:00
### Technical Descriptor: Consolidated Inducing Point Selection with KDTree Optimization

function generate_inducing_points(coords::AbstractMatrix, n_inducing::Int; method::String="kmeans", seed::Int=42)
    # Purpose: Selects a representative subset of coordinates to serve as inducing points for sparse GPs.
    # Rationale: Inducing points must be actual or representative locations in the input space. 
    # Systematic methods (quantile/regular) generate ideal targets that are then mapped to the 
    # nearest available data points using an efficient KDTree search to ensure spatial fidelity.

    n_obs, n_dims = size(coords)

    if n_inducing >= n_obs
        return coords
    end

    Random.seed!(seed)

    if method == "random"
        # Simple stochastic selection
        selected_idx = StatsBase.sample(1:n_obs, n_inducing, replace=false)
        return coords[selected_idx, :]

    elseif method == "kmeans"
        # Centroid-based selection via Clustering.jl
        # kmeans expects observations in columns: [dims x obs]
        kmeans_res = Clustering.kmeans(coords', n_inducing; maxiter=200, display=:none)
        return kmeans_res.centers'

    elseif method == "quantile" || method == "regular"
        # Systematic mapping methods requiring KDTree for efficiency
        
        target_pts = zeros(Float64, n_inducing, n_dims)
        
        if method == "quantile"
            # Density-aware target generation using marginal quantiles
            probs = range(0.0, stop=1.0, length=n_inducing)
            for d in 1:n_dims
                target_pts[:, d] = Statistics.quantile(coords[:, d], probs)
            end
        else 
            # method == "regular"
            # Grid-like target generation across marginal ranges
            for d in 1:n_dims
                v_min, v_max = extrema(coords[:, d])
                target_pts[:, d] = range(v_min, stop=v_max, length=n_inducing)
            end
        end

        # Efficient Nearest Neighbor Search
        # Build KDTree from the data points
        tree = KDTree(coords')
        
        # Find the single nearest observation for each target coordinate
        # knn returns (indices, distances)
        nn_indices_vec, _ = knn(tree, target_pts', 1, true)
        
        # Extract the scalar index from each neighbor search result and deduplicate
        unique_nn_indices = unique([idx_list[1] for idx_list in nn_indices_vec])
        
        return coords[unique_nn_indices, :]

    else
        @warn "Inducing point method '$method' not recognized. Falling back to random selection."
        selected_idx = StatsBase.sample(1:n_obs, n_inducing, replace=false)
        return coords[selected_idx, :]
    end
end
 


function _stable_logsubexp(a::Real, b::Real)
    # Purpose: Numerically stable computation of log(exp(a) - exp(b)).
    # Rationale: Avoids overflow and underflow by factoring out the larger term.
    # v1.0.0 (2026-07-16)
    #            This is equivalent to LogExpFunctions.logsubexp.
    # Inputs:
    #   - a, b: Real numbers.
    # Outputs: log(exp(a) - exp(b)).
    if a <= b
        return -Inf
    end
    return a + log1mexp(b - a)
end

 


function create_pc_prior(param_name::Symbol, constraint::Tuple)
    # Purpose: Creates a Penalized Complexity (PC) prior distribution from a user-specified quantile constraint.
    # Rationale: Translates an intuitive belief (e.g., "P(sigma > 1.0) = 0.05") into a formal prior distribution.
    #            This version is updated to use the base parameter name directly, as the `_prior` suffix
    #            is no longer part of the API for component structs.
    # v1.0.0 (2026-07-20)
    # Assumptions: `param_name` is one of the recognized types (:sigma, :rho, etc.).
    # Inputs:
    #   - param_name: The base name of the parameter.
    #   - constraint: A tuple `(U, α)` or `(U, α, direction)`.
    # Outputs: A `Distribution` object.

    
    direction = :upper
    if length(constraint) == 2; U, α = constraint; elseif length(constraint) == 3; U, α, direction = constraint; else; error("PC prior constraint must be a tuple of (U, α) or (U, α, direction)."); end
    
    if param_name == :sigma || endswith(string(param_name), "_sigma")
        direction != :upper && error("PC prior for sigma only supports upper tail constraints.")
        λ = -log(α) / U
        return Exponential(λ)
    elseif param_name == :rho || endswith(string(param_name), "_rho")
        direction != :upper && error("PC prior for 'rho' only supports upper tail constraints.")
        λ = log(α) / log(1.0 - U)
        return Exponential(λ)
    elseif param_name == :lengthscale || endswith(string(param_name), "_lengthscale")
        direction != :lower && error("PC prior for 'lengthscale' only supports lower tail constraints.")
        λ = -U * log(α)
        return Exponential(λ)
    elseif param_name == :kappa || endswith(string(param_name), "_kappa")
        direction != :upper && error("PC prior for kappa only supports upper tail constraints.")
        λ = -log(α) / U
        return Exponential(λ)
    else
        sigma = -U / quantile(Normal(0, 1), α / 2)
        return Normal(0, sigma)
    end
end


# Version 1.2.0 (2026-08-06)
# Purpose: Automatically constructs an efficient composite Gibbs sampler for a `bstm` model.
# Rationale: This version implements a sophisticated block-Gibbs strategy. It groups component-specific
#            parameters (e.g., a spatial field and its hyperparameters) into a single block for joint
#            sampling with NUTS, which is crucial for handling the "funnel" geometry in hierarchical models.
#            For remaining parameters, it assigns specialized samplers based on their prior's support:
#            ESS for Gaussian, Slice for bounded, and PG for discrete parameters. This hybrid approach
#            significantly improves sampling efficiency and convergence over a single, general-purpose sampler.
#            It also adds an `adtype` keyword to allow user control over the AD backend for NUTS.
function get_optimal_sampler(
    model_obj::DynamicPPL.Model;
    sampler_choice=:auto,
    sampler_map::Dict{Symbol, <:AbstractMCMC.AbstractSampler}=Dict{Symbol, AbstractMCMC.AbstractSampler}(),
    adtype::ADTypes.AbstractADType=ADTypes.AutoForwardDiff(),
    target_acceptance=0.8,
    adaptation_steps=1000,
    group_components::Bool=true,
    n_particles=20,
    hmc_leapfrog_steps=10
)
    # This function constructs a composite Gibbs sampler by assigning optimal MCMC algorithms
    # to different blocks of parameters based on their characteristics. The process follows a
    # clear hierarchy to provide both flexibility and efficiency.
    #
    # Sampler Selection Workflow:
    # 1. Manual Override: If a specific sampler is provided via `sampler_choice`, it is used directly.
    #    If a `sampler_map` is provided, it assigns specific samplers to designated parameters,
    #    taking the highest precedence.
    #
    # 2. Component Grouping: If `group_components=true`, the function identifies all parameters
    #    belonging to the same model component (e.g., a spatial field and its hyperparameters).
    #    These are grouped into a single block and assigned a NUTS sampler. This is crucial for
    #    efficiently exploring the correlated posterior geometry of hierarchical components.
    #
    # 3. Default Categorization: Any remaining parameters are categorized by their prior's support:
    #    - `:discrete` (e.g., from Categorical priors) are assigned Particle Gibbs (PG).
    #    - `:gaussian` (from Normal or MvNormal priors) are assigned Elliptical Slice Sampling (ESS).
    #    - `:bounded` (from priors like Beta, Exponential) are assigned Slice sampling.
    #    - `:other_continuous` (unbounded, non-Gaussian) are assigned NUTS.
    #
    # This strategy balances the use of efficient gradient-based samplers (NUTS) for complex,
    # high-dimensional blocks with robust gradient-free samplers for simpler or constrained parameters.

    if sampler_choice isa AbstractMCMC.AbstractSampler
        @info "Using user-specified sampler: $(typeof(sampler_choice))"
        return sampler_choice
    end

    vi = DynamicPPL.VarInfo(model_obj)
    vns = DynamicPPL.keys(vi)

    sampler_assignments = []
    all_processed_vns = Set{VarName}()

    # --- Stage 1: Handle user-provided sampler map (highest precedence) ---
    for (param_sym, sampler) in sampler_map
        sym_vns = filter(vn -> DynamicPPL.getsym(vn) == param_sym, vns)
        if !isempty(sym_vns)
            push!(sampler_assignments, Tuple(sym_vns) => sampler)
            union!(all_processed_vns, sym_vns)
            @info "Applying user-defined sampler $(typeof(sampler)) for parameter group: $(param_sym)"
        else
            @warn "Parameter :$(param_sym) in sampler_map not found in model."
        end
    end

    # --- Stage 2: Group parameters by model component if enabled ---
    if group_components
        @info "Component grouping enabled. Grouping hyperparameters and latent fields for joint sampling."
        component_groups = Dict{String, Set{VarName}}()
        
        # This list contains the standard prefixes for parameters generated by `bstm`.
        # It is used to parse variable names like `sigma_spatial_main` into a prefix (`sigma`)
        # and a component key (`spatial_main`).
        all_prefixes = [
            "sigma", "rho", "rho1", "rho2", "kappa", "ls", "range", "period",
            "amplitude", "phase", "velocity", "diffusion", "pca_sd", "pdef_sd",
            "L_corr", "sigma_effects", "r", "K", "q", "M_nat", "alpha", "beta", "gamma", "delta",
            "raw", "innov", "latent", "struct", "iid", "beta_cos", "beta_sin", "rho_field",
            "W", "b", "v_raw", "factors_flat", "thresh_raw", "W1", "b1", "W2", "amplitude_raw",
            "innov_predator"
        ]
        prefix_regex = Regex("^(" * join(all_prefixes, "|") * ")_(.+)\$")

        for vn in vns
            if vn in all_processed_vns; continue; end

            vn_str = string(DynamicPPL.getsym(vn))
            m = match(prefix_regex, vn_str)
            
            if !isnothing(m)
                component_key = m.captures[2]
                if !haskey(component_groups, component_key)
                    component_groups[component_key] = Set{VarName}()
                end
                push!(component_groups[component_key], vn)
            end
        end

        # Assign a NUTS sampler to each identified component group.
        for (key, params_vns) in component_groups
            if !isempty(params_vns)
                sampler = NUTS(adaptation_steps, target_acceptance; adtype=adtype)
                push!(sampler_assignments, Tuple(params_vns) => sampler)
                union!(all_processed_vns, params_vns)
                param_syms = Set(DynamicPPL.getsym.(params_vns))
                @info "Created NUTS block for component '$(key)' with parameters: $(param_syms)"
            end
        end
    end

    # --- Stage 3: Assign samplers to remaining parameters based on their prior's support ---
    remaining_vns = filter(vn -> !(vn in all_processed_vns), vns)
    if !isempty(remaining_vns)
        param_groups = Dict(:discrete => Set{VarName}(), :gaussian => Set{VarName}(), :bounded => Set{VarName}(), :other_continuous => Set{VarName}())

        for vn in remaining_vns
            try
                dist = DynamicPPL.getdist(vi, vn)
                support = Distributions.value_support(typeof(dist))
                if support isa Distributions.Discrete
                    push!(param_groups[:discrete], vn)
                elseif support isa Distributions.Continuous
                    if dist isa Union{Normal, MvNormal, Truncated{<:Normal}}
                        push!(param_groups[:gaussian], vn)
                    elseif isfinite(minimum(dist)) || isfinite(maximum(dist))
                        push!(param_groups[:bounded], vn)
                    else
                        push!(param_groups[:other_continuous], vn)
                    end
                end
            catch e
                # If we can't determine the distribution, default to the 'other' category.
                push!(param_groups[:other_continuous], vn)
            end
        end

        # Assign appropriate samplers to each category.
        if !isempty(param_groups[:discrete]); params = Tuple(param_groups[:discrete]); push!(sampler_assignments, params => PG(n_particles)); @info "Using Particle Gibbs (PG) for: $(DynamicPPL.getsym.(params))"; end
        if !isempty(param_groups[:gaussian]); params = Tuple(param_groups[:gaussian]); push!(sampler_assignments, params => ESS()); @info "Using Elliptical Slice Sampling (ESS) for: $(DynamicPPL.getsym.(params))"; end
        if !isempty(param_groups[:bounded]); params = Tuple(param_groups[:bounded]); push!(sampler_assignments, params => Slice()); @info "Using Slice sampling for: $(DynamicPPL.getsym.(params))"; end
        if !isempty(param_groups[:other_continuous]); params = Tuple(param_groups[:other_continuous]); push!(sampler_assignments, params => NUTS(adaptation_steps, target_acceptance; adtype=adtype)); @info "Using NUTS for remaining continuous parameters: $(DynamicPPL.getsym.(params))"; end
    end

    # --- Stage 4: Construct and return the final composite sampler ---
    if isempty(sampler_assignments)
        @warn "Could not identify any parameters to sample. Defaulting to a single NUTS sampler for all parameters."
        return NUTS(adaptation_steps, target_acceptance; adtype=adtype)
    elseif length(sampler_assignments) == 1
        # If only one group was formed, return that sampler directly, not wrapped in Gibbs.
        return sampler_assignments[1][2]
    else
        return Gibbs(sampler_assignments...)
    end
end




function create_fixed_design(
    formula_rhs::AbstractString, 
    data::DataFrame, 
    calling_module::Module; 
    contrasts=Dict{Symbol, Any}()
)
    # Purpose: Creates the fixed-effects design matrix (`X`) from a formula string.
    # Rationale: This version prevents the full DataFrame from being printed to the console
    #            on a `MethodError` by catching the exception and logging a summarized
    #            warning message instead of the full error object. It also uses the
    #            `calling_module` context to correctly evaluate the formula, which
    #            resolves the underlying `MethodError`.
    df_internal = copy(data)
    final_rhs_string = strip(formula_rhs)

    if isempty(final_rhs_string)
        return NamedArray(zeros(size(df_internal, 1), 0), (1:size(df_internal, 1), Symbol[])), nothing
    end

    if final_rhs_string == "1"
        return NamedArray(ones(size(df_internal, 1), 1), (1:size(df_internal, 1), [:Intercept])), nothing
    end

    try
        # Use a placeholder response variable for formula parsing.
        placeholder_name = :__y_placeholder
        if !hasproperty(df_internal, placeholder_name)
            df_internal[!, placeholder_name] = zeros(size(df_internal, 1))
        end

        formula_expression = Meta.parse("@formula($placeholder_name ~ $final_rhs_string)")
        
        # Evaluate the formula in the correct calling module to avoid world-age issues.
        dynamic_formula = Core.eval(calling_module, formula_expression)

        data_schema = StatsModels.schema(dynamic_formula, df_internal, contrasts)
        applied_formula = StatsModels.apply_schema(dynamic_formula, data_schema, StatsModels.RegressionModel)

        _, model_matrix_numeric = StatsModels.modelcols(applied_formula, df_internal)
        coefficient_labels = StatsModels.coefnames(applied_formula.rhs)

        label_vector = coefficient_labels isa AbstractString ? [Symbol(coefficient_labels)] : Symbol.(coefficient_labels)

        return NamedArray(model_matrix_numeric, (1:size(model_matrix_numeric, 1), label_vector)), applied_formula

    catch design_error
        # The `design_error` object can be large and contain the full DataFrame.
        # To prevent printing it, we log a concise message with only the error type.
        @warn "BSTM Registry: create_fixed_design expansion failed for: '$final_rhs_string'. Error of type '$(typeof(design_error))' occurred. Check formula syntax and variable names."
        return NamedArray(zeros(size(df_internal, 1), 0), (1:size(df_internal, 1), Symbol[])), nothing
    end
end




function householder_to_eigenvector(v_mat::AbstractMatrix{T}, nU, n_factors) where {T}
    # Purpose: Constructs an orthonormal loadings matrix (eigenvectors) from a matrix of Householder reflector vectors.
    # Rationale: Provides a differentiable and numerically stable way to parameterize an orthonormal matrix for Bayesian PCA.
    # v1.0.0 (2026-07-16)
    # Assumptions: `v_mat` contains the reflector vectors.
    # Inputs:
    #   - v_mat: Matrix of reflector vectors.
    #   - nU: Number of variables.
    #   - n_factors: Number of factors.
    # Outputs: An orthonormal loadings matrix `[nU x n_factors]`.
    U = Matrix{T}(I, nU, nU)

    for k in 1:n_factors
        vk = v_mat[:, k]
        norm_v = LinearAlgebra.norm(vk)
        
        if norm_v > 1e-9
            vk = vk / norm_v
            v_transpose_U = vk' * U
            U = U - 2.0 .* vk * v_transpose_U
        end
    end

    return U[:, 1:n_factors]
end

function show_model(m::DynamicPPL.Model)
    # Purpose: Displays a comprehensive summary of the `bstm` model configuration and a pseudo-code representation.
    # Rationale: Provides a user-friendly way to inspect the model structure before and after fitting.
    # v1.0.0 (2026-07-16)
    # Assumptions: `m` is a Turing model generated by the `bstm` framework.
    # Inputs:
    #   - m: The Turing model instance.
    # Outputs: None (prints to console).
    println("\n--- Model Summary ---\n")
    config = m.args.M
    println("Model Name: ", get(config, :model_name, nameof(m.f)))
    println("Model Architecture: ", get(config, :model_arch, "N/A"))
    println("Likelihood Family: ", get(config.likelihood_specs[1], :family, "N/A"))
    println("Number of observations: ", get(config, :y_N, "N/A"))
    println("Number of spatial units: ", get(config, :s_N, "N/A"))
    println("Number of time units: ", get(config, :t_N, "N/A"))
    println("\nFixed Effects:")
    if get(config, :Xfixed_N, 0) > 0
        println("  Variables: ", join(string.(get(config, :Xfixed_names, ["N/A"])), ", "))
    else
        println("  None")
    end
    println("\nComponents:\n")
    if haskey(config, :components) && !isempty(config.components)
        for spec in config.components
            println("  - Key: ", spec.key)
            println("    Structure: ", spec.structure)
            println("    Variable: ", spec.var)
            println("    Component Type: ", typeof(spec.component_obj))
            println("    Parameters:")
            for (p_key, p_val) in pairs(spec.params)
                println("      ", p_key, ": ", p_val)
            end
        end
    else
        println("  None")
    end
    if haskey(config, :generated_model_code)
        println("\n--- Generated Model Source ---\n")
        println(config.generated_model_code)
        println("\n--- End Generated Model Source ---")
    else
        println("\n--- Reconstructed Model Source (Pseudo-code) ---\n")
        println(model_pseudocode(m))
        println("\n--- End Reconstructed Model Source ---")
    end
    println("\n--- End Model Summary ---")
    return nothing
end


function model_pseudocode(m::DynamicPPL.Model)
    # Purpose: Reconstructs a pseudo-code representation of the Turing model definition.
    # Rationale: For inspection and clarity. This version is updated to use the simplified
    #            hyperparameter names (e.g., `sigma` instead of `sigma_prior`) and to be
    #            more comprehensive in the hyperpriors it checks for.
    # v1.0.0 (2026-07-20)
    # Assumptions: `m` is a Turing model generated by the `bstm` framework.
    # Inputs:
    #   - m: The Turing model instance.
    # Outputs: A string containing the pseudo-code.
    config = m.args.M
    model_name = get(config, :model_name, nameof(m.f))
    
    lines = ["@model function $model_name(M)"]
    push!(lines, "    # --- Priors & Hyperparameters ---")

    family = get(config.likelihood_specs[1], :family, "gaussian")
    if family == "negbin"
        push!(lines, "    r_nb ~ Exponential(1.0)")
    end
    if get(config, :use_zi, false)
        push!(lines, "    phi_zi ~ Beta(1, 1)")
    end
    if family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"] && !get(config, :use_sv, false)
        push!(lines, "    y_sigma ~ Exponential(1.0)")
    end

    if haskey(config, :components) && !isempty(config.components)
        for spec in config.components
            m_obj = spec.component_obj
            m_type_str = string(typeof(m_obj))
            key = spec.key
            
            push!(lines, "\n    # Priors for component: $(key) ($(m_type_str))")
            
            possible_hyperpriors = [
                (:sigma, "sigma_$(key)"),
                (:rho, "rho_$(key)"),
                (:lengthscale, "ls_$(key)"),
                (:kappa, "kappa_$(key)"),
                (:amplitude, "amp_$(key)"),
                (:phase, "phase_$(key)"),
                (:range, "range_$(key)"),
                (:pca_sd, "pca_sd_$(key)"),
                (:pdef_sd, "pdef_sd_$(key)")
            ]

            for (field_sym, name_str) in possible_hyperpriors
                if hasproperty(m_obj, field_sym)
                    prior_dist = getfield(m_obj, field_sym)
                    if !isnothing(prior_dist)
                        push!(lines, "    $(name_str) ~ $(prior_dist)")
                    end
                end
            end
        end
    end

    if get(config, :Xfixed_N, 0) > 0
        push!(lines, "\n    # Prior for fixed effects")
        push!(lines, "    Xfixed_beta ~ MvNormal(0, 5.0 * I)")
    end

    push!(lines, "\n    # --- Latent Field Definitions & Linear Predictor Assembly ---")
    eta_parts = haskey(config, :log_offset) && !all(iszero, get(config, :log_offset, [])) ? ["M.log_offset"] : []

    if get(config, :add_intercept, false)
        push!(eta_parts, "intercept")
    end
    
    model_st = get(config, :model_st, "none")
    if model_st != "none"
        push!(eta_parts, "spacetime_interaction")
    end

    if haskey(config, :components)
        for spec in config.components
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


# ==============================================================================
# SECTION 10: ADVANCED MODELING UTILITIES 
# ==============================================================================

"""
    bstm_bspline_basis(x::AbstractVector, n_basis::Int, degree::Int; knot_method::Symbol=:quantile, custom_knots::Union{AbstractVector, Nothing}=nothing)

Generates a B-spline basis matrix of a specified degree. This function implements the
De Boor-Cox recursive formula in an iterative manner to create basis functions.

# Arguments
- `x`: The vector of data points for which the basis is evaluated.
- `n_basis`: The number of basis functions to generate.
- `degree`: The polynomial degree of the B-spline (e.g., 1 for linear, 3 for cubic).
- `knot_method`: The method for placing interior knots. Can be `:quantile` (default) for knots placed at data quantiles, or `:range` for knots spaced evenly over the data range.
- `custom_knots`: An optional vector of pre-defined interior knots.

# Returns
- A tuple `(Matrix, Int)` containing the basis matrix of size `(length(x), n_basis)` and the final number of basis functions used.
"""
function bstm_bspline_basis(x::AbstractVector, n_basis::Int, degree::Int; knot_method::Symbol=:quantile, custom_knots::Union{AbstractVector, Nothing}=nothing)
    p = degree
    if n_basis <= p
        error("Number of basis functions (nbins) must be greater than the spline degree. Got n_basis=$n_basis, degree=$p.")
    end

    # The number of interior knots is determined by the number of basis functions and the degree.
    n_interior_knots = n_basis - p

    local knots
    if !isnothing(custom_knots)
        knots = custom_knots
    else
        # Place interior knots based on the chosen method.
        if n_interior_knots > 0
            if knot_method == :quantile
                # Ensure that quantile probabilities are unique to avoid issues with discrete data
                probs = range(0, 1, length=n_interior_knots + 2)[2:end-1]
                knots = quantile(x, probs)
            else # :range
                knots = range(minimum(x), maximum(x), length=n_interior_knots + 2)[2:end-1]
            end
        else
            knots = Float64[]
        end
    end

    # Define the full knot vector by adding boundary knots.
    boundary_knots = [minimum(x), maximum(x)]
    all_knots = sort(unique(vcat(boundary_knots, knots)))

    # --- Robustness Check for Low-Variability Data ---
    # The number of basis functions that can be constructed depends on the number of unique knots.
    # If the data has low variability, quantile-based knots can be identical, reducing the
    # number of unique knots and thus the possible number of basis functions.
    n_basis_possible = length(all_knots) + p - 1
    
    if n_basis > n_basis_possible
        @warn "Requested n_basis ($n_basis) is too high for the number of unique knots ($(length(all_knots))) and degree ($p) supported by the data's variability. Reducing n_basis to the maximum possible value: $n_basis_possible."
        n_basis = n_basis_possible
    end
    # --- End Robustness Check ---

    # Augment the knot vector by repeating the boundary knots `p` times at each end.
    # This is required for the De Boor-Cox recursion.
    t = vcat(fill(all_knots[1], p), all_knots, fill(all_knots[end], p))
    
    N = length(x)
    # The number of basis functions is length of augmented knots - degree - 1.
    num_total_basis = length(t) - p - 1
    B = zeros(N, num_total_basis)

    # Base case: degree 0 splines are piecewise constant.
    for j in 1:num_total_basis
        B[:, j] = (t[j] .<= x .< t[j+1])
    end
    # Ensure the last point is included in the final basis function.
    if !isempty(x) && t[end] == maximum(x)
        B[x .== t[end], num_total_basis] .= 1.0
    end

    # Iteratively compute higher-degree splines from lower-degree ones.
    for d in 1:p
        for j in 1:(num_total_basis - d)
            w1 = zeros(N)
            denom1 = t[j+d] - t[j]
            if denom1 > 1e-9 # Avoid division by zero
                w1 = (x .- t[j]) ./ denom1
            end
            
            w2 = zeros(N)
            denom2 = t[j+d+1] - t[j+1]
            if denom2 > 1e-9 # Avoid division by zero
                w2 = (t[j+d+1] .- x) ./ denom2
            end
            
            B[:, j] = w1 .* B[:, j] + w2 .* B[:, j+1]
        end
    end

    # Return the final basis matrix and the actual number of basis functions.
    return (B[:, 1:n_basis], n_basis)
end

"""
    bstm_tensor_product_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int}, degrees_per_dim::Vector{Int}; ...)

Generates a tensor product B-spline basis matrix for multidimensional data.

# Rationale for Update
This version fixes a `MethodError` that occurred when unsupported keyword arguments
were passed down to the `bstm_bspline_basis` function. The `kwargs...` splat has been
removed from the call to `bstm_bspline_basis`, and only the relevant keyword arguments
(`knot_method`, `custom_knots`) are explicitly passed. It also correctly handles the
tuple `(matrix, n_basis)` returned by `bstm_bspline_basis` to ensure the tensor
product is computed on the basis matrices.
"""
function bstm_tensor_product_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int}, degrees_per_dim::Vector{Int}; knot_method::Symbol=:quantile, kwargs...)
    n_dims = size(coords, 2)
    if length(nbins_per_dim) != n_dims || length(degrees_per_dim) != n_dims
        error("Number of dimensions in coords must match length of nbins_per_dim and degrees_per_dim.")
    end

    # Filter kwargs to only include those accepted by bstm_bspline_basis.
    # This prevents passing unsupported arguments like `structure`, `model`, etc.
    bspline_kwargs = Dict{Symbol, Any}()
    if haskey(kwargs, :custom_knots)
        bspline_kwargs[:custom_knots] = kwargs[:custom_knots]
    end

    # Generate 1D B-spline basis matrices for each dimension.
    basis_matrices_1D = Vector{Matrix{Float64}}(undef, n_dims)
    for i in 1:n_dims
        # For multi-dimensional custom knots, we need to pass the correct slice.
        local_bspline_kwargs = copy(bspline_kwargs)
        if haskey(local_bspline_kwargs, :custom_knots) && local_bspline_kwargs[:custom_knots] isa Tuple
            local_bspline_kwargs[:custom_knots] = local_bspline_kwargs[:custom_knots][i]
        end
        
        # bstm_bspline_basis returns a tuple (matrix, n_basis). We only need the matrix here.
        basis_mat, _ = bstm_bspline_basis(
            coords[:, i], 
            nbins_per_dim[i], 
            degrees_per_dim[i]; 
            knot_method=knot_method, 
            local_bspline_kwargs...
        ) # Pass custom_knots explicitly
        basis_matrices_1D[i] = basis_mat
    end

    if isempty(basis_matrices_1D)
        return zeros(size(coords, 1), 0)
    end

    # Initialize with the first basis matrix.
    B_final = basis_matrices_1D[1]

    # Iteratively compute the tensor product with the remaining matrices.
    for i in 2:n_dims
        B_next = basis_matrices_1D[i]
        
        n_obs, n_cols_final = size(B_final)
        _, n_cols_next = size(B_next)
        
        # Reshape for broadcasting to compute row-wise outer products.
        B_final_reshaped = reshape(B_final, n_obs, n_cols_final, 1)
        B_next_reshaped = reshape(B_next, n_obs, 1, n_cols_next)
        
        tensor_prod = B_final_reshaped .* B_next_reshaped
        
        B_final = reshape(tensor_prod, n_obs, n_cols_final * n_cols_next)
    end
    
    return B_final
end



function _reconstruct_wavelet_function_from_filters(h::Vector{Float64}, g::Vector{Float64}, n_iterations::Int)
    # Dynamically load Interpolations to ensure it's available in the execution scope.
    Interpolations = Base.require(Base.Main, :Interpolations)

    L = length(h) 
    x_min_support = 0.0
    x_max_support = L > 1 ? L - 1.0 : 1.0

    num_points_final_grid = max(2, (2^n_iterations) * max(1, L - 1) + 1)
    x_grid_final = collect(range(x_min_support, stop=x_max_support, length=num_points_final_grid))

    phi_current_vals = zeros(length(x_grid_final))
    for i in eachindex(x_grid_final)
        if 0.0 <= x_grid_final[i] < 1.0; phi_current_vals[i] = 1.0; end
    end
    # Qualify call with the dynamically loaded module.
    phi_itp = Interpolations.linear_interpolation(x_grid_final, phi_current_vals, extrapolation_bc=Interpolations.Flat())

    psi_next_vals = zeros(length(x_grid_final))

    for iter in 1:n_iterations
        phi_next_vals = zeros(length(x_grid_final))
        for idx in eachindex(x_grid_final)
            x_val = x_grid_final[idx]
            phi_sum = 0.0
            psi_sum = 0.0
            for k_filter in 0:(L-1)
                phi_sum += h[k_filter+1] * phi_itp(2.0 * x_val - k_filter)
                psi_sum += g[k_filter+1] * phi_itp(2.0 * x_val - k_filter)
            end
            phi_next_vals[idx] = sqrt(2.0) * phi_sum
            psi_next_vals[idx] = sqrt(2.0) * psi_sum
        end
        # Qualify call with the dynamically loaded module.
        phi_itp = Interpolations.linear_interpolation(x_grid_final, phi_next_vals, extrapolation_bc=Interpolations.Flat())
    end
    
    return x_grid_final, psi_next_vals
end



function bstm_wavelet_basis_1D(vals::AbstractVector, nbins::Int, family::Symbol, lengthscale::Float64)
    # Dynamically load Interpolations to ensure it's available in the execution scope.
    Interpolations = Base.require(Base.Main, :Interpolations)

    n_obs = length(vals)
    B = zeros(Float64, n_obs, nbins)
    v_min, v_max = minimum(vals), maximum(vals)
    v_range = v_max - v_min
    if v_range < 1e-9; v_range = 1.0; end

    local wt_type
    try
        wt_type = getfield(Wavelets.WT, family)
    catch e
        @error "Could not resolve wavelet family ':$family'. Error: $e. Defaulting to db4."
        wt_type = Wavelets.WT.db4
    end

    local wt_instance
    try
        wt_instance = Wavelets.wavelet(wt_type)
    catch e
        error("Failed to instantiate wavelet object from type '$wt_type'. Error: $e")
    end

    h_filter = wt_instance.qmf
    
    L = length(h_filter)
    g_filter = similar(h_filter)
    for i in 1:L
        g_filter[i] = (-1.0)^(i-1) * h_filter[L - (i-1)]
    end

    n_reconstruction_iterations = 8
    x_psi_grid, psi_vals = _reconstruct_wavelet_function_from_filters(h_filter, g_filter, n_reconstruction_iterations)

    # Qualify call with the dynamically loaded module.
    itp = Interpolations.linear_interpolation(x_psi_grid, psi_vals, extrapolation_bc=Interpolations.Flat())

    n_scales = max(1, floor(Int, log2(nbins/4)))
    bins_per_scale = div(nbins, n_scales)
    
    current_bin = 1
    for j in 1:n_scales
        scale_factor = lengthscale * (2.0^(j-1))
        
        n_translations = (j == n_scales) ? (nbins - current_bin + 1) : bins_per_scale
        if n_translations <= 0; continue; end

        probs = n_translations == 1 ? [0.5] : range(0, 1, length=n_translations)
        centers = quantile(vals, probs)
        
        for k in 1:n_translations
            if current_bin > nbins; break; end
            
            transformed_vals = (vals .- centers[k]) ./ (scale_factor * v_range)
            B[:, current_bin] = itp.(transformed_vals)
            current_bin += 1
        end
    end
    return B
end



# Dependent function for tensor product wavelet basis (no changes needed)
function bstm_tensor_product_wavelet_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int}, family::Symbol, lengthscale::Union{Real, AbstractVector})
    n_dims = size(coords, 2)
    if length(nbins_per_dim) != n_dims; error("Length of `nbins_per_dim` must match coordinate dimensions."); end
    
    ls_vec = if lengthscale isa Real
        fill(Float64(lengthscale), n_dims)
    else
        if length(lengthscale) != n_dims; error("Length of lengthscale vector must match coordinate dimensions."); end
        lengthscale
    end

    basis_matrices_1D = [bstm_wavelet_basis_1D(coords[:, i], nbins_per_dim[i], family, ls_vec[i]) for i in 1:n_dims]
    
    if isempty(basis_matrices_1D); return zeros(size(coords, 1), 0); end

    B_final = basis_matrices_1D[1]
    for i in 2:n_dims
        B_next = basis_matrices_1D[i]
        n_obs, n_cols_final = size(B_final)
        _, n_cols_next = size(B_next)
        B_final_reshaped = reshape(B_final, n_obs, n_cols_final, 1)
        B_next_reshaped = reshape(B_next, n_obs, 1, n_cols_next)
        tensor_prod = B_final_reshaped .* B_next_reshaped
        B_final = reshape(tensor_prod, n_obs, n_cols_final * n_cols_next)
    end
    return B_final
end


"""
    bstm_barycentric_basis_4D(coords::AbstractMatrix, knots::Vector{Point4D}, n_marginal::Int)

Generates a 4D basis matrix using multilinear interpolation on a regular grid of knots.

# Rationale
A true barycentric interpolation in 4D would require a Delaunay triangulation of the
knot points to form a mesh of 4-simplices (pentatopes). This is computationally
prohibitive and not supported by standard Julia libraries.

This function provides a practical and efficient approximation by constructing a basis
from the tensor product of 1D linear "tent" functions centered at the provided knot
points. This is equivalent to multilinear interpolation within the 4D hyper-rectangles
defined by the grid of knots.

# Arguments
- `coords`: An `N x 4` matrix of data points.
- `knots`: A vector of `Point4D` knot points, assumed to form a regular grid.
- `n_marginal`: The number of knots along each of the 4 dimensions.

# Returns
- A basis matrix of size `(N, length(knots))`.
"""
function bstm_barycentric_basis_4D(coords::AbstractMatrix, knots::Vector{Point4D}, n_marginal::Int)
    n_obs = size(coords, 1)
    n_knots = length(knots)
    B = zeros(Float64, n_obs, n_knots)

    if n_knots != n_marginal^4
        error("Number of knots must equal n_marginal^4 for 4D tensor product basis. Got n_knots=$n_knots and n_marginal^4=$(n_marginal^4).")
    end

    # Extract unique knot coordinates along each dimension and sort them
    k1 = sort(unique([k.x for k in knots]))
    k2 = sort(unique([k.y for k in knots]))
    k3 = sort(unique([k.z for k in knots]))
    k4 = sort(unique([k.t for k in knots]))

    # Calculate grid spacing (h) for each dimension
    h1 = (maximum(k1) - minimum(k1)) / (n_marginal > 1 ? (n_marginal - 1) : 1); h1 = h1 > 0 ? h1 : 1.0
    h2 = (maximum(k2) - minimum(k2)) / (n_marginal > 1 ? (n_marginal - 1) : 1); h2 = h2 > 0 ? h2 : 1.0
    h3 = (maximum(k3) - minimum(k3)) / (n_marginal > 1 ? (n_marginal - 1) : 1); h3 = h3 > 0 ? h3 : 1.0
    h4 = (maximum(k4) - minimum(k4)) / (n_marginal > 1 ? (n_marginal - 1) : 1); h4 = h4 > 0 ? h4 : 1.0

    idx = 1
    # The loop order must match the order in which the knot_points vector was created.
    # The standard is x-fastest, then y, then z, etc.
    for l in 1:n_marginal
        for k in 1:n_marginal
            for j in 1:n_marginal
                for i in 1:n_marginal
                    b1 = max.(0.0, 1.0 .- abs.(coords[:, 1] .- k1[i]) ./ h1)
                    b2 = max.(0.0, 1.0 .- abs.(coords[:, 2] .- k2[j]) ./ h2)
                    b3 = max.(0.0, 1.0 .- abs.(coords[:, 3] .- k3[k]) ./ h3)
                    b4 = max.(0.0, 1.0 .- abs.(coords[:, 4] .- k4[l]) ./ h4)
                    B[:, idx] .= b1 .* b2 .* b3 .* b4
                    idx += 1
                end
            end
        end
    end
    
    return B
end


 
"""
    bstm_smooth_basis_1D(type::String, vals::AbstractVector, nbins::Int, degree::Int; ...)

Generates a 1D basis matrix for various smoothers. This is an updated version.

# Rationale for Update
The implementation for `pspline` and `bspline` now calls `bstm_bspline_basis` to generate
a proper B-spline basis of the specified `degree`. The previous implementation was
hardcoded to a linear spline. The original linear "tent function" behavior is retained
for the aliases `smooth`, `barycentric`, and `linear` for backward compatibility.
"""
function bstm_smooth_basis_1D(type::String, vals::AbstractVector, nbins::Int, degree::Int; W=nothing, knot_method::Symbol = :quantile, custom_knots::Union{AbstractVector, Nothing} = nothing, kwargs...)
    n_obs = length(vals)
    
    v_min = minimum(vals)
    v_max = maximum(vals)
    v_std = std(vals) + 1e-9
    use_regular_grid = type in ["invdist", "kriging", "tps", "spherical"]
    local knots
    if knot_method == :custom && !isnothing(custom_knots)
        knots = custom_knots
    elseif knot_method == :range || use_regular_grid
        knots = collect(range(v_min, stop=v_max, length=nbins))
    else # :quantile or any other default
        knots = quantile(vals, range(0, 1, length=nbins))
    end
    local B_out
    local actual_nbins_generated = nbins # Default to nbins, will be updated for splines
    if type in ["pspline", "bspline"]
        # bstm_bspline_basis now returns a tuple (basis_matrix, actual_nbins).
        B_out, actual_nbins_generated = bstm_bspline_basis(vals, nbins, degree; knot_method=knot_method, custom_knots=custom_knots)
        # The slicing is already done inside bstm_bspline_basis, so no further action is needed here.
    else # For other types, generate B and assume nbins is the actual count
        B_out = zeros(Float64, n_obs, nbins)
        if type in ["smooth", "barycentric", "linear"]
            # Retain the original linear tent function implementation for these aliases.
            h = (v_max - v_min) / (nbins > 1 ? (nbins - 1) : 1)
            h = h > 0 ? h : 1.0
            for m in 1:nbins
                dist = abs.(vals .- knots[m]) ./ h
                mask = dist .< 1.0
                B_out[mask, m] .= 1.0 .- dist[mask]
            end
        elseif type == "tps"
            # The radial basis function for 1D TPS (m=2, d=1) is r^3.
            for m in 1:nbins
                r = abs.(vals .- knots[m])
                B_out[:, m] .= r.^3
            end
        elseif type == "rff"
            m_rff = nbins
            ls = get(kwargs, :lengthscale, v_std)
            Omega = randn(1, m_rff) ./ ls
            Phi_phases = rand(m_rff) .* (2.0 * pi)
            B_out .= sqrt(2.0 / m_rff) .* cos.((vals * Omega) .+ Phi_phases')
        elseif type == "fft"
            ls = get(kwargs, :lengthscale, v_std)
            t_coords = vals ./ ls
            for m in 1:div(nbins, 2)
                B_out[:, 2m-1] .= sin.(2.0 * pi * m .* t_coords)
                B_out[:, 2m]   .= cos.(2.0 * pi * m .* t_coords)
            end
        elseif type == "wavelet"
            family = get(kwargs, :family, :db4)
            lengthscale = get(kwargs, :lengthscale, 0.1)
            B_out = bstm_wavelet_basis_1D(vals, nbins, family, lengthscale)
        elseif type == "spherical"
            range_r = get(kwargs, :range, v_std * 2.0)
            for m in 1:nbins
                h = abs.(vals .- knots[m]) ./ range_r
                mask = h .< 1.0
                B_out[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
            end
        elseif type == "decay"
            ls = get(kwargs, :lengthscale, v_std)
            for m in 1:nbins
                B_out[:, m] .= exp.(-abs.(vals .- knots[m]) ./ ls)
            end
        elseif type == "invdist"
            for m in 1:nbins
                dist_sq = (vals .- knots[m]).^2
                B_out[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
            end
        elseif type == "kriging"
            ls = get(kwargs, :lengthscale, v_std)
            for m in 1:nbins
                dist_sq = (vals .- knots[m]).^2
                B_out[:, m] .= exp.(-dist_sq ./ (2 * ls^2))
            end
        else
            B_out = ones(n_obs, 1)
            actual_nbins_generated = 1
        end
    end
    return B_out, actual_nbins_generated
end



"""
    bstm_barycentric_basis_2D(coords::AbstractMatrix, knots::Vector{Point2D})

Generates a 2D barycentric basis matrix based on a Delaunay triangulation of knot points.

# Rationale
This provides a true triangulation-based barycentric interpolation, aligning the
implementation with the documentation's reference to "Delaunay/Voronoi" methods.
It is more flexible for irregularly spaced data than the previous grid-based
bilinear interpolation.

# Arguments
- `coords`: An `N x 2` matrix of data points.
- `knots`: A vector of `Point2D` knot points (vertices for the triangulation).

# Returns
- A sparse basis matrix of size `(N, length(knots))`.
"""
function bstm_barycentric_basis_2D(coords::AbstractMatrix, knots::Vector{Point2D})
    n_obs = size(coords, 1)
    n_knots = length(knots)
    B = spzeros(Float64, n_obs, n_knots)

    # 1. Perform Delaunay triangulation on the knot points
    triangles = _delaunay_triangulation(knots)
    if isempty(triangles)
        @warn "Delaunay triangulation failed or resulted in no triangles. Returning an empty basis."
        return B
    end

    # 2. For each observation, find its enclosing triangle and barycentric coordinates
    for i in 1:n_obs
        obs_point = Point2D(coords[i, 1], coords[i, 2])
        
        for tri in triangles
            v1_idx, v2_idx, v3_idx = tri.v1, tri.v2, tri.v3
            p1, p2, p3 = knots[v1_idx], knots[v2_idx], knots[v3_idx]

            if _is_inside_triangle(obs_point, p1, p2, p3)
                bary_coords = _get_barycentric_coords(obs_point, p1, p2, p3)
                if !isnothing(bary_coords)
                    w1, w2, w3 = bary_coords
                    B[i, v1_idx] = w1
                    B[i, v2_idx] = w2
                    B[i, v3_idx] = w3
                end
                break # Found the enclosing triangle
            end
        end
    end
    return B
end



# Helper to get barycentric coordinates of a point in a triangle
function _get_barycentric_coords(p::Point2D, p1::Point2D, p2::Point2D, p3::Point2D)
    # Using the formula based on areas
    area_total = abs((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))
    if area_total < 1e-9 return nothing end

    # Area of sub-triangles
    area1 = abs((p2.x - p.x) * (p3.y - p.y) - (p3.x - p.x) * (p2.y - p.y)) # for p1
    area2 = abs((p3.x - p.x) * (p1.y - p.y) - (p1.x - p.x) * (p3.y - p.y)) # for p2
    area3 = abs((p1.x - p.x) * (p2.y - p.y) - (p2.x - p.x) * (p1.y - p.y)) # for p3

    w1 = area1 / area_total
    w2 = area2 / area_total
    w3 = area3 / area_total
    
    # Normalize to ensure sum is 1, accounting for float precision
    w_sum = w1 + w2 + w3
    return (w1/w_sum, w2/w_sum, w3/w_sum)
end



# Helper function to calculate the circumcenter and squared radius of a triangle
function _get_circumcircle(p1::Point2D, p2::Point2D, p3::Point2D)
    D = 2 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))
    if abs(D) < 1e-9
        return nothing, nothing # Collinear points
    end

    p1_sq = p1.x^2 + p1.y^2
    p2_sq = p2.x^2 + p2.y^2
    p3_sq = p3.x^2 + p3.y^2

    center_x = (p1_sq * (p2.y - p3.y) + p2_sq * (p3.y - p1.y) + p3_sq * (p1.y - p2.y)) / D
    center_y = (p1_sq * (p3.x - p2.x) + p2_sq * (p1.x - p3.x) + p3_sq * (p2.x - p1.x)) / D
    
    center = Point2D(center_x, center_y)
    radius_sq = (p1.x - center.x)^2 + (p1.y - center.y)^2
    
    return center, radius_sq
end

# Helper to check if a point is inside the circumcircle of a triangle
function _is_in_circumcircle(p::Point2D, p1::Point2D, p2::Point2D, p3::Point2D)
    center, radius_sq = _get_circumcircle(p1, p2, p3)
    if isnothing(center)
        return false # Cannot be in the circumcircle of collinear points
    end
    dist_sq = (p.x - center.x)^2 + (p.y - center.y)^2
    return dist_sq < radius_sq
end


# Bowyer-Watson algorithm for Delaunay triangulation
function _delaunay_triangulation(points::Vector{Point2D})
    n = length(points)
    if n < 3
        return []
    end

    # Determine a "super-triangle" that encloses all points
    min_x = minimum(p.x for p in points)
    max_x = maximum(p.x for p in points)
    min_y = minimum(p.y for p in points)
    max_y = maximum(p.y for p in points)
    
    dx = max_x - min_x
    dy = max_y - min_y
    delta_max = max(dx, dy)
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    # Define vertices of the super-triangle
    p_super1 = Point2D(mid_x - 20 * delta_max, mid_y - delta_max)
    p_super2 = Point2D(mid_x + 20 * delta_max, mid_y - delta_max)
    p_super3 = Point2D(mid_x, mid_y + 20 * delta_max)
    
    # The indices of the super-triangle vertices will be n+1, n+2, n+3
    super_triangle = Triangle(n + 1, n + 2, n + 3)
    all_points = [points; p_super1; p_super2; p_super3]

    triangulation = [super_triangle]

    for i in 1:n
        point = points[i]
        bad_triangles = []
        
        for tri in triangulation
            p1 = all_points[tri.v1]
            p2 = all_points[tri.v2]
            p3 = all_points[tri.v3]
            if _is_in_circumcircle(point, p1, p2, p3)
                push!(bad_triangles, tri)
            end
        end

        polygon = []
        for tri in bad_triangles
            edges = [(tri.v1, tri.v2), (tri.v2, tri.v3), (tri.v3, tri.v1)]
            for edge in edges
                is_shared = false
                for other_tri in bad_triangles
                    if tri === other_tri continue end
                    other_edges = [(other_tri.v1, other_tri.v2), (other_tri.v2, other_tri.v3), (other_tri.v3, other_tri.v1)]
                    if (edge in other_edges) || ((edge[2], edge[1]) in other_edges)
                        is_shared = true
                        break
                    end
                end
                if !is_shared
                    push!(polygon, edge)
                end
            end
        end

        # Remove bad triangles from triangulation
        filter!(t -> !(t in bad_triangles), triangulation)

        # Form new triangles from the polygon edges to the new point
        for edge in polygon
            push!(triangulation, Triangle(edge[1], edge[2], i))
        end
    end

    # Remove triangles that include vertices of the super-triangle
    filter!(t -> !(t.v1 > n || t.v2 > n || t.v3 > n), triangulation)

    return triangulation
end

# Helper to check if a point is inside a triangle
function _is_inside_triangle(p::Point2D, p1::Point2D, p2::Point2D, p3::Point2D)
    # Using barycentric coordinates. A point is inside if all coordinates are non-negative.
    coords = _get_barycentric_coords(p, p1, p2, p3)
    return !isnothing(coords) && all(c -> c >= -1e-9, coords) # Allow for small float inaccuracies
end


"""
    bstm_smooth_basis_2D(type::String, coords::AbstractMatrix, nbins::Union{Int, Vector{Int}}; ...)

Generates a 2D basis matrix for various smoothers.

# Rationale for Update
This version unifies the `barycentric` smoother with the `smooth` and `linear` types.
The previous implementation for `barycentric` used a complex and likely incorrect
Delaunay triangulation on a regular grid. The corrected version now uses the robust
and standard tensor product of 1D linear "tent" functions (bilinear interpolation),
which is the correct interpretation for a barycentric-style basis on a grid.
"""
function bstm_smooth_basis_2D(type::String, coords::AbstractMatrix, nbins::Union{Int, Vector{Int}}; W=nothing, knot_method::Symbol = :quantile, custom_knots::Union{Tuple{AbstractVector, AbstractVector}, Nothing} = nothing, kwargs...)
    n_obs = size(coords, 1)
    
    local n_marginal_x, n_marginal_y
    if nbins isa Int
        n_marginal_x = nbins
        n_marginal_y = nbins
    elseif nbins isa Vector{Int} && length(nbins) == 2
        n_marginal_x = nbins[1]
        n_marginal_y = nbins[2]
    else
        error("For a 2D smooth, `nbins` must be an Int or a Vector{Int} of length 2.")
    end
    total_bins = n_marginal_x * n_marginal_y

    c_min = [minimum(coords[:, 1]), minimum(coords[:, 2])]
    c_max = [maximum(coords[:, 1]), maximum(coords[:, 2])]
    c_std = [std(coords[:, 1]), std(coords[:, 2])] .+ 1e-9

    ls_x = get(kwargs, :ls_x, c_std[1])
    ls_y = get(kwargs, :ls_y, c_std[2])

    local kx, ky
    use_regular_grid = type in ["invdist", "kriging", "tps", "spherical"]

    if knot_method == :custom && !isnothing(custom_knots)
        kx, ky = custom_knots
    elseif knot_method == :quantile && !use_regular_grid
        kx = quantile(coords[:, 1], range(0, 1, length=n_marginal_x))
        ky = quantile(coords[:, 2], range(0, 1, length=n_marginal_y))
    else # :range or if regular grid is required
        kx = collect(range(c_min[1], stop=c_max[1], length=n_marginal_x))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_marginal_y))
    end
    
    B = zeros(Float64, n_obs, total_bins)

    if type in ["pspline", "bspline"]
        degree_val = get(kwargs, :degree, 3)
        nbins_per_dim = [n_marginal_x, n_marginal_y]
        degrees_per_dim = [degree_val, degree_val]
        return bstm_tensor_product_basis(coords, nbins_per_dim, degrees_per_dim; knot_method=knot_method, kwargs...)
    end

    if type in ["smooth", "barycentric", "linear"]
        hx = (c_max[1] - c_min[1]) / (n_marginal_x > 1 ? (n_marginal_x - 1) : 1); hx = hx > 0 ? hx : 1.0
        hy = (c_max[2] - c_min[2]) / (n_marginal_y > 1 ? (n_marginal_y - 1) : 1); hy = hy > 0 ? hy : 1.0
        idx = 1
        for j in 1:n_marginal_y, i in 1:n_marginal_x
            if idx > total_bins; break; end
            b_x = max.(0.0, 1.0 .- abs.(coords[:, 1] .- kx[i]) ./ hx)
            b_y = max.(0.0, 1.0 .- abs.(coords[:, 2] .- ky[j]) ./ hy)
            B[:, idx] .= b_x .* b_y
            idx += 1
        end
    elseif type == "wavelet"
        family = get(kwargs, :family, :db4)
        lengthscale = get(kwargs, :lengthscale, 0.1)
        nbins_per_dim = [n_marginal_x, n_marginal_y]
        return bstm_tensor_product_wavelet_basis(coords, nbins_per_dim, family, lengthscale)
    end

    if type == "tps"
        centers = [(x, y) for y in ky for x in kx][:]
        for m in 1:total_bins
            dx = coords[:, 1] .- centers[m][1]
            dy = coords[:, 2] .- centers[m][2]
            r = sqrt.(dx.^2 .+ dy.^2)
            B[:, m] .= (r.^2) .* log.(r .+ 1e-9)
        end
        return B
    end

    if type == "rff" || type == "anisotropic"
        Omega = randn(2, total_bins)
        Omega[1, :] ./= ls_x
        Omega[2, :] ./= ls_y
        Phi_phases = rand(total_bins) .* (2.0 * pi)
        B = sqrt(2.0 / total_bins) .* cos.((coords * Omega) .+ Phi_phases')
    elseif type == "fft"
        nx = coords[:, 1] ./ ls_x
        ny = coords[:, 2] ./ ls_y
        idx = 1
        for my in 1:n_marginal_y, mx in 1:n_marginal_x
            if idx + 1 <= total_bins
                arg = mx .* nx .+ my .* ny
                B[:, idx]   .= sin.(2.0 * pi * arg)
                B[:, idx+1] .= cos.(2.0 * pi * arg)
                idx += 2
            end
        end
    end

    if type == "spherical"
        centers = [(x, y) for x in kx, y in ky][:]
        range_r = get(kwargs, :range, mean(c_std))
        for m in 1:total_bins
            dx = coords[:, 1] .- centers[m][1]
            dy = coords[:, 2] .- centers[m][2]
            h = sqrt.(dx.^2 .+ dy.^2) ./ range_r
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end
    elseif  type == "invdist"
        centers = [(x, y) for x in kx, y in ky][:]
        for m in 1:total_bins
            dist_sq = (coords[:, 1] .- centers[m][1]).^2 .+ (coords[:, 2] .- centers[m][2]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end
    end

    if  type == "kriging"
        centers = [(x, y) for x in kx, y in ky][:]
        for m in 1:total_bins
            dist_sq = ((coords[:, 1] .- centers[m][1]).^2 ./ ls_x^2) .+ ((coords[:, 2] .- centers[m][2]).^2 ./ ls_y^2)
            B[:, m] .= exp.(-dist_sq ./ 2.0)
        end
    end

    return B[:, 1:min(total_bins, size(B, 2))]
end

"""
    bstm_smooth_basis_3D(type::String, coords::AbstractMatrix, nbins::Union{Int, Vector{Int}}; ...)

Generates a 3D basis matrix for various smoothers.

# Rationale for Update
This version fixes a critical bug where the `barycentric` smoother attempted to call
a non-existent function `bstm_barycentric_basis_3D`. The logic has been corrected
to unify `barycentric` with the `smooth` and `linear` types, which correctly implement
a tensor product of 1D linear basis functions (trilinear interpolation).
"""
function bstm_smooth_basis_3D(type::String, coords::AbstractMatrix, nbins::Union{Int, Vector{Int}}; W=nothing, knot_method::Symbol = :quantile, custom_knots::Union{Tuple{AbstractVector, AbstractVector, AbstractVector}, Nothing} = nothing, kwargs...)
    n_obs = size(coords, 1)

    local n_marginal_x, n_marginal_y, n_marginal_z
    if nbins isa Int
        n_marginal_x = nbins
        n_marginal_y = nbins
        n_marginal_z = nbins
    elseif nbins isa Vector{Int} && length(nbins) == 3
        n_marginal_x = nbins[1]
        n_marginal_y = nbins[2]
        n_marginal_z = nbins[3]
    else
        error("For a 3D smooth, `nbins` must be an Int or a Vector{Int} of length 3.")
    end
    total_bins = n_marginal_x * n_marginal_y * n_marginal_z

    c_min = [minimum(coords[:, i]) for i in 1:3]
    c_max = [maximum(coords[:, i]) for i in 1:3]
    c_std = [std(coords[:, i]) for i in 1:3] .+ 1e-9

    ls_x = get(kwargs, :ls_x, c_std[1])
    ls_y = get(kwargs, :ls_y, c_std[2])
    ls_z = get(kwargs, :ls_z, c_std[3])

    local kx, ky, kz
    use_regular_grid = type in ["invdist", "kriging", "tps", "spherical"]

    if knot_method == :custom && !isnothing(custom_knots)
        if length(custom_knots) != 3 || length(custom_knots[1]) != n_marginal_x || length(custom_knots[2]) != n_marginal_y || length(custom_knots[3]) != n_marginal_z
            @warn "Custom knots for 3D smoother must be a Tuple of 3 AbstractVectors with lengths matching `nbins`. Falling back to :quantile method."
            kx = quantile(coords[:, 1], range(0, 1, length=n_marginal_x))
            ky = quantile(coords[:, 2], range(0, 1, length=n_marginal_y))
            kz = quantile(coords[:, 3], range(0, 1, length=n_marginal_z))
        else
            kx, ky, kz = custom_knots
        end
    elseif knot_method == :quantile && !use_regular_grid
        kx = quantile(coords[:, 1], range(0, 1, length=n_marginal_x))
        ky = quantile(coords[:, 2], range(0, 1, length=n_marginal_y))
        kz = quantile(coords[:, 3], range(0, 1, length=n_marginal_z))
    else # :range or if regular grid is required
        kx = collect(range(c_min[1], stop=c_max[1], length=n_marginal_x))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_marginal_y))
        kz = collect(range(c_min[3], stop=c_max[3], length=n_marginal_z))
    end

    B = zeros(Float64, n_obs, total_bins)

    if type in ["pspline", "bspline"]
        nbins_per_dim = [n_marginal_x, n_marginal_y, n_marginal_z]
        degree_val = get(kwargs, :degree, 3)
        degrees_per_dim = [degree_val, degree_val, degree_val]
        full_B = bstm_tensor_product_basis(coords, nbins_per_dim, degrees_per_dim; knot_method=knot_method)
        return full_B[:, 1:min(total_bins, size(full_B, 2))]
    elseif type == "wavelet"
        family = get(kwargs, :family, :db4)
        lengthscale = get(kwargs, :lengthscale, 0.1)
        nbins_per_dim = [n_marginal_x, n_marginal_y, n_marginal_z]
        full_B = bstm_tensor_product_wavelet_basis(coords, nbins_per_dim, family, lengthscale)
        return full_B[:, 1:min(total_bins, size(full_B, 2))]
    end

    if type in ["smooth", "barycentric", "linear"]
        hx = (c_max[1] - c_min[1]) / (n_marginal_x > 1 ? (n_marginal_x - 1) : 1); hx = hx > 0 ? hx : 1.0
        hy = (c_max[2] - c_min[2]) / (n_marginal_y > 1 ? (n_marginal_y - 1) : 1); hy = hy > 0 ? hy : 1.0
        hz = (c_max[3] - c_min[3]) / (n_marginal_z > 1 ? (n_marginal_z - 1) : 1); hz = hz > 0 ? hz : 1.0

        idx = 1
        for k_idx in 1:n_marginal_z, j_idx in 1:n_marginal_y, i_idx in 1:n_marginal_x
            if idx > total_bins; break; end
            b_x = max.(0.0, 1.0 .- abs.(coords[:, 1] .- kx[i_idx]) ./ hx)
            b_y = max.(0.0, 1.0 .- abs.(coords[:, 2] .- ky[j_idx]) ./ hy)
            b_z = max.(0.0, 1.0 .- abs.(coords[:, 3] .- kz[k_idx]) ./ hz)
            B[:, idx] .= b_x .* b_y .* b_z
            idx += 1
        end
    elseif type == "tps"
        centers = [(x, y, z) for z in kz for y in ky for x in kx][:]
        for m in 1:total_bins
            dx = coords[:, 1] .- centers[m][1]
            dy = coords[:, 2] .- centers[m][2]
            dz = coords[:, 3] .- centers[m][3]
            r = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
            B[:, m] .= r
        end
    elseif type == "rff"
        Omega = randn(3, total_bins)
        Omega[1, :] ./= ls_x; Omega[2, :] ./= ls_y; Omega[3, :] ./= ls_z
        Phi_phases = rand(total_bins) .* (2.0 * pi)
        B .= sqrt(2.0 / total_bins) .* cos.((coords * Omega) .+ Phi_phases')
    elseif type == "fft"
        nx = coords[:, 1] ./ ls_x
        ny = coords[:, 2] ./ ls_y
        nz = coords[:, 3] ./ ls_z
        idx = 1
        for mz in 1:n_marginal_z, my in 1:n_marginal_y, mx in 1:n_marginal_x
            if idx + 1 <= total_bins
                arg = mx .* nx .+ my .* ny .+ mz .* nz
                B[:, idx]   .= sin.(2.0 * pi * arg)
                B[:, idx+1] .= cos.(2.0 * pi * arg)
                idx += 2
            end
        end
    elseif type == "spherical"
        centers = [(x, y, z) for z in kz for y in ky for x in kx][:]
        range_r = get(kwargs, :range, mean(c_std))
        for m in 1:total_bins
            dx = coords[:, 1] .- centers[m][1]
            dy = coords[:, 2] .- centers[m][2]
            dz = coords[:, 3] .- centers[m][3]
            h = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2) ./ range_r
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end
    elseif type == "invdist"
        centers = [(x, y, z) for z in kz for y in ky for x in kx][:]
        for m in 1:total_bins
            dist_sq = (coords[:, 1] .- centers[m][1]).^2 .+ (coords[:, 2] .- centers[m][2]).^2 .+ (coords[:, 3] .- centers[m][3]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end
    elseif type == "kriging"
        centers = [(x,y,z) for z in kz for y in ky for x in kx][:]
        for m in 1:total_bins
            dist_sq = ((coords[:, 1] .- centers[m][1]).^2 ./ ls_x^2) .+ ((coords[:, 2] .- centers[m][2]).^2 ./ ls_y^2) .+ ((coords[:, 3] .- centers[m][3]).^2 ./ ls_z^2)
            B[:, m] .= exp.(-dist_sq ./ 2.0)
        end
    else
        B = ones(n_obs, total_bins)
    end

    return B[:, 1:min(total_bins, size(B, 2))]
end


function bstm_smooth_basis_4D(type::String, coords::AbstractMatrix, nbins::Union{Int, Vector{Int}}; W=nothing, knot_method::Symbol = :quantile, custom_knots::Union{Tuple{AbstractVector, AbstractVector, AbstractVector, AbstractVector}, Nothing} = nothing, kwargs...)
    n_obs = size(coords, 1)

    local n_marginal_1, n_marginal_2, n_marginal_3, n_marginal_4
    if nbins isa Int
        n_marginal_1 = nbins
        n_marginal_2 = nbins
        n_marginal_3 = nbins
        n_marginal_4 = nbins
    elseif nbins isa Vector{Int} && length(nbins) == 4
        n_marginal_1 = nbins[1]
        n_marginal_2 = nbins[2]
        n_marginal_3 = nbins[3]
        n_marginal_4 = nbins[4]
    else
        error("For a 4D smooth, `nbins` must be an Int or a Vector{Int} of length 4.")
    end
    total_bins = n_marginal_1 * n_marginal_2 * n_marginal_3 * n_marginal_4

    c_min = [minimum(coords[:, i]) for i in 1:4]
    c_max = [maximum(coords[:, i]) for i in 1:4]
    c_std = [std(coords[:, i]) for i in 1:4] .+ 1e-9

    ls_1 = get(kwargs, :ls_1, c_std[1])
    ls_2 = get(kwargs, :ls_2, c_std[2])
    ls_3 = get(kwargs, :ls_3, c_std[3])
    ls_4 = get(kwargs, :ls_4, c_std[4])

    local k1, k2, k3, k4

    use_regular_grid = type in ["invdist", "kriging", "tps", "spherical"]

    if knot_method == :custom && !isnothing(custom_knots)
        if length(custom_knots) != 4 || length(custom_knots[1]) != n_marginal_1 || length(custom_knots[2]) != n_marginal_2 || length(custom_knots[3]) != n_marginal_3 || length(custom_knots[4]) != n_marginal_4
            @warn "Custom knots for 4D smoother must be a Tuple of 4 AbstractVectors with lengths matching `nbins`. Falling back to :quantile method."
            k1 = quantile(coords[:, 1], range(0, 1, length=n_marginal_1))
            k2 = quantile(coords[:, 2], range(0, 1, length=n_marginal_2))
            k3 = quantile(coords[:, 3], range(0, 1, length=n_marginal_3))
            k4 = quantile(coords[:, 4], range(0, 1, length=n_marginal_4))
        else
            k1 = custom_knots[1]
            k2 = custom_knots[2]
            k3 = custom_knots[3]
            k4 = custom_knots[4]
        end
    elseif knot_method == :quantile && !use_regular_grid
        k1 = quantile(coords[:, 1], range(0, 1, length=n_marginal_1)); k2 = quantile(coords[:, 2], range(0, 1, length=n_marginal_2)); k3 = quantile(coords[:, 3], range(0, 1, length=n_marginal_3)); k4 = quantile(coords[:, 4], range(0, 1, length=n_marginal_4))
    else # :range or if regular grid is required
        k1 = collect(range(c_min[1], stop=c_max[1], length=n_marginal_1)); k2 = collect(range(c_min[2], stop=c_max[2], length=n_marginal_2)); k3 = collect(range(c_min[3], stop=c_max[3], length=n_marginal_3)); k4 = collect(range(c_min[4], stop=c_max[4], length=n_marginal_4))
    end

    B = zeros(Float64, n_obs, total_bins)

    if type in ["pspline", "bspline"]
        nbins_per_dim = [n_marginal_1, n_marginal_2, n_marginal_3, n_marginal_4]
        degree_val = get(kwargs, :degree, 3)
        degrees_per_dim = fill(degree_val, 4)
        full_B = bstm_tensor_product_basis(coords, nbins_per_dim, degrees_per_dim; knot_method=knot_method)
        return full_B[:, 1:min(total_bins, size(full_B, 2))]
    elseif type == "wavelet"
        family = get(kwargs, :family, :db4)
        lengthscale = get(kwargs, :lengthscale, 0.1)
        nbins_per_dim = [n_marginal_1, n_marginal_2, n_marginal_3, n_marginal_4]
        full_B = bstm_tensor_product_wavelet_basis(coords, nbins_per_dim, family, lengthscale)
        return full_B[:, 1:min(total_bins, size(full_B, 2))]
    end

    if type == "barycentric"
        knot_points = [Point4D(x, y, z, t) for t in k4 for z in k3 for y in k2 for x in k1]
        full_B = bstm_barycentric_basis_4D(coords, knot_points, n_marginal_1) # Assumes cubic grid
        return full_B[:, 1:min(total_bins, size(full_B, 2))]
    end

    if type in [ "smooth", "linear"]
        hx1 = (c_max[1] - c_min[1]) / (n_marginal_1 > 1 ? (n_marginal_1 - 1) : 1); hx1 = hx1 > 0 ? hx1 : 1.0
        hx2 = (c_max[2] - c_min[2]) / (n_marginal_2 > 1 ? (n_marginal_2 - 1) : 1); hx2 = hx2 > 0 ? hx2 : 1.0
        hx3 = (c_max[3] - c_min[3]) / (n_marginal_3 > 1 ? (n_marginal_3 - 1) : 1); hx3 = hx3 > 0 ? hx3 : 1.0
        hx4 = (c_max[4] - c_min[4]) / (n_marginal_4 > 1 ? (n_marginal_4 - 1) : 1); hx4 = hx4 > 0 ? hx4 : 1.0

        idx = 1
        for l_idx in 1:n_marginal_4, k_idx in 1:n_marginal_3, j_idx in 1:n_marginal_2, i_idx in 1:n_marginal_1
            if idx > total_bins; break; end
            b1 = max.(0.0, 1.0 .- abs.(coords[:, 1] .- k1[i_idx]) ./ hx1)
            b2 = max.(0.0, 1.0 .- abs.(coords[:, 2] .- k2[j_idx]) ./ hx2)
            b3 = max.(0.0, 1.0 .- abs.(coords[:, 3] .- k3[k_idx]) ./ hx3)
            b4 = max.(0.0, 1.0 .- abs.(coords[:, 4] .- k4[l_idx]) ./ hx4)
            B[:, idx] .= b1 .* b2 .* b3 .* b4
            idx += 1
        end

    elseif type == "tps"
        centers = [(k1_i, k2_i, k3_i, k4_i) for k4_i in k4 for k3_i in k3 for k2_i in k2 for k1_i in k1][:]

        for m in 1:total_bins
            d1 = coords[:, 1] .- centers[m][1]
            d2 = coords[:, 2] .- centers[m][2]
            d3 = coords[:, 3] .- centers[m][3]
            d4 = coords[:, 4] .- centers[m][4]
            r = sqrt.(d1.^2 .+ d2.^2 .+ d3.^2 .+ d4.^2)
            B[:, m] .= log.(r .+ 1e-9)
        end

    elseif type == "rff"
        Omega = randn(4, total_bins)
        Omega[1, :] ./= ls_1; Omega[2, :] ./= ls_2; Omega[3, :] ./= ls_3; Omega[4, :] ./= ls_4
        Phi_phases = rand(total_bins) .* (2.0 * pi)
        B .= sqrt(2.0 / total_bins) .* cos.((coords * Omega) .+ Phi_phases')

    elseif type == "fft"
        nx1 = coords[:, 1] ./ ls_1
        nx2 = coords[:, 2] ./ ls_2
        nx3 = coords[:, 3] ./ ls_3
        nx4 = coords[:, 4] ./ ls_4
        idx = 1
        for m4 in 1:n_marginal_4, m3 in 1:n_marginal_3, m2 in 1:n_marginal_2, m1 in 1:n_marginal_1
            if idx + 1 <= total_bins
                arg = m1 .* nx1 .+ m2 .* nx2 .+ m3 .* nx3 .+ m4 .* nx4
                B[:, idx]   .= sin.(2.0 * pi * arg)
                B[:, idx+1] .= cos.(2.0 * pi * arg)
                idx += 2
            end
        end
 
    elseif type == "spherical"
        centers = [(k1_i, k2_i, k3_i, k4_i) for k4_i in k4 for k3_i in k3 for k2_i in k2 for k1_i in k1][:]
        range_r = get(kwargs, :range, mean(c_std))

        for m in 1:total_bins
            d1 = coords[:, 1] .- centers[m][1]
            d2 = coords[:, 2] .- centers[m][2]
            d3 = coords[:, 3] .- centers[m][3]
            d4 = coords[:, 4] .- centers[m][4]
            h = sqrt.(d1.^2 .+ d2.^2 .+ d3.^2 .+ d4.^2) ./ range_r
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end
 
    elseif type == "invdist"
        centers = [(w,x,y,z) for w in k1, x in k2, y in k3, z in k4][:]
        for m in 1:total_bins
            dist_sq = (coords[:, 1] .- centers[m][1]).^2 .+ (coords[:, 2] .- centers[m][2]).^2 .+ (coords[:, 3] .- centers[m][3]).^2 .+ (coords[:, 4] .- centers[m][4]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end

    elseif type == "kriging"
        centers = [(w,x,y,z) for w in k1, x in k2, y in k3, z in k4][:]
        for m in 1:total_bins
            dist_sq = ((coords[:, 1] .- centers[m][1]).^2 ./ ls_1^2) .+ ((coords[:, 2] .- centers[m][2]).^2 ./ ls_2^2) .+ ((coords[:, 3] .- centers[m][3]).^2 ./ ls_3^2) .+ ((coords[:, 4] .- centers[m][4]).^2 ./ ls_4^2)
            B[:, m] .= exp.(-dist_sq ./ 2.0)
        end

    else
        B = ones(n_obs, total_bins)
    end

    return B[:, 1:min(total_bins, size(B, 2))]
end

"""
    evaluate_kernel_matrix(coords::AbstractMatrix, param_val::Real, ls::Union{Real, AbstractVector}, kernel_type::Symbol, noise::Real; wavelet_levels=3)

Computes the covariance kernel matrix for a given set of coordinates.

# Rationale
This function is a core utility for all GP-based components and is not deprecated.
It is designed to be AD-friendly by using type promotion and `convert` to handle
numeric types dynamically, which is crucial for compatibility with `ForwardDiff.jl`.
It supports both isotropic and ARD (Automatic Relevance Determination) kernels by
accepting a scalar or vector `lengthscale`. This version is consistent with the
refactored architecture.

# Arguments
- `coords::AbstractMatrix`: An `N x D` matrix of data points.
- `param_val::Real`: The signal variance (σ²) of the kernel.
- `ls::Union{Real, AbstractVector}`: The lengthscale(s) of the kernel.
- `kernel_type::Symbol`: The type of kernel to evaluate (e.g., `:se`, `:matern32`).
- `noise::Real`: A small jitter or nugget term added to the diagonal for numerical stability.
- `wavelet_levels`: The number of levels for the wavelet kernel.

# Returns
- A dense `N x N` covariance matrix.
"""
function evaluate_kernel_matrix(coords::AbstractMatrix, param_val::Real, ls::Union{Real, AbstractVector}, kernel_type::Symbol, noise::Real; wavelet_levels=3)
    T = promote_type(eltype(coords), typeof(param_val), eltype(ls), typeof(noise))
    coords_T = convert(AbstractMatrix{T}, coords)
    ls_T = convert(typeof(ls) <: Real ? T : AbstractVector{T}, ls)

    dist_sq = if ls isa AbstractVector # ARD case
        if size(coords_T, 2) != length(ls_T)
            error("Dimension mismatch for ARD kernel: Number of coordinate dimensions ($(size(coords_T, 2))) does not match number of lengthscales ($(length(ls_T))).")
        end
        # Calculate weighted squared Euclidean distance
        pairwise(SqEuclidean(), coords_T ./ ls_T', dims=1)
    else # Isotropic case
        pairwise(SqEuclidean(), coords_T, dims=1) ./ ls_T^2
    end

    # Gaussian / Squared Exponential
    if kernel_type == :gaussian || kernel_type == :se
        return param_val^2 .* exp.(-one(T)/2 .* dist_sq) .+ (noise * I)
    
    # Exponential / Matern 1/2
    elseif kernel_type == :exponential || kernel_type == :matern12 
        d = sqrt.(dist_sq)
        return param_val^2 .* exp.(-d) .+ (noise * I)
    
    # Matern 3/2
    elseif kernel_type == :matern32 
        d = sqrt.(dist_sq)
        val = sqrt(convert(T, 3.0)) .* d
        return param_val^2 .* (one(T) .+ val) .* exp.(-val) .+ (noise * I)
    
    # Matern 5/2
    elseif kernel_type == :matern52 
        d = sqrt.(dist_sq)
        val = sqrt(convert(T, 5.0)) .* d
        return param_val^2 .* (one(T) .+ val .+ (val.^2 ./ convert(T, 3.0))) .* exp.(-val) .+ (noise * I)

    # Constant Kernel (Identity innovation)
    elseif kernel_type == :constant
        return fill(convert(T, param_val^2), size(dist_sq))

    # Linear Kernel
    elseif kernel_type == :linear
        return param_val^2 .* dist_sq

    # Wavelet Multiscale Kernel
    elseif kernel_type == :wavelet
        K_accum = zeros(T, size(dist_sq))
        for wv_scale in 1:wavelet_levels
            ls_scale_sq = (ls isa Real ? ls_T^2 : one(T)) / (convert(T, 4.0)^(wv_scale-1))
            weight_scale = param_val^2 * exp(convert(T, -wv_scale) / ls_T)
            K_accum .+= weight_scale .* exp.(-one(T)/2 .* dist_sq ./ ls_scale_sq)
        end
        return K_accum + (noise * I)

    # Fallback Dispatch
    else
        return param_val^2 .* exp.(-one(T)/2 .* dist_sq) .+ (noise * I)
    end
end



"""
    recompose_precision(m_type::Symbol, template_s::AbstractMatrix, param_val::Real; extra_param=nothing, noise=1e-4, kwargs...)

An internal factory function that constructs a final precision matrix from a template
and sampled hyperparameters. This is a core part of the dynamic model generation.

# Rationale
This function is not deprecated and is essential for components whose precision
structure is not static (e.g., depends on a `rho` or `kappa` parameter). It is
designed to be AD-friendly by using type promotion and avoiding hard-coded types.
This version is consistent with the refactored architecture and includes detailed
comments clarifying the mathematical formulation for each model type.

# Arguments
- `m_type::Symbol`: The symbol representing the model type (e.g., `:leroux`, `:spde`).
- `template_s::AbstractMatrix`: The parameter-free structure matrix (e.g., a graph Laplacian).
- `param_val::Real`: A generic parameter, typically the scale `sigma`, though its role can vary.
- `extra_param`: An additional hyperparameter, typically `rho` or `kappa`.
- `noise`: A small jitter term for numerical stability.
- `kwargs...`: Additional keyword arguments for specific models (e.g., `flow_direction`).

# Returns
- A `Symmetric` precision matrix.
"""
function recompose_precision(m_type::Symbol, template_s::AbstractMatrix, param_val::Real; extra_param=nothing, noise=1e-4, kwargs...)
    n_s = size(template_s, 1)
    T_num = promote_type(typeof(param_val), typeof(noise), eltype(template_s), typeof(extra_param))

    if m_type == :SPDE
        # For SPDE, Q = (κ²I + G)ᵀ(κ²I + G), where G is the geometric structure matrix.
        kappa = isnothing(extra_param) ? one(T_num) : extra_param
        Q_kappa = if kappa isa Real
            kappa^2 * I(n_s) # UniformScaling promotes correctly
        else
            if length(kappa) != n_s; error("Anisotropic kappa vector length must match number of spatial units."); end
            Diagonal(kappa.^2) # Let promotion handle eltype
        end
        L_spde = Q_kappa + template_s
        return Symmetric(L_spde' * L_spde)
    end

    if m_type == :None || m_type == :FIXED
        return Symmetric(sparse(I, n_s, n_s))
    end

    if m_type == :Besag || m_type == :ICAR || m_type == :Cyclic
        # These are intrinsic models where the structure is fixed.
        return Symmetric(template_s)
    end

    if m_type == :AR1
        # Precision for a stationary AR(1) process.
        rho = isnothing(extra_param) ? zero(T_num) : extra_param
        Q = spdiagm(0 => fill(one(T_num) + rho^2, n_s))
        if n_s > 0
            Q[1, 1] = one(T_num)
            Q[n_s, n_s] = one(T_num)
        end
        Q .+= rho .* template_s # template_s is the adjacency matrix for the time series
        return Symmetric(Q)
    end

    if m_type == :RW1 || m_type == :RW2
        error("recompose_precision should not be called for $(m_type) models. Use the state-space implementation.")
    end

    if m_type == :Leroux || m_type == :LocalAdaptive
        # Leroux precision: Q = ρ*Q_icar + (1-ρ)*I
        lambda_val = isnothing(extra_param) ? convert(T_num, 0.5) : extra_param
        I_prom = spdiagm(0 => ones(T_num, n_s))
        return Symmetric(lambda_val .* template_s + (one(T_num) - lambda_val) .* I_prom)
    end

    if m_type == :ST_I || m_type == :ST_II || m_type == :ST_III || m_type == :ST_IV
        error("recompose_precision should not be called for ST_I/II/III/IV models directly.")
    end

    if m_type == :NetworkFlow
        # SAR-like precision for directed graphs: Q = (I - ρW)ᵀ(I - ρW)
        rho_net = isnothing(extra_param) ? convert(T_num, 0.8) : extra_param
        W_net = template_s
        flow_direction = get(kwargs, :flow_direction, :bidirectional)
        
        L_op = if flow_direction == :upstream
            I(n_s) - rho_net .* W_net'
        elseif flow_direction == :downstream
            I(n_s) - rho_net .* W_net
        else # :bidirectional or default
            # Symmetrize for bidirectional flow.
            W_symm = sparse((W_net + W_net') .> 0)
            I(n_s) - rho_net .* W_symm
        end
        return Symmetric(L_op' * L_op)
    end

    if m_type == :SAR || m_type == :DAG
        # Standard SAR precision: Q = (I - ρW)ᵀ(I - ρW)
        rho_p = isnothing(extra_param) ? convert(T_num, 0.8) : extra_param
        L_op = I(n_s) - rho_p .* template_s
        return Symmetric(L_op' * L_op)
    end

    if m_type == :GP
        # For a full GP, the precision is the inverse of the covariance matrix.
        ls = isnothing(extra_param) ? one(T_num) : extra_param
        K = param_val^2 .* exp.(-(Matrix(template_s).^2) ./ (convert(T_num, 2.0) * ls^2 + convert(T_num, noise)))
        return inv(Symmetric(K))
    end

    if m_type == :RFF || m_type == :FFT || m_type == :BSpline || m_type == :PSpline || m_type == :TPS
        # For basis function models, the template is the penalty matrix on the coefficients.
        return Symmetric(template_s)
    end

    # Fallback for any other case.
    return Symmetric(template_s)
end


"""
    _distribution_to_string(d::Distribution)

Converts a `Distribution` object into a type-stable string representation of its
constructor call, suitable for dynamic code generation within a Turing `@model`.

# Rationale
This function is a core utility for the `bstm` code generation engine and is not
deprecated. It ensures that prior distributions defined as Julia objects can be
correctly translated into AD-compatible code.

Key features of this implementation:
1.  **AD Compatibility**: It generates constructor strings like `Normal{T}(...)`, where
    `T` is the generic numeric type used by the Turing model. This allows the model
    to work seamlessly with automatic differentiation libraries like `ForwardDiff.jl`,
    which use `Dual` numbers.
2.  **Robustness**: It uses accessor functions like `meanlog` and `stdlog` where
    possible, making the code less dependent on the internal structure of `Distribution`
    objects.
3.  **Completeness**: It handles a wide range of univariate distributions, as well as
    wrapper types like `Truncated`, `Product`, and `Fill`, which are common in complex
    Bayesian models.
4.  **Efficiency**: For `Product` distributions containing identical inner distributions,
    it generates a more efficient `filldist` call.

# Version
v1.0.2 (2026-08-09)

# Arguments
- `d::Distribution`: The distribution object to convert.

# Returns
- `String`: A string representing the constructor call for the distribution.
"""
function _distribution_to_string(d::Distribution)
    dist_name = string(typeof(d).name.name)
    if d isa Exponential
        return "$(dist_name){T}($(rate(d)))"
    elseif d isa Normal
        return "$(dist_name){T}($(mean(d)), $(std(d)))"
    elseif d isa LogNormal
        # Use accessors for robustness instead of params(d)
        return "$(dist_name){T}($(meanlog(d)), $(stdlog(d)))"
    elseif d isa Beta
        return "$(dist_name){T}($(params(d)[1]), $(params(d)[2]))"
    elseif d isa InverseGamma
        return "$(dist_name){T}($(Distributions.shape(d)), $(Distributions.scale(d)))"
    elseif d isa Gamma
        return "$(dist_name){T}($(Distributions.shape(d)), $(Distributions.scale(d)))"
    elseif d isa Uniform
        return "$(dist_name){T}($(minimum(d)), $(maximum(d)))"
    elseif d isa TDist
        return "$(dist_name){T}($(d.df))"
    elseif d isa Dirac
        return "$(dist_name){T}($(d.value))"
    elseif d isa Truncated
        inner_dist_str = _distribution_to_string(d.untruncated)
        # Handle potential -Inf/Inf which don't need type casting
        lower_str = isinf(d.lower) ? string(d.lower) : "T($(d.lower))"
        upper_str = isinf(d.upper) ? string(d.upper) : "T($(d.upper))"
        return "truncated($(inner_dist_str), $(lower_str), $(upper_str))"
    elseif d isa Product
        if hasproperty(d, :v) && d.v isa Fill
            inner_dist_str = _distribution_to_string(d.v.value)
            n = length(d.v)
            return "filldist($(inner_dist_str), $(n))"
        elseif hasproperty(d, :v) && all(dist -> dist == d.v[1], d.v)
             inner_dist_str = _distribution_to_string(d.v[1])
             n = length(d.v)
             return "filldist($(inner_dist_str), $(n))"
        else
            inner_strs = [_distribution_to_string(dist) for dist in d.v]
            return "Product([$(join(inner_strs, ", "))])"
        end
    elseif d isa Fill
        inner_dist_str = _distribution_to_string(d.value)
        n = length(d)
        return "filldist($(inner_dist_str), $(n))"
    else
        @warn "String conversion for distribution type `$(typeof(d))` is not explicitly handled. The generated code may not be type-stable."
        return string(d)
    end
end


  
"""
    bstm_sample(model, sampler, n_samples; kwargs...)

A wrapper around `Turing.sample` that suppresses world-age warnings by temporarily
redirecting `stderr` to `devnull`.

# Rationale
Dynamically generated models from `@bstm` can trigger non-fatal world-age warnings
when interacting with pre-compiled library functions inside Turing.jl. This wrapper
provides a clean way to run the sampler without printing these warnings to the console.

# Arguments
- `model`: The Turing model object.
- `sampler`: The MCMC sampler to use.
- `n_samples`: The number of samples to draw.
- `kwargs...`: Additional keyword arguments passed directly to `Turing.sample`.

# Returns
- The MCMC chain object returned by `Turing.sample`.

# Example
```julia
m = @bstm(likelihood(y) ~ 1, data)
chn = bstm_sample(m, NUTS(), 1000)
```
"""
function bstm_sample(model, sampler, n_samples; kwargs...)
    local chain
    redirect_stderr(devnull) do
        chain = sample(model, sampler, n_samples; kwargs...)
    end
    return chain
end


"""
    _generate_likelihood_section(M::NamedTuple, is_multivariate::Bool)

Generates Turing code for all likelihood-specific priors.

# Rationale for Update
This function is updated to be fully consistent with the refactored architecture by
including the necessary prior definitions for ordinal regression models. The previous
version was missing the logic to generate priors for the ordered cut-points (`alphas`)
and the degrees of freedom (`df`) for the latent Student's T distribution. This
omission would cause any ordinal model to fail.

This corrected version now:
1.  Checks if an `ordinal` family is specified in the likelihood.
2.  If so, it generates the priors for the first cut-point (`ordinal_alpha_raw_1`) and
    the positive differences for subsequent cut-points (`ordinal_alpha_diffs`), which
    is a standard parameterization to enforce ordering.
3.  It also generates the prior for the degrees of freedom (`ordinal_df`) if the
    latent distribution for the ordinal model is Student's T.
4.  Retains all existing logic for other likelihood-specific parameters (e.g., `r_nb`,
    `y_sigma`, `L_corr`).

This change ensures that ordinal models are correctly specified and aligns the function
with the full feature set of the `bstm` framework.
"""
function _generate_likelihood_section(M::NamedTuple, is_multivariate::Bool)
    families = [string(get(spec, :family, "gaussian")) for spec in M.likelihood_specs]
    
    prior_blocks = String[]

    # Prior for Negative Binomial dispersion
    if any(f -> f == "negbin", families)
        push!(prior_blocks, "r_nb ~ NamedDist(Exponential(1.0), :r_nb)")
    end

    # Prior for Zero-Inflation or Hurdle probability
    if get(M, :user_provided_hurdle, false)
        push!(prior_blocks, "lik_phi_hurdle ~ NamedDist(Beta(1,1), :lik_phi_hurdle)")
    elseif get(M, :use_zi, false)
        push!(prior_blocks, "lik_phi_zi ~ NamedDist(Beta(1,1), :lik_phi_zi)")
    end

    # Prior for Student's T degrees of freedom
    if any(f -> f == "student_t", families)
        push!(prior_blocks, "lik_nu_student_t ~ NamedDist(Exponential(1.0), :lik_nu_student_t)")
    end

    # Prior for observation standard deviation (for Gaussian-like families)
    if any(f -> f in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"], families)
        y_sigma_prior_str = _distribution_to_string(Exponential(1.0))
        if is_multivariate
            push!(prior_blocks, "y_sigma ~ NamedDist(filldist($(y_sigma_prior_str), K), :y_sigma)")
        else
            push!(prior_blocks, "y_sigma ~ NamedDist($(y_sigma_prior_str), :y_sigma)")
        end
    end
    
    # Prior for extra parameters (e.g., Gamma shape, Beta precision)
    if any(f -> f in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"], families)
        push!(prior_blocks, "lik_extra_params ~ NamedDist(Exponential(1.0), :lik_extra_params)")
    end

    # Prior for multivariate correlation matrix
    if is_multivariate
        push!(prior_blocks, "L_corr ~ NamedDist(LKJCholesky(K, 1.0), :L_corr)")
    end

    # --- Priors for Ordinal Model ---
    ordinal_spec_idx = findfirst(s -> string(get(s, :family, "")) == "ordinal", M.likelihood_specs)
    if !isnothing(ordinal_spec_idx)
        spec = M.likelihood_specs[ordinal_spec_idx]
        K = get(spec, :K, 0)
        latent_dist = get(spec, :latent_dist, :logistic)

        if K > 2
            # Prior for the first cut-point and the positive differences for subsequent cut-points.
            push!(prior_blocks, "ordinal_alpha_raw_1 ~ NamedDist(Normal(0, 5), :ordinal_alpha_raw_1)")
            push!(prior_blocks, "ordinal_alpha_diffs ~ NamedDist(filldist(Exponential(1.0), $(K - 2)), :ordinal_alpha_diffs)")
        elseif K == 2
            # For a binary ordinal model, only one cut-point is needed.
            push!(prior_blocks, "ordinal_alpha_raw_1 ~ NamedDist(Normal(0, 5), :ordinal_alpha_raw_1)")
        end

        # Prior for the degrees of freedom if using a Student's T latent distribution.
        if latent_dist == :student_t
            push!(prior_blocks, "ordinal_df ~ NamedDist(Exponential(1.0), :ordinal_df)")
        end
    end

    return join(prior_blocks, "\n    ")
end


"""
    _generate_univariate_likelihood_block(M::NamedTuple)

Generates the likelihood block for a univariate model.

# Rationale
This function is a core part of the code generation pipeline and is not deprecated.
It is consistent with the refactored architecture, which requires a clear separation
between the linear predictor assembly and the final likelihood evaluation.

This function correctly implements the following key features:
1.  **Correct API Usage**: It generates code that uses the refactored
    `bstm_Likelihood(eta, ...)` constructor, passing the linear predictor `eta` as
    the main parameter and the observed data `y` to the `logpdf` function.
2.  **Dynamic Keyword Arguments**: It dynamically builds the keyword arguments for the
    `bstm_Likelihood` constructor, ensuring that optional parameters (e.g., `sigma_y`,
    `r_nb`, `trial`, `weight`, `censor_lower`, `censor_upper`, `hurdle`, `phi_zi`,
    `extra_params`) are only included when required by the specified likelihood family
    and enabled in the model configuration.
3.  **AD-Safety**: The generated code is compatible with automatic differentiation as it
    correctly separates model parameters from fixed data.

# Arguments
- `M`: The main model configuration `NamedTuple`.

# Returns
- A `String` containing the generated Turing code for the univariate likelihood block.
"""
function _generate_univariate_likelihood_block(M::NamedTuple)
    family = string(M.likelihood_specs[1][:family])
    
    # Determine which optional parameters are needed for this family
    needs_sigma = family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
    needs_rnb = family == "negbin"
    needs_nu = family == "student_t"
    needs_extra = family in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"]

    # Dynamically build the keyword arguments string
    kwargs_parts = String[]
    if needs_sigma; push!(kwargs_parts, "sigma_y=y_sigma"); end
    if needs_rnb; push!(kwargs_parts, "r_nb=r_nb"); end
    if get(M, :user_provided_trials, false); push!(kwargs_parts, "trial=Int(M.trials[i, 1])"); end
    if get(M, :user_provided_weights, false); push!(kwargs_parts, "weight=M.weights[i, 1]"); end
    if get(M, :user_provided_censor_lower, false); push!(kwargs_parts, "censor_lower=M.censor_lower[i, 1]"); end
    if get(M, :user_provided_censor_upper, false); push!(kwargs_parts, "censor_upper=M.censor_upper[i, 1]"); end
    
    if get(M, :user_provided_hurdle, false)
        push!(kwargs_parts, "hurdle=M.hurdle[i, 1]")
        push!(kwargs_parts, "phi_hurdle=lik_phi_hurdle")
    elseif get(M, :use_zi, false)
        push!(kwargs_parts, "phi_zi=lik_phi_zi")
    end

    extra_param_logic = ""
    if needs_nu || needs_extra
        extra_param_logic = if needs_nu; "local extra_p = lik_nu_student_t"
        else; "local extra_p = lik_extra_params"; end
        push!(kwargs_parts, "extra_params=extra_p")
    end

    kwargs_str = join(kwargs_parts, ", ")

    return """
    family = M.likelihood_specs[1][:family]
    $(extra_param_logic)
    for i in 1:M.y_N
        # Construct distribution with the linear predictor `eta[i]`
        d_lik = bstm_Likelihood(family, eta[i]; $(kwargs_str))
        
        # Evaluate logpdf at the corresponding observation `M.y_obs[i]`
        Turing.@addlogprob! Distributions.logpdf(d_lik, M.y_obs[i])
    end
    """
end



"""
    _generate_multivariate_likelihood_block(M::NamedTuple)

Generates the likelihood block for a multivariate model.

# Rationale for Update
This version corrects and completes the likelihood generation for multivariate models.
The previous implementation was incomplete, missing support for several likelihood
modifications like weights, censoring, and hurdles. It also contained a bug where
`extra_params` (e.g., for Student's T or Gamma distributions) were not correctly
included in the generated code.

This updated function now:
1.  **Dynamically Builds All Keyword Arguments**: It iterates through all possible
    likelihood parameters (`sigma_y`, `r_nb`, `trial`, `weight`, `censor_lower`,
    `censor_upper`, `hurdle`, `phi_zi`, `extra_params`) and includes them in the
    `bstm_Likelihood` constructor only if required by the specific likelihood family
    and enabled in the model configuration.
2.  **Fixes `extra_params` Bug**: The logic for including `extra_params` has been
    corrected to ensure it is properly added to the list of keyword arguments before
    the code string is generated.
3.  **Improves Readability**: The code generation is refactored to build an array of
    likelihood blocks, which are then joined, improving clarity over repeated string
    concatenation.
4.  **Removes `local` Keyword**: Aligns with the project's coding standards by removing
    unnecessary `local` variable declarations.

This ensures that the generated code is robust, correct, and fully supports the
feature set of the `bstm` framework.
"""
function _generate_multivariate_likelihood_block(M::NamedTuple)
    # This block handles the case where all outcomes are categories of a single multinomial response.
    if any(s -> s[:family] == "dirichlet_multinomial", M.likelihood_specs)
        return """
        eta_correlated = eta_latent * L_corr.L
        family = M.likelihood_specs[1][:family]
        for i in 1:M.y_N
            # For each observation, y_obs[i, :] is the vector of counts across categories.
            # The total number of trials is the sum of counts for that observation.
            d_lik = bstm_Likelihood(family, eta_correlated[i, :]; trial=Int(sum(M.y_obs[i, :])))
            Turing.@addlogprob! Distributions.logpdf(d_lik, M.y_obs[i, :])
        end
        """
    end

    # This block handles multiple, independent univariate-style outcomes.
    loop_body_parts = String[]
    for k in 1:M.outcomes_N
        family = string(M.likelihood_specs[k][:family])
        needs_sigma = family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
        needs_rnb = family == "negbin"
        needs_nu = family == "student_t"
        needs_extra = family in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"]

        kwargs_parts = String[]
        if needs_sigma; push!(kwargs_parts, "sigma_y=y_sigma[$k]"); end
        if needs_rnb; push!(kwargs_parts, "r_nb=r_nb"); end # Assumes r_nb is a vector if multivariate
        if get(M, :user_provided_trials, false); push!(kwargs_parts, "trial=Int(M.trials[i, $k])"); end
        if get(M, :user_provided_weights, false); push!(kwargs_parts, "weight=M.weights[i, $k]"); end
        if get(M, :user_provided_censor_lower, false); push!(kwargs_parts, "censor_lower=M.censor_lower[i, $k]"); end
        if get(M, :user_provided_censor_upper, false); push!(kwargs_parts, "censor_upper=M.censor_upper[i, $k]"); end
        
        if get(M, :user_provided_hurdle, false)
            push!(kwargs_parts, "hurdle=M.hurdle[i, $k]")
            push!(kwargs_parts, "phi_hurdle=lik_phi_hurdle")
        elseif get(M, :use_zi, false)
            push!(kwargs_parts, "phi_zi=lik_phi_zi")
        end
        
        extra_param_logic = ""
        if needs_nu || needs_extra
             extra_param_logic = if needs_nu; "extra_p = lik_nu_student_t"
             else; "extra_p = lik_extra_params"; end
             push!(kwargs_parts, "extra_params=extra_p")
        end

        kwargs_str = join(kwargs_parts, ", ")

        outcome_block = """
        # Likelihood for outcome $(k)
        let family_k = M.likelihood_specs[$k][:family]
            $(extra_param_logic)
            for i in 1:M.y_N
                d_lik = bstm_Likelihood(family_k, eta_correlated[i, $k]; $(kwargs_str))
                Turing.@addlogprob! Distributions.logpdf(d_lik, M.y_obs[i, $k])
            end
        end
        """
        push!(loop_body_parts, outcome_block)
    end

    loop_body = join(loop_body_parts, "\n\n")

    return """
    eta_correlated = eta_latent * L_corr.L
    $(loop_body)
    """
end



 
"""
    _generate_final_likelihood_block(M::NamedTuple, is_multivariate::Bool)

Generates the final likelihood block for the Turing model, dispatching to the
appropriate helper based on the model architecture and handling special cases like
ordinal models and custom component likelihoods.

# Rationale
This function is a core part of the code generation pipeline and is not deprecated.
It ensures that the correct likelihood is generated for the specified model, including
the complex logic required for ordinal regression with proportional and non-proportional
effects. This version improves the clarity of the generated code for ordinal models by
removing unnecessary `local` keywords and adding comments, aligning it with the
refactoring's goal of improved readability.

# Arguments
- `M`: The main model configuration `NamedTuple`.
- `is_multivariate`: A boolean indicating if the model is multivariate.

# Returns
- A `String` containing the generated Turing code for the likelihood block.
"""
function _generate_final_likelihood_block(M::NamedTuple, is_multivariate::Bool)
    # Purpose: Top-level dispatcher for likelihood code generation.
    # Rationale: This function routes to the correct generator based on model architecture
    #            and handles special cases like custom likelihoods from components.
    
    # Check if a component like LGCP handles its own likelihood.
    has_custom_likelihood_from_component = any(spec -> any(T -> spec.component_obj isa T, [LGCP, LogGammaCoxProcess, ShotNoiseCoxProcess]), M.components)
    if has_custom_likelihood_from_component
        return "" # The component's `get_updates` method will add the log-probability.
    end

    if is_multivariate
        return _generate_multivariate_likelihood_block(M)
    else
        # For univariate models, check for special families like ordinal.
        family = string(M.likelihood_specs[1][:family])
        
        if family == "ordinal"
            K = M.likelihood_specs[1][:K]
            latent_dist_val = M.likelihood_specs[1][:latent_dist]
            if K < 2; return ""; end

            non_prop_terms = get(M, :non_proportional_effects, Symbol[])
            is_npo = !isempty(non_prop_terms)
            
            npo_indices = findall(x -> x in non_prop_terms, M.Xfixed_names)
            n_npo_vars = length(npo_indices)

            assignment_lines = ""
            for j in 1:(K-1)
                assignment_lines *= "ordinal_alpha_$(j) = alphas_computed[$(j)]\n"
            end

            npo_update_block = ""
            if is_npo && n_npo_vars > 0
                npo_update_block = """
                # Non-proportional effects calculation
                X_npo = M.Xfixed[:, $(npo_indices)]
                beta_npo_matrix = reshape(beta_npo, $(n_npo_vars), $(K-1))
                eta_npo = X_npo * beta_npo_matrix
                """
            end

            return """
            # Proportional Odds Likelihood
            let
                # Reconstruct the ordered cut-points from their raw parameters.
                alphas_computed = if $(K > 2)
                    cumsum([ordinal_alpha_raw_1; ordinal_alpha_diffs])
                else
                    [ordinal_alpha_raw_1]
                end

                $(assignment_lines)
                latent_dist_symbol = :$(latent_dist_val)
                $(npo_update_block)

                for i in 1:M.y_N
                    # The proportional part of the linear predictor for this observation.
                    linear_predictor_prop = eta[i]
                    
                    # The full linear predictor for each category's threshold.
                    linear_predictor_vec = if $(is_npo && n_npo_vars > 0)
                        # Combine proportional and non-proportional parts for each cut-point.
                        linear_predictor_prop .+ view(eta_npo, i, :)
                    else
                        # If fully proportional, broadcast the single predictor.
                        fill(linear_predictor_prop, $(K-1))
                    end
                    
                    # Calculate cumulative probabilities using the inverse link function.
                    cumulative_probs = if latent_dist_symbol == :normal
                        Distributions.cdf.(Normal(), alphas_computed .- linear_predictor_vec)
                    elseif latent_dist_symbol == :logistic
                        logistic.(alphas_computed .- linear_predictor_vec)
                    elseif latent_dist_symbol == :student_t
                        Distributions.cdf.(TDist(ordinal_df), alphas_computed .- linear_predictor_vec)
                    else
                        error("Unsupported latent distribution ':\$(latent_dist_symbol)' for ordinal model.")
                    end
                    
                    # Convert cumulative probabilities to individual category probabilities.
                    probs = Vector{T}(undef, $(K))
                    if $(K > 1)
                        probs[1] = cumulative_probs[1]
                        for j in 2:($(K-1))
                            probs[j] = max(0.0, cumulative_probs[j] - cumulative_probs[j-1])
                        end
                        probs[$(K)] = max(0.0, 1.0 - cumulative_probs[$(K-1)])
                    else
                        probs[1] = 1.0
                    end

                    # Normalize to ensure probabilities sum to 1, for numerical stability.
                    probs ./= (sum(probs) + 1e-9)
                    
                    # Add the log-probability of the observed category.
                    Turing.@addlogprob! logpdf(Categorical(probs), M.y_obs[i])
                end
            end
            """
        else
            return _generate_univariate_likelihood_block(M)
        end
    end
end


"""
    _generate_intercept_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)

Generates the Turing code for the global intercept's prior and its addition to the
linear predictor.

# Rationale for Update
This version is updated for consistency with the refactored architecture. The explicit
`for` loop in the multivariate case has been replaced with a more efficient and
concise broadcasting operation (`.+= intercept'`). The unnecessary `local` keyword
has also been removed to align with the project's coding standards.

# Arguments
- `M`: The main model configuration `NamedTuple`.
- `is_multivariate`: A boolean indicating if the model is multivariate.
- `eta_name`: The name of the linear predictor variable (`eta` or `eta_latent`).

# Returns
- A tuple `(priors_code::String, updates_code::String)`.
"""
function _generate_intercept_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    if !get(M, :add_intercept, false)
        return "", ""
    end
    
    intercept_prior_obj = get(M, :intercept_prior, Normal(0, 5))
    
    dist_str, update_code = if is_multivariate
        # For multivariate, the intercept is a vector of length K.
        # We use broadcasting with a transpose to add it to each row of the eta_latent matrix.
        ("filldist($(_distribution_to_string(intercept_prior_obj)), K)",
         "$(eta_name) .+= intercept'")
    else
        # For univariate, the intercept is a scalar.
        (_distribution_to_string(intercept_prior_obj),
         "$(eta_name) .+= intercept")
    end
    
    prior_code = "intercept ~ NamedDist($(dist_str), :intercept)"
    
    return prior_code, update_code
end



"""
    _generate_offset_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)

Generates the Turing code for adding log-offsets to the linear predictor.

# Rationale for Update
This version corrects a bug where the function was using an incorrect key (`:log_offset`)
and did not properly handle the matrix dimensions of the offset data. The `bstm`
framework standardizes observation-level parameters like offsets into an `[N x K]`
matrix stored under the key `:log_offsets`. This updated function now correctly:
1.  Uses the consistent key `:log_offsets` for both univariate and multivariate cases.
2.  Adds a check to return an empty string if all offsets are zero, avoiding
    unnecessary operations in the model.
3.  In the univariate case, it correctly selects the first column of the
    `M.log_offsets` matrix (`M.log_offsets[:, 1]`) to match the dimensions of the
    `eta` vector, preventing a `DimensionMismatch`.
4.  In the multivariate case, it correctly adds the full `[N x K]` offset matrix
    to the `eta_latent` matrix.

This aligns the function with the refactored data structures and ensures type
stability and dimensional correctness.
"""
function _generate_offset_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    # Check if offsets are provided and are non-trivial.
    if !haskey(M, :log_offsets) || all(iszero, M.log_offsets)
        return ""
    end
    
    if is_multivariate
        # For multivariate models, eta_name is `eta_latent` (an N x K matrix),
        # and M.log_offsets is also an N x K matrix.
        return "$(eta_name) .+= M.log_offsets"
    else
        # For univariate models, eta_name is `eta` (an N-element vector).
        # M.log_offsets is an N x 1 matrix, so we must select the first column.
        return "$(eta_name) .+= M.log_offsets[:, 1]"
    end
end



"""
    _generate_fixed_effects_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)

Generates the Turing code for the priors and linear predictor updates for all
fixed effects.

# Rationale for Update
This version is updated to be fully consistent with the refactored architecture by
re-introducing the logic to handle non-proportional effects for ordinal models.
When an ordinal likelihood is used with `non_proportional_effects=true` on a
`fixed()` term, this function correctly separates the proportional and non-proportional
coefficients, defines their respective priors, and generates the appropriate update
code. This resolves a functional regression and ensures ordinal models are correctly specified.
The use of the `local` keyword has also been removed for stylistic consistency.

# Arguments
- `M`: The main model configuration `NamedTuple`.
- `is_multivariate`: A boolean indicating if the model is multivariate.
- `eta_name`: The name of the linear predictor variable (`eta` or `eta_latent`).

# Returns
- A tuple `(priors_code::String, updates_code::String)`.
"""
function _generate_fixed_effects_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    if get(M, :Xfixed_N, 0) == 0
        return "", ""
    end

    priors_vec = get(M, :Xfixed_priors_vec, [Normal(0, 5) for _ in 1:M.Xfixed_N])
    
    is_ordinal = any(spec -> string(get(spec, :family, "")) == "ordinal", M.likelihood_specs)
    non_prop_terms = is_ordinal ? get(M, :non_proportional_effects, Symbol[]) : Symbol[]
    
    prop_indices = collect(1:M.Xfixed_N)
    npo_indices = Int[]

    if is_ordinal && !isempty(non_prop_terms)
        npo_indices = findall(x -> x in non_prop_terms, M.Xfixed_names)
        prop_indices = setdiff(prop_indices, npo_indices)
    end

    n_prop = length(prop_indices)
    n_npo = length(npo_indices)
    K_ordinal = is_ordinal ? get(M.likelihood_specs[1], :K, 0) : 0

    prior_parts = String[]
    update_parts = String[]

    # --- Proportional Effects ---
    if n_prop > 0
        priors_prop = priors_vec[prop_indices]
        all_same_prop = !isempty(priors_prop) && all(p -> p == priors_prop[1], priors_prop)
        
        if is_multivariate
            beta_prop_name = "Xfixed_beta_prop_flat"
            update_code = "$(eta_name) .+= M.Xfixed[:, $(prop_indices)] * reshape($(beta_prop_name), $(n_prop), M.outcomes_N)"
            push!(update_parts, update_code)
            
            if all_same_prop
                prior_str = _distribution_to_string(priors_prop[1])
                push!(prior_parts, "$(beta_prop_name) ~ NamedDist(filldist($(prior_str), $(n_prop * M.outcomes_N)), :Xfixed_beta_prop)")
            else
                full_priors_list = vcat([priors_prop for _ in 1:M.outcomes_N]...)
                priors_str_list = [_distribution_to_string(p) for p in full_priors_list]
                push!(prior_parts, "$(beta_prop_name) ~ NamedDist(Product([$(join(priors_str_list, ", "))]), :Xfixed_beta_prop)")
            end
        else
            beta_prop_name = "Xfixed_beta_prop"
            update_code = "$(eta_name) .+= M.Xfixed[:, $(prop_indices)] * $(beta_prop_name)"
            push!(update_parts, update_code)
            
            if all_same_prop
                prior_str = _distribution_to_string(priors_prop[1])
                push!(prior_parts, "$(beta_prop_name) ~ NamedDist(filldist($(prior_str), $(n_prop)), :Xfixed_beta_prop)")
            else
                priors_str_list = [_distribution_to_string(p) for p in priors_prop]
                push!(prior_parts, "$(beta_prop_name) ~ NamedDist(Product([$(join(priors_str_list, ", "))]), :Xfixed_beta_prop)")
            end
        end
    end

    # --- Non-Proportional Effects (for Ordinal Models) ---
    if n_npo > 0 && K_ordinal > 1
        priors_npo = priors_vec[npo_indices]
        all_same_npo = !isempty(priors_npo) && all(p -> p == priors_npo[1], priors_npo)
        beta_npo_name = "beta_npo"
        n_npo_params = n_npo * (K_ordinal - 1)

        if all_same_npo
            prior_str = _distribution_to_string(priors_npo[1])
            push!(prior_parts, "$(beta_npo_name) ~ NamedDist(filldist($(prior_str), $(n_npo_params)), :beta_npo)")
        else
            full_priors_list = vcat([priors_npo for _ in 1:(K_ordinal-1)]...)
            priors_str_list = [_distribution_to_string(p) for p in full_priors_list]
            push!(prior_parts, "$(beta_npo_name) ~ NamedDist(Product([$(join(priors_str_list, ", "))]), :beta_npo)")
        end
        # The update logic for non-proportional effects is handled within the ordinal likelihood block.
    end

    priors_code = join(prior_parts, "\n    ")
    updates_code = join(update_parts, "\n    ")
    
    return priors_code, updates_code
end



"""
    compare_models(loo_a_report, loo_b_report; model_names=["Model_A", "Model_B"])

A utility for formal model comparison between two fitted `bstm` models. It uses
their PSIS-LOO results to compute the difference in Expected Log Pointwise
Predictive Density (ELPD) and provides a statistical basis for model selection.

# Rationale
This function is updated to be consistent with the refactored `bstm` framework,
which uses the term "component" instead of the deprecated "manifold". The function
name and internal print statements have been updated accordingly. The core logic,
which relies on `PosteriorStats.compare`, remains unchanged as it is correct.

# Arguments
- `loo_a_report`, `loo_b_report`: The output `NamedTuple` from `bstm_loo` for each model.
- `model_names`: A vector of strings with names for the models being compared.

# Returns
- A `NamedTuple` containing the comparison table, ELPD difference, and the original LOO objects.
"""
function compare_models(loo_a_report, loo_b_report; model_names=["Model_A", "Model_B"])
    println("--- Starting BSTM Model Comparison ---")

    # 1. LOO Object Extraction
    loo_a = loo_a_report.loo_obj
    loo_b = loo_b_report.loo_obj

    # 2. Formal Selection Metric Calculation
    comparison_stats = nothing
    try
        comparison_stats = compare([loo_a, loo_b])
    catch e
        @error "BSTM Comparison Error: Selection suite failed. Error: " * string(e)
        return nothing
    end

    # 3. Parameter and Diagnostic Extraction
    p_loo_a = loo_a_report.metrics.p_loo
    p_loo_b = loo_b_report.metrics.p_loo
    elpd_a = loo_a_report.metrics.elpd
    elpd_b = loo_b_report.metrics.elpd

    # 4. Report Generation
    println("\n--- BSTM Model Selection Registry ---")
    println("Model A (", model_names[1], "): ELPD = ", round(elpd_a, digits=2), " | p_loo = ", round(p_loo_a, digits=2))
    println("Model B (", model_names[2], "): ELPD = ", round(elpd_b, digits=2), " | p_loo = ", round(p_loo_b, digits=2))
    diff_elpd = elpd_a - elpd_b
    println("\nELPD Delta (A - B): ", round(diff_elpd, digits=2))

    if abs(diff_elpd) > 4.0
        winning_model = diff_elpd > 0 ? model_names[1] : model_names[2]
        println("CONCLUSION: ", winning_model, " is statistically preferred based on predictive density.")
    else
        println("CONCLUSION: Competing models provide indistinguishable predictive density.")
    end

    # 5. Table Construction
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



function process_spatial_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `spatial()` module call.
    # Rationale: Handles the setup of the adjacency matrix `W` and spatial indices `s_idx`.
    # v1.0.0 (2026-07-16)
    #            If `W` is not provided, it attempts to infer it from coordinates.
    # Assumptions: `data` is present in `opt_dict`.
    # Inputs:
    #   - opt_dict: The main configuration dictionary.
    #   - mod_data: The parsed data for this specific module.
    #   - registries, hyperpriors: Not used here, but part of the standard processor signature.
    # Outputs: None (mutates `opt_dict`).
    data = opt_dict[:data]
    # Ensure data is a DataFrame for consistency
    if !(data isa DataFrame); error("Input `data` must be a DataFrame."); end

    params = mod_data[:params]
    variables = mod_data[:variables]
    
    # # Check if this is a specialized point process like LGCP
    point_process = get(params, :point_process, nothing)
    
    if point_process == :lgcp
        # # Re-tag module type for correct builder dispatch
        mod_data[:type] = :lgcp
        # # Ensure the underlying spatial model is specified
        if !haskey(params, :model)
            params[:model] = :icar
            @warn "LGCP point process requested without a model. Defaulting to :icar."
        end
        # Point process logic has been moved to process_pointprocess_module!
        # This function should not handle it.
        @warn "Point process logic detected in process_spatial_module!. This should be handled by process_pointprocess_module!."
        return false # Do not create a spatial component if it's a point process
    end

    # 1. Resolve the adjacency matrix `W`.
    # Prioritize `W` from the module's parameters.
    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr || w_val isa Symbol
            calling_mod = get(opt_dict, :calling_module, Main)
            try
                opt_dict[:W] = Core.eval(calling_mod, w_val)
            catch e
                error("Could not evaluate `W` argument `$(w_val)` in spatial module. Error: $e")
            end
        else
            opt_dict[:W] = w_val
        end
    end
    
    # 2. If `W` is still not available, attempt to infer it from coordinates.
    if !haskey(opt_dict, :W)
        @warn "Adjacency matrix 'W' not provided for spatial module. Attempting to infer from coordinates."
        if hasproperty(data, :s_x) && hasproperty(data, :s_y)
            # `assign_spatial_units` will create AUs and a W matrix.
            au = assign_spatial_units(Matrix(data[!, [:s_x, :s_y]]); target_units=get(params, :target_units, 50))
            opt_dict[:W] = au.W
            opt_dict[:s_idx] = au.s_idx
            opt_dict[:s_N] = size(au.W, 1)
            opt_dict[:centroids] = au.centroids
            opt_dict[:centroids] = au.centroids # Store centroids for potential future use (e.g., localadaptive)
        else
            error("Cannot infer spatial structure without 'W' or coordinate columns 's_x', 's_y'.")
        end
    else
    end

    # 3. Set `s_N` and `s_idx` based on the resolved `W`.
    # This ensures `s_N` is always consistent with the `W` matrix.
    if haskey(opt_dict, :W)
        opt_dict[:s_N] = size(opt_dict[:W], 1)
        if !isempty(variables)
            s_var_sym = Symbol(variables[1])
            if hasproperty(data, s_var_sym)
                opt_dict[:s_idx] = data[!, s_var_sym]
            else
                @warn "Spatial index variable ':$s_var_sym' not found. Ensure data is aligned with W."
                @warn "Spatial index variable ':$s_var_sym' not found. Defaulting to 1:s_N."
                # If s_idx is not explicitly provided, default to 1:s_N for simplicity.
                # This assumes a direct mapping from observations to spatial units.
                opt_dict[:s_idx] = repeat(1:opt_dict[:s_N], inner=cld(nrow(data), opt_dict[:s_N]))[1:nrow(data)]
            end
        elseif !haskey(opt_dict, :s_idx)
            # If no spatial variable is given, and s_idx is not set, default to 1:s_N.
            opt_dict[:s_idx] = repeat(1:opt_dict[:s_N], inner=cld(nrow(data), opt_dict[:s_N]))[1:nrow(data)]
        end
    else
        # If W could not be resolved, s_N and s_idx cannot be reliably set.
        error("Spatial module requires an adjacency matrix `W` to define spatial units.")
    end
    
    # # Grid Area Resolution for LGCP
    if point_process == :lgcp
        if haskey(params, :grid_areas)
            ga_val = params[:grid_areas]
            if ga_val isa Symbol && hasproperty(data, ga_val)
                opt_dict[:grid_areas] = data[!, ga_val]
            elseif ga_val isa AbstractVector
                opt_dict[:grid_areas] = ga_val
            else
                # # Fallback to evaluating symbol in calling module
                calling_mod = get(opt_dict, :calling_module, Main)
                try
                    opt_dict[:grid_areas] = Core.eval(calling_mod, ga_val)
                catch
                    @warn "Could not resolve grid_areas. Defaulting to unit areas."
                    opt_dict[:grid_areas] = ones(opt_dict[:s_N])
                end
            end
        else
            opt_dict[:grid_areas] = ones(opt_dict[:s_N])
        end
    end

    return true
end
 

"""
    process_smooth_module!(opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict)

Processes a module with `structure=:smooth`, which is typically invoked via
`random(var, structure=:smooth, model=...)`. This function is responsible for
generating basis matrices for various static smoothers or setting up coordinate
data for continuous and dynamic kernel-based models.

# Rationale for Update
This function is updated to be consistent with the refactored component processing
pipeline. Key changes include:
1.  **Standardized Signature**: The function signature is updated to the standard
    `(opt_dict, mod_data, registries, hyperpriors)` format.
2.  **Direct Registry Access**: It now directly accesses `opt_dict[:basis_matrices]`
    instead of using a complex registry argument.
3.  **Model Categorization Fix**: The `kriging` model, which is a continuous kernel
    model, has been correctly moved to the `continuous_kernel_models` list.
4.  **Clarity and Comments**: Added comments to delineate the logic for different
    smoother types (static basis, dynamic basis, continuous kernel, GMRF-on-bins).
5.  **Robustness**: Removed fragile `@isdefined` checks and unnecessary `local` keywords.
6.  **Consistency**: The logic for handling different `nbins` specifications (Int, Vector,
    Symbol) and for re-tagging GMRF-on-bins models as `:mixed` is preserved and clarified.

# Arguments
- `opt_dict`: The main model configuration dictionary (`M`).
- `mod_data`: The parsed data for the `random(structure=:smooth)` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries (not used here).

# Returns
- `true` to indicate that a component object should be created.
"""
function process_smooth_module!(
    opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict
)
    # This processor handles `random(..., structure=:smooth)` calls.
    # It pre-computes basis matrices for static models or prepares coordinate
    # data for dynamic/continuous models.

    basis_registry = opt_dict[:basis_matrices]
    data = opt_dict[:data]
    params = mod_data[:params]
    model_param = get(params, :model, "pspline")
    original_nbins_param = get(params, :nbins, 20)
    variables = mod_data[:variables]
    n_vars = length(variables)
    
    nbins_per_dim_vec = Int[]
    total_bins_for_component_obj = 0

    if n_vars > 0
        if original_nbins_param isa Int
            nbins_per_dim_vec = fill(original_nbins_param, n_vars)
            total_bins_for_component_obj = original_nbins_param^n_vars
        elseif original_nbins_param isa Vector{Int}
            if length(original_nbins_param) != n_vars
                error("`nbins` vector length must match number of variables for smooth. Got $(length(original_nbins_param)) for $n_vars variables.")
            end
            nbins_per_dim_vec = original_nbins_param
            total_bins_for_component_obj = prod(original_nbins_param)
        else
            # Handle symbolic or expression-based nbins
            calling_mod = get(opt_dict, :calling_module, Main)
            try
                evaluated_nbins = Core.eval(calling_mod, original_nbins_param)
                temp_params = copy(params)
                temp_params[:nbins] = evaluated_nbins
                temp_mod_data = Dict(:type => mod_data[:type], :params => temp_params, :variables => variables)
                # Recursive call with evaluated nbins
                return process_smooth_module!(opt_dict, temp_mod_data, registries, hyperpriors)
            catch e
                error("Could not evaluate `nbins` parameter `$(original_nbins_param)`. Error: $e")
            end
        end
        mod_data[:params][:nbins] = total_bins_for_component_obj
        if !isempty(nbins_per_dim_vec)
            mod_data[:params][:nbins_per_dim] = nbins_per_dim_vec
        end
    end

    # Categorize models to determine processing path
    basis_models = ["pspline", "bspline", "tps", "moran", "spherical", "barycentric", "decay", "linear", "invdist"]
    dynamic_basis_models = ["wavelet", "fft"]
    continuous_kernel_models = ["gp", "fitc", "svgp", "nystrom", "warp", "spde", "exponentialdecay", "rff", "kriging"]
    gmrfs_on_bins_models = ["rw1", "rw2", "ar1", "icar", "besag", "cyclic"]
    
    model_str = string(model_param)

    # --- Path 1: Dynamic Basis Models (e.g., wavelet, fft) ---
    # For these, the basis depends on a hyperparameter (e.g., lengthscale).
    # We do not pre-compute the basis matrix. Instead, we pass the raw coordinates
    # to the component, and the basis is constructed inside the Turing model.
    if model_str in dynamic_basis_models
        if all(v -> hasproperty(data, Symbol(v)), mod_data[:variables])
            coords = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
            mod_data[:params][:coords] = coords
        else
            error("Coordinate variables for smooth model not found in data: $(mod_data[:variables])")
        end
        return true
    end

    # --- Path 2: Static Basis Models (e.g., pspline, tps) ---
    # For these, the basis matrix is fixed and can be pre-computed.
    if model_str in basis_models
        if !isempty(mod_data[:variables]) 
            reg_key = Symbol(join(mod_data[:variables], "_"))
            if all(hasproperty(data, Symbol(v)) for v in mod_data[:variables])
                local_kwargs = Dict(params)
                delete!(local_kwargs, :nbins)
                
                B_smooth_matrix, actual_nbins_for_component = if n_vars == 1
                    v_vec = data[!, Symbol(mod_data[:variables][1])]
                    bstm_smooth_basis_1D(model_str, v_vec, nbins_per_dim_vec[1], get(params, :degree, 3); local_kwargs...)
                else
                    c_mat = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
                    B_matrix = if n_vars == 2; bstm_smooth_basis_2D(model_str, c_mat, nbins_per_dim_vec; local_kwargs...);
                    elseif n_vars == 3; bstm_smooth_basis_3D(model_str, c_mat, nbins_per_dim_vec; local_kwargs...);
                    elseif n_vars == 4; bstm_smooth_basis_4D(model_str, c_mat, nbins_per_dim_vec; local_kwargs...);
                    else; error("Smoothers with more than 4 dimensions are not supported for this basis type."); end
                    (B_matrix, size(B_matrix, 2)) # Return matrix and its column count
                end
                
                basis_registry[reg_key] = B_smooth_matrix
                mod_data[:params][:nbins] = actual_nbins_for_component # Update nbins with the actual count
            end
        end
    
    # --- Path 3: Continuous Kernel Models (e.g., gp, fitc) ---
    # These models operate on coordinates directly, not on a basis matrix.
    # We store the coordinates and, for sparse models, pre-compute inducing points.
    elseif model_str in continuous_kernel_models
        if all(v -> hasproperty(data, Symbol(v)), mod_data[:variables])
            coords = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
            mod_data[:params][:coords] = coords
            if model_str in ["fitc", "svgp", "nystrom", "gp"]
                n_inducing_default = min(100, size(coords, 1))
                n_inducing = get(mod_data[:params], :n_inducing, n_inducing_default)
                Z_inducing = generate_inducing_points(coords, n_inducing; seed=42, method="kmeans")
                mod_data[:params][:Z_inducing] = Z_inducing
            end
        else
            @warn "Continuous kernel smooth specified, but coordinate variables not found in data. Component may be misspecified."
        end

    # --- Path 4: GMRFs on Binned Covariates (e.g., rw2 on age) ---
    # This re-tags the module as a `:mixed` effect on a discretized variable.
    elseif model_str in gmrfs_on_bins_models
        vars = mod_data[:variables]
        if length(vars) != 1; @warn "GMRF smooth on $(join(vars, ",")) requires exactly 1 variable. Skipping."; return true; end
        
        var_sym = Symbol(vars[1])
        nbins = get(mod_data[:params], :nbins, 20)
        _, indices = apply_discretization_logic(data[!, var_sym], nbins)
        
        index_key = Symbol("mixed_idx_$(string(vars[1]))")
        opt_dict[index_key] = indices
        
        mod_data[:params][:indices] = indices
        mod_data[:params][:n_cat] = length(unique(indices))
        mod_data[:type] = :mixed # Re-tag for the mixed effect processor
    end    
    
    mod_data[:params][:model] = model_param
    return true
end

  
"""
    process_pointprocess_module!(opt_dict, mod_data, registries, hyperpriors)

Processes the `pointprocess()` module, setting up the necessary spatial context and
parameters for point process models like Log-Gaussian Cox Processes (LGCP).

# Rationale
This processor is designed to handle modules created by the `pointprocess() ∘ random()`
syntax. It assumes the parser has created a single module of type `:pointprocess` and
has merged the arguments from both the `pointprocess()` and `random()` calls.

The function performs two main tasks:
1.  **Spatial Context Setup**: It delegates to `process_random_module!` to ensure the
    underlying spatial context (e.g., adjacency matrix `W`, spatial indices `s_idx`,
    and spatial dimensions `s_N`) is correctly established. This is crucial because
    point process models are built upon a latent spatial field.
2.  **Point Process Parameter Resolution**: It handles parameters specific to the point
    process model. For an LGCP (`model=:lgcp`), this involves resolving the `grid_areas`
    parameter, which is necessary for correctly integrating the latent intensity
    surface. It defaults to unit areas if not provided.

This function ensures that all necessary data structures and parameters are in place
before the component object (e.g., `LGCP`) is instantiated by `resolve_technical_primitive`.

# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the `pointprocess()` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries.

# Returns
- `true` to indicate that a component object should be created.
"""
function process_pointprocess_module!(opt_dict, mod_data, registries, hyperpriors)
    # This processor handles modules created by `pointprocess()`.
    # It sets up the spatial context and resolves point process-specific parameters.

    # 1. The underlying structure is spatial, so delegate to the random module processor
    #    to set up W, s_idx, s_N, etc. The `random` processor will correctly infer
    #    the structure as :spatial based on the presence of W or spatial coordinates.
    process_random_module!(opt_dict, mod_data, registries, hyperpriors)
    
    # 2. Handle parameters specific to the point process model.
    #    The specific model (e.g., :lgcp) is passed as a parameter.
    model_type = get(mod_data[:params], :model, :lgcp) # Default to LGCP

    if model_type == :lgcp
        # For an LGCP, we need to resolve the `grid_areas` parameter.
        if haskey(mod_data[:params], :grid_areas)
            ga_val = mod_data[:params][:grid_areas]
            if ga_val isa Symbol && hasproperty(opt_dict[:data], ga_val)
                opt_dict[:grid_areas] = opt_dict[:data][!, ga_val]
            elseif ga_val isa AbstractVector
                opt_dict[:grid_areas] = ga_val
            else
                # Fallback to evaluating the symbol in the calling module's scope.
                calling_mod = get(opt_dict, :calling_module, Main)
                try
                    opt_dict[:grid_areas] = Core.eval(calling_mod, ga_val)
                catch
                    @warn "Could not resolve `grid_areas` for LGCP. Defaulting to unit areas."
                    opt_dict[:grid_areas] = ones(opt_dict[:s_N])
                end
            end
        else
            # If grid_areas is not specified, default to ones(s_N).
            # This requires s_N to have been set by the spatial processor call above.
            if !haskey(opt_dict, :s_N)
                error("Cannot default `grid_areas` for LGCP because `s_N` is not yet established. Ensure a spatial context is defined.")
            end
            opt_dict[:grid_areas] = ones(opt_dict[:s_N])
        end
    end

    # Add logic for other point process models here if needed.

    # Return true to indicate that the component object (e.g., LGCP) should be created.
    return true
end

# src/tmp.jl


"""
    process_pointprocess_module!(opt_dict, mod_data, registries, hyperpriors)

Processes the `pointprocess()` module, setting up the necessary spatial context and
parameters for point process models like Log-Gaussian Cox Processes (LGCP).

# Rationale
This processor is designed to handle modules created by the `pointprocess() ∘ random()`
syntax. It assumes the parser has created a single module of type `:pointprocess` and
has merged the arguments from both the `pointprocess()` and `random()` calls.

The function performs two main tasks:
1.  **Spatial Context Setup**: It delegates to `process_random_module!` to ensure the
    underlying spatial context (e.g., adjacency matrix `W`, spatial indices `s_idx`,
    and spatial dimensions `s_N`) is correctly established. This is crucial because
    point process models are built upon a latent spatial field.
2.  **Point Process Parameter Resolution**: It handles parameters specific to the point
    process model. For an LGCP (`model=:lgcp`), this involves resolving the `grid_areas`
    parameter, which is necessary for correctly integrating the latent intensity
    surface. It defaults to unit areas if not provided.

This function ensures that all necessary data structures and parameters are in place
before the component object (e.g., `LGCP`) is instantiated by `resolve_technical_primitive`.

# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the `pointprocess()` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries.

# Returns
- `true` to indicate that a component object should be created.
"""
function process_pointprocess_module!(opt_dict, mod_data, registries, hyperpriors)
    # This processor handles modules created by `pointprocess()`.
    # It sets up the spatial context and resolves point process-specific parameters.

    # 1. The underlying structure is spatial, so delegate to the random module processor
    #    to set up W, s_idx, s_N, etc. The `random` processor will correctly infer
    #    the structure as :spatial based on the presence of W or spatial coordinates.
    #    It's crucial that `process_random_module!` does NOT handle point_process-specific
    #    parameters when called from here.
    process_random_module!(opt_dict, mod_data, registries, hyperpriors)
    
    # 2. Handle parameters specific to the point process model.
    #    The specific model (e.g., :lgcp) is passed as a parameter.
    model_type = get(mod_data[:params], :model, :lgcp) # Default to LGCP

    if model_type == :lgcp
        # For an LGCP, we need to resolve the `grid_areas` parameter.
        if haskey(mod_data[:params], :grid_areas)
            ga_val = mod_data[:params][:grid_areas]
            if ga_val isa Symbol && hasproperty(opt_dict[:data], ga_val)
                opt_dict[:grid_areas] = opt_dict[:data][!, ga_val]
            elseif ga_val isa AbstractVector
                opt_dict[:grid_areas] = ga_val
            else
                # Fallback to evaluating the symbol in the calling module's scope.
                calling_mod = get(opt_dict, :calling_module, Main)
                try
                    opt_dict[:grid_areas] = Core.eval(calling_mod, ga_val)
                catch
                    @warn "Could not resolve `grid_areas` for LGCP. Defaulting to unit areas."
                    opt_dict[:grid_areas] = ones(opt_dict[:s_N])
                end
            end
        else
            # If grid_areas is not specified, default to ones(s_N).
            # This requires s_N to have been set by the spatial processor call above.
            if !haskey(opt_dict, :s_N)
                error("Cannot default `grid_areas` for LGCP because `s_N` is not yet established. Ensure a spatial context is defined.")
            end
            opt_dict[:grid_areas] = ones(opt_dict[:s_N])
        end
    end

    # Add logic for other point process models here if needed.

    # Return true to indicate that the component object (e.g., LGCP) should be created.
    return true
end

"""
    process_eigen_module!(opt_dict, mod_data, registries, hyperpriors)

Processes the `eigen()` module call.

# Rationale for Update
The original implementation did not extract the data for the variables specified in the
`eigen()` call, making it impossible to perform PCA. This updated version correctly
extracts the relevant columns from the data frame, centers them, and stores the
resulting matrix in the module's parameters for use by the model builder and code
generator. It also validates the number of factors against the number of variables.
"""
function process_eigen_module!(opt_dict, mod_data, registries, hyperpriors)
    params = mod_data[:params]
    vars_str = mod_data[:variables]
    vars_sym = Symbol.(vars_str)
    
    if isempty(vars_sym)
        error("The `eigen()` module was called without any variables specified.")
    end

    data = opt_dict[:data]
    if !all(hasproperty(data, v) for v in vars_sym)
        missing_vars = filter(v -> !hasproperty(data, v), vars_sym)
        error("Eigen module variables not found in data: $(missing_vars)")
    end
    
    # Extract the data and center it (a standard assumption for PCA).
    eigen_data_matrix = Matrix(data[!, vars_sym])
    eigen_data_matrix .-= mean(eigen_data_matrix, dims=1)
    
    # Store the data matrix in the module's parameters for the builder to access.
    mod_data[:params][:eigen_data] = eigen_data_matrix
    
    n_vars = length(vars_sym)
    n_factors = get(params, :n_factors, 1)
    if n_factors >= n_vars
        @warn "Number of factors ($n_factors) for eigen() module should be less than the number of variables ($n_vars). Setting to $(n_vars - 1)."
        n_factors = n_vars - 1
    end
    
    # Pre-calculate indices for the lower-triangular part of the Householder matrix.
    ltri_mask = [r >= c for r in 1:n_vars, c in 1:n_factors]
    ltri_indices = findall(vec(ltri_mask))
    
    mod_data[:params][:ltri_indices] = ltri_indices
    mod_data[:params][:n_factors] = n_factors
    mod_data[:params][:n_vars] = n_vars
    
    return true # Proceed with component creation.
end

"""
    process_mixed_module!(opt_dict, mod_data, registries, hyperpriors)

Processes the `mixed()` module for random effects. This version uses `StatsModels.jl`
to correctly parse the effects part of the mixed model formula (e.g., `intercept() + cov1`).
It handles `intercept()` by converting it to `1` and correctly identifies all terms, which
are then passed to the code generator as a vector of strings. This fixes a bug where
`intercept()` was treated as a string literal, causing a `ParseError`.

# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the `mixed()` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries.

# Returns
- `true` to indicate that a `Mixed` object should be created.
"""
function process_mixed_module!(opt_dict, mod_data, registries, hyperpriors)
    data = opt_dict[:data]
    vars = mod_data[:variables]
    
    response_var = Symbol(opt_dict[:outcomes][1])

    effect_expr, group_var_str = if !isempty(vars) && vars[1] isa Expr && vars[1].head == :call && vars[1].args[1] == :|
        # Handles the `effect | group` syntax.
        (vars[1].args[2], string(vars[1].args[3]))
    elseif length(vars) >= 2
        # Handles the `effect, group` syntax.
        (vars[1], string(vars[2]))
    else
        @warn "The mixed() module requires syntax `mixed(effect | group)` or `mixed(effect, group)`. Skipping."
        return false
    end

    # Replace bstm-specific modules like `intercept()` with `1` for StatsModels.jl.
    effect_expr_mod = _replace_bstm_modules_in_expr(effect_expr)
    schema = StatsModels.schema(data)
    
    terms = if effect_expr_mod isa Number
        StatsModels.term(effect_expr_mod)
    else
        calling_mod = get(opt_dict, :calling_module, Main)
        form = Core.eval(calling_mod, :(@formula($response_var ~ $(effect_expr_mod))))
        applied_form = StatsModels.apply_schema(form, schema)
        applied_form.rhs
    end

    # Decompose the parsed formula into a vector of individual term strings.
    # This correctly handles multi-term effects like `(1 + cov1 | group)`.
    term_vec = if terms isa StatsModels.TupleTerm
        terms.terms
    elseif terms isa StatsModels.AbstractTerm
        (terms,) # Wrap single term in a tuple for consistent iteration.
    elseif terms isa Tuple
        collect(terms)
    else
        [terms]
    end
    
    effect_names = String[]
    for term in term_vec
        if term isa StatsModels.InterceptTerm{true}
            push!(effect_names, "1")
        elseif term isa StatsModels.InterceptTerm{false}
            continue # Skip if intercept is explicitly removed with `0`.
        else
            push!(effect_names, _canonical_term_string(term))
        end
    end

    group_var_sym = Symbol(group_var_str)
    if !hasproperty(data, group_var_sym)
        error("Grouping variable ':$group_var_sym' for mixed() module not found in dataset.")
    end
    
    # Create integer indices for the grouping variable.
    group_data = data[!, group_var_sym]
    unique_levels = unique(group_data)
    group_map = Dict(v => i for (i, v) in enumerate(unique_levels))
    indices = [group_map[v] for v in group_data]

    # Store the generated indices and parameters in the configuration dictionaries
    # for the code generator to use.
    index_key = Symbol("mixed_idx_$(group_var_str)")
    opt_dict[index_key] = indices
    
    mod_data[:params][:indices] = indices
    mod_data[:params][:n_cat] = length(unique_levels)
    mod_data[:params][:lhs] = effect_names
    mod_data[:variables] = [group_var_str]
    
    return true
end


"""
    process_spacetime_module!(opt_dict, mod_data, registries, hyperpriors)

Processes the `spacetime()` module. **NOTE: This module is considered deprecated.**
Its functionality is superseded by the more explicit `random(...) ⊗ random(...)`
syntax, which is handled by `process_interact_module!`.

This processor determines the Knorr-Held interaction type based on the specified
spatial and temporal models and sets the global `:model_st` flag. It does not
create a component itself.

# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the `spacetime()` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries.

# Returns
- `false`, indicating that no `Component` object should be created for this module.
"""
function process_spacetime_module!(opt_dict, mod_data, registries, hyperpriors)
    models = get(mod_data[:params], :model, (:iid, :iid))
    
    spatial_model, temporal_model = if models isa Expr && models.head == :tuple && length(models.args) == 2
        (string(models.args[1]), string(models.args[2]))
    elseif models isa Tuple && length(models) == 2
        (string(models[1]), string(models[2]))
    else
        error("The `model` for a spacetime interaction must be a tuple of two models, e.g., `model=(icar, ar1)`.")
    end

    has_structured_space = (spatial_model != "iid")
    has_structured_time = (temporal_model != "iid")

    opt_dict[:model_st] = if has_structured_space && has_structured_time
        "IV"
    elseif !has_structured_space && has_structured_time
        "II"
    elseif has_structured_space && !has_structured_time
        "III"
    else # !has_structured_space && !has_structured_time
        "I"
    end

    if haskey(mod_data[:params], :sigma)
        prior_val = mod_data[:params][:sigma]
        calling_mod = get(opt_dict, :calling_module, Main)
        if prior_val isa Tuple
            opt_dict[:st_interaction_sigma_prior] = create_pc_prior(:sigma, prior_val)
        elseif prior_val isa Expr
            try
                opt_dict[:st_interaction_sigma_prior] = Core.eval(calling_mod, prior_val)
            catch e
                error("Could not evaluate `prior` argument `$(prior_val)` for spacetime interaction. Error: $e")
            end
        else
            opt_dict[:st_interaction_sigma_prior] = prior_val
        end
    end
    
    # This module only sets flags; it does not create a component itself.
    return false
end

"""
    process_fixed_module!(opt_dict, mod_data, registries, hyperpriors)

Processes the `fixed()` module.

# Rationale for Update
This version is updated to collect the variable names from the `fixed()` call
and store them in `opt_dict[:fixed_effects_from_modules]`. This ensures that
fixed effects specified via `fixed()` are correctly included in the model's
design matrix.
"""
function process_fixed_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `fixed()` module.
    # Rationale: Gathers information about fixed effects, including custom contrasts and priors.
    #            This version is updated to collect the variable names from the `fixed()` call
    #            and store them in `opt_dict[:fixed_effects_from_modules]`. This ensures that
    #            fixed effects specified via `fixed()` are correctly included in the model's
    #            design matrix.
    # v1.1.0 (2026-08-03)
    if !haskey(opt_dict, :fixed_effects); opt_dict[:fixed_effects] = String[]; end
    if !haskey(opt_dict, :contrasts); opt_dict[:contrasts] = Dict{Symbol, Any}(); end
    if !haskey(opt_dict, :fixed_effects_priors); opt_dict[:fixed_effects_priors] = Dict{Symbol, Any}(); end
    if !haskey(opt_dict, :vars_to_categorize); opt_dict[:vars_to_categorize] = Set{Symbol}(); end
    
    params = mod_data[:params]
    vars = mod_data[:variables]

    # --- NEW: Collect fixed effect variables ---
    if !haskey(opt_dict, :fixed_effects_from_modules)
        opt_dict[:fixed_effects_from_modules] = String[]
    end
    for var in vars
        push!(opt_dict[:fixed_effects_from_modules], string(var))
    end
    # --- END NEW ---

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
    return false # fixed() modules do not create components in the main loop.
end


"""
    process_custom_module!(opt_dict, mod_data, registries, hyperpriors)

Processes the `custom()` module.

# Rationale
This function serves as the processor for the `custom()` module. It is intentionally
minimal because the `custom()` module's logic is entirely self-contained within the
user-provided `code_fragment`. Its only role is to return `true`, which signals to
the configuration engine that a `Custom` component object should be instantiated. This
object then acts as a wrapper for the user's code, which is handled by a specialized
code generator later in the pipeline. The function is not deprecated and is a necessary
part of the component processing system.

# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the `custom()` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries (not used here).

# Returns
- `true` to indicate that a `Custom` component object should be created.
"""
function process_custom_module!(opt_dict, mod_data, registries, hyperpriors)
    # This module is a placeholder for user-defined code.
    # It performs no data processing itself but signals that a component
    # object should be created to hold the user's code fragment.
    return true
end

 

"""
    process_localadaptive_module!(opt_dict, mod_data, registries, hyperpriors)

Processes the `localadaptive` model, ensuring centroids and cluster assignments are
correctly computed for all spatial units.

# Rationale for Update
This function is updated to be consistent with the refactored component system. It
no longer calls the deprecated `process_spatial_module!`. Instead, it assumes that
the main `process_random_module!` has already established the necessary spatial
context (`s_N`, `s_idx`, `W`). This function's sole responsibility is now to handle
the logic specific to the `localadaptive` model, which includes resolving the
centroids for all spatial units and performing k-means clustering to generate the
`cluster_assignments`. This change removes redundancy and aligns the function with
the modular design of the refactor. The robust logic for resolving and validating
centroids from the previous version is retained.

# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the `random(model=:localadaptive)` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries (not used here).

# Returns
- `true` to indicate that a `LocalAdaptive` component object should be created.
"""
function process_localadaptive_module!(opt_dict, mod_data, registries, hyperpriors)
    s_N = get(opt_dict, :s_N, 0)
    if s_N == 0
        error("The `localadaptive` model requires a spatial context (`s_N`), but it has not been established. Ensure a spatial variable and adjacency matrix `W` are provided.")
    end
    
    data = opt_dict[:data]

    # The localadaptive model requires centroids for clustering.
    if !haskey(opt_dict, :centroids)
        @info "Centroids not found for localadaptive model. Attempting to compute from s_x and s_y coordinates."
        if hasproperty(data, :s_x) && hasproperty(data, :s_y) && hasproperty(data, :s_idx)
            
            # Create a map from each spatial index to its unique coordinate.
            coord_map = Dict{Int, Point2D}()
            for i in 1:nrow(data)
                idx = data.s_idx[i]
                if !haskey(coord_map, idx)
                    coord_map[idx] = Point2D(data.s_x[i], data.s_y[i])
                end
            end

            # Check if we have coordinates for all s_N units.
            if length(coord_map) < s_N
                error("The `localadaptive` model requires coordinates for all $(s_N) spatial units, but only found coordinates for $(length(coord_map)) unique units in the data. Please provide a complete `centroids` vector as a keyword argument.")
            end

            # Reconstruct the full, ordered list of centroids.
            centroids = Vector{Point2D}(undef, s_N)
            for i in 1:s_N
                if !haskey(coord_map, i)
                     error("Coordinate for spatial unit `s_idx = $i` not found in the data. The `localadaptive` model requires complete coordinate information.")
                end
                centroids[i] = coord_map[i]
            end
            opt_dict[:centroids] = centroids
        else
            error("The `localadaptive()` model requires centroids for clustering. Provide them via the `centroids` keyword argument, or ensure spatial coordinates (s_x, s_y) and indices (s_idx) are in the data frame.")
        end
    end
    
    centroids = opt_dict[:centroids]
    if length(centroids) != s_N
        error("The number of provided centroids ($(length(centroids))) does not match the number of spatial units s_N ($(s_N)).")
    end

    params = mod_data[:params]
    n_clusters = get(params, :n_clusters, 5)
    
    if length(centroids) < n_clusters
        @warn "Number of spatial units ($(length(centroids))) is less than the requested number of clusters ($n_clusters). Adjusting n_clusters to $(length(centroids))."
        n_clusters = length(centroids)
    end
    
    # Clustering.jl expects a [dims x n_points] matrix
    centroids_matrix = hcat([c.x for c in centroids], [c.y for c in centroids])'
    
    # Perform k-means clustering on the centroids
    kmeans_result = kmeans(centroids_matrix, n_clusters; maxiter=200, display=:none)
    
    # The assignments map each of the s_N centroids to a cluster. This vector now has the correct length.
    opt_dict[:cluster_assignments] = assignments(kmeans_result)
    opt_dict[:n_clusters] = nclusters(kmeans_result)
    
    return true
end


"""
    process_nested_module!(opt_dict, mod_data, registries, hyperpriors)

Processes the `nested()` module for multi-fidelity or joint models. This function
recursively calls `bstm_config` to create a complete and independent configuration
for the sub-model defined within the `nested()` call.

# Rationale
This processor is the core of the multi-fidelity and joint modeling capabilities. It
validates the `data_source` and `formula` arguments for the sub-model, correctly
evaluating them in the user's calling module if necessary. It then orchestrates a
recursive call to `bstm_config`, creating a self-contained configuration for the
sub-model which is then stored in the main configuration's `:nested_components`
registry. This function does not create a component in the main model's processing
loop (returns `false`), as the nested model's effects are handled by a specialized
code generator. This version is consistent with the refactored architecture.

# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the `nested()` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries.

# Returns
- `false`, as the `nested()` module itself does not create a `Component` object.
"""
function process_nested_module!(opt_dict, mod_data, registries, hyperpriors)
    if !haskey(opt_dict, :nested_components)
        opt_dict[:nested_components] = Dict{Symbol, Any}()
    end
    
    var = Symbol(mod_data[:variables][1])
    params = mod_data[:params]
    sub_formula_raw = get(params, :formula, "")
    data_source_sym = get(params, :data_source, :data)

    if !haskey(opt_dict, data_source_sym)
        @warn "Data source ':$data_source_sym' for nested module on '$var' not found. Skipping."
        return false
    end

    sub_data = opt_dict[data_source_sym]
    
    # Prepare keyword arguments for the recursive bstm_config call.
    # Exclude keys that are specific to the parent model.
    sub_config_kwargs = Dict(pairs(NamedTuple(opt_dict)))
    delete!(sub_config_kwargs, :data)
    delete!(sub_config_kwargs, data_source_sym)
    delete!(sub_config_kwargs, :formula)
    
    # Robustly handle the formula argument, which can be a String, Symbol, or Expr.
    sub_formula_str::String = if sub_formula_raw isa String
        sub_formula_raw
    elseif sub_formula_raw isa Symbol
        calling_mod = get(opt_dict, :calling_module, Main)
        try
            Core.eval(calling_mod, sub_formula_raw)
        catch e
            error("Could not evaluate the `formula` variable `$(sub_formula_raw)` for the nested module. Ensure it is defined in the calling scope. Error: $e")
        end
    elseif sub_formula_raw isa Expr
        string(sub_formula_raw)
    else
        error("Unsupported type for `formula` argument in nested module: $(typeof(sub_formula_raw))")
    end

    if !(sub_formula_str isa String)
        error("The `formula` argument for the nested module must resolve to a String. Got type: $(typeof(sub_formula_str))")
    end
    
    # Recursively call bstm_config to create a full configuration for the sub-model.
    sub_config = bstm_config(sub_formula_str, sub_data; sub_config_kwargs...)
    
    # Store the complete sub-model configuration.
    opt_dict[:nested_components][var] = sub_config
    
    # The nested module itself does not create a component in the main model loop.
    return false
end


"""
    process_random_module!(opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict)

The main processor for the `random()` module, updated to support hierarchical mosaic
and bipartite graph (`bcgn`) models.

# Rationale for Update
This version is updated to be consistent with the refactored component system.
- It now includes a dedicated block to handle `model=:bcgn`. When specified, it
  resolves the adjacency matrix `W`, converts it to a bipartite representation, and
  stores the resulting graph structures in the component's parameters for use by the
  `BCGN` component builder. This makes the standalone `process_bcgn_module!` obsolete.
- The redundant logic for handling point process parameters (e.g., `grid_areas`) has been
  removed, as this is now the sole responsibility of the `process_pointprocess_module!`.
- The logic for inferring structure, handling special model types like `:localadaptive`,
  and processing mosaic components is preserved.
"""
function process_random_module!(opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict)
    params = mod_data[:params]
    data = opt_dict[:data]
    variables = mod_data[:variables]
    
    # --- 1. Infer Structure and Handle Special Models ---
    is_anisotropic = get(params, :anisotropic, false)
    ls_prior = get(params, :lengthscale, get(params, :kappa, nothing))
    
    if ls_prior isa Expr && ls_prior.head == :vect && length(ls_prior.args) > 1
        is_anisotropic = true
    end
    params[:anisotropic] = is_anisotropic

    if !isempty(variables); params[:in_dims] = length(variables); end

    if !haskey(params, :structure)
        args_for_inference = copy(params)
        args_for_inference[:vars] = get(mod_data, :variables, [])
        params[:structure] = _infer_structure_from_args(args_for_inference)
    end
    structure = params[:structure]
    
    model_name = get(params, :model, :iid)

    # Handle special models that have their own dedicated processors
    if model_name == :localadaptive
        process_localadaptive_module!(opt_dict, mod_data, registries, hyperpriors)
        return true
    elseif model_name == :bcgn
        # --- BCGN Logic ---
        # This block handles the setup for the Bipartite Graph Convolutional Network model.
        W = get(params, :W, get(opt_dict, :W, nothing))
        if isnothing(W)
            error("The `bcgn` model requires an adjacency matrix `W`.")
        end
        bipartite_info = adjacency_to_bipartite(W)
        mod_data[:params][:bipartite_adj] = bipartite_info.bipartite_adj
        mod_data[:params][:set1_indices] = bipartite_info.set1
        mod_data[:params][:set2_indices] = bipartite_info.set2
        # The number of latent units is the size of the first partition.
        opt_dict[:s_N] = length(bipartite_info.set1)
        return true
    elseif structure == :svar
        process_svar_module!(opt_dict, mod_data, registries, hyperpriors)
        return true
    end

    # --- 2. Set up Context Based on Structure ---
    if structure == :spatial
        if haskey(params, :W)
            w_val = params[:W]
            if w_val isa Expr || w_val isa Symbol
                calling_mod = get(opt_dict, :calling_module, Main)
                try; opt_dict[:W] = Core.eval(calling_mod, w_val); catch e; error("Could not evaluate `W` argument `$(w_val)`. Error: $e"); end
            else
                opt_dict[:W] = w_val
            end
        end
        if !haskey(opt_dict, :W); error("Spatial model requires an adjacency matrix `W`."); end
        
        opt_dict[:s_N] = size(opt_dict[:W], 1)
        if !isempty(variables)
            s_var_sym = Symbol(variables[1])
            if hasproperty(data, s_var_sym); opt_dict[:s_idx] = data[!, s_var_sym]; else; @warn "Spatial index ':$s_var_sym' not found."; end
        end

        # --- 2a. Process Mosaic Component (now that spatial context exists) ---
        if haskey(params, :mosaic)
            grouping_info = _process_mosaic_grouping!(opt_dict, mod_data)
            
            mosaic_key = "mosaic_$(mod_data[:key])"
            group_col_name = grouping_info.group_col_name

            mosaic_params = Dict(
                :model => :iid,
                :n_cat => grouping_info.n_regions,
                :lhs => ["1"] # Specifies a random intercept.
            )
            
            group_data = opt_dict[:data][!, group_col_name]
            unique_levels = unique(group_data)
            group_map = Dict(v => i for (i, v) in enumerate(unique_levels))
            indices = [group_map[v] for v in group_data]
            index_key = Symbol("mixed_idx_$(group_col_name)")
            opt_dict[index_key] = indices
            mosaic_params[:indices] = indices

            mosaic_mod_data = Dict(
                :type => :mixed,
                :variables => [group_col_name],
                :params => mosaic_params,
                :key => mosaic_key
            )
            
            mosaic_comp_obj = resolve_technical_primitive(mosaic_mod_data, opt_dict, hyperpriors, opt_dict[:prior_scheme])
            mosaic_spec_built = build_model(mosaic_comp_obj, opt_dict, mosaic_mod_data)
            
            final_mosaic_spec = (
                key=Symbol(mosaic_key), 
                structure=:mixed, 
                var=string(grouping_info.group_col_name), 
                component_obj=mosaic_comp_obj, 
                params=mosaic_mod_data[:params], 
                Q_template=mosaic_spec_built.Q_template, 
                scaling_factor=mosaic_spec_built.scaling_factor,
                hyper=mosaic_spec_built.hyper
            )
            push!(opt_dict[:components], final_mosaic_spec)

            delete!(params, :mosaic)
            if haskey(params, :n_regions); delete!(params, :n_regions); end
        end

    elseif structure == :temporal
        if model_name == :tar
            mod_data[:type] = :tar
            if !haskey(params, :threshold_var)
                error("The `tar` model requires a `threshold_var` parameter, e.g., `random(time, model=tar, threshold_var=my_covariate)`.")
            end
        end
        continuous_kernel_models = [:gp, :rff, :fitc, :svgp, :nystrom, :warp, :spde, :kriging]
        is_continuous_kernel_on_time = model_name in continuous_kernel_models
        is_discrete_seasonal = model_name == :cyclic
        is_continuous_periodic = model_name == :harmonic || haskey(params, :period)

        if !isempty(variables)
            t_var_sym = Symbol(variables[1])
            if hasproperty(data, t_var_sym)
                if is_continuous_kernel_on_time
                    coords = Matrix{Float64}(data[!, [t_var_sym]])
                    params[:coords] = coords
                    mod_data[:type] = :smooth 
                    if model_name in [:fitc, :svgp, :nystrom]
                        n_inducing = get(params, :n_inducing, min(100, size(coords, 1)))
                        params[:n_inducing] = n_inducing
                        params[:Z_inducing] = generate_inducing_points(coords, n_inducing)
                    end
                elseif is_discrete_seasonal || is_continuous_periodic
                    opt_dict[:u_idx] = data[!, t_var_sym]
                    opt_dict[:u_N] = length(unique(opt_dict[:u_idx]))
                    opt_dict[:u_idx_var] = t_var_sym
                else
                    time_opts = Dict(:time_method => get(params, :time_method, "regular"))
                    tu_meta = assign_time_units(data[!, t_var_sym]; time_opts...)
                    opt_dict[:t_idx] = tu_meta.idx
                    opt_dict[:t_N] = tu_meta.N_cat
                    opt_dict[:t_idx_var] = t_var_sym
                end
            else
                @warn "Temporal index ':$t_var_sym' not found."
            end
        end

    elseif structure == :smooth
        continuous_kernel_models = [:gp, :rff, :fitc, :svgp, :nystrom, :warp, :spde, :kriging]
        if model_name in continuous_kernel_models
            if all(v -> hasproperty(data, Symbol(v)), variables)
                coords = Matrix{Float64}(data[!, Symbol.(variables)])
                params[:coords] = coords
                if model_name in [:fitc, :svgp, :nystrom]
                    n_inducing = get(params, :n_inducing, min(100, size(coords, 1)))
                    params[:n_inducing] = n_inducing
                    params[:Z_inducing] = generate_inducing_points(coords, n_inducing)
                end
            else
                error("Coordinate variables for smooth model not found in data: $(variables)")
            end
        else
            process_smooth_module!(opt_dict, mod_data, opt_dict[:basis_matrices], opt_dict[:components])
        end
    
    elseif string(structure) == "spacetime"
        process_spacetime_module!(opt_dict, mod_data, registries, hyperpriors)
        return false
        
    else
        @warn "Processing for structure ':$structure' is not fully implemented in `process_random_module!`. A default component will be created."
    end

    return true
end


"""
    process_interact_module!(opt_dict, mod_data, registries, hyperpriors)

Processes interaction modules created by operators like `|>`, `∘`, and `⊗`.

# Rationale
This function is a core part of the component processing pipeline and is not
deprecated. It acts as the central dispatcher for all compositional syntax,
enabling complex model structures. It correctly identifies various interaction
patterns and orchestrates their setup by calling other processors or setting global
configuration flags. This version is consistent with the refactored architecture,
including the use of `process_random_module!` for establishing the context of
child components.

# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the interaction module.
- `registries`, `hyperpriors`: Additional configuration dictionaries.

# Returns
- `true` if a component object should be created for this interaction.
- `false` if the interaction is handled globally (e.g., Kronecker product).
"""
function process_interact_module!(opt_dict, mod_data, registries, hyperpriors)
    op = get(mod_data, :operator, get(mod_data[:params], :operator, nothing))
    components = get(mod_data, :components, get(mod_data[:params], :components, []))
    if isnothing(op) || isempty(components); return false; end
    
    if op == :composition && length(components) == 2
        outer_node, inner_node = components[1], components[2]
        
        if outer_node.module_type == :pointprocess
            return true
        end

        is_nonstationary_variance = outer_node.module_type == :random && get(outer_node.args, :structure, :none) == :spatial && inner_node.module_type == :random && get(inner_node.args, :structure, :none) == :smooth
        if is_nonstationary_variance
            modifier_vars = get(inner_node.args, :positional_args, [])
            if isempty(modifier_vars); @warn "The modifier component (smooth) of a composition operator is missing variables. Skipping."; return false; end
            
            smooth_mod_data = Dict(:type => :smooth, :variables => modifier_vars, :params => inner_node.args)
            process_smooth_module!(opt_dict, smooth_mod_data, opt_dict[:basis_matrices], opt_dict[:components])
            
            mod_data[:type] = :nonstationary_variance
            mod_data[:params][:base_node] = outer_node
            mod_data[:params][:modifier_node] = inner_node
            mod_data[:params][:modifier_basis_key] = Symbol(join(modifier_vars, "_"))
            
            return true 
        end
    end

    if op == :pipe && length(components) == 2
        node1, node2 = components[1], components[2]
        
        if node1.module_type == :random && !haskey(node1.args, :structure)
            args1 = copy(node1.args); args1[:vars] = get(node1.args, :positional_args, [])
            node1.args[:structure] = _infer_structure_from_args(args1)
        end
        if node2.module_type == :random && !haskey(node2.args, :structure)
            args2 = copy(node2.args); args2[:vars] = get(node2.args, :positional_args, [])
            node2.args[:structure] = _infer_structure_from_args(args2)
        end

        is_spatially_varying_curve = node1.module_type == :random && get(node1.args, :structure, :none) == :smooth &&
                                     node2.module_type == :random && get(node2.args, :structure, :none) == :spatial

        is_svc = node1.module_type == :fixed && node2.module_type == :random && get(node2.args, :structure, :none) == :spatial
        is_tvc = node1.module_type == :fixed && node2.module_type == :random && get(node2.args, :structure, :none) == :temporal
        is_svar = node1.module_type == :random && get(node1.args, :structure, :none) == :temporal && node2.module_type == :random && get(node2.args, :structure, :none) == :spatial

        if is_spatially_varying_curve
            dynamic_node = node1
            dynamic_vars = get(dynamic_node.args, :positional_args, [])
            if isempty(dynamic_vars); error("The dynamic part of a pipe operator (e.g., a smoother) must have a variable."); end
            
            smooth_mod_data = Dict(:type => :smooth, :variables => dynamic_vars, :params => dynamic_node.args)
            process_smooth_module!(opt_dict, smooth_mod_data, opt_dict[:basis_matrices], opt_dict[:components])
            
            state_node = node2
            # Corrected call: Use process_random_module! instead of the deprecated process_spatial_module!
            spatial_mod_data = Dict(:type => :random, :variables => get(state_node.args, :positional_args, []), :params => state_node.args)
            process_random_module!(opt_dict, spatial_mod_data, registries, hyperpriors)
            
            mod_data[:params][:dynamic_component_node] = dynamic_node
            mod_data[:params][:state_component_node] = state_node
            
            return true

        elseif is_svc
            covariate_node = node1
            spatial_node = node2
            cov_args = get(covariate_node.args, :positional_args, [])
            if isempty(cov_args); @warn "SVC model is missing a covariate. Skipping."; return false; end
            covariate_name = cov_args[1]
            
            mod_data[:type] = :svc
            mod_data[:variables] = [covariate_name, get(spatial_node.args, :positional_args, [])...]
            mod_data[:params][:covariate] = covariate_name
            mod_data[:params][:spatial_model_spec] = spatial_node
            return true

        elseif is_tvc
            covariate_node = node1
            temporal_node = node2
            cov_args = get(covariate_node.args, :positional_args, [])
            if isempty(cov_args); @warn "TVC model is missing a covariate. Skipping."; return false; end
            covariate_name = cov_args[1]

            mod_data[:type] = :tvc
            mod_data[:variables] = [covariate_name, get(temporal_node.args, :positional_args, [])...]
            mod_data[:params][:covariate] = covariate_name
            mod_data[:params][:temporal_model_spec] = temporal_node
            return true
            
        elseif is_svar
            temporal_node = node1
            spatial_node = node2
            
            process_random_module!(opt_dict, Dict(:type => :temporal, :params => temporal_node.args, :variables => get(temporal_node.args, :positional_args, [])), registries, hyperpriors)
            process_random_module!(opt_dict, Dict(:type => :spatial, :params => spatial_node.args, :variables => get(spatial_node.args, :positional_args, [])), registries, hyperpriors)

            mod_data[:type] = :svar
            mod_data[:params][:rho_spatial_node] = spatial_node
            mod_data[:params][:base_temporal_node] = temporal_node
            return true
        end
    end

    if op == :kronecker_product
        if haskey(mod_data[:params], :sigma)
            prior_val = mod_data[:params][:sigma]
            calling_mod = get(opt_dict, :calling_module, Main)
            if prior_val isa Tuple
                opt_dict[:st_interaction_sigma_prior] = create_pc_prior(:sigma, prior_val)
            elseif prior_val isa Expr
                opt_dict[:st_interaction_sigma_prior] = Core.eval(calling_mod, prior_val)
            else
                opt_dict[:st_interaction_sigma_prior] = prior_val
            end
        end
        
        if length(components) == 2
            s_idx = get(opt_dict, :s_idx, nothing); t_idx = get(opt_dict, :t_idx, nothing); s_N = get(opt_dict, :s_N, nothing)
            if !isnothing(s_idx) && !isnothing(t_idx) && !isnothing(s_N); mod_data[:params][:indices] = [(t - 1) * s_N + s for (s, t) in zip(s_idx, t_idx)];
            else @warn "Could not compute Kronecker product indices for '$(mod_data[:variables])'. Ensure spatial and temporal components are defined."; end
            
            c1_type = get(components[1], :module_type, :unknown); c2_type = get(components[2], :module_type, :unknown)
            
            spatial_node = c1_type == :random && get(components[1].args, :structure, :none) == :spatial ? components[1] : (c2_type == :random && get(components[2].args, :structure, :none) == :spatial ? components[2] : nothing)
            temporal_node = c1_type == :random && get(components[1].args, :structure, :none) == :temporal ? components[1] : (c2_type == :random && get(components[2].args, :structure, :none) == :temporal ? components[2] : nothing)
            
            if !isnothing(spatial_node) && !isnothing(temporal_node)
                spatial_model_str = string(get(spatial_node.args, :model, :iid)); temporal_model_str = string(get(temporal_node.args, :model, :iid))
                has_structured_space = spatial_model_str != "iid"; has_structured_time = temporal_model_str != "iid"
                if has_structured_space && has_structured_time; opt_dict[:model_st] = "IV";
                elseif !has_structured_space && has_structured_time; opt_dict[:model_st] = "II";
                elseif has_structured_space && !has_structured_time; opt_dict[:model_st] = "III";
                else opt_dict[:model_st] = "I"; end
            end
            return false
        else
            @warn "Kronecker product with more than 2 components is not yet supported in process_interact_module!."
        end
    end
    
    return true
end




 """
    _process_mosaic_grouping!(opt_dict, mod_data)

A helper function to handle the grouping logic for mosaic models. It partitions
the spatial domain based on the `mosaic` parameter in a `random()` call.

This function is introduced to centralize the logic for creating mosaic groups.
It supports two modes:
1.  `:kmeans`: Performs k-means clustering on the spatial coordinates.
2.  A `Symbol` pointing to a column in the data that contains pre-defined group assignments.

It creates a new column in the main DataFrame for the observation-level group
assignments and returns the necessary information for the main processor to create
the hierarchical model components.
"""
function _process_mosaic_grouping!(opt_dict, mod_data)
    s_N = get(opt_dict, :s_N, 0)
    if s_N == 0
        error("Mosaic models require a spatial context (`s_N`) to be established first.")
    end
    
    data = opt_dict[:data]
    params = mod_data[:params]
    mosaic_param = get(params, :mosaic, :none)
    
    cluster_assignments::Vector{Int}
    n_regions::Int
    
    group_col_name = Symbol("mosaic_group_for_", mod_data[:key])

    if mosaic_param == :kmeans
        # --- K-Means Clustering Logic ---
        if !haskey(opt_dict, :centroids)
            if hasproperty(data, :s_x) && hasproperty(data, :s_y) && hasproperty(data, :s_idx)
                coord_map = Dict{Int, Point2D}()
                for i in 1:nrow(data)
                    idx = data.s_idx[i]
                    if !haskey(coord_map, idx)
                        coord_map[idx] = Point2D(data.s_x[i], data.s_y[i])
                    end
                end
                if length(coord_map) < s_N
                    error("Mosaic k-means requires coordinates for all $(s_N) spatial units, but only found $(length(coord_map)).")
                end
                opt_dict[:centroids] = [coord_map[i] for i in 1:s_N]
            else
                error("Mosaic k-means requires centroids or `s_x`/`s_y` coordinates.")
            end
        end
        
        centroids = opt_dict[:centroids]
        if length(centroids) != s_N
            error("Number of centroids ($(length(centroids))) does not match s_N ($(s_N)).")
        end

        n_regions_raw = get(params, :n_regions, 5)
        n_regions_req::Int = if n_regions_raw isa Symbol
            calling_mod = get(opt_dict, :calling_module, Main)
            try
                Core.eval(calling_mod, n_regions_raw)
            catch e
                error("Could not evaluate `n_regions` variable `:$(n_regions_raw)`. Ensure it is defined. Error: $e")
            end
        elseif n_regions_raw isa Int
            n_regions_raw
        else
            error("`n_regions` must be an Integer or a Symbol pointing to an Integer. Got: $(typeof(n_regions_raw))")
        end

        if length(centroids) < n_regions_req
            n_regions_req = length(centroids)
        end
        
        centroids_matrix = hcat([c.x for c in centroids], [c.y for c in centroids])'
        kmeans_result = kmeans(centroids_matrix, n_regions_req; maxiter=200, display=:none)
        
        cluster_assignments = assignments(kmeans_result)
        n_regions = nclusters(kmeans_result)

    elseif mosaic_param isa Symbol
        # --- Pre-defined Grouping Column Logic ---
        if !hasproperty(data, mosaic_param)
            error("The specified mosaic grouping column `:$(mosaic_param)` was not found in the data.")
        end
        group_data = data[!, mosaic_param]
        unique_levels = unique(group_data)
        n_regions = length(unique_levels)
        level_map = Dict(level => i for (i, level) in enumerate(unique_levels))
        
        s_idx_to_group = Dict{Int, Int}()
        for i in 1:nrow(data)
            s_idx_i = data.s_idx[i]
            group_val = group_data[i]
            if !haskey(s_idx_to_group, s_idx_i)
                s_idx_to_group[s_idx_i] = level_map[group_val]
            end
        end
        cluster_assignments = [get(s_idx_to_group, i, 1) for i in 1:s_N]
    else
        error("Invalid `mosaic` parameter. Must be `:kmeans` or a Symbol pointing to a grouping column.")
    end

    # Create the observation-level grouping column needed by the `mixed` processor.
    opt_dict[:data][!, group_col_name] = cluster_assignments[opt_dict[:data].s_idx]
    
    return (group_col_name=group_col_name, n_regions=n_regions)
end


"""
    process_sciml_module!(opt_dict, mod_data, registries, hyperpriors)

Processes the `sciml()` module call from the formula.

# Rationale
This function is a necessary component processor for integrating models from the SciML
ecosystem. It is not deprecated and is consistent with the refactored architecture.
Its primary role is to validate the arguments provided to the `sciml()` module and
to establish the necessary temporal context (`t_idx`, `t_N`, `t_coords`) in the main
model configuration. It ensures that all required parameters for defining a SciML
problem (`model_func`, `u0_prior`, `p_priors`, `tspan`, `solver`) are present before
the model building phase.

# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the `sciml()` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries (not used here).

# Returns
- `true` to indicate that a `SciMLComponent` object should be created.
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



"""
    MODULE_PROCESSORS

A constant dictionary that serves as the central dispatch table for processing modules
parsed from the `bstm` formula. This registry maps a module's type (a `Symbol` like
`:random` or `:mixed`) to its corresponding processor function.

# Rationale
This dispatch mechanism is a core element of the refactored component-based
architecture. It replaces a monolithic series of `if/elseif` checks with a clean,
extensible, and maintainable system. When the `bstm_config` function encounters a
module in the parsed formula, it looks up the module's type in this dictionary and
calls the associated function to perform data-dependent setup and validation.

This design is consistent with the refactor's goals:
- **Modularity**: Each module's setup logic is encapsulated in its own processor function.
- **Consolidation**: It reflects the consolidation of older modules like `:spatial`,
  `:temporal`, and `:smooth` into the unified `:random` module, which is handled by
  `process_random_module!`.
- **Extensibility**: New modules can be added to the framework by defining a new
  processor function and adding an entry to this dictionary.

This constant is the definitive registry for the current system.
"""
const MODULE_PROCESSORS = Dict{Symbol, Function}(
    :random => process_random_module!,
    :fixed => process_fixed_module!,
    :mixed => process_mixed_module!,
    :nested => process_nested_module!,
    :eigen => process_eigen_module!,
    :dynamics => process_dynamics_module!,
    :interact => process_interact_module!,
    :pointprocess => process_pointprocess_module!,
    :custom => process_custom_module!,
    :sciml => process_sciml_module!
)
