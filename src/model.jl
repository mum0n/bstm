


function Base.:|>(m1::Component, m2::Component)
    return Composed([m1, m2], :pipe)
end

composition(m1::Component, m2::Component) = Composed([m1, m2], :composition)
∘(m1::Component, m2::Component) = Composed([m1, m2], :composition)

otimes(m1::Component, m2::Component) = Composed([m1, m2], :kronecker_product)
⊗(m1::Component, m2::Component) = Composed([m1, m2], :kronecker_product)

 
"""
    apply_transformation(func_name::Symbol, data_vector::AbstractVector)

Applies a specified data transformation to a vector.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function is a core utility for the formula parsing engine, acting as the
implementation backend for the transformation syntax (e.g., `zscore(x) |> ...`).
It is called by `_rewrite_transformations!` to apply common pre-processing steps
to covariate data directly within the model formula. This centralizes the
transformation logic and provides a consistent, extensible interface for adding
new transformations.

# Mathematical Formulations
- **`:zscore`**: \$ (x - \\text{mean}(x)) / \\text{std}(x) \$
- **`:log`**: \$ \\log(x - \\min(x) + 1) \$
- **`:center`**: \$ x - \\text{mean}(x) \$
- **`:scale`**: \$ (x - \\min(x)) / (\\max(x) - \\min(x)) \$

# Arguments
- `func_name::Symbol`: The name of the transformation (e.g., `:zscore`, `:log`).
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

Recursively traverses the formula's Abstract Syntax Tree (AST), applies data
transformations defined by pipe operators (`|>`), and rewrites the AST nodes to
use the newly created data columns.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function is a core part of the formula parsing engine. It enables a clean,
expressive syntax for pre-processing covariates directly within the model formula
(e.g., `zscore(temperature) |> random(...)`). It works by traversing the parsed AST
and identifying `pipe` operators where the left-hand side is a recognized
transformation function (e.g., `:zscore`, `:log`).

When such a transformation is found, this function:
1.  Applies the transformation to the specified data column.
2.  Adds a new column to the input `DataFrame` to store the transformed data.
3.  Rewrites the right-hand side of the pipe operator to use this new column as its
    input variable.

This process ensures that all subsequent model components operate on the correctly
transformed data, without requiring the user to manually pre-process their data frame.

# Arguments
- `nodes::Vector`: A vector of AST nodes to be processed.
- `data::DataFrame`: The input data frame. **This argument is mutated** by adding
  new columns for the transformed data.

# Returns
- `Vector`: A new vector of rewritten AST nodes.
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
    generate_rff_params(in_dims::Int, n_features::Int, lengthscale::Union{Real, AbstractVector}, kernel_name::String)

Generates random projection weights (W) and biases (b) for the Random Fourier
Features (RFF) approximation. This version is CPU-only.
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





"""
    split_terms_at_depth(input::AbstractString, sep::AbstractString)

Splits a string by a given separator, but only when the separator is not nested
inside parentheses `()` or brackets `[]`.

# Version
v1.1.2 (2026-08-14)

# Rationale
This function is a core utility for the formula parsing engine. It is responsible
for correctly decomposing a formula string into its top-level additive terms
(e.g., splitting `a + b` but not `random(a, b)`). It achieves this by maintaining
a `depth` counter to track nesting within parentheses and brackets, ensuring that
the separator `sep` is only acted upon when at the top level (depth 0).

This version corrects a `UndefVarError` by replacing a typo (`current_arg`) with
the correct variable name (`current_term`).

# Arguments
- `input::AbstractString`: The string to be split.
- `sep::AbstractString`: The separator string to split by.

# Returns
- `Vector{String}`: A vector of the resulting terms.
"""
function split_terms_at_depth(input::AbstractString, sep::AbstractString)
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
    _infer_structure_from_args(variables, params)

Infers the structural type of a `random()` component (e.g., :spatial, :temporal)
based on the model name and the variables provided.
"""
function _infer_structure_from_args(variables::Vector, params::Dict)::Symbol
    model_name = get(params, :model, :iid) 

    # Use the central registry to find the structure for unambiguous models
    if haskey(MODEL_TO_STRUCTURE_MAP, model_name)
        return MODEL_TO_STRUCTURE_MAP[model_name]
    end
    
    # Fallback for legacy or unregistered models
    if haskey(KNOWN_UNAMBIGUOUS_MODELS, model_name)
        return KNOWN_UNAMBIGUOUS_MODELS[model_name]
    end
    
    # For truly ambiguous models, infer from variable names
    if model_name in AMBIGUOUS_MODELS
        if any(v -> occursin("year", string(v)) || occursin("time", string(v)), variables)
            return :temporal
        elseif any(v -> occursin(r"lon|lat|coord"i, string(v)), variables)
            return :smooth
        elseif any(v -> occursin(r"region|idx|area"i, string(v)), variables)
            return :spatial
        end
    end
    
    # Default to a smooth structure if no other context is available
    return :smooth 
end



"""
    _parse_arguments_from_expr(args::Vector{Any})

Parses the arguments from a Julia expression (specifically, the `args` field of a
`:call` expression) into a dictionary of keyword arguments and a list of positional
arguments.

# Version
v1.1.0 (2026-08-13)

# Rationale
This function is a core part of the formula parsing engine that works directly with
Julia's Abstract Syntax Tree (AST). It correctly distinguishes between positional
arguments (like variable names) and keyword arguments (like `model=:bym2`). It
preserves expressions (e.g., `prior=Normal(0,1)`) for later evaluation, which is
crucial for allowing complex objects as parameters.   use
the key `:positional_args` for positional arguments, ensuring consistency with the
rest of the parsing engine.

# Arguments
- `args`: A vector of arguments from an `Expr` object.

# Returns
- A `Dict{Symbol, Any}` where keyword arguments are stored by their key, and
  positional arguments are stored under the `:positional_args` key.
"""
function _parse_arguments_from_expr(args::Vector{Any})
    parsed_args = Dict{Symbol, Any}()
    positional_args = []

    for arg in args
        if arg isa Expr && arg.head == :kw
            # This is a keyword argument, e.g., `model=:bym2`.
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

    # Store positional arguments under the conventional `:positional_args` key for consistency.
    if !isempty(positional_args)
        parsed_args[:positional_args] = positional_args
    end

    return parsed_args
end


"""
    _parse_value(val_str::AbstractString)

Parses a string value from a formula argument into an appropriate Julia type.

# Version
v1.0.2 (2026-08-13)

# Rationale
This function is a core utility for the formula parsing engine. It is responsible
for converting the string representation of a value from a module's argument list
into a concrete Julia type (`Symbol`, `String`, `Bool`, `Number`, or `Expr`). This
is a critical step that enables the `bstm` formula to accept a wide range of
user inputs.   more robust in parsing symbol literals,
ensuring that `model=:bym2` is always parsed as the symbol `:bym2`.

# Arguments
- `val_str::AbstractString`: The string value to parse.

# Returns
- The parsed Julia object. The type can be `Symbol`, `String`, `Bool`, `Number`,
  or `Expr`.
"""
function _parse_value(val_str::AbstractString)
    val_str = strip(val_str)

    # Priority 1: Handle symbol literals like `:foo`.
    if startswith(val_str, ":")
        return Symbol(val_str[2:end])
    # Priority 2: Handle quoted string literals.
    elseif (startswith(val_str, "'") && endswith(val_str, "'")) || (startswith(val_str, "\"") && endswith(val_str, "\""))
        return String(val_str[2:end-1])
    # Priority 3: Handle boolean literals.
    elseif val_str == "true"
        return true
    elseif val_str == "false"
        return false
    # Priority 4: Handle bare words that are valid identifiers (e.g., variable names).
    elseif occursin(r"^[a-zA-Z_][a-zA-Z0-9_]*$", val_str)
        return Symbol(val_str)
    else
        # Fallback: Attempt to parse as a Julia expression (e.g., a number, a vector, or a function call like `Normal(0,1)`).
        try
            return Meta.parse(val_str)
        catch
            # If all else fails, treat it as a string that was not quoted.
            return String(val_str)
        end
    end
end

_parse_value(val_str::SubString{String}) = _parse_value(String(val_str))


"""
    _add_parsed_arg!(args_dict::Dict{Symbol, Any}, positional_args::Vector{Any}, arg_val::AbstractString)

A helper function that parses a single argument string and adds it to either the
keyword argument dictionary or the positional argument list.

# Version
v1.0.2 (2026-08-13)

# Rationale
This function is a core utility for the formula parsing engine. It centralizes the
logic for distinguishing between `key=value` pairs and positional arguments within
a module's call string.   accept `AbstractString` to
handle both `String` and `SubString` types, resolving a potential `MethodError`,
and its documentation has been expanded for clarity.

# Arguments
- `args_dict::Dict{Symbol, Any}`: The dictionary to which keyword arguments will be added.
- `positional_args::Vector{Any}`: The vector to which positional arguments will be added.
- `arg_val::AbstractString`: The raw argument string to parse (e.g., "model=bym2" or "s_idx").

# Returns
- `nothing`. The `args_dict` and `positional_args` collections are mutated.
"""
function _add_parsed_arg!(args_dict::Dict{Symbol, Any}, positional_args::Vector{Any}, arg_val::AbstractString)
    if contains(arg_val, "=")
        # It's a keyword argument.
        key_val = Base.split(arg_val, "=", limit=2)
        key = Symbol(Base.strip(key_val[1]))
        val_str = String(Base.strip(key_val[2]))
        args_dict[key] = _parse_value(val_str)
    else
        # It's a positional argument.
        push!(positional_args, _parse_value(arg_val))
    end
end


"""
    _parse_arguments_string(args_str::String)

Parses the inner content of a module call string (e.g., "s_idx, model=bym2") into
a dictionary of keyword arguments and a list of positional arguments.

# Version
v1.1.1 (2026-08-13)

# Rationale
This function is a core utility for the formula parsing engine. It is responsible
for breaking down the argument string from within a module's parentheses into a
structured dictionary.   correctly handle commas within
string literals, preventing incorrect splitting of arguments. The docstring has
also been expanded to detail the internal workflow.

# Workflow
The function implements a simple state machine that iterates through the argument
string character by character:
1.  It maintains a `depth` counter for parentheses `()` and brackets `[]`. A comma
    is only considered a top-level argument separator if `depth` is zero.
2.  It maintains an `in_string` flag to detect whether the parser is currently
    inside a single or double-quoted string. Commas encountered while `in_string`
    is true are ignored as separators.
3.  When a valid top-level comma is found, the accumulated argument string is
    processed by `_add_parsed_arg!`, which separates it into either a positional
    or keyword argument.
4.  After the loop, the final accumulated argument is processed.
5.  Positional arguments are collected and stored in the final dictionary under the
    key `:positional_args`.

# Arguments
- `args_str::String`: The string of arguments from inside a module's parentheses.

# Returns
- `Dict{Symbol, Any}`: A dictionary containing parsed keyword arguments and a
  `:positional_args` key for positional arguments.
"""
function _parse_arguments_string(args_str::String)
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


"""
    _sanitize_variablename(name::String)

Sanitizes a string to be a valid Julia variable name, suitable for use in
dynamically generated code.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function is a core utility for the formula parsing engine. The parser can
generate keys for model components from formula syntax that includes characters
invalid in Julia variable names (e.g., `cov |> spatial` might produce a key
containing `|`). This function cleans those keys before they are used by the
code generator to create variable names for latent effects, priors, and other
parameters within the Turing `@model`.

It specifically handles:
- Standard invalid characters: `|`, `(`, `)`, `+`, `*`, `&`, `:`, and whitespace.
- Special `bstm` operators: `⊗` (Kronecker product) and `∘` (composition).
- Formatting artifacts: Ensures no leading/trailing underscores and collapses
  multiple consecutive underscores into a single one.

# Arguments
- `name::String`: The raw string to be sanitized.

# Returns
- `String`: A sanitized version of the name suitable for use as a variable.
"""
function _sanitize_variablename(name::String)
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



"""
    _parse_single_component_term(term_str::AbstractString)

Parses a single module call string (e.g., "random(s_idx, model=bym2)") into its
constituent parts: the module name and its arguments.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function is a low-level utility within the formula parsing engine. It is
responsible for the initial decomposition of a single component call. It uses a
regular expression to reliably separate the component's name from the string
containing its arguments. The argument string is then passed to
`_parse_arguments_string` for more detailed parsing. This separation of concerns
makes the parsing engine more modular and easier to debug.

# Arguments
- `term_str::AbstractString`: The string representing a single module call.

# Returns
- A `NamedTuple` of the form `(module_type::Symbol, args::Dict)`.
"""
function _parse_single_component_term(term_str::AbstractString)
    term_str = Base.strip(term_str)
    # Regex to capture the module name and the content inside the parentheses.
    m = match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$", term_str)
    
    if m === nothing
        # This error is thrown when a term without parentheses is incorrectly passed
        # to this function, indicating a logic error in the parent parser.
        error("Internal Parser Error: _parse_single_component_term was called with a non-module term '$term_str'.")
    end

    module_name = Symbol(m.captures[1])
    args_str = String(m.captures[2])
    args_dict = _parse_arguments_string(args_str)

    return (module_type = module_name, args = args_dict)
end



"""
    resolve_hyperpriors(model_name::String, global_priors::Dict, local_params::Dict, scheme::Symbol, calling_mod::Module)

Resolves the prior distribution for each hyperparameter of a model component by
checking for specifications at the local, global, and scheme levels.

# Version
v1.1.0 (2026-08-13)

# Rationale
This function is a core utility in the model configuration pipeline. It implements a
3-level precedence system to resolve the prior distribution for every hyperparameter
of a model component:
1.  **Local**: A prior specified directly in a component's formula call (e.g.,
    `random(..., sigma=Normal(0,1))`).
2.  **Global**: A prior specified in the `hyperpriors` dictionary passed to `@bstm`.
3.  **Scheme**: A default prior from a pre-defined scheme (`:pcpriors`,
    `:informative`, `:uninformative`).

  comprehensive, resolving priors for a wide range of
parameters used across all `bstm` components, including those for spatial, temporal,
smoother, and mechanistic models. It correctly handles anisotropic priors for
multi-dimensional components, normalizes common aliases (e.g., `ls` for `lengthscale`),
and evaluates user-provided expressions in the correct module scope.

# Arguments
- `model_name::String`: The name of the component model being processed.
- `global_priors::Dict`: A dictionary of globally specified hyperpriors.
- `local_params::Dict`: A dictionary of parameters from the component's formula call.
- `scheme::Symbol`: The active prior scheme (e.g., `:pcpriors`).
- `calling_mod::Module`: The module context for evaluating symbols or expressions.

# Returns
- A `NamedTuple` containing the resolved prior distributions for the component.
"""
function resolve_hyperpriors(model_name::String, global_priors::Dict, local_params::Dict, scheme::Symbol, calling_mod::Module)
    prior_defaults = if scheme == :pcpriors
        PC_PRIORS
    elseif scheme == :informative
        INFORMATIVE_PRIORS
    else
        UNINFORMATIVE_PRIORS
    end

    # Normalize aliases for consistency (e.g., ls -> lengthscale).
    local_params_norm = Dict(local_params)
    if haskey(local_params_norm, :ls)
        local_params_norm[:lengthscale] = local_params_norm[:ls]
        delete!(local_params_norm, :ls)
    end

    is_anisotropic = get(local_params_norm, :anisotropic, false)
    in_dims = get(local_params_norm, :in_dims, 0)

    # Comprehensive list of all possible hyperpriors across all components.
    possible_priors = [
        :sigma, :rho, :rho1, :rho2, :unconstrained_rho, :kappa, :lengthscale, 
        :range, :period, :amplitude, :phase, :velocity, :diffusion, :pca_sd, 
        :pdef_sd, :L_corr, :sigma_effects, :r, :K, :q, :M_nat, :alpha, :beta, 
        :gamma, :delta, :curvature
    ]

    resolved = Dict{Symbol, Any}()

    for p_sym in possible_priors
        is_ard_param = p_sym in [:lengthscale, :kappa]

        if is_ard_param && is_anisotropic
            if in_dims == 0; error("Cannot resolve anisotropic prior for '$p_sym' because input dimensionality is unknown."); end
            
            prior_val = get(local_params_norm, p_sym, nothing)
            
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
                else; get(prior_defaults, string(p_sym), nothing); end
                
                if !(single_prior isa Distribution); error("Resolved prior for '$p_sym' is not a Distribution."); end
                resolved[p_sym] = [single_prior for _ in 1:in_dims]
            end
            continue
        end

        if haskey(local_params_norm, p_sym)
            prior_val = local_params_norm[p_sym]
            if prior_val isa Tuple
                resolved[p_sym] = create_pc_prior(p_sym, prior_val)
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

        global_key_model = Symbol(model_name, "_", p_sym)
        global_key_param = p_sym
        
        if haskey(global_priors, global_key_model)
            resolved[p_sym] = global_priors[global_key_model]
        elseif haskey(global_priors, global_key_param)
            resolved[p_sym] = global_priors[global_key_param]
        elseif haskey(prior_defaults, string(p_sym))
            resolved[p_sym] = prior_defaults[string(p_sym)]
        end
    end

    return NamedTuple(resolved)
end



"""
    _is_outermost_grouping_parentheses(s::AbstractString)

Checks if a string is fully and exclusively enclosed by a single pair of
grouping parentheses.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function is a core utility for the recursive descent parser (`_parse_rhs_expression`).
It determines if the parser should strip the outer parentheses from an expression
and recurse on its content. For example, it should return `true` for `(a + b)`
and `((a + b))`, but `false` for an expression like `(a) + (b)` (which would not be
passed to this function directly, as the parser would split on `+` first).

This version corrects a bug in the original implementation where the depth check
incorrectly failed for valid nested expressions. The logic now correctly starts
the depth count at 1 (to account for the initial opening parenthesis) and checks if
the depth returns to zero *before* the end of the inner string, which robustly
identifies expressions that are not exclusively grouped by the outermost parentheses.

# Arguments
- `s::AbstractString`: The string to check.

# Returns
- `Bool`: `true` if the string is fully enclosed by grouping parentheses, `false` otherwise.
"""
function _is_outermost_grouping_parentheses(s::AbstractString)
    if !startswith(s, "(") || !endswith(s, ")")
        return false
    end
    
    # Start depth at 1 to account for the initial opening parenthesis.
    depth = 1
    
    # Iterate over the content *inside* the first and last parentheses.
    inner_s = SubString(s, nextind(s, 1), prevind(s, lastindex(s)))
    
    for (idx, char) in enumerate(inner_s)
        if char == '('
            depth += 1
        elseif char == ')'
            depth -= 1
        end
        
        # If depth becomes zero before the end of the inner string,
        # it means a closing parenthesis matches the initial opening one
        # prematurely. e.g., for "(a) + (b)", depth becomes 0 after ')'.
        # This indicates the outer parentheses are not the sole grouping operator.
        if depth == 0 && idx < length(inner_s)
            return false
        end
    end
    
    # After iterating through the whole inner string, the depth must be exactly 1
    # to match the final closing parenthesis that was not part of the iteration.
    # A final depth of 0 would mean an imbalance like `(a))`.
    return depth == 1
end


"""
    _parse_rhs_expression(term_str::AbstractString)

Recursively parses a term from the right-hand side (RHS) of a formula into an
Abstract Syntax Tree (AST) node, respecting operator precedence and grouping.

# Version
v1.1.0 (2026-08-13)

# Rationale
This function is a recursive descent parser that forms the core of the formula
parsing engine. It is responsible for translating a formula string into a structured
tree that the model configuration engine can interpret.

This version corrects two critical issues from the previous implementation:
1.  **Operator Precedence**: The order of parsing has been reversed to correctly
    implement operator precedence. The parser now splits by the lowest-precedence
    operator first, in the order: `∘` (composition), then `⊗` (Kronecker product),
    then `|>` (pipe). This ensures that expressions are grouped correctly according
    to standard mathematical and programming language conventions.
2.  **Parentheses Handling**: A robust helper function, `_is_fully_enclosed`, has
    been added to correctly identify and strip grouping parentheses. This prevents
    parsing errors for complex, nested expressions like `(a |> b) ⊗ c`.

# Operator Precedence & Associativity
- **Composition (`∘`)**: Lowest precedence, right-associative. `a ∘ b ∘ c` is parsed as `a ∘ (b ∘ c)`.
- **Kronecker Product (`⊗`)**: Medium precedence, left-associative. `a ⊗ b ⊗ c` is parsed as `(a ⊗ b) ⊗ c`.
- **Pipe (`|>`)**: Highest precedence, left-associative. `a ⊗ b |> c` is parsed as `a ⊗ (b |> c)`.

# Arguments
- `term_str::AbstractString`: A string representing a single term or a composition of terms from the formula's RHS.

# Returns
- A `NamedTuple` representing a node in the AST.
"""
function _parse_rhs_expression(term_str::AbstractString)
    term_str_stripped = Base.strip(term_str)

    # Helper to check if the string is fully enclosed by a single pair of parentheses.
    function _is_fully_enclosed(s::AbstractString)
        if !startswith(s, "(") || !endswith(s, ")")
            return false
        end
        depth = 0
        # Check the content *inside* the outer parentheses.
        inner_s = SubString(s, nextind(s, 1), prevind(s, lastindex(s)))
        for (idx, char) in enumerate(inner_s)
            if char == '('; depth += 1;
            elseif char == ')'; depth -= 1; end
            
            if depth < 0; return false; end # Unbalanced
            # If depth is zero before the end, the outer parentheses are not the sole group.
            if depth == 0 && idx < length(inner_s); return false; end
        end
        return depth == 0
    end

    # If the expression is wrapped in grouping parentheses, parse the inner content recursively.
    if _is_fully_enclosed(term_str_stripped)
        inner_content = SubString(term_str_stripped, nextind(term_str_stripped, 1), prevind(term_str_stripped, lastindex(term_str_stripped)))
        return _parse_rhs_expression(inner_content)
    end

    # Proceed with parsing based on operator precedence (lowest precedence first).
    parts = split_terms_at_depth(term_str_stripped, " ∘ ")
    if length(parts) > 1
        # Composition is right-associative: a ∘ b ∘ c -> a ∘ (b ∘ c)
        return (type=:operator, op=:composition, children=[_parse_rhs_expression(parts[1]), _parse_rhs_expression(join(parts[2:end], " ∘ "))])
    end

    parts = split_terms_at_depth(term_str_stripped, " ⊗ ")
    if length(parts) > 1
        # Kronecker product is left-associative.
        return (type=:operator, op=:kronecker_product, children=[_parse_rhs_expression(p) for p in parts])
    end

    parts = split_terms_at_depth(term_str_stripped, " |> ")
    if length(parts) > 1
        # Pipe is left-associative: a |> b |> c -> (a |> b) |> c
        return (type=:operator, op=:pipe, children=[_parse_rhs_expression(join(parts[1:end-1], " |> ")), _parse_rhs_expression(parts[end])])
    end

    # If no operators are found, parse as a single module or a fixed effect.
    if occursin(r"\(.*\)", term_str_stripped)
        return _parse_single_component_term(term_str_stripped)
    else
        # Treat bare terms as fixed effects.
        return (module_type = :fixed, args = Dict(:positional_args => [term_str_stripped]))
    end
end




"""
    _generate_unique_module_key(base_key::String, existing_modules::Dict)

Generates a unique, sanitized module key by ensuring it does not conflict with
existing keys in the `existing_modules` dictionary.

# Arguments
- `base_key::String`: The initial, unsanitized key string.
- `existing_modules::Dict`: The dictionary of modules already processed.

# Returns
- `String`: A unique and sanitized module key.
"""
function _generate_unique_module_key(base_key::String, existing_modules::Dict)
    sanitized_base_key = _sanitize_variablename(base_key)
    module_key = sanitized_base_key
    counter = 1
    while haskey(existing_modules, module_key)
        counter += 1
        module_key = "$(sanitized_base_key)_$(counter)"
    end
    return module_key
end



"""
    _categorize_rhs_nodes!(nodes, modules, fixed_effects)

Recursively traverses the Abstract Syntax Tree (AST) of the right-hand side (RHS)
of a formula, categorizing its nodes into `bstm` modules (components) or bare
fixed effects.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function is a core component of the formula parsing engine. It translates
the structured representation of the formula into a categorized list of components
and fixed effects, which is then used by the model configuration engine. This
version improves the specificity of generated keys for composed modules and
extracts repetitive unique key generation logic into a helper function for
better readability and maintainability.

# Arguments
- `nodes`: A vector of AST nodes representing the RHS of the formula.
- `modules`: A dictionary to store categorized `bstm` modules.
- `fixed_effects`: A list to store bare fixed effect terms.

# Returns
- `nothing`. The `modules` and `fixed_effects` collections are mutated.
"""
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
                module_key = _generate_unique_module_key(raw_key, modules)
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
                    # Include module type in the key for better specificity
                    return isempty(pos_args) ? string(n.module_type) : "$(string(n.module_type))_$(join(string.(pos_args), "_"))"
                else
                    return "unknown"
                end
            end
            raw_key = _get_simplified_composed_node_key(node)
            module_key = _generate_unique_module_key(raw_key, modules)
            
            modules[module_key] = (module_type = :interact, args = Dict(:operator => node.op, :components => node.children))

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

                module_key = _generate_unique_module_key(raw_key, modules)
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


"""
    _parse_lhs_term(term::String)

Parses a single term from the left-hand side (LHS) of the formula string. This
function is a core part of the formula parsing engine.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function is designed to handle the various ways outcome variables can be
specified in a `bstm` formula. It correctly distinguishes between bare outcome
variables and outcomes defined within a `likelihood()` block. It also supports
the `+` operator for specifying multiple outcomes, which is essential for both
multivariate and compositional models. The function's output is a standardized
vector of dictionaries, which provides a clean, intermediate representation for the
downstream model configuration engine.

# Arguments
- `term::String`: A string representing one part of the LHS (e.g., `"y"`,
  `"y1 + y2"`, or `"likelihood(y, family=:poisson)"`).

# Returns
- `Vector{Dict{Symbol, Any}}`: A vector where each dictionary represents a single
  outcome specification. Each dictionary contains:
  - `:var`: The name of the outcome variable as a `String`.
  - `:params`: A `Dict` of parameters parsed from the `likelihood()` block. This
    is empty if the outcome was specified as a bare variable.
"""
function _parse_lhs_term(term::String)
    term = Base.strip(term)
    m = match(r"likelihood\((.*)\)", term)
    specs = Dict{Symbol, Any}[]
    if !isnothing(m)
        # Case 1: The term is a likelihood() block.
        inner_content = m.captures[1]
        args = split_terms_at_depth(inner_content, ",")
        if isempty(args); return specs; end
        
        # The first argument(s) are the outcome variables.
        outcome_var_str = Base.strip(args[1])
        # The rest are keyword parameters for the likelihood.
        params_str = join(args[2:end], ",")
        params = _parse_arguments_string(params_str)
        
        # Handle multiple outcomes specified with `+` inside the likelihood block.
        for ov in [Base.strip(s) for s in Base.split(outcome_var_str, '+')]
            push!(specs, Dict(:var => ov, :params => params))
        end
    else
        # Case 2: The term is one or more bare outcome variables.
        for ov in [Base.strip(s) for s in Base.split(term, '+')]
            push!(specs, Dict(:var => ov, :params => Dict()))
        end
    end
    return specs
end



"""
    decompose_bstm_formula(formula_str::String, data::DataFrame)

Decomposes a `bstm` formula string into its constituent parts, including outcomes,
likelihood specifications, model components, and fixed effects.

# Version
v1.0.2 (2026-08-13)

# Rationale
This function is the main entry point for the formula parsing engine. It translates
the user-facing formula into a structured, intermediate representation that the
model configuration engine can process.   handle formulas
that do not contain a right-hand side (RHS). If the `~` operator is missing, it now
defaults the RHS to "1" (an intercept-only model), preventing a `BoundsError` during
parsing. The docstring has also been expanded to detail the internal workflow.

# Workflow
1.  **LHS/RHS Splitting**: The formula string is split by the `~` operator. If no `~`
    is present, the RHS defaults to `"1"`.
2.  **LHS Parsing**: The LHS is parsed to identify outcome variables and their
    `likelihood()` specifications.
3.  **RHS Parsing**: The RHS is parsed to handle intercept control (`0`, `-1`,
    `intercept()`) and to collect all other terms as model components.
4.  **AST Construction**: An Abstract Syntax Tree (AST) is built from the RHS terms,
    respecting operator precedence (`|>`, `⊗`, `∘`) to correctly represent
    component compositions.
5.  **Data Transformation**: The AST is traversed to find and apply data
    transformations (e.g., `zscore(x)`). The `DataFrame` is mutated to include new
    columns with the transformed data, and the AST is rewritten to use these new
    columns.
6.  **Node Categorization**: The final AST is traversed to categorize each node as
    either a `bstm` module (e.g., `random`, `mixed`) or a bare fixed effect term.
7.  **Output**: A `NamedTuple` is returned containing the structured lists of
    outcomes, modules, fixed effects, and intercept information.

# Arguments
- `formula_str::String`: The model formula string.
- `data::DataFrame`: The input DataFrame, which may be mutated by transformation functions.

# Returns
- A `NamedTuple` with the fields `:outcomes`, `:modules`, `:fixed_effects`,
  `:has_intercept`, and `:intercept_prior`.
"""
function decompose_bstm_formula(formula_str::String, data::DataFrame)
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

 
"""
    _precompute_likelihood_params!(M::Dict)

Ensures all observation-level likelihood parameters are consistently formatted as
matrices of size `(N, K)`.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function is a core utility in the model configuration pipeline. It standardizes
all observation-level parameters (e.g., `offsets`, `weights`, `trials`, censoring
bounds) into a consistent `(N, K)` matrix format, where `N` is the number of
observations and `K` is the number of outcomes. This simplifies the downstream code
generator, which can then access these parameters with simple indexing `[i, k]`
inside the model's observation loop, avoiding inefficient runtime conditional checks
and ensuring dimensional correctness.

# Arguments
- `M::Dict`: The model configuration dictionary, which is mutated by this function.

# Returns
- `nothing`.
"""
function _precompute_likelihood_params!(M::Dict)
    N = M[:y_N]
    K = M[:outcomes_N]

    param_specs = [
        (key=:censor_lower, default=-Inf, is_scalar_per_outcome=true),
        (key=:censor_upper, default=Inf, is_scalar_per_outcome=true),
        (key=:hurdle, default=-Inf, is_scalar_per_outcome=true),
        (key=:trials, default=1, is_scalar_per_outcome=false),
        (key=:weights, default=1.0, is_scalar_per_outcome=false),
        (key=:log_offsets, default=0.0, is_scalar_per_outcome=false)
    ]

    for spec in param_specs
        key = spec.key
        default_val = spec.default
        is_scalar_per_outcome = spec.is_scalar_per_outcome

        final_matrix = Matrix{typeof(default_val)}(undef, N, K)

        if !haskey(M, key)
            fill!(final_matrix, default_val)
        else
            val = M[key]
            if val isa Real
                # Handle scalar input for both parameter types.
                fill!(final_matrix, val)
            elseif is_scalar_per_outcome
                # Parameter is defined per outcome, broadcast across observations.
                if val isa AbstractVector && length(val) == K
                    final_matrix = repeat(val', N, 1)
                else
                    @warn "Scalar-per-outcome parameter `:$key` has unexpected type or dimensions. Expected a scalar or a vector of length $K. Using default."
                    fill!(final_matrix, default_val)
                end
            else # Parameter is defined per observation.
                if val isa AbstractVector && length(val) == N
                    final_matrix = repeat(val, 1, K)
                elseif val isa AbstractMatrix && size(val) == (N, K)
                    final_matrix = val
                else
                    @warn "Per-observation parameter `:$key` has incorrect dimensions. Expected a scalar, vector of length $N, or matrix of size ($N, $K). Using default."
                    fill!(final_matrix, default_val)
                end
            end
        end
        M[key] = final_matrix
    end
end



 

"""
    bstm_config(formula::String, data::DataFrame; calling_module::Module=Main, kwargs...)

Constructs the complete model configuration from a formula and data. 
"""
function bstm_config(
    formula::String, data::DataFrame; calling_module::Module=Main, kwargs...
)
    df_processed = deepcopy(data)
    decomposed_formula = decompose_bstm_formula(formula, df_processed)

    M = _initialize_config(
        df_processed,
        merge(Dict(kwargs), Dict(:calling_module => calling_module))
    )
    M[:formula] = formula
    
    _process_lhs!(M, decomposed_formula.outcomes)
    
    is_multivariate = get(M, :model_arch, "univariate") == "multivariate"
    if is_multivariate
        for (key, mod_data_nt) in decomposed_formula.modules
            model_name = get(mod_data_nt.args, :model, :none)
            if mod_data_nt.module_type == :dynamics && model_name in [
                :leslie_matrix, :delay_difference, :generalized_lotka_volterra,
                :generalized_leslie_matrix
            ]
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

    for (key, mod_data_nt) in decomposed_formula.modules
        mod_type = mod_data_nt.module_type
        mod_data_dict = Dict(
            :key => key,
            :type => mod_type,
            :variables => get(mod_data_nt.args, :positional_args, []),
            :params => mod_data_nt.args
        )

        processor! = get(MODULE_PROCESSORS, mod_type, nothing)
        
        create_component = true
        if !isnothing(processor!)
            create_component = processor!(M, mod_data_dict, M, M[:hyperpriors])
            if mod_type == :random && !haskey(mod_data_dict[:params], :structure)
                mod_data_dict[:params][:structure] = _infer_structure_from_args(mod_data_dict[:params])
            end
        end

        if !create_component; continue; end

        component_obj = resolve_technical_primitive(
            mod_data_dict, M, M[:hyperpriors], M[:prior_scheme]
        )
        mod_data_dict[:component_obj] = component_obj

        M_nt = NamedTuple(M)
        precomputes = get_precomputes(component_obj, M_nt, mod_data_dict)

        spec = (
            key=Symbol(key), 
            structure=mod_data_dict[:params][:structure], 
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

    # Pre-compute Cholesky factorizations for static components.
    _precompute_static_components!(M)

    _finalize_config!(M)
    
    return NamedTuple(M)
end
 



"""
    generate_full_variable_names(spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates a NamedTuple of full variable names for a given component.

# Rationale for Update
This function is updated to correctly handle the distinction between hyperparameters
(which can be shared across outcomes in a multivariate model) and latent fields
(which are always unique to each outcome in a multivariate model). The previous
implementation used a single suffixing rule, which failed for shared-hyperparameter
models.

This version introduces two separate suffixing rules:
1.  `hyperparam_suffix`: Appended to hyperparameters (`sigma`, `rho`, `ls`, etc.) only
    when the model is multivariate AND the component is not shared (`!is_shared`).
2.  `latent_field_suffix`: Appended to latent fields (`innovations`, `latent`, etc.)
    whenever the model is multivariate, regardless of the `shared` flag.

This change ensures that variable names are generated correctly for all model
architectures, resolving a key source of `FieldError` during model construction.
The list of parameters has also been updated to use `innovations` consistently,
deprecating `raw` and `innov`, and to include other specialized parameter names
found across the component library.

# Version
v1.1.0 (2026-08-11)

# Arguments
- `spec`: The component's specification.
- `arch`: The model architecture (`"univariate"` or `"multivariate"`).
- `outcome_idx`: The index of the outcome for multivariate models.
- `prefix`: An optional prefix for nested models.

# Returns
- A `NamedTuple` containing all necessary variable names as Symbols.
"""
function generate_full_variable_names(spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    base_key = string(spec.key)
    full_key = isempty(prefix) ? base_key : "$(prefix)_$(base_key)"

    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    # Suffix for hyperparameters, which can be shared across outcomes.
    # It is only added if the model is multivariate AND the component is not shared.
    hyperparam_suffix = (is_multivariate && !is_shared) ? "_$(outcome_idx)" : ""
    
    # Suffix for latent fields, which are always per-outcome in a multivariate model.
    latent_field_suffix = is_multivariate ? "_$(outcome_idx)" : ""

    names = Dict{Symbol, Symbol}()
    
    # --- Hyperparameters ---
    # These parameters may be shared across outcomes in a multivariate model.
    hyperparameters = [
        :sigma, :rho, :rho1, :rho2, :unconstrained_rho, :kappa, :ls, :range, :period,
        :amplitude, :phase, :velocity, :diffusion, :pca_sd, :pdef_sd, :L_corr,
        :sigma_effects, :r, :K, :q, :M_nat, :alpha, :beta, :gamma, :delta, :curvature
    ]
    for p in hyperparameters
        names[p] = Symbol("$(p)_$(full_key)$(hyperparam_suffix)")
    end

    # --- Latent Fields & Innovations ---
    # These are always unique per outcome in a multivariate model.
    latent_fields = [
        :innovations, :latent, :struct, :iid, :beta_cos, :beta_sin, :rho_field,
        :W, :b, :v_raw, :factors_flat, :thresh_raw, :W1, :b1, :W2, :amplitude_raw,
        :innov_predator, :cluster_innovations, :inducing_innovations, :diag_innovations
    ]
    for p in latent_fields
        names[p] = Symbol("$(p)_$(full_key)$(latent_field_suffix)")
    end

    # Deprecated names, kept for temporary backward compatibility.
    # They point to the new 'innovations' name to ensure old code does not break.
    names[:raw] = names[:innovations]
    names[:innov] = names[:innovations]

    return NamedTuple(names)
end



"""
    _generate_st_interaction_block(M::NamedTuple, s_spec, t_spec, is_multivariate::Bool, eta_name::String)

Generates Turing code for a spatiotemporal interaction effect. 

# Version
v1.1.0 (2026-08-18)
 

# Arguments
- `M`: The main model configuration `NamedTuple`.
- `s_spec`, `t_spec`: The specifications for the spatial and temporal components.
- `is_multivariate`: A boolean indicating if the model is multivariate.
- `eta_name`: The name of the linear predictor variable.

# Returns
- A `String` containing the generated Turing code for the interaction block.
"""
function _generate_st_interaction_block(M::NamedTuple, s_spec, t_spec, is_multivariate::Bool, eta_name::String)
    if get(M, :model_st, "none") == "none" 
        return ""
    end

    if isnothing(s_spec) || isnothing(t_spec)
        @warn "Spatiotemporal interaction requested but marginal specifications are missing."
        return ""
    end

    s_key = string(s_spec.key)
    t_key = string(t_spec.key)
    
    s_chol_access = get(s_spec, :is_static, false) ? "spec_registry[:$(s_key)].cholesky_factor" : "cholesky(Symmetric(spec_registry[:$(s_key)].hyper.Q_template + noise * I))"
    t_chol_access = get(t_spec, :is_static, false) ? "spec_registry[:$(t_key)].cholesky_factor" : "cholesky(Symmetric(spec_registry[:$(t_key)].hyper.Q_template + noise * I))"

    K = get(M, :outcomes_N, 1)

    if is_multivariate
        interaction_code = """
    # --- Spatiotemporal Interaction Priors ---
    local st_sigma_prior_dist_str = haskey(M, :st_interaction_sigma_prior) ? _distribution_to_string(M.st_interaction_sigma_prior) : "Exponential(1.0)"
    st_interaction_sigma ~ NamedDist(filldist($(st_sigma_prior_dist_str), $K), :st_interaction_sigma)
    
    # --- Spatiotemporal Interaction Innovations ---
    st_interaction_raw ~ NamedDist(MvNormal(fill!(Array{T}(undef, M.s_N * M.t_N * $K), 0), I), :st_interaction_raw)

    let
        C_s = $s_chol_access
        C_t = $t_chol_access
        
        Z_tensor = reshape(st_interaction_raw, M.s_N, M.t_N, $K)
        
        for k in 1:$K
            Z_k = view(Z_tensor, :, :, k)
            
            tmp_spatial = C_s.U \\ Z_k
            st_field_k_unscaled = (transpose(C_t.U \\ transpose(tmp_spatial)))
            
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * (M.s_N * M.t_N)), sum(st_field_k_unscaled))
            
            st_field_k = st_field_k_unscaled .* st_interaction_sigma[k]

            # Vectorized update to the linear predictor
            effect_k = vec(st_field_k)[M.st_idx]
            $(eta_name)[:, k] .+= effect_k
        end
    end
    """
    else
        interaction_code = """
    # --- Spatiotemporal Interaction Priors ---
    local st_sigma_prior_dist_str = haskey(M, :st_interaction_sigma_prior) ? _distribution_to_string(M.st_interaction_sigma_prior) : "Exponential(1.0)"
    st_interaction_sigma ~ NamedDist($(st_sigma_prior_dist_str), :st_interaction_sigma)

    st_interaction_raw ~ NamedDist(MvNormal(fill!(Array{T}(undef, M.s_N * M.t_N), 0), I), :st_interaction_raw)

    let
        C_s = $s_chol_access
        C_t = $t_chol_access
        
        Z_matrix = reshape(st_interaction_raw, M.s_N, M.t_N)
        
        tmp_spatial = C_s.U \\ Z_matrix
        st_field_unscaled = (transpose(C_t.U \\ transpose(tmp_spatial)))
        
        Turing.@addlogprob! logpdf(Normal(0, 0.001 * (M.s_N * M.t_N)), sum(st_field_unscaled))
        
        st_field = st_field_unscaled .* st_interaction_sigma

        # Vectorized update to the linear predictor
        effect = vec(st_field)[M.st_idx]
        $(eta_name) .+= effect
    end
    """
    end
    
    return interaction_code
end
 

 
"""
    _generate_householder_reflection_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)

Generates Turing code for the Householder reflection (spectral orientation) feature.
This allows for rotating the latent space in multivariate models to better align signals,
which can be useful for processes with directional dependencies. This is controlled by
the `spectral_orientation=true` keyword argument.
"""
function _generate_householder_reflection_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    if !is_multivariate || !get(M, :spectral_orientation, false)
        return "", ""
    end

    K = M[:outcomes_N]
     
    priors_str = """
    # Householder reflection for spectral orientation
    v_raw_reflection ~ NamedDist(MvNormal(fill!(Array{T}(undef, $(K)), 0), I), :v_raw_reflection)
    """

    update_str = """
    let
        v_reflection = v_raw_reflection / (norm(v_raw_reflection) + 1e-9)
        H_reflection = I - 2.0 * v_reflection * v_reflection'
        $(eta_name) = $(eta_name) * H_reflection
    end
    """
    return priors_str, update_str
end




"""
    _generate_nested_model_block(M::NamedTuple, is_multivariate::Bool, main_eta_name::String)

Generates the code block for nested sub-models.

# Version
v1.0.0 (2026-08-17)

# Rationale
This function restores the functionality for handling nested models, which was lost
during a previous refactoring. It generates the necessary Turing code for each
sub-model defined in `M.nested_components`.

The implementation follows a similar pattern to the main `bstm_text_assembler`. For
each nested component, it:
1.  Defines a prior for the linking parameter `rho_nested`, which scales the sub-model's effect.
2.  Creates a `let` block to scope the sub-model's variables.
3.  Generates the priors for the sub-model's intercept, fixed effects, and all of its components. All variable names are prefixed with the nested component's key to avoid name collisions.
4.  Assembles the sub-model's linear predictor (`eta_sub`).
5.  Adds the sub-model's likelihood evaluation to the main model's likelihood block.
6.  Links the assembled sub-model effect to the main model's linear predictor.

This implementation assumes that the sub-model's coordinates align with the main
model's coordinates, as interpolation logic is not yet part of the model generation step.
"""
function _generate_nested_model_block(M::NamedTuple, is_multivariate::Bool, main_eta_name::String)
    if haskey(M, :nested_components) && !isempty(M.nested_components)
        priors_acc = String[]
        updates_acc = String[]
        likelihood_acc = String[]

        for (key, sub_M) in M.nested_components
            prefix = string(key)
            
            # 1. Define the linking parameter rho
            rho_name = "rho_nested_$(key)"
            push!(priors_acc, "$(rho_name) ~ Normal(1.0, 0.5)")

            # --- Start generating code for the sub-model ---
            sub_priors_acc = String[]
            sub_updates_acc = String[]
            
            sub_arch = get(sub_M, :model_arch, "univariate")
            is_sub_multivariate = sub_arch == "multivariate"
            sub_eta_name = is_sub_multivariate ? "eta_latent_sub_$(key)" : "eta_sub_$(key)"
            
            # --- Generate Priors for sub-model ---
            # Intercept
            if get(sub_M, :add_intercept, false)
                intercept_var_name = "intercept_$(prefix)"
                dist_str = is_sub_multivariate ? "filldist($(_distribution_to_string(sub_M.intercept_prior)), $(sub_M.outcomes_N))" : _distribution_to_string(sub_M.intercept_prior)
                push!(sub_priors_acc, "$(intercept_var_name) ~ DynamicPPL.NamedDist($(dist_str), :$(intercept_var_name))")
            end
            
            # Fixed Effects
            if get(sub_M, :Xfixed_N, 0) > 0
                beta_name = is_sub_multivariate ? "Xfixed_beta_prop_flat_$(prefix)" : "Xfixed_beta_prop_$(prefix)"
                n_params = is_sub_multivariate ? sub_M.Xfixed_N * sub_M.outcomes_N : sub_M.Xfixed_N
                prior_str = _distribution_to_string(sub_M.Xfixed_priors_vec[1])
                push!(sub_priors_acc, "$(beta_name) ~ DynamicPPL.NamedDist(filldist($(prior_str), $(n_params)), :$(beta_name))")
            end

            # Components
            for (i, sub_spec) in enumerate(sub_M.components)
                prefixed_sub_spec = merge(sub_spec, (key=Symbol(prefix, "_", sub_spec.key),))
                for k in 1:sub_M.outcomes_N
                    sub_outcome_idx = is_sub_multivariate ? k : nothing
                    priors_str = get_priors(sub_spec.component_obj, prefixed_sub_spec, sub_arch, sub_outcome_idx, sub_M)
                    push!(sub_priors_acc, replace(priors_str, "spec_registry[:$(prefixed_sub_spec.key)]" => "sub_M.components[$(i)]"))
                end
            end
            
            # --- Assemble sub-model updates ---
            sub_eta_init = if get(sub_M, :add_intercept, false)
                is_sub_multivariate ? "intercept_$(prefix)' .+ fill!(Array{T}(undef, sub_M.y_N, sub_M.outcomes_N), 0)" : "intercept_$(prefix) .+ fill!(Array{T}(undef, sub_M.y_N), 0)"
            else
                is_sub_multivariate ? "fill!(Array{T}(undef, sub_M.y_N, sub_M.outcomes_N), 0)" : "fill!(Array{T}(undef, sub_M.y_N), 0)"
            end
            push!(sub_updates_acc, "local $(sub_eta_name) = $(sub_eta_init)")

            # Add component updates
            for (i, sub_spec) in enumerate(sub_M.components)
                prefixed_sub_spec = merge(sub_spec, (key=Symbol(prefix, "_", sub_spec.key),))
                for k in 1:sub_M.outcomes_N
                    sub_outcome_idx = is_sub_multivariate ? k : nothing
                    updates_str = get_updates(sub_spec.component_obj, prefixed_sub_spec, sub_arch, sub_outcome_idx, sub_M)
                    updates_str_final = replace(updates_str, r"eta_latent|eta" => sub_eta_name, "spec_registry[:$(prefixed_sub_spec.key)]" => "sub_M.components[$(i)]")
                    push!(sub_updates_acc, updates_str_final)
                end
            end

            # --- Sub-model Likelihood ---
            sub_lik_code = _generate_final_likelihood_block(sub_M, is_sub_multivariate)
            sub_lik_code_final = replace(sub_lik_code, r"eta_latent|eta" => sub_eta_name, r"\bM\." => "sub_M.")
            
            # --- Assemble final code blocks ---
            push!(priors_acc, join(sub_priors_acc, "\n"))
            let_block = "let\n    local sub_M = M.nested_components[:$(key)]\n    $(join(sub_updates_acc, "\n    "))\n    $(main_eta_name) .+= $(rho_name) .* $(sub_eta_name)\nend"
            push!(updates_acc, let_block)
            push!(likelihood_acc, sub_lik_code_final)
        end

        return join(priors_acc, "\n\n"), join(updates_acc, "\n\n"), join(likelihood_acc, "\n\n")
    end
    return "", "", ""
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


"""
    _canonical_term_string(term::StatsModels.AbstractTerm)

Creates a canonical string representation for a `StatsModels.AbstractTerm`. This is
used internally to map priors to the correct fixed-effect coefficients, especially
for interaction terms where the order of variables does not matter.

# Arguments
- `term::StatsModels.AbstractTerm`: A term from a `StatsModels.FormulaTerm`.

# Returns
- `String`: A standardized string representation of the term.
"""
function _canonical_term_string(term::StatsModels.AbstractTerm)
    if term isa StatsModels.InteractionTerm
        # Sort term names for canonical representation, e.g., "a&b" is the same as "b&a".
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


"""
    _precompute_static_components!(M::Dict)

Pre-computes matrix factorizations for static model components. This version is
CPU-only.
"""
function _precompute_static_components!(M::Dict)
    noise = M[:noise]
    new_components = []
    static_component_types = [IID, ICAR, Besag, RW1, RW2, Cyclic, PSpline, TPS, BSpline, Eigen, Moran, Barycentric, TensorProductSmooth]
 
    for spec_in in M[:components]
        current_spec = spec_in
        m_obj = current_spec.component_obj
 
        if m_obj isa Mixed
            inner_model = m_obj.model
            is_inner_static = any(T -> inner_model isa T, static_component_types)
            if is_inner_static && hasproperty(current_spec.hyper, :Q_template) && !isnothing(current_spec.hyper.Q_template) && size(current_spec.hyper.Q_template, 1) > 0
                try
                    Q_concrete = current_spec.hyper.Q_template
                    Q_dense = Matrix(Q_concrete)
                    F = cholesky(Symmetric(Q_dense + noise * I))
                    final_spec = merge(current_spec, (is_static=true, cholesky_factor=F))
                    push!(new_components, final_spec)
                    continue
                catch e
                    @warn "Cholesky factorization failed for static inner model in $(current_spec.key). Reverting to dynamic computation. Error: $e"
                end
            end
        end

        is_main_static = !(current_spec.component_obj isa Composed) && any(T -> current_spec.component_obj isa T, static_component_types)

        if is_main_static && hasproperty(current_spec.hyper, :Q_template) && !isnothing(current_spec.hyper.Q_template) && size(current_spec.hyper.Q_template, 1) > 0
            try
                Q_concrete = current_spec.hyper.Q_template
                Q_dense = Matrix(Q_concrete)
                F = cholesky(Symmetric(Q_dense + noise * I))
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


 
"""
    _initialize_config(data::DataFrame, kwargs)

Creates the initial model configuration dictionary (`M`) 

# Version
v1.2.0 (2026-08-16)

# Rationale
This function is refactored to handle only the initial setup. 

# Arguments
- `data::DataFrame`: The input data for the model.
- `kwargs`: A dictionary of keyword arguments passed from the main `@bstm` call.

# Returns
- `Dict{Symbol, Any}`: The initial model configuration dictionary `M`.
""" 
function _initialize_config(data::DataFrame, kwargs)
    M = Dict{Symbol, Any}()
    M[:data] = data
    M[:y_N] = size(data, 1)
    
    # Set defaults that can be overridden by user-provided kwargs.
    M[:noise] = 1e-6
    M[:hyperpriors] = Dict{Symbol, Any}()
    M[:prior_scheme] = :pcpriors
    M[:fixed_effects_priors] = Dict{Symbol, Any}()
    M[:spectral_orientation] = true

    # Merge user-provided keyword arguments, overriding defaults.
    for (k, v) in kwargs; M[k] = v; end
    
    # Initialize containers for components and basis matrices.
    M[:calling_module] = get(kwargs, :calling_module, Main)
    M[:components] = []
    M[:basis_matrices] = Dict{Symbol, Any}()
    
    return M
end
 
 

"""
    _process_lhs!(M::Dict, outcome_specs::Vector{Dict{Symbol, Any}})

Processes the Left-Hand Side (LHS) of the formula, setting up outcomes, likelihoods,
and observation-level parameters in the main model configuration.

# Version
v1.1.0 (2026-08-13)

# Rationale
This function is a core part of the model configuration pipeline. It is responsible
for interpreting the `likelihood()` module(s) on the LHS of the formula. It correctly
handles multiple model architectures:
1.  **Univariate**: A single outcome variable.
2.  **Multivariate**: Multiple outcome variables specified with `+`, which are modeled
    jointly with a shared correlation structure.
3.  **Compositional**: Multiple variables specified with `+` under a
    `dirichlet_multinomial` family are treated as categories of a single response.
4.  **Ordinal**: A single outcome variable with the `ordinal` family is recoded, and
    the necessary parameters for the cut-points are configured.

The function also resolves all observation-level parameters (e.g., `offsets`,
`weights`, `trials`, censoring bounds) by calling specialized helper functions.
This version updates the call to `_resolve_outcome_scalar_param` to be consistent
with the refactored, non-mutating function name.

# Arguments
- `M::Dict`: The main model configuration dictionary, which is mutated by this function.
- `outcome_specs::Vector{Dict{Symbol, Any}}`: A vector of parsed outcome specifications
  from the formula parser.

# Returns
- `nothing`.
"""
function _process_lhs!(M::Dict, outcome_specs::Vector{Dict{Symbol, Any}})
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
        
        # Merge parameters from all likelihood specs, giving precedence to the first ones.
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
                @warn "Likelihood `family` not specified for outcome '$(outcomes[i])'. Defaulting to `family=gaussian`."
            end
            
            if string(get(spec, :family, "")) == "ordinal"
                if M[:outcomes_N] > 1; error("The `ordinal` family is currently only supported for univariate models."); end
                outcome_var = M[:outcomes][i]
                if !hasproperty(M[:data], outcome_var); error("Ordinal outcome variable ':$outcome_var' not found in data."); end
                
                outcome_data = M[:data][!, outcome_var]
                if !(eltype(outcome_data) <: Integer)
                    @warn "Ordinal outcome variable ':$outcome_var' is not of integer type. Attempting to convert."
                    try; M[:data][!, outcome_var] = round.(Int, outcome_data); catch; error("Could not convert ordinal outcome variable ':$outcome_var' to integers."); end
                end
                
                unique_levels = sort(unique(M[:data][!, outcome_var]))
                K = length(unique_levels)
                if K < 2; error("Ordinal outcome variable ':$outcome_var' must have at least 2 unique levels."); end
                
                spec[:latent_dist] = get(spec, :latent_dist, :logistic)
                spec[:K] = K
                
                level_map = Dict(level => i for (i, level) in enumerate(unique_levels))
                M[:data][!, outcome_var] = [level_map[val] for val in M[:data][!, outcome_var]]
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
    
    # Merge all likelihood parameters to resolve global settings like offsets, weights, etc.
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
            # Corrected call to non-mutating version
            val = _resolve_outcome_scalar_param(spec_params, key, calling_mod)
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



"""
    _resolve_outcome_scalar_param(params::Dict, key::Symbol, calling_mod::Module)

Resolves a likelihood parameter that must be a scalar value for a given outcome.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function is a core utility for the model configuration engine. It handles
scalar parameters in the `likelihood()` module, such as `censor_lower` or `hurdle`.
It is designed to be robust, correctly parsing values that are provided as direct
numeric literals, as symbols pointing to a variable in the user's scope, or as
expressions that evaluate to a number. The `!` has been removed from the function
name as it does not mutate its arguments, adhering to Julia's style conventions.

# Arguments
- `params::Dict`: The dictionary of parameters from the parsed `likelihood()` module.
- `key::Symbol`: The symbol for the parameter to resolve (e.g., `:censor_lower`).
- `calling_mod::Module`: The module context for evaluating symbols or expressions.

# Returns
- The resolved scalar `Number`, or `nothing` if the parameter is not found or invalid.
"""
function _resolve_outcome_scalar_param(params::Dict, key::Symbol, calling_mod::Module)
    if !haskey(params, key)
        return nothing
    end

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



"""
    _resolve_obs_param!(opt_dict, params, data, param_keys, target_key)

Resolves an observation-level parameter (e.g., offsets, weights) from the
likelihood parameters and sets it in the main configuration dictionary.

# Version
v1.0.2 (2026-08-13)

# Rationale
This function is a core utility for the model configuration engine. It robustly
parses observation-level parameters from the `likelihood()` module, such as
`log_offsets`, `weights`, and `trials`. It is designed to handle multiple input
styles from the user:
1.  A `Symbol` referring directly to a column in the data frame (e.g., `log_offsets=my_col`).
2.  A `Symbol` referring to a variable in the user's scope, which itself contains
    a column name or a data vector (e.g., `my_var = :my_col; ... log_offsets=my_var`).
3.  A literal `Number` or `AbstractVector`.

This flexibility, particularly the use of `Core.eval` to resolve variables in the
user's scope, is critical for the macro's usability in both interactive and
programmatic contexts.

# Arguments
- `opt_dict`: The main model configuration dictionary (`M`), which is mutated.
- `params`: The dictionary of parameters from the parsed `likelihood()` module.
- `data`: The input `DataFrame`.
- `param_keys`: A list of possible keys for the parameter (e.g., `[:log_offsets, :offsets]`).
- `target_key`: The key to set in `opt_dict` (e.g., `:log_offsets`).

# Returns
- `nothing`.
"""
function _resolve_obs_param!(opt_dict, params, data, param_keys, target_key)
    for key in param_keys
        if haskey(params, key)
            val = params[key]
            if val isa Symbol
                if hasproperty(data, val)
                    # Case 1: The value is a symbol that directly matches a column name.
                    opt_dict[target_key] = data[!, val]
                    opt_dict[Symbol("user_provided_", target_key)] = true
                else
                    # Case 2: The symbol might refer to a variable in the calling scope.
                    calling_mod = get(opt_dict, :calling_module, Main)
                    try
                        evaluated_val = Core.eval(calling_mod, val)
                        if evaluated_val isa Symbol && hasproperty(data, evaluated_val)
                            # e.g., my_var = :my_col; log_offsets = my_var
                            opt_dict[target_key] = data[!, evaluated_val]
                            opt_dict[Symbol("user_provided_", target_key)] = true
                        elseif evaluated_val isa String && hasproperty(data, Symbol(evaluated_val))
                            # e.g., my_var = "my_col"; log_offsets = my_var
                            opt_dict[target_key] = data[!, Symbol(evaluated_val)]
                            opt_dict[Symbol("user_provided_", target_key)] = true
                        elseif evaluated_val isa Number || evaluated_val isa AbstractVector
                            # e.g., my_vec = [0.1, ...]; log_offsets = my_vec
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
                # Case 3: The value is a literal number or vector.
                opt_dict[target_key] = val
                opt_dict[Symbol("user_provided_", target_key)] = true
            else
                @warn "Observation parameter '$val' for '$target_key' is not a valid column name, vector, or scalar. Ignoring."
            end
            return # Stop after finding the first matching key.
        end
    end
end


"""
    _resolve_boolean_obs_param!(opt_dict, params, param_key, target_key)

Resolves a boolean flag from the likelihood parameters and sets it in the main
configuration dictionary.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function is a core utility for the model configuration engine. It handles
boolean flags in the `likelihood()` module, such as `zero_inflated=true` or
`volatility=true`. It is designed to be robust, correctly parsing values that are
provided as direct booleans (`true`), as symbols pointing to a boolean variable
in the user's scope (`my_flag`), or as expressions that evaluate to a boolean.
This ensures a flexible and predictable user experience.

# Arguments
- `opt_dict`: The main model configuration dictionary (`M`), which is mutated.
- `params`: The dictionary of parameters from the parsed `likelihood()` module.
- `param_key`: The key for the boolean flag to look for in `params`.
- `target_key`: The key to set in `opt_dict`.

# Returns
- `nothing`.
"""
function _resolve_boolean_obs_param!(opt_dict, params, param_key, target_key) 
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



"""
    _process_fixed_effects_priors!(M::Dict)

Resolves and stores the prior distributions for each fixed effect coefficient.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function is a core part of the model configuration pipeline. It is responsible
for correctly mapping the priors specified in `fixed()` modules within the formula
to the final coefficients generated by `StatsModels.jl`. This is a non-trivial
task because `StatsModels.jl` can expand a single term (e.g., a categorical
variable) into multiple coefficients.

This function correctly handles:
1.  **Canonical Naming**: It normalizes interaction terms (e.g., `a&b` is treated
    the same as `b&a`) to ensure priors are applied consistently.
2.  **Scoped Evaluation**: It uses the `calling_module` context to correctly
    evaluate `Expr` objects for priors (e.g., `prior=Normal(mu, sd)`), where `mu`
    and `sd` are variables in the user's scope.
3.  **PC Priors**: It correctly dispatches to `create_pc_prior` when a tuple
    constraint is provided.
4.  **Defaults**: It assigns a default prior to any coefficient that does not have
    an explicit prior.

# Arguments
- `M::Dict`: The main model configuration dictionary, which is mutated by this function.

# Returns
- `nothing`.
"""
function _process_fixed_effects_priors!(M::Dict) 
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
            prior_obj = if prior_val isa Expr
                try
                    Core.eval(calling_mod, prior_val)
                catch e
                    error("Could not evaluate `prior` argument `$(prior_val)` for fixed effect '$canonical_name'. Error: $e")
                end
            else
                prior_val
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

    # Assign default priors to any coefficients that were not explicitly assigned one.
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


"""
    _finalize_config!(M::Dict)

Ensures the model configuration dictionary has all necessary keys with default values
before being passed to the code generator.

# Version
v1.1.0 (2026-08-13)

# Rationale
This function is the final step in the configuration pipeline. It prevents `KeyError`
exceptions in the model assembler by providing safe defaults for any configuration
keys that were not set during the formula parsing or component processing stages.

  consistent with the refactored architecture. It no
longer sets defaults for observation-level parameters like `log_offsets`, `weights`,
or `trials`, as these are now robustly handled by the `_precompute_likelihood_params!`
function earlier in the pipeline. Its primary role is to provide fallbacks for
structural indices (`s_idx`, `t_idx`, `u_idx`) and global settings like the prior scheme.

# Arguments
- `M::Dict`: The model configuration dictionary, which is mutated by this function.

# Returns
- `nothing`.
"""
function _finalize_config!(M::Dict)
    # This function provides safe defaults for keys that may not have been set
    # during the main configuration process.
    defaults = Dict(
        :s_N => 0, 
        :t_N => 0, 
        :u_N => 0,
        :s_idx => ones(Int, M[:y_N]),
        :t_idx => ones(Int, M[:y_N]),
        :u_idx => ones(Int, M[:y_N]),
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

 
"""
    _replace_bstm_modules_in_expr(ex)

Recursively traverses a Julia expression and replaces `bstm`-specific modules
with their `StatsModels.jl` equivalents for parsing within other modules like `mixed()`.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function is a core utility for the formula parsing engine, specifically for
handling the `mixed()` module. The `mixed()` module's formula (e.g., `1 + x | g`)
is parsed internally by `StatsModels.jl`. However, users can write `bstm`-specific
syntax like `intercept()` or `fixed(x)` within the `mixed()` call. This function
acts as a pre-processor, traversing the expression tree and replacing these `bstm`
modules with their `StatsModels.jl` equivalents (`1` and `x`, respectively) before
the expression is passed to `StatsModels.jl` for parsing. This ensures compatibility
and a consistent user experience.

# Arguments
- `ex`: A Julia expression, symbol, or literal.

# Returns
- The modified expression, ready for parsing by `StatsModels.jl`.
"""
function _replace_bstm_modules_in_expr(ex)
    if ex isa Expr && ex.head == :call
        if ex.args[1] == :intercept
            return 1
        elseif ex.args[1] == :fixed && length(ex.args) > 1
            # Return the variable inside fixed(), e.g., fixed(x) -> x
            return ex.args[2]
        end
        # Recursively process the arguments of any other function call.
        return Expr(ex.head, _replace_bstm_modules_in_expr.(ex.args)...)
    elseif ex isa Expr
        # Recursively process the arguments of any other expression type.
        return Expr(ex.head, _replace_bstm_modules_in_expr.(ex.args)...)
    else
        # Base case: return non-expression atoms (symbols, literals) as is.
        return ex
    end
end


  
"""
    _bstm_error_handler(e, model)

Provides detailed, user-friendly diagnostics when the initial prior predictive check
(`rand(model)`) fails.

# Version
v1.1.0 (2026-08-13)

# Rationale
This function is a critical part of the user experience. Instead of presenting a
raw Julia error, it intercepts common exception types (`DimensionMismatch`,
`PosDefException`, `MethodError`, etc.) and provides specific, actionable advice
tailored to the `bstm` framework. This helps users quickly identify issues related
to data structure, model specification, or AD compatibility. This version adds
handlers for `MethodError`, `ArgumentError`, and `UndefVarError` and refines the
existing diagnostics for clarity.

# Arguments
- `e`: The exception object caught during the prior predictive check.
- `model`: The instantiated Turing model object.

# Returns
- `nothing`. The function prints a diagnostic report to the console.
"""
function _bstm_error_handler(e, model)
    println("\nERROR during prior predictive check (rand(m)):")
    showerror(stdout, e, stacktrace(catch_backtrace()))
    println("\n\n--- bstm Diagnosis ---")

    if e isa DimensionMismatch
        println("A `DimensionMismatch` error occurred. This often points to an issue in the model's structure.")
        println("Potential Causes:")
        println("  1. Latent Field vs. Index Mismatch: The number of latent variables (e.g., `s_N`) does not match the number of unique levels in the corresponding index variable (e.g., `s_idx`).")
        println("  2. Matrix Multiplication: An operation like `X * beta` has incompatible dimensions.")
    elseif e isa BoundsError
        println("A `BoundsError` occurred. This means an index is out of range for an array.")
        println("Potential Causes:")
        println("  - This is common with `mixed()` or `spatial()` effects. The latent field vector is smaller than the maximum index in the corresponding index vector (e.g., `M.s_idx`).")
        println("  - Check for off-by-one errors or miscalculated numbers of levels (`n_cat`, `s_N`).")
    elseif e isa PosDefException
        println("A `PosDefException` occurred. A matrix that must be positive definite is not.")
        println("Potential Causes:")
        println("  1. GMRF Precision Matrix: For `icar` or `besag` models, ensure your adjacency matrix `W` corresponds to a single connected graph. Disconnected spatial 'islands' will cause this error.")
        println("  2. GP Covariance Matrix: A kernel matrix `K` is not positive definite, often due to very close data points. Try increasing the `noise` keyword argument (e.g., `noise=1e-5`).")
    elseif e isa KeyError
        println("A `KeyError` occurred. The model tried to access a parameter in the configuration that does not exist.")
        println("Potential Causes:")
        println("  - A typo in a variable name within the formula string.")
        println("  - A required parameter (e.g., `W` for a spatial model) was not passed as a keyword argument.")
    elseif e isa MethodError
        println("A `MethodError` occurred. This is often due to a type instability, especially with Automatic Differentiation (AD).")
        println("Potential Causes:")
        println("  - A custom function or prior distribution is not compatible with Turing's AD backend (e.g., ForwardDiff.jl). Ensure custom code handles `Dual` number types.")
        println("  - A parameter was expected to be one type (e.g., a Vector) but was another (e.g., a scalar).")
    elseif e isa ArgumentError
        println("An `ArgumentError` occurred. A function or distribution was called with an invalid argument (e.g., a negative standard deviation).")
        println("Potential Causes:")
        println("  - A prior distribution is misspecified (e.g., `Normal(0, -1)`). Check priors in your formula.")
    elseif e isa UndefVarError
        println("An `UndefVarError` occurred: `$(e.var)` is not defined.")
        println("Potential Causes:")
        println("  - A variable in the formula (e.g., `fixed(my_var)`) is not a column in the DataFrame.")
        println("  - A variable for a keyword argument (e.g., `W=my_matrix`) is not defined in the script's scope.")
    else
        println("An unexpected error occurred. General debugging tips:")
        println("  - Review the generated model code printed above the error.")
        println("  - Use `show_model(m)` to inspect the full model configuration.")
        println("  - Simplify your model formula by removing components one by one to isolate the error.")
    end
    println("------------------------")

    # Suggest simplified formulas to help the user debug.
    println("\n--- Suggested Debugging Steps ---")
    try
        formula_str = model.args.M.formula
        lhs, rhs_raw = split(formula_str, '~')
        lhs = Base.strip(lhs)

        rhs_normalized = replace(Base.strip(rhs_raw), r"\s*-\s*" => " + -")
        all_terms = split_terms_at_depth(rhs_normalized, " + ")

        has_intercept = !any(in.(Base.strip.(all_terms), (["0", "-1"],))) && !any(startswith.(Base.strip.(all_terms), "intercept(false"))

        base_rhs = has_intercept ? "1" : "0"
        println("1. Start with the simplest possible model to isolate the issue.")
        println("   This helps determine if the error is in your `likelihood()` definition or in the model components.")
        println("\n   Suggested base model:")
        println("   @bstm(\n       $lhs ~ $base_rhs,\n       data, ...\n   )")

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
        println("\nError during suggestion generation: $e_sugg")
    end
    println("---------------------------------\n")
end


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

    # Construct the core function call. 
    core_logic = :(bstm_core($formula_str, $data_esc, $(__module__); $(kwargs_esc...)))

    # Return the appropriate expression
    if !isnothing(var_name)
        return :($(esc(var_name)) = $core_logic)
    else
        return core_logic
    end
end




"""
    _print_param(name, value, status; indent=4)

A helper function to print a single parameter with its value and status in a
standardized format. It ensures consistent indentation and safely truncates long
string representations to maintain readability.

# Arguments
- `name`: The name of the parameter.
- `value`: The value of the parameter.
- `status`: A `Symbol` indicating if the value was `:user` provided or a `:default`.
- `indent`: The indentation level for printing.
"""
function _print_param(name, value, status; indent=4)
    indent_str = " " ^ indent
    status_str = status == :user ? "(User-provided)" : "(Default)"
    
    value_str = string(value)
    # Safely truncate long values to prevent cluttering the console.
    if length(value_str) > 70
        value_str = first(value_str, 67) * "..."
    end
    
    println("$indent_str- $(rpad(name, 20)): $(value_str)  $status_str")
end

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
        _print_param(:prior, get(config, :intercept_prior, Normal(0,5)), is_user_provided ? :user : :default; indent=2)
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
            _print_param(name, prior_obj, is_default ? :default : :user; indent=4)
        end
    end

    # 4. Model Components (random, smooth, etc.)
    if !isempty(config.components)
        println("\n[ Model Components ]")
        for spec in config.components
            println("  --- Component: $(spec.key) ---")
            component_obj = spec.component_obj
            model_type_sym = Symbol(lowercase(string(typeof(component_obj))))
            println("    - Type:      $(typeof(component_obj))")
            println("    - Structure: $(spec.structure)")
            println("    - Variable:  $(spec.var)")
            
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
                if param_name in [:positional_args, :structure, :in_dims]; continue; end
                
                local final_val, status
                if param_name in fieldnames(typeof(component_obj))
                    # It's a hyperparameter (prior)
                    final_val = getfield(component_obj, param_name)
                    status = haskey(user_provided_params_raw, param_name) ? :user : :default
                else
                    # It's a configuration argument
                    default_val = get(COMPONENT_CONFIG_ARGS, model_type_sym, Dict()) |> d -> get(d, param_name, "N/A")
                    final_val = get(user_provided_params_raw, param_name, default_val)
                    status = haskey(user_provided_params_raw, param_name) ? :user : :default
                end
                
                if final_val isa Vector{<:UnivariateDistribution}
                    println("      - $(rpad(param_name, 20)): [")
                    for (idx, p_dist) in enumerate(final_val)
                        println("        $idx: $p_dist")
                    end
                    println("      ] $(status == :user ? "(User-provided)" : "(Default)")")
                else
                    _print_param(param_name, final_val, status; indent=6)
                end
            end
        end
    end
    println("\n-------------------------------------\n")
end

"""
    bstm_core(formula::String, data::DataFrame, calling_module::Module; kwargs...)

The main entry point for the `@bstm` macro. This function orchestrates the configuration,
code generation, and instantiation of a Turing model.

# Version
v1.1.3 (2026-08-17)

# Rationale
This function is the core of the `bstm` framework. It provides a robust and complete
workflow for dynamic model creation. This version resolves a "world age" warning by
changing how the dynamically generated model function is retrieved. Instead of using
`Core.eval` to define the function and then `getfield` to retrieve it (which can
trigger the warning), this version uses a `quote` block within `Core.eval` to define
and then immediately return the function object. This is a more robust pattern for
runtime code generation in Julia and eliminates the world-age warning.

# Workflow
1.  **Configuration**: Calls `bstm_config` to translate the user's formula and
    data into a detailed model specification.
2.  **Code Generation**: Dynamically generates a unique Turing `@model` definition
    by calling `bstm_text_assembler`.
3.  **Scoped Evaluation**: Evaluates the generated model in the current module's
    scope and directly captures the returned function object.
4.  **Safe Instantiation**: Uses `Base.invokelatest` to instantiate the model.
5.  **Validation**: Automatically runs a prior predictive check (`rand(model)`) and
    provides detailed, user-friendly diagnostics if the check fails.
 
# Arguments
- `formula::String`: The model formula.
- `data::DataFrame`: The input data.
- `calling_module::Module`: The module in which the model was called (used for context).
- `kwargs...`: Additional keyword arguments passed to `bstm_config`.

# Returns
- An instantiated and validated Turing model object.
"""
function bstm_core(formula::String, data::DataFrame, calling_module::Module; kwargs...)
    # --- 1. Configuration ---
    # Generate model configuration dictionary based on formula syntax and data schema.
    options = bstm_config(formula, data; calling_module=calling_module, kwargs...) # Pass kwargs to bstm_config

    # --- 2. Code Generation ---
    # Generate a unique name for the model function to avoid world age issues.
    random_suffix = rand(10000:99999)
    model_func_name = Symbol("bstm_dynamic_model_$(random_suffix)")

    # Call the text assembler to get the model's source code, expression, and registry.
    model_string, expr, registry = bstm_text_assembler(options, model_func_name)

    # Update the configuration with the generated model code for inspection.
    config_dict = Dict(pairs(options))
    config_dict[:generated_model_code] = model_string
    new_config = NamedTuple(config_dict)

    if get(new_config, :verbose, true)
        _print_finalized_parameters(new_config)
    end

    # --- 3. Model Evaluation and Instantiation ---
    # Evaluate the generated @model expression and get the function object back directly.
    # This pattern avoids world-age issues that can arise from using `getfield`.
    model_func = Core.eval(@__MODULE__, quote
        $(expr) # The parsed @model expression from bstm_text_assembler
        $(model_func_name) # Return the function object itself
    end)

    # Instantiate the Turing Model Object using invokelatest for type stability.
    model_instance = Base.invokelatest(model_func, new_config, registry)

    # --- 4. Prior Predictive Check and Validation ---
    if get(new_config, :verbose, true)
        println("\n--- Running prior predictive check ---")
    end

    prior_sample = nothing
    try
        # Run a single draw from the prior to validate the model structure.
        redirect_stderr(devnull) do
            prior_sample = Base.invokelatest(rand, model_instance)
        end
    
        if get(new_config, :verbose, true) && !isnothing(prior_sample)
            println("Prior sample check successful. Sample values:")
            display(prior_sample)
        end
    catch e 
        # Provide detailed, user-friendly error diagnostics if the check fails.
        _bstm_error_handler(e, model_instance)
    end

    if get(new_config, :verbose, true)
        println("--------------------------------------\n")
    end

    # Return the fully configured and validated model object.
    return model_instance
end




"""
    bstm_core(formula::String, data::DataFrame; kwargs...)

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
function bstm_core(formula::String, data::DataFrame; kwargs...)
    # This overload defaults the calling_module to `Main`, which is a sensible
    # default for interactive use. The `@bstm` macro should be preferred for
    # use inside other modules to ensure correct scope resolution.
    return bstm_core(formula, data, Main; kwargs...)
end
 

"""
    bstm_text_assembler(config::NamedTuple, model_func_name::Symbol)

Assembles the full Turing model code as a string and a Julia `Expr` from the
provided configuration.

# Version
v1.1.4 (2026-08-15)

# Rationale
This function is the core of the `bstm` code generation pipeline. It translates the
high-level `config` object into a complete, executable Turing `@model` definition.
This version corrects a scoping issue that caused an `UndefVarError: Turing not defined`
during dynamic model evaluation. By adding `using Turing` (which is valid in a local
scope) to the top of the generated function body, it ensures that the `Turing` module
is explicitly available in the scope where the `@model` macro is expanded, resolving
the error without causing a syntax error. This version also ensures device-aware
memory allocation for the linear predictor `eta`.

# Workflow
1.  **Initialization**: Determines the model architecture and sets up the AD-safe
    initialization string for `eta`, ensuring allocation on the correct device.
2.  **Fragment Generation**: Calls helper functions to generate code strings for
    different parts of the model (intercept, fixed effects, etc.).
3.  **Component Loop**: Iterates through all components, calling `get_priors` and
    `get_updates` for each to generate their specific code fragments.
4.  **Assembly**: Consolidates all generated code fragments into a single,
    well-formatted model string.
5.  **Parsing**: Parses the final model string into a Julia `Expr` object.

# Arguments
- `M::NamedTuple`: The complete model configuration object.
- `model_func_name::Symbol`: The unique name for the generated Turing model function.

# Returns
- A tuple `(model_string, expr, registry)`.
"""
function bstm_text_assembler(M::NamedTuple, model_func_name::Symbol)
    arch = get(M, :model_arch, "univariate")
    is_multivariate = arch == "multivariate"
    eta_name = is_multivariate ? "eta_latent" : "eta"
 
    eta_init = if get(M, :add_intercept, false)
        is_multivariate ? "intercept' .+ fill!(Array{T}(undef, N, K), 0)" : "intercept .+ fill!(Array{T}(undef, N), 0)"
    else
        is_multivariate ? "fill!(Array{T}(undef, N, K), 0)" : "fill!(Array{T}(undef, N), 0)"
    end

    outcomes_N = get(M, :outcomes_N, 1)
    spec_registry = Dict{Symbol, Any}()
    
    priors_acc = String[]
    updates_acc = String[]
    likelihood_acc = String[]

    main_spatial_spec = nothing
    main_temporal_spec = nothing
    
    has_custom_likelihood_from_component = any(spec -> spec.component_obj isa PointProcess, M.components)
    has_custom_likelihood_from_family = any(spec -> string(get(spec, :family, "")) == "ordinal", M.likelihood_specs)
    has_custom_likelihood = has_custom_likelihood_from_component || has_custom_likelihood_from_family

    # --- Generate all code fragments ---
    intercept_priors, _ = _generate_intercept_block(M, is_multivariate, eta_name)
    push!(priors_acc, intercept_priors)
    
    push!(updates_acc, _generate_offset_block(M, is_multivariate, eta_name))
    
    fixed_effects_priors, fixed_effects_update = _generate_fixed_effects_block(M, is_multivariate, eta_name)
    push!(priors_acc, fixed_effects_priors)
    push!(updates_acc, fixed_effects_update)

    if get(M, :is_multivariate_dynamics, false)
        mv_dyn_key = M[:multivariate_dynamics_key]
        spec_idx = findfirst(s -> string(s.key) == mv_dyn_key, M.components)
        if !isnothing(spec_idx)
            spec = M.components[spec_idx]
            spec_registry[spec.key] = spec
            push!(priors_acc, get_priors(spec.component_obj, spec, arch, nothing, M))
            push!(updates_acc, get_updates(spec.component_obj, spec, arch, nothing, M))
        end
    end

    for spec in M.components
        if get(M, :is_multivariate_dynamics, false) && string(spec.key) == M[:multivariate_dynamics_key]
            continue
        end
        spec_registry[spec.key] = spec
        for k in 1:outcomes_N
            outcome_idx = is_multivariate ? k : nothing
            push!(priors_acc, get_priors(spec.component_obj, spec, arch, outcome_idx, M))
            push!(updates_acc, get_updates(spec.component_obj, spec, arch, outcome_idx, M))
        end

        if spec.structure == :spatial && isnothing(main_spatial_spec); main_spatial_spec = spec; end
        if spec.structure == :temporal && isnothing(main_temporal_spec); main_temporal_spec = spec; end
    end

    push!(priors_acc, _generate_likelihood_section(M, is_multivariate))
    
    st_interaction_block = _generate_st_interaction_block(M, main_spatial_spec, main_temporal_spec, is_multivariate, eta_name)
    push!(updates_acc, st_interaction_block)

    householder_priors, householder_update = _generate_householder_reflection_block(M, is_multivariate, eta_name)
    push!(priors_acc, householder_priors)
    push!(updates_acc, householder_update)
    
    nested_priors, nested_updates, nested_likelihoods = _generate_nested_model_block(M, is_multivariate, eta_name)
    push!(priors_acc, nested_priors)
    push!(updates_acc, nested_updates)
    push!(likelihood_acc, nested_likelihoods)

    final_likelihood = if has_custom_likelihood
        has_custom_likelihood_from_family ? _generate_final_likelihood_block(M, is_multivariate) : ""
    else
        _generate_final_likelihood_block(M, is_multivariate)
    end
    push!(likelihood_acc, final_likelihood)
    
    # --- Assemble the final model string ---
    function _indent_block(text::String, level=1)
        if isempty(strip(text)) return "" end
        indent_str = "    " ^ level
        return indent_str * replace(strip(text), "\n" => "\n" * indent_str)
    end

    priors_code = join(filter(s -> !isempty(strip(s)), priors_acc), "\n\n")
    updates_code = join(filter(s -> !isempty(strip(s)), updates_acc), "\n\n")
    likelihood_code = join(filter(s -> !isempty(strip(s)), likelihood_acc), "\n\n")

    model_string = """
    @model function $(model_func_name)(M, spec_registry)
        noise = M.noise
        N = M.y_N
        K = $(outcomes_N)
        T = Float64

        # --- Priors & Hyperparameters ---
    $(_indent_block(priors_code))

        # --- Linear Predictor Assembly ---
        $(eta_name) = $(eta_init)

    $(_indent_block(updates_code))

        # --- Likelihood ---
    $(_indent_block(likelihood_code))
    end
    """
 
    try
        return model_string, Meta.parse(model_string), spec_registry
    catch e
        println("BSTM Assembler Error: Failed to parse the generated model string.")
        println(model_string)
        rethrow(e)
    end
end

 
 
"""
    resolve_technical_primitive(module_metadata::Dict{Symbol, Any}, M, priors_dict, scheme::Symbol)

Instantiates a `ComponentModel` object from its parsed formula representation. This function
acts as a factory, resolving hyperpriors and calling the appropriate constructor from the
`COMPONENT_CONSTRUCTORS` registry.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function is the central factory in the `bstm` component system. It translates
the parsed representation of a formula module (e.g., `random(s_idx, model=:bym2)`)
into a concrete `ComponentModel` struct instance (e.g., `BYM2(...)`). It achieves
this by looking up the specified model name in the `COMPONENT_CONSTRUCTORS` registry
and calling the appropriate constructor with resolved hyperpriors. It also handles
the recursive instantiation of `Composed` components for operators like `⊗` and `|>`,
making it a critical part of the model configuration pipeline.

# Arguments
- `module_metadata::Dict`: The parsed data for the module from the formula.
- `M`: The main model configuration dictionary.
- `priors_dict::Dict`: A dictionary of globally specified hyperpriors.
- `scheme::Symbol`: The active prior scheme (e.g., `:pcpriors`).

# Returns
- A `ComponentModel` object (e.g., an instance of `BYM2`, `AR1`, or `Composed`).
"""
function resolve_technical_primitive(module_metadata::Dict{Symbol, Any}, M, priors_dict, scheme::Symbol)
    m_type = module_metadata[:type]
    m_params = module_metadata[:params]
    calling_mod = get(M, :calling_module, Main)

    # Handle composed components recursively.
    if m_type == :interact
        op = m_params[:operator]
        components_data = m_params[:components]
        components_metadata = map(c_node -> Dict(:key=>"temp", :type => c_node.module_type, :params => c_node.args, :variables => get(c_node.args, :positional_args, [])), components_data)
        resolved_components = [resolve_technical_primitive(comp_meta, M, priors_dict, scheme) for comp_meta in components_metadata]
        return Composed(resolved_components, op)
    end

    # Handle standard components.
    model_name = if haskey(m_params, :model)
        m_params[:model]
    else
        # Infer default model based on the structure if no model is specified.
        if m_type == :spatial
            haskey(M, :W) ? :bym2 : :iid
        elseif m_type == :temporal
            :rw2
        else
            :iid
        end
    end

    model_name_str = string(model_name)
    resolved_priors = resolve_hyperpriors(model_name_str, priors_dict, m_params, scheme, calling_mod)
    
    if !haskey(COMPONENT_CONSTRUCTORS, model_name)
        error("Component model ':$model_name' is not a recognized model type.")
    end
    
    constructor_func = COMPONENT_CONSTRUCTORS[model_name]
    return constructor_func(resolved_priors, m_params)
end


"""
    build_structure_template(model_type::Symbol, n::Int; W::Union{AbstractMatrix, Nothing}=nothing)

Creates a precision matrix template and its spectral decomposition for a GMRF model. 
"""
function build_structure_template(model_type::Symbol, n::Int; W::Union{AbstractMatrix, Nothing}=nothing)
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

    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values

    if rank_deficiency > 0
        scaling_factor = _compute_scaling_factor(L, rank_deficiency)
        Q_template = Q_template ./ scaling_factor
        L = L ./ scaling_factor
    else
        scaling_factor = 1.0
    end

    return (matrix=Q_template, scaling_factor=scaling_factor, U=U, L=L)
end
 




"""
    _compute_scaling_factor(evals::Vector{Float64}, rank_deficiency::Int)

Computes a robust scaling factor for a precision matrix from its eigenvalues.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function is a core utility for scaling precision matrices in intrinsic Gaussian
Markov Random Field (GMRF) models (e.g., ICAR, RW1, RW2). These models often have
singular precision matrices (i.e., they are rank-deficient), meaning some eigenvalues
are zero. A proper scaling factor is essential for ensuring the identifiability of
the model and for numerical stability during sampling. This method avoids using a
fixed tolerance to identify zero eigenvalues, which can be sensitive to floating-point
noise. Instead, it leverages the known `rank_deficiency` of the GMRF model to
correctly identify the structural zero eigenvalues.

# Mathematical Formulation
The scaling factor `c` is defined as the geometric mean of the `n - rank_deficiency`
non-zero eigenvalues. This ensures that the determinant of the scaled precision
matrix is 1.
\$c = \\exp\\left( \\frac{1}{n - \\text{rank\\_deficiency}} \\sum_{i=\\text{rank\\_deficiency}+1}^{n} \\log(\\lambda_i) \\right)\$
where \$\\lambda_i\$ are the non-zero eigenvalues.

# Arguments
- `evals::Vector{Float64}`: A vector of eigenvalues from a symmetric matrix.
- `rank_deficiency::Int`: The known rank deficiency of the matrix (number of zero eigenvalues).

# Returns
- `Float64`: The computed scaling factor.
"""
function _compute_scaling_factor(evals::Vector{Float64}, rank_deficiency::Int)
    # Sort eigenvalues in ascending order to easily discard the smallest ones,
    # which correspond to the null space.
    sorted_evals = sort(evals)
    
    n = length(sorted_evals)
    if n <= rank_deficiency
        # If the number of eigenvalues is less than or equal to the rank deficiency,
        # it implies all eigenvalues are effectively zero or the matrix is too small.
        return 1.0
    end
    
    # Select the eigenvalues that are not part of the null space.
    # These are the `n - rank_deficiency` largest eigenvalues.
    positive_evals = sorted_evals[(rank_deficiency + 1):end]
    
    if isempty(positive_evals)
        # If no positive eigenvalues are found, return 1.0 to avoid errors.
        return 1.0
    end
    
    # The scaling factor is the geometric mean of the positive eigenvalues.
    # This is a standard method for ensuring the determinant of the scaled
    # precision matrix is 1.
    return exp(mean(log.(positive_evals)))
end


"""
    evaluate_cross_kernel_matrix(coords1::AbstractMatrix, coords2::AbstractMatrix, param_val::Real, ls::Union{Real, AbstractVector}, kernel_type::Symbol)
 

# Version
v1.5.7 (2026-08-13)

# Rationale
This function is a core utility for Gaussian Process (GP) components, particularly
for tasks such as out-of-sample prediction or computing cross-covariances in multi-output
GP models. It calculates the covariance between two distinct sets of input points
(`coords1` and `coords2`) based on a specified kernel function and hyperparameters.
This version is consistent with `evaluate_kernel_matrix`, including support for
various kernel types and handling of both isotropic and Automatic Relevance
Determination (ARD) lengthscales.

# Arguments
- `coords1::AbstractMatrix`: An `N1 x D` matrix of data points.
- `coords2::AbstractMatrix`: An `N2 x D` matrix of data points.
- `param_val::Real`: The signal variance (\$\\sigma^2\$) of the kernel.
- `ls::Union{Real, AbstractVector}`: The lengthscale(s) (\$\\ell\$) of the kernel.
  A `Real` value assumes an isotropic kernel, while a `Vector` of length `D` enables
  ARD with a separate lengthscale for each dimension.
- `kernel_type::Symbol`: The type of kernel to evaluate.

# Supported Kernels and Mathematical Formulation
- `:gaussian`, `:se`, `:rbf`: Squared Exponential kernel.
  \$k(x, x') = \\sigma^2 \\exp\\left(-\\frac{\\|x - x'\\|^2}{2\\ell^2}\\right)\$
- `:exponential`, `:matern12`: Exponential kernel (Matérn with \$\\nu=1/2\$).
  \$k(x, x') = \\sigma^2 \\exp\\left(-\\frac{\\|x - x'\\|}{\\ell}\\right)\$
- `:matern32`: Matérn kernel with \$\\nu=3/2\$.
  \$k(x, x') = \\sigma^2 \\left(1 + \\frac{\\sqrt{3}\\|x - x'\\|}{\\ell}\\right) \\exp\\left(-\\frac{\\sqrt{3}\\|x - x'\\|}{\\ell}\\right)\$
- `:matern52`: Matérn kernel with \$\\nu=5/2\$.
  \$k(x, x') = \\sigma^2 \\left(1 + \\frac{\\sqrt{5}\\|x - x'\\|}{\\ell} + \\frac{5\\|x - x'\\|^2}{3\\ell^2}\\right) \\exp\\left(-\\frac{\\sqrt{5}\\|x - x'\\|}{\\ell}\\right)\$
- `:spherical`: Spherical kernel.
  \$k(x, x') = \\sigma^2 \\left(1 - \\frac{3}{2}\\frac{\\|x - x'\\|}{\\ell} + \\frac{1}{2}\\left(\\frac{\\|x - x'\\|}{\\ell}\\right)^3\\right)\$ for \$\\|x - x'\\| < \\ell\$, else \$0\$.
- `:cosine`: Cosine kernel.
  \$k(x, x') = \\sigma^2 \\cos\\left(\\frac{2\\pi\\|x - x'\\|}{\\ell}\\right)\$
- `:linear`: Linear kernel.
  \$k(x, x') = \\sigma^2 x^T x'\$
- `:constant`: Constant kernel.
  \$k(x, x') = \\sigma^2\$
"""
function evaluate_cross_kernel_matrix(coords1::AbstractMatrix, coords2::AbstractMatrix, param_val::Real, ls::Union{Real, AbstractVector}, kernel_type::Symbol)
    T = promote_type(eltype(coords1), eltype(coords2), typeof(param_val), eltype(ls))
    coords1_T = convert(AbstractMatrix{T}, coords1)
    coords2_T = convert(AbstractMatrix{T}, coords2)
    ls_T = convert(typeof(ls) <: Real ? T : AbstractVector{T}, ls)

    if kernel_type == :linear
        return param_val^2 .* (coords1_T * coords2_T')
    end

    function _sqeuclidean_broadcast_cross(X1::AbstractMatrix, X2::AbstractMatrix)
        sum(X1.^2, dims=2) .- 2 * (X1 * X2') .+ sum(X2.^2, dims=2)'
    end

    local dist_sq
    if ls isa AbstractVector # ARD case
        if size(coords1_T, 2) != length(ls_T) || size(coords2_T, 2) != length(ls_T)
            error("Dimension mismatch for ARD kernel: Number of coordinate dimensions ($(size(coords1_T, 2))) does not match number of lengthscales ($(length(ls_T))).")
        end
        dist_sq = _sqeuclidean_broadcast_cross(coords1_T ./ ls_T', coords2_T ./ ls_T')
    else # Isotropic case
        dist_sq = _sqeuclidean_broadcast_cross(coords1_T, coords2_T) ./ ls_T^2
    end
    
    dist_sq[diagind(dist_sq)] .= max.(0, diag(dist_sq))

    if kernel_type == :gaussian || kernel_type == :se || kernel_type == :rbf
        return param_val^2 .* exp.(-one(T)/2 .* dist_sq)
    
    elseif kernel_type == :exponential || kernel_type == :matern12
        d = sqrt.(dist_sq)
        return param_val^2 .* exp.(-d)
    
    elseif kernel_type == :matern32
        d = sqrt.(dist_sq)
        val = sqrt(convert(T, 3.0)) .* d
        return param_val^2 .* (one(T) .+ val) .* exp.(-val)
    
    elseif kernel_type == :matern52
        d = sqrt.(dist_sq)
        val = sqrt(convert(T, 5.0)) .* d
        return param_val^2 .* (one(T) .+ val .+ (val.^2 ./ convert(T, 3.0))) .* exp.(-val)

    elseif kernel_type == :spherical
        d = sqrt.(dist_sq)
        K = zeros(T, size(d))
        mask = d .< one(T)
        K[mask] = param_val^2 .* (one(T) .- 1.5 .* d[mask] .+ 0.5 .* d[mask].^3)
        return K

    elseif kernel_type == :cosine
        if ls isa AbstractVector
            @warn "Cosine kernel with ARD lengthscale is not standard. Using the first lengthscale for an isotropic kernel."
            ls_T = ls_T[1]
        end
        d_euclidean = sqrt.(_sqeuclidean_broadcast_cross(coords1_T, coords2_T))
        return param_val^2 .* cos.(2.0 * pi .* d_euclidean ./ ls_T)

    elseif kernel_type == :constant
        return fill(convert(T, param_val^2), size(dist_sq))

    else
        @warn "Kernel '$(kernel_type)' not explicitly handled in evaluate_cross_kernel_matrix. Defaulting to Squared Exponential."
        return param_val^2 .* exp.(-one(T)/2 .* dist_sq)
    end
end

 




"""
    observation_volatility(M::NamedTuple)

Generates Turing code fragments for the observation error variance, handling both
constant variance and a spatiotemporal stochastic volatility (SV) model.

# Version
v1.1.0 (2026-08-13)

# Rationale
This function is a core part of the code generation pipeline. It provides a switch
to model observation noise either as a simple constant parameter or as a complex,
spatiotemporally varying field. The stochastic volatility option is crucial for
heteroscedastic data, where the measurement error changes across space and time.
This is implemented using Random Fourier Features (RFF) to create a flexible,
non-parametric model for the log-variance.   correctly
handle both univariate and multivariate models, ensuring the `y_sigma` parameter
has the correct dimensions in all cases.

# Mathematical Formulation
- **Constant Variance**: \$\\sigma_y\$ is a single parameter.
- **Stochastic Volatility**: The log-variance is modeled as a GP approximated by RFFs:
  `log_var(s, t) = Z(s, t) * β`
  where `Z` is the RFF basis matrix and `β` are coefficients. The standard deviation
  is then `σ_y(s, t) = exp(log_var(s, t) / 2)`.

# Arguments
- `M::NamedTuple`: The model configuration object.

# Returns
- A `NamedTuple` with code strings for `:priors` and `:calculation`.
"""
function observation_volatility(M::NamedTuple)
    is_multivariate = get(M, :model_arch, "univariate") == "multivariate"
    
    if get(M, :volatility, false)
        required_keys = [:M_rff_sigma, :W_sigma_fixed, :b_sigma_fixed, :coords_st]
        if !all(k -> haskey(M, k), required_keys)
            error("Stochastic volatility is enabled, but required keys are missing from the model configuration: $required_keys.")
        end

        priors_str = """
        sigma_log_var ~ DynamicPPL.NamedDist(Exponential(1.0), :sigma_log_var)
        beta_vol ~ DynamicPPL.NamedDist(MvNormal(fill!(Array{T}(undef, M.M_rff_sigma), 0), sigma_log_var^2 * I), :beta_vol)
        """

        calc_str = """
        # Stochastic Volatility Calculation
        vol_proj = (M.coords_st * M.W_sigma_fixed) .+ M.b_sigma_fixed'
        log_var_latent = sqrt(2.0 / M.M_rff_sigma) .* cos.(vol_proj) * beta_vol
        y_sigma_sv = exp.(log_var_latent ./ 2.0)
        """
        
        final_calc_str = is_multivariate ? "y_sigma = y_sigma_sv .* y_sigma_const'" : "y_sigma = y_sigma_const .* y_sigma_sv"
        
        return (priors=priors_str, calculation="$(calc_str)\n    $(final_calc_str)")
    else
        priors_str = ""
        
        calc_str = if is_multivariate
            "y_sigma = y_sigma_const'"
        else
            "y_sigma = fill(y_sigma_const, N)"
        end
        return (priors=priors_str, calculation=calc_str)
    end
end



  
"""
    generate_inducing_points(coords::AbstractMatrix, n_inducing::Int; method::String="kmeans", seed::Int=42)

Selects a representative subset of coordinates to serve as inducing points for sparse
Gaussian Process (GP) models.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function is a core utility for all sparse GP components (e.g., `FITC`, `SVGP`).
Sparse GPs reduce the O(N³) complexity of exact GPs by summarizing the data through a
smaller set of `M` inducing points, where `M << N`. The quality of the GP approximation
depends heavily on the placement of these points. This function provides several
standard methods for their selection, offering a trade-off between computational
speed and representativeness. The implementation is consistent with the refactored
architecture and provides robust, well-tested methods for this critical task.

# Methods
- `:kmeans`: (Default) Uses k-means clustering to find the centroids of the data
  points. This is often the most effective method as it places inducing points in
  high-density areas.
- `:random`: Simple and fast random sub-sampling of the data points.
- `:quantile`: Generates a systematic grid of target points based on the marginal
  quantiles of the data and finds the nearest actual data points to these targets.
  This ensures good coverage of the data's distributional space.
- `:regular`: Generates a regular grid of target points over the data's range and
  finds the nearest actual data points. This ensures good spatial coverage.

# Arguments
- `coords::AbstractMatrix`: An `N x D` matrix of data point coordinates.
- `n_inducing::Int`: The number of inducing points to select.
- `method::String`: The selection method to use.
- `seed::Int`: A random seed for reproducibility of `:random` and `:kmeans`.

# Returns
- An `M x D` matrix of inducing point coordinates, where `M <= n_inducing`.
"""
function generate_inducing_points(
    coords::AbstractMatrix, 
    n_inducing::Int; 
    method::String="kmeans", 
    seed::Int=42
)
    n_obs, n_dims = size(coords)

    if n_inducing >= n_obs
        return coords
    end

    Random.seed!(seed)

    if method == "random"
        # Simple stochastic selection without replacement.
        selected_idx = StatsBase.sample(1:n_obs, n_inducing, replace=false)
        return coords[selected_idx, :]

    elseif method == "kmeans"
        # Centroid-based selection via Clustering.jl.
        # kmeans expects observations in columns: [dims x obs].
        kmeans_res = Clustering.kmeans(coords', n_inducing; maxiter=200, display=:none)
        return kmeans_res.centers'

    elseif method == "quantile" || method == "regular"
        # Systematic mapping methods requiring KDTree for efficiency.
        target_pts = zeros(Float64, n_inducing, n_dims)
        
        if method == "quantile"
            # Density-aware target generation using marginal quantiles.
            probs = range(0.0, stop=1.0, length=n_inducing)
            for d in 1:n_dims
                target_pts[:, d] = Statistics.quantile(coords[:, d], probs)
            end
        else # method == "regular"
            # Grid-like target generation across marginal ranges.
            for d in 1:n_dims
                v_min, v_max = extrema(coords[:, d])
                target_pts[:, d] = range(v_min, stop=v_max, length=n_inducing)
            end
        end

        # Efficient Nearest Neighbor Search using KDTree.
        tree = KDTree(coords')
        
        # Find the single nearest observation for each target coordinate.
        nn_indices_vec, _ = knn(tree, target_pts', 1, true)
        
        # Extract the scalar index from each neighbor search result and deduplicate.
        unique_nn_indices = unique([idx_list[1] for idx_list in nn_indices_vec])
        
        return coords[unique_nn_indices, :]

    else
        @warn "Inducing point method '$method' not recognized. Falling back to random selection."
        selected_idx = StatsBase.sample(1:n_obs, n_inducing, replace=false)
        return coords[selected_idx, :]
    end
end


"""
    create_pc_prior(param_name::Symbol, constraint::Tuple)

Creates a Penalized Complexity (PC) prior distribution from a user-specified quantile constraint.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function is a core utility for the prior specification engine. It translates an
intuitive belief about a parameter's scale (e.g., "the probability that sigma is
greater than 1.0 should be 5%") into a formal, well-defined prior distribution.
This aligns with the principles of Penalized Complexity priors, which favor simpler
models by shrinking parameters towards a base case (e.g., zero variance or no
correlation) unless the data provides strong evidence to the contrary. This version
is consistent with the refactored architecture and includes detailed mathematical
documentation.

# Mathematical Formulation
The function maps a quantile constraint `(U, α)` to the hyperparameter `λ` of an
`Exponential(λ)` prior. The specific formula depends on the parameter type:

- **For `sigma` or `kappa` (scale parameters)**:
  - Constraint: \$P(\\text{param} > U) = \\alpha\$
  - Derivation: \$e^{-\\lambda U} = \\alpha \\implies \\lambda = -\\log(\\alpha) / U\$

- **For `rho` (correlation parameter on [0, 1])**:
  - The prior is placed on a transformed parameter \$\\theta = -\\log(1-\\rho) \\sim \\text{Exponential}(\\lambda)\$.
  - Constraint: \$P(\\rho > U) = \\alpha\$
  - Derivation: \$P(\\theta > -\\log(1-U)) = e^{-\\lambda(-\\log(1-U))} = (1-U)^{\\lambda} = \\alpha \\implies \\lambda = \\log(\\alpha) / \\log(1-U)\$

- **For `lengthscale`**:
  - The prior is placed on the inverse \$\\theta = 1/\\ell \\sim \\text{Exponential}(\\lambda)\$.
  - Constraint: \$P(\\ell < U) = \\alpha\$
  - Derivation: \$P(\\theta > 1/U) = e^{-\\lambda/U} = \\alpha \\implies \\lambda = -U \\log(\\alpha)\$

- **For other parameters**:
  - A symmetric `Normal(0, σ)` prior is assumed, where the standard deviation `σ` is
    derived from a two-sided constraint \$P(|\\text{param}| > U) = \\alpha\$.

# Arguments
- `param_name::Symbol`: The base name of the parameter (e.g., `:sigma`, `:rho`).
- `constraint::Tuple`: A tuple `(U, α)` or `(U, α, direction)` defining the quantile constraint.

# Returns
- A `Distribution` object representing the calculated prior.
"""
function create_pc_prior(param_name::Symbol, constraint::Tuple)
    direction = :upper
    if length(constraint) == 2
        U, α = constraint
    elseif length(constraint) == 3
        U, α, direction = constraint
    else
        error("PC prior constraint must be a tuple of (U, α) or (U, α, direction).")
    end
    
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
        # Fallback for other parameters, assuming a symmetric Normal prior.
        # P(|param| > U) = α  => P(Z > U/σ) = α/2
        sigma = -U / quantile(Normal(0, 1), α / 2)
        return Normal(0, sigma)
    end
end

 

"""
    create_fixed_design(formula_rhs::AbstractString, data::DataFrame, calling_module::Module; contrasts=Dict{Symbol, Any}())

Creates a fixed-effects design matrix (`X`) from a formula string using `StatsModels.jl`.

# Version
v1.0.3 (2026-08-15)

# Rationale
This function is a core utility for the model configuration engine. It acts as a
robust wrapper around `StatsModels.jl` to parse a formula's right-hand side and
generate the corresponding design matrix.   prevent a
`LoadError` by explicitly evaluating the formula expression in the `bstm` module's
scope (`@__MODULE__`) instead of the `calling_module`'s scope. This ensures the
`StatsModels.@formula` macro can always be found, as `StatsModels` is a direct
dependency of `bstm`.

# Arguments
- `formula_rhs::AbstractString`: A string representing the right-hand side of the model formula (e.g., "0 + x + y*z").
- `data::DataFrame`: The input data frame containing the variables.
- `calling_module::Module`: The module in which the formula should be evaluated (used by `StatsModels` to resolve variables).
- `contrasts`: An optional dictionary specifying contrast coding for categorical variables.

# Returns
- A tuple `(NamedArray, Union{StatsModels.FormulaTerm, Nothing})`.
"""
function create_fixed_design(
    formula_rhs::AbstractString, 
    data::DataFrame, 
    calling_module::Module; 
    contrasts=Dict{Symbol, Any}()
)
    df_internal = copy(data)
    final_rhs_string = strip(formula_rhs)

    if isempty(final_rhs_string)
        return NamedArray(zeros(size(df_internal, 1), 0), (1:size(df_internal, 1), Symbol[])), nothing
    end

    if final_rhs_string == "1"
        return NamedArray(ones(size(df_internal, 1), 1), (1:size(df_internal, 1), [:Intercept])), nothing
    end

    try
        placeholder_name = :__y_placeholder
        if !hasproperty(df_internal, placeholder_name)
            df_internal[!, placeholder_name] = zeros(size(df_internal, 1))
        end

        # Explicitly qualify the @formula macro to prevent LoadError.
        formula_expression = Meta.parse("StatsModels.@formula($placeholder_name ~ $final_rhs_string)")
        
        # Evaluate in the current module's scope to ensure StatsModels is found.
        # The variables in the formula string are resolved later by apply_schema.
        dynamic_formula = Core.eval(@__MODULE__, formula_expression)

        data_schema = StatsModels.schema(dynamic_formula, df_internal, contrasts)
        applied_formula = StatsModels.apply_schema(dynamic_formula, data_schema, StatsModels.RegressionModel)

        _, model_matrix_numeric = StatsModels.modelcols(applied_formula, df_internal)
        coefficient_labels = StatsModels.coefnames(applied_formula.rhs)

        label_vector = coefficient_labels isa AbstractString ? [Symbol(coefficient_labels)] : Symbol.(coefficient_labels)

        return NamedArray(model_matrix_numeric, (1:size(model_matrix_numeric, 1), label_vector)), applied_formula

    catch design_error
        @warn "BSTM Registry: create_fixed_design expansion failed for: '$final_rhs_string'. Error of type '$(typeof(design_error))' occurred. Check formula syntax and variable names."
        return NamedArray(zeros(size(df_internal, 1), 0), (1:size(df_internal, 1), Symbol[])), nothing
    end
end


 
 

"""
    show_model(m::DynamicPPL.Model)

Displays a comprehensive and well-formatted summary of the `bstm` model configuration,
including likelihoods, priors, and component-specific parameters.

# Version
v1.1.0 (2026-08-13)

# Rationale
This function provides a user-friendly way to inspect the full specification of a
model before or after fitting. This updated version replaces the previous, more basic
implementation with a detailed, structured output. It iterates through all parts of
the model configuration (`likelihood`, `intercept`, `fixed effects`, and all `components`)
and prints their finalized parameters, clearly indicating whether each value was
provided by the user or assigned a default. This is crucial for debugging and for
verifying that the model was specified as intended.

# Arguments
- `m`: The Turing model instance generated by `@bstm`.

# Returns
- `nothing`. The function prints the summary to the console.
"""
function show_model(m::DynamicPPL.Model)
    println("\n--- Model Summary ---\n")
    config = m.args.M
    println("Model Name:           ", get(config, :model_name, nameof(m.f)))
    println("Model Architecture:   ", get(config, :model_arch, "N/A"))
    
    # Likelihood Configuration
    println("\n[ Likelihood ]")
    for (i, spec) in enumerate(config.likelihood_specs)
        outcome = config.outcomes[i]
        println("  Outcome: $outcome")
        user_params = spec
        
        lik_params_to_show = [
            :family, :log_offsets, :weights, :trials, :zero_inflated, 
            :volatility, :censor_lower, :censor_upper, :hurdle, :latent_dist
        ]
        for p_name in lik_params_to_show
            if haskey(user_params, p_name)
                 _print_param(p_name, user_params[p_name], :user; indent=4)
            end
        end
    end

    # Intercept Configuration
    println("\n[ Intercept ]")
    if get(config, :add_intercept, false)
        is_user_provided = haskey(config, :intercept_prior) && config.intercept_prior != Normal(0, 5)
        _print_param(:prior, get(config, :intercept_prior, Normal(0,5)), is_user_provided ? :user : :default; indent=2)
    else
        println("  - Intercept removed from model.")
    end

    # Fixed Effects Configuration
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
            _print_param(name, prior_obj, is_default ? :default : :user; indent=4)
        end
    end

    # Model Components
    if haskey(config, :components) && !isempty(config.components)
        println("\n[ Model Components ]")
        for spec in config.components
            println("  --- Component: $(spec.key) ---")
            component_obj = spec.component_obj
            println("    - Type:      $(typeof(component_obj))")
            println("    - Structure: $(spec.structure)")
            println("    - Variable:  $(spec.var)")
            
            println("    - Parameters:")
            user_provided_params_raw = spec.params 
            
            all_param_names = Set(fieldnames(typeof(component_obj)))
            union!(all_param_names, keys(user_provided_params_raw))
            
            for param_name in sort(collect(all_param_names))
                if param_name in [:positional_args, :structure, :in_dims]; continue; end
                
                local final_val, status
                if param_name in fieldnames(typeof(component_obj))
                    final_val = getfield(component_obj, param_name)
                    status = haskey(user_provided_params_raw, param_name) ? :user : :default
                else
                    final_val = get(user_provided_params_raw, param_name, "N/A")
                    status = haskey(user_provided_params_raw, param_name) ? :user : :default
                end
                
                if final_val isa Vector{<:UnivariateDistribution}
                    println("      - $(rpad(param_name, 20)): [")
                    for (idx, p_dist) in enumerate(final_val)
                        println("        $idx: $p_dist")
                    end
                    println("      ] $(status == :user ? "(User-provided)" : "(Default)")")
                else
                    _print_param(param_name, final_val, status; indent=6)
                end
            end
        end
    else
        println("\n[ Model Components ]")
        println("  None")
    end

    # Generated Code
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
    config = m.args.M
    model_name = get(config, :model_name, nameof(m.f))
    
    lines = String[]
    push!(lines, "@model function $(model_name)(M; T::Type=Float64)")
    push!(lines, "    # --- Priors & Hyperparameters ---")

    # Likelihood-specific priors
    family = string(get(config.likelihood_specs[1], :family, "gaussian"))
    
    if family == "negbin"
        push!(lines, "    r_nb ~ $(_distribution_to_string(Exponential(1.0)))")
    end
    if get(config, :use_zi, false)
        push!(lines, "    phi_zi ~ $(_distribution_to_string(Beta(1, 1)))")
    end
    if get(config, :user_provided_hurdle, false)
        push!(lines, "    lik_phi_hurdle ~ $(_distribution_to_string(Beta(1,1)))")
    end
    if family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"] && !get(config, :volatility, false)
        push!(lines, "    y_sigma ~ $(_distribution_to_string(Exponential(1.0)))")
    end
    if get(config, :volatility, false)
        push!(lines, "    sigma_log_var ~ $(_distribution_to_string(Exponential(1.0)))")
        # Simplified representation of beta_vol for pseudo-code
        push!(lines, "    beta_vol ~ MvNormal(zeros(T, M.M_rff_sigma), sigma_log_var^2 * I)")
    end
    if get(config, :outcomes_N, 1) > 1
        push!(lines, "    L_corr ~ $(_distribution_to_string(LKJCholesky(get(config, :outcomes_N, 1), 1.0)))")
    end
    if family == "student_t"
        push!(lines, "    lik_nu_student_t ~ $(_distribution_to_string(Exponential(1.0)))")
    end
    if family in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"]
        push!(lines, "    lik_extra_params ~ $(_distribution_to_string(Exponential(1.0)))")
    end

    # Ordinal-specific priors
    ordinal_spec_idx = findfirst(s -> string(get(s, :family, "")) == "ordinal", config.likelihood_specs)
    if !isnothing(ordinal_spec_idx)
        spec = config.likelihood_specs[ordinal_spec_idx]
        K_ordinal = get(spec, :K, 0)
        if K_ordinal > 2
            push!(lines, "    ordinal_alpha_raw_1 ~ $(_distribution_to_string(Normal(0, 5)))")
            push!(lines, "    ordinal_alpha_diffs ~ $(_distribution_to_string(Fill(Exponential(1.0), K_ordinal - 2)))")
        elseif K_ordinal == 2
            push!(lines, "    ordinal_alpha_raw_1 ~ $(_distribution_to_string(Normal(0, 5)))")
        end
        if get(spec, :latent_dist, :logistic) == :student_t
            push!(lines, "    ordinal_df ~ $(_distribution_to_string(Exponential(1.0)))")
        end
    end

    # Intercept prior
    if get(config, :add_intercept, false)
        intercept_prior_obj = get(config, :intercept_prior, Normal(0, 5))
        push!(lines, "    intercept ~ $(_distribution_to_string(intercept_prior_obj))")
    end

    # Fixed effects priors
    if get(config, :Xfixed_N, 0) > 0
        push!(lines, "    # Priors for fixed effects coefficients")
        is_multivariate = config.model_arch == "multivariate"
        n_fixed = config.Xfixed_N
        outcomes_N = config.outcomes_N
        
        # Proportional fixed effects
        prop_indices = collect(1:n_fixed)
        npo_indices = Int[]
        if !isnothing(ordinal_spec_idx) && haskey(config, :non_proportional_effects) && !isempty(config.non_proportional_effects)
            npo_indices = findall(x -> x in config.non_proportional_effects, config.Xfixed_names)
            prop_indices = setdiff(prop_indices, npo_indices)
        end

        if !isempty(prop_indices)
            priors_prop = get(config, :Xfixed_priors_vec, [Normal(0, 5) for _ in 1:n_fixed])[prop_indices]
            if is_multivariate
                # Simplified for pseudo-code: assume all outcomes have the same prior structure
                prior_str = _distribution_to_string(priors_prop[1]) # Take first as representative
                push!(lines, "    Xfixed_beta_prop_flat ~ filldist($(prior_str), $(length(prop_indices) * outcomes_N))")
            else
                prior_str_list = [_distribution_to_string(p) for p in priors_prop]
                push!(lines, "    Xfixed_beta_prop ~ Product([$(join(prior_str_list, ", "))])")
            end
        end

        # Non-proportional fixed effects (for ordinal)
        if !isempty(npo_indices) && K_ordinal > 1
            priors_npo = get(config, :Xfixed_priors_vec, [Normal(0, 5) for _ in 1:n_fixed])[npo_indices]
            prior_str_list = [_distribution_to_string(p) for p in priors_npo]
            push!(lines, "    beta_npo ~ Product([$(join(prior_str_list, ", "))])")
        end
    end

    # Component-specific priors
    if haskey(config, :components) && !isempty(config.components)
        for spec in config.components
            m_obj = spec.component_obj
            m_type_str = string(typeof(m_obj).name.name)
            key = spec.key
            
            push!(lines, "\n    # Priors for component: $(key) ($(m_type_str))")
            
            # This list should be comprehensive for all possible hyperparameters
            # that might have priors in any component.
            all_possible_hyperpriors = [
                :sigma, :rho, :rho1, :rho2, :unconstrained_rho, :kappa, :ls, :range, :period,
                :amplitude, :phase, :velocity, :diffusion, :pca_sd, :pdef_sd,
                :sigma_effects, :r, :K, :q, :M_nat, :alpha, :beta, :gamma, :delta, :curvature,
                :lengthscale # Added for consistency with GP/RFF
            ]

            for field_sym in all_possible_hyperpriors
                if hasproperty(m_obj, field_sym)
                    prior_dist = getfield(m_obj, field_sym)
                    if prior_dist isa Distribution || (prior_dist isa Vector && !isempty(prior_dist) && all(d -> d isa Distribution, prior_dist))
                        p_names = generate_full_variable_names(spec, config.model_arch, 1) # Use 1 for outcome_idx for pseudo-code simplicity
                        param_name_sym = get(p_names, field_sym, Symbol("$(field_sym)_$(key)")) # Fallback if not in p_names

                        if prior_dist isa Vector
                            dist_str = "Product([$(join([_distribution_to_string(d) for d in prior_dist], ", "))])"
                            push!(lines, "    $(param_name_sym) ~ $(dist_str)")
                        else
                            push!(lines, "    $(param_name_sym) ~ $(_distribution_to_string(prior_dist))")
                        end
                    end
                end
            end
            # Add innovations prior for components that have them
            p_names = generate_full_variable_names(spec, config.model_arch, 1)
            if hasproperty(spec.hyper, :n_latent) && spec.hyper.n_latent > 0
                push!(lines, "    $(p_names.innovations) ~ MvNormal(zeros(T, $(spec.hyper.n_latent)), I)")
            end
        end
    end

    push!(lines, "\n    # --- Linear Predictor Assembly ---")
    eta_parts = String[]
    is_multivariate = config.model_arch == "multivariate"
    eta_var_name = is_multivariate ? "eta_latent" : "eta"

    # Initialize eta with intercept and offsets
    if get(config, :add_intercept, false)
        push!(eta_parts, "intercept")
    end
    if haskey(config, :log_offsets) && !all(iszero, config.log_offsets)
        push!(eta_parts, "M.log_offsets")
    end
    
    eta_init_str = isempty(eta_parts) ? "zeros(T, M.y_N, $(config.outcomes_N))" : join(eta_parts, " .+ ")
    push!(lines, "    $(eta_var_name) = $(eta_init_str)")

    # Add fixed effects
    if get(config, :Xfixed_N, 0) > 0
        if is_multivariate
            push!(lines, "    $(eta_var_name) .+= M.Xfixed * reshape(Xfixed_beta_prop_flat, M.Xfixed_N, M.outcomes_N)")
        else
            push!(lines, "    $(eta_var_name) .+= M.Xfixed * Xfixed_beta_prop")
        end
    end

    # Add component effects
    if haskey(config, :components) && !isempty(config.components)
        for spec in config.components
            p_names = generate_full_variable_names(spec, config.model_arch, 1) # Use 1 for pseudo-code
            if hasproperty(p_names, :latent)
                # This is a simplification; actual update logic is more complex and depends on structure.
                # For pseudo-code, we show a generic addition.
                if is_multivariate
                    push!(lines, "    $(eta_var_name) .+= $(p_names.latent)[M.s_idx, :]") # Example for spatial/temporal
                else
                    push!(lines, "    $(eta_var_name) .+= $(p_names.latent)[M.s_idx]")
                end
            end
        end
    end

    # Add spacetime interaction
    model_st = get(config, :model_st, "none")
    if model_st != "none"
        if is_multivariate
            push!(lines, "    $(eta_var_name) .+= spacetime_interaction[M.s_idx, M.t_idx, :]")
        else
            push!(lines, "    $(eta_var_name) .+= spacetime_interaction[M.s_idx, M.t_idx]")
        end
    end

    # Apply multivariate correlation if applicable
    if is_multivariate
        push!(lines, "    eta = $(eta_var_name) * L_corr.L'")
    end

    push!(lines, "\n    # --- Likelihood ---")
    # Construct a more complete bstm_Likelihood call for pseudo-code
    lik_kwargs_parts = String[]
    if family == "negbin"; push!(lik_kwargs_parts, "r_nb=r_nb"); end
    if get(config, :user_provided_hurdle, false); push!(lik_kwargs_parts, "phi_hurdle=lik_phi_hurdle"); end
    if get(config, :use_zi, false); push!(lik_kwargs_parts, "phi_zi=phi_zi"); end
    if family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]; push!(lik_kwargs_parts, "sigma_y=y_sigma"); end
    if family == "student_t"; push!(lik_kwargs_parts, "nu=lik_nu_student_t"); end
    if family in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"]; push!(lik_kwargs_parts, "extra_params=lik_extra_params"); end
    if get(config, :user_provided_trials, false); push!(lik_kwargs_parts, "trial=M.trials"); end
    if get(config, :user_provided_weights, false); push!(lik_kwargs_parts, "weight=M.weights"); end
    if get(config, :user_provided_censor_lower, false); push!(lik_kwargs_parts, "censor_lower=M.censor_lower"); end
    if get(config, :user_provided_censor_upper, false); push!(lik_kwargs_parts, "censor_upper=M.censor_upper"); end

    lik_kwargs_str = isempty(lik_kwargs_parts) ? "" : "; $(join(lik_kwargs_parts, ", "))"

    if is_multivariate
        push!(lines, "    M.y_obs ~ bstm_Likelihood(\"$(family)\", eta $(lik_kwargs_str))")
    else
        push!(lines, "    M.y_obs ~ bstm_Likelihood(\"$(family)\", eta $(lik_kwargs_str))")
    end
    push!(lines, "end")

    return join(lines, "\n")
end


"""
    bstm_bspline_basis(x::AbstractVector, n_basis::Int, degree::Int; ...)

Generates a B-spline basis matrix. This version is CPU-only.
"""
function bstm_bspline_basis(x::AbstractVector, n_basis::Int, degree::Int; knot_method::Symbol=:quantile, custom_knots::Union{AbstractVector, Nothing}=nothing)
    p = degree
    if n_basis <= p
        error("Number of basis functions (nbins) must be greater than the spline degree. Got n_basis=$n_basis, degree=$p.")
    end

    n_interior_knots = n_basis - p

    knots = if !isnothing(custom_knots)
        custom_knots
    else
        if n_interior_knots > 0
            if knot_method == :quantile
                probs = range(0, 1, length=n_interior_knots + 2)[2:end-1]
                quantile(x, probs)
            else
                range(minimum(x), maximum(x), length=n_interior_knots + 2)[2:end-1]
            end
        else
            Float64[]
        end
    end

    boundary_knots = [minimum(x), maximum(x)]
    all_knots = sort(unique(vcat(boundary_knots, knots)))

    n_basis_possible = length(all_knots) + p - 1
    if n_basis > n_basis_possible
        @warn "Requested n_basis ($n_basis) is too high for the number of unique knots ($(length(all_knots))) and degree ($p). Reducing to $n_basis_possible."
        n_basis = n_basis_possible
    end

    t = vcat(fill(all_knots[1], p), all_knots, fill(all_knots[end], p))
    
    N = length(x)
    num_total_basis = length(t) - p - 1
    B = Matrix{Float64}(undef, N, num_total_basis); fill!(B, 0.0)

    for j in 1:num_total_basis
        B[:, j] = (t[j] .<= x .< t[j+1])
    end
    if !isempty(x) && t[end] == maximum(x)
        B[x .== t[end], num_total_basis] .= 1.0
    end

    for d in 1:p
        for j in 1:(num_total_basis - d)
            w1 = Vector{Float64}(undef, N); fill!(w1, 0.0)
            denom1 = t[j+d] - t[j]
            if denom1 > 1e-9
                w1 = (x .- t[j]) ./ denom1
            end
            
            w2 = Vector{Float64}(undef, N); fill!(w2, 0.0)
            denom2 = t[j+d+1] - t[j+1]
            if denom2 > 1e-9
                w2 = (t[j+d+1] .- x) ./ denom2
            end
            
            B[:, j] = w1 .* B[:, j] + w2 .* B[:, j+1]
        end
    end

    return (B[:, 1:n_basis], n_basis)
end



"""
    bstm_tensor_product_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int}, degrees_per_dim::Vector{Int}; ...)

Generates a tensor product B-spline basis matrix. This version is CPU-only.
"""
function bstm_tensor_product_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int}, degrees_per_dim::Vector{Int}; knot_method::Symbol=:quantile, kwargs...)
    n_dims = size(coords, 2)
    if length(nbins_per_dim) != n_dims || length(degrees_per_dim) != n_dims
        error("Number of dimensions in coords must match length of nbins_per_dim and degrees_per_dim.")
    end

    bspline_kwargs = Dict{Symbol, Any}()
    if haskey(kwargs, :custom_knots)
        bspline_kwargs[:custom_knots] = kwargs[:custom_knots]
    end

    basis_matrices_1D = Vector{AbstractMatrix{Float64}}(undef, n_dims)
    for i in 1:n_dims
        local_bspline_kwargs = copy(bspline_kwargs)
        if haskey(local_bspline_kwargs, :custom_knots) && local_bspline_kwargs[:custom_knots] isa Tuple
            local_bspline_kwargs[:custom_knots] = local_bspline_kwargs[:custom_knots][i]
        end
        
        basis_mat, _ = bstm_bspline_basis(
            coords[:, i], 
            nbins_per_dim[i], 
            degrees_per_dim[i]; 
            knot_method=knot_method, 
            local_bspline_kwargs...
        )
        basis_matrices_1D[i] = basis_mat
    end

    if isempty(basis_matrices_1D)
        return Matrix{Float64}(undef, size(coords, 1), 0)
    end

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
    bstm_wavelet_basis_1D(vals::AbstractVector, nbins::Int, family::Symbol, lengthscale::Float64)

Generates a 1D wavelet basis matrix. This version is CPU-only.
"""
function bstm_wavelet_basis_1D(vals::AbstractVector, nbins::Int, family::Symbol, lengthscale::Float64)
    Interpolations = Base.require(Base.Main, :Interpolations)

    n_obs = length(vals)
    B = Matrix{Float64}(undef, n_obs, nbins); fill!(B, 0.0)
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




"""
    _reconstruct_wavelet_function_from_filters(h::Vector{Float64}, g::Vector{Float64}, n_iterations::Int)

Reconstructs the mother wavelet function from its quadrature mirror filters using the cascade algorithm.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function is a core utility for the `Wavelet` component. While the `Wavelets.jl`
package provides the filter coefficients for various wavelet families, it does not
offer a direct method to evaluate the mother wavelet function \$\\psi(t)\$ on an
arbitrary grid. This function implements the cascade algorithm (also known as the
refining algorithm or pyramid algorithm) to numerically approximate \$\\psi(t)\$ from
the low-pass (`h`) and high-pass (`g`) filters.

# Mathematical Formulation
The algorithm starts with the scaling function \$\\phi_0(t)\$ as a box function
(\$1\$ on `[0,1)`, \$0\$ otherwise) and iteratively refines it using the two-scale relation:
\$\\phi_{j+1}(t) = \\sqrt{2} \\sum_k h_k \\phi_j(2t - k)\$
In each iteration, the wavelet function \$\\psi(t)\$ is computed from the current scaling
function:
\$\\psi_{j+1}(t) = \\sqrt{2} \\sum_k g_k \\phi_j(2t - k)\$
After `n_iterations`, the function returns the final approximation of \$\\psi(t)\$.

# Arguments
- `h::Vector{Float64}`: The low-pass filter coefficients (scaling function filter).
- `g::Vector{Float64}`: The high-pass filter coefficients (wavelet function filter).
- `n_iterations::Int`: The number of refinement iterations to perform.

# Returns
- A tuple `(x_grid_final, psi_next_vals)` where `x_grid_final` is the coordinate
  grid and `psi_next_vals` are the corresponding values of the wavelet function.
"""
function _reconstruct_wavelet_function_from_filters(h::Vector{Float64}, g::Vector{Float64}, n_iterations::Int)
    # Dynamically load Interpolations to ensure it's available in the execution scope.
    Interpolations = Base.require(Base.Main, :Interpolations)

    L = length(h) 
    x_min_support = 0.0
    x_max_support = L > 1 ? L - 1.0 : 1.0

    # Define a fine grid for the final reconstruction.
    num_points_final_grid = max(2, (2^n_iterations) * max(1, L - 1) + 1)
    x_grid_final = collect(range(x_min_support, stop=x_max_support, length=num_points_final_grid))

    # Initialize the scaling function phi_0 as a box function.
    phi_current_vals = zeros(length(x_grid_final))
    for i in eachindex(x_grid_final)
        if 0.0 <= x_grid_final[i] < 1.0; phi_current_vals[i] = 1.0; end
    end
    
    # Create an interpolant for the current scaling function.
    phi_itp = Interpolations.linear_interpolation(x_grid_final, phi_current_vals, extrapolation_bc=Interpolations.Flat())

    psi_next_vals = zeros(length(x_grid_final))

    # Iteratively refine the scaling and wavelet functions.
    for iter in 1:n_iterations
        phi_next_vals = zeros(length(x_grid_final))
        for idx in eachindex(x_grid_final)
            x_val = x_grid_final[idx]
            phi_sum = 0.0
            psi_sum = 0.0
            # Apply the two-scale relation using the filter coefficients.
            for k_filter in 0:(L-1)
                arg = 2.0 * x_val - k_filter
                phi_val_at_arg = phi_itp(arg)
                phi_sum += h[k_filter+1] * phi_val_at_arg
                psi_sum += g[k_filter+1] * phi_val_at_arg
            end
            phi_next_vals[idx] = sqrt(2.0) * phi_sum
            psi_next_vals[idx] = sqrt(2.0) * psi_sum
        end
        # Update the interpolant for the next iteration.
        phi_itp = Interpolations.linear_interpolation(x_grid_final, phi_next_vals, extrapolation_bc=Interpolations.Flat())
    end
    
    return x_grid_final, psi_next_vals
end

 

"""
    bstm_tensor_product_wavelet_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int}, family::Symbol, lengthscale::Union{Real, AbstractVector})

Generates a multi-dimensional wavelet basis matrix via a tensor product of 1D bases.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function provides a method for constructing high-dimensional basis functions
from their 1D counterparts. The tensor product is a standard mathematical operation
for achieving this. The implementation is correct and consistent with the refactor,
efficiently using broadcasting and reshaping to compute the product. This version
adds comprehensive documentation to clarify its purpose and usage.

# Mathematical Formulation
Given 1D basis matrices \$B_1, B_2, \\dots, B_D\$, the tensor product basis \$B\$ is
constructed such that each column of \$B\$ is the element-wise product of one column
from each of the 1D basis matrices. This is equivalent to the Kronecker product of
the rows of the 1D basis matrices.

# Arguments
- `coords::AbstractMatrix`: An `N x D` matrix of data points.
- `nbins_per_dim::Vector{Int}`: A vector specifying the number of basis functions for each dimension.
- `family::Symbol`: The wavelet family to use for the 1D bases.
- `lengthscale::Union{Real, AbstractVector}`: The lengthscale(s) for the wavelets.

# Returns
- A basis matrix of size `(N, prod(nbins_per_dim))`.
"""
function bstm_tensor_product_wavelet_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int}, family::Symbol, lengthscale::Union{Real, AbstractVector})
    n_dims = size(coords, 2)
    if length(nbins_per_dim) != n_dims; error("Length of `nbins_per_dim` must match coordinate dimensions."); end
    
    ls_vec = if lengthscale isa Real
        fill(Float64(lengthscale), n_dims)
    else
        if length(lengthscale) != n_dims; error("Length of lengthscale vector must match coordinate dimensions."); end
        lengthscale
    end

    # Generate a 1D wavelet basis matrix for each dimension.
    basis_matrices_1D = [bstm_wavelet_basis_1D(coords[:, i], nbins_per_dim[i], family, ls_vec[i]) for i in 1:n_dims]
    
    if isempty(basis_matrices_1D); return zeros(size(coords, 1), 0); end

    # Initialize the final basis with the matrix from the first dimension.
    B_final = basis_matrices_1D[1]

    # Iteratively compute the tensor product with the remaining basis matrices.
    for i in 2:n_dims
        B_next = basis_matrices_1D[i]
        n_obs, n_cols_final = size(B_final)
        _, n_cols_next = size(B_next)
        
        # Reshape for broadcasting to compute row-wise outer products.
        B_final_reshaped = reshape(B_final, n_obs, n_cols_final, 1)
        B_next_reshaped = reshape(B_next, n_obs, 1, n_cols_next)
        
        # The element-wise product creates the tensor product of the rows.
        tensor_prod = B_final_reshaped .* B_next_reshaped
        
        # Reshape the result into the final 2D basis matrix.
        B_final = reshape(tensor_prod, n_obs, n_cols_final * n_cols_next)
    end
    
    return B_final
end


 

"""
    bstm_smooth_basis_1D(type::String, vals::AbstractVector, nbins::Int, degree::Int; ...)

Generates a 1D basis matrix. This version is CPU-only.
"""
function bstm_smooth_basis_1D(
    type::String, 
    vals::AbstractVector, 
    nbins::Int, 
    degree::Int; 
    W=nothing, 
    knot_method::Symbol = :quantile, 
    custom_knots::Union{AbstractVector, Nothing} = nothing, 
    kwargs...
)
    n_obs = length(vals)
    
    v_min = minimum(vals)
    v_max = maximum(vals)
    v_std = std(vals) + 1e-9
    use_regular_grid = type in ["invdist", "kriging", "tps", "gp"]

    knots = if knot_method == :custom && !isnothing(custom_knots)
        custom_knots
    elseif knot_method == :range || use_regular_grid
        collect(range(v_min, stop=v_max, length=nbins))
    else
        quantile(vals, range(0, 1, length=nbins))
    end

    if type in ["pspline", "bspline"]
        return bstm_bspline_basis(vals, nbins, degree; knot_method=knot_method, custom_knots=custom_knots)
    end

    B_out = Matrix{Float64}(undef, n_obs, nbins); fill!(B_out, 0.0)
    actual_nbins_generated = nbins

    if type in ["smooth", "barycentric", "linear"]
        h = (v_max - v_min) / (nbins > 1 ? (nbins - 1) : 1.0)
        h = h > 0 ? h : 1.0
        for m in 1:nbins
            dist = abs.(vals .- knots[m]) ./ h
            mask = dist .< 1.0
            B_out[mask, m] .= 1.0 .- dist[mask]
        end
    elseif type == "tps"
        for m in 1:nbins
            r = abs.(vals .- knots[m])
            B_out[:, m] .= r.^3
        end
    elseif type == "rff"
        ls = get(kwargs, :lengthscale, v_std)
        Omega = randn(1, nbins) ./ ls
        Phi_phases = rand(nbins) .* (2.0 * pi)
        B_out .= sqrt(2.0 / nbins) .* cos.((vals * Omega) .+ Phi_phases')
    elseif type == "fft"
        ls = get(kwargs, :lengthscale, v_std)
        t_coords = vals ./ ls
        idx = 1
        for m in 1:div(nbins, 2)
            arg = m .* t_coords
            B_out[:, idx] = sin.(2.0 * pi * arg)
            B_out[:, idx+1] = cos.(2.0 * pi * arg)
            idx += 2
        end
        if isodd(nbins) && idx <= nbins
            m = div(nbins, 2) + 1
            arg = m .* t_coords
            B_out[:, idx] = sin.(2.0 * pi * arg)
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
    else
        B_out = ones(Float64, n_obs, 1)
        actual_nbins_generated = 1
    end

    return B_out, actual_nbins_generated
end


"""
    bstm_smooth_basis_2D(type::String, coords::AbstractMatrix, nbins::Union{Int, Vector{Int}}; ...)

Generates a 2D basis matrix. This version is CPU-only.
"""
function bstm_smooth_basis_2D(
    type::String, 
    coords::AbstractMatrix, 
    nbins::Union{Int, Vector{Int}}; 
    W=nothing, 
    knot_method::Symbol = :quantile, 
    custom_knots::Union{Tuple{AbstractVector, AbstractVector}, Nothing} = nothing, 
    kwargs...
)
    n_obs = size(coords, 1)
    
    n_marginal_x, n_marginal_y = if nbins isa Int
        (nbins, nbins)
    elseif nbins isa Vector{Int} && length(nbins) == 2
        (nbins[1], nbins[2])
    else
        error("For a 2D smooth, `nbins` must be an Int or a Vector{Int} of length 2.")
    end
    total_bins = n_marginal_x * n_marginal_y

    c_min = [minimum(coords[:, 1]), minimum(coords[:, 2])]
    c_max = [maximum(coords[:, 1]), maximum(coords[:, 2])]
    c_std = [std(coords[:, 1]), std(coords[:, 2])] .+ 1e-9

    ls_x = get(kwargs, :ls_x, c_std[1])
    ls_y = get(kwargs, :ls_y, c_std[2])

    use_regular_grid = type in ["invdist", "kriging", "tps", "spherical"]

    kx, ky = if knot_method == :custom && !isnothing(custom_knots)
        custom_knots
    elseif knot_method == :quantile && !use_regular_grid
        (quantile(coords[:, 1], range(0, 1, length=n_marginal_x)),
         quantile(coords[:, 2], range(0, 1, length=n_marginal_y)))
    else
        (collect(range(c_min[1], stop=c_max[1], length=n_marginal_x)),
         collect(range(c_min[2], stop=c_max[2], length=n_marginal_y)))
    end
    
    B = Matrix{Float64}(undef, n_obs, total_bins); fill!(B, 0.0)

    if type == "barycentric"
        knot_points = [Point2D(kx[i], ky[j]) for j in 1:n_marginal_y for i in 1:n_marginal_x]
        B = bstm_barycentric_basis_2D(coords, knot_points)
    elseif type in ["pspline", "bspline"]
        degree_val = get(kwargs, :degree, 3)
        B = bstm_tensor_product_basis(coords, [n_marginal_x, n_marginal_y], [degree_val, degree_val]; knot_method=knot_method, kwargs...)
    elseif type == "wavelet"
        family = get(kwargs, :family, :db4)
        lengthscale = get(kwargs, :lengthscale, 0.1)
        B = bstm_tensor_product_wavelet_basis(coords, [n_marginal_x, n_marginal_y], family, lengthscale)
    elseif type in ["smooth", "linear"]
        hx = (c_max[1] - c_min[1]) / (n_marginal_x > 1 ? (n_marginal_x - 1) : 1.0); hx = hx > 0 ? hx : 1.0
        hy = (c_max[2] - c_min[2]) / (n_marginal_y > 1 ? (n_marginal_y - 1) : 1.0); hy = hy > 0 ? hy : 1.0
        idx = 1
        for j in 1:n_marginal_y, i in 1:n_marginal_x
            if idx > total_bins; break; end
            b_x = max.(0.0, 1.0 .- abs.(coords[:, 1] .- kx[i]) ./ hx)
            b_y = max.(0.0, 1.0 .- abs.(coords[:, 2] .- ky[j]) ./ hy)
            B[:, idx] .= b_x .* b_y
            idx += 1
        end
    elseif type == "tps"
        centers = [(x, y) for y in ky for x in kx][:]
        for m in 1:total_bins
            dx = coords[:, 1] .- centers[m][1]
            dy = coords[:, 2] .- centers[m][2]
            r = sqrt.(dx.^2 .+ dy.^2)
            B[:, m] .= (r.^2) .* log.(r .+ 1e-9)
        end
    elseif type == "rff" || type == "anisotropic"
        Omega = randn(2, total_bins)
        Omega[1, :] ./= ls_x
        Omega[2, :] ./= ls_y
        Phi_phases = rand(total_bins) .* (2.0 * pi)
        B .= sqrt(2.0 / total_bins) .* cos.((coords * Omega) .+ Phi_phases')
    else
        @warn "Basis type '$type' not recognized for 2D smooth. Returning an empty basis matrix."
        return Matrix{Float64}(undef, n_obs, 0)
    end

    return B[:, 1:min(total_bins, size(B, 2))]
end


"""
    bstm_smooth_basis_3D(type::String, coords::AbstractMatrix, nbins::Union{Int, Vector{Int}}; ...)

Generates a 3D basis matrix. This version is CPU-only.
"""
function bstm_smooth_basis_3D(
    type::String, 
    coords::AbstractMatrix, 
    nbins::Union{Int, Vector{Int}}; 
    W=nothing, 
    knot_method::Symbol = :quantile, 
    custom_knots::Union{Tuple{AbstractVector, AbstractVector, AbstractVector}, Nothing} = nothing, 
    kwargs...
)
    n_obs = size(coords, 1)

    n_marginal_x, n_marginal_y, n_marginal_z = if nbins isa Int
        (nbins, nbins, nbins)
    elseif nbins isa Vector{Int} && length(nbins) == 3
        (nbins[1], nbins[2], nbins[3])
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

    kx, ky, kz = if knot_method == :custom && !isnothing(custom_knots)
        custom_knots
    elseif knot_method == :quantile
        (quantile(coords[:, 1], range(0, 1, length=n_marginal_x)),
         quantile(coords[:, 2], range(0, 1, length=n_marginal_y)),
         quantile(coords[:, 3], range(0, 1, length=n_marginal_z)))
    else
        (collect(range(c_min[1], stop=c_max[1], length=n_marginal_x)),
         collect(range(c_min[2], stop=c_max[2], length=n_marginal_y)),
         collect(range(c_min[3], stop=c_max[3], length=n_marginal_z)))
    end

    B = Matrix{Float64}(undef, n_obs, total_bins); fill!(B, 0.0)

    if type in ["pspline", "bspline"]
        degree_val = get(kwargs, :degree, 3)
        B = bstm_tensor_product_basis(coords, [n_marginal_x, n_marginal_y, n_marginal_z], fill(degree_val, 3); knot_method=knot_method, kwargs...)
    elseif type == "wavelet"
        family = get(kwargs, :family, :db4)
        lengthscale = get(kwargs, :lengthscale, 0.1)
        B = bstm_tensor_product_wavelet_basis(coords, [n_marginal_x, n_marginal_y, n_marginal_z], family, lengthscale)
    elseif type in ["smooth", "barycentric", "linear"]
        hx = (c_max[1] - c_min[1]) / (n_marginal_x > 1 ? (n_marginal_x - 1) : 1.0); hx = hx > 0 ? hx : 1.0
        hy = (c_max[2] - c_min[2]) / (n_marginal_y > 1 ? (n_marginal_y - 1) : 1.0); hy = hy > 0 ? hy : 1.0
        hz = (c_max[3] - c_min[3]) / (n_marginal_z > 1 ? (n_marginal_z - 1) : 1.0); hz = hz > 0 ? hz : 1.0

        idx = 1
        for k_idx in 1:n_marginal_z, j_idx in 1:n_marginal_y, i_idx in 1:n_marginal_x
            if idx > total_bins; break; end
            b_x = max.(0.0, 1.0 .- abs.(coords[:, 1] .- kx[i_idx]) ./ hx)
            b_y = max.(0.0, 1.0 .- abs.(coords[:, 2] .- ky[j_idx]) ./ hy)
            b_z = max.(0.0, 1.0 .- abs.(coords[:, 3] .- kz[k_idx]) ./ hz)
            B[:, idx] .= b_x .* b_y .* b_z
            idx += 1
        end
    elseif type == "rff"
        Omega = randn(3, total_bins)
        Omega[1, :] ./= ls_x; Omega[2, :] ./= ls_y; Omega[3, :] ./= ls_z
        Phi_phases = rand(total_bins) .* (2.0 * pi)
        B .= sqrt(2.0 / total_bins) .* cos.((coords * Omega) .+ Phi_phases')
    else
        B = ones(Float64, n_obs, total_bins)
    end

    return B[:, 1:min(total_bins, size(B, 2))]
end


"""
    bstm_smooth_basis_4D(type::String, coords::AbstractMatrix, nbins::Union{Int, Vector{Int}}; ...)

Generates a 4D basis matrix. This version is CPU-only.
"""
function bstm_smooth_basis_4D(
    type::String, 
    coords::AbstractMatrix, 
    nbins::Union{Int, Vector{Int}}; 
    W=nothing, 
    knot_method::Symbol = :quantile, 
    custom_knots::Union{Tuple{AbstractVector, AbstractVector, AbstractVector, AbstractVector}, Nothing} = nothing, 
    kwargs...
)
    n_obs = size(coords, 1)

    n_marginal_1, n_marginal_2, n_marginal_3, n_marginal_4 = if nbins isa Int
        (nbins, nbins, nbins, nbins)
    elseif nbins isa Vector{Int} && length(nbins) == 4
        (nbins[1], nbins[2], nbins[3], nbins[4])
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

    k1, k2, k3, k4 = if knot_method == :custom && !isnothing(custom_knots)
        custom_knots
    elseif knot_method == :quantile
        (quantile(coords[:, 1], range(0, 1, length=n_marginal_1)),
         quantile(coords[:, 2], range(0, 1, length=n_marginal_2)),
         quantile(coords[:, 3], range(0, 1, length=n_marginal_3)),
         quantile(coords[:, 4], range(0, 1, length=n_marginal_4)))
    else
        (collect(range(c_min[1], stop=c_max[1], length=n_marginal_1)),
         collect(range(c_min[2], stop=c_max[2], length=n_marginal_2)),
         collect(range(c_min[3], stop=c_max[3], length=n_marginal_3)),
         collect(range(c_min[4], stop=c_max[4], length=n_marginal_4)))
    end

    B = Matrix{Float64}(undef, n_obs, total_bins); fill!(B, 0.0)

    if type in ["pspline", "bspline"]
        degree_val = get(kwargs, :degree, 3)
        B = bstm_tensor_product_basis(coords, [n_marginal_1, n_marginal_2, n_marginal_3, n_marginal_4], fill(degree_val, 4); knot_method=knot_method, kwargs...)
    elseif type == "wavelet"
        family = get(kwargs, :family, :db4)
        lengthscale = get(kwargs, :lengthscale, 0.1)
        B = bstm_tensor_product_wavelet_basis(coords, [n_marginal_1, n_marginal_2, n_marginal_3, n_marginal_4], family, lengthscale)
    elseif type in ["smooth", "linear", "barycentric"]
        hx1 = (c_max[1] - c_min[1]) / (n_marginal_1 > 1 ? (n_marginal_1 - 1) : 1.0); hx1 = hx1 > 0 ? hx1 : 1.0
        hx2 = (c_max[2] - c_min[2]) / (n_marginal_2 > 1 ? (n_marginal_2 - 1) : 1.0); hx2 = hx2 > 0 ? hx2 : 1.0
        hx3 = (c_max[3] - c_min[3]) / (n_marginal_3 > 1 ? (n_marginal_3 - 1) : 1.0); hx3 = hx3 > 0 ? hx3 : 1.0
        hx4 = (c_max[4] - c_min[4]) / (n_marginal_4 > 1 ? (n_marginal_4 - 1) : 1.0); hx4 = hx4 > 0 ? hx4 : 1.0

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
    elseif type == "rff"
        Omega = randn(4, total_bins)
        Omega[1, :] ./= ls_1; Omega[2, :] ./= ls_2; Omega[3, :] ./= ls_3; Omega[4, :] ./= ls_4
        Phi_phases = rand(total_bins) .* (2.0 * pi)
        B .= sqrt(2.0 / total_bins) .* cos.((coords * Omega) .+ Phi_phases')
    else
        B = ones(Float64, n_obs, total_bins)
    end

    return B[:, 1:min(total_bins, size(B, 2))]
end


"""
    evaluate_kernel_matrix(coords::AbstractMatrix, param_val::Real, ls::Union{Real, AbstractVector}, kernel_type::Symbol, noise::Real; wavelet_levels=3)

Computes the covariance kernel matrix for a given set of coordinates.

# Version
v1.5.7 (2026-08-13)

# Rationale
This function is a core utility for all GP-based components. It acts as a factory
to generate dense covariance matrices for various kernel functions. This version is
updated to include the `:exponential` (or `:matern12`) kernel and fixes a bug in
the `:wavelet` kernel's handling of anisotropic lengthscales. The documentation has
been expanded to include mathematical formulations for clarity.

# Arguments
- `coords::AbstractMatrix`: An `N x D` matrix of data points, where `N` is the
  number of points and `D` is the number of dimensions.
- `param_val::Real`: The signal variance (\$\\sigma^2\$) of the kernel. This controls
  the overall amplitude of the function.
- `ls::Union{Real, AbstractVector}`: The lengthscale(s) (\$\\ell\$) of the kernel.
  Controls the "wiggliness" or correlation distance. A `Real` value assumes an
  isotropic kernel, while a `Vector` of length `D` enables Automatic Relevance
  Determination (ARD) with a separate lengthscale for each dimension.
- `kernel_type::Symbol`: The type of kernel to evaluate.
- `noise::Real`: A small jitter or "nugget" term added to the diagonal for
  numerical stability, representing observation noise.
- `wavelet_levels`: The number of levels for the wavelet kernel.

# Supported Kernels
- `:gaussian`, `:se`, `:rbf`: Squared Exponential kernel.
  \$k(x, x') = \\sigma^2 \\exp\\left(-\\frac{\\|x - x'\\|^2}{2\\ell^2}\\right)\$
- `:exponential`, `:matern12`: Exponential kernel (Matérn with \$\\nu=1/2\$).
  \$k(x, x') = \\sigma^2 \\exp\\left(-\\frac{\\|x - x'\\|}{\\ell}\\right)\$
- `:matern32`: Matérn kernel with \$\\nu=3/2\$.
- `:matern52`: Matérn kernel with \$\\nu=5/2\$.
- `:spherical`: Spherical kernel, which is compactly supported (zero beyond range \$\\ell\$).
- `:cosine`: Cosine kernel for periodic functions.
- `:linear`: Linear kernel.
- `:constant`: Constant kernel.
- `:wavelet`: A multi-scale kernel constructed from a sum of SE kernels.

# Returns
- A dense `N x N` covariance matrix.
 
Computes the covariance kernel matrix for a given set of coordinates 
"""
function evaluate_kernel_matrix(coords::AbstractMatrix, param_val::Real, ls::Union{Real, AbstractVector}, kernel_type::Symbol, noise::Real; wavelet_levels=3)
    T = promote_type(eltype(coords), typeof(param_val), eltype(ls), typeof(noise))
    coords_T = convert(AbstractMatrix{T}, coords)
    ls_T = convert(typeof(ls) <: Real ? T : AbstractVector{T}, ls)
    N = size(coords_T, 1)

    if kernel_type == :linear
        return param_val^2 .* (coords_T * coords_T') .+ (noise * I)
    end

    function _sqeuclidean_broadcast(X::AbstractMatrix)
        sum(X.^2, dims=2) .- 2 * (X * X') .+ sum(X.^2, dims=2)'
    end

    local dist_sq
    if ls isa AbstractVector # ARD case
        if size(coords_T, 2) != length(ls_T)
            error("Dimension mismatch for ARD kernel: Number of coordinate dimensions ($(size(coords_T, 2))) does not match number of lengthscales ($(length(ls_T))).")
        end
        dist_sq = _sqeuclidean_broadcast(coords_T ./ ls_T')
    else # Isotropic case
        dist_sq = _sqeuclidean_broadcast(coords_T) ./ ls_T^2
    end
    
    dist_sq[diagind(dist_sq)] .= max.(0, diag(dist_sq))

    if kernel_type == :gaussian || kernel_type == :se || kernel_type == :rbf
        return param_val^2 .* exp.(-one(T)/2 .* dist_sq) .+ (noise * I)
    
    elseif kernel_type == :exponential || kernel_type == :matern12
        d = sqrt.(dist_sq)
        return param_val^2 .* exp.(-d) .+ (noise * I)
    
    elseif kernel_type == :matern32
        d = sqrt.(dist_sq)
        val = sqrt(convert(T, 3.0)) .* d
        return param_val^2 .* (one(T) .+ val) .* exp.(-val) .+ (noise * I)
    
    elseif kernel_type == :matern52
        d = sqrt.(dist_sq)
        val = sqrt(convert(T, 5.0)) .* d
        return param_val^2 .* (one(T) .+ val .+ (val.^2 ./ convert(T, 3.0))) .* exp.(-val) .+ (noise * I)

    elseif kernel_type == :spherical
        d = sqrt.(dist_sq)
        K = zeros(T, size(d))
        mask = d .< one(T)
        K[mask] = param_val^2 .* (one(T) .- 1.5 .* d[mask] .+ 0.5 .* d[mask].^3)
        return K .+ (noise * I)

    elseif kernel_type == :cosine
        if ls isa AbstractVector
            @warn "Cosine kernel with ARD lengthscale is not standard. Using the first lengthscale for an isotropic kernel."
            ls_T = ls_T[1]
        end
        d_euclidean = sqrt.(_sqeuclidean_broadcast(coords_T))
        return param_val^2 .* cos.(2.0 * pi .* d_euclidean ./ ls_T) .+ (noise * I)

    elseif kernel_type == :constant
        return fill(convert(T, param_val^2), size(dist_sq)) .+ (noise * I)

    elseif kernel_type == :wavelet
        local_ls = ls isa Real ? ls_T : ls_T[1]
        if ls isa AbstractVector
            @warn "Wavelet kernel with ARD lengthscale is not standard. Using the first lengthscale for decay."
        end
        K_accum = zeros(T, size(dist_sq))
        for wv_scale in 1:wavelet_levels
            ls_scale_sq = (ls isa Real ? ls_T^2 : one(T)) / (convert(T, 4.0)^(wv_scale-1))
            weight_scale = param_val^2 * exp(convert(T, -wv_scale) / local_ls)
            K_accum .+= weight_scale .* exp.(-one(T)/2 .* dist_sq ./ ls_scale_sq)
        end
        return K_accum + (noise * I)

    else
        @warn "Kernel '$(kernel_type)' not explicitly handled in evaluate_kernel_matrix. Defaulting to Squared Exponential."
        return param_val^2 .* exp.(-one(T)/2 .* dist_sq) .+ (noise * I)
    end
end




"""
    recompose_precision(m_type::Symbol, template_s::AbstractMatrix, param_val::Real; ...)

Constructs a final precision matrix from a template and sampled hyperparameters.
This version is CPU-only.
"""
function recompose_precision(m_type::Symbol, template_s::AbstractMatrix, param_val::Real; extra_param=nothing, noise=1e-4, kwargs...)
    n_s = size(template_s, 1)
    T_num = promote_type(typeof(param_val), typeof(noise), eltype(template_s), typeof(extra_param))

    if m_type == :SPDE
        kappa = isnothing(extra_param) ? one(T_num) : extra_param
        Q_kappa = if kappa isa Real
            kappa^2 * I
        else
            if length(kappa) != n_s; error("Anisotropic kappa vector length must match number of spatial units."); end
            Diagonal(kappa.^2)
        end
        L_spde = Q_kappa + template_s
        return Symmetric(L_spde' * L_spde)
    end

    if m_type == :None || m_type == :FIXED
        return Symmetric(sparse(I, n_s, n_s))
    end

    if m_type == :Besag || m_type == :ICAR || m_type == :Cyclic
        return Symmetric(template_s)
    end

    if m_type == :AR1
        rho = isnothing(extra_param) ? zero(T_num) : extra_param
        Q = (one(T_num) + rho^2) * I + rho .* template_s
        if n_s > 0
            Q[1, 1] = one(T_num)
            Q[n_s, n_s] = one(T_num)
        end
        return Symmetric(Q)
    end

    if m_type == :Leroux || m_type == :LocalAdaptive
        lambda_val = isnothing(extra_param) ? convert(T_num, 0.5) : extra_param
        I_prom = I
        return Symmetric(lambda_val .* template_s + (one(T_num) - lambda_val) .* I_prom)
    end

    if m_type == :NetworkFlow
        rho_net = isnothing(extra_param) ? convert(T_num, 0.8) : extra_param
        W_net = template_s
        flow_direction = get(kwargs, :flow_direction, :bidirectional)
        
        L_op = if flow_direction == :upstream
            I - rho_net .* W_net'
        elseif flow_direction == :downstream
            I - rho_net .* W_net
        else
            W_symm = (W_net + W_net') ./ 2
            I - rho_net .* W_symm
        end
        return Symmetric(L_op' * L_op)
    end

    if m_type == :SAR || m_type == :DAG
        rho_p = isnothing(extra_param) ? convert(T_num, 0.8) : extra_param
        L_op = I - rho_p .* template_s
        return Symmetric(L_op' * L_op)
    end

    if m_type == :GP
        ls = isnothing(extra_param) ? one(T_num) : extra_param
        K = param_val^2 .* exp.(-(template_s) ./ (convert(T_num, 2.0) * ls^2))
        return inv(Symmetric(K + (noise * I)))
    end

    if m_type in [:RFF, :FFT, :BSpline, :PSpline, :TPS]
        return Symmetric(template_s)
    end

    return Symmetric(template_s)
end






"""
    _distribution_to_string(d::Distribution)

Converts a `Distribution` object into a type-stable string representation of its
constructor call, suitable for dynamic code generation within a Turing `@model`.

# Rationale
This function is a core utility for the `bstm` code generation engine. It ensures
that prior distributions defined as Julia objects can be correctly translated into
AD-compatible code.

Key features of this implementation:
1.  **AD Compatibility**: It generates constructor strings like `Normal{T}(...)`, where
    `T` is the generic numeric type used by the Turing model. This allows the model
    to work seamlessly with automatic differentiation libraries like `ForwardDiff.jl`.
2.  **Robustness**: It uses accessor functions (`meanlog`, `stdlog`, `shape`, `scale`)
    where possible, making the code less dependent on the internal structure of
    `Distribution` objects.
3.  **Completeness**: It handles a wide range of univariate and multivariate
    distributions, as well as wrapper types like `Truncated`, `Product`, and `Fill`.
4.  **Efficiency**: For `Product` distributions containing identical inner distributions,
    it generates a more efficient `filldist` call.

# Version
v1.1.0 (2026-08-13)

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
        return "$(dist_name){T}($(meanlog(d)), $(stdlog(d)))"
    elseif d isa Beta
        # Access alpha and beta parameters directly for Beta distribution
        return "$(dist_name){T}($(d.α), $(d.β))"
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
    elseif d isa Categorical
        return "$(dist_name)($(d.p))" # Probabilities are fixed, no need for {T}
    elseif d isa LKJCholesky
        # Essential for multivariate models (e.g., mixed effects)
        return "$(dist_name){T}($(d.d), $(d.eta))"
    elseif d isa MvNormal
        # Handle common cases for MvNormal, especially with UniformScaling
        mean_str = string(mean(d))
        cov_str = if d.Σ isa UniformScaling
            "$(d.Σ.λ) * I"
        else
            string(d.Σ) # Fallback for other matrix types
        end
        return "$(dist_name)(T.($(mean_str)), $(cov_str))"
    elseif d isa Truncated
        inner_dist_str = _distribution_to_string(d.untruncated)
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
    get_optimal_sampler(model_obj::DynamicPPL.Model; ...)

Constructs an efficient composite MCMC sampler for a `bstm` model by assigning
specialized samplers to different parameter blocks.

# Version
v1.5.1 (2026-08-20)

# Rationale
This function is a core utility for the `bstm` inference engine. It replaces a
simple, one-size-fits-all sampler selection with a sophisticated, multi-stage
process that builds a composite `Gibbs` sampler tailored to the model's structure.

This version has been rewritten to use the modern, public API of `DynamicPPL.jl`,
resolving `MethodError` and deprecation warnings from previous versions. The key
corrections are:
1.  **Correct Parameter Name Matching**: Uses `vn.name` to match parameter symbols
    (e.g., in `sampler_map`) and `string(vn)` for string-based matching (e.g., for
    component grouping), replacing the incorrect use of `vn.optic`.
2.  **Public API for `VarInfo`**: Uses `keys(vi)` and `get(vi, vn)` which are part of
    the public API, ensuring forward compatibility.
3.  **Robust Type Inference**: Uses `eltype(get(vi, vn))` for determining parameter
    types, which correctly handles the underlying data structures of `VarInfo`.
4.  **Generalization**: The logic for categorizing parameters has been made more
    general and less reliant on internal implementation details of `DynamicPPL`.
5.  **Dense Mass Matrix Adaptation**: Introduces an option to use `DenseEuclideanMetric`
    for `NUTS` samplers assigned to model component groups, where parameters are
    often highly correlated, leading to more efficient exploration of the posterior.
6.  **Refined AD Backend Selection**: The heuristic for selecting the AD backend
    (e.g., `AutoForwardDiff` vs. `AutoEnzyme`) is now applied at the block level,
    allowing for more granular optimization based on the size of each parameter block.

# Workflow
The function constructs the sampler in a series of stages, with each stage having
higher precedence than the next:
1.  **User-Specified Sampler**: If a complete sampler is passed via `sampler_choice`,
    it is used directly, overriding all other logic.
2.  **Automatic AD Engine Selection**: If the user has not specified a non-default
    `adtype`, the function inspects the model's size. For large models (default > 100
    parameters), it automatically switches the primary AD engine to a more performant
    one like `ADTypes.AutoEnzyme()`.
3.  **Manual Sampler Map**: The user can provide a `sampler_map` to assign specific
    samplers to specific parameter groups.
4.  **Component Grouping**: If `group_components=true`, the function identifies all
    parameters belonging to the same model component and assigns them to a joint `NUTS` sampler.
    For these blocks, a `DenseEuclideanMetric` can be optionally used.
5.  **Default Categorization**: Any remaining parameters are categorized by their support,
    inferred from their value type and transform strategy:
    - **Discrete**: Inferred if the parameter's value is an `Integer`. Assigned to `PG`.
    - **Bounded Continuous**: Inferred if the parameter's transform is `Exp` or `Logistic`. Assigned to `Slice`.
    - **Unbounded Continuous**: All other continuous parameters. Assigned to `NUTS`.

# Arguments
- `model_obj::DynamicPPL.Model`: The instantiated Turing model.
- `sampler_choice`: A specific sampler instance to use, or `:auto` for automatic selection.
- `sampler_map`: A `Dict` mapping parameter `Symbol`s to specific samplers.
- `adtype`: The automatic differentiation backend to use for gradient-based samplers.
- `group_components`: A `Bool` indicating whether to group component parameters for joint sampling.
- `use_dense_metric_for_components::Bool`: If `true`, `DenseEuclideanMetric` is used for `NUTS` samplers assigned to component groups. Otherwise, `DiagEuclideanMetric` is used.
- `adaptation_steps`: The number of adaptation steps for HMC-based samplers.
- `target_acceptance`: The target acceptance rate for HMC-based samplers.
- `n_particles`: The number of particles for the `PG` sampler.
- `n_chains::Int`: The number of chains to be run. This parameter is primarily used by the `sample` function and does not directly affect the construction of the `Gibbs` or `NUTS` samplers themselves.

# Returns
- An `AbstractMCMC.AbstractSampler` object.
"""
function get_optimal_sampler(
    model_obj::DynamicPPL.Model;
    sampler_choice::Union{Symbol, AbstractMCMC.AbstractSampler}=:auto,
    sampler_map::Dict{Symbol, <:AbstractMCMC.AbstractSampler}=Dict{Symbol, AbstractMCMC.AbstractSampler}(),
    adtype::ADTypes.AbstractADType=ADTypes.AutoForwardDiff(),
    group_components::Bool=true,
    use_dense_metric_for_components::Bool=true, # New argument for dense mass matrix
    adaptation_steps::Int=500,
    target_acceptance::Float64=0.8,
    n_particles::Int=20,
    n_chains::Int=1 # Kept for consistency, but primarily for `sample` call
)
    # --- Stage 0: Handle direct sampler choice ---
    # If a specific sampler instance is provided, use it directly.
    if sampler_choice isa AbstractMCMC.AbstractSampler
        return sampler_choice
    end

    # --- Initialize VarInfo and determine global AD backend ---
    # Instantiate and populate the VarInfo object to get variable names and types.
    vi = DynamicPPL.VarInfo(model_obj)
    vns = keys(vi)
    num_params = length(vns)

    # Determine the global AD backend to use based on model size.
    # For large models, Enzyme is often more performant.
    adtype_to_use = adtype
    param_threshold = 100
    if adtype isa ADTypes.AutoForwardDiff && num_params > param_threshold
        adtype_to_use = ADTypes.AutoEnzyme()
        @info "Model has > $(param_threshold) parameters. Switching global AD backend to Enzyme for performance."
    end

    # Initialize lists to store sampler assignments and track processed variables.
    sampler_assignments = []
    all_processed_vns = Set{VarName}()

    # --- Stage 1: Handle user-provided sampler map (highest precedence) ---
    # Iterate through the user's sampler map and assign samplers to specified parameters.
    for (param_sym, sampler) in sampler_map
        # Match VarNames by their symbolic name (vn.name).
        sym_vns = filter(vn -> vn.name == param_sym, vns)
        if !isempty(sym_vns)
            push!(sampler_assignments, Tuple(sym_vns) => sampler)
            union!(all_processed_vns, sym_vns)
        else
            @warn "Parameter :$(param_sym) in sampler_map not found in model."
        end
    end

    # --- Stage 2: Group parameters by model component if enabled ---
    # Identify parameters belonging to the same model component and group them for joint sampling.
    if group_components && hasproperty(model_obj.args.M, :components)
        # Get component keys from the model configuration.
        component_keys = string.([spec.key for spec in model_obj.args.M.components])
        # Sort by length in reverse to prioritize longer, more specific keys (e.g., "spatial_key" before "key").
        sort!(component_keys, by=length, rev=true)

        component_groups = Dict{String, Set{VarName}}()
        # Only consider variables not yet processed by the user's sampler map.
        vns_to_check = setdiff(vns, all_processed_vns)

        for vn in vns_to_check
            # Use `string(vn)` for matching, which produces names like `param[1]`.
            # The symbol is `vn.name`.
            vn_str = string(vn)
            found_key = nothing
            for key in component_keys
                # Match if the variable name ends with `_key` or `_key_d` (for indexed parameters).
                # Or if it starts with `key_` (for parameters directly named after the component).
                if occursin(Regex("_$(key)(_\\d+)?\$"), vn_str) || startswith(vn_str, "$(key)_")
                    found_key = key
                    break
                end
            end
            
            if !isnothing(found_key)
                if !haskey(component_groups, found_key)
                    component_groups[found_key] = Set{VarName}()
                end
                push!(component_groups[found_key], vn)
            end
        end

        for (key, params_vns) in component_groups
            # Ensure we don't re-process variables already handled by the user map.
            params_to_process = setdiff(params_vns, all_processed_vns)
            if isempty(params_to_process); continue; end

            # Refined AD backend selection for this specific block.
            # Use ForwardDiff for very small blocks, otherwise the global AD backend.
            block_adtype = if length(params_to_process) <= 10
                ADTypes.AutoForwardDiff()
            else
                adtype_to_use
            end

            # Use DenseEuclideanMetric for component groups if enabled, as parameters
            # within a component are often highly correlated.
            metric_type = use_dense_metric_for_components ? DenseEuclideanMetric() : DiagEuclideanMetric()
            sampler = NUTS(adaptation_steps, target_acceptance; adtype=block_adtype, metric=metric_type)
            push!(sampler_assignments, Tuple(params_to_process) => sampler)
            union!(all_processed_vns, params_to_process)
        end
    end

    # --- Stage 3: Assign samplers to remaining parameters based on their support ---
    # Categorize any remaining parameters (not yet assigned) by their type and support.
    remaining_vns = setdiff(vns, all_processed_vns)
    if !isempty(remaining_vns)
        param_groups = Dict(
            :discrete => Set{VarName}(), 
            :bounded => Set{VarName}(), 
            :other_continuous => Set{VarName}()
        )

        for vn in remaining_vns
            try
                # Check for discrete parameters (e.g., integer types).
                if eltype(get(vi, vn)) <: Integer
                    push!(param_groups[:discrete], vn)
                    continue
                end

                # Check for bounded parameters by inspecting their transform strategy.
                # This is the most reliable way to infer support from DynamicPPL.
                if haskey(vi.transform_strategy, vn)
                    transform = vi.transform_strategy[vn]
                    if transform isa Bijectors.Exp || transform isa Bijectors.Logistic
                        push!(param_groups[:bounded], vn)
                    else 
                        # If a transform exists but is not Exp or Logistic, assume unbounded continuous.
                        push!(param_groups[:other_continuous], vn)
                    end
                else
                    # If no specific transform, assume unbounded continuous.
                    push!(param_groups[:other_continuous], vn)
                end
            catch e
                @warn "Could not categorize parameter $(vn). Defaulting to NUTS. Error: $e"
                push!(param_groups[:other_continuous], vn)
            end
        end

        # Assign PG sampler to discrete parameters.
        if !isempty(param_groups[:discrete])
            push!(sampler_assignments, Tuple(param_groups[:discrete]) => PG(n_particles))
        end

        # Assign Slice sampler to bounded continuous parameters.
        if !isempty(param_groups[:bounded])
            push!(sampler_assignments, Tuple(param_groups[:bounded]) => Slice())
        end

        # Assign NUTS sampler to other unbounded continuous parameters.
        if !isempty(param_groups[:other_continuous])
            params = Tuple(param_groups[:other_continuous])
            
            # Refined AD backend selection for this block.
            block_adtype = if length(params) <= 10
                ADTypes.AutoForwardDiff()
            else
                adtype_to_use
            end

            # Default to DiagEuclideanMetric for general continuous blocks.
            push!(sampler_assignments, 
                params => NUTS(adaptation_steps, target_acceptance; adtype=block_adtype, metric=DiagEuclideanMetric())
            )
        end
    end

    # --- Stage 4: Construct and return the final composite sampler ---
    # If no parameters were identified for sampling, default to a single NUTS sampler.
    if isempty(sampler_assignments)
        @warn "Could not identify any parameters to sample. Defaulting to a single NUTS sampler for all parameters."
        return NUTS(adaptation_steps, target_acceptance; adtype=adtype_to_use)
    # If only one sampler assignment exists, return that sampler directly.
    elseif length(sampler_assignments) == 1
        return sampler_assignments[1][2]
    # Otherwise, construct a Gibbs sampler from all the assignments.
    else
        return Gibbs(sampler_assignments...)
    end
end



function _extract_flexichain_param_samples(chain, param_name::String)
    param_sym = Symbol(param_name)
    base_param_sym = Symbol(first(Base.split(param_name, '[')))
 
    chain_keys = names(DataFrame(chain))

    if !(base_param_sym in chain_keys)
        error("Parameter ':$param_sym' or base ':$base_param_sym' not found in the MCMC chain.")
    end

    param_data_array = Array(chain[base_param_sym])

    num_iterations = size(param_data_array, 1)
    num_chains = size(param_data_array, ndims(param_data_array))
    total_samples = num_iterations * num_chains

    if num_chains == 1 && ndims(param_data_array) > 1 && size(param_data_array, 2) > 1 && ndims(param_data_array) < 3
        return param_data_array
    end

    perm_dims = (1, ndims(param_data_array), 2:(ndims(param_data_array)-1)...)
    permuted_data = permutedims(param_data_array, perm_dims)

    param_dims = size(permuted_data)[3:end]
    
    if isempty(param_dims) # Scalar parameter
        return reshape(permuted_data, total_samples, 1)
    else
        if prod(param_dims) == 1 && length(param_dims) > 0 # Handle vector of length 1
            return reshape(permuted_data, total_samples, 1)
        else
            return reshape(permuted_data, total_samples, prod(param_dims))
        end
    end
end

 
# get_params_vector is for scalar parameters (expected_len = 1)
function get_params_vector(chain, param_name::String, expected_len::Int)
    if expected_len != 1
        error("get_params_vector is intended for scalar parameters (expected_len=1). Use get_params_matrix for vector parameters.")
    end
    samples = _extract_flexichain_param_samples(chain, param_name)
    return samples # Should be (total_samples, 1)
end

# get_params_matrix is for vector parameters (expected_len > 1)
function get_params_matrix(chain, param_name::String, expected_len::Int)
    if expected_len == 1
        error("get_params_matrix is intended for vector parameters (expected_len > 1). Use get_params_vector for scalar parameters.")
    end
    samples = _extract_flexichain_param_samples(chain, param_name)
    # Check if the extracted samples have the correct expected_len
    if size(samples, 2) != expected_len
        error("Parameter '$param_name' has dimension $(size(samples, 2)), but expected $expected_len.")
    end
    return samples # Should be (total_samples, expected_len)
end



"""
    bstm_sample(model::DynamicPPL.Model, n_samples::Int; kwargs...)

A convenience wrapper for `bstm_sample` that automatically selects an optimal sampler
if one is not provided.

# Rationale
This method simplifies the user experience by automatically calling `get_optimal_sampler`
when a specific sampler is not provided. It calculates a default number of adaptation
steps for HMC-based samplers as 25% of the total number of samples, which is a
reasonable heuristic for balancing warmup and sampling time.

# Arguments
- `model`: The Turing model object.
- `n_samples`: The number of samples to draw per chain.
- `kwargs...`: Additional keyword arguments passed to `get_optimal_sampler` and `Turing.sample`.

# Returns
- The MCMC chain object returned by `Turing.sample`.
"""
function bstm_sample(model::DynamicPPL.Model, n_samples::Int; kwargs...)
    # Separate kwargs for get_optimal_sampler and the main sample call.
    sampler_kwargs = Dict{Symbol, Any}()
    sample_kwargs = Dict{Symbol, Any}()

    # List of keywords known to get_optimal_sampler
    known_sampler_keys = [
        :sampler_choice, :sampler_map, :adtype, :group_components,
        :adaptation_steps, :target_acceptance, :n_particles, :n_chains
    ]

    for (key, value) in kwargs
        if key in known_sampler_keys
            sampler_kwargs[key] = value
        else
            sample_kwargs[key] = value
        end
    end

    # Set default adaptation_steps if not provided by the user.
    if !haskey(sampler_kwargs, :adaptation_steps)
        adaptation_steps = floor(Int, n_samples * 0.25)
        sampler_kwargs[:adaptation_steps] = adaptation_steps
        @info "Automatically setting `adaptation_steps` to $(adaptation_steps) (25% of `n_samples`)."
    end

    @info "Sampler not provided. Automatically selecting an optimal sampler..."
    sampler = get_optimal_sampler(model; sampler_kwargs...)
    
    # The `n_chains` kwarg, if present, will be in sample_kwargs or its default will be used.
    return bstm_sample(model, sampler, n_samples; sample_kwargs...)
end
 
"""
    bstm_sample(model, sampler, n_samples; n_chains=1, kwargs...)

A wrapper around `Turing.sample` that defaults to the `MCMCThreads` backend for
both single and multi-chain sampling, ensuring consistent output dimensionality.

# Rationale
Dynamically generated models from `@bstm` can trigger non-fatal world-age warnings,
which this wrapper suppresses. By defaulting to `MCMCThreads()` for all sampling,
it ensures that the returned chain object is always a 3-dimensional array of
`[iterations, parameters, chains]`, even for `n_chains=1`. This prevents
`DimensionMismatch` errors in downstream post-processing functions.

# Arguments
- `model`: The Turing model object.
- `sampler`: The MCMC sampler to use.
- `n_samples`: The number of samples to draw per chain.
- `n_chains::Int`: The number of MCMC chains to run. Default: `1`.
- `kwargs...`: Additional keyword arguments passed directly to `Turing.sample`.

# Returns
- A 3-dimensional MCMC chain object.
"""
function bstm_sample(model, sampler, n_samples; n_chains::Int=1, kwargs...)
    local chain
    redirect_stderr(devnull) do
        @info "Running $(n_chains) chain(s) using MCMCThreads() backend."
        chain = sample(model, sampler, MCMCThreads(), n_samples, n_chains; kwargs...)
    end
    return chain
end

# Forward single-chain syntax: sample(model, sampler, N; kwargs...) 
# directly to the multi-chain MCMCThreads() signature.
function AbstractMCMC.sample(
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    N::Integer;
    kwargs...
)
    # Default to 1 chain run via MCMCThreads() if n_chains isn't specified
    return AbstractMCMC.sample(model, sampler, MCMCThreads(), N, 1; kwargs...)
end

# Forward multi-chain syntax without ensemble strategy: sample(model, sampler, N, n_chains; kwargs...)
function AbstractMCMC.sample(
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    N::Integer,
    n_chains::Integer;
    kwargs...
)
    return AbstractMCMC.sample(model, sampler, MCMCThreads(), N, n_chains; kwargs...)
end



"""
    _generate_likelihood_section(M::NamedTuple, is_multivariate::Bool)

Generates Turing code for all likelihood-specific priors.

# Rationale
This function is a core part of the code generation pipeline and is not deprecated.
It is responsible for defining the priors for all parameters that are specific to
the chosen likelihood family, but are not part of a `ComponentModel`. This includes
parameters for dispersion, zero-inflation, observation noise, and the complex
cut-point parameterization required for ordinal models.

This version is fully consistent with the refactored architecture and correctly:
1.  Generates priors for standard likelihood parameters like `r_nb` (Negative Binomial),
    `y_sigma` (Gaussian-like families), and `lik_phi_zi` (Zero-Inflation).
2.  Handles multivariate models by correctly specifying priors for shared correlation
    matrices (`L_corr`) and outcome-specific variance parameters.
3.  Includes the necessary prior definitions for ordinal regression models, including
    the ordered cut-points (`alphas`) and the degrees of freedom (`ordinal_df`) for
    the latent Student's T distribution.

This function ensures that all necessary likelihood-level priors are declared before
the main model components are processed.

# Arguments
- `M`: The main model configuration `NamedTuple`.
- `is_multivariate`: A boolean indicating if the model is multivariate.

# Returns
- A `String` containing the generated Turing code for the likelihood priors.
"""
function _generate_likelihood_section(M::NamedTuple, is_multivariate::Bool)
    families = [string(get(spec, :family, "gaussian")) for spec in M.likelihood_specs]
    
    prior_blocks = String[]

    # Prior for Negative Binomial dispersion
    if any(f -> f == "negbin", families)
        push!(prior_blocks, "r_nb ~ DynamicPPL.NamedDist(Exponential(1.0), :r_nb)")
    end

    # Prior for Zero-Inflation or Hurdle probability
    if get(M, :user_provided_hurdle, false)
        push!(prior_blocks, "lik_phi_hurdle ~ DynamicPPL.NamedDist(Beta(1,1), :lik_phi_hurdle)")
    elseif get(M, :use_zi, false)
        push!(prior_blocks, "lik_phi_zi ~ DynamicPPL.NamedDist(Beta(1,1), :lik_phi_zi)")
    end

    # Prior for Student's T degrees of freedom
    if any(f -> f == "student_t", families)
        push!(prior_blocks, "lik_nu_student_t ~ DynamicPPL.NamedDist(Exponential(1.0), :lik_nu_student_t)")
    end

    # Prior for observation standard deviation (for Gaussian-like families)
    if any(f -> f in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"], families)
        y_sigma_prior_str = _distribution_to_string(Exponential(1.0))
        if is_multivariate
            push!(prior_blocks, "y_sigma ~ DynamicPPL.NamedDist(filldist($(y_sigma_prior_str), K), :y_sigma)")
        else
            push!(prior_blocks, "y_sigma ~ DynamicPPL.NamedDist($(y_sigma_prior_str), :y_sigma)")
        end
    end
    
    # Prior for extra parameters (e.g., Gamma shape, Beta precision)
    if any(f -> f in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"], families)
        push!(prior_blocks, "lik_extra_params ~ DynamicPPL.NamedDist(Exponential(1.0), :lik_extra_params)")
    end

    # Prior for multivariate correlation matrix
    if is_multivariate
        push!(prior_blocks, "L_corr ~ DynamicPPL.NamedDist(LKJCholesky(K, 1.0), :L_corr)")
    end

    # --- Priors for Ordinal Model ---
    ordinal_spec_idx = findfirst(s -> string(get(s, :family, "")) == "ordinal", M.likelihood_specs)
    if !isnothing(ordinal_spec_idx)
        spec = M.likelihood_specs[ordinal_spec_idx]
        K = get(spec, :K, 0)
        latent_dist = get(spec, :latent_dist, :logistic)

        if K > 2
            # Prior for the first cut-point and the positive differences for subsequent cut-points.
            push!(prior_blocks, "ordinal_alpha_raw_1 ~ DynamicPPL.NamedDist(Normal(0, 5), :ordinal_alpha_raw_1)")
            push!(prior_blocks, "ordinal_alpha_diffs ~ DynamicPPL.NamedDist(filldist(Exponential(1.0), $(K - 2)), :ordinal_alpha_diffs)")
        elseif K == 2
            # For a binary ordinal model, only one cut-point is needed.
            push!(prior_blocks, "ordinal_alpha_raw_1 ~ DynamicPPL.NamedDist(Normal(0, 5), :ordinal_alpha_raw_1)")
        end

        # Prior for the degrees of freedom if using a Student's T latent distribution.
        if latent_dist == :student_t
            push!(prior_blocks, "ordinal_df ~ DynamicPPL.NamedDist(Exponential(1.0), :ordinal_df)")
        end
    end

    return join(prior_blocks, "\n    ")
end



function _generate_univariate_likelihood_block(M::NamedTuple)
    family = string(M.likelihood_specs[1][:family])
    family_symbol = QuoteNode(Symbol(family))

    # Determine which kwargs are needed based on the family
    needs_sigma = family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
    needs_rnb = family == "negbin"
    needs_nu = family == "student_t"
    needs_extra = family in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"]

    kwargs_parts = String[]
    extra_param_logic = ""

    if needs_sigma; push!(kwargs_parts, "sigma_y=y_sigma"); end
    if needs_rnb; push!(kwargs_parts, "r_nb=r_nb"); end
    if get(M, :use_zi, false); push!(kwargs_parts, "phi_zi=lik_phi_zi"); end
    if get(M, :user_provided_hurdle, false); push!(kwargs_parts, "phi_hurdle=lik_phi_hurdle"); end
    
    if needs_nu || needs_extra
        extra_param_logic = if needs_nu; "extra_p = lik_nu_student_t"
        else; "extra_p = lik_extra_params"; end
        push!(kwargs_parts, "extra_params=extra_p")
    end

    if get(M, :user_provided_trials, false); push!(kwargs_parts, "trial=M.trials[:, 1]"); end
    if get(M, :user_provided_weights, false); push!(kwargs_parts, "weight=M.weights[:, 1]"); end
    if get(M, :user_provided_censor_lower, false); push!(kwargs_parts, "censor_lower=M.censor_lower[:, 1]"); end
    if get(M, :user_provided_censor_upper, false); push!(kwargs_parts, "censor_upper=M.censor_upper[:, 1]"); end
    if get(M, :user_provided_hurdle, false); push!(kwargs_parts, "hurdle=M.hurdle[:, 1]"); end

    kwargs_str = join(kwargs_parts, ", ")

    block_content = """
        $(extra_param_logic)
        d_lik_vec = bstm_Likelihood.($(family_symbol), eta; $(kwargs_str))
        log_lik_sum = sum(Distributions.logpdf.(d_lik_vec, M.y_obs))
    """
    
    return """
    let
        local log_lik_sum
        $(block_content)
        Turing.@addlogprob! log_lik_sum
    end
    """
end


function _generate_multivariate_likelihood_block(M::NamedTuple)
    if any(s -> string(get(s, :family, "")) == "dirichlet_multinomial", M.likelihood_specs)
        return """
        let
            eta_correlated = eta_latent * L_corr.L
            family = :dirichlet_multinomial
            
            trials_vec = sum.(eachrow(M.y_obs))
            d_lik_vec = bstm_Likelihood.(family, eachrow(eta_correlated); trial=trials_vec)
            
            Turing.@addlogprob! sum(logpdf.(d_lik_vec, eachrow(M.y_obs)))
        end
        """
    end

    loop_body_parts = String[]
    for k in 1:M.outcomes_N
        family = string(M.likelihood_specs[k][:family])
        family_symbol = QuoteNode(Symbol(family))

        needs_sigma = family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
        needs_rnb = family == "negbin"
        needs_nu = family == "student_t"
        needs_extra = family in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"]

        kwargs_parts = String[]
        extra_param_logic = ""

        if needs_sigma; push!(kwargs_parts, "sigma_y=y_sigma[$k]"); end
        if needs_rnb; push!(kwargs_parts, "r_nb=r_nb"); end
        if get(M, :use_zi, false); push!(kwargs_parts, "phi_zi=lik_phi_zi"); end
        if get(M, :user_provided_hurdle, false); push!(kwargs_parts, "phi_hurdle=lik_phi_hurdle"); end
        
        if needs_nu || needs_extra
            extra_param_logic = if needs_nu; "extra_p = lik_nu_student_t"
            else; "extra_p = lik_extra_params"; end
            push!(kwargs_parts, "extra_params=extra_p")
        end

        if get(M, :user_provided_trials, false); push!(kwargs_parts, "trial=M.trials[:, $k]"); end
        if get(M, :user_provided_weights, false); push!(kwargs_parts, "weight=M.weights[:, $k]"); end
        if get(M, :user_provided_censor_lower, false); push!(kwargs_parts, "censor_lower=M.censor_lower[:, $k]"); end
        if get(M, :user_provided_censor_upper, false); push!(kwargs_parts, "censor_upper=M.censor_upper[:, $k]"); end
        if get(M, :user_provided_hurdle, false); push!(kwargs_parts, "hurdle=M.hurdle[:, $k]"); end

        kwargs_str = join(kwargs_parts, ", ")

        block_content = """
            $(extra_param_logic)
            d_lik_vec_k = bstm_Likelihood.($(family_symbol), view(eta_correlated, :, $k); $(kwargs_str))
            Turing.@addlogprob! sum(Distributions.logpdf.(d_lik_vec_k, view(M.y_obs, :, $k)))
        """

        outcome_block = """
        # Likelihood for outcome $(k)
        let
            $(block_content)
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
    _generate_ordinal_likelihood_block(M::NamedTuple)

Generates the Turing code block for an ordinal regression likelihood. This version
is CPU-only.
"""
function _generate_ordinal_likelihood_block(M::NamedTuple)
    spec = M.likelihood_specs[1]
    K = get(spec, :K, 0)
    if K < 2; return ""; end

    latent_dist_val = get(spec, :latent_dist, :logistic)
    non_prop_terms = get(M, :non_proportional_effects, Symbol[])
    is_npo = !isempty(non_prop_terms)
     
    npo_indices = findall(x -> x in non_prop_terms, M.Xfixed_names)
    n_npo_vars = length(npo_indices)

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
    # Ordinal Likelihood Block
    let
        # Reconstruct the ordered cut-points from their raw parameters.
        alphas_computed = if $(K > 2)
            cumsum([ordinal_alpha_raw_1; ordinal_alpha_diffs])
        else
            [ordinal_alpha_raw_1]
        end

        latent_dist_symbol = :$(latent_dist_val)
        $(npo_update_block)

        # Proportional effect for all observations
        eta_prop = eta

        # Calculate cumulative probabilities for all observations in a vectorized manner
        eta_matrix = if $(is_npo && n_npo_vars > 0)
            eta_prop .+ eta_npo
        else
            # Broadcast the proportional effect across all cut-points
            eta_prop .* ones(T, 1, $(K-1))
        end
        
        # linear_predictor_vec is now a matrix of size [N_obs, K-1]
        linear_predictor_matrix = alphas_computed' .- eta_matrix

        cumulative_probs_matrix = if latent_dist_symbol == :normal
            Distributions.cdf.(Normal(), linear_predictor_matrix)
        elseif latent_dist_symbol == :logistic
            LogExpFunctions.logistic.(linear_predictor_matrix)
        elseif latent_dist_symbol == :student_t
            Distributions.cdf.(TDist(ordinal_df), linear_predictor_matrix)
        else
            error("Unsupported latent distribution ':\$(latent_dist_symbol)' for ordinal model.")
        end
        
        # Calculate probabilities for each category for all observations
        probs_matrix = Array{T}(undef, M.y_N, $(K))
        if $(K > 1)
            probs_matrix[:, 1] = cumulative_probs_matrix[:, 1]
            for j in 2:($(K-1))
                probs_matrix[:, j] = max.(0.0, cumulative_probs_matrix[:, j] .- cumulative_probs_matrix[:, j-1])
            end
            probs_matrix[:, $(K)] = max.(0.0, 1.0 .- cumulative_probs_matrix[:, $(K-1)])
        else
            probs_matrix[:, 1] .= 1.0
        end

        # Normalize probabilities row-wise
        probs_matrix ./= (sum(probs_matrix, dims=2) .+ 1e-9)
        
        # Use broadcasting to apply logpdf to each observation
        log_likelihoods = logpdf.(Categorical.(eachrow(probs_matrix)), M.y_obs)
        Turing.@addlogprob! sum(log_likelihoods)
    end
    """
end



"""
    _generate_final_likelihood_block(M::NamedTuple, is_multivariate::Bool)

Generates the final likelihood block for the Turing model, dispatching to the
appropriate helper based on the model architecture and handling special cases.

# Rationale
This function is a core part of the code generation pipeline. This version has been
refactored to improve modularity. The complex logic for ordinal models has been
extracted into a dedicated helper function, `_generate_ordinal_likelihood_block`.
This change makes the function a clean, high-level dispatcher that routes to the
correct likelihood generator based on the model's configuration, aligning with the
refactoring's goal of improved readability and maintainability.

# Arguments
- `M`: The main model configuration `NamedTuple`.
- `is_multivariate`: A boolean indicating if the model is multivariate.

# Returns
- A `String` containing the generated Turing code for the likelihood block.
"""

function _generate_final_likelihood_block(M::NamedTuple, is_multivariate::Bool)
    # Check if a component like a PointProcess handles its own likelihood.
    # This check is now consistent with the one in `bstm_text_assembler`.
    has_custom_likelihood_from_component = any(spec -> spec.component_obj isa PointProcess, M.components)
    if has_custom_likelihood_from_component
        return "" # The component's `get_updates` method will add the log-probability.
    end

    if is_multivariate
        return _generate_multivariate_likelihood_block(M)
    else
        # For univariate models, check for special families like ordinal.
        family = string(M.likelihood_specs[1][:family])
        if family == "ordinal"
            return _generate_ordinal_likelihood_block(M)
        else
            return _generate_univariate_likelihood_block(M)
        end
    end
end




"""
    _generate_intercept_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)

Generates the Turing code for the global intercept's prior.

# Rationale for Update
This version is updated for consistency with the refactored architecture. The
explicit update to the linear predictor (`eta`) has been removed from this function.
Instead, the intercept is now added during the initialization of `eta` in the main
model assembler (`bstm_text_assembler`). This change is critical for ensuring
automatic differentiation (AD) type stability, as it guarantees that `eta` is
initialized with the correct `Dual` number type when using gradient-based samplers.
The unnecessary `local` keyword has also been removed.

# Arguments
- `M`: The main model configuration `NamedTuple`.
- `is_multivariate`: A boolean indicating if the model is multivariate.
- `eta_name`: The name of the linear predictor variable (unused, but kept for signature consistency).

# Returns
- A tuple `(priors_code::String, updates_code::String)`, where `updates_code` is always empty.
"""
function _generate_intercept_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    if !get(M, :add_intercept, false)
        return "", ""
    end
    
    intercept_prior_obj = get(M, :intercept_prior, Normal(0, 5))
    
    dist_str = if is_multivariate
        "filldist($(_distribution_to_string(intercept_prior_obj)), K)"
    else
        _distribution_to_string(intercept_prior_obj)
    end
    
    prior_code = "intercept ~ DynamicPPL.NamedDist($(dist_str), :intercept)"
    
    # The update code is intentionally empty. The intercept is added during the
    # initialization of the `eta` vector in the model assembler to ensure AD type stability.
    update_code = ""
    
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
fixed effects. This version is CPU-only.
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
        
        beta_prop_name = is_multivariate ? "Xfixed_beta_prop_flat" : "Xfixed_beta_prop"
        n_params_prop = is_multivariate ? n_prop * M.outcomes_N : n_prop
        prior_label = is_multivariate ? :Xfixed_beta_prop_flat : :Xfixed_beta_prop

        # Generate prior string
        if all_same_prop
            prior_str = _distribution_to_string(priors_prop[1])
            push!(prior_parts, "$(beta_prop_name) ~ DynamicPPL.NamedDist(filldist($(prior_str), $(n_params_prop)), $(QuoteNode(prior_label)))")
        else
            priors_to_use = is_multivariate ? vcat([priors_prop for _ in 1:M.outcomes_N]...) : priors_prop
            priors_str_list = [_distribution_to_string(p) for p in priors_to_use]
            push!(prior_parts, "$(beta_prop_name) ~ DynamicPPL.NamedDist(Product([$(join(priors_str_list, ", "))]), $(QuoteNode(prior_label)))")
        end

        # Generate update string
        update_code = if is_multivariate
            "$(eta_name) .+= M.Xfixed[:, $(prop_indices)] * reshape($(beta_prop_name), $(n_prop), M.outcomes_N)"
        else
            "$(eta_name) .+= M.Xfixed[:, $(prop_indices)] * $(beta_prop_name)"
        end
        push!(update_parts, update_code)
    end

    # --- Non-Proportional Effects (for Ordinal Models) ---
    if n_npo > 0 && K_ordinal > 1
        priors_npo = priors_vec[npo_indices]
        all_same_npo = !isempty(priors_npo) && all(p -> p == priors_npo[1], priors_npo)
        beta_npo_name = "beta_npo"
        n_npo_params = n_npo * (K_ordinal - 1)

        if all_same_npo
            prior_str = _distribution_to_string(priors_npo[1])
            push!(prior_parts, "$(beta_npo_name) ~ DynamicPPL.NamedDist(filldist($(prior_str), $(n_npo_params)), :beta_npo)")
        else
            full_priors_list = vcat([priors_npo for _ in 1:(K_ordinal-1)]...)
            priors_str_list = [_distribution_to_string(p) for p in full_priors_list]
            push!(prior_parts, "$(beta_npo_name) ~ DynamicPPL.NamedDist(Product([$(join(priors_str_list, ", "))]), :beta_npo)")
        end
    end

    priors_code = join(prior_parts, "\n    ")
    updates_code = join(update_parts, "\n    ")
    
    return priors_code, updates_code
end



"""
    process_smooth_module!(opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict)

Processes a module with `structure=:smooth`, which is typically invoked via
`random(var, structure=:smooth, model=...)`. This function is responsible for
generating basis matrices for various static smoothers or setting up coordinate
data for continuous and dynamic kernel-based models.

# Rationale for Update
  consistent with the refactored component processing
pipeline. It removes a recursive call pattern for handling symbolic `nbins` arguments,
replacing it with a more direct, linear control flow that improves code clarity.

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
    basis_registry = opt_dict[:basis_matrices]
    data = opt_dict[:data]
    params = mod_data[:params]
    model_param = get(params, :model, "pspline")
    original_nbins_param = get(params, :nbins, 20)
    variables = mod_data[:variables]
    n_vars = length(variables)
    
    # Resolve the nbins parameter at the beginning to avoid recursion.
    local nbins_resolved
    if original_nbins_param isa Int || original_nbins_param isa Vector{Int}
        nbins_resolved = original_nbins_param
    else
        calling_mod = get(opt_dict, :calling_module, Main)
        try
            nbins_resolved = Core.eval(calling_mod, original_nbins_param)
        catch e
            error("Could not evaluate `nbins` parameter `$(original_nbins_param)`. Error: $e")
        end
    end

    nbins_per_dim_vec = Int[]
    total_bins_for_component_obj = 0

    if n_vars > 0
        if nbins_resolved isa Int
            nbins_per_dim_vec = fill(nbins_resolved, n_vars)
            total_bins_for_component_obj = nbins_resolved^n_vars
        elseif nbins_resolved isa Vector{Int}
            if length(nbins_resolved) != n_vars
                error("`nbins` vector length must match number of variables for smooth. Got $(length(nbins_resolved)) for $n_vars variables.")
            end
            nbins_per_dim_vec = nbins_resolved
            total_bins_for_component_obj = prod(nbins_resolved)
        else
            error("Resolved `nbins` parameter is not an Int or Vector{Int}. Got type $(typeof(nbins_resolved))")
        end
        mod_data[:params][:nbins] = total_bins_for_component_obj
        if !isempty(nbins_per_dim_vec)
            mod_data[:params][:nbins_per_dim] = nbins_per_dim_vec
        end
    end

    # Categorize models to determine processing path
    basis_models = ["pspline", "bspline", "tps", "moran", "gp", "barycentric", "linear", "invdist"]
    dynamic_basis_models = ["wavelet", "fft"]
    continuous_kernel_models = ["gp", "fitc", "svgp", "nystrom", "warp", "spde", "exponentialdecay", "rff", "kriging"]
    gmrfs_on_bins_models = ["rw1", "rw2", "ar1", "icar", "besag", "cyclic"]
    
    model_str = string(model_param)

    # --- Path 1: Dynamic Basis Models (e.g., wavelet, fft) ---
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
    process_eigen_module!(opt_dict, mod_data, registries, hyperpriors)

Processes the `eigen()` module, which performs Bayesian Principal Component Analysis
(PCA) for dimensionality reduction.

# Version
v1.1.0 (2026-08-13)

# Mathematical Summary
The `eigen()` component models a set of \$P\$ observed variables \$\\mathbf{Y}\$ (an \$N \\times P\$
matrix) as a linear combination of \$K\$ latent factors (principal components)
\$\\mathbf{F}\$ (an \$N \\times K\$ matrix) plus residual noise:
\$\\mathbf{Y} = \\mathbf{F} \\mathbf{L}^T + \\mathbf{E}\$
where:
- \$\\mathbf{L}\$ is the \$P \\times K\$ matrix of factor loadings (eigenvectors).
- \$\\mathbf{E}\$ is the residual noise matrix.

This processor prepares the data for the model by:
1.  Extracting the specified variables from the main data frame.
2.  Centering the data matrix by subtracting the column means, a standard
    pre-processing step for PCA.
3.  Validating that the number of requested factors is less than the number of
    input variables.
4.  Pre-calculating indices needed for the Householder transformation, which is used
    to construct the orthonormal loadings matrix \$\\mathbf{L}\$ in a numerically stable way.

# Inputs (from `mod_data`)
- `variables`: A `Vector` of `Symbol`s specifying the columns in the data to be
  used for PCA.
- `params[:n_factors]`: `Int`, the number of latent factors to extract.

# Outputs (mutates `mod_data[:params]`)
- `eigen_data::Matrix`: The centered \$N \\times P\$ data matrix.
- `n_vars::Int`: The number of input variables, \$P\$.
- `n_factors::Int`: The number of latent factors, \$K\$.
- `ltri_indices::Vector{Int}`: Indices for parameterizing the Householder reflectors.
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
    
    # Check for missing values in the specified columns.
    if any(col -> any(ismissing, data[!, col]), vars_sym)
        error("Columns for eigen() module contain missing values. Please handle them before calling bstm().")
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
    # This parameterizes the reflector vectors for constructing the orthonormal loadings matrix.
    ltri_mask = [r >= c for r in 1:n_vars, c in 1:n_factors]
    ltri_indices = findall(vec(ltri_mask))
    
    mod_data[:params][:ltri_indices] = ltri_indices
    mod_data[:params][:n_factors] = n_factors
    mod_data[:params][:n_vars] = n_vars
    
    return true # Proceed with component creation.
end



"""
    process_mixed_module!(opt_dict, mod_data, registries, hyperpriors)

Processes the `mixed()` module for random effects.

# Rationale for Update
This version enhances robustness by adding a check for the number of unique levels
in the grouping variable. If the number of levels is suspiciously high (suggesting
a continuous variable was passed by mistake), it issues a warning to the user.
The core logic, which uses `StatsModels.jl` to correctly parse the random effects
formula (e.g., `1 + cov1 | group`), remains consistent with the refactored architecture.

# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the `mixed()` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries.

# Returns
- `true` to indicate that a `Mixed` component object should be created.
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
    n_levels = length(unique_levels)
    n_obs = nrow(data)

    # Add a warning if the number of levels is suspiciously high.
    if n_levels > n_obs / 2 && n_levels > 50 # Heuristic threshold
        @warn "Grouping variable ':$group_var_sym' has a large number of unique levels ($n_levels for $n_obs observations). If this is a continuous variable, the model may be very large and slow. Consider binning the variable or ensuring it is a categorical factor."
    end

    group_map = Dict(v => i for (i, v) in enumerate(unique_levels))
    indices = [group_map[v] for v in group_data]

    # Store the generated indices and parameters in the configuration dictionaries
    # for the code generator to use.
    index_key = Symbol("mixed_idx_$(group_var_str)")
    opt_dict[index_key] = indices
    
    mod_data[:params][:indices] = indices
    mod_data[:params][:n_cat] = n_levels
    mod_data[:params][:lhs] = effect_names
    mod_data[:variables] = [group_var_str]
    
    return true
end



"""
    process_fixed_module!(opt_dict, mod_data, registries, hyperpriors)

Processes the `fixed()` module, gathering information about fixed effects, custom
contrasts, and priors.

# Rationale for Update
This version improves the robustness of contrast handling. The previous implementation
only applied a specified `contrast` to the first variable in a `fixed()` call. This
has been corrected to iterate over all variables provided in the module and apply the
contrast to each one, making the behavior more intuitive and consistent. The
initialization block has also been simplified using `get!`.

# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the `fixed()` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries.

# Returns
- `false`, as the `fixed()` module itself does not create a `Component` object.
"""
function process_fixed_module!(opt_dict, mod_data, registries, hyperpriors)
    # Ensure necessary keys exist in the configuration dictionary.
    get!(opt_dict, :fixed_effects_from_modules, String[])
    get!(opt_dict, :contrasts, Dict{Symbol, Any}())
    get!(opt_dict, :fixed_effects_priors, Dict{Symbol, Any}())
    get!(opt_dict, :vars_to_categorize, Set{Symbol}())
    
    params = mod_data[:params]
    vars = mod_data[:variables]

    # Collect all variables specified in this fixed() call.
    for var in vars
        push!(opt_dict[:fixed_effects_from_modules], string(var))
    end

    # Handle custom contrast coding.
    if haskey(params, :contrast)
        if !isempty(vars)
            contrast_sym = params[:contrast]
            contrast_obj = if haskey(STATSMODELS_CONTRASTS, contrast_sym)
                STATSMODELS_CONTRASTS[contrast_sym]
            else
                @warn "Unknown contrast coding ':$contrast_sym'. Using default (DummyCoding)."
                StatsModels.DummyCoding()
            end
            # Apply the contrast to all variables in this fixed() call.
            for var in vars
                opt_dict[:contrasts][Symbol(var)] = contrast_obj
            end
        else
            @warn "A 'contrast' was specified in a fixed() module with no variable. Ignoring."
        end
    end

    # Handle custom priors.
    if haskey(params, :prior)
        for var in vars
            opt_dict[:fixed_effects_priors][Symbol(var)] = params[:prior]
        end
    end

    # Handle explicit categorization.
    if get(params, :model, nothing) == :categorical || haskey(params, :contrast)
        for var in vars
            push!(opt_dict[:vars_to_categorize], Symbol(var))
        end
    end
    
    # The fixed() module only configures the model; it does not create a component itself.
    return false
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

This version adds a validation step to ensure that the required `code_fragment`
argument is provided, failing early with an informative error message if it is missing.

# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the `custom()` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries (not used here).

# Returns
- `true` to indicate that a `Custom` component object should be created.
"""
function process_custom_module!(opt_dict, mod_data, registries, hyperpriors)
    # Validate that the essential `code_fragment` parameter is present.
    if !haskey(mod_data[:params], :code_fragment)
        error("The `custom()` module requires a `code_fragment` argument containing the user-defined Turing code string.")
    end
    
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
the modular design of the refactor. The logic for resolving and validating
centroids from the previous version is retained and made more efficient.

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
            
            # Efficiently create a map from each spatial index to its unique coordinate using DataFrames.
            gdf = groupby(data, :s_idx)
            unique_coords_df = combine(gdf, [:s_x, :s_y] .=> first, renamecols=false)
            coord_map = Dict(row.s_idx => Point2D(row.s_x, row.s_y) for row in eachrow(unique_coords_df))

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
    sub_config_kwargs = copy(opt_dict)
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
    process_random_module!(opt_dict, mod_data, registries, hyperpriors)

Processes a `random()` module call from the formula. This function is a central
part of the model configuration pipeline. Its primary responsibility is to set up
the structural context (e.g., :spatial, :temporal) for a random effect.

It infers the structure, validates and processes index variables (like `s_idx` or
`t_idx`), and populates the main configuration dictionary (`opt_dict`) with shared
information like the number of spatial units (`s_N`) or the adjacency matrix (`W`).

This processor fully handles the creation and registration of the component and
returns `false` to signal to the main configuration loop that no further processing
is needed for this component.
"""
function process_random_module!(
    opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict
)
    # 1. Infer the structure (:spatial, :temporal, :smooth, etc.)
    structure = _infer_structure_from_args(mod_data[:variables], mod_data[:params])
    mod_data[:params][:structure] = structure

    data = opt_dict[:data]

    # 2. Process based on the inferred structure, setting up shared indices in opt_dict
    if structure == :spatial
        variables = mod_data[:variables]
        if isempty(variables)
            error("A spatial `random()` component requires a spatial index variable (e.g., `random(region, ...)`).")
        end
        
        s_var_sym = Symbol(variables[1])

        if !hasproperty(data, s_var_sym)
            error("Spatial index variable ':$s_var_sym' not found in data.")
        end

        if !haskey(opt_dict, :s_idx)
            s_idx = data[!, s_var_sym]
            opt_dict[:s_idx] = s_idx
            opt_dict[:s_N] = length(unique(s_idx))
        end
        
        if !haskey(opt_dict, :W) && haskey(mod_data[:params], :W)
            W_arg = mod_data[:params][:W]
            if W_arg isa Symbol || W_arg isa Expr
                calling_mod = get(opt_dict, :calling_module, Main)
                try
                    opt_dict[:W] = Core.eval(calling_mod, W_arg)
                catch e
                    error("Could not evaluate `W` argument `$(W_arg)` for component '$(mod_data[:key])'. Error: $e")
                end
            else
                opt_dict[:W] = W_arg
            end
        end

    elseif structure == :temporal
        variables = mod_data[:variables]
        if isempty(variables)
            error("A temporal `random()` component requires a time index variable (e.g., `random(year, ...)`).")
        end
        
        t_var_sym = Symbol(variables[1])
        if !hasproperty(data, t_var_sym)
            error("Temporal index variable ':$t_var_sym' not found in data.")
        end

        if !haskey(opt_dict, :t_idx)
            time_opts = Dict(:time_method => get(mod_data[:params], :time_method, "regular"))
            tu_meta = assign_time_units(data[!, t_var_sym]; time_opts...)
            opt_dict[:t_idx] = tu_meta.idx
            opt_dict[:t_N] = tu_meta.N_cat
            opt_dict[:t_idx_var] = t_var_sym
        end

    elseif structure == :spacetime
        if length(mod_data[:variables]) < 2
            error("A spacetime `random()` component requires both a spatial and a temporal index variable, e.g., `random(s, t, ...)`.")
        end
        
        s_var_sym = Symbol(mod_data[:variables][1])
        if !haskey(opt_dict, :s_idx)
            if !hasproperty(data, s_var_sym)
                error("Spatial index variable ':$s_var_sym' not found in data.")
            end
            opt_dict[:s_idx] = data[!, s_var_sym]
            opt_dict[:s_N] = length(unique(opt_dict[:s_idx]))
        end

        t_var_sym = Symbol(mod_data[:variables][2])
        if !haskey(opt_dict, :t_idx)
            if !hasproperty(data, t_var_sym)
                error("Temporal index variable ':$t_var_sym' not found in data.")
            end
            time_opts = Dict(:time_method => get(mod_data[:params], :time_method, "regular"))
            tu_meta = assign_time_units(data[!, t_var_sym]; time_opts...)
            opt_dict[:t_idx] = tu_meta.idx
            opt_dict[:t_N] = tu_meta.N_cat
            opt_dict[:t_idx_var] = t_var_sym
        end

        if !haskey(opt_dict, :st_idx)
            opt_dict[:st_idx] = (opt_dict[:t_idx] .- 1) .* opt_dict[:s_N] .+ opt_dict[:s_idx]
        end
    end

    # 3. Create the component object
    component_obj = resolve_technical_primitive(
        mod_data, NamedTuple(opt_dict), hyperpriors, opt_dict[:prior_scheme]
    )
    mod_data[:component_obj] = component_obj

    # 4. Get precomputes
    M_nt = NamedTuple(opt_dict)
    precomputes = get_precomputes(component_obj, M_nt, mod_data)

    # 5. Create the final specification object
    spec = (
        key=Symbol(mod_data[:key]), 
        structure=mod_data[:params][:structure], 
        var=join(string.(mod_data[:variables]), "_"), 
        component_obj=component_obj, 
        params=mod_data[:params], 
        hyper=precomputes
    )
    
    # 6. Add the final spec to the main components list
    push!(registries[:components], spec)

    # 7. Return false to signal to bstm_config that this component is fully processed
    return false
end



function adjacency_to_bipartite(W::AbstractMatrix; force_bipartite::Bool=true)
    # # Process: Translates a unipartite graph into a bipartite representation.
    # # Rationale: Required for models like BCGN that operate on inter-set connectivity.
    
    rows, cols = size(W)
    if rows != cols
        error("Input matrix must be square to represent a unipartite adjacency structure.")
    end
    
    n = rows
    g = SimpleGraph(W)
    
    # # Coloring Algorithm: Attempt to find a natural 2-coloring (bipartition)
    # # nodes are assigned to set 0 or set 1
    colors = fill(-1, n)
    is_bipartite = true
    
    for start_node in 1:n
        if colors[start_node] != -1
            continue
        end
        
        colors[start_node] = 0
        queue = [start_node]
        
        while !isempty(queue)
            u = popfirst!(queue)
            for v in Neighbors(g, u)
                if colors[v] == -1
                    colors[v] = 1 - colors[u]
                    push!(queue, v)
                elseif colors[v] == colors[u]
                    is_bipartite = false
                    if !force_bipartite
                        error("Graph is not bipartite and force_bipartite is false.")
                    end
                end
            end
        end
    end
    
    # # Fallback: If not bipartite, use a greedy degree-based partition to maximize cut
    if !is_bipartite
        @warn "Graph is not naturally bipartite. Applying greedy partitioning to maximize inter-set edges."
        colors = fill(0, n)
        node_degrees = degree(g)
        sorted_nodes = sortperm(node_degrees, rev=true)
        
        for u in sorted_nodes
            # # Count neighbors already in set 0 and set 1
            n0 = 0
            n1 = 0
            for v in Neighbors(g, u)
                if colors[v] == 0
                    n0 += 1
                else
                    n1 += 1
                end
            end
            # # Assign to the set that maximizes connections to the other set
            colors[u] = n0 >= n1 ? 1 : 0
        end
    end
    
    # # Extraction: Construct the bipartite matrix B
    set1_indices = findall(==(0), colors)
    set2_indices = findall(==(1), colors)
    
    n1 = length(set1_indices)
    n2 = length(set2_indices)
    
    if n1 == 0 || n2 == 0
        error("Partitioning failed to create two non-empty sets. Check graph connectivity.")
    end
    
    # # B is n1 x n2 matrix representing connections from Set 1 to Set 2
    B = spzeros(Float64, n1, n2)
    
    for (i, u) in enumerate(set1_indices)
        for (j, v) in enumerate(set2_indices)
            if W[u, v] > 0
                B[i, j] = Float64(W[u, v])
            end
        end
    end
    
    return (
        bipartite_adj = B,
        set1 = set1_indices,
        set2 = set2_indices,
        is_natural = is_bipartite
    )
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
            
            # Corrected call: Use process_random_module! for the smooth component.
            smooth_mod_data = Dict(:type => :random, :variables => modifier_vars, :params => inner_node.args)
            process_random_module!(opt_dict, smooth_mod_data, registries, hyperpriors)
            
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
            
            # Corrected call: Use process_random_module! for the smooth component.
            smooth_mod_data = Dict(:type => :random, :variables => dynamic_vars, :params => dynamic_node.args)
            process_random_module!(opt_dict, smooth_mod_data, registries, hyperpriors)
            
            state_node = node2
            # Corrected call: Use process_random_module! for the spatial component.
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


function _process_mosaic_grouping!(opt_dict, mod_data)
    # Purpose: Handles the grouping logic for mosaic models, partitioning the spatial domain.
    # Rationale: Centralizes the logic for creating mosaic groups, supporting k-means clustering
    #            or pre-defined grouping columns. This version enhances robustness and clarity,
    #            especially for pre-defined groups, by validating consistency and completeness.
    # v1.0.1 (2026-08-13)
    # Inputs:
    #   - opt_dict: The main model configuration dictionary.
    #   - mod_data: The parsed data for the `random()` module call.
    # Outputs: A NamedTuple containing the new observation-level grouping column name and the number of regions.

    s_N = get(opt_dict, :s_N, 0)
    if s_N == 0
        error("Mosaic models require a spatial context (`s_N`) to be established first. Ensure a spatial variable and adjacency matrix `W` are provided.")
    end
    
    data = opt_dict[:data]
    params = mod_data[:params]
    mosaic_param = get(params, :mosaic, :none)
    
    cluster_assignments::Vector{Int}
    n_regions::Int
    
    group_col_name = Symbol("mosaic_group_for_", mod_data[:key])

    if mosaic_param == :kmeans
        # --- K-Means Clustering Logic ---
        # This path uses spatial coordinates to perform clustering.
        if !haskey(opt_dict, :centroids)
            # Attempt to derive centroids from s_x, s_y if not explicitly provided.
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
                error("Mosaic k-means requires centroids or `s_x`/`s_y` coordinates in the data.")
            end
        end
        
        centroids = opt_dict[:centroids]
        if length(centroids) != s_N
            error("Number of centroids ($(length(centroids))) does not match s_N ($(s_N)).")
        end

        # Resolve n_regions, allowing it to be a symbol pointing to a variable.
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
            @warn "Number of centroids ($(length(centroids))) is less than the requested number of regions ($n_regions_req). Adjusting n_regions to $(length(centroids))."
            n_regions_req = length(centroids)
        end
        
        centroids_matrix = hcat([c.x for c in centroids], [c.y for c in centroids])'
        kmeans_result = kmeans(centroids_matrix, n_regions_req; maxiter=200, display=:none)
        
        cluster_assignments = assignments(kmeans_result)
        n_regions = nclusters(kmeans_result)

    elseif mosaic_param isa Symbol
        # --- Pre-defined Grouping Column Logic (Enhanced Robustness) ---
        # This path uses a column in the DataFrame to define the spatial groups.
        if !hasproperty(data, mosaic_param)
            error("The specified mosaic grouping column `:$(mosaic_param)` was not found in the data.")
        end

        # Create a mapping from s_idx to the group value.
        # Ensure each s_idx maps to a unique group value.
        s_idx_group_pairs = unique(data[!, [:s_idx, mosaic_param]])
        
        # Validate consistency: each s_idx must map to only one group value.
        s_idx_counts = countmap(s_idx_group_pairs.s_idx)
        for (s_id, count) in s_idx_counts
            if count > 1
                error("Spatial unit `s_idx = $s_id` has multiple conflicting values in the grouping column `:$(mosaic_param)`. Each spatial unit must belong to exactly one group.")
            end
        end

        # Map unique group levels to integers 1:n_regions.
        unique_group_levels = unique(s_idx_group_pairs[!, mosaic_param])
        n_regions = length(unique_group_levels)
        level_to_int_map = Dict(level => i for (i, level) in enumerate(unique_group_levels))
        
        # Build the final cluster_assignments vector for all s_N units.
        cluster_assignments = Vector{Int}(undef, s_N)
        s_idx_to_group_map = Dict{Int, Int}()
        for row in eachrow(s_idx_group_pairs)
            s_idx_to_group_map[row.s_idx] = level_to_int_map[row[mosaic_param]]
        end

        for i in 1:s_N
            if !haskey(s_idx_to_group_map, i)
                error("Spatial unit `s_idx = $i` is missing from the grouping column `:$(mosaic_param)`. All spatial units from 1 to `s_N` must have a defined group.")
            end
            cluster_assignments[i] = s_idx_to_group_map[i]
        end

    else
        error("Invalid `mosaic` parameter. Must be `:kmeans` or a Symbol pointing to a grouping column.")
    end

    # Create the observation-level grouping column needed by the `mixed` processor.
    # This column assigns each observation to its spatial group based on its s_idx.
    opt_dict[:data][!, group_col_name] = cluster_assignments[opt_dict[:data].s_idx]
    
    return (group_col_name=group_col_name, n_regions=n_regions)
end




"""
    process_sciml_module!(opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict)

Processes the `sciml()` module call, validating arguments and setting up temporal context.
This version is CPU-only.
"""
function process_sciml_module!(
    opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict
)
    data = opt_dict[:data]
    params = mod_data[:params]
    variables = mod_data[:variables]
    calling_mod = get(opt_dict, :calling_module, Main)

    # 1. Set up temporal context from the time index variable.
    if isempty(variables)
        error("The `sciml()` module requires a time index variable, e.g., `sciml(year, ...)`.")
    end

    time_var_sym = Symbol(variables[1])
    if !hasproperty(data, time_var_sym)
        error("Time index variable ':$time_var_sym' for sciml() module not found in data.")
    end

    time_opts = Dict(:time_method => get(params, :time_method, "regular"))
    tu_meta = assign_time_units(data[!, time_var_sym]; time_opts...)
    opt_dict[:t_idx] = tu_meta.idx
    opt_dict[:t_N] = tu_meta.N_cat
    opt_dict[:t_idx_var] = time_var_sym
    opt_dict[:t_coords] = data[!, time_var_sym] # Store original time coordinates for interpolation.

    # 2. Validate and evaluate all required SciML parameters.
    required_args = [:model_func, :u0_prior, :p_priors, :tspan, :solver]
    for arg in required_args
        if !haskey(params, arg)
            error("The `sciml()` module is missing the required keyword argument `:$arg`.")
        end
        
        raw_val = params[arg]
        try
            evaluated_val = Core.eval(calling_mod, raw_val)
            params[arg] = evaluated_val
        catch e
            error("Could not evaluate the `$(arg)` argument `$(raw_val)` for the sciml() module. Ensure it is defined in the calling scope. Error: $e")
        end
    end

    # 3. Evaluate optional SciML keyword arguments.
    optional_args = [:saveat, :de_kwargs]
    for arg in optional_args
        if haskey(params, arg)
            raw_val = params[arg]
            if raw_val isa Expr || raw_val isa Symbol
                try
                    params[arg] = Core.eval(calling_mod, raw_val)
                catch e
                    @warn "Could not evaluate `$(arg)` argument for sciml() module. Using default. Error: $e"
                    if arg == :de_kwargs; params[arg] = Dict(); end
                end
            end
        end
    end

    # 4. Store necessary evaluated parameters in the main model configuration.
    opt_dict[:sciml_solver] = params[:solver]
    opt_dict[:sciml_tspan] = params[:tspan]
    opt_dict[:sciml_saveat] = get(params, :saveat, 0.1) # Default saveat

    # 5. Create and store a problem template.
    u0_prior = params[:u0_prior]
    p_priors = params[:p_priors]
    
    u0_mean = mean(u0_prior)
    u0_placeholder = u0_mean isa Number ? [u0_mean] : vec(u0_mean)
    p_placeholder = [mean(p) for p in p_priors]

    prob_func = getfield(calling_mod, params[:model_func])
    de_kwargs = get(params, :de_kwargs, Dict())
    prob_template = ODEProblem(prob_func, u0_placeholder, params[:tspan], p_placeholder; de_kwargs...)

    if !haskey(opt_dict, :sciml_problem_templates)
        opt_dict[:sciml_problem_templates] = Dict{Symbol, Any}()
    end
    opt_dict[:sciml_problem_templates][Symbol(mod_data[:key])] = prob_template

    return true
end



"""
    process_dynamics_module!(opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict)

Processes the `dynamics()` module, ensuring spatial and temporal contexts are established.

  self-contained and consistent with the refactored
architecture. It no longer calls deprecated processors. Instead, it directly
handles the setup of spatial and temporal indices from its own arguments, resolves
the adjacency matrix `W`, and validates all necessary parameters for the specified
mechanistic model.

# Arguments
- `opt_dict`: The main model configuration dictionary (`M`).
- `mod_data`: The parsed data for the `dynamics()` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries.

# Returns
- `true` to indicate that a `Dynamics` component object should be created.
"""
function process_dynamics_module!(
    opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict
)
    params = mod_data[:params]
    data = opt_dict[:data]
    variables = mod_data[:variables] # Positional arguments from the formula, e.g., s_idx, year

    # 1. Validate and set up spatial and temporal indices from formula arguments.
    # The dynamics module expects at least two positional arguments: spatial_index, temporal_index
    if length(variables) < 2
        error("The `dynamics()` module requires at least two positional arguments: a spatial index and a temporal index (e.g., `dynamics(s_idx, year, ...)`).")
    end

    spatial_idx_var = Symbol(variables[1])
    temporal_idx_var = Symbol(variables[2])

    # Resolve spatial index and its number of levels
    if !hasproperty(data, spatial_idx_var)
        error("Spatial index variable ':$spatial_idx_var' for dynamics module not found in data.")
    end
    opt_dict[:s_idx] = data[!, spatial_idx_var]
    opt_dict[:s_N] = length(unique(opt_dict[:s_idx])) # Number of unique spatial units

    # Resolve temporal index and its number of levels
    if !hasproperty(data, temporal_idx_var)
        error("Temporal index variable ':$temporal_idx_var' for dynamics module not found in data.")
    end
    opt_dict[:t_idx] = data[!, temporal_idx_var]
    opt_dict[:t_N] = length(unique(opt_dict[:t_idx])) # Number of unique time steps

    # 2. Resolve adjacency matrix `W`.
    # Prioritize `W` from the module's parameters, then fallback to global opt_dict.
    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr || w_val isa Symbol
            calling_mod = get(opt_dict, :calling_module, Main)
            try
                opt_dict[:W] = Core.eval(calling_mod, w_val)
            catch e
                error("Could not evaluate `W` argument `$(w_val)` for dynamics module. Error: $e")
            end
        else
            opt_dict[:W] = w_val
        end
    end
    if !haskey(opt_dict, :W)
        error("Dynamics models require an adjacency matrix `W`, passed either to the module (e.g., `dynamics(..., W=my_W)`) or as a keyword argument to the main `@bstm` call (e.g., `@bstm(..., W=my_W)`).")
    end
    if opt_dict[:s_N] != size(opt_dict[:W], 1)
        error("Number of unique spatial indices ($(opt_dict[:s_N])) does not match the dimension of the provided adjacency matrix W ($(size(opt_dict[:W], 1))).")
    end

    # 3. Model Type Verification
    model_type = string(get(params, :model, "none"))
    if model_type == "none"
        error("Dynamics module requires a 'model' parameter (e.g., model='advection').")
    end

    # 4. Covariate and Parameter Validation
    if model_type in ["advection", "advection_diffusion"]
        if !haskey(params, :velocity_prior) && !haskey(opt_dict[:hyperpriors], "velocity")
            @warn "Advection model specified without explicit velocity priors. Using system defaults."
        end
    end
    if model_type in ["diffusion", "advection_diffusion"]
        if !haskey(params, :diffusion_prior) && !haskey(opt_dict[:hyperpriors], "diffusion")
            @warn "Diffusion model specified without explicit diffusion priors. Using system defaults."
        end
    end

    # 5. Mapping Spatiotemporal State
    # We pre-calculate the spatiotemporal flat index (st_idx) to allow the
    # code generator to map the [s_N, t_N] state matrix to the observation vector N.
    s_idx = opt_dict[:s_idx]
    t_idx = opt_dict[:t_idx]
    s_N = opt_dict[:s_N]
    opt_dict[:st_idx] = [(t_val - 1) * s_N + s_val for (s_val, t_val) in zip(s_idx, t_idx)]

    return true
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
    :custom => process_custom_module!,
    :sciml => process_sciml_module!
)
