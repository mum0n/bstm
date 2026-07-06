using Turing, DataFrames, StatsModels, CategoricalArrays, LinearAlgebra, SparseArrays, Distributions, NamedArrays, FlexiChains, PosteriorStats, JLD2, Random, Statistics, Logging

# This file contains the corrected core logic for the bstm formula interface.
# It is designed to replace the faulty parsing and configuration functions in
# spatiotemporal_functions.jl and bstm_formula_parsing_utils.jl.

# It resolves the persistent `UndefVarError` by correctly handling the evaluation scope
# for expressions within the formula.

# --------------------------------------------------------------------------------------
# SECTION 1: CORE FORMULA PARSING UTILITIES
# These functions are dependencies for `decompose_bstm_formula`.
# --------------------------------------------------------------------------------------

const BSTM_MODULE_KEYWORDS = Set([
    "intercept", "spatial", "temporal", "smooth",
    "nested", "eigen", "fixed", "mixed", "dynamics", "likelihood", "spacetime"
])

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

function decompose_bstm_formula(formula_str::String)
    parts = Base.split(formula_str, "~")
    lhs = strip(parts[1])
    rhs = strip(parts[2])

    outcome_specs = []
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
            push!(outcome_specs, Dict(:var => outcome_var, :params => params))
        else
            push!(outcome_specs, Dict(:var => term, :params => Dict()))
        end
    end

    modules = Dict{String, Any}()
    fixed_effects = String[]
    has_intercept = false
    rhs_terms = split_terms_at_depth(rhs, "+")

    for term in rhs_terms
        term = strip(term)
        if term == "1" || term == "intercept()"
            has_intercept = true
            continue
        end

        # Generalize to handle all algebraic operators (⊗, ⊕, ∘, |>)
        # This regex captures a left-hand side, an operator, and a right-hand side.
        op_match = match(r"(.+)\s*(⊗|⊕|∘|\|>|otimes|oplus|compose|pipe)\s*(.+)", term)
        
        if !isnothing(op_match)
            left_str, op_str, right_str = op_match.captures
            
            # Map the operator string to its corresponding symbol
            op_sym_map = Dict("⊗"=>:kronecker_product, "⊕"=>:direct_sum, "∘"=>:composition, "|>"=>:pipe,
                              "otimes"=>:kronecker_product, "oplus"=>:direct_sum, "compose"=>:composition, "pipe"=>:pipe)
            op_sym = op_sym_map[op_str]

            # Recursively parse the left and right components of the operation
            left_config = _parse_single_manifold_term(strip(left_str))
            right_config = _parse_single_manifold_term(strip(right_str))

            # Create a generic :interaction module that will be handled by `process_interaction_module!`
            key = "interaction_$(op_sym)_$(left_config.module_type)_$(right_config.module_type)"
            modules[key] = (
                module_type = :interaction,
                args = Dict(
                    :operator => op_sym,
                    :components => [left_config, right_config]
                )
            )
        else
            m_match = match(r"^\s*(\w+)\s*\((.*)\)\s*$", term)
            
            # FIX: The check was `Symbol(m_match.captures[1]) in BSTM_MODULE_KEYWORDS`, which is incorrect
            # because BSTM_MODULE_KEYWORDS is a Set of Strings, not Symbols.
            if !isnothing(m_match) && m_match.captures[1] in BSTM_MODULE_KEYWORDS
                module_config = _parse_single_manifold_term(term)
                key_parts = [string(module_config.module_type)]
                if haskey(module_config.args, :positional_args) && !isempty(module_config.args[:positional_args])
                    push!(key_parts, string(module_config.args[:positional_args][1]))
                end
                base_key = join(key_parts, "_")
                module_key = base_key
                counter = 1
                while haskey(modules, module_key)
                    counter += 1
                    module_key = base_key * "_$counter"
                end
                modules[module_key] = module_config
            elseif !isempty(term)
                push!(fixed_effects, term)
            end
        end
    end

    return (outcomes=outcome_specs, modules=modules, fixed_effects=fixed_effects, has_intercept=has_intercept)
end

# --------------------------------------------------------------------------------------
# SECTION 2: CORE BSTM ENTRY POINTS AND CONFIGURATION ENGINE
# --------------------------------------------------------------------------------------

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
    options = bstm_config(formula, data, calling_module; kwargs...)
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

"""
    bstm_config(formula::String, data::DataFrame, calling_module::Module; kwargs...)

The main configuration engine for `bstm`. It parses the formula, evaluates parameters
in the correct scope, processes modules, and assembles a `NamedTuple` containing all
specifications required to build the Turing model.
"""
function bstm_config(formula::String, data::DataFrame, calling_module::Module; kwargs...)
    # --- 1. Initial Setup ---
    metadata = decompose_bstm_formula(formula)
    opt_dict = Dict{Symbol, Any}(kwargs)
    opt_dict[:data] = data
    opt_dict[:formula] = formula

    # Initialize registries
    basis_matrices_registry = Dict{Symbol, Any}()
    manifolds_registry = []
    registries = Dict(:manifolds => manifolds_registry, :basis_matrices => basis_matrices_registry)
    hyperprior_registry = Dict{String, Any}(string(k) => v for (k, v) in get(opt_dict, :hyperpriors, Dict()))
    opt_dict[:hyperpriors] = hyperprior_registry

    # --- 2. Process LHS: Likelihood Specifications ---
    outcome_specs = metadata.outcomes
    n_outcomes = length(outcome_specs)
    opt_dict[:outcomes_N] = n_outcomes
    outcome_vars = [Symbol(spec[:var]) for spec in outcome_specs]
    opt_dict[:y_obs_vars] = outcome_vars
    y_N = size(data, 1)
    opt_dict[:y_N] = y_N

    likelihood_specs = []
    for spec in outcome_specs
        params = spec[:params]
        resolved_params = Dict{Symbol, Any}()
        for (k, v) in params
            if k == :family && v isa Symbol
                resolved_params[k] = string(v)
            elseif v isa Symbol
                if hasproperty(data, v)
                    resolved_params[k] = data[!, v]
                elseif isdefined(calling_module, v)
                    resolved_params[k] = Core.eval(calling_module, v)
                else
                    @warn "Could not resolve symbol ':$v' for likelihood parameter '$k'. It is not a column in the data or a variable in the current scope. Using symbol literally."
                    resolved_params[k] = v
                end
            elseif v isa Expr
                try
                    resolved_params[k] = Core.eval(calling_module, v)
                catch e
                    @warn "Could not evaluate likelihood parameter '$k' with value '$v'. Using literal value. Error: $e"
                    resolved_params[k] = v
                end
            else
                resolved_params[k] = v
            end
        end
        
        global_family = get(opt_dict, :model_family, "gaussian")
        resolved_params[:family] = get(resolved_params, :family, global_family)
        if resolved_params[:family] isa Symbol || resolved_params[:family] isa QuoteNode
            resolved_params[:family] = string(resolved_params[:family])
        end
        push!(likelihood_specs, resolved_params)
    end
    opt_dict[:likelihood_specs] = likelihood_specs

    # --- 3. Pre-process RHS: Dual Temporal/Seasonal Modules ---
    processed_modules = Dict{String, Any}()
    for (key, mod) in metadata.modules
        if mod.module_type == :temporal && get(mod.args, :model, "") isa Tuple
            models = mod.args[:model]
            vars = get(mod.args, :positional_args, [])
            if !isempty(vars)
                time_mod_args = merge(mod.args, Dict(:model => models[1]))
                processed_modules[key * "_time"] = (module_type=:temporal, args=time_mod_args)
            end
            if length(models) > 1 && length(vars) > 1
                season_mod_args = merge(mod.args, Dict(:positional_args => [vars[2]], :model => models[2]))
                processed_modules[key * "_season"] = (module_type=:seasonal, args=season_mod_args)
            end
        else
            processed_modules[key] = mod
        end
    end
    metadata = (outcomes=metadata.outcomes, modules=processed_modules, fixed_effects=metadata.fixed_effects, has_intercept=metadata.has_intercept)

    # --- 4. Main Module Processing Loop ---
    opt_dict[:add_intercept] = metadata.has_intercept
    if !haskey(opt_dict, :u_N); opt_dict[:u_N] = 0; end
    if !haskey(opt_dict, :u_idx); opt_dict[:u_idx] = nothing; end

    for (key_str, mod) in metadata.modules
        m_type = mod.module_type
        mod_data_dict = Dict{Symbol, Any}(:type => m_type, :params => mod.args)
        mod_data_dict[:variables] = haskey(mod.args, :positional_args) ? [string(arg) for arg in mod.args[:positional_args]] : []

        if haskey(mod_data_dict, :params)
            for (param_key, param_val) in mod_data_dict[:params]
                if param_val isa Expr || (param_val isa Symbol && param_val != :auto && isdefined(calling_module, param_val))
                    try
                        mod_data_dict[:params][param_key] = Core.eval(calling_module, param_val)
                    catch e
                        @error "Failed to evaluate formula parameter `$param_key` with expression `$param_val` in module `$key_str`."; rethrow(e)
                    end
                end
            end
        end

        if haskey(MODULE_PROCESSORS, m_type)
            processor_func = MODULE_PROCESSORS[m_type]
            processor_func(opt_dict, mod_data_dict, registries, hyperprior_registry)
        end

        if m_type in [:interaction_composition, :likelihood, :intercept, :nested, :fixed, :spacetime]
            continue
        end

        manifold_obj = resolve_technical_primitive(mod_data_dict, opt_dict, hyperprior_registry, :pcpriors)

        
        # Manifold Registration Block
        # This section handles the registration of both simple and composed manifolds.
        if manifold_obj isa ComposedManifold
            # For composed manifolds (from ⊗, ⊕, etc.), we register the object and pre-compute
            # combined indices or component metadata where applicable.
            
            local combined_indices = nothing
            local component_dims = nothing
            local component_indices = nothing

            if manifold_obj.operator == :kronecker_product
                # This logic specifically handles the common case of a spatiotemporal Kronecker product.
                # It computes the flattened 1D index for each observation.
                if haskey(opt_dict, :s_idx) && haskey(opt_dict, :t_idx) && haskey(opt_dict, :s_N)
                    s_idx = opt_dict[:s_idx]
                    t_idx = opt_dict[:t_idx]
                    s_N = opt_dict[:s_N]
                    
                    # Formula: (time_index - 1) * num_spatial_units + spatial_index
                    combined_indices = (t_idx .- 1) .* s_N .+ s_idx
                else
                    @warn "Could not compute combined indices for Kronecker product manifold '$key_str'. Spatial and temporal indices (s_idx, t_idx, s_N) not found in configuration."
                end
            elseif manifold_obj.operator == :direct_sum
                # For direct sums, we need to provide the dimensions and index vectors for each component.
                component_dims = []
                component_indices = []
                for comp in manifold_obj.components
                    # This is a simplified heuristic. A more robust implementation might need to inspect `comp` more deeply.
                    if comp isa Union{ICAR, Besag, BYM2, Leroux, SAR} # Spatial manifolds
                        push!(component_dims, get(opt_dict, :s_N, 0))
                        push!(component_indices, get(opt_dict, :s_idx, nothing))
                    elseif comp isa Union{AR1, RW1, RW2} # Temporal manifolds
                        push!(component_dims, get(opt_dict, :t_N, 0))
                        push!(component_indices, get(opt_dict, :t_idx, nothing))
                    end
                end
            end

            spec = (
                key = Symbol(key_str),
                domain = m_type,
                var = :none,
                indices = combined_indices,
                component_dims = component_dims,
                component_indices = component_indices,
                manifold_obj = manifold_obj,
                params = mod_data_dict[:params]
            )
            push!(manifolds_registry, spec)
            continue # Skip the simple manifold logic below
        end


        n_units = if m_type == :spatial; get(opt_dict, :s_N, 0)
                  elseif m_type == :temporal; get(opt_dict, :t_N, 0)
                  elseif m_type == :seasonal; get(opt_dict, :u_N, 0)
                  elseif m_type == :smooth; size(get(basis_matrices_registry, Symbol(join(mod_data_dict[:variables], "_")), zeros(0,0)), 2)
                  elseif m_type == :mixed; get(mod_data_dict[:params], :n_cat, 0)
                  else 0 end

        if n_units == 0 && !(manifold_obj isa GP); @warn "Could not determine size for manifold '$key_str'. Skipping."; continue; end

        template_info = build_structure_template(Symbol(lowercase(string(typeof(manifold_obj)))), n_units; W=get(opt_dict, :W, nothing))

        spec = (
            key = Symbol(key_str),
            domain = m_type,
            var = isempty(mod_data_dict[:variables]) ? :none : Symbol(mod_data_dict[:variables][1]),
            manifold_obj = manifold_obj,
            Q_template = template_info.matrix,
            scaling_factor = template_info.scaling_factor,
            params = mod_data_dict[:params]
        )
        push!(manifolds_registry, spec)
    end

    opt_dict[:manifolds] = manifolds_registry
    opt_dict[:basis_matrices] = basis_matrices_registry

    # --- 5. Final Configuration Assembly ---
    if get(opt_dict, :use_sv, false)
        if hasproperty(data, :s_x) && hasproperty(data, :s_y) && haskey(opt_dict, :t_idx)
            m_rff_sigma = get(opt_dict, :M_rff_sigma, 20)
            coords_st = hcat(data.s_x, data.s_y, opt_dict[:t_idx])
            W_sigma, b_sigma = generate_informed_rff_params(coords_st, m_rff_sigma)
            opt_dict[:vol_proj] = (coords_st * W_sigma) .+ b_sigma'
        else
             @warn "Stochastic volatility requires spatial coordinates (:s_x, :s_y) and a temporal index. SV disabled."
             opt_dict[:use_sv] = false
        end
    end

    fixed_effects_formula_part = join(metadata.fixed_effects, " + ")
    if metadata.has_intercept
        fixed_effects_formula_part = isempty(strip(fixed_effects_formula_part)) ? "1" : "1 + " * fixed_effects_formula_part
    end
    if !isempty(strip(fixed_effects_formula_part))
        opt_dict[:Xfixed] = create_fixed_design(fixed_effects_formula_part, data; contrasts=get(opt_dict, :contrasts, Dict()))
        opt_dict[:Xfixed_N] = size(opt_dict[:Xfixed], 2)
    else
        opt_dict[:Xfixed] = NamedArray(zeros(size(data, 1), 0), (1:size(data,1), Symbol[]))
        opt_dict[:Xfixed_N] = 0
    end

    if n_outcomes == 1
        opt_dict[:y_obs] = data[!, outcome_vars[1]]
        opt_dict[:model_arch] = get(opt_dict, :model_arch, "univariate")
    else
        opt_dict[:y_obs] = Matrix(data[!, outcome_vars])
        opt_dict[:model_arch] = "multivariate"
    end

    log_offsets_mat = zeros(y_N, n_outcomes)
    weights_mat = ones(y_N, n_outcomes)
    trials_mat = ones(Int, y_N, n_outcomes)

    for (k, spec) in enumerate(likelihood_specs)
        offset_val = get(spec, :log_offsets, get(spec, :offsets, nothing))
        if !isnothing(offset_val); log_offsets_mat[:, k] .= offset_val; end

        weight_val = get(spec, :weights, get(spec, :weight, nothing))
        if !isnothing(weight_val); weights_mat[:, k] .= weight_val; end

        trial_val = get(spec, :trials, get(spec, :trial, nothing))
        if !isnothing(trial_val); trials_mat[:, k] .= trial_val; end
    end

    opt_dict[:log_offset] = all(c -> c == log_offsets_mat[:, 1], eachcol(log_offsets_mat)) ? log_offsets_mat[:, 1] : log_offsets_mat
    opt_dict[:weights] = all(c -> c == weights_mat[:, 1], eachcol(weights_mat)) ? weights_mat[:, 1] : weights_mat
    opt_dict[:trials] = all(c -> c == trials_mat[:, 1], eachcol(trials_mat)) ? trials_mat[:, 1] : trials_mat

    return NamedTuple(opt_dict)
end


const BSTM_MODULE_KEYWORDS = Set([
    "spatial",
    "temporal",
    "seasonal",
    "smooth",
    "intercept",
    "dynamics",
    "eigen",
    "mixed",
    "nested",
    "interaction",
    "spacetime",
    "transform"
])


function process_intercept_module!(opt_dict, mod_data, registries, hyperpriors)
    #
    # Processes the `intercept()` module to handle the global intercept and its prior.
    #
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
    #
    # Processes the `spatial()` module. It identifies the spatial index, adjacency matrix (W),
    # and coordinates for GP models. It then resolves the manifold object and registers it.
    #
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
        if haskey(opt_dict, :s_x) && haskey(opt_dict, :s_y)
            s_N = get(opt_dict, :s_N, 0)
            if s_N > 0
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

function process_temporal_module!(opt_dict, mod_data, registries, hyperpriors)
    #
    # Processes the `temporal()` module. It identifies the primary time index variable,
    # runs `assign_time_units` to create discrete time steps, and registers the manifold.
    #
    data = opt_dict[:data]

    if !isempty(mod_data[:variables])
        t_var_sym = Symbol(mod_data[:variables][1])
        if hasproperty(data, t_var_sym)
            time_opts = Dict(:time_method => get(mod_data[:params], :time_method, "regular"))
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
    #
    # Processes a `seasonal()` module. This module is typically created by the pre-processor
    # from a dual `temporal()` call (e.g., `temporal(year, month)`).
    #
    data = opt_dict[:data]

    if !haskey(opt_dict, :u_idx); opt_dict[:u_idx] = nothing; end
    if !haskey(opt_dict, :u_N); opt_dict[:u_N] = 0; end

    if !isempty(mod_data[:variables])
        u_var_sym = Symbol(mod_data[:variables][1])
        if hasproperty(data, u_var_sym)
            opt_dict[:u_idx] = data[!, u_var_sym]
            opt_dict[:u_N] = length(unique(opt_dict[:u_idx]))
        else
            @warn "Seasonal index variable ':$u_var_sym' not found."
        end
    end
end

function process_smooth_module!(opt_dict, mod_data, registries, hyperpriors)
    #
    # Processes `smooth()` modules for non-linear covariate effects. It handles various
    # basis function models, continuous kernel models, and GMRF-on-bins models.
    #
    data = opt_dict[:data]
    basis_matrices_registry = registries[:basis_matrices]
    params = mod_data[:params]
    model_param = get(params, :model, "pspline")
    model_str = string(model_param)

    basis_models = ["pspline", "bspline", "tps", "rff", "fft", "moran", "spherical", "barycentric", "decay", "wavelet", "linear", "invdist", "kriging"]
    continuous_kernel_models = ["gp", "fitc", "svgp", "nystrom", "warp", "spde", "exponentialdecay"]
    gmrfs_on_bins_models = ["rw1", "rw2", "ar1", "icar", "besag", "cyclic"]

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
        if all(v -> hasproperty(data, Symbol(v)), mod_data[:variables])
            mod_data[:params][:coords] = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
        else
            @warn "Continuous kernel smooth specified, but coordinate variables not found in data. Manifold may be misspecified."
        end
    elseif model_str in gmrfs_on_bins_models
        vars = mod_data[:variables]
        if length(vars) == 1
            var_sym = Symbol(vars[1])
            nbins = get(mod_data[:params], :nbins, 20)
            _, indices = apply_discretization_logic(data[!, var_sym], nbins)
            mod_data[:params][:indices] = indices
            mod_data[:params][:n_cat] = length(unique(indices))
            mod_data[:type] = :mixed # Re-brand for main loop
        else
            @warn "GMRF smooth on $(join(vars, ",")) requires exactly 1 variable. Skipping."
            return
        end
    end
end

function process_dynamics_module!(opt_dict, mod_data, registries, hyperpriors)
    #
    # Processes the `dynamics()` module for mechanistic state-space models.
    #
    # This function currently acts as a placeholder. The actual manifold object is
    # resolved and registered in the main bstm_config loop.
end

function process_eigen_module!(opt_dict, mod_data, registries, hyperpriors)
    #
    # Processes the `eigen()` module for Bayesian PCA factor models. It calculates
    # the indices for the lower-triangular part of the Householder reflector matrix.
    #
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
    #
    # Processes the `mixed()` module for random effects (random intercepts/slopes).
    #
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
    mod_data[:variables] = [effect_var_str] # The effect variable is the primary variable
end

function process_nested_module!(opt_dict, mod_data, registries, hyperpriors)
    #
    # Processes the `nested()` supervisor module for multi-fidelity models.
    #
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
    :spacetime => process_spacetime_module!
);

