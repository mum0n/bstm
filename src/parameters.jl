# =============================================================================
# ParamRegistry: Centralized Parameter and Variable Name Architecture for BSTM
# =============================================================================

"""
    ParamDescriptor

Detailed semantic metadata for a single model parameter / random variable.

# Fields
- `symbol::Symbol`: The canonical Julia symbol used in Turing code (e.g., `:sigma_s_idx`, `:innovations_s_idx_1`).
- `name::String`: Canonical string representation of the symbol.
- `component_key::Symbol`: Owner component key (e.g., `:s_idx`, `:t_idx`, `:intercept`, `:fixed`, `:st_interaction`, `:likelihood`).
- `role::Symbol`: Semantic role (e.g., `:sigma`, `:rho`, `:innovations`, `:latent`, `:W1`, `:b1`, `:intercept`, `:fixed_coef`, `:scale`).
- `outcome_idx::Union{Int, Nothing}`: 1-indexed outcome number for multivariate models, or `nothing` if univariate / shared.
- `is_shared::Bool`: `true` if this hyperparameter is shared across outcomes in a multivariate model.
- `shape::Tuple{Vararg{Int}}`: Expected dimension tuple (e.g., `(1,)` for scalar, `(20,)` for vector, `(10, 20)` for matrix).
- `prior::Union{Distribution, Nothing}`: The prior distribution object if known.
"""
struct ParamDescriptor
    symbol::Symbol
    name::String
    component_key::Symbol
    role::Symbol
    outcome_idx::Union{Int, Nothing}
    is_shared::Bool
    shape::Tuple{Vararg{Int}}
    prior::Union{Distribution, Nothing}
end

# Convenience constructor with defaults
function ParamDescriptor(
    symbol::Symbol;
    component_key::Symbol = :unknown,
    role::Symbol = :param,
    outcome_idx::Union{Int, Nothing} = nothing,
    is_shared::Bool = false,
    shape::Tuple{Vararg{Int}} = (1,),
    prior::Union{Distribution, Nothing} = nothing
)
    return ParamDescriptor(
        symbol,
        string(symbol),
        component_key,
        role,
        outcome_idx,
        is_shared,
        shape,
        prior
    )
end

"""
    ParamRegistry

Central registry that stores parameter descriptors, semantic hierarchies, and chain index mappings.
Provides canonical name resolution and sample extraction across all MCMC chain formats.
"""
mutable struct ParamRegistry
    names::Vector{String}                                              # Canonical string names
    descriptors::Dict{Symbol, ParamDescriptor}                         # Symbol -> ParamDescriptor
    by_component::Dict{Symbol, Dict{Symbol, Vector{ParamDescriptor}}}  # component_key -> role -> [descriptors]
    by_base::Dict{String, Vector{String}}                              # base name -> list of matching full names
    name_to_key::Dict{String, Any}                                     # String/Symbol in chain -> actual indexing key
end

# Default empty constructor
function ParamRegistry()
    return ParamRegistry(
        String[],
        Dict{Symbol, ParamDescriptor}(),
        Dict{Symbol, Dict{Symbol, Vector{ParamDescriptor}}}(),
        Dict{String, Vector{String}}(),
        Dict{String, Any}()
    )
end

# Backward-compatible 3-argument constructor
function ParamRegistry(names::Vector{String}, by_base::Dict{String, Vector{String}}, name_to_key::Dict{String, Any})
    reg = ParamRegistry()
    reg.names = names
    reg.by_base = by_base
    reg.name_to_key = name_to_key
    for n in names
        sym = Symbol(n)
        reg.descriptors[sym] = ParamDescriptor(sym)
    end
    return reg
end

# =============================================================================
# Registration & Mutation Helpers
# =============================================================================

"""
    add_descriptor!(reg::ParamRegistry, desc::ParamDescriptor)

Registers a `ParamDescriptor` in the registry and updates all indexing tables.
"""
function add_descriptor!(reg::ParamRegistry, desc::ParamDescriptor)
    reg.descriptors[desc.symbol] = desc
    
    if !(desc.name in reg.names)
        push!(reg.names, desc.name)
    end

    # Register under by_component hierarchy: component_key -> role -> [descriptors]
    if !haskey(reg.by_component, desc.component_key)
        reg.by_component[desc.component_key] = Dict{Symbol, Vector{ParamDescriptor}}()
    end
    role_dict = reg.by_component[desc.component_key]
    if !haskey(role_dict, desc.role)
        role_dict[desc.role] = ParamDescriptor[]
    end
    # Replace existing with same symbol or append
    existing_idx = findfirst(d -> d.symbol == desc.symbol, role_dict[desc.role])
    if isnothing(existing_idx)
        push!(role_dict[desc.role], desc)
    else
        role_dict[desc.role][existing_idx] = desc
    end

    # Index by canonical base (e.g. before '[')
    base = first(Base.split(desc.name, '['))
    if !haskey(reg.by_base, base)
        reg.by_base[base] = String[]
    end
    if !(desc.name in reg.by_base[base])
        push!(reg.by_base[base], desc.name)
    end

    # Also index by underscore-numeric suffix base if present (e.g., intercept_1 -> intercept)
    parts = Base.split(desc.name, '_')
    if length(parts) > 1 && all(isdigit.(collect(parts[end])))
        underscore_base = join(parts[1:end-1], "_")
        if !haskey(reg.by_base, underscore_base)
            reg.by_base[underscore_base] = String[]
        end
        if !(desc.name in reg.by_base[underscore_base])
            push!(reg.by_base[underscore_base], desc.name)
        end
    end

    # Map name to indexing key
    reg.name_to_key[desc.name] = desc.symbol
    return reg
end

# =============================================================================
# Multi-Source ParamRegistry Builders
# =============================================================================

"""
    build_param_registry(M::NamedTuple)

Builds a static `ParamRegistry` from the model configuration `M`.
"""
function build_param_registry(M::NamedTuple)
    reg = ParamRegistry()
    arch = get(M, :model_arch, "univariate")
    is_multivariate = arch == "multivariate"
    outcomes_N = get(M, :outcomes_N, 1)

    # 1. Intercept
    if get(M, :add_intercept, false)
        intercept_prior = get(M, :intercept_prior, Normal(0, 5))
        if is_multivariate
            for k in 1:outcomes_N
                sym = Symbol("intercept_$(k)")
                add_descriptor!(reg, ParamDescriptor(
                    sym;
                    component_key = :intercept,
                    role = :intercept,
                    outcome_idx = k,
                    is_shared = false,
                    shape = (1,),
                    prior = intercept_prior
                ))
            end
        else
            add_descriptor!(reg, ParamDescriptor(
                :intercept;
                component_key = :intercept,
                role = :intercept,
                outcome_idx = nothing,
                is_shared = true,
                shape = (1,),
                prior = intercept_prior
            ))
        end
    end

    # 2. Fixed Effects
    if get(M, :Xfixed_N, 0) > 0
        n_fixed = M.Xfixed_N
        if is_multivariate
            add_descriptor!(reg, ParamDescriptor(
                :beta_flat;
                component_key = :fixed,
                role = :fixed_coef,
                outcome_idx = nothing,
                is_shared = false,
                shape = (n_fixed * outcomes_N,)
            ))
        else
            add_descriptor!(reg, ParamDescriptor(
                :beta;
                component_key = :fixed,
                role = :fixed_coef,
                outcome_idx = nothing,
                is_shared = true,
                shape = (n_fixed,)
            ))
        end
    end

    # 3. Model Components
    if haskey(M, :components) && !isempty(M.components)
        for spec in M.components
            comp_obj = spec.component_obj
            comp_key = spec.key
            is_shared = get(spec.params, :shared, false)

            # Detect all hyperparameters on component struct
            for f in fieldnames(typeof(comp_obj))
                val = getfield(comp_obj, f)
                if val isa Distribution
                    if is_multivariate && !is_shared
                        for k in 1:outcomes_N
                            sym = Symbol("$(f)_$(comp_key)_$(k)")
                            add_descriptor!(reg, ParamDescriptor(
                                sym;
                                component_key = comp_key,
                                role = f,
                                outcome_idx = k,
                                is_shared = false,
                                shape = (1,),
                                prior = val
                            ))
                        end
                    else
                        sym = Symbol("$(f)_$(comp_key)")
                        add_descriptor!(reg, ParamDescriptor(
                            sym;
                            component_key = comp_key,
                            role = f,
                            outcome_idx = is_multivariate ? 1 : nothing,
                            is_shared = is_shared,
                            shape = (1,),
                            prior = val
                        ))
                    end
                end
            end

            # Detect latent fields & innovations
            n_latent = if hasproperty(spec.hyper, :n_latent)
                spec.hyper.n_latent
            elseif hasproperty(spec.hyper, :in_dim) && hasproperty(comp_obj, :hidden_dim) && hasproperty(comp_obj, :nbins)
                comp_obj.nbins
            else
                0
            end

            latent_roles = [:ure, :sre, :W1, :b1, :W2, :W, :b, :v_unscaled, :thresh_unscaled, :amplitude_unscaled]
            for role in latent_roles
                for k in 1:outcomes_N
                    outcome_k = is_multivariate ? k : nothing
                    p_names = generate_full_variable_names(spec, arch, outcome_k)
                    if hasproperty(p_names, role)
                        sym = getfield(p_names, role)
                        shape_val = if role in [:ure, :sre]
                            (n_latent > 0 ? n_latent : 1,)
                        elseif role == :W1 && hasproperty(comp_obj, :hidden_dim) && hasproperty(spec.hyper, :in_dim)
                            (spec.hyper.in_dim * comp_obj.hidden_dim,)
                        elseif role == :b1 && hasproperty(comp_obj, :hidden_dim)
                            (comp_obj.hidden_dim,)
                        elseif role == :W2 && hasproperty(comp_obj, :hidden_dim) && hasproperty(comp_obj, :nbins)
                            (comp_obj.hidden_dim * comp_obj.nbins,)
                        else
                            (1,)
                        end

                        add_descriptor!(reg, ParamDescriptor(
                            sym;
                            component_key = comp_key,
                            role = role,
                            outcome_idx = outcome_k,
                            is_shared = false,
                            shape = shape_val
                        ))
                    end
                end
            end
        end
    end

    # 4. Spatiotemporal Interaction
    if get(M, :model_st, "none") != "none"
        st_prior = if haskey(M, :sigma_st_interaction_prior)
            M.sigma_st_interaction_prior
        elseif haskey(M, :st_interaction_sigma_prior)
            M.st_interaction_sigma_prior
        else
            Exponential(1.0)
        end
        s_N = get(M, :s_N, 1)
        t_N = get(M, :t_N, 1)
        add_descriptor!(reg, ParamDescriptor(
            :sigma_st_interaction;
            component_key = :st_interaction,
            role = :sigma,
            outcome_idx = nothing,
            is_shared = false,
            shape = is_multivariate ? (outcomes_N,) : (1,),
            prior = st_prior
        ))
        add_descriptor!(reg, ParamDescriptor(
            :ure_st_interaction;
            component_key = :st_interaction,
            role = :ure,
            outcome_idx = nothing,
            is_shared = false,
            shape = is_multivariate ? (s_N * t_N * outcomes_N,) : (s_N * t_N,)
        ))
    end

    # 5. Spectral Orientation (Householder Reflection)
    if is_multivariate && get(M, :spectral_orientation, false)
        add_descriptor!(reg, ParamDescriptor(
            :v_unscaled_reflection;
            component_key = :spectral_orientation,
            role = :v_unscaled,
            outcome_idx = nothing,
            is_shared = true,
            shape = (outcomes_N,)
        ))
    end

    # 6. Likelihood parameters
    if is_multivariate && outcomes_N > 1
        add_descriptor!(reg, ParamDescriptor(
            :L_corr;
            component_key = :likelihood,
            role = :correlation_cholesky,
            outcome_idx = nothing,
            is_shared = true,
            shape = (outcomes_N, outcomes_N)
        ))
    end

    return reg
end

"""
    build_param_registry(sample::Any, M::Union{NamedTuple, Nothing}=nothing)

Builds or augments a `ParamRegistry` from a prior predictive draw (e.g., `rand(model)` returning `NamedTuple` or `VarNamedTuple`).
"""
function build_param_registry(sample::Any, M::Union{NamedTuple, Nothing}=nothing)
    reg = !isnothing(M) ? build_param_registry(M) : ParamRegistry()
    return calibrate_param_registry(reg, sample)
end

"""
    build_param_registry(model::DynamicPPL.Model)

Builds a `ParamRegistry` directly from an instantiated Turing `DynamicPPL.Model`.
"""
function build_param_registry(model::DynamicPPL.Model)
    if hasproperty(model.args, :M)
        reg = build_param_registry(model.args.M)
        # Check VarInfo if available
        try
            vi = DynamicPPL.VarInfo(model)
            for vn in keys(vi)
                sym = _get_varname_symbol(vn)
                if !haskey(reg.descriptors, sym)
                    add_descriptor!(reg, ParamDescriptor(sym))
                end
            end
        catch
        end
        return reg
    else
        reg = ParamRegistry()
        try
            vi = DynamicPPL.VarInfo(model)
            for vn in keys(vi)
                sym = _get_varname_symbol(vn)
                add_descriptor!(reg, ParamDescriptor(sym))
            end
        catch
        end
        return reg
    end
end

"""
    build_param_registry(chain::Any)

Builds a `ParamRegistry` from an MCMC chain (e.g., FlexiChain, MCMCChains.Chains, DataFrame, or Dict).
"""
function build_param_registry(chain::Any)
    reg = ParamRegistry()
    raw_names = _extract_chain_column_names(chain)
    names_str = string.(raw_names)

    for (i, nstr) in enumerate(names_str)
        orig_key = raw_names[i]
        key_for_indexing = orig_key isa String ? Symbol(nstr) : orig_key
        sym = Symbol(nstr)

        base = first(Base.split(nstr, '['))
        if !haskey(reg.by_base, base); reg.by_base[base] = String[]; end
        push!(reg.by_base[base], nstr)

        parts = Base.split(nstr, '_')
        if length(parts) > 1 && all(isdigit.(collect(parts[end])))
            underscore_base = join(parts[1:end-1], "_")
            if !haskey(reg.by_base, underscore_base); reg.by_base[underscore_base] = String[]; end
            push!(reg.by_base[underscore_base], nstr)
        end

        reg.name_to_key[nstr] = key_for_indexing
        if !(nstr in reg.names); push!(reg.names, nstr); end
        
        reg.descriptors[sym] = ParamDescriptor(sym)
    end

    return reg
end

# =============================================================================
# Calibration Helper
# =============================================================================

"""
    calibrate_param_registry(reg::ParamRegistry, sample::Any)

Updates descriptors and shapes in `reg` based on actual realized values in `sample` (from `rand(model)`).
Supports `NamedTuple`, `DynamicPPL.VarNamedTuple`, `AbstractDict`, etc.
"""
function calibrate_param_registry(reg::ParamRegistry, sample::Any)
    pairs_iter = try
        pairs(sample)
    catch
        try
            pairs(NamedTuple(sample))
        catch
            Dict(k => getproperty(sample, k) for k in propertynames(sample))
        end
    end

    for (sym, val) in pairs_iter
        val_shape = val isa AbstractArray ? size(val) : (1,)
        nstr = string(sym)
        sym_key = Symbol(sym)

        if haskey(reg.descriptors, sym_key)
            existing = reg.descriptors[sym_key]
            reg.descriptors[sym_key] = ParamDescriptor(
                existing.symbol,
                existing.name,
                existing.component_key,
                existing.role,
                existing.outcome_idx,
                existing.is_shared,
                val_shape,
                existing.prior
            )
        else
            # New parameter discovered in sample
            add_descriptor!(reg, ParamDescriptor(
                sym_key;
                shape = val_shape
            ))
        end

        reg.name_to_key[nstr] = sym_key
    end
    return reg
end

# =============================================================================
# Universal Sample Extraction API
# =============================================================================

"""
    get_samples(chain, reg::ParamRegistry, component_key::Symbol, role::Symbol; outcome=nothing, expected_len=nothing)

Extracts an `(n_samples, param_dim)` matrix of posterior samples for a specific component and semantic role.
Automatically handles scalar, vector, and matrix parameters across FlexiChains, MCMCChains, and DataFrames.
"""
function get_samples(
    chain,
    reg::ParamRegistry,
    component_key::Symbol,
    role::Symbol;
    outcome::Union{Int, Nothing} = nothing,
    expected_len::Union{Int, Nothing} = nothing
)
    # Map role to canonical role if alias was used
    canonical_role = if role in [:innovations, :innov, :raw, :unscaled, :iid]
        :ure
    elseif role in [:latent, :struct]
        :sre
    elseif role in [:unconstrained_rho, :rho_unconstrained]
        :rho_unconstrained
    elseif role in [:unconstrained_rho1, :rho1_unconstrained]
        :rho1_unconstrained
    elseif role in [:unconstrained_rho2, :rho2_unconstrained]
        :rho2_unconstrained
    elseif role in [:unconstrained_sigma1, :sigma1_unconstrained]
        :sigma1_unconstrained
    elseif role in [:unconstrained_sigma2, :sigma2_unconstrained]
        :sigma2_unconstrained
    elseif role in [:v_raw, :v_unscaled]
        :v_unscaled
    elseif role in [:thresh_raw, :thresh_unscaled]
        :thresh_unscaled
    elseif role in [:amplitude_raw, :amplitude_unscaled]
        :amplitude_unscaled
    else
        role
    end

    # 1. Lookup matching descriptor from by_component
    has_canonical = haskey(reg.by_component, component_key) && haskey(reg.by_component[component_key], canonical_role)
    has_orig = haskey(reg.by_component, component_key) && haskey(reg.by_component[component_key], role)

    if !has_canonical && !has_orig
        # Fallback to base name search
        fallback_name = isnothing(outcome) ? "$(role)_$(component_key)" : "$(role)_$(component_key)_$(outcome)"
        target_name = find_chain_param(reg, fallback_name; outcome_idx = outcome)
        if isempty(target_name)
            # Try with canonical role name as well
            fallback_canonical = isnothing(outcome) ? "$(canonical_role)_$(component_key)" : "$(canonical_role)_$(component_key)_$(outcome)"
            target_name = find_chain_param(reg, fallback_canonical; outcome_idx = outcome)
        end
        if isempty(target_name)
            error("Parameter with component :$(component_key) and role :$(role) (outcome: $(outcome)) not found in ParamRegistry.")
        end
        return get_param_samples(chain, reg, target_name; expected_len = expected_len)
    end

    lookup_role = has_canonical ? canonical_role : role
    candidates = reg.by_component[component_key][lookup_role]
    selected_desc = nothing

    if !isnothing(outcome)
        # Find candidate with matching outcome_idx
        for d in candidates
            if d.outcome_idx == outcome
                selected_desc = d
                break
            end
        end
    end

    if isnothing(selected_desc) && !isempty(candidates)
        selected_desc = first(candidates)
    end

    if isnothing(selected_desc)
        error("No suitable descriptor found for component :$(component_key), role :$(role), outcome $(outcome).")
    end

    exp_len = isnothing(expected_len) ? prod(selected_desc.shape) : expected_len
    return get_param_samples(chain, reg, selected_desc.name; expected_len = exp_len)
end

"""
    get_param_samples(chain, reg::ParamRegistry, param_name::Union{String, Symbol}; expected_len::Union{Int, Nothing}=nothing)

Extracts an `(n_samples, param_dim)` matrix of posterior samples for a specific parameter name.
"""
function get_param_samples(
    chain,
    reg::ParamRegistry,
    param_name::Union{String, Symbol};
    expected_len::Union{Int, Nothing} = nothing
)
    nstr = string(param_name)
    actual_col = find_chain_param(reg, nstr)
    lookup_key = !isempty(actual_col) ? chain_index_key(reg, actual_col) : Symbol(nstr)

    return _extract_samples_from_chain(chain, lookup_key, nstr; expected_len = expected_len)
end

# =============================================================================
# Name Resolution & Lookup Helpers
# =============================================================================

"""
    find_chain_param(reg::ParamRegistry, requested::String; outcome_idx::Union{Int, Nothing}=nothing)

Finds the best matching actual chain column name for a requested canonical name.
Returns `String` matching actual name or `""` if not found.
"""
function find_chain_param(reg::ParamRegistry, requested::String; outcome_idx::Union{Int, Nothing}=nothing)
    # 1) Exact match in registered names
    if requested in reg.names
        return requested
    end

    # 2) If outcome index provided, try "_k" or "[k]"
    if !isnothing(outcome_idx)
        s1 = "$(requested)_$(outcome_idx)"
        s2 = "$(requested)[$(outcome_idx)]"
        if s1 in reg.names; return s1; end
        if s2 in reg.names; return s2; end
    end

    # 3) Base match in by_base
    base = first(Base.split(requested, '['))
    if haskey(reg.by_base, base)
        candidates = reg.by_base[base]

        # Plain candidate equal to base
        if base in candidates; return base; end

        # Bracketed candidates
        for c in candidates
            if startswith(c, base * "["); return c; end
        end

        # Underscore numeric suffix candidates
        for c in candidates
            if startswith(c, base * "_"); return c; end
        end

        return first(candidates)
    end

    # 4) Alias transformations
    aliases = String[]
    if requested == "beta"
        push!(aliases, "Xfixed_beta_prop", "beta_prop", "Xfixed_beta")
    elseif requested == "beta_flat"
        push!(aliases, "Xfixed_beta_prop_flat", "beta_prop_flat")
    elseif requested == "Xfixed_beta_prop"
        push!(aliases, "beta")
    elseif requested == "Xfixed_beta_prop_flat"
        push!(aliases, "beta_flat")
    elseif startswith(requested, "ure_")
        push!(aliases, replace(requested, r"^ure_" => "innovations_"), replace(requested, r"^ure_" => "innov_"), replace(requested, r"^ure_" => "raw_"))
    elseif startswith(requested, "sre_")
        push!(aliases, replace(requested, r"^sre_" => "latent_"), replace(requested, r"^sre_" => "struct_"))
    elseif requested == "sigma_st_interaction"
        push!(aliases, "st_interaction_sigma")
    elseif requested == "ure_st_interaction"
        push!(aliases, "st_interaction_raw")
    end

    for a in aliases
        res = find_chain_param(reg, a; outcome_idx = outcome_idx)
        if !isempty(res)
            return res
        end
    end

    # 5) Substring fallback
    for n in reg.names
        if occursin(requested, n)
            return n
        end
    end

    return ""
end

"""
    chain_index_key(reg::ParamRegistry, actual_name::String)

Returns the exact indexing key (Symbol or String) for indexing into the MCMC chain.
"""
function chain_index_key(reg::ParamRegistry, actual_name::String)
    return get(reg.name_to_key, actual_name, Symbol(actual_name))
end

# =============================================================================
# Low-Level Chain Extraction Dispatch Helpers
# =============================================================================

function _extract_chain_column_names(chain::Any)
    if chain isa DataFrame
        return names(chain)
    elseif chain isa Dict
        return collect(keys(chain))
    elseif chain isa NamedTuple
        return collect(keys(chain))
    else
        # FlexiChain / MCMCChains.Chains
        try
            return names(DataFrame(chain))
        catch
            try
                return collect(keys(chain))
            catch
                return String[]
            end
        end
    end
end

function _extract_samples_from_chain(chain::Dict, key::Any, nstr::String; expected_len=nothing)
    # Mock chain dictionary support for testing
    sym_key = Symbol(key)
    if haskey(chain, sym_key)
        data = chain[sym_key]
        if data isa Matrix
            # [dim, n_samples] or [n_samples, dim]
            if size(data, 1) == 1 && size(data, 2) > 1
                return reshape(data, :, 1)
            elseif !isnothing(expected_len) && size(data, 1) == expected_len
                return Matrix(data')
            else
                return data
            end
        elseif data isa Vector
            return reshape(data, :, 1)
        else
            return fill(Float64(data), 1, 1)
        end
    end
    error("Parameter :$(sym_key) not found in mock chain.")
end

function _extract_samples_from_chain(chain::NamedTuple, key::Any, nstr::String; expected_len=nothing)
    sym_key = Symbol(key)
    if hasproperty(chain, sym_key)
        data = getproperty(chain, sym_key)
        if data isa Matrix
            if size(data, 1) == 1 && size(data, 2) > 1
                return reshape(data, :, 1)
            elseif !isnothing(expected_len) && size(data, 1) == expected_len
                return Matrix(data')
            else
                return data
            end
        elseif data isa Vector
            return reshape(data, :, 1)
        else
            return fill(Float64(data), 1, 1)
        end
    end
    error("Parameter :$(sym_key) not found in chain NamedTuple.")
end

function _extract_samples_from_chain(chain::Any, key::Any, nstr::String; expected_len=nothing)
    # Use BSTM's core extraction helper _extract_flexichain_param_samples
    base_name = first(Base.split(nstr, '['))
    return _extract_flexichain_param_samples(chain, base_name)
end