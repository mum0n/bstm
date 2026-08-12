"""
    Custom <: ComponentModel

A component for injecting user-defined arbitrary code into the `bstm` model.
This provides an "escape hatch" for advanced users who need to implement logic
not covered by the standard components.

# Version
v1.1.0 (2026-08-11)

# Mathematical Summary
This component does not have a fixed mathematical form. It is a "blank canvas"
where the user provides a `code_fragment` string containing valid Turing.jl model
code. This allows for the implementation of novel statistical relationships, custom
priors, or likelihoods not covered by the standard `bstm` components. The user is
responsible for defining the mathematical logic within this code fragment.

For example, a user could implement a custom non-linear function:
`y ~ Normal(a * x^b, sigma)`
by providing a `code_fragment` that defines priors for `a` and `b` and adds the
result to the linear predictor `eta`.

# Assumptions
- The user-provided `code_fragment` is syntactically correct Turing.jl code.
- The user is responsible for ensuring parameter names are unique to avoid
  collisions with other model components.
- The code must correctly modify the `eta` variable if it is intended to be part
  of the linear predictor.
- The user is responsible for ensuring the provided code is AD-compatible if using
  gradient-based samplers like NUTS.

# Best Use Case
Prototyping new models, implementing highly specialized statistical processes, or
integrating logic from other libraries directly into a `bstm` model without needing
to create a full custom component file from scratch. It serves as a powerful didactic
tool for learning how `bstm` constructs models.

# Inputs
- **Required**:
  - `code_fragment`: A `String` containing valid Turing.jl model code. This can be
    passed directly or as a variable name (Symbol) that resolves to a string in the
    calling module.
- **Optional**:
  - `params`: A `Dict{Symbol, Any}` for user-defined parameters. A special key,
    `:reconstruct_func`, can be included. Its value should be a function that
    implements the posterior reconstruction logic for the custom component.

# Outputs (Parameter Names)
- All parameter names are defined by the user within the `code_fragment`. There are
  no standard output parameter names for this component.

# Key References
- Turing.jl Documentation: https://turing.ml/dev/docs/using-turing/
"""
struct Custom <: ComponentModel
    code_fragment::String
    params::Dict{Symbol, Any}
end

COMPONENT_TYPE_REGISTRY[:custom] = Custom

COMPONENT_CONSTRUCTORS[:custom] = (p, params) -> begin
    code_fragment_val = get(params, :code_fragment, "")
    Custom(string(code_fragment_val), get(params, :params, Dict{Symbol, Any}()))
end

MODEL_TO_STRUCTURE_MAP[:custom] = :any


"""
    get_datastructures!(m_type::Type{<:Custom}, M::Dict, mod_data::Dict)::Bool

Resolves the `code_fragment` string from the formula, evaluating it if it is
provided as a variable name or expression.

# Security Warning
This function uses `Core.eval`, which can execute arbitrary code. Only use this
component with trusted formula inputs.
"""
function get_datastructures!(
    m_type::Type{<:Custom}, M::Dict, mod_data::Dict
)::Bool
    params = mod_data[:params]
    code_fragment_val = get(params, :code_fragment, "")
    
    local final_code_fragment::String
    if code_fragment_val isa Symbol
        @warn "Evaluating `code_fragment` from symbol `$(code_fragment_val)`. Ensure this is from a trusted source."
        calling_mod = get(M, :calling_module, Main)
        try
            final_code_fragment = Core.eval(calling_mod, code_fragment_val)
        catch e
            error("Could not evaluate `code_fragment` variable `$(code_fragment_val)`. Error: $e")
        end
    elseif code_fragment_val isa String
        final_code_fragment = code_fragment_val
    elseif code_fragment_val isa Expr
        @warn "Evaluating `code_fragment` from expression. Ensure this is from a trusted source."
        try
            final_code_fragment = Core.eval(get(M, :calling_module, Main), code_fragment_val)
        catch e
            error("Could not evaluate `code_fragment` expression `$(code_fragment_val)`. Error: $e")
        end
    else
        error("Unsupported type for `code_fragment`: $(typeof(code_fragment_val))")
    end

    if !(final_code_fragment isa String)
        error("`code_fragment` must resolve to a String. Got: $(typeof(final_code_fragment))")
    end
    
    params[:code_fragment] = final_code_fragment
    return true
end


"""
    get_precomputes(m::Custom, M::NamedTuple, mod_data::Dict)::NamedTuple

Returns an empty `NamedTuple`. All pre-computation must be handled within the
user's code fragment.
"""
function get_precomputes(m::Custom, M::NamedTuple, mod_data::Dict)::NamedTuple
    return (;)
end

"""
    get_priors(m::Custom, spec::NamedTuple, arch::String, outcome_idx, M)::String

Returns an empty string. All priors must be defined within the user's
`code_fragment`.
"""
function get_priors(
    m::Custom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    return ""
end

"""
    get_updates(m::Custom, spec::NamedTuple, arch::String, outcome_idx, M)::String

Injects the user-provided `code_fragment` directly into the model's update block,
wrapped in a `let` block to prevent variable scope leakage.
"""
function get_updates(
    m::Custom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    user_code = m.code_fragment
    
    if isempty(strip(user_code))
        @warn "Custom component '$(spec.key)' has an empty `code_fragment` and will have no effect."
        return ""
    end

    return """
        # --- Custom Code Block for $(spec.key) ---
        let
            $(user_code)
        end
    """
end



"""
    get_effects(m::Custom, chain, M::NamedTuple, ...)::NamedTuple

Calls a user-provided `reconstruct_func` for posterior reconstruction. If no
function is provided, it returns a zero-effect with a warning. The user's function
will receive the full list of parameter names and the model's multivariate status
to allow for correct use of `_find_parameter`.
"""
function get_effects(
    m::Custom, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    reconstruct_func = get(m.params, :reconstruct_func, nothing)

    if !isnothing(reconstruct_func) && isa(reconstruct_func, Function)
        try
            # Pass additional arguments to the user's function to enable
            # correct use of _find_parameter.
            p_names_vec = string.(FlexiChains.parameters(chain))
            is_multivariate_model = M.model_arch == "multivariate"
            
            return reconstruct_func(
                chain, M, n_samples, outcomes_N, spec, PS, N_total,
                p_names_vec, is_multivariate_model
            )
        catch e
            @error "The custom reconstruction function for component '$(spec.key)' failed."
            rethrow(e)
        end
    else
        @warn "Reconstruction for custom component '$(spec.key)' is not defined. " *
              "Returning a zero-effect. Provide a `reconstruct_func` to the " *
              "`custom()` module to enable posterior reconstruction."
        structured_effects = [
            zeros(Float64, N_total, n_samples) for _ in 1:outcomes_N
        ]
        return (structured=structured_effects, noisy=structured_effects)
    end
end
