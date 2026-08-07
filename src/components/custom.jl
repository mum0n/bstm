# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    Custom <: ComponentModel

A component for injecting user-defined arbitrary code into the `bstm` model.
This provides an "escape hatch" for advanced users who need to implement logic
not covered by the standard components.

# Fields
- `code_fragment::String`: A string containing valid Turing.jl model code.
- `params::Dict{Symbol, Any}`: A dictionary for user-defined parameters, which can
  include a `reconstruct_func` for posterior reconstruction.
"""
struct Custom <: ComponentModel
    code_fragment::String
    params::Dict{Symbol, Any}
end

# Add to the central component constructor registry.
COMPONENT_CONSTRUCTORS[:custom] = (p, params) -> begin
    code_fragment_val = get(params, :code_fragment, "")
    
    # The code fragment could be a variable in the user's scope, so it needs to be resolved.
    # This is handled in get_datastructures!, but we do a preliminary check here.
    # For the constructor, we just store the raw value.
    
    Custom(string(code_fragment_val), get(params, :params, Dict{Symbol, Any}()))
end

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[Custom] = :any

"""
    get_datastructures!(m_type::Type{<:Custom}, M::Dict, mod_data::Dict)::Bool

A pass-through function for the `Custom`. It resolves the `code_fragment`
if it is provided as a variable name.
"""
function get_datastructures!(m_type::Type{<:Custom}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    code_fragment_val = get(params, :code_fragment, "")
    
    local final_code_fragment::String
    if code_fragment_val isa Symbol
        calling_mod = get(M, :calling_module, Main)
        try
            final_code_fragment = Core.eval(calling_mod, code_fragment_val)
        catch e
            error("Could not evaluate `code_fragment` variable `$(code_fragment_val)`. Error: $e")
        end
    elseif code_fragment_val isa String
        final_code_fragment = code_fragment_val
    elseif code_fragment_val isa Expr
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
    
    # Store the resolved string back into the parameters for the constructor to use.
    params[:code_fragment] = final_code_fragment

    return true
end

"""
    get_precomputes(m::Custom, M::NamedTuple, mod_data::Dict)::NamedTuple

A pass-through function for the `Custom`. No pre-computations are performed
by the framework.
"""
function get_precomputes(m::Custom, M::NamedTuple, mod_data::Dict)::NamedTuple
    # No standard precomputes. The user's code fragment must be self-contained.
    return (;)
end

"""
    get_priors(m::Custom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Returns an empty string. Priors for a `Custom` must be defined within the
`code_fragment` provided by the user.
"""
function get_priors(m::Custom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    # The user is expected to define all priors within the code_fragment.
    return ""
end

"""
    get_updates(m::Custom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Injects the user-provided `code_fragment` directly into the model's update block.
"""
function get_updates(m::Custom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    user_code = m.code_fragment
    
    if isempty(strip(user_code))
        @warn "Custom component '$(spec.key)' was specified but the `code_fragment` is empty. This component will have no effect."
        return ""
    end

    # The entire user code is treated as an update block.
    # Turing doesn't distinguish between prior and update sections inside the @model macro,
    # so this is a valid approach. The user must ensure their code is correct and
    # that any new parameter names are unique to avoid collisions.
    return """
        # --- Custom Code Block for $(spec.key) ---
        $(user_code)
    """
end

"""
    get_effects(m::Custom, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Attempts to call a user-provided `reconstruct_func` for posterior reconstruction.
If no function is provided, it returns a zero-effect with a warning.
"""
function get_effects(m::Custom, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    reconstruct_func = get(m.params, :reconstruct_func, nothing)

    if !isnothing(reconstruct_func) && isa(reconstruct_func, Function)
        try
            return reconstruct_func(chain, M, n_samples, outcomes_N, p_names, spec, PS, N_total)
        catch e
            @error "The custom reconstruction function for component '$(spec.key)' failed."
            rethrow(e)
        end
    else
        @warn "Reconstruction for custom component '$(spec.key)' is not defined. Returning a zero-effect. Provide a `reconstruct_func` to the `custom()` module to enable posterior reconstruction."
        structured_effects = (mean=zeros(N_total), lower=zeros(N_total), upper=zeros(N_total))
        return (structured=structured_effects,)
    end
end
