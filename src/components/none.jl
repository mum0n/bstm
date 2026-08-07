# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    NoneComponent <: ComponentModel

A placeholder component that has no effect on the model. It is used internally
by the parser for components that are handled globally (like spatiotemporal interactions)
or for situations where a component is syntactically required but no effect is desired.
"""
struct NoneComponent <: ComponentModel end

# Add to the central component constructor registry.
COMPONENT_CONSTRUCTORS[:none] = (p, params) -> NoneComponent()

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[NoneComponent] = :none

"""
    get_datastructures!(m_type::Type{<:NoneComponent}, M::Dict, mod_data::Dict)::Bool

A pass-through function for the `NoneComponent`. It performs no actions and returns `true`.
"""
function get_datastructures!(m_type::Type{<:NoneComponent}, M::Dict, mod_data::Dict)::Bool
    # This component does not require any data setup.
    return true
end

"""
    get_precomputes(m::NoneComponent, M::NamedTuple, mod_data::Dict)::NamedTuple

Returns an empty `NamedTuple` as the `NoneComponent` has no pre-computations.
"""
function get_precomputes(m::NoneComponent, M::NamedTuple, mod_data::Dict)::NamedTuple
    # No precomputes are necessary for a null component.
    return (;)
end

"""
    get_priors(m::NoneComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Returns an empty string as the `NoneComponent` has no parameters and thus no priors.
"""
function get_priors(m::NoneComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    # No priors are defined for a null component.
    return ""
end

"""
    get_updates(m::NoneComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Returns an empty string as the `NoneComponent` does not contribute to the linear predictor.
"""
function get_updates(m::NoneComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    # This component has no effect on the linear predictor.
    return ""
end

"""
    get_effects(m::NoneComponent, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Returns a zero-effect `NamedTuple` as the `NoneComponent` has no effect to reconstruct.
"""
function get_effects(m::NoneComponent, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    # The effect is always zero.
    zero_effect = (mean=zeros(N_total), lower=zeros(N_total), upper=zeros(N_total))
    return (structured=zero_effect,)
end
