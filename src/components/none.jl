"""
    None <: ComponentModel

A placeholder component that has no effect on the model. It is used internally
by the parser for components that are handled globally (like spatiotemporal
interactions) or for situations where a component is syntactically required but no
effect is desired.

# Version
v1.1.0 (2026-08-19)

# Mathematical Summary
This component has no mathematical form and adds nothing to the model's linear
predictor or its likelihood. It is equivalent to adding zero.

# Inputs
- **Required**: None.
- **Optional**: None.

# Outputs (Parameter Names)
- This component produces no parameters.

# Key References
- This component implements the Null Object pattern.
 
"""
struct None <: ComponentModel end

COMPONENT_TYPE_REGISTRY[:none] = None
COMPONENT_CONSTRUCTORS[:none] = (p, params) -> None()
MODEL_TO_STRUCTURE_MAP[:none] = :none 

"""
    get_precomputes(m::None, M::NamedTuple, mod_data::Dict)::NamedTuple

Returns an empty `NamedTuple` as the `None` component has no pre-computations.
"""
function get_precomputes(m::None, M::NamedTuple, mod_data::Dict)::NamedTuple
    # No precomputes are necessary for a null component.
    return (;)
end

"""
    get_priors(m::None, spec::NamedTuple, arch::String, outcome_idx, M)::String

Returns an empty string as the `None` component has no parameters and thus no priors.
"""
function get_priors(
    m::None, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    # No priors are defined for a null component.
    return ""
end

"""
    get_updates(m::None, spec::NamedTuple, arch::String, outcome_idx, M)::String

Returns an empty string as the `None` component does not contribute to the linear
predictor.
"""
function get_updates(
    m::None, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    # This component has no effect on the linear predictor.
    return ""
end

"""
    get_effects(m::None, chain, spec::NamedTuple, M::NamedTuple, PS)

Returns a zero-effect `NamedTuple` as the `None` component has no effect to
reconstruct. This version is CPU-only and uses modern chain accessors.
"""
function get_effects(
    m::None, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = if occursin("FlexiChain", string(typeof(chain)))
        size(chain, 1) * FlexiChains.nchains(chain)
    else
        size(chain, 1) * size(chain, 3)
    end
    outcomes_N = M.outcomes_N
    N_total = M.y_N + (isnothing(PS) ? 0 : size(PS.data, 1))

    # The effect is always zero.
    structured_effects = [zeros(Float64, N_total, n_samples) for _ in 1:outcomes_N]
    return (structured=structured_effects, noisy=structured_effects)
end 
