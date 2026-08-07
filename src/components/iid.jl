# --- Implementation of the Explicit Interface for IID ---

"""
    struct IID <: ComponentModel

A simple Independent and Identically Distributed (IID) random effect, representing
unstructured noise or heterogeneity.

# Fields
- `sigma::UnivariateDistribution`: The prior for the standard deviation of the effect.
"""
struct IID <: ComponentModel
    sigma::UnivariateDistribution
end

# update COMPONENT_CONSTRUCTORS 
COMPONENT_CONSTRUCTORS[:iid] = (p, params) -> IID(p.sigma)

# helper to map to classes of methods (data structures), :any mean it can be used in many approaches
MODEL_TO_STRUCTURE_MAP[:iid] = :any


function get_datastructures!(m_type::Type{IID}, M::Dict, mod_data::Dict)::Bool
    # The IID component has no specific data requirements beyond what is handled
    # by the main configuration engine (e.g., inferring `s_N` or `t_N`).
    return true
end

function get_precomputes(m::IID, M::NamedTuple, mod_data::Dict)::NamedTuple
    # For an IID component, the precision matrix `Q` is the identity matrix.
    structure = get(mod_data, :type, :spatial)
    
    n = if structure == :spatial
        get(M, :s_N, 0)
    elseif structure == :temporal
        get(M, :t_N, 0)
    elseif structure == :mixed
        get(mod_data[:params], :n_cat, 0)
    else # smooth, etc.
        get(mod_data[:params], :nbins, 0)
    end

    if n == 0
        @warn "Could not determine dimension for IID component '$(mod_data[:key])'. The component will have no effect."
    end

    template = build_structure_template(:iid, n)
    return (Q_template=template.matrix, U=template.U, L=template.L, scaling_factor=template.scaling_factor)
end

function get_priors(m::IID, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    # Generates priors for `sigma` and the standard normal `raw` innovations.
    v = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = size(spec.Q_template, 1)
    is_multivariate = (arch == "multivariate")
    is_shared = get(spec.params, :shared, false)
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ $(_distribution_to_string(m.sigma))")
    end
    push!(priors_acc, "$(v.raw) ~ MvNormal(zeros(T, $(n_latent)), I)")
    return join(priors_acc, "\n    ")
end

function get_updates(m::IID, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    # Generates code to scale the raw innovations and add the effect to eta.
    v = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    index_var = if spec.structure == :spatial
        "s_idx"
    elseif spec.structure == :temporal
        "t_idx"
    elseif spec.structure == :mixed
        "mixed_idx_$(spec.var)"
    else
        string(spec.structure) * "_idx"
    end

    return """
    # --- IID Component: $(spec.key) ---
    $(v.latent) = $(v.raw) .* $(v.sigma)
    $(eta_target) .+= view($(v.latent), M.$(index_var))
    """
end

function get_effects(m::IID, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    # Reconstructs the posterior effect from the MCMC chain.
    structured_effects = []
    is_multivariate = outcomes_N > 1
    is_shared = get(spec.params, :shared, false)

    for k in 1:outcomes_N
        outcome_idx = is_multivariate ? k : nothing
        v = generate_full_variable_names(spec, M.model_arch, outcome_idx)
        sigma_var_name = (is_multivariate && is_shared) ? string(generate_full_variable_names(spec, M.model_arch, nothing).sigma) : string(v.sigma)
        
        sigma_samples = get_params_vector(chain, sigma_var_name, 1)
        raw_samples = get_params_vector(chain, string(v.raw), size(spec.Q_template, 1))
        
        effect_k = raw_samples .* sigma_samples
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end

