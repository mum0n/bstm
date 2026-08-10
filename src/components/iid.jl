 """
    IID <: ComponentModel

A simple Independent and Identically Distributed (IID) random effect, representing
unstructured noise or heterogeneity. Each latent effect is drawn independently from
the same normal distribution.

# Version
v1.0.2 (2026-08-10)

# Mathematical Summary
The IID component models a latent field \$\\phi\$ where each element \$\\phi_i\$ is drawn
independently from a zero-mean normal distribution with a shared standard deviation
\$\\sigma\$:
\$\\phi_i \\sim \\mathcal{N}(0, \\sigma^2)\$

The joint distribution is therefore a multivariate normal with a diagonal covariance
matrix:
\$\\boldsymbol{\\phi} \\sim \\mathcal{N}(\\mathbf{0}, \\sigma^2 I)\$
where \$I\$ is the identity matrix.

# Assumptions
- The random effects are independent of each other.
- The random effects are drawn from the same distribution (identically distributed).

# Best Use Case
Modeling unstructured random effects for groups (e.g., random intercepts in a mixed
effects model), accounting for overdispersion in count models, or serving as the
unstructured component in more complex spatial models like the BYM2.

# Key References
- Wikipedia: Independent and identically distributed random variables

# Fields
- `sigma::UnivariateDistribution`: The prior for the standard deviation of the effect.
- `method::Symbol`: The parameterization method. Can be `:noncentered` (default,
  recommended) or `:centered` (didactic alternative).
"""
struct IID <: ComponentModel
    sigma::UnivariateDistribution
    method::Symbol
end

# Add to the central component constructor registry.
COMPONENT_TYPE_REGISTRY[:iid] = IID
COMPONENT_CONSTRUCTORS[:iid] = (p, params) -> IID(
    p.sigma, get(params, :method, :noncentered)
)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[:iid] = :any


"""
    get_datastructures!(m_type::Type{<:IID}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `IID` component. This component has no
specific data requirements beyond what is handled by the main configuration engine
(e.g., inferring `s_N` or `t_N` for the dimension of the effect).
"""
function get_datastructures!(m_type::Type{<:IID}, M::Dict, mod_data::Dict)::Bool
    return true
end

"""
    get_precomputes(m::IID, M::NamedTuple, mod_data::Dict)::NamedTuple

For an IID component, the precision matrix `Q_template` is the identity matrix.
This function determines the dimension of the effect based on its structure
(e.g., spatial, temporal) and returns the appropriate identity matrix.
"""
function get_precomputes(m::IID, M::NamedTuple, mod_data::Dict)::NamedTuple
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
        @warn "Could not determine dimension for IID component '$(mod_data[:key])'. " *
              "The component will have no effect."
    end

    template = build_structure_template(:iid, n)
    return (
        Q_template=template.matrix,
        U=template.U,
        L=template.L,
        scaling_factor=template.scaling_factor,
        n_latent=n
    )
end

"""
    get_priors(m::IID, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `sigma`. For the `:noncentered` method, it also defines a
prior for the standard normal `raw` innovations.
"""
function get_priors(
    m::IID, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    is_multivariate = (arch == "multivariate")
    is_shared = get(spec.params, :shared, false)
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ $(_distribution_to_string(m.sigma))")
    end

    if m.method == :noncentered
        push!(priors_acc, "$(v.raw) ~ MvNormal(zeros($(n_latent)), I)")
    end
    
    return join(priors_acc, "\n    ")
end


"""
    get_updates(m::IID, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to construct the IID effect. Supports two methods:
- `:noncentered` (default): Samples standard normal noise and transforms it. This
  is generally more efficient for MCMC.
- `:centered`: Samples the latent field directly from the `MvNormal` distribution.
  This can be less efficient due to posterior correlations but is a useful
  didactic alternative.
"""
function get_updates(
    m::IID, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    index_var = if spec.structure == :spatial
        "s_idx"
    elseif spec.structure == :temporal
        "t_idx"
    elseif spec.structure == :mixed
        "mixed_idx_$(spec.var)"
    else
        string(spec.structure) * "_idx"
    end

    noncentered_code = """
        # --- IID Component (Non-Centered): $(spec.key) ---
        $(v.latent) = $(v.raw) .* $(v.sigma)
        $(eta_target) .+= view($(v.latent), M.$(index_var))
    """

    centered_code = """
        # --- IID Component (Centered): $(spec.key) ---
        # This is a didactic alternative. It can be less efficient for MCMC sampling.
        $(v.latent) ~ MvNormal(zeros($(spec.hyper.n_latent)), $(v.sigma)^2 * I)
        $(eta_target) .+= view($(v.latent), M.$(index_var))
    """

    if m.method == :noncentered
        return noncentered_code
    elseif m.method == :centered
        return centered_code
    else
        error("Unsupported method '$(m.method)' for IID component.")
    end
end

"""
    get_effects(m::IID, chain, M::NamedTuple, ...)::NamedTuple

Reconstructs the posterior effect from the MCMC chain's posterior samples,
dispatching on the method used during sampling.
"""
function get_effects(
    m::IID, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    is_multivariate = outcomes_N > 1
    is_shared = get(spec.params, :shared, false)
    n_latent = spec.hyper.n_latent

    for k in 1:outcomes_N
        outcome_idx = is_multivariate ? k : nothing
        v = generate_full_variable_names(spec, M.model_arch, outcome_idx)
        
        local effect_k
        if m.method == :noncentered
            sigma_var_name = if is_multivariate && is_shared
                string(generate_full_variable_names(spec, M.model_arch, nothing).sigma)
            else
                string(v.sigma)
            end
            sigma_samples = get_params_vector(chain, sigma_var_name, 1)
            raw_samples = get_params_vector(chain, string(v.raw), n_latent)
            effect_k = raw_samples .* sigma_samples
        else # :centered
            effect_k = get_params_vector(chain, string(v.latent), n_latent)
        end
        
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
