"""
    IID <: ComponentModel

A simple Independent and Identically Distributed (IID) random effect, representing
unstructured noise or heterogeneity. Each latent effect is drawn independently from
the same normal distribution.

# Version
v1.1.1 (2026-08-12)

# Mathematical Summary
The IID component models a latent field \$\\phi\$ where each element \$\\phi_i\$ is drawn
independently from a zero-mean normal distribution with a shared standard deviation
\$\\sigma\$:
\$\\phi_i \\sim \\mathcal{N}(0, \\sigma^2)\$

The joint distribution is therefore a multivariate normal with a diagonal covariance
matrix:
\$\\boldsymbol{\\phi} \\sim \\mathcal{N}(\\mathbf{0}, \\sigma^2 I)\$
where \$I\$ is the identity matrix.

# Computational Methods
- `:noncentered` (Default, AD-friendly): Samples standard normal innovations and
  scales them by `sigma`. Recommended for gradient-based samplers like NUTS.
- `:centered` (Didactic, Not AD-friendly): Samples the latent effects directly from
  the `MvNormal` distribution defined by `sigma`. This can be less efficient for
  MCMC due to posterior correlations.

# Inputs
- **Required**:
  - An index variable (e.g., `group_id`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the effect. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:noncentered` or `:centered`). Default: `:noncentered`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the IID effect.
- `innovations_<key>`: The raw standard normal innovations for the effect (for `:noncentered`).
- `latent_<key>`: The latent IID effect (for `:centered`).
"""
struct IID <: ComponentModel
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:iid] = IID
COMPONENT_CONSTRUCTORS[:iid] = (p, params) -> IID(
    p.sigma, get(params, :method, :noncentered)
)

MODEL_TO_STRUCTURE_MAP[:iid] = :any

function get_datastructures!(m_type::Type{<:IID}, M::Dict, mod_data::Dict)::Bool
    return true
end

function get_precomputes(m::IID, M::NamedTuple, mod_data::Dict)::NamedTuple
    structure = get(mod_data, :type, :spatial)
    
    n = if structure == :spatial
        get(M, :s_N, 0)
    elseif structure == :temporal
        get(M, :t_N, 0)
    elseif structure == :seasonal
        get(M, :u_N, 0)
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

function get_priors(
    m::IID, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    is_multivariate = (arch == "multivariate")
    is_shared = get(spec.params, :shared, false)
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")
    end

    if m.method == :noncentered
        # Removed explicit `T` from `zeros` for better AD compatibility.
        push!(priors_acc, "$(p_names.innovations) ~ NamedDist(MvNormal(zeros($(n_latent)), I), :$(p_names.innovations))")
    end
    
    return join(priors_acc, "\n    ")
end

function get_updates(
    m::IID, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    index_var = if spec.structure == :spatial
        "s_idx"
    elseif spec.structure == :temporal
        "t_idx"
    elseif spec.structure == :seasonal
        "u_idx"
    elseif spec.structure == :mixed
        "mixed_idx_$(spec.var)"
    else
        string(spec.structure) * "_idx"
    end

    noncentered_code = """
        # --- IID Component (Non-Centered): $(spec.key) ---
        let
            $(p_names.latent) = $(p_names.innovations) .* $(p_names.sigma)
            $(eta_target) .+= view($(p_names.latent), M.$(index_var))
        end
    """

    centered_code = """
        # --- IID Component (Centered): $(spec.key) ---
        let
            # Removed explicit `T` from `zeros` for better AD compatibility.
            $(p_names.latent) ~ MvNormal(zeros($(spec.hyper.n_latent)), $(p_names.sigma)^2 * I)
            $(eta_target) .+= view($(p_names.latent), M.$(index_var))
        end
    """

    if m.method == :noncentered
        return noncentered_code
    elseif m.method == :centered
        return centered_code
    else
        error("Unsupported method '$(m.method)' for IID component.")
    end
end

function get_effects(
    m::IID, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))
    n_latent = spec.hyper.n_latent

    index_var_sym = if spec.structure == :spatial
        :s_idx
    elseif spec.structure == :temporal
        :t_idx
    elseif spec.structure == :seasonal
        :u_idx
    elseif spec.structure == :mixed
        Symbol("mixed_idx_$(spec.var)")
    else
        Symbol(string(spec.structure) * "_idx")
    end

    idx_full = if haskey(M, index_var_sym)
        isnothing(PS) || !haskey(PS, index_var_sym) ? M[index_var_sym] : vcat(M[index_var_sym], PS[index_var_sym])
    else
        ones(Int, N_total)
    end

    for k in 1:outcomes_N
        sigma_name = _find_parameter(p_names_vec, string(spec.key), "sigma", k, is_multivariate_model)
        
        if isempty(sigma_name)
            @warn "Sigma parameter for IID component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        
        local latent_field_samples
        if m.method == :noncentered
            innovations_name = _find_parameter(p_names_vec, string(spec.key), "innovations", k, is_multivariate_model)
            if isempty(innovations_name)
                @warn "Innovations for IID component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            innovations_samples = get_params_vector(chain, innovations_name, n_latent)
            latent_field_samples = innovations_samples' .* sigma_samples
        else # :centered
            latent_name = _find_parameter(p_names_vec, string(spec.key), "latent", k, is_multivariate_model)
            if isempty(latent_name)
                @warn "Latent field for IID component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            latent_field_samples = get_params_vector(chain, latent_name, n_latent)'
        end
        
        effect_k = latent_field_samples[idx_full, :]
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
