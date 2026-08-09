
"""
    RW1 <: ComponentModel

A component for a first-order random walk (RW1) model. This model assumes that the
current value of a latent temporal field is the previous value plus a random
innovation. It is an intrinsic Gaussian Markov Random Field (GMRF) with a rank
deficiency of 1, implying a sum-to-zero constraint for identifiability.

# Version
v1.9.2 (2026-08-08)

# Mathematical Summary
The RW1 model defines a latent temporal field \$\\phi\$ where the value at time \$t\$ is
conditionally dependent on its immediate predecessor:
\$\\phi_t | \\phi_{t-1} \\sim \\mathcal{N}(\\phi_{t-1}, \\sigma^2)\$
This can be written as \$\\phi_t - \\phi_{t-1} = \\epsilon_t\$, where
\$\\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2)\$.

The joint precision matrix \$\\mathbf{Q}\$ for this process is singular (rank-deficient),
making it an "intrinsic" GMRF. To ensure the model is identifiable from a global
intercept, a sum-to-zero constraint (\$\\sum_i \\phi_i = 0\$) is imposed on the
latent field.

# Assumptions
- The temporal process is non-stationary and exhibits local smoothness.
- The time points are regularly spaced.

# Best Use Case
Modeling smooth but non-stationary temporal trends, or processes that exhibit
step-like changes. It is a fundamental building block for time series analysis.

# Key References
- Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and
  Applications*. CRC Press.
- Wikipedia: Random walk

# Fields
- `sigma::UnivariateDistribution`: The prior for the standard deviation of the
  innovations.
- `method::Symbol`: The computational method, either `:statespace` (default, most
  efficient) or `:spectral`.
"""
struct RW1 <: ComponentModel
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:rw1] = RW1

COMPONENT_CONSTRUCTORS[:rw1] = (p, params) -> RW1(
    p.sigma, get(params, :method, :statespace)
)

MODEL_TO_STRUCTURE_MAP[:rw1] = :temporal

"""
    get_datastructures!(m_type::Type{<:RW1}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `RW1` component. It establishes the temporal
context by identifying the time variable, creating discrete time indices (`t_idx`),
and determining the total number of time steps (`t_N`).
"""
function get_datastructures!(m_type::Type{<:RW1}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]
    if isempty(variables)
        error("The RW1 model requires a time index variable, e.g., `random(year, model=:rw1)`.")
    end

    time_var_sym = Symbol(variables[1])
    if !hasproperty(M[:data], time_var_sym)
        error("Time index variable ':$time_var_sym' for RW1 model not found in data.")
    end

    time_opts = Dict(:time_method => get(mod_data[:params], :time_method, "regular"))
    tu_meta = assign_time_units(M[:data][!, time_var_sym]; time_opts...)
    
    M[:t_idx] = tu_meta.idx
    M[:t_N] = tu_meta.N_cat
    M[:t_idx_var] = time_var_sym
    
    return true
end

"""
    get_precomputes(m::RW1, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the structure matrix for the `RW1` component. The precision matrix
template defines the first-order differences. This function calls the central
`build_structure_template` utility to generate this matrix and its spectral
decomposition (`U`, `L`), which are essential for the `:spectral` sampling method.
"""
function get_precomputes(m::RW1, M::NamedTuple, mod_data::Dict)::NamedTuple
    t_N = get(M, :t_N, 0)
    if t_N == 0
        @warn "Could not determine number of time steps for RW1 component " *
              "'$(mod_data[:key])'. The component will have no effect."
    end
    template = build_structure_template(:rw1, t_N)
    return (
        Q_template=template.matrix,
        U=template.U,
        L=template.L,
        scaling_factor=template.scaling_factor,
        n_latent=t_N
    )
end

"""
    get_priors(m::RW1, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the `RW1` component's priors. This function creates
the code strings for sampling the `sigma` hyperparameter and the standard normal
innovations (`raw`) for the latent field.
"""
function get_priors(
    m::RW1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
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
    push!(priors_acc, "$(v.raw) ~ MvNormal(zeros($(n_latent)), I)")
    return join(priors_acc, "\n    ")
end

"""
    get_updates(m::RW1, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the `RW1` component's update logic, dispatching on
the chosen `method` (`:statespace` or `:spectral`).
"""
function get_updates(
    m::RW1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent

    if m.method == :statespace
        return """
            # --- RW1 Component: $(key) (State-Space Method) ---
            # This block constructs the RW1 effect by taking the cumulative sum
            # of standard normal innovations. This is the most efficient method.
            let
                innovations = $(p_names.raw)
                latent_field_raw = cumsum(innovations)
                
                # Apply a soft sum-to-zero constraint for identifiability.
                Turing.@addlogprob! logpdf(
                    Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_raw)
                )
                
                $(p_names.latent) = latent_field_raw .* $(p_names.sigma)
                $(eta_target) .+= view($(p_names.latent), M.t_idx)
            end
        """
    elseif m.method == :spectral
        return """
            # --- RW1 Component: $(key) (Spectral Method) ---
            # This block constructs the RW1 effect using a non-centered
            # parameterization based on the spectral decomposition of the
            # precision matrix. This method is AD-safe and efficient.
            let
                local U = spec.hyper.U
                local L = spec.hyper.L
                local diag_D = $(p_names.sigma) ./ sqrt.(L .+ M.noise)
                
                # Enforce sum-to-zero constraint by zeroing out the component
                # corresponding to the null space (the first eigenvalue).
                diag_D[1] = 0.0
                
                $(p_names.latent) = U * (diag_D .* $(p_names.raw))
                $(eta_target) .+= view($(p_names.latent), M.t_idx)
            end
        """
    else
        error("Unsupported method '$(m.method)' for RW1. Use :statespace or :spectral.")
    end
end

"""
    get_effects(m::RW1, chain, M::NamedTuple, ...)

Reconstructs the `RW1` component's effect from posterior samples, applying a
sum-to-zero constraint for identifiability.
"""
function get_effects(
    m::RW1, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    n_latent = spec.hyper.n_latent

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        raw_samples = get_params_vector(chain, string(p_names.raw), n_latent)

        effect_k = zeros(Float64, n_latent, n_samples)

        if m.method == :statespace
            for j in 1:n_samples
                latent_field_raw = cumsum(raw_samples[j, :])
                latent_field_centered = latent_field_raw .- mean(latent_field_raw)
                effect_k[:, j] = latent_field_centered .* sigma_samples[j]
            end
        else # :spectral
            U = spec.hyper.U
            L = spec.hyper.L
            for j in 1:n_samples
                diag_D = sigma_samples[j] ./ sqrt.(L .+ M.noise)
                diag_D[1] = 0.0 # Enforce sum-to-zero
                effect_k[:, j] = U * (diag_D .* raw_samples[j, :])
            end
        end
        
        t_idx_full = isnothing(PS) ? M.t_idx : vcat(M.t_idx, PS.t_idx)
        indexed_effects = effect_k[t_idx_full, :]
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
