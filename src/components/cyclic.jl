"""
    Cyclic <: ComponentModel

A component model for cyclic temporal effects, typically used for seasonal patterns.
It implements a first-order cyclic random walk (RW1 on a circle), where the last
point smoothly connects back to the first. This is a type of Gaussian Markov
Random Field (GMRF) with a circulant precision matrix.

# Version
v1.0.0 (2026-08-08)

# Mathematical Summary
The cyclic random walk models a latent field \$\\phi\$ where the value at time \$t\$ is
conditionally dependent on its neighbors, with the first and last points
considered neighbors. The conditional distribution is:
\$\\phi_t | \\phi_{-t} \\sim \\mathcal{N}\\left( \\frac{1}{2}(\\phi_{t-1} + \\phi_{t+1}), \\frac{\\sigma^2}{2} \\right)\$
(indices are taken modulo the period).

The joint precision matrix \$Q\$ is a circulant matrix corresponding to this structure.
Like the standard RW1, this is an intrinsic GMRF with a rank deficiency of 1, so a
sum-to-zero constraint is imposed on the latent field for identifiability.

# Assumptions
- The effect is periodic with a known `period`.
- The effect is smooth, with values at adjacent time points being similar.

# Best Use Case
Modeling smooth, repeating patterns where the end of a cycle influences the
beginning, such as day-of-the-year effects, day-of-week effects, or other cyclical
phenomena where a simple harmonic function is not flexible enough.

# Key References
- Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and
  Applications*. CRC Press. (For GMRFs and circulant precision matrices).
- Wikipedia: Circulant matrix

# Fields
- `period::Int`: The length of the cycle (e.g., 12 for months, 7 for days).
- `sigma::Distribution`: The prior for the standard deviation of the cyclic effect.
- `method::Symbol`: The computational method, either `:spectral` (default) or
  `:cholesky`.
"""
struct Cyclic <: ComponentModel
    period::Int
    sigma::Distribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:cyclic] = Cyclic

COMPONENT_CONSTRUCTORS[:cyclic] = (p, params) -> Cyclic(
    get(params, :period, 12), p.sigma, get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:cyclic] = :seasonal

"""
    get_datastructures!(m_type::Type{<:Cyclic}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Cyclic` component. It ensures a seasonal
index variable is provided and that its number of unique levels matches the
component's `period`.

# Assumptions
- The input data column specified in the `random()` call contains discrete,
  integer-like indices representing seasonal units (e.g., month 1-12).
"""
function get_datastructures!(
    m_type::Type{<:Cyclic}, M::Dict, mod_data::Dict
)::Bool
    variables = mod_data[:variables]
    if isempty(variables)
        error(
            "The Cyclic model requires a seasonal index variable, e.g., " *
            "`random(month, model=:cyclic)`."
        )
    end

    u_var_sym = Symbol(variables[1])
    if !hasproperty(M[:data], u_var_sym)
        error("Seasonal index variable ':$u_var_sym' for Cyclic model not found in data.")
    end
    
    M[:u_idx] = M[:data][!, u_var_sym]
    M[:u_N] = length(unique(M[:u_idx]))
    M[:u_idx_var] = u_var_sym
    
    period = get(mod_data[:params], :period, 12)
    if period != M[:u_N]
        error(
            "Cyclic `period` ($period) does not match the number of unique levels " *
            "in the index variable `$(u_var_sym)` ($(M[:u_N]))."
        )
    end

    return true
end

"""
    get_precomputes(m::Cyclic, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the circulant precision matrix (`Q_template`) for the cyclic random
walk, along with its spectral decomposition (`U`, `L`) and Cholesky factorization.
"""
function get_precomputes(m::Cyclic, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = m.period
    
    template = build_structure_template(:cyclic, n)
    F = cholesky(Symmetric(Matrix(template.matrix) + M.noise * I))
    
    return (
        Q_template=template.matrix,
        scaling_factor=template.scaling_factor,
        U=template.U,
        L=template.L,
        n_latent=n,
        cholesky_factor=F
    )
end

"""
    get_priors(m::Cyclic, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for the scale parameter `sigma` and the raw innovations `raw`.
"""
function get_priors(
    m::Cyclic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    
    return """
    $(p_names.sigma) ~ $(_distribution_to_string(m.sigma))
    $(p_names.raw) ~ MvNormal(zeros(T, $(n_latent)), I)
    """
end

"""
    get_updates(m::Cyclic, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to sample the latent cyclic field using either the `:spectral` or
`:cholesky` method, applying a sum-to-zero constraint for identifiability.
"""
function get_updates(
    m::Cyclic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent

    if m.method == :spectral
        return """
            # --- Cyclic Component: $(key) (Spectral Method) ---
            let
                local U = spec_registry[:$(key)].hyper.U
                local L = spec_registry[:$(key)].hyper.L
                local diag_D = $(p_names.sigma) ./ sqrt.(L .+ M.noise)
                diag_D[1] = 0.0 # Enforce sum-to-zero constraint
                local latent_field = U * (diag_D .* $(p_names.raw))
                $(eta_target) .+= view(latent_field, M.u_idx)
            end
        """
    else # :cholesky method
        return """
            # --- Cyclic Component: $(key) (Cholesky Method) ---
            let
                local F = spec_registry[:$(key)].hyper.cholesky_factor
                local latent_field_raw = F.L' \\ $(p_names.raw)
                
                Turing.@addlogprob! logpdf(
                    Normal(0.0,0.001 * $(n_latent)), sum(latent_field_raw)
                )
                
                local latent_field = latent_field_raw .* $(p_names.sigma)
                $(eta_target) .+= view(latent_field, M.u_idx)
            end
        """
    end
end

"""
    get_effects(m::Cyclic, chain, M::NamedTuple, ...)::NamedTuple

Reconstructs the `Cyclic` component's effect from posterior samples, applying a
sum-to-zero constraint for identifiability.
"""
function get_effects(
    m::Cyclic, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    n_latent = spec.hyper.n_latent
    noise = M.noise

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        raw_samples = get_params_vector(chain, string(p_names.raw), n_latent)

        effect_k = zeros(Float64, n_latent, n_samples)

        if m.method == :spectral
            U = spec.hyper.U
            L = spec.hyper.L
            for j in 1:n_samples
                diag_D = sigma_samples[j] ./ sqrt.(L .+ noise)
                diag_D[1] = 0.0 # Enforce sum-to-zero
                effect_k[:, j] = U * (diag_D .* raw_samples[j, :])
            end
        else # :cholesky method
            F = spec.hyper.cholesky_factor
            for j in 1:n_samples
                latent_field_raw = F.L' \ raw_samples[j, :]
                latent_field_centered = latent_field_raw .- mean(latent_field_raw)
                effect_k[:, j] = latent_field_centered .* sigma_samples[j]
            end
        end
        
        u_idx_full = isnothing(PS) ? M.u_idx : vcat(M.u_idx, PS.u_idx)
        indexed_effects = effect_k[u_idx_full, :]
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
