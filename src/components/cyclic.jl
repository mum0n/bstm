"""
    Cyclic <: ComponentModel

A component model for cyclic temporal effects, typically used for seasonal patterns.
It implements a first-order cyclic random walk (RW1 on a circle), where the last
point smoothly connects back to the first. This is a type of Gaussian Markov
Random Field (GMRF) with a circulant precision matrix.

# Version
v1.1.0 (2026-08-11)

# Mathematical Summary
The cyclic random walk models a latent field \$\\phi\$ where the value at time \$t\$ is
conditionally dependent on its neighbors, with the first and last points
considered neighbors. The conditional distribution is:
\$\\phi_t | \\phi_{-t} \\sim \\mathcal{N}\\left( \\frac{1}{2}(\\phi_{t-1} + \\phi_{t+1}), \\frac{\\sigma^2}{2} \\right)\$
(indices are taken modulo the period).

The joint precision matrix \$Q\$ is a circulant matrix corresponding to this structure.
Like the standard RW1, this is an intrinsic GMRF with a rank deficiency of 1, so a
sum-to-zero constraint is imposed on the latent field for identifiability.

# Computational Methods
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the circulant precision matrix. Recommended for NUTS.
- `:cholesky` (AD-friendly): Uses a pre-computed dense Cholesky factorization.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization,
  which is not compatible with most AD backends.

# Inputs
- **Required**:
  - A seasonal index variable (e.g., `month`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `period`: `Int`, the length of the cycle. Must match the number of unique
    levels in the index variable. Default: `12`.
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the
    cyclic effect. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`).
    Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the cyclic effect.
- `innovations_<key>`: The raw standard normal innovations for the effect.
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

Generates priors for the scale parameter `sigma` and the raw innovations `innovations`.

# Inputs
- `m::Cyclic`: The `Cyclic` component instance.
- `spec::NamedTuple`: The component's specification, including its `key` and `hyper` parameters.
- `arch::String`: The model architecture (`"univariate"` or `"multivariate"`).
- `outcome_idx::Union{Int, Nothing}`: The index of the outcome for multivariate models.
- `M::NamedTuple`: The main model configuration.

# Outputs
- `String`: Turing code for the priors of `sigma` and `innovations`.
"""
function get_priors(
    m::Cyclic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    
    return """
    $(p_names.sigma) ~ $(_distribution_to_string(m.sigma))
    $(p_names.innovations) ~ MvNormal(zeros(T, $(n_latent)), I)
    """
end

"""
    get_updates(m::Cyclic, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to sample the latent cyclic field. Supports three methods:
- `:spectral` (default): An efficient, AD-safe method using spectral decomposition.
- `:cholesky`: An AD-safe didactic alternative using dense Cholesky factorization.
- `:cholesky_sparse`: A non-AD-safe didactic method using sparse Cholesky
  factorization, suitable for gradient-free samplers.

# Inputs
- `m::Cyclic`: The `Cyclic` component instance.
- `spec::NamedTuple`: The component's specification, including its `key` and `hyper` parameters.
- `arch::String`: The model architecture (`"univariate"` or `"multivariate"`).
- `outcome_idx::Union{Int, Nothing}`: The index of the outcome for multivariate models.
- `M::NamedTuple`: The main model configuration.

# Outputs
- `String`: Turing code for constructing the latent cyclic field and adding it to the linear predictor.
"""
function get_updates(
    m::Cyclic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent

    spectral_code = """
        # --- Cyclic Component: $(key) (Spectral Method) ---
        let
            hyper = spec_registry[:$(key)].hyper
            U, L = hyper.U, hyper.L
            diag_D = $(p_names.sigma) ./ sqrt.(L .+ M.noise)
            diag_D[1] = 0.0 # Enforce sum-to-zero constraint
            latent_field = U * (diag_D .* $(p_names.innovations))
            $(eta_target) .+= view(latent_field, M.u_idx)
        end
    """

    cholesky_code = """
        # --- Cyclic Component: $(key) (Cholesky Method, AD-Safe) ---
        let
            hyper = spec_registry[:$(key)].hyper
            F = hyper.cholesky_factor
            latent_field_raw = F.L' \\ $(p_names.innovations)
            
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_raw)
            )
            
            latent_field = latent_field_raw .* $(p_names.sigma)
            $(eta_target) .+= view(latent_field, M.u_idx)
        end
    """

    cholesky_sparse_code = """
        # --- Cyclic Component: $(key) (Sparse Cholesky, Not AD-Safe): ---
        # This method is for didactic purposes and is NOT compatible with
        # automatic differentiation (e.g., NUTS sampler).
        let
            hyper = spec_registry[:$(key)].hyper
            Q = hyper.Q_template
            F = cholesky(Symmetric(Q + M.noise * I))
            latent_field_raw = F.L' \\ $(p_names.innovations)
            
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_raw)
            )
            
            latent_field = latent_field_raw .* $(p_names.sigma)
            $(eta_target) .+= view(latent_field, M.u_idx)
        end
    """

    if m.method == :spectral
        return spectral_code
    elseif m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    else
        error("Unsupported method '$(m.method)' for Cyclic component. Supported methods are :spectral, :cholesky, and :cholesky_sparse.")
    end
end

"""
    get_effects(m::Cyclic, chain, M::NamedTuple, n_samples, outcomes_N, spec, PS, N_total)::NamedTuple

Reconstructs the `Cyclic` component's effect from posterior samples, applying a
sum-to-zero constraint for identifiability. This function dispatches on the method
used during sampling.

# Inputs
- `m::Cyclic`: The `Cyclic` component instance.
- `chain`: The MCMC chain object.
- `M::NamedTuple`: The main model configuration.
- `n_samples::Int`: The number of posterior samples.
- `outcomes_N::Int`: The number of outcome variables.
- `spec::NamedTuple`: The component's specification, including its `key` and `hyper` parameters.
- `PS::Union{NamedTuple, Nothing}`: The prediction set configuration object, if applicable.
- `N_total::Int`: The total number of observations (training + prediction).

# Outputs
- `NamedTuple`: A NamedTuple with `structured` and `noisy` effects, each a vector of matrices
  `[N_total x n_samples]`.
"""
function get_effects(
    m::Cyclic, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    n_latent = spec.hyper.n_latent
    noise = M.noise
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))

    for k in 1:outcomes_N
        sigma_name = _find_parameter(p_names_vec, string(spec.key), "sigma", k, is_multivariate_model)
        innovations_name = _find_parameter(p_names_vec, string(spec.key), "innovations", k, is_multivariate_model)

        if isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for Cyclic component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innovations_name, n_latent)

        effect_k = zeros(Float64, n_latent, n_samples)

        if m.method == :spectral
            U = spec.hyper.U
            L = spec.hyper.L
            for j in 1:n_samples
                diag_D = sigma_samples[j] ./ sqrt.(L .+ noise)
                diag_D[1] = 0.0 # Enforce sum-to-zero
                effect_k[:, j] = U * (diag_D .* innovations_samples[j, :])
            end
        else # :cholesky or :cholesky_sparse
            # For reconstruction, we can use the pre-computed dense factor for both
            # Cholesky methods as it does not involve AD.
            F = spec.hyper.cholesky_factor
            for j in 1:n_samples
                latent_field_raw = F.L' \ innovations_samples[j, :]
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
