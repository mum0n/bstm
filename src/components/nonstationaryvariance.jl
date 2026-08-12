"""
    NonStationaryVariance <: ComponentModel

A component model for non-stationary variance, where the standard deviation of a base
spatial effect varies across space according to a modifier spatial effect.
It acts as an orchestrator, combining a `base_model` (typically a spatial GMRF)
with a `modifier_model` (typically a spatial smoother) to create a spatially
varying standard deviation.

# Version
v1.1.0 (2026-08-11)

# Mathematical Summary
The component models a non-stationary spatial field \$\\phi(s)\$ where the local
standard deviation is a function of some covariates. The effect is a product of a
stationary base process \$\\phi_{base}\$ and a spatially varying scale \$\\sigma(s)\$:
\$\\phi(s) = \\phi_{base}(s) \\cdot \\sigma(s)\$

1.  **Base Process (\$\\phi_{base}\$)**: This is a standard, zero-mean Gaussian Markov
    Random Field (GMRF) with unit variance, such as an `ICAR` or `Besag` model.
    \$\\phi_{base} \\sim \\mathcal{N}(0, Q_{base}^{-1})\$

2.  **Scale Process (\$\\sigma(s)\$)**: The logarithm of the scale is modeled as a
    smooth function of one or more covariates \$x\$, defined by the `modifier_model`:
    \$\\log(\\sigma(s)) = f(x(s))\$
    where \$f(x)\$ is typically a P-spline or Gaussian Process smoother. Exponentiating
    ensures the standard deviation is always positive.

# Computational Methods (for Base Model)
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the base model's precision matrix. Recommended for NUTS.
- `:cholesky` (AD-friendly): Uses a dense Cholesky factorization of the base model's
  precision matrix.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization,
  which is not compatible with most AD backends.

# Inputs
- **Required**:
  - A composition of two `random()` modules using the `∘` operator, e.g.,
    `random(s_idx, model=icar) ∘ random(cov, model=pspline)`.
  - The base model must be a spatial GMRF (e.g., `icar`, `bym2`).
  - The modifier model must be a smoother (e.g., `pspline`, `gp`).
- **Optional**:
  - `method`: `Symbol`, computational method for the base model. Default: `:spectral`.

# Outputs (Parameter Names)
- Parameters are inherited from the child components, prefixed with the main key
  and either `_base` or `_modifier`. For example:
  - `sigma_<key>_modifier`: The standard deviation of the modifier smoother.
  - `rho_<key>_base`: The mixing parameter of the base spatial model (if applicable).
  - `innovations_<key>_modifier`: Innovations for the modifier smoother.
  - `innovations_<key>_base`: Innovations for the base spatial model.
"""
struct NonStationaryVariance <: ComponentModel
    base_model::ComponentModel
    modifier_model::ComponentModel
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:nonstationaryvariance] = NonStationaryVariance
COMPONENT_CONSTRUCTORS[:nonstationaryvariance] = (p, params) -> begin
    base_model_obj = get(
        params, :base_model_obj,
        error("NonStationaryVariance requires a `base_model_obj`.")
    )
    modifier_model_obj = get(
        params, :modifier_model_obj,
        error("NonStationaryVariance requires a `modifier_model_obj`.")
    )
    method = get(params, :method, :spectral)
    NonStationaryVariance(base_model_obj, modifier_model_obj, method)
end

MODEL_TO_STRUCTURE_MAP[:nonstationaryvariance] = :spatial

function get_datastructures!(
    m_type::Type{<:NonStationaryVariance}, M::Dict, mod_data::Dict
)::Bool
    params = mod_data[:params]

    base_model_spec_node = get(params, :base_model_spec, nothing)
    if isnothing(base_model_spec_node)
        error("NonStationaryVariance requires a `base_model_spec` parameter.")
    end
    base_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_base"),
        :type => base_model_spec_node.module_type,
        :variables => get(base_model_spec_node.args, :positional_args, []),
        :params => base_model_spec_node.args,
        :component_obj => mod_data[:component_obj].base_model
    )
    base_model_type = typeof(mod_data[:component_obj].base_model)
    get_datastructures!(base_model_type, M, base_mod_data)

    modifier_model_spec_node = get(params, :modifier_model_spec, nothing)
    if isnothing(modifier_model_spec_node)
        error("NonStationaryVariance requires a `modifier_model_spec` parameter.")
    end
    modifier_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_modifier"),
        :type => modifier_model_spec_node.module_type,
        :variables => get(modifier_model_spec_node.args, :positional_args, []),
        :params => modifier_model_spec_node.args,
        :component_obj => mod_data[:component_obj].modifier_model
    )
    modifier_model_type = typeof(mod_data[:component_obj].modifier_model)
    get_datastructures!(modifier_model_type, M, modifier_mod_data)

    return true
end

function get_precomputes(
    m::NonStationaryVariance, M::NamedTuple, mod_data::Dict
)::NamedTuple
    params = mod_data[:params]

    base_model_spec_node = get(params, :base_model_spec, nothing)
    base_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_base"),
        :type => base_model_spec_node.module_type,
        :variables => get(base_model_spec_node.args, :positional_args, []),
        :params => base_model_spec_node.args
    )
    base_precomputes = get_precomputes(m.base_model, M, base_mod_data)

    modifier_model_spec_node = get(params, :modifier_model_spec, nothing)
    modifier_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_modifier"),
        :type => modifier_model_spec_node.module_type,
        :variables => get(modifier_model_spec_node.args, :positional_args, []),
        :params => modifier_model_spec_node.args
    )
    modifier_precomputes = get_precomputes(m.modifier_model, M, modifier_mod_data)

    modifier_basis_key = Symbol(join(modifier_mod_data[:variables], "_"))

    return (
        base_precomputes=base_precomputes,
        modifier_precomputes=modifier_precomputes,
        modifier_basis_key=modifier_basis_key
    )
end

function get_priors(
    m::NonStationaryVariance, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    base_spec_key = Symbol("$(spec.key)_base")
    modifier_spec_key = Symbol("$(spec.key)_modifier")

    base_spec_for_priors = (
        key = base_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.base_model)],
        var = spec.var,
        component_obj = m.base_model,
        params = spec.params,
        hyper = spec.hyper.base_precomputes
    )
    modifier_spec_for_priors = (
        key = modifier_spec_key,
        structure = MODEL_TO_STRUCTURE_MAP[typeof(m.modifier_model)],
        var = spec.var,
        component_obj = m.modifier_model,
        params = spec.params,
        hyper = spec.hyper.modifier_precomputes
    )

    base_priors = get_priors(m.base_model, base_spec_for_priors, arch, outcome_idx, M)
    base_priors_cleaned = replace(base_priors, r".*sigma.*" => "")
    
    modifier_priors = get_priors(
        m.modifier_model, modifier_spec_for_priors, arch, outcome_idx, M
    )

    base_p_names = generate_full_variable_names(base_spec_for_priors, arch, outcome_idx)
    n_latent_base = spec.hyper.base_precomputes.n_latent
    base_innov_prior = "$(base_p_names.innovations) ~ MvNormal(zeros(T, $(n_latent_base)), I)"

    return """
        # --- Priors for NonStationaryVariance component: $(spec.key) ---
        $(base_priors_cleaned)
        $(modifier_priors)
        $(base_innov_prior)
    """
end
"""
    get_updates(m::NonStationaryVariance, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates Turing code for the `NonStationaryVariance` component.
"""
function get_updates(
    m::NonStationaryVariance, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    # Define keys for child components
    base_spec_key = Symbol("$(spec.key)_base")
    modifier_spec_key = Symbol("$(spec.key)_modifier")

    # Generate variable names for both components
    base_p_names = generate_full_variable_names((key=base_spec_key,), arch, outcome_idx)
    modifier_p_names = generate_full_variable_names((key=modifier_spec_key,), arch, outcome_idx)

    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"

    # --- 1. Generate code for the MODIFIER component (log-sigma field) ---
    modifier_model = m.modifier_model
    modifier_hyper = spec.hyper.modifier_precomputes
    modifier_basis_key = spec.hyper.modifier_basis_key
    
    local modifier_effect_code
    modifier_method = get(modifier_model, :method, :spectral)

    if modifier_method == :spectral
        modifier_effect_code = """
            # Modifier model: $(modifier_spec_key) (Spectral)
            local modifier_hyper = spec_registry[:$(modifier_spec_key)].hyper
            local B_modifier = M.basis_matrices[:$(modifier_basis_key)]
            
            local diag_D_mod = $(modifier_p_names.sigma) ./ sqrt.(modifier_hyper.L .+ M.noise)
            if $(typeof(modifier_model)) in [RW1, ICAR, Besag]; diag_D_mod[1] = 0.0; end
            if $(typeof(modifier_model)) in [PSpline, RW2]; diag_D_mod[1] = 0.0; diag_D_mod[2] = 0.0; end
            
            local modifier_coeffs = modifier_hyper.U * (diag_D_mod .* $(modifier_p_names.innovations))
            local log_sigma_field = B_modifier * modifier_coeffs
        """
    else # :cholesky or :cholesky_sparse
        modifier_effect_code = """
            # Modifier model: $(modifier_spec_key) (Cholesky, AD-Safe)
            local modifier_hyper = spec_registry[:$(modifier_spec_key)].hyper
            local B_modifier = M.basis_matrices[:$(modifier_basis_key)]
            local Q_mod = modifier_hyper.Q_template
            
            local F_mod = cholesky(Symmetric(Matrix(Q_mod) + M.noise * I))
            local coeffs_raw_mod = F_mod.L' \\ $(modifier_p_names.innovations)
            
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(modifier_hyper.n_latent)), sum(coeffs_raw_mod))
            
            local modifier_coeffs = $(modifier_p_names.sigma) .* coeffs_raw_mod
            local log_sigma_field = B_modifier * modifier_coeffs
        """
    end

    # --- 2. Generate code for the BASE component ---
    base_model_type_sym = Symbol(lowercase(string(typeof(m.base_model))))
    n_latent_base = spec.hyper.base_precomputes.n_latent

    local base_latent_reconstruction_code
    if m.method == :spectral
        base_latent_reconstruction_code = """
            local base_hyper = spec_registry[:$(base_spec_key)].hyper
            local diag_D_base = 1.0 ./ sqrt.(base_hyper.L .+ M.noise)
            if $(base_model_type_sym) in [:icar, :besag]; diag_D_base[1] = 0.0; end
            local base_latent_raw = base_hyper.U * (diag_D_base .* $(base_p_names.innovations))
        """
    else # :cholesky or :cholesky_sparse
        base_latent_reconstruction_code = """
            local Q_base = spec_registry[:$(base_spec_key)].hyper.Q_template
            local F_base = cholesky(Symmetric(Matrix(Q_base) + M.noise * I))
            local base_latent_raw = F_base.L' \\ $(base_p_names.innovations)
        """
    end

    sum_to_zero_constraint = if base_model_type_sym in [:icar, :besag]
        "Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_latent_base)), sum(base_latent_raw))"
    else
        ""
    end

    # --- 3. Assemble the final update block ---
    return """
        # --- NonStationaryVariance Component: $(spec.key) ---
        let
            # 1. Realize the log-standard deviation field from the modifier model.
            $(modifier_effect_code)
            local spatially_varying_sigma = exp.(log_sigma_field)
            
            # 2. Realize the raw latent field from the base model.
            $(base_latent_reconstruction_code)
            $(sum_to_zero_constraint)
            
            # 3. Combine raw latent field with spatially varying sigma.
            local final_effect_latent = base_latent_raw .* spatially_varying_sigma
            
            # 4. Add the final effect to the linear predictor.
            $(eta_target) .+= view(final_effect_latent, M.s_idx)
        end
    """
end


function get_effects(
    m::NonStationaryVariance, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))
    
    base_spec_key = Symbol("$(spec.key)_base")
    modifier_spec_key = Symbol("$(spec.key)_modifier")

    s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)

    for k in 1:outcomes_N
        outcome_idx = outcomes_N > 1 ? k : nothing
        
        modifier_spec = (
            key = modifier_spec_key,
            structure = MODEL_TO_STRUCTURE_MAP[typeof(m.modifier_model)],
            var = spec.var,
            component_obj = m.modifier_model,
            params = spec.params,
            hyper = spec.hyper.modifier_precomputes
        )
        modifier_results = get_effects(
            m.modifier_model, chain, M, n_samples, outcomes_N, modifier_spec, PS, N_total
        )
        log_sigma_field_samples = modifier_results.structured[k]
        spatially_varying_sigma_samples = exp.(log_sigma_field_samples)

        base_spec = (
            key = base_spec_key,
            structure = MODEL_TO_STRUCTURE_MAP[typeof(m.base_model)],
            var = spec.var,
            component_obj = m.base_model,
            params = spec.params,
            hyper = spec.hyper.base_precomputes
        )
        
        base_innovations_name = _find_parameter(p_names_vec, string(base_spec.key), "innovations", k, is_multivariate_model)
        if isempty(base_innovations_name)
            @warn "Base innovations for NonStationaryVariance component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end
        base_innovations_samples = get_params_vector(chain, base_innovations_name, base_spec.hyper.n_latent)
        
        n_latent_base = base_spec.hyper.n_latent
        base_latent_raw_samples = zeros(n_samples, n_latent_base)

        if m.method == :spectral
            U, L = base_spec.hyper.U, base_spec.hyper.L
            diag_D = 1.0 ./ sqrt.(L .+ M.noise)
            if typeof(m.base_model) in [ICAR, Besag]; diag_D[1] = 0.0; end
            for i in 1:n_samples
                base_latent_raw_samples[i, :] = U * (diag_D .* base_innovations_samples[i, :])
            end
        else # :cholesky or :cholesky_sparse
            Q_base_template = base_spec.hyper.Q_template
            F_base = cholesky(Symmetric(Matrix(Q_base_template) + M.noise * I))
            for i in 1:n_samples
                base_latent_raw_samples[i, :] = F_base.L' \ base_innovations_samples[i, :]
            end
        end

        final_effect_k = zeros(N_total, n_samples)
        for i in 1:n_samples
            base_field_at_obs = view(base_latent_raw_samples[i, :], s_idx_full)
            sigma_at_obs = spatially_varying_sigma_samples[:, i]
            final_effect_k[:, i] = base_field_at_obs .* sigma_at_obs
        end
        
        push!(structured_effects, final_effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
