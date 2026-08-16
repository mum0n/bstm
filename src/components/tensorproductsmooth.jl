"""
    TensorProductSmooth <: ComponentModel

A component for creating inseparable spatiotemporal effects by taking the tensor
product of marginal (e.g., spatial and temporal) components. This is typically
specified in the formula via the Kronecker product operator `⊗`.

# Version
v1.1.1 (2026-08-14)

# Mathematical Summary
This component models an inseparable spatiotemporal random effect \$\\boldsymbol{\\delta}\$
as a zero-mean Gaussian Process with a separable covariance structure. Given a
spatial precision matrix \$\\mathbf{Q}_s\$ and a temporal precision matrix \$\\mathbf{Q}_t\$,
the joint spatiotemporal precision matrix is the Kronecker product of the marginals:

\$\\mathbf{Q}_{st} = \\mathbf{Q}_t \\otimes \\mathbf{Q}_s\$

This structure implies that the covariance function is separable, i.e.,
\$K_{st}((s_1, t_1), (s_2, t_2)) = K_s(s_1, s_2) \\cdot K_t(t_1, t_2)\$. This is the
basis for the Knorr-Held Type IV interaction model.

# Computational Methods
- `:cholesky` (Default, AD-friendly): An AD-safe method using dense Cholesky factorization
  of the marginal precision matrices. Recommended for gradient-based samplers.
- `:cholesky_sparse` (Didactic, Not AD-friendly): A more memory-efficient method using
  sparse Cholesky factorization, suitable for gradient-free samplers.

# Inputs
- **Required**:
  - A composition of two `random()` modules using the `⊗` operator, e.g.,
    `random(s_idx, model=icar) ⊗ random(year, model=ar1)`.
- **Optional**:
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the
    interaction effect. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:cholesky` or `:cholesky_sparse`).
    Default: `:cholesky`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the interaction effect.
- `innovations_<key>`: The raw standard normal innovations for the interaction field.

# Key References
- Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time variation
  in disease risk. *Statistical Methods in Medical Research*, 9(3), 205-220.
"""
struct TensorProductSmooth <: ComponentModel
    components::Vector{ComponentModel}
    sigma::Distribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:tensorproductsmooth] = TensorProductSmooth
COMPONENT_CONSTRUCTORS[:tensorproductsmooth] = (p, params) -> begin
    components = get(params, :components, error("TensorProductSmooth requires child components."))
    TensorProductSmooth(components, p.sigma, get(params, :method, :cholesky))
end
COMPONENT_CONSTRUCTORS[:interaction] = COMPONENT_CONSTRUCTORS[:tensorproductsmooth]

MODEL_TO_STRUCTURE_MAP[:tensorproductsmooth] = :spacetime

function get_precomputes(m::TensorProductSmooth, M::NamedTuple, mod_data::Dict)::NamedTuple
    if isempty(m.components) || length(m.components) != 2
        error("TensorProductSmooth requires exactly two child components.")
    end

    if !hasproperty(M, :st_idx)
        error("TensorProductSmooth requires spatiotemporal index `st_idx` to be pre-computed by the model processor.")
    end

    child_nodes = get(mod_data[:params], :components, [])
    child_specs_list = []
    n_latent_total = 1

    for (i, child_node) in enumerate(child_nodes)
        child_component_obj = m.components[i]
        
        child_mod_data = Dict(
            :key => Symbol("$(mod_data[:key])_child_$(i)"),
            :variables => get(child_node.args, :positional_args, []),
            :params => child_node.args
        )
        
        precomputes = get_precomputes(child_component_obj, M, child_mod_data)
        n_latent_total *= precomputes.n_latent

        child_spec = (
            key = child_mod_data[:key],
            structure = MODEL_TO_STRUCTURE_MAP[typeof(child_component_obj)],
            var = join(child_mod_data[:variables], "_"),
            component_obj = child_component_obj,
            params = child_mod_data[:params],
            hyper = precomputes
        )
        push!(child_specs_list, child_spec)
    end

    return (
        child_specs = child_specs_list,
        n_latent = n_latent_total
    )
end

function get_priors(
    m::TensorProductSmooth, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    key = spec.key
    
    return """
    # Priors for Spatiotemporal Interaction: $(spec.key)
    $(p_names.sigma) ~ $(_distribution_to_string(m.sigma))
    $(p_names.innovations) ~ MvNormal(
        zeros(T, spec_registry[:$(key)].hyper.n_latent), I
    )
    """
end

function get_updates(
    m::TensorProductSmooth, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    
    child_specs = spec.hyper.child_specs
    s_spec, t_spec = child_specs[1], child_specs[2]
    
    s_model_type = Symbol(lowercase(string(typeof(s_spec.component_obj))))
    t_model_type = Symbol(lowercase(string(typeof(t_spec.component_obj))))
    
    s_rho_val = hasproperty(s_spec.component_obj, :rho) ? string(generate_full_variable_names(s_spec, arch, outcome_idx).rho) : "nothing"
    t_rho_val = hasproperty(t_spec.component_obj, :rho) ? string(generate_full_variable_names(t_spec, arch, outcome_idx).rho) : "nothing"

    s_N = s_spec.hyper.n_latent
    t_N = t_spec.hyper.n_latent

    cholesky_base_code = """
        local parent_hyper = spec_registry[:$(key)].hyper
        local s_spec_hyper = parent_hyper.child_specs[1].hyper
        local t_spec_hyper = parent_hyper.child_specs[2].hyper
        local Q_s = recompose_precision(:$(s_model_type), s_spec_hyper.Q_template, 1.0; extra_param=$(s_rho_val))
        local Q_t = recompose_precision(:$(t_model_type), t_spec_hyper.Q_template, 1.0; extra_param=$(t_rho_val))
    """

    cholesky_dense_code = """
        # --- Spatiotemporal Interaction (Cholesky, AD-Safe): $(spec.key) ---
        let
            $(cholesky_base_code)
            local C_s = cholesky(Symmetric(Matrix(Q_s) + M.noise * I))
            local C_t = cholesky(Symmetric(Matrix(Q_t) + M.noise * I))
            local Z_matrix = reshape($(p_names.innovations), $(s_N), $(t_N))
            local tmp_spatial = C_s.L' \\ Z_matrix
            local st_field_unscaled = transpose(C_t.L' \\ transpose(tmp_spatial))
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * ($(s_N) * $(t_N))), sum(st_field_unscaled))
            local st_field = st_field_unscaled .* $(p_names.sigma)
            $(eta_target) .+= view(st_field, M.st_idx)
        end
    """

    cholesky_sparse_code = """
        # --- Spatiotemporal Interaction (Sparse Cholesky, Not AD-Safe): $(spec.key) ---
        let
            $(cholesky_base_code)
            local C_s = cholesky(Symmetric(Q_s + M.noise * I))
            local C_t = cholesky(Symmetric(Q_t + M.noise * I))
            local Z_matrix = reshape($(p_names.innovations), $(s_N), $(t_N))
            local tmp_spatial = C_s.L' \\ Z_matrix
            local st_field_unscaled = transpose(C_t.L' \\ transpose(tmp_spatial))
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * ($(s_N) * $(t_N))), sum(st_field_unscaled))
            local st_field = st_field_unscaled .* $(p_names.sigma)
            $(eta_target) .+= view(st_field, M.st_idx)
        end
    """

    if m.method == :cholesky
        return cholesky_dense_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    else
        @warn "Method '$(m.method)' for TensorProductSmooth not supported. Defaulting to :cholesky."
        return cholesky_dense_code
    end
end

function get_effects(
    m::TensorProductSmooth, chain, M::NamedTuple, n_samples::Int, is_multivariate_model::Bool,
    outcomes_N::Int, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    child_specs = spec.hyper.child_specs
    s_spec, t_spec = child_specs[1], child_specs[2]
    s_N, t_N = s_spec.hyper.n_latent, t_spec.hyper.n_latent
    noise = M.noise

    s_model_type = Symbol(lowercase(string(typeof(s_spec.component_obj))))
    t_model_type = Symbol(lowercase(string(typeof(t_spec.component_obj))))
    
    p_names_vec = string.(FlexiChains.parameters(chain))

    s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, get(PS, :s_idx, []))
    t_idx_full = isnothing(PS) ? M.t_idx : vcat(M.t_idx, get(PS, :t_idx, []))
    st_idx_full = (t_idx_full .- 1) .* s_N .+ s_idx_full

    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        s_v = generate_full_variable_names(s_spec, M.model_arch, k)
        t_v = generate_full_variable_names(t_spec, M.model_arch, k)

        sigma_name = _find_parameter(p_names_vec, v.sigma, k, is_multivariate_model)
        innovations_name = _find_parameter(p_names_vec, v.innovations, k, is_multivariate_model)
        
        s_rho_name = hasproperty(s_spec.component_obj, :rho) ? _find_parameter(p_names_vec, s_v.rho, k, is_multivariate_model) : ""
        t_rho_name = hasproperty(t_spec.component_obj, :rho) ? _find_parameter(p_names_vec, t_v.rho, k, is_multivariate_model) : ""

        if isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for TensorProductSmooth component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innovations_name, s_N * t_N)
        
        s_rho_samples = !isempty(s_rho_name) ? get_params_vector(chain, s_rho_name, 1)[:, 1] : nothing
        t_rho_samples = !isempty(t_rho_name) ? get_params_vector(chain, t_rho_name, 1)[:, 1] : nothing

        st_field_samples = zeros(Float64, s_N * t_N, n_samples)

        for i in 1:n_samples
            s_rho_val = isnothing(s_rho_samples) ? nothing : s_rho_samples[i]
            t_rho_val = isnothing(t_rho_samples) ? nothing : t_rho_samples[i]
            
            Q_s = recompose_precision(s_model_type, s_spec.hyper.Q_template, 1.0; extra_param=s_rho_val)
            Q_t = recompose_precision(t_model_type, t_spec.hyper.Q_template, 1.0; extra_param=t_rho_val)
            
            C_s = cholesky(Symmetric(Matrix(Q_s) + noise * I))
            C_t = cholesky(Symmetric(Matrix(Q_t) + noise * I))
            
            Z_matrix = reshape(innovations_samples[i, :], s_N, t_N)
            tmp_spatial = C_s.L' \ Z_matrix
            st_field_unscaled = transpose(C_t.L' \ transpose(tmp_spatial))
            
            st_field = st_field_unscaled .* sigma_samples[i]
            st_field_samples[:, i] = vec(st_field)
        end
        
        indexed_effects = st_field_samples[st_idx_full, :]
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
