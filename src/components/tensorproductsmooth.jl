"""
    TensorProductSmooth <: ComponentModel

A component for creating inseparable spatiotemporal effects by taking the tensor
product of marginal (e.g., spatial and temporal) components. This is typically
specified in the formula via the Kronecker product operator `⊗`.

# Version
v1.0.1 (2026-08-09)

# Mathematical Summary
This component models an inseparable spatiotemporal random effect \$\\boldsymbol{\\delta}\$
as a zero-mean Gaussian Process with a separable covariance structure. Given a
spatial precision matrix \$\\mathbf{Q}_s\$ and a temporal precision matrix \$\\mathbf{Q}_t\$,
the joint spatiotemporal precision matrix is the Kronecker product of the marginals:

\$\\mathbf{Q}_{st} = \\mathbf{Q}_t \\otimes \\mathbf{Q}_s\$

This structure implies that the covariance function is separable, i.e.,
\$K_{st}((s_1, t_1), (s_2, t_2)) = K_s(s_1, s_2) \\cdot K_t(t_1, t_2)\$. This is
different from the additive structure of the Knorr-Held Type IV interaction, which
is \$\\mathbf{Q}_{st} = \\mathbf{Q}_t \\otimes \\mathbf{I}_s + \\mathbf{I}_t \\otimes \\mathbf{Q}_s\$.

For efficient sampling, the model uses a non-centered parameterization and solves
the system \$\\mathbf{L}_{st}' \\boldsymbol{\\delta} = \\boldsymbol{\\epsilon}\$ (where
\$\\mathbf{Q}_{st} = \\mathbf{L}_{st}\\mathbf{L}_{st}'\$ and \$\\boldsymbol{\\epsilon} \\sim N(0,I)\$)
by leveraging the identity \$\\text{vec}(\\mathbf{A X B}) = (\\mathbf{B}^T \\otimes \\mathbf{A}) \\text{vec}(\\mathbf{X})\$.
This allows the use of efficient solvers on the smaller marginal matrices.

# Best Use Case
Modeling spatiotemporal interactions where the temporal evolution of the spatial
pattern can be considered multiplicative in the covariance domain. It is a
computationally efficient way to introduce flexible, inseparable space-time dynamics.

# Key References
- Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time variation
  in disease risk. *Statistics in medicine*, 19(17-18), 2555-2567.
- Saatçi, Y. (2011). *Scalable and exact Gaussian process inference*. (PhD thesis,
  University of Cambridge). Discusses Kronecker methods for GPs.
- Wikipedia: Kronecker product.

# Fields
- `components::Vector{ComponentModel}`: The child components being combined.
- `sigma::Distribution`: The prior for the std. dev. of the interaction effect.
"""
struct TensorProductSmooth <: ComponentModel
    components::Vector{ComponentModel}
    sigma::Distribution
end

COMPONENT_TYPE_REGISTRY[:tensorproductsmooth] = TensorProductSmooth
COMPONENT_CONSTRUCTORS[:tensorproductsmooth] = (p, params) -> begin
    components = get(params, :components, error("TensorProductSmooth requires child components."))
    TensorProductSmooth(components, p.sigma)
end
COMPONENT_CONSTRUCTORS[:interaction] = COMPONENT_CONSTRUCTORS[:tensorproductsmooth]

MODEL_TO_STRUCTURE_MAP[:tensorproductsmooth] = :spacetime

"""
    get_datastructures!(m_type::Type{<:TensorProductSmooth}, M::Dict, mod_data::Dict)::Bool

Delegates data structure setup to child components and then computes the combined
spatiotemporal index `st_idx` required for mapping the effect.
"""
function get_datastructures!(m_type::Type{<:TensorProductSmooth}, M::Dict, mod_data::Dict)::Bool
    child_nodes = get(mod_data[:params], :components, [])
    
    if !haskey(mod_data, :component_obj)
        error("TensorProductSmooth's get_datastructures! requires a temporary component object.")
    end

    for (i, child_node) in enumerate(child_nodes)
        child_component_obj = mod_data[:component_obj].components[i]
        child_component_type = typeof(child_component_obj)
        
        child_mod_data = Dict(
            :key => Symbol("$(mod_data[:key])_child_$(i)"),
            :type => child_node.module_type,
            :variables => get(child_node.args, :positional_args, []),
            :params => child_node.args,
            :component_obj => child_component_obj
        )
        
        get_datastructures!(child_component_type, M, child_mod_data)
    end

    if haskey(M, :s_idx) && haskey(M, :t_idx) && haskey(M, :s_N)
        M[:st_idx] = (M[:t_idx] .- 1) .* M[:s_N] .+ M[:s_idx]
    else
        @warn "Could not compute spatiotemporal index `st_idx` for '$(mod_data[:key])'."
    end
    
    return true
end

"""
    get_precomputes(m::TensorProductSmooth, M::NamedTuple, mod_data::Dict)::NamedTuple

Delegates pre-computation to child components and aggregates their results.
"""
function get_precomputes(m::TensorProductSmooth, M::NamedTuple, mod_data::Dict)::NamedTuple
    child_nodes = get(mod_data[:params], :components, [])
    child_precomputes_list = []
    child_specs_list = []

    for (i, child_node) in enumerate(child_nodes)
        child_component_obj = m.components[i]
        child_mod_data = Dict(
            :key => Symbol("$(mod_data[:key])_child_$(i)"),
            :type => child_node.module_type,
            :variables => get(child_node.args, :positional_args, []),
            :params => child_node.args
        )
        precomputes = get_precomputes(child_component_obj, M, child_mod_data)
        push!(child_precomputes_list, precomputes)

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

    return (child_specs = child_specs_list,)
end

"""
    get_priors(m::TensorProductSmooth, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for the interaction scale `sigma` and the latent innovations `raw`.
"""
function get_priors(m::TensorProductSmooth, spec::NamedTuple, arch::String, outcome_idx, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    child_specs = spec.hyper.child_specs
    
    s_N = child_specs[1].hyper.n_latent
    t_N = child_specs[2].hyper.n_latent
    
    return """
    # Priors for Spatiotemporal Interaction: $(spec.key)
    $(p_names.sigma) ~ $(_distribution_to_string(m.sigma))
    $(p_names.raw) ~ MvNormal(zeros($(s_N * t_N)), I)
    """
end

"""
    get_updates(m::TensorProductSmooth, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the spatiotemporal interaction effect using a
Kronecker solver.
"""
function get_updates(m::TensorProductSmooth, spec::NamedTuple, arch::String, outcome_idx, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    child_specs = spec.hyper.child_specs

    s_spec = child_specs[1]
    t_spec = child_specs[2]
    
    s_model_type = Symbol(lowercase(string(typeof(s_spec.component_obj))))
    t_model_type = Symbol(lowercase(string(typeof(t_spec.component_obj))))
    
    s_rho_val = hasproperty(s_spec.component_obj, :rho) ? generate_full_variable_names(s_spec, arch, outcome_idx).rho : "nothing"
    t_rho_val = hasproperty(t_spec.component_obj, :rho) ? generate_full_variable_names(t_spec, arch, outcome_idx).rho : "nothing"

    return """
    # --- Spatiotemporal Interaction Update: $(spec.key) ---
    local Q_s = recompose_precision(:$(s_model_type), spec.hyper.child_specs[1].hyper.Q_template, 1.0; extra_param=$(s_rho_val))
    local Q_t = recompose_precision(:$(t_model_type), spec.hyper.child_specs[2].hyper.Q_template, 1.0; extra_param=$(t_rho_val))
    
    local C_s = cholesky(Symmetric(Matrix(Q_s) + M.noise * I))
    local C_t = cholesky(Symmetric(Matrix(Q_t) + M.noise * I))
    
    local Z_matrix = reshape($(p_names.raw), M.s_N, M.t_N)
    
    local tmp_spatial = C_s.L' \\ Z_matrix
    local st_field_unscaled = transpose(C_t.L' \\ transpose(tmp_spatial))
    
    Turing.@addlogprob! logpdf(Normal(0, 0.001 * (M.s_N * M.t_N)), sum(st_field_unscaled))
    
    local st_field = st_field_unscaled .* $(p_names.sigma)
    
    $(eta_target) .+= view(st_field, M.st_idx)
    """
end

"""
    get_effects(m::TensorProductSmooth, chain, M, n_samples, outcomes_N, spec, PS, N_total)::NamedTuple

Reconstructs the posterior of the spatiotemporal interaction field.
"""
function get_effects(m::TensorProductSmooth, chain, M, n_samples, outcomes_N, spec, PS, N_total)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    child_specs = spec.hyper.child_specs
    s_spec = child_specs[1]
    t_spec = child_specs[2]
    
    s_N = s_spec.hyper.n_latent
    t_N = t_spec.hyper.n_latent
    noise = M.noise

    s_model_type = Symbol(lowercase(string(typeof(s_spec.component_obj))))
    t_model_type = Symbol(lowercase(string(typeof(t_spec.component_obj))))
    
    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        s_p_names = generate_full_variable_names(s_spec, M.model_arch, k)
        t_p_names = generate_full_variable_names(t_spec, M.model_arch, k)

        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)
        raw_samples = get_params_vector(chain, string(p_names.raw), s_N * t_N)
        
        s_rho_samples = hasproperty(s_spec.component_obj, :rho) ? get_params_vector(chain, string(s_p_names.rho), 1) : nothing
        t_rho_samples = hasproperty(t_spec.component_obj, :rho) ? get_params_vector(chain, string(t_p_names.rho), 1) : nothing

        st_field_samples = zeros(Float64, s_N * t_N, n_samples)

        for i in 1:n_samples
            s_rho_val = isnothing(s_rho_samples) ? nothing : s_rho_samples[i, 1]
            t_rho_val = isnothing(t_rho_samples) ? nothing : t_rho_samples[i, 1]

            Q_s = recompose_precision(s_model_type, s_spec.hyper.Q_template, 1.0; extra_param=s_rho_val)
            Q_t = recompose_precision(t_model_type, t_spec.hyper.Q_template, 1.0; extra_param=t_rho_val)
            
            C_s = cholesky(Symmetric(Matrix(Q_s) + noise * I))
            C_t = cholesky(Symmetric(Matrix(Q_t) + noise * I))
            
            Z_matrix = reshape(raw_samples[i, :], s_N, t_N)
            
            tmp_spatial = C_s.L' \ Z_matrix
            st_field_unscaled = transpose(C_t.L' \ transpose(tmp_spatial))
            
            st_field = st_field_unscaled .* sigma_samples[i, 1]
            st_field_samples[:, i] = vec(st_field)
        end
        push!(structured_effects, st_field_samples)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
