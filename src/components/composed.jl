"""
    Composed <: ComponentModel

An orchestrator component that combines multiple child components using a specified
operator. This component is the backbone for creating complex interactions, such as
spatiotemporal models, spatially varying coefficients, and non-stationary variance
models.

# Version
v1.0.0 (2026-08-08)

# Mathematical Summary & Operators

The `Composed` component implements different model structures based on the `operator`
field:

1.  **`:pipe` (`|>`): Spatially Varying Curves & State-Space Models**
    -   **Use Case**: `random(t, model=pspline) |> random(s, model=icar)`
    -   **Math**: Models a set of basis coefficients \$\\beta(s)\$ as a spatial field.
        The final effect is \$f(s,t) = \\sum_j B_j(t) \\beta_j(s)\$, where \$B_j(t)\$ are
        temporal basis functions and \$\\beta_j(s)\$ are spatially correlated
        coefficients.

2.  **`:kronecker_product` (`⊗`): Spatiotemporal Interactions**
    -   **Use Case**: `random(s, model=icar) ⊗ random(t, model=ar1)`
    -   **Math**: Implements Knorr-Held interaction models. The joint precision
        matrix is the Kronecker product of the marginal precision matrices:
        \$Q_{st} = Q_s \\otimes Q_t\$. This creates a fully structured spatiotemporal
        random effect.

3.  **`:composition` (`∘`): Non-Stationary Variance & Functional Composition**
    -   **Use Case**: `random(s, model=icar) ∘ random(x, model=pspline)`
    -   **Math**: Models non-stationary processes where one component modulates the
        parameters of another. A common use is for non-stationary variance, where a
        smoother on a covariate `x` defines a spatially varying standard deviation
        \$\\sigma(x)\$ for a base spatial field \$\\phi(s)\$. The final effect is
        \$\\phi(s) \\cdot \\sigma(x)\$.

# Key References
- Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time variation
  in disease risk. *Statistical Methods in Medical Research*. (For Kronecker
  product interactions).
- Gelfand, A. E., et al. (2003). Spatial modeling with spatially varying
  coefficient processes. *Journal of the American Statistical Association*. (For
  concepts related to spatially varying curves).

# Fields
- `components::Vector{ComponentModel}`: The child components being combined.
- `operator::Symbol`: The operator defining the composition (e.g., `:pipe`,
  `:kronecker_product`, `:composition`).
"""
struct Composed <: ComponentModel
    components::Vector{ComponentModel}
    operator::Symbol
end

COMPONENT_TYPE_REGISTRY[:composed] = Composed

COMPONENT_CONSTRUCTORS[:composed] = (p, params) -> begin
    components = get(params, :components, error("Composed requires child components."))
    operator = get(params, :operator, error("Composed requires an operator."))
    Composed(components, operator)
end

MODEL_TO_STRUCTURE_MAP[:composed] = :any

"""
    get_datastructures!(m_type::Type{<:Composed}, M::Dict, mod_data::Dict)::Bool

Delegates data structure setup to the child components, ensuring that all necessary
indices and data are processed before the pre-computation step.
"""
function get_datastructures!(
    m_type::Type{<:Composed}, M::Dict, mod_data::Dict
)::Bool
    child_nodes = get(mod_data[:params], :components, [])
    
    if !haskey(mod_data, :component_obj)
        error(
            "Composed's get_datastructures! requires a temporary component object " *
            "in mod_data."
        )
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
    return true
end

"""
    get_precomputes(m::Composed, M::NamedTuple, mod_data::Dict)::NamedTuple

Delegates pre-computation to child components and aggregates their results into a
`child_specs` list for use by the code generators.
"""
function get_precomputes(m::Composed, M::NamedTuple, mod_data::Dict)::NamedTuple
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
            structure = get(
                MODEL_TO_STRUCTURE_MAP, typeof(child_component_obj), :unknown
            ),
            var = join(child_mod_data[:variables], "_"),
            component_obj = child_component_obj,
            params = child_mod_data[:params],
            hyper = precomputes
        )
        push!(child_specs_list, child_spec)
    end

    return (
        operator = m.operator,
        child_precomputes = child_precomputes_list,
        child_specs = child_specs_list
    )
end

"""
    get_priors(m::Composed, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for the `Composed` component, dispatching on the operator to
create the correct hierarchical prior structure.
"""
function get_priors(
    m::Composed, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    child_specs = spec.hyper.child_specs
    
    if m.operator == :pipe # Spatially varying curve
        dynamic_spec = child_specs[1]
        state_spec = child_specs[2]
        
        state_priors = get_priors(
            state_spec.component_obj, state_spec, arch, outcome_idx, M
        )
        
        n_spatial = state_spec.hyper.n_latent
        n_basis = dynamic_spec.hyper.n_latent
        coeffs_prior = "$(p_names.raw) ~ MvNormal(zeros(T, $(n_spatial * n_basis)), I)"
        
        return """
        # Priors for Spatially Varying Curve: $(spec.key)
        $(state_priors)
        $(coeffs_prior)
        """
    elseif m.operator == :kronecker_product # Spatiotemporal interaction
        st_sigma_prior = get(spec.params, :sigma, Exponential(1.0))
        s_N = child_specs[1].hyper.n_latent
        t_N = child_specs[2].hyper.n_latent
        
        return """
        # Priors for Spatiotemporal Interaction: $(spec.key)
        $(p_names.sigma) ~ $(_distribution_to_string(st_sigma_prior))
        $(p_names.raw) ~ MvNormal(zeros(T, $(s_N * t_N)), I)
        """
    elseif m.operator == :composition # Non-stationary variance
        modifier_spec = child_specs[1]
        base_spec = child_specs[2]

        modifier_priors = get_priors(
            modifier_spec.component_obj, modifier_spec, arch, outcome_idx, M
        )

        base_priors_acc = []
        base_p_names = generate_full_variable_names(base_spec, arch, outcome_idx)
        if hasproperty(base_spec.component_obj, :rho)
            rho_prior_str = _distribution_to_string(base_spec.component_obj.rho)
            push!(base_priors_acc, "$(base_p_names.rho) ~ $(rho_prior_str)")
        end
        
        n_latent_base = base_spec.hyper.n_latent
        push!(
            base_priors_acc,
            "$(base_p_names.raw) ~ MvNormal(zeros(T, $(n_latent_base)), I)"
        )
        base_priors = join(base_priors_acc, "\n    ")

        return """
        # Priors for Composition (NonStationaryVariance): $(spec.key)
        $(modifier_priors)
        $(base_priors)
        """
    else
        all_priors = [
            get_priors(cs.component_obj, cs, arch, outcome_idx, M)
            for cs in child_specs
        ]
        return join(filter(!isempty, all_priors), "\n    ")
    end
end

"""
    get_updates(m::Composed, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the effect of the `Composed` component, dispatching
on the operator to construct the correct interaction logic.
"""
function get_updates(
    m::Composed, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    child_specs = spec.hyper.child_specs

    if m.operator == :pipe # Spatially varying curve
        dynamic_spec = child_specs[1]
        state_spec = child_specs[2]
        
        state_p_names = generate_full_variable_names(state_spec, arch, outcome_idx)
        state_model_type = Symbol(lowercase(string(typeof(state_spec.component_obj))))
        rho_value = hasproperty(state_spec.component_obj, :rho) ?
                    string(state_p_names.rho) : "nothing"
        
        dynamic_basis_key = Symbol(join(get(dynamic_spec.params, :positional_args, []), "_"))
        basis_matrix = "M.basis_matrices[:$(dynamic_basis_key)]"
        
        n_spatial = state_spec.hyper.n_latent
        n_basis = dynamic_spec.hyper.n_latent

        return """
        # --- Spatially Varying Curve Update: $(spec.key) ---
        local Q_spatial_template = spec_registry[:$(state_spec.key)].hyper.Q_template
        local Q_spatial = recompose_precision(
            :$(state_model_type), Q_spatial_template, 1.0; extra_param=$(rho_value)
        )
        local F_spatial = cholesky(Symmetric(Matrix(Q_spatial) + M.noise * I))
        
        local coeffs_raw_matrix = reshape($(p_names.raw), $(n_spatial), $(n_basis))
        
        local spatial_coeffs = F_spatial.L' \\ coeffs_raw_matrix
        
        local final_effect = sum(T.($(basis_matrix)) .* spatial_coeffs[M.s_idx, :], dims=2)
        
        $(eta_target) .+= final_effect
        """
    elseif m.operator == :kronecker_product # Spatiotemporal interaction
        s_spec = child_specs[1]
        t_spec = child_specs[2]
        
        s_model_type = Symbol(lowercase(string(typeof(s_spec.component_obj))))
        t_model_type = Symbol(lowercase(string(typeof(t_spec.component_obj))))
        
        s_rho_val = hasproperty(s_spec.component_obj, :rho) ?
                    string(generate_full_variable_names(s_spec, arch, outcome_idx).rho) :
                    "nothing"
        t_rho_val = hasproperty(t_spec.component_obj, :rho) ?
                    string(generate_full_variable_names(t_spec, arch, outcome_idx).rho) :
                    "nothing"

        return """
        # --- Spatiotemporal Interaction Update: $(spec.key) ---
        local Q_s = recompose_precision(
            :$(s_model_type), spec_registry[:$(s_spec.key)].hyper.Q_template,
            1.0; extra_param=$(s_rho_val)
        )
        local Q_t = recompose_precision(
            :$(t_model_type), spec_registry[:$(t_spec.key)].hyper.Q_template,
            1.0; extra_param=$(t_rho_val)
        )
        
        local C_s = cholesky(Symmetric(Matrix(Q_s) + M.noise * I))
        local C_t = cholesky(Symmetric(Matrix(Q_t) + M.noise * I))
        
        local Z_matrix = reshape($(p_names.raw), M.s_N, M.t_N)
        
        local tmp_spatial = C_s.L' \\ Z_matrix
        local st_field_unscaled = transpose(C_t.L' \\ transpose(tmp_spatial))
        
        Turing.@addlogprob! logpdf(
            Normal(0, 0.001 * (M.s_N * M.t_N)), sum(st_field_unscaled)
        )
        
        local st_field = st_field_unscaled .* $(p_names.sigma)
        
        $(eta_target) .+= st_field[M.st_idx]
        """
    elseif m.operator == :composition # Non-stationary variance
        modifier_spec = child_specs[1]
        base_spec = child_specs[2]

        modifier_frags = get_updates(
            modifier_spec.component_obj, modifier_spec, arch, outcome_idx, M
        )
        modifier_update_cleaned = replace(modifier_frags, Regex("$(eta_target) \\.\\+= .*") => "")
        modifier_latent_var = generate_full_variable_names(
            modifier_spec, arch, outcome_idx
        ).latent

        base_p_names = generate_full_variable_names(base_spec, arch, outcome_idx)
        base_model_type = Symbol(lowercase(string(typeof(base_spec.component_obj))))
        
        base_latent_reconstruction_code = """
            local Q_base_template = spec_registry[:$(base_spec.key)].hyper.Q_template
            local F_base = cholesky(Symmetric(Matrix(Q_base_template) + M.noise * I))
            local base_latent_raw::Vector{T} = F_base.L' \\ $(base_p_names.raw)
            
            if $(base_model_type) in [:icar, :besag]
                Turing.@addlogprob! logpdf(
                    Normal(0.0, 0.001 * $(base_spec.hyper.n_latent)),
                    sum(base_latent_raw)
                )
            end
        """
        
        modifier_basis_key = Symbol(join(get(modifier_spec.params, :positional_args, []), "_"))

        return """
        # --- Composition (NonStationaryVariance) Update: $(spec.key) ---
        
        $(modifier_update_cleaned)
        local log_sigma_field::Vector{T} = M.basis_matrices[:$(modifier_basis_key)] * $(modifier_latent_var)
        local spatially_varying_sigma::Vector{T} = exp.(log_sigma_field)
        
        $(base_latent_reconstruction_code)
        
        local final_effect_latent = base_latent_raw .* spatially_varying_sigma
        
        $(eta_target) .+= final_effect_latent[M.s_idx]
        """
    end
    
    return "# Operator '$(m.operator)' not implemented for get_updates in Composed"
end

"""
    get_effects(m::Composed, chain, M::NamedTuple, ...)::NamedTuple

Reconstructs the `Composed`'s effect from the MCMC chain's posterior samples.
"""
function get_effects(
    m::Composed, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    child_specs = spec.hyper.child_specs
    structured_effects = [zeros(Float64, N_total, n_samples) for _ in 1:outcomes_N]

    if m.operator == :pipe # Spatially varying curve
        dynamic_spec = child_specs[1]
        state_spec = child_specs[2]
        
        dynamic_basis_key = Symbol(join(get(dynamic_spec.params, :positional_args, []), "_"))
        B_dynamic_train = M.basis_matrices[dynamic_basis_key]
        B_dynamic_full = if !isnothing(PS) && haskey(PS, :basis_matrices) &&
                            haskey(PS.basis_matrices, dynamic_basis_key)
            vcat(B_dynamic_train, PS.basis_matrices[dynamic_basis_key])
        else
            B_dynamic_train
        end
        
        n_spatial = state_spec.hyper.n_latent
        n_basis = dynamic_spec.hyper.n_latent
        s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)

        for k in 1:outcomes_N
            p_names = generate_full_variable_names(spec, M.model_arch, k)
            state_p_names = generate_full_variable_names(state_spec, M.model_arch, k)
            
            coeffs_raw_samples = get_params_vector(
                chain, string(p_names.raw), n_spatial * n_basis
            )
            
            state_model_type = Symbol(lowercase(string(typeof(state_spec.component_obj))))
            Q_spatial_template = state_spec.hyper.Q_template
            
            effect_k = zeros(Float64, N_total, n_samples)

            for i in 1:n_samples
                rho_val = hasproperty(state_spec.component_obj, :rho) ?
                          get_params_vector(chain, string(state_p_names.rho), 1)[i, 1] :
                          nothing
                Q_spatial = recompose_precision(
                    state_model_type, Q_spatial_template, 1.0; extra_param=rho_val
                )
                F_spatial = cholesky(Symmetric(Matrix(Q_spatial) + M.noise * I))
                
                coeffs_raw_matrix = reshape(coeffs_raw_samples[i, :], n_spatial, n_basis)
                spatial_coeffs = F_spatial.L' \ coeffs_raw_matrix
                
                effect_k[:, i] = sum(B_dynamic_full .* spatial_coeffs[s_idx_full, :], dims=2)
            end
            structured_effects[k] = effect_k
        end
        return (structured=structured_effects, noisy=structured_effects)

    elseif m.operator == :composition # Non-stationary variance
        modifier_spec = child_specs[1]
        base_spec = child_specs[2]
        
        for k in 1:outcomes_N
            modifier_p_names = generate_full_variable_names(
                modifier_spec, M.model_arch, k
            )
            modifier_raw_samples = get_params_vector(
                chain, string(modifier_p_names.raw), modifier_spec.hyper.n_latent
            )
            modifier_sigma_samples = get_params_vector(
                chain, string(modifier_p_names.sigma), 1
            )[:, 1]

            U_mod, L_mod = modifier_spec.hyper.U, modifier_spec.hyper.L
            diff_order_mod = get(modifier_spec.component_obj, :diff_order, 0)
            modifier_basis_key = Symbol(join(get(modifier_spec.params, :positional_args, []), "_"))
            B_modifier = M.basis_matrices[modifier_basis_key]

            log_sigma_field_samples = zeros(n_samples, M.s_N)
            for i in 1:n_samples
                diag_D_mod = modifier_sigma_samples[i] ./ sqrt.(L_mod .+ M.noise)
                for j in 1:diff_order_mod; diag_D_mod[j] = 0.0; end
                modifier_coeffs = U_mod * (diag_D_mod .* modifier_raw_samples[i, :])
                log_sigma_field_samples[i, :] = B_modifier * modifier_coeffs
            end
            spatially_varying_sigma_samples = exp.(log_sigma_field_samples)

            base_p_names = generate_full_variable_names(base_spec, M.model_arch, k)
            base_raw_samples = get_params_vector(
                chain, string(base_p_names.raw), base_spec.hyper.n_latent
            )
            Q_base_template = base_spec.hyper.Q_template
            F_base = cholesky(Symmetric(Matrix(Q_base_template) + M.noise * I))
            base_latent_raw_samples = hcat([F_base.L' \ s for s in eachrow(base_raw_samples)]...)'

            reconstructed_effects = base_latent_raw_samples .* spatially_varying_sigma_samples
            structured_effects[k] = reconstructed_effects'
        end
        return (structured=structured_effects, noisy=structured_effects)
    else
        @warn "Posterior reconstruction for Composed with operator :$(m.operator) " *
              "is not fully implemented. Returning zero effects."
        return (structured=structured_effects, noisy=structured_effects)
    end
end
