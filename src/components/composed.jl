"""
    Composed <: ComponentModel

An orchestrator component that combines multiple child components using a specified
operator. This component is the backbone for creating complex interactions, such as
spatiotemporal models, spatially varying coefficients, and non-stationary variance
models.

# Version
v1.1.0 (2026-08-11)

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

# Inputs
- **Required**:
  - Two or more `random()` modules combined with an operator (`|>`, `⊗`, `∘`).
- **Optional**:
  - `method`: A `Symbol` specifying the computational method.

# Outputs (Parameter Names)
- **`:pipe`**: `innovations_<key>`, `sigma_<state_key>`, `rho_<state_key>`
- **`:kronecker_product`**: `sigma_<key>`, `innovations_<key>`
- **`:composition`**: Parameters from the child components.
"""
struct Composed <: ComponentModel
    components::Vector{ComponentModel}
    operator::Symbol
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:composed] = Composed

COMPONENT_CONSTRUCTORS[:composed] = (p, params) -> begin
    components = get(params, :components, error("Composed requires child components."))
    operator = get(params, :operator, error("Composed requires an operator."))
    
    default_method = if operator == :kronecker_product
        :spectral
    elseif operator == :pipe
        :cholesky
    else
        :none
    end
    
    method = get(params, :method, default_method)
    Composed(components, operator, method)
end

MODEL_TO_STRUCTURE_MAP[:composed] = :any

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
        coeffs_prior = "$(p_names.innovations) ~ MvNormal(zeros(T, $(n_spatial * n_basis)), I)"
        
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
        $(p_names.innovations) ~ MvNormal(zeros(T, $(s_N * t_N)), I)
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
            "$(base_p_names.innovations) ~ MvNormal(zeros(T, $(n_latent_base)), I)"
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

function get_updates(
    m::Composed, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    child_specs = spec.hyper.child_specs

    if m.operator == :pipe
        dynamic_spec = child_specs[1]
        state_spec = child_specs[2]
        
        state_p_names = generate_full_variable_names(state_spec, arch, outcome_idx)
        state_model_type = Symbol(lowercase(string(typeof(state_spec.component_obj))))
        rho_value = hasproperty(state_spec.component_obj, :rho) ? string(state_p_names.rho) : "nothing"
        
        dynamic_basis_key = Symbol(join(get(dynamic_spec.params, :positional_args, []), "_"))
        basis_matrix = "M.basis_matrices[:$(dynamic_basis_key)]"
        
        n_spatial = state_spec.hyper.n_latent
        n_basis = dynamic_spec.hyper.n_latent

        cholesky_base_code = """
            Q_spatial_template = spec_registry[:$(state_spec.key)].hyper.Q_template
            Q_spatial = recompose_precision(:$(state_model_type), Q_spatial_template, 1.0; extra_param=$(rho_value))
        """

        cholesky_dense_code = """
            # --- Spatially Varying Curve (Cholesky): $(spec.key) ---
            let
                $(cholesky_base_code)
                F_spatial = cholesky(Symmetric(Matrix(Q_spatial) + M.noise * I))
                coeffs_raw_matrix = reshape($(p_names.innovations), $(n_spatial), $(n_basis))
                spatial_coeffs = F_spatial.L' \\ coeffs_raw_matrix
                final_effect = sum($(basis_matrix) .* spatial_coeffs[M.s_idx, :], dims=2)
                $(eta_target) .+= final_effect
            end
        """

        cholesky_sparse_code = """
            # --- Spatially Varying Curve (Sparse Cholesky, Not AD-Safe): $(spec.key) ---
            let
                $(cholesky_base_code)
                F_spatial = cholesky(Symmetric(Q_spatial + M.noise * I))
                coeffs_raw_matrix = reshape($(p_names.innovations), $(n_spatial), $(n_basis))
                spatial_coeffs = F_spatial.L' \\ coeffs_raw_matrix
                final_effect = sum($(basis_matrix) .* spatial_coeffs[M.s_idx, :], dims=2)
                $(eta_target) .+= final_effect
            end
        """
        
        if m.method == :cholesky; return cholesky_dense_code;
        elseif m.method == :cholesky_sparse; return cholesky_sparse_code;
        else; error("Unsupported method '$(m.method)' for pipe operator."); end

    elseif m.operator == :kronecker_product
        s_spec = child_specs[1]
        t_spec = child_specs[2]
        
        s_model_type = Symbol(lowercase(string(typeof(s_spec.component_obj))))
        t_model_type = Symbol(lowercase(string(typeof(t_spec.component_obj))))
        
        s_rho_val = hasproperty(s_spec.component_obj, :rho) ? string(generate_full_variable_names(s_spec, arch, outcome_idx).rho) : "nothing"
        t_rho_val = hasproperty(t_spec.component_obj, :rho) ? string(generate_full_variable_names(t_spec, arch, outcome_idx).rho) : "nothing"

        spectral_code = """
            # --- Spatiotemporal Interaction (Spectral): $(spec.key) ---
            let
                s_hyper = spec_registry[:$(s_spec.key)].hyper
                t_hyper = spec_registry[:$(t_spec.key)].hyper
                
                diag_Ls = (1.0 .- $(s_rho_val)) .+ $(s_rho_val) .* s_hyper.L
                diag_Lt = (1.0 .- $(t_rho_val)) .+ $(t_rho_val) .* t_hyper.L
                
                diag_D_s = $(p_names.sigma) ./ sqrt.(diag_Ls .+ M.noise)
                diag_D_t = 1.0 ./ sqrt.(diag_Lt .+ M.noise)
                
                Z_matrix = reshape($(p_names.innovations), M.s_N, M.t_N)
                
                tmp = s_hyper.U' * Z_matrix * t_hyper.U
                transformed = (diag_D_s .* tmp) .* diag_D_t'
                st_field = s_hyper.U * transformed * t_hyper.U'
                
                $(eta_target) .+= st_field[M.st_idx]
            end
        """

        cholesky_base_code = """
            Q_s = recompose_precision(:$(s_model_type), spec_registry[:$(s_spec.key)].hyper.Q_template, 1.0; extra_param=$(s_rho_val))
            Q_t = recompose_precision(:$(t_model_type), spec_registry[:$(t_spec.key)].hyper.Q_template, 1.0; extra_param=$(t_rho_val))
        """

        cholesky_dense_code = """
            # --- Spatiotemporal Interaction (Cholesky): $(spec.key) ---
            let
                $(cholesky_base_code)
                C_s = cholesky(Symmetric(Matrix(Q_s) + M.noise * I))
                C_t = cholesky(Symmetric(Matrix(Q_t) + M.noise * I))
                Z_matrix = reshape($(p_names.innovations), M.s_N, M.t_N)
                tmp_spatial = C_s.L' \\ Z_matrix
                st_field_unscaled = transpose(C_t.L' \\ transpose(tmp_spatial))
                Turing.@addlogprob! logpdf(Normal(0, 0.001 * (M.s_N * M.t_N)), sum(st_field_unscaled))
                st_field = st_field_unscaled .* $(p_names.sigma)
                $(eta_target) .+= st_field[M.st_idx]
            end
        """

        cholesky_sparse_code = """
            # --- Spatiotemporal Interaction (Sparse Cholesky, Not AD-Safe): $(spec.key) ---
            let
                $(cholesky_base_code)
                C_s = cholesky(Symmetric(Q_s + M.noise * I))
                C_t = cholesky(Symmetric(Q_t + M.noise * I))
                Z_matrix = reshape($(p_names.innovations), M.s_N, M.t_N)
                tmp_spatial = C_s.L' \\ Z_matrix
                st_field_unscaled = transpose(C_t.L' \\ transpose(tmp_spatial))
                Turing.@addlogprob! logpdf(Normal(0, 0.001 * (M.s_N * M.t_N)), sum(st_field_unscaled))
                st_field = st_field_unscaled .* $(p_names.sigma)
                $(eta_target) .+= st_field[M.st_idx]
            end
        """

        if m.method == :spectral; return spectral_code;
        elseif m.method == :cholesky; return cholesky_dense_code;
        elseif m.method == :cholesky_sparse; return cholesky_sparse_code;
        else; error("Unsupported method '$(m.method)' for Kronecker product operator."); end

    elseif m.operator == :composition # Non-stationary variance
        modifier_spec = child_specs[1]
        base_spec = child_specs[2]

        modifier_frags = get_updates(modifier_spec.component_obj, modifier_spec, arch, outcome_idx, M)
        modifier_update_cleaned = replace(modifier_frags, Regex("$(eta_target) \\.\\+= .*") => "")
        modifier_latent_var = generate_full_variable_names(modifier_spec, arch, outcome_idx).latent

        base_p_names = generate_full_variable_names(base_spec, arch, outcome_idx)
        base_model_type = Symbol(lowercase(string(typeof(base_spec.component_obj))))
        
        base_latent_reconstruction_code = """
            Q_base_template = spec_registry[:$(base_spec.key)].hyper.Q_template
            F_base = cholesky(Symmetric(Matrix(Q_base_template) + M.noise * I))
            base_latent_raw = F_base.L' \\ $(base_p_names.innovations)
            if $(base_model_type) in [:icar, :besag]
                Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(base_spec.hyper.n_latent)), sum(base_latent_raw))
            end
        """
        
        modifier_basis_key = Symbol(join(get(modifier_spec.params, :positional_args, []), "_"))

        return """
        # --- Composition (NonStationaryVariance) Update: $(spec.key) ---
        let
            $(modifier_update_cleaned)
            log_sigma_field = M.basis_matrices[:$(modifier_basis_key)] * $(modifier_latent_var)
            spatially_varying_sigma = exp.(log_sigma_field)
            $(base_latent_reconstruction_code)
            final_effect_latent = base_latent_raw .* spatially_varying_sigma
            $(eta_target) .+= final_effect_latent[M.s_idx]
        end
        """
    end
    
    return "# Operator '$(m.operator)' not implemented for get_updates in Composed"
end

function get_effects(
    m::Composed, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    child_specs = spec.hyper.child_specs
    structured_effects = [zeros(Float64, N_total, n_samples) for _ in 1:outcomes_N]
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))

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
            innovations_name = _find_parameter(p_names_vec, string(spec.key), "innovations", k, is_multivariate_model)
            if isempty(innovations_name)
                @warn "Innovations for Composed component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                continue
            end
            innovations_samples = get_params_vector(chain, innovations_name, n_spatial * n_basis)
            
            state_p_names = generate_full_variable_names(state_spec, M.model_arch, k)
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
                
                coeffs_raw_matrix = reshape(innovations_samples[i, :], n_spatial, n_basis)
                spatial_coeffs = F_spatial.L' \ coeffs_raw_matrix
                
                effect_k[:, i] = sum(B_dynamic_full .* spatial_coeffs[s_idx_full, :], dims=2)
            end
            structured_effects[k] = effect_k
        end
        return (structured=structured_effects, noisy=structured_effects)

    elseif m.operator == :kronecker_product
        s_spec = child_specs[1]
        t_spec = child_specs[2]
        s_N = s_spec.hyper.n_latent
        t_N = t_spec.hyper.n_latent
        st_idx_full = (t_idx_full .- 1) .* s_N .+ s_idx_full

        for k in 1:outcomes_N
            sigma_name = _find_parameter(p_names_vec, string(spec.key), "sigma", k, is_multivariate_model)
            innovations_name = _find_parameter(p_names_vec, string(spec.key), "innovations", k, is_multivariate_model)
            if isempty(sigma_name) || isempty(innovations_name)
                @warn "Parameters for Kronecker product component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                continue
            end
            sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
            innovations_samples = get_params_vector(chain, innovations_name, s_N * t_N)

            effect_k = zeros(Float64, N_total, n_samples)

            for i in 1:n_samples
                local st_field
                if m.method == :spectral
                    s_hyper = s_spec.hyper; t_hyper = t_spec.hyper
                    s_rho_val = hasproperty(s_spec.component_obj, :rho) ? get_params_vector(chain, string(generate_full_variable_names(s_spec, M.model_arch, k).rho), 1)[i, 1] : nothing
                    t_rho_val = hasproperty(t_spec.component_obj, :rho) ? get_params_vector(chain, string(generate_full_variable_names(t_spec, M.model_arch, k).rho), 1)[i, 1] : nothing
                    
                    diag_Ls = (1.0 .- s_rho_val) .+ s_rho_val .* s_hyper.L
                    diag_Lt = (1.0 .- t_rho_val) .+ t_rho_val .* t_hyper.L
                    diag_D_s = sigma_samples[i] ./ sqrt.(diag_Ls .+ M.noise)
                    diag_D_t = 1.0 ./ sqrt.(diag_Lt .+ M.noise)
                    
                    Z_matrix = reshape(innovations_samples[i, :], s_N, t_N)
                    tmp = s_hyper.U' * Z_matrix * t_hyper.U
                    transformed = (diag_D_s .* tmp) .* diag_D_t'
                    st_field = s_hyper.U * transformed * t_hyper.U'
                else # cholesky or cholesky_sparse
                    s_model_type = Symbol(lowercase(string(typeof(s_spec.component_obj))))
                    t_model_type = Symbol(lowercase(string(typeof(t_spec.component_obj))))
                    s_rho_val = hasproperty(s_spec.component_obj, :rho) ? get_params_vector(chain, string(generate_full_variable_names(s_spec, M.model_arch, k).rho), 1)[i, 1] : nothing
                    t_rho_val = hasproperty(t_spec.component_obj, :rho) ? get_params_vector(chain, string(generate_full_variable_names(t_spec, M.model_arch, k).rho), 1)[i, 1] : nothing
                    
                    Q_s = recompose_precision(s_model_type, s_spec.hyper.Q_template, 1.0; extra_param=s_rho_val)
                    Q_t = recompose_precision(t_model_type, t_spec.hyper.Q_template, 1.0; extra_param=t_rho_val)
                    
                    C_s = cholesky(Symmetric(Matrix(Q_s) + M.noise * I))
                    C_t = cholesky(Symmetric(Matrix(Q_t) + M.noise * I))
                    
                    Z_matrix = reshape(innovations_samples[i, :], s_N, t_N)
                    tmp_spatial = C_s.L' \ Z_matrix
                    st_field_unscaled = transpose(C_t.L' \ transpose(tmp_spatial))
                    st_field = st_field_unscaled .* sigma_samples[i]
                end
                effect_k[:, i] = st_field[st_idx_full]
            end
            structured_effects[k] = effect_k
        end
        return (structured=structured_effects, noisy=structured_effects)

    elseif m.operator == :composition # Non-stationary variance
        modifier_spec = child_specs[1]
        base_spec = child_specs[2]
        
        for k in 1:outcomes_N
            # Reconstruct modifier effect
            modifier_effects = get_effects(modifier_spec.component_obj, chain, M, n_samples, outcomes_N, modifier_spec, PS, N_total)
            log_sigma_field_samples = modifier_effects.structured[k]
            spatially_varying_sigma_samples = exp.(log_sigma_field_samples)

            # Reconstruct base effect
            base_p_names = generate_full_variable_names(base_spec, M.model_arch, k)
            base_innovations_name = _find_parameter(p_names_vec, string(base_spec.key), "innovations", k, is_multivariate_model)
            if isempty(base_innovations_name)
                @warn "Base innovations for Composed component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                continue
            end
            base_innovations_samples = get_params_vector(chain, base_innovations_name, base_spec.hyper.n_latent)
            
            Q_base_template = base_spec.hyper.Q_template
            F_base = cholesky(Symmetric(Matrix(Q_base_template) + M.noise * I))
            base_latent_raw_samples = hcat([F_base.L' \ s for s in eachrow(base_innovations_samples)]...)'

            # Combine
            reconstructed_effects = base_latent_raw_samples' .* spatially_varying_sigma_samples
            structured_effects[k] = reconstructed_effects
        end
        return (structured=structured_effects, noisy=structured_effects)
    end
    
    @warn "Posterior reconstruction for Composed with operator :$(m.operator) is not fully implemented. Returning zero effects."
    return (structured=structured_effects, noisy=structured_effects)
end
