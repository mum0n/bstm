"""
    Composed <: ComponentModel

An orchestrator component that combines multiple child components using a specified
operator. This component is the backbone for creating complex interactions, such as
spatiotemporal models, spatially varying coefficients, and non-stationary variance
models.

# Version
v1.1.3 (2026-08-15)

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
            structure = MODEL_TO_STRUCTURE_MAP[child_node.module_type],
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
        coeffs_prior = "$(p_names.innovations) ~ MvNormal(zeros(T, " *
                       "$(n_spatial * n_basis)), I)"
        
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
        modifier_priors = get_priors(
            child_specs[1].component_obj, child_specs[1], arch, outcome_idx, M
        )
        base_priors = get_priors(
            child_specs[2].component_obj, child_specs[2], arch, outcome_idx, M
        )

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
        rho_value = hasproperty(state_spec.component_obj, :rho) ? 
                    string(state_p_names.rho) : "nothing"
        
        basis_matrix = "spec_registry[:$(dynamic_spec.key)].hyper.basis_matrix"
        
        n_spatial = state_spec.hyper.n_latent
        n_basis = dynamic_spec.hyper.n_latent

        cholesky_base_code = """
            Q_spatial_template = spec_registry[:$(state_spec.key)].hyper.Q_template
            Q_spatial = recompose_precision(
                :$(state_model_type), Q_spatial_template, 1.0; extra_param=$(rho_value)
            )
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
        
        s_p_names = generate_full_variable_names(s_spec, arch, outcome_idx)
        t_p_names = generate_full_variable_names(t_spec, arch, outcome_idx)

        s_rho_val = hasproperty(s_spec.component_obj, :rho) ? 
                    string(s_p_names.rho) : "nothing"
        t_rho_val = hasproperty(t_spec.component_obj, :rho) ? 
                    string(t_p_names.rho) : "nothing"

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
                
                st_idx = (M.t_idx .- 1) .* M.s_N .+ M.s_idx
                $(eta_target) .+= view(st_field, st_idx)
            end
        """

        cholesky_base_code = """
            Q_s = recompose_precision(:$(s_model_type), 
                spec_registry[:$(s_spec.key)].hyper.Q_template, 1.0; extra_param=$(s_rho_val))
            Q_t = recompose_precision(:$(t_model_type), 
                spec_registry[:$(t_spec.key)].hyper.Q_template, 1.0; extra_param=$(t_rho_val))
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
                st_idx = (M.t_idx .- 1) .* M.s_N .+ M.s_idx
                $(eta_target) .+= view(st_field, st_idx)
            end
        """

        if m.method == :spectral; return spectral_code;
        elseif m.method == :cholesky; return cholesky_dense_code;
        else; error("Unsupported method '$(m.method)' for Kronecker product operator."); end

    elseif m.operator == :composition # Non-stationary variance
        modifier_spec = child_specs[1]
        base_spec = child_specs[2]

        modifier_updates = get_updates(
            modifier_spec.component_obj, modifier_spec, arch, outcome_idx, M
        )
        modifier_latent_var = generate_full_variable_names(
            modifier_spec, arch, outcome_idx
        ).latent
        modifier_code = replace(modifier_updates, Regex("$(eta_target) .\\+= .*") => "")

        base_updates = get_updates(
            base_spec.component_obj, base_spec, arch, outcome_idx, M
        )
        base_latent_var = generate_full_variable_names(
            base_spec, arch, outcome_idx
        ).latent
        base_code = replace(base_updates, Regex("$(eta_target) .\\+= .*") => "")

        base_structure = base_spec.structure
        indexed_base_effect = if base_structure == :spatial
            "view($(base_latent_var), M.s_idx)"
        elseif base_structure == :temporal
            "view($(base_latent_var), M.t_idx)"
        elseif base_structure == :seasonal
            "view($(base_latent_var), M.u_idx)"
        else # :smooth or :any
            "$(base_latent_var)"
        end

        return """
        # --- Composition (NonStationaryVariance) Update: $(spec.key) ---
        let
            # Compute the modulating field (e.g., log_sigma(x))
            $(modifier_code)
            
            # Compute the base field (e.g., spatial effect z(s))
            $(base_code)

            # Modulate the base field and add to eta
            final_effect = $(indexed_base_effect) .* exp.($(modifier_latent_var))
            $(eta_target) .+= final_effect
        end
        """
    end
    
    return "# Operator '$(m.operator)' not implemented for get_updates in Composed"
end

function get_effects(
    m::Composed, chain::Chains, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device
    noise = M.noise
    child_specs = spec.hyper.child_specs

    # --- Operator-Specific Reconstruction ---
    if m.operator == :pipe # Spatially varying curve
        structured_effects = Vector{Matrix{Float64}}()
        dynamic_spec = child_specs[1]
        state_spec = child_specs[2]
        
        # Handle basis matrix for training and prediction sets
        B_dynamic_train = dynamic_spec.hyper.basis_matrix
        B_dynamic_full = if !isnothing(PS)
            coord_vars = get(dynamic_spec.params, :positional_args, [])
            if !isempty(coord_vars) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
                coords_pred_cpu = Matrix{Float64}(PS.data[!, Symbol.(coord_vars)])
                coords_pred_device = to_device(coords_pred_cpu)
                dynamic_comp_obj = dynamic_spec.component_obj
                B_pred, _ = bstm_bspline_basis(
                    coords_pred_device[:, 1], dynamic_comp_obj.nbins, dynamic_comp_obj.degree
                )
                vcat(B_dynamic_train, B_pred)
            else
                B_dynamic_train
            end
        else
            B_dynamic_train
        end
        N_total = size(B_dynamic_full, 1)
        
        n_spatial = state_spec.hyper.n_latent
        n_basis = dynamic_spec.hyper.n_latent
        s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, to_device(get(PS.data, :s_idx, [])))

        for k in 1:outcomes_N
            p_names_k = generate_full_variable_names(spec, M.model_arch, k)
            innov_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            if isempty(innov_name)
                @warn "Innovations for Composed component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            innov_samples_cpu = get_params_matrix(chain, innov_name, n_spatial * n_basis)
            
            state_p_names = generate_full_variable_names(state_spec, M.model_arch, k)
            state_model_type = Symbol(lowercase(string(typeof(state_spec.component_obj))))
            Q_spatial_template = state_spec.hyper.Q_template
            
            effect_k_device = to_device(zeros(Float64, N_total, n_samples))

            for i in 1:n_samples
                rho_val = hasproperty(state_spec.component_obj, :rho) ?
                          get_params_vector(chain, string(state_p_names.rho), 1)[i, 1] :
                          nothing
                Q_spatial = recompose_precision(state_model_type, Q_spatial_template, 1.0; extra_param=rho_val)
                F_spatial = cholesky(Symmetric(Q_spatial + M.noise * I))
                
                coeffs_raw_matrix_device = to_device(reshape(innov_samples_cpu[i, :], n_spatial, n_basis))
                spatial_coeffs = F_spatial.L' \ coeffs_raw_matrix_device
                
                effect_k_device[:, i] = sum(B_dynamic_full .* spatial_coeffs[s_idx_full, :], dims=2)
            end
            push!(structured_effects, Array(effect_k_device))
        end
        return (structured=structured_effects, noisy=structured_effects)

    elseif m.operator == :kronecker_product
        structured_effects = Vector{Matrix{Float64}}()
        s_spec = child_specs[1]
        t_spec = child_specs[2]
        s_N = s_spec.hyper.n_latent
        t_N = t_spec.hyper.n_latent
        
        s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, to_device(get(PS.data, :s_idx, [])))
        t_idx_full = isnothing(PS) ? M.t_idx : vcat(M.t_idx, to_device(get(PS.data, :t_idx, [])))
        st_idx_full = (t_idx_full .- 1) .* s_N .+ s_idx_full
        N_total = length(st_idx_full)

        for k in 1:outcomes_N
            p_names_k = generate_full_variable_names(spec, M.model_arch, k)
            sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
            innov_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            if isempty(sigma_name) || isempty(innov_name)
                @warn "Parameters for Kronecker product component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
            innov_samples_cpu = get_params_matrix(chain, innov_name, s_N * t_N)

            effect_k_device = to_device(zeros(Float64, N_total, n_samples))

            for i in 1:n_samples
                local st_field_device
                if m.method == :spectral
                    s_hyper = s_spec.hyper; t_hyper = t_spec.hyper
                    s_p_names = generate_full_variable_names(s_spec, M.model_arch, k)
                    t_p_names = generate_full_variable_names(t_spec, M.model_arch, k)
                    s_rho_val = hasproperty(s_spec.component_obj, :rho) ? get_params_vector(chain, string(s_p_names.rho), 1)[i, 1] : nothing
                    t_rho_val = hasproperty(t_spec.component_obj, :rho) ? get_params_vector(chain, string(t_p_names.rho), 1)[i, 1] : nothing
                    
                    diag_Ls = (1.0 .- s_rho_val) .+ s_rho_val .* s_hyper.L
                    diag_Lt = (1.0 .- t_rho_val) .+ t_rho_val .* t_hyper.L
                    diag_D_s = sigma_samples_cpu[i] ./ sqrt.(diag_Ls .+ noise)
                    diag_D_t = 1.0 ./ sqrt.(diag_Lt .+ noise)
                    
                    Z_matrix_device = to_device(reshape(innov_samples_cpu[i, :], s_N, t_N))
                    tmp = s_hyper.U' * Z_matrix_device * t_hyper.U
                    transformed = (diag_D_s .* tmp) .* diag_D_t'
                    st_field_device = s_hyper.U * transformed * t_hyper.U'
                else # cholesky
                    s_model_type = Symbol(lowercase(string(typeof(s_spec.component_obj))))
                    t_model_type = Symbol(lowercase(string(typeof(t_spec.component_obj))))
                    s_p_names = generate_full_variable_names(s_spec, M.model_arch, k)
                    t_p_names = generate_full_variable_names(t_spec, M.model_arch, k)
                    s_rho_val = hasproperty(s_spec.component_obj, :rho) ? get_params_vector(chain, string(s_p_names.rho), 1)[i, 1] : nothing
                    t_rho_val = hasproperty(t_spec.component_obj, :rho) ? get_params_vector(chain, string(t_p_names.rho), 1)[i, 1] : nothing
                    
                    Q_s = recompose_precision(s_model_type, s_spec.hyper.Q_template, 1.0; extra_param=s_rho_val)
                    Q_t = recompose_precision(t_model_type, t_spec.hyper.Q_template, 1.0; extra_param=t_rho_val)
                    
                    C_s = cholesky(Symmetric(Q_s + noise * I))
                    C_t = cholesky(Symmetric(Q_t + noise * I))
                    
                    Z_matrix_device = to_device(reshape(innov_samples_cpu[i, :], s_N, t_N))
                    tmp_spatial = C_s.L' \ Z_matrix_device
                    st_field_unscaled = transpose(C_t.L' \ transpose(tmp_spatial))
                    st_field_device = st_field_unscaled .* sigma_samples_cpu[i]
                end
                effect_k_device[:, i] = view(st_field_device, st_idx_full)
            end
            push!(structured_effects, Array(effect_k_device))
        end
        return (structured=structured_effects, noisy=structured_effects)

    elseif m.operator == :composition # Non-stationary variance
        structured_effects = Vector{Matrix{Float64}}()
        modifier_spec = child_specs[1]
        base_spec = child_specs[2]
        
        # Recursive calls with the correct, modern signature
        modifier_effects_all = get_effects(modifier_spec.component_obj, chain, modifier_spec, M, PS)
        base_effects_all = get_effects(base_spec.component_obj, chain, base_spec, M, PS)

        for k in 1:outcomes_N
            log_sigma_field = modifier_effects_all.structured[k]
            base_field = base_effects_all.structured[k]
            
            reconstructed_effects = base_field .* exp.(log_sigma_field)
            push!(structured_effects, reconstructed_effects)
        end
        return (structured=structured_effects, noisy=structured_effects)
    end
    
    N_total = isnothing(PS) ? M.y_N : M.y_N + size(PS.data, 1)
    structured_effects = [zeros(Float64, N_total, n_samples) for _ in 1:outcomes_N]
    @warn "Posterior reconstruction for Composed with operator :$(m.operator) is not fully implemented. Returning zero effects."
    return (structured=structured_effects, noisy=structured_effects)
end
