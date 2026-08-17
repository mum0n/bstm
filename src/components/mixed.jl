"""
    Mixed <: ComponentModel

An operator component that models random effects (intercepts and/or slopes) for a
specified grouping variable. The correlation structure of the effects is determined
by an inner `ComponentModel`.

# Version
v1.1.2 (2026-08-14)

# Mathematical Summary
The `Mixed` component models effects that vary across the levels of a grouping
variable. It supports both simple (uncorrelated) and correlated random effects.

1.  **Simple Random Effects** (e.g., `random(1 | group)`):
    A single random effect \$\\phi\$ (e.g., an intercept) is modeled for each of the
    \$G\$ levels of the grouping variable. The structure of these effects is
    determined by the inner model. For an `IID` inner model, this is:
    \$\\phi_g \\sim \\mathcal{N}(0, \\sigma^2)\$ for \$g = 1, \\dots, G\$.

2.  **Correlated Random Effects** (e.g., `random(1 + x | group)`):
    A vector of \$K\$ random effects, \$\\boldsymbol{\\beta}_g = [\\beta_{0g}, \\beta_{1g}, \\dots]^T\$,
    is modeled for each group level \$g\$. These effects are assumed to be drawn from
    a multivariate normal distribution with a shared covariance structure:
    \$\\boldsymbol{\\beta}_g \\sim \\mathcal{N}(\\mathbf{0}, \\Sigma)\$
    The covariance matrix \$\\Sigma\$ is decomposed into a set of standard deviations
    \$\\boldsymbol{\\sigma}\$ and a correlation matrix \$\\mathbf{R}\$:
    \$\\Sigma = \\text{diag}(\\boldsymbol{\\sigma}) \\mathbf{R} \\text{diag}(\\boldsymbol{\\sigma})\$
    A prior is placed on the Cholesky factor of \$\\mathbf{R}\$ using the `LKJCholesky`
    distribution. The structure of the effects across group levels (e.g., IID, spatial)
    is determined by the inner `ComponentModel`.

# Computational Methods (for Correlated Effects)
- `:spectral` (default): An efficient, AD-safe method using spectral decomposition of
  the group-level precision matrix.
- `:cholesky`: An AD-safe didactic alternative using dense Cholesky factorization.
- `:cholesky_sparse`: A non-AD-safe didactic method using sparse Cholesky
  factorization, suitable for gradient-free samplers.

# Inputs
- **Required**:
  - A grouping variable (e.g., `group_id`) passed to `random()`.
  - One or more terms for the random effects (e.g., `1` for intercept, `covariate` for slope).
- **Optional (in `random()` call)**:
  - `model`: An inner `ComponentModel` defining the structure across groups (e.g., `iid()`, `ar1()`). Default: `iid()`.
  - `method`: `Symbol`, computational method for correlated effects (`:spectral`, `:cholesky`, `:cholesky_sparse`). Default: `:spectral`.

# Outputs (Parameter Names)
- **Simple Effects**: Same as the inner model (e.g., `sigma_<key>`, `innovations_<key>`).
- **Correlated Effects**:
  - `L_corr_<key>`: The Cholesky factor of the correlation matrix for the effects.
  - `sigma_effects_<key>`: The standard deviations for each random effect term.
  - `innovations_<key>`: The raw standard normal innovations for the coefficients.
"""
struct Mixed <: ComponentModel
    group_var::Symbol
    lhs::Vector{String}
    model::ComponentModel
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:mixed] = Mixed
COMPONENT_CONSTRUCTORS[:mixed] = (p, params) -> begin
    group_var = get(params, :group_var, error("Mixed requires a `group_var`."))
    lhs = get(params, :lhs, ["1"])
    inner_model_obj = get(
        params, :inner_model_obj, error("Mixed requires an `inner_model_obj`.")
    )
    method = get(params, :method, :spectral)
    
    Mixed(group_var, lhs, inner_model_obj, method)
end

MODEL_TO_STRUCTURE_MAP[:mixed] = :mixed

function get_precomputes(m::Mixed, M::NamedTuple, mod_data::Dict)::NamedTuple
    n_cat = get(mod_data[:params], :n_cat, 0)
    if n_cat == 0
        error("Number of categories for Mixed not determined in get_datastructures!.")
    end

    inner_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_inner"),
        :type => :mixed,
        :variables => [m.group_var],
        :params => Dict(:n_cat => n_cat)
    )

    inner_precomputes = get_precomputes(m.model, M, inner_mod_data)
    
    return (
        inner_precomputes = inner_precomputes,
        n_latent = n_cat * length(m.lhs)
    )
end

function get_priors(
    m::Mixed, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_terms = length(m.lhs)
    n_groups = spec.hyper.inner_precomputes.n_latent

    if n_terms == 1
        inner_spec = (
            key = spec.key,
            structure = :mixed,
            var = spec.var,
            component_obj = m.model,
            params = spec.params,
            hyper = spec.hyper.inner_precomputes
        )
        return get_priors(m.model, inner_spec, arch, outcome_idx, M)
    else
        return """
        # Priors for Correlated Mixed Effects: $(spec.key)
        $(p_names.L_corr) ~ LKJCholesky($(n_terms), 1.0)
        $(p_names.sigma_effects) ~ filldist(Exponential(1.0), $(n_terms))
        $(p_names.innovations) ~ DynamicPPL.NamedDist(MvNormal(zeros($(n_groups * n_terms)), I), :$(p_names.innovations))
        """
    end
end

function get_updates(
    m::Mixed, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "mixed_idx_$(m.group_var)"
    n_terms = length(m.lhs)
    n_groups = spec.hyper.inner_precomputes.n_latent

    inner_hyper_access = "spec_registry[:$(spec.key)].hyper.inner_precomputes"

    if n_terms == 1
        lhs_str = m.lhs[1]
        inner_model = m.model
        
        local latent_field_code
        if inner_model isa IID
            latent_field_code = "$(p_names.latent) = $(p_names.innovations) .* $(p_names.sigma)"
        elseif inner_model isa Union{ICAR, Besag, RW1, RW2, Leroux}
            if m.method == :spectral
                latent_field_code = """
                diag_D = $(p_names.sigma) ./ sqrt.($(inner_hyper_access).L .+ M.noise)
                if $(inner_model isa Union{ICAR, Besag, RW1}); diag_D[1] = 0.0; end
                if $(inner_model isa RW2); diag_D[1] = 0.0; diag_D[2] = 0.0; end
                $(p_names.latent) = $(inner_hyper_access).U * (diag_D .* $(p_names.innovations))
                """
            else # :cholesky or :cholesky_sparse
                latent_field_code = """
                F_groups = $(inner_hyper_access).cholesky_factor
                latent_field_raw = F_groups.L' \\ $(p_names.innovations)
                if $(inner_model isa Union{ICAR, Besag, RW1, RW2})
                    Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_groups)), sum(latent_field_raw))
                end
                $(p_names.latent) = latent_field_raw .* $(p_names.sigma)
                """
            end
        else
            latent_field_code = "$(p_names.latent) = $(p_names.innovations) .* $(p_names.sigma)"
        end

        local application_code
        if lhs_str == "1" || lhs_str == "intercept()"
            application_code = "$(eta_target) .+= view($(p_names.latent), M.$(index_var))"
        else
            application_code = """
            let cov_data = M.data[!, :$(Symbol(lhs_str))]
                for i in 1:length($(eta_target))
                    $(eta_target)[i] += cov_data[i] * $(p_names.latent)[M.$(index_var)[i]]
                end
            end
            """
        end

        return """
        # --- Mixed Effect (Single): $(lhs_str) | $(m.group_var) ---
        let
            $(latent_field_code)
            $(application_code)
        end
        """
    else
        application_loop = ""
        for i in 1:n_terms
            term = m.lhs[i]
            if term == "1" || term == "intercept()"
                application_loop *= "for j in 1:length($(eta_target)); $(eta_target)[j] += effects_matrix[M.$(index_var)[j], $(i)]; end\n"
            else
                application_loop *= "let cov_data_$(i) = M.data[!, :$(Symbol(term))]; for j in 1:length($(eta_target)); $(eta_target)[j] += cov_data_$(i)[j] * effects_matrix[M.$(index_var)[j], $(i)]; end; end\n"
            end
        end

        common_correlated_code = """
            L_effects_t = ($(p_names.L_corr).L' * Diagonal($(p_names.sigma_effects)))
            innovations_matrix = reshape($(p_names.innovations), $(n_groups), $(n_terms))
        """

        spectral_code = """
            # --- Correlated Mixed Effects (Spectral): $(spec.key) ---
            let
                $(common_correlated_code)
                inner_hyper = $(inner_hyper_access)
                diag_D = 1.0 ./ sqrt.(inner_hyper.L .+ M.noise)
                if $(m.model isa ICAR || m.model isa Besag); diag_D[1] = 0.0; end
                
                gamma_matrix = inner_hyper.U * (diag_D .* innovations_matrix)
                effects_matrix = gamma_matrix * L_effects_t
                
                $(application_loop)
            end
        """

        cholesky_code = """
            # --- Correlated Mixed Effects (Cholesky): $(spec.key) ---
            let
                $(common_correlated_code)
                F_groups = $(inner_hyper_access).cholesky_factor
                gamma_matrix = F_groups.L' \\ innovations_matrix
                effects_matrix = gamma_matrix * L_effects_t
                
                $(application_loop)
            end
        """

        cholesky_sparse_code = """
            # --- Correlated Mixed Effects (Sparse Cholesky, Not AD-Safe): $(spec.key) ---
            let
                $(common_correlated_code)
                Q_groups = $(inner_hyper_access).Q_template
                F_groups = cholesky(Symmetric(Q_groups + M.noise * I))
                gamma_matrix = F_groups.L' \\ innovations_matrix
                effects_matrix = gamma_matrix * L_effects_t
                
                $(application_loop)
            end
        """

        if m.method == :spectral; return spectral_code;
        elseif m.method == :cholesky; return cholesky_code;
        elseif m.method == :cholesky_sparse; return cholesky_sparse_code;
        else; error("Unsupported method '$(m.method)' for correlated Mixed component."); end
    end
end

function _get_model_symbol(m_obj::ComponentModel)
    for (sym, typ) in COMPONENT_TYPE_REGISTRY
        if m_obj isa typ
            return sym
        end
    end
    return :unknown
end
function get_effects(
    m::Mixed, chain::Chains, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device
    noise = M.noise
    
    n_terms = length(m.lhs)
    n_groups_train = spec.hyper.inner_precomputes.n_latent
    
    # --- Grouping Level and Index Handling ---
    group_var = m.group_var
    train_levels = unique(M.data[!, group_var])
    all_levels = train_levels
    has_new_levels = false

    if !isnothing(PS) && hasproperty(PS.data, group_var)
        pred_levels = unique(PS.data[!, group_var])
        if !isempty(setdiff(pred_levels, train_levels))
            has_new_levels = true
            all_levels = unique(vcat(train_levels, pred_levels))
        end
    end
    n_all_groups = length(all_levels)
    level_map = Dict(level => i for (i, level) in enumerate(all_levels))
    
    # Create the full index vector on the CPU first, then move to device
    full_indices_cpu = if !isnothing(PS) && hasproperty(PS.data, group_var)
        [level_map[v] for v in vcat(M.data[!, group_var], PS.data[!, group_var])]
    else
        [level_map[v] for v in M.data[!, group_var]]
    end
    full_indices_device = to_device(full_indices_cpu)

    # --- Reconstruction Logic ---
    if n_terms == 1
        # --- Case 1: Simple (Uncorrelated) Random Effects ---
        effects_per_outcome = Vector{Matrix{Float64}}()
        for k in 1:outcomes_N
            p_names_k = generate_full_variable_names(spec, M.model_arch, k)
            sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)

            if isempty(sigma_name) || isempty(innovations_name)
                @warn "Parameters for simple Mixed component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(effects_per_outcome, zeros(Float64, n_all_groups, n_samples))
                continue
            end

            # Extract samples (CPU)
            sigma_samples_cpu = get_params_matrix(chain, sigma_name, 1)[:, 1]
            innovations_samples_cpu = get_params_matrix(chain, innovations_name, n_groups_train)
            
            # Move to device for computation
            innovations_device = to_device(innovations_samples_cpu)
            sigma_device = to_device(sigma_samples_cpu)
            
            # Broadcasting on the device
            latent_samples_train_device = innovations_device' .* sigma_device'
            
            full_effects_device = to_device(zeros(Float64, n_all_groups, n_samples))
            train_indices_map_device = to_device([level_map[level] for level in train_levels])
            full_effects_device[train_indices_map_device, :] = latent_samples_train_device

            if has_new_levels
                new_level_indices = setdiff(1:n_all_groups, train_indices_map_device)
                new_effects_device = to_device(randn(Float32, length(new_level_indices), n_samples)) .* sigma_device'
                full_effects_device[new_level_indices, :] = new_effects_device
            end
            
            # Move final result for this outcome back to CPU
            push!(effects_per_outcome, Array(full_effects_device))
        end
        return (type=:simple, effects=effects_per_outcome, lhs=m.lhs[1], indices=full_indices_cpu)
    else
        # --- Case 2: Correlated Random Effects ---
        effects_by_term = Dict{Symbol, Vector{Matrix{Float64}}}()
        for term in m.lhs
            term_key = (term == "1" || term == "intercept()") ? :intercept : Symbol("slope_$(term)")
            effects_by_term[term_key] = [zeros(Float64, n_all_groups, n_samples) for _ in 1:outcomes_N]
        end

        for k in 1:outcomes_N
            p_names_k = generate_full_variable_names(spec, M.model_arch, k)
            l_corr_name = _find_parameter(p_names, string(p_names_k.L_corr), k, is_multivariate_model)
            sigma_effects_name = _find_parameter(p_names, string(p_names_k.sigma_effects), k, is_multivariate_model)
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)

            if isempty(l_corr_name) || isempty(sigma_effects_name) || isempty(innovations_name)
                @warn "Parameters for correlated Mixed component $(spec.key) (outcome $k) not found. Skipping."
                continue
            end

            # Extract samples (CPU)
            l_corr_samples_cpu = get_params_matrix(chain, l_corr_name, n_terms * n_terms)
            sigma_effects_samples_cpu = get_params_matrix(chain, sigma_effects_name, n_terms)
            innovations_samples_cpu = get_params_matrix(chain, innovations_name, n_groups_train * n_terms)
            
            inner_precomputes = spec.hyper.inner_precomputes

            for s in 1:n_samples
                # Move current sample's parameters to device
                l_corr_s_device = to_device(reshape(l_corr_samples_cpu[s,:], n_terms, n_terms))
                sigma_effects_s_device = to_device(sigma_effects_samples_cpu[s,:])
                innov_matrix_s_device = to_device(reshape(innovations_samples_cpu[s,:], n_groups_train, n_terms))
                
                # Perform computations on device
                L_effects_t_device = (l_corr_s_device' * Diagonal(sigma_effects_s_device))
                
                local gamma_matrix_device
                if m.method == :spectral
                    diag_D_device = 1.0 ./ sqrt.(inner_precomputes.L .+ noise)
                    if m.model isa Union{ICAR, Besag, RW1, RW2}; diag_D_device[1] = 0.0; end
                    if m.model isa RW2; diag_D_device[2] = 0.0; end
                    gamma_matrix_device = inner_precomputes.U * (diag_D_device .* innov_matrix_s_device)
                else # :cholesky or :cholesky_sparse
                    F_groups_device = inner_precomputes.cholesky_factor
                    gamma_matrix_device = F_groups_device.L' \ innov_matrix_s_device
                end
                
                effects_matrix_train_device = gamma_matrix_device * L_effects_t_device
                
                # Move results for this sample back to CPU to populate the dictionary
                effects_matrix_train_cpu = Array(effects_matrix_train_device)
                train_indices_map = [level_map[level] for level in train_levels]
                
                for i in 1:n_terms
                    term_key = (m.lhs[i] == "1" || m.lhs[i] == "intercept()") ? :intercept : Symbol("slope_$(m.lhs[i])")
                    effects_by_term[term_key][k][train_indices_map, s] = effects_matrix_train_cpu[:, i]
                end

                if has_new_levels
                    new_level_indices = setdiff(1:n_all_groups, train_indices_map)
                    n_new = length(new_level_indices)
                    new_innovs_device = to_device(randn(Float32, n_new, n_terms))
                    new_gamma_device = new_innovs_device # Simplified assumption for new levels
                    new_effects_device = new_gamma_device * L_effects_t_device
                    new_effects_cpu = Array(new_effects_device)
                    for i in 1:n_terms
                        term_key = (m.lhs[i] == "1" || m.lhs[i] == "intercept()") ? :intercept : Symbol("slope_$(m.lhs[i])")
                        effects_by_term[term_key][k][new_level_indices, s] = new_effects_cpu[:, i]
                    end
                end
            end
        end
        return (type=:correlated, effects=effects_by_term, lhs=m.lhs, indices=full_indices_cpu)
    end
end
