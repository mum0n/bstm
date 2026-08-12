"""
    Mixed <: ComponentModel

An operator component that models random effects (intercepts and/or slopes) for a
specified grouping variable. The correlation structure of the effects is determined
by an inner `ComponentModel`.

# Version
v1.1.1 (2026-08-12)

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

function get_datastructures!(m_type::Type{<:Mixed}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    group_var = get(params, :group_var, nothing)

    if isnothing(group_var)
        error("Mixed is missing the grouping variable.")
    end

    if !hasproperty(M[:data], group_var)
        error("Grouping variable ':$group_var' for Mixed not found in data.")
    end

    group_data = M[:data][!, group_var]
    if !(group_data isa CategoricalArray)
        group_data = categorical(group_data)
    end
    
    unique_levels = levels(group_data)
    n_cat = length(unique_levels)
    
    index_key = Symbol("mixed_idx_$(group_var)")
    M[index_key] = levelcode.(group_data)
    
    params[:n_cat] = n_cat
    
    return true
end

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
        # Removed explicit `T` from `zeros` for better AD compatibility.
        return """
        # Priors for Correlated Mixed Effects: $(spec.key)
        $(p_names.L_corr) ~ LKJCholesky($(n_terms), 1.0)
        $(p_names.sigma_effects) ~ filldist(Exponential(1.0), $(n_terms))
        $(p_names.innovations) ~ MvNormal(zeros($(n_groups * n_terms)), I)
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

function get_effects(
    m::Mixed, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    n_terms = length(m.lhs)
    n_groups_train = spec.hyper.inner_precomputes.n_latent
    noise = M.noise
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))
    
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
    
    full_indices = if !isnothing(PS) && hasproperty(PS.data, group_var)
        [level_map[v] for v in vcat(M.data[!, group_var], PS.data[!, group_var])]
    else
        [level_map[v] for v in M.data[!, group_var]]
    end

    if n_terms == 1
        effects_per_outcome = Vector{Matrix{Float64}}()
        for k in 1:outcomes_N
            sigma_name = _find_parameter(p_names_vec, string(spec.key), "sigma", k, is_multivariate_model)
            innovations_name = _find_parameter(p_names_vec, string(spec.key), "innovations", k, is_multivariate_model)

            if isempty(sigma_name) || isempty(innovations_name)
                @warn "Parameters for simple Mixed component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(effects_per_outcome, zeros(Float64, n_all_groups, n_samples))
                continue
            end

            sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
            innovations_samples = get_params_vector(chain, innovations_name, n_groups_train)
            
            latent_samples_train = innovations_samples' .* sigma_samples'
            
            full_effects = zeros(Float64, n_all_groups, n_samples)
            train_indices_map = [level_map[level] for level in train_levels]
            full_effects[train_indices_map, :] = latent_samples_train

            if has_new_levels
                new_level_indices = setdiff(1:n_all_groups, train_indices_map)
                new_effects = randn(length(new_level_indices), n_samples) .* sigma_samples'
                full_effects[new_level_indices, :] = new_effects
            end
            push!(effects_per_outcome, full_effects)
        end
        return (type=:simple, effects=effects_per_outcome, lhs=m.lhs[1], indices=full_indices)
    else
        effects_by_term = Dict{Symbol, Vector{Matrix{Float64}}}()
        for term in m.lhs
            term_key = (term == "1" || term == "intercept()") ? :intercept : Symbol("slope_$(term)")
            effects_by_term[term_key] = [zeros(n_all_groups, n_samples) for _ in 1:outcomes_N]
        end

        for k in 1:outcomes_N
            l_corr_samples_name = _find_parameter(p_names_vec, string(spec.key), "L_corr", k, is_multivariate_model)
            sigma_effects_samples_name = _find_parameter(p_names_vec, string(spec.key), "sigma_effects", k, is_multivariate_model)
            innovations_samples_name = _find_parameter(p_names_vec, string(spec.key), "innovations", k, is_multivariate_model)

            if isempty(l_corr_samples_name) || isempty(sigma_effects_samples_name) || isempty(innovations_samples_name)
                @warn "Parameters for correlated Mixed component $(spec.key) (outcome $k) not found. Skipping."
                continue
            end

            l_corr_samples = get_params_vector(chain, l_corr_samples_name, n_terms * n_terms)
            sigma_effects_samples = get_params_vector(chain, sigma_effects_samples_name, n_terms)
            innovations_samples = get_params_vector(chain, innovations_samples_name, n_groups_train * n_terms)
            
            inner_precomputes = spec.hyper.inner_precomputes

            for s in 1:n_samples
                L_effects_t = (reshape(l_corr_samples[s,:], n_terms, n_terms)' * Diagonal(sigma_effects_samples[s,:]))
                innov_matrix = reshape(innovations_samples[s,:], n_groups_train, n_terms)
                
                local gamma_matrix
                if m.method == :spectral
                    diag_D = 1.0 ./ sqrt.(inner_precomputes.L .+ noise)
                    if m.model isa ICAR || m.model isa Besag; diag_D[1] = 0.0; end
                    gamma_matrix = inner_precomputes.U * (diag_D .* innov_matrix)
                else # :cholesky or :cholesky_sparse
                    F_groups = inner_precomputes.cholesky_factor
                    gamma_matrix = F_groups.L' \ innov_matrix
                end
                
                effects_matrix_train = gamma_matrix * L_effects_t
                train_indices_map = [level_map[level] for level in train_levels]
                
                for i in 1:n_terms
                    term_key = (m.lhs[i] == "1" || m.lhs[i] == "intercept()") ? :intercept : Symbol("slope_$(m.lhs[i])")
                    effects_by_term[term_key][k][train_indices_map, s] = effects_matrix_train[:, i]
                end

                if has_new_levels
                    new_level_indices = setdiff(1:n_all_groups, train_indices_map)
                    n_new = length(new_level_indices)
                    new_innovs = randn(n_new, n_terms)
                    new_gamma = new_innovs # Simplified assumption
                    new_effects = new_gamma * L_effects_t
                    for i in 1:n_terms
                        term_key = (m.lhs[i] == "1" || m.lhs[i] == "intercept()") ? :intercept : Symbol("slope_$(m.lhs[i])")
                        effects_by_term[term_key][k][new_level_indices, s] = new_effects[:, i]
                    end
                end
            end
        end
        return (type=:correlated, effects=effects_by_term, lhs=m.lhs, indices=full_indices)
    end
end
