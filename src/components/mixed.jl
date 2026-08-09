
"""
    Mixed <: ComponentModel

An operator component that models random effects (intercepts and/or slopes) for a
specified grouping variable. The correlation structure of the effects is determined
by an inner `ComponentModel`.

# Version
v1.0.0 (2026-08-08)

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
    distribution, which is a standard choice for modeling correlation matrices. The
    structure of the effects across group levels (e.g., IID, spatial) is determined
    by the inner `ComponentModel`.

# Assumptions
- The grouping variable is categorical.
- For correlated effects, the number of terms is small enough for the `LKJCholesky`
  prior to be computationally feasible.

# Best Use Case
Modeling hierarchical or repeated-measures data where observations are nested within
groups. Examples include random intercepts for patients in a clinical trial, random
slopes for students within schools, or spatially varying intercepts for regions.

# Key References
- Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting Linear
  Mixed-Effects Models Using lme4. *Journal of Statistical Software*, 67(1), 1-48.
- Wikipedia: Mixed model
- Wikipedia: Random effects model

# Fields
- `group_var::Symbol`: The symbol of the grouping variable.
- `lhs::Vector{String}`: A vector of strings representing the effects to be randomized
  (e.g., `["1", "cov1"]` for a random intercept and slope).
- `model::ComponentModel`: The inner component model that defines the structure of the
  random effects across the group levels (e.g., `IID`, `AR1`).
"""
struct Mixed <: ComponentModel
    group_var::Symbol
    lhs::Vector{String}
    model::ComponentModel
end

# Add to the central component constructor registry.
# This is called by `resolve_technical_primitive`.
COMPONENT_TYPE_REGISTRY[:mixed] = Mixed
COMPONENT_CONSTRUCTORS[:mixed] = (p, params) -> begin
    group_var = get(params, :group_var, error("Mixed requires a `group_var`."))
    lhs = get(params, :lhs, ["1"])
    inner_model_obj = get(
        params, :inner_model_obj, error("Mixed requires an `inner_model_obj`.")
    )
    
    Mixed(group_var, lhs, inner_model_obj)
end

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[:mixed] = :mixed

"""
    get_datastructures!(m_type::Type{<:Mixed}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Mixed`. It validates the grouping
variable, creates a numeric index for its levels, and stores this information in
the main model configuration `M`.
"""
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
    
    # Create and store the integer index for the grouping variable.
    index_key = Symbol("mixed_idx_$(group_var)")
    M[index_key] = levelcode.(group_data)
    
    # Store the number of categories for use in precomputes.
    params[:n_cat] = n_cat
    
    return true
end

"""
    get_precomputes(m::Mixed, M::NamedTuple, mod_data::Dict)::NamedTuple

Delegates pre-computation to the inner model to get the penalty matrix `Q_template`
for the random effect coefficients.
"""
function get_precomputes(m::Mixed, M::NamedTuple, mod_data::Dict)::NamedTuple
    n_cat = get(mod_data[:params], :n_cat, 0)
    if n_cat == 0
        error("Number of categories for Mixed not determined in get_datastructures!.")
    end

    # The inner model defines the structure across the group levels.
    # We create a temporary module data dictionary to call its get_precomputes method.
    inner_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_inner"),
        :type => :mixed, # The structure is over the mixed effect groups
        :variables => [m.group_var],
        :params => Dict(:n_cat => n_cat) # Pass n_cat for the inner model's dimension
    )

    inner_precomputes = get_precomputes(m.model, M, inner_mod_data)
    
    return (
        inner_precomputes = inner_precomputes,
        n_latent = n_cat * length(m.lhs) # Total number of latent coefficients
    )
end

"""
    get_priors(m::Mixed, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for the `Mixed`, handling both simple and correlated effects.
"""
function get_priors(
    m::Mixed, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_terms = length(m.lhs)
    n_groups = spec.hyper.inner_precomputes.n_latent

    if n_terms == 1
        # Simple random effect (uncorrelated intercept or slope)
        # Delegate to the inner model's prior generator.
        inner_spec = (
            key = spec.key, # Use the same key
            structure = :mixed,
            var = spec.var,
            component_obj = m.model,
            params = spec.params,
            hyper = spec.hyper.inner_precomputes
        )
        return get_priors(m.model, inner_spec, arch, outcome_idx, M)
    else
        # Correlated random effects
        return """
        # Priors for Correlated Mixed Effects: $(spec.key)
        $(p_names.L_corr) ~ LKJCholesky($(n_terms), 1.0)
        $(p_names.sigma_effects) ~ filldist(Exponential(1.0), $(n_terms))
        $(p_names.raw) ~ MvNormal(zeros($(n_groups * n_terms)), I)
        """
    end
end

"""
    get_updates(m::Mixed, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for constructing and applying the mixed effect.
"""
function get_updates(
    m::Mixed, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = Symbol("mixed_idx_$(m.group_var)")
    n_terms = length(m.lhs)
    n_groups = spec.hyper.inner_precomputes.n_latent

    if n_terms == 1
        # Simple random effect
        inner_spec = (
            key = spec.key,
            structure = :mixed,
            var = spec.var,
            component_obj = m.model,
            params = spec.params,
            hyper = spec.hyper.inner_precomputes
        )
        inner_update_frags = get_updates(m.model, inner_spec, arch, outcome_idx, M)
        
        # The inner update will use a generic index. We need to replace it with the correct one.
        update_inner_cleaned = replace(
            inner_update_frags, Regex("M\\.\\w+_idx") => "M.$(index_var)"
        )
        
        lhs_str = m.lhs[1]
        if lhs_str == "1" || lhs_str == "intercept()"
            return update_inner_cleaned # The inner update already handles the intercept
        else
            # It's a random slope. The inner update created the latent field, now we apply it.
            update_without_eta = replace(
                update_inner_cleaned, Regex("$(eta_target) \\.\\+= .*") => ""
            )
            return """
            $(update_without_eta)
            local cov_data = M.data[!, :$(Symbol(lhs_str))]
            for i in 1:length($(eta_target))
                $(eta_target)[i] += cov_data[i] * $(p_names.latent)[M.$(index_var)[i]]
            end
            """
        end
    else
        # Correlated random effects
        application_loop = ""
        for i in 1:n_terms
            term = m.lhs[i]
            if term == "1" || term == "intercept()"
                application_loop *= "for j in 1:length($(eta_target)); $(eta_target)[j] += effects_matrix[M.$(index_var)[j], $(i)]; end\n"
            else
                application_loop *= "local cov_data_$(i) = M.data[!, :$(Symbol(term))]; for j in 1:length($(eta_target)); $(eta_target)[j] += cov_data_$(i)[j] * effects_matrix[M.$(index_var)[j], $(i)]; end\n"
            end
        end

        return """
        # --- Correlated Mixed Effects Update: $(spec.key) ---
        local L_effects_t = ($(p_names.L_corr).L' * Diagonal($(p_names.sigma_effects)))
        
        local inner_precomputes = spec_registry[:$(spec.key)].hyper.inner_precomputes
        local F_groups = cholesky(Symmetric(Matrix(inner_precomputes.Q_template) + M.noise * I))
        local L_groups_cov_inv_t = F_groups.L'
        
        local innovations_matrix = reshape($(p_names.raw), $(n_groups), $(n_terms))
        local gamma_matrix = L_groups_cov_inv_t \\ innovations_matrix
        local effects_matrix = gamma_matrix * L_effects_t
        
        $(application_loop)
        """
    end
end

"""
    get_effects(m::Mixed, chain, M::NamedTuple, ...)

Reconstructs the posterior effects for each level of the grouping variable.
This version returns the raw posterior samples, not summaries, in a structure
that distinguishes between simple and correlated effects.
"""
function get_effects(
    m::Mixed, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    n_terms = length(m.lhs)
    n_groups = spec.hyper.inner_precomputes.n_latent
    noise = M.noise
    
    group_var = m.group_var
    train_levels = unique(M.data[!, group_var])
    all_levels = train_levels
    has_new_levels = false
    if !isnothing(PS) && hasproperty(PS.data, group_var)
        pred_levels = unique(PS.data[!, group_var])
        if !isempty(setdiff(pred_levels, train_levels))
            has_new_levels = true
        end
        all_levels = unique(vcat(train_levels, pred_levels))
    end
    n_all_groups = length(all_levels)
    
    level_map = Dict(level => i for (i, level) in enumerate(all_levels))
    
    full_indices = if !isnothing(PS) && hasproperty(PS.data, group_var)
        [level_map[v] for v in vcat(M.data[!, group_var], PS.data[!, group_var])]
    else
        [level_map[v] for v in M.data[!, group_var]]
    end

    if n_terms == 1
        # --- Simple (uncorrelated) random effects ---
        effects_per_outcome = Vector{Matrix{Float64}}()
        for k in 1:outcomes_N
            p_names = generate_full_variable_names(spec, M.model_arch, k)
            latent_samples = get_params_vector(chain, string(p_names.latent), n_groups)
            push!(effects_per_outcome, latent_samples')
        end
        return (type=:simple, effects=effects_per_outcome, lhs=m.lhs[1], indices=full_indices)
    else
        # --- Correlated random effects ---
        effects_by_term = Dict{Symbol, Vector{Matrix{Float64}}}()
        for term in m.lhs
            term_key = (term == "1" || term == "intercept()") ?
                       :intercept : Symbol("slope_$(term)")
            effects_by_term[term_key] = [zeros(n_groups, n_samples) for _ in 1:outcomes_N]
        end

        for k in 1:outcomes_N
            p_names = generate_full_variable_names(spec, M.model_arch, k)
            l_corr_samples = get_params_vector(
                chain, string(p_names.L_corr), n_terms * n_terms
            )
            sigma_effects_samples = get_params_vector(
                chain, string(p_names.sigma_effects), n_terms
            )
            raw_samples = get_params_vector(
                chain, string(p_names.raw), n_groups * n_terms
            )
            
            F_groups = cholesky(
                Symmetric(Matrix(spec.hyper.inner_precomputes.Q_template) + noise * I)
            )
            L_groups_cov_inv_t = F_groups.L'

            for s in 1:n_samples
                L_effects_t = (
                    reshape(l_corr_samples[s,:], n_terms, n_terms)' *
                    Diagonal(sigma_effects_samples[s,:])
                )
                innov_matrix = reshape(raw_samples[s,:], n_groups, n_terms)
                gamma_matrix = L_groups_cov_inv_t \ innov_matrix
                effects_matrix = gamma_matrix * L_effects_t
                
                for i in 1:n_terms
                    term_key = (m.lhs[i] == "1" || m.lhs[i] == "intercept()") ?
                               :intercept : Symbol("slope_$(m.lhs[i])")
                    effects_by_term[term_key][k][:, s] = effects_matrix[:, i]
                end
            end
        end
        return (type=:correlated, effects=effects_by_term, lhs=m.lhs, indices=full_indices)
    end
end
