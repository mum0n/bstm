# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    Eigen <: ComponentModel

A component model for Bayesian Principal Component Analysis (PCA) or Factor Analysis.
It decomposes a set of multivariate outcomes into a smaller set of orthogonal latent
factors, and uses the sum of these factors as a predictor in the main model.

# Fields
- `n_vars::Int`: The number of variables (outcomes) to be decomposed.
- `n_factors::Int`: The number of latent factors to extract.
- `pca_sd::Distribution`: The prior for the standard deviations of the principal components (factor scores).
- `pdef_sd::Distribution`: The prior for the standard deviation of the uniqueness/residual noise for each variable.
- `ltri_indices::Vector{Int}`: Pre-calculated indices for the lower-triangular part of the Householder matrix.
"""
struct Eigen <: ComponentModel
    n_vars::Int
    n_factors::Int
    pca_sd::Distribution
    pdef_sd::Distribution
    ltri_indices::Vector{Int}
end

# Add to the central component constructor registry.
# The parameters are populated by `get_datastructures!`.
COMPONENT_CONSTRUCTORS[:eigen] = (p, params) -> Eigen(
    get(params, :n_vars, 0),
    get(params, :n_factors, 1),
    p.pca_sd,
    p.pdef_sd,
    get(params, :ltri_indices, Int[])
)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[Eigen] = :any

"""
    get_datastructures!(m_type::Type{<:Eigen}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Eigen` component. It extracts the multivariate
data to be decomposed, validates dimensions, and pre-calculates indices for the
Householder transformation used to ensure orthonormal loadings.
"""
function get_datastructures!(m_type::Type{<:Eigen}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    variables = mod_data[:variables]
    
    if isempty(variables)
        error("The `eigen()` module was called without any variables specified.")
    end

    vars_sym = Symbol.(variables)
    if !all(hasproperty(M[:data], v) for v in vars_sym)
        missing_vars = filter(v -> !hasproperty(M[:data], v), vars_sym)
        error("Eigen module variables not found in data: $(missing_vars)")
    end
    
    # Extract the data and center it (a standard assumption for PCA).
    eigen_data_matrix = Matrix(M[:data][!, vars_sym])
    eigen_data_matrix .-= mean(eigen_data_matrix, dims=1)
    
    # Store the data matrix in the module's parameters for the builder to access.
    params[:eigen_data] = eigen_data_matrix
    
    n_vars = length(vars_sym)
    n_factors = get(params, :n_factors, 1)
    if n_factors >= n_vars
        @warn "Number of factors ($n_factors) for eigen() module should be less than the number of variables ($n_vars). Setting to $(n_vars - 1)."
        n_factors = n_vars - 1
    end
    
    # Pre-calculate indices for the lower-triangular part of the Householder matrix.
    ltri_mask = [r >= c for r in 1:n_vars, c in 1:n_factors]
    ltri_indices = findall(vec(ltri_mask))
    
    params[:ltri_indices] = ltri_indices
    params[:n_factors] = n_factors
    params[:n_vars] = n_vars
    
    return true # Proceed with component creation.
end

"""
    get_precomputes(m::Eigen, M::NamedTuple, mod_data::Dict)::NamedTuple

For the `Eigen` component, this function stores the centered data matrix and
dimensional information required by the code generators.
"""
function get_precomputes(m::Eigen, M::NamedTuple, mod_data::Dict)::NamedTuple
    eigen_data = get(mod_data[:params], :eigen_data, nothing)
    if isnothing(eigen_data)
        error("Eigen data matrix not found in module metadata. This indicates an issue in `get_datastructures!`.")
    end

    n_obs = size(eigen_data, 1)

    return (
        eigen_data=eigen_data,
        n_latent=n_obs # The main effect is per-observation
    )
end

"""
    get_priors(m::Eigen, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code for the priors on the `Eigen` component's parameters.
"""
function get_priors(m::Eigen, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    pca_sd_prior_str = _distribution_to_string(m.pca_sd)
    pdef_sd_prior_str = _distribution_to_string(m.pdef_sd)
    
    n_obs = size(spec.precomputes.eigen_data, 1)

    return """
        # Priors for Eigen component: $(spec.key)
        $(p_names.v_raw) ~ NamedDist(MvNormal(zeros(T, $(length(m.ltri_indices))), T(1.0)), :$(p_names.v_raw))
        $(p_names.pca_sd) ~ NamedDist(filldist($(pca_sd_prior_str), $(m.n_factors)), :$(p_names.pca_sd))
        $(p_names.pdef_sd) ~ NamedDist(filldist($(pdef_sd_prior_str), $(m.n_vars)), :$(p_names.pdef_sd))
        $(p_names.factors_flat) ~ NamedDist(MvNormal(zeros(T, $(n_obs * m.n_factors)), I), :$(p_names.factors_flat))
    """
end

"""
    get_updates(m::Eigen, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code for the `Eigen` component. This includes the likelihood
for the factor analysis and the construction of the effect to be added to `eta`.
"""
function get_updates(m::Eigen, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    n_obs = size(spec.precomputes.eigen_data, 1)

    return """
        # --- Factor Model for Eigen Component: $(spec.key) ---
        local v_mat = zeros(T, $(m.n_vars), $(m.n_factors))
        v_mat[$(m.ltri_indices)] .= $(p_names.v_raw)
        
        local U = householder_to_eigenvector(v_mat, $(m.n_vars), $(m.n_factors))
        local L = U * Diagonal($(p_names.pca_sd))
        local F = reshape($(p_names.factors_flat), $(n_obs), $(m.n_factors))
        local Y_hat = F * L'
        local Psi = Diagonal($(p_names.pdef_sd).^2) + (T(M.noise) * I)
        
        local Y_eigen_data = spec_registry[:$(spec.key)].precomputes.eigen_data
        for i in 1:$(n_obs)
            # This component has its own likelihood for the factor analysis part.
            Turing.@addlogprob! logpdf(MvNormal(Y_hat[i, :], Psi), Y_eigen_data[i, :])
        end
        
        # The effect added to the main model is the sum of the factor scores.
        $(eta_target) .+= sum(F, dims=2)
    """
end

"""
    get_effects(m::Eigen, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `Eigen` component's effect (sum of factor scores) from the MCMC chain.
"""
function get_effects(m::Eigen, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    factors_samples = get(chain, p_names.factors_flat)
    
    n_obs_train = size(spec.precomputes.eigen_data, 1)
    n_factors = m.n_factors

    # The effect is the sum of factor scores for each observation.
    total_effect = zeros(n_samples, n_obs_train)
    for j in 1:n_samples
        F_matrix_j = reshape(factors_samples[j, :], n_obs_train, n_factors)
        total_effect[j, :] = sum(F_matrix_j, dims=2)
    end

    # Handle prediction set (PS) if provided.
    # For out-of-sample prediction, the latent factors are unknown.
    # A common approach is to use the mean effect (which is zero for standard normal factors).
    # Here, we will pad with zeros, assuming the effect is centered.
    if !isnothing(PS)
        n_pred = PS.y_N
        pred_effect = zeros(n_samples, n_pred)
        total_effect = hcat(total_effect, pred_effect)
    end

    mean_effect = mean(total_effect, dims=1)[:]
    lower_ci = quantile(total_effect, 0.025, dims=1)[:]
    upper_ci = quantile(total_effect, 0.975, dims=1)[:]

    return (structured=(mean=mean_effect, lower=lower_ci, upper=upper_ci),)
end 
