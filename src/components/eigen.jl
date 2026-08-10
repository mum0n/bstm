"""
    Eigen <: ComponentModel

A component model for Bayesian Principal Component Analysis (PCA) or Factor Analysis.
It decomposes a set of multivariate outcomes into a smaller set of orthogonal latent
factors, and uses the dominant latent factor as a predictor in the main model.

# Version
v1.0.2 (2026-08-10)

# Mathematical Summary
The `Eigen` component models a set of \$N_{vars}\$ observed variables \$Y \\in \\mathbb{R}^{N_{obs} \\times N_{vars}}\$
as a linear combination of \$N_{factors}\$ latent factors \$F \\in \\mathbb{R}^{N_{obs} \\times N_{factors}}\$
and a loadings matrix \$L \\in \\mathbb{R}^{N_{vars} \\times N_{factors}}\$, plus idiosyncratic noise \$\\Psi\$:
\$Y = F L^T + \\Psi\$
where \$F_{i,j} \\sim \\mathcal{N}(0, \\sigma_{pca,j}^2)\$ and \$\\Psi_{i,j} \\sim \\mathcal{N}(0, \\sigma_{pdef,j}^2)\$.

To ensure the loadings matrix \$L\$ is identifiable and orthonormal, it is constructed
using a sequence of Householder reflections. This parameterization ensures \$L^T L = I\$.
The latent factors \$F\$ are typically sampled from standard normal distributions.

The effect added to the linear predictor `eta` of the main model is the first
(dominant) latent factor \$F_{:,1}\$.

# Assumptions
- The input variables are centered (mean-subtracted). This is handled automatically
  by `get_datastructures!`.
- The number of factors `n_factors` is less than the number of variables `n_vars`.
- The Householder transformation provides a stable and differentiable way to
  parameterize orthonormal matrices.

# Best Use Case
Dimensionality reduction for multivariate data, identifying the dominant shared
signal across multiple correlated variables, and using this signal as a predictor
in a larger model. Useful in ecology for species assemblages, in finance for
portfolio analysis, or in neuroscience for brain activity patterns.

# Key References
- **Bayesian PCA**: Tipping, M. E., & Bishop, C. M. (1999). *Probabilistic Principal Component Analysis*. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 61(3), 611-622.
- **Householder Transformation**: Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*. Johns Hopkins University Press.
- **Wikipedia**: Householder transformation

# Fields
- `n_vars::Int`: The number of variables (outcomes) to be decomposed.
- `n_factors::Int`: The number of latent factors to extract.
- `pca_sd::Distribution`: The prior for the standard deviations of the principal components (factor scores).
- `pdef_sd::Distribution`: The prior for the standard deviation of the uniqueness/residual noise for each variable.
- `ltri_indices::Vector{Int}`: Pre-calculated indices for the lower-triangular part of the Householder matrix.
- `method::Symbol`: The parameterization method for latent factors. Can be `:noncentered` (default, recommended) or `:centered` (didactic alternative).
"""
struct Eigen <: ComponentModel
    n_vars::Int
    n_factors::Int
    pca_sd::Distribution
    pdef_sd::Distribution
    ltri_indices::Vector{Int}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:eigen] = Eigen

# Add to the central component constructor registry.
# The parameters are populated by `get_datastructures!`.
COMPONENT_CONSTRUCTORS[:eigen] = (p, params) -> Eigen(
    get(params, :n_vars, 0),
    get(params, :n_factors, 1),
    p.pca_sd,
    p.pdef_sd,
    get(params, :ltri_indices, Int[]),
    get(params, :method, :noncentered)
)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[:eigen] = :any

"""
    get_datastructures!(m_type::Type{<:Eigen}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Eigen` component. It extracts the multivariate
data to be decomposed, centers it, validates dimensions, and pre-calculates indices
for the Householder transformation used to ensure orthonormal loadings.

# Assumptions
- The `eigen()` call provides one or more variables representing the multivariate data.
- The data should be continuous.
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
    # This is used to map the flat `v_raw` vector to the `v_mat` for Householder.
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
        n_latent=n_obs # The main effect added to eta is per-observation
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

    priors = String[]
    push!(priors, "$(p_names.v_raw) ~ NamedDist(MvNormal(zeros(T, $(length(m.ltri_indices))), 1.0), :$(p_names.v_raw))")
    push!(priors, "$(p_names.pca_sd) ~ NamedDist(filldist($(pca_sd_prior_str), $(m.n_factors)), :$(p_names.pca_sd))")
    push!(priors, "$(p_names.pdef_sd) ~ NamedDist(filldist($(pdef_sd_prior_str), $(m.n_vars)), :$(p_names.pdef_sd))")
    
    if m.method == :noncentered
        push!(priors, "$(p_names.factors_flat) ~ NamedDist(MvNormal(zeros(T, $(n_obs * m.n_factors)), I), :$(p_names.factors_flat))")
    end

    return join(priors, "\n    ")
end


"""
    get_updates(m::Eigen, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code for the `Eigen` component. This includes the likelihood
for the factor analysis and the construction of the dominant latent factor to be
added to the linear predictor `eta`.
"""
function get_updates(m::Eigen, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    n_obs = size(spec.precomputes.eigen_data, 1)
    n_factors = m.n_factors

    common_code = """
        local v_mat = zeros(T, $(m.n_vars), $(n_factors))
        v_mat[$(m.ltri_indices)] .= $(p_names.v_raw)
        
        local U = householder_to_eigenvector(v_mat, $(m.n_vars), $(n_factors))
        local L = U * Diagonal($(p_names.pca_sd))
        local Psi = Diagonal($(p_names.pdef_sd).^2) + (T(M.noise) * I)
        
        local Y_eigen_data = spec_registry[:$(spec.key)].precomputes.eigen_data
    """

    noncentered_code = """
        # --- Factor Model for Eigen Component (Non-Centered): $(spec.key) ---
        let
            $(common_code)
            local F = reshape($(p_names.factors_flat), $(n_obs), $(n_factors))
            local Y_hat = F * L'
            
            for i in 1:$(n_obs)
                Turing.@addlogprob! logpdf(MvNormal(Y_hat[i, :], Psi), Y_eigen_data[i, :])
            end
            
            $(eta_target) .+= view(F, :, 1)
        end
    """

    centered_code = """
        # --- Factor Model for Eigen Component (Centered): $(spec.key) ---
        # This is a didactic alternative. It can be less efficient for MCMC sampling.
        let
            $(common_code)
            local F = zeros(T, $(n_obs), $(n_factors))
            local Cov_F_row = Symmetric(L * L') # Covariance for each row of F
            
            for i in 1:$(n_obs)
                F[i, :] ~ MvNormal(zeros(T, $(n_factors)), Cov_F_row)
            end
            
            local Y_hat = F * L'
            
            for i in 1:$(n_obs)
                Turing.@addlogprob! logpdf(MvNormal(Y_hat[i, :], Psi), Y_eigen_data[i, :])
            end
            
            $(eta_target) .+= view(F, :, 1)
        end
    """

    if m.method == :noncentered
        return noncentered_code
    elseif m.method == :centered
        return centered_code
    else
        error("Unsupported method '$(m.method)' for Eigen component.")
    end
end

"""
    get_effects(m::Eigen, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `Eigen` component's effect (the dominant latent factor) from the MCMC chain.
"""
function get_effects(m::Eigen, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    n_obs_train = size(spec.precomputes.eigen_data, 1)
    n_factors = m.n_factors

    # Handle prediction set (PS) if provided.
    # For out-of-sample prediction, the latent factors are unknown.
    # A common approach is to assume they are zero or sample from their prior.
    # Here, we pad with zeros for prediction points.
    n_obs_full = N_total # Total observations (train + pred)
    
    # Reconstruct the first latent factor for each sample.
    first_factor_samples = zeros(Float64, n_obs_full, n_samples)
    
    for j in 1:n_samples
        local F_matrix_j
        if m.method == :noncentered
            factors_samples = get_params_vector(chain, string(p_names.factors_flat), n_obs_train * n_factors)
            F_matrix_j = reshape(factors_samples[j, :], n_obs_train, n_factors)
        else # :centered
            # For centered, factors are sampled directly as F[i,:]
            # We need to extract the full F matrix.
            # This assumes Turing stores F as a single flattened vector or a matrix.
            # If it's stored as F[i,j] or F[i,:], we need to adapt.
            # Assuming it's stored as F[i,j] and we can reconstruct the matrix.
            # This is a simplification; actual extraction might need to loop over indices.
            F_matrix_j_flat = get_params_vector(chain, string(p_names.latent), n_obs_train * n_factors)
            F_matrix_j = reshape(F_matrix_j_flat[j, :], n_obs_train, n_factors)
        end
        
        # The effect is the first latent factor (first column of F).
        first_factor_samples[1:n_obs_train, j] = view(F_matrix_j, :, 1)
        
        # Prediction points (if any) are assumed to have zero effect for factors.
        # This is a simplification; a more complex model might sample them or impute.
        if n_obs_full > n_obs_train
            first_factor_samples[(n_obs_train+1):n_obs_full, j] .= 0.0
        end
    end

    # The Eigen effect is univariate (the dominant factor); it applies the same effect to all outcomes.
    # We replicate the single reconstructed effect for each outcome.
    for k in 1:outcomes_N
        push!(structured_effects, first_factor_samples)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
