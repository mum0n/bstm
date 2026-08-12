"""
    Eigen <: ComponentModel

A component model for Bayesian Principal Component Analysis (PCA) or Factor Analysis.
It decomposes a set of multivariate outcomes into a smaller set of orthogonal latent
factors, and uses the dominant latent factor as a predictor in the main model.

# Version
v1.1.0 (2026-08-11)

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

# Computational Methods
- `:noncentered` (Default, AD-friendly): A non-centered parameterization where the
  latent factors are constructed from standard normal innovations. Recommended for NUTS.
- `:centered` (Didactic, Not AD-friendly): A centered parameterization where the
  latent factors are sampled directly from their scaled distribution. This can be
  less efficient for MCMC and is retained for didactic purposes.

# Inputs
- **Required**:
  - One or more variables passed to `eigen()` representing the multivariate data to be decomposed.
- **Optional (in `eigen()` call)**:
  - `n_factors`: `Int`, the number of latent factors to extract. Default: `1`.
  - `pca_sd`: `UnivariateDistribution`, prior for the standard deviations of the factors. Default: `Exponential(1.0)`.
  - `pdef_sd`: `UnivariateDistribution`, prior for the standard deviations of the uniquenesses. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:noncentered` or `:centered`). Default: `:noncentered`.

# Outputs (Parameter Names)
- `v_raw_<key>`: Raw vectors for the Householder transformation.
- `pca_sd_<key>`: Standard deviations of the principal components (factors).
- `pdef_sd_<key>`: Standard deviations of the uniquenesses (residuals).
- `factors_flat_<key>`: Raw standard normal innovations for the latent factors (for `:noncentered`).
- `latent_<key>`: The latent factors (for `:centered`).
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

COMPONENT_CONSTRUCTORS[:eigen] = (p, params) -> Eigen(
    get(params, :n_vars, 0),
    get(params, :n_factors, 1),
    p.pca_sd,
    p.pdef_sd,
    get(params, :ltri_indices, Int[]),
    get(params, :method, :noncentered)
)

MODEL_TO_STRUCTURE_MAP[:eigen] = :any

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
    
    eigen_data_matrix = Matrix(M[:data][!, vars_sym])
    eigen_data_matrix .-= mean(eigen_data_matrix, dims=1)
    
    params[:eigen_data] = eigen_data_matrix
    
    n_vars = length(vars_sym)
    n_factors = get(params, :n_factors, 1)
    if n_factors >= n_vars
        @warn "Number of factors ($n_factors) for eigen() module should be less than the number of variables ($n_vars). Setting to $(n_vars - 1)."
        n_factors = n_vars - 1
    end
    
    ltri_mask = [r >= c for r in 1:n_vars, c in 1:n_factors]
    ltri_indices = findall(vec(ltri_mask))
    
    params[:ltri_indices] = ltri_indices
    params[:n_factors] = n_factors
    params[:n_vars] = n_vars
    
    return true
end

function get_precomputes(m::Eigen, M::NamedTuple, mod_data::Dict)::NamedTuple
    eigen_data = get(mod_data[:params], :eigen_data, nothing)
    if isnothing(eigen_data)
        error("Eigen data matrix not found in module metadata. This indicates an issue in `get_datastructures!`.")
    end

    n_obs = size(eigen_data, 1)

    return (
        eigen_data=eigen_data,
        n_latent=n_obs
    )
end

function get_priors(m::Eigen, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    pca_sd_prior_str = _distribution_to_string(m.pca_sd)
    pdef_sd_prior_str = _distribution_to_string(m.pdef_sd)
    
    n_obs = size(spec.hyper.eigen_data, 1)

    priors = String[]
    push!(priors, "$(p_names.v_raw) ~ MvNormal(zeros(T, $(length(m.ltri_indices))), I)")
    push!(priors, "$(p_names.pca_sd) ~ filldist($(pca_sd_prior_str), $(m.n_factors))")
    push!(priors, "$(p_names.pdef_sd) ~ filldist($(pdef_sd_prior_str), $(m.n_vars))")
    
    if m.method == :noncentered
        push!(priors, "$(p_names.factors_flat) ~ MvNormal(zeros(T, $(n_obs * m.n_factors)), I)")
    end

    return join(priors, "\n    ")
end

function get_updates(m::Eigen, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    
    n_obs = size(spec.hyper.eigen_data, 1)
    n_factors = m.n_factors
    n_vars = m.n_vars

    common_code = """
        v_mat = zeros(T, $(n_vars), $(n_factors))
        v_mat[$(m.ltri_indices)] .= $(p_names.v_raw)
        
        U = householder_to_eigenvector(v_mat, $(n_vars), $(n_factors))
        L = U * Diagonal($(p_names.pca_sd))
        Psi = Diagonal($(p_names.pdef_sd).^2) + (T(M.noise) * I)
        
        Y_eigen_data = spec_registry[:$(key)].hyper.eigen_data
    """

    noncentered_code = """
        # --- Factor Model for Eigen Component (Non-Centered): $(key) ---
        let
            $(common_code)
            F = reshape($(p_names.factors_flat), $(n_obs), $(n_factors))
            Y_hat = F * L'
            
            for i in 1:$(n_obs)
                Turing.@addlogprob! logpdf(MvNormal(Y_hat[i, :], Psi), Y_eigen_data[i, :])
            end
            
            $(eta_target) .+= view(F, :, 1)
        end
    """

    centered_code = """
        # --- Factor Model for Eigen Component (Centered): $(key) ---
        let
            $(common_code)
            # Define the latent factors matrix that will be sampled row-by-row
            $(p_names.latent) = Matrix{T}(undef, $(n_obs), $(n_factors))
            Cov_F_row = Symmetric(L * L')
            
            for i in 1:$(n_obs)
                $(p_names.latent)[i, :] ~ MvNormal(zeros(T, $(n_factors)), Cov_F_row)
            end
            
            Y_hat = $(p_names.latent) * L'
            
            for i in 1:$(n_obs)
                Turing.@addlogprob! logpdf(MvNormal(Y_hat[i, :], Psi), Y_eigen_data[i, :])
            end
            
            $(eta_target) .+= view($(p_names.latent), :, 1)
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

function get_effects(
    m::Eigen, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    p_names_vec = string.(FlexiChains.parameters(chain))
    
    n_obs_train = size(spec.hyper.eigen_data, 1)
    n_factors = m.n_factors
    n_obs_full = N_total

    local factor_samples
    if m.method == :noncentered
        factors_flat_name = _find_parameter(p_names_vec, string(spec.key), "factors_flat", nothing, false)
        if isempty(factors_flat_name)
            @warn "Parameter 'factors_flat' for Eigen component $(spec.key) not found. Returning zero-matrix."
            factor_samples = zeros(Float64, n_obs_full, n_samples)
        else
            factor_samples_train = get_params_vector(chain, factors_flat_name, n_obs_train * n_factors)
            factor_samples = zeros(Float64, n_obs_full, n_samples)
            for j in 1:n_samples
                F_matrix_j = reshape(factor_samples_train[j, :], n_obs_train, n_factors)
                factor_samples[1:n_obs_train, j] = F_matrix_j[:, 1]
            end
        end
    else # :centered
        latent_name = _find_parameter(p_names_vec, string(spec.key), "latent", nothing, false)
        if isempty(latent_name)
            @warn "Parameter 'latent' for Eigen component $(spec.key) not found. Returning zero-matrix."
            factor_samples = zeros(Float64, n_obs_full, n_samples)
        else
            latent_samples_train = get_params_vector(chain, latent_name, n_obs_train * n_factors)
            factor_samples = zeros(Float64, n_obs_full, n_samples)
            for j in 1:n_samples
                F_matrix_j = reshape(latent_samples_train[j, :], n_obs_train, n_factors)
                factor_samples[1:n_obs_train, j] = F_matrix_j[:, 1]
            end
        end
    end

    for k in 1:outcomes_N
        push!(structured_effects, factor_samples)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
