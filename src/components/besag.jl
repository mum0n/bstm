"""
    Besag <: ComponentModel

A component for a Besag model, also known as an Intrinsic Conditional Autoregressive
(ICAR) model. This is a fundamental model for spatial data on a lattice or graph,
where the value at a location is assumed to be conditionally dependent on the
average of its neighbors.

# Version
v1.2.1 (2026-08-11)

# Mathematical Summary
The Besag model defines a Gaussian Markov Random Field (GMRF) with a singular
precision matrix (the graph Laplacian), making it an "intrinsic" GMRF. The
conditional distribution of the spatial effect \$\\phi_i\$ at location \$i\$, given all
other locations, is:
\$\\phi_i | \\phi_{j \\ne i} \\sim \\mathcal{N}\\left( \\frac{1}{d_i} \\sum_{j \\sim i} \\phi_j, \\frac{\\sigma^2}{d_i} \\right)\$
where \$j \\sim i\$ denotes that \$j\$ is a neighbor of \$i\$, and \$d_i\$ is the number of
neighbors.

The joint precision matrix is the graph Laplacian, \$Q = D - W\$, where \$D\$ is the
diagonal degree matrix and \$W\$ is the adjacency matrix. Because \$Q\$ is
rank-deficient (its rows sum to zero), a sum-to-zero constraint
(\$\\sum_i \\phi_i = 0\$) is imposed on the latent field to ensure identifiability
from the global intercept.

# Assumptions
- The spatial process is locally smooth, with values at neighboring locations being
  similar.
- The provided adjacency matrix `W` represents a connected graph. Disconnected
  "islands" will lead to a rank deficiency greater than 1 and cause the model to
  fail.

# Best Use Case
Modeling structured spatial random effects for areal or lattice data, such as
disease mapping, real estate analysis, or ecological modeling, where there is a
strong prior belief in local spatial smoothing.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `s_idx`).
  - An adjacency matrix `W` passed as a keyword argument to `@bstm`.
- **Optional (in `random()` call)**:
  - `sigma`: `UnivariateDistribution`, prior for the marginal standard deviation. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`). Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The marginal standard deviation of the latent field.
- `innovations_<key>`: The raw standard normal innovations for the latent field.
- `latent_<key>`: The reconstructed latent spatial field.


# Key References
- Besag, J. (1974). Spatial interaction and the statistical analysis of lattice
  systems. *Journal of the Royal Statistical Society: Series B (Methodological)*,
  36(2), 192-225.
- Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and
  Applications*. CRC Press.
- Wikipedia: Conditional autoregressive model

# Fields
- `sigma::UnivariateDistribution`: The prior for the marginal standard deviation of
  the conditional spatial effect.
- `method::Symbol`: The computational method. Can be `:spectral` (default, AD-safe),
  `:cholesky` (AD-safe, dense), or `:cholesky_sparse` (didactic, not AD-safe).
"""
struct Besag <: ComponentModel
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:besag] = Besag
COMPONENT_CONSTRUCTORS[:besag] = (p, params) -> Besag(
    p.sigma, get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:besag] = :spatial

"""
    get_datastructures!(m_type::Type{<:Besag}, M::Dict, mod_data::Dict)::Bool

Ensures a spatial context (`s_idx`, `s_N`, `W`) is established by calling the main
spatial processor.

# Assumptions
- A base adjacency matrix `W` must be provided in the main `@bstm` call or within
  the `random()` module.
- A spatial index variable must be provided in the `random()` call.
"""
function get_datastructures!(m_type::Type{<:Besag}, M::Dict, mod_data::Dict)::Bool
    process_spatial_module!(M, mod_data, Dict(), Dict())
    return true
end

"""
    get_precomputes(m::Besag, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the graph Laplacian (`Q_template`), its Cholesky factorization, and its
spectral decomposition (`U`, `L`) for use by different sampling methods.
"""
function get_precomputes(m::Besag, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.s_N
    W = M.W
    
    template = build_structure_template(:besag, n; W=W)
    
    # Pre-compute the dense Cholesky factor for the :cholesky method.
    F = cholesky(Symmetric(Matrix(template.matrix) + M.noise * I))
    
    return (
        Q_template=template.matrix, 
        scaling_factor=template.scaling_factor, 
        U=template.U, 
        L=template.L, 
        n_latent=n, 
        cholesky_factor=F
    )
end

"""
    get_priors(m::Besag, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for the scale parameter `sigma` and the raw innovations `innovations`.
"""
function get_priors(
    m::Besag, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    
    return """
    $(p_names.sigma) ~ $(_distribution_to_string(m.sigma)) # Prior for the marginal standard deviation
    $(p_names.innovations) ~ MvNormal(zeros(T, $(n_latent)), I) # Raw standard normal innovations for the latent field
    """
end


"""
    get_updates(m::Besag, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to sample the latent spatial field. Supports three methods:
- `:spectral` (default): An efficient, AD-safe method using spectral decomposition.
- `:cholesky`: An AD-safe didactic alternative using dense Cholesky factorization.
- `:cholesky_sparse`: A non-AD-safe didactic method using sparse Cholesky
  factorization, suitable for gradient-free samplers.
"""
function get_updates(m::PointProcess, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    if m.method == :lgcp
        return """
        # LGCP Model: $(spec.key)
        let
            hyper = spec_registry[:$(spec.key)].hyper
            Q_lgcp = hyper.inner_spec.Q_template
            F_lgcp = cholesky(Symmetric(Matrix(Q_lgcp) + M.noise * I))
            spatial_component = $(p_names.sigma) .* (F_lgcp.L' \\ $(p_names.innovations))
            
            log_intensity_surface = eta .+ spatial_component[M.s_idx]
            
            for i in 1:M.y_N
                y_i = M.y_obs[i]
                A_i = hyper.areas[M.s_idx[i]]
                lambda_i = exp(log_intensity_surface[i])
                
                Turing.@addlogprob! logpdf(Poisson(lambda_i * A_i), y_i)
            end
        end
        M.likelihood_handled = true
        """
    elseif m.method == :lgmcp
        return """
        # LGMCP Model: $(spec.key)
        let
            hyper = spec_registry[:$(spec.key)].hyper
            Q_lgmcp = hyper.inner_spec.Q_template
            F_lgmcp = cholesky(Symmetric(Matrix(Q_lgmcp) + M.noise * I))
            spatial_component = exp.(F_lgmcp.L' \\ $(p_names.innovations))
            
            mean_intensity_surface = exp.(eta) .* spatial_component[M.s_idx]
            
            for i in 1:M.y_N
                y_i = M.y_obs[i]
                A_i = hyper.areas[M.s_idx[i]]
                mu = mean_intensity_surface[i] * A_i
                
                r_nb = $(p_names.shape)
                p_nb = r_nb / (r_nb + mu)
                
                Turing.@addlogprob! logpdf(NegativeBinomial(r_nb, p_nb), y_i)
            end
        end
        M.likelihood_handled = true
        """
    elseif m.method == :sncp
        return """
        # SNCP Model: $(spec.key)
        let
            hyper = spec_registry[:$(spec.key)].hyper
            obs_locs = M.centroids
            parent_locs = hcat($(p_names.parent_locs_x), $(p_names.parent_locs_y))
            n_parents = length($(p_names.parent_locs_x))
            
            intensity_at_obs = zeros(T, M.s_N)
            for i in 1:M.s_N
                intensity_i = zero(T)
                for j in 1:n_parents
                    dist_sq = (obs_locs[i].x - parent_locs[j, 1])^2 + (obs_locs[i].y - parent_locs[j, 2])^2
                    kernel_val = exp(-0.5 * dist_sq / ($(p_names.ls)^2))
                    intensity_i += $(p_names.amplitude)[j] * kernel_val
                end
                intensity_at_obs[i] = intensity_i
            end
            
            for i in 1:M.y_N
                y_i = M.y_obs[i]
                s_i = M.s_idx[i]
                A_s = hyper.areas[s_i]
                lambda_s = intensity_at_obs[s_i] * A_s
                
                Turing.@addlogprob! logpdf(Poisson(lambda_s), y_i)
            end
        end
        M.likelihood_handled = true
        """
    end
    return ""
end



"""
    get_effects(m::Besag, chain, M::NamedTuple, ...)::NamedTuple

Reconstructs the `Besag` component's effect from posterior samples, applying a
sum-to-zero constraint for identifiability. This function dispatches on the method
used during sampling.
"""
function get_effects(
    m::Besag, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    n_latent = spec.hyper.n_latent
    noise = M.noise
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))

    for k in 1:outcomes_N
        sigma_name = _find_parameter(p_names_vec, string(spec.key), "sigma", k, is_multivariate_model)
        innovations_name = _find_parameter(p_names_vec, string(spec.key), "innovations", k, is_multivariate_model)

        if isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for Besag component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innovations_name, n_latent)

        effect_k = zeros(Float64, n_latent, n_samples)

        if m.method == :spectral
            U = spec.hyper.U
            L = spec.hyper.L
            for j in 1:n_samples # Iterate over posterior samples
                diag_D = sigma_samples[j] ./ sqrt.(L .+ noise) # Scale by sigma and add jitter
                diag_D[1] = 0.0 # Enforce sum-to-zero constraint for the intrinsic GMRF
                effect_k[:, j] = U * (diag_D .* innovations_samples[j, :]) # Apply spectral transformation
            end
        else # :cholesky or :cholesky_sparse
            # For reconstruction, the pre-computed dense Cholesky factor is used for
            # both dense and sparse Cholesky methods, as AD is not involved here.
            F = spec.hyper.cholesky_factor
            for j in 1:n_samples
                latent_field_raw = F.L' \ innovations_samples[j, :]
                latent_field_centered = latent_field_raw .- mean(latent_field_raw)
                effect_k[:, j] = latent_field_centered .* sigma_samples[j]
            end
        end
        
        s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx) # Combine spatial indices from training and prediction sets
        indexed_effects = effect_k[s_idx_full, :]
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
