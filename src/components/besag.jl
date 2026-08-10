"""
    Besag <: ComponentModel

A component for a Besag model, also known as an Intrinsic Conditional Autoregressive
(ICAR) model. This is a fundamental model for spatial data on a lattice or graph,
where the value at a location is assumed to be conditionally dependent on the
average of its neighbors.

# Version
v1.1.2 (2026-08-10)

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

# Add to the central component constructor registry.
COMPONENT_TYPE_REGISTRY[:besag] = Besag
COMPONENT_CONSTRUCTORS[:besag] = (p, params) -> Besag(
    p.sigma, get(params, :method, :spectral)
)

# Add to the model-to-structure map.
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

Generates priors for the scale parameter `sigma` and the raw innovations `raw`.
"""
function get_priors(
    m::Besag, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    
    return """
    $(p_names.sigma) ~ $(_distribution_to_string(m.sigma))
    $(p_names.raw) ~ MvNormal(zeros($(n_latent)), I)
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
function get_updates(
    m::Besag, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent

    spectral_code = """
        # --- Besag (ICAR) Component: $(key) (Spectral Method) ---
        let
            local U = spec_registry[:$(key)].hyper.U
            local L = spec_registry[:$(key)].hyper.L
            local diag_D = $(p_names.sigma) ./ sqrt.(L .+ M.noise)
            diag_D[1] = 0.0 # Enforce sum-to-zero constraint
            local latent_field = U * (diag_D .* $(p_names.raw))
            $(eta_target) .+= view(latent_field, M.s_idx)
        end
    """

    cholesky_code = """
        # --- Besag (ICAR) Component: $(key) (Cholesky Method, AD-Safe) ---
        let
            local F = spec_registry[:$(key)].hyper.cholesky_factor
            local latent_field_raw = F.L' \\ $(p_names.raw)
            
            # Apply soft sum-to-zero constraint for identifiability
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_raw)
            )
            
            local latent_field = latent_field_raw .* $(p_names.sigma)
            
            $(eta_target) .+= view(latent_field, M.s_idx)
        end
    """

    cholesky_sparse_code = """
        # --- Besag (ICAR) Component: $(key) (Sparse Cholesky, Not AD-Safe) ---
        # WARNING: This method is for didactic purposes and is NOT compatible with
        # automatic differentiation (e.g., NUTS sampler).
        let
            local Q = spec_registry[:$(key)].hyper.Q_template
            local F = cholesky(Symmetric(Q + M.noise * I))
            local latent_field_raw = F.L' \\ $(p_names.raw)
            
            # Apply soft sum-to-zero constraint for identifiability
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_raw)
            )
            
            local latent_field = latent_field_raw .* $(p_names.sigma)
            
            $(eta_target) .+= view(latent_field, M.s_idx)
        end
    """

    if m.method == :spectral
        return spectral_code
    elseif m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    else
        error(
            "Unsupported method '$(m.method)' for Besag component. Supported " *
            "methods are :spectral, :cholesky, and :cholesky_sparse."
        )
    end
end

"""
    get_effects(m::Besag, chain, M::NamedTuple, ...)::NamedTuple

Reconstructs the `Besag` component's spatial effect from posterior samples, applying a
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

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        raw_samples = get_params_vector(chain, string(p_names.raw), n_latent)

        effect_k = zeros(Float64, n_latent, n_samples)

        if m.method == :spectral
            U = spec.hyper.U
            L = spec.hyper.L
            for j in 1:n_samples
                diag_D = sigma_samples[j] ./ sqrt.(L .+ noise)
                diag_D[1] = 0.0 # Enforce sum-to-zero
                effect_k[:, j] = U * (diag_D .* raw_samples[j, :])
            end
        else # :cholesky or :cholesky_sparse
            # For reconstruction, the pre-computed dense Cholesky factor is used for
            # both dense and sparse Cholesky methods, as AD is not involved here.
            F = spec.hyper.cholesky_factor
            for j in 1:n_samples
                latent_field_raw = F.L' \ raw_samples[j, :]
                latent_field_centered = latent_field_raw .- mean(latent_field_raw)
                effect_k[:, j] = latent_field_centered .* sigma_samples[j]
            end
        end
        
        s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)
        indexed_effects = effect_k[s_idx_full, :]
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
