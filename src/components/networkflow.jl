"""
    NetworkFlow <: ComponentModel

A component model for network flow processes, where dependencies are directional or
influenced by underlying landscape features. It parameterizes a Gaussian Markov
Random Field (GMRF) on a graph with data-driven edge weights, often representing
habitat conductivity or resistance.

# Version
v1.0.0 (2026-08-08)

# Mathematical Summary
The component models a latent spatial field \$\\phi\$ as a GMRF,
\$\\phi \\sim \\mathcal{N}(0, (\\frac{1}{\\sigma^2} Q_{\\beta})^{-1})\$. The precision
matrix \$Q_{\\beta}\$ is a graph Laplacian that depends on a learned parameter \$\\beta\$.

The edge weight \$w_{ij}\$ between two connected spatial units \$i\$ and \$j\$ is defined
as a function of a habitat variable \$H\$ at those locations:
\$w_{ij} = \\exp\\left( \\beta \\cdot \\frac{H_i + H_j}{2} \\right)\$
A positive \$\\beta\$ implies that higher habitat values correspond to stronger
connectivity (lower movement cost), while a negative \$\\beta\$ implies higher values
correspond to stronger resistance.

The precision matrix is then constructed as the weighted graph Laplacian:
\$Q_{\\beta} = D_{\\beta} - W_{\\beta}\$
where \$W_{\\beta}\$ is the matrix of weights \$w_{ij}\$ and \$D_{\\beta}\$ is the diagonal
matrix of row sums of \$W_{\\beta}\$. By estimating \$\\beta\$, the model learns the
degree to which the habitat influences the spatial correlation structure.

# Assumptions
- The provided adjacency matrix `W` represents the base connectivity of the graph.
- The `habitat` data represents a measure of conductivity or resistivity.

# Best Use Case
Modeling animal movement, disease spread, or any spatial process where correlation
is not uniform but is facilitated or impeded by environmental features. It is a
powerful tool for creating non-stationary spatial models where the correlation
structure is learned from data.

# Key References
- McRae, B. H. (2006). Isolation by resistance. *Evolution*, 60(8), 1551-1561.
  (For the concept of resistance surfaces in ecology).
- Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and
  Applications*. CRC Press. (For the GMRF formulation).

# Fields
- `beta::UnivariateDistribution`: Prior for the parameter \$\\beta\$, which controls
  the influence of the habitat on connectivity.
- `sigma::UnivariateDistribution`: Prior for the marginal standard deviation of the
  latent field.
"""
struct NetworkFlow <: ComponentModel
    beta::UnivariateDistribution
    sigma::UnivariateDistribution
end

# Add to the central component constructor registry.
COMPONENT_TYPE_REGISTRY[:networkflow] = NetworkFlow
COMPONENT_CONSTRUCTORS[:networkflow] = (p, params) -> NetworkFlow(p.beta, p.sigma)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[:networkflow] = :spatial

"""
    get_datastructures!(m_type::Type{<:NetworkFlow}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `NetworkFlow` component. It establishes the
spatial context (s_idx, s_N, W) and resolves the `habitat` data from the provided
DataFrame or keyword arguments.

# Assumptions
- A base adjacency matrix `W` must be provided.
- The `habitat` data must be provided either as a column name in the DataFrame or
  as a vector of length `s_N` via keyword arguments.
"""
function get_datastructures!(
    m_type::Type{<:NetworkFlow}, M::Dict, mod_data::Dict
)::Bool
    process_spatial_module!(M, mod_data, Dict(), Dict())

    params = mod_data[:params]
    data = M[:data]
    s_N = M[:s_N]

    if !haskey(params, :habitat)
        error(
            "The `networkflow` model requires a `habitat` parameter specifying the " *
            "conductivity/resistivity data."
        )
    end

    habitat_val = params[:habitat]
    local habitat_data::Vector{Float64}

    if habitat_val isa Symbol
        if !hasproperty(data, habitat_val)
            error("Habitat variable ':$habitat_val' not found in the data frame.")
        end
        # Aggregate per-observation habitat data to per-unit data by taking the mean.
        habitat_per_obs = data[!, habitat_val]
        habitat_aggregated = zeros(Float64, s_N)
        counts = zeros(Int, s_N)
        for i in 1:M[:y_N]
            s_i = M[:s_idx][i]
            habitat_aggregated[s_i] += habitat_per_obs[i]
            counts[s_i] += 1
        end
        habitat_data = habitat_aggregated ./ max.(1, counts)

    elseif habitat_val isa AbstractVector
        if length(habitat_val) != s_N
            error(
                "Provided `habitat` vector length ($(length(habitat_val))) does not " *
                "match the number of spatial units s_N ($(s_N))."
            )
        end
        habitat_data = convert(Vector{Float64}, habitat_val)
    else
        error(
            "The `habitat` parameter must be a Symbol (column name) or a Vector of " *
            "length s_N."
        )
    end

    M[Symbol("habitat_", mod_data[:key])] = habitat_data
    return true
end

"""
    get_precomputes(m::NetworkFlow, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the sparse structure (I, J vectors) of the adjacency matrix `W`. This
avoids repeatedly finding non-zero elements inside the Turing model loop.
"""
function get_precomputes(m::NetworkFlow, M::NamedTuple, mod_data::Dict)::NamedTuple
    W = M.W
    I, J, _ = findnz(W)
    return (W_I=I, W_J=J, n_latent=size(W, 1))
end

"""
    get_priors(m::NetworkFlow, spec::NamedTuple, arch::String, outcome_idx, M)

Generates the Turing code for the priors on `beta`, `sigma`, and the raw innovations.
"""
function get_priors(
    m::NetworkFlow, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    priors = String[]

    push!(priors, "$(p_names.beta) ~ $(_distribution_to_string(m.beta))")
    push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")
    push!(priors, "$(p_names.raw) ~ MvNormal(zeros(M.s_N), I)")

    return join(priors, "\n    ")
end

"""
    get_updates(m::NetworkFlow, spec::NamedTuple, arch::String, outcome_idx, M)

Generates the Turing code to construct the habitat-weighted precision matrix, sample
the latent field, and add it to the linear predictor `eta`.
"""
function get_updates(
    m::NetworkFlow, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

    return """
        # --- NetworkFlow Component: $(key) ---
        let
            # 1. Construct the weighted adjacency matrix based on habitat and beta.
            local W_I = spec_registry[:$(key)].hyper.W_I
            local W_J = spec_registry[:$(key)].hyper.W_J
            local habitat = M[Symbol("habitat_$(key)")]
            
            local V_beta = exp.($(p_names.beta) .* (habitat[W_I] .+ habitat[W_J]) ./ 2.0)
            local W_beta = sparse(W_I, W_J, V_beta, M.s_N, M.s_N)
            
            # 2. Construct the weighted graph Laplacian (precision matrix).
            local D_beta = Diagonal(vec(sum(W_beta, dims=2)))
            local Q_beta = D_beta - W_beta
            
            # 3. Sample the latent field using a non-centered parameterization.
            #    The Cholesky factor of the precision matrix is used to transform
            #    standard normal noise.
            local F = cholesky(Symmetric(Matrix(Q_beta) + M.noise * I))
            local latent_field = $(p_names.sigma) .* (F.L' \\ $(p_names.raw))
            
            # 4. Add the effect to the linear predictor.
            $(eta_target) .+= view(latent_field, M.s_idx)
        end
    """
end

"""
    get_effects(m::NetworkFlow, chain, M::NamedTuple, ...)

Reconstructs the `NetworkFlow` component's effect from the MCMC chain's posterior
samples.
"""
function get_effects(
    m::NetworkFlow, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    key = spec.key
    
    W_I = spec.hyper.W_I
    W_J = spec.hyper.W_J
    habitat = M[Symbol("habitat_", key)]
    s_N = M.s_N

    for k_outcome in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k_outcome)

        beta_samples = get_params_vector(chain, string(p_names.beta), 1)[:, 1]
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        raw_samples = get_params_vector(chain, string(p_names.raw), s_N)

        reconstructed_effects_k = zeros(Float64, s_N, n_samples)

        for i in 1:n_samples
            # Reconstruct the precision matrix for each posterior sample
            V_beta_i = exp.(beta_samples[i] .* (habitat[W_I] .+ habitat[W_J]) ./ 2.0)
            W_beta_i = sparse(W_I, W_J, V_beta_i, s_N, s_N)
            D_beta_i = Diagonal(vec(sum(W_beta_i, dims=2)))
            Q_beta_i = D_beta_i - W_beta_i
            
            F_i = cholesky(Symmetric(Matrix(Q_beta_i) + M.noise * I))
            
            # Reconstruct the latent field for this sample
            reconstructed_effects_k[:, i] = sigma_samples[i] .*
                                            (F_i.L' \ raw_samples[i, :])
        end
        
        s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)
        indexed_effects = reconstructed_effects_k[s_idx_full, :]
        push!(structured_effects, indexed_effects)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
