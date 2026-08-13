"""
    NetworkFlow <: ComponentModel

A component model for network flow processes, where dependencies are directional or
influenced by underlying landscape features. It parameterizes a Gaussian Markov
Random Field (GMRF) on a graph with data-driven edge weights, often representing
habitat conductivity or resistance.

# Version
v1.1.0 (2026-08-11)

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

# Computational Methods
- `:cholesky` (Default, AD-friendly): An AD-safe method using dense Cholesky factorization.
- `:cholesky_sparse` (Didactic, Not AD-friendly): A more memory-efficient method using
  sparse Cholesky factorization, suitable for gradient-free samplers.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@bstm`.
  - `habitat`: A `Symbol` pointing to a column in the data, or a `Vector` of length `s_N`.
- **Optional (in `random()` call)**:
  - `beta`: `UnivariateDistribution`, prior for the habitat effect parameter. Default: `Normal(0, 1)`.
  - `sigma`: `UnivariateDistribution`, prior for the marginal standard deviation. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:cholesky` or `:cholesky_sparse`). Default: `:cholesky`.

# Outputs (Parameter Names)
- `beta_<key>`: The habitat effect parameter.
- `sigma_<key>`: The marginal standard deviation of the latent field.
- `innovations_<key>`: The raw standard normal innovations for the latent field.
"""
struct NetworkFlow <: ComponentModel
    beta::UnivariateDistribution
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:networkflow] = NetworkFlow
COMPONENT_CONSTRUCTORS[:networkflow] = (p, params) -> NetworkFlow(
    p.beta, p.sigma, get(params, :method, :cholesky)
)

MODEL_TO_STRUCTURE_MAP[:networkflow] = :spatial

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

function get_precomputes(m::NetworkFlow, M::NamedTuple, mod_data::Dict)::NamedTuple
    W = M.W
    I, J, _ = findnz(W)
    return (W_I=I, W_J=J, n_latent=size(W, 1))
end

function get_priors(
    m::NetworkFlow, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    priors = String[]

    push!(priors, "$(p_names.beta) ~ $(_distribution_to_string(m.beta))")
    push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")
    push!(priors, "$(p_names.innovations) ~ DynamicPPL.NamedDist(MvNormal(zeros(T, M.s_N), I), :$(p_names.innovations))") # Raw standard normal innovations

    return join(priors, "\n    ")
end

function get_updates(
    m::NetworkFlow, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

    common_code = """
        # 1. Construct the weighted adjacency matrix based on habitat and beta.
        W_I = spec_registry[:$(key)].hyper.W_I
        W_J = spec_registry[:$(key)].hyper.W_J
        habitat = M[Symbol("habitat_$(key)")]
        
        V_beta = exp.($(p_names.beta) .* (habitat[W_I] .+ habitat[W_J]) ./ 2.0)
        W_beta = sparse(W_I, W_J, V_beta, M.s_N, M.s_N)
        
        # 2. Construct the weighted graph Laplacian (precision matrix).
        D_beta = Diagonal(vec(sum(W_beta, dims=2)))
        Q_beta = D_beta - W_beta
    """

    cholesky_code = """
        # --- NetworkFlow Component (Cholesky, AD-Safe): $(key) ---
        let
            $(common_code)
            
            F = cholesky(Symmetric(Matrix(Q_beta) + M.noise * I))
            $(p_names.latent) = $(p_names.sigma) .* (F.L' \\ $(p_names.innovations))
            
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """

    cholesky_sparse_code = """
        # --- NetworkFlow Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            $(common_code)
            
            F = cholesky(Symmetric(Q_beta + M.noise * I))
            $(p_names.latent) = $(p_names.sigma) .* (F.L' \\ $(p_names.innovations))
            
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """

    if m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    else
        error("Unsupported method '$(m.method)' for NetworkFlow component.")
    end
end

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
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))

    for k_outcome in 1:outcomes_N
        beta_name = _find_parameter(p_names_vec, string(key), "beta", k_outcome, is_multivariate_model)
        sigma_name = _find_parameter(p_names_vec, string(key), "sigma", k_outcome, is_multivariate_model)
        innovations_name = _find_parameter(p_names_vec, string(key), "innovations", k_outcome, is_multivariate_model)

        if isempty(beta_name) || isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for NetworkFlow component $(key) (outcome $k_outcome) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        beta_samples = get_params_vector(chain, beta_name, 1)[:, 1]
        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innovations_name, s_N)

        reconstructed_effects_k = zeros(Float64, s_N, n_samples)

        for i in 1:n_samples
            V_beta_i = exp.(beta_samples[i] .* (habitat[W_I] .+ habitat[W_J]) ./ 2.0)
            W_beta_i = sparse(W_I, W_J, V_beta_i, s_N, s_N)
            D_beta_i = Diagonal(vec(sum(W_beta_i, dims=2)))
            Q_beta_i = D_beta_i - W_beta_i
            
            F_i = cholesky(Symmetric(Matrix(Q_beta_i) + M.noise * I))
            
            reconstructed_effects_k[:, i] = sigma_samples[i] .*
                                            (F_i.L' \ innovations_samples[i, :])
        end
        
        s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)
        indexed_effects = reconstructed_effects_k[s_idx_full, :]
        push!(structured_effects, indexed_effects)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
