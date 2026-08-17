"""
    NetworkFlow <: ComponentModel

A component model for network flow processes, where dependencies are directional or
influenced by underlying landscape features. It parameterizes a Gaussian Markov
Random Field (GMRF) on a graph with data-driven edge weights, often representing
habitat conductivity or resistance.

# Version
v1.1.1 (2026-08-14)

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
- `latent_<key>`: The reconstructed latent spatial field.
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

function get_precomputes(m::NetworkFlow, M::NamedTuple, mod_data::Dict)::NamedTuple
    # Ensure spatial context is established by process_random_module!
    s_N = M.s_N
    W = M.W
    
    params = mod_data[:params]
    data = M.data

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
        # Aggregate habitat data to spatial units if it's per-observation
        habitat_per_obs = data[!, habitat_val]
        habitat_aggregated = zeros(Float64, s_N)
        counts = zeros(Int, s_N)
        for i in 1:M.y_N
            s_i = M.s_idx[i]
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

    I, J, _ = findnz(W)
    return (W_I=I, W_J=J, n_latent=s_N, s_N=s_N, habitat_data=habitat_data)
end

function get_priors(
    m::NetworkFlow, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    priors = String[]

    push!(priors, "$(p_names.beta) ~ $(_distribution_to_string(m.beta))")
    push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")
    push!(priors, "$(p_names.innovations) ~ DynamicPPL.NamedDist(MvNormal(zeros(T, spec.hyper.n_latent), I), :$(p_names.innovations))")

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
        hyper = spec_registry[:$(key)].hyper
        W_I = hyper.W_I
        W_J = hyper.W_J
        habitat = hyper.habitat_data
        s_N = hyper.s_N
        
        V_beta = exp.($(p_names.beta) .* (habitat[W_I] .+ habitat[W_J]) ./ 2.0)
        W_beta = sparse(W_I, W_J, V_beta, s_N, s_N)
        
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
    m::NetworkFlow, chain::Chains, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device
    
    key = spec.key
    hyper = spec.hyper
    W_I = hyper.W_I
    W_J = hyper.W_J
    habitat_cpu = Array(hyper.habitat_data) # Ensure habitat data is on CPU for sparse construction
    s_N = hyper.s_N
    noise = M.noise

    # --- Index Handling: Combine training and prediction sets on device ---
    s_idx_train = M.s_idx # Already on device
    s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx)
        s_idx_pred_cpu = get(PS.data, :s_idx, [])
        vcat(s_idx_train, to_device(s_idx_pred_cpu))
    else
        s_idx_train
    end
    N_total = length(s_idx_full)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k_outcome in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k_outcome)
        beta_name = _find_parameter(p_names, string(p_names_k.beta), k_outcome, is_multivariate_model)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k_outcome, is_multivariate_model)
        innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k_outcome, is_multivariate_model)

        if isempty(beta_name) || isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for NetworkFlow component $(key) (outcome $k_outcome) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        beta_samples_cpu = get_params_vector(chain, beta_name, 1)[:, 1]
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples_cpu = get_params_matrix(chain, innovations_name, s_N)

        # Initialize the output matrix for the full latent field on the target device
        reconstructed_effects_k_device = to_device(zeros(Float64, s_N, n_samples))

        # --- Sample-wise Reconstruction on the Target Device ---
        for i in 1:n_samples
            # Construct precision matrix on CPU first, then move to device
            V_beta_i = exp.(beta_samples_cpu[i] .* (habitat_cpu[W_I] .+ habitat_cpu[W_J]) ./ 2.0)
            W_beta_i = sparse(W_I, W_J, V_beta_i, s_N, s_N)
            D_beta_i = Diagonal(vec(sum(W_beta_i, dims=2)))
            Q_beta_i = D_beta_i - W_beta_i
            
            # Move matrix to device and perform factorization
            Q_beta_i_device = to_device(Matrix(Q_beta_i))
            F_i = cholesky(Symmetric(Q_beta_i_device + noise * I))
            
            # Move innovations to device for the solve
            innov_i_device = to_device(innovations_samples_cpu[i, :])
            
            reconstructed_effects_k_device[:, i] = sigma_samples_cpu[i] .* (F_i.L' \ innov_i_device)
        end
        
        # Index the reconstructed effects for the full observation set and move back to CPU
        indexed_effects_device = reconstructed_effects_k_device[s_idx_full, :]
        push!(structured_effects, Array(indexed_effects_device))
    end

    return (structured=structured_effects, noisy=structured_effects)
end
