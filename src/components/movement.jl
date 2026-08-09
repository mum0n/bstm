"""
    Movement <: ComponentModel

A component model for a latent spatial field whose correlation structure is
determined by a habitat conductivity or resistivity layer. It parameterizes
spatiotemporal paths or potentials by defining a Gaussian Markov Random Field (GMRF)
on a graph with data-driven edge weights. This allows for modeling non-stationary
spatial processes where movement or correlation is facilitated or impeded by
underlying environmental features.

# Version
v1.0.1 (2026-08-09)

# Mathematical Summary
The component models a latent spatial field \$\\phi\$ as a GMRF,
\$\\boldsymbol{\\phi} \\sim \\mathcal{N}(\\mathbf{0}, (\\sigma^2 \\mathbf{Q}_{\\beta})^{-1})\$,
where the precision matrix \$\\mathbf{Q}_{\\beta}\$ is a graph Laplacian that depends on a
learned parameter \$\\beta\$.

The edge weight \$w_{ij}\$ between two connected spatial units \$i\$ and \$j\$ is defined
as a function of the habitat conductivity/resistivity \$H\$ at those locations:
\$w_{ij} = \\exp\\left( \\beta \\cdot \\frac{H_i + H_j}{2} \\right)\$
A positive \$\\beta\$ implies that higher habitat values correspond to stronger
connectivity (lower movement cost), treating \$H\$ as a conductivity layer. A
negative \$\\beta\$ implies the opposite, treating \$H\$ as a resistivity layer.

The precision matrix is then constructed as the weighted graph Laplacian:
\$\\mathbf{Q}_{\\beta} = \\mathbf{D}_{\\beta} - \\mathbf{W}_{\\beta}\$
where \$\\mathbf{W}_{\\beta}\$ is the matrix of weights \$w_{ij}\$ and \$\\mathbf{D}_{\\beta}\$ is the
diagonal matrix of row sums of \$\\mathbf{W}_{\\beta}\$.

By estimating \$\\beta\$, the model learns the degree to which the habitat influences
the spatial correlation structure. This approach is closely related to the use of
circuit theory in landscape ecology.

# Assumptions
- The provided adjacency matrix `W` (or the grid structure from `habitat_raster`)
  represents the base connectivity of the spatial graph.
- The `habitat` data represents a measure of conductivity or resistivity, and the
  sign of the estimated `beta` will reflect this relationship.

# Best Use Case
Modeling animal movement, disease spread, or any spatial process where movement or
correlation is not uniform but is facilitated or impeded by underlying environmental
features. It is a powerful tool for creating non-stationary spatial models where the
correlation structure is learned from data.

# Key References
- McRae, B. H., Dickson, B. G., Keitt, T. H., & Shah, V. B. (2008). Using circuit
  theory to model connectivity in ecology, evolution, and conservation. *Ecology*,
  89(10), 2712-2724.
- Wikipedia: Laplacian matrix

# Fields
- `beta::UnivariateDistribution`: Prior for the parameter \$\\beta\$, which controls
  the influence of the habitat on connectivity.
- `sigma::UnivariateDistribution`: Prior for the marginal standard deviation of the
  latent field.
"""
struct Movement <: ComponentModel
    beta::UnivariateDistribution
    sigma::UnivariateDistribution
end

COMPONENT_TYPE_REGISTRY[:movement] = Movement
COMPONENT_CONSTRUCTORS[:movement] = (p, params) -> Movement(p.beta, p.sigma)
MODEL_TO_STRUCTURE_MAP[:movement] = :spatial

function _raster_to_graph(raster::AbstractMatrix)
    rows, cols = size(raster)
    n_units = rows * cols
    W = spzeros(Int, n_units, n_units)
    
    for r in 1:rows, c in 1:cols
        idx = (c - 1) * rows + r
        # 8-neighbor connectivity (Queen's case)
        for dr in -1:1, dc in -1:1
            if dr == 0 && dc == 0; continue; end
            nr, nc = r + dr, c + dc
            if 1 <= nr <= rows && 1 <= nc <= cols
                n_idx = (nc - 1) * rows + nr
                W[idx, n_idx] = 1
            end
        end
    end
    return W
end

"""
    get_datastructures!(m_type::Type{<:Movement}, M::Dict, mod_data::Dict)::Bool

Establishes the spatial context. If an adjacency matrix `W` is not provided, it
attempts to generate one from a `habitat_raster` parameter, assuming a regular
grid structure. It also resolves the `habitat` data vector.
"""
function get_datastructures!(m_type::Type{<:Movement}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    data = M[:data]

    if !haskey(M, :W)
        if haskey(params, :habitat_raster)
            raster = params[:habitat_raster]
            if !(raster isa AbstractMatrix); error("`habitat_raster` must be a matrix."); end
            M[:W] = _raster_to_graph(raster)
            M[:s_N] = size(raster, 1) * size(raster, 2)
            # For raster-based models, s_idx must map observations to grid cells.
            # This requires coordinate columns in the data.
            if !hasproperty(data, :s_x) || !hasproperty(data, :s_y)
                error("Raster-based Movement model requires `s_x` and `s_y` columns in data to map observations to grid cells.")
            end
            # This part would require a function to map continuous coordinates to grid cell indices.
            # For simplicity, we assume this mapping is pre-computed and passed as s_idx.
            if !haskey(M, :s_idx); error("`s_idx` must be provided for raster-based Movement models."); end
        else
            error("The `movement` model requires either an adjacency matrix `W` or a `habitat_raster` parameter.")
        end
    end
    
    # Now that W is guaranteed to exist, run the standard spatial processor.
    process_spatial_module!(M, mod_data, Dict(), Dict())
    s_N = M[:s_N]

    if !haskey(params, :habitat)
        error("The `movement` model requires a `habitat` parameter specifying the conductivity/resistivity data.")
    end

    habitat_val = params[:habitat]
    local habitat_data::Vector{Float64}

    if habitat_val isa Symbol
        if !hasproperty(data, habitat_val); error("Habitat variable ':$habitat_val' not found in data."); end
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
        if length(habitat_val) != s_N; error("Provided `habitat` vector length ($(length(habitat_val))) does not match s_N ($(s_N))."); end
        habitat_data = convert(Vector{Float64}, habitat_val)
    else
        error("The `habitat` parameter must be a Symbol (column name) or a Vector of length s_N.")
    end

    M[Symbol("habitat_", mod_data[:key])] = habitat_data
    return true
end

"""
    get_precomputes(m::Movement, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the sparse structure (I, J vectors) of the adjacency matrix `W`.
"""
function get_precomputes(m::Movement, M::NamedTuple, mod_data::Dict)::NamedTuple
    W = M.W
    I, J, _ = findnz(W)
    return (W_I=I, W_J=J, n_latent=size(W,1))
end

"""
    get_priors(m::Movement, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `beta`, `sigma`, and the raw innovations.
"""
function get_priors(m::Movement, spec::NamedTuple, arch::String, outcome_idx, M)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    priors = String[]

    push!(priors, "$(p_names.beta) ~ $(_distribution_to_string(m.beta))")
    push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")
    push!(priors, "$(p_names.raw) ~ MvNormal(zeros(spec.hyper.n_latent), I)")

    return join(priors, "\n    ")
end

"""
    get_updates(m::Movement, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to construct the habitat-weighted precision matrix, sample the
latent field, and add it to the linear predictor `eta`.
"""
function get_updates(m::Movement, spec::NamedTuple, arch::String, outcome_idx, M)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

    return """
        # --- Movement Component: $(key) ---
        let
            local W_I = spec.hyper.W_I
            local W_J = spec.hyper.W_J
            local habitat = M[Symbol("habitat_$(key)")]
            
            local V_beta = exp.($(p_names.beta) .* (habitat[W_I] .+ habitat[W_J]) ./ 2.0)
            local W_beta = sparse(W_I, W_J, V_beta, M.s_N, M.s_N)
            
            local D_beta = Diagonal(vec(sum(W_beta, dims=2)))
            local Q_beta = D_beta - W_beta
            
            local F = cholesky(Symmetric(Matrix(Q_beta) + M.noise * I))
            local latent_field = $(p_names.sigma) .* (F.L' \\ $(p_names.raw))
            
            $(eta_target) .+= view(latent_field, M.s_idx)
        end
    """
end

"""
    get_effects(m::Movement, chain, M, n_samples, outcomes_N, spec, PS, N_total)::NamedTuple

Reconstructs the `Movement` component's effect from posterior samples.
"""
function get_effects(m::Movement, chain, M, n_samples, outcomes_N, spec, PS, N_total)::NamedTuple
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
            V_beta_i = exp.(beta_samples[i] .* (habitat[W_I] .+ habitat[W_J]) ./ 2.0)
            W_beta_i = sparse(W_I, W_J, V_beta_i, s_N, s_N)
            D_beta_i = Diagonal(vec(sum(W_beta_i, dims=2)))
            Q_beta_i = D_beta_i - W_beta_i
            
            F_i = cholesky(Symmetric(Matrix(Q_beta_i) + M.noise * I))
            reconstructed_effects_k[:, i] = sigma_samples[i] .* (F_i.L' \ raw_samples[i, :])
        end
        
        s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)
        indexed_effects = reconstructed_effects_k[s_idx_full, :]
        push!(structured_effects, indexed_effects)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
