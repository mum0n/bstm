# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    BCGN <: ComponentModel

A component model for a Bipartite Graph Convolutional Network (BCGN) effect.
This model induces a GMRF structure on one set of nodes based on their shared
connections in the other partition of a bipartite graph.

# Fields
- `sigma::Distribution`: The prior distribution for the standard deviation of the effect.
- `bipartite_adj::AbstractMatrix`: The bipartite adjacency matrix. This is typically
  set by the `get_datastructures!` method.
"""
struct BCGN <: ComponentModel
    sigma::Distribution
    bipartite_adj::AbstractMatrix
end

# Add to the central component constructor registry.
# The `bipartite_adj` parameter is populated by `get_datastructures!`.
COMPONENT_CONSTRUCTORS[:bcgn] = (p, params) -> BCGN(p.sigma, get(params, :bipartite_adj, sparse(zeros(1,1))))

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[BCGN] = :spatial

"""
    get_datastructures!(m_type::Type{<:BCGN}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `BCGN` component. It expects a unipartite
adjacency matrix `W` and converts it to a bipartite representation, which is then
stored in the module's parameters. It also sets up the spatial context (`s_idx`, `s_N`).
"""
function get_datastructures!(m_type::Type{<:BCGN}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    variables = mod_data[:variables]

    # Ensure W is available, either directly in params or in M
    if !haskey(params, :W)
        error("The `bcgn()` module requires a unipartite adjacency matrix `W` to be provided.")
    end

    W_uni = params[:W]
    if W_uni isa Expr || W_uni isa Symbol
        calling_mod = get(M, :calling_module, Main)
        try
            W_uni = Core.eval(calling_mod, W_uni)
        catch e
            error("Could not evaluate `W` argument `$(W_uni)` for BCGN component. Error: $e")
        end
    end

    if !isa(W_uni, AbstractMatrix) || isempty(W_uni)
        error("Provided `W` for BCGN model is not a valid non-empty matrix.")
    end

    # Convert the unipartite graph to a bipartite representation.
    # This function is assumed to be available in the execution environment.
    bipartite_res = adjacency_to_bipartite(W_uni)
    params[:bipartite_adj] = bipartite_res.bipartite_adj
    
    # The latent effect is on the first set of nodes from the bipartition.
    M[:s_N] = size(bipartite_res.bipartite_adj, 1)
    
    # The user provides an index into the original graph's nodes. We need to map this
    # to the indices of the first partition set.
    if !isempty(variables)
        s_var_sym = Symbol(variables[1])
        if !hasproperty(M[:data], s_var_sym)
            error("Spatial index variable ':$s_var_sym' for BCGN model not found in data.")
        end
        
        original_indices = M[:data][!, s_var_sym]
        set1_map = Dict(original_node => new_idx for (new_idx, original_node) in enumerate(bipartite_res.set1))
        
        # Map the observation indices to the new, smaller set of latent indices.
        # If an observation's index is not in set1, its effect will be zero.
        M[:s_idx] = [get(set1_map, idx, 0) for idx in original_indices]
        
        if any(iszero, M[:s_idx])
            @warn "Some observations in the BCGN component do not map to the primary node partition. Their spatial effect will be zero."
        end
    else
        error("BCGN model requires a spatial index variable to map observations to the graph nodes.")
    end

    return true
end

"""
    get_precomputes(m::BCGN, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `BCGN` component. It constructs
the precision matrix `Q_template` from the one-mode projection of the bipartite graph
and computes its spectral decomposition for efficient sampling.
"""
function get_precomputes(m::BCGN, M::NamedTuple, mod_data::Dict)::NamedTuple
    B = m.bipartite_adj
    if isempty(B)
        error("BCGN component has an empty bipartite adjacency matrix.")
    end

    # Create the precision matrix from the one-mode projection onto the first set of nodes.
    W_proj = B * B'
    
    # For a standard graph Laplacian, self-loops (diagonal elements) are set to zero.
    W_proj[diagind(W_proj)] .= 0
    W_proj = dropzeros(W_proj)

    # Build the graph Laplacian from the projected adjacency matrix: Q = D - W
    D_proj = spdiagm(0 => vec(sum(W_proj, dims=2)))
    Q_template = D_proj - W_proj
    
    n_latent = size(Q_template, 1)
    rank_deficiency = 1 # The one-mode projection typically results in a connected graph.

    # Compute eigendecomposition for spectral sampling
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values

    # Compute scaling factor
    scaling_factor = _compute_scaling_factor(L, rank_deficiency)
    
    # Rescale Q_template and eigenvalues
    Q_template_scaled = Q_template ./ scaling_factor
    L_scaled = L ./ scaling_factor

    return (Q_template=Q_template_scaled, scaling_factor=scaling_factor, U=U, L=L_scaled, n_latent=n_latent)
end

"""
    get_priors(m::BCGN, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `BCGN` component's priors.
It defines the prior for `sigma` and the latent field `raw`.
"""
function get_priors(m::BCGN, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    return """
        $(p_names.sigma) ~ NamedDist($(sigma_prior_str), :$(p_names.sigma))
        $(p_names.raw) ~ NamedDist(MvNormal(zeros(T, spec_registry[:$(spec.key)].precomputes.n_latent), I), :$(p_names.raw))
    """
end

"""
    get_updates(m::BCGN, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `BCGN` component's effect
and adding it to the linear predictor (`eta`), using the spectral method.
"""
function get_updates(m::BCGN, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- BCGN Component: $(spec.key) (Spectral Method) ---
        local diag_D = $(p_names.sigma) ./ sqrt.(spec_registry[:$(spec.key)].precomputes.L .+ M.noise)
        diag_D[1] = zero(T) # Enforce sum-to-zero constraint
        local $(p_names.latent) = spec_registry[:$(spec.key)].precomputes.U * (diag_D .* $(p_names.raw))
        
        # Handle cases where an observation's index is not in the primary partition (s_idx=0)
        for i in 1:length($(eta_target))
            if M.s_idx[i] > 0
                $(eta_target)[i] += $(p_names.latent)[M.s_idx[i]]
            end
        end
    """
end

"""
    get_effects(m::BCGN, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `BCGN` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::BCGN, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    sigma_samples = get(chain, p_names.sigma)
    raw_samples = get(chain, p_names.raw)

    n_latent = spec.precomputes.n_latent
    noise = M.noise
    U = spec.precomputes.U
    L = spec.precomputes.L

    # Determine indices for reconstruction (training or prediction)
    idx_to_use = isnothing(PS) ? M.s_idx : PS.s_idx
    
    reconstructed_effects = zeros(n_samples, n_latent)

    for i in 1:n_samples
        current_sigma = sigma_samples[i]
        current_raw = raw_samples[i, :]
        diag_D = current_sigma ./ sqrt.(L .+ noise)
        diag_D[1] = 0.0 # Enforce sum-to-zero
        reconstructed_effects[i, :] = U * (diag_D .* current_raw)
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    # Handle observations that do not map to the primary partition (index 0)
    indexed_mean = [idx > 0 ? mean_effect[idx] : 0.0 for idx in idx_to_use]
    indexed_lower = [idx > 0 ? lower_ci[idx] : 0.0 for idx in idx_to_use]
    indexed_upper = [idx > 0 ? upper_ci[idx] : 0.0 for idx in idx_to_use]

    return (structured=(mean=indexed_mean, lower=indexed_lower, upper=indexed_upper),)
end
