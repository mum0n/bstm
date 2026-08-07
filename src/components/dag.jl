# This file contains the proposed new and updated functions for the bstm refactoring.
 
 
"""
    DAG <: ComponentModel

A component model for a Directed Acyclic Graph (DAG) structure. This model is useful for
capturing causal or directional dependencies between spatial or other units. The effect at
each node is a linear combination of its parents' effects plus an independent innovation.

# Fields
- `rho::Distribution`: The prior for the autoregressive parameter `rho`, controlling the strength of parental influence.
- `sigma::Distribution`: The prior for the standard deviation of the innovations at each node.
"""
struct DAG <: ComponentModel
    rho::Distribution
    sigma::Distribution
end

# Add to the central component constructor registry.
COMPONENT_CONSTRUCTORS[:dag] = (p, params) -> DAG(p.rho, p.sigma)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[DAG] = :spatial

"""
    get_datastructures!(m_type::Type{<:DAG}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `DAG` component. It ensures that an adjacency
matrix `W` is provided, warns if it is not lower triangular (a requirement for the
forward-substitution algorithm), and sets up the spatial context (`s_idx`, `s_N`).
"""
function get_datastructures!(m_type::Type{<:DAG}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    variables = mod_data[:variables]

    # Ensure W is available, either directly in params or in M
    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr || w_val isa Symbol
            calling_mod = get(M, :calling_module, Main)
            try
                M[:W] = Core.eval(calling_mod, w_val)
            catch e
                error("Could not evaluate `W` argument `$(w_val)` for DAG component. Error: $e")
            end
        else
            M[:W] = w_val
        end
    end

    if !haskey(M, :W)
        error("DAG model requires an adjacency matrix `W` to be provided.")
    end

    W = M[:W]
    if !istril(W)
        @warn "The adjacency matrix `W` for the DAG component is not lower triangular. The model assumes a causal ordering. Results may be incorrect if the graph contains cycles."
    end

    M[:s_N] = size(W, 1)

    if isempty(variables)
        M[:s_idx] = collect(1:M[:s_N])
        @warn "Spatial index variable not provided for DAG. Assuming `s_idx = 1:s_N`."
    else
        s_var_sym = Symbol(variables[1])
        if !hasproperty(M[:data], s_var_sym)
            error("Spatial index variable ':$s_var_sym' for DAG model not found in data.")
        end
        M[:s_idx] = M[:data][!, s_var_sym]
    end

    return true
end

"""
    get_precomputes(m::DAG, M::NamedTuple, mod_data::Dict)::NamedTuple

For the `DAG` component, this function stores the adjacency matrix `W` as the `Q_template`.
The model's forward substitution logic relies on this matrix representing the causal graph structure.
"""
function get_precomputes(m::DAG, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.s_N
    W = sparse(M.W) # Ensure W is sparse

    # For DAG, the Q_template is the adjacency matrix itself.
    # The model assumes a causal ordering (e.g., lower triangular).
    return (Q_template=W, n_latent=n)
end

"""
    get_priors(m::DAG, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code for the priors on the `DAG` component's parameters.
This includes priors for `rho`, `sigma`, and the independent innovations for each node.
"""
function get_priors(m::DAG, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    rho_prior_str = _distribution_to_string(m.rho)
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    # Latent field prior for the innovations (non-centered parameterization)
    # innov ~ MvNormal(zeros(T, n_latent), I)
    
    return """
        $(p_names.rho) ~ NamedDist($(rho_prior_str), :$(p_names.rho))
        $(p_names.sigma) ~ NamedDist($(sigma_prior_str), :$(p_names.sigma))
        $(p_names.innov) ~ NamedDist(MvNormal(zeros(T, spec_registry[:$(spec.key)].precomputes.n_latent), I), :$(p_names.innov))
    """
end

"""
    get_updates(m::DAG, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code to construct the `DAG` component's effect using forward substitution
and adds the result to the linear predictor `eta`.
"""
function get_updates(m::DAG, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- DAG Component: $(spec.key) ---
        local W_dag = spec_registry[:$(spec.key)].precomputes.Q_template
        local innovations = $(p_names.innov)
        local rho_val = $(p_names.rho)
        local sigma_val = $(p_names.sigma)
        local n_latent = spec_registry[:$(spec.key)].precomputes.n_latent

        local T_num = promote_type(typeof(rho_val), eltype(innovations))
        local $(p_names.latent) = zeros(T_num, n_latent)

        # Assumes W_dag is lower triangular, representing a valid DAG ordering.
        for i in 1:n_latent
            local parent_effect = zero(T_num)
            # Efficiently iterate over non-zero elements in the i-th column of the sparse matrix.
            # This corresponds to summing over parent nodes j -> i.
            for j_ptr in nzrange(W_dag, i)
                parent_idx = W_dag.rowval[j_ptr]
                parent_effect += W_dag.nzval[j_ptr] * $(p_names.latent)[parent_idx]
            end
            $(p_names.latent)[i] = rho_val * parent_effect + innovations[i]
        end
        $(p_names.latent) .*= sigma_val
        
        $(eta_target) .+= $(p_names.latent)[M.s_idx]
    """
end

"""
    get_effects(m::DAG, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `DAG` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::DAG, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    rho_samples = get(chain, p_names.rho)
    sigma_samples = get(chain, p_names.sigma)
    innov_samples = get(chain, p_names.innov)

    n_latent = spec.precomputes.n_latent
    W_dag = spec.precomputes.Q_template

    # Determine indices for reconstruction (training or prediction)
    idx_to_use = isnothing(PS) ? M.s_idx : PS.s_idx
    
    reconstructed_effects = zeros(n_samples, n_latent)

    for i in 1:n_samples
        current_rho = rho_samples[i]
        current_sigma = sigma_samples[i]
        current_innov = innov_samples[i, :]
        
        latent_field_i = zeros(eltype(current_rho), n_latent)
        
        for j in 1:n_latent
            parent_effect = zero(eltype(current_rho))
            for j_ptr in nzrange(W_dag, j)
                parent_idx = W_dag.rowval[j_ptr]
                parent_effect += W_dag.nzval[j_ptr] * latent_field_i[parent_idx]
            end
            latent_field_i[j] = current_rho * parent_effect + current_innov[j]
        end
        
        reconstructed_effects[i, :] = latent_field_i .* current_sigma
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    indexed_mean = mean_effect[idx_to_use]
    indexed_lower = lower_ci[idx_to_use]
    indexed_upper = upper_ci[idx_to_use]

    return (structured=(mean=indexed_mean, lower=indexed_lower, upper=indexed_upper),)
end
