# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    NetworkFlow <: ComponentModel

A component model for network flow processes, where dependencies are directional.
This is a variation of a Simultaneous Autoregressive (SAR) model that can account
for upstream, downstream, or bidirectional influence on a graph.

# Fields
- `sigma::Distribution`: The prior for the standard deviation of the innovations.
- `rho::Distribution`: The prior for the spatial/network autocorrelation parameter `rho`.
- `flow_direction::Symbol`: The direction of influence, one of `:downstream`, `:upstream`, or `:bidirectional`.
"""
struct NetworkFlow <: ComponentModel
    sigma::Distribution
    rho::Distribution
    flow_direction::Symbol
end

# Add to the central component constructor registry.
COMPONENT_CONSTRUCTORS[:networkflow] = (p, params) -> NetworkFlow(p.sigma, p.rho, get(params, :flow_direction, :bidirectional))

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[NetworkFlow] = :spatial

"""
    get_datastructures!(m_type::Type{<:NetworkFlow}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `NetworkFlow` component.
It ensures that an adjacency matrix `W` is provided and sets up the spatial context
(`s_idx`, `s_N`) in the main model configuration `M`.
"""
function get_datastructures!(m_type::Type{<:NetworkFlow}, M::Dict, mod_data::Dict)::Bool
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
                error("Could not evaluate `W` argument `$(w_val)` for NetworkFlow component. Error: $e")
            end
        else
            M[:W] = w_val
        end
    end

    if !haskey(M, :W)
        error("NetworkFlow model requires an adjacency matrix `W` to be provided.")
    end

    if !isa(M[:W], AbstractMatrix) || isempty(M[:W])
        error("Provided `W` for NetworkFlow model is not a valid non-empty matrix.")
    end

    M[:s_N] = size(M[:W], 1)

    if isempty(variables)
        M[:s_idx] = collect(1:M[:s_N])
        @warn "Spatial index variable not provided for NetworkFlow. Assuming `s_idx = 1:s_N`."
    else
        s_var_sym = Symbol(variables[1])
        if !hasproperty(M[:data], s_var_sym)
            error("Spatial index variable ':$s_var_sym' for NetworkFlow model not found in data.")
        end
        M[:s_idx] = M[:data][!, s_var_sym]
    end

    return true
end

"""
    get_precomputes(m::NetworkFlow, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `NetworkFlow` component.
The `Q_template` is the raw adjacency matrix `W`. The full precision matrix is
constructed dynamically within the model based on the `flow_direction`.
"""
function get_precomputes(m::NetworkFlow, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.s_N
    W = sparse(M.W) # Ensure W is sparse

    # For NetworkFlow, the Q_template is the adjacency matrix itself.
    # The full precision matrix is constructed dynamically based on flow_direction.
    return (Q_template=W, n_latent=n)
end

"""
    get_priors(m::NetworkFlow, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `NetworkFlow` component's priors.
It defines the priors for `rho`, `sigma`, and the latent field `raw`.
"""
function get_priors(m::NetworkFlow, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    rho_prior_str = _distribution_to_string(m.rho)
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    return """
        $(p_names.rho) ~ NamedDist($(rho_prior_str), :$(p_names.rho))
        $(p_names.sigma) ~ NamedDist($(sigma_prior_str), :$(p_names.sigma))
        $(p_names.raw) ~ NamedDist(MvNormal(zeros(T, spec_registry[:$(spec.key)].precomputes.n_latent), I), :$(p_names.raw))
    """
end

"""
    get_updates(m::NetworkFlow, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `NetworkFlow` component's effect.
It dynamically builds the precision matrix based on `flow_direction` and uses Cholesky
decomposition for AD-compatible sampling.
"""
function get_updates(m::NetworkFlow, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    # Logic to construct the operator based on flow direction
    operator_logic = if m.flow_direction == :downstream
        "I(n_latent) - $(p_names.rho) * W_net"
    elseif m.flow_direction == :upstream
        "I(n_latent) - $(p_names.rho) * W_net'"
    else # :bidirectional
        "I(n_latent) - $(p_names.rho) * sparse((W_net + W_net') .> 0)"
    end

    return """
        # --- NetworkFlow Component: $(spec.key) ---
        local W_net = spec_registry[:$(spec.key)].precomputes.Q_template
        local n_latent = spec_registry[:$(spec.key)].precomputes.n_latent
        
        # Construct the operator (I - rho*W_effective)
        local L_op = $(operator_logic)
        
        # Form the precision matrix Q_final = (L_op' * L_op) / sigma^2
        local Q_final = Symmetric((L_op' * L_op) / ($(p_names.sigma)^2) + M.noise * I)
        
        # Perform Cholesky decomposition for non-centered parameterization.
        local F = cholesky(Matrix(Q_final))
        
        # Sample latent field: latent = inv(L') * raw
        local $(p_names.latent) = F.L' \\ $(p_names.raw)
        
        $(eta_target) .+= $(p_names.latent)[M.s_idx]
    """
end

"""
    get_effects(m::NetworkFlow, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `NetworkFlow` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::NetworkFlow, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    rho_samples = get(chain, p_names.rho)
    sigma_samples = get(chain, p_names.sigma)
    raw_samples = get(chain, p_names.raw)

    n_latent = spec.precomputes.n_latent
    noise = M.noise
    W_net = spec.precomputes.Q_template

    idx_to_use = isnothing(PS) ? M.s_idx : PS.s_idx
    
    reconstructed_effects = zeros(n_samples, n_latent)

    for i in 1:n_samples
        current_rho = rho_samples[i]
        current_sigma = sigma_samples[i]
        current_raw = raw_samples[i, :]
        
        # Reconstruct the operator and precision matrix for the current sample
        L_op = if m.flow_direction == :downstream
            I(n_latent) - current_rho * W_net
        elseif m.flow_direction == :upstream
            I(n_latent) - current_rho * W_net'
        else # :bidirectional
            I(n_latent) - current_rho * sparse((W_net + W_net') .> 0)
        end
        
        Q_final = Symmetric((L_op' * L_op) / (current_sigma^2) + noise * I)
        
        F = cholesky(Matrix(Q_final))
        reconstructed_effects[i, :] = F.L' \ current_raw
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    indexed_mean = mean_effect[idx_to_use]
    indexed_lower = lower_ci[idx_to_use]
    indexed_upper = upper_ci[idx_to_use]

    return (structured=(mean=indexed_mean, lower=indexed_lower, upper=indexed_upper),)
end
