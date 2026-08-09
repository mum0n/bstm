"""
    DAG <: ComponentModel

A component for a Directed Acyclic Graph (DAG) structure, also known as a
unilateral autoregressive model. This model is useful for capturing causal or
directional dependencies between spatial or other units. The effect at each node
is a linear combination of its parents' effects plus an independent innovation.

# Version
v1.0.0 (2026-08-08)

# Mathematical Summary
The DAG model defines a recursive relationship for the latent field \$\\phi\$:
\$\\phi_i = \\rho \\sum_{j \\in pa(i)} W_{ij} \\phi_j + \\epsilon_i\$
where \$pa(i)\$ are the parents of node \$i\$, \$W\$ is an adjacency matrix representing
the directed graph, \$\\rho\$ is a dependence parameter, and \$\\epsilon_i\$ are
independent innovations, \$\\epsilon_i \\sim N(0, \\sigma^2)\$.

This can be written in matrix form as \$\\phi = \\rho W \\phi + \\epsilon\$, which implies
\$(I - \\rho W)\\phi = \\epsilon\$. The corresponding precision matrix is
\$Q = (I - \\rho W)^T (I - \\rho W) / \\sigma^2\$.

A key feature of this model is that if \$W\$ is strictly lower (or upper)
triangular, representing a valid topological ordering of the DAG, the latent field
\$\\phi\$ can be sampled efficiently using forward substitution, avoiding the need to
form or invert the dense precision matrix. This makes it highly scalable.

# Assumptions
- The adjacency matrix `W` represents a true DAG (i.e., it is acyclic). For the
  forward substitution algorithm to be valid, `W` must be strictly lower or upper
  triangular, which implies a topological ordering of the nodes.
- The dependencies are linear and additive.

# Best Use Case
Modeling processes with known directional influence, such as river networks (where
flow is downstream), gene regulatory networks, or certain causal inference problems.
It is a computationally efficient alternative to dense GPs or symmetric GMRFs when a
causal or directional ordering of nodes is known a priori.

# Key References
- **Bayesian Networks**: Koller, D., & Friedman, N. (2009). *Probabilistic
  Graphical Models: Principles and Techniques*. MIT Press.
- **Spatial Statistics**: Vecchia, A. V. (1988). Estimation and model
  identification for continuous spatial processes. *Journal of the Royal
  Statistical Society: Series B (Methodological)*, 50(2), 297-312. (For related
  ideas on conditional approximations in spatial statistics).
- **Wikipedia**: Directed acyclic graph

# Fields
- `rho::Distribution`: The prior for the autoregressive parameter `rho`,
  controlling the strength of parental influence.
- `sigma::Distribution`: The prior for the standard deviation of the innovations at
  each node.
"""
struct DAG <: ComponentModel
    rho::Distribution
    sigma::Distribution
end

COMPONENT_TYPE_REGISTRY[:dag] = DAG
COMPONENT_CONSTRUCTORS[:dag] = (p, params) -> DAG(p.rho, p.sigma)
MODEL_TO_STRUCTURE_MAP[:dag] = :spatial

"""
    get_datastructures!(m_type::Type{<:DAG}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `DAG` component. It ensures that an
adjacency matrix `W` is provided, warns if it is not strictly triangular (a
requirement for the forward-substitution algorithm), and sets up the spatial
context (`s_idx`, `s_N`).

# Assumptions
- The adjacency matrix `W` must be strictly lower or upper triangular to represent
  a valid topological ordering for the forward substitution algorithm.
"""
function get_datastructures!(m_type::Type{<:DAG}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    variables = mod_data[:variables]

    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr || w_val isa Symbol
            calling_mod = get(M, :calling_module, Main)
            try
                M[:W] = Core.eval(calling_mod, w_val)
            catch e
                error(
                    "Could not evaluate `W` argument `$(w_val)` for DAG component. " *
                    "Error: $e"
                )
            end
        else
            M[:W] = w_val
        end
    end

    if !haskey(M, :W)
        error("DAG model requires an adjacency matrix `W` to be provided.")
    end

    W = M[:W]
    if !istril(W) && !istriu(W)
        @warn "The adjacency matrix `W` for the DAG component is not strictly " *
              "triangular. The model assumes a causal ordering. Results may be " *
              "incorrect if the graph contains cycles."
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

For the `DAG` component, this function stores the adjacency matrix `W` as the
`Q_template`. The model's forward substitution logic relies on this matrix
representing the causal graph structure.
"""
function get_precomputes(m::DAG, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.s_N
    W = sparse(M.W) # Ensure W is sparse

    return (Q_template=W, n_latent=n)
end

"""
    get_priors(m::DAG, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the priors on the `DAG` component's parameters.
This includes priors for `rho`, `sigma`, and the independent innovations for each
node.
"""
function get_priors(
    m::DAG, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    rho_prior_str = _distribution_to_string(m.rho)
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    return """
        $(p_names.rho) ~ $(rho_prior_str)
        $(p_names.sigma) ~ $(sigma_prior_str)
        $(p_names.innov) ~ MvNormal(
            zeros(T, spec_registry[:$(spec.key)].hyper.n_latent), I
        )
    """
end

"""
    get_updates(m::DAG, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code to construct the `DAG` component's effect using forward
substitution and adds the result to the linear predictor `eta`.
"""
function get_updates(
    m::DAG, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- DAG Component: $(spec.key) ---
        let
            local W_dag = spec_registry[:$(spec.key)].hyper.Q_template
            local innovations = $(p_names.innov)
            local rho_val = $(p_names.rho)
            local sigma_val = $(p_names.sigma)
            local n_latent = spec_registry[:$(spec.key)].hyper.n_latent

            local T_num = promote_type(typeof(rho_val), eltype(innovations))
            local $(p_names.latent) = zeros(T_num, n_latent)

            # Assumes W_dag is lower triangular, representing a valid DAG ordering.
            for i in 1:n_latent
                local parent_effect = zero(T_num)
                # Efficiently iterate over non-zero elements in the i-th column.
                for j_ptr in nzrange(W_dag, i)
                    parent_idx = W_dag.rowval[j_ptr]
                    parent_effect += W_dag.nzval[j_ptr] * $(p_names.latent)[parent_idx]
                end
                $(p_names.latent)[i] = rho_val * parent_effect + innovations[i]
            end
            $(p_names.latent) .*= sigma_val
            
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """
end

"""
    get_effects(m::DAG, chain, M::NamedTuple, ...)::NamedTuple

Reconstructs the `DAG` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(
    m::DAG, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        rho_samples = get_params_vector(chain, string(p_names.rho), 1)[:, 1]
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        innov_samples = get_params_vector(
            chain, string(p_names.innov), spec.hyper.n_latent
        )

        n_latent = spec.hyper.n_latent
        W_dag = spec.hyper.Q_template
        s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)
        
        effect_k = zeros(Float64, N_total, n_samples)

        for i in 1:n_samples
            latent_field_i = zeros(Float64, n_latent)
            for j in 1:n_latent
                parent_effect = 0.0
                for j_ptr in nzrange(W_dag, j)
                    parent_idx = W_dag.rowval[j_ptr]
                    parent_effect += W_dag.nzval[j_ptr] * latent_field_i[parent_idx]
                end
                latent_field_i[j] = rho_samples[i] * parent_effect + innov_samples[i, j]
            end
            latent_field_i .*= sigma_samples[i]
            effect_k[:, i] = view(latent_field_i, s_idx_full)
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
