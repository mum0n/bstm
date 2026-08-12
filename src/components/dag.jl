"""
    DAG <: ComponentModel

A component for a Directed Acyclic Graph (DAG) structure, also known as a
unilateral autoregressive model. This model is useful for capturing causal or
directional dependencies between spatial or other units.

# Version
v1.0.2 (2026-08-11)

# Mathematical Summary
The DAG model defines a recursive relationship for the latent field \$\\phi\$:
\$\\phi_i = \\rho \\sum_{j \\in pa(i)} W_{ij} \\phi_j + \\epsilon_i\$
where \$pa(i)\$ are the parents of node \$i\$, \$W\$ is an adjacency matrix representing
the directed graph, \$\\rho\$ is a dependence parameter, and \$\\epsilon_i\$ are
independent innovations, \$\\epsilon_i \\sim N(0, \\sigma^2)\$.

This can be written in matrix form as \$(I - \\rho W)\\phi = \\epsilon\$. The corresponding
precision matrix is \$Q = (I - \\rho W)^T (I - \\rho W) / \\sigma^2\$.

# Computational Methods
- `:forward_substitution` (Default, AD-safe): If `W` is strictly triangular (representing a
  valid topological ordering), the latent field can be sampled efficiently using
  forward substitution. This is the recommended, AD-safe method.
- `:precision` (Didactic, AD-safe): Explicitly constructs the dense precision matrix `Q` and
  samples the field from the corresponding `MvNormal`. This is less efficient but
  conceptually clear and also AD-safe.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `s_idx`).
  - An adjacency matrix `W` passed as a keyword argument to `@bstm`.
- **Optional (in `random()` call)**:
  - `rho`: `UnivariateDistribution`, prior for the autoregressive parameter. Default: `Normal(0, 0.5)`.
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the innovations. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:forward_substitution` or `:precision`). Default: `:forward_substitution`.

# Outputs (Parameter Names)
- `rho_<key>`: The autoregressive parameter.
- `sigma_<key>`: The standard deviation of the innovations.
- `innov_<key>`: The standard normal innovations for the latent field (for `:forward_substitution`).
- `raw_<key>`: The standard normal innovations for the latent field (for `:precision`).
- `latent_<key>`: The reconstructed latent DAG effect.
"""
struct DAG <: ComponentModel
    rho::Distribution
    sigma::Distribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:dag] = DAG
COMPONENT_CONSTRUCTORS[:dag] = (p, params) -> DAG(
    get(p, :rho, Normal(0, 0.5)), # Default prior for rho
    get(p, :sigma, Exponential(1.0)), # Default prior for sigma
    get(params, :method, :forward_substitution)
)

MODEL_TO_STRUCTURE_MAP[:dag] = :spatial

"""
    get_datastructures!(m_type::Type{<:DAG}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `DAG` component. It ensures that an
adjacency matrix `W` is provided, warns if it is not strictly triangular (a
requirement for the forward-substitution algorithm), and sets up the spatial
context (`s_idx`, `s_N`).

# Assumptions
- The adjacency matrix `W` should ideally be strictly lower or upper triangular to
  represent a valid topological ordering for the forward substitution algorithm.
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

    if !haskey(M, :W) || isnothing(M[:W])
        error("DAG model requires an adjacency matrix `W` to be provided.")
    end

    W = M[:W]
    if !istril(W) && !istriu(W)
        @warn "The adjacency matrix `W` for the DAG component is not strictly " *
              "triangular. The model assumes a causal ordering. Results may be " *
              "incorrect if the graph contains cycles. Consider reordering nodes " *
              "or using a different spatial model."
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
    W = sparse(M.W) # Ensure W is sparse for efficiency

    return (Q_template=W, n_latent=n)
end

"""
    get_priors(m::DAG, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `rho`, `sigma`, and the raw innovations. The name of the
innovation parameter (`innov` or `raw`) depends on the chosen method.

# Inputs
- `m::DAG`: The `DAG` component instance.
- `spec::NamedTuple`: The component's specification, including its `key` and `hyper` parameters.
- `arch::String`: The model architecture (`"univariate"` or `"multivariate"`).
- `outcome_idx::Union{Int, Nothing}`: The index of the outcome for multivariate models.
- `M::NamedTuple`: The main model configuration.

# Outputs
- `String`: Turing code for the priors of `rho`, `sigma`, and the innovation parameter.
"""
function get_priors(
    m::DAG, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = [
        "$(p_names.rho) ~ $(_distribution_to_string(m.rho))",
        "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))"
    ]
    
    # The innovation parameter has a different conceptual role depending on the method.
    # :forward_substitution -> innovations in the state-space equation.
    # :precision -> standard normal noise for non-centered parameterization.
    innov_param = m.method == :forward_substitution ? p_names.innov : p_names.raw
    push!(priors, "$(innov_param) ~ MvNormal(zeros(T, $(spec.hyper.n_latent)), I)")
    
    return join(priors, "\n    ")
end

"""
    get_updates(m::DAG, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates Turing code to construct the `DAG` effect, dispatching on the chosen method.

# Inputs
- `m::DAG`: The `DAG` component instance.
- `spec::NamedTuple`: The component's specification, including its `key` and `hyper` parameters.
- `arch::String`: The model architecture (`"univariate"` or `"multivariate"`).
- `outcome_idx::Union{Int, Nothing}`: The index of the outcome for multivariate models.
- `M::NamedTuple`: The main model configuration.

# Outputs
- `String`: Turing code for constructing the latent DAG effect and adding it to the linear predictor.
"""
function get_updates(
    m::DAG, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    forward_sub_code = """
        # --- DAG Component (Forward Substitution): $(spec.key) ---
        let
            local W_dag = spec_registry[:$(spec.key)].hyper.Q_template
            local innovations = $(p_names.innov)
            local rho_val = $(p_names.rho)
            local sigma_val = $(p_names.sigma)
            local n_latent = spec_registry[:$(spec.key)].hyper.n_latent

            local T_num = promote_type(typeof(rho_val), eltype(innovations))
            local $(p_names.latent) = zeros(T_num, n_latent)

            # Iterate through nodes in topological order (assumed by W_dag structure)
            for i in 1:n_latent
                local parent_effect = zero(T_num) # Initialize with zero of correct type
                # Sum contributions from parents (non-zero elements in the row of W_dag)
                for j_ptr in nzrange(W_dag, i)
                    parent_idx = W_dag.rowval[j_ptr]
                    parent_effect += W_dag.nzval[j_ptr] * $(p_names.latent)[parent_idx]
                end
                # Recursive relationship: phi_i = rho * sum(W_ij * phi_j) + epsilon_i
                $(p_names.latent)[i] = rho_val * parent_effect + innovations[i]
            end
            $(p_names.latent) .*= sigma_val # Scale the entire field by sigma
            
            $(eta_target) .+= view($(p_names.latent), M.s_idx) # Apply to linear predictor
        end
    """

    precision_code = """
        # --- DAG Component (Precision Matrix): $(spec.key) ---
        # This is a didactic alternative that is less efficient than forward substitution.
        let
            local W_dag = spec_registry[:$(spec.key)].hyper.Q_template
            local L_op = I - $(p_names.rho) * W_dag # Operator (I - rho * W)
            local Q = L_op' * L_op # Precision matrix Q = (I - rho * W)' * (I - rho * W)
            
            # Use dense Cholesky for AD-safety, as sparse Cholesky might not support Dual numbers.
            local F = cholesky(Symmetric(Matrix(Q) + M.noise * I))
            
            # Non-centered parameterization: latent = sigma * L_inv * raw_innovations
            $(p_names.latent) = $(p_names.sigma) .* (F.U \\ $(p_names.raw))
            
            $(eta_target) .+= view($(p_names.latent), M.s_idx) # Apply to linear predictor
        end
    """

    if m.method == :forward_substitution
        return forward_sub_code
    elseif m.method == :precision
        return precision_code
    else
        error("Unsupported method '$(m.method)' for DAG component. Supported methods are `:forward_substitution` and `:precision`.")
    end
end

"""
    get_effects(m::DAG, chain, M::NamedTuple, n_samples, outcomes_N, spec, PS, N_total)::NamedTuple

Reconstructs the `DAG` component's effect from posterior samples, dispatching
on the method used during sampling.

# Inputs
- `m::DAG`: The `DAG` component instance.
- `chain`: The MCMC chain object.
- `M::NamedTuple`: The main model configuration.
- `n_samples::Int`: The number of posterior samples.
- `outcomes_N::Int`: The number of outcome variables.
- `spec::NamedTuple`: The component's specification, including its `key` and `hyper` parameters.
- `PS::Union{NamedTuple, Nothing}`: The prediction set configuration object, if applicable.
- `N_total::Int`: The total number of observations (training + prediction).

# Outputs
- `NamedTuple`: A NamedTuple with `structured` and `noisy` effects, each a vector of matrices
  `[N_total x n_samples]`.
"""
function get_effects(
    m::DAG, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))
    n_latent = spec.hyper.n_latent
    W_dag = spec.hyper.Q_template
    s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)
    
    for k in 1:outcomes_N
        rho_name = _find_parameter(p_names_vec, string(spec.key), "rho", k, is_multivariate_model)
        sigma_name = _find_parameter(p_names_vec, string(spec.key), "sigma", k, is_multivariate_model)
        
        # Determine which innovation parameter name to look for based on the method
        innov_param_name = m.method == :forward_substitution ? "innov" : "raw"
        innov_or_raw_name = _find_parameter(p_names_vec, string(spec.key), innov_param_name, k, is_multivariate_model)

        if isempty(rho_name) || isempty(sigma_name) || isempty(innov_or_raw_name)
            @warn "Parameters for DAG component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        rho_samples = get_params_vector(chain, rho_name, 1)[:, 1]
        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        innov_or_raw_samples = get_params_vector(chain, innov_or_raw_name, n_latent)
        
        effect_k = zeros(Float64, N_total, n_samples)

        if m.method == :forward_substitution
            for i in 1:n_samples
                latent_field_i = zeros(Float64, n_latent)
                for j in 1:n_latent
                    parent_effect = 0.0
                    for j_ptr in nzrange(W_dag, j)
                        parent_idx = W_dag.rowval[j_ptr]
                        parent_effect += W_dag.nzval[j_ptr] * latent_field_i[parent_idx]
                    end
                    latent_field_i[j] = rho_samples[i] * parent_effect + innov_or_raw_samples[i, j]
                end
                latent_field_i .*= sigma_samples[i]
                effect_k[:, i] = view(latent_field_i, s_idx_full)
            end
        else # :precision
            for i in 1:n_samples
                L_op = I - rho_samples[i] * W_dag
                Q = L_op' * L_op
                F = cholesky(Symmetric(Matrix(Q) + M.noise * I))
                latent_field_i = sigma_samples[i] .* (F.U \ innov_or_raw_samples[i, :])
                effect_k[:, i] = view(latent_field_i, s_idx_full)
            end
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
