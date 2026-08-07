
"""
    RW2 <: ComponentModel

v1.9.1 (2026-08-07) - The Second-Order Random Walk (RW2) model.
This model assumes that the current value of a latent field is twice the previous
value minus the value before that, plus a random innovation. It is an intrinsic
Gaussian Markov Random Field (GMRF) with a rank deficiency of 2, implying two
sum-to-zero constraints for identifiability. It produces a smoother field than RW1.

# Fields
- `sigma::UnivariateDistribution`: The prior for the standard deviation of the innovations.
"""
struct RW2 <: ComponentModel
    sigma::UnivariateDistribution
end

COMPONENT_CONSTRUCTORS[:rw2] = (p, params) -> RW2(p.sigma)

# helper to map to classes of methods (data structures), :any mean it can be used in many approaches
MODEL_TO_STRUCTURE_MAP[::rw2] = :temporal


# --- Implementation of the Explicit Interface for RW2 ---

"""
    get_datastructures!(m_type::Type{RW2}, M::Dict, mod_data::Dict)::Bool

v1.9.1 (2026-08-07) - Data-dependent setup for the RW2 component.
This method establishes the temporal context for the model, similar to AR1. It identifies
the time variable, creates discrete time indices (`t_idx`), and determines the total
number of time steps (`t_N`), adding this information to the main model configuration `M`.
"""
function get_datastructures!(m_type::Type{RW2}, M::Dict, mod_data::Dict)::Bool
    data = M[:data]
    params = mod_data[:params]
    variables = mod_data[:variables]

    if isempty(variables)
        error("The RW2 model requires a time index variable, e.g., `random(year, model=:rw2)`.")
    end

    time_var_sym = Symbol(variables[1])
    if !hasproperty(data, time_var_sym)
        error("Time index variable ':$time_var_sym' for RW2 model not found in data.")
    end

    time_opts = Dict(:time_method => get(params, :time_method, "regular"))
    tu_meta = assign_time_units(data[!, time_var_sym]; time_opts...)
    
    M[:t_idx] = tu_meta.idx
    M[:t_N] = tu_meta.N_cat
    M[:t_idx_var] = time_var_sym
    
    return true
end

"""
    get_precomputes(m::RW2, M::NamedTuple, mod_data::Dict)::NamedTuple

v1.9.1 (2026-08-07) - Pre-computes the structure matrix for the RW2 component.
For an RW2 model, the precision matrix template defines the second-order differences.
This function calls the central `build_structure_template` utility to generate this
matrix and its spectral decomposition (`U`, `L`), which are essential for the
AD-safe spectral sampling method.
"""
function get_precomputes(m::RW2, M::NamedTuple, mod_data::Dict)::NamedTuple
    t_N = get(M, :t_N, 0)
    if t_N == 0
        @warn "Could not determine number of time steps for RW2 component '$(mod_data[:key])'. The component will have no effect."
    end
    template = build_structure_template(:rw2, t_N)
    return (Q_template=template.matrix, U=template.U, L=template.L, scaling_factor=template.scaling_factor)
end

"""
    get_priors(m::RW2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

v1.9.1 (2026-08-07) - Generates the Turing code for the RW2 component's priors.
This function creates the code strings for sampling the `sigma` hyperparameter and the
standard normal innovations (`struct`) for the latent field.
"""
function get_priors(m::RW2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = size(spec.Q_template, 1)
    is_multivariate = (arch == "multivariate")
    is_shared = get(spec.params, :shared, false)
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ $(_distribution_to_string(m.sigma))")
    end
    push!(priors_acc, "$(v.struct) ~ MvNormal(zeros(T, $(n_latent)), I)")
    return join(priors_acc, "\n    ")
end

"""
    get_updates(m::RW2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

v1.9.1 (2026-08-07) - Generates the Turing code for the RW2 component's update logic using spectral sampling.
This function implements the AD-safe spectral sampling method for the RW2 model. It generates
code that constructs the latent field by transforming standard normal innovations using the
pre-computed spectral decomposition (`U`, `L`) of the RW2 precision matrix, avoiding any
`cholesky` calls on parameter-dependent matrices. It also enforces the sum-to-zero constraints.
"""
function get_updates(m::RW2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx"
    n_latent = size(spec.Q_template, 1)

    return """
    # --- RW2 Spectral Assembly: $(spec.key) ---
    # This block constructs the RW2 temporal effect using a non-centered parameterization
    # based on the spectral decomposition of the RW2 precision matrix. This method is
    # efficient and safe for automatic differentiation (AD).

    # 1. Construct the diagonal of the spectral transformation matrix D.
    #    The standard deviation is sigma / sqrt(L_j).
    local diag_D = $(v.sigma) ./ sqrt.(spec.L .+ T(noise))

    # 2. Enforce sum-to-zero constraints by zeroing out components corresponding to the null space.
    #    RW2 has a rank deficiency of 2, so the first two eigenvalues are zero.
    diag_D[1] = zero(T)
    diag_D[2] = zero(T)

    # 3. Apply the spectral transformation: latent = U * D * z
    $(v.latent) = spec.U * (diag_D .* $(v.struct))

    # 4. Add the final effect to the linear predictor.
    $(eta_target) .+= view($(v.latent), M.$(index_var))
    """
end

"""
    get_effects(m::RW2, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

v1.9.1 (2026-08-07) - Reconstructs the RW2 component's effect from posterior samples.
This function extracts the posterior samples for `sigma` and `struct` from the MCMC
chain. It then reconstructs the full posterior distribution of the latent RW2 field by
re-running the spectral transformation for each posterior sample.
"""
function get_effects(m::RW2, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    structured_effects = []
    is_multivariate = outcomes_N > 1
    is_shared = get(spec.params, :shared, false)
    n_latent = size(spec.Q_template, 1)

    U = spec.hyper.U
    L_eig = spec.hyper.L

    for k in 1:outcomes_N
        outcome_idx = is_multivariate ? k : nothing
        v = generate_full_variable_names(spec, M.model_arch, outcome_idx)
        
        sigma_var_name = (is_multivariate && is_shared) ? string(generate_full_variable_names(spec, M.model_arch, nothing).sigma) : string(v.sigma)
        struct_var_name = (is_multivariate && is_shared) ? string(generate_full_variable_names(spec, M.model_arch, nothing).struct) : string(v.struct)

        sigma_samples = get_params_vector(chain, sigma_var_name, 1)
        struct_samples = get_params_vector(chain, struct_var_name, n_latent)
        
        T = eltype(chain.value)
        effect_k = Matrix{T}(undef, n_latent, n_samples)

        diag_D = one(T) ./ sqrt.(L_eig .+ T(1e-9))
        diag_D[1] = zero(T) # Enforce sum-to-zero
        diag_D[2] = zero(T) # Enforce sum-to-zero

        for s in 1:n_samples
            sigma_s = sigma_samples[s, 1]
            struct_s = struct_samples[s, :]
            effect_k[:, s] = U * (diag_D .* struct_s) .* sigma_s
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end