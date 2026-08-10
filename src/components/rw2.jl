"""
    RW2 <: ComponentModel

A component for a second-order random walk (RW2) model. This model assumes that the
second differences of a latent temporal field follow a random innovation. It is an
intrinsic Gaussian Markov Random Field (GMRF) with a rank deficiency of 2, implying
two sum-to-zero constraints for identifiability. It produces a smoother field than
an RW1 model.

# Version
v1.9.3 (2026-08-10)

# Mathematical Summary
The RW2 model defines a latent temporal field \$\\phi\$ where the value at time \$t\$ is
a linear extrapolation from its two immediate predecessors, plus a random innovation:
\$\\phi_t | \\phi_{t-1}, \\phi_{t-2} \\sim \\mathcal{N}(2\\phi_{t-1} - \\phi_{t-2}, \\sigma^2)\$
This can be written as \$\\phi_t - 2\\phi_{t-1} + \\phi_{t-2} = \\epsilon_t\$, where
\$\\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2)\$.

The joint precision matrix \$\\mathbf{Q}\$ for this process is singular (rank-deficient),
making it an "intrinsic" GMRF. To ensure the model is identifiable from a global
intercept and linear trend, two sum-to-zero constraints are imposed on the latent
field.

# Computational Methods
- `:statespace` (default): The most efficient method, constructing the RW2 process
  via a state-space recurrence relation. AD-safe.
- `:spectral`: An efficient, AD-safe method using spectral decomposition of the
  precision matrix.
- `:cholesky`: An AD-safe didactic alternative using dense Cholesky factorization.
- `:cholesky_sparse`: A non-AD-safe didactic method using sparse Cholesky
  factorization, suitable for gradient-free samplers.

# Fields
- `sigma::UnivariateDistribution`: The prior for the standard deviation of the
  innovations.
- `method::Symbol`: The computational method, one of `:statespace`, `:spectral`,
  `:cholesky`, or `:cholesky_sparse`.
"""
struct RW2 <: ComponentModel
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:rw2] = RW2

COMPONENT_CONSTRUCTORS[:rw2] = (p, params) -> RW2(
    p.sigma, get(params, :method, :statespace)
)


MODEL_TO_STRUCTURE_MAP[:rw2] = :temporal

"""
    get_datastructures!(m_type::Type{<:RW2}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `RW2` component. It establishes the temporal
context by identifying the time variable, creating discrete time indices (`t_idx`),
and determining the total number of time steps (`t_N`).
"""
function get_datastructures!(m_type::Type{<:RW2}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]
    if isempty(variables)
        error("The RW2 model requires a time index variable, e.g., `random(year, model=:rw2)`.")
    end

    time_var_sym = Symbol(variables[1])
    if !hasproperty(M[:data], time_var_sym)
        error("Time index variable ':$time_var_sym' for RW2 model not found in data.")
    end

    time_opts = Dict(:time_method => get(mod_data[:params], :time_method, "regular"))
    tu_meta = assign_time_units(M[:data][!, time_var_sym]; time_opts...)
    
    M[:t_idx] = tu_meta.idx
    M[:t_N] = tu_meta.N_cat
    M[:t_idx_var] = time_var_sym
    
    return true
end

"""
    get_precomputes(m::RW2, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the RW2 precision matrix template, its spectral decomposition (`U`, `L`),
and its dense Cholesky factorization (`cholesky_factor`).
"""
function get_precomputes(m::RW2, M::NamedTuple, mod_data::Dict)::NamedTuple
    t_N = get(M, :t_N, 0)
    if t_N == 0
        @warn "Could not determine number of time steps for RW2 component " *
              "'$(mod_data[:key])'. The component will have no effect."
    end
    template = build_structure_template(:rw2, t_N)
    
    # Pre-compute the dense Cholesky factor for the :cholesky method.
    F = cholesky(Symmetric(Matrix(template.matrix) + M.noise * I))
    
    return (
        Q_template=template.matrix,
        U=template.U,
        L=template.L,
        scaling_factor=template.scaling_factor,
        n_latent=t_N,
        cholesky_factor=F
    )
end

"""
    get_priors(m::RW2, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the `RW2` component's priors. This function creates
the code strings for sampling the `sigma` hyperparameter and the standard normal
innovations (`raw`) for the latent field.
"""
function get_priors(
    m::RW2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    is_multivariate = (arch == "multivariate")
    is_shared = get(spec.params, :shared, false)
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ $(_distribution_to_string(m.sigma))")
    end
    push!(priors_acc, "$(v.raw) ~ MvNormal(zeros($(n_latent)), I)")
    return join(priors_acc, "\n    ")
end

"""
    get_updates(m::RW2, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the `RW2` component's update logic, dispatching on
the chosen `method`.
"""
function get_updates(
    m::RW2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent

    statespace_code = """
        # --- RW2 Component: $(key) (State-Space Method) ---
        let
            innovations = $(p_names.raw)
            latent_field_raw = Vector{T}(undef, $(n_latent))
            if $(n_latent) > 0; latent_field_raw[1] = innovations[1]; end
            if $(n_latent) > 1; latent_field_raw[2] = 2*latent_field_raw[1] + innovations[2]; end
            for t in 3:$(n_latent)
                latent_field_raw[t] = 2*latent_field_raw[t-1] - latent_field_raw[t-2] + innovations[t]
            end
            if $(n_latent) > 0; Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_raw)); end
            $(p_names.latent) = latent_field_raw .* $(p_names.sigma)
            $(eta_target) .+= view($(p_names.latent), M.t_idx)
        end
    """

    spectral_code = """
        # --- RW2 Component: $(key) (Spectral Method) ---
        let
            local U = spec_registry[:$(key)].hyper.U
            local L = spec_registry[:$(key)].hyper.L
            local diag_D = $(p_names.sigma) ./ sqrt.(L .+ M.noise)
            diag_D[1] = 0.0; diag_D[2] = 0.0
            $(p_names.latent) = U * (diag_D .* $(p_names.raw))
            $(eta_target) .+= view($(p_names.latent), M.t_idx)
        end
    """

    cholesky_code = """
        # --- RW2 Component: $(key) (Cholesky Method, AD-Safe) ---
        let
            local F = spec_registry[:$(key)].hyper.cholesky_factor
            local latent_field_raw = F.L' \\ $(p_names.raw)
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_raw))
            $(p_names.latent) = latent_field_raw .* $(p_names.sigma)
            $(eta_target) .+= view($(p_names.latent), M.t_idx)
        end
    """

    cholesky_sparse_code = """
        # --- RW2 Component: $(key) (Sparse Cholesky, Not AD-Safe) ---
        let
            local Q = spec_registry[:$(key)].hyper.Q_template
            local F = cholesky(Symmetric(Q + M.noise * I))
            local latent_field_raw = F.L' \\ $(p_names.raw)
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_raw))
            $(p_names.latent) = latent_field_raw .* $(p_names.sigma)
            $(eta_target) .+= view($(p_names.latent), M.t_idx)
        end
    """

    if m.method == :statespace; return statespace_code;
    elseif m.method == :spectral; return spectral_code;
    elseif m.method == :cholesky; return cholesky_code;
    elseif m.method == :cholesky_sparse; return cholesky_sparse_code;
    else; error("Unsupported method '$(m.method)' for RW2. Use :statespace, :spectral, :cholesky, or :cholesky_sparse."); end
end

"""
    get_effects(m::RW2, chain, M::NamedTuple, ...)

Reconstructs the `RW2` component's effect from posterior samples, applying a
sum-to-zero constraint for identifiability.
"""
function get_effects(
    m::RW2, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    n_latent = spec.hyper.n_latent

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        raw_samples = get_params_vector(chain, string(p_names.raw), n_latent)

        effect_k = zeros(Float64, n_latent, n_samples)

        if m.method == :statespace
            for j in 1:n_samples
                latent_field_raw = Vector{Float64}(undef, n_latent)
                innovations_j = raw_samples[j, :]
                if n_latent > 0; latent_field_raw[1] = innovations_j[1]; end
                if n_latent > 1; latent_field_raw[2] = 2*latent_field_raw[1] + innovations_j[2]; end
                for i in 3:n_latent
                    latent_field_raw[i] = 2*latent_field_raw[i-1] - latent_field_raw[i-2] + innovations_j[i]
                end
                latent_field_centered = latent_field_raw .- mean(latent_field_raw)
                effect_k[:, j] = latent_field_centered .* sigma_samples[j]
            end
        elseif m.method == :spectral
            U = spec.hyper.U
            L = spec.hyper.L
            for j in 1:n_samples
                diag_D = sigma_samples[j] ./ sqrt.(L .+ M.noise)
                diag_D[1] = 0.0; diag_D[2] = 0.0
                effect_k[:, j] = U * (diag_D .* raw_samples[j, :])
            end
        else # :cholesky or :cholesky_sparse
            F = spec.hyper.cholesky_factor
            for j in 1:n_samples
                latent_field_raw = F.L' \ raw_samples[j, :]
                latent_field_centered = latent_field_raw .- mean(latent_field_raw)
                effect_k[:, j] = latent_field_centered .* sigma_samples[j]
            end
        end
        
        t_idx_full = isnothing(PS) ? M.t_idx : vcat(M.t_idx, PS.t_idx)
        indexed_effects = effect_k[t_idx_full, :]
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
