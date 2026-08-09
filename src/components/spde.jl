"""
    SPDE <: ComponentModel

A component model for a spatial field based on the Stochastic Partial Differential
Equation (SPDE) approach. This method provides a direct link between a continuous
Gaussian Process with a Matérn covariance function and a discrete Gaussian Markov
Random Field (GMRF), enabling scalable and principled spatial modeling.

# Version
v1.0.0 (2026-08-08)

# Mathematical Summary
The SPDE approach models a Gaussian Field \$u(s)\$ as the solution to the SPDE:
\$(\\kappa^2 - \\Delta)^{\\alpha/2} u(s) = \\mathcal{W}(s)\$
where:
- \$\\Delta\$ is the Laplacian operator.
- \$\\kappa > 0\$ controls the spatial range of the process.
- \$\\alpha\$ controls the smoothness of the process.
- \$\\mathcal{W}(s)\$ is Gaussian white noise.

For a discrete spatial domain represented by a graph, the Laplacian \$\\Delta\$ is
approximated by the graph Laplacian \$\\mathbf{Q}_{ICAR} = D - W\$. For the common case
where \$\\alpha = 2\$ (which corresponds to a Matérn field with smoothness \$\\nu=1\$),
the precision matrix \$\\mathbf{Q}\$ of the latent field \$\\boldsymbol{\\phi}\$ is given by:
\$\\mathbf{Q} = (\\kappa^2 \\mathbf{I} + \\mathbf{Q}_{ICAR})^T (\\kappa^2 \\mathbf{I} + \\mathbf{Q}_{ICAR})\$
The model then samples the latent field from \$\\boldsymbol{\\phi} \\sim \\mathcal{N}(0, (\\sigma^2 \\mathbf{Q})^{-1})\$.

# Assumptions
- The provided adjacency matrix `W` represents a connected graph that is a reasonable
  discretization of the continuous spatial domain.
- The smoothness parameter \$\\alpha\$ is fixed (typically at 2).

# Best Use Case
Modeling continuous spatial processes on regular or irregular lattices where a Matérn
covariance is desired. It is a computationally efficient alternative to a full GP,
as it leverages a sparse precision matrix.

# Key References
- Lindgren, F., Rue, H., & Lindström, J. (2011). An explicit link between
  Gaussian fields and Gaussian Markov random fields: The SPDE approach. *Journal
  of the Royal Statistical Society: Series B (Statistical Methodology)*, 73(4),
  423-498.
- Wikipedia: Matérn covariance function

# Fields
- `sigma::Distribution`: The prior for the marginal standard deviation of the SPDE effect.
- `kappa::Union{Distribution, Vector{<:Distribution}}`: The prior for the `kappa`
  parameter, which controls the spatial range. Can be a single distribution or a
  vector for anisotropic effects.
"""
struct SPDE <: ComponentModel
    sigma::Distribution
    kappa::Union{Distribution, Vector{<:Distribution}}
end

# Add to the central component constructor registry.
COMPONENT_TYPE_REGISTRY[:spde] = SPDE
COMPONENT_CONSTRUCTORS[:spde] = (p, params) -> SPDE(p.sigma, p.kappa)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[:spde] = :spatial

"""
    get_datastructures!(m_type::Type{<:SPDE}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `SPDE` component. It ensures that an
adjacency matrix `W` is provided and sets up the spatial context (`s_idx`, `s_N`).
"""
function get_datastructures!(m_type::Type{<:SPDE}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    variables = mod_data[:variables]

    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr || w_val isa Symbol
            calling_mod = get(M, :calling_module, Main)
            try
                M[:W] = Core.eval(calling_mod, w_val)
            catch e
                error("Could not evaluate `W` argument `$(w_val)` for SPDE. Error: $e")
            end
        else
            M[:W] = w_val
        end
    end

    if !haskey(M, :W) || !isa(M[:W], AbstractMatrix) || isempty(M[:W])
        error("SPDE model requires a valid, non-empty adjacency matrix `W`.")
    end

    M[:s_N] = size(M[:W], 1)

    if isempty(variables)
        M[:s_idx] = collect(1:M[:s_N])
        @warn "Spatial index not provided for SPDE. Assuming `s_idx = 1:s_N`."
    else
        s_var_sym = Symbol(variables[1])
        if !hasproperty(M[:data], s_var_sym)
            error("Spatial index ':$s_var_sym' for SPDE not found in data.")
        end
        M[:s_idx] = M[:data][!, s_var_sym]
    end

    return true
end

"""
    get_precomputes(m::SPDE, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the graph Laplacian `Q_template` and its spectral decomposition.
"""
function get_precomputes(m::SPDE, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.s_N
    W = M.W

    W_sym = sparse((W + W') .> 0)
    D = spdiagm(0 => vec(sum(W_sym, dims=2)))
    Q_template = D - W_sym

    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values
    scaling_factor = _compute_scaling_factor(L, 1)
    
    Q_template_scaled = Q_template ./ scaling_factor
    L_scaled = L ./ scaling_factor

    return (
        Q_template=Q_template_scaled,
        scaling_factor=scaling_factor,
        U=U,
        L=L_scaled,
        n_latent=n
    )
end

"""
    get_priors(m::SPDE, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `sigma`, `kappa`, and the `raw` innovations.
"""
function get_priors(
    m::SPDE, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = String[]
    push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")

    if m.kappa isa Vector
        kappa_priors_str = join([_distribution_to_string(p) for p in m.kappa], ", ")
        push!(priors, "$(p_names.kappa) ~ Product([$(kappa_priors_str)])")
    else
        kappa_prior_str = _distribution_to_string(m.kappa)
        push!(priors, "$(p_names.kappa) ~ $(kappa_prior_str)")
    end
    
    push!(
        priors,
        "$(p_names.raw) ~ MvNormal(zeros(spec.precomputes.n_latent), I)"
    )

    return join(priors, "\n    ")
end

"""
    get_updates(m::SPDE, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to construct the SPDE precision matrix, sample the latent field,
and add it to the linear predictor `eta`.
"""
function get_updates(
    m::SPDE, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- SPDE Component: $(spec.key) ---
        let
            local Q_laplacian = spec_registry[:$(spec.key)].precomputes.Q_template
            local n_latent = spec_registry[:$(spec.key)].precomputes.n_latent
            
            local kappa_val = $(p_names.kappa)
            local Q_kappa_term
            if kappa_val isa AbstractVector
                Q_kappa_term = Diagonal(kappa_val.^2)
            else
                Q_kappa_term = kappa_val^2 * I(n_latent)
            end

            local L_operator = Q_kappa_term + Q_laplacian
            local Q_final = Symmetric(L_operator' * L_operator)
            
            local F = cholesky(Matrix(Q_final) + M.noise * I)
            
            local $(p_names.latent) = $(p_names.sigma) .* (F.L' \\ $(p_names.raw))
            
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """
end

"""
    get_effects(m::SPDE, chain, M::NamedTuple, ...)

Reconstructs the `SPDE` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(
    m::SPDE, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    n_latent = spec.hyper.n_latent
    noise = M.noise
    Q_laplacian = spec.hyper.Q_template
    s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        kappa_samples = get_params_vector(
            chain, string(p_names.kappa), m.kappa isa Vector ? length(m.kappa) : 1
        )
        raw_samples = get_params_vector(chain, string(p_names.raw), n_latent)

        effect_k = zeros(Float64, N_total, n_samples)

        for i in 1:n_samples
            current_sigma = sigma_samples[i]
            current_kappa = kappa_samples[i, :]
            current_raw = raw_samples[i, :]
            
            local Q_kappa_term
            if m.kappa isa Vector
                Q_kappa_term = Diagonal(current_kappa.^2)
            else
                Q_kappa_term = current_kappa[1]^2 * I(n_latent)
            end
            
            L_operator = Q_kappa_term + Q_laplacian
            Q_final = Symmetric(L_operator' * L_operator)
            
            F = cholesky(Matrix(Q_final) + noise * I)
            
            latent_field = current_sigma .* (F.L' \ current_raw)
            effect_k[:, i] = view(latent_field, s_idx_full)
        end
        push!(structured_effects, effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
