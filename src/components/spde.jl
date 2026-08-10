"""
    SPDE <: ComponentModel

A component model for a spatial field based on the Stochastic Partial Differential
Equation (SPDE) approach. This method provides a direct link between a continuous
Gaussian Process with a Matérn covariance function and a discrete Gaussian Markov
Random Field (GMRF), enabling scalable and principled spatial modeling.

# Version
v1.0.1 (2026-08-10)

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

# Computational Methods
- `:spectral` (default): An efficient, AD-safe method using spectral decomposition.
  Only applicable for isotropic `kappa` priors.
- `:cholesky`: An AD-safe didactic alternative using dense Cholesky factorization.
- `:cholesky_sparse`: A non-AD-safe didactic method using sparse Cholesky
  factorization, suitable for gradient-free samplers.

# Fields
- `sigma::Distribution`: The prior for the marginal standard deviation of the SPDE effect.
- `kappa::Union{Distribution, Vector{<:Distribution}}`: The prior for the `kappa`
  parameter, which controls the spatial range. Can be a single distribution or a
  vector for anisotropic effects.
- `method::Symbol`: The computational method for the SPDE solver.
"""
struct SPDE <: ComponentModel
    sigma::Distribution
    kappa::Union{Distribution, Vector{<:Distribution}}
    method::Symbol
end

# Add to the central component constructor registry.
COMPONENT_TYPE_REGISTRY[:spde] = SPDE
COMPONENT_CONSTRUCTORS[:spde] = (p, params) -> SPDE(
    p.sigma, p.kappa, get(params, :method, :spectral)
)

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
and add it to the linear predictor `eta`, dispatching on the chosen method.
"""
function get_updates(
    m::SPDE, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent

    # The spectral method is only simple for the isotropic case.
    use_spectral = m.method == :spectral && !(m.kappa isa Vector)

    spectral_code = """
        # --- SPDE Component (Spectral): $(key) ---
        let
            local U = spec_registry[:$(key)].hyper.U
            local L = spec_registry[:$(key)].hyper.L
            
            # Eigenvalues of the final precision matrix are (kappa^2 + L_i)^2
            local diag_vals = ($(p_names.kappa)^2 .+ L).^2
            local diag_D = $(p_names.sigma) ./ sqrt.(diag_vals .+ M.noise)
            
            local $(p_names.latent) = U * (diag_D .* $(p_names.raw))
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """

    cholesky_base_code = """
        local Q_laplacian = spec_registry[:$(key)].hyper.Q_template
        local kappa_val = $(p_names.kappa)
        local Q_kappa_term = if kappa_val isa AbstractVector
            Diagonal(kappa_val.^2)
        else
            kappa_val^2 * I
        end
        local L_operator = Q_kappa_term + Q_laplacian
        local Q_final = Symmetric(L_operator' * L_operator)
    """

    cholesky_code = """
        # --- SPDE Component (Cholesky, AD-Safe): $(key) ---
        let
            $(cholesky_base_code)
            local F = cholesky(Matrix(Q_final) + M.noise * I)
            local $(p_names.latent) = $(p_names.sigma) .* (F.L' \\ $(p_names.raw))
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """

    cholesky_sparse_code = """
        # --- SPDE Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            $(cholesky_base_code)
            local F = cholesky(Q_final + M.noise * I)
            local $(p_names.latent) = $(p_names.sigma) .* (F.L' \\ $(p_names.raw))
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """

    if use_spectral
        return spectral_code
    elseif m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    else
        @warn "SPDE method '$(m.method)' with anisotropic kappa is not supported by spectral method. Falling back to dense Cholesky."
        return cholesky_code
    end
end

"""
    get_effects(m::SPDE, chain, M::NamedTuple, ...)

Reconstructs the `SPDE` component's effect from posterior samples, dispatching
on the method used during sampling.
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
    use_spectral = m.method == :spectral && !(m.kappa isa Vector)

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        kappa_samples = get_params_vector(
            chain, string(p_names.kappa), m.kappa isa Vector ? length(m.kappa) : 1
        )
        raw_samples = get_params_vector(chain, string(p_names.raw), n_latent)

        effect_k = zeros(Float64, N_total, n_samples)

        for i in 1:n_samples
            local latent_field
            if use_spectral
                U, L = spec.hyper.U, spec.hyper.L
                kappa_val = kappa_samples[i, 1]
                diag_vals = (kappa_val^2 .+ L).^2
                diag_D = sigma_samples[i] ./ sqrt.(diag_vals .+ noise)
                latent_field = U * (diag_D .* raw_samples[i, :])
            else # Cholesky methods
                current_kappa = kappa_samples[i, :]
                Q_kappa_term = if m.kappa isa Vector
                    Diagonal(current_kappa.^2)
                else
                    current_kappa[1]^2 * I(n_latent)
                end
                L_operator = Q_kappa_term + Q_laplacian
                Q_final = Symmetric(L_operator' * L_operator)
                F = cholesky(Matrix(Q_final) + noise * I)
                latent_field = sigma_samples[i] .* (F.L' \ raw_samples[i, :])
            end
            effect_k[:, i] = view(latent_field, s_idx_full)
        end
        push!(structured_effects, effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
