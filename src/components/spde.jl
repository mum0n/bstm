"""
    SPDE <: ComponentModel

A component model for a spatial field based on the Stochastic Partial Differential
Equation (SPDE) approach. This method provides a direct link between a continuous
Gaussian Process with a Matérn covariance function and a discrete Gaussian Markov
Random Field (GMRF), enabling scalable and principled spatial modeling.

# Version
v1.1.1 (2026-08-14)

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
- `:spectral` (Default, AD-friendly): An efficient, AD-safe method using spectral decomposition.
  Only applicable for isotropic `kappa` priors.
- `:cholesky` (AD-friendly): An AD-safe didactic alternative using dense Cholesky factorization.
- `:cholesky_sparse` (Didactic, Not AD-friendly): A non-AD-safe didactic method using sparse Cholesky
  factorization, suitable for gradient-free samplers.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@bstm`.
- **Optional (in `random()` call)**:
  - `sigma`: `UnivariateDistribution`, prior for the marginal standard deviation. Default: `Exponential(1.0)`.
  - `kappa`: `UnivariateDistribution` or `Vector{<:UnivariateDistribution}`, prior for the `kappa`
    parameter(s). Default: `LogNormal(0, 1)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`).
    Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The marginal standard deviation of the SPDE effect.
- `kappa_<key>`: The spatial range parameter(s).
- `innovations_<key>`: The raw standard normal innovations for the latent field.
- `latent_<key>`: The reconstructed latent SPDE effect.

# Key References
- Lindgren, F., Rue, H., & Lindström, J. (2011). *An explicit link between
  Gaussian fields and Gaussian Markov random fields: The SPDE approach*. Journal
  of the Royal Statistical Society: Series B (Statistical Methodology), 73(4), 423-498.
"""
struct SPDE <: ComponentModel
    sigma::Distribution
    kappa::Union{Distribution, Vector{<:Distribution}}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:spde] = SPDE
COMPONENT_CONSTRUCTORS[:spde] = (p, params) -> SPDE(
    p.sigma, p.kappa, get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:spde] = :spatial

function get_precomputes(m::SPDE, M::NamedTuple, mod_data::Dict)::NamedTuple
    # Data validation moved from get_datastructures!
    if !hasproperty(M, :W) || !isa(M.W, AbstractMatrix) || isempty(M.W)
        error("SPDE model requires a valid, non-empty adjacency matrix `W` provided via keyword.")
    end

    s_N = size(M.W, 1)

    # The processor is now responsible for creating s_idx.
    if !hasproperty(M, :s_idx)
        error(
            "SPDE component '$(mod_data[:key])' failed: s_idx not found in model " *
            "configuration. This should have been set by the model processor."
        )
    end

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
        n_latent=s_N
    )
end

function get_priors(
    m::SPDE, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    key = spec.key
    
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
        "$(p_names.innovations) ~ MvNormal(zeros(T, spec_registry[:$(key)].hyper.n_latent), I)"
    )

    return join(priors, "\n    ")
end

function get_updates(
    m::SPDE, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

    use_spectral = m.method == :spectral && !(m.kappa isa Vector)

    spectral_code = """
        # --- SPDE Component (Spectral): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            U = hyper.U
            L = hyper.L
            
            diag_vals = ($(p_names.kappa)^2 .+ L).^2
            diag_D = $(p_names.sigma) ./ sqrt.(diag_vals .+ M.noise)
            
            $(p_names.latent) = U * (diag_D .* $(p_names.innovations))
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """

    cholesky_base_code = """
        local hyper = spec_registry[:$(key)].hyper
        local Q_laplacian = hyper.Q_template
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
            $(p_names.latent) = $(p_names.sigma) .* (F.L' \\ $(p_names.innovations))
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """

    cholesky_sparse_code = """
        # --- SPDE Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            $(cholesky_base_code)
            local F = cholesky(Q_final + M.noise * I)
            $(p_names.latent) = $(p_names.sigma) .* (F.L' \\ $(p_names.innovations))
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

function get_effects(
    m::SPDE, chain, M::NamedTuple, n_samples::Int, is_multivariate_model::Bool,
    outcomes_N::Int, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    n_latent = spec.hyper.n_latent
    noise = M.noise
    Q_laplacian = spec.hyper.Q_template
    s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, get(PS, :s_idx, []))
    use_spectral = m.method == :spectral && !(m.kappa isa Vector)
    p_names_vec = string.(FlexiChains.parameters(chain))

    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names_vec, v.sigma, k, is_multivariate_model)
        kappa_name = _find_parameter(p_names_vec, v.kappa, k, is_multivariate_model)
        innovations_name = _find_parameter(p_names_vec, v.innovations, k, is_multivariate_model)

        if isempty(sigma_name) || isempty(kappa_name) || isempty(innovations_name)
            @warn "Parameters for SPDE component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        kappa_dim = m.kappa isa Vector ? length(m.kappa) : 1
        kappa_samples = get_params_vector(chain, kappa_name, kappa_dim)
        innovations_samples = get_params_vector(chain, innovations_name, n_latent)

        effect_k = zeros(Float64, N_total, n_samples)

        for i in 1:n_samples
            local latent_field
            if use_spectral
                U, L = spec.hyper.U, spec.hyper.L
                kappa_val = kappa_samples[i, 1]
                diag_vals = (kappa_val^2 .+ L).^2
                diag_D = sigma_samples[i] ./ sqrt.(diag_vals .+ noise)
                latent_field = U * (diag_D .* innovations_samples[i, :])
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
                latent_field = sigma_samples[i] .* (F.L' \ innovations_samples[i, :])
            end
            effect_k[:, i] = view(latent_field, s_idx_full)
        end
        push!(structured_effects, effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
