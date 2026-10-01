"""
    Cyclic <: ComponentModel

A component model for cyclic temporal effects, typically used for seasonal patterns.
It implements a first-order cyclic random walk (RW1 on a circle), where the last
point smoothly connects back to the first. This is a type of Gaussian Markov
Random Field (GMRF) with a circulant precision matrix.

# Version
v1.0.0

# Mathematical Summary
The cyclic random walk models a latent field \$\\phi\$ where the value at time \$t\$ is
conditionally dependent on its neighbors, with the first and last points
considered neighbors. The conditional distribution is:
\$\\phi_t | \\phi_{-t} \\sim \\mathcal{N}\\left( \\frac{1}{2}(\\phi_{t-1} + \\phi_{t+1}),
  \\frac{\\sigma^2}{2} \\right)\$
(indices are taken modulo the period).

The joint precision matrix \$Q\$ is a circulant matrix corresponding to this structure.
Like the standard RW1, this is an intrinsic GMRF with a rank deficiency of 1, so a
sum-to-zero constraint is imposed on the latent field for identifiability.

# Computational Methods
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the circulant precision matrix. Recommended for NUTS.
- `:cholesky` (AD-friendly): Uses a pre-computed dense Cholesky factorization.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization,
  which is not compatible with most AD backends.

# Inputs
- **Required**:
  - A seasonal index variable (e.g., `month`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `period`: `Int`, the length of the cycle. Must match the number of unique
    levels in the index variable. Default: `12`.
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the
    cyclic effect. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`).
    Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the cyclic effect.
- `innovations_<key>`: The raw standard normal innovations for the effect.

# Key References
- Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and
  Applications*. CRC Press.
- Wikipedia: Random walk
"""
struct Cyclic <: ComponentModel
    period::Int
    sigma::Union{Distribution, Real}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:cyclic] = Cyclic

COMPONENT_CONSTRUCTORS[:cyclic] = (p, params) -> Cyclic(
    get(params, :period, 12), p.sigma, get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:cyclic] = :seasonal

"""
    get_precomputes(m::Cyclic, M::NamedTuple, mod_data::Dict)::NamedTuple

Validates the seasonal index variable and pre-computes the circulant precision
matrix (`Q_template`) for the cyclic random walk, along with its spectral
decomposition (`U`, `L`) and Cholesky factorization. This is a CPU-only implementation.
"""
function get_precomputes(m::Cyclic, M::NamedTuple, mod_data::Dict)::NamedTuple
    raw_vars = get(mod_data, :variables, [])
    variables = raw_vars isa AbstractVector ? raw_vars : [raw_vars]
    
    u_N = get(M, :u_N, 0)
    if u_N == 0 && !isempty(variables) && hasproperty(M, :data) && hasproperty(M.data,
        Symbol(variables[1]))
        u_N = length(unique(M.data[!, Symbol(variables[1])]))
    end
    if u_N == 0
        u_N = Int(m.period > 0 ? m.period : 12)
    end

    n = Int(u_N)
    template = build_structure_template(:cyclic, n)
    noise_val = get(M, :noise, 1e-6)
    F = cholesky(Symmetric(Matrix(template.matrix) + noise_val * I))
    
    return (
        Q_template=template.matrix,
        scaling_factor=template.scaling_factor,
        U=template.U,
        L=template.L,
        n_latent=n,
        cholesky_factor=F,
        model_type=:cyclic
    )
end

"""
    _cyclic_log_marginal_likelihood(y_residual, u_idx, u_N, Q_template, L_eig, sigma,
      y_sigma, noise=1e-6)

Computes the exact log marginal likelihood for a Cyclic seasonal process integrated out
  analytically.
"""
function _cyclic_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    u_idx::AbstractVector{Int},
    u_N::Int,
    Q_template::AbstractMatrix,
    L_eig::AbstractVector,
    sigma::T,
    y_sigma::T,
    noise::Real=1e-6
) where {T}
    N = length(y_residual)
    T_num = promote_type(T, typeof(noise))
    
    # Pre-accumulate observation counts and sums per seasonal index
    N_u = zeros(T_num, u_N)
    S_u = zeros(T_num, u_N)
    for i in 1:N
        u = u_idx[i]
        if 1 <= u <= u_N
            N_u[u] += one(T_num)
            S_u[u] += y_residual[i]
        end
    end
    
    inv_sigma_y2 = one(T_num) / (y_sigma^2 + T_num(noise))
    scale = sigma^2 + T_num(noise)
    
    Q_base = Matrix{T_num}(Q_template)
    for u in 1:u_N
        Q_base[u, u] += T_num(noise) + N_u[u] * inv_sigma_y2 * scale
    end
    
    F = cholesky(Symmetric(Q_base))
    
    # Determinant term (Cyclic has rank deficiency 1)
    log_det_prior = sum(log.(L_eig[2:end] .+ T_num(noise)))
    log_det_diff = - (u_N - 1) * log(scale) + log_det_prior - 2 * sum(log.(diag(F.U)))
    
    # Quadratic term
    b = S_u .* inv_sigma_y2
    v = F.L \ b
    quad_term = scale * dot(v, v)
    
    log_lik = - (N / 2) * log(2 * T_num(pi) * (y_sigma^2 + T_num(noise))) -
              (inv_sigma_y2 / 2) * dot(y_residual, y_residual) +
              (1 / 2) * log_det_diff +
              (1 / 2) * quad_term
              
    return log_lik
end

"""
    get_priors(m::Cyclic, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for the scale parameter `sigma` and the raw innovations `innovations`.
"""
function get_priors(
    m::Cyclic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    if m.method == :marginalized
        return """
        # Priors for Cyclic component: $(spec.key)
        $(_prior_or_constant(p_names.sigma, m.sigma))
        """
    else
        return """
        # Priors for Cyclic component: $(spec.key)
        $(_prior_or_constant(p_names.sigma, m.sigma))
        $(p_names.innovations) ~ MvNormal(zeros(T, $(n_latent)), I)
        """
    end
end

function get_updates(
    m::Cyclic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent

    spectral_code = """
        # --- Cyclic Component: $(key) (Spectral Method) ---
        let
            hyper = spec_registry[:$(key)].hyper
            U, L = hyper.U, hyper.L
            diag_D = $(p_names.sigma) ./ sqrt.(L .+ M.noise)
            diag_D[L .<= (1e-10 * maximum(L))] .= 0.0   # zero EVERY null direction (a disconnected graph has more than one)
            $(p_names.latent_field) = U * (diag_D .* $(p_names.innovations))
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.u_idx)
        end
    """

    cholesky_code = """
        # --- Cyclic Component: $(key) (Cholesky Method, AD-Safe) ---
        let
            hyper = spec_registry[:$(key)].hyper
            F = hyper.cholesky_factor
            latent_field_unscaled = F.L' \\ $(p_names.innovations)
            
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_unscaled)
            )
            
            $(p_names.latent_field) = latent_field_unscaled .* $(p_names.sigma)
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.u_idx)
        end
    """

    cholesky_sparse_code = """
        # --- Cyclic Component: $(key) (Sparse Cholesky, Not AD-Safe): ---
        let
            hyper = spec_registry[:$(key)].hyper
            Q = hyper.Q_template
            F = cholesky(Symmetric(Q + M.noise * I))
            latent_field_unscaled = F.L' \\ $(p_names.innovations)
            
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_unscaled)
            )
            
            $(p_names.latent_field) = latent_field_unscaled .* $(p_names.sigma)
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.u_idx)
        end
    """

    marginalized_code = """
        # --- Cyclic Component: $(key) (Marginalized Method) ---
        let
            hyper = spec_registry[:$(key)].hyper
            y_residual = M.y_obs .- $(eta_target)
            log_lik_marginalized_$(key) = _cyclic_log_marginal_likelihood(
                y_residual,
                M.u_idx,
                hyper.n_latent,
                hyper.Q_template,
                hyper.L,
                $(p_names.sigma),
                @isdefined(y_sigma) ? y_sigma : 0.0,
                M.noise
            )
            Turing.@addlogprob! log_lik_marginalized_$(key)
        end
    """

    if m.method == :spectral
        return spectral_code
    elseif m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    elseif m.method == :marginalized
        return marginalized_code
    else
        error("Unsupported method '$(m.method)' for Cyclic component. Supported methods are :spectral, :cholesky, :cholesky_sparse, and :marginalized.")
    end
end


"""
    get_effects(m::Cyclic, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the `Cyclic` component's effect from posterior samples, applying a
sum-to-zero constraint for identifiability. This version is CPU-only and uses
modern chain accessors.
"""
function get_effects(
    m::Cyclic, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    n_samples = _get_chain_n_samples(chain)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    noise = M.noise
    n_latent = spec.hyper.n_latent

    # --- Coordinate/Index Handling: Combine training and prediction sets on CPU ---
    u_idx_train = M.u_idx
    u_idx_full = if !isnothing(PS)
        if hasproperty(PS, :u_idx)
            vcat(u_idx_train, PS.u_idx)
        elseif hasproperty(PS, :data) && haskey(M, :u_idx_var) &&
               hasproperty(PS.data, M.u_idx_var)
            raw_new = PS.data[!, M.u_idx_var]
            if haskey(M, :u_values) && !isnothing(M.u_values)
                u_map = Dict(v => i for (i, v) in enumerate(M.u_values))
                vcat(u_idx_train, [get(u_map, v, 1) for v in raw_new])
            else
                vcat(u_idx_train, Int.(raw_new))
            end
        elseif hasproperty(PS, :data) && hasproperty(PS.data, :u_idx)
            vcat(u_idx_train, Int.(PS.data.u_idx))
        else
            u_idx_train
        end
    else
        u_idx_train
    end
    N_total = length(u_idx_full)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        sigma_samples = _resolve_hyper_samples(
            chain, p_names, p_names_k.sigma, m.sigma, k,
            is_multivariate_model, n_samples;
            as_matrix = true)

        if isnothing(sigma_samples)
            @warn "Parameters for $(spec.key) (outcome $k) not resolved, and " *
                  "sigma is not a pinned constant. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end


        # Initialize the output matrix for latent effects
        effect_k_matrix = zeros(Float64, n_latent, n_samples)

        # --- Sample-wise Reconstruction ---
        if m.method == :marginalized
            y_sigma_name = _find_parameter(p_names, "y_sigma", k, is_multivariate_model)
            y_sigma_samples = if !isempty(y_sigma_name)
                get_params_vector(chain, y_sigma_name, 1)[:, 1]
            else
                fill(1.0, n_samples)
            end
            
            y_vec = M.y_obs isa AbstractMatrix ? M.y_obs[:, k] : M.y_obs
            
            N_u = zeros(Float64, n_latent)
            S_u = zeros(Float64, n_latent)
            for i in 1:length(u_idx_train)
                u = u_idx_train[i]
                if 1 <= u <= n_latent
                    N_u[u] += 1.0
                    S_u[u] += y_vec[i]
                end
            end
            
            Q_template = spec.hyper.Q_template
            
            for j in 1:n_samples
                sig = sigma_samples[j, 1]
                y_sig = y_sigma_samples[j]
                
                scale = sig^2 + noise
                inv_sigma_y2 = 1.0 / (y_sig^2 + noise)
                
                Q_base = Matrix{Float64}(Q_template)
                for u in 1:n_latent
                    Q_base[u, u] += noise + N_u[u] * inv_sigma_y2 * scale
                end
                
                F = cholesky(Symmetric(Q_base))
                b = S_u .* inv_sigma_y2
                mu = scale .* (F \ b)
                
                z = randn(n_latent)
                x_train = mu .+ sqrt(max(scale, 1e-12)) .* (F.U \ z)
                x_train .-= mean(x_train)
                effect_k_matrix[:, j] = x_train
            end
        else
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            innovations_samples = get_params_matrix(chain, innovations_name, n_latent)

            if m.method == :spectral
                U = spec.hyper.U
                L = spec.hyper.L
                
                for j in 1:n_samples
                    sigma_j = sigma_samples[j, 1]
                    innov_j = innovations_samples[j, :]
                    
                    diag_D = sigma_j ./ sqrt.(L .+ noise)
                    _zero_null_modes!(diag_D, L)   # zero EVERY null direction; a disconnected graph has more than one
                    effect_k_matrix[:, j] = U * (diag_D .* innov_j)
                end
            else # :cholesky or :cholesky_sparse
                F = spec.hyper.cholesky_factor
                for j in 1:n_samples
                    sigma_j = sigma_samples[j, 1]
                    innov_j = innovations_samples[j, :]

                    latent_field_unscaled = F.L' \ innov_j
                    latent_field_centered = latent_field_unscaled .- mean(latent_field_unscaled)
                    effect_k_matrix[:, j] = latent_field_centered .* sigma_j
                end
            end
        end

        # Index the reconstructed latent effects to match the observation indices
        indexed_effects = effect_k_matrix[u_idx_full, :]
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end 
