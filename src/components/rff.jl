"""
    RFF <: ComponentModel

A component model for a Random Fourier Features (RFF) smoother. This component
approximates a stationary kernel (like Squared Exponential or Matérn) by projecting
the input coordinates into a randomized feature space. This transforms the GP into a
more scalable Bayesian linear regression problem.

# Version
v1.0.0

# Mathematical Summary
The RFF method approximates a stationary kernel \$k(\\tau) = k(x - x')\$ by using
Bochner's theorem. The feature map \$\\phi(x)\$ is:
\$\\phi(x) = \\sqrt{2/M} [\\cos(\\omega_1^T x + b_1), \\dots, \\cos(\\omega_M^T x + b_M)]\$
where \$\\{\\omega_j\\}\$ are random frequencies and \$\\{b_j\\}\$ are random phase shifts.
The final effect is a linear combination of these features: \$f(x) = \\phi(x)^T \\beta\$.

# Computational Methods
- `:fixed` (Default, AD-friendly): The RFF weights `W` and biases `b` are pre-computed
  based on the prior mean of the lengthscale and are fixed during sampling. This is
  the most efficient and numerically stable method.
- `:adaptive` (AD-friendly): The RFF weights `W` and biases `b` are treated as
  parameters and sampled from priors centered on the fixed features. This allows the
  model to learn the feature space but is computationally more intensive.
- `:centered` (Didactic, Not AD-friendly): Uses fixed features but samples the
  coefficients `β` directly from a scaled Normal distribution, which can be less
  efficient for MCMC due to posterior correlations.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`, `y`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `n_features`: `Int`, the number of random features to use. Default: `20`.
  - `kernel`: `String`, the name of the kernel to approximate (e.g., `"se"`, `"matern32"`).
    Default: `"se"`.
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the RFF
    coefficients. Default: `Exponential(1.0)`.
  - `lengthscale`: `UnivariateDistribution` or `Vector{<:UnivariateDistribution}`, prior for
    the kernel lengthscale(s). Default: `Gamma(2, 0.5)`.
  - `method`: `Symbol`, computational method (`:fixed`, `:adaptive`, or `:centered`).
    Default: `:fixed`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the RFF coefficients.
- `ls_<key>`: The kernel lengthscale(s).
- `W_<key>`: The learned RFF projection weights (for `:adaptive` method).
- `b_<key>`: The learned RFF biases (for `:adaptive` method).
- `innovations_<key>`: Raw standard normal innovations for the RFF coefficients (for
  `:fixed` and `:adaptive`).
- `latent_<key>`: The RFF coefficients (for `:centered`).
"""
struct RFF <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    n_features::Int
    kernel::String
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:rff] = RFF

COMPONENT_CONSTRUCTORS[:rff] = (p, params) -> RFF(
    p.lengthscale,
    p.sigma,
    get(params, :n_features, 20),
    string(get(params, :kernel, "se")),
    get(params, :method, :fixed)
)

MODEL_TO_STRUCTURE_MAP[:rff] = :smooth

function _generate_rff_fixed_params(
    in_dims::Int, n_features::Int, lengthscale::Union{Real, AbstractVector},
    kernel_name::String
)
    b = rand(Uniform(0, 2 * pi), n_features)
    W = Matrix{Float64}(undef, in_dims, n_features)
    k_name = lowercase(kernel_name)

    if k_name in ["se", "gaussian", "rbf"]
        if lengthscale isa Real
            W .= rand(Normal(0, 1.0 / lengthscale), in_dims, n_features)
        else
            if length(lengthscale) != in_dims
                error("ARD lengthscale vector length mismatch.")
            end
            for d in 1:in_dims
                W[d, :] = rand(Normal(0, 1.0 / lengthscale[d]), n_features)
            end
        end
    elseif occursin("matern", k_name)
        nu = if k_name == "matern12"
            0.5
        elseif k_name == "matern32"
            1.5
        else
            2.5
        end
        df = 2 * nu
        if lengthscale isa Real
            W .= (sqrt(df) / lengthscale) .* rand(TDist(df), in_dims, n_features)
        else
            if length(lengthscale) != in_dims
                error("ARD lengthscale vector length mismatch.")
            end
            for d in 1:in_dims
                W[d, :] = (sqrt(df) / lengthscale[d]) .* rand(TDist(df), n_features)
            end
        end
    else
        @warn "Kernel '$kernel_name' not recognized for RFF. Defaulting to SE."
        return _generate_rff_fixed_params(in_dims, n_features, lengthscale, "se")
    end
    return W, b
end

function get_precomputes(m::RFF, M::NamedTuple, mod_data::Dict)::NamedTuple
    variables = mod_data[:variables]

    if isempty(variables)
        error("The RFF model requires coordinate variables, e.g., `random(x, y, model=:rff)`.")
    end

    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for RFF model not found in data.")
        end
    end

    coords = Matrix{Float64}(M.data[!, Symbol.(variables)])
    in_dims = size(coords, 2)
    
    ls_prior = m.lengthscale
    local ls_initial
    if ls_prior isa Vector
        ls_initial = [mean(p isa Truncated ? untruncated(p) : p) for p in ls_prior]
    else
        ls_initial = mean(ls_prior isa Truncated ? untruncated(ls_prior) : ls_prior)
    end

    W_fixed, b_fixed = _generate_rff_fixed_params(
        in_dims, m.n_features, ls_initial, m.kernel
    )

    return (
        coords=coords,
        W_fixed=W_fixed,
        b_fixed=b_fixed,
        n_latent=m.n_features,
        in_dims=in_dims
    )
end

"""
    _rff_log_marginal_likelihood(y_residual, Phi, sigma, y_sigma, noise=1e-6)

Computes the exact log marginal likelihood for a Random Fourier Features smoother with
  coefficients integrated out analytically.
"""
function _rff_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    Phi::AbstractMatrix,
    sigma::T,
    y_sigma::T,
    noise::Real=1e-6
) where {T}
    N = length(y_residual)
    M_feat = size(Phi, 2)
    T_num = promote_type(T, typeof(noise))
    
    inv_sigma_y2 = one(T_num) / (y_sigma^2 + T_num(noise))
    scale = sigma^2 + T_num(noise)
    
    PTP = Matrix{T_num}(Phi' * Phi)
    PTy = Vector{T_num}(Phi' * y_residual)
    
    Q_base = Matrix{T_num}(I, M_feat, M_feat) .+ (scale * inv_sigma_y2) .* PTP
    for k in 1:M_feat
        Q_base[k, k] += T_num(noise)
    end
    
    F = cholesky(Symmetric(Q_base))
    log_det_diff = - M_feat * log(scale) - 2 * sum(log.(diag(F.U)))
    
    b = PTy .* inv_sigma_y2
    v = F.L \ b
    quad_term = scale * dot(v, v)
    
    log_lik = - (N / 2) * log(2 * T_num(pi) * (y_sigma^2 + T_num(noise))) -
              (inv_sigma_y2 / 2) * dot(y_residual, y_residual) +
              (1 / 2) * log_det_diff +
              (1 / 2) * quad_term
              
    return log_lik
end

function get_priors(
    m::RFF, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    key = spec.key
    
    priors = String[]
    push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")

    if m.method != :marginalized
        if m.lengthscale isa Vector
            ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
            push!(priors, "$(p_names.ls) ~ Product([$(ls_priors_str)])")
        else
            ls_prior_str = _distribution_to_string(m.lengthscale)
            push!(priors, "$(p_names.ls) ~ $(ls_prior_str)")
        end
        
        if m.method == :adaptive
            push!(priors, "$(p_names.W) ~ DynamicPPL.NamedDist(MvNormal(vec(spec_registry[:$(key)].hyper.W_fixed), 0.1), :$(p_names.W))")
            push!(priors, "$(p_names.b) ~ NamedDist(MvNormal(spec_registry[:$(key)].hyper.b_fixed, 0.1), :$(p_names.b))")
        end

        if m.method in [:fixed, :adaptive]
            push!(priors,
                "$(p_names.ure) ~ MvNormal(zeros(T, spec_registry[:$(key)].hyper.n_latent), I)")
        end
    end

    return join(priors, "\n    ")
end

function get_updates(
    m::RFF, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    
    hyper_access = "spec_registry[:$(key)].hyper"
    in_dims = spec.hyper.in_dims
    n_latent = spec.hyper.n_latent

    phi_code(W_expr, b_expr) = """
        X_coords = $(hyper_access).coords
        W_matrix = reshape($(W_expr), $(in_dims), $(n_latent))
        Phi = sqrt(2.0 / $(n_latent)) .* cos.((X_coords * W_matrix) .+ $(b_expr)')
    """

    fixed_code = """
        # --- RFF Smoother (Fixed Features): $(key) ---
        let
            $(phi_code("$(hyper_access).W_fixed", "$(hyper_access).b_fixed"))
            scaled_coeffs = $(p_names.ure) .* $(p_names.sigma)
            $(p_names.sre) = Phi * scaled_coeffs
            $(eta_target) = $(eta_target) .+ $(p_names.sre)
        end
    """

    adaptive_code = """
        # --- RFF Smoother (Adaptive Features): $(key) ---
        let
            $(phi_code(string(p_names.W), string(p_names.b)))
            scaled_coeffs = $(p_names.ure) .* $(p_names.sigma)
            $(p_names.sre) = Phi * scaled_coeffs
            $(eta_target) = $(eta_target) .+ $(p_names.sre)
        end
    """

    centered_code = """
        # --- RFF Smoother (Centered): $(key) ---
        let
            $(phi_code("$(hyper_access).W_fixed", "$(hyper_access).b_fixed"))
            $(p_names.sre) ~ MvNormal(zeros(T, $(n_latent)), $(p_names.sigma)^2 * I)
            rff_effect = Phi * $(p_names.sre)
            $(eta_target) = $(eta_target) .+ rff_effect
        end
    """

    marginalized_code = """
        # --- RFF Smoother (Marginalized): $(key) ---
        let
            $(phi_code("$(hyper_access).W_fixed", "$(hyper_access).b_fixed"))
            y_residual = M.y_obs .- $(eta_target)
            log_lik_marginalized_$(key) = _rff_log_marginal_likelihood(
                y_residual,
                Phi,
                $(p_names.sigma),
                y_sigma,
                M.noise
            )
            Turing.@addlogprob! log_lik_marginalized_$(key)
        end
    """

    if m.method == :fixed
        return fixed_code
    elseif m.method == :adaptive
        return adaptive_code
    elseif m.method == :centered
        return centered_code
    elseif m.method == :marginalized
        return marginalized_code
    else; error("Unsupported method '$(m.method)' for RFF component. Use :fixed, :adaptive, :centered, or :marginalized."); end
end

"""
    get_effects(m::RFF, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the RFF effect from posterior samples. This version is CPU-only,
uses modern chain accessors, and is optimized to use vectorized operations for
fixed-feature methods, avoiding inefficient per-sample loops.
"""
function get_effects(
    m::RFF, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = _get_chain_n_samples(chain)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    noise_val = get(M, :noise, 1e-6)
    
    hyper = spec.hyper
    in_dims = hyper.in_dims
    n_features = hyper.n_latent

    # --- Coordinate Handling: Combine training and prediction sets on CPU ---
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        coords_pred_cpu = Matrix{Float64}(PS.data[!, Symbol.(coord_vars)])
        vcat(hyper.coords, coords_pred_cpu)
    else
        hyper.coords
    end
    N_total = size(coords_full, 1)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k_outcome in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k_outcome)

        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k_outcome,
            is_multivariate_model)
        if isempty(sigma_name)
            @warn "Sigma parameter for RFF component $(spec.key) (outcome $k_outcome) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end
        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]

        # Initialize the output matrix for the full effect on the CPU
        effect_k = zeros(Float64, N_total, n_samples)

        if m.method == :marginalized
            y_sigma_name = _find_parameter(p_names, "y_sigma", k_outcome, is_multivariate_model)
            y_sigma_samples = if !isempty(y_sigma_name)
                get_params_vector(chain, y_sigma_name, 1)[:, 1]
            else
                fill(1.0, n_samples)
            end
            
            W_matrix = hyper.W_fixed
            b_vec = hyper.b_fixed
            Phi_train = sqrt(2.0 / n_features) .* cos.((hyper.coords * W_matrix) .+ b_vec')
            Phi_full = sqrt(2.0 / n_features) .* cos.((coords_full * W_matrix) .+ b_vec')
            
            y_vec = M.y_obs isa AbstractMatrix ? M.y_obs[:, k_outcome] : M.y_obs
            PTP = Matrix{Float64}(Phi_train' * Phi_train)
            PTy = Vector{Float64}(Phi_train' * y_vec)
            
            coeffs_samples = zeros(Float64, n_features, n_samples)
            for i in 1:n_samples
                sig = sigma_samples[i]
                y_sig = y_sigma_samples[i]
                
                scale = sig^2 + noise_val
                inv_sigma_y2 = 1.0 / (y_sig^2 + noise_val)
                
                Q_base = Matrix{Float64}(I, n_features,
                    n_features) .+ (scale * inv_sigma_y2) .* PTP
                for j in 1:n_features
                    Q_base[j, j] += noise_val
                end
                
                F = cholesky(Symmetric(Q_base))
                b = PTy .* inv_sigma_y2
                mu = scale .* (F \ b)
                
                z = randn(n_features)
                coeffs_samples[:, i] = mu .+ sqrt(max(scale, 1e-12)) .* (F.U \ z)
            end
            effect_k = Phi_full * coeffs_samples

        elseif m.method == :adaptive
            # --- Adaptive Method: Per-sample loop is necessary as W and b change ---
            W_name = _find_parameter(p_names, string(p_names_k.W), k_outcome,
                is_multivariate_model)
            b_name = _find_parameter(p_names, string(p_names_k.b), k_outcome,
                is_multivariate_model)
            ure_name = _find_parameter(p_names, string(p_names_k.ure), k_outcome,
                is_multivariate_model)
            
            if isempty(W_name) || isempty(b_name) || isempty(ure_name)
                @warn "Adaptive RFF parameters for component $(spec.key) (outcome $k_outcome) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            
            W_samples = get_params_matrix(chain, W_name, in_dims * n_features)
            b_samples = get_params_matrix(chain, b_name, n_features)
            ure_samples = get_params_matrix(chain, ure_name, n_features)

            for i in 1:n_samples
                W_matrix = reshape(W_samples[i, :], in_dims, n_features)
                b_vec = b_samples[i, :]
                innov_i = ure_samples[i, :]
                sigma_i = sigma_samples[i]

                Phi = sqrt(2.0 / n_features) .* cos.((coords_full * W_matrix) .+ b_vec')
                scaled_coeffs = innov_i .* sigma_i
                effect_k[:, i] = Phi * scaled_coeffs
            end

        else # :fixed or :centered methods
            W_matrix = hyper.W_fixed
            b_vec = hyper.b_fixed
            Phi = sqrt(2.0 / n_features) .* cos.((coords_full * W_matrix) .+ b_vec')

            if m.method == :fixed
                ure_name = _find_parameter(p_names, string(p_names_k.ure), k_outcome,
                    is_multivariate_model)
                if isempty(ure_name)
                    @warn "ure for RFF component $(spec.key) (outcome $k_outcome) not found. Returning zero-matrix."
                    push!(structured_effects, zeros(Float64, N_total, n_samples))
                    continue
                end
                ure_samples = get_params_matrix(chain, ure_name, n_features)
                
                scaled_coeffs = ure_samples' .* sigma_samples'
                effect_k = Phi * scaled_coeffs

            else # :centered
                sre_name = _find_parameter(p_names, string(p_names_k.sre), k_outcome,
                    is_multivariate_model)
                if isempty(sre_name)
                    @warn "Latent coefficients for centered RFF component $(spec.key) (outcome $k_outcome) not found. Returning zero-matrix."
                    push!(structured_effects, zeros(Float64, N_total, n_samples))
                    continue
                end
                coeffs_samples = get_params_matrix(chain, sre_name, n_features)
                
                effect_k = Phi * coeffs_samples'
            end
        end
        
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
