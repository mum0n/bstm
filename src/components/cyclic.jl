"""
    Cyclic <: ComponentModel

A component model for cyclic temporal effects, typically used for seasonal patterns.
It implements a first-order cyclic random walk (RW1 on a circle), where the last
point smoothly connects back to the first. This is a type of Gaussian Markov
Random Field (GMRF) with a circulant precision matrix.

# Version
v1.1.3 (2026-08-15)

# Mathematical Summary
The cyclic random walk models a latent field \$\\phi\$ where the value at time \$t\$ is
conditionally dependent on its neighbors, with the first and last points
considered neighbors. The conditional distribution is:
\$\\phi_t | \\phi_{-t} \\sim \\mathcal{N}\\left( \\frac{1}{2}(\\phi_{t-1} + \\phi_{t+1}), \\frac{\\sigma^2}{2} \\right)\$
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
    sigma::Distribution
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
decomposition (`U`, `L`) and Cholesky factorization.
"""
function get_precomputes(m::Cyclic, M::NamedTuple, mod_data::Dict)::NamedTuple
    # The `process_random_module!` is expected to have set up `M.u_N` and `M.u_idx`
    # based on the seasonal index variable provided in the formula.
    u_N = get(M, :u_N, 0)
    if u_N == 0
        error(
            "The Cyclic model requires a seasonal context (`u_N`), but it has not " *
            "been established. Ensure a seasonal index variable is provided."
        )
    end

    # Validate the period against the number of unique levels.
    if m.period != u_N
        @warn "The specified period ($(m.period)) does not match the number of " *
              "unique levels in the seasonal index variable ($(u_N)). " *
              "Setting period to $u_N."
        n = u_N
    else
        n = m.period
    end
    
    template = build_structure_template(:cyclic, n)
    F = cholesky(Symmetric(Matrix(template.matrix) + M.noise * I))
    
    return (
        Q_template=template.matrix,
        scaling_factor=template.scaling_factor,
        U=template.U,
        L=template.L,
        n_latent=n,
        cholesky_factor=F
    )
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
    return """
    # Priors for Cyclic component: $(spec.key)
    $(p_names.sigma) ~ $(_distribution_to_string(m.sigma))
    $(p_names.innovations) ~ MvNormal(zeros($(n_latent)), I)
    """
end

"""
    get_updates(m::Cyclic, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to sample the latent cyclic field. Supports three methods:
- `:spectral` (default): An efficient, AD-safe method using spectral decomposition.
- `:cholesky`: An AD-safe didactic alternative using dense Cholesky factorization.
- `:cholesky_sparse`: A non-AD-safe didactic method using sparse Cholesky
  factorization, suitable for gradient-free samplers.
"""
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
            diag_D[1] = 0.0 # Enforce sum-to-zero constraint
            latent_field = U * (diag_D .* $(p_names.innovations))
            $(eta_target) .+= view(latent_field, M.u_idx)
        end
    """

    cholesky_code = """
        # --- Cyclic Component: $(key) (Cholesky Method, AD-Safe) ---
        let
            hyper = spec_registry[:$(key)].hyper
            F = hyper.cholesky_factor
            latent_field_raw = F.L' \\ $(p_names.innovations)
            
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_raw)
            )
            
            latent_field = latent_field_raw .* $(p_names.sigma)
            $(eta_target) .+= view(latent_field, M.u_idx)
        end
    """

    cholesky_sparse_code = """
        # --- Cyclic Component: $(key) (Sparse Cholesky, Not AD-Safe): ---
        let
            hyper = spec_registry[:$(key)].hyper
            Q = hyper.Q_template
            F = cholesky(Symmetric(Q + M.noise * I))
            latent_field_raw = F.L' \\ $(p_names.innovations)
            
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_raw)
            )
            
            latent_field = latent_field_raw .* $(p_names.sigma)
            $(eta_target) .+= view(latent_field, M.u_idx)
        end
    """

    if m.method == :spectral
        return spectral_code
    elseif m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    else
        error(
            "Unsupported method '$(m.method)' for Cyclic component. Supported " *
            "methods are :spectral, :cholesky, and :cholesky_sparse."
        )
    end
end


"""
    get_effects(m::Cyclic, chain, spec, M, PS)

Reconstructs the `Cyclic` component's effect from posterior samples, applying a
sum-to-zero constraint for identifiability.   handle
GPU arrays by moving sampled parameters to the device for computation and moving
the final results back to the CPU.
"""
function get_effects(
    m::Cyclic, chain::Chains, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device
    noise = M.noise
    n_latent = spec.hyper.n_latent

    # --- Coordinate/Index Handling: Combine training and prediction sets ---
    u_idx_full_device = if !isnothing(PS) && hasproperty(PS.data, :u_idx)
        # M.u_idx is already on the device. PS.data.u_idx is on CPU.
        vcat(M.u_idx, to_device(PS.data.u_idx))
    else
        M.u_idx
    end
    N_total = length(u_idx_full_device)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        
        # Find parameter names in the MCMC chain
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for Cyclic component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples_cpu = get_params_matrix(chain, innovations_name, n_latent)

        # Initialize the output matrix for latent effects on the target device
        effect_k_device = to_device(zeros(Float64, n_latent, n_samples))

        # --- Sample-wise Reconstruction on the Target Device ---
        if m.method == :spectral
            U = spec.hyper.U # Already on device
            L = spec.hyper.L # Already on device
            
            for j in 1:n_samples
                sigma_j = sigma_samples_cpu[j] # CPU scalar
                innov_j_device = to_device(innovations_samples_cpu[j, :])
                
                diag_D = sigma_j ./ sqrt.(L .+ noise)
                diag_D[1] = 0.0 # Enforce sum-to-zero constraint
                effect_k_device[:, j] = U * (diag_D .* innov_j_device)
            end
        else # :cholesky or :cholesky_sparse
            F = spec.hyper.cholesky_factor # Already on device
            
            for j in 1:n_samples
                sigma_j = sigma_samples_cpu[j]
                innov_j_device = to_device(innovations_samples_cpu[j, :])

                latent_field_raw = F.L' \ innov_j_device
                latent_field_centered = latent_field_raw .- mean(latent_field_raw)
                effect_k_device[:, j] = latent_field_centered .* sigma_j
            end
        end
        
        # Indexing on the device and moving the final result to CPU
        indexed_effects_device = effect_k_device[u_idx_full_device, :]
        push!(structured_effects, Array(indexed_effects_device))
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
