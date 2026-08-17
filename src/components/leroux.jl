"""
    Leroux <: ComponentModel

A component for a Leroux model, which is a proper Conditional Autoregressive (CAR)
model. It defines spatial correlation as a convex combination of a spatially
structured (ICAR) component and an unstructured (IID) component, controlled by a
single mixing parameter, `rho`.

# Version
v2.2.0 (2026-08-17)

# Mathematical Summary
The Leroux model is a proper CAR model, meaning its precision matrix is always
positive definite. It defines the precision matrix \$\\mathbf{Q}\$ as a convex
combination of an identity matrix \$\\mathbf{I}\$ and a scaled ICAR precision matrix
\$\\mathbf{Q}^*\$ (where \$\\mathbf{Q}^* = D - W\$):
\$\\mathbf{Q} = (1-\\rho)\\mathbf{I} + \\rho\\mathbf{Q}^*\$
This structure allows the model to smoothly interpolate between unstructured random
effects (\$\\rho=0\$) and a fully structured ICAR model (\$\\rho=1\$), providing a
flexible way to model spatial autocorrelation.

# Computational Methods
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the ICAR precision matrix. Recommended for gradient-based samplers.
- `:cholesky` (AD-friendly): Uses a dense Cholesky factorization of the
  full Leroux precision matrix, computed on-the-fly.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization,
  computed on-the-fly.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@bstm`.
- **Optional (in `random()` call)**:
  - `rho`: A `UnivariateDistribution` for the prior on the mixing parameter. Default: `Beta(1,1)`.
  - `sigma`: A `UnivariateDistribution` for the prior on the overall standard deviation. Default: `Exponential(1.0)`.
  - `method`: A `Symbol` specifying the computational method. Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The overall marginal standard deviation.
- `rho_<key>`: The mixing parameter.
- `innovations_<key>`: The raw standard normal innovations for the effect.
- `latent_<key>`: The reconstructed latent spatial field.

# Key References
- Leroux, B. G., Lei, X., & Breslow, N. (2000). Estimation of disease rates in
  small areas: a new mixed model for spatial dependence. In *Statistical models
  in epidemiology, the environment, and clinical trials* (pp. 179-191). Springer.
- Wikipedia: Conditional autoregressive model
"""
struct Leroux <: ComponentModel
    rho::UnivariateDistribution
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:leroux] = Leroux

COMPONENT_CONSTRUCTORS[:leroux] = (p, params) -> Leroux(
    p.rho, p.sigma, get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:leroux] = :spatial

function get_precomputes(m::Leroux, M::NamedTuple, mod_data::Dict)::NamedTuple
    s_N = get(M, :s_N, 0)
    W = get(M, :W, nothing)
    if s_N == 0 || isnothing(W)
        error(
            "Could not perform pre-computation for Leroux component because " *
            "spatial context (s_N and W) is missing."
        )
    end

    # Get the device transfer function
    to_device = M.to_device

    # build_structure_template returns CPU arrays
    template = build_structure_template(:icar, s_N; W=W)
    
    # Move precomputed structures to the target device.
    # Do not pre-compute Cholesky factor as it depends on `rho`.
    return (
        Q_template=to_device(template.matrix),
        U=to_device(template.U),
        L=to_device(template.L),
        scaling_factor=template.scaling_factor,
        n_latent=s_N
    )
end

function get_priors(
    m::Leroux, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    is_multivariate = (arch == "multivariate")
    is_shared = get(spec.params, :shared, false)
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")
        push!(priors_acc, "$(p_names.rho) ~ $(_distribution_to_string(m.rho))")
    end
    push!(priors_acc, "$(p_names.innovations) ~ MvNormal(zeros(T, $(n_latent)), I)")
    return join(priors_acc, "\n    ")
end


"""
    get_updates(m::Leroux, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to compute the Leroux effect and add it to the linear predictor `eta`.
Supports three methods: `:spectral`, `:cholesky`, and `:cholesky_sparse`.
"""
function get_updates(
    m::Leroux, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "s_idx"
    key = spec.key

    spectral_code = """
        # --- Leroux Spectral Assembly: $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            diag_D = $(p_names.sigma) ./ sqrt.((1.0 .- $(p_names.rho)) .+ 
                                              $(p_names.rho) .* hyper.L .+ M.noise)
            $(p_names.latent) = hyper.U * (diag_D .* $(p_names.innovations))
            $(eta_target) .+= view($(p_names.latent), M.$(index_var))
        end
        """

    cholesky_code = """
        # --- Leroux Cholesky Assembly (Dense, AD-Safe): $(key) ---
        let
            Q_template = spec_registry[:$(key)].hyper.Q_template
            rho_val = $(p_names.rho)
            Q_final = (1.0 - rho_val) .* I(size(Q_template, 1)) .+ rho_val .* Q_template
            F = cholesky(Symmetric(Matrix(Q_final) + M.noise * I))
            $(p_names.latent) = $(p_names.sigma) .* (F.U \\ $(p_names.innovations))
            $(eta_target) .+= view($(p_names.latent), M.$(index_var))
        end
        """

    cholesky_sparse_code = """
        # --- Leroux Cholesky Assembly (Sparse, Not AD-Safe): $(key) ---
        let
            Q_template = spec_registry[:$(key)].hyper.Q_template
            rho_val = $(p_names.rho)
            Q_final = (1.0 - rho_val) .* sparse(I, size(Q_template)...) .+ 
                      rho_val .* Q_template
            F = cholesky(Symmetric(Q_final + M.noise * I))
            $(p_names.latent) = $(p_names.sigma) .* (F.U \\ $(p_names.innovations))
            $(eta_target) .+= view($(p_names.latent), M.$(index_var))
        end
        """

    if m.method == :spectral
        return spectral_code
    elseif m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    else
        error("Unsupported method '$(m.method)' for Leroux component.")
    end
end


function get_effects(
    m::Leroux, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device

    structured_effects = Vector{Matrix{Float64}}()
    n_latent = spec.hyper.n_latent
    noise = M.noise

    # --- Coordinate/Index Handling: Combine training and prediction sets on device ---
    s_idx_train_device = M.s_idx # Already on device
    s_idx_full_device = if !isnothing(PS) && hasproperty(PS.data, :s_idx)
        s_idx_pred_cpu = get(PS.data, :s_idx, [])
        vcat(s_idx_train_device, to_device(s_idx_pred_cpu))
    else
        s_idx_train_device
    end
    N_total = length(s_idx_full_device)

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        rho_name = _find_parameter(p_names, string(p_names_k.rho), k, is_multivariate_model)
        innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(rho_name) || isempty(innovations_name)
            @warn "Parameters for Leroux component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        rho_samples_cpu = get_params_vector(chain, rho_name, 1)[:, 1]
        innovations_samples_cpu = get_params_matrix(chain, innovations_name, n_latent)
        
        # Initialize the output matrix for the full latent field on the target device
        effect_k_latent_device = to_device(zeros(Float64, n_latent, n_samples))

        # --- Sample-wise Reconstruction on the Target Device ---
        for s in 1:n_samples
            # These are scalars, no need to move to device explicitly
            sigma_s = sigma_samples_cpu[s]
            rho_s = rho_samples_cpu[s]
            # Move vector to device
            innov_s_device = to_device(innovations_samples_cpu[s, :])

            if m.method == :spectral
                U_device = spec.hyper.U # Already on device
                L_eig_device = spec.hyper.L # Already on device
                diag_D_s = sigma_s ./ sqrt.((1.0 - rho_s) .+ rho_s .* L_eig_device .+ noise)
                effect_k_latent_device[:, s] = U_device * (diag_D_s .* innov_s_device)
            else # :cholesky or :cholesky_sparse
                Q_template_device = spec.hyper.Q_template # Already on device
                I_device = to_device(Matrix{Float64}(I, n_latent, n_latent)) # Create identity matrix on device
                Q_final_device = (1.0 - rho_s) .* I_device .+ rho_s .* Q_template_device
                F_device = cholesky(Symmetric(Q_final_device + noise * I_device))
                effect_k_latent_device[:, s] = sigma_s .* (F_device.U \ innov_s_device)
            end
        end
        
        # Index the reconstructed effects for the full observation set and move back to CPU
        indexed_effects_device = effect_k_latent_device[s_idx_full_device, :]
        push!(structured_effects, Array(indexed_effects_device))
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end

