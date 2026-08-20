"""
    Leroux <: ComponentModel

A component for a Leroux model, which is a proper Conditional Autoregressive (CAR)
model. It defines spatial correlation as a convex combination of a spatially
structured (ICAR) component and an unstructured (IID) component, controlled by a
single mixing parameter, `rho`.

# Version
v2.3.0 (2026-08-19)

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
  - `method`: `Symbol`, specifying the computational method. Default: `:spectral`.

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

"""
    get_precomputes(m::Leroux, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-dependent setup for the Leroux model. This version is CPU-only.
It pre-computes the ICAR precision matrix template and its spectral decomposition.
"""
function get_precomputes(m::Leroux, M::NamedTuple, mod_data::Dict)::NamedTuple
    s_N = get(M, :s_N, 0)
    W = get(M, :W, nothing)
    if s_N == 0 || isnothing(W)
        error(
            "Could not perform pre-computation for Leroux component because " *
            "spatial context (s_N and W) is missing."
        )
    end

    # build_structure_template returns CPU arrays
    template = build_structure_template(:icar, s_N; W=W)
    
    # All precomputed structures remain on the CPU.
    # Do not pre-compute Cholesky factor as it depends on `rho`.
    return (
        Q_template=template.matrix,
        U=template.U,
        L=template.L,
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

"""
    get_effects(m::Leroux, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the `Leroux` component's effect from posterior samples. This version
is CPU-only and uses modern chain accessors.
"""
function get_effects(
    m::Leroux, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = if occursin("FlexiChain", string(typeof(chain)))
        size(chain, 1) * FlexiChains.nchains(chain)
    else
        size(chain, 1) * size(chain, 3)
    end
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    noise = M.noise
    n_latent = spec.hyper.n_latent

    # --- Coordinate/Index Handling: Combine training and prediction sets on CPU ---
    s_idx_train = M.s_idx # Spatial indices for training data
    s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx) # If prediction set is provided
        vcat(s_idx_train, PS.data.s_idx) # Combine training and prediction indices
    else
        s_idx_train # Otherwise, use only training indices
    end
    N_total = length(s_idx_full) # Total number of observations (training + prediction)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        rho_name = _find_parameter(p_names, string(p_names_k.rho), k, is_multivariate_model)
        innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(rho_name) || isempty(innovations_name)
            @warn "Parameters for Leroux component $(spec.key) (outcome ) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        sigma_samples = get_params_vector(chain, sigma_name, 1) # (n_samples, 1)
        rho_samples = get_params_vector(chain, rho_name, 1) # (n_samples, 1)
        innovations_samples = get_params_matrix(chain, innovations_name, n_latent) # (n_samples, n_latent)
        
        # Initialize the output matrix for the full latent field
        effect_k_latent = zeros(Float64, n_latent, n_samples)

        # --- Sample-wise Reconstruction ---
        for s in 1:n_samples # Iterate over each posterior sample
            sigma_s = sigma_samples[s, 1] # Sigma for current sample
            rho_s = rho_samples[s, 1] # Rho for current sample
            innov_s = innovations_samples[s, :] # Innovations for current sample

            if m.method == :spectral
                U = spec.hyper.U
                L_eig = spec.hyper.L
                diag_D_s = sigma_s ./ sqrt.((1.0 - rho_s) .+ rho_s .* L_eig .+ noise)
                effect_k_latent[:, s] = U * (diag_D_s .* innov_s)
            else # :cholesky or :cholesky_sparse (use pre-computed dense Cholesky factor)
                Q_template = spec.hyper.Q_template
                I_mat = Matrix{Float64}(I, n_latent, n_latent)
                Q_final = (1.0 - rho_s) .* I_mat .+ rho_s .* Q_template
                F = cholesky(Symmetric(Q_final + noise * I_mat))
                effect_k_latent[:, s] = sigma_s .* (F.U \ innov_s)
            end
        end
        # Index the reconstructed effects for the full observation set to match observation indices
        indexed_effects = effect_k_latent[s_idx_full, :]
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end