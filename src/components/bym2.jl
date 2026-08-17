"""
    BYM2 <: ComponentModel

The Besag-York-Mollié 2 (BYM2) model, which provides an intuitive and well-identified
parameterization for spatial effects by separating them into a structured (ICAR)
and an unstructured (IID) component.

# Version
v2.1.0 (2026-08-17)

# Mathematical Summary
The BYM2 model decomposes a spatial random effect \$\\boldsymbol{\\phi}\$ into two parts:
a spatially structured component \$\\boldsymbol{\\theta}\$ and an unstructured (IID) component \$\\boldsymbol{\\epsilon}\$:

\$\\boldsymbol{\\phi} = \\sigma \\left( \\sqrt{\\rho} \\boldsymbol{\\theta}_{scaled} + \\sqrt{1 - \\rho} \\boldsymbol{\\epsilon} \\right)\$

where:
- \$\\boldsymbol{\\theta}_{scaled}\$ is a scaled intrinsic CAR (ICAR) process with unit variance.
- \$\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(0, \\mathbf{I})\$ is IID Gaussian noise.
- \$\\rho \\in [0, 1]\$ is a mixing parameter controlling the proportion of variance attributed to the structured spatial effect. It is parameterized on an unconstrained scale via `unconstrained_rho`.
- \$\\sigma > 0\$ is the overall marginal standard deviation of the total spatial effect.

# Computational Methods
- `:spectral` (Default, AD-friendly): An efficient, AD-safe method using spectral decomposition of the ICAR precision matrix.
- `:cholesky` (AD-friendly): A didactic alternative using dense Cholesky factorization.
- `:cholesky_sparse` (Didactic, Not AD-friendly): A non-AD-safe didactic method using sparse Cholesky factorization.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@bstm`.
- **Optional (in `random()` call)**:
  - `unconstrained_rho`: `UnivariateDistribution`, prior for the unconstrained mixing parameter. Default: `Normal(0, 0.5)`.
  - `sigma`: `UnivariateDistribution`, prior for the marginal standard deviation. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`). Default: `:spectral`.

# Outputs (Parameter Names)
- `unconstrained_rho_<key>`: The unconstrained mixing parameter.
- `sigma_<key>`: The marginal standard deviation.
- `struct_<key>`: Raw standard normal innovations for the structured (ICAR) component.
- `iid_<key>`: Raw standard normal innovations for the unstructured (IID) component.
- `latent_<key>`: The reconstructed latent BYM2 effect.

# Key References
- Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H. (2016). *An intuitive joint prior for variance parameters in hierarchical models*. Statistical Science, 31(1), 114-135.
- Besag, J., York, J., & Mollié, A. (1991). *Bayesian image restoration, with applications in spatial statistics*. Annals of the Institute of Statistical Mathematics, 43(1), 1-20.
"""
struct BYM2 <: ComponentModel
    unconstrained_rho::UnivariateDistribution
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:bym2] = BYM2

COMPONENT_CONSTRUCTORS[:bym2] = (p, params) -> BYM2(
    get(p, :unconstrained_rho, Normal(0, 0.5)),
    p.sigma,
    get(params, :method, :spectral)
)
 

MODEL_TO_STRUCTURE_MAP[:bym2] = :spatial

function get_precomputes(m::BYM2, M::NamedTuple, mod_data::Dict)::NamedTuple
    if !hasproperty(M, :W) || !isa(M.W, AbstractMatrix) || isempty(M.W)
        error("BYM2 model requires a valid, non-empty adjacency matrix `W` " *
              "provided via keyword.")
    end

    s_N = size(M.W, 1)

    if !hasproperty(M, :s_idx)
        error("BYM2 component '$(mod_data[:key])' failed: s_idx not found in " *
              "model configuration.")
    end

    # Get the device transfer function
    to_device = M.to_device

    # build_structure_template returns CPU arrays
    template = build_structure_template(:bym2, s_N; W=M.W)
    
    # Move precomputed structures to the target device
    Q_template_device = to_device(template.matrix)
    U_device = to_device(template.U)
    L_device = to_device(template.L)
    
    # Pre-compute the dense Cholesky factor for the :cholesky method on the target device
    F_device = cholesky(Symmetric(Matrix(Q_template_device) + M.noise * I))

    return (
        Q_template=Q_template_device,
        U=U_device,
        L=L_device,
        scaling_factor=template.scaling_factor,
        n_latent=s_N,
        cholesky_factor=F_device
    )
end

function get_priors(
    m::BYM2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, 
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent

    priors = String[]
    push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")
    push!(priors, "$(p_names.unconstrained_rho) ~ " *
                    "$(_distribution_to_string(m.unconstrained_rho))")
    
    # Priors for the raw innovations for the structured and unstructured components.
    push!(priors, "$(p_names.struct) ~ MvNormal(zeros(T, $(n_latent)), I)")
    push!(priors, "$(p_names.iid) ~ MvNormal(zeros(T, $(n_latent)), I)")
    
    return join(priors, "\n    ")
end

function get_updates(
    m::BYM2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, 
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent

    spectral_code = """
        # --- BYM2 Component (Spectral): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            rho = logistic($(p_names.unconstrained_rho))
            
            # Construct the diagonal of the spectral transformation matrix D.
            diag_D_structured = 1.0 ./ sqrt.(hyper.L .+ M.noise)
            diag_D_structured[1] = 0.0 # Enforce sum-to-zero constraint

            # Apply the spectral transformation: latent = U * D * z
            structured_effect = hyper.U * (diag_D_structured .* $(p_names.struct))
            
            # Combine structured and unstructured components
            $(p_names.latent) = $(p_names.sigma) .* (sqrt(rho) .* structured_effect .+ 
                                sqrt(1.0 - rho) .* $(p_names.iid))
            
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """

    cholesky_code = """
        # --- BYM2 Component (Cholesky, AD-Safe): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            rho = logistic($(p_names.unconstrained_rho))
            F = hyper.cholesky_factor
            
            struct_latent_raw = F.L' \\ $(p_names.struct)
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_latent)), 
                                       sum(struct_latent_raw))
            
            $(p_names.latent) = $(p_names.sigma) .* (sqrt(rho) .* struct_latent_raw .+ 
                                sqrt(1.0 - rho) .* $(p_names.iid))
            
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """

    cholesky_sparse_code = """
        # --- BYM2 Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            rho = logistic($(p_names.unconstrained_rho))
            Q_penalty = hyper.Q_template
            F = cholesky(Symmetric(Q_penalty + M.noise * I))
            
            struct_latent_raw = F.L' \\ $(p_names.struct)
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_latent)), 
                                       sum(struct_latent_raw))
            
            $(p_names.latent) = $(p_names.sigma) .* (sqrt(rho) .* struct_latent_raw .+ 
                                sqrt(1.0 - rho) .* $(p_names.iid))
            
            $(eta_target) .+= view($(p_names.latent), M.s_idx)
        end
    """

    if m.method == :spectral; return spectral_code;
    elseif m.method == :cholesky; return cholesky_code;
    elseif m.method == :cholesky_sparse; return cholesky_sparse_code;
    else; error("Unsupported method '$(m.method)' for BYM2 component."); end
end


function get_effects(
    m::BYM2, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{Nothing, NamedTuple}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device
    noise = M.noise
    hyper = spec.hyper
    n_latent = hyper.n_latent

    # --- Coordinate/Index Handling: Combine training and prediction sets ---
    s_idx_train_device = M.s_idx # Already on device
    s_idx_full_device = if !isnothing(PS) && hasproperty(PS.data, :s_idx)
        s_idx_pred_cpu = get(PS.data, :s_idx, [])
        vcat(s_idx_train_device, to_device(s_idx_pred_cpu))
    else
        s_idx_train_device
    end
    N_total = length(s_idx_full_device)

    structured_effects = Vector{Matrix{Float64}}()
    unstructured_effects = Vector{Matrix{Float64}}()
    noisy_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        
        # Find parameter names in the MCMC chain
        sigma_name = _find_parameter(p_names, string(v.sigma), k, is_multivariate_model)
        rho_name = _find_parameter(p_names, string(v.unconstrained_rho), k, is_multivariate_model)
        struct_name = _find_parameter(p_names, string(v.struct), k, is_multivariate_model)
        iid_name = _find_parameter(p_names, string(v.iid), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(rho_name) || isempty(struct_name) || isempty(iid_name)
            @warn "Parameters for BYM2 component $(spec.key) (outcome $k) not found. Returning zero-matrices."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            push!(unstructured_effects, zeros(Float64, N_total, n_samples))
            push!(noisy_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        rho_samples_cpu = logistic.(get_params_vector(chain, rho_name, 1)[:, 1])
        struct_samples_cpu = get_params_matrix(chain, struct_name, n_latent)
        iid_samples_cpu = get_params_matrix(chain, iid_name, n_latent)

        # Initialize output matrices on the target device
        struct_effect_device = to_device(zeros(Float64, n_latent, n_samples))
        unstruct_effect_device = to_device(zeros(Float64, n_latent, n_samples))

        # --- Sample-wise Reconstruction on the Target Device ---
        for s in 1:n_samples
            # Move current sample's innovations to the device
            struct_innov_s = to_device(struct_samples_cpu[s, :])
            iid_innov_s = to_device(iid_samples_cpu[s, :])
            
            # Reconstruct the structured component
            local struct_latent_s
            if m.method == :spectral
                U_device, L_device = hyper.U, hyper.L # Already on device
                diag_D = 1.0 ./ sqrt.(L_device .+ noise)
                diag_D[1] = 0.0 # Enforce sum-to-zero constraint
                struct_latent_s = U_device * (diag_D .* struct_innov_s)
            else # :cholesky or :cholesky_sparse
                F_device = hyper.cholesky_factor # Already on device
                struct_latent_s = F_device.L' \ struct_innov_s
            end
            
            struct_latent_centered = struct_latent_s .- mean(struct_latent_s)
            
            # Combine components on the device
            sigma_s = sigma_samples_cpu[s] # CPU scalar
            rho_s = rho_samples_cpu[s]     # CPU scalar
            
            struct_effect_device[:, s] = (sqrt(rho_s) .* struct_latent_centered) .* sigma_s
            unstruct_effect_device[:, s] = (sqrt(1.0 - rho_s) .* iid_innov_s) .* sigma_s
        end
        
        # Indexing on the device and moving the final results to CPU
        push!(structured_effects, Array(struct_effect_device[s_idx_full_device, :]))
        push!(unstructured_effects, Array(unstruct_effect_device[s_idx_full_device, :]))
        
        total_effect_device = struct_effect_device .+ unstruct_effect_device
        push!(noisy_effects, Array(total_effect_device[s_idx_full_device, :]))
    end
    
    return (structured=structured_effects, unstructured=unstructured_effects, 
            noisy=noisy_effects)
end

