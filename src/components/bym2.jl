"""
    BYM2 <: ComponentModel

The Besag-York-Mollié 2 (BYM2) model, which provides an intuitive and well-identified
parameterization for spatial effects by separating them into a structured (ICAR)
and an unstructured (IID) component.

# Version
v2.0.3 (2026-08-16)

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

    template = build_structure_template(:bym2, s_N; W=M.W)
    
    F = cholesky(Symmetric(Matrix(template.matrix) + M.noise * I))

    return (
        Q_template=template.matrix,
        U=template.U,
        L=template.L,
        scaling_factor=template.scaling_factor,
        n_latent=s_N,
        cholesky_factor=F
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
    push!(priors, "$(p_names.struct) ~ MvNormal(zeros($(n_latent)), I)")
    push!(priors, "$(p_names.iid) ~ MvNormal(zeros($(n_latent)), I)")
    
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
    m::BYM2, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, 
    p_names::Vector{String}, spec::NamedTuple, PS::Union{Nothing, NamedTuple}, 
    N_total::Int, is_multivariate_model::Bool
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    unstructured_effects = Vector{Matrix{Float64}}()
    noisy_effects = Vector{Matrix{Float64}}()
    
    hyper = spec.hyper
    n_latent = hyper.n_latent
    noise = M.noise
    
    s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, get(PS, :s_idx, []))

    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_name = _find_parameter(p_names, string(v.sigma), k, is_multivariate_model)
        rho_name = _find_parameter(
            p_names, string(v.unconstrained_rho), k, is_multivariate_model
        )
        struct_name = _find_parameter(p_names, string(v.struct), k, is_multivariate_model)
        iid_name = _find_parameter(p_names, string(v.iid), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(rho_name) || isempty(struct_name) || 
           isempty(iid_name)
            @warn "Parameters for BYM2 component $(spec.key) (outcome $k) not found. " *
                  "Returning zero-matrices."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            push!(unstructured_effects, zeros(Float64, N_total, n_samples))
            push!(noisy_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        rho_samples = logistic.(get_params_vector(chain, rho_name, 1)[:, 1])
        struct_samples = get_params_matrix(chain, struct_name, n_latent)
        iid_samples = get_params_matrix(chain, iid_name, n_latent)

        struct_effect = zeros(Float64, n_latent, n_samples)
        unstruct_effect = zeros(Float64, n_latent, n_samples)

        for s in 1:n_samples
            local struct_latent_s
            if m.method == :spectral
                U, L = hyper.U, hyper.L
                diag_D = 1.0 ./ sqrt.(L .+ noise)
                diag_D[1] = 0.0
                struct_latent_s = U * (diag_D .* struct_samples[s, :])
            else # cholesky or cholesky_sparse
                F = hyper.cholesky_factor
                struct_latent_s = F.L' \ struct_samples[s, :]
            end
            
            struct_latent_centered = struct_latent_s .- mean(struct_latent_s)
            
            struct_effect[:, s] = (sqrt(rho_samples[s]) .* struct_latent_centered) .* 
                                  sigma_samples[s]
            unstruct_effect[:, s] = (sqrt(1.0 - rho_samples[s]) .* iid_samples[s, :]) .* 
                                    sigma_samples[s]
        end
        
        push!(structured_effects, struct_effect[s_idx_full, :])
        push!(unstructured_effects, unstruct_effect[s_idx_full, :])
        push!(noisy_effects, (struct_effect .+ unstruct_effect)[s_idx_full, :])
    end
    
    return (structured=structured_effects, unstructured=unstructured_effects, 
            noisy=noisy_effects)
end
