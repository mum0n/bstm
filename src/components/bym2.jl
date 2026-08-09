
"""
    BYM2 <: ComponentModel

The Besag-York-Mollié 2 (BYM2) model, which provides an intuitive and
well-identified parameterization for spatial effects by separating them into a
structured (ICAR) and an unstructured (IID) component.

# Version
v1.9.4 (2026-08-08)

# Mathematical Summary
The BYM2 model decomposes a spatial random effect \$\\phi\$ into two independent
components: a structured spatial effect \$\\phi_{str}\$ and an unstructured (IID)
effect \$\\phi_{iid}\$. The final effect is a scaled combination of these two:
\$\\phi = \\sigma (\\sqrt{\\rho} \\cdot \\phi_{str}^* + \\sqrt{1-\\rho} \\cdot \\phi_{iid})\$
where:
- \$\\phi_{str}^*\$ is a scaled Intrinsic Conditional Autoregressive (ICAR) field,
  such that `Var(phi_str^*) = 1`.
- \$\\phi_{iid}\$ is standard Gaussian noise, \$\\phi_{iid} \\sim N(0, I)\$.
- \$\\sigma\$ is the overall marginal standard deviation of the total spatial effect.
- \$\\rho \\in [0, 1]\$ is a mixing parameter that controls the proportion of variance
  attributed to the structured spatial component.

This parameterization, proposed by Riebler et al. (2016), is preferred over the
original BYM model because it avoids confounding between the two spatial components,
leading to better MCMC convergence and more interpretable hyperparameters.

# Assumptions
- The spatial process can be decomposed into a locally smooth component and an
  uncorrelated noise component.
- The provided adjacency matrix `W` represents a single connected graph.

# Best Use Case
The standard and recommended model for disease mapping and general-purpose spatial
smoothing of areal data. It is more robust than a pure ICAR model as it can
account for both spatial clustering and localized heterogeneity.

# Key References
- Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H. (2016). An intuitive
  Bayesian spatial model for disease mapping that is exactly specified as a
  Gaussian Markov random field. *Statistical Methods in Medical Research*, 25(2),
  611-630.
- Wikipedia: Conditional autoregressive model

# Fields
- `rho::UnivariateDistribution`: The prior for the mixing parameter `rho`.
- `sigma::UnivariateDistribution`: The prior for the overall marginal standard
  deviation `sigma`.
- `method::Symbol`: The computational method for sampling the latent field. Can be
  `:spectral` (default, efficient, and AD-safe) or `:cholesky` (dense
  factorization, AD-safe but less efficient).
"""
struct BYM2 <: ComponentModel
    rho::UnivariateDistribution
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:bym2] = BYM2

COMPONENT_CONSTRUCTORS[:bym2] = (p, params) -> BYM2(
    p.rho, p.sigma, get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:bym2] = :spatial

"""
    get_datastructures!(m_type::Type{<:BYM2}, M::Dict, mod_data::Dict)::Bool

Establishes the spatial context for the BYM2 model. It ensures a spatial index
variable is provided and that a valid adjacency matrix `W` is available, then sets
the spatial index (`s_idx`) and number of spatial units (`s_N`) in the main
configuration `M`.

# Assumptions
- A base adjacency matrix `W` must be provided.
- A spatial index variable must be provided in the `random()` call.
"""
function get_datastructures!(m_type::Type{<:BYM2}, M::Dict, mod_data::Dict)::Bool
    data = M[:data]
    params = mod_data[:params]
    variables = mod_data[:variables]

    if isempty(variables)
        error(
            "The BYM2 model requires a spatial index variable, e.g., " *
            "`random(region, model=:bym2)`."
        )
    end

    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr || w_val isa Symbol
            calling_mod = get(M, :calling_module, Main)
            try
                M[:W] = Core.eval(calling_mod, w_val)
            catch e
                error(
                    "Could not evaluate `W` argument `$(w_val)` in BYM2 module. " *
                    "Error: $e"
                )
            end
        else
            M[:W] = w_val
        end
    end

    if !haskey(M, :W) || isnothing(M[:W])
        error("The BYM2 model requires an adjacency matrix `W`.")
    end

    s_var_sym = Symbol(variables[1])
    if !hasproperty(data, s_var_sym)
        error("Spatial index variable ':$s_var_sym' for BYM2 model not found in data.")
    end
    
    M[:s_idx] = data[!, s_var_sym]
    M[:s_N] = size(M[:W], 1)

    if M[:s_N] != length(unique(M[:s_idx]))
        @warn(
            "The number of spatial units implied by `W` ($(M[:s_N])) does not " *
            "match the number of unique levels in the index variable " *
            "`$(s_var_sym)` ($(length(unique(M[:s_idx])))). Ensure this is intentional."
        )
    end

    return true
end

"""
    get_precomputes(m::BYM2, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the structure matrix for the BYM2 component. The precision matrix
template is based on the ICAR structure. This function calls the central
`build_structure_template` utility to generate this matrix and its spectral
decomposition (`U`, `L`), which are essential for the AD-safe spectral sampling method.
"""
function get_precomputes(m::BYM2, M::NamedTuple, mod_data::Dict)::NamedTuple
    s_N = get(M, :s_N, 0)
    W = get(M, :W, nothing)

    if s_N == 0 || isnothing(W)
        error(
            "Could not perform pre-computation for BYM2 component " *
            "'$(mod_data[:key])' because spatial context (s_N and W) is missing."
        )
    end

    template = build_structure_template(:bym2, s_N; W=W)
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

"""
    get_priors(m::BYM2, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the BYM2 component's priors. This defines priors for
`sigma` and `rho`, and for the standard normal innovations for the structured
(`struct`) and unstructured (`iid`) components.
"""
function get_priors(
    m::BYM2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = size(spec.hyper.Q_template, 1)
    is_multivariate = (arch == "multivariate")
    is_shared = get(spec.params, :shared, false)
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ $(_distribution_to_string(m.sigma))")
        push!(priors_acc, "$(v.rho) ~ $(_distribution_to_string(m.rho))")
    end
    push!(priors_acc, "$(v.struct) ~ MvNormal(zeros($(n_latent)), I)")
    push!(priors_acc, "$(v.iid) ~ MvNormal(zeros($(n_latent)), I)")
    return join(priors_acc, "\n    ")
end

"""
    get_updates(m::BYM2, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the BYM2 component's update logic, dispatching on
the chosen `method` (`:spectral` or `:cholesky`). Both methods correctly implement
the Riebler parameterization.
"""
function get_updates(
    m::BYM2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "s_idx"

    if m.method == :spectral
        return """
        # --- BYM2 Spectral Assembly: $(spec.key) ---
        # This block constructs the BYM2 spatial effect using a non-centered
        # parameterization based on the spectral decomposition of the ICAR
        # precision matrix. This method is the default as it is efficient and
        # safe for automatic differentiation (AD).

        # 1. Construct the diagonal of the spectral transformation matrix D for
        #    the structured part. The structured part has precision Q_star, so
        #    its covariance has eigenvalues 1/L_j. The standard deviation is
        #    1/sqrt(L_j).
        local diag_D_structured = 1.0 ./ sqrt.(spec.hyper.L .+ M.noise)
        # The first eigenvalue is 0, corresponding to the null space (intercept).
        # Set its contribution to 0 to enforce the sum-to-zero constraint.
        diag_D_structured[1] = 0.0

        # 2. Apply the spectral transformation to the standard normal innovations.
        local structured_effect = spec.hyper.U * (diag_D_structured .* $(v.struct))

        # 3. Combine structured and unstructured components using the Riebler parameterization.
        $(v.latent) = $(v.sigma) .* (sqrt($(v.rho)) .* structured_effect .+
                                     sqrt(1.0 - $(v.rho)) .* $(v.iid))

        # 4. Add the final effect to the linear predictor.
        $(eta_target) .+= view($(v.latent), M.$(index_var))
        """
    elseif m.method == :cholesky
        return """
        # --- BYM2 Cholesky Assembly: $(spec.key) ---
        # This block constructs the BYM2 spatial effect using a dense Cholesky
        # decomposition of the static ICAR precision matrix. This method is
        # AD-safe but can be less efficient than the spectral method for large N.

        # 1. Get the pre-computed Cholesky factor of the ICAR precision matrix.
        local F_struct = spec.hyper.cholesky_factor
        
        # 2. Sample the structured component using the Cholesky factor.
        local structured_effect = F_struct.L' \\ $(v.struct)
        
        # 3. Apply a soft sum-to-zero constraint for identifiability.
        Turing.@addlogprob! logpdf(
            Normal(0.0, 0.001 * $(spec.hyper.n_latent)),
            sum(structured_effect)
        )
        
        # 4. Combine components using the Riebler parameterization.
        $(v.latent) = $(v.sigma) .* (sqrt($(v.rho)) .* structured_effect .+
                                     sqrt(1.0 - $(v.rho)) .* $(v.iid))
        
        # 5. Add the final effect to the linear predictor.
        $(eta_target) .+= view($(v.latent), M.$(index_var))
        """
    else
        error(
            "Unsupported method '$(m.method)' for BYM2 component. Supported " *
            "methods are :spectral and :cholesky."
        )
    end
end

"""
    get_effects(m::BYM2, chain, M::NamedTuple, n_samples, outcomes_N, spec, PS, N_total)::NamedTuple

Reconstructs the BYM2 component's effect from posterior samples. This function
re-runs the Riebler parameterization logic for each sample. The reconstruction uses
the spectral method for efficiency, which is valid regardless of the fitting method.
"""
function get_effects(
    m::BYM2, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    is_multivariate = outcomes_N > 1
    is_shared = get(spec.params, :shared, false)
    n_latent = size(spec.hyper.Q_template, 1)

    U = spec.hyper.U
    L_eig = spec.hyper.L
    T_el = eltype(chain.value)

    for k in 1:outcomes_N
        outcome_idx = is_multivariate ? k : nothing
        v = generate_full_variable_names(spec, M.model_arch, outcome_idx)
        
        sigma_var_name = (is_multivariate && is_shared) ?
            string(generate_full_variable_names(spec, M.model_arch, nothing).sigma) :
            string(v.sigma)
        rho_var_name = (is_multivariate && is_shared) ?
            string(generate_full_variable_names(spec, M.model_arch, nothing).rho) :
            string(v.rho)

        sigma_samples = get_params_vector(chain, sigma_var_name, 1)
        rho_samples = get_params_vector(chain, rho_var_name, 1)
        struct_samples = get_params_vector(chain, string(v.struct), n_latent)
        iid_samples = get_params_vector(chain, string(v.iid), n_latent)
        
        effect_k = Matrix{T_el}(undef, n_latent, n_samples)

        diag_D_structured = 1.0 ./ sqrt.(L_eig .+ M.noise)
        diag_D_structured[1] = 0.0

        for s in 1:n_samples
            sigma_s = sigma_samples[s, 1]
            rho_s = rho_samples[s, 1]
            struct_innov_s = struct_samples[s, :]
            iid_s = iid_samples[s, :]

            structured_effect_s = U * (diag_D_structured .* struct_innov_s)
            
            effect_k[:, s] = sigma_s .* (sqrt(rho_s) .* structured_effect_s .+
                                         sqrt(1.0 - rho_s) .* iid_s)
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
