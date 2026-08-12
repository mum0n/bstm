"""
    BYM2 <: ComponentModel

The Besag-York-Mollié 2 (BYM2) model, which provides an intuitive and
well-identified parameterization for spatial effects by separating them into a
structured (ICAR) and an unstructured (IID) component.

# Version
v2.1.0 (2026-08-12)

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

# Computational Methods
- `:spectral` (Default, AD-friendly): Constructs the structured component using a
  spectral decomposition of the ICAR precision matrix. Recommended for NUTS.
- `:cholesky` (AD-friendly): Uses a pre-computed dense Cholesky factorization.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization,
  which is not compatible with most AD backends.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@bstm`.
- **Optional (in `random()` call)**:
  - `rho`: A `Distribution` for the prior on the mixing parameter. Default: `Beta(1,1)`.
  - `sigma`: A `Distribution` for the prior on the overall standard deviation. Default: `Exponential(1.0)`.
  - `method`: A `Symbol` specifying the computational method. Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The overall marginal standard deviation.
- `rho_<key>`: The mixing parameter.
- `struct_innovations_<key>`: The raw standard normal innovations for the structured component.
- `iid_innovations_<key>`: The raw standard normal innovations for the unstructured component.

# Key References
- Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H. (2016). *An intuitive Bayesian spatial model with two hyperparameters*. Statistical Methods in Medical Research, 25(2), 1145-1160.
- Wikipedia: Besag-York-Mollié model
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

function get_priors(
    m::BYM2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    is_multivariate = (arch == "multivariate")
    is_shared = get(spec.params, :shared, false)
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))

    struct_innov_name = "struct_innovations_$(spec.key)"
    iid_innov_name = "iid_innovations_$(spec.key)"
    if is_multivariate && !is_shared
        struct_innov_name *= "_$(outcome_idx)"
        iid_innov_name *= "_$(outcome_idx)"
    end

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")
        push!(priors_acc, "$(p_names.rho) ~ $(_distribution_to_string(m.rho))")
    end
    push!(priors_acc, "$(struct_innov_name) ~ MvNormal(zeros(T, $(n_latent)), I)")
    push!(priors_acc, "$(iid_innov_name) ~ MvNormal(zeros(T, $(n_latent)), I)")
    return join(priors_acc, "\n    ")
end

function get_updates(
    m::BYM2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "s_idx"
    key = spec.key
    n_latent = spec.hyper.n_latent

    is_multivariate = (arch == "multivariate")
    is_shared = get(spec.params, :shared, false)
    struct_innov_name = "struct_innovations_$(spec.key)"
    iid_innov_name = "iid_innovations_$(spec.key)"
    if is_multivariate && !is_shared
        struct_innov_name *= "_$(outcome_idx)"
        iid_innov_name *= "_$(outcome_idx)"
    end

    spectral_code = """
        # --- BYM2 Spectral Assembly: $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            diag_D_structured = 1.0 ./ sqrt.(hyper.L .+ M.noise)
            diag_D_structured[1] = 0.0
            structured_effect = hyper.U * (diag_D_structured .* $(struct_innov_name))
            
            $(p_names.latent) = $(p_names.sigma) .* (sqrt($(p_names.rho)) .* structured_effect .+
                                         sqrt(1.0 - $(p_names.rho)) .* $(iid_innov_name))
            
            $(eta_target) .+= view($(p_names.latent), M.$(index_var))
        end
        """

    cholesky_code = """
        # --- BYM2 Cholesky Assembly (Dense, AD-Safe): $(key) ---
        let
            F_struct = spec_registry[:$(key)].hyper.cholesky_factor
            structured_effect = F_struct.L' \\ $(struct_innov_name)
            
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(structured_effect)
            )
            
            $(p_names.latent) = $(p_names.sigma) .* (sqrt($(p_names.rho)) .* structured_effect .+
                                         sqrt(1.0 - $(p_names.rho)) .* $(iid_innov_name))
            
            $(eta_target) .+= view($(p_names.latent), M.$(index_var))
        end
        """

    cholesky_sparse_code = """
        # --- BYM2 Cholesky Assembly (Sparse, Not AD-Safe): $(key) ---
        let
            Q_template = spec_registry[:$(key)].hyper.Q_template
            F_struct = cholesky(Symmetric(Q_template + M.noise * I))
            structured_effect = F_struct.L' \\ $(struct_innov_name)
            
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(structured_effect)
            )
            
            $(p_names.latent) = $(p_names.sigma) .* (sqrt($(p_names.rho)) .* structured_effect .+
                                         sqrt(1.0 - $(p_names.rho)) .* $(iid_innov_name))
            
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
        error(
            "Unsupported method '$(m.method)' for BYM2 component. Supported " *
            "methods are :spectral, :cholesky, and :cholesky_sparse."
        )
    end
end

function get_effects(
    m::BYM2, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    unstructured_effects = Vector{Matrix{Float64}}()
    noisy_effects = Vector{Matrix{Float64}}()
    
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))
    n_latent = spec.hyper.n_latent
    noise = M.noise
    s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)

    for k in 1:outcomes_N
        sigma_name = _find_parameter(p_names_vec, string(spec.key), "sigma", k, is_multivariate_model)
        rho_name = _find_parameter(p_names_vec, string(spec.key), "rho", k, is_multivariate_model)
        struct_innov_name = _find_parameter(p_names_vec, string(spec.key), "struct_innovations", k, is_multivariate_model)
        iid_innov_name = _find_parameter(p_names_vec, string(spec.key), "iid_innovations", k, is_multivariate_model)

        if isempty(sigma_name) || isempty(rho_name) || isempty(struct_innov_name) || isempty(iid_innov_name)
            @warn "Parameters for BYM2 component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            push!(unstructured_effects, zeros(Float64, N_total, n_samples))
            push!(noisy_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        rho_samples = get_params_vector(chain, rho_name, 1)[:, 1]
        struct_innov_samples = get_params_vector(chain, struct_innov_name, n_latent)
        iid_samples = get_params_vector(chain, iid_innov_name, n_latent)
        
        struct_effect_k = zeros(Float64, n_latent, n_samples)
        unstruct_effect_k = zeros(Float64, n_latent, n_samples)

        for s in 1:n_samples
            sigma_s = sigma_samples[s]
            rho_s = rho_samples[s]
            
            local structured_effect_s
            if m.method == :spectral
                U = spec.hyper.U
                L_eig = spec.hyper.L
                diag_D_structured = 1.0 ./ sqrt.(L_eig .+ noise)
                diag_D_structured[1] = 0.0
                structured_effect_s = U * (diag_D_structured .* struct_innov_samples[s, :])
            else # :cholesky or :cholesky_sparse
                F_struct = spec.hyper.cholesky_factor
                structured_effect_raw = F_struct.L' \ struct_innov_samples[s, :]
                structured_effect_s = structured_effect_raw .- mean(structured_effect_raw)
            end
            
            struct_effect_k[:, s] = sigma_s .* sqrt(rho_s) .* structured_effect_s
            unstruct_effect_k[:, s] = sigma_s .* sqrt(1.0 - rho_s) .* iid_samples[s, :]
        end
        
        push!(structured_effects, struct_effect_k[s_idx_full, :])
        push!(unstructured_effects, unstruct_effect_k[s_idx_full, :])
        push!(noisy_effects, (struct_effect_k .+ unstruct_effect_k)[s_idx_full, :])
    end
    
    return (structured=structured_effects, unstructured=unstructured_effects, noisy=noisy_effects)
end
