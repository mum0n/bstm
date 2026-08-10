"""
    Leroux <: ComponentModel

A component for a Leroux model, which is a proper Conditional Autoregressive (CAR)
model. It defines spatial correlation as a convex combination of a spatially
structured (ICAR) component and an unstructured (IID) component, controlled by a
single mixing parameter, `rho`.

# Version
v1.9.3 (2026-08-10)

# Mathematical Summary
The Leroux model is a proper CAR model, meaning its precision matrix is always
positive definite. It defines the precision matrix \$\\mathbf{Q}\$ as a convex
combination of an identity matrix \$\\mathbf{I}\$ and a scaled ICAR precision matrix
\$\\mathbf{Q}^*\$ (where \$\\mathbf{Q}^* = D - W\$):
\$\\mathbf{Q} = (1-\\rho)\\mathbf{I} + \\rho\\mathbf{Q}^*\$
This structure allows the model to smoothly interpolate between unstructured random
effects (\$\\rho=0\$) and a fully structured ICAR model (\$\\rho=1\$), providing a
flexible way to model spatial autocorrelation.

# Assumptions
- The provided adjacency matrix `W` represents a single connected graph.

# Best Use Case
Modeling structured spatial random effects for areal data, particularly when there
is uncertainty about the true strength of the spatial correlation. It is a robust
alternative to the `ICAR` or `BYM2` models as it avoids the rank-deficiency of
the ICAR model and has a more direct parameterization than BYM2.

# Key References
- Leroux, B. G., Lei, X., & Breslow, N. (2000). Estimation of disease rates in
  small areas: a new mixed model for spatial dependence. In *Statistical models
  in epidemiology, the environment, and clinical trials* (pp. 179-191). Springer.
- Wikipedia: Conditional autoregressive model

# Fields
- `rho::UnivariateDistribution`: The prior for the spatial correlation parameter,
  bounded on `[0, 1]`.
- `sigma::UnivariateDistribution`: The prior for the overall marginal standard
  deviation.
- `method::Symbol`: The computational method. Can be `:spectral` (default, AD-safe),
  `:cholesky` (AD-safe, dense), or `:cholesky_sparse` (didactic, not AD-safe).
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
    get_datastructures!(m_type::Type{<:Leroux}, M::Dict, mod_data::Dict)::Bool

Establishes the spatial context for the Leroux model. It ensures a spatial index
variable is provided and that a valid adjacency matrix `W` is available, then sets
the spatial index (`s_idx`) and number of spatial units (`s_N`) in the main
configuration `M`.
"""
function get_datastructures!(m_type::Type{<:Leroux}, M::Dict, mod_data::Dict)::Bool
    data = M[:data]
    params = mod_data[:params]
    variables = mod_data[:variables]

    if isempty(variables)
        error(
            "The Leroux model requires a spatial index variable, e.g., " *
            "`random(region, model=:leroux)`."
        )
    end

    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr || w_val isa Symbol
            calling_mod = get(M, :calling_module, Main)
            try
                M[:W] = Core.eval(calling_mod, w_val)
            catch e
                error("Could not evaluate `W` argument `$(w_val)`. Error: $e")
            end
        else
            M[:W] = w_val
        end
    end

    if !haskey(M, :W) || isnothing(M[:W])
        error("The Leroux model requires an adjacency matrix `W`.")
    end

    s_var_sym = Symbol(variables[1])
    if !hasproperty(data, s_var_sym)
        error("Spatial index variable ':$s_var_sym' not found in data.")
    end
    
    M[:s_idx] = data[!, s_var_sym]
    M[:s_N] = size(M[:W], 1)

    return true
end

"""
    get_precomputes(m::Leroux, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the structure matrix for the Leroux component. The precision matrix
template is based on the ICAR structure. This function calls the central
`build_structure_template` utility to generate this matrix and its spectral
decomposition (`U`, `L`), which are essential for the AD-safe spectral sampling method.
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

    template = build_structure_template(:leroux, s_N; W=W)
    return (
        Q_template=template.matrix,
        U=template.U,
        L=template.L,
        scaling_factor=template.scaling_factor,
        n_latent=s_N
    )
end

"""
    get_priors(m::Leroux, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the Leroux component's priors. This function creates
the code strings for sampling the hyperparameters `sigma` and `rho`, as well as
the standard normal innovations (`raw`) for the latent field.
"""
function get_priors(
    m::Leroux, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    is_multivariate = (arch == "multivariate")
    is_shared = get(spec.params, :shared, false)
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ $(_distribution_to_string(m.sigma))")
        push!(priors_acc, "$(v.rho) ~ $(_distribution_to_string(m.rho))")
    end
    push!(priors_acc, "$(v.raw) ~ MvNormal(zeros($(n_latent)), I)")
    return join(priors_acc, "\n    ")
end


"""
    get_updates(m::Leroux, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the Leroux component's update logic. It supports
three methods for sampling the latent field, controlled by `m.method`:

- `:spectral` (default): Uses a non-centered parameterization based on the
  spectral decomposition of the ICAR precision matrix. This method is highly
  efficient and AD-safe.
- `:cholesky`: Uses a non-centered parameterization based on the Cholesky
  decomposition of the full Leroux precision matrix. For AD-compatibility, the
  precision matrix is converted to a dense matrix before factorization.
- `:cholesky_sparse`: A didactic method using sparse Cholesky factorization. It is
  not compatible with gradient-based samplers (e.g., NUTS) but can be used with
  gradient-free methods.
"""
function get_updates(
    m::Leroux, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "s_idx"

    spectral_code = """
        # --- Leroux Spectral Assembly: $(spec.key) ---
        # This block constructs the Leroux spatial effect using a non-centered
        # parameterization based on the spectral decomposition of the ICAR
        # precision matrix. This is the default and recommended AD-safe method.
        
        diag_D = $(v.sigma) ./ sqrt.((1.0 .- $(v.rho)) .+ $(v.rho) .* spec_registry[:$(spec.key)].hyper.L .+ M.noise)
        $(v.latent) = spec_registry[:$(spec.key)].hyper.U * (diag_D .* $(v.raw))
        $(eta_target) .+= view($(v.latent), M.$(index_var))
        """

    cholesky_code = """
        # --- Leroux Cholesky Assembly (Dense, AD-Safe): $(spec.key) ---
        let
            local Q_template = spec_registry[:$(spec.key)].hyper.Q_template
            local rho_val = $(v.rho)
            local Q_final = (1.0 - rho_val) .* I(size(Q_template, 1)) .+ rho_val .* Q_template
            local F = cholesky(Symmetric(Matrix(Q_final) + M.noise * I))
            $(v.latent) = $(v.sigma) .* (F.U \\ $(v.raw))
            $(eta_target) .+= view($(v.latent), M.$(index_var))
        end
        """

    cholesky_sparse_code = """
        # --- Leroux Cholesky Assembly (Sparse, Not AD-Safe): $(spec.key) ---
        # WARNING: This method is for didactic purposes and is NOT compatible with
        # automatic differentiation (e.g., NUTS sampler).
        let
            local Q_template = spec_registry[:$(spec.key)].hyper.Q_template
            local rho_val = $(v.rho)
            local Q_final = (1.0 - rho_val) .* sparse(I, size(Q_template)...) .+ rho_val .* Q_template
            local F = cholesky(Symmetric(Q_final + M.noise * I))
            $(v.latent) = $(v.sigma) .* (F.U \\ $(v.raw))
            $(eta_target) .+= view($(v.latent), M.$(index_var))
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
            "Unsupported method '$(m.method)' for Leroux component. Supported " *
            "methods are :spectral, :cholesky, and :cholesky_sparse."
        )
    end
end



"""
    get_effects(m::Leroux, chain, M::NamedTuple, ...)

Reconstructs the Leroux component's effect from posterior samples. This function
is updated to dispatch on the `method` used during sampling to ensure the
reconstruction logic is consistent with the model definition.
"""
function get_effects(
    m::Leroux, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    is_multivariate = outcomes_N > 1
    is_shared = get(spec.params, :shared, false)
    n_latent = spec.hyper.n_latent
    
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
        raw_samples = get_params_vector(chain, string(v.raw), n_latent)
        
        effect_k = Matrix{Float64}(undef, n_latent, n_samples)

        for s in 1:n_samples
            sigma_s = sigma_samples[s, 1]
            rho_s = rho_samples[s, 1]
            raw_s = raw_samples[s, :]

            if m.method == :spectral
                U = spec.hyper.U
                L_eig = spec.hyper.L
                diag_D_s = sigma_s ./ sqrt.((1.0 - rho_s) .+ rho_s .* L_eig .+ M.noise)
                effect_k[:, s] = U * (diag_D_s .* raw_s)
            else # :cholesky or :cholesky_sparse
                Q_template = spec.hyper.Q_template
                Q_final = (1.0 - rho_s) .* I(n_latent) .+ rho_s .* Q_template
                F = cholesky(Symmetric(Matrix(Q_final) + M.noise * I))
                effect_k[:, s] = sigma_s .* (F.U \ raw_s)
            end
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
