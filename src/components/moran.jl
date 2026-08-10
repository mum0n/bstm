"""
    Moran <: ComponentModel

A component model for Moran's I Eigenvector Maps (MEM). This component decomposes
spatial autocorrelation into a set of orthogonal spatial patterns (eigenvectors)
derived from the Moran operator `(I - 11'/n)W(I - 11'/n)`. The effect is a linear
combination of these eigenvectors, providing a spectral basis for modeling spatial
processes.

# Version
v1.0.1 (2026-08-10)

# Mathematical Summary
The Moran component models a spatial field \$\\phi\$ as a linear combination of the
eigenvectors of the Moran operator \$\\mathbf{M}\$:
\$\\boldsymbol{\\phi} = \\mathbf{E} \\boldsymbol{\\beta}\$
where:
1.  \$\\mathbf{W}\$ is the spatial adjacency matrix.
2.  \$\\mathbf{H} = \\mathbf{I} - \\frac{1}{n}\\mathbf{1}\\mathbf{1}^T\$ is a centering matrix.
3.  The Moran operator is \$\\mathbf{M} = \\mathbf{HWH}\$.
4.  \$\\mathbf{E}\$ is the matrix whose columns are the eigenvectors of \$\\mathbf{M}\$.
5.  \$\\boldsymbol{\\beta}\$ is a vector of coefficients, which are given a hierarchical
    prior: \$\\beta_k \\sim \\mathcal{N}(0, \\sigma^2)\$.

# Computational Methods
- `:noncentered` (default): A non-centered parameterization where coefficients are
  constructed from standard normal innovations. Recommended for AD.
- `:centered`: A centered parameterization where coefficients are sampled directly
  from `N(0, sigma^2)`. Didactic, can be less efficient.

# Fields
- `sigma::Distribution`: The prior for the standard deviation of the coefficients.
- `method::Symbol`: The parameterization method, `:noncentered` or `:centered`.
"""
struct Moran <: ComponentModel
    sigma::Distribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:moran] = Moran
COMPONENT_CONSTRUCTORS[:moran] = (p, params) -> Moran(
    p.sigma, get(params, :method, :noncentered)
)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[:moran] = :spatial

"""
    get_datastructures!(m_type::Type{<:Moran}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Moran` component.
It ensures that an adjacency matrix `W` is provided and sets up the spatial context
(`s_idx`, `s_N`) in the main model configuration `M`.
"""
function get_datastructures!(m_type::Type{<:Moran}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    variables = mod_data[:variables]

    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr || w_val isa Symbol
            calling_mod = get(M, :calling_module, Main)
            try
                M[:W] = Core.eval(calling_mod, w_val)
            catch e
                error("Could not evaluate `W` argument `$(w_val)` for Moran. Error: $e")
            end
        else
            M[:W] = w_val
        end
    end

    if !haskey(M, :W) || !isa(M[:W], AbstractMatrix) || isempty(M[:W])
        error("Moran model requires a valid, non-empty adjacency matrix `W`.")
    end

    M[:s_N] = size(M[:W], 1)

    if isempty(variables)
        M[:s_idx] = collect(1:M[:s_N])
        @warn "Spatial index not provided for Moran. Assuming `s_idx = 1:s_N`."
    else
        s_var_sym = Symbol(variables[1])
        if !hasproperty(M[:data], s_var_sym)
            error("Spatial index ':$s_var_sym' for Moran not found in data.")
        end
        M[:s_idx] = M[:data][!, s_var_sym]
    end

    return true
end

"""
    get_precomputes(m::Moran, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `Moran` component.
It computes the Moran operator `M = (I - 11'/n)W(I - 11'/n)` and calculates its
eigenvectors, which serve as the spatial basis functions.
"""
function get_precomputes(m::Moran, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.s_N
    W = M.W

    # Create the centering matrix H = I - (1/n) * 1*1'
    H = I - (1/n) * ones(n, n)
    
    # Compute the Moran operator M = HWH
    W_mat = Matrix(W)
    moran_operator = H * W_mat * H
    
    # Compute the eigenvectors of the symmetric Moran operator
    eig_result = eigen(Symmetric(moran_operator))
    moran_eigenvectors = eig_result.vectors
    
    n_latent = size(moran_eigenvectors, 2)

    return (moran_eigenvectors=moran_eigenvectors, n_latent=n_latent)
end


"""
    get_priors(m::Moran, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `sigma`. For the `:noncentered` method, it also defines a
prior for the `raw` innovations.
"""
function get_priors(
    m::Moran, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = ["$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))"]

    if m.method == :noncentered
        push!(
            priors,
            "$(p_names.raw) ~ MvNormal(zeros($(spec.precomputes.n_latent)), I)"
        )
    end
    
    return join(priors, "\n    ")
end


"""
    get_updates(m::Moran, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code to construct the `Moran` effect, dispatching on the
chosen method.
"""
function get_updates(
    m::Moran, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    
    common_code = """
        local moran_eigenvectors = spec_registry[:$(key)].precomputes.moran_eigenvectors
    """

    noncentered_code = """
        # --- Moran Eigenvector Component (Non-Centered): $(key) ---
        let
            $(common_code)
            local scaled_coeffs = $(p_names.raw) .* $(p_names.sigma)
            local latent_field = moran_eigenvectors * scaled_coeffs
            $(eta_target) .+= view(latent_field, M.s_idx)
        end
    """

    centered_code = """
        # --- Moran Eigenvector Component (Centered): $(key) ---
        let
            $(common_code)
            local n_latent = spec_registry[:$(key)].precomputes.n_latent
            local coeffs ~ MvNormal(zeros(n_latent), $(p_names.sigma)^2 * I)
            local latent_field = moran_eigenvectors * coeffs
            $(eta_target) .+= view(latent_field, M.s_idx)
        end
    """

    if m.method == :noncentered
        return noncentered_code
    elseif m.method == :centered
        return centered_code
    else
        error("Unsupported method '$(m.method)' for Moran component.")
    end
end

"""
    get_effects(m::Moran, chain, M::NamedTuple, ...)

Reconstructs the `Moran` component's effect from posterior samples, dispatching
on the method used during sampling.
"""
function get_effects(
    m::Moran, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    eigenvectors = spec.precomputes.moran_eigenvectors
    n_latent = spec.precomputes.n_latent
    s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        effect_k = zeros(Float64, N_total, n_samples)
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]

        if m.method == :noncentered
            raw_samples = get_params_vector(chain, string(p_names.raw), n_latent)
            for i in 1:n_samples
                scaled_coeffs = raw_samples[i, :] .* sigma_samples[i]
                spatial_field = eigenvectors * scaled_coeffs
                effect_k[:, i] = view(spatial_field, s_idx_full)
            end
        else # :centered
            coeffs_samples = get_params_vector(chain, string(p_names.latent), n_latent)
            for i in 1:n_samples
                spatial_field = eigenvectors * coeffs_samples[i, :]
                effect_k[:, i] = view(spatial_field, s_idx_full)
            end
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
