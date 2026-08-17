"""
    Moran <: ComponentModel

A component model for Moran's I Eigenvector Maps (MEM). This component decomposes
spatial autocorrelation into a set of orthogonal spatial patterns (eigenvectors)
derived from the Moran operator \$(I - 11'/n)W(I - 11'/n)\$. The effect is a linear
combination of these eigenvectors, providing a spectral basis for modeling spatial
processes.

# Version
v1.1.1 (2026-08-14)

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
- `:noncentered` (Default, AD-friendly): A non-centered parameterization where coefficients are
  constructed from standard normal innovations. Recommended for gradient-based samplers.
- `:centered` (Didactic, Not AD-friendly): A centered parameterization where coefficients are sampled directly
  from `N(0, sigma^2)`. This can be less efficient for MCMC.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@bstm`.
- **Optional (in `random()` call)**:
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the coefficients. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:noncentered` or `:centered`). Default: `:noncentered`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the eigenvector coefficients.
- `innovations_<key>`: The raw standard normal innovations for the coefficients (for `:noncentered`).
- `latent_<key>`: The latent coefficients (for `:centered`).

# Key References
- Griffith, D. A. (2003). *Spatial autocorrelation and spatial filtering: gaining
  understanding through theory and practice*. Springer Science & Business Media.
"""
struct Moran <: ComponentModel
    sigma::Distribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:moran] = Moran
COMPONENT_CONSTRUCTORS[:moran] = (p, params) -> Moran(
    p.sigma, get(params, :method, :noncentered)
)

MODEL_TO_STRUCTURE_MAP[:moran] = :spatial

function get_precomputes(m::Moran, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.s_N
    W = M.W

    H = I - (1/n) * ones(n, n)
    W_mat = Matrix(W)
    moran_operator = H * W_mat * H
    
    eig_result = eigen(Symmetric(moran_operator))
    moran_eigenvectors = eig_result.vectors
    
    n_latent = size(moran_eigenvectors, 2)

    return (moran_eigenvectors=moran_eigenvectors, n_latent=n_latent)
end

function get_priors(
    m::Moran, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = ["$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))"]

    if m.method == :noncentered
        push!(
            priors,
            "$(p_names.innovations) ~ DynamicPPL.NamedDist(MvNormal(zeros(T, spec.hyper.n_latent), I), :$(p_names.innovations))"
        )
    end
    
    return join(priors, "\n    ")
end

function get_updates(
    m::Moran, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent
    
    common_code = """
        moran_eigenvectors = spec_registry[:$(key)].hyper.moran_eigenvectors
    """

    noncentered_code = """
        # --- Moran Eigenvector Component (Non-Centered): $(key) ---
        let
            $(common_code)
            scaled_coeffs = $(p_names.innovations) .* $(p_names.sigma)
            latent_field = moran_eigenvectors * scaled_coeffs
            $(eta_target) .+= view(latent_field, M.s_idx)
        end
    """

    centered_code = """
        # --- Moran Eigenvector Component (Centered): $(key) ---
        let
            $(common_code)
            $(p_names.latent) ~ MvNormal(zeros(T, $(n_latent)), $(p_names.sigma)^2 * I)
            latent_field = moran_eigenvectors * $(p_names.latent)
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
    get_effects(m::Moran, chain, spec, M, PS)

Reconstructs the Moran eigenvector effect from posterior samples. This version is
updated to handle GPU arrays by moving sampled parameters to the device for
computation and moving the final results back to the CPU. It also uses a more
efficient vectorized approach for reconstruction.
"""
function get_effects(
    m::Moran, chain::Chains, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device

    structured_effects = Vector{Matrix{Float64}}()
    
    eigenvectors = spec.hyper.moran_eigenvectors # This is already on the correct device
    n_latent = spec.hyper.n_latent

    # --- Index Handling: Combine training and prediction sets on device ---
    s_idx_train = M.s_idx # Already on device
    s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx)
        s_idx_pred_cpu = get(PS.data, :s_idx, [])
        vcat(s_idx_train, to_device(s_idx_pred_cpu))
    else
        s_idx_train
    end
    N_total = length(s_idx_full)

    # --- Reconstruction Loop ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        
        if isempty(sigma_name)
            @warn "Sigma parameter for Moran component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Samples are always on CPU
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        
        local latent_field_device
        if m.method == :noncentered
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            if isempty(innovations_name)
                @warn "Innovations for Moran component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            innovations_samples_cpu = get_params_matrix(chain, innovations_name, n_latent)
            
            # Move to device for computation
            innovations_device = to_device(innovations_samples_cpu)
            sigma_device = to_device(sigma_samples_cpu)
            
            # Vectorized reconstruction on device
            scaled_coeffs_device = innovations_device' .* sigma_device'
            latent_field_device = eigenvectors * scaled_coeffs_device

        else # :centered
            latent_name = _find_parameter(p_names, string(p_names_k.latent), k, is_multivariate_model)
            if isempty(latent_name)
                @warn "Latent coefficients for Moran component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            coeffs_samples_cpu = get_params_matrix(chain, latent_name, n_latent)
            
            # Move to device for computation
            coeffs_device = to_device(coeffs_samples_cpu)
            
            # Vectorized reconstruction on device
            latent_field_device = eigenvectors * coeffs_device'
        end
        
        # Indexing on the device
        effect_k_device = latent_field_device[s_idx_full, :]
        
        # Move final result back to CPU
        push!(structured_effects, Array(effect_k_device))
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end

