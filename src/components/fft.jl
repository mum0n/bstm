"""
    FFT <: ComponentModel

A component model for a Fourier-based smoother. This component creates a basis of
sine and cosine functions at different frequencies. The effect is a linear
combination of these basis functions, with coefficients regularized by a random
walk prior to ensure smoothness.

# Version
v1.0.7 (2026-08-15)

# Mathematical Summary
The component models a smooth function \$f(x)\$ as a linear combination of Fourier
basis functions:
\$f(x) = \\sum_{j=1}^{M} \\left( \\beta_{s,j} \\sin\\left(\\frac{2\\pi j x}{\\ell}\\right) + \\beta_{c,j} \\cos\\left(\\frac{2\\pi j x}{\\ell}\\right) \\right)\$
where:
- \$M\$ is half the number of bins (`nbins`), representing the number of sine/cosine pairs.
- \$\\ell\$ is the `lengthscale` that controls the periodicity of the basis functions.
- \$\\beta_{s,j}\$ and \$\\beta_{c,j}\$ are the Fourier coefficients.

To ensure smoothness, a penalty is applied to the coefficients, typically a
second-order random walk (RW2) prior, which penalizes deviations from a linear
trend in the coefficient space. The coefficients are sampled from a Gaussian
Markov Random Field (GMRF) with a precision matrix derived from an RW2 penalty,
scaled by \$\\sigma^2\$.

# Computational Methods
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the RW2 penalty matrix. This method is fully compatible with
  automatic differentiation (AD) and thus suitable for gradient-based samplers
  like NUTS.
- `:cholesky` (AD-friendly): Uses a dense Cholesky factorization of the RW2
  penalty matrix. This method is also AD-compatible but can be less efficient
  than spectral decomposition for a large number of basis functions.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky
  factorization. This method is generally not AD-compatible for gradient-based
  samplers but is retained as a didactic alternative for use with gradient-free
  samplers.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `nbins`: `Int`, the total number of basis functions (sine/cosine pairs). Default: `20`.
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the Fourier
    coefficients. Default: `Exponential(1.0)`.
  - `lengthscale`: `UnivariateDistribution` or `Vector{<:UnivariateDistribution}`,
    prior for the lengthscale(s) controlling the periodicity. Default: `LogNormal(0, 1)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`).
    Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the Fourier coefficients.
- `ls_<key>`: The lengthscale(s) controlling the periodicity of the basis functions.
- `innovations_<key>`: The raw standard normal innovations for the Fourier coefficients.
- `latent_<key>`: The reconstructed latent smooth effect.
"""
struct FFT <: ComponentModel
    sigma::Distribution
    nbins::Int
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:fft] = FFT

COMPONENT_CONSTRUCTORS[:fft] = (p, params) -> FFT(
    p.sigma,
    get(params, :nbins, 20),
    get(p, :lengthscale, LogNormal(0.0, 1.0)), # Default prior for lengthscale
    get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:fft] = :smooth

"""
    get_precomputes(m::FFT, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs all data-dependent setup and pre-computation for the `FFT` component.
This function validates that coordinate variables exist, extracts them, and then
pre-computes the penalty matrix and its spectral decomposition for the Fourier
coefficients.
"""
function get_precomputes(m::FFT, M::NamedTuple, mod_data::Dict)::NamedTuple
    variables = mod_data[:variables]
    if isempty(variables)
        error("The FFT model requires coordinate variables, e.g., `random(x, model=:fft)`.")
    end

    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for FFT model not found in data.")
        end
    end

    coords = Matrix{Float64}(M.data[!, Symbol.(variables)])
    
    n_latent = m.nbins
    n_dims = size(coords, 2)

    # Determine nbins_per_dim for tensor product basis
    nbins_per_dim = fill(round(Int, n_latent^(1/n_dims)), n_dims)
    # Adjust if product is less than n_latent (e.g., due to rounding)
    while prod(nbins_per_dim) < n_latent
        nbins_per_dim[1] += 1
    end

    # The RW2 penalty is commonly used for Fourier coefficients to ensure smoothness
    template = build_structure_template(:rw2, n_latent)
    Q_template = template.matrix
    
    rank_deficiency = 2 # RW2 penalty has a rank deficiency of 2
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values
    scaling_factor = _compute_scaling_factor(L, rank_deficiency)
    
    Q_template_scaled = Q_template ./ scaling_factor
    L_scaled = L ./ scaling_factor

    # Pre-compute dense Cholesky factor for the :cholesky method
    F = cholesky(Symmetric(Matrix(Q_template_scaled) + M.noise * I))

    return (
        coords=coords,
        nbins_per_dim=nbins_per_dim,
        Q_template=Q_template_scaled,
        scaling_factor=scaling_factor,
        U=U,
        L=L_scaled,
        n_latent=n_latent,
        cholesky_factor=F
    )
end

"""
    get_priors(m::FFT, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `sigma`, `lengthscale` (`ls`), and the `innovations` coefficients.
"""
function get_priors(
    m::FFT, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = String[]
    push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")

    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors, "$(p_names.ls) ~ Product([$(ls_priors_str)])")
    else
        ls_prior_str = _distribution_to_string(m.lengthscale)
        push!(priors, "$(p_names.ls) ~ $(ls_prior_str)")
    end
    
    push!(
        priors,
        "$(p_names.innovations) ~ MvNormal(zeros($(spec.hyper.n_latent)), I)"
    )

    return join(priors, "\n    ")
end

"""
    bstm_fourier_basis(coords, nbins_per_dim, lengthscale)

Helper function to generate a tensor product Fourier basis matrix.
"""
function bstm_fourier_basis(
    coords::AbstractMatrix, nbins_per_dim::Vector{Int},
    lengthscale::Union{Real, AbstractVector}
)
    n_obs, n_dims = size(coords)
    
    ls_vec = if lengthscale isa Real
        fill(Float64(lengthscale), n_dims)
    else
        if length(lengthscale) != n_dims
            error("Length of lengthscale vector must match coordinate dimensions.")
        end
        lengthscale
    end

    basis_matrices_1D = []
    for i in 1:n_dims
        vals = coords[:, i]
        ls_val = ls_vec[i]
        t_coords = vals ./ ls_val
        n_basis_1d = nbins_per_dim[i]
        B_1d = zeros(eltype(t_coords), n_obs, n_basis_1d)
        
        idx = 1
        for m in 1:div(n_basis_1d, 2)
            arg = (2.0 * pi * m) .* t_coords
            B_1d[:, idx] = sin.(arg)
            B_1d[:, idx+1] = cos.(arg)
            idx += 2
        end
        if isodd(n_basis_1d) && idx <= n_basis_1d
            m = div(n_basis_1d, 2) + 1
            arg = (2.0 * pi * m) .* t_coords
            B_1d[:, idx] = sin.(arg)
        end
        push!(basis_matrices_1D, B_1d)
    end
    
    if isempty(basis_matrices_1D); return zeros(n_obs, 0); end

    # Combine 1D basis matrices using Kronecker product for multi-dimensional input
    B_final = basis_matrices_1D[1]
    for i in 2:n_dims
        B_next = basis_matrices_1D[i]
        n_obs_i, n_cols_final = size(B_final)
        _, n_cols_next = size(B_next)
        
        # Reshape for broadcasting to compute row-wise outer products
        B_final_reshaped = reshape(B_final, n_obs_i, n_cols_final, 1)
        B_next_reshaped = reshape(B_next, n_obs_i, 1, n_cols_next)
        
        tensor_prod = B_final_reshaped .* B_next_reshaped
        B_final = reshape(tensor_prod, n_obs_i, n_cols_final * n_cols_next)
    end
    
    return B_final
end

"""
    get_updates(m::FFT, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for constructing the `FFT` smooth effect. It supports
three methods for regularizing the Fourier coefficients.
"""
function get_updates(
    m::FFT, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent # Retrieve n_latent from spec.hyper
    
    common_basis_code = """
        B_fft = bstm_fourier_basis(
            spec_registry[:$(key)].hyper.coords,
            spec_registry[:$(key)].hyper.nbins_per_dim,
            $(p_names.ls)
        )
    """

    spectral_code = """
        # --- FFT Smoother Component (Spectral): $(key) ---
        let
            $(common_basis_code)
            
            diag_D = $(p_names.sigma) ./ sqrt.(spec_registry[:$(key)].hyper.L .+ M.noise)
            # Enforce sum-to-zero constraints for RW2 penalty
            diag_D[1] = 0.0
            diag_D[2] = 0.0
            
            coeffs = spec_registry[:$(key)].hyper.U * (diag_D .* $(p_names.innovations))
            $(p_names.latent) = B_fft * coeffs
            
            $(eta_target) .+= $(p_names.latent)
        end
    """

    cholesky_code = """
        # --- FFT Smoother Component (Cholesky, AD-Safe): $(key) ---
        let
            $(common_basis_code)
            
            Q_penalty = spec_registry[:$(key)].hyper.Q_template
            F = cholesky(Symmetric(Matrix(Q_penalty) + M.noise * I))
            
            coeffs_raw = F.L' \\ $(p_names.innovations)
            
            # Apply soft sum-to-zero constraints for RW2 penalty
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(coeffs_raw[1:2])
            )
            
            coeffs = $(p_names.sigma) .* coeffs_raw
            $(p_names.latent) = B_fft * coeffs
            
            $(eta_target) .+= $(p_names.latent)
        end
    """

    cholesky_sparse_code = """
        # --- FFT Smoother Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        # WARNING: This method is for didactic purposes and is NOT compatible with
        # automatic differentiation (e.g., NUTS sampler).
        let
            $(common_basis_code)
            
            Q_penalty = spec_registry[:$(key)].hyper.Q_template
            F = cholesky(Symmetric(Q_penalty + M.noise * I))
            
            coeffs_raw = F.L' \\ $(p_names.innovations)
            
            # Apply soft sum-to-zero constraints for RW2 penalty
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(coeffs_raw[1:2])
            )
            
            coeffs = $(p_names.sigma) .* coeffs_raw
            $(p_names.latent) = B_fft * coeffs
            
            $(eta_target) .+= $(p_names.latent)
        end
    """

    if m.method == :spectral
        return spectral_code
    elseif m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    else
        error("Unsupported method '$(m.method)' for FFT component.")
    end
end

"""
    get_effects(m::FFT, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_total, is_multivariate_model)

Reconstructs the `FFT` component's effect from posterior samples, dispatching on
the method used during sampling.
"""
function get_effects(
    m::FFT, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    p_names::Vector{String}, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, 
    N_total::Int, is_multivariate_model::Bool
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()

    hyper = spec.hyper
    noise = M.noise
    n_latent = hyper.n_latent
    nbins_per_dim = hyper.nbins_per_dim
    
    coords_train = hyper.coords
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        coords_train
    end

    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        ls_name = _find_parameter(p_names, string(p_names_k.ls), k, is_multivariate_model)
        innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(ls_name) || isempty(innovations_name)
            @warn "Parameters for FFT component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end
        
        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        ls_samples = get_params_matrix(chain, ls_name, m.lengthscale isa Vector ? length(m.lengthscale) : 1)
        innovations_samples = get_params_matrix(chain, innovations_name, n_latent)

        effect_k = zeros(Float64, size(coords_full, 1), n_samples)

        for i in 1:n_samples
            current_ls = if m.lengthscale isa Vector; ls_samples[i, :]; else ls_samples[i]; end
            
            B_fft_i = bstm_fourier_basis(coords_full, nbins_per_dim, current_ls)
            
            local coeffs
            if m.method == :spectral
                U = hyper.U
                L = hyper.L
                diag_D = sigma_samples[i] ./ sqrt.(L .+ noise)
                diag_D[1] = 0.0 # Enforce sum-to-zero constraints for RW2 penalty
                diag_D[2] = 0.0
                coeffs = U * (diag_D .* innovations_samples[i, :])
            else # :cholesky or :cholesky_sparse
                # For reconstruction, dense Cholesky is fine for both methods as AD is not involved here.
                F = hyper.cholesky_factor
                coeffs_raw = F.L' \ innovations_samples[i, :]
                # Apply sum-to-zero constraints for RW2 penalty
                coeffs_centered = coeffs_raw .- mean(coeffs_raw[1:2]) # Centering based on first two elements
                coeffs = sigma_samples[i] .* coeffs_centered
            end
            
            effect_k[:, i] = B_fft_i * coeffs
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
