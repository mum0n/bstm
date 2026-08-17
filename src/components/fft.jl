"""
    FFT <: ComponentModel

A component model for a Fourier-based smoother. This component creates a basis of
sine and cosine functions at different frequencies. The effect is a linear
combination of these basis functions, with coefficients regularized by a random
walk prior to ensure smoothness.

# Version
v1.1.0 (2026-08-17)

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
  penalty matrix, computed on-the-fly. This method is also AD-compatible.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky
  factorization, computed on-the-fly. This method is generally not AD-compatible.

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
coefficients, moving large arrays to the target device.
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

    # Get the device transfer function
    to_device = M.to_device

    # Ensure coords are on CPU for initial processing
    coords_cpu = Matrix{Float64}(M.data[!, Symbol.(variables)])
    
    n_latent = m.nbins
    n_dims = size(coords_cpu, 2)

    # Determine nbins_per_dim for tensor product basis
    nbins_per_dim = fill(round(Int, n_latent^(1/n_dims)), n_dims)
    while prod(nbins_per_dim) < n_latent
        nbins_per_dim[1] += 1
    end

    # The RW2 penalty is commonly used for Fourier coefficients to ensure smoothness
    # build_structure_template returns CPU arrays
    template = build_structure_template(:rw2, n_latent)
    Q_template_cpu = template.matrix
    
    rank_deficiency = 2 # RW2 penalty has a rank deficiency of 2
    eig_decomp = eigen(Symmetric(Matrix(Q_template_cpu)))
    U_cpu = eig_decomp.vectors
    L_cpu = eig_decomp.values
    scaling_factor = _compute_scaling_factor(L_cpu, rank_deficiency)
    
    Q_template_scaled_cpu = Q_template_cpu ./ scaling_factor
    L_scaled_cpu = L_cpu ./ scaling_factor

    # Move large, static arrays to the target device
    return (
        coords = to_device(coords_cpu),
        nbins_per_dim = nbins_per_dim,
        Q_template = to_device(Q_template_scaled_cpu),
        scaling_factor = scaling_factor,
        U = to_device(U_cpu),
        L = to_device(L_scaled_cpu),
        n_latent = n_latent
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
        "$(p_names.innovations) ~ MvNormal(zeros(T, $(spec.hyper.n_latent)), I)"
    )

    return join(priors, "\n    ")
end

"""
    bstm_fourier_basis(coords, nbins_per_dim, lengthscale)

Helper function to generate a tensor product Fourier basis matrix. This function
is compatible with GPU arrays.
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
        B_1d = similar(t_coords, n_obs, n_basis_1d)
        
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
    
    if isempty(basis_matrices_1D); return similar(coords, n_obs, 0); end

    # Combine 1D basis matrices using broadcasting for a row-wise Kronecker product
    B_final = basis_matrices_1D[1]
    for i in 2:n_dims
        B_next = basis_matrices_1D[i]
        n_obs_i, n_cols_final = size(B_final)
        _, n_cols_next = size(B_next)
        
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
    n_latent = spec.hyper.n_latent
    
    common_basis_code = """
        local B_fft = bstm_fourier_basis(
            spec_registry[:$(key)].hyper.coords,
            spec_registry[:$(key)].hyper.nbins_per_dim,
            $(p_names.ls)
        )
    """

    spectral_code = """
        # --- FFT Smoother Component (Spectral): $(key) ---
        let
            $(common_basis_code)
            local hyper = spec_registry[:$(key)].hyper
            
            local diag_D = $(p_names.sigma) ./ sqrt.(hyper.L .+ M.noise)
            # Enforce sum-to-zero constraints for RW2 penalty
            diag_D[1] = 0.0
            diag_D[2] = 0.0
            
            local coeffs = hyper.U * (diag_D .* $(p_names.innovations))
            $(p_names.latent) = B_fft * coeffs
            
            $(eta_target) .+= $(p_names.latent)
        end
    """

    cholesky_code = """
        # --- FFT Smoother Component (Cholesky, AD-Safe): $(key) ---
        let
            $(common_basis_code)
            local Q_penalty = spec_registry[:$(key)].hyper.Q_template
            local F = cholesky(Symmetric(Matrix(Q_penalty) + M.noise * I))
            
            local coeffs_raw = F.L' \\ $(p_names.innovations)
            
            # Apply soft sum-to-zero constraint for RW2 penalty
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(coeffs_raw)
            )
            
            local coeffs = $(p_names.sigma) .* coeffs_raw
            $(p_names.latent) = B_fft * coeffs
            
            $(eta_target) .+= $(p_names.latent)
        end
    """

    cholesky_sparse_code = """
        # --- FFT Smoother Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            $(common_basis_code)
            local Q_penalty = spec_registry[:$(key)].hyper.Q_template
            local F = cholesky(Symmetric(Q_penalty + M.noise * I))
            
            local coeffs_raw = F.L' \\ $(p_names.innovations)
            
            # Apply soft sum-to-zero constraint for RW2 penalty
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(coeffs_raw)
            )
            
            local coeffs = $(p_names.sigma) .* coeffs_raw
            $(p_names.latent) = B_fft * coeffs
            
            $(eta_target) .+= $(p_names.latent)
        end
    """

    if m.method == :spectral; return spectral_code;
    elseif m.method == :cholesky; return cholesky_code;
    elseif m.method == :cholesky_sparse; return cholesky_sparse_code;
    else; error("Unsupported method '$(m.method)' for FFT component."); end
end


"""
    get_effects(m::FFT, chain, spec, M, PS)

Reconstructs the `FFT` component's effect from posterior samples, dispatching on
the method used during sampling. Handles GPU arrays by moving sampled parameters
to the device for computation and moving the final results back to the CPU.
"""
function get_effects(
    m::FFT, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device
    
    hyper = spec.hyper
    noise = M.noise
    n_latent = hyper.n_latent
    nbins_per_dim = hyper.nbins_per_dim
    
    # --- Coordinate/Index Handling: Combine training and prediction sets on device ---
    coords_train_device = hyper.coords # Already on device
    coord_vars = get(spec.params, :positional_args, [])
    coords_full_device = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        coords_pred_cpu = Matrix{Float64}(PS.data[!, Symbol.(coord_vars)])
        vcat(coords_train_device, to_device(coords_pred_cpu))
    else
        coords_train_device
    end
    N_total = size(coords_full_device, 1)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        
        # Find parameter names in the MCMC chain
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        ls_name = _find_parameter(p_names, string(p_names_k.ls), k, is_multivariate_model)
        innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(ls_name) || isempty(innovations_name)
            @warn "Parameters for FFT component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end
        
        # Extract posterior samples (these are on the CPU)
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        ls_dim = m.lengthscale isa Vector ? length(m.lengthscale) : 1
        ls_samples_cpu = get_params_vector(chain, ls_name, ls_dim)
        innovations_samples_cpu = get_params_vector(chain, innovations_name, n_latent)

        # Initialize the output matrix for the full effect on the target device
        effect_k_device = to_device(zeros(Float64, N_total, n_samples))

        # --- Sample-wise Reconstruction on the Target Device ---
        for i in 1:n_samples
            current_ls_cpu = if m.lengthscale isa Vector; ls_samples_cpu[i, :]; else ls_samples_cpu[i, 1]; end
            
            # Generate Fourier basis matrix on the target device
            B_fft_i_device = bstm_fourier_basis(coords_full_device, nbins_per_dim, current_ls_cpu)
            
            # Move current sample's innovations to the device
            innov_i_device = to_device(innovations_samples_cpu[i, :])
            
            local coeffs_device
            if m.method == :spectral
                U_device = hyper.U
                L_device = hyper.L
                diag_D_device = sigma_samples_cpu[i] ./ sqrt.(L_device .+ noise)
                # Enforce sum-to-zero constraints for RW2 penalty
                diag_D_device[1] = 0.0
                diag_D_device[2] = 0.0
                coeffs_device = U_device * (diag_D_device .* innov_i_device)
            else # :cholesky or :cholesky_sparse
                # For reconstruction, we can re-compute the dense Cholesky factor on the device.
                Q_penalty_device = hyper.Q_template
                F_device = cholesky(Symmetric(Matrix(Q_penalty_device) + noise * I))
                coeffs_raw_device = F_device.L' \ innov_i_device
                # Apply sum-to-zero constraint for RW2 penalty
                coeffs_centered_device = coeffs_raw_device .- mean(coeffs_raw_device)
                coeffs_device = sigma_samples_cpu[i] .* coeffs_centered_device
            end
            
            # Compute the effect for this sample on the device
            effect_k_device[:, i] = B_fft_i_device * coeffs_device
        end
        
        # Move the final result matrix for this outcome back to the CPU
        push!(structured_effects, Array(effect_k_device))
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end