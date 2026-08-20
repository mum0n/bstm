"""
    Wavelet <: ComponentModel

A component for a wavelet-based smoother. This component creates a basis of
wavelet functions at different scales and locations across the covariate space. The
effect is a linear combination of these basis functions, with coefficients regularized by
a random walk prior to ensure smoothness.

# Version
v1.1.1 (2026-08-14)

# Mathematical Summary
The wavelet smoother models a function \$f(x)\$ as a linear combination of scaled and
translated mother wavelets \$\\psi\$:
\$f(x) = \\sum_{j,k} \\beta_{jk} \\psi_{jk}(x)\$
where \$\\psi_{jk}(x) = 2^{j/2} \\psi(2^j x - k)\$ represents a wavelet at scale \$j\$ and
location \$k\$. The basis functions \$B_m(x)\$ are generated from a chosen mother
wavelet family (e.g., Daubechies 'db4'). The effect is then:
\$f(x) = \\sum_{m=1}^{M} \\beta_m B_m(x)\$

The coefficients \$\\boldsymbol{\\beta}\$ are given a smoothing prior, typically a
second-order random walk (RW2), to regularize the function:
\$\\boldsymbol{\\beta} \\sim \\mathcal{N}(\\mathbf{0}, (\\tau \\mathbf{Q}_{RW2})^{-1})\$

# Computational Methods
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the RW2 penalty matrix. Recommended for gradient-based samplers.
- `:cholesky` (AD-friendly): Uses a pre-computed dense Cholesky factorization of the
  penalty matrix.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization,
  which is not compatible with most AD backends.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `nbins`: `Int`, the total number of basis functions (wavelets) to generate. Default: `32`.
  - `family`: `Symbol`, the wavelet family to use (e.g., `:db4`, `:haar`). Default: `:db4`.
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the wavelet coefficients. Default: `Exponential(1.0)`.
  - `lengthscale`: `UnivariateDistribution` or `Vector{<:UnivariateDistribution}`, prior for the
    lengthscale(s), which control the dilation of the wavelets. Default: `Gamma(2, 0.5)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`).
    Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the wavelet coefficients.
- `ls_<key>`: The lengthscale(s) controlling the wavelet dilation.
- `innovations_<key>`: The raw standard normal innovations for the coefficients.
- `latent_<key>`: The final smooth effect vector.

# Key References
- Nason, G. P. (2008). *Wavelet Methods in Statistics with R*. Springer.
- Daubechies, I. (1992). *Ten Lectures on Wavelets*. SIAM.
"""
struct Wavelet <: ComponentModel
    family::Symbol
    nbins::Int
    sigma::Distribution
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:wavelet] = Wavelet

COMPONENT_CONSTRUCTORS[:wavelet] = (p, params) -> Wavelet(
    get(params, :family, :db4),
    get(params, :nbins, 32),
    p.sigma,
    p.lengthscale,
    get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:wavelet] = :smooth

function get_precomputes(m::Wavelet, M::NamedTuple, mod_data::Dict)::NamedTuple
    # Data validation moved from get_datastructures!
    variables = mod_data[:variables]
    if isempty(variables)
        error("Wavelet model requires coordinate variables, e.g., `random(x, model=:wavelet)`.")
    end

    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Wavelet model not found in data.")
        end
    end

    coords = Matrix{Float64}(M.data[!, Symbol.(variables)])
    
    n_latent = m.nbins
    n_dims = size(coords, 2)

    nbins_per_dim = fill(round(Int, n_latent^(1/n_dims)), n_dims)
    while prod(nbins_per_dim) < n_latent
        nbins_per_dim[1] += 1
    end

    template = build_structure_template(:rw2, n_latent)
    Q_template = template.matrix
    
    rank_deficiency = 2 # RW2 penalty
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values
    scaling_factor = _compute_scaling_factor(L, rank_deficiency)
    
    Q_template_scaled = Q_template ./ scaling_factor
    L_scaled = L ./ scaling_factor

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

function get_priors(
    m::Wavelet, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    key = spec.key
    
    priors = String[]
    push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")

    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors, "$(p_names.ls) ~ Product([$(ls_priors_str)])")
    else
        ls_prior_str = _distribution_to_string(m.lengthscale)
        push!(priors, "$(p_names.ls) ~ $(ls_prior_str)")
    end
    
    push!(priors, "$(p_names.ure) ~ MvNormal(zeros(T, spec_registry[:$(key)].hyper.n_latent), I)")

    return join(priors, "\n    ")
end

function get_updates(
    m::Wavelet, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    
    common_basis_code = """
        hyper = spec_registry[:$(key)].hyper
        B_wavelet = bstm_tensor_product_wavelet_basis(
            hyper.coords,
            hyper.nbins_per_dim,
            Symbol("$(m.family)"),
            $(p_names.ls)
        )
    """

    spectral_code = """
        # --- Wavelet Smoother Component (Spectral): $(key) ---
        let
            $(common_basis_code)
            diag_D = $(p_names.sigma) ./ sqrt.(hyper.L .+ M.noise)
            diag_D[1] = 0.0; diag_D[2] = 0.0
            coeffs = hyper.U * (diag_D .* $(p_names.ure))
            $(p_names.sre) = B_wavelet * coeffs
            $(eta_target) .+= $(p_names.sre)
        end
    """

    cholesky_code = """
        # --- Wavelet Smoother Component (Cholesky, AD-Safe): $(key) ---
        let
            $(common_basis_code)
            F = hyper.cholesky_factor
            coeffs_unscaled = F.L' \\ $(p_names.ure)
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * hyper.n_latent), sum(coeffs_unscaled))
            coeffs = $(p_names.sigma) .* (coeffs_unscaled .- mean(coeffs_unscaled))
            $(p_names.sre) = B_wavelet * coeffs
            $(eta_target) .+= $(p_names.sre)
        end
    """

    cholesky_sparse_code = """
        # --- Wavelet Smoother Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            $(common_basis_code)
            Q_penalty = hyper.Q_template
            F = cholesky(Symmetric(Q_penalty + M.noise * I))
            coeffs_unscaled = F.L' \\ $(p_names.ure)
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * hyper.n_latent), sum(coeffs_unscaled))
            coeffs = $(p_names.sigma) .* coeffs_unscaled
            $(p_names.sre) = B_wavelet * coeffs
            $(eta_target) .+= $(p_names.sre)
        end
    """

    if m.method == :spectral; return spectral_code;
    elseif m.method == :cholesky; return cholesky_code;
    elseif m.method == :cholesky_sparse; return cholesky_sparse_code;
    else; error("Unsupported method '$(m.method)' for Wavelet component."); end
end

function get_effects(
    m::Wavelet, chain, spec::NamedTuple, M::NamedTuple,
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
    
    hyper = spec.hyper
    noise = M.noise
    n_latent = hyper.n_latent
    nbins_per_dim = hyper.nbins_per_dim

    # --- Coordinate Handling: Combine training and prediction sets on CPU ---
    coords_train_cpu = hyper.coords
    coord_vars = get(spec.params, :positional_args, [])
    coords_full_cpu = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train_cpu, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        coords_train_cpu
    end
    N_total = size(coords_full_cpu, 1)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        ls_name = _find_parameter(p_names, string(p_names_k.ls), k, is_multivariate_model)
        ure_name = _find_parameter(p_names, string(p_names_k.ure), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(ls_name) || isempty(ure_name)
            @warn "Parameters for Wavelet component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        ls_dim = m.lengthscale isa Vector ? length(m.lengthscale) : 1
        ls_samples_cpu = get_params_matrix(chain, ls_name, ls_dim)
        ure_samples_cpu = get_params_matrix(chain, ure_name, n_latent)

        # Initialize the output matrix for the full effect on the CPU
        effect_k_cpu = zeros(Float64, N_total, n_samples)

        # --- Sample-wise Reconstruction ---
        for i in 1:n_samples
            # 1. Generate basis matrix on CPU
            current_ls_cpu = ls_dim > 1 ? ls_samples_cpu[i, :] : ls_samples_cpu[i, 1]
            B_wavelet_i_cpu = bstm_tensor_product_wavelet_basis(
                coords_full_cpu, nbins_per_dim, m.family, current_ls_cpu
            )
            
            innov_i_cpu = ure_samples_cpu[i, :]
            sigma_i_cpu = sigma_samples_cpu[i]
            
            # 3. Reconstruct coefficients on CPU
            local coeffs_cpu
            if m.method == :spectral
                U = hyper.U
                L = hyper.L
                diag_D = sigma_i_cpu ./ sqrt.(L .+ noise)
                diag_D[1] = 0.0; diag_D[2] = 0.0
                coeffs_cpu = U * (diag_D .* innov_i_cpu)
            else # :cholesky or :cholesky_sparse
                F = hyper.cholesky_factor
                coeffs_unscaled = F.L' \ innov_i_cpu
                coeffs_centered = coeffs_unscaled .- mean(coeffs_unscaled)
                coeffs_cpu = sigma_i_cpu .* coeffs_centered
            end
            
            # 4. Compute effect for this sample on CPU
            effect_k_cpu[:, i] = B_wavelet_i_cpu * coeffs_cpu
        end
        
        push!(structured_effects, effect_k_cpu)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end 