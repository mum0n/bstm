"""
    Wavelet <: ComponentModel

A component for a wavelet-based smoother. This component creates a basis of
wavelet functions at different scales and locations across the covariate space. The
effect is a linear combination of these basis functions, with coefficients regularized by
a random walk prior to ensure smoothness.

# Version
v1.0.1 (2026-08-10)

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
- `:spectral` (default): An efficient, AD-safe method using spectral decomposition.
- `:cholesky`: An AD-safe didactic alternative using dense Cholesky factorization.
- `:cholesky_sparse`: A non-AD-safe didactic method using sparse Cholesky
  factorization, suitable for gradient-free samplers.

# Fields
- `family::Symbol`: The wavelet family to use (e.g., `:db4`, `:haar`).
- `nbins::Int`: The total number of basis functions (wavelets) to generate.
- `sigma::Distribution`: The prior for the std. dev. of the wavelet coefficients.
- `lengthscale::Union{Distribution, Vector{<:Distribution}}`: Prior for the
  lengthscale(s), which control the dilation of the wavelets.
- `method::Symbol`: The computational method for regularizing coefficients.
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

"""
    get_datastructures!(m_type::Type{<:Wavelet}, M::Dict, mod_data::Dict)::Bool

Ensures that coordinate variables are provided and stores them in the module data.
"""
function get_datastructures!(m_type::Type{<:Wavelet}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]
    if isempty(variables)
        error("Wavelet model requires coordinate variables, e.g., `random(x, model=:wavelet)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Wavelet model not found in data.")
        end
    end

    mod_data[:params][:coords] = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    return true
end

"""
    get_precomputes(m::Wavelet, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the penalty matrix, its spectral decomposition, and its dense
Cholesky factorization for the wavelet coefficients.
"""
function get_precomputes(m::Wavelet, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("Wavelet component precomputes failed: coordinates not found.")
    end
    
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


"""
    get_priors(m::Wavelet, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `sigma`, `lengthscale`, and the `raw` coefficients.
"""
function get_priors(m::Wavelet, spec::NamedTuple, arch::String, outcome_idx, M)::String
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
    
    push!(priors, "$(p_names.raw) ~ MvNormal(zeros(spec.precomputes.n_latent), I)")

    return join(priors, "\n    ")
end

"""
    get_updates(m::Wavelet, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code to construct the wavelet smooth effect, dispatching on
the chosen method.
"""
function get_updates(
    m::Wavelet, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    
    common_basis_code = """
        local precomputes = spec_registry[:$(key)].hyper
        local B_wavelet = bstm_tensor_product_wavelet_basis(
            precomputes.coords,
            precomputes.nbins_per_dim,
            Symbol("$(m.family)"),
            $(p_names.ls)
        )
    """

    spectral_code = """
        # --- Wavelet Smoother Component (Spectral): $(key) ---
        let
            $(common_basis_code)
            local diag_D = $(p_names.sigma) ./ sqrt.(precomputes.L .+ M.noise)
            diag_D[1] = 0.0; diag_D[2] = 0.0
            local coeffs = precomputes.U * (diag_D .* $(p_names.raw))
            local $(p_names.latent) = B_wavelet * coeffs
            $(eta_target) .+= $(p_names.latent)
        end
    """

    cholesky_code = """
        # --- Wavelet Smoother Component (Cholesky, AD-Safe): $(key) ---
        let
            $(common_basis_code)
            local F = precomputes.cholesky_factor
            local coeffs_raw = F.L' \\ $(p_names.raw)
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * precomputes.n_latent), sum(coeffs_raw))
            local coeffs = $(p_names.sigma) .* coeffs_raw
            local $(p_names.latent) = B_wavelet * coeffs
            $(eta_target) .+= $(p_names.latent)
        end
    """

    cholesky_sparse_code = """
        # --- Wavelet Smoother Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            $(common_basis_code)
            local Q_penalty = precomputes.Q_template
            local F = cholesky(Symmetric(Q_penalty + M.noise * I))
            local coeffs_raw = F.L' \\ $(p_names.raw)
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * precomputes.n_latent), sum(coeffs_raw))
            local coeffs = $(p_names.sigma) .* coeffs_raw
            local $(p_names.latent) = B_wavelet * coeffs
            $(eta_target) .+= $(p_names.latent)
        end
    """

    if m.method == :spectral; return spectral_code;
    elseif m.method == :cholesky; return cholesky_code;
    elseif m.method == :cholesky_sparse; return cholesky_sparse_code;
    else; error("Unsupported method '$(m.method)' for Wavelet component."); end
end

"""
    get_effects(m::Wavelet, chain, M::NamedTuple, ...)

Reconstructs the `Wavelet` component's effect from posterior samples, dispatching
on the method used during sampling.
"""
function get_effects(
    m::Wavelet, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()

    precomputes = spec.hyper
    noise = M.noise
    n_latent = precomputes.n_latent
    nbins_per_dim = precomputes.nbins_per_dim
    
    coords_train = precomputes.coords
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        coords_train
    end

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        ls_samples = get_params_vector(
            chain, string(p_names.ls), m.lengthscale isa Vector ? length(m.lengthscale) : 1
        )
        raw_samples = get_params_vector(chain, string(p_names.raw), n_latent)

        effect_k = zeros(Float64, size(coords_full, 1), n_samples)

        for i in 1:n_samples
            current_ls = if m.lengthscale isa Vector; ls_samples[i, :]; else ls_samples[i, 1]; end
            B_wavelet_i = bstm_tensor_product_wavelet_basis(
                coords_full, nbins_per_dim, m.family, current_ls
            )
            
            local coeffs
            if m.method == :spectral
                U, L = precomputes.U, precomputes.L
                diag_D = sigma_samples[i] ./ sqrt.(L .+ noise)
                diag_D[1] = 0.0; diag_D[2] = 0.0
                coeffs = U * (diag_D .* raw_samples[i, :])
            else # :cholesky or :cholesky_sparse
                F = precomputes.cholesky_factor
                coeffs_raw = F.L' \ raw_samples[i, :]
                coeffs_centered = coeffs_raw .- mean(coeffs_raw)
                coeffs = sigma_samples[i] .* coeffs_centered
            end
            
            effect_k[:, i] = B_wavelet_i * coeffs
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
