# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    FFT <: ComponentModel

A component model for a Fourier-based smoother. This component creates a basis of
sine and cosine functions at different frequencies. The effect is a linear combination
of these basis functions, with coefficients regularized by a random walk prior to
ensure smoothness.

# Fields
- `sigma::Distribution`: The prior for the standard deviation of the Fourier coefficients.
- `nbins::Int`: The total number of basis functions (sine/cosine pairs).
- `lengthscale::Union{Distribution, Vector{<:Distribution}}`: The prior for the lengthscale(s),
  which control the period of the basis functions.
"""
struct FFT <: ComponentModel
    sigma::Distribution
    nbins::Int
    lengthscale::Union{Distribution, Vector{<:Distribution}}
end

# Add to the central component constructor registry.
COMPONENT_CONSTRUCTORS[:fft] = (p, params) -> FFT(
    p.sigma,
    get(params, :nbins, 20),
    p.lengthscale
)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[FFT] = :smooth

"""
    get_datastructures!(m_type::Type{<:FFT}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `FFT` component.
It ensures that coordinate variables are provided and stores them in the module data.
"""
function get_datastructures!(m_type::Type{<:FFT}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error("The FFT model requires coordinate variables, e.g., `random(x, model=:fft)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for FFT model not found in data.")
        end
    end

    # Store the coordinates matrix in the module's parameters for later use.
    mod_data[:params][:coords] = Matrix{Float64}(M[:data][!, Symbol.(variables)])

    return true
end

"""
    get_precomputes(m::FFT, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `FFT` component.
This involves storing the coordinate matrix and pre-computing the penalty matrix
and its spectral decomposition for the Fourier coefficients.
"""
function get_precomputes(m::FFT, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("FFT component precomputes failed: coordinates not found in module data.")
    end
    
    n_latent = m.nbins
    n_dims = size(coords, 2)

    # Determine number of bins per dimension for basis generation
    nbins_per_dim = fill(round(Int, n_latent^(1/n_dims)), n_dims)
    while prod(nbins_per_dim) < n_latent
        nbins_per_dim[1] += 1
    end

    # Create penalty matrix Q for the coefficients (RW2 is a common choice for smoothness)
    template = build_structure_template(:rw2, n_latent)
    Q_template = template.matrix
    
    # Spectral decomposition for AD-friendly sampling of coefficients
    rank_deficiency = 2 # RW2
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values
    scaling_factor = _compute_scaling_factor(L, rank_deficiency)
    
    Q_template_scaled = Q_template ./ scaling_factor
    L_scaled = L ./ scaling_factor

    return (
        coords=coords,
        nbins_per_dim=nbins_per_dim,
        Q_template=Q_template_scaled,
        scaling_factor=scaling_factor,
        U=U,
        L=L_scaled,
        n_latent=n_latent
    )
end

"""
    get_priors(m::FFT, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `FFT` component's priors.
It defines priors for `sigma`, `lengthscale`, and the `raw` coefficients.
"""
function get_priors(m::FFT, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = String[]
    push!(priors, "$(p_names.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(p_names.sigma))")

    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors, "$(p_names.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(p_names.ls))")
    else
        ls_prior_str = _distribution_to_string(m.lengthscale)
        push!(priors, "$(p_names.ls) ~ NamedDist($(ls_prior_str), :$(p_names.ls))")
    end
    
    push!(priors, "$(p_names.raw) ~ NamedDist(MvNormal(zeros(T, spec.precomputes.n_latent), I), :$(p_names.raw))")

    return join(priors, "\n    ")
end

"""
    bstm_fourier_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int}, lengthscale::Union{Real, AbstractVector})

Helper function to generate a tensor product Fourier basis matrix.
"""
function bstm_fourier_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int}, lengthscale::Union{Real, AbstractVector})
    n_obs, n_dims = size(coords)
    
    ls_vec = if lengthscale isa Real
        fill(Float64(lengthscale), n_dims)
    else
        if length(lengthscale) != n_dims; error("Length of lengthscale vector must match coordinate dimensions."); end
        lengthscale
    end

    basis_matrices_1D = []
    for i in 1:n_dims
        vals = coords[:, i]
        ls_val = ls_vec[i]
        t_coords = vals ./ ls_val
        n_basis_1d = nbins_per_dim[i]
        B_1d = zeros(eltype(t_coords), n_obs, n_basis_1d)
        
        for m in 1:div(n_basis_1d, 2)
            if (2*m) <= n_basis_1d
                arg = (2.0 * pi * m) .* t_coords
                B_1d[:, 2*m-1] = sin.(arg)
                B_1d[:, 2*m]   = cos.(arg)
            end
        end
        push!(basis_matrices_1D, B_1d)
    end
    
    if isempty(basis_matrices_1D); return zeros(n_obs, 0); end

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
    get_updates(m::FFT, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `FFT` smooth effect.
"""
function get_updates(m::FFT, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- FFT Smoother Component: $(spec.key) ---
        local precomputes = spec_registry[:$(spec.key)].precomputes
        
        # 1. Dynamically generate the Fourier basis matrix using the sampled lengthscale
        local B_fft = bstm_fourier_basis(
            T.(precomputes.coords),
            precomputes.nbins_per_dim,
            $(p_names.ls)
        )
        
        # 2. Reconstruct latent coefficients using spectral decomposition of the penalty matrix
        local diag_D = $(p_names.sigma) ./ sqrt.(precomputes.L .+ M.noise)
        # Enforce sum-to-zero constraints for RW2 penalty
        diag_D[1] = zero(T)
        diag_D[2] = zero(T)
        
        local coeffs = precomputes.U * (diag_D .* $(p_names.raw))
        
        # 3. Compute final effect by multiplying basis matrix with coefficients
        local $(p_names.latent) = B_fft * coeffs
        
        $(eta_target) .+= $(p_names.latent)
    """
end

"""
    get_effects(m::FFT, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `FFT` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::FFT, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    sigma_samples = get(chain, p_names.sigma)
    ls_samples = get(chain, p_names.ls)
    raw_samples = get(chain, p_names.raw)

    precomputes = spec.precomputes
    U = precomputes.U
    L = precomputes.L
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

    reconstructed_effects = zeros(n_samples, size(coords_full, 1))

    for i in 1:n_samples
        current_sigma = sigma_samples[i]
        current_ls = if m.lengthscale isa Vector; ls_samples[i, :]; else ls_samples[i]; end
        current_raw = raw_samples[i, :]
        
        # Reconstruct basis matrix for the current lengthscale sample
        B_fft_i = bstm_fourier_basis(coords_full, nbins_per_dim, current_ls)
        
        # Reconstruct coefficients
        diag_D = current_sigma ./ sqrt.(L .+ noise)
        diag_D[1] = 0.0
        diag_D[2] = 0.0
        coeffs = U * (diag_D .* current_raw)
        
        reconstructed_effects[i, :] = B_fft_i * coeffs
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    return (structured=(mean=mean_effect, lower=lower_ci, upper=upper_ci),)
end
