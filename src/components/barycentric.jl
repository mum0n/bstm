# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    Barycentric <: ComponentModel

A component model for a barycentric interpolation smoother. This component creates a basis
of linear "tent" functions centered at knots distributed across the covariate space. The
effect is a linear combination of these basis functions, with coefficients regularized by
a random walk prior to ensure smoothness.

# Fields
- `sigma::Distribution`: The prior distribution for the standard deviation of the coefficients.
- `nbins::Union{Int, Vector{Int}}`: The number of bins (and basis functions) for each dimension.
- `diff_order::Int`: The order of the random walk penalty on the coefficients (1 for RW1, 2 for RW2).
"""
struct Barycentric <: ComponentModel
    sigma::Distribution
    nbins::Union{Int, Vector{Int}}
    diff_order::Int
end

# Add to the central component constructor registry.
COMPONENT_CONSTRUCTORS[:barycentric] = (p, params) -> Barycentric(p.sigma, get(params, :nbins, 20), get(params, :diff_order, 2))

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[Barycentric] = :smooth

"""
    get_datastructures!(m_type::Type{<:Barycentric}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `Barycentric` component.
It ensures that coordinate variables are provided and stores them in the module data.
"""
function get_datastructures!(m_type::Type{<:Barycentric}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error("The Barycentric model requires coordinate variables, e.g., `random(x, y, model=:barycentric)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Barycentric model not found in data.")
        end
    end

    # Store the coordinates matrix in the module's parameters for later use.
    mod_data[:params][:coords] = Matrix{Float64}(M[:data][!, Symbol.(variables)])

    return true
end

"""
    get_precomputes(m::Barycentric, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `Barycentric` component.
This involves creating the tensor product basis of linear "tent" functions and
the precision matrix template for the random walk penalty on the coefficients.
"""
function get_precomputes(m::Barycentric, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("Barycentric component precomputes failed: coordinates not found in module data.")
    end
    
    n_obs, n_dims = size(coords)

    # Determine number of bins per dimension
    local nbins_per_dim::Vector{Int}
    if m.nbins isa Int
        nbins_per_dim = fill(m.nbins, n_dims)
    elseif m.nbins isa Vector{Int} && length(m.nbins) == n_dims
        nbins_per_dim = m.nbins
    else
        error("For a $(n_dims)D Barycentric smooth, `nbins` must be an Int or a Vector{Int} of length $(n_dims).")
    end

    # --- Helper function to create 1D linear tent basis ---
    function _create_1d_tent_basis(vals::AbstractVector, n_knots::Int)
        B_1d = zeros(length(vals), n_knots)
        v_min, v_max = extrema(vals)
        knots = range(v_min, stop=v_max, length=n_knots)
        h = (v_max - v_min) / (n_knots > 1 ? (n_knots - 1) : 1.0)
        h = h > 0 ? h : 1.0
        
        for i in 1:n_knots
            dist = abs.(vals .- knots[i]) ./ h
            mask = dist .< 1.0
            B_1d[mask, i] .= 1.0 .- dist[mask]
        end
        return B_1d
    end

    # --- Create basis matrix B ---
    # Generate 1D basis matrices for each dimension
    basis_matrices_1D = [_create_1d_tent_basis(coords[:, i], nbins_per_dim[i]) for i in 1:n_dims]

    # Compute the tensor product efficiently
    B_final = basis_matrices_1D[1]
    for i in 2:n_dims
        B_next = basis_matrices_1D[i]
        
        n_obs_i, n_cols_final = size(B_final)
        _, n_cols_next = size(B_next)
        
        # Reshape for broadcasting to compute row-wise outer products.
        B_final_reshaped = reshape(B_final, n_obs_i, n_cols_final, 1)
        B_next_reshaped = reshape(B_next, n_obs_i, 1, n_cols_next)
        
        tensor_prod = B_final_reshaped .* B_next_reshaped
        
        B_final = reshape(tensor_prod, n_obs_i, n_cols_final * n_cols_next)
    end
    
    n_latent = size(B_final, 2)

    # --- Create penalty matrix Q ---
    penalty_type = m.diff_order == 1 ? :rw1 : :rw2
    template = build_structure_template(penalty_type, n_latent)
    Q_template = template.matrix
    
    # --- Spectral decomposition for AD-friendly sampling ---
    rank_deficiency = m.diff_order
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values
    scaling_factor = _compute_scaling_factor(L, rank_deficiency)
    
    Q_template_scaled = Q_template ./ scaling_factor
    L_scaled = L ./ scaling_factor

    return (
        basis_matrix=B_final,
        Q_template=Q_template_scaled,
        scaling_factor=scaling_factor,
        U=U,
        L=L_scaled,
        n_latent=n_latent
    )
end

"""
    get_priors(m::Barycentric, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `Barycentric` component's priors.
It defines the prior for `sigma` and the `raw` coefficients for the basis functions.
"""
function get_priors(m::Barycentric, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    return """
        $(p_names.sigma) ~ NamedDist($(sigma_prior_str), :$(p_names.sigma))
        $(p_names.raw) ~ NamedDist(MvNormal(zeros(T, spec_registry[:$(spec.key)].precomputes.n_latent), I), :$(p_names.raw))
    """
end

"""
    get_updates(m::Barycentric, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `Barycentric` smooth effect.
"""
function get_updates(m::Barycentric, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- Barycentric Smoother Component: $(spec.key) ---
        local precomputes = spec_registry[:$(spec.key)].precomputes
        
        # Reconstruct latent coefficients using spectral decomposition of the penalty matrix
        local diag_D = $(p_names.sigma) ./ sqrt.(precomputes.L .+ M.noise)
        # Enforce sum-to-zero constraint(s) on coefficients
        for i in 1:$(m.diff_order); diag_D[i] = zero(T); end
        
        local coeffs = precomputes.U * (diag_D .* $(p_names.raw))
        
        # Compute final effect by multiplying basis matrix with coefficients
        local $(p_names.latent) = T.(precomputes.basis_matrix) * coeffs
        
        $(eta_target) .+= $(p_names.latent)
    """
end

"""
    get_effects(m::Barycentric, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `Barycentric` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::Barycentric, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    sigma_samples = get(chain, p_names.sigma)
    raw_samples = get(chain, p_names.raw)

    precomputes = spec.precomputes
    U = precomputes.U
    L = precomputes.L
    noise = M.noise
    n_latent = precomputes.n_latent

    # The `predict` function is responsible for creating the basis matrix for the prediction set.
    # Here, we assume it is available in PS.basis_matrices if a prediction set is provided.
    B_train = precomputes.basis_matrix
    B_full = if !isnothing(PS) && haskey(PS, :basis_matrices) && haskey(PS.basis_matrices, spec.key)
        vcat(B_train, PS.basis_matrices[spec.key])
    else
        B_train
    end
    
    if size(B_full, 1) != N_total
        @warn "Barycentric effect reconstruction: dimension mismatch. Using in-sample effects only."
        B_full = B_train
    end

    reconstructed_effects = zeros(n_samples, size(B_full, 1))

    for i in 1:n_samples
        current_sigma = sigma_samples[i]
        current_raw = raw_samples[i, :]
        
        diag_D = current_sigma ./ sqrt.(L .+ noise)
        for j in 1:m.diff_order; diag_D[j] = 0.0; end
        
        coeffs = U * (diag_D .* current_raw)
        reconstructed_effects[i, :] = B_full * coeffs
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    return (structured=(mean=mean_effect, lower=lower_ci, upper=upper_ci),)
end
