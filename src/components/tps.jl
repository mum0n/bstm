"""
    TPS <: ComponentModel

A component model for a Thin Plate Spline (TPS) smoother. This component creates a
basis of radial basis functions centered at knots distributed across the covariate
space. The effect is a linear combination of these basis functions, with coefficients
regularized by a random walk prior to ensure smoothness.

# Version
v1.0.1 (2026-08-09)

# Mathematical Summary
A Thin Plate Spline models a function \$f(\\mathbf{x})\$ as a linear combination of
radial basis functions \$\\phi\$, centered at a set of \$M\$ knots \$\\mathbf{c}_k\$:

\$f(\\mathbf{x}) = \\sum_{k=1}^{M} \\beta_k \\phi(\\|\\mathbf{x} - \\mathbf{c}_k\\|)\$

The radial basis function \$\\phi(r)\$ depends on the dimensionality \$d\$ of the input
space \$\\mathbf{x}\$:
- For \$d=1\$, \$\\phi(r) = r^3\$.
- For \$d=2\$, \$\\phi(r) = r^2 \\log(r)\$.
- For odd \$d > 2\$, \$\\phi(r) = r^{2m-d}\$ (with \$m=2\$, this is \$r^{4-d}\$).
- For even \$d > 2\$, \$\\phi(r) = r^{2m-d} \\log(r)\$.

The coefficients \$\\boldsymbol{\\beta} = (\\beta_1, \\dots, \\beta_M)\$ are given a smoothing
prior to regularize the function. This implementation uses a second-order random
walk (RW2) prior as an approximation of the TPS penalty:

\$\\boldsymbol{\\beta} \\sim \\mathcal{N}(\\mathbf{0}, (\\tau \\mathbf{Q}_{RW2})^{-1})\$

# Distinction from other smoothers
- **P-Splines**: Use a basis of B-spline functions, which are localized polynomials
  with compact support. TPS basis functions have global support.
- **Gaussian Processes**: Assume a global correlation structure defined by a kernel.
  TPS can be shown to be equivalent to a GP with a specific non-stationary kernel.

# Best Use Case
Flexible, non-parametric smoothing for low-dimensional covariates (typically 1D, 2D,
or 3D), especially for interpolating scattered data points. It is a classic choice
for spatial smoothing when a GMRF on a lattice is not appropriate.

# Key References
- Duchon, J. (1977). Splines minimizing rotation-invariant semi-norms in Sobolev
  spaces. In *Constructive Theory of Functions of Several Variables* (pp. 85-100).
  Springer.
- Wood, S. N. (2003). Thin plate regression splines. *Journal of the Royal
  Statistical Society: Series B*, 65(1), 95-114.
- Wikipedia: Thin plate spline

# Fields
- `nbins::Int`: The number of knots (and basis functions) to use.
- `sigma::Distribution`: The prior for the std. dev. of the TPS coefficients.
"""
struct TPS <: ComponentModel
    nbins::Int
    sigma::Distribution
end

COMPONENT_TYPE_REGISTRY[:tps] = TPS

COMPONENT_CONSTRUCTORS[:tps] = (p, params) -> TPS(get(params, :nbins, 20), p.sigma)

MODEL_TO_STRUCTURE_MAP[:tps] = :smooth

"""
    get_datastructures!(m_type::Type{<:TPS}, M::Dict, mod_data::Dict)::Bool

Ensures that coordinate variables are provided and stores them in the module data.
"""
function get_datastructures!(m_type::Type{<:TPS}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error("The TPS model requires coordinate variables, e.g., `random(x, y, model=:tps)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for TPS model not found in data.")
        end
    end

    mod_data[:params][:coords] = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    return true
end

"""
    get_precomputes(m::TPS, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes knot locations, the radial basis function matrix, and the spectral
decomposition of the penalty matrix for the TPS coefficients.
"""
function get_precomputes(m::TPS, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("TPS component precomputes failed: coordinates not found in module data.")
    end
    
    n_obs, n_dims = size(coords)
    n_latent = m.nbins

    knot_method = get(mod_data[:params], :knot_method, :kmeans)
    knots = generate_inducing_points(coords, n_latent; method=knot_method)
    actual_n_knots = size(knots, 1)
    if actual_n_knots < n_latent
        @warn "TPS: Could only generate $(actual_n_knots) unique knots, requested $(n_latent). Using $(actual_n_knots)."
        n_latent = actual_n_knots
    end

    B = zeros(Float64, n_obs, n_latent)
    if n_dims == 1
        for i in 1:n_latent
            r = abs.(coords[:, 1] .- knots[i, 1])
            B[:, i] .= r.^3
        end
    elseif n_dims == 2
        for i in 1:n_latent
            dist_sq = (coords[:, 1] .- knots[i, 1]).^2 .+ (coords[:, 2] .- knots[i, 2]).^2
            r = sqrt.(dist_sq)
            B[:, i] .= (r.^2) .* log.(r .+ 1e-9)
        end
    else
        for i in 1:n_latent
            dist_sq = sum((coords .- knots[i, :]').^2, dims=2)
            r = sqrt.(dist_sq)
            if isodd(n_dims)
                B[:, i] .= r.^(4 - n_dims)
            else
                B[:, i] .= (r.^(4 - n_dims)) .* log.(r .+ 1e-9)
            end
        end
    end
    
    template = build_structure_template(:rw2, n_latent)
    Q_template = template.matrix
    
    rank_deficiency = 2
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values
    scaling_factor = _compute_scaling_factor(L, rank_deficiency)
    
    Q_template_scaled = Q_template ./ scaling_factor
    L_scaled = L ./ scaling_factor

    return (
        basis_matrix=B,
        Q_template=Q_template_scaled,
        scaling_factor=scaling_factor,
        U=U,
        L=L_scaled,
        n_latent=n_latent,
        knots=knots
    )
end

"""
    get_priors(m::TPS, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `sigma` and the `raw` coefficients for the basis functions.
"""
function get_priors(m::TPS, spec::NamedTuple, arch::String, outcome_idx, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    return """
        $(p_names.sigma) ~ $(sigma_prior_str)
        $(p_names.raw) ~ MvNormal(zeros(spec.hyper.n_latent), I)
    """
end

"""
    get_updates(m::TPS, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for constructing the `TPS` smooth effect.
"""
function get_updates(m::TPS, spec::NamedTuple, arch::String, outcome_idx, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- Thin Plate Spline (TPS) Smoother Component: $(spec.key) ---
        local precomputes = spec.hyper
        
        # Reconstruct latent coefficients using spectral decomposition of the penalty.
        local diag_D = $(p_names.sigma) ./ sqrt.(precomputes.L .+ M.noise)
        diag_D[1] = 0.0
        diag_D[2] = 0.0
        
        local coeffs = precomputes.U * (diag_D .* $(p_names.raw))
        
        # Compute final effect by multiplying basis matrix with coefficients.
        $(p_names.latent) = precomputes.basis_matrix * coeffs
        
        $(eta_target) .+= $(p_names.latent)
    """
end

"""
    get_effects(m::TPS, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_total)::NamedTuple

Reconstructs the `TPS` component's effect from posterior samples.
"""
function get_effects(m::TPS, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_total)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()

    precomputes = spec.hyper
    U = precomputes.U
    L = precomputes.L
    noise = M.noise
    n_latent = precomputes.n_latent
    knots = precomputes.knots
    n_dims = size(knots, 2)

    B_train = precomputes.basis_matrix
    B_full = if !isnothing(PS) && haskey(PS, :basis_matrices) && haskey(PS.basis_matrices, spec.key)
        vcat(B_train, PS.basis_matrices[spec.key])
    else
        B_train
    end
    
    if size(B_full, 1) != N_total
        @warn "TPS effect reconstruction: dimension mismatch. Using in-sample effects only."
        B_full = B_train
    end

    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(v.sigma), 1)
        raw_samples = get_params_vector(chain, string(v.raw), n_latent)

        effect_k = Matrix{Float64}(undef, size(B_full, 1), n_samples)

        for i in 1:n_samples
            current_sigma = sigma_samples[i, 1]
            current_raw = raw_samples[i, :]
            
            diag_D = current_sigma ./ sqrt.(L .+ noise)
            diag_D[1] = 0.0
            diag_D[2] = 0.0
            
            coeffs = U * (diag_D .* current_raw)
            effect_k[:, i] = B_full * coeffs
        end
        push!(structured_effects, effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
