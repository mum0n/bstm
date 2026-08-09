

"""
    RFF <: ComponentModel

A component model for a Random Fourier Features (RFF) smoother. This component
approximates a stationary kernel (like Squared Exponential or Matérn) by projecting
the input coordinates into a randomized feature space. This transforms the GP into a
more scalable Bayesian linear regression problem.

# Version
v1.0.0 (2026-08-08)

# Mathematical Summary
The RFF method approximates a stationary kernel \$k(\\tau) = k(x - x')\$ by using
Bochner's theorem, which states that \$k(\\tau)\$ is the Fourier transform of a
non-negative measure \$p(\\omega)\$, the spectral density. The kernel can be written as:
\$k(\\tau) = \\mathbb{E}_{\\omega \\sim p}[\\cos(\\omega^T \\tau)]\$

This expectation is approximated via Monte Carlo by drawing \$M\$ random frequencies
\$\\{\\omega_j\\}_{j=1}^M\$ from \$p(\\omega)\$. The feature map \$\\phi(x)\$ is then:
\$\\phi(x) = \\sqrt{2/M} [\\cos(\\omega_1^T x + b_1), \\dots, \\cos(\\omega_M^T x + b_M)]\$
where \$b_j \\sim \\text{Uniform}(0, 2\\pi)\$. The kernel is then approximated as
\$k(x, x') \\approx \\phi(x)^T \\phi(x')\$. The final effect is a linear combination
of these features: \$f(x) = \\phi(x)^T \\beta\$, where \$\\beta\$ are learnable weights.

# Distinction from other GP approximations
- **Nystrom**: Approximates the full kernel matrix \$K_{XX}\$ with a low-rank version
  \$\\tilde{K}_{XX} = K_{XU} K_{UU}^{-1} K_{UX}\$. It's a low-rank approximation of the
  covariance matrix itself.
- **RFF (Random Fourier Features)**: Approximates the kernel *function* \$k(x, x')\$
  with a finite-dimensional feature map \$\\phi(x)^T \\phi(x')\$. It transforms the problem
  into a linear model in a high-dimensional feature space.
- **Full GP**: Computes the exact kernel matrix \$K_{XX}\$ and performs inference directly,
  which is \$O(N^3)\$ and memory-intensive (\$O(N^2)\$). SVGP (and FITC) reduce this to
  \$O(NM^2 + M^3)\$ for computation and \$O(NM)\$ for memory.

# Assumptions
- The underlying process is stationary.
- The number of features `n_features` is sufficient to provide a good approximation
  of the true kernel.

# Best Use Case
A scalable alternative to a full Gaussian Process for modeling smooth, non-linear
effects of continuous covariates, especially when the number of observations is
large.

# Key References
- Rahimi, A., & Recht, B. (2007). *Random features for large-scale kernel
  machines*. In NIPS.
- Wikipedia: Random Fourier features

# Fields
- `lengthscale::Union{Distribution, Vector{<:Distribution}}`: The prior for the
  lengthscale(s) of the kernel.
- `sigma::Distribution`: The prior for the standard deviation of the RFF coefficients.
- `n_features::Int`: The number of random features to use for the approximation.
- `kernel::String`: The name of the kernel to approximate (e.g., "se", "matern32").
"""
struct RFF <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    n_features::Int
    kernel::String
end

COMPONENT_TYPE_REGISTRY[:rff] = RFF

COMPONENT_CONSTRUCTORS[:rff] = (p, params) -> RFF(
    p.lengthscale,
    p.sigma,
    get(params, :n_features, 20),
    string(get(params, :kernel, "se"))
)

MODEL_TO_STRUCTURE_MAP[:rff] = :smooth

"""
    get_datastructures!(m_type::Type{<:RFF}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `RFF` component.
It ensures that coordinate variables are provided and stores them in the module data.
"""
function get_datastructures!(m_type::Type{<:RFF}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error("The RFF model requires coordinate variables, e.g., `random(x, y, model=:rff)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for RFF model not found in data.")
        end
    end

    mod_data[:params][:coords] = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    return true
end

"""
    _generate_rff_fixed_params(in_dims, n_features, lengthscale, kernel_name)

Generates fixed random projection weights (W) and biases (b) for RFF approximation.
This helper function is used in the precomputation stage.
"""
function _generate_rff_fixed_params(
    in_dims::Int, n_features::Int, lengthscale::Union{Real, AbstractVector},
    kernel_name::String
)
    b = rand(Uniform(0, 2 * pi), n_features)
    W = Matrix{Float64}(undef, in_dims, n_features)
    k_name = lowercase(kernel_name)

    if k_name in ["se", "gaussian", "rbf"]
        if lengthscale isa Real
            W .= rand(Normal(0, 1.0 / lengthscale), in_dims, n_features)
        else
            if length(lengthscale) != in_dims
                error("ARD lengthscale vector length mismatch.")
            end
            for d in 1:in_dims
                W[d, :] = rand(Normal(0, 1.0 / lengthscale[d]), n_features)
            end
        end
    elseif occursin("matern", k_name)
        nu = if k_name == "matern12"; 0.5; elseif k_name == "matern32"; 1.5; else 2.5; end
        df = 2 * nu
        if lengthscale isa Real
            W .= (sqrt(df) / lengthscale) .* rand(TDist(df), in_dims, n_features)
        else
            if length(lengthscale) != in_dims
                error("ARD lengthscale vector length mismatch.")
            end
            for d in 1:in_dims
                W[d, :] = (sqrt(df) / lengthscale[d]) .* rand(TDist(df), n_features)
            end
        end
    else
        @warn "Kernel '$kernel_name' not recognized for RFF. Defaulting to SE."
        return _generate_rff_fixed_params(in_dims, n_features, lengthscale, "se")
    end
    return W, b
end

"""
    get_precomputes(m::RFF, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `RFF` component.
It generates the fixed random features (`W_fixed`, `b_fixed`) that serve as the
means for the priors on the adaptive feature parameters.
"""
function get_precomputes(m::RFF, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("RFF component precomputes failed: coordinates not found in module data.")
    end
    
    in_dims = size(coords, 2)
    
    ls_prior = m.lengthscale
    local ls_initial
    if ls_prior isa Vector
        ls_initial = [mean(p isa Truncated ? untruncated(p) : p) for p in ls_prior]
    else
        ls_initial = mean(ls_prior isa Truncated ? untruncated(ls_prior) : ls_prior)
    end

    W_fixed, b_fixed = _generate_rff_fixed_params(
        in_dims, m.n_features, ls_initial, m.kernel
    )

    return (
        coords=coords,
        W_fixed=W_fixed,
        b_fixed=b_fixed,
        n_latent=m.n_features,
        in_dims=in_dims
    )
end

"""
    get_priors(m::RFF, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `sigma`, `lengthscale`, the adaptive feature parameters `W`
and `b`, and the `raw` coefficients.
"""
function get_priors(
    m::RFF, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
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
        "$(p_names.W) ~ MvNormal(vec(spec.precomputes.W_fixed), 0.1)"
    )
    push!(
        priors,
        "$(p_names.b) ~ MvNormal(spec.precomputes.b_fixed, 0.1)"
    )
    push!(
        priors,
        "$(p_names.raw) ~ MvNormal(zeros(T, spec.precomputes.n_latent), I)"
    )

    return join(priors, "\n    ")
end

"""
    get_updates(m::RFF, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code string for constructing the `RFF` smooth effect.
"""
function get_updates(
    m::RFF, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    precomputes = "spec.precomputes"
    
    return """
        # --- RFF Smoother Component: $(spec.key) ---
        let
            local X_coords = $(precomputes).coords
            local W_matrix = reshape($(p_names.W), $(precomputes).in_dims, $(precomputes).n_latent)
            
            # Compute the feature matrix Phi
            local Phi = sqrt(2.0 / $(precomputes).n_latent) .* cos.((X_coords * W_matrix) .+ $(p_names.b)')
            
            # Scale the raw coefficients and compute the final effect
            local scaled_coeffs = $(p_names.raw) .* $(p_names.sigma)
            local $(p_names.latent) = Phi * scaled_coeffs
            
            $(eta_target) .+= $(p_names.latent)
        end
    """
end

"""
    get_effects(m::RFF, chain, M::NamedTuple, ...)::NamedTuple

Reconstructs the `RFF` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(
    m::RFF, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    precomputes = spec.precomputes
    in_dims = precomputes.in_dims
    n_features = precomputes.n_latent

    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(precomputes.coords, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        precomputes.coords
    end

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)
        raw_samples = get_params_vector(chain, string(p_names.raw), n_features)
        W_samples = get_params_vector(chain, string(p_names.W), in_dims * n_features)
        b_samples = get_params_vector(chain, string(p_names.b), n_features)

        effect_k = zeros(Float64, size(coords_full, 1), n_samples)

        for i in 1:n_samples
            current_sigma = sigma_samples[i, 1]
            current_raw = raw_samples[i, :]
            current_W = reshape(W_samples[i, :], in_dims, n_features)
            current_b = b_samples[i, :]
            
            Phi = sqrt(2.0 / n_features) .* cos.((coords_full * current_W) .+ current_b')
            
            scaled_coeffs = current_raw .* current_sigma
            effect_k[:, i] = Phi * scaled_coeffs
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
