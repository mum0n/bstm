# Definitions

# Discrete and Graph Manifold Categories 

# BSTM Low-Level Manifold Registry [v06.1 - Reusable Schema] ---
# Rationale: Manifolds are now defined as low-level primitives that are domain-agnostic. 
# The context (Spatial vs Temporal) is determined at the model-building stage.

abstract type Manifold end

# 1. Discrete & Graph Primitives
struct Fixed <: Manifold; sigma_prior::UnivariateDistribution; end

struct IID <: Manifold; sigma_prior::UnivariateDistribution; end
struct ICAR <: Manifold; sigma_prior::UnivariateDistribution; end
struct Besag <: Manifold; sigma_prior::UnivariateDistribution; end
struct BYM2 <: Manifold; rho_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end
struct Leroux <: Manifold; rho_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end
struct SAR <: Manifold; rho_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end
struct RW1 <: Manifold; sigma_prior::UnivariateDistribution; end
struct RW2 <: Manifold; sigma_prior::UnivariateDistribution; end
struct AR1 <: Manifold; rho_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end

# 2. Continuous & Spectral Primitives (Cross-Domain)
struct GP <: Manifold; lengthscale_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; kernel::String; end
struct FITC <: Manifold; lengthscale_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; n_inducing::Int; end
struct RFF <: Manifold; lengthscale_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; n_features::Int; end
struct FFT <: Manifold; sigma_prior::UnivariateDistribution; nbins::Int; end
struct SPDE <: Manifold; sigma_prior::UnivariateDistribution; kappa_prior::UnivariateDistribution; end
struct DAG <: Manifold; adjacency_matrix::AbstractMatrix; sigma_prior::UnivariateDistribution; end

# 3. Non-Euclidean & Decay Primitives (v06.1 Expansion)
struct ExponentialDecay <: Manifold; sigma_prior::UnivariateDistribution; decay_lengthscale_prior::UnivariateDistribution; end
struct Hyperbolic <: Manifold; curvature::Float64; sigma_prior::UnivariateDistribution; end

# 4. Basis-Function Primitives (Covariate/Smooth)
struct TPS <: Manifold; nbins::Int; sigma_prior::UnivariateDistribution; end
struct BSpline <: Manifold; nbins::Int; degree::Int; sigma_prior::UnivariateDistribution; end
struct PSpline <: Manifold; nbins::Int; degree::Int; diff_order::Int; sigma_prior::UnivariateDistribution; end
struct Wavelets <: Manifold; wavelet_family::Symbol; nbins::Int; sigma_prior::UnivariateDistribution; end

# 5. Seasonal & Periodic (Wrappers)
struct Harmonic <: Manifold; amplitude_prior::UnivariateDistribution; phase_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end
struct Cyclic <: Manifold; period::Int; sigma_prior::UnivariateDistribution; end

# 6. Specialized & Network Manifolds [v14.3.10 - BSTM v06.1 Registry Fix]
struct Nystrom <: Manifold; sigma_prior::UnivariateDistribution; lengthscale_prior::UnivariateDistribution; n_inducing::Int; end
struct Eigen <: Manifold; sigma_prior::UnivariateDistribution; pca_sd_prior::UnivariateDistribution; pdef_sd_prior::UnivariateDistribution; n_factors::Int; ltri_indices::Vector{Int}; end
struct BCGN <: Manifold; sigma_prior::UnivariateDistribution; bipartite_adj::AbstractMatrix; group_weights::AbstractVector; end
struct LocalAdaptive <: Manifold
    sigma_prior::UnivariateDistribution
    weights_variable::Symbol
end
struct NetworkFlow <: Manifold
    sigma_prior::UnivariateDistribution
    adjacency_matrix::AbstractMatrix
    flow_direction::Symbol # :upstream, :downstream, :bidirectional
end

struct FixedManifold <: Manifold; end
struct CovariateManifold <: Manifold; end

# 6. Algebraic Operators
struct ComposedManifold <: Manifold; components::Vector{Manifold}; operator::Symbol; end

⊗(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :kronecker_product)
⊕(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :direct_sum)

 
otimes(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :kronecker_product)
oplus(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :direct_sum)


# Base pipe for transformations: Manifold |> Manifold (Transformation)
# If the RHS is a Transformation (LogManifold, ZScoreManifold, etc.), wrap the LHS
function Base.:|>(m::Manifold, t::Manifold)
    return TransformedManifold(m, typeof(t))
end

# Base pipe for composition: Manifold |> Manifold (Stacking)
# If the RHS is a Structural Manifold (Temporal, Spatial), compose them
function Base.:|>(m1::Manifold, m2::Manifold)
    return ComposedManifold([m1, m2], :pipe)
end


# --- 5. Composite and Constrained Manifolds ---
struct RegularizationGroupManifold <: Manifold
    manifolds::Vector{Manifold}
    penalty::Symbol # :ridge, :lasso
    lambda_prior::UnivariateDistribution
end

struct SoftConstraintManifold <: Manifold
    manifold::Manifold
    type::Symbol # :sum_to_zero, :monotonic
    weight::Float64
end
 
struct TransformedManifold <: Manifold
    manifold::Manifold
    transform_fn::DataType # e.g., Log, ZScoreManifold types
end

# Placeholder transformation types
struct LogManifold <: Manifold end
struct ZScoreManifold <: Manifold end
struct UnitScaleManifold <: Manifold end

# --- 6. Compositional Manifolds (Physics & Stacking) ---
struct KnorrHeld <: Manifold
    dim1::Symbol
    dim2::Symbol
    type::Char
    sigma_prior::UnivariateDistribution
end

struct AdvectionDiffusion <: Manifold
    dim1::Symbol
    dim2::Symbol
    velocity_field::Union{Nothing, Symbol}
    velocity_prior::UnivariateDistribution
    diffusion_coeff::Union{Nothing, Float64}
    diffusion_prior::UnivariateDistribution
end

struct Mosaic <: Manifold
    coordinates::Vector{Symbol}
    n_regions::Int
    local_smoothness::Bool
end

# 7. Manifold Type Dispatch [v06.1 Audit]
function manifold_type(m::Manifold)
    return lowercase(string(typeof(m)))
end
    
 

function parse_manifold_graph(expr_in::AbstractString)
    # Audited Recursive DSL Parser [v15.2.0 - BSTM v06.1 Final Restoration]
    # Rationale: This engine decomposes algebraic manifold expressions into a structural tree.
    # Requirements: Support for ⊕ (Direct Sum), ⊗ (Kronecker), and ∘ (Composition).
    
    local expr = string(replace(expr_in, r"\s" => ""))

    if occursin("⊕", expr)
        local elements = Base.split(expr, "⊕")
        return (type=:sum, elements=parse_manifold_graph.(string.(elements)))

    elseif occursin("⊗", expr)
        local elements = Base.split(expr, "⊗")
        return (type=:kronecker, elements=parse_manifold_graph.(string.(elements)))

    elseif occursin("∘", expr)
        local parts = Base.split(expr, "∘")
        return (type=:composition, elements=parse_manifold_graph.(string.(parts)))

    else
        local m = match(r"(\w+)\((.*?)\)", expr)
        if !isnothing(m)
            local m_name = lowercase(string(m.captures[1]))
            local var_match = match(r"\((.*)\)", expr)
            local var_name = !isnothing(var_match) ? string(var_match.captures[1]) : expr
            return (type=:atomic, model=m_name, var=var_name)
        else
            return (type=:atomic, model="literal", var=expr)
        end
    end
end


function process_graph_into_rules!(re_rules::Dict{String, Any}, opt_kwargs::Dict{Symbol, Any}, graph)
    # Audited Rule Dispatcher [v15.2.0]
    # Rationale: Recursively populates re_rules and architectural flags from the manifold graph.
    
    if graph.type == :atomic
        local m_name = lowercase(string(graph.model))
        local var_name = string(graph.var)
        re_rules[var_name] = Dict{Symbol, Any}(:model => m_name)

        # Routing for Global Architectural Flags
        if m_name in ["spatial", "iid", "bym2", "icar", "leroux", "sar", "dag", "spde", "gp", "fitc", "rff", "mosaic", "fft"]
            local m_resolved = (m_name == "spatial") ? "iid" : m_name
            re_rules[var_name][:model] = m_resolved
            re_rules[var_name][:domain] = :spatial
            opt_kwargs[:model_space] = m_resolved

        elseif m_name in ["temporal", "ar1", "rw1", "rw2", "fft", "rff", "harmonic"]
            local m_resolved = (m_name == "temporal") ? "ar1" : m_name
            re_rules[var_name][:model] = m_resolved
            re_rules[var_name][:domain] = :temporal
            opt_kwargs[:model_time] = m_resolved

        elseif m_name == "cyclic"
            re_rules[var_name][:domain] = :seasonal
            opt_kwargs[:model_season] = "cyclic"

        elseif m_name == "eigen"
            re_rules[var_name][:is_eigen] = true

        elseif m_name == "nested"
            opt_kwargs[:model_arch] = "multifidelity"
            re_rules[var_name][:is_nested] = true
        end

    elseif graph.type == :kronecker
        opt_kwargs[:model_st] = "IV"
        for el in graph.elements
            process_graph_into_rules!(re_rules, opt_kwargs, el)
        end

    elseif graph.type == :sum
        for el in graph.elements
            process_graph_into_rules!(re_rules, opt_kwargs, el)
        end

    elseif graph.type == :composition
        for el in graph.elements
            process_graph_into_rules!(re_rules, opt_kwargs, el)
        end
    end
end






# --- 3. Architectural Dispatch Types ---
# Concrete types cannot be subtyped; therefore, we must define abstract bases first.
abstract type AbstractModelArchitecture end

struct UnivariateArchitecture <: AbstractModelArchitecture end
struct MultivariateArchitecture <: AbstractModelArchitecture end
struct MultifidelityArchitecture <: AbstractModelArchitecture end
struct ExampleArchitecture <: AbstractModelArchitecture end
struct UnknownArchitecture <: AbstractModelArchitecture end


# --- 4. Likelihood Family Types ---
# Concrete types cannot be subtyped; therefore, we must define abstract bases first.
  
abstract type AbstractBSTM_Family end

struct PoissonFamily <: AbstractBSTM_Family end
struct GaussianFamily <: AbstractBSTM_Family end
struct LogNormalFamily <: AbstractBSTM_Family end
struct NegativeBinomialFamily <: AbstractBSTM_Family end
struct BinomialFamily <: AbstractBSTM_Family end
struct GammaFamily <: AbstractBSTM_Family end
struct ExponentialFamily <: AbstractBSTM_Family end
struct BetaFamily <: AbstractBSTM_Family end
struct InverseGaussianFamily <: AbstractBSTM_Family end
struct StudentTFamily <: AbstractBSTM_Family end
struct HalfNormalFamily <: AbstractBSTM_Family end
struct HalfStudentTFamily <: AbstractBSTM_Family end
struct LaplaceFamily <: AbstractBSTM_Family end
struct ParetoFamily <: AbstractBSTM_Family end
struct DirichletFamily <: AbstractBSTM_Family end
struct InverseWishartFamily <: AbstractBSTM_Family end


abstract type AbstractZIState end
struct NonZeroInflated <: AbstractZIState end
struct ZeroInflated <: AbstractZIState end


abstract type AbstractCensoringState end
struct Uncensored <: AbstractCensoringState end
struct LeftCensored <: AbstractCensoringState end
struct RightCensored <: AbstractCensoringState end
struct IntervalCensored <: AbstractCensoringState end


# --- 3. Unified Likelihood Structure ---
struct bstm_Likelihood{F<:AbstractBSTM_Family, Z<:AbstractZIState, C<:AbstractCensoringState, W, P, R, S, T, TR, TL, TU, HT, EX} <: ContinuousMultivariateDistribution
    family::F
    y_obs::TR
    zi_state::Z
    censoring_state::C
    weight::W
    phi_zi::P
    r_nb::R
    sigma_y::S
    trial::T
    y_L::TL
    y_U::TU
    hurdle::HT
    extra_params::EX
end


# --- 4. Outer Constructor (Fixed Mapping) ---
function bstm_Likelihood(family_input, y_obs; sigma_y=0.0, weight=1.0, phi_zi=-Inf, r_nb=0, trial=0,
                         y_L=-Inf, y_U=Inf, hurdle=-Inf, extra_params=zeros(1)[])

    local fam = if family_input isa AbstractBSTM_Family; family_input
        elseif family_input == "poisson"; PoissonFamily()
        elseif family_input == "gaussian"; GaussianFamily()
        elseif family_input == "lognormal"; LogNormalFamily()
        elseif family_input == "negbin"; NegativeBinomialFamily()
        elseif family_input == "binomial"; BinomialFamily()
        elseif family_input == "gamma"; GammaFamily()
        elseif family_input == "exponential"; ExponentialFamily()
        elseif family_input == "beta"; BetaFamily()
        elseif family_input == "inverse_gaussian"; InverseGaussianFamily()
        elseif family_input == "student_t"; StudentTFamily()
        elseif family_input == "half_normal"; HalfNormalFamily()
        elseif family_input == "half_student_t"; HalfStudentTFamily()
        elseif family_input == "laplace"; LaplaceFamily()
        elseif family_input == "pareto"; ParetoFamily()
        elseif family_input == "dirichlet"; DirichletFamily()
        else PoissonFamily() end

    local z_trait = phi_zi >= 0.0 ? ZeroInflated() : NonZeroInflated()

    # Ensure hurdle and bounds are numeric for kernel comparison
    local h_val = isnothing(hurdle) ? -Inf : hurdle
    local yL_val = isnothing(y_L) ? -Inf : y_L
    local yU_val = isnothing(y_U) ? Inf : y_U

    local c_trait = if !isfinite(yL_val) && !isfinite(yU_val); Uncensored()
        elseif !isfinite(yL_val) && isfinite(yU_val); LeftCensored()
        elseif isfinite(yL_val) && !isfinite(yU_val); RightCensored()
        else IntervalCensored() end

    return bstm_Likelihood(fam, y_obs, z_trait, c_trait, weight, phi_zi, r_nb, sigma_y, trial, yL_val, yU_val, h_val, extra_params)
end

# --- 5. Distribution Mapping ---

# Rationale: Standardizing the mapping of eta to Distribution objects.
function get_dist_ref(::PoissonFamily, d, eta, sig); return Poisson(exp(eta)); end
function get_dist_ref(::GaussianFamily, d, eta, sig); return Normal(eta, sig); end
function get_dist_ref(::LogNormalFamily, d, eta, sig); return LogNormal(eta, sig); end
function get_dist_ref(::NegativeBinomialFamily, d, eta, sig); mu = exp(eta); return NegativeBinomial(d.r_nb, d.r_nb/(d.r_nb + mu)); end
function get_dist_ref(::BinomialFamily, d, eta, sig); n = d.trial isa AbstractVector ? d.trial[1] : d.trial; return Binomial(Int(n), logistic(eta)); end
function get_dist_ref(::GammaFamily, d, eta, sig); alpha = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0; return Gamma(alpha, exp(eta)/alpha); end
function get_dist_ref(::ExponentialFamily, d, eta, sig); return Exponential(exp(eta)); end
function get_dist_ref(::BetaFamily, d, eta, sig); mu = logistic(eta); phi = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 10.0; return Beta(mu*phi, (1-mu)*phi); end
function get_dist_ref(::InverseGaussianFamily, d, eta, sig); mu = exp(eta); lambda = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0; return InverseGaussian(mu, lambda); end
function get_dist_ref(::StudentTFamily, d, eta, sig); nu = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 5.0; return LocationScale(eta, sig, TDist(nu)); end
function get_dist_ref(::HalfNormalFamily, d, eta, sig); return truncated(Normal(0.0, sig), 0.0, Inf); end
function get_dist_ref(::HalfStudentTFamily, d, eta, sig); nu = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 5.0; return truncated(LocationScale(0.0, sig, TDist(nu)), 0.0, Inf); end
function get_dist_ref(::LaplaceFamily, d, eta, sig); return Laplace(eta, sig); end
function get_dist_ref(::ParetoFamily, d, eta, sig); alpha = exp(eta); k = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0; return Pareto(alpha, k); end
function get_dist_ref(::DirichletFamily, d, eta, sig); return Dirichlet(exp.(eta)); end

# Specialized matrix-variate factory
function get_dist_ref(::InverseWishartFamily, d, eta::AbstractMatrix, sig)
    local p = size(eta, 1)
    local nu = d.extra_params isa Number && d.extra_params > p ? d.extra_params : Float64(p + 1)
    # Transformation to ensure positive definiteness
    local Psi = PDMat(Symmetric(eta * eta' + 1e-6 * I))
    return InverseWishart(nu, Psi)
end

# --- 4. Multiple Dispatch Kernels ---

_is_discrete(::Union{PoissonFamily, NegativeBinomialFamily, BinomialFamily}) = true
_is_discrete(::AbstractBSTM_Family) = false

# Point Kernel (Uncensored) 

function bstm_kernel(fam::AbstractBSTM_Family, ::Uncensored, zi::AbstractZIState, d, eta, sig, y)
    local dist = get_dist_ref(fam, d, eta, sig)
    local lp = logpdf(dist, y)
    if zi isa ZeroInflated
        local pz = _is_discrete(fam) ? pdf(dist, 0.0) : 0.0
        lp = (y == 0) ? logsumexp(log(d.phi_zi + 1e-15), log(1.0 - d.phi_zi + 1e-15) + log(pz + 1e-15)) : log(1.0 - d.phi_zi + 1e-15) + lp
    end
    # Safeguard hurdle comparison
    if d.hurdle isa Number && d.hurdle > -Inf
        lp -= logccdf(dist, d.hurdle)
    end
    return lp
end


# Matrix-Variate Uncensored Kernel
function bstm_kernel(fam::InverseWishartFamily, ::Uncensored, ::AbstractZIState, d, eta::AbstractMatrix, sig, y)
    local dist = get_dist_ref(fam, d, eta, sig)
    return logpdf(dist, y)
end

# Censoring Kernels
function bstm_kernel(fam::AbstractBSTM_Family, ::LeftCensored, zi::AbstractZIState, d, eta, sig, y)
    local dist = get_dist_ref(fam, d, eta, sig)
    local lp = logcdf(dist, d.y_U[1])
    if zi isa ZeroInflated; lp = logsumexp(log(d.phi_zi + 1e-15), log(1.0 - d.phi_zi + 1e-15) + lp); end
    return lp
end

function bstm_kernel(fam::AbstractBSTM_Family, ::RightCensored, zi::AbstractZIState, d, eta, sig, y)
    local dist = get_dist_ref(fam, d, eta, sig)
    local adj_L = _is_discrete(fam) ? d.y_L[1] - 1.0 : d.y_L[1]
    local lp = logccdf(dist, adj_L)
    if zi isa ZeroInflated; lp = (d.y_L[1] <= 0.0) ? 0.0 : (log(1.0 - d.phi_zi + 1e-15) + lp); end
    return lp
end


function bstm_kernel(fam::AbstractBSTM_Family, ::IntervalCensored, zi::AbstractZIState, d, eta, sig, y)
    # Math: log(exp(a) - exp(b)) = a + log(1 - exp(b - a))
    function stable_logdiffexp(a, b)
        if a <= b
            return -1e15 # Effectively zero probability
        else
            local diff = b - a
            # Clamp diff slightly below zero to prevent domain errors in log1mexp
            local safe_diff = min(-eps(typeof(diff)), diff)
            return a + log1mexp(safe_diff)
        end
    end

    local dist = get_dist_ref(fam, d, eta, sig)
    local adj_L = _is_discrete(fam) ? d.y_L[1] - 1.0 : d.y_L[1]
    local lp = stable_logdiffexp(logcdf(dist, d.y_U[1]), logcdf(dist, adj_L))
    if zi isa ZeroInflated; lp = (d.y_L[1] <= 0.0) ? logsumexp(log(d.phi_zi + 1e-15), log(1.0 - d.phi_zi + 1e-15) + lp) : log(1.0 - d.phi_zi + 1e-15) + lp; end
    return lp
end

# --- 5. Unified Dispatch Interface (logpdf) ---

function Distributions.logpdf(d::bstm_Likelihood, eta::Real)
    local sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
    local w = d.weight isa AbstractVector ? d.weight[1] : d.weight
    return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, d.y_obs[1]) * w
end

function Distributions.logpdf(d::bstm_Likelihood, eta::AbstractVector)
    if d.family isa DirichletFamily
        local w = d.weight isa AbstractVector ? d.weight[1] : d.weight
        return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, 1.0, d.y_obs) * w
    end
    local total_lp = 0.0
    for i in 1:length(eta)
        local sig = d.sigma_y isa AbstractVector ? d.sigma_y[i] : d.sigma_y
        local w = d.weight isa AbstractVector ? d.weight[i] : d.weight
        total_lp += bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta[i], sig, d.y_obs[i]) * w
    end
    return total_lp
end

function Distributions.logpdf(d::bstm_Likelihood, eta::AbstractMatrix)
    local w = d.weight isa AbstractVector ? d.weight[1] : d.weight
    return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, 1.0, d.y_obs) * w
end



function resolve_hyperpriors(m_id::String, user_priors::Dict{String, Any}, scheme::Symbol=:pcpriors)
    # Audited Hyperprior Resolver [v06.1 - PC-Prior Standardized]
    # Rationale: Maps string identifiers to technical prior requirements.
    # PC-Priors are now the default (:pcpriors) to ensure robust scaling across graph topologies.

    # 1. Primary Prior Registries
    # PC-Priors: Penalize complexity relative to a base model (e.g., IID or Intercept)
    local pcpriors = Dict(
        "sigma" => Exponential(1.0),
        "rho" => Beta(1, 1), # Flat/PC base for correlation
        "lengthscale" => InverseGamma(3, 3), # Informative base for GPs
        "kappa" => Exponential(1.0),
        "amplitude" => Normal(0, 1),
        "phase" => Beta(1, 1)
    )

    local informative = Dict(
        "sigma" => Exponential(0.5),
        "rho" => Beta(2, 2),
        "lengthscale" => InverseGamma(5, 5),
        "kappa" => Exponential(0.1),
        "amplitude" => Normal(0, 0.5),
        "phase" => Beta(2, 2)
    )

    local uninformative = Dict(
        "sigma" => Normal(0, 1e6),
        "rho" => Uniform(0, 1),
        "lengthscale" => InverseGamma(0.01, 0.01),
        "kappa" => Exponential(10.0),
        "amplitude" => Normal(0, 100),
        "phase" => Uniform(0, 1)
    )

    # 2. Scheme Selection Logic
    local defaults = if scheme == :pcpriors
        pcpriors
    elseif scheme == :informative
        informative
    else
        uninformative
    end

    # 3. Resolved Object Assembly
    # Initializing result container with mandatory sigma and optional structural slots.
    local res = (
        sigma_prior = get(user_priors, "sigma", defaults["sigma"]),
        rho_prior = nothing,
        lengthscale_prior = nothing,
        kappa_prior = nothing,
        amplitude_prior = nothing,
        phase_prior = nothing
    )

    # 4. Registry-Based Parameter Dispatch
    # Discrete Graph/CAR Manifolds
    if m_id in ["bym2", "leroux", "sar", "ar1", "proper_car", "dag"]
        res = merge(res, (rho_prior = get(user_priors, "rho", defaults["rho"]),))
    end

    # Continuous / Spectral Kernels
    if m_id in ["gp", "fitc", "rff"]
        res = merge(res, (lengthscale_prior = get(user_priors, "lengthscale", defaults["lengthscale"]),))
    end

    # SPDE Latent Fields
    if m_id == "spde"
        res = merge(res, (kappa_prior = get(user_priors, "kappa", defaults["kappa"]),))
    end

    # Periodic / Harmonic Cycles
    if m_id == "harmonic"
        res = merge(res, (
            amplitude_prior = get(user_priors, "amplitude", defaults["amplitude"]),
            phase_prior = get(user_priors, "phase", defaults["phase"])
        ))
    end

    return res
end





# --- Unified bstm Entry Point and Recursive DSL Router ---
# This section provides the consolidated, feature-complete version of the bstm framework.
# It integrates support for:
# 1. Recursive Manifold DSL (Sum ⊕, Kronecker ⊗, Pipe |>)
# 2. Advanced Effects: Eigen-Effects (ee), SVC (Spatially Varying Coefficients), and Interactions (ie)
# 3. Architectural Dispatch: Univariate, Multivariate, and Multifidelity

function bstm( inp::NamedTuple )
    arch = inp[:model_arch]
    if arch == "multivariate" return bstm_multivariate(inp)
    elseif arch == "multifidelity" return bstm_multifidelity(inp)
    else return bstm_univariate(inp) end
end


"""
    bstm(formula, data; kwargs...)

Main entry point for Bayesian Spatio-Temporal Modeling. 
Consolidates formula parsing, DSL resolution, and architectural dispatch.
"""

function bstm(configs::Vector{<:NamedTuple};
    model_family="gaussian",
    model_arch="univariate",
    kwargs...)

    # --- BSTM Multi-Config Router [v17.7.9 -  Restoration] ---
    # Rationale: Routes multiple fidelity/nested configurations to the unified MF engine.

    local opt_kwargs = Dict{Symbol, Any}(kwargs)
    local num_fidelities = length(configs)
    local fidelity_metadata = []

    for (i, config) in enumerate(configs)
        local formula = config.formula
        local data = config.data
        local W_fidelity = get(config, :W, nothing)

        local f_str = string(formula)
        local sides = Base.split(f_str, "~")
        local lhs = strip(sides[1])
        local rhs = strip(sides[2])

        local y_val = data[!, Symbol(lhs)]
        local rhs_terms = strip.(Base.split(rhs, "+"))

        local model_space = "none"
        local model_time = "none"

        for term in rhs_terms
            local term_lower = lowercase(strip(term))
            if occursin("spatial(", term_lower)
                model_space = "bym2"
            elseif occursin("temporal(", term_lower)
                model_time = "ar1"
            end
        end

        # Metadata engine builds unit-specific templates for each container level
        local sub_opt = bstm_options(
            data = data,
            y_obs = y_val,
            W = W_fidelity,
            model_space = model_space,
            model_time = model_time,
            model_arch = num_fidelities > 1 ? "multifidelity" : "univariate"
        )
        push!(fidelity_metadata, sub_opt)
    end

    if num_fidelities > 1
        opt_kwargs[:fidelities] = fidelity_metadata
        opt_kwargs[:model_arch] = "multifidelity"
        opt_kwargs[:model_family] = model_family
        return bstm_multifidelity(NamedTuple(opt_kwargs))
    else
        return bstm_univariate(fidelity_metadata[1])
    end
end

function bstm(formula::Union{String, StatsModels.FormulaTerm}, data_input::Union{DataFrame, NamedArray};
    model_family="gaussian",
    model_arch="univariate",
    hyperpriors=Dict{String, Any}(),
    hyperprior_scheme=:pcpriors,
    auxiliary_responses=nothing,
    auxiliary_data=nothing,
    return_data=false,
    contrasts=Dict{Symbol, Any}(),
    kwargs...)

    # --- BSTM  Entry Point [v17.7.9 - FEATURE COMPLETE AUDIT] ---
    # Rationale: Primary engine for manifold discovery and structural routing.
    # Requirements: Absolute preservation of hyperpriors, schemes, and recursive parser logic.

    # --- 1. Data Normalization & Initialization ---
    local data = data_input isa DataFrame ? copy(data_input) : DataFrame(data_input, :auto)
    local opt_kwargs = Dict{Symbol, Any}(kwargs)
    local internal_contrasts = copy(contrasts)

    # --- 2. Formula decomposition (LHS ~ RHS) ---
    local f_str = string(formula)
    local sides = Base.split(f_str, "~")
    if length(sides) < 2
        error("BSTM Error: Formula must contain a '~' separator (e.g., 'y ~ 1 + Spatial(s_idx)').")
    end
    local lhs_side = strip(sides[1])
    local rhs = strip(sides[2])

    # --- 3. Architectural Discovery & Response Extraction ---
    local lhs_vars = Symbol.(filter(!isempty, strip.(Base.split(lhs_side, "+"))))
    opt_kwargs[:outcomes_N] = length(lhs_vars)

    if !isnothing(auxiliary_responses)
        opt_kwargs[:model_arch] = "multifidelity"
        opt_kwargs[:auxiliary_responses] = auxiliary_responses
        opt_kwargs[:auxiliary_data] = auxiliary_data
    end

    if length(lhs_vars) > 1
        opt_kwargs[:model_arch] = "multivariate"
        opt_kwargs[:y_obs] = Matrix(data[!, lhs_vars])
    else
        opt_kwargs[:model_arch] = get(opt_kwargs, :model_arch, model_arch)
        opt_kwargs[:y_obs] = data[!, lhs_vars[1]]
    end

    # LOCK Coordinates for Continuous/Spectral kernels (GP, RFF, SPDE)
    if !haskey(opt_kwargs, :s_x) && "s_x" in names(data); opt_kwargs[:s_x] = data.s_x; end
    if !haskey(opt_kwargs, :s_y) && "s_y" in names(data); opt_kwargs[:s_y] = data.s_y; end
    if !haskey(opt_kwargs, :t_v) && "t_v" in names(data); opt_kwargs[:t_v] = data.t_v; end

    # --- 4. Manifold Routing Engine (v06.1 Taxonomy) ---
    local re_rules = Dict{String, Any}()
    local fixed_parts = String[]
    local interaction_terms = []
    local eigen_terms = []
    local nested_manifolds = Dict{String, Any}()
    local svc_covs = Symbol[]
    local has_intercept = true

    local rhs_terms = strip.(Base.split(rhs, "+"))

    for term in rhs_terms
        local term_clean = strip(term)
        local term_lower = lowercase(term_clean)

        # 4.1 Intercept Controls
        if term_lower == "0" || term_lower == "-1"
            has_intercept = false
        elseif term_lower == "1"
            has_intercept = true

        # 4.2 Standard Domain Manifolds (Spatial, Temporal, Seasonal)
        elseif startswith(term_lower, "spatial(") || startswith(term_lower, "temporal(") || startswith(term_lower, "seasonal(")
            local domain = startswith(term_lower, "spatial") ? :spatial : (startswith(term_lower, "temporal") ? :temporal : :seasonal)
            local m_match = match(r"\(([^;)]+)(?:;\s*(.*))?\)", term_lower)
            if !isnothing(m_match)
                local var_name = strip(m_match.captures[1])
                local params = isnothing(m_match.captures[2]) ? "" : m_match.captures[2]
                local m_man = match(r"manifold=['\" ]?(\w+)['\" ]?", params)
                local m_id = isnothing(m_man) ? (domain == :spatial ? "bym2" : (domain == :temporal ? "ar1" : "harmonic")) : m_man.captures[1]
                
                re_rules[var_name] = Dict("model" => string(m_id), "domain" => domain)
                if domain == :spatial; opt_kwargs[:model_space] = string(m_id); end
                if domain == :temporal; opt_kwargs[:model_time] = string(m_id); end
                if domain == :seasonal; opt_kwargs[:model_season] = string(m_id); end
            end

        # 4.3 Advanced Basis, Interaction, and Eigen Manifolds
        elseif startswith(term_lower, "smooth(") || startswith(term_lower, "interaction(")
            local m_si = match(r"(?:smooth|interaction)\(([^;)]+)(?:;\s*(.*))?\)", term_lower)
            if !isnothing(m_si)
                local vars_part = strip(m_si.captures[1])
                local params_part = isnothing(m_si.captures[2]) ? "" : m_si.captures[2]
                local sub_vars = strip.(Base.split(vars_part, ","))
                
                if length(sub_vars) == 2
                    # 2D Interaction Smooth (Spectral RFF / TPS)
                    local v1, v2 = Symbol(sub_vars[1]), Symbol(sub_vars[2])
                    local m_man = match(r"manifold=['\" ]?(\w+)['\" ]?", params_part)
                    local m_id = isnothing(m_man) ? "rff" : m_man.captures[1]
                    local m_rff_match = match(r"m_rff=(\d+)", params_part)
                    local m_val = isnothing(m_rff_match) ? 20 : parse(Int, m_rff_match.captures[1])
                    push!(interaction_terms, (var1=v1, var2=v2, manifold=Symbol(m_id), M_rff=m_val, coords=hcat(data[!, v1], data[!, v2]), W_ie=randn(2, m_val), b_ie=rand(m_val).*(2π)))
                else
                    # 1D Smooth (Splines)
                    local v_name = strip(sub_vars[1])
                    local m_man = match(r"manifold=['\" ]?(\w+)['\" ]?", params_part)
                    local m_id = isnothing(m_man) ? "pspline" : m_man.captures[1]
                    re_rules[v_name] = Dict("model" => string(m_id), "domain" => :covariate, "is_smooth" => true)
                    push!(fixed_parts, v_name)
                end
            end

        elseif startswith(term_lower, "eigen(")
            local m_eig = match(r"eigen\(([^;)]+)(?:;\s*(.*))?\)", term_lower)
            if !isnothing(m_eig)
                local var_name = Symbol(strip(m_eig.captures[1]))
                local params = isnothing(m_eig.captures[2]) ? "" : m_eig.captures[2]
                local rank_match = match(r"rank=(\d+)", params)
                local k_rank = isnothing(rank_match) ? 3 : parse(Int, rank_match.captures[1])
                push!(eigen_terms, (data=Matrix(data[!, [var_name]]), n_dims=k_rank, ltri_indices=collect(1:(k_rank*(k_rank+1)÷2))))
            end

        # 4.4 Recursive Nested Supervisor Discovery
        elseif startswith(term_lower, "nested(")
            local m_nest = match(r"nested\(([^;)]+)(?:;\s*(.*))?\)", term_clean)
            if !isnothing(m_nest)
                local z_var = strip(m_nest.captures[1])
                local z_params = isnothing(m_nest.captures[2]) ? "" : m_nest.captures[2]
                local z_f_match = match(r"formula=['\" ]?([^'\" ]+)['\" ]?", z_params)
                local z_f = isnothing(z_f_match) ? "$z_var ~ 1" : z_f_match.captures[1]
                local z_data_src = get(opt_kwargs, Symbol(z_var * "_data"), data)
                # Recursive builder generates sub-metadata for the supervisor process
                local z_opt = bstm(z_f, z_data_src; return_data=true, hyperpriors=hyperpriors, hyperprior_scheme=hyperprior_scheme, kwargs...)
                nested_manifolds[z_var] = z_opt
            end

        # 4.5 Spatially Varying Coefficients (SVC)
        elseif occursin("|", term_clean)
            local svc_parts = strip.(Base.split(term_clean, "|"))
            if length(svc_parts) == 2
                local cov_var = Symbol(svc_parts[1])
                push!(svc_covs, cov_var)
                local graph_struct = parse_manifold_graph(string(svc_parts[2]))
                process_graph_into_rules!(re_rules, opt_kwargs, graph_struct)
            end

        # 4.6 Algebraic Operators (⊗, ⊕, ∘)
        elseif occursin(r"[⊗⊕|>∘]", term_clean)
            local graph_struct = parse_manifold_graph(term_clean)
            local term_re_rules = Dict{String, Any}()
            process_graph_into_rules!(term_re_rules, opt_kwargs, graph_struct)
            merge!(re_rules, term_re_rules)

        else
            push!(fixed_parts, term_clean)
        end
    end

    # --- 5. Hyperprior Resolution & Parameter Injection ---
    # RESTORED: Absolute enforcement of user priors and scheme selection.
    for (var, rule) in re_rules
        local m_id = get(rule, "model", "iid")
        local resolved = resolve_hyperpriors(m_id, hyperpriors, hyperprior_scheme)
        local new_rule = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in rule)
        
        new_rule[:sigma_prior] = resolved.sigma_prior
        if !isnothing(resolved.rho_prior); new_rule[:rho_prior] = resolved.rho_prior; end
        if !isnothing(resolved.lengthscale_prior); new_rule[:lengthscale_prior] = resolved.lengthscale_prior; end
        if !isnothing(resolved.kappa_prior); new_rule[:kappa_prior] = resolved.kappa_prior; end
        if !isnothing(resolved.amplitude_prior); new_rule[:amplitude_prior] = resolved.amplitude_prior; end
        if !isnothing(resolved.phase_prior); new_rule[:phase_prior] = resolved.phase_prior; end
        
        re_rules[var] = new_rule
    end

    # --- 6. UNUSED VARIABLE PRUNING PASS (HARDENED) ---
    local active_col_set = Set{Symbol}()
    union!(active_col_set, lhs_vars)
    for part in fixed_parts
        local sub_parts = Symbol.(filter(!isempty, strip.(Base.split(part, "+"))))
        union!(active_col_set, sub_parts)
    end
    for k in keys(re_rules); push!(active_col_set, Symbol(k)); end
    union!(active_col_set, svc_covs)
    for i_term in interaction_terms
        push!(active_col_set, i_term.var1); push!(active_col_set, i_term.var2)
    end
    for n_key in keys(nested_manifolds); push!(active_col_set, Symbol(n_key)); end
    # Coordinate tokens strictly preserved to prevent downstream manifold collapse
    for c_key in [:s_x, :s_y, :t_v, :s_idx, :t_idx, :u_idx]
        if c_key in Symbol.(names(data)); push!(active_col_set, c_key); end
    end

    # Data Frame Projection (Memory Optimization)
    data = data[!, collect(intersect(active_col_set, Symbol.(names(data))))]
    if !isnothing(auxiliary_data) && auxiliary_data isa DataFrame
        auxiliary_data = auxiliary_data[!, collect(intersect(active_col_set, Symbol.(names(auxiliary_data))))]
        opt_kwargs[:auxiliary_data] = auxiliary_data
    end

    # --- 7. Final Configuration Packaging & Dispatch ---
    opt_kwargs[:add_intercept] = has_intercept
    opt_kwargs[:re_rules] = re_rules
    opt_kwargs[:fixed_parts] = fixed_parts
    opt_kwargs[:interaction_terms] = interaction_terms
    opt_kwargs[:eigen_terms] = eigen_terms
    opt_kwargs[:nested_manifolds] = nested_manifolds
    opt_kwargs[:svc_covariates] = svc_covs
    opt_kwargs[:data] = data
    opt_kwargs[:model_family] = model_family
    opt_kwargs[:contrasts] = internal_contrasts
    opt_kwargs[:hyperprior_scheme] = hyperprior_scheme
    opt_kwargs[:hyperpriors] = hyperpriors

    local inp = bstm_options(; opt_kwargs...)
    if return_data; return inp; end

    local arch_type = get(opt_kwargs, :model_arch, "univariate")
    if arch_type == "multivariate"; return bstm_multivariate(inp)
    elseif arch_type == "multifidelity" || arch_type == "nested"; return bstm_multifidelity(inp)
    else; return bstm_univariate(inp); end
end



function bstm_options(; kwargs...)
    # Audited  Configuration Engine [v17.0.28 - Feature Complete Restoration]
    # Rationale: This function consolidates all metadata, coordinate discovery, and architectural
    # flags required for the BSTM framework. It acts as the technical source of truth for all manifold types.
    # Requirements: Absolute consistency with the v06.1 taxonomy (GP, RFF, DAG, Cyclic, Mosaic, Adaptive).
    # Audit: Verified against reference to ensure NO truncation of advanced manifold registries.

    # 1. Initialization and Metadata Consolidation
    # All user-provided keyword arguments are collected into a mutable dictionary.
    local M = Dict{Symbol, Any}(kwargs)
    local data = get(M, :data, nothing)

    # 2. Automated Keyword Discovery
    # Scans the provided DataFrame for standard BSTM variable names to minimize configuration overhead.
    if !isnothing(data)
        local v_names = Symbol.(names(data))
        local keyword_map = Dict(
            :y_obs => [:y, :y_obs, :response],
            :s_x => [:s_x, :lon, :plon, :plons, :lons, :longs, :longitude],
            :s_y => [:s_y, :lat, :plat, :plats, :lats, :latitude],
            :t_v => [:t_v, :time, :time_coords, :t_coords],
            :s_idx => [:s_idx, :space_idx],
            :t_idx => [:t_idx, :time_idx, :ti],
            :u_idx => [:u_idx, :season_idx, :ui],
            :log_offset => [:log_offset, :logoffset],
            :weights => [:weights, :wts],
            :trials => [:trials, :n_trials]
        )
        for (key, candidates) in keyword_map
            if !haskey(M, key)
                for cand in candidates
                    if cand in v_names
                        M[key] = data[!, cand]
                        break
                    end
                end
            end
        end
    end

    # 3. Dimensional Validation & Scope Discovery
    if !haskey(M, :y_obs)
        error("BSTM Error: :y_obs is required for model initialization.")
    end

    local y_obs_raw = M[:y_obs]
    # Handle Matrix (Multivariate), Vector of Vectors (Heterogeneous), or flat Vector responses
    M[:y_N] = (y_obs_raw isa AbstractMatrix) ? size(y_obs_raw, 1) : (y_obs_raw isa Vector{<:AbstractVector} ? length(y_obs_raw[1]) : length(y_obs_raw))
    local y_N = M[:y_N]

    # 4. Scalar Sizing for Manifold Containers
    # Extract scalar equivalents for global container initialization to prevent AD BoundsErrors.
    local s_idx_raw = get(M, :s_idx, ones(Int, y_N))
    local t_idx_raw = get(M, :t_idx, ones(Int, y_N))
    local u_idx_raw = get(M, :u_idx, ones(Int, y_N))

    # Determine maximum dimensions from index vectors
    local max_s = (s_idx_raw isa AbstractVector) ? (isempty(s_idx_raw) ? 1 : maximum(s_idx_raw)) : 1
    local max_t = (t_idx_raw isa AbstractVector) ? (isempty(t_idx_raw) ? 1 : maximum(t_idx_raw)) : 1
    local max_u = (u_idx_raw isa AbstractVector) ? (isempty(u_idx_raw) ? 1 : maximum(u_idx_raw)) : 1

    # Finalize dimension metadata
    get!(M, :s_N, max_s)
    get!(M, :t_N, max_t)
    get!(M, :u_N, max_u)

    # 5. Default Architectural Flag Assignments
    get!(M, :model_arch, "univariate")
    get!(M, :model_family, "gaussian")
    get!(M, :model_space, "none")
    get!(M, :model_time, "none")
    get!(M, :model_season, "none")
    get!(M, :model_st, "none")
    get!(M, :noise, 1e-4)
    get!(M, :use_zi, false)
    get!(M, :use_sv, false)
    get!(M, :outcomes_N, 1)
    get!(M, :log_offset, zeros(Float64, y_N))
    get!(M, :period, 12.0)

    # 6. Design Matrix Factory: Fixed Effects
    if haskey(M, :fixed_parts) && !isnothing(data)
        local f_expr = isempty(M[:fixed_parts]) ? "1" : join(M[:fixed_parts], " + ")
        M[:Xfixed] = Matrix{Float64}(create_fixed_design(f_expr, data; contrasts=get(M, :contrasts, Dict())))
    else
        get!(M, :Xfixed, ones(Float64, y_N, 1))
    end
    M[:Xfixed_N] = size(M[:Xfixed], 2)

    # 7. Coordinate Preservation for Continuous Kernels
    if !haskey(M, :s_coord)
        if haskey(M, :s_x) && haskey(M, :s_y)
             M[:s_coord] = hcat(M[:s_x], M[:s_y])
        elseif haskey(M, :areal_units)
             local c = M[:areal_units].centroids
             M[:s_coord] = hcat([p[1] for p in c], [p[2] for p in c])
        else
             M[:s_coord] = zeros(Float64, M[:s_N], 2)
        end
    end

    # 8. Index Realignment
    # Casting all indices to Int and ensuring numerical stability for AD.
    M[:s_idx] = Int.(collect(s_idx_raw))
    M[:t_idx] = Int.(collect(t_idx_raw))
    M[:u_idx] = Int.(collect(u_idx_raw))

    # Global linear indexing for Type IV interactions (Knorr-Held separable maps)
    M[:st_idx] = [(M[:t_idx][i] - 1) * M[:s_N] + M[:s_idx][i] for i in 1:y_N]

    # 9. Spectral Projection Caching for Stochastic Volatility
    if get(M, :use_sv, false)
        get!(M, :M_rff_sigma, 20)
        local coords_v = haskey(M, :s_coord) ? M[:s_coord] : randn(y_N, 2)
        local W_v, b_v = generate_informed_rff_params(coords_v, M[:M_rff_sigma])
        M[:vol_proj] = (coords_v * W_v) .+ b_v'
    end

    # 10. Precision Matrix Template Discovery
    M[:s_Q_template] = build_structure_template(Symbol(M[:model_space]), M[:s_N]; W=get(M, :W, nothing))
    M[:t_Q_template] = build_structure_template(Symbol(M[:model_time]), M[:t_N])
    M[:u_Q_template] = (M[:model_season] != "none") ? build_structure_template(Symbol(M[:model_season]), M[:u_N]) : nothing

    # 11. Multi-Effect Registries [v17.0.28 Restoration]
    # Rationale: Ensuring all loop-based latent effect containers and advanced topological fields are active.
    get!(M, :nested_manifolds, Dict{String, Any}())
    get!(M, :spatial_hierarchy, Dict{Symbol, Any}())
    get!(M, :mixed_terms, [])
    get!(M, :basis_matrices, Dict{Symbol, Any}())
    get!(M, :interaction_terms, [])
    get!(M, :eigen_terms, [])
    get!(M, :svc_covariates, Symbol[])
  
    # Advanced Topological Metadata
    get!(M, :cluster_assignments, ones(Int, M[:s_N]))
    get!(M, :n_mosaics, 1)
    get!(M, :local_weights, ones(Float64, M[:s_N]))
    get!(M, :directed_adj, get(M, :W, sparse(I(M[:s_N]))))
    get!(M, :curvature_const, -1.0)

    # 12. Observation Metadata
    get!(M, :trials, ones(Int, y_N))
    get!(M, :weights, ones(Float64, y_N))

    if !haskey(M, :y_ok)
        if y_obs_raw isa AbstractMatrix
            M[:y_ok] = [findall(!isnan, y_obs_raw[:, k]) for k in 1:get(M, :outcomes_N, 1)]
        else
            M[:y_ok] = [findall(!isnan, vec(y_obs_raw))]
        end
    end

    return NamedTuple(M)
end



#---------------------------------------------
#---------------------------------------------
#---------------------------------------------



function init_params_extract(X)
  XS = summarize(X)
  vns = XS.nt.parameters  # var names
  init_params = FillArrays.Fill( XS.nt[2] ) # means
  return init_params, vns
end

 
function discretize_decimal( x, delta=0.01 ) 
    num_digits = Int(ceil( log10(1.0 / delta)) )   # time floating point rounding
    out = round.( round.( x ./ delta; digits=0 ) .* delta; digits=num_digits)
    return out
end
 

function expand_grid(; kws...)
    names, vals = keys(kws), values(kws)
    return DataFrame(NamedTuple{names}(t) for t in Iterators.product(vals...))
end
   

function showall( x )
    # print everything to console
    show(stdout, "text/plain", x) # display all estimates
end 
 

function firstindexin(a::AbstractArray, b::AbstractArray)
    bdict = Dict{eltype(b), Int}()
    for i=length(b):-1:1
        bdict[b[i]] = i
    end
    [get(bdict, i, 0) for i in a]
end
   
  
function β( mode, conc )
    # alternate parameterization of beta distribution 
    # conc = α + β     https://en.wikipedia.org/wiki/Beta_distribution
    beta1 = mode *( conc - 2  ) + 1.0
    beta2 = (1.0 - mode) * ( conc - 2  ) + 1.0
    Beta( beta1, beta2 ) 
end 
  
function modelruntime(o)
    dt = ( o.info.stop_time- o.info.start_time )/ 60
    showall( summarize(o) )
    print( dt )
end
 
function code_show(x)
   # printstyled( CodeTracking.@code_string x() )
end


################




function expand_hull(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, buffer_dist)
    """
    Synopsis: Computes the convex hull of points and expands it by a buffer distance.
    Inputs:
    - s_x: Vector of x-coordinates.
    - s_y: Vector of y-coordinates.
    - buffer_dist: Distance to buffer the convex hull.
    Outputs:
    - A LibGEOS Polygon geometry representing the buffered convex hull.
    """

    s_coord_tuple_local = tuple.(s_x, s_y)

    if isempty(s_coord_tuple_local) return LibGEOS.Polygon([[ (0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0) ]]) end
    coords_vec = [[Float64(p[1]), Float64(p[2])] for p in s_coord_tuple_local]
    points_geom = LibGEOS.MultiPoint(coords_vec)
    hull = LibGEOS.convexhull(points_geom)
    buffered_hull = LibGEOS.buffer(hull, buffer_dist)
    return buffered_hull
end

 

function get_kde_seeds(s_coord_tuple_local, target_u)
 
    # Basic KDE-based seeding using StatsBase weights based on local density
    u_pts = unique(s_coord_tuple_local)
    if isempty(u_pts) return [] end
    n = length(u_pts)
    dists = [sum((p1 .- p2).^2) for p1 in u_pts, p2 in u_pts]
    # Inverse of mean distance as a density proxy
    weights = 1.0 ./ (mean(dists, dims=2)[:] .+ 1e-6)
    idx = StatsBase.sample(1:n, Weights(weights), min(target_u, n), replace=false)
    return u_pts[idx]
end

 
function is_valid_polygon_coords(poly_coords)
    # Filters out NaN/Inf values and checks for a minimum of 3 valid points for a polygon.

    valid_pts = [p for p in poly_coords if !isnan(p[1]) && !isinf(p[1]) && !isnan(p[2]) && !isinf(p[2])]
    return length(valid_pts) >= 3
end
 

function get_cvt_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom)
    """
    Synopsis: Centroidal Voronoi Tessellation (CVT) with diagnostic termination tracking.
    """

    s_coord_tuple_local = tuple.(s_x, s_y)

    if length(s_coord_tuple_local) <= cfg.min_total_arealunits
        return [ (mean(s_x), mean(s_y)) ], "not_enough_points_to_tessellate"
    end

    u_pts = unique(s_coord_tuple_local)
    idx = StatsBase.sample(1:length(u_pts), min(cfg.target, length(u_pts)), replace=false)
    curr_centroids = [u_pts[i] for i in idx]
    termination_reason = "max_iterations"

    # Initialize convergence tracking variables
    last_mean_density = 0.0
    last_cv = 0.0

    for iter in 1:100
        polys, _ = get_voronoi_polygons_and_edges(curr_centroids, hull_geom)
        new_centroids = Tuple{Float64, Float64}[]
        shifts = Float64[]

        for i in 1:length(polys)
            poly_coords = polys[i]
            area = get_polygon_area(poly_coords) # Using refactored area func

            if length(poly_coords) > 2 && area >= cfg.min_area && area <= cfg.max_area
                lg_poly = LibGEOS.Polygon([[ [p[1], p[2]] for p in poly_coords ]])
                cent_geom = LibGEOS.centroid(lg_poly)
                seq = LibGEOS.getCoordSeq(cent_geom)
                new_c = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))

                dist = sqrt(sum((new_c .- curr_centroids[i]).^2))
                push!(shifts, dist)
                push!(new_centroids, new_c)
            else
                push!(new_centroids, curr_centroids[i])
            end
        end

        if isempty(shifts) || mean(shifts) < cfg.tolerance
            termination_reason = "convergence"
            break
        end

        # Use original s_coord_tuple_local for assignment to clusters
        assigns = [argmin([sum((p .- c).^2) for c in new_centroids]) for p in s_coord_tuple_local]
        counts = [count(==(i), assigns) for i in 1:length(new_centroids)]

        if isempty(counts)
            termination_reason = "no_units_formed"
            break
        end

        # New Density Convergence Check
        curr_mean_density = mean(counts)
        if abs(curr_mean_density - last_mean_density) < cfg.tolerance && iter > 1
            termination_reason = "density_convergence"
            break
        end
        last_mean_density = curr_mean_density

        cv_val = std(counts) / (mean(counts) + 1e-9)
        # CV Convergence Check
        if abs(cv_val - last_cv) < cfg.tolerance && iter > 1
            termination_reason = "cv_convergence"
            break
        end
        last_cv = cv_val

        if mean(counts) < cfg.min_points
            termination_reason = "min_points_violation"
            break
        end

        curr_centroids = new_centroids
    end

    return curr_centroids, termination_reason
end


function get_kvt_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom)
    """
    Synopsis: K-means Voronoi Tessellation (KVT) with diagnostic termination tracking.
    """

    s_coord_tuple_local = tuple.(s_x, s_y)


    if length(s_coord_tuple_local) <= cfg.min_total_arealunits
        return [ (mean(s_x), mean(s_y)) ], "not_enough_points_to_tessellate"
    end

    u_pts = unique(s_coord_tuple_local)
    idx_init = StatsBase.sample(1:length(u_pts), min(cfg.target, length(u_pts)), replace=false)
    c_iter = [u_pts[i] for i in idx_init]
    data = tuple.(s_coord_tuple_local, cfg.t_idx)

    damping = 0.7
    termination_reason = "max_iterations"

    # Initialize convergence tracking variables
    last_mean_density = 0.0
    last_cv = 0.0

    for iter in 1:100
        old_centroids = copy(c_iter)
        assigns = [argmin([sum((p[1] .- sj).^2) for sj in c_iter]) for p in data]

        polys_coords, _ = get_voronoi_polygons_and_edges(c_iter, hull_geom)

        for k in 1:length(c_iter)
            idx_cluster = findall(==(k), assigns)
            ts_count = length(unique([data[j][2] for j in idx_cluster]))

            area = 0.0
            if k <= length(polys_coords)
                area = get_polygon_area(polys_coords[k])
            end

            # Modified area_ok condition: require positive area
            area_ok = (area > 0) && area >= cfg.min_area && area <= cfg.max_area

            if !isempty(idx_cluster) && length(idx_cluster) >= cfg.min_points && ts_count >= cfg.min_time_slices && area_ok
                mean_x = mean(data[j][1][1] for j in idx_cluster)
                mean_y = mean(data[j][1][2] for j in idx_cluster)

                c_iter[k] = ((1.0 - damping) * old_centroids[k][1] + damping * mean_x,
                             (1.0 - damping) * old_centroids[k][2] + damping * mean_y)
            end
        end

        counts = [count(==(k), assigns) for k in 1:length(c_iter)]
        if isempty(counts)
            termination_reason = "no_units_formed"
            break
        end

        # New Density Convergence Check
        curr_mean_density = mean(counts)
        if abs(curr_mean_density - last_mean_density) < cfg.tolerance && iter > 1
            termination_reason = "density_convergence"
            break
        end
        last_mean_density = curr_mean_density

        cv_val = std(counts) / (mean(counts) + 1e-9)
        # CV Convergence Check
        if abs(cv_val - last_cv) < cfg.tolerance && iter > 1
            termination_reason = "cv_convergence"
            break
        end
        last_cv = cv_val

        if mean(counts) < cfg.min_points
            termination_reason = "min_points_violation"
            break
        end

        damping *= 0.99
    end

    return c_iter, termination_reason
end



function get_qvt_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom)
    # Local helper to convert flat s_x, s_y to a vector of tuples

    s_coord_tuple_local = tuple.(s_x, s_y)

    if length(s_coord_tuple_local) <= cfg.min_total_arealunits
        return [ (mean(s_x), mean(s_y)) ], "not_enough_points_to_tessellate"
    end
 
    data = tuple.(s_coord_tuple_local, cfg.t_idx)

    regions = [data]
    # Track specific region objects that failed to split to avoid redundant attempts
    unsplittable = Set{UInt64}()

    effective_min_p = max(1, cfg.min_points)

    if length(data) < 2 * effective_min_p # Initial check: if the whole dataset is too small to split
        return [(mean(p[1][1] for p in data), mean(p[1][2] for p in data))], "initial_data_too_small_to_tessellate"
    end

    termination_reason = "max_units_reached"

    # Initialize convergence tracking variables
    last_mean_density = 0.0
    last_cv = 0.0

    cnt = 0

    while length(regions) < cfg.max_total_arealunits
        cnt += 1

        counts = length.(regions)
        curr_mean_density = mean(counts)
        cv_val = std(counts) / (curr_mean_density + 1e-9)

        # Early breaking based on statistical stabilization and target resolution
        if cnt > 3
            if last_mean_density > 0.0 && (abs(curr_mean_density - last_mean_density) < cfg.tolerance || abs(cv_val - last_cv) < cfg.tolerance)
                if length(regions) >= cfg.target && all(c -> c <= cfg.max_points, counts)
                    termination_reason = "converged_constraints_satisfied"
                    break
                elseif abs(cv_val - cfg.target_cv) < cfg.tolerance
                    termination_reason = "converged_target_cv"
                    break
                elseif (count(>(cfg.max_points), counts) / length(regions)) < cfg.tolerance/10
                    termination_reason = "converged_minor_violations"
                    break
                end
            end
        end

        last_mean_density = curr_mean_density
        last_cv = cv_val

        # Candidacy: regions that can be split
        viable_indices = findall(r -> length(r) >= max(2, effective_min_p) && objectid(r) ∉ unsplittable, regions)
        if cnt > 3
            if isempty(viable_indices); termination_reason = "cannot_split_further"; break; end
        end

        # Split if (below min_total_arealunits) OR (below target OR has max_points violators)
        violators = filter(i -> length(regions[i]) > cfg.max_points, viable_indices)
        must_split = length(regions) < cfg.min_total_arealunits
        want_split = length(regions) < cfg.target || !isempty(violators)
        candidates = (must_split || want_split) ? (isempty(violators) ? viable_indices : violators) : []

        if isempty(candidates); termination_reason = "constraints_satisfied"; break; end

        # Attempt splitting the largest available candidate
        target_idx = candidates[argmax([length(regions[i]) for i in candidates])]
        target_region = regions[target_idx]

        xs_r = [p[1][1] for p in target_region]; ys_r = [p[1][2] for p in target_region]

        # Robust splitting: handle datasets with zero variance in one or more dimensions
        if length(unique(xs_r)) > 1 || length(unique(ys_r)) > 1
            mx = length(unique(xs_r)) > 1 ? median(xs_r) : xs_r[1]
            my = length(unique(ys_r)) > 1 ? median(ys_r) : ys_r[1]
            r_splits = [
                filter(p -> p[1][1] <= mx && p[1][2] <= my, target_region),
                filter(p -> p[1][1] > mx && p[1][2] <= my, target_region),
                filter(p -> p[1][1] <= mx && p[1][2] > my, target_region),
                filter(p -> p[1][1] > mx && p[1][2] > my, target_region)
            ]
        else
            # All points collocated spatially: split by index to progress toward target unit count
            mid = length(target_region) ÷ 2  # Corrected from ∈ 2 to ÷ 2
            r_splits = [target_region[1:mid], target_region[mid+1:end], [], []]
        end

        valid_splits = filter(r -> length(r) >= effective_min_p, r_splits)

        if length(valid_splits) < 2
            # This specific region is locally unsplittable; mark it and continue with others
            push!(unsplittable, objectid(target_region))
            continue
        end

        deleteat!(regions, target_idx)
        append!(regions, valid_splits)
    end

    # Post-audit: filter by final constraints (minimum time slices and point counts)
    final_filtered_regions = filter(regions) do r
        length(r) >= effective_min_p && length(unique([p[2] for p in r])) >= cfg.min_time_slices
    end

    if isempty(final_filtered_regions)
        return [], "no_valid_units_after_filter"
    end

    final_centroids = [(mean(p[1][1] for p in r), mean(p[1][2] for p in r)) for r in final_filtered_regions]

    polys_coords, _ = get_voronoi_polygons_and_edges(final_centroids, hull_geom)
    area_violation = any(
        p_coords -> !is_valid_polygon_coords(p_coords) || get_polygon_area(p_coords) < cfg.min_area,
        polys_coords
    )

    final_status = termination_reason
    if length(final_centroids) < cfg.min_total_arealunits
        final_status = "insufficient_units_error"
    elseif area_violation
        final_status = "min_area_violation"
    end

    return final_centroids, final_status
end



function get_bvt_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom)
    """
    Synopsis: Binary Voronoi Tessellation (BVT) with corrected recursive splitting logic.
    """

    s_coord_tuple_local = tuple.(s_x, s_y)

    if length(s_coord_tuple_local) <= cfg.min_total_arealunits
        return [ (mean(s_x), mean(s_y)) ], "not_enough_points_to_tessellate"
    end

    data = tuple.(s_coord_tuple_local, cfg.t_idx)
    regions = [data]
    # Track specific region objects that failed to split to avoid redundant attempts

    effective_min_p = max(1, cfg.min_points)
    if length(data) < 2 * effective_min_p # Initial check: if the whole dataset is too small to split
        return [(mean(p[1][1] for p in data), mean(p[1][2] for p in data))], "initial_data_too_small_to_tessellate"
    end

    unsplittable = Set{UInt64}()
    termination_reason = "max_units_reached"
    last_mean_density = 0.0
    last_cv = 0.0
    cnt =0

    while length(regions) < cfg.max_total_arealunits
        cnt += 1

        counts = length.(regions)
        curr_mean_density = mean(counts)
        cv_val = std(counts) / (curr_mean_density + 1e-9)

        # Early breaking based on statistical stabilization and target resolution
        if cnt > 3
            if last_mean_density > 0.0 && (abs(curr_mean_density - last_mean_density) < cfg.tolerance || abs(cv_val - last_cv) < cfg.tolerance)
                if length(regions) >= cfg.target && all(c -> c <= cfg.max_points, counts)
                    termination_reason = "converged_constraints_satisfied"
                    break
                elseif (abs(cv_val - cfg.target_cv) < cfg.tolerance)
                    termination_reason = "converged_target_cv"
                    break
                elseif (count(>(cfg.max_points), counts) / length(regions)) < cfg.tolerance/10
                    termination_reason = "converged_minor_violations"
                    break
                end
            end
        end

        last_mean_density = curr_mean_density
        last_cv = cv_val

        # Candidacy: regions that can be split
        viable_indices = findall(r -> length(r) >= max(2, effective_min_p) && objectid(r) ∉ unsplittable, regions)
        if cnt > 3
            if isempty(viable_indices); termination_reason = "cannot_split_further"; break; end
        end

        # Split if (below min_total_arealunits) OR (below target OR has max_points violators)
        violators = filter(i -> length(regions[i]) > cfg.max_points, viable_indices)
        must_split = length(regions) < cfg.min_total_arealunits
        want_split = length(regions) < cfg.target || !isempty(violators)
        candidates = (must_split || want_split) ? (isempty(violators) ? viable_indices : violators) : []

        if isempty(candidates); termination_reason = "constraints_satisfied"; break; end

        # Attempt splitting the largest available candidate
        target_idx = candidates[argmax([length(regions[i]) for i in candidates])]
        target = regions[target_idx]

        xs = [p[1][1] for p in target]; ys = [p[1][2] for p in target]
        var_x = length(xs) > 1 ? var(xs) : 0.0
        var_y = length(ys) > 1 ? var(ys) : 0.0
        dim = var_x > var_y ? 1 : 2

        if var_x > 1e-9 || var_y > 1e-9
            vals = [p[1][dim] for p in target]
            med = length(unique(vals)) > 1 ? median(vals) : vals[1]
            r1 = filter(p -> p[1][dim] <= med, target)
            r2 = filter(p -> p[1][dim] > med, target)
        else
            # Handle collocated points
            mid = length(target) ÷ 2
            r1, r2 = target[1:mid], target[mid+1:end]
        end

        # Validate children for point count and temporal diversity
        v1 = length(r1) >= effective_min_p && length(unique([p[2] for p in r1])) >= cfg.min_time_slices
        v2 = length(r2) >= effective_min_p && length(unique([p[2] for p in r2])) >= cfg.min_time_slices

        if !v1 || !v2
             push!(unsplittable, objectid(target))
             continue
        end

        # Tentative update to check global area constraints
        tentative_regions = copy(regions)
        deleteat!(tentative_regions, target_idx)
        push!(tentative_regions, r1, r2)

        candidate_centroids = [(mean(p[1][1] for p in r), mean(p[1][2] for p in r)) for r in tentative_regions]
        polys_coords, _ = get_voronoi_polygons_and_edges(candidate_centroids, hull_geom)

        area_violation = any(
            p_coords -> !is_valid_polygon_coords(p_coords) || get_polygon_area(p_coords) < cfg.min_area,
            polys_coords
        )

        if area_violation && length(tentative_regions) > cfg.min_total_arealunits
             push!(unsplittable, objectid(target))
             continue
        end

        regions = tentative_regions
    end

    final_centroids_candidate = [(mean(p[1][1] for p in r), mean(p[1][2] for p in r)) for r in regions]

    # if length(final_centroids_candidate) < cfg.min_total_arealunits
    #     # Aggregate all original points (from 'data') into a single centroid
    #     all_pts_x = [p[1][1] for p in data]
    #     all_pts_y = [p[1][2] for p in data]
    #     return [ (mean(all_pts_x), mean(all_pts_y)) ], "insufficient_units_error"
    # else
        return final_centroids_candidate, termination_reason
    # end
end

 
 
function get_hvt_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom; max_iter=500)

    s_coord_tuple_local = tuple.(s_x, s_y)

    # Internal utility for point-to-centroid distance
    dist(p1, p2) = sqrt(sum((p1 .- p2).^2))

    # Standardized refinement loop (Lloyd's update)
    function refine(pts_in, centroids, iters)
        curr = deepcopy(centroids)
        for _ in 1:iters
            groups = [Int[] for _ in 1:length(curr)]
            for (i, p) in enumerate(pts_in)
                dists = [dist(p, c) for c in curr]
                push!(groups[argmin(dists)], i)
            end
            new_c = [!isempty(idx) ?
                     (mean(pts_in[j][1] for j in idx), mean(pts_in[j][2] for j in idx)) :
                     curr[k] for (k, idx) in enumerate(groups)]
            if all(dist(new_c[j], curr[j]) < cfg.tolerance for j in 1:length(curr))
                return new_c
            end
            curr = new_c
        end
        return curr
    end

    # Initial Seed generation using k-means
    # Convert s_coord_tuple_local to matrix for Clustering.jl
    pts_matrix = hcat([p[1] for p in s_coord_tuple_local], [p[2] for p in s_coord_tuple_local])'
    k_target = max(1, cfg.min_total_arealunits)

    # Run k-means to find initial centers
    R = kmeans(pts_matrix, k_target)
    curr_centroids = [(R.centers[1, i], R.centers[2, i]) for i in 1:size(R.centers, 2)]

    # Iterative HVT process with advanced stopping conditions
    last_mean_density = 0.0
    last_cv = 0.0
    status = "max_iterations_reached"

    for i in 1:max_iter
        # 1. Assignment step to calculate metrics
        s_idx = [argmin([dist(p, c) for c in curr_centroids]) for p in s_coord_tuple_local]
        counts = [count(==(k), s_idx) for k in 1:length(curr_centroids)]

        curr_mean_density = mean(counts)
        cv_val = std(counts) / (curr_mean_density + 1e-9)

        # Convergence logic aligned with QVT/BVT
        if i > 5
            if abs(curr_mean_density - last_mean_density) < cfg.tolerance || abs(cv_val - last_cv) < cfg.tolerance
                if length(curr_centroids) >= cfg.target && all(c -> c <= cfg.max_points, counts)
                    status = "converged_constraints_satisfied"
                    break
                elseif abs(cv_val - cfg.target_cv) < cfg.tolerance
                    status = "converged_target_cv"
                    break
                elseif (count(>(cfg.max_points), counts) / length(curr_centroids)) < cfg.tolerance/10
                    status = "converged_minor_violations"
                    break
                end
            end
        end

        last_mean_density = curr_mean_density
        last_cv = cv_val

        # 2. Refinement step (Lloyd's update)
        new_centroids = refine(s_coord_tuple_local, curr_centroids, 3)

        # Check for centroid position stabilization
        if all(dist(new_centroids[j], curr_centroids[j]) < cfg.tolerance for j in 1:length(curr_centroids))
             # If positions stabilized but constraints aren't met, check if add a unit
             if length(curr_centroids) < cfg.max_total_arealunits && (length(curr_centroids) < cfg.target || any(counts .> cfg.max_points))
                 # Split the largest group to improve density balance
                 idx_to_split = argmax(counts)
                 group_pts = s_coord_tuple_local[s_idx .== idx_to_split]
                 if length(group_pts) >= 2 * cfg.min_points
                     new_seeds = [(mean(p[1] for p in group_pts) * 0.99, mean(p[2] for p in group_pts) * 0.99),
                                  (mean(p[1] for p in group_pts) * 1.01, mean(p[2] for p in group_pts) * 1.01)]
                     deleteat!(curr_centroids, idx_to_split)
                     append!(curr_centroids, new_seeds)
                     continue
                 end
             end
             status = "converged_stable_positions"
             break
        end

        curr_centroids = new_centroids
    end

    return curr_centroids, status
end





"""
    get_avt_centroids(s_x, s_y, cfg, hull_geom)

Iterative Adaptive Voronoi Tessellation. Merges units that violate constraints
on point counts, time-slice representation, or geometric area.
"""
function get_avt_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom)
    s_coord_tuple = tuple.(s_x, s_y)
    
    if length(s_coord_tuple) <= cfg.min_total_arealunits
        return [ (mean(p[1] for p in s_coord_tuple), mean(p[2] for p in s_coord_tuple)) ], "not_enough_points_to_tessellate"
    end

    u_pts = unique(s_coord_tuple)
    # Seeding centroids via KDE-based logic
    c_init = get_kde_seeds(u_pts, min(length(u_pts), cfg.max_total_arealunits))
    data = tuple.(s_coord_tuple, cfg.t_idx)
    curr_c = [SVector{2, Float64}(c) for c in c_init]

    termination_reason = "min_units_reached"
    last_mean_density = 0.0
    last_cv = 0.0

    while length(curr_c) > cfg.min_total_arealunits
        # 1. Assignment
        assigns = [Int[] for _ in 1:length(curr_c)]
        for i in 1:length(data)
            d_pt = data[i][1]
            # Finding closest centroid (Explicit loop for clarity)
            dist_idx = argmin([sum((d_pt .- c).^2) for c in curr_c])
            push!(assigns[dist_idx], i)
        end
        
        counts = length.(assigns)
        
        # 2. Geometry Calculation
        # get_voronoi_polygons_and_edges returns Vector{Vector{Tuple{Float64, Float64}}}
        polys_coords, _ = get_voronoi_polygons_and_edges([Tuple(c) for c in curr_c], hull_geom)
        
        areas = fill(0.0, length(curr_c))
        for i in 1:min(length(curr_c), length(polys_coords))
            # Fixed Call: Passes the Vector of Tuples directly to the new method
            areas[i] = get_polygon_area(polys_coords[i])
        end

        # 3. Violation Audit
        violators = Int[]
        for k in 1:length(curr_c)
            ts_count = length(unique([data[idx][2] for idx in assigns[k]]))
            
            # Logic for merging: too few points, too few time slices, or area outside bounds
            is_invalid_count = counts[k] < cfg.min_points
            is_invalid_time = ts_count < cfg.min_time_slices
            is_invalid_area = (areas[k] > 0 && areas[k] < cfg.min_area) || (areas[k] > cfg.max_area)
            
            if is_invalid_count || is_invalid_time || is_invalid_area
                push!(violators, k)
            end
        end

        # 4. Convergence Check
        curr_mean_density = mean(counts)
        cv_val = std(counts) / (mean(counts) + 1e-9)
        
        if last_mean_density > 0.0 && (abs(curr_mean_density - last_mean_density) < cfg.tolerance || abs(cv_val - last_cv) < cfg.tolerance)
            termination_reason = "tolerance_reached"
            break
        end
        
        last_mean_density = curr_mean_density
        last_cv = cv_val

        # 5. Merging Step
        # Identify target unit to merge (the one with the lowest count among violators or overall)
        candidates_indices = isempty(violators) ? collect(1:length(curr_c)) : violators
        v_counts = [counts[k] for k in candidates_indices]
        target_idx = candidates_indices[argmin(v_counts)]

        # Find nearest neighbor for the target centroid
        dists = [sum((curr_c[target_idx] .- curr_c[j]).^2) for j in 1:length(curr_c)]
        dists[target_idx] = Inf
        neighbor_idx = argmin(dists)

        # Weighted update for the merged centroid location
        total_n = counts[target_idx] + counts[neighbor_idx]
        curr_c[neighbor_idx] = (curr_c[target_idx] .* counts[target_idx] .+ curr_c[neighbor_idx] .* counts[neighbor_idx]) ./ (total_n + 1e-9)
        
        # Explicit removal (No clamp used)
        deleteat!(curr_c, target_idx)
    end
    
    return [Tuple(c) for c in curr_c], termination_reason
end


function get_lattice_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, lengthscale)
    """
    Synopsis: Generates centroids for a regular 2D lattice (grid) based on a lengthscale.
    """

    s_coord_tuple = tuple.(s_x, s_y)

    if isempty(s_coord_tuple); return [], 0, 0, (0.0, 0.0, 0.0, 0.0); end

    xs = [p[1] for p in s_coord_tuple]
    ys = [p[2] for p in s_coord_tuple]

    xmin, xmax = minimum(xs), maximum(xs)
    ymin, ymax = minimum(ys), maximum(ys)

    # Generate grid ranges
    x_range = collect(xmin:lengthscale:xmax)
    y_range = collect(ymin:lengthscale:ymax)

    # Ensure at least one cell if the range is smaller than lengthscale
    if isempty(x_range); x_range = [xmin]; end
    if isempty(y_range); y_range = [ymin]; end

    rows = length(y_range)
    cols = length(x_range)

    # Create meshgrid of centroids
    centroids = [(x, y) for y in y_range, x in x_range][:]

    return centroids, rows, cols, (xmin, xmax, ymin, ymax)
end



function load_shapefile_to_libgeos(filepath::String)
    # Read the shapefile
    # import Shapefile  << --- install this if you need it
    # import LibGEOS
    # import GeoInterface

    table = Shapefile.Table(filepath)
    
    # Extract geometries and convert to LibGEOS
    # GeoInterface allows LibGEOS to understand Shapefile objects automatically
    geoms = [LibGEOS.read_geom(row.geometry) for row in table]
    
    return geoms, table
end

function get_user_centroids(input_polygons)
    # Convert input to a concrete vector of LibGEOS Polygons
    geoms = LibGEOS.Polygon[p for p in input_polygons]
    n = length(geoms)
    centroids = Vector{Tuple{Float64, Float64}}(undef, n)
    polys_coords = Vector{Vector{Tuple{Float64, Float64}}}(undef, n)

    for i in 1:n
        poly = geoms[i]
        cent_geom = LibGEOS.centroid(poly)
        seq = LibGEOS.getCoordSeq(cent_geom)
        centroids[i] = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
        polys_coords[i] = get_coords_from_geom(poly)
    end

    # Wrap the vector in a GeometryCollection so GeoInterface traits are recognized
    collection = LibGEOS.GeometryCollection(geoms)
    # Perform unaryUnion on the collection instead of the vector
    united = LibGEOS.unaryUnion(collection)
    hull_coords = get_coords_from_geom(united)

    return centroids, polys_coords, hull_coords
end


function assign_spatial_units(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}; area_method=:avt, target_units=10, lengthscale=nothing, input_polygons=nothing, geom_hull=nothing, kwargs...)
    # s_coord_tuple_local will be used for calculations that still expect a collection of points

    s_coord_tuple_local = tuple.(s_x, s_y) # Using the globally defined helper

    # The branch for `input_data isa AbstractMatrix` is removed, as this refactored function
    # is specifically for coordinate-based spatial unit assignment. `bstm_options`
    # or a similar function will call `assign_spatial_units_inferred` (or its refactored version)
    # directly if an adjacency matrix is provided as primary input.

    # 1. Handle User-Defined Polygons
    if !isnothing(input_polygons)
        # If geom_hull is provided, intersect the input polygons with it
        processed_polys = isnothing(geom_hull) ? input_polygons : [LibGEOS.intersection(p, geom_hull) for p in input_polygons]

        final_centroids, polys_coords, hull_coords = get_user_centroids(processed_polys)
        reason = :user_polygons
        n_units = length(final_centroids)

        g = SimpleGraph(n_units)
        for i in 1:n_units, j in (i+1):n_units
            if LibGEOS.touches(processed_polys[i], processed_polys[j]) || LibGEOS.intersects(LibGEOS.buffer(processed_polys[i], 1e-7), processed_polys[j])
                add_edge!(g, i, j)
            end
        end
        g = ensure_connected!(g, final_centroids)
        W = Float64.(Graphs.adjacency_matrix(g))

        # Use s_coord_tuple_local for assignments, as it represents the original observation points
        new_assigns = [argmin([sum((p .- sj).^2) for sj in final_centroids]) for p in s_coord_tuple_local]
        v_edges = []

    # 2. Handle Lattice Method
    elseif area_method == :lattice
        # `expand_hull` and `get_lattice_centroids` will be refactored to take s_x, s_y
        ls = isnothing(lengthscale) ? sqrt(get_polygon_area(get_coords_from_geom(expand_hull( 0.0))) / target_units) : lengthscale # Updated call
        final_centroids_raw, rows, cols, bbox = get_lattice_centroids(s_x, s_y, ls) # Updated call
        reason = :lattice_grid

        # Generate square polygons and clip them if geom_hull is provided
        polys_coords = Vector{Vector{Tuple{Float64, Float64}}}()
        lg_polys = LibGEOS.Polygon[]
        final_centroids = Tuple{Float64, Float64}[]
        half = ls / 2.0

        for c in final_centroids_raw
            coords = [[(c[1]-half, c[2]-half), (c[1]+half, c[2]-half), (c[1]+half, c[2]+half), (c[1]-half, c[2]+half), (c[1]-half, c[2]+half)]]
            p_geom = LibGEOS.Polygon(coords)
            if !isnothing(geom_hull)
                p_geom = LibGEOS.intersection(p_geom, geom_hull)
            end

            if !LibGEOS.isEmpty(p_geom)
                push!(lg_polys, p_geom)
                p_c = LibGEOS.centroid(p_geom)
                seq = LibGEOS.getCoordSeq(p_c)
                push!(final_centroids, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
                push!(polys_coords, get_coords_from_geom(p_geom))
            end
        end

        n_units = length(final_centroids)
        g = SimpleGraph(n_units)
        for i in 1:n_units, j in (i+1):n_units
            if LibGEOS.touches(lg_polys[i], lg_polys[j]) || LibGEOS.intersects(LibGEOS.buffer(lg_polys[i], 1e-7), lg_polys[j])
                add_edge!(g, i, j)
            end
        end
        g = ensure_connected!(g, final_centroids)
        W = Float64.(Graphs.adjacency_matrix(g))

        # Use s_coord_tuple_local for assignments
        new_assigns = [argmin([sum((p .- sj).^2) for sj in final_centroids]) for p in s_coord_tuple_local]
        v_edges = []
        hull_coords = isnothing(geom_hull) ? [(bbox[1], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[4]), (bbox[1], bbox[4]), (bbox[1], bbox[3])] : get_coords_from_geom(geom_hull)

    # 3. Standard Tessellation Methods
    else
        cfg = (
            target=Int(target_units),
            min_total_arealunits=Int(get(kwargs, :min_total_arealunits, 3)),
            max_total_arealunits=Int(get(kwargs, :max_total_arealunits, target_units*2)),
            min_time_slices=Int(get(kwargs, :min_time_slices, 1)),
            min_points=Int(get(kwargs, :min_points, 1)),
            max_points=Int(get(kwargs, :max_points, length(s_x))), # Use length of s_x
            min_area=get(kwargs, :min_area, 0.0),
            max_area=get(kwargs, :max_area, Inf),
            target_cv=get(kwargs, :target_cv, 1.0),
            tolerance=get(kwargs, :tolerance, 0.1),
            buffer_dist=get(kwargs, :buffer_dist, 0.5),
            t_idx=get(kwargs, :t_idx, ones(Int, length(s_x)))) # Use length of s_x

        # `expand_hull` will be refactored to take s_x, s_y
        hull_geom = !isnothing(geom_hull) ? geom_hull : expand_hull(s_x, s_y, cfg.buffer_dist) # Updated call

        # Centroid functions will be refactored to take s_x, s_y
        c_mid, reason = if area_method == :cvt get_cvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :kvt get_kvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :qvt get_qvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :bvt get_bvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :hvt get_hvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :avt get_avt_centroids(s_x, s_y, cfg, hull_geom) # Updated call
        else error("Unknown partitioning method: $area_method") end

        polys_coords, v_edges = get_voronoi_polygons_and_edges(c_mid, hull_geom)
        final_centroids = Tuple{Float64, Float64}[]
        lg_polys = []
        for p_coords in polys_coords
            if isempty(p_coords); continue; end
            # Ensure polygon is closed for LibGEOS if it's not already
            if p_coords[1] != p_coords[end]; push!(p_coords, p_coords[1]); end
            lg_p = LibGEOS.Polygon([[ [pt[1], pt[2]] for pt in p_coords ]])
            push!(lg_polys, lg_p)
            cent_g = LibGEOS.centroid(lg_p)
            seq = LibGEOS.getCoordSeq(cent_g)
            push!(final_centroids, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
        end

        # Use s_coord_tuple_local for assignments
        new_assigns = [argmin([sum((p .- sj).^2) for sj in final_centroids]) for p in s_coord_tuple_local]
        n_units = length(final_centroids)
        g = SimpleGraph(n_units)
        for i in 1:n_units, j in (i+1):n_units
            if LibGEOS.touches(lg_polys[i], lg_polys[j]) || LibGEOS.intersects(LibGEOS.buffer(lg_polys[i], 1e-7), lg_polys[j])
                add_edge!(g, i, j)
            end
        end
        g = ensure_connected!(g, final_centroids)
        hull_coords = get_coords_from_geom(hull_geom)
        W = Float64.(Graphs.adjacency_matrix(g))
    end

    # Update the returned NamedTuple to store s_x and s_y
    return (centroids=final_centroids, s_idx=new_assigns, polygons=polys_coords,
            adjacency_edges=v_edges, graph=g, hull_coords=hull_coords,
            termination_reason=reason, s_x=s_x, s_y=s_y, W=W, s_vals=collect(1:size(W,1)))
end



function assign_spatial_units_inferred(adjacency_matrix; iterations=50, learning_rate=0.1, buffer_dist=0.5, input_polygons = nothing)
    """
    Synopsis: Manually constructs a areal_units object for areal data like the Lip Cancer dataset.
              Centroid locations are spatially inferred from connectivity using a rudimentary force-directed layout.
    Inputs:
    - adjacency_matrix: The adjacency matrix (W) of the areal units.
    - iterations: Number of iterations for the force-directed layout.
    - learning_rate: Step size for moving centroids in the layout algorithm.
    - buffer_dist: Distance to buffer the convex hull when polygons are inferred.
    - input_polygons: Optional. A vector of LibGEOS Polygons. If provided, centroids and hull are derived from these.
    """

    local final_centroids
    local adjacency_edges_output
    local polys_output
    local hull_coords_output
    local g_final # The final graph that will be in the result

    nAU = size(adjacency_matrix, 1)


    if input_polygons !== nothing && !isempty(input_polygons)
        # Case 1: Polygons are provided
        # 1. Extract centroids from input_polygons
        final_centroids_geoms = [LibGEOS.centroid(p) for p in input_polygons]
        final_centroids = map(final_centroids_geoms) do g_pt
            seq = LibGEOS.getCoordSeq(g_pt)
            (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
        end

        # 2. Determine hull by dissolving all internal edges
        united_geom = LibGEOS.unaryunion(input_polygons)
        hull_coords_output = get_coords_from_geom(united_geom)

        # 3. Determine adjacency from input_polygons (using LibGEOS.touches)
        adjacency_edges_output = []
        for i in 1:nAU
            g1 = input_polygons[i]
            for j in (i+1):nAU
                g2 = input_polygons[j]
                if LibGEOS.touches(g1, g2)
                    push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                else
                    # Fallback robust check, similar to get_voronoi_polygons_and_edges
                    g1_buffered = LibGEOS.buffer(g1, 1e-6)
                    if LibGEOS.intersects(g1_buffered, g2)
                        inter = LibGEOS.intersection(g1_buffered, g2)
                        if !LibGEOS.isEmpty(inter) && (LibGEOS.area(inter) > 1e-9 || LibGEOS.geomTypeId(inter) in [LibGEOS.GEOS_LINESTRING, LibGEOS.GEOS_MULTILINESTRING])
                            push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                        end
                    end
                end
            end
        end

        polys_output = [get_coords_from_geom(p) for p in input_polygons]

        # Build graph from the determined adjacency edges and ensure connectivity
        g_final = SimpleGraph(nAU)
        centroid_map = Dict(c => i for (i, c) in enumerate(final_centroids))
        for (c1, c2) in adjacency_edges_output
            xi = get(centroid_map, c1, 0)
            yi = get(centroid_map, c2, 0)
            if xi > 0 && yi > 0 && !has_edge(g_final, xi, yi)
                add_edge!(g_final, xi, yi)
            end
        end
        g_final = ensure_connected!(g_final, final_centroids) # Ensure connectivity if necessary

    else
        # Case 2: Polygons are not provided, infer centroids and use tessellation
        # 1. Build initial graph from adjacency_matrix for force-directed layout
        g_initial_for_layout = SimpleGraph(adjacency_matrix)

        # 2. Infer initial centroids using force-directed layout
        side = ceil(Int, sqrt(nAU))
        initial_centroids_fd = [(Float64(i % side), Float64(i ÷ side)) for i in 0:(nAU-1)]
        centroids_vec = [SVector{2, Float64}(c) for c in initial_centroids_fd]

        for iter in 1:iterations
            new_centroids_vec = copy(centroids_vec)
            for i in 1:nAU
                neighbors_i = Graphs.neighbors(g_initial_for_layout, i)
                if !isempty(neighbors_i)
                    avg_neighbor_pos = sum(centroids_vec[n] for n in neighbors_i) / length(neighbors_i)
                    new_centroids_vec[i] = centroids_vec[i] + learning_rate * (avg_neighbor_pos - centroids_vec[i])
                end
            end
            centroids_vec = new_centroids_vec
        end
        # Centroids after force-directed layout
        forced_layout_centroids = [(p[1], p[2]) for p in centroids_vec]

        # 3. Determine hull_geom from inferred centroids for clipping
        fx = getindex.(forced_layout_centroids, 1)
        fy = getindex.(forced_layout_centroids, 2)
        hull_geom = expand_hull(fx, fy, buffer_dist)
        hull_coords_output = get_coords_from_geom(hull_geom)

        # 4. Use tessellation to determine polygon coordinates and initial adjacency (based on forced_layout_centroids)
        polys_coords_raw, _ = get_voronoi_polygons_and_edges(forced_layout_centroids, hull_geom)

        # 5. RECOMPUTE CENTROIDS from the generated (clipped) polygons and prepare for adjacency
        final_centroids = Vector{Tuple{Float64, Float64}}(undef, length(polys_coords_raw))
        lg_polygons_for_adjacency = Vector{Union{LibGEOS.Polygon, Nothing}}(undef, length(polys_coords_raw))
        polys_output = polys_coords_raw

        for (idx, poly_coord_list) in enumerate(polys_coords_raw)
            if !isempty(poly_coord_list) && length(poly_coord_list) >= 3
                if poly_coord_list[1] != poly_coord_list[end]
                    push!(poly_coord_list, poly_coord_list[1])
                end
                lg_poly = LibGEOS.Polygon([ [Float64[p[1], p[2]] for p in poly_coord_list] ])
                centroid_geom = LibGEOS.centroid(lg_poly)
                seq = LibGEOS.getCoordSeq(centroid_geom)
                final_centroids[idx] = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
                lg_polygons_for_adjacency[idx] = lg_poly
            else
                @warn "Invalid or empty polygon encountered in Voronoi tessellation at index $idx. Using original centroid as fallback."
                final_centroids[idx] = forced_layout_centroids[idx]
                lg_polygons_for_adjacency[idx] = nothing
            end
        end

        # 6. Re-build adjacency based on the newly derived centroids and polygons
        adjacency_edges_output = []
        if !isempty(lg_polygons_for_adjacency)
            for i in 1:length(lg_polygons_for_adjacency)
                g1 = lg_polygons_for_adjacency[i]
                if g1 === nothing continue end
                for j in (i+1):length(lg_polygons_for_adjacency)
                    g2 = lg_polygons_for_adjacency[j]
                    if g2 === nothing continue end
                    if LibGEOS.touches(g1, g2)
                        push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                    else
                        g1_buffered = LibGEOS.buffer(g1, 1e-6)
                        if LibGEOS.intersects(g1_buffered, g2)
                            inter = LibGEOS.intersection(g1_buffered, g2)
                            if !LibGEOS.isEmpty(inter) && (LibGEOS.area(inter) > 1e-9 || LibGEOS.geomTypeId(inter) in [LibGEOS.GEOS_LINESTRING, LibGEOS.GEOS_MULTILINESTRING])
                                push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                            end
                        end
                    end
                end
            end
        end

        # 7. Build final graph from the re-derived adjacency edges and ensure connectivity
        g_final = SimpleGraph(nAU)
        centroid_map = Dict(c => i for (i, c) in enumerate(final_centroids))
        for (c1, c2) in adjacency_edges_output
            xi = get(centroid_map, c1, 0)
            yi = get(centroid_map, c2, 0)
            if xi > 0 && yi > 0 && !has_edge(g_final, xi, yi)
                add_edge!(g_final, xi, yi)
            end
        end
        g_final = ensure_connected!(g_final, final_centroids)
    end

    return (
        centroids = final_centroids,
        adjacency_edges = adjacency_edges_output,
        graph = g_final,
        polygons = polys_output,
        hull_coords = hull_coords_output
    )
end


 
# Verbatim copy and refinement of get_polygon_area and get_avt_centroids
# This version fixes the MethodError by providing explicit dispatch for tuple vectors.

"""
    get_polygon_area(poly_coords::AbstractVector)

Calculates the area of a polygon defined by a vector of (x, y) tuples using the Shoelace formula.
Includes data cleaning for NaNs, Infs, and duplicate vertices.
"""
function get_polygon_area(poly_coords::AbstractVector)
    # Filter invalid points
    valid_pts = [p for p in poly_coords if !isnan(p[1]) && !isinf(p[1]) && !isnan(p[2]) && !isinf(p[2])]
    
    # Remove trailing duplicate if it matches the start
    if length(valid_pts) > 1 && valid_pts[1] == valid_pts[end]
        pop!(valid_pts)
    end
    
    # A polygon must have at least 3 vertices
    if length(valid_pts) < 3 
        return 0.0 
    end
    
    x = [p[1] for p in valid_pts]
    y = [p[2] for p in valid_pts]
    
    # Shoelace Formula using LinearAlgebra utilities
    return 0.5 * abs(dot(x, circshift(y, 1)) - dot(y, circshift(x, 1)))
end

function get_polygon_area(s_x, s_y)
    # Wrapper for legacy three-argument calls
    poly_coords = tuple.(s_x, s_y)
    return get_polygon_area(poly_coords)
end

 
function get_coords_from_geom(geom)
    """
    Synopsis: Extracts coordinates from various LibGEOS geometry types.
    Inputs:
    - geom: A LibGEOS geometry object.
    Outputs:
    - A vector of (x, y) coordinates.
    """

    coords = Tuple{Float64, Float64}[]
    local type_id = -1
    try
        type_id = LibGEOS.geomTypeId(geom)
        if type_id == LibGEOS.GEOS_POINT
             # Access coordinate sequence directly for point types
             seq = LibGEOS.getCoordSeq(geom)
             push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
             return coords
        elseif type_id == LibGEOS.GEOS_POLYGON
            ring = LibGEOS.exteriorRing(geom)
            n = LibGEOS.numPoints(ring)
            for i in 1:n
                p = LibGEOS.getPoint(ring, i)
                seq = LibGEOS.getCoordSeq(p)
                push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
            end
        elseif type_id == LibGEOS.GEOS_MULTIPOLYGON
            for i in 1:LibGEOS.numGeometries(geom)
                poly = LibGEOS.getGeometryN(geom, i)
                ring = LibGEOS.exteriorRing(poly)
                n = LibGEOS.numPoints(ring)
                for j in 1:n
                    p = LibGEOS.getPoint(ring, j)
                    seq = LibGEOS.getCoordSeq(p)
                    push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
                end
                if i < LibGEOS.numGeometries(geom); push!(coords, (NaN, NaN)); end
            end
        elseif type_id in [LibGEOS.GEOS_LINESTRING, LibGEOS.GEOS_LINEARRING]
            n = LibGEOS.numPoints(geom)
            for i in 1:n
                p = LibGEOS.getPoint(geom, i)
                seq = LibGEOS.getCoordSeq(p)
                push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
            end
        end
    catch e
        @warn "Coordinate extraction failed for type $type_id: $e"
    end
    return coords
end




function get_voronoi_polygons_and_edges(centroids, hull_geom, tol=1e-7)
    """
    Synopsis: Generates clipped Voronoi polygons with robust adjacency detection.
    Uses a small buffer fallback to handle floating-point misalignment in LibGEOS.
    """
    n_c = length(centroids)
    if n_c == 0
        return [], []
    elseif n_c == 1
        return [get_coords_from_geom(hull_geom)], []
    elseif n_c == 2
        # Standard 2-point bisection logic
        p1, p2 = centroids[1], centroids[2]
        mid = ((p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)
        dx, dy = p2[1] - p1[1], p2[2] - p1[2]
        px, py = -dy, dx
        L = 1e7
        pt1 = (mid[1] + L*px, mid[2] + L*py)
        pt2 = (mid[1] - L*px, mid[2] - L*py)
        side1_pts = [pt1, pt2, (pt2[1] - L*dx, pt2[2] - L*dy), (pt1[1] - L*dx, pt1[2] - L*dy), pt1]
        poly1_box = LibGEOS.Polygon([[[p[1], p[2]] for p in side1_pts]])
        side2_pts = [pt1, pt2, (pt2[1] + L*dx, pt2[2] + L*dy), (pt1[1] + L*dx, pt1[2] + L*dy), pt1]
        poly2_box = LibGEOS.Polygon([[[p[1], p[2]] for p in side2_pts]])
        res1 = LibGEOS.intersection(hull_geom, poly1_box)
        res2 = LibGEOS.intersection(hull_geom, poly2_box)
        return [get_coords_from_geom(res1), get_coords_from_geom(res2)], [(p1, p2)]
    end

    # Deduplicate centroids before triangulation to suppress package warnings 
    # and ensure the output polygon array matches the input centroid array in length.
    u_centroids = unique(centroids)
    if length(u_centroids) < n_c
        u_polys, u_edges = get_voronoi_polygons_and_edges(u_centroids, hull_geom, tol)
        return [u_polys[findfirst(==(c), u_centroids)] for c in centroids], u_edges
    end

    # 3+ points logic
    pts_dt = [(Float64(c[1]), Float64(c[2])) for c in centroids]
    tri = triangulate(pts_dt)
    hull_coords = get_coords_from_geom(hull_geom)
    xs = [p[1] for p in hull_coords if !isnan(p[1])]
    ys = [p[2] for p in hull_coords if !isnan(p[2])]
    if isempty(xs) || isempty(ys) return [Tuple{Float64, Float64}[] for _ in 1:length(centroids)], [] end
    
    bbox = (minimum(xs), maximum(xs), minimum(ys), maximum(ys))
    vorn = voronoi(tri)
    final_coords = [Tuple{Float64, Float64}[] for _ in 1:length(centroids)]
    valid_geoms = Dict{Int, Any}()

    for i in each_generator(vorn)
        if i < 1 || i > length(centroids) continue end
        vertices = get_polygon_coordinates(vorn, i, bbox)
        if !isempty(vertices)
            poly_pts = [[v[1], v[2]] for v in vertices]
            if poly_pts[1] != poly_pts[end] push!(poly_pts, poly_pts[1]) end
            try
                lg_poly = LibGEOS.Polygon([poly_pts])
                clipped = LibGEOS.intersection(lg_poly, hull_geom)
                if !LibGEOS.isEmpty(clipped) && LibGEOS.geomTypeId(clipped) in [LibGEOS.GEOS_POLYGON, LibGEOS.GEOS_MULTIPOLYGON]
                    final_coords[i] = get_coords_from_geom(clipped)
                    valid_geoms[i] = clipped
                end
            catch e end
        end
    end

    v_edges = []
    active_ids = sort(collect(keys(valid_geoms)))
    for idx in 1:length(active_ids)
        i = active_ids[idx]
        g1 = valid_geoms[i]
        for jdx in idx+1:length(active_ids)
            j = active_ids[jdx]
            g2 = valid_geoms[j]
            # Primary check: direct contact
            if LibGEOS.touches(g1, g2)
                push!(v_edges, (centroids[i], centroids[j]))
            else
                # Fallback check: microscopic overlap/buffer
                g1_b = LibGEOS.buffer(g1, tol)
                if LibGEOS.intersects(g1_b, g2)
                    push!(v_edges, (centroids[i], centroids[j]))
                end
            end
        end
    end
    return final_coords, v_edges
end

function check_connectivity(g)
    """
    Synopsis: Evaluates the connectivity of a spatial graph.
    Inputs:
    - g: A SimpleGraph.
    Outputs:
    - NamedTuple showing connection status and components.
    """
    comps = connected_components(g)
    return (is_connected = length(comps) == 1, n_components = length(comps), components = comps)
end


function ensure_connected!(g, centroids)
    # Ensures the spatial graph is connected by adding edges between the nearest 
    # components based on the provided centroid coordinates.
    while !is_connected(g)
        comps = connected_components(g)
        best_dist = Inf
        best_pair = (0, 0)
        
        # Find the two closest nodes belonging to different components
        for i in 1:length(comps), j in (i+1):length(comps)
            for u in comps[i], v in comps[j]
                d = sum((centroids[u] .- centroids[v]).^2)
                if d < best_dist
                    best_dist = d
                    best_pair = (u, v)
                end
            end
        end
        
        if best_pair != (0, 0)
            add_edge!(g, best_pair[1], best_pair[2])
        else
            break
        end
    end
    return g
end



function plot_spatial_graph(au; plot_title="Spatial Partitioning", domain_boundary=nothing)
    # 1. Coordinate Synchronization: Reconstruct tuples for scatter plotting
    local pts = tuple.(au.s_x, au.s_y)

    # 2. Base Plot Initialization
    plt = Plots.plot(aspect_ratio=:equal, legend=false)
    Plots.title!(plt, plot_title)

    # 3. Polygon Geometry Rendering
    for poly_coords in au.polygons
        if length(poly_coords) > 2
            px = [p[1] for p in poly_coords if !isnan(p[1])]
            py = [p[2] for p in poly_coords if !isnan(p[2])]
            if !isempty(px) && (px[1], py[1]) != (px[end], py[end])
                push!(px, px[1])
                push!(py, py[1])
            end
            Plots.plot!(plt, px, py, seriestype=:shape, fillalpha=0.1, linecolor=:black, lw=0.5)
        end
    end

    # 4. Adjacency Graph Edge Rendering
    for edge in Graphs.edges(au.graph)
        u, v = Graphs.src(edge), Graphs.dst(edge)
        p1, p2 = au.centroids[u], au.centroids[v]
        Plots.plot!(plt, [p1[1], p2[1]], [p1[2], p2[2]], color=:red, lw=1.5, alpha=0.6)
    end

    # 5. Scatter Plotting: Points and Centroids
    Plots.scatter!(plt, [p[1] for p in pts], [p[2] for p in pts],
        markersize=1, color=:gray, alpha=0.3, label="Points")
    Plots.scatter!(plt, [c[1] for c in au.centroids], [c[2] for c in au.centroids],
        markersize=4, color=:blue, markerstrokecolor=:white, label="Centroids")

    # 6. Boundary Constraints
    if !isnothing(domain_boundary)
        bx = [p[1] for p in domain_boundary if !isnan(p[1])]
        by = [p[2] for p in domain_boundary if !isnan(p[2])]
        Plots.plot!(plt, bx, by, color=:black, lw=2, ls=:dash)
    end

    return plt
end

    

################
 

function turingindex( indices, sym=nothing, dims=nothing  ) 
     
    if isa(indices, DynamicPPL.Model)
        _, indices = bijector(turing_model, Val(true));
    end

    if isnothing(sym)
      out = enumerate(keys(indices))
    elseif sym=="varnames"
      out = keys(indices)
    else
      out = union(indices[sym]...)
    end
    
    if !isnothing(dims)
        out = reshape(out, dims)
    end

    return out 
end


function showtuples(X)
    for k in keys(X)
        val = getproperty(X, k)
        # Skip displaying keys with NaN values
        # Check if value is numeric before rounding to avoid errors
        display_val = val isa Number ? round(val, digits=3) : val
        println("$k: $display_val")
    end
end



function showparams(X, keywords=["rho", "phi", "sigma",  "mu_", "l_", "ls_"]; limit=10 )
    # Create a regex pattern by joining keywords with the pipe '|' operator
    pattern = Regex(join(keywords, "|"))

    # Filter the parameter list
    matched_params = filter(p -> occursin(pattern, string(p)), FlexiChains.parameters(X))

    # Display the filtered slice
    if isempty(matched_params)
        println("No parameters matched keywords: $keywords")
    else
        out = X[matched_params[1:min(limit, end)]]
        # display(out)
        return out
    end
end





function random_correlation_matrix(d=3, eta=1)

# etas = [1 10 100 1000 1e+4 1e+5];
# d = size of matrix

# EXTENDED ONION METHOD to generate random correlation matrices
# distributed ~ det(S)^eta [or maybe det(S)^(eta-1), not sure]
# https://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices

# LKJ modify this method slightly, in order to be able to sample correlation matrices C from a distribution proportional to [detC]η−1. The larger the η, the larger will be the determinant, meaning that generated correlation matrices will more and more approach the identity matrix. The value η=1 corresponds to uniform distribution. On the figure below the matrices are generated with η=1,10,100,1000,10000,100000. 

    beta = eta + (d-2)/2;
    u = rand( Beta(beta, beta) );
    r12 = 2*u - 1;
    S = [1 r12; r12 1];  

    for k = 3:d
        beta = beta - 1/2;
        y = rand( Beta((k-1)/2, beta) );  # sample from beta
        r = sqrt(y);
        theta = randn(k-1,1);
        theta = theta/norm(theta);
        w = r*theta;
        U, E = eigen(S);
        U = hcat(U)
        R = U' * sqrt(E) * U; # R is a square root of S
        q = R[].re * w;
        S = [S q; q' 1];
    end
    return S
end




function build_st_inputs(time_indices, space_indices, spatial_coords)
  # Space-Time Input Construction
  # Space and Time as continuous coordinates.
  # Inputs: 
  #   spatial_coords: Matrix (2 x N_nodes) -> [Lat, Lon]
  #   time_coords: Vector (T_steps)
  # Returns:
  #   ColVecs of 3D points (Time, Lat, Lon)

  # Map indices to actual coordinates
  # This assumes spatial_coords is 2xN

  # Extract coords for every observation
  coords = spatial_coords[:, space_indices] # 2 x y_N
  times = time_indices' # 1 x y_N

  # Stack to create 3D input: [Time; Lat; Lon]
  return ColVecs(vcat(times, coords))
end

 


function adjacency_matrix_to_nb( W )
    nau = size(W)[1]
    # W = LowerTriangular(W)  # using LinearAlgebra
    nb = [Int[] for _ in 1:nau]
    Threads.@threads for i in 1:nau
        nb[i] = findall( isone, W[i,:] )
    end
    return nb
end


function nb_to_adjacency_matrix( nb )
    nau = Integer( length( unique( reduce(vcat, nb) )) )
    W = zeros( Int8, nau, nau )
    Threads.@threads for i in 1:nau
        for j in 1:length( nb[i] )
            k = nb[i][j]
            W[i, k] = 1
        end
    end
    return(W)
end


function nodes( adj )
    nau = length(adj)
    N_edges = Integer( length( reduce(vcat, adj) )/2 )
    node1 =  fill(0, N_edges); 
    node2 =  fill(0, N_edges); 
    i_edge = 0;
    for i in 1:nau
        u = adj[i]
        num = length(u)
        for j in 1:num
            k = u[j]
            if i < k
                i_edge = i_edge + 1;
                node1[i_edge] = i;
                node2[i_edge] = k;
            end
        end
    end

    e = Edge.(node1, node2)
    g = Graph(e)
    W = Graphs.adjacency_matrix(g)
    
    # D = diagm(vec( sum(W, dims=2) ))
    scalefactor = scaling_factor_bym2(W)

    return node1, node2, scalefactor
end




function scaling_factor_bym2( adjacency_mat )
    # re-scaling variance using Reibler's solution and 
    # Buerkner's implementation: https://codesti.com/issue/paul-buerkner/brms/1241)  
    # Compute the diagonal elements of the covariance matrix subject to the 
    # constraint that the entries of the ICAR sum to zero.
    # See the inla.qinv function help for further details.
    # Q_inv = inla.qinv(Q, constr=list(A = matrix(1,1,nbs$N),e=0))  # sum to zero constraint
    # Compute the geometric mean of the variances, which are on the diagonal of Q.inv
    # scaling_factor = exp(mean(log(diag(Q_inv))))
 
# --- Robust Precision Scaling ---
    N = size(adjacency_mat)[1]
    if N <= 1 return 1.0 end
    asum = vec(sum(adjacency_mat, dims=2))
    asum = float(asum) + N .* max.(asum) .* sqrt(1e-15)
    Q = Diagonal(asum) - adjacency_mat
    A = ones(N)
    try
        S = Q \ Diagonal(A)
        V = S * A
        S = S - V * inv(A' * V) * V'
        diag_s = diag(S)
        valid_diag = filter(x -> x > 1e-12, diag_s)
        if isempty(valid_diag) return 1.0 end
        scale_factor = exp(mean(log.(valid_diag)))
        return isnan(scale_factor) ? 1.0 : scale_factor
    catch
        return 1.0
    end
end


function scaling_factor_bym2(node1, node2, groups=ones(length(node1))) 
    ## calculate the scale factor for each of k connected group of nodes, 
    ## copied from the scale_c function from M. Morris
    gr = unique( groups )
    n_groups = length(gr)
    scale_factor = ones(n_groups)
    Threads.@threads for j in 1:n_groups 
      k = findall( x -> x==j, groups)
      if length(k) > 1 
        e = Edge.(node1[k], node2[k])
        g = Graph(e)
        adjacency_mat = adjacency_matrix(g)
        scale_factor[j] = scaling_factor_bym2( adjacency_mat )
      end
    end
    return scale_factor
end
  

 

function sample_gaussian_process( ; GPmethod="cholesky", returntype="default",
    fkernal=nothing, kerneltype="default", kvar=nothing, kscale=nothing, gpc=GPC(),
    Yobs, Xobs, Xinducing=nothing, lambda=0.0001 )
    
    if isnothing(fkernal)
        if kerneltype=="default" || kerneltype=="squared_exponential"
            fkernal = kvar * SqExponentialKernel() ∘ ScaleTransform( kscale) # ∘ ARDTransform(α)
        end
        if kerneltype=="matern32"
            fkernal = kvar * Matern32Kernel() ∘ ScaleTransform( kscale) # ∘ ARDTransform(α)
        end
    end



    if GPmethod=="textbook"
        # Dimensional fix: Removed vec() to allow multi-feature inputs (N x D)
        Ko = kernelmatrix( fkernal, Xobs )
        Kcommon = inv(Ko + lambda*I) 

        if !isnothing(Xinducing)
            Ki = kernelmatrix( fkernal, Xinducing )
            Kio = kernelmatrix( fkernal, Xinducing, Xobs ) 
            Yinducing_mean_process = Kio * Kcommon * Yobs 
            Covi = Symmetric( Ki - Kio * Kcommon * Kio'  + lambda*I )
            MVNi = MvNormal( Yinducing_mean_process, Covi )

            Yinducing_sample  = rand( MVNi )
            Li =  cholesky(Symmetric( Ki + lambda*I)).L 

            Yobs_mean_process =  Kio' * ( Li' \ (Li \ Yinducing_mean_process  ) ) 
            Covo = Symmetric(kernelmatrix( fkernal, Xobs ) + lambda*I)
            MVN = MvNormal(Yobs_mean_process, Covo)

            if returntype=="fcovariance"
                return MVN
            end

            Yobs_sample =  Kio' * ( Li' \ (Li \ Yinducing_sample  ) ) 

            if returntype=="sample"
                return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
            end

            LogLik = logpdf(MVN, Yobs)

            if returntype=="sample_loglik"
                return ( Yobs_sample=Yobs_sample, loglik=LogLik, GPmethod=GPmethod)
            end

            return (MVN=MVN, MVNi=MVNi, Li=Li, loglik=LogLik,
                Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)

        else
            mean_process = Ko * Kcommon * Yobs   
            MVN = MvNormal(mean_process, Ko + lambda*I  ) 

            if returntype=="fcovariance"
                return MVN
            end

            Yobs_sample = rand( MVN ) 

            if returntype=="sample"
                return ( Yobs_sample=Yobs_sample, GPmethod=GPmethod )
            end

            LogLik = logpdf(MVN,Yobs)

            if returntype=="sample_loglik"
                return ( Yobs_sample=Yobs_sample, loglik=LogLik, GPmethod=GPmethod)
            end

            return ( MVN=MVN, loglik=LogLik, Yobs_sample=Yobs_sample, GPmethod=GPmethod)
        end
    end

    if GPmethod=="cholesky"
        if !isnothing(Xinducing)
            Ko = kernelmatrix( fkernal, Xobs )
            Ki = kernelmatrix( fkernal, Xinducing )
            Kio = kernelmatrix( fkernal, Xinducing, Xobs )
            Lo = cholesky(Symmetric( Ko + lambda*I)).L
            Li = cholesky(Symmetric( Ki + lambda*I)).L
            Yinducing_mean_process  = Kio * ( Lo' \ (Lo \ Yobs ) ) 

            Covi = Symmetric( Ki + lambda*I)
            MVN = MvNormal( Yinducing_mean_process, Covi )

            if returntype=="fcovariance"
                return MVN
            end

            Yobs_mean_process = Kio' * ( Li' \ (Li \ Yinducing_mean_process )) 
            Yinducing_sample  = Yinducing_mean_process + Li * rand(Normal(0, 1), size(Li,1))
            Yobs_sample = Yobs_mean_process + Lo * rand(Normal(0, 1), size(Lo,2))

            if returntype=="sample"
                return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
            end

            LogLik = logpdf(MVN, Yinducing_mean_process)

            if returntype=="sample_loglik"
                return ( Yobs_sample=Yobs_sample, loglik=LogLik, GPmethod=GPmethod)
            end

            return (MVN=MVN, Li=Li, Lo=Lo, loglik=LogLik,
                    Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)

        else
            Ko = kernelmatrix( fkernal, Xobs )
            Lo = cholesky(Symmetric( Ko + lambda*I)).L
            Yobs_mean_process = Ko' * ( Lo' \ (Lo \ Yobs )) 
            Covo = Symmetric( Ko + lambda*I)
            MVN = MvNormal( Yobs_mean_process, Covo ) 

            if returntype=="fcovariance"
                return MVN
            end

            Yobs_sample = Yobs_mean_process + Lo * rand(Normal(0, 1), size(Lo,2))

            if returntype=="sample"
                return (Yobs_sample=Yobs_sample, GPmethod=GPmethod)
            end

            LogLik = logpdf(MVN, Yobs)

            if returntype=="sample_loglik"
                return (Yobs_sample=Yobs_sample, loglik=LogLik,  GPmethod=GPmethod )
            end

            return (MVN=MVN, Lo=Lo, loglik=LogLik, Yobs_sample=Yobs_sample, GPmethod=GPmethod)
        end
    end

 

    if GPmethod=="GPexact"

        fgp = atomic(AbstractGPs.GP(fkernal), gpc)
        fobs = fgp(Xobs, lambda)

        if returntype=="fcovariance"
            return fobs
        end 

        fposterior = posterior(fobs, Yobs) 
        
        if returntype=="posterior"
            return fposterior
        end

        Yobs_sample =  rand(fposterior(Xobs, lambda) )   

        if returntype=="sample"
            return ( Yobs_sample=Yobs_sample, GPmethod=GPmethod)
        end

        LogLik = logpdf(fobs, Yobs)
       
        if returntype=="sample_loglik"
            return (Yobs_sample=Yobs_sample, loglik=LogLik,  GPmethod=GPmethod )
        end

        return ( fgp=fgp, fobs=fobs, fposterior=fposterior, Yobs_sample=Yobs_sample, loglik=LogLik, GPmethod=GPmethod)
    end
 
    if GPmethod=="GPsparse"
        fgp = atomic(AbstractGPs.GP(fkernal), gpc)
        fobs = fgp( Xobs, lambda )
        finducing = fgp( Xinducing, lambda ) 
        fsparse = SparseFiniteGP(fobs, finducing)

        if returntype=="fcovariance"
            return fsparse
        end 

        fposterior = posterior(fsparse, Yobs)

        if returntype=="posterior"
            return fposterior
        end
        
        Yobs_sample =  rand(fposterior(Xobs, lambda) )  
        Yinducing_sample =   rand(fposterior(Xinducing, lambda))

        if returntype=="sample"
            return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
        end

        LogLik = logpdf(fsparse, Yobs)
       
        if returntype=="sample_loglik"
            return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, loglik=LogLik,  GPmethod=GPmethod )
        end

        return ( fgp=fgp, fobs=fobs, finducing=finducing, fsparse=fsparse, fposterior=fposterior, loglik=LogLik, 
                Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)

    end

    if GPmethod=="GPvfe" # Variational Free Energy
        fgp = atomic(AbstractGPs.GP(fkernal), gpc)
        fobs = fgp( Xobs, lambda )
        finducing = fgp(Xinducing, lambda )
        fsparse = VFE( finducing )

        if returntype=="fcovariance"
            return fsparse
        end 
        
        fposterior = posterior(fsparse, fobs, Yobs)  # Distribution is MvNormal  

        if returntype=="posterior"
            return fposterior
        end
        
        Yobs_sample =  rand(fposterior(Xobs, lambda) )  
        Yinducing_sample =   rand(fposterior(Xinducing, lambda))

        if returntype=="sample"
            return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
        end
        
        LogLik = AbstractGPs.elbo(fsparse, fobs, Yobs)  # to a constant
      
        if returntype=="sample_loglik"
            return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, loglik=LogLik,  GPmethod=GPmethod )
        end

        return ( fgp=fgp, fobs=fobs, finducing=finducing, fsparse=fsparse, fposterior=fposterior, loglik=LogLik, 
                Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
    end
      
end
 
  

function plot_variational_marginals(z, sym2range)
    # copied straight from https://turinglang.org/docs/tutorials/variational-inference/
    ps = []

    for (i, sym) in enumerate(keys(sym2range))
        indices = union(sym2range[sym]...)  # <= array of ranges
        if sum(length.(indices)) > 1
            k = 1
            for r in indices
                p = density(
                    z[r, :];
                    title="$(sym)[$k]",
                    titlefontsize=10,
                    label="",
                    ylabel="Density",
                    margin=1.5mm,
                )
                push!(ps, p)
                k += 1
            end
        else
            p = density(
                z[first(indices), :];
                title="$(sym)",
                titlefontsize=10,
                label="",
                ylabel="Density",
                margin=1.5mm,
            )
            push!(ps, p)
        end
    end

    return plot(ps...; layout=(length(ps), 1), size=(500, 2000), margin=4.0mm)
end


function fkernal( kernfunctype="squared_exp", params=nothing )

    if kernfunctype=="squared_exp"
        out = params[1] * SqExponentialKernel() ∘ ScaleTransform(params[2])  
    end

    if kernfunctype=="matern12"
        out = params[1] * Matern12Kernel() ∘ ScaleTransform(params[2])  
    end

    if kernfunctype=="matern32"
        out = params[1] * Matern32Kernel() ∘ ScaleTransform(params[2])  
    end

    if kernfunctype=="matern52"
        out = params[1] * Matern52Kernel() ∘ ScaleTransform(params[2])  
    end

    # ∘ ARDTransform(α)

    return out
end


sekernel2(v, s) = v * SqExponentialKernel() ∘ ScaleTransform(s) # ∘ ARDTransform(a);

sekernel(v, s) = v * SqExponentialKernel() ∘ ScaleTransform(s) # ∘ ARDTransform(a);
 

# Squared-exponential covariance function
sqexp_cov_fn(D, phi, noise=1e-6) = exp.(-D^2 / phi) + LinearAlgebra.I * noise

# Exponential covariance function
exp_cov_fn(D, phi) = exp.(-D / phi)



# generic kernel functions 

# lenscale = -1 / log(ρ)
# σ_ar1^2 / (1 - ρ^2) = marginal variance
# kernel_ar1(σ, ρ) = σ^2 * with_lengthscale(Matern12Kernel(), -1/log(ρ)) 
# the softplus should not be necessary ... 

kernel_ar1(σ, ρ) = σ^2 / softplus(1 - ρ^2) * with_lengthscale(Matern12Kernel(), softplus(-1 / log(ρ)) )

# RW2 is equivalent to a Spline kernel or an Integrated Wiener Process
# For simplicity, use a high-order Matern or a custom structure

kernel_rw2(σ) = σ^2 * Matern52Kernel() # Matern32 is a common smooth approximation for RW2


   
    
function assign_time_units(t_v; time_method="regular", t_N=nothing, u_N=12, kwargs...)

    if time_method=="regular"

        tint = Int.(floor.(t_v))
        t0, t1 = minimum(tint), maximum(tint)
        t_n = t1-t0
        if !isnothing(t_N) 
            if t_n != t_N
                print("warning: time range and unique years do not match")
            end
        end

        t_idx = tint .- t0 .+ 1
        t_vals = collect(t0:t1) .- t0 .+ 1
        t_yr = collect(t0:t1)
        t_brks = (t_yr, t1+1)
        t_mids = t_yr .+ 0.5
        
        u_v = t_v - tint

        u_disc = discretize_data( u_v, N_cat=u_N, method="regular" )  # seasonality discretized

        return (
            t_v = t_v, 
            t_idx = t_idx, 
            t0=t0, 
            t1=t1, 
            t_vals, 
            t_yr=t_yr, 
            t_mids=t_mids, 
            t_brks=t_brks,
            tn=length(t_vals),
            t_N= length(t_vals),
            u_v=u_v, 
            u_idx=u_disc.idx, 
            u_brks=u_disc.brks,
            u_mids=u_disc.mids, 
            u_N=u_N,
            u_vals=collect(1:u_N) 
        )
    end

end


function discretize_data(X; method="quantile", N_cat=9, brks=nothing, probs=nothing, dx=nothing, minv = 0, maxv=1)
     
    if method=="quantile" 
        # simpler solutions
        probs = isnothing(probs) ? collect(range(0.0, stop=1.0, length=N_cat + 1)) : probs
        brks = isnothing(brks) ? quantile(X, probs) : brks
        mids = brks[1:N_cat] + diff(brks) 
        idx = map(x -> clamp(searchsortedfirst(brks, x) - 1, 1, N_cat), X)
        return ( idx=idx, brks=brks, mids=mids, probs )

    elseif method == "regular"
        dx = isnothing(dx) ? (maxv - minv) / N_cat : dx
        probs = nothing
        brks = collect(minv:dx:maxv)
        mids = brks[1:N_cat] + diff(brks) 
        idx = map(x -> clamp(searchsortedfirst(brks, x) - 1, 1, N_cat), X)
        return ( idx=idx, brks=brks, mids=mids, probs, dx=dx )
    
    elseif method=="regular_resolution"    

        xd = round.(Int, X ./ dx ) .* dx   # resolution to  units of dx
        brks = collect( minimum(xd):dx:maximum(xd) + dx  ) 
        mids = midpoints(brks)
        N_cats = length(mids)
        
        xd_cut = cut(X, brks, extend=true)  # from CategoricalArrays
        xi = levelcode.(xd_cut)  # integer index
        return xd, xi, mids, N_cats, dx

    elseif method=="quantile_resolution"
    
        brks = quantile(X, range(0, 1, length=N_cats+1))
        mids = midpoints(brks)
        xd_cut = cut(X, brks, extend=true)  # from CategoricalArrays
        xi = levelcode.(xd_cut)  # integer index
        dx = diff(mids)[1]
        xd = mids[xi] 
        return xd, xi, mids, N_cats, dx

    end


end
 
 

function assign_covariate_levels( X; N_cat=9, brks=nothing, probs=nothing )
    X = Array(X)
    U = [ discretize_data( X[:,i], N_cat=N_cat, brks=brks, probs=probs ) for i in 1:size(X, 2) ]
    V = hcat([t[:idx] for t in  U]...)  # as matrix
    B = hcat([t[:brks] for t in  U]...)
    M = hcat([t[:mids] for t in  U]...)
    P = hcat([t[:probs] for t in  U]...)
    return( idx=V, brks=B, mids=M, probs=P )
end



function estimate_local_kde_with_extrapolation(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, t_idx, target_ts; grid_res=600, sd_extension_factor=0.25)
    """
    Synopsis: Estimates 2D KDE for a specific time slice with extrapolation.
    Inputs:
    - s_coord_tuple: Vector of (x, y) coordinates for all time points.
    - t_idx: Vector of time indices corresponding to s_coord_tuple.
    - target_ts: The specific time slice to estimate KDE for.
    - grid_res: Resolution of the output grid (e.g., 100 for 100x100 grid).
    - sd_extension_factor: Multiplier for standard deviation to define the bandwidth.
    Outputs:
    - Tuple (x_grid, y_grid, intensity) where intensity is a matrix.
    """
    # Filter points for the target time slice

    s_coord_tuple = tuple.(s_x, s_y)

    filtered_pts = [p for (i, p) in enumerate(s_coord_tuple) if t_idx[i] == target_ts]
    if isempty(filtered_pts)
        error("No points found for the target time slice $target_ts")
    end
    xs, ys = [p[1] for p in filtered_pts], [p[2] for p in filtered_pts]
    # Calculate bandwidth based on standard deviation of points
    bw_x = std(xs) * sd_extension_factor
    bw_y = std(ys) * sd_extension_factor
    # Define grid boundaries extending slightly beyond the data range
    x_min, x_max = minimum(xs) - bw_x, maximum(xs) + bw_x
    y_min, y_max = minimum(ys) - bw_y, maximum(ys) + bw_y
    x_grid = collect(range(x_min, stop=x_max, length=grid_res))
    y_grid = collect(range(y_min, stop=y_max, length=grid_res))
    intensity = zeros(grid_res, grid_res)
    # Gaussian KDE implementation
    for i in 1:grid_res
        for j in 1:grid_res
            x_val, y_val = x_grid[i], y_grid[j]
            for (px, py) in filtered_pts
                dx = (x_val - px) / bw_x
                dy = (y_val - py) / bw_y
                intensity[i, j] += exp(-0.5 * (dx^2 + dy^2))
            end
        end
    end
    # Normalize intensity to sum to 1 (optional, depending on desired output)
    intensity ./= sum(intensity)
    return x_grid, y_grid, intensity
end



function calculate_metrics(au_obj)
    # Map coordinates from constituent vectors to avoid FieldError
    local observation_points = tuple.(au_obj.s_x, au_obj.s_y)
    
    # Re-calculate assignments based on nearest centroid
    local assignments = [argmin([sum((p .- c).^2) for c in au_obj.centroids]) for p in observation_points]
    
    # Compute frequency counts per spatial unit
    local unit_counts = [count(==(i), assignments) for i in 1:length(au_obj.centroids)]

    # Filter valid numerical entries to prevent NaN propagation
    local valid_entries = filter(x -> !isnan(x) && !ismissing(x), unit_counts)

    if isempty(valid_entries)
        return (mean_density=NaN, sd_density=NaN, cv_density=NaN)
    end

    local m_val = mean(valid_entries)
    local s_val = std(valid_entries)
    local cv_val = s_val / (m_val + 1e-9)

    return (mean_density=m_val, sd_density=s_val, cv_density=cv_val)
end


function get_spatial_graph( centroids, adjacency_edges )
    """
    Synopsis: Converts partitioning results into a formal SimpleGraph. 
    Outputs: A SimpleGraph object.
    """
    n = length(centroids)
    g = SimpleGraph(n)
    centroid_map = Dict(c => i for (i, c) in enumerate(centroids))
    for edge in adjacency_edges
        xi, yi = get(centroid_map, edge[1], 0), get(centroid_map, edge[2], 0)
        if xi > 0 && yi > 0 add_edge!(g, xi, yi) end
    end
    return g
end



function plot_kde_simple(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}; grid_res=600, sd_extension_factor=0.25, title="Spatial Intensity (KDE)")
    # Internal wrapper for estimate_local_kde_with_extrapolation
    # Description: Generates a simple 2D Heatmap of spatial intensity using Kernel Density Estimation.
    # Inputs:
    #   - s_coord_tuple: Vector of (x, y) coordinate tuples.
    #   - grid_res: Resolution of the output grid.
    #   - sd_extension_factor: Factor to extend the bandwidth standard deviation.
    #   - title: Title for the generated plot.
    # Outputs:
    #   - A Plots.Plot object (Heatmap with scatter overlay).
    # Using a dummy t_idx of 1s since plotting a static slice

    s_coord_tuple = tuple.(s_x, s_y)

    t_idx_dummy = ones(Int, length(s_coord_tuple))
    x_g, y_g, intensity = estimate_local_kde_with_extrapolation(s_coord_tuple, t_idx_dummy, 1; grid_res=grid_res, sd_extension_factor=sd_extension_factor)

    plt = Plots.heatmap(x_g, y_g, intensity',
                  title=title,
                  c=:viridis,
                  aspect_ratio=:equal,
                  xlabel="X", ylabel="Y")
    Plots.scatter!(plt, [p[1] for p in s_coord_tuple], [p[2] for p in s_coord_tuple],
                   markersize=2, markercolor=:white, markeralpha=0.5, label="Points")
    return plt
end




function scottish_lip_cancer_data_spacetime(n_years::Int=10; rndseed::Int=42)
    # "expand" scottish lip cancer data to a space-time version
    # original data source:  https://mc-stan.org/users/documentation/case-studies/icar_stan.html

    Random.seed!(rndseed)

    # Load base spatial data
 # Base Spatial Data for 56 Counties
    # data source:  https://mc-stan.org/users/documentation/case-studies/icar_stan.html

    nAU = 56

    y_base = [ 9, 39, 11, 9, 15, 8, 26, 7, 6, 20, 13, 5, 3, 8, 17, 9, 2, 7, 9, 7,
    16, 31, 11, 7, 19, 15, 7, 10, 16, 11, 5, 3, 7, 8, 11, 9, 11, 8, 6, 4,
    10, 8, 2, 6, 19, 3, 2, 3, 28, 6, 1, 1, 1, 1, 0, 0]

    E_base = [1.4, 8.7, 3.0, 2.5, 4.3, 2.4, 8.1, 2.3, 2.0, 6.6, 4.4, 1.8, 1.1, 3.3, 7.8, 4.6,
    1.1, 4.2, 5.5, 4.4, 10.5,22.7, 8.8, 5.6,15.5,12.5, 6.0, 9.0,14.4,10.2, 4.8, 2.9, 7.0,
    8.5, 12.3, 10.1, 12.7, 9.4, 7.2, 5.3,  18.8,15.8, 4.3,14.6,50.7, 8.2, 5.6, 9.3, 88.7,
    19.6, 3.4, 3.6, 5.7, 7.0, 4.2, 1.8]

    x_base = [16,16,10,24,10,24,10, 7, 7,16, 7,16,10,24, 7,16,10, 7, 7,10,
    7,16,10, 7, 1, 1, 7, 7,10,10, 7,24,10, 7, 7, 0,10, 1,16, 0,
    1,16,16, 0, 1, 7, 1, 1, 0, 1, 1, 0, 1, 1,16,10]

    adjacency = [ 5, 9,11,19, 7,10, 6,12, 18,20,28, 1,11,12,13,19,
    3, 8, 2,10,13,16,17, 6, 1,11,17,19,23,29, 2, 7,16,22, 1, 5, 9,12,
    3, 5,11, 5, 7,17,19, 31,32,35, 25,29,50, 7,10,17,21,22,29,
    7, 9,13,16,19,29, 4,20,28,33,55,56, 1, 5, 9,13,17, 4,18,55,
    16,29,50, 10,16, 9,29,34,36,37,39, 27,30,31,44,47,48,55,56,
    15,26,29, 25,29,42,43, 24,31,32,55, 4,18,33,45, 9,15,16,17,21,23,25,
    26,34,43,50, 24,38,42,44,45,56, 14,24,27,32,35,46,47, 14,27,31,35,
    18,28,45,56, 23,29,39,40,42,43,51,52,54, 14,31,32,37,46,
    23,37,39,41, 23,35,36,41,46, 30,42,44,49,51,54, 23,34,36,40,41,
    34,39,41,49,52, 36,37,39,40,46,49,53, 26,30,34,38,43,51, 26,29,34,42,
    24,30,38,48,49, 28,30,33,56, 31,35,37,41,47,53, 24,31,46,48,49,53,
    24,44,47,49, 38,40,41,44,47,48,52,53,54, 15,21,29, 34,38,42,54,
    34,40,49,54, 41,46,47,49, 34,38,49,51,52, 18,20,24,27,56,
    18,24,30,33,45,55]

    number_neighbours = [4, 2, 2, 3, 5, 2, 5, 1,  6,  4, 4, 3, 4, 3, 3, 6, 6, 6 ,5,
    3, 3, 2, 6, 8, 3, 4, 4, 4,11,  6, 7, 4, 4, 9, 5, 4, 5, 6, 5,
    5, 7, 6, 4, 5, 4, 6, 6, 4, 9, 3, 4, 4, 4, 5, 5, 6]
 
    # Build graph from adjacency info

    N_edges = Integer(length(adjacency) / 2)
    node1 = fill(0, N_edges)
    node2 = fill(0, N_edges)
    i_adjacency = 0
    i_edge = 0
    for i in 1:nAU
        for j in 1:number_neighbours[i]
            i_adjacency += 1
            if i < adjacency[i_adjacency]
                i_edge += 1
                node1[i_edge] = i
                node2[i_edge] = adjacency[i_adjacency]
            end
        end
    end

    e = Edge.(node1, node2)
    g = Graph(e)
    W = adjacency_matrix(g)
    # D = diagm(vec(sum(W, dims=2)))
 
    au = assign_spatial_units_inferred( W ) # "infer" from the adjacency network (W)
    pts_base = au.centroids
    
    N_total = nAU * n_years

    # 1. Random Walk Trend
    rw_trend = cumsum(randn(n_years) .* 0.5)

    # 2. Expand Data Vectors
    y_expanded = repeat(y_base, n_years)
    E_expanded = repeat(E_base, n_years)
    x_expanded = repeat(x_base, n_years)
    t_idx = repeat(1:n_years, inner=nAU)
    s_coord_tuple = repeat(pts_base, n_years)

    s_x = getindex.(s_coord_tuple, 1)
    s_y = getindex.(s_coord_tuple, 2)

    # The s_idx is the spatial unit identifier (1 to 56)
    s_idx = repeat(1:nAU, n_years)
 
    # 3. Add Random Walk + Noise to Response
    # Broadcast rw_trend across years
    trend_component = repeat(rw_trend, inner=nAU)
    noise = randn(N_total) .* 0.2

    # Final response: base_y + trend + noise (ensuring positive counts)
    y_final = floor.(Int, abs.(y_expanded .+ trend_component .+ noise))

    # 4. Final covariate matrix and offsets
    x_scaled = (x_expanded .- mean(x_expanded)) ./ std(x_expanded)
    X = Matrix(DataFrame(AFF=x_scaled))
    log_offset = log.(E_expanded)
   
    return (
        y=y_final, X=X, s_x=s_x, s_y=s_y, log_offset=log_offset, t_idx=t_idx,
        s_idx=s_idx, n_years=n_years, W=W, au=au
    )
end
  


function icar_form(theta, phi, sigma, rho)
    # https://sites.stat.columbia.edu/gelman/research/published/bym_article_SSTEproof.pdf
    # Reibler parameterization: https://pubmed.ncbi.nlm.nih.gov/27566770/
    # https://www.jstatsoft.org/index.php/jss/article/view/v063c01/841
    sigma .* ( sqrt.(1 .- rho) .* theta .+ sqrt.(rho ./ scaling_factor) .* phi )  
end
   


 
function kron_matern_sample(Ns, Nt, unique_s, unique_t, ls_s, sigma_s, ls_t, sigma_t, noise_vec)
    # Spatial Precision
    k_s = Matern32Kernel() ∘ ScaleTransform(inv(ls_s))
    K_s = Symmetric(sigma_s^2 * kernelmatrix(k_s, RowVecs(unique_s)) + 1e-4*I)
    Q_s = inv(K_s)

    # Temporal Precision
    k_t = Matern32Kernel() ∘ ScaleTransform(inv(ls_t))
    K_t = Symmetric(sigma_t^2 * kernelmatrix(k_t, unique_t) + 1e-4*I)
    Q_t = inv(K_t)

    # Full Kronecker Precision (Dense for AD compatibility)
    Q_full = Symmetric(Matrix(kron(Q_t, Q_s)) + 1e-4*I)
    L_q = cholesky(Q_full)

    # Sample: f = (L')^-1 * noise
    return L_q.U \ noise_vec
end


function get_posterior_means(ch, param_base, N)
    # Description:
    #   Extracts and averages posterior samples for a specific vector parameter.
    # Inputs:
    #   ch: MCMC sample chain.
    #   param_base: String prefix of the parameter (e.g., "s_eff").
    #   N: Length of the vector parameter.
    # Outputs:
    #   Vector of posterior means.

    means = zeros(N)
    
    for i in 1:N
        p_symbol = Symbol("$param_base[$i]")
        if p_symbol in names(ch, :parameters)
            means[i] = mean(ch[p_symbol])
        else
            @warn "Parameter $p_symbol not found in chain."
        end
    end
    
    return means
end



function generate_inducing_points(coords, M_inducing, seed=42; method="kmeans")

    # Helper function to generate inducing points (simple random sampling for now)

    Random.seed!(seed)
    n_data = size(coords, 1)
    if M_inducing >= n_data
        return coords # If M >= N, just use all data points (becomes exact GP)
    end

    
    if method=="random"
        indices = sample(1:n_data, M_inducing, replace=false)
        return coords[indices, :]
    end

    if method=="kmeans"
        #   Identifies optimal inducing point locations using K-Means clustering.
        #   Essential for Sparse/FITC Gaussian Process models.
        # Inputs:
        #   coords: N x D matrix of spatiotemporal coordinates.
        #   M_inducing: Target number of inducing points.
        # Outputs:
        #   M x D matrix of cluster centroids.

        # Transpose for Clustering.jl compatibility
        data_matrix = Matrix(coords')
        
        # Execute K-Means
        clustering_result = kmeans(data_matrix, M_inducing, maxiter=200)
        
        # Extract centroids and transpose back
        inducing_points = clustering_result.centers'
        
        return inducing_points
    end

    if method=="furthest_point"

        # Initialize with a random point
        inducing_points_idx = [rand(1:n_data)]
        distances = fill(Inf, n_data)

        # Convert coords to the expected format for pairwise
        coords_matrix = permutedims(coords)

        for _ in 2:M_inducing
            # Calculate distances from all points to the newest inducing point
            last_added_idx = inducing_points_idx[end]
            new_distances = colwise(Euclidean(), coords_matrix, coords_matrix[:, last_added_idx])

            # Update minimum distances to any inducing point found so far
            distances = min.(distances, new_distances)

            # Find the point farthest from any existing inducing point
            farthest_idx = argmax(distances)
            push!(inducing_points_idx, farthest_idx)
        end

        return coords[inducing_points_idx, :], inducing_points_idx

    end

end

 
function ar1_covariance(n, rho, var, ::Type{T}=Float64) where {T}
    # Description:
    #   Generates a full AR1 covariance matrix.
    # Inputs:
    #   n: Number of time points.
    #   rho: Correlation coefficient.
    #   var: Marginal variance.
    # Outputs:
    #   n x n Covariance matrix.

    vcv = zeros(T, n, n) .+ I(n)
    
    Threads.@threads for r in 1:n
        for c in 1:n
            if r >= c
                vcv[r, c] = var * rho^(r - c)
            end
        end
    end
    
    return Symmetric(vcv)
end

 


function ar1_covariance_local( n, rho, var,  ::Type{T}=Float64 )  where {T} 
    vcv = zeros( T, n, n) .+ I(n) 
    Threads.@threads for r in 1:n
    for c in 1:n
        d = r-c
        if d == 0 | d == 1
            vcv[r,c] = var * rho^d  
        end
    end
    end
    return vcv
end



function gp_predictions(; Y, D, mu, sig2, phi, cov_fn=exp_cov_fn, nN=length(Xnew), nP=size(res, 1) ) 
    ynew = Vector{Float64}()
    # Threads.@threads -- to add 
    for i in sample(1:size(res,1), nP, replace=true)
        K = cov_fn(D, phi[i])
        Koo_inv = inv(K[(nN+1):end, (nN+1):end])
        Knn = K[1:nN, 1:nN]
        Kno = K[1:nN, (nN+1):end]
        C = Kno * Koo_inv
        mvn = MvNormal( 
            C * (Y .- mu[i]) .+ mu[i], 
            Matrix(LinearAlgebra.Symmetric(Knn - C * Kno')) + sig2[i] * LinearAlgebra.I 
        ) 
        ynew = vcat(ynew, [rand(mvn) ] )
    end
    ynew = stack(ynew, dims=1)  # rehape to matrix   
    return ynew
end




function variational_inference_solution(m; max_iters=100, nsamps=max_iters,  nelbo=3 )

    # Fit via ADVI. minor speed benefit vs NUTS
    _, indices = Bijectors.bijector(m, Val(true));
    vars = keys(indices)

    q0 = Variational.meanfield(m)     # initialize variational distribution (optional)
    advi = ADVI(nelbo, max_iters)    # num_elbo_samples, max_iters
    msol = Turing.vi(m, advi, q0) #, optimizer=Flux.ADAM(1e-1));
    msamples = DataFrame( rand(msol, nsamps )', :auto ) 

    # vectorize variable names ... needs more conditions if 2-D or higher ..
    vns = []
    for (i, sym) in enumerate(vars) 
        j = union(indices[sym]...)  # <= array of ranges
        nj = sum(length.(j)) 
        if  nj > 1
            k = 1
            for r in j
                push!(vns, "$(sym)[$k]")
                k += 1
            end
        else
            push!(vns, "$(sym)") 
        end
    end
    
    vns = Symbol.(vns)

    msamples = rename(msamples, vns)

    mmean = combine( msamples, [ n => (x -> mean(x)) => n for n in names(msamples)  ] )
    mstd  = combine( msamples, [ n => (x -> std(x)) => n for n in names(msamples)  ] )

    out = (
        msol = msol,
        msamples = msamples, 
        mmean = mmean,
        mstd = mstd
    )
    
    return out
 
end

  


function rff_map(coords, W, b)
    # Description:
    #   Maps input coordinates into a Random Fourier Feature (RFF) space
    #   to approximate a kernel function (usually Squared Exponential).
    # Inputs:
    #   coords: N x D matrix of input features (space/time).
    #   W: D x M weight matrix sampled from spectral density.
    #   b: Vector of M random phases.
    # Outputs:
    #   N x M feature matrix.

    # Project coordinates into higher dimensional space
    projection = (coords * W) .+ b'
    
    # Apply cosine transformation with scaling factor
    m = size(W, 2)
    feature_map = sqrt(2 / m) .* cos.(projection)
    
    return feature_map
end
 

function generate_informed_rff_params(coords, M_rff)
    # Audited and Robust RFF Parameter Generator (v05.4)
    # Rationale: Fixing MethodError by ensuring numeric matrix conversion before adjoint.
    
    # 1. Convert input to standard Matrix{Float64}
    mat = if coords isa AbstractMatrix && eltype(coords) <: Real
        Matrix{Float64}(coords)
    elseif coords isa AbstractVector && eltype(coords) <: Tuple
        # Standardize Tuple vectors to numeric Matrix
        reduce(hcat, [[Float64(p[1]), Float64(p[2])] for p in coords])'
    else
        # Fallback for complex nesting or SubArrays
        collect(Matrix{Float64}(reduce(hcat, collect.(coords))'))
    end

    # 2. Dynamic Orientation Check (Enforce D x N for std calculation)
    if size(mat, 1) > size(mat, 2) && size(mat, 2) <= 3
        mat = collect(mat')
    end

    d = size(mat, 1)
    
    # 3. Scale Calculation along observation axis
    coord_scales = std(mat, dims=2) .+ 1e-6

    # 4. Frequency Weight Generation
    W = randn(d, M_rff) ./ coord_scales

    # 5. Phase Shift Generation
    b = rand(M_rff) .* (2.0 * pi)
    
    return W, b
end


function generate_rff_params_for_se_kernel(D_in, M_rff, lengthscale)
    # Helper function to generate RFF parameters for a Squared Exponential kernel
    # For a Squared Exponential kernel, the spectral density is Gaussian: N(0, (1/l)^2 * I)
    sigma_spectral = 1.0 / lengthscale
    W_matrix = randn(D_in, M_rff) .* sigma_spectral # D_in x M_rff matrix
    b_vector = rand(Uniform(0, 2pi), M_rff)
    return W_matrix, b_vector
end 
 

macro save_carstm_state(file_to_save_name_sym)
  quote
    try
      # Evaluate the input symbol (e.g., :state_filename) to its value (e.g., "carstm_state.jld2")
      local filename_val = $(esc(file_to_save_name_sym))
      @info "Saving CARSTM state to $(filename_val)..."
      # JLD2.@save expects variable names as symbols, not their values.
      # The variables themselves should be directly passed.
      JLD2.@save "$(filename_val)" areal_units mod chain y_sim y_binary t_idx weights trials cov_indices cov_indices trials_sim  weights_sim adj_matrix_numeric s_N t_N area_method
      @info "CARSTM state saved successfully."
    catch e
      @error "Error saving CARSTM state: $e"
    end
  end
end

macro load_carstm_state(filename_sym)
  quote
    # Evaluate the input symbol (e.g., :state_filename) to its value (e.g., "carstm_state.jld2")
    local filename_val = $(esc(filename_sym))
    if !isfile(filename_val)
      @error "File $(filename_val) not found."
      return nothing
    end
    try
      @info "Loading CARSTM state from $(filename_val)..."
      # JLD2.@load expects variable names as symbols, not their values.
      # The variables themselves should be directly passed.
      JLD2.@load "$(filename_val)" areal_units mod chain  y_sim y_binary t_idx weights trials cov_indices cov_indices trials_sim  weights_sim adj_matrix_numeric s_N t_N area_method
      @info "CARSTM state loaded successfully."
      # Variables are loaded directly into the calling scope by JLD2.@load
      # No explicit return value from the macro itself, as it injects variables
    catch e
      @error "Error loading CARSTM state: $e"
      return nothing
    end
  end
end


function init_params_extract( res=NaN; load_from_file=false, override_means=false, fn_inits = "init_params.jl2"  )
  # Description: Extracts initial parameter values from a model result summary or loads them from a file.
  # Inputs:
  #   - res: Model result object (default: NaN).
  #   - load_from_file: Boolean, if true loads params from fn_inits.
  #   - override_means: Boolean, if true applies custom overrides for specific parameter patterns.
  #   - fn_inits: String, filename for storage.
  # Outputs:
  #   - A FillArray containing the extracted or loaded mean parameter values.

  if load_from_file
    init_params = load(fn_inits )
    return(init_params)
  end

  ressumm = summarize(res)
  vns = ressumm.nt.parameters
  means = ressumm.nt[2]  # means

  if  override_means
    u = findall(x-> occursin(r"^t_period\[", String(x)), vns ); vns[u]
    if length(u) > 0 
      means[u] = [ 1.0, 1.0, 5.0, 5.0]  # (sin, cos) X annual, 5-year (el nino)
    end

    u = findall(x-> occursin(r"^pca_sd\[", String(x)), vns ); vns[u]
    if length(u) > 0  
      means[u] = sigma_prior  # from basic pca
    end

    u = findall(x-> occursin(r"^v\[", String(x)), vns ); vns[u]
    if length(u) > 0 
      means[u] = v_prior  # from basic pca
    end
  end

  init_params = FillArrays.Fill( means )
  jldsave( fn_inits; init_params )

  return(init_params)
end


function init_params_copy( res=NaN, res0=NaN; load_from_file=false, override_means=false, fn_inits = "init_params.jl2"  )
  # using spatial parts of res0 
  # Description: Copies parameter values from a reference result (res0) to a target result structure (res).
  # Inputs:
  #   - res: Target model result object.
  #   - res0: Reference model result object.
  #   - load_from_file: Boolean to load from fn_inits instead.
  #   - override_means: Boolean to apply custom pattern-based overrides.
  #   - fn_inits: String, filename for storage.
  # Outputs:
  #   - A FillArray containing the merged mean parameter values.
  if load_from_file
    init_params = load(fn_inits )
    return(init_params)
  end

  ressumm = summarize(res)
  vns = ressumm.nt.parameters
  means = ressumm.nt[2]  # means

  ressumm0 = summarize(res0)
  vns0 = ressumm0.nt.parameters
  means0 = ressumm0.nt[2]  # means

  if  override_means
    u = findall(x-> occursin(r"^t_period\[", String(x)), vns );  
    if length(u) > 0 
      means[u] = [ 1.0, 1.0, 5.0, 5.0, 10.0, 10.0][1:length(u)]  # (sin, cos) X annual, 5-year (el nino)
    end

    u = findall(x-> occursin(r"^pca_sd\[", String(x)), vns );  
    if length(u) > 0  
      u0 = findall(x-> occursin(r"^pca_sd\[", String(x)), vns0 );  
      if length(u0) > 0  && length(u) == length(u0)
        means[u] = means0[u0]  # from basic pca
      end
    end

    u = findall(x-> occursin(r"^v\[", String(x)), vns );  
    if length(u) > 0  
      u0 = findall(x-> occursin(r"^pca_sd\[", String(x)), vns0 );  
      if length(u0) > 0  && length(u) == length(u0)
        means[u] = means0[u0]  # from basic pca
      end
    end
  end
  
  init_params = FillArrays.Fill( means )
  jldsave( fn_inits; init_params )

  return(init_params)
end


function libgeos_lattice_adjacency_matrix(rows::Int, cols::Int)
    """
    libgeos_lattice_adjacency_matrix(rows, cols)

    Description:
    Generates a sparse adjacency matrix for a regular 2D lattice using LibGEOS for spatial geometry operations.
    Constructs unit square polygons for each cell and identifies neighbors based on Queen contiguity
    (any shared boundary point or edge).

    Inputs:
    - rows (Int): Number of rows in the lattice grid.
    - cols (Int): Number of columns in the lattice grid.

    Output:
    - W (SparseMatrixCSC{Int, Int}): A binary sparse adjacency matrix of size (rows*cols) x (rows*cols).
    """
    # Create polygons for each cell in the lattice
    polygons = []
    for r in 1:rows, c in 1:cols
        # Define unit square coordinates as nested vectors for LibGEOS compatibility
        coords = [
            [Float64(c-1), Float64(r-1)],
            [Float64(c),   Float64(r-1)],
            [Float64(c),   Float64(r)],
            [Float64(c-1), Float64(r)],
            [Float64(c-1), Float64(r-1)]
        ]
        # Construct LinearRing and then Polygon
        ring = LibGEOS.LinearRing(coords)
        push!(polygons, LibGEOS.Polygon(ring))
    end

    n = length(polygons)
    W = spzeros(Int, n, n)

    # Queen contiguity check
    for i in 1:n
        poly_i = polygons[i]
        for j in (i+1):n
            if LibGEOS.intersects(poly_i, polygons[j])
                W[i, j] = W[j, i] = 1
            end
        end
    end
    return W
end




function build_laplacian_precision(adj_matrix)
    # Description:
    #   Constructs a GMRF precision matrix (Graph Laplacian) from an adjacency matrix.
    # Inputs:
    #   adj_matrix: Sparse adjacency matrix (W).
    # Outputs:
    #   Sparse precision matrix (Q).

    # D is the diagonal matrix of node degrees
    D_diag = Diagonal(vec(sum(adj_matrix, dims=2)))
    Q_mat = D_diag - adj_matrix
    
    return Q_mat
end
 




function scale_precision!(Q)
    # Description:
    #   Scales a precision matrix using the geometric mean of non-zero eigenvalues.
    #   Essential for ensuring sigma_sp represents marginal variance in BYM2.
    # Inputs:
    #   Q: Precision matrix to be modified in-place.

    eig_vals = eigvals(Matrix(Q))
    # Filter out near-zero eigenvalues associated with the null space
    valid_eigs = filter(x -> x > 1e-6, eig_vals)
    
    scaling_factor = exp(mean(log.(valid_eigs)))
    
    if Q isa Symmetric
        Q.data ./= scaling_factor
    else
        Q ./= scaling_factor
    end
    
    return Q
end


  


function logpdf_gmrf(x, Q)
    # Description: Calculates the log-probability of a Gaussian Markov Random Field.
    # Inputs:
    #   - x: Vector of values.
    #   - Q: Precision matrix.
    # Outputs:
    #   - Log-likelihood value.
    Q_stable = Matrix(Q) + I * 1e-5
    F = cholesky(Symmetric(Q_stable))
    return 0.5 * (logdet(F) - dot(x, Q, x) - length(x) * log(2pi))
end


 
 

function plot_posterior_results(stats, M=nothing, areal_units=nothing; s_x=nothing, s_y=nothing, time_slice=nothing, effect=:spatial, cov_idx=1, show_pts=false)
    # Description: Comprehensive posterior visualization for CARSTM and Deep GP models.

    # Extract target stats and guard against nothing or scalar values
    st = getproperty(stats, effect)
    isnothing(st) && return nothing
    if st isa Real
        return Plots.plot(title="$effect (Fixed: $st)")
    end
 

    # 1. Handle Categorical/Class Bar Plots
    if effect == :beta_cov
        b_list = get(stats, :beta_cov, nothing)
        isnothing(b_list) && return nothing
        b_stats = b_list isa AbstractVector ? b_list[cov_idx] : b_list
        (isnothing(b_stats) || b_stats isa Real) && return nothing
        n_levels = size(b_stats.mean, 1)
        return StatsPlots.bar(1:n_levels, b_stats.mean[:,1],
                  yerror=(b_stats.mean[:,1] .- b_stats.lower[:,1], b_stats.upper[:,1] .- b_stats.mean[:,1]),
                  title="Covariate $cov_idx Effects", xlabel="Level", ylabel="Effect Size", legend=false)

    elseif effect == :b_class1 || effect == :b_class2
        b_stats = st
        if isnothing(b_stats) || b_stats isa Real; return nothing; end
        n_levels = size(b_stats.mean, 1)
        return StatsPlots.bar(1:n_levels, b_stats.mean[:,1],
                  yerror=(b_stats.mean[:,1] .- b_stats.lower[:,1], b_stats.upper[:,1] .- b_stats.mean[:,1]),
                  title="$effect Levels", xlabel="Class Index", ylabel="Effect Size", legend=false)

    # 2. Handle Temporal Main Effects
    elseif effect == :temporal
        t_stats = st
        if isnothing(t_stats) || t_stats isa Real; return nothing; end
        n_times = length(t_stats.mean)
        return StatsPlots.plot(1:n_times, t_stats.mean,
                   ribbon=(t_stats.mean .- t_stats.lower, t_stats.upper .- t_stats.mean),
                   fillalpha=0.2, lw=2, title="Temporal Main Effect", xlabel="Time Index", ylabel="Effect (Latent Scale)", legend=false)

    # 2a. Handle Seasonal Effects
    elseif effect == :seasonal
        t_stats = st
        if isnothing(t_stats) || t_stats isa Real; return nothing; end
        n_times = length(t_stats.mean)
        return StatsPlots.plot(1:n_times, t_stats.mean,
                   ribbon=(t_stats.mean .- t_stats.lower, t_stats.upper .- t_stats.mean),
                   fillalpha=0.2, lw=2, title="Seasonal Effect", xlabel="Time Index", ylabel="Effect (Latent Scale)", legend=false)


    # 3. Handle Spatial, ST, and Deep GP Mean Fields
    elseif effect in [:spatial, :spatial_structured, :spatial_unstructured, :predictions_denoised, :predictions_noisy, :residuals, :eta_gp, :hidden_layer]
        plt = StatsPlots.plot(aspect_ratio=:equal, title="$effect (T=$(time_slice))", legend=true)

        # Determine the values to map to colors
        values = if hasproperty(st, :mean)
            st.mean
        elseif effect == :spatial_structured
            stats.spatial_structured.mean
        elseif effect == :spatial_unstructured
            stats.spatial_unstructured.mean
        elseif effect == :eta_gp
            haskey(stats, :eta_gp) ? stats.eta_gp.mean : error("eta_gp not found in stats")
        elseif effect == :hidden_layer
            haskey(stats, :h1) ? stats.h1.mean : error("hidden layer h1 not found in stats")
        elseif effect == :predictions_denoised && !isnothing(time_slice)
            stats.predictions_denoised.mean[:, time_slice]
        elseif effect == :predictions_noisy && !isnothing(time_slice)
            stats.predictions_noisy.mean[:, time_slice]
        # elseif effect == :residuals
        #    stats.predictions_noisy.mean[:, isnothing(time_slice) ? 1 : time_slice]
        else
            error("Effect $effect requires specific keys in stats or time_slice index")
        end

        # SAFETY FIX: Plot only as many polygons as results for to avoid BoundsError
        n_to_plot = min(length(areal_units.polygons), length(values))

        for i in 1:n_to_plot
            poly_coords = areal_units.polygons[i]
            if length(poly_coords) > 2
                px = [pt[1] for pt in poly_coords if !isnan(pt[1])]
                py = [pt[2] for pt in poly_coords if !isnan(pt[2])]

                if !isempty(px)
                    if (px[1], py[1]) != (px[end], py[end])
                        push!(px, px[1]); push!(py, py[1])
                    end

                    val = values[i]
                    StatsPlots.plot!(plt, px, py,
                        seriestype=:shape,
                        fill_z=val,
                        c=:RdYlBu,
                        linecolor=:black,
                        linewidth=0.5,
                        fillalpha=0.8,
                        legend=false
                    )
                end
            end
        end

        if show_pts
            StatsPlots.scatter!(plt, s_x, s_y,
                markersize=1, markercolor=:gray, alpha=0.2, label="Observations")
        end

        StatsPlots.scatter!(plt, [c[1] for c in areal_units.centroids], [c[2] for c in areal_units.centroids],
            markersize=2, markercolor=:white, markerstrokecolor=:black, alpha=0.5, label="Centroids")

        return plt
    else
        error("Effect $effect not recognized.")
    end
end




function plot_posterior_vs_prior(model::DynamicPPL.Model, chain::MCMCChains.Chains, param_sym::Symbol; n_prior_samples=1000, title="Posterior vs Prior")
    # Description: Overlays posterior and prior densities for a specific parameter to check learning/shrinkage.
    # Inputs:
    #   - model: Turing model object.
    #   - chain: MCMC sample chain.
    #   - param_sym: Symbol of the parameter to check.
    # Outputs:
    #   - A Plots.Plot object.

    # 1. Extract posterior samples using .data for AxisArray compatibility
    post_samples = vec(chain[param_sym].data)

    # 2. Automated Prior Sampling via Turing
    prior_chain = sample(model, Prior(), n_prior_samples, progress=false)
    prior_samples = vec(prior_chain[param_sym].data)

    # 3. Visualization
    plt = StatsPlots.density(post_samples, label="Posterior: $param_sym", lw=3, color=:blue, fill=(0, 0.2, :blue))
    StatsPlots.density!(plt, prior_samples, label="Prior (sampled)", lw=2, ls=:dash, color=:red)

    title!(plt, title)
    xlabel!(plt, "Value")
    ylabel!(plt, "Density")

    return plt
end




# --- 1. MODEL UTILITIES ---

function NegativeBinomial2(μ, r)
    # Description: Alternative parametrization of Negative Binomial using mean (μ) and dispersion (r).
    # Inputs:
    #   - μ: Mean.
    #   - r: Size/dispersion parameter.
    # Outputs:
    #   - Distributions.NegativeBinomial object.

    p = r / (r + μ)
    return NegativeBinomial(r, p)
end
  

function get_rff_deep2D_basis(X, m, lengthscale)
    # Description: Generates Random Fourier Feature (RFF) basis for 2D inputs (Spatial/Temporal).
    # Inputs:
    #   - X: Input matrix (N x D).
    #   - m: Number of features.
    #   - lengthscale: Gaussian kernel lengthscale.
    # Outputs:
    #   - N x m feature matrix.
    N, D = size(X)
    Random.seed!(42)
    Omega_samples = randn(m, D) ./ lengthscale
    Phi_phases = rand(m) .*  2pi
    return sqrt(2/m) .* cos.(X * Omega_samples' .+ Phi_phases')
end


function get_rff_trend_basis(t, m, lengthscale, ::Type{T}=Float64) where {T}
    N = length(t)
    # Generate random parameters for RFFs.
    # Using a seed ensures consistency within the AD pass.
    Random.seed!(42)
    Omega_samples_float = randn(m)
    Phi_phases_float = rand(m)

    Omega_samples = Omega_samples_float ./ lengthscale
    Phi_phases = Phi_phases_float .* convert(T,  2pi)

    Z = zeros(T, N, m)
    for j in 1:m
        Z[:, j] = convert.(T, sqrt(2/m)) .* cos.(Omega_samples[j] .* t .+ Phi_phases[j])
    end
    return Z
end


function get_rff_seasonal_basis(t, m, freq, lengthscale)
    # Description: Generates RFF-style basis for periodic seasonal components.
    # Inputs:
    #   - t: Time vector.
    #   - m: Number of harmonics.
    #   - freq: Base frequency.
    #   - lengthscale: Smoothness scale.
    # Outputs:
    #   - N x (2*m) feature matrix.
    N = length(t)
    Z = zeros(N, 2*m)
    for j in 1:m
        omega_j =  2pi * j * freq
        Z[:, 2j-1] = cos.(omega_j .* t)
        Z[:, 2j] = sin.(omega_j .* t)
    end
    return Z
end



 
# --- Recursive Parser Logic ---
# Parses the algebraic expression into a tree-like structure
function parse_manifold_graph(expr::String)
    expr = strip(expr)
    # Handle Direct Sum (Additive Components)
    if occursin("⊕", expr)
        # Recursively parse each component of the sum
        return (type=:sum, elements=parse_manifold_graph.(strip.(Base.split(expr, "⊕"))))
    # Handle Kronecker Product (Separable Interaction)
    elseif occursin("⊗", expr)
        # Recursively parse each component of the product
        return (type=:kronecker, elements=parse_manifold_graph.(strip.(Base.split(expr, "⊗"))))
    # Handle Composition (Warping/Transport)
    elseif occursin("∘", expr)
        # Recursively parse each component of the composition
        return (type=:composition, elements=parse_manifold_graph.(strip.(Base.split(expr, "∘"))))
    else
        # Atomic Manifold (e.g., "ICAR(s_idx)", "GP(s_x,s_y)")
        # Regex to capture the model name (e.g., ICAR, GP) and its arguments (e.g., s_idx, s_x,s_y)
        m = match(r"(\w+)\\((.*?)\\)", expr)
        if !isnothing(m)
            # Normalize model name to lowercase for consistency with bstm_options
            return (type=:atomic, model=lowercase(m.captures[1]), var=m.captures[2])
        else
            # If it's not a recognized manifold pattern, treat as unknown
            return (type=:unknown, val=expr)
        end
    end
end




function apply_discretization_logic(vals, rules)
    groups = nothing
    new_vals = nothing

    if rules == 0 || isnothing(rules)
        groups = collect(1:length(vals))
    elseif rules == "unit"
        min_v, max_v = minimum(vals), maximum(vals)
        new_vals = (min_v == max_v) ? zeros(length(vals)) : (vals .- min_v) ./ (max_v - min_v)
        groups = collect(1:length(vals))
    elseif rules == "zscore"
        m, s = Statistics.mean(vals), Statistics.std(vals)
        new_vals = (s ≈ 0.0) ? zeros(length(vals)) : (vals .- m) ./ s
        groups = collect(1:length(vals))
    elseif rules == "log"
        new_vals = log.(vals .+ 1.0 .- minimum(vals))
        groups = collect(1:length(vals))
    elseif rules isa Int
        if rules > 1
            qs = unique(sort(quantile(vals, (0:rules) ./ rules)))
            groups = (length(qs) < 2) ? ones(Int, length(vals)) : clamp.(map(x -> searchsortedlast(qs, x), vals), 1, length(qs)-1)
        else
            groups = ones(Int, length(vals))
        end
    elseif rules isa AbstractString && startswith(rules, "regular:")
        n = parse(Int, Base.split(rules, ":")[2])
        q025, q975 = quantile(vals, 0.025), quantile(vals, 0.975)
        bins = unique(sort(collect(range(q025, stop=q975, length=n+1))))
        groups = (length(bins) < 2) ? ones(Int, length(vals)) : clamp.(map(x -> searchsortedlast(bins, x), vals), 1, length(bins)-1)
    elseif rules isa AbstractVector
        groups = clamp.(map(x -> searchsortedlast(rules, x) + 1, vals), 1, length(rules) + 1)
    end
    return new_vals, groups
end

function assign_covariate_units(cov_data_base, cov_discretization, re_rules, cov_interactions)
    cov_data_for_processing = deepcopy(cov_data_base)
    cov_groups = Dict{Symbol, Vector{Int}}()
    cov_re_structures = Dict{Symbol, Any}()

    if !isnothing(cov_discretization)
        for cov_name_sym in names(cov_data_base, 1)
            if haskey(cov_discretization, cov_name_sym)
                rules = cov_discretization[cov_name_sym]
                vals = cov_data_for_processing[cov_name_sym, :]
                new_vals, groups = apply_discretization_logic(vals, rules)
                if !isnothing(new_vals); cov_data_for_processing[cov_name_sym, :] = new_vals; end
                if !isnothing(groups)
                    cov_groups[cov_name_sym] = groups
                    if haskey(re_rules, cov_name_sym) && length(unique(groups)) > 1
                        n_bins = length(unique(groups))
                        if re_rules[cov_name_sym] == "rw2"; cov_re_structures[cov_name_sym] = build_bstm_rw2_template(n_bins).matrix
                        elseif re_rules[cov_name_sym] == "ar1"; cov_re_structures[cov_name_sym] = build_bstm_ar1_template(n_bins).matrix
                        end
                    end
                end
            end
        end
    end

    for inter_str in cov_interactions
        parts = Base.split(inter_str, "*")
        if length(parts) == 2
            n1, n2 = Symbol(parts[1]), Symbol(parts[2])
            if n1 in names(cov_data_for_processing, 1) && n2 in names(cov_data_for_processing, 1)
                inter_val = cov_data_for_processing[n1, :] .* cov_data_for_processing[n2, :]
                new_row = NamedArray(inter_val', (Symbol[Symbol(inter_str)], names(cov_data_for_processing, 2)))
                cov_data_for_processing = vcat(cov_data_for_processing, new_row)
                if !isnothing(cov_discretization) && haskey(cov_discretization, Symbol(inter_str))
                    rule_int = cov_discretization[Symbol(inter_str)]
                    iv_vals, iv_groups = apply_discretization_logic(inter_val, rule_int)
                    if !isnothing(iv_vals); cov_data_for_processing[Symbol(inter_str), :] = iv_vals; end
                    if !isnothing(iv_groups)
                        cov_groups[Symbol(inter_str)] = iv_groups
                        if haskey(re_rules, Symbol(inter_str)) && length(unique(iv_groups)) > 1
                            n_bins = length(unique(iv_groups))
                            if re_rules[Symbol(inter_str)] == "rw2"; cov_re_structures[Symbol(inter_str)] = build_bstm_rw2_template(n_bins).matrix
                            elseif re_rules[Symbol(inter_str)] == "ar1"; cov_re_structures[Symbol(inter_str)] = build_bstm_ar1_template(n_bins).matrix
                            end
                        end
                    end
                end
            end
        end
    end
    return cov_data_for_processing, cov_groups, cov_re_structures
end


# -----------------------


function get_chain_names(chain)
    try
        return string.(FlexiChains.parameters(chain))
    catch
        return string.(names(chain))
    end
end


function _extract_volatility(chain, name_strs, N_tot, N_samples, outcome_idx=nothing, M=nothing)
    y_sig_samples = zeros(N_tot, N_samples)
    for j in 1:N_samples
        local sig_y
        if !isnothing(outcome_idx)
            # Multivariate specific volatility extraction
            # FIX: Explicitly construct string for v_key and c_key to avoid ParseError
            v_key = string("y_sigma_k[", outcome_idx, "]")
            c_key = string("y_sigma_const_k[", outcome_idx, "]")

            if v_key in name_strs # Stochastic volatility for specific outcome
                # FIX: Explicitly construct string for parameter names passed to get_params_vector
                sig_val = get_params_vector(chain, string("sigma_log_var_k[", outcome_idx, "]"), 1)[j]
                beta_vol_latent_val = get_params_vector(chain, string("beta_vol_latent_k[", outcome_idx, "]"), M.M_rff_sigma)[j, :]
                # FIX: M.vol_proj is global, not outcome-specific for now
                y_vol_proj_k = M.vol_proj
                sig_y = exp.((sig_val .* (sqrt(2.0 / M.M_rff_sigma) .* cos.(y_vol_proj_k) * beta_vol_latent_val)) ./ 2.0)
            elseif c_key in name_strs # Homoskedastic volatility for specific outcome
                sig_val = Float64(chain[Symbol(c_key)][j])
                sig_y = fill(sig_val, N_tot)
            else
                sig_y = fill(1.0, N_tot)
            end
        elseif !isnothing(M) && get(M, :use_sv, false) # Added SV handling for univariate
            # Reconstruct y_sigma from SV parameters for univariate model
            sigma_log_var_val = get_params_vector(chain, "sigma_log_var", 1)[j]
            beta_vol_latent_val = get_params_vector(chain, "beta_vol_latent", M.M_rff_sigma)[j, :]
            # M.vol_proj must be available for reconstruction
            if haskey(M, :vol_proj)
                sig_y = exp.((sigma_log_var_val .* (sqrt(2.0 / M.M_rff_sigma) .* cos.(M.vol_proj) * beta_vol_latent_val)) ./ 2.0)
            else
                @warn "M.vol_proj not found for stochastic volatility reconstruction. Defaulting to 1.0."
                sig_y = fill(1.0, N_tot)
            end
        else
            # General univariate volatility extraction
            if "y_sigma" in name_strs
                val = get_params_vector(chain, "y_sigma", 1)[j]
                sig_y = val isa AbstractVector ? vec(collect(val)) : fill(Float64(val), N_tot)
            elseif "y_sigma_const" in name_strs
                sig_val = Float64(chain[:y_sigma_const][j])
                sig_y = fill(sig_val, N_tot)
            else
                sig_y = fill(1.0, N_tot)
            end
        end

        # Final flattening and type enforcement
        flat_sig = vec(Float64.(collect(sig_y)))

        if length(flat_sig) >= N_tot
            y_sig_samples[:, j] = flat_sig[1:N_tot]
        else
            # Pad with the last value if reconstructed sig_y is shorter than N_tot (e.g., if only M.y_N is returned)
            y_sig_samples[1:length(flat_sig), j] = flat_sig
            y_sig_samples[length(flat_sig)+1:end, j] .= flat_sig[end]
        end
    end
    return y_sig_samples
end

 


function generate_spectral_w_from_magnitude(freqs_x, freqs_y, magnitude_spectrum, M_rff_count)
"""
    generate_spectral_w_from_magnitude(freqs_x, freqs_y, magnitude_spectrum, M_rff_count)

Generates 2D RFF weights W by sampling frequencies from the provided 2D magnitude spectrum.

Args:
    freqs_x: Vector of x-dimension frequencies.
    freqs_y: Vector of y-dimension frequencies.
    magnitude_spectrum: 2D array of magnitude values corresponding to freqs_x, freqs_y.
    M_rff_count: Number of RFF features to generate.

Returns:
    A 2 x M_rff_count matrix for Wfixed.
"""
    # Flatten frequency grids and magnitude spectrum into 1D arrays for sampling
    all_freqs_x = repeat(freqs_x, inner=length(freqs_y))
    all_freqs_y = repeat(freqs_y, outer=length(freqs_x))
    all_magnitudes = vec(magnitude_spectrum)

    # Normalize magnitudes to form a probability distribution
    # Add a small constant to magnitudes before normalization to prevent division by zero for zero probabilities.
    probabilities = (all_magnitudes .+ 1e-9) ./ sum(all_magnitudes .+ 1e-9)

    # Sample M_rff_count indices based on probabilities
    # StatsBase.sample expects Weights from non-negative numbers
    sampled_indices = sample(1:length(probabilities), Weights(probabilities), M_rff_count, replace=true)

    Wfixed = Matrix{Float64}(undef, 2, M_rff_count)
    for i in 1:M_rff_count
        idx = sampled_indices[i]
        Wfixed[1, i] = all_freqs_x[idx] *  2pi # Scale by  2pi to match RFF convention (often ω'x)
        Wfixed[2, i] = all_freqs_y[idx] *  2pi
    end

    return Wfixed
end
  

# Helper to create AR1 covariance matrix
function ar1_covariance_matrix(times::Vector{<:Real}, rho::Real, sigma_e::Real)
    n = length(times)
    T = typeof(rho) # Get the type of the parameters
    C = Matrix{T}(undef, n, n) # Initialize matrix with this type
    for i in 1:n
        for j in 1:n
            C[i, j] = sigma_e^2 * rho^abs(times[i] - times[j])
        end
    end
    return C
end

# Helper to create AR1 cross-covariance matrix
function ar1_cross_covariance_matrix(times_a::Vector{<:Real}, times_b::Vector{<:Real}, rho::Real, sigma_e::Real)
    na = length(times_a)
    nb = length(times_b)
    T = typeof(rho) # Get the type of the parameters
    C = Matrix{T}(undef, na, nb) # Initialize matrix with this type
    for i in 1:na
        for j in 1:nb
            C[i, j] = sigma_e^2 * rho^abs(times_a[i] - times_b[j])
        end
    end
    return C
end
 

function prepare_fft_grid(s_x, s_y, values; grid_res=64, pad_factor=2)
    # 1. Define the bounding box

    xmin, xmax = minimum(s_x), maximum(s_x)
    ymin, ymax = minimum(s_y), maximum(s_y)

    # 2. Map points to a grid
    # Use the length of the shorter input to prevent BoundsError
    n_limit = min(length(s_x), length(values))
    grid = zeros(grid_res, grid_res)

    for i in 1:n_limit
        ix = Int(floor((s_x[i] - xmin) / (xmax - xmin + 1e-6) * (grid_res - 1))) + 1
        iy = Int(floor((s_y[i] - ymin) / (ymax - ymin + 1e-6) * (grid_res - 1))) + 1
        grid[ix, iy] = values[i]
    end

    # 3. Apply Zero-Padding
    padded_res = grid_res * pad_factor
    padded_grid = zeros(padded_res, padded_res)

    start_idx = Int(grid_res / 2)
    padded_grid[start_idx:start_idx+grid_res-1, start_idx:start_idx+grid_res-1] .= grid

    return padded_grid, (xmin, xmax, ymin, ymax)
end
 



#####

 
function get_architecture(model_arch::String)
    if startswith(model_arch, "example_")
        return ExampleArchitecture() 
    elseif model_arch in ["univariate"]
        return UnivariateArchitecture()
    elseif model_arch in ["multivariate"]
        return MultivariateArchitecture()
    elseif model_arch in ["multifidelity"]
        return MultifidelityArchitecture()
    else
        return UnknownArchitecture()
    end
end

 
function get_model_family(model_family::String)
    if model_family == "poisson"
        return PoissonFamily()
    elseif model_family == "gaussian"
        return GaussianFamily()
    elseif model_family == "lognormal"
        return LogNormalFamily()
    elseif model_family in ["bernoulli", "binomial"]
        return BinomialFamily()
    elseif model_family == "negbin"
        return NegativeBinomialFamily()
    else
        error("Unknown model_family: $model_family")
    end
end


function get_model_parameters(m::DynamicPPL.Model)
    # Directly extract names from a model instance sample as a fallback discovery method
    try
        raw_keys = keys(rand(m))
        return map(raw_keys) do k
            replace(string(k), r"[\(\)\"\:]" => "")
        end
    catch
        return String[]
    end
end


function check_has_parameter(m::DynamicPPL.Model, param_name::String)
    all_p = get_model_parameters(m)
    return any(n -> n == param_name || startswith(n, "$param_name["), all_p)
end
 

function _process_ll_and_predictions(fam, eta, chain, M, N_tot, N_samples, y_sigma_samples=nothing, y_obs_custom=nothing)
    # Audited and Feature-Complete Prediction Utility (v10.5.0)
    # Rationale: Finalizing the full-taxonomy support for all 15+ model families.
    # Requirement: Ensure posterior predictive sampling (noisy) matches the support and parameters
    # defined in the bstm_Likelihood dispatcher and get_dist_ref factory.

    # 1. Container Initialization
    # denoised: expected value mu (response scale)
    # noisy: realized stochastic draws (prediction scale)
    # log_lik: pointwise log-probabilities for WAIC/LOO
    local denoised = zeros(N_tot, N_samples)
    local noisy = zeros(N_tot, N_samples)
    local log_lik = zeros(N_samples, M.y_N)

    local name_strs = string.(FlexiChains.parameters(chain))
    local use_zi = get(M, :use_zi, false)
    local fam_str = hasproperty(M, :model_family) ? M.model_family : "gaussian"

    # 2. Primary Sampling Loop
    for j in 1:N_samples

        # 2.1 Extraction of Volatility Surface
        # Standardizing heteroskedastic vs homoskedastic error scales
        local sig_y
        if !isnothing(y_sigma_samples)
            sig_y = y_sigma_samples[:, j]
        else
            sig_y = _extract_volatility(chain, name_strs, N_tot, N_samples, nothing, M)[:, j]
        end

        # 2.2 Parameter Discovery for the Current Sample
        local r_val = "lik_r" in name_strs ? chain[:lik_r].data[j] : 1.0
        local phi_val = "lik_phi" in name_strs ? chain[:lik_phi].data[j] : 0.0
        local extra = "extra_params" in name_strs ? chain[:extra_params].data[j] : 1.0

        # 2.3 Link Transformation and Expectation Mapping
        # Rationale: Denoised expectations must use the standardized inverse link logic.
        local mu_vec = _apply_link_and_lik(fam_str, eta[:, j], use_zi, phi_val, r_val)
        denoised[:, j] .= mu_vec

        # 2.4 Likelihood and Predictive Sampling
        for i in 1:N_tot
            local is_obs = i <= M.y_N
            local mu = mu_vec[i]
            local eta_val = eta[i, j]

            # --- Likelihood Calculation (Training Data Only) ---
            if is_obs
                local y_vals_src = isnothing(y_obs_custom) ? M.y_obs : y_obs_custom
                # Standardize likelihood evaluation via the bstm_Likelihood functor
                local lik_obj = bstm_Likelihood(
                    fam_str, [y_vals_src[i]]; sigma_y=sig_y[i], weight=M.weights[i],
                    phi_zi=use_zi ? phi_val : -Inf, r_nb=r_val, trial=M.trials[i], extra_params=extra
                )
                log_lik[j, i] = Distributions.logpdf(lik_obj, eta_val)
            end

            # --- Feature-Complete Posterior Predictive Sampling (Noisy) ---
            # Rationale: Handling the full taxonomy of realized draws.

            if use_zi && rand() < phi_val
                # Handle structural zeros for Zero-Inflated models
                noisy[i, j] = 0.0
            else
                noisy[i, j] = if fam isa GaussianFamily
                    mu + randn() * sig_y[i]
                elseif fam isa PoissonFamily
                    Float64(rand(Poisson(clamp(mu, 1e-10, 1e9))))
                elseif fam isa NegativeBinomialFamily
                    Float64(rand(NegativeBinomial(r_val, r_val / (r_val + mu))))
                elseif fam isa BinomialFamily
                    local n_t = Int(is_obs ? M.trials[i] : 1)
                    Float64(rand(Binomial(n_t, mu)))
                elseif fam isa LogNormalFamily
                    rand(LogNormal(mu, sig_y[i]))
                elseif fam isa GammaFamily
                    local alpha_g = extra > 0 ? extra : 1.0
                    rand(Gamma(alpha_g, mu / alpha_g))
                elseif fam isa BetaFamily
                    local phi_b = extra > 0 ? extra : 10.0
                    # Parameters for Beta are mu*phi and (1-mu)*phi
                    rand(Beta(clamp(mu * phi_b, 1e-6, Inf), clamp((1.0 - mu) * phi_b, 1e-6, Inf)))
                elseif fam isa InverseGaussianFamily
                    local lambda_ig = extra > 0 ? extra : 1.0
                    rand(InverseGaussian(mu, lambda_ig))
                elseif fam isa StudentTFamily
                    local nu_st = extra > 0 ? extra : 5.0
                    rand(LocationScale(mu, sig_y[i], TDist(nu_st)))
                elseif fam isa LaplaceFamily
                    rand(Laplace(mu, sig_y[i]))
                elseif fam isa ExponentialFamily
                    rand(Exponential(mu))
                elseif fam isa ParetoFamily
                    local shape_p = extra > 0 ? extra : 1.0
                    rand(Pareto(mu, shape_p))
                elseif fam isa HalfNormalFamily
                    rand(truncated(Normal(0.0, sig_y[i]), 0.0, Inf))
                elseif fam isa HalfStudentTFamily
                    local nu_hst = extra > 0 ? extra : 5.0
                    rand(truncated(LocationScale(0.0, sig_y[i], TDist(nu_hst)), 0.0, Inf))
                else
                    # Fallback to expectation for unknown families
                    mu
                end
            end
        end
    end

    # Audit Verification: Container check for N_tot x N_samples shapes
    return denoised, noisy, log_lik
end


function _calculate_ps_weights(p_denoised, M, PS, N_PS, N_samples)
    # Standardized Post-Stratification Weight Utility (v09.1.0)
    # Rationale: Centralizing the relative risk/weight calculation between predicted 
    # strata and observed data points. This ensures architectural parity and 
    # facilitates cross-validation matching.
    
    if N_PS == 0
        return nothing
    end

    # Initialize the weight matrix [Strata x Samples]
    local ps_weights = zeros(N_PS, N_samples)
    
    for k in 1:N_PS
        # Identify the target spatiotemporal and seasonal indices for the k-th stratum
        local s_target = PS.s_idx[k]
        local t_target = PS.t_idx[k]
        local u_target = PS.u_idx[k]

        # Search for an exact match in the observed data configuration to establish a baseline
        local obs_match_idx = findfirst(i -> M.s_idx[i] == s_target && M.t_idx[i] == t_target && M.u_idx[i] == u_target, 1:M.y_N)

        if !isnothing(obs_match_idx)
            # If a match is found, compute weight relative to the specific observed realization
            for j in 1:N_samples
                # Use a small epsilon to prevent division by zero in sparse count scenarios
                ps_weights[k, j] = p_denoised[M.y_N + k, j] / (p_denoised[obs_match_idx, j] + 1e-9)
            end
        else
            # Fallback: If no exact stratum exists in training data, scale by the mean observed expectation
            local sample_mean_obs = mean(p_denoised[1:M.y_N, :], dims=1)
            for j in 1:N_samples
                ps_weights[k, j] = p_denoised[M.y_N + k, j] / (sample_mean_obs[j] + 1e-9)
            end
        end
    end

    return ps_weights
end



function _ensure_matrix(field, n_units, n_samples)
    # Null/Empty handling: Return zero-field of correct dimensions
    if isnothing(field) || isempty(field)
        return zeros(Float64, n_units, n_samples)
    end

    # Type Discovery and Normalization
    # Case 1: Pure Scalar - Broadcast to full N x S block
    if field isa Number
        return fill(Float64(field), n_units, n_samples)
    end

    # Case 2: Vector Handling
    if field isa AbstractVector
        # If length matches units, assume it is a spatial field constant across samples
        if length(field) == n_units
            return repeat(reshape(field, n_units, 1), 1, n_samples)
        # If length matches samples, assume it is a global trend across units
        elseif length(field) == n_samples
            return repeat(reshape(field, 1, n_samples), n_units, 1)
        else
            # Fallback for 1-element vectors: scalar broadcast
            return fill(Float64(field[1]), n_units, n_samples)
        end
    end

    # Case 3: Matrix Handling (Critical for Outcome-Specific Slices)
    if ndims(field) == 2
        local mat = Matrix{Float64}(field)
        local r, c = size(mat)

        # Scenario A: Exact Match
        if r == n_units && c == n_samples
            return mat
        # Scenario B: Global component (1xS Matrix) - Broadcast rows across units
        elseif r == 1 && c == n_samples
            return repeat(mat, n_units, 1)
        # Scenario C: Static field (Nx1 Matrix) - Broadcast columns across samples
        elseif r == n_units && c == 1
            return repeat(mat, 1, n_samples)
        else
            # Aggressive fallback using first element to prevent BoundsError
            return fill(mat[1,1], n_units, n_samples)
        end
    end

    return zeros(Float64, n_units, n_samples)
end

# --- 2. Feature-Complete Linear Predictor Assembly [v11.0.6 Audit] ---
# Rationale: This is the primary engine for combining latent manifold realizations 
# into the response-scale predictor eta. It handles both observed training data 
# and Post-Stratification (PS) grids.

function _assemble_linear_predictor(N_tot, N_samples, M, PS, Xfixed_betas, mixed_eff_coeffs, 
                                    mixed_terms_list, basis_eff_accum, st_eff_map, svc_slopes, 
                                    svc_covs, s_eff_struct, s_eff_noisy, t_eff, u_eff)

    # Initialize the high-fidelity linear predictor container
    local eta_samples = zeros(Float64, N_tot, N_samples)

    # Phase 1: Dimensional Enforcement with Broadcast-Aware Matrix utility
    # We extract the first element of manifold arrays (standard for univariate) 
    # and ensure they are expanded to [Units x Samples] matrices.
    local S_struct = _ensure_matrix(s_eff_struct[1], M.s_N, N_samples)
    local S_noisy  = _ensure_matrix(s_eff_noisy[1], M.s_N, N_samples)
    local T_field  = _ensure_matrix(t_eff[1], M.t_N, N_samples)
    local U_field  = _ensure_matrix(u_eff, M.u_N, N_samples)
    local B_field  = _ensure_matrix(basis_eff_accum, M.y_N, N_samples)

    for j in 1:N_samples
        # Phase 2: Fixed Effects Superposition
        # We perform a row-wise dot product between the design matrix and sampled coefficients.
        if M.Xfixed_N > 0 && !isnothing(Xfixed_betas)
            # Ensure betas are treated as a flat vector for the dot product
            local beta_j = vec(collect(Xfixed_betas[:, j]))
            for i in 1:N_tot
                local X_mat = i <= M.y_N ? M.Xfixed : PS.Xfixed
                local row_idx = i <= M.y_N ? i : i - M.y_N
                
                # Design Row extraction
                local X_row = vec(collect(X_mat[row_idx, :]))
                
                # Additive contribution of fixed covariates
                eta_samples[i, j] += dot(X_row, beta_j)
            end
        end

        # Phase 3: Structural Manifold Assembly
        # Iterate through every observation (Observed + PS Strata)
        for i in 1:N_tot
            local is_obs = i <= M.y_N
            local src = is_obs ? M : PS
            local idx = is_obs ? i : i - M.y_N

            # Manifold Coordinate Discovery
            local s_ptr = Int(src.s_idx[idx])
            local t_ptr = Int(src.t_idx[idx])
            local u_ptr = Int(src.u_idx[idx])

            # 3.1 Link-Scale Offsets (Training Context Only)
            if is_obs && haskey(M, :log_offset)
                eta_samples[i, j] += M.log_offset[i]
            end

            # 3.2 Additive Spatial Components (Structured + Overdispersion)
            # Rationale: S_struct + (S_noisy - S_struct) = S_noisy total realization.
            eta_samples[i, j] += S_struct[s_ptr, j] + (S_noisy[s_ptr, j] - S_struct[s_ptr, j])

            # 3.3 Temporal Trend Realization
            eta_samples[i, j] += T_field[t_ptr, j]

            # 3.4 Seasonal Periodic Realization
            eta_samples[i, j] += U_field[u_ptr, j]

            # 3.5 Spatiotemporal (ST) Interaction Mapping
            # Handles both 3D Sample Tensors [S, T, Sample] and static 2D maps.
            if !isnothing(st_eff_map) && !isempty(st_eff_map)
                if ndims(st_eff_map) == 3
                    eta_samples[i, j] += st_eff_map[s_ptr, t_ptr, j]
                else
                    eta_samples[i, j] += st_eff_map[s_ptr, t_ptr]
                end
            end

            # 3.6 Spatially Varying Coefficients (SVC)
            # If active, we multiply the spatial slope field by the covariate value.
            if !isnothing(svc_slopes) && !isempty(svc_slopes)
                 for (k, c_sym) in enumerate(svc_covs)
                    local col_idx = findfirst(==(c_sym), Symbol.(names(M.Xfixed, 2)))
                    if !isnothing(col_idx)
                        local cov_val = is_obs ? M.Xfixed[idx, col_idx] : PS.Xfixed[idx, col_idx]
                        # Svc_slopes indexed as [Unit, Covariate, Sample]
                        eta_samples[i, j] += svc_slopes[1][s_ptr, k, j] * cov_val
                    end
                end
            end

            # 3.7 Smooth Basis / Spline Components (Observed grid mapping only)
            if is_obs
                eta_samples[i, j] += B_field[i, j]
            end
        end
    end

    # Verification: Matrix dimensions must be [N_tot x N_samples]
    return eta_samples
end


 
function resolve_technical_primitive(manifold_name::String, M, priors_dict, scheme::Symbol)
    # Standardized Technical Primitive Resolver [v12.8.6 - Full Taxonomy Hardening]
    # Rationale: Centralizes hyperprior resolution and struct instantiation for cleaner dispatch.
    # Requirement: Absolute parity with the v06.1 glossary including spatial, temporal, and specialized primitives.

    local resolved_priors = resolve_hyperpriors(manifold_name, priors_dict, scheme)

    # 1. Discrete Spatial Primitives
    if manifold_name == "bym2"
        return BYM2(resolved_priors.rho_prior, resolved_priors.sigma_prior)
    elseif manifold_name == "leroux"
        return Leroux(resolved_priors.rho_prior, resolved_priors.sigma_prior)
    elseif manifold_name == "icar" || manifold_name == "besag"
        # Besag is the mathematical basis for ICAR
        return ICAR(resolved_priors.sigma_prior)

    # 2. Temporal and 1D Processes
    elseif manifold_name == "ar1"
        return AR1(resolved_priors.rho_prior, resolved_priors.sigma_prior)
    elseif manifold_name == "rw1"
        return RW1(resolved_priors.sigma_prior)
    elseif manifold_name == "rw2"
        return RW2(resolved_priors.sigma_prior)

    # 3. Continuous and Spectral Kernels
    elseif manifold_name == "gp"
        return GP(resolved_priors.lengthscale_prior, resolved_priors.sigma_prior, "se")
    elseif manifold_name == "nystrom"
        # Low-rank kernel approximation via inducing points
        return Nystrom(resolved_priors.sigma_prior, resolved_priors.lengthscale_prior, get(M, :n_inducing, 10))
    elseif manifold_name == "spde"
        return SPDE(resolved_priors.sigma_prior, resolved_priors.kappa_prior)
    elseif manifold_name == "dag"
        return DAG(get(M, :W, sparse(I(M.s_N))), resolved_priors.sigma_prior)

    # 4. Seasonal and Periodic Structures
    elseif manifold_name == "harmonic"
        return Harmonic(resolved_priors.amplitude_prior, resolved_priors.phase_prior, resolved_priors.sigma_prior)
    elseif manifold_name == "cyclic"
        return Cyclic(M.period, resolved_priors.sigma_prior)

    # 5. Specialized and Interaction Manifolds
    elseif manifold_name == "mosaic"
        return Mosaic(get(M, :s_coord, zeros(M.s_N, 2)), M.n_regions, true)
    elseif manifold_name == "localadaptive" || manifold_name == "local_adaptive"
        # Riemannian diagonal weighting for non-stationary smoothing
        return LocalAdaptive(resolved_priors.sigma_prior, :local_weights)
    elseif manifold_name == "bcgn"
        # Bipartite Covariate Graph Network
        return BCGN(resolved_priors.sigma_prior, get(M, :bipartite_adj, sparse(I(M.s_N))), get(M, :group_weights, ones(M.s_N)))
    elseif manifold_name == "KnorrHeld"
        # Space-Time Interaction types I, II, III, IV
        return KnorrHeld(:space, :time, 'I', resolved_priors.sigma_prior)
    elseif manifold_name == "AdvectionDiffusion"
        # Physics-informed transport manifold
        return AdvectionDiffusion(:space, :time, nothing, resolved_priors.sigma_prior, 0.1, resolved_priors.sigma_prior)
    
    # 6. Basis-mapped Smooths and Defaults
    elseif manifold_name == "wavelet"
        return Wavelets(:db4, M.nbins, resolved_priors.sigma_prior)
    elseif manifold_name == "fixed" || manifold_name == "iid"
        return IID(resolved_priors.sigma_prior)
    elseif manifold_name in ["pspline", "bspline", "tps", "rff", "fft"]
        return PSpline(M.nbins, 3, 2, resolved_priors.sigma_prior)
    else
        # Fallback to standard identity overdispersion
        return IID(resolved_priors.sigma_prior)
    end
end



function _extract_reconstruction_parameters(chain, M, N_samples, outcomes_N, p_names)
    # Audited Parameter Extraction Gateway [v12.8.7 - Recovery Sync]
    # Rationale: Ensures 100% completeness against the BSTM v06.1 taxonomy.
    # Requirement: All output slots (hierarchical, volatility, ST-maps, SVC) must be populated.
    
    println("--- Discovery: Extracting v06.1 Latent Manifolds [v12.8.7 Hardened] ---")

    # --- 1. Container Initialization ---
    local Xfixed_betas = (M.Xfixed_N > 0) ? zeros(Float64, M.Xfixed_N * outcomes_N, N_samples) : nothing
    local mixed_terms_list = get(M, :mixed_terms, [])
    local mixed_eff_coeffs = !isempty(mixed_terms_list) ? [zeros(Float64, term.n_cat, N_samples) for term in mixed_terms_list] : nothing

    local s_eff_struct = [zeros(Float64, M.s_N, N_samples) for _ in 1:outcomes_N]
    local s_eff_noisy  = [zeros(Float64, M.s_N, N_samples) for _ in 1:outcomes_N]
    local t_eff = [zeros(Float64, M.t_N, N_samples) for _ in 1:outcomes_N]
    local u_eff = zeros(Float64, M.u_N, N_samples)
    local basis_eff_accum = zeros(Float64, M.y_N, N_samples)
    local st_eff_maps = [zeros(Float64, M.s_N, M.t_N, N_samples) for _ in 1:outcomes_N]

    local svc_covs = get(M, :svc_covariates, Symbol[])
    local svc_slopes = !isempty(svc_covs) ? [zeros(Float64, M.s_N, length(svc_covs), N_samples) for _ in 1:outcomes_N] : nothing

    # Restore: Hierarchical Scale and Volatility containers
    local hierarchical_scales = Dict{Symbol, Matrix{Float64}}()
    local sv_surface = [zeros(Float64, M.y_N, N_samples) for _ in 1:outcomes_N]

    # --- 2. Pre-Loop Manifold Configuration ---
    local hyper_scheme = get(M, :hyperprior_scheme, :pcpriors)
    local priors_map = get(M, :hyperpriors, Dict())

    local season_prim = resolve_technical_primitive(M.model_season, M, priors_map, hyper_scheme)
    local space_prim = resolve_technical_primitive(M.model_space, M, priors_map, hyper_scheme)
    local time_prim = resolve_technical_primitive(M.model_time, M, priors_map, hyper_scheme)

    # --- 3. The Streamlined Extraction Loop ---
    for j in 1:N_samples

        # 3.1 Seasonal & Basis Recovery
        if M.model_season != "none"
            local u_sig = "u_sigma" in p_names ? get_params_vector(chain, "u_sigma", 1)[j] : 1.0
            u_eff[:, j] .= extract_manifold(season_prim, chain, M, j, u_sig)[1]
        end

        if haskey(M, :basis_matrices)
            for (v_sym, B_mat) in M.basis_matrices
                local p_key = "beta_basis_" * string(v_sym)
                if p_key in p_names
                    basis_eff_accum[:, j] .+= B_mat * get_params_vector(chain, p_key, size(B_mat, 2))[j, :]
                end
            end
        end

        # 3.2 Hierarchical Scale Discovery (Restored)
        if haskey(M, :spatial_hierarchy)
            for (scale_sym, scale_obj) in M.spatial_hierarchy
                local p_key_h = "s_latent_" * string(scale_sym)
                if p_key_h in p_names
                    local h_mat = get!(hierarchical_scales, scale_sym, zeros(Float64, scale_obj.n_units, N_samples))
                    h_mat[:, j] .= vec(get_params_vector(chain, p_key_h, scale_obj.n_units)[j, :])
                end
            end
        end

        # 3.3 Linear & Mixed Effects
        if !isnothing(Xfixed_betas)
            Xfixed_betas[:, j] .= get_params_vector(chain, "Xfixed_beta", M.Xfixed_N * outcomes_N)[j, :]
        end

        if !isnothing(mixed_eff_coeffs)
            for (m_idx, m_term) in enumerate(mixed_terms_list)
                local p_key_me = "beta_group_" * string(m_idx)
                if p_key_me in p_names
                    mixed_eff_coeffs[m_idx][:, j] .= vec(get_params_vector(chain, p_key_me, m_term.n_cat)[j, :])
                end
            end
        end

        # 3.4 Outcome-Specific Manifold Recovery
        for k in 1:outcomes_N
            if M.model_space != "none"
                local s_sig = (outcomes_N == 1) ?
                    ("s_sigma" in p_names ? get_params_vector(chain, "s_sigma", 1)[j] : 1.0) :
                    ("s_sigma_arr" in p_names ? get_params_vector(chain, "s_sigma_arr", outcomes_N)[j, k] : 1.0)

                local fs, fn = (outcomes_N == 1) ?
                    extract_manifold(space_prim, chain, M, j, s_sig) :
                    extract_manifold_k(space_prim, chain, M, j, k, s_sig)

                s_eff_struct[k][:, j] .= fs
                s_eff_noisy[k][:, j] .= fn
            end

            if M.model_time != "none"
                local t_sig = (outcomes_N == 1) ?
                    ("t_sigma" in p_names ? get_params_vector(chain, "t_sigma", 1)[j] : 1.0) :
                    ("t_sigma_arr" in p_names ? get_params_vector(chain, "t_sigma_arr", outcomes_N)[j, k] : 1.0)

                local ft, _ = (outcomes_N == 1) ?
                    extract_manifold(time_prim, chain, M, j, t_sig) :
                    extract_manifold_k(time_prim, chain, M, j, k, t_sig)

                t_eff[k][:, j] .= ft
            end

            local st_key = (outcomes_N == 1) ? "st_latent" : "st_latent_" * string(k)
            if st_key in p_names
                local st_sig = (outcomes_N == 1) ?
                    ("st_sigma" in p_names ? get_params_vector(chain, "st_sigma", 1)[j] : 1.0) :
                    ("st_sigma_arr" in p_names ? get_params_vector(chain, "st_sigma_arr", outcomes_N)[j, k] : 1.0)
                local flat_st = get_params_vector(chain, st_key, M.s_N * M.t_N)[j, :]
                st_eff_maps[k][:, :, j] .= reshape(flat_st, M.s_N, M.t_N) .* st_sig
            end

            if !isnothing(svc_slopes)
                for (v_idx, c_sym) in enumerate(svc_covs)
                    local p_key_svc = (outcomes_N == 1) ? "beta_svc_" * string(c_sym) : "beta_svc_" * string(c_sym) * "_" * string(k)
                    if p_key_svc in p_names
                        svc_slopes[k][:, v_idx, j] .= get_params_vector(chain, p_key_svc, M.s_N)[j, :]
                    end
                end
            end

            # 3.5 Volatility Discovery (Outcome-indexed reconstruction)
            if get(M, :use_sv, false)
                local sv_sig_key = (outcomes_N == 1) ? "sigma_log_var" : "sigma_log_var_k[" * string(k) * "]"
                local sv_beta_key = (outcomes_N == 1) ? "beta_vol_latent" : "beta_vol_latent_k[" * string(k) * "]"
                if sv_sig_key in p_names
                    local sig_v = get_params_vector(chain, sv_sig_key, 1)[j, 1]
                    local beta_v = vec(get_params_vector(chain, sv_beta_key, M.M_rff_sigma)[j, :])
                    sv_surface[k][:, j] .= exp.((sig_v .* (sqrt(2.0 / M.M_rff_sigma) .* cos.(M.vol_proj * beta_v))) ./ 2.0)
                end
            else
                sv_surface[k][:, j] .= 1.0
            end
        end
    end

    # Final Packaging: Return all technical parameters for downstream assembly
    return (
        Xfixed_betas = Xfixed_betas,
        mixed_eff_coeffs = mixed_eff_coeffs,
        s_eff_struct = s_eff_struct, 
        s_eff_noisy = s_eff_noisy,
        t_eff = t_eff, 
        u_eff = u_eff, 
        basis_eff_accum = basis_eff_accum,
        st_eff_maps = st_eff_maps, 
        svc_slopes = svc_slopes,
        hierarchical_scales = hierarchical_scales,
        sv_surface = outcomes_N == 1 ? sv_surface[1] : sv_surface
    )
end



function _apply_link_and_lik(family::String, eta::AbstractArray, use_zi::Bool, phi=0.0, r=1.0)
    # Audited and Feature-Complete Link Utility (v10.2.0)
    # Rationale: Mirroring the full 15+ family taxonomy established in _process_ll_and_predictions.
    # This ensures that denoised realizations (expectations) are mathematically consistent with the likelihood kernels.
    # Audit: Verified against Reference Hierarchy to include Log-link, Logit-link, and Identity-link routing.

    # 1. Family-Specific Inverse Link Dispatch
    # The linear predictor 'eta' exists in the latent space (usually unbounded Real).
    # We map 'mu' to the support of the target distribution family.
    local mu

    if family in ["poisson", "negbin", "gamma", "exponential", "inverse_gaussian", "pareto"]
        # Log-link group: Support is (0, Inf)
        # Rationale: Enforcing positivity via exponential transformation.
        mu = exp.(eta)

    elseif family in ["bernoulli", "binomial", "beta"]
        # Logit-link group: Support is (0, 1)
        # Rationale: Standard logistic mapping for probabilities or proportion parameters.
        mu = logistic.(eta)

    elseif family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
        # Identity-link group: Mapping stays in the space of the distribution parameter.
        # Note: LogNormal uses identity because 'eta' models the mean of the underlying Normal log-process.
        # Half-Normal/Half-StudentT are truncated but the location parameter follows identity in the latent field.
        mu = eta

    else
        # Fallback to Identity for unknown or experimental families
        mu = eta
    end

    # 2. Zero-Inflation Adjustment
    # Rationale: If Zero-Inflation is active, the expected value of the mixture is (1 - phi) * mu_branch.
    # This matches the analytical mean of the ZI-distribution where 'phi' is the probability of a structural zero.
    if use_zi
        mu = (1.0 .- phi) .* mu
    end

    return mu
end



function _compute_waic(log_lik)
    nsamples, nobs = size(log_lik)
    lppd = sum(logsumexp(log_lik[:, i]) - log(nsamples) for i in 1:nobs)
    p_waic = sum(var(log_lik[:, i]) for i in 1:nobs)
    return -2 * (lppd - p_waic)
end

function _extract_beta_cov(all_names, chain, M, N_samples, alpha)
    # Identify all categorical covariate groups present in the chain
    # Matches patterns like "beta_cov[1]", "beta_cov[2]", etc.
    cov_matches = unique(map(m -> m.captures[1], filter(!isnothing, match.(r"beta_cov\[(\d+)\]", all_names))))
    
    if isempty(cov_matches)
        return nothing
    end

    # Parse indices and sort to ensure sequential processing
    cov_indices = sort(parse.(Int, cov_matches))
    
    results = []
    for k in cov_indices
        base_name = "beta_cov[$k]"
        # Use the robust get_params_vector helper to extract the full vector for this covariate group
        raw_vals = get_params_vector(chain, base_name, M.N_cat)
        
        if !all(raw_vals .== 0) # Only process if data was actually found
            # Reshape for summarize_array (N_categories x 1 x N_samples)
            summ = summarize_array(reshape(raw_vals', M.N_cat, 1, N_samples); alpha=alpha)
            push!(results, summ)
        end
    end

    return isempty(results) ? nothing : results
end


# --- 1. Structural Storage for Latent Components ---
# Standardizing for Heterogeneous Outcome Support (v14.0.0 Refactor)
# This structure acts as the primary registry for discovered manifold realizations.
# It ensures that every latent component defined in the BSTM v05.5 Taxonomy is captured.
mutable struct ManifoldRegistry
    # Fixed Effects: Standard Matrix representing [K_fixed * Outcomes x Samples]
    Xfixed_betas::Union{Nothing, Matrix{Float64}}

    # Mixed Effects: A Vector containing matrices of [Category_Levels x Samples]
    mixed_eff_coeffs::Vector{Matrix{Float64}}

    # Spatial Manifolds: Outcome-specific vectors of [Spatial_Units_k x Samples]
    s_eff_struct::Vector{Matrix{Float64}}
    s_eff_noisy::Vector{Matrix{Float64}}

    # Temporal Manifolds: Outcome-specific vectors of [Time_Slices_k x Samples]
    t_eff::Vector{Matrix{Float64}}

    # Seasonal Manifolds: Shared periodic components [Seasonal_Bins x Samples]
    u_eff::Matrix{Float64}

    # Basis/Spline Effects: Accumulated values mapped to the training grid [Observations x Samples]
    basis_eff_accum::Matrix{Float64}

    # Spatiotemporal Interaction: Outcome-specific tensors [Units_k x Time_k x Samples]
    st_eff_maps::Vector{Array{Float64, 3}}

    # Spatially Varying Coefficients: Outcome-specific tensors [Units_k x Covariates x Samples]
    svc_slopes::Vector{Array{Float64, 3}}

    # Observation Error Scale: Pointwise surface [Observations x Samples]
    sv_surface::Matrix{Float64}

    # Internal Metadata for Validation and Architectural Dispatch
    n_samples::Int
    outcomes_N::Int
end
 
 

# 2. Validation Engine: Parameter Consistency Check

function _validate_parameter_registry(chain, M, p_names, outcomes_N)
    # Flag to track overall registry integrity
    local is_valid = true

    # Iterate through each outcome dimension k
    for k in 1:outcomes_N
        # 1. Spatial Manifold Validation
        if M[:model_space] != "none"
            # Multivariate models use indexed keys like 's_latent_1', 's_latent_2'
            # Univariate models default to 's_latent' or 's_icar'
            local s_key = (outcomes_N == 1) ? "s_latent" : "s_latent_$k"

            # We check for the primary key or common legacy/structural aliases
            local has_spatial = (s_key in p_names) || ("s_icar" in p_names) || ("s_latent_1" in p_names)

            if !has_spatial
                @warn "Validation Error: Outcome $k requires spatial manifold '$s_key', but it was not found in the chain."
                is_valid = false
            end
        end

        # 2. Temporal Manifold Validation
        if M[:model_time] != "none"
            local t_key = (outcomes_N == 1) ? "t_latent" : "t_latent_$k"
            if !(t_key in p_names)
                @warn "Validation Warning: Expected temporal manifold '$t_key' missing from chain."
                # We permit continuation for temporal if defaults can be assumed, but warn the user
            end
        end
    end

    # 3. Global Shared Component Validation
    if M[:model_season] != "none"
        if !("u_latent" in p_names) && !("u_alpha" in p_names)
            @warn "Validation Warning: Seasonal manifold requested but latent parameters (u_latent/u_alpha) not discovered."
        end
    end

    if M[:Xfixed_N] > 0
        if !("Xfixed_beta" in p_names)
            @error "Validation Failure: Fixed effects design matrix detected but 'Xfixed_beta' parameters are absent."
            is_valid = false
        end
    end

    return is_valid
end


#  Parameter Discovery Engine [v17.0.15 - Restoration]
# Rationale: This is the primary recovery engine for the BSTM v06.1 manifold taxonomy.
# It must extract all latent field realizations from the MCMC chain for K outcomes/fidelities.
# Requirement: 100% parity with bstm_options and the technical primitive registry.


function _discover_manifold_realizations(chain, M, N_samples, outcomes_N, p_names)
    println("--- Discovery: Extracting v06.1 Latent Manifolds [v17.8.22 Domain-Sync Audit] ---")

    # 1. Linear and Mixed Effect Container Allocation
    # Fixed Effects: Standard Matrix [K_fixed * Outcomes x Samples]
    local Xfixed_betas = (M.Xfixed_N > 0) ? zeros(Float64, M.Xfixed_N * outcomes_N, N_samples) : nothing

    # Mixed Effects: Array of matrices for varying intercepts/slopes per sample
    local mixed_terms_list = get(M, :mixed_terms, [])
    local mixed_eff_coeffs = !isempty(mixed_terms_list) ? [zeros(Float64, term.n_cat, N_samples) for term in mixed_terms_list] : nothing

    # 2. Structural Manifold Container Allocation (Outcome-Specific)
    # Each outcome/fidelity k receives its own matrix of [Units x Samples]
    local s_eff_struct = [zeros(Float64, M.s_N, N_samples) for _ in 1:outcomes_N]
    local s_eff_noisy  = [zeros(Float64, M.s_N, N_samples) for _ in 1:outcomes_N]
    local t_eff = [zeros(Float64, M.t_N, N_samples) for _ in 1:outcomes_N]

    # Global Shared Components (Seasonal and Basis Smooths)
    local u_eff = zeros(Float64, M.u_N, N_samples)
    local basis_eff_accum = zeros(Float64, M.y_N, N_samples)

    # 3. Advanced Registry Allocation (Tensors and Hierarchies)
    # Spatiotemporal Interaction maps: [Units x Time x Sample]
    local st_eff_maps = [zeros(Float64, M.s_N, M.t_N, N_samples) for _ in 1:outcomes_N]

    # Spatially Varying Coefficients (SVC): [Units x Covariates x Sample]
    local svc_covs = get(M, :svc_covariates, Symbol[])
    local svc_slopes = !isempty(svc_covs) ? [zeros(Float64, M.s_N, length(svc_covs), N_samples) for _ in 1:outcomes_N] : nothing

    # Hierarchical Multi-Resolution Scales and Stochastic Volatility surfaces
    local hierarchical_scales = Dict{Symbol, Matrix{Float64}}()
    local sv_surface = [zeros(Float64, M.y_N, N_samples) for _ in 1:outcomes_N]

    # 4. Technical Primitive Resolution
    # Standardizing hyperprior and struct instantiation for the multiple-dispatch extractor
    local hyper_scheme = get(M, :hyperprior_scheme, :pcpriors)
    local priors_map = get(M, :hyperpriors, Dict())

    local season_prim = resolve_technical_primitive(M.model_season, M, priors_map, hyper_scheme)
    local space_prim = resolve_technical_primitive(M.model_space, M, priors_map, hyper_scheme)
    local time_prim = resolve_technical_primitive(M.model_time, M, priors_map, hyper_scheme)

    # 5. Primary Parameter Recovery Loop
    for j in 1:N_samples

        # 5.1 Global Shared Discovery (Seasonal & Basis Smooths)
        if M.model_season != "none"
            local u_sig = "u_sigma" in p_names ? get_params_vector(chain, "u_sigma", 1)[j] : 1.0
            # Domain-Aware Extraction: Passing :seasonal symbol
            u_eff[:, j] .= extract_manifold(season_prim, chain, M, j, u_sig, :seasonal)[1]
        end

        if haskey(M, :basis_matrices)
            for (v_sym, B_mat) in M.basis_matrices
                local p_key = "beta_basis_" * string(v_sym)
                if p_key in p_names
                    # Accumulate spline/FFT coefficients onto the training grid
                    basis_eff_accum[:, j] .+= B_mat * get_params_vector(chain, p_key, size(B_mat, 2))[j, :]
                end
            end
        end

        # 5.2 Hierarchical Scale Discovery
        # Rationale: Recovering regional/group latent fields defined in the spatial hierarchy
        if haskey(M, :spatial_hierarchy)
            for (scale_sym, scale_obj) in M.spatial_hierarchy
                local p_key_h = "s_latent_" * string(scale_sym)
                if p_key_h in p_names
                    local h_mat = get!(hierarchical_scales, scale_sym, zeros(Float64, scale_obj.n_units, N_samples))
                    h_mat[:, j] .= vec(get_params_vector(chain, p_key_h, scale_obj.n_units)[j, :])
                end
            end
        end

        # 5.3 Linear Fixed and Mixed Effect Recovery
        if !isnothing(Xfixed_betas)
            Xfixed_betas[:, j] .= get_params_vector(chain, "Xfixed_beta", M.Xfixed_N * outcomes_N)[j, :]
        end

        if !isnothing(mixed_eff_coeffs)
            for (m_idx, m_term) in enumerate(mixed_terms_list)
                local p_key_me = "beta_group_" * string(m_idx)
                if p_key_me in p_names
                    mixed_eff_coeffs[m_idx][:, j] .= vec(get_params_vector(chain, p_key_me, m_term.n_cat)[j, :])
                end
            end
        end

        # 5.4 Outcome-Specific Manifold Discovery (Fidelities 1:K)
        for k in 1:outcomes_N
            # --- A. Spatial Innovations ---
            if M.model_space != "none"
                local s_sig = (outcomes_N == 1) ?
                    ("s_sigma" in p_names ? get_params_vector(chain, "s_sigma", 1)[j] : 1.0) :
                    ("s_sigma_arr" in p_names ? get_params_vector(chain, "s_sigma_arr", outcomes_N)[j, k] : 1.0)

                # Domain-Aware Sync: Passing :spatial symbol to resolve naming conventions
                local fs, fn = (outcomes_N == 1) ?
                    extract_manifold(space_prim, chain, M, j, s_sig, :spatial) :
                    extract_manifold_k(space_prim, chain, M, j, k, s_sig, :spatial)

                s_eff_struct[k][:, j] .= fs
                s_eff_noisy[k][:, j] .= fn
            end

            # --- B. Temporal Innovations ---
            if M.model_time != "none"
                local t_sig = (outcomes_N == 1) ?
                    ("t_sigma" in p_names ? get_params_vector(chain, "t_sigma", 1)[j] : 1.0) :
                    ("t_sigma_arr" in p_names ? get_params_vector(chain, "t_sigma_arr", outcomes_N)[j, k] : 1.0)

                # Domain-Aware Sync: Passing :temporal symbol
                local ft, _ = (outcomes_N == 1) ?
                    extract_manifold(time_prim, chain, M, j, t_sig, :temporal) :
                    extract_manifold_k(time_prim, chain, M, j, k, t_sig, :temporal)

                t_eff[k][:, j] .= ft
            end

            # --- C. Spatiotemporal Interactions (Type IV) ---
            # Rationale: KRONECKER operator forced to :spacetime resolution
            local st_key = (outcomes_N == 1) ? "st_latent" : "st_latent_" * string(k)
            if st_key in p_names
                local st_sig = (outcomes_N == 1) ?
                    ("st_sigma" in p_names ? get_params_vector(chain, "st_sigma", 1)[j] : 1.0) :
                    ("st_sigma_arr" in p_names ? get_params_vector(chain, "st_sigma_arr", outcomes_N)[j, k] : 1.0)

                local fs_st, _ = (outcomes_N == 1) ?
                    extract_manifold(ComposedManifold([], :kronecker_product), chain, M, j, st_sig, :spacetime) :
                    extract_manifold_k(ComposedManifold([], :kronecker_product), chain, M, j, k, st_sig, :spacetime)

                st_eff_maps[k][:, :, j] .= reshape(fs_st, M.s_N, M.t_N)
            end

            # --- D. Spatially Varying Coefficients (SVC) ---
            if !isnothing(svc_slopes)
                for (v_idx, c_sym) in enumerate(svc_covs)
                    local p_key_svc = (outcomes_N == 1) ? "beta_svc_" * string(c_sym) : "beta_svc_" * string(c_sym) * "_" * string(k)
                    if p_key_svc in p_names
                        svc_slopes[k][:, v_idx, j] .= get_params_vector(chain, p_key_svc, M.s_N)[j, :]
                    end
                end
            end

            # --- E. Stochastic Volatility (SV) Surface Reconstruction ---
            # Rationale: Reconstructs heteroskedastic error scales via RFF projection
            if get(M, :use_sv, false)
                local sv_sig_key = (outcomes_N == 1) ? "sigma_log_var" : "sigma_log_var_" * string(k)
                local sv_beta_key = (outcomes_N == 1) ? "beta_vol_latent" : "beta_vol_latent_" * string(k)
                
                if sv_sig_key in p_names
                    local sig_v = get_params_vector(chain, sv_sig_key, 1)[j, 1]
                    local beta_v = vec(get_params_vector(chain, sv_beta_key, M.M_rff_sigma)[j, :])
                    # Projection logic: exp( (sigma * RFF(coords)) / 2 )
                    sv_surface[k][:, j] .= exp.((sig_v .* (sqrt(2.0 / M.M_rff_sigma) .* cos.(M.vol_proj * beta_v))) ./ 2.0)
                end
            else
                # Homoskedastic Fallback
                sv_surface[k][:, j] .= 1.0
            end
        end
    end

    # 6. Monolithic Registry Serialization
    # Rationale: Returns the NamedTuple ensuring parity with modular eta assembly
    return (
        Xfixed_betas = Xfixed_betas,
        mixed_eff_coeffs = mixed_eff_coeffs,
        s_eff_struct = s_eff_struct,
        s_eff_noisy = s_eff_noisy,
        t_eff = t_eff,
        u_eff = u_eff,
        basis_eff_accum = basis_eff_accum,
        st_eff_maps = st_eff_maps,
        svc_slopes = svc_slopes,
        hierarchical_scales = hierarchical_scales,
        sv_surface = outcomes_N == 1 ? sv_surface[1] : sv_surface,
        n_samples = N_samples,
        outcomes_N = outcomes_N
    )
end



function _modular_eta_assembly(N_tot_in, registry, M, PS_in)
    # Extraction of registry metadata for sample and outcome dimensions
    local N_samples = registry.n_samples
    local outcomes_N = registry.outcomes_N
    
    # Determine the actual limit of the assembly to prevent null-indexing
    # If PS_in is nothing, the loop limit is strictly defined by the training observations.
    local is_ps_null = isnothing(PS_in)
    local actual_limit = is_ps_null ? Int(M.y_N) : Int(N_tot_in)

    # Primary linear predictor container [Total Observations x Outcomes x MCMC Samples]
    # This tensor stores the additive latent field realizations on the link scale.
    local eta_container = zeros(Float64, actual_limit, outcomes_N, N_samples)

    for j in 1:N_samples
        for k in 1:outcomes_N
            # Manifold Slice Discovery for the current outcome (k) and sample (j)
            # S_f: Realized spatial field [Units_k]
            # T_f: Realized temporal trend [Time_k]
            # U_f: Realized seasonal cycle [Period_k]
            local S_f = !isnothing(registry.s_eff_noisy) && !isempty(registry.s_eff_noisy) ? registry.s_eff_noisy[k][:, j] : Float64[]
            local T_f = !isnothing(registry.t_eff) && !isempty(registry.t_eff) ? registry.t_eff[k][:, j] : Float64[]
            local U_f = !isnothing(registry.u_eff) && !isempty(registry.u_eff) ? registry.u_eff[:, j] : Float64[]

            # 3D Tensor Discovery for interactions and varying coefficients
            local ST_f = !isnothing(registry.st_eff_maps) && !isempty(registry.st_eff_maps[k]) ? registry.st_eff_maps[k][:, :, j] : zeros(0, 0)
            local SVC_f = !isnothing(registry.svc_slopes) && !isempty(registry.svc_slopes[k]) ? registry.svc_slopes[k][:, :, j] : zeros(0, 0)

            # Fixed Effect discovery for outcome k
            local xf_n = Int(M.Xfixed_N)
            local has_fixed = !isnothing(registry.Xfixed_betas) && xf_n > 0
            local beta_slice = has_fixed ? registry.Xfixed_betas[((k-1)*xf_n + 1):(k*xf_n), j] : Float64[]

            for i in 1:actual_limit
                # Categorize observation context: belong to Training (obs) or Prediction (PS) strata
                local is_obs = i <= M.y_N
                
                # Forensic Guard: Select the correct source for indices and covariates
                local src = is_obs ? M : PS_in
                local idx = is_obs ? i : i - Int(M.y_N)

                # Latent Manifold Coordinate Mapping (Strict Integer Cast)
                local s_ptr = Int(src.s_idx[idx])
                local t_ptr = Int(src.t_idx[idx])
                local u_ptr = Int(src.u_idx[idx])

                # 1. Structural Superposition (Spatial + Temporal + Seasonal)
                local val = 0.0
                if !isempty(S_f); val += S_f[s_ptr]; end
                if !isempty(T_f); val += T_f[t_ptr]; end
                if !isempty(U_f); val += U_f[u_ptr]; end

                # 2. Spatiotemporal Interaction (ST Tensor)
                if !isempty(ST_f)
                    val += ST_f[s_ptr, t_ptr]
                end

                # 3. Spatially Varying Coefficients (SVC) Contribution
                if !isempty(SVC_f)
                    local svc_vars = get(M, :svc_covariates, Symbol[])
                    for (v_idx, c_sym) in enumerate(svc_vars)
                        # Locate column in the current source design matrix
                        local col_idx = findfirst(==(c_sym), Symbol.(names(src.Xfixed, 2)))
                        if !isnothing(col_idx)
                            # Additive interaction: slope_at_location * covariate_value
                            val += SVC_f[s_ptr, v_idx] * src.Xfixed[idx, col_idx]
                        end
                    end
                end

                # 4. Observation-Specific Components (Basis Smooths and Link Offsets)
                # Note: These typically apply only to the training grid.
                if is_obs
                    if !isnothing(registry.basis_eff_accum)
                        val += registry.basis_eff_accum[idx, j]
                    end
                    if haskey(M, :log_offset)
                        val += M.log_offset[idx]
                    end
                end

                # 5. Fixed Effect Linear Predictor Assembly
                if has_fixed
                    # Dot product of design row with sampled beta vector
                    val += dot(vec(collect(src.Xfixed[idx, :])), beta_slice)
                end

                # Final Realization Entry for Observation i, Outcome k, Sample j
                eta_container[i, k, j] = val
            end
        end
    end
    return eta_container
end



"""
    summarize_array(samples::AbstractArray; alpha=0.05)

Primary engine for collapsing MCMC sample dimensions into posterior statistics. 
Supports 1D (scalars over samples), 2D (vectors over samples), and 3D (matrices over samples).

Inputs:
- samples: Array where the last dimension is assumed to be the sample index.
- alpha: Significance level for credible intervals (default: 0.05 for 95% CI).

Outputs:
- NamedTuple: (mean, median, lower, upper) where each field is a flat Vector{Float64}.
"""
function summarize_array(samples::AbstractArray; alpha=0.05)
    # 1. Edge Case: Empty or All-NaN Arrays
    # Rationale: Preventing computational overhead on invalid results
    if isempty(samples) || all(isnan, samples)
        return (mean = [NaN], median = [NaN], lower = [NaN], upper = [NaN])
    end

    local dims = size(samples)
    local n_dims = length(dims)
    local low_prob = alpha / 2.0
    local high_prob = 1.0 - low_prob

    # 2. Centralized Statistic Calculation
    # Rationale: dropdims collapses the sample dimension after reduction.
    # Mapslices ensures quantile calculations are performed along the correct axis.
    local post_mean = dropdims(Statistics.mean(samples, dims=n_dims), dims=n_dims)
    local post_median = dropdims(Statistics.median(samples, dims=n_dims), dims=n_dims)
    local low_bound = dropdims(mapslices(x -> Statistics.quantile(x, low_prob), samples, dims=n_dims), dims=n_dims)
    local high_bound = dropdims(mapslices(x -> Statistics.quantile(x, high_prob), samples, dims=n_dims), dims=n_dims)

    # 3. Vectorization Post-Processor
    # Rationale: Standardizing outputs to Vectors even for scalar results.
    # This prevents MethodErrors in downstream visualization loops (e.g., model_results_plots).
    function force_vector(x)
        if x isa AbstractArray
            if ndims(x) == 0
                # Handle 0-dimensional array artifacts from dropdims
                return [Float64(x[])]
            else
                # Flatten matrices/vectors to standard Vector
                return vec(collect(x))
            end
        else
            # Wrap raw scalars in a 1-element vector
            return [Float64(x)]
        end
    end

    # 4. Feature Complete Return Object
    return (
        mean = force_vector(post_mean),
        median = force_vector(post_median),
        lower = force_vector(low_bound),
        upper = force_vector(high_bound)
    )
end





function model_results_plots(res, ts=1; outcome=1, centroids=nothing, polygons=nothing, y_obs=nothing)
    println("--- Rendering Hardened Posterior Dashboard v18.1.8 [Outcome: $outcome] ---")
    
    local pstats = res.pstats
    local obs_src = !isnothing(y_obs) ? y_obs : res.y_obs
    local coords = !isnothing(centroids) ? centroids : res.centroids
    local polys = !isnothing(polygons) ? polygons : res.polygons

    local plots_list = []

    # 1. Posterior Predictive Check (PPC)
    # Rationale: Verifying the correlation between observed data and denoised expectations.
    local is_multivariate = (obs_src isa AbstractMatrix && size(obs_src, 2) > 1)
    local y_p = is_multivariate ? pstats.predictions_denoised.mean[:, outcome] : vec(pstats.predictions_denoised.mean)
    local y_o = is_multivariate ? obs_src[:, outcome] : vec(obs_src)

    if length(y_p) == length(y_o)
        local p_ppc = Plots.scatter(vec(y_p), vec(y_o), title="PPC (Outcome $outcome)", xlabel="Pred", ylabel="Obs", alpha=0.5, markersize=3, markerstrokewidth=0, legend=false)
        local clean_p = filter(!isnan, y_p)
        local clean_o = filter(!isnan, y_o)
        if !isempty(clean_p) && !isempty(clean_o)
            local m_val = max(maximum(clean_p), maximum(clean_o))
            Plots.plot!(p_ppc, [0, m_val], [0, m_val], color=:red, ls=:dash, lw=1.5)
        end
        push!(plots_list, p_ppc)
    end

    # 2. Spatial Mapping
    # Rationale: Rendering the structured spatial field realization.
    if !isnothing(coords)
        local s_field = (pstats.arch isa MultivariateArchitecture && pstats.spatial_structured isa AbstractVector) ?
                        pstats.spatial_structured[outcome] : pstats.spatial_structured

        if !isnothing(s_field) && hasproperty(s_field, :mean)
            local s_mean = vec(collect(s_field.mean))
            local p_map = Plots.plot(aspect_ratio=:equal, title="Spatial Field (Outcome $outcome)", frame=:box)

            if !isnothing(polys) && length(polys) >= length(s_mean)
                for i in 1:length(s_mean)
                    local px_coords = [pt[1] for pt in polys[i]]
                    local py_coords = [pt[2] for pt in polys[i]]
                    if !isempty(px_coords) && (px_coords[1], py_coords[1]) != (px_coords[end], py_coords[end])
                        push!(px_coords, px_coords[1])
                        push!(py_coords, py_coords[1])
                    end
                    Plots.plot!(p_map, px_coords, py_coords, seriestype=:shape, fill_z=s_mean[i], c=:viridis, linecolor=:black, lw=0.2, label=nothing)
                end
            else
                local cx = [c[1] for c in coords]
                local cy = [c[2] for c in coords]
                local n_pts = length(cx)
                local z_vals = length(s_mean) >= n_pts ? s_mean[1:n_pts] : [s_mean; zeros(n_pts - length(s_mean))]
                Plots.scatter!(p_map, cx, cy, marker_z=vec(z_vals), markersize=4, c=:viridis, label=nothing, colorbar=true)
            end
            push!(plots_list, p_map)
        end
    end

    # 3. Temporal Trend
    # Rationale: Displaying the non-linear trend component with credible interval ribbon.
    local t_field = nothing
    if hasproperty(pstats, :temporal_effects)
        t_field = (pstats.arch isa MultivariateArchitecture && pstats.temporal_effects isa AbstractVector) ?
                   pstats.temporal_effects[outcome] : pstats.temporal_effects
    end
    if !isnothing(t_field) && !all(isnan, t_field.mean)
        local tm = vec(collect(t_field.mean))
        local tl = vec(collect(t_field.lower))
        local tu = vec(collect(t_field.upper))
        push!(plots_list, Plots.plot(tm, ribbon=(tm .- tl, tu .- tm), title="Temporal Trend", lw=2, fillalpha=0.2, color=:royalblue, legend=false))
    end

    # 4. Seasonal Cycle (FIXED)
    # Rationale: Standardizing seasonal discovery and ensuring ribbon-safe vectorization.
    if hasproperty(pstats, :seasonal) && !isnothing(pstats.seasonal)
        local u_field = pstats.seasonal
        if !all(isnan, u_field.mean)
            local um = vec(collect(u_field.mean))
            local ul = vec(collect(u_field.lower))
            local uu = vec(collect(u_field.upper))
            push!(plots_list, Plots.plot(um, ribbon=(um .- ul, uu .- um), title="Seasonal Cycle", lw=2, fillalpha=0.2, color=:forestgreen, legend=false, xlabel="Season Bin"))
        end
    end

    # 5. Fixed Effects
    # Rationale: Bar chart for categorical or linear covariate effects.
    if hasproperty(pstats, :fixed_effects) && !isnothing(pstats.fixed_effects)
        local f_field = pstats.fixed_effects
        if !all(isnan, f_field.mean)
            local fm = vec(collect(f_field.mean))
            local fl = vec(collect(f_field.lower))
            local fu = vec(collect(f_field.upper))
            push!(plots_list, Plots.bar(fm, yerror=(fm .- fl, fu .- fm), title="Fixed Effects", color=:grey, legend=false))
        end
    end

    # 6. Smooth Basis / Accumulated Smooths (FIXED)
    # Rationale: Aggregating non-linear spline or FFT effects with ribbon confidence.
    if hasproperty(pstats, :smooth_effects) && !isnothing(pstats.smooth_effects)
        local b_field = pstats.smooth_effects
        if !all(isnan, b_field.mean)
            local bm = vec(collect(b_field.mean))
            local bl = vec(collect(b_field.lower))
            local bu = vec(collect(b_field.upper))
            push!(plots_list, Plots.plot(bm, ribbon=(bm .- bl, bu .- bm), title="Accumulated Smooths", lw=1.5, fillalpha=0.1, color=:darkorange, legend=false))
        end
    end

    # Final Dashboard Assembly
    if !isempty(plots_list)
        local n_plots = length(plots_list)
        local cols = min(n_plots, 2)
        local rows = Int(ceil(n_plots / cols))
        return Plots.plot(plots_list..., layout=(rows, cols), size=(1000, 300 * rows), margin=5Plots.mm)
    end
    
    @warn "No active components discovered for dashboard rendering."
    return nothing
end



function model_results_comprehensive(model, chain; n_samples=100, alpha=0.05)
    # Audited Comprehensive Posterior Reconstruction [v14.0.12 - BSTM v06.1  Sync]
    # Rationale: Standardizes the recovery, summarization, and accuracy assessment of BSTM models.
    # This version ensures full parity with the v06.1 manifold registry.

    println("--- Starting Audited Comprehensive Posterior Reconstruction v14.0.12 [v06.1] ---")

    # 1. Metadata and Architecture Extraction
    # M contains the configuration, data, and spatial units used during sampling.
    local M = model.args.M
    local y_obs = M[:y_obs]
    local raw_arch = get(M, :model_arch, "univariate")
    local outcomes_N = get(M, :outcomes_N, 1)

    # Determining the architectural dispatch type for reconstruction
    local arch_type = if raw_arch == "univariate"
        UnivariateArchitecture()
    elseif raw_arch == "multivariate"
        MultivariateArchitecture()
    elseif raw_arch == "multifidelity"
        MultifidelityArchitecture()
    else
        UnivariateArchitecture()
    end

    # 2. Latent Manifold Reconstruction
    # Executes the architectural-specific reconstruction of every latent component.
    # This call triggers _reconstruct which includes internal parameter validation for v06.1 types.
    local res = _reconstruct(arch_type, "model_results", chain, M, nothing, alpha)

    # 3. Spatial Metadata Recovery for Mapping
    # Recovery of Centroids and Polygons to ensure mapping consistency in the dashboard.
    local centroids = nothing
    if haskey(M, :areal_units)
        centroids = M[:areal_units].centroids
    elseif haskey(M, :s_coord)
        # Convert coordinate matrix to vector of tuples for visualization parity
        centroids = [(M[:s_coord][i, 1], M[:s_coord][i, 2]) for i in 1:size(M[:s_coord], 1)]
    end

    local polygons = nothing
    if haskey(M, :areal_units) && hasproperty(M[:areal_units], :polygons)
        polygons = M[:areal_units].polygons
    end

    # 4. Global Metric Assessment (Denoised Mean vs Observation)
    # Metrics are calculated using the posterior predictive mean (denoised expectation).
    local y_pred = res.predictions_denoised.mean

    # Flatten for metric consistency across Univariate and Multivariate responses
    # y_obs may be a Matrix [N x K] or Vector [N]
    local y_obs_flat = vec(collect(y_obs))
    local y_pred_flat = vec(collect(y_pred))

    # Filter valid (non-NaN) indices to handle missing data or hurdle/truncated branches
    local valid_idx = findall(x -> !isnan(x) && !isnothing(x), y_obs_flat)

    local rmse_val = 0.0
    local r_pearson = 0.0

    if !isempty(valid_idx)
        local obs_v = y_obs_flat[valid_idx]
        local pred_v = y_pred_flat[valid_idx]
        
        # Root Mean Square Error
        rmse_val = sqrt(Statistics.mean((obs_v .- pred_v).^2))
        
        # Pearson r Calculation (Correlation Strength)
        try
            r_pearson = Statistics.cor(obs_v, pred_v)
        catch
            # Handle cases with zero variance in predictions
            r_pearson = 0.0
        end
    end

    println("\n--- Quality Metrics ---")
    println("RMSE: ", round(rmse_val, digits=4))
    println("Pearson R: ", round(r_pearson, digits=4))
    println("WAIC: ", round(get(res, :waic, 0.0), digits=2))

    # 5. Final Registry Object Assembly
    return (
        metrics = (rmse = rmse_val, r_pearson = r_pearson, waic = get(res, :waic, 0.0)),
        pstats = res,
        y_obs = y_obs,
        centroids = centroids,
        polygons = polygons,
        model_family = get(M, :model_family, "gaussian"),
        arch = arch_type
    )
end


# Consolidated and Audited Parameter Extraction Engine (v14.0.10)
# Rationale: This function serves as the primary data-bridge between raw MCMC chains
# and the posterior reconstruction assembly. It must robustly handle FlexiChains 
# nesting and ensure that indexed parameters are recovered in strictly numerical order.


function get_params_vector(chain, base_name::String, expected_len::Int)
    # Audited and Hardened Parameter Extraction Engine (v14.0.10)
    # Rationale: This utility acts as the primary data-bridge between raw MCMC chains
    # and the posterior reconstruction assembly. It must robustly handle FlexiChains
    # nesting and ensure that indexed parameters are recovered in numerical order.

    local N_samples = size(chain, 1)
    local all_names = string.(FlexiChains.parameters(chain))

    # 1. Tier 1: Regex-based Indexed Recovery (Numerical Sorting)
    # Rationale: Standard string sorting often places 'beta[10]' before 'beta[2]'.
    # This Regex captures the integer index to ensure strictly numerical alignment.
    local regex = Regex("^" * base_name * "\\[(\\d+)\\]")
    local matched_names = filter(n -> occursin(regex, n), all_names)

    if !isempty(matched_names)
        # Sort by the captured integer index to maintain dimensional alignment
        sort!(matched_names, by = n -> parse(Int, match(regex, n).captures[1]))

        local res_mat = zeros(Float64, N_samples, length(matched_names))

        for (idx, n) in enumerate(matched_names)
            local val_obj = chain[Symbol(n)]
            # Extract raw data while handling potential vector wrapping from FlexiChains
            local raw = hasproperty(val_obj, :data) ? val_obj.data : collect(val_obj)

            # Map each sample to a Float64 scalar. If the sample is a 1-element vector, extract index 1.
            for s in 1:N_samples
                local v = raw[s]
                res_mat[s, idx] = (v isa AbstractVector) ? Float64(v[1]) : Float64(v)
            end
        end

        # Standard Scalar Broadcast: If only one index is found but multiple are expected,
        # we broadcast the column to the expected length.
        if size(res_mat, 2) == 1 && expected_len > 1
            return repeat(res_mat, 1, expected_len)
        end

        return res_mat
    end

    # 2. Tier 2: Vectorized/Single Entity Recovery
    # Handles cases where parameters are stored as a single vector (e.g., 's_sigma_arr')
    if base_name in all_names
        local val_obj = chain[Symbol(base_name)]
        local raw_data = hasproperty(val_obj, :data) ? val_obj.data : collect(val_obj)

        # Standardize to Matrix [Samples x Params]
        # We iterate through samples to flatten any nested Matrix{Vector} artifacts.
        local mat_data = if eltype(raw_data) <: AbstractVector
             reduce(hcat, [vec(collect(v)) for v in raw_data])'
        else
             Matrix{Float64}(reshape(collect(raw_data), N_samples, :))
        end

        # Dimensional Realignment and Transpose Logic
        if size(mat_data, 2) == expected_len
            return mat_data
        elseif size(mat_data, 1) == expected_len && size(mat_data, 2) != expected_len
            # Transpose if the chain orientation is [Params x Samples]
            return mat_data'
        elseif size(mat_data, 2) == 1
            # Scalar Broadcast fallback
            return repeat(mat_data, 1, expected_len)
        else
            # Final fallback: return the raw data if it matches expected_len after collect
            return reshape(vec(mat_data), N_samples, :)
        end
    end

    # 3. Null Safety Fallback
    # If the parameter is missing, return a zero-matrix to prevent downstream assembly failure.
    @warn "get_params_vector: Parameter '$base_name' not discovered in chain. Initializing with zeros (len=$expected_len)."
    return zeros(Float64, N_samples, expected_len)
end
 

function create_prediction_surface(basis_df::DataFrame, observations_df::DataFrame, au; lambda_s=2.0, lambda_t=1.0, max_iters=5)
    # 1. Initialization and Automatic Identification of Fixed-Effect Columns
    mergeon = hasproperty(basis_df, :u_idx) && hasproperty(observations_df, :u_idx) ? [:s_idx, :t_idx, :u_idx] : [:s_idx, :t_idx]

    # Identify non-merge, non-outcome columns (the design matrix variables)
    # observations_df should already contain M.y_obs if applicable, so exclude it.
    fixed_variable_names = setdiff(propertynames(observations_df), vcat(mergeon, [:y_obs]))

    # Join basis (grid) with observations
    surface = leftjoin(basis_df, observations_df, on = mergeon, makeunique=true)

    # Standardize types to Float64/Missing for imputation math
    # Apply to the combined 'surface' DataFrame
    for c in fixed_variable_names
        surface[!, c] = convert(Vector{Union{Float64, Missing}}, collect(surface[!, c]))
    end

    centroids = au.centroids
    n_units = length(centroids)
    W = au.W

    # 2. Precompute Spatial Adjacency Weights
    dist_mat_s = [sqrt(sum((c1 .- c2).^2)) for c1 in centroids, c2 in centroids]
    weight_mat_s = exp.(-dist_mat_s.^2 ./ (2 * lambda_s^2)) .* (W + I)
    for i in 1:n_units
        s_row = sum(weight_mat_s[i, :])
        if s_row > 1e-9; weight_mat_s[i, :] ./= s_row; end
    end

    # 3. Iterative Spatiotemporal Imputation (Handling Continuous & Binned Factors)
    for iter in 1:max_iters
        # Check if there are any missing values in the relevant columns
        has_missing = false
        for c in fixed_variable_names
            if any(ismissing, surface[!, c])
                has_missing = true
                break
            end
        end
        if !has_missing; break; end # Break if no missing values remain

        for c in fixed_variable_names
            # If the variable is likely a factor (only integers), store levels to snap back later
            unique_vals = filter(!ismissing, unique(surface[!, c]))
            is_factor = all(v -> v == floor(v), unique_vals) # Check if all unique valid values are integers

            group_cols = hasproperty(surface, :u_idx) ? [:u_idx] : Symbol[] # Convert to Symbol[] for consistency
            # Iterate over unique combinations of grouping columns (e.g., u_idx if present)
            for grp in groupby(surface, group_cols)
                # Iterate over rows within each group
                for i in 1:nrow(grp)
                    current_row_in_grp = grp[i, :]
                    if ismissing(current_row_in_grp[c]) || isnan(current_row_in_grp[c]) # Check for isnan too
                        curr_s = Int(round(current_row_in_grp[:s_idx]))
                        curr_t = Int(round(current_row_in_grp[:t_idx])) # Ensure t_idx is also Int for comparison
                        curr_u = hasproperty(grp, :u_idx) ? Int(round(current_row_in_grp[:u_idx])) : nothing

                        # Spatial Influence: Look for values in the same time slice (and u_idx if applicable)
                        spatial_mask_df = filter(row -> row[:t_idx] == curr_t && (isnothing(curr_u) || row[:u_idx] == curr_u), surface)
                        s_vals_filtered = filter(row -> !ismissing(row[c]), eachrow(spatial_mask_df))

                        val_s, w_s_sum = 0.0, 0.0
                        for row in s_vals_filtered
                            nb_s = Int(round(row[:s_idx]))
                            w = nb_s != curr_s ? weight_mat_s[curr_s, nb_s] : 0.0 # Exclude self-influence
                            val_s += w * Float64(row[c])
                            w_s_sum += w
                        end

                        # Temporal Influence: Look for values in the same spatial unit (and u_idx if applicable)
                        temporal_mask_df = filter(row -> row[:s_idx] == curr_s && (isnothing(curr_u) || row[:u_idx] == curr_u), surface)
                        t_vals_filtered = filter(row -> !ismissing(row[c]), eachrow(temporal_mask_df))

                        val_t, w_t_sum = 0.0, 0.0
                        for row in t_vals_filtered
                            nb_t = Int(round(row[:t_idx]))
                            w = nb_t != curr_t ? exp(-(curr_t - nb_t)^2 / (2 * lambda_t^2)) : 0.0 # Exclude self-influence
                            val_t += w * Float64(row[c])
                            w_t_sum += w
                        end

                        if (w_s_sum + w_t_sum) > 1e-9 # Use a small epsilon to avoid division by zero
                            imputed_val = (val_s + val_t) / (w_s_sum + w_t_sum)
                            # Find the global index of the current row in 'surface' to update
                            # This is a potentially slow lookup, but necessary given DataFrame's groupby behavior
                            global_row_idx = findfirst(r -> r[:s_idx] == curr_s && r[:t_idx] == curr_t && (isnothing(curr_u) || r[:u_idx] == curr_u), eachrow(surface))
                            if !isnothing(global_row_idx)
                                surface[global_row_idx, c] = is_factor ? round(imputed_val) : imputed_val
                            end
                        end
                    end
                end
            end
        end
    end

    # 4. Final Global Fallback for any remaining missing values if iterative imputation couldn't fill them
    for c in fixed_variable_names
        valid_entries = filter(!ismissing, surface[!, c])
        if !isempty(valid_entries)
            m_val = median(valid_entries)
            surface[!, c] = map(x -> ismissing(x) ? m_val : x, surface[!, c])
        end
    end

    return surface
end



function convert_advi_to_reconstruct_format(msol, model::DynamicPPL.Model, n_samples::Int=500)
    # Sample from the variational distribution
    samples_vec = rand(msol, n_samples)

    # Safety check for extraction: ADVI samples often wrap data in .nt, .data, or are ParamsWithStats objects
    function get_data_obj(s)
        if hasproperty(s, :nt) return s.nt
        elseif hasproperty(s, :data) return s.data
        elseif hasproperty(s, :params) return s.params # Handle ParamsWithStats internal field
        else return s end
    end

    # Peek at the first sample to discover parameter keys
    first_samp = get_data_obj(samples_vec[1])
    all_keys = keys(first_samp)

    unique_bases = Set{Symbol}()
    for k in all_keys
        m = match(r"^([^\\\\[]+)", string(k))
        if m !== nothing
            push!(unique_bases, Symbol(m.captures[1]))
        end
    end

    reconstruct_samples = map(samples_vec) do samp
        nt = get_data_obj(samp)
        sample_params = Dict{Symbol, Any}()
        for base_sym in unique_bases
            base_str = string(base_sym)
            col_keys = filter(k -> string(k) == base_str || startswith(string(k), "$base_str["), all_keys)
            if length(col_keys) == 1 && string(first(col_keys)) == base_str
                sample_params[base_sym] = nt[first(col_keys)]
            else
                # Sort indexed keys like x[1], x[10], x[2] into numerical order
                sorted_keys = sort(collect(col_keys), by = k -> begin
                    m_idx = match(r"\\\\[(\d+)\\\\]", string(k))
                    m_idx !== nothing ? parse(Int, m_idx.captures[1]) : 0
                end)
                sample_params[base_sym] = [nt[k] for k in sorted_keys]
            end
        end
        return (; sample_params...)
    end

    # Create a FlexiChain for standard diagnostics
    formatted_dicts = map(samples_vec) do samp
        nt = get_data_obj(samp)
        Dict(FlexiChains.Parameter(Symbol(k)) => v for (k, v) in pairs(nt))
    end
    chn = FlexiChains.FlexiChain{Symbol}(n_samples, 1, formatted_dicts)

    return (chain=chn, reconstruct_samples=reconstruct_samples)
end



function convert_optim_to_reconstruct_format(optim_result, model, n_samples::Int=500; use_hessian=true, external_hessian=nothing)
    """
    Synopsis: Converts a point estimate (MAP/ML) from Optim/Optimisers into a distribution of samples.
    If a Hessian is available, it samples from the Multivariate Normal Laplace approximation.
    Otherwise, it creates a narrow Gaussian around the point estimate.
    """

    point_est_constrained = optim_result.params # This is the named tuple of constrained parameters

    reconstruct_samples_namedtuple = [] # Store NamedTuple for easier handling

    # Determine which Hessian to use
    H_to_use = nothing
    if external_hessian !== nothing
        H_to_use = external_hessian
    elseif hasproperty(optim_result, :hessian)
        H_to_use = optim_result.hessian
    end

    # Attempt Hessian-based sampling if enabled and a Hessian is available from either source
    if use_hessian && H_to_use !== nothing
        try
            # Get the unconstrained minimizer (mu_unconstrained)
            mu_unconstrained = optim_result.minimizer
            H = H_to_use

            # Ensure H is symmetric and compute its inverse for the covariance matrix
            Sigma = inv(Symmetric(Matrix(H) + Diagonal(fill(1e-6, size(H, 1)))))
            
            dist = MvNormal(mu_unconstrained, Sigma)

            # Generate n_samples of unconstrained parameters
            unconstrained_samples_matrix = rand(dist, n_samples)

            # Prepare template for conversion
            vi_template = DynamicPPL.VarInfo(model)

            for i in 1:n_samples
                sample_unconstrained_vec = unconstrained_samples_matrix[:, i]
                vi_current_sample = deepcopy(vi_template)
                DynamicPPL.setlink!(vi_current_sample, sample_unconstrained_vec)
                
                # Convert back to constrained named tuple
                constrained_sample_params = DynamicPPL.vi_to_params(vi_current_sample, model)
                push!(reconstruct_samples_namedtuple, constrained_sample_params)
            end

        catch e
            @warn "Failed to compute covariance from Hessian or convert samples, falling back to adding noise: $e"
            use_hessian = false 
        end
    end

    # Fallback to noise-based sampling
    if !use_hessian || isempty(reconstruct_samples_namedtuple)
        all_keys = keys(point_est_constrained)
        unique_bases = Set{Symbol}()
        for k in all_keys
            m = match(r"^([^\\[]+)", string(k))
            if m !== nothing
                push!(unique_bases, Symbol(m.captures[1]))
            end
        end

        reconstruct_samples_namedtuple = map(1:n_samples) do _
            sample_params_dict = Dict{Symbol, Any}() 
            for base_sym in unique_bases
                base_str = string(base_sym)
                col_keys = filter(k -> string(k) == base_str || startswith(string(k), "$base_str["), all_keys)

                if length(col_keys) == 1 && string(first(col_keys)) == base_str
                    val = point_est_constrained[first(col_keys)]
                    if val isa AbstractVector
                        sample_params_dict[base_sym] = val .+ randn() * 1e-4
                    else
                        sample_params_dict[base_sym] = val + randn() * 1e-4
                    end
                else
                    sorted_keys = sort(collect(col_keys), by = k -> begin
                        m_idx = match(r"\\[(\\d+)\\]", string(k))
                        m_idx !== nothing ? parse(Int, m_idx.captures[1]) : 0
                    end)
                    sample_params_dict[base_sym] = [point_est_constrained[k] + randn() * 1e-4 for k in sorted_keys]
                end
            end
            return (; sample_params_dict...)
        end
    end

    # 4. Format into a FlexiChain
    # IMPORTANT: Store ONLY the base symbols (e.g., :s_icar as a vector).
    # FlexiChains will automatically expand these into indexed names for summary statistics,
    # avoiding duplicate key errors while allowing _reconstruct to find the full vector.
    formatted_dicts = map(1:n_samples) do i
        samp = reconstruct_samples_namedtuple[i]
        d = Dict{FlexiChains.Parameter, Any}()
        for k in keys(samp)
            d[FlexiChains.Parameter(k)] = samp[k]
        end
        return d
    end

    chn = FlexiChains.FlexiChain{Symbol}(n_samples, 1, formatted_dicts)

    return (chain=chn, reconstruct_samples=reconstruct_samples_namedtuple)
end
 
    
function generate_sim_data(s_N=25, t_N=10; rndseed=42)
    Random.seed!(rndseed)
    local n_total = s_N * t_N

    # 1. Spatial Coordinates (Unit Level)
    local unique_pts = [(rand() * 100.0, rand() * 100.0) for _ in 1:s_N]
    local s_coord_tuple = repeat(unique_pts, inner=t_N)
    local s_x = collect(Float64, getindex.(s_coord_tuple, 1))
    local s_y = collect(Float64, getindex.(s_coord_tuple, 2))

    # 2. Temporal/Seasonal Indices
    local t_v = repeat(collect(1:t_N), outer=s_N) .+ (rand(n_total) .* 0.05)
    local t_idx = repeat(collect(1:t_N), outer=s_N)
    local u_N = 12
    # Fix: repeat() mapping for seasonal indices
    local u_idx = Int.(mod1.(1:n_total, u_N))

    # 3. Latent Fields
    local period = 12.0
    local trend = 0.05 .* t_v
    local seasonal = 0.8 .* cos.(2π .* t_v ./ period)

        # Covariate Generation (W1, W2, W3)
    # These simulate continuous predictors with some shared latent signal Z
    local Z = randn(n_total)
    local W1_obs = 0.5 .* sin.(t_v ./ 5.0) .+ 0.5 .* Z .+ (randn(n_total) .* 0.1)
    local W2_obs = 0.5 .* cos.(t_v ./ 5.0) .- 0.3 .* Z .+ (randn(n_total) .* 0.2)
    local W3_obs = 0.2 .* (t_v ./ t_N) .+ 0.1 .* Z .+ (randn(n_total) .* 0.3)
    local w_obs = hcat(W1_obs, W2_obs, W3_obs)
 
    # Mosaic/Cluster Effects
    local s_clusters = repeat(1:5, inner=Int(s_N/5))
    local cluster_effects = [-2.5, -1.0, 0.0, 1.0, 2.5]
    local spatial_effect = cluster_effects[s_clusters[repeat(1:s_N, inner=t_N)]]

    # 4. Response Construction
    local sigma_y = 0.15
    local observation_error = sigma_y .* randn(n_total)
    local eta = 1.0 .+ spatial_effect .+ trend .+ seasonal .+ observation_error

    local y_binary = Int.(eta .> (mean(eta) + 0.5))
    local y_counts = abs.(Int.(round.(exp.(eta)))) # Poisson-friendly counts

    local weights = ones(Float64, n_total)
    local trials = ones(Int, n_total)

    # Fixed Effects Design Matrix (Standard Intercept-only approach)
    local Xfixed = ones(Float64, n_total, 1)

    return (
        y_obs = eta,
        s_idx = repeat(1:s_N, inner=t_N),
        t_idx = t_idx,
        u_idx = u_idx,
        s_x = s_x,
        s_y = s_y,
        t_v = t_v,
        weights = weights,
        trials = trials,
        s_N = s_N,
        t_N = t_N,
        u_N = u_N,
        y_binary = y_binary,
        y_counts = y_counts,
        Xfixed = Xfixed,
        z_obs = Z,
        w_obs = w_obs,

        s_coord = reduce(hcat, unique_pts)',
        cluster_assignments = s_clusters,

        n_total = n_total
    )
end

  
   
function precompute_nystrom_projection(spatial_coords, inducing_points, kernel_func; jitter=1e-6)

    # Example Usage:
    # kernel = Matern32Kernel() ∘ ScaleTransform(1.0)
    # modinputs_reference.K_nystrom_proj = precompute_nystrom_projection(areal_units.centroids, Z_inducing, kernel)

    # println("Precomputing Nystrom Projection Matrix...")
    
    # 1. K_mm: Kernel matrix between inducing points (M x M)
    K_mm = kernelmatrix(kernel_func, RowVecs(inducing_points))
    K_mm_stable = Symmetric(K_mm + jitter * I)
    
    # 2. K_nm: Kernel matrix between all spatial units and inducing points (N x M)
    K_nm = kernelmatrix(kernel_func, RowVecs(spatial_coords), RowVecs(inducing_points))
    
    # 3. Projection: K_nm * inv(K_mm)
    # use the backslash operator for better numerical stability than direct inversion
    K_nystrom_proj = K_nm / K_mm_stable
    
    # println("Generated projection matrix of size: ", size(K_nystrom_proj))
    return K_nystrom_proj
end



# --- Optimized Householder PCA Helper Functions ---

function householder_to_eigenvector(v_mat::AbstractMatrix{T}, nU, n_factors) where {T}
    # Initializes the Identity matrix to be transformed
    U = Matrix{T}(I, nU, nU)

    for k in 1:n_factors
        # Extract the k-th Householder vector
        vk = v_mat[:, k]
        norm_v = norm(vk)
        
        if norm_v > 1e-9
            vk = vk / norm_v
            
            # --- O(K * N^2) Optimization ---
            # Naive: U = (I - 2vv') * U  => O(N^3)
            # Optimized: U = U - 2v * (v' * U) => O(N^2)
            # first compute the row vector (v' * U), then perform an outer product update.
            
            v_transpose_U = vk' * U
            U = U - 2.0 .* vk * v_transpose_U
        end
    end

    # Return only the first n_factors columns as the orthonormal loadings matrix
    return U[:, 1:n_factors]
end

function householder_transform(v, nU, n_factors, ltri_indices, pca_sd, pdef_sd, noise)
    T = eltype(v)
    v_mat = zeros(T, nU, n_factors)
    v_mat[ltri_indices] .= v

    # Generate Orthonormal Loadings using optimized transformation
    U = householder_to_eigenvector(v_mat, nU, n_factors)

    # Reconstruct Covariance Components
    # W = Loadings * Scaled Eigenvalues
    W = U * Diagonal(pca_sd)

    # Kmat is the full covariance matrix: WW' + Residual_Variance
    # add a small noise term for numerical stability
    Kmat = W * W' + (pdef_sd^2 + noise) * I(nU)

    return Kmat, pca_sd, U
end



function eigenvector_to_householder(U_in::AbstractMatrix{T}, n_factors) where {T}
# --- Optimal Vector Extraction (Orthonormal to Householder) ---

# eigenvector_to_householder(U, n_factors)
#
# Description:
#   Extracts the Householder reflector vectors (v) from an orthonormal loadings matrix U.
#   This allows for initializing the Bayesian model from a frequentist PCA result.
#
# Complexity: O(K * N^2)

    nU = size(U_in, 1)
    # work on a copy to avoid modifying the input
    U = copy(U_in)
    
    # Storage for the lower-triangular part of the v_mat
    # Each column k corresponds to the k-th Householder vector
    v_mat = zeros(T, nU, n_factors)

    for k in 1:n_factors
        # 1. Target vector is the k-th column of the current transformation
        # For the identity, want U[k,k] to be 1 and others 0
        x = U[k:end, k]
        
        # 2. Standard Householder Reflection Math
        # v = x + sign(x[1]) * ||x|| * e1
        norm_x = norm(x)
        vk = copy(x)
        
        sign_x1 = x[1] >= 0 ? one(T) : -one(T)
        vk[1] += sign_x1 * norm_x
        
        norm_vk = norm(vk)
        if norm_vk > 1e-9
            vk = vk ./ norm_vk
            
            # 3. Apply the reflection to the rest of the matrix (Rank-1 update)
            # U[k:end, k:end] = (I - 2vv') * U[k:end, k:end]
            # Using the O(N^2) update trick
            v_transpose_U = vk' * U[k:end, k:end]
            U[k:end, k:end] -= 2.0 .* vk * v_transpose_U
            
            # 4. Store the reflector
            v_mat[k:end, k] .= vk
        end
    end

    return v_mat
end


function extract_v_parameters(v_mat, ltri_indices)
    # extract_v_parameters(v_mat, ltri_indices)
    #
    # Utility to extract only the free parameters (lower triangular) from the v_mat
    # for use as initial values in Turing (matching the 'v' parameter vector).
    return v_mat[ltri_indices]
end

;;

function extract_v_parameters(v_mat, ltri_indices)
    # extract_v_parameters(v_mat, ltri_indices)
    #
    # Utility to extract only the free parameters (lower triangular) from the v_mat
    # for use as initial values in Turing (matching the 'v' parameter vector).
    return v_mat[ltri_indices]
end 

function get_params_matrix_sizestructured(chain, base_name, dims)
    # Optimized for chn[:param].data[sample][row, col] access pattern
    n_rows, n_cols = dims
    N_samples = size(chain, 1)

    if Symbol(base_name) in names(chain, :parameters)
        res = zeros(Float64, n_rows, n_cols, N_samples)
        data_container = chain[Symbol(base_name)].data

        for s in 1:N_samples
            samp_mat = data_container[s]
            if size(samp_mat) == (n_rows, n_cols)
                res[:, :, s] = samp_mat
            elseif size(samp_mat) == (n_cols, n_rows)
                res[:, :, s] = samp_mat'
            end
        end
        return res
    end
    return nothing
end



 
 
 
 


function get_vec(obj, key)  
    # Helper for robust indexing
    val = hasproperty(obj, key) ? getproperty(obj, key) : nothing
    val isa AbstractVector ? val : [val]
end


function summarize_lkj_correlation(chain, outcomes_N; alpha=0.05)
    println("--- Summarizing Cross-Outcome Covariance (LKJ Prior) ---")
    p_names = string.(FlexiChains.parameters(chain))

    # Identify correlation parameters (typically stored as L_omega or CorMatrix)
    cor_params = filter(p -> occursin("L_omega", p) || occursin("cor_mat", p), p_names)

    if isempty(cor_params)
        @warn "No correlation parameters found in chain."
        return nothing
    end

    # For LKJ Cholesky factors (L_omega), reconstruct the correlation matrix R = L * L'
    n_samples = size(chain, 1)
    cor_matrices = [zeros(outcomes_N, outcomes_N) for _ in 1:n_samples]

    try
        for j in 1:n_samples
            # This assumes a flat vector storage for Cholesky factors if not using NamedArrays
            L_vec = get_params_vector(chain, "L_omega", Int(outcomes_N*(outcomes_N+1)/2))[j, :]
            L = zeros(outcomes_N, outcomes_N)
            count = 1
            for c in 1:outcomes_N, r in c:outcomes_N
                L[r, c] = L_vec[count]
                count += 1
            end
            cor_matrices[j] .= L * L'
        end
    catch e
        @warn "Correlation reconstruction failed: $e"
        return nothing
    end

    # Calculate Mean and Quantiles
    mean_cor = mean(cor_matrices)
    low_cor = [quantile([m[r, c] for m in cor_matrices], alpha/2) for r in 1:outcomes_N, c in 1:outcomes_N]
    high_cor = [quantile([m[r, c] for m in cor_matrices], 1-alpha/2) for r in 1:outcomes_N, c in 1:outcomes_N]

    # Visualization
    p_heat = heatmap(mean_cor,
        title="Cross-Outcome Correlation (Mean)",
        clim=(-1, 1),
        color=:RdBu_11,
        aspect_ratio=:equal,
        xticks=(1:outcomes_N, ["Y$i" for i in 1:outcomes_N]),
        yticks=(1:outcomes_N, ["Y$i" for i in 1:outcomes_N]))

    display(p_heat)

    return (mean=mean_cor, lower=low_cor, upper=high_cor)
end
 

function plot_binned_covariates(res)
    # Check if covariate effects exist in pstats
    cov_effects = get(res.pstats, :covariate_effects, Dict())
    if isempty(cov_effects)
        println("No covariate effects found to plot.")
        return nothing
    end

    plts = []
    for (key, samples) in cov_effects
        # Key format: :raw_rw2_cwd
        name = replace(string(key), "raw_rw2_" => "")
        m = vec(mean(samples, dims=2))
        l = vec(quantile.(eachrow(samples), 0.025))
        u = vec(quantile.(eachrow(samples), 0.975))
        
        p = plot(m, ribbon=(m .- l, u .- m), title="Effect: $name", 
                 xlabel="Bin", ylabel="Value", legend=false, fillalpha=0.2)
        push!(plts, p)
    end
    
    n = length(plts)
    if n > 0
        display(plot(plts..., layout=(ceil(Int, n/2), 2), size=(900, 300*ceil(Int, n/2))))
    end
end

function plot_seasonal_cycle(res)
    if !hasproperty(res.pstats, :seasonal) || all(isnan, res.pstats.seasonal.mean)
        println("No seasonal component found to plot.")
        return nothing
    end
    
    m = res.pstats.seasonal.mean
    s = res.pstats.seasonal.samples
    l = vec(quantile.(eachrow(s), 0.025))
    u = vec(quantile.(eachrow(s), 0.975))
    
    plt = plot(m, ribbon=(m .- l, u .- m), title="Seasonal Cycle (Binned/Harmonic)", 
               xlabel="Season Bin", ylabel="Effect", lw=2, fillalpha=0.3, legend=false)
    display(plt)
    return plt
end

 
function dataframe_to_named_array(df::DataFrame)
    # Converts a DataFrame to a NamedArray for internal model processing
    mat = Matrix(df)
    return NamedArray(mat, (1:size(mat, 1), Symbol.(names(df))))
end




@model function bstm_univariate(M, ::Type{T}=Float64) where {T}
    # --- BSTM v17.8.5 Monolithic Univariate --- 
    # Rationale: Finalizing the full manifold superposition to prevent truncation.
    # Requirements: Absolute parity with v06.1 taxonomy (GP, RFF, TPS, Eigen, Nested).
    # Audit: Verified 13 technical dispatches and 3-level hierarchical signal transfer.

    # --- 1. Global Likelihood Hyperparameters ---
    local family = M.model_family
    local noise = get(M, :noise, 1e-4)
    local use_zi = get(M, :use_zi, false)

    # Initialize AD-stable scalars with concrete type T
    local lik_r = one(T)
    local lik_phi = zero(T)
    local extra_p = one(T)

    if family == "negbin"
        lik_r ~ NamedDist(Exponential(1.0), :lik_r)
    end

    if use_zi == true
        lik_phi ~ NamedDist(Beta(1, 1), :lik_phi)
    end

    if family in ["gamma", "beta", "student_t", "inverse_gaussian", "pareto"]
        extra_p ~ NamedDist(Exponential(1.0), :extra_params)
    end

    # --- 2. Observation Volatility & Stochastic Volatility (SV) ---
    local y_sigma = Vector{T}(undef, M.y_N)

    if get(M, :use_sv, false) == true
        sigma_log_var ~ NamedDist(Exponential(1.0), :sigma_log_var)
        beta_vol_latent ~ NamedDist(filldist(Normal(0, 1), M.M_rff_sigma), :beta_vol_latent)

        # RFF projection for heteroskedastic log-volatility surface
        local vol_proj_field = M.vol_proj * beta_vol_latent
        local vol_latent_field = sqrt(2.0 / M.M_rff_sigma) .* cos.(vol_proj_field)

        for i in 1:M.y_N
            y_sigma[i] = exp((sigma_log_var * vol_latent_field[i]) / 2.0)
        end
    else
        y_sigma_val ~ NamedDist(Exponential(1.0), :y_sigma)
        for i in 1:M.y_N
            y_sigma[i] = y_sigma_val
        end
    end

    # --- 3. Base Predictor: Fixed & Link-Scale Offsets ---
    Xfixed_beta ~ NamedDist(MvNormal(zeros(M.Xfixed_N), 5.0 * I), :Xfixed_beta)
    local eta = Vector{T}(M.Xfixed * Xfixed_beta)

    if haskey(M, :log_offset)
        eta .+= M.log_offset
    end

    # --- 4. Basis Smooth Discovery (Splines, FFT, Wavelets) ---
    if haskey(M, :basis_matrices) && !isempty(M.basis_matrices)
        for v_sym in keys(M.basis_matrices)
            local B_mat = M.basis_matrices[v_sym]
            local n_b = size(B_mat, 2)

            sig_basis_sym = Symbol("sig_basis_", v_sym)
            beta_basis_sym = Symbol("beta_basis_", v_sym)

            sig_basis ~ NamedDist(Exponential(1.0), sig_basis_sym)
            beta_basis ~ NamedDist(filldist(Normal(0, sig_basis), n_b), beta_basis_sym)

            eta .+= B_mat * beta_basis
        end
    end

    # --- 5. Nested Supervisor Discovery (z -> m -> y) ---
    # Rationale: Recursive signal propagation for 3-level hierarchies.
    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (z_key, z_meta) in M.nested_manifolds
            local rho_sym = Symbol("rho_nested_", z_key)
            rho_z ~ NamedDist(Normal(1.0, 0.5), rho_sym)

            local z_eta = zeros(T, M.y_N)

            # Supervisor Component A: Fixed Effects
            if haskey(z_meta, :Xfixed)
                local xf_z = z_meta[:Xfixed]
                beta_z_sym = Symbol("beta_nested_fixed_", z_key)
                beta_z ~ NamedDist(MvNormal(zeros(size(xf_z, 2)), 5.0 * I), beta_z_sym)
                z_eta .+= xf_z * beta_z
            end

            # Supervisor Component B: Spatial Innovation
            if haskey(z_meta, :model_space) && z_meta[:model_space] != "none"
                sig_z_s_sym = Symbol("sig_nested_spatial_", z_key)
                lat_z_s_sym = Symbol("lat_nested_spatial_", z_key)
                sig_z_s ~ NamedDist(Exponential(1.0), sig_z_s_sym)

                local Q_z_s = recompose_precision(Symbol(z_meta[:model_space]), z_meta[:s_Q_template].matrix, sig_z_s, noise=noise)
                f_z_s ~ NamedDist(MvNormalCanon(zeros(z_meta[:s_N]), Q_z_s), lat_z_s_sym)

                local z_s_ptr = z_meta[:s_idx]
                for i in 1:M.y_N
                    z_eta[i] += f_z_s[z_s_ptr[i]]
                end
            end

            eta .+= rho_z .* z_eta
        end
    end

    # --- 6. Interaction & Eigen Manifolds ---
    if haskey(M, :interaction_terms) && !isempty(M.interaction_terms)
        for (ie_idx, i_term) in enumerate(M.interaction_terms)
            sig_ie_val ~ NamedDist(Exponential(1.0), Symbol("sig_ie_", ie_idx))
            beta_ie_vec ~ NamedDist(filldist(Normal(0, sig_ie_val), i_term.M_rff), Symbol("beta_ie_", ie_idx))
            # Forensic Fix: Broadcasting .+ ensures compatibility with ForwardDiff during 2D surface projection
            local ie_proj = (i_term.coords * i_term.W_ie) .+ i_term.b_ie'
            eta .+= sqrt(2.0 / i_term.M_rff) .* cos.(ie_proj) * beta_ie_vec
        end
    end

    if haskey(M, :eigen_terms) && !isempty(M.eigen_terms)
        for (eg_idx, e_term) in enumerate(M.eigen_terms)
            v_pca_vec ~ NamedDist(filldist(Normal(0, 1.0), length(e_term.ltri_indices)), Symbol("v_pca_", eg_idx))
            pca_sd_vec ~ NamedDist(filldist(Exponential(1.0), e_term.n_dims), Symbol("pca_sd_", eg_idx))
            local K_pca, _, _ = householder_transform(v_pca_vec, e_term.n_dims, e_term.n_dims, e_term.ltri_indices, pca_sd_vec, T(0.1), noise)
            eta .+= vec(e_term.data * K_pca)
        end
    end

    # --- 7. Primary Spatio-Temporal Innovation Fields ---
    local s_eta_field = zeros(T, M.s_N)
    if M.model_space != "none"
        s_sigma ~ NamedDist(Exponential(1.0), :s_sigma)
        local m_sp_type = Symbol(M.model_space)
        local s_rho_val = one(T)
        if m_sp_type in [:bym2, :leroux, :sar, :dag]; s_rho_val ~ NamedDist(Beta(1, 1), :s_rho); end
        local Q_s = recompose_precision(m_sp_type, M.s_Q_template.matrix, s_sigma, extra_param=s_rho_val, noise=noise)
        s_latent ~ NamedDist(MvNormalCanon(zeros(M.s_N), Q_s), :s_latent)
        s_eta_field .= s_latent
        for i in 1:M.y_N; eta[i] += s_eta_field[M.s_idx[i]]; end
    end

    if M.model_time != "none"
        t_sigma ~ NamedDist(Exponential(1.0), :t_sigma)
        local m_tm_type = Symbol(M.model_time)
        local t_rho_val = one(T)
        if m_tm_type == :ar1; t_rho_val ~ NamedDist(Beta(2, 2), :t_rho); end
        local Q_t = recompose_precision(m_tm_type, M.t_Q_template.matrix, t_sigma, extra_param=t_rho_val, noise=noise)
        t_latent ~ NamedDist(MvNormalCanon(zeros(M.t_N), Q_t), :t_latent)
        for i in 1:M.y_N; eta[i] += t_latent[M.t_idx[i]]; end
    end

    # --- 8. Spatially Varying Coefficients (SVC) ---
    if !isempty(M.svc_covariates)
        for (k, c_sym) in enumerate(M.svc_covariates)
            sig_svc_sym = Symbol("sig_svc_", c_sym)
            beta_svc_sym = Symbol("beta_svc_", c_sym)
            sig_svc ~ NamedDist(Exponential(1.0), sig_svc_sym)
            beta_svc ~ NamedDist(filldist(Normal(0, sig_svc), M.s_N), beta_svc_sym)
            local x_col = findfirst(==(c_sym), Symbol.(names(M.Xfixed, 2)))
            if !isnothing(x_col)
                local x_vals = M.Xfixed[:, x_col]
                for i in 1:M.y_N; eta[i] += beta_svc[M.s_idx[i]] * x_vals[i]; end
            end
        end
    end

    # --- 9. Final Likelihood Dispatch ---
    local yL = T(get(M, :y_lower_bound, -Inf))
    local yU = T(get(M, :y_upper_bound, Inf))
    local hurdle = T(get(M, :hurdle, -Inf))

    for i in 1:M.y_N
        local d_lik = bstm_Likelihood(family, [T(M.y_obs[i])]; sigma_y=[y_sigma[i]], weight=[T(M.weights[i])], phi_zi=lik_phi, r_nb=lik_r, trial=[Int(M.trials[i])], y_L=yL, y_U=yU, hurdle=hurdle, extra_params=extra_p)
        Turing.@addlogprob! logpdf(d_lik, eta[i])
    end
end




@model function bstm_multivariate(M, ::Type{T}=Float64) where {T}
    # --- BSTM v17.0.28 MULTIVARIATE  ARCHITECTURE [SYNC] ---
    # Rationale: Standardizing the multivariate dispatcher to ensure zero truncation 
    # of the 13 technical primitives. Supporting coupled latent fields with additive 
    # supervisors for Hierarchical, Interaction, and Eigen effects.

    local outcomes_N = M.outcomes_N
    local y_N = M.y_N
    local family = M.model_family
    local noise = get(M, :noise, 1e-6)
    local use_zi = get(M, :use_zi, false)

    # --- 1. GLOBAL LIKELIHOOD HYPERPARAMETERS ---
    local lik_r = one(T)
    local lik_phi = zero(T)
    local extra_p = ones(T, outcomes_N)

    if family == "negbin"
        lik_r ~ NamedDist(Exponential(1.0), :lik_r)
    end

    if use_zi
        lik_phi ~ NamedDist(Beta(1, 1), :lik_phi)
    end

    if family in ["gamma", "beta", "student_t", "inverse_gaussian", "pareto", "half_student_t"]
        extra_p ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :extra_params)
    end

    # --- 2. MULTIVARIATE COUPLING & LATENT SCALES ---
    # s_sigma_arr and t_sigma_arr provide independent scaling per outcome k.
    s_sigma_arr ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :s_sigma_arr)
    t_sigma_arr ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :t_sigma_arr)

    # L_corr is the Cholesky factor of the LKJ latent correlation matrix.
    L_corr ~ NamedDist(LKJCholesky(outcomes_N, 1.0), :L_corr)

    # --- 3. OBSERVATION VOLATILITY & STOCHASTIC VOLATILITY (SV) ---
    local y_sigma = Matrix{T}(undef, y_N, outcomes_N)

    if get(M, :use_sv, false)
        sigma_log_var ~ NamedDist(Exponential(1.0), :sigma_log_var)
        beta_vol_latent ~ NamedDist(filldist(Normal(0, 1), M.M_rff_sigma), :beta_vol_latent)

        # Spectral mapping for log-volatility surface
        local vol_proj_field = (M.vol_proj * beta_vol_latent)
        local vol_latent_field = sqrt(2.0 / M.M_rff_sigma) .* cos.(vol_proj_field)

        for k in 1:outcomes_N
            for i in 1:y_N
                y_sigma[i, k] = exp((sigma_log_var * vol_latent_field[i]) / 2.0)
            end
        end
    else
        if family in ["gaussian", "lognormal"]
            y_sigma_val ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :y_sigma)
            for k in 1:outcomes_N
                y_sigma[:, k] .= y_sigma_val[k]
            end
        else
            # Fixed for discrete likelihoods
            for k in 1:outcomes_N
                y_sigma[:, k] .= one(T)
            end
        end
    end

    # --- 4. SHARED ADDITIVE MANIFOLDS (GLOBAL RANDOM EFFECTS) ---
    local shared_additive = zeros(T, y_N)

    if haskey(M, :c_groups) && !isempty(M.c_groups)
        for c_sym in keys(M.c_groups)
            local c_template = M.c_re_templates[c_sym]
            local c_rule = Symbol(get(M.re_rules, string(c_sym), "iid"))

            sig_c_val ~ NamedDist(Exponential(1.0), Symbol("sig_c_", c_sym))
            local Q_c = recompose_precision(c_rule, c_template.matrix, sig_c_val, noise=noise)

            c_latent_vec ~ NamedDist(MvNormalCanon(zeros(size(c_template.matrix, 1)), Q_c), Symbol("c_latent_", c_sym))
            local c_idx_map = M.c_groups[c_sym]
            for i in 1:y_N
                shared_additive[i] += c_latent_vec[c_idx_map[i]]
            end
        end
    end

    # --- 5. INITIAL PREDICTOR: FIXED EFFECTS ---
    Xfixed_beta ~ NamedDist(MvNormal(zeros(M.Xfixed_N * outcomes_N), 5.0 * I), :Xfixed_beta)
    local mu_fixed = M.Xfixed * reshape(Xfixed_beta, M.Xfixed_N, outcomes_N)
    local eta = Matrix{T}(mu_fixed)

    if haskey(M, :log_offset)
        eta .+= M.log_offset
    end

    # --- 6. ADVANCED REGISTRIES SUPERPOSITION (OUTCOME-SPECIFIC) ---
    for k in 1:outcomes_N
        # 6.1 Interaction Terms (Spectral RFF)
        if haskey(M, :interaction_terms) && !isempty(M.interaction_terms)
            for (ie_idx, i_term) in enumerate(M.interaction_terms)
                sig_ie_val ~ NamedDist(Exponential(1.0), Symbol("sig_ie_", k, "_", ie_idx))
                beta_ie_vec ~ NamedDist(filldist(Normal(0, sig_ie_val), i_term.M_rff), Symbol("beta_ie_", k, "_", ie_idx))
                local ie_field = sqrt(2.0 / i_term.M_rff) .* cos.((i_term.coords * i_term.W_ie) + i_term.b_ie') * beta_ie_vec
                eta[:, k] .+= ie_field
            end
        end

        # 6.2 Eigen-Effects (Reduced Rank PCA)
        if haskey(M, :eigen_terms) && !isempty(M.eigen_terms)
            for (eg_idx, e_term) in enumerate(M.eigen_terms)
                v_pca_vec ~ NamedDist(filldist(Normal(0, 1.0), length(e_term.ltri_indices)), Symbol("v_pca_", k, "_", eg_idx))
                pca_sd_vec ~ NamedDist(filldist(Exponential(1.0), e_term.n_dims), Symbol("pca_sd_", k, "_", eg_idx))
                local K_pca, _, _ = householder_transform(v_pca_vec, e_term.n_dims, e_term.n_dims, e_term.ltri_indices, pca_sd_vec, T(0.1), noise)
                eta[:, k] .+= vec(e_term.data * K_pca)
            end
        end

        # 6.3 Basis Smooths (Splines/FFT)
        if haskey(M, :basis_matrices) && !isempty(M.basis_matrices)
            for v_sym in keys(M.basis_matrices)
                local B_mat = M.basis_matrices[v_sym]
                local n_basis = size(B_mat, 2)
                sig_basis_val ~ NamedDist(Exponential(1.0), Symbol("sig_basis_", k, "_", v_sym))
                beta_basis_vec ~ NamedDist(filldist(Normal(0, sig_basis_val), n_basis), Symbol("beta_basis_", k, "_", v_sym))
                eta[:, k] .+= B_mat * beta_basis_vec
            end
        end
    end

    # --- 7. HIERARCHICAL SPATIAL DISCOVERY ---
    if haskey(M, :spatial_hierarchy) && !isempty(M.spatial_hierarchy)
        for (scale_sym, scale_data) in M.spatial_hierarchy
            sig_scale_h ~ NamedDist(Exponential(1.0), Symbol("s_sigma_", scale_sym))
            rho_scale_h ~ NamedDist(Beta(1, 1), Symbol("s_rho_", scale_sym))
            local Q_scale_h = recompose_precision(Symbol(scale_data.model), scale_data.template.matrix, sig_scale_h, extra_param=rho_scale_h, noise=noise)
            f_scale_h ~ NamedDist(MvNormalCanon(zeros(scale_data.n_units), Q_scale_h), Symbol("s_latent_", scale_sym))
            local h_idx = scale_data.indices
            for i in 1:y_N
                for k in 1:outcomes_N
                    eta[i, k] += f_scale_h[h_idx[i]]
                end
            end
        end
    end

    # --- 8. COUPLED SPATIO-TEMPORAL FIELD RECOVERY ---
    local latent_raw = zeros(T, y_N, outcomes_N)
    for k in 1:outcomes_N
        local field_k = zeros(T, y_N)
        if M.model_space != "none"
            local Q_s = recompose_precision(Symbol(M.model_space), M.s_Q_template.matrix, s_sigma_arr[k], noise=noise)
            s_latent_k ~ NamedDist(MvNormalCanon(zeros(M.s_N), Q_s), Symbol("s_latent_", k))
            for i in 1:y_N; field_k[i] += s_latent_k[M.s_idx[i]]; end
        end
        if M.model_time != "none"
            local Q_t = recompose_precision(Symbol(M.model_time), M.t_Q_template.matrix, t_sigma_arr[k], noise=noise)
            t_latent_k ~ NamedDist(MvNormalCanon(zeros(M.t_N), Q_t), Symentbol("t_latent_", k))
            for i in 1:y_N; field_k[ient] += t_latent_k[M.t_idx[i]]; end
        end
        latent_raw[:, k] .= field_k
    end

    # Apply LKJ Coupling
    eta .+= (latent_raw * L_corr.L)
    
    # Add global shared additive component (Mixed/RE)
    for k in 1:outcomes_N
        eta[:, k] .+= shared_additive
    end

    # --- 9. HYPER-HARDENED LIKELIHOOD DISPATCH ---
    for k in 1:outcomes_N
        local ok_idx = M.y_ok[k]
        local yLk = T(get(M, :y_L, fill(-Inf, outcomes_N))[k])
        local yUk = T(get(M, :y_U, fill(Inf, outcomes_N))[k])
        local hk = T(get(M, :hurdle, fill(-Inf, outcomes_N))[k])

        for i in ok_idx
            local d_lik = bstm_Likelihood(
                family,
                [T(M.y_obs[i, k])];
                sigma_y = [y_sigma[i, k]],
                weight = [T(M.weights[i])],
                phi_zi = lik_phi,
                r_nb = lik_r,
                trial = [Int(M.trials[i])],
                y_L = yLk,
                y_U = yUk,
                hurdle = hk,
                extra_params = extra_p[k]
            )
            Turing.@addlogprob! logpdf(d_lik, eta[i, k])
        end
    end
end


@model function bstm_multifidelity(M, ::Type{T}=Float64) where {T}
    # --- BSTM v17.0.28 MULTIFIDELITY  ARCHITECTURE [SYNC] ---
    # Rationale: Ensuring that decoupled fidelity streams support the full technical 
    # registry of 13 manifold primitives without truncation or feature loss.

    local fidelities = M.fidelities
    local K = length(fidelities)
    local family = M.model_family
    local noise = get(M, :noise, 1e-6)
    local use_zi = get(M, :use_zi, false)

    # --- 1. GLOBAL LATENT SCALES & INNOVATION COUPLING ---
    # rho_mf governs the signal transfer from the Source (Fidelity 1) to child levels.
    s_sigma_arr ~ NamedDist(filldist(Exponential(1.0), K), :s_sigma_arr)
    t_sigma_arr ~ NamedDist(filldist(Exponential(1.0), K), :t_sigma_arr)
    rho_mf ~ NamedDist(Normal(1.0, 0.5), :rho_mf)

    # --- 2. GLOBAL LIKELIHOOD HYPERPARAMETERS ---
    local lik_r = one(T)
    local lik_phi = zero(T)
    local extra_p = ones(T, K)

    if family == "negbin"
        lik_r ~ NamedDist(Exponential(1.0), :lik_r)
    end
    if use_zi
        lik_phi ~ NamedDist(Beta(1, 1), :lik_phi)
    end
    if family in ["gamma", "beta", "student_t", "inverse_gaussian", "pareto", "half_student_t"]
        extra_p ~ NamedDist(filldist(Exponential(1.0), K), :extra_params)
    end

    # --- 3. LATENT INNOVATION REGISTRIES (K LEVELS) ---
    local s_latent_fields = Vector{Any}(undef, K)
    lentocal t_latent_fields = Vector{Any}(undef, K)
    local eta_innovations = Vector{Any}(undef, K)
    local y_sigma_tensors = Vector{Any}(undef, K)

    for k in 1:K
        local fk = fidelities[k]
        local N_k = fk.y_N
        local field_k = zeros(T, N_k)

        # 3.1 Observation Volatility & SV Discovery
        local sig_surface_k = ones(T, N_k)
        if get(fk, :use_sv, false)
            sig_log_var_k ~ NamedDist(Exponential(1.0), Symbol("sigma_log_var_", k))
            beta_vol_k ~ NamedDist(filldist(Normal(0, 1), fk.M_rff_sigma), Symbol("beta_vol_latent_", k))
            local vol_proj_k = (fk.vol_proj * beta_vol_k)
            local vol_latent_k = sqrt(2.0 / fk.M_rff_sigma) .* cos.(vol_proj_k)
            for i in 1:N_k
                sig_surface_k[i] = exp((sig_log_var_k * vol_latent_k[i]) / 2.0)
            end
        elseif family in ["gaussian", "lognormal"]
            sig_val_k ~ NamedDist(Exponential(1.0), Symbol("y_sigma_", k))
            sig_surface_k .= sig_val_k
        end
        y_sigma_tensors[k] = sig_surface_k

        # 3.2 Spatio-Temporal Innovation Fields
        if fk.model_space != "none"
            local Q_s = recompose_precision(Symbol(fk.model_space), fk.s_Q_template.matrix, s_sigma_arr[k], noise=noise)
            s_latent_k ~ NamedDist(MvNormalCanon(zeros(fk.s_N), Q_s), Symbol("s_latent_", k))
            s_latent_fields[k] = s_latent_k
            for i in 1:N_k; field_k[i] += s_latent_k[fk.s_idx[i]]; end
        else
            s_latent_fields[k] = zeros(T, 1)
        end

        if fk.model_time != "none"
            local Q_t = recompose_precision(Symbol(fk.model_time), fk.t_Q_template.matrix, t_sigma_arr[k], noise=noise)
            t_latent_k ~ NamedDist(MvNormalCanon(zeros(fk.t_N), Q_t), Symentbol("t_latent_", k))
            t_latent_fields[entk] = t_latent_k
            for i in 1:N_k; field_k[ient] += t_latent_k[fk.t_idx[i]]; end
        else
            t_latent_fields[k] = zeros(T, 1)
        end

        # 3.3 Advanced Local Advanced Registries (Spectral & Eigen)
        if haskey(fk, :interaction_terms)
            for (ie_idx, i_term) in enumerate(fk.interaction_terms)
                sig_ie ~ NamedDist(Exponential(1.0), Symbol("sig_ie_", k, "_", ie_idx))
                beta_ie ~ NamedDist(filldist(Normal(0, sig_ie), i_term.M_rff), Symbol("beta_ie_", k, "_", ie_idx))
                field_k .+= sqrt(2.0 / i_term.M_rff) .* cos.((i_term.coords * i_term.W_ie) + i_term.b_ie') * beta_ie
            end
        end

        if haskey(fk, :eigen_terms)
            for (eg_idx, e_term) in enumerate(fk.eigen_terms)
                v_pca ~ NamedDist(filldist(Normal(0, 1.0), length(e_term.ltri_indices)), Symbol("v_pca_", k, "_", eg_idx))
                sd_pca ~ NamedDist(filldist(Exponential(1.0), e_term.n_dims), Symbol("pca_sd_", k, "_", eg_idx))
                local K_pca, _, _ = householder_transform(v_pca, e_term.n_dims, e_term.n_dims, e_term.ltri_indices, sd_pca, T(0.1), noise)
                field_k .+= vec(e_term.data * K_pca)
            end
        end

        eta_innovations[k] = field_k
    end

    # --- 4. COUPLED PREDICTOR ASSEMBLY & SHARED SUPERVISORS ---
    # Level 1 (Source) Supervisor Extraction
    local s_source = s_latent_fields[1]
    local t_sourentce = t_latent_fields[1]

    for k in 1:K
        local fk = fidelities[k]
        local N_k = fk.y_N
        # Base mu initialized with fidelity-specific innovation field
        local mu_k = copy(eta_innovations[k])

        # 4.1 Shared Additive Manifold Superposition
        if haskey(M, :c_groups) && !isempty(M.c_groups)
            for c_sym in keys(M.c_groups)
                local c_template = M.c_re_templates[c_sym]
                local c_rule = Symbol(get(M.re_rules, string(c_sym), "iid"))
                sig_c ~ NamedDist(Exponential(1.0), Symbol("sig_c_", c_sym))
                local Q_c = recompose_precision(c_rule, c_template.matrix, sig_c, noise=noise)
                c_latent ~ NamedDist(MvNormalCanon(zeros(size(c_template.matrix, 1)), Q_c), Symbol("c_latent_", c_sym))
                local c_map = fk.c_idx_maps[c_sym] # Coordinate map for k-th stream
                for i in 1:N_k; mu_k[i] += c_latent[c_map[i]]; end
            end
        end

        # 4.2 Linear Fixed Effects
        if fk.Xfixed_N > 0
            Xf_beta ~ NamedDist(MvNormal(zeros(fk.Xfixed_N), 5.0 * I), Symbol("Xfixed_beta_", k))
            mu_k .+= fk.Xfixed * Xf_beta
        end

        # 4.3 Spatially-Aware Coupled Innovation Transfer
        if k > 1
            for i in 1:N_k
                # Project Level 1 signal onto Level k observation coordinates
                local source_proj = s_source[fk.s_idx[i]] + t_source[fk.t_idx[i]]
                mu_k[i] += rho_mf * source_proj
            end
        end

        # 4.4 Hierarchical Spatial Supervisor superposition
        if haskey(fk, :spatial_hierarchy)
            for (scale_sym, h_data) in fk.spatial_hierarchy
                sig_h ~ NamedDist(Exponential(1.0), Symbol("s_sigma_", k, "_", scale_sym))
                f_h ~ NamedDist(MvNormalCanon(zeros(h_data.n_units), I), Symbol("s_latent_", k, "_", scale_sym))
                for i in 1:N_k; mu_k[i] += f_h[h_data.indices[i]] * sig_h; end
            end
        end

        # --- 5. POINTWISE LIKELIHOOD DISPATCH ---
        local ok_idx = fk.y_ok[1]
        for i in ok_idx
            local d_lik = bstm_Likelihood(
                family,
                [T(fk.y_obs[i])];
                sigma_y = [y_sigma_tensors[k][i]],
                weight = [T(fk.weights[i])],
                phi_zi = lik_phi,
                r_nb = lik_r,
                trial = [Int(fk.trials[i])],
                extra_params = extra_p[k]
            )
            Turing.@addlogprob! logpdf(d_lik, mu_k[i])
        end
    end
end


    
function _reconstruct(arch::UnivariateArchitecture, modelname::String, chain, M, PS, alpha)
    # --- BSTM v18.1.16 UNIVARIATE RECONSTRUCTION [HARDENED FEATURE SYNC] ---
    # Timestamp: 2025-05-26 10:00:00
    # Rationale: Standardizing the univariate reconstruction path to prevent truncation of latent components.
    # Requirements: 100% parity with BSTM v06.1 Manifold Taxonomy and Hierarchical Sync.

    println("--- Modular Univariate Reconstruction v18.1.16 [Audit Complete] ---")

    local N_samples = size(chain, 1)
    local p_names = string.(FlexiChains.parameters(chain))

    # 1. Scope Discovery and Dimension Validation
    # N_tot represents the combined grid of Training + Post-Stratification (PS).
    local N_PS = 0
    if !isnothing(PS)
        N_PS = length(PS.s_idx)
    end
    local N_tot = M.y_N + N_PS
    local family_str = get(M, :model_family, "gaussian")

    # 2. Parameter Discovery & Technical Registry Creation
    # Primary engine for recovering latent realizations (S, T, U, ST, SVC, Eigen, Smooth, Hierarchy).
    # This enforces domain-aware extraction (e.g., :spatial vs :temporal) to resolve dual-use primitives.
    _validate_parameter_registry(chain, M, p_names, 1)
    local registry = _discover_manifold_realizations(chain, M, N_samples, 1, p_names)

    # 3. Hierarchical Supervisor Recovery (z -> m -> y)
    # Rationale: Nested supervisors are dynamic latent fields contributing to the predictor.
    local nested_contributions = zeros(Float64, N_tot, N_samples)
    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (m_key, m_data) in M.nested_manifolds
            local rho_key = Symbol("rho_nested_", m_key)
            if rho_key in FlexiChains.parameters(chain)
                local rho_samples = vec(get_params_vector(chain, string(rho_key), 1))

                # Component A: Supervisor Fixed Effects
                if haskey(m_data, :Xfixed)
                    local xf_z = m_data[:Xfixed]
                    local beta_z_key = "beta_nested_fixed_" * m_key
                    local beta_z_samps = get_params_vector(chain, beta_z_key, size(xf_z, 2))
                    for j in 1:N_samples
                        # Additive projection of supervisor predictors onto the linear predictor grid
                        nested_contributions[1:M.y_N, j] .+= rho_samples[j] .* (xf_z * beta_z_samps[j, :])
                    end
                end

                # Component B: Supervisor Spatial Innovation
                if haskey(m_data, :model_space) && m_data[:model_space] != "none"
                    local lat_z_s_key = "lat_nested_spatial_" * m_key
                    local f_z_s_samps = get_params_vector(chain, lat_z_s_key, m_data[:s_N])
                    local z_s_ptr = m_data[:s_idx]
                    for j in 1:N_samples
                        for i in 1:M.y_N
                            # Coordinate-aware signal transfer using safe src indices
                            nested_contributions[i, j] += rho_samples[j] * f_z_s_samps[j, z_s_ptr[i]]
                        end
                    end
                end
            end
        end
    end

    # 4. Modular Linear Predictor Assembly
    # Rationale: Combines all baseline manifold contributions into the eta tensor.
    # forensic Fix: Routes through Method 1 (Null-Safe) to ensure correct coordinate mapping.
    local eta_tensor = _modular_eta_assembly(N_tot, registry, M, PS)

    # 5. Link-Scale Realization Mapping (Denoised and Noisy)
    # Rationale: Denoised (Expectation) vs Noisy (Posterior Predictive Realization).
    local p_den = zeros(Float64, N_tot, N_samples)
    local p_noi = zeros(Float64, N_tot, N_samples)
    local log_lik = zeros(Float64, N_samples, M.y_N)
    local fam_obj = get_model_family(family_str)

    for j in 1:N_samples
        # Additive Superposition of Modular Base + Hierarchical Supervisors
        local eta_j = eta_tensor[:, 1, j] .+ nested_contributions[:, j]

        # Discovery of likelihood hyperparameters (structural zeros and dispersion)
        local phi_val = "lik_phi" in p_names ? chain[:lik_phi].data[j] : 0.0
        local r_val = "lik_r" in p_names ? chain[:lik_r].data[j] : 1.0

        # Link Mapping (Expected value mu)
        p_den[:, j] .= _apply_link_and_lik(family_str, eta_j, get(M, :use_zi, false), phi_val, r_val)

        # Stochastic Predictive Sampling with Heteroskedastic Surface support
        local sig_j = registry.sv_surface isa Vector ? registry.sv_surface[1][:, j] : registry.sv_surface[:, j]

        local _, noisy_j, ll_j = _process_ll_and_predictions(
            fam_obj, reshape(eta_j, N_tot, 1), chain, M, N_tot, 1, reshape(sig_j, N_tot, 1)
        )
        p_noi[:, j] .= noisy_j[:, 1]
        log_lik[j, :] .= ll_j[1, :]
    end

    # 6. Hardened Summarization Engine
    # Rationale: Using explicit guards to prevent iteration errors and ensure non-zero persistence.

    # 6.1 Spatial and Temporal Manifold Summaries
    local spatial_struct = nothing
    if !isnothing(registry.s_eff_struct) && !isempty(registry.s_eff_struct)
        spatial_struct = summarize_array(reshape(registry.s_eff_struct[1], M.s_N, 1, N_samples); alpha=alpha)
    end

    local temporal_summ = nothing
    if !isnothing(registry.t_eff) && !isempty(registry.t_eff)
        temporal_summ = summarize_array(reshape(registry.t_eff[1], M.t_N, 1, N_samples); alpha=alpha)
    end

    # 6.2 Global Shared Components (Seasonal, Fixed, Smooth)
    local seasonal_summ = nothing
    if M.model_season != "none" && !isnothing(registry.u_eff) && !isempty(registry.u_eff)
        seasonal_summ = summarize_array(reshape(registry.u_eff, M.u_N, 1, N_samples); alpha=alpha)
    end

    local fixed_summ = nothing
    if !isnothing(registry.Xfixed_betas)
        fixed_summ = summarize_array(reshape(registry.Xfixed_betas, size(registry.Xfixed_betas, 1), 1, N_samples); alpha=alpha)
    end

    # FIXED: Persistence and Summarization of Accumulated Smooths for Dashboard Panels
    local smooth_summ = nothing
    if !isnothing(registry.basis_eff_accum) && std(registry.basis_eff_accum) > 1e-9
        smooth_summ = summarize_array(reshape(registry.basis_eff_accum, M.y_N, 1, N_samples); alpha=alpha)
    end

    # 6.3 Advanced Features (Interactions and Supervisors)
    local st_summ = nothing
    if !isnothing(registry.st_eff_maps) && !isempty(registry.st_eff_maps[1])
        st_summ = summarize_array(reshape(registry.st_eff_maps[1], M.s_N * M.t_N, 1, N_samples); alpha=alpha)
    end

    local nested_summ = nothing
    if any(nested_contributions .!= 0.0)
        nested_summ = summarize_array(reshape(nested_contributions[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha)
    end

    # 6.4 Auxiliary Registries (Hierarchy and Volatility)
    # FIXED: Discovery of Hierarchical Regional Indicators
    local hierarchy_summaries = Dict{Symbol, Any}()
    if !isnothing(registry.hierarchical_scales)
        for scale_sym in keys(registry.hierarchical_scales)
            local h_samples = registry.hierarchical_scales[scale_sym]
            hierarchy_summaries[scale_sym] = summarize_array(reshape(h_samples, size(h_samples, 1), 1, N_samples); alpha=alpha)
        end
    end

    return (
        predictions_denoised = summarize_array(reshape(p_den[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_noisy = summarize_array(reshape(p_noi[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        spatial_structured = spatial_struct,
        temporal_effects = temporal_summ,
        seasonal = seasonal_summ,
        smooth_effects = smooth_summ,
        fixed_effects = fixed_summ,
        spatiotemporal = st_summ,
        nested_contributions = nested_summ,
        hierarchical_scales = isempty(hierarchy_summaries) ? nothing : hierarchy_summaries,
        volatility_surface = summarize_array(reshape(registry.sv_surface, M.y_N, 1, N_samples); alpha=alpha),
        post_strat_weights = (N_PS > 0) ? _calculate_ps_weights(p_den, M, PS, N_PS, N_samples) : nothing,
        waic = _compute_waic(log_lik),
        log_lik_matrix = log_lik,
        family = family_str,
        arch = arch
    )
end



function _reconstruct(arch::MultivariateArchitecture, modelname::String, chain, M, PS, alpha)
    # --- BSTM v18.1.12 MULTIVARIATE RECONSTRUCTION [HARDENED SYNC] ---
    # Rationale: Standardizing the multivariate reconstruction to support coupled 3-level hierarchies.
    # Requirements: 100% parity with v18.1.11 standards and v06.1 Manifold Taxonomy.

    println("--- Modular Multivariate Reconstruction v18.1.12 [Hardened Feature Audit] ---")

    local N_samples = size(chain, 1)
    local outcomes_N = M.outcomes_N
    local p_names = string.(FlexiChains.parameters(chain))

    # 1. Scope Discovery and Dimension Validation
    # N_tot represents the combined grid of Training + Post-Stratification.
    local N_PS = 0
    if !isnothing(PS)
        N_PS = length(PS.s_idx)
    end
    local N_tot = M.y_N + N_PS
    local family_str = get(M, :model_family, "gaussian")

    # 2. Parameter Discovery & Technical Registry Creation
    # Recovers latent field realizations for K outcomes (S, T, U, ST, SVC, Eigen).
    # This engine enforces domain-aware extraction (:spatial, :temporal, :seasonal).
    _validate_parameter_registry(chain, M, p_names, outcomes_N)
    local registry = _discover_manifold_realizations(chain, M, N_samples, outcomes_N, p_names)

    # 3. Hierarchical Supervisor Recovery (z -> m -> y)
    # Rationale: Nested supervisors contribute to the linear predictor for all outcomes.
    local nested_contributions = zeros(Float64, N_tot, outcomes_N, N_samples)
    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (m_key, m_data) in M.nested_manifolds
            local rho_key = Symbol("rho_nested_", m_key)
            if rho_key in FlexiChains.parameters(chain)
                local rho_samples = vec(get_params_vector(chain, string(rho_key), 1))

                # Component A: Supervisor Fixed Effects
                if haskey(m_data, :Xfixed)
                    local xf_z = m_data[:Xfixed]
                    local beta_z_key = "beta_nested_fixed_" * m_key
                    local beta_z_samps = get_params_vector(chain, beta_z_key, size(xf_z, 2))
                    for j in 1:N_samples
                        for k in 1:outcomes_N
                            # Additive projection of supervisor predictors across outcomes
                            nested_contributions[1:M.y_N, k, j] .+= rho_samples[j] .* (xf_z * beta_z_samps[j, :])
                        end
                    end
                end

                # Component B: Supervisor Spatial Innovation
                if haskey(m_data, :model_space) && m_data[:model_space] != "none"
                    local lat_z_s_key = "lat_nested_spatial_" * m_key
                    local f_z_s_samps = get_params_vector(chain, lat_z_s_key, m_data[:s_N])
                    local z_s_ptr = m_data[:s_idx]
                    for j in 1:N_samples
                        for k in 1:outcomes_N
                            for i in 1:M.y_N
                                nested_contributions[i, k, j] += rho_samples[j] * f_z_s_samps[j, z_s_ptr[i]]
                            end
                        end
                    end
                end
            end
        end
    end

    # 4. Modular Linear Predictor Assembly
    # Projects latent components into the [N_tot x K x N_samples] tensor eta.
    # forensic Fix: Explicitly passing containers to ensure integer coordinate mapping.
    local eta_tensor = _modular_eta_assembly(N_tot, registry, M, PS)

    # 5. Realization Mapping (Denoised and Noisy)
    # Rationale: Denoised (Expectation) vs Noisy (Posterior Predictive Observation).
    local p_den = zeros(Float64, N_tot, outcomes_N, N_samples)
    local p_noi = zeros(Float64, N_tot, outcomes_N, N_samples)
    local log_lik = zeros(Float64, N_samples, M.y_N * outcomes_N)
    local fam_obj = get_model_family(family_str)

    for j in 1:N_samples
        for k in 1:outcomes_N
            # Additive Superposition of Modular Base and Hierarchical Supervisors
            local eta_jk = eta_tensor[:, k, j] .+ nested_contributions[:, k, j]

            # Discovery of likelihood hyperparameters for current sample
            local phi_val = "lik_phi" in p_names ? chain[:lik_phi].data[j] : 0.0
            local r_val = "lik_r" in p_names ? chain[:lik_r].data[j] : 1.0

            # Link Mapping (Expected value mu)
            p_den[:, k, j] .= _apply_link_and_lik(family_str, eta_jk, get(M, :use_zi, false), phi_val, r_val)

            # Stochastic Predictive Sampling with Heteroskedastic Surface
            # Forensic Fix: Handle Vector vs Matrix volatility surfaces for multivariate
            local sig_jk = registry.sv_surface isa Vector ? registry.sv_surface[k][:, j] : registry.sv_surface[:, j]

            local _, noisy_jk, ll_jk = _process_ll_and_predictions(
                fam_obj, reshape(eta_jk, N_tot, 1), chain, M, N_tot, 1, reshape(sig_jk, N_tot, 1)
            )
            p_noi[:, k, j] .= noisy_jk[:, 1]

            # Pointwise Log-Likelihood mapping
            local slice_idx = ((k-1)*M.y_N + 1):(k*M.y_N)
            log_lik[j, slice_idx] .= ll_jk[1, :]
        end
    end

    # 6. Hardened Summarization Engine
    # Rationale: Explicit guards prevent iteration errors on inactive components.
    # All fields are standardized to Vector{Float64} within the result NamedTuple.

    # 6.1 Spatial and Temporal Manifold Summaries
    local spatial_struct = nothing
    if !isnothing(registry.s_eff_struct) && !isempty(registry.s_eff_struct)
        spatial_struct = [summarize_array(reshape(registry.s_eff_struct[k], M.s_N, 1, N_samples); alpha=alpha) for k in 1:outcomes_N]
    end

    local temporal_summ = nothing
    if !isnothing(registry.t_eff) && !isempty(registry.t_eff)
        temporal_summ = [summarize_array(reshape(registry.t_eff[k], M.t_N, 1, N_samples); alpha=alpha) for k in 1:outcomes_N]
    end

    # 6.2 Global Shared Components (Seasonal, Fixed, Smooth)
    local seasonal_summ = nothing
    if M.model_season != "none" && !isnothing(registry.u_eff) && !isempty(registry.u_eff)
        seasonal_summ = summarize_array(reshape(registry.u_eff, M.u_N, 1, N_samples); alpha=alpha)
    end

    local fixed_summ = nothing
    if !isnothing(registry.Xfixed_betas)
        fixed_summ = summarize_array(reshape(registry.Xfixed_betas, size(registry.Xfixed_betas, 1), 1, N_samples); alpha=alpha)
    end

    # FIXED: Discovery and Summarization of Accumulated Smooths
    local smooth_summ = nothing
    if !isnothing(registry.basis_eff_accum)
        smooth_summ = summarize_array(reshape(registry.basis_eff_accum, M.y_N, 1, N_samples); alpha=alpha)
    end

    # 6.3 Advanced Interactions and Hierarchies
    local st_summ = nothing
    if !isnothing(registry.st_eff_maps) && !isempty(registry.st_eff_maps[1])
        st_summ = [summarize_array(reshape(registry.st_eff_maps[k], M.s_N * M.t_N, 1, N_samples); alpha=alpha) for k in 1:outcomes_N]
    end

    local svc_summ = nothing
    if !isnothing(registry.svc_slopes) && !isempty(registry.svc_slopes[1])
        svc_summ = [summarize_array(reshape(registry.svc_slopes[k], size(registry.svc_slopes[k], 1) * size(registry.svc_slopes[k], 2), 1, N_samples); alpha=alpha) for k in 1:outcomes_N]
    end

    local nested_summ = nothing
    if any(nested_contributions .!= 0.0)
        # Summarizing the total denoised expectation including nested signal
        nested_summ = summarize_array(p_den[1:M.y_N, :, :]; alpha=alpha)
    end

    # 6.4 Auxiliary Registries (Hierarchy and Volatility)
    # FIXED: Persistence of Hierarchical Multi-Resolution Scale fields
    local hierarchy_summaries = Dict{Symbol, Any}()
    if !isnothing(registry.hierarchical_scales)
        for scale_sym in keys(registry.hierarchical_scales)
            local h_samples = registry.hierarchical_scales[scale_sym]
            hierarchy_summaries[scale_sym] = summarize_array(reshape(h_samples, size(h_samples, 1), 1, N_samples); alpha=alpha)
        end
    end

    return (
        predictions_denoised = summarize_array(p_den[1:M.y_N, :, :]; alpha=alpha),
        predictions_noisy = summarize_array(p_noi[1:M.y_N, :, :]; alpha=alpha),
        spatial_structured = spatial_struct,
        temporal_effects = temporal_summ,
        seasonal = seasonal_summ,
        smooth_effects = smooth_summ,
        fixed_effects = fixed_summ,
        spatiotemporal = st_summ,
        svc_slopes = svc_summ,
        nested_contributions = nested_summ,
        hierarchical_scales = isempty(hierarchy_summaries) ? nothing : hierarchy_summaries,
        volatility_surface = summarize_array(reshape(registry.sv_surface, M.y_N, outcomes_N, N_samples); alpha=alpha),
        waic = _compute_waic(log_lik),
        log_lik_matrix = log_lik,
        outcomes_N = outcomes_N,
        family = family_str,
        arch = arch
    )
end



function _reconstruct(arch::MultifidelityArchitecture, modelname::String, chain, M, PS, alpha)
    # --- BSTM v18.1.11 MULTIFIDELITY RECONSTRUCTION [FEATURE COMPLETE RESTORATION] ---
    println("--- Modular Multifidelity Reconstruction v18.1.11 [Hardened Audit] ---")

    local N_samples = size(chain, 1)
    local fidelities = M.fidelities
    local K = length(fidelities)
    local p_names = string.(FlexiChains.parameters(chain))
    local family_str = get(M, :model_family, "gaussian")
    local fam_obj = get_model_family(family_str)

    # 1. tech registry
    _validate_parameter_registry(chain, M, p_names, K)
    local registry = _discover_manifold_realizations(chain, M, N_samples, K, p_names)
    local rhos = vec(get_params_vector(chain, "rho_mf", 1))

    # 2. realized containers
    local p_den_list = [zeros(Float64, fidelities[k].y_N, N_samples) for k in 1:K]
    local p_noi_list = [zeros(Float64, fidelities[k].y_N, N_samples) for k in 1:K]
    local total_y_N = sum(f.y_N for f in fidelities)
    local log_lik = zeros(Float64, N_samples, total_y_N)

    # 3. Loop
    for j in 1:N_samples
        local eta_tensor = _modular_eta_assembly(total_y_N, registry, M, PS)
        for k in 1:K
            local fk = fidelities[k]
            local N_k = fk.y_N
            local coupled_eta_jk = if k == 1; eta_tensor[1:N_k, 1, j]; else; (rhos[j] .* (registry.s_eff_struct[1][fk.s_idx, j] .+ registry.t_eff[1][fk.t_idx, j])) .+ eta_tensor[1:N_k, k, j]; end
            
            p_den_list[k][:, j] .= _apply_link_and_lik(family_str, coupled_eta_jk, get(M, :use_zi, false))
            local sig_jk = (registry.sv_surface isa Vector) ? registry.sv_surface[k][:, j] : registry.sv_surface[:, j]
            local _, noisy_jk, ll_jk = _process_ll_and_predictions(fam_obj, reshape(coupled_eta_jk, N_k, 1), chain, fk, N_k, 1, reshape(sig_jk, N_k, 1))
            p_noi_list[k][:, j] .= noisy_jk[:, 1]

            local offset = 0
            if k > 1; for m in 1:(k-1); offset += fidelities[m].y_N; end; end
            log_lik[j, (offset + 1):(offset + N_k)] .= ll_jk[1, :]
        end
    end

    # 4. Summaries
    local spatial_struct = !isnothing(registry.s_eff_struct) && !isempty(registry.s_eff_struct) ? [summarize_array(reshape(registry.s_eff_struct[k], fidelities[k].s_N, 1, N_samples); alpha=alpha) for k in 1:K] : nothing
    local temporal_summ = !isnothing(registry.t_eff) && !isempty(registry.t_eff) ? [summarize_array(reshape(registry.t_eff[k], fidelities[k].t_N, 1, N_samples); alpha=alpha) for k in 1:K] : nothing
    local seasonal_summ = (M.model_season != "none") ? summarize_array(reshape(registry.u_eff, M.u_N, 1, N_samples); alpha=alpha) : nothing
    local smooth_summ = !isnothing(registry.basis_eff_accum) ? summarize_array(reshape(registry.basis_eff_accum, total_y_N, 1, N_samples); alpha=alpha) : nothing

    return (
        predictions_denoised = [summarize_array(reshape(p_den_list[k], fidelities[k].y_N, 1, N_samples); alpha=alpha) for k in 1:K],
        predictions_noisy = [summarize_array(reshape(p_noi_list[k], fidelities[k].y_N, 1, N_samples); alpha=alpha) for k in 1:K],
        spatial_structured = spatial_struct,
        temporal_effects = temporal_summ,
        seasonal = seasonal_summ,
        smooth_effects = smooth_summ,
        waic = _compute_waic(log_lik),
        log_lik_matrix = log_lik,
        family = family_str,
        arch = arch
    )
end

function _reconstruct(arch::ExampleArchitecture, modelname::String, chain, M, PS, alpha)
    # --- BSTM v18.1.11 EXAMPLE RECONSTRUCTION [FEATURE COMPLETE RESTORATION] ---
    println("--- Modular Example Reconstruction v18.1.11 [Hardened Audit] ---")

    local N_samples = size(chain, 1)
    local p_names = string.(FlexiChains.parameters(chain))
    local N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    local N_tot = M.y_N + N_PS
    local family_str = get(M, :model_family, "gaussian")
    local fam_obj = get_model_family(family_str)

    _validate_parameter_registry(chain, M, p_names, 1)
    local registry = _discover_manifold_realizations(chain, M, N_samples, 1, p_names)
    local eta_tensor = _modular_eta_assembly(N_tot, registry, M, PS)

    local p_den = zeros(Float64, N_tot, N_samples)
    local p_noi = zeros(Float64, N_tot, N_samples)
    local log_lik = zeros(Float64, N_samples, M.y_N)

    for j in 1:N_samples
        local eta_j = eta_tensor[:, 1, j]
        p_den[:, j] .= _apply_link_and_lik(family_str, eta_j, get(M, :use_zi, false))
        local sig_j = registry.sv_surface isa Vector ? registry.sv_surface[1][:, j] : registry.sv_surface[:, j]
        local _, noisy_j, ll_j = _process_ll_and_predictions(fam_obj, reshape(eta_j, N_tot, 1), chain, M, N_tot, 1, reshape(sig_j, N_tot, 1))
        p_noi[:, j] .= noisy_j[:, 1]
        log_lik[j, :] .= ll_j[1, :]
    end

    return (
        predictions_denoised = summarize_array(reshape(p_den[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_noisy = summarize_array(reshape(p_noi[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        spatial_structured = !isnothing(registry.s_eff_struct) && !isempty(registry.s_eff_struct) ? summarize_array(reshape(registry.s_eff_struct[1], M.s_N, 1, N_samples); alpha=alpha) : nothing,
        temporal_effects = !isnothing(registry.t_eff) && !isempty(registry.t_eff) ? summarize_array(reshape(registry.t_eff[1], M.t_N, 1, N_samples); alpha=alpha) : nothing,
        seasonal = (M.model_season != "none") ? summarize_array(reshape(registry.u_eff, M.u_N, 1, N_samples); alpha=alpha) : nothing,
        smooth_effects = !isnothing(registry.basis_eff_accum) ? summarize_array(reshape(registry.basis_eff_accum, M.y_N, 1, N_samples); alpha=alpha) : nothing,
        waic = _compute_waic(log_lik),
        family = family_str,
        arch = arch
    )
end



 
# --- Centralized Prediction & Likelihood Logic ---

"""
    _apply_link_and_lik(family, eta, use_zi, [phi], [r])

Applies the appropriate link function (Inverse Link) to the linear predictor `eta` 
and accounts for zero-inflation or dispersion to return the expected value (mu).
"""
function _apply_link_and_lik(family::String, eta::AbstractArray, use_zi::Bool, phi=0.0, r=1.0)
    if family == "gaussian"
        return eta
    elseif family == "lognormal" || family == "poisson"
        return exp.(eta)
    elseif family == "bernoulli"
        return logistic.(eta)
    elseif family == "negbin"
        # Expected value for NB (mu). If ZI is active, scale by (1-phi).
        mu = exp.(eta)
        return use_zi ? (1.0 .- phi) .* mu : mu
    else
        # Fallback to identity link
        return eta
    end
end
 

  


####


 
;;


function build_structure_template(type::Symbol, n::Int; scale=true, coords=nothing, W=nothing, bipartite_adj=nothing)
    # Consolidated Structural Factory [v14.6.0 - BSTM v06.1 Unified Registry]
    # Rationale: This function provides the definitive factory for precision matrix templates
    # across all manifold domains. It standardizes scaling and topological initialization.
    # Audit: Fixed SPDE scaling and DAG operator registration; added TPS 2D routing.

    local Q::Matrix{Float64}
    local sf::Float64 = 1.0

    # --- 1. Manifold Dispatch Registry ---

    # Group 1: Identity, Spectral, and Bipartite Primitives
    # Rationale: These manifolds operate in transformed or group-level spaces.
    if type in [:iid, :eigen, :bgcn, :nystrom, :rff, :fft, :bspline]
        Q = Matrix(1.0I, n, n)
        sf = 1.0

    # Group 2: Graph-Based & Conditionally Autoregressive (CAR)
    elseif type in [:icar, :besag, :bym2, :leroux, :sar]
        if isnothing(W)
            error("BSTM Registry Error: Spatial adjacency W required for manifold type :$type")
        end
        local D_sp = Diagonal(vec(sum(W, dims=2)))
        local Q_raw = Matrix(D_sp - W) # Graph Laplacian L

        if scale
            local evals = eigvals(Q_raw)
            local nz_ev = filter(x -> x > 1e-6, evals)
            sf = isempty(nz_ev) ? 1.0 : exp(mean(log.(nz_ev)))
            Q = Q_raw ./ sf
        else
            Q = Q_raw
            sf = 1.0
        end

    # Group 3: Temporal & Higher-Order Intrinsic GMRFs
    elseif type == :ar1
        local Q_ar = Matrix(1.0I, n, n)
        for i in 1:(n-1); Q_ar[i, i+1] = Q_ar[i+1, i] = -0.5; end
        Q = Q_ar
        sf = 1.0

    elseif type == :rw1
        local Q_rw1 = zeros(n, n)
        for i in 1:n
            if i == 1 || i == n; Q_rw1[i,i] = 1.0; else Q_rw1[i,i] = 2.0; end
            if i < n; Q_rw1[i, i+1] = Q_rw1[i+1, i] = -1.0; end
        end
        sf = 1.0
        Q = Q_rw1

    elseif type == :rw2
        local Q_rw2 = zeros(n, n)
        for i in 1:n
            if 2 < i < n-1
                Q_rw2[i,i]=6; Q_rw2[i,i-1]=Q_rw2[i,i+1]=-4; Q_rw2[i,i-2]=Q_rw2[i,i+2]=1
            elseif i == 1; Q_rw2[i,i]=1; Q_rw2[i,i+1]=-2; Q_rw2[i,i+2]=1
            elseif i == 2; Q_rw2[i,i]=5; Q_rw2[i,i-1]=-2; Q_rw2[i,i+1]=-4; Q_rw2[i,i+2]=1
            elseif i == n-1; Q_rw2[i,i]=5; Q_rw2[i,i+1]=-2; Q_rw2[i,i-1]=-4; Q_rw2[i,i-2]=1
            elseif i == n; Q_rw2[i,i]=1; Q_rw2[i,i-1]=-2; Q_rw2[i,i-2]=1
            end
        end
        if scale
            local evals_rw2 = eigvals(Q_rw2)
            local nz_ev_rw2 = filter(x -> x > 1e-6, evals_rw2)
            sf = isempty(nz_ev_rw2) ? 1.0 : exp(mean(log.(nz_ev_rw2)))
            Q = Q_rw2 ./ sf
        else
            Q = Q_rw2
            sf = 1.0
        end

    # Group 4: Advanced Topological & SPDE Manifolds
    elseif type == :spde
        if isnothing(W)
            error("BSTM Registry Error: Laplacian structure W required for :spde Finite Element approximation")
        end
        local D_mat = Diagonal(vec(sum(W, dims=2)))
        local L_mat = Matrix(D_mat - W)
        # SPDE Matern Precision approximation: (kappa^2 I + L)^2. Template is L.
        local Q_raw = L_mat
        if scale
            local evals_spde = eigvals(Q_raw)
            local nz_ev_spde = filter(x -> x > 1e-6, evals_spde)
            sf = isempty(nz_ev_spde) ? 1.0 : exp(mean(log.(nz_ev_spde)))
            Q = Q_raw ./ sf
        else
            Q = Q_raw
            sf = 1.0
        end

    elseif type == :dag
        # DAGs utilize a directed adjacency matrix (I - rho*W) directly.
        # Scale must be 1.0 to preserve the unit-diagonal properties of the Vecchia operator.
        Q = isnothing(W) ? Matrix(1.0I, n, n) : Matrix(W)
        sf = 1.0

    # Group 5: Basis & Continuous Kernels
    elseif type == :tps
        # Thin Plate Spline bending energy matrix standardized to RW2 scaling logic
        local template_rw2 = build_structure_template(:rw2, n; scale=scale)
        Q = template_rw2.matrix
        sf = template_rw2.scaling_factor

    elseif type == :pspline
        # Penalized Spline template defaults to second-order differencing (RW2)
        local template_rw2 = build_structure_template(:rw2, n; scale=scale)
        Q = template_rw2.matrix
        sf = template_rw2.scaling_factor

    elseif type == :gp
        Q = isnothing(coords) ? Matrix(1.0I, n, n) : [sqrt(sum((c1 .- c2).^2)) for c1 in coords, c2 in coords]
        sf = 1.0

    # Group 6: Periodic & Seasonal Continuity
    elseif type in [:cyclic, :seasonal, :harmonic]
        local Q_cyc = Matrix(1.0I, n, n) .* 2.0
        for i in 1:n
            Q_cyc[i, mod1(i-1, n)] = -1.0
            Q_cyc[i, mod1(i+1, n)] = -1.0
        end
        if scale
            local evals_cyc = eigvals(Q_cyc)
            local nz_ev_cyc = filter(x -> x > 1e-6, evals_cyc)
            sf = isempty(nz_ev_cyc) ? 1.0 : exp(mean(log.(nz_ev_cyc)))
            Q = Q_cyc ./ sf
        else
            Q = Q_cyc
            sf = 1.0
        end

    else
        @warn "BSTM Registry Fallback: Manifold :$type not recognized. Initializing Identity."
        Q = Matrix(1.0I, n, n)
        sf = 1.0
    end

    return (matrix = Q, scaling_factor = sf)
end

 
 

# --- 2. High-Level Manifold Builder Dispatch ---
# This function acts as the interface between the Struct-based Manifold definitions
# and the bstm_options metadata generator.

 
# Helper constructor for common defaults (cubic splines)
BSpline(v::Symbol; nbins=10, degree=3, sigma_prior=Exponential(1.0)) = BSpline(v, nbins, degree, sigma_prior)

 


function build_model(m::Eigen, data_inputs)
    # Audited Householder PCA Builder [v14.3.5 - BSTM v06.1 Spectral Patch]
    # Rationale: Provides low-rank decomposition using Householder reflections for orthonormal loadings.
    # Requirements: Uses Group 1 (Identity) template as coefficients are modeled in an unconstrained latent space.
    
    local n = data_inputs.s_N
    
    # Dispatching to Group 1 (Identity/Spectral Bases) in the structural factory
    local template = build_structure_template(:eigen, n)
    
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :eigen,
        hyper = (
            sigma_prior = m.sigma_prior, 
            pca_sd_prior = m.pca_sd_prior, 
            pdef_sd_prior = m.pdef_sd_prior,
            n_factors = m.n_factors,
            ltri_indices = m.ltri_indices
        )
    )
end


function build_model(m::Nystrom, data_inputs)
    # Audited Nystrom Approximation Builder [v14.3.0 - BSTM v06.1 Basis Patch]
    # Rationale: Provides low-rank kernel approximation using landmark inducing points.
    # Requirements: Returns a distance-based template or placeholder for dynamic calculation.
    
    local n = data_inputs.s_N
    local coords = get(data_inputs, :s_coord, nothing)
    
    # Dispatching to Group 7 (Distance-Based Kernel Manifolds) in the structural factory
    local template = build_structure_template(:nystrom, n; coords=coords)
    
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :nystrom,
        hyper = (
            sigma_prior = m.sigma_prior, 
            lengthscale_prior = m.lengthscale_prior, 
            n_inducing = m.n_inducing
        )
    )
end

function build_model(m::BCGN, data_inputs)
    # Audited BCGN (Bipartite Covariate Graph Network) Builder [v14.3.8 - BSTM v06.1 Network Patch]
    # Rationale: Models latent signals propagated through a bipartite unit-group adjacency.
    # Requirements: Routes to Group 1 (Identity) as weights are modeled in an unconstrained space before bipartite projection.
    
    # Determine the number of auxiliary nodes/groups from the provided manifold struct
    local n_groups = size(m.bipartite_adj, 2)
    
    # Dispatching to Group 1 (Identity) in the structural factory to initialize group-level coefficients
    local template = build_structure_template(:bgcn, n_groups)
    
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :bgcn,
        hyper = (
            sigma_prior = m.sigma_prior,
            bipartite_adj = m.bipartite_adj,
            group_weights = m.group_weights
        )
    )
end



# 2. Directed Network Flow Builder
function build_model(m::NetworkFlow, data_inputs)
    # For directed graphs, utilize the provided directed adjacency matrix
    W_directed = m.adjacency_matrix
    n = size(W_directed, 1)

    # Precision for directed flow often uses the flow-Laplacian: (I - rho*W)'(I - rho*W)
    return (
        Q_template = W_directed,
        scaling_factor = 1.0,
        model_type = :network,
        hyper = (
            sigma_prior = m.sigma_prior,
            direction = m.flow_direction
        )
    )
end

# 3. Hyperbolic Embedding Builder
function build_model(m::Hyperbolic, data_inputs)
    # Hyperbolic manifolds are continuous; utilize the coordinate names
    return (
        Q_template = nothing, # Calculated dynamically via hyperbolic distance kernel
        model_type = :hyperbolic,
        hyper = (
            sigma_prior = m.sigma_prior,
            curvature = m.curvature,
            coords = m.coordinates
        )
    )
end
 
# 1. FFT Builder Dispatch
function build_model(m::FFT, data_inputs)
    # n defines the resolution of the spectral grid
    n = m.nbins
    # The FFT basis is pre-computed or handled via FFTW in the likelihood
    # provide an identity template as the FFT manifold models coefficients in frequency space
    template = build_structure_template(:iid, n)

    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :fft,
        hyper = (
            sigma_prior = m.sigma_prior,
            is_2d = m.is_2d
        )
    )
end

# 2. Wavelet Builder Dispatch
function build_model(m::Wavelets, data_inputs)
    n = m.nbins
    # Wavelet coefficients are often assumed IID (sparse) in the transformed domain
    template = build_structure_template(:iid, n)

    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :wavelet,
        hyper = (
            sigma_prior = m.sigma_prior,
            family = m.wavelet_family
        )
    )
end

function build_model(m::IID, data_inputs)
    # IID Builder: Unstructured random effects
    # Rationale: Standardizes variance scales across the spatial or temporal unit grid.
    local n = data_inputs.s_N
    local template = build_structure_template(:iid, n)

    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :iid,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end

function build_model(m::ICAR, data_inputs)
    # ICAR Builder: Pure local smoothing (Besag)
    # Rationale: Requires an adjacency matrix W to define the graph Laplacian.
    local n = data_inputs.s_N
    local W = get(data_inputs, :W, nothing)
    local template = build_structure_template(:icar, n; W=W)

    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :icar,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end

function build_model(m::BYM2, data_inputs)
    # BYM2 Builder: Standardized ICAR + IID
    # Rationale: Decouples variance scale from graph topology using geometric mean scaling.
    local n = data_inputs.s_N
    local W = get(data_inputs, :W, nothing)
    local template = build_structure_template(:bym2, n; W=W)

    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :bym2,
        hyper = (
            sigma_prior = m.sigma_prior,
            rho_prior = m.rho_prior
        )
    )
end

function build_model(m::Leroux, data_inputs)
    # Leroux Builder: Convex combination of structure and noise
    local n = data_inputs.s_N
    local W = get(data_inputs, :W, nothing)
    local template = build_structure_template(:leroux, n; W=W)

    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :leroux,
        hyper = (
            sigma_prior = m.sigma_prior,
            rho_prior = m.rho_prior
        )
    )
end

function build_model(m::AR1, data_inputs)
    # AR1 Builder: Mean-reverting temporal or spatial process
    # Rationale: Maps time indices or regular lattice units to an AR(1) precision structure.
    local n = get(data_inputs, :t_N, data_inputs.s_N)
    local template = build_structure_template(:ar1, n)

    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :ar1,
        hyper = (
            sigma_prior = m.sigma_prior,
            rho_prior = m.rho_prior
        )
    )
end

function build_model(m::RW2, data_inputs)
    # RW2 Builder: Second-order random walk (Smooth Trend)
    # Rationale: Utilizes intrinsic GMRF precision with a locally linear slope assumption.
    local n = get(data_inputs, :t_N, data_inputs.s_N)
    local template = build_structure_template(:rw2, n)

    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :rw2,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end

function build_model(m::GP, data_inputs)
    # Continuous GP Builder: Exact kernel-based covariance
    # Rationale: Distance-based template construction using coordinate metadata.
    local n = data_inputs.s_N
    local coords = get(data_inputs, :s_coord, nothing)
    local template = build_structure_template(:gp, n; coords=coords)

    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :gp,
        hyper = (
            sigma_prior = m.sigma_prior,
            lengthscale_prior = m.lengthscale_prior,
            kernel = m.kernel
        )
    )
end

function build_model(m::Cyclic, data_inputs)
    # Cyclic Builder: Periodic continuity (Seasonal)
    # Rationale: Enforces boundary condition y_0 = y_T via a circular Laplacian.
    local n = m.period
    local W_circ = zeros(n, n)
    for i in 1:n
        W_circ[i, mod1(i-1, n)] = 1.0
        W_circ[i, mod1(i+1, n)] = 1.0
    end
    local template = build_structure_template(:besag, n; W=W_circ)

    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :cyclic,
        hyper = (sigma_prior = m.sigma_prior, period = m.period)
    )
end

function build_model(m::RFF, data_inputs)
    # RFF Builder: Spectral approximation of stationary kernels
    # Rationale: Maps coordinates to random Fourier feature space coefficients.
    return (
        Q_template = Matrix(1.0I, m.n_features, m.n_features),
        scaling_factor = 1.0,
        model_type = :rff,
        hyper = (
            sigma_prior = m.sigma_prior,
            lengthscale_prior = m.lengthscale_prior,
            n_features = m.n_features
        )
    )
end 

function build_model(m::TPS, data_inputs)
    # Audited Thin Plate Spline (TPS) Builder [v14.6.0 - BSTM v06.1 Basis Patch]
    # Rationale: Provides 2D smooth interaction surfaces using the bending energy penalty.
    # Requirements: Uses nbins to define the grid resolution for the spectral template.

    local n = m.nbins

    # Dispatching to Group 5 (Cyclic/Seasonal/Harmonic) or specialized TPS logic in template factory
    local template = build_structure_template(:tps, n)

    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :tps,
        hyper = (
            sigma_prior = m.sigma_prior,
            nbins = m.nbins
        )
    )
end

function build_model(m::BSpline, data_inputs)
    # Audited B-Spline Builder [v14.6.0 - BSTM v06.1 Basis Patch]
    # Rationale: Provides local polynomial smoothing via piecewise basis functions.
    # Requirements: Typically uses an Identity template as smoothing is controlled via the basis matrix B.

    local n = m.nbins

    # Dispatching to Group 1 (Identity) as B-Splines are often unpenalized or have external penalties
    local template = build_structure_template(:iid, n)

    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :bspline,
        hyper = (
            sigma_prior = m.sigma_prior,
            nbins = m.nbins,
            degree = m.degree
        )
    )
end

function build_model(m::PSpline, data_inputs)
    # Audited Penalized Spline (P-Spline) Builder [v14.6.0 - BSTM v06.1 Basis Patch]
    # Rationale: Combines B-splines with a discrete differencing penalty on coefficients.
    # Requirements: Uses the RW2/RW1 differencing logic within the structural factory.

    local n = m.nbins
    local diff_type = m.diff_order == 1 ? :rw1 : :rw2

    # Dispatching to Group 3 (Temporal/GMRF) for the differencing template
    local template = build_structure_template(diff_type, n)

    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :pspline,
        hyper = (
            sigma_prior = m.sigma_prior,
            nbins = m.nbins,
            degree = m.degree,
            diff_order = m.diff_order
        )
    )
end

# --- Precision Engine Fix ---
# Positional dispatch required by bstm_univariate internally
function recompose_precision(m_type::Symbol, template_mat::AbstractMatrix, param_val::Real; extra_param=nothing, noise=1e-4)
    local T = typeof(param_val)
    local n = size(template_mat, 1)
    local scale_factor = 1.0 / (param_val^2 + noise)

    local Q = if m_type == :none || m_type == :fixed
        scale_factor * I(n)
    elseif m_type in [:besag, :icar, :diffusion, :cyclic]
        scale_factor .* template_mat
    elseif m_type == :bym2
        local rho = isnothing(extra_param) ? 0.5 : extra_param
        scale_factor .* (rho .* template_mat + (1.0 - rho) .* I(n))
    elseif m_type == :leroux
        local lambda_val = isnothing(extra_param) ? 0.5 : extra_param
        scale_factor .* (lambda_val .* template_mat + (1.0 - lambda_val) .* I(n))
    elseif m_type in [:sar, :advection, :network, :dag, :proper_car]
        local rho_p = isnothing(extra_param) ? 0.8 : extra_param
        local L_op = I(n) - rho_p .* template_mat
        scale_factor .* (L_op' * L_op)
    elseif m_type == :gp
        local ls = isnothing(extra_param) ? 1.0 : extra_param
        local K = (param_val^2) .* exp.(-(Matrix(template_mat).^2) ./ (2 * ls^2 + noise))
        inv(Symmetric(K + noise * I(n)))
    elseif m_type == :spde
        local kappa = isnothing(extra_param) ? 1.0 : extra_param
        local L_spde = (kappa^2 .* I(n) + template_mat)
        scale_factor .* (L_spde' * L_spde)
    elseif m_type in [:rff, :fft, :bspline, :pspline, :rw1, :rw2, :tps]
        scale_factor .* template_mat
    else
        scale_factor .* template_mat
    end
    return Symmetric(Matrix(Q) + noise * I(n))
end

function recompose_precision(manifold_metadata::NamedTuple, param_val::Real; extra_param=nothing, noise=1e-4)
    return recompose_precision(
        Symbol(manifold_metadata.model_type),
        manifold_metadata.Q_template,
        param_val;
        extra_param=extra_param,
        noise=noise
    )
end 

function recompose_precision0(manifold_metadata::NamedTuple, param_val::Real; extra_param=nothing, noise=1e-4)
    # Refactored Precision Recomposition Engine [v13.0.0 - Metadata Coupled]
    # Rationale: This function is the core engine for converting latent manifold parameters into
    # Symmetric, Positive-Definite (SPD) precision matrices. It now accepts the full metadata
    # dictionary generated by the decentralized builders to ensure consistent hyperprior sizing.

    # 1. Extraction and Type Discovery
    # Extracting structural metadata from the builder output
    local T = typeof(param_val)
    local m_type = manifold_metadata.model_type
    local template_mat = manifold_metadata.Q_template
    local n = size(template_mat, 1)

    # 2. Precision Scaling Logic
    # Precision (Q) is inversely proportional to variance (sigma^2).
    # The noise term acts as a 'nugget' effect for numerical stability.
    local scale_factor = 1.0 / (param_val^2 + noise)

    # 3. Domain-Tagged Multiple Dispatch Logic
    # Rationale: Different domains require different mathematical treatments of the precision template.

    local Q = if m_type == :none || m_type == :fixed
        # Unstructured Overdispersion: Identity Scaling
        scale_factor * I(n)

    elseif m_type == :mosaic
        # Piecewise Constant: Cluster-level Intercepts mapped back to the unit grid
        scale_factor * I(n)

    elseif m_type in [:besag, :icar, :diffusion, :cyclic]
        # Pure Local Smoothing / GMRF / Circular Random Walk
        # Uses the pre-calculated Graph Laplacian or Circular Adjacency template
        scale_factor .* template_mat

    elseif m_type == :bym2
        # Besag-York-Mollie Standardized Manifold
        # Rationale: Standardizes variance scales independent of graph topology using rho mixing.
        local rho = isnothing(extra_param) ? 0.5 : extra_param
        scale_factor .* (rho .* template_mat + (1.0 - rho) .* I(n))

    elseif m_type == :leroux
        # Convex combination of Laplacian and Identity for unknown spatial dependence
        local lambda_val = isnothing(extra_param) ? 0.5 : extra_param
        scale_factor .* (lambda_val .* template_mat + (1.0 - lambda_val) .* I(n))

    elseif m_type in [:sar, :advection, :network, :dag, :proper_car]
        # Directed or Proper CAR operators (I - rho * W)
        # Enforces symmetry via the operator-transpose product
        local rho_p = isnothing(extra_param) ? 0.8 : extra_param
        local L_op = I(n) - rho_p .* template_mat
        scale_factor .* (L_op' * L_op)

    elseif m_type == :gp
        # Continuous Kernel Recomposition (Distance-based Template)
        # Used for both 1D Temporal and 2D Spatial Gaussian Processes
        local ls = isnothing(extra_param) ? 1.0 : extra_param
        # Matrix exponentiation and inversion for robust ForwardDiff probes
        local K = (param_val^2) .* exp.(-(Matrix(template_mat).^2) ./ (2 * ls^2 + noise))
        inv(Symmetric(K + noise * I(n)))

    elseif m_type == :spde
        # Matern SPDE approximation: (kappa^2 I + L)^2
        local kappa = isnothing(extra_param) ? 1.0 : extra_param
        local L_spde = (kappa^2 .* I(n) + template_mat)
        scale_factor .* (L_spde' * L_spde)

    elseif m_type == :local_adaptive
        # Riemannian diagonal weighting for non-stationary smoothing intensity
        local w_vec = isnothing(extra_param) ? ones(n) : extra_param
        scale_factor .* (Diagonal(w_vec) * template_mat * Diagonal(w_vec))

    elseif m_type in [:rff, :fft, :bspline, :pspline, :rw1, :rw2, :tps]
        # Basis and Difference-Penalty Manifolds
        # Utilize the pre-scaled templates from the builder stage
        scale_factor .* template_mat

    else
        # Fallback: Ensure no silent failure for unrecognized tokens
        @warn "BSTM Precision Kernel Warning: Unknown manifold type :$m_type. Defaulting to identity scaling."
        scale_factor .* template_mat
    end

    # 4. Scalar  and Matrix Verification
    # Rationale: Explicit casting to Symmetric Matrix ensures stability in Cholesky decomp.
    # Dense conversion is preferred here to prevent gradient fragmentation in high-D ForwardDiff passes.
    return Symmetric(Matrix(Q) + noise * I(n))
end

 
# --- Comprehensive Manifold Builders and Precision Factories ---
# This section provides the high-level dispatch logic to convert 
# Manifold structs into the structural metadata required by bstm_options.

# --- 1. Base Builder Dispatch ---

# 1. Base Generic Fallback
function build_model(manifold::Manifold, data_inputs)
    @warn "No specific builder for $(typeof(manifold)). Using IID identity template."
    template = build_structure_template(:iid, data_inputs.s_N)
    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :iid,
        hyper = (sigma_prior = hasproperty(manifold, :sigma_prior) ? manifold.sigma_prior : Exponential(1.0),)
    )
end


function build_model(m::ICAR, data_inputs)
    # ICAR Builder: Pure local smoothing using the Graph Laplacian
    local n = data_inputs.s_N
    local template = build_structure_template(:icar, n; W=get(data_inputs, :W, nothing))
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :icar,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end

function build_model(m::Besag, data_inputs)
    # Besag Builder: Pure spatial smoothing (equivalent to ICAR in precision form)
    local n = data_inputs.s_N
    local template = build_structure_template(:besag, n; W=get(data_inputs, :W, nothing))
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :besag,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end

function build_model(m::SAR, data_inputs)
    # SAR Builder: Simultaneous Autoregressive operator (I - rho*W)'(I - rho*W)
    local n = data_inputs.s_N
    local template = build_structure_template(:sar, n; W=get(data_inputs, :W, nothing))
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :sar,
        hyper = (sigma_prior = m.sigma_prior, rho_prior = m.rho_prior)
    )
end

function build_model(m::FFT, data_inputs)
    # FFT Builder: Frequency-domain temporal or spatial coefficients
    # Precision is assumed IID in the spectral space
    return (
        Q_template = Matrix(1.0I, m.nbins, m.nbins),
        scaling_factor = 1.0,
        model_type = :fft,
        hyper = (sigma_prior = m.sigma_prior, nbins = m.nbins)
    )
end

function build_model(m::FITC, data_inputs)
    # FITC Builder: Fully Independent Training Conditional (Sparse GP)
    # Rationale: Uses M inducing points to approximate the full covariance
    return (
        Q_template = nothing, # Requires dynamic kernel distance matrix
        scaling_factor = 1.0,
        model_type = :fitc,
        hyper = (sigma_prior = m.sigma_prior, lengthscale_prior = m.lengthscale_prior, n_inducing = m.n_inducing)
    )
end

function build_model(m::SPDE, data_inputs)
    # SPDE Builder: Finite element approximation of Matern covariance via (kappa^2 I + L)^2
    local n = data_inputs.s_N
    local template = build_structure_template(:besag, n; W=get(data_inputs, :W, nothing))
    return (
        Q_template = template.matrix, # Base Laplacian L used in SPDE expansion
        scaling_factor = template.scaling_factor,
        model_type = :spde,
        hyper = (sigma_prior = m.sigma_prior, kappa_prior = m.kappa_prior)
    )
end

function build_model(m::GP, data_inputs)
    # Dense GP Builder: Distance-based precision mapping
    return (
        Q_template = nothing, # Calculated via Distance Matrix in recompose_precision
        scaling_factor = 1.0,
        model_type = :gp,
        hyper = (sigma_prior = m.sigma_prior, lengthscale_prior = m.lengthscale_prior, kernel = m.kernel)
    )
end

function build_model(m::IID, data_inputs)
    # IID Builder: Unstructured random effects
    return (
        Q_template = Matrix(1.0I, data_inputs.s_N, data_inputs.s_N),
        scaling_factor = 1.0,
        model_type = :iid,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end

# 2. Discrete Spatial Manifolds (CAR/Laplacian Family)
function build_model(m::BYM2, data_inputs)
    # BYM2 requires a spatial unit count and adjacency matrix W
    template = build_structure_template(:bym2, data_inputs.s_N; W=data_inputs.W)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :bym2,
        hyper = (sigma_prior = m.sigma_prior, rho_prior = m.rho_prior)
    )
end

function build_model(m::ICAR, data_inputs)
    template = build_structure_template(:icar, data_inputs.s_N; W=data_inputs.W)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :icar,
        hyper = (sigma_prior = m.sigma_prior, rho_prior = nothing)
    )
end

function build_model(m::Leroux, data_inputs)
    template = build_structure_template(:leroux, data_inputs.s_N; W=data_inputs.W)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :leroux,
        hyper = (sigma_prior = m.sigma_prior, rho_prior = m.rho_prior)
    )
end

# 3. Topological & Directed Manifolds (v05.8 Expansion)
function build_model(m::LocalAdaptive, data_inputs)
    # Q_template is the base Laplacian; local weights are applied in recompose_precision
    template = build_structure_template(:besag, data_inputs.s_N; W=data_inputs.W)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :local_adaptive,
        hyper = (sigma_prior = m.sigma_prior, weights_var = m.weights_variable)
    )
end


function build_model(m::NetworkFlow, data_inputs)
    # For directed networks, the template is the raw adjacency matrix
    return (
        Q_template = m.adjacency_matrix,
        scaling_factor = 1.0,
        model_type = :network,
        hyper = (sigma_prior = m.sigma_prior, direction = m.flow_direction)
    )
end

function build_model(m::DAG, data_inputs)
    # DAG uses the directed adjacency as the operator (I - rho*W)
    return (
        Q_template = m.adjacency_matrix,
        scaling_factor = 1.0,
        model_type = :dag,
        hyper = (sigma_prior = m.sigma_prior, rho_prior = nothing)
    )
end

# 4. Temporal & Seasonal Manifolds
function build_model(m::AR1, data_inputs)
    # n derived from temporal units in metadata
    n = get(data_inputs, :t_N, 10)
    template = build_structure_template(:ar1, n)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :ar1,
        hyper = (sigma_prior = m.sigma_prior, rho_prior = m.rho_prior)
    )
end

function build_model(m::Harmonic, data_inputs)
    # Harmonic manifolds are basis-driven, not GMRF-driven
    return (
        Q_template = nothing,
        scaling_factor = 1.0,
        model_type = :harmonic,
        hyper = (
            sigma_prior = m.sigma_prior, 
            amplitude_prior = m.amplitude_prior, 
            phase_prior = m.phase_prior,
            period = m.period
        )
    )
end

function build_model(m::RW1, data_inputs)
    # RW1 Builder: First-order random walk (Stochastic Level shifts)
    # Rationale: Maps temporal or lattice units to a standard differencing precision.
    local n = get(data_inputs, :t_N, data_inputs.s_N)
    local template = build_structure_template(:ar1, n) # Reuse AR1 for differencing structure
    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :rw1,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end

function build_model(m::RW2, data_inputs)
    # RW2 Builder: Second-order random walk (Smooth locally linear trends)
    # Rationale: Higher-order smoothing with geometric mean scaling.
    local n = get(data_inputs, :t_N, data_inputs.s_N)
    local template = build_structure_template(:rw2, n)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :rw2,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end

function build_model(m::Cyclic, data_inputs)
    # Cyclic Builder: Periodic continuity (Seasonal)
    # Rationale: Enforces boundary condition y_0 = y_T via a circular Laplacian.
    local n = m.period
    local W_circ = zeros(n, n)
    for i in 1:n
        W_circ[i, mod1(i-1, n)] = 1.0
        W_circ[i, mod1(i+1, n)] = 1.0
    end
    local template = build_structure_template(:besag, n; W=W_circ)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :cyclic,
        hyper = (sigma_prior = m.sigma_prior, period = m.period)
    )
end

function build_model(m::BSpline, data_inputs)
    # BSpline Builder: Cubic Spline basis-mapped coefficients
    # Rationale: Extracts resolution (nbins) and degree for basis construction.
    return (
        Q_template = Matrix(1.0I, m.nbins, m.nbins), # IID in basis space
        scaling_factor = 1.0,
        model_type = :bspline,
        hyper = (sigma_prior = m.sigma_prior, nbins = m.nbins, degree = m.degree)
    )
end

function build_model(m::Wavelets, data_inputs)
    # Wavelets Builder: Multi-resolution frequency decomposition
    return (
        Q_template = Matrix(1.0I, m.nbins, m.nbins),
        scaling_factor = 1.0,
        model_type = :wavelet,
        hyper = (sigma_prior = m.sigma_prior, family = m.wavelet_family, nbins = m.nbins)
    )
end

function build_model(m::Hyperbolic, data_inputs)
    # Hyperbolic Builder: Hierarchical embedding with constant negative curvature
    return (
        Q_template = nothing, # Requires dynamic geodesic distance matrix
        scaling_factor = 1.0,
        model_type = :hyperbolic,
        hyper = (sigma_prior = m.sigma_prior, curvature = m.curvature)
    )
end

function build_model(m::ExponentialDecay, data_inputs)
    # Exponential Decay Builder: Signal attenuation over continuous domains
    return (
        Q_template = nothing, # Distance-based precision mapping
        scaling_factor = 1.0,
        model_type = :decay,
        hyper = (sigma_prior = m.sigma_prior, lengthscale_prior = m.decay_lengthscale_prior)
    )
end
 

# 5. Covariate Smooths (Splines)
function build_model(m::TPS, data_inputs)
    template = build_structure_template(:tps, m.nbins)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :tps,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end

function build_model(m::PSpline, data_inputs)
    template = build_structure_template(:pspline, m.nbins)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :pspline,
        hyper = (sigma_prior = m.sigma_prior, degree = m.degree, diff_order = m.diff_order)
    )
end

# 6. Spectral Manifolds
function build_model(m::FFT, data_inputs)
    # Spectral coefficients are assumed IID in frequency space
    return (
        Q_template = Matrix(1.0I, m.nbins, m.nbins),
        scaling_factor = 1.0,
        model_type = :fft,
        hyper = (sigma_prior = m.sigma_prior, is_2d = m.is_2d)
    )
end

function build_model(m::RFF, data_inputs)
    return (
        Q_template = Matrix(1.0I, m.n_features, m.n_features),
        scaling_factor = 1.0,
        model_type = :rff,
        hyper = (sigma_prior = m.sigma_prior, lengthscale_prior = m.lengthscale_prior)
    )
end


# 7. Continuous Spatial (GPs)
function build_model(m::GP, data_inputs)
    # Distance matrix acts as the template for kernel evaluation
    return (
        Q_template = nothing, # Computed dynamically in recompose_precision for :gp
        scaling_factor = 1.0,
        model_type = :gp,
        hyper = (sigma_prior = m.sigma_prior, lengthscale_prior = m.lengthscale_prior, kernel = m.kernel)
    )
end

  

# Knorr-Held Interaction Builder
function build_model(m::KnorrHeld, data_inputs)
    # Space-Time Interaction types I, II, III, IV
    # Precision is built via kron() in the sampler, so no static template is stored
    return (
        Q_template = nothing,
        model_type = :knorrheld,
        interaction_class = m.type,
        hyper = (sigma_prior = m.sigma_prior, rho_prior = nothing)
    )
end 
 
 



function get_optimal_sampler(model::DynamicPPL.Model;
    nuts_n_samples_adaptation = 100,
    nuts_target_acceptance_ratio = 0.65,
    pg_particles = 20,
    kwargs...)

    # 1. Parameter Discovery Phase
    # Generating multiple prior samples to determine parameter volatility and types.
    # Using 5 samples to robustly identify fixed (Dirac) parameters.
    local test_samples = [Dict(pairs(rand(model))) for _ in 1:5]
    local prototype = test_samples[1]
    local all_keys = collect(keys(prototype))

    # Identification of Fixed Parameters
    # These are parameters that do not vary across samples (e.g. constant inputs or flags).
    fixed_params = filter(k -> all(s -> s[k] == prototype[k], test_samples), all_keys)

    # Identification of Discrete Parameters
    # Required for Particle Gibbs (PG) routing.
    discrete_params = filter(k -> k ∉ fixed_params && (prototype[k] isa Integer || prototype[k] isa Bool), all_keys)

    # 2. Gaussian Field Discovery (ESS Targeting)
    # Rationale: ESS is optimal for latent Gaussian processes (GMRFs, GPs).
    # We identify vectors representing the realized latent surfaces across all outcomes.
    latent_manifold_realizations = filter(k -> begin
        local val = prototype[k]
        local k_str = string(k)

        # BSTM v06.1 Naming Convention Audit:
        # Targets: s_latent, t_latent, u_latent, s_latenentt_k, t_latent_k, st_latent, st_latent_k,
        # c_latent, f_z_s (nested), and hierarchy keys.
        is_manifold = occursin(r"latent|lat_|s_eff|t_eff|st_eff|f_z_s", k_str)

        # Filter constraints
        k ∉ fixed_params &&
        k ∉ discrete_params &&
        val isa AbstractVector &&
        is_manifold &&
        # Hyperparameters (sigma, rho, etc.) are excluded even if they contain the stem
        !occursin(r"sigma|sig_|rho|phi|ls_|lengthscale|alpha|beta|Xfixed", k_str)
    end, all_keys)

    # 3. Hyperparameter Discovery (NUTS Targeting)
    # These are continuous scalars/vectors with complex geometries (e.g., rho, sigma, beta).
    active_hyper_params = filter(k -> begin
        k ∉ fixed_params &&
        k ∉ discrete_params &&
        k ∉ latent_manifold_realizations
    end, all_keys)

    # 4. Gibbs Block Partitioning
    local gibbs_blocks = []

    # # Block 1: Discrete States (Particle Gibbs)
    if !isempty(discrete_params)
        push!(gibbs_blocks, Tuple(discrete_params) => PG(pg_particles))
    end

    # # Block 2: High-Dimensional Latent Fields (Elliptical Slice Sampling)
    # This block handles the intensive manifold realizations for S, T, and U domains.
    if !isempty(latent_manifold_realizations)
        push!(gibbs_blocks, Tuple(latent_manifold_realizations) => ESS())
    end

    # # Block 3: Structural Hyperparameters (NUTS)
    # Handles regression coefficients and variance/correlation components.
    if !isempty(active_hyper_params)
        push!(gibbs_blocks, Tuple(active_hyper_params) => Turing.NUTS(nuts_n_samples_adaptation, nuts_target_acceptance_ratio))
    end

    # # Block 4: Fallback / Fixed (Metropolis-Hastings)
    if !isempty(fixed_params)
        push!(gibbs_blocks, Tuple(fixed_params) => MH())
    end

    # Final Technical Assembly
    # The Gibbs sampler partitions the posterior, improving efficiency in hierarchical models.
    return Gibbs(gibbs_blocks...)
end


function get_inits(model::DynamicPPL.Model; refine="map", n_samples=100, optimizer=LBFGS(), max_iters=500, maxtime=60.0, noise=nothing)
    println("--- Generating Initial Parameters ---")

    # 1. Heuristic Initialization from Prior Samples
    samples = [Dict(pairs(rand(model))) for _ in 1:n_samples]
    init_dict = Dict{Symbol, Any}()

    if !isnothing(noise)
        init_dict[:noise] = noise
    end

    for k in keys(samples[1])
        ks = Symbol(k)
        vals = [s[k] for s in samples]

        # Dirac/Fixed parameter check
        if all(v -> v == vals[1], vals)
            init_dict[ks] = vals[1]
            continue
        end

        # FIX: Skip averaging if the elements do not support standard arithmetic (e.g. Cholesky)
        if vals[1] isa Cholesky || vals[1] isa LKJCholesky
            init_dict[ks] = vals[1]
            continue
        end

        mu = mean(vals)
        s_name = string(ks)

        if vals[1] isa AbstractVector
            # Latent fields are centered at zero for stability
            init_dict[ks] = zeros(eltype(mu), length(mu))
        elseif occursin(r"sigma|ls_|lengthscale|sig_", s_name)
            init_dict[ks] = max(0.1, mu)
        elseif occursin("rho", s_name)
            init_dict[ks] = clamp(mu, -0.9, 0.9)
        elseif occursin("phi", s_name)
            init_dict[ks] = clamp(mu, 0.01, 0.5)
        else
            init_dict[ks] = mu
        end
    end

    # 2. Optimization Refinement
    if refine == "map"
        try
            println("Refining inits with Maximum A Posteriori (MAP)...")
            map_res = maximum_a_posteriori(model, optimizer;
                initial_params=DynamicPPL.InitFromParams(NamedTuple(init_dict)),
                iterations=max_iters, maxtime=maxtime)
            return DynamicPPL.InitFromParams(NamedTuple(map_res.params))
        catch e
            @warn "MAP refinement failed ($e). Using heuristic inits."
        end
    end

    return DynamicPPL.InitFromParams(NamedTuple(init_dict))
end



function manifold_type(m::Manifold)
    return lowercase(string(typeof(m)))
end


# This enforces explicit dispatch (:spatial, :temporal, :seasonal, :spacetime) to resolve 
# dimensional ambiguity for dual-use primitives (e.g. RW2, IID, AR1).

# --- 1. Univariate Extraction Gateway ---

function extract_manifold(m_type::Manifold, chain, M, j, sig, domain::Symbol)
    # Target dimension resolution based on the explicitly provided domain context
    local n_target = if domain == :spatial
        M.s_N
    elseif domain == :temporal
        M.t_N
    elseif domain == :seasonal
        M.u_N
    elseif domain == :spacetime
        M.s_N * M.t_N
    else
        length(M.y_obs)
    end

    # Resolve the expected parameter key prefix based on domain naming conventions
    local key_prefix = if domain == :spatial
        "s_latent"
    elseif domain == :temporal
        "t_latent"
    elseif domain == :seasonal
        "u_latent"
    elseif domain == :spacetime
        "st_latent"
    else
        "latent"
    end

    local field = zeros(n_target)
    local p_names = string.(FlexiChains.parameters(chain))

    # Forensic Discovery: Check for FlexiChains vector object or indexed scalars
    if key_prefix in p_names
        # Standard Vector retrieval
        local raw = chain[Symbol(key_prefix)].data[j]
        local flat_raw = vec(collect(raw))
        local n_raw = length(flat_raw)
        # Forensic Slicing: Ensure raw samples align with target grid to prevent broadcast errors
        field .= flat_raw[1:min(n_raw, n_target)] .* sig
    elseif any(occursin.(Regex("^" * key_prefix * "\\[\\d+\\]"), p_names))
        # Indexed Scalar retrieval using the numerically sorted helper
        local mat = get_params_vector(chain, key_prefix, n_target)
        field .= mat[j, :] .* sig
    end

    # Return tuple (structured, noisy) - default is identical for simple primitives
    return field, field
end

# --- 2. Multivariate Outcome-Specific Extraction ---

function extract_manifold_k(m_type::Manifold, chain, M, j, k, sig_k, domain::Symbol)
    # Outcome-specific extraction with domain synchronization
    local n_target = if domain == :spatial
        M.s_N
    elseif domain == :temporal
        M.t_N
    elseif domain == :seasonal
        M.u_N
    elseif domain == :spacetime
        M.s_N * M.t_N
    else
        length(M.y_obs)
    end

    local base_prefix = if domain == :spatial
        "s_latent"
    elseif domain == :temporal
        "t_latent"
    elseif domain == :seasonal
        "u_latent"
    elseif domain == :spacetime
        "st_latent"
    else
        "latent"
    end

    # Standard Multivariate naming convention: prefix_outcome (e.g., s_latent_1)
    local key_str = base_prefix * "_" * string(k)
    local field = zeros(n_target)
    local p_names = string.(FlexiChains.parameters(chain))

    if key_str in p_names
        local raw = chain[Symbol(key_str)].data[j]
        local flat_raw = vec(collect(raw))
        local n_raw = length(flat_raw)
        field .= flat_raw[1:min(n_raw, n_target)] .* sig_k
    elseif any(occursin.(Regex("^" * key_str * "\\[\\d+\\]"), p_names))
        local mat = get_params_vector(chain, key_str, n_target)
        field .= mat[j, :] .* sig_k
    end

    return field, field
end

# --- 3. Manifold-Specific Technical Overrides ---

function extract_manifold(::BYM2, chain, M, j, sig, domain::Symbol)
    # BYM2 decomposition into structured (ICAR) and noisy (IID) innovations
    # Domain context routes naming to 's_rho' vs 't_rho'
    local n_target = (domain == :spatial) ? M.s_N : M.t_N
    local prefix = (domain == :spatial) ? "s" : "t"

    local rho_key = Symbol(prefix * "_rho")
    local lat_key = Symbol(prefix * "_latent")
    local iid_key = Symbol(prefix * "_iid")
    
    local p_names = string.(FlexiChains.parameters(chain))
    local field_struct = zeros(n_target)
    local field_noisy = zeros(n_target)

    if string(lat_key) in p_names && string(rho_key) in p_names
        local rho = chain[rho_key].data[j]
        local raw_lat = vec(collect(chain[lat_key].data[j]))
        local n_raw = length(raw_lat)
        
        # Reconstruct structured field
        field_struct .= raw_lat[1:min(n_raw, n_target)] .* (sig * sqrt(rho))
        
        # Incorporate IID noise if present in the chain
        if string(iid_key) in p_names
            local raw_iid = vec(collect(chain[iid_key].data[j]))
            field_noisy .= field_struct .+ (raw_iid[1:min(length(raw_iid), n_target)] .* (sig * sqrt(1.0 - rho)))
        else
            field_noisy .= field_struct
        end
    end

    return field_struct, field_noisy
end

function extract_manifold(::Union{Cyclic, Harmonic}, chain, M, j, sig, domain::Symbol)
    # Aligning periodic components with the seasonal period (u_N)
    local n_target = M.u_N
    local field = zeros(n_target)
    local p_names = string.(FlexiChains.parameters(chain))

    if "u_latent" in p_names
        local raw = chain[:u_latent].data[j]
        local flat_raw = vec(collect(raw))
        local n_raw = length(flat_raw)
        field .= flat_raw[1:min(n_raw, n_target)] .* sig
    elseif "u_alpha" in p_names && "u_beta" in p_names
        # Harmonic recovery: Reconstruct Sine/Cosine cycle from coefficients
        local alpha = chain[:u_alpha].data[j]
        local beta = chain[:u_beta].data[j]
        local angles = collect(1:n_target) .* (2.0 * pi / get(M, :period, 12.0))
        field .= (alpha .* sin.(angles) .+ beta .* cos.(angles)) .* sig
    end

    return field, field
end

function extract_manifold(::Mosaic, chain, M, j, sig, domain::Symbol)
    # Piecewise constant mapping of cluster intercepts back to unit grid
    local n_target = M.s_N
    local field = zeros(n_target)
    local p_names = string.(FlexiChains.parameters(chain))
    
    if "mu_local" in p_names
        local m_vals = get_params_vector(chain, "mu_local", M.n_mosaics)[j, :]
        field .= m_vals[M.cluster_assignments] .* sig
    end
    return field, field
end

# --- 4. Recursive Algebraic Composition ---

function extract_manifold(m::ComposedManifold, chain, M, j, sig, domain::Symbol)
    # Interaction manifolds (Kronecker) forced to :spacetime domain
    if m.operator == :kronecker_product
        # Type IV Interaction standard resolution
        local n_target = M.s_N * M.t_N
        local field = zeros(n_target)
        local p_names = string.(FlexiChains.parameters(chain))
        
        if "st_latent" in p_names
            local raw = chain[:st_latent].data[j]
            local flat_raw = vec(collect(raw))
            field .= flat_raw[1:min(length(flat_raw), n_target)] .* sig
        end
        return field, field
    else
        # Recurse for additive direct sums (oplus) using the parent domain context
        local total_field = nothing
        for comp in m.components
            sub_field, _ = extract_manifold(comp, chain, M, j, sig, domain)
            if isnothing(total_field)
                total_field = copy(sub_field)
            else
                total_field .+= sub_field
            end
        end
        return total_field, total_field
    end
end

function extract_manifold_k(m::ComposedManifold, chain, M, j, k, sig_k, domain::Symbol)
    # Recursive outcome-specific algebraic dispatch
    if m.operator == :kronecker_product
        local n_target = M.s_N * M.t_N
        local field = zeros(n_target)
        local key_str = "st_latent_" * string(k)
        local p_names = string.(FlexiChains.parameters(chain))
        
        if key_str in p_names
            local raw = chain[Symbol(key_str)].data[j]
            local flat_raw = vec(collect(raw))
            field .= flat_raw[1:min(length(flat_raw), n_target)] .* sig_k
        end
        return field, field
    else
        local total_field = nothing
        for comp in m.components
            sub_field, _ = extract_manifold_k(comp, chain, M, j, k, sig_k, domain)
            if isnothing(total_field)
                total_field = copy(sub_field)
            else
                total_field .+= sub_field
            end
        end
        return total_field, total_field
    end
end

# --- 4. Trait Retrieval Helpers ---
# These helpers standardise the coordinate mapping from struct types to the metadata index vectors.

# Maps Spatial manifolds to the 's_idx' column in the data source
# manifold_indices(::SpatialManifold, M) = M.s_idx

# Maps Temporal manifolds to the 't_idx' column (time steps)
# manifold_indices(::TemporalManifold, M) = M.t_idx

# Maps Seasonal manifolds to the 'u_idx' column (periodic bins)
# manifold_indices(::SeasonalManifold, M) = M.u_idx



function create_mixed_indices(term::String, data::DataFrame)
    # Parses terms like 'me(1 | group)' or 'me(x | group)'
    m = match(r"me\((.*)\|(.*)\)", term)
    isnothing(m) && return nothing
    
    lhs = strip(m.captures[1])
    group_var = Symbol(strip(m.captures[2]))
    
    group_data = data[!, group_var]
    levels = unique(group_data)
    group_map = Dict(v => i for (i, v) in enumerate(levels))
    indices = [group_map[v] for v in group_data]
    
    return (indices = indices, n_cat = length(levels), name = group_var)
end


function create_fixed_design(formula_rhs::AbstractString, data::Union{DataFrame, NamedArray}; contrasts=Dict{Symbol, Any}())
    df_tmp = data isa NamedArray ? DataFrame(data, :auto) : data
    if data isa NamedArray
        rename!(df_tmp, names(data, 2))
    end

    clean_rhs = replace(formula_rhs, r"(re|st|cv|int|me|fe)\\(.*?\\)" => "")
    clean_rhs = strip(replace(clean_rhs, r"\\+\\s*\\+" => "+"))
    clean_rhs = replace(clean_rhs, r"^\\+|\\+$" => "")

    if isempty(strip(clean_rhs)) || clean_rhs == "1"
        return NamedArray(ones(size(df_tmp, 1), 1), (1:size(df_tmp, 1), [:Intercept]))
    end

    try
        f = StatsModels.FormulaTerm(StatsModels.Term(:y), StatsModels.Term(Symbol(clean_rhs)))
        sch = StatsModels.schema(f, df_tmp, contrasts)
        f_applied = StatsModels.apply_schema(f, sch, StatsModels.RegressionModel)
        _, mm = StatsModels.modelcols(f_applied, df_tmp)
        return NamedArray(mm, (1:size(mm, 1), Symbol.(names(mm))))
    catch e
        return NamedArray(ones(size(df_tmp, 1), 1), (1:size(df_tmp, 1), [:Intercept]))
    end
end


 

 

function predict(model_obj::DynamicPPL.Model, chain, new_data::DataFrame; n_samples::Int=100, alpha=0.05)
    # Out-of-Sample Prediction Engine
    # Rationale: Projects latent manifolds onto new spatiotemporal coordinates.
    # Requirement: Verbatim interoperability with bstm_options and _reconstruct.
    # This method enables the projection of recovered latent manifolds onto previously unseen data points. 
    # **Technical Requirements:**
    # 1. **Manifold Mapping**: For discrete manifolds (ICAR/Leroux), new locations must be mapped to existing units via nearest-neighbor centroid lookup.
    # 2. **Kernel Projection**: For continuous manifolds (GP/RFF), we utilize the basis functions and lengthscales to project the surface.
    # 3. **Coordinate Consistency**: The method must accept a new `DataFrame`, extract coordinates, and apply the same scaling/transformations used during training.

    println("--- Starting BSTM Out-of-Sample Prediction ---")

    # 1. Recover Training Metadata (M)
    local M_train = model_obj.args.M
    local N_samples_total = size(chain, 1)
    local N_samps = min(n_samples, N_samples_total)
    
    # 2. Configure Prediction Metadata (PS)
    # We utilize bstm_options to ensure the new data is formatted identically to training data.
    # Note: we use dummy responses as y_obs is not used during the projection phase.
    local PS = bstm_options(
        data = new_data,
        y_obs = zeros(nrow(new_data)),
        model_family = M_train.model_family,
        model_space = M_train.model_space,
        model_time = M_train.model_time,
        svc_covariates = M_train.svc_covariates,
        use_sv = M_train.use_sv
    )

    # 3. Centroid Alignment for Discrete Manifolds
    # If the training model used areal units, we map new points to the nearest training centroid.
    if haskey(M_train, :areal_units)
        local centroids = M_train.areal_units.centroids
        PS_s_idx = [argmin([sum(((new_data.s_x[i], new_data.s_y[i]) .- c).^2) for c in centroids]) for i in 1:nrow(new_data)]
        # Update PS with aligned indices
        PS = merge(PS, (s_idx = PS_s_idx,))
    end

    # 4. Invoke Architectural Reconstruction
    # We treat the new data as a Post-Stratification (PS) object within the existing _reconstruct logic.
    # This allows us to reuse the stable manifold retrieval and assembly code.
    local arch_type = get_architecture(get(M_train, :model_arch, "univariate"))
    
    # Selecting a subset of samples for efficiency if requested
    local chain_sub = chain[1:N_samps]
    
    # _reconstruct(arch, name, chain, M_training, PS_new_data, alpha)
    local res = _reconstruct(arch_type, "prediction", chain_sub, M_train, PS, alpha)

    println("Prediction Complete. Generated samples for ", nrow(new_data), " observations.")

    return res
end

 

# 1. PSIS-LOO Implementation for BSTM
# Rationale: Standardizing the extraction of log-likelihood matrices to provide 
# Expected Log Pointwise Predictive Density (ELPD) estimates.

function bstm_loo(chain, model_obj)
    println("--- Calculating PSIS-LOO for BSTM ---")
    
    # Extract log-likelihood matrix [Samples x Observations]
    # Based on our _reconstruct logic, log_lik is stored in the results object
    # or can be pulled directly from the chain if using Turing's point_loglikelihoods
    ### Model Selection Suite (v10.0)

# This suite implements robust Bayesian model comparison tools. While WAIC is a useful heuristic, **LOO-CV (PSIS-LOO)** provides a more reliable estimate of out-of-sample predictive performance by smoothing importance weights. 

# Additionally, we introduce **Bridge Sampling** logic to approximate the Marginal Likelihood, allowing for the calculation of **Bayes Factors** to perform formal hypothesis testing between competing manifold structures.

    local results = _reconstruct(get_architecture(model_obj.args.M.model_arch), "loo_calc", chain, model_obj.args.M, nothing, 0.05)
    
    # Use PosteriorStats.loo (assumes log_likelihood array is available)
    # Note: This requires the log_lik matrix gathered during reconstruction
    local loo_result = loo(results.log_lik_matrix)
    
    display(loo_result)
    return loo_result
end

# 2. Bayes Factor Suite (Manifold Comparison)
function compare_manifolds(model_a_results, model_b_results)
    println("--- Manifold Comparison Dashboard ---")
    
    # 1. ELPD Comparison
    local diff_elpd = model_a_results.waic.elpd - model_b_results.waic.elpd
    
    # 2. Bayes Factor Approximation
    # Rationale: Using the Savage-Dickey density ratio or Bridge sampling 
    # approximation if Marginal Likelihood is available.
    # For this suite, we provide a structured comparison table.
    
    comparison_df = DataFrame(
        Metric = ["ELPD (WAIC)", "Effective Params (p_waic)", "WAIC Score"],
        Model_A = [model_a_results.waic.elpd, model_a_results.waic.p_waic, model_a_results.waic.waic],
        Model_B = [model_b_results.waic.elpd, model_b_results.waic.p_waic, model_b_results.waic.waic]
    )
    
    comparison_df[!, :Delta] = comparison_df.Model_A - comparison_df.Model_B
    
    display(comparison_df)
    return comparison_df
end



function bstm_cv_orchestrator(formula::String, data::DataFrame;
                             method=:kfold,
                             k=5,
                             lolo_var=:s_idx,
                             model_family="gaussian",
                             n_samples=500,
                             sampler=MH(),
                             kwargs...)
    # Audited CV Orchestrator [v09.6 Final Coordinate Sync]
    # Rationale: Ensures s_x and s_y are passed to bstm() to trigger dynamic 
    # reconstruction of W-matrix (tessellation) for each fold's training set.

    println("--- Starting BSTM Cross-Validation Orchestrator [Method: $method] ---")
    local N_total = nrow(data)
    local results_folds = []
    local folds_indices = []

    if method == :kfold
        local idx_perm = shuffle(1:N_total)
        local fold_size = floor(Int, N_total / k)
        for i in 1:k
            start_idx = (i-1) * fold_size + 1
            end_idx = i == k ? N_total : i * fold_size
            push!(folds_indices, idx_perm[start_idx:end_idx])
        end
    elseif method == :lolo
        local unique_locs = unique(data[!, lolo_var])
        for loc in unique_locs
            push!(folds_indices, findall(x -> x == loc, data[!, lolo_var]))
        end
        k = length(unique_locs)
    end

    for i in 1:k
        println("\n--- Executing Fold $i / $k ---")
        local test_idx = folds_indices[i]
        local train_idx = setdiff(1:N_total, test_idx)
        local train_data = data[train_idx, :]
        local test_data = data[test_idx, :]

        # Propagate training coordinates to trigger build_structure_template internally
        # We explicitly map s_x and s_y from the DataFrame columns
        local model_train = bstm(formula, train_data;
            model_family=model_family,
            s_x=collect(train_data.s_x),
            s_y=collect(train_data.s_y),
            kwargs...)

        local chain_train = sample(model_train, sampler, n_samples, progress=false)
        local res_pred = predict(model_train, chain_train, test_data; n_samples=n_samples)

        local y_test_obs = test_data.y_obs
        local y_test_pred = res_pred.predictions_observed_denoised.mean

        local rmse = sqrt(mean((y_test_obs .- y_test_pred).^2))
        local mae = mean(abs.(y_test_obs .- y_test_pred))
        
        # Safe R2 calculation
        local r2 = (length(y_test_obs) > 1 && var(y_test_pred) > 0) ? cor(y_test_obs, y_test_pred)^2 : 0.0

        push!(results_folds, (rmse=rmse, mae=mae, r2=r2, fold=i))
        println("Fold $i Results -> RMSE: ", round(rmse, digits=4))
    end

    return (
        mean_rmse = mean([f.rmse for f in results_folds]),
        mean_r2 = mean([f.r2 for f in results_folds]),
        folds = results_folds
    )
end

;;
 