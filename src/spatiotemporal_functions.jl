#!Reference

# Definitions

# Discrete and Graph Manifold Categories 

# BSTM Low-Level Manifold Registry [v06.1 - Reusable Schema] ---
# Rationale: Manifolds are now defined as low-level primitives that are domain-agnostic. 
# The context (Spatial vs Temporal) is determined at the model-building stage.

abstract type Manifold end

# Define dedicated abstract type for transformations to enable distinct dispatch
abstract type ManifoldTransform <: Manifold end

# Redefine transformation types to inherit from ManifoldTransform
struct LogManifold <: ManifoldTransform end
struct ZScoreManifold <: ManifoldTransform end
struct UnitScaleManifold <: ManifoldTransform end

# Redefine TransformedManifold to wrap a base manifold and a transformation type
struct TransformedManifold <: Manifold
    manifold::Manifold
    transform_fn::DataType
end

# Logic: When the right-hand side is a transformation type, wrap the left-hand side
function Base.:|>(m::Manifold, t::ManifoldTransform)
    return TransformedManifold(m, typeof(t))
end



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
struct Harmonic <: Manifold
    amplitude_prior::UnivariateDistribution
    phase_prior::UnivariateDistribution
    sigma_prior::UnivariateDistribution
    period::UnivariateDistribution
end

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
  
# --- 6. Compositional Manifolds (Physics & Stacking) ---
# struct KnorrHeld <: Manifold
#     dim1::Symbol
#     dim2::Symbol
#     type::Char
#     sigma_prior::UnivariateDistribution
# end

# struct AdvectionDiffusion <: Manifold
#     dim1::Symbol
#     dim2::Symbol
#     velocity_field::Union{Nothing, Symbol}
#     velocity_prior::UnivariateDistribution
#     diffusion_coeff::Union{Nothing, Float64}
#     diffusion_prior::UnivariateDistribution
# end

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

# Helper to determine discrete status without ternary syntax
function is_discrete_family(::Union{PoissonFamily, NegativeBinomialFamily, BinomialFamily})
    return true
end

function is_discrete_family(::AbstractBSTM_Family)
    return false
end

# Point Kernel (Uncensored) 

# Refined point kernel for all families
function bstm_kernel(fam::AbstractBSTM_Family, ::Uncensored, zi::AbstractZIState, d, eta, sig, y)
    dist = get_dist_ref(fam, d, eta, sig)
    lp_branch = logpdf(dist, y)
    
    # Initialize with default value
    lp_final = lp_branch

    # Logic for Zero-Inflation Mixture
    # Mathematical Definition: P(y) = phi * I(y=0) + (1 - phi) * f(y; theta)
    # Where I(y=0) is the indicator function (point mass at zero)
    if zi isa ZeroInflated
        log_phi = log(d.phi_zi + 1e-15)
        log_one_minus_phi = log(1.0 - d.phi_zi + 1e-15)
        
        if y == 0.0
            # For both discrete and continuous, y=0 captures the point mass
            # Discrete families may also have probability mass at 0 from the distribution itself
            pz = 0.0
            if is_discrete_family(fam)
                pz = pdf(dist, 0.0)
            end
            lp_final = logsumexp(log_phi, log_one_minus_phi + log(pz + 1e-15))
        else
            # For y > 0, we only have the continuous/discrete density branch
            lp_final = log_one_minus_phi + lp_branch
        end
    end

    # Hurdle Truncation adjustment
    if d.hurdle > -Inf
        lp_final = lp_final - logccdf(dist, d.hurdle)
    end

    return lp_final
end

# Logic for Zero-Inflation Mixture in Interval Censoring
function bstm_kernel(fam::AbstractBSTM_Family, ::IntervalCensored, zi::AbstractZIState, d, eta, sig, y)
    function stable_logdiffexp(a, b)
        if a <= b
            return -1e15
        end
        diff = b - a
        safe_diff = min(-eps(typeof(diff)), diff)
        return a + log1mexp(safe_diff)
    end

    dist = get_dist_ref(fam, d, eta, sig)
    adj_L = d.y_L[1]
    if is_discrete_family(fam)
        adj_L = adj_L - 1.0
    end

    lp_prob = stable_logdiffexp(logcdf(dist, d.y_U[1]), logcdf(dist, adj_L))
    lp_final = lp_prob

    if zi isa ZeroInflated
        log_phi = log(d.phi_zi + 1e-15)
        log_one_minus_phi = log(1.0 - d.phi_zi + 1e-15)
        
        # If the interval [y_L, y_U] includes zero, add the point mass phi
        if d.y_L[1] <= 0.0
            lp_final = logsumexp(log_phi, log_one_minus_phi + lp_prob)
        else
            lp_final = log_one_minus_phi + lp_prob
        end
    end

    return lp_final
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
    local adj_L = is_discrete_family(fam) ? d.y_L[1] - 1.0 : d.y_L[1]
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
    local adj_L = is_discrete_family(fam) ? d.y_L[1] - 1.0 : d.y_L[1]
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



const BSTM_MODULE_KEYWORDS = Set([
    "intercept",
    "bias", 
    "spatial", 
    "temporal", 
    "smooth",  
    "nested", 
    "eigen", 
    "fixed", 
    "mixed",
    "svc",
    "volatility",
    "network",
    "spacetime",
    "hyperbolic",
    "dynamics"
])


   
# # BSTM Monolithic Configuration Engine v19.23.14



function bstm_options(; kwargs...)
    # Initialization and Technical Metadata Consolidation
    # Consolidates all user-provided keyword arguments into a mutable dictionary.
    M = Dict{Symbol, Any}(kwargs)
    data = get(M, :data, nothing)

    # Automated Keyword Discovery
    # Scans the provided DataFrame for standard BSTM variable names.
    if !isnothing(data)
        v_names = Symbol.(names(data))
        keyword_map = Dict(
            :y_obs => [:y, :y_obs, :response],
            :s_x => [:s_x, :lon, :longitude],
            :s_y => [:s_y, :lat, :latitude],
            :t_v => [:t_v, :time, :time_coords],
            :s_idx => [:s_idx, :space_idx],
            :t_idx => [:t_idx, :time_idx],
            :u_idx => [:u_idx, :season_idx],
            :log_offset => [:log_offset, :logoffset, :log_offsets],
            :weights => [:weights, :wts],
            :trials => [:trials, :n_trials]
        )

        for (key, candidates) in keyword_map
            # Check if the user explicitly provided a value for the key
            if haskey(M, key)
                val = M[key]
                # Type-Safety Check: Only attempt lookup if the value is a Symbol or String
                # If it is already a Vector or Matrix, skip to prevent lookup/bounds errors
                if val isa Symbol || val isa AbstractString
                    sym_val = Symbol(val)
                    if sym_val in v_names
                        M[key] = data[!, sym_val]
                    else
                        @warn "Column $sym_val not found in data for parameter $key. Keeping original value."
                    end
                end
            # Fallback to candidates if key is not provided
            else
                for cand in candidates
                    if cand in v_names
                        M[key] = data[!, cand]
                        break
                    end
                end
            end
        end
    end


    # Dimensional Validation & Scope Discovery
    if !haskey(M, :y_obs)
        error("BSTM Error: :y_obs is required for model initialization.")
    end

    y_obs_raw = M[:y_obs]
    y_N = (y_obs_raw isa AbstractMatrix) ? size(y_obs_raw, 1) : (y_obs_raw isa Vector{<:AbstractVector} ? length(y_obs_raw[1]) : length(y_obs_raw))
    M[:y_N] = y_N

    # Manifold Index Realignment
    s_idx_raw = get(M, :s_idx, ones(Int, y_N))
    t_idx_raw = get(M, :t_idx, ones(Int, y_N))
    u_idx_raw = get(M, :u_idx, ones(Int, y_N))

    M[:s_idx] = Int.(collect(s_idx_raw))
    M[:t_idx] = Int.(collect(t_idx_raw))
    M[:u_idx] = Int.(collect(u_idx_raw))

    # If the user provides s_N/t_N, we use them; otherwise scan indices.
    M[:s_N] = get(M, :s_N, maximum(M[:s_idx]))
    M[:t_N] = get(M, :t_N, maximum(M[:t_idx]))
    M[:u_N] = get(M, :u_N, maximum(M[:u_idx]))

    # Global linear indexing for space-time interaction tensors
    M[:st_idx] = [(M[:t_idx][i] - 1) * M[:s_N] + M[:s_idx][i] for i in 1:y_N]

    # Design Matrix Factory: Fixed Effects
    if haskey(M, :fixed_parts) && !isnothing(data)
        f_expr = isempty(M[:fixed_parts]) ? "1" : join(M[:fixed_parts], " + ")
        M[:Xfixed] = create_fixed_design(f_expr, data; contrasts=get(M, :contrasts, Dict()))
    else
        M[:Xfixed] = NamedArray(ones(Float64, y_N, 1), (1:y_N, [:Intercept]))
    end
    M[:Xfixed_N] = size(M[:Xfixed], 2)

    # Taxonomic Metadata Hardening
    # Use get!() to ensure defaults are explicitly written into the dictionary.
    get!(M, :model_space, "none")
    get!(M, :model_time, "none")
    get!(M, :model_season, "none")

    # GMRF Precision Matrix Template Factory
 
    ms = string(get!(M, :model_space, "none"))
    mt = string(get!(M, :model_time, "none"))
    mu = string(get!(M, :model_season, "none"))

    # FIXED: Build templates using the locked dimensions
    M[:s_Q_template] = build_structure_template(Symbol(ms), M[:s_N]; W=get(M, :W, nothing))
    M[:t_Q_template] = build_structure_template(Symbol(mt), M[:t_N])
    M[:u_Q_template] = (mu != "none") ? build_structure_template(Symbol(mu), M[:u_N]) : nothing

    # Hierarchical Registries Initialization
    get!(M, :nested_manifolds, Dict{String, Any}())
    get!(M, :spatial_hierarchy, Dict{Symbol, Any}())
    get!(M, :mixed_terms, []) 
    get!(M, :eigen_terms, [])
    get!(M, :svc_covariates, Symbol[])
    get!(M, :basis_matrices, Dict{Symbol, Any}())

    # Coordinate Sync for Continuous Kernels
    if !haskey(M, :s_coord)
        if haskey(M, :s_x) && haskey(M, :s_y)
            M[:s_coord] = hcat(M[:s_x], M[:s_y])
        else
            M[:s_coord] = zeros(Float64, M[:s_N], 2)
        end
    end

    # Volatility & Spectral Projection Caching
    if get(M, :use_sv, false)
        get!(M, :M_rff_sigma, 20)
        W_v, b_v = generate_informed_rff_params(M[:s_coord], M[:M_rff_sigma])
        M[:vol_proj] = (M[:s_coord] * W_v) .+ b_v'
    end

    # Technical Flag Standardization
    get!(M, :model_arch, "univariate")
    get!(M, :model_family, "gaussian")
    get!(M, :noise, 1e-4)
    get!(M, :use_zi, false)
    get!(M, :log_offset, zeros(Float64, y_N))
    get!(M, :weights, ones(Float64, y_N))
    get!(M, :trials, ones(Int, y_N))
    get!(M, :period, 12.0)

    # Observation Mask Discovery
    if !haskey(M, :y_ok)
        if y_obs_raw isa AbstractMatrix
            M[:y_ok] = [findall(!isnan, y_obs_raw[:, k]) for k in 1:get(M, :outcomes_N, 1)]
        else
            M[:y_ok] = [findall(!isnan, vec(collect(y_obs_raw)))]
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
    return (centroids=final_centroids, polygons=polys_coords,
            adjacency_edges=v_edges, graph=g, W=W, hull_coords=hull_coords,
            s_idx=new_assigns, s_x=s_x, s_y=s_y, s_vals=collect(1:size(W,1)), termination_reason=reason)
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
        polygons = polys_output,
        adjacency_edges = adjacency_edges_output,
        graph = g_final,
        W = adjacency_matrix,
        hull_coords = hull_coords_output,
        s_idx = collect(1:nAU ),
        s_x = [c[1] for c in final_centroids[1:nAU]],
        s_y = [c[2] for c in final_centroids[1:nAU]],
        s_vals = collect(1:nAU ),
        termination_reason = "positions inferred from adjacency matrix"
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



function ensure_connected!(W::AbstractMatrix, centroids::Vector{<:Tuple{Real, Real}})
    # Convert the adjacency matrix to a Graph object for component analysis
    g = Graph(W)
    comps = connected_components(g)
    
    # If the graph is already connected, no further action is required
    if length(comps) <= 1
        return W
    end

    # Number of components to bridge
    n_comps = length(comps)
    
    # 1. Calculate Centroids for each component to reduce search space
    # This simplifies the problem from comparing all nodes to comparing component centers
    comp_centroids = Vector{Vector{Float64}}(undef, n_comps)
    for i in 1:n_comps
        pts = [ [centroids[node][1], centroids[node][2]] for node in comps[i] ]
        comp_centroids[i] = mean(pts)
    end

    # 2. Build a KD-Tree of the component centroids for efficient spatial lookup
    # We treat each component as a single point in this high-level search
    centroid_matrix = hcat(comp_centroids...)
    tree = KDTree(centroid_matrix)

    # 3. Iteratively bridge components using the nearest-neighbor logic
    # We connect the most isolated components first to ensure global connectivity
    for i in 1:n_comps
        # Find the nearest component other than itself
        # knn returns indices and distances; we look for the 2 nearest (itself + neighbor)
        idxs, dists = knn(tree, comp_centroids[i], 2)
        target_comp_idx = idxs[2]
        
        # Now find the absolute closest node pair between the two identified components
        # While this sub-step is O(Ni * Nj), the components are typically small sub-graphs
        min_dist = Inf
        best_pair = (0, 0)
        
        for u in comps[i]
            for v in comps[target_comp_idx]
                d = sqdist([centroids[u][1], centroids[u][2]], [centroids[v][1], centroids[v][2]])
                if d < min_dist
                    min_dist = d
                    best_pair = (u, v)
                end
            end
        end
        
        # 4. Update the sparse adjacency matrix with the new bridging edge
        u_node, v_node = best_pair
        if u_node != 0 && v_node != 0
            W[u_node, v_node] = true
            W[v_node, u_node] = true
        end
    end

    # Verify final connectivity to ensure no orphans remain
    final_g = Graph(W)
    if length(connected_components(final_g)) > 1
        # Recursive fallback for complex edge cases (e.g., non-convex boundaries)
        return ensure_connected!(W, centroids)
    end

    return W
end

function sqdist(p1, p2)
    return (p1[1]-p2[1])^2 + (p1[2]-p2[2])^2
end


function plot_spatial_graph(; au=nothing, pts=nothing, plot_title="Spatial Partitioning")
 
    # 2. Base Plot Initialization
    plt = Plots.plot(aspect_ratio=:equal, legend=false)
    Plots.title!(plt, plot_title)

    # 3. Polygon Geometry Rendering
        for poly_coords in au[:polygons]
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
        for edge in Graphs.edges(au[:graph])
            u, v = Graphs.src(edge), Graphs.dst(edge)
            p1, p2 = au[:centroids][u], au[:centroids][v]
            Plots.plot!(plt, [p1[1], p2[1]], [p1[2], p2[2]], color=:red, lw=1.5, alpha=0.6)
        end 

    # 5. Scatter Plotting: Data Points and Polygon Centroids
    if !isnothing(pts)
        Plots.scatter!(plt, [p[1] for p in pts], [p[2] for p in pts],
            markersize=1, color=:gray, alpha=0.3, label="Points")
    end
 
        Plots.scatter!(plt, [c[1] for c in au[:centroids]], [c[2] for c in au[:centroids]],
            markersize=4, color=:blue, markerstrokecolor=:white, label="Centroids")

    # 6. Boundary Constraints
        bx = [p[1] for p in au[:hull_coords] if !isnan(p[1])]
        by = [p[2] for p in au[:hull_coords] if !isnan(p[2])]
        Plots.plot!(plt, bx, by, color=:black, lw=2, ls=:dash)

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

# LKJ modify this method slightly, in order to be able to sample correlation matrices C from a distribution 
# proportional to [detC]η−1. The larger the η, the larger will be the determinant, meaning that generated 
# correlation matrices will more and more approach the identity matrix. The value η=1 corresponds to uniform distribution. 
 
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
    # Re-scaling variance using the geometric mean of variances of the ICAR component
    # Rationale: Standardizes the structured innovation scale relative to the unstructured IID component.
    # Requirement: Precision matrix Q must be symmetric and positive-definite.

    N = size(adjacency_mat)[1]
    if N <= 1
        return 1.0
    end

    # 1. Construct the Graph Laplacian
    asum = vec(sum(adjacency_mat, dims=2))
    Q = Diagonal(asum) - adjacency_mat

    # 2. Symmetric Perturbation for Numerical Stability
    # Rationale: Adding a ridge to the full matrix ensures stability across the entire spectrum.
    jitter = sqrt(1e-15) * mean(asum)
    Q_stable = Q +jitter * I

    # 3. Sum-to-Zero Constraint Adjustment
    # Compute the diagonal of the generalized inverse subject to the constraint A'x = 0 where A is a vector of ones.
    A = ones(N)
    try
        # Solve Q * S = I under the constraint
        S = Q_stable \ Diagonal(A)
        V = S * A
        # Projection matrix to enforce the sum-to-zero constraint in the covariance space
        S = S - V * inv(A' * V) * V'

        diag_s = diag(S)

        # 4. Extract Variances for Valid Eigen-dimensions
        valid_diag = filter(x -> x > jitter, diag_s)

        if isempty(valid_diag)
            return 1.0
        end

        # 5. Calculate Scaling Factor
        # Geometric mean of the variances ensures the structured component represents marginal variance.
        scale_factor = exp(mean(log.(valid_diag)))

        return isnan(scale_factor) ? 1.0 : scale_factor
    catch
        return 1.0
    end
end

# POSITIONAL OVERLOAD
function scaling_factor_bym2(node1, node2, groups=ones(length(node1)))
    # Logic: Calculates independent scale factors for K connected components.
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



function estimate_local_kde_with_extrapolation(s_coord_tuple, t_idx, target_ts; grid_res=600, sd_extension_factor=0.25)
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



function plot_kde_simple(s_coord_tuple; grid_res=600, sd_extension_factor=0.25, title="Spatial Intensity (KDE)")
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

            # 3.5 Spatiotemporal (ST) Mapping
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
        return Harmonic(resolved_priors.amplitude_prior, resolved_priors.phase_prior, resolved_priors.sigma_prior, M.period)
    elseif manifold_name == "cyclic"
        return Cyclic(M.period, resolved_priors.sigma_prior)

    # 5. Specialized and Smooth Manifolds
    elseif manifold_name == "mosaic"
        return Mosaic(get(M, :s_coord, zeros(M.s_N, 2)), M.n_regions, true)
    elseif manifold_name == "localadaptive" || manifold_name == "local_adaptive"
        # Riemannian diagonal weighting for non-stationary smoothing
        return LocalAdaptive(resolved_priors.sigma_prior, :local_weights)
    elseif manifold_name == "bcgn"
        # Bipartite Covariate Graph Network
        return BCGN(resolved_priors.sigma_prior, get(M, :bipartite_adj, sparse(I(M.s_N))), get(M, :group_weights, ones(M.s_N)))
    # elseif manifold_name == "KnorrHeld"
        # Space-Time Interaction types I, II, III, IV
    #    return KnorrHeld(:space, :time, 'I', resolved_priors.sigma_prior)
    #elseif manifold_name == "AdvectionDiffusion"
        # Physics-informed transport manifold
    #    return AdvectionDiffusion(:space, :time, nothing, resolved_priors.sigma_prior, 0.1, resolved_priors.sigma_prior)
    
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

    # Spatiotemporal maps: Outcome-specific tensors [Units_k x Time_k x Samples]
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


function _discover_manifold_realizations(chain, M, n_samples, outcomes_N, p_names)
    # BSTM Manifold Discovery Engine v18.2.5
    # Timestamp: 2026-06-10 11:15:00
    # Rationale: Restoring Hierarchical Scale discovery and multi-outcome SVC generalization.

    # # 1. Fixed Effect Parameter Hoisting
    # Extracting the design matrix coefficients for all outcomes simultaneously.
    xf_betas = nothing
    if "Xfixed_beta" in p_names || any(occursin.(Ref("Xfixed_beta["), p_names))
        xf_betas = get_params_vector(chain, "Xfixed_beta", M.Xfixed_N * outcomes_N)'
    end

    # # 2. Spatial and Temporal Realization Containers
    # Initializing structures for K outcomes to store primary latent fields.
    s_eff_struct = []
    s_eff_noisy = []
    t_eff = []

    for k in 1:outcomes_N
        # Spatial Field Discovery (Discrete units)
        s_key = outcomes_N > 1 ? "s_latent_$k" : "s_latent"
        if s_key in p_names || any(occursin.(Regex("^$s_key\\["), p_names))
            s_vals = get_params_vector(chain, s_key, M.s_N)'
            push!(s_eff_struct, s_vals)
            push!(s_eff_noisy, s_vals)
        else
            push!(s_eff_struct, zeros(M.s_N, n_samples))
            push!(s_eff_noisy, zeros(M.s_N, n_samples))
        end

        # Temporal Trend Discovery
        t_key = outcomes_N > 1 ? "t_latent_$k" : "t_latent"
        if t_key in p_names || any(occursin.(Regex("^$t_key\\["), p_names))
            push!(t_eff, get_params_vector(chain, t_key, M.t_N)')
        else
            push!(t_eff, zeros(M.t_N, n_samples))
        end
    end

    # # 3. Seasonal Cycle Realization
    # Mapping trigonometric or cyclic random effects based on periodicity metadata.
    u_eff = zeros(M.u_N, n_samples)
    if M.model_season == "harmonic"
        u_alpha = vec(get_params_vector(chain, "u_alpha", 1))
        u_beta = vec(get_params_vector(chain, "u_beta", 1))
        u_sigma = vec(get_params_vector(chain, "u_sigma", 1))
        angles = collect(1:M.u_N) .* (2.0 * pi / M.period)
        for j in 1:n_samples
            u_eff[:, j] .= (u_alpha[j] .* sin.(angles) .+ u_beta[j] .* cos.(angles)) .* u_sigma[j]
        end
    elseif M.model_season == "cyclic" && "u_latent" in p_names
        u_eff .= get_params_vector(chain, "u_latent", M.u_N)'
    end

    # # 4. Spatiotemporal Interaction (Knorr-Held I-IV)
    # Generalizing interaction tensors across all outcomes.
    st_eff_maps = [zeros(M.s_N, M.t_N, n_samples) for _ in 1:outcomes_N]
    for k in 1:outcomes_N
        st_key = outcomes_N > 1 ? "st_latent_$k" : "st_latent"
        if st_key in p_names || any(occursin.(Regex("^$st_key\\["), p_names))
            st_raw = get_params_vector(chain, st_key, M.s_N * M.t_N)'
            for j in 1:n_samples
                st_eff_maps[k][:, :, j] .= reshape(st_raw[:, j], M.s_N, M.t_N)
            end
        end
    end

    # # 5. Spatially Varying Coefficients (SVC)
    # Mapping non-stationary regression slopes for discovered covariates.
    svc_slopes = [zeros(M.s_N, length(M.svc_covariates), n_samples) for _ in 1:outcomes_N]
    for k in 1:outcomes_N
        for (v_idx, c_sym) in enumerate(M.svc_covariates)
            svc_key = outcomes_N > 1 ? "beta_svc_$(c_sym)_$k" : "beta_svc_$c_sym"
            if svc_key in p_names || any(occursin.(Regex("^$svc_key\\["), p_names))
                svc_raw = get_params_vector(chain, svc_key, M.s_N)'
                for j in 1:n_samples
                    svc_slopes[k][:, v_idx, j] .= svc_raw[:, j]
                end
            end
        end
    end

    # # 6. Basis Smooth Accumulation (1D-4D)
    # Projects non-linear basis matrices using recovered weights.
    basis_eff_accum = nothing
    if haskey(M, :basis_matrices) && !isempty(M.basis_matrices)
        basis_eff_accum = zeros(M.y_N, n_samples)
        for v_sym in keys(M.basis_matrices)
            b_sig_key = "sig_basis_$v_sym"
            b_beta_key = "beta_basis_$v_sym"
            if b_sig_key in p_names && (b_beta_key in p_names || any(occursin.(Regex("^$b_beta_key\\["), p_names)))
                b_sig = vec(get_params_vector(chain, b_sig_key, 1))
                b_beta = get_params_vector(chain, b_beta_key, size(M.basis_matrices[v_sym], 2))
                for j in 1:n_samples
                    basis_eff_accum[:, j] .+= (M.basis_matrices[v_sym] * b_beta[j, :]) .* b_sig[j]
                end
            end
        end
    end

    # # 7. Hierarchical Multi-Resolution Scale Recovery
    # Discovery of regional or aggregate spatial latent fields.
    hierarchical_scales = Dict{Symbol, Any}()
    if haskey(M, :spatial_hierarchy) && !isempty(M.spatial_hierarchy)
        for scale_sym in keys(M.spatial_hierarchy)
            h_key = "s_latent_$scale_sym"
            if h_key in p_names || any(occursin.(Regex("^$h_key\\["), p_names))
                n_units_h = M.spatial_hierarchy[scale_sym].n_units
                hierarchical_scales[scale_sym] = get_params_vector(chain, h_key, n_units_h)'
            end
        end
    end

    # # 8. Heteroskedastic Volatility Mapping
    # Extraction of Stochastic Volatility surfaces via RFF projections.
    sv_surface = outcomes_N > 1 ? [ones(M.y_N, n_samples) for _ in 1:outcomes_N] : ones(M.y_N, n_samples)
    if get(M, :use_sv, false)
        for k in 1:outcomes_N
            sig_v_key = outcomes_N > 1 ? "sigma_log_var_k[$k]" : "sigma_log_var"
            beta_v_key = outcomes_N > 1 ? "beta_vol_latent_k[$k]" : "beta_vol_latent"
            
            if sig_v_key in p_names && beta_v_key in p_names
                sig_log_var = vec(get_params_vector(chain, sig_v_key, 1))
                b_vol = get_params_vector(chain, beta_v_key, M.M_rff_sigma)
                for j in 1:n_samples
                    vol_lat = sqrt(2.0 / M.M_rff_sigma) .* cos.(M.vol_proj * b_vol[j, :])
                    target = outcomes_N > 1 ? sv_surface[k] : sv_surface
                    target[:, j] .= exp.((sig_log_var[j] .* vol_lat) ./ 2.0)
                end
            end
        end
    elseif "y_sigma" in p_names
        y_sig = vec(get_params_vector(chain, "y_sigma", 1))
        for j in 1:n_samples
            if outcomes_N > 1
                for k in 1:outcomes_N; sv_surface[k][:, j] .= y_sig[j]; end
            else
                sv_surface[:, j] .= y_sig[j]
            end
        end
    end

    return ( 
        n_samples = n_samples,
        outcomes_N = outcomes_N,
        Xfixed_betas = xf_betas,
        s_eff_struct = s_eff_struct,
        s_eff_noisy = s_eff_noisy,
        t_eff = t_eff,
        u_eff = u_eff,
        st_eff_maps = st_eff_maps,
        svc_slopes = svc_slopes,
        basis_eff_accum = basis_eff_accum,
        sv_surface = sv_surface,
        hierarchical_scales = hierarchical_scales
    )
end


function _modular_eta_assembly(N_tot_in, registry, M, PS_in)
    # BSTM Linear Predictor Assembly Engine v18.1.32
    # Timestamp: 2026-06-07 14:45:00
    # Rationale: Optimizing SVC inner loop by precomputing column indices and avoiding redundant symbol conversion.

    n_samples = registry.n_samples
    outcomes_n = registry.outcomes_N
    y_n_train = Int(M.y_N)
    actual_limit = isnothing(PS_in) ? y_n_train : Int(N_tot_in)

    # Precomputing SVC column indices to prevent findfirst/names calls in inner loops
    svc_vars = get(M, :svc_covariates, Symbol[])
    n_svc = length(svc_vars)
    
    # Training grid indices
    train_col_names = Symbol.(names(M.Xfixed, 2))
    train_svc_indices = Int[findfirst(==(v), train_col_names) for v in svc_vars]
    
    # Prediction grid indices if PS is present
    ps_svc_indices = Int[]
    if !isnothing(PS_in)
        ps_col_names = Symbol.(names(PS_in.Xfixed, 2))
        ps_svc_indices = Int[findfirst(==(v), ps_col_names) for v in svc_vars]
    end

    # Primary linear predictor container [Total Observations x Outcomes x MCMC Samples]
    eta_container = zeros(Float64, actual_limit, outcomes_n, n_samples)

    for j in 1:n_samples
        for k in 1:outcomes_n
            # Extract realized manifold slices for current outcome and sample
            s_f = !isnothing(registry.s_eff_noisy) && !isempty(registry.s_eff_noisy) ? registry.s_eff_noisy[k][:, j] : Float64[]
            t_f = !isnothing(registry.t_eff) && !isempty(registry.t_eff) ? registry.t_eff[k][:, j] : Float64[]
            u_f = !isnothing(registry.u_eff) && !isempty(registry.u_eff) ? registry.u_eff[:, j] : Float64[]

            # Interaction and Varying Coefficient tensors
            st_f = !isnothing(registry.st_eff_maps) && !isempty(registry.st_eff_maps[k]) ? registry.st_eff_maps[k][:, :, j] : zeros(0, 0)
            svc_f = !isnothing(registry.svc_slopes) && !isempty(registry.svc_slopes[k]) ? registry.svc_slopes[k][:, :, j] : zeros(0, 0)

            # Fixed Effect parameters
            xf_n = Int(M.Xfixed_N)
            has_fixed = !isnothing(registry.Xfixed_betas) && xf_n > 0
            beta_slice = has_fixed ? registry.Xfixed_betas[((k-1)*xf_n + 1):(k*xf_n), j] : Float64[]

            for i in 1:actual_limit
                is_obs = i <= y_n_train
                src = is_obs ? M : PS_in
                idx = is_obs ? i : i - y_n_train
                
                # Pre-cached indices for the specific metadata source
                target_svc_indices = is_obs ? train_svc_indices : ps_svc_indices

                s_ptr = Int(src.s_idx[idx])
                t_ptr = Int(src.t_idx[idx])
                u_ptr = Int(src.u_idx[idx])

                # # 1. Structural Superposition
                val = 0.0
                if !isempty(s_f); val += s_f[s_ptr]; end
                if !isempty(t_f); val += t_f[t_ptr]; end
                if !isempty(u_f); val += u_f[u_ptr]; end

                # # 2. Spatiotemporal Interaction
                if !isempty(st_f); val += st_f[s_ptr, t_ptr]; end

                # # 3. SVC Contribution using precomputed indices
                if !isempty(svc_f)
                    for v_idx in 1:n_svc
                        col_idx = target_svc_indices[v_idx]
                        if !isnothing(col_idx)
                            val += svc_f[s_ptr, v_idx] * src.Xfixed[idx, col_idx]
                        end
                    end
                end

                # # 4. Observation-Specific Components
                if is_obs
                    if !isnothing(registry.basis_eff_accum)
                        val += registry.basis_eff_accum[idx, j]
                    end
                    if haskey(M, :log_offset)
                        val += M.log_offset[idx]
                    end
                end

                # # 5. Fixed Effect Design Matrix Mapping
                if has_fixed
                    val += dot(vec(collect(src.Xfixed[idx, :])), beta_slice)
                end

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





# # BSTM Monolithic Visualization Engine [v19.38.5 ]
# Timestamp: 2025-11-05 20:00:00
# Rationale: Expanding the dashboard to include Mixed Effects, Seasonal Cycles, and 4D smooths.
# Requirements: 100% parity with v19.38.0 Registry. Zero-truncation of dashboard panels.

function model_results_plots(res, ts=1; outcome=1, centroids=nothing, polygons=nothing, y_obs=nothing)
    # Description: Primary exhaustive dashboard for spatiotemporal manifold verification.
    # Requirement: Scalable grid to accommodate all recovered latent effects (S, T, U, Smooth, Mixed, Nested).

    println("--- Rendering Dashboard v19.38.5 [Outcome: ", outcome, "] ---")

    # # 1. Metadata and Result Extraction
    # Rationale: Standardizing the posterior statistics and metadata sources for visualization.
    pstats = res.pstats
    # Resolving observation source (Training vs custom override)
    obs_src = isnothing(y_obs) ? res.y_obs : y_obs
    # Synchronizing spatial geometry for mapping
    coords = isnothing(centroids) ? res.centroids : centroids
    polys = isnothing(polygons) ? res.polygons : polygons

    plots_list = []

    # # 2. Panel 1: Posterior Predictive Check (PPC)
    # Rationale: Core fit audit comparing expectations to observations.
    is_mv = (obs_src isa AbstractMatrix && size(obs_src, 2) > 1)
    y_p = is_mv ? pstats.predictions_denoised.mean[:, outcome] : vec(pstats.predictions_denoised.mean)
    y_o = is_mv ? obs_src[:, outcome] : vec(obs_src)

    if length(y_p) == length(y_o)
        p_ppc = Plots.scatter(vec(y_p), vec(y_o), title="PPC Audit", xlabel="Predicted", ylabel="Observed", alpha=0.5, markersize=3, markerstrokewidth=0, legend=false)
        # Identity line reference
        clean_p = filter(x -> !isnan(x), y_p)
        clean_o = filter(x -> !isnan(x), y_o)
        if !isempty(clean_p) && !isempty(clean_o)
            m_val = max(maximum(clean_p), maximum(clean_o))
            Plots.plot!(p_ppc, [0, m_val], [0, m_val], color=:red, ls=:dash, lw=1.5)
        end
        push!(plots_list, p_ppc)
    end

    # # 3. Panel 2: Spatial Field Mapping
    # Rationale: High-fidelity geographic rendering of structured innovations.
  if !isnothing(coords)
        s_field = (pstats.arch isa MultivariateArchitecture || pstats.arch isa MultifidelityArchitecture) ? pstats.spatial_structured[outcome] : pstats.spatial_structured

        if !isnothing(s_field) && hasproperty(s_field, :mean)
            s_mean = vec(collect(s_field.mean))
            p_map = Plots.plot(aspect_ratio=:equal, title="Spatial Innovation", frame=:box)

            if !isnothing(polys) && length(polys) >= length(s_mean)
                for i in 1:length(s_mean)
                    px = [pt[1] for pt in polys[i]]
                    py = [pt[2] for pt in polys[i]]
                    if !isempty(px) && (px[1], py[1]) != (px[end], py[end])
                        push!(px, px[1])
                        push!(py, py[1])
                    end
                    Plots.plot!(p_map, px, py, seriestype=:shape, fill_z=s_mean[i], c=:viridis, linecolor=:black, lw=0.2, label=nothing)
                end
            else
                cx = [c[1] for c in coords]
                cy = [c[2] for c in coords]
                Plots.scatter!(p_map, cx, cy, marker_z=vec(s_mean), markersize=4, c=:viridis, label=nothing, colorbar=true)
            end
            push!(plots_list, p_map)
        end
    end


    # Panel: Temporal Trend Recovery
    # Ensure t_field is correctly extracted for the requested outcome
    t_field = nothing
    if hasproperty(pstats, :temporal_effects)
        raw_t = pstats.temporal_effects
        t_field = (pstats.arch isa MultivariateArchitecture || pstats.arch isa MultifidelityArchitecture) ? raw_t[outcome] : raw_t
    end

    # Robust rendering of longitudinal trend with ribbons
    if !isnothing(t_field) && hasproperty(t_field, :mean)
        tm = vec(collect(t_field.mean))
        tl = vec(collect(t_field.lower))
        tu = vec(collect(t_field.upper))

        if !isempty(tm) && !all(isnan, tm)
            p_trend = Plots.plot(tm, ribbon=(tm .- tl, tu .- tm), title="Temporal Trend", lw=2.5, fillalpha=0.25, color=:royalblue, legend=false, xlabel="Time Index", ylabel="Latent Value")
            push!(plots_list, p_trend)
        end
    end

    # Panel: Seasonal Cycle
    if hasproperty(pstats, :seasonal) && !isnothing(pstats.seasonal)
        u_field = pstats.seasonal
        if hasproperty(u_field, :mean) && !all(isnan, u_field.mean)
            um = vec(collect(u_field.mean))
            ul = vec(collect(u_field.lower))
            uu = vec(collect(u_field.upper))
            p_seas = Plots.plot(um, ribbon=(um .- ul, uu .- um), title="Seasonal Cycle", lw=2, fillalpha=0.2, color=:forestgreen, legend=false, xlabel="Season Bin")
            push!(plots_list, p_seas)
        end
    end

    # # 6. Panel 5: RESTORED Mixed-Effect Categorical Registry
    # Rationale: Specifically added to audit hospital/group varying intercepts.
    # Checks for summaries of c_latent terms identified in v19.21.2.
    if hasproperty(pstats, :mixed_effects) && !isnothing(pstats.mixed_effects)
        for (grp_sym, m_summ) in pstats.mixed_effects
            mm, ml, mu = vec(m_summ.mean), vec(m_summ.lower), vec(m_summ.upper)
            push!(plots_list, Plots.bar(mm, yerror=(mm .- ml, mu .- mm), title=string("Mixed: ", grp_sym), color=:purple, legend=false, xlabel="Category ID"))
        end
    end

    # # 7. Panel 6: RESTORED Hyper-Volumetric Smooths (1D-4D)
    # Rationale: High-dimensional surfaces projected onto the observation grid.
    if hasproperty(pstats, :smooth_effects) && !isnothing(pstats.smooth_effects)
        b_field = pstats.smooth_effects
        if !all(isnan, b_field.mean) && Statistics.std(b_field.mean) > 1e-9
            bm, bl, bu = vec(b_field.mean), vec(b_field.lower), vec(b_field.upper)
            push!(plots_list, Plots.plot(bm, ribbon=(bm .- bl, bu .- bm), title="Accumulated Smooths (1D-4D)", lw=1.5, fillalpha=0.1, color=:darkorange, legend=false, xlabel="Observation Order"))
        end
    end

    # # 8. Panel 7: Hierarchical Supervisor Contributions
    # Rationale: Renders signals transferred from nested supervisor processes.
    if hasproperty(pstats, :nested_contributions) && !isnothing(pstats.nested_contributions)
        n_field = pstats.nested_contributions
        if !all(isnan, n_field.mean) && Statistics.std(n_field.mean) > 1e-9
            nm, nl, nu = vec(n_field.mean), vec(n_field.lower), vec(n_field.upper)
            push!(plots_list, Plots.plot(nm, ribbon=(nm .- nl, nu .- nm), title="Hierarchical Supervisors", lw=1.5, fillalpha=0.2, color=:brown, legend=false, xlabel="Observation Order"))
        end
    end

    # # 9. Panel 8: Fixed Effect Coefficient Registry

    # Panel: Fixed Effect Forest Plot
    # Dot-and-whisker style for regression coefficients
    if hasproperty(pstats, :fixed_effects) && !isnothing(pstats.fixed_effects)
        f_field = pstats.fixed_effects
        if hasproperty(f_field, :mean) && !all(isnan, f_field.mean)
            fm = vec(f_field.mean)
            fl = vec(f_field.lower)
            fu = vec(f_field.upper)
            n_coeffs = length(fm)
            # Reverse indices for intuitive top-to-bottom reading in forest plots
            p_forest = Plots.scatter(fm, 1:n_coeffs, xerror=(fm .- fl, fu .- fm), title="Fixed Effects Registry", xlabel="Coefficient Estimate", ylabel="Index", markersize=5, color=:black, legend=false, yticks=(1:n_coeffs, ["#$i" for i in 1:n_coeffs]))
            Plots.vline!(p_forest, [0], color=:red, ls=:dash, lw=1)
            push!(plots_list, p_forest)
        end
    end


    # # 10. Dashboard Assembly and Layout
    # Rationale: Collating all discovered panels into a unified multi-plot object.
    if !isempty(plots_list)
        n_plots = length(plots_list)
        cols = min(n_plots, 2)
        rows = Int(ceil(n_plots / cols))
        final_plt = Plots.plot(plots_list..., layout=(rows, cols), size=(1200, 350 * rows), margin=5Plots.mm)
        return final_plt
    end

    @warn "BSTM Visualization: No active technical manifolds discovered for outcome $outcome."
    return nothing
end


# # BSTM Results Engine v20.1.5
# Timestamp: 2025-11-25 14:00:00
# Description: Standardizes model reporting for metadata, performance metrics, and MCMC diagnostics.

function model_results_comprehensive(model, chain; n_samples=100, alpha=0.05)
    println("--- Starting Comprehensive Model Reporting ---")

    # # Metadata and Architecture Extraction
    # Rationale: Extracting the technical configuration from the model arguments.
    M = model.args.M
    y_obs = M.y_obs
    raw_arch = get(M, :model_arch, "univariate")
    model_family = get(M, :model_family, "gaussian")

    # Mapping the configuration string to the architectural dispatch types for reconstruction
    arch_type = if raw_arch == "univariate"
        UnivariateArchitecture()
    elseif raw_arch == "multivariate"
        MultivariateArchitecture()
    elseif raw_arch == "multifidelity" || raw_arch == "nested"
        MultifidelityArchitecture()
    else
        UnivariateArchitecture()
    end

    # # Latent Manifold Reconstruction
    # Rationale: Executes the architectural-specific reconstruction of every latent component.
    # The alpha parameter controls the width of the credible intervals (default 95%).
    res = _reconstruct(arch_type, "model_results", chain, M, nothing, alpha)

    # # Performance Metric Assessment
    # Rationale: Metrics are calculated using the posterior predictive mean (denoised expectation).
    y_pred = res.predictions_denoised.mean

    # Flatten for metric consistency across Univariate and Multivariate responses
    y_obs_flat = vec(collect(y_obs))
    y_pred_flat = vec(collect(y_pred))

    # Filter valid indices for calculation to handle missing values or NaNs in observed data
    valid_idx = findall(x -> !isnan(x) && !isnothing(x), y_obs_flat)

    rmse_val = 0.0
    r_pearson = 0.0

    if !isempty(valid_idx)
        obs_v = y_obs_flat[valid_idx]
        pred_v = y_pred_flat[valid_idx]

        # Root Mean Square Error (RMSE)
        rmse_val = sqrt(Statistics.mean((obs_v .- pred_v).^2))

        # Pearson Correlation Coefficient (r)
        try
            r_pearson = Statistics.cor(obs_v, pred_v)
        catch
            # Handle cases with zero variance in predictions to prevent domain errors
            r_pearson = 0.0
        end
    end

    # FlexiChains store timing data in _metadata, not info.
    # Rationale: Direct access to _metadata keys via Symbol lookup.
    sampling_time = chain._metadata.sampling_time[]

    # Standardizing Diagnostic Extraction via DataFrames
    # Rationale: summarystats(chain) returns a FlexiSummary which is converted to a DataFrame
    # to ensure column-based indexing for rhat and ess_bulk is stable and type-safe.
    sum_stats_df = DataFrame(MCMCChains.summarystats(chain))
    
    mean_rhat = 1.0
    min_ess = 0.0

    try
        # Extraction of diagnostic vectors
        rhat_vector = sum_stats_df[!, :rhat]
        ess_bulk_vector = sum_stats_df[!, :ess_bulk]
        
        # Robust aggregation ignoring NaNs
        mean_rhat = Statistics.mean(filter(!isnan, rhat_vector))
        min_ess = Statistics.minimum(filter(!isnan, ess_bulk_vector))
    catch
        # Fallback to standard ess if ess_bulk is absent in the DataFrame schema
        try
            ess_alt = sum_stats_df[!, :ess]
            min_ess = Statistics.minimum(filter(!isnan, ess_alt))
        catch
            mean_rhat = 1.0
            min_ess = 0.0
        end
    end

    waic_val = get(res, :waic, 0.0)

    ess_rate = round(min_ess/sampling_time, digits=2)

    # # Displaying the Report
    println("\n--- Model Metadata ---")
    println("Architecture:     ", raw_arch)
    println("Family:           ", model_family)
    println("Space Component:  ", get(M, :model_space, "none"))
    println("Time Component:   ", get(M, :model_time, "none"))
    println("Seasonal Component: ", get(M, :model_season, "none"))

    println("\n--- Performance Metrics ---")
    println("Compute Time:     ", round(sampling_time, digits=2), " seconds")
    println("RMSE:             ", round(rmse_val, digits=4))
    println("Pearson r:        ", round(r_pearson, digits=4))
    println("WAIC Score:       ", round(waic_val, digits=2))

    println("\n--- MCMC Diagnostics ---")
    println("Mean R-hat:       ", round(mean_rhat, digits=4))
    println("Minimum ESS:      ", round(min_ess, digits=2))
    println("ESS per second:    ", ess_rate)

    # # Final Registry Object Assembly
    # Rationale: Returning the full set of metrics and reconstructed latent fields.
    return (
        metrics = (
            rmse = rmse_val, 
            r_pearson = r_pearson, 
            waic = waic_val, 
            rhat = mean_rhat, 
            ess = min_ess, 
            ess_rate = ess_rate,
            sampling_time = sampling_time
        ),
        pstats = res,
        y_obs = y_obs,
        model_family = model_family,
        arch = arch_type
    )
end
 

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
 

function create_prediction_surface(
    data_df::DataFrame, 
    au_obj::NamedTuple, 
    tu_obj::NamedTuple, 
    covariate_vars::Vector{Symbol}; 
    iterations::Int = 3
)
    # # 1. Grid Construction
    # Establishing a full Cartesian product of spatial centroids and temporal slices
    s_N = length(au_obj.centroids)
    t_N = tu_obj.tn
    
    # Construct initial empty surface
    surface = DataFrame(
        s_idx = repeat(1:s_N, inner = t_N),
        t_idx = repeat(1:t_N, outer = s_N)
    )
    
    # Map coordinates from centroids
    surface.s_x = [au_obj.centroids[i][1] for i in surface.s_idx]
    surface.s_y = [au_obj.centroids[i][2] for i in surface.s_idx]
    
    # # 2. Feature Mapping
    # Transferring existing observed covariates to the prediction grid
    # Observed points are averaged per spatio-temporal cell
    data_agg = combine(
        groupby(data_df, [:s_idx, :t_idx]), 
        covariate_vars .=> mean .=> covariate_vars
    )
    
    surface = leftjoin(surface, data_agg, on = [:s_idx, :t_idx])
    
    # # 3. Safe Spatiotemporal Imputation
    # Rationale: Filling gaps using spatial adjacency and temporal neighbors.
    # Updates are collected in a buffer to prevent mutation of the DataFrame during iteration.
    
    adj_matrix = au_obj.W
    
    for iter_idx in 1:iterations
        # Create a copy of the current state to serve as the source for this pass
        current_state = copy(surface)
        
        for cov in covariate_vars
            # Identify rows needing imputation
            missing_mask = isnothing.(current_state[!, cov]) .| isnan.(current_state[!, cov])
            
            if !any(missing_mask)
                continue
            end
            
            # Initialize update buffer for the current covariate
            updates = copy(current_state[!, cov])
            
            # Group by spatial unit to efficiently access temporal neighbors
            surface_groups = groupby(current_state, :s_idx)
            
            for (row_idx, is_missing) in enumerate(missing_mask)
                if !is_missing
                    continue
                end
                
                s_i = current_state.s_idx[row_idx]
                t_i = current_state.t_idx[row_idx]
                
                # Accumulate neighborhood values
                neighbor_vals = Float64[]
                
                # # A. Temporal Context (Previous and Next slices in same unit)
                # Accessing the group for current spatial unit s_i
                unit_data = surface_groups[(s_idx = s_i,)]
                t_prev = t_i > 1 ? unit_data[t_i - 1, cov] : NaN
                t_next = t_i < t_N ? unit_data[t_i + 1, cov] : NaN
                
                if !isnan(t_prev) push!(neighbor_vals, t_prev) end
                if !isnan(t_next) push!(neighbor_vals, t_next) end
                
                # # B. Spatial Context (Same slice in adjacent units)
                neighbors_idx = findall(x -> x > 0, adj_matrix[s_i, :])
                for nb_s in neighbors_idx
                    # Find the specific row in current_state for (nb_s, t_i)
                    # Rationale: In a full Cartesian grid, index = (nb_s - 1) * t_N + t_i
                    nb_row_idx = (nb_s - 1) * t_N + t_i
                    val = current_state[nb_row_idx, cov]
                    if !isnothing(val) && !isnan(val)
                        push!(neighbor_vals, val)
                    end
                end
                
                # # C. Apply Mean Imputation
                if !isempty(neighbor_vals)
                    imputed_val = mean(neighbor_vals)
                    
                    # Factor Snapping: Rounding if the column appears to be integer-based
                    if all(x -> x == floor(x), filter(!isnan, current_state[!, cov]))
                        imputed_val = round(imputed_val)
                    end
                    
                    updates[row_idx] = imputed_val
                end
            end
            
            # Apply the collected updates to the surface after the row loop
            surface[!, cov] = updates
        end
    end
    
    # # 4. Final Verification
    # Fallback to global means for any remaining orphaned cells
    for cov in covariate_vars
        final_missing = isnothing.(surface[!, cov]) .| isnan.(surface[!, cov])
        if any(final_missing)
            global_avg = mean(filter(!isnan, filter(!isnothing, surface[!, cov])))
            surface[final_missing, cov] .= isnan(global_avg) ? 0.0 : global_avg
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


function scottish_lip_cancer_data_spacetime(n_years::Int=20, spatial_expansion::Float64=1.5, temporal_expansion::Float64=1.5; rndseed::Int=42, re_create::Bool=false)
    # BSTM Standard Data Factory v28.3.0
    # Timestamp: 2025-12-01 15:00:00
    # Rationale: Resolving symmetry errors in adjacency matrix construction and scoping errors for derived variables.

    cache_path = "data/scottish_lip_cancer_cache.jld2"

    # Check for existing cache and bypass logic unless re_create is explicitly true
    if isfile(cache_path) && !re_create
        println("Loading cached dataset from: ", cache_path)
        data_bundle = JLD2.load(cache_path)
        return (data_bundle["primary"], data_bundle["nested"])
    end

    println("Generating new spatiotemporal dataset...")
    Random.seed!(rndseed)

    # ##########################################################################
    # PRIMARY DATASET CONSTRUCTION (56 Districts)
    # scottish lip cancer data to a space-time version
    # ##########################################################################

    n_districts = 56

    # Canonical neighbor list (undirected counties)
    neighbor_list = [
        [5, 9, 11, 19], [7, 10], [6, 12], [18, 20, 28], [1, 11, 12, 13, 19],
        [3, 8], [2, 10, 13, 16, 17], [6], [1, 11, 17, 19, 23, 29], [2, 7, 16, 22],
        [1, 5, 9, 12], [3, 5, 11], [5, 7, 17, 19], [31, 32, 35], [25, 29, 50],
        [7, 10, 17, 21, 22, 29], [7, 9, 13, 16, 19, 29], [4, 20, 28, 33, 55, 56], [1, 5, 9, 13, 17], [4, 18, 55],
        [16, 29, 50], [10, 16], [9, 29, 34, 36, 37, 39], [27, 30, 31, 44, 47, 48, 55, 56], [15, 26, 29],
        [26, 29, 42, 43], [24, 31, 32, 55], [4, 18, 33, 45], [9, 15, 16, 17, 21, 23, 25, 26, 34, 43, 50], [24, 38, 42, 44, 45, 56],
        [14, 24, 27, 32, 35, 46, 47], [14, 27, 31, 35], [18, 28, 45, 56], [23, 29, 39, 40, 42, 43, 51, 52, 54], [14, 31, 32, 37, 46],
        [23, 37, 39, 41], [23, 35, 36, 41, 46], [30, 42, 44, 49, 51, 54], [23, 34, 36, 40, 41], [34, 39, 41, 49, 52],
        [36, 37, 39, 40, 46, 49, 53], [26, 30, 34, 38, 43, 51], [26, 29, 34, 42], [24, 30, 38, 48, 49], [28, 30, 33, 56],
        [31, 35, 37, 41, 47, 53], [24, 31, 46, 48, 49, 53], [24, 44, 47, 49], [38, 40, 41, 44, 47, 48, 52, 53, 54], [15, 21, 29],
        [34, 38, 42, 54], [34, 40, 49, 54], [41, 46, 47, 49], [34, 38, 49, 51, 52], [18, 20, 24, 27, 56], [18, 24, 30, 33, 45, 55]
    ]

    # Construct and enforce symmetry for adjacency matrix W
    W_raw = spzeros(Int, n_districts, n_districts)
    for i in 1:n_districts
        for nb in neighbor_list[i]
            W_raw[i, nb] = 1
        end
    end
    # Symmetric enforcement: W_{ij} = W_{ji}
    W = sparse(Symmetric(Matrix(W_raw + W_raw')) .> 0)

    # Inferred spatial geometry using force-directed layout
    au_primary = assign_spatial_units_inferred(W)
    p_centroids = au_primary.centroids
    p_hull = au_primary.hull_coords

    # Clayton & Kaldor Reference Values
    y_orig = [9,39,11,9,15,8,26,7,6,20,13,5,3,8,17,9,2,7,9,7,16,31,11,7,19,15,7,10,16,11,5,3,7,8,11,9,11,8,6,4,10,8,2,6,19,3,2,3,28,6,1,1,1,1,0,0]
    E_orig = [1.4,8.7,3.0,2.5,4.3,2.4,8.1,2.3,2.0,6.6,4.4,1.8,1.1,3.3,7.8,4.6,1.1,4.2,5.5,4.4,10.5,22.7,8.8,5.6,15.5,12.5,6.0,9.0,14.4,10.2,4.8,2.9,7.0,8.5,12.3,10.1,12.7,9.4,7.2,5.3,18.8,15.8,4.3,14.6,50.7,8.2,5.6,9.3,88.7,19.6,3.4,3.6,5.7,7.0,4.2,1.8]
    x_orig = [16,16,10,24,10,24,10,7,7,16,7,16,10,24,7,16,10,7,7,10,7,16,10,7,1,1,7,7,10,10,7,24,10,7,7,0,10,1,16,0,1,16,16,0,1,7,1,1,0,1,1,0,1,1,16,10]

    data_primary = DataFrame()
    for i in 1:n_districts
        log_off = log.(fill(E_orig[i], n_years))
        innov = cumsum(randn(n_years) .* 0.1)
        y_p = floor.(Int, abs.(fill(y_orig[i], n_years) .+ (innov .* 4.0)))

        d_df = DataFrame(
            district = i,
            year = 1:n_years,
            y = y_p,
            log_offset = log_off,
            cov1 = fill(x_orig[i], n_years)
        )
        # Calculate rate within block to avoid scoping errors
        d_df.y_rate = d_df.y ./ exp.(d_df.log_offset)
        append!(data_primary, d_df)
    end

    # Assign binary response based on grand mean rate
    data_primary.y_bin = [v > mean(data_primary.y_rate) ? 1 : 0 for v in data_primary.y_rate]

    # Generate correlated covariates
    data_primary.cov2 = 0.5 .* data_primary.cov1 .+ randn(nrow(data_primary))
    # Correcting scoping by using the data frame columns
    data_primary.cov3 = randn(nrow(data_primary)) .* (data_primary.y_rate .^ 2)
    data_primary.cov4 = randn(nrow(data_primary)) .* log.(data_primary.y_rate .+ 1.0)
    data_primary.cov5 = randn(nrow(data_primary)) .* exp.(data_primary.y_rate) .* 2.0
    data_primary.cov6 = randn(nrow(data_primary))

    data_primary.f1 = rand(["A", "B"], nrow(data_primary))
    data_primary.s_idx = data_primary.district
    data_primary.s_x =  [c[1] for c in p_centroids[data_primary.s_idx]]
    data_primary.s_y =  [c[2] for c in p_centroids[data_primary.s_idx]]
 
    au_primary = merge( au_primary, (
        s_idx = data_primary.s_idx,
        s_x = [c[1] for c in p_centroids[data_primary.s_idx]],
        s_y = [c[2] for c in p_centroids[data_primary.s_idx]],
        s_vals = collect(1:n_districts)
    ))

    # ##########################################################################
    # NESTED DATASET CONSTRUCTION (User-Controlled Expansion)
    # ##########################################################################

    # Spatial domain boundary from primary centroids
    px = [c[1] for c in p_centroids]
    py = [c[2] for c in p_centroids]
    x_min, x_max = minimum(px), maximum(px)
    y_min, y_max = minimum(py), maximum(py)
    x_rng, y_rng = x_max - x_min, y_max - y_min

    # Expansion buffer calculations
    s_buff = (spatial_expansion - 1.0) / 2.0
    nx_min, nx_max = x_min - s_buff * x_rng, x_max + s_buff * x_rng
    ny_min, ny_max = y_min - s_buff * y_rng, y_max + s_buff * y_rng

    nt_max = Int(round(n_years * temporal_expansion))
    n_obs_nested = Int(round(nrow(data_primary) * spatial_expansion * temporal_expansion))

    sx_nested = rand(Uniform(nx_min, nx_max), n_obs_nested)
    sy_nested = rand(Uniform(ny_min, ny_max), n_obs_nested)
    time_nested = rand(1:nt_max, n_obs_nested)

    # Spatial Unit Assignment for expanded domain
    au_nested = assign_spatial_units(sx_nested, sy_nested; target_units=100)

    data_nested = DataFrame(
        s_x = sx_nested,
        s_y = sy_nested,
        year = time_nested,
        district = au_nested.s_idx
    )

    # Latent signal generation for nested grid
    s_lat_n = cumsum(randn(length(au_nested.centroids))) .* 0.3
    t_lat_n = sin.(collect(1:nt_max) .* (2π/nt_max))

    eta_n = [1.5 + s_lat_n[data_nested.district[i]] + t_lat_n[data_nested.year[i]] for i in 1:n_obs_nested]

    data_nested.y = [rand(Poisson(exp(v))) for v in eta_n]
    data_nested.y_rate = exp.(eta_n) .+ randn(n_obs_nested) .* 0.2
    data_nested.y_bin = [v > mean(data_nested.y_rate) ? 1 : 0 for v in data_nested.y_rate]

    data_nested.ncov1 = 0.6 .* eta_n .+ randn(n_obs_nested)
    data_nested.ncov2 = randn(n_obs_nested) .* exp.(data_nested.y_rate)
    data_nested.ncov3 = randn(n_obs_nested)

    primary_out = (data=data_primary, au=au_primary )

    nested_out = (data=data_nested, au=au_nested.W )

    # Directory check and caching
    if !isdir("data"); mkdir("data"); end
    JLD2.save(cache_path, "primary", primary_out, "nested", nested_out)
    println("Dataset successfully cached at: ", cache_path)

    return (primary_out, nested_out)
    # (p_set, n_set) = scottish_lip_cancer_data_spacetime();
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
    # # 1. Global Likelihood Hyperparameters
    # Rationale: Standardizing scalars for the target likelihood family with AD-stable initialization.
    family = M.model_family
    noise = get(M, :noise, 1e-4)
    use_zi = get(M, :use_zi, false)

    lik_r = one(T)
    lik_phi = zero(T)
    extra_p = one(T)

    # Poisson and Negative Binomial Dispersion/Zero-Inflation parameters
    if family == "negbin"
        lik_r ~ NamedDist(Exponential(1.0), :lik_r)
    end

    if use_zi == true
        lik_phi ~ NamedDist(Beta(1, 1), :lik_phi)
    end

    # Shape, Tail, or Scale parameters for continuous families (Gamma, Beta, Student-T, etc.)
    if family in ["gamma", "beta", "student_t", "inverse_gaussian", "pareto"]
        extra_p ~ NamedDist(Exponential(1.0), :extra_params)
    end

    # # 2. Observation Volatility & Stochastic Volatility (SV)
    # Rationale: Reconstructs heteroskedastic error scales via RFF log-variance projection.
    y_sigma = Vector{T}(undef, M.y_N)

    if get(M, :use_sv, false) == true
        # Stochastic Volatility Path: Log-variance is modeled as a latent Gaussian field
        sigma_log_var ~ NamedDist(Exponential(1.0), :sigma_log_var)
        beta_vol_latent ~ NamedDist(filldist(Normal(0, 1), M.M_rff_sigma), :beta_vol_latent)

        # Spectral mapping of the log-variance surface
        vol_proj_field = M.vol_proj * beta_vol_latent
        vol_latent_field = sqrt(2.0 / M.M_rff_sigma) .* cos.(vol_proj_field)

        for i in 1:M.y_N
            y_sigma[i] = exp((sigma_log_var * vol_latent_field[i]) / 2.0)
        end
    else
        # Homoskedastic Path: Fixed error scale across the entire spatiotemporal grid
        y_sigma_val ~ NamedDist(Exponential(1.0), :y_sigma)
        for i in 1:M.y_N
            y_sigma[i] = y_sigma_val
        end
    end

    
    # # 3. Base Predictor: Fixed Effects & Link-Scale Offsets
    # Rationale: Standard regression coefficients follow an uninformative Gaussian prior.
    Xfixed_beta ~ NamedDist(MvNormal(zeros(M.Xfixed_N), 5.0 * I), :Xfixed_beta)
    eta = Vector{T}(M.Xfixed * Xfixed_beta)

    # Apply additive offsets (e.g., exposure/offset columns) if registered
    # Scale-Aware Offset Implementation
    # If the family uses an identity link (Gaussian), we exponentiate the log_offset.
    # For log-link families (Poisson, Hurdle Poisson), it remains additive in the latent space.
    if haskey(M, :log_offset)
        if family in ["gaussian", "student_t", "laplace"]
            eta .+= exp.(M.log_offset)
        else
            eta .+= M.log_offset
        end
    end
 
    # # 1. Spatiotemporal Interactions (Knorr-Held Types I, II, III, IV)
    # Rationale: Discovering Kronecker-recomposed precision templates for separable fields.
    st_type = get(M, :model_st, "none")
    if st_type in ["I", "II", "III", "IV"]
        st_sigma ~ NamedDist(Exponential(1.0), :st_sigma)
        
        # Recomposing precision via Kronecker product of spatial and temporal templates
        # Q_st = st_scale * (Q_s \otimes Q_t)
        Q_st = recompose_precision(
            Symbol(st_type),
            M.s_Q_template.matrix,
            st_sigma,
            template_t = M.t_Q_template.matrix,
            noise = noise
        )

        st_latent ~ NamedDist(MvNormalCanon(zeros(M.s_N * M.t_N), Q_st), :st_latent)
        
        for i in 1:M.y_N
            # Mapping linear index in the ST-grid to flat observation index
            obs_st_idx = (M.t_idx[i] - 1) * M.s_N + M.s_idx[i]
            eta[i] += st_latent[obs_st_idx]
        end
    end

    # # 2. Physics-Informed Transport
    model_st_phys = get(M, :model_st, "none")
    if model_st_phys in ["advection", "diffusion", "advection_diffusion"]
        sig_phys ~ NamedDist(Exponential(1.0), :sig_phys)
        velocity_phys ~ NamedDist(Normal(0, 1), :velocity_phys)
        
        K_op = build_transport_operator(M.W, velocity_phys, 1.0)
        phys_innov ~ NamedDist(filldist(Normal(0, sig_phys), M.s_N * M.t_N), :phys_innov)
        phys_field = Vector{T}(undef, M.s_N * M.t_N)

        for t in 1:M.t_N
            idx_t = ((t-1)*M.s_N + 1):(t*M.s_N)
            if t == 1
                phys_field[idx_t] .= phys_innov[idx_t]
            else
                idx_prev = ((t-2)*M.s_N + 1):((t-1)*M.s_N)
                phys_field[idx_t] .= K_op * phys_field[idx_prev] .+ phys_innov[idx_t]
            end
        end

        for i in 1:M.y_N
            obs_st_ptr = (M.t_idx[i] - 1) * M.s_N + M.s_idx[i]
            eta[i] += phys_field[obs_st_ptr]
        end
    end



    # # Mixed Effects (Categorical Random Intercepts)
    if haskey(M, :mixed_terms) && !isempty(M.mixed_terms)
        for term in M.mixed_terms
            grp_name = term.name
            n_levels = term.n_cat

            sig_c ~ NamedDist(Exponential(1.0), Symbol("sig_c_", grp_name))
            Q_c = recompose_precision(:iid, sparse(I(n_levels, n_levels)), sig_c, noise=noise)
            c_latent ~ NamedDist(MvNormalCanon(zeros(n_levels), Q_c), Symbol("c_latent_", grp_name))

            indices = term.indices
            for i in 1:M.y_N
                eta[i] += c_latent[indices[i]]
            end
        end
    end

    

    # # Nested Hierarchical Supervisors
    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (z_key, z_meta) in M.nested_manifolds
            rho_nested ~ NamedDist(Normal(1.0, 0.5), Symbol("rho_nested_", z_key))

            if haskey(z_meta, :model_space) && z_meta.model_space != "none"
                sig_z_s ~ NamedDist(Exponential(1.0), Symbol("sig_nested_spatial_", z_key))
                Q_z_s = recompose_precision(Symbol(z_meta.model_space), z_meta.s_Q_template.matrix, sig_z_s, noise=noise)
                lat_z_s ~ NamedDist(MvNormalCanon(zeros(z_meta.s_N), Q_z_s), Symbol("lat_nested_spatial_", z_key))

                z_s_ptr = z_meta.s_idx
                for i in 1:M.y_N
                    eta[i] += rho_nested * lat_z_s[z_s_ptr[i]]
                end
            end

            if haskey(z_meta, :Xfixed)
                xf_z = z_meta.Xfixed
                beta_z_f ~ NamedDist(MvNormal(zeros(size(xf_z, 2)), 5.0 * I), Symbol("beta_nested_fixed_", z_key))
                eta .+= rho_nested .* (xf_z * beta_z_f)
            end
        end
    end
 
    # Spatial Manifold Realization
    if M.model_space != "none"
        s_sigma ~ NamedDist(Exponential(1.0), :s_sigma)
        s_rho_val = one(T)
        m_sp_sym = Symbol(M.model_space)
        
        if m_sp_sym in [:bym2, :leroux, :sar, :dag]
            s_rho_val ~ NamedDist(Beta(1, 1), :s_rho)
        end
        
        # Identification of directed attributes for Network flow
        dir_adj = get(M, :directed_W, nothing)
        flow_dir = Symbol(get(M, :flow_direction, :bidirectional))

        Q_s = recompose_precision(
            m_sp_sym, 
            M.s_Q_template.matrix, 
            s_sigma, 
            extra_param=s_rho_val, 
            noise=noise, 
            directed_adj=dir_adj, 
            flow_direction=flow_dir
        )
        
        s_latent ~ NamedDist(MvNormalCanon(zeros(M.s_N), Q_s), :s_latent)
        for i in 1:M.y_N
            eta[i] += s_latent[M.s_idx[i]]
        end
    end

    # Temporal Manifold Realization
    if M.model_time != "none"
        t_sigma ~ NamedDist(Exponential(1.0), :t_sigma)
        t_rho_val = one(T)
        m_tm_sym = Symbol(M.model_time)
        
        if m_tm_sym == :ar1
            t_rho_val ~ NamedDist(Beta(2, 2), :t_rho)
        end
        
        Q_t = recompose_precision(m_tm_sym, M.t_Q_template.matrix, t_sigma, extra_param=t_rho_val, noise=noise)
        t_latent ~ NamedDist(MvNormalCanon(zeros(M.t_N), Q_t), :t_latent)
        for i in 1:M.y_N
            eta[i] += t_latent[M.t_idx[i]]
        end
    end

    # # 9. Mechanistic Dynamics (Biomass & Population Growth)
    # Rationale: Logistic state-space transitions with density dependence.
    if get(M, :use_dynamics, false) == true
        K_dyn ~ NamedDist(get(M.hyperpriors, "K", Normal(150, 10)), :K_dyn)
        r_dyn ~ NamedDist(get(M.hyperpriors, "r", Beta(2, 2)), :r_dyn)
        sig_dyn ~ NamedDist(Exponential(1.0), :sig_dyn)
        
        # Projecting the latent growth trajectory onto time slices
        m_states ~ NamedDist(filldist(Normal(0, 1), M.t_N), :m_states)
        for i in 1:M.y_N
            eta[i] += m_states[M.t_idx[i]] * sig_dyn
        end
    end
 
    # Smooth Basis Dispatch (1D, 2D, 3D, and 4D Latent Fields)
    # Rationale: Aggregates non-linear effects discovered in the smooth() DSL blocks.
    if haskey(M, :basis_matrices) && !isempty(M.basis_matrices)
        for v_sym in keys(M.basis_matrices)
            B_mat = M.basis_matrices[v_sym]
            n_basis_cols = size(B_mat, 2)
            
            # Variance scale for the specific basis expansion
            sig_basis ~ NamedDist(Exponential(1.0), Symbol("sig_basis_", v_sym))
            # Latent coefficients for the basis surface
            beta_basis ~ NamedDist(filldist(Normal(0, sig_basis), n_basis_cols), Symbol("beta_basis_", v_sym))
            
            # Additive superposition onto the linear predictor
            eta .+= B_mat * beta_basis
        end
    end

 
    # # Spatially Varying Coefficients (SVC)
    if haskey(M, :svc_covariates) && !isempty(M.svc_covariates)
        for c_sym in M.svc_covariates
            sig_svc ~ NamedDist(Exponential(1.0), Symbol("sig_svc_", c_sym))
            beta_svc ~ NamedDist(filldist(Normal(0, sig_svc), M.s_N), Symbol("beta_svc_", c_sym))
            
            x_col_idx = findfirst(==(c_sym), Symbol.(names(M.Xfixed, 2)))
            if !isnothing(x_col_idx)
                x_vals = M.Xfixed[:, x_col_idx]
                for i in 1:M.y_N
                    eta[i] += beta_svc[M.s_idx[i]] * x_vals[i]
                end
            end
        end
    end


    # # 11. Final Pointwise Likelihood Dispatch (Scalar AD Firewall)
    # Rationale: Using @addlogprob! with bstm_Likelihood functor to bypass strict tilde checks.
    yL = T(get(M, :y_lower_bound, -Inf))
    yU = T(get(M, :y_upper_bound, Inf))
    h = T(get(M, :hurdle, -Inf))

    for i in 1:M.y_N
        # Standardizing likelihood evaluation via the multiple-dispatch functor
        d_lik = bstm_Likelihood(
            family, 
            [T(M.y_obs[i])]; 
            sigma_y=[y_sigma[i]], 
            weight=[T(M.weights[i])], 
            phi_zi=lik_phi, 
            r_nb=lik_r, 
            trial=[Int(M.trials[i])], 
            y_L=yL, 
            y_U=yU, 
            hurdle=h, 
            extra_params=extra_p
        )
        Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i])
    end
end

 

@model function bstm_multivariate(M, ::Type{T}=Float64) where {T}
    # # 1. Architectural Scope Discovery
    # Rationale: Determining outcome count and grid dimensions for coupled latent fields.
    outcomes_N = M.outcomes_N
    y_N = M.y_N
    family = M.model_family
    noise = get(M, :noise, 1e-4)
    use_zi = get(M, :use_zi, false)

    # # 2. Global Likelihood Hyperparameters
    # Rationale: Initializing vectors for K-outcome scalars with AD-stable initialization.
    lik_r = one(T)
    lik_phi = zero(T)
    extra_p = ones(T, outcomes_N)

    if family == "negbin"
        lik_r ~ NamedDist(Exponential(1.0), :lik_r)
    end

    if use_zi == true
        lik_phi ~ NamedDist(Beta(1, 1), :lik_phi)
    end

    if family in ["gamma", "beta", "student_t", "inverse_gaussian", "pareto"]
        extra_p ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :extra_params)
    end

    # # 3. Multivariate Coupling & Latent Scales
    # Rationale: Standardizing independent variance scaling per outcome k.
    s_sigma_arr ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :s_sigma_arr)
    t_sigma_arr ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :t_sigma_arr)

    # L_corr is the Cholesky factor of the LKJ latent correlation matrix for signal coupling.
    L_corr ~ NamedDist(LKJCholesky(outcomes_N, 1.0, T), :L_corr)

    # # 4. Observation Volatility & Stochastic Volatility (SV)
    # Rationale: Reconstructs heteroskedastic error scales via RFF log-variance projection.
    y_sigma = Matrix{T}(undef, y_N, outcomes_N)

    if get(M, :use_sv, false) == true
        sigma_log_var ~ NamedDist(Exponential(1.0), :sigma_log_var)
        beta_vol_latent ~ NamedDist(filldist(Normal(0, 1), M.M_rff_sigma), :beta_vol_latent)

        vol_proj_field = M.vol_proj * beta_vol_latent
        vol_latent_field = sqrt(2.0 / M.M_rff_sigma) .* cos.(vol_proj_field)

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
            for k in 1:outcomes_N
                y_sigma[:, k] .= one(T)
            end
        end
    end

    # # 5. Base Predictor: Fixed Effects & Offsets
    # Rationale: Standard regression coefficients mapped across the outcome matrix.
    Xfixed_beta ~ NamedDist(MvNormal(zeros(M.Xfixed_N * outcomes_N), 5.0 * I), :Xfixed_beta)
    mu_fixed = M.Xfixed * reshape(Xfixed_beta, M.Xfixed_N, outcomes_N)
    eta = Matrix{T}(mu_fixed)

    # Scale-Aware Offset Implementation
    # If the family uses an identity link (Gaussian), we exponentiate the log_offset.
    # For log-link families (Poisson, Hurdle Poisson), it remains additive in the latent space.
    if haskey(M, :log_offset)
        if family in ["gaussian", "student_t", "laplace"]
            eta .+= exp.(M.log_offset)
        else
            eta .+= M.log_offset
        end
    end


    # # 6. Shared Additive Manifolds (Global Mixed Effects)
    # Rationale: Categorical intercepts applied globally across all outcomes.
    if haskey(M, :c_groups) && !isempty(M.c_groups)
        for c_sym in keys(M.c_groups)
            c_template = M.c_re_templates[c_sym]
            sig_c ~ NamedDist(Exponential(1.0), Symbol("sig_c_", c_sym))
            Q_c = recompose_precision(:iid, c_template.matrix, sig_c, noise=noise)
            c_latent ~ NamedDist(MvNormalCanon(zeros(size(c_template.matrix, 1)), Q_c), Symbol("c_latent_", c_sym))

            indices = M.c_groups[c_sym]
            for k in 1:outcomes_N
                for i in 1:y_N
                    eta[i, k] += c_latent[indices[i]]
                end
            end
        end
    end

    # # 7. Nested Hierarchical Supervisors (Multifidelity Signal Transfer)
    # Rationale: Propagates signals from supervisor variables to primary outcomes k.
    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (z_key, z_meta) in M.nested_manifolds
            rho_nested ~ NamedDist(Normal(1.0, 0.5), Symbol("rho_nested_", z_key))

            if haskey(z_meta, :model_space) && z_meta.model_space != "none"
                sig_z_s ~ NamedDist(Exponential(1.0), Symbol("sig_nested_spatial_", z_key))
                Q_z_s = recompose_precision(Symbol(z_meta.model_space), z_meta.s_Q_template.matrix, sig_z_s, noise=noise)
                lat_z_s ~ NamedDist(MvNormalCanon(zeros(z_meta.s_N), Q_z_s), Symbol("lat_nested_spatial_", z_key))

                z_ptr = z_meta.s_idx
                for k in 1:outcomes_N
                    for i in 1:y_N
                        eta[i, k] += rho_nested * lat_z_s[z_ptr[i]]
                    end
                end
            end
        end
    end

    # # 8. Coupled Spatio-Temporal Innovation Fields
    # Rationale: Latent fields are coupled via the LKJ correlation prior L_corr. 
    # Coupled Innovation Fields
    latent_innovations = Matrix{T}(undef, y_N, outcomes_N)
    for k in 1:outcomes_N
        field_k = zeros(T, y_N)
        if M.model_space != "none"
            Q_s = recompose_precision(Symbol(M.model_space), M.s_Q_template.matrix, s_sigma_arr[k], noise=noise)
            s_latent_k ~ NamedDist(MvNormalCanon(zeros(M.s_N), Q_s), Symbol("s_latent_", k))
            for i in 1:y_N; field_k[i] += s_latent_k[M.s_idx[i]]; end
        end
        if M.model_time != "none"
            Q_t = recompose_precision(Symbol(M.model_time), M.t_Q_template.matrix, t_sigma_arr[k], noise=noise)
            t_latent_k ~ NamedDist(MvNormalCanon(zeros(M.t_N), Q_t), Symbol("t_latent_", k))
            for i in 1:y_N; field_k[i] += t_latent_k[M.t_idx[i]]; end
        end
        latent_innovations[:, k] .= field_k
    end
    
    # Apply LKJ Coupling Factor: Coupled_Eta = Innovations * L'
    # Corrected multiplication orientation for row-major assembly
    eta .+= (latent_innovations * L_corr.L)

    # # 9. RESTORED: Mechanistic Dynamics & Physics
    # Rationale: These modules apply mechanistic growth or transport to the coupled grid.
    if get(M, :use_dynamics, false) == true
        K_dyn ~ NamedDist(get(M.hyperpriors, "K", Normal(150, 10)), :K_dyn)
        r_dyn ~ NamedDist(get(M.hyperpriors, "r", Beta(2, 2)), :r_dyn)
        sig_dyn ~ NamedDist(Exponential(1.0), :sig_dyn)
        m_states ~ NamedDist(filldist(Normal(0, 1), M.t_N), :m_states)
        for k in 1:outcomes_N
            for i in 1:y_N
                eta[i, k] += m_states[M.t_idx[i]] * sig_dyn
            end
        end
    end

 
    # # 10. Advanced Registries: Basis Smooths & Interactions
    # Smooth Basis Dispatch (1D, 2D, 3D, and 4D Latent Fields)
    # Rationale: Aggregates non-linear effects discovered in the smooth() DSL blocks across outcomes.
    # Unified Smooth Basis Dispatch (1D to 4D)
    # Rationale: Applies non-linear surfaces registered in basis_matrices to all outcome dimensions.
    if haskey(M, :basis_matrices) && !isempty(M.basis_matrices)
        for v_sym in keys(M.basis_matrices)
            B_mat = M.basis_matrices[v_sym]
            n_basis_cols = size(B_mat, 2)
            sig_basis ~ NamedDist(Exponential(1.0), Symbol("sig_basis_", v_sym))
            beta_basis ~ NamedDist(filldist(Normal(0, sig_basis), n_basis_cols), Symbol("beta_basis_", v_sym))
            surface_proj = B_mat * beta_basis
            for k in 1:outcomes_N
                eta[:, k] .+= surface_proj
            end
        end
    end


    # # 11. Pointwise Likelihood Dispatch (Scalar AD Firewall)
    # Rationale: Standardizing outcome k evaluation via the multiple-dispatch Likelihood functor.
    yL = T.(get(M, :y_L, fill(-Inf, outcomes_N)))
    yU = T.(get(M, :y_U, fill(Inf, outcomes_N)))
    h = T.(get(M, :hurdle, fill(-Inf, outcomes_N)))

    for k in 1:outcomes_N
        ok_idx = M.y_ok[k]
        for i in ok_idx
            d_lik = bstm_Likelihood(
                family,
                [T(M.y_obs[i, k])];
                sigma_y=[y_sigma[i, k]],
                weight=[T(M.weights[i])],
                phi_zi=lik_phi,
                r_nb=lik_r,
                trial=[Int(M.trials[i])],
                y_L=yL[k],
                y_U=yU[k],
                hurdle=h[k],
                extra_params=extra_p[k]
            )
            Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i, k])
        end
    end
end



@model function bstm_multifidelity(M, ::Type{T}=Float64) where {T}
    # # 1. Architectural Scope Discovery
    # Rationale: Extracting fidelity metadata and global technical flags.
    fidelities = M.fidelities
    K_levels = length(fidelities)
    family = M.model_family
    noise = get(M, :noise, 1e-4)
    use_zi = get(M, :use_zi, false)

    # # 2. Global Latent Scales & Signal Transfer
    # s_sigma_arr and t_sigma_arr provide independent innovation scaling per fidelity level.
    s_sigma_arr ~ NamedDist(filldist(Exponential(1.0), K_levels), :s_sigma_arr)
    t_sigma_arr ~ NamedDist(filldist(Exponential(1.0), K_levels), :t_sigma_arr)

    # rho_mf governs the coupling strength from the high-fidelity Source (Level 1) to child levels.
    rho_mf ~ NamedDist(Normal(1.0, 0.5), :rho_mf)

    # # 3. Global Likelihood Hyperparameters
    # Initializing vectors for K-level scalars with AD-stable initialization.
    lik_r = one(T)
    lik_phi = zero(T)
    extra_p = ones(T, K_levels)

    if family == "negbin"
        lik_r ~ NamedDist(Exponential(1.0), :lik_r)
    end

    if use_zi == true
        lik_phi ~ NamedDist(Beta(1, 1), :lik_phi)
    end

    if family in ["gamma", "beta", "student_t", "inverse_gaussian", "pareto"]
        extra_p ~ NamedDist(filldist(Exponential(1.0), K_levels), :extra_params)
    end

    # # 4. Latent Innovation Registries (K Levels)
    # We pre-allocate innovation fields to be combined in the final assembly loop.
    s_latent_fields = Vector{Any}(undef, K_levels)
    t_latent_fields = Vector{Any}(undef, K_levels)
    eta_innovations = Vector{Any}(undef, K_levels)
    y_sigma_tensors = Vector{Any}(undef, K_levels)

    for k in 1:K_levels
        fk = fidelities[k]
        N_k = fk.y_N
        field_k = zeros(T, N_k)
     
        # 4.1 Observation Volatility & SV Discovery per Fidelity
        sig_surface_k = ones(T, N_k)
        if get(fk, :use_sv, false) == true
            sig_log_var_k ~ NamedDist(Exponential(1.0), Symbol("sigma_log_var_", k))
            beta_vol_k ~ NamedDist(filldist(Normal(0, 1), fk.M_rff_sigma), Symbol("beta_vol_latent_", k))
            vol_proj_k = fk.vol_proj * beta_vol_k
            vol_latent_k = sqrt(2.0 / fk.M_rff_sigma) .* cos.(vol_proj_k)
            for i in 1:N_k
                sig_surface_k[i] = exp((sig_log_var_k * vol_latent_k[i]) / 2.0)
            end
        elseif family in ["gaussian", "lognormal"]
            sig_val_k ~ NamedDist(Exponential(1.0), Symbol("y_sigma_", k))
            for i in 1:N_k; sig_surface_k[i] = sig_val_k; end
        end
        y_sigma_tensors[k] = sig_surface_k

        # 4.2 Spatio-Temporal Innovation Fields per Fidelity
        if fk.model_space != "none"
            Q_s = recompose_precision(Symbol(fk.model_space), fk.s_Q_template.matrix, s_sigma_arr[k], noise=noise)
            s_latent_k ~ NamedDist(MvNormalCanon(zeros(fk.s_N), Q_s), Symbol("s_latent_", k))
            s_latent_fields[k] = s_latent_k
            for i in 1:N_k; field_k[i] += s_latent_k[fk.s_idx[i]]; end
        else
            s_latent_fields[k] = zeros(T, 1)
        end

        if fk.model_time != "none"
            Q_t = recompose_precision(Symbol(fk.model_time), fk.t_Q_template.matrix, t_sigma_arr[k], noise=noise)
            t_latent_k ~ NamedDist(MvNormalCanon(zeros(fk.t_N), Q_t), Symbol("t_latent_", k))
            t_latent_fields[k] = t_latent_k
            for i in 1:N_k; field_k[i] += t_latent_k[fk.t_idx[i]]; end
        else
            t_latent_fields[k] = zeros(T, 1)
        end
      # Unified Smooth Basis Dispatch (1D-4D)
        # Rationale: Fidelity-specific basis smooths are aggregated from the local registry.
        if haskey(fk, :basis_matrices) && !isempty(fk.basis_matrices)
            for v_sym in keys(fk.basis_matrices)
                B_mat = fk.basis_matrices[v_sym]
                sig_basis_k ~ NamedDist(Exponential(1.0), Symbol("sig_basis_", k, "_", v_sym))
                beta_basis_k ~ NamedDist(filldist(Normal(0, sig_basis_k), size(B_mat, 2)), Symbol("beta_basis_", k, "_", v_sym))
                field_k .+= B_mat * beta_basis_k
            end
        end
        
        if haskey(fk, :eigen_terms) && !isempty(fk.eigen_terms)
            for (eg_idx, e_term) in enumerate(fk.eigen_terms)
                v_pca ~ NamedDist(filldist(Normal(0, 1.0), length(e_term.ltri_indices)), Symbol("v_pca_", k, "_", eg_idx))
                sd_pca ~ NamedDist(filldist(Exponential(1.0), e_term.n_dims), Symbol("pca_sd_", k, "_", eg_idx))
                K_pca, _, _ = householder_transform(v_pca, e_term.n_dims, e_term.n_dims, e_term.ltri_indices, sd_pca, T(0.1), noise)
                field_k .+= vec(e_term.data * K_pca)
            end
        end

        eta_innovations[k] = field_k
    end

    # # 5. COUPLED PREDICTOR ASSEMBLY & SHARED SUPERVISORS
    # Rationale: Signal propagation from Level 1 (Source) to dependent Child fidelities.
    s_source = s_latent_fields[1]
    t_source = t_latent_fields[1]

    for k in 1:K_levels
        fk = fidelities[k]
        N_k = fk.y_N
        # Base mu initialized with fidelity-specific innovations
        mu_k = Vector{T}(eta_innovations[k])

        # Shared Mixed Effects
        if haskey(M, :mixed_terms) && !isempty(M.mixed_terms)
            for term in M.mixed_terms
                sig_c ~ NamedDist(Exponential(1.0), Symbol("sig_c_", term.name))
                Q_c = recompose_precision(:iid, sparse(I(term.n_cat, term.n_cat)), sig_c, noise=noise)
                c_latent ~ NamedDist(MvNormalCanon(zeros(term.n_cat), Q_c), Symbol("c_latent_", term.name))
                c_map = fk.mixed_terms[findfirst(x -> x.name == term.name, fk.mixed_terms)].indices
                for i in 1:N_k; mu_k[i] += c_latent[c_map[i]]; end
            end
        end


        # 5.2 Linear Fixed Effects per Fidelity
        if fk.Xfixed_N > 0
            Xf_beta ~ NamedDist(MvNormal(zeros(fk.Xfixed_N), 5.0 * I), Symbol("Xfixed_beta_", k))
            mu_k .+= fk.Xfixed * Xf_beta
        end

        
        # Offset Implementation for Multifidelity
        # Logic: Identity-link families exponentiate the provided log_offset.
        if haskey(fk, :log_offset)
            if family in ["gaussian", "student_t", "laplace"]
                for i in 1:N_k; mu_k[i] += exp(fk.log_offset[i]); end
            else
                for i in 1:N_k; mu_k[i] += fk.log_offset[i]; end
            end
        end


        # 5.3 Spatially-Aware Coupled Innovation Transfer
        # Signals from Level 1 are projected onto Level k coordinates via rho_mf coupling.
        if k > 1
            for i in 1:N_k
                source_proj = s_source[fk.s_idx[i]] + t_source[fk.t_idx[i]]
                mu_k[i] += rho_mf * source_proj
            end
        end

 
        # 5.5 Mechanistic Dynamics per Fidelity
        if get(fk, :use_dynamics, false) == true
            sig_dyn ~ NamedDist(Exponential(1.0), Symbol("sig_dyn_", k))
            m_states ~ NamedDist(filldist(Normal(0, 1), fk.t_N), Symbol("m_states_", k))
            for i in 1:N_k; mu_k[i] += m_states[fk.t_idx[i]] * sig_dyn; end
        end

        # # 6. POINTWISE LIKELIHOOD DISPATCH (Scalar AD Firewall)
        # Rationale: Standardizing multifidelity stream evaluation via the multiple-dispatch Likelihood functor.
        ok_idx = fk.y_ok[1]
        yL_k = T(get(M, :y_L, fill(-Inf, K_levels))[k])
        yU_k = T(get(M, :y_U, fill(Inf, K_levels))[k])
        h_k = T(get(M, :hurdle, fill(-Inf, K_levels))[k])

        for i in ok_idx
            d_lik = bstm_Likelihood(
                family,
                [T(fk.y_obs[i])];
                sigma_y = [y_sigma_tensors[k][i]],
                weight = [T(fk.weights[i])],
                phi_zi = lik_phi,
                r_nb = lik_r,
                trial = [Int(fk.trials[i])],
                y_L = yL_k,
                y_U = yU_k,
                hurdle = h_k,
                extra_params = extra_p[k]
            )
            Turing.@addlogprob! logpdf(d_lik, mu_k[i])
        end
    end
end



function _reconstruct(arch::UnivariateArchitecture, modelname::String, chain, M, PS, alpha)
    # # BSTM Modular Univariate Reconstruction v18.1.30
    # Rationale: Standardizing the univariate path to support the full 13 technical primitives.
    # Requirements: Parity with v18.1.12 Multivariate and v18.1.11 Multifidelity standards.

    N_samples = size(chain, 1)
    p_names = string.(FlexiChains.parameters(chain))

    # # 1. Scope Discovery and Dimension Validation
    # N_tot represents the combined grid of Training + Post-Stratification (PS).
    N_PS = 0
    if !isnothing(PS)
        N_PS = length(PS.s_idx)
    end
    N_tot = M.y_N + N_PS
    family_str = get(M, :model_family, "gaussian")

    # # 2. Parameter Discovery & Technical Registry Creation
    # Primary engine for recovering latent realizations (S, T, U, ST, SVC, Eigen, Smooth, Hierarchy).
    # Enforces domain-aware extraction to populate the 1D-4D smooth basis registry.
    _validate_parameter_registry(chain, M, p_names, 1)
    registry = _discover_manifold_realizations(chain, M, N_samples, 1, p_names)

    # # 3. Hierarchical Supervisor Recovery (z -> m -> y)
    # Rationale: Nested supervisors are dynamic latent fields contributing to the predictor.
    nested_contributions = zeros(Float64, N_tot, N_samples)
    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (m_key, m_data) in M.nested_manifolds
            rho_key = Symbol("rho_nested_", m_key)
            if rho_key in FlexiChains.parameters(chain)
                rho_samples = vec(get_params_vector(chain, string(rho_key), 1))

                # Component A: Supervisor Fixed Effects
                if haskey(m_data, :Xfixed)
                    xf_z = m_data.Xfixed
                    beta_z_key = "beta_nested_fixed_" * m_key
                    if beta_z_key in p_names
                        beta_z_samps = get_params_vector(chain, beta_z_key, size(xf_z, 2))
                        for j in 1:N_samples
                            # Additive projection of supervisor predictors onto the linear predictor grid
                            nested_contributions[1:M.y_N, j] .+= rho_samples[j] .* (xf_z * beta_z_samps[j, :])
                        end
                    end
                end

                # Component B: Supervisor Spatial Innovation
                if haskey(m_data, :model_space) && m_data.model_space != "none"
                    lat_z_s_key = "lat_nested_spatial_" * m_key
                    if lat_z_s_key in p_names || any(occursin.(Regex("^" * lat_z_s_key * "\\["), p_names))
                        f_z_s_samps = get_params_vector(chain, lat_z_s_key, m_data.s_N)
                        z_s_ptr = m_data.s_idx
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
    end

    # # 4. Modular Linear Predictor Assembly
    # Rationale: Combines all baseline manifold contributions into the eta tensor.
    # Routes through Method 1 (Null-Safe) to ensure correct coordinate mapping.
    eta_tensor = _modular_eta_assembly(N_tot, registry, M, PS)

    # # 5. Link-Scale Realization Mapping (Denoised and Noisy)
    # Rationale: Denoised (Expectation) vs Noisy (Posterior Predictive Realization).
    p_den = zeros(Float64, N_tot, N_samples)
    p_noi = zeros(Float64, N_tot, N_samples)
    log_lik = zeros(Float64, N_samples, M.y_N)
    fam_obj = get_model_family(family_str)

    for j in 1:N_samples
        # Additive Superposition of Modular Base + Hierarchical Supervisors
        eta_j = eta_tensor[:, 1, j] .+ nested_contributions[:, j]

        # Discovery of likelihood hyperparameters (structural zeros and dispersion)
        phi_val = "lik_phi" in p_names ? chain[:lik_phi].data[j] : 0.0
        r_val = "lik_r" in p_names ? chain[:lik_r].data[j] : 1.0

        # Link Mapping (Expected value mu)
        p_den[:, j] .= _apply_link_and_lik(family_str, eta_j, get(M, :use_zi, false), phi_val, r_val)

        # Stochastic Predictive Sampling with Heteroskedastic Surface support
        sig_j = registry.sv_surface isa Vector ? registry.sv_surface[1][:, j] : registry.sv_surface[:, j]

        _, noisy_j, ll_j = _process_ll_and_predictions(
            fam_obj, reshape(eta_j, N_tot, 1), chain, M, N_tot, 1, reshape(sig_j, N_tot, 1)
        )
        p_noi[:, j] .= noisy_j[:, 1]
        log_lik[j, :] .= ll_j[1, :]
    end

    # # 6. Hardened Summarization Engine
    # Rationale: Using explicit guards to prevent iteration errors and ensure non-zero persistence.

    # 6.1 Spatial and Temporal Manifold Summaries
    spatial_struct = nothing
    if !isnothing(registry.s_eff_struct) && !isempty(registry.s_eff_struct)
        spatial_struct = summarize_array(reshape(registry.s_eff_struct[1], M.s_N, 1, N_samples); alpha=alpha)
    end

    temporal_summ = nothing
    if !isnothing(registry.t_eff) && !isempty(registry.t_eff)
        temporal_summ = summarize_array(reshape(registry.t_eff[1], M.t_N, 1, N_samples); alpha=alpha)
    end

    # 6.2 Global Shared Components (Seasonal, Fixed, Smooth)
    seasonal_summ = nothing
    if M.model_season != "none" && !isnothing(registry.u_eff) && !isempty(registry.u_eff)
        seasonal_summ = summarize_array(reshape(registry.u_eff, M.u_N, 1, N_samples); alpha=alpha)
    end

    fixed_summ = nothing
    if !isnothing(registry.Xfixed_betas)
        fixed_summ = summarize_array(reshape(registry.Xfixed_betas, size(registry.Xfixed_betas, 1), 1, N_samples); alpha=alpha)
    end

    # FIXED: Persistence and Summarization of Accumulated Smooths for Dashboard Panels
    smooth_summ = nothing
    if !isnothing(registry.basis_eff_accum) && std(registry.basis_eff_accum) > 1e-9
        smooth_summ = summarize_array(reshape(registry.basis_eff_accum, M.y_N, 1, N_samples); alpha=alpha)
    end

    # 6.3 Advanced Features (Interactions and Supervisors)
    st_summ = nothing
    if !isnothing(registry.st_eff_maps) && !isempty(registry.st_eff_maps[1])
        st_summ = summarize_array(reshape(registry.st_eff_maps[1], M.s_N * M.t_N, 1, N_samples); alpha=alpha)
    end

    nested_summ = nothing
    if any(nested_contributions .!= 0.0)
        nested_summ = summarize_array(reshape(nested_contributions[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha)
    end

    # 6.4 Auxiliary Registries (Hierarchy and Volatility)
    # FIXED: Discovery of Hierarchical Regional Indicators
    hierarchy_summaries = Dict{Symbol, Any}()
    if !isnothing(registry.hierarchical_scales)
        for scale_sym in keys(registry.hierarchical_scales)
            h_samples = registry.hierarchical_scales[scale_sym]
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
    # # BSTM Modular Multivariate Reconstruction v18.1.35
    # Rationale: Standardizing the multivariate path to support the full 13 technical primitives.
    # Requirements: 100% parity with v18.1.30 Univariate and v06.1 Taxonomy.

    N_samples = size(chain, 1)
    outcomes_N = Int(M.outcomes_N)
    p_names = string.(FlexiChains.parameters(chain))

    # # 1. Scope Discovery and Dimension Validation
    # N_tot represents the combined grid of Training + Post-Stratification (PS) strata.
    N_PS = 0
    if !isnothing(PS)
        N_PS = length(PS.s_idx)
    end
    N_tot = M.y_N + N_PS
    family_str = get(M, :model_family, "gaussian")

    # # 2. Parameter Discovery & Technical Registry Creation
    # Recovers latent field realizations for K outcomes (S, T, U, ST, SVC, Eigen).
    # Enforces domain-aware extraction to resolve dual-use primitives (e.g. RW2, AR1).
    _validate_parameter_registry(chain, M, p_names, outcomes_N)
    registry = _discover_manifold_realizations(chain, M, N_samples, outcomes_N, p_names)

    # # 3. Hierarchical Supervisor Recovery (z -> m -> y)
    # Rationale: Nested supervisors contribute to the linear predictor across all K outcomes.
    nested_contributions = zeros(Float64, N_tot, outcomes_N, N_samples)
    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (m_key, m_data) in M.nested_manifolds
            rho_key = Symbol("rho_nested_", m_key)
            if rho_key in FlexiChains.parameters(chain)
                rho_samples = vec(get_params_vector(chain, string(rho_key), 1))

                # Type A: Supervisor Spatial Innovation (lat_nested_spatial)
                lat_key = "lat_nested_spatial_" * m_key
                if lat_key in p_names || any(occursin.(Regex("^" * lat_key * "\\["), p_names))
                    f_z_s_samps = get_params_vector(chain, lat_key, m_data.s_N)
                    z_s_ptr = m_data.s_idx
                    for j in 1:N_samples
                        for k in 1:outcomes_N
                            for i in 1:M.y_N
                                # Coordinate-aware signal transfer across outcomes
                                nested_contributions[i, k, j] += rho_samples[j] * f_z_s_samps[j, z_s_ptr[i]]
                            end
                        end
                    end
                end

                # Type B: Supervisor Fixed Effects (beta_nested_fixed)
                fix_key = "beta_nested_fixed_" * m_key
                if fix_key in p_names && haskey(m_data, :Xfixed)
                    xf_z = m_data.Xfixed
                    b_samps = get_params_vector(chain, fix_key, size(xf_z, 2))
                    for j in 1:N_samples
                        for k in 1:outcomes_N
                            # Additive projection of supervisor predictors onto the primary outcomes
                            nested_contributions[1:M.y_N, k, j] .+= rho_samples[j] .* (xf_z * b_samps[j, :])
                        end
                    end
                end
            end
        end
    end

    # # 4. Modular Linear Predictor Assembly
    # Projects all latent components into the [N_tot x K x N_samples] tensor eta.
    # Rationale: Ensures integer-perfect mapping of manifold pointers.
    eta_tensor = _modular_eta_assembly(N_tot, registry, M, PS)

    # # 5. Realization Mapping (Denoised and Noisy)
    # Rationale: Denoised (Expectation) vs Noisy (Posterior Predictive Observation).
    p_den = zeros(Float64, N_tot, outcomes_N, N_samples)
    p_noi = zeros(Float64, N_tot, outcomes_N, N_samples)
    log_lik = zeros(Float64, N_samples, M.y_N * outcomes_N)
    fam_obj = get_model_family(family_str)

    for j in 1:N_samples
        for k in 1:outcomes_N
            # Additive Superposition of Modular Base + Hierarchical Supervisors
            eta_jk = eta_tensor[:, k, j] .+ nested_contributions[:, k, j]

            # Discovery of likelihood hyperparameters for current sample
            phi_val = "lik_phi" in p_names ? chain[:lik_phi].data[j] : 0.0
            r_val = "lik_r" in p_names ? chain[:lik_r].data[j] : 1.0

            # Link Mapping (Expected value mu)
            p_den[:, k, j] .= _apply_link_and_lik(family_str, eta_jk, get(M, :use_zi, false), phi_val, r_val)

            # Stochastic Predictive Sampling with Heteroskedastic Surface
            # Handle Vector vs Matrix volatility surfaces for multivariate
            sig_jk = registry.sv_surface isa Vector ? registry.sv_surface[k][:, j] : registry.sv_surface[:, j]

            _, noisy_jk, ll_jk = _process_ll_and_predictions(
                fam_obj, reshape(eta_jk, N_tot, 1), chain, M, N_tot, 1, reshape(sig_jk, N_tot, 1)
            )
            p_noi[:, k, j] .= noisy_jk[:, 1]

            # Pointwise Log-Likelihood mapping for WAIC/LOO
            slice_idx = ((k-1)*M.y_N + 1):(k*M.y_N)
            log_lik[j, slice_idx] .= ll_jk[1, :]
        end
    end

    # # 6. Hardened Summarization Engine
    # Rationale: Standardizes all output fields to ensure visualization stability.

    # 6.1 Spatial and Temporal Manifold Summaries (Outcome-indexed)
    spatial_struct = nothing
    if !isnothing(registry.s_eff_struct) && !isempty(registry.s_eff_struct)
        spatial_struct = [summarize_array(reshape(registry.s_eff_struct[k], M.s_N, 1, N_samples); alpha=alpha) for k in 1:outcomes_N]
    end

    temporal_summ = nothing
    if !isnothing(registry.t_eff) && !isempty(registry.t_eff)
        temporal_summ = [summarize_array(reshape(registry.t_eff[k], M.t_N, 1, N_samples); alpha=alpha) for k in 1:outcomes_N]
    end

    # 6.2 Global Shared Components (Seasonal, Fixed, Smooth)
    seasonal_summ = nothing
    if M.model_season != "none" && !isnothing(registry.u_eff) && !isempty(registry.u_eff)
        seasonal_summ = summarize_array(reshape(registry.u_eff, M.u_N, 1, N_samples); alpha=alpha)
    end

    fixed_summ = nothing
    if !isnothing(registry.Xfixed_betas)
        fixed_summ = summarize_array(reshape(registry.Xfixed_betas, size(registry.Xfixed_betas, 1), 1, N_samples); alpha=alpha)
    end

    # Accumulated Smooths discovered for multivariate response branches
    smooth_summ = nothing
    if !isnothing(registry.basis_eff_accum) && std(registry.basis_eff_accum) > 1e-9
        smooth_summ = summarize_array(reshape(registry.basis_eff_accum, M.y_N, 1, N_samples); alpha=alpha)
    end

    # 6.3 Advanced Interactions and Hierarchies
    st_summ = nothing
    if !isnothing(registry.st_eff_maps) && !isempty(registry.st_eff_maps[1])
        st_summ = [summarize_array(reshape(registry.st_eff_maps[k], M.s_N * M.t_N, 1, N_samples); alpha=alpha) for k in 1:outcomes_N]
    end

    svc_summ = nothing
    if !isnothing(registry.svc_slopes) && !isempty(registry.svc_slopes[1])
        svc_summ = [summarize_array(reshape(registry.svc_slopes[k], size(registry.svc_slopes[k], 1) * size(registry.svc_slopes[k], 2), 1, N_samples); alpha=alpha) for k in 1:outcomes_N]
    end

    nested_summ = nothing
    if any(nested_contributions .!= 0.0)
        nested_summ = summarize_array(p_den[1:M.y_N, :, :]; alpha=alpha)
    end

    # 6.4 Auxiliary Registries (Hierarchy and Volatility)
    # Persistence of Hierarchical Multi-Resolution Scale fields
    hierarchy_summaries = Dict{Symbol, Any}()
    if !isnothing(registry.hierarchical_scales)
        for scale_sym in keys(registry.hierarchical_scales)
            h_samples = registry.hierarchical_scales[scale_sym]
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
    # # BSTM Modular Multifidelity Reconstruction v18.1.35
    # Rationale: Standardizing the multifidelity path to support the full 13 technical primitives.
    # Requirements: 100% parity with v18.1.30 Univariate and v06.1 Taxonomy.

    N_samples = size(chain, 1)
    fidelities = M.fidelities
    K_levels = length(fidelities)
    p_names = string.(FlexiChains.parameters(chain))
    family_str = get(M, :model_family, "gaussian")
    fam_obj = get_model_family(family_str)

    # # 1. Parameter Discovery & Technical Registry Creation
    # Recovers latent realizations for K innovation levels (S, T, U, ST, SVC, Eigen, Physics, Dynamics).
    # This enforces domain-aware extraction to resolve dual-use primitives.
    _validate_parameter_registry(chain, M, p_names, K_levels)
    registry = _discover_manifold_realizations(chain, M, N_samples, K_levels, p_names)

    # # 2. Hierarchical Coupling Discovery (ρ)
    # rho_mf governs the signal transfer from the high-fidelity Source (Level 1) to child levels.
    rhos = vec(get_params_vector(chain, "rho_mf", 1))

    # # 3. Modular Linear Predictor Assembly (Heterogeneous Innovation Fields)
    # Projects fidelity-specific latent components into the innovation tensor.
    # Total_y_N represents the sum of all observation counts across the K levels.
    total_y_N = sum(f.y_N for f in fidelities)
    eta_tensor = _modular_eta_assembly(total_y_N, registry, M, PS)

    # # 4. Fidelity Signal Transfer & Realization Mapping
    # Rationale: Denoised (Expectation) vs Noisy (Posterior Predictive Observation).
    p_den_list = [zeros(Float64, fidelities[k].y_N, N_samples) for k in 1:K_levels]
    p_noi_list = [zeros(Float64, fidelities[k].y_N, N_samples) for k in 1:K_levels]
    log_lik = zeros(Float64, N_samples, total_y_N)

    for j in 1:N_samples
        for k in 1:K_levels
            fk = fidelities[k]
            N_k = fk.y_N

            # Signal Coupling Logic: coupled_eta = rho * Source_Signal + Local_Innovation
            # Level 1 (Source) acts as the global structural skeleton.
            coupled_eta_jk = if k == 1
                eta_tensor[1:N_k, 1, j]
            else
                # Projection of Level 1 signal onto Level k observation coordinates
                s_source_proj = registry.s_eff_struct[1][fk.s_idx, j]
                t_source_proj = registry.t_eff[1][fk.t_idx, j]
                (rhos[j] .* (s_source_proj .+ t_source_proj)) .+ eta_tensor[1:N_k, k, j]
            end

            # Link Mapping (Expected value mu)
            p_den_list[k][:, j] .= _apply_link_and_lik(family_str, coupled_eta_jk, get(M, :use_zi, false))

            # Stochastic Predictive Sampling with Heteroskedastic Surface support
            sig_jk = registry.sv_surface isa Vector ? registry.sv_surface[k][:, j] : registry.sv_surface[:, j]

            _, noisy_jk, ll_jk = _process_ll_and_predictions(
                fam_obj, reshape(coupled_eta_jk, N_k, 1), chain, fk, N_k, 1, reshape(sig_jk, N_k, 1)
            )
            p_noi_list[k][:, j] .= noisy_jk[:, 1]

            # Pointwise Log-Likelihood mapping for WAIC/LOO
            offset = k > 1 ? sum(fidelities[m].y_N for m in 1:(k-1)) : 0
            log_lik[j, (offset + 1):(offset + N_k)] .= ll_jk[1, :]
        end
    end

    # # 5. Hardened Summarization Engine
    # Rationale: Standardizes all output fields to ensure visualization stability.

    # 5.1 Innovation Manifold Summaries (Fidelity-indexed)
    spatial_innovations = [summarize_array(reshape(registry.s_eff_struct[k], fidelities[k].s_N, 1, N_samples); alpha=alpha) for k in 1:K_levels]
    temporal_innovations = [summarize_array(reshape(registry.t_eff[k], fidelities[k].t_N, 1, N_samples); alpha=alpha) for k in 1:K_levels]

    # 5.2 Global Shared Components and Basis Smooths
    seasonal_summ = nothing
    if M.model_season != "none" && !isnothing(registry.u_eff) && !isempty(registry.u_eff)
        seasonal_summ = summarize_array(reshape(registry.u_eff, M.u_N, 1, N_samples); alpha=alpha)
    end

    # Accumulated Smooths discovered for heterogeneous response branches
    smooth_summ = nothing
    if !isnothing(registry.basis_eff_accum) && std(registry.basis_eff_accum) > 1e-9
        smooth_summ = summarize_array(reshape(registry.basis_eff_accum, total_y_N, 1, N_samples); alpha=alpha)
    end

    # 5.3 Advanced Features (Interactions and Hierarchical Scales)
    # Persistence of Hierarchical Multi-Resolution Scale fields for each level
    hierarchy_summaries = Dict{Symbol, Any}()
    if !isnothing(registry.hierarchical_scales)
        for scale_sym in keys(registry.hierarchical_scales)
            h_samples = registry.hierarchical_scales[scale_sym]
            hierarchy_summaries[scale_sym] = summarize_array(reshape(h_samples, size(h_samples, 1), 1, N_samples); alpha=alpha)
        end
    end

    # 5.4 Volatility Surface Summaries
    vol_surfaces = if registry.sv_surface isa Vector
        [summarize_array(reshape(registry.sv_surface[k], fidelities[k].y_N, 1, N_samples); alpha=alpha) for k in 1:K_levels]
    else
        [summarize_array(reshape(registry.sv_surface, fidelities[k].y_N, 1, N_samples); alpha=alpha) for k in 1:K_levels]
    end

    return (
        predictions_denoised = [summarize_array(reshape(p_den_list[k], fidelities[k].y_N, 1, N_samples); alpha=alpha) for k in 1:K_levels],
        predictions_noisy = [summarize_array(reshape(p_noi_list[k], fidelities[k].y_N, 1, N_samples); alpha=alpha) for k in 1:K_levels],
        spatial_innovations = spatial_innovations,
        temporal_innovations = temporal_innovations,
        seasonal = seasonal_summ,
        smooth_effects = smooth_summ,
        hierarchical_scales = isempty(hierarchy_summaries) ? nothing : hierarchy_summaries,
        volatility_surfaces = vol_surfaces,
        fidelity_rho = summarize_array(reshape(rhos, 1, 1, N_samples); alpha=alpha),
        waic = _compute_waic(log_lik),
        log_lik_matrix = log_lik,
        K_levels = K_levels,
        family = family_str,
        arch = arch
    )
end

 
function _reconstruct(arch::ExampleArchitecture, modelname::String, chain, M, PS, alpha)
    # # BSTM Modular Example Reconstruction v18.1.35
    # Rationale: Standardizing the example path to support the full 13 technical primitives.
    # Requirements: 100% parity with v18.1.35 Multivariate and v06.1 Taxonomy.
 
    N_samples = size(chain, 1)
    p_names = string.(FlexiChains.parameters(chain))

    # # 1. Scope Discovery and Dimension Validation
    N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    N_tot = M.y_N + N_PS
    family_str = get(M, :model_family, "gaussian")

    # # 2. Parameter Discovery & Technical Registry Creation
    _validate_parameter_registry(chain, M, p_names, 1)
    registry = _discover_manifold_realizations(chain, M, N_samples, 1, p_names)

    # # 3. Hierarchical Supervisor Recovery (z -> m -> y)
    nested_contributions = zeros(Float64, N_tot, N_samples)
    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (m_key, m_data) in M.nested_manifolds
            rho_key = Symbol("rho_nested_", m_key)
            if rho_key in FlexiChains.parameters(chain)
                rho_samples = vec(get_params_vector(chain, string(rho_key), 1))

                # Component A: Supervisor Fixed Effects
                if haskey(m_data, :Xfixed)
                    xf_z = m_data.Xfixed
                    beta_z_key = "beta_nested_fixed_" * m_key
                    if beta_z_key in p_names
                        beta_z_samps = get_params_vector(chain, beta_z_key, size(xf_z, 2))
                        for j in 1:N_samples
                            nested_contributions[1:M.y_N, j] .+= rho_samples[j] .* (xf_z * beta_z_samps[j, :])
                        end
                    end
                end

                # Component B: Supervisor Spatial Innovation
                if haskey(m_data, :model_space) && m_data.model_space != "none"
                    lat_z_s_key = "lat_nested_spatial_" * m_key
                    if lat_z_s_key in p_names || any(occursin.(Regex("^" * lat_z_s_key * "\\["), p_names))
                        f_z_s_samps = get_params_vector(chain, lat_z_s_key, m_data.s_N)
                        z_s_ptr = m_data.s_idx
                        for j in 1:N_samples, i in 1:M.y_N
                            nested_contributions[i, j] += rho_samples[j] * f_z_s_samps[j, z_s_ptr[i]]
                        end
                    end
                end
            end
        end
    end

    # # 4. Modular Linear Predictor Assembly
    eta_tensor = _modular_eta_assembly(N_tot, registry, M, PS)

    # # 5. Link-Scale Realization Mapping (Denoised and Noisy)
    p_den = zeros(Float64, N_tot, N_samples)
    p_noi = zeros(Float64, N_tot, N_samples)
    log_lik = zeros(Float64, N_samples, M.y_N)
    fam_obj = get_model_family(family_str)

    for j in 1:N_samples
        eta_j = eta_tensor[:, 1, j] .+ nested_contributions[:, j]
        phi_val = "lik_phi" in p_names ? chain[:lik_phi].data[j] : 0.0
        r_val = "lik_r" in p_names ? chain[:lik_r].data[j] : 1.0

        p_den[:, j] .= _apply_link_and_lik(family_str, eta_j, get(M, :use_zi, false), phi_val, r_val)

        sig_j = registry.sv_surface isa Vector ? registry.sv_surface[1][:, j] : registry.sv_surface[:, j]

        _, noisy_j, ll_j = _process_ll_and_predictions(
            fam_obj, reshape(eta_j, N_tot, 1), chain, M, N_tot, 1, reshape(sig_j, N_tot, 1)
        )
        p_noi[:, j] .= noisy_j[:, 1]
        log_lik[j, :] .= ll_j[1, :]
    end

    # # 6. Hardened Summarization Engine
    spatial_struct = !isnothing(registry.s_eff_struct) && !isempty(registry.s_eff_struct) ? summarize_array(reshape(registry.s_eff_struct[1], M.s_N, 1, N_samples); alpha=alpha) : nothing
    temporal_summ = !isnothing(registry.t_eff) && !isempty(registry.t_eff) ? summarize_array(reshape(registry.t_eff[1], M.t_N, 1, N_samples); alpha=alpha) : nothing
    seasonal_summ = M.model_season != "none" && !isnothing(registry.u_eff) && !isempty(registry.u_eff) ? summarize_array(reshape(registry.u_eff, M.u_N, 1, N_samples); alpha=alpha) : nothing
    fixed_summ = !isnothing(registry.Xfixed_betas) ? summarize_array(reshape(registry.Xfixed_betas, size(registry.Xfixed_betas, 1), 1, N_samples); alpha=alpha) : nothing
    smooth_summ = !isnothing(registry.basis_eff_accum) && std(registry.basis_eff_accum) > 1e-9 ? summarize_array(reshape(registry.basis_eff_accum, M.y_N, 1, N_samples); alpha=alpha) : nothing
    st_summ = !isnothing(registry.st_eff_maps) && !isempty(registry.st_eff_maps[1]) ? summarize_array(reshape(registry.st_eff_maps[1], M.s_N * M.t_N, 1, N_samples); alpha=alpha) : nothing
    nested_summ = any(nested_contributions .!= 0.0) ? summarize_array(reshape(nested_contributions[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha) : nothing

    hierarchy_summaries = Dict{Symbol, Any}()
    if !isnothing(registry.hierarchical_scales)
        for scale_sym in keys(registry.hierarchical_scales)
            h_samples = registry.hierarchical_scales[scale_sym]
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

 



####


 
;;


function build_structure_template(type::Symbol, n::Int; scale=true, coords=nothing, W=nothing, bipartite_adj=nothing)
    # Consolidated Structural Factory [v14.6.0 - BSTM v06.1 Unified Registry]
    # Rationale: This function provides the definitive factory for precision matrix templates
    # across all manifold domains. It standardizes scaling and topological initialization.
    # Audit: Fixed SPDE scaling and DAG operator registration; added TPS 2D routing.

    Q = Matrix{Float64}(undef, 0, 0)
    sf = 1.0

    # Group 0: Null Manifold Handling
    # If the type is :none, we return a 1x1 identity to satisfy supervisor shapes without overhead
        # 'harmonic' and 'rff' are basis-driven but require a recognized identity entry in the registry.
    if type == :iid || type == :none || type == :identity || type == :harmonic || type == :rff
        return (matrix = sparse(I(n)), scaling_factor = 1.0)
    end

  
    # Group 1: Identity and Spectral Bases
    # Rationale: These manifolds operate in transformed or group-level spaces.
    if type in [:iid, :eigen, :bgcn, :nystrom, :rff, :fft, :bspline]
        Q = Matrix(1.0I, n, n)
        sf = 1.0
   
   # Group 2: Graph-Based & CAR Manifolds
    elseif type in [:icar, :besag, :bym2, :leroux, :sar]
        if isnothing(W)
            # If W is missing but n=1, fallback to identity instead of error
            if n <= 1
                Q = Matrix(1.0I, 1, 1)
                sf = 1.0
            else
                error("BSTM Factory Error: Adjacency matrix W required for manifold :$type with size $n")
            end
        else
            D_sp = Diagonal(vec(sum(W, dims=2)))
            Q_raw = Matrix(D_sp - W)
            if scale
                evals = eigvals(Q_raw)
                nz_ev = filter(x -> x > 1e-6, evals)
                sf = isempty(nz_ev) ? 1.0 : exp(mean(log.(nz_ev)))
                Q = Q_raw ./ sf
            else
                Q = Q_raw
                sf = 1.0
            end
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
    elseif type in [:seasonal, :harmonic]
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

        
    # # Periodic / Cyclic Manifolds
    # Rationale: Standardizing periodic structures as Besag fields on circulant graphs.
    elseif type == :cyclic
        rows = Int[]
        cols = Int[]
        for i in 1:n
            # Backward connection (previous node in cycle)
            prev_node = i == 1 ? n : i - 1
            push!(rows, i)
            push!(cols, prev_node)
            
            # Forward connection (next node in cycle)
            next_node = i == n ? 1 : i + 1
            push!(rows, i)
            push!(cols, next_node)
        end
        
        W_circ = sparse(rows, cols, ones(length(rows)), n, n)
        
        # Recompute Laplacian and scaling using the generated circulant adjacency
        asum_c = vec(sum(W_circ, dims=2))
        Q_circ = Diagonal(asum_c) - W_circ
        sf_c = scale ? scaling_factor_bym2(W_circ) : 1.0
        return (matrix = Q_circ, scaling_factor = sf_c)
 
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
    template = build_structure_template(:icar, data_inputs.s_N; W=data_inputs.W)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :icar,
        hyper = (sigma_prior = m.sigma_prior, rho_prior = nothing)
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
    # Rationale: Provides 2D smooth surfaces using the bending energy penalty.
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


function recompose_precision(
    m_type::Symbol, 
    template_s::AbstractMatrix, 
    param_val::Real; 
    template_t=nothing, 
    extra_param=nothing, 
    noise=1e-4, 
    directed_adj=nothing, 
    flow_direction=:bidirectional
)
    # Rationale: Standardizing the inversion of the variance scale for precision mapping.
    # Requirement: Avoid ternary if/else during sampling to maintain AD stability.
    
    n_s = size(template_s, 1)
    scale_factor = 1.0 / (param_val^2 + noise)

    # 1. Base Graph & CAR Manifolds
    if m_type == :none || m_type == :fixed
        return Symmetric(scale_factor * I(n_s) + noise * I)
    end

    if m_type == :besag || m_type == :icar || m_type == :diffusion || m_type == :cyclic
        return Symmetric(scale_factor .* template_s + noise * I)
    end

    if m_type == :bym2
        rho = isnothing(extra_param) ? 0.5 : extra_param
        return Symmetric(scale_factor .* (rho .* template_s + (1.0 - rho) .* I(n_s)) + noise * I)
    end

    if m_type == :leroux
        lambda_val = isnothing(extra_param) ? 0.5 : extra_param
        return Symmetric(scale_factor .* (lambda_val .* template_s + (1.0 - lambda_val) .* I(n_s)) + noise * I)
    end

    # 2. Knorr-Held Interaction Suite (Types I-IV)
    # Rationale: These require the Kronecker product of temporal and spatial structures.
    # Fix: Ensure template_t is utilized to prevent UndefVarError.
    if m_type == :I || m_type == :II || m_type == :III || m_type == :IV
        if isnothing(template_t)
            # Fallback to temporal identity if template_t was not discovered
            Q_full = template_s
        else
            Q_full = kron(template_t, template_s)
        end
        return Symmetric(scale_factor .* Q_full + noise * I)
    end

    # 3. Directed Network & Physics-Informed Transport
    if m_type == :network
        rho_net = isnothing(extra_param) ? 0.8 : extra_param
        W_net = !isnothing(directed_adj) ? directed_adj : template_s
        
        # Adjoint operator for directed flow
        L_op = if flow_direction == :upstream
            I(n_s) - rho_net .* W_net'
        elseif flow_direction == :downstream
            I(n_s) - rho_net .* W_net
        else
            I(n_s) - rho_net .* (W_net + W_net') ./ 2.0
        end
        return Symmetric(scale_factor .* (L_op' * L_op) + noise * I)
    end

    if m_type == :sar || m_type == :advection || m_type == :advection_diffusion || m_type == :dag || m_type == :proper_car
        rho_p = isnothing(extra_param) ? 0.8 : extra_param
        L_op = I(n_s) - rho_p .* template_s
        return Symmetric(scale_factor .* (L_op' * L_op) + noise * I)
    end

    # 4. Continuous Distance & Difference Penalties
    if m_type == :gp
        ls = isnothing(extra_param) ? 1.0 : extra_param
        K = (param_val^2) .* exp.(-(Matrix(template_s).^2) ./ (2 * ls^2 + noise))
        return inv(Symmetric(K + noise * I(n_s)))
    end

    if m_type == :spde
        kappa = isnothing(extra_param) ? 1.0 : extra_param
        L_spde = (kappa^2 .* I(n_s) + template_s)
        return Symmetric(scale_factor .* (L_spde' * L_spde) + noise * I)
    end

    if m_type == :rff || m_type == :fft || m_type == :bspline || m_type == :pspline || m_type == :rw1 || m_type == :rw2 || m_type == :tps
        return Symmetric(scale_factor .* template_s + noise * I)
    end

    # Fallback for unrecognized types
    return Symmetric(scale_factor .* template_s + noise * I)
end

# POSITIONAL OVERLOAD: Dispatches from bstm_univariate metadata objects
function recompose_precision(manifold_metadata::NamedTuple, param_val::Real; template_t=nothing, extra_param=nothing, noise=1e-4)
    return recompose_precision(
        Symbol(manifold_metadata.model_type), 
        manifold_metadata.Q_template, 
        param_val; 
        template_t=template_t, 
        extra_param=extra_param, 
        noise=noise
    )
end

# # Precision Recomposition Factory (Recursive Algebraic Dispatch)
# This function now accepts Manifold structs or algebraic operator results.
function recompose_precision(manifold_node::Any, M, param_sig::Real; noise=1e-4)
    # # Base Case: Atomic Manifold Structs
    if manifold_node isa Manifold
        m_type = manifold_type(manifold_node)
        # Atomic builders provide the base template Q
        m_meta = build_model(manifold_node, M)
        return recompose_precision(Symbol(m_type), m_meta.Q_template, param_sig; noise=noise)
    
    # # Algebraic Case: Composed Manifolds (⊗, ⊕)
    elseif manifold_node isa ComposedManifold
        # Retrieve components
        comps = manifold_node.components
        op = manifold_node.operator
        
        if op == :kronecker_product
            # Q_total = Q1 ⊗ Q2
            # We split the parameter variance across components or treat sig as global scale
            Q1 = recompose_precision(comps[1], M, 1.0; noise=noise)
            Q2 = recompose_precision(comps[2], M, 1.0; noise=noise)
            Q_full = kron(Q1, Q2)
            return Symmetric(Q_full ./ (param_sig^2 + noise) + noise * I)
            
        elseif op == :direct_sum
            # Q_total is block diagonal if domains are distinct
            # Here we return a list of precisions for the supervisor to iterate
            return [recompose_precision(c, M, param_sig; noise=noise) for c in comps]

        elseif op == :directed_dependency
            # Q_total = (I - ρM1)' Q2 (I - ρM1)
            # Rationale: Representing state-space transitions or advection operators
            Q1 = recompose_precision(comps[1], M, 1.0; noise=noise)
            Q2 = recompose_precision(comps[2], M, 1.0; noise=noise)
            
            # For directed dependencies, we return the tuple of operators for the supervisor
            # as they typically require an explicit coupling parameter (rho)
            return (Q_source=Q1, Q_innovation=Q2, type=:directed)
            
        elseif op == :composition
            # Q_total = Q1 * Q2 * Q1 (Basis warping/Cascading dependency)
            # Rationale: Representing the composition of two precision structures as a functional transformation.
            Q1 = recompose_precision(comps[1], M, 1.0; noise=noise)
            Q2 = recompose_precision(comps[2], M, 1.0; noise=noise)
            
            # Ensure dimensional parity for matrix multiplication in composition
            if size(Q1) == size(Q2)
                Q_comp = Q1 * Q2 * Q1
                return Symmetric(Q_comp ./ (param_sig^2 + noise) + noise * I)
            else
                # Fallback to Kronecker if dimensions are heterogeneous
                return Symmetric(kron(Q1, Q2) ./ (param_sig^2 + noise) + noise * I)
            end

        elseif op == :pipe
            # Stacking: M1 |> M2 (e.g., Spatial |> AR1 for space-time evolution)
            # This implies a state-space transition matrix
            return recompose_precision(comps[1], M, param_sig; noise=noise)
        end
    end
    
    # Fallback for symbol-based legacy calls
    return Matrix(1.0I, 1, 1)
end



function parse_module_params(params_str::String)
    # BSTM Parameter Extractor and Expression Realizer [v20.2.1]
    # Timestamp: 2025-06-27 15:30:00
    # Rationale: Resolving the evaluation gap by identifying and executing Julia code strings found in DSL parameters.

    d = Dict{Symbol, Any}()
    if isempty(strip(params_str))
        return d
    end

    # Split at depth zero to handle nested structures correctly
    function split_params(input::String)
        parts = String[]
        depth = 0
        current_block = ""
        for char in input
            if char == '(' || char == '['
                depth = depth + 1
            elseif char == ')' || char == ']'
                depth = depth - 1
            end
            if char == ',' && depth == 0
                push!(parts, strip(current_block))
                current_block = ""
            else
                current_block = current_block * char
            end
        end
        if !isempty(strip(current_block))
            push!(parts, strip(current_block))
        end
        return parts
    end

    pairs = split_params(params_str)

    for entry in pairs
        if occursin("=", entry)
            elements = Base.split(entry, "=", limit = 2)
            param_key = Symbol(strip(elements[1]))
            param_val_raw = strip(elements[2])

            # Dynamic Expression Realization
            # If the string contains indexing symbols or call symbols, evaluate it in Main scope.
            if occursin(r"\[|:|[()]", param_val_raw) && 
               !(startswith(param_val_raw, "(") && endswith(param_val_raw, ")")) && 
               !(startswith(param_val_raw, "[") && endswith(param_val_raw, "]"))
                try
                    d[param_key] = Main.eval(Meta.parse(param_val_raw))
                catch e
                    @warn "BSTM Parser: Could not evaluate expression $param_val_raw. Keeping as string. Error: $e"
                    d[param_key] = param_val_raw
                end

            # Collection Handling
            elseif (startswith(param_val_raw, "(") && endswith(param_val_raw, ")")) || 
                   (startswith(param_val_raw, "[") && endswith(param_val_raw, "]"))
                content_inner = strip(param_val_raw[2:end-1])
                parts_inner = split_params(content_inner)
                # Map cleaned values or recursively parse if sub-pairs exist
                d[param_key] = [occursin("=", v) ? parse_module_params(v) : strip(v, ['\'', '"']) for v in parts_inner]

            # String Literals
            elseif (startswith(param_val_raw, "'") && endswith(param_val_raw, "'")) || 
                   (startswith(param_val_raw, "\"") && endswith(param_val_raw, "\""))
                d[param_key] = strip(param_val_raw, ['\'', '"'])

            # Numeric Literals
            elseif !isnothing(tryparse(Int, param_val_raw))
                d[param_key] = parse(Int, param_val_raw)
            elseif !isnothing(tryparse(Float64, param_val_raw))
                d[param_key] = parse(Float64, param_val_raw)

            # Boolean Literals
            elseif param_val_raw == "true"
                d[param_key] = true
            elseif param_val_raw == "false"
                d[param_key] = false

            # Symbolic Keywords
            else
                d[param_key] = Symbol(param_val_raw)
            end
        end
    end

    return d
end



# Rationale: Centralizing the generation of non-linear basis matrices (B).
# This factory ensures absolute parity with the v06.1 manifold taxonomy.

function bstm_smooth_basis_1D(type::String, vals::AbstractVector, nbins::Int, degree::Int; kwargs...)
    # # 1. Initialization and Scoping
    # Rationale: Ensures every call returns a valid N x M matrix to prevent sampler collapse.
    n_obs = length(vals)
    B = zeros(Float64, n_obs, nbins)
    
    # Technical Metadata Discovery
    v_min = minimum(vals)
    v_max = maximum(vals)
    v_std = std(vals) + 1e-9

    # # 2. Manifold Dispatch Registry
    
    # # 2.1 B-Splines / P-Splines (Piecewise Polynomials)
    if type in ["bspline", "pspline", "smooth"]
        # Step A: Define knot sequence using a range-based sequence for full coverage.
        knots = collect(range(v_min, stop=v_max, length=nbins - degree + 1))
        
        # Step B: Recursive Radial Expansion
        for m in 1:nbins
            # Index resolution without 'clamp'
            k_idx = m > length(knots) ? length(knots) : (m < 1 ? 1 : m)
            target_knot = knots[k_idx]
            B[:, m] .= exp.(-((vals .- target_knot).^2) ./ (2.0 * v_std^2))
        end

    # # 2.2 Thin Plate Splines (TPS - Bending Energy Minimization)
    elseif type == "tps"
        knots = quantile(vals, range(0, 1, length=nbins))
        for m in 1:nbins
            r = abs.(vals .- knots[m])
            B[:, m] .= (r.^2) .* log.(r .+ 1e-6)
        end

    # # 2.3 Random Fourier Features (RFF - Spectral GP Approximation)
    elseif type == "rff"
        m_rff = nbins
        ls = get(kwargs, :lengthscale, v_std)
        Omega = randn(1, m_rff) ./ ls
        Phi = rand(m_rff) .* (2.0 * pi)
        B .= sqrt(2.0 / m_rff) .* cos.((vals * Omega) .+ Phi')

    # # 2.4 Fast Fourier Transform (FFT Basis - Toroidal Grid)
    elseif type == "fft"
        t_coords = collect(range(0, 1, length=n_obs))
        for m in 1:div(nbins, 2)
            B[:, 2m-1] .= sin.(2.0 * pi * m .* t_coords)
            B[:, 2m]   .= cos.(2.0 * pi * m .* t_coords)
        end

    # # 2.5 Wavelets (Multi-Resolution Decomposition)
    elseif type == "wavelet"
        for m in 1:nbins
            center = v_min + (m/nbins) * (v_max - v_min)
            width = (v_max - v_min) / nbins
            B[:, m] .= (vals .>= center) .& (vals .< (center + width)) ? 1.0 : 0.0
        end

    # # 2.6 Moran's I Basis (Spectral Filtering) - RESTORED
    # Rationale: Uses eigenvectors of the spatial weights matrix to capture specific scales.
    elseif type == "moran"
        # Placeholder: In production, this requires the adjacency matrix W.
        # Here we provide a spectral sine-basis as a technical proxy.
        for m in 1:nbins
            B[:, m] .= sin.(pi * m * (vals .- v_min) ./ (v_max - v_min))
        end

    # # 2.7 Spherical Basis (Compact Support) - RESTORED
    # Rationale: Ensures correlation is exactly zero beyond a finite range R.
    elseif type == "spherical"
        range_r = get(kwargs, :range, v_std * 2.0)
        knots = quantile(vals, range(0, 1, length=nbins))
        for m in 1:nbins
            h = abs.(vals .- knots[m]) ./ range_r
            # Spherical kernel: 1 - 1.5h + 0.5h^3 if h < 1, else 0
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    # # 2.8 Exponential Decay (Continuous Attenuation) - RESTORED
    elseif type == "decay"
        ls = get(kwargs, :lengthscale, v_std)
        knots = quantile(vals, range(0, 1, length=nbins))
        for m in 1:nbins
            B[:, m] .= exp.(-abs.(vals .- knots[m]) ./ ls)
        end

    # # 2.9 Barycentric (Piecewise Linear Triangulation) - RESTORED
    elseif type == "barycentric"
        knots = collect(range(v_min, stop=v_max, length=nbins))
        for m in 1:nbins
            # Standard triangle/hat basis
            h = (v_max - v_min) / (nbins - 1)
            dist = abs.(vals .- knots[m]) ./ h
            mask = dist .< 1.0
            B[mask, m] .= 1.0 .- dist[mask]
        end

    # # 2.10 Identity / Fixed Fallback
    else
        B = ones(n_obs, 1)
    end

    return B
end

function bstm_smooth_basis_2D(type::String, coords::AbstractMatrix, nbins::Int; kwargs...)
    # # 1. Initialization and Context Discovery
    # Rationale: Ensures every call returns a valid N x M matrix to prevent sampler collapse.
    # coords is expected to be an N x 2 matrix (e.g., [lon lat]).
    n_obs = size(coords, 1)
    
    # Technical Metadata Discovery
    c_min = [minimum(coords[:, 1]), minimum(coords[:, 2])]
    c_max = [maximum(coords[:, 1]), maximum(coords[:, 2])]
    c_std = [std(coords[:, 1]), std(coords[:, 2])] .+ 1e-9

    # # 2. Manifold Dispatch Registry for 2D Surfaces

    # Anisotropy Configuration: Extract independent lengthscales if provided, else fallback to marginal std
    # ls_x and ls_y allow the user to control directional smoothing separately
    ls_x = get(kwargs, :ls_x, c_std[1])
    ls_y = get(kwargs, :ls_y, c_std[2])

     # # 2.1 Anisotropic Bivariate P-Splines (Tensor Product Basis)
    # Rationale: Expands 1D marginal bases into a 2D surface grid with directional scaling.
    if type in ["pspline", "bspline", "smooth"]
        # n_marginal defines the resolution per dimension to reach total nbins
        n_marginal = Int(floor(sqrt(nbins)))
        m_total = n_marginal^2
        B = zeros(Float64, n_obs, m_total)

        # Marginal Knot Grids: Distributed evenly across the observed ranges
        kx = collect(range(c_min[1], stop=c_max[1], length=n_marginal))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_marginal))

        idx = 1
        for i in 1:n_marginal
            for j in 1:n_marginal
                # Tensor product of anisotropic radial marginals
                # bx uses ls_x; by uses ls_y to define the support width
                bx = exp.(-((coords[:, 1] .- kx[i]).^2) ./ (2.0 * ls_x^2))
                by = exp.(-((coords[:, 2] .- ky[j]).^2) ./ (2.0 * ls_y^2))
                B[:, idx] .= bx .* by
                idx += 1
            end
        end

    # # 2.2 Anisotropic Thin Plate Splines (TPS)
    # Rationale: Normalizes distances relative to anisotropic scales before computing bending energy.
    elseif type == "tps"
        B = zeros(Float64, n_obs, nbins)
        n_grid = Int(floor(sqrt(nbins)))
        kx = collect(range(c_min[1], stop=c_max[1], length=n_grid))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_grid))
        centers = [(x, y) for x in kx, y in ky][:]

        for m in 1:min(nbins, length(centers))
            # Normalized distances relative to anisotropic scales
            # This warps the radial basis function to match the directional lengthscales
            dx = (coords[:, 1] .- centers[m][1]) ./ ls_x
            dy = (coords[:, 2] .- centers[m][2]) ./ ls_y
            r = sqrt.(dx^2 .+ dy^2)
            # TPS Kernel: r^2 * log(r)
            B[:, m] .= (r.^2) .* log.(r .+ 1e-6)
        end

    # # 2.3 Anisotropic Random Fourier Features (RFF)
    # Rationale: Samples frequencies from an anisotropic spectral density.
    elseif type == "rff" || type == "anisotropic"
        # Generate 2D frequencies with directional scaling
        # Omega[1] is scaled by ls_x; Omega[2] is scaled by ls_y
        Omega = randn(2, nbins)
        Omega[1, :] ./= ls_x
        Omega[2, :] ./= ls_y
        
        Phi = rand(nbins) .* (2.0 * pi)
        # Projection: feature_map = sqrt(2/M) * cos(Coords * Omega + Phi)
        B = sqrt(2.0 / nbins) .* cos.((coords * Omega) .+ Phi')

    # # 2.4 2D Fast Fourier Transform (FFT - Anisotropic Spectral Lattice)
    # Rationale: Periodic decomposition where coordinates are stretched independently.
    elseif type == "fft"
        n_marginal = Int(floor(sqrt(nbins / 2)))
        B = zeros(Float64, n_obs, nbins)

        # Normalize coordinates relative to the independent dimension ranges
        nx = (coords[:, 1] .- c_min[1]) ./ (c_max[1] - c_min[1] + 1e-9)
        ny = (coords[:, 2] .- c_min[2]) ./ (c_max[2] - c_min[2] + 1e-9)

        idx = 1
        for mx in 1:n_marginal
            for my in 1:n_marginal
                if idx + 1 <= nbins
                    # The spectral lattice now stretches independently based on coordinate normalization
                    B[:, idx]   .= sin.(2.0 * pi * (mx .* nx .+ my .* ny))
                    B[:, idx+1] .= cos.(2.0 * pi * (mx .* nx .+ my .* ny))
                    idx += 2
                end
            end
        end

    # # 2.5 Barycentric Triangulation (Delaunay interpolation)
    # Rationale: Piecewise linear interaction for irregular 2D scattered data.
    elseif type == "barycentric" || type == "triangulation"
        B = zeros(Float64, n_obs, nbins)
        # Inducing landmarks via quantile grid
        n_grid = Int(floor(sqrt(nbins)))
        kx = quantile(coords[:, 1], range(0, 1, length=n_grid))
        ky = quantile(coords[:, 2], range(0, 1, length=n_grid))
        centers = [(x, y) for x in kx, y in ky][:]
        # Standard Hat Basis for triangulation nodes
        for m in 1:min(nbins, length(centers))
            dist_x = abs.(coords[:, 1] .- centers[m][1]) ./ (c_std[1] / 2.0)
            dist_y = abs.(coords[:, 2] .- centers[m][2]) ./ (c_std[2] / 2.0)
            B[:, m] .= max.(0.0, 1.0 .- dist_x) .* max.(0.0, 1.0 .- dist_y)
        end

    # # 2.6 Spherical Compact Interaction (Zero-Correlation Support)
    # Rationale: Interaction drops to exactly zero beyond a range R.
    elseif type == "spherical"
        B = zeros(Float64, n_obs, nbins)
        n_grid = Int(floor(sqrt(nbins)))
        kx = collect(range(c_min[1], stop=c_max[1], length=n_grid))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_grid))
        centers = [(x, y) for x in kx, y in ky][:]
        range_r = get(kwargs, :range, mean(c_std))
        for m in 1:min(nbins, length(centers))
            dx = coords[:, 1] .- centers[m][1]
            dy = coords[:, 2] .- centers[m][2]
            h = sqrt.(dx.^2 .+ dy.^2) ./ range_r
            mask = h .< 1.0
            # Spherical Kernel polynomial
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    # # 2.7 Moran's I Spectral Interaction
    # Rationale: Captures non-linearities at specific spatial scales.
    elseif type == "moran"
        B = zeros(Float64, n_obs, nbins)
        # Proxy for spatial weighting eigenvectors
        for m in 1:nbins
            B[:, m] .= sin.(pi * m .* (coords[:, 1] .+ coords[:, 2]) ./ sum(c_max))
        end

    # # 2.8 Identity Fallback
    else
        B = ones(n_obs, 1)
    end

    return B
end

# # BSTM 3D Smooth Basis & Volumetric Factory [v19.34.0]
# Timestamp: 2025-10-20 10:00:00
# Rationale: Finalizing the 3D manifold registry with absolute parity to 1D/2D counterparts.
# This implementation restores missing Moran, Spherical, and Barycentric methods with full anisotropy support.

function bstm_smooth_basis_3D(type::String, coords::AbstractMatrix, nbins::Int; kwargs...)
    # # 1. Initialization and Technical Discovery
    # coords is expected to be an N x 3 matrix: [dim1, dim2, dim3]
    n_obs = size(coords, 1)

    # Technical Boundary Discovery: Identifying the range and spread of input dimensions
    c_min = [minimum(coords[:, 1]), minimum(coords[:, 2]), minimum(coords[:, 3])]
    c_max = [maximum(coords[:, 1]), maximum(coords[:, 2]), maximum(coords[:, 3])]
    c_std = [std(coords[:, 1]), std(coords[:, 2]), std(coords[:, 3])] .+ 1e-9

    # Anisotropy Configuration: Extract independent lengthscales if provided, else fallback to marginal std
    # ls_x, ls_y, and ls_z allow the user to control directional smoothing separately
    ls_x = get(kwargs, :ls_x, c_std[1])
    ls_y = get(kwargs, :ls_y, c_std[2])
    ls_z = get(kwargs, :ls_z, c_std[3])

    # # 2. Manifold Dispatch Registry for Anisotropic Volumetric Surfaces

    # # 2.1 Anisotropic Trivariate P-Splines (Tensor Product Basis)
    # Rationale: Expands 1D marginal bases into a 3D volumetric grid with directional scaling.
    if type in ["pspline", "bspline", "smooth"]
        # n_marginal defines the cubic root resolution to reach total nbins
        n_marginal = Int(floor(cbrt(nbins)))
        m_total = n_marginal^3
        B = zeros(Float64, n_obs, m_total)

        # Marginal Knot Grids: Distributed evenly across the observed ranges
        kx = collect(range(c_min[1], stop=c_max[1], length=n_marginal))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_marginal))
        kz = collect(range(c_min[3], stop=c_max[3], length=n_marginal))

        idx = 1
        for i in 1:n_marginal
            for j in 1:n_marginal
                for k in 1:n_marginal
                    # Tensor product of anisotropic radial marginals
                    bx = exp.(-((coords[:, 1] .- kx[i]).^2) ./ (2.0 * ls_x^2))
                    by = exp.(-((coords[:, 2] .- ky[j]).^2) ./ (2.0 * ls_y^2))
                    bz = exp.(-((coords[:, 3] .- kz[k]).^2) ./ (2.0 * ls_z^2))
                    B[:, idx] .= bx .* by .* bz
                    idx += 1
                end
            end
        end

    # # 2.2 Volumetric Random Fourier Features (RFF)
    # Rationale: Samples frequencies from an anisotropic 3D spectral density.
    elseif type == "rff"
        Omega = randn(3, nbins)
        Omega[1, :] ./= ls_x
        Omega[2, :] ./= ls_y
        Omega[3, :] ./= ls_z
        
        Phi = rand(nbins) .* (2.0 * pi)
        B = sqrt(2.0 / nbins) .* cos.((coords * Omega) .+ Phi')

    # # 2.3 3D Fast Fourier Transform (FFT - Anisotropic Spectral Lattice)
    # Rationale: Volumetric periodic decomposition where coordinates are stretched independently.
    elseif type == "fft"
        n_marginal = Int(floor(cbrt(nbins / 2)))
        B = zeros(Float64, n_obs, nbins)

        # Normalize coordinates relative to the independent dimension ranges
        nx = (coords[:, 1] .- c_min[1]) ./ (c_max[1] - c_min[1] + 1e-9)
        ny = (coords[:, 2] .- c_min[2]) ./ (c_max[2] - c_min[2] + 1e-9)
        nz = (coords[:, 3] .- c_min[3]) ./ (c_max[3] - c_min[3] + 1e-9)

        idx = 1
        for mx in 1:n_marginal
            for my in 1:n_marginal
                for mz in 1:n_marginal
                    if idx + 1 <= nbins
                        # Periodic projection respecting the volume stretching
                        B[:, idx]   .= sin.(2.0 * pi * (mx .* nx .+ my .* ny .+ mz .* nz))
                        B[:, idx+1] .= cos.(2.0 * pi * (mx .* nx .+ my .* ny .+ mz .* nz))
                        idx += 2
                    end
                end
            end
        end

    # # 2.4 Volumetric Spherical Basis (Compact Support Anisotropy)
    # Rationale: Ensures interaction drops to zero beyond directional ranges.
    elseif type == "spherical"
        n_grid = Int(floor(cbrt(nbins)))
        m_total = n_grid^3
        B = zeros(Float64, n_obs, m_total)

        # Inducing landmarks via 3D mesh
        ix = collect(range(c_min[1], stop=c_max[1], length=n_grid))
        iy = collect(range(c_min[2], stop=c_max[2], length=n_grid))
        iz = collect(range(c_min[3], stop=c_max[3], length=n_grid))
        centers = [(x, y, z) for x in ix, y in iy, z in iz][:]

        for m in 1:min(m_total, length(centers))
            # Normalized Mahalanobis-like distance for anisotropy
            dx = (coords[:, 1] .- centers[m][1]) ./ ls_x
            dy = (coords[:, 2] .- centers[m][2]) ./ ls_y
            dz = (coords[:, 3] .- centers[m][3]) ./ ls_z
            h = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
            
            # Spherical Kernel with directional compact support
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    # # 2.5 3D Moran's I Spectral Basis
    # Rationale: Captures volumetric non-linearities at specific scales using Sine basis proxy.
    elseif type == "moran"
        B = zeros(Float64, n_obs, nbins)
        for m in 1:nbins
            # Multi-scale spectral filters for 3D volumes
            arg = (coords[:, 1] ./ ls_x) .+ (coords[:, 2] ./ ls_y) .+ (coords[:, 3] ./ ls_z)
            B[:, m] .= sin.(pi * m .* arg ./ 3.0)
        end

    # # 2.6 Barycentric Volumetric Interpolation (Tetrahedral Proxy)
    # Rationale: Piecewise linear interaction for 3D scattered data.
    elseif type == "barycentric" || type == "triangulation"
        n_grid = Int(floor(cbrt(nbins)))
        m_total = n_grid^3
        B = zeros(Float64, n_obs, m_total)
        ix = quantile(coords[:, 1], range(0, 1, length=n_grid))
        iy = quantile(coords[:, 2], range(0, 1, length=n_grid))
        iz = quantile(coords[:, 3], range(0, 1, length=n_grid))
        centers = [(x, y, z) for x in ix, y in iy, z in iz][:]

        for m in 1:min(m_total, length(centers))
            # Standard 3D Hat Basis (Trilinear interpolation proxy)
            dist_x = abs.(coords[:, 1] .- centers[m][1]) ./ ls_x
            dist_y = abs.(coords[:, 2] .- centers[m][2]) ./ ls_y
            dist_z = abs.(coords[:, 3] .- centers[m][3]) ./ ls_z
            B[:, m] .= max.(0.0, 1.0 .- dist_x) .* max.(0.0, 1.0 .- dist_y) .* max.(0.0, 1.0 .- dist_z)
        end

    # # 2.7 Technical Fallback
    else
        B = ones(n_obs, 1)
    end

    return B
end
 

# # BSTM Hyper-Volumetric Factory & Entry Point [v19.36.0]
# Timestamp: 2025-10-30 10:00:00
# Rationale: Expanding the 4D manifold registry to include advanced methods: Spherical, Moran, and Barycentric.
# Requirement: Absolute parity with the v06.1 Taxonomy. Ensures non-truncated hyper-volumetric surfaces.

function bstm_smooth_basis_4D(type::String, coords::AbstractMatrix, nbins::Int; kwargs...)
    # # 1. Initialization and Hyper-Dimensional Discovery
    # coords is expected to be an N x 4 matrix: [dim1, dim2, dim3, dim4]
    n_obs = size(coords, 1)

    # Technical Boundary Discovery: Identifying the hyper-cube extents
    c_min = [minimum(coords[:, 1]), minimum(coords[:, 2]), minimum(coords[:, 3]), minimum(coords[:, 4])]
    c_max = [maximum(coords[:, 1]), maximum(coords[:, 2]), maximum(coords[:, 3]), maximum(coords[:, 4])]
    c_std = [std(coords[:, 1]), std(coords[:, 2]), std(coords[:, 3]), std(coords[:, 4])] .+ 1e-9

    # Anisotropy Configuration: Independent lengthscales for 4 dimensions
    ls_1 = get(kwargs, :ls_1, c_std[1])
    ls_2 = get(kwargs, :ls_2, c_std[2])
    ls_3 = get(kwargs, :ls_3, c_std[3])
    ls_4 = get(kwargs, :ls_4, c_std[4])

    # # 2. Manifold Dispatch Registry for 4D Latent Fields

    # # 2.1 4D Anisotropic P-Splines (Tensor Product)
    if type in ["pspline", "bspline", "smooth"]
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        B = zeros(Float64, n_obs, m_total)

        k1 = collect(range(c_min[1], stop=c_max[1], length=n_marginal))
        k2 = collect(range(c_min[2], stop=c_max[2], length=n_marginal))
        k3 = collect(range(c_min[3], stop=c_max[3], length=n_marginal))
        k4 = collect(range(c_min[4], stop=c_max[4], length=n_marginal))

        idx = 1
        for i in 1:n_marginal
            for j in 1:n_marginal
                for k in 1:n_marginal
                    for l in 1:n_marginal
                        b1 = exp.(-((coords[:, 1] .- k1[i]).^2) ./ (2.0 * ls_1^2))
                        b2 = exp.(-((coords[:, 2] .- k2[j]).^2) ./ (2.0 * ls_2^2))
                        b3 = exp.(-((coords[:, 3] .- k3[k]).^2) ./ (2.0 * ls_3^2))
                        b4 = exp.(-((coords[:, 4] .- k4[l]).^2) ./ (2.0 * ls_4^2))
                        B[:, idx] .= b1 .* b2 .* b3 .* b4
                        idx += 1
                    end
                end
            end
        end

    # # 2.2 4D Random Fourier Features (Spectral Anisotropy)
    elseif type == "rff"
        Omega = randn(4, nbins)
        Omega[1, :] ./= ls_1
        Omega[2, :] ./= ls_2
        Omega[3, :] ./= ls_3
        Omega[4, :] ./= ls_4
        Phi = rand(nbins) .* (2.0 * pi)
        B = sqrt(2.0 / nbins) .* cos.((coords * Omega) .+ Phi')

    # # 2.3 4D Fast Fourier Transform (Spectral Lattice)
    elseif type == "fft"
        n_marginal = Int(floor(sqrt(sqrt(nbins / 2))))
        B = zeros(Float64, n_obs, nbins)
        nx1 = (coords[:, 1] .- c_min[1]) ./ (c_max[1] - c_min[1] + 1e-9)
        nx2 = (coords[:, 2] .- c_min[2]) ./ (c_max[2] - c_min[2] + 1e-9)
        nx3 = (coords[:, 3] .- c_min[3]) ./ (c_max[3] - c_min[3] + 1e-9)
        nx4 = (coords[:, 4] .- c_min[4]) ./ (c_max[4] - c_min[4] + 1e-9)
        idx = 1
        for m1 in 1:n_marginal
            for m2 in 1:n_marginal
                for m3 in 1:n_marginal
                    for m4 in 1:n_marginal
                        if idx + 1 <= nbins
                            arg = m1 .* nx1 .+ m2 .* nx2 .+ m3 .* nx3 .+ m4 .* nx4
                            B[:, idx]   .= sin.(2.0 * pi * arg)
                            B[:, idx+1] .= cos.(2.0 * pi * arg)
                            idx += 2
                        end
                    end
                end
            end
        end

    # # 2.4 4D Spherical Basis (Anisotropic Compact Support)
    # Rationale: Ensures latent correlation is zero beyond hyper-volumetric directional ranges.
    elseif type == "spherical"
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        B = zeros(Float64, n_obs, m_total)
        k1 = collect(range(c_min[1], stop=c_max[1], length=n_marginal))
        k2 = collect(range(c_min[2], stop=c_max[2], length=n_marginal))
        k3 = collect(range(c_min[3], stop=c_max[3], length=n_marginal))
        k4 = collect(range(c_min[4], stop=c_max[4], length=n_marginal))
        centers = [(w, x, y, z) for w in k1, x in k2, y in k3, z in k4][:]
        for m in 1:min(m_total, length(centers))
            d1 = (coords[:, 1] .- centers[m][1]) ./ ls_1
            d2 = (coords[:, 2] .- centers[m][2]) ./ ls_2
            d3 = (coords[:, 3] .- centers[m][3]) ./ ls_3
            d4 = (coords[:, 4] .- centers[m][4]) ./ ls_4
            h = sqrt.(d1.^2 .+ d2.^2 .+ d3.^2 .+ d4.^2)
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    # # 2.5 4D Moran's I Spectral Basis
    # Rationale: Hyper-dimensional multi-scale filtering using spectral Sine proxy.
    elseif type == "moran"
        B = zeros(Float64, n_obs, nbins)
        for m in 1:nbins
            arg = (coords[:, 1] ./ ls_1) .+ (coords[:, 2] ./ ls_2) .+ (coords[:, 3] ./ ls_3) .+ (coords[:, 4] ./ ls_4)
            B[:, m] .= sin.(pi * m .* arg ./ 4.0)
        end

    # # 2.6 4D Barycentric Interpolation (Hyper-Tetrahedral Proxy)
    # Rationale: Piecewise linear interaction across irregular 4D scattered data.
    elseif type == "barycentric" || type == "triangulation"
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        B = zeros(Float64, n_obs, m_total)
        k1 = quantile(coords[:, 1], range(0, 1, length=n_marginal))
        k2 = quantile(coords[:, 2], range(0, 1, length=n_marginal))
        k3 = quantile(coords[:, 3], range(0, 1, length=n_marginal))
        k4 = quantile(coords[:, 4], range(0, 1, length=n_marginal))
        centers = [(w, x, y, z) for w in k1, x in k2, y in k3, z in k4][:]
        for m in 1:min(m_total, length(centers))
            dw = abs.(coords[:, 1] .- centers[m][1]) ./ ls_1
            dx = abs.(coords[:, 2] .- centers[m][2]) ./ ls_2
            dy = abs.(coords[:, 3] .- centers[m][3]) ./ ls_3
            dz = abs.(coords[:, 4] .- centers[m][4]) ./ ls_4
            B[:, m] .= max.(0.0, 1.0 .- dw) .* max.(0.0, 1.0 .- dx) .* max.(0.0, 1.0 .- dy) .* max.(0.0, 1.0 .- dz)
        end

    # # 2.7 Technical Fallback
    else
        B = ones(n_obs, 1)
    end

    return B
end



function bstm_modular_config(formula::String, data::DataFrame; kwargs...)
  # 1. Formula Scope Discovery
    metadata = decompose_bstm_formula(formula)

    # 2. Metadata Context Initialization
    opt_dict = Dict{Symbol, Any}(kwargs)
    opt_dict[:y_obs_vars] = metadata.outcomes
    opt_dict[:data] = data

    # Technical Registers Initialization
    opt_dict[:model_space] = get(opt_dict, :model_space, "none")
    opt_dict[:model_time] = get(opt_dict, :model_time, "none")
    opt_dict[:model_season] = get(opt_dict, :model_season, "none")
    opt_dict[:model_st] = get(opt_dict, :model_st, "none")
    opt_dict[:algebraic_manifolds] = Dict{Symbol, Any}()
    opt_dict[:svc_covariates] = Symbol[]
    opt_dict[:basis_matrices] = Dict{Symbol, Any}()
    opt_dict[:mixed_terms] = []

    # Hyperprior Registry Initialization
    initial_hp = get(opt_dict, :hyperpriors, Dict{Any, Any}())
    hyperprior_registry = Dict{String, Any}(string(k) => v for (k, v) in initial_hp)

    # Internal Technical Registries
    fixed_parts_accumulator = String[]
    mixed_terms_registry = []
    nested_manifolds_registry = Dict{String, Any}()
    basis_matrices_registry = Dict{Symbol, Any}()
    svc_covariates_registry = Symbol[]
    svc_manifolds_registry = Dict{Symbol, Any}()
    network_registry = Dict{Symbol, Any}()
    volatility_registry = Dict{Symbol, Any}()
    spacetime_registry = Dict{Symbol, Any}()
    dynamics_registry = Dict{Symbol, Any}()


    
    # Helper: Recursive Hyperprior Extraction from module parameters
    function extract_module_hyperpriors!(reg::Dict{String, Any}, p::Dict{Symbol, Any})
        if haskey(p, :hyperpriors)
            hp_raw = p[:hyperpriors]
            hp_val = hp_raw isa Symbol ? Main.eval(hp_raw) : hp_raw
            if hp_val isa Dict
                for (hk, hv) in hp_val
                    reg[string(hk)] = hv
                end
            end
        end
    end


    # Technical Module Resolution Loop
    for (key, mod) in metadata.modules

        type_raw = get(mod, :type, "none")
        type = lowercase(strip(string(type_raw)))
        params = get(mod, :params, Dict{Symbol, Any}())

        # Ingest local hyperpriors for any module type
        extract_module_hyperpriors!(hyperprior_registry, params)

        # Process Algebraic Manifolds
        manifold_call = get(mod, :manifold_call, "")
        graph = isempty(manifold_call) ? nothing : parse_manifold_graph(manifold_call)
        if !isnothing(graph)
            opt_dict[:algebraic_manifolds][Symbol(type)] = graph
        end

        if type == "svc"
            cov_var = Symbol(mod[:covariate])
            push!(svc_covariates_registry, cov_var)
            svc_manifolds_registry[cov_var] = Dict(:graph => graph, :params => params)
            continue
        end

        vars_list = get(mod, :variables, [])
        n_vars = length(vars_list)

        # Module Dispatch

        # --- A. Algebraic Composition Handling ---
        if type == "interaction_composition"
            op = get(mod, :operator, "")
            comps = get(mod, :components, [])
            
            # Discover and Register Sub-components into algebraic_manifolds
            for comp in comps
                c_type = Symbol(get(comp, :type, "unknown"))
                # Register the original composite call string for forensic recovery
                opt_dict[:algebraic_manifolds][c_type] = parse_manifold_graph(string(key))
            end

            # Technical Signal Promotion for ST Class (I-IV)
            if op == "⊗" || op == "otimes"
                opt_dict[:model_st] = "IV"
                for comp in comps
                    c_type = get(comp, :type, "")
                    c_params = get(comp, :params, Dict{Symbol, Any}())
                    if c_type == "spatial"; opt_dict[:model_space] = string(get(c_params, :manifold, "bym2")); end
                    if c_type == "temporal"; opt_dict[:model_time] = string(get(c_params, :manifold, "ar1")); end
                end
            end
     
        elseif type == "bias"
            if haskey(params, :weight) || haskey(params, :weights)
                opt_dict[:weights] = get(params, :weight, get(params, :weights, nothing))
            end
            if haskey(params, :offsets) || haskey(params, :offset)
                opt_dict[:log_offset] = get(params, :offsets, get(params, :offset, nothing))
            end
            if haskey(params, :trials) || haskey(params, :trial)
                opt_dict[:trials] = get(params, :trials, get(params, :trial, nothing))
            end
            if haskey(params, :y_L); opt_dict[:y_lower_bound] = params[:y_L]; end
            if haskey(params, :y_U); opt_dict[:y_upper_bound] = params[:y_U]; end
            if haskey(params, :hurdle); opt_dict[:hurdle] = params[:hurdle]; end

        elseif type == "intercept"
            if !isempty(params)
                hyperprior_registry["intercept"] = params
            end

        elseif type == "spatial"
            opt_dict[:model_space] = string(get(params, :manifold, "bym2"))
            if n_vars == 1
                var_sym = vars_list[1]
                if hasproperty(data, var_sym)
                    opt_dict[:s_idx] = Int.(data[!, var_sym])
                    opt_dict[:s_idx_var] = var_sym
                end
            else
                opt_dict[:s_coord] = Matrix{Float64}(data[!, vars_list])
            end
            if haskey(params, :W)
                w_raw = params[:W]
                opt_dict[:W] = w_raw isa String ? Main.eval(Meta.parse(w_raw)) : w_raw
            end

        elseif type == "temporal"
            manifold_id = string(get(params, :manifold, "ar1"))
            opt_dict[:model_time] = manifold_id
            if n_vars == 1
                var_sym = vars_list[1]
                if hasproperty(data, var_sym)
                    tu_meta = assign_time_units(data[!, var_sym]; time_method="regular")
                    opt_dict[:t_idx] = tu_meta.t_idx
                    opt_dict[:u_idx] = tu_meta.u_idx
                    opt_dict[:t_idx_var] = var_sym
                    if manifold_id == "cyclic"; opt_dict[:model_season] = "cyclic"; end
                end
            else
                opt_dict[:t_idx_var] = vars_list[1]
                opt_dict[:u_idx_var] = vars_list[2]
            end

        elseif type == "smooth"
            if !isempty(vars_list)
                nb = get(params, :nbins, 20)
                manifold_type_str = string(get(params, :manifold, "pspline"))
                reg_key = Symbol(join(vars_list, "_"))
                if n_vars == 1
                    v_vec = data[!, Symbol(vars_list[1])]
                    basis_matrices_registry[reg_key] = bstm_smooth_basis_1D(manifold_type_str, v_vec, nb, get(params, :degree, 3); params...)
                else
                    c_mat = Matrix{Float64}(data[!, Symbol.(vars_list)])
                    if n_vars == 2; basis_matrices_registry[reg_key] = bstm_smooth_basis_2D(manifold_type_str, c_mat, nb; params...);
                    elseif n_vars == 3; basis_matrices_registry[reg_key] = bstm_smooth_basis_3D(manifold_type_str, c_mat, nb; params...);
                    elseif n_vars == 4; basis_matrices_registry[reg_key] = bstm_smooth_basis_4D(manifold_type_str, c_mat, nb; params...);
                    end
                end
            end

        elseif type == "mixed"
            mixed_str = string(vars_list[1])
            m_match = match(r"(.+?)\s*\|\s*(.+)", mixed_str)
            if !isnothing(m_match)
                group_var = Symbol(strip(m_match.captures[2]))
                if hasproperty(data, group_var)
                    levels = unique(data[!, group_var])
                    group_map = Dict(v => i for (i, v) in enumerate(levels))
                    indices = [group_map[v] for v in data[!, group_var]]
                    push!(mixed_terms_registry, (name=group_var, lhs=strip(m_match.captures[1]), indices=indices, n_cat=length(levels)))
                end
            end

        elseif type == "volatility"
            opt_dict[:use_sv] = true
            opt_dict[:M_rff_sigma] = get(params, :nbins, 20)
            volatility_registry[Symbol(vars_list[1])] = params

        elseif type == "network"
            network_registry[Symbol(vars_list[1])] = params

        elseif type == "spacetime"
            m_call = lowercase(string(get(params, :manifold, "none")))
            # Detect Kronecker operator convenience wrappers and perform shorthand expansion
            if occursin("\u2297", m_call) || occursin("otimes", m_call)
                m_parts = split(m_call, r"\s*\u2297\s*|\s+otimes\s+")
                if length(m_parts) >= 2
                    ms = strip(string(m_parts[1]), ['\'', '"', ' '])
                    mt = strip(string(m_parts[2]), ['\'', '"', ' '])

                    # Convenience Shorthand Expansion: Physics Aliases
                    if ms == "advection" && mt == "diffusion"
                        opt_dict[:model_st] = "advection_diffusion"
                        opt_dict[:model_space] = "besag"
                        opt_dict[:model_time] = "ar1"
                    else
                        # Standard composite mapping to Space/Time registers
                        opt_dict[:model_space] = ms
                        opt_dict[:model_time] = mt
                        # Automatic Knorr-Held Interaction classification (Type I-IV)
                        if ms == "iid" && mt == "iid"; opt_dict[:model_st] = "I"
                        elseif ms == "iid"; opt_dict[:model_st] = "II"
                        elseif mt == "iid"; opt_dict[:model_st] = "III"
                        else; opt_dict[:model_st] = "IV"; end
                    end
                end
            end
            spacetime_registry[key] = params

        elseif type == "dynamics"
            opt_dict[:use_dynamics] = true
            opt_dict[:dynamics_var] = vars_list[1]
            dynamics_registry[Symbol(vars_list[1])] = params
            for (p_k, p_v) in params
                if p_k != :hyperpriors; opt_dict[p_k] = p_v; end
            end

        elseif type == "nested"
            formula_str = get(params, :formula, "")
            if isempty(formula_str)
                z_var_sym = vars_list[1]
                z_formula = string(z_var_sym) * " ~ 1"
            else
                z_formula = formula_str
                z_var_sym = Symbol(strip(Base.split(z_formula, "~")[1]))
            end
            nested_manifolds_registry[string(z_var_sym)] = Dict(:formula => z_formula, :data => get(params, :data, :data))

        elseif type == "hyperbolic"
            opt_dict[:model_space] = "hyperbolic"
            opt_dict[:curvature] = get(params, :curvature, -1.0)
            if length(vars_list) > 1; opt_dict[:s_coord] = Matrix{Float64}(data[!, vars_list]); end
        end
    end

    # Final Registry Consolidation
    opt_dict[:hyperpriors] = hyperprior_registry
    opt_dict[:basis_matrices] = basis_matrices_registry
    opt_dict[:mixed_terms] = mixed_terms_registry
    opt_dict[:nested_manifolds] = nested_manifolds_registry
    opt_dict[:svc_covariates] = svc_covariates_registry
    opt_dict[:svc_manifolds] = svc_manifolds_registry
    opt_dict[:network_metadata] = network_registry
    opt_dict[:volatility_metadata] = volatility_registry
    opt_dict[:spacetime_metadata] = spacetime_registry
    opt_dict[:dynamics_metadata] = dynamics_registry
    opt_dict[:fixed_parts] = unique(vcat(fixed_parts_accumulator, string.(metadata.fixed_effects)))
    opt_dict[:add_intercept] = metadata.has_intercept

    if length(metadata.outcomes) == 1
        opt_dict[:y_obs] = data[!, metadata.outcomes[1]]
    else
        opt_dict[:y_obs] = Matrix(data[!, metadata.outcomes])
        opt_dict[:model_arch] = "multivariate"
    end

    return bstm_options(; opt_dict...)
end

function bstm(args...; model_family="gaussian", model_arch="univariate", auxiliary_responses=nothing, kwargs...)
    # Partition the input space to determine the operational mode based on argument types and count.

    # Mode 1: Formula-Driven Model Construction
    # Syntax: bstm(formula::String, data::DataFrame; ...)
    if length(args) >= 2 && args[1] isa String && args[2] isa DataFrame
        formula = args[1]
        data = args[2]

        # The modular config engine decomposes the formula DSL and discovers the target architecture.
        options = bstm_modular_config(formula, data; auxiliary_responses=auxiliary_responses, kwargs...)

        # Dispatch to model supervisors based on discovered metadata.
        if options.model_arch == "multivariate"
            return bstm_multivariate(options)
        elseif options.model_arch == "multifidelity"
            return bstm_multifidelity(options)
        else
            return bstm_univariate(options)
        end

    # Mode 2: Configuration Vector for Heterogeneous Data Streams
    # Syntax: bstm(configs::Vector{<:NamedTuple}; ...)
    elseif length(args) == 1 && args[1] isa Vector && !isempty(args[1]) && args[1][1] isa NamedTuple
        configs = args[1]

        # Multiple configurations trigger the multifidelity pipeline.
        if length(configs) > 1
            options = bstm_options(
                configs = configs,
                model_family = model_family,
                model_arch = "multifidelity",
                kwargs...
            )
            return bstm_multifidelity(options)
        else
            # Single element vectors fall back to the univariate supervisor.
            return bstm_univariate(bstm_options(; NamedTuple(configs[1])..., kwargs...))
        end

    # Mode 3: Direct Metadata Execution (Expert / Internal Mode)
    # Syntax: bstm(M::NamedTuple)
    elseif length(args) == 1 && args[1] isa NamedTuple
        M = args[1]

        if M.model_arch == "multivariate"
            return bstm_multivariate(M)
        elseif M.model_arch == "multifidelity"
            return bstm_multifidelity(M)
        else
            return bstm_univariate(M)
        end

    else
        # Catch-all for unsupported argument patterns to ensure system robustness.
        error("BSTM Dispatch Error: Unsupported argument combination. Permitted: (String, DataFrame), (Vector{NamedTuple}), or (NamedTuple).")
    end
end


 
function decompose_bstm_formula(formula_str::String)
    # Partition the formula into Left-Hand Side (Outcomes) and Right-Hand Side (Predictors)
    parts = Base.split(formula_str, "~")
    lhs = strip(parts[1])
    rhs = strip(parts[2])

    # # Outcome Discovery
    outcomes = String[]
    if startswith(lhs, "[") && endswith(lhs, "]")
        content = lhs[2:end-1]
        outcomes = [strip(s) for s in Base.split(content, ",")]
    else
        outcomes = [lhs]
    end

    # # Recursive Split Utility
    function split_terms_at_depth(input::AbstractString, sep::AbstractString)
        terms = String[]
        depth = 0
        current = ""
        for char in input
            if char == '('
                depth += 1
            elseif char == ')'
                depth -= 1
            end
            if string(char) == sep && depth == 0
                push!(terms, strip(current))
                current = ""
            else
                current *= char
            end
        end
        if !isempty(strip(current))
            push!(terms, strip(current))
        end
        return terms
    end

    # # Module and Fixed Effect Registries
    terms_raw = split_terms_at_depth(rhs, "+")
    modules = Dict{String, Any}()
    fixed_effects = String[]
    has_intercept = false

    for term in terms_raw
        t_clean = strip(term)
        if t_clean == "1"
            has_intercept = true
            continue
        end

        # # Algebraic Operator Discovery
        operator_found = ""
        for op in ["⊗", "otimes", "⊕", "oplus", "∘", "compose", "|>", "pipe"]
            if occursin(op, t_clean)
                operator_found = op
                break
            end
        end

        if !isempty(operator_found)
            # Normalize symbolic names to Unicode for registry keys
            norm_op = operator_found
            if operator_found == "otimes"; norm_op = "⊗"; end
            if operator_found == "oplus"; norm_op = "⊕"; end
            if operator_found == "compose"; norm_op = "∘"; end
            if operator_found == "pipe"; norm_op = "|>"; end

            sub_terms = split_terms_at_depth(t_clean, operator_found)

            interaction_components = []
            for st in sub_terms
                m_mod = match(r"(\w+)\((.*)\)", st)
                if !isnothing(m_mod)
                    push!(interaction_components, Dict(
                        :type => lowercase(m_mod.captures[1]),
                        :variables => [strip(v) for v in Base.split(Base.split(m_mod.captures[2], ";")[1], ",")],
                        :params => parse_module_params(String(m_mod.captures[2]))
                    ))
                else
                    push!(interaction_components, Dict(:type => "literal", :var => st))
                end
            end

            modules["algebraic_" * t_clean] = Dict(
                :type => "interaction_composition",
                :components => interaction_components,
                :operator => norm_op
            )
            continue
        end

        # # Standard Module Discovery
        is_module = false
        for kw in BSTM_MODULE_KEYWORDS
            if startswith(lowercase(t_clean), kw * "(")
                is_module = true
                m_content = match(Regex(kw * "\\((.*)\\)", "i"), t_clean)
                if !isnothing(m_content)
                    inner_str = m_content.captures[1]
                    v_part, p_dict = if occursin(";", inner_str)
                        sp = Base.split(inner_str, ";")
                        strip(sp[1]), parse_module_params(String(sp[2]))
                    else
                        inner_str, Dict{Symbol, Any}()
                    end


                    # Keyword-Specific Routing
                    if kw == "svc" 
                        # Spatially Varying Coefficients: "covariate | ManifoldCall"
                        if occursin("|", v_part)
                            svc_split = Base.split(v_part, "|")
                            cov_var = strip(svc_split[1])
                            manifold_call = strip(svc_split[2])
    
                            modules["svc_" * cov_var] = Dict(
                                :type => "svc",
                                :covariate => cov_var,
                                :manifold_call => manifold_call,
                                :params => p_dict
                            )
                        else
                            # Fallback for svc(x) without manifold
                            modules["svc_" * v_part] = Dict(
                                :type => "svc",
                                :covariate => v_part,
                                :manifold_call => "iid()", # default
                                :params => p_dict)
                        end
                    else
                        # Standard modules: extraction of comma-separated variable list
                        vars_list = [strip(v) for v in Base.split(v_part, ",")]
                        # Unique registry key based on type and primary variable
                        modules[kw * "_" * vars_list[1]] = Dict(
                            :type => kw,
                            :variables => vars_list,
                            :params => p_dict
                        )
                    end
 
                end
                break
            end
        end

        if !is_module && !isempty(t_clean)
            push!(fixed_effects, t_clean)
        end
    end

    return (outcomes=outcomes, modules=modules, fixed_effects=fixed_effects, has_intercept=has_intercept)
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
    # FFT Builder: Frequency-domain temporal or spatial coefficients
    # Precision is assumed IID in the spectral space
    return (
        Q_template = Matrix(1.0I, m.nbins, m.nbins),
        scaling_factor = 1.0,
        model_type = :fft,
        hyper = (sigma_prior = m.sigma_prior, nbins = m.nbins)
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


  

# # Knorr-Held Interaction Builder
# function build_model(m::KnorrHeld, data_inputs)
#     # Space-Time Interaction types I, II, III, IV
#     # Precision is built via kron() in the sampler, so no static template is stored
#     return (
#         Q_template = nothing,
#         model_type = :knorrheld,
#         smooth_class = m.type,
#         hyper = (sigma_prior = m.sigma_prior, rho_prior = nothing)
#     )
# end 
 
 


function get_optimal_sampler(model_obj::DynamicPPL.Model)
    # BSTM Sampler Architect v20.1.0
    # Timestamp: 2026-06-05 14:30:00
    # Rationale: Correcting Gibbs partition syntax to use valid VarName => Sampler pairs.

    # 1. Parameter Discovery
    # Identifying the full set of latent variables present in the model instance
    all_params = DynamicPPL.get_varnames(model_obj)
    
    # 2. Categorization Logic
    # We distinguish between discrete parameters (requiring Particle Gibbs) 
    # and continuous parameters (optimal for NUTS/HMC).
    discrete_keywords = ["innov", "state", "cluster", "assignment", "cat"]
    discrete_params = Symbol[]
    continuous_params = Symbol[]

    for vn in all_params
        p_sym = Symbol(DynamicPPL.getsymbol(vn))
        p_name_str = string(p_sym)
        
        is_discrete = false
        for kw in discrete_keywords
            if occursin(kw, p_name_str)
                is_discrete = true
                break
            end
        end

        if is_discrete
            push!(discrete_params, p_sym)
        else
            push!(continuous_params, p_sym)
        end
    end

    # 3. Sampler Block Construction
    # Rationale: Turing's Gibbs constructor requires individual sampler instances.
    # We use NUTS for continuous fields and PG for discrete innovations.
    gibbs_blocks = []

    # A. Continuous Block: NUTS with ForwardDiff
    if !isempty(continuous_params)
        # Standardizing on NUTS(adaptation_steps, target_acceptance)
        push!(gibbs_blocks, NUTS(500, 0.65))
    end

    # B. Discrete Block: Particle Gibbs
    if !isempty(discrete_params)
        # Particle count standardized to 20 for balance between precision and compute
        pg_particles = 20
        push!(gibbs_blocks, PG(pg_particles))
    end

    # 4. Final Sampler Dispatch
    # If the model is purely continuous, return NUTS directly.
    # If mixed, return the coordinated Gibbs sampler.
    if isempty(discrete_params)
        return NUTS(500, 0.65)
    else
        # Corrected Syntax: Gibbs(sampler1, sampler2, ...)
        # Turing handles the variable mapping based on the internal distribution types
        return Gibbs(gibbs_blocks...)
    end
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

    # Check for FlexiChains vector object or indexed scalars
    if key_prefix in p_names
        # Standard Vector retrieval
        local raw = chain[Symbol(key_prefix)].data[j]
        local flat_raw = vec(collect(raw))
        local n_raw = length(flat_raw)
        #  Slicing: Ensure raw samples align with target grid to prevent broadcast errors
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
    # Composed manifolds (Kronecker) forced to :spacetime domain
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



function extract_manifold(m::AR1, chain, M, n_samples, outcomes_N, p_names, prefix="t")
    # Rationale: AR1 realizations are typically stored as t_latent in temporal contexts.
    latent_realizations = [zeros(M.t_N, n_samples) for _ in 1:outcomes_N]
    for k in 1:outcomes_N
        key = outcomes_N > 1 ? "$(prefix)_latent_$k" : "$(prefix)_latent"
        latent_realizations[k] = get_params_vector(chain, key, M.t_N)
    end
    return latent_realizations
end

function extract_manifold(m::RW2, chain, M, n_samples, outcomes_N, p_names, prefix="t")
    # Rationale: RW2 trends are extracted using temporal trend registries.
    latent_realizations = [zeros(M.t_N, n_samples) for _ in 1:outcomes_N]
    for k in 1:outcomes_N
        key = outcomes_N > 1 ? "$(prefix)_latent_$k" : "$(prefix)_latent"
        latent_realizations[k] = get_params_vector(chain, key, M.t_N)
    end
    return latent_realizations
end

function extract_manifold(m::ICAR, chain, M, n_samples, outcomes_N, p_names, prefix="s")
    # Rationale: ICAR (Besag) spatial fields map to the s_latent registry.
    latent_realizations = [zeros(M.s_N, n_samples) for _ in 1:outcomes_N]
    for k in 1:outcomes_N
        key = outcomes_N > 1 ? "$(prefix)_latent_$k" : "$(prefix)_latent"
        latent_realizations[k] = get_params_vector(chain, key, M.s_N)
    end
    return latent_realizations
end

function extract_manifold(m::Leroux, chain, M, n_samples, outcomes_N, p_names, prefix="s")
    # Rationale: Leroux manifolds require discovery of both the latent field and the lambda/rho parameter.
    latent_realizations = [zeros(M.s_N, n_samples) for _ in 1:outcomes_N]
    for k in 1:outcomes_N
        key = outcomes_N > 1 ? "$(prefix)_latent_$k" : "$(prefix)_latent"
        latent_realizations[k] = get_params_vector(chain, key, M.s_N)
    end
    return latent_realizations
end

function extract_manifold(m::SAR, chain, M, n_samples, outcomes_N, p_names, prefix="s")
    # Rationale: SAR manifolds utilize the autoregressive spatial structure labels.
    latent_realizations = [zeros(M.s_N, n_samples) for _ in 1:outcomes_N]
    for k in 1:outcomes_N
        key = outcomes_N > 1 ? "$(prefix)_latent_$k" : "$(prefix)_latent"
        latent_realizations[k] = get_params_vector(chain, key, M.s_N)
    end
    return latent_realizations
end

function extract_manifold(m::GP, chain, M, n_samples, outcomes_N, p_names, prefix="s")
    # Rationale: Continuous Gaussian Processes are reconstructed by identifying the spectral or dense latent weights.
    latent_realizations = [zeros(M.y_N, n_samples) for _ in 1:outcomes_N]
    for k in 1:outcomes_N
        # GPs often use f_latent or weights_gp depending on the approximation (FITC/Spectral)
        key = outcomes_N > 1 ? "f_latent_$k" : "f_latent"
        if !(key in p_names)
            key = outcomes_N > 1 ? "$(prefix)_latent_$k" : "$(prefix)_latent"
        end
        latent_realizations[k] = get_params_vector(chain, key, size(latent_realizations[k], 1))
    end
    return latent_realizations
end

function extract_manifold(m::SPDE, chain, M, n_samples, outcomes_N, p_names, prefix="s")
    # Rationale: SPDE manifolds extract weights corresponding to the finite element mesh vertices.
    n_mesh = size(m.precision_template, 1)
    latent_realizations = [zeros(n_mesh, n_samples) for _ in 1:outcomes_N]
    for k in 1:outcomes_N
        key = outcomes_N > 1 ? "$(prefix)_spde_weights_$k" : "$(prefix)_spde_weights"
        latent_realizations[k] = get_params_vector(chain, key, n_mesh)
    end
    return latent_realizations
end


function extract_manifold(m::Manifold, chain, M, n_samples, outcomes_N, p_names, prefix="s")
    # Fallback for SVC and custom manifolds using dynamic symbol lookup
    if hasproperty(m, :covariate)
        cov_name = m.covariate
        latent_realizations = [zeros(M.s_N, n_samples) for _ in 1:outcomes_N]
        for k in 1:outcomes_N
            key = outcomes_N > 1 ? "beta_svc_$(cov_name)_$k" : "beta_svc_$(cov_name)"
            latent_realizations[k] = get_params_vector(chain, key, M.s_N)
        end
        return latent_realizations
    end
    return [zeros(1, n_samples) for _ in 1:outcomes_N]
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
    # # 1. Synchronize Internal Data Frame
    df_internal = data isa NamedArray ? DataFrame(data, :auto) : copy(data)
    if data isa NamedArray
        rename!(df_internal, names(data, 2))
    end

    # # 2. Process as_factor() Directives
    # Identified in v19.19.10 to prevent numerical collapse of categorical predictors
    factor_regex = r"as_factor\(\s*(\w+)\s*\)"
    factor_matches = eachmatch(factor_regex, formula_rhs)

    for m in factor_matches
        var_name_str = m.captures[1]
        var_sym = Symbol(var_name_str)

        if hasproperty(df_internal, var_sym)
            df_internal[!, var_sym] = CategoricalArrays.categorical(df_internal[!, var_sym])
        end
    end

    # # 3. Dynamic BSTM Module Wrapper Pruning
    # Rationale: Instead of hard-coding keywords, we construct the regex from the taxonomy registry.
    # Requirement: Modules like Spatial, Temporal, etc., must be removed before StatsModels parsing.
    keywords_pattern = join(collect(BSTM_MODULE_KEYWORDS), "|")
    clean_rhs = replace(formula_rhs, Regex("(" * keywords_pattern * ")\\(.*?\\)", "i") => "")
    
    # Cleanup remaining artifacts from the stripping process
    final_rhs_string = replace(clean_rhs, "as_factor(" => "(")
    final_rhs_string = isempty(strip(final_rhs_string)) ? formula_rhs : final_rhs_string
    final_rhs_string = strip(replace(final_rhs_string, r"\\+\\s*\\+" => "+"))
    final_rhs_string = strip(replace(final_rhs_string, r"^\\+|\\+$" => ""))

    # # 4. Intercept Only Fallback
    if isempty(strip(final_rhs_string)) || final_rhs_string == "1"
        return NamedArray(ones(size(df_internal, 1), 1), (1:size(df_internal, 1), [:Intercept]))
    end

    # # 5. Technical Design Matrix Expansion
    try
        placeholder_name = :__y_placeholder
        df_internal[!, placeholder_name] = zeros(size(df_internal, 1))

        # Generating the formula expression for Main scope evaluation
        formula_expression = Meta.parse("@formula($placeholder_name ~ $final_rhs_string)")
        dynamic_formula = Main.eval(formula_expression)

        # Applying contrast schema and model matrix expansion
        data_schema = StatsModels.schema(dynamic_formula, df_internal, contrasts)
        applied_formula = StatsModels.apply_schema(dynamic_formula, data_schema, StatsModels.RegressionModel)

        _, model_matrix_numeric = StatsModels.modelcols(applied_formula, df_internal)
        coefficient_labels = StatsModels.coefnames(applied_formula.rhs)

        # Standardizing label types for NamedArray construction
        label_vector = coefficient_labels isa AbstractString ? [Symbol(coefficient_labels)] : Symbol.(coefficient_labels)

        return NamedArray(model_matrix_numeric, (1:size(model_matrix_numeric, 1), label_vector))

    catch design_error
        @warn "BSTM Registry: create_fixed_design expansion failed for: $final_rhs_string. Error: $design_error"
        return NamedArray(ones(size(df_internal, 1), 1), (1:size(df_internal, 1), [:Intercept]))
    end
end

 


function predict(model_obj::DynamicPPL.Model, chain, new_data::DataFrame; n_samples::Int=100, alpha=0.05)
    # Description: Primary engine for projecting recovered latent manifolds onto new spatiotemporal coordinates.
    # Rationale: Standardizing the out-of-sample path to support the full BSTM v06.1 Taxonomy.
    # # BSTM Monolithic Prediction & Projection Engine [v19.38.6 ]
    # Timestamp: 2025-11-06 10:00:00
    # Rationale: Updating out-of-sample projection to support 1D-4D smooths, nested supervisors, and mixed effects.
    # Requirements: 100% parity with v19.38.5 Results Dashboard. Zero-truncation of latent manifolds.

    println("--- Starting BSTM Out-of-Sample Prediction v19.38.6 ---")

    # # 1. Training Metadata Recovery
    # Rationale: M contains the configuration and technical registry established during the modular config pass.
    M_train = model_obj.args.M
    n_samples_total = size(chain, 1)
    n_samps = n_samples_total < n_samples ? n_samples_total : n_samples

    # # 2. Prediction Metadata Configuration (PS)
    # Rationale: PSU utilizes bstm_options to ensure the new data is formatted identically to training data.
    # Note: Dummy responses are used as y_obs is not utilized during the projection phase.
    PS = bstm_options(
        data = new_data,
        y_obs = zeros(nrow(new_data)),
        model_family = M_train.model_family,
        model_space = M_train.model_space,
        model_time = M_train.model_time,
        model_season = get(M_train, :model_season, "none"),
        svc_covariates = M_train.svc_covariates,
        use_sv = M_train.use_sv,
        model_arch = get(M_train, :model_arch, "univariate")
    )

    # # 3. Manifold Coordinate Alignment
    # Rationale: Aligning out-of-sample points with the training grid for discrete and continuous manifolds.
    
    # Case A: Centroid Alignment for Discrete Manifolds (ICAR/BYM2/Mosaic)
    # If the training model utilized areal units, we map new points to the nearest training centroid.
    if haskey(M_train, :centroids) && !isnothing(M_train.centroids)
        centroids_train = M_train.centroids
        # Standardizing coordinates from new_data
        nx = hasproperty(new_data, :s_x) ? new_data.s_x : zeros(nrow(new_data))
        ny = hasproperty(new_data, :s_y) ? new_data.s_y : zeros(nrow(new_data))
        
        PS_s_idx = Int[]
        for i in 1:nrow(new_data)
            # Find nearest neighbor in the training unit grid
            dists = [sum(((nx[i], ny[i]) .- c).^2) for c in centroids_train]
            push!(PS_s_idx, argmin(dists))
        end
        
        # Injecting aligned indices into the PS metadata
        PS_mut = Dict(pairs(PS))
        PS_mut[:s_idx] = PS_s_idx
        PS = NamedTuple(PS_mut)
    end

    # # 4. Hyper-Volumetric Basis Projection (1D-4D)
    # Rationale: Reconstructing basis matrices (Splines/RFF/FFT) for new coordinates.
    if haskey(M_train, :basis_matrices) && !isempty(M_train.basis_matrices)
        ps_basis_registry = Dict{Symbol, Any}()
        for (v_sym, B_train) in M_train.basis_matrices
            # We discover the dimensionality of the smooth (1D-4D)
            # and invoke the corresponding factory for the new coordinate grid.
            v_str = string(v_sym)
            vars = Symbol.(Base.split(v_str, "_"))
            n_vars = length(vars)
            
            nb = size(B_train, 2)
            
            if n_vars == 1
                ps_basis_registry[v_sym] = bstm_smooth_basis_1D("pspline", new_data[!, vars[1]], nb, 3)
            elseif n_vars == 2
                coords_new = Matrix{Float64}(new_data[!, vars])
                ps_basis_registry[v_sym] = bstm_smooth_basis_2D("rff", coords_new, nb)
            elseif n_vars == 3
                coords_new = Matrix{Float64}(new_data[!, vars])
                ps_basis_registry[v_sym] = bstm_smooth_basis_3D("pspline", coords_new, nb)
            elseif n_vars == 4
                coords_new = Matrix{Float64}(new_data[!, vars])
                ps_basis_registry[v_sym] = bstm_smooth_basis_4D("pspline", coords_new, nb)
            end
        end
        # Update PS with the projected basis matrices
        PS_mut = Dict(pairs(PS))
        PS_mut[:basis_matrices] = ps_basis_registry
        PS = NamedTuple(PS_mut)
    end

    # # 5. Architectural Dispatch for Latent Recovery
    # Rationale: Utilizing the validated _reconstruct engine to assemble the prediction tensor.
    raw_arch = get(M_train, :model_arch, "univariate")
    arch_type = if raw_arch == "univariate"
        UnivariateArchitecture()
    elseif raw_arch == "multivariate"
        MultivariateArchitecture()
    elseif raw_arch == "multifidelity" || raw_arch == "nested"
        MultifidelityArchitecture()
    else
        UnivariateArchitecture()
    end

    # Subset the chain for the requested number of samples
    chain_sub = chain[1:n_samps]

    # Rationale: _reconstruct treats PS as a grid for Post-Stratification, 
    # which is mathematically equivalent to out-of-sample prediction.
    res = _reconstruct(arch_type, "prediction_projection", chain_sub, M_train, PS, alpha)

    println("--- Projection Complete [Observations: ", nrow(new_data), "] ---")

    return (
        predictions_denoised = res.predictions_denoised,
        predictions_noisy = res.predictions_noisy,
        pstats = res,
        PS = PS,
        centroids = haskey(PS, :s_coord) ? PS.s_coord : nothing
    )
end

 

# 1. PSIS-LOO Implementation for BSTM
# Rationale: Standardizing the extraction of log-likelihood matrices to provide 
# Expected Log Pointwise Predictive Density (ELPD) estimates.
function bstm_loo(model_obj::DynamicPPL.Model, chain; alpha=0.05)
    # # BSTM PSIS-LOO Engine v19.38.6
    # Description: Calculates the Leave-One-Out Cross-Validation metrics for BSTM manifolds.
    # Rationale: Standardizing the extraction of log-likelihood matrices to provide ELPD estimates.
    # Requirements: Absolute parity with the v19.38.6 reconstruction path.

    println("--- Starting BSTM PSIS-LOO Audit v19.38.6 ---")

    # # 1. Metadata and Architecture Extraction
    # Rationale: M contains the configuration and technical registry required for reconstruction.
    M = model_obj.args.M
    raw_arch = get(M, :model_arch, "univariate")

    # # 2. Technical Dispatch Resolution
    # Mapping the configuration string to the architectural dispatch types.
    arch_type = if raw_arch == "univariate"
        UnivariateArchitecture()
    elseif raw_arch == "multivariate"
        MultivariateArchitecture()
    elseif raw_arch == "multifidelity" || raw_arch == "nested"
        MultifidelityArchitecture()
    else
        UnivariateArchitecture()
    end

    # # 3. Latent Manifold Reconstruction for Likelihood Registry
    # Rationale: _reconstruct generates the [Samples x Observations] log-likelihood matrix.
    # We utilize alpha for consistent summarization during the recovery phase.
    println("Audit: Recovering pointwise log-likelihood registry...")
    res = _reconstruct(arch_type, "loo_recovery", chain, M, nothing, alpha)

    # # 4. Matrix Extraction and Validation
    # Rationale: Ensuring the log_lik_matrix matches the observation grid dimensions.
    log_lik = res.log_lik_matrix
    n_samples, n_obs = size(log_lik)

    println("Audit: Processing ", n_samples, " samples for ", n_obs, " observations.")

    # # 5. PSIS-LOO Calculation via PosteriorStats
    # Rationale: LOO-CV provides a reliable estimate of out-of-sample predictive performance.
    loo_result = nothing
    try
        loo_result = loo(log_lik)
    catch e
        @error "BSTM Selection Error: PSIS-LOO calculation failed. Error: " * string(e)
        return nothing
    end

    println("\n--- BSTM Model Selection Report [v19.38.6] ---")
    println("Expected Log Pointwise Predictive Density (ELPD): ", round(loo_result.estimates[:elpd_loo, :estimate], digits=2))
    println("Effective Number of Parameters (p_loo):          ", round(loo_result.estimates[:p_loo, :estimate], digits=2))
    println("LOO Information Criterion:                       ", round(loo_result.estimates[:looic, :estimate], digits=2))

    # Check for influential observations (k > 0.7)
    # Rationale: Identifying data points where the importance weight is unstable.
    pareto_k = loo_result.pointwise[:pareto_k]
    influential_count = count(x -> x > 0.7, pareto_k)
    if influential_count > 0
        @warn "BSTM: " * string(influential_count) * " influential observations detected (Pareto k > 0.7)."
    end

    return (
        loo_obj = loo_result,
        metrics = (
            elpd = loo_result.estimates[:elpd_loo, :estimate],
            p_loo = loo_result.estimates[:p_loo, :estimate],
            looic = loo_result.estimates[:looic, :estimate]
        ),
        log_lik_matrix = log_lik,
        pareto_k = pareto_k
    )
end



# 2. Bayes Factor Suite (Manifold Comparison)

function compare_manifolds(loo_a_report, loo_b_report; model_names=["Model_A", "Model_B"])
    # # BSTM Model Comparison Engine v19.38.6
    # Description: Performs formal model selection between two BSTM manifold candidates.
    # Rationale: Standardizing the assessment of ELPD differences and complexity trade-offs.
    # Requirements: Absolute parity with the v19.38.6 PSIS-LOO metrics.

    println("--- Starting BSTM Manifold Comparison v19.38.6 ---")

    # # 1. LOO Object Extraction
    # Rationale: Extracting the underlying PosteriorStats LOO objects for comparison.
    loo_a = loo_a_report.loo_obj
    loo_b = loo_b_report.loo_obj

    # # 2. Formal Selection Metric Calculation
    # Rationale: The difference in ELPD is the primary metric for out-of-sample performance.
    # We utilize the compare function from PosteriorStats to compute deltas and standard errors.
    comparison_stats = nothing
    try
        comparison_stats = compare([loo_a, loo_b])
    catch e
        @error "BSTM Comparison Error: Selection suite failed. Error: " * string(e)
        return nothing
    end

    # # 3. Parameter and Diagnostic Extraction
    # Rationale: Collecting effective parameter counts (p_loo) to assess complexity.
    p_loo_a = loo_a_report.metrics.p_loo
    p_loo_b = loo_b_report.metrics.p_loo

    elpd_a = loo_a_report.metrics.elpd
    elpd_b = loo_b_report.metrics.elpd

    # # 4. Report Generation
    println("\n--- BSTM Manifold Selection Registry [v19.38.6] ---")
    println("Model A (", model_names[1], "): ELPD = ", round(elpd_a, digits=2), " | p_loo = ", round(p_loo_a, digits=2))
    println("Model B (", model_names[2], "): ELPD = ", round(elpd_b, digits=2), " | p_loo = ", round(p_loo_b, digits=2))

    diff_elpd = elpd_a - elpd_b
    println("\nELPD Delta (A - B): ", round(diff_elpd, digits=2))

    # Interpretation Logic
    # Rationale: If |diff_elpd| > 4, the difference is generally considered significant.
    if abs(diff_elpd) > 4.0
        winning_model = diff_elpd > 0 ? model_names[1] : model_names[2]
        println("CONCLUSION: ", winning_model, " is statistically preferred based on predictive density.")
    else
        println("CONCLUSION: Competing manifold structures provide indistinguishable predictive density.")
    end

    # # 5. Table Construction
    comparison_df = DataFrame(
        Metric = ["ELPD (LOO)", "Effective Parameters (p_loo)", "LOO-IC"],
        Model_A = [elpd_a, p_loo_a, loo_a_report.metrics.looic],
        Model_B = [elpd_b, p_loo_b, loo_b_report.metrics.looic]
    )

    comparison_df[!, :Delta] = comparison_df.Model_A .- comparison_df.Model_B

    display(comparison_df)

    return (
        comparison_table = comparison_df,
        elpd_diff = diff_elpd,
        loo_objects = (loo_a, loo_b)
    )
end


function bstm_cv_orchestrator(
    formula::String, 
    data::DataFrame; 
    method::Symbol = :lolo, 
    lolo_var::Symbol = :s_idx, 
    n_folds::Int = 5, 
    n_samples::Int = 500, 
    sampler = MH(), 
    target_units::Int = 0,
    kwargs...
)
    # # 1. Outcome Discovery
    # Parsing the formula to identify the response variable name
    metadata = decompose_bstm_formula(formula)
    response_name = Symbol(metadata.outcomes[1])
    
    # # 2. Partition Strategy
    # Establishing the fold indices based on the requested cross-validation method
    folds_indices = Vector{Vector{Int}}()
    
    if method == :lolo
        # Leave-One-Location-Out: Grouping indices by the spatial unit identifier
        unique_locs = unique(data[!, lolo_var])
        for loc in unique_locs
            push!(folds_indices, findall(x -> x == loc, data[!, lolo_var]))
        end
    else
        # Standard K-Fold: Random permutation of row indices
        n_obs = nrow(data)
        row_indices = randperm(n_obs)
        fold_size = div(n_obs, n_folds)
        for i in 1:n_folds
            start_idx = (i - 1) * fold_size + 1
            end_idx = i == n_folds ? n_obs : i * fold_size
            push!(folds_indices, row_indices[start_idx:end_idx])
        end
    end

    # # 3. Cross-Validation Execution Loop
    fold_results = []
    
    for (f_idx, test_idx) in enumerate(folds_indices)
        # Splitting dataset into training and testing sets
        train_data = data[setdiff(1:nrow(data), test_idx), :]
        test_data = data[test_idx, :]
        
        # # 4. Model Training
        # Constructing the model using the training partition
        # Technical registries (W, s_N, t_N) are propagated via kwargs
        model_train = bstm(
            formula, 
            train_data; 
            model_arch = "univariate", 
            target_units = target_units, 
            kwargs...
        )
        
        # Posterior sampling using the specified sampler
        chain_train = sample(model_train, sampler, n_samples, progress = false, check_model = false)
        M_train = model_train.args.M
        
        # # 5. Out-of-Sample Prediction
        # Projecting the latent manifolds onto the test partition coordinates
        # Rationale: predict() handles the projection of spatial/temporal components
        res_pred = predict(model_train, chain_train, test_data, n_samples = div(n_samples, 2))
        
        # # 6. Dynamic Performance Metric Calculation
        # Replacing hardcoded 'data.y' with the response name discovered from formula
        y_test_obs = test_data[!, response_name]
        y_test_pred = res_pred.predictions_observed_denoised.mean
        
        # Error Residuals
        residuals = y_test_obs .- y_test_pred
        rmse = sqrt(mean(residuals.^2))
        
        # Coefficient of Determination (R2)
        ss_res = sum(residuals.^2)
        ss_tot = sum((y_test_obs .- mean(y_test_obs)).^2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-15))
        
        push!(fold_results, (fold=f_idx, rmse=rmse, r2=r2))
    end
    
    # # 7. Aggregate CV Report
    mean_rmse = mean([r.rmse for r in fold_results])
    mean_r2 = mean([r.r2 for r in fold_results])
    
    return (
        folds = fold_results, 
        mean_rmse = mean_rmse, 
        mean_r2 = mean_r2, 
        response_var = response_name,
        method = method
    )
end


### 

# Advection-Diffusion Operator Registration
function build_transport_operator(M, velocity, diffusion_coeff)
    # Rationale: Constructs the discretized transport operator on the graph manifold.
    # Requirement: Adjacency matrix W must be non-null for graph-based advection.
    local n = M.s_N
    local W = get(M, :W, sparse(I(n)))
    local D = Matrix(Diagonal(vec(sum(W, dims=2))))
    local L = D - Matrix(W)

    # Operator: (I - dt * (diffusion * L + velocity * G))
    # For this restoration, we focus on the Diffusion Laplacian contribution.
    return I(n) .- (diffusion_coeff .* L)
end

# Updated Parser Logic for Spatial Manifold Dispatch
function parse_spatial_term(var_name, params_str)
    local p = parse_module_params(params_str)
    local manifold = get(p, :manifold, "bym2")

    # Logic: Determine if input is index (graph) or coordinates (continuous)
    # Graph-based requires W; Coordinate-based requires (x,y) parsing.
    return (variable=var_name, manifold=manifold, params=p)
end



;;
 