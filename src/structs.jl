#!Reference

# Definitions
const BSTM_MODULE_KEYWORDS = Set([
    "intercept", "observationprocess", "spatial", "temporal",
    "smooth", "nested", "eigen", "fixed", "mixed", "dynamics"
])

const BSTM_TRANSFORM_KEYWORDS = Dict(
    "log" => x -> log.(x),
    "zscore" => x -> (x .- mean(x)) ./ std(x),
    "unit" => x -> (x .- minimum(x)) ./ (maximum(x) - minimum(x))
)
# BSTM Low-Level Manifold Registry [v06.1 - Reusable Schema] ---
# Rationale: ManifoldModels are now defined as low-level primitives that are domain-agnostic. 
# The context (Spatial vs Temporal) is determined at the model-building stage.

# --- 1. Core Abstract Types ---
abstract type Manifold end
abstract type ManifoldModel <: Manifold end
abstract type ManifoldOperator <: Manifold end
struct Fixed <: ManifoldModel end
struct Covariate <: ManifoldModel end
struct NoneManifold <: ManifoldModel end

# 2.2 Discrete & Graph Primitives
struct IID <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct ICAR <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct Besag <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct BYM2 <: ManifoldModel; rho_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end
struct Leroux <: ManifoldModel; rho_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end
struct SAR <: ManifoldModel; rho_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end
struct RW1 <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct RW2 <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct AR1 <: ManifoldModel; rho_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end

struct GP <: ManifoldModel; lengthscale_prior::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; sigma_prior::UnivariateDistribution; kernel::String; end
struct FITC <: ManifoldModel; lengthscale_prior::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; sigma_prior::UnivariateDistribution; n_inducing::Int; kernel::String; end
struct RFF <: ManifoldModel; lengthscale_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; n_features::Int; kernel::String; end
struct FFT <: ManifoldModel; sigma_prior::UnivariateDistribution; nbins::Int; kernel::String; lengthscale_prior::UnivariateDistribution; end
struct SPDE <: ManifoldModel; sigma_prior::UnivariateDistribution; kappa_prior::UnivariateDistribution; end
struct SVGP <: ManifoldModel; lengthscale_prior::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; sigma_prior::UnivariateDistribution; n_inducing::Int; kernel::String; end
struct Warp <: ManifoldModel; lengthscale_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; n_features::Int; kernel::String; end
struct Nystrom <: ManifoldModel; lengthscale_prior::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; sigma_prior::UnivariateDistribution; n_inducing::Int; kernel::String; end
struct Hyperbolic <: ManifoldModel; curvature::Real; sigma_prior::UnivariateDistribution; end
struct ExponentialDecay <: ManifoldModel; sigma_prior::UnivariateDistribution; lengthscale_prior::UnivariateDistribution; end

struct PSpline <: ManifoldModel
    nbins::Int
    degree::Int
    diff_order::Int
    sigma_prior::UnivariateDistribution
end

struct Wavelet <: ManifoldModel
    family::Symbol
    nbins::Int
    sigma_prior::UnivariateDistribution
end

struct Eigen <: ManifoldModel; n_factors::Int; pca_sd_prior::UnivariateDistribution; pdef_sd_prior::UnivariateDistribution; ltri_indices::Vector{Int}; end
struct Moran <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct Spherical <: ManifoldModel; sigma_prior::UnivariateDistribution; range_prior::UnivariateDistribution; end
struct Barycentric <: ManifoldModel; sigma_prior::UnivariateDistribution; end

struct BCGN <: ManifoldModel; sigma_prior::UnivariateDistribution; bipartite_adj::AbstractMatrix; end
struct NetworkFlow <: ManifoldModel; sigma_prior::UnivariateDistribution; adjacency_matrix::AbstractMatrix; flow_direction::Symbol; end
struct LocalAdaptive <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct Mosaic <: ManifoldModel; sigma_prior::UnivariateDistribution; n_regions::Int; end
struct TensorProductSmooth <: ManifoldModel; sigma_prior::UnivariateDistribution; Q_template::AbstractMatrix; end

struct TPS <: ManifoldModel; nbins::Int; sigma_prior::UnivariateDistribution; end
struct BSpline <: ManifoldModel; nbins::Int; degree::Int; sigma_prior::UnivariateDistribution; end

struct Harmonic <: ManifoldModel
    amplitude_prior::UnivariateDistribution
    phase_prior::UnivariateDistribution
    sigma_prior::UnivariateDistribution
    period::Union{Real, UnivariateDistribution}
end
struct Cyclic <: ManifoldModel; period::Int; sigma_prior::UnivariateDistribution; end

struct ST_I <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct ST_II <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct ST_III <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct ST_IV <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct DynamicsManifold <: ManifoldModel
    model::String
    params::Dict{Symbol, Any}
end

struct ComposedManifold <: ManifoldOperator; components::Vector{Manifold}; operator::Symbol; end
struct ChangePointManifold <: ManifoldOperator
    manifold::ManifoldModel              # The process within each segment (e.g., AR1)
    n_changepoints::Int                  # Number of change points to infer
    changepoint_prior::UnivariateDistribution # Prior on the location of change points
end
struct TransformedManifold <: ManifoldOperator
    manifold::Manifold
    transform_fn::Symbol
end
struct VaryingInteractionManifold <: ManifoldOperator
    interaction_vars::Vector{Symbol} # e.g., [:temperature, :salinity]
    model::ManifoldModel             # e.g., AR1(...) or RW2(...)
end
struct SVCManifold <: ManifoldOperator
    covariate::Symbol
    model::ManifoldModel
end
struct MixedManifold <: ManifoldOperator
    group_var::Symbol
    lhs::String
    model::ManifoldModel
end
struct RegularizationGroupManifold{T<:Union{UnivariateDistribution, Nothing}} <: ManifoldOperator
    manifolds::Vector{Manifold}
    penalty::Symbol # :ridge, :lasso, :elastic_net
    lambda_prior::UnivariateDistribution
    alpha_prior::T # Mixing parameter for Elastic Net
end

RegularizationGroupManifold(manifolds, penalty, lambda_prior; alpha_prior=nothing) = RegularizationGroupManifold(manifolds, penalty, lambda_prior, alpha_prior)

struct SoftConstraintManifold <: ManifoldOperator
    manifold::Manifold
    type::Symbol # :sum_to_zero, :monotonic_increasing, :monotonic_decreasing, :periodicity, :non_negative, :convex, :concave
    weight::Float64
end

abstract type ManifoldSupervisor <: Manifold end
struct NestedManifold <: ManifoldSupervisor
    var::Symbol
    formula::String
    data_source::Symbol
end

function Base.:|>(m1::Manifold, m2::Manifold)
    return ComposedManifold([m1, m2], :pipe)
end

⊗(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :kronecker_product)
⊕(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :direct_sum)

otimes(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :kronecker_product)
oplus(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :direct_sum)

const PC_PRIORS = Dict(
    "sigma" => Exponential(1.0),
    "rho" => Beta(1, 1),
    "lengthscale" => InverseGamma(3, 3),
    "kappa" => Exponential(1.0),
    "amplitude" => Normal(0, 1),
    "phase" => Beta(1, 1),
    "pca_sd" => Exponential(1.0), 
    "pdef_sd" => Exponential(1.0),
    "range" => InverseGamma(3,3)
)

const INFORMATIVE_PRIORS = Dict(
    "sigma" => Exponential(0.5),
    "rho" => Beta(2, 2),
    "lengthscale" => InverseGamma(5, 5),
    "kappa" => Exponential(0.1),
    "amplitude" => Normal(0, 0.5),
    "phase" => Beta(2, 2),
    "pca_sd" => Exponential(0.5), 
    "pdef_sd" => Exponential(0.5),
    "range" => InverseGamma(5,5)
)

const UNINFORMATIVE_PRIORS = Dict(
    "sigma" => Normal(0, 1e6),
    "rho" => Uniform(0, 1),
    "lengthscale" => InverseGamma(0.01, 0.01),
    "kappa" => Exponential(10.0),
    "amplitude" => Normal(0, 100),
    "phase" => Uniform(0, 1),
    "pca_sd" => Normal(0, 1e6), 
    "pdef_sd" => Normal(0, 1e6),
    "range" => InverseGamma(0.01, 0.01)
)

const MANIFOLD_CONSTRUCTORS = Dict{Symbol, Function}(
    :none => (p, params) -> NoneManifold(),
    :iid => (p, params) -> IID(p.sigma_prior),
    :icar => (p, params) -> ICAR(p.sigma_prior),
    :besag => (p, params) -> Besag(p.sigma_prior),
    :bym2 => (p, params) -> BYM2(p.rho_prior, p.sigma_prior),
    :leroux => (p, params) -> Leroux(p.rho_prior, p.sigma_prior),
    :sar => (p, params) -> SAR(p.rho_prior, p.sigma_prior),
    :ar1 => (p, params) -> AR1(p.rho_prior, p.sigma_prior),
    :rw1 => (p, params) -> RW1(p.sigma_prior),
    :rw2 => (p, params) -> RW2(p.sigma_prior),
    :fitc => (p, params) -> FITC(p.lengthscale_prior, p.sigma_prior, get(params, :n_inducing, 20), string(get(params, :kernel, "se"))),
    :svgp => (p, params) -> SVGP(p.lengthscale_prior, p.sigma_prior, get(params, :n_inducing, 20), string(get(params, :kernel, "se"))),
    :nystrom => (p, params) -> Nystrom(p.lengthscale_prior, p.sigma_prior, get(params, :n_inducing, 20), string(get(params, :kernel, "se"))),
    :warp => (p, params) -> Warp(p.lengthscale_prior, p.sigma_prior, get(params, :n_features, 20), string(get(params, :kernel, "se"))),
    :hyperbolic => (p, params) -> Hyperbolic(get(params, :curvature, -1.0), p.sigma_prior),
    :decay => (p, params) -> ExponentialDecay(p.sigma_prior, p.lengthscale_prior),
    :gp => (p, params) -> GP(p.lengthscale_prior, p.sigma_prior, string(get(params, :kernel, "se"))),
    :rff => (p, params) -> RFF(p.lengthscale_prior, p.sigma_prior, get(params, :n_features, 20), string(get(params, :kernel, "se"))),
    :fft => (p, params) -> FFT(p.sigma_prior, get(params, :nbins, 20), string(get(params, :kernel, "se")), p.lengthscale_prior),
    :spde => (p, params) -> SPDE(p.sigma_prior, p.kappa_prior),
    :cyclic => (p, params) -> Cyclic(get(params, :period, 12), p.sigma_prior),
    :harmonic => (p, params) -> Harmonic(p.amplitude_prior, p.phase_prior, p.sigma_prior, get(params, :period, 12.0)),
    :pspline => (p, params) -> PSpline(get(params, :nbins, 20), get(params, :degree, 3), get(params, :diff_order, 2), p.sigma_prior),
    :bspline => (p, params) -> BSpline(get(params, :nbins, 10), get(params, :degree, 3), p.sigma_prior),
    :tps => (p, params) -> TPS(get(params, :nbins, 20), p.sigma_prior),
    :wavelet => (p, params) -> Wavelet(get(params, :family, :db4), get(params, :nbins, 32), p.sigma_prior),
    :eigen => (p, params) -> Eigen(get(params, :n_factors, 1), p.pca_sd_prior, p.pdef_sd_prior, get(params, :ltri_indices, Int[])),
    :moran => (p, params) -> Moran(p.sigma_prior),
    :spherical => (p, params) -> Spherical(p.sigma_prior, p.range_prior),
    :barycentric => (p, params) -> Barycentric(p.sigma_prior),
    :bcgn => (p, params) -> BCGN(p.sigma_prior, get(params, :bipartite_adj, sparse(zeros(1,1)))),
    :networkflow => (p, params) -> NetworkFlow(p.sigma_prior, get(params, :adjacency_matrix, sparse(zeros(1,1))), get(params, :flow_direction, :bidirectional)),
    :localadaptive => (p, params) -> LocalAdaptive(p.sigma_prior),
    :mosaic => (p, params) -> Mosaic(p.sigma_prior, get(params, :n_regions, 4)),
    :tensorproductsmooth => (p, params) -> TensorProductSmooth(p.sigma_prior, get(params, :Q_template, sparse(zeros(1,1)))),
    :dynamics => (p, params) -> DynamicsManifold(string(get(params, :model, "none")), params)
)

# --- 3. Architectural Dispatch Types ---
abstract type AbstractModelArchitecture end

struct UnivariateArchitecture <: AbstractModelArchitecture end
struct MultivariateArchitecture <: AbstractModelArchitecture end
struct MultifidelityArchitecture <: AbstractModelArchitecture end
struct ExampleArchitecture <: AbstractModelArchitecture end
struct UnknownArchitecture <: AbstractModelArchitecture end


# --- 4. Likelihood Family Types ---
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


const BSTM_FAMILY_REGISTRY = Dict{String, AbstractBSTM_Family}(
    "poisson" => PoissonFamily(),
    "gaussian" => GaussianFamily(),
    "lognormal" => LogNormalFamily(),
    "bernoulli" => BinomialFamily(),
    "binomial" => BinomialFamily(),
    "negbin" => NegativeBinomialFamily(),
    "gamma" => GammaFamily(),
    "exponential" => ExponentialFamily(),
    "beta" => BetaFamily(),
    "inverse_gaussian" => InverseGaussianFamily(),
    "student_t" => StudentTFamily(),
    "half_normal" => HalfNormalFamily(),
    "half_student_t" => HalfStudentTFamily(),
    "laplace" => LaplaceFamily(),
    "pareto" => ParetoFamily(),
    "dirichlet" => DirichletFamily(),
    "inverse_wishart" => InverseWishartFamily()
)


abstract type AbstractZIState end
struct NonZeroInflated <: AbstractZIState end
struct ZeroInflated <: AbstractZIState end

abstract type AbstractCensoringState end
struct Uncensored <: AbstractCensoringState end
struct LeftCensored <: AbstractCensoringState end
struct RightCensored <: AbstractCensoringState end
struct IntervalCensored <: AbstractCensoringState end

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