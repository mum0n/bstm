# --- 1. Abstract Base Type Hierarchy --- 
# XR: Establishing the primary abstract manifold hierarchy.
# Concrete types cannot be subtyped; therefore, we must define abstract bases first.
abstract type AbstractModelArchitecture end
abstract type ModelFamily end
abstract type Manifold end

# Discrete Manifold Categories
abstract type Spatial <: Manifold end
abstract type Temporal <: Manifold end
abstract type Seasonal <: Manifold end
abstract type Covariate <: Manifold end
abstract type Compositional <: Manifold end


# Continuous Manifold Categories
abstract type ContinuousTemporal <: Temporal end
abstract type ContinuousSpatial <: Spatial end



# Define Concrete Traits for Dispatch
struct SpatialTrait <: Spatial end
struct TemporalTrait <: Temporal end
struct SeasonalTrait <: Seasonal end
struct FixedTrait <: Covariate end


# --- 2. Architectural Dispatch Types ---
struct UnivariateArchitecture <: AbstractModelArchitecture end
struct MultivariateArchitecture <: AbstractModelArchitecture end
struct MultifidelityArchitecture <: AbstractModelArchitecture end
struct ExampleArchitecture <: AbstractModelArchitecture end
struct UnknownArchitecture <: AbstractModelArchitecture end

# --- 3. Likelihood Family Types ---
struct PoissonFamily <: ModelFamily end
struct GaussianFamily <: ModelFamily end
struct LogNormalFamily <: ModelFamily end
struct BinomialFamily <: ModelFamily end
struct NegativeBinomialFamily <: ModelFamily end

# --- 4. Composite and Constrained Manifolds ---
# XR: Fixed subtyping error here by ensuring these depend on 'abstract type Manifold'.
struct RegularizationGroup <: Manifold
    manifolds::Vector{Manifold}
    penalty::Symbol # :ridge, :lasso
    lambda_prior::UnivariateDistribution
end

struct SoftConstraint <: Manifold
    manifold::Manifold
    type::Symbol # :sum_to_zero, :monotonic
    weight::Float64
end

struct ConstrainedBlock <: Manifold
    manifolds::Vector{Manifold}
    constraint::Symbol # :sum_to_zero
end

struct SelectBest <: Manifold
    models::Vector{Manifold}
    cv_folds::Int
end

struct ComposedManifold <: Manifold
    components::Vector{Manifold}
    operator::Symbol  # :pipe, :add, :kronecker_product, :direct_sum
end

struct TransformedManifold <: Manifold
    manifold::Manifold
    transform_fn::DataType # e.g., Log, ZScore types
end

# Placeholder transformation types
struct Log <: Manifold end
struct ZScore <: Manifold end
struct UnitScale <: Manifold end
struct SumToZero <: Manifold end
struct ProperConstraint <: Manifold end

# --- 5. Hierarchical and Multi-Fidelity Terms ---
struct Intercept <: Manifold
    prior::Union{Nothing, UnivariateDistribution}
end

# Helper constructor for Intercept
Intercept(; prior=nothing) = Intercept(prior)

struct HierarchicalLink <: Manifold
    target_manifold::Manifold
    noise_scale::Float64
    layers::Union{Nothing, Int}
end

struct DeepLayer <: Manifold
    n_features::Int
end

# --- 6. Concrete Spatial Manifolds ---
struct BYM2 <: Spatial
    index::Symbol
    sigma_prior::UnivariateDistribution
    rho_prior::UnivariateDistribution
end
 
# 1. Concrete Struct Definitions (Verified)
struct Besag <: Spatial
    index::Symbol
    sigma_prior::UnivariateDistribution
end

struct Leroux <: Spatial
    index::Symbol
    sigma_prior::UnivariateDistribution
    rho_prior::UnivariateDistribution
end

struct SAR <: Spatial
    index::Symbol
    sigma_prior::UnivariateDistribution
    rho_prior::UnivariateDistribution
end
 
struct ICAR <: Spatial
    index::Symbol
    sigma_prior::UnivariateDistribution
end

struct ProperCAR <: Spatial
    index::Symbol
    sigma_prior::UnivariateDistribution
    rho_prior::UnivariateDistribution
end

struct RW1 <: Spatial
    index::Symbol
    sigma_prior::UnivariateDistribution
end

struct RW2 <: Spatial
    index::Symbol
    sigma_prior::UnivariateDistribution
end

struct GCN <: Spatial
    index::Symbol
    sigma_prior::UnivariateDistribution
    n_layers::Int
end

struct ExponentialDecay <: Spatial
    index::Symbol
    sigma_prior::UnivariateDistribution
    decay_lengthscale::Float64
    decay_lengthscale_prior::UnivariateDistribution
end

# XR: Topological Manifolds (v05.4 Addition)
struct LocalAdaptive <: Spatial
    index::Symbol
    weights_variable::Symbol
    sigma_prior::UnivariateDistribution
end

struct DirectedAcyclicGraph <: Spatial
    index::Symbol
    adjacency_matrix::AbstractMatrix
    sigma_prior::UnivariateDistribution
end

struct NetworkFlow <: Spatial
    index::Symbol
    adjacency_matrix::AbstractMatrix
    flow_direction::Symbol # :upstream, :downstream, or :bidirectional
    sigma_prior::UnivariateDistribution
end

# --- 7. Continuous Spatial Manifolds (GPs) ---
struct GaussianProcess <: ContinuousSpatial
    coordinates::Vector{Symbol}
    sigma_prior::UnivariateDistribution
    kernel::Union{Nothing, String}
    nu::Union{Nothing, Float64}
    lengthscale_prior::UnivariateDistribution
end

struct FITC <: ContinuousSpatial
    coordinates::Vector{Symbol}
    sigma_prior::UnivariateDistribution
    kernel::Union{Nothing, String}
    nu::Union{Nothing, Float64}
    lengthscale_prior::UnivariateDistribution
    n_inducing::Int
end

struct RandomFourierFeatures <: ContinuousSpatial
    coordinates::Vector{Symbol}
    sigma_prior::UnivariateDistribution
    kernel::Union{Nothing, String}
    n_features::Int
end

struct SPDE <: ContinuousSpatial
    coordinates::Vector{Symbol}
    sigma_prior::UnivariateDistribution
    smoothness::Float64
end

struct Hyperbolic <: ContinuousSpatial
    coordinates::Vector{Symbol}
    curvature::Float64
    sigma_prior::UnivariateDistribution
end

# --- 8. Temporal and Seasonal Manifolds ---
struct AR1 <: Temporal
    index::Symbol
    sigma_prior::UnivariateDistribution
    rho_prior::UnivariateDistribution
end

struct RW1T <: Temporal
    index::Symbol
    sigma_prior::UnivariateDistribution
end

struct RW2T <: Temporal
    index::Symbol
    sigma_prior::UnivariateDistribution
end

struct IIDT <: Temporal
    index::Symbol
    sigma_prior::UnivariateDistribution
end

struct HarmonicSeasonal <: Seasonal
    period::Int
    sigma_prior::UnivariateDistribution
    amplitude_prior::UnivariateDistribution
    phase_prior::UnivariateDistribution
end

struct DiscreteSeasonal <: Seasonal
    index::Symbol
    sigma_prior::UnivariateDistribution
    manifold::Symbol # e.g., :rw1, :iid
end

# --- 9. Covariate Manifolds ---
struct Fixed <: Covariate
    variable::Union{Symbol, Vector{Symbol}}
    contrasts::Union{Nothing, String}
    shared_prior::Bool
end

# Constructor for flexible instantiation within the parser
Fixed(v::Symbol; contrasts=nothing, shared_prior=false) = Fixed(v, contrasts, shared_prior)

struct Smooth <: Covariate
    variable::Symbol
    manifold::Symbol
    nbins::Union{Nothing, Int}
    transform::Union{Nothing, Function}
    grouped_by::Union{Nothing, Symbol}
end

struct TPS <: Covariate
    variable::Symbol
    nbins::Int
    sigma_prior::UnivariateDistribution
end

struct BSpline <: Covariate
    variable::Symbol
    nbins::Int
    degree::Int
    sigma_prior::UnivariateDistribution
end

struct PSpline <: Covariate
    variable::Symbol
    nbins::Int
    degree::Int
    diff_order::Int
    sigma_prior::UnivariateDistribution
end

struct FFT <: Covariate
    variable::Symbol
    nbins::Int
    sigma_prior::UnivariateDistribution
    is_2d::Bool
end

struct Wavelets <: Covariate
    variable::Symbol
    wavelet_family::Symbol
    nbins::Int
    sigma_prior::UnivariateDistribution
end

struct RFF <: Covariate
    variable::Union{Symbol, Vector{Symbol}}
    n_features::Int
    sigma_prior::UnivariateDistribution
    lengthscale_prior::UnivariateDistribution
end

# --- 10. Compositional Manifolds ---
struct KnorrHeld <: Compositional
    dim1::Symbol
    dim2::Symbol
    type::Char
    sigma_prior::UnivariateDistribution
end

struct NoInteraction <: Compositional
    dim1::Symbol
    dim2::Symbol
end

struct Advection <: Compositional
    dim1::Symbol
    dim2::Symbol
    velocity_field::Union{Nothing, Symbol}
    velocity_prior::UnivariateDistribution
end

struct Diffusion <: Compositional
    dim1::Symbol
    dim2::Symbol
    diffusion_coeff::Union{Nothing, Float64}
    diffusion_prior::UnivariateDistribution
end

struct AdvectionDiffusion <: Compositional
    dim1::Symbol
    dim2::Symbol
    velocity_field::Union{Nothing, Symbol}
    velocity_prior::UnivariateDistribution
    diffusion_coeff::Union{Nothing, Float64}
    diffusion_prior::UnivariateDistribution
end

struct NonSeparableRFF <: Compositional
    coordinates::Vector{Symbol}
    n_features::Int
    lengthscale_priors::Vector{<:UnivariateDistribution}
end

struct Mosaic <: Compositional
    coordinates::Vector{Symbol}
    n_regions::Int
    local_smoothness::Bool
end

struct Warping <: Compositional
    coordinates::Vector{Symbol}
    warp_features::Int
    spatial_features::Int
end

# --- 11. Hyperprior Metadata ---
struct SpatialHyper
    sigma_prior::UnivariateDistribution
    rho_prior::UnivariateDistribution
end

# --- 12. Standard Registry Dispatch ---
# XR: Consolidating manifold_type routing to resolve previous UndefVarErrors.
# --- Spatial Manifolds (Discrete & Topological) ---
function manifold_type(m::BYM2) return :bym2 end
function manifold_type(m::ICAR) return :icar end
function manifold_type(m::ProperCAR) return :proper_car end
function manifold_type(m::Besag) return :besag end
function manifold_type(m::Leroux) return :leroux end
function manifold_type(m::SAR) return :sar end
function manifold_type(m::RW1) return :rw1 end
function manifold_type(m::RW2) return :rw2 end
function manifold_type(m::GCN) return :gcn end
function manifold_type(m::ExponentialDecay) return :exponential_decay end
function manifold_type(m::LocalAdaptive) return :local_adaptive end
function manifold_type(m::NetworkFlow) return :network end
function manifold_type(m::DirectedAcyclicGraph) return :dag end

# --- Continuous Spatial Manifolds (GPs & Kernels) ---
function manifold_type(m::GaussianProcess) return :gp end
function manifold_type(m::FITC) return :fitc end
function manifold_type(m::RandomFourierFeatures) return :rff end
function manifold_type(m::SPDE) return :spde end
function manifold_type(m::Hyperbolic) return :hyperbolic end

# --- Temporal & Seasonal Manifolds ---
function manifold_type(m::AR1) return :ar1 end
function manifold_type(m::RW1T) return :rw1 end
function manifold_type(m::RW2T) return :rw2 end
function manifold_type(m::IIDT) return :iid end
function manifold_type(m::HarmonicSeasonal) return :harmonic end
function manifold_type(m::DiscreteSeasonal) return :seasonal end

# --- Covariate Smooths & Basis Manifolds ---
function manifold_type(m::TPS) return :tps end
function manifold_type(m::BSpline) return :bspline end
function manifold_type(m::PSpline) return :pspline end
function manifold_type(m::FFT) return :fft end
function manifold_type(m::Wavelets) return :wavelet end
function manifold_type(m::RFF) return :rff end

# --- Compositional Manifolds (Physics & Stacking) ---
function manifold_type(m::KnorrHeld) return :st_interaction end
function manifold_type(m::NoInteraction) return :none end
function manifold_type(m::Advection) return :advection end
function manifold_type(m::Diffusion) return :diffusion end
function manifold_type(m::AdvectionDiffusion) return :advection_diffusion end
function manifold_type(m::NonSeparableRFF) return :ns_rff end
function manifold_type(m::Mosaic) return :mosaic end
function manifold_type(m::Warping) return :warping end

# --- 13. Metadata Helpers ---
dimension(m::Spatial) = "spatial"
dimension(m::Temporal) = "temporal"
dimension(m::Seasonal) = "seasonal"

 